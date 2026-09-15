"""Behavioral tests for the assembled public EventFlow model interface."""

from dataclasses import replace
from inspect import signature

import pytest
import torch

from silent_cascade.models.types import ExternalFeatures
from silent_cascade.schemas import Mode


def _external_features(*, event_kinds: tuple[int, ...]) -> ExternalFeatures:
    batch_size = len(event_kinds)
    return ExternalFeatures(
        subject_ids=torch.arange(1, batch_size + 1, dtype=torch.int64),
        object_ids=torch.arange(2, batch_size + 2, dtype=torch.int64),
        record_kind_ids=torch.zeros(batch_size, dtype=torch.int64),
        hazard_ids=torch.full((batch_size,), 4, dtype=torch.int64),
        provenance_ids=torch.zeros(batch_size, dtype=torch.int64),
        record_scalar_features=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.2, 0.8]] * batch_size,
            dtype=torch.float32,
        ),
        activation_entity_ids=torch.arange(4, 4 + batch_size, dtype=torch.int64),
        event_kinds=torch.tensor(event_kinds, dtype=torch.int64),
        time_features=torch.tensor([[0.5, 0.0]] * batch_size),
    )


def test_public_model_method_signatures_are_frozen() -> None:
    from silent_cascade.models.event_flow import EventFlowModel

    assert str(signature(EventFlowModel.initial_context)) == (
        "(self, batch_size: 'int', *, device: 'str') -> 'ModelContext'"
    )
    assert tuple(signature(EventFlowModel.observe).parameters) == ("self", "context", "event")
    assert tuple(signature(EventFlowModel.preview_and_control).parameters) == ("self", "context")
    assert tuple(signature(EventFlowModel.recall_scores).parameters) == ("self", "context")
    assert tuple(signature(EventFlowModel.compose).parameters) == ("self", "context")
    assert tuple(signature(EventFlowModel.action).parameters) == ("self", "context")
    assert tuple(signature(EventFlowModel.jump).parameters) == ("self", "context", "event_kinds")


def test_assembler_has_exact_shared_parameter_ownership(neural_config) -> None:
    from silent_cascade.models.event_flow import EventFlowModel

    model = EventFlowModel(neural_config.neural)
    assert model.scorer.mode_embedding is model.mode_embedding
    assert model.controller.mode_embedding is model.mode_embedding
    assert model.shared_jump.mode_embedding is model.mode_embedding
    assert model.compose_heads.mode_embedding is model.mode_embedding
    assert model.action_heads.mode_embedding is model.mode_embedding
    assert model.external_encoder.record_encoder is model.record_encoder
    parameter_names = tuple(name for name, _ in model.named_parameters())
    assert sum(name.endswith("mode_embedding.weight") for name in parameter_names) == 1
    assert sum(name.endswith("entity_embedding.weight") for name in parameter_names) == 1
    assert sum(key.endswith("entity_embedding.weight") for key in model.state_dict()) == 1


def test_initial_context_is_empty_observing_public_state(neural_config) -> None:
    from silent_cascade.models.event_flow import EventFlowModel

    context = EventFlowModel(neural_config.neural).initial_context(3, device="cpu")
    assert context.workspace.latent.shape == (3, 456)
    assert context.memory_embeddings.shape == (3, 64, 96)
    assert not context.eligibility.any()
    assert not context.support_mask.any()
    assert context.active_slot_indices.eq(-1).all()
    assert context.modes.eq(tuple(Mode).index(Mode.OBSERVING)).all()
    assert context.time_features.eq(0.0).all()
    assert context.hypothesis_features.eq(0.0).all()


def test_observe_injects_without_retrieval_and_activate_installs_focus(neural_config) -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter
    from silent_cascade.models.event_flow import EventFlowModel

    model = EventFlowModel(neural_config.neural)
    context = model.initial_context(2, device="cpu")
    event = _external_features(event_kinds=(0, 1))
    with NeuralComputeMeter(model) as meter:
        observed = model.observe(context, event)
    snapshot = meter.snapshot()
    assert snapshot.records_scored == 0
    assert snapshot.jump_applications == 2
    assert observed.workspace.accumulators.eq(0.0).all()
    assert torch.equal(observed.workspace.latent[0, 320:], context.workspace.latent[0, 320:])
    expected_focus = torch.tanh(
        model.focus_projection(
            model.record_encoder.encode_entities(event.activation_entity_ids[1:])
        )
    )[0]
    torch.testing.assert_close(observed.workspace.latent[1, 328:392], expected_focus)
    assert observed.modes.tolist() == [0, 1]
    assert torch.equal(observed.time_features, event.time_features)


def test_mixed_preview_scores_only_postactivation_rows(model_context, neural_config) -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter
    from silent_cascade.models.event_flow import EventFlowModel

    model = EventFlowModel(neural_config.neural)
    mixed = replace(model_context, modes=torch.tensor([0, 1], dtype=torch.int64))
    with NeuralComputeMeter(model) as meter:
        preview, parameters = model.preview_and_control(mixed)
    snapshot = meter.snapshot()
    assert preview.features[0].eq(0.0).all()
    assert not preview.has_candidate[0]
    assert snapshot.records_scored == 64
    assert snapshot.eligible_records == int(mixed.eligibility[1].sum())
    assert snapshot.opportunities == 1
    assert parameters.flow_targets.shape == (2, 456)
    assert parameters.guard_targets[0].lt(1.0).all()


def test_recall_does_not_score_observing_memory(neural_config) -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter
    from silent_cascade.models.event_flow import EventFlowModel

    model = EventFlowModel(neural_config.neural)
    observing = model.initial_context(2, device="cpu")
    with NeuralComputeMeter(model) as meter:
        scores = model.recall_scores(observing)
    assert meter.snapshot().records_scored == 0
    assert scores.raw_scores.eq(0.0).all()
    assert scores.masked_logits.isneginf().all()
    assert not scores.eligible_mask.any()


def test_model_budget(neural_config) -> None:
    from silent_cascade.eval.compute import parameter_counts
    from silent_cascade.models.event_flow import EventFlowModel

    counts = parameter_counts(EventFlowModel(neural_config.neural))
    assert counts["entity_table"] == 64 * 32
    assert 1_000_000 <= counts["total"] <= 5_000_000
    assert counts["total"] == counts["entity_table"] + counts["non_entity"]


def test_model_prediction_methods_return_neutral_outputs(model_context, neural_config) -> None:
    from silent_cascade.models.event_flow import EventFlowModel

    model = EventFlowModel(neural_config.neural)
    assert model.recall_scores(model_context).raw_scores.shape == (2, 64)
    assert model.compose(model_context).role_logits.shape == (2, 5)
    assert model.action(model_context).class_logits.shape == (2, 5)
    assert model.jump(model_context, torch.tensor([0, 2])).latent.shape == (2, 456)


@torch.no_grad()
def _assert_mps_float32(*values: torch.Tensor) -> None:
    for value in values:
        assert value.device.type == "mps"
        assert value.dtype is torch.float32
        assert torch.isfinite(value).all()


@torch.no_grad()
def _mps_external_features() -> ExternalFeatures:
    return ExternalFeatures(
        subject_ids=torch.tensor([1, 2], dtype=torch.int64, device="mps"),
        object_ids=torch.tensor([2, 3], dtype=torch.int64, device="mps"),
        record_kind_ids=torch.tensor([0, 0], dtype=torch.int64, device="mps"),
        hazard_ids=torch.tensor([4, 4], dtype=torch.int64, device="mps"),
        provenance_ids=torch.tensor([0, 0], dtype=torch.int64, device="mps"),
        record_scalar_features=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.2, 0.8]] * 2,
            device="mps",
        ),
        activation_entity_ids=torch.tensor([4, 5], dtype=torch.int64, device="mps"),
        event_kinds=torch.tensor([1, 1], dtype=torch.int64, device="mps"),
        time_features=torch.tensor([[0.5, 0.0], [0.6, 0.0]], device="mps"),
    )


@torch.no_grad()
def _mps_memory_context(context):
    return replace(
        context,
        memory_embeddings=torch.zeros(2, 64, 96, device="mps"),
        eligibility=torch.ones(2, 64, dtype=torch.bool, device="mps"),
        active_slot_indices=torch.tensor([0, 1], dtype=torch.int64, device="mps"),
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_assembled_model_executes_complete_native_mps_graph(neural_config) -> None:
    from dataclasses import fields

    from silent_cascade.eval.compute import NeuralComputeMeter, parameter_counts
    from silent_cascade.models.dynamics import flow_batch
    from silent_cascade.models.event_flow import EventFlowModel

    model = EventFlowModel(neural_config.neural).to("mps")
    context = _mps_memory_context(model.initial_context(2, device="mps"))
    with NeuralComputeMeter(model) as meter:
        observed = model.observe(context, _mps_external_features())
        preview, parameters = model.preview_and_control(observed)
        scores = model.recall_scores(observed)
        composition = model.compose(observed)
        action = model.action(observed)
        jumped = model.jump(observed, torch.tensor([0, 2], dtype=torch.int64, device="mps"))
        flowed = flow_batch(jumped, parameters, torch.full((2,), 0.1, device="mps"))
        loss = (
            preview.features.sum()
            + scores.raw_scores.sum()
            + sum(getattr(composition, field.name).sum() for field in fields(composition))
            + action.class_logits.sum()
            + action.lead_fraction.sum()
            + flowed.latent.sum()
        )
        loss.backward()
    _assert_mps_float32(
        observed.workspace.latent,
        preview.features,
        parameters.flow_targets,
        scores.raw_scores,
        composition.role_logits,
        action.class_logits,
        jumped.latent,
        flowed.latent,
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    snapshot = meter.snapshot()
    assert snapshot.records_scored == 256
    assert snapshot.foundation_model_calls == 0
    assert snapshot.mps_peak_allocation_bytes is None
    assert parameter_counts(model) == {
        "total": 2_781_042,
        "entity_table": 2_048,
        "non_entity": 2_778_994,
    }
