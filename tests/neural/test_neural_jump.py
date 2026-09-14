"""Behavioral tests for bounded internal and external learned jumps."""

from dataclasses import fields

import pytest
import torch
from torch import nn

from silent_cascade.models.errors import NeuralError
from silent_cascade.models.types import ModelContext, TensorWorkspace


def _replace_workspace(context: ModelContext, workspace: TensorWorkspace) -> ModelContext:
    return ModelContext(
        workspace=workspace,
        memory_embeddings=context.memory_embeddings,
        eligibility=context.eligibility,
        support_mask=context.support_mask,
        active_slot_indices=context.active_slot_indices,
        modes=context.modes,
        time_features=context.time_features,
        hypothesis_features=context.hypothesis_features,
    )


def _external_features(batch_size: int = 2, *, device: str = "cpu"):
    from silent_cascade.models.types import ExternalFeatures

    return ExternalFeatures(
        subject_ids=torch.tensor([1, 0], dtype=torch.int64, device=device)[:batch_size],
        object_ids=torch.tensor([2, 0], dtype=torch.int64, device=device)[:batch_size],
        record_kind_ids=torch.zeros(batch_size, dtype=torch.int64, device=device),
        hazard_ids=torch.full((batch_size,), 4, dtype=torch.int64, device=device),
        provenance_ids=torch.zeros(batch_size, dtype=torch.int64, device=device),
        record_scalar_features=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.5, 0.8], [0.0] * 6],
            dtype=torch.float32,
            device=device,
        )[:batch_size],
        activation_entity_ids=torch.tensor([0, 3], dtype=torch.int64, device=device)[:batch_size],
        event_kinds=torch.tensor([0, 1], dtype=torch.int64, device=device)[:batch_size],
        time_features=torch.tensor([[0.5, 0.0], [1.0, 0.0]], device=device)[:batch_size],
    )


def test_shared_jump_is_bounded_resets_guards_and_has_finite_gradients(
    model_context, neural_config
) -> None:
    from silent_cascade.models.jump import SharedJump

    latent = torch.linspace(-1.0, 1.0, 912).reshape(2, 456)
    accumulators = torch.full((2, 3), 0.75)
    context = _replace_workspace(model_context, TensorWorkspace(latent, accumulators))
    jump = SharedJump(neural_config.neural)
    result = jump(context, torch.tensor([0, 2], dtype=torch.int64))
    assert result.latent.shape == (2, 456)
    assert result.latent.ge(-1.0).all() and result.latent.le(1.0).all()
    assert result.accumulators.eq(0.0).all()
    result.latent.square().sum().backward()
    gradients = [parameter.grad for parameter in jump.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_repeated_shared_jumps_remain_finite_and_bounded(model_context, neural_config) -> None:
    from silent_cascade.models.jump import SharedJump

    jump = SharedJump(neural_config.neural)
    context = model_context
    for _ in range(128):
        workspace = jump(context, torch.tensor([0, 1], dtype=torch.int64))
        context = _replace_workspace(context, workspace)
    assert torch.isfinite(workspace.latent).all()
    assert workspace.latent.ge(-1.0).all() and workspace.latent.le(1.0).all()
    assert workspace.accumulators.eq(0.0).all()


def test_shared_jump_uses_nonowning_mode_table_and_active_record(
    model_context, neural_config
) -> None:
    from silent_cascade.models.jump import SharedJump

    table = nn.Embedding(6, 8)
    jump = SharedJump(neural_config.neural, mode_embedding=table)
    context = ModelContext(
        workspace=model_context.workspace,
        memory_embeddings=model_context.memory_embeddings,
        eligibility=model_context.eligibility,
        support_mask=model_context.support_mask,
        active_slot_indices=torch.tensor([0, -1], dtype=torch.int64),
        modes=model_context.modes,
        time_features=model_context.time_features,
        hypothesis_features=model_context.hypothesis_features,
    )
    assert jump.mode_embedding is table
    assert all("mode_embedding" not in name for name, _ in jump.named_parameters())
    assert jump(context, torch.tensor([0, 1], dtype=torch.int64)).latent.shape == (2, 456)


@pytest.mark.parametrize(
    "event_kinds",
    [torch.tensor([0, 3]), torch.tensor([0.0, 1.0]), torch.tensor([0])],
)
def test_shared_jump_rejects_non_endogenous_event_kind_batches(
    model_context, neural_config, event_kinds
) -> None:
    from silent_cascade.models.jump import SharedJump

    with pytest.raises(NeuralError):
        SharedJump(neural_config.neural)(model_context, event_kinds)


def test_external_encoder_uses_current_public_fact_or_activation_only(
    neural_config,
) -> None:
    from silent_cascade.memory.encoder import RecordEncoder
    from silent_cascade.models.jump import ExternalEncoder

    record_encoder = RecordEncoder(neural_config.neural)
    external = ExternalEncoder(neural_config.neural, record_encoder)
    features = _external_features()
    output = external(features)
    assert output.shape == (2, 640)
    assert torch.isfinite(output).all()
    assert external.record_encoder is record_encoder
    assert all("record_encoder" not in name for name, _ in external.named_parameters())
    output.square().sum().backward()
    assert any(parameter.grad is not None for parameter in record_encoder.parameters())


def test_external_feature_schema_cannot_represent_padding_or_private_event() -> None:
    from silent_cascade.models.types import ExternalFeatures

    assert tuple(field.name for field in fields(ExternalFeatures)) == (
        "subject_ids",
        "object_ids",
        "record_kind_ids",
        "hazard_ids",
        "provenance_ids",
        "record_scalar_features",
        "activation_entity_ids",
        "event_kinds",
        "time_features",
    )
    values = {
        field.name: getattr(_external_features(), field.name) for field in fields(ExternalFeatures)
    }
    values["event_kinds"] = torch.tensor([0, 2], dtype=torch.int64)
    with pytest.raises(NeuralError):
        ExternalFeatures(**values)


def test_external_injection_updates_only_fast_slow_and_resets_guards() -> None:
    from silent_cascade.models.jump import inject_external

    state = TensorWorkspace(torch.full((2, 456), 0.25), torch.full((2, 3), 0.8))
    raw = torch.cat((torch.full((2, 320), -2.0), torch.full((2, 320), 2.0)), dim=1)
    result = inject_external(state, raw)
    assert result.latent[:, :320].lt(0.25).all()
    assert torch.equal(result.latent[:, 320:], state.latent[:, 320:])
    assert result.accumulators.eq(0.0).all()
    assert result.latent.ge(-1.0).all() and result.latent.le(1.0).all()


def test_external_injection_uses_trusted_functional_workspace_boundary(monkeypatch) -> None:
    from silent_cascade.models.jump import inject_external

    state = TensorWorkspace.zeros(2, "cpu")

    def reject_public_revalidation(self) -> None:
        raise AssertionError("hot functional updates must not invoke public constructor validation")

    monkeypatch.setattr(TensorWorkspace, "__post_init__", reject_public_revalidation)
    result = inject_external(state, torch.zeros(2, 640))
    assert result.latent.shape == (2, 456)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_external_and_internal_jumps_are_native_float32_mps(neural_config) -> None:
    from silent_cascade.memory.encoder import RecordEncoder
    from silent_cascade.models.jump import ExternalEncoder, SharedJump, inject_external

    config = neural_config.neural
    record_encoder = RecordEncoder(config).to("mps")
    external = ExternalEncoder(config, record_encoder).to("mps")
    workspace = TensorWorkspace.zeros(2, "mps")
    injected = inject_external(workspace, external(_external_features(device="mps")))
    context = ModelContext(
        workspace=injected,
        memory_embeddings=torch.zeros(2, 64, 96, device="mps"),
        eligibility=torch.zeros(2, 64, dtype=torch.bool, device="mps"),
        support_mask=torch.zeros(2, 64, dtype=torch.bool, device="mps"),
        active_slot_indices=torch.full((2,), -1, dtype=torch.int64, device="mps"),
        modes=torch.tensor([0, 1], dtype=torch.int64, device="mps"),
        time_features=torch.zeros(2, 2, device="mps"),
        hypothesis_features=torch.zeros(2, 8, device="mps"),
    )
    result = SharedJump(config).to("mps")(context, torch.tensor([0, 1], device="mps"))
    assert result.latent.device.type == "mps"
    assert result.latent.dtype is torch.float32
