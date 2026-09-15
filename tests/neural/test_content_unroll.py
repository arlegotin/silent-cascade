"""Real model trajectories for immediate, predict-before-teacher content learning."""

import sys
from dataclasses import fields, replace

import pytest
import torch

from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.schemas import InternalEventKind
from silent_cascade.train.batches import pack_training_examples
from silent_cascade.train.curriculum_data import CurriculumKey, make_curriculum_example


@pytest.fixture
def model(neural_config):
    torch.manual_seed(11)
    return EventFlowModel(neural_config.neural)


@pytest.fixture
def examples(neural_config):
    return tuple(
        make_curriculum_example(
            neural_config, CurriculumKey("ofd-one-hop-v1", "debug", 311, 331, i, "one_hop")
        )
        for i in range(4)
    )


def positive_batch(examples):
    return pack_training_examples(
        (next(e for e in examples if e.solution.hazard_type is not None),)
    )


def test_content_keeps_activation_clocks_and_only_real_external_flow(model, examples, monkeypatch):
    from silent_cascade.models.dynamics import crossings_batch
    from silent_cascade.train import observations
    from silent_cascade.train.content_unroll import teacher_forced_content_unroll

    batch = pack_training_examples(examples)
    flow_deltas, control_modes, jump_records, crossings = [], [], [], []
    real_flow = observations.flow_batch

    def capture_flow(workspace, parameters, delta):
        flow_deltas.append(delta.clone())
        return real_flow(workspace, parameters, delta)

    monkeypatch.setattr(observations, "flow_batch", capture_flow)
    previous_profile = sys.getprofile()

    def capture_crossings(frame, event, argument):
        if event == "call" and frame.f_code is crossings_batch.__code__:
            crossings.append(frame.f_locals)

    handles = [
        model.controller.register_forward_pre_hook(
            lambda module, args: control_modes.append(args[0].modes.clone())
        ),
        model.shared_jump.register_forward_pre_hook(
            lambda module, args: jump_records.append(
                (args[0].active_slot_indices.clone(), args[1].clone())
            )
        ),
    ]
    try:
        sys.setprofile(capture_crossings)
        result = teacher_forced_content_unroll(model, batch)
    finally:
        sys.setprofile(previous_profile)
        for handle in handles:
            handle.remove()
    assert len({example.variant for example in examples}) == 3
    activation = result.observations[-1].context.time_features
    expected = batch.targets.real_step_mask & (batch.targets.kind < 2)
    inputs = result.loss_inputs()
    assert torch.equal(inputs.boundary_mask, expected)
    assert torch.equal(inputs.recall_mask, expected & (batch.targets.kind == 0))
    assert len(result.steps) == batch.targets.kind.shape[1]
    assert expected.sum(1).tolist() == [
        len(t.steps) - int(t.steps[-1].kind == InternalEventKind.ACT) for t in batch.teacher_traces
    ]
    for col, step in enumerate(result.steps):
        assert torch.equal(step.row_mask, expected[:, col])
        torch.testing.assert_close(
            step.prediction_context.time_features, activation, rtol=0, atol=0
        )
        torch.testing.assert_close(step.post_context.time_features, activation, rtol=0, atol=0)
    external_flows = int(batch.public.observation_mask.sum()) - len(examples)
    assert result.compute.flow_evaluations == external_flows
    assert sum(delta.numel() for delta in flow_deltas) == external_flows
    assert all((delta > 0).all() for delta in flow_deltas)
    assert result.compute.jump_applications == int(
        batch.public.observation_mask.sum() + expected.sum()
    )
    assert result.compute.module_calls.get("ActionHeads", 0) == 0
    assert not crossings
    assert all((modes != 2).all() and (modes != 4).all() for modes in control_modes)
    assert all((slots >= 0).all() for slots, _ in jump_records)
    assert sum(kind.numel() for _, kind in jump_records) == int(expected.sum())
    assert result.compute.foundation_model_calls == 0
    assert result.diagnostic_label == "teacher-forced-untimed-content"


def test_active_record_lifetime_consumption_and_terminal_exclusion(model, examples):
    from silent_cascade.train.content_unroll import teacher_forced_content_unroll

    batch = pack_training_examples(examples)
    result = teacher_forced_content_unroll(model, batch)
    for row, trace in enumerate(batch.teacher_traces):
        consumed = []
        for col in range(len(trace.steps)):
            step = result.steps[col]
            if not step.row_mask[row]:
                assert batch.targets.kind[row, col] == 2
                continue
            if step.kind[row] == 0:
                assert step.prediction_context.active_slot_indices[row] == -1
                assert not step.retrieval.eligible_mask[row, consumed].any()
                slot = int(batch.targets.selected_slot[row, col])
                assert step.post_context.active_slot_indices[row] == slot
                assert step.post_context.modes[row] == 2
            else:
                assert step.prediction_context.active_slot_indices[row] == slot
                assert step.post_context.active_slot_indices[row] == -1
                assert not step.post_context.eligibility[row, slot]
                consumed.append(slot)
            assert torch.equal(step.post_context.support_mask[row], batch.support_after[row, col])
        for col in range(len(trace.steps), len(result.steps)):
            torch.testing.assert_close(
                result.steps[col].post_context.workspace.latent[row],
                result.steps[col - 1].post_context.workspace.latent[row],
                rtol=0,
                atol=0,
            )
    assert (result.final_context.active_slot_indices == -1).all()


def test_teacher_clocks_deadline_and_future_labels_cannot_change_content_predictions(
    model, examples
):
    from silent_cascade.train.content_unroll import teacher_forced_content_unroll

    batch = positive_batch(examples)
    first = teacher_forced_content_unroll(model, batch)
    altered = replace(
        batch,
        teacher_times=batch.teacher_times + 10000,
        targets=replace(
            batch.targets,
            delta=batch.targets.delta + 200,
            normalized_deadline=batch.targets.normalized_deadline + 700,
        ),
    )
    second = teacher_forced_content_unroll(model, altered)
    for a, b in zip(first.steps, second.steps, strict=True):
        torch.testing.assert_close(
            a.prediction_context.workspace.latent,
            b.prediction_context.workspace.latent,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            a.post_context.hypothesis_features, b.post_context.hypothesis_features, rtol=0, atol=0
        )
    selected = batch.targets.selected_slot.clone()
    slots = batch.public.records.valid_mask[0].nonzero(as_tuple=True)[0]
    selected[0, 0] = next(slot for slot in slots if slot != selected[0, 0])
    changed = teacher_forced_content_unroll(
        model, replace(batch, targets=replace(batch.targets, selected_slot=selected))
    )
    torch.testing.assert_close(
        first.steps[0].retrieval.raw_scores, changed.steps[0].retrieval.raw_scores, rtol=0, atol=0
    )
    torch.testing.assert_close(
        first.steps[0].raw_recall_target, changed.steps[0].raw_recall_target, rtol=0, atol=0
    )
    hazard = batch.targets.hazard_type.clone()
    hazard[:, 3] = (hazard[:, 3] + 1) % 4
    changed = teacher_forced_content_unroll(
        model, replace(batch, targets=replace(batch.targets, hazard_type=hazard))
    )
    torch.testing.assert_close(
        first.loss_inputs().hazard_logits, changed.loss_inputs().hazard_logits, rtol=0, atol=0
    )
    terminal = first.steps[3].post_context.hypothesis_features
    torch.testing.assert_close(terminal[:, 7], batch.targets.log_delay[:, 3], rtol=0, atol=0)


def test_ragged_batch_matches_independent_rows_and_slot_permutation(model, examples):
    from silent_cascade.train.content_unroll import teacher_forced_content_unroll

    batch = pack_training_examples(examples)
    result = teacher_forced_content_unroll(model, batch)
    permutation = torch.arange(63, -1, -1).expand(len(examples), -1)
    permuted = teacher_forced_content_unroll(model, batch.permute_slots(permutation))
    torch.testing.assert_close(
        result.loss_inputs().retrieval_logits.flip(-1),
        permuted.loss_inputs().retrieval_logits,
        atol=3e-6,
        rtol=3e-5,
    )
    torch.testing.assert_close(
        result.final_context.workspace.latent,
        permuted.final_context.workspace.latent,
        atol=3e-6,
        rtol=3e-5,
    )
    for row, example in enumerate(examples):
        single = teacher_forced_content_unroll(model, pack_training_examples((example,)))
        torch.testing.assert_close(
            result.final_context.workspace.latent[row],
            single.final_context.workspace.latent[0],
            atol=3e-6,
            rtol=3e-5,
        )


def test_public_contexts_match_when_prior_real_learned_choices_agree(model, examples):
    from silent_cascade.train.component_eval import predict_components
    from silent_cascade.train.content_unroll import teacher_forced_content_unroll

    # This seeded public prediction makes a LINK/continue choice. Stage that
    # actual choice in the private teacher fixture, without altering predictions.
    batch = pack_training_examples((examples[3],))
    controller_contexts, compose_contexts, jump_contexts = [], [], []
    handles = [
        model.controller.register_forward_pre_hook(
            lambda module, args: controller_contexts.append(args[0])
        ),
        model.compose_heads.register_forward_pre_hook(
            lambda module, args: compose_contexts.append(args[0])
        ),
        model.shared_jump.register_forward_pre_hook(
            lambda module, args: jump_contexts.append((args[0], args[1]))
        ),
    ]
    try:
        public = predict_components(model, batch.public).rows[0]
    finally:
        for handle in handles:
            handle.remove()
    decision = public.content[0]
    assert decision.role == 0 and decision.continue_search
    selected, focus = batch.targets.selected_slot.clone(), batch.targets.focus.clone()
    selected[0, 0] = batch.public.records.record_ids[0].index(decision.record_id)
    focus[0, 1] = decision.focus
    support_after = batch.support_after.clone()
    support_after[0, 1] = jump_contexts[1][0].support_mask[0]
    batch = replace(
        batch,
        targets=replace(batch.targets, selected_slot=selected, focus=focus),
        support_after=support_after,
    )
    result = teacher_forced_content_unroll(model, batch)
    observation_count = int(batch.public.observation_mask.sum())

    def compare(actual, expected):
        for item in fields(actual):
            a, b = getattr(actual, item.name), getattr(expected, item.name)
            if item.name == "workspace":
                for field in fields(a):
                    torch.testing.assert_close(
                        getattr(a, field.name), getattr(b, field.name), rtol=0, atol=0
                    )
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    compare(result.steps[0].prediction_context, controller_contexts[observation_count])
    compare(result.steps[1].prediction_context, compose_contexts[0])
    compare(result.steps[2].prediction_context, controller_contexts[observation_count + 1])


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_content_gradients_reach_early_encoding_and_both_jumps_with_fresh_graphs(
    model, examples, device
):
    from silent_cascade.models.content_loss import content_loss
    from silent_cascade.train.content_unroll import teacher_forced_content_unroll

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    model = model.to(device)
    batch = positive_batch(examples).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = teacher_forced_content_unroll(model, batch)
        watched = [
            result.observations[0].context.workspace.latent,
            result.steps[0].post_context.workspace.latent,
            result.steps[1].post_context.workspace.latent,
        ]
        for tensor in watched:
            tensor.retain_grad()
        inputs = result.loss_inputs()
        assert all(getattr(inputs, field.name).device.type == device for field in fields(inputs))
        loss = content_loss(inputs)
        loss.total.backward()
        for tensor in watched:
            assert (
                tensor.grad is not None
                and torch.isfinite(tensor.grad).all()
                and tensor.grad.abs().sum() > 0
            )
        for parameter in (
            model.record_encoder.network[0].weight,
            model.external_encoder.injection.weight,
            model.shared_jump.network[0].weight,
        ):
            assert (
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and parameter.grad.abs().sum() > 0
            )
        optimizer.step()
