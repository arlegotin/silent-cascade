"""Real trace regressions for causal, differentiable teacher forcing."""

from dataclasses import fields, replace

import pytest
import torch

from silent_cascade.models.event_flow import EventFlowModel
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


def test_future_teacher_choices_cannot_change_earlier_predictions(model, examples):
    from silent_cascade.train.unroll import teacher_forced_unroll

    batch = positive_batch(examples)
    assert batch.targets.kind.shape == (1, 5)
    hazard = batch.targets.hazard_type.clone()
    hazard[:, 3] = (hazard[:, 3] + 1) % 4
    altered = replace(batch, targets=replace(batch.targets, hazard_type=hazard))
    first = teacher_forced_unroll(model, batch)
    second = teacher_forced_unroll(model, altered)
    for i in range(4):
        torch.testing.assert_close(
            first.boundaries[i].raw_guard_targets,
            second.boundaries[i].raw_guard_targets,
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        first.loss_inputs().hazard_logits, second.loss_inputs().hazard_logits, rtol=0, atol=0
    )


def test_future_public_fact_does_not_change_observation_prefix(model, examples):
    from silent_cascade.train.unroll import teacher_forced_unroll

    batch = positive_batch(examples)
    records = batch.public.records
    values = records.subject_ids.clone()
    slot = int(batch.public.observation_slots[0, 1])
    values[0, slot] = (values[0, slot] + 1) % 64
    public = replace(batch.public, records=replace(records, subject_ids=values))
    first = teacher_forced_unroll(model, batch)
    second = teacher_forced_unroll(model, replace(batch, public=public))
    torch.testing.assert_close(
        first.observations[0].context.workspace.latent,
        second.observations[0].context.workspace.latent,
        rtol=0,
        atol=0,
    )
    assert not first.observations[0].preview.features.any()
    assert (first.observations[0].parameters.guard_targets < 1).all()


def test_final_action_retains_early_encoder_and_jump_gradients(model, examples):
    from silent_cascade.train.unroll import teacher_forced_unroll

    result = teacher_forced_unroll(model, positive_batch(examples))
    early = result.observations[0].context.workspace.latent
    early.retain_grad()
    inputs = result.loss_inputs()
    assert all(isinstance(getattr(inputs, f.name), torch.Tensor) for f in fields(inputs))
    torch.nn.functional.cross_entropy(
        inputs.action_logits[inputs.action_mask], inputs.action_target[inputs.action_mask]
    ).backward()
    assert early.grad is not None and early.grad.abs().sum() > 0
    for parameter in (
        model.record_encoder.network[0].weight,
        model.external_encoder.injection.weight,
        model.shared_jump.network[0].weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_batched_rows_match_independent_unroll_and_skip_padding(model, examples):
    from silent_cascade.train.unroll import teacher_forced_unroll

    batch = pack_training_examples(examples)
    evaluated_modes = []
    handle = model.controller.register_forward_pre_hook(
        lambda module, args: evaluated_modes.append(args[0].modes)
    )
    try:
        result = teacher_forced_unroll(model, batch)
    finally:
        handle.remove()
    inputs = result.loss_inputs()
    assert result.diagnostic_label == "teacher-forced"
    assert result.compute.jump_applications == int(
        batch.public.observation_mask.sum() + batch.targets.real_step_mask.sum()
    )
    assert result.compute.foundation_model_calls == 0
    assert sum(modes.numel() for modes in evaluated_modes) == int(
        batch.public.observation_mask.sum() + batch.targets.real_step_mask.sum()
    )
    assert sum(int((modes == 4).sum()) for modes in evaluated_modes) == len(examples)
    for row, example in enumerate(examples):
        single = teacher_forced_unroll(model, pack_training_examples((example,)))
        torch.testing.assert_close(
            result.final_context.workspace.latent[row],
            single.final_context.workspace.latent[0],
            atol=3e-6,
            rtol=3e-5,
        )
        last = len(batch.teacher_traces[row].steps) - 1
        torch.testing.assert_close(
            inputs.final_guard_targets[row, last],
            result.steps[last].post_parameters.raw_guard_targets[row],
        )
    assert inputs.final_dormancy_mask.sum() == len(examples)
    assert inputs.abstention_mask.sum() == 2
    assert result.compute.module_calls["FlowGuardController"] == (
        batch.public.observation_mask.shape[1] + batch.targets.kind.shape[1]
    )


def test_recall_is_rescored_after_flow_and_before_record_choice(model, examples):
    from silent_cascade.train.unroll import teacher_forced_unroll

    batch = positive_batch(examples)
    result = teacher_forced_unroll(model, batch)
    step = result.steps[0]
    expected = model.recall_scores(step.prediction_context)
    torch.testing.assert_close(result.loss_inputs().retrieval_logits[:, 0], expected.masked_logits)
    assert (step.prediction_context.active_slot_indices == -1).all()
    assert torch.equal(step.post_context.active_slot_indices, batch.targets.selected_slot[:, 0])
    assert not torch.equal(
        step.prediction_context.workspace.latent, result.boundaries[0].context.workspace.latent
    )


def test_slot_permutation_preserves_predictions_and_teacher_selection(model, examples):
    from silent_cascade.train.unroll import teacher_forced_unroll

    batch = pack_training_examples(examples)
    permutation = torch.arange(63, -1, -1).expand(len(examples), -1)
    first = teacher_forced_unroll(model, batch)
    second = teacher_forced_unroll(model, batch.permute_slots(permutation))
    torch.testing.assert_close(
        first.final_context.workspace.latent,
        second.final_context.workspace.latent,
        atol=3e-6,
        rtol=3e-5,
    )
    torch.testing.assert_close(
        first.loss_inputs().retrieval_logits.flip(-1),
        second.loss_inputs().retrieval_logits,
        atol=3e-6,
        rtol=3e-5,
    )


def test_two_optimizer_steps_build_fresh_graphs(model, examples):
    from silent_cascade.train.unroll import teacher_forced_unroll

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch = positive_batch(examples)
    losses = []
    for _ in range(2):
        optimizer.zero_grad()
        inputs = teacher_forced_unroll(model, batch).loss_inputs()
        loss = torch.nn.functional.cross_entropy(
            inputs.action_logits[inputs.action_mask], inputs.action_target[inputs.action_mask]
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(losses))) and losses[0] != losses[1]


def test_absolute_clock_translation_preserves_hypothesis_features(model, examples):
    from silent_cascade.train.unroll import teacher_forced_unroll

    batch = positive_batch(examples)
    shift = 1_000_000_000.0
    records = replace(
        batch.public.records,
        observed_at=tuple(
            tuple(None if value is None else value + shift for value in row)
            for row in batch.public.records.observed_at
        ),
    )
    public = replace(
        batch.public,
        records=records,
        observation_times=batch.public.observation_times + shift,
        initial_times=tuple(t + shift for t in batch.public.initial_times),
    )
    shifted = replace(batch, public=public, teacher_times=batch.teacher_times + shift)
    first = teacher_forced_unroll(model, batch)
    second = teacher_forced_unroll(model, shifted)
    torch.testing.assert_close(
        first.steps[3].post_context.hypothesis_features,
        second.steps[3].post_context.hypothesis_features,
        atol=2e-6,
        rtol=2e-6,
    )


def test_negative_action_uses_final_post_compose_and_no_extra_tick(model, examples):
    from silent_cascade.train.unroll import teacher_forced_unroll

    for example in examples:
        if example.solution.hazard_type is not None:
            continue
        result = teacher_forced_unroll(model, pack_training_examples((example,)))
        assert result.steps[-1].post_context.modes.item() == 4
        expected = model.action(result.steps[-1].post_context)
        torch.testing.assert_close(result.steps[-1].action.class_logits, expected.class_logits)
        assert result.compute.module_calls["ActionHeads"] == 1
        assert result.compute.module_calls["RetrievalScorer"] == 1 + len(result.steps) + (
            len(result.steps) // 2
        )


def test_teacher_record_is_disclosed_only_after_recall_scores(model, examples):
    from silent_cascade.train.unroll import teacher_forced_unroll

    batch = positive_batch(examples)
    selected = batch.targets.selected_slot.clone()
    alternatives = batch.public.records.valid_mask[0].nonzero(as_tuple=True)[0]
    selected[0, 0] = next(slot for slot in alternatives if slot != selected[0, 0])
    altered = replace(batch, targets=replace(batch.targets, selected_slot=selected))
    first = teacher_forced_unroll(model, batch)
    second = teacher_forced_unroll(model, altered)
    torch.testing.assert_close(
        first.steps[0].retrieval.raw_scores, second.steps[0].retrieval.raw_scores, rtol=0, atol=0
    )
    torch.testing.assert_close(
        first.boundaries[0].raw_guard_targets,
        second.boundaries[0].raw_guard_targets,
        rtol=0,
        atol=0,
    )


def test_null_hypothesis_is_absent_and_terminal_jump_sees_disclosed_deadline(model, examples):
    from silent_cascade.train.unroll import teacher_forced_unroll

    jump_inputs = []
    handle = model.shared_jump.register_forward_pre_hook(
        lambda module, args: jump_inputs.append(args[0].hypothesis_features)
    )
    try:
        result = teacher_forced_unroll(model, pack_training_examples(examples))
    finally:
        handle.remove()
    for row, example in enumerate(examples):
        if example.solution.terminal_kind == "disconnected":
            assert not result.final_context.hypothesis_features[row].any()
    # At the fourth position, both positive and safe rows remain active.
    rows = result.targets.real_step_mask[:, 3].nonzero(as_tuple=True)[0]
    torch.testing.assert_close(
        jump_inputs[3], result.steps[3].post_context.hypothesis_features[rows]
    )


def test_compose_jump_keeps_active_record_until_after_prediction_and_jump(model, examples):
    from silent_cascade.train.unroll import teacher_forced_unroll

    seen = []
    handle = model.shared_jump.register_forward_pre_hook(
        lambda module, args: seen.append((args[0], args[1]))
    )
    try:
        result = teacher_forced_unroll(model, positive_batch(examples))
    finally:
        handle.remove()
    for col in (1, 3):
        jump_context, kind = seen[col]
        before = result.steps[col].prediction_context
        assert kind.item() == 1
        assert torch.equal(jump_context.active_slot_indices, before.active_slot_indices)
        assert jump_context.active_slot_indices.item() >= 0
        assert result.steps[col].post_context.active_slot_indices.item() == -1
        slot = int(before.active_slot_indices.item())
        assert not before.support_mask[0, slot]
        assert jump_context.support_mask[0, slot]
        assert not result.steps[col].post_context.eligibility[0, slot]


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_complete_length_four_trace(model, neural_config, device):
    from silent_cascade.train.unroll import teacher_forced_unroll

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS is not exposed")
    for i in range(40):
        example = make_curriculum_example(
            neural_config, CurriculumKey("ofd-one-hop-v1", "debug", 311, 331, i, "primary")
        )
        if len(example.solution.link_record_ids) == 4 and example.solution.hazard_type is not None:
            break
    else:
        pytest.fail("fixture did not yield a positive four-link trace")
    result = teacher_forced_unroll(model.to(device), pack_training_examples((example,)).to(device))
    inputs = result.loss_inputs()
    assert inputs.boundary_mask.sum() == 11
    inputs.action_logits[inputs.action_mask].sum().backward()
    assert torch.isfinite(model.shared_jump.network[0].weight.grad).all()
