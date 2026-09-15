import math

import pytest
import torch

from silent_cascade.models.config import LossWeights
from silent_cascade.models.errors import NeuralError
from silent_cascade.models.types import LossInputs


def _float(value):
    tensor = value.detach().clone() if isinstance(value, torch.Tensor) else torch.tensor(value)
    return tensor.to(torch.float32).requires_grad_(True)


def loss_inputs(**overrides) -> LossInputs:
    shape = (2, 2)
    values = {
        "raw_guard_targets": _float(torch.zeros((*shape, 3))),
        "crossing_offsets": _float(torch.full((*shape, 3), torch.inf)),
        "legal_guard_mask": torch.zeros((*shape, 3), dtype=torch.bool),
        "boundary_mask": torch.zeros(shape, dtype=torch.bool),
        "kind_target": torch.full(shape, -1, dtype=torch.int64),
        "delta_target": torch.zeros(shape, dtype=torch.float32),
        "final_guard_targets": _float(torch.zeros((*shape, 3))),
        "final_dormancy_mask": torch.zeros(shape, dtype=torch.bool),
        "retrieval_logits": _float(torch.zeros((*shape, 64))),
        "retrieval_eligible_mask": torch.zeros((*shape, 64), dtype=torch.bool),
        "retrieval_target": torch.full(shape, -1, dtype=torch.int64),
        "retrieval_mask": torch.zeros(shape, dtype=torch.bool),
        "role_logits": _float(torch.zeros((*shape, 5))),
        "role_target": torch.full(shape, -1, dtype=torch.int64),
        "role_mask": torch.zeros(shape, dtype=torch.bool),
        "status_logits": _float(torch.zeros((*shape, 3))),
        "status_target": torch.full(shape, -1, dtype=torch.int64),
        "status_mask": torch.zeros(shape, dtype=torch.bool),
        "confidence_logit": _float(torch.zeros(shape)),
        "confidence_target": torch.zeros(shape, dtype=torch.float32),
        "confidence_mask": torch.zeros(shape, dtype=torch.bool),
        "append_support_logit": _float(torch.zeros(shape)),
        "append_support_target": torch.zeros(shape, dtype=torch.float32),
        "append_support_mask": torch.zeros(shape, dtype=torch.bool),
        "continue_search_logit": _float(torch.zeros(shape)),
        "continue_search_target": torch.zeros(shape, dtype=torch.float32),
        "continue_search_mask": torch.zeros(shape, dtype=torch.bool),
        "next_focus_logits": _float(torch.zeros((*shape, 64))),
        "focus_target": torch.full(shape, -1, dtype=torch.int64),
        "focus_mask": torch.zeros(shape, dtype=torch.bool),
        "hazard_logits": _float(torch.zeros((*shape, 4))),
        "hazard_target": torch.full(shape, -1, dtype=torch.int64),
        "hazard_mask": torch.zeros(shape, dtype=torch.bool),
        "log_delay": _float(torch.zeros(shape)),
        "log_delay_target": torch.zeros(shape, dtype=torch.float32),
        "log_delay_mask": torch.zeros(shape, dtype=torch.bool),
        "normalized_deadline": _float(torch.zeros(shape)),
        "normalized_deadline_target": torch.zeros(shape, dtype=torch.float32),
        "normalized_deadline_mask": torch.zeros(shape, dtype=torch.bool),
        "action_logits": _float(torch.zeros((*shape, 5))),
        "action_target": torch.full(shape, -1, dtype=torch.int64),
        "action_mask": torch.zeros(shape, dtype=torch.bool),
        "abstention_mask": torch.zeros(shape, dtype=torch.bool),
        "lead_fraction": _float(torch.zeros(shape)),
        "lead_target": torch.zeros(shape, dtype=torch.float32),
        "lead_mask": torch.zeros(shape, dtype=torch.bool),
        "pre_jump_latent": _float(torch.zeros((2, 2, 456))),
        "post_jump_latent": _float(torch.zeros((2, 2, 456))),
        "jump_mask": torch.zeros((2, 2), dtype=torch.bool),
    }
    values.update(overrides)
    return LossInputs(**values)


def hand_calculated_inputs() -> LossInputs:
    boundary = torch.tensor([[True, True], [False, True]])
    legal = boundary.unsqueeze(-1).expand(-1, -1, 3).clone()
    raw = torch.tensor(
        [[[1.0, 0.8, 1.0], [1.1, 0.8, 0.7]], [[float("nan")] * 3, [0.95, 0.85, 0.9]]]
    )
    crossings = torch.tensor(
        [
            [[2.0, 2.02, float("inf")], [float("inf")] * 3],
            [[float("nan")] * 3, [0.4, 0.8, 0.5]],
        ]
    )
    final = torch.tensor(
        [
            [[float("nan")] * 3, [1.0, 0.8, 0.7]],
            [[float("nan")] * 3, [0.9, 1.2, 1.0]],
        ]
    )
    retrieval_logits = torch.full((2, 2, 64), float("nan"))
    retrieval_logits[0, 0] = -torch.inf
    retrieval_logits[0, 0, :3] = torch.tensor([0.0, math.log(2.0), 0.0])
    eligible = torch.zeros((2, 2, 64), dtype=torch.bool)
    eligible[0, 0, :3] = True
    role_logits = torch.full((2, 2, 5), float("nan"))
    role_logits[0, 1] = 0.0
    status_logits = torch.full((2, 2, 3), float("nan"))
    status_logits[0, 1] = 0.0
    binary = torch.full((2, 2), float("nan"))
    binary[0, 1] = 0.0
    focus_logits = torch.full((2, 2, 64), float("nan"))
    focus_logits[0, 1] = 0.0
    hazard_logits = torch.full((2, 2, 4), float("nan"))
    hazard_logits[1, 1] = 0.0
    log_delay = torch.full((2, 2), float("nan"))
    log_delay[1, 1] = 0.5
    deadline = torch.full((2, 2), float("nan"))
    deadline[1, 1] = 2.0
    action_logits = torch.full((2, 2, 5), float("nan"))
    action_logits[1, 1] = 0.0
    action_logits[0, 1] = torch.tensor([0.0, 0.0, 0.0, 0.0, math.log(4.0)])
    lead = torch.full((2, 2), float("nan"))
    lead[1, 1] = 0.675
    pre = torch.full((2, 2, 456), float("nan"))
    post = torch.full((2, 2, 456), float("nan"))
    pre[0, 0] = 0.0
    post[0, 0] = 0.0
    post[0, 0, 256:320] = 0.5
    pre[1, 1] = 0.0
    pre[1, 1, 256:320] = 0.25
    post[1, 1] = 0.0
    post[1, 1, 0] = 1.5
    post[1, 1, 256:320] = 0.75
    return loss_inputs(
        raw_guard_targets=_float(raw),
        crossing_offsets=_float(crossings),
        legal_guard_mask=legal,
        boundary_mask=boundary,
        kind_target=torch.tensor([[0, 1], [-1, 2]]),
        delta_target=torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
        final_guard_targets=_float(final),
        final_dormancy_mask=torch.tensor([[False, True], [False, True]]),
        retrieval_logits=_float(retrieval_logits),
        retrieval_eligible_mask=eligible,
        retrieval_target=torch.tensor([[1, -1], [-1, -1]]),
        retrieval_mask=torch.tensor([[True, False], [False, False]]),
        role_logits=_float(role_logits),
        role_target=torch.tensor([[-1, 2], [-1, -1]]),
        role_mask=torch.tensor([[False, True], [False, False]]),
        status_logits=_float(status_logits),
        status_target=torch.tensor([[-1, 1], [-1, -1]]),
        status_mask=torch.tensor([[False, True], [False, False]]),
        confidence_logit=_float(binary.clone()),
        confidence_target=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        confidence_mask=torch.tensor([[False, True], [False, False]]),
        append_support_logit=_float(binary.clone()),
        append_support_target=torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        append_support_mask=torch.tensor([[False, True], [False, False]]),
        continue_search_logit=_float(binary.clone()),
        continue_search_target=torch.tensor([[0.0, 1.0], [0.0, 0.0]]),
        continue_search_mask=torch.tensor([[False, True], [False, False]]),
        next_focus_logits=_float(focus_logits),
        focus_target=torch.tensor([[-1, 17], [-1, -1]]),
        focus_mask=torch.tensor([[False, True], [False, False]]),
        hazard_logits=_float(hazard_logits),
        hazard_target=torch.tensor([[-1, -1], [-1, 3]]),
        hazard_mask=torch.tensor([[False, False], [False, True]]),
        log_delay=_float(log_delay),
        log_delay_target=torch.zeros((2, 2)),
        log_delay_mask=torch.tensor([[False, False], [False, True]]),
        normalized_deadline=_float(deadline),
        normalized_deadline_target=torch.zeros((2, 2)),
        normalized_deadline_mask=torch.tensor([[False, False], [False, True]]),
        action_logits=_float(action_logits),
        action_target=torch.tensor([[-1, 4], [-1, 2]]),
        action_mask=torch.tensor([[False, False], [False, True]]),
        abstention_mask=torch.tensor([[False, True], [False, False]]),
        lead_fraction=_float(lead),
        lead_target=torch.tensor([[0.0, 0.0], [0.0, 0.175]]),
        lead_mask=torch.tensor([[False, False], [False, True]]),
        pre_jump_latent=_float(pre),
        post_jump_latent=_float(post),
        jump_mask=torch.tensor([[True, False], [False, True]]),
    )


def test_masked_mean_does_not_evaluate_invalid_padding():
    from silent_cascade.models.losses import masked_mean

    values = torch.tensor([2.0, float("nan")], requires_grad=True)
    result = masked_mean(values, torch.tensor([True, False]))
    assert result.item() == 2.0
    result.backward()
    torch.testing.assert_close(values.grad, torch.tensor([1.0, 0.0]))


def test_hand_calculated_guard_reductions_and_boundary_denominators():
    from silent_cascade.models.losses import event_flow_loss

    result = event_flow_loss(hand_calculated_inputs(), LossWeights())
    expected_time = 0.5 * math.log(2.0) ** 2
    expected = {
        "guard_active_margin": (0.6, 3),
        "guard_inactive_margin": (0.175, 3),
        "guard_time": (2.0 * expected_time, 2),
        "guard_race": (0.105, 2),
        "guard_final_dormancy": (1.0 / 6.0, 2),
    }
    for name, (numerator, denominator) in expected.items():
        assert result.numerators[name].item() == pytest.approx(numerator, abs=2e-6)
        assert result.denominators[name].item() == denominator
        assert result.subterms[name].item() == pytest.approx(numerator / denominator, abs=2e-6)
    assert result.terms["guard"].item() == pytest.approx(
        0.6 / 3 + 0.175 / 3 + expected_time + 0.105 / 2 + 1.0 / 12.0,
        abs=2e-6,
    )
    assert result.per_position["guard_time"].shape == (2, 2)
    assert result.per_position["guard_time"][0, 1].item() == 0.0


def test_hand_calculated_classification_and_composition_reductions():
    from silent_cascade.models.losses import event_flow_loss

    result = event_flow_loss(hand_calculated_inputs(), LossWeights())
    assert result.terms["retrieval"].item() == pytest.approx(math.log(2.0), abs=2e-6)
    assert result.terms["compose_type"].item() == pytest.approx(
        (math.log(5.0) + math.log(3.0) + 3.0 * math.log(2.0)) / 5.0,
        abs=2e-6,
    )
    assert result.terms["focus"].item() == pytest.approx(math.log(64.0), abs=2e-6)
    assert result.terms["hazard"].item() == pytest.approx(math.log(4.0), abs=2e-6)
    for name in (
        "retrieval",
        "compose_role",
        "compose_status",
        "compose_confidence",
        "compose_append_support",
        "compose_continue_search",
        "focus",
        "hazard",
    ):
        assert result.denominators[name].item() == 1


def test_hand_calculated_regression_action_state_and_cost_reductions():
    from silent_cascade.models.losses import event_flow_loss

    result = event_flow_loss(hand_calculated_inputs(), LossWeights())
    assert result.terms["deadline"].item() == pytest.approx((0.125 + 1.5) / 2.0)
    assert result.terms["action"].item() == pytest.approx(
        (math.log(5.0) + math.log(2.0)) / 2.0, abs=2e-6
    )
    assert result.terms["action_time"].item() == pytest.approx(0.125)
    assert result.terms["state"].item() == pytest.approx(0.25 + 0.125 / 456.0)
    assert result.terms["event_cost"].item() == pytest.approx(0.1 / 9.0, abs=2e-6)
    expected_total = sum(
        result.terms[name].item() * getattr(LossWeights(), name)
        for name in (
            "guard",
            "retrieval",
            "compose_type",
            "focus",
            "hazard",
            "deadline",
            "action",
            "action_time",
            "state",
            "event_cost",
        )
    )
    assert result.total.item() == pytest.approx(expected_total, abs=2e-6)


def test_empty_masks_select_before_all_arithmetic_and_remain_graph_connected():
    from silent_cascade.models.losses import event_flow_loss

    inputs = loss_inputs(
        raw_guard_targets=_float(torch.full((2, 2, 3), float("nan"))),
        crossing_offsets=_float(torch.full((2, 2, 3), float("inf"))),
        final_guard_targets=_float(torch.full((2, 2, 3), float("nan"))),
        retrieval_logits=_float(torch.full((2, 2, 64), float("-inf"))),
        role_logits=_float(torch.full((2, 2, 5), float("nan"))),
        status_logits=_float(torch.full((2, 2, 3), float("nan"))),
        confidence_logit=_float(torch.full((2, 2), float("nan"))),
        append_support_logit=_float(torch.full((2, 2), float("nan"))),
        continue_search_logit=_float(torch.full((2, 2), float("nan"))),
        next_focus_logits=_float(torch.full((2, 2, 64), float("nan"))),
        hazard_logits=_float(torch.full((2, 2, 4), float("nan"))),
        log_delay=_float(torch.full((2, 2), float("nan"))),
        normalized_deadline=_float(torch.full((2, 2), float("nan"))),
        action_logits=_float(torch.full((2, 2, 5), float("nan"))),
        lead_fraction=_float(torch.full((2, 2), float("nan"))),
        pre_jump_latent=_float(torch.full((2, 2, 456), float("nan"))),
        post_jump_latent=_float(torch.full((2, 2, 456), float("nan"))),
    )
    result = event_flow_loss(inputs, LossWeights())
    assert result.total.item() == 0.0
    assert set(result.terms) == {
        "guard",
        "retrieval",
        "compose_type",
        "focus",
        "hazard",
        "deadline",
        "action",
        "action_time",
        "state",
        "event_cost",
    }
    result.total.backward()
    for value in (
        inputs.raw_guard_targets,
        inputs.crossing_offsets,
        inputs.retrieval_logits,
        inputs.action_logits,
        inputs.post_jump_latent,
    ):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert not value.grad.any()


def test_masked_padding_and_race_losers_have_only_intended_gradients():
    from silent_cascade.models.losses import event_flow_loss

    inputs = hand_calculated_inputs()
    result = event_flow_loss(inputs, LossWeights())
    result.total.backward()
    assert not inputs.raw_guard_targets.grad[1, 0].any()
    assert not inputs.crossing_offsets.grad[1, 0].any()
    assert not inputs.retrieval_logits.grad[1, 0].any()
    assert inputs.raw_guard_targets.grad[0, 1, 1].item() < 0.0
    assert inputs.crossing_offsets.grad[0, 0, 1].item() < 0.0
    assert inputs.crossing_offsets.grad[1, 1, 0].item() < 0.0
    assert inputs.crossing_offsets.grad[1, 1, 1].item() == 0.0
    assert inputs.action_logits.grad[0, 1].abs().sum() > 0
    assert inputs.action_logits.grad[1, 1].abs().sum() > 0


def test_retrieval_target_requires_a_legally_eligible_slot():
    from silent_cascade.models.losses import event_flow_loss

    inputs = loss_inputs(
        boundary_mask=torch.tensor([[True, False], [False, False]]),
        kind_target=torch.tensor([[0, -1], [-1, -1]]),
        delta_target=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        retrieval_mask=torch.tensor([[True, False], [False, False]]),
        retrieval_target=torch.tensor([[4, -1], [-1, -1]]),
    )
    with pytest.raises(NeuralError, match="retrieval target"):
        event_flow_loss(inputs, LossWeights())


def test_loss_contract_rejects_malformed_masks_with_neural_error():
    from silent_cascade.models.losses import event_flow_loss

    inputs = loss_inputs(boundary_mask=torch.zeros((2, 2), dtype=torch.int64))
    with pytest.raises(NeuralError, match="boundary_mask"):
        event_flow_loss(inputs, LossWeights())

    with pytest.raises(TypeError, match="LossInputs"):
        event_flow_loss(object(), LossWeights())
