"""Hand-derived auxiliary reductions; invalid storage must never enter arithmetic."""

import math
from dataclasses import fields, replace

import pytest
import torch

from silent_cascade.models.errors import NeuralError


def content_fixture():
    from silent_cascade.models.content_types import ContentLossInputs

    shape = (2, 3)

    def mask(*positions):
        value = torch.zeros(shape, dtype=torch.bool)
        for position in positions:
            value[position] = True
        return value

    recall = mask((0, 0))
    compose = mask((0, 1), (1, 0), (1, 1))
    terminal = mask((1, 0), (1, 1))
    values = {"boundary_mask": recall | compose, "recall_mask": recall}
    raw = torch.full(shape, float("nan"))
    raw[recall] = 0.8
    values["raw_recall_target"] = raw.requires_grad_()
    eligible = torch.zeros((*shape, 64), dtype=torch.bool)
    eligible[0, 0, [2, 7]] = True
    logits = torch.full((*shape, 64), float("nan"))
    logits[eligible] = 0
    values["retrieval_logits"] = logits.requires_grad_()
    values["retrieval_eligible_mask"] = eligible
    values["retrieval_target"] = torch.tensor([[7, -999, -999], [-999, -999, -999]])
    for name, prediction, width, selected, labels in (
        ("role", "role_logits", 5, compose, [0, 1, 2]),
        ("status", "status_logits", 3, terminal, [0, 1]),
        ("confidence", "confidence_logit", None, terminal, [1.0, 0.0]),
        ("append_support", "append_support_logit", None, compose, [1.0, 1.0, 1.0]),
        ("continue_search", "continue_search_logit", None, mask((0, 1)), [0.0]),
        ("focus", "next_focus_logits", 64, mask((0, 1)), [13]),
        ("hazard", "hazard_logits", 4, mask((1, 0)), [2]),
        ("log_delay", "log_delay", None, mask((1, 0)), [2.0]),
    ):
        predictions = torch.full((*shape, width) if width else shape, float("nan"))
        predictions[selected] = 0.0
        target = (
            torch.full(shape, -999, dtype=torch.int64) if width else torch.full(shape, float("nan"))
        )
        target[selected] = torch.tensor(labels, dtype=target.dtype)
        values[prediction] = predictions.requires_grad_()
        values[name + "_target"] = target
        values[name + "_mask"] = selected
    return ContentLossInputs(**values)


def test_every_auxiliary_reduction_matches_hand_derived_masks_and_coefficients():
    from silent_cascade.models.content_loss import content_loss

    fixture = content_fixture()
    loss = content_loss(fixture)
    expected = {
        "recall_available": (0.3, 1, "recall"),
        "retrieval": (math.log(2), 1, "recall"),
        "compose_role": (3 * math.log(5), 3, "role"),
        "compose_status": (2 * math.log(3), 2, "status"),
        "compose_confidence": (2 * math.log(2), 2, "confidence"),
        "compose_append_support": (3 * math.log(2), 3, "append_support"),
        "compose_continue_search": (math.log(2), 1, "continue_search"),
        "focus": (math.log(64), 1, "focus"),
        "hazard": (math.log(4), 1, "hazard"),
        "log_delay": (1.5, 1, "log_delay"),
    }
    for name, (numerator, denominator, mask_name) in expected.items():
        assert loss.numerators[name].item() == pytest.approx(numerator)
        assert loss.denominators[name].item() == denominator
        assert loss.subterms[name].item() == pytest.approx(numerator / denominator)
        per_position = loss.per_position[name]
        assert per_position.shape == (2, 3)
        selected = getattr(fixture, mask_name + "_mask")
        assert (per_position[~selected] == 0).all()
        torch.testing.assert_close(per_position[selected].sum(), loss.numerators[name])
    compose = (math.log(5) + math.log(3) + 3 * math.log(2)) / 5
    assert loss.terms["compose_type"].item() == pytest.approx(compose)
    assert set(loss.terms) == {
        "recall_available",
        "retrieval",
        "compose_type",
        "focus",
        "hazard",
        "log_delay",
    }
    assert loss.total.item() == pytest.approx(
        0.3 + math.log(2) + compose + 0.5 * math.log(64) + math.log(4) + 0.125 * 1.5
    )
    loss.total.backward()
    assert fixture.raw_recall_target.grad[0, 0].item() == pytest.approx(-1.0)
    for field in fields(fixture):
        tensor = getattr(fixture, field.name)
        if tensor.requires_grad:
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_ineligible_retrieval_target_is_rejected_even_with_large_logit():
    from silent_cascade.models.content_loss import content_loss

    fixture = content_fixture()
    logits = fixture.retrieval_logits.detach().clone()
    logits[0, 0, 9] = 1000
    target = fixture.retrieval_target.clone()
    target[0, 0] = 9
    with pytest.raises(NeuralError, match="eligible"):
        content_loss(replace(fixture, retrieval_logits=logits, retrieval_target=target))


def test_empty_selections_are_graph_connected_zero_despite_nan_storage():
    from silent_cascade.models.content_loss import content_loss

    fixture = content_fixture()
    fixture = replace(
        fixture,
        **{
            item.name: torch.zeros_like(getattr(fixture, item.name))
            for item in fields(fixture)
            if item.name.endswith("_mask")
        },
    )
    loss = content_loss(fixture)
    assert loss.total.item() == 0 and loss.total.requires_grad
    assert all(value.item() == 0 for value in loss.denominators.values())
    assert all((value == 0).all() for value in loss.per_position.values())
    loss.total.backward()
    for item in fields(fixture):
        tensor = getattr(fixture, item.name)
        if tensor.requires_grad:
            assert tensor.grad is not None and (tensor.grad == 0).all()


@pytest.mark.parametrize("field", ["recall_mask", "role_mask", "log_delay_mask"])
def test_loss_masks_cannot_select_padding(field):
    from silent_cascade.models.content_loss import content_loss

    fixture = content_fixture()
    mask = getattr(fixture, field).clone()
    mask[0, 2] = True
    with pytest.raises(NeuralError, match="padding"):
        content_loss(replace(fixture, **{field: mask}))


@pytest.mark.parametrize("field", ["role_logits", "hazard_target", "retrieval_eligible_mask"])
def test_loss_requires_exact_shape_dtype_and_device(field):
    from silent_cascade.models.content_loss import content_loss

    fixture = content_fixture()
    with pytest.raises(NeuralError, match=field):
        content_loss(replace(fixture, **{field: getattr(fixture, field).to(torch.float64)}))
