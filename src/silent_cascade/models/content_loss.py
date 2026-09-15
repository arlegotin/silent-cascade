"""Fixed auxiliary content loss, separate from the unchanged timed objective."""

from dataclasses import fields
from types import MappingProxyType

import torch
from torch.nn import functional as F

from silent_cascade.models.content_types import ContentLossInputs
from silent_cascade.models.errors import NeuralError
from silent_cascade.models.losses import (
    LossBreakdown,
    _binary,
    _classification,
    _reduce_selected,
    _require_finite,
    _smooth_l1,
)


def _require_inputs(inputs):
    if not isinstance(inputs, ContentLossInputs):
        raise TypeError("content_loss requires ContentLossInputs")
    boundary = inputs.boundary_mask
    if not isinstance(boundary, torch.Tensor) or boundary.ndim != 2:
        raise NeuralError("boundary_mask must be a rank-two tensor")
    shape = tuple(boundary.shape)
    if not 1 <= shape[0] <= 128 or not 1 <= shape[1] <= 11:
        raise NeuralError("boundary_mask must have bounded [B,T] shape")
    if boundary.dtype is not torch.bool or boundary.device.type not in {"cpu", "mps"}:
        raise NeuralError("boundary_mask must be a boolean CPU/MPS tensor")
    widths = {
        "retrieval_logits": 64,
        "retrieval_eligible_mask": 64,
        "role_logits": 5,
        "status_logits": 3,
        "next_focus_logits": 64,
        "hazard_logits": 4,
    }
    integer_names = {
        "retrieval_target",
        "role_target",
        "status_target",
        "focus_target",
        "hazard_target",
    }
    for item in fields(inputs):
        name, value = item.name, getattr(inputs, item.name)
        expected = (*shape, widths[name]) if name in widths else shape
        dtype = (
            torch.bool
            if name.endswith("_mask")
            else torch.int64
            if name in integer_names
            else torch.float32
        )
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != expected
            or value.dtype is not dtype
            or value.device != boundary.device
        ):
            raise NeuralError(f"{name} must match the content loss shape, dtype, and device")
        if name.endswith("_mask") and name != "retrieval_eligible_mask":
            if bool((value & ~boundary).any()):
                raise NeuralError(f"{name} cannot select padding")
            if name not in {"boundary_mask", "recall_mask"} and bool(
                (value & inputs.recall_mask).any()
            ):
                raise NeuralError(f"{name} may select only COMPOSE positions")
    return shape


def _eligible_retrieval(inputs, shape):
    mask = inputs.recall_mask
    losses = []
    for row, col in mask.nonzero(as_tuple=False).tolist():
        eligible = inputs.retrieval_eligible_mask[row, col]
        target = int(inputs.retrieval_target[row, col].item())
        if not 0 <= target < 64 or not bool(eligible[target]):
            raise NeuralError("retrieval target must identify a legally eligible slot")
        logits = inputs.retrieval_logits[row, col, eligible]
        _require_finite(logits, "retrieval logits")
        local = (eligible.nonzero(as_tuple=True)[0] == target).nonzero(as_tuple=True)[0]
        losses.append(F.cross_entropy(logits.unsqueeze(0), local, reduction="none")[0])
    selected = torch.stack(losses) if losses else inputs.retrieval_logits[mask].sum(dim=1)
    return _reduce_selected(selected, mask, position_shape=shape)


def content_loss(inputs: ContentLossInputs) -> LossBreakdown:
    """Mean content subterms with fixed weights 1, 1, 1, .5, 1, .125.

    As in the timed loss, compose_type averages five independently masked
    means. Its component numerators/denominators live under compose_* keys;
    aggregating their raw numerators would change the objective on ragged data.
    """
    shape = _require_inputs(inputs)
    raw = inputs.raw_recall_target[inputs.recall_mask]
    _require_finite(raw, "raw recall target")
    reductions = {
        "recall_available": _reduce_selected(
            F.relu(1.10 - raw), inputs.recall_mask, position_shape=shape
        ),
        "retrieval": _eligible_retrieval(inputs, shape),
    }
    for name, logits in (
        ("role", inputs.role_logits),
        ("status", inputs.status_logits),
        ("focus", inputs.next_focus_logits),
        ("hazard", inputs.hazard_logits),
    ):
        key = "compose_" + name if name in {"role", "status"} else name
        reductions[key] = _classification(
            logits, getattr(inputs, name + "_target"), getattr(inputs, name + "_mask"), name
        )
    for name in ("confidence", "append_support", "continue_search"):
        reductions["compose_" + name] = _binary(
            getattr(inputs, name + "_logit"),
            getattr(inputs, name + "_target"),
            getattr(inputs, name + "_mask"),
            name,
        )
    reductions["log_delay"] = _smooth_l1(
        inputs.log_delay, inputs.log_delay_target, inputs.log_delay_mask, "log delay"
    )
    compose = ("role", "status", "confidence", "append_support", "continue_search")
    terms = {
        name: reductions[name].value
        for name in ("recall_available", "retrieval", "focus", "hazard", "log_delay")
    }
    terms["compose_type"] = torch.stack(
        [reductions["compose_" + name].value for name in compose]
    ).mean()
    total = (
        terms["recall_available"]
        + terms["retrieval"]
        + terms["compose_type"]
        + 0.5 * terms["focus"]
        + terms["hazard"]
        + 0.125 * terms["log_delay"]
    )
    if not bool(torch.isfinite(total)):
        raise NeuralError("content loss must be finite")
    return LossBreakdown(
        total=total,
        terms=MappingProxyType(terms),
        subterms=MappingProxyType({name: value.value for name, value in reductions.items()}),
        numerators=MappingProxyType({name: value.numerator for name, value in reductions.items()}),
        denominators=MappingProxyType(
            {name: value.denominator for name, value in reductions.items()}
        ),
        per_position=MappingProxyType(
            {name: value.per_position for name, value in reductions.items()}
        ),
    )
