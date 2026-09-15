"""Masked differentiable objectives for teacher-forced EventFlow traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch.nn import functional as F

from silent_cascade.models.config import LossWeights
from silent_cascade.models.errors import NeuralError
from silent_cascade.models.types import LossInputs

_TERM_NAMES = (
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


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    """Complete differentiable objective and its unrounded diagnostic reductions."""

    total: torch.Tensor
    terms: Mapping[str, torch.Tensor]
    subterms: Mapping[str, torch.Tensor]
    numerators: Mapping[str, torch.Tensor]
    denominators: Mapping[str, torch.Tensor]
    per_position: Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class _Reduction:
    value: torch.Tensor
    numerator: torch.Tensor
    denominator: torch.Tensor
    per_position: torch.Tensor


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean only selected entries without evaluating invalid padding values."""
    if not isinstance(values, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("masked_mean requires tensors")
    if mask.dtype is not torch.bool or mask.device != values.device or mask.shape != values.shape:
        raise NeuralError("masked_mean mask must match values")
    selected = values[mask]
    return selected.mean() if selected.numel() else selected.sum()


def _require_loss_inputs(inputs: LossInputs) -> tuple[tuple[int, int], int, torch.device]:
    if not isinstance(inputs, LossInputs):
        raise TypeError("event_flow_loss requires LossInputs")
    boundary = inputs.boundary_mask
    if not isinstance(boundary, torch.Tensor) or boundary.ndim != 2:
        raise NeuralError("boundary_mask must be a rank-two tensor")
    shape = tuple(boundary.shape)
    if not 1 <= shape[0] <= 128 or not 1 <= shape[1] <= 11:
        raise NeuralError("boundary_mask must have bounded [B,T] shape")
    if boundary.dtype is not torch.bool or boundary.device.type not in {"cpu", "mps"}:
        raise NeuralError("boundary_mask must be a boolean CPU/MPS tensor")
    device = boundary.device
    float_shapes = {
        "raw_guard_targets": (*shape, 3),
        "crossing_offsets": (*shape, 3),
        "final_guard_targets": (*shape, 3),
        "retrieval_logits": (*shape, 64),
        "role_logits": (*shape, 5),
        "status_logits": (*shape, 3),
        "confidence_logit": shape,
        "confidence_target": shape,
        "append_support_logit": shape,
        "append_support_target": shape,
        "continue_search_logit": shape,
        "continue_search_target": shape,
        "next_focus_logits": (*shape, 64),
        "hazard_logits": (*shape, 4),
        "log_delay": shape,
        "log_delay_target": shape,
        "normalized_deadline": shape,
        "normalized_deadline_target": shape,
        "action_logits": (*shape, 5),
        "lead_fraction": shape,
        "lead_target": shape,
        "delta_target": shape,
    }
    integer_shapes = {
        "kind_target": shape,
        "retrieval_target": shape,
        "role_target": shape,
        "status_target": shape,
        "focus_target": shape,
        "hazard_target": shape,
        "action_target": shape,
    }
    boolean_shapes = {
        "legal_guard_mask": (*shape, 3),
        "boundary_mask": shape,
        "final_dormancy_mask": shape,
        "retrieval_eligible_mask": (*shape, 64),
        "retrieval_mask": shape,
        "role_mask": shape,
        "status_mask": shape,
        "confidence_mask": shape,
        "append_support_mask": shape,
        "continue_search_mask": shape,
        "focus_mask": shape,
        "hazard_mask": shape,
        "log_delay_mask": shape,
        "normalized_deadline_mask": shape,
        "action_mask": shape,
        "abstention_mask": shape,
        "lead_mask": shape,
    }
    for name, expected in float_shapes.items():
        value = getattr(inputs, name)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != expected
            or value.dtype is not torch.float32
            or value.device != device
        ):
            raise NeuralError(f"{name} must match the loss shape, float32 dtype, and device")
    for name, expected in integer_shapes.items():
        value = getattr(inputs, name)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != expected
            or value.dtype is not torch.int64
            or value.device != device
        ):
            raise NeuralError(f"{name} must match the loss shape, int64 dtype, and device")
    for name, expected in boolean_shapes.items():
        value = getattr(inputs, name)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != expected
            or value.dtype is not torch.bool
            or value.device != device
        ):
            raise NeuralError(f"{name} must match the loss shape, boolean dtype, and device")
    pre, post, jump = inputs.pre_jump_latent, inputs.post_jump_latent, inputs.jump_mask
    if (
        not isinstance(pre, torch.Tensor)
        or pre.ndim != 3
        or pre.shape[0] != shape[0]
        or pre.shape[2] != 456
        or pre.shape[1] < 1
        or pre.dtype is not torch.float32
        or pre.device != device
        or not isinstance(post, torch.Tensor)
        or post.shape != pre.shape
        or post.dtype is not torch.float32
        or post.device != device
        or not isinstance(jump, torch.Tensor)
        or jump.shape != pre.shape[:2]
        or jump.dtype is not torch.bool
        or jump.device != device
    ):
        raise NeuralError("jump tensors must align on bounded [B,J,456] float32 storage")
    for name in (
        "final_dormancy_mask",
        "retrieval_mask",
        "role_mask",
        "status_mask",
        "confidence_mask",
        "append_support_mask",
        "continue_search_mask",
        "focus_mask",
        "hazard_mask",
        "log_delay_mask",
        "normalized_deadline_mask",
        "action_mask",
        "abstention_mask",
        "lead_mask",
    ):
        if bool((getattr(inputs, name) & ~boundary).any()):
            raise NeuralError(f"{name} cannot select padding")
    if bool((inputs.legal_guard_mask & ~boundary.unsqueeze(-1)).any()):
        raise NeuralError("legal_guard_mask cannot select padding")
    if bool((inputs.action_mask & inputs.abstention_mask).any()):
        raise NeuralError("positive action and abstention masks must be disjoint")
    if bool((inputs.lead_mask & ~inputs.action_mask).any()):
        raise NeuralError("lead_mask may select only positive ACT positions")
    kinds = inputs.kind_target[boundary]
    if bool(((kinds < 0) | (kinds >= 3)).any()):
        raise NeuralError("kind_target must encode a real endogenous guard")
    delta = inputs.delta_target[boundary]
    if not bool(torch.isfinite(delta).all()) or bool((delta <= 0.0).any()):
        raise NeuralError("delta_target must be finite and positive at real boundaries")
    return shape, pre.shape[1], device


def _require_finite(values: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(values).all()):
        raise NeuralError(f"selected {name} values must be finite")


def _reduce_selected(
    selected: torch.Tensor, mask: torch.Tensor, *, position_shape: tuple[int, ...]
) -> _Reduction:
    numerator = selected.sum()
    denominator = mask.sum()
    value = numerator / denominator.to(numerator.dtype) if selected.numel() else numerator
    if selected.numel():
        per_position = selected.new_zeros(position_shape)
        per_position = per_position.masked_scatter(mask, selected)
    else:
        per_position = numerator.expand(position_shape)
    return _Reduction(value, numerator, denominator, per_position)


def _classification(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, name: str
) -> _Reduction:
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    if selected_logits.numel():
        _require_finite(selected_logits, name)
        if bool(((selected_targets < 0) | (selected_targets >= logits.shape[-1])).any()):
            raise NeuralError(f"selected {name} target is outside the class range")
        losses = F.cross_entropy(selected_logits, selected_targets, reduction="none")
    else:
        losses = selected_logits.sum(dim=1)
    return _reduce_selected(losses, mask, position_shape=tuple(mask.shape))


def _binary(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, name: str
) -> _Reduction:
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    _require_finite(selected_logits, name)
    _require_finite(selected_targets, f"{name} target")
    if bool(((selected_targets < 0.0) | (selected_targets > 1.0)).any()):
        raise NeuralError(f"selected {name} targets must be in [0, 1]")
    losses = F.binary_cross_entropy_with_logits(selected_logits, selected_targets, reduction="none")
    return _reduce_selected(losses, mask, position_shape=tuple(mask.shape))


def _smooth_l1(
    predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, name: str
) -> _Reduction:
    selected_predictions = predictions[mask]
    selected_targets = targets[mask]
    _require_finite(selected_predictions, name)
    _require_finite(selected_targets, f"{name} target")
    losses = F.smooth_l1_loss(selected_predictions, selected_targets, reduction="none")
    return _reduce_selected(losses, mask, position_shape=tuple(mask.shape))


def _retrieval(inputs: LossInputs, shape: tuple[int, int]) -> _Reduction:
    mask = inputs.retrieval_mask
    rows = mask.nonzero(as_tuple=False)
    losses = []
    for row, col in rows.tolist():
        eligible = inputs.retrieval_eligible_mask[row, col]
        target = int(inputs.retrieval_target[row, col].item())
        if not 0 <= target < inputs.retrieval_logits.shape[-1] or not bool(eligible[target]):
            raise NeuralError("retrieval target must identify a legally eligible slot")
        logits = inputs.retrieval_logits[row, col, eligible]
        _require_finite(logits, "retrieval logits")
        eligible_slots = eligible.nonzero(as_tuple=True)[0]
        local_target = (eligible_slots == target).nonzero(as_tuple=True)[0]
        losses.append(F.cross_entropy(logits.unsqueeze(0), local_target, reduction="none")[0])
    selected = torch.stack(losses) if losses else inputs.retrieval_logits[mask].sum(dim=1)
    return _reduce_selected(selected, mask, position_shape=shape)


def _guard_reductions(inputs: LossInputs, shape: tuple[int, int]) -> dict[str, _Reduction]:
    boundary = inputs.boundary_mask
    raw = inputs.raw_guard_targets[boundary]
    kinds = inputs.kind_target[boundary]
    _require_finite(raw, "raw guard targets")
    positions = torch.arange(raw.shape[0], device=raw.device)
    correct = raw[positions, kinds]
    active = _reduce_selected(F.relu(1.10 - correct), boundary, position_shape=shape)

    correct_mask = F.one_hot(kinds, num_classes=3).to(torch.bool)
    inactive_values = F.relu(raw[~correct_mask].reshape(-1, 2) - 0.90).mean(dim=1)
    inactive = _reduce_selected(inactive_values, boundary, position_shape=shape)

    crossing = inputs.crossing_offsets[boundary]
    if bool((torch.isnan(crossing) | torch.isneginf(crossing)).any()):
        raise NeuralError("real crossing offsets cannot contain NaN or negative infinity")
    correct_crossing = crossing[positions, kinds]
    temporal_rows = torch.isfinite(correct_crossing)
    temporal_mask = torch.zeros_like(boundary)
    temporal_mask[boundary] = temporal_rows
    temporal_offsets = correct_crossing[temporal_rows]
    temporal_targets = inputs.delta_target[boundary][temporal_rows]
    if temporal_offsets.numel() and bool((temporal_offsets <= 0.0).any()):
        raise NeuralError("finite correct guard crossings must be positive")
    time_losses = F.smooth_l1_loss(
        torch.log(temporal_offsets), torch.log(temporal_targets), reduction="none"
    )
    time = _reduce_selected(time_losses, temporal_mask, position_shape=shape)

    legal = inputs.legal_guard_mask[boundary]
    race_losses = []
    race_rows = []
    boundary_indices = boundary.nonzero(as_tuple=False)
    for index in temporal_rows.nonzero(as_tuple=True)[0].tolist():
        wrong_valid = legal[index] & ~correct_mask[index] & torch.isfinite(crossing[index])
        wrong = crossing[index, wrong_valid]
        if wrong.numel():
            race_losses.append(F.relu(0.05 + correct_crossing[index] - wrong).mean())
            race_rows.append(boundary_indices[index])
    race_mask = torch.zeros_like(boundary)
    if race_rows:
        race_indices = torch.stack(race_rows)
        race_mask[race_indices[:, 0], race_indices[:, 1]] = True
        race_selected = torch.stack(race_losses)
    else:
        race_selected = crossing[:, 0][:0]
    race = _reduce_selected(race_selected, race_mask, position_shape=shape)

    final_values = inputs.final_guard_targets[inputs.final_dormancy_mask]
    _require_finite(final_values, "final guard targets")
    final_losses = F.relu(final_values - 0.90).mean(dim=1)
    final = _reduce_selected(final_losses, inputs.final_dormancy_mask, position_shape=shape)
    return {
        "guard_active_margin": active,
        "guard_inactive_margin": inactive,
        "guard_time": time,
        "guard_race": race,
        "guard_final_dormancy": final,
    }


def _guard_group(inputs: LossInputs, reductions: Mapping[str, _Reduction]) -> _Reduction:
    boundary_penalties = torch.stack(
        [
            reductions[name].per_position
            for name in (
                "guard_active_margin",
                "guard_inactive_margin",
                "guard_time",
                "guard_race",
            )
        ]
    ).sum(dim=0)
    final_penalties = reductions["guard_final_dormancy"].per_position
    boundary_selected = boundary_penalties[inputs.boundary_mask]
    final_selected = final_penalties[inputs.final_dormancy_mask]
    numerator = boundary_selected.sum() + final_selected.sum()
    denominator = inputs.boundary_mask.sum() + inputs.final_dormancy_mask.sum()
    selected_count = boundary_selected.numel() + final_selected.numel()
    value = numerator / denominator.to(numerator.dtype) if selected_count else numerator
    return _Reduction(
        value=value,
        numerator=numerator,
        denominator=denominator,
        per_position=torch.stack((boundary_penalties, final_penalties), dim=-1),
    )


def _state_reductions(inputs: LossInputs, jump_count: int) -> dict[str, _Reduction]:
    mask = inputs.jump_mask
    pre = inputs.pre_jump_latent[mask]
    post = inputs.post_jump_latent[mask]
    _require_finite(pre, "pre-jump latent")
    _require_finite(post, "post-jump latent")
    slow_losses = (post[:, 256:320] - pre[:, 256:320]).square().mean(dim=1)
    bound_losses = F.relu(post.abs() - 1.0).square().mean(dim=1)
    shape = (mask.shape[0], jump_count)
    return {
        "state_slow_change": _reduce_selected(slow_losses, mask, position_shape=shape),
        "state_bound": _reduce_selected(bound_losses, mask, position_shape=shape),
    }


def _event_cost(inputs: LossInputs, shape: tuple[int, int]) -> _Reduction:
    mask = inputs.legal_guard_mask & inputs.boundary_mask.unsqueeze(-1)
    selected = inputs.raw_guard_targets[mask]
    _require_finite(selected, "event-cost guard targets")
    losses = F.relu(selected - 1.0)
    numerator = losses.sum()
    denominator = mask.sum()
    value = numerator / denominator.to(numerator.dtype) if losses.numel() else numerator
    per_position = selected.new_zeros(shape)
    if losses.numel():
        boundary_rows = inputs.boundary_mask.nonzero(as_tuple=False)
        row_losses = []
        row_mask = torch.zeros_like(inputs.boundary_mask)
        for row, col in boundary_rows.tolist():
            legal = inputs.legal_guard_mask[row, col]
            if bool(legal.any()):
                row_losses.append(F.relu(inputs.raw_guard_targets[row, col, legal] - 1.0).mean())
                row_mask[row, col] = True
        if row_losses:
            per_position = per_position.masked_scatter(row_mask, torch.stack(row_losses))
    return _Reduction(value, numerator, denominator, per_position)


def _weighted_total(terms: Mapping[str, torch.Tensor], weights: LossWeights) -> torch.Tensor:
    return torch.stack([terms[name] * getattr(weights, name) for name in _TERM_NAMES]).sum()


def event_flow_loss(inputs: LossInputs, weights: LossWeights) -> LossBreakdown:
    """Build the complete masked ten-group trace objective without detaching tensors."""
    if not isinstance(weights, LossWeights):
        raise TypeError("event_flow_loss requires LossWeights")
    shape, jump_count, _ = _require_loss_inputs(inputs)
    reductions = _guard_reductions(inputs, shape)
    reductions["retrieval"] = _retrieval(inputs, shape)
    reductions.update(
        {
            "compose_role": _classification(
                inputs.role_logits, inputs.role_target, inputs.role_mask, "compose role"
            ),
            "compose_status": _classification(
                inputs.status_logits, inputs.status_target, inputs.status_mask, "compose status"
            ),
            "compose_confidence": _binary(
                inputs.confidence_logit,
                inputs.confidence_target,
                inputs.confidence_mask,
                "compose confidence",
            ),
            "compose_append_support": _binary(
                inputs.append_support_logit,
                inputs.append_support_target,
                inputs.append_support_mask,
                "compose append support",
            ),
            "compose_continue_search": _binary(
                inputs.continue_search_logit,
                inputs.continue_search_target,
                inputs.continue_search_mask,
                "compose continue search",
            ),
            "focus": _classification(
                inputs.next_focus_logits, inputs.focus_target, inputs.focus_mask, "focus"
            ),
            "hazard": _classification(
                inputs.hazard_logits, inputs.hazard_target, inputs.hazard_mask, "hazard"
            ),
            "deadline_log_delay": _smooth_l1(
                inputs.log_delay,
                inputs.log_delay_target,
                inputs.log_delay_mask,
                "log delay",
            ),
            "deadline_normalized": _smooth_l1(
                inputs.normalized_deadline,
                inputs.normalized_deadline_target,
                inputs.normalized_deadline_mask,
                "normalized deadline",
            ),
            "action_positive": _classification(
                inputs.action_logits, inputs.action_target, inputs.action_mask, "positive action"
            ),
            "action_abstention": _classification(
                inputs.action_logits,
                inputs.action_target,
                inputs.abstention_mask,
                "action abstention",
            ),
            "action_time": _smooth_l1(
                inputs.lead_fraction, inputs.lead_target, inputs.lead_mask, "action lead"
            ),
        }
    )
    if inputs.abstention_mask.any() and bool(
        (inputs.action_target[inputs.abstention_mask] != 4).any()
    ):
        raise NeuralError("abstention positions must target the abstain class")
    reductions.update(_state_reductions(inputs, jump_count))
    reductions["event_cost"] = _event_cost(inputs, shape)
    reductions["guard"] = _guard_group(inputs, reductions)

    compose_names = (
        "compose_role",
        "compose_status",
        "compose_confidence",
        "compose_append_support",
        "compose_continue_search",
    )
    positive, abstention = reductions["action_positive"], reductions["action_abstention"]
    if positive.denominator.item() and abstention.denominator.item():
        action = (positive.value + abstention.value) / 2.0
    else:
        action = positive.value + abstention.value
    terms = {
        "guard": reductions["guard"].value,
        "retrieval": reductions["retrieval"].value,
        "compose_type": torch.stack([reductions[name].value for name in compose_names]).mean(),
        "focus": reductions["focus"].value,
        "hazard": reductions["hazard"].value,
        "deadline": torch.stack(
            [
                reductions["deadline_log_delay"].value,
                reductions["deadline_normalized"].value,
            ]
        ).mean(),
        "action": action,
        "action_time": reductions["action_time"].value,
        "state": reductions["state_slow_change"].value + reductions["state_bound"].value,
        "event_cost": reductions["event_cost"].value,
    }
    total = _weighted_total(terms, weights)
    if not bool(torch.isfinite(total)):
        raise NeuralError("event-flow loss must be finite")
    return LossBreakdown(
        total=total,
        terms=MappingProxyType(terms),
        subterms=MappingProxyType(
            {name: reduction.value for name, reduction in reductions.items()}
        ),
        numerators=MappingProxyType(
            {name: reduction.numerator for name, reduction in reductions.items()}
        ),
        denominators=MappingProxyType(
            {name: reduction.denominator for name, reduction in reductions.items()}
        ),
        per_position=MappingProxyType(
            {name: reduction.per_position for name, reduction in reductions.items()}
        ),
    )
