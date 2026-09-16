"""Public decoded decisions and shared learned jumps, without teacher access."""

from dataclasses import dataclass

import torch

from silent_cascade.eval.compute import _record_functional_operations
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.models.types import ModelContext, TensorWorkspace


@dataclass(frozen=True, slots=True)
class RecallTransition:
    selected_slots: torch.Tensor
    modes: torch.Tensor


@dataclass(frozen=True, slots=True)
class ComposeTransition:
    support_mask: torch.Tensor
    modes: torch.Tensor
    hypothesis_features: torch.Tensor
    focus_ids: torch.Tensor
    focus_mask: torch.Tensor


def apply_transition_context(
    model: EventFlowModel,
    context: ModelContext,
    kinds: torch.Tensor,
    recall: RecallTransition,
    compose: ComposeTransition,
) -> ModelContext:
    """Install decoded registers, jump once, inject LINK focus, clear COMPOSE slot.

    The mixed-row entry point preserves teacher batching and arithmetic order.
    Decisions describe every row's final registers; no labels or clocks are read.
    Eligibility/consumption remain the caller's responsibility.
    """
    active = torch.where(kinds == 0, recall.selected_slots, context.active_slot_indices)
    updated = context._updated(
        active_slot_indices=active,
        support_mask=compose.support_mask,
        modes=compose.modes,
        hypothesis_features=compose.hypothesis_features,
    )
    workspace = model.jump(updated, kinds)
    focus_rows = compose.focus_mask.nonzero(as_tuple=True)[0]
    if focus_rows.numel():
        focus = torch.tanh(
            model.focus_projection(
                model.record_encoder.encode_entities(compose.focus_ids[focus_rows])
            )
        )
        _record_functional_operations(tanh_ops=focus.numel())
        latent = torch.cat(
            (
                workspace.latent[:, :328],
                workspace.latent[:, 328:392].index_copy(0, focus_rows, focus),
                workspace.latent[:, 392:],
            ),
            1,
        )
        workspace = TensorWorkspace._from_functional_update(latent, workspace.accumulators)
    return updated._updated(
        workspace=workspace, active_slot_indices=torch.where(kinds == 1, -1, active)
    )


def apply_recall_context(
    model: EventFlowModel,
    context: ModelContext,
    decision: RecallTransition,
) -> ModelContext:
    return apply_transition_context(
        model,
        context,
        torch.zeros_like(context.modes),
        decision,
        ComposeTransition(
            context.support_mask,
            decision.modes,
            context.hypothesis_features,
            torch.zeros_like(context.modes),
            torch.zeros_like(context.modes, dtype=torch.bool),
        ),
    )


def apply_compose_context(
    model: EventFlowModel,
    context: ModelContext,
    decision: ComposeTransition,
) -> ModelContext:
    return apply_transition_context(
        model,
        context,
        torch.ones_like(context.modes),
        RecallTransition(context.active_slot_indices, context.modes),
        decision,
    )


def apply_action_context(model: EventFlowModel, context: ModelContext) -> ModelContext:
    modes = torch.full_like(context.modes, 4)
    return apply_transition_context(
        model,
        context,
        torch.full_like(modes, 2),
        RecallTransition(context.active_slot_indices, modes),
        ComposeTransition(
            context.support_mask,
            modes,
            context.hypothesis_features,
            torch.zeros_like(modes),
            torch.zeros_like(modes, dtype=torch.bool),
        ),
    )
