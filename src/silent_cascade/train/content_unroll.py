"""Training-private content decisions in the public evaluator's immediate contexts."""

from dataclasses import dataclass, fields

import torch
from torch.nn import functional as F

from silent_cascade.eval.compute import (
    NeuralComputeMeter,
    NeuralComputeSnapshot,
    _record_functional_operations,
)
from silent_cascade.memory.retrieval import RetrievalScores
from silent_cascade.models.content_types import ContentLossInputs
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.models.heads import ComposePredictions
from silent_cascade.models.types import ModelContext, TensorWorkspace
from silent_cascade.train.batches import TeacherTargets, TrainingBatch
from silent_cascade.train.observations import Boundary, observe_public


@dataclass(frozen=True, slots=True)
class ContentOperation:
    prediction_context: ModelContext
    post_context: ModelContext
    row_mask: torch.Tensor
    kind: torch.Tensor
    raw_recall_target: torch.Tensor
    retrieval: RetrievalScores
    composition: ComposePredictions


@dataclass(frozen=True, slots=True)
class ContentUnrollResult:
    """Forward-only counted diagnostics; an optimizer must meter backward itself."""

    observations: tuple[Boundary, ...]
    steps: tuple[ContentOperation, ...]
    final_context: ModelContext
    targets: TeacherTargets
    compute: NeuralComputeSnapshot
    diagnostic_label: str = "teacher-forced-untimed-content"

    def loss_inputs(self) -> ContentLossInputs:
        boundary = torch.stack([step.row_mask for step in self.steps], 1)
        values = {
            "boundary_mask": boundary,
            "recall_mask": boundary & (self.targets.kind == 0),
            "raw_recall_target": torch.stack([step.raw_recall_target for step in self.steps], 1),
            "retrieval_logits": torch.stack(
                [step.retrieval.masked_logits for step in self.steps], 1
            ),
            "retrieval_eligible_mask": torch.stack(
                [step.retrieval.eligible_mask for step in self.steps], 1
            ),
            "retrieval_target": self.targets.selected_slot,
        }
        for item in fields(ComposePredictions):
            if item.name != "normalized_deadline":
                values[item.name] = torch.stack(
                    [getattr(step.composition, item.name) for step in self.steps], 1
                )
        for name in (
            "role",
            "status",
            "confidence",
            "append_support",
            "continue_search",
            "focus",
            "hazard",
            "log_delay",
        ):
            source = "hazard_type" if name == "hazard" else name
            values[name + "_target"] = getattr(self.targets, source)
            values[name + "_mask"] = self.targets.validity[source] & boundary
        return ContentLossInputs(**values)


def _scatter(predictions, rows, size):
    return type(predictions)(
        **{
            item.name: getattr(predictions, item.name)
            .new_zeros((size, *getattr(predictions, item.name).shape[1:]))
            .index_copy(0, rows, getattr(predictions, item.name))
            for item in fields(predictions)
        }
    )


def _compose_teacher(model, current, batch, rows, col):
    """Apply already-predicted content; no teacher world clock or normalized deadline."""
    target = batch.targets
    status = target.status[rows, col]
    hazard_mask = target.validity["hazard_type"][rows, col]
    hypothesis = torch.cat(
        (
            (status != 2).float()[:, None],
            F.one_hot(target.hazard_type[rows, col].clamp_min(0), 4).float() * hazard_mask[:, None],
            (status == 1).float()[:, None],
            (target.confidence[rows, col] * (status != 2))[:, None],
            # At activation, signed log1p(relative deadline) is log1p(delay).
            # This terminal register enters the final jump only, never a prediction.
            (target.log_delay[rows, col] * hazard_mask)[:, None],
        ),
        1,
    )
    current = current._updated(
        support_mask=batch.support_after[rows, col],
        modes=batch.post_modes[rows, col],
        hypothesis_features=torch.where(
            target.validity["status"][rows, col, None], hypothesis, current.hypothesis_features
        ),
    )
    workspace = model.jump(current, torch.ones_like(rows))
    focus_rows = target.validity["focus"][rows, col].nonzero(as_tuple=True)[0]
    if focus_rows.numel():
        focus = torch.tanh(
            model.focus_projection(
                model.record_encoder.encode_entities(target.focus[rows[focus_rows], col])
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
    flat = torch.arange(rows.numel(), device=current.device) * 64 + current.active_slot_indices
    eligible = current.eligibility.flatten().index_fill(0, flat, False).reshape(-1, 64)
    return current._updated(
        workspace=workspace, eligibility=eligible, active_slot_indices=torch.full_like(rows, -1)
    )


def teacher_forced_content_unroll(
    model: EventFlowModel, batch: TrainingBatch
) -> ContentUnrollResult:
    """Fresh public observation graph, then teacher content with no internal flow.

    Preserve the target T layout for consumers: ACT/padding columns are inert
    storage with false row masks, not events or prediction-module calls.
    """
    if not isinstance(model, EventFlowModel) or not isinstance(batch, TrainingBatch):
        raise TypeError("content unroll requires EventFlowModel and TrainingBatch")
    with NeuralComputeMeter(model) as meter:
        context, _, _, observations, *_ = observe_public(model, batch.public_inputs())
        size, count = batch.targets.kind.shape
        device = context.device
        steps = []
        for col in range(count):
            kind = batch.targets.kind[:, col]
            mask = batch.targets.real_step_mask[:, col] & ((kind == 0) | (kind == 1))
            recall_rows = (mask & (kind == 0)).nonzero(as_tuple=True)[0]
            compose_rows = (mask & (kind == 1)).nonzero(as_tuple=True)[0]
            prediction_context = context
            # Empty slices still have a path back to the fresh observation graph.
            anchor = context.workspace.latent.sum(dim=1) * 0.0
            raw = anchor
            retrieval = RetrievalScores(
                anchor[:, None].expand(-1, 64),
                anchor[:, None].expand(-1, 64),
                torch.zeros(size, 64, dtype=torch.bool, device=device),
            )
            composition = ComposePredictions(
                *[
                    anchor[:, None].expand(-1, width) if width else anchor
                    for width in (5, 64, 4, None, None, 3, None, None, None)
                ]
            )
            if recall_rows.numel():
                current = prediction_context._gather(recall_rows)
                _, parameters = model.preview_and_control(current)
                raw = raw.index_copy(0, recall_rows, parameters.raw_guard_targets[:, 0])
                retrieval = _scatter(model.recall_scores(current), recall_rows, size)
                # First disclosure of the teacher-selected record follows scoring.
                current = current._updated(
                    active_slot_indices=batch.targets.selected_slot[recall_rows, col],
                    modes=torch.full_like(recall_rows, 2),
                )
                current = current._updated(
                    workspace=model.jump(current, torch.zeros_like(recall_rows))
                )
                context = context._scatter(recall_rows, current)
            if compose_rows.numel():
                current = prediction_context._gather(compose_rows)
                composition = _scatter(model.compose(current), compose_rows, size)
                context = context._scatter(
                    compose_rows, _compose_teacher(model, current, batch, compose_rows, col)
                )
            steps.append(
                ContentOperation(
                    prediction_context, context, mask, kind, raw, retrieval, composition
                )
            )
    return ContentUnrollResult(observations, tuple(steps), context, batch.targets, meter.snapshot())
