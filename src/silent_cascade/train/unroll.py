"""Training-private, predict-before-teacher differentiable event traces."""

from dataclasses import dataclass, fields

import torch
from torch.nn import functional as F

from silent_cascade.eval.compute import (
    NeuralComputeMeter,
    NeuralComputeSnapshot,
    _record_functional_operations,
)
from silent_cascade.eventflow.guards import allowed_mode_mask
from silent_cascade.memory.retrieval import RetrievalPreview, RetrievalScores
from silent_cascade.models.dynamics import (
    BatchedSegmentParameters,
    crossings_batch,
    flow_batch,
)
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.models.heads import ActionPredictions, ComposePredictions
from silent_cascade.models.types import LossInputs, ModelContext, TensorWorkspace
from silent_cascade.schemas import Mode
from silent_cascade.train.batches import TeacherTargets, TrainingBatch
from silent_cascade.train.observations import Boundary, _parameters, observe_public


@dataclass(frozen=True, slots=True)
class Operation:
    prediction_context: ModelContext
    post_context: ModelContext
    post_parameters: BatchedSegmentParameters
    crossings: torch.Tensor
    legal_guards: torch.Tensor
    retrieval: RetrievalScores
    composition: ComposePredictions
    action: ActionPredictions


@dataclass(frozen=True, slots=True)
class UnrollResult:
    """Teacher-forced diagnostics, never validation accuracy or model inputs.

    Compute covers this forward unroll only. Callers measuring optimization
    must wrap their forward plus backward in an outer NeuralComputeMeter.
    """

    observations: tuple[Boundary, ...]
    boundaries: tuple[Boundary, ...]
    steps: tuple[Operation, ...]
    final_context: ModelContext
    targets: TeacherTargets
    pre_jump_latent: torch.Tensor
    post_jump_latent: torch.Tensor
    jump_mask: torch.Tensor
    compute: NeuralComputeSnapshot
    diagnostic_label: str = "teacher-forced"

    def loss_inputs(self) -> LossInputs:
        """Project tensor references and graph-preserving stacks, without a reducer."""
        targets = self.targets
        values = {
            "raw_guard_targets": torch.stack([b.raw_guard_targets for b in self.boundaries], 1),
            "crossing_offsets": torch.stack([s.crossings for s in self.steps], 1),
            "legal_guard_mask": torch.stack([s.legal_guards for s in self.steps], 1),
            "boundary_mask": targets.real_step_mask,
            "kind_target": targets.kind,
            "delta_target": targets.delta,
            "final_guard_targets": torch.stack(
                [s.post_parameters.raw_guard_targets for s in self.steps], 1
            ),
            "final_dormancy_mask": targets.final_dormancy_mask,
            "retrieval_logits": torch.stack([s.retrieval.masked_logits for s in self.steps], 1),
            "retrieval_eligible_mask": torch.stack(
                [s.retrieval.eligible_mask for s in self.steps], 1
            ),
            "action_logits": torch.stack([s.action.class_logits for s in self.steps], 1),
            "action_mask": targets.kind == 2,
            "abstention_mask": targets.action_class == 4,
            "lead_fraction": torch.stack([s.action.lead_fraction for s in self.steps], 1),
            "pre_jump_latent": self.pre_jump_latent,
            "post_jump_latent": self.post_jump_latent,
            "jump_mask": self.jump_mask,
        }
        for item in fields(ComposePredictions):
            values[item.name] = torch.stack(
                [getattr(s.composition, item.name) for s in self.steps], 1
            )
        aliases = {
            "retrieval": "selected_slot",
            "hazard": "hazard_type",
            "action": "action_class",
            "lead": "action_lead",
        }
        for name in (
            "retrieval",
            "role",
            "status",
            "confidence",
            "append_support",
            "continue_search",
            "focus",
            "hazard",
            "log_delay",
            "normalized_deadline",
            "action",
            "lead",
        ):
            source = aliases.get(name, name)
            values[name + "_target"] = getattr(targets, source)
            if name != "action":
                values[name + "_mask"] = targets.validity[source]
        return LossInputs(**values)


def _scatter_predictions(predictions, rows, size):
    return type(predictions)(
        **{
            item.name: getattr(predictions, item.name)
            .new_zeros((size, *getattr(predictions, item.name).shape[1:]))
            .index_copy(0, rows, getattr(predictions, item.name))
            for item in fields(predictions)
        }
    )


def _hypothesis_time(features, deadline, elapsed):
    difference = deadline - elapsed
    relative = difference.sign() * torch.log1p(difference.abs())
    return torch.cat(
        (
            features[:, :7],
            (relative * features[:, 1:5].sum(1))[:, None],
        ),
        1,
    )


def _time_context(context, times, initial, activation, deadline):
    # Absolute clocks stay host float64; only derived public features enter models.
    features = torch.stack(
        (torch.log1p(times - initial), torch.log1p((times - activation).clamp_min(0))), 1
    )
    elapsed = (times - activation).to(device=context.device, dtype=torch.float32)
    return context._updated(
        time_features=features.to(context.device, dtype=torch.float32),
        hypothesis_features=_hypothesis_time(context.hypothesis_features, deadline, elapsed),
    )


def _apply_teacher(model, context, batch, col, rows, deadline, activation):
    """Disclosure boundary: called only after every supervised prediction."""
    target = batch.targets
    kind = target.kind[rows, col]
    recall = kind == 0
    selected = target.selected_slot[rows, col]
    active = torch.where(recall, selected, context.active_slot_indices)
    hypothesis = context.hypothesis_features
    status_mask = target.validity["status"][rows, col]
    status = target.status[rows, col]
    hazard_mask = target.validity["hazard_type"][rows, col]
    hazard = F.one_hot(target.hazard_type[rows, col].clamp_min(0), 4).to(torch.float32)
    new_hypothesis = torch.cat(
        (
            (status != 2).to(torch.float32)[:, None],
            hazard * hazard_mask[:, None],
            (status == 1).to(torch.float32)[:, None],
            target.confidence[rows, col, None] * (status != 2)[:, None],
            torch.zeros_like(status[:, None], dtype=torch.float32),
        ),
        1,
    )
    hypothesis = torch.where(status_mask[:, None], new_hypothesis, hypothesis)
    # Keep the register relative to activation: no absolute float32 clocks.
    # Use the normalized teacher deadline only AFTER its composition prediction.
    elapsed = (batch.teacher_times[rows.cpu(), col] - activation[rows.cpu()]).to(
        context.device, dtype=torch.float32
    )
    proposed = target.normalized_deadline[rows, col] * (1 + elapsed)
    deadline = deadline.index_copy(0, rows, torch.where(hazard_mask, proposed, deadline[rows]))
    updated = context._updated(
        active_slot_indices=active,
        support_mask=batch.support_after[rows, col],
        modes=batch.post_modes[rows, col],
        hypothesis_features=_hypothesis_time(hypothesis, deadline[rows], elapsed),
    )
    workspace = model.jump(updated, kind)
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
    return updated._updated(
        workspace=workspace, active_slot_indices=torch.where(kind == 1, -1, active)
    ), deadline


def teacher_forced_unroll(model: EventFlowModel, batch: TrainingBatch) -> UnrollResult:
    """Execute actual public/teacher positions with fresh, fully recurrent graphs."""
    if not isinstance(model, EventFlowModel) or not isinstance(batch, TrainingBatch):
        raise TypeError("unroll requires EventFlowModel and TrainingBatch")
    with NeuralComputeMeter(model) as meter:
        result = _unroll(model, batch)
    return UnrollResult(**result, compute=meter.snapshot())


def _unroll(model, batch):
    public = batch.public_inputs()
    size, count = batch.targets.kind.shape
    device = public.records.device
    context, parameters, preview, observations, before, after, masks = observe_public(model, public)
    initial = torch.tensor(public.initial_times, dtype=torch.float64)
    activation = torch.tensor(
        [
            public.observation_times[row, public.observation_kind[row].cpu() == 1].item()
            for row in range(size)
        ],
        dtype=torch.float64,
    )
    times = activation.clone()
    deadline = torch.zeros(size, device=device)
    consumed = torch.zeros_like(context.eligibility)
    refractory = torch.zeros(size, 64, dtype=torch.float64)
    boundaries, steps = [], []

    legal_table = torch.tensor([allowed_mode_mask(mode) for mode in Mode], device=device)
    for col in range(count):
        mask = batch.targets.real_step_mask[:, col]
        rows = mask.nonzero(as_tuple=True)[0]
        boundaries.append(Boundary(context, parameters, preview, mask))
        current = context._gather(rows)
        fixed = _parameters(parameters, rows)
        crossing = crossings_batch(
            current.workspace.accumulators, fixed.guard_targets, fixed.guard_rates
        )
        crossings = torch.full((size, 3), torch.inf, device=device).index_copy(0, rows, crossing)
        legal = legal_table[context.modes] & mask[:, None]
        current = current._updated(
            workspace=flow_batch(current.workspace, fixed, batch.targets.delta[rows, col])
        )
        times = times.index_copy(0, rows.cpu(), batch.teacher_times[rows.cpu(), col])
        eligibility = (
            public.records.valid_mask & ~consumed & (refractory <= times[:, None]).to(device)
        )
        current = _time_context(
            current, times[rows.cpu()], initial[rows.cpu()], activation[rows.cpu()], deadline[rows]
        )
        current = current._updated(eligibility=eligibility[rows])
        prediction_context = context._scatter(rows, current)
        # Kind dispatch is the first teacher disclosure; it selects a head only.
        kinds = batch.targets.kind[:, col]
        retrieval_rows = (mask & (kinds == 0)).nonzero(as_tuple=True)[0]
        compose_rows = (mask & (kinds == 1)).nonzero(as_tuple=True)[0]
        action_rows = (mask & (kinds == 2)).nonzero(as_tuple=True)[0]
        retrieval = RetrievalScores(
            torch.zeros(size, 64, device=device),
            torch.full((size, 64), -torch.inf, device=device),
            torch.zeros(size, 64, dtype=torch.bool, device=device),
        )
        if retrieval_rows.numel():
            scored = model.recall_scores(prediction_context._gather(retrieval_rows))
            retrieval = RetrievalScores(
                retrieval.raw_scores.index_copy(0, retrieval_rows, scored.raw_scores),
                retrieval.masked_logits.index_copy(0, retrieval_rows, scored.masked_logits),
                retrieval.eligible_mask.index_copy(0, retrieval_rows, scored.eligible_mask),
            )
        composition = ComposePredictions(
            *[
                torch.zeros((size, *tail), device=device)
                for tail in ((5,), (64,), (4,), (), (), (3,), (), (), ())
            ]
        )
        if compose_rows.numel():
            composition = _scatter_predictions(
                model.compose(prediction_context._gather(compose_rows)), compose_rows, size
            )
        action = ActionPredictions(
            torch.zeros(size, 5, device=device), torch.zeros(size, device=device)
        )
        if action_rows.numel():
            action = _scatter_predictions(
                model.action(prediction_context._gather(action_rows)), action_rows, size
            )

        post, deadline = _apply_teacher(model, current, batch, col, rows, deadline, activation)
        if retrieval_rows.numel():
            slots = batch.targets.selected_slot[retrieval_rows, col]
            flat = retrieval_rows.cpu() * 64 + slots.cpu()
            refractory = (
                refractory.flatten()
                .index_copy(0, flat, times[retrieval_rows.cpu()] + 1e-3)
                .reshape(size, 64)
            )
        if compose_rows.numel():
            slots = prediction_context.active_slot_indices[compose_rows]
            flat = compose_rows * 64 + slots
            consumed = consumed.flatten().index_fill(0, flat, True).reshape(size, 64)
        eligibility = (
            public.records.valid_mask & ~consumed & (refractory <= times[:, None]).to(device)
        )
        post = post._updated(eligibility=eligibility[rows])
        post = _time_context(
            post, times[rows.cpu()], initial[rows.cpu()], activation[rows.cpu()], deadline[rows]
        )
        active_preview, active_parameters = model.preview_and_control(post)
        context = context._scatter(rows, post)
        parameters = _parameters(parameters, rows, active_parameters)
        preview = RetrievalPreview(
            preview.features.index_copy(0, rows, active_preview.features),
            preview.has_candidate.index_copy(0, rows, active_preview.has_candidate),
        )
        negative_rows = (batch.targets.action_class[:, col] == 4).nonzero(as_tuple=True)[0]
        if negative_rows.numel():
            negative = model.action(context._gather(negative_rows))
            action = ActionPredictions(
                action.class_logits.index_copy(0, negative_rows, negative.class_logits),
                action.lead_fraction.index_copy(0, negative_rows, negative.lead_fraction),
            )
        steps.append(
            Operation(
                prediction_context,
                context,
                parameters,
                crossings,
                legal,
                retrieval,
                composition,
                action,
            )
        )
        before.append(prediction_context.workspace.latent)
        after.append(context.workspace.latent)
        masks.append(mask)
    return dict(
        observations=tuple(observations),
        boundaries=tuple(boundaries),
        steps=tuple(steps),
        final_context=context,
        targets=batch.targets,
        pre_jump_latent=torch.stack(before, 1),
        post_jump_latent=torch.stack(after, 1),
        jump_mask=torch.stack(masks, 1),
    )
