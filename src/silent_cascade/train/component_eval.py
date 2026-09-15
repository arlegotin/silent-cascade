"""Public-only bounded content predictions, followed by private-label scoring.

This is unassisted content evaluation, not the EventFlow world-time clock.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from silent_cascade.eval.compute import NeuralComputeMeter, NeuralComputeSnapshot
from silent_cascade.memory.retrieval import select_record_ids
from silent_cascade.models.errors import NeuralError
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.models.types import ModelContext, PublicInputBatch, TensorWorkspace
from silent_cascade.train.observations import observe_public

if TYPE_CHECKING:
    from silent_cascade.train.curriculum_data import ComponentCorpus, ComponentTarget


@dataclass(frozen=True, slots=True)
class ContentDecision:
    record_id: int
    role: int
    focus: int
    hazard_type: int
    log_delay: float
    normalized_deadline: float
    status: int
    confidence: float
    append_support: bool
    continue_search: bool


@dataclass(frozen=True, slots=True)
class ComponentPrediction:
    record_ids: tuple[int, ...] = ()
    content: tuple[ContentDecision, ...] = ()
    stop_reason: str = "capped"
    atomic_transitions: int = 0
    terminal_class: int = 4
    terminal_status: int = 2
    action_class: int = 4
    action_lead_fraction: float = 0.0
    action_logits: tuple[float, ...] = ()
    recall_guard_margins: tuple[float, ...] = ()
    retrieval_scores: tuple[tuple[tuple[int, float], ...], ...] = ()
    observation_events: int = 0
    scorer_calls: int = 0
    records_scored: int = 0
    foundation_model_calls: int = 0

    @property
    def failed(self) -> bool:
        return self.stop_reason in {"capped", "malformed"}


def _primitive(value):
    if isinstance(value, dict):
        return {key: _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ComponentPredictions:
    rows: tuple[ComponentPrediction, ...]
    evaluation_mode: str = "unassisted_content"

    def to_primitive(self):
        return {
            "evaluation_mode": self.evaluation_mode,
            "timed": False,
            "rows": [
                {
                    **_primitive(asdict(row)),
                    "failed": row.failed,
                    "capped": row.stop_reason == "capped",
                    "stopped": row.stop_reason != "capped",
                }
                for row in self.rows
            ],
        }


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise NeuralError(f"Nonfinite internal prediction {name}")


def _check_context(context):
    _finite(context.workspace.latent, "latent")
    _finite(context.workspace.accumulators, "accumulators")
    for name in ("memory_embeddings", "time_features", "hypothesis_features"):
        _finite(getattr(context, name), name)
    # Validate trusted functional updates at this external evaluation boundary.
    ModelContext(**{item.name: getattr(context, item.name) for item in fields(context)})


def predict_components(
    model: EventFlowModel, public_inputs: PublicInputBatch
) -> ComponentPredictions:
    """Four atomic transitions maximum; no targets or private imports on this path."""
    if not isinstance(model, EventFlowModel) or not isinstance(public_inputs, PublicInputBatch):
        raise TypeError("prediction requires EventFlowModel and PublicInputBatch")
    with torch.no_grad():
        context, _, _, observations, *_ = observe_public(model, public_inputs)
        for boundary in observations:
            _check_context(boundary.context)
            for item in fields(boundary.parameters):
                _finite(getattr(boundary.parameters, item.name), item.name)
        size, device = context.batch_size, context.device
        records = [[] for _ in range(size)]
        contents = [[] for _ in range(size)]
        margins = [[] for _ in range(size)]
        scores_log = [[] for _ in range(size)]
        reasons = [None] * size
        classes, statuses = [4] * size, [2] * size
        transitions = [0] * size
        scorer_calls = [1] * size  # The actual activation preview in observe_public.
        for _ in range(2):
            rows = torch.tensor(
                [i for i, reason in enumerate(reasons) if reason is None], device=device
            )
            if not rows.numel():
                break
            current = context._gather(rows)
            preview, parameters = model.preview_and_control(current)
            for item in fields(parameters):
                _finite(getattr(parameters, item.name), item.name)
            proceed = []
            for local, row in enumerate(rows.tolist()):
                scorer_calls[row] += 1
                margin = float(parameters.guard_targets[local, 0] - 1)
                margins[row].append(margin)
                if margin <= 0 or not bool(preview.has_candidate[local]):
                    reasons[row] = "dormant"
                else:
                    proceed.append(row)
            if not proceed:
                continue
            rows = torch.tensor(proceed, device=device)
            current = context._gather(rows)
            scores = model.recall_scores(current)
            _finite(scores.raw_scores, "retrieval scores")
            ids = tuple(public_inputs.records.record_ids[row] for row in proceed)
            selected = select_record_ids(scores, ids)
            slots = []
            for local, row in enumerate(proceed):
                scorer_calls[row] += 1
                record = selected[local]
                if record is None:
                    raise NeuralError("Eligible learned recall produced no record")
                slots.append(ids[local].index(record))
                records[row].append(record)
                scores_log[row].append(
                    tuple(
                        (record_id, float(scores.raw_scores[local, slot]))
                        for slot, record_id in enumerate(ids[local])
                        if bool(scores.eligible_mask[local, slot]) and record_id is not None
                    )
                )
                transitions[row] += 1
            slot_tensor = torch.tensor(slots, device=device)
            current = current._updated(
                active_slot_indices=slot_tensor, modes=torch.full_like(rows, 2)
            )
            current = current._updated(workspace=model.jump(current, torch.zeros_like(rows)))
            _check_context(current)
            predicted = model.compose(current)
            for item in fields(predicted):
                _finite(getattr(predicted, item.name), item.name)
            role = predicted.role_logits.argmax(1)
            focus = predicted.next_focus_logits.argmax(1)
            hazard = predicted.hazard_logits.argmax(1)
            status = predicted.status_logits.argmax(1)
            support = predicted.append_support_logit >= 0
            continuing = predicted.continue_search_logit >= 0
            confidence = predicted.confidence_logit.sigmoid()
            support_mask = current.support_mask.clone()
            support_mask[torch.arange(len(proceed), device=device), slot_tensor] |= support
            modes = torch.where(role == 1, 3, torch.where((role == 0) & continuing, 1, 4))
            terminal = (role == 1) | (role == 2) | ((role == 0) & ~continuing)
            hypothesis = torch.cat(
                (
                    (status != 2).float()[:, None],
                    F.one_hot(hazard, 4).float() * (role == 1)[:, None],
                    (status == 1).float()[:, None],
                    confidence[:, None] * (status != 2)[:, None],
                    (
                        predicted.normalized_deadline.sign()
                        * torch.log1p(predicted.normalized_deadline.abs())
                        * (role == 1)
                    )[:, None],
                ),
                1,
            )
            current = current._updated(
                support_mask=support_mask,
                modes=modes,
                hypothesis_features=torch.where(
                    terminal[:, None], hypothesis, current.hypothesis_features
                ),
            )
            workspace = model.jump(current, torch.ones_like(rows))
            learned_focus = torch.tanh(
                model.focus_projection(model.record_encoder.encode_entities(focus))
            )
            latent = torch.cat(
                (
                    workspace.latent[:, :328],
                    torch.where((role == 0)[:, None], learned_focus, workspace.latent[:, 328:392]),
                    workspace.latent[:, 392:],
                ),
                1,
            )
            current = current._updated(
                workspace=TensorWorkspace._from_functional_update(latent, workspace.accumulators),
                active_slot_indices=torch.full_like(rows, -1),
            )
            eligible = current.eligibility.clone()
            eligible[torch.arange(len(proceed), device=device), slot_tensor] = False
            current = current._updated(eligibility=eligible)
            _check_context(current)
            context = context._scatter(rows, current)
            for local, row in enumerate(proceed):
                decision = ContentDecision(
                    records[row][-1],
                    int(role[local]),
                    int(focus[local]),
                    int(hazard[local]),
                    float(predicted.log_delay[local]),
                    float(predicted.normalized_deadline[local]),
                    int(status[local]),
                    float(confidence[local]),
                    bool(support[local]),
                    bool(continuing[local]),
                )
                contents[row].append(decision)
                transitions[row] += 1
                if decision.role in (1, 2):
                    reasons[row] = "hazard" if decision.role == 1 else "safe"
                    classes[row] = decision.hazard_type if decision.role == 1 else 4
                    statuses[row] = decision.status
                elif decision.role == 0 and not decision.continue_search:
                    reasons[row] = "explicit_null"
                    statuses[row] = decision.status
                elif decision.role in (3, 4):
                    reasons[row] = "malformed"
        action = model.action(context)
        _finite(action.class_logits, "action logits")
        _finite(action.lead_fraction, "action lead")
        return ComponentPredictions(
            tuple(
                ComponentPrediction(
                    tuple(records[i]),
                    tuple(contents[i]),
                    reasons[i] or "capped",
                    transitions[i],
                    classes[i],
                    statuses[i],
                    int(action.class_logits[i].argmax()),
                    float(action.lead_fraction[i]),
                    tuple(action.class_logits[i].tolist()),
                    tuple(margins[i]),
                    tuple(scores_log[i]),
                    int(public_inputs.observation_mask[i].sum()),
                    scorer_calls[i],
                    scorer_calls[i] * 64,
                )
                for i in range(size)
            )
        )


@dataclass(frozen=True, slots=True)
class ScoredComponent:
    recall_correct: tuple[bool, ...]
    composition_correct: tuple[bool, ...]
    chain_correct: bool
    action_correct: bool
    category: str


@dataclass(frozen=True, slots=True)
class ComponentMetrics:
    episode_count: int
    required_recall_count: int
    required_composition_count: int
    correct_recall_count: int
    correct_composition_count: int
    correct_chain_count: int
    action_correct_count: int
    categories: tuple[tuple[str, int, int], ...]
    log_delay_absolute_errors: tuple[float, ...]
    normalized_deadline_absolute_errors: tuple[float, ...]
    rows: tuple[ScoredComponent, ...]

    @property
    def required_recall_accuracy(self):
        return (
            self.correct_recall_count / self.required_recall_count
            if self.required_recall_count
            else 0.0
        )

    @property
    def required_composition_accuracy(self):
        return (
            self.correct_composition_count / self.required_composition_count
            if self.required_composition_count
            else 0.0
        )

    @property
    def complete_chain_accuracy(self):
        return self.correct_chain_count / self.episode_count if self.episode_count else 0.0

    @property
    def gates_pass(self):
        return all(
            value > 0.99
            for value in (
                self.required_recall_accuracy,
                self.required_composition_accuracy,
                self.complete_chain_accuracy,
            )
        )

    def to_primitive(self):
        return {
            **_primitive(asdict(self)),
            "required_recall_accuracy": self.required_recall_accuracy,
            "required_composition_accuracy": self.required_composition_accuracy,
            "complete_chain_accuracy": self.complete_chain_accuracy,
            "gates_pass": self.gates_pass,
            "category_metrics": {
                name: {
                    "count": count,
                    "correct": correct,
                    "accuracy": correct / count if count else 0.0,
                }
                for name, count, correct in self.categories
            },
            "action_accuracy": self.action_correct_count / self.episode_count
            if self.episode_count
            else 0.0,
        }


def score_components(
    predictions: ComponentPredictions, targets: tuple[ComponentTarget, ...]
) -> ComponentMetrics:
    if len(predictions.rows) != len(targets):
        raise ValueError("Prediction and target row counts differ")
    rows, delay, deadline = [], [], []
    for prediction, target in zip(predictions.rows, targets, strict=True):
        recall = tuple(
            i < len(prediction.record_ids) and prediction.record_ids[i] == record
            for i, record in enumerate(target.record_ids)
        )
        compositions = []
        for i, expected in enumerate(target.content):
            if i >= len(prediction.content):
                compositions.append(False)
                continue
            actual = prediction.content[i]
            final = i == len(target.content) - 1
            continuation = (
                actual.continue_search == expected.continue_search
                if expected.continue_search is not None
                else True
            )
            if final and target.terminal_status == 2 and actual.role == 0:
                continuation = prediction.stop_reason in {"explicit_null", "dormant"} and len(
                    prediction.content
                ) == len(target.content)
            correct = (
                actual.record_id == expected.record_id
                and actual.role == expected.role
                and actual.append_support == expected.append_support
                and continuation
                and (expected.focus is None or actual.focus == expected.focus)
                and (expected.hazard_type is None or actual.hazard_type == expected.hazard_type)
                and (
                    expected.status is None
                    or actual.status == expected.status
                    or (
                        final
                        and target.terminal_status == 2
                        and prediction.stop_reason == "dormant"
                    )
                )
            )
            compositions.append(correct)
            if expected.log_delay is not None:
                delay.append(abs(actual.log_delay - expected.log_delay))
            if expected.normalized_deadline is not None:
                deadline.append(abs(actual.normalized_deadline - expected.normalized_deadline))
        chain = (
            all(recall)
            and all(compositions)
            and len(prediction.record_ids) == len(target.record_ids)
            and len(prediction.content) == len(target.content)
            and not prediction.failed
            and prediction.terminal_class == target.terminal_class
            and prediction.terminal_status == target.terminal_status
        )
        category = ("positive", "safe", "disconnected")[target.terminal_status]
        rows.append(
            ScoredComponent(
                recall,
                tuple(compositions),
                chain,
                prediction.action_class == target.terminal_class,
                category,
            )
        )
    return ComponentMetrics(
        len(rows),
        sum(len(t.record_ids) for t in targets),
        sum(len(t.content) for t in targets),
        sum(sum(r.recall_correct) for r in rows),
        sum(sum(r.composition_correct) for r in rows),
        sum(r.chain_correct for r in rows),
        sum(r.action_correct for r in rows),
        tuple(
            (
                name,
                sum(r.category == name for r in rows),
                sum(r.category == name and r.chain_correct for r in rows),
            )
            for name in ("positive", "safe", "disconnected")
        ),
        tuple(delay),
        tuple(deadline),
        tuple(rows),
    )


@dataclass(frozen=True, slots=True)
class ComponentEvaluation:
    predictions: ComponentPredictions
    metrics: ComponentMetrics
    compute: tuple[NeuralComputeSnapshot, ...]


def evaluate_components(
    model: EventFlowModel, corpus: ComponentCorpus, *, batch_size: int
) -> ComponentEvaluation:
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("Evaluation batch size must be 1 to 128")
    training = model.training
    rows, compute = [], []
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(corpus.public_examples), batch_size):
                public = corpus.public_batch(
                    start, min(batch_size, len(corpus.public_examples) - start)
                )
                public = public.to(str(next(model.parameters()).device))
                with NeuralComputeMeter(model) as meter:
                    rows.extend(predict_components(model, public).rows)
                compute.append(meter.snapshot())
    finally:
        model.train(training)
    predictions = ComponentPredictions(tuple(rows))
    return ComponentEvaluation(
        predictions, score_components(predictions, corpus.targets), tuple(compute)
    )
