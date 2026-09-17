"""One versioned training objective shared by optimization and native verification."""

from dataclasses import dataclass
from typing import Protocol

import torch

from silent_cascade.models.config import LossWeights
from silent_cascade.models.content_loss import content_loss
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.models.losses import LossBreakdown, event_flow_loss
from silent_cascade.train.batches import TrainingBatch
from silent_cascade.train.content_unroll import ContentUnrollResult, teacher_forced_content_unroll
from silent_cascade.train.unroll import UnrollResult, teacher_forced_unroll


class ObjectiveRecipe(Protocol):
    @property
    def loss_weights(self) -> LossWeights: ...

    @property
    def content_auxiliary_weight(self) -> float: ...

    @property
    def objective_version(self) -> str: ...


@dataclass(frozen=True, slots=True)
class TrainingObjective:
    total: torch.Tensor
    timed: UnrollResult
    timed_loss: LossBreakdown
    content: ContentUnrollResult | None
    content_loss: LossBreakdown | None
    objective_version: str
    auxiliary_coefficient: float


def training_objective(
    model: EventFlowModel, batch: TrainingBatch, training: ObjectiveRecipe
) -> TrainingObjective:
    timed = teacher_forced_unroll(model, batch)
    timed_loss = event_flow_loss(timed.loss_inputs(), training.loss_weights)
    content = auxiliary = None
    total = timed_loss.total
    if training.content_auxiliary_weight:
        content = teacher_forced_content_unroll(model, batch)
        auxiliary = content_loss(content.loss_inputs())
        total = total + training.content_auxiliary_weight * auxiliary.total
    return TrainingObjective(
        total,
        timed,
        timed_loss,
        content,
        auxiliary,
        training.objective_version,
        training.content_auxiliary_weight,
    )
