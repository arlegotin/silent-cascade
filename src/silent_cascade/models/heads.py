"""Prediction-only composition and action heads over public model context."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional

from silent_cascade.models.common import context_features
from silent_cascade.models.config import NeuralModelConfig
from silent_cascade.models.errors import NeuralError
from silent_cascade.models.types import ModelContext


@dataclass(frozen=True, slots=True)
class ComposePredictions:
    """Neutral differentiable composition predictions in canonical label order."""

    role_logits: torch.Tensor
    next_focus_logits: torch.Tensor
    hazard_logits: torch.Tensor
    log_delay: torch.Tensor
    normalized_deadline: torch.Tensor
    status_logits: torch.Tensor
    confidence_logit: torch.Tensor
    append_support_logit: torch.Tensor
    continue_search_logit: torch.Tensor


@dataclass(frozen=True, slots=True)
class ActionPredictions:
    """Neutral action-class logits, including abstain, and public lead fraction."""

    class_logits: torch.Tensor
    lead_fraction: torch.Tensor


class _ContextHead(nn.Module):
    def __init__(
        self,
        config: NeuralModelConfig,
        *,
        input_dim: int,
        mode_embedding: nn.Embedding | None,
    ) -> None:
        super().__init__()
        if not isinstance(config, NeuralModelConfig):
            raise TypeError("config must be a NeuralModelConfig")
        self.config = config
        if mode_embedding is None:
            self._owned_mode_embedding = nn.Embedding(config.mode_count, config.mode_dim)
        else:
            if not isinstance(mode_embedding, nn.Embedding):
                raise TypeError("mode_embedding must be an nn.Embedding")
            if (
                mode_embedding.num_embeddings != config.mode_count
                or mode_embedding.embedding_dim != config.mode_dim
            ):
                raise NeuralError("mode embedding does not match the neural configuration")
            object.__setattr__(self, "_injected_mode_embedding", mode_embedding)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.SiLU(),
        )

    @property
    def mode_embedding(self) -> nn.Embedding:
        owned = self._modules.get("_owned_mode_embedding")
        if owned is not None:
            return owned  # type: ignore[return-value]
        return object.__getattribute__(self, "_injected_mode_embedding")

    def _features(self, context: ModelContext) -> torch.Tensor:
        if not isinstance(context, ModelContext):
            raise TypeError("context must be a ModelContext")
        if next(self.trunk.parameters()).device != context.device:
            raise NeuralError("prediction head and context must use one device")
        return context_features(context, self.mode_embedding)


class ComposeHeads(_ContextHead):
    """Predict record interpretation and learned composition choices."""

    def __init__(
        self,
        config: NeuralModelConfig,
        *,
        mode_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__(
            config,
            input_dim=config.composition_input_dim,
            mode_embedding=mode_embedding,
        )
        hidden = config.head_hidden_dim
        self.role = nn.Linear(hidden, 5)
        self.next_focus = nn.Linear(hidden, config.entity_count)
        self.hazard = nn.Linear(hidden, 4)
        self.delay = nn.Linear(hidden, 1)
        self.deadline = nn.Linear(hidden, 1)
        self.status = nn.Linear(hidden, 3)
        self.confidence = nn.Linear(hidden, 1)
        self.append_support = nn.Linear(hidden, 1)
        self.continue_search = nn.Linear(hidden, 1)

    def forward(self, context: ModelContext) -> ComposePredictions:
        features = self._features(context)
        safe_indices = context.active_slot_indices.clamp_min(0)
        rows = torch.arange(context.batch_size, device=context.device)
        active = context.memory_embeddings[rows, safe_indices]
        active = active * (context.active_slot_indices >= 0).unsqueeze(1)
        hidden = self.trunk(torch.cat((features, active), dim=1))
        return ComposePredictions(
            role_logits=self.role(hidden),
            next_focus_logits=self.next_focus(hidden),
            hazard_logits=self.hazard(hidden),
            log_delay=functional.softplus(self.delay(hidden)).squeeze(1),
            normalized_deadline=self.deadline(hidden).squeeze(1),
            status_logits=self.status(hidden),
            confidence_logit=self.confidence(hidden).squeeze(1),
            append_support_logit=self.append_support(hidden).squeeze(1),
            continue_search_logit=self.continue_search(hidden).squeeze(1),
        )


class ActionHeads(_ContextHead):
    """Predict four action classes plus abstention and an action lead fraction."""

    def __init__(
        self,
        config: NeuralModelConfig,
        *,
        mode_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__(
            config,
            input_dim=config.action_input_dim,
            mode_embedding=mode_embedding,
        )
        self.classifier = nn.Linear(config.head_hidden_dim, 5)
        self.lead = nn.Linear(config.head_hidden_dim, 1)

    def forward(self, context: ModelContext) -> ActionPredictions:
        hidden = self.trunk(self._features(context))
        return ActionPredictions(
            class_logits=self.classifier(hidden),
            lead_fraction=torch.sigmoid(self.lead(hidden)).squeeze(1),
        )
