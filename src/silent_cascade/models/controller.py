"""Learned bounded flow and endogenous-guard controller."""

import torch
from torch import nn
from torch.nn import functional as functional

from silent_cascade.eventflow.guards import allowed_mode_mask
from silent_cascade.memory.retrieval import RetrievalPreview
from silent_cascade.models.common import context_features
from silent_cascade.models.config import NeuralModelConfig
from silent_cascade.models.dynamics import BatchedSegmentParameters
from silent_cascade.models.errors import NeuralError
from silent_cascade.models.types import ModelContext
from silent_cascade.schemas import Mode


class FlowGuardController(nn.Module):
    """Predict one fixed, bounded analytic segment from current public context."""

    def __init__(
        self,
        config: NeuralModelConfig,
        *,
        mode_embedding: nn.Embedding | None = None,
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
        hidden = config.controller_hidden_dim
        self.network = nn.Sequential(
            nn.Linear(config.controller_input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.controller_output_dim),
        )
        masks = torch.tensor(
            [allowed_mode_mask(mode) for mode in Mode],
            dtype=torch.bool,
        )
        self.register_buffer("_allowed_guard_masks", masks, persistent=False)

    @property
    def mode_embedding(self) -> nn.Embedding:
        owned = self._modules.get("_owned_mode_embedding")
        if owned is not None:
            return owned  # type: ignore[return-value]
        return object.__getattribute__(self, "_injected_mode_embedding")

    def forward(self, context: ModelContext, preview: RetrievalPreview) -> BatchedSegmentParameters:
        if not isinstance(context, ModelContext):
            raise TypeError("context must be a ModelContext")
        if not isinstance(preview, RetrievalPreview):
            raise TypeError("preview must be a RetrievalPreview")
        if (
            not isinstance(preview.features, torch.Tensor)
            or preview.features.shape != (context.batch_size, self.config.preview_dim)
            or preview.features.dtype is not torch.float32
            or preview.features.device != context.device
            or not isinstance(preview.has_candidate, torch.Tensor)
            or preview.has_candidate.shape != (context.batch_size,)
            or preview.has_candidate.dtype is not torch.bool
            or preview.has_candidate.device != context.device
        ):
            raise NeuralError("preview must match the context batch, dtype, and device")
        if not bool(torch.isfinite(preview.features).all()):
            raise NeuralError("preview features must be finite")
        if next(self.network.parameters()).device != context.device:
            raise NeuralError("controller and context must use one device")
        raw = self.network(
            torch.cat((context_features(context, self.mode_embedding), preview.features), dim=1)
        )
        latent_targets, latent_rates, guard_logits, guard_rate_logits = raw.split(
            (self.config.latent_dim, self.config.latent_dim, 3, 3), dim=1
        )
        flow_targets = torch.tanh(latent_targets)
        flow_rates = torch.clamp(functional.softplus(latent_rates) + 1e-5, 1e-5, 20.0)
        epsilon = torch.finfo(torch.float32).eps
        raw_guard_targets = torch.clamp(2.0 * torch.sigmoid(guard_logits), epsilon, 2.0 - epsilon)
        guard_rates = torch.clamp(functional.softplus(guard_rate_logits) + 1e-5, 1e-5, 500.0)
        allowed = self._allowed_guard_masks[context.modes]
        guard_targets = torch.where(allowed, raw_guard_targets, raw_guard_targets.new_tensor(0.5))
        return BatchedSegmentParameters(
            flow_targets=flow_targets,
            flow_rates=flow_rates,
            raw_guard_targets=raw_guard_targets,
            guard_targets=guard_targets,
            guard_rates=guard_rates,
        )
