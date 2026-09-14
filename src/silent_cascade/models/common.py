"""Shared public-context feature assembly for neural components."""

import torch
from torch import nn

from silent_cascade.models.errors import NeuralError
from silent_cascade.models.types import ModelContext


def context_features(context: ModelContext, mode_embedding: nn.Embedding) -> torch.Tensor:
    """Assemble the canonical 570-wide public context without mutating state."""
    if not isinstance(context, ModelContext):
        raise TypeError("context must be a ModelContext")
    if not isinstance(mode_embedding, nn.Embedding):
        raise TypeError("mode_embedding must be an nn.Embedding")
    if mode_embedding.num_embeddings != 6 or mode_embedding.embedding_dim != 8:
        raise NeuralError("mode embedding must have shape (6, 8)")
    if (
        mode_embedding.weight.device != context.device
        or mode_embedding.weight.dtype is not torch.float32
    ):
        raise NeuralError("mode embedding and context must use float32 on one device")

    support_weights = context.support_mask.unsqueeze(-1).to(context.memory_embeddings.dtype)
    support_counts = support_weights.sum(dim=1).clamp_min(1.0)
    support_summary = (context.memory_embeddings * support_weights).sum(dim=1) / support_counts
    return torch.cat(
        (
            context.workspace.latent,
            mode_embedding(context.modes),
            support_summary,
            context.time_features,
            context.hypothesis_features,
        ),
        dim=1,
    )
