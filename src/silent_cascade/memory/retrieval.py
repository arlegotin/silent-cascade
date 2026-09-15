"""Learned memory scoring, safe previews, and host-side selection decoding."""

import math
from dataclasses import dataclass

import torch
from torch import nn

from silent_cascade.models.config import NeuralModelConfig
from silent_cascade.models.errors import NeuralError
from silent_cascade.models.types import ModelContext


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    """Per-slot learned scores and their schema-legality mask."""

    raw_scores: torch.Tensor
    masked_logits: torch.Tensor
    eligible_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class RetrievalPreview:
    """Bounded differentiable summary consumed by the controller."""

    features: torch.Tensor
    has_candidate: torch.Tensor


def _preview_pool(
    scores: torch.Tensor,
    embeddings: torch.Tensor,
    eligible: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Soft-pool top records, including every exact tie at the boundary."""
    live = eligible.any(dim=1)
    output = embeddings.new_zeros((embeddings.shape[0], embeddings.shape[2]))
    if not bool(live.any()):
        return output
    live_scores = scores[live]
    live_eligible = eligible[live]
    logits = live_scores.masked_fill(~live_eligible, -torch.inf)
    boundary = logits.topk(k=top_k, dim=1).values[:, -1:]
    selected = live_eligible & (logits >= boundary)
    weights = logits.masked_fill(~selected, -torch.inf).softmax(dim=1)
    pooled = (weights.unsqueeze(-1) * embeddings[live]).sum(dim=1)
    return output.index_copy(0, live.nonzero(as_tuple=True)[0], pooled)


class RetrievalScorer(nn.Module):
    """Score all bounded memory slots from public state with one shared network."""

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
            # The model assembler owns injected shared tables. Keeping only a plain
            # reference here prevents duplicate module/state-dict registration.
            object.__setattr__(self, "_injected_mode_embedding", mode_embedding)
        self.query_network = nn.Sequential(
            nn.Linear(config.retrieval_query_input_dim, config.query_dim),
            nn.SiLU(),
        )
        self.record_projection = nn.Linear(
            config.record_dim, config.bilinear_record_output_dim, bias=False
        )
        self.pair_network = nn.Sequential(
            nn.Linear(config.retrieval_pair_input_dim, config.query_dim),
            nn.SiLU(),
            nn.Linear(config.query_dim, 1),
        )

    @property
    def mode_embedding(self) -> nn.Embedding:
        """Return the owned or externally owned shared mode table."""
        owned = self._modules.get("_owned_mode_embedding")
        if owned is not None:
            return owned  # type: ignore[return-value]
        return object.__getattribute__(self, "_injected_mode_embedding")

    def _query(self, context: ModelContext) -> torch.Tensor:
        latent = context.workspace.latent
        z_fast = latent[:, :256]
        z_slow = latent[:, 256:320]
        drives = latent[:, 320:328]
        focus_key = latent[:, 328:392]
        support_weights = context.support_mask.unsqueeze(-1).to(context.memory_embeddings.dtype)
        support_counts = support_weights.sum(dim=1).clamp_min(1.0)
        support_summary = (context.memory_embeddings * support_weights).sum(dim=1) / support_counts
        inputs = torch.cat(
            (
                focus_key,
                z_fast,
                z_slow,
                drives,
                self.mode_embedding(context.modes),
                support_summary,
            ),
            dim=1,
        )
        return self.query_network(inputs)

    def forward(self, context: ModelContext) -> RetrievalScores:
        """Return finite learned scores plus logits masked only by legal eligibility."""
        if not isinstance(context, ModelContext):
            raise TypeError("context must be a ModelContext")
        if next(self.parameters()).device != context.device:
            raise NeuralError("retrieval scorer and context must use one device")
        query = self._query(context)
        projected_records = self.record_projection(context.memory_embeddings)
        bilinear = (query.unsqueeze(1) * projected_records).sum(dim=2) / math.sqrt(
            self.config.query_dim
        )
        expanded_query = query.unsqueeze(1).expand(-1, self.config.memory_slots, -1)
        pair_inputs = torch.cat((expanded_query, context.memory_embeddings), dim=2)
        pair_scores = self.pair_network(pair_inputs).squeeze(2)
        raw_scores = bilinear + pair_scores
        return RetrievalScores(
            raw_scores=raw_scores,
            masked_logits=raw_scores.masked_fill(~context.eligibility, -torch.inf),
            eligible_mask=context.eligibility,
        )

    def preview(self, context: ModelContext) -> RetrievalPreview:
        """Re-score and summarize current state without changing memory or workspace."""
        scores = self(context)
        eligible = scores.eligible_mask
        live = eligible.any(dim=1)
        zero_anchor = scores.raw_scores.sum(dim=1, keepdim=True) * 0.0
        features = zero_anchor.expand(-1, self.config.preview_dim).clone()
        if bool(live.any()):
            live_logits = scores.masked_logits[live]
            live_eligible = eligible[live]
            counts = live_eligible.sum(dim=1)
            top_values = live_logits.topk(k=2, dim=1).values
            margins = torch.where(
                counts > 1,
                top_values[:, 0] - top_values[:, 1],
                torch.zeros_like(top_values[:, 0]),
            )
            probabilities = live_logits.softmax(dim=1)
            entropy = torch.logsumexp(live_logits, dim=1) - (
                probabilities * scores.raw_scores[live]
            ).sum(dim=1)
            metrics = torch.stack(
                (
                    top_values[:, 0],
                    margins,
                    entropy,
                    counts.to(scores.raw_scores.dtype) / self.config.memory_slots,
                ),
                dim=1,
            )
            pooled = _preview_pool(
                scores.raw_scores,
                context.memory_embeddings,
                eligible,
                self.config.top_k,
            )[live]
            from silent_cascade.eval.compute import _record_functional_operations

            _record_functional_operations(
                softmax_ops=2 * live_logits.numel(),
                logsumexp_ops=live_logits.numel(),
            )
            live_features = torch.cat((metrics, pooled), dim=1)
            features = features.index_copy(0, live.nonzero(as_tuple=True)[0], live_features)
        return RetrievalPreview(features=features, has_candidate=live)


def select_record_ids(
    scores: RetrievalScores,
    record_ids: tuple[tuple[int | None, ...], ...],
) -> tuple[int | None, ...]:
    """Decode scores on the host, resolving exact ties by lowest record ID."""
    if not isinstance(scores, RetrievalScores):
        raise TypeError("scores must be RetrievalScores")
    batch_size, slots = scores.raw_scores.shape
    if (
        not isinstance(record_ids, tuple)
        or len(record_ids) != batch_size
        or any(not isinstance(row, tuple) or len(row) != slots for row in record_ids)
    ):
        raise NeuralError("record_ids must match the score batch and slot dimensions")
    raw_rows = scores.raw_scores.detach().cpu().tolist()
    eligible_rows = scores.eligible_mask.detach().cpu().tolist()
    selected: list[int | None] = []
    for raw_row, eligible_row, id_row in zip(raw_rows, eligible_rows, record_ids, strict=True):
        candidates: list[tuple[float, int]] = []
        for raw_score, is_eligible, record_id in zip(raw_row, eligible_row, id_row, strict=True):
            if not is_eligible:
                continue
            if type(record_id) is not int:
                raise NeuralError("every eligible slot must have an exact integer record ID")
            candidates.append((raw_score, record_id))
        if not candidates:
            selected.append(None)
            continue
        maximum = max(raw_score for raw_score, _ in candidates)
        selected.append(
            min(record_id for raw_score, record_id in candidates if raw_score == maximum)
        )
    return tuple(selected)
