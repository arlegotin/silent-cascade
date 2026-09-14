"""Assembly of the bounded trainable EventFlow neural components."""

from __future__ import annotations

import torch
from torch import nn

from silent_cascade.memory.encoder import RecordEncoder
from silent_cascade.memory.retrieval import RetrievalPreview, RetrievalScorer, RetrievalScores
from silent_cascade.models.config import NeuralModelConfig
from silent_cascade.models.controller import FlowGuardController
from silent_cascade.models.dynamics import BatchedSegmentParameters
from silent_cascade.models.errors import NeuralError
from silent_cascade.models.heads import (
    ActionHeads,
    ActionPredictions,
    ComposeHeads,
    ComposePredictions,
)
from silent_cascade.models.jump import ExternalEncoder, SharedJump, inject_external
from silent_cascade.models.types import ExternalFeatures, ModelContext, TensorWorkspace

_OBSERVING = 0
_SEARCHING = 1


def _select_context(context: ModelContext, rows: torch.Tensor) -> ModelContext:
    workspace = TensorWorkspace._from_functional_update(
        context.workspace.latent[rows], context.workspace.accumulators[rows]
    )
    return ModelContext(
        workspace=workspace,
        memory_embeddings=context.memory_embeddings[rows],
        eligibility=context.eligibility[rows],
        support_mask=context.support_mask[rows],
        active_slot_indices=context.active_slot_indices[rows],
        modes=context.modes[rows],
        time_features=context.time_features[rows],
        hypothesis_features=context.hypothesis_features[rows],
    )


class EventFlowModel(nn.Module):
    """Own exactly one copy of every learned EventFlow component and shared table."""

    def __init__(self, config: NeuralModelConfig) -> None:
        super().__init__()
        if not isinstance(config, NeuralModelConfig):
            raise TypeError("config must be a NeuralModelConfig")
        self.config = config
        self.mode_embedding = nn.Embedding(config.mode_count, config.mode_dim)
        self.record_encoder = RecordEncoder(config)
        self.scorer = RetrievalScorer(config, mode_embedding=self.mode_embedding)
        self.controller = FlowGuardController(config, mode_embedding=self.mode_embedding)
        self.shared_jump = SharedJump(config, mode_embedding=self.mode_embedding)
        self.external_encoder = ExternalEncoder(config, self.record_encoder)
        self.focus_projection = nn.Linear(config.entity_dim, 64)
        self.compose_heads = ComposeHeads(config, mode_embedding=self.mode_embedding)
        self.action_heads = ActionHeads(config, mode_embedding=self.mode_embedding)

    def initial_context(self, batch_size: int, *, device: str) -> ModelContext:
        workspace = TensorWorkspace.zeros(batch_size, device)
        if self.mode_embedding.weight.device != workspace.device:
            raise NeuralError("model and requested initial context must use one device")
        return ModelContext(
            workspace=workspace,
            memory_embeddings=torch.zeros(
                batch_size,
                self.config.memory_slots,
                self.config.record_dim,
                dtype=torch.float32,
                device=workspace.device,
            ),
            eligibility=torch.zeros(
                batch_size,
                self.config.memory_slots,
                dtype=torch.bool,
                device=workspace.device,
            ),
            support_mask=torch.zeros(
                batch_size,
                self.config.memory_slots,
                dtype=torch.bool,
                device=workspace.device,
            ),
            active_slot_indices=torch.full(
                (batch_size,), -1, dtype=torch.int64, device=workspace.device
            ),
            modes=torch.full((batch_size,), _OBSERVING, dtype=torch.int64, device=workspace.device),
            time_features=torch.zeros(batch_size, 2, dtype=torch.float32, device=workspace.device),
            hypothesis_features=torch.zeros(
                batch_size, 8, dtype=torch.float32, device=workspace.device
            ),
        )

    def observe(self, context: ModelContext, event: ExternalFeatures) -> ModelContext:
        if not isinstance(context, ModelContext):
            raise TypeError("context must be a ModelContext")
        if not isinstance(event, ExternalFeatures):
            raise TypeError("event must be ExternalFeatures")
        if event.batch_size != context.batch_size or event.device != context.device:
            raise NeuralError("external event and context must share a batch and device")
        workspace = inject_external(context.workspace, self.external_encoder(event))
        activation = event.event_kinds == 1
        latent = workspace.latent
        if bool(activation.any()):
            activation_rows = activation.nonzero(as_tuple=True)[0]
            activation_focus = torch.tanh(
                self.focus_projection(
                    self.record_encoder.encode_entities(
                        event.activation_entity_ids[activation_rows]
                    )
                )
            )
            focus = latent[:, 328:392].index_copy(0, activation_rows, activation_focus)
            latent = torch.cat((latent[:, :328], focus, latent[:, 392:]), dim=1)
            workspace = TensorWorkspace._from_functional_update(latent, workspace.accumulators)
        modes = torch.where(activation, torch.full_like(context.modes, _SEARCHING), context.modes)
        return ModelContext(
            workspace=workspace,
            memory_embeddings=context.memory_embeddings,
            eligibility=context.eligibility,
            support_mask=context.support_mask,
            active_slot_indices=context.active_slot_indices,
            modes=modes,
            time_features=event.time_features,
            hypothesis_features=context.hypothesis_features,
        )

    def _active_rows(self, context: ModelContext) -> torch.Tensor:
        if not isinstance(context, ModelContext):
            raise TypeError("context must be a ModelContext")
        return (context.modes != _OBSERVING).nonzero(as_tuple=True)[0]

    def _active_preview(self, context: ModelContext) -> RetrievalPreview:
        rows = self._active_rows(context)
        zero_anchor = context.workspace.latent.sum(dim=1, keepdim=True) * 0.0
        features = zero_anchor.expand(-1, self.config.preview_dim).clone()
        has_candidate = torch.zeros(context.batch_size, dtype=torch.bool, device=context.device)
        if rows.numel() == 0:
            return RetrievalPreview(features=features, has_candidate=has_candidate)
        active_preview = self.scorer.preview(_select_context(context, rows))
        return RetrievalPreview(
            features=features.index_copy(0, rows, active_preview.features),
            has_candidate=has_candidate.index_copy(0, rows, active_preview.has_candidate),
        )

    def preview_and_control(
        self, context: ModelContext
    ) -> tuple[RetrievalPreview, BatchedSegmentParameters]:
        preview = self._active_preview(context)
        return preview, self.controller(context, preview)

    def recall_scores(self, context: ModelContext) -> RetrievalScores:
        rows = self._active_rows(context)
        raw_scores = context.workspace.latent.sum(dim=1, keepdim=True) * 0.0
        raw_scores = raw_scores.expand(-1, self.config.memory_slots).clone()
        eligible = torch.zeros(
            context.batch_size,
            self.config.memory_slots,
            dtype=torch.bool,
            device=context.device,
        )
        masked_logits = torch.full_like(raw_scores, -torch.inf)
        if rows.numel() == 0:
            return RetrievalScores(raw_scores, masked_logits, eligible)
        active_scores = self.scorer(_select_context(context, rows))
        return RetrievalScores(
            raw_scores=raw_scores.index_copy(0, rows, active_scores.raw_scores),
            masked_logits=masked_logits.index_copy(0, rows, active_scores.masked_logits),
            eligible_mask=eligible.index_copy(0, rows, active_scores.eligible_mask),
        )

    def compose(self, context: ModelContext) -> ComposePredictions:
        return self.compose_heads(context)

    def action(self, context: ModelContext) -> ActionPredictions:
        return self.action_heads(context)

    def jump(self, context: ModelContext, event_kinds: torch.Tensor) -> TensorWorkspace:
        return self.shared_jump(context, event_kinds)
