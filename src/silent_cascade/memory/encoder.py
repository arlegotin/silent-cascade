"""Differentiable encoder for public memory records."""

import torch
from torch import nn

from silent_cascade.memory.tensor_store import RecordTensorBatch
from silent_cascade.models.config import NeuralModelConfig
from silent_cascade.models.errors import NeuralError


class RecordEncoder(nn.Module):
    """Encode public record fields with categorical tables shared by the model."""

    def __init__(self, config: NeuralModelConfig) -> None:
        super().__init__()
        if not isinstance(config, NeuralModelConfig):
            raise TypeError("config must be a NeuralModelConfig")
        self.entity_embedding = nn.Embedding(config.entity_count, config.entity_dim)
        self.kind_embedding = nn.Embedding(3, config.kind_dim)
        self.hazard_embedding = nn.Embedding(5, config.hazard_dim)
        self.provenance_embedding = nn.Embedding(3, config.provenance_dim)
        self.network = nn.Sequential(
            nn.Linear(config.record_input_dim, config.record_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.record_hidden_dim, config.record_dim),
        )

    def encode_entities(self, ids: torch.Tensor) -> torch.Tensor:
        """Encode entity IDs through the same table used for record subjects and objects."""
        if not isinstance(ids, torch.Tensor) or ids.dtype is not torch.int64:
            raise NeuralError("entity IDs must be an int64 tensor")
        if ids.device != self.entity_embedding.weight.device:
            raise NeuralError("entity IDs and encoder must use one device")
        if bool(((ids < 0) | (ids >= self.entity_embedding.num_embeddings)).any()):
            raise NeuralError("entity IDs must be in the configured entity range")
        return self.entity_embedding(ids)

    def forward(self, records: RecordTensorBatch) -> torch.Tensor:
        """Return one fresh embedding per slot, with exact zeros for padding."""
        if not isinstance(records, RecordTensorBatch):
            raise TypeError("records must be a RecordTensorBatch")
        if records.device != self.entity_embedding.weight.device:
            raise NeuralError("record tensors and encoder must use one device")
        object_present = records.scalar_features[..., :1]
        features = torch.cat(
            (
                self.encode_entities(records.subject_ids),
                self.encode_entities(records.object_ids) * object_present,
                self.kind_embedding(records.kind_ids),
                self.hazard_embedding(records.hazard_ids),
                self.provenance_embedding(records.provenance_ids),
                records.scalar_features,
            ),
            dim=-1,
        )
        return self.network(features) * records.valid_mask.unsqueeze(-1)
