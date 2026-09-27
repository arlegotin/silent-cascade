"""Public-memory recurrent competitor for the exploratory Phase 5A pilot."""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from pydantic import Field
from torch import nn

from silent_cascade.eval.compute import (
    RuntimeCompute,
    _record_ponder_retrieval,
    parameter_counts,
)
from silent_cascade.memory.encoder import RecordEncoder
from silent_cascade.models.config import NeuralModelConfig
from silent_cascade.models.types import PublicInputBatch
from silent_cascade.schemas import Action
from silent_cascade.validation import StrictModel

MATCHED_EVENTFLOW_PARAMETERS = 2_781_042
PONDER_CAPS = (4, 8, 12, 16, 24)
_WIDTHS = (640, 672, 704, 736, 768, 800)
_HEAD_SIZES = {
    "role": 5,
    "focus": 64,
    "hazard": 4,
    "log_delay": 1,
    "deadline": 1,
    "status": 3,
    "confidence": 1,
    "support": 1,
    "continue": 1,
    "action": 5,
    "action_offset": 1,
    "halt": 1,
}


def _record_config() -> NeuralModelConfig:
    return NeuralModelConfig()


class PonderModelConfig(StrictModel):
    width: Literal[640, 672, 704, 736, 768, 800] = 800
    max_parameters: Literal[5_000_000] = 5_000_000
    record_config: NeuralModelConfig = Field(default_factory=_record_config)


@dataclass(frozen=True, slots=True)
class PonderEncoded:
    record_embeddings: torch.Tensor
    keys: torch.Tensor
    valid_mask: torch.Tensor
    record_ids: tuple[tuple[int | None, ...], ...]
    activation_embedding: torch.Tensor
    time_features: torch.Tensor
    activation_times: tuple[float, ...]
    hidden: torch.Tensor


@dataclass(frozen=True, slots=True)
class PonderStepOutput:
    retrieval_logits: torch.Tensor
    selected_slots: torch.Tensor
    selected_record_ids: tuple[int | None, ...]
    hidden: torch.Tensor
    heads: dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class PonderForward:
    retrieval_logits: torch.Tensor
    heads: dict[str, torch.Tensor]
    hidden: torch.Tensor


@dataclass(frozen=True, slots=True)
class PonderStep:
    cognitive_timestamp: float
    selected_record_id: int | None
    halt_probability: float
    action_class: int
    action_offset: float


@dataclass(frozen=True, slots=True)
class PonderDecision:
    action: Action | None
    steps: tuple[PonderStep, ...]
    stop_reason: Literal["halt", "cap_reached"]
    compute: RuntimeCompute


class ActivationPonderModel(nn.Module):
    """Full-memory retrieval plus one shared GRU update and readout per step."""

    def __init__(self, config: PonderModelConfig) -> None:
        super().__init__()
        if not isinstance(config, PonderModelConfig):
            raise TypeError("ponderer requires a PonderModelConfig")
        self.config = config
        self.record_encoder = RecordEncoder(config.record_config)
        input_width = config.record_config.record_dim + config.record_config.entity_dim + 2
        context_width = config.width + config.record_config.entity_dim + 2
        self.initial = nn.Linear(input_width, config.width)
        self.query = nn.Linear(context_width, 256)
        self.query_activation = nn.SiLU()
        self.key = nn.Linear(config.record_config.record_dim, 256)
        self.null_key = nn.Parameter(torch.empty(256))
        self.null_embedding = nn.Parameter(torch.empty(config.record_config.record_dim))
        self.gru = nn.GRUCell(input_width, config.width)
        self.trunk = nn.Linear(context_width, 256)
        self.trunk_activation = nn.SiLU()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(256, size) for name, size in _HEAD_SIZES.items()}
        )
        nn.init.normal_(self.null_key, std=0.02)
        nn.init.normal_(self.null_embedding, std=0.02)
        if parameter_counts(self)["total"] > config.max_parameters:
            raise ValueError("ponderer exceeds the fixed 5M parameter ceiling")

    def encode_public(self, public: PublicInputBatch) -> PonderEncoded:
        if not isinstance(public, PublicInputBatch):
            raise TypeError("ponderer receives only a PublicInputBatch")
        if public.records.device != next(self.parameters()).device:
            raise ValueError("public records and ponderer must use one device")
        lengths = public.observation_mask.sum(dim=1).to(torch.int64)
        if bool((lengths < 1).any()):
            raise ValueError("ponderer requires public activation")
        rows = torch.arange(public.records.batch_size, device=public.records.device)
        final = lengths - 1
        if not bool((public.observation_kind[rows, final] == 1).all()):
            raise ValueError("last public observation must be ACTIVATE")
        records = self.record_encoder(public.records)
        mask = public.records.valid_mask
        mean = records.sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
        entity = self.record_encoder.encode_entities(public.activation_entities[rows, final])
        time = public.observation_time_features[rows, final]
        hidden = torch.tanh(self.initial(torch.cat((mean, entity, time), dim=-1)))
        return PonderEncoded(
            record_embeddings=records,
            keys=self.key(records),
            valid_mask=mask,
            record_ids=public.records.record_ids,
            activation_embedding=entity,
            time_features=time,
            activation_times=tuple(
                float(public.observation_times[row, int(final[row])])
                for row in range(public.records.batch_size)
            ),
            hidden=hidden,
        )

    @staticmethod
    def _stable_selection(
        logits: torch.Tensor, record_ids: tuple[tuple[int | None, ...], ...]
    ) -> torch.Tensor:
        selected = []
        for row, scores in enumerate(logits.detach().cpu().tolist()):
            maximum = max(scores)
            tied = [index for index, value in enumerate(scores) if value == maximum]
            chosen = min(
                tied,
                key=lambda index: (
                    record_ids[row][index]
                    if index < 64 and record_ids[row][index] is not None
                    else 2**63,
                    index,
                ),
            )
            selected.append(chosen)
        return torch.tensor(selected, dtype=torch.int64, device=logits.device)

    def transition(
        self,
        encoded: PonderEncoded,
        hidden: torch.Tensor,
        teacher_slot: torch.Tensor | None = None,
    ) -> PonderStepOutput:
        if hidden.shape != (len(encoded.record_ids), self.config.width):
            raise ValueError("ponder hidden state has wrong shape")
        context = torch.cat((hidden, encoded.activation_embedding, encoded.time_features), dim=-1)
        query = self.query_activation(self.query(context))
        scores = torch.einsum("bd,bsd->bs", query, encoded.keys) / math.sqrt(256)
        null_score = (query * self.null_key).sum(dim=-1, keepdim=True) / math.sqrt(256)
        _record_ponder_retrieval(encoded.valid_mask, 256)
        logits = torch.cat(
            (scores.masked_fill(~encoded.valid_mask, torch.finfo(scores.dtype).min), null_score),
            dim=-1,
        )
        if teacher_slot is None:
            chosen = self._stable_selection(logits, encoded.record_ids)
        elif bool((teacher_slot < 0).any()):
            chosen = torch.where(
                teacher_slot >= 0,
                teacher_slot,
                self._stable_selection(logits, encoded.record_ids),
            )
        else:
            chosen = teacher_slot
        if (
            chosen.shape != (hidden.shape[0],)
            or chosen.dtype != torch.int64
            or bool(((chosen < 0) | (chosen > 64)).any())
        ):
            raise ValueError("ponder selected slot must be an int64 record or null")
        if bool(
            (
                (chosen < 64)
                & ~encoded.valid_mask.gather(1, chosen.clamp_max(63)[:, None]).squeeze(1)
            ).any()
        ):
            raise ValueError("ponder cannot read a padded memory slot")
        gathered = encoded.record_embeddings.gather(
            1, chosen.clamp_max(63)[:, None, None].expand(-1, 1, 96)
        ).squeeze(1)
        read = torch.where((chosen == 64)[:, None], self.null_embedding, gathered)
        next_hidden = self.gru(
            torch.cat((read, encoded.activation_embedding, encoded.time_features), dim=-1),
            hidden,
        )
        trunk = self.trunk_activation(
            self.trunk(
                torch.cat(
                    (next_hidden, encoded.activation_embedding, encoded.time_features), dim=-1
                )
            )
        )
        heads = {name: layer(trunk) for name, layer in self.heads.items()}
        ids = tuple(
            None if int(slot) == 64 else encoded.record_ids[row][int(slot)]
            for row, slot in enumerate(chosen.detach().cpu().tolist())
        )
        return PonderStepOutput(logits, chosen, ids, next_hidden, heads)

    def forward_teacher(
        self, public: PublicInputBatch, teacher_slots: torch.Tensor, step_mask: torch.Tensor
    ) -> PonderForward:
        if teacher_slots.ndim != 2 or teacher_slots.shape != step_mask.shape:
            raise ValueError("teacher slots and masks must align as [B,T]")
        encoded = self.encode_public(public)
        hidden = encoded.hidden
        retrieval = []
        heads = {name: [] for name in _HEAD_SIZES}
        states = []
        for index in range(teacher_slots.shape[1]):
            output = self.transition(encoded, hidden, teacher_slots[:, index])
            hidden = output.hidden
            retrieval.append(output.retrieval_logits)
            states.append(hidden)
            for name in heads:
                heads[name].append(output.heads[name])
        return PonderForward(
            retrieval_logits=torch.stack(retrieval, dim=1),
            heads={name: torch.stack(values, dim=1) for name, values in heads.items()},
            hidden=torch.stack(states, dim=1),
        )


def matched_ponder_config() -> PonderModelConfig:
    """Choose once by exact parameter arithmetic, breaking ties toward smaller width."""
    with torch.random.fork_rng(devices=[]):
        candidates = []
        for width in _WIDTHS:
            config = PonderModelConfig(width=width)
            count = parameter_counts(ActivationPonderModel(config))["total"]
            if count <= 5_000_000 and abs(count - MATCHED_EVENTFLOW_PARAMETERS) <= (
                MATCHED_EVENTFLOW_PARAMETERS * 0.10
            ):
                candidates.append((abs(count - MATCHED_EVENTFLOW_PARAMETERS), width))
    if not candidates:
        raise ValueError("no fixed ponderer width satisfies the parameter envelope")
    return PonderModelConfig(width=min(candidates)[1])
