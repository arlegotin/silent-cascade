from pathlib import Path

import pytest
import torch

from silent_cascade.config import resolve_config
from silent_cascade.memory.store import BoundedMemory, append_perceived_fact
from silent_cascade.schemas import (
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    LinkFact,
    SafeFact,
)
from silent_cascade.train.config import Phase3Config

PHASE3_CONFIG_PATHS = (
    Path("configs/base.yaml"),
    Path("configs/data/primary.yaml"),
    Path("configs/model/event_flow.yaml"),
    Path("configs/model/neural_components.yaml"),
    Path("configs/train/one_hop.yaml"),
)


@pytest.fixture(scope="module")
def resolved_neural_config():
    return resolve_config(Phase3Config, PHASE3_CONFIG_PATHS)


@pytest.fixture(scope="module")
def neural_config(resolved_neural_config):
    return resolved_neural_config.config


@pytest.fixture
def public_memories() -> tuple[BoundedMemory, BoundedMemory]:
    """Hand-authored public memories with every semantic record kind."""
    first = BoundedMemory(64)
    events = (
        ExternalEvent(11, 1.0, ExternalEventKind.FACT, LinkFact(1, 2, confidence=0.8)),
        ExternalEvent(12, 2.5, ExternalEventKind.FACT, HazardFact(2, 3, 4.0, confidence=0.7)),
        ExternalEvent(13, 4.0, ExternalEventKind.FACT, SafeFact(3, confidence=0.6)),
    )
    for event in events:
        first = append_perceived_fact(first, event)
    second = append_perceived_fact(
        BoundedMemory(64),
        ExternalEvent(21, 10.0, ExternalEventKind.FACT, SafeFact(4, confidence=0.9)),
    )
    return first, second


@pytest.fixture
def model_context(public_memories, neural_config):
    """Public-only neural context with no support or hypothesis state."""
    from silent_cascade.memory.encoder import RecordEncoder
    from silent_cascade.memory.tensor_store import pack_memory
    from silent_cascade.models.types import ModelContext, TensorWorkspace

    records = pack_memory(public_memories, (0.0, 9.0), device="cpu")
    memory_embeddings = RecordEncoder(neural_config.neural)(records).detach()
    return ModelContext(
        workspace=TensorWorkspace.zeros(2, "cpu"),
        memory_embeddings=memory_embeddings,
        eligibility=records.valid_mask,
        support_mask=torch.zeros((2, 64), dtype=torch.bool),
        active_slot_indices=torch.full((2,), -1, dtype=torch.int64),
        modes=torch.tensor([1, 4], dtype=torch.int64),
        time_features=torch.tensor(
            [[torch.log1p(torch.tensor(5.0)), torch.log1p(torch.tensor(1.0))], [0.0, 0.0]],
            dtype=torch.float32,
        ),
        hypothesis_features=torch.zeros((2, 8), dtype=torch.float32),
    )
