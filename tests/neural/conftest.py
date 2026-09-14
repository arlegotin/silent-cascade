from pathlib import Path

import pytest

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
