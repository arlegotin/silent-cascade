"""Immutable public event schemas for Observation-Free Deadline episodes."""

import math
from dataclasses import dataclass
from enum import StrEnum

from silent_cascade.errors import EpisodeInvariantError

MAX_ENTITY_ID = 63
MAX_HAZARD_TYPE = 3


class RecordKind(StrEnum):
    LINK = "link"
    HAZARD = "hazard"
    SAFE = "safe"


class ExternalEventKind(StrEnum):
    FACT = "fact"
    ACTIVATE = "activate"
    OUTCOME = "outcome"
    END = "end"


class InternalEventKind(StrEnum):
    RECALL = "recall"
    COMPOSE = "compose"
    ACT = "act"
    NOOP = "noop"


class Condition(StrEnum):
    ORACLE = "oracle"
    RANDOM = "random"
    ONE_SHOT = "one_shot_all_memory"
    PONDER_AT_ACTIVATION = "ponder_at_activation"
    FLOW_ONLY = "flow_only"
    FIXED_TICK = "fixed_tick_recurrent"
    PERIODIC_MATCHED = "periodic_matched_count"
    RANDOM_TIME_MATCHED = "random_time_matched_count"
    PERSISTENT_FROZEN = "persistent_frozen"
    EVENT_FLOW = "event_flow"


class Provenance(StrEnum):
    PERCEIVED = "perceived"
    INFERRED = "inferred"
    SIMULATED = "simulated"


class Mode(StrEnum):
    OBSERVING = "observing"
    SEARCHING = "searching"
    HAVE_MEMORY = "have_memory"
    HOLDING_HAZARD = "holding_hazard"
    QUIESCENT = "quiescent"
    TERMINAL = "terminal"


def _require_int(value: object, name: str, *, minimum: int = 0, maximum: int | None = None) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is out of bounds")


def _require_float(value: object, name: str, *, minimum: float | None = None) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be finite and within bounds")


def _require_confidence(value: object) -> None:
    _require_float(value, "confidence")
    if not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be in [0, 1]")


def _require_ids(value: object, name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    for item in value:
        _require_int(item, name)


@dataclass(frozen=True, slots=True)
class LinkFact:
    source_node: int
    target_node: int
    confidence: float = 1.0

    def __post_init__(self) -> None:
        _require_int(self.source_node, "source_node", maximum=MAX_ENTITY_ID)
        _require_int(self.target_node, "target_node", maximum=MAX_ENTITY_ID)
        _require_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class HazardFact:
    node: int
    hazard_type: int
    delay: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        _require_int(self.node, "node", maximum=MAX_ENTITY_ID)
        _require_int(self.hazard_type, "hazard_type", maximum=MAX_HAZARD_TYPE)
        _require_float(self.delay, "delay", minimum=0.0)
        _require_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class SafeFact:
    node: int
    confidence: float = 1.0

    def __post_init__(self) -> None:
        _require_int(self.node, "node", maximum=MAX_ENTITY_ID)
        _require_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class ActivationPayload:
    start_node: int

    def __post_init__(self) -> None:
        _require_int(self.start_node, "start_node", maximum=MAX_ENTITY_ID)


type FactPayload = LinkFact | HazardFact | SafeFact
type ExternalPayload = FactPayload | ActivationPayload | None


@dataclass(frozen=True, slots=True)
class AgentInit:
    episode_public_id: str
    memory_capacity: int
    hazard_type_count: int
    initial_time: float

    def __post_init__(self) -> None:
        if not isinstance(self.episode_public_id, str) or not self.episode_public_id:
            raise ValueError("episode_public_id must be a nonempty string")
        _require_int(self.memory_capacity, "memory_capacity", minimum=1, maximum=64)
        _require_int(self.hazard_type_count, "hazard_type_count", minimum=1, maximum=4)
        _require_float(self.initial_time, "initial_time")


@dataclass(frozen=True, slots=True)
class ExternalEvent:
    event_id: int
    timestamp: float
    kind: ExternalEventKind
    payload: ExternalPayload

    def __post_init__(self) -> None:
        _require_int(self.event_id, "event_id")
        _require_float(self.timestamp, "timestamp")
        if not isinstance(self.kind, ExternalEventKind):
            raise TypeError("kind must be an ExternalEventKind")
        allowed_payloads: dict[ExternalEventKind, tuple[type[object], ...]] = {
            ExternalEventKind.FACT: (LinkFact, HazardFact, SafeFact),
            ExternalEventKind.ACTIVATE: (ActivationPayload,),
            ExternalEventKind.OUTCOME: (),
            ExternalEventKind.END: (),
        }
        expected = allowed_payloads[self.kind]
        if expected:
            if not isinstance(self.payload, expected):
                raise ValueError("event kind and payload do not match")
        elif self.payload is not None:
            raise ValueError("event kind and payload do not match")


@dataclass(frozen=True, slots=True)
class InternalEvent:
    event_id: int
    parent_event_id: int
    timestamp: float
    kind: InternalEventKind
    guard_index: int
    predicted_delta: float

    def __post_init__(self) -> None:
        _require_int(self.event_id, "event_id")
        _require_int(self.parent_event_id, "parent_event_id")
        _require_float(self.timestamp, "timestamp")
        if not isinstance(self.kind, InternalEventKind):
            raise TypeError("kind must be an InternalEventKind")
        _require_int(self.guard_index, "guard_index")
        _require_float(self.predicted_delta, "predicted_delta", minimum=0.0)


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    record_id: int
    kind: RecordKind
    subject_id: int
    object_id: int | None
    hazard_type: int | None
    delay: float | None
    observed_at: float
    confidence: float
    provenance: Provenance
    support_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_int(self.record_id, "record_id")
        if not isinstance(self.kind, RecordKind):
            raise TypeError("kind must be a RecordKind")
        _require_int(self.subject_id, "subject_id", maximum=MAX_ENTITY_ID)
        _require_float(self.observed_at, "observed_at")
        _require_confidence(self.confidence)
        if not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be a Provenance")
        _require_ids(self.support_ids, "support_ids")
        if self.kind is RecordKind.LINK:
            if self.object_id is None or self.hazard_type is not None or self.delay is not None:
                raise ValueError("LINK records require only object_id")
            _require_int(self.object_id, "object_id", maximum=MAX_ENTITY_ID)
        elif self.kind is RecordKind.HAZARD:
            if self.object_id is not None or self.hazard_type is None or self.delay is None:
                raise ValueError("HAZARD records require hazard_type and delay")
            _require_int(self.hazard_type, "hazard_type", maximum=MAX_HAZARD_TYPE)
            _require_float(self.delay, "delay", minimum=0.0)
        elif self.object_id is not None or self.hazard_type is not None or self.delay is not None:
            raise ValueError("SAFE records have no object, hazard, or delay")


@dataclass(frozen=True, slots=True)
class Hypothesis:
    hazard_type: int | None
    deadline: float | None
    confidence: float
    provenance: Provenance
    support_ids: tuple[int, ...]
    is_safe: bool

    def __post_init__(self) -> None:
        if self.hazard_type is not None:
            _require_int(self.hazard_type, "hazard_type", maximum=MAX_HAZARD_TYPE)
        if self.deadline is not None:
            _require_float(self.deadline, "deadline")
        _require_confidence(self.confidence)
        if not isinstance(self.provenance, Provenance):
            raise TypeError("provenance must be a Provenance")
        _require_ids(self.support_ids, "support_ids")
        if self.provenance is Provenance.INFERRED and (
            not self.support_ids or len(set(self.support_ids)) != len(self.support_ids)
        ):
            raise ValueError("inferred hypotheses require nonempty unique support_ids")
        if type(self.is_safe) is not bool:
            raise TypeError("is_safe must be a bool")
        if self.is_safe and (self.hazard_type is not None or self.deadline is not None):
            raise ValueError("safe hypotheses have no hazard or deadline")
        if not self.is_safe and (self.hazard_type is None or self.deadline is None):
            raise ValueError("hazard hypotheses require hazard and deadline")


@dataclass(frozen=True, slots=True)
class Action:
    hazard_type: int
    timestamp: float
    caused_by_event_id: int

    def __post_init__(self) -> None:
        _require_int(self.hazard_type, "hazard_type", maximum=MAX_HAZARD_TYPE)
        _require_float(self.timestamp, "timestamp")
        _require_int(self.caused_by_event_id, "caused_by_event_id")


def fact_event_to_memory_record(event: ExternalEvent) -> MemoryRecord:
    """Convert one public fact event into its corresponding memory record."""
    if event.kind is not ExternalEventKind.FACT:
        raise EpisodeInvariantError("only FACT events become memory records")
    payload = event.payload
    if isinstance(payload, LinkFact):
        kind, subject, object_id, hazard_type, delay = (
            RecordKind.LINK,
            payload.source_node,
            payload.target_node,
            None,
            None,
        )
    elif isinstance(payload, HazardFact):
        kind, subject, object_id, hazard_type, delay = (
            RecordKind.HAZARD,
            payload.node,
            None,
            payload.hazard_type,
            payload.delay,
        )
    elif isinstance(payload, SafeFact):
        kind, subject, object_id, hazard_type, delay = (
            RecordKind.SAFE,
            payload.node,
            None,
            None,
            None,
        )
    else:
        raise EpisodeInvariantError("FACT has an invalid payload")
    return MemoryRecord(
        record_id=event.event_id,
        kind=kind,
        subject_id=subject,
        object_id=object_id,
        hazard_type=hazard_type,
        delay=delay,
        observed_at=event.timestamp,
        confidence=payload.confidence,
        provenance=Provenance.PERCEIVED,
        support_ids=(),
    )
