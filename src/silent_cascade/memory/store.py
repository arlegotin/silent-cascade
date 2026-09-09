"""Immutable bounded runtime metadata for public memory records."""

import math
from dataclasses import dataclass, replace

from silent_cascade.errors import DynamicsError
from silent_cascade.schemas import (
    ExternalEvent,
    MemoryRecord,
    RecordKind,
    fact_event_to_memory_record,
)

_PRIMARY_MEMORY_CAPACITY = 64


def _require_finite_float(value: object, name: str) -> float:
    if type(value) is not float:
        raise DynamicsError(f"{name} must be an exact float")
    if not math.isfinite(value):
        raise DynamicsError(f"{name} must be finite")
    return value


def _require_record_id(value: object) -> int:
    if type(value) is not int or value < 0:
        raise DynamicsError("record_id must be a nonnegative exact integer")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeMemoryRecord:
    """Immutable runtime metadata that wraps one immutable semantic record."""

    record: MemoryRecord
    refractory_until: float = 0.0
    valid: bool = True
    consumed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.record, MemoryRecord):
            raise TypeError("record must be a MemoryRecord")
        _require_finite_float(self.refractory_until, "refractory_until")
        if type(self.valid) is not bool:
            raise TypeError("valid must be a bool")
        if type(self.consumed) is not bool:
            raise TypeError("consumed must be a bool")


@dataclass(frozen=True, slots=True)
class BoundedMemory:
    """A fixed-capacity public record store in presentation order."""

    capacity: int
    records: tuple[RuntimeMemoryRecord, ...] = ()

    def __post_init__(self) -> None:
        if type(self.capacity) is not int or self.capacity != _PRIMARY_MEMORY_CAPACITY:
            raise DynamicsError("memory capacity must be exactly 64")
        if not isinstance(self.records, tuple):
            raise TypeError("records must be a tuple")
        if len(self.records) > self.capacity:
            raise DynamicsError("memory capacity exceeded")
        record_ids: set[int] = set()
        for slot in self.records:
            if not isinstance(slot, RuntimeMemoryRecord):
                raise TypeError("records must contain RuntimeMemoryRecord values")
            record_id = slot.record.record_id
            if record_id in record_ids:
                raise DynamicsError("duplicate memory record_id")
            record_ids.add(record_id)

    def lookup(self, record_id: int) -> RuntimeMemoryRecord:
        """Return one stored record, failing loudly when its ID is absent."""
        requested_id = _require_record_id(record_id)
        for slot in self.records:
            if slot.record.record_id == requested_id:
                return slot
        raise DynamicsError("memory record_id is missing")

    def legal_records(
        self,
        *,
        at_time: float,
        subject_id: int | None = None,
        kinds: frozenset[RecordKind] | None = None,
    ) -> tuple[RuntimeMemoryRecord, ...]:
        """Return legal records in arrival order without selecting or ranking them."""
        return legal_records(self, at_time=at_time, subject_id=subject_id, kinds=kinds)


def append_perceived_fact(memory: BoundedMemory, event: ExternalEvent) -> BoundedMemory:
    """Append one public FACT as a new immutable perceived record."""
    if not isinstance(memory, BoundedMemory):
        raise TypeError("memory must be a BoundedMemory")
    if not isinstance(event, ExternalEvent):
        raise TypeError("event must be an ExternalEvent")
    record = fact_event_to_memory_record(event)
    if len(memory.records) == memory.capacity:
        raise DynamicsError("memory capacity exceeded")
    if any(slot.record.record_id == record.record_id for slot in memory.records):
        raise DynamicsError("duplicate memory record_id")
    return replace(memory, records=(*memory.records, RuntimeMemoryRecord(record)))


def mark_recalled(
    memory: BoundedMemory,
    record_id: int,
    *,
    refractory_until: float,
) -> BoundedMemory:
    """Set a record's refractory bound without ever shortening it."""
    if not isinstance(memory, BoundedMemory):
        raise TypeError("memory must be a BoundedMemory")
    requested_id = _require_record_id(record_id)
    requested_until = _require_finite_float(refractory_until, "refractory_until")
    slot = memory.lookup(requested_id)
    replacement = replace(slot, refractory_until=max(slot.refractory_until, requested_until))
    return _replace_slot(memory, requested_id, replacement)


def mark_consumed(memory: BoundedMemory, record_id: int) -> BoundedMemory:
    """Mark a record consumed; this metadata transition is monotone."""
    if not isinstance(memory, BoundedMemory):
        raise TypeError("memory must be a BoundedMemory")
    requested_id = _require_record_id(record_id)
    slot = memory.lookup(requested_id)
    return _replace_slot(memory, requested_id, replace(slot, consumed=True))


def legal_records(
    memory: BoundedMemory,
    *,
    at_time: float,
    subject_id: int | None = None,
    kinds: frozenset[RecordKind] | None = None,
) -> tuple[RuntimeMemoryRecord, ...]:
    """Apply explicit legality masks while preserving record-arrival order."""
    if not isinstance(memory, BoundedMemory):
        raise TypeError("memory must be a BoundedMemory")
    current_time = _require_finite_float(at_time, "at_time")
    if subject_id is not None:
        _require_record_id(subject_id)
    if kinds is not None and (
        not isinstance(kinds, frozenset) or not all(isinstance(kind, RecordKind) for kind in kinds)
    ):
        raise TypeError("kinds must be a frozenset of RecordKind")
    return tuple(
        slot
        for slot in memory.records
        if slot.valid
        and not slot.consumed
        and slot.refractory_until <= current_time
        and (subject_id is None or slot.record.subject_id == subject_id)
        and (kinds is None or slot.record.kind in kinds)
    )


def _replace_slot(
    memory: BoundedMemory,
    record_id: int,
    replacement: RuntimeMemoryRecord,
) -> BoundedMemory:
    return replace(
        memory,
        records=tuple(
            replacement if slot.record.record_id == record_id else slot for slot in memory.records
        ),
    )
