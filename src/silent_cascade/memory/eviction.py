"""Explicit deterministic eviction used only by memory-pressure stress tests."""

from dataclasses import dataclass, replace

from silent_cascade.errors import DynamicsError
from silent_cascade.memory.store import (
    BoundedMemory,
    RuntimeMemoryRecord,
    append_perceived_fact,
)
from silent_cascade.schemas import ExternalEvent, fact_event_to_memory_record


@dataclass(frozen=True, slots=True)
class EvictionResult:
    """An immutable stress append result and the displaced host record ID, if any."""

    memory: BoundedMemory
    evicted_record_id: int | None


def _validated_memory(memory: object) -> BoundedMemory:
    if not isinstance(memory, BoundedMemory):
        raise TypeError("memory must be a BoundedMemory")
    validated = replace(memory)
    for slot in validated.records:
        replace(slot)
        replace(slot.record)
    return validated


def _validated_protected_ids(memory: BoundedMemory, protected_ids: object) -> frozenset[int]:
    if not isinstance(protected_ids, frozenset):
        raise TypeError("protected_ids must be a frozenset")
    if any(type(record_id) is not int or record_id < 0 for record_id in protected_ids):
        raise DynamicsError("protected IDs must be nonnegative exact integers")
    stored_ids = {slot.record.record_id for slot in memory.records}
    if not protected_ids <= stored_ids:
        raise DynamicsError("protected IDs must refer to stored memory records")
    return protected_ids


def _eviction_id(records: tuple[RuntimeMemoryRecord, ...], protected_ids: frozenset[int]) -> int:
    candidates = [slot.record for slot in records if slot.record.record_id not in protected_ids]
    if not candidates:
        raise DynamicsError("No unprotected memory record can be evicted")
    return min(
        candidates,
        key=lambda record: (record.confidence, record.observed_at, record.record_id),
    ).record_id


def append_with_stress_eviction(
    memory: BoundedMemory,
    event: ExternalEvent,
    *,
    protected_ids: frozenset[int],
) -> EvictionResult:
    """Append one perceived FACT, evicting a deterministic unprotected full slot."""
    validated_memory = _validated_memory(memory)
    validated_protected_ids = _validated_protected_ids(validated_memory, protected_ids)
    if len(validated_memory.records) == validated_memory.capacity and all(
        slot.record.record_id in validated_protected_ids for slot in validated_memory.records
    ):
        raise DynamicsError("No unprotected memory record can be evicted")
    if not isinstance(event, ExternalEvent):
        raise TypeError("event must be an ExternalEvent")
    validated_event = replace(event)
    incoming = fact_event_to_memory_record(validated_event)
    if any(slot.record.record_id == incoming.record_id for slot in validated_memory.records):
        raise DynamicsError("duplicate memory record_id")
    if len(validated_memory.records) < validated_memory.capacity:
        return EvictionResult(append_perceived_fact(validated_memory, validated_event), None)

    victim_id = _eviction_id(validated_memory.records, validated_protected_ids)
    replacement = RuntimeMemoryRecord(incoming)
    updated = replace(
        validated_memory,
        records=tuple(
            replacement if slot.record.record_id == victim_id else slot
            for slot in validated_memory.records
        ),
    )
    return EvictionResult(updated, victim_id)
