"""Behavioral tests for explicit memory-pressure stress eviction."""

from dataclasses import replace

import pytest

from silent_cascade.errors import DynamicsError
from silent_cascade.memory.store import BoundedMemory, RuntimeMemoryRecord, append_perceived_fact
from silent_cascade.schemas import (
    ExternalEvent,
    ExternalEventKind,
    MemoryRecord,
    Provenance,
    RecordKind,
    SafeFact,
)


def _safe_event(record_id: int, *, observed_at: float | None = None) -> ExternalEvent:
    return ExternalEvent(
        record_id,
        float(record_id) if observed_at is None else observed_at,
        ExternalEventKind.FACT,
        SafeFact(record_id % 64),
    )


def _full_memory() -> BoundedMemory:
    memory = BoundedMemory(64)
    for record_id in range(64):
        memory = append_perceived_fact(memory, _safe_event(record_id))
    return memory


def _replace_rank(
    memory: BoundedMemory, record_id: int, *, confidence: float, observed_at: float
) -> BoundedMemory:
    return replace(
        memory,
        records=tuple(
            replace(
                slot,
                record=replace(slot.record, confidence=confidence, observed_at=observed_at),
            )
            if slot.record.record_id == record_id
            else slot
            for slot in memory.records
        ),
    )


def test_nonfull_stress_append_does_not_evict_or_change_primary_behavior() -> None:
    from silent_cascade.memory.eviction import append_with_stress_eviction

    original = BoundedMemory(64)
    result = append_with_stress_eviction(original, _safe_event(1), protected_ids=frozenset())

    assert result.evicted_record_id is None
    assert [slot.record.record_id for slot in result.memory.records] == [1]
    assert original.records == ()


def test_eviction_uses_confidence_then_observation_time_then_record_id() -> None:
    from silent_cascade.memory.eviction import append_with_stress_eviction

    base = _full_memory()
    confidence = _replace_rank(base, 7, confidence=0.1, observed_at=60.0)
    confidence = _replace_rank(confidence, 8, confidence=0.2, observed_at=0.0)
    assert (
        append_with_stress_eviction(
            confidence, _safe_event(64), protected_ids=frozenset()
        ).evicted_record_id
        == 7
    )

    recency = _replace_rank(base, 7, confidence=0.1, observed_at=2.0)
    recency = _replace_rank(recency, 8, confidence=0.1, observed_at=1.0)
    assert (
        append_with_stress_eviction(
            recency, _safe_event(64), protected_ids=frozenset()
        ).evicted_record_id
        == 8
    )

    record_id = _replace_rank(base, 7, confidence=0.1, observed_at=1.0)
    record_id = _replace_rank(record_id, 8, confidence=0.1, observed_at=1.0)
    assert (
        append_with_stress_eviction(
            record_id, _safe_event(64), protected_ids=frozenset()
        ).evicted_record_id
        == 7
    )


def test_eviction_protects_active_and_support_ids_and_rebuilds_victim_slot() -> None:
    from silent_cascade.memory.eviction import append_with_stress_eviction
    from silent_cascade.memory.tensor_store import pack_memory

    memory = _replace_rank(_full_memory(), 0, confidence=0.0, observed_at=0.0)
    memory = _replace_rank(memory, 1, confidence=0.1, observed_at=1.0)
    before = pack_memory((memory,), (0.0,), device="cpu")
    result = append_with_stress_eviction(
        memory, _safe_event(64, observed_at=64.0), protected_ids=frozenset({0})
    )
    after = pack_memory((result.memory,), (0.0,), device="cpu")

    assert result.evicted_record_id == 1
    assert result.memory.records[1].record.record_id == 64
    assert before.record_ids[0][1] == 1
    assert after.record_ids[0][1] == 64
    assert result.memory.lookup(0).record.record_id == 0
    with pytest.raises(DynamicsError, match="missing"):
        result.memory.lookup(1)


def test_all_protected_duplicate_and_invalid_inputs_fail_without_mutation() -> None:
    from silent_cascade.memory.eviction import append_with_stress_eviction

    memory = _full_memory()
    before = memory
    with pytest.raises(DynamicsError, match="No unprotected"):
        append_with_stress_eviction(memory, _safe_event(64), protected_ids=frozenset(range(64)))
    with pytest.raises(DynamicsError, match="duplicate"):
        append_with_stress_eviction(memory, _safe_event(1), protected_ids=frozenset())
    with pytest.raises(DynamicsError, match="protected"):
        append_with_stress_eviction(memory, _safe_event(64), protected_ids=frozenset({99}))
    assert memory == before


def test_full_protection_is_validated_before_the_incoming_event() -> None:
    from silent_cascade.memory.eviction import append_with_stress_eviction

    memory = _full_memory()
    with pytest.raises(DynamicsError, match="No unprotected"):
        append_with_stress_eviction(memory, _safe_event(1), protected_ids=frozenset(range(64)))


def test_malformed_existing_record_fails_before_victim_selection() -> None:
    from silent_cascade.memory.eviction import append_with_stress_eviction

    record = MemoryRecord(
        1, RecordKind.SAFE, 1, None, None, None, 0.0, 1.0, Provenance.PERCEIVED, ()
    )
    object.__setattr__(record, "provenance", "perceived")
    memory = BoundedMemory(64, (RuntimeMemoryRecord(record),))
    object.__setattr__(memory, "records", memory.records * 64)

    with pytest.raises((TypeError, DynamicsError, ValueError), match=r"provenance|duplicate"):
        append_with_stress_eviction(memory, _safe_event(2), protected_ids=frozenset())
