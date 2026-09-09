"""Behavioral contracts for bounded public runtime memory."""

from dataclasses import replace

import pytest

from silent_cascade.errors import DynamicsError, ProvenanceError
from silent_cascade.memory import (
    BoundedMemory,
    RuntimeMemoryRecord,
    append_perceived_fact,
    legal_records,
    mark_consumed,
    mark_recalled,
    require_provenance_preserved,
    require_support_ledger,
)
from silent_cascade.schemas import (
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    Hypothesis,
    LinkFact,
    MemoryRecord,
    Provenance,
    RecordKind,
    SafeFact,
)


def _fact(event_id: int, payload: LinkFact | HazardFact | SafeFact) -> ExternalEvent:
    return ExternalEvent(event_id, float(event_id), ExternalEventKind.FACT, payload)


def _record(
    record_id: int,
    *,
    kind: RecordKind = RecordKind.LINK,
    subject_id: int = 1,
    provenance: Provenance = Provenance.PERCEIVED,
    support_ids: tuple[int, ...] = (),
) -> MemoryRecord:
    if kind is RecordKind.LINK:
        return MemoryRecord(
            record_id,
            kind,
            subject_id,
            2,
            None,
            None,
            float(record_id),
            1.0,
            provenance,
            support_ids,
        )
    if kind is RecordKind.HAZARD:
        return MemoryRecord(
            record_id,
            kind,
            subject_id,
            None,
            0,
            1.0,
            float(record_id),
            1.0,
            provenance,
            support_ids,
        )
    return MemoryRecord(
        record_id,
        kind,
        subject_id,
        None,
        None,
        None,
        float(record_id),
        1.0,
        provenance,
        support_ids,
    )


def test_append_converts_public_facts_in_presentation_order() -> None:
    memory = BoundedMemory(64)

    memory = append_perceived_fact(memory, _fact(11, LinkFact(1, 2)))
    memory = append_perceived_fact(memory, _fact(3, HazardFact(2, 0, 1.5)))

    assert [slot.record.record_id for slot in memory.records] == [11, 3]
    assert [slot.record.provenance for slot in memory.records] == [
        Provenance.PERCEIVED,
        Provenance.PERCEIVED,
    ]
    assert memory.records[1].record.delay == 1.5


def test_memory_requires_exact_primary_capacity_and_rejects_the_65th_record() -> None:
    with pytest.raises(DynamicsError, match="capacity"):
        BoundedMemory(63)

    memory = BoundedMemory(64)
    for record_id in range(64):
        memory = append_perceived_fact(memory, _fact(record_id, SafeFact(record_id)))

    with pytest.raises(DynamicsError, match="capacity"):
        append_perceived_fact(memory, _fact(64, SafeFact(0)))


def test_duplicate_record_ids_are_rejected_without_eviction() -> None:
    memory = append_perceived_fact(BoundedMemory(64), _fact(7, LinkFact(1, 2)))

    with pytest.raises(DynamicsError, match="duplicate"):
        append_perceived_fact(memory, _fact(7, SafeFact(1)))


def test_legal_records_masks_invalid_consumed_subject_kind_and_refractory_slots() -> None:
    eligible = RuntimeMemoryRecord(_record(1, subject_id=3))
    invalid = RuntimeMemoryRecord(_record(2, subject_id=3), valid=False)
    consumed = RuntimeMemoryRecord(_record(3, subject_id=3), consumed=True)
    wrong_subject = RuntimeMemoryRecord(_record(4, subject_id=4))
    wrong_kind = RuntimeMemoryRecord(_record(5, kind=RecordKind.HAZARD, subject_id=3))
    refractory = RuntimeMemoryRecord(_record(6, subject_id=3), refractory_until=2.0)
    memory = BoundedMemory(64, (eligible, invalid, consumed, wrong_subject, wrong_kind, refractory))

    records = legal_records(
        memory,
        at_time=1.0,
        subject_id=3,
        kinds=frozenset({RecordKind.LINK}),
    )

    assert records == (eligible,)
    assert (
        memory.legal_records(
            at_time=1.0,
            subject_id=3,
            kinds=frozenset({RecordKind.LINK}),
        )
        == records
    )


def test_recall_only_extends_a_record_refractory_window() -> None:
    original = BoundedMemory(64, (RuntimeMemoryRecord(_record(1), refractory_until=3.0),))

    not_shortened = mark_recalled(original, 1, refractory_until=2.0)
    extended = mark_recalled(not_shortened, 1, refractory_until=4.0)

    assert original.records[0].refractory_until == 3.0
    assert not_shortened.records[0].refractory_until == 3.0
    assert extended.records[0].refractory_until == 4.0


def test_consumption_is_monotone_and_masks_the_record() -> None:
    memory = BoundedMemory(64, (RuntimeMemoryRecord(_record(1)),))

    consumed = mark_consumed(memory, 1)
    consumed_again = mark_consumed(consumed, 1)

    assert memory.records[0].consumed is False
    assert consumed.records[0].consumed is True
    assert consumed_again.records[0].consumed is True
    assert legal_records(consumed_again, at_time=0.0) == ()


def test_lookup_of_a_missing_record_id_fails_loudly() -> None:
    memory = BoundedMemory(64)

    with pytest.raises(DynamicsError, match="missing"):
        memory.lookup(99)


def test_runtime_wrappers_and_perceived_records_are_immutable() -> None:
    record = _record(1)
    slot = RuntimeMemoryRecord(record)

    with pytest.raises(AttributeError):
        slot.valid = False  # type: ignore[misc]
    with pytest.raises(AttributeError):
        record.provenance = Provenance.INFERRED  # type: ignore[misc]


def test_support_ledger_requires_unique_existing_valid_support_ids() -> None:
    memory = BoundedMemory(
        64,
        (
            RuntimeMemoryRecord(_record(1)),
            RuntimeMemoryRecord(_record(2), valid=False),
        ),
    )
    hypothesis = Hypothesis(0, 3.0, 1.0, Provenance.INFERRED, (1,), False)

    assert require_support_ledger(memory, hypothesis.support_ids) is None
    with pytest.raises(ProvenanceError, match="unknown"):
        require_support_ledger(memory, (9,))
    with pytest.raises(ProvenanceError, match="invalid"):
        require_support_ledger(memory, (2,))
    with pytest.raises(ProvenanceError, match="unique"):
        require_support_ledger(memory, (1, 1))


def test_provenance_cannot_escalate_to_perceived_or_change() -> None:
    inferred = _record(1, provenance=Provenance.INFERRED, support_ids=(2,))
    simulated = _record(2, provenance=Provenance.SIMULATED, support_ids=(1,))
    perceived = _record(1, provenance=Provenance.PERCEIVED)

    with pytest.raises(ProvenanceError, match="provenance"):
        require_provenance_preserved(inferred, perceived)
    with pytest.raises(ProvenanceError, match="provenance"):
        require_provenance_preserved(simulated, replace(simulated, provenance=Provenance.PERCEIVED))
