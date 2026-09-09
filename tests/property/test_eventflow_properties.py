"""State-machine properties for the Phase 2 event-flow runtime."""

from dataclasses import replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from silent_cascade.errors import DynamicsError, ProvenanceError
from silent_cascade.memory import (
    BoundedMemory,
    append_perceived_fact,
    mark_consumed,
    mark_recalled,
    require_support_ledger,
)
from silent_cascade.schemas import ExternalEvent, ExternalEventKind, SafeFact


@given(st.lists(st.sampled_from(("append", "recall", "consume")), min_size=1, max_size=96))
def test_runtime_memory_transitions_preserve_capacity_unique_ids_and_monotone_metadata(
    operations: list[str],
) -> None:
    """Removing capacity/ID checks or reverting metadata makes this fail."""
    memory = BoundedMemory(64)
    next_id = 0
    consumed_ids: set[int] = set()
    refractory_by_id: dict[int, float] = {}

    for operation in operations:
        if operation == "append":
            event = ExternalEvent(next_id, float(next_id), ExternalEventKind.FACT, SafeFact(0))
            if next_id == 64:
                try:
                    append_perceived_fact(memory, event)
                except DynamicsError:
                    pass
                else:
                    raise AssertionError("the 65th record must fail loudly")
            else:
                memory = append_perceived_fact(memory, event)
                next_id += 1
        elif memory.records:
            record_id = memory.records[0].record.record_id
            if operation == "recall":
                requested_until = float(next_id + len(memory.records))
                previous_until = refractory_by_id.get(record_id, 0.0)
                memory = mark_recalled(memory, record_id, refractory_until=requested_until)
                refractory_by_id[record_id] = max(previous_until, requested_until)
            else:
                memory = mark_consumed(memory, record_id)
                consumed_ids.add(record_id)

        assert len(memory.records) <= 64
        assert len({slot.record.record_id for slot in memory.records}) == len(memory.records)
        for slot in memory.records:
            assert slot.record.support_ids == ()
            assert slot.refractory_until >= refractory_by_id.get(slot.record.record_id, 0.0)
            assert slot.consumed is (slot.record.record_id in consumed_ids)


@given(
    initial_record_count=st.integers(min_value=1, max_value=16),
    operations=st.lists(st.sampled_from(("append", "recall", "consume")), max_size=48),
    data=st.data(),
)
def test_runtime_memory_support_ledgers_only_name_unique_existing_valid_records(
    initial_record_count: int,
    operations: list[str],
    data: st.DataObject,
) -> None:
    """Removing support validation admits duplicate, missing, or invalid derivations."""
    memory = BoundedMemory(64)
    next_id = 0
    for _ in range(initial_record_count):
        memory = append_perceived_fact(
            memory,
            ExternalEvent(next_id, float(next_id), ExternalEventKind.FACT, SafeFact(0)),
        )
        next_id += 1

    for operation in operations:
        record_ids = tuple(slot.record.record_id for slot in memory.records)
        if operation == "append" and next_id < 64:
            memory = append_perceived_fact(
                memory,
                ExternalEvent(next_id, float(next_id), ExternalEventKind.FACT, SafeFact(0)),
            )
            next_id += 1
        else:
            selected_id = data.draw(st.sampled_from(record_ids), label="selected_record_id")
            if operation == "recall":
                memory = mark_recalled(memory, selected_id, refractory_until=float(next_id))
            else:
                memory = mark_consumed(memory, selected_id)

    record_by_id = {slot.record.record_id: slot for slot in memory.records}
    support_ids = tuple(
        data.draw(
            st.lists(
                st.sampled_from(tuple(record_by_id)),
                min_size=1,
                max_size=min(4, len(record_by_id)),
                unique=True,
            ),
            label="derived_support_ids",
        )
    )

    assert len(support_ids) == len(set(support_ids))
    assert set(support_ids).issubset(record_by_id)
    assert all(record_by_id[support_id].valid for support_id in support_ids)
    assert require_support_ledger(memory, support_ids) is None

    invalid_support_id = support_ids[0]
    invalid_memory = replace(
        memory,
        records=tuple(
            replace(slot, valid=False) if slot.record.record_id == invalid_support_id else slot
            for slot in memory.records
        ),
    )
    with pytest.raises(ProvenanceError, match="invalid"):
        require_support_ledger(invalid_memory, support_ids)
    with pytest.raises(ProvenanceError, match="unknown"):
        require_support_ledger(memory, (999,))
    with pytest.raises(ProvenanceError, match="unique"):
        require_support_ledger(memory, (support_ids[0], support_ids[0]))
