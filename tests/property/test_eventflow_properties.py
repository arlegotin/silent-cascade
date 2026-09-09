"""State-machine properties for the Phase 2 event-flow runtime."""

from hypothesis import given
from hypothesis import strategies as st

from silent_cascade.errors import DynamicsError
from silent_cascade.memory import BoundedMemory, append_perceived_fact, mark_consumed, mark_recalled
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
