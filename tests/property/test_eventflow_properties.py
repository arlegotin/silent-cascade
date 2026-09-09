"""State-machine properties for the Phase 2 event-flow runtime."""

import math
from dataclasses import replace

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from silent_cascade.errors import DynamicsError, ProvenanceError
from silent_cascade.eventflow.flow import advance_to, start_segment, state_at
from silent_cascade.eventflow.guards import crossing_offset_host
from silent_cascade.eventflow.state import (
    AnalyticSegment,
    ContinuousChannels,
    RuntimeCore,
    RuntimeState,
    SegmentParameters,
    make_initial_continuous_state,
)
from silent_cascade.memory import (
    BoundedMemory,
    append_perceived_fact,
    mark_consumed,
    mark_recalled,
    require_support_ledger,
)
from silent_cascade.schemas import ExternalEvent, ExternalEventKind, SafeFact


def runtime() -> RuntimeState:
    dimensions = {
        "z_fast": 256,
        "z_slow": 64,
        "drives": 8,
        "focus_key": 64,
        "hypothesis_latent": 64,
    }
    parameters = SegmentParameters(
        flow_targets=ContinuousChannels(
            **{name: torch.full((size,), 0.75) for name, size in dimensions.items()}
        ),
        flow_rates=ContinuousChannels(
            **{name: torch.full((size,), 0.7) for name, size in dimensions.items()}
        ),
        guard_targets=torch.tensor([0.5, 1.25, 1.75]),
        guard_rates=torch.tensor([0.1, 2.0, 500.0]),
    )
    return start_segment(
        RuntimeCore(make_initial_continuous_state()),
        parameters,
        time=0.0,
        parent_event_id=0,
        prediction_snapshot_sha256="a" * 64,
    )


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


@settings(max_examples=30, deadline=None)
@given(
    first=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    second=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    origin=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    target=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    rate=st.floats(min_value=1e-5, max_value=20.0, allow_nan=False, allow_infinity=False),
)
def test_continuous_semigroup_bounds_and_nonmutation(
    first: float,
    second: float,
    origin: float,
    target: float,
    rate: float,
) -> None:
    """Reanchoring, aliasing, bad signs, or unbounded flow break these invariants."""
    initial = runtime()
    state = replace(initial.core.continuous, z_fast=torch.full((256,), origin))
    param = replace(
        initial.segment.parameters,
        flow_targets=replace(
            initial.segment.parameters.flow_targets, z_fast=torch.full((256,), target)
        ),
        flow_rates=replace(initial.segment.parameters.flow_rates, z_fast=torch.full((256,), rate)),
    )
    initial = RuntimeState(
        replace(initial.core, continuous=state),
        AnalyticSegment(0.0, state, param, 0, "a" * 64),
        0.0,
    )
    before = initial.core.continuous.z_fast.clone()
    direct = advance_to(initial, first + second)
    middle = advance_to(initial, first)
    anchored = advance_to(middle, first + second)
    split_origin = RuntimeState(
        middle.core, AnalyticSegment(first, middle.core.continuous, param, 0, "a" * 64), first
    )
    split = advance_to(split_origin, first + second)
    result = direct.core.continuous.z_fast
    assert torch.equal(result, anchored.core.continuous.z_fast)
    torch.testing.assert_close(result, split.core.continuous.z_fast, rtol=5e-7, atol=2e-7)
    assert torch.isfinite(result).all()
    assert (result >= -1.0).all() and (result <= 1.0).all()
    assert initial.time <= middle.time <= anchored.time
    assert torch.equal(initial.core.continuous.z_fast, before)
    snapshot = state_at(middle, first + second)
    assert snapshot.z_fast.data_ptr() != middle.core.continuous.z_fast.data_ptr()


@settings(max_examples=20, deadline=None)
@given(dt=st.floats(min_value=1e6, max_value=1e12, allow_nan=False, allow_infinity=False))
def test_huge_time_flow_stays_finite_and_converges(dt: float) -> None:
    initial = runtime()
    advanced = advance_to(initial, dt)
    assert torch.isfinite(advanced.core.continuous.z_fast).all()
    torch.testing.assert_close(
        advanced.core.continuous.z_fast, initial.segment.parameters.flow_targets.z_fast
    )
    torch.testing.assert_close(
        advanced.core.continuous.guard_accumulators, initial.segment.parameters.guard_targets
    )


@settings(max_examples=30, deadline=None)
@given(
    accumulator=st.floats(min_value=0.0, max_value=0.999999, allow_nan=False, allow_infinity=False),
    asymptote=st.floats(
        min_value=1.000001, max_value=1.999999, allow_nan=False, allow_infinity=False
    ),
    rate=st.floats(min_value=1.0e-5, max_value=500.0, allow_nan=False, allow_infinity=False),
)
def test_active_guard_host_crossing_is_finite_and_positive(
    accumulator: float, asymptote: float, rate: float
) -> None:
    """Changing the active formula or accepting dormant values violates this scalar invariant."""
    crossing = crossing_offset_host(accumulator, asymptote, rate)
    assert crossing is not None
    assert math.isfinite(crossing)
    assert crossing > 0.0
