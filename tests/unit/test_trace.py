"""Causal records retain provenance and exclude polls, tensors and private labels."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from test_jumps import internal, runtime

from silent_cascade.errors import DynamicsError, TimeOrderError
from silent_cascade.eventflow.flow import advance_to
from silent_cascade.eventflow.jumps import (
    ComposeDecision,
    ComposeRole,
    apply_act,
    apply_compose,
    apply_fact,
    begin_post_jump_segment,
)
from silent_cascade.eventflow.scheduling import TieCandidate, TieResolution
from silent_cascade.eventflow.state import ComputeCounters
from silent_cascade.logging.trace import (
    CausalTrace,
    SegmentSummary,
    TraceRecorder,
    sanitized_crash_events,
    tensor_sha256,
)
from silent_cascade.schemas import (
    ExternalEvent,
    ExternalEventKind,
    InternalEventKind,
    LinkFact,
    Mode,
)


def recorded_fact(recorder: TraceRecorder, state, event_id: int = 4, time: float = 2.0):
    before = advance_to(state, time)
    event = ExternalEvent(event_id, time, ExternalEventKind.FACT, LinkFact(3, 4))
    post = apply_fact(before, event)
    after = begin_post_jump_segment(post, before.segment.parameters, time=time)
    summary = recorder.record(event, before, after, selected_record_id=1, selected_rank=2)
    return summary, after


def test_record_has_causal_delta_modes_metadata_and_detached_state_hash() -> None:
    recorder = TraceRecorder()
    state = runtime()
    state = replace(state, core=replace(state.core, last_event_time=1.0))
    paused = advance_to(state, 1.5)
    record, after = recorded_fact(recorder, paused)
    assert record.event_id == 4 and record.parent_event_id == 3
    assert record.timestamp == 2.0 and record.delta == 1.0
    assert record.pre_mode is record.post_mode is Mode.OBSERVING
    assert (record.selected_record_id, record.selected_rank) == (1, 2)
    assert record.support_before == record.support_after == ()
    assert record.hypothesis_before is record.hypothesis_after is None
    assert record.actions == ()
    assert record.counter_delta.jump_applications == 1
    assert record.prediction_snapshot_sha256 == "a" * 64
    assert record.segment == SegmentSummary.from_segment(after.segment)
    assert record.state_norm > 0 and len(record.state_sha256) == 64
    trace = recorder.snapshot()
    original = trace.sha256
    after.core.continuous.z_fast.zero_()
    assert trace.sha256 == original
    with pytest.raises(FrozenInstanceError):
        record.delta = 0.0
    assert len(trace.events) == 1


def test_trace_rejects_duplicate_wrong_parent_and_reversed_events() -> None:
    recorder = TraceRecorder()
    first, state = recorded_fact(recorder, runtime())
    for bad in (first, replace(first, event_id=5, parent_event_id=99)):
        with pytest.raises(DynamicsError):
            recorder.append(bad)
    with pytest.raises(TimeOrderError):
        recorder.append(replace(first, event_id=5, parent_event_id=4, timestamp=1.0))
    second, _ = recorded_fact(recorder, state, 5, 2.0)
    assert second.delta == 0.0
    assert [item.event_id for item in recorder.snapshot().events] == [4, 5]


@pytest.mark.parametrize("terminal_kind", [ExternalEventKind.OUTCOME, ExternalEventKind.END])
def test_online_terminal_and_bounded_crash_projection(terminal_kind) -> None:
    recorder = TraceRecorder()
    state = runtime()
    for event_id in range(4, 27):
        _, state = recorded_fact(recorder, state, event_id, float(event_id))
    event = ExternalEvent(27, 27.0, terminal_kind, None)
    before = advance_to(state, 27.0)
    post = replace(before.core, mode=Mode.TERMINAL, last_event_id=27, last_event_time=27.0)
    recorder.record(event, before, post)
    trace = recorder.snapshot()
    payload = trace.to_payload()
    assert payload["events"][-1]["kind"] == "terminal"
    assert payload["events"][-1]["payload"] is None
    assert payload["events"][-1]["segment"] is None
    encoded = json.dumps(payload)
    for secret in ("outcome", '"end"', "variant", "action_window", '"path"'):
        assert secret not in encoded
    crashed = sanitized_crash_events(trace)
    assert len(crashed) == 20
    assert crashed[0]["event_id"] == 8
    assert crashed[-1]["kind"] == "terminal"


def test_tensor_hash_frames_dtype_shape_and_contiguous_values() -> None:
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert tensor_sha256(tensor) == tensor_sha256(tensor.T.contiguous().T)
    assert tensor_sha256(tensor) != tensor_sha256(tensor.reshape(3, 2))
    assert tensor_sha256(tensor) != tensor_sha256(tensor.view(torch.int32))
    tensor[0, 0] = -0.0
    assert tensor_sha256(tensor) != tensor_sha256(
        torch.arange(6, dtype=torch.float32).reshape(2, 3)
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="native MPS unavailable")
def test_tensor_hash_matches_cpu_for_native_mps_float32() -> None:
    value = torch.arange(16, dtype=torch.float32)
    assert tensor_sha256(value) == tensor_sha256(value.to("mps"))


def test_canonical_hash_uses_sorted_json_and_is_hash_seed_independent() -> None:
    recorder = TraceRecorder()
    recorded_fact(recorder, runtime())
    trace = recorder.snapshot()
    payload = trace.to_payload()
    expected = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
    assert trace.sha256 == expected
    assert CausalTrace(trace.events).sha256 == expected
    script = (
        "import sys; sys.path.insert(0, 'tests/unit'); from test_trace import *; "
        "r=TraceRecorder(); recorded_fact(r,runtime()); print(r.snapshot().sha256)"
    )
    for seed in ("1", "321"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
            capture_output=True,
            check=True,
        )
        assert result.stdout.strip() == expected


@pytest.mark.parametrize("kind", ["outcome", "end", "checkpoint", "pause"])
def test_summary_rejects_private_or_noncausal_kind(kind: str) -> None:
    record, _ = recorded_fact(TraceRecorder(), runtime())
    with pytest.raises(DynamicsError):
        replace(record, kind=kind)


def test_summary_rejects_mutable_nested_values_before_they_enter_trace() -> None:
    record, _ = recorded_fact(TraceRecorder(), runtime())
    for changes in ({"support_after": [1]}, {"payload": {"secret": "label"}}, {"actions": []}):
        with pytest.raises(DynamicsError):
            replace(record, **changes)


def test_counter_deltas_include_initialization_and_resume_from_prior_post_state() -> None:
    state = runtime()
    state = replace(
        state,
        core=replace(state.core, counters=ComputeCounters(controller_calls=1, guard_predictions=1)),
    )
    recorder = TraceRecorder()
    first, state = recorded_fact(recorder, state)
    assert first.counter_delta.controller_calls == 1
    assert first.counter_delta.guard_predictions == 1
    assert first.counter_delta.flow_evaluations == 1
    resumed = TraceRecorder(recorder.snapshot())
    state = replace(
        state,
        core=replace(
            state.core,
            counters=replace(state.core.counters, controller_calls=2, guard_predictions=2),
        ),
    )
    second, _ = recorded_fact(resumed, state, 5, 3.0)
    assert second.counter_delta.controller_calls == 1
    assert second.counter_delta.guard_predictions == 1
    assert second.counter_delta.flow_evaluations == 1
    assert second.counters_after.jump_applications == 2


def test_pause_cursor_and_checkpoint_counter_do_not_change_causal_trace_hash() -> None:
    initial = runtime()
    direct = TraceRecorder()
    recorded_fact(direct, initial)
    paused = advance_to(initial, 1.5)
    paused = replace(
        paused, core=replace(paused.core, counters=ComputeCounters(checkpoint_flow_evaluations=1))
    )
    recorder = TraceRecorder()
    recorded_fact(recorder, paused)
    assert direct.snapshot().sha256 == recorder.snapshot().sha256
    assert recorder.snapshot().events[0].counter_delta.checkpoint_flow_evaluations == 0


def test_real_composition_and_action_trace_preserves_predicted_metadata() -> None:
    state = runtime(Mode.HAVE_MEMORY)
    event = internal(InternalEventKind.COMPOSE)
    decision = ComposeDecision(ComposeRole.HAZARD, hazard_type=3, deadline=10.0)
    post = apply_compose(state, event, decision)
    after = begin_post_jump_segment(post, state.segment.parameters, time=1.0)
    recorder = TraceRecorder()
    composition = recorder.record(event, state, after, selected_record_id=1, selected_rank=4)
    assert composition.support_before == () and composition.support_after == (1,)
    assert composition.hypothesis_before is None
    assert composition.hypothesis_after.hazard_type == 3
    assert composition.selected_rank == 4
    before_act = advance_to(after, 2.0)
    act = replace(
        internal(InternalEventKind.ACT, timestamp=2.0),
        event_id=event.event_id + 1,
        parent_event_id=event.event_id,
    )
    acted = apply_act(before_act, act, hazard_type=0)
    after_act = begin_post_jump_segment(acted, after.segment.parameters, time=2.0)
    action = recorder.record(act, before_act, after_act)
    assert action.pre_mode is Mode.HOLDING_HAZARD and action.post_mode is Mode.QUIESCENT
    assert action.actions[0].hazard_type == 0
    assert action.actions[0].caused_by_event_id == act.event_id
    assert action.hypothesis_after.hazard_type == 3
    assert action.delta == 1.0


def test_trace_rejects_same_time_priority_reversal() -> None:
    recorder = TraceRecorder()
    record, _ = recorded_fact(recorder, runtime())
    with pytest.raises(DynamicsError):
        recorder.append(
            replace(
                record,
                event_id=5,
                parent_event_id=4,
                kind="terminal",
                payload=None,
                segment=None,
                delta=0.0,
                counter_delta=ComputeCounters(),
            )
        )


def test_trace_records_full_near_tie_and_rejects_wrong_winner() -> None:
    state = advance_to(runtime(), 2.0)
    event = ExternalEvent(4, 2.0, ExternalEventKind.FACT, LinkFact(3, 4))
    post = apply_fact(state, event)
    after = begin_post_jump_segment(post, state.segment.parameters, time=2.0)
    tie = TieResolution(
        (TieCandidate(4, 2.0, "external"), TieCandidate(1 << 62, 2.0 - 0.5e-9, "act")), 4
    )
    record = TraceRecorder().record(event, state, after, tie=tie)
    assert record.tie_id == tie.tie_id
    assert record.to_payload()["tie"]["candidates"][1]["timestamp"] == 2.0 - 0.5e-9
    with pytest.raises(DynamicsError):
        TraceRecorder().record(event, state, after, tie=replace(tie, winner_event_id=1 << 62))


def test_summary_rejects_mutable_hypothesis_or_tie() -> None:
    record, _ = recorded_fact(TraceRecorder(), runtime())
    with pytest.raises(DynamicsError):
        replace(record, hypothesis_after={"hidden": True})
    with pytest.raises(DynamicsError):
        replace(record, tie={"hidden": True})


def test_tie_candidates_cannot_leak_private_terminal_kind_or_mutable_container() -> None:
    with pytest.raises(DynamicsError):
        TieCandidate(1, 2.0, "outcome")
    with pytest.raises(DynamicsError):
        TieResolution([TieCandidate(1, 2.0, "terminal")], 1)


def test_record_rejects_a_post_event_state_materialized_beyond_the_event() -> None:
    before = advance_to(runtime(), 2.0)
    event = ExternalEvent(4, 2.0, ExternalEventKind.FACT, LinkFact(3, 4))
    post = apply_fact(before, event)
    after = begin_post_jump_segment(post, before.segment.parameters, time=2.0)
    with pytest.raises(TimeOrderError):
        TraceRecorder().record(event, before, advance_to(after, 3.0))
