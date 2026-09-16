from dataclasses import replace

import pytest

from .test_neural_agent import prepare, state_at


def test_queries_are_pure_and_dormancy_advances_directly(runtime_case, controlled_model):
    controlled_model.dormant = True
    agent, state = prepare(runtime_case, controlled_model)
    calls = len(controlled_model.calls)
    for time in (5.0, 6.0, 15.0, 28.0):
        assert agent.next_internal_event(state_at(state, time)) is None
    assert len(controlled_model.calls) == calls
    assert state.core.executed_internal_events == 0


def test_external_terminal_preempts_learned_crossing(runtime_case, controlled_model):
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent

    controlled_model.guard_rate = 0.001
    result = runtime_case.run(
        NeuralEventFlowAgent(controlled_model, identity=runtime_case.identity, device="cpu")
    )
    assert result.actions == ()
    assert result.counters.jump_applications == 6  # 5 public + private terminal


def test_refractory_and_engine_gap_clamp_are_preserved(runtime_case, controlled_model):
    from silent_cascade.eventflow.engine import EventEngine

    agent, state = prepare(runtime_case, controlled_model)
    refractory = replace(
        state, core=replace(state.core, same_kind_refractory_until=(5.4, 0.0, 0.0))
    )
    assert agent.next_internal_event(refractory).timestamp == 5.4
    engine = EventEngine(runtime_case.engine_config)
    # Retain a mathematically tiny crossing in the current segment; engine alone clamps it.
    targets = state.segment.parameters.guard_targets.clone()
    targets[0] = 1.999
    origin = replace(
        state.segment.origin, guard_accumulators=state.segment.origin.guard_accumulators + 0.999999
    )
    from silent_cascade.eventflow.guards import allowed_mode_mask, prediction_snapshot_sha256
    from silent_cascade.eventflow.state import AnalyticSegment, RuntimeState

    parameters = replace(state.segment.parameters, guard_targets=targets)
    core = replace(state.core, continuous=origin)
    digest = prediction_snapshot_sha256(
        origin,
        parameters,
        started_at=state.time,
        allowed_mode_mask=allowed_mode_mask(core.mode),
        parent_event_id=50,
    )
    # Synthetic materialized segment isolates the scheduler boundary; learned
    # resets start at zero and their legal rates ordinarily exceed this gap.
    tiny = RuntimeState(
        core, AnalyticSegment(state.time, origin, parameters, 50, digest), state.time
    )
    event = agent.next_internal_event(tiny)
    assert event.timestamp - tiny.time < 0.001
    # The public agent never installs the scheduler's clamp itself.
    assert tiny.core.consecutive_gap_clamps == 0
    from silent_cascade.eventflow.scheduling import normalize_internal_gap

    gap = engine.config.guards.minimum_internal_gap
    clamped, was_clamped = normalize_internal_gap(event, origin_time=tiny.time, minimum_gap=gap)
    assert was_clamped
    assert clamped.timestamp == pytest.approx(tiny.time + gap)


def test_neural_callback_preserves_internal_cap(runtime_case, controlled_model):
    from silent_cascade.models.errors import NeuralError

    from .test_neural_agent import step

    agent, state = prepare(runtime_case, controlled_model)
    state = replace(state, core=replace(state.core, executed_internal_events=64))
    with pytest.raises(NeuralError) as caught:
        step(agent, state)
    assert "limit" in str(caught.value.__cause__)


def test_engine_private_terminal_wins_exact_learned_tie(runtime_case, controlled_model):
    agent, state = prepare(runtime_case, controlled_model)
    delay = agent.next_internal_event(state).timestamp - 5.0
    result = runtime_case.run(agent, negative=True, terminal_delay=delay)
    assert result.actions == ()
    assert result.counters.jump_applications == 6
