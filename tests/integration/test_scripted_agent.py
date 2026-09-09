"""Public rule-system integration: non-neural runtime engineering evidence."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from silent_cascade.config import resolve_config
from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.episode import EpisodeVariant, scale_episode_time
from silent_cascade.env.generator import IndependentEpisodeRequest, generate_independent_episode
from silent_cascade.errors import DynamicsError
from silent_cascade.eventflow.config import Phase2Config
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.flow import advance_to, state_at
from silent_cascade.eventflow.guards import (
    allowed_mode_mask,
    next_crossings,
    prediction_snapshot_sha256,
)
from silent_cascade.eventflow.scripted import ScriptedEventFlowAgent
from silent_cascade.rng import AllocationLabelKey, allocate_independent_variants
from silent_cascade.schemas import (
    ActivationPayload,
    AgentInit,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    InternalEventKind,
    LinkFact,
    Mode,
)


@pytest.fixture(scope="module")
def config():
    return resolve_config(
        Phase2Config,
        [
            Path("configs/base.yaml"),
            Path("configs/data/primary.yaml"),
            Path("configs/model/event_flow.yaml"),
        ],
    ).config


def generated(config, length, variant, suite=SuiteName.IID_PRIMARY):
    variants = allocate_independent_variants(
        AllocationLabelKey("ofd-v1", SplitNamespace.DEBUG, suite, 41, length, 0)
    )
    member = variants.index(variant)
    request = IndependentEpisodeRequest(
        SplitNamespace.DEBUG, suite, 41, member, length, variant, 0, member
    )
    return generate_independent_episode(config, request, public_id_seed=91)


@pytest.mark.parametrize("length", [2, 3, 4])
@pytest.mark.parametrize("variant", list(EpisodeVariant))
@pytest.mark.parametrize(
    "suite", [SuiteName.IID_PRIMARY, SuiteName.OOD_SHORT_DELAY, SuiteName.OOD_LONG_DELAY]
)
def test_generated_episodes_follow_public_chains_and_guard_crossings(
    config, length, variant, suite
):
    bundle = generated(config, length, variant, suite)
    agent = ScriptedEventFlowAgent()
    engine = EventEngine(config.event_flow)
    session = engine.start_episode(bundle, agent)
    activation_time = bundle.public.events[-1].timestamp
    if suite is SuiteName.IID_PRIMARY:
        assert bundle.truth.recipe.distractor_link_count > 0
    while session.state.core.mode is not Mode.TERMINAL:
        proposal = engine.next_internal_event(session, agent)
        before = session.state
        if before.time < activation_time:
            assert proposal is None
        if proposal is not None:
            (crossing,) = next_crossings(before)
            assert proposal.timestamp == crossing.timestamp
            assert proposal.parent_event_id == before.segment.parent_event_id
            flowed = state_at(before, proposal.timestamp)
            assert flowed.guard_accumulators[proposal.guard_index].item() == pytest.approx(
                1.0, abs=2e-6
            )
            if proposal.kind in {InternalEventKind.RECALL, InternalEventKind.COMPOSE}:
                assert 0.05 - 1e-8 <= proposal.timestamp - before.time <= 0.068 + 1e-8
        engine.step(session, agent)
    trace = session.trace.snapshot().events
    internals = [row for row in trace if row.kind in {"recall", "compose", "act"}]
    grammar = ["recall", "compose"] * length
    if variant is not EpisodeVariant.DISCONNECTED_NEGATIVE:
        grammar += ["recall", "compose"]
    if variant is EpisodeVariant.POSITIVE:
        grammar += ["act"]
    assert [row.kind for row in internals] == grammar
    gaps = [round(row.delta, 6) for row in internals if row.kind != "act"]
    assert len(set(gaps)) > 1
    focus = bundle.public.events[-1].payload.start_node
    records = session.state.core.memory
    for row in internals:
        if row.kind == "recall":
            record = records.lookup(row.selected_record_id).record
            assert record.subject_id == focus
            assert row.selected_rank == 1
            if record.object_id is not None:
                focus = record.object_id
    assert session.terminal_score.timed_success
    if variant is EpisodeVariant.POSITIVE:
        (action,) = session.state.core.actions
        assert action.hazard_type == bundle.truth.relevant_hazard_type
        assert bundle.truth.action_window_start <= action.timestamp < bundle.truth.action_window_end
    else:
        assert session.state.core.actions == ()
        if variant is EpisodeVariant.DISCONNECTED_NEGATIVE:
            assert session.state.core.hypothesis is None
            assert internals[-1].post_mode is Mode.QUIESCENT
    assert session.state.core.counters.controller_calls == len(trace)
    assert session.state.core.counters.jump_applications == len(trace)
    assert agent.compute_counters().foundation_model_calls == 0
    assert session.state.core.counters.foundation_model_calls == 0


@pytest.mark.parametrize("length", [2, 3, 4])
@pytest.mark.parametrize("variant", list(EpisodeVariant))
@pytest.mark.parametrize(
    "suite,factor", [(SuiteName.CLOCK_SCALE_0_1X, 0.1), (SuiteName.CLOCK_SCALE_10X, 10.0)]
)
def test_clock_scaled_pairs_keep_outcomes_and_public_support(
    config, length, variant, suite, factor
):
    parent = generated(config, length, variant)
    child = scale_episode_time(parent, suite, "00000000-0000-4000-8000-000000000001")
    results = [
        EventEngine(config.event_flow).run_episode(item, ScriptedEventFlowAgent())
        for item in (parent, child)
    ]
    assert all(result.score.timed_success for result in results)
    assert [row.selected_record_id for row in results[0].trace.events] == [
        row.selected_record_id for row in results[1].trace.events
    ]
    if variant is EpisodeVariant.POSITIVE:
        assert results[1].actions[0].timestamp == pytest.approx(
            results[0].actions[0].timestamp * factor, rel=2e-7
        )


def public_state(agent, *, facts, start=0):
    state = agent.initialize(AgentInit("public-only", 64, 4, 0.0))
    for index, (record_id, payload) in enumerate(facts):
        time = (index + 1) * 0.1
        state = agent.on_external(
            advance_to(state, time), ExternalEvent(record_id, time, ExternalEventKind.FACT, payload)
        )
    return agent.on_external(
        advance_to(state, 1.0),
        ExternalEvent(100, 1.0, ExternalEventKind.ACTIVATE, ActivationPayload(start)),
    )


def execute_next(agent, state):
    event = agent.next_internal_event(state)
    assert event is not None
    after, actions = agent.on_internal(advance_to(state, event.timestamp), event)
    return after, actions


def test_initial_state_is_dormant_with_real_digest_and_one_controller_call():
    agent = ScriptedEventFlowAgent()
    init = AgentInit("public-only", 64, 4, 12.0)
    state = agent.initialize(init)
    assert state.time == state.segment.started_at == 12.0
    assert state.core.last_event_id is state.core.last_event_time is None
    assert agent.next_internal_event(state) is None
    assert bool((state.segment.parameters.guard_targets <= 0.90).all())
    assert not bool(torch.count_nonzero(state.core.continuous.guard_accumulators))
    assert state.segment.prediction_snapshot_sha256 == prediction_snapshot_sha256(
        state.core.continuous,
        state.segment.parameters,
        started_at=12.0,
        allowed_mode_mask=allowed_mode_mask(Mode.OBSERVING),
        parent_event_id=0,
    )
    assert agent.compute_counters().controller_calls == 1


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 0, 2)])
def test_confidence_then_record_id_selection_ignores_presentation_order(order):
    agent = ScriptedEventFlowAgent()
    facts = [(9, LinkFact(0, 9, 0.8)), (8, LinkFact(0, 8, 0.9)), (7, LinkFact(0, 7, 0.9))]
    state = public_state(agent, facts=[facts[index] for index in order])
    recalled, _ = execute_next(agent, state)
    assert recalled.core.active_record_id == 7
    assert recalled.core.active_record_rank == 1
    assert recalled.core.counters.records_scored == 6  # three at scheduling, three at RECALL
    composed, _ = execute_next(agent, recalled)
    assert composed.core.focus_node_id == 7
    assert composed.core.support_ids == (7,)
    assert composed.core.mode is Mode.QUIESCENT
    assert composed.core.active_record_id is composed.core.active_record_rank is None


def test_recall_recomputes_choice_from_current_public_memory_and_preserves_counters():
    agent = ScriptedEventFlowAgent()
    state = public_state(agent, facts=[(1, LinkFact(0, 1, 0.9)), (2, LinkFact(0, 2, 0.8))])
    event = agent.next_internal_event(state)
    flowed = advance_to(state, event.timestamp)
    slots = flowed.core.memory.records
    memory = replace(flowed.core.memory, records=(replace(slots[0], valid=False), slots[1]))
    flowed = replace(
        flowed,
        core=replace(
            flowed.core,
            memory=memory,
            counters=replace(
                flowed.core.counters, guard_predictions=123, checkpoint_flow_evaluations=7
            ),
        ),
    )
    recalled, _ = agent.on_internal(flowed, event)
    assert recalled.core.active_record_id == 2
    assert recalled.core.counters.guard_predictions == 123
    assert recalled.core.counters.checkpoint_flow_evaluations == 7
    assert recalled.core.counters.records_scored == 3


@pytest.mark.parametrize("delay,target", [(2.0, 2.65), (8.0, 7.6), (0.2, 1.165)])
def test_public_hazard_delay_alone_moves_action_clock(delay, target):
    agent = ScriptedEventFlowAgent()
    state = public_state(agent, facts=[(1, HazardFact(0, 2, delay))])
    state, _ = execute_next(agent, state)
    state, _ = execute_next(agent, state)
    assert state.core.hypothesis.deadline == 1.0 + delay
    state, actions = execute_next(agent, state)
    assert len(actions) == 1
    assert actions[0].hazard_type == 2
    assert actions[0].timestamp == pytest.approx(target, abs=2e-7)
    assert agent.next_internal_event(state) is None


def test_compressed_public_delay_uses_shorter_but_legal_guard_gaps():
    agent = ScriptedEventFlowAgent()
    state = public_state(agent, facts=[(1, LinkFact(0, 1)), (2, HazardFact(1, 2, 0.2))])
    for _ in range(4):
        event = agent.next_internal_event(state)
        assert 0.01 - 1e-8 <= event.timestamp - state.time <= 0.0136 + 1e-8
        assert bool((state.segment.parameters.guard_rates <= 500.0).all())
        state, _ = execute_next(agent, state)
    state, actions = execute_next(agent, state)
    assert actions[0].timestamp == pytest.approx(1.165, abs=1e-7)


def test_past_action_target_fails_loudly_instead_of_acting_immediately():
    agent = ScriptedEventFlowAgent()
    state = public_state(agent, facts=[(1, HazardFact(0, 2, 2.0))])
    recalled, _ = execute_next(agent, state)
    proposal = agent.next_internal_event(recalled)
    event = replace(proposal, timestamp=2.9, predicted_delta=2.9 - recalled.time)
    with pytest.raises(DynamicsError, match="strictly after"):
        agent.on_internal(advance_to(recalled, 2.9), event)


def test_impossibly_short_public_delay_fails_without_clipping_guard_rate():
    agent = ScriptedEventFlowAgent()
    with pytest.raises(DynamicsError, match="rate"):
        public_state(agent, facts=[(1, HazardFact(0, 2, 0.001))])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_scripted_cpu_mps_semantic_agreement(config):
    bundle = generated(config, 4, EpisodeVariant.POSITIVE)
    cpu, mps = [
        EventEngine(config.event_flow).run_episode(bundle, ScriptedEventFlowAgent(device=device))
        for device in ("cpu", "mps")
    ]
    assert cpu.score == mps.score
    assert cpu.actions == mps.actions
    assert [(row.kind, row.selected_record_id, row.support_after) for row in cpu.trace.events] == [
        (row.kind, row.selected_record_id, row.support_after) for row in mps.trace.events
    ]
