"""Finite causal execution and the private scorer/public callback boundary."""

from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import pytest
import torch

from silent_cascade.config import resolve_config
from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    EpisodeBundle,
    EpisodeKey,
    EpisodeRecipe,
    EpisodeTruth,
    EpisodeVariant,
    MatchedEpisodeCoordinate,
    PublicEpisode,
)
from silent_cascade.errors import DynamicsError, TimeOrderError
from silent_cascade.eventflow.config import Phase2Config
from silent_cascade.eventflow.engine import EpisodeResult, EventEngine, RuntimeSession
from silent_cascade.eventflow.flow import start_segment, state_at
from silent_cascade.eventflow.guards import allowed_mode_mask, prediction_snapshot_sha256
from silent_cascade.eventflow.jumps import (
    ComposeDecision,
    ComposeRole,
    apply_act,
    apply_activate,
    apply_compose,
    apply_fact,
    apply_recall,
    begin_post_jump_segment,
)
from silent_cascade.eventflow.state import (
    INTERNAL_EVENT_ID_BASE,
    ComputeCounters,
    ContinuousChannels,
    RuntimeCore,
    RuntimeState,
    SegmentParameters,
    make_initial_continuous_state,
)
from silent_cascade.schemas import (
    Action,
    ActivationPayload,
    AgentInit,
    Condition,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    InternalEvent,
    InternalEventKind,
    Mode,
)


def engine() -> EventEngine:
    config = resolve_config(
        Phase2Config,
        [
            Path("configs/base.yaml"),
            Path("configs/data/primary.yaml"),
            Path("configs/model/event_flow.yaml"),
        ],
    ).config.event_flow
    return EventEngine(config)


def bundle(*, positive: bool = True, delay: float = 10.0) -> EpisodeBundle:
    public = PublicEpisode(
        AgentInit("00000000-0000-4000-8000-000000000001", 64, 4, 0.0),
        (
            ExternalEvent(1, 1.0, ExternalEventKind.FACT, HazardFact(0, 2, delay)),
            ExternalEvent(2, 2.0, ExternalEventKind.ACTIVATE, ActivationPayload(0)),
        ),
    )
    truth = EpisodeTruth(
        EpisodeKey(
            "ofd-v1",
            SplitNamespace.DEBUG,
            SuiteName.IID_PRIMARY,
            998_877,
            MatchedEpisodeCoordinate("matched", 776_655, 443_322),
        ),
        EpisodeRecipe(
            1,
            EpisodeVariant.POSITIVE if positive else EpisodeVariant.SAFE_NEGATIVE,
            0,
            SuiteName.IID_PRIMARY,
            887_766,
        ),
        (61, 0),
        (1,),
        1 if positive else None,
        2 if positive else None,
        ExternalEvent(
            993_311,
            2.0 + delay,
            ExternalEventKind.OUTCOME if positive else ExternalEventKind.END,
            None,
        ),
        2.0,
        delay,
        2.0 + 0.75 * delay if positive else None,
        2.0 + 0.90 * delay if positive else None,
        2.0 + 0.825 * delay if positive else None,
        1,
        ("PRIVATE-REJECTION-SENTINEL",),
    )
    return EpisodeBundle(public, truth)


def walk(value):
    """Touch every nested dataclass field actually delivered to a callback."""
    yield value
    if is_dataclass(value):
        for item in fields(value):
            yield item.name
            yield from walk(getattr(value, item.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from walk(item)


def test_engine_copies_explicit_nonfirst_rank_without_assuming_agent_order() -> None:
    class RankedSpy(SpyAgent):
        def on_internal(self, state, event):
            if event.kind is InternalEventKind.RECALL:
                core = apply_recall(state, event, record_id=1, selected_rank=3)
                return self.install(core, state), []
            return super().on_internal(state, event)

    result = engine().run_episode(bundle(), RankedSpy())
    selected = [row for row in result.trace.events if row.selected_record_id is not None]
    assert [row.selected_rank for row in selected] == [3, 3]


class SpyAgent:
    """A three-event test trajectory, using real shared jumps and installations."""

    name = Condition.EVENT_FLOW

    def __init__(self, *, dormant: bool = False, action_time: float = 10.25, device="cpu"):
        self.dormant = dormant
        self.action_time = action_time
        self.device = device
        self.calls = []
        self.observed = []
        self.predictions = []

    def observe(self, name, *arguments):
        self.calls.append((name, arguments))
        for argument in arguments:
            self.observed.extend(walk(argument))

    def initialize(self, init: AgentInit) -> RuntimeState:
        self.observe("initialize", init)
        continuous = make_initial_continuous_state(device=self.device)
        core = RuntimeCore(continuous, counters=ComputeCounters(controller_calls=1))
        params = SegmentParameters(
            ContinuousChannels(
                **{f.name: getattr(continuous, f.name) for f in fields(ContinuousChannels)}
            ),
            ContinuousChannels(
                **{
                    f.name: torch.ones_like(getattr(continuous, f.name))
                    for f in fields(ContinuousChannels)
                }
            ),
            torch.full((3,), 0.5, device=self.device),
            torch.ones(3, device=self.device),
        )
        digest = prediction_snapshot_sha256(
            continuous,
            params,
            started_at=init.initial_time,
            allowed_mode_mask=allowed_mode_mask(core.mode),
            parent_event_id=0,
        )
        return start_segment(
            core,
            params,
            time=init.initial_time,
            parent_event_id=0,
            prediction_snapshot_sha256=digest,
        )

    def install(self, core, state):
        core = replace(
            core,
            counters=replace(core.counters, controller_calls=core.counters.controller_calls + 1),
        )
        return begin_post_jump_segment(core, state.segment.parameters, time=state.time)

    def on_external(self, state: RuntimeState, event: ExternalEvent) -> RuntimeState:
        self.observe("on_external", state, event)
        core = (
            apply_fact(state, event)
            if event.kind is ExternalEventKind.FACT
            else (apply_activate(state, event))
        )
        return self.install(core, state)

    def next_internal_event(self, state: RuntimeState) -> InternalEvent | None:
        self.observe("next_internal_event", state)
        choices = {
            Mode.SEARCHING: (InternalEventKind.RECALL, 0, 3.0),
            Mode.HAVE_MEMORY: (InternalEventKind.COMPOSE, 1, 4.0),
            Mode.HOLDING_HAZARD: (InternalEventKind.ACT, 2, self.action_time),
        }
        if self.dormant or state.core.mode not in choices:
            self.predictions.append(None)
            return None
        kind, index, timestamp = choices[state.core.mode]
        event = InternalEvent(
            INTERNAL_EVENT_ID_BASE + state.core.executed_internal_events,
            state.segment.parent_event_id,
            timestamp,
            kind,
            index,
            timestamp - state.time,
        )
        self.predictions.append(event)
        return event

    def on_internal(self, state: RuntimeState, event: InternalEvent):
        self.observe("on_internal", state, event)
        if event.kind is InternalEventKind.RECALL:
            core = apply_recall(state, event, record_id=1)
        elif event.kind is InternalEventKind.COMPOSE:
            core = apply_compose(
                state, event, ComposeDecision(ComposeRole.HAZARD, hazard_type=2, deadline=12.0)
            )
        else:
            core = apply_act(state, event, hazard_type=2)
        return self.install(core, state), list(core.actions[len(state.core.actions) :])

    def compute_counters(self):
        # Deliberately stale: reading this would overwrite the engine's counters.
        return ComputeCounters()


@pytest.mark.parametrize("positive", [False, True])
def test_sentinel_spy_and_private_terminal_score_only_inside_engine(positive, monkeypatch):
    import silent_cascade.eventflow.engine as module

    episode, agent, runtime_engine = bundle(positive=positive), SpyAgent(), engine()
    scored = []
    real_score = module.score_actions

    def score(truth, actions):
        scored.append((truth, actions))
        return real_score(truth, actions)

    monkeypatch.setattr(module, "score_actions", score)
    result = runtime_engine.run_episode(episode, agent)
    assert isinstance(result, EpisodeResult)
    assert agent.calls[0] == ("initialize", (episode.public.init,))
    assert agent.calls[0][1][0] is episode.public.init
    assert [args[1] for name, args in agent.calls if name == "on_external"] == list(
        episode.public.events
    )
    for name, args in agent.calls[1:]:
        assert type(args[0]) is RuntimeState
        assert len(args) == (1 if name == "next_internal_event" else 2)
        if name == "on_internal":
            assert any(args[1] is predicted for predicted in agent.predictions)
    private_names = {
        "truth",
        "private_terminal",
        "action_window_start",
        "action_window_end",
        "recipe",
        "root_seed",
        "relevant_node_path",
        "accepted_attempt",
    }
    assert not private_names.intersection(x for x in agent.observed if isinstance(x, str))
    secrets = {998_877, 776_655, 443_322, 887_766, 993_311, "PRIVATE-REJECTION-SENTINEL"}
    assert not secrets.intersection(x for x in agent.observed if type(x) in (int, str))
    assert not any(
        isinstance(x, (EpisodeTruth, EpisodeBundle, RuntimeSession)) for x in walk(result)
    )
    assert len(scored) == 1 and scored[0][0] is episode.truth
    assert scored[0][1] == result.actions
    assert result.score.timed_success is positive
    assert result.trace.events[-1].kind == "terminal"
    assert result.trace.events[-1].segment is None


def test_all_dormant_long_silence_has_no_polling_or_fixed_grid():
    runtime_engine, agent = engine(), SpyAgent(dormant=True)
    session = runtime_engine.start_episode(bundle(positive=False, delay=1_000_000.0), agent)
    assert not runtime_engine.step(session, agent)
    assert session.state.time == 1.0
    assert not runtime_engine.step(session, agent)
    assert session.state.time == 2.0
    assert runtime_engine.step(session, agent)
    assert session.state.time == 1_000_002.0 and session.state.core.mode is Mode.TERMINAL
    assert session.terminal_score.timed_success
    assert len(agent.predictions) == 3
    assert session.state.core.counters == ComputeCounters(
        flow_evaluations=3, controller_calls=3, guard_predictions=3, jump_applications=3
    )
    assert session.state.core.last_event_time == 1_000_002.0
    assert session.state.core.last_event_id == 993_311
    assert torch.count_nonzero(session.state.core.continuous.guard_accumulators) == 0
    assert session.state.segment.started_at == 2.0


def test_internal_events_run_before_terminal_with_once_per_boundary_predictions():
    runtime_engine, agent = engine(), SpyAgent()
    session = runtime_engine.start_episode(bundle(), agent)
    for _ in range(2):
        assert not runtime_engine.step(session, agent)
    first = runtime_engine.next_internal_event(session, agent)
    assert runtime_engine.next_internal_event(session, agent) is first
    assert len(agent.predictions) == 3
    for timestamp in (3.0, 4.0, 10.25):
        assert not runtime_engine.step(session, agent)
        assert session.state.time == timestamp
    assert runtime_engine.step(session, agent)
    assert session.state.core.counters == ComputeCounters(
        flow_evaluations=6, controller_calls=6, guard_predictions=6, jump_applications=6
    )
    assert [event.kind for event in session.trace.snapshot().events] == [
        "fact",
        "activate",
        "recall",
        "compose",
        "act",
        "terminal",
    ]
    assert [event.selected_record_id for event in session.trace.snapshot().events] == [
        None,
        None,
        1,
        1,
        None,
        None,
    ]


def test_losing_preactivation_prediction_is_rejected_before_external_preemption():
    class Preempted(SpyAgent):
        def next_internal_event(self, state):
            if state.core.mode is Mode.OBSERVING:
                self.observe("next_internal_event", state)
                event = InternalEvent(
                    INTERNAL_EVENT_ID_BASE,
                    state.segment.parent_event_id,
                    20.0,
                    InternalEventKind.RECALL,
                    0,
                    20.0 - state.time,
                )
                self.predictions.append(event)
                return event
            return super().next_internal_event(state)

    runtime_engine, agent = engine(), Preempted()
    session = runtime_engine.start_episode(bundle(), agent)
    with pytest.raises(DynamicsError, match="runtime invariant"):
        runtime_engine.next_internal_event(session, agent)
    assert not session.prediction_cache.is_computed
    assert session.state.core.executed_internal_events == 0
    assert session.state.time == 0.0
    assert not any(name == "on_internal" for name, args in agent.calls)


@pytest.mark.parametrize(
    "timestamp,success", [(9.499, False), (9.5, True), (10.25, True), (11.0, False)]
)
def test_positive_score_has_half_open_window(timestamp, success):
    result = engine().run_episode(bundle(), SpyAgent(action_time=timestamp))
    assert result.score.timed_success is success
    assert result.actions[0].timestamp == timestamp


@pytest.mark.parametrize("offset", [0.0, -0.5e-9, 0.5e-9])
def test_act_tied_with_private_outcome_loses(offset):
    agent = SpyAgent(action_time=12.0 + offset)
    result = engine().run_episode(bundle(), agent)
    assert result.actions == () and not result.score.timed_success
    assert [event.kind for event in result.trace.events] == [
        "fact",
        "activate",
        "recall",
        "compose",
        "terminal",
    ]
    assert result.trace.events[-1].tie.winner_event_id == 993_311
    assert not any(
        name == "on_internal" and args[1].kind is InternalEventKind.ACT
        for name, args in agent.calls
    )


def test_terminal_refuses_pause_and_never_calls_agent_again():
    runtime_engine, agent = engine(), SpyAgent(dormant=True)
    session = runtime_engine.start_episode(bundle(positive=False), agent)
    runtime_engine.run_until(session, agent, 12.0)
    before = len(agent.calls)
    assert runtime_engine.step(session, agent)
    assert runtime_engine.next_internal_event(session, agent) is None
    with pytest.raises(DynamicsError):
        runtime_engine.run_until(session, agent, 12.0)
    assert len(agent.calls) == before


def test_pause_inside_terminal_tie_keeps_preempted_prediction_without_time_rewind():
    runtime_engine, episode = engine(), bundle()
    expected = runtime_engine.run_episode(episode, SpyAgent(action_time=12.0 - 0.5e-9))
    agent = SpyAgent(action_time=12.0 - 0.5e-9)
    session = runtime_engine.start_episode(episode, agent)
    pause = 12.0 - 0.25e-9
    runtime_engine.run_until(session, agent, pause)
    cached = runtime_engine.next_internal_event(session, agent)
    assert cached.timestamp < session.state.time == pause
    assert session.state.core.actions == ()
    assert runtime_engine.step(session, agent)
    assert session.trace.snapshot().sha256 == expected.trace.sha256


def test_equal_time_public_events_apply_each_jump_without_an_extra_flow_charge():
    episode = bundle(positive=False)
    episode = replace(
        episode,
        public=replace(
            episode.public,
            events=(replace(episode.public.events[0], timestamp=2.0), episode.public.events[1]),
        ),
    )
    result = engine().run_episode(episode, SpyAgent(dormant=True))
    assert [event.delta for event in result.trace.events] == [2.0, 0.0, 10.0]
    assert result.counters.flow_evaluations == 2
    assert result.counters.jump_applications == 3


@pytest.mark.parametrize("change", ["mode", "time", "type"])
def test_initialization_rejects_nonpublic_runtime_start(change):
    class InvalidInit(SpyAgent):
        def initialize(self, init):
            state = super().initialize(init)
            if change == "mode":
                return replace(state, core=replace(state.core, mode=Mode.TERMINAL))
            if change == "time":
                return replace(state, time=1.0)
            return object()

    with pytest.raises(DynamicsError):
        engine().start_episode(bundle(), InvalidInit())


@pytest.mark.parametrize("change", ["counter", "mode", "type", "action"])
def test_invalid_external_callback_result_cannot_commit(change):
    class InvalidCallback(SpyAgent):
        def on_external(self, state, event):
            after = super().on_external(state, event)
            if change == "counter":
                return replace(after, core=replace(after.core, counters=ComputeCounters()))
            if change == "mode":
                return replace(after, core=replace(after.core, mode=Mode.TERMINAL))
            if change == "action":
                return replace(
                    after,
                    core=replace(after.core, actions=(Action(2, event.timestamp, event.event_id),)),
                )
            return object()

    runtime_engine, agent = engine(), InvalidCallback()
    session = runtime_engine.start_episode(bundle(), agent)
    with pytest.raises(DynamicsError):
        runtime_engine.step(session, agent)
    assert session.trace.snapshot().events == ()
    assert session.state.time == 0.0


def test_pauses_preserve_segment_cache_and_causal_trace():
    episode, runtime_engine, agent = bundle(), engine(), SpyAgent()
    expected = runtime_engine.run_episode(episode, SpyAgent())
    session = runtime_engine.start_episode(episode, agent)
    for pause in (0.25, 0.75, 1.0, 1.5, 2.0, 2.25, 2.75, 3.0, 4.0, 9.0, 10.25, 11.0):
        runtime_engine.run_until(session, agent, pause)
        assert session.state.time == pause and session.pause_cursor == pause
        assert session.state.segment.started_at <= pause
    runtime_engine.run_until(session, agent, 12.0)
    assert session.trace.snapshot().sha256 == expected.trace.sha256
    assert len(agent.predictions) == 6
    assert session.state.core.counters.checkpoint_flow_evaluations == 7


def test_dormant_pause_lookahead_never_consumes_external_or_repredicts():
    runtime_engine, agent = engine(), SpyAgent(dormant=True)
    session = runtime_engine.start_episode(bundle(positive=False), agent)
    original = session.state.segment
    queue = session.external_queue.snapshot()
    for time in (0.1, 0.2, 0.9):
        runtime_engine.run_until(session, agent, time)
        assert session.external_queue.snapshot() == queue
        assert (
            session.state.segment.prediction_snapshot_sha256 == original.prediction_snapshot_sha256
        )
        assert session.state.core.counters.flow_evaluations == 0
        assert len(session.trace.snapshot().events) == 0
    assert len(agent.predictions) == 1
    with pytest.raises(TimeOrderError):
        runtime_engine.run_until(session, agent, 0.8)


def test_causal_advance_at_materialized_boundary_charges_origin_elapsed_time_once():
    runtime_engine, agent = engine(), SpyAgent(dormant=True)
    session = runtime_engine.start_episode(bundle(positive=False), agent)
    initial = session.state
    materialized = replace(
        initial,
        time=1.0,
        core=replace(
            initial.core,
            continuous=state_at(initial, 1.0),
            counters=replace(initial.core.counters, checkpoint_flow_evaluations=1),
        ),
    )
    flowed = runtime_engine.advance_to(materialized, 1.0)
    assert flowed.core.counters.flow_evaluations == 1
    assert flowed.segment.started_at == 0.0
    assert runtime_engine.advance_to(initial, 0.0).core.counters.flow_evaluations == 0
    with pytest.raises(TimeOrderError):
        runtime_engine.advance_to(flowed, 0.5)


def test_emitted_actions_must_equal_new_state_actions_and_failure_does_not_commit():
    class MissingEmission(SpyAgent):
        def on_internal(self, state, event):
            after, _ = super().on_internal(state, event)
            return after, []

    runtime_engine, agent = engine(), MissingEmission()
    session = runtime_engine.start_episode(bundle(), agent)
    runtime_engine.run_until(session, agent, 4.0)
    before = session.trace.snapshot().sha256
    with pytest.raises(DynamicsError):
        runtime_engine.step(session, agent)
    assert session.state.core.actions == ()
    assert session.trace.snapshot().sha256 == before


def test_callback_failure_does_not_consume_external_or_append_trace():
    class Broken(SpyAgent):
        def on_external(self, state, event):
            raise DynamicsError("test callback failure")

    runtime_engine, agent = engine(), Broken()
    session = runtime_engine.start_episode(bundle(), agent)
    queued = session.external_queue.snapshot()
    with pytest.raises(DynamicsError, match="test callback failure"):
        runtime_engine.step(session, agent)
    assert session.external_queue.snapshot() == queued
    assert session.trace.snapshot().events == ()
    assert session.state.time == 0.0
    assert session.state.core.counters.guard_predictions == 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="native MPS unavailable")
def test_native_mps_executes_private_terminal_without_device_float64():
    result = engine().run_episode(bundle(), SpyAgent(device="mps"))
    assert result.score.timed_success
    assert result.counters.foundation_model_calls == 0
