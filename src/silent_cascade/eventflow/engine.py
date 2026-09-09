"""Finite causal execution with an engine-private episode and terminal boundary.

RuntimeCore counters are authoritative. The engine owns guard prediction and
flow counts; agents preserve these and charge actual controller/retrieval work
on the incoming core. Shared jumps own transition/internal-event counts. The
protocol's compute_counters() snapshot is deliberately never synchronized back.
"""

from dataclasses import dataclass, replace

import torch

from silent_cascade.env.episode import EpisodeBundle, EpisodeTruth
from silent_cascade.env.reward import EpisodeScore, score_actions
from silent_cascade.errors import DynamicsError, TimeOrderError
from silent_cascade.eventflow.config import EventFlowConfig
from silent_cascade.eventflow.flow import state_at
from silent_cascade.eventflow.protocols import AgentCondition
from silent_cascade.eventflow.scheduling import (
    ExternalEventQueue,
    PredictionCache,
    ScheduledChoice,
    choose_next_event,
    event_priority_kind,
)
from silent_cascade.eventflow.state import (
    ComputeCounters,
    RuntimeCore,
    RuntimeState,
    require_time,
)
from silent_cascade.logging.trace import CausalTrace, TraceRecorder
from silent_cascade.schemas import Action, ExternalEvent, InternalEvent, InternalEventKind, Mode


@dataclass(slots=True)
class RuntimeSession:
    """Engine-private execution handle; never a callback argument or public result."""

    public_id: str
    truth: EpisodeTruth
    state: RuntimeState
    external_queue: ExternalEventQueue
    prediction_cache: PredictionCache
    trace: TraceRecorder
    pause_cursor: float
    terminal_score: EpisodeScore | None = None


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Post-outcome result without truth, session, or private terminal objects."""

    public_id: str
    score: EpisodeScore
    actions: tuple[Action, ...]
    counters: ComputeCounters
    trace: CausalTrace


class EventEngine:
    def __init__(self, config: EventFlowConfig):
        self.config = config

    def start_episode(self, bundle: EpisodeBundle, agent: AgentCondition) -> RuntimeSession:
        queue = ExternalEventQueue(bundle.public, bundle.truth.private_terminal)
        state = agent.initialize(bundle.public.init)
        if not isinstance(state, RuntimeState) or state.core.mode is not Mode.OBSERVING:
            raise DynamicsError("initialization must return an observing RuntimeState")
        if state.time != bundle.public.init.initial_time or state.segment.started_at != state.time:
            raise TimeOrderError("initial state and segment must start at the public initial time")
        return RuntimeSession(
            bundle.public.init.episode_public_id,
            bundle.truth,
            state,
            queue,
            PredictionCache(),
            TraceRecorder(),
            state.time,
        )

    def next_internal_event(
        self, session: RuntimeSession, agent: AgentCondition
    ) -> InternalEvent | None:
        """Cache dormancy as well as events; pass only public runtime state."""
        if session.state.core.mode is Mode.TERMINAL:
            return None

        def predict(state: RuntimeState) -> InternalEvent | None:
            counters = replace(
                state.core.counters, guard_predictions=state.core.counters.guard_predictions + 1
            )
            session.state = replace(state, core=replace(state.core, counters=counters))
            return agent.next_internal_event(session.state)

        return session.prediction_cache.get_or_predict(session.state, predict)

    def _next_choice(self, session: RuntimeSession, agent: AgentCondition) -> ScheduledChoice:
        internal = self.next_internal_event(session, agent)
        # A pause may pass a cached losing near-tie proposal before its later
        # winner. Resolve from the causal origin, then forbid an actual rewind.
        choice = choose_next_event(
            session.external_queue,
            internal,
            condition=agent.name,
            current_time=session.state.segment.started_at,
        )
        if choice is None:
            raise DynamicsError("nonterminal session has no next causal event")
        if choice.timestamp < session.state.time:
            raise TimeOrderError("selected event precedes the runtime cursor")
        return choice

    def advance_to(self, state: RuntimeState, target_time: float) -> RuntimeState:
        """Materialize one causal advance, charging by segment elapsed time.

        This is an event-execution operation, not a pause/snapshot query. Even
        if a checkpoint already materialized this timestamp, a positive causal
        interval costs one flow evaluation. Equal-time causal jumps cost zero.
        """
        require_time(target_time, "target_time")
        if state.core.mode is Mode.TERMINAL:
            raise DynamicsError("cannot advance a terminal state")
        if target_time < state.time:
            raise TimeOrderError("advance time cannot precede the current runtime time")
        elapsed = target_time - state.segment.started_at
        if elapsed == 0.0:
            return state
        counters = replace(
            state.core.counters, flow_evaluations=state.core.counters.flow_evaluations + 1
        )
        return replace(
            state,
            time=target_time,
            core=replace(state.core, continuous=state_at(state, target_time), counters=counters),
        )

    @staticmethod
    def _require_callback_result(before: RuntimeState, after: RuntimeState) -> None:
        if not isinstance(after, RuntimeState) or after.core.mode is Mode.TERMINAL:
            raise DynamicsError("callback must return a nonterminal RuntimeState")
        for name in ("flow_evaluations", "checkpoint_flow_evaluations", "guard_predictions"):
            if getattr(after.core.counters, name) != getattr(before.core.counters, name):
                raise DynamicsError("callback changed an engine-owned counter")
        if after.core.actions[: len(before.core.actions)] != before.core.actions:
            raise DynamicsError("callback rewrote an already emitted action")

    @staticmethod
    def _require_emitted_matches_state(
        before: RuntimeState, after: RuntimeState, emitted: list[Action]
    ) -> None:
        if (
            not isinstance(emitted, list)
            or tuple(emitted) != after.core.actions[len(before.core.actions) :]
        ):
            raise DynamicsError("emitted actions do not match the new runtime actions")

    @staticmethod
    def _terminalize(before: RuntimeState, event: ExternalEvent) -> RuntimeCore:
        return replace(
            before.core,
            mode=Mode.TERMINAL,
            last_event_id=event.event_id,
            last_event_time=event.timestamp,
            continuous=replace(
                before.core.continuous,
                guard_accumulators=torch.zeros_like(before.core.continuous.guard_accumulators),
            ),
            counters=replace(
                before.core.counters, jump_applications=before.core.counters.jump_applications + 1
            ),
        )

    def _execute_choice(
        self, session: RuntimeSession, agent: AgentCondition, choice: ScheduledChoice
    ) -> bool:
        """Dispatch exactly one selected causal event, committing after validation."""
        event = choice.event
        before = self.advance_to(session.state, choice.timestamp)
        score = None
        if event_priority_kind(event) == "terminal":
            score = score_actions(session.truth, before.core.actions)
            after = self._terminalize(before, event)
        elif isinstance(event, ExternalEvent):
            after = agent.on_external(before, event)
            self._require_callback_result(before, after)
            self._require_emitted_matches_state(before, after, [])
        else:
            after, emitted = agent.on_internal(before, event)
            self._require_callback_result(before, after)
            self._require_emitted_matches_state(before, after, emitted)
        selected_record_id = None
        if isinstance(event, InternalEvent):
            if event.kind is InternalEventKind.RECALL:
                selected_record_id = after.core.active_record_id
            elif event.kind is InternalEventKind.COMPOSE:
                selected_record_id = before.core.active_record_id
        session.trace.record(
            event, before, after, selected_record_id=selected_record_id, tie=choice.tie
        )
        if isinstance(event, ExternalEvent):
            session.external_queue.consume(event)
        session.state = replace(before, core=after) if isinstance(after, RuntimeCore) else after
        session.terminal_score = score
        session.pause_cursor = session.state.time
        session.prediction_cache.invalidate()
        return session.state.core.mode is Mode.TERMINAL

    def step(self, session: RuntimeSession, agent: AgentCondition) -> bool:
        """Execute at most one causal event; already-terminal sessions are inert."""
        if session.state.core.mode is Mode.TERMINAL:
            return True
        return self._execute_choice(session, agent, self._next_choice(session, agent))

    def run_until(self, session: RuntimeSession, agent: AgentCondition, pause_time: float) -> None:
        """Run causal events through a host time, then materialize without an event."""
        require_time(pause_time, "pause_time")
        if session.state.core.mode is Mode.TERMINAL:
            raise DynamicsError("cannot pause an already-terminal session")
        if pause_time < session.state.time:
            raise TimeOrderError("pause cannot precede the current runtime time")
        while True:
            choice = self._next_choice(session, agent)
            if choice.timestamp > pause_time:
                state = session.state
                if pause_time > state.time:
                    counters = replace(
                        state.core.counters,
                        checkpoint_flow_evaluations=(
                            state.core.counters.checkpoint_flow_evaluations + 1
                        ),
                    )
                    session.state = replace(
                        state,
                        time=pause_time,
                        core=replace(
                            state.core, continuous=state_at(state, pause_time), counters=counters
                        ),
                    )
                session.pause_cursor = pause_time
                return
            if self._execute_choice(session, agent, choice):
                return

    def run_episode(self, bundle: EpisodeBundle, agent: AgentCondition) -> EpisodeResult:
        session = self.start_episode(bundle, agent)
        while not self.step(session, agent):
            pass
        if session.terminal_score is None:
            raise DynamicsError("terminal session is missing its score")
        return EpisodeResult(
            session.public_id,
            session.terminal_score,
            session.state.core.actions,
            session.state.core.counters,
            session.trace.snapshot(),
        )
