"""Finite causal execution with an engine-private episode and terminal boundary.

RuntimeCore counters are authoritative. The engine owns guard prediction and
flow counts; agents preserve these and charge actual controller/retrieval work
on the incoming core. Shared jumps own transition/internal-event counts. The
protocol's compute_counters() snapshot is deliberately never synchronized back.
"""

import json
import re
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from functools import wraps
from pathlib import Path

import torch

from silent_cascade.env.episode import EpisodeBundle, EpisodeTruth
from silent_cascade.env.reward import EpisodeScore, score_actions
from silent_cascade.errors import CrashBundleError, DynamicsError, ProvenanceError, TimeOrderError
from silent_cascade.eventflow.config import EventFlowConfig
from silent_cascade.eventflow.flow import state_at
from silent_cascade.eventflow.invariants import (
    validate_post_jump,
    validate_runtime_state,
    validate_session_boundary,
)
from silent_cascade.eventflow.protocols import AgentCondition
from silent_cascade.eventflow.scheduling import (
    ExternalEventQueue,
    PredictionCache,
    ScheduledChoice,
    choose_next_event,
    event_priority_kind,
    normalize_internal_gap,
    selected_gap_clamp_streak,
)
from silent_cascade.eventflow.state import (
    ComputeCounters,
    RuntimeCore,
    RuntimeState,
    require_time,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.crash_bundle import CrashContext, write_crash_bundle
from silent_cascade.logging.trace import (
    CausalTrace,
    SegmentSummary,
    TraceRecorder,
    sanitized_crash_events,
    tensor_sha256,
)
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
    segment_anchor: SegmentSummary | None = None
    public_state_anchor: bytes | None = None
    failure: DynamicsError | None = None


def _core_metadata(state: RuntimeState) -> dict:
    return {
        item.name: (
            asdict(value)
            if is_dataclass(value)
            else tuple(asdict(entry) if is_dataclass(entry) else entry for entry in value)
            if isinstance(value, tuple)
            else value
        )
        for item in fields(state.core)
        if item.name != "continuous"
        for value in (getattr(state.core, item.name),)
    }


def _public_state_anchor(state: RuntimeState) -> bytes:
    metadata = _core_metadata(state)
    # Operational counters can change on lookahead/pause, while this captures
    # the immutable public semantic history at the most recent causal jump.
    del metadata["counters"]
    del metadata["consecutive_gap_clamps"]
    return canonical_json_bytes(metadata)


def _runtime_signature(state: RuntimeState) -> tuple:
    """Capture input storage before a callback can mutate shared tensor/record data."""
    return (
        state.time,
        SegmentSummary.from_segment(state.segment),
        canonical_json_bytes(_core_metadata(state)),
        tuple(
            tensor_sha256(getattr(state.core.continuous, item.name))
            for item in fields(state.core.continuous)
        ),
    )


def _failure_boundary(function):
    @wraps(function)
    def guarded(self, context, *args, **kwargs):
        try:
            if isinstance(context, RuntimeSession) and context.failure is not None:
                raise context.failure
            return function(self, context, *args, **kwargs)
        except (DynamicsError, ProvenanceError, TypeError, ValueError, AttributeError) as caught:
            error = (
                caught
                if isinstance(caught, DynamicsError)
                else DynamicsError("runtime callback or boundary is invalid").with_traceback(
                    caught.__traceback__
                )
            )
            if isinstance(context, RuntimeSession):
                if context.failure is not None:
                    raise
                context.failure = error
            self._publish_failure(context, error)
            if error is caught:
                raise
            raise error from caught

    return guarded


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Post-outcome result without truth, session, or private terminal objects."""

    public_id: str
    score: EpisodeScore
    actions: tuple[Action, ...]
    counters: ComputeCounters
    trace: CausalTrace


class EventEngine:
    def __init__(
        self,
        config: EventFlowConfig,
        *,
        crash_root: Path | None = None,
        source_revision: str | None = None,
    ):
        self.config = config
        if source_revision is not None and not re.fullmatch(
            r"[0-9a-f]{40}|[0-9a-f]{64}", source_revision
        ):
            raise DynamicsError(
                "source revision must be a canonical full commit digest",
                context={"invariant": "source_revision_format"},
            )
        self.crash_root = crash_root
        self.source_revision = source_revision

    def _publish_failure(
        self, context: RuntimeSession | EpisodeBundle, error: DynamicsError
    ) -> None:
        if self.crash_root is None:
            return
        public_id = (
            context.public_id
            if isinstance(context, RuntimeSession)
            else context.public.init.episode_public_id
        )
        last_events = (
            tuple(
                json.loads(
                    canonical_json_bytes(
                        {"events": sanitized_crash_events(context.trace.snapshot())}
                    )
                )["events"]
            )
            if isinstance(context, RuntimeSession)
            else ()
        )
        try:
            write_crash_bundle(
                self.crash_root,
                error=error,
                sanitize_diagnostics=True,
                context=CrashContext(
                    episode_public_id=public_id,
                    config_sha256=sha256_bytes(canonical_json_bytes(self.config)),
                    source_revision=self.source_revision,
                    last_events=last_events,
                ),
            )
        except CrashBundleError as publication_error:
            raise publication_error from error

    @staticmethod
    def _check_session(session: RuntimeSession) -> None:
        cached = session.prediction_cache.snapshot()
        validate_session_boundary(
            session.state,
            prediction_snapshot_sha256=cached.prediction_snapshot_sha256 if cached else None,
        )
        if cached is not None and cached.parent_event_id != session.state.segment.parent_event_id:
            raise DynamicsError(
                "cached prediction parent differs from the causal segment",
                context={"invariant": "cached_prediction_parent"},
            )
        if session.segment_anchor != SegmentSummary.from_segment(session.state.segment):
            raise DynamicsError(
                "causal segment storage changed between engine boundaries",
                context={"invariant": "segment_storage_changed"},
            )
        if session.public_state_anchor != _public_state_anchor(session.state):
            raise DynamicsError(
                "public causal history changed between engine boundaries",
                context={"invariant": "public_history_changed"},
            )

    @_failure_boundary
    def start_episode(self, bundle: EpisodeBundle, agent: AgentCondition) -> RuntimeSession:
        queue = ExternalEventQueue(bundle.public, bundle.truth.private_terminal)
        state = agent.initialize(bundle.public.init)
        if not isinstance(state, RuntimeState) or state.core.mode is not Mode.OBSERVING:
            raise DynamicsError(
                "initialization must return an observing RuntimeState",
                context={"invariant": "initial_runtime_mode"},
            )
        if state.time != bundle.public.init.initial_time or state.segment.started_at != state.time:
            raise TimeOrderError(
                "initial state and segment must start at the public initial time",
                context={"invariant": "initial_runtime_time"},
            )
        validate_runtime_state(state)
        if (
            state.core.last_event_id is not None
            or state.core.executed_internal_events
            or state.core.actions
            or state.core.memory.records
        ):
            raise DynamicsError(
                "initialization must not invent events, actions or perceived facts",
                context={"invariant": "initial_public_history"},
            )
        return RuntimeSession(
            bundle.public.init.episode_public_id,
            bundle.truth,
            state,
            queue,
            PredictionCache(),
            TraceRecorder(),
            state.time,
            segment_anchor=SegmentSummary.from_segment(state.segment),
            public_state_anchor=_public_state_anchor(state),
        )

    @_failure_boundary
    def next_internal_event(
        self, session: RuntimeSession, agent: AgentCondition
    ) -> InternalEvent | None:
        """Cache dormancy as well as events; pass only public runtime state."""
        self._check_session(session)
        if session.state.core.mode is Mode.TERMINAL:
            return None

        def predict(state: RuntimeState) -> InternalEvent | None:
            counters = replace(
                state.core.counters, guard_predictions=state.core.counters.guard_predictions + 1
            )
            session.state = replace(state, core=replace(state.core, counters=counters))
            signature = _runtime_signature(session.state)
            result = agent.next_internal_event(session.state)
            if _runtime_signature(session.state) != signature:
                raise DynamicsError(
                    "prediction callback mutated its runtime input",
                    context={"invariant": "prediction_input_mutated"},
                )
            validate_runtime_state(session.state)
            return result

        return session.prediction_cache.get_or_predict(session.state, predict)

    def _next_choice(self, session: RuntimeSession, agent: AgentCondition) -> ScheduledChoice:
        internal = self.next_internal_event(session, agent)
        normalized, clamped = (
            (None, False)
            if internal is None
            else normalize_internal_gap(
                internal,
                origin_time=session.state.segment.started_at,
                minimum_gap=self.config.guards.minimum_internal_gap,
            )
        )
        # A pause may pass a cached losing near-tie proposal before its later
        # winner. Resolve from the causal origin, then forbid an actual rewind.
        choice = choose_next_event(
            session.external_queue,
            normalized,
            condition=agent.name,
            current_time=session.state.segment.started_at,
        )
        if choice is None:
            raise DynamicsError(
                "nonterminal session has no next causal event",
                context={"invariant": "missing_next_event"},
            )
        if choice.timestamp < session.state.time:
            raise TimeOrderError(
                "selected event precedes the runtime cursor",
                context={"invariant": "selected_event_time"},
            )
        if isinstance(choice.event, InternalEvent):
            choice = replace(
                choice,
                was_gap_clamped=clamped,
                raw_predicted_delta=internal.predicted_delta if clamped else None,
            )
        return choice

    def advance_to(self, state: RuntimeState, target_time: float) -> RuntimeState:
        """Materialize one causal advance, charging by segment elapsed time.

        This is an event-execution operation, not a pause/snapshot query. Even
        if a checkpoint already materialized this timestamp, a positive causal
        interval costs one flow evaluation. Equal-time causal jumps cost zero.
        """
        require_time(target_time, "target_time")
        validate_runtime_state(state)
        if state.core.mode is Mode.TERMINAL:
            raise DynamicsError(
                "cannot advance a terminal state", context={"invariant": "advance_after_terminal"}
            )
        if target_time < state.time:
            raise TimeOrderError(
                "advance time cannot precede the current runtime time",
                context={"invariant": "advance_time"},
            )
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
            raise DynamicsError(
                "callback must return a nonterminal RuntimeState",
                context={"invariant": "callback_runtime_result"},
            )
        for name in ("flow_evaluations", "checkpoint_flow_evaluations", "guard_predictions"):
            if getattr(after.core.counters, name) != getattr(before.core.counters, name):
                raise DynamicsError(
                    "callback changed an engine-owned counter",
                    context={"invariant": "callback_counter_owner"},
                )
        if after.core.actions[: len(before.core.actions)] != before.core.actions:
            raise DynamicsError(
                "callback rewrote an already emitted action",
                context={"invariant": "callback_actions_rewritten"},
            )

    @staticmethod
    def _require_emitted_matches_state(
        before: RuntimeState, after: RuntimeState, emitted: list[Action]
    ) -> None:
        if (
            not isinstance(emitted, list)
            or tuple(emitted) != after.core.actions[len(before.core.actions) :]
        ):
            raise DynamicsError(
                "emitted actions do not match the new runtime actions",
                context={"invariant": "callback_emission_mismatch"},
            )

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
        self._check_session(session)
        event = choice.event
        validate_session_boundary(session.state, next_event=event)
        if isinstance(event, InternalEvent):
            if session.state.core.executed_internal_events >= self.config.max_internal_events:
                raise DynamicsError(
                    "internal event limit exceeded", context={"invariant": "internal_event_cap"}
                )
            if event.timestamp < session.state.core.same_kind_refractory_until[event.guard_index]:
                raise DynamicsError(
                    "internal event kind is refractory",
                    context={"invariant": "same_kind_refractory"},
                )
        streak = selected_gap_clamp_streak(session.state.core.consecutive_gap_clamps, choice)
        prepared = replace(
            session.state, core=replace(session.state.core, consecutive_gap_clamps=streak)
        )
        before = self.advance_to(prepared, choice.timestamp)
        signature = _runtime_signature(before)
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
        if _runtime_signature(before) != signature:
            raise DynamicsError(
                "jump callback mutated its runtime input",
                context={"invariant": "jump_input_mutated"},
            )
        validate_post_jump(before, event, after)
        selected_record_id = None
        selected_rank = None
        if isinstance(event, InternalEvent):
            if event.kind is InternalEventKind.RECALL:
                selected_record_id = after.core.active_record_id
                selected_rank = after.core.active_record_rank
            elif event.kind is InternalEventKind.COMPOSE:
                selected_record_id = before.core.active_record_id
                selected_rank = before.core.active_record_rank
        session.trace.record(
            event,
            before,
            after,
            selected_record_id=selected_record_id,
            selected_rank=selected_rank,
            tie=choice.tie,
            was_gap_clamped=choice.was_gap_clamped,
            raw_predicted_delta=choice.raw_predicted_delta,
        )
        if isinstance(event, ExternalEvent):
            session.external_queue.consume(event)
        session.state = replace(before, core=after) if isinstance(after, RuntimeCore) else after
        session.terminal_score = score
        session.pause_cursor = session.state.time
        session.prediction_cache.invalidate()
        session.segment_anchor = SegmentSummary.from_segment(session.state.segment)
        session.public_state_anchor = _public_state_anchor(session.state)
        return session.state.core.mode is Mode.TERMINAL

    @_failure_boundary
    def step(self, session: RuntimeSession, agent: AgentCondition) -> bool:
        """Execute at most one causal event; already-terminal sessions are inert."""
        if session.state.core.mode is Mode.TERMINAL:
            self._check_session(session)
            return True
        return self._execute_choice(session, agent, self._next_choice(session, agent))

    @_failure_boundary
    def run_until(self, session: RuntimeSession, agent: AgentCondition, pause_time: float) -> None:
        """Run causal events through a host time, then materialize without an event."""
        require_time(pause_time, "pause_time")
        if session.state.core.mode is Mode.TERMINAL:
            raise DynamicsError(
                "cannot pause an already-terminal session",
                context={"invariant": "pause_after_terminal"},
            )
        if pause_time < session.state.time:
            raise TimeOrderError(
                "pause cannot precede the current runtime time", context={"invariant": "pause_time"}
            )
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
            raise DynamicsError(
                "terminal session is missing its score",
                context={"invariant": "missing_terminal_score"},
            )
        return EpisodeResult(
            session.public_id,
            session.terminal_score,
            session.state.core.actions,
            session.state.core.counters,
            session.trace.snapshot(),
        )
