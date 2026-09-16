"""Private shared runtime state codec; no learned policy or training imports."""

import math
from dataclasses import fields, replace
from typing import Literal

import torch

from silent_cascade.env.episode import PublicEpisode
from silent_cascade.env.reward import score_actions
from silent_cascade.errors import ReplayError
from silent_cascade.eventflow.checkpoint_rng import (
    decode_rng,
)
from silent_cascade.eventflow.engine import (
    EventEngine,
    RuntimeSession,
    _core_metadata,
    _public_state_anchor,
)
from silent_cascade.eventflow.flow import state_at
from silent_cascade.eventflow.guards import next_crossings
from silent_cascade.eventflow.invariants import (
    validate_post_jump,
    validate_runtime_state,
    validate_session_boundary,
)
from silent_cascade.eventflow.replay import CausalTraceArtifact
from silent_cascade.eventflow.scheduling import (
    ExternalEventQueue,
    PredictionCache,
    choose_next_event,
    normalize_internal_gap,
    selected_gap_clamp_streak,
)
from silent_cascade.eventflow.state import (
    INTERNAL_EVENT_ID_BASE,
    AnalyticSegment,
    ComputeCounters,
    ContinuousChannels,
    ContinuousState,
    RuntimeCore,
    RuntimeState,
    SegmentParameters,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.trace import (
    CausalEventSummary,
    CausalTrace,
    SegmentSummary,
    TraceRecorder,
    Trajectory,
    tensor_sha256,
)
from silent_cascade.memory import BoundedMemory
from silent_cascade.schemas import (
    Action,
    AgentInit,
    ExternalEvent,
    ExternalEventKind,
    Hypothesis,
    InternalEvent,
    InternalEventKind,
    Mode,
)
from silent_cascade.validation import StrictModel


class CoreMetadata(StrictModel):
    mode: Mode
    memory: BoundedMemory
    focus_node_id: int | None
    active_record_id: int | None
    active_record_rank: int | None
    support_ids: tuple[int, ...]
    hypothesis: Hypothesis | None
    activation_time: float | None
    actions: tuple[Action, ...]
    same_kind_refractory_until: tuple[float, float, float]
    executed_internal_events: int
    consecutive_gap_clamps: int
    last_event_id: int | None
    last_event_time: float | None
    counters: ComputeCounters


class StateMetadata(StrictModel):
    core: CoreMetadata
    time: float
    segment_started_at: float
    segment_parent_event_id: int
    prediction_snapshot_sha256: str


class TensorMetadata(StrictModel):
    dtype: Literal["torch.float32", "torch.uint8"]
    shape: tuple[int, ...]
    sha256: str


def _state_metadata(
    state: RuntimeState, prefix: str, tensors: dict[str, torch.Tensor]
) -> StateMetadata:
    for group, channels in (
        ("current", state.core.continuous),
        ("origin", state.segment.origin),
        ("targets", state.segment.parameters.flow_targets),
        ("rates", state.segment.parameters.flow_rates),
    ):
        for item in fields(channels):
            tensors[f"{prefix}.{group}.{item.name}"] = (
                getattr(channels, item.name).detach().cpu().contiguous().clone()
            )
    for name in ("guard_targets", "guard_rates"):
        tensors[f"{prefix}.{name}"] = (
            getattr(state.segment.parameters, name).detach().cpu().contiguous().clone()
        )
    return StateMetadata(
        core=CoreMetadata.model_validate_json(canonical_json_bytes(_core_metadata(state))),
        time=state.time,
        segment_started_at=state.segment.started_at,
        segment_parent_event_id=state.segment.parent_event_id,
        prediction_snapshot_sha256=state.segment.prediction_snapshot_sha256,
    )


def _tensor_metadata(name: str, tensor: torch.Tensor) -> TensorMetadata:
    return TensorMetadata(
        dtype=str(tensor.dtype),
        shape=tuple(tensor.shape),
        sha256=sha256_bytes(
            canonical_json_bytes({"name": name, "tensor_sha256": tensor_sha256(tensor)})
        ),
    )


def _metadata_hash(payload: dict) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {key: value for key, value in payload.items() if key != "metadata_sha256"}
        )
    )


def _decode_state(
    metadata: StateMetadata, prefix: str, tensors: dict[str, torch.Tensor], device: str
) -> RuntimeState:
    def channels(group, cls):
        return cls(
            **{
                item.name: tensors.pop(f"{prefix}.{group}.{item.name}").to(device)
                for item in fields(cls)
            }
        )

    current, origin = channels("current", ContinuousState), channels("origin", ContinuousState)
    parameters = SegmentParameters(
        channels("targets", ContinuousChannels),
        channels("rates", ContinuousChannels),
        tensors.pop(f"{prefix}.guard_targets").to(device),
        tensors.pop(f"{prefix}.guard_rates").to(device),
    )
    core = RuntimeCore(
        continuous=current,
        **{name: getattr(metadata.core, name) for name in CoreMetadata.model_fields},
    )
    state = RuntimeState(
        core,
        AnalyticSegment(
            metadata.segment_started_at,
            origin,
            parameters,
            metadata.segment_parent_event_id,
            metadata.prediction_snapshot_sha256,
        ),
        metadata.time,
    )
    validate_runtime_state(state)
    return state


def _trace_from_metadata(metadata: CausalTraceArtifact) -> CausalTrace:
    rows = tuple(
        CausalEventSummary(
            **{name: getattr(row, name) for name in CausalEventSummary.__dataclass_fields__}
        )
        for row in metadata.events
    )
    trace = CausalTrace(rows)
    if trace.sha256 != metadata.sha256:
        raise ReplayError("checkpoint trace hash differs")
    recorder = TraceRecorder()
    for row in rows:
        recorder.append(row)
        if row.tie_id != metadata.events[len(recorder.snapshot().events) - 1].tie_id:
            raise ReplayError("checkpoint tie hash differs")
    return trace


def _validate_history(
    state: RuntimeState,
    anchors: tuple[RuntimeState, ...],
    trace: CausalTrace,
    *,
    pending: tuple[ExternalEvent, ...],
    terminal: ExternalEvent,
) -> None:
    if len(anchors) != len(trace.events) + 1:
        raise ReplayError("checkpoint trajectory and trace lengths differ")
    initial = anchors[0]
    if (
        initial.core.last_event_id is not None
        or initial.core.actions
        or initial.core.memory.records
        or initial.core.mode is not Mode.OBSERVING
        or initial.core.counters.jump_applications
    ):
        raise ReplayError("invalid initial causal anchor")
    recorder = TraceRecorder()
    external = tuple(
        ExternalEvent(row.event_id, row.timestamp, ExternalEventKind(row.kind), row.payload)
        for row in trace.events
        if row.kind in {"fact", "activate"}
    )
    queue = ExternalEventQueue.from_snapshot(
        (*external, *pending, *((terminal,) if state.core.mode is Mode.TERMINAL else ()))
    )
    for previous, after, row in zip(anchors, anchors[1:], trace.events, strict=False):
        if row.kind in {"fact", "activate", "terminal"}:
            event = ExternalEvent(
                row.event_id,
                row.timestamp,
                terminal.kind if row.kind == "terminal" else ExternalEventKind(row.kind),
                terminal.payload if row.kind == "terminal" else row.payload,
            )
        else:
            kind = InternalEventKind(row.kind)
            event = InternalEvent(
                row.event_id,
                row.parent_event_id,
                row.timestamp,
                kind,
                ("recall", "compose", "act").index(row.kind),
                row.raw_predicted_delta if row.was_gap_clamped else row.delta,
            )
        prediction = _analytic_prediction_for_validation(previous)
        normalized, clamped = (
            (None, False)
            if prediction is None
            else normalize_internal_gap(prediction, origin_time=previous.segment.started_at)
        )
        choice = choose_next_event(queue, normalized, current_time=previous.segment.started_at)
        if choice is None or choice.event != event or choice.tie != row.tie:
            raise ReplayError("checkpoint historical event differs from the analytic scheduler")
        internal = isinstance(event, InternalEvent)
        if internal:
            choice = replace(
                choice,
                was_gap_clamped=clamped,
                raw_predicted_delta=prediction.predicted_delta if clamped else None,
            )
        if (
            choice.was_gap_clamped != row.was_gap_clamped
            or choice.raw_predicted_delta != row.raw_predicted_delta
        ):
            raise ReplayError("checkpoint historical clamp differs from scheduler")
        streak = selected_gap_clamp_streak(previous.core.consecutive_gap_clamps, choice)
        counters = replace(
            previous.core.counters,
            guard_predictions=previous.core.counters.guard_predictions + 1,
            flow_evaluations=previous.core.counters.flow_evaluations
            + int(row.timestamp > previous.segment.started_at),
            checkpoint_flow_evaluations=after.core.counters.checkpoint_flow_evaluations,
        )
        if (
            after.core.counters.checkpoint_flow_evaluations
            < previous.core.counters.checkpoint_flow_evaluations
        ):
            raise ReplayError("checkpoint diagnostic flow counter reversed")
        before = replace(
            previous,
            time=row.timestamp,
            core=replace(
                previous.core,
                continuous=state_at(previous, row.timestamp),
                counters=counters,
                consecutive_gap_clamps=streak,
            ),
        )
        post = after.core if row.kind == "terminal" else after
        validate_post_jump(before, event, post)
        if row.kind == "terminal":
            if event != terminal or after.core.counters != replace(
                counters, jump_applications=counters.jump_applications + 1
            ):
                raise ReplayError("checkpoint terminal identity or counters differ")
        else:
            EventEngine._require_callback_result(before, after)
        actual = recorder.record(
            event,
            before,
            after.core if row.kind == "terminal" else after,
            selected_record_id=row.selected_record_id,
            selected_rank=row.selected_rank,
            tie=row.tie,
            was_gap_clamped=row.was_gap_clamped,
            raw_predicted_delta=row.raw_predicted_delta,
        )
        if canonical_json_bytes(actual.to_payload()) != canonical_json_bytes(row.to_payload()):
            raise ReplayError("checkpoint causal anchor differs from trace")
        if isinstance(event, ExternalEvent):
            queue.consume(event)
    last = anchors[-1]
    if _public_state_anchor(last) != _public_state_anchor(state) or SegmentSummary.from_segment(
        last.segment
    ) != SegmentSummary.from_segment(state.segment):
        raise ReplayError("checkpoint current state differs from causal anchor")
    for item in fields(ComputeCounters):
        a, b = getattr(last.core.counters, item.name), getattr(state.core.counters, item.name)
        if item.name == "checkpoint_flow_evaluations":
            valid = b >= a
        elif item.name == "guard_predictions":
            valid = b in {a, a + 1}
        else:
            valid = a == b
        if not valid:
            raise ReplayError("checkpoint counters differ from causal anchor")


def _analytic_prediction_for_validation(anchor: RuntimeState) -> InternalEvent | None:
    """Pure diagnostic root/kind/refractory check; never install a proposal.

    The existing host-float equations are evaluated at the immutable causal
    anchor, including when a paused cursor has passed a losing near-tie ACT.
    This neither calls an agent nor touches cache, counters or generator state.
    """
    crossings = next_crossings(anchor)
    if not crossings:
        return None
    crossing = crossings[0]
    return InternalEvent(
        crossing.event_id,
        crossing.parent_event_id,
        crossing.timestamp,
        crossing.kind,
        crossing.guard_index,
        crossing.predicted_delta,
    )


def _validate_tensor_layout(metadata, tensors, config):
    expected = {"rng.cpu"}
    if metadata.rng.has_mps:
        expected.add("rng.mps")
    for prefix in ("state", *(f"trajectory.{index}" for index in range(len(metadata.trajectory)))):
        for group, cls in (
            ("current", ContinuousState),
            ("origin", ContinuousState),
            ("targets", ContinuousChannels),
            ("rates", ContinuousChannels),
        ):
            for item in fields(cls):
                name = f"{prefix}.{group}.{item.name}"
                expected.add(name)
                tensor = tensors[name]
                if (
                    tensor.dtype is not torch.float32
                    or tensor.shape != (getattr(config.dimensions, item.name),)
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise ReplayError("checkpoint continuous tensor layout differs")
        for name in ("guard_targets", "guard_rates"):
            key = f"{prefix}.{name}"
            expected.add(key)
            tensor = tensors[key]
            if (
                tensor.dtype is not torch.float32
                or tensor.shape != (3,)
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ReplayError("checkpoint guard tensor layout differs")
    if expected != set(tensors):
        raise ReplayError("checkpoint tensor names differ from schema")


def _validate_external_history(metadata) -> None:
    """Bind pending public events to the consumed prefix without private replay."""
    consumed = tuple(
        ExternalEvent(row.event_id, row.timestamp, ExternalEventKind(row.kind), row.payload)
        for row in metadata.trace.events
        if row.kind in {"fact", "activate"}
    )
    pending = tuple(
        event
        for event in metadata.external_queue
        if event.kind in {ExternalEventKind.FACT, ExternalEventKind.ACTIVATE}
    )
    public = PublicEpisode(
        AgentInit(metadata.public_id, 64, 4, metadata.trajectory[0].time), (*consumed, *pending)
    )
    if public.events[-1].timestamp != metadata.truth.activation_time:
        raise ReplayError("checkpoint public and private activation times differ")
    facts = {event.event_id for event in public.events if event.kind is ExternalEventKind.FACT}
    if not set(metadata.truth.relevant_record_ids).issubset(facts):
        raise ReplayError("checkpoint private references include absent public facts")


def validate_runtime_contents(metadata, tensors, config, device):
    _validate_tensor_layout(metadata, tensors, config)
    _validate_external_history(metadata)
    if device is None:
        # Portable inspection stops at schema, storage and integrity checks.
        # Backend-dependent analytic/cross-anchor validation belongs to
        # same-device runnable restoration, before any caller mutation.
        _trace_from_metadata(metadata.trace)
        ExternalEventQueue.from_snapshot(metadata.external_queue)
        return None, decode_rng(
            metadata.rng,
            {name: value for name, value in tensors.items() if name.startswith("rng.")},
        )
    if device not in {"cpu", "mps"} or (device == "mps" and not torch.backends.mps.is_available()):
        raise ReplayError("checkpoint device is unavailable")
    remaining = dict(tensors)
    state = _decode_state(metadata.state, "state", remaining, device)
    anchors = tuple(
        _decode_state(item, f"trajectory.{index}", remaining, device)
        for index, item in enumerate(metadata.trajectory)
    )
    rng = decode_rng(metadata.rng, remaining)
    trace = _trace_from_metadata(metadata.trace)
    trajectory = Trajectory.from_snapshots(anchors)
    _validate_history(
        state,
        anchors,
        trace,
        pending=metadata.external_queue,
        terminal=metadata.truth.private_terminal,
    )
    if metadata.pause_cursor != state.time:
        raise ReplayError("checkpoint pause cursor differs")
    queue = ExternalEventQueue.from_snapshot(metadata.external_queue)
    terminal = metadata.truth.private_terminal
    pending = queue.snapshot()
    is_terminal = state.core.mode is Mode.TERMINAL
    if (
        not is_terminal
        and tuple(
            event
            for event in pending
            if event.kind in {ExternalEventKind.OUTCOME, ExternalEventKind.END}
        )
        != (terminal,)
    ) or (is_terminal and pending):
        raise ReplayError("checkpoint terminal queue differs")
    if any(
        event.timestamp < state.time or event.event_id in {row.event_id for row in trace.events}
        for event in pending
    ):
        raise ReplayError("checkpoint queue overlaps causal history")
    if (metadata.terminal_score is not None) != is_terminal:
        raise ReplayError("checkpoint terminal score differs")
    if is_terminal and metadata.terminal_score != score_actions(metadata.truth, state.core.actions):
        raise ReplayError("checkpoint terminal score differs from deterministic scoring")
    cache = PredictionCache.from_snapshot(metadata.prediction_cache)
    session = RuntimeSession(
        metadata.public_id,
        metadata.truth,
        state,
        queue,
        cache,
        TraceRecorder(trace),
        metadata.pause_cursor,
        trajectory,
        terminal_score=metadata.terminal_score,
        segment_anchor=SegmentSummary.from_segment(state.segment),
        public_state_anchor=_public_state_anchor(state),
    )
    EventEngine._check_session(session)
    if cache.is_computed:
        cached = cache.snapshot()
        if cached.event != _analytic_prediction_for_validation(anchors[-1]):
            raise ReplayError("checkpoint cached proposal differs from analytic guard root")
        if cached.event is not None:
            # A losing near-tie event may precede a pause cursor. Validate at
            # its causal anchor and race there before checking the winner.
            validate_session_boundary(anchors[-1], next_event=cached.event)
            if (
                cached.event.event_id
                != INTERNAL_EVENT_ID_BASE + state.core.executed_internal_events
            ):
                raise ReplayError("checkpoint cached event ordinal differs")
            if not math.isclose(
                cached.event.predicted_delta,
                cached.event.timestamp - state.segment.started_at,
                rel_tol=0.0,
                abs_tol=2 * math.ulp(cached.event.timestamp),
            ):
                raise ReplayError("checkpoint cached delta differs from its causal origin")
            normalized, _ = normalize_internal_gap(
                cached.event, origin_time=state.segment.started_at
            )
            winner = choose_next_event(queue, normalized, current_time=state.segment.started_at)
            if winner is None or winner.timestamp < state.time:
                raise ReplayError("checkpoint cached winner precedes cursor")
    if state.core.counters.guard_predictions != anchors[-1].core.counters.guard_predictions + int(
        cache.is_computed
    ):
        raise ReplayError("checkpoint prediction counter differs from cache")
    if metadata.agent_counters.foundation_model_calls or any(
        getattr(metadata.agent_counters, item.name) > getattr(state.core.counters, item.name)
        for item in fields(ComputeCounters)
    ):
        raise ReplayError("checkpoint agent counters exceed authoritative runtime counters")
    return session, rng
