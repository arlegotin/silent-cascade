"""Immutable causal summaries, canonical tensor hashes and sanitized live traces.

Import this module directly. The logging package initializer remains part of the
frozen Phase 1 analysis import closure. Private replay envelopes belong elsewhere.
"""

import hashlib
import math
import struct
from dataclasses import asdict, dataclass, fields

import torch

from silent_cascade.errors import DynamicsError, TimeOrderError
from silent_cascade.eventflow.scheduling import (
    Event,
    TieResolution,
    causal_priority,
    event_priority_kind,
)
from silent_cascade.eventflow.state import (
    AnalyticSegment,
    ComputeCounters,
    ContinuousState,
    RuntimeCore,
    RuntimeState,
    require_time,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.schemas import (
    Action,
    ActivationPayload,
    ExternalEvent,
    ExternalPayload,
    HazardFact,
    Hypothesis,
    InternalEvent,
    LinkFact,
    Mode,
    SafeFact,
)


def _frame(payload: bytes) -> bytes:
    return struct.pack(">Q", len(payload)) + payload


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash version, dtype, shape and contiguous CPU bytes with length framing."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(b"silent-cascade-tensor-v1\0")
    digest.update(_frame(str(value.dtype).encode("ascii")))
    digest.update(
        _frame(
            struct.pack(">Q", value.ndim)
            + b"".join(struct.pack(">Q", dimension) for dimension in value.shape)
        )
    )
    digest.update(_frame(value.reshape(-1).view(torch.uint8).numpy().tobytes()))
    return digest.hexdigest()


def _state_summary(state: ContinuousState) -> tuple[float, str]:
    hashes = {item.name: tensor_sha256(getattr(state, item.name)) for item in fields(state)}
    # Widen on CPU only; hashing preserves exact original float32 values.
    squared = sum(
        float(getattr(state, item.name).detach().cpu().double().square().sum())
        for item in fields(state)
    )
    return math.sqrt(squared), sha256_bytes(canonical_json_bytes(hashes))


@dataclass(frozen=True, slots=True)
class SegmentSummary:
    started_at: float
    parent_event_id: int
    prediction_snapshot_sha256: str
    origin_sha256: str
    parameter_hashes: tuple[tuple[str, str], ...]

    @classmethod
    def from_segment(cls, segment: AnalyticSegment) -> "SegmentSummary":
        parameters = segment.parameters
        values = []
        for group in ("flow_targets", "flow_rates"):
            channels = getattr(parameters, group)
            values.extend(
                (f"{group}.{item.name}", tensor_sha256(getattr(channels, item.name)))
                for item in fields(channels)
            )
        values.extend(
            (name, tensor_sha256(getattr(parameters, name)))
            for name in ("guard_targets", "guard_rates")
        )
        return cls(
            segment.started_at,
            segment.parent_event_id,
            segment.prediction_snapshot_sha256,
            _state_summary(segment.origin)[1],
            tuple(sorted(values)),
        )


@dataclass(frozen=True, slots=True)
class CausalEventSummary:
    event_id: int
    parent_event_id: int | None
    timestamp: float
    kind: str
    payload: ExternalPayload
    pre_mode: Mode
    post_mode: Mode
    delta: float
    selected_record_id: int | None
    selected_rank: int | None
    support_before: tuple[int, ...]
    support_after: tuple[int, ...]
    hypothesis_before: Hypothesis | None
    hypothesis_after: Hypothesis | None
    actions: tuple[Action, ...]
    state_norm: float
    state_sha256: str
    counter_delta: ComputeCounters
    counters_after: ComputeCounters
    prediction_snapshot_sha256: str
    segment: SegmentSummary | None
    tie: TieResolution | None = None
    was_gap_clamped: bool = False
    raw_predicted_delta: float | None = None

    def __post_init__(self) -> None:
        if type(self.was_gap_clamped) is not bool:
            raise DynamicsError("gap clamp marker must be a boolean")
        if self.raw_predicted_delta is not None and (
            type(self.raw_predicted_delta) is not float
            or not math.isfinite(self.raw_predicted_delta)
            or self.raw_predicted_delta < 0.0
        ):
            raise DynamicsError("raw predicted delta must be finite and nonnegative")
        if self.was_gap_clamped and (
            self.kind not in {"recall", "compose", "act"} or self.raw_predicted_delta is None
        ):
            raise DynamicsError("a clamped trace requires an internal raw prediction")
        if self.kind not in {"terminal", "fact", "activate", "act", "compose", "recall", "noop"}:
            raise DynamicsError("trace kind must be a sanitized causal event kind")
        payload_types = {
            "fact": (LinkFact, HazardFact, SafeFact),
            "activate": (ActivationPayload,),
        }
        if self.kind in payload_types:
            if not isinstance(self.payload, payload_types[self.kind]):
                raise DynamicsError("trace payload must be the immutable public event payload")
        elif self.payload is not None:
            raise DynamicsError("internal and terminal trace payloads must be None")
        for supports in (self.support_before, self.support_after):
            if not isinstance(supports, tuple) or any(
                type(value) is not int or value < 0 for value in supports
            ):
                raise DynamicsError("trace supports must be an immutable tuple of record IDs")
        if not isinstance(self.actions, tuple) or any(
            not isinstance(action, Action) for action in self.actions
        ):
            raise DynamicsError("trace actions must be an immutable action tuple")
        if self.kind == "terminal" and self.segment is not None:
            raise DynamicsError("terminal must not carry a next segment")
        for hypothesis in (self.hypothesis_before, self.hypothesis_after):
            if hypothesis is not None and not isinstance(hypothesis, Hypothesis):
                raise DynamicsError("trace hypothesis must be an immutable Hypothesis")
        if self.tie is not None and not isinstance(self.tie, TieResolution):
            raise DynamicsError("trace tie must be an immutable TieResolution")

    @property
    def tie_id(self) -> str | None:
        return self.tie.tie_id if self.tie else None

    def to_payload(self) -> dict:
        payload = asdict(self)
        payload["tie_id"] = self.tie_id
        return payload


def _causal_counters(counters: ComputeCounters) -> ComputeCounters:
    """Checkpoint materializations are operational diagnostics, not causal work."""
    return ComputeCounters(
        **{
            item.name: 0
            if item.name == "checkpoint_flow_evaluations"
            else getattr(counters, item.name)
            for item in fields(counters)
        }
    )


def _validate_append(events: tuple[CausalEventSummary, ...], event: CausalEventSummary) -> None:
    require_time(event.timestamp, "trace timestamp")
    if any(previous.event_id == event.event_id for previous in events):
        raise DynamicsError("causal trace event IDs must be unique")
    if events:
        previous = events[-1]
        if event.timestamp < previous.timestamp:
            raise TimeOrderError("causal trace timestamps cannot reverse")
        if previous.kind == "terminal":
            raise DynamicsError("no causal event may follow terminal")
        if event.parent_event_id != previous.event_id:
            raise DynamicsError("causal trace parent must be the preceding event")
        if event.delta != event.timestamp - previous.timestamp:
            raise DynamicsError("causal delta must use the preceding event time")
        if event.timestamp == previous.timestamp and (
            causal_priority(event.kind),
            event.event_id,
        ) < (causal_priority(previous.kind), previous.event_id):
            raise DynamicsError("same-time causal events must follow priority and event ID")
    if event.kind == "terminal" and (event.payload is not None or event.segment is not None):
        raise DynamicsError("online terminal may contain no payload or next segment")


@dataclass(frozen=True, slots=True)
class CausalTrace:
    events: tuple[CausalEventSummary, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple):
            raise DynamicsError("causal trace requires an immutable event tuple")
        for index, event in enumerate(self.events):
            _validate_append(self.events[:index], event)

    def to_payload(self) -> dict:
        return {"schema_version": 1, "events": [event.to_payload() for event in self.events]}

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_payload()))


class TraceRecorder:
    """Record only executed jumps; snapshots do not append trace events.

    Counter deltas start at zero and then the previous recorded post-event
    counters, so initialization, prediction and inter-event flow count once.
    A resumed causal prefix restores this baseline. Checkpoint-only flow counts
    remain outside causal JSON; the engine must charge pauses to that counter.
    """

    def __init__(
        self, trace: CausalTrace | None = None, *, initial_counters: ComputeCounters | None = None
    ) -> None:
        self._events = (trace or CausalTrace()).events
        if self._events and initial_counters is not None:
            raise DynamicsError("a trace prefix already supplies the counter baseline")
        self._counters = (
            self._events[-1].counters_after
            if self._events
            else _causal_counters(initial_counters or ComputeCounters())
        )

    def snapshot(self) -> CausalTrace:
        return CausalTrace(self._events)

    def append(self, summary: CausalEventSummary) -> None:
        _validate_append(self._events, summary)
        for item in fields(ComputeCounters):
            if getattr(summary.counters_after, item.name) - getattr(
                self._counters, item.name
            ) != getattr(summary.counter_delta, item.name):
                raise DynamicsError("trace counter delta does not match its prior baseline")
        self._events = (*self._events, summary)
        self._counters = summary.counters_after

    def record(
        self,
        event: Event,
        before: RuntimeState,
        after: RuntimeCore | RuntimeState,
        *,
        selected_record_id: int | None = None,
        selected_rank: int | None = None,
        tie: TieResolution | None = None,
        was_gap_clamped: bool = False,
        raw_predicted_delta: float | None = None,
    ) -> CausalEventSummary:
        """Accept explicit agent selection metadata, never an oracle or private label."""
        post = after.core if isinstance(after, RuntimeState) else after
        if isinstance(after, RuntimeState) and after.time != event.timestamp:
            raise TimeOrderError("post-event state must be materialized at the event time")
        if event.timestamp != before.time or post.last_event_time != event.timestamp:
            raise TimeOrderError("record requires both states at the executed event time")
        if post.last_event_id != event.event_id:
            raise DynamicsError("post-event state does not identify the executed event")
        parent = before.core.last_event_id
        if isinstance(event, InternalEvent) and event.parent_event_id != parent:
            raise DynamicsError("internal event parent does not match the causal origin")
        origin_time = before.core.last_event_time
        if origin_time is None:
            origin_time = before.segment.started_at
        delta = event.timestamp - origin_time
        if delta < 0.0:
            raise TimeOrderError("event precedes its causal origin")
        terminal = event_priority_kind(event) == "terminal"
        if terminal and (post.mode is not Mode.TERMINAL or isinstance(after, RuntimeState)):
            raise DynamicsError("terminal requires TERMINAL core and no new segment")
        if not terminal and not isinstance(after, RuntimeState):
            raise DynamicsError("a causal public jump requires its installed next segment")
        if isinstance(after, RuntimeState) and (
            after.segment.parent_event_id != event.event_id
            or after.segment.started_at != event.timestamp
        ):
            raise DynamicsError("post-event segment must be anchored to the executed event")
        for value in (selected_record_id, selected_rank):
            if value is not None and (type(value) is not int or value < 0):
                raise DynamicsError("selection metadata must be nonnegative integer or None")
        if tie is not None and tie.winner_event_id != event.event_id:
            raise DynamicsError("tie winner must identify the executed event")
        counters_after = _causal_counters(post.counters)
        counter_delta = ComputeCounters(
            **{
                item.name: getattr(counters_after, item.name) - getattr(self._counters, item.name)
                for item in fields(ComputeCounters)
            }
        )
        state_norm, state_hash = _state_summary(post.continuous)
        summary = CausalEventSummary(
            event.event_id,
            parent,
            event.timestamp,
            "terminal" if terminal else event.kind.value,
            event.payload if isinstance(event, ExternalEvent) and not terminal else None,
            before.core.mode,
            post.mode,
            delta,
            selected_record_id,
            selected_rank,
            before.core.support_ids,
            post.support_ids,
            before.core.hypothesis,
            post.hypothesis,
            post.actions[len(before.core.actions) :],
            state_norm,
            state_hash,
            counter_delta,
            counters_after,
            before.segment.prediction_snapshot_sha256,
            SegmentSummary.from_segment(after.segment) if isinstance(after, RuntimeState) else None,
            tie,
            was_gap_clamped,
            raw_predicted_delta,
        )
        self.append(summary)
        return summary


def sanitized_crash_events(trace: CausalTrace) -> tuple[dict, ...]:
    """At most twenty already-sanitized causal records; no private replay envelope."""
    return tuple(event.to_payload() for event in trace.events[-20:])
