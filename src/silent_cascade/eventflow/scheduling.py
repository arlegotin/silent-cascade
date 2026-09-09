"""Engine-private queue, prediction cache and pure host-float tie selection.

None of these objects is an agent input. Queue/cache snapshots are private
checkpoint material; selection never consumes an event or reserves an ID.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from silent_cascade.errors import DynamicsError, TimeOrderError
from silent_cascade.eventflow.state import INTERNAL_EVENT_ID_BASE, RuntimeState, require_time
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.schemas import (
    Condition,
    ExternalEvent,
    ExternalEventKind,
    InternalEvent,
    InternalEventKind,
)

if TYPE_CHECKING:
    from silent_cascade.env.episode import PublicEpisode

NEAR_TIE_TOLERANCE = 1e-9
_TERMINAL = (ExternalEventKind.OUTCOME, ExternalEventKind.END)
_PRIORITY = {"terminal": 0, "external": 1, "act": 2, "compose": 3, "recall": 4, "noop": 5}
type Event = ExternalEvent | InternalEvent


def causal_priority(kind: str) -> int:
    """Shared priority of sanitized scheduling or trace kinds."""
    return _PRIORITY["external" if kind in {"fact", "activate"} else kind]


def event_priority_kind(event: Event) -> str:
    """Return the public-safe priority category, masking private terminal type."""
    if isinstance(event, ExternalEvent):
        return "terminal" if event.kind in _TERMINAL else "external"
    return event.kind.value


class ExternalEventQueue:
    """Engine-owned heap; the explicit snapshot must never be passed to an agent."""

    __slots__ = ("_heap",)

    def __init__(self, episode: PublicEpisode, terminal: ExternalEvent) -> None:
        if terminal.kind not in _TERMINAL:
            raise DynamicsError("queue requires one private terminal")
        events = (*episode.events, terminal)
        if len({event.event_id for event in events}) != len(events):
            raise DynamicsError("queue event IDs must be unique")
        for event in events:
            require_time(event.timestamp, "external timestamp")
            if event.event_id >= INTERNAL_EVENT_ID_BASE:
                raise DynamicsError("external ID occupies the internal namespace")
        self._heap = [
            (event.timestamp, causal_priority(event_priority_kind(event)), event.event_id, event)
            for event in events
        ]
        heapq.heapify(self._heap)

    def snapshot(self) -> tuple[ExternalEvent, ...]:
        """Private, immutable chronological checkpoint snapshot."""
        return tuple(item[3] for item in sorted(self._heap))

    def consume(self, event: ExternalEvent) -> None:
        """Remove exactly the selected external at execution, never during lookahead."""
        for index, item in enumerate(self._heap):
            if item[3] == event:
                self._heap[index] = self._heap[-1]
                self._heap.pop()
                heapq.heapify(self._heap)
                return
        raise DynamicsError("external event is not pending")


@dataclass(frozen=True, slots=True)
class PredictionSnapshot:
    parent_event_id: int
    prediction_snapshot_sha256: str
    event: InternalEvent | None


_UNCOMPUTED = object()


class PredictionCache:
    """Cache one prediction or explicit dormancy for one unchanged causal segment."""

    __slots__ = ("_value",)

    def __init__(self) -> None:
        self._value: PredictionSnapshot | object = _UNCOMPUTED

    @property
    def is_computed(self) -> bool:
        return self._value is not _UNCOMPUTED

    def snapshot(self) -> PredictionSnapshot | None:
        """None means uncomputed; a snapshot whose event is None means dormant."""
        return self._value if isinstance(self._value, PredictionSnapshot) else None

    def invalidate(self) -> None:
        """Engine calls after an executed event, never for a noncausal pause."""
        self._value = _UNCOMPUTED

    def get_or_predict(
        self, state: RuntimeState, predictor: Callable[[RuntimeState], InternalEvent | None]
    ) -> InternalEvent | None:
        segment = state.segment
        previous = self.snapshot()
        if previous is not None and (
            previous.parent_event_id == segment.parent_event_id
            and previous.prediction_snapshot_sha256 == segment.prediction_snapshot_sha256
        ):
            return previous.event
        self.invalidate()
        prediction = predictor(state)
        if prediction is not None:
            if not isinstance(prediction, InternalEvent):
                raise DynamicsError("prediction must be an InternalEvent or dormant None")
            if prediction.parent_event_id != segment.parent_event_id:
                raise DynamicsError("prediction parent does not match the segment")
            if prediction.timestamp < state.time:
                raise TimeOrderError("prediction precedes the runtime cursor")
        self._value = PredictionSnapshot(
            segment.parent_event_id, segment.prediction_snapshot_sha256, prediction
        )
        return prediction


@dataclass(frozen=True, slots=True)
class TieCandidate:
    event_id: int
    timestamp: float
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in _PRIORITY:
            raise DynamicsError("tie candidate requires a sanitized priority kind")
        require_time(self.timestamp, "tie candidate timestamp")
        if type(self.event_id) is not int or self.event_id < 0:
            raise DynamicsError("tie candidate ID must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class TieResolution:
    candidates: tuple[TieCandidate, ...]
    winner_event_id: int
    tolerance: float = NEAR_TIE_TOLERANCE

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple) or any(
            not isinstance(candidate, TieCandidate) for candidate in self.candidates
        ):
            raise DynamicsError("tie candidates must be an immutable candidate tuple")
        candidate_ids = {candidate.event_id for candidate in self.candidates}
        if len(self.candidates) < 2 or len(candidate_ids) != len(self.candidates):
            raise DynamicsError("tie requires at least two candidates with unique IDs")
        if type(self.winner_event_id) is not int or self.winner_event_id not in candidate_ids:
            raise DynamicsError("tie winner must be an exact integer ID in the candidate set")
        if type(self.tolerance) is not float or self.tolerance != NEAR_TIE_TOLERANCE:
            raise DynamicsError("tie tolerance must be the declared host float 1e-9")

    @property
    def tie_id(self) -> str:
        return sha256_bytes(canonical_json_bytes(asdict(self)))


@dataclass(frozen=True, slots=True)
class ScheduledChoice:
    event: Event
    timestamp: float
    tie: TieResolution | None = None


def choose_next_event(
    queue: ExternalEventQueue,
    internal: InternalEvent | tuple[InternalEvent, ...] | None = None,
    *,
    condition: Condition = Condition.EVENT_FLOW,
    current_time: float = 0.0,
) -> ScheduledChoice | None:
    """Choose without mutation from the earliest timestamp's tolerance cluster.

    Compare host floats, retain original timestamps, then priority, time and ID.
    Scan the heap's cluster too: a near-tied terminal cannot hide behind its root.
    Clusters are anchored to the earliest time (tolerance is not transitive).
    NOOP is rejected for EventFlow before considering who wins.
    """
    require_time(current_time, "current_time")
    internals = (
        () if internal is None else (internal,) if isinstance(internal, InternalEvent) else internal
    )
    if not isinstance(condition, Condition):
        raise DynamicsError("condition must be a Condition")
    if any(not isinstance(event, InternalEvent) for event in internals):
        raise DynamicsError("internal candidates must be InternalEvent values")
    if condition is Condition.EVENT_FLOW and any(
        event.kind is InternalEventKind.NOOP for event in internals
    ):
        raise DynamicsError("EventFlow rejects NOOP, including losing predictions")
    candidates = (*queue.snapshot(), *internals)
    if not candidates:
        return None
    if len({event.event_id for event in candidates}) != len(candidates):
        raise DynamicsError("candidate event IDs must be unique")
    for event in candidates:
        require_time(event.timestamp, "candidate timestamp")
        if event.timestamp < current_time:
            raise TimeOrderError("candidate precedes the runtime cursor")
    first = min(event.timestamp for event in candidates)
    tied = sorted(
        (event for event in candidates if event.timestamp - first <= NEAR_TIE_TOLERANCE),
        key=lambda event: (
            causal_priority(event_priority_kind(event)),
            event.timestamp,
            event.event_id,
        ),
    )
    winner = tied[0]
    tie = None
    if len(tied) > 1:
        tie = TieResolution(
            tuple(
                TieCandidate(event.event_id, event.timestamp, event_priority_kind(event))
                for event in tied
            ),
            winner.event_id,
        )
    return ScheduledChoice(winner, winner.timestamp, tie)
