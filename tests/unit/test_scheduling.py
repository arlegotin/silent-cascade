"""Private scheduling, complete ties and once-per-segment predictions."""

from dataclasses import replace
from itertools import combinations

import pytest
from test_jumps import runtime

from silent_cascade.env.episode import PublicEpisode
from silent_cascade.errors import DynamicsError, TimeOrderError
from silent_cascade.eventflow.flow import advance_to
from silent_cascade.eventflow.jumps import begin_post_jump_segment
from silent_cascade.eventflow.scheduling import (
    ExternalEventQueue,
    PredictionCache,
    TieCandidate,
    TieResolution,
    choose_next_event,
)
from silent_cascade.eventflow.state import INTERNAL_EVENT_ID_BASE
from silent_cascade.schemas import (
    ActivationPayload,
    AgentInit,
    Condition,
    ExternalEvent,
    ExternalEventKind,
    InternalEvent,
    InternalEventKind,
    LinkFact,
    Mode,
)


def queue(time: float = 2.0, terminal_time: float = 10.0) -> ExternalEventQueue:
    public = PublicEpisode(
        AgentInit("public", 64, 4, 0.0),
        (ExternalEvent(1, time, ExternalEventKind.ACTIVATE, ActivationPayload(0)),),
    )
    return ExternalEventQueue(public, ExternalEvent(2, terminal_time, ExternalEventKind.END, None))


def internal(kind: InternalEventKind, time: float, event_id: int = 0) -> InternalEvent:
    return InternalEvent(INTERNAL_EVENT_ID_BASE + event_id, 3, time, kind, 0, 1.0)


def test_queue_pops_only_selected_event_and_preserves_public_episode() -> None:
    public = PublicEpisode(
        AgentInit("public", 64, 4, 0.0),
        (ExternalEvent(1, 2.0, ExternalEventKind.ACTIVATE, ActivationPayload(0)),),
    )
    terminal = ExternalEvent(2, 10.0, ExternalEventKind.END, None)
    pending = ExternalEventQueue(public, terminal)
    snapshot = pending.snapshot()
    predicted = internal(InternalEventKind.RECALL, 1.0)
    assert choose_next_event(pending, predicted).event == predicted
    assert pending.snapshot() == snapshot
    assert choose_next_event(pending).event == public.events[0]
    assert pending.snapshot() == snapshot
    pending.consume(public.events[0])
    assert choose_next_event(pending).event == terminal
    pending.consume(terminal)
    assert choose_next_event(pending) is None
    assert public.events == snapshot[:-1]
    with pytest.raises(TypeError):
        iter(pending)


def test_queue_rejects_duplicate_terminal_id_and_public_terminal_substitute() -> None:
    public = PublicEpisode(
        AgentInit("public", 64, 4, 0.0),
        (ExternalEvent(1, 2.0, ExternalEventKind.ACTIVATE, ActivationPayload(0)),),
    )
    with pytest.raises(DynamicsError):
        ExternalEventQueue(public, ExternalEvent(1, 10.0, ExternalEventKind.END, None))
    with pytest.raises(DynamicsError):
        ExternalEventQueue(public, public.events[0])


PRIORITY = (
    "terminal",
    "external",
    *(
        InternalEventKind.ACT,
        InternalEventKind.COMPOSE,
        InternalEventKind.RECALL,
        InternalEventKind.NOOP,
    ),
)


@pytest.mark.parametrize("higher,lower", list(combinations(PRIORITY, 2)))
@pytest.mark.parametrize("offset", [-0.5e-9, 0.0, 0.5e-9, 1.1e-9])
def test_complete_tie_matrix_records_original_times(higher, lower, offset: float) -> None:
    higher_time, lower_time = 2.0 + offset, 2.0
    public_time = higher_time if higher == "external" else lower_time
    pending = queue(public_time if "external" in (higher, lower) else 20.0, 30.0)
    if higher == "terminal":
        pending = queue(public_time if lower == "external" else 20.0, higher_time)
    candidates = tuple(
        internal(kind, time, index)
        for index, (kind, time) in enumerate(((higher, higher_time), (lower, lower_time)))
        if isinstance(kind, InternalEventKind)
    )
    choice = choose_next_event(pending, candidates, condition=Condition.FIXED_TICK)
    want = lower if offset == 1.1e-9 else higher
    actual = choice.event.kind
    actual = "terminal" if actual is ExternalEventKind.END else actual
    actual = "external" if actual is ExternalEventKind.ACTIVATE else actual
    assert actual == want
    assert choice.timestamp == (lower_time if want == lower else higher_time)
    if offset == 1.1e-9:
        assert choice.tie is None
    else:
        assert choice.tie.tolerance == 1e-9
        assert choice.tie.winner_event_id == choice.event.event_id
        assert sorted(item.timestamp for item in choice.tie.candidates) == sorted(
            [higher_time, lower_time]
        )
        assert len(choice.tie.tie_id) == 64


@pytest.mark.parametrize("time", [1.0, 2.0, 100.0])
def test_eventflow_rejects_noop_even_when_it_loses(time: float) -> None:
    pending = queue()
    before = pending.snapshot()
    with pytest.raises(DynamicsError, match="NOOP"):
        choose_next_event(pending, internal(InternalEventKind.NOOP, time))
    assert pending.snapshot() == before


def test_selection_never_reverses_time_and_does_not_mutate_on_error() -> None:
    pending = queue()
    with pytest.raises(TimeOrderError):
        choose_next_event(pending, current_time=3.0)
    assert len(pending.snapshot()) == 2
    with pytest.raises(TimeOrderError):
        choose_next_event(pending, current_time=2.0 + 0.5e-9)
    assert choose_next_event(pending, current_time=2.0).timestamp == 2.0


@pytest.mark.parametrize("dormant", [False, True])
def test_cache_binds_parent_and_snapshot_and_survives_pause(dormant: bool) -> None:
    seed = runtime(Mode.SEARCHING)
    state = begin_post_jump_segment(
        replace(seed.core, last_event_time=seed.time), seed.segment.parameters, time=seed.time
    )
    cache = PredictionCache()
    calls = []

    def predict(current):
        calls.append(current.segment.parent_event_id)
        return (
            None
            if dormant
            else replace(internal(InternalEventKind.RECALL, 5.0), predicted_delta=4.0)
        )

    expected = cache.get_or_predict(state, predict)
    paused = advance_to(state, 1.5)
    assert cache.get_or_predict(paused, predict) == expected
    assert calls == [3]
    pending = queue()
    assert choose_next_event(pending, expected).event.event_id == 1
    cache.invalidate()
    changed = begin_post_jump_segment(
        replace(state.core, last_event_id=1), state.segment.parameters, time=state.time
    )
    cache.get_or_predict(changed, lambda current: calls.append(1))
    assert calls == [3, 1]
    changed = begin_post_jump_segment(
        changed.core,
        replace(changed.segment.parameters, guard_rates=changed.segment.parameters.guard_rates * 2),
        time=changed.time,
    )
    cache.get_or_predict(changed, lambda current: calls.append(1))
    assert calls == [3, 1, 1]


def test_cache_rejects_prediction_for_another_parent_without_caching_failure() -> None:
    cache = PredictionCache()
    with pytest.raises(DynamicsError):
        cache.get_or_predict(
            runtime(),
            lambda state: replace(internal(InternalEventKind.ACT, 2.0), parent_event_id=99),
        )
    assert not cache.is_computed


def test_near_tied_same_priority_events_keep_time_order_after_consumption() -> None:
    public = PublicEpisode(
        AgentInit("public", 64, 4, 0.0),
        (
            ExternalEvent(2, 2.0, ExternalEventKind.FACT, LinkFact(0, 1)),
            ExternalEvent(1, 2.0 + 0.5e-9, ExternalEventKind.FACT, LinkFact(1, 2)),
            ExternalEvent(3, 3.0, ExternalEventKind.ACTIVATE, ActivationPayload(0)),
        ),
    )
    pending = ExternalEventQueue(public, ExternalEvent(4, 10.0, ExternalEventKind.END, None))
    first = choose_next_event(pending)
    assert first.event.event_id == 2
    pending.consume(first.event)
    second = choose_next_event(pending, current_time=first.timestamp)
    assert second.event.event_id == 1
    assert second.timestamp > first.timestamp
    pending.consume(second.event)
    assert choose_next_event(pending, current_time=second.timestamp).event.event_id == 3


def test_exact_time_same_priority_events_use_event_id() -> None:
    candidates = (
        internal(InternalEventKind.RECALL, 1.0, 2),
        internal(InternalEventKind.RECALL, 1.0, 1),
    )
    assert choose_next_event(queue(), candidates).event.event_id == INTERNAL_EVENT_ID_BASE + 1


@pytest.mark.parametrize("winner", [99, True, 1.0, -1, None])
def test_tie_requires_exact_integer_winner_in_candidate_set(winner) -> None:
    candidates = (TieCandidate(1, 2.0, "external"), TieCandidate(2, 2.0, "recall"))
    with pytest.raises(DynamicsError):
        TieResolution(candidates, winner)


@pytest.mark.parametrize(
    "candidates",
    [
        (),
        (TieCandidate(1, 2.0, "external"),),
        (TieCandidate(1, 2.0, "external"), TieCandidate(1, 2.0, "recall")),
    ],
)
def test_tie_requires_at_least_two_unique_candidates(candidates) -> None:
    with pytest.raises(DynamicsError):
        TieResolution(candidates, 1)


@pytest.mark.parametrize("tolerance", [0.0, 1.1e-9, -1e-9, 1, True, float("nan"), float("inf")])
def test_tie_requires_exact_declared_host_float_tolerance(tolerance) -> None:
    candidates = (TieCandidate(1, 2.0, "external"), TieCandidate(2, 2.0, "recall"))
    with pytest.raises(DynamicsError):
        TieResolution(candidates, 1, tolerance)
