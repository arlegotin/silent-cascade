"""Behavioral contracts for immutable public OFD schemas."""

import math

import pytest

from silent_cascade.errors import EpisodeInvariantError
from silent_cascade.schemas import (
    Action,
    ActivationPayload,
    AgentInit,
    Condition,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    Hypothesis,
    InternalEvent,
    InternalEventKind,
    LinkFact,
    MemoryRecord,
    Mode,
    Provenance,
    RecordKind,
    SafeFact,
    fact_event_to_memory_record,
)


@pytest.mark.parametrize(
    ("enum_type", "values"),
    [
        (RecordKind, ("link", "hazard", "safe")),
        (ExternalEventKind, ("fact", "activate", "outcome", "end")),
        (InternalEventKind, ("recall", "compose", "act", "noop")),
        (
            Condition,
            (
                "oracle",
                "random",
                "one_shot_all_memory",
                "ponder_at_activation",
                "flow_only",
                "fixed_tick_recurrent",
                "periodic_matched_count",
                "random_time_matched_count",
                "persistent_frozen",
                "event_flow",
            ),
        ),
        (Provenance, ("perceived", "inferred", "simulated")),
        (
            Mode,
            (
                "observing",
                "searching",
                "have_memory",
                "holding_hazard",
                "quiescent",
                "terminal",
            ),
        ),
    ],
)
def test_public_enum_vocabulary_is_exact(enum_type: type[object], values: tuple[str, ...]) -> None:
    assert tuple(member.value for member in enum_type) == values  # type: ignore[union-attr]


@pytest.mark.parametrize("invalid", [True, -1, 1.5])
def test_link_fact_rejects_non_exact_nonnegative_entity_ids(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        LinkFact(source_node=invalid, target_node=2, confidence=1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("payload", [LinkFact(1, 2), HazardFact(2, 1, 0.5), SafeFact(3)])
def test_fact_events_convert_one_to_one_to_perceived_memory_records(payload: object) -> None:
    event = ExternalEvent(7, 1.5, ExternalEventKind.FACT, payload)  # type: ignore[arg-type]

    record = fact_event_to_memory_record(event)

    assert record.record_id == event.event_id
    assert record.observed_at == event.timestamp
    assert record.confidence == payload.confidence  # type: ignore[union-attr]
    assert record.provenance is Provenance.PERCEIVED
    assert record.support_ids == ()
    assert (
        record.kind,
        record.subject_id,
        record.object_id,
        record.hazard_type,
        record.delay,
    ) == {
        LinkFact: (RecordKind.LINK, 1, 2, None, None),
        HazardFact: (RecordKind.HAZARD, 2, None, 1, 0.5),
        SafeFact: (RecordKind.SAFE, 3, None, None, None),
    }[type(payload)]


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        (ExternalEventKind.FACT, ActivationPayload(1)),
        (ExternalEventKind.ACTIVATE, LinkFact(1, 2)),
        (ExternalEventKind.OUTCOME, SafeFact(1)),
        (ExternalEventKind.END, HazardFact(1, 0, 1.0)),
    ],
)
def test_external_events_reject_invalid_kind_payload_pairings(
    kind: ExternalEventKind, payload: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ExternalEvent(1, 0.0, kind, payload)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", [math.inf, -math.inf, math.nan])
def test_public_times_reject_nonfinite_values(invalid: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        AgentInit("00000000-0000-4000-8000-000000000001", 64, 4, invalid)

    with pytest.raises((TypeError, ValueError)):
        InternalEvent(1, 0, invalid, InternalEventKind.RECALL, 0, 0.1)


def test_records_actions_hypotheses_and_events_are_frozen_with_immutable_tuples() -> None:
    record = MemoryRecord(
        1, RecordKind.LINK, 1, 2, None, None, 0.0, 1.0, Provenance.PERCEIVED, (3, 4)
    )
    hypothesis = Hypothesis(1, 2.0, 1.0, Provenance.INFERRED, (1, 2), False)
    action = Action(1, 1.0, 7)

    with pytest.raises(AttributeError):
        record.confidence = 0.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        hypothesis.support_ids += (3,)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        action.timestamp = 2.0  # type: ignore[misc]


@pytest.mark.parametrize("support_ids", [(), (3, 3)])
def test_inferred_hypotheses_require_nonempty_unique_support_ids(
    support_ids: tuple[int, ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        Hypothesis(1, 2.0, 1.0, Provenance.INFERRED, support_ids, False)


def test_non_fact_event_cannot_become_memory_record() -> None:
    event = ExternalEvent(2, 0.0, ExternalEventKind.ACTIVATE, ActivationPayload(1))

    with pytest.raises(EpisodeInvariantError, match="only FACT"):
        fact_event_to_memory_record(event)
