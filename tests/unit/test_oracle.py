"""Behavioral contracts for the independent facts-derived OFD oracle."""

import inspect
import math
from dataclasses import replace

import numpy as np
import pytest

from silent_cascade.env.config import OracleTimingConfig, SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    EpisodeKey,
    EpisodeRecipe,
    EpisodeTruth,
    EpisodeVariant,
    MatchedEpisodeCoordinate,
    PublicEpisode,
)
from silent_cascade.env.oracle import (
    OraclePolicy,
    OracleSolution,
    OracleTerminalKind,
    build_oracle_trace,
    oracle_actions,
    scale_oracle_trace,
    solve_public_episode,
    verify_oracle_truth,
)
from silent_cascade.errors import OracleError
from silent_cascade.schemas import (
    ActivationPayload,
    AgentInit,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    InternalEventKind,
    LinkFact,
    SafeFact,
)


def public_episode(
    facts: tuple[LinkFact | HazardFact | SafeFact, ...],
    *,
    activation_node: int = 1,
    activation_time: float = 100.0,
) -> PublicEpisode:
    events = tuple(
        ExternalEvent(index + 10, float(index + 1), ExternalEventKind.FACT, fact)
        for index, fact in enumerate(facts)
    )
    return PublicEpisode(
        AgentInit("00000000-0000-4000-8000-000000000005", 64, 4, 0.0),
        (
            *events,
            ExternalEvent(
                len(events) + 10,
                activation_time,
                ExternalEventKind.ACTIVATE,
                ActivationPayload(activation_node),
            ),
        ),
    )


def positive_public() -> PublicEpisode:
    return public_episode(
        (
            LinkFact(7, 2),
            HazardFact(5, 3, 24.0),
            LinkFact(1, 6),
            SafeFact(4),
            LinkFact(2, 5),
            LinkFact(9, 7),
            HazardFact(6, 1, 24.0),
        ),
        activation_node=9,
    )


def positive_truth() -> EpisodeTruth:
    return EpisodeTruth(
        key=EpisodeKey(
            "ofd-v1",
            SplitNamespace.PHASE1_GATE,
            SuiteName.IID_PRIMARY,
            31,
            MatchedEpisodeCoordinate("matched", 1, 0),
        ),
        recipe=EpisodeRecipe(4, EpisodeVariant.POSITIVE, 3, SuiteName.IID_PRIMARY, 0),
        relevant_node_path=(9, 7, 2, 5),
        relevant_record_ids=(15, 10, 14, 11),
        terminal_record_id=11,
        relevant_hazard_type=3,
        private_terminal=ExternalEvent(99, 124.0, ExternalEventKind.OUTCOME, None),
        activation_time=100.0,
        episode_delay=24.0,
        action_window_start=118.0,
        action_window_end=121.6,
        action_target=119.8,
        rejection_count=0,
        rejection_reasons=(),
    )


def test_solver_derives_canonical_path_only_from_public_facts() -> None:
    solution = solve_public_episode(positive_public())

    assert solution == OracleSolution(
        terminal_kind=OracleTerminalKind.HAZARD,
        node_path=(9, 7, 2, 5),
        link_record_ids=(15, 10, 14),
        terminal_record_id=11,
        hazard_type=3,
        public_delay=24.0,
    )
    assert tuple(inspect.signature(solve_public_episode).parameters) == ("public", "policy")


@pytest.mark.parametrize(
    ("facts", "message"),
    [
        ((LinkFact(1, 2), LinkFact(1, 2)), "duplicate equivalent"),
        ((LinkFact(1, 2), LinkFact(1, 3)), "branch"),
        ((LinkFact(1, 2), LinkFact(2, 1)), "cycle"),
        ((LinkFact(1, 2), HazardFact(2, 0, 4.0), SafeFact(2)), "multiple terminals"),
        (
            (LinkFact(1, 2), HazardFact(2, 0, 4.0), HazardFact(2, 1, 4.0)),
            "multiple terminals",
        ),
    ],
)
def test_primary_solver_rejects_nonunique_legal_traces(
    facts: tuple[LinkFact | HazardFact | SafeFact, ...], message: str
) -> None:
    with pytest.raises(OracleError, match=message):
        solve_public_episode(public_episode(facts))


def test_solver_rejects_corrupt_fact_payload() -> None:
    corrupt = object.__new__(ExternalEvent)
    object.__setattr__(corrupt, "event_id", 10)
    object.__setattr__(corrupt, "timestamp", 1.0)
    object.__setattr__(corrupt, "kind", ExternalEventKind.FACT)
    object.__setattr__(corrupt, "payload", ActivationPayload(2))
    public = PublicEpisode(
        AgentInit("00000000-0000-4000-8000-000000000006", 64, 4, 0.0),
        (
            corrupt,
            ExternalEvent(11, 100.0, ExternalEventKind.ACTIVATE, ActivationPayload(1)),
        ),
    )

    with pytest.raises(OracleError, match="invalid FACT payload"):
        solve_public_episode(public)


def test_primary_solver_does_not_implement_future_stress_policies() -> None:
    with pytest.raises(OracleError, match="not implemented"):
        solve_public_episode(positive_public(), OraclePolicy.BRANCHING)
    with pytest.raises(OracleError, match="not implemented"):
        solve_public_episode(positive_public(), OraclePolicy.CONTRADICTION)


def test_truth_verification_fails_loudly_without_repairing_private_truth() -> None:
    solution = solve_public_episode(positive_public())
    truth = positive_truth()

    verify_oracle_truth(solution, truth)
    corrupt = replace(truth, relevant_node_path=(9, 7, 2, 6))
    with pytest.raises(OracleError, match="truth mismatch"):
        verify_oracle_truth(solution, corrupt)
    assert corrupt.relevant_node_path == (9, 7, 2, 6)


def test_trace_records_support_focus_parentage_and_action_target() -> None:
    public = positive_public()
    solution = solve_public_episode(public)
    terminal = ExternalEvent(99, 124.0, ExternalEventKind.OUTCOME, None)
    trace = build_oracle_trace(
        public,
        solution,
        terminal,
        OracleTimingConfig(jitter_log_std=0.0),
        np.random.default_rng(1),
    )

    assert [step.kind for step in trace.steps] == [
        InternalEventKind.RECALL,
        InternalEventKind.COMPOSE,
        InternalEventKind.RECALL,
        InternalEventKind.COMPOSE,
        InternalEventKind.RECALL,
        InternalEventKind.COMPOSE,
        InternalEventKind.RECALL,
        InternalEventKind.COMPOSE,
        InternalEventKind.ACT,
    ]
    assert [step.selected_record_id for step in trace.steps] == [
        15,
        15,
        10,
        10,
        14,
        14,
        11,
        11,
        None,
    ]
    assert [(step.focus_before, step.focus_after) for step in trace.steps[:-1]] == [
        (9, 9),
        (9, 7),
        (7, 7),
        (7, 2),
        (2, 2),
        (2, 5),
        (5, 5),
        (5, 5),
    ]
    assert [step.trace_step_id for step in trace.steps] == list(range(9))
    assert [step.parent_trace_step_id for step in trace.steps] == [
        None,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    assert trace.steps[-1].timestamp == 119.8
    assert trace.steps[-1].delta == 119.8 - trace.steps[-2].timestamp
    assert trace.actions[0].timestamp == 119.8
    assert trace.actions[0].caused_by_event_id == 8
    assert oracle_actions(trace) == trace.actions


def test_nondefault_timing_controls_compression_target_and_feasibility() -> None:
    public = positive_public()
    solution = solve_public_episode(public)
    terminal = ExternalEvent(99, 103.0, ExternalEventKind.OUTCOME, None)
    timing = OracleTimingConfig(
        delta_0=0.6,
        delta_min=0.1,
        delta_max=1.2,
        jitter_log_std=0.0,
        terminal_compose_fraction=0.4,
        action_window_start_fraction=0.6,
        action_target_fraction=0.7,
        action_window_end_fraction=0.8,
    )
    compressed = build_oracle_trace(
        public, replace(solution, public_delay=3.0), terminal, timing, np.random.default_rng(2)
    )

    assert math.isclose(sum(step.delta for step in compressed.steps[:-1]), 1.2)
    assert all(step.delta >= 0.1 for step in compressed.steps[:-1])
    assert compressed.steps[-2].timestamp == pytest.approx(101.2)
    assert compressed.actions[0].timestamp == 102.1
    assert 101.8 <= compressed.actions[0].timestamp < 102.4

    infeasible_timing = timing.model_copy(
        update={"delta_min": 0.5, "delta_0": 0.5, "terminal_compose_fraction": 0.2}
    )
    with pytest.raises(OracleError, match="temporally infeasible"):
        build_oracle_trace(
            public_episode((LinkFact(1, 2), LinkFact(2, 3))),
            OracleSolution(
                OracleTerminalKind.DISCONNECTED,
                (1, 2, 3),
                (10, 11),
                None,
                None,
                None,
            ),
            ExternalEvent(99, 105.0, ExternalEventKind.END, None),
            infeasible_timing,
            np.random.default_rng(3),
        )


def test_scale_trace_multiplies_only_temporal_values() -> None:
    public = positive_public()
    trace = build_oracle_trace(
        public,
        solve_public_episode(public),
        ExternalEvent(99, 124.0, ExternalEventKind.OUTCOME, None),
        OracleTimingConfig(jitter_log_std=0.0),
        np.random.default_rng(4),
    )

    scaled = scale_oracle_trace(trace, 10.0)

    assert scaled.solution == trace.solution
    assert [step.timestamp for step in scaled.steps] == [
        step.timestamp * 10.0 for step in trace.steps
    ]
    assert [step.delta for step in scaled.steps] == [step.delta * 10.0 for step in trace.steps]
    assert [action.timestamp for action in scaled.actions] == [
        action.timestamp * 10.0 for action in trace.actions
    ]
    assert [
        (
            step.kind,
            step.selected_record_id,
            step.focus_before,
            step.focus_after,
        )
        for step in scaled.steps
    ] == [
        (
            step.kind,
            step.selected_record_id,
            step.focus_before,
            step.focus_after,
        )
        for step in trace.steps
    ]


@pytest.mark.parametrize("factor", [0.0, -1.0, math.inf, math.nan])
def test_scale_trace_rejects_invalid_factors(factor: float) -> None:
    public = positive_public()
    trace = build_oracle_trace(
        public,
        solve_public_episode(public),
        ExternalEvent(99, 124.0, ExternalEventKind.OUTCOME, None),
        OracleTimingConfig(),
        np.random.default_rng(4),
    )

    with pytest.raises(OracleError, match="scale factor"):
        scale_oracle_trace(trace, factor)
