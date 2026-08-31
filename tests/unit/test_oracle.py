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
    OracleTrace,
    OracleTraceStep,
    build_oracle_trace,
    oracle_actions,
    scale_oracle_trace,
    solve_public_episode,
    verify_oracle_truth,
)
from silent_cascade.errors import OracleError
from silent_cascade.schemas import (
    Action,
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


def negative_origin_episode(
    facts: tuple[LinkFact | HazardFact | SafeFact, ...],
) -> PublicEpisode:
    events = tuple(
        ExternalEvent(index + 10, -19.0 + index, ExternalEventKind.FACT, fact)
        for index, fact in enumerate(facts)
    )
    return PublicEpisode(
        AgentInit("00000000-0000-4000-8000-000000000008", 64, 4, -20.0),
        (
            *events,
            ExternalEvent(
                len(events) + 10,
                -10.0,
                ExternalEventKind.ACTIVATE,
                ActivationPayload(1),
            ),
        ),
    )


def positive_public(*, delay: float = 24.0) -> PublicEpisode:
    return public_episode(
        (
            LinkFact(7, 2),
            HazardFact(5, 3, delay),
            LinkFact(1, 6),
            SafeFact(4),
            LinkFact(2, 5),
            LinkFact(9, 7),
            HazardFact(6, 1, delay),
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


def unchecked_solution(solution: OracleSolution, **updates: object) -> OracleSolution:
    """Bypass construction solely to exercise public boundary revalidation."""
    corrupt = object.__new__(OracleSolution)
    for field in (
        "terminal_kind",
        "node_path",
        "link_record_ids",
        "terminal_record_id",
        "hazard_type",
        "public_delay",
        "superseded_terminal_record_ids",
    ):
        object.__setattr__(corrupt, field, updates.get(field, getattr(solution, field)))
    return corrupt


def built_positive_trace() -> OracleTrace:
    public = positive_public()
    return build_oracle_trace(
        public,
        solve_public_episode(public),
        ExternalEvent(99, 124.0, ExternalEventKind.OUTCOME, None),
        OracleTimingConfig(jitter_log_std=0.0),
        np.random.default_rng(4),
    )


def built_negative_trace() -> OracleTrace:
    public = public_episode((LinkFact(1, 2), SafeFact(2)))
    return build_oracle_trace(
        public,
        solve_public_episode(public),
        ExternalEvent(99, 124.0, ExternalEventKind.END, None),
        OracleTimingConfig(jitter_log_std=0.0),
        np.random.default_rng(5),
    )


def built_disconnected_trace() -> OracleTrace:
    public = public_episode((LinkFact(1, 2), LinkFact(2, 3), SafeFact(9)))
    return build_oracle_trace(
        public,
        solve_public_episode(public),
        ExternalEvent(99, 124.0, ExternalEventKind.END, None),
        OracleTimingConfig(jitter_log_std=0.0),
        np.random.default_rng(6),
    )


def unchecked_trace(
    trace: OracleTrace,
    *,
    steps: tuple[OracleTraceStep, ...] | None = None,
    actions: tuple[Action, ...] | None = None,
) -> OracleTrace:
    """Bypass aggregate construction solely to test boundary revalidation."""
    corrupt = object.__new__(OracleTrace)
    object.__setattr__(corrupt, "solution", trace.solution)
    object.__setattr__(corrupt, "steps", trace.steps if steps is None else steps)
    object.__setattr__(corrupt, "actions", trace.actions if actions is None else actions)
    return corrupt


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


@pytest.mark.parametrize(
    "facts",
    [
        (LinkFact(1, 2), SafeFact(2), LinkFact(2, 3), HazardFact(3, 1, 10.0)),
        (LinkFact(1, 2), SafeFact(2), LinkFact(2, 3), LinkFact(2, 4)),
        (LinkFact(1, 2), HazardFact(2, 1, 10.0), LinkFact(2, 1)),
    ],
    ids=("sequential-terminals", "terminal-to-branch", "terminal-to-cycle"),
)
def test_primary_solver_rejects_any_reachable_continuation_after_terminal(
    facts: tuple[LinkFact | HazardFact | SafeFact, ...],
) -> None:
    with pytest.raises(OracleError, match=r"terminal.*continuation"):
        solve_public_episode(public_episode(facts))


def test_primary_solver_ignores_unreachable_branches_and_cycles() -> None:
    public = public_episode(
        (
            LinkFact(1, 2),
            SafeFact(2),
            LinkFact(30, 31),
            LinkFact(30, 32),
            LinkFact(40, 41),
            LinkFact(41, 40),
        )
    )

    assert solve_public_episode(public) == OracleSolution(
        OracleTerminalKind.SAFE,
        (1, 2),
        (10,),
        11,
        None,
        None,
    )


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


@pytest.mark.parametrize(
    "updates",
    [
        {"hazard_type": 2},
        {"hazard_type": None},
        {"public_delay": 12.0},
        {"public_delay": None},
    ],
    ids=("wrong-hazard", "missing-hazard", "wrong-delay", "missing-delay"),
)
def test_truth_verification_rejects_wrong_or_missing_positive_hazard_fields(
    updates: dict[str, object],
) -> None:
    solution = unchecked_solution(solve_public_episode(positive_public()), **updates)

    with pytest.raises(OracleError):
        verify_oracle_truth(solution, positive_truth())


@pytest.mark.parametrize(
    ("updates", "terminal_time"),
    [
        ({"hazard_type": 2}, 124.0),
        ({"hazard_type": None}, 124.0),
        ({"public_delay": 12.0}, 112.0),
        ({"public_delay": None}, 124.0),
    ],
    ids=("wrong-hazard", "missing-hazard", "wrong-delay", "missing-delay"),
)
def test_trace_builder_binds_hazard_fields_to_selected_public_fact(
    updates: dict[str, object], terminal_time: float
) -> None:
    public = positive_public()
    solution = unchecked_solution(solve_public_episode(public), **updates)

    with pytest.raises(OracleError):
        build_oracle_trace(
            public,
            solution,
            ExternalEvent(99, terminal_time, ExternalEventKind.OUTCOME, None),
            OracleTimingConfig(),
            np.random.default_rng(19),
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"terminal_kind": "hazard"},
        {"node_path": [9, 7, 2, 5]},
        {"node_path": (9, 7, 2, True)},
        {"link_record_ids": (15, 10, True)},
        {"terminal_record_id": True},
        {"hazard_type": True},
        {"hazard_type": 4},
        {"public_delay": 24},
        {"public_delay": 0.0},
        {"public_delay": -1.0},
        {"public_delay": math.nan},
        {"public_delay": math.inf},
        {"superseded_terminal_record_ids": (True,)},
    ],
)
def test_oracle_solution_requires_exact_finite_declared_values(
    updates: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "terminal_kind": OracleTerminalKind.HAZARD,
        "node_path": (9, 7, 2, 5),
        "link_record_ids": (15, 10, 14),
        "terminal_record_id": 11,
        "hazard_type": 3,
        "public_delay": 24.0,
        "superseded_terminal_record_ids": (),
    }
    values.update(updates)

    with pytest.raises((TypeError, ValueError)):
        OracleSolution(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "values",
    [
        (OracleTerminalKind.HAZARD, 11, None, 24.0),
        (OracleTerminalKind.HAZARD, 11, 3, None),
        (OracleTerminalKind.SAFE, 11, 3, None),
        (OracleTerminalKind.SAFE, 11, None, 24.0),
        (OracleTerminalKind.DISCONNECTED, 11, None, None),
    ],
)
def test_oracle_solution_enforces_terminal_discriminants(
    values: tuple[OracleTerminalKind, int | None, int | None, float | None],
) -> None:
    terminal_kind, terminal_record_id, hazard_type, public_delay = values

    with pytest.raises(ValueError):
        OracleSolution(
            terminal_kind,
            (1, 2),
            (10,),
            terminal_record_id,
            hazard_type,
            public_delay,
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"trace_step_id": True},
        {"parent_trace_step_id": True},
        {"kind": "recall"},
        {"timestamp": 100},
        {"timestamp": math.nan},
        {"timestamp": math.inf},
        {"delta": True},
        {"delta": -0.1},
        {"delta": math.nan},
        {"delta": math.inf},
        {"selected_record_id": True},
        {"focus_before": True},
        {"focus_after": True},
    ],
)
def test_oracle_trace_step_requires_exact_finite_declared_values(
    updates: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "trace_step_id": 0,
        "parent_trace_step_id": None,
        "kind": InternalEventKind.RECALL,
        "timestamp": 100.1,
        "delta": 0.1,
        "selected_record_id": 10,
        "focus_before": 1,
        "focus_after": 1,
    }
    values.update(updates)

    with pytest.raises((TypeError, ValueError)):
        OracleTraceStep(**values)  # type: ignore[arg-type]


def test_oracle_trace_revalidates_contained_steps_and_actions() -> None:
    positive = built_positive_trace()
    negative = built_negative_trace()
    corrupt_step = object.__new__(OracleTraceStep)
    for field in (
        "trace_step_id",
        "parent_trace_step_id",
        "kind",
        "timestamp",
        "delta",
        "selected_record_id",
        "focus_before",
        "focus_after",
    ):
        object.__setattr__(corrupt_step, field, getattr(negative.steps[0], field))
    object.__setattr__(corrupt_step, "timestamp", math.inf)
    corrupt_action = object.__new__(Action)
    object.__setattr__(corrupt_action, "hazard_type", 3)
    object.__setattr__(corrupt_action, "timestamp", math.nan)
    object.__setattr__(corrupt_action, "caused_by_event_id", positive.steps[-1].trace_step_id)

    with pytest.raises((TypeError, ValueError)):
        OracleTrace(negative.solution, (corrupt_step, *negative.steps[1:]), ())
    with pytest.raises((TypeError, ValueError)):
        OracleTrace(positive.solution, positive.steps, (corrupt_action,))
    with pytest.raises(ValueError):
        OracleTrace(positive.solution, positive.steps[:-1], ())
    with pytest.raises(ValueError):
        OracleTrace(negative.solution, (*negative.steps, positive.steps[-1]), positive.actions)


@pytest.mark.parametrize("positive", [True, False])
def test_negative_absolute_time_origin_builds_actions_and_scales(positive: bool) -> None:
    terminal_fact: HazardFact | SafeFact
    terminal_kind: ExternalEventKind
    if positive:
        terminal_fact = HazardFact(2, 2, 5.0)
        terminal_kind = ExternalEventKind.OUTCOME
    else:
        terminal_fact = SafeFact(2)
        terminal_kind = ExternalEventKind.END
    public = negative_origin_episode((LinkFact(1, 2), terminal_fact))
    solution = solve_public_episode(public)
    trace = build_oracle_trace(
        public,
        solution,
        ExternalEvent(99, -5.0, terminal_kind, None),
        OracleTimingConfig(jitter_log_std=0.0),
        np.random.default_rng(20),
    )

    assert all(
        type(step.timestamp) is float and math.isfinite(step.timestamp) for step in trace.steps
    )
    assert all(step.timestamp < 0.0 and step.delta >= 0.0 for step in trace.steps)
    if positive:
        assert trace.actions == (Action(2, -5.875, trace.steps[-1].trace_step_id),)
    else:
        assert trace.actions == ()

    scaled = scale_oracle_trace(trace, 2.0)
    assert [step.timestamp for step in scaled.steps] == [
        step.timestamp * 2.0 for step in trace.steps
    ]
    assert [step.delta for step in scaled.steps] == [step.delta * 2.0 for step in trace.steps]
    assert [action.timestamp for action in scaled.actions] == [
        action.timestamp * 2.0 for action in trace.actions
    ]


@pytest.mark.parametrize("variant", ["positive", "safe", "disconnected"])
def test_solution_derived_trace_grammar_accepts_every_primary_variant(variant: str) -> None:
    trace = {
        "positive": built_positive_trace,
        "safe": built_negative_trace,
        "disconnected": built_disconnected_trace,
    }[variant]()

    assert oracle_actions(trace) == trace.actions
    assert scale_oracle_trace(trace, 2.0).solution == trace.solution


def test_trace_constructor_rejects_missing_noop_and_act_only_grammars() -> None:
    safe = built_negative_trace()
    positive = built_positive_trace()
    noop = OracleTraceStep(0, None, InternalEventKind.NOOP, 100.1, 0.1, None, None, None)
    act = OracleTraceStep(0, None, InternalEventKind.ACT, 119.8, 19.8, None, None, None)

    with pytest.raises(ValueError, match="grammar"):
        OracleTrace(safe.solution, (), ())
    with pytest.raises(ValueError, match="grammar"):
        OracleTrace(safe.solution, (noop,), ())
    with pytest.raises(ValueError, match="grammar"):
        OracleTrace(positive.solution, (act,), (Action(3, 119.8, 0),))
    with pytest.raises(ValueError, match="grammar"):
        OracleTrace(safe.solution, safe.steps[:2], ())


def test_trace_constructor_rejects_extra_reordered_support_and_focus() -> None:
    safe = built_negative_trace()
    extra_recall = OracleTraceStep(
        4,
        3,
        InternalEventKind.RECALL,
        safe.steps[-1].timestamp + 0.1,
        0.1,
        10,
        1,
        1,
    )
    extra_compose = OracleTraceStep(
        5,
        4,
        InternalEventKind.COMPOSE,
        safe.steps[-1].timestamp + 0.2,
        0.1,
        10,
        1,
        2,
    )
    reordered = (
        replace(safe.steps[0], kind=InternalEventKind.COMPOSE),
        replace(safe.steps[1], kind=InternalEventKind.RECALL),
        *safe.steps[2:],
    )
    wrong_support = (
        replace(safe.steps[0], selected_record_id=11),
        *safe.steps[1:],
    )
    wrong_focus = (
        replace(safe.steps[0], focus_before=2, focus_after=2),
        *safe.steps[1:],
    )

    for steps in (
        (*safe.steps, extra_recall, extra_compose),
        reordered,
        wrong_support,
        wrong_focus,
    ):
        with pytest.raises(ValueError, match="grammar"):
            OracleTrace(safe.solution, steps, ())


def test_trace_boundaries_reject_bypassed_grammar_and_forged_actions() -> None:
    safe = built_negative_trace()
    positive = built_positive_trace()
    noop = OracleTraceStep(0, None, InternalEventKind.NOOP, 100.1, 0.1, None, None, None)
    corrupt_action = object.__new__(Action)
    object.__setattr__(corrupt_action, "hazard_type", 3)
    object.__setattr__(corrupt_action, "timestamp", math.nan)
    object.__setattr__(corrupt_action, "caused_by_event_id", positive.steps[-1].trace_step_id)
    corrupt_traces = (
        unchecked_trace(safe, steps=()),
        unchecked_trace(safe, steps=(noop,)),
        unchecked_trace(
            safe,
            steps=(
                replace(safe.steps[0], kind=InternalEventKind.COMPOSE),
                replace(safe.steps[1], kind=InternalEventKind.RECALL),
                *safe.steps[2:],
            ),
        ),
        unchecked_trace(positive, actions=(corrupt_action,)),
    )

    for trace in corrupt_traces:
        with pytest.raises(OracleError):
            scale_oracle_trace(trace, 2.0)
        with pytest.raises(OracleError):
            oracle_actions(trace)


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
    public = positive_public(delay=3.0)
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
    compressed = build_oracle_trace(public, solution, terminal, timing, np.random.default_rng(2))

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
    trace = built_positive_trace()

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


@pytest.mark.parametrize("factor", [True, 0.0, -1.0, math.inf, math.nan])
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


@pytest.mark.parametrize("positive", [True, False])
def test_scale_trace_rejects_finite_overflow_for_every_terminal_kind(positive: bool) -> None:
    trace = built_positive_trace() if positive else built_negative_trace()

    with pytest.raises(OracleError, match="nonfinite"):
        scale_oracle_trace(trace, 1.0e308)
