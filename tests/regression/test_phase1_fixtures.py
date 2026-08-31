"""Independent hand-authored Phase 1 oracle fixture regression tests."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    OracleTerminalKind,
    build_oracle_trace,
    solve_public_episode,
    verify_oracle_truth,
)
from silent_cascade.env.reward import score_actions
from silent_cascade.rng import CounterSeedKey, SeedStream, local_generator
from silent_cascade.schemas import (
    ActivationPayload,
    AgentInit,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    LinkFact,
    SafeFact,
)

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "phase1" / "oracle_cases.json"


@dataclass(frozen=True, slots=True)
class FixtureCase:
    raw: dict[str, Any]
    public: PublicEpisode
    truth: EpisodeTruth
    private_terminal: ExternalEvent
    trace_seed_key: CounterSeedKey
    fact_payloads: tuple[LinkFact | HazardFact | SafeFact, ...]


def _fact_payload(raw: list[Any]) -> LinkFact | HazardFact | SafeFact:
    if raw[0] == "link":
        return LinkFact(raw[1], raw[2])
    if raw[0] == "hazard":
        return HazardFact(raw[1], raw[2], raw[3])
    if raw[0] == "safe":
        return SafeFact(raw[1])
    raise AssertionError(f"unknown hand-authored fact kind: {raw[0]}")


def _public_from_payloads(
    payloads: tuple[LinkFact | HazardFact | SafeFact, ...],
    raw: dict[str, Any],
) -> PublicEpisode:
    events = tuple(
        ExternalEvent(index + 10, float(index + 1), ExternalEventKind.FACT, payload)
        for index, payload in enumerate(payloads)
    )
    return PublicEpisode(
        AgentInit("00000000-0000-4000-8000-000000000007", 64, 4, 0.0),
        (
            *events,
            ExternalEvent(
                len(events) + 10,
                raw["activation_time"],
                ExternalEventKind.ACTIVATE,
                ActivationPayload(raw["activation_node"]),
            ),
        ),
    )


def _truth_for_case(raw: dict[str, Any], public: PublicEpisode) -> EpisodeTruth:
    variant = EpisodeVariant(raw["variant"])
    terminal_kind = (
        ExternalEventKind.OUTCOME if variant is EpisodeVariant.POSITIVE else ExternalEventKind.END
    )
    terminal = ExternalEvent(99, raw["terminal_time"], terminal_kind, None)
    selected_ids = tuple(index + 10 for index in raw["expected_record_fact_indexes"])
    terminal_index = raw["expected_terminal_fact_index"]
    window = raw["expected_window"]
    return EpisodeTruth(
        key=EpisodeKey(
            "ofd-v1",
            SplitNamespace.PHASE1_GATE,
            SuiteName.IID_PRIMARY,
            raw["trace_root_seed"],
            MatchedEpisodeCoordinate("matched", 0, 0),
        ),
        recipe=EpisodeRecipe(
            len(raw["expected_path"]) - 1,
            variant,
            len(raw["facts"]) - len(selected_ids),
            SuiteName.IID_PRIMARY,
            0,
        ),
        relevant_node_path=tuple(raw["expected_path"]),
        relevant_record_ids=selected_ids,
        terminal_record_id=None if terminal_index is None else terminal_index + 10,
        relevant_hazard_type=raw["expected_hazard_type"],
        private_terminal=terminal,
        activation_time=raw["activation_time"],
        episode_delay=raw["terminal_time"] - raw["activation_time"],
        action_window_start=None if window is None else window[0],
        action_window_end=None if window is None else window[1],
        action_target=raw["expected_target"],
        rejection_count=0,
        rejection_reasons=(),
    )


def load_fixture_cases() -> dict[str, FixtureCase]:
    decoded = json.loads(FIXTURE_PATH.read_text())
    result: dict[str, FixtureCase] = {}
    for raw in decoded:
        payloads = tuple(_fact_payload(fact) for fact in raw["facts"])
        public = _public_from_payloads(payloads, raw)
        truth = _truth_for_case(raw, public)
        result[raw["variant"]] = FixtureCase(
            raw=raw,
            public=public,
            truth=truth,
            private_terminal=truth.private_terminal,
            trace_seed_key=CounterSeedKey(
                "ofd-v1",
                SplitNamespace.PHASE1_GATE,
                SuiteName.IID_PRIMARY,
                raw["trace_root_seed"],
                0,
                0,
                SeedStream.TRACE_JITTER,
                0,
            ),
            fact_payloads=payloads,
        )
    return result


@pytest.mark.parametrize(
    ("path_length", "variant", "expected_events"),
    [(3, "positive", 9), (2, "safe_negative", 6), (2, "disconnected_negative", 4)],
)
def test_oracle_trace_lengths(path_length: int, variant: str, expected_events: int) -> None:
    case = load_fixture_cases()[variant]
    solution = solve_public_episode(case.public)
    trace = build_oracle_trace(
        case.public,
        solution,
        case.private_terminal,
        OracleTimingConfig(),
        local_generator(case.trace_seed_key),
    )

    assert len(solution.link_record_ids) == path_length
    assert len(trace.steps) == expected_events == case.raw["expected_event_count"]


@pytest.mark.parametrize("variant", tuple(EpisodeVariant))
def test_hand_authored_fixture_solution_timing_and_scoring_kat(variant: EpisodeVariant) -> None:
    case = load_fixture_cases()[variant.value]
    solution = solve_public_episode(case.public)
    trace = build_oracle_trace(
        case.public,
        solution,
        case.private_terminal,
        OracleTimingConfig(),
        local_generator(case.trace_seed_key),
    )

    assert solution.terminal_kind is OracleTerminalKind(case.raw["expected_terminal"])
    assert solution.node_path == tuple(case.raw["expected_path"])
    assert solution.hazard_type == case.raw["expected_hazard_type"]
    assert [step.delta for step in trace.steps if step.kind.value != "act"] == pytest.approx(
        case.raw["expected_non_action_deltas"], abs=1.0e-15
    )
    verify_oracle_truth(solution, case.truth)
    assert score_actions(case.truth, trace.actions).timed_success


def test_solver_and_normalized_action_are_invariant_under_128_record_permutations() -> None:
    case = load_fixture_cases()[EpisodeVariant.POSITIVE.value]
    permutation_rng = local_generator(
        CounterSeedKey(
            "ofd-v1",
            SplitNamespace.PHASE1_GATE,
            SuiteName.IID_PRIMARY,
            6001,
            0,
            0,
            SeedStream.PRESENTATION,
            0,
        )
    )
    expected_selected_payloads = tuple(
        case.fact_payloads[index] for index in case.raw["expected_record_fact_indexes"]
    )

    for _ in range(128):
        order = tuple(int(index) for index in permutation_rng.permutation(len(case.fact_payloads)))
        permuted_payloads = tuple(case.fact_payloads[index] for index in order)
        public = _public_from_payloads(permuted_payloads, case.raw)
        solution = solve_public_episode(public)
        selected_by_id = {
            event.event_id: event.payload
            for event in public.events
            if event.kind is ExternalEventKind.FACT
        }
        selected_payloads = tuple(
            selected_by_id[record_id]
            for record_id in (*solution.link_record_ids, solution.terminal_record_id)
            if record_id is not None
        )
        trace = build_oracle_trace(
            public,
            solution,
            case.private_terminal,
            OracleTimingConfig(),
            local_generator(case.trace_seed_key),
        )

        assert selected_payloads == expected_selected_payloads
        assert solution.node_path == (9, 7, 2, 5)
        assert solution.terminal_kind is OracleTerminalKind.HAZARD
        assert (solution.hazard_type, solution.public_delay) == (3, 24.0)
        assert (trace.actions[0].timestamp - 100.0) / 24.0 == pytest.approx(0.825)
