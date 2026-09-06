"""Adversarial terminal-selection and deadline-boundary contracts."""

import math
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    EpisodeVariant,
    PublicEpisode,
    canonical_episode_bytes,
    episode_from_bytes,
    public_projection,
)
from silent_cascade.env.generator import IndependentEpisodeRequest, generate_stress_episode
from silent_cascade.env.oracle import OraclePolicy, OracleTerminalKind, solve_public_episode
from silent_cascade.errors import EpisodeInvariantError, OracleError
from silent_cascade.rng import (
    AllocationLabelKey,
    IndependentCounterSeedKey,
    SeedStream,
    allocate_independent_variants,
    independent_local_generator,
)
from silent_cascade.schemas import (
    ActivationPayload,
    AgentInit,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    LinkFact,
    SafeFact,
)


@pytest.fixture
def config() -> Phase1Config:
    return resolve_config(
        Phase1Config,
        [
            Path("configs/base.yaml"),
            Path("configs/data/primary.yaml"),
            Path("configs/data/stress.yaml"),
        ],
    ).config


def _request(suite: SuiteName, episode_index: int = 41) -> IndependentEpisodeRequest:
    quartet_member_index = {
        SuiteName.CONTRADICTION_STRESS: 2,
        SuiteName.MINIMUM_DURATION_STRESS: 1,
        SuiteName.CHECKPOINT_STRESS: 1,
    }[suite]
    variants = allocate_independent_variants(
        AllocationLabelKey(
            "ofd-v1",
            SplitNamespace.DEBUG,
            suite,
            20260831,
            3,
            10,
        )
    )
    assert variants[quartet_member_index] is EpisodeVariant.POSITIVE
    return IndependentEpisodeRequest(
        split_namespace=SplitNamespace.DEBUG,
        suite=suite,
        root_seed=20260831,
        episode_index=episode_index,
        requested_path_length=3,
        variant=variants[quartet_member_index],
        allocation_quartet_index=10,
        quartet_member_index=quartet_member_index,
    )


def test_contradiction_stress_keeps_explicit_stale_records_and_selects_current(
    config: Phase1Config,
) -> None:
    bundle = generate_stress_episode(config, _request(SuiteName.CONTRADICTION_STRESS), 91)
    solution = solve_public_episode(bundle.public, OraclePolicy.CONTRADICTION)
    target = bundle.truth.relevant_node_path[-1]
    records = tuple(
        event
        for event in bundle.public.events[:-1]
        if isinstance(event.payload, (HazardFact, SafeFact)) and event.payload.node == target
    )

    assert config.stress is not None
    assert (
        config.stress.contradiction_records[0]
        <= len(records)
        <= config.stress.contradiction_records[1]
    )
    assert len({(event.timestamp, event.payload.confidence) for event in records}) == len(records)
    assert {type(event.payload) for event in records} == {HazardFact, SafeFact}
    selected = max(
        records,
        key=lambda event: (event.timestamp, event.payload.confidence, event.event_id),
    )
    assert solution.terminal_kind is OracleTerminalKind.HAZARD
    assert solution.terminal_record_id == selected.event_id
    assert solution.superseded_terminal_record_ids == tuple(
        event.event_id for event in records if event.event_id != selected.event_id
    )
    assert bundle.truth.terminal_record_id == selected.event_id


def test_contradiction_policy_uses_lexicographic_time_confidence_id_and_refuses_ambiguity() -> None:
    public = PublicEpisode(
        AgentInit("00000000-0000-4000-8000-000000000099", 64, 4, 0.0),
        (
            ExternalEvent(10, 1.0, ExternalEventKind.FACT, LinkFact(1, 2)),
            ExternalEvent(11, 5.0, ExternalEventKind.FACT, HazardFact(2, 1, 8.0, confidence=0.4)),
            ExternalEvent(12, 5.0, ExternalEventKind.FACT, SafeFact(2, confidence=0.9)),
            ExternalEvent(13, 100.0, ExternalEventKind.ACTIVATE, ActivationPayload(1)),
        ),
    )
    solution = solve_public_episode(public, OraclePolicy.CONTRADICTION)
    assert solution.terminal_kind is OracleTerminalKind.SAFE
    assert solution.terminal_record_id == 12
    assert solution.superseded_terminal_record_ids == (11,)

    ambiguous = list(public.events)
    ambiguous[1] = type(ambiguous[1])(
        ambiguous[1].event_id, 5.0, ambiguous[1].kind, SafeFact(2, confidence=0.9)
    )
    with pytest.raises(OracleError, match="duplicate equivalent"):
        solve_public_episode(
            type(public)(public.init, tuple(ambiguous)), OraclePolicy.CONTRADICTION
        )


def test_minimum_duration_is_the_float64_feasibility_boundary(config: Phase1Config) -> None:
    from silent_cascade.env.generator import minimum_duration_is_feasible

    bundle = generate_stress_episode(config, _request(SuiteName.MINIMUM_DURATION_STRESS), 91)
    metadata = bundle.truth.stress_metadata
    assert metadata is not None and metadata.minimum_feasible_delay is not None
    boundary = metadata.minimum_feasible_delay
    delay = bundle.truth.episode_delay
    assert minimum_duration_is_feasible(bundle, delay)
    assert not minimum_duration_is_feasible(bundle, math.nextafter(boundary, -math.inf))
    assert config.stress is not None
    assert delay - boundary >= config.stress.minimum_duration_epsilon
    assert delay - boundary <= config.stress.minimum_duration_epsilon + 4 * math.ulp(delay)


def test_checkpoint_pause_is_private_and_strictly_between_trace_events(
    config: Phase1Config,
) -> None:
    from silent_cascade.env.oracle import build_oracle_trace

    bundle = generate_stress_episode(config, _request(SuiteName.CHECKPOINT_STRESS), 91)
    metadata = bundle.truth.stress_metadata
    assert metadata is not None and metadata.proposed_checkpoint_pause_time is not None
    coordinate = bundle.truth.key.coordinate
    trace = build_oracle_trace(
        bundle.public,
        solve_public_episode(bundle.public),
        bundle.truth.private_terminal,
        bundle.truth.recipe.oracle_timing,
        independent_local_generator(
            IndependentCounterSeedKey(
                "ofd-v1",
                bundle.truth.key.split_namespace,
                bundle.truth.key.suite,
                bundle.truth.key.root_seed,
                coordinate.episode_index,
                SeedStream.TRACE_JITTER,
                bundle.truth.recipe.accepted_attempt,
            )
        ),
    )
    pause = metadata.proposed_checkpoint_pause_time
    assert any(left.timestamp < pause < right.timestamp for left, right in pairwise(trace.steps))
    assert "checkpoint" not in repr(public_projection(bundle)).lower()
    assert "pause" not in repr(public_projection(bundle)).lower()


def test_terminal_and_timing_stress_metadata_round_trips_only_privately(
    config: Phase1Config,
) -> None:
    from silent_cascade.env.invariants import validate_episode_invariants

    wrong_member_by_suite = {
        SuiteName.MINIMUM_DURATION_STRESS: 0,
        SuiteName.CHECKPOINT_STRESS: 0,
    }
    for suite in (SuiteName.MINIMUM_DURATION_STRESS, SuiteName.CHECKPOINT_STRESS):
        bundle = generate_stress_episode(config, _request(suite), 91)
        rebuilt = episode_from_bytes(canonical_episode_bytes(bundle))
        assert rebuilt == bundle
        assert (
            rebuilt.truth.key.coordinate.quartet_member_index
            == bundle.truth.key.coordinate.quartet_member_index
        )
        public = public_projection(bundle)
        public_text = repr(public).lower()
        assert "minimum_feasible" not in public_text
        assert "checkpoint" not in public_text
        assert "stress_metadata" not in public_text
        corrupted = replace(
            bundle,
            truth=replace(
                bundle.truth,
                key=replace(
                    bundle.truth.key,
                    coordinate=replace(
                        bundle.truth.key.coordinate,
                        quartet_member_index=wrong_member_by_suite[suite],
                    ),
                ),
            ),
        )
        with pytest.raises(EpisodeInvariantError):
            validate_episode_invariants(corrupted, config)
