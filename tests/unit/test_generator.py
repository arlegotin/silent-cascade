"""Behavioral contracts for matched Phase 1 cohort construction."""

import re
from collections import Counter
from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import EpisodeVariant
from silent_cascade.env.generator import CohortRequest
from silent_cascade.env.oracle import solve_public_episode, verify_oracle_truth
from silent_cascade.errors import GenerationError
from silent_cascade.rng import PublicIdBatchKey, allocate_public_ids
from silent_cascade.schemas import ExternalEventKind, HazardFact, LinkFact, SafeFact


@pytest.fixture
def config() -> Phase1Config:
    return resolve_config(
        Phase1Config,
        [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
    ).config


def cohort_request(index: int = 17) -> CohortRequest:
    return CohortRequest(
        split_namespace=SplitNamespace.DEBUG,
        suite=SuiteName.IID_PRIMARY,
        root_seed=41,
        cohort_index=index,
        requested_path_length=3,
    )


def _link_signature(bundle: object) -> tuple[tuple[int, int], ...]:
    public = bundle.public  # type: ignore[union-attr]
    links = [
        event.payload
        for event in public.events
        if event.kind is ExternalEventKind.FACT and isinstance(event.payload, LinkFact)
    ]
    degrees: dict[int, list[int]] = {}
    for link in links:
        degrees.setdefault(link.source_node, [0, 0])[1] += 1
        degrees.setdefault(link.target_node, [0, 0])[0] += 1
    return tuple(sorted((degree[0], degree[1]) for degree in degrees.values()))


def _fact_gaps(bundle: object) -> tuple[float, ...]:
    public = bundle.public  # type: ignore[union-attr]
    previous = public.init.initial_time
    gaps: list[float] = []
    for event in public.events:
        gaps.append(event.timestamp - previous)
        previous = event.timestamp
    return tuple(gaps)


def test_matched_cohort_preserves_exact_nuisance_and_terminal_matching(
    config: Phase1Config,
) -> None:
    """Changing member construction must not break matched nuisance controls."""
    from silent_cascade.env.generator import generate_matched_cohort

    cohort = generate_matched_cohort(config, cohort_request(), public_id_seed=91)

    assert Counter(bundle.truth.recipe.variant for bundle in cohort.episodes) == {
        EpisodeVariant.POSITIVE: 2,
        EpisodeVariant.SAFE_NEGATIVE: 1,
        EpisodeVariant.DISCONNECTED_NEGATIVE: 1,
    }
    assert {bundle.truth.recipe.accepted_attempt for bundle in cohort.episodes} == {
        cohort.accepted_attempt
    }
    assert len({_link_signature(bundle) for bundle in cohort.episodes}) == 1
    assert len({_fact_gaps(bundle) for bundle in cohort.episodes}) == 1
    assert len({bundle.truth.episode_delay for bundle in cohort.episodes}) == 1
    assert {
        sum(
            isinstance(event.payload, HazardFact)
            for event in bundle.public.events
            if event.kind is ExternalEventKind.FACT
        )
        for bundle in cohort.episodes
    } == {2}
    assert {
        sum(
            isinstance(event.payload, SafeFact)
            for event in bundle.public.events
            if event.kind is ExternalEventKind.FACT
        )
        for bundle in cohort.episodes
    } == {1}
    assert all(
        [event.event_id for event in bundle.public.events if event.kind is ExternalEventKind.FACT]
        == list(range(len(bundle.public.events) - 1))
        and bundle.public.events[-1].event_id == len(bundle.public.events) - 1
        and bundle.truth.private_terminal.event_id == len(bundle.public.events)
        for bundle in cohort.episodes
    )


def test_matched_cohort_is_private_safe_and_oracle_consistent(config: Phase1Config) -> None:
    """Changing the truth binding must not let public facts disagree with the scorer."""
    from silent_cascade.env.generator import generate_matched_cohort

    cohort = generate_matched_cohort(config, cohort_request(), public_id_seed=91)

    assert len(cohort.seed_tokens) == 16
    assert len(set(cohort.seed_tokens)) == len(cohort.seed_tokens)
    for bundle in cohort.episodes:
        solution = solve_public_episode(bundle.public)
        verify_oracle_truth(solution, bundle.truth)
        public_view = repr(bundle.public)
        for private_name in (
            "root_seed",
            "cohort_index",
            "accepted_attempt",
            "relevant_node_path",
            "private_terminal",
        ):
            assert private_name not in public_view


def test_matched_regeneration_requires_accepted_attempt_and_public_id(config: Phase1Config) -> None:
    """Changing regeneration coordinates must not select a different cohort member."""
    from silent_cascade.env.generator import (
        generate_matched_cohort,
        regenerate_matched_episode,
    )

    request = cohort_request()
    cohort = generate_matched_cohort(config, request, public_id_seed=91)
    expected = cohort.episodes[2]

    assert (
        regenerate_matched_episode(
            config,
            request,
            public_id_seed=91,
            expected_public_id=expected.public.init.episode_public_id,
            expected_accepted_attempt=cohort.accepted_attempt,
        )
        == expected
    )
    with pytest.raises(GenerationError, match="regeneration"):
        regenerate_matched_episode(
            config,
            request,
            public_id_seed=91,
            expected_public_id="00000000-0000-4000-8000-000000000000",
            expected_accepted_attempt=cohort.accepted_attempt,
        )


def test_invalid_matched_attempt_retries_the_complete_cohort(
    config: Phase1Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing retries to member-local must fail this matched-cohort contract."""
    import silent_cascade.env.generator as generator

    original = generator.sample_cohort_template

    def reject_first_two(config: Phase1Config, request: CohortRequest, attempt: int) -> object:
        if attempt < 2:
            raise ValueError("forced template rejection")
        return original(config, request, attempt)

    monkeypatch.setattr(generator, "sample_cohort_template", reject_first_two)
    cohort = generator.generate_matched_cohort(config, cohort_request(), public_id_seed=91)

    assert cohort.accepted_attempt == 2
    assert {bundle.truth.recipe.accepted_attempt for bundle in cohort.episodes} == {2}
    generated_public_ids = tuple(bundle.public.init.episode_public_id for bundle in cohort.episodes)
    assert generated_public_ids == allocate_public_ids(
        PublicIdBatchKey(
            generator_version="ofd-v1",
            split_namespace=SplitNamespace.DEBUG,
            suite=SuiteName.IID_PRIMARY,
            public_id_seed=91,
            cohort_index=17,
            accepted_attempt=2,
        )
    )
    assert cohort.rejections == (generator.RejectionDiagnostic("forced template rejection", 2),)


def test_matched_generation_exhaustion_is_opaque_and_aggregated(
    config: Phase1Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing exhaustion errors to expose private coordinates must fail this contract."""
    import silent_cascade.env.generator as generator

    def always_reject(*_: object) -> object:
        raise ValueError("forced template rejection")

    monkeypatch.setattr(generator, "sample_cohort_template", always_reject)
    short_config = config.model_copy(
        update={"data": config.data.model_copy(update={"max_generation_attempts": 3})}
    )

    with pytest.raises(GenerationError, match="matched cohort generation exhausted") as raised:
        generator.generate_matched_cohort(short_config, cohort_request(), public_id_seed=91)

    payload = raised.value.to_payload()
    assert payload["context"]["attempt_count"] == 3
    assert payload["context"]["rejection_reasons"] == {"forced template rejection": 3}
    cohort_hash = payload["context"]["cohort_hash"]
    assert isinstance(cohort_hash, str) and re.fullmatch(r"[0-9a-f]{64}", cohort_hash)
    assert set(payload["context"]) == {"cohort_hash", "attempt_count", "rejection_reasons"}
