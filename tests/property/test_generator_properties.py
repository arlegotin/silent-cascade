"""Property coverage for independent primary validation and deterministic cohorts."""

from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.generator import CohortRequest, generate_matched_cohort

PRIMARY_PATHS = {
    SuiteName.VALIDATION: (2, 3, 4),
    SuiteName.IID_PRIMARY: (2, 3, 4),
    SuiteName.OOD_DEPTH: (5, 6, 7, 8),
    SuiteName.OOD_SHORT_DELAY: (2, 3, 4),
    SuiteName.OOD_LONG_DELAY: (2, 3, 4),
    SuiteName.DISTRACTOR_FLOOD: (2, 3, 4),
}


def _config() -> Phase1Config:
    return resolve_config(
        Phase1Config,
        [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
    ).config


@settings(max_examples=12, deadline=None)
@given(
    suite=st.sampled_from(tuple(PRIMARY_PATHS)),
    root_seed=st.integers(min_value=0, max_value=2**32 - 1),
    cohort_index=st.integers(min_value=0, max_value=500),
)
def test_primary_generation_is_independently_valid_and_bounded(
    suite: SuiteName, root_seed: int, cohort_index: int
) -> None:
    """Dropping an invariant or capacity bound must fail a sampled primary cohort."""
    from silent_cascade.env.invariants import validate_cohort_invariants

    length = PRIMARY_PATHS[suite][cohort_index % len(PRIMARY_PATHS[suite])]
    cohort = generate_matched_cohort(
        _config(),
        CohortRequest(SplitNamespace.DEBUG, suite, root_seed, cohort_index, length),
        public_id_seed=91,
    )

    reports = validate_cohort_invariants(cohort.episodes, _config())
    assert all(report.valid and report.fact_count <= 64 for report in reports)
    assert {report.requested_path_length for report in reports} == {length}
    assert all(
        np.isfinite(event.timestamp) for bundle in cohort.episodes for event in bundle.public.events
    )


def test_matched_generation_is_order_chunk_and_global_rng_independent() -> None:
    """Using ambient RNG or construction order would change canonical bundle bytes."""
    from silent_cascade.env.episode import canonical_episode_bytes
    from silent_cascade.env.invariants import validate_cohort_invariants

    config = _config()
    requests = tuple(
        CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 41, index, 3)
        for index in (7, 8, 9)
    )
    np.random.seed(2026)
    before = np.random.get_state()
    forward = {
        request.cohort_index: generate_matched_cohort(config, request, public_id_seed=91)
        for request in requests
    }
    after = np.random.get_state()
    reverse = {
        request.cohort_index: generate_matched_cohort(config, request, public_id_seed=91)
        for request in reversed(requests)
    }

    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    for index in forward:
        assert tuple(map(canonical_episode_bytes, forward[index].episodes)) == tuple(
            map(canonical_episode_bytes, reverse[index].episodes)
        )
        validate_cohort_invariants(forward[index].episodes, config)


def _direct_summary(cohort: object) -> tuple[object, ...]:
    episodes = cohort.episodes  # type: ignore[union-attr]
    variants = tuple(sorted(bundle.truth.recipe.variant.value for bundle in episodes))
    delays = tuple(bundle.truth.episode_delay for bundle in episodes)
    fact_counts = tuple(len(bundle.public.events) - 1 for bundle in episodes)
    link_counts = tuple(
        sum(type(event.payload).__name__ == "LinkFact" for event in bundle.public.events[:-1])
        for bundle in episodes
    )
    gaps = tuple(
        tuple(
            event.timestamp
            - (
                bundle.public.init.initial_time
                if index == 0
                else bundle.public.events[index - 1].timestamp
            )
            for index, event in enumerate(bundle.public.events)
        )
        for bundle in episodes
    )
    return variants, delays, fact_counts, link_counts, gaps


@settings(max_examples=10, deadline=None)
@given(
    suite=st.sampled_from(tuple(PRIMARY_PATHS)),
    root_seed=st.integers(min_value=0, max_value=2**32 - 1),
    chunk_size=st.sampled_from((1, 3, 7)),
)
def test_primary_generation_is_chunk_equivalent_with_direct_matched_summaries(
    suite: SuiteName, root_seed: int, chunk_size: int
) -> None:
    """Changing call/chunk topology must not alter independent primary artifacts."""
    from silent_cascade.env.episode import canonical_episode_bytes

    config = _config()
    length = PRIMARY_PATHS[suite][0]
    requests = tuple(
        CohortRequest(SplitNamespace.DEBUG, suite, root_seed, index, length) for index in range(11)
    )
    np.random.seed(20260831)
    before = np.random.get_state()
    direct = {
        request.cohort_index: generate_matched_cohort(config, request, public_id_seed=91)
        for request in requests
    }
    chunked: dict[int, object] = {}
    for start in range(0, len(requests), chunk_size):
        for request in requests[start : start + chunk_size]:
            chunked[request.cohort_index] = generate_matched_cohort(
                config, request, public_id_seed=91
            )

    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]

    for index, expected in direct.items():
        actual = chunked[index]
        assert _direct_summary(actual) == _direct_summary(expected)
        assert tuple(map(canonical_episode_bytes, actual.episodes)) == tuple(
            map(canonical_episode_bytes, expected.episodes)
        )
