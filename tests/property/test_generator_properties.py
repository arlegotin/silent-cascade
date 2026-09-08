"""Property coverage for independent primary validation and deterministic cohorts."""

from dataclasses import replace
from itertools import groupby, islice
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import canonical_episode_bytes, scale_episode_time
from silent_cascade.env.generator import (
    PHASE1_GATE_ALLOCATION,
    CohortRequest,
    IndependentEpisodeRequest,
    generate_independent_episode,
    generate_matched_cohort,
    generate_stress_episode,
    independent_seed_tokens,
    iter_independent_requests,
    iter_phase1_gate_requests,
    regenerate_independent_episode,
)
from silent_cascade.rng import AllocationLabelKey, allocate_independent_variants

PRIMARY_PATHS = {
    SuiteName.VALIDATION: (2, 3, 4),
    SuiteName.IID_PRIMARY: (2, 3, 4),
    SuiteName.OOD_DEPTH: (5, 6, 7, 8),
    SuiteName.OOD_SHORT_DELAY: (2, 3, 4),
    SuiteName.OOD_LONG_DELAY: (2, 3, 4),
    SuiteName.DISTRACTOR_FLOOD: (2, 3, 4),
}
STRUCTURAL_STRESS_SUITES = (
    SuiteName.BRANCHING_STRESS,
    SuiteName.CYCLES_STRESS,
    SuiteName.MEMORY_OVERFLOW_STRESS,
    SuiteName.NULL_NEAR_MISS_STRESS,
)


def _config() -> Phase1Config:
    return resolve_config(
        Phase1Config,
        [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
    ).config


def _stress_config() -> Phase1Config:
    return resolve_config(
        Phase1Config,
        [
            Path("configs/base.yaml"),
            Path("configs/data/primary.yaml"),
            Path("configs/data/stress.yaml"),
        ],
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


def _allocation_quartet(
    split_namespace: SplitNamespace, root_seed: int, block_index: int, quartet_offset: int
) -> tuple[object, ...]:
    first_request_index = (
        sum(block.episode_count for block in PHASE1_GATE_ALLOCATION.blocks[:block_index])
        + quartet_offset * 4
    )
    if split_namespace is SplitNamespace.PHASE1_GATE:
        requests = iter_phase1_gate_requests(root_seed)
    else:
        allocation = PHASE1_GATE_ALLOCATION.model_copy(update={"split_namespace": split_namespace})
        requests = iter_independent_requests(allocation, root_seed)
    return tuple(islice(requests, first_request_index, first_request_index + 4))


def test_every_frozen_gate_request_forms_a_valid_audit_quartet() -> None:
    """All 100k real requests obey the suite-scoped Task 15 quartet contract."""
    from silent_cascade.env.leakage import (
        AuditSourceDescriptor,
        _StoredExample,
        _validate_independent_quartets,
    )

    descriptor = AuditSourceDescriptor(
        schema_version="leakage-source-v1",
        generation_mode="independent",
        allocation_id=PHASE1_GATE_ALLOCATION.allocation_id,
        allocation_or_manifest_sha256="1" * 64,
        split_namespace=SplitNamespace.PHASE1_GATE,
        root_seed=41,
        public_id_seed_sha256="2" * 64,
        config_sha256="3" * 64,
        generator_source_sha256="4" * 64,
        episode_count=100_000,
    )

    quartet_count = 0
    episode_count = 0
    for quartet_index, grouped in groupby(
        iter_phase1_gate_requests(41),
        key=lambda request: request.allocation_quartet_index,
    ):
        requests = tuple(grouped)
        rows = tuple(
            _StoredExample(
                public_id=f"00000000-0000-4000-8000-{request.episode_index:012d}",
                digest=f"{position + 1:064x}",
                group_id=f"independent:{quartet_index}",
                suite=request.suite,
                path_length=request.requested_path_length,
                variant=request.variant,
                hazard_class=0 if request.variant.value == "positive" else None,
                block=quartet_index,
                position=request.episode_index,
                quartet_member_index=request.quartet_member_index,
            )
            for position, request in enumerate(requests)
        )
        _validate_independent_quartets(rows, descriptor)
        quartet_count += 1
        episode_count += len(rows)

    assert quartet_count == 25_000
    assert episode_count == 100_000


@settings(max_examples=8, deadline=None)
@given(
    root_seed=st.integers(min_value=0, max_value=2**32 - 1),
    block_index=st.integers(min_value=0, max_value=len(PHASE1_GATE_ALLOCATION.blocks) - 1),
    quartet_selector=st.integers(min_value=0, max_value=10_000),
)
def test_independent_gate_and_frozen_requests_share_one_deterministic_primitive(
    root_seed: int, block_index: int, quartet_selector: int
) -> None:
    """Namespace changes must preserve APIs while domain-separating all independent artifacts."""
    from silent_cascade.env.invariants import validate_episode_invariants

    config = _config()
    block = PHASE1_GATE_ALLOCATION.blocks[block_index]
    quartet_offset = quartet_selector % (block.episode_count // 4)
    gate_requests = _allocation_quartet(
        SplitNamespace.PHASE1_GATE, root_seed, block_index, quartet_offset
    )
    frozen_requests = _allocation_quartet(
        SplitNamespace.FROZEN, root_seed, block_index, quartet_offset
    )
    assert all(
        replace(gate, split_namespace=SplitNamespace.FROZEN, variant=frozen.variant) == frozen
        for gate, frozen in zip(gate_requests, frozen_requests, strict=True)
    )
    np.random.seed(20260901)
    before = np.random.get_state()
    gate_forward = {
        request.episode_index: generate_independent_episode(config, request, 91)
        for request in gate_requests
    }
    frozen_forward = {
        request.episode_index: generate_independent_episode(config, request, 91)
        for request in frozen_requests
    }
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]

    for chunk_size in (1, 3, 7):
        gate_chunked = {
            request.episode_index: generate_independent_episode(config, request, 91)
            for start in range(0, len(gate_requests), chunk_size)
            for request in gate_requests[start : start + chunk_size]
        }
        gate_reverse = {
            request.episode_index: generate_independent_episode(config, request, 91)
            for request in reversed(gate_requests)
        }
        for gate_request, frozen_request in zip(gate_requests, frozen_requests, strict=True):
            gate = gate_forward[gate_request.episode_index]
            frozen = frozen_forward[frozen_request.episode_index]
            assert canonical_episode_bytes(
                gate_chunked[gate_request.episode_index]
            ) == canonical_episode_bytes(gate)
            assert canonical_episode_bytes(
                gate_reverse[gate_request.episode_index]
            ) == canonical_episode_bytes(gate)
            assert validate_episode_invariants(gate, config).valid
            assert validate_episode_invariants(frozen, config).valid
            assert independent_seed_tokens(gate_request, 0) != independent_seed_tokens(
                frozen_request, 0
            )
            assert gate.public.init.episode_public_id != frozen.public.init.episode_public_id
            assert (
                regenerate_independent_episode(
                    config,
                    gate_request,
                    91,
                    gate.public.init.episode_public_id,
                    gate.truth.recipe.accepted_attempt,
                )
                == gate
            )
            assert (
                regenerate_independent_episode(
                    config,
                    frozen_request,
                    91,
                    frozen.public.init.episode_public_id,
                    frozen.truth.recipe.accepted_attempt,
                )
                == frozen
            )


@settings(max_examples=8, deadline=None)
@given(
    root_seed=st.integers(min_value=0, max_value=2**32 - 1),
    episode_index=st.integers(min_value=0, max_value=7_999),
    target_suite=st.sampled_from((SuiteName.CLOCK_SCALE_0_1X, SuiteName.CLOCK_SCALE_10X)),
)
def test_paired_clock_scaling_preserves_oracle_semantics_and_normalized_windows(
    root_seed: int, episode_index: int, target_suite: SuiteName
) -> None:
    """Scaling a paired IID source must not change its decision or normalized window."""
    request = next(islice(iter_phase1_gate_requests(root_seed), episode_index, episode_index + 1))
    bundle = generate_independent_episode(_config(), request, 91)
    scaled = scale_episode_time(
        bundle,
        target_suite,
        "10000000-0000-4000-8000-000000000001",
    )

    assert scaled.truth.key == bundle.truth.key
    assert scaled.truth.key.coordinate.quartet_member_index == request.quartet_member_index
    assert scaled.truth.recipe.variant is bundle.truth.recipe.variant
    assert scaled.truth.relevant_node_path == bundle.truth.relevant_node_path
    assert scaled.truth.relevant_record_ids == bundle.truth.relevant_record_ids
    if bundle.truth.action_window_start is not None:
        assert scaled.truth.action_window_start is not None
        assert scaled.truth.action_window_end is not None
        assert scaled.truth.action_target is not None
        assert scaled.truth.action_window_start - scaled.truth.activation_time == pytest.approx(
            (bundle.truth.action_window_start - bundle.truth.activation_time)
            * scaled.truth.recipe.clock_scale
        )


@settings(max_examples=8, deadline=None)
@given(
    suite=st.sampled_from(STRUCTURAL_STRESS_SUITES),
    root_seed=st.integers(min_value=0, max_value=2**32 - 1),
    episode_index=st.integers(min_value=0, max_value=500),
    quartet_member_index=st.integers(min_value=0, max_value=3),
)
def test_structural_stress_episodes_remain_independently_valid_and_private(
    suite: SuiteName,
    root_seed: int,
    episode_index: int,
    quartet_member_index: int,
) -> None:
    """Removing stress-specific graph checks would accept a branch or cycle in the wrong place."""
    from silent_cascade.env.episode import EpisodeVariant
    from silent_cascade.env.invariants import validate_episode_invariants
    from silent_cascade.env.oracle import OraclePolicy, OracleTerminalKind, solve_public_episode

    expected_variant = (
        EpisodeVariant.DISCONNECTED_NEGATIVE
        if suite is SuiteName.NULL_NEAR_MISS_STRESS
        else EpisodeVariant.POSITIVE
    )
    allocation_quartet_index = episode_index // 4
    variants = allocate_independent_variants(
        AllocationLabelKey(
            "ofd-v1",
            SplitNamespace.DEBUG,
            suite,
            root_seed,
            3,
            allocation_quartet_index,
        )
    )
    variant = variants[quartet_member_index]
    assume(variant is expected_variant)
    request = IndependentEpisodeRequest(
        SplitNamespace.DEBUG,
        suite,
        root_seed,
        episode_index,
        3,
        variant,
        allocation_quartet_index,
        quartet_member_index,
    )
    bundle = generate_stress_episode(_stress_config(), request, 91)
    policy = OraclePolicy.BRANCHING if suite is SuiteName.BRANCHING_STRESS else OraclePolicy.PRIMARY
    expected_terminal = (
        OracleTerminalKind.DISCONNECTED
        if suite is SuiteName.NULL_NEAR_MISS_STRESS
        else OracleTerminalKind.HAZARD
    )

    assert validate_episode_invariants(bundle, _stress_config()).valid
    assert bundle.truth.key.coordinate.quartet_member_index == quartet_member_index
    assert solve_public_episode(bundle.public, policy).terminal_kind is expected_terminal
    assert "stress_metadata" not in repr(bundle.public)


@settings(max_examples=8, deadline=None)
@given(
    root_seed=st.integers(min_value=0, max_value=2**32 - 1),
    cohort_index=st.integers(min_value=0, max_value=500),
)
def test_public_shortcut_features_are_repeatable_and_finite(
    root_seed: int, cohort_index: int
) -> None:
    """Changing extraction order or admitting nonfinite values must fail this property."""
    from silent_cascade.env.leakage import AuditExample, extract_shortcut_features

    cohort = generate_matched_cohort(
        _config(),
        CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, root_seed, cohort_index, 3),
        public_id_seed=91,
    )
    example = AuditExample(
        cohort.episodes[0], cohort_index, "matched", cohort_index, 0, quartet_member_index=0
    )
    first = extract_shortcut_features(example, 32)
    second = extract_shortcut_features(example, 32)

    assert first.episode_public_id == second.episode_public_id
    assert first.audit_group_id == second.audit_group_id
    assert all(
        np.array_equal(first.vectors[group], second.vectors[group]) for group in first.vectors
    )
    assert all(np.isfinite(vector).all() for vector in first.vectors.values())
