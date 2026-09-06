"""Contracts for suite allocation and cohort-level sampling."""

import ast
from collections import Counter
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import EpisodeVariant
from silent_cascade.env.generator import (
    PHASE1_GATE_ALLOCATION,
    VALIDATION_ALLOCATION,
    ClockEpisodeBlock,
    CohortAllocation,
    CohortBlock,
    CohortTemplate,
    EpisodeBlock,
    IndependentAllocation,
    IndependentEpisodeRequest,
    iter_cohort_requests,
    iter_independent_requests,
    iter_phase1_gate_requests,
    sample_cohort_template,
    suite_spec,
    validate_phase1_gate_allocation,
    validate_validation_allocation,
)
from silent_cascade.rng import AllocationLabelKey, allocate_independent_variants


@pytest.fixture
def config() -> Phase1Config:
    return resolve_config(
        Phase1Config,
        [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
    ).config


def test_validation_allocation_starts_at_exact_stratified_cohorts(config: Phase1Config) -> None:
    requests = tuple(iter_cohort_requests(VALIDATION_ALLOCATION, root_seed=41))

    assert [(item.cohort_index, item.requested_path_length) for item in requests[:3]] == [
        (0, 2),
        (1, 2),
        (2, 2),
    ]
    assert [(item.cohort_index, item.requested_path_length) for item in requests[832:835]] == [
        (832, 2),
        (833, 2),
        (834, 3),
    ]
    assert [(item.cohort_index, item.requested_path_length) for item in requests[1665:1668]] == [
        (1665, 3),
        (1666, 3),
        (1667, 4),
    ]
    assert len(requests) == 2_500
    assert Counter(item.requested_path_length for item in requests) == {2: 834, 3: 833, 4: 833}
    validate_validation_allocation(VALIDATION_ALLOCATION, config)


def test_phase1_gate_allocation_has_exact_frozen_blocks(config: Phase1Config) -> None:
    requests = tuple(iter_independent_requests(PHASE1_GATE_ALLOCATION, root_seed=41))

    assert len(requests) == 100_000
    assert sum(request.suite is SuiteName.IID_PRIMARY for request in requests) == 24_000
    assert Counter((request.suite, request.requested_path_length) for request in requests) == {
        (SuiteName.IID_PRIMARY, 2): 8_000,
        (SuiteName.IID_PRIMARY, 3): 8_000,
        (SuiteName.IID_PRIMARY, 4): 8_000,
        (SuiteName.OOD_DEPTH, 5): 4_000,
        (SuiteName.OOD_DEPTH, 6): 4_000,
        (SuiteName.OOD_DEPTH, 7): 4_000,
        (SuiteName.OOD_DEPTH, 8): 4_000,
        (SuiteName.OOD_SHORT_DELAY, 2): 8_000,
        (SuiteName.OOD_SHORT_DELAY, 3): 8_000,
        (SuiteName.OOD_SHORT_DELAY, 4): 8_000,
        (SuiteName.OOD_LONG_DELAY, 2): 4_000,
        (SuiteName.OOD_LONG_DELAY, 3): 4_000,
        (SuiteName.OOD_LONG_DELAY, 4): 4_000,
        (SuiteName.DISTRACTOR_FLOOD, 2): 8_000,
        (SuiteName.DISTRACTOR_FLOOD, 3): 8_000,
        (SuiteName.DISTRACTOR_FLOOD, 4): 8_000,
    }
    assert [(item.suite, item.episode_index) for item in requests[7_999:8_001]] == [
        (SuiteName.IID_PRIMARY, 7_999),
        (SuiteName.IID_PRIMARY, 8_000),
    ]
    assert all(
        Counter(item.variant for item in requests[offset : offset + 4])
        == Counter(
            {
                EpisodeVariant.POSITIVE: 2,
                EpisodeVariant.SAFE_NEGATIVE: 1,
                EpisodeVariant.DISCONNECTED_NEGATIVE: 1,
            }
        )
        for offset in range(0, len(requests), 4)
    )
    assert [(item.requested_path_length, item.episode_index) for item in requests[:4]] == [
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
    ]
    assert tuple(iter_phase1_gate_requests(41)) == requests
    validate_phase1_gate_allocation(PHASE1_GATE_ALLOCATION, config)


def test_independent_block_members_are_local_explicit_and_label_bound() -> None:
    """An unaligned block start cannot change or hide quartet-member identity."""
    allocation = IndependentAllocation(
        allocation_id="test-explicit-members",
        split_namespace=SplitNamespace.DEBUG,
        blocks=(
            EpisodeBlock(
                suite=SuiteName.IID_PRIMARY,
                requested_path_length=2,
                first_episode_index=5,
                episode_count=8,
            ),
        ),
    )

    requests = tuple(iter_independent_requests(allocation, root_seed=41))

    assert tuple(request.episode_index for request in requests) == tuple(range(5, 13))
    assert tuple(request.quartet_member_index for request in requests) == (0, 1, 2, 3) * 2
    for request in requests:
        variants = allocate_independent_variants(
            AllocationLabelKey(
                "ofd-v1",
                request.split_namespace,
                request.suite,
                request.root_seed,
                request.requested_path_length,
                request.allocation_quartet_index,
            )
        )
        assert request.variant is variants[request.quartet_member_index]


@pytest.mark.parametrize("quartet_member_index", (False, -1, 4))
def test_independent_request_requires_an_exact_bounded_quartet_member(
    quartet_member_index: object,
) -> None:
    """Malformed explicit members must fail before independent generation."""
    with pytest.raises((TypeError, ValueError), match="quartet_member_index"):
        IndependentEpisodeRequest(
            split_namespace=SplitNamespace.DEBUG,
            suite=SuiteName.IID_PRIMARY,
            root_seed=41,
            episode_index=5,
            requested_path_length=2,
            variant=EpisodeVariant.POSITIVE,
            allocation_quartet_index=0,
            quartet_member_index=quartet_member_index,  # type: ignore[arg-type]
        )


def test_independent_coordinate_paths_do_not_reconstruct_quartet_members() -> None:
    """Lossy position/rank/variant reconstruction cannot re-enter identity paths."""
    paths = (
        Path("src/silent_cascade/env/generator.py"),
        Path("src/silent_cascade/env/invariants.py"),
        Path("src/silent_cascade/logging/manifest.py"),
        Path("src/silent_cascade/env/services.py"),
        Path("src/silent_cascade/env/reproducibility.py"),
    )
    forbidden: list[str] = []
    reconstruction_names = {"episode_index", "within_block", "rank"}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Mod)
                and isinstance(node.right, ast.Constant)
                and node.right.value == 4
                and any(
                    (isinstance(child, ast.Name) and child.id in reconstruction_names)
                    or (isinstance(child, ast.Attribute) and child.attr in reconstruction_names)
                    for child in ast.walk(node.left)
                )
            ):
                forbidden.append(f"{path}:{node.lineno}:modulo-four member reconstruction")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "index"
                and any(
                    (isinstance(child, ast.Name) and child.id == "variant")
                    or (isinstance(child, ast.Attribute) and child.attr == "variant")
                    for child in ast.walk(node)
                )
            ):
                forbidden.append(f"{path}:{node.lineno}:variant-rank member reconstruction")
        assert not any(
            isinstance(node, ast.Name) and node.id == "_INDEPENDENT_ALLOCATION_BLOCKS"
            for node in ast.walk(tree)
        ), f"{path} retains the Phase 1 invariant block table"

    assert forbidden == []


def test_gate_clock_blocks_are_nested_iid_prefixes() -> None:
    assert [
        (
            block.requested_path_length,
            block.source_first_episode_index,
            block.scale_0_1x_episode_count,
            block.scale_10x_episode_count,
        )
        for block in PHASE1_GATE_ALLOCATION.clock_blocks
    ] == [
        (2, 0, 1_668, 668),
        (3, 8_000, 1_668, 668),
        (4, 16_000, 1_664, 664),
    ]


def test_suite_specs_change_only_the_declared_axis() -> None:
    assert suite_spec(SuiteName.VALIDATION).path_lengths == (2, 3, 4)
    assert suite_spec(SuiteName.IID_PRIMARY).delay_log_uniform == (8.0, 64.0)
    assert suite_spec(SuiteName.OOD_DEPTH).path_lengths == (5, 6, 7, 8)
    assert suite_spec(SuiteName.OOD_DEPTH).delay_log_uniform == (16.0, 128.0)
    assert suite_spec(SuiteName.OOD_SHORT_DELAY).delay_log_uniform == (1.0, 8.0)
    assert suite_spec(SuiteName.OOD_LONG_DELAY).delay_log_uniform == (64.0, 1024.0)
    assert suite_spec(SuiteName.DISTRACTOR_FLOOD).distractor_link_records == (16, 48)


def test_template_contains_one_shared_unlabelled_link_topology(config: Phase1Config) -> None:
    request = next(
        item
        for item in iter_cohort_requests(
            CohortAllocation(
                allocation_id="test-template",
                split_namespace=SplitNamespace.DEBUG,
                blocks=(
                    CohortBlock(
                        suite=SuiteName.DISTRACTOR_FLOOD,
                        requested_path_length=4,
                        first_cohort_index=0,
                        cohort_count=1,
                    ),
                ),
            ),
            root_seed=41,
        )
    )
    template = sample_cohort_template(config, request, attempt=0)

    assert template.requested_path_length == 4
    assert len(template.relevant_edges) == 4
    assert 16 <= len(template.distractor_edges) <= 48
    assert not set(template.relevant_nodes) & set(template.distractor_source_nodes)
    assert not set(template.relevant_nodes) & set(template.unreachable_terminal_nodes)
    assert len(template.hazard_class_multiset) == 2
    assert all(0 <= item < 4 for item in template.hazard_class_multiset)
    assert template.episode_delay >= 8.0
    assert template.episode_delay <= 64.0
    assert all(0.1 <= gap <= 8.0 for gap in template.fact_gap_sequence)
    assert 0.1 <= template.activation_gap <= 8.0


def test_template_is_cohort_scoped_and_stable(config: Phase1Config) -> None:
    allocation = CohortAllocation(
        allocation_id="test-cohort-stability",
        split_namespace=SplitNamespace.DEBUG,
        blocks=(
            CohortBlock(
                suite=SuiteName.VALIDATION,
                requested_path_length=3,
                first_cohort_index=4,
                cohort_count=1,
            ),
        ),
    )
    request = next(iter_cohort_requests(allocation, root_seed=91))

    assert sample_cohort_template(config, request, attempt=0) == sample_cohort_template(
        config, request, attempt=0
    )
    template = sample_cohort_template(config, request, attempt=0)
    assert template.fact_gap_sequence
    assert len(template.hazard_class_multiset) == 2


def test_generic_debug_allocations_are_accepted_but_frozen_guards_reject_them(
    config: Phase1Config,
) -> None:
    debug_cohorts = CohortAllocation(
        allocation_id="test-debug-cohorts",
        split_namespace=SplitNamespace.DEBUG,
        blocks=(
            CohortBlock(
                suite=SuiteName.VALIDATION,
                requested_path_length=2,
                first_cohort_index=0,
                cohort_count=8,
            ),
        ),
    )
    debug_episodes = IndependentAllocation(
        allocation_id="test-debug-episodes",
        split_namespace=SplitNamespace.DEBUG,
        blocks=(
            EpisodeBlock(
                suite=SuiteName.IID_PRIMARY,
                requested_path_length=2,
                first_episode_index=0,
                episode_count=16,
            ),
        ),
    )

    assert len(tuple(iter_cohort_requests(debug_cohorts, root_seed=41))) == 8
    assert len(tuple(iter_independent_requests(debug_episodes, root_seed=41))) == 16
    with pytest.raises(ValueError, match="frozen validation allocation"):
        validate_validation_allocation(debug_cohorts, config)
    with pytest.raises(ValueError, match="frozen phase1 gate allocation"):
        validate_phase1_gate_allocation(debug_episodes, config)


def test_structural_allocations_reject_overlap_invalid_counts_and_non_nested_clock() -> None:
    with pytest.raises(ValueError, match="overlap"):
        IndependentAllocation(
            allocation_id="test-overlap",
            split_namespace=SplitNamespace.DEBUG,
            blocks=(
                EpisodeBlock(
                    suite=SuiteName.IID_PRIMARY,
                    requested_path_length=2,
                    first_episode_index=0,
                    episode_count=8,
                ),
                EpisodeBlock(
                    suite=SuiteName.IID_PRIMARY,
                    requested_path_length=3,
                    first_episode_index=4,
                    episode_count=8,
                ),
            ),
        )
    with pytest.raises(ValueError, match="divisible by four"):
        EpisodeBlock(
            suite=SuiteName.IID_PRIMARY,
            requested_path_length=2,
            first_episode_index=0,
            episode_count=6,
        )
    with pytest.raises(ValueError, match="nested"):
        IndependentAllocation(
            allocation_id="test-clock",
            split_namespace=SplitNamespace.DEBUG,
            blocks=(
                EpisodeBlock(
                    suite=SuiteName.IID_PRIMARY,
                    requested_path_length=2,
                    first_episode_index=0,
                    episode_count=8,
                ),
            ),
            clock_blocks=(
                ClockEpisodeBlock(
                    requested_path_length=2,
                    source_first_episode_index=0,
                    scale_0_1x_episode_count=4,
                    scale_10x_episode_count=5,
                ),
            ),
        )


def test_frozen_guards_reject_config_drift(config: Phase1Config) -> None:
    altered = config.model_copy(
        update={"data": config.data.model_copy(update={"train_delay_log_uniform": (8.0, 63.0)})}
    )
    with pytest.raises(ValueError, match="frozen"):
        validate_validation_allocation(VALIDATION_ALLOCATION, altered)


def valid_cohort_template(**changes: object) -> CohortTemplate:
    values: dict[str, object] = {
        "requested_path_length": 2,
        "relevant_nodes": (0, 1, 2),
        "relevant_edges": ((0, 1), (1, 2)),
        "distractor_edges": ((3, 4),),
        "unreachable_terminal_nodes": (61, 62, 63),
        "episode_delay": 8.0,
        "fact_gap_sequence": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        "activation_gap": 0.7,
        "hazard_class_multiset": (0, 1),
    }
    values.update(changes)
    return CohortTemplate(**values)  # type: ignore[arg-type]


def test_cohort_template_is_deeply_immutable_and_sampler_output_is_valid(
    config: Phase1Config,
) -> None:
    template = valid_cohort_template()
    with pytest.raises(FrozenInstanceError):
        template.episode_delay = 9.0  # type: ignore[misc]

    with pytest.raises(TypeError, match="tuple"):
        valid_cohort_template(
            relevant_nodes=[0, 1, 2],
            relevant_edges=[(0, 1), (1, 2)],
            distractor_edges=[],
            unreachable_terminal_nodes=[61, 62, 63],
            fact_gap_sequence=[0.1],
            hazard_class_multiset=[0, 1],
        )

    request = next(iter_cohort_requests(VALIDATION_ALLOCATION, root_seed=41))
    sampled = sample_cohort_template(config, request, attempt=0)
    assert sampled == CohortTemplate(
        requested_path_length=sampled.requested_path_length,
        relevant_nodes=sampled.relevant_nodes,
        relevant_edges=sampled.relevant_edges,
        distractor_edges=sampled.distractor_edges,
        unreachable_terminal_nodes=sampled.unreachable_terminal_nodes,
        episode_delay=sampled.episode_delay,
        fact_gap_sequence=sampled.fact_gap_sequence,
        activation_gap=sampled.activation_gap,
        hazard_class_multiset=sampled.hazard_class_multiset,
    )


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"requested_path_length": True}, "exact integer"),
        ({"relevant_nodes": (0, True, 2)}, "exact integer"),
        ({"episode_delay": 8}, "exact float"),
        ({"activation_gap": 1}, "exact float"),
        ({"fact_gap_sequence": (0.1, 1, 0.3, 0.4, 0.5, 0.6)}, "exact float"),
        ({"episode_delay": float("nan")}, "positive and finite"),
        ({"activation_gap": float("inf")}, "positive and finite"),
        ({"relevant_nodes": (0, 2, 3)}, "canonical"),
        ({"relevant_edges": ((0, 1), (1, 3))}, "canonical"),
        ({"relevant_edges": ((0, 1, 2),)}, "edge"),
        (
            {
                "distractor_edges": ((3, 4), (3, 4)),
                "fact_gap_sequence": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
            },
            "duplicate",
        ),
        (
            {
                "distractor_edges": ((3, 4), (4, 3)),
                "fact_gap_sequence": (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7),
            },
            "acyclic",
        ),
        ({"distractor_edges": ((2, 3),)}, "touch"),
        ({"unreachable_terminal_nodes": (61, 62)}, "exactly 3"),
        ({"unreachable_terminal_nodes": (61, 61, 63)}, "distinct"),
        ({"unreachable_terminal_nodes": (2, 62, 63)}, "distinct"),
        ({"hazard_class_multiset": (0,)}, "exactly 2"),
        ({"hazard_class_multiset": (0, True)}, "exact integer"),
        ({"hazard_class_multiset": (0, 4)}, "hazard"),
        ({"hazard_class_multiset": (1, 0)}, "canonical"),
    ],
)
def test_cohort_template_rejects_malformed_or_non_strict_values(
    changes: dict[str, object], error: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        valid_cohort_template(**changes)
