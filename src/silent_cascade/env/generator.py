"""Frozen suite specifications and deterministic Phase 1 allocation primitives.

This module deliberately stops before episode construction.  It establishes the
publicly stable allocation coordinates and cohort-scoped template recipe that
later generator tasks consume without consulting the oracle or private truth.
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from pydantic import Field, model_validator

from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import EpisodeVariant
from silent_cascade.rng import (
    AllocationLabelKey,
    CounterSeedKey,
    SeedStream,
    allocate_independent_variants,
    local_generator,
)
from silent_cascade.validation import StrictModel

_PRIMARY_SUITES = frozenset(
    {
        SuiteName.VALIDATION,
        SuiteName.IID_PRIMARY,
        SuiteName.OOD_DEPTH,
        SuiteName.OOD_SHORT_DELAY,
        SuiteName.OOD_LONG_DELAY,
        SuiteName.DISTRACTOR_FLOOD,
    }
)
_OBSERVATION_GAP_RANGE = (0.1, 8.0)
_TERMINAL_RECORD_COUNT = 3


def _require_exact_int(value: object, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _require_root_seed(root_seed: object) -> int:
    _require_exact_int(root_seed, "root_seed")
    if root_seed >= 2**128:
        raise ValueError("root_seed must be below 2**128")
    return root_seed


def _log_uniform(generator: np.random.Generator, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return float(math.exp(generator.uniform(math.log(lower), math.log(upper))))


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    """The one factor each primary suite changes from the IID recipe."""

    path_lengths: tuple[int, ...]
    delay_log_uniform: tuple[float, float]
    distractor_link_records: tuple[int, int]


_SUITE_SPECS = {
    SuiteName.VALIDATION: SuiteSpec((2, 3, 4), (8.0, 64.0), (0, 12)),
    SuiteName.IID_PRIMARY: SuiteSpec((2, 3, 4), (8.0, 64.0), (0, 12)),
    SuiteName.OOD_DEPTH: SuiteSpec((5, 6, 7, 8), (16.0, 128.0), (0, 12)),
    SuiteName.OOD_SHORT_DELAY: SuiteSpec((2, 3, 4), (1.0, 8.0), (0, 12)),
    SuiteName.OOD_LONG_DELAY: SuiteSpec((2, 3, 4), (64.0, 1024.0), (0, 12)),
    SuiteName.DISTRACTOR_FLOOD: SuiteSpec((2, 3, 4), (8.0, 64.0), (16, 48)),
}


def suite_spec(suite: SuiteName) -> SuiteSpec:
    """Return the fixed primary recipe for ``suite``."""

    if not isinstance(suite, SuiteName):
        raise TypeError("suite must be a SuiteName")
    try:
        return _SUITE_SPECS[suite]
    except KeyError as error:
        raise ValueError(f"suite {suite.value} has no primary suite specification") from error


class CohortBlock(StrictModel):
    suite: SuiteName
    requested_path_length: int
    first_cohort_index: int
    cohort_count: int

    @model_validator(mode="after")
    def validate_block(self) -> "CohortBlock":
        _validate_block_values(
            self.suite,
            self.requested_path_length,
            self.first_cohort_index,
            self.cohort_count,
            "cohort",
        )
        return self


class EpisodeBlock(StrictModel):
    suite: SuiteName
    requested_path_length: int
    first_episode_index: int
    episode_count: int

    @model_validator(mode="after")
    def validate_block(self) -> "EpisodeBlock":
        _validate_block_values(
            self.suite,
            self.requested_path_length,
            self.first_episode_index,
            self.episode_count,
            "episode",
        )
        if self.episode_count % 4:
            raise ValueError("independent episode_count must be divisible by four")
        return self


class ClockEpisodeBlock(StrictModel):
    requested_path_length: int
    source_first_episode_index: int
    scale_0_1x_episode_count: int
    scale_10x_episode_count: int

    @model_validator(mode="after")
    def validate_block(self) -> "ClockEpisodeBlock":
        _require_exact_int(self.requested_path_length, "requested_path_length", minimum=1)
        _require_exact_int(self.source_first_episode_index, "source_first_episode_index")
        _require_exact_int(self.scale_0_1x_episode_count, "scale_0_1x_episode_count", minimum=1)
        _require_exact_int(self.scale_10x_episode_count, "scale_10x_episode_count", minimum=1)
        if self.scale_10x_episode_count > self.scale_0_1x_episode_count:
            raise ValueError("clock episode counts must be nested")
        return self


def _validate_block_values(
    suite: SuiteName,
    requested_path_length: int,
    first_index: int,
    count: int,
    label: str,
) -> None:
    if suite not in _PRIMARY_SUITES:
        raise ValueError(f"{label} blocks support only primary suites")
    _require_exact_int(requested_path_length, "requested_path_length", minimum=1)
    if requested_path_length not in suite_spec(suite).path_lengths:
        raise ValueError("requested_path_length is unsupported for suite")
    _require_exact_int(first_index, f"first_{label}_index")
    _require_exact_int(count, f"{label}_count", minimum=1)


def _validate_nonoverlap(
    blocks: tuple[CohortBlock, ...] | tuple[EpisodeBlock, ...], *, index_name: str
) -> None:
    by_suite: dict[SuiteName, list[tuple[int, int]]] = {}
    for block in blocks:
        start = getattr(block, index_name)
        count = getattr(
            block, "cohort_count" if index_name == "first_cohort_index" else "episode_count"
        )
        by_suite.setdefault(block.suite, []).append((start, start + count))
    for intervals in by_suite.values():
        previous_end = -1
        for start, end in sorted(intervals):
            if start < previous_end:
                raise ValueError("same-suite allocation blocks may not overlap")
            previous_end = end


class CohortAllocation(StrictModel):
    allocation_id: str = Field(min_length=1)
    split_namespace: SplitNamespace
    blocks: tuple[CohortBlock, ...]

    @model_validator(mode="after")
    def validate_allocation(self) -> "CohortAllocation":
        if not self.blocks:
            raise ValueError("cohort allocation must contain blocks")
        _validate_nonoverlap(self.blocks, index_name="first_cohort_index")
        return self


class IndependentAllocation(StrictModel):
    allocation_id: str = Field(min_length=1)
    split_namespace: SplitNamespace
    blocks: tuple[EpisodeBlock, ...]
    clock_blocks: tuple[ClockEpisodeBlock, ...] = ()

    @model_validator(mode="after")
    def validate_allocation(self) -> "IndependentAllocation":
        if not self.blocks:
            raise ValueError("independent allocation must contain blocks")
        _validate_nonoverlap(self.blocks, index_name="first_episode_index")
        for clock_block in self.clock_blocks:
            matching = [
                block
                for block in self.blocks
                if block.suite is SuiteName.IID_PRIMARY
                and block.requested_path_length == clock_block.requested_path_length
                and block.first_episode_index <= clock_block.source_first_episode_index
                and clock_block.source_first_episode_index + clock_block.scale_0_1x_episode_count
                <= block.first_episode_index + block.episode_count
            ]
            if len(matching) != 1:
                raise ValueError("clock episode blocks must be nested in one IID source block")
        return self


@dataclass(frozen=True, slots=True)
class CohortRequest:
    split_namespace: SplitNamespace
    suite: SuiteName
    root_seed: int
    cohort_index: int
    requested_path_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.split_namespace, SplitNamespace) or not isinstance(
            self.suite, SuiteName
        ):
            raise TypeError("split_namespace and suite must be strict enums")
        _require_root_seed(self.root_seed)
        _require_exact_int(self.cohort_index, "cohort_index")
        _validate_block_values(
            self.suite, self.requested_path_length, self.cohort_index, 1, "cohort"
        )


@dataclass(frozen=True, slots=True)
class IndependentEpisodeRequest:
    split_namespace: SplitNamespace
    suite: SuiteName
    root_seed: int
    episode_index: int
    requested_path_length: int
    variant: EpisodeVariant
    allocation_quartet_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.split_namespace, SplitNamespace) or not isinstance(
            self.suite, SuiteName
        ):
            raise TypeError("split_namespace and suite must be strict enums")
        _require_root_seed(self.root_seed)
        _require_exact_int(self.episode_index, "episode_index")
        _require_exact_int(self.allocation_quartet_index, "allocation_quartet_index")
        _validate_block_values(
            self.suite, self.requested_path_length, self.episode_index, 1, "episode"
        )
        if not isinstance(self.variant, EpisodeVariant):
            raise TypeError("variant must be an EpisodeVariant")


@dataclass(frozen=True, slots=True)
class CohortTemplate:
    requested_path_length: int
    relevant_nodes: tuple[int, ...]
    relevant_edges: tuple[tuple[int, int], ...]
    distractor_edges: tuple[tuple[int, int], ...]
    unreachable_terminal_nodes: tuple[int, int, int]
    episode_delay: float
    fact_gap_sequence: tuple[float, ...]
    activation_gap: float
    hazard_class_multiset: tuple[int, int]

    @property
    def distractor_source_nodes(self) -> tuple[int, ...]:
        """Sources are exposed for topology validation, never as labels."""

        return tuple(source for source, _ in self.distractor_edges)


VALIDATION_ALLOCATION = CohortAllocation(
    allocation_id="phase1-validation-v1",
    split_namespace=SplitNamespace.VALIDATION,
    blocks=(
        CohortBlock(
            suite=SuiteName.VALIDATION,
            requested_path_length=2,
            first_cohort_index=0,
            cohort_count=834,
        ),
        CohortBlock(
            suite=SuiteName.VALIDATION,
            requested_path_length=3,
            first_cohort_index=834,
            cohort_count=833,
        ),
        CohortBlock(
            suite=SuiteName.VALIDATION,
            requested_path_length=4,
            first_cohort_index=1667,
            cohort_count=833,
        ),
    ),
)


PHASE1_GATE_ALLOCATION = IndependentAllocation(
    allocation_id="phase1-gate-v1",
    split_namespace=SplitNamespace.PHASE1_GATE,
    blocks=(
        EpisodeBlock(
            suite=SuiteName.IID_PRIMARY,
            requested_path_length=2,
            first_episode_index=0,
            episode_count=8_000,
        ),
        EpisodeBlock(
            suite=SuiteName.IID_PRIMARY,
            requested_path_length=3,
            first_episode_index=8_000,
            episode_count=8_000,
        ),
        EpisodeBlock(
            suite=SuiteName.IID_PRIMARY,
            requested_path_length=4,
            first_episode_index=16_000,
            episode_count=8_000,
        ),
        EpisodeBlock(
            suite=SuiteName.OOD_DEPTH,
            requested_path_length=5,
            first_episode_index=0,
            episode_count=4_000,
        ),
        EpisodeBlock(
            suite=SuiteName.OOD_DEPTH,
            requested_path_length=6,
            first_episode_index=4_000,
            episode_count=4_000,
        ),
        EpisodeBlock(
            suite=SuiteName.OOD_DEPTH,
            requested_path_length=7,
            first_episode_index=8_000,
            episode_count=4_000,
        ),
        EpisodeBlock(
            suite=SuiteName.OOD_DEPTH,
            requested_path_length=8,
            first_episode_index=12_000,
            episode_count=4_000,
        ),
        EpisodeBlock(
            suite=SuiteName.OOD_SHORT_DELAY,
            requested_path_length=2,
            first_episode_index=0,
            episode_count=8_000,
        ),
        EpisodeBlock(
            suite=SuiteName.OOD_SHORT_DELAY,
            requested_path_length=3,
            first_episode_index=8_000,
            episode_count=8_000,
        ),
        EpisodeBlock(
            suite=SuiteName.OOD_SHORT_DELAY,
            requested_path_length=4,
            first_episode_index=16_000,
            episode_count=8_000,
        ),
        EpisodeBlock(
            suite=SuiteName.OOD_LONG_DELAY,
            requested_path_length=2,
            first_episode_index=0,
            episode_count=4_000,
        ),
        EpisodeBlock(
            suite=SuiteName.OOD_LONG_DELAY,
            requested_path_length=3,
            first_episode_index=4_000,
            episode_count=4_000,
        ),
        EpisodeBlock(
            suite=SuiteName.OOD_LONG_DELAY,
            requested_path_length=4,
            first_episode_index=8_000,
            episode_count=4_000,
        ),
        EpisodeBlock(
            suite=SuiteName.DISTRACTOR_FLOOD,
            requested_path_length=2,
            first_episode_index=0,
            episode_count=8_000,
        ),
        EpisodeBlock(
            suite=SuiteName.DISTRACTOR_FLOOD,
            requested_path_length=3,
            first_episode_index=8_000,
            episode_count=8_000,
        ),
        EpisodeBlock(
            suite=SuiteName.DISTRACTOR_FLOOD,
            requested_path_length=4,
            first_episode_index=16_000,
            episode_count=8_000,
        ),
    ),
    clock_blocks=(
        ClockEpisodeBlock(
            requested_path_length=2,
            source_first_episode_index=0,
            scale_0_1x_episode_count=1_668,
            scale_10x_episode_count=668,
        ),
        ClockEpisodeBlock(
            requested_path_length=3,
            source_first_episode_index=8_000,
            scale_0_1x_episode_count=1_668,
            scale_10x_episode_count=668,
        ),
        ClockEpisodeBlock(
            requested_path_length=4,
            source_first_episode_index=16_000,
            scale_0_1x_episode_count=1_664,
            scale_10x_episode_count=664,
        ),
    ),
)


def iter_cohort_requests(
    allocation: CohortAllocation,
    root_seed: int,
) -> Iterator[CohortRequest]:
    """Yield allocation coordinates without drawing episode-local randomness."""

    if not isinstance(allocation, CohortAllocation):
        raise TypeError("allocation must be a CohortAllocation")
    _require_root_seed(root_seed)
    for block in allocation.blocks:
        for offset in range(block.cohort_count):
            yield CohortRequest(
                split_namespace=allocation.split_namespace,
                suite=block.suite,
                root_seed=root_seed,
                cohort_index=block.first_cohort_index + offset,
                requested_path_length=block.requested_path_length,
            )


def iter_independent_requests(
    allocation: IndependentAllocation,
    root_seed: int,
) -> Iterator[IndependentEpisodeRequest]:
    """Yield independent episode coordinates with private quartet labels only."""

    if not isinstance(allocation, IndependentAllocation):
        raise TypeError("allocation must be an IndependentAllocation")
    _require_root_seed(root_seed)
    quartet_index = 0
    for block in allocation.blocks:
        for offset in range(0, block.episode_count, 4):
            variants = allocate_independent_variants(
                AllocationLabelKey(
                    generator_version="ofd-v1",
                    split_namespace=allocation.split_namespace,
                    suite=block.suite,
                    root_seed=root_seed,
                    requested_path_length=block.requested_path_length,
                    allocation_quartet_index=quartet_index,
                )
            )
            for within_quartet, variant in enumerate(variants):
                yield IndependentEpisodeRequest(
                    split_namespace=allocation.split_namespace,
                    suite=block.suite,
                    root_seed=root_seed,
                    episode_index=block.first_episode_index + offset + within_quartet,
                    requested_path_length=block.requested_path_length,
                    variant=variant,
                    allocation_quartet_index=quartet_index,
                )
            quartet_index += 1


def iter_phase1_gate_requests(root_seed: int) -> Iterator[IndependentEpisodeRequest]:
    """Exact convenience iterator for the production Phase 1 gate."""

    yield from iter_independent_requests(PHASE1_GATE_ALLOCATION, root_seed)


def _cohort_stream(request: CohortRequest, stream: SeedStream, attempt: int) -> np.random.Generator:
    return local_generator(
        CounterSeedKey(
            generator_version="ofd-v1",
            split_namespace=request.split_namespace,
            suite=request.suite,
            root_seed=request.root_seed,
            cohort_index=request.cohort_index,
            member_index=-1,
            stream=stream,
            attempt=attempt,
        )
    )


def sample_cohort_template(
    config: Phase1Config, request: CohortRequest, attempt: int
) -> CohortTemplate:
    """Sample the complete matched topology from cohort-only RNG domains.

    No member-local stream or oracle result participates: those operations are
    intentionally deferred to construction and validation tasks.
    """

    if not isinstance(config, Phase1Config):
        raise TypeError("config must be a Phase1Config")
    if not isinstance(request, CohortRequest):
        raise TypeError("request must be a CohortRequest")
    _require_exact_int(attempt, "attempt")
    if attempt >= config.data.max_generation_attempts:
        raise ValueError("attempt exceeds config.max_generation_attempts")

    spec = suite_spec(request.suite)
    if request.requested_path_length not in spec.path_lengths:
        raise ValueError("request path length is unsupported for suite")
    template_rng = _cohort_stream(request, SeedStream.TEMPLATE, attempt)
    distractor_count = int(
        template_rng.integers(spec.distractor_link_records[0], spec.distractor_link_records[1] + 1)
    )
    episode_delay = _log_uniform(template_rng, spec.delay_log_uniform)
    hazard_class_multiset = tuple(
        sorted(
            int(template_rng.integers(0, config.data.hazard_types))
            for _ in range(config.data.terminal_hazard_records)
        )
    )
    if len(hazard_class_multiset) != 2:
        raise ValueError("Phase 1 requires exactly two hazard records")

    relevant_nodes = tuple(range(request.requested_path_length + 1))
    relevant_edges = tuple(pairwise(relevant_nodes))
    unreachable_terminal_nodes = (61, 62, 63)
    distractor_nodes = tuple(
        node
        for node in range(request.requested_path_length + 1, config.data.max_entities - 3)
        if node not in unreachable_terminal_nodes
    )
    candidates = tuple(
        (source, target)
        for source in distractor_nodes
        for target in distractor_nodes
        if source < target
    )
    if distractor_count > len(candidates):
        raise ValueError("distractor topology cannot fit available entity IDs")
    structure_rng = _cohort_stream(request, SeedStream.STRUCTURE, attempt)
    selected = structure_rng.choice(len(candidates), size=distractor_count, replace=False)
    distractor_edges = tuple(sorted(candidates[int(index)] for index in selected))
    _validate_template_topology(
        relevant_nodes, relevant_edges, distractor_edges, unreachable_terminal_nodes
    )

    fact_count = len(relevant_edges) + len(distractor_edges) + _TERMINAL_RECORD_COUNT
    if fact_count > config.data.primary_memory_capacity:
        raise ValueError("template facts exceed the fixed memory capacity")
    timestamps_rng = _cohort_stream(request, SeedStream.TIMESTAMPS, attempt)
    fact_gap_sequence = tuple(
        _log_uniform(timestamps_rng, _OBSERVATION_GAP_RANGE) for _ in range(fact_count)
    )
    activation_gap = _log_uniform(timestamps_rng, _OBSERVATION_GAP_RANGE)
    return CohortTemplate(
        requested_path_length=request.requested_path_length,
        relevant_nodes=relevant_nodes,
        relevant_edges=relevant_edges,
        distractor_edges=distractor_edges,
        unreachable_terminal_nodes=unreachable_terminal_nodes,
        episode_delay=episode_delay,
        fact_gap_sequence=fact_gap_sequence,
        activation_gap=activation_gap,
        hazard_class_multiset=(hazard_class_multiset[0], hazard_class_multiset[1]),
    )


def _validate_template_topology(
    relevant_nodes: tuple[int, ...],
    relevant_edges: tuple[tuple[int, int], ...],
    distractor_edges: tuple[tuple[int, int], ...],
    unreachable_terminal_nodes: tuple[int, int, int],
) -> None:
    relevant = set(relevant_nodes)
    if len(relevant) != len(relevant_nodes) or len(set(relevant_edges)) != len(relevant_edges):
        raise ValueError("relevant topology has duplicate nodes or edges")
    if relevant_edges != tuple(pairwise(relevant_nodes)):
        raise ValueError("relevant topology must be one canonical path")
    if set(unreachable_terminal_nodes) & relevant or len(set(unreachable_terminal_nodes)) != 3:
        raise ValueError("unreachable terminals must be distinct from the relevant component")
    if len(set(distractor_edges)) != len(distractor_edges):
        raise ValueError("distractor topology has duplicate edges")
    for source, target in distractor_edges:
        if source in relevant or target in relevant:
            raise ValueError("distractors may not touch the relevant component")
        if source >= target:
            raise ValueError("distractor topology must be acyclic")


def _validate_frozen_suite_config(config: Phase1Config) -> None:
    data = config.data
    if (
        data.train_path_lengths != (2, 3, 4)
        or data.iid_test_path_lengths != (2, 3, 4)
        or data.ood_depth_path_lengths != (5, 6, 7, 8)
        or data.ood_short_delay_path_lengths != (2, 3, 4)
        or data.train_delay_log_uniform != (8.0, 64.0)
        or data.ood_depth_delay_log_uniform != (16.0, 128.0)
        or data.ood_short_delay_log_uniform != (1.0, 8.0)
        or data.ood_long_delay_log_uniform != (64.0, 1024.0)
        or data.train_distractor_link_records != (0, 12)
        or data.ood_distractor_link_records != (16, 48)
        or data.observation_gap_log_uniform != _OBSERVATION_GAP_RANGE
        or data.max_entities != 64
        or data.hazard_types != 4
        or data.terminal_hazard_records != 2
        or data.terminal_safe_records != 1
        or data.primary_memory_capacity != 64
    ):
        raise ValueError("config does not match the frozen Phase 1 suite specification")


def validate_validation_allocation(allocation: CohortAllocation, config: Phase1Config) -> None:
    """Reject every non-production validation allocation at a publication seam."""

    if not isinstance(allocation, CohortAllocation) or allocation != VALIDATION_ALLOCATION:
        raise ValueError("allocation must equal the frozen validation allocation")
    if not isinstance(config, Phase1Config):
        raise TypeError("config must be a Phase1Config")
    _validate_frozen_suite_config(config)
    if config.data.manifest_sizes.validation != 10_000:
        raise ValueError("config does not match the frozen validation allocation")
    if sum(block.cohort_count for block in allocation.blocks) != 2_500:
        raise ValueError("frozen validation allocation must contain 2,500 cohorts")


def validate_phase1_gate_allocation(
    allocation: IndependentAllocation, config: Phase1Config
) -> None:
    """Reject every non-production independent allocation at a gate seam."""

    if not isinstance(allocation, IndependentAllocation) or allocation != PHASE1_GATE_ALLOCATION:
        raise ValueError("allocation must equal the frozen phase1 gate allocation")
    if not isinstance(config, Phase1Config):
        raise TypeError("config must be a Phase1Config")
    _validate_frozen_suite_config(config)
    expected_gate = {
        "iid_primary": {2: 8_000, 3: 8_000, 4: 8_000},
        "ood_depth": {5: 4_000, 6: 4_000, 7: 4_000, 8: 4_000},
        "ood_short_delay": {2: 8_000, 3: 8_000, 4: 8_000},
        "ood_long_delay": {2: 4_000, 3: 4_000, 4: 4_000},
        "distractor_flood": {2: 8_000, 3: 8_000, 4: 8_000},
        "clock_parent_episodes": {2: 1_668, 3: 1_668, 4: 1_664},
        "clock_10x_episodes": {2: 668, 3: 668, 4: 664},
    }
    if config.data.phase1_gate.model_dump() != expected_gate:
        raise ValueError("config does not match the frozen phase1 gate allocation")
    if sum(block.episode_count for block in allocation.blocks) != 100_000:
        raise ValueError("frozen phase1 gate allocation must contain 100,000 episodes")
