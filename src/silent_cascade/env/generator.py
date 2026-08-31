"""Frozen suite specifications and deterministic Phase 1 allocation primitives.

This module deliberately stops before episode construction.  It establishes the
publicly stable allocation coordinates and cohort-scoped template recipe that
later generator tasks consume without consulting the oracle or private truth.
"""

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import pairwise

import numpy as np
from pydantic import Field, model_validator

from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    EpisodeBundle,
    EpisodeKey,
    EpisodeRecipe,
    EpisodeTruth,
    EpisodeVariant,
    IndependentEpisodeCoordinate,
    MatchedEpisodeCoordinate,
    PublicEpisode,
    StressMetadata,
)
from silent_cascade.env.invariants import validate_cohort_invariants, validate_episode_invariants
from silent_cascade.env.timing import action_window
from silent_cascade.errors import GenerationError
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.rng import (
    COHORT_SCOPED_STREAMS,
    AllocationLabelKey,
    CounterSeedKey,
    IndependentCounterSeedKey,
    IndependentPublicIdKey,
    PublicIdBatchKey,
    SeedStream,
    allocate_independent_public_id,
    allocate_independent_variants,
    allocate_public_ids,
    derive_counter_seed,
    derive_independent_counter_seed,
    independent_local_generator,
    local_generator,
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
_STRUCTURAL_STRESS_SUITES = frozenset(
    {
        SuiteName.BRANCHING_STRESS,
        SuiteName.CYCLES_STRESS,
        SuiteName.MEMORY_OVERFLOW_STRESS,
        SuiteName.NULL_NEAR_MISS_STRESS,
        SuiteName.CONTRADICTION_STRESS,
        SuiteName.MINIMUM_DURATION_STRESS,
        SuiteName.CHECKPOINT_STRESS,
    }
)
_STRESS_RESERVED_NODES = frozenset(range(49, 61))
_OBSERVATION_GAP_RANGE = (0.1, 8.0)
_TERMINAL_RECORD_COUNT = 3
_MAX_ENTITY_ID = 63
_HAZARD_TYPE_COUNT = 4


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


def _require_positive_float(value: object, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _require_int_tuple(
    value: object,
    name: str,
    *,
    length: int | None = None,
    minimum: int = 0,
    maximum: int = _MAX_ENTITY_ID,
) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if length is not None and len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} items")
    for item in value:
        _require_exact_int(item, name, minimum=minimum)
        if item > maximum:
            raise ValueError(f"{name} is out of bounds")
    return value


def _require_edge_tuple(value: object, name: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    for edge in value:
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise TypeError(f"{name} must contain two-item tuple edges")
        _require_exact_int(edge[0], name)
        _require_exact_int(edge[1], name)
        if edge[0] > _MAX_ENTITY_ID or edge[1] > _MAX_ENTITY_ID:
            raise ValueError(f"{name} is out of bounds")
    return value


def _require_positive_float_tuple(value: object, name: str, *, length: int) -> tuple[float, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} items")
    for item in value:
        _require_positive_float(item, name)
    return value


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
    SuiteName.BRANCHING_STRESS: SuiteSpec((2, 3, 4), (8.0, 64.0), (0, 12)),
    SuiteName.CYCLES_STRESS: SuiteSpec((2, 3, 4), (8.0, 64.0), (0, 12)),
    SuiteName.MEMORY_OVERFLOW_STRESS: SuiteSpec((2, 3, 4), (8.0, 64.0), (0, 12)),
    SuiteName.NULL_NEAR_MISS_STRESS: SuiteSpec((2, 3, 4), (8.0, 64.0), (0, 12)),
    SuiteName.CONTRADICTION_STRESS: SuiteSpec((2, 3, 4), (8.0, 64.0), (0, 12)),
    SuiteName.MINIMUM_DURATION_STRESS: SuiteSpec((2, 3, 4), (8.0, 64.0), (0, 12)),
    SuiteName.CHECKPOINT_STRESS: SuiteSpec((2, 3, 4), (8.0, 64.0), (0, 12)),
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
        if self.suite not in _PRIMARY_SUITES | _STRUCTURAL_STRESS_SUITES:
            raise ValueError("independent episode suite is not supported")
        _require_exact_int(self.requested_path_length, "requested_path_length", minimum=1)
        if self.requested_path_length not in suite_spec(self.suite).path_lengths:
            raise ValueError("requested_path_length is unsupported for suite")
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

    def __post_init__(self) -> None:
        _require_exact_int(self.requested_path_length, "requested_path_length", minimum=1)
        relevant_nodes = _require_int_tuple(
            self.relevant_nodes,
            "relevant_nodes",
            length=self.requested_path_length + 1,
        )
        relevant_edges = _require_edge_tuple(self.relevant_edges, "relevant_edges")
        distractor_edges = _require_edge_tuple(self.distractor_edges, "distractor_edges")
        unreachable_terminal_nodes = _require_int_tuple(
            self.unreachable_terminal_nodes,
            "unreachable_terminal_nodes",
            length=_TERMINAL_RECORD_COUNT,
        )
        _require_positive_float(self.episode_delay, "episode_delay")
        fact_count = len(relevant_edges) + len(distractor_edges) + _TERMINAL_RECORD_COUNT
        if fact_count > 64:
            raise ValueError("template facts exceed the fixed memory capacity")
        _require_positive_float_tuple(
            self.fact_gap_sequence, "fact_gap_sequence", length=fact_count
        )
        _require_positive_float(self.activation_gap, "activation_gap")
        hazard_class_multiset = _require_int_tuple(
            self.hazard_class_multiset,
            "hazard_class_multiset",
            length=2,
            maximum=_HAZARD_TYPE_COUNT - 1,
        )
        if relevant_nodes != tuple(range(self.requested_path_length + 1)):
            raise ValueError("relevant_nodes must be one canonical path")
        if relevant_edges != tuple(pairwise(relevant_nodes)):
            raise ValueError("relevant_edges must be one canonical path")
        if len(set(unreachable_terminal_nodes)) != _TERMINAL_RECORD_COUNT:
            raise ValueError("unreachable terminal nodes must be distinct")
        if set(unreachable_terminal_nodes) & set(relevant_nodes):
            raise ValueError("unreachable terminal nodes must be distinct from relevant nodes")
        if hazard_class_multiset != tuple(sorted(hazard_class_multiset)):
            raise ValueError("hazard_class_multiset must be canonical")
        _validate_template_topology(
            relevant_nodes,
            relevant_edges,
            distractor_edges,
            (
                unreachable_terminal_nodes[0],
                unreachable_terminal_nodes[1],
                unreachable_terminal_nodes[2],
            ),
        )

    @property
    def distractor_source_nodes(self) -> tuple[int, ...]:
        """Sources are exposed for topology validation, never as labels."""

        return tuple(source for source, _ in self.distractor_edges)


@dataclass(frozen=True, slots=True)
class RejectionDiagnostic:
    """One aggregate, environment-private reason for a rejected cohort draw."""

    reason: str
    count: int

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("rejection reason must be a nonempty string")
        _require_exact_int(self.count, "rejection count", minimum=1)


@dataclass(frozen=True, slots=True)
class GenerationCohort:
    """The four private bundles produced from one matched validation recipe."""

    request: CohortRequest
    accepted_attempt: int
    episodes: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle]
    seed_tokens: tuple[str, ...]
    rejections: tuple[RejectionDiagnostic, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, CohortRequest):
            raise TypeError("request must be a CohortRequest")
        _require_exact_int(self.accepted_attempt, "accepted_attempt")
        if len(self.episodes) != 4 or not all(
            isinstance(bundle, EpisodeBundle) for bundle in self.episodes
        ):
            raise ValueError("matched generation requires exactly four episode bundles")
        if not isinstance(self.seed_tokens, tuple) or not all(
            isinstance(token, str) for token in self.seed_tokens
        ):
            raise TypeError("seed_tokens must be a tuple of strings")
        if not isinstance(self.rejections, tuple) or not all(
            isinstance(item, RejectionDiagnostic) for item in self.rejections
        ):
            raise TypeError("rejections must be rejection diagnostics")


VALIDATION_ALLOCATION = CohortAllocation(
    allocation_id="validation-v1",
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
    allocation_id="phase1-independent-gate-v1",
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


_MATCHED_VARIANTS = (
    EpisodeVariant.POSITIVE,
    EpisodeVariant.POSITIVE,
    EpisodeVariant.SAFE_NEGATIVE,
    EpisodeVariant.DISCONNECTED_NEGATIVE,
)


def _member_stream(
    request: CohortRequest,
    member_index: int,
    stream: SeedStream,
    attempt: int,
) -> np.random.Generator:
    if stream in COHORT_SCOPED_STREAMS:
        raise ValueError("member streams cannot use a cohort-scoped RNG domain")
    return local_generator(
        CounterSeedKey(
            generator_version="ofd-v1",
            split_namespace=request.split_namespace,
            suite=request.suite,
            root_seed=request.root_seed,
            cohort_index=request.cohort_index,
            member_index=member_index,
            stream=stream,
            attempt=attempt,
        )
    )


def _seed_token(
    request: CohortRequest,
    member_index: int,
    stream: SeedStream,
    attempt: int,
) -> str:
    return derive_counter_seed(
        CounterSeedKey(
            generator_version="ofd-v1",
            split_namespace=request.split_namespace,
            suite=request.suite,
            root_seed=request.root_seed,
            cohort_index=request.cohort_index,
            member_index=member_index,
            stream=stream,
            attempt=attempt,
        )
    ).token


def _matched_seed_tokens(request: CohortRequest, attempt: int) -> tuple[str, ...]:
    cohort_tokens = tuple(
        _seed_token(request, -1, stream, attempt)
        for stream in (
            SeedStream.LABEL,
            SeedStream.TEMPLATE,
            SeedStream.STRUCTURE,
            SeedStream.TIMESTAMPS,
        )
    )
    member_tokens = tuple(
        _seed_token(request, member_index, stream, attempt)
        for member_index in range(4)
        for stream in (
            SeedStream.NODE_PERMUTATION,
            SeedStream.TERMINALS,
            SeedStream.PRESENTATION,
        )
    )
    return (*cohort_tokens, *member_tokens)


def _matched_variants(request: CohortRequest, attempt: int) -> tuple[EpisodeVariant, ...]:
    label_rng = _cohort_stream(request, SeedStream.LABEL, attempt)
    order = label_rng.permutation(len(_MATCHED_VARIANTS))
    return tuple(_MATCHED_VARIANTS[int(index)] for index in order)


def _build_matched_member(
    config: Phase1Config,
    request: CohortRequest,
    template: CohortTemplate,
    variant: EpisodeVariant,
    member_index: int,
    attempt: int,
    rejection_reasons: tuple[str, ...],
    public_id: str,
) -> EpisodeBundle:
    node_rng = _member_stream(request, member_index, SeedStream.NODE_PERMUTATION, attempt)
    node_permutation = tuple(int(item) for item in node_rng.permutation(config.data.max_entities))
    relabel = {
        canonical: node_permutation[canonical] for canonical in range(config.data.max_entities)
    }
    relevant_nodes = tuple(relabel[node] for node in template.relevant_nodes)
    facts: list[tuple[LinkFact | HazardFact | SafeFact, str]] = [
        (LinkFact(relabel[source], relabel[target]), "relevant_link")
        for source, target in template.relevant_edges
    ]
    facts.extend(
        (LinkFact(relabel[source], relabel[target]), "distractor_link")
        for source, target in template.distractor_edges
    )

    terminals_rng = _member_stream(request, member_index, SeedStream.TERMINALS, attempt)
    hazard_types = tuple(
        int(item) for item in terminals_rng.permutation(template.hazard_class_multiset)
    )
    unreachable_nodes = tuple(
        relabel[template.unreachable_terminal_nodes[int(index)]]
        for index in terminals_rng.permutation(_TERMINAL_RECORD_COUNT)
    )
    target = relevant_nodes[-1]
    if variant is EpisodeVariant.POSITIVE:
        facts.extend(
            (
                (HazardFact(target, hazard_types[0], template.episode_delay), "reachable_terminal"),
                (
                    HazardFact(unreachable_nodes[0], hazard_types[1], template.episode_delay),
                    "decoy_terminal",
                ),
                (SafeFact(unreachable_nodes[1]), "decoy_terminal"),
            )
        )
    elif variant is EpisodeVariant.SAFE_NEGATIVE:
        facts.extend(
            (
                (SafeFact(target), "reachable_terminal"),
                (
                    HazardFact(unreachable_nodes[0], hazard_types[0], template.episode_delay),
                    "decoy_terminal",
                ),
                (
                    HazardFact(unreachable_nodes[1], hazard_types[1], template.episode_delay),
                    "decoy_terminal",
                ),
            )
        )
    else:
        facts.extend(
            (
                (
                    HazardFact(unreachable_nodes[0], hazard_types[0], template.episode_delay),
                    "decoy_terminal",
                ),
                (
                    HazardFact(unreachable_nodes[1], hazard_types[1], template.episode_delay),
                    "decoy_terminal",
                ),
                (SafeFact(unreachable_nodes[2]), "decoy_terminal"),
            )
        )

    presentation_rng = _member_stream(request, member_index, SeedStream.PRESENTATION, attempt)
    order = presentation_rng.permutation(len(facts))
    presentation = tuple(facts[int(index)] for index in order)
    if len(presentation) != len(template.fact_gap_sequence):
        raise ValueError("fact presentation and timestamp template disagree")
    timestamp = 0.0
    events: list[ExternalEvent] = []
    relevant_link_ids: dict[int, int] = {}
    terminal_record_id: int | None = None
    for event_id, ((payload, role), gap) in enumerate(
        zip(presentation, template.fact_gap_sequence, strict=True)
    ):
        timestamp += gap
        events.append(ExternalEvent(event_id, timestamp, ExternalEventKind.FACT, payload))
        if role == "relevant_link":
            assert isinstance(payload, LinkFact)
            relevant_link_ids[payload.source_node] = event_id
        elif role == "reachable_terminal":
            terminal_record_id = event_id
    activation_time = timestamp + template.activation_gap
    fact_count = len(events)
    events.append(
        ExternalEvent(
            fact_count,
            activation_time,
            ExternalEventKind.ACTIVATE,
            ActivationPayload(relevant_nodes[0]),
        )
    )
    relevant_record_ids = tuple(relevant_link_ids[node] for node in relevant_nodes[:-1])
    if variant is not EpisodeVariant.DISCONNECTED_NEGATIVE:
        if terminal_record_id is None:
            raise ValueError("reachable terminal record was not constructed")
        relevant_record_ids = (*relevant_record_ids, terminal_record_id)
    relevant_hazard_type = hazard_types[0] if variant is EpisodeVariant.POSITIVE else None
    private_terminal = ExternalEvent(
        fact_count + 1,
        activation_time + template.episode_delay,
        ExternalEventKind.OUTCOME if variant is EpisodeVariant.POSITIVE else ExternalEventKind.END,
        None,
    )
    configured_window = action_window(
        activation_time,
        template.episode_delay,
        config.data.oracle_timing,
    )
    action_start = configured_window.start if variant is EpisodeVariant.POSITIVE else None
    action_end = configured_window.end if variant is EpisodeVariant.POSITIVE else None
    action_target = configured_window.target if variant is EpisodeVariant.POSITIVE else None
    truth = EpisodeTruth(
        key=EpisodeKey(
            "ofd-v1",
            request.split_namespace,
            request.suite,
            request.root_seed,
            MatchedEpisodeCoordinate("matched", request.cohort_index, member_index),
        ),
        recipe=EpisodeRecipe(
            request.requested_path_length,
            variant,
            len(template.distractor_edges),
            request.suite,
            attempt,
            oracle_timing=config.data.oracle_timing,
        ),
        relevant_node_path=relevant_nodes,
        relevant_record_ids=relevant_record_ids,
        terminal_record_id=terminal_record_id,
        relevant_hazard_type=relevant_hazard_type,
        private_terminal=private_terminal,
        activation_time=activation_time,
        episode_delay=template.episode_delay,
        action_window_start=action_start,
        action_window_end=action_end,
        action_target=action_target,
        rejection_count=attempt,
        rejection_reasons=rejection_reasons,
    )
    return EpisodeBundle(
        PublicEpisode(
            AgentInit(
                public_id,
                config.data.primary_memory_capacity,
                config.data.hazard_types,
                0.0,
            ),
            tuple(events),
        ),
        truth,
    )


def _validate_matched_member_shape(bundle: EpisodeBundle, config: Phase1Config) -> None:
    facts = bundle.public.events[:-1]
    if len(facts) > config.data.primary_memory_capacity:
        raise ValueError("primary fact capacity exceeded")
    if any(event.kind is not ExternalEventKind.FACT for event in facts):
        raise ValueError("public facts must precede activation")
    if [event.event_id for event in facts] != list(range(len(facts))):
        raise ValueError("FACT IDs must be contiguous after presentation")
    if bundle.public.events[-1].event_id != len(facts):
        raise ValueError("ACTIVATE ID must follow FACT IDs")
    if bundle.truth.private_terminal.event_id != len(facts) + 1:
        raise ValueError("private terminal ID must follow ACTIVATE")
    fact_payloads = tuple(event.payload for event in facts)
    if bundle.truth.key.suite is SuiteName.CONTRADICTION_STRESS:
        target = bundle.truth.relevant_node_path[-1]
        contradictions = tuple(
            payload
            for payload in fact_payloads
            if isinstance(payload, (HazardFact, SafeFact)) and payload.node == target
        )
        if not 2 <= len(contradictions) <= 4 or {type(payload) for payload in contradictions} != {
            HazardFact,
            SafeFact,
        }:
            raise ValueError("contradiction stress requires explicit conflicting terminal records")
    elif (
        sum(isinstance(payload, HazardFact) for payload in fact_payloads) != 2
        or sum(isinstance(payload, SafeFact) for payload in fact_payloads) != 1
    ):
        raise ValueError("primary members require two hazards and one safe fact")


def _opaque_cohort_hash(request: CohortRequest) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "generator_version": "ofd-v1",
                "split_namespace": request.split_namespace.value,
                "suite": request.suite.value,
                "root_seed": request.root_seed,
                "cohort_index": request.cohort_index,
                "requested_path_length": request.requested_path_length,
            }
        )
    ).hexdigest()


def generate_matched_cohort(
    config: Phase1Config,
    request: CohortRequest,
    public_id_seed: int,
) -> GenerationCohort:
    """Generate one fully matched four-member validation cohort."""
    if not isinstance(config, Phase1Config):
        raise TypeError("config must be a Phase1Config")
    if not isinstance(request, CohortRequest):
        raise TypeError("request must be a CohortRequest")
    rejection_counts: dict[str, int] = {}
    for attempt in range(config.data.max_generation_attempts):
        try:
            template = sample_cohort_template(config, request, attempt)
            variants = _matched_variants(request, attempt)
            rejection_reasons = tuple(rejection_counts)
            candidates = tuple(
                _build_matched_member(
                    config,
                    request,
                    template,
                    variant,
                    member_index,
                    attempt,
                    rejection_reasons,
                    "pending-public-id",
                )
                for member_index, variant in enumerate(variants)
            )
            for candidate in candidates:
                _validate_matched_member_shape(candidate, config)
        except ValueError as error:
            reason = str(error)
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue

        public_ids = allocate_public_ids(
            PublicIdBatchKey(
                generator_version="ofd-v1",
                split_namespace=request.split_namespace,
                suite=request.suite,
                public_id_seed=public_id_seed,
                cohort_index=request.cohort_index,
                accepted_attempt=attempt,
            )
        )
        episodes = tuple(
            EpisodeBundle(
                PublicEpisode(
                    AgentInit(
                        public_id,
                        candidate.public.init.memory_capacity,
                        candidate.public.init.hazard_type_count,
                        candidate.public.init.initial_time,
                    ),
                    candidate.public.events,
                ),
                candidate.truth,
            )
            for candidate, public_id in zip(candidates, public_ids, strict=True)
        )
        complete_episodes = (episodes[0], episodes[1], episodes[2], episodes[3])
        for episode in complete_episodes:
            validate_episode_invariants(episode, config)
        validate_cohort_invariants(complete_episodes, config)
        return GenerationCohort(
            request=request,
            accepted_attempt=attempt,
            episodes=complete_episodes,
            seed_tokens=_matched_seed_tokens(request, attempt),
            rejections=tuple(
                RejectionDiagnostic(reason, count) for reason, count in rejection_counts.items()
            ),
        )
    raise GenerationError(
        "matched cohort generation exhausted",
        context={
            "cohort_hash": _opaque_cohort_hash(request),
            "attempt_count": config.data.max_generation_attempts,
            "rejection_reasons": rejection_counts,
        },
    )


def regenerate_matched_episode(
    config: Phase1Config,
    request: CohortRequest,
    public_id_seed: int,
    expected_public_id: str,
    expected_accepted_attempt: int,
) -> EpisodeBundle:
    """Rebuild a matched cohort and return only the authenticated requested member."""
    _require_exact_int(expected_accepted_attempt, "expected_accepted_attempt")
    if not isinstance(expected_public_id, str):
        raise TypeError("expected_public_id must be a string")
    cohort = generate_matched_cohort(config, request, public_id_seed)
    if cohort.accepted_attempt != expected_accepted_attempt:
        raise GenerationError("matched cohort regeneration attempt mismatch")
    for bundle in cohort.episodes:
        if bundle.public.init.episode_public_id == expected_public_id:
            return bundle
    raise GenerationError("matched cohort regeneration public ID mismatch")


@dataclass(frozen=True, slots=True)
class _IndependentEpisodeTemplate:
    """One episode-local primary topology and timing draw.

    This is deliberately not a cohort template: every value comes from an
    ``IndependentCounterSeedKey`` for the request's own episode index.
    """

    relevant_nodes: tuple[int, ...]
    relevant_edges: tuple[tuple[int, int], ...]
    distractor_edges: tuple[tuple[int, int], ...]
    structural_edges: tuple[tuple[int, int], ...]
    unreachable_terminal_nodes: tuple[int, int, int]
    episode_delay: float
    fact_gap_sequence: tuple[float, ...]
    activation_gap: float
    hazard_class_multiset: tuple[int, int]
    terminal_record_count: int


def _independent_stream(
    request: IndependentEpisodeRequest, stream: SeedStream, attempt: int
) -> np.random.Generator:
    if stream is SeedStream.LABEL:
        raise ValueError("independent episodes cannot consume allocation label streams")
    return independent_local_generator(
        IndependentCounterSeedKey(
            generator_version="ofd-v1",
            split_namespace=request.split_namespace,
            suite=request.suite,
            root_seed=request.root_seed,
            episode_index=request.episode_index,
            stream=stream,
            attempt=attempt,
        )
    )


def _independent_seed_token(
    request: IndependentEpisodeRequest, stream: SeedStream, attempt: int
) -> str:
    if stream is SeedStream.LABEL:
        raise ValueError("independent episodes cannot consume allocation label streams")
    return derive_independent_counter_seed(
        IndependentCounterSeedKey(
            generator_version="ofd-v1",
            split_namespace=request.split_namespace,
            suite=request.suite,
            root_seed=request.root_seed,
            episode_index=request.episode_index,
            stream=stream,
            attempt=attempt,
        )
    ).token


def independent_seed_tokens(
    request: IndependentEpisodeRequest, accepted_attempt: int
) -> tuple[str, ...]:
    """Return every construction token for one independent accepted draw."""

    if not isinstance(request, IndependentEpisodeRequest):
        raise TypeError("request must be an IndependentEpisodeRequest")
    _require_exact_int(accepted_attempt, "accepted_attempt")
    if accepted_attempt >= 1_000:
        raise ValueError("accepted_attempt must be below 1000")
    return tuple(
        _independent_seed_token(request, stream, accepted_attempt)
        for stream in (
            SeedStream.TEMPLATE,
            SeedStream.STRUCTURE,
            SeedStream.TIMESTAMPS,
            SeedStream.NODE_PERMUTATION,
            SeedStream.TERMINALS,
            SeedStream.PRESENTATION,
        )
    )


def _sample_independent_template(
    config: Phase1Config, request: IndependentEpisodeRequest, attempt: int
) -> _IndependentEpisodeTemplate:
    """Sample the complete primary nuisance recipe for one episode only."""

    spec = suite_spec(request.suite)
    template_rng = _independent_stream(request, SeedStream.TEMPLATE, attempt)
    distractor_count = int(
        template_rng.integers(spec.distractor_link_records[0], spec.distractor_link_records[1] + 1)
    )
    episode_delay = _log_uniform(template_rng, spec.delay_log_uniform)
    hazard_classes = tuple(
        sorted(
            int(template_rng.integers(0, config.data.hazard_types))
            for _ in range(config.data.terminal_hazard_records)
        )
    )
    if len(hazard_classes) != _TERMINAL_RECORD_COUNT - 1:
        raise ValueError("Phase 1 requires exactly two hazard records")

    relevant_nodes = tuple(range(request.requested_path_length + 1))
    relevant_edges = tuple(pairwise(relevant_nodes))
    unreachable_terminal_nodes = (61, 62, 63)
    reserved_nodes = _STRESS_RESERVED_NODES if request.suite in _STRUCTURAL_STRESS_SUITES else ()
    distractor_nodes = tuple(
        node
        for node in range(request.requested_path_length + 1, config.data.max_entities - 3)
        if node not in unreachable_terminal_nodes and node not in reserved_nodes
    )
    candidates = tuple(
        (source, target)
        for source in distractor_nodes
        for target in distractor_nodes
        if source < target
    )
    if distractor_count > len(candidates):
        raise ValueError("distractor topology cannot fit available entity IDs")
    structure_rng = _independent_stream(request, SeedStream.STRUCTURE, attempt)
    selected = structure_rng.choice(len(candidates), size=distractor_count, replace=False)
    distractor_edges = tuple(sorted(candidates[int(index)] for index in selected))
    structural_edges = _sample_structural_edges(config, request, structure_rng, distractor_edges)
    terminal_record_count = _stress_terminal_record_count(config, request, structure_rng)
    _validate_template_topology(
        relevant_nodes, relevant_edges, distractor_edges, unreachable_terminal_nodes
    )

    fact_count = (
        len(relevant_edges) + len(distractor_edges) + len(structural_edges) + terminal_record_count
    )
    if request.suite is SuiteName.MEMORY_OVERFLOW_STRESS:
        if config.stress is None or not (
            config.stress.overflow_record_count[0]
            <= fact_count
            <= config.stress.overflow_record_count[1]
        ):
            raise ValueError("memory overflow facts are outside the configured range")
    elif fact_count > config.data.primary_memory_capacity:
        raise ValueError("independent facts exceed the fixed memory capacity")
    timestamps_rng = _independent_stream(request, SeedStream.TIMESTAMPS, attempt)
    fact_gaps = tuple(
        _log_uniform(timestamps_rng, _OBSERVATION_GAP_RANGE) for _ in range(fact_count)
    )
    activation_gap = _log_uniform(timestamps_rng, _OBSERVATION_GAP_RANGE)
    return _IndependentEpisodeTemplate(
        relevant_nodes,
        relevant_edges,
        distractor_edges,
        structural_edges,
        unreachable_terminal_nodes,
        episode_delay,
        fact_gaps,
        activation_gap,
        (hazard_classes[0], hazard_classes[1]),
        terminal_record_count,
    )


def _sample_structural_edges(
    config: Phase1Config,
    request: IndependentEpisodeRequest,
    structure_rng: np.random.Generator,
    distractor_edges: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Derive only the declared structural stress shape from the local structure stream."""

    if request.suite not in _STRUCTURAL_STRESS_SUITES:
        return ()
    if config.stress is None:
        raise ValueError("structural stress suites require StressDataConfig")
    if request.suite is SuiteName.BRANCHING_STRESS:
        count = int(
            structure_rng.integers(
                config.stress.branching_records[0], config.stress.branching_records[1] + 1
            )
        )
        targets = tuple(range(55, 61))[:count]
        sources = tuple(
            int(structure_rng.integers(0, request.requested_path_length)) for _ in targets
        )
        return tuple(zip(sources, targets, strict=True))
    if request.suite is SuiteName.CYCLES_STRESS:
        length = int(
            structure_rng.integers(
                config.stress.irrelevant_cycle_length[0],
                config.stress.irrelevant_cycle_length[1] + 1,
            )
        )
        nodes = tuple(range(49, 55))[:length]
        return tuple((nodes[index], nodes[(index + 1) % length]) for index in range(length))
    if request.suite is SuiteName.MEMORY_OVERFLOW_STRESS:
        total_count = int(
            structure_rng.integers(
                config.stress.overflow_record_count[0],
                config.stress.overflow_record_count[1] + 1,
            )
        )
        edge_count = total_count - request.requested_path_length - len(distractor_edges) - 3
        candidates = tuple(
            (source, target)
            for source in range(request.requested_path_length + 1, 61)
            for target in range(request.requested_path_length + 1, 61)
            if source < target and (source, target) not in distractor_edges
        )
        if not 0 <= edge_count <= len(candidates):
            raise ValueError("memory overflow topology cannot fit the configured fact count")
        selected = structure_rng.choice(len(candidates), size=edge_count, replace=False)
        return tuple(sorted(candidates[int(index)] for index in selected))
    if request.suite is SuiteName.NULL_NEAR_MISS_STRESS:
        return ()
    if request.suite in {
        SuiteName.CONTRADICTION_STRESS,
        SuiteName.MINIMUM_DURATION_STRESS,
        SuiteName.CHECKPOINT_STRESS,
    }:
        return ()
    raise ValueError("unsupported structural stress suite")


def _stress_terminal_record_count(
    config: Phase1Config,
    request: IndependentEpisodeRequest,
    structure_rng: np.random.Generator,
) -> int:
    """Draw the sole terminal-count deviation in the contradiction suite."""

    if request.suite is not SuiteName.CONTRADICTION_STRESS:
        return _TERMINAL_RECORD_COUNT
    if config.stress is None:
        raise ValueError("contradiction stress suites require StressDataConfig")
    return int(
        structure_rng.integers(
            config.stress.contradiction_records[0],
            config.stress.contradiction_records[1] + 1,
        )
    )


def _build_independent_candidate(
    config: Phase1Config,
    request: IndependentEpisodeRequest,
    attempt: int,
    rejection_reasons: tuple[str, ...],
) -> EpisodeBundle:
    """Build one unbound independent candidate from its episode-local streams."""

    template = _sample_independent_template(config, request, attempt)
    node_rng = _independent_stream(request, SeedStream.NODE_PERMUTATION, attempt)
    permutation = tuple(int(item) for item in node_rng.permutation(config.data.max_entities))
    relabel = {canonical: permutation[canonical] for canonical in range(config.data.max_entities)}
    relevant_nodes = tuple(relabel[node] for node in template.relevant_nodes)
    facts: list[tuple[LinkFact | HazardFact | SafeFact, str]] = [
        (LinkFact(relabel[source], relabel[target]), "relevant_link")
        for source, target in template.relevant_edges
    ]
    facts.extend(
        (LinkFact(relabel[source], relabel[target]), "distractor_link")
        for source, target in template.distractor_edges
    )
    facts.extend(
        (LinkFact(relabel[source], relabel[target]), "structural_link")
        for source, target in template.structural_edges
    )

    terminals_rng = _independent_stream(request, SeedStream.TERMINALS, attempt)
    hazard_types = tuple(
        int(item) for item in terminals_rng.permutation(template.hazard_class_multiset)
    )
    terminal_nodes = tuple(
        relabel[template.unreachable_terminal_nodes[int(index)]]
        for index in terminals_rng.permutation(_TERMINAL_RECORD_COUNT)
    )
    near_miss_hazard_node: int | None = None
    target = relevant_nodes[-1]
    if request.variant is EpisodeVariant.POSITIVE:
        if request.suite is SuiteName.CONTRADICTION_STRESS:
            facts.extend(
                (SafeFact(target, confidence=0.01 * (index + 1)), "contradiction_terminal")
                for index in range(template.terminal_record_count)
            )
        else:
            facts.extend(
                (
                    (
                        HazardFact(target, hazard_types[0], template.episode_delay),
                        "reachable_terminal",
                    ),
                    (
                        HazardFact(terminal_nodes[0], hazard_types[1], template.episode_delay),
                        "decoy_terminal",
                    ),
                    (SafeFact(terminal_nodes[1]), "decoy_terminal"),
                )
            )
    elif request.variant is EpisodeVariant.SAFE_NEGATIVE:
        facts.extend(
            (
                (SafeFact(target), "reachable_terminal"),
                (
                    HazardFact(terminal_nodes[0], hazard_types[0], template.episode_delay),
                    "decoy_terminal",
                ),
                (
                    HazardFact(terminal_nodes[1], hazard_types[1], template.episode_delay),
                    "decoy_terminal",
                ),
            )
        )
    elif request.variant is EpisodeVariant.DISCONNECTED_NEGATIVE:
        if request.suite is SuiteName.NULL_NEAR_MISS_STRESS:
            near_miss_hazard_node = relabel[55]
            facts.extend(
                (
                    (
                        HazardFact(near_miss_hazard_node, hazard_types[0], template.episode_delay),
                        "near_miss_terminal",
                    ),
                    (
                        HazardFact(terminal_nodes[1], hazard_types[1], template.episode_delay),
                        "decoy_terminal",
                    ),
                    (SafeFact(terminal_nodes[2]), "decoy_terminal"),
                )
            )
        else:
            facts.extend(
                (
                    (
                        HazardFact(terminal_nodes[0], hazard_types[0], template.episode_delay),
                        "decoy_terminal",
                    ),
                    (
                        HazardFact(terminal_nodes[1], hazard_types[1], template.episode_delay),
                        "decoy_terminal",
                    ),
                    (SafeFact(terminal_nodes[2]), "decoy_terminal"),
                )
            )
    else:
        raise ValueError("independent request has an unsupported primary variant")

    presentation_rng = _independent_stream(request, SeedStream.PRESENTATION, attempt)
    presentation = [facts[int(index)] for index in presentation_rng.permutation(len(facts))]
    contradiction_terminal_id: int | None = None
    if request.suite is SuiteName.CONTRADICTION_STRESS:
        contradiction_positions = tuple(
            index
            for index, (_, role) in enumerate(presentation)
            if role == "contradiction_terminal"
        )
        replacement = _contradiction_terminal_payloads(
            target,
            template.terminal_record_count,
            hazard_types[0],
            template.episode_delay,
        )
        if len(contradiction_positions) != len(replacement):
            raise ValueError("contradiction terminal presentation is incomplete")
        for position, payload in zip(contradiction_positions, replacement, strict=True):
            presentation[position] = (payload, "contradiction_terminal")
        contradiction_terminal_id = contradiction_positions[-1]
    if len(presentation) != len(template.fact_gap_sequence):
        raise ValueError("independent presentation and timestamps disagree")
    timestamp = 0.0
    events: list[ExternalEvent] = []
    relevant_link_ids: dict[int, int] = {}
    terminal_record_id: int | None = None
    for event_id, ((payload, role), gap) in enumerate(
        zip(presentation, template.fact_gap_sequence, strict=True)
    ):
        timestamp += gap
        events.append(ExternalEvent(event_id, timestamp, ExternalEventKind.FACT, payload))
        if role == "relevant_link":
            assert isinstance(payload, LinkFact)
            relevant_link_ids[payload.source_node] = event_id
        elif role == "reachable_terminal" or (
            role == "contradiction_terminal" and event_id == contradiction_terminal_id
        ):
            terminal_record_id = event_id
    activation_time = timestamp + template.activation_gap
    fact_count = len(events)
    events.append(
        ExternalEvent(
            fact_count,
            activation_time,
            ExternalEventKind.ACTIVATE,
            ActivationPayload(relevant_nodes[0]),
        )
    )
    relevant_record_ids = tuple(relevant_link_ids[node] for node in relevant_nodes[:-1])
    if request.variant is not EpisodeVariant.DISCONNECTED_NEGATIVE:
        if terminal_record_id is None:
            raise ValueError("reachable terminal record was not constructed")
        relevant_record_ids = (*relevant_record_ids, terminal_record_id)
    configured_window = action_window(
        activation_time, template.episode_delay, config.data.oracle_timing
    )
    positive = request.variant is EpisodeVariant.POSITIVE
    stress_metadata = (
        StressMetadata(over_capacity_record_count=fact_count)
        if request.suite is SuiteName.MEMORY_OVERFLOW_STRESS
        else (
            StressMetadata(
                near_miss_missing_edges=config.stress.near_miss_missing_edges,
                near_miss_hazard_node=near_miss_hazard_node,
            )
            if request.suite is SuiteName.NULL_NEAR_MISS_STRESS
            else (
                StressMetadata(minimum_feasible_delay=template.episode_delay)
                if request.suite is SuiteName.MINIMUM_DURATION_STRESS
                else (
                    StressMetadata(
                        proposed_checkpoint_pause_time=(
                            activation_time + template.episode_delay / 2.0
                        )
                    )
                    if request.suite is SuiteName.CHECKPOINT_STRESS
                    else None
                )
            )
        )
    )
    truth = EpisodeTruth(
        key=EpisodeKey(
            "ofd-v1",
            request.split_namespace,
            request.suite,
            request.root_seed,
            IndependentEpisodeCoordinate(
                "independent", request.episode_index, request.allocation_quartet_index
            ),
        ),
        recipe=EpisodeRecipe(
            request.requested_path_length,
            request.variant,
            len(template.distractor_edges) + len(template.structural_edges),
            request.suite,
            attempt,
            oracle_timing=config.data.oracle_timing,
        ),
        relevant_node_path=relevant_nodes,
        relevant_record_ids=relevant_record_ids,
        terminal_record_id=terminal_record_id,
        relevant_hazard_type=hazard_types[0] if positive else None,
        private_terminal=ExternalEvent(
            fact_count + 1,
            activation_time + template.episode_delay,
            ExternalEventKind.OUTCOME if positive else ExternalEventKind.END,
            None,
        ),
        activation_time=activation_time,
        episode_delay=template.episode_delay,
        action_window_start=configured_window.start if positive else None,
        action_window_end=configured_window.end if positive else None,
        action_target=configured_window.target if positive else None,
        rejection_count=attempt,
        rejection_reasons=rejection_reasons,
        stress_metadata=stress_metadata,
    )
    bundle = EpisodeBundle(
        PublicEpisode(
            AgentInit(
                "pending-public-id",
                config.data.primary_memory_capacity,
                config.data.hazard_types,
                0.0,
            ),
            tuple(events),
        ),
        truth,
    )
    if request.suite is SuiteName.MINIMUM_DURATION_STRESS:
        return _with_minimum_duration(bundle, config)
    if request.suite is SuiteName.CHECKPOINT_STRESS:
        return _with_checkpoint_pause(bundle)
    return bundle


def _contradiction_terminal_payloads(
    target: int,
    count: int,
    current_hazard_type: int,
    delay: float,
) -> tuple[HazardFact | SafeFact, ...]:
    """Keep stale terminal assertions explicit, with the latest assertion current."""

    if not 2 <= count <= 4:
        raise ValueError("contradiction stress requires two through four terminal records")
    stale: list[HazardFact | SafeFact] = [SafeFact(target, confidence=0.10)]
    for index in range(1, count - 1):
        stale.append(
            HazardFact(
                target,
                (current_hazard_type + index) % _HAZARD_TYPE_COUNT,
                delay,
                confidence=0.10 + 0.10 * index,
            )
        )
    return (*stale, HazardFact(target, current_hazard_type, delay, confidence=0.90))


def _stress_trace_raw_intervals(bundle: EpisodeBundle, delay: float) -> tuple[float, ...]:
    """Recompute the Task 5 urgency/difficulty intervals from a fixed jitter draw."""

    if type(delay) is not float or not math.isfinite(delay) or delay <= 0.0:
        return ()
    truth = bundle.truth
    timing = truth.recipe.oracle_timing
    coordinate = truth.key.coordinate
    if not isinstance(coordinate, IndependentEpisodeCoordinate):
        raise ValueError("stress timing requires an independent episode coordinate")
    facts_by_id = {
        event.event_id: event
        for event in bundle.public.events[:-1]
        if event.kind is ExternalEventKind.FACT
    }
    selected = tuple(facts_by_id[record_id] for record_id in truth.relevant_record_ids)
    facts = tuple(facts_by_id.values())
    jitter_rng = _independent_stream(
        IndependentEpisodeRequest(
            truth.key.split_namespace,
            truth.key.suite,
            truth.key.root_seed,
            coordinate.episode_index,
            truth.recipe.requested_path_length,
            truth.recipe.variant,
            coordinate.allocation_quartet_index,
        ),
        SeedStream.TRACE_JITTER,
        truth.recipe.accepted_attempt,
    )
    elapsed = 0.0
    raw_intervals: list[float] = []
    event_count = 2 * len(selected)
    for index in range(event_count):
        event = selected[index // 2]
        payload = event.payload
        if not isinstance(payload, (LinkFact, HazardFact, SafeFact)):
            raise ValueError("stress trace selected record is invalid")
        competitors = sum(
            1
            for candidate in facts
            if candidate is not event and _stress_competes(payload, candidate.payload)
        )
        remaining_budget = timing.terminal_compose_fraction * delay - elapsed
        remaining_count = event_count - index
        urgency = min(
            1.0,
            remaining_count * timing.delta_0 / max(remaining_budget, timing.delta_min),
        )
        raw = timing.delta_0 * (1.0 + 0.15 * competitors) / (1.0 + 0.5 * urgency)
        jittered = raw * math.exp(float(jitter_rng.normal(0.0, timing.jitter_log_std)))
        if not math.isfinite(jittered) or jittered <= 0.0:
            raise ValueError("stress trace jitter produced an invalid interval")
        raw_intervals.append(jittered)
        elapsed += min(timing.delta_max, max(timing.delta_min, jittered))
    return tuple(raw_intervals)


def _stress_competes(selected: LinkFact | HazardFact | SafeFact, candidate: object) -> bool:
    if not isinstance(candidate, type(selected)):
        return False
    if isinstance(selected, LinkFact):
        assert isinstance(candidate, LinkFact)
        return (
            selected.source_node == candidate.source_node
            or selected.target_node == candidate.target_node
        )
    if isinstance(selected, HazardFact):
        assert isinstance(candidate, HazardFact)
        return (
            selected.node == candidate.node
            or selected.hazard_type == candidate.hazard_type
            or math.floor(math.log2(selected.delay)) == math.floor(math.log2(candidate.delay))
        )
    assert isinstance(selected, SafeFact) and isinstance(candidate, SafeFact)
    return selected.node == candidate.node


def minimum_duration_is_feasible(bundle: EpisodeBundle, delay: float) -> bool:
    """Return the fixed-jitter feasibility predicate used by minimum-duration stress."""

    if not isinstance(bundle, EpisodeBundle):
        raise TypeError("bundle must be an EpisodeBundle")
    if type(delay) is not float or not math.isfinite(delay) or delay <= 0.0:
        return False
    raw = _stress_trace_raw_intervals(bundle, delay)
    if not raw:
        return False
    timing = bundle.truth.recipe.oracle_timing
    budget_required = timing.delta_min * sum(raw) / min(raw)
    return timing.terminal_compose_fraction * delay >= budget_required


def _minimum_duration_boundary(bundle: EpisodeBundle, config: Phase1Config) -> float:
    if config.stress is None:
        raise ValueError("minimum-duration stress suites require StressDataConfig")
    timing = bundle.truth.recipe.oracle_timing
    event_count = 2 * len(bundle.truth.relevant_record_ids)
    floor = event_count * timing.delta_min / timing.terminal_compose_fraction
    upper = floor
    while not minimum_duration_is_feasible(bundle, upper):
        upper *= 2.0
        if upper > config.stress.minimum_duration_search_upper:
            raise ValueError("minimum-duration feasibility exceeds configured search upper bound")
    points = config.stress.minimum_duration_monotonic_grid_points
    values = tuple(floor + (upper - floor) * index / (points - 1) for index in range(points))
    feasible = tuple(minimum_duration_is_feasible(bundle, value) for value in values)
    if any(left and not right for left, right in pairwise(feasible)):
        raise ValueError("minimum-duration feasibility is not monotonic")
    first = next(index for index, value in enumerate(feasible) if value)
    if first == 0:
        return floor
    lo, hi = values[first - 1], values[first]
    while math.nextafter(lo, math.inf) != hi:
        midpoint = lo + (hi - lo) / 2.0
        if minimum_duration_is_feasible(bundle, midpoint):
            hi = midpoint
        else:
            lo = midpoint
    return hi


def _stress_trace_deltas(bundle: EpisodeBundle, delay: float) -> tuple[float, ...]:
    raw = _stress_trace_raw_intervals(bundle, delay)
    timing = bundle.truth.recipe.oracle_timing
    deltas = [min(timing.delta_max, max(timing.delta_min, value)) for value in raw]
    compose_budget = timing.terminal_compose_fraction * delay
    elapsed = sum(deltas)
    if elapsed > compose_budget:
        scale = compose_budget / elapsed
        deltas = [value * scale for value in deltas]
    return tuple(deltas)


def _with_minimum_duration(bundle: EpisodeBundle, config: Phase1Config) -> EpisodeBundle:
    boundary = _minimum_duration_boundary(bundle, config)
    epsilon = config.stress.minimum_duration_epsilon if config.stress is not None else 0.0
    delay = math.nextafter(boundary + epsilon, math.inf)
    deltas = _stress_trace_deltas(bundle, delay)
    timing = bundle.truth.recipe.oracle_timing
    if (
        not deltas
        or any(value < timing.delta_min for value in deltas)
        or sum(deltas) > timing.terminal_compose_fraction * delay
        or timing.action_target_fraction * delay < sum(deltas)
    ):
        raise ValueError("minimum-duration trace cannot satisfy the OFD timing contract")
    events = tuple(
        replace(event, payload=replace(event.payload, delay=delay))
        if isinstance(event.payload, HazardFact)
        else event
        for event in bundle.public.events
    )
    truth = bundle.truth
    window = action_window(truth.activation_time, delay, timing)
    updated_truth = replace(
        truth,
        private_terminal=replace(truth.private_terminal, timestamp=truth.activation_time + delay),
        episode_delay=delay,
        action_window_start=window.start,
        action_window_end=window.end,
        action_target=window.target,
        stress_metadata=StressMetadata(minimum_feasible_delay=boundary),
    )
    return EpisodeBundle(replace(bundle.public, events=events), updated_truth)


def _with_checkpoint_pause(bundle: EpisodeBundle) -> EpisodeBundle:
    if bundle.truth.recipe.variant is not EpisodeVariant.POSITIVE:
        raise ValueError("checkpoint stress requires a positive hazard path")
    delay = bundle.truth.episode_delay
    deltas = _stress_trace_deltas(bundle, delay)
    timestamps: list[float] = []
    current = bundle.truth.activation_time
    for delta in deltas:
        current += delta
        timestamps.append(current)
    timestamps.append(
        bundle.truth.activation_time
        + bundle.truth.recipe.oracle_timing.action_target_fraction * delay
    )
    if len(timestamps) < 2:
        raise ValueError("checkpoint stress requires adjacent oracle events")
    coordinate = bundle.truth.key.coordinate
    assert isinstance(coordinate, IndependentEpisodeCoordinate)
    index = coordinate.episode_index % (len(timestamps) - 1)
    pause = (timestamps[index] + timestamps[index + 1]) / 2.0
    if not timestamps[index] < pause < timestamps[index + 1]:
        raise ValueError("checkpoint pause must lie strictly between oracle events")
    return EpisodeBundle(
        bundle.public,
        replace(bundle.truth, stress_metadata=StressMetadata(proposed_checkpoint_pause_time=pause)),
    )


def _validate_independent_episode_shape(bundle: EpisodeBundle, config: Phase1Config) -> None:
    """Generator-local primary shape checks before binding its public ID."""

    facts = bundle.public.events[:-1]
    if bundle.truth.key.suite is SuiteName.MEMORY_OVERFLOW_STRESS:
        if config.stress is None or not (
            config.stress.overflow_record_count[0]
            <= len(facts)
            <= config.stress.overflow_record_count[1]
        ):
            raise ValueError("memory overflow fact count is outside the configured range")
        metadata = bundle.truth.stress_metadata
        if metadata is None or metadata.over_capacity_record_count != len(facts):
            raise ValueError("memory overflow metadata does not match retained facts")
    else:
        _validate_matched_member_shape(bundle, config)
    minimum_trace_events = 2 * (
        bundle.truth.recipe.requested_path_length
        + (0 if bundle.truth.recipe.variant is EpisodeVariant.DISCONNECTED_NEGATIVE else 1)
    )
    if (
        minimum_trace_events * bundle.truth.recipe.oracle_timing.delta_min
        > bundle.truth.recipe.oracle_timing.terminal_compose_fraction * bundle.truth.episode_delay
    ):
        raise ValueError("minimum legal oracle trace cannot fit the episode delay")


def _opaque_independent_request_hash(request: IndependentEpisodeRequest) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "generator_version": "ofd-v1",
                "split_namespace": request.split_namespace.value,
                "suite": request.suite.value,
                "root_seed": request.root_seed,
                "episode_index": request.episode_index,
                "requested_path_length": request.requested_path_length,
                "variant": request.variant.value,
                "allocation_quartet_index": request.allocation_quartet_index,
            }
        )
    ).hexdigest()


def generate_independent_episode(
    config: Phase1Config,
    request: IndependentEpisodeRequest,
    public_id_seed: int,
) -> EpisodeBundle:
    """Generate one independent primary claim-data episode without cohort nuisance sharing."""

    if not isinstance(config, Phase1Config):
        raise TypeError("config must be a Phase1Config")
    if not isinstance(request, IndependentEpisodeRequest):
        raise TypeError("request must be an IndependentEpisodeRequest")
    if request.suite not in _PRIMARY_SUITES:
        raise ValueError("generate_independent_episode supports only primary suites")
    return _generate_independent_bundle(config, request, public_id_seed)


def generate_stress_episode(
    config: Phase1Config,
    request: IndependentEpisodeRequest,
    public_id_seed: int,
) -> EpisodeBundle:
    """Generate one declared structural stress episode from local RNG domains."""

    if not isinstance(config, Phase1Config):
        raise TypeError("config must be a Phase1Config")
    if not isinstance(request, IndependentEpisodeRequest):
        raise TypeError("request must be an IndependentEpisodeRequest")
    if request.suite not in _STRUCTURAL_STRESS_SUITES:
        raise ValueError("generate_stress_episode supports only structural stress suites")
    positive_suites = _STRUCTURAL_STRESS_SUITES - {SuiteName.NULL_NEAR_MISS_STRESS}
    if request.suite in positive_suites and request.variant is not EpisodeVariant.POSITIVE:
        raise ValueError("structural stress suites require a positive hazard path")
    if (
        request.suite is SuiteName.NULL_NEAR_MISS_STRESS
        and request.variant is not EpisodeVariant.DISCONNECTED_NEGATIVE
    ):
        raise ValueError("null near-miss stress suites require a disconnected negative path")
    return _generate_independent_bundle(config, request, public_id_seed)


def _generate_independent_bundle(
    config: Phase1Config,
    request: IndependentEpisodeRequest,
    public_id_seed: int,
) -> EpisodeBundle:
    """Execute the shared bounded local retry and public-ID binding path."""

    rejection_counts: dict[str, int] = {}
    for attempt in range(config.data.max_generation_attempts):
        try:
            candidate = _build_independent_candidate(
                config, request, attempt, tuple(rejection_counts)
            )
            _validate_independent_episode_shape(candidate, config)
        except ValueError as error:
            reason = str(error)
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        public_id = allocate_independent_public_id(
            IndependentPublicIdKey(
                generator_version="ofd-v1",
                split_namespace=request.split_namespace,
                suite=request.suite,
                public_id_seed=public_id_seed,
                episode_index=request.episode_index,
                accepted_attempt=attempt,
            )
        )
        bundle = EpisodeBundle(
            PublicEpisode(
                AgentInit(
                    public_id,
                    candidate.public.init.memory_capacity,
                    candidate.public.init.hazard_type_count,
                    candidate.public.init.initial_time,
                ),
                candidate.public.events,
            ),
            candidate.truth,
        )
        validate_episode_invariants(bundle, config)
        return bundle
    raise GenerationError(
        "independent episode generation exhausted",
        context={
            "request_hash": _opaque_independent_request_hash(request),
            "attempt_count": config.data.max_generation_attempts,
            "rejection_reasons": rejection_counts,
        },
    )


def regenerate_independent_episode(
    config: Phase1Config,
    request: IndependentEpisodeRequest,
    public_id_seed: int,
    expected_public_id: str,
    expected_accepted_attempt: int,
) -> EpisodeBundle:
    """Rebuild one independent episode and authenticate its stored public identity."""

    _require_exact_int(expected_accepted_attempt, "expected_accepted_attempt")
    if not isinstance(expected_public_id, str):
        raise TypeError("expected_public_id must be a string")
    bundle = generate_independent_episode(config, request, public_id_seed)
    if bundle.truth.recipe.accepted_attempt != expected_accepted_attempt:
        raise GenerationError("independent episode regeneration attempt mismatch")
    if bundle.public.init.episode_public_id != expected_public_id:
        raise GenerationError("independent episode regeneration public ID mismatch")
    return bundle


def regenerate_stress_episode(
    config: Phase1Config,
    request: IndependentEpisodeRequest,
    public_id_seed: int,
    expected_public_id: str,
    expected_accepted_attempt: int,
) -> EpisodeBundle:
    """Rebuild and authenticate one structural stress episode."""

    _require_exact_int(expected_accepted_attempt, "expected_accepted_attempt")
    if not isinstance(expected_public_id, str):
        raise TypeError("expected_public_id must be a string")
    bundle = generate_stress_episode(config, request, public_id_seed)
    if bundle.truth.recipe.accepted_attempt != expected_accepted_attempt:
        raise GenerationError("stress episode regeneration attempt mismatch")
    if bundle.public.init.episode_public_id != expected_public_id:
        raise GenerationError("stress episode regeneration public ID mismatch")
    return bundle


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
