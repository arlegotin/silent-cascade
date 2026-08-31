"""Independent, fail-closed validation for generated primary OFD episodes.

This module intentionally derives its graph and timing conclusions from event
facts.  It does not import the generator or the oracle: agreement between
those three implementations is a meaningful correctness control.
"""

import hashlib
import math
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from itertools import pairwise, permutations

import numpy as np

from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    EpisodeBundle,
    EpisodeVariant,
    IndependentEpisodeCoordinate,
    MatchedEpisodeCoordinate,
)
from silent_cascade.env.timing import action_window
from silent_cascade.errors import EpisodeInvariantError
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.rng import (
    CounterSeedKey,
    IndependentCounterSeedKey,
    SeedStream,
    independent_local_generator,
    local_generator,
)
from silent_cascade.schemas import (
    ActivationPayload,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    LinkFact,
    SafeFact,
)

_TIME_TOLERANCE = 1.0e-9
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
_TERMINAL_RECORD_COUNT = 3

_INDEPENDENT_ALLOCATION_BLOCKS = (
    (SuiteName.IID_PRIMARY, 2, 0, 8_000, 0),
    (SuiteName.IID_PRIMARY, 3, 8_000, 8_000, 2_000),
    (SuiteName.IID_PRIMARY, 4, 16_000, 8_000, 4_000),
    (SuiteName.OOD_DEPTH, 5, 0, 4_000, 6_000),
    (SuiteName.OOD_DEPTH, 6, 4_000, 4_000, 7_000),
    (SuiteName.OOD_DEPTH, 7, 8_000, 4_000, 8_000),
    (SuiteName.OOD_DEPTH, 8, 12_000, 4_000, 9_000),
    (SuiteName.OOD_SHORT_DELAY, 2, 0, 8_000, 10_000),
    (SuiteName.OOD_SHORT_DELAY, 3, 8_000, 8_000, 12_000),
    (SuiteName.OOD_SHORT_DELAY, 4, 16_000, 8_000, 14_000),
    (SuiteName.OOD_LONG_DELAY, 2, 0, 4_000, 16_000),
    (SuiteName.OOD_LONG_DELAY, 3, 4_000, 4_000, 17_000),
    (SuiteName.OOD_LONG_DELAY, 4, 8_000, 4_000, 18_000),
    (SuiteName.DISTRACTOR_FLOOD, 2, 0, 8_000, 19_000),
    (SuiteName.DISTRACTOR_FLOOD, 3, 8_000, 8_000, 21_000),
    (SuiteName.DISTRACTOR_FLOOD, 4, 16_000, 8_000, 23_000),
)


@dataclass(frozen=True, slots=True)
class InvariantReport:
    """Counts and independently established primary episode properties."""

    valid: bool
    fact_count: int
    link_count: int
    hazard_count: int
    safe_count: int
    reachable_node_count: int
    reachable_terminal_count: int
    requested_path_length: int
    check_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Analysis:
    report: InvariantReport
    path: tuple[int, ...]
    link_record_ids: tuple[int, ...]
    terminal: ExternalEvent | None
    fact_gaps: tuple[float, ...]
    activation_gap: float
    hazard_signature: tuple[tuple[int, float], ...]
    link_signature: tuple[tuple[int, int], tuple[int, ...], tuple[tuple[int, int], ...]]


class _InvariantViolation(EpisodeInvariantError):
    """Private check marker for stable non-strict reports without error detail."""

    def __init__(self, message: str, check_id: str = "invalid") -> None:
        super().__init__(message)
        self.check_id = check_id


def _fail(message: str, *, check_id: str = "invalid") -> None:
    raise _InvariantViolation(message, check_id)


def _is_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=_TIME_TOLERANCE)


def _require_exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{name} must be an in-range exact integer")
    return value


def _require_finite_float(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or (positive and value <= 0.0):
        _fail(f"{name} must be finite" if not positive else f"{name} must be positive and finite")
    return value


def _validate_public_id(value: object) -> None:
    if not isinstance(value, str):
        _fail("public episode ID must be a canonical UUID v4")
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        _fail("public episode ID must be a canonical UUID v4")
    if parsed.version != 4 or str(parsed) != value:
        _fail("public episode ID must be a canonical UUID v4")


def _fact_identity(payload: object) -> tuple[object, ...]:
    if isinstance(payload, LinkFact):
        return ("link", payload.source_node, payload.target_node, payload.confidence)
    if isinstance(payload, HazardFact):
        return ("hazard", payload.node, payload.hazard_type, payload.delay, payload.confidence)
    if isinstance(payload, SafeFact):
        return ("safe", payload.node, payload.confidence)
    _fail("FACT payload must be a primary fact")


def _suite_parameters(
    config: Phase1Config, suite: SuiteName
) -> tuple[tuple[int, ...], tuple[float, float], tuple[int, int]]:
    data = config.data
    if suite in {SuiteName.VALIDATION, SuiteName.IID_PRIMARY}:
        return (
            data.iid_test_path_lengths,
            data.train_delay_log_uniform,
            data.train_distractor_link_records,
        )
    if suite is SuiteName.OOD_DEPTH:
        return (
            data.ood_depth_path_lengths,
            data.ood_depth_delay_log_uniform,
            data.train_distractor_link_records,
        )
    if suite is SuiteName.OOD_SHORT_DELAY:
        return (
            data.ood_short_delay_path_lengths,
            data.ood_short_delay_log_uniform,
            data.train_distractor_link_records,
        )
    if suite is SuiteName.OOD_LONG_DELAY:
        return (
            data.iid_test_path_lengths,
            data.ood_long_delay_log_uniform,
            data.train_distractor_link_records,
        )
    if suite is SuiteName.DISTRACTOR_FLOOD:
        return (
            data.iid_test_path_lengths,
            data.train_delay_log_uniform,
            data.ood_distractor_link_records,
        )
    if suite in _STRUCTURAL_STRESS_SUITES:
        return (
            data.iid_test_path_lengths,
            data.train_delay_log_uniform,
            data.train_distractor_link_records,
        )
    _fail("primary suite is not supported by invariant validation")


def _allocation_variants(
    split_namespace: SplitNamespace,
    suite: SuiteName,
    root_seed: int,
    requested_path_length: int,
    allocation_quartet_index: int,
) -> tuple[EpisodeVariant, EpisodeVariant, EpisodeVariant, EpisodeVariant]:
    """Independently reproduce the fixed four-label allocation contract."""

    variants = tuple(
        sorted(
            (
                EpisodeVariant.POSITIVE,
                EpisodeVariant.POSITIVE,
                EpisodeVariant.SAFE_NEGATIVE,
                EpisodeVariant.DISCONNECTED_NEGATIVE,
            ),
            key=lambda variant: variant.value,
        )
    )
    permutations_without_duplicates: list[
        tuple[EpisodeVariant, EpisodeVariant, EpisodeVariant, EpisodeVariant]
    ] = []
    for candidate in permutations(variants):
        typed_candidate = (candidate[0], candidate[1], candidate[2], candidate[3])
        if typed_candidate not in permutations_without_duplicates:
            permutations_without_duplicates.append(typed_candidate)
    payload = canonical_json_bytes(
        {
            "domain": "silent-cascade/ofd-v1/allocation-label/v1",
            "generator_version": "ofd-v1",
            "split_namespace": split_namespace.value,
            "suite": suite.value,
            "root_seed": root_seed,
            "requested_path_length": requested_path_length,
            "allocation_quartet_index": allocation_quartet_index,
        }
    )
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    selected_index = int(generator.integers(len(permutations_without_duplicates)))
    return permutations_without_duplicates[selected_index]


def _validate_independent_allocation_provenance(bundle: EpisodeBundle) -> None:
    """Authenticate fixed gate/frozen allocation position and its private label."""

    truth = bundle.truth
    key = truth.key
    recipe = truth.recipe
    coordinate = key.coordinate
    if not isinstance(coordinate, IndependentEpisodeCoordinate):
        _fail("invalid RNG provenance", check_id="rng_provenance")
    if key.split_namespace not in {SplitNamespace.PHASE1_GATE, SplitNamespace.FROZEN}:
        return
    matching = tuple(
        block
        for block in _INDEPENDENT_ALLOCATION_BLOCKS
        if block[0] is key.suite
        and block[1] == recipe.requested_path_length
        and block[2] <= coordinate.episode_index < block[2] + block[3]
    )
    if len(matching) != 1:
        _fail("invalid RNG provenance", check_id="rng_provenance")
    _, _, first_episode_index, _, first_quartet_index = matching[0]
    within_block = coordinate.episode_index - first_episode_index
    expected_quartet = first_quartet_index + within_block // 4
    if coordinate.allocation_quartet_index != expected_quartet:
        _fail("invalid RNG provenance", check_id="rng_provenance")
    expected_variants = _allocation_variants(
        key.split_namespace,
        key.suite,
        key.root_seed,
        recipe.requested_path_length,
        expected_quartet,
    )
    if recipe.variant is not expected_variants[within_block % 4]:
        _fail("invalid RNG provenance", check_id="rng_provenance")


def _member_generator(bundle: EpisodeBundle, stream: SeedStream):
    coordinate = bundle.truth.key.coordinate
    if not isinstance(coordinate, MatchedEpisodeCoordinate) or coordinate.member_index not in range(
        4
    ):
        _fail("matched member coordinate is invalid")
    return local_generator(
        CounterSeedKey(
            generator_version="ofd-v1",
            split_namespace=bundle.truth.key.split_namespace,
            suite=bundle.truth.key.suite,
            root_seed=bundle.truth.key.root_seed,
            cohort_index=coordinate.cohort_index,
            member_index=coordinate.member_index,
            stream=stream,
            attempt=bundle.truth.recipe.accepted_attempt,
        )
    )


def _cohort_generator(bundle: EpisodeBundle, stream: SeedStream):
    return local_generator(
        CounterSeedKey(
            generator_version="ofd-v1",
            split_namespace=bundle.truth.key.split_namespace,
            suite=bundle.truth.key.suite,
            root_seed=bundle.truth.key.root_seed,
            cohort_index=bundle.truth.key.coordinate.cohort_index,
            member_index=-1,
            stream=stream,
            attempt=bundle.truth.recipe.accepted_attempt,
        )
    )


def _log_uniform(generator: object, bounds: tuple[float, float]) -> float:
    return float(math.exp(generator.uniform(math.log(bounds[0]), math.log(bounds[1]))))  # type: ignore[union-attr]


def _expected_member_payloads(
    bundle: EpisodeBundle, config: Phase1Config
) -> tuple[tuple[object, ...], tuple[float, ...], float, int]:
    """Reconstruct one member using only the stable RNG contract and public types."""

    truth = bundle.truth
    key = truth.key
    recipe = truth.recipe
    coordinate = key.coordinate
    if (
        recipe.evaluation_suite is not key.suite
        or key.generator_version != config.data.generator_version
        or not isinstance(key.split_namespace, SplitNamespace)
        or not isinstance(key.suite, SuiteName)
        or not isinstance(coordinate, MatchedEpisodeCoordinate)
    ):
        _fail("invalid RNG provenance", check_id="rng_provenance")
    if (
        type(key.root_seed) is not int
        or not 0 <= key.root_seed < 2**128
        or type(coordinate.cohort_index) is not int
        or coordinate.cohort_index < 0
        or type(coordinate.member_index) is not int
        or coordinate.member_index not in range(4)
        or type(recipe.accepted_attempt) is not int
        or not 0 <= recipe.accepted_attempt < 1_000
        or type(truth.rejection_count) is not int
        or not 0 <= truth.rejection_count < 1_000
        or truth.rejection_count != recipe.accepted_attempt
        or type(coordinate.mode) is not str
        or coordinate.mode != "matched"
    ):
        _fail("invalid RNG provenance", check_id="rng_provenance")
    path_lengths, delay_bounds, distractor_bounds = _suite_parameters(config, key.suite)
    if recipe.requested_path_length not in path_lengths:
        _fail("requested path length is invalid for the suite")
    template_rng = _cohort_generator(bundle, SeedStream.TEMPLATE)
    distractor_count = int(template_rng.integers(distractor_bounds[0], distractor_bounds[1] + 1))
    episode_delay = _log_uniform(template_rng, delay_bounds)
    hazard_classes = tuple(
        sorted(int(template_rng.integers(0, config.data.hazard_types)) for _ in range(2))
    )
    relevant_nodes = tuple(range(recipe.requested_path_length + 1))
    candidates = tuple(
        (source, target)
        for source in range(recipe.requested_path_length + 1, config.data.max_entities - 3)
        for target in range(recipe.requested_path_length + 1, config.data.max_entities - 3)
        if source < target
    )
    structure_rng = _cohort_generator(bundle, SeedStream.STRUCTURE)
    selected = structure_rng.choice(len(candidates), size=distractor_count, replace=False)
    distractor_edges = tuple(sorted(candidates[int(index)] for index in selected))
    timestamps_rng = _cohort_generator(bundle, SeedStream.TIMESTAMPS)
    fact_gaps = tuple(
        _log_uniform(timestamps_rng, config.data.observation_gap_log_uniform)
        for _ in range(recipe.requested_path_length + distractor_count + 3)
    )
    activation_gap = _log_uniform(timestamps_rng, config.data.observation_gap_log_uniform)
    permutation = tuple(
        int(value)
        for value in _member_generator(bundle, SeedStream.NODE_PERMUTATION).permutation(64)
    )
    relabel = {canonical: permutation[canonical] for canonical in range(64)}
    if truth.relevant_node_path != tuple(relabel[node] for node in relevant_nodes):
        _fail(
            "member node permutation provenance disagrees",
            check_id="node_permutation_provenance",
        )
    facts: list[object] = [
        LinkFact(relabel[source], relabel[target]) for source, target in pairwise(relevant_nodes)
    ]
    facts.extend(LinkFact(relabel[source], relabel[target]) for source, target in distractor_edges)
    terminals_rng = _member_generator(bundle, SeedStream.TERMINALS)
    hazard_types = tuple(int(value) for value in terminals_rng.permutation(hazard_classes))
    terminal_nodes = tuple(
        relabel[(61, 62, 63)[int(index)]] for index in terminals_rng.permutation(3)
    )
    target = relabel[relevant_nodes[-1]]
    if recipe.variant is EpisodeVariant.POSITIVE:
        facts.extend(
            (
                HazardFact(target, hazard_types[0], episode_delay),
                HazardFact(terminal_nodes[0], hazard_types[1], episode_delay),
                SafeFact(terminal_nodes[1]),
            )
        )
    elif recipe.variant is EpisodeVariant.SAFE_NEGATIVE:
        facts.extend(
            (
                SafeFact(target),
                HazardFact(terminal_nodes[0], hazard_types[0], episode_delay),
                HazardFact(terminal_nodes[1], hazard_types[1], episode_delay),
            )
        )
    elif recipe.variant is EpisodeVariant.DISCONNECTED_NEGATIVE:
        facts.extend(
            (
                HazardFact(terminal_nodes[0], hazard_types[0], episode_delay),
                HazardFact(terminal_nodes[1], hazard_types[1], episode_delay),
                SafeFact(terminal_nodes[2]),
            )
        )
    else:
        _fail("episode variant is invalid")
    presentation = _member_generator(bundle, SeedStream.PRESENTATION).permutation(len(facts))
    return (
        tuple(facts[int(index)] for index in presentation),
        fact_gaps,
        activation_gap,
        distractor_count,
    )


def _independent_generator(bundle: EpisodeBundle, stream: SeedStream):
    coordinate = bundle.truth.key.coordinate
    if not isinstance(coordinate, IndependentEpisodeCoordinate):
        _fail("independent coordinate is invalid", check_id="rng_provenance")
    return independent_local_generator(
        IndependentCounterSeedKey(
            generator_version="ofd-v1",
            split_namespace=bundle.truth.key.split_namespace,
            suite=bundle.truth.key.suite,
            root_seed=bundle.truth.key.root_seed,
            episode_index=coordinate.episode_index,
            stream=stream,
            attempt=bundle.truth.recipe.accepted_attempt,
        )
    )


def _expected_structural_edges(
    bundle: EpisodeBundle,
    config: Phase1Config,
    structure_rng: object,
    distractor_edges: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Reconstruct the declared stress shape without calling the generator."""

    suite = bundle.truth.key.suite
    if suite not in _STRUCTURAL_STRESS_SUITES:
        return ()
    if config.stress is None:
        _fail("structural stress suite requires stress configuration")
    length = bundle.truth.recipe.requested_path_length
    if suite is SuiteName.BRANCHING_STRESS:
        count = int(
            structure_rng.integers(  # type: ignore[union-attr]
                config.stress.branching_records[0], config.stress.branching_records[1] + 1
            )
        )
        targets = tuple(range(55, 61))[:count]
        sources = tuple(
            int(structure_rng.integers(0, length))  # type: ignore[union-attr]
            for _ in targets
        )
        return tuple(zip(sources, targets, strict=True))
    if suite is SuiteName.MEMORY_OVERFLOW_STRESS:
        total_count = int(
            structure_rng.integers(  # type: ignore[union-attr]
                config.stress.overflow_record_count[0], config.stress.overflow_record_count[1] + 1
            )
        )
        edge_count = total_count - length - len(distractor_edges) - 3
        candidates = tuple(
            (source, target)
            for source in range(length + 1, 61)
            for target in range(length + 1, 61)
            if source < target and (source, target) not in distractor_edges
        )
        if not 0 <= edge_count <= len(candidates):
            _fail("memory overflow topology cannot fit the configured fact count")
        selected = structure_rng.choice(  # type: ignore[union-attr]
            len(candidates), size=edge_count, replace=False
        )
        return tuple(sorted(candidates[int(index)] for index in selected))
    if suite in {
        SuiteName.NULL_NEAR_MISS_STRESS,
        SuiteName.CONTRADICTION_STRESS,
        SuiteName.MINIMUM_DURATION_STRESS,
        SuiteName.CHECKPOINT_STRESS,
    }:
        return ()
    cycle_length = int(
        structure_rng.integers(  # type: ignore[union-attr]
            config.stress.irrelevant_cycle_length[0],
            config.stress.irrelevant_cycle_length[1] + 1,
        )
    )
    nodes = tuple(range(49, 55))[:cycle_length]
    return tuple((nodes[index], nodes[(index + 1) % cycle_length]) for index in range(cycle_length))


def _expected_independent_payloads(
    bundle: EpisodeBundle, config: Phase1Config
) -> tuple[tuple[object, ...], tuple[float, ...], float, int]:
    """Reconstruct independent construction from only its episode-local key domains."""

    truth = bundle.truth
    key = truth.key
    recipe = truth.recipe
    coordinate = key.coordinate
    if (
        recipe.evaluation_suite is not key.suite
        or key.generator_version != config.data.generator_version
        or not isinstance(key.split_namespace, SplitNamespace)
        or not isinstance(key.suite, SuiteName)
        or not isinstance(coordinate, IndependentEpisodeCoordinate)
    ):
        _fail("invalid RNG provenance", check_id="rng_provenance")
    if (
        type(key.root_seed) is not int
        or not 0 <= key.root_seed < 2**128
        or type(coordinate.episode_index) is not int
        or coordinate.episode_index < 0
        or type(coordinate.allocation_quartet_index) is not int
        or coordinate.allocation_quartet_index < 0
        or type(recipe.accepted_attempt) is not int
        or not 0 <= recipe.accepted_attempt < 1_000
        or type(truth.rejection_count) is not int
        or not 0 <= truth.rejection_count < 1_000
        or truth.rejection_count != recipe.accepted_attempt
        or type(coordinate.mode) is not str
        or coordinate.mode != "independent"
    ):
        _fail("invalid RNG provenance", check_id="rng_provenance")
    _validate_independent_allocation_provenance(bundle)
    path_lengths, delay_bounds, distractor_bounds = _suite_parameters(config, key.suite)
    if recipe.requested_path_length not in path_lengths:
        _fail("requested path length is invalid for the suite")
    template_rng = _independent_generator(bundle, SeedStream.TEMPLATE)
    distractor_count = int(template_rng.integers(distractor_bounds[0], distractor_bounds[1] + 1))
    episode_delay = _log_uniform(template_rng, delay_bounds)
    hazard_classes = tuple(
        sorted(int(template_rng.integers(0, config.data.hazard_types)) for _ in range(2))
    )
    if key.suite is SuiteName.MINIMUM_DURATION_STRESS:
        # The sampled template delay is intentionally replaced by the separately
        # searched boundary delay; validate that search below from public facts.
        episode_delay = truth.episode_delay
    relevant_nodes = tuple(range(recipe.requested_path_length + 1))
    reserved_nodes = _STRESS_RESERVED_NODES if key.suite in _STRUCTURAL_STRESS_SUITES else ()
    candidates = tuple(
        (source, target)
        for source in range(recipe.requested_path_length + 1, config.data.max_entities - 3)
        for target in range(recipe.requested_path_length + 1, config.data.max_entities - 3)
        if source < target and source not in reserved_nodes and target not in reserved_nodes
    )
    structure_rng = _independent_generator(bundle, SeedStream.STRUCTURE)
    selected = structure_rng.choice(len(candidates), size=distractor_count, replace=False)
    distractor_edges = tuple(sorted(candidates[int(index)] for index in selected))
    structural_edges = _expected_structural_edges(bundle, config, structure_rng, distractor_edges)
    terminal_record_count = _TERMINAL_RECORD_COUNT
    if key.suite is SuiteName.CONTRADICTION_STRESS:
        if config.stress is None:
            _fail("contradiction stress suite requires stress configuration")
        terminal_record_count = int(
            structure_rng.integers(
                config.stress.contradiction_records[0],
                config.stress.contradiction_records[1] + 1,
            )
        )
    timestamps_rng = _independent_generator(bundle, SeedStream.TIMESTAMPS)
    fact_gaps = tuple(
        _log_uniform(timestamps_rng, config.data.observation_gap_log_uniform)
        for _ in range(
            recipe.requested_path_length
            + distractor_count
            + len(structural_edges)
            + terminal_record_count
        )
    )
    activation_gap = _log_uniform(timestamps_rng, config.data.observation_gap_log_uniform)
    permutation = tuple(
        int(value)
        for value in _independent_generator(bundle, SeedStream.NODE_PERMUTATION).permutation(64)
    )
    relabel = {canonical: permutation[canonical] for canonical in range(64)}
    if truth.relevant_node_path != tuple(relabel[node] for node in relevant_nodes):
        _fail("independent node permutation provenance disagrees")
    facts: list[object] = [
        LinkFact(relabel[source], relabel[target]) for source, target in pairwise(relevant_nodes)
    ]
    facts.extend(LinkFact(relabel[source], relabel[target]) for source, target in distractor_edges)
    facts.extend(LinkFact(relabel[source], relabel[target]) for source, target in structural_edges)
    terminals_rng = _independent_generator(bundle, SeedStream.TERMINALS)
    hazard_types = tuple(int(value) for value in terminals_rng.permutation(hazard_classes))
    terminal_nodes = tuple(
        relabel[(61, 62, 63)[int(index)]] for index in terminals_rng.permutation(3)
    )
    target = relabel[relevant_nodes[-1]]
    if recipe.variant is EpisodeVariant.POSITIVE:
        if key.suite is SuiteName.CONTRADICTION_STRESS:
            facts.extend(
                SafeFact(target, confidence=0.01 * (index + 1))
                for index in range(terminal_record_count)
            )
        else:
            facts.extend(
                (
                    HazardFact(target, hazard_types[0], episode_delay),
                    HazardFact(terminal_nodes[0], hazard_types[1], episode_delay),
                    SafeFact(terminal_nodes[1]),
                )
            )
    elif recipe.variant is EpisodeVariant.SAFE_NEGATIVE:
        facts.extend(
            (
                SafeFact(target),
                HazardFact(terminal_nodes[0], hazard_types[0], episode_delay),
                HazardFact(terminal_nodes[1], hazard_types[1], episode_delay),
            )
        )
    elif recipe.variant is EpisodeVariant.DISCONNECTED_NEGATIVE:
        if key.suite is SuiteName.NULL_NEAR_MISS_STRESS:
            facts.extend(
                (
                    HazardFact(relabel[55], hazard_types[0], episode_delay),
                    HazardFact(terminal_nodes[1], hazard_types[1], episode_delay),
                    SafeFact(terminal_nodes[2]),
                )
            )
        else:
            facts.extend(
                (
                    HazardFact(terminal_nodes[0], hazard_types[0], episode_delay),
                    HazardFact(terminal_nodes[1], hazard_types[1], episode_delay),
                    SafeFact(terminal_nodes[2]),
                )
            )
    else:
        _fail("episode variant is invalid")
    presentation = _independent_generator(bundle, SeedStream.PRESENTATION).permutation(len(facts))
    expected_payloads = [facts[int(index)] for index in presentation]
    if key.suite is SuiteName.CONTRADICTION_STRESS:
        positions = tuple(
            index
            for index, payload in enumerate(expected_payloads)
            if isinstance(payload, (HazardFact, SafeFact)) and payload.node == target
        )
        replacement = _contradiction_terminal_payloads(
            target, terminal_record_count, hazard_types[0], episode_delay
        )
        if len(positions) != len(replacement):
            _fail("contradiction terminal presentation provenance is invalid")
        for index, payload in zip(positions, replacement, strict=True):
            expected_payloads[index] = payload
    return (
        tuple(expected_payloads),
        fact_gaps,
        activation_gap,
        distractor_count + len(structural_edges),
    )


def _contradiction_terminal_payloads(
    target: int,
    count: int,
    current_hazard_type: int,
    delay: float,
) -> tuple[HazardFact | SafeFact, ...]:
    """Independently reconstruct explicit stale/current terminal assertions."""

    if not 2 <= count <= 4:
        _fail("contradiction terminal count is outside the configured range")
    stale: list[HazardFact | SafeFact] = [SafeFact(target, confidence=0.10)]
    for index in range(1, count - 1):
        stale.append(
            HazardFact(
                target,
                (current_hazard_type + index) % 4,
                delay,
                confidence=0.10 + 0.10 * index,
            )
        )
    return (*stale, HazardFact(target, current_hazard_type, delay, confidence=0.90))


def _require_acyclic_links(links: tuple[ExternalEvent, ...]) -> None:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for event in links:
        assert isinstance(event.payload, LinkFact)
        adjacency[event.payload.source_node].add(event.payload.target_node)
    visiting: set[int] = set()
    complete: set[int] = set()

    def visit(node: int) -> None:
        if node in visiting:
            _fail("LINK topology contains a cycle")
        if node in complete:
            return
        visiting.add(node)
        for target in adjacency[node]:
            visit(target)
        visiting.remove(node)
        complete.add(node)

    for node in tuple(adjacency):
        visit(node)


def _directed_cycles(links: tuple[ExternalEvent, ...]) -> tuple[tuple[int, ...], ...]:
    """Return canonical directed cycles so the cycle exception stays narrowly scoped."""

    adjacency: dict[int, list[int]] = defaultdict(list)
    for event in links:
        assert isinstance(event.payload, LinkFact)
        adjacency[event.payload.source_node].append(event.payload.target_node)
    cycles: set[tuple[int, ...]] = set()

    def visit(node: int, path: tuple[int, ...]) -> None:
        for target in adjacency[node]:
            if target in path:
                cycle = path[path.index(target) :]
                rotations = tuple(cycle[index:] + cycle[:index] for index in range(len(cycle)))
                cycles.add(min(rotations))
            else:
                visit(target, (*path, target))

    for node in tuple(adjacency):
        visit(node, (node,))
    return tuple(sorted(cycles))


def _activation_reachable_nodes(
    start: int, links_by_source: dict[int, list[ExternalEvent]]
) -> set[int]:
    reachable = {start}
    pending = [start]
    while pending:
        node = pending.pop()
        for event in links_by_source.get(node, []):
            assert isinstance(event.payload, LinkFact)
            target = event.payload.target_node
            if target not in reachable:
                reachable.add(target)
                pending.append(target)
    return reachable


def _unique_branching_path(
    start: int,
    links_by_source: dict[int, list[ExternalEvent]],
    terminals_by_node: dict[int, list[ExternalEvent]],
) -> tuple[tuple[int, ...], tuple[ExternalEvent, ...], ExternalEvent]:
    """Derive the sole terminal path while allowing terminal-free branches."""

    solutions: list[tuple[tuple[int, ...], tuple[ExternalEvent, ...], ExternalEvent]] = []

    def visit(node: int, nodes: tuple[int, ...], links: tuple[ExternalEvent, ...]) -> None:
        if node in nodes[:-1]:
            _fail("activation-reachable graph contains a cycle")
        terminals = terminals_by_node.get(node, [])
        outgoing = links_by_source.get(node, [])
        if len(terminals) > 1:
            _fail("activation-reachable graph has multiple terminals")
        if terminals:
            if outgoing:
                _fail("reachable terminal has an outgoing continuation")
            solutions.append((nodes, links, terminals[0]))
            return
        for event in outgoing:
            assert isinstance(event.payload, LinkFact)
            visit(event.payload.target_node, (*nodes, event.payload.target_node), (*links, event))

    visit(start, (start,), ())
    if len(solutions) != 1:
        _fail("branching stress episode must have exactly one reachable terminal solution")
    return solutions[0]


def _link_signature(
    links: tuple[ExternalEvent, ...],
) -> tuple[tuple[int, int], tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Return a label-free component/degree signature for matched comparison."""

    degrees: dict[int, list[int]] = {}
    adjacency: dict[int, set[int]] = defaultdict(set)
    for event in links:
        payload = event.payload
        if not isinstance(payload, LinkFact):
            _fail("link signature requires LINK facts")
        degrees.setdefault(payload.source_node, [0, 0])[1] += 1
        degrees.setdefault(payload.target_node, [0, 0])[0] += 1
        adjacency[payload.source_node].add(payload.target_node)
        adjacency[payload.target_node].add(payload.source_node)
    component_sizes: list[int] = []
    unseen = set(adjacency)
    while unseen:
        pending = [unseen.pop()]
        size = 0
        while pending:
            node = pending.pop()
            size += 1
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    pending.append(neighbor)
        component_sizes.append(size)
    return (
        (len(links), len(degrees)),
        tuple(sorted(component_sizes)),
        tuple(sorted((in_degree, out_degree) for in_degree, out_degree in degrees.values())),
    )


def _stress_raw_intervals(bundle: EpisodeBundle, delay: float) -> tuple[float, ...]:
    """Independently reproduce fixed-jitter Task 5 scheduling inputs."""

    truth = bundle.truth
    timing = truth.recipe.oracle_timing
    facts_by_id = {event.event_id: event for event in bundle.public.events[:-1]}
    selected = tuple(facts_by_id[record_id] for record_id in truth.relevant_record_ids)
    if not isinstance(truth.key.coordinate, IndependentEpisodeCoordinate):
        _fail("stress timing requires an independent coordinate")
    rng = _independent_generator(bundle, SeedStream.TRACE_JITTER)
    elapsed = 0.0
    raw: list[float] = []
    event_count = 2 * len(selected)
    for index in range(event_count):
        event = selected[index // 2]
        payload = event.payload
        if not isinstance(payload, (LinkFact, HazardFact, SafeFact)):
            _fail("stress trace contains an invalid selected record")
        competitors = sum(
            1
            for candidate in facts_by_id.values()
            if candidate is not event and _stress_competes(payload, candidate.payload)
        )
        remaining = timing.terminal_compose_fraction * delay - elapsed
        urgency = min(
            1.0,
            (event_count - index) * timing.delta_0 / max(remaining, timing.delta_min),
        )
        interval = (
            timing.delta_0
            * (1.0 + 0.15 * competitors)
            / (1.0 + 0.5 * urgency)
            * math.exp(float(rng.normal(0.0, timing.jitter_log_std)))
        )
        if not math.isfinite(interval) or interval <= 0.0:
            _fail("stress trace jitter produced an invalid interval")
        raw.append(interval)
        elapsed += min(timing.delta_max, max(timing.delta_min, interval))
    return tuple(raw)


def _stress_competes(selected: object, candidate: object) -> bool:
    if type(selected) is not type(candidate):
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


def _minimum_duration_feasible(bundle: EpisodeBundle, delay: float) -> bool:
    raw = _stress_raw_intervals(bundle, delay)
    timing = bundle.truth.recipe.oracle_timing
    return timing.terminal_compose_fraction * delay >= timing.delta_min * sum(raw) / min(raw)


def _minimum_duration_boundary(bundle: EpisodeBundle, config: Phase1Config) -> float:
    if config.stress is None:
        _fail("minimum-duration stress suite requires stress configuration")
    timing = bundle.truth.recipe.oracle_timing
    floor = 2 * len(bundle.truth.relevant_record_ids) * timing.delta_min
    floor /= timing.terminal_compose_fraction
    upper = floor
    while not _minimum_duration_feasible(bundle, upper):
        upper *= 2.0
        if upper > config.stress.minimum_duration_search_upper:
            _fail("minimum-duration search exceeded its configured upper bound")
    points = config.stress.minimum_duration_monotonic_grid_points
    grid = tuple(floor + (upper - floor) * index / (points - 1) for index in range(points))
    feasibility = tuple(_minimum_duration_feasible(bundle, value) for value in grid)
    if any(left and not right for left, right in pairwise(feasibility)):
        _fail("minimum-duration feasibility is not monotonic")
    first = next(index for index, value in enumerate(feasibility) if value)
    if first == 0:
        return floor
    lo, hi = grid[first - 1], grid[first]
    while math.nextafter(lo, math.inf) != hi:
        midpoint = lo + (hi - lo) / 2.0
        if _minimum_duration_feasible(bundle, midpoint):
            hi = midpoint
        else:
            lo = midpoint
    return hi


def _stress_trace_timestamps(bundle: EpisodeBundle) -> tuple[float, ...]:
    timing = bundle.truth.recipe.oracle_timing
    raw = _stress_raw_intervals(bundle, bundle.truth.episode_delay)
    deltas = [min(timing.delta_max, max(timing.delta_min, value)) for value in raw]
    budget = timing.terminal_compose_fraction * bundle.truth.episode_delay
    if sum(deltas) > budget:
        scale = budget / sum(deltas)
        deltas = [value * scale for value in deltas]
    current = bundle.truth.activation_time
    timestamps: list[float] = []
    for delta in deltas:
        current += delta
        timestamps.append(current)
    timestamps.append(
        bundle.truth.activation_time + timing.action_target_fraction * bundle.truth.episode_delay
    )
    return tuple(timestamps)


def _invalid_report(bundle: EpisodeBundle, check_id: str = "invalid") -> InvariantReport:
    facts = tuple(
        event
        for event in bundle.public.events
        if isinstance(event, ExternalEvent) and event.kind is ExternalEventKind.FACT
    )
    payloads = tuple(event.payload for event in facts)
    truth = bundle.truth
    return InvariantReport(
        valid=False,
        fact_count=len(facts),
        link_count=sum(isinstance(payload, LinkFact) for payload in payloads),
        hazard_count=sum(isinstance(payload, HazardFact) for payload in payloads),
        safe_count=sum(isinstance(payload, SafeFact) for payload in payloads),
        reachable_node_count=0,
        reachable_terminal_count=0,
        requested_path_length=getattr(truth.recipe, "requested_path_length", 0),
        check_ids=(check_id,),
    )


def _analyze_episode(bundle: EpisodeBundle, config: Phase1Config) -> _Analysis:
    if not isinstance(bundle, EpisodeBundle):
        raise TypeError("bundle must be an EpisodeBundle")
    if not isinstance(config, Phase1Config):
        raise TypeError("config must be a Phase1Config")
    public = bundle.public
    truth = bundle.truth
    if not isinstance(public.events, tuple) or not public.events:
        _fail("public events must be a nonempty tuple")
    _validate_public_id(public.init.episode_public_id)
    if public.init.memory_capacity != config.data.primary_memory_capacity:
        _fail("public memory capacity must equal the configured primary capacity")
    if public.init.hazard_type_count != config.data.hazard_types:
        _fail("public hazard count must equal the configured hazard count")
    initial_time = _require_finite_float(public.init.initial_time, "initial time")

    activation = public.events[-1]
    if (
        not isinstance(activation, ExternalEvent)
        or activation.kind is not ExternalEventKind.ACTIVATE
        or not isinstance(activation.payload, ActivationPayload)
    ):
        _fail("public episode must finish with activation")
    facts = public.events[:-1]
    capacity = min(config.data.primary_memory_capacity, 64)
    if truth.key.suite is SuiteName.MEMORY_OVERFLOW_STRESS:
        metadata = truth.stress_metadata
        if (
            config.stress is None
            or not config.stress.overflow_record_count[0]
            <= len(facts)
            <= config.stress.overflow_record_count[1]
            or len(facts) <= capacity
            or metadata is None
            or metadata.over_capacity_record_count != len(facts)
            or any(
                value is not None
                for value in (
                    metadata.near_miss_missing_edges,
                    metadata.near_miss_hazard_node,
                    metadata.proposed_checkpoint_pause_time,
                    metadata.minimum_feasible_delay,
                )
            )
        ):
            _fail("memory overflow stress metadata or retained fact count is invalid")
    elif len(facts) > capacity:
        _fail("primary fact capacity exceeded")
    previous_time = initial_time
    identities: set[tuple[object, ...]] = set()
    links: list[ExternalEvent] = []
    terminals: list[ExternalEvent] = []
    for expected_id, event in enumerate(facts):
        if not isinstance(event, ExternalEvent) or event.kind is not ExternalEventKind.FACT:
            _fail("all pre-activation events must be FACT events")
        if event.event_id != expected_id:
            _fail("FACT IDs must be contiguous", check_id="record_identity")
        timestamp = _require_finite_float(event.timestamp, "FACT timestamp")
        if timestamp <= previous_time:
            _fail("FACT timestamps must be strictly increasing")
        previous_time = timestamp
        identity = _fact_identity(event.payload)
        if truth.key.suite is SuiteName.CONTRADICTION_STRESS and isinstance(
            event.payload, (HazardFact, SafeFact)
        ):
            identity = (*identity, event.payload.confidence)
        if identity in identities:
            _fail("duplicate equivalent FACT record")
        identities.add(identity)
        if isinstance(event.payload, LinkFact):
            links.append(event)
        else:
            terminals.append(event)
    if activation.event_id != len(facts):
        _fail("activation ID must follow FACT IDs", check_id="record_identity")
    activation_time = _require_finite_float(activation.timestamp, "activation timestamp")
    if activation_time <= previous_time:
        _fail("activation timestamp must follow FACT timestamps")
    if not _is_close(activation_time, truth.activation_time):
        _fail("truth activation time must match public activation")

    observation_lower, observation_upper = config.data.observation_gap_log_uniform
    fact_gaps = tuple(
        event.timestamp - (initial_time if index == 0 else facts[index - 1].timestamp)
        for index, event in enumerate(facts)
    )
    activation_gap = activation_time - previous_time
    if any(
        not observation_lower <= gap <= observation_upper for gap in (*fact_gaps, activation_gap)
    ):
        _fail(
            "FACT and activation gaps must stay in the configured observation interval",
            check_id="observation_gap",
        )
    link_endpoints: set[tuple[int, int]] = set()
    for event in links:
        assert isinstance(event.payload, LinkFact)
        endpoints = (event.payload.source_node, event.payload.target_node)
        if endpoints in link_endpoints:
            _fail("LINK endpoints must be unique regardless of confidence")
        link_endpoints.add(endpoints)
    if truth.key.suite is SuiteName.CYCLES_STRESS:
        cycles = _directed_cycles(tuple(links))
        if (
            config.stress is None
            or len(cycles) != 1
            or not (
                config.stress.irrelevant_cycle_length[0]
                <= len(cycles[0])
                <= config.stress.irrelevant_cycle_length[1]
            )
        ):
            _fail("cycles stress episode must contain one bounded directed cycle")
    else:
        _require_acyclic_links(tuple(links))
    for event in terminals:
        if isinstance(event.payload, HazardFact) and not _is_close(
            event.payload.delay, truth.episode_delay
        ):
            _fail("every hazard terminal delay must match private episode delay")

    coordinate = truth.key.coordinate
    if isinstance(coordinate, MatchedEpisodeCoordinate):
        expected = _expected_member_payloads(bundle, config)
    elif isinstance(coordinate, IndependentEpisodeCoordinate):
        expected = _expected_independent_payloads(bundle, config)
    else:
        _fail("invalid RNG provenance", check_id="rng_provenance")
    (
        expected_payloads,
        expected_fact_gaps,
        expected_activation_gap,
        expected_distractor_count,
    ) = expected
    observed_distractor_count = len(links) - truth.recipe.requested_path_length
    if (
        truth.recipe.distractor_link_count != observed_distractor_count
        or truth.recipe.distractor_link_count != expected_distractor_count
    ):
        _fail(
            "private distractor count disagrees with independently parsed facts",
            check_id="recipe_distractor_count",
        )
    if tuple(event.payload for event in facts) != expected_payloads:
        _fail("member presentation provenance disagrees", check_id="presentation_provenance")
    if (
        len(fact_gaps) != len(expected_fact_gaps)
        or any(
            not _is_close(actual, expected)
            for actual, expected in zip(fact_gaps, expected_fact_gaps, strict=True)
        )
        or not _is_close(activation_gap, expected_activation_gap)
    ):
        _fail("member timing provenance disagrees", check_id="timing_provenance")

    if truth.key.suite in (_PRIMARY_SUITES | _STRUCTURAL_STRESS_SUITES) - {
        SuiteName.CONTRADICTION_STRESS
    }:
        if len(facts) != len(links) + 3:
            _fail("primary episodes require exactly three terminal facts")
        if sum(isinstance(event.payload, HazardFact) for event in terminals) != 2:
            _fail("primary episodes require exactly two hazard facts")
        if sum(isinstance(event.payload, SafeFact) for event in terminals) != 1:
            _fail("primary episodes require exactly one safe fact")
    elif truth.key.suite is SuiteName.CONTRADICTION_STRESS:
        if config.stress is None:
            _fail("contradiction stress suite requires stress configuration")
        target = truth.relevant_node_path[-1]
        contradictions = tuple(event for event in terminals if event.payload.node == target)
        if (
            not config.stress.contradiction_records[0]
            <= len(contradictions)
            <= config.stress.contradiction_records[1]
            or {type(event.payload) for event in contradictions} != {HazardFact, SafeFact}
            or len({(event.timestamp, event.payload.confidence) for event in contradictions})
            != len(contradictions)
        ):
            _fail("contradiction stress terminals are not explicit and distinct")

    links_by_source: dict[int, list[ExternalEvent]] = defaultdict(list)
    terminals_by_node: dict[int, list[ExternalEvent]] = defaultdict(list)
    for event in links:
        assert isinstance(event.payload, LinkFact)
        links_by_source[event.payload.source_node].append(event)
    for event in terminals:
        payload = event.payload
        assert isinstance(payload, (HazardFact, SafeFact))
        terminals_by_node[payload.node].append(event)

    current = activation.payload.start_node
    if truth.key.suite is SuiteName.BRANCHING_STRESS:
        branch_path, branch_links, branch_terminal = _unique_branching_path(
            current, links_by_source, terminals_by_node
        )
        path = list(branch_path)
        selected_links = list(branch_links)
        terminal: ExternalEvent | None = branch_terminal
        seen = set(path)
    else:
        path = [current]
        selected_links = []
        terminal = None
        seen = set()
        while True:
            if current in seen:
                _fail("activation-reachable graph contains a cycle")
            seen.add(current)
            current_terminals = terminals_by_node.get(current, [])
            if len(current_terminals) > 1 and truth.key.suite is not SuiteName.CONTRADICTION_STRESS:
                _fail("activation-reachable graph has multiple terminals")
            outgoing = links_by_source.get(current, [])
            if current_terminals:
                if outgoing:
                    _fail("reachable terminal has an outgoing continuation")
                terminal = max(
                    current_terminals,
                    key=lambda event: (
                        event.timestamp,
                        event.payload.confidence,
                        event.event_id,
                    ),
                )
                break
            if len(outgoing) > 1:
                _fail("activation-reachable graph branches")
            if not outgoing:
                break
            selected = outgoing[0]
            assert isinstance(selected.payload, LinkFact)
            selected_links.append(selected)
            current = selected.payload.target_node
            path.append(current)

    selected_ids = {event.event_id for event in selected_links}
    if truth.key.suite is SuiteName.BRANCHING_STRESS:
        if config.stress is None:
            _fail("branching stress suite requires stress configuration")
        branch_links = tuple(
            event
            for event in links
            if event.event_id not in selected_ids and event.payload.source_node in seen
        )
        branch_targets = {event.payload.target_node for event in branch_links}
        reachable = _activation_reachable_nodes(activation.payload.start_node, links_by_source)
        if (
            not (
                config.stress.branching_records[0]
                <= len(branch_links)
                <= config.stress.branching_records[1]
            )
            or reachable != seen | branch_targets
        ):
            _fail("branching stress episode has an invalid reachable branch shape")
        for event in branch_links:
            payload = event.payload
            assert isinstance(payload, LinkFact)
            if (
                payload.target_node in seen
                or payload.target_node in terminals_by_node
                or links_by_source.get(payload.target_node)
            ):
                _fail("branching stress branches must be terminal-free dead ends")
    else:
        for event in links:
            if event.event_id in selected_ids:
                continue
            payload = event.payload
            assert isinstance(payload, LinkFact)
            if payload.source_node in seen or payload.target_node in seen:
                _fail("distractor link may not touch the reachable component")
        if truth.key.suite is SuiteName.CYCLES_STRESS:
            cycles = _directed_cycles(tuple(links))
            if any(node in seen for node in cycles[0]):
                _fail("cycles stress cycle must be outside activation reachability")

    if truth.key.suite is SuiteName.NULL_NEAR_MISS_STRESS:
        metadata = truth.stress_metadata
        expected_near_miss_node = tuple(
            int(value)
            for value in _independent_generator(bundle, SeedStream.NODE_PERMUTATION).permutation(64)
        )[55]
        if (
            config.stress is None
            or metadata is None
            or metadata.near_miss_missing_edges != config.stress.near_miss_missing_edges
            or metadata.near_miss_hazard_node != expected_near_miss_node
            or expected_near_miss_node in seen
            or (path[-1], expected_near_miss_node) in link_endpoints
            or not any(
                isinstance(event.payload, HazardFact)
                and event.payload.node == expected_near_miss_node
                for event in terminals
            )
        ):
            _fail("null near-miss stress provenance is invalid")

    requested_length = _require_exact_int(
        truth.recipe.requested_path_length, "requested path length", minimum=1
    )
    if len(selected_links) != requested_length:
        _fail("reachable path length must equal the requested LINK-edge count")
    expected_terminal_kind = {
        EpisodeVariant.POSITIVE: HazardFact,
        EpisodeVariant.SAFE_NEGATIVE: SafeFact,
        EpisodeVariant.DISCONNECTED_NEGATIVE: type(None),
    }.get(truth.recipe.variant)
    if expected_terminal_kind is None:
        _fail("episode variant must be a primary variant")
    if expected_terminal_kind is type(None):
        if terminal is not None:
            _fail("disconnected primary episode must have no reachable terminal")
    elif terminal is None or not isinstance(terminal.payload, expected_terminal_kind):
        _fail("reachable terminal does not match the private episode variant")

    expected_record_ids = tuple(event.event_id for event in selected_links) + (
        () if terminal is None else (terminal.event_id,)
    )
    if truth.relevant_node_path != tuple(path):
        _fail("private truth path disagrees with independently parsed facts")
    trace_ids_are_unique = len(set(truth.relevant_record_ids)) == len(truth.relevant_record_ids)
    if truth.relevant_record_ids != expected_record_ids or not trace_ids_are_unique:
        _fail("private truth record trace is not the unique public trace")
    expected_terminal_id = None if terminal is None else terminal.event_id
    if truth.terminal_record_id != expected_terminal_id:
        _fail("private terminal record disagrees with public facts")

    delay = _require_finite_float(truth.episode_delay, "episode delay", positive=True)
    if truth.private_terminal.event_id != len(facts) + 1:
        _fail("private terminal ID must follow activation")
    if not _is_close(truth.private_terminal.timestamp, activation_time + delay):
        _fail("private terminal timestamp disagrees with public activation and delay")
    private_kind = (
        ExternalEventKind.OUTCOME
        if truth.recipe.variant is EpisodeVariant.POSITIVE
        else ExternalEventKind.END
    )
    if (
        truth.private_terminal.kind is not private_kind
        or truth.private_terminal.payload is not None
    ):
        _fail("private terminal kind does not match the episode variant")
    if truth.recipe.accepted_attempt != truth.rejection_count:
        _fail("accepted attempt must equal the persisted rejection count")

    if terminal is not None and isinstance(terminal.payload, HazardFact):
        if truth.relevant_hazard_type != terminal.payload.hazard_type:
            _fail("private hazard type disagrees with reachable hazard")
        if not _is_close(terminal.payload.delay, delay):
            _fail("reachable hazard delay disagrees with private truth")
        expected_window = action_window(activation_time, delay, truth.recipe.oracle_timing)
        if not (
            _is_close(truth.action_window_start, expected_window.start)
            and _is_close(truth.action_target, expected_window.target)
            and _is_close(truth.action_window_end, expected_window.end)
        ):
            _fail("private positive action window disagrees with OFD timing")
    elif any(
        value is not None
        for value in (
            truth.relevant_hazard_type,
            truth.action_window_start,
            truth.action_target,
            truth.action_window_end,
        )
    ):
        _fail("negative private truth must not contain action or hazard fields")

    if truth.key.suite is SuiteName.MINIMUM_DURATION_STRESS:
        metadata = truth.stress_metadata
        if config.stress is None or metadata is None or metadata.minimum_feasible_delay is None:
            _fail("minimum-duration stress metadata is missing")
        boundary = _minimum_duration_boundary(bundle, config)
        expected_delay = math.nextafter(
            boundary + config.stress.minimum_duration_epsilon,
            math.inf,
        )
        if (
            metadata.minimum_feasible_delay != boundary
            or delay != expected_delay
            or not _minimum_duration_feasible(bundle, delay)
            or _minimum_duration_feasible(bundle, math.nextafter(boundary, -math.inf))
        ):
            _fail("minimum-duration stress timing provenance is invalid")
    if truth.key.suite is SuiteName.CHECKPOINT_STRESS:
        metadata = truth.stress_metadata
        if metadata is None or metadata.proposed_checkpoint_pause_time is None:
            _fail("checkpoint stress metadata is missing")
        timestamps = _stress_trace_timestamps(bundle)
        coordinate = truth.key.coordinate
        assert isinstance(coordinate, IndependentEpisodeCoordinate)
        index = coordinate.episode_index % (len(timestamps) - 1)
        expected_pause = (timestamps[index] + timestamps[index + 1]) / 2.0
        if metadata.proposed_checkpoint_pause_time != expected_pause:
            _fail("checkpoint pause provenance is invalid")

    hazard_signature = tuple(
        sorted(
            (event.payload.hazard_type, event.payload.delay)
            for event in terminals
            if isinstance(event.payload, HazardFact)
        )
    )
    report = InvariantReport(
        valid=True,
        fact_count=len(facts),
        link_count=len(links),
        hazard_count=sum(isinstance(event.payload, HazardFact) for event in terminals),
        safe_count=sum(isinstance(event.payload, SafeFact) for event in terminals),
        reachable_node_count=len(path),
        reachable_terminal_count=int(terminal is not None),
        requested_path_length=requested_length,
        check_ids=(
            "public_shape",
            "record_identity",
            "primary_counts",
            "reachable_graph",
            "distractor_isolation",
            "private_truth",
            "timing",
        ),
    )
    return _Analysis(
        report=report,
        path=tuple(path),
        link_record_ids=tuple(event.event_id for event in selected_links),
        terminal=terminal,
        fact_gaps=fact_gaps,
        activation_gap=activation_gap,
        hazard_signature=hazard_signature,
        link_signature=_link_signature(tuple(links)),
    )


def validate_episode_invariants(
    bundle: EpisodeBundle, config: Phase1Config, strict: bool = True
) -> InvariantReport:
    """Independently validate one episode, raising on the first strict violation."""

    if type(strict) is not bool:
        raise TypeError("strict must be a bool")
    try:
        return _analyze_episode(bundle, config).report
    except EpisodeInvariantError as error:
        if strict:
            raise
        if not isinstance(bundle, EpisodeBundle):
            raise
        return _invalid_report(bundle, getattr(error, "check_id", "invalid"))


def validate_cohort_invariants(
    episodes: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
    config: Phase1Config,
    strict: bool = True,
) -> tuple[InvariantReport, InvariantReport, InvariantReport, InvariantReport]:
    """Validate each member and the exact primary matched-cohort controls."""

    if type(strict) is not bool:
        raise TypeError("strict must be a bool")
    if (
        not isinstance(episodes, tuple)
        or len(episodes) != 4
        or not all(isinstance(bundle, EpisodeBundle) for bundle in episodes)
    ):
        raise TypeError("episodes must be a four-member EpisodeBundle tuple")
    try:
        analyses = tuple(_analyze_episode(bundle, config) for bundle in episodes)
        variants = Counter(bundle.truth.recipe.variant for bundle in episodes)
        if variants != Counter(
            {
                EpisodeVariant.POSITIVE: 2,
                EpisodeVariant.SAFE_NEGATIVE: 1,
                EpisodeVariant.DISCONNECTED_NEGATIVE: 1,
            }
        ):
            _fail("matched cohort must contain the exact primary variant multiset")
        public_ids = tuple(bundle.public.init.episode_public_id for bundle in episodes)
        if len(set(public_ids)) != len(public_ids):
            _fail("matched cohort public IDs must be unique")
        coordinates = tuple(bundle.truth.key.coordinate for bundle in episodes)
        if not all(isinstance(coordinate, MatchedEpisodeCoordinate) for coordinate in coordinates):
            _fail("matched cohort members require matched coordinates")
        matched_coordinates = tuple(
            coordinate
            for coordinate in coordinates
            if isinstance(coordinate, MatchedEpisodeCoordinate)
        )
        if {coordinate.member_index for coordinate in matched_coordinates} != {0, 1, 2, 3}:
            _fail("matched cohort member indexes must be a complete permutation")
        key_signature = {
            (
                bundle.truth.key.generator_version,
                bundle.truth.key.split_namespace,
                bundle.truth.key.suite,
                bundle.truth.key.root_seed,
                bundle.truth.key.coordinate.cohort_index,
                bundle.truth.recipe.accepted_attempt,
            )
            for bundle in episodes
        }
        if len(key_signature) != 1:
            _fail("matched cohort keys and accepted attempt must agree")
        reports = tuple(analysis.report for analysis in analyses)
        for values, name in (
            ({report.requested_path_length for report in reports}, "requested path length"),
            ({report.fact_count for report in reports}, "fact count"),
            ({report.link_count for report in reports}, "LINK count"),
            ({report.hazard_count for report in reports}, "hazard count"),
            ({report.safe_count for report in reports}, "safe count"),
            (
                {bundle.truth.recipe.distractor_link_count for bundle in episodes},
                "distractor count",
            ),
            ({bundle.truth.episode_delay for bundle in episodes}, "episode delay"),
            ({analysis.fact_gaps for analysis in analyses}, "FACT gap template"),
            ({analysis.activation_gap for analysis in analyses}, "activation gap"),
            ({analysis.hazard_signature for analysis in analyses}, "hazard signature"),
            ({analysis.link_signature for analysis in analyses}, "label-free LINK signature"),
        ):
            if len(values) != 1:
                _fail(f"matched cohort must share the {name}")
        return (reports[0], reports[1], reports[2], reports[3])
    except EpisodeInvariantError as error:
        if strict:
            raise
        reports = tuple(
            validate_episode_invariants(bundle, config, strict=False) for bundle in episodes
        )
        failure_check_id = getattr(error, "check_id", "invalid")
        invalid_reports = tuple(
            replace(
                report,
                valid=False,
                check_ids=(*report.check_ids, failure_check_id, "cohort_matching"),
            )
            for report in reports
        )
        return (
            invalid_reports[0],
            invalid_reports[1],
            invalid_reports[2],
            invalid_reports[3],
        )
