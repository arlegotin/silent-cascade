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
        _fail("member node permutation provenance disagrees")
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
    relevant_nodes = tuple(range(recipe.requested_path_length + 1))
    candidates = tuple(
        (source, target)
        for source in range(recipe.requested_path_length + 1, config.data.max_entities - 3)
        for target in range(recipe.requested_path_length + 1, config.data.max_entities - 3)
        if source < target
    )
    structure_rng = _independent_generator(bundle, SeedStream.STRUCTURE)
    selected = structure_rng.choice(len(candidates), size=distractor_count, replace=False)
    distractor_edges = tuple(sorted(candidates[int(index)] for index in selected))
    timestamps_rng = _independent_generator(bundle, SeedStream.TIMESTAMPS)
    fact_gaps = tuple(
        _log_uniform(timestamps_rng, config.data.observation_gap_log_uniform)
        for _ in range(recipe.requested_path_length + distractor_count + 3)
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
    terminals_rng = _independent_generator(bundle, SeedStream.TERMINALS)
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
    presentation = _independent_generator(bundle, SeedStream.PRESENTATION).permutation(len(facts))
    return (
        tuple(facts[int(index)] for index in presentation),
        fact_gaps,
        activation_gap,
        distractor_count,
    )


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
    if len(facts) > min(config.data.primary_memory_capacity, 64):
        _fail("primary fact capacity exceeded")
    previous_time = initial_time
    identities: set[tuple[object, ...]] = set()
    links: list[ExternalEvent] = []
    terminals: list[ExternalEvent] = []
    for expected_id, event in enumerate(facts):
        if not isinstance(event, ExternalEvent) or event.kind is not ExternalEventKind.FACT:
            _fail("all pre-activation events must be FACT events")
        if event.event_id != expected_id:
            _fail("FACT IDs must be contiguous")
        timestamp = _require_finite_float(event.timestamp, "FACT timestamp")
        if timestamp <= previous_time:
            _fail("FACT timestamps must be strictly increasing")
        previous_time = timestamp
        identity = _fact_identity(event.payload)
        if identity in identities:
            _fail("duplicate equivalent FACT record")
        identities.add(identity)
        if isinstance(event.payload, LinkFact):
            links.append(event)
        else:
            terminals.append(event)
    if activation.event_id != len(facts):
        _fail("activation ID must follow FACT IDs")
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
        _fail("FACT and activation gaps must stay in the configured observation interval")
    link_endpoints: set[tuple[int, int]] = set()
    for event in links:
        assert isinstance(event.payload, LinkFact)
        endpoints = (event.payload.source_node, event.payload.target_node)
        if endpoints in link_endpoints:
            _fail("LINK endpoints must be unique regardless of confidence")
        link_endpoints.add(endpoints)
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
        _fail("member presentation provenance disagrees")
    if (
        len(fact_gaps) != len(expected_fact_gaps)
        or any(
            not _is_close(actual, expected)
            for actual, expected in zip(fact_gaps, expected_fact_gaps, strict=True)
        )
        or not _is_close(activation_gap, expected_activation_gap)
    ):
        _fail("member timing provenance disagrees")

    if truth.key.suite in _PRIMARY_SUITES:
        if len(facts) != len(links) + 3:
            _fail("primary episodes require exactly three terminal facts")
        if sum(isinstance(event.payload, HazardFact) for event in terminals) != 2:
            _fail("primary episodes require exactly two hazard facts")
        if sum(isinstance(event.payload, SafeFact) for event in terminals) != 1:
            _fail("primary episodes require exactly one safe fact")

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
    path = [current]
    selected_links: list[ExternalEvent] = []
    terminal: ExternalEvent | None = None
    seen: set[int] = set()
    while True:
        if current in seen:
            _fail("activation-reachable graph contains a cycle")
        seen.add(current)
        current_terminals = terminals_by_node.get(current, [])
        if len(current_terminals) > 1:
            _fail("activation-reachable graph has multiple terminals")
        outgoing = links_by_source.get(current, [])
        if current_terminals:
            if outgoing:
                _fail("reachable terminal has an outgoing continuation")
            terminal = current_terminals[0]
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
    for event in links:
        if event.event_id in selected_ids:
            continue
        payload = event.payload
        assert isinstance(payload, LinkFact)
        if payload.source_node in seen or payload.target_node in seen:
            _fail("distractor link may not touch the reachable component")

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
