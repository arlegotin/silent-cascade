"""Independent mutation checks for primary OFD episode bundles."""

from copy import copy
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    EpisodeBundle,
    EpisodeVariant,
    IndependentEpisodeCoordinate,
    MatchedEpisodeCoordinate,
    PublicEpisode,
)
from silent_cascade.env.generator import (
    CohortRequest,
    generate_independent_episode,
    generate_matched_cohort,
    iter_phase1_gate_requests,
)
from silent_cascade.errors import EpisodeInvariantError
from silent_cascade.rng import CounterSeedKey, SeedStream, local_generator
from silent_cascade.schemas import (
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
        [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
    ).config


@pytest.fixture
def cohort(
    config: Phase1Config,
) -> tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle]:
    request = CohortRequest(
        split_namespace=SplitNamespace.DEBUG,
        suite=SuiteName.IID_PRIMARY,
        root_seed=41,
        cohort_index=17,
        requested_path_length=3,
    )
    return generate_matched_cohort(config, request, public_id_seed=91).episodes


@pytest.fixture
def independent_bundle(config: Phase1Config) -> EpisodeBundle:
    request = next(iter_phase1_gate_requests(41))
    return generate_independent_episode(config, request, public_id_seed=91)


def _bundle_with(
    bundle: EpisodeBundle,
    *,
    events: tuple[ExternalEvent, ...] | None = None,
    **truth_changes: object,
) -> EpisodeBundle:
    """Build a type-correct bundle and then emulate artifact-field corruption."""

    public = bundle.public if events is None else PublicEpisode(bundle.public.init, events)
    truth = replace(bundle.truth)
    for field, value in truth_changes.items():
        object.__setattr__(truth, field, value)
    corrupted = object.__new__(EpisodeBundle)
    object.__setattr__(corrupted, "public", public)
    object.__setattr__(corrupted, "truth", truth)
    return corrupted


def _fact_events(bundle: EpisodeBundle) -> list[ExternalEvent]:
    return list(bundle.public.events[:-1])


def _replace_fact(bundle: EpisodeBundle, index: int, payload: object) -> EpisodeBundle:
    events = _fact_events(bundle)
    old = events[index]
    events[index] = ExternalEvent(old.event_id, old.timestamp, ExternalEventKind.FACT, payload)
    return _bundle_with(bundle, events=(*events, bundle.public.events[-1]))


def _disconnected_link_indexes(bundle: EpisodeBundle) -> tuple[int, ...]:
    relevant_ids = set(bundle.truth.relevant_record_ids)
    return tuple(
        index
        for index, event in enumerate(_fact_events(bundle))
        if isinstance(event.payload, LinkFact) and event.event_id not in relevant_ids
    )


def _unused_link_pair(bundle: EpisodeBundle) -> tuple[int, int]:
    endpoints = {
        (event.payload.source_node, event.payload.target_node)
        for event in _fact_events(bundle)
        if isinstance(event.payload, LinkFact)
    }
    reachable = set(bundle.truth.relevant_node_path)
    nodes = tuple(node for node in range(64) if node not in reachable)
    for source in nodes:
        for target in nodes:
            if source != target and (source, target) not in endpoints:
                return source, target
    raise AssertionError("fixture has no unused disconnected LINK endpoints")


def _identity_node_permutation_bundle(bundle: EpisodeBundle) -> EpisodeBundle:
    coordinate = bundle.truth.key.coordinate
    assert hasattr(coordinate, "cohort_index") and hasattr(coordinate, "member_index")
    permutation = tuple(
        int(value)
        for value in local_generator(
            CounterSeedKey(
                generator_version="ofd-v1",
                split_namespace=bundle.truth.key.split_namespace,
                suite=bundle.truth.key.suite,
                root_seed=bundle.truth.key.root_seed,
                cohort_index=coordinate.cohort_index,
                member_index=coordinate.member_index,
                stream=SeedStream.NODE_PERMUTATION,
                attempt=bundle.truth.recipe.accepted_attempt,
            )
        ).permutation(64)
    )
    inverse = {value: index for index, value in enumerate(permutation)}

    def remap(payload: object) -> object:
        if isinstance(payload, LinkFact):
            return LinkFact(inverse[payload.source_node], inverse[payload.target_node])
        if isinstance(payload, HazardFact):
            return HazardFact(inverse[payload.node], payload.hazard_type, payload.delay)
        assert isinstance(payload, SafeFact)
        return SafeFact(inverse[payload.node])

    events = _fact_events(bundle)
    for index, event in enumerate(events):
        events[index] = ExternalEvent(
            event.event_id, event.timestamp, event.kind, remap(event.payload)
        )
    path = tuple(inverse[node] for node in bundle.truth.relevant_node_path)
    return _bundle_with(
        bundle,
        events=(*events, bundle.public.events[-1]),
        relevant_node_path=path,
    )


def _construction_order_bundle(bundle: EpisodeBundle) -> EpisodeBundle:
    facts = _fact_events(bundle)
    ordered_payloads = tuple(sorted((event.payload for event in facts), key=repr))
    assert ordered_payloads != tuple(event.payload for event in facts)
    rebuilt = tuple(
        ExternalEvent(event_id, facts[event_id].timestamp, ExternalEventKind.FACT, payload)
        for event_id, payload in enumerate(ordered_payloads)
    )
    by_payload = {event.payload: event.event_id for event in rebuilt}
    path = bundle.truth.relevant_node_path
    link_ids = tuple(by_payload[LinkFact(source, target)] for source, target in pairwise(path))
    terminal_id = next(
        event.event_id
        for event in rebuilt
        if isinstance(event.payload, (HazardFact, SafeFact)) and event.payload.node == path[-1]
    )
    return _bundle_with(
        bundle,
        events=(*rebuilt, bundle.public.events[-1]),
        relevant_record_ids=(*link_ids, terminal_id),
        terminal_record_id=terminal_id,
    )


def _with_public_id(bundle: EpisodeBundle, public_id: str) -> EpisodeBundle:
    init = copy(bundle.public.init)
    object.__setattr__(init, "episode_public_id", public_id)
    public = object.__new__(PublicEpisode)
    object.__setattr__(public, "init", init)
    object.__setattr__(public, "events", bundle.public.events)
    corrupted = object.__new__(EpisodeBundle)
    object.__setattr__(corrupted, "public", public)
    object.__setattr__(corrupted, "truth", bundle.truth)
    return corrupted


def _fact_overflow_bundle(bundle: EpisodeBundle) -> EpisodeBundle:
    facts = _fact_events(bundle)
    timestamp = facts[-1].timestamp
    for event_id in range(len(facts), 65):
        timestamp += 0.1
        facts.append(
            ExternalEvent(
                event_id,
                timestamp,
                ExternalEventKind.FACT,
                LinkFact((event_id * 3) % 64, (event_id * 3 + 1) % 64),
            )
        )
    timestamp += 0.1
    activation = ExternalEvent(
        len(facts), timestamp, ExternalEventKind.ACTIVATE, bundle.public.events[-1].payload
    )
    return _bundle_with(bundle, events=(*facts, activation))


def test_valid_matched_members_have_hand_derived_reports(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """Removing an independent primary check must make this report incomplete."""
    from silent_cascade.env.invariants import (
        validate_cohort_invariants,
        validate_episode_invariants,
    )

    reports = validate_cohort_invariants(cohort, config)

    assert reports == tuple(validate_episode_invariants(bundle, config) for bundle in cohort)
    assert all(report.valid for report in reports)
    assert {report.fact_count - report.link_count for report in reports} == {3}
    assert all(report.link_count >= report.requested_path_length for report in reports)
    assert {report.hazard_count for report in reports} == {2}
    assert {report.safe_count for report in reports} == {1}
    assert {report.reachable_node_count for report in reports} == {4}
    assert {report.reachable_terminal_count for report in reports} == {0, 1}
    assert {report.requested_path_length for report in reports} == {3}
    assert all(report.check_ids for report in reports)


def test_independent_allocation_quartet_and_variant_provenance_fail_closed(
    config: Phase1Config, independent_bundle: EpisodeBundle
) -> None:
    """Changing private label provenance must reject even when episode-local facts are unchanged."""
    from silent_cascade.env.invariants import validate_episode_invariants

    coordinate = independent_bundle.truth.key.coordinate
    assert isinstance(coordinate, IndependentEpisodeCoordinate)
    wrong_variant = next(
        variant
        for variant in EpisodeVariant
        if variant is not independent_bundle.truth.recipe.variant
    )
    corruptions = (
        _bundle_with(
            independent_bundle,
            key=replace(
                independent_bundle.truth.key,
                coordinate=IndependentEpisodeCoordinate(
                    "independent", coordinate.episode_index, coordinate.allocation_quartet_index + 1
                ),
            ),
        ),
        _bundle_with(
            independent_bundle,
            key=replace(
                independent_bundle.truth.key,
                coordinate=IndependentEpisodeCoordinate(
                    "independent", coordinate.episode_index, 999_999
                ),
            ),
        ),
        _bundle_with(
            independent_bundle,
            recipe=replace(independent_bundle.truth.recipe, variant=wrong_variant),
        ),
    )

    for corrupted in corruptions:
        with pytest.raises(EpisodeInvariantError) as raised:
            validate_episode_invariants(corrupted, config)
        assert raised.value.context == {}
        report = validate_episode_invariants(corrupted, config, strict=False)
        assert report.valid is False
        assert "rng_provenance" in report.check_ids


def test_independent_node_permutation_provenance_has_stable_check_id(
    config: Phase1Config,
) -> None:
    """A counterfactually relabeled private path must name its provenance check."""
    from silent_cascade.env.generator import IndependentEpisodeRequest
    from silent_cascade.env.invariants import validate_episode_invariants

    bundle = generate_independent_episode(
        config,
        IndependentEpisodeRequest(
            split_namespace=SplitNamespace.PHASE1_GATE,
            suite=SuiteName.IID_PRIMARY,
            root_seed=2026083011,
            episode_index=32,
            requested_path_length=2,
            variant=EpisodeVariant.POSITIVE,
            allocation_quartet_index=8,
        ),
        public_id_seed=2026083012,
    )
    source = bundle.truth.relevant_node_path[0]
    relabel = {source: 0, 0: source}
    counterfactual_path = tuple(relabel.get(node, node) for node in bundle.truth.relevant_node_path)
    assert counterfactual_path != bundle.truth.relevant_node_path
    corrupted = _bundle_with(bundle, relevant_node_path=counterfactual_path)

    with pytest.raises(
        EpisodeInvariantError,
        match="independent node permutation provenance disagrees",
    ) as raised:
        validate_episode_invariants(corrupted, config)
    assert raised.value.check_id == "node_permutation_provenance"

    report = validate_episode_invariants(corrupted, config, strict=False)
    assert report.valid is False
    assert report.check_ids == ("node_permutation_provenance",)


@pytest.mark.parametrize(
    ("mutation", "description"),
    [
        (
            lambda bundle: _replace_fact(
                bundle,
                0,
                LinkFact(bundle.truth.relevant_node_path[0], bundle.truth.relevant_node_path[-1]),
            ),
            "shortcut",
        ),
        (
            lambda bundle: _replace_fact(
                bundle,
                0,
                LinkFact(bundle.truth.relevant_node_path[0], bundle.truth.relevant_node_path[1]),
            ),
            "duplicate fact",
        ),
        (
            lambda bundle: _replace_fact(
                bundle,
                0,
                LinkFact(bundle.truth.relevant_node_path[-1], 60),
            ),
            "reachable-source distractor",
        ),
        (
            lambda bundle: _bundle_with(
                bundle, relevant_node_path=bundle.truth.relevant_node_path[:-1]
            ),
            "corrupted truth path",
        ),
        (
            lambda bundle: _bundle_with(bundle, terminal_record_id=0),
            "corrupted private terminal record",
        ),
    ],
)
def test_episode_validator_rejects_realistic_artifact_corruption(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
    mutation: object,
    description: str,
) -> None:
    """Changing the named public/truth field must not survive independent validation."""
    from silent_cascade.env.invariants import validate_episode_invariants

    assert callable(mutation), description
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(mutation(cohort[0]), config)  # type: ignore[operator]


def test_episode_validator_rejects_a_reachable_second_terminal(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """Dropping the terminal uniqueness branch would let a primary ambiguity through."""
    from silent_cascade.env.invariants import validate_episode_invariants

    positive = next(item for item in cohort if item.truth.recipe.variant is EpisodeVariant.POSITIVE)
    terminal_node = positive.truth.relevant_node_path[-1]
    facts = _fact_events(positive)
    old = facts[0]
    facts[0] = ExternalEvent(
        old.event_id,
        old.timestamp,
        ExternalEventKind.FACT,
        SafeFact(terminal_node),
    )
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(
            _bundle_with(positive, events=(*facts, positive.public.events[-1])), config
        )


def test_episode_validator_rejects_bad_terminal_counts_and_unequal_public_timing(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """Removing kind-count or timing checks would accept broken primary evidence."""
    from silent_cascade.env.invariants import validate_episode_invariants

    positive = next(item for item in cohort if item.truth.recipe.variant is EpisodeVariant.POSITIVE)
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(_replace_fact(positive, 0, SafeFact(59)), config)

    facts = _fact_events(positive)
    old = facts[1]
    facts[1] = ExternalEvent(old.event_id, facts[0].timestamp, old.kind, old.payload)
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(
            _bundle_with(positive, events=(*facts, positive.public.events[-1])), config
        )


def test_episode_validator_rejects_capacity_truth_window_and_id_corruption(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """Validation must catch corruption that bypasses immutable constructor guards."""
    from silent_cascade.env.invariants import validate_episode_invariants

    positive = next(item for item in cohort if item.truth.recipe.variant is EpisodeVariant.POSITIVE)
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(_bundle_with(positive, action_window_end=999.0), config)

    overflow = _bundle_with(positive)
    object.__setattr__(overflow.public.init, "memory_capacity", 1)
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(overflow, config)

    invalid_id = _bundle_with(positive)
    object.__setattr__(invalid_id.public.init, "episode_public_id", "not-a-uuid")
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(invalid_id, config)


def test_cohort_validator_rejects_cross_member_mismatch_and_trace_ambiguity(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """Changing matching or reusing a record ID must fail the cohort contract."""
    from silent_cascade.env.invariants import (
        validate_cohort_invariants,
        validate_episode_invariants,
    )

    changed_delay = _bundle_with(cohort[0], episode_delay=cohort[0].truth.episode_delay + 1.0)
    with pytest.raises(EpisodeInvariantError):
        validate_cohort_invariants((changed_delay, cohort[1], cohort[2], cohort[3]), config)

    trace_ambiguous = _bundle_with(
        cohort[0], relevant_record_ids=(cohort[0].truth.relevant_record_ids[0],) * 4
    )
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(trace_ambiguous, config)


def test_cohort_validator_rejects_variant_and_fact_signature_mismatch(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """A mismatched four-member allocation must never be accepted as a cohort."""
    from silent_cascade.env.invariants import validate_cohort_invariants

    with pytest.raises(EpisodeInvariantError):
        validate_cohort_invariants((cohort[0], cohort[0], cohort[2], cohort[3]), config)


def test_episode_validator_rejects_recipe_key_and_gap_contract_corruption(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """Removing recipe/key/gap checks would accept structurally valid damaged artifacts."""
    from silent_cascade.env.invariants import validate_episode_invariants

    bundle = cohort[0]
    recipe = replace(bundle.truth.recipe, evaluation_suite=SuiteName.OOD_DEPTH)
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(_bundle_with(bundle, recipe=recipe), config)

    facts = _fact_events(bundle)
    first = facts[0]
    facts[0] = ExternalEvent(first.event_id, 0.01, first.kind, first.payload)
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(
            _bundle_with(bundle, events=(*facts, bundle.public.events[-1])), config
        )


def test_episode_validator_rejects_every_terminal_delay(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """Each terminal delay, including unreachable hazards, is a primary invariant."""
    from silent_cascade.env.invariants import validate_episode_invariants

    bundle = cohort[0]
    facts = _fact_events(bundle)
    hazard_index = next(
        index
        for index, event in enumerate(facts)
        if isinstance(event.payload, HazardFact)
        and event.payload.node != bundle.truth.relevant_node_path[-1]
    )
    hazard = facts[hazard_index]
    assert isinstance(hazard.payload, HazardFact)
    facts[hazard_index] = ExternalEvent(
        hazard.event_id,
        hazard.timestamp,
        hazard.kind,
        HazardFact(hazard.payload.node, hazard.payload.hazard_type, hazard.payload.delay + 1.0),
    )
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(
            _bundle_with(bundle, events=(*facts, bundle.public.events[-1])), config
        )


def test_episode_validator_rejects_disconnected_link_cycles(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """Irrelevant LINK components may not contain cycles after artifact corruption."""
    from silent_cascade.env.invariants import validate_episode_invariants

    bundle = cohort[0]
    first, second = _disconnected_link_indexes(bundle)[:2]
    facts = _fact_events(bundle)
    source, target = _unused_link_pair(bundle)
    for index, payload in ((first, LinkFact(source, target)), (second, LinkFact(target, source))):
        old = facts[index]
        facts[index] = ExternalEvent(old.event_id, old.timestamp, old.kind, payload)
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(
            _bundle_with(bundle, events=(*facts, bundle.public.events[-1])), config
        )


def test_episode_validator_rejects_duplicate_endpoints_and_a_genuine_branch(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """Confidence cannot distinguish duplicate LINK structure or hide a branch."""
    from silent_cascade.env.invariants import validate_episode_invariants

    bundle = cohort[0]
    facts = _fact_events(bundle)
    original_link = next(event for event in facts if isinstance(event.payload, LinkFact))
    duplicate_index = _disconnected_link_indexes(bundle)[0]
    assert isinstance(original_link.payload, LinkFact)
    old = facts[duplicate_index]
    facts[duplicate_index] = ExternalEvent(
        old.event_id,
        old.timestamp,
        old.kind,
        LinkFact(
            original_link.payload.source_node,
            original_link.payload.target_node,
            confidence=0.5,
        ),
    )
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(
            _bundle_with(bundle, events=(*facts, bundle.public.events[-1])), config
        )


def test_episode_validator_rejects_a_genuine_reachable_branch(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """An added outgoing edge at a reachable node cannot be treated as a distractor."""
    from silent_cascade.env.invariants import validate_episode_invariants

    bundle = cohort[0]

    branch_index = _disconnected_link_indexes(bundle)[0]
    _, target = _unused_link_pair(bundle)
    facts = _fact_events(bundle)
    old = facts[branch_index]
    facts[branch_index] = ExternalEvent(
        old.event_id,
        old.timestamp,
        old.kind,
        LinkFact(bundle.truth.relevant_node_path[0], target),
    )
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(
            _bundle_with(bundle, events=(*facts, bundle.public.events[-1])), config
        )


def test_episode_validator_rejects_wrong_length_overflow_and_all_invalid_id_forms(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """Primary size and public-ID checks must fail even after constructor bypass."""
    from silent_cascade.env.invariants import validate_episode_invariants

    bundle = cohort[0]
    wrong_recipe = replace(bundle.truth.recipe, requested_path_length=2)
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(_bundle_with(bundle, recipe=wrong_recipe), config)

    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(_fact_overflow_bundle(bundle), config)

    for public_id in (
        "",
        "not-a-uuid",
        "00000000-0000-1000-8000-000000000000",
        bundle.public.init.episode_public_id.upper(),
    ):
        with pytest.raises(EpisodeInvariantError):
            validate_episode_invariants(_with_public_id(bundle, public_id), config)


def test_cohort_validator_rejects_identity_and_construction_order_provenance(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """Matched summaries alone cannot certify member-local RNG provenance."""
    from silent_cascade.env.invariants import validate_cohort_invariants

    with pytest.raises(EpisodeInvariantError):
        validate_cohort_invariants(
            (_identity_node_permutation_bundle(cohort[0]), *cohort[1:]), config
        )
    with pytest.raises(EpisodeInvariantError):
        validate_cohort_invariants((_construction_order_bundle(cohort[0]), *cohort[1:]), config)


def test_episode_validator_rejects_structural_coordinate_substitution(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """Changing only a member coordinate must invalidate its authenticated RNG provenance."""
    from silent_cascade.env.invariants import validate_episode_invariants

    bundle = cohort[0]
    coordinate = bundle.truth.key.coordinate
    assert isinstance(coordinate, MatchedEpisodeCoordinate)
    altered_key = replace(
        bundle.truth.key,
        coordinate=MatchedEpisodeCoordinate(
            "matched", coordinate.cohort_index, (coordinate.member_index + 1) % 4
        ),
    )
    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(_bundle_with(bundle, key=altered_key), config)


def test_non_strict_cohort_failure_marks_every_report_invalid(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """Cohort-only mismatch cannot be hidden by individually valid reports."""
    from silent_cascade.env.invariants import validate_cohort_invariants

    changed_recipe = replace(
        cohort[0].truth.recipe,
        distractor_link_count=cohort[0].truth.recipe.distractor_link_count + 1,
    )
    reports = validate_cohort_invariants(
        (_bundle_with(cohort[0], recipe=changed_recipe), *cohort[1:]), config, strict=False
    )
    assert all(not report.valid for report in reports)
    assert all("cohort_matching" in report.check_ids for report in reports)


def test_validators_reject_uniform_private_distractor_count_corruption(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
) -> None:
    """Recipe counts must equal independently parsed non-reachable LINK records."""
    from silent_cascade.env.invariants import (
        validate_cohort_invariants,
        validate_episode_invariants,
    )

    corrupted = tuple(
        _bundle_with(
            bundle,
            recipe=replace(
                bundle.truth.recipe,
                distractor_link_count=bundle.truth.recipe.distractor_link_count + 1,
            ),
        )
        for bundle in cohort
    )
    assert len(corrupted) == 4
    for bundle in corrupted:
        with pytest.raises(EpisodeInvariantError):
            validate_episode_invariants(bundle, config)
        report = validate_episode_invariants(bundle, config, strict=False)
        assert not report.valid
        assert "recipe_distractor_count" in report.check_ids
    with pytest.raises(EpisodeInvariantError):
        validate_cohort_invariants(corrupted, config)
    reports = validate_cohort_invariants(corrupted, config, strict=False)
    assert all(not report.valid for report in reports)
    assert all("recipe_distractor_count" in report.check_ids for report in reports)
    assert all("cohort_matching" in report.check_ids for report in reports)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda bundle: object.__setattr__(bundle.truth.key, "root_seed", -1),
        lambda bundle: object.__setattr__(bundle.truth.key.coordinate, "cohort_index", -1),
        lambda bundle: object.__setattr__(bundle.truth.key.coordinate, "member_index", 4),
        lambda bundle: (
            object.__setattr__(bundle.truth.recipe, "accepted_attempt", 1_000),
            object.__setattr__(bundle.truth, "rejection_count", 1_000),
        ),
        lambda bundle: (
            object.__setattr__(bundle.truth.key, "split_namespace", "malformed"),
            object.__setattr__(bundle.truth.key, "suite", "malformed"),
            object.__setattr__(bundle.truth.recipe, "evaluation_suite", "malformed"),
        ),
    ],
)
def test_invalid_persisted_rng_coordinates_raise_safe_typed_invariant_errors(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
    mutation: object,
) -> None:
    """Corrupted RNG fields must not expose lower-layer validation errors or values."""
    from silent_cascade.env.invariants import validate_episode_invariants

    assert callable(mutation)
    bundle = _bundle_with(cohort[0])
    mutation(bundle)  # type: ignore[operator]
    with pytest.raises(EpisodeInvariantError) as raised:
        validate_episode_invariants(bundle, config)
    assert raised.value.context == {}
    message = raised.value.message
    assert "-1" not in message
    assert "1000" not in message
    assert "malformed" not in message


@pytest.mark.parametrize(
    "mutation",
    [
        lambda bundle: object.__setattr__(bundle.truth, "rejection_count", False),
        lambda bundle: object.__setattr__(bundle.truth, "rejection_count", -1),
        lambda bundle: object.__setattr__(bundle.truth, "rejection_count", 1_000),
        lambda bundle: object.__setattr__(bundle.truth, "rejection_count", "invalid"),
        lambda bundle: object.__setattr__(bundle.truth, "rejection_count", 1),
        lambda bundle: object.__setattr__(bundle.truth.key.coordinate, "mode", "wrong"),
        lambda bundle: object.__setattr__(bundle.truth.key.coordinate, "mode", ""),
        lambda bundle: object.__setattr__(bundle.truth.key.coordinate, "mode", True),
        lambda bundle: object.__setattr__(bundle.truth.key.coordinate, "mode", None),
    ],
)
def test_rejection_and_coordinate_provenance_is_strict_and_fail_closed(
    config: Phase1Config,
    cohort: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle],
    mutation: object,
) -> None:
    """Every persisted retry/mode corruption has one safe invariant failure surface."""
    from silent_cascade.env.invariants import (
        validate_cohort_invariants,
        validate_episode_invariants,
    )

    assert callable(mutation)
    corrupted = _bundle_with(cohort[0])
    mutation(corrupted)  # type: ignore[operator]
    with pytest.raises(EpisodeInvariantError) as raised:
        validate_episode_invariants(corrupted, config)
    assert raised.value.context == {}
    assert raised.value.message == "invalid RNG provenance"

    report = validate_episode_invariants(corrupted, config, strict=False)
    assert not report.valid
    assert "rng_provenance" in report.check_ids

    corrupted_cohort = (corrupted, *cohort[1:])
    with pytest.raises(EpisodeInvariantError) as raised:
        validate_cohort_invariants(corrupted_cohort, config)
    assert raised.value.context == {}
    assert raised.value.message == "invalid RNG provenance"

    reports = validate_cohort_invariants(corrupted_cohort, config, strict=False)
    assert all(not item.valid for item in reports)
    assert all("rng_provenance" in item.check_ids for item in reports)
    assert all("cohort_matching" in item.check_ids for item in reports)
