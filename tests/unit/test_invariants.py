"""Independent mutation checks for primary OFD episode bundles."""

from dataclasses import replace
from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import EpisodeBundle, EpisodeVariant, PublicEpisode
from silent_cascade.env.generator import CohortRequest, generate_matched_cohort
from silent_cascade.errors import EpisodeInvariantError
from silent_cascade.schemas import (
    ExternalEvent,
    ExternalEventKind,
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
