"""Private Phase 1 manifest construction and authenticated regeneration."""

from typing import Literal

from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    EpisodeBundle,
    EpisodeVariant,
    episode_sha256,
    scale_episode_time,
)
from silent_cascade.env.generator import (
    VALIDATION_ALLOCATION,
    CohortAllocation,
    CohortRequest,
    IndependentEpisodeRequest,
    generate_matched_cohort,
    iter_cohort_requests,
    regenerate_independent_episode,
    regenerate_matched_episode,
    validate_validation_allocation,
)
from silent_cascade.logging.manifest import (
    EpisodeManifest,
    EpisodeManifestEntry,
    IndependentManifestCoordinate,
    ManifestAccessClass,
    MatchedManifestCoordinate,
)
from silent_cascade.provenance import EvidenceProvenance, public_id_seed_sha256
from silent_cascade.rng import AllocationLabelKey, allocate_independent_variants


def _validate_manifest_boundary(
    allocation: CohortAllocation,
    provenance: EvidenceProvenance,
    root_seed: int,
    public_id_seed: int,
    access_class: ManifestAccessClass,
) -> None:
    if type(root_seed) is not int or root_seed != provenance.root_seed:
        raise ValueError("root seed does not match provenance")
    if (
        type(public_id_seed) is not int
        or public_id_seed < 0
        or provenance.public_id_seed_sha256 != public_id_seed_sha256(public_id_seed)
    ):
        raise ValueError("public ID seed does not match provenance")
    if provenance.generation_mode != "matched":
        raise ValueError("matched manifest builder requires matched provenance")
    if provenance.allocation_id != allocation.allocation_id:
        raise ValueError("allocation does not match provenance")
    if provenance.split_namespace is not allocation.split_namespace:
        raise ValueError("split namespace does not match provenance")
    if access_class is ManifestAccessClass.DEBUG:
        if (
            allocation.split_namespace is not SplitNamespace.DEBUG
            or not allocation.allocation_id.startswith("test-")
        ):
            raise ValueError("debug manifests require a test- DEBUG allocation")
    elif access_class is not ManifestAccessClass.VALIDATION:
        raise ValueError("cohort builder can create only DEBUG or VALIDATION manifests")


def build_cohort_manifest(
    config: Phase1Config,
    allocation: CohortAllocation,
    provenance: EvidenceProvenance,
    root_seed: int,
    public_id_seed: int,
    *,
    access_class: Literal[ManifestAccessClass.DEBUG, ManifestAccessClass.VALIDATION],
) -> EpisodeManifest:
    """Build only immutable recipe entries; episode payloads never leave this seam."""
    if not isinstance(config, Phase1Config) or not isinstance(allocation, CohortAllocation):
        raise TypeError("config and allocation have invalid types")
    if not isinstance(provenance, EvidenceProvenance) or not isinstance(
        access_class, ManifestAccessClass
    ):
        raise TypeError("provenance and access class have invalid types")
    # The validation guard needs the exact config; keep DEBUG structurally reusable.
    if access_class is ManifestAccessClass.VALIDATION:
        validate_validation_allocation(allocation, config)
    _validate_manifest_boundary(allocation, provenance, root_seed, public_id_seed, access_class)
    suites = {block.suite for block in allocation.blocks}
    if len(suites) != 1:
        raise ValueError("a manifest must contain exactly one suite")
    entries: list[EpisodeManifestEntry] = []
    for request in iter_cohort_requests(allocation, root_seed):
        cohort = generate_matched_cohort(config, request, public_id_seed)
        for member_index, bundle in enumerate(cohort.episodes):
            entries.append(
                EpisodeManifestEntry(
                    episode_public_id=bundle.public.init.episode_public_id,
                    split_namespace=request.split_namespace,
                    suite=request.suite,
                    coordinate=MatchedManifestCoordinate(
                        cohort_index=request.cohort_index, member_index=member_index
                    ),
                    requested_path_length=request.requested_path_length,
                    accepted_attempt=cohort.accepted_attempt,
                    episode_sha256=episode_sha256(bundle),
                )
            )
    return EpisodeManifest(
        schema_version=1,
        experiment_version=config.experiment_version,
        access_class=access_class,
        provenance=provenance,
        suite=next(iter(suites)),
        public_id_seed=public_id_seed,
        episode_count=len(entries),
        entries=tuple(entries),
    )


def build_validation_manifest(
    config: Phase1Config,
    provenance: EvidenceProvenance,
    root_seed: int,
    public_id_seed: int,
) -> EpisodeManifest:
    """Build the sole production validation allocation and reject dirty evidence."""
    if provenance.source_dirty:
        raise ValueError("production validation provenance must be clean")
    manifest = build_cohort_manifest(
        config,
        VALIDATION_ALLOCATION,
        provenance,
        root_seed,
        public_id_seed,
        access_class=ManifestAccessClass.VALIDATION,
    )
    counts = {variant: 0 for variant in EpisodeVariant}
    for entry in manifest.entries:
        bundle = regenerate_entry(config, manifest, entry)
        counts[bundle.truth.recipe.variant] += 1
    if counts != {
        EpisodeVariant.POSITIVE: 5_000,
        EpisodeVariant.SAFE_NEGATIVE: 2_500,
        EpisodeVariant.DISCONNECTED_NEGATIVE: 2_500,
    }:
        raise ValueError("frozen validation manifest has an invalid variant allocation")
    return manifest


def _assert_entry(bundle: EpisodeBundle, entry: EpisodeManifestEntry) -> EpisodeBundle:
    if (
        bundle.public.init.episode_public_id != entry.episode_public_id
        or bundle.truth.recipe.accepted_attempt != entry.accepted_attempt
        or episode_sha256(bundle) != entry.episode_sha256
    ):
        raise ValueError("regenerated episode does not match manifest entry")
    return bundle


def regenerate_entry(
    config: Phase1Config, manifest: EpisodeManifest, entry: EpisodeManifestEntry
) -> EpisodeBundle:
    """Regenerate one authenticated recipe, never inferring a quartet position."""
    if entry not in manifest.entries:
        raise ValueError("entry is not a member of the manifest")
    if entry.split_namespace is not manifest.provenance.split_namespace:
        raise ValueError("entry split namespace differs from manifest provenance")
    coordinate = entry.coordinate
    if isinstance(coordinate, MatchedManifestCoordinate):
        request = CohortRequest(
            split_namespace=entry.split_namespace,
            suite=entry.suite,
            root_seed=manifest.provenance.root_seed,
            cohort_index=coordinate.cohort_index,
            requested_path_length=entry.requested_path_length,
        )
        return _assert_entry(
            regenerate_matched_episode(
                config,
                request,
                manifest.public_id_seed,
                entry.episode_public_id,
                entry.accepted_attempt,
            ),
            entry,
        )
    if not isinstance(coordinate, IndependentManifestCoordinate):
        raise TypeError("unsupported manifest coordinate")
    source_suite = (
        SuiteName.IID_PRIMARY
        if entry.suite
        in {
            SuiteName.CLOCK_SCALE_0_1X,
            SuiteName.CLOCK_SCALE_10X,
        }
        else entry.suite
    )
    variants = allocate_independent_variants(
        AllocationLabelKey(
            generator_version="ofd-v1",
            split_namespace=entry.split_namespace,
            suite=source_suite,
            root_seed=manifest.provenance.root_seed,
            requested_path_length=entry.requested_path_length,
            allocation_quartet_index=coordinate.allocation_quartet_index,
        )
    )
    request = IndependentEpisodeRequest(
        split_namespace=entry.split_namespace,
        suite=source_suite,
        root_seed=manifest.provenance.root_seed,
        episode_index=coordinate.episode_index,
        requested_path_length=entry.requested_path_length,
        variant=variants[coordinate.quartet_member_index],
        allocation_quartet_index=coordinate.allocation_quartet_index,
    )
    if entry.suite in {SuiteName.CLOCK_SCALE_0_1X, SuiteName.CLOCK_SCALE_10X}:
        if entry.parent_public_id is None or entry.parent_episode_sha256 is None:
            raise ValueError("clock manifest entry is missing its parent")
        parent = regenerate_independent_episode(
            config, request, manifest.public_id_seed, entry.parent_public_id, entry.accepted_attempt
        )
        if episode_sha256(parent) != entry.parent_episode_sha256:
            raise ValueError("clock parent hash does not match manifest entry")
        return _assert_entry(
            scale_episode_time(parent, entry.suite, entry.episode_public_id), entry
        )
    return _assert_entry(
        regenerate_independent_episode(
            config,
            request,
            manifest.public_id_seed,
            entry.episode_public_id,
            entry.accepted_attempt,
        ),
        entry,
    )
