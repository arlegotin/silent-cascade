"""Task 14 contracts for private validation-manifest construction."""

from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.generator import CohortAllocation, CohortBlock
from silent_cascade.logging.manifest import ManifestAccessClass
from silent_cascade.provenance import (
    GENERATOR_SOURCE_PATHS,
    PHASE1_ANALYSIS_SOURCE_PATHS,
    EvidenceProvenance,
    SourceTreeFingerprint,
    public_id_seed_sha256,
)


def _config() -> Phase1Config:
    return resolve_config(
        Phase1Config, [Path("configs/base.yaml"), Path("configs/data/primary.yaml")]
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


def _provenance(
    *, allocation_id: str = "test-validation-v1", dirty: bool = False
) -> EvidenceProvenance:
    source = SourceTreeFingerprint(
        frame_version="sc-source-tree-v1",
        scope="generator",
        paths=GENERATOR_SOURCE_PATHS,
        sha256="a" * 64,
    )
    analysis = SourceTreeFingerprint(
        frame_version="sc-source-tree-v1",
        scope="phase1_analysis",
        paths=PHASE1_ANALYSIS_SOURCE_PATHS,
        sha256="b" * 64,
    )
    return EvidenceProvenance(
        schema_version="phase1-evidence-provenance-v1",
        plan_base_revision="c" * 40,
        source_commit="d" * 40,
        source_dirty=dirty,
        generator_version="ofd-v1",
        generation_mode="matched",
        allocation_id=allocation_id,
        split_namespace=SplitNamespace.DEBUG,
        config_sha256="e" * 64,
        generator_source=source,
        analysis_source=analysis,
        root_seed=41,
        public_id_seed_sha256=public_id_seed_sha256(91),
    )


def _allocation() -> CohortAllocation:
    return CohortAllocation(
        allocation_id="test-validation-v1",
        split_namespace=SplitNamespace.DEBUG,
        blocks=(
            CohortBlock(
                suite=SuiteName.IID_PRIMARY,
                requested_path_length=2,
                first_cohort_index=0,
                cohort_count=2,
            ),
        ),
    )


def test_debug_cohort_manifest_round_trips_every_entry() -> None:
    """A persisted coordinate, attempt, ID, and digest authenticate regeneration."""
    from silent_cascade.env.services import build_cohort_manifest, regenerate_entry

    manifest = build_cohort_manifest(
        _config(), _allocation(), _provenance(), 41, 91, access_class=ManifestAccessClass.DEBUG
    )

    assert manifest.episode_count == 8
    assert manifest.access_class is ManifestAccessClass.DEBUG
    for entry in manifest.entries:
        bundle = regenerate_entry(_config(), manifest, entry)
        assert bundle.public.init.episode_public_id == entry.episode_public_id
        assert bundle.truth.recipe.accepted_attempt == entry.accepted_attempt


def test_cohort_builder_fails_closed_on_provenance_mismatch() -> None:
    """Generation must not begin with an unauthenticated recipe boundary."""
    from silent_cascade.env.services import build_cohort_manifest

    with pytest.raises(ValueError, match="allocation"):
        build_cohort_manifest(
            _config(),
            _allocation(),
            _provenance(allocation_id="test-other"),
            41,
            91,
            access_class=ManifestAccessClass.DEBUG,
        )


@pytest.mark.parametrize(
    "suite",
    (SuiteName.BRANCHING_STRESS, SuiteName.NULL_NEAR_MISS_STRESS),
)
def test_regenerate_entry_dispatches_structural_stress_suites(suite: SuiteName) -> None:
    """Independent manifest recipes use the declared stress regeneration path."""
    from silent_cascade.env.episode import EpisodeVariant, episode_sha256
    from silent_cascade.env.generator import IndependentEpisodeRequest, generate_stress_episode
    from silent_cascade.env.services import regenerate_entry
    from silent_cascade.logging.manifest import (
        EpisodeManifest,
        EpisodeManifestEntry,
        IndependentManifestCoordinate,
        ManifestAccessClass,
    )
    from silent_cascade.rng import AllocationLabelKey, allocate_independent_variants

    variants = allocate_independent_variants(
        AllocationLabelKey("ofd-v1", SplitNamespace.DEBUG, suite, 41, 2, 2)
    )
    expected_variant = (
        EpisodeVariant.DISCONNECTED_NEGATIVE
        if suite is SuiteName.NULL_NEAR_MISS_STRESS
        else EpisodeVariant.POSITIVE
    )
    member_index = variants.index(expected_variant)
    request = IndependentEpisodeRequest(
        SplitNamespace.DEBUG, suite, 41, 5 + member_index, 2, expected_variant, 2
    )
    bundle = generate_stress_episode(_stress_config(), request, 91)
    provenance = _provenance().model_copy(
        update={"generation_mode": "independent", "allocation_id": "test-stress-v1"}
    )
    entries = tuple(
        EpisodeManifestEntry(
            episode_public_id=(
                bundle.public.init.episode_public_id
                if index == member_index
                else f"00000000-0000-4000-8000-0000000000{index + 10:02d}"
            ),
            split_namespace=SplitNamespace.DEBUG,
            suite=suite,
            coordinate=IndependentManifestCoordinate(
                episode_index=5 + index,
                allocation_quartet_index=2,
                quartet_member_index=index,
            ),
            requested_path_length=2,
            accepted_attempt=(bundle.truth.recipe.accepted_attempt if index == member_index else 0),
            episode_sha256=(
                episode_sha256(bundle) if index == member_index else f"{index + 1:x}" * 64
            ),
        )
        for index in range(4)
    )
    entry = entries[member_index]
    manifest = EpisodeManifest(
        schema_version=1,
        experiment_version="v1",
        access_class=ManifestAccessClass.DEBUG,
        provenance=provenance,
        suite=suite,
        public_id_seed=91,
        episode_count=4,
        entries=entries,
    )
    assert regenerate_entry(_stress_config(), manifest, entry) == bundle
