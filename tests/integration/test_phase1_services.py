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
