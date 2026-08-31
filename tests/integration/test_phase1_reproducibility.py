"""Task 14 deterministic order/chunk/fresh-process contracts."""

from pathlib import Path

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.generator import EpisodeBlock, IndependentAllocation
from silent_cascade.provenance import (
    GENERATOR_SOURCE_PATHS,
    PHASE1_ANALYSIS_SOURCE_PATHS,
    EvidenceProvenance,
    SourceTreeFingerprint,
    public_id_seed_sha256,
)


def _resolved():
    return resolve_config(
        Phase1Config, [Path("configs/base.yaml"), Path("configs/data/primary.yaml")]
    )


def _provenance(resolved) -> EvidenceProvenance:
    return EvidenceProvenance(
        schema_version="phase1-evidence-provenance-v1",
        plan_base_revision="c" * 40,
        source_commit="d" * 40,
        source_dirty=False,
        generator_version="ofd-v1",
        generation_mode="independent",
        allocation_id="test-independent-v1",
        split_namespace=SplitNamespace.DEBUG,
        config_sha256=resolved.sha256,
        generator_source=SourceTreeFingerprint(
            frame_version="sc-source-tree-v1",
            scope="generator",
            paths=GENERATOR_SOURCE_PATHS,
            sha256="a" * 64,
        ),
        analysis_source=SourceTreeFingerprint(
            frame_version="sc-source-tree-v1",
            scope="phase1_analysis",
            paths=PHASE1_ANALYSIS_SOURCE_PATHS,
            sha256="b" * 64,
        ),
        root_seed=41,
        public_id_seed_sha256=public_id_seed_sha256(91),
    )


def test_independent_reproducibility_is_order_chunk_and_hash_seed_stable() -> None:
    from silent_cascade.env.generator import generate_independent_episode, iter_independent_requests
    from silent_cascade.env.reproducibility import (
        IndependentAllocationReproducibilitySource,
        ReproducibilityDependencies,
        ReproducibilityRequest,
        check_reproducibility,
    )
    from silent_cascade.logging.manifest import load_manifest

    resolved = _resolved()
    allocation = IndependentAllocation(
        allocation_id="test-independent-v1",
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
    provenance = _provenance(resolved)
    deps = ReproducibilityDependencies.for_test(
        allocation,
        provenance,
        collect_provenance=lambda *_args, **_kwargs: provenance,
        load_verified_manifest=load_manifest,
        iter_independent=iter_independent_requests,
        generate_independent=generate_independent_episode,
    )
    report = check_reproducibility(
        ReproducibilityRequest(
            source=IndependentAllocationReproducibilitySource(
                allocation_id=allocation.allocation_id, root_seed=41, public_id_seed=91
            ),
            sample_size=8,
            chunk_sizes=(1, 3, 7),
            python_hash_seeds=(0, 1),
            verify_all_source_entries=True,
        ),
        resolved,
        deps=deps,
    )

    assert report.passed and report.mismatch_count == 0
    assert report.verified_source_entries == 8
    assert report.provenance == provenance
