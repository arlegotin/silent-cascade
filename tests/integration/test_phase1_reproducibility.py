"""Task 14 deterministic order/chunk/fresh-process contracts."""

from dataclasses import replace
from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.generator import EpisodeBlock, IndependentAllocation
from silent_cascade.errors import (
    ArtifactIntegrityError,
    ConfigurationError,
    ManifestAccessError,
    ProvenanceError,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
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


def test_production_validation_reproducibility_uses_freeze_provenance() -> None:
    """The frozen manifest and its verifier must authenticate the same final source scope."""
    from silent_cascade.env.generator import VALIDATION_ALLOCATION
    from silent_cascade.env.reproducibility import PRODUCTION_REPRODUCIBILITY_DEPENDENCIES
    from silent_cascade.env.services import PRODUCTION_DEPENDENCIES

    resolved = _resolved()
    arguments = {
        "repo_root": Path.cwd(),
        "generation_mode": "matched",
        "allocation_id": VALIDATION_ALLOCATION.allocation_id,
        "split_namespace": SplitNamespace.VALIDATION,
        "root_seed": 2026083001,
        "public_id_seed": 2026083002,
        "analysis_seeds": {},
    }

    freeze = PRODUCTION_DEPENDENCIES.collect_provenance(resolved, **arguments)  # type: ignore[arg-type]
    reproducibility = PRODUCTION_REPRODUCIBILITY_DEPENDENCIES.collect_provenance(
        resolved,
        **arguments,  # type: ignore[arg-type]
    )

    assert freeze == reproducibility
    assert reproducibility.analysis_source.scope == "phase1_analysis"


def test_production_independent_reproducibility_uses_common_gate_provenance() -> None:
    """Independent reproducibility must share oracle/leakage's final source authority."""
    from silent_cascade.env.generator import PHASE1_GATE_ALLOCATION
    from silent_cascade.env.reproducibility import PRODUCTION_REPRODUCIBILITY_DEPENDENCIES
    from silent_cascade.env.services import PRODUCTION_DEPENDENCIES

    resolved = _resolved()
    arguments = {
        "repo_root": Path.cwd(),
        "generation_mode": "independent",
        "allocation_id": PHASE1_GATE_ALLOCATION.allocation_id,
        "split_namespace": SplitNamespace.PHASE1_GATE,
        "root_seed": 2026083011,
        "public_id_seed": 2026083012,
        "analysis_seeds": {},
    }

    gate = PRODUCTION_DEPENDENCIES.collect_provenance(resolved, **arguments)  # type: ignore[arg-type]
    reproducibility = PRODUCTION_REPRODUCIBILITY_DEPENDENCIES.collect_provenance(
        resolved,
        **arguments,  # type: ignore[arg-type]
    )

    assert gate == reproducibility
    assert reproducibility.analysis_source.scope == "phase1_analysis"


def test_independent_reproducibility_is_order_chunk_and_hash_seed_stable() -> None:
    from silent_cascade.env.generator import generate_independent_episode, iter_independent_requests
    from silent_cascade.env.reproducibility import (
        FreshWorkOrder,
        IndependentAllocationReproducibilitySource,
        ReproducibilityDependencies,
        ReproducibilityRequest,
        _entry_for,
        _run_fresh_process,
        check_reproducibility,
        independent_sample_membership_sha256,
        select_independent_reproducibility_sample,
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
    observed_work_orders: list[tuple[tuple[int, int, int], ...]] = []

    def run_and_observe(work_order: bytes, hash_seed: int) -> bytes:
        payload = FreshWorkOrder.model_validate_json(work_order)
        observed_work_orders.append(
            tuple(
                (
                    entry.coordinate.episode_index,
                    entry.coordinate.allocation_quartet_index,
                    entry.coordinate.quartet_member_index,
                )
                for entry in payload.entries
            )
        )
        return _run_fresh_process(work_order, hash_seed)

    deps = replace(deps, run_fresh_process=run_and_observe)
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
    assert report.schema_version == "phase1-reproducibility-v2"
    assert report.verified_source_entries == 8
    assert report.provenance == provenance
    assert report.namespace_evidence.generation_mode == "independent"
    assert report.namespace_evidence.public_id_seed == 91
    assert report.namespace_evidence.accepted_draw_count == 8
    assert report.namespace_evidence.seed_token_count == 56
    assert report.namespace_evidence.base_public_id_count == 8
    assert report.namespace_evidence.clock_public_id_count == 0
    requests = tuple(iter_independent_requests(allocation, 41))
    known_sample = select_independent_reproducibility_sample(requests, "1" * 64, 4)
    assert tuple(item.episode_index for item in known_sample) == (9, 10, 8, 5)
    assert independent_sample_membership_sha256("1" * 64, known_sample) == (
        "8ea03fe787a7330a598867792072f0f5120e0fc207da7e5ac833e2542cf848e0"
    )
    assert _entry_for(generate_independent_episode(resolved.config, requests[0], 91))
    expected_coordinates = tuple(
        (
            request.episode_index,
            request.allocation_quartet_index,
            request.quartet_member_index,
        )
        for request in requests
    )
    assert tuple(member for _, _, member in expected_coordinates) == (0, 1, 2, 3) * 2
    assert len(observed_work_orders) == 2
    assert observed_work_orders[0] == observed_work_orders[1]
    assert tuple(sorted(observed_work_orders[0])) == expected_coordinates


def test_independent_rejects_dirty_provenance_before_generation() -> None:
    """A dirty scientific source is not reproducible evidence and must fail closed."""
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
    dirty = _provenance(resolved).model_copy(update={"source_dirty": True})
    calls = 0

    def never_generate(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return generate_independent_episode(*args, **kwargs)

    deps = ReproducibilityDependencies.for_test(
        allocation,
        _provenance(resolved),
        collect_provenance=lambda *_args, **_kwargs: dirty,
        load_verified_manifest=load_manifest,
        iter_independent=iter_independent_requests,
        generate_independent=never_generate,  # type: ignore[arg-type]
    )
    request = ReproducibilityRequest(
        source=IndependentAllocationReproducibilitySource(
            allocation_id=allocation.allocation_id, root_seed=41, public_id_seed=91
        ),
        sample_size=8,
        chunk_sizes=(1,),
        python_hash_seeds=(0,),
        verify_all_source_entries=True,
    )

    with pytest.raises(ProvenanceError, match="provenance"):
        check_reproducibility(request, resolved, deps=deps)
    assert calls == 0


def test_reproducibility_rejects_empty_matrix_before_generation() -> None:
    """A report cannot claim order/process coverage it did not execute."""
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
    deps = ReproducibilityDependencies.for_test(
        allocation,
        _provenance(resolved),
        collect_provenance=lambda *_args, **_kwargs: _provenance(resolved),
        load_verified_manifest=load_manifest,
        iter_independent=iter_independent_requests,
        generate_independent=generate_independent_episode,
    )
    request = ReproducibilityRequest(
        source=IndependentAllocationReproducibilitySource(
            allocation_id=allocation.allocation_id, root_seed=41, public_id_seed=91
        ),
        sample_size=8,
        chunk_sizes=(),
        python_hash_seeds=(),
        verify_all_source_entries=True,
    )

    with pytest.raises(ConfigurationError, match="matrix"):
        check_reproducibility(request, resolved, deps=deps)


@pytest.mark.parametrize(
    "update",
    [
        {"sample_size": 999},
        {"chunk_sizes": (2,)},
        {"python_hash_seeds": (3,)},
        {"verify_all_source_entries": False},
    ],
)
def test_reproducibility_rejects_altered_production_matrix_before_generation(
    update: dict[str, object],
) -> None:
    """The production evidence matrix is a frozen protocol input, not a tuning flag."""
    from silent_cascade.env.reproducibility import (
        IndependentAllocationReproducibilitySource,
        ReproducibilityRequest,
        _require_execution_matrix,
    )

    request = ReproducibilityRequest(
        source=IndependentAllocationReproducibilitySource(
            allocation_id="phase1-independent-gate-v1", root_seed=41, public_id_seed=91
        ),
        sample_size=1_000,
        chunk_sizes=(1, 3, 7),
        python_hash_seeds=(0, 1),
        verify_all_source_entries=True,
    ).model_copy(update=update)

    with pytest.raises(ConfigurationError, match="production matrix"):
        _require_execution_matrix(request, production_mode=True)


def test_manifest_source_executes_the_full_authenticated_matrix(tmp_path: Path) -> None:
    """Manifest mode must genuinely execute chunks and both fresh hash-seed runs."""
    from silent_cascade.env.generator import CohortAllocation, CohortBlock
    from silent_cascade.env.reproducibility import (
        PRODUCTION_REPRODUCIBILITY_DEPENDENCIES,
        ManifestReproducibilitySource,
        ReproducibilityRequest,
        _run_fresh_process,
        check_reproducibility,
        manifest_sample_membership_sha256,
        select_manifest_reproducibility_sample,
    )
    from silent_cascade.env.services import build_cohort_manifest, regenerate_entry
    from silent_cascade.logging.manifest import ManifestAccessClass, load_manifest, publish_manifest

    resolved = _resolved()
    provenance = _provenance(resolved).model_copy(
        update={"generation_mode": "matched", "allocation_id": "test-manifest-v1"}
    )
    allocation = CohortAllocation(
        allocation_id="test-manifest-v1",
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
    manifest = build_cohort_manifest(
        resolved.config, allocation, provenance, 41, 91, access_class=ManifestAccessClass.DEBUG
    )
    path = tmp_path / "debug-manifest.json"
    publish_manifest(path, manifest)
    runs: list[int] = []

    def runner(work_order: bytes, hash_seed: int) -> bytes:
        runs.append(hash_seed)
        return _run_fresh_process(work_order, hash_seed)

    deps = replace(
        PRODUCTION_REPRODUCIBILITY_DEPENDENCIES,
        production_mode=False,
        collect_provenance=lambda *_args, **_kwargs: provenance,
        load_verified_manifest=load_manifest,
        regenerate_manifest_entry=regenerate_entry,
        run_fresh_process=runner,
    )
    report = check_reproducibility(
        ReproducibilityRequest(
            source=ManifestReproducibilitySource(manifest_path=path),
            sample_size=8,
            chunk_sizes=(1, 3, 7),
            python_hash_seeds=(0, 1),
            verify_all_source_entries=True,
        ),
        resolved,
        deps=deps,
    )

    assert report.modes == ("forward", "reverse", "chunked", "fresh_process")
    assert runs == [0, 1]
    assert report.schema_version == "phase1-reproducibility-v2"
    assert report.namespace_evidence.generation_mode == "matched"
    assert report.namespace_evidence.public_id_seed == 91
    assert report.namespace_evidence.accepted_draw_count == 2
    assert report.namespace_evidence.seed_token_count == 40
    assert report.namespace_evidence.base_public_id_count == 8
    source_hash = sha256_bytes(canonical_json_bytes(manifest))
    selected = select_manifest_reproducibility_sample(manifest, source_hash, 4)
    assert len(selected) == 4
    assert manifest_sample_membership_sha256(source_hash, selected) != "0" * 64


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("split_namespace", SplitNamespace.VALIDATION),
        ("suite", SuiteName.OOD_DEPTH),
        ("cohort_index", 99),
        ("requested_path_length", 3),
        ("accepted_attempt", 1),
    ),
)
def test_matched_namespace_rejects_mixed_cohort_signatures(
    field: str,
    replacement: object,
) -> None:
    """One token stream may authenticate only four entries from the same cohort draw."""
    from silent_cascade.env.generator import CohortAllocation, CohortBlock
    from silent_cascade.env.reproducibility import _matched_manifest_namespace
    from silent_cascade.env.services import build_cohort_manifest
    from silent_cascade.logging.manifest import ManifestAccessClass

    resolved = _resolved()
    provenance = _provenance(resolved).model_copy(
        update={"generation_mode": "matched", "allocation_id": "test-signature-v1"}
    )
    allocation = CohortAllocation(
        allocation_id="test-signature-v1",
        split_namespace=SplitNamespace.DEBUG,
        blocks=(
            CohortBlock(
                suite=SuiteName.IID_PRIMARY,
                requested_path_length=2,
                first_cohort_index=0,
                cohort_count=1,
            ),
        ),
    )
    manifest = build_cohort_manifest(
        resolved.config, allocation, provenance, 41, 91, access_class=ManifestAccessClass.DEBUG
    )
    entries = list(manifest.entries)
    target = entries[1]
    if field == "cohort_index":
        target = target.model_copy(
            update={"coordinate": target.coordinate.model_copy(update={field: replacement})}
        )
    else:
        target = target.model_copy(update={field: replacement})
    entries[1] = target
    mixed = manifest.model_copy(update={"entries": tuple(entries)})

    with pytest.raises(ArtifactIntegrityError, match="namespace draw is inconsistent"):
        _matched_manifest_namespace(mixed)


def test_reproducibility_rejects_colliding_actual_construction_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproducibility must audit derived tokens, not trust unique allocation coordinates."""
    import silent_cascade.env.reproducibility as reproducibility
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
    monkeypatch.setattr(
        reproducibility,
        "independent_seed_tokens",
        lambda _request, _attempt: ("0" * 64,) * 7,
    )

    with pytest.raises(ProvenanceError, match="construction token collision"):
        check_reproducibility(
            ReproducibilityRequest(
                source=IndependentAllocationReproducibilitySource(
                    allocation_id=allocation.allocation_id,
                    root_seed=41,
                    public_id_seed=91,
                ),
                sample_size=8,
                chunk_sizes=(1, 3, 7),
                python_hash_seeds=(0, 1),
                verify_all_source_entries=True,
            ),
            resolved,
            deps=deps,
        )


def test_reproducibility_rejects_noncanonical_fresh_worker_output() -> None:
    """A parseable result with leaked fields cannot become evidence."""
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
    deps = ReproducibilityDependencies.for_test(
        allocation,
        _provenance(resolved),
        collect_provenance=lambda *_args, **_kwargs: _provenance(resolved),
        load_verified_manifest=load_manifest,
        iter_independent=iter_independent_requests,
        generate_independent=generate_independent_episode,
    )
    deps = replace(
        deps,
        run_fresh_process=lambda *_args, **_kwargs: b'{"leaked":true}',
    )
    request = ReproducibilityRequest(
        source=IndependentAllocationReproducibilitySource(
            allocation_id=allocation.allocation_id, root_seed=41, public_id_seed=91
        ),
        sample_size=8,
        chunk_sizes=(1,),
        python_hash_seeds=(0,),
        verify_all_source_entries=True,
    )

    with pytest.raises(ArtifactIntegrityError, match="malformed"):
        check_reproducibility(request, resolved, deps=deps)


def test_clean_worktree_debug_manifest_is_rejected_by_production_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean source tree cannot make a DEBUG manifest production evidence."""
    import subprocess

    from silent_cascade.config import resolve_config
    from silent_cascade.env.generator import CohortAllocation, CohortBlock
    from silent_cascade.env.reproducibility import (
        PRODUCTION_REPRODUCIBILITY_DEPENDENCIES,
        ManifestReproducibilitySource,
        ReproducibilityRequest,
        check_reproducibility,
    )
    from silent_cascade.env.services import build_cohort_manifest
    from silent_cascade.logging.manifest import ManifestAccessClass, publish_manifest
    from silent_cascade.provenance import collect_evidence_provenance

    repository = Path.cwd()
    worktree = tmp_path / "clean"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    try:
        monkeypatch.chdir(worktree)
        resolved = resolve_config(
            Phase1Config,
            [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
        )
        allocation = CohortAllocation(
            allocation_id="test-manifest-v1",
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
        provenance = collect_evidence_provenance(
            resolved,
            repo_root=worktree,
            generation_mode="matched",
            allocation_id=allocation.allocation_id,
            split_namespace=allocation.split_namespace,
            root_seed=41,
            public_id_seed=91,
            analysis_seeds={},
        )
        assert not provenance.source_dirty
        manifest = build_cohort_manifest(
            resolved.config,
            allocation,
            provenance,
            41,
            91,
            access_class=ManifestAccessClass.DEBUG,
        )
        path = worktree / "debug-manifest.json"
        publish_manifest(path, manifest)

        with pytest.raises(ManifestAccessError, match="canonical validation"):
            check_reproducibility(
                ReproducibilityRequest(
                    source=ManifestReproducibilitySource(manifest_path=path),
                    sample_size=1_000,
                    chunk_sizes=(1, 3, 7),
                    python_hash_seeds=(0, 1),
                    verify_all_source_entries=True,
                ),
                resolved,
                deps=PRODUCTION_REPRODUCIBILITY_DEPENDENCIES,
            )
    finally:
        monkeypatch.chdir(repository)
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=repository,
            check=True,
            capture_output=True,
        )
