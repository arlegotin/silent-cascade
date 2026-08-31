"""Task 14 contracts for private validation-manifest construction."""

from contextlib import contextmanager
from dataclasses import replace
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


def test_inspection_request_requires_exactly_one_selector() -> None:
    """Changing selector validation would allow an ambiguous private inspection."""
    from silent_cascade.env.services import ConfigSelection, InspectEpisodeRequest

    with pytest.raises(ValueError, match="select exactly one"):
        InspectEpisodeRequest(config=ConfigSelection(), manifest_path=Path("manifest.json"))


def test_freeze_with_test_dependencies_publishes_immutable_report(tmp_path: Path) -> None:
    """Dropping the report-only publication boundary would make reuse mutate evidence."""
    from silent_cascade.env.services import (
        ConfigSelection,
        FreezeValidationRequest,
        Phase1ServiceDependencies,
        freeze_validation,
    )

    allocation = _allocation()

    def collect(*args: object, **kwargs: object) -> EvidenceProvenance:
        del kwargs
        return _provenance().model_copy(update={"config_sha256": args[0].sha256})  # type: ignore[attr-defined]

    built: list[object] = []

    def build(
        config: Phase1Config, provenance: EvidenceProvenance, root_seed: int, public_id_seed: int
    ) -> object:
        from silent_cascade.env.services import build_cohort_manifest

        manifest = build_cohort_manifest(
            config,
            allocation,
            provenance,
            root_seed,
            public_id_seed,
            access_class=ManifestAccessClass.DEBUG,
        )
        built.append(manifest)
        return manifest

    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=allocation,
        collect_provenance=collect,
        build_manifest=build,
    )
    request = FreezeValidationRequest(
        config=ConfigSelection(),
        output_path=tmp_path / "freeze.json",
        episode_count=8,
        root_seed=41,
        public_id_seed=91,
    )
    first = freeze_validation(request, deps=deps)
    saved = request.output_path.read_bytes()
    second = freeze_validation(request, deps=deps)

    assert first.report.episode_count == 8
    assert first.publication.created is True
    assert second.publication.created is False
    assert request.output_path.read_bytes() == saved
    assert first.report.provenance.foundation_model_calls == 0
    assert b'"created"' not in saved
    assert all(
        entry.episode_public_id.encode() not in saved
        for entry in built[0].entries  # type: ignore[attr-defined]
    )
    divergent = request.model_copy(
        update={
            "config": ConfigSelection(
                set_overrides=("data.leakage_audit.test.permutation_replicates=1",)
            )
        }
    )
    with pytest.raises(ValueError, match="different immutable report"):
        freeze_validation(divergent, deps=deps)
    assert request.output_path.read_bytes() == saved


@pytest.mark.parametrize(
    "output",
    (
        Path("manifests/validation/report.json"),
        Path("manifests/validation/nested/report.json"),
        Path.cwd() / "manifests/validation/absolute/report.json",
        Path("manifests/debug/../validation/traversal/report.json"),
    ),
)
def test_test_dependencies_refuse_canonical_validation_tree(output: Path) -> None:
    """Changing ancestor checking would let DEBUG evidence impersonate validation output."""
    from silent_cascade.env.services import _require_test_output_boundary

    with pytest.raises(ValueError, match="manifests/validation"):
        _require_test_output_boundary(output, type("Deps", (), {"production_mode": False})())


@pytest.mark.parametrize(
    "output",
    (
        Path("manifests/validation-copy/report.json"),
        Path("manifests/validation/../debug/report.json"),
    ),
)
def test_test_dependencies_allow_paths_outside_canonical_validation_tree(output: Path) -> None:
    """Prefix or pre-normalization checks would reject legitimate DEBUG output."""
    from silent_cascade.env.services import _require_test_output_boundary

    _require_test_output_boundary(output, type("Deps", (), {"production_mode": False})())


def test_service_dependency_identity_and_test_size_bounds_are_sealed() -> None:
    """Trusting a production flag or unbounded fixture would cross the service boundary."""
    from silent_cascade.env.services import (
        PRODUCTION_DEPENDENCIES,
        Phase1ServiceDependencies,
        _require_service_dependencies,
    )

    with pytest.raises(ValueError, match="sealed"):
        _require_service_dependencies(replace(PRODUCTION_DEPENDENCIES))
    valid = Phase1ServiceDependencies.for_test(
        validation_allocation=_allocation(),
        collect_provenance=lambda *args, **kwargs: _provenance(),
        build_manifest=lambda *args: pytest.fail("not used"),
    )
    too_small = replace(valid, validation_allocation=_sized_allocation(4))
    too_large = replace(valid, validation_allocation=_sized_allocation(20))
    for invalid in (too_small, too_large):
        with pytest.raises(ValueError, match="8-16"):
            _require_service_dependencies(invalid)


def test_request_models_reject_unknown_source_modes_and_ambiguous_selectors() -> None:
    """Permissive request parsing would let invalid mode combinations reach generation."""
    from silent_cascade.env.services import ConfigSelection, OracleEvaluationRequest

    with pytest.raises(ValueError):
        OracleEvaluationRequest.model_validate(
            {"config": ConfigSelection(), "source": {"mode": "unknown"}}
        )


def test_debug_inspection_can_include_oracle_without_private_truth(tmp_path: Path) -> None:
    """Changing the public projection would leak private recipe fields to inspection."""
    from silent_cascade.env.services import (
        ConfigSelection,
        InspectEpisodeRequest,
        Phase1ServiceDependencies,
        inspect_episode,
    )
    from silent_cascade.logging.manifest import publish_manifest

    allocation = _allocation()

    def collect(*args: object, **kwargs: object) -> EvidenceProvenance:
        del kwargs
        return _provenance().model_copy(update={"config_sha256": args[0].sha256})  # type: ignore[attr-defined]

    def build(
        config: Phase1Config, provenance: EvidenceProvenance, root_seed: int, public_id_seed: int
    ) -> object:
        from silent_cascade.env.services import build_cohort_manifest

        return build_cohort_manifest(
            config,
            allocation,
            provenance,
            root_seed,
            public_id_seed,
            access_class=ManifestAccessClass.DEBUG,
        )

    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=allocation, collect_provenance=collect, build_manifest=build
    )
    resolved = resolve_config(
        Phase1Config, [Path("configs/base.yaml"), Path("configs/data/primary.yaml")]
    )
    manifest = build(resolved.config, collect(resolved), 41, 91)
    path = tmp_path / "debug-manifest.json"
    publish_manifest(path, manifest)  # type: ignore[arg-type]

    report = inspect_episode(
        InspectEpisodeRequest(
            config=ConfigSelection(), manifest_path=path, entry_index=0, include_oracle=True
        ),
        deps=deps,
    )

    assert report.oracle is not None
    assert "truth" not in report.model_dump_json()


def test_oracle_evaluation_streams_bound_test_allocation(tmp_path: Path) -> None:
    """Removing per-episode verification would let a corrupt generated bundle enter evidence."""
    from silent_cascade.env.generator import EpisodeBlock, IndependentAllocation
    from silent_cascade.env.services import (
        ConfigSelection,
        OracleEvaluationRequest,
        Phase1GateCorpusSource,
        Phase1ServiceDependencies,
        evaluate_oracle,
    )

    allocation = _allocation()
    independent = IndependentAllocation(
        allocation_id="test-independent-v1",
        split_namespace=SplitNamespace.DEBUG,
        blocks=tuple(
            EpisodeBlock(
                suite=SuiteName.IID_PRIMARY,
                requested_path_length=path,
                first_episode_index=(path - 2) * 4,
                episode_count=4,
            )
            for path in (2, 3, 4)
        ),
    )

    def collect(*args: object, **kwargs: object) -> EvidenceProvenance:
        del kwargs
        resolved = args[0]
        return _provenance(allocation_id=independent.allocation_id).model_copy(
            update={
                "generation_mode": "independent",
                "config_sha256": resolved.sha256,  # type: ignore[attr-defined]
            }
        )

    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=allocation,
        independent_allocation=independent,
        collect_provenance=collect,
        build_manifest=lambda *args: pytest.fail("not used"),
    )
    result = evaluate_oracle(
        OracleEvaluationRequest(
            config=ConfigSelection(),
            source=Phase1GateCorpusSource(
                allocation_id=independent.allocation_id, root_seed=41, public_id_seed=91
            ),
            output_path=tmp_path / "oracle.json",
        ),
        deps=deps,
    )

    assert result.report.verified_episode_count == 12
    assert result.report.oracle_successes == 12
    assert result.report.corpus_sha256


def test_manifest_oracle_accepts_real_authenticated_clock_children(tmp_path: Path) -> None:
    """Treating a paired child as an unscaled draw would reject valid manifest evidence."""
    from silent_cascade.env.services import (
        ConfigSelection,
        ManifestCorpusSource,
        OracleEvaluationRequest,
        Phase1ServiceDependencies,
        evaluate_oracle,
    )
    from silent_cascade.logging.manifest import publish_manifest

    manifest, _parents = _clock_manifest()
    path = tmp_path / "clock-manifest.json"
    output = tmp_path / "oracle.json"
    publish_manifest(path, manifest)

    def collect(*args: object, **kwargs: object) -> EvidenceProvenance:
        del kwargs
        resolved = args[0]
        return manifest.provenance.model_copy(
            update={"config_sha256": resolved.sha256}  # type: ignore[attr-defined]
        )

    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=_allocation(),
        independent_allocation=_independent_allocation(8),
        collect_provenance=collect,
        build_manifest=lambda *args: pytest.fail("not used"),
    )

    result = evaluate_oracle(
        OracleEvaluationRequest(
            config=ConfigSelection(),
            source=ManifestCorpusSource(manifest_path=path),
            output_path=output,
        ),
        deps=deps,
    )

    assert result.report.clock_0_1x_episode_count == 8
    assert result.report.clock_10x_episode_count == 0
    assert result.report.clock_decision_mismatches == 0


def test_manifest_oracle_refuses_a_different_valid_clock_transform_before_publication(
    tmp_path: Path,
) -> None:
    """Removing the independent clock pass would trust the corpus regenerator twice."""
    from silent_cascade.env.episode import scale_episode_time
    from silent_cascade.env.services import (
        ConfigSelection,
        ManifestCorpusSource,
        OracleEvaluationRequest,
        Phase1ServiceDependencies,
        evaluate_oracle,
        regenerate_entry,
    )
    from silent_cascade.logging.manifest import publish_manifest

    manifest, parents = _clock_manifest()
    path = tmp_path / "clock-manifest.json"
    output = tmp_path / "oracle.json"
    publish_manifest(path, manifest)
    calls: dict[str, int] = {}

    def regenerate(config: Phase1Config, current: object, entry: object) -> object:
        public_id = entry.episode_public_id  # type: ignore[attr-defined]
        calls[public_id] = calls.get(public_id, 0) + 1
        if calls[public_id] == 2 and entry == manifest.entries[0]:
            return scale_episode_time(
                parents[1],
                entry.suite,
                entry.episode_public_id,  # type: ignore[attr-defined]
            )
        return regenerate_entry(config, current, entry)  # type: ignore[arg-type]

    def collect(*args: object, **kwargs: object) -> EvidenceProvenance:
        del kwargs
        resolved = args[0]
        return manifest.provenance.model_copy(
            update={"config_sha256": resolved.sha256}  # type: ignore[attr-defined]
        )

    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=_allocation(),
        independent_allocation=_independent_allocation(8),
        collect_provenance=collect,
        build_manifest=lambda *args: pytest.fail("not used"),
        regenerate_manifest_entry=regenerate,
    )

    with pytest.raises(ValueError, match="clock"):
        evaluate_oracle(
            OracleEvaluationRequest(
                config=ConfigSelection(),
                source=ManifestCorpusSource(manifest_path=path),
                output_path=output,
            ),
            deps=deps,
        )
    assert not output.exists()


def test_leakage_anchor_authority_stays_pinned_when_source_authentication_is_forged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Copying source authentication would let one compromised capability endorse itself."""
    import silent_cascade.env.services as services
    from silent_cascade.env.leakage import (
        LeakageAuditProfileName,
        audit_source_descriptor_sha256,
    )
    from silent_cascade.env.services import (
        ConfigSelection,
        LeakageAuditRequest,
        Phase1GateCorpusSource,
        Phase1ServiceDependencies,
        _bind_independent_audit_source,
        run_leakage_audit,
    )
    from silent_cascade.provenance import LeakageAuditEvidenceAnchor

    independent = _audit_independent_allocation()
    holder: dict[str, Phase1ServiceDependencies] = {}

    def collect(*args: object, **kwargs: object) -> EvidenceProvenance:
        resolved = args[0]
        return _provenance(allocation_id=independent.allocation_id).model_copy(
            update={
                "generation_mode": "independent",
                "config_sha256": resolved.sha256,  # type: ignore[attr-defined]
                "root_seed": kwargs["root_seed"],
                "public_id_seed_sha256": public_id_seed_sha256(kwargs["public_id_seed"]),
                "analysis_seeds": kwargs["analysis_seeds"],
            }
        )

    def clean_source(resolved: object, raw_source: object, provenance: object):
        return _bind_independent_audit_source(
            resolved,
            raw_source,
            provenance,
            holder["deps"],  # type: ignore[arg-type]
        )

    def anchor_authority(
        resolved: object, raw_source: object, provenance: object, profile: object
    ) -> LeakageAuditEvidenceAnchor:
        bound = clean_source(resolved, raw_source, provenance)
        descriptor = bound.descriptor
        authentication = bound.authentication
        return LeakageAuditEvidenceAnchor(
            schema_version="phase1-leakage-audit-anchor-v1",
            profile=profile.value,  # type: ignore[attr-defined]
            allocation_id=descriptor.allocation_id,
            allocation_or_manifest_sha256=descriptor.allocation_or_manifest_sha256,
            config_sha256=descriptor.config_sha256,
            descriptor_sha256=audit_source_descriptor_sha256(descriptor),
            source_manifest_sha256=authentication.source_manifest_sha256,
            suite_path_denominators=authentication.suite_path_denominators,
            clock_pair_manifest_sha256=authentication.clock_pair_manifest_sha256,
            clock_scale_pair_counts=authentication.clock_scale_pair_counts,
            episode_count=descriptor.episode_count,
        )

    def forged_source(resolved: object, raw_source: object, provenance: object, profile: object):
        del profile
        bound = clean_source(resolved, raw_source, provenance)
        alternate_raw = Phase1GateCorpusSource(
            allocation_id=independent.allocation_id,
            root_seed=42,
            public_id_seed=92,
        )
        alternate_provenance = collect(
            resolved,
            root_seed=42,
            public_id_seed=92,
            analysis_seeds=provenance.analysis_seeds,  # type: ignore[attr-defined]
        )
        alternate = clean_source(resolved, alternate_raw, alternate_provenance)
        return replace(bound, authentication=alternate.authentication)

    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=_allocation(),
        independent_allocation=independent,
        collect_provenance=collect,
        build_manifest=lambda *args: pytest.fail("not used"),
        build_audit_source=forged_source,
        build_audit_anchor=anchor_authority,
    )
    holder["deps"] = deps

    def observe_anchor(
        source: object,
        config: object,
        profile: object,
        provenance: object,
        path: Path,
    ):
        del config, profile, path
        assert provenance.leakage_audit is not None  # type: ignore[attr-defined]
        assert (
            provenance.leakage_audit.source_manifest_sha256  # type: ignore[attr-defined]
            != source.authentication.source_manifest_sha256  # type: ignore[attr-defined]
        )
        raise RuntimeError("anchor observed")

    real_audit_leakage = services.audit_leakage
    monkeypatch.setattr(services, "audit_leakage", observe_anchor)
    with pytest.raises(RuntimeError, match="anchor observed"):
        run_leakage_audit(
            LeakageAuditRequest(
                config=ConfigSelection(),
                source=Phase1GateCorpusSource(
                    allocation_id=independent.allocation_id,
                    root_seed=41,
                    public_id_seed=91,
                ),
                profile=LeakageAuditProfileName.TEST,
            ),
            deps=deps,
        )
    monkeypatch.setattr(services, "audit_leakage", real_audit_leakage)
    output = tmp_path / "forged-source-must-not-exist.json"
    with pytest.raises(ValueError, match=r"authentication|anchor"):
        run_leakage_audit(
            LeakageAuditRequest(
                config=ConfigSelection(),
                source=Phase1GateCorpusSource(
                    allocation_id=independent.allocation_id,
                    root_seed=41,
                    public_id_seed=91,
                ),
                profile=LeakageAuditProfileName.TEST,
                output_path=output,
            ),
            deps=deps,
        )
    assert not output.exists()


def test_manifest_audit_binds_current_execution_separately_from_embedded_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requiring current provenance equality would make every immutable manifest stale."""
    import silent_cascade.env.services as services
    from silent_cascade.env.leakage import LeakageAuditProfileName
    from silent_cascade.env.services import (
        ManifestCorpusSource,
        Phase1ServiceDependencies,
        _production_audit_source,
    )
    from silent_cascade.logging.manifest import publish_manifest

    manifest = _matched_manifest()
    path = tmp_path / "matched-manifest.json"
    publish_manifest(path, manifest)
    current = manifest.provenance.model_copy(
        update={
            "source_commit": "f" * 40,
            "analysis_seeds": {
                "audit_seed": 2026083091,
                "positive_control_seed": 2026083092,
            },
        }
    )
    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=_audit_validation_allocation(),
        independent_allocation=_audit_independent_allocation(),
        collect_provenance=lambda *args, **kwargs: current,
        build_manifest=lambda *args: pytest.fail("not used"),
    )
    monkeypatch.setattr(services, "PRODUCTION_DEPENDENCIES", deps)

    source = _production_audit_source(
        resolve_config(
            Phase1Config,
            [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
        ),
        ManifestCorpusSource(manifest_path=path),
        current,
        LeakageAuditProfileName.TEST,
    )

    assert source.descriptor.generation_mode == "matched"
    assert source.authentication.clock_scale_pair_counts["scale_0_1x"] > 0
    assert source.authentication.clock_scale_pair_counts["scale_10x"] > 0


@pytest.mark.parametrize("source_mode", ("manifest", "phase1_gate"))
@pytest.mark.parametrize("scientific_failure", (False, True))
def test_leakage_service_executes_task15_and_publishes_complete_scientific_results(
    source_mode: str,
    scientific_failure: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bypassing Task 15 would miss typed counterfactuals, two passes, and inner cleanup."""
    import numpy as np

    import silent_cascade.env.leakage as leakage
    from silent_cascade.env.leakage import (
        CounterfactualCheckId,
        LeakageAuditProfileName,
        ShortcutFeatureGroup,
        ShortcutProbeResult,
        ShortcutTask,
    )
    from silent_cascade.env.services import (
        ConfigSelection,
        LeakageAuditRequest,
        ManifestCorpusSource,
        Phase1GateCorpusSource,
        Phase1ServiceDependencies,
        _anchor_for_bound_source,
        _bind_independent_audit_source,
        _bind_manifest_audit_source,
        evaluate_oracle,
        run_leakage_audit,
    )
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
    from silent_cascade.logging.manifest import load_manifest, publish_manifest

    config_selection = ConfigSelection(
        set_overrides=(
            "data.leakage_audit.test.episode_count=12",
            "data.leakage_audit.test.permutation_replicates=1",
            "data.leakage_audit.test.minimum_test_examples_per_class=1",
        )
    )
    independent = _audit_independent_allocation()
    manifest = _matched_manifest(
        resolve_config(
            Phase1Config,
            [config_selection.base_path, config_selection.data_path],
            set_overrides=config_selection.set_overrides,
        )
    )
    manifest_path = tmp_path / "matched-manifest.json"
    publish_manifest(manifest_path, manifest)
    holder: dict[str, Phase1ServiceDependencies] = {}

    def collect(*args: object, **kwargs: object) -> EvidenceProvenance:
        resolved = args[0]
        return _provenance(allocation_id=kwargs["allocation_id"]).model_copy(
            update={
                "generation_mode": kwargs["generation_mode"],
                "config_sha256": resolved.sha256,  # type: ignore[attr-defined]
                "root_seed": kwargs["root_seed"],
                "public_id_seed_sha256": public_id_seed_sha256(kwargs["public_id_seed"]),
                "analysis_seeds": kwargs["analysis_seeds"],
            }
        )

    def build_source(resolved: object, raw_source: object, provenance: object, profile: object):
        del profile
        if isinstance(raw_source, ManifestCorpusSource):
            return _bind_manifest_audit_source(
                resolved,
                load_manifest(raw_source.manifest_path),
                holder["deps"],  # type: ignore[arg-type]
            )
        return _bind_independent_audit_source(
            resolved,
            raw_source,
            provenance,
            holder["deps"],  # type: ignore[arg-type]
        )

    def build_anchor(resolved: object, raw_source: object, provenance: object, profile: object):
        return _anchor_for_bound_source(
            build_source(resolved, raw_source, provenance, profile),
            profile,  # type: ignore[arg-type]
        )

    workspace = tmp_path / f"workspace-{source_mode}-{scientific_failure}"
    cleanup_observed: list[tuple[Path, ...]] = []

    @contextmanager
    def create_workspace():
        workspace.mkdir()
        yield workspace
        cleanup_observed.append(tuple(workspace.iterdir()))

    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=_audit_validation_allocation(),
        independent_allocation=independent,
        collect_provenance=collect,
        build_manifest=lambda *args: pytest.fail("not used"),
        build_audit_source=build_source,
        build_audit_anchor=build_anchor,
        create_audit_workspace=create_workspace,
    )
    holder["deps"] = deps

    def small_split(*args: object, **kwargs: object):
        del args, kwargs
        return (
            np.arange(8, dtype=np.int64),
            np.arange(8, 12, dtype=np.int64),
            sha256_bytes(canonical_json_bytes({"test": "small-task16-split"})),
        )

    failed_probe = ShortcutProbeResult(
        task=ShortcutTask.POSITIVE_BINARY,
        feature_group=ShortcutFeatureGroup.ID_POSITION,
        feature_dimension=17,
        train_examples=8,
        test_examples=4,
        train_class_counts={"0": 4, "1": 4},
        test_class_counts={"0": 2, "1": 2},
        raw_accuracy=1.0,
        balanced_accuracy=1.0,
        balanced_chance=0.5,
        raw_permutation_p=0.001,
        holm_adjusted_p=0.001,
        optimizer_iterations=1,
        optimizer_converged=True,
        passed=False,
    )

    def small_probes(*args: object, **kwargs: object):
        del args
        return [failed_probe] if scientific_failure and kwargs.get("label_shuffled") else []

    monkeypatch.setattr(leakage, "_split_memberships", small_split)
    monkeypatch.setattr(leakage, "_run_probes", small_probes)
    output = tmp_path / f"leakage-{source_mode}-{scientific_failure}.json"
    source = (
        ManifestCorpusSource(manifest_path=manifest_path)
        if source_mode == "manifest"
        else Phase1GateCorpusSource(
            allocation_id=independent.allocation_id,
            root_seed=41,
            public_id_seed=91,
        )
    )
    result = run_leakage_audit(
        LeakageAuditRequest(
            config=config_selection,
            source=source,
            profile=LeakageAuditProfileName.TEST,
            output_path=output,
        ),
        deps=deps,
    )

    assert result.report.passed is (not scientific_failure)
    assert tuple(item.check_id for item in result.report.counterfactual_checks) == tuple(
        CounterfactualCheckId
    )
    assert all(item.passed for item in result.report.counterfactual_checks)
    assert result.report.provenance.analysis_seeds == {
        "audit_seed": 2026083091,
        "positive_control_seed": 2026083092,
    }
    assert result.publication is not None and result.publication.created
    assert output.exists()
    assert cleanup_observed == [()]
    if source_mode == "phase1_gate" and not scientific_failure:
        from silent_cascade.env.generator import (
            generate_independent_episode,
            iter_independent_requests,
        )
        from silent_cascade.env.reproducibility import (
            IndependentAllocationReproducibilitySource,
            ReproducibilityDependencies,
            ReproducibilityRequest,
            check_reproducibility,
        )
        from silent_cascade.env.services import OracleEvaluationRequest

        oracle = evaluate_oracle(
            OracleEvaluationRequest(config=config_selection, source=source),
            deps=deps,
        )
        resolved = resolve_config(
            Phase1Config,
            [config_selection.base_path, config_selection.data_path],
            set_overrides=config_selection.set_overrides,
        )
        reproducibility_provenance = collect(
            resolved,
            generation_mode="independent",
            allocation_id=independent.allocation_id,
            split_namespace=independent.split_namespace,
            root_seed=41,
            public_id_seed=91,
            analysis_seeds={},
        )
        reproducibility = check_reproducibility(
            ReproducibilityRequest(
                source=IndependentAllocationReproducibilitySource(
                    allocation_id=independent.allocation_id,
                    root_seed=41,
                    public_id_seed=91,
                ),
                sample_size=12,
                chunk_sizes=(1, 3, 7),
                python_hash_seeds=(0, 1),
                verify_all_source_entries=True,
            ),
            resolved,
            deps=ReproducibilityDependencies.for_test(
                independent,
                reproducibility_provenance,
                collect_provenance=collect,
                load_verified_manifest=load_manifest,
                iter_independent=iter_independent_requests,
                generate_independent=generate_independent_episode,
            ),
        )
        assert (
            oracle.report.corpus_sha256
            == result.report.corpus_hash
            == reproducibility.reference_corpus_sha256
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


def _sized_allocation(episode_count: int) -> CohortAllocation:
    return CohortAllocation(
        allocation_id=f"test-sized-{episode_count}",
        split_namespace=SplitNamespace.DEBUG,
        blocks=(
            CohortBlock(
                suite=SuiteName.IID_PRIMARY,
                requested_path_length=2,
                first_cohort_index=0,
                cohort_count=episode_count // 4,
            ),
        ),
    )


def _audit_validation_allocation() -> CohortAllocation:
    return CohortAllocation(
        allocation_id="test-validation-audit-v1",
        split_namespace=SplitNamespace.DEBUG,
        blocks=tuple(
            CohortBlock(
                suite=SuiteName.IID_PRIMARY,
                requested_path_length=path,
                first_cohort_index=path - 2,
                cohort_count=1,
            )
            for path in (2, 3, 4)
        ),
    )


def _matched_manifest(resolved=None):
    from silent_cascade.env.services import build_cohort_manifest

    if resolved is None:
        resolved = resolve_config(
            Phase1Config, [Path("configs/base.yaml"), Path("configs/data/primary.yaml")]
        )
    allocation = _audit_validation_allocation()
    provenance = _provenance(allocation_id=allocation.allocation_id).model_copy(
        update={"config_sha256": resolved.sha256}
    )
    return build_cohort_manifest(
        resolved.config,
        allocation,
        provenance,
        41,
        91,
        access_class=ManifestAccessClass.DEBUG,
    )


def _independent_allocation(episode_count: int = 12):
    from silent_cascade.env.generator import EpisodeBlock, IndependentAllocation

    blocks = tuple(
        EpisodeBlock(
            suite=SuiteName.IID_PRIMARY,
            requested_path_length=path,
            first_episode_index=(path - 2) * 4,
            episode_count=4,
        )
        for path in (2, 3, 4)
    )
    return IndependentAllocation(
        allocation_id="test-independent-v1",
        split_namespace=SplitNamespace.DEBUG,
        blocks=blocks[: episode_count // 4],
    )


def _audit_independent_allocation():
    from silent_cascade.env.generator import ClockEpisodeBlock, IndependentAllocation

    allocation = _independent_allocation()
    return IndependentAllocation(
        allocation_id=allocation.allocation_id,
        split_namespace=allocation.split_namespace,
        blocks=allocation.blocks,
        clock_blocks=(
            ClockEpisodeBlock(
                requested_path_length=2,
                source_first_episode_index=0,
                scale_0_1x_episode_count=1,
                scale_10x_episode_count=1,
            ),
        ),
    )


def _clock_manifest():
    from silent_cascade.env.episode import episode_sha256, scale_episode_time
    from silent_cascade.env.generator import generate_independent_episode, iter_independent_requests
    from silent_cascade.logging.manifest import (
        EpisodeManifest,
        EpisodeManifestEntry,
        IndependentManifestCoordinate,
    )
    from silent_cascade.rng import IndependentPublicIdKey, allocate_independent_public_id

    allocation = _independent_allocation(8)
    resolved = resolve_config(
        Phase1Config, [Path("configs/base.yaml"), Path("configs/data/primary.yaml")]
    )
    provenance = _provenance(allocation_id="test-clock-v1").model_copy(
        update={"generation_mode": "independent", "config_sha256": resolved.sha256}
    )
    parents = tuple(
        generate_independent_episode(resolved.config, request, 91)
        for request in iter_independent_requests(allocation, 41)
    )
    entries = []
    for request, parent in zip(iter_independent_requests(allocation, 41), parents, strict=True):
        child_id = allocate_independent_public_id(
            IndependentPublicIdKey(
                generator_version="ofd-v1",
                split_namespace=SplitNamespace.DEBUG,
                suite=SuiteName.CLOCK_SCALE_0_1X,
                public_id_seed=91,
                episode_index=request.episode_index,
                accepted_attempt=parent.truth.recipe.accepted_attempt,
            )
        )
        child = scale_episode_time(parent, SuiteName.CLOCK_SCALE_0_1X, child_id)
        entries.append(
            EpisodeManifestEntry(
                episode_public_id=child_id,
                split_namespace=SplitNamespace.DEBUG,
                suite=SuiteName.CLOCK_SCALE_0_1X,
                coordinate=IndependentManifestCoordinate(
                    episode_index=request.episode_index,
                    allocation_quartet_index=request.allocation_quartet_index,
                    quartet_member_index=request.episode_index % 4,
                ),
                requested_path_length=request.requested_path_length,
                accepted_attempt=parent.truth.recipe.accepted_attempt,
                episode_sha256=episode_sha256(child),
                parent_public_id=parent.public.init.episode_public_id,
                parent_episode_sha256=episode_sha256(parent),
                clock_scale=0.1,
            )
        )
    return (
        EpisodeManifest(
            schema_version=1,
            experiment_version="v1",
            access_class=ManifestAccessClass.DEBUG,
            provenance=provenance,
            suite=SuiteName.CLOCK_SCALE_0_1X,
            public_id_seed=91,
            episode_count=len(entries),
            entries=tuple(entries),
        ),
        parents,
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
