"""Task 14 contracts for private validation-manifest construction."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.generator import CohortAllocation, CohortBlock
from silent_cascade.errors import (
    ArtifactError,
    ConfigurationError,
    EpisodeError,
    ManifestAccessError,
    ManifestError,
    ProvenanceError,
)
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
    from uuid import UUID

    from silent_cascade.env.services import ConfigSelection, InspectEpisodeRequest

    with pytest.raises(ValueError, match="select exactly one"):
        InspectEpisodeRequest(config=ConfigSelection(), manifest_path=Path("manifest.json"))

    with pytest.raises(ValueError, match="select exactly one"):
        InspectEpisodeRequest(
            config=ConfigSelection(),
            manifest_path=Path("manifest.json"),
            episode_public_id=UUID("00000000-0000-4000-8000-000000000001"),
            entry_index=0,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("successes", -1),
        ("successes", 9),
        ("total", 0),
        ("observed_rate", 0.0),
        ("expected_rate", 0.5),
        ("absolute_error", 0.5),
        ("exact_binomial_p", 0.0),
        ("passed", False),
    ),
)
def test_exact_random_check_rejects_every_underived_field(
    field: str,
    replacement: object,
) -> None:
    """A report cannot publish hand-written random diagnostics or pass state."""
    from scipy.stats import binomtest

    from silent_cascade.env.services import ExactRandomCheck

    payload = {
        "successes": 1,
        "total": 8,
        "observed_rate": 0.125,
        "expected_rate": 0.125,
        "absolute_error": 0.0,
        "exact_binomial_p": float(binomtest(1, 8, 0.125).pvalue),
        "passed": True,
    }
    payload[field] = replacement

    with pytest.raises(ValidationError):
        ExactRandomCheck.model_validate(payload)


def test_freeze_with_test_dependencies_publishes_an_immutable_loadable_manifest(
    tmp_path: Path,
) -> None:
    """Publishing the summary instead would break every downstream manifest consumer."""
    from silent_cascade.env.services import (
        ConfigSelection,
        FreezeValidationRequest,
        Phase1ServiceDependencies,
        freeze_validation,
    )
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes, sha256_file
    from silent_cascade.logging.manifest import load_manifest

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
    saved_mtime_ns = request.output_path.stat().st_mtime_ns
    loaded = load_manifest(request.output_path)
    second = freeze_validation(request, deps=deps)

    assert first.report.episode_count == 8
    assert first.publication.created is True
    assert second.publication.created is False
    assert request.output_path.read_bytes() == saved
    assert request.output_path.stat().st_mtime_ns == saved_mtime_ns
    assert loaded == built[0]
    assert first.report.manifest_payload_sha256 == sha256_bytes(canonical_json_bytes(loaded))
    assert first.publication.payload_sha256 == first.report.manifest_payload_sha256
    assert first.publication.file_sha256 == sha256_file(request.output_path)
    assert first.report.provenance.foundation_model_calls == 0
    assert all(
        entry.episode_public_id.encode() in saved
        for entry in built[0].entries  # type: ignore[attr-defined]
    )
    divergent = request.model_copy(
        update={
            "config": ConfigSelection(
                set_overrides=("data.leakage_audit.test.permutation_replicates=1",)
            )
        }
    )
    with pytest.raises(ManifestError, match="different immutable manifest"):
        freeze_validation(divergent, deps=deps)
    assert request.output_path.read_bytes() == saved
    assert request.output_path.stat().st_mtime_ns == saved_mtime_ns

    with pytest.raises(ManifestError, match=r"publication|manifest|target"):
        freeze_validation(request.model_copy(update={"output_path": tmp_path}), deps=deps)


def test_dirty_production_validation_provenance_is_a_typed_refusal() -> None:
    """Dirty evidence is an expected publication refusal, not a raw builder traceback."""
    from silent_cascade.env.services import build_validation_manifest

    provenance = _provenance(allocation_id="validation-v1", dirty=True).model_copy(
        update={"split_namespace": SplitNamespace.VALIDATION}
    )

    with pytest.raises(ProvenanceError, match=r"clean|dirty|provenance"):
        build_validation_manifest(_config(), provenance, 41, 91)


def test_report_publication_directory_is_a_typed_artifact_refusal(tmp_path: Path) -> None:
    """An existing directory target must not escape as IsADirectoryError."""
    from silent_cascade.env.services import ConfigSelection, _publish_report

    with pytest.raises(ArtifactError, match=r"publication|report|target"):
        _publish_report(tmp_path, ConfigSelection())


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

    with pytest.raises(ManifestAccessError, match="manifests/validation"):
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


def test_default_freeze_selects_the_sealed_production_adapter_without_running_task18(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The 10,000 adapter identity is covered without executing the Task 18 corpus."""
    import silent_cascade.env.services as services
    from silent_cascade.env.generator import VALIDATION_ALLOCATION
    from silent_cascade.env.services import ConfigSelection, FreezeValidationRequest

    observed: list[tuple[Phase1Config, EvidenceProvenance, int, int]] = []

    def collect(*args: object, **kwargs: object) -> EvidenceProvenance:
        resolved = args[0]
        return _provenance(allocation_id="validation-v1").model_copy(
            update={
                "split_namespace": SplitNamespace.VALIDATION,
                "config_sha256": resolved.sha256,  # type: ignore[attr-defined]
                "root_seed": kwargs["root_seed"],
                "public_id_seed_sha256": public_id_seed_sha256(kwargs["public_id_seed"]),
            }
        )

    def stop_at_adapter(
        config: Phase1Config,
        provenance: EvidenceProvenance,
        root_seed: int,
        public_id_seed: int,
    ) -> object:
        observed.append((config, provenance, root_seed, public_id_seed))
        raise RuntimeError("production adapter selected")

    monkeypatch.setattr(
        services,
        "PRODUCTION_DEPENDENCIES",
        replace(
            services.PRODUCTION_DEPENDENCIES,
            validation_allocation=VALIDATION_ALLOCATION,
            collect_provenance=collect,
            build_manifest=stop_at_adapter,
        ),
    )
    with pytest.raises(RuntimeError, match="production adapter selected"):
        services.freeze_validation(
            FreezeValidationRequest(
                config=ConfigSelection(),
                output_path=tmp_path / "production-spy.json",
                episode_count=10_000,
                root_seed=41,
                public_id_seed=91,
            )
        )
    assert len(observed) == 1
    assert observed[0][2:] == (41, 91)


@pytest.mark.parametrize("service", ("oracle", "leakage"))
@pytest.mark.parametrize(
    "manifest_kind",
    ("debug_test", "wrong_identity", "short_validation"),
)
def test_default_production_manifest_services_refuse_noncanonical_sources_before_work(
    service: str,
    manifest_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Removing the service-entry guard would let test evidence reach production work."""
    import silent_cascade.env.services as services
    from silent_cascade.env.leakage import LeakageAuditProfileName
    from silent_cascade.env.services import (
        ConfigSelection,
        LeakageAuditRequest,
        ManifestCorpusSource,
        OracleEvaluationRequest,
    )
    from silent_cascade.logging.manifest import publish_manifest

    if manifest_kind == "debug_test":
        manifest = _matched_manifest()
    elif manifest_kind == "wrong_identity":
        debug = _matched_manifest()
        manifest = debug.model_copy(
            update={
                "access_class": ManifestAccessClass.FIXTURE,
                "provenance": debug.provenance.model_copy(
                    update={"allocation_id": "wrong-production-v1"}
                ),
            }
        )
    else:
        manifest = _small_access_manifest(
            SplitNamespace.VALIDATION,
            ManifestAccessClass.VALIDATION,
            "validation-v1",
            SuiteName.VALIDATION,
        )
    manifest_path = tmp_path / f"{manifest_kind}.json"
    output = tmp_path / f"{service}-must-not-exist.json"
    publish_manifest(manifest_path, manifest)

    def forbidden_regeneration(*args: object, **kwargs: object):
        del args, kwargs
        raise RuntimeError("production regeneration reached")

    monkeypatch.setattr(
        services,
        "PRODUCTION_DEPENDENCIES",
        replace(
            services.PRODUCTION_DEPENDENCIES,
            collect_provenance=_manifest_current_collector(manifest),
            regenerate_manifest_entry=forbidden_regeneration,
        ),
    )

    if service == "oracle":
        with pytest.raises(ManifestAccessError, match=r"production|canonical|manifest"):
            services.evaluate_oracle(
                OracleEvaluationRequest(
                    config=ConfigSelection(),
                    source=ManifestCorpusSource(manifest_path=manifest_path),
                    output_path=output,
                )
            )
    else:
        monkeypatch.setattr(
            services,
            "audit_leakage",
            lambda *args, **kwargs: pytest.fail("Task 15 must not run"),
        )
        with pytest.raises(ManifestAccessError, match=r"production|canonical|manifest"):
            services.run_leakage_audit(
                LeakageAuditRequest(
                    config=ConfigSelection(),
                    source=ManifestCorpusSource(manifest_path=manifest_path),
                    profile=LeakageAuditProfileName.TEST,
                    output_path=output,
                )
            )
    assert not output.exists()


@pytest.mark.parametrize("service", ("oracle", "leakage"))
@pytest.mark.parametrize("manifest_kind", ("cross_allocation", "validation_access"))
def test_injected_manifest_services_refuse_unbound_access_and_allocation(
    service: str,
    manifest_kind: str,
    tmp_path: Path,
) -> None:
    """Injected DEBUG authority is limited to its exact bound allocation."""
    from silent_cascade.env.leakage import LeakageAuditProfileName
    from silent_cascade.env.services import (
        ConfigSelection,
        LeakageAuditRequest,
        ManifestCorpusSource,
        OracleEvaluationRequest,
        Phase1ServiceDependencies,
        evaluate_oracle,
        run_leakage_audit,
    )
    from silent_cascade.logging.manifest import publish_manifest

    manifest = (
        _matched_manifest()
        if manifest_kind == "cross_allocation"
        else _small_access_manifest(
            SplitNamespace.VALIDATION,
            ManifestAccessClass.VALIDATION,
            "validation-v1",
            SuiteName.VALIDATION,
        )
    )
    path = tmp_path / f"{manifest_kind}.json"
    output = tmp_path / f"{service}-must-not-exist.json"
    publish_manifest(path, manifest)

    def forbidden(*args: object, **kwargs: object):
        del args, kwargs
        raise RuntimeError("unbound manifest reached work")

    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=_allocation(),
        independent_allocation=_independent_allocation(8),
        collect_provenance=_manifest_current_collector(manifest),
        build_manifest=lambda *args: pytest.fail("not used"),
        regenerate_manifest_entry=forbidden,
        build_audit_source=forbidden,
        build_audit_anchor=forbidden,
    )
    if service == "oracle":
        with pytest.raises(ManifestAccessError, match=r"test|bound|manifest|DEBUG"):
            evaluate_oracle(
                OracleEvaluationRequest(
                    config=ConfigSelection(),
                    source=ManifestCorpusSource(manifest_path=path),
                    output_path=output,
                ),
                deps=deps,
            )
    else:
        with pytest.raises(ManifestAccessError, match=r"test|bound|manifest|DEBUG"):
            run_leakage_audit(
                LeakageAuditRequest(
                    config=ConfigSelection(),
                    source=ManifestCorpusSource(manifest_path=path),
                    profile=LeakageAuditProfileName.TEST,
                    output_path=output,
                ),
                deps=deps,
            )
    assert not output.exists()


def test_production_manifest_boundary_accepts_the_exact_validation_recipe() -> None:
    """The boundary guard must not turn Task 18's canonical source into a false refusal."""
    from silent_cascade.env.services import (
        PRODUCTION_DEPENDENCIES,
        _require_manifest_source_boundary,
    )

    _require_manifest_source_boundary(
        _canonical_validation_manifest_recipe(),
        PRODUCTION_DEPENDENCIES,
    )


@pytest.mark.parametrize("service", ("oracle", "leakage"))
@pytest.mark.parametrize("refusal", ("config", "provenance"))
def test_manifest_evidence_services_type_routine_config_and_provenance_refusals(
    service: str,
    refusal: str,
    tmp_path: Path,
) -> None:
    """User-selected config and current-source drift are expected service refusals."""
    from silent_cascade.env.leakage import LeakageAuditProfileName
    from silent_cascade.env.services import (
        ConfigSelection,
        LeakageAuditRequest,
        ManifestCorpusSource,
        OracleEvaluationRequest,
        Phase1ServiceDependencies,
        evaluate_oracle,
        run_leakage_audit,
    )
    from silent_cascade.logging.manifest import publish_manifest

    manifest = _matched_manifest()
    path = tmp_path / f"{service}-{refusal}.json"
    publish_manifest(path, manifest)
    selection = (
        ConfigSelection(set_overrides=("data.leakage_audit.test.permutation_replicates=1",))
        if refusal == "config"
        else ConfigSelection()
    )

    def collect(*args: object, **kwargs: object) -> EvidenceProvenance:
        del args, kwargs
        return manifest.provenance.model_copy(
            update={
                "generator_source": manifest.provenance.generator_source.model_copy(
                    update={"sha256": "f" * 64}
                )
            }
        )

    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=_audit_validation_allocation(),
        independent_allocation=_audit_independent_allocation(),
        collect_provenance=(
            collect if refusal == "provenance" else _manifest_current_collector(manifest)
        ),
        build_manifest=lambda *args: pytest.fail("not used"),
        build_audit_source=lambda *args: pytest.fail("audit source work reached"),
        build_audit_anchor=lambda *args: pytest.fail("audit anchor work reached"),
    )
    expected = ConfigurationError if refusal == "config" else ProvenanceError
    source = ManifestCorpusSource(manifest_path=path)

    with pytest.raises(expected, match=r"config|provenance|authenticate"):
        if service == "oracle":
            evaluate_oracle(
                OracleEvaluationRequest(config=selection, source=source),
                deps=deps,
            )
        else:
            run_leakage_audit(
                LeakageAuditRequest(
                    config=selection,
                    source=source,
                    profile=LeakageAuditProfileName.TEST,
                ),
                deps=deps,
            )


def test_manifest_source_change_before_publication_is_a_typed_integrity_refusal(
    tmp_path: Path,
) -> None:
    """Both oracle and leakage rely on one typed manifest recheck boundary."""
    from silent_cascade.env.services import (
        Phase1ServiceDependencies,
        _bound_manifest_sha256,
    )
    from silent_cascade.logging.manifest import publish_manifest

    manifest = _matched_manifest()
    path = tmp_path / "source-changed.json"
    publish_manifest(path, manifest)
    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=_audit_validation_allocation(),
        independent_allocation=_audit_independent_allocation(),
        collect_provenance=_manifest_current_collector(manifest),
        build_manifest=lambda *args: pytest.fail("not used"),
    )

    with pytest.raises(ManifestError, match=r"changed|manifest|source"):
        _bound_manifest_sha256(path, deps, expected_sha256="0" * 64)


def test_production_manifest_test_profile_selects_the_frozen_1200_in_canonical_order() -> None:
    """Auditing all 10,000 rows or sampling partial cohorts would invalidate TEST."""
    from collections import Counter, defaultdict

    from silent_cascade.env.services import (
        PRODUCTION_DEPENDENCIES,
        _manifest_test_profile_entries,
    )
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
    from silent_cascade.logging.manifest import MatchedManifestCoordinate

    manifest = _canonical_validation_manifest_recipe()
    selected = _manifest_test_profile_entries(
        manifest,
        PRODUCTION_DEPENDENCIES,
        audit_seed=2026083091,
    )
    coordinates = []
    members: dict[tuple[int, int], set[int]] = defaultdict(set)
    for entry in selected:
        assert isinstance(entry.coordinate, MatchedManifestCoordinate)
        coordinate = (
            entry.requested_path_length,
            entry.coordinate.cohort_index,
            entry.coordinate.member_index,
        )
        coordinates.append(coordinate)
        members[coordinate[:2]].add(coordinate[2])

    assert len(selected) == 1_200
    assert Counter(entry.requested_path_length for entry in selected) == {
        2: 400,
        3: 400,
        4: 400,
    }
    assert len(members) == 300
    assert all(value == {0, 1, 2, 3} for value in members.values())
    assert coordinates == sorted(coordinates)
    assert (
        sha256_bytes(canonical_json_bytes({"coordinates": coordinates}))
        == "135c962422b96f5b8f43bfee3d25cab994878f87d2dfdfce5225789cba0c5015"
    )


def test_production_manifest_test_source_binds_full_identity_and_subset_witnesses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A self-consistent 1,200-row subset must still authenticate its complete manifest."""
    import silent_cascade.env.services as services
    from silent_cascade.env.leakage import LeakageAuditProfileName
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

    resolved = resolve_config(
        Phase1Config,
        [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
    )
    manifest = _canonical_validation_manifest_recipe()
    debug_manifest = _matched_manifest(resolved)
    template = services.regenerate_entry(
        resolved.config,
        debug_manifest,
        debug_manifest.entries[0],
    )
    deps = replace(
        services.PRODUCTION_DEPENDENCIES,
        regenerate_manifest_entry=lambda *args: template,
    )
    monkeypatch.setattr(services, "_require_manifest_bundle_matches_entry", lambda *args: None)

    source = services._bind_manifest_audit_source(resolved, manifest, deps)
    independent_authority = services._bind_manifest_audit_source(resolved, manifest, deps)
    anchor = services._anchor_for_bound_source(source, LeakageAuditProfileName.TEST)

    assert source.descriptor.episode_count == 1_200
    assert source.descriptor.allocation_or_manifest_sha256 == sha256_bytes(
        canonical_json_bytes(manifest)
    )
    assert source.authentication.suite_path_denominators == {
        "iid_primary:2": 400,
        "iid_primary:3": 400,
        "iid_primary:4": 400,
    }
    assert source.authentication.clock_scale_pair_counts["scale_0_1x"] > 0
    assert source.authentication.clock_scale_pair_counts["scale_10x"] > 0
    assert independent_authority.descriptor == source.descriptor
    assert independent_authority.authentication == source.authentication
    assert anchor.source_manifest_sha256 == source.authentication.source_manifest_sha256
    assert anchor.clock_pair_manifest_sha256 == source.authentication.clock_pair_manifest_sha256
    assert anchor.episode_count == 1_200


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


@pytest.mark.parametrize("refusal", ("config", "index", "uuid"))
def test_inspection_routine_artifact_and_selector_refusals_are_typed(
    refusal: str,
    tmp_path: Path,
) -> None:
    """Routine manifest selection failures must cross the public service seam typed."""
    from uuid import UUID

    from silent_cascade.env.services import (
        ConfigSelection,
        InspectEpisodeRequest,
        Phase1ServiceDependencies,
        inspect_episode,
    )
    from silent_cascade.logging.manifest import publish_manifest

    manifest = _matched_manifest()
    path = tmp_path / "inspect-routine.json"
    publish_manifest(path, manifest)
    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=_audit_validation_allocation(),
        independent_allocation=_audit_independent_allocation(),
        collect_provenance=_manifest_current_collector(manifest),
        build_manifest=lambda *args: pytest.fail("not used"),
    )
    selection = (
        ConfigSelection(set_overrides=("data.leakage_audit.test.permutation_replicates=1",))
        if refusal == "config"
        else ConfigSelection()
    )
    request = InspectEpisodeRequest(
        config=selection,
        manifest_path=path,
        entry_index=99 if refusal in {"config", "index"} else None,
        episode_public_id=(
            UUID("ffffffff-ffff-4fff-bfff-ffffffffffff") if refusal == "uuid" else None
        ),
    )
    error_type = ConfigurationError if refusal == "config" else ManifestAccessError

    with pytest.raises(error_type, match=r"config|index|public ID|manifest"):
        inspect_episode(request, deps=deps)


def test_inspection_regenerated_entry_mismatch_is_a_typed_episode_refusal(
    tmp_path: Path,
) -> None:
    """A manifest recipe mismatch is corrupt artifact evidence, not a programmer ValueError."""
    from silent_cascade.env.services import (
        ConfigSelection,
        InspectEpisodeRequest,
        Phase1ServiceDependencies,
        inspect_episode,
    )
    from silent_cascade.logging.manifest import publish_manifest

    manifest = _matched_manifest()
    changed_entry = manifest.entries[0].model_copy(update={"episode_sha256": "0" * 64})
    changed = manifest.model_copy(update={"entries": (changed_entry, *manifest.entries[1:])})
    path = tmp_path / "inspect-corrupt-entry.json"
    publish_manifest(path, changed)
    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=_audit_validation_allocation(),
        independent_allocation=_audit_independent_allocation(),
        collect_provenance=_manifest_current_collector(changed),
        build_manifest=lambda *args: pytest.fail("not used"),
    )

    with pytest.raises(EpisodeError, match=r"regenerated|manifest entry"):
        inspect_episode(
            InspectEpisodeRequest(config=ConfigSelection(), manifest_path=path, entry_index=0),
            deps=deps,
        )


def test_validation_inspection_is_authorized_but_frozen_access_survives_rename(
    tmp_path: Path,
) -> None:
    """Access authority comes from authenticated content, never a copied filename."""
    from silent_cascade.env.services import (
        ConfigSelection,
        InspectEpisodeRequest,
        Phase1ServiceDependencies,
        inspect_episode,
    )
    from silent_cascade.errors import ManifestAccessError
    from silent_cascade.logging.manifest import publish_manifest

    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=_allocation(),
        independent_allocation=_independent_allocation(8),
        collect_provenance=lambda *args, **kwargs: _provenance(),
        build_manifest=lambda *args: pytest.fail("not used"),
    )
    validation = _small_access_manifest(
        SplitNamespace.VALIDATION,
        ManifestAccessClass.VALIDATION,
        "validation-v1",
        SuiteName.VALIDATION,
    )
    validation_path = tmp_path / "validation-copy.json"
    publish_manifest(validation_path, validation)
    authorized = inspect_episode(
        InspectEpisodeRequest(
            config=ConfigSelection(),
            manifest_path=validation_path,
            entry_index=0,
            include_oracle=True,
        ),
        deps=deps,
    )
    assert authorized.oracle is not None
    assert authorized.access_class is ManifestAccessClass.VALIDATION
    assert "truth" not in authorized.model_dump_json()

    frozen = _small_access_manifest(
        SplitNamespace.FROZEN,
        ManifestAccessClass.FROZEN_TEST,
        "frozen-task16-v1",
        SuiteName.IID_PRIMARY,
    )
    renamed = tmp_path / "harmless-debug-name.json"
    publish_manifest(renamed, frozen)
    public_only = inspect_episode(
        InspectEpisodeRequest(config=ConfigSelection(), manifest_path=renamed, entry_index=0),
        deps=deps,
    )
    serialized = public_only.model_dump_json()
    assert public_only.oracle is None
    assert public_only.access_class is ManifestAccessClass.FROZEN_TEST
    assert all(
        sentinel not in serialized
        for sentinel in (
            "truth",
            "root_seed",
            "public_id_seed",
            "accepted_attempt",
            "rejection",
            "action_target",
        )
    )
    with pytest.raises(ManifestAccessError, match="forbidden"):
        inspect_episode(
            InspectEpisodeRequest(
                config=ConfigSelection(),
                manifest_path=renamed,
                entry_index=0,
                include_oracle=True,
            ),
            deps=deps,
        )


def test_oracle_evaluation_streams_bound_test_allocation(tmp_path: Path) -> None:
    """Removing per-episode verification would let a corrupt generated bundle enter evidence."""
    from scipy.stats import binomtest

    from silent_cascade.env.generator import EpisodeBlock, IndependentAllocation
    from silent_cascade.env.services import (
        ConfigSelection,
        OracleEvaluationReport,
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
    output = tmp_path / "oracle.json"
    request = OracleEvaluationRequest(
        config=ConfigSelection(),
        source=Phase1GateCorpusSource(
            allocation_id=independent.allocation_id, root_seed=41, public_id_seed=91
        ),
        output_path=output,
    )
    result = evaluate_oracle(request, deps=deps)
    saved = output.read_bytes()
    reused = evaluate_oracle(request, deps=deps)

    assert result.report.verified_episode_count == 12
    assert result.report.schema_version == "oracle-evaluation-report-v2"
    assert result.report.oracle_successes == 12
    assert result.report.namespace_evidence.generation_mode == "independent"
    assert result.report.namespace_evidence.public_id_seed == 91
    assert result.report.namespace_evidence.accepted_draw_count == 12
    assert result.report.namespace_evidence.seed_token_count == 84
    assert result.report.namespace_evidence.base_public_id_count == 12
    assert result.report.namespace_evidence.clock_public_id_count == 0
    assert (
        result.report.seed_token_collisions
        == result.report.namespace_evidence.seed_token_collision_count
    )
    assert (
        result.report.public_id_collisions
        == result.report.namespace_evidence.public_id_collision_count
    )
    assert result.report.corpus_sha256
    assert result.report.provenance.foundation_model_calls == 0
    assert result.report.provenance.analysis_source.scope == "phase1_analysis"
    assert "src/silent_cascade/env/services.py" in result.report.provenance.analysis_source.paths
    assert "src/silent_cascade/env/leakage.py" in result.report.provenance.analysis_source.paths
    assert result.report.random_positive.total == result.report.positive_count
    assert result.report.random_positive.expected_rate == 0.125
    assert result.report.random_positive.exact_binomial_p == pytest.approx(
        binomtest(
            result.report.random_positive.successes,
            result.report.positive_count,
            0.125,
            alternative="two-sided",
        ).pvalue
    )
    negative_count = result.report.safe_negative_count + result.report.disconnected_negative_count
    assert result.report.random_negative.total == negative_count
    assert result.report.random_negative.expected_rate == 0.5
    assert result.report.random_negative.exact_binomial_p == pytest.approx(
        binomtest(
            result.report.random_negative.successes,
            negative_count,
            0.5,
            alternative="two-sided",
        ).pvalue
    )
    assert result.report.random_pooled_expected_rate == 0.3125
    assert result.report.generation_attempt_count == (
        result.report.verified_episode_count + result.report.rejected_draw_count
    )
    assert sum(result.report.rejection_reason_counts.values()) == result.report.rejected_draw_count
    assert result.report.rejected_draw_rate == pytest.approx(
        result.report.rejected_draw_count / result.report.generation_attempt_count
    )
    report_payload = result.report.model_dump(mode="python")
    invalid_reports = (
        {**report_payload, "requested_episode_count": 13},
        {**report_payload, "oracle_successes": 11},
        {
            **report_payload,
            "random_pooled_observed_rate": (
                1.0 if report_payload["random_pooled_observed_rate"] != 1.0 else 0.0
            ),
        },
        {**report_payload, "generation_attempt_count": 1},
        {
            **report_payload,
            "rejected_draw_rate": (1.0 if report_payload["rejected_draw_rate"] != 1.0 else 0.0),
        },
        {**report_payload, "passed": not report_payload["passed"]},
    )
    for invalid in invalid_reports:
        with pytest.raises(ValidationError):
            OracleEvaluationReport.model_validate(invalid)
    assert reused.publication is not None and reused.publication.created is False
    assert output.read_bytes() == saved
    divergent = request.model_copy(
        update={
            "config": ConfigSelection(
                set_overrides=("data.leakage_audit.test.permutation_replicates=1",)
            )
        }
    )
    with pytest.raises(ArtifactError, match="different immutable report"):
        evaluate_oracle(divergent, deps=deps)
    assert output.read_bytes() == saved


def test_oracle_refuses_episode_outside_the_bound_allocation_before_publication(
    tmp_path: Path,
) -> None:
    """A valid episode from another seed is not evidence for the requested allocation."""
    from silent_cascade.env.generator import generate_independent_episode
    from silent_cascade.env.services import (
        ConfigSelection,
        OracleEvaluationRequest,
        Phase1GateCorpusSource,
        evaluate_oracle,
    )
    from silent_cascade.rng import AllocationLabelKey, allocate_independent_variants

    allocation = _independent_allocation()

    def generate_shifted(config: Phase1Config, request: object, public_id_seed: int):
        shifted = replace(request, root_seed=request.root_seed + 1)  # type: ignore[attr-defined]
        variants = allocate_independent_variants(
            AllocationLabelKey(
                "ofd-v1",
                shifted.split_namespace,
                shifted.suite,
                shifted.root_seed,
                shifted.requested_path_length,
                shifted.allocation_quartet_index,
            )
        )
        return generate_independent_episode(
            config,
            replace(shifted, variant=variants[shifted.quartet_member_index]),
            public_id_seed,
        )

    output = tmp_path / "wrong-allocation-must-not-exist.json"
    with pytest.raises(ValueError, match=r"allocation|request|coordinate"):
        evaluate_oracle(
            OracleEvaluationRequest(
                config=ConfigSelection(),
                source=Phase1GateCorpusSource(
                    allocation_id=allocation.allocation_id,
                    root_seed=41,
                    public_id_seed=91,
                ),
                output_path=output,
            ),
            deps=_oracle_dependencies(allocation, generate_independent=generate_shifted),
        )
    assert not output.exists()


def test_oracle_refuses_a_same_label_episode_with_the_wrong_explicit_member(
    tmp_path: Path,
) -> None:
    """Allocation services must distinguish the two positive quartet members."""
    from silent_cascade.env.generator import generate_independent_episode
    from silent_cascade.env.services import (
        ConfigSelection,
        OracleEvaluationRequest,
        Phase1GateCorpusSource,
        evaluate_oracle,
    )

    allocation = _independent_allocation()

    def forge_member(config: Phase1Config, request: object, public_id_seed: int):
        bundle = generate_independent_episode(config, request, public_id_seed)  # type: ignore[arg-type]
        if request.episode_index != 1:  # type: ignore[attr-defined]
            return bundle
        assert request.quartet_member_index == 1  # type: ignore[attr-defined]
        assert bundle.truth.recipe.variant.value == "positive"
        coordinate = bundle.truth.key.coordinate
        return replace(
            bundle,
            truth=replace(
                bundle.truth,
                key=replace(
                    bundle.truth.key,
                    coordinate=replace(coordinate, quartet_member_index=3),
                ),
            ),
        )

    output = tmp_path / "wrong-member-must-not-exist.json"
    with pytest.raises(ValueError, match=r"allocation|request|coordinate"):
        evaluate_oracle(
            OracleEvaluationRequest(
                config=ConfigSelection(),
                source=Phase1GateCorpusSource(
                    allocation_id=allocation.allocation_id,
                    root_seed=41,
                    public_id_seed=91,
                ),
                output_path=output,
            ),
            deps=_oracle_dependencies(allocation, generate_independent=forge_member),
        )
    assert not output.exists()


@pytest.mark.parametrize("service", ("oracle", "leakage"))
def test_allocation_source_mismatch_is_a_typed_configuration_refusal(service: str) -> None:
    """A caller-selected allocation ID cannot become an untyped generation traceback."""
    from silent_cascade.env.services import (
        ConfigSelection,
        Phase1GateCorpusSource,
        _audit_provenance,
        _independent_provenance,
    )

    allocation = _audit_independent_allocation()
    deps = _oracle_dependencies(allocation)
    resolved = resolve_config(
        Phase1Config,
        [ConfigSelection().base_path, ConfigSelection().data_path],
    )
    source = Phase1GateCorpusSource(
        allocation_id="wrong-allocation-v1",
        root_seed=41,
        public_id_seed=91,
    )

    with pytest.raises(ConfigurationError, match=r"allocation|config|source"):
        if service == "oracle":
            _independent_provenance(resolved, source, deps)
        else:
            _audit_provenance(resolved, source, deps)


def test_oracle_refuses_incomplete_denominator_before_publication(tmp_path: Path) -> None:
    from silent_cascade.env.generator import iter_independent_requests
    from silent_cascade.env.services import (
        ConfigSelection,
        OracleEvaluationRequest,
        Phase1GateCorpusSource,
        evaluate_oracle,
    )

    allocation = _independent_allocation()

    def truncated(current: object, root_seed: int):
        return iter(tuple(iter_independent_requests(current, root_seed))[:-4])  # type: ignore[arg-type]

    output = tmp_path / "incomplete-must-not-exist.json"
    with pytest.raises(ValueError, match="denominator"):
        evaluate_oracle(
            OracleEvaluationRequest(
                config=ConfigSelection(),
                source=Phase1GateCorpusSource(
                    allocation_id=allocation.allocation_id,
                    root_seed=41,
                    public_id_seed=91,
                ),
                output_path=output,
            ),
            deps=_oracle_dependencies(allocation, iter_independent=truncated),
        )
    assert not output.exists()


def test_oracle_refuses_independently_invalid_episode_before_publication(
    tmp_path: Path,
) -> None:
    from silent_cascade.env.generator import generate_independent_episode
    from silent_cascade.env.services import (
        ConfigSelection,
        OracleEvaluationRequest,
        Phase1GateCorpusSource,
        evaluate_oracle,
    )
    from silent_cascade.errors import EpisodeInvariantError

    allocation = _independent_allocation()

    def corrupt(config: Phase1Config, request: object, public_id_seed: int):
        bundle = generate_independent_episode(config, request, public_id_seed)  # type: ignore[arg-type]
        first = bundle.public.events[0]
        changed = replace(
            first,
            timestamp=(bundle.public.init.initial_time + first.timestamp) / 2.0,
        )
        return replace(
            bundle,
            public=replace(bundle.public, events=(changed, *bundle.public.events[1:])),
        )

    output = tmp_path / "invalid-invariant-must-not-exist.json"
    with pytest.raises(EpisodeInvariantError, match=r"timestamp|gap|timing|invariant"):
        evaluate_oracle(
            OracleEvaluationRequest(
                config=ConfigSelection(),
                source=Phase1GateCorpusSource(
                    allocation_id=allocation.allocation_id,
                    root_seed=41,
                    public_id_seed=91,
                ),
                output_path=output,
            ),
            deps=_oracle_dependencies(allocation, generate_independent=corrupt),
        )
    assert not output.exists()


def test_manifest_oracle_runs_the_independent_cohort_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import silent_cascade.env.services as services
    from silent_cascade.env.services import (
        ConfigSelection,
        ManifestCorpusSource,
        OracleEvaluationRequest,
        Phase1ServiceDependencies,
        evaluate_oracle,
    )
    from silent_cascade.logging.manifest import publish_manifest

    manifest = _matched_manifest()
    path = tmp_path / "matched-cohorts.json"
    publish_manifest(path, manifest)
    observed: list[int] = []
    real_validate = services.validate_cohort_invariants

    def validate(cohort: object, config: Phase1Config):
        observed.append(len(cohort))  # type: ignore[arg-type]
        return real_validate(cohort, config)  # type: ignore[arg-type]

    monkeypatch.setattr(services, "validate_cohort_invariants", validate)

    def collect(*args: object, **kwargs: object) -> EvidenceProvenance:
        del kwargs
        resolved = args[0]
        return manifest.provenance.model_copy(
            update={"config_sha256": resolved.sha256}  # type: ignore[attr-defined]
        )

    result = evaluate_oracle(
        OracleEvaluationRequest(
            config=ConfigSelection(), source=ManifestCorpusSource(manifest_path=path)
        ),
        deps=Phase1ServiceDependencies.for_test(
            validation_allocation=_audit_validation_allocation(),
            independent_allocation=_independent_allocation(),
            collect_provenance=collect,
            build_manifest=lambda *args: pytest.fail("not used"),
        ),
    )
    assert observed == [4, 4, 4]
    assert result.report.invariant_failures == 0
    assert result.report.namespace_evidence.generation_mode == "matched"
    assert result.report.namespace_evidence.accepted_draw_count == 3
    assert result.report.namespace_evidence.seed_token_count == 60
    assert result.report.namespace_evidence.base_public_id_count == 12


def test_matched_oracle_counts_one_forced_retry_once_per_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four matched members must not multiply one construction-draw rejection by four."""
    import silent_cascade.env.generator as generator
    from silent_cascade.env.services import (
        ConfigSelection,
        ManifestCorpusSource,
        OracleEvaluationRequest,
        Phase1ServiceDependencies,
        evaluate_oracle,
    )
    from silent_cascade.logging.manifest import publish_manifest

    real_validate = generator._validate_counterfactual_family

    def force_first_attempt_retry(family: tuple[object, ...]) -> None:
        first = family[0]
        if first.truth.recipe.accepted_attempt == 0:  # type: ignore[attr-defined]
            raise ValueError("forced matched review retry")
        real_validate(family)  # type: ignore[arg-type]

    monkeypatch.setattr(generator, "_validate_counterfactual_family", force_first_attempt_retry)
    manifest = _matched_manifest()
    path = tmp_path / "matched-forced-retry.json"
    publish_manifest(path, manifest)

    def collect(*args: object, **_kwargs: object) -> EvidenceProvenance:
        resolved = args[0]
        return manifest.provenance.model_copy(
            update={"config_sha256": resolved.sha256}  # type: ignore[attr-defined]
        )

    report = evaluate_oracle(
        OracleEvaluationRequest(
            config=ConfigSelection(), source=ManifestCorpusSource(manifest_path=path)
        ),
        deps=Phase1ServiceDependencies.for_test(
            validation_allocation=_audit_validation_allocation(),
            independent_allocation=_independent_allocation(),
            collect_provenance=collect,
            build_manifest=lambda *args: pytest.fail("not used"),
        ),
    ).report

    assert report.namespace_evidence.accepted_draw_count == 3
    assert report.namespace_evidence.rejected_draw_count == 3
    assert report.namespace_evidence.generation_attempt_count == 6
    assert report.rejected_draw_count == 3
    assert report.generation_attempt_count == 6
    assert report.rejection_reason_counts == {"forced matched review retry": 3}


def test_oracle_rejects_colliding_actual_construction_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unique coordinates cannot substitute for collision-free derived token evidence."""
    import silent_cascade.env.services as services
    from silent_cascade.env.generator import EpisodeBlock, IndependentAllocation
    from silent_cascade.env.services import (
        ConfigSelection,
        OracleEvaluationRequest,
        Phase1GateCorpusSource,
        evaluate_oracle,
    )

    allocation = IndependentAllocation(
        allocation_id="test-independent-v1",
        split_namespace=SplitNamespace.DEBUG,
        blocks=(
            EpisodeBlock(
                suite=SuiteName.IID_PRIMARY,
                requested_path_length=2,
                first_episode_index=0,
                episode_count=8,
            ),
        ),
    )
    monkeypatch.setattr(
        services,
        "independent_seed_tokens",
        lambda _request, _attempt: ("0" * 64,) * 7,
    )

    with pytest.raises(ProvenanceError, match="construction token collision"):
        evaluate_oracle(
            OracleEvaluationRequest(
                config=ConfigSelection(),
                source=Phase1GateCorpusSource(
                    allocation_id=allocation.allocation_id,
                    root_seed=41,
                    public_id_seed=91,
                ),
                output_path=tmp_path / "must-not-exist.json",
            ),
            deps=_oracle_dependencies(allocation),
        )


@pytest.mark.parametrize(
    ("suite", "expected"),
    (
        (SuiteName.BRANCHING_STRESS, "branching"),
        (SuiteName.CONTRADICTION_STRESS, "contradiction"),
        (SuiteName.IID_PRIMARY, "primary"),
        (SuiteName.CLOCK_SCALE_0_1X, "primary"),
    ),
)
def test_oracle_suite_policy_uses_the_exact_authority(suite: SuiteName, expected: str) -> None:
    from silent_cascade.env.services import _oracle_policy_for_suite

    assert _oracle_policy_for_suite(suite).value == expected


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


def test_oracle_service_builds_and_scores_authenticated_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthesizing midpoint actions would bypass the accepted TRACE_JITTER schedule."""
    import silent_cascade.env.oracle as oracle_module
    import silent_cascade.env.services as services
    from silent_cascade.env.generator import EpisodeBlock, IndependentAllocation
    from silent_cascade.env.services import (
        ConfigSelection,
        OracleEvaluationRequest,
        Phase1GateCorpusSource,
        evaluate_oracle,
    )

    allocation = IndependentAllocation(
        allocation_id="test-independent-v1",
        split_namespace=SplitNamespace.DEBUG,
        blocks=(
            EpisodeBlock(
                suite=SuiteName.IID_PRIMARY,
                requested_path_length=2,
                first_episode_index=0,
                episode_count=8,
            ),
        ),
    )
    built_actions: list[tuple[object, ...]] = []
    scored_actions: list[tuple[object, ...]] = []
    real_build = oracle_module.build_oracle_trace
    real_score = services.score_actions

    def build(*args: object, **kwargs: object):
        trace = real_build(*args, **kwargs)  # type: ignore[arg-type]
        built_actions.append(trace.actions)
        return trace

    def score(truth: object, actions: tuple[object, ...]):
        scored_actions.append(actions)
        return real_score(truth, actions)  # type: ignore[arg-type]

    monkeypatch.setattr(services, "build_oracle_trace", build)
    monkeypatch.setattr(services, "score_actions", score)
    result = evaluate_oracle(
        OracleEvaluationRequest(
            config=ConfigSelection(),
            source=Phase1GateCorpusSource(
                allocation_id=allocation.allocation_id,
                root_seed=41,
                public_id_seed=91,
            ),
            output_path=tmp_path / "trace-scored-oracle.json",
        ),
        deps=_oracle_dependencies(allocation),
    )

    assert result.report.oracle_successes == 8
    assert len(built_actions) == 8
    assert all(actions in scored_actions for actions in built_actions)


@pytest.mark.parametrize(
    "clock_suite",
    (SuiteName.CLOCK_SCALE_0_1X, SuiteName.CLOCK_SCALE_10X),
)
def test_oracle_service_builds_parent_traces_and_scales_clock_traces(
    clock_suite: SuiteName, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision-only clock checks cannot authenticate the complete paired event schedule."""
    import silent_cascade.env.oracle as oracle_module
    import silent_cascade.env.services as services
    from silent_cascade.env.episode import EpisodeVariant
    from silent_cascade.env.services import (
        ConfigSelection,
        ManifestCorpusSource,
        OracleEvaluationRequest,
        Phase1ServiceDependencies,
        evaluate_oracle,
    )
    from silent_cascade.logging.manifest import publish_manifest

    manifest, parents = _clock_manifest(clock_suite)
    path = tmp_path / "clock-trace-manifest.json"
    publish_manifest(path, manifest)
    built: list[tuple[object, object, object]] = []
    scaled: list[tuple[object, float]] = []
    scored: list[tuple[object, object]] = []
    real_build = services._build_authenticated_trace
    real_scale = oracle_module.scale_oracle_trace
    real_score = services.score_actions

    def build(bundle: object, solution: object):
        trace = real_build(bundle, solution)  # type: ignore[arg-type]
        built.append((bundle, solution, trace))
        return trace

    def scale(trace: object, factor: float):
        scaled.append((trace, factor))
        return real_scale(trace, factor)  # type: ignore[arg-type]

    def score(truth: object, actions: object):
        scored.append((truth, actions))
        return real_score(truth, actions)  # type: ignore[arg-type]

    monkeypatch.setattr(services, "_build_authenticated_trace", build)
    monkeypatch.setattr(services, "scale_oracle_trace", scale)
    monkeypatch.setattr(services, "score_actions", score)

    def collect(*args: object, **kwargs: object) -> EvidenceProvenance:
        del kwargs
        resolved = args[0]
        return manifest.provenance.model_copy(
            update={"config_sha256": resolved.sha256}  # type: ignore[attr-defined]
        )

    evaluate_oracle(
        OracleEvaluationRequest(
            config=ConfigSelection(),
            source=ManifestCorpusSource(manifest_path=path),
        ),
        deps=Phase1ServiceDependencies.for_test(
            validation_allocation=_allocation(),
            independent_allocation=_independent_allocation(8),
            collect_provenance=collect,
            build_manifest=lambda *args: pytest.fail("not used"),
        ),
    )

    child_ids = tuple(entry.episode_public_id for entry in manifest.entries)
    parent_ids = tuple(parent.public.init.episode_public_id for parent in parents)
    expected_build_ids = (
        *child_ids,
        *(public_id for pair in zip(parent_ids, child_ids, strict=True) for public_id in pair),
    )
    assert tuple(item[0].public.init.episode_public_id for item in built) == expected_build_ids  # type: ignore[union-attr]
    assert all(item[2].solution is item[1] for item in built)  # type: ignore[union-attr]
    assert len(scaled) == len(manifest.entries)
    for pair_index, (scaled_trace, factor) in enumerate(scaled):
        parent_build = built[len(child_ids) + 2 * pair_index]
        child_build = built[len(child_ids) + 2 * pair_index + 1]
        assert scaled_trace is parent_build[2]
        assert factor == child_build[0].truth.recipe.clock_scale  # type: ignore[union-attr]

    assert len(scored) == 4 * len(manifest.entries)
    for index, child_build in enumerate(built[: len(child_ids)]):
        truth, actions = scored[2 * index]
        assert truth is child_build[0].truth  # type: ignore[union-attr]
        assert actions is child_build[2].actions  # type: ignore[union-attr]
    paired_score_offset = 2 * len(manifest.entries)
    for pair_index in range(len(manifest.entries)):
        parent_build = built[len(child_ids) + 2 * pair_index]
        child_build = built[len(child_ids) + 2 * pair_index + 1]
        parent_score = scored[paired_score_offset + 2 * pair_index]
        child_score = scored[paired_score_offset + 2 * pair_index + 1]
        assert parent_score[0] is parent_build[0].truth  # type: ignore[union-attr]
        assert parent_score[1] is parent_build[2].actions  # type: ignore[union-attr]
        assert child_score[0] is child_build[0].truth  # type: ignore[union-attr]
        assert child_score[1] is child_build[2].actions  # type: ignore[union-attr]

    assert {parent.truth.recipe.variant for parent in parents} == {
        EpisodeVariant.POSITIVE,
        EpisodeVariant.SAFE_NEGATIVE,
        EpisodeVariant.DISCONNECTED_NEGATIVE,
    }


def test_clock_trace_comparison_rejects_corrupted_child_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scaling only the parent trace would conceal a corrupted child scheduler."""
    import silent_cascade.env.services as services
    from silent_cascade.env.episode import scale_episode_time

    manifest, parents = _clock_manifest()
    entry = manifest.entries[0]
    parent = parents[0]
    child = scale_episode_time(parent, entry.suite, entry.episode_public_id)
    child_timing = child.truth.recipe.oracle_timing.model_copy(
        update={"delta_0": child.truth.recipe.oracle_timing.delta_0 * 1.5}
    )
    corrupted = replace(
        child,
        truth=replace(
            child.truth,
            recipe=replace(child.truth.recipe, oracle_timing=child_timing),
        ),
    )
    monkeypatch.setattr(
        services,
        "_build_authenticated_trace",
        lambda *_args: pytest.fail("trace built before scaled timing validation"),
    )

    assert not services._clock_decision_matches(parent, corrupted)


def test_clock_trace_comparison_rejects_forged_parent_episode_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A matching parent ID cannot authenticate a child bound to another parent payload."""
    import silent_cascade.env.services as services
    from silent_cascade.env.episode import scale_episode_time

    manifest, parents = _clock_manifest()
    entry = manifest.entries[0]
    parent = parents[0]
    child = scale_episode_time(parent, entry.suite, entry.episode_public_id)
    corrupted = replace(
        child,
        truth=replace(
            child.truth,
            recipe=replace(
                child.truth.recipe,
                parent_episode_sha256="0" * 64,
            ),
        ),
    )
    monkeypatch.setattr(
        services,
        "_build_authenticated_trace",
        lambda *_args: pytest.fail("trace built before parent provenance validation"),
    )

    assert not services._clock_decision_matches(parent, corrupted)


@pytest.mark.parametrize(
    "timing_update",
    (
        {"jitter_log_std": 0.2},
        {"terminal_compose_fraction": 0.55},
        {"action_window_start_fraction": 0.70},
        {"action_target_fraction": 0.80},
        {"action_window_end_fraction": 0.95},
    ),
)
def test_clock_trace_comparison_rejects_altered_normalized_window_fractions(
    timing_update: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consistent absolute child windows cannot excuse changed normalized timing."""
    import silent_cascade.env.services as services
    from silent_cascade.env.episode import EpisodeVariant, scale_episode_time
    from silent_cascade.env.timing import action_window

    manifest, parents = _clock_manifest()
    pair_index = next(
        index
        for index, parent in enumerate(parents)
        if parent.truth.recipe.variant is EpisodeVariant.POSITIVE
    )
    entry = manifest.entries[pair_index]
    parent = parents[pair_index]
    child = scale_episode_time(parent, entry.suite, entry.episode_public_id)
    child_timing = child.truth.recipe.oracle_timing.model_copy(update=timing_update)
    consistent_window = action_window(
        child.truth.activation_time,
        child.truth.episode_delay,
        child_timing,
    )
    corrupted = replace(
        child,
        truth=replace(
            child.truth,
            recipe=replace(child.truth.recipe, oracle_timing=child_timing),
            action_window_start=consistent_window.start,
            action_window_end=consistent_window.end,
            action_target=consistent_window.target,
        ),
    )
    monkeypatch.setattr(
        services,
        "_build_authenticated_trace",
        lambda *_args: pytest.fail("trace built before normalized timing validation"),
    )

    assert not services._clock_decision_matches(parent, corrupted)


def test_manifest_oracle_authenticates_every_regenerated_entry_before_publication(
    tmp_path: Path,
) -> None:
    """A cohort-preserving permutation is valid data but not the requested manifest recipe."""
    from silent_cascade.env.services import (
        ConfigSelection,
        ManifestCorpusSource,
        OracleEvaluationRequest,
        Phase1ServiceDependencies,
        evaluate_oracle,
        regenerate_entry,
    )
    from silent_cascade.logging.manifest import MatchedManifestCoordinate, publish_manifest

    manifest = _matched_manifest()
    manifest_path = tmp_path / "matched-manifest.json"
    output = tmp_path / "wrong-manifest-recipe-must-not-exist.json"
    publish_manifest(manifest_path, manifest)

    def collect(*args: object, **kwargs: object) -> EvidenceProvenance:
        del kwargs
        resolved = args[0]
        return manifest.provenance.model_copy(
            update={"config_sha256": resolved.sha256}  # type: ignore[attr-defined]
        )

    def permute(config: Phase1Config, current: object, entry: object):
        coordinate = entry.coordinate  # type: ignore[attr-defined]
        assert isinstance(coordinate, MatchedManifestCoordinate)
        replacement = next(
            candidate
            for candidate in manifest.entries
            if isinstance(candidate.coordinate, MatchedManifestCoordinate)
            and candidate.coordinate.cohort_index == coordinate.cohort_index
            and candidate.coordinate.member_index == (coordinate.member_index + 1) % 4
        )
        return regenerate_entry(config, current, replacement)  # type: ignore[arg-type]

    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=_audit_validation_allocation(),
        independent_allocation=_independent_allocation(),
        collect_provenance=collect,
        build_manifest=lambda *args: pytest.fail("not used"),
        regenerate_manifest_entry=permute,
    )
    with pytest.raises(EpisodeError, match=r"manifest|entry|coordinate"):
        evaluate_oracle(
            OracleEvaluationRequest(
                config=ConfigSelection(),
                source=ManifestCorpusSource(manifest_path=manifest_path),
                output_path=output,
            ),
            deps=deps,
        )
    assert not output.exists()


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
        _anchor_for_bound_source,
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
    assert source.descriptor.episode_count == 12
    assert source.authentication.suite_path_denominators == {
        "iid_primary:2": 4,
        "iid_primary:3": 4,
        "iid_primary:4": 4,
    }
    assert source.authentication.clock_scale_pair_counts["scale_0_1x"] > 0
    assert source.authentication.clock_scale_pair_counts["scale_10x"] > 0
    with pytest.raises(ConfigurationError, match=r"profile|source"):
        _anchor_for_bound_source(source, LeakageAuditProfileName.PHASE1_GATE)


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
        _publish_report,
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
    request = LeakageAuditRequest(
        config=config_selection,
        source=source,
        profile=LeakageAuditProfileName.TEST,
        output_path=output,
    )
    result = run_leakage_audit(request, deps=deps)
    saved = output.read_bytes()
    reused = _publish_report(output, result.report)

    assert result.report.passed is (not scientific_failure)
    assert result.report.schema_version == "leakage-report-v2"
    assert result.report.namespace_evidence.public_id_seed == 91
    assert result.report.namespace_evidence.accepted_draw_count == (
        3 if source_mode == "manifest" else 12
    )
    assert result.report.namespace_evidence.seed_token_count == (
        60 if source_mode == "manifest" else 84
    )
    assert result.report.namespace_evidence.base_public_id_count == 12
    assert result.report.namespace_evidence.clock_public_id_count == (
        24 if source_mode == "manifest" else 2
    )
    assert tuple(item.check_id for item in result.report.counterfactual_checks) == tuple(
        CounterfactualCheckId
    )
    assert all(
        item.passed
        and item.checked_pairs > 0
        and item.decision_mismatch_count == 0
        and item.temporal_mismatch_count == 0
        and len(item.result_payload_sha256) == 64
        for item in result.report.counterfactual_checks
    )
    assert result.report.provenance.analysis_seeds == {
        "audit_seed": 2026083091,
        "positive_control_seed": 2026083092,
    }
    assert result.publication is not None and result.publication.created
    assert reused.created is False
    assert output.exists()
    assert output.read_bytes() == saved
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
        assert oracle.report.clock_0_1x_episode_count == 1
        assert oracle.report.clock_10x_episode_count == 1
        assert oracle.report.clock_decision_mismatches == 0
        assert sum(oracle.report.suite_path_denominators.values()) == 12
        assert oracle.report.generation_attempt_count == (
            oracle.report.verified_episode_count + oracle.report.rejected_draw_count
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
        divergent = result.report.model_copy(update={"feature_schema_hash": "f" * 64})
        with pytest.raises(ArtifactError, match="different immutable report"):
            _publish_report(output, divergent)
        assert output.read_bytes() == saved
        assert cleanup_observed == [()]


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


def _canonical_validation_manifest_recipe():
    from uuid import UUID

    from silent_cascade.env.generator import VALIDATION_ALLOCATION
    from silent_cascade.logging.manifest import (
        EpisodeManifest,
        EpisodeManifestEntry,
        MatchedManifestCoordinate,
    )

    resolved = resolve_config(
        Phase1Config, [Path("configs/base.yaml"), Path("configs/data/primary.yaml")]
    )
    provenance = _provenance(allocation_id=VALIDATION_ALLOCATION.allocation_id).model_copy(
        update={
            "split_namespace": SplitNamespace.VALIDATION,
            "config_sha256": resolved.sha256,
        }
    )
    entries = []
    rank = 0
    for block in VALIDATION_ALLOCATION.blocks:
        for cohort_index in range(
            block.first_cohort_index,
            block.first_cohort_index + block.cohort_count,
        ):
            for member_index in range(4):
                rank += 1
                entries.append(
                    EpisodeManifestEntry(
                        episode_public_id=str(UUID(int=rank, version=4)),
                        split_namespace=SplitNamespace.VALIDATION,
                        suite=SuiteName.VALIDATION,
                        coordinate=MatchedManifestCoordinate(
                            cohort_index=cohort_index,
                            member_index=member_index,
                        ),
                        requested_path_length=block.requested_path_length,
                        accepted_attempt=0,
                        episode_sha256=f"{rank:064x}",
                    )
                )
    return EpisodeManifest(
        schema_version=1,
        experiment_version=resolved.config.experiment_version,
        access_class=ManifestAccessClass.VALIDATION,
        provenance=provenance,
        suite=SuiteName.VALIDATION,
        public_id_seed=91,
        episode_count=len(entries),
        entries=tuple(entries),
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


def _manifest_current_collector(manifest: object):
    def collect(*args: object, **kwargs: object) -> EvidenceProvenance:
        resolved = args[0]
        return manifest.provenance.model_copy(  # type: ignore[attr-defined]
            update={
                "config_sha256": resolved.sha256,  # type: ignore[attr-defined]
                "analysis_seeds": kwargs["analysis_seeds"],
            }
        )

    return collect


def _oracle_dependencies(
    allocation: object,
    *,
    iter_independent=None,
    generate_independent=None,
):
    from silent_cascade.env.generator import (
        generate_independent_episode,
        iter_independent_requests,
    )
    from silent_cascade.env.services import Phase1ServiceDependencies

    def collect(*args: object, **kwargs: object) -> EvidenceProvenance:
        resolved = args[0]
        return _provenance(allocation_id=allocation.allocation_id).model_copy(  # type: ignore[attr-defined]
            update={
                "generation_mode": "independent",
                "config_sha256": resolved.sha256,  # type: ignore[attr-defined]
                "root_seed": kwargs["root_seed"],
                "public_id_seed_sha256": public_id_seed_sha256(kwargs["public_id_seed"]),
                "analysis_seeds": kwargs["analysis_seeds"],
            }
        )

    return Phase1ServiceDependencies.for_test(
        validation_allocation=_allocation(),
        independent_allocation=allocation,  # type: ignore[arg-type]
        collect_provenance=collect,
        build_manifest=lambda *args: pytest.fail("not used"),
        iter_independent=iter_independent or iter_independent_requests,
        generate_independent=generate_independent or generate_independent_episode,
    )


def _small_access_manifest(
    split: SplitNamespace,
    access_class: ManifestAccessClass,
    allocation_id: str,
    suite: SuiteName,
):
    from silent_cascade.env.episode import episode_sha256
    from silent_cascade.env.generator import (
        CohortAllocation,
        CohortBlock,
        EpisodeBlock,
        IndependentAllocation,
        generate_independent_episode,
        generate_matched_cohort,
        iter_cohort_requests,
        iter_independent_requests,
    )
    from silent_cascade.logging.manifest import (
        EpisodeManifest,
        EpisodeManifestEntry,
        IndependentManifestCoordinate,
        MatchedManifestCoordinate,
    )

    resolved = resolve_config(
        Phase1Config, [Path("configs/base.yaml"), Path("configs/data/primary.yaml")]
    )
    provenance = _provenance(allocation_id=allocation_id).model_copy(
        update={
            "generation_mode": (
                "matched" if access_class is ManifestAccessClass.VALIDATION else "independent"
            ),
            "split_namespace": split,
            "config_sha256": resolved.sha256,
        }
    )
    entries = []
    if access_class is ManifestAccessClass.VALIDATION:
        allocation = CohortAllocation(
            allocation_id=allocation_id,
            split_namespace=split,
            blocks=(
                CohortBlock(
                    suite=suite,
                    requested_path_length=2,
                    first_cohort_index=0,
                    cohort_count=2,
                ),
            ),
        )
        for request in iter_cohort_requests(allocation, 41):
            cohort = generate_matched_cohort(resolved.config, request, 91)
            for member_index, bundle in enumerate(cohort.episodes):
                entries.append(
                    EpisodeManifestEntry(
                        episode_public_id=bundle.public.init.episode_public_id,
                        split_namespace=split,
                        suite=suite,
                        coordinate=MatchedManifestCoordinate(
                            cohort_index=request.cohort_index,
                            member_index=member_index,
                        ),
                        requested_path_length=2,
                        accepted_attempt=cohort.accepted_attempt,
                        episode_sha256=episode_sha256(bundle),
                    )
                )
    else:
        allocation = IndependentAllocation(
            allocation_id=allocation_id,
            split_namespace=split,
            blocks=(
                EpisodeBlock(
                    suite=suite,
                    requested_path_length=2,
                    first_episode_index=0,
                    episode_count=8,
                ),
            ),
        )
        for request in iter_independent_requests(allocation, 41):
            bundle = generate_independent_episode(resolved.config, request, 91)
            entries.append(
                EpisodeManifestEntry(
                    episode_public_id=bundle.public.init.episode_public_id,
                    split_namespace=split,
                    suite=suite,
                    coordinate=IndependentManifestCoordinate(
                        episode_index=request.episode_index,
                        allocation_quartet_index=request.allocation_quartet_index,
                        quartet_member_index=request.quartet_member_index,
                    ),
                    requested_path_length=2,
                    accepted_attempt=bundle.truth.recipe.accepted_attempt,
                    episode_sha256=episode_sha256(bundle),
                )
            )
    return EpisodeManifest(
        schema_version=1,
        experiment_version=resolved.config.experiment_version,
        access_class=access_class,
        provenance=provenance,
        suite=suite,
        public_id_seed=91,
        episode_count=len(entries),
        entries=tuple(entries),
    )


def _clock_manifest(
    target_suite: SuiteName = SuiteName.CLOCK_SCALE_0_1X,
):
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
    provenance = _provenance(allocation_id=allocation.allocation_id).model_copy(
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
                suite=target_suite,
                public_id_seed=91,
                episode_index=request.episode_index,
                accepted_attempt=parent.truth.recipe.accepted_attempt,
            )
        )
        child = scale_episode_time(parent, target_suite, child_id)
        entries.append(
            EpisodeManifestEntry(
                episode_public_id=child_id,
                split_namespace=SplitNamespace.DEBUG,
                suite=target_suite,
                coordinate=IndependentManifestCoordinate(
                    episode_index=request.episode_index,
                    allocation_quartet_index=request.allocation_quartet_index,
                    quartet_member_index=request.quartet_member_index,
                ),
                requested_path_length=request.requested_path_length,
                accepted_attempt=parent.truth.recipe.accepted_attempt,
                episode_sha256=episode_sha256(child),
                parent_public_id=parent.public.init.episode_public_id,
                parent_episode_sha256=episode_sha256(parent),
                clock_scale=(0.1 if target_suite is SuiteName.CLOCK_SCALE_0_1X else 10.0),
            )
        )
    return (
        EpisodeManifest(
            schema_version=1,
            experiment_version="v1",
            access_class=ManifestAccessClass.DEBUG,
            provenance=provenance,
            suite=target_suite,
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
    member_index = next(
        index for index, variant in enumerate(variants) if variant is expected_variant
    )
    request = IndependentEpisodeRequest(
        SplitNamespace.DEBUG,
        suite,
        41,
        5 + member_index,
        2,
        expected_variant,
        2,
        member_index,
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
    regenerated = regenerate_entry(_stress_config(), manifest, entry)
    assert regenerated == bundle
    assert regenerated.truth.key.coordinate.quartet_member_index == member_index
