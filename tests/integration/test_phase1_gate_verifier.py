"""Task 17 contracts for local, read-only Phase 1 evidence verification."""

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from runpy import run_path

import pytest

from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.episode import CorpusDigestEntry, corpus_sha256
from silent_cascade.env.leakage import (
    NAMED_LEAK_INJECTORS,
    AuditSourceDescriptor,
    CounterfactualCheckId,
    CounterfactualCheckResult,
    LeakageAuditProfileName,
    LeakageReport,
    PositiveControlResult,
    audit_source_descriptor_sha256,
)
from silent_cascade.env.reproducibility import IndependentSourceDescriptor, ReproducibilityReport
from silent_cascade.env.services import ExactRandomCheck, OracleEvaluationReport
from silent_cascade.errors import ArtifactIntegrityError
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.manifest import (
    EpisodeManifest,
    EpisodeManifestEntry,
    ManifestAccessClass,
    ManifestEnvelope,
    MatchedManifestCoordinate,
)
from silent_cascade.provenance import (
    GENERATOR_SOURCE_PATHS,
    PHASE1_ANALYSIS_SOURCE_PATHS,
    EvidenceProvenance,
    LeakageAuditEvidenceAnchor,
    SourceTreeFingerprint,
    public_id_seed_sha256,
)

verify_phase1_gate_artifacts = run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/verify_phase1_gate_artifacts.py")
)["verify_phase1_gate_artifacts"]

GATE_DENOMINATORS = {
    "distractor_flood:2": 8_000,
    "distractor_flood:3": 8_000,
    "distractor_flood:4": 8_000,
    "iid_primary:2": 8_000,
    "iid_primary:3": 8_000,
    "iid_primary:4": 8_000,
    "ood_depth:5": 4_000,
    "ood_depth:6": 4_000,
    "ood_depth:7": 4_000,
    "ood_depth:8": 4_000,
    "ood_long_delay:2": 4_000,
    "ood_long_delay:3": 4_000,
    "ood_long_delay:4": 4_000,
    "ood_short_delay:2": 8_000,
    "ood_short_delay:3": 8_000,
    "ood_short_delay:4": 8_000,
}
GATE_ALLOCATION_SHA256 = "9e032f6993af9f2d53c1dde3a3e3e72e097cf143ae2e72d02609f2d2d6f1ce18"
INDEPENDENT_SOURCE_PAYLOAD_SHA256 = (
    "799f4add88eae1e541eff6edf7e3afa35ec22a68e29f7a8309b05f6b315cc121"
)


def _provenance(*, independent: bool) -> EvidenceProvenance:
    return EvidenceProvenance(
        schema_version="phase1-evidence-provenance-v1",
        plan_base_revision="a" * 40,
        source_commit="b" * 40,
        source_dirty=False,
        generator_version="ofd-v1",
        generation_mode="independent" if independent else "matched",
        allocation_id="phase1-independent-gate-v1" if independent else "validation-v1",
        split_namespace=(SplitNamespace.PHASE1_GATE if independent else SplitNamespace.VALIDATION),
        config_sha256="c" * 64,
        generator_source=SourceTreeFingerprint(
            frame_version="sc-source-tree-v1",
            scope="generator",
            paths=GENERATOR_SOURCE_PATHS,
            sha256="d" * 64,
        ),
        analysis_source=SourceTreeFingerprint(
            frame_version="sc-source-tree-v1",
            scope="phase1_analysis",
            paths=PHASE1_ANALYSIS_SOURCE_PATHS,
            sha256="e" * 64,
        ),
        root_seed=2026083011 if independent else 2026083001,
        public_id_seed_sha256=public_id_seed_sha256(2026083012 if independent else 2026083002),
    )


def _validation_manifest() -> EpisodeManifest:
    provenance = _provenance(independent=False)
    entries = tuple(
        EpisodeManifestEntry(
            episode_public_id=f"00000000-0000-4000-8000-{index + 1:012x}",
            split_namespace=SplitNamespace.VALIDATION,
            suite=SuiteName.VALIDATION,
            coordinate=MatchedManifestCoordinate(
                cohort_index=index // 4,
                member_index=index % 4,
            ),
            requested_path_length=(2 if index // 4 < 834 else 3 if index // 4 < 1_667 else 4),
            accepted_attempt=0,
            episode_sha256=f"{index + 1:064x}",
        )
        for index in range(10_000)
    )
    return EpisodeManifest(
        schema_version=1,
        experiment_version="v1",
        access_class=ManifestAccessClass.VALIDATION,
        provenance=provenance,
        suite=SuiteName.VALIDATION,
        public_id_seed=2026083002,
        episode_count=10_000,
        entries=entries,
    )


def _counterfactual(
    check_id: CounterfactualCheckId,
    checked_pairs: int,
    marker: str,
) -> CounterfactualCheckResult:
    return CounterfactualCheckResult(
        check_id=check_id,
        checked_pairs=checked_pairs,
        decision_mismatch_count=0,
        temporal_mismatch_count=0,
        result_payload_sha256=marker * 64,
        passed=True,
    )


def _positive_controls() -> tuple[PositiveControlResult, ...]:
    return tuple(
        PositiveControlResult(
            control_id=injector.control_id,
            target_task=injector.target_task,
            expected_detector_id=injector.expected_detector_id,
            observed_detector_ids=(injector.expected_detector_id,),
            base_subset_corpus_sha256="1" * 64,
            injected_corpus_sha256=f"{index + 17:064x}",
            split_membership_sha256="3" * 64,
            balanced_accuracy=1.0,
            holm_adjusted_p=0.0054,
            passed=True,
        )
        for index, injector in enumerate(NAMED_LEAK_INJECTORS)
    )


@pytest.fixture(scope="session")
def consistent_artifact_bytes() -> dict[str, bytes]:
    manifest = _validation_manifest()
    manifest_payload_sha256 = sha256_bytes(canonical_json_bytes(manifest))
    envelope = ManifestEnvelope(
        payload=manifest,
        payload_sha256=manifest_payload_sha256,
    )
    validation_corpus = corpus_sha256(
        (
            CorpusDigestEntry(entry.episode_public_id, entry.episode_sha256)
            for entry in manifest.entries
        ),
        expected_count=10_000,
    )
    gate_corpus = "f" * 64
    independent = _provenance(independent=True)
    descriptor = AuditSourceDescriptor(
        schema_version="leakage-source-v1",
        generation_mode="independent",
        allocation_id=independent.allocation_id,
        allocation_or_manifest_sha256=GATE_ALLOCATION_SHA256,
        split_namespace=independent.split_namespace,
        root_seed=independent.root_seed,
        public_id_seed_sha256=independent.public_id_seed_sha256,
        config_sha256=independent.config_sha256,
        generator_source_sha256=independent.generator_source.sha256,
        episode_count=100_000,
    )
    oracle = OracleEvaluationReport(
        schema_version="oracle-evaluation-report-v1",
        provenance=independent,
        source_mode="phase1_gate",
        requested_episode_count=100_000,
        verified_episode_count=100_000,
        positive_count=50_000,
        safe_negative_count=25_000,
        disconnected_negative_count=25_000,
        suite_path_denominators=GATE_DENOMINATORS,
        invariant_failures=0,
        oracle_ambiguities=0,
        seed_token_collisions=0,
        public_id_collisions=0,
        oracle_successes=100_000,
        oracle_failures=0,
        random_positive=ExactRandomCheck(
            successes=6_250,
            total=50_000,
            observed_rate=0.125,
            expected_rate=0.125,
            absolute_error=0.0,
            exact_binomial_p=1.0,
            passed=True,
        ),
        random_negative=ExactRandomCheck(
            successes=25_000,
            total=50_000,
            observed_rate=0.5,
            expected_rate=0.5,
            absolute_error=0.0,
            exact_binomial_p=1.0,
            passed=True,
        ),
        random_pooled_observed_rate=0.3125,
        random_pooled_expected_rate=0.3125,
        clock_0_1x_episode_count=5_000,
        clock_10x_episode_count=2_000,
        clock_decision_mismatches=0,
        rejection_reason_counts={},
        generation_attempt_count=100_000,
        rejected_draw_count=0,
        rejected_draw_rate=0.0,
        corpus_sha256=gate_corpus,
        passed=True,
    )
    anchor = LeakageAuditEvidenceAnchor(
        schema_version="phase1-leakage-audit-anchor-v1",
        profile="phase1_gate",
        allocation_id="phase1-independent-gate-v1",
        allocation_or_manifest_sha256=GATE_ALLOCATION_SHA256,
        config_sha256=independent.config_sha256,
        descriptor_sha256=audit_source_descriptor_sha256(descriptor),
        source_manifest_sha256="3" * 64,
        suite_path_denominators=GATE_DENOMINATORS,
        clock_pair_manifest_sha256="4" * 64,
        clock_scale_pair_counts={"scale_0_1x": 5_000, "scale_10x": 2_000},
        episode_count=100_000,
    )
    leakage = LeakageReport(
        schema_version="leakage-report-v1",
        provenance=independent.model_copy(
            update={
                "analysis_seeds": {
                    "audit_seed": 2026083091,
                    "positive_control_seed": 2026083092,
                },
                "leakage_audit": anchor,
            }
        ),
        generation_mode="independent",
        profile=LeakageAuditProfileName.PHASE1_GATE,
        corpus_hash=gate_corpus,
        feature_schema_hash="5" * 64,
        split_membership_hash="6" * 64,
        train_membership_hash="7" * 64,
        test_membership_hash="8" * 64,
        episode_count=100_000,
        randomization_block_count=25_000,
        suite_path_denominators=GATE_DENOMINATORS,
        construction_checks={"complete": True},
        probes=(),
        positive_controls=_positive_controls(),
        counterfactual_checks=(
            _counterfactual(CounterfactualCheckId.TERMINAL_DELAY_SWAP, 25_000, "9"),
            _counterfactual(CounterfactualCheckId.PRESENTATION_PERMUTATION, 100_000, "a"),
            _counterfactual(CounterfactualCheckId.PAIRED_CLOCK_SCALE, 7_000, "b"),
        ),
        label_shuffled_control_passed=True,
        passed=True,
    )
    validation_reproducibility = ReproducibilityReport(
        schema_version="phase1-reproducibility-v1",
        source_mode="manifest",
        source_payload_sha256=manifest_payload_sha256,
        sample_size=1_000,
        verified_source_entries=10_000,
        modes=("forward", "reverse", "chunked", "fresh_process"),
        chunk_sizes=(1, 3, 7),
        python_hash_seeds=(0, 1),
        reference_corpus_sha256=validation_corpus,
        sample_membership_sha256="c" * 64,
        mismatch_count=0,
        provenance=manifest.provenance,
        passed=True,
    )
    independent_reproducibility = ReproducibilityReport(
        schema_version="phase1-reproducibility-v1",
        source_mode="independent_allocation",
        source_payload_sha256=INDEPENDENT_SOURCE_PAYLOAD_SHA256,
        sample_size=1_000,
        verified_source_entries=100_000,
        modes=("forward", "reverse", "chunked", "fresh_process"),
        chunk_sizes=(1, 3, 7),
        python_hash_seeds=(0, 1),
        reference_corpus_sha256=gate_corpus,
        sample_membership_sha256="e" * 64,
        mismatch_count=0,
        provenance=independent,
        passed=True,
    )
    return {
        "validation.json": canonical_json_bytes(envelope),
        "oracle.json": canonical_json_bytes(oracle),
        "leakage.json": canonical_json_bytes(leakage),
        "validation-reproducibility.json": canonical_json_bytes(validation_reproducibility),
        "independent-reproducibility.json": canonical_json_bytes(independent_reproducibility),
    }


def _write_artifacts(tmp_path: Path, artifacts: dict[str, bytes]) -> dict[str, Path]:
    paths = {name: tmp_path / name for name in artifacts}
    for name, payload in artifacts.items():
        paths[name].write_bytes(payload)
    return paths


def _verify(paths: dict[str, Path]):
    return verify_phase1_gate_artifacts(
        validation_path=paths["validation.json"],
        oracle_path=paths["oracle.json"],
        leakage_path=paths["leakage.json"],
        validation_reproducibility_path=paths["validation-reproducibility.json"],
        independent_reproducibility_path=paths["independent-reproducibility.json"],
    )


def _mutate_json(payload: bytes, mutation: Callable[[dict[str, object]], None]) -> bytes:
    value = json.loads(payload)
    mutation(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _mutate_validation_manifest(
    artifacts: dict[str, bytes], mutation: Callable[[dict[str, object]], None]
) -> None:
    envelope = json.loads(artifacts["validation.json"])
    mutation(envelope["payload"])
    envelope["payload_sha256"] = sha256_bytes(canonical_json_bytes(envelope["payload"]))
    artifacts["validation.json"] = canonical_json_bytes(envelope)

    def rebind_reproducibility(value: dict[str, object]) -> None:
        value["source_payload_sha256"] = envelope["payload_sha256"]

    artifacts["validation-reproducibility.json"] = _mutate_json(
        artifacts["validation-reproducibility.json"], rebind_reproducibility
    )


def _independent_descriptor_sha256(**updates: object) -> str:
    values: dict[str, object] = {
        "schema_version": "phase1-independent-source-v1",
        "allocation_id": "phase1-independent-gate-v1",
        "allocation_sha256": GATE_ALLOCATION_SHA256,
        "split_namespace": SplitNamespace.PHASE1_GATE,
        "root_seed": 2026083011,
        "public_id_seed_sha256": public_id_seed_sha256(2026083012),
        "config_sha256": "c" * 64,
        "generator_source_sha256": "d" * 64,
    }
    values.update(updates)
    descriptor = IndependentSourceDescriptor.model_validate(values)
    return sha256_bytes(canonical_json_bytes(descriptor))


def _replace_validation_seeds(
    artifacts: dict[str, bytes], *, root_seed: int, public_id_seed: int
) -> None:
    envelope = json.loads(artifacts["validation.json"])
    provenance = envelope["payload"]["provenance"]
    provenance["root_seed"] = root_seed
    provenance["public_id_seed_sha256"] = public_id_seed_sha256(public_id_seed)
    envelope["payload"]["public_id_seed"] = public_id_seed
    envelope["payload_sha256"] = sha256_bytes(canonical_json_bytes(envelope["payload"]))
    artifacts["validation.json"] = canonical_json_bytes(envelope)

    def rebind(value: dict[str, object]) -> None:
        report_provenance = value["provenance"]  # type: ignore[assignment]
        report_provenance["root_seed"] = root_seed  # type: ignore[index]
        report_provenance["public_id_seed_sha256"] = public_id_seed_sha256(  # type: ignore[index]
            public_id_seed
        )
        value["source_payload_sha256"] = envelope["payload_sha256"]

    artifacts["validation-reproducibility.json"] = _mutate_json(
        artifacts["validation-reproducibility.json"], rebind
    )


def _replace_independent_seeds(
    artifacts: dict[str, bytes], *, root_seed: int, public_id_seed: int
) -> None:
    fingerprint = public_id_seed_sha256(public_id_seed)
    for artifact_name in (
        "oracle.json",
        "leakage.json",
        "independent-reproducibility.json",
    ):

        def rebind(value: dict[str, object], *, name: str = artifact_name) -> None:
            provenance = value["provenance"]  # type: ignore[assignment]
            provenance["root_seed"] = root_seed  # type: ignore[index]
            provenance["public_id_seed_sha256"] = fingerprint  # type: ignore[index]
            if name == "leakage.json":
                descriptor = AuditSourceDescriptor(
                    schema_version="leakage-source-v1",
                    generation_mode="independent",
                    allocation_id="phase1-independent-gate-v1",
                    allocation_or_manifest_sha256=GATE_ALLOCATION_SHA256,
                    split_namespace=SplitNamespace.PHASE1_GATE,
                    root_seed=root_seed,
                    public_id_seed_sha256=fingerprint,
                    config_sha256="c" * 64,
                    generator_source_sha256="d" * 64,
                    episode_count=100_000,
                )
                provenance["leakage_audit"]["descriptor_sha256"] = (  # type: ignore[index]
                    audit_source_descriptor_sha256(descriptor)
                )
            elif name == "independent-reproducibility.json":
                value["source_payload_sha256"] = _independent_descriptor_sha256(
                    root_seed=root_seed,
                    public_id_seed_sha256=fingerprint,
                )

        artifacts[artifact_name] = _mutate_json(artifacts[artifact_name], rebind)


def test_complete_consistent_fixture_set_returns_one_canonical_success_object(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    paths = _write_artifacts(tmp_path, consistent_artifact_bytes)

    result = _verify(paths)

    assert result.model_dump(mode="json") == {
        "schema_version": "phase1-gate-verification-v1",
        "validation_episode_count": 10_000,
        "independent_episode_count": 100_000,
        "validation_manifest_payload_sha256": json.loads(
            consistent_artifact_bytes["validation.json"]
        )["payload_sha256"],
        "independent_corpus_sha256": "f" * 64,
        "foundation_model_calls": 0,
        "passed": True,
    }
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_phase1_gate_artifacts.py",
            "--validation",
            str(paths["validation.json"]),
            "--oracle",
            str(paths["oracle.json"]),
            "--leakage",
            str(paths["leakage.json"]),
            "--validation-reproducibility",
            str(paths["validation-reproducibility.json"]),
            "--independent-reproducibility",
            str(paths["independent-reproducibility.json"]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == result.model_dump(mode="json")
    assert completed.stdout.count("\n") == 1
    assert completed.stderr == ""


def test_verifier_refuses_a_missing_artifact(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    paths = _write_artifacts(tmp_path, consistent_artifact_bytes)
    paths["oracle.json"].unlink()

    with pytest.raises(ArtifactIntegrityError, match="oracle"):
        _verify(paths)


def test_verifier_refuses_a_tampered_manifest_envelope(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def tamper(value: dict[str, object]) -> None:
        value["payload"]["experiment_version"] = "tampered"  # type: ignore[index]

    artifacts["validation.json"] = _mutate_json(artifacts["validation.json"], tamper)

    with pytest.raises(ArtifactIntegrityError, match="validation"):
        _verify(_write_artifacts(tmp_path, artifacts))


def test_verifier_refuses_a_failed_inner_report(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    artifacts = dict(consistent_artifact_bytes)
    artifacts["oracle.json"] = _mutate_json(
        artifacts["oracle.json"], lambda value: value.__setitem__("passed", False)
    )

    with pytest.raises(ArtifactIntegrityError, match=r"oracle.*passed"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("plan_base_revision", "0" * 40),
        ("source_commit", "1" * 40),
        ("config_sha256", "2" * 64),
        ("generator_version", "other"),
    ],
)
def test_verifier_refuses_common_provenance_disagreement(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    field: str,
    replacement: str,
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        value["provenance"][field] = replacement  # type: ignore[index]

    artifacts["independent-reproducibility.json"] = _mutate_json(
        artifacts["independent-reproducibility.json"], mutate
    )

    with pytest.raises(ArtifactIntegrityError, match=r"provenance|schema"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize(
    ("artifact_name", "field", "replacement"),
    [
        ("validation-reproducibility.json", "sample_size", 1),
        ("validation-reproducibility.json", "verified_source_entries", 9_999),
        (
            "validation-reproducibility.json",
            "modes",
            ["forward", "reverse", "chunked"],
        ),
        ("validation-reproducibility.json", "chunk_sizes", []),
        ("validation-reproducibility.json", "chunk_sizes", [1, 31, 257]),
        ("validation-reproducibility.json", "python_hash_seeds", []),
        ("validation-reproducibility.json", "python_hash_seeds", [0, 2]),
        ("validation-reproducibility.json", "mismatch_count", 1),
        ("validation-reproducibility.json", "passed", False),
        ("independent-reproducibility.json", "sample_size", 999),
        ("independent-reproducibility.json", "verified_source_entries", 99_999),
        (
            "independent-reproducibility.json",
            "modes",
            ["forward", "reverse", "fresh_process"],
        ),
        ("independent-reproducibility.json", "chunk_sizes", []),
        ("independent-reproducibility.json", "chunk_sizes", [1, 31, 257]),
        ("independent-reproducibility.json", "python_hash_seeds", []),
        ("independent-reproducibility.json", "python_hash_seeds", [0, 2]),
        ("independent-reproducibility.json", "mismatch_count", 1),
        ("independent-reproducibility.json", "passed", False),
    ],
)
def test_verifier_refuses_nonfrozen_reproducibility_execution_matrix(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    artifact_name: str,
    field: str,
    replacement: object,
) -> None:
    """Every execution coordinate is evidence; declaring mode names is insufficient."""
    artifacts = dict(consistent_artifact_bytes)
    artifacts[artifact_name] = _mutate_json(
        artifacts[artifact_name],
        lambda value: value.__setitem__(field, replacement),
    )

    with pytest.raises(ArtifactIntegrityError, match="reproducibility"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize(
    "descriptor_updates",
    [
        pytest.param(None, id="arbitrary-payload-hash"),
        pytest.param({"allocation_id": "unrelated-allocation"}, id="allocation-id"),
        pytest.param({"allocation_sha256": "0" * 64}, id="allocation-hash"),
        pytest.param({"split_namespace": SplitNamespace.DEBUG}, id="namespace"),
        pytest.param({"root_seed": 47}, id="root-seed"),
        pytest.param(
            {"public_id_seed_sha256": public_id_seed_sha256(53)},
            id="public-id-seed",
        ),
        pytest.param({"config_sha256": "0" * 64}, id="config-hash"),
        pytest.param({"generator_source_sha256": "0" * 64}, id="generator-hash"),
    ],
)
def test_verifier_refuses_wrong_independent_reproducibility_source_descriptor(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    descriptor_updates: dict[str, object] | None,
) -> None:
    """A matching corpus hash cannot substitute for the sealed Task 14 source identity."""
    artifacts = dict(consistent_artifact_bytes)
    wrong_hash = (
        "0" * 64
        if descriptor_updates is None
        else _independent_descriptor_sha256(**descriptor_updates)
    )
    artifacts["independent-reproducibility.json"] = _mutate_json(
        artifacts["independent-reproducibility.json"],
        lambda value: value.__setitem__("source_payload_sha256", wrong_hash),
    )

    with pytest.raises(ArtifactIntegrityError, match="independent reproducibility"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize("scope", ["validation", "independent", "both"])
def test_verifier_refuses_coordinated_alternate_evidence_seeds(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    scope: str,
) -> None:
    """Internally consistent republishing cannot replace the predeclared evidence seeds."""
    artifacts = dict(consistent_artifact_bytes)
    if scope in {"validation", "both"}:
        _replace_validation_seeds(artifacts, root_seed=41, public_id_seed=43)
    if scope in {"independent", "both"}:
        _replace_independent_seeds(artifacts, root_seed=47, public_id_seed=53)

    with pytest.raises(ArtifactIntegrityError, match=r"seed|source provenance|validation"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("profile", "test"),
        ("allocation_id", "unrelated-allocation"),
        ("allocation_or_manifest_sha256", "0" * 64),
        ("config_sha256", "0" * 64),
        ("descriptor_sha256", "0" * 64),
        ("suite_path_denominators", {"unrelated:1": 100_000}),
        ("clock_scale_pair_counts", {"scale_0_1x": 1, "scale_10x": 1}),
    ],
)
def test_verifier_refuses_leakage_anchor_that_contradicts_gate_authority(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    field: str,
    replacement: object,
) -> None:
    """The source's report cannot replace the separately sealed gate authority."""
    artifacts = dict(consistent_artifact_bytes)

    def weaken(value: dict[str, object]) -> None:
        value["provenance"]["leakage_audit"][field] = replacement  # type: ignore[index]

    artifacts["leakage.json"] = _mutate_json(artifacts["leakage.json"], weaken)

    with pytest.raises(ArtifactIntegrityError, match="leakage"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda payload: payload["entries"][0].__setitem__(  # type: ignore[index,union-attr]
                "requested_path_length", 1
            ),
            id="wrong-path",
        ),
        pytest.param(
            lambda payload: [
                entry["coordinate"].__setitem__(  # type: ignore[index,union-attr]
                    "cohort_index",
                    entry["coordinate"]["cohort_index"] + 1,  # type: ignore[index,operator]
                )
                for entry in payload["entries"]  # type: ignore[union-attr]
            ],
            id="wrong-cohort-boundary",
        ),
        pytest.param(
            lambda payload: payload.__setitem__(  # type: ignore[union-attr]
                "entries",
                [
                    payload["entries"][1],  # type: ignore[index]
                    payload["entries"][0],  # type: ignore[index]
                    *payload["entries"][2:],  # type: ignore[index]
                ],
            ),
            id="noncanonical-coordinate-order",
        ),
        pytest.param(
            lambda payload: payload.__setitem__(  # type: ignore[union-attr]
                "experiment_version", "different-validation-recipe"
            ),
            id="wrong-experiment-version",
        ),
    ],
)
def test_verifier_refuses_nonfrozen_validation_allocation_recipe(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    mutation: Callable[[dict[str, object]], None],
) -> None:
    """A rehashed 10,000-entry manifest is not enough without the frozen recipe."""
    artifacts = dict(consistent_artifact_bytes)
    _mutate_validation_manifest(artifacts, mutation)

    with pytest.raises(ArtifactIntegrityError, match="validation"):
        _verify(_write_artifacts(tmp_path, artifacts))


def test_verifier_refuses_nonzero_foundation_calls_even_before_consistency_checks(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        value["provenance"]["foundation_model_calls"] = 1  # type: ignore[index]

    artifacts["oracle.json"] = _mutate_json(artifacts["oracle.json"], mutate)

    with pytest.raises(ArtifactIntegrityError, match=r"oracle.*schema"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize("failure", ["absent", "failed"])
def test_verifier_refuses_absent_or_failed_counterfactual_check(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    failure: str,
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        checks = value["counterfactual_checks"]  # type: ignore[assignment]
        if failure == "absent":
            checks.pop()  # type: ignore[union-attr]
        else:
            checks[-1]["decision_mismatch_count"] = 1  # type: ignore[index]
            checks[-1]["passed"] = False  # type: ignore[index]
            value["passed"] = False

    artifacts["leakage.json"] = _mutate_json(artifacts["leakage.json"], mutate)

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|counterfactual"):
        _verify(_write_artifacts(tmp_path, artifacts))


def test_verifier_refuses_an_incomplete_positive_control_family(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        controls = value["positive_controls"]  # type: ignore[assignment]
        controls.pop()  # type: ignore[union-attr]

    artifacts["leakage.json"] = _mutate_json(artifacts["leakage.json"], mutate)

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|positive.control"):
        _verify(_write_artifacts(tmp_path, artifacts))


def test_verifier_refuses_a_wrong_frozen_denominator(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    artifacts = dict(consistent_artifact_bytes)
    artifacts["oracle.json"] = _mutate_json(
        artifacts["oracle.json"],
        lambda value: value.__setitem__("verified_episode_count", 99_999),
    )

    with pytest.raises(ArtifactIntegrityError, match="denominator"):
        _verify(_write_artifacts(tmp_path, artifacts))


def test_verifier_refuses_unequal_independent_gate_corpus_hashes(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    artifacts = dict(consistent_artifact_bytes)
    artifacts["independent-reproducibility.json"] = _mutate_json(
        artifacts["independent-reproducibility.json"],
        lambda value: value.__setitem__("reference_corpus_sha256", "0" * 64),
    )

    with pytest.raises(ArtifactIntegrityError, match="corpus"):
        _verify(_write_artifacts(tmp_path, artifacts))


def test_script_emits_one_typed_error_without_traceback(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/verify_phase1_gate_artifacts.py",
            "--validation",
            str(missing),
            "--oracle",
            str(missing),
            "--leakage",
            str(missing),
            "--validation-reproducibility",
            str(missing),
            "--independent-reproducibility",
            str(missing),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert json.loads(completed.stderr)["code"] == "artifact_integrity_error"
    assert completed.stderr.count("\n") == 1
    assert "Traceback" not in completed.stderr
