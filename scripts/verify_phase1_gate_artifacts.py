"""Verify the five immutable Phase 1 gate artifacts without regeneration."""

import argparse
import sys
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.episode import CorpusDigestEntry, corpus_sha256
from silent_cascade.env.leakage import (
    NAMED_LEAK_INJECTORS,
    AuditSourceDescriptor,
    CounterfactualCheckId,
    LeakageAuditProfileName,
    LeakageReport,
    ShortcutFeatureGroup,
    ShortcutTask,
    audit_source_descriptor_sha256,
)
from silent_cascade.env.reproducibility import IndependentSourceDescriptor, ReproducibilityReport
from silent_cascade.env.services import OracleEvaluationReport
from silent_cascade.errors import ArtifactIntegrityError, ManifestError, SilentCascadeError
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.manifest import (
    EpisodeManifest,
    ManifestAccessClass,
    MatchedManifestCoordinate,
    load_manifest,
)
from silent_cascade.provenance import EvidenceProvenance
from silent_cascade.validation import StrictModel

_VALIDATION_EPISODE_COUNT = 10_000
_INDEPENDENT_EPISODE_COUNT = 100_000
_INDEPENDENT_VARIANT_COUNTS = (50_000, 25_000, 25_000)
_CLOCK_COUNTS = (5_000, 2_000)
_COUNTERFACTUAL_COUNTS = {
    CounterfactualCheckId.TERMINAL_DELAY_SWAP: 25_000,
    CounterfactualCheckId.PRESENTATION_PERMUTATION: 100_000,
    CounterfactualCheckId.PAIRED_CLOCK_SCALE: 7_000,
}
_GATE_DENOMINATORS = {
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
_REPRODUCIBILITY_MODES = ("forward", "reverse", "chunked", "fresh_process")
_REPRODUCIBILITY_SAMPLE_SIZE = 1_000
_REPRODUCIBILITY_CHUNK_SIZES = (1, 3, 7)
_REPRODUCIBILITY_PYTHON_HASH_SEEDS = (0, 1)
_VALIDATION_EXPERIMENT_VERSION = "v1"
_VALIDATION_COHORT_BLOCKS = ((0, 834, 2), (834, 1_667, 3), (1_667, 2_500, 4))
_PHASE1_GATE_ALLOCATION_SHA256 = "9e032f6993af9f2d53c1dde3a3e3e72e097cf143ae2e72d02609f2d2d6f1ce18"
_CLOCK_PAIR_COUNTS = {"scale_0_1x": 5_000, "scale_10x": 2_000}
_VALIDATION_ROOT_SEED = 2026083001
_VALIDATION_PUBLIC_ID_SEED = 2026083002
_VALIDATION_PUBLIC_ID_SEED_SHA256 = (
    "0454fca622eb08379a5d88ecbe0a5ef70f6a15e9ca7d333acecf840f2df38802"
)
_INDEPENDENT_ROOT_SEED = 2026083011
_INDEPENDENT_PUBLIC_ID_SEED_SHA256 = (
    "f21ac562825bfd96e875eedf90c8ba5ed09883ebe2c18449843b03acebec778a"
)


class Phase1GateVerificationResult(StrictModel):
    """Canonical summary proving the five files agree at their public boundaries."""

    schema_version: Literal["phase1-gate-verification-v1"]
    validation_episode_count: Literal[10_000]
    independent_episode_count: Literal[100_000]
    validation_manifest_payload_sha256: str
    independent_corpus_sha256: str
    foundation_model_calls: Literal[0]
    passed: Literal[True]


def _artifact_error(message: str, *, path: Path | None = None) -> ArtifactIntegrityError:
    context = {} if path is None else {"path": str(path)}
    return ArtifactIntegrityError(message, context=context)


def _load_report[ReportT: StrictModel](path: Path, model: type[ReportT], *, name: str) -> ReportT:
    try:
        raw = path.read_bytes()
        report = model.model_validate_json(raw)
    except (OSError, ValidationError, TypeError, ValueError) as error:
        raise _artifact_error(f"{name} artifact schema verification failed", path=path) from error
    if raw != canonical_json_bytes(report):
        raise _artifact_error(f"{name} artifact is not canonical", path=path)
    return report


def _load_validation(path: Path) -> EpisodeManifest:
    try:
        return load_manifest(path)
    except ManifestError as error:
        raise _artifact_error("validation artifact verification failed", path=path) from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _artifact_error(message)


def _common_provenance_key(provenance: EvidenceProvenance) -> tuple[object, ...]:
    return (
        provenance.plan_base_revision,
        provenance.source_commit,
        provenance.source_dirty,
        provenance.generator_version,
        provenance.config_sha256,
        provenance.generator_source,
        provenance.analysis_source,
    )


def _independent_source_key(provenance: EvidenceProvenance) -> tuple[object, ...]:
    return (
        provenance.generation_mode,
        provenance.allocation_id,
        provenance.split_namespace,
        provenance.root_seed,
        provenance.public_id_seed_sha256,
    )


def _require_zero_call_common_provenance(
    provenances: tuple[EvidenceProvenance, ...],
) -> None:
    _require(
        all(item.foundation_model_calls == 0 for item in provenances),
        "Phase 1 evidence records nonzero foundation-model calls",
    )
    _require(
        all(not item.source_dirty for item in provenances),
        "Phase 1 evidence provenance records a dirty source tree",
    )
    expected = _common_provenance_key(provenances[0])
    _require(
        all(_common_provenance_key(item) == expected for item in provenances[1:]),
        "Phase 1 plan/source/config/generator provenance disagrees",
    )


def _validation_path_length(cohort_index: int) -> int | None:
    for first, stop, path_length in _VALIDATION_COHORT_BLOCKS:
        if first <= cohort_index < stop:
            return path_length
    return None


def _require_validation_recipe(manifest: EpisodeManifest) -> None:
    _require(
        manifest.experiment_version == _VALIDATION_EXPERIMENT_VERSION,
        "validation artifact has the wrong frozen experiment version",
    )
    for entry_index, entry in enumerate(manifest.entries):
        coordinate = entry.coordinate
        expected_cohort_index, expected_member_index = divmod(entry_index, 4)
        _require(
            isinstance(coordinate, MatchedManifestCoordinate)
            and coordinate.cohort_index == expected_cohort_index
            and coordinate.member_index == expected_member_index
            and entry.requested_path_length == _validation_path_length(expected_cohort_index),
            "validation artifact does not match the ordered 834/833/833 allocation recipe",
        )


def _has_frozen_reproducibility_matrix(report: ReproducibilityReport) -> bool:
    return (
        report.sample_size == _REPRODUCIBILITY_SAMPLE_SIZE
        and report.modes == _REPRODUCIBILITY_MODES
        and report.chunk_sizes == _REPRODUCIBILITY_CHUNK_SIZES
        and report.python_hash_seeds == _REPRODUCIBILITY_PYTHON_HASH_SEEDS
        and report.mismatch_count == 0
    )


def _require_validation_artifacts(
    manifest: EpisodeManifest,
    reproducibility: ReproducibilityReport,
) -> str:
    _require(
        manifest.access_class is ManifestAccessClass.VALIDATION
        and manifest.suite is SuiteName.VALIDATION
        and manifest.provenance.generation_mode == "matched"
        and manifest.provenance.allocation_id == "validation-v1"
        and manifest.provenance.split_namespace is SplitNamespace.VALIDATION
        and manifest.provenance.root_seed == _VALIDATION_ROOT_SEED
        and manifest.public_id_seed == _VALIDATION_PUBLIC_ID_SEED
        and manifest.provenance.public_id_seed_sha256 == _VALIDATION_PUBLIC_ID_SEED_SHA256
        and manifest.episode_count == _VALIDATION_EPISODE_COUNT,
        "validation artifact does not have the frozen 10,000-episode recipe",
    )
    _require_validation_recipe(manifest)
    manifest_payload_sha256 = sha256_bytes(canonical_json_bytes(manifest))
    expected_corpus = corpus_sha256(
        (
            CorpusDigestEntry(entry.episode_public_id, entry.episode_sha256)
            for entry in manifest.entries
        ),
        expected_count=_VALIDATION_EPISODE_COUNT,
    )
    _require(
        reproducibility.passed
        and reproducibility.source_mode == "manifest"
        and _has_frozen_reproducibility_matrix(reproducibility)
        and reproducibility.verified_source_entries == _VALIDATION_EPISODE_COUNT
        and reproducibility.source_payload_sha256 == manifest_payload_sha256
        and reproducibility.reference_corpus_sha256 == expected_corpus
        and reproducibility.modes == _REPRODUCIBILITY_MODES,
        "validation reproducibility artifact is failed or inconsistent",
    )
    _require(
        reproducibility.provenance == manifest.provenance,
        "validation manifest and reproducibility provenance disagree",
    )
    return manifest_payload_sha256


def _require_oracle(oracle: OracleEvaluationReport) -> None:
    _require(oracle.passed, "oracle inner report passed flag is false")
    _require(
        oracle.source_mode == "phase1_gate"
        and oracle.requested_episode_count == _INDEPENDENT_EPISODE_COUNT
        and oracle.verified_episode_count == _INDEPENDENT_EPISODE_COUNT
        and (
            oracle.positive_count,
            oracle.safe_negative_count,
            oracle.disconnected_negative_count,
        )
        == _INDEPENDENT_VARIANT_COUNTS
        and oracle.oracle_successes == _INDEPENDENT_EPISODE_COUNT
        and oracle.random_positive.total == 50_000
        and oracle.random_negative.total == 50_000
        and oracle.random_positive.passed
        and oracle.random_negative.passed
        and (
            oracle.clock_0_1x_episode_count,
            oracle.clock_10x_episode_count,
        )
        == _CLOCK_COUNTS
        and oracle.suite_path_denominators == _GATE_DENOMINATORS,
        "oracle artifact has a wrong frozen denominator or failed diagnostic",
    )


def _require_leakage(leakage: LeakageReport) -> None:
    _require(leakage.passed, "leakage inner report passed flag is false")
    _require(
        leakage.generation_mode == "independent"
        and leakage.profile is LeakageAuditProfileName.PHASE1_GATE
        and leakage.episode_count == _INDEPENDENT_EPISODE_COUNT
        and leakage.randomization_block_count == 25_000
        and leakage.suite_path_denominators == _GATE_DENOMINATORS
        and bool(leakage.construction_checks)
        and all(leakage.construction_checks.values())
        and leakage.label_shuffled_control_passed,
        "leakage artifact has a wrong frozen denominator or failed check",
    )
    expected_probes = tuple(
        (task, group) for task in ShortcutTask for group in ShortcutFeatureGroup
    )
    observed_probes = tuple((probe.task, probe.feature_group) for probe in leakage.probes)
    _require(
        observed_probes == expected_probes
        and all(
            probe.passed
            and probe.optimizer_converged
            and not (
                probe.balanced_accuracy > probe.balanced_chance and probe.holm_adjusted_p < 0.01
            )
            for probe in leakage.probes
        ),
        "leakage clean-probe family is absent, incomplete, or failed",
    )
    checks = {item.check_id: item for item in leakage.counterfactual_checks}
    _require(
        set(checks) == set(CounterfactualCheckId),
        "leakage counterfactual family is absent or incomplete",
    )
    _require(
        all(
            checks[check_id].passed
            and checks[check_id].checked_pairs == expected_count
            and checks[check_id].decision_mismatch_count == 0
            and checks[check_id].temporal_mismatch_count == 0
            for check_id, expected_count in _COUNTERFACTUAL_COUNTS.items()
        ),
        "leakage counterfactual check failed or has a wrong denominator",
    )
    expected_controls = tuple(
        (injector.control_id, injector.target_task, injector.expected_detector_id)
        for injector in NAMED_LEAK_INJECTORS
    )
    observed_controls = tuple(
        (control.control_id, control.target_task, control.expected_detector_id)
        for control in leakage.positive_controls
    )
    _require(
        observed_controls == expected_controls
        and all(
            control.passed
            and control.expected_detector_id in control.observed_detector_ids
            and control.balanced_accuracy is not None
            and control.balanced_accuracy >= 0.95
            and control.holm_adjusted_p is not None
            and control.holm_adjusted_p < 0.01
            for control in leakage.positive_controls
        )
        and len({control.base_subset_corpus_sha256 for control in leakage.positive_controls}) == 1
        and len({control.split_membership_sha256 for control in leakage.positive_controls}) == 1,
        "leakage positive-control family is absent, incomplete, or failed",
    )
    anchor = leakage.provenance.leakage_audit
    _require(anchor is not None, "leakage provenance is missing its frozen audit authority")
    assert anchor is not None
    expected_descriptor = AuditSourceDescriptor(
        schema_version="leakage-source-v1",
        generation_mode="independent",
        allocation_id="phase1-independent-gate-v1",
        allocation_or_manifest_sha256=_PHASE1_GATE_ALLOCATION_SHA256,
        split_namespace=SplitNamespace.PHASE1_GATE,
        root_seed=leakage.provenance.root_seed,
        public_id_seed_sha256=leakage.provenance.public_id_seed_sha256,
        config_sha256=leakage.provenance.config_sha256,
        generator_source_sha256=leakage.provenance.generator_source.sha256,
        episode_count=_INDEPENDENT_EPISODE_COUNT,
    )
    _require(
        leakage.provenance.analysis_seeds
        == {"audit_seed": 2026083091, "positive_control_seed": 2026083092}
        and anchor.profile == LeakageAuditProfileName.PHASE1_GATE.value
        and anchor.allocation_id == leakage.provenance.allocation_id == "phase1-independent-gate-v1"
        and anchor.allocation_or_manifest_sha256 == _PHASE1_GATE_ALLOCATION_SHA256
        and anchor.config_sha256 == leakage.provenance.config_sha256
        and anchor.descriptor_sha256 == audit_source_descriptor_sha256(expected_descriptor)
        and anchor.suite_path_denominators == leakage.suite_path_denominators == _GATE_DENOMINATORS
        and anchor.clock_scale_pair_counts == _CLOCK_PAIR_COUNTS
        and anchor.episode_count == leakage.episode_count == _INDEPENDENT_EPISODE_COUNT,
        "leakage provenance contradicts its frozen audit authority",
    )


def _require_independent_reproducibility(report: ReproducibilityReport) -> None:
    _require(
        report.passed
        and report.source_mode == "independent_allocation"
        and _has_frozen_reproducibility_matrix(report)
        and report.verified_source_entries == _INDEPENDENT_EPISODE_COUNT
        and report.mismatch_count == 0,
        "independent reproducibility artifact is failed or has a wrong denominator",
    )
    expected_source = IndependentSourceDescriptor(
        schema_version="phase1-independent-source-v1",
        allocation_id="phase1-independent-gate-v1",
        allocation_sha256=_PHASE1_GATE_ALLOCATION_SHA256,
        split_namespace=SplitNamespace.PHASE1_GATE,
        root_seed=report.provenance.root_seed,
        public_id_seed_sha256=report.provenance.public_id_seed_sha256,
        config_sha256=report.provenance.config_sha256,
        generator_source_sha256=report.provenance.generator_source.sha256,
    )
    _require(
        report.source_payload_sha256 == sha256_bytes(canonical_json_bytes(expected_source)),
        "independent reproducibility source descriptor disagrees with frozen provenance",
    )


def verify_phase1_gate_artifacts(
    *,
    validation_path: Path,
    oracle_path: Path,
    leakage_path: Path,
    validation_reproducibility_path: Path,
    independent_reproducibility_path: Path,
) -> Phase1GateVerificationResult:
    """Strictly load and cross-check the five frozen Phase 1 evidence files."""
    manifest = _load_validation(validation_path)
    oracle = _load_report(oracle_path, OracleEvaluationReport, name="oracle")
    leakage = _load_report(leakage_path, LeakageReport, name="leakage")
    validation_reproducibility = _load_report(
        validation_reproducibility_path,
        ReproducibilityReport,
        name="validation reproducibility",
    )
    independent_reproducibility = _load_report(
        independent_reproducibility_path,
        ReproducibilityReport,
        name="independent reproducibility",
    )

    manifest_payload_sha256 = _require_validation_artifacts(manifest, validation_reproducibility)
    _require_oracle(oracle)
    _require_leakage(leakage)
    _require_independent_reproducibility(independent_reproducibility)

    provenances = (
        manifest.provenance,
        oracle.provenance,
        leakage.provenance,
        validation_reproducibility.provenance,
        independent_reproducibility.provenance,
    )
    _require_zero_call_common_provenance(provenances)
    independent_key = _independent_source_key(oracle.provenance)
    _require(
        all(
            _independent_source_key(item) == independent_key
            for item in (leakage.provenance, independent_reproducibility.provenance)
        )
        and independent_key
        == (
            "independent",
            "phase1-independent-gate-v1",
            SplitNamespace.PHASE1_GATE,
            _INDEPENDENT_ROOT_SEED,
            _INDEPENDENT_PUBLIC_ID_SEED_SHA256,
        ),
        "independent gate source provenance disagrees",
    )
    _require(
        oracle.corpus_sha256
        == leakage.corpus_hash
        == independent_reproducibility.reference_corpus_sha256,
        "independent gate corpus SHA-256 values disagree",
    )
    return Phase1GateVerificationResult(
        schema_version="phase1-gate-verification-v1",
        validation_episode_count=_VALIDATION_EPISODE_COUNT,
        independent_episode_count=_INDEPENDENT_EPISODE_COUNT,
        validation_manifest_payload_sha256=manifest_payload_sha256,
        independent_corpus_sha256=oracle.corpus_sha256,
        foundation_model_calls=0,
        passed=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the five immutable Phase 1 gate artifacts locally."
    )
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--leakage", type=Path, required=True)
    parser.add_argument("--validation-reproducibility", type=Path, required=True)
    parser.add_argument("--independent-reproducibility", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = verify_phase1_gate_artifacts(
            validation_path=arguments.validation,
            oracle_path=arguments.oracle,
            leakage_path=arguments.leakage,
            validation_reproducibility_path=arguments.validation_reproducibility,
            independent_reproducibility_path=arguments.independent_reproducibility,
        )
    except SilentCascadeError as error:
        sys.stderr.buffer.write(canonical_json_bytes(error.to_payload()) + b"\n")
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
