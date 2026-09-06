"""Verify the five immutable Phase 1 gate artifacts without regeneration."""

import argparse
import sys
import uuid
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.episode import CorpusDigestEntry, corpus_sha256
from silent_cascade.env.generator import (
    PHASE1_GATE_ALLOCATION,
    VALIDATION_ALLOCATION,
    CohortRequest,
    IndependentEpisodeRequest,
    independent_seed_tokens,
    iter_independent_requests,
    matched_seed_tokens,
)
from silent_cascade.env.leakage import (
    NAMED_LEAK_INJECTORS,
    AuditSourceDescriptor,
    CounterfactualCheckId,
    CounterfactualCheckResult,
    LeakageAuditProfileName,
    LeakageReport,
    PositiveControlResult,
    ShortcutFeatureGroup,
    ShortcutProbeResult,
    ShortcutTask,
    audit_source_descriptor_sha256,
)
from silent_cascade.env.reproducibility import (
    IndependentSourceDescriptor,
    ReproducibilityReport,
    independent_sample_membership_sha256,
    manifest_sample_membership_sha256,
    select_independent_reproducibility_sample,
    select_manifest_reproducibility_sample,
)
from silent_cascade.env.services import OracleEvaluationReport
from silent_cascade.errors import (
    ArtifactIntegrityError,
    ManifestError,
    ProvenanceError,
    SilentCascadeError,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.manifest import (
    EpisodeManifest,
    ManifestAccessClass,
    MatchedManifestCoordinate,
    load_manifest,
)
from silent_cascade.provenance import (
    AcceptedAttemptRun,
    ConstructionNamespaceBuilder,
    ConstructionNamespaceEvidence,
    EvidenceProvenance,
    authenticate_final_phase1_provenance,
    require_namespace_evidence_provenance,
)
from silent_cascade.rng import (
    IndependentPublicIdKey,
    PublicIdBatchKey,
    allocate_independent_public_id,
    allocate_public_ids,
)
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
_INDEPENDENT_PUBLIC_ID_SEED = 2026083012
_INDEPENDENT_PUBLIC_ID_SEED_SHA256 = (
    "f21ac562825bfd96e875eedf90c8ba5ed09883ebe2c18449843b03acebec778a"
)
_PRIMARY_CONFIG_SHA256 = "8eede957c7d69cc85d33bddcd1af69eb76bc5d7bba48bd1cbbe166bbb757ecd4"
_PHASE1_PERMUTATION_DENOMINATOR = 5_000
_PHASE1_FEATURE_DIMENSIONS = {
    ShortcutTask.POSITIVE_BINARY: (17, 256, 278, 4, 768, 194, 10, 33, 1_560),
    ShortcutTask.VARIANT_THREE_WAY: (17, 256, 278, 4, 768, 194, 10, 33, 1_560),
    ShortcutTask.POSITIVE_HAZARD_CLASS: (17, 256, 270, 4, 704, 194, 4, 33, 1_482),
}
_CONSTRUCTION_CHECK_IDS = (
    "provenance",
    "public_ids",
    "seed_tokens",
    "invariants",
    "finite_features",
    "two_pass_identity",
)
_CONSTRUCTION_CHECKS = {key: True for key in sorted(_CONSTRUCTION_CHECK_IDS)}
_FEATURE_SCHEMA_SHA256 = sha256_bytes(
    canonical_json_bytes(
        {
            "schema_version": "leakage-features-v1",
            "dimensions": (17, 256, 278, 4, 768, 194, 10, 33),
            "groups": [group.value for group in ShortcutFeatureGroup],
        }
    )
)


class Phase1GateVerificationResult(StrictModel):
    """Canonical summary proving the five files agree at their public boundaries."""

    schema_version: Literal["phase1-gate-verification-v2"]
    validation_episode_count: Literal[10_000]
    independent_episode_count: Literal[100_000]
    matched_accepted_draw_count: Literal[2_500]
    independent_accepted_draw_count: Literal[100_000]
    matched_public_id_seed: Literal[2026083002]
    independent_public_id_seed: Literal[2026083012]
    construction_token_count: Literal[750_000]
    public_id_count: Literal[117_000]
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


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _holm_adjusted_p_values(raw_p_values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = sorted(enumerate(raw_p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(raw_p_values)
    running = 0.0
    for rank, (index, raw_p) in enumerate(ordered):
        running = max(running, min(1.0, (len(raw_p_values) - rank) * raw_p))
        adjusted[index] = running
    return tuple(adjusted)


def _probe_primitive_types_are_exact(probe: ShortcutProbeResult) -> bool:
    return (
        type(probe.task) is ShortcutTask
        and type(probe.feature_group) is ShortcutFeatureGroup
        and type(probe.feature_dimension) is int
        and type(probe.train_examples) is int
        and type(probe.test_examples) is int
        and type(probe.train_class_counts) is dict
        and type(probe.test_class_counts) is dict
        and all(
            type(key) is str and type(value) is int
            for counts in (probe.train_class_counts, probe.test_class_counts)
            for key, value in counts.items()
        )
        and type(probe.raw_accuracy) is float
        and type(probe.balanced_accuracy) is float
        and type(probe.balanced_chance) is float
        and type(probe.raw_permutation_p) is float
        and type(probe.holm_adjusted_p) is float
        and type(probe.optimizer_iterations) is int
        and type(probe.optimizer_converged) is bool
        and type(probe.passed) is bool
    )


def _positive_control_primitive_types_are_exact(control: PositiveControlResult) -> bool:
    return (
        type(control.control_id) is str
        and type(control.target_task) is ShortcutTask
        and type(control.expected_detector_id) is str
        and type(control.observed_detector_ids) is tuple
        and all(type(value) is str for value in control.observed_detector_ids)
        and _is_sha256(control.base_subset_corpus_sha256)
        and _is_sha256(control.injected_corpus_sha256)
        and _is_sha256(control.split_membership_sha256)
        and type(control.balanced_accuracy) is float
        and type(control.holm_adjusted_p) is float
        and type(control.probes) is tuple
        and type(control.passed) is bool
    )


def _namespace_evidence_primitive_types_are_exact(
    evidence: ConstructionNamespaceEvidence,
) -> bool:
    integer_fields = (
        evidence.public_id_seed,
        evidence.accepted_draw_count,
        evidence.rejected_draw_count,
        evidence.generation_attempt_count,
        evidence.seed_token_count,
        evidence.seed_token_collision_count,
        evidence.base_public_id_count,
        evidence.clock_public_id_count,
        evidence.total_public_id_count,
        evidence.public_id_collision_count,
    )
    return (
        type(evidence) is ConstructionNamespaceEvidence
        and type(evidence.schema_version) is str
        and evidence.schema_version == "construction-namespace-evidence-v1"
        and type(evidence.generation_mode) is str
        and all(type(value) is int for value in integer_fields)
        and type(evidence.accepted_attempt_runs) is tuple
        and all(
            type(run) is AcceptedAttemptRun
            and type(run.first_draw_index) is int
            and type(run.draw_count) is int
            and type(run.accepted_attempt) is int
            for run in evidence.accepted_attempt_runs
        )
        and _is_sha256(evidence.seed_token_sequence_sha256)
        and _is_sha256(evidence.public_id_sequence_sha256)
    )


def _require_namespace_evidence(
    evidence: ConstructionNamespaceEvidence,
    provenance: EvidenceProvenance,
    *,
    generation_mode: Literal["matched", "independent"],
    public_id_seed: int,
    accepted_draw_count: int,
    seed_token_count: int,
    base_public_id_count: int,
    clock_public_id_count: int,
) -> None:
    try:
        require_namespace_evidence_provenance(evidence, provenance)
    except (TypeError, ValueError) as error:
        raise _artifact_error("construction namespace evidence contradicts provenance") from error
    _require(
        _namespace_evidence_primitive_types_are_exact(evidence)
        and evidence.generation_mode == generation_mode
        and evidence.public_id_seed == public_id_seed
        and evidence.accepted_draw_count == accepted_draw_count
        and evidence.seed_token_count == seed_token_count
        and evidence.base_public_id_count == base_public_id_count
        and evidence.clock_public_id_count == clock_public_id_count
        and evidence.total_public_id_count == base_public_id_count + clock_public_id_count,
        "construction namespace evidence has a wrong frozen seed or denominator",
    )


def _leakage_report_primitive_types_are_exact(leakage: LeakageReport) -> bool:
    digest_fields = (
        leakage.corpus_hash,
        leakage.feature_schema_hash,
        leakage.split_membership_hash,
        leakage.train_membership_hash,
        leakage.test_membership_hash,
    )
    return (
        type(leakage) is LeakageReport
        and type(leakage.schema_version) is str
        and leakage.schema_version == "leakage-report-v2"
        and type(leakage.provenance) is EvidenceProvenance
        and _namespace_evidence_primitive_types_are_exact(leakage.namespace_evidence)
        and type(leakage.generation_mode) is str
        and type(leakage.profile) is LeakageAuditProfileName
        and all(_is_sha256(value) for value in digest_fields)
        and type(leakage.episode_count) is int
        and type(leakage.randomization_block_count) is int
        and type(leakage.suite_path_denominators) is dict
        and all(
            type(key) is str and type(value) is int
            for key, value in leakage.suite_path_denominators.items()
        )
        and type(leakage.construction_check_ids) is tuple
        and all(type(value) is str for value in leakage.construction_check_ids)
        and type(leakage.construction_checks) is dict
        and all(
            type(key) is str and type(value) is bool
            for key, value in leakage.construction_checks.items()
        )
        and type(leakage.probes) is tuple
        and type(leakage.label_shuffled_probes) is tuple
        and type(leakage.positive_controls) is tuple
        and type(leakage.counterfactual_checks) is tuple
        and type(leakage.label_shuffled_control_passed) is bool
        and type(leakage.passed) is bool
    )


def _counterfactual_primitive_types_are_exact(result: CounterfactualCheckResult) -> bool:
    return (
        type(result) is CounterfactualCheckResult
        and type(result.check_id) is CounterfactualCheckId
        and type(result.checked_pairs) is int
        and type(result.decision_mismatch_count) is int
        and type(result.temporal_mismatch_count) is int
        and _is_sha256(result.result_payload_sha256)
        and type(result.passed) is bool
    )


def _expected_class_counts(
    task: ShortcutTask,
    episode_count: int,
) -> tuple[dict[str, int] | None, dict[str, int] | None]:
    if task is ShortcutTask.POSITIVE_BINARY:
        return (
            {"0": episode_count * 2 // 5, "1": episode_count * 2 // 5},
            {"0": episode_count // 10, "1": episode_count // 10},
        )
    if task is ShortcutTask.VARIANT_THREE_WAY:
        return (
            {"0": episode_count * 2 // 5, "1": episode_count // 5, "2": episode_count // 5},
            {"0": episode_count // 10, "1": episode_count // 20, "2": episode_count // 20},
        )
    if episode_count == 8_000:
        return (
            {str(index): 800 for index in range(4)},
            {str(index): 200 for index in range(4)},
        )
    return None, None


def _probe_metadata_is_consistent(
    probe: ShortcutProbeResult,
    *,
    episode_count: int,
) -> bool:
    if not _probe_primitive_types_are_exact(probe):
        return False
    group_index = tuple(ShortcutFeatureGroup).index(probe.feature_group)
    task_examples = (
        episode_count // 2 if probe.task is ShortcutTask.POSITIVE_HAZARD_CLASS else episode_count
    )
    expected_train = task_examples * 4 // 5
    expected_test = task_examples - expected_train
    expected_labels = {
        ShortcutTask.POSITIVE_BINARY: {"0", "1"},
        ShortcutTask.VARIANT_THREE_WAY: {"0", "1", "2"},
        ShortcutTask.POSITIVE_HAZARD_CLASS: {"0", "1", "2", "3"},
    }[probe.task]
    expected_chance = {
        ShortcutTask.POSITIVE_BINARY: 0.5,
        ShortcutTask.VARIANT_THREE_WAY: 1.0 / 3.0,
        ShortcutTask.POSITIVE_HAZARD_CLASS: 0.25,
    }[probe.task]
    exact_train, exact_test = _expected_class_counts(probe.task, episode_count)
    scaled_p = probe.raw_permutation_p * _PHASE1_PERMUTATION_DENOMINATOR
    permutation_numerator = round(scaled_p)
    bounded_values = (
        probe.raw_accuracy,
        probe.balanced_accuracy,
        probe.balanced_chance,
        probe.raw_permutation_p,
        probe.holm_adjusted_p,
    )
    return (
        all(0.0 <= value <= 1.0 for value in bounded_values)
        and probe.balanced_chance == expected_chance
        and probe.feature_dimension == _PHASE1_FEATURE_DIMENSIONS[probe.task][group_index]
        and probe.train_examples == expected_train
        and probe.test_examples == expected_test
        and set(probe.train_class_counts) == expected_labels
        and set(probe.test_class_counts) == expected_labels
        and all(value >= 0 for value in probe.train_class_counts.values())
        and all(value >= 0 for value in probe.test_class_counts.values())
        and sum(probe.train_class_counts.values()) == probe.train_examples
        and sum(probe.test_class_counts.values()) == probe.test_examples
        and min(probe.train_class_counts.values()) >= 200
        and min(probe.test_class_counts.values()) >= 200
        and (exact_train is None or probe.train_class_counts == exact_train)
        and (exact_test is None or probe.test_class_counts == exact_test)
        and 0 <= probe.optimizer_iterations <= 500
        and probe.optimizer_converged
        and 1 <= permutation_numerator <= _PHASE1_PERMUTATION_DENOMINATOR
        and probe.raw_permutation_p == permutation_numerator / _PHASE1_PERMUTATION_DENOMINATOR
    )


def _probe_family_is_consistent(
    probes: tuple[ShortcutProbeResult, ...],
    *,
    episode_count: int,
) -> bool:
    expected = tuple((task, group) for task in ShortcutTask for group in ShortcutFeatureGroup)
    if tuple((probe.task, probe.feature_group) for probe in probes) != expected:
        return False
    adjusted = _holm_adjusted_p_values(tuple(probe.raw_permutation_p for probe in probes))
    signatures: dict[ShortcutTask, tuple[object, ...]] = {}
    for index, probe in enumerate(probes):
        if (
            not _probe_metadata_is_consistent(probe, episode_count=episode_count)
            or probe.holm_adjusted_p != adjusted[index]
            or probe.passed
            != (not (probe.balanced_accuracy > probe.balanced_chance and adjusted[index] < 0.01))
        ):
            return False
        signature = (
            probe.train_examples,
            probe.test_examples,
            probe.train_class_counts,
            probe.test_class_counts,
        )
        if signatures.setdefault(probe.task, signature) != signature:
            return False
    return True


def _positive_control_is_consistent(control: PositiveControlResult) -> bool:
    if not _positive_control_primitive_types_are_exact(control):
        return False
    expected_family = tuple((control.target_task, group) for group in ShortcutFeatureGroup)
    observed_family = tuple((probe.task, probe.feature_group) for probe in control.probes)
    if observed_family != expected_family or any(
        not _probe_metadata_is_consistent(probe, episode_count=8_000) for probe in control.probes
    ):
        return False
    adjusted = _holm_adjusted_p_values(tuple(probe.raw_permutation_p for probe in control.probes))
    if any(
        not probe.optimizer_converged
        or probe.holm_adjusted_p != adjusted[index]
        or probe.passed
        != (not (probe.balanced_accuracy > probe.balanced_chance and adjusted[index] < 0.01))
        for index, probe in enumerate(control.probes)
    ):
        return False
    derived_observed = tuple(
        f"{probe.task.value}:{probe.feature_group.value}"
        for index, probe in enumerate(control.probes)
        if probe.balanced_accuracy >= 0.95
        and probe.raw_permutation_p <= 0.05
        and adjusted[index] < 0.01
    )
    expected_group = ShortcutFeatureGroup(control.expected_detector_id.split(":", 1)[1])
    expected_probe = next(
        (probe for probe in control.probes if probe.feature_group is expected_group),
        None,
    )
    derived_pass = control.expected_detector_id in derived_observed
    return (
        control.observed_detector_ids == derived_observed
        and expected_probe is not None
        and control.balanced_accuracy == expected_probe.balanced_accuracy
        and control.holm_adjusted_p == expected_probe.holm_adjusted_p
        and control.passed is derived_pass
    )


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


def _accepted_attempts(
    evidence: ConstructionNamespaceEvidence,
) -> tuple[int, ...]:
    attempts = tuple(
        run.accepted_attempt
        for run in evidence.accepted_attempt_runs
        for _ in range(run.draw_count)
    )
    _require(
        len(attempts) == evidence.accepted_draw_count,
        "construction attempt runs do not cover the declared corpus",
    )
    return attempts


def _reconstruct_matched_namespace(
    manifest: EpisodeManifest,
) -> tuple[ConstructionNamespaceEvidence, set[bytes], set[bytes]]:
    builder = ConstructionNamespaceBuilder(
        generation_mode="matched",
        public_id_seed=_VALIDATION_PUBLIC_ID_SEED,
        accepted_draw_count=2_500,
        clock_public_id_count=0,
    )
    token_values: set[bytes] = set()
    public_id_values: set[bytes] = set()
    for cohort_index, request in enumerate(
        CohortRequest(
            split_namespace=VALIDATION_ALLOCATION.split_namespace,
            suite=block.suite,
            root_seed=_VALIDATION_ROOT_SEED,
            cohort_index=index,
            requested_path_length=block.requested_path_length,
        )
        for block in VALIDATION_ALLOCATION.blocks
        for index in range(
            block.first_cohort_index,
            block.first_cohort_index + block.cohort_count,
        )
    ):
        members = manifest.entries[cohort_index * 4 : cohort_index * 4 + 4]
        attempt = members[0].accepted_attempt
        _require(
            len(members) == 4 and all(member.accepted_attempt == attempt for member in members),
            "validation cohort accepted attempts disagree",
        )
        expected_ids = allocate_public_ids(
            PublicIdBatchKey(
                generator_version="ofd-v1",
                split_namespace=request.split_namespace,
                suite=request.suite,
                public_id_seed=_VALIDATION_PUBLIC_ID_SEED,
                cohort_index=request.cohort_index,
                accepted_attempt=attempt,
            )
        )
        actual_ids = tuple(member.episode_public_id for member in members)
        _require(
            actual_ids == expected_ids,
            "validation public ID is not derivable from its recipe",
        )
        tokens = matched_seed_tokens(request, attempt)
        builder.add_draw(attempt, tokens, actual_ids)
        token_values.update(bytes.fromhex(token) for token in tokens)
        public_id_values.update(uuid.UUID(public_id).bytes for public_id in actual_ids)
    return builder.finalize(), token_values, public_id_values


def _independent_clock_public_ids(
    requests: tuple[IndependentEpisodeRequest, ...],
    attempts: tuple[int, ...],
) -> tuple[str, ...]:
    request_attempts = {
        (request.suite, request.requested_path_length, request.episode_index): attempt
        for request, attempt in zip(requests, attempts, strict=True)
    }
    identifiers: list[str] = []
    for suite, count_field in (
        (SuiteName.CLOCK_SCALE_0_1X, "scale_0_1x_episode_count"),
        (SuiteName.CLOCK_SCALE_10X, "scale_10x_episode_count"),
    ):
        for block in PHASE1_GATE_ALLOCATION.clock_blocks:
            for offset in range(getattr(block, count_field)):
                episode_index = block.source_first_episode_index + offset
                key = (SuiteName.IID_PRIMARY, block.requested_path_length, episode_index)
                attempt = request_attempts.get(key)
                _require(attempt is not None, "independent clock parent is absent")
                identifiers.append(
                    allocate_independent_public_id(
                        IndependentPublicIdKey(
                            generator_version="ofd-v1",
                            split_namespace=SplitNamespace.PHASE1_GATE,
                            suite=suite,
                            public_id_seed=_INDEPENDENT_PUBLIC_ID_SEED,
                            episode_index=episode_index,
                            accepted_attempt=attempt,
                        )
                    )
                )
    return tuple(identifiers)


def _reconstruct_independent_namespace(
    evidence: ConstructionNamespaceEvidence,
    *,
    matched_tokens: set[bytes],
    matched_public_ids: set[bytes],
) -> ConstructionNamespaceEvidence:
    requests = tuple(iter_independent_requests(PHASE1_GATE_ALLOCATION, _INDEPENDENT_ROOT_SEED))
    attempts = _accepted_attempts(evidence)
    _require(
        len(requests) == len(attempts) == _INDEPENDENT_EPISODE_COUNT,
        "independent namespace draw count is incomplete",
    )
    clock_ids = _independent_clock_public_ids(requests, attempts)
    builder = ConstructionNamespaceBuilder(
        generation_mode="independent",
        public_id_seed=_INDEPENDENT_PUBLIC_ID_SEED,
        accepted_draw_count=_INDEPENDENT_EPISODE_COUNT,
        clock_public_id_count=len(clock_ids),
    )
    for request, attempt in zip(requests, attempts, strict=True):
        tokens = independent_seed_tokens(request, attempt)
        public_id = allocate_independent_public_id(
            IndependentPublicIdKey(
                generator_version="ofd-v1",
                split_namespace=request.split_namespace,
                suite=request.suite,
                public_id_seed=_INDEPENDENT_PUBLIC_ID_SEED,
                episode_index=request.episode_index,
                accepted_attempt=attempt,
            )
        )
        _require(
            not any(bytes.fromhex(token) in matched_tokens for token in tokens),
            "matched and independent construction tokens collide",
        )
        _require(
            uuid.UUID(public_id).bytes not in matched_public_ids,
            "matched and independent base public IDs collide",
        )
        builder.add_draw(attempt, tokens, (public_id,))
    for public_id in clock_ids:
        _require(
            uuid.UUID(public_id).bytes not in matched_public_ids,
            "matched and independent clock public IDs collide",
        )
        builder.add_clock_public_id(public_id)
    return builder.finalize()


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
) -> tuple[str, set[bytes], set[bytes]]:
    _require_namespace_evidence(
        reproducibility.namespace_evidence,
        reproducibility.provenance,
        generation_mode="matched",
        public_id_seed=_VALIDATION_PUBLIC_ID_SEED,
        accepted_draw_count=2_500,
        seed_token_count=50_000,
        base_public_id_count=10_000,
        clock_public_id_count=0,
    )
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
    selected = select_manifest_reproducibility_sample(
        manifest,
        manifest_payload_sha256,
        _REPRODUCIBILITY_SAMPLE_SIZE,
    )
    expected_membership = manifest_sample_membership_sha256(
        manifest_payload_sha256,
        selected,
    )
    _require(
        reproducibility.passed
        and reproducibility.source_mode == "manifest"
        and _has_frozen_reproducibility_matrix(reproducibility)
        and reproducibility.verified_source_entries == _VALIDATION_EPISODE_COUNT
        and reproducibility.source_payload_sha256 == manifest_payload_sha256
        and reproducibility.reference_corpus_sha256 == expected_corpus
        and reproducibility.sample_membership_sha256 == expected_membership
        and reproducibility.modes == _REPRODUCIBILITY_MODES,
        "validation reproducibility artifact is failed or inconsistent",
    )
    _require(
        reproducibility.provenance == manifest.provenance,
        "validation manifest and reproducibility provenance disagree",
    )
    try:
        namespace, tokens, public_ids = _reconstruct_matched_namespace(manifest)
    except (TypeError, ValueError, ProvenanceError) as error:
        raise _artifact_error("validation namespace reconstruction failed") from error
    _require(
        reproducibility.namespace_evidence == namespace,
        "validation namespace evidence is not derivable from the manifest",
    )
    return manifest_payload_sha256, tokens, public_ids


def _require_oracle(oracle: OracleEvaluationReport) -> None:
    try:
        reparsed = OracleEvaluationReport.model_validate(oracle.model_dump(mode="python"))
    except ValidationError as error:
        raise _artifact_error("oracle derived report validation failed") from error
    _require(reparsed == oracle, "oracle report is not its strict derived form")
    _require_namespace_evidence(
        oracle.namespace_evidence,
        oracle.provenance,
        generation_mode="independent",
        public_id_seed=_INDEPENDENT_PUBLIC_ID_SEED,
        accepted_draw_count=_INDEPENDENT_EPISODE_COUNT,
        seed_token_count=700_000,
        base_public_id_count=_INDEPENDENT_EPISODE_COUNT,
        clock_public_id_count=sum(_CLOCK_COUNTS),
    )
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
        and oracle.oracle_failures == 0
        and oracle.invariant_failures == 0
        and oracle.oracle_ambiguities == 0
        and oracle.seed_token_collisions == 0
        and oracle.public_id_collisions == 0
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
    _require_namespace_evidence(
        leakage.namespace_evidence,
        leakage.provenance,
        generation_mode="independent",
        public_id_seed=_INDEPENDENT_PUBLIC_ID_SEED,
        accepted_draw_count=_INDEPENDENT_EPISODE_COUNT,
        seed_token_count=700_000,
        base_public_id_count=_INDEPENDENT_EPISODE_COUNT,
        clock_public_id_count=sum(_CLOCK_COUNTS),
    )
    _require(
        _leakage_report_primitive_types_are_exact(leakage),
        "leakage outer report types or containers are invalid",
    )
    _require(
        type(leakage.passed) is bool and leakage.passed,
        "leakage inner report passed flag is false or mistyped",
    )
    _require(
        leakage.generation_mode == "independent"
        and leakage.profile is LeakageAuditProfileName.PHASE1_GATE
        and leakage.episode_count == _INDEPENDENT_EPISODE_COUNT
        and leakage.randomization_block_count == 25_000
        and leakage.suite_path_denominators == _GATE_DENOMINATORS
        and type(leakage.construction_checks) is dict
        and leakage.construction_check_ids == _CONSTRUCTION_CHECK_IDS
        and tuple(leakage.construction_checks) == tuple(sorted(_CONSTRUCTION_CHECK_IDS))
        and leakage.construction_checks == _CONSTRUCTION_CHECKS
        and all(type(value) is bool for value in leakage.construction_checks.values())
        and type(leakage.label_shuffled_control_passed) is bool
        and leakage.label_shuffled_control_passed
        and type(leakage.feature_schema_hash) is str
        and leakage.feature_schema_hash == _FEATURE_SCHEMA_SHA256,
        "leakage artifact has a wrong frozen denominator or failed check",
    )
    _require(
        _probe_family_is_consistent(
            leakage.probes,
            episode_count=_INDEPENDENT_EPISODE_COUNT,
        )
        and all(probe.passed for probe in leakage.probes),
        "leakage clean-probe family is absent, incomplete, or failed",
    )
    _require(
        _probe_family_is_consistent(
            leakage.label_shuffled_probes,
            episode_count=_INDEPENDENT_EPISODE_COUNT,
        )
        and all(probe.passed for probe in leakage.label_shuffled_probes)
        and leakage.label_shuffled_control_passed,
        "leakage label-shuffled probe family is absent, incomplete, or failed",
    )
    expected_counterfactuals = tuple(_COUNTERFACTUAL_COUNTS.items())
    _require(
        len(leakage.counterfactual_checks) == len(expected_counterfactuals)
        and all(
            _counterfactual_primitive_types_are_exact(result)
            and result.check_id is expected_id
            and result.checked_pairs == expected_count
            and result.decision_mismatch_count == 0
            and result.temporal_mismatch_count == 0
            and result.passed
            for result, (expected_id, expected_count) in zip(
                leakage.counterfactual_checks,
                expected_counterfactuals,
                strict=True,
            )
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
            _positive_control_is_consistent(control) and control.passed
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
    _require_namespace_evidence(
        report.namespace_evidence,
        report.provenance,
        generation_mode="independent",
        public_id_seed=_INDEPENDENT_PUBLIC_ID_SEED,
        accepted_draw_count=_INDEPENDENT_EPISODE_COUNT,
        seed_token_count=700_000,
        base_public_id_count=_INDEPENDENT_EPISODE_COUNT,
        clock_public_id_count=sum(_CLOCK_COUNTS),
    )
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
    requests = tuple(iter_independent_requests(PHASE1_GATE_ALLOCATION, _INDEPENDENT_ROOT_SEED))
    selected = select_independent_reproducibility_sample(
        requests,
        report.source_payload_sha256,
        _REPRODUCIBILITY_SAMPLE_SIZE,
    )
    _require(
        report.sample_membership_sha256
        == independent_sample_membership_sha256(report.source_payload_sha256, selected),
        "independent reproducibility sample membership is not derivable",
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

    manifest_payload_sha256, matched_tokens, matched_public_ids = _require_validation_artifacts(
        manifest, validation_reproducibility
    )
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
    _require(
        all(item.config_sha256 == _PRIMARY_CONFIG_SHA256 for item in provenances),
        "Phase 1 evidence does not use the frozen primary configuration",
    )
    for provenance in provenances:
        try:
            authenticate_final_phase1_provenance(provenance, repo_root=Path.cwd())
        except (TypeError, ProvenanceError) as error:
            raise _artifact_error("historical Git provenance authentication failed") from error
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
    _require(
        oracle.namespace_evidence
        == leakage.namespace_evidence
        == independent_reproducibility.namespace_evidence,
        "independent namespace evidence disagrees across reports",
    )
    try:
        independent_namespace = _reconstruct_independent_namespace(
            oracle.namespace_evidence,
            matched_tokens=matched_tokens,
            matched_public_ids=matched_public_ids,
        )
    except (TypeError, ValueError, ProvenanceError) as error:
        raise _artifact_error("independent namespace reconstruction failed") from error
    _require(
        independent_namespace == oracle.namespace_evidence,
        "independent namespace evidence is not derivable from frozen coordinates",
    )
    construction_token_count = (
        validation_reproducibility.namespace_evidence.seed_token_count
        + independent_namespace.seed_token_count
    )
    public_id_count = (
        validation_reproducibility.namespace_evidence.total_public_id_count
        + independent_namespace.total_public_id_count
    )
    _require(
        construction_token_count == 750_000 and public_id_count == 117_000,
        "combined Phase 1 namespace counts are not exact",
    )
    return Phase1GateVerificationResult(
        schema_version="phase1-gate-verification-v2",
        validation_episode_count=_VALIDATION_EPISODE_COUNT,
        independent_episode_count=_INDEPENDENT_EPISODE_COUNT,
        matched_accepted_draw_count=2_500,
        independent_accepted_draw_count=_INDEPENDENT_EPISODE_COUNT,
        matched_public_id_seed=_VALIDATION_PUBLIC_ID_SEED,
        independent_public_id_seed=_INDEPENDENT_PUBLIC_ID_SEED,
        construction_token_count=construction_token_count,
        public_id_count=public_id_count,
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
