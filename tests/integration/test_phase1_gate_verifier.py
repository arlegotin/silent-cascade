"""Task 17 contracts for local, read-only Phase 1 evidence verification."""

import json
import subprocess
import sys
from collections.abc import Callable
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from runpy import run_path

import pytest
from pydantic import ValidationError

from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.episode import CorpusDigestEntry, corpus_sha256
from silent_cascade.env.generator import (
    PHASE1_GATE_ALLOCATION,
    VALIDATION_ALLOCATION,
    CohortRequest,
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
    _independent_clock_public_ids,
    independent_sample_membership_sha256,
    manifest_sample_membership_sha256,
    select_independent_reproducibility_sample,
    select_manifest_reproducibility_sample,
)
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
    TASK14_ANALYSIS_SOURCE_PATHS,
    ConstructionNamespaceBuilder,
    ConstructionNamespaceEvidence,
    EvidenceProvenance,
    LeakageAuditEvidenceAnchor,
    SourceTreeFingerprint,
    public_id_seed_sha256,
    source_tree_sha256_at_revision,
)
from silent_cascade.rng import (
    IndependentPublicIdKey,
    PublicIdBatchKey,
    allocate_independent_public_id,
    allocate_public_ids,
)

_VERIFIER = run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/verify_phase1_gate_artifacts.py")
)
verify_phase1_gate_artifacts = _VERIFIER["verify_phase1_gate_artifacts"]
_require_leakage = _VERIFIER["_require_leakage"]
Phase1GateVerificationResult = _VERIFIER["Phase1GateVerificationResult"]

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
PRIMARY_CONFIG_SHA256 = "8eede957c7d69cc85d33bddcd1af69eb76bc5d7bba48bd1cbbe166bbb757ecd4"
PHASE1_FEATURE_DIMENSIONS = {
    ShortcutTask.POSITIVE_BINARY: (17, 256, 278, 4, 768, 194, 10, 33, 1_560),
    ShortcutTask.VARIANT_THREE_WAY: (17, 256, 278, 4, 768, 194, 10, 33, 1_560),
    ShortcutTask.POSITIVE_HAZARD_CLASS: (17, 256, 270, 4, 704, 194, 4, 33, 1_482),
}
LEAKAGE_CONSTRUCTION_CHECK_IDS = (
    "provenance",
    "public_ids",
    "seed_tokens",
    "invariants",
    "finite_features",
    "two_pass_identity",
)
LEAKAGE_CONSTRUCTION_CHECKS = {key: True for key in sorted(LEAKAGE_CONSTRUCTION_CHECK_IDS)}
LEAKAGE_FEATURE_SCHEMA_SHA256 = sha256_bytes(
    canonical_json_bytes(
        {
            "schema_version": "leakage-features-v1",
            "dimensions": (17, 256, 278, 4, 768, 194, 10, 33),
            "groups": [group.value for group in ShortcutFeatureGroup],
        }
    )
)


class _GenerationMode(StrEnum):
    INDEPENDENT = "independent"


def _probe_workload(
    task: ShortcutTask,
    *,
    episode_count: int,
) -> tuple[int, int, dict[str, int], dict[str, int]]:
    if task is ShortcutTask.POSITIVE_BINARY:
        return (
            episode_count * 4 // 5,
            episode_count // 5,
            {"0": episode_count * 2 // 5, "1": episode_count * 2 // 5},
            {"0": episode_count // 10, "1": episode_count // 10},
        )
    if task is ShortcutTask.VARIANT_THREE_WAY:
        return (
            episode_count * 4 // 5,
            episode_count // 5,
            {"0": episode_count * 2 // 5, "1": episode_count // 5, "2": episode_count // 5},
            {"0": episode_count // 10, "1": episode_count // 20, "2": episode_count // 20},
        )
    positive_count = episode_count // 2
    return (
        positive_count * 4 // 5,
        positive_count // 5,
        {str(index): positive_count // 5 for index in range(4)},
        {str(index): positive_count // 20 for index in range(4)},
    )


def _provenance(*, independent: bool) -> EvidenceProvenance:
    repository = Path.cwd()
    head = subprocess.run(
        ("git", "rev-parse", "--verify", "HEAD^{commit}"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_commit = subprocess.run(
        (
            "git",
            "log",
            "-1",
            "--format=%H",
            head,
            "--",
            *PHASE1_ANALYSIS_SOURCE_PATHS,
        ),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    plan_base = subprocess.run(
        (
            "git",
            "log",
            "-1",
            "--format=%H",
            source_commit,
            "--",
            "docs/superpowers/plans/2026-08-30-phase-1-generator-oracle.md",
        ),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return EvidenceProvenance(
        schema_version="phase1-evidence-provenance-v1",
        plan_base_revision=plan_base,
        source_commit=source_commit,
        source_dirty=False,
        generator_version="ofd-v1",
        generation_mode="independent" if independent else "matched",
        allocation_id="phase1-independent-gate-v1" if independent else "validation-v1",
        split_namespace=(SplitNamespace.PHASE1_GATE if independent else SplitNamespace.VALIDATION),
        config_sha256=PRIMARY_CONFIG_SHA256,
        generator_source=SourceTreeFingerprint(
            frame_version="sc-source-tree-v1",
            scope="generator",
            paths=GENERATOR_SOURCE_PATHS,
            sha256=source_tree_sha256_at_revision(
                repository, source_commit, GENERATOR_SOURCE_PATHS
            ),
        ),
        analysis_source=SourceTreeFingerprint(
            frame_version="sc-source-tree-v1",
            scope="phase1_analysis",
            paths=PHASE1_ANALYSIS_SOURCE_PATHS,
            sha256=source_tree_sha256_at_revision(
                repository, source_commit, PHASE1_ANALYSIS_SOURCE_PATHS
            ),
        ),
        root_seed=2026083011 if independent else 2026083001,
        public_id_seed_sha256=public_id_seed_sha256(2026083012 if independent else 2026083002),
    )


@lru_cache(maxsize=2)
def _namespace(*, independent: bool) -> ConstructionNamespaceEvidence:
    accepted_draws = 100_000 if independent else 2_500
    builder = ConstructionNamespaceBuilder(
        generation_mode="independent" if independent else "matched",
        public_id_seed=2026083012 if independent else 2026083002,
        accepted_draw_count=accepted_draws,
        clock_public_id_count=7_000 if independent else 0,
    )
    if independent:
        requests = tuple(iter_independent_requests(PHASE1_GATE_ALLOCATION, 2026083011))
        attempts: dict[tuple[str, int, int, int, int], int] = {}
        for request in requests:
            public_id = allocate_independent_public_id(
                IndependentPublicIdKey(
                    "ofd-v1",
                    request.split_namespace,
                    request.suite,
                    2026083012,
                    request.episode_index,
                    0,
                )
            )
            builder.add_draw(0, independent_seed_tokens(request, 0), (public_id,))
            attempts[
                (
                    request.suite.value,
                    request.requested_path_length,
                    request.episode_index,
                    request.allocation_quartet_index,
                    request.quartet_member_index,
                )
            ] = 0
        for public_id in _independent_clock_public_ids(
            PHASE1_GATE_ALLOCATION,
            requests,
            2026083012,
            attempts,
        ):
            builder.add_clock_public_id(public_id)
    else:
        for block in VALIDATION_ALLOCATION.blocks:
            for cohort_index in range(
                block.first_cohort_index,
                block.first_cohort_index + block.cohort_count,
            ):
                request = CohortRequest(
                    SplitNamespace.VALIDATION,
                    block.suite,
                    2026083001,
                    cohort_index,
                    block.requested_path_length,
                )
                public_ids = allocate_public_ids(
                    PublicIdBatchKey(
                        "ofd-v1",
                        SplitNamespace.VALIDATION,
                        block.suite,
                        2026083002,
                        cohort_index,
                        0,
                    )
                )
                builder.add_draw(0, matched_seed_tokens(request, 0), public_ids)
    return builder.finalize()


def _validation_manifest() -> EpisodeManifest:
    provenance = _provenance(independent=False)
    entries: list[EpisodeManifestEntry] = []
    for block in VALIDATION_ALLOCATION.blocks:
        for cohort_index in range(
            block.first_cohort_index,
            block.first_cohort_index + block.cohort_count,
        ):
            public_ids = allocate_public_ids(
                PublicIdBatchKey(
                    "ofd-v1",
                    SplitNamespace.VALIDATION,
                    block.suite,
                    2026083002,
                    cohort_index,
                    0,
                )
            )
            entries.extend(
                EpisodeManifestEntry(
                    episode_public_id=public_id,
                    split_namespace=SplitNamespace.VALIDATION,
                    suite=SuiteName.VALIDATION,
                    coordinate=MatchedManifestCoordinate(
                        cohort_index=cohort_index,
                        member_index=member_index,
                    ),
                    requested_path_length=block.requested_path_length,
                    accepted_attempt=0,
                    episode_sha256=f"{cohort_index * 4 + member_index + 1:064x}",
                )
                for member_index, public_id in enumerate(public_ids)
            )
    return EpisodeManifest(
        schema_version=1,
        experiment_version="v1",
        access_class=ManifestAccessClass.VALIDATION,
        provenance=provenance,
        suite=SuiteName.VALIDATION,
        public_id_seed=2026083002,
        episode_count=10_000,
        entries=tuple(entries),
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
            holm_adjusted_p=9 * 0.0002,
            probes=tuple(
                ShortcutProbeResult(
                    task=injector.target_task,
                    feature_group=group,
                    feature_dimension=PHASE1_FEATURE_DIMENSIONS[injector.target_task][
                        tuple(ShortcutFeatureGroup).index(group)
                    ],
                    train_examples=_probe_workload(injector.target_task, episode_count=8_000)[0],
                    test_examples=_probe_workload(injector.target_task, episode_count=8_000)[1],
                    train_class_counts=_probe_workload(injector.target_task, episode_count=8_000)[
                        2
                    ],
                    test_class_counts=_probe_workload(injector.target_task, episode_count=8_000)[3],
                    raw_accuracy=(
                        1.0
                        if group.value == injector.expected_detector_id.split(":", 1)[1]
                        else 0.5
                    ),
                    balanced_accuracy=(
                        1.0
                        if group.value == injector.expected_detector_id.split(":", 1)[1]
                        else {
                            ShortcutTask.POSITIVE_BINARY: 0.5,
                            ShortcutTask.VARIANT_THREE_WAY: 1.0 / 3.0,
                            ShortcutTask.POSITIVE_HAZARD_CLASS: 0.25,
                        }[injector.target_task]
                    ),
                    balanced_chance={
                        ShortcutTask.POSITIVE_BINARY: 0.5,
                        ShortcutTask.VARIANT_THREE_WAY: 1.0 / 3.0,
                        ShortcutTask.POSITIVE_HAZARD_CLASS: 0.25,
                    }[injector.target_task],
                    raw_permutation_p=(
                        0.0002
                        if group.value == injector.expected_detector_id.split(":", 1)[1]
                        else 1.0
                    ),
                    holm_adjusted_p=(
                        9 * 0.0002
                        if group.value == injector.expected_detector_id.split(":", 1)[1]
                        else 1.0
                    ),
                    optimizer_iterations=1,
                    optimizer_converged=True,
                    passed=group.value != injector.expected_detector_id.split(":", 1)[1],
                )
                for group in ShortcutFeatureGroup
            ),
            passed=True,
        )
        for index, injector in enumerate(NAMED_LEAK_INJECTORS)
    )


def _clean_probes() -> tuple[ShortcutProbeResult, ...]:
    return tuple(
        ShortcutProbeResult(
            task=task,
            feature_group=group,
            feature_dimension=PHASE1_FEATURE_DIMENSIONS[task][
                tuple(ShortcutFeatureGroup).index(group)
            ],
            train_examples=_probe_workload(task, episode_count=100_000)[0],
            test_examples=_probe_workload(task, episode_count=100_000)[1],
            train_class_counts=_probe_workload(task, episode_count=100_000)[2],
            test_class_counts=_probe_workload(task, episode_count=100_000)[3],
            raw_accuracy={
                ShortcutTask.POSITIVE_BINARY: 0.5,
                ShortcutTask.VARIANT_THREE_WAY: 1.0 / 3.0,
                ShortcutTask.POSITIVE_HAZARD_CLASS: 0.25,
            }[task],
            balanced_accuracy={
                ShortcutTask.POSITIVE_BINARY: 0.5,
                ShortcutTask.VARIANT_THREE_WAY: 1.0 / 3.0,
                ShortcutTask.POSITIVE_HAZARD_CLASS: 0.25,
            }[task],
            balanced_chance={
                ShortcutTask.POSITIVE_BINARY: 0.5,
                ShortcutTask.VARIANT_THREE_WAY: 1.0 / 3.0,
                ShortcutTask.POSITIVE_HAZARD_CLASS: 0.25,
            }[task],
            raw_permutation_p=1.0,
            holm_adjusted_p=1.0,
            optimizer_iterations=1,
            optimizer_converged=True,
            passed=True,
        )
        for task in ShortcutTask
        for group in ShortcutFeatureGroup
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
        schema_version="oracle-evaluation-report-v2",
        provenance=independent,
        namespace_evidence=_namespace(independent=True),
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
        schema_version="leakage-report-v2",
        provenance=independent.model_copy(
            update={
                "analysis_seeds": {
                    "audit_seed": 2026083091,
                    "positive_control_seed": 2026083092,
                },
                "leakage_audit": anchor,
            }
        ),
        namespace_evidence=_namespace(independent=True),
        generation_mode="independent",
        profile=LeakageAuditProfileName.PHASE1_GATE,
        corpus_hash=gate_corpus,
        feature_schema_hash=LEAKAGE_FEATURE_SCHEMA_SHA256,
        split_membership_hash="6" * 64,
        train_membership_hash="7" * 64,
        test_membership_hash="8" * 64,
        episode_count=100_000,
        randomization_block_count=25_000,
        suite_path_denominators=GATE_DENOMINATORS,
        construction_check_ids=LEAKAGE_CONSTRUCTION_CHECK_IDS,
        construction_checks=LEAKAGE_CONSTRUCTION_CHECKS,
        probes=_clean_probes(),
        label_shuffled_probes=_clean_probes(),
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
        schema_version="phase1-reproducibility-v2",
        source_mode="manifest",
        source_payload_sha256=manifest_payload_sha256,
        sample_size=1_000,
        verified_source_entries=10_000,
        modes=("forward", "reverse", "chunked", "fresh_process"),
        chunk_sizes=(1, 3, 7),
        python_hash_seeds=(0, 1),
        reference_corpus_sha256=validation_corpus,
        sample_membership_sha256=manifest_sample_membership_sha256(
            manifest_payload_sha256,
            select_manifest_reproducibility_sample(manifest, manifest_payload_sha256, 1_000),
        ),
        mismatch_count=0,
        namespace_evidence=_namespace(independent=False),
        provenance=manifest.provenance,
        passed=True,
    )
    independent_descriptor = IndependentSourceDescriptor(
        schema_version="phase1-independent-source-v1",
        allocation_id="phase1-independent-gate-v1",
        allocation_sha256=GATE_ALLOCATION_SHA256,
        split_namespace=SplitNamespace.PHASE1_GATE,
        root_seed=2026083011,
        public_id_seed_sha256=independent.public_id_seed_sha256,
        config_sha256=independent.config_sha256,
        generator_source_sha256=independent.generator_source.sha256,
    )
    independent_source_hash = sha256_bytes(canonical_json_bytes(independent_descriptor))
    independent_requests = tuple(iter_independent_requests(PHASE1_GATE_ALLOCATION, 2026083011))
    independent_reproducibility = ReproducibilityReport(
        schema_version="phase1-reproducibility-v2",
        source_mode="independent_allocation",
        source_payload_sha256=independent_source_hash,
        sample_size=1_000,
        verified_source_entries=100_000,
        modes=("forward", "reverse", "chunked", "fresh_process"),
        chunk_sizes=(1, 3, 7),
        python_hash_seeds=(0, 1),
        reference_corpus_sha256=gate_corpus,
        sample_membership_sha256=independent_sample_membership_sha256(
            independent_source_hash,
            select_independent_reproducibility_sample(
                independent_requests, independent_source_hash, 1_000
            ),
        ),
        mismatch_count=0,
        namespace_evidence=_namespace(independent=True),
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


def _replace_common_provenance(
    artifacts: dict[str, bytes],
    mutation: Callable[[dict[str, object]], None],
) -> None:
    envelope = json.loads(artifacts["validation.json"])
    mutation(envelope["payload"]["provenance"])
    envelope["payload_sha256"] = sha256_bytes(canonical_json_bytes(envelope["payload"]))
    artifacts["validation.json"] = canonical_json_bytes(envelope)
    changed_manifest = EpisodeManifest.model_validate_json(
        canonical_json_bytes(envelope["payload"])
    )
    changed_membership = manifest_sample_membership_sha256(
        envelope["payload_sha256"],
        select_manifest_reproducibility_sample(
            changed_manifest,
            envelope["payload_sha256"],
            1_000,
        ),
    )
    for name in (
        "oracle.json",
        "leakage.json",
        "validation-reproducibility.json",
        "independent-reproducibility.json",
    ):

        def replace(value: dict[str, object], *, artifact_name: str = name) -> None:
            mutation(value["provenance"])  # type: ignore[arg-type,index]
            if artifact_name == "validation-reproducibility.json":
                value["source_payload_sha256"] = envelope["payload_sha256"]
                value["sample_membership_sha256"] = changed_membership

        artifacts[name] = _mutate_json(artifacts[name], replace)


def _independent_descriptor_sha256(**updates: object) -> str:
    provenance = _provenance(independent=True)
    values: dict[str, object] = {
        "schema_version": "phase1-independent-source-v1",
        "allocation_id": "phase1-independent-gate-v1",
        "allocation_sha256": GATE_ALLOCATION_SHA256,
        "split_namespace": SplitNamespace.PHASE1_GATE,
        "root_seed": 2026083011,
        "public_id_seed_sha256": public_id_seed_sha256(2026083012),
        "config_sha256": provenance.config_sha256,
        "generator_source_sha256": provenance.generator_source.sha256,
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
        value["namespace_evidence"]["public_id_seed"] = public_id_seed  # type: ignore[index]
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
            value["namespace_evidence"]["public_id_seed"] = public_id_seed  # type: ignore[index]
            if name == "leakage.json":
                current = _provenance(independent=True)
                descriptor = AuditSourceDescriptor(
                    schema_version="leakage-source-v1",
                    generation_mode="independent",
                    allocation_id="phase1-independent-gate-v1",
                    allocation_or_manifest_sha256=GATE_ALLOCATION_SHA256,
                    split_namespace=SplitNamespace.PHASE1_GATE,
                    root_seed=root_seed,
                    public_id_seed_sha256=fingerprint,
                    config_sha256=current.config_sha256,
                    generator_source_sha256=current.generator_source.sha256,
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
        "schema_version": "phase1-gate-verification-v2",
        "validation_episode_count": 10_000,
        "independent_episode_count": 100_000,
        "matched_accepted_draw_count": 2_500,
        "independent_accepted_draw_count": 100_000,
        "matched_public_id_seed": 2026083002,
        "independent_public_id_seed": 2026083012,
        "construction_token_count": 750_000,
        "public_id_count": 117_000,
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


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("validation_manifest_payload_sha256", "A" * 64),
        ("validation_manifest_payload_sha256", "a" * 63),
        ("independent_corpus_sha256", "G" * 64),
        ("independent_corpus_sha256", False),
    ),
)
def test_outer_gate_result_requires_strict_lowercase_sha256_digests(
    field: str,
    replacement: object,
) -> None:
    """The success envelope cannot emit malformed or coerced artifact digests."""
    payload = {
        "schema_version": "phase1-gate-verification-v2",
        "validation_episode_count": 10_000,
        "independent_episode_count": 100_000,
        "matched_accepted_draw_count": 2_500,
        "independent_accepted_draw_count": 100_000,
        "matched_public_id_seed": 2026083002,
        "independent_public_id_seed": 2026083012,
        "construction_token_count": 750_000,
        "public_id_count": 117_000,
        "validation_manifest_payload_sha256": "a" * 64,
        "independent_corpus_sha256": "b" * 64,
        "foundation_model_calls": 0,
        "passed": True,
    }
    payload[field] = replacement

    with pytest.raises(ValidationError):
        Phase1GateVerificationResult.model_validate(payload)


@pytest.mark.parametrize(
    ("artifact_name", "stale_schema"),
    (
        ("oracle.json", "oracle-evaluation-report-v1"),
        ("leakage.json", "leakage-report-v1"),
        ("validation-reproducibility.json", "phase1-reproducibility-v1"),
        ("independent-reproducibility.json", "phase1-reproducibility-v1"),
    ),
)
def test_verifier_rejects_stale_inner_v1_reports_without_compatibility_loader(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    artifact_name: str,
    stale_schema: str,
) -> None:
    artifacts = dict(consistent_artifact_bytes)
    artifacts[artifact_name] = _mutate_json(
        artifacts[artifact_name],
        lambda value: value.__setitem__("schema_version", stale_schema),
    )

    with pytest.raises(ArtifactIntegrityError, match="schema"):
        _verify(_write_artifacts(tmp_path, artifacts))


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

    with pytest.raises(ArtifactIntegrityError, match=r"oracle.*(?:schema|passed)"):
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
    "artifact_name",
    (
        "validation.json",
        "oracle.json",
        "validation-reproducibility.json",
        "independent-reproducibility.json",
    ),
)
@pytest.mark.parametrize("field", ("analysis_seeds", "leakage_audit"))
def test_verifier_requires_exact_empty_non_leakage_metadata(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    artifact_name: str,
    field: str,
) -> None:
    """Only the leakage report may carry audit seeds or a leakage authority anchor."""
    artifacts = dict(consistent_artifact_bytes)
    leakage = json.loads(artifacts["leakage.json"])
    replacement = (
        {"unexpected_seed": 1}
        if field == "analysis_seeds"
        else leakage["provenance"]["leakage_audit"]
    )
    if artifact_name == "validation.json":
        envelope = json.loads(artifacts[artifact_name])
        envelope["payload"]["provenance"][field] = replacement
        envelope["payload_sha256"] = sha256_bytes(canonical_json_bytes(envelope["payload"]))
        artifacts[artifact_name] = canonical_json_bytes(envelope)
    else:
        artifacts[artifact_name] = _mutate_json(
            artifacts[artifact_name],
            lambda value: value["provenance"].__setitem__(field, replacement),  # type: ignore[index,union-attr]
        )

    with pytest.raises(ArtifactIntegrityError, match="non-leakage provenance metadata"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize(
    "mutation",
    (
        pytest.param(
            lambda value: value.__setitem__("source_commit", "f" * 40),
            id="missing-source-commit",
        ),
        pytest.param(
            lambda value: value.__setitem__("plan_base_revision", "f" * 40),
            id="wrong-historical-plan-base",
        ),
        pytest.param(
            lambda value: value["analysis_source"].__setitem__(  # type: ignore[index,union-attr]
                "sha256", "f" * 64
            ),
            id="wrong-historical-analysis-blob-hash",
        ),
        pytest.param(
            lambda value: value.__setitem__(
                "analysis_source",
                {
                    "frame_version": "sc-source-tree-v1",
                    "scope": "phase1_task14_analysis",
                    "paths": list(TASK14_ANALYSIS_SOURCE_PATHS),
                    "sha256": source_tree_sha256_at_revision(
                        Path.cwd(),
                        subprocess.run(
                            ("git", "rev-parse", "HEAD^{commit}"),
                            check=True,
                            capture_output=True,
                            text=True,
                        ).stdout.strip(),
                        TASK14_ANALYSIS_SOURCE_PATHS,
                    ),
                },
            ),
            id="coordinated-task14-scope-downgrade",
        ),
    ),
)
def test_verifier_authenticates_coordinated_provenance_against_git_history(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    mutation: Callable[[dict[str, object]], None],
) -> None:
    artifacts = dict(consistent_artifact_bytes)
    _replace_common_provenance(artifacts, mutation)

    with pytest.raises(ArtifactIntegrityError, match=r"historical Git|source|schema"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize(
    "artifact_name",
    ("validation-reproducibility.json", "independent-reproducibility.json"),
)
def test_verifier_independently_recomputes_reproducibility_membership(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    artifact_name: str,
) -> None:
    artifacts = dict(consistent_artifact_bytes)
    artifacts[artifact_name] = _mutate_json(
        artifacts[artifact_name],
        lambda value: value.__setitem__("sample_membership_sha256", "0" * 64),
    )

    with pytest.raises(ArtifactIntegrityError, match=r"membership|reproducibility"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize(
    "artifact_names",
    (
        ("oracle.json",),
        ("oracle.json", "leakage.json", "independent-reproducibility.json"),
    ),
)
def test_verifier_rederives_independent_namespace_instead_of_trusting_reports(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    artifact_names: tuple[str, ...],
) -> None:
    artifacts = dict(consistent_artifact_bytes)
    for name in artifact_names:
        artifacts[name] = _mutate_json(
            artifacts[name],
            lambda value: value["namespace_evidence"].__setitem__(  # type: ignore[index,union-attr]
                "seed_token_sequence_sha256", "0" * 64
            ),
        )

    with pytest.raises(ArtifactIntegrityError, match=r"namespace"):
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


def test_verifier_refuses_an_incomplete_clean_probe_family(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        probes = value["probes"]  # type: ignore[assignment]
        probes.pop()  # type: ignore[union-attr]

    artifacts["leakage.json"] = _mutate_json(artifacts["leakage.json"], mutate)

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|probe"):
        _verify(_write_artifacts(tmp_path, artifacts))


def test_verifier_derives_clean_probe_chance_from_the_task(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        probe = value["probes"][0]  # type: ignore[index]
        probe["balanced_accuracy"] = 0.99
        probe["balanced_chance"] = 1.0
        probe["raw_permutation_p"] = 0.0002
        probe["holm_adjusted_p"] = 0.0002
        probe["passed"] = True

    artifacts["leakage.json"] = _mutate_json(artifacts["leakage.json"], mutate)

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|probe|chance"):
        _verify(_write_artifacts(tmp_path, artifacts))


def test_verifier_recomputes_clean_probe_holm_values_and_pass_states(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        probe = value["probes"][0]  # type: ignore[index]
        probe["balanced_accuracy"] = 0.99
        probe["raw_permutation_p"] = 0.0002
        probe["holm_adjusted_p"] = 1.0
        probe["passed"] = True

    artifacts["leakage.json"] = _mutate_json(artifacts["leakage.json"], mutate)

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|probe|Holm"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize(
    "field,replacement",
    [("balanced_accuracy", 2.0), ("holm_adjusted_p", -0.1)],
)
def test_verifier_refuses_out_of_range_positive_control_metrics(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    field: str,
    replacement: float,
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        value["positive_controls"][0][field] = replacement  # type: ignore[index]

    artifacts["leakage.json"] = _mutate_json(artifacts["leakage.json"], mutate)

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|control|schema"):
        _verify(_write_artifacts(tmp_path, artifacts))


def test_verifier_recomputes_positive_control_holm_and_pass_state(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        control = value["positive_controls"][0]  # type: ignore[index]
        expected = next(probe for probe in control["probes"] if probe["feature_group"] == "counts")
        expected["balanced_accuracy"] = 0.5
        expected["raw_permutation_p"] = 1.0
        expected["holm_adjusted_p"] = 1.0
        expected["passed"] = True

    artifacts["leakage.json"] = _mutate_json(artifacts["leakage.json"], mutate)

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|control|evidence"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize(
    "target,field,replacement",
    [
        ("clean", "feature_dimension", -1),
        ("control", "feature_dimension", -1),
        ("clean", "optimizer_iterations", -1),
        ("clean", "optimizer_iterations", 501),
        ("control", "optimizer_iterations", -1),
        ("control", "optimizer_iterations", 501),
    ],
)
def test_verifier_refuses_impossible_probe_shape_values(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    target: str,
    field: str,
    replacement: int,
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        family = value["probes"] if target == "clean" else value["positive_controls"][0]["probes"]  # type: ignore[index]
        family[0][field] = replacement

    artifacts["leakage.json"] = _mutate_json(artifacts["leakage.json"], mutate)

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|probe|schema|shape"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize("target", ["clean", "control"])
def test_verifier_refuses_underpowered_probe_workload_metadata(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    target: str,
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        family = value["probes"] if target == "clean" else value["positive_controls"][0]["probes"]  # type: ignore[index]
        family[0]["train_examples"] = 1
        family[0]["test_examples"] = 1
        family[0]["train_class_counts"] = {"0": 1}
        family[0]["test_class_counts"] = {"0": 1}

    artifacts["leakage.json"] = _mutate_json(artifacts["leakage.json"], mutate)

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|probe|workload|class"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize("target", ["clean", "control"])
@pytest.mark.parametrize("raw_p", [0.00021, 0.0])
def test_verifier_refuses_off_grid_phase1_permutation_p_values(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    target: str,
    raw_p: float,
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        if target == "clean":
            probe = value["probes"][0]  # type: ignore[index]
            probe["raw_permutation_p"] = raw_p
            probe["holm_adjusted_p"] = 27 * raw_p
        else:
            control = value["positive_controls"][0]  # type: ignore[index]
            expected = next(
                probe for probe in control["probes"] if probe["feature_group"] == "counts"
            )
            expected["raw_permutation_p"] = raw_p
            expected["holm_adjusted_p"] = 9 * raw_p
            control["holm_adjusted_p"] = 9 * raw_p

    artifacts["leakage.json"] = _mutate_json(artifacts["leakage.json"], mutate)

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|probe|permutation|grid"):
        _verify(_write_artifacts(tmp_path, artifacts))


def test_verifier_refuses_a_nonconverged_positive_control_probe(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        value["positive_controls"][0]["probes"][0]["optimizer_converged"] = False  # type: ignore[index]

    artifacts["leakage.json"] = _mutate_json(artifacts["leakage.json"], mutate)

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|control|converg"):
        _verify(_write_artifacts(tmp_path, artifacts))


def test_standalone_verifier_rejects_nonconverged_control_without_schema_help(
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    report = LeakageReport.model_validate_json(consistent_artifact_bytes["leakage.json"])
    control = report.positive_controls[0]
    probe = control.probes[0].model_copy(update={"optimizer_converged": False})
    mutated_control = control.model_copy(update={"probes": (probe, *control.probes[1:])})
    mutated = report.model_copy(
        update={"positive_controls": (mutated_control, *report.positive_controls[1:])}
    )

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|control|converg"):
        _require_leakage(mutated)


def test_verifier_requires_full_label_shuffled_probe_evidence(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        value.pop("label_shuffled_probes", None)

    artifacts["leakage.json"] = _mutate_json(artifacts["leakage.json"], mutate)

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|label|shuffl"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("task", "positive_binary"),
        ("feature_group", "id_position"),
        ("feature_dimension", 17.0),
        ("train_examples", 80_000.0),
        ("test_examples", 20_000.0),
        ("train_class_counts", {"0": 40_000.0, "1": 40_000.0}),
        ("test_class_counts", {"0": 10_000.0, "1": 10_000.0}),
        ("raw_accuracy", True),
        ("balanced_accuracy", True),
        ("raw_permutation_p", True),
        ("holm_adjusted_p", True),
        ("optimizer_iterations", True),
        ("optimizer_converged", 1),
        ("passed", 1),
    ],
)
def test_standalone_verifier_rejects_schema_bypassed_probe_primitive_types(
    consistent_artifact_bytes: dict[str, bytes],
    field: str,
    replacement: object,
) -> None:
    report = LeakageReport.model_validate_json(consistent_artifact_bytes["leakage.json"])
    probe = report.probes[0].model_copy(update={field: replacement})
    mutated = report.model_copy(update={"probes": (probe, *report.probes[1:])})

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|probe|type|schema"):
        _require_leakage(mutated)


def test_standalone_verifier_rejects_schema_bypassed_control_summary_boolean(
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    report = LeakageReport.model_validate_json(consistent_artifact_bytes["leakage.json"])
    control = report.positive_controls[0].model_copy(update={"balanced_accuracy": True})
    mutated = report.model_copy(
        update={"positive_controls": (control, *report.positive_controls[1:])}
    )

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|control|type|schema"):
        _require_leakage(mutated)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("passed", 1),
        ("label_shuffled_control_passed", 1),
        (
            "construction_checks",
            {
                "provenance": 1,
                "public_ids": 1,
                "seed_tokens": 1,
                "invariants": 1,
                "finite_features": 1,
                "two_pass_identity": 1,
            },
        ),
    ],
)
def test_standalone_verifier_rejects_schema_bypassed_report_booleans(
    consistent_artifact_bytes: dict[str, bytes],
    field: str,
    replacement: object,
) -> None:
    report = LeakageReport.model_validate_json(consistent_artifact_bytes["leakage.json"])
    mutated = report.model_copy(update={field: replacement})

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|check|type|schema"):
        _require_leakage(mutated)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("construction_checks", {"invented": True}),
        ("feature_schema_hash", "0" * 64),
    ],
)
def test_verifier_requires_exact_construction_and_feature_schema_identities(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    field: str,
    replacement: object,
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        value[field] = replacement

    artifacts["leakage.json"] = _mutate_json(artifacts["leakage.json"], mutate)

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|construction|feature|schema"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("construction_checks", {"invented": True}),
        ("feature_schema_hash", "0" * 64),
    ],
)
def test_standalone_verifier_requires_exact_construction_and_feature_schema_identities(
    consistent_artifact_bytes: dict[str, bytes],
    field: str,
    replacement: object,
) -> None:
    report = LeakageReport.model_validate_json(consistent_artifact_bytes["leakage.json"])
    mutated = report.model_copy(update={field: replacement})

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|construction|feature|schema"):
        _require_leakage(mutated)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("schema_version", lambda _report: 1),
        ("generation_mode", lambda _report: _GenerationMode.INDEPENDENT),
        ("corpus_hash", lambda _report: 1),
        ("split_membership_hash", lambda _report: 1),
        ("train_membership_hash", lambda _report: 1),
        ("test_membership_hash", lambda _report: 1),
        ("episode_count", lambda report: float(report.episode_count)),
        ("randomization_block_count", lambda report: float(report.randomization_block_count)),
        (
            "suite_path_denominators",
            lambda report: {
                key: float(value) for key, value in report.suite_path_denominators.items()
            },
        ),
        ("probes", lambda report: list(report.probes)),
        ("label_shuffled_probes", lambda report: list(report.label_shuffled_probes)),
        ("positive_controls", lambda report: list(report.positive_controls)),
        ("counterfactual_checks", lambda report: list(report.counterfactual_checks)),
    ],
)
def test_standalone_verifier_rejects_schema_bypassed_outer_report_types(
    consistent_artifact_bytes: dict[str, bytes],
    field: str,
    replacement: Callable[[LeakageReport], object],
) -> None:
    report = LeakageReport.model_validate_json(consistent_artifact_bytes["leakage.json"])
    mutated = report.model_copy(update={field: replacement(report)})

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|report|type|schema"):
        _require_leakage(mutated)


@pytest.mark.parametrize("mutation", ["payload_type", "reversed", "duplicate"])
def test_standalone_verifier_requires_exact_ordered_counterfactual_evidence(
    consistent_artifact_bytes: dict[str, bytes],
    mutation: str,
) -> None:
    report = LeakageReport.model_validate_json(consistent_artifact_bytes["leakage.json"])
    checks = report.counterfactual_checks
    if mutation == "payload_type":
        first = checks[0].model_copy(update={"result_payload_sha256": 1})
        replacement = (first, *checks[1:])
    elif mutation == "reversed":
        replacement = tuple(reversed(checks))
    else:
        replacement = (*checks, checks[0])
    mutated = report.model_copy(update={"counterfactual_checks": replacement})

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|counterfactual|order|type"):
        _require_leakage(mutated)


def test_standalone_verifier_rejects_reordered_construction_check_mapping(
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    report = LeakageReport.model_validate_json(consistent_artifact_bytes["leakage.json"])
    reversed_checks = dict(reversed(tuple(report.construction_checks.items())))
    mutated = report.model_copy(update={"construction_checks": reversed_checks})

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|construction|order"):
        _require_leakage(mutated)


def test_verifier_requires_ordered_construction_check_identity_evidence(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
) -> None:
    artifacts = dict(consistent_artifact_bytes)

    def mutate(value: dict[str, object]) -> None:
        value.pop("construction_check_ids", None)

    artifacts["leakage.json"] = _mutate_json(artifacts["leakage.json"], mutate)

    with pytest.raises(ArtifactIntegrityError, match=r"leakage|construction|order|schema"):
        _verify(_write_artifacts(tmp_path, artifacts))


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.__setitem__("requested_episode_count", 99_999),
        lambda value: value.__setitem__("positive_count", 49_999),
        lambda value: value["suite_path_denominators"].__setitem__(  # type: ignore[index,union-attr]
            "iid_primary:2", 7_999
        ),
        lambda value: value.__setitem__("oracle_successes", 99_999),
        lambda value: value["random_positive"].__setitem__(  # type: ignore[index,union-attr]
            "successes", 6_249
        ),
        lambda value: value["random_negative"].__setitem__(  # type: ignore[index,union-attr]
            "exact_binomial_p", 0.5
        ),
        lambda value: value.__setitem__("random_pooled_observed_rate", 0.3),
        lambda value: value.__setitem__("clock_0_1x_episode_count", 4_999),
        lambda value: value.__setitem__("rejection_reason_counts", {"forged": 1}),
        lambda value: value.__setitem__("generation_attempt_count", 100_001),
        lambda value: value.__setitem__("rejected_draw_count", 1),
        lambda value: value.__setitem__("rejected_draw_rate", 0.5),
        lambda value: value.__setitem__("passed", False),
    ),
)
def test_verifier_rejects_each_underived_oracle_report_relationship(
    tmp_path: Path,
    consistent_artifact_bytes: dict[str, bytes],
    mutation: Callable[[dict[str, object]], None],
) -> None:
    artifacts = dict(consistent_artifact_bytes)
    artifacts["oracle.json"] = _mutate_json(artifacts["oracle.json"], mutation)

    with pytest.raises(ArtifactIntegrityError, match=r"oracle.*(?:schema|denominator|failed)"):
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

    with pytest.raises(ArtifactIntegrityError, match=r"oracle.*(?:schema|denominator)"):
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
