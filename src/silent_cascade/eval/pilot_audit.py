"""Private Phase 4 pilot evidence, independent of the frozen Phase 1 gate."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from silent_cascade.env.config import LeakageAuditProfileConfig
from silent_cascade.env.episode import CorpusDigestEntry, corpus_sha256, episode_sha256
from silent_cascade.env.leakage import (
    AuditExample,
    ShortcutProbeResult,
    _audit_public_shortcut_controls,
    _expected_probe_dimension,
    _PublicShortcutControls,
    _statistical_probe_evidence_is_consistent,
    audit_public_shortcuts,
)
from silent_cascade.env.pilot import curriculum_to_bundle
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.schemas import HazardFact, LinkFact, SafeFact
from silent_cascade.train.pilot_config import Phase4Config
from silent_cascade.train.pilot_data import (
    Hash,
    PilotManifest,
    _publish_pilot_bytes,
    iter_pilot_examples,
)
from silent_cascade.validation import StrictModel

PILOT_AUDIT_PROFILE = LeakageAuditProfileConfig(
    episode_count=10_000,
    permutation_replicates=4_999,
    positive_control_episode_count=10_000,
    positive_control_permutation_replicates=4_999,
    minimum_test_examples_per_class=200,
    enforce_clean_statistical_gate=True,
)


class PilotAuditReport(StrictModel):
    schema_version: Literal["phase4-pilot-audit-v1"] = "phase4-pilot-audit-v1"
    profile: Literal["phase4-pilot-10000", "debug-non-acceptance"]
    manifest_sha256: Hash
    config_sha256: Hash
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    executing_source_files: dict[str, Hash]
    executing_source_sha256: Hash
    audit_config_sha256: Hash
    audit_profile_sha256: Hash
    corpus_sha256: Hash
    projected_episode_sha256s: tuple[Hash, ...]
    oracle_trace_sha256s: tuple[Hash, ...]
    count: int
    oracle_verified_count: int
    parent_verified_count: int
    feasible_trace_count: int
    unique_public_id_count: int
    allocation_key_separation_verified: Literal[True] = True
    variant_counts: dict[str, int]
    path_counts: dict[str, int]
    record_count_histogram: dict[str, int]
    hazard_record_count: int
    safe_record_count: int
    link_record_count: int
    statistical_status: Literal["passed", "failed", "insufficient-debug-corpus"]
    shortcut_probes: tuple[ShortcutProbeResult, ...]
    controls: _PublicShortcutControls | None
    acceptance: bool
    foundation_model_calls: Literal[0] = 0

    @model_validator(mode="after")
    def derive_acceptance(self):
        if not (
            self.count
            == self.oracle_verified_count
            == self.parent_verified_count
            == self.feasible_trace_count
            == self.unique_public_id_count
            == len(self.projected_episode_sha256s)
            == len(self.oracle_trace_sha256s)
        ):
            raise ValueError("pilot audit evidence denominator mismatch")
        if self.executing_source_sha256 != sha256_bytes(
            canonical_json_bytes(self.executing_source_files)
        ):
            raise ValueError("pilot audit source hash mismatch")
        if self.hazard_record_count != self.count * 2 or self.safe_record_count != self.count:
            raise ValueError("pilot audit terminal inventory mismatch")
        if self.variant_counts != {
            "positive": self.count // 2,
            "safe_negative": self.count // 4,
            "disconnected_negative": self.count // 4,
        }:
            raise ValueError("pilot audit allocation mismatch")
        if (
            sum(self.path_counts.values()) != self.count
            or sum(self.record_count_histogram.values()) != self.count
        ):
            raise ValueError("pilot audit record/path denominator mismatch")
        if self.profile == "debug-non-acceptance":
            if (
                self.count != 16
                or self.shortcut_probes
                or self.controls is not None
                or self.statistical_status != "insufficient-debug-corpus"
                or self.acceptance
            ):
                raise ValueError("debug audit cannot certify statistical acceptance")
        else:
            if (
                self.count != 10_000
                or len(self.shortcut_probes) != 27
                or self.controls is None
                or len(self.controls.shuffled_probes) != 27
            ):
                raise ValueError("pilot acceptance requires complete full-profile probes/controls")
            if (
                self.audit_profile_sha256 != sha256_bytes(canonical_json_bytes(PILOT_AUDIT_PROFILE))
                or self.controls.profile_sha256 != self.audit_profile_sha256
                or self.controls.config_sha256 != self.audit_config_sha256
                or self.controls.train_examples + self.controls.test_examples != self.count
            ):
                raise ValueError("pilot statistical profile/configuration mismatch")
            for family, complete in (
                (self.shortcut_probes, True),
                (self.controls.shuffled_probes, True),
                ((self.controls.injected_probe,), False),
            ):
                if not _statistical_probe_evidence_is_consistent(family, require_complete=complete):
                    raise ValueError("pilot statistical probe family is inconsistent")
                for probe in family:
                    divisor = 2 if probe.task.value == "positive_hazard_class" else 1
                    if (
                        probe.permutation_replicate_count != 4_999
                        or min(probe.test_class_counts.values()) < 200
                        or probe.train_examples * divisor != self.controls.train_examples
                        or probe.test_examples * divisor != self.controls.test_examples
                        or probe.feature_dimension
                        != _expected_probe_dimension(probe.task, probe.feature_group)
                    ):
                        raise ValueError("pilot probe does not meet the full statistical profile")
            if (
                self.controls.injected_probe.task.value,
                self.controls.injected_probe.feature_group.value,
            ) != ("positive_binary", "counts"):
                raise ValueError("pilot injected probe is not the predeclared detector")
            passed = (
                all(p.passed for p in self.shortcut_probes)
                and all(p.passed for p in self.controls.shuffled_probes)
                and self.controls.positive_control_passed
            )
            if self.acceptance != passed or self.statistical_status != (
                "passed" if passed else "failed"
            ):
                raise ValueError("pilot acceptance must derive from statistical evidence")
        return self


def audit_pilot_manifest(
    manifest: PilotManifest, *, config: Phase4Config, output_dir: Path
) -> PilotAuditReport:
    """Authenticate every projection before fitting any shortcut probe.

    A failed statistical report is preserved with acceptance=False. Callers must
    require acceptance before training. Sixteen-row debug data never lowers the
    full-profile statistical thresholds and never claims acceptance.
    """
    examples = []
    trace_hashes = []
    record_counts: Counter[str] = Counter()
    links = hazards = safe = 0
    for index, example in enumerate(iter_pilot_examples(manifest, config=config)):
        bundle = curriculum_to_bundle(example, config=config)
        examples.append(
            AuditExample(
                bundle, index, "independent", index // 4, index, quartet_member_index=index % 4
            )
        )
        trace_hashes.append(sha256_bytes(canonical_json_bytes(asdict(example.oracle_trace))))
        record_counts[str(len(bundle.public.events) - 1)] += 1
        links += sum(isinstance(e.payload, LinkFact) for e in bundle.public.events)
        hazards += sum(isinstance(e.payload, HazardFact) for e in bundle.public.events)
        safe += sum(isinstance(e.payload, SafeFact) for e in bundle.public.events)
    projected_hashes = tuple(episode_sha256(e.bundle) for e in examples)
    corpus_hash = corpus_sha256(
        (
            CorpusDigestEntry(e.bundle.public.init.episode_public_id, digest)
            for e, digest in zip(examples, projected_hashes, strict=True)
        ),
        expected_count=manifest.count,
    )
    probes: tuple[ShortcutProbeResult, ...] = ()
    controls = None
    if manifest.split == "validation":
        probes = audit_public_shortcuts(
            examples,
            config=config.data.leakage_audit,
            profile=PILOT_AUDIT_PROFILE,
            corpus_hash=corpus_hash,
        )
        controls = _audit_public_shortcut_controls(
            examples,
            config=config.data.leakage_audit,
            profile=PILOT_AUDIT_PROFILE,
            corpus_hash=corpus_hash,
        )
    accepted = bool(
        controls
        and controls.positive_control_passed
        and all(p.passed for p in probes)
        and all(p.passed for p in controls.shuffled_probes)
    )
    source_root = Path(__file__).resolve().parents[1]
    source_files = {
        path.relative_to(source_root).as_posix(): sha256_bytes(path.read_bytes())
        for path in sorted(source_root.rglob("*.py"))
    }
    report = PilotAuditReport(
        profile="phase4-pilot-10000" if manifest.split == "validation" else "debug-non-acceptance",
        manifest_sha256=sha256_bytes(canonical_json_bytes(manifest)),
        config_sha256=manifest.config_hash,
        source_commit=manifest.source_commit,
        executing_source_files=source_files,
        executing_source_sha256=sha256_bytes(canonical_json_bytes(source_files)),
        audit_config_sha256=sha256_bytes(canonical_json_bytes(config.data.leakage_audit)),
        audit_profile_sha256=sha256_bytes(canonical_json_bytes(PILOT_AUDIT_PROFILE)),
        corpus_sha256=corpus_hash,
        projected_episode_sha256s=projected_hashes,
        oracle_trace_sha256s=tuple(trace_hashes),
        count=manifest.count,
        oracle_verified_count=len(examples),
        parent_verified_count=len(examples),
        feasible_trace_count=len(examples),
        unique_public_id_count=len({e.bundle.public.init.episode_public_id for e in examples}),
        variant_counts=dict(Counter(e.bundle.truth.recipe.variant.value for e in examples)),
        path_counts=dict(
            Counter(str(e.bundle.truth.recipe.requested_path_length) for e in examples)
        ),
        record_count_histogram=dict(record_counts),
        hazard_record_count=hazards,
        safe_record_count=safe,
        link_record_count=links,
        shortcut_probes=probes,
        controls=controls,
        statistical_status=("passed" if accepted else "failed")
        if controls
        else "insufficient-debug-corpus",
        acceptance=accepted,
    )
    _publish_pilot_bytes(output_dir / "manifest.json", canonical_json_bytes(manifest))
    payload = canonical_json_bytes(report)
    _publish_pilot_bytes(output_dir / "report.json", payload)
    _publish_pilot_bytes(
        output_dir / "index.json",
        canonical_json_bytes(
            {
                "schema_version": "phase4-pilot-audit-index-v1",
                "report_sha256": sha256_bytes(payload),
                "manifest_sha256": report.manifest_sha256,
            }
        ),
    )
    return report
