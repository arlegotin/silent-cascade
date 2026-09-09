"""Streaming, source-bound non-neural runtime engineering evidence.

Raw compact witnesses are retained for all executions. Only the current trace
and the three selected CPU replay archives survive an iteration. Witness
integrity is unkeyed: replacing evidence together with every Git trust anchor
is outside the threat model; no learned-model claim follows from this gate.
"""

import hashlib
import json
import os
import stat
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from silent_cascade.config import ResolvedConfig
from silent_cascade.env.episode import EpisodeArtifact
from silent_cascade.env.reward import EpisodeScore
from silent_cascade.env.services import regenerate_entry
from silent_cascade.errors import ArtifactIntegrityError, AtomicWriteError, SilentCascadeError
from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at
from silent_cascade.eventflow.config import Phase2Config
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.provenance import (
    MAX_MANIFEST_BYTES,
    VALIDATION_FILE_SHA256,
    VALIDATION_MANIFEST_PATH,
    Phase2EvidenceProvenance,
    Phase2EvidenceProvenanceData,
    collect_phase2_evidence_provenance,
    parse_gate_config,
)
from silent_cascade.eventflow.replay import (
    CausalTraceArtifact,
    EpisodeResultArtifact,
    ReplayArtifact,
    parse_replay_artifact_bytes,
    verify_replay,
)
from silent_cascade.eventflow.scripted import ScriptedEventFlowAgent
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.crash_bundle import CrashContext, write_crash_bundle
from silent_cascade.logging.manifest import (
    ManifestAccessClass,
    ManifestEnvelope,
    MatchedManifestCoordinate,
    verify_manifest,
)
from silent_cascade.schemas import Condition
from silent_cascade.validation import StrictModel

MAX_GATE_BYTES = 32 * 1024 * 1024
type SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
VARIANTS = ("positive", "safe_negative", "disconnected_negative")


class EpisodeGateWitnessData(StrictModel):
    """Strict serialized observations; no producer score arithmetic."""

    entry_index: int = Field(ge=0, lt=10_000)
    episode_public_id: str = Field(min_length=36, max_length=36)
    episode_sha256: SHA256
    manifest_entry_sha256: SHA256
    coordinate: MatchedManifestCoordinate
    variant: Literal["positive", "safe_negative", "disconnected_negative"]
    config_sha256: SHA256
    trace_sha256: SHA256
    completed: bool
    score: EpisodeScore | None
    dynamics_failure: bool
    replay_failure: bool
    provenance_failure: bool
    internal_event_count: int = Field(ge=0)
    minimum_internal_gap: float | None
    gap_clamp_count: int = Field(ge=0)
    event_cap_failure: bool
    time_reversal_count: int = Field(ge=0)
    foundation_model_calls: int = Field(ge=0)


class EpisodeGateWitness(EpisodeGateWitnessData):
    @model_validator(mode="after")
    def consistent_result(self) -> Self:
        if self.completed != (self.score is not None):
            raise ValueError("completed witness must have exactly one scored terminal")
        if (self.internal_event_count == 0) != (self.minimum_internal_gap is None):
            raise ValueError("internal gap must exist exactly when internal events exist")
        if self.score is not None:
            score = self.score
            positive = self.variant == "positive"
            if score.is_positive != positive or score.action_count < 0:
                raise ValueError("score and variant disagree")
            expected_success = (
                score.action_count == 1 and score.correct_class is True and score.in_window is True
                if positive
                else score.action_count == 0
            )
            if score.timed_success != expected_success:
                raise ValueError("timed success must follow primitive score fields")
            if score.false_action != (not positive and score.action_count > 0):
                raise ValueError("false action must follow primitive action count")
        return self


def _chain(rows: tuple[EpisodeGateWitness, ...]) -> str:
    digest = hashlib.sha256(b"silent-cascade/phase2/episode-chain/v1\0")
    for index, row in enumerate(rows):
        raw = canonical_json_bytes(row)
        digest.update(index.to_bytes(8, "big"))
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _totals(rows: tuple[EpisodeGateWitness, ...]) -> dict:
    gaps = [row.minimum_internal_gap for row in rows if row.minimum_internal_gap is not None]
    return {
        "completed_episode_count": sum(row.completed for row in rows),
        **{f"{variant}_count": sum(row.variant == variant for row in rows) for variant in VARIANTS},
        "timed_success_count": sum(
            row.score is not None and row.score.timed_success for row in rows
        ),
        "false_action_count": sum(row.score is not None and row.score.false_action for row in rows),
        "dynamics_failure_count": sum(row.dynamics_failure for row in rows),
        "replay_failure_count": sum(row.replay_failure for row in rows),
        "internal_event_count": sum(row.internal_event_count for row in rows),
        "maximum_episode_internal_events": max(
            (row.internal_event_count for row in rows), default=0
        ),
        "minimum_internal_gap": min(gaps, default=None),
        "gap_clamp_count": sum(row.gap_clamp_count for row in rows),
        "event_cap_failure_count": sum(row.event_cap_failure for row in rows),
        "time_reversal_count": sum(row.time_reversal_count for row in rows),
        "provenance_failure_count": sum(row.provenance_failure for row in rows),
        "foundation_model_calls": sum(row.foundation_model_calls for row in rows),
    }


def _passes(totals: dict, requested: int, provenance: Phase2EvidenceProvenance) -> bool:
    return (
        totals["completed_episode_count"] == totals["timed_success_count"] == requested
        and tuple(totals[f"{v}_count"] for v in VARIANTS)
        == (requested // 2, requested // 4, requested // 4)
        and all(
            totals[field] == 0
            for field in (
                "false_action_count",
                "dynamics_failure_count",
                "replay_failure_count",
                "gap_clamp_count",
                "event_cap_failure_count",
                "time_reversal_count",
                "provenance_failure_count",
                "foundation_model_calls",
            )
        )
        and totals["maximum_episode_internal_events"] <= 64
        and totals["minimum_internal_gap"] is not None
        and totals["minimum_internal_gap"] >= 1e-4
        and not provenance.source_dirty
    )


class _GateFields(StrictModel):
    provenance: Phase2EvidenceProvenanceData
    validation_manifest_payload_sha256: SHA256
    validation_manifest_file_sha256: SHA256
    config_sha256: SHA256
    generator_version: Literal["ofd-v1"]
    condition: Literal["scripted_event_flow"]
    agent_implementation: Literal["scripted-event-flow-v1"]
    requested_episode_count: int = Field(gt=0, le=10_000, multiple_of=4)
    completed_episode_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    safe_negative_count: int = Field(ge=0)
    disconnected_negative_count: int = Field(ge=0)
    timed_success_count: int = Field(ge=0)
    false_action_count: int = Field(ge=0)
    dynamics_failure_count: int = Field(ge=0)
    replay_failure_count: int = Field(ge=0)
    internal_event_count: int = Field(ge=0)
    maximum_episode_internal_events: int = Field(ge=0)
    minimum_internal_gap: float | None
    gap_clamp_count: int = Field(ge=0)
    event_cap_failure_count: int = Field(ge=0)
    time_reversal_count: int = Field(ge=0)
    provenance_failure_count: int = Field(ge=0)
    trace_chain_sha256: SHA256
    episode_witnesses: tuple[EpisodeGateWitnessData, ...] = Field(min_length=4, max_length=10_000)
    selected_replay_samples: tuple[ReplayArtifact, ...] = Field(min_length=3, max_length=3)
    selected_replay_trace_hashes: tuple[SHA256, ...] = Field(min_length=3, max_length=3)
    foundation_model_calls: Literal[0]
    passed: bool

    @field_validator("foundation_model_calls", "requested_episode_count", mode="before")
    @classmethod
    def exact_integers(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("gate counts must be exact integers")
        return value


class _ValidatedGateFields(_GateFields):
    provenance: Phase2EvidenceProvenance
    episode_witnesses: tuple[EpisodeGateWitness, ...] = Field(min_length=4, max_length=10_000)

    @model_validator(mode="after")
    def derive_evidence(self) -> Self:
        rows = self.episode_witnesses
        if len(rows) != self.requested_episode_count:
            raise ValueError("raw witnesses must cover requested denominator")
        if tuple(row.entry_index for row in rows) != tuple(range(len(rows))):
            raise ValueError("raw witnesses must retain manifest order")
        for name in ("episode_public_id", "episode_sha256", "trace_sha256"):
            if len({getattr(row, name) for row in rows}) != len(rows):
                raise ValueError("episode identities and all trace hashes must be unique")
        for name in (
            "config_sha256",
            "validation_manifest_file_sha256",
            "validation_manifest_payload_sha256",
        ):
            if getattr(self, name) != getattr(self.provenance, name):
                raise ValueError("report and provenance hashes disagree")
        if any(row.config_sha256 != self.config_sha256 for row in rows):
            raise ValueError("episode witness configuration mismatch")
        totals = _totals(rows)
        if any(getattr(self, name) != value for name, value in totals.items()):
            raise ValueError("report totals differ from raw witnesses")
        if self.trace_chain_sha256 != _chain(rows):
            raise ValueError("ordered framed trace chain mismatch")
        runtime = parse_gate_config(self.provenance.config_canonical_json).event_flow
        by_id = {row.episode_public_id: row for row in rows}
        for variant, sample, listed in zip(
            VARIANTS, self.selected_replay_samples, self.selected_replay_trace_hashes, strict=True
        ):
            parse_replay_artifact_bytes(canonical_json_bytes(sample))
            bundle = sample.episode.to_bundle()
            row = by_id.get(sample.expected_result.public_id)
            if (
                row is None
                or row.variant != variant
                or bundle.truth.recipe.variant.value != variant
                or row.trace_sha256 != sample.trace.sha256
                or listed != sample.trace.sha256
                or row.episode_sha256 != sha256_bytes(canonical_json_bytes(sample.episode))
                or row.score != sample.expected_result.score
            ):
                raise ValueError("selected replay must match its raw episode witness and variant")
            if sample.config_canonical_json != canonical_json_bytes(runtime).decode():
                raise ValueError(
                    "selected runtime configuration differs from full gate configuration"
                )
        if self.passed != _passes(totals, self.requested_episode_count, self.provenance):
            raise ValueError("gate passed must be derived from raw evidence")
        return self


class Phase2EngineGateData(_GateFields):
    """Production serialized schema for independent consumers, not a gate verdict."""

    schema_version: Literal["phase2-engine-gate-v1"]
    access_class: Literal["validation_private"]
    profile: Literal["production"]
    requested_episode_count: Literal[10_000]


class Phase2EngineGateDebugData(_GateFields):
    """Explicit debug serialized schema, without producer-derived validation."""

    schema_version: Literal["phase2-engine-gate-debug-v1"]
    access_class: Literal["debug"]
    profile: Literal["debug"]


class Phase2EngineGateReport(_ValidatedGateFields):
    schema_version: Literal["phase2-engine-gate-v1"]
    access_class: Literal["validation_private"]
    profile: Literal["production"]
    requested_episode_count: Literal[10_000]


class Phase2EngineGateDebugReport(_ValidatedGateFields):
    schema_version: Literal["phase2-engine-gate-debug-v1"]
    access_class: Literal["debug"]
    profile: Literal["debug"]


def parse_phase2_gate_bytes(raw: bytes, *, allow_debug: bool = False):
    """Parse exactly one bounded canonical snapshot; debug dispatch is explicit."""
    try:
        if len(raw) > MAX_GATE_BYTES:
            raise ValueError("gate byte limit")
        payload = json.loads(raw)
        schema = payload.get("schema_version") if isinstance(payload, dict) else None
        if schema == "phase2-engine-gate-v1":
            model = Phase2EngineGateReport
        elif allow_debug and schema == "phase2-engine-gate-debug-v1":
            model = Phase2EngineGateDebugReport
        else:
            raise ValueError("closed gate schema")
        report = model.model_validate_json(raw)
        if raw != canonical_json_bytes(report):
            raise ValueError("gate must be canonical JSON without duplicate keys")
        return report
    except (ValueError, TypeError, RecursionError, SilentCascadeError) as error:
        raise ArtifactIntegrityError("invalid Phase 2 gate artifact") from error


def _read(path: Path, cap: int) -> bytes:
    try:
        with archive_parent(path, error_factory=ArtifactIntegrityError) as (parent, name):
            return read_archive_at(
                parent, name, max_bytes=cap, error_factory=ArtifactIntegrityError
            )
    except OSError as error:
        raise ArtifactIntegrityError("gate input must be a bounded regular file") from error


def _remove_owned_gate_at(
    parent: int, name: str, owned: os.stat_result, raw: bytes | None = None
) -> None:
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    identity = (owned.st_dev, owned.st_ino)
    if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != identity:
        raise ValueError("replacement is not our publication inode")
    if raw is not None:
        observed = read_archive_at(
            parent, name, max_bytes=MAX_GATE_BYTES, error_factory=ArtifactIntegrityError
        )
        if observed != raw:
            raise ValueError("publication contents changed")
    after = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if (after.st_dev, after.st_ino) != identity:
        raise ValueError("publication identity changed during cleanup")
    os.unlink(name, dir_fd=parent)


def _publish_gate(path: Path, raw: bytes) -> None:
    """Publish a retained owned inode, then roll back only that inode on failure.

    Parent descriptors pin every operation. Keeping the staged descriptor open
    prevents inode reuse; identical-byte replacements are foreign too. Hostile
    replacement between the final identity check and unlink remains outside the
    documented local single-writer boundary. Shared Phase 1 I/O is unchanged.
    """
    with archive_parent(path, error_factory=ArtifactIntegrityError) as (parent, name):
        temporary = f".{name}.{uuid4().hex}.tmp"
        descriptor = None
        owned = None
        temporary_exists = False
        published = False
        try:
            descriptor = os.open(
                temporary, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent
            )
            temporary_exists = True
            owned = os.fstat(descriptor)
            with os.fdopen(os.dup(descriptor), "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
            published = True
            _remove_owned_gate_at(parent, temporary, owned)
            temporary_exists = False
            os.fsync(parent)
        except (OSError, ValueError, SilentCascadeError) as error:
            cleanup_errors = []
            for leaf, should_remove, expected_bytes in (
                (name, published, raw),
                (temporary, temporary_exists, None),
            ):
                if should_remove:
                    try:
                        if owned is None:
                            # Before-link identity capture failed. The retained
                            # descriptor still identifies our original temp file.
                            owned = os.fstat(descriptor)
                        _remove_owned_gate_at(parent, leaf, owned, expected_bytes)
                    except (OSError, ValueError, SilentCascadeError) as cleanup_error:
                        cleanup_errors.append(str(cleanup_error))
            if published or temporary_exists:
                try:
                    os.fsync(parent)
                except OSError as cleanup_error:
                    cleanup_errors.append(str(cleanup_error))
            if cleanup_errors:
                raise ArtifactIntegrityError(
                    "gate publication failed; rollback unconfirmed and residual output may remain",
                    context={
                        "published": published,
                        "rollback_failed": True,
                        "cleanup_errors": cleanup_errors,
                    },
                ) from error
            raise AtomicWriteError(
                "gate publication failed",
                context={"published": published, "rollback_failed": False},
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)


def collect_phase2_gate(
    *,
    repo_root: Path,
    resolved: ResolvedConfig[Phase2Config],
    manifest_path: Path,
    expected_source_commit: str,
    expected_plan_base_revision: str,
    output_path: Path,
    profile: Literal["production", "debug"] = "production",
    requested_episode_count: int = 10_000,
) -> Phase2EngineGateReport | Phase2EngineGateDebugReport:
    """Execute authenticated manifest recipes once, and each retained sample twice."""
    crash_root = output_path.with_name(output_path.name + ".crashes")
    try:
        # Refuse existing targets before execution; the owned-inode link remains
        # the authoritative no-clobber check against a concurrent writer.
        if output_path.exists() or output_path.is_symlink():
            raise ArtifactIntegrityError("gate output already exists")
        if profile not in {"production", "debug"} or type(requested_episode_count) is not int:
            raise ArtifactIntegrityError("invalid gate profile")
        if not 0 < requested_episode_count <= 10_000 or requested_episode_count % 4:
            raise ArtifactIntegrityError("gate denominator must be a positive multiple of four")
        raw = _read(manifest_path, MAX_MANIFEST_BYTES)
        envelope = ManifestEnvelope.model_validate_json(raw)
        if raw != canonical_json_bytes(envelope):
            raise ArtifactIntegrityError("manifest envelope must be canonical")
        manifest = verify_manifest(envelope)
        if manifest.episode_count != requested_episode_count:
            raise ArtifactIntegrityError("manifest denominator mismatch")
        if profile == "production":
            if (
                requested_episode_count != 10_000
                or manifest.access_class is not ManifestAccessClass.VALIDATION
                or sha256_bytes(raw) != VALIDATION_FILE_SHA256
                or manifest_path.absolute() != (repo_root / VALIDATION_MANIFEST_PATH).absolute()
            ):
                raise ArtifactIntegrityError(
                    "production gate requires the exact committed Phase 1 validation manifest"
                )
        elif manifest.access_class is not ManifestAccessClass.DEBUG:
            raise ArtifactIntegrityError("debug profile requires a separate debug manifest")
        if any(
            not isinstance(entry.coordinate, MatchedManifestCoordinate)
            for entry in manifest.entries
        ):
            raise ArtifactIntegrityError("gate requires matched manifest recipes")
        provenance_args = dict(
            repo_root=repo_root,
            manifest_path=manifest_path,
            manifest_raw=raw,
            manifest_payload_sha256=envelope.payload_sha256,
            expected_source_commit=expected_source_commit,
            expected_plan_base_revision=expected_plan_base_revision,
        )
        provenance = collect_phase2_evidence_provenance(resolved, **provenance_args)
        phase1_payload = resolved.config.model_dump(mode="json")
        del phase1_payload["event_flow"]
        if sha256_bytes(canonical_json_bytes(phase1_payload)) != manifest.provenance.config_sha256:
            raise ArtifactIntegrityError("manifest generator configuration mismatch")
        rows, samples = [], {}
        for index, entry in enumerate(manifest.entries):
            bundle = regenerate_entry(resolved.config, manifest, entry)
            engine = EventEngine(
                resolved.config.event_flow,
                crash_root=crash_root,
                source_revision=expected_source_commit,
            )
            result = engine.run_episode(bundle, ScriptedEventFlowAgent(device="cpu"))
            trace = result.trace.events
            internal = [event for event in trace if event.kind in {"recall", "compose", "act"}]
            variant = bundle.truth.recipe.variant.value
            row = EpisodeGateWitness(
                entry_index=index,
                episode_public_id=entry.episode_public_id,
                episode_sha256=entry.episode_sha256,
                manifest_entry_sha256=sha256_bytes(canonical_json_bytes(entry)),
                coordinate=entry.coordinate,
                variant=variant,
                config_sha256=resolved.sha256,
                trace_sha256=result.trace.sha256,
                completed=True,
                score=result.score,
                dynamics_failure=False,
                replay_failure=False,
                provenance_failure=False,
                internal_event_count=len(internal),
                minimum_internal_gap=min((event.delta for event in internal), default=None),
                gap_clamp_count=sum(event.was_gap_clamped for event in trace),
                event_cap_failure=False,
                time_reversal_count=sum(
                    right.timestamp < left.timestamp for left, right in pairwise(trace)
                ),
                foundation_model_calls=result.counters.foundation_model_calls,
            )
            rows.append(row)
            if variant not in samples:
                runtime_raw = canonical_json_bytes(resolved.config.event_flow)
                payload = dict(
                    schema_version="phase2-replay-v1",
                    access_class="validation_private",
                    episode=EpisodeArtifact.from_bundle(bundle).model_dump(mode="json"),
                    config_canonical_json=runtime_raw.decode(),
                    config_sha256=sha256_bytes(runtime_raw),
                    condition=Condition.EVENT_FLOW,
                    agent_implementation="scripted-event-flow-v1",
                    expected_result=EpisodeResultArtifact.from_result(result).model_dump(
                        mode="json"
                    ),
                    trace=CausalTraceArtifact.from_trace(result.trace).model_dump(mode="json"),
                )
                payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
                sample = parse_replay_artifact_bytes(canonical_json_bytes(payload))
                for _ in range(2):
                    verify_replay(sample)
                samples[variant] = sample
            # Full trace and trajectory are released here, before the next episode.
            del result, trace, internal, bundle, engine
        witnesses = tuple(rows)
        if set(samples) != set(VARIANTS):
            raise ArtifactIntegrityError("gate is missing a variant replay witness")
        totals = _totals(witnesses)
        final_provenance = collect_phase2_evidence_provenance(resolved, **provenance_args)
        if final_provenance != provenance or _read(manifest_path, MAX_MANIFEST_BYTES) != raw:
            raise ArtifactIntegrityError("gate source or manifest changed during execution")
        model = Phase2EngineGateReport if profile == "production" else Phase2EngineGateDebugReport
        report = model(
            schema_version="phase2-engine-gate-v1"
            if profile == "production"
            else "phase2-engine-gate-debug-v1",
            access_class="validation_private" if profile == "production" else "debug",
            profile=profile,
            provenance=provenance,
            validation_manifest_payload_sha256=envelope.payload_sha256,
            validation_manifest_file_sha256=sha256_bytes(raw),
            config_sha256=resolved.sha256,
            generator_version="ofd-v1",
            condition="scripted_event_flow",
            agent_implementation="scripted-event-flow-v1",
            requested_episode_count=requested_episode_count,
            episode_witnesses=witnesses,
            trace_chain_sha256=_chain(witnesses),
            selected_replay_samples=tuple(samples[v] for v in VARIANTS),
            selected_replay_trace_hashes=tuple(samples[v].trace.sha256 for v in VARIANTS),
            passed=_passes(totals, requested_episode_count, provenance),
            **totals,
        )
        encoded = canonical_json_bytes(report)
        if len(encoded) > MAX_GATE_BYTES:
            raise ArtifactIntegrityError("gate exceeds output byte limit")
        _publish_gate(output_path, encoded)
        return report
    except Exception as caught:
        error = (
            caught
            if isinstance(caught, SilentCascadeError)
            else ArtifactIntegrityError("Phase 2 gate collection failed")
        )
        write_crash_bundle(
            crash_root,
            error=error,
            context=CrashContext(
                config_sha256=resolved.sha256, source_revision=expected_source_commit
            ),
            sanitize_diagnostics=True,
        )
        if error is caught:
            raise
        raise error from caught
