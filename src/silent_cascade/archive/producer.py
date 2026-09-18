"""Durable offline producer handoff through the existing authenticated session."""

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import model_validator

from silent_cascade.archive.catalog import _pinned_directory, _safe_logical_path
from silent_cascade.archive.ledger import StorageBlocked
from silent_cascade.archive.types import (
    ArchivePolicy,
    EpisodeCommit,
    FileEntry,
    Hash,
    UnitIdentity,
    UnitRef,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.io import atomic_create_bytes
from silent_cascade.validation import StrictModel

EVALUATION_BYTES = 128 * 1024**2
PILOT_BYTES = 64 * 1024**2
TELEMETRY_BYTES = 128
COMMIT_BYTES = 64 * 1024
ALLOCATION_OVERHEAD = 256 * 1024


class JournalSegmentCommit(StrictModel):
    schema_version: Literal["phase4-journal-segment-commit-v1"] = "phase4-journal-segment-commit-v1"
    base_head: Hash | None
    sealed_head: Hash
    record_sha256s: tuple[Hash, ...]

    @model_validator(mode="after")
    def validate_chain_shape(self):
        if (
            not self.record_sha256s
            or self.record_sha256s[0] != self.sealed_head
            or len(set(self.record_sha256s)) != len(self.record_sha256s)
            or self.base_head in self.record_sha256s
        ):
            raise ValueError("journal segment heads or unique record order differ")
        return self


def seal_journal_segments(
    *,
    run_dir: Path,
    control_dir: Path,
    base_head: str | None,
    sealed_head: str,
    policy: ArchivePolicy,
    session,
    identity: UnitIdentity,
) -> tuple[UnitRef, ...]:
    """Seal exact immutable chain pieces through the already-authenticated session."""
    from silent_cascade.train.pilot_data import _read_pilot_bytes

    if session.run_dir != run_dir.absolute() or session.control_dir != control_dir.absolute():
        raise ValueError("journal session roots differ")
    groups, current, size, seen = [], [], 0, set()
    cursor = sealed_head
    while cursor != base_head:
        if cursor is None or cursor in seen or not re.fullmatch(r"[0-9a-f]{64}", cursor):
            raise ValueError("journal segment predecessor missing or cyclic")
        seen.add(cursor)
        raw = _read_pilot_bytes(run_dir / f"journal-{cursor}.json")
        record = json.loads(raw)
        if sha256_bytes(raw) != cursor or not isinstance(record, dict) or "prior" not in record:
            raise ValueError("journal record hash or predecessor differs")
        if len(raw) > policy.journal_bytes:
            raise StorageBlocked("storage_blocked: individual journal exceeds segment policy")
        if current and (
            len(current) == policy.journal_records or size + len(raw) > policy.journal_bytes
        ):
            groups.append((tuple(current), cursor))
            current, size = [], 0
        current.append(cursor)
        size += len(raw)
        cursor = record["prior"]
    if current:
        groups.append((tuple(current), base_head))
    refs = []
    for hashes, predecessor in reversed(groups):
        commit = JournalSegmentCommit(
            base_head=predecessor, sealed_head=hashes[0], record_sha256s=hashes
        )
        bound = identity.model_copy(
            update={"evidence_identity_sha256": sha256_bytes(canonical_json_bytes(commit))}
        )
        response = session._request(
            "seal",
            {
                "logical_root": ".",
                "paths": sorted(f"journal-{digest}.json" for digest in hashes),
                "kind": "journal",
                "identity": bound.model_dump(mode="json"),
                "episode_groups": [],
                "borrowed": [],
            },
        )
        if set(response) != {"ref"}:
            raise ValueError("journal seal response differs")
        ref = UnitRef(**response["ref"])
        if ref.file_count != len(hashes) or ref.kind != "journal" or ref.logical_root != ".":
            raise ValueError("journal seal ownership differs")
        refs.append(ref)
    return tuple(refs)


@dataclass(frozen=True)
class ArtifactOutputBounds:
    """Existing serializer limits times source-derived possible publication counts.

    An episode has one sidecar, one trajectory, at most one crash JSON and one
    runtime checkpoint; shared weights are separately pinned. A row can consume
    the existing evaluation format limit, never an invented smaller row cap.
    """

    episode_owned: int = 4 * EVALUATION_BYTES + TELEMETRY_BYTES
    episode_row: int = EVALUATION_BYTES
    shared_weights: int = EVALUATION_BYTES
    journal_record: int = PILOT_BYTES
    commit: int = COMMIT_BYTES


def before_work_bounds(operation):
    """Parent-derived additional maxima; no child-supplied capacity numbers."""
    bounds = ArtifactOutputBounds()
    if operation == "episode":
        return {
            "spool": bounds.episode_owned + bounds.episode_row + ALLOCATION_OVERHEAD,
            "pinned": bounds.shared_weights + ALLOCATION_OVERHEAD,
            "metadata": bounds.commit + ALLOCATION_OVERHEAD,
        }
    if operation == "update":
        return {"spool": bounds.journal_record + ALLOCATION_OVERHEAD}
    if operation == "evaluation":
        return {"spool": 2 * EVALUATION_BYTES + ALLOCATION_OVERHEAD}
    if operation == "checkpoint":
        return {"spool": 2 * EVALUATION_BYTES + ALLOCATION_OVERHEAD}
    raise ValueError("unknown producer admission operation")


def _entry(root, path, expected=None):
    from silent_cascade.eval.artifacts import read_evaluation_artifact

    relative = path.absolute().relative_to(root.absolute()).as_posix()
    _safe_logical_path(relative, field="published evidence")
    raw = read_evaluation_artifact(path)
    digest = sha256_bytes(raw)
    if expected is not None and digest != expected:
        raise ValueError("published evidence hash differs")
    return FileEntry(relative, digest, len(raw))


def verify_crash(*, root, logical_root, row, crash, identity):
    """Authenticate explicit manifest/checkpoint/weights links without discovery."""
    from silent_cascade.eval.artifacts import read_evaluation_artifact
    from silent_cascade.eventflow.neural_checkpoint import NeuralRuntimeMetadata
    from silent_cascade.eventflow.neural_weights import WeightsMetadata, decode_archive
    from silent_cascade.logging.crash_bundle import CrashBundleManifest

    if (row.error is None) != (crash is None):
        raise ValueError("crash ownership differs from row outcome")
    if crash is None:
        return None
    manifest_entry = _entry(root, crash.manifest.path, crash.manifest.sha256)
    manifest = CrashBundleManifest.model_validate_json(
        read_evaluation_artifact(crash.manifest.path)
    )
    if (
        manifest.context.episode_public_id != row.public_id
        or manifest.context.source_revision != identity.execution_source_revision
    ):
        raise ValueError("crash manifest identity differs")
    if crash.checkpoint is None:
        if manifest.context.checkpoint_ref is not None or row.causal_trace_sha256 is not None:
            raise ValueError("initialization crash checkpoint differs")
        if crash.shared_weights is not None:
            raise ValueError("initialization crash cannot own a weights reference")
    else:
        if crash.shared_weights is None or (
            crash.checkpoint.path.parent != crash.manifest.path.parent
            or crash.checkpoint.path.name != manifest.context.checkpoint_ref
        ):
            raise ValueError("crash checkpoint reference differs")
        _entry(root, crash.checkpoint.path, crash.checkpoint.sha256)
        _entry(root, crash.shared_weights.path, crash.shared_weights.sha256)
        metadata, _ = decode_archive(
            read_evaluation_artifact(crash.checkpoint.path),
            crash.checkpoint.sha256,
            NeuralRuntimeMetadata,
            "runtime",
        )
        weights, _ = decode_archive(
            read_evaluation_artifact(crash.shared_weights.path),
            crash.shared_weights.sha256,
            WeightsMetadata,
            "weights",
        )
        if (
            metadata.public_init.episode_public_id != row.public_id
            or metadata.source_revision != identity.execution_source_revision
            or metadata.weights_ref != crash.shared_weights.path.name
            or metadata.weights_sha256 != crash.shared_weights.sha256
            or metadata.weights != weights
            or weights.identity != identity.model_identity
            or metadata.experiment_config_canonical_json
            != identity.evaluation_config_canonical_json
        ):
            raise ValueError("crash checkpoint/weights identity differs")
    return {
        "path": Path(manifest_entry.path).relative_to(logical_root).as_posix(),
        "sha256": manifest_entry.sha256,
    }


class ArchiveProducer:
    def __init__(self, *, session):
        self.session = session
        self.run_dir = session.run_dir
        self.control_dir = session.control_dir
        self.policy = session.policy
        self._evaluations = {}
        self._resident = {}
        self._metadata = {}
        self._training_identity = None
        self._journal_base = None
        self._journal_head = None
        self._journal_pending = []
        self._journal_bytes = 0

    def bind_training(self, *, source_commit, config_sha256, journal_head=None):
        self._training_identity = UnitIdentity(
            run_id=self.session._identity["run_id"],
            source_commit=source_commit,
            config_sha256=config_sha256,
            evidence_identity_sha256=config_sha256,
            checkpoint_sha256=None,
            writer_stopped=False,
            checkpoint_committed=False,
        )
        self._journal_base = self._journal_head = journal_head

    def _flush_journals(self):
        if not self._journal_pending:
            return
        refs = seal_journal_segments(
            run_dir=self.run_dir,
            control_dir=self.control_dir,
            base_head=self._journal_base,
            sealed_head=self._journal_head,
            policy=self.policy,
            session=self.session,
            identity=self._training_identity,
        )
        # Each flush is at most one segment. All unrelated known files stay resident.
        if len(refs) != 1:
            raise ValueError("journal rolling tail exceeded one segment")
        self._archive(refs[0], evict=True, owned=tuple(self._journal_pending))
        self._journal_base = self._journal_head
        self._journal_pending = []
        self._journal_bytes = 0

    def after_journal(self, logical_path: str, journal_sha256: str) -> None:
        from silent_cascade.train.pilot_data import _read_pilot_bytes

        if logical_path != f"journal-{journal_sha256}.json":
            raise ValueError("journal publication path differs")
        raw = _read_pilot_bytes(self.run_dir / logical_path)
        record = json.loads(raw)
        if sha256_bytes(raw) != journal_sha256 or record["prior"] != self._journal_head:
            raise ValueError("journal publication predecessor differs")
        if len(raw) > self.policy.journal_bytes:
            raise StorageBlocked("storage_blocked: journal record exceeds policy")
        # Include the newly durable record in retained custody while closing the old tail.
        self.track_file(self.run_dir / logical_path, journal_sha256)
        if self._journal_pending and self._journal_bytes + len(raw) > self.policy.journal_bytes:
            self._flush_journals()
        self._journal_head = journal_sha256
        self._journal_pending.append(logical_path)
        self._journal_bytes += len(raw)
        if (
            len(self._journal_pending) >= self.policy.journal_records
            or self._journal_bytes >= self.policy.journal_bytes
        ):
            self._flush_journals()

    def after_checkpoint(self, descriptor, progress) -> None:
        if descriptor.global_step != progress.global_step:
            raise ValueError("checkpoint progress differs")
        self.track_file(self.run_dir / descriptor.path, descriptor.sha256)
        self.track_file(self.run_dir / "checkpoint-index.json")
        # Checkpoint pruning is owned by the existing trainer and leaves its descriptors.
        self._resident = {
            path: entry for path, entry in self._resident.items() if (self.run_dir / path).exists()
        }
        self._flush_journals()
        identity = self._training_identity.model_copy(
            update={
                "checkpoint_sha256": descriptor.sha256,
                "checkpoint_committed": True,
                "evidence_identity_sha256": sha256_bytes(canonical_json_bytes(progress)),
            }
        )
        ref = self._seal(
            logical_root=".",
            paths=(descriptor.path, "checkpoint-index.json"),
            kind="control_snapshot",
            identity=identity.model_dump(mode="json"),
        )
        self._archive(ref, evict=False)

    def after_validation(self, logical_root: str) -> None:
        evaluation_root = logical_root + "/autonomous"
        identity = self._evaluations[evaluation_root]
        paths = [logical_root + "/validation.json"]
        if identity.stage == "one_hop":
            paths.append(logical_root + "/components.json")
        for path in paths:
            self.track_file(self.run_dir / path)
        ref = self._seal(
            logical_root=logical_root,
            paths=tuple(paths),
            kind="diagnostic",
            identity=self._identity(evaluation_root),
        )
        self._archive(ref, evict=False)

    def logical_root(self, directory):
        logical = directory.absolute().relative_to(self.run_dir).as_posix()
        _safe_logical_path(logical, field="evaluation root")
        return logical

    def bind_evaluation(self, logical_root, identity):
        if logical_root in self._evaluations:
            raise ValueError("same-directory evaluation resume is forbidden")
        self._evaluations[logical_root] = identity

    def _before(self, operation):
        if self.session._request("status", {"before_work": operation}) != {"admitted": True}:
            raise StorageBlocked("storage_blocked: producer admission not authenticated")

    def before_update(self, global_step: int) -> None:
        if type(global_step) is not int or global_step < 0:
            raise ValueError("invalid update step")
        self._before("update")

    def before_evaluation(self, logical_root: str) -> None:
        self._evaluations[logical_root]
        self._before("evaluation")

    def before_episode(self, logical_root: str, ordinal: int) -> None:
        identity = self._evaluations[logical_root]
        if type(ordinal) is not int or not 0 <= ordinal < len(identity.episodes):
            raise ValueError("episode ordinal differs")
        if ArtifactOutputBounds().episode_owned > self.policy.episode_bytes:
            raise StorageBlocked("storage_blocked: complete episode exceeds policy admission")
        self._before("episode")

    def make_commit(self, *, logical_root, ordinal, row, offset, payload, publication):
        root = self.run_dir / logical_root
        owned = [
            _entry(self.run_dir, root / row.neural_trace_ref, row.neural_trace_sha256),
            _entry(self.run_dir, root / publication.telemetry_ref),
        ]
        if row.full_trace_ref is not None:
            owned.append(_entry(self.run_dir, root / row.full_trace_ref, row.full_trace_sha256))
        crash = publication.crash
        verify_crash(
            root=self.run_dir,
            logical_root=logical_root,
            row=row,
            crash=crash,
            identity=self._evaluations[logical_root],
        )
        if crash is not None:
            for item in (crash.manifest, crash.checkpoint):
                if item is not None:
                    owned.append(_entry(self.run_dir, item.path, item.sha256))
        borrowed = (
            ()
            if publication.shared_weights is None
            else (
                _entry(
                    self.run_dir, publication.shared_weights.path, publication.shared_weights.sha256
                ),
            )
        )
        commit = EpisodeCommit(
            "phase4-evaluation-episode-commit-v1",
            row.identity_sha256,
            ordinal,
            row.public_id,
            row.episode_sha256,
            offset,
            len(payload),
            sha256_bytes(payload),
            tuple(sorted(owned, key=lambda entry: entry.path)),
            borrowed,
        )
        raw = canonical_json_bytes(asdict(commit))
        if len(raw) > COMMIT_BYTES:
            raise ValueError("episode commit exceeds fixed ownership metadata bound")
        atomic_create_bytes(self.control_dir / "episode-pending.json", raw)
        return commit

    def _identity(self, logical_root):
        identity = self._evaluations[logical_root]
        return UnitIdentity(
            run_id=self.session._identity["run_id"],
            source_commit=identity.execution_source_revision,
            config_sha256=sha256_bytes(identity.evaluation_config_canonical_json.encode()),
            evidence_identity_sha256=identity.sha256,
            checkpoint_sha256=identity.checkpoint_sha256,
            writer_stopped=False,
            checkpoint_committed=False,
        ).model_dump(mode="json")

    def _seal(self, *, logical_root, paths, kind, identity, groups=(), borrowed=()):
        response = self.session._request(
            "seal",
            {
                "logical_root": logical_root,
                "paths": list(paths),
                "kind": kind,
                "identity": identity,
                "episode_groups": list(groups),
                "borrowed": [asdict(entry) for entry in borrowed],
            },
        )
        if set(response) != {"ref"}:
            raise ValueError("invalid unit seal response")
        return UnitRef(**response["ref"])

    def _archive(self, ref, *, evict, owned=()):
        owned = set(owned)
        retained = tuple(
            entry
            for path, entry in sorted(self._resident.items())
            if path not in owned
            and (ref.logical_root == "." or Path(path).is_relative_to(ref.logical_root))
        )
        response = self.session._request(
            "archive",
            {
                "ref": asdict(ref),
                "evict": evict,
                "retained": [asdict(entry) for entry in retained] if evict else [],
            },
        )
        if set(response) != {"receipt_sha256"} or not re.fullmatch(
            r"[0-9a-f]{64}", response["receipt_sha256"]
        ):
            raise ValueError("archive receipt response differs")
        if evict:
            for path in owned:
                self._resident.pop(path, None)

    def track_file(self, path, digest=None):
        entry = _entry(self.run_dir, path, digest)
        self._resident[entry.path] = entry
        return entry

    def after_episode(self, logical_root: str, commit: EpisodeCommit) -> None:
        paths = tuple(entry.path for entry in commit.owned)
        raw = canonical_json_bytes(asdict(commit))
        if (self.control_dir / "episode-pending.json").read_bytes() != raw:
            raise ValueError("durable episode envelope differs")
        for name in ("identity.json", "retention.json", ".rows.pending.jsonl"):
            self.track_file(self.run_dir / logical_root / name)
        for entry in (*commit.owned, *commit.borrowed):
            self._resident[entry.path] = entry
        ref = self._seal(
            logical_root=logical_root,
            paths=paths,
            kind="episode_pack",
            identity=self._identity(logical_root),
            groups=(paths,),
            borrowed=commit.borrowed,
        )
        response = self.session._request(
            "seal",
            {
                "logical_root": logical_root,
                "unit_ref": asdict(ref),
                "commit": asdict(commit),
            },
        )
        if response != {"commit_sha256": sha256_bytes(raw)}:
            raise ValueError("episode binding digest differs")
        self._archive(ref, evict=True, owned=paths)
        # The verified parent binding now holds these exact canonical bytes.
        with _pinned_directory(self.control_dir) as descriptor:
            os.unlink("episode-pending.json", dir_fd=descriptor)
            os.fsync(descriptor)

    def after_evaluation(self, logical_root: str) -> None:
        root = self.run_dir / logical_root
        done = json.loads((root / "DONE").read_bytes())
        paths = [
            str(Path(logical_root) / name)
            for name in (
                "identity.json",
                "retention.json",
                "rows.jsonl",
                "index.json",
                "metrics.json",
                "DONE",
            )
        ]
        if "crashes/index.json" in done["artifact_hashes"]:
            paths.append(str(Path(logical_root) / "crashes/index.json"))
        ref = self._seal(
            logical_root=logical_root,
            paths=tuple(sorted(paths)),
            kind="evaluation_metadata",
            identity=self._identity(logical_root),
        )
        # Rows are needed immediately by the unchanged ValidationRecord writer.
        self._archive(ref, evict=False)
        self._resident.pop(str(Path(logical_root) / ".rows.pending.jsonl"), None)
        for path in paths:
            self.track_file(self.run_dir / path)
        self._metadata[logical_root] = (ref, tuple(paths))
