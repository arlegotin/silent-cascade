"""Read-only verification of pilot evidence; never executes training or evaluation."""

import gzip
import hashlib
import json
import math
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from silent_cascade.eval.artifacts import (
    EvaluationIdentity,
    _verify_evidence,
    read_evaluation_artifact,
)
from silent_cascade.eval.metrics import PilotMetrics, TimedEpisodeRow
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.crash_bundle import CrashBundleManifest


@dataclass(frozen=True)
class EvaluationHeader:
    identity: EvaluationIdentity
    retention: dict
    index: tuple[dict, ...]
    metrics: PilotMetrics
    crashes: dict
    done: dict
    metadata_paths: frozenset[str]

    @property
    def hashes(self):
        return self.done["artifact_hashes"]


@dataclass(frozen=True)
class IndexedRow:
    row: TimedEpisodeRow
    ordinal: int
    offset: int
    raw: bytes


@dataclass(frozen=True)
class VerifiedEpisode:
    indexed_row: IndexedRow
    sidecar: dict
    owned: frozenset[str]
    shared: frozenset[str]
    crash: dict | None
    trajectory: dict | None = None

    @property
    def row(self):
        return self.indexed_row.row


@dataclass
class EvaluationScanAccumulator:
    """Only counters, identities and classified paths survive episode leases."""

    count: int = 0
    offset: int = 0
    owned: set[str] = field(default_factory=set)
    shared: set[str] = field(default_factory=set)
    failures: set[str] = field(default_factory=set)
    public_ids: set[str] = field(default_factory=set)
    episode_ids: set[str] = field(default_factory=set)
    identity_ids: set[str] = field(default_factory=set)
    totals: Counter = field(default_factory=Counter)
    misses: Counter = field(default_factory=Counter)
    eligible: bool = True
    stream: object = field(default_factory=hashlib.sha256)

    def add(self, episode: VerifiedEpisode):
        indexed, row = episode.indexed_row, episode.row
        if indexed.ordinal != self.count or indexed.offset != self.offset:
            raise ValueError("evaluation scan ordinal/offset differs")
        if self.owned & (episode.owned | episode.shared) or self.shared & episode.owned:
            raise ValueError("duplicate evaluation evidence ownership")
        self.owned.update(episode.owned)
        self.shared.update(episode.shared)
        self.count += 1
        self.offset += len(indexed.raw)
        self.stream.update(indexed.raw)
        self.public_ids.add(row.public_id)
        self.episode_ids.add(row.episode_sha256)
        self.identity_ids.add(row.identity_sha256)
        self.eligible &= row.gate_eligible
        self.totals.update(
            success=int(row.error is None and row.timed_success),
            errors=int(row.error is not None),
            negatives=int(not row.is_positive),
            false_actions=int(not row.is_positive and bool(row.actions)),
            foundation=row.compute.foundation_model_calls,
            positive_success=int(row.is_positive and row.timed_success),
            negative_success=int(not row.is_positive and row.timed_success),
            dynamics=int(
                row.error is not None and row.error.code in {"dynamics_error", "time_order_error"}
            ),
        )
        if row.error is not None:
            self.failures.add(row.public_id)
        if row.miss_category:
            self.misses[row.miss_category] += 1

    def metrics(self):
        n, t = self.count, self.totals
        eligible = (
            n == 10_000
            and t["negatives"] == 5_000
            and self.eligible
            and len(self.public_ids) == n
            and len(self.episode_ids) == n
            and len(self.identity_ids) == 1
        )
        return PilotMetrics(
            episode_count=n,
            timed_success_count=t["success"],
            positive_count=n - t["negatives"],
            negative_count=t["negatives"],
            positive_success_count=t["positive_success"],
            negative_success_count=t["negative_success"],
            false_action_count=t["false_actions"],
            dynamics_error_count=t["dynamics"],
            error_count=t["errors"],
            foundation_model_calls=t["foundation"],
            timed_success_rate=t["success"] / n if n else 0.0,
            negative_false_action_rate=t["false_actions"] / t["negatives"]
            if t["negatives"]
            else 0.0,
            miss_counts=tuple(sorted(self.misses.items())),
            validation_valid=bool(n) and t["errors"] == 0 and t["foundation"] == 0,
            gate_passed=eligible
            and t["success"] >= 9_000
            and t["false_actions"] <= 500
            and t["errors"] == 0
            and t["foundation"] == 0,
        )


def child(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("unsafe artifact path")
    if any(part in {"..", ".", "frozen", "frozen_test"} for part in name.split("/")):
        raise ValueError("unsafe artifact path")
    return root / name


def read_json(path: Path):
    return _decode_json(read_evaluation_artifact(path))


def _decode_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate artifact key")
            value[key] = item
        return value

    return json.loads(
        raw,
        object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def verify_hashes(root: Path, hashes: dict, *, retain=()) -> dict[str, bytes]:
    """Hash every bounded file; keep raw bytes only for explicitly requested inputs."""
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("missing artifact inventory")
    verified = {}
    for name, digest in hashes.items():
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid artifact hash")
        raw = read_evaluation_artifact(child(root, name))
        if sha256_bytes(raw) != digest:
            raise ValueError(f"artifact integrity mismatch: {name}")
        if name in retain:
            verified[name] = raw
    return verified


def _verify_row_identity(row, binding, identity):
    if (
        row.public_id,
        row.episode_sha256,
        row.identity_sha256,
        row.manifest_sha256,
        row.checkpoint_sha256,
        row.model_state_sha256,
        row.producing_source_revision,
        row.execution_source_revision,
        row.config_sha256,
        row.purpose,
        row.gate_eligible,
    ) != (
        binding.public_id,
        binding.episode_sha256,
        identity.sha256,
        identity.manifest_sha256,
        identity.checkpoint_sha256,
        identity.model_identity.model_state_sha256,
        identity.model_identity.source_revision,
        identity.execution_source_revision,
        sha256_bytes(identity.evaluation_config_canonical_json.encode()),
        identity.purpose,
        identity.gate_eligible,
    ):
        raise ValueError("row artifact identity mismatch")
    if (
        sha256_bytes(canonical_json_bytes(row.model_dump(mode="json")["truth"]))
        != binding.scoring_truth_sha256
        or row.truth.recipe.variant.value != binding.variant
        or row.truth.recipe.requested_path_length != binding.path_length
    ):
        raise ValueError("row artifact scoring truth mismatch")


def load_evaluation_header(root: Path) -> EvaluationHeader:
    """Authenticate the original completion controls without opening episode payloads."""
    done = read_json(root / "DONE")
    if (
        set(done)
        != {
            "identity_sha256",
            "artifact_hashes",
            "execution_complete",
            "gate_passed",
            "validation_valid",
        }
        or done["execution_complete"] is not True
    ):
        raise ValueError("incomplete evaluation artifact")
    hashes = done["artifact_hashes"]
    required = {"identity.json", "retention.json", "rows.jsonl", "index.json", "metrics.json"}
    if not isinstance(hashes, dict) or not required <= hashes.keys():
        raise ValueError("missing evaluation artifact")
    for name, digest in hashes.items():
        child(root, name)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid artifact hash")
    metadata = required - {"rows.jsonl"}
    if "crashes/index.json" in hashes:
        metadata.add("crashes/index.json")
    verified = verify_hashes(root, {name: hashes[name] for name in metadata}, retain=metadata)
    identity = EvaluationIdentity.model_validate_json(verified["identity.json"])
    if identity.sha256 != done["identity_sha256"]:
        raise ValueError("evaluation identity hash mismatch")
    retention = _decode_json(verified["retention.json"])
    if retention != {
        "stratified_public_ids": sorted(identity.retained_public_ids()),
        "report_example_public_ids": list(identity.report_example_public_ids),
        "every_failure": True,
        "every_delay_swap": True,
    }:
        raise ValueError("evaluation retention differs from identity")
    index = _decode_json(verified["index.json"])
    if not isinstance(index, dict) or set(index) != {"rows"} or not isinstance(index["rows"], list):
        raise ValueError("evaluation index mismatch")
    if len(index["rows"]) != len(identity.episodes):
        raise ValueError("incomplete evaluation denominator")
    offset = 0
    for entry, binding in zip(index["rows"], identity.episodes, strict=True):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"public_id", "offset", "bytes", "sha256"}
            or entry["public_id"] != binding.public_id
            or type(entry["offset"]) is not int
            or entry["offset"] != offset
            or type(entry["bytes"]) is not int
            or entry["bytes"] <= 0
            or not isinstance(entry["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        ):
            raise ValueError("evaluation index mismatch")
        offset += entry["bytes"]
        if offset > 128 * 1024**2:
            raise ValueError("evaluation row log exceeds format bound")
    crashes = (
        _decode_json(verified["crashes/index.json"]) if "crashes/index.json" in verified else {}
    )
    if not isinstance(crashes, dict):
        raise ValueError("invalid crash index")
    return EvaluationHeader(
        identity,
        retention,
        tuple(index["rows"]),
        PilotMetrics.model_validate_json(verified["metrics.json"]),
        crashes,
        done,
        frozenset(metadata | {"rows.jsonl"}),
    )


def iter_evaluation_rows(root: Path, header: EvaluationHeader):
    """Read exact indexed slices, with final raw stream authentication at exhaustion."""
    from silent_cascade.eventflow.archive_io import archive_parent

    with archive_parent(root / "rows.jsonl", error_factory=ValueError) as (parent, name):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 128 * 1024**2:
                raise ValueError("invalid or oversized evaluation row log")
            stream = hashlib.sha256()
            for ordinal, entry in enumerate(header.index):
                line = handle.read(entry["bytes"])
                if (
                    len(line) != entry["bytes"]
                    or not line.endswith(b"\n")
                    or b"\n" in line[:-1]
                    or sha256_bytes(line) != entry["sha256"]
                ):
                    raise ValueError("evaluation index row hash/length mismatch")
                stream.update(line)
                row = TimedEpisodeRow.model_validate_json(line)
                _verify_row_identity(row, header.identity.episodes[ordinal], header.identity)
                yield IndexedRow(row, ordinal, entry["offset"], line)
            if handle.read(1) or stream.hexdigest() != header.hashes["rows.jsonl"]:
                raise ValueError("evaluation row stream closure mismatch")


def verify_evaluation_episode(
    root: Path,
    *,
    header: EvaluationHeader,
    indexed_row: IndexedRow,
    decode_trajectory=False,
    borrowed_paths=None,
):
    """Consume semantic and crash evidence completely within the caller's episode lease."""
    row, identity = indexed_row.row, header.identity
    owned, shared = set(), set()

    def bound(name, *, shared_input=False):
        if name not in header.hashes:
            raise ValueError("unbound runtime artifact")
        raw = verify_hashes(root, {name: header.hashes[name]}, retain={name})[name]
        (shared if shared_input else owned).add(name)
        return raw

    for reference in (row.neural_trace_ref, row.full_trace_ref):
        if reference is not None:
            bound(reference)
    sidecar = _verify_evidence(
        root,
        row,
        retain=(
            row.public_id in identity.retained_public_ids()
            or row.public_id in identity.report_example_public_ids
            or identity.purpose == "delay_swap"
        ),
    )
    telemetry = _decode_json(bound(f"episodes/{indexed_row.ordinal:05d}.telemetry.json"))
    if (
        not isinstance(telemetry, dict)
        or set(telemetry) != {"elapsed_seconds"}
        or type(telemetry["elapsed_seconds"]) not in {int, float}
        or not math.isfinite(telemetry["elapsed_seconds"])
        or telemetry["elapsed_seconds"] < 0
    ):
        raise ValueError("invalid episode telemetry")
    weights_name = f"crashes/weights-{identity.checkpoint_sha256}.safetensors"
    if weights_name in header.hashes and (
        row.causal_trace_sha256 is not None
        or borrowed_paths is None
        or weights_name in borrowed_paths
    ):
        from silent_cascade.eventflow.neural_weights import WeightsMetadata, decode_archive

        weights, _ = decode_archive(
            bound(weights_name, shared_input=True),
            identity.checkpoint_sha256,
            WeightsMetadata,
            "weights",
        )
        if weights.identity != identity.model_identity:
            raise ValueError("shared weights identity differs")
    crash = header.crashes.get(row.public_id)
    if (row.error is None) != (crash is None):
        raise ValueError("crash denominator mismatch")
    if crash is not None:
        from silent_cascade.archive.producer import verify_crash
        from silent_cascade.eventflow.neural_checkpoint import NeuralRuntimeMetadata
        from silent_cascade.eventflow.neural_weights import decode_archive
        from silent_cascade.logging.crash_bundle import PublishedCrash, PublishedCrashFile

        if not isinstance(crash, dict) or set(crash) != {"path", "sha256"}:
            raise ValueError("invalid crash index entry")
        raw = bound(crash["path"])
        if sha256_bytes(raw) != crash["sha256"]:
            raise ValueError("crash index hash differs")
        manifest = CrashBundleManifest.model_validate_json(raw)
        checkpoint = weights = None
        if manifest.context.checkpoint_ref is not None:
            name = child(Path("crashes"), manifest.context.checkpoint_ref).as_posix()
            checkpoint_raw = bound(name)
            metadata, _ = decode_archive(
                checkpoint_raw, header.hashes[name], NeuralRuntimeMetadata, "runtime"
            )
            checkpoint = PublishedCrashFile(root / name, header.hashes[name])
            weights_name = child(Path("crashes"), metadata.weights_ref).as_posix()
            bound(weights_name, shared_input=True)
            weights = PublishedCrashFile(root / weights_name, header.hashes[weights_name])
        verify_crash(
            root=root,
            logical_root=".",
            row=row,
            identity=identity,
            crash=PublishedCrash(
                PublishedCrashFile(root / crash["path"], crash["sha256"]), checkpoint, weights
            ),
        )
    trajectory = None
    if decode_trajectory and row.full_trace_ref is not None:
        # The same bounded member has already passed full trajectory validation above.
        trajectory = _decode_json(gzip.decompress(bound(row.full_trace_ref)))
    return VerifiedEpisode(
        indexed_row, sidecar, frozenset(owned), frozenset(shared), crash, trajectory
    )


def finish_evaluation_scan(header: EvaluationHeader, accumulator: EvaluationScanAccumulator):
    if accumulator.count != len(header.identity.episodes):
        raise ValueError("incomplete evaluation denominator")
    if accumulator.stream.hexdigest() != header.hashes["rows.jsonl"]:
        raise ValueError("evaluation row stream closure mismatch")
    if set(header.crashes) != accumulator.failures:
        raise ValueError("crash denominator mismatch")
    if (
        header.metadata_paths & (accumulator.owned | accumulator.shared)
        or header.metadata_paths | accumulator.owned | accumulator.shared != header.hashes.keys()
    ):
        raise ValueError("unclassified or duplicate evaluation evidence")
    metrics = accumulator.metrics()
    if (
        metrics != header.metrics
        or metrics.gate_passed != header.done["gate_passed"]
        or metrics.validation_valid != header.done["validation_valid"]
    ):
        raise ValueError("evaluation metrics differ from raw rows")


def load_evaluation(root: Path, *, evidence_context=None):
    """Compatibility API: deliberately materializes rows, using the common scanner."""
    from silent_cascade.archive.readers import scan_evaluation

    rows = []
    header = scan_evaluation(
        root, evidence_context=evidence_context, consume=lambda item: rows.append(item.row)
    )
    return header.identity, tuple(rows), header.metrics, header.hashes


def evaluation_directories(run_dir: Path, *, evidence_context=None) -> tuple[Path, ...]:
    """Notice partial or damaged corpora too; never silently omit missing evidence."""
    if run_dir.is_symlink():
        raise ValueError("symbolic pilot artifact path")
    if evidence_context is not None:
        from silent_cascade.archive.readers import logical_root

        if logical_root(run_dir, evidence_context) != ".":
            raise ValueError("evaluation discovery requires the session run root")
        roots = {run_dir / name for name in evidence_context.evaluation_roots()}
        if not roots:
            raise ValueError("missing pilot evaluation artifacts")
        return tuple(sorted(roots))
    roots = set()
    for path in run_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError("symbolic pilot artifact path")
        if path.name in {"identity.json", "DONE", "rows.jsonl", "metrics.json"} and (
            path.parent.name == "autonomous" or "eval" in path.relative_to(run_dir).parts
        ):
            roots.add(path.parent)
    if not roots:
        raise ValueError("missing pilot evaluation artifacts")
    return tuple(sorted(roots))


def training_evaluation_status(run_dir: Path, training, *, evidence_context=None):
    """Authenticate committed roots and abandoned attempts from retained journals."""
    hashes = training["artifact_hashes"]

    def bound(name):
        if name not in hashes:
            raise ValueError("unbound recovery artifact")
        if evidence_context is None:
            raw = verify_hashes(run_dir, {name: hashes[name]}, retain={name})[name]
        else:
            from silent_cascade.archive.readers import evidence_path

            with evidence_path(run_dir, name, evidence_context=evidence_context) as path:
                raw = read_evaluation_artifact(path)
            if sha256_bytes(raw) != hashes[name]:
                raise ValueError("recovery artifact integrity mismatch")
        return _decode_json(raw)

    def journal(name):
        match = re.fullmatch(r"journal-([0-9a-f]{64})\.json", name)
        if match is None or hashes.get(name) != match[1]:
            raise ValueError("recovery journal hash mismatch")
        record = bound(name)
        if (
            record.get("kind") not in {"update", "validation"}
            or not re.fullmatch(r"attempt-[0-9a-f]{32}", record.get("attempt", ""))
            or type(record.get("global_step")) is not int
            or record["global_step"] <= 0
            or (
                record.get("prior") is not None
                and not re.fullmatch(r"[0-9a-f]{64}", record["prior"])
            )
        ):
            raise ValueError("invalid recovery journal")
        return record

    committed, required, steps = set(), set(), []
    if evidence_context is None:

        def committed_records():
            cursor = training["progress"]["journal_sha256"]
            while cursor is not None:
                name = f"journal-{cursor}.json"
                record = journal(name)
                yield name, record
                cursor = record["prior"]

    else:
        from silent_cascade.archive.readers import iter_journal_records

        def committed_records():
            cursor = training["progress"]["journal_sha256"]
            for record in iter_journal_records(run_dir, cursor, evidence_context=evidence_context):
                name = f"journal-{cursor}.json"
                if hashes.get(name) != cursor:
                    raise ValueError("unbound recovery artifact")
                yield name, record
                cursor = record["prior"]

    for name, record in committed_records():
        if name in committed:
            raise ValueError("cyclic recovery journal")
        committed.add(name)
        if record["kind"] == "update":
            steps.append(record["global_step"])
        else:
            artifacts = record.get("artifacts")
            if not isinstance(artifacts, dict) or not artifacts:
                raise ValueError("missing committed validation artifacts")
            for path, digest in artifacts.items():
                parts = child(run_dir, path).relative_to(run_dir).parts
                if parts[0] != record["attempt"] or hashes.get(path) != digest:
                    raise ValueError("committed validation artifact mismatch")
                if len(parts) >= 4 and parts[2] == "autonomous":
                    required.add(run_dir.joinpath(*parts[:3]))
    if steps != list(range(training["progress"]["global_step"], 0, -1)):
        raise ValueError("missing or duplicate committed recovery updates")
    abandoned = set()
    if evidence_context is None:
        restart_names = tuple(path.name for path in sorted(run_dir.glob("restart-*.json")))
    else:
        restart_names = tuple(
            sorted(
                entry.path
                for entry in evidence_context.entries()
                if "/" not in entry.path
                and entry.path.startswith("restart-")
                and entry.path.endswith(".json")
            )
        )
    for name in restart_names:
        restart = bound(name)
        if (
            set(restart) != {"last_durable_step", "uncommitted_journal_tail"}
            or type(restart["last_durable_step"]) is not int
            or not 0 <= restart["last_durable_step"] <= training["progress"]["global_step"]
            or not isinstance(restart["uncommitted_journal_tail"], list)
        ):
            raise ValueError("invalid restart artifact")
        tail = restart["uncommitted_journal_tail"]
        if any(not isinstance(name, str) for name in tail) or len(set(tail)) != len(tail):
            raise ValueError("invalid restart journal inventory")
        records = {}
        for name in tail:
            if name in committed:
                raise ValueError("restart labels committed journal abandoned")
            record = journal(name)
            records[name] = record
            prior = record["prior"]
            if prior is not None and f"journal-{prior}.json" not in committed | set(tail):
                raise ValueError("missing abandoned journal predecessor")
            abandoned.add(record["attempt"])
        if evidence_context is not None and tail:
            referenced = {
                f"journal-{record['prior']}.json"
                for record in records.values()
                if record["prior"] is not None and f"journal-{record['prior']}.json" in records
            }
            heads = set(tail) - referenced
            authenticated = set()
            for head in sorted(heads):
                cursor = head.removeprefix("journal-").removesuffix(".json")
                for record in iter_journal_records(
                    run_dir, cursor, evidence_context=evidence_context
                ):
                    current = f"journal-{cursor}.json"
                    if hashes.get(current) != cursor:
                        raise ValueError("unbound recovery artifact")
                    authenticated.add(current)
                    cursor = record["prior"]
            if not set(tail) <= authenticated:
                raise ValueError("abandoned journal authentication is incomplete")
    return required, abandoned


def load_abandoned_evaluation(
    root: Path, *, run_dir: Path, training, abandoned, evidence_context=None
):
    """Describe retained partial evidence, never infer outcomes for unwritten rows."""
    relative = root.relative_to(run_dir)
    if (
        len(relative.parts) != 3
        or relative.parts[0] not in abandoned
        or not relative.parts[1].startswith("validation-")
        or relative.parts[2] != "autonomous"
    ):
        raise ValueError("incomplete evaluation is not authenticated abandoned evidence")
    retain = {"identity.json", "rows.jsonl", ".rows.pending.jsonl"}
    if evidence_context is None:
        hashes = {}
        for path in root.rglob("*"):
            if path.is_file():
                name = str(path.relative_to(run_dir))
                if name not in training["artifact_hashes"]:
                    raise ValueError("unbound abandoned evaluation artifact")
                hashes[str(path.relative_to(root))] = training["artifact_hashes"][name]
        raw = verify_hashes(root, hashes, retain=retain)
    else:
        from silent_cascade.archive.readers import evidence_path, logical_root

        name = logical_root(root, evidence_context)
        prefix = name + "/"
        archived = {
            entry.path.removeprefix(prefix): entry.sha256
            for entry in evidence_context.entries()
            if entry.path.startswith(prefix)
        }
        expected = {
            path.removeprefix(prefix): digest
            for path, digest in training["artifact_hashes"].items()
            if path.startswith(prefix)
        }
        if archived != expected:
            raise ValueError("unbound abandoned evaluation artifact")
        hashes, raw = archived, {}
        for relative_name, digest in hashes.items():
            with evidence_path(
                run_dir, prefix + relative_name, evidence_context=evidence_context
            ) as path:
                payload = read_evaluation_artifact(path)
            if sha256_bytes(payload) != digest:
                raise ValueError("artifact integrity mismatch")
            if relative_name in retain or (
                relative_name.startswith("crashes/") and relative_name.endswith(".json")
            ):
                raw[relative_name] = payload
    if "identity.json" not in raw:
        raise ValueError("missing abandoned evaluation identity")
    identity = EvaluationIdentity.model_validate_json(raw["identity.json"])
    if (
        "rows.jsonl" in raw
        and ".rows.pending.jsonl" in raw
        and raw["rows.jsonl"] != raw[".rows.pending.jsonl"]
    ):
        raise ValueError("conflicting abandoned row publications")
    rows_name = "rows.jsonl" if "rows.jsonl" in raw else ".rows.pending.jsonl"
    lines = raw.get(rows_name, b"").splitlines(keepends=True)
    trailing = 0
    if lines and not lines[-1].endswith(b"\n"):
        trailing = len(lines.pop())
    if len(lines) > len(identity.episodes):
        raise ValueError("extra abandoned evaluation rows")
    if evidence_context is not None and lines:
        raise ValueError("cold complete abandoned rows require semantic episode verification")
    errors, completed = set(), set()
    for line, binding in zip(lines, identity.episodes, strict=False):
        row = TimedEpisodeRow.model_validate_json(line)
        _verify_row_identity(row, binding, identity)
        for reference in (row.neural_trace_ref, row.full_trace_ref):
            if reference is not None and reference not in hashes:
                raise ValueError("unbound abandoned runtime artifact")
        _verify_evidence(
            root,
            row,
            retain=(
                row.public_id in identity.retained_public_ids()
                or row.public_id in identity.report_example_public_ids
                or identity.purpose == "delay_swap"
            ),
        )
        (errors if row.error is not None else completed).add(row.public_id)
    crashes = {}
    episode_ids = {binding.public_id for binding in identity.episodes}
    for name in hashes:
        if (
            not name.startswith("crashes/")
            or not name.endswith(".json")
            or name == "crashes/index.json"
        ):
            continue
        manifest = CrashBundleManifest.model_validate_json(
            raw[name] if evidence_context is not None else read_evaluation_artifact(root / name)
        )
        public_id = manifest.context.episode_public_id
        if (
            public_id not in episode_ids
            or public_id in crashes
            or public_id in completed
            or manifest.context.source_revision != identity.execution_source_revision
        ):
            raise ValueError("inconsistent abandoned crash evidence")
        reference = manifest.context.checkpoint_ref
        if reference is not None:
            checkpoint = child(Path("crashes"), reference).as_posix()
            if checkpoint not in hashes:
                raise ValueError("missing abandoned crash evidence checkpoint")
        crashes[public_id] = {"path": name, "sha256": hashes[name]}
    if not errors <= crashes.keys():
        raise ValueError("missing abandoned crash evidence")
    if "crashes/index.json" in hashes:
        crash_index = (
            _decode_json(raw["crashes/index.json"])
            if evidence_context is not None
            else read_json(root / "crashes/index.json")
        )
        if crash_index != crashes:
            raise ValueError("abandoned crash evidence index mismatch")
    return dict(
        corpus=str(relative),
        status="abandoned_incomplete",
        identity_sha256=identity.sha256,
        planned_episodes=len(identity.episodes),
        retained_rows=len(lines),
        retained_errors=len(errors),
        unknown_episodes=len(identity.episodes) - len(lines),
        unparsed_trailing_bytes=trailing,
        raw_rows=str(relative / rows_name) if rows_name in raw else None,
        artifact_hashes=hashes,
    )


def load_training_result(run_dir: Path, path: Path, *, evidence_context=None):
    """Verify the complete trainer inventory without importing its executor."""
    from silent_cascade.archive.readers import evidence_path
    from silent_cascade.train.pilot_artifact_index import expand_training_result

    name = path.absolute().relative_to(run_dir.absolute()).as_posix()
    with evidence_path(run_dir, name, evidence_context=evidence_context) as local:
        payload = read_json(local)
    result = expand_training_result(run_dir, payload, evidence_context=evidence_context)
    validate_training_result(result)
    for name, digest in result["artifact_hashes"].items():
        with evidence_path(run_dir, name, evidence_context=evidence_context) as local:
            if sha256_bytes(read_evaluation_artifact(local)) != digest:
                raise ValueError(f"artifact integrity mismatch: {name}")
    for key in ("latest_weights", "selected_weights", "selected_checkpoint"):
        if result[key] is not None:
            descriptor = result[key]
            with evidence_path(
                run_dir, descriptor["path"], evidence_context=evidence_context
            ) as local:
                if sha256_bytes(read_evaluation_artifact(local)) != descriptor["sha256"]:
                    raise ValueError("training descriptor hash differs")
    return result


def validate_training_result(result):
    """Structural/selection checks shared with portable evidence authentication."""
    from silent_cascade.eventflow.neural import NeuralModelIdentity
    from silent_cascade.train.pilot_state import PilotCheckpointDescriptor, PilotProgress

    if set(result) != {
        "status",
        "progress",
        "selected_checkpoint",
        "selected_weights",
        "latest_weights",
        "artifact_hashes",
        "model_identity",
        "gate_eligible",
    }:
        raise ValueError("invalid training result artifact")
    progress = PilotProgress.model_validate_json(canonical_json_bytes(result["progress"]))
    if result["status"] != progress.status or progress.status == "running":
        raise ValueError("incomplete training result artifact")
    for key in ("latest_weights", "selected_weights", "selected_checkpoint"):
        if result[key] is not None:
            PilotCheckpointDescriptor.model_validate_json(canonical_json_bytes(result[key]))
    NeuralModelIdentity(**result["model_identity"])
    if (
        result["selected_weights"] != progress.model_dump(mode="json")["selected"]
        or result["latest_weights"]["global_step"] != progress.global_step
        or result["latest_weights"]["model_state_sha256"]
        != result["model_identity"]["model_state_sha256"]
    ):
        raise ValueError("training selection/weights disagree with progress")
    if result["gate_eligible"] != (result["status"] == "robustness_complete"):
        raise ValueError("training gate status mismatch")
    return result
