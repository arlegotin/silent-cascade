"""Independently verify Phase 2 historical evidence without executing episodes.

This checks raw witness arithmetic, framing, replay commitments, historical
regular Git blobs, and the immutable Phase 1 manifest. It trusts committed
episode witness observations rather than rerunning 10,000 episodes. Coherent
replacement of evidence and all Git trust anchors is outside its threat model.
"""

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Literal

import yaml

from silent_cascade.errors import ArtifactIntegrityError, SilentCascadeError
from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at
from silent_cascade.eventflow.config import Phase2Config
from silent_cascade.eventflow.evidence import Phase2EngineGateDebugReport, Phase2EngineGateReport
from silent_cascade.eventflow.provenance import PHASE2_ENGINE_SOURCE_PATHS
from silent_cascade.logging.manifest import ManifestEnvelope
from silent_cascade.validation import StrictModel

_GATE_LIMIT = 32 * 1024 * 1024
_MANIFEST_LIMIT = 16 * 1024 * 1024
_BLOB_LIMIT = 16 * 1024 * 1024
_PLAN = "docs/superpowers/plans/2026-09-09-phase-2-flow-event-engine.md"
_SPEC = "docs/superpowers/specs/2026-08-30-silent-cascade-design.md"
_VALIDATION = "manifests/validation/v1/ofd-primary-10000.json"
_VALIDATION_HASH = "84926a3217b27b04d7ed3e41038447533636aea06bcdc85b787f148f3d737e07"
_CONFIG_HASH = "29e4afac2118bdac5a7a01896167805f1fd1f7379a13e5b21566fef2c34fd546"
_VARIANTS = ("positive", "safe_negative", "disconnected_negative")


class Phase2GateVerificationResult(StrictModel):
    schema_version: Literal["phase2-gate-verification-v1"] = "phase2-gate-verification-v1"
    profile: Literal["production", "debug"]
    passed: bool
    artifact_file_sha256: str
    source_commit: str
    completed_episode_count: int
    foundation_model_calls: Literal[0] = 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactIntegrityError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path, limit: int) -> bytes:
    try:
        with archive_parent(path, error_factory=ArtifactIntegrityError) as (parent, name):
            return read_archive_at(
                parent, name, max_bytes=limit, error_factory=ArtifactIntegrityError
            )
    except OSError as error:
        raise ArtifactIntegrityError("evidence input must be a bounded regular file") from error


def _git(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments], cwd=root, check=True, capture_output=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ArtifactIntegrityError("historical Git authentication failed") from error


def _commit(root: Path, revision: str) -> None:
    _require(
        type(revision) is str and re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
        "revision must be a full lowercase commit ID",
    )
    _require(
        _git(root, "rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip() == revision,
        "revision must resolve exactly",
    )


def _blob(root: Path, revision: str, path: str) -> bytes:
    _require(
        bool(path) and not Path(path).is_absolute() and ".." not in Path(path).parts,
        "historical path must be repository relative",
    )
    entry = _git(root, "ls-tree", "-z", revision, "--", path)
    try:
        header, found = entry.rstrip(b"\0").split(b"\t")
        mode, kind, object_id = header.split()
    except ValueError as error:
        raise ArtifactIntegrityError("missing historical regular blob") from error
    _require(
        found == path.encode() and mode in {b"100644", b"100755"} and kind == b"blob",
        "historical input must be a regular Git blob",
    )
    _require(
        int(_git(root, "cat-file", "-s", object_id.decode())) <= _BLOB_LIMIT,
        "historical blob exceeds byte limit",
    )
    return _git(root, "cat-file", "blob", object_id.decode())


class _UniqueYamlLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        pairs = self.construct_pairs(node, deep=deep)
        mapping = {}
        for key, value in pairs:
            if key in mapping:
                raise ValueError("duplicate historical YAML key")
            mapping[key] = value
        return mapping


def _historical_config(root: Path, source: str) -> bytes:
    """Resolve only the three authenticated protocol blobs, independently."""

    def merge(base, overlay):
        result = dict(base)
        for key, value in overlay.items():
            result[key] = (
                merge(result[key], value)
                if isinstance(result.get(key), dict) and isinstance(value, dict)
                else value
            )
        return result

    try:
        values = {}
        for path in (
            "configs/base.yaml",
            "configs/data/primary.yaml",
            "configs/model/event_flow.yaml",
        ):
            overlay = yaml.load(_blob(root, source, path), Loader=_UniqueYamlLoader)
            _require(isinstance(overlay, dict), "historical YAML must be a mapping")
            values = merge(values, overlay)
        raw = _canonical(Phase2Config.model_validate(values).model_dump(mode="json"))
        _require(
            _hash(raw) == _CONFIG_HASH, "historical YAML differs from fixed gate configuration"
        )
        return raw
    except (ValueError, TypeError, yaml.YAMLError) as error:
        raise ArtifactIntegrityError("invalid historical Phase 2 configuration") from error


def _trust(
    root: Path,
    payload: dict,
    manifest_path: Path,
    manifest_raw: bytes,
    expected_source: str | None,
    expected_plan: str | None,
) -> None:
    provenance = payload["provenance"]
    source = provenance["source_commit"]
    plan = provenance["phase2_plan_base_revision"]
    _commit(root, source)
    _commit(root, plan)
    _require(expected_source is None or source == expected_source, "unexpected source commit")
    _require(expected_plan is None or plan == expected_plan, "unexpected plan base revision")
    _require(provenance["source_dirty"] is False, "dirty source provenance")
    _git(root, "merge-base", "--is-ancestor", plan, source)
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    _git(root, "merge-base", "--is-ancestor", source, head)
    _require(
        _git(root, "log", "-1", "--format=%H", source, "--", _PLAN).decode().strip() == plan,
        "plan base must be latest approved plan commit at source",
    )
    plan_raw = _blob(root, source, _PLAN)
    _require(
        b"User-approved; completion tracked in docs/PLAN.md" in plan_raw,
        "plan header is not durably approved",
    )
    _require(
        plan_raw == _blob(root, plan, _PLAN) == _blob(root, head, _PLAN),
        "plan changed since evidence source",
    )
    _require(_hash(plan_raw) == provenance["approved_plan_sha256"], "approved plan hash mismatch")
    spec_raw = _blob(root, source, _SPEC)
    _require(spec_raw == _blob(root, head, _SPEC), "specification changed since source")
    _require(_hash(spec_raw) == provenance["specification_sha256"], "specification hash mismatch")
    paths = tuple(provenance["source_paths"])
    _require(
        paths == PHASE2_ENGINE_SOURCE_PATHS and tuple(sorted(set(paths))) == paths,
        "source closure must be canonical and exact",
    )
    digest = hashlib.sha256(b"silent-cascade/phase2/source-tree/v1\0")
    digest.update(len(paths).to_bytes(4, "big"))
    for path in paths:
        raw = _blob(root, source, path)
        _require(raw == _blob(root, head, path), "source changed since evidence collection")
        encoded = path.encode()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    _require(
        digest.hexdigest() == provenance["source_tree_sha256"], "historical source hash mismatch"
    )
    try:
        relative = manifest_path.absolute().relative_to(root.absolute()).as_posix()
    except ValueError as error:
        raise ArtifactIntegrityError("manifest must be inside historical repository") from error
    _require(
        _blob(root, source, relative) == manifest_raw == _blob(root, head, relative),
        "manifest differs from historical regular blob",
    )
    if payload["profile"] == "production":
        _require(
            relative == _VALIDATION and _hash(manifest_raw) == _VALIDATION_HASH,
            "production requires the immutable Phase 1 validation artifact",
        )
    config_raw = provenance["config_canonical_json"].encode()
    _require(
        config_raw == _historical_config(root, source), "historical YAML configuration mismatch"
    )
    _require(
        _hash(config_raw)
        == _CONFIG_HASH
        == provenance["config_sha256"]
        == payload["config_sha256"],
        "full Phase 2 configuration hash mismatch",
    )


def _independent_witnesses(payload: dict, manifest: dict) -> bool:
    """Deliberately independent of producer summary, pass, and framing helpers."""
    rows = payload["episode_witnesses"]
    requested = payload["requested_episode_count"]
    _require(len(rows) == requested == manifest["episode_count"], "raw denominator mismatch")
    _require(
        payload["profile"] != "production" or requested == 10_000, "wrong production denominator"
    )
    counts = Counter()
    gaps = []
    maximum = 0
    seen = {name: set() for name in ("episode_public_id", "episode_sha256", "trace_sha256")}
    chain = hashlib.sha256(b"silent-cascade/phase2/episode-chain/v1\0")
    by_id = {}
    for index, (row, entry) in enumerate(zip(rows, manifest["entries"], strict=True)):
        _require(
            row["entry_index"] == index and row["coordinate"] == entry["coordinate"],
            "witness order/coordinate mismatch",
        )
        _require(
            row["manifest_entry_sha256"] == _hash(_canonical(entry)), "entry commitment mismatch"
        )
        for name in ("episode_public_id", "episode_sha256"):
            _require(row[name] == entry[name], "manifest episode identity mismatch")
        _require(row["config_sha256"] == payload["config_sha256"], "witness configuration mismatch")
        for name, values in seen.items():
            _require(row[name] not in values, "duplicate episode identity or trace hash")
            values.add(row[name])
        raw = _canonical(row)
        chain.update(index.to_bytes(8, "big"))
        chain.update(len(raw).to_bytes(8, "big"))
        chain.update(raw)
        variant = row["variant"]
        counts[variant + "_count"] += 1
        score = row["score"]
        counts["completed_episode_count"] += row["completed"]
        if score is not None:
            positive = variant == "positive"
            success = (
                (
                    score["action_count"] == 1
                    and score["correct_class"] is True
                    and score["in_window"] is True
                )
                if positive
                else score["action_count"] == 0
            )
            false_action = not positive and score["action_count"] > 0
            _require(
                score["is_positive"] == positive
                and score["timed_success"] == success
                and score["false_action"] == false_action,
                "primitive score arithmetic mismatch",
            )
            counts["timed_success_count"] += success
            counts["false_action_count"] += false_action
        for primitive, aggregate in (
            ("dynamics_failure", "dynamics_failure_count"),
            ("replay_failure", "replay_failure_count"),
            ("provenance_failure", "provenance_failure_count"),
            ("event_cap_failure", "event_cap_failure_count"),
            ("gap_clamp_count", "gap_clamp_count"),
            ("time_reversal_count", "time_reversal_count"),
            ("foundation_model_calls", "foundation_model_calls"),
            ("internal_event_count", "internal_event_count"),
        ):
            counts[aggregate] += row[primitive]
        maximum = max(maximum, row["internal_event_count"])
        if row["minimum_internal_gap"] is not None:
            gaps.append(row["minimum_internal_gap"])
        by_id[row["episode_public_id"]] = row
    _require(
        all(payload[name] == count for name, count in counts.items()), "aggregate totals mismatch"
    )
    _require(
        payload["maximum_episode_internal_events"] == maximum
        and payload["minimum_internal_gap"] == min(gaps, default=None),
        "gap/event extrema mismatch",
    )
    _require(
        payload["trace_chain_sha256"] == chain.hexdigest(), "ordered framed trace chain mismatch"
    )
    runtime_json = _canonical(
        json.loads(payload["provenance"]["config_canonical_json"])["event_flow"]
    )
    for variant, sample, listed in zip(
        _VARIANTS,
        payload["selected_replay_samples"],
        payload["selected_replay_trace_hashes"],
        strict=True,
    ):
        row = by_id.get(sample["expected_result"]["public_id"])
        _require(row is not None and row["variant"] == variant, "missing variant replay witness")
        _require(
            _hash(_canonical(sample["episode"])) == row["episode_sha256"],
            "sample episode hash mismatch",
        )
        _require(
            sample["episode"]["truth"]["recipe"]["variant"] == variant, "sample variant mismatch"
        )
        _require(
            sample["config_canonical_json"].encode() == runtime_json
            and sample["config_sha256"] == _hash(runtime_json),
            "nested runtime config scope mismatch",
        )
        _require(
            sample["payload_sha256"]
            == _hash(_canonical({k: v for k, v in sample.items() if k != "payload_sha256"})),
            "replay envelope hash mismatch",
        )
        trace = sample["trace"]
        events = trace["events"]
        online = {
            "schema_version": trace["schema_version"],
            "events": [{k: v for k, v in event.items() if k != "source"} for event in events],
        }
        _require(
            _hash(_canonical(online)) == trace["sha256"] == listed == row["trace_sha256"],
            "sample trace commitment mismatch",
        )
        _require(events and events[-1]["kind"] == "terminal", "sample lacks scored terminal")
        _require(
            sample["expected_result"]["score"] == row["score"], "sample score witness mismatch"
        )
        internal = [event for event in events if event["source"] == "internal"]
        _require(
            len(internal) == row["internal_event_count"]
            and min((e["delta"] for e in internal), default=None) == row["minimum_internal_gap"]
            and sum(e["was_gap_clamped"] for e in events) == row["gap_clamp_count"],
            "sample dynamics witness mismatch",
        )
        _require(
            sum(b["timestamp"] < a["timestamp"] for a, b in pairwise(events))
            == row["time_reversal_count"],
            "sample time order mismatch",
        )
        counters = sample["expected_result"]["counters"]
        _require(
            counters["foundation_model_calls"] == row["foundation_model_calls"],
            "sample call count mismatch",
        )
    passed = (
        counts["completed_episode_count"] == counts["timed_success_count"] == requested
        and tuple(counts[v + "_count"] for v in _VARIANTS)
        == (requested // 2, requested // 4, requested // 4)
        and not any(
            counts[name]
            for name in (
                "false_action_count",
                "dynamics_failure_count",
                "replay_failure_count",
                "provenance_failure_count",
                "event_cap_failure_count",
                "gap_clamp_count",
                "time_reversal_count",
                "foundation_model_calls",
            )
        )
        and maximum <= 64
        and bool(gaps)
        and min(gaps) >= 1e-4
        and payload["provenance"]["source_dirty"] is False
    )
    _require(
        payload["passed"] == passed, "supplied pass flag differs from independently derived gate"
    )
    return passed


def verify_phase2_gate_artifact(
    *,
    artifact_path: Path,
    repo_root: Path,
    manifest_path: Path | None = None,
    expected_source_commit: str | None = None,
    expected_plan_base_revision: str | None = None,
    allow_debug: bool = False,
) -> Phase2GateVerificationResult:
    try:
        raw = _read(artifact_path, _GATE_LIMIT)
        payload = json.loads(raw)
        _require(isinstance(payload, dict), "gate must be an object")
        schema = payload.get("schema_version")
        if schema == "phase2-engine-gate-v1":
            model = Phase2EngineGateReport
        elif allow_debug and schema == "phase2-engine-gate-debug-v1":
            model = Phase2EngineGateDebugReport
        else:
            raise ArtifactIntegrityError("closed production gate schema required")
        parsed = model.model_validate_json(raw)
        _require(raw == _canonical(parsed.model_dump(mode="json")), "gate must be canonical JSON")
        manifest_path = manifest_path or repo_root / _VALIDATION
        manifest_raw = _read(manifest_path, _MANIFEST_LIMIT)
        manifest = json.loads(manifest_raw)
        envelope = ManifestEnvelope.model_validate_json(manifest_raw)
        _require(
            manifest_raw == _canonical(envelope.model_dump(mode="json")),
            "manifest must be canonical JSON",
        )
        manifest_hash = _hash(_canonical(manifest["payload"]))
        _require(
            manifest_hash
            == manifest["payload_sha256"]
            == payload["validation_manifest_payload_sha256"]
            == payload["provenance"]["validation_manifest_payload_sha256"],
            "manifest payload hash mismatch",
        )
        _require(
            _hash(manifest_raw)
            == payload["validation_manifest_file_sha256"]
            == payload["provenance"]["validation_manifest_file_sha256"],
            "manifest file hash mismatch",
        )
        access = "debug" if payload["profile"] == "debug" else "validation"
        _require(manifest["payload"]["access_class"] == access, "manifest access/profile mismatch")
        config = json.loads(payload["provenance"]["config_canonical_json"])
        del config["event_flow"]
        _require(
            _hash(_canonical(config)) == manifest["payload"]["provenance"]["config_sha256"],
            "manifest generation config mismatch",
        )
        _trust(
            repo_root,
            payload,
            manifest_path,
            manifest_raw,
            expected_source_commit,
            expected_plan_base_revision,
        )
        passed = _independent_witnesses(payload, manifest["payload"])
        return Phase2GateVerificationResult(
            profile=payload["profile"],
            passed=passed,
            artifact_file_sha256=_hash(raw),
            source_commit=payload["provenance"]["source_commit"],
            completed_episode_count=payload["completed_episode_count"],
        )
    except SilentCascadeError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise ArtifactIntegrityError("invalid Phase 2 evidence schema or witness") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-plan-base-revision")
    parser.add_argument(
        "--allow-debug", action="store_true", help="explicitly verify a separate debug profile"
    )
    args = parser.parse_args()
    try:
        result = verify_phase2_gate_artifact(
            artifact_path=args.artifact,
            repo_root=args.repo_root,
            manifest_path=args.manifest,
            expected_source_commit=args.expected_source_commit,
            expected_plan_base_revision=args.expected_plan_base_revision,
            allow_debug=args.allow_debug,
        )
    except SilentCascadeError:
        print("Phase 2 artifact verification failed")
        return 1
    print(_canonical(result.model_dump(mode="json")).decode())
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
