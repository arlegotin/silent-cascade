"""Read-only verification of pilot evidence; never executes training or evaluation."""

import json
import re
from pathlib import Path

from silent_cascade.eval.artifacts import (
    EvaluationIdentity,
    _verify_evidence,
    read_evaluation_artifact,
)
from silent_cascade.eval.metrics import PilotMetrics, TimedEpisodeRow, summarize_timed_rows
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes


def child(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("unsafe artifact path")
    if any(part in {"..", ".", "frozen", "frozen_test"} for part in name.split("/")):
        raise ValueError("unsafe artifact path")
    return root / name


def read_json(path: Path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate artifact key")
            value[key] = item
        return value

    return json.loads(
        read_evaluation_artifact(path),
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


def load_evaluation(root: Path):
    """Recompute every denominator and validate ordered rows against retained evidence."""
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
    verified = verify_hashes(
        root,
        done["artifact_hashes"],
        retain={"identity.json", "retention.json", "rows.jsonl", "index.json", "metrics.json"},
    )
    if (
        not {"identity.json", "retention.json", "rows.jsonl", "index.json", "metrics.json"}
        <= verified.keys()
    ):
        raise ValueError("missing evaluation artifact")
    identity = EvaluationIdentity.model_validate_json(verified["identity.json"])
    if identity.sha256 != done["identity_sha256"]:
        raise ValueError("evaluation identity hash mismatch")
    rows = tuple(
        TimedEpisodeRow.model_validate_json(line) for line in verified["rows.jsonl"].splitlines()
    )
    if len(rows) != len(identity.episodes):
        raise ValueError("incomplete evaluation denominator")
    indices, offset = [], 0
    for row, binding, line in zip(
        rows, identity.episodes, verified["rows.jsonl"].splitlines(keepends=True), strict=True
    ):
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
        for reference in (row.neural_trace_ref, row.full_trace_ref):
            if reference is not None and reference not in done["artifact_hashes"]:
                raise ValueError("unbound runtime artifact")
        _verify_evidence(
            root,
            row,
            retain=(
                row.public_id in identity.retained_public_ids()
                or row.public_id in identity.report_example_public_ids
                or identity.purpose == "delay_swap"
            ),
        )
        indices.append(
            dict(public_id=row.public_id, offset=offset, bytes=len(line), sha256=sha256_bytes(line))
        )
        offset += len(line)
    if read_json(root / "index.json") != {"rows": indices}:
        raise ValueError("evaluation index mismatch")
    if any(r.error is not None for r in rows):
        crashes = read_json(root / "crashes/index.json")
        if set(crashes) != {r.public_id for r in rows if r.error is not None}:
            raise ValueError("crash denominator mismatch")
        for entry in crashes.values():
            verify_hashes(root, {entry["path"]: entry["sha256"]})
    metrics = summarize_timed_rows(rows)
    if (
        metrics != PilotMetrics.model_validate_json(verified["metrics.json"])
        or metrics.gate_passed != done["gate_passed"]
        or metrics.validation_valid != done["validation_valid"]
    ):
        raise ValueError("evaluation metrics differ from raw rows")
    return identity, rows, metrics, done["artifact_hashes"]


def evaluation_directories(run_dir: Path) -> tuple[Path, ...]:
    """Also notice partial evaluations, whose missing DONE must fail reporting."""
    if run_dir.is_symlink():
        raise ValueError("symbolic pilot artifact path")
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


def load_training_result(run_dir: Path, path: Path):
    """Verify the complete trainer inventory without importing its executor."""
    from silent_cascade.eventflow.neural import NeuralModelIdentity
    from silent_cascade.train.pilot_state import PilotCheckpointDescriptor, PilotProgress

    result = read_json(path)
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
    verify_hashes(run_dir, result["artifact_hashes"])
    progress = PilotProgress.model_validate_json(canonical_json_bytes(result["progress"]))
    if result["status"] != progress.status or progress.status == "running":
        raise ValueError("incomplete training result artifact")
    for key in ("latest_weights", "selected_weights", "selected_checkpoint"):
        if result[key] is not None:
            descriptor = PilotCheckpointDescriptor.model_validate_json(
                canonical_json_bytes(result[key])
            )
            verify_hashes(run_dir, {descriptor.path: descriptor.sha256})
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
