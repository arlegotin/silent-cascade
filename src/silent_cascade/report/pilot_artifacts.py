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
from silent_cascade.logging.crash_bundle import CrashBundleManifest


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
        _verify_row_identity(row, binding, identity)
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
    """Notice partial or damaged corpora too; never silently omit missing evidence."""
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


def training_evaluation_status(run_dir: Path, training):
    """Authenticate committed roots and abandoned attempts from retained journals."""
    hashes = training["artifact_hashes"]

    def bound(name):
        if name not in hashes:
            raise ValueError("unbound recovery artifact")
        return _decode_json(verify_hashes(run_dir, {name: hashes[name]}, retain={name})[name])

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

    cursor = training["progress"]["journal_sha256"]
    committed, required, steps = set(), set(), []
    while cursor is not None:
        name = f"journal-{cursor}.json"
        if name in committed:
            raise ValueError("cyclic recovery journal")
        committed.add(name)
        record = journal(name)
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
        cursor = record["prior"]
    if steps != list(range(training["progress"]["global_step"], 0, -1)):
        raise ValueError("missing or duplicate committed recovery updates")
    abandoned = set()
    for path in sorted(run_dir.glob("restart-*.json")):
        restart = bound(path.name)
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
        for name in tail:
            if name in committed:
                raise ValueError("restart labels committed journal abandoned")
            record = journal(name)
            prior = record["prior"]
            if prior is not None and f"journal-{prior}.json" not in committed | set(tail):
                raise ValueError("missing abandoned journal predecessor")
            abandoned.add(record["attempt"])
    return required, abandoned


def load_abandoned_evaluation(root: Path, *, run_dir: Path, training, abandoned):
    """Describe retained partial evidence, never infer outcomes for unwritten rows."""
    relative = root.relative_to(run_dir)
    if (
        len(relative.parts) != 3
        or relative.parts[0] not in abandoned
        or not relative.parts[1].startswith("validation-")
        or relative.parts[2] != "autonomous"
    ):
        raise ValueError("incomplete evaluation is not authenticated abandoned evidence")
    hashes = {}
    for path in root.rglob("*"):
        if path.is_file():
            name = str(path.relative_to(run_dir))
            if name not in training["artifact_hashes"]:
                raise ValueError("unbound abandoned evaluation artifact")
            hashes[str(path.relative_to(root))] = training["artifact_hashes"][name]
    raw = verify_hashes(root, hashes, retain={"identity.json", "rows.jsonl", ".rows.pending.jsonl"})
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
        manifest = CrashBundleManifest.model_validate_json(read_evaluation_artifact(root / name))
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
    if "crashes/index.json" in hashes and read_json(root / "crashes/index.json") != crashes:
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


def load_training_result(run_dir: Path, path: Path):
    """Verify the complete trainer inventory without importing its executor."""
    from silent_cascade.train.pilot_artifact_index import expand_training_result

    result = expand_training_result(run_dir, read_json(path))
    validate_training_result(result)
    verify_hashes(run_dir, result["artifact_hashes"])
    for key in ("latest_weights", "selected_weights", "selected_checkpoint"):
        if result[key] is not None:
            descriptor = result[key]
            verify_hashes(run_dir, {descriptor["path"]: descriptor["sha256"]})
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
