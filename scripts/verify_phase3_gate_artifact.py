"""Independent artifact arithmetic and historical authentication; never runs a model."""

import argparse
import ast
import hashlib
import json
import math
import struct
import subprocess
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

from pydantic import ValidationError

from silent_cascade.errors import ArtifactIntegrityError, SilentCascadeError
from silent_cascade.train.evidence_types import (
    MAX_ARTIFACT_BYTES,
    MAX_METADATA_BYTES,
    ComponentGateReport,
    Phase3VerificationResult,
    WeightMetadataData,
    decode_json,
    read_bytes,
)

PLAN = "docs/superpowers/plans/2026-09-09-phase-3-neural-components.md"
SPEC = "docs/superpowers/specs/2026-08-30-silent-cascade-design.md"


def _require(condition, message):
    if not condition:
        raise ArtifactIntegrityError(message)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _git(root, *args):
    try:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ArtifactIntegrityError("historical Git authentication failed") from error


def _path(name):
    path = Path(name)
    _require(
        bool(path.parts)
        and not path.is_absolute()
        and path.as_posix() == name
        and not any(part in (".", "..") for part in path.parts)
        and "\\" not in name,
        "unsafe relative path",
    )
    return path


def _blob(root, commit, name):
    _path(name)
    entry = _git(root, "ls-tree", "-z", commit, "--", name)
    try:
        meta, actual = entry.rstrip(b"\0").split(b"\t")
        mode, kind, oid = meta.split()
    except ValueError as error:
        raise ArtifactIntegrityError("missing historical regular blob") from error
    _require(
        actual.decode() == name and mode in (b"100644", b"100755") and kind == b"blob",
        "historical input must be a regular Git blob",
    )
    _require(
        int(_git(root, "cat-file", "-s", oid.decode())) <= MAX_ARTIFACT_BYTES, "Git blob byte limit"
    )
    return _git(root, "cat-file", "blob", oid.decode())


def _trust(report, manifest_raw, root, expected):
    p = report["provenance"]
    source, plan, intro = p["source_commit"], p["plan_revision"], p["validation_manifest_commit"]
    if expected is not None:
        _require(source == expected, "unexpected source revision")
    for revision in (source, plan, intro, p["evidence_base_revision"]):
        _require(
            _git(root, "rev-parse", f"{revision}^{{commit}}").decode().strip() == revision,
            "revision is not an exact commit",
        )
    for left, right in (
        (plan, source),
        (source, intro),
        (intro, p["evidence_base_revision"]),
        (p["evidence_base_revision"], "HEAD"),
    ):
        _git(root, "merge-base", "--is-ancestor", left, right)
    _require(
        _git(root, "log", "-1", "--format=%H", source, "--", PLAN).decode().strip() == plan,
        "plan revision is not the latest effective approval",
    )
    source_module = _blob(root, source, "src/silent_cascade/train/provenance.py")
    tuples = [
        ast.literal_eval(node.value)
        for node in ast.parse(source_module).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PHASE3_SOURCE_PATHS" for t in node.targets)
    ]
    _require(len(tuples) == 1, "source closure lacks one literal declaration")
    paths = tuples[0]
    _require(
        tuple(p["source_paths"]) == paths == tuple(sorted(set(paths))),
        "source closure substitution",
    )
    digest = hashlib.sha256(b"silent-cascade/phase3/source-tree/v1\0")
    for name in paths:
        raw, encoded = _blob(root, source, name), name.encode()
        _require(
            raw == read_bytes(root / name, MAX_ARTIFACT_BYTES),
            f"current reviewed source changed: {name}",
        )
        digest.update(len(encoded).to_bytes(4, "big") + encoded)
        digest.update(len(raw).to_bytes(8, "big") + raw)
    _require(digest.hexdigest() == p["source_tree_sha256"], "source closure digest mismatch")
    for name, key in ((SPEC, "specification_sha256"), (PLAN, "approved_plan_sha256")):
        raw = _blob(root, source, name)
        _require(_sha(raw) == p[key], "specification or plan hash mismatch")
    _require(
        b"User-approved; completion tracked in docs/PLAN.md" in _blob(root, plan, PLAN),
        "unapproved plan",
    )
    _require(
        _sha(manifest_raw) == p["validation_manifest_sha256"]
        and manifest_raw == _blob(root, intro, p["validation_manifest_path"]),
        "manifest introduction identity mismatch",
    )
    _require(
        _git(
            root,
            "log",
            "-1",
            "--format=%H",
            p["evidence_base_revision"],
            "--",
            p["validation_manifest_path"],
        )
        .decode()
        .strip()
        == intro,
        "manifest introduction revision mismatch",
    )
    # Configuration classes and corpus recipes contain no learned computation.
    from silent_cascade.config import resolve_config
    from silent_cascade.train.config import Phase3Config

    production = report["publication"] == "production"
    paths = [
        "configs/base.yaml",
        "configs/data/primary.yaml",
        "configs/model/event_flow.yaml",
        "configs/model/neural_components.yaml",
        "configs/train/one_hop.yaml" if production else "configs/train/smoke.yaml",
    ]
    _require(p["config_paths"] == paths, "configuration layer order mismatch")
    resolved = resolve_config(Phase3Config, tuple(root / name for name in paths))
    _require(
        resolved.sha256 == p["config_sha256"]
        and resolved.canonical_json.decode() == p["config_canonical_json"],
        "historical canonical configuration mismatch",
    )
    return resolved


def _score(prediction, target):
    recall = sum(
        i < len(prediction["record_ids"]) and prediction["record_ids"][i] == value
        for i, value in enumerate(target["record_ids"])
    )
    composition, errors, deadlines = 0, [], []
    for i, wanted in enumerate(target["content"]):
        if i >= len(prediction["content"]):
            continue
        got = prediction["content"][i]
        final_disconnected = i == len(target["content"]) - 1 and target["terminal_status"] == 2
        continuation = (
            wanted["continue_search"] is None or got["continue_search"] == wanted["continue_search"]
        )
        if final_disconnected and got["role"] == 0:
            continuation = prediction["stop_reason"] in ("explicit_null", "dormant") and len(
                prediction["content"]
            ) == len(target["content"])
        status = (
            wanted["status"] is None
            or got["status"] == wanted["status"]
            or (final_disconnected and prediction["stop_reason"] == "dormant")
        )
        equal = (
            got["record_id"] == wanted["record_id"]
            and got["role"] == wanted["role"]
            and got["append_support"] == wanted["append_support"]
            and continuation
            and status
        )
        for name in ("focus", "hazard_type"):
            equal = equal and (wanted[name] is None or got[name] == wanted[name])
        composition += int(equal)
        if wanted["log_delay"] is not None:
            errors.append(abs(got["log_delay"] - wanted["log_delay"]))
        if wanted["normalized_deadline"] is not None:
            deadlines.append(abs(got["normalized_deadline"] - wanted["normalized_deadline"]))
    required_r, required_c = len(target["record_ids"]), len(target["content"])
    capped, invalid = (
        prediction["stop_reason"] == "capped",
        prediction["stop_reason"] == "malformed",
    )
    chain = (
        recall == required_r
        and composition == required_c
        and len(prediction["record_ids"]) == required_r
        and len(prediction["content"]) == required_c
        and not capped
        and not invalid
        and prediction["terminal_class"] == target["terminal_class"]
        and prediction["terminal_status"] == target["terminal_status"]
    )
    return dict(
        required_recall_count=required_r,
        correct_recall_count=recall,
        required_composition_count=required_c,
        correct_composition_count=composition,
        complete_chain_correct=chain,
        action_correct=prediction["action_class"] == target["terminal_class"],
        capped=capped,
        invalid_prediction=invalid,
        log_delay_absolute_errors=errors,
        normalized_deadline_absolute_errors=deadlines,
    )


def _above_99_percent(correct, required):
    if (
        type(correct) is not int
        or type(required) is not int
        or not 0 <= correct <= required
        or required == 0
    ):
        return False
    return 100 * correct > 99 * required


def _summary(rows):
    result = {
        key: sum(row[key] for row in rows)
        for key in (
            "required_recall_count",
            "correct_recall_count",
            "required_composition_count",
            "correct_composition_count",
            "complete_chain_correct",
        )
    }
    n = len(rows)
    result.update(
        episode_count=n,
        action_correct_count=sum(row["action_correct"] for row in rows),
        variant_counts={
            v: sum(row["variant"] == v for row in rows)
            for v in ("positive", "safe", "disconnected")
        },
        cap_count=sum(row["capped"] for row in rows),
        invalid_prediction_count=sum(row["invalid_prediction"] for row in rows),
    )
    result["required_recall_accuracy"] = (
        result["correct_recall_count"] / result["required_recall_count"]
        if result["required_recall_count"]
        else 0.0
    )
    result["required_composition_accuracy"] = (
        result["correct_composition_count"] / result["required_composition_count"]
        if result["required_composition_count"]
        else 0.0
    )
    result["complete_chain_accuracy"] = result["complete_chain_correct"] / n
    result["action_accuracy"] = result["action_correct_count"] / n
    result["content_gates_pass"] = all(
        _above_99_percent(c, r)
        for c, r in (
            (result["correct_recall_count"], result["required_recall_count"]),
            (result["correct_composition_count"], result["required_composition_count"]),
            (result["complete_chain_correct"], n),
        )
    )
    return result


def _numeric(report, config):
    from silent_cascade.train.curriculum_data import CurriculumKey, make_curriculum_example

    training = config.config.training

    def expected_batch(counter):
        return [
            make_curriculum_example(
                config.config,
                CurriculumKey(
                    training.curriculum_version,
                    "train",
                    training.train_root_seed,
                    training.train_public_id_seed,
                    counter * training.batch_size + offset,
                    "one_hop",
                ),
            ).example_hash
            for offset in range(training.batch_size)
        ]

    p, numeric = report["provenance"], report["numeric"]
    parity = numeric["parity"]
    for label in ("forward", "losses", "gradients", "updated_weights"):
        _comparison(parity[label], 1e-4 if label in ("forward", "losses") else 1e-3, 1e-5)
    _require(
        parity["recalled_record_mismatches"] == parity["action_class_mismatches"] == 0,
        "discrete device mismatch",
    )
    _require(
        min(parity[name] for name in ("active_crossings", "dormant_crossings", "empty_memory_rows"))
        > 0,
        "missing actual native parity coverage",
    )
    _require(
        parity["example_hashes"] == expected_batch(0),
        "parity batch identity mismatch",
    )
    continuation = expected_batch(1)
    for name, device, rtol, atol in (
        ("cpu_resume", "cpu", 0.0, 0.0),
        ("mps_resume", "mps", 1e-4, 1e-5),
    ):
        data = numeric[name]
        _require(data["example_hashes"] == continuation, "resume batch identity mismatch")
        _require(
            data["schema_version"] == f"phase3-{device}-resume-v1" and data["device"] == device,
            "wrong resume device/schema",
        )
        _require(
            data["config_sha256"] == p["config_sha256"]
            and data["source_commit"] == p["source_commit"],
            "resume source/config mismatch",
        )
        _require(
            data["rng_mismatches"] == data["batch_hash_mismatches"] == 0, "resume identity mismatch"
        )
        for kind in ("parameters", "optimizer", "losses"):
            _comparison(data[kind], rtol, atol)
    for data in (parity, numeric["cpu_resume"], numeric["mps_resume"]):
        _require(
            (data["python_version"], data["torch_version"], data["threads"])
            == (p["python_version"], p["torch_version"], p["threads"]),
            "numeric environment mismatch",
        )


def _comparison(data, rtol, atol):
    _require(data["rtol"] == rtol and data["atol"] == atol, "numeric tolerance changed")
    _require(
        data["compared"] > 0
        and data["mismatches"] == 0
        and not data["failed_names"]
        and bool(data["tested_names"])
        and len(set(data["tested_names"])) == len(data["tested_names"]),
        "numerical comparison failed or empty",
    )
    if rtol == atol == 0:
        _require(
            data["max_absolute_error"] == data["max_relative_error"] == 0, "CPU resume is not exact"
        )


def _training(report, config):
    p, data = report["provenance"], report["training"]
    _path(data["run_path"])
    _path(data["result_path"])
    _require(
        data["validation_manifest_sha256"] == p["validation_manifest_sha256"],
        "training manifest mismatch",
    )
    _require(
        0 < data["step_count"] == data["next_batch_counter"] <= config.config.training.max_steps,
        "training progress mismatch",
    )
    for name in ("latest", "selected", "weights"):
        descriptor = data[name]
        _path(descriptor["relative_path"])
        _require(
            descriptor["source_commit"] == p["source_commit"]
            and descriptor["config_sha256"] == p["config_sha256"],
            "checkpoint provenance mismatch",
        )
        prefix = "weights" if name == "weights" else "training"
        _require(
            descriptor["relative_path"] == f"{prefix}-{descriptor['file_sha256']}.safetensors",
            "checkpoint content-address mismatch",
        )
    _require(
        data["latest"]["optimizer_step"] == data["step_count"]
        and data["weights"]["optimizer_step"] == 0,
        "portable and latest progress mismatch",
    )
    _require(
        data["selected"]["model_state_sha256"] == data["weights"]["model_state_sha256"],
        "selected model mismatch",
    )
    history = data["validation_history"]
    _require(bool(history), "missing committed validation history")
    _require(
        [v["step"] for v in history]
        == list(
            range(
                config.config.training.validation_every_steps,
                data["step_count"] + 1,
                config.config.training.validation_every_steps,
            )
        ),
        "validation cadence mismatch",
    )
    best = max(history, key=lambda row: (row["chain"], row["composition"], -row["step"]))
    _require(
        (best["step"], best["chain"], best["composition"])
        == (
            data["selected"]["optimizer_step"],
            data["selected"]["validation_metric"],
            data["selected"]["validation_composition_metric"],
        ),
        "checkpoint selection ranking mismatch",
    )
    patience, ranking = 0, None
    for point in history:
        count = 10000 if report["publication"] == "production" else 16
        _require(point["episode_count"] == count, "validation episode denominator mismatch")
        _require(
            all(
                point[name] == report["summary"][name]
                for name in ("required_recall_count", "required_composition_count")
            ),
            "validation required denominator differs from fixed corpus",
        )
        triples = (
            ("recall", "correct_recall_count", "required_recall_count"),
            ("composition", "correct_composition_count", "required_composition_count"),
            ("chain", "complete_chain_correct", "episode_count"),
        )
        for ratio, correct, required in triples:
            _require(
                0 <= point[correct] <= point[required]
                and point[required] > 0
                and point[ratio] == point[correct] / point[required],
                "validation history arithmetic mismatch",
            )
        passing = all(
            _above_99_percent(point[correct], point[required]) for _, correct, required in triples
        )
        _require(point["gates_pass"] is passing, "validation history pass mismatch")
        current = (point["chain"], point["composition"], -point["step"])
        if ranking is None or current > ranking:
            ranking, patience = current, 0
        else:
            patience += 1
        if passing or patience >= 15:
            _require(point["step"] == data["step_count"], "training continued beyond stopping rule")
    _require(data["patience_counter"] == patience, "training patience counter mismatch")
    reason = (
        "component_gate"
        if history[-1]["gates_pass"] and history[-1]["step"] == data["step_count"]
        else "early_stopping"
        if patience >= 15
        else "step_ceiling"
    )
    _require(
        data["stop_reason"] == reason
        and (reason != "step_ceiling" or data["step_count"] == config.config.training.max_steps),
        "training stop rule mismatch",
    )


def _weights(path, report):
    raw = read_bytes(path, MAX_ARTIFACT_BYTES)
    descriptor = report["training"]["weights"]
    _require(_sha(raw) == descriptor["file_sha256"], "weight archive byte identity mismatch")
    _require(len(raw) >= 8, "truncated weights archive")
    size = struct.unpack("<Q", raw[:8])[0]
    _require(size <= MAX_METADATA_BYTES and size <= len(raw) - 8, "weight metadata bound")
    header = decode_json(raw[8 : 8 + size])
    meta = header.pop("__metadata__", None)
    _require(
        isinstance(meta, dict) and set(meta) == {"training"} and type(meta["training"]) is str,
        "weight metadata schema",
    )
    metadata = decode_json(meta["training"].encode())
    validated_metadata = WeightMetadataData.model_validate_json(meta["training"])
    _require(
        _canonical(validated_metadata.model_dump(mode="json")) == _canonical(metadata),
        "weight metadata primitive types differ",
    )
    _require(
        set(metadata)
        == {
            "schema_version",
            "config_json",
            "config_sha256",
            "source_commit",
            "model_state_sha256",
            "aliases",
            "training",
            "environment",
            "owner_directory",
            "progress",
            "groups",
            "optimizer_names",
            "rng",
        },
        "unknown weight metadata fields",
    )
    _require(
        metadata["schema_version"] == "phase3-model-weights-v1"
        and metadata["progress"] is None
        and metadata["rng"] is None
        and metadata["groups"] == []
        and metadata["optimizer_names"] == [],
        "weights contain training state",
    )
    for name in ("source_commit", "config_sha256", "model_state_sha256"):
        _require(metadata[name] == descriptor[name], f"weight {name} mismatch")
    _require(
        metadata["config_json"] == report["provenance"]["config_canonical_json"],
        "weight config mismatch",
    )
    intervals = []
    widths = {"F32": 4, "BOOL": 1, "I64": 8}
    dtype_names = {"F32": "torch.float32", "BOOL": "torch.bool", "I64": "torch.int64"}
    aliases = metadata["aliases"]
    _require(
        isinstance(aliases, dict)
        and all(type(value) is str and value in header for value in aliases.values()),
        "invalid model aliases",
    )
    model_digest = hashlib.sha256(_canonical(aliases))
    for name in sorted(header):
        tensor = header[name]
        _require(
            set(tensor) == {"dtype", "shape", "data_offsets"}
            and tensor["dtype"] in widths
            and name.startswith(("parameter/", "buffer/"))
            and (not name.startswith("parameter/") or tensor["dtype"] == "F32"),
            "invalid model tensor header",
        )
        _require(
            all(type(v) is int and v >= 0 for v in tensor["shape"] + tensor["data_offsets"])
            and len(tensor["data_offsets"]) == 2,
            "invalid tensor dimensions/offsets",
        )
        start, end = tensor["data_offsets"]
        _require(
            0 <= start <= end <= len(raw) - 8 - size
            and end - start == widths[tensor["dtype"]] * math.prod(tensor["shape"]),
            "tensor payload length mismatch",
        )
        intervals.append((start, end))
        content = raw[8 + size + start : 8 + size + end]
        if tensor["dtype"] == "F32":
            _require(
                all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", content)),
                "nonfinite weight value",
            )
        elif tensor["dtype"] == "BOOL":
            _require(all(value in (0, 1) for value in content), "invalid boolean model buffer")
        model_digest.update(
            _canonical(
                {"name": name, "dtype": dtype_names[tensor["dtype"]], "shape": tensor["shape"]}
            )
        )
        model_digest.update(content)
    intervals.sort()
    _require(
        bool(intervals)
        and intervals[0][0] == 0
        and intervals[-1][1] == len(raw) - 8 - size
        and all(a[1] == b[0] for a, b in pairwise(intervals)),
        "noncontiguous model payload",
    )
    _require(
        model_digest.hexdigest() == descriptor["model_state_sha256"],
        "weight model-state digest mismatch",
    )


def verify_phase3_gate_artifact(
    *,
    artifact_path: Path,
    manifest_path: Path,
    repo_root: Path,
    weights_path: Path | None = None,
    expected_source_commit: str | None = None,
    allow_debug: bool = False,
) -> Phase3VerificationResult:
    try:
        artifact_raw = read_bytes(artifact_path, MAX_ARTIFACT_BYTES)
        raw = decode_json(artifact_raw, MAX_ARTIFACT_BYTES)
        typed = ComponentGateReport.model_validate_json(artifact_raw)
        _require(
            _canonical(typed.model_dump(mode="json")) == _canonical(raw),
            "noncanonical primitive types",
        )
        _require(raw["timed"] is False, "component evidence must remain untimed")
        _require(
            len(_canonical({k: v for k, v in raw.items() if k not in ("rows", "compute")}))
            <= MAX_METADATA_BYTES,
            "gate metadata byte limit",
        )
        _require(
            allow_debug or raw["publication"] == "production", "debug evidence is non-acceptance"
        )
        manifest_raw = read_bytes(manifest_path, MAX_ARTIFACT_BYTES)
        decode_json(manifest_raw, MAX_ARTIFACT_BYTES)
        config = _trust(raw, manifest_raw, repo_root, expected_source_commit)
        from silent_cascade.train.curriculum_data import ComponentManifest

        manifest = ComponentManifest.model_validate_json(manifest_raw)
        p = raw["provenance"]
        production = raw["publication"] == "production"
        count = 10000 if production else 16
        _require(
            manifest.publication == raw["publication"]
            and manifest.count == count == len(raw["rows"]),
            "fixed denominator mismatch",
        )
        _require(
            (manifest.source_revision, manifest.plan_revision, manifest.config_hash)
            == (p["source_commit"], p["plan_revision"], p["config_sha256"]),
            "manifest provenance mismatch",
        )
        for i, entry in enumerate(manifest.entries):
            key = entry.key
            _require(
                (key.split, key.root_seed, key.public_id_seed, key.episode_index, key.stage)
                == ("validation" if production else "debug", 313, 337, i, "one_hop"),
                "manifest key identity mismatch",
            )
        corpus = manifest.build_corpus(config.config)
        chain = hashlib.sha256(b"silent-cascade/phase3/rows/v1\0").digest()
        for i, row in enumerate(raw["rows"]):
            target = json.loads(_canonical(asdict(corpus.targets[i])))
            _require(
                row["target"] == target
                and row["position"] == i
                and row["public_id"] == corpus.public_examples[i].init.episode_public_id
                and row["example_hash"] == manifest.entries[i].example_hash,
                "raw example identity/target mismatch",
            )
            _require(
                row["variant"] == ("positive", "safe", "disconnected")[target["terminal_status"]],
                "raw variant mismatch",
            )
            for name, value in _score(row["prediction"], target).items():
                _require(row[name] == value, f"raw row derived field mismatch: {i}.{name}")
            _require(
                row["source_commit"] == p["source_commit"]
                and row["config_sha256"] == p["config_sha256"]
                and row["model_state_sha256"] == raw["training"]["weights"]["model_state_sha256"],
                "raw model identity mismatch",
            )
            encoded = _canonical(row)
            chain = hashlib.sha256(
                chain + i.to_bytes(4, "big") + len(encoded).to_bytes(8, "big") + encoded
            ).digest()
        _require(chain.hex() == raw["rows_sha256"], "ordered framed row chain mismatch")
        summary = _summary(raw["rows"])
        _require(raw["summary"] == summary, "summary differs from independently reconstructed rows")
        _require(
            summary["variant_counts"]
            == (
                {"positive": 5000, "safe": 2500, "disconnected": 2500}
                if production
                else {"positive": 8, "safe": 4, "disconnected": 4}
            ),
            "fixed variant denominator mismatch",
        )
        _numeric(raw, config)
        _training(raw, config)
        offline = raw["offline"]
        _require(
            offline["training_steps"] == 1
            and offline["predicted_rows"] == 8
            and offline["backward_macs"] > 0
            and offline["blocked_import_attempts"] == offline["network_attempts"] == 0
            and not offline["forbidden_modules"],
            "offline/import execution boundary failed",
        )
        from silent_cascade.config import resolve_config
        from silent_cascade.train.config import Phase3Config

        smoke_paths = [*raw["provenance"]["config_paths"][:-1], "configs/train/smoke.yaml"]
        _require(
            offline["config_sha256"]
            == resolve_config(Phase3Config, tuple(repo_root / path for path in smoke_paths)).sha256,
            "offline smoke config identity mismatch",
        )
        for i, batch in enumerate(raw["compute"]):
            start = i * config.config.training.batch_size
            _require(
                batch["start"] == start
                and batch["count"] == min(config.config.training.batch_size, count - start),
                "compute batch attribution mismatch",
            )
            _require(
                0 < batch["parameters"] <= config.config.neural.max_trainable_parameters
                and batch["backward_macs"] == 0,
                "invalid evaluation compute",
            )
            for row in raw["rows"][start : start + batch["count"]]:
                _require(
                    row["compute_batch"] == i and row["prediction"]["atomic_transitions"] <= 4,
                    "row compute attribution or transition cap mismatch",
                )
        _require(sum(b["count"] for b in raw["compute"]) == count, "missing compute batches")
        _require(
            raw["repeated_prediction_count"] == count
            and raw["repeated_prediction_mismatches"] == 0,
            "CPU prediction repetition failed",
        )
        _require(
            {s["variant"] for s in raw["replay_samples"]} == {"positive", "safe", "disconnected"}
            and len({s["position"] for s in raw["replay_samples"]}) == 3,
            "replay samples lack distinct variants",
        )
        for sample in raw["replay_samples"]:
            row = raw["rows"][sample["position"]]
            _require(
                sample["variant"] == row["variant"]
                and sample["output_sha256"] == _sha(_canonical(row["prediction"])),
                "replay output hash mismatch",
            )
        passed = (
            production
            and summary["content_gates_pass"]
            and summary["cap_count"] == summary["invalid_prediction_count"] == 0
        )
        _require(raw["passed"] is passed, "stored acceptance flag mismatch")
        if weights_path is not None:
            _weights(weights_path, raw)
        return Phase3VerificationResult(
            valid=True,
            passed=passed,
            publication=raw["publication"],
            episode_count=count,
            source_commit=p["source_commit"],
            artifact_file_sha256=_sha(artifact_raw),
            checkpoint_sha256=raw["training"]["weights"]["file_sha256"],
        )
    except ArtifactIntegrityError:
        raise
    except (
        SilentCascadeError,
        ValidationError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        RecursionError,
        OSError,
    ) as error:
        raise ArtifactIntegrityError(f"invalid Phase 3 artifact: {error}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--expected-source-commit")
    args = parser.parse_args()
    try:
        result = verify_phase3_gate_artifact(
            artifact_path=args.artifact,
            manifest_path=args.manifest,
            weights_path=args.weights,
            expected_source_commit=args.expected_source_commit,
            repo_root=Path(__file__).resolve().parents[1],
        )
        print(result.model_dump_json())
        return 0 if result.passed else 1
    except SilentCascadeError as error:
        print(json.dumps({"error": error.code, "message": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
