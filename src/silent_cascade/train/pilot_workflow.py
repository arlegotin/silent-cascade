"""Authenticated local pilot orchestration and resumable single-writer ownership."""

import fcntl
import json
import math
import os
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import psutil

from silent_cascade.env.pilot import curriculum_to_bundle
from silent_cascade.eval.artifacts import EvaluationIdentity
from silent_cascade.eval.pilot_audit import audit_pilot_manifest
from silent_cascade.eval.runner import evaluate_episodes
from silent_cascade.eventflow.archive_io import archive_parent
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.neural import NeuralEventFlowAgent, NeuralModelIdentity
from silent_cascade.eventflow.neural_replay import verify_neural_replay, write_neural_replay
from silent_cascade.eventflow.neural_weights import load_neural_weights
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.report.pilot_artifacts import (
    child,
    load_evaluation,
    load_training_result,
    read_json,
)
from silent_cascade.train.pilot_artifact_index import training_result_payload
from silent_cascade.train.pilot_checkpoints import load_pilot_checkpoint
from silent_cascade.train.pilot_config import resolve_pilot_config
from silent_cascade.train.pilot_data import (
    _check_path,
    _publish_pilot_bytes,
    _read_pilot_bytes,
    freeze_pilot_manifest,
    iter_pilot_examples,
    load_pilot_manifest,
)
from silent_cascade.train.pilot_provenance import authenticate_pilot_source, verify_pilot_data
from silent_cascade.train.pilot_state import STAGES, PilotCheckpointDescriptor, PilotProgress
from silent_cascade.train.pilot_trainer import (
    PilotTrainingResult,
    run_pilot_training,
    verify_journal,
)
from silent_cascade.train.provenance import git


def resolve_pilot_path(path: Path):
    _check_path(path)
    choices = {
        Path("configs/train/pilot.yaml").absolute(): "phase4_pilot",
        Path("configs/train/pilot_smoke.yaml").absolute(): "phase4_smoke",
    }
    if path.absolute() not in choices:
        raise ValueError(
            "unsupported pilot config; use the closed pilot.yaml or pilot_smoke.yaml overlay"
        )
    _read_pilot_bytes(path)
    return resolve_pilot_config(choices[path.absolute()])


def _source(config):
    root = Path(__file__).resolve().parents[3]
    source = git(root, "rev-parse", "HEAD").decode().strip()
    identity = authenticate_pilot_source(repo_root=root, source_commit=source, config=config)
    return root, source, identity


def freeze_pilot(*, config_path: Path, stage: str, output: Path):
    config = resolve_pilot_path(config_path)
    _, source, _ = _source(config)
    return freeze_pilot_manifest(config, stage=stage, output_path=output, source_commit=source)


def _inputs(config, manifest_dir, *, prepare):
    root, source, identity = _source(config)
    manifests = {stage: manifest_dir / f"{stage}.json" for stage in STAGES}
    paths = [
        p
        for stage, path in manifests.items()
        for p in (path, manifest_dir / "audits" / stage / "report.json")
    ]
    for path in paths:
        _check_path(path)
    created = False
    if prepare:
        for stage, path in manifests.items():
            if not path.exists():
                freeze_pilot_manifest(config, stage=stage, output_path=path, source_commit=source)
                created = True
            audit = manifest_dir / "audits" / stage / "report.json"
            if not audit.exists():
                manifest = load_pilot_manifest(path, config=config)
                raw_dir = manifest_dir / "audit-raw" / stage
                report = audit_pilot_manifest(manifest, config=config.config, output_dir=raw_dir)
                raw = _read_pilot_bytes(raw_dir / "report.json")
                if raw != canonical_json_bytes(report):
                    raise ValueError("audit artifact mismatch")
                _publish_pilot_bytes(audit, raw)
                created = True
    message = (
        "commit-data precondition: introduce these exact manifest/audit files in Git, "
        "then create a distinct compatible training revision before optimization:\n"
        + "\n".join(str(p) for p in paths)
    )
    if created:
        raise ValueError(message)
    try:
        loaded, introductions = verify_pilot_data(
            repo_root=root, source_commit=source, config=config, manifests=manifests
        )
    except (ValueError, OSError) as error:
        raise ValueError(f"{message}\nVerification: {error}") from error
    return (
        source,
        identity.model_copy(update={"data_introductions": introductions}),
        manifests,
        loaded,
    )


def process_identity(pid: int):
    """Absolute process creation time distinguishes PID reuse, including after reboot."""
    try:
        process = psutil.Process(pid)
        return {"create_time": process.create_time()}
    except psutil.NoSuchProcess:
        return None
    except (psutil.AccessDenied, PermissionError) as error:
        raise ValueError(f"cannot inspect process identity for PID {pid}") from error


@dataclass(frozen=True)
class StoppedPilotOwner:
    prior_owner_canonical_json: str
    prior_owner_sha256: str
    checkpoint: PilotCheckpointDescriptor


@contextmanager
def pilot_ownership(run_dir: Path, *, run_identity: str, recover=None, capture_stopped=False):
    """A permanent inode lock prevents unlink/recreate races; records survive crashes."""
    _check_path(run_dir)
    lock = run_dir.parent / ("." + run_dir.name + ".owner.json")
    # Safe create-only publication pins all parents before creating the lock inode.
    if not lock.exists():
        _publish_pilot_bytes(lock, b"{}")
    with archive_parent(lock, error_factory=ValueError) as (parent, name):
        descriptor = os.open(name, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        acquired = False
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("ownership record must be a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                existing = read_json(lock)
                if existing.get("run_identity") != run_identity:
                    raise ValueError(
                        "ownership run identity mismatch; new-run-path required"
                    ) from None
                raise ValueError(f"live pilot writer: PID {existing.get('pid')}") from None
            raw = os.read(descriptor, 65537)
            if len(raw) > 65536:
                raise ValueError("oversized ownership record")
            existing = json.loads(raw)
            stopped = None
            recovered = False
            if not isinstance(existing, dict):
                raise ValueError("invalid ownership record")
            if existing:
                start = existing.get("process_start")
                if (
                    set(existing) != {"pid", "process_start", "run_identity", "active"}
                    or type(existing["pid"]) is not int
                    or existing["pid"] <= 0
                    or type(existing["active"]) is not bool
                    or not isinstance(start, dict)
                    or set(start) != {"create_time"}
                    or type(start["create_time"]) is not float
                    or not math.isfinite(start["create_time"])
                    or start["create_time"] <= 0
                ):
                    raise ValueError("invalid ownership record")
                if existing.get("run_identity") != run_identity:
                    raise ValueError("ownership run identity mismatch; new-run-path required")
                if existing.get("active"):
                    if process_identity(existing["pid"]) == existing["process_start"]:
                        raise ValueError(f"live matching pilot process: PID {existing['pid']}")
                    if run_dir.exists():
                        if recover is None:
                            raise ValueError(
                                "stale ownership requires verified last durable checkpoint"
                            )
                        recover()
                        recovered = True
                if capture_stopped and run_dir.exists():
                    if recover is None:
                        raise ValueError("stopped custody requires durable recovery")
                    if not recovered:
                        recover()
                    checkpoint_descriptor = PilotCheckpointDescriptor.model_validate_json(
                        canonical_json_bytes(read_json(run_dir / "checkpoint-index.json")["latest"])
                    )
                    if raw != canonical_json_bytes(existing):
                        raise ValueError("prior pilot owner is not canonical")
                    stopped = StoppedPilotOwner(
                        raw.decode(), sha256_bytes(raw), checkpoint_descriptor
                    )
            record = dict(
                pid=os.getpid(),
                process_start=process_identity(os.getpid()),
                run_identity=run_identity,
                active=True,
            )

            def write(value):
                os.lseek(descriptor, 0, os.SEEK_SET)
                os.ftruncate(descriptor, 0)
                os.write(descriptor, canonical_json_bytes(value))
                os.fsync(descriptor)

            write(record)
            try:
                yield stopped
            finally:
                write(record | {"active": False})
        finally:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _durable(run_dir, config, source, device, *, result=None, evidence_context=None):
    from silent_cascade.archive.readers import evidence_path

    with evidence_path(
        run_dir, "checkpoint-index.json", evidence_context=evidence_context
    ) as local:
        index = read_json(local)
    descriptor = PilotCheckpointDescriptor.model_validate_json(
        canonical_json_bytes(index["latest"])
    )
    path = child(run_dir, descriptor.path)
    with evidence_path(run_dir, descriptor.path, evidence_context=evidence_context) as local:
        session = load_pilot_checkpoint(
            local, expected_sha256=descriptor.sha256, config=config, source=source, device=device
        )
    verify_journal(run_dir, session.progress, evidence_context=evidence_context)
    if (
        session.progress.global_step != descriptor.global_step
        or NeuralModelIdentity.from_model(
            session.model, source_revision=source.source_commit
        ).model_state_sha256
        != descriptor.model_state_sha256
    ):
        raise ValueError("last durable checkpoint descriptor mismatch")
    if result is not None:
        progress = PilotProgress.model_validate_json(canonical_json_bytes(result["progress"]))
        if (
            progress.model_dump(exclude={"latest"})
            != session.progress.model_dump(exclude={"latest"})
            or progress.latest != descriptor
            or NeuralModelIdentity(**result["model_identity"])
            != NeuralModelIdentity.from_model(session.model, source_revision=source.source_commit)
        ):
            raise ValueError("training result differs from last durable checkpoint")
    return path


def _restore_result(payload):
    values = dict(payload)
    values["progress"] = PilotProgress.model_validate_json(canonical_json_bytes(values["progress"]))
    values["model_identity"] = NeuralModelIdentity(**values["model_identity"])
    for name in ("selected_checkpoint", "selected_weights", "latest_weights"):
        if values[name] is not None:
            values[name] = PilotCheckpointDescriptor.model_validate_json(
                canonical_json_bytes(values[name])
            )
    return PilotTrainingResult(**values)


def _train(
    config,
    manifests,
    run_dir,
    device,
    source_commit,
    source,
    resume,
    *,
    archive_producer=None,
    evidence_context=None,
):
    from silent_cascade.archive.readers import logical_root

    candidates = set()
    completed = (run_dir / "training-result.json").exists()
    if evidence_context is not None:
        if logical_root(run_dir, evidence_context) != ".":
            raise ValueError("training context requires the session run root")
        # Exhaust the validated inventory, retaining only completed result paths.
        for entry in evidence_context.entries():
            if entry.path == "training-result.json":
                completed = True
            elif re.fullmatch(r"attempt-[0-9a-f]{32}/result\.json", entry.path):
                candidates.add(child(run_dir, entry.path))
    candidates.update(
        path
        for path in run_dir.glob("attempt-*/result.json")
        if re.fullmatch(r"attempt-[0-9a-f]{32}", path.parent.name)
    )
    if completed:
        payload = load_training_result(
            run_dir, run_dir / "training-result.json", evidence_context=evidence_context
        )
        durable = _durable(
            run_dir, config, source, device, result=payload, evidence_context=evidence_context
        )
        if resume is not None and resume.absolute() != durable.absolute():
            raise ValueError("resume must name the last durable checkpoint")
        return _restore_result(payload)
    if candidates or (run_dir.exists() and (archive_producer is None or any(run_dir.iterdir()))):
        durable = _durable(run_dir, config, source, device, evidence_context=evidence_context)
        if resume is not None and resume.absolute() != durable.absolute():
            raise ValueError("resume must name the last durable checkpoint")
        # Recover publication interrupted after the trainer's final result.
        if candidates:
            results = [
                (p, load_training_result(run_dir, p, evidence_context=evidence_context))
                for p in sorted(candidates)
            ]
            payload = max(results, key=lambda item: item[1]["progress"]["global_step"])[1]
            _durable(
                run_dir, config, source, device, result=payload, evidence_context=evidence_context
            )
            result = _restore_result(payload)
        else:
            result = run_pilot_training(
                config,
                manifests=manifests,
                run_dir=run_dir,
                device=device,
                source_commit=source_commit,
                resume=durable,
                archive_producer=archive_producer,
                evidence_context=evidence_context,
            )
    else:
        if resume is not None:
            raise ValueError("resume requires its existing run directory")
        result = run_pilot_training(
            config,
            manifests=manifests,
            run_dir=run_dir,
            device=device,
            source_commit=source_commit,
            archive_producer=archive_producer,
            evidence_context=evidence_context,
        )
    _publish_pilot_bytes(
        run_dir / "training-result.json",
        canonical_json_bytes(training_result_payload(result, run_dir=run_dir)),
    )
    return result


def train_pilot(
    *,
    config_path: Path,
    manifest_dir: Path,
    run_dir: Path,
    device: str,
    seed: int = 11,
    resume: Path | None = None,
    archive_producer=None,
    evidence_context=None,
):
    config = resolve_pilot_path(config_path)
    if seed != 11:
        raise ValueError("pilot model seed must be 11")
    return _workflow(
        config,
        manifest_dir,
        run_dir,
        device,
        resume=resume,
        complete=False,
        archive_producer=archive_producer,
        evidence_context=evidence_context,
    )


def _workflow(
    config,
    manifest_dir,
    run_dir,
    device,
    *,
    resume=None,
    complete,
    archive_producer=None,
    evidence_context=None,
):
    if device not in {"cpu", "mps"}:
        raise ValueError("pilot device must be cpu or mps")
    _check_path(manifest_dir)
    root, _, source = _source(config)
    locator = manifest_dir.absolute().relative_to(root).as_posix()
    identity = sha256_bytes(
        canonical_json_bytes(
            {
                "config_sha256": config.sha256,
                "executable_sha256": source.source_sha256,
                "device": device,
                "manifest_directory": locator,
            }
        )
    )

    def recover():
        _, authenticated, _, _ = _inputs(config, manifest_dir, prepare=False)
        _durable(run_dir, config, authenticated, device, evidence_context=evidence_context)

    with pilot_ownership(
        run_dir,
        run_identity=identity,
        recover=recover,
        capture_stopped=archive_producer is not None,
    ) as stopped:
        source_commit, source, manifests, loaded = _inputs(config, manifest_dir, prepare=complete)
        if stopped is not None:
            archive_producer.adopt_stopped(
                stopped, source_commit=source_commit, config_sha256=config.sha256
            )
        metadata = dict(
            schema_version="phase4-pilot-workflow-v1",
            config_sha256=config.sha256,
            source_commit=source_commit,
            device=device,
            seed=11,
            manifest_hashes={
                stage: sha256_bytes(canonical_json_bytes(m)) for stage, m in loaded.items()
            },
        )
        result = _train(
            config,
            manifests,
            run_dir,
            device,
            source_commit,
            source,
            resume,
            archive_producer=archive_producer,
            evidence_context=evidence_context,
        )
        _publish_pilot_bytes(run_dir / "workflow.json", canonical_json_bytes(metadata))
        if complete:
            from silent_cascade.report.pilot import build_pilot_report
            from silent_cascade.train.pilot_checks import _run_pilot_checks_owned

            _run_pilot_checks_owned(
                run_dir=run_dir,
                config=config,
                output_path=run_dir / "phase4-gate.json",
                archive_producer=archive_producer,
                evidence_context=evidence_context,
            )
            build_pilot_report(
                run_dir=run_dir, output_dir=run_dir / "report", evidence_context=evidence_context
            )
        return result


def run_pilot(
    *,
    config_path: Path,
    manifest_dir: Path,
    run_dir: Path,
    device: str,
    archive_producer=None,
    evidence_context=None,
) -> PilotTrainingResult:
    return _workflow(
        resolve_pilot_path(config_path),
        manifest_dir,
        run_dir,
        device,
        complete=True,
        archive_producer=archive_producer,
        evidence_context=evidence_context,
    )


def evaluate_pilot(*, checkpoint: Path, manifest: Path, output: Path, device: str):
    """Authenticate current source, introduced data and a concrete portable descriptor."""
    for path in (checkpoint, manifest, output):
        _check_path(path)
    descriptor = PilotCheckpointDescriptor.model_validate_json(_read_pilot_bytes(checkpoint))
    raw_manifest = read_json(manifest)
    config = resolve_pilot_config(
        "phase4_pilot" if raw_manifest.get("split") == "validation" else "phase4_smoke"
    )
    source_commit, source, _, loaded = _inputs(config, manifest.parent, prepare=False)
    stage = raw_manifest.get("stage")
    if stage not in loaded or loaded[stage] != load_pilot_manifest(manifest, config=config):
        raise ValueError("unsupported evaluation manifest")
    if config.config.pilot.is_production and not descriptor.eligible:
        raise ValueError("production evaluation requires eligible selected weights")
    result_path = checkpoint.parent / "training-result.json"
    candidates = (
        [result_path]
        if result_path.exists()
        else list(checkpoint.parent.glob("attempt-*/result.json"))
    )
    matched = []
    for path in candidates:
        payload = load_training_result(checkpoint.parent, path)
        expected = (
            payload["selected_weights"]
            if config.config.pilot.is_production
            else payload["latest_weights"]
        )
        if expected is not None and descriptor.model_dump(mode="json") == expected:
            matched.append(payload)
    if len(matched) != 1:
        raise ValueError("checkpoint descriptor differs from authenticated trainer result")
    _durable(checkpoint.parent, config, source, device, result=matched[0])
    weights_path = child(checkpoint.parent, descriptor.path)
    weights = load_neural_weights(weights_path, expected_sha256=descriptor.sha256, device=device)
    if (
        weights.identity.model_state_sha256 != descriptor.model_state_sha256
        or weights.identity.source_revision != source_commit
        or weights.identity.model_config_json != canonical_json_bytes(config.config.neural).decode()
    ):
        raise ValueError("evaluation checkpoint source/config/model identity mismatch")
    identity = EvaluationIdentity(
        experiment="phase4-pilot-v1",
        stage=stage,
        split=loaded[stage].split,
        purpose="pilot_validation" if config.config.pilot.is_production else "debug",
        manifest_schema=loaded[stage].schema_version,
        manifest_sha256=sha256_bytes(canonical_json_bytes(loaded[stage])),
        episodes=tuple(entry.projected for entry in loaded[stage].entries),
        checkpoint_sha256=descriptor.sha256,
        model_identity=weights.identity,
        evaluation_config_canonical_json=config.canonical_json.decode(),
        execution_source_revision=source_commit,
    )
    _check_path(output)
    _publish_pilot_bytes(
        output.parent / ("." + output.name + ".evaluation-request.json"),
        canonical_json_bytes({"identity_sha256": identity.sha256, "device": device}),
    )
    if output.exists():
        previous, _, metrics, _ = load_evaluation(output)
        if previous != identity:
            raise ValueError("incompatible evaluation; new-run-path required")
    else:
        evaluation = evaluate_episodes(
            weights.model,
            identity=identity,
            config=config.config,
            episodes=(
                curriculum_to_bundle(e, config=config.config)
                for e in iter_pilot_examples(loaded[stage], config=config.config)
            ),
            output_dir=output,
            device=device,
        )
        metrics = evaluation.metrics
    replay_path = output / "replay.json"
    if not replay_path.exists():
        # Replay one successful execution; a dynamics failure remains in its raw denominator.
        _, rows, _, _ = load_evaluation(output)
        candidate = next((i for i, row in enumerate(rows) if row.error is None), None)
        if candidate is not None:
            examples = iter_pilot_examples(loaded[stage], config=config.config)
            bundle = next(
                curriculum_to_bundle(e, config=config.config)
                for i, e in enumerate(examples)
                if i == candidate
            )
            agent = NeuralEventFlowAgent(weights.model, identity=weights.identity, device=device)
            result = EventEngine(config.config.event_flow).run_episode(bundle, agent)
            write_neural_replay(
                replay_path,
                bundle=bundle,
                result=result,
                identity=weights.identity,
                config=config.config.event_flow,
                weights=weights_path,
                source_revision=source_commit,
                experiment_config_canonical_json=config.canonical_json.decode(),
            )
    if replay_path.exists():
        verify_neural_replay(replay_path, weights_path=weights_path)
    return metrics
