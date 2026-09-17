"""Fresh, offline curriculum optimization with a single resumable global budget."""

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from silent_cascade.config import ResolvedConfig
from silent_cascade.env.pilot import curriculum_to_bundle
from silent_cascade.eval.artifacts import EvaluationIdentity, read_evaluation_artifact
from silent_cascade.eval.compute import NeuralComputeMeter
from silent_cascade.eval.runner import evaluate_episodes
from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at
from silent_cascade.eventflow.neural import NeuralModelIdentity
from silent_cascade.eventflow.neural_weights import MAX_ARCHIVE_BYTES, save_neural_weights
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.rng import seed_all
from silent_cascade.train.archive_tensors import snapshot_optimizer
from silent_cascade.train.checkpoints import _device, _publish_bytes_at
from silent_cascade.train.component_eval import evaluate_components
from silent_cascade.train.objective import training_objective
from silent_cascade.train.pilot_checkpoints import load_pilot_checkpoint, save_pilot_checkpoint
from silent_cascade.train.pilot_config import Phase4Config
from silent_cascade.train.pilot_data import (
    _publish_pilot_bytes,
    _read_pilot_bytes,
    iter_pilot_examples,
    next_pilot_batch,
    pilot_component_corpus,
)
from silent_cascade.train.pilot_provenance import authenticate_pilot_source, verify_pilot_data
from silent_cascade.train.pilot_state import (
    PilotCheckpointDescriptor,
    PilotProgress,
    ValidationRecord,
    advance_progress,
    apply_validation,
)
from silent_cascade.train.state import TrainingError
from silent_cascade.train.trainer import (
    StepResult,
    _check_objective,
    _finite_state,
    _json,
    make_optimizer,
)


def pilot_train_one_step(model, optimizer, batch, config) -> StepResult:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with NeuralComputeMeter(model) as meter:
        objective = training_objective(model, batch, config.pilot)
        _check_objective(objective)
        objective.total.backward()
        for p in model.parameters():
            if p.grad is not None:
                _finite_state(p.grad)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        for p in model.parameters():
            _finite_state(p)
        _finite_state(optimizer.state)
    branches = [("timed/", objective.timed_loss), ("content/", objective.content_loss)]
    values = {
        name: {
            prefix + key: float(value.detach())
            for prefix, loss in branches
            for key, value in getattr(loss, name).items()
        }
        for name in ("terms", "subterms", "numerators", "denominators")
    }
    return StepResult(
        float(objective.total.detach()),
        **values,
        per_position={
            prefix + key: value.detach().cpu().tolist()
            for prefix, loss in branches
            for key, value in loss.per_position.items()
        },
        gradient_norm=float(norm),
        compute=meter.snapshot(),
        objective_version=objective.objective_version,
        auxiliary_coefficient=objective.auxiliary_coefficient,
        timed_loss=float(objective.timed_loss.total.detach()),
        content_loss=float(objective.content_loss.total.detach()),
        timed_compute=objective.timed.compute,
        content_compute=objective.content.compute,
    )


def publish_json(path, value):
    raw = canonical_json_bytes(_json(value))
    _publish_pilot_bytes(path, raw)
    return sha256_bytes(raw)


def _write_index(parent, value):
    _publish_bytes_at(parent, "checkpoint-index.json", canonical_json_bytes(value), replace=True)


def _publish_selected(run_dir, progress, config):
    if not config.config.pilot.is_production or progress.status != "robustness_complete":
        return
    selected = progress.selected
    if selected is None or not selected.eligible or selected.stage != "robustness":
        raise TrainingError("completed pilot has no eligible portable selection")
    path = Path(selected.path)
    if path.is_absolute() or ".." in path.parts:
        raise TrainingError("selected weights are not owned by this run")
    if sha256_bytes(_read_pilot_bytes(run_dir / path)) != selected.sha256:
        raise TrainingError("selected weights hash differs")
    publish_json(run_dir / "selected.json", selected)


def _publish_checkpoint(run_dir, pending, step, digest):
    """Publish and clean up only owned names beneath a single pinned run directory."""
    relative = pending.relative_to(run_dir)
    if (
        len(relative.parts) != 2
        or not relative.parts[0].startswith("attempt-")
        or relative.name != f"checkpoint-{step}.safetensors"
    ):
        raise TrainingError("pending checkpoint is not owned by this run")
    name = f"training-{step}-{digest}.safetensors"
    with archive_parent(run_dir / name, error_factory=TrainingError) as (parent, name):
        source = os.open(
            relative.parts[0], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
        )
        try:
            raw = read_archive_at(
                source, relative.name, max_bytes=MAX_ARCHIVE_BYTES, error_factory=TrainingError
            )
            if sha256_bytes(raw) != digest:
                raise TrainingError("pending checkpoint hash differs")
            _publish_bytes_at(parent, name, raw, replace=False)
            os.unlink(relative.name, dir_fd=source)
            os.fsync(source)
        finally:
            os.close(source)
    return name


def _journal(run_dir, progress, value):
    record = {"prior": progress.journal_sha256, "global_step": progress.global_step, **value}
    raw = canonical_json_bytes(record)
    digest = sha256_bytes(raw)
    _publish_pilot_bytes(run_dir / f"journal-{digest}.json", raw)
    return progress.model_copy(update={"journal_sha256": digest})


def verify_journal(run_dir, progress):
    cursor, seen, steps = progress.journal_sha256, set(), []
    while cursor is not None:
        if cursor in seen:
            raise TrainingError("cyclic pilot journal")
        seen.add(cursor)
        raw = _read_pilot_bytes(run_dir / f"journal-{cursor}.json")
        if sha256_bytes(raw) != cursor:
            raise TrainingError("pilot journal hash mismatch")
        record = json.loads(raw)
        if record["kind"] == "update":
            steps.append(record["global_step"])
        for name, digest in record.get("artifacts", {}).items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise TrainingError("unsafe journal artifact path")
            if sha256_bytes(_read_pilot_bytes(run_dir / name)) != digest:
                raise TrainingError("pilot journal artifact mismatch")
        cursor = record["prior"]
    if steps != list(range(progress.global_step, 0, -1)):
        raise TrainingError("pilot journal has missing or duplicate committed updates")
    return seen


def retain_checkpoints(run_dir, descriptors, latest):
    """Retain best three in the current stage plus latest, preserving pruned descriptors."""
    candidates = [d for d in descriptors if d.stage == latest.stage and d.rank]
    best = sorted(candidates, key=lambda d: d.rank, reverse=True)[:3]
    kept = {d.path for d in (*best, latest)}
    history = [d.model_dump(mode="json") for d in descriptors]
    with archive_parent(run_dir / "checkpoint-index.json", error_factory=TrainingError) as (
        parent,
        _,
    ):
        prunable = []
        for descriptor in descriptors:
            name = descriptor.path
            if name != f"training-{descriptor.global_step}-{descriptor.sha256}.safetensors":
                raise TrainingError("checkpoint is not owned by this run")
            try:
                raw = read_archive_at(
                    parent, name, max_bytes=MAX_ARCHIVE_BYTES, error_factory=TrainingError
                )
            except FileNotFoundError:
                if name in kept:
                    raise TrainingError("retained checkpoint is missing") from None
                continue
            if sha256_bytes(raw) != descriptor.sha256:
                raise TrainingError("refusing to retain or prune changed checkpoint")
            if name not in kept:
                prunable.append(name)
        _write_index(
            parent,
            {
                "latest": latest.model_dump(mode="json"),
                "best": [d.model_dump(mode="json") for d in best],
                "history": history,
            },
        )
        for name in prunable:
            os.unlink(name, dir_fd=parent)
        os.fsync(parent)
    return best


def _validation(model, manifest, *, step, config, identity, weights_sha, directory, device):
    manifest_sha = sha256_bytes(canonical_json_bytes(manifest))
    artifacts = {}
    if manifest.stage == "one_hop":
        evaluation = evaluate_components(
            model,
            pilot_component_corpus(manifest, config=config.config),
            batch_size=config.config.pilot.batch_size,
        )
        rows = tuple(json.loads(canonical_json_bytes(asdict(r))) for r in evaluation.metrics.rows)
        artifacts["components.json"] = publish_json(
            directory / "components.json", _json(evaluation)
        )
        kind = "content_validation"
    evaluation_identity = EvaluationIdentity(
        experiment="phase4-pilot-v1",
        stage=manifest.stage,
        split=manifest.split,
        purpose="pilot_validation" if config.config.pilot.is_production else "debug",
        manifest_schema=manifest.schema_version,
        manifest_sha256=manifest_sha,
        episodes=tuple(e.projected for e in manifest.entries),
        checkpoint_sha256=weights_sha,
        model_identity=identity,
        evaluation_config_canonical_json=config.canonical_json.decode(),
        execution_source_revision=identity.source_revision,
    )
    evaluation = evaluate_episodes(
        model,
        identity=evaluation_identity,
        config=config.config,
        episodes=(
            curriculum_to_bundle(e, config=config.config)
            for e in iter_pilot_examples(manifest, config=config.config)
        ),
        output_dir=directory / "autonomous",
        device=device,
    )
    artifacts.update({"autonomous/" + p: h for p, h in evaluation.artifact_hashes})
    timed_rows = tuple(
        json.loads(line)
        for line in read_evaluation_artifact(directory / "autonomous/rows.jsonl").splitlines()
    )
    if manifest.stage == "one_hop":
        rows = tuple(
            {**content, "error": timed["error"]}
            for content, timed in zip(rows, timed_rows, strict=True)
        )
    if manifest.stage != "one_hop":
        rows = timed_rows
        kind = "autonomous_validation"
    record = ValidationRecord(
        stage=manifest.stage,
        global_step=step,
        manifest_sha256=manifest_sha,
        checkpoint_sha256=weights_sha,
        model_state_sha256=identity.model_state_sha256,
        rows=rows,
        rows_sha256=sha256_bytes(canonical_json_bytes({"rows": rows})),
        production=config.config.pilot.is_production,
        evidence_kind=kind,
    )
    artifacts["validation.json"] = publish_json(directory / "validation.json", record)
    return record, {str(directory / p): h for p, h in artifacts.items()}


@dataclass(frozen=True)
class PilotTrainingResult:
    status: str
    progress: PilotProgress
    selected_checkpoint: PilotCheckpointDescriptor | None
    selected_weights: PilotCheckpointDescriptor | None
    latest_weights: PilotCheckpointDescriptor
    artifact_hashes: dict[str, str]
    model_identity: NeuralModelIdentity
    gate_eligible: bool


def run_pilot_training(
    config: ResolvedConfig[Phase4Config],
    *,
    manifests: Mapping[str, Path],
    run_dir: Path,
    device: str,
    source_commit: str,
    resume: Path | None = None,
) -> PilotTrainingResult:
    root = Path(__file__).resolve().parents[3]
    source = authenticate_pilot_source(repo_root=root, source_commit=source_commit, config=config)
    loaded, introductions = verify_pilot_data(
        repo_root=root, source_commit=source_commit, config=config, manifests=manifests
    )
    source = source.model_copy(update={"data_introductions": introductions})
    _device(device)
    hashes = {stage: sha256_bytes(canonical_json_bytes(m)) for stage, m in loaded.items()}
    descriptors = []
    if resume is None:
        if run_dir.exists():
            raise TrainingError("fresh pilot attempt requires a new directory")
        run_dir.mkdir(parents=True)
        seed_all(11)
        model = EventFlowModel(config.config.neural).to(device)
        optimizer = make_optimizer(model, config.config.pilot)
        progress = PilotProgress(manifest_hashes=hashes)
    else:
        index = json.loads(_read_pilot_bytes(run_dir / "checkpoint-index.json"))
        latest = PilotCheckpointDescriptor.model_validate_json(
            canonical_json_bytes(index["latest"])
        )
        if (
            latest.path != f"training-{latest.global_step}-{latest.sha256}.safetensors"
            or resume.absolute() != (run_dir / latest.path).absolute()
        ):
            raise TrainingError("resume must use latest durable checkpoint")
        session = load_pilot_checkpoint(
            resume, expected_sha256=latest.sha256, config=config, source=source, device=device
        )
        model, optimizer, progress = session.model, session.optimizer, session.progress
        identity = NeuralModelIdentity.from_model(model, source_revision=source_commit)
        if (
            latest.global_step != progress.global_step
            or latest.model_state_sha256 != identity.model_state_sha256
        ):
            raise TrainingError("latest descriptor disagrees with checkpoint")
        progress = progress.model_copy(update={"latest": latest})
        if progress.manifest_hashes != hashes:
            raise TrainingError("resume manifests differ")
        committed = verify_journal(run_dir, progress)
        abandoned = sorted(
            p.name
            for p in run_dir.glob("journal-*.json")
            if p.name.removeprefix("journal-").removesuffix(".json") not in committed
        )
        publish_json(
            run_dir / f"restart-{uuid.uuid4().hex}.json",
            {"last_durable_step": progress.global_step, "uncommitted_journal_tail": abandoned},
        )
        descriptors = [
            PilotCheckpointDescriptor.model_validate_json(canonical_json_bytes(d))
            for d in index["history"]
        ]
    attempt = "attempt-" + uuid.uuid4().hex
    attempt_dir = run_dir / attempt
    attempt_dir.mkdir()

    def durable(rank=(), eligible=False, evaluated_stage=None):
        current = authenticate_pilot_source(
            repo_root=root, source_commit=source_commit, config=config
        )
        if current != source.model_copy(update={"data_introductions": {}}):
            raise TrainingError("executing source changed before publication")
        snapshot_optimizer(model, optimizer, config.config.pilot, progress.global_step)
        pending = attempt_dir / f"checkpoint-{progress.global_step}.safetensors"
        digest = save_pilot_checkpoint(
            pending,
            model=model,
            optimizer=optimizer,
            progress=progress,
            config=config,
            source=source,
        )
        name = _publish_checkpoint(run_dir, pending, progress.global_step, digest)
        identity = NeuralModelIdentity.from_model(model, source_revision=source_commit)
        descriptor = PilotCheckpointDescriptor(
            path=name,
            sha256=digest,
            model_state_sha256=identity.model_state_sha256,
            global_step=progress.global_step,
            stage=evaluated_stage or progress.stage,
            rank=rank,
            eligible=eligible,
        )
        descriptors.append(descriptor)
        retain_checkpoints(run_dir, descriptors, descriptor)
        return descriptor

    if resume is None:
        durable()
    try:
        while progress.status == "running" and progress.global_step < config.config.pilot.max_steps:
            batch = next_pilot_batch(
                config.config, stage=progress.stage, batch_counter=progress.batch_counter
            ).to(device)
            result = pilot_train_one_step(model, optimizer, batch, config.config)
            old_stage = progress.stage
            progress = advance_progress(progress, ceiling=config.config.pilot.max_steps)
            progress = _journal(
                run_dir,
                progress,
                {
                    "kind": "update",
                    "attempt": attempt,
                    "stage": old_stage,
                    "batch_counter": progress.batch_counter,
                    "batch_sha256": sha256_bytes(
                        canonical_json_bytes({"examples": batch.example_hashes})
                    ),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "source": source.source_sha256,
                    "config": config.sha256,
                    "result": _json(result),
                },
            )
            scheduled = progress.global_step % config.config.pilot.validation_every_steps == 0
            rank, eligible = (), False
            if scheduled:
                identity = NeuralModelIdentity.from_model(model, source_revision=source_commit)
                weights_path = attempt_dir / f"weights-{progress.global_step}.safetensors"
                weights_sha = save_neural_weights(weights_path, model=model, identity=identity)
                progress = progress.model_copy(
                    update={
                        "evaluation_weights": PilotCheckpointDescriptor(
                            path=str(weights_path.relative_to(run_dir)),
                            sha256=weights_sha,
                            model_state_sha256=identity.model_state_sha256,
                            global_step=progress.global_step,
                            stage=old_stage,
                        )
                    }
                )
                record, artifacts = _validation(
                    model,
                    loaded[old_stage],
                    step=progress.global_step,
                    config=config,
                    identity=identity,
                    weights_sha=weights_sha,
                    directory=attempt_dir / f"validation-{progress.global_step}-{old_stage}",
                    device=device,
                )
                primary = None
                if old_stage == "robustness":
                    primary, more = _validation(
                        model,
                        loaded["primary"],
                        step=progress.global_step,
                        config=config,
                        identity=identity,
                        weights_sha=weights_sha,
                        directory=attempt_dir / f"validation-{progress.global_step}-primary",
                        device=device,
                    )
                    artifacts.update(more)
                prior = progress
                progress = apply_validation(
                    progress,
                    record,
                    primary=primary,
                    interval=config.config.pilot.validation_every_steps,
                )
                rank, eligible = record.rank_and_gate()
                if old_stage == "robustness":
                    rank = progress.best_rank if progress.status == "robustness_complete" else ()
                    eligible = progress.status == "robustness_complete"
                artifacts[str(weights_path)] = weights_sha
                if len(progress.promotion_hashes) > len(prior.promotion_hashes):
                    certificate = {"validation": record, "primary": primary}
                    cert_path = attempt_dir / f"promotion-{progress.global_step}.json"
                    artifacts[str(cert_path)] = publish_json(cert_path, certificate)
                progress = _journal(
                    run_dir,
                    progress,
                    {
                        "kind": "validation",
                        "attempt": attempt,
                        "artifacts": {
                            str(Path(p).relative_to(run_dir)): h for p, h in artifacts.items()
                        },
                    },
                )
                if old_stage == "robustness" and eligible:
                    progress = progress.model_copy(
                        update={
                            "selected": PilotCheckpointDescriptor(
                                path=str(weights_path.relative_to(run_dir)),
                                sha256=weights_sha,
                                model_state_sha256=identity.model_state_sha256,
                                global_step=progress.global_step,
                                stage=old_stage,
                                rank=rank,
                                eligible=True,
                            )
                        }
                    )
            if (
                progress.status == "running"
                and progress.global_step >= config.config.pilot.max_steps
            ):
                progress = progress.model_copy(update={"status": "step_ceiling"})
            if scheduled or progress.status != "running":
                descriptor = durable(rank, eligible, old_stage)
                progress = progress.model_copy(update={"latest": descriptor})
    except Exception as error:
        publish_json(
            attempt_dir / "failure.json",
            {
                "step": progress.global_step,
                "type": type(error).__name__,
                "message": str(error),
                "source": source,
                "status": "uncommitted_since_last_checkpoint",
            },
        )
        raise
    identity = NeuralModelIdentity.from_model(model, source_revision=source_commit)
    final_weights_path = attempt_dir / "latest-weights.safetensors"
    final_weights_sha = save_neural_weights(final_weights_path, model=model, identity=identity)
    latest_weights = PilotCheckpointDescriptor(
        path=str(final_weights_path.relative_to(run_dir)),
        sha256=final_weights_sha,
        model_state_sha256=identity.model_state_sha256,
        global_step=progress.global_step,
        stage=progress.stage,
    )
    _publish_selected(run_dir, progress, config)
    artifacts = {
        str(p.relative_to(run_dir)): sha256_bytes(_read_pilot_bytes(p))
        for p in sorted(run_dir.rglob("*"))
        if p.is_file()
    }
    result = PilotTrainingResult(
        progress.status,
        progress,
        progress.latest if progress.status == "robustness_complete" else None,
        progress.selected,
        latest_weights,
        artifacts,
        identity,
        config.config.pilot.is_production and progress.status == "robustness_complete",
    )
    publish_json(attempt_dir / "result.json", _json(result))
    return result
