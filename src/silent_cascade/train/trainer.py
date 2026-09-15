"""Bounded offline optimization, validation, and checkpoint-driven continuation."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path

import torch

from silent_cascade.config import ResolvedConfig
from silent_cascade.eval.compute import NeuralComputeMeter, NeuralComputeSnapshot
from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.models.config import NeuralModelConfig
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.rng import seed_all
from silent_cascade.train.batches import next_training_batch, pack_training_examples
from silent_cascade.train.checkpoints import (
    _publish_bytes_at,
    export_weights,
    load_checkpoint_index,
    load_training_checkpoint,
    prune_training_checkpoints,
    save_training_checkpoint,
    update_checkpoint_index,
)
from silent_cascade.train.component_eval import evaluate_components
from silent_cascade.train.config import Phase3Config
from silent_cascade.train.curriculum_data import (
    ComponentCorpus,
    CurriculumKey,
    make_curriculum_example,
)
from silent_cascade.train.objective import training_objective
from silent_cascade.train.state import CheckpointDescriptor, TrainingError, TrainProgress
from silent_cascade.train.traces import build_teacher_trace, component_target


@dataclass(frozen=True, slots=True)
class StepResult:
    loss: float
    terms: dict[str, float]
    subterms: dict[str, float]
    numerators: dict[str, float]
    denominators: dict[str, float]
    per_position: dict[str, list]
    gradient_norm: float
    compute: NeuralComputeSnapshot
    objective_version: str
    auxiliary_coefficient: float
    timed_loss: float
    content_loss: float | None
    timed_compute: NeuralComputeSnapshot
    content_compute: NeuralComputeSnapshot | None


def make_optimizer(model, training):
    return torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        betas=training.betas,
        eps=training.epsilon,
        weight_decay=training.weight_decay,
        foreach=training.foreach,
        fused=training.fused,
    )


def _check_unroll(unroll):
    for boundary in (*unroll.observations, *unroll.boundaries):
        tensors = [
            boundary.context.workspace.latent,
            boundary.context.workspace.accumulators,
            boundary.context.memory_embeddings,
            boundary.context.hypothesis_features,
            *(getattr(boundary.parameters, item.name) for item in fields(boundary.parameters)),
        ]
        if not all(bool(torch.isfinite(t).all()) for t in tensors):
            raise TrainingError("Nonfinite internal training state")
    for step in unroll.steps:
        if bool((torch.isnan(step.crossings) | torch.isneginf(step.crossings)).any()):
            raise TrainingError("Nonfinite internal crossing offset")
        tensors = [
            step.post_context.workspace.latent,
            step.post_context.workspace.accumulators,
            step.post_context.hypothesis_features,
            step.post_context.memory_embeddings,
            *(getattr(step.post_parameters, item.name) for item in fields(step.post_parameters)),
        ]
        if not all(bool(torch.isfinite(t).all()) for t in tensors):
            raise TrainingError("Nonfinite internal post-jump state")


def _finite_state(value):
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise TrainingError("Nonfinite internal training state or loss")
    elif is_dataclass(value):
        for item in fields(value):
            _finite_state(getattr(value, item.name))
    elif isinstance(value, Mapping):
        for child in value.values():
            _finite_state(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _finite_state(child)


def _check_objective(objective):
    _check_unroll(objective.timed)
    _finite_state(objective.timed.final_context)
    _finite_state(objective.timed_loss)
    if objective.content is not None:
        for observation in objective.content.observations:
            _finite_state(observation.context)
            _finite_state(observation.parameters)
        for step in objective.content.steps:
            _finite_state(step.prediction_context)
            _finite_state(step.post_context)
            _finite_state(step.composition)
        _finite_state(objective.content.final_context)
        _finite_state(objective.content_loss)
    _finite_state(objective.total)


def train_one_step(model, optimizer, batch, config) -> StepResult:
    """Full trace graph, finite checks, backward, clip, then the AdamW update."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with NeuralComputeMeter(model) as meter:
        objective = training_objective(model, batch, config.training)
        _check_objective(objective)
        objective.total.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise TrainingError(f"Nonfinite gradient in {name}")
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.gradient_clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        for name, parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise TrainingError(f"Nonfinite parameter after update in {name}")
    branches = (
        [("", objective.timed_loss)]
        if objective.content is None
        else [("timed/", objective.timed_loss), ("content/", objective.content_loss)]
    )
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
        content_loss=None
        if objective.content_loss is None
        else float(objective.content_loss.total.detach()),
        timed_compute=objective.timed.compute,
        content_compute=None if objective.content is None else objective.content.compute,
    )


def stopping_reason(*, step, ceiling, patience, scheduled, gates_pass):
    if scheduled and gates_pass:
        return "component_gate"
    if patience >= 15:
        return "early_stopping"
    if step >= ceiling:
        return "step_ceiling"
    return None


def _json(value):
    if is_dataclass(value):
        return {item.name: _json(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _write_json(path, value):
    payload = canonical_json_bytes(_json(value))
    with archive_parent(path, error_factory=TrainingError) as (parent, name):
        _publish_bytes_at(parent, name, payload, replace=False)
    return hashlib.sha256(payload).hexdigest()


def _read_json(path):
    with archive_parent(path, error_factory=TrainingError) as (parent, name):
        return json.loads(
            read_archive_at(parent, name, max_bytes=32 * 1024 * 1024, error_factory=TrainingError)
        )


def _read_bound_json(path, digest):
    with archive_parent(path, error_factory=TrainingError) as (parent, name):
        payload = read_archive_at(
            parent, name, max_bytes=32 * 1024 * 1024, error_factory=TrainingError
        )
    if hashlib.sha256(payload).hexdigest() != digest:
        raise TrainingError("Training journal hash mismatch")
    return json.loads(payload)


def _verified_history(run_dir, progress):
    cursor = progress.training_journal_sha256
    if cursor is None:
        raise TrainingError("Run checkpoint has no bound training journal")
    seen, history = set(), []
    expected_step = progress.optimizer_step
    while cursor is not None:
        if (
            not isinstance(cursor, str)
            or not re.fullmatch(r"[0-9a-f]{64}", cursor)
            or cursor in seen
        ):
            raise TrainingError("Invalid training journal chain")
        seen.add(cursor)
        if len(seen) > 75001:
            raise TrainingError("Training journal chain exceeds step ceiling")
        record = _read_bound_json(run_dir / f"journal-{cursor}.json", cursor)
        if record["step"] != expected_step or not re.fullmatch(
            r"attempt-[a-zA-Z0-9_-]+", record["attempt"]
        ):
            raise TrainingError("Invalid training journal progress or attempt path")
        attempt = run_dir / record["attempt"]
        for item in reversed(record["steps"]):
            if item["path"] != f"step-{expected_step}.json":
                raise TrainingError("Noncontiguous training journal steps")
            logged = _read_bound_json(attempt / item["path"], item["sha256"])
            if logged["step"] != expected_step:
                raise TrainingError("Training journal step mismatch")
            expected_step -= 1
        if record["validation"] is not None:
            artifact = record["validation_file"]
            if artifact["path"] != f"validation-{record['step']}.json":
                raise TrainingError("Invalid validation journal path")
            _read_bound_json(attempt / artifact["path"], artifact["sha256"])
            history.append(record["validation"])
        cursor = record["prior"]
    if expected_step != 0:
        raise TrainingError("Incomplete training journal history")
    history.reverse()
    return history


def _manifest_corpus(config, path, source_commit):
    from silent_cascade.train.curriculum_data import MAX_COMPONENT_MANIFEST_BYTES, ComponentManifest

    with archive_parent(path, error_factory=TrainingError) as (parent, name):
        payload = read_archive_at(
            parent, name, max_bytes=MAX_COMPONENT_MANIFEST_BYTES, error_factory=TrainingError
        )
    manifest = ComponentManifest.model_validate_json(payload)
    training = config.config.training
    production = training.is_production
    if manifest.publication != ("production" if production else "debug"):
        raise TrainingError("Validation manifest publication does not match training profile")
    if manifest.source_revision != source_commit:
        raise TrainingError("Validation manifest source mismatch")
    if manifest.config_hash != config.sha256:
        raise TrainingError("Validation manifest configuration mismatch")
    if manifest.count != training.fixed_validation_episodes:
        raise TrainingError("Validation manifest count does not match training profile")
    for index, entry in enumerate(manifest.entries):
        key = entry.key
        if (
            key.stage != "one_hop"
            or key.split != ("validation" if production else "debug")
            or key.root_seed != training.validation_root_seed
            or key.public_id_seed != training.validation_public_id_seed
            or key.episode_index != index
        ):
            raise TrainingError("Validation manifest key/split/seed/stage/order mismatch")
    return manifest.build_corpus(config.config), hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    progress: TrainProgress
    stop_reason: str
    latest: CheckpointDescriptor
    selected: CheckpointDescriptor
    weights: CheckpointDescriptor
    validation_history: tuple[dict, ...]
    device: str
    evaluation_mode: str = "unassisted_content"
    selection_rule: str = "highest_chain_then_composition_then_earliest_step"
    checkpoint_cadence: str = "initial_scheduled_validation_and_final"


def run_training(
    config: ResolvedConfig[Phase3Config],
    *,
    validation_manifest: Path,
    run_dir: Path,
    source_commit: str,
    resume: Path | None = None,
    device: str | None = None,
) -> TrainingRunResult:
    """Train smoke/one-hop; only indexed completed-step archives consume counters.

    Journals are immutable per attempt. A crash can replay at most one validation
    interval (1000 production steps or 2 smoke steps), including pending commit.
    """
    if not isinstance(config, ResolvedConfig) or not isinstance(config.config, Phase3Config):
        raise TrainingError("Training requires a resolved Phase3Config")
    training = config.config.training
    stage = training.curriculum_stage
    if device is not None and (type(device) is not str or device not in ("cpu", "mps")):
        raise TrainingError("Training device must be cpu or mps")
    if device == "mps" and not torch.backends.mps.is_available():
        raise TrainingError("Requested MPS device is unavailable")
    corpus, manifest_hash = _manifest_corpus(config, validation_manifest, source_commit)
    device = device or next(
        device
        for device in config.config.runtime.device_preference
        if device == "cpu" or torch.backends.mps.is_available()
    )
    run_dir = run_dir.absolute()
    history = []
    prior = None
    if resume is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise TrainingError("Existing run directory requires explicit resume")
        run_dir.mkdir(parents=True, exist_ok=True)
        seed_all(training.model_seed)
        model = EventFlowModel(config.config.neural).to(device)
        optimizer = make_optimizer(model, training)
        progress = TrainProgress(
            optimizer_step=0,
            next_batch_counter=0,
            stage=stage,
            train_root_seed=training.train_root_seed,
            train_public_id_seed=training.train_public_id_seed,
            patience_counter=0,
            retained_checkpoints=(),
            validation_manifest_sha256=manifest_hash,
        )
    else:
        restored = load_training_checkpoint(
            resume,
            expected_config_sha256=config.sha256,
            expected_source_commit=source_commit,
            device=device,
        )
        progress = restored.progress
        if progress.stage not in ("one_hop", "smoke"):
            raise TrainingError("Phase boundary: higher curriculum training is not supported")
        if progress.stage != stage or progress.validation_manifest_sha256 != manifest_hash:
            raise TrainingError("Resume stage or validation manifest mismatch")
        index = load_checkpoint_index(run_dir)
        if index.latest != restored.descriptor:
            raise TrainingError("Resume requires the run directory's latest committed checkpoint")
        prior = restored.descriptor
        history = _verified_history(run_dir, progress)
        model, optimizer = restored.model, restored.optimizer
        restored.restore_rng()
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=run_dir))
    _write_json(
        attempt / "start.json", {"prior": prior, "next_batch_counter": progress.next_batch_counter}
    )
    segment_steps = []
    validation_file = None

    def checkpoint(validation=None):
        nonlocal prior, segment_steps, progress
        journal = {
            "schema_version": "phase3-training-journal-v1",
            "step": progress.optimizer_step,
            "prior": progress.training_journal_sha256,
            "attempt": attempt.name,
            "steps": segment_steps,
            "validation": validation,
            "validation_file": validation_file,
        }
        digest = hashlib.sha256(canonical_json_bytes(journal)).hexdigest()
        _write_json(run_dir / f"journal-{digest}.json", journal)
        progress = TrainProgress.model_validate(
            {**progress.model_dump(), "training_journal_sha256": digest}
        )
        descriptor = save_training_checkpoint(
            run_dir, model, optimizer, progress, config=config, source_commit=source_commit
        )
        update_checkpoint_index(run_dir, descriptor)
        prior, segment_steps = descriptor, []
        prune_training_checkpoints(run_dir)

    if prior is None:
        checkpoint()
    best = max(((row["chain"], row["composition"], -row["step"]) for row in history), default=None)
    reason = stopping_reason(
        step=progress.optimizer_step,
        ceiling=training.max_steps,
        patience=progress.patience_counter,
        scheduled=bool(history and history[-1]["step"] == progress.optimizer_step),
        gates_pass=bool(history and history[-1]["gates_pass"]),
    )
    while reason is None:
        batch = None
        try:
            batch = next_training_batch(
                config.config, stage="one_hop", batch_counter=progress.next_batch_counter
            ).to(device)
            result = train_one_step(model, optimizer, batch, config.config)
            step = progress.optimizer_step + 1
            step_name = f"step-{step}.json"
            digest = _write_json(
                attempt / step_name,
                {
                    "step": step,
                    "batch_keys": batch.example_keys,
                    "example_hashes": batch.example_hashes,
                    "result": result,
                    "prior": prior,
                },
            )
            segment_steps.append({"path": step_name, "sha256": digest})
            progress = TrainProgress.model_validate(
                {
                    **progress.model_dump(),
                    "optimizer_step": step,
                    "next_batch_counter": batch.next_batch_counter,
                    "validation_metric": None,
                    "validation_composition_metric": None,
                }
            )
            scheduled = step % training.validation_every_steps == 0
            validation = None
            if scheduled:
                evaluation = evaluate_components(model, corpus, batch_size=training.batch_size)
                metrics = evaluation.metrics
                ranking = (
                    metrics.complete_chain_accuracy,
                    metrics.required_composition_accuracy,
                    -step,
                )
                improved = best is None or ranking > best
                if improved:
                    best = ranking
                validation = {
                    "step": step,
                    "chain": metrics.complete_chain_accuracy,
                    "composition": metrics.required_composition_accuracy,
                    "recall": metrics.required_recall_accuracy,
                    "gates_pass": metrics.gates_pass,
                }
                digest = _write_json(
                    attempt / f"validation-{step}.json",
                    {
                        "predictions": evaluation.predictions.to_primitive(),
                        "metrics": metrics.to_primitive(),
                        "compute": evaluation.compute,
                        "targets": corpus.targets,
                    },
                )
                validation_file = {"path": f"validation-{step}.json", "sha256": digest}
                history.append(validation)
                progress = TrainProgress.model_validate(
                    {
                        **progress.model_dump(),
                        "best_metric": best[0],
                        "best_step": -best[2],
                        "validation_metric": metrics.complete_chain_accuracy,
                        "validation_composition_metric": metrics.required_composition_accuracy,
                        "patience_counter": 0 if improved else progress.patience_counter + 1,
                    }
                )
            reason = stopping_reason(
                step=step,
                ceiling=training.max_steps,
                patience=progress.patience_counter,
                scheduled=scheduled,
                gates_pass=bool(validation and validation["gates_pass"]),
            )
            if scheduled or reason is not None:
                checkpoint(validation)
        except Exception as error:
            failure = {
                "batch_counter": progress.next_batch_counter
                if batch is None
                else batch.next_batch_counter - 1,
                "batch_keys": () if batch is None else batch.example_keys,
                "example_hashes": () if batch is None else batch.example_hashes,
                "prior_checkpoint": prior,
                "exception_type": type(error).__name__,
                "message": str(error),
            }
            _write_json(attempt / "failure.json", failure)
            raise TrainingError(
                str(error),
                context={
                    "failure_path": str(attempt / "failure.json"),
                    "prior_checkpoint": prior.relative_path,
                    "batch_counter": failure["batch_counter"],
                },
            ) from error
    history = _verified_history(run_dir, progress)
    index = load_checkpoint_index(run_dir)
    if not index.best:
        raise TrainingError("Training stopped without a scheduled validation checkpoint")
    selected = index.best[0]
    update_checkpoint_index(run_dir, index.latest, protected=(selected,))
    selected_restore = load_training_checkpoint(
        run_dir / selected.relative_path,
        expected_config_sha256=config.sha256,
        expected_source_commit=source_commit,
        device=device,
    )
    weights = export_weights(
        run_dir, selected_restore.model, config=config, source_commit=source_commit
    )
    result = TrainingRunResult(
        progress, reason, index.latest, selected, weights, tuple(history), device
    )
    _write_json(attempt / "result.json", result)
    return result


@dataclass(frozen=True, slots=True)
class TinyOverfitResult:
    episodes: int
    steps: int
    complete_chain_accuracy: float
    history: tuple[tuple[int, float, float, float, float], ...]
    evidence_kind: str = "overfit_engineering_regression"


def run_tiny_overfit(config, *, device: str) -> TinyOverfitResult:
    """Actual fixed debug examples, fixed AdamW, and a separate 1000-step limit."""
    torch.manual_seed(config.training.model_seed)
    debug = NeuralModelConfig(
        architecture_profile="debug",
        record_hidden_dim=96,
        external_hidden_dim=64,
        query_dim=64,
        controller_hidden_dim=128,
        jump_hidden_dim=128,
        head_hidden_dim=64,
    )
    model = EventFlowModel(debug).to(device)
    optimizer = make_optimizer(model, config.training)
    examples = tuple(
        make_curriculum_example(
            config, CurriculumKey("ofd-one-hop-v1", "debug", 311, 331, i, "one_hop")
        )
        for i in range(64)
    )
    batch = pack_training_examples(examples).to(device)
    corpus = ComponentCorpus(
        tuple(e.public for e in examples),
        tuple(component_target(build_teacher_trace(e)) for e in examples),
    )
    history = []
    for step in range(1, 1001):
        result = train_one_step(model, optimizer, batch, config)
        if step % 25 == 0:
            metrics = evaluate_components(model, corpus, batch_size=64).metrics
            history.append(
                (
                    step,
                    result.loss,
                    metrics.required_recall_accuracy,
                    metrics.required_composition_accuracy,
                    metrics.complete_chain_accuracy,
                )
            )
            if metrics.complete_chain_accuracy >= 0.99:
                break
    return TinyOverfitResult(64, step, metrics.complete_chain_accuracy, tuple(history))
