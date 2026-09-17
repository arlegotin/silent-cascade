"""The separately budgeted, fixed64 training-exposed engineering diagnostic."""

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from silent_cascade.config import ResolvedConfig
from silent_cascade.env.pilot import curriculum_to_bundle
from silent_cascade.env.reward import score_actions
from silent_cascade.errors import DynamicsError
from silent_cascade.eval.artifacts import EpisodeBinding
from silent_cascade.eval.compute import NeuralComputeMeter, aggregate_runtime_compute
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.neural import NeuralEventFlowAgent, NeuralModelIdentity
from silent_cascade.eventflow.neural_weights import save_neural_weights
from silent_cascade.eventflow.state import ComputeCounters
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.runtime_diagnostics import runtime_diagnostic_identity
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.rng import seed_all
from silent_cascade.train.batches import pack_training_examples
from silent_cascade.train.checkpoints import _device, _environment
from silent_cascade.train.component_eval import evaluate_components
from silent_cascade.train.curriculum_data import (
    CURRICULUM_VERSION,
    ComponentCorpus,
    CurriculumKey,
    make_curriculum_example,
)
from silent_cascade.train.objective import training_objective
from silent_cascade.train.pilot_config import Phase4Config
from silent_cascade.train.pilot_provenance import authenticate_pilot_source
from silent_cascade.train.pilot_trainer import pilot_train_one_step, publish_json
from silent_cascade.train.traces import build_teacher_trace, component_target
from silent_cascade.train.trainer import _check_objective, _json, make_optimizer


def fixed_overfit_examples(config):
    if config.config.pilot.profile != "phase4_smoke":
        raise ValueError("fixed64 diagnostic requires the closed debug base configuration")
    return tuple(
        make_curriculum_example(
            config.config, CurriculumKey(CURRICULUM_VERSION, "debug", 449, 457, i, "one_hop")
        )
        for i in range(64)
    )


def evaluate_overfit_boundary(
    model,
    examples,
    *,
    config,
    device,
    output_dir,
    source_commit,
    update,
    ordered_data_sha256,
    weights_sha256,
):
    """Use the existing engine to completion; this corpus is not a pilot manifest."""
    identity = NeuralModelIdentity.from_model(model, source_revision=source_commit)
    corpus = ComponentCorpus(
        tuple(e.public for e in examples),
        tuple(component_target(build_teacher_trace(e)) for e in examples),
    )
    component = evaluate_components(model, corpus, batch_size=64)
    batch = pack_training_examples(examples, next_batch_counter=1).to(device)
    model.zero_grad(set_to_none=True)
    with NeuralComputeMeter(model) as loss_meter:
        objective = training_objective(model, batch, config.config.pilot)
        _check_objective(objective)
        objective.total.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float("inf"), error_if_nonfinite=True
        )
    losses = {
        "total": float(objective.total.detach()),
        "gradient_norm": float(norm),
        "compute": _json(loss_meter.snapshot()),
    }
    for prefix, loss in (("timed", objective.timed_loss), ("content", objective.content_loss)):
        losses[prefix] = {
            "total": float(loss.total.detach()),
            "terms": {k: float(v.detach()) for k, v in loss.terms.items()},
            "subterms": {k: float(v.detach()) for k, v in loss.subterms.items()},
            "per_position": {k: v.detach().cpu().tolist() for k, v in loss.per_position.items()},
        }
    model.zero_grad(set_to_none=True)
    inference = copy.deepcopy(model).to(device)
    agent = NeuralEventFlowAgent(inference, identity=identity, device=device)
    engine = EventEngine(
        config.config.event_flow,
        crash_root=output_dir / "crashes",
        source_revision=source_commit,
        experiment_config_canonical_json=config.canonical_json.decode(),
    )
    outcomes = []
    for ordinal, example in enumerate(examples):
        bundle = curriculum_to_bundle(example, config=config.config)
        session, error = None, None
        before = engine.operational_compute_snapshot()
        with NeuralComputeMeter(inference) as meter, torch.no_grad():
            try:
                session = engine.start_episode(bundle, agent)
                engine.run_until(session, agent, bundle.truth.private_terminal.timestamp)
            except DynamicsError as caught:
                error = asdict(runtime_diagnostic_identity(caught))
        actions = session.state.core.actions if session else ()
        score = score_actions(bundle.truth, actions)
        trace = session.trace.snapshot() if session else None
        compute = aggregate_runtime_compute(
            session.state.core.counters if session else ComputeCounters(),
            (meter.snapshot(),),
            entity_parameters=inference.record_encoder.entity_embedding.weight.numel(),
            causal_flow_evaluations=engine.operational_compute_snapshot().causal_flow_evaluations
            - before.causal_flow_evaluations,
        )
        row = {
            "ordinal": ordinal,
            "episode": EpisodeBinding.from_bundle(bundle).model_dump(mode="json"),
            "truth": asdict(bundle.truth),
            "actions": [asdict(a) for a in actions],
            "score": asdict(score),
            "error": error,
            "timed_success": error is None and score.timed_success,
            "causal_trace_sha256": trace.sha256 if trace else None,
            "causal_events": [e.to_payload() for e in trace.events] if trace else [],
            "compute": compute.model_dump(mode="json"),
            "checkpoint_sha256": weights_sha256,
            "model_identity": asdict(identity),
            "ordered_data_sha256": ordered_data_sha256,
            "source_commit": source_commit,
            "config_sha256": config.sha256,
            "model_seed": 11,
        }
        publish_json(output_dir / f"episode-{ordinal:03d}.json", row)
        outcomes.append(row)
    metrics = {
        "episode_count": len(outcomes),
        "timed_success_count": sum(r["timed_success"] for r in outcomes),
        "false_action_count": sum(
            not r["score"]["is_positive"] and bool(r["actions"]) for r in outcomes
        ),
        "error_count": sum(r["error"] is not None for r in outcomes),
        "foundation_model_calls": 0,
    }
    passed = (
        len(outcomes) == 64
        and component.metrics.correct_chain_count == 64
        and metrics["timed_success_count"] >= 58
        and metrics["false_action_count"] <= 3
        and metrics["error_count"] == 0
    )
    return {
        "update": update,
        "model_identity": asdict(identity),
        "weights_sha256": weights_sha256,
        "ordered_data_sha256": ordered_data_sha256,
        "component": _json(component),
        "autonomous": metrics,
        "outcomes": outcomes,
        "losses": losses,
        "passed": passed,
        "gate_eligible": False,
    }


@dataclass(frozen=True)
class PilotOverfitResult:
    evidence_kind: Literal["phase4_overfit_engineering_diagnostic"]
    recipe: str
    episode_count: int
    optimizer_steps: int
    passed: bool
    gate_eligible: Literal[False]
    workload: dict
    boundaries: tuple[dict, ...]
    artifact_hashes: dict[str, str]
    source_commit: str
    config_sha256: str
    model_identity: NeuralModelIdentity


def run_pilot_tiny_overfit(
    config: ResolvedConfig[Phase4Config], *, device: str, output_dir: Path, source_commit: str
) -> PilotOverfitResult:
    root = Path(__file__).resolve().parents[3]
    source = authenticate_pilot_source(repo_root=root, source_commit=source_commit, config=config)
    _device(device)
    if output_dir.exists():
        raise ValueError("diagnostic output must be fresh; attempts cannot be silently restarted")
    output_dir.mkdir(parents=True)
    examples = fixed_overfit_examples(config)
    bundles = tuple(curriculum_to_bundle(e, config=config.config) for e in examples)
    ordered = [
        {
            "key": asdict(e.key),
            "example_sha256": e.example_hash,
            "public_sha256": e.public_hash,
            "target_sha256": e.target_hash,
            "projection": EpisodeBinding.from_bundle(b),
        }
        for e, b in zip(examples, bundles, strict=True)
    ]
    workload = {
        "recipe": "phase4-overfit-debug-v1",
        "model_seed": 11,
        "episode_count": 64,
        "batch_size": 64,
        "maximum_updates": 1000,
        "diagnostic_interval": 25,
        "split": "debug",
        "stage": "one_hop",
        "root_seed": 449,
        "public_id_seed": 457,
        "indices": list(range(64)),
        "positive": 32,
        "safe": 16,
        "disconnected": 16,
        "training_exposed": True,
        "gate_eligible": False,
        "base_config_sha256": config.sha256,
        "base_config_json": config.canonical_json.decode(),
        "source": source,
        "environment": _environment(device).model_dump(),
        "ordered_data": ordered,
    }
    artifacts = {"workload.json": publish_json(output_dir / "workload.json", workload)}
    data_hash = sha256_bytes(
        canonical_json_bytes(
            {
                "examples": [
                    {**e, "projection": e["projection"].model_dump(mode="json")} for e in ordered
                ]
            }
        )
    )
    seed_all(11)
    model = EventFlowModel(config.config.neural).to(device)
    optimizer = make_optimizer(model, config.config.pilot)
    batch = pack_training_examples(examples, next_batch_counter=1).to(device)
    boundaries, passed, step = [], False, 0
    last_update = None
    try:
        for step in range(1001):
            if step:
                last_update = pilot_train_one_step(model, optimizer, batch, config.config)
                name = f"update-{step:04d}.json"
                artifacts[name] = publish_json(
                    output_dir / name,
                    {
                        "update": step,
                        "batch_sha256": sha256_bytes(
                            canonical_json_bytes({"examples": batch.example_hashes})
                        ),
                        "ordered_data_sha256": data_hash,
                        "source_sha256": source.source_sha256,
                        "config_sha256": config.sha256,
                        "model_seed": 11,
                        "parameter_count": sum(p.numel() for p in model.parameters()),
                        "result": _json(last_update),
                    },
                )
            if step % 25:
                continue
            current_source = authenticate_pilot_source(
                repo_root=root, source_commit=source_commit, config=config
            )
            if current_source != source:
                raise ValueError("diagnostic executing source changed")
            identity = NeuralModelIdentity.from_model(model, source_revision=source_commit)
            weights_name = f"weights-{step:04d}.safetensors"
            weights_sha = save_neural_weights(
                output_dir / weights_name, model=model, identity=identity
            )
            artifacts[weights_name] = weights_sha
            autonomous_dir = output_dir / f"autonomous-{step:04d}"
            boundary = evaluate_overfit_boundary(
                model,
                examples,
                config=config,
                device=device,
                output_dir=autonomous_dir,
                source_commit=source_commit,
                update=step,
                ordered_data_sha256=data_hash,
                weights_sha256=weights_sha,
            )
            boundary["last_update"] = _json(last_update) if last_update else None
            passed = boundary["passed"]
            name = f"boundary-{step:04d}.json"
            artifacts[name] = publish_json(output_dir / name, boundary)
            artifacts.update(
                {
                    str(p.relative_to(output_dir)): sha256_bytes(p.read_bytes())
                    for p in sorted(autonomous_dir.rglob("*"))
                    if p.is_file()
                }
            )
            boundaries.append(
                {
                    "step": step,
                    "chain_correct": boundary["component"]["metrics"]["correct_chain_count"],
                    "timed_success": boundary["autonomous"]["timed_success_count"],
                    "false_actions": boundary["autonomous"]["false_action_count"],
                    "errors": boundary["autonomous"]["error_count"],
                    "passed": passed,
                    "artifact": name,
                    "sha256": artifacts[name],
                }
            )
            print(json.dumps(boundaries[-1]), flush=True)
            if passed:
                break
    except Exception as error:
        publish_json(
            output_dir / "failure.json",
            {
                "update": step,
                "type": type(error).__name__,
                "message": str(error),
                "source": source,
                "boundaries": boundaries,
                "gate_eligible": False,
            },
        )
        raise
    result = PilotOverfitResult(
        "phase4_overfit_engineering_diagnostic",
        "phase4-overfit-debug-v1",
        64,
        step,
        passed,
        False,
        workload,
        tuple(boundaries),
        artifacts,
        source_commit,
        config.sha256,
        identity,
    )
    publish_json(output_dir / "result.json", _json(result))
    return result
