"""Private, offline action readouts: historical, teacher and actual autonomous ACT.

No optimizer is constructed. Original file authentication precedes every readout;
teacher labels never cross the autonomous evaluator's public callback boundary.
"""

import copy
import gzip
import json
import math
import shutil
import subprocess
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import torch

from silent_cascade.env.pilot import curriculum_to_bundle
from silent_cascade.eval.artifacts import (
    EpisodeBinding,
    EvaluationIdentity,
    read_evaluation_artifact,
)
from silent_cascade.eval.runner import evaluate_episodes
from silent_cascade.eventflow.flow import state_at
from silent_cascade.eventflow.guards import crossing_offset_host
from silent_cascade.eventflow.neural import NeuralModelIdentity
from silent_cascade.eventflow.neural_context import continuous_from_workspace
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.io import atomic_create_bytes
from silent_cascade.logging.neural_trace import _host, _trajectory_state
from silent_cascade.logging.trace import SegmentSummary, _state_summary, tensor_sha256
from silent_cascade.models.dynamics import crossings_batch
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.models.losses import event_flow_loss
from silent_cascade.train.batches import pack_training_examples
from silent_cascade.train.checkpoints import load_weight_bundle
from silent_cascade.train.component_eval import predict_components
from silent_cascade.train.curriculum_data import (
    MAX_COMPONENT_MANIFEST_BYTES,
    ComponentCorpus,
    ComponentManifest,
    CurriculumExample,
    load_component_manifest,
    make_curriculum_example,
)
from silent_cascade.train.pilot_config import Phase4Config
from silent_cascade.train.traces import build_teacher_trace, component_target
from silent_cascade.train.unroll import teacher_forced_unroll

HISTORICAL_WEIGHTS_SHA256 = "ba6f929f1e5bfa9cf00b8c31b948f35582dc6662a23203db02c9e5f17ee93e3c"
HISTORICAL_MANIFEST_SHA256 = "2e1d367c6ac072d682afd241f280b5ff729491ca523c2106a4bb176be308d6d2"
CONTEXTS = (
    "historical_immediate_five_way",
    "teacher_context_diagnostic",
    "autonomous_legal_shield",
    "autonomous_act_raw_five_way",
)


@dataclass(frozen=True)
class ActionContextSummary:
    label: str
    episode_count: int
    positive_count: int
    action_count: int
    missing_action_count: int
    correct_count: int
    positive_correct_count: int
    timed_success_count: int | None
    error_count: int
    # True classes 0..4; predicted columns 0..4 and 5 = missing ACT/readout.
    confusion: tuple[tuple[int, ...], ...]
    by_variant: dict


@dataclass(frozen=True)
class ActionDiagnosticReport:
    provenance: dict
    immediate: ActionContextSummary
    teacher_timed: ActionContextSummary
    autonomous: ActionContextSummary
    autonomous_raw: ActionContextSummary
    gradient_diagnostics: dict
    loss_diagnostics: dict
    detailed_public_ids: tuple[str, ...]
    observer_prediction_equivalence: bool
    historical_reproduction: bool
    artifact_hashes: dict
    output_dir: Path
    elapsed_seconds: float
    interpretation: dict
    foundation_model_calls: int = 0
    gate_eligible: bool = False
    diagnostic_only: bool = True


def _authenticate_files(
    *,
    source_weights_path,
    expected_source_weights_sha256,
    source_manifest_path,
    expected_source_manifest_sha256,
    device,
):
    """One original archive load; original-config regeneration, never Phase4 forgery."""
    restored = load_weight_bundle(
        source_weights_path, expected_sha256=expected_source_weights_sha256, device=device
    )
    raw = read_evaluation_artifact(source_manifest_path, max_bytes=MAX_COMPONENT_MANIFEST_BYTES)
    if sha256_bytes(raw) != expected_source_manifest_sha256:
        raise ValueError("original manifest hash mismatch")
    manifest = load_component_manifest(source_manifest_path, restored.config.config)
    if manifest != ComponentManifest.model_validate_json(raw):
        raise ValueError("original manifest changed between authenticated and loaded snapshots")
    # The closed loader validates regeneration; retain the corresponding private
    # examples here because its public ComponentCorpus intentionally drops them.
    examples = tuple(
        make_curriculum_example(restored.config.config, e.key) for e in manifest.entries
    )
    if (
        sha256_bytes(
            read_evaluation_artifact(source_manifest_path, max_bytes=MAX_COMPONENT_MANIFEST_BYTES)
        )
        != expected_source_manifest_sha256
    ):
        raise ValueError("original manifest hash changed during authentication")
    return restored, manifest, examples


def diagnose_actions(
    model: EventFlowModel,
    *,
    examples: Sequence[CurriculumExample],
    identity: NeuralModelIdentity,
    config: Phase4Config,
    source_weights_path: Path,
    expected_source_weights_sha256: str,
    source_manifest_path: Path,
    expected_source_manifest_sha256: str,
    output_dir: Path,
) -> ActionDiagnosticReport:
    """Authenticate actual original files and all supplied live inputs before work."""
    restored, manifest, original = _authenticate_files(
        source_weights_path=source_weights_path,
        expected_source_weights_sha256=expected_source_weights_sha256,
        source_manifest_path=source_manifest_path,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        device=str(next(model.parameters()).device),
    )
    authenticated = NeuralModelIdentity.from_model(
        restored.model, source_revision=restored.descriptor.source_commit
    )
    if (
        type(model) is not EventFlowModel
        or identity != authenticated
        or (
            NeuralModelIdentity.from_model(model, source_revision=identity.source_revision)
            != authenticated
        )
    ):
        raise ValueError("live model or identity differs from authenticated original weights")
    if tuple(examples) != original:
        raise ValueError("ordered examples differ from authenticated original corpus")
    return _diagnose_authenticated(
        restored=restored,
        manifest=manifest,
        examples=original,
        config=config,
        manifest_sha256=expected_source_manifest_sha256,
        output_dir=output_dir,
    )


def _execution_identity():
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    paths = sorted(
        (
            *root.glob("src/silent_cascade/**/*.py"),
            *root.glob("configs/**/*.yaml"),
            root / "scripts/diagnose_phase4_actions.py",
            root / "pyproject.toml",
            root / "uv.lock",
        )
    )
    hashes = {str(p.relative_to(root)): sha256_bytes(p.read_bytes()) for p in paths}
    dirty = bool(
        subprocess.check_output(
            [
                "git",
                "status",
                "--porcelain",
                "--",
                "src",
                "scripts",
                "configs",
                "pyproject.toml",
                "uv.lock",
            ],
            cwd=root,
        )
    )
    return {
        "source_revision": revision,
        "source_files_sha256": hashes,
        "source_tree_sha256": sha256_bytes(canonical_json_bytes(hashes)),
        "source_dirty": dirty,
    }


def _context_snapshot(context, row, *, detailed=False):
    selected = context._gather(torch.tensor([row], device=context.device))
    value = _host(selected)
    hypothesis = context.hypothesis_features[row, 1:5].detach()
    result = {
        "state_sha256": sha256_bytes(canonical_json_bytes(value)),
        "state_hash_schema": "action-diagnostic-model-context-v1",
        "continuous_state_sha256": _state_summary(continuous_from_workspace(selected.workspace))[1],
        "context_hypothesis_class": int(hypothesis.argmax()) if bool(hypothesis.sum()) else None,
    }
    if detailed:
        result["context"] = value
    return result


def _guard(parameters, context, row, *, crossing=None):
    values = (
        crossings_batch(
            context.workspace.accumulators, parameters.guard_targets, parameters.guard_rates
        )
        if crossing is None
        else crossing
    )
    return {
        "guard_targets": parameters.guard_targets[row].detach().cpu().tolist(),
        "guard_rates": parameters.guard_rates[row].detach().cpu().tolist(),
        "guard_crossings": [
            float(x) if math.isfinite(float(x)) else None for x in values[row].detach().cpu()
        ],
        "crossing_null_means": "no_finite_crossing",
        "segment_sha256": sha256_bytes(
            canonical_json_bytes(
                {
                    "origin": tensor_sha256(context.workspace.latent[row]),
                    "accumulators": tensor_sha256(context.workspace.accumulators[row]),
                    "parameters": {
                        f.name: tensor_sha256(getattr(parameters, f.name)[row])
                        for f in fields(parameters)
                    },
                }
            )
        ),
    }


def _readout(example, kind, logits, *, act_occurred=False, **context):
    true_class = example.solution.hazard_type if example.variant.value == "positive" else 4
    if logits is not None:
        if len(logits) != 5 or not all(math.isfinite(x) for x in logits):
            raise ValueError("nonfinite or malformed action logits")
        raw = max(range(5), key=lambda i: logits[i])
        legal = max(range(4), key=lambda i: logits[i])
        margin = logits[true_class] - max(x for i, x in enumerate(logits) if i != true_class)
        ordered = sorted(logits, reverse=True)
        top_margin = ordered[0] - ordered[1]
        maximum = max(logits)
        loss = math.log(sum(math.exp(x - maximum) for x in logits)) + maximum - logits[true_class]
    else:
        raw = legal = margin = top_margin = loss = None
    return {
        "public_id": example.init.episode_public_id,
        "example_sha256": example.example_hash,
        "variant": example.variant.value,
        "context_kind": kind,
        "raw_five_way_choice": raw,
        "legal_act_shield_choice": legal,
        "logits": logits,
        "true_class": true_class,
        "act_occurred": act_occurred,
        "true_class_margin": margin,
        "top_two_margin": top_margin,
        "cross_entropy": loss,
        "segment_sha256": None,
        "state_sha256": None,
        "predicted_hypothesis_class": None,
        "context_hypothesis_class": None,
        "guard_targets": None,
        "guard_rates": None,
        "guard_crossings": None,
        "elapsed_since_activation": None,
        "timestamp": None,
        **context,
    }


def _summarize(rows, kind):
    selected = [r for r in rows if r["context_kind"] == kind]
    confusion = [[0] * 6 for _ in range(5)]
    strata = {}
    correct = positives = positive_correct = actions = missing = successes = errors = 0
    for row in selected:
        choice = (
            row["legal_act_shield_choice"] if kind == CONTEXTS[2] else row["raw_five_way_choice"]
        )
        positive = row["true_class"] < 4
        # Raw-at-ACT never treats missing ACT as a correct abstention. Actual
        # autonomous negatives are scored by the real no-action rule separately.
        right = (choice == row["true_class"]) if choice is not None else False
        if kind == CONTEXTS[2] and not positive:
            right = not row["act_occurred"] and row.get("error") is None
        if row.get("error") is not None:
            right = False
        confusion[row["true_class"]][choice if choice is not None else 5] += 1
        correct += right
        positives += positive
        positive_correct += right and positive
        actions += row["act_occurred"]
        missing += positive and choice is None
        successes += bool(row.get("timed_success", False))
        errors += row.get("error") is not None
        item = strata.setdefault(
            row["variant"], {"count": 0, "correct": 0, "confusion": [[0] * 6 for _ in range(5)]}
        )
        item["count"] += 1
        item["correct"] += int(right)
        item["confusion"][row["true_class"]][choice if choice is not None else 5] += 1
    return ActionContextSummary(
        kind,
        len(selected),
        positives,
        actions,
        missing,
        correct,
        positive_correct,
        successes if kind == CONTEXTS[2] else None,
        errors,
        tuple(tuple(r) for r in confusion),
        strata,
    )


def _immediate(model, examples, output_dir, detailed_count):
    corpus = ComponentCorpus(
        tuple(e.public for e in examples),
        tuple(component_target(build_teacher_trace(e)) for e in examples),
    )
    rows = []
    for start in range(0, len(examples), 128):
        count = min(128, len(examples) - start)
        public = corpus.public_batch(start, count).to(str(next(model.parameters()).device))
        snapshots = []

        def observe(module, inputs, output, *, snapshots=snapshots, start=start, count=count):
            snapshots.extend(
                _context_snapshot(inputs[0], i, detailed=start + i < detailed_count)
                for i in range(count)
            )
            # An observer must return None: replacing a prediction is forbidden.

        hook = model.action_heads.register_forward_hook(observe)
        try:
            predicted = predict_components(model, public)
        finally:
            hook.remove()
        plain = predict_components(model, public)
        if plain != predicted or len(snapshots) != count:
            raise ValueError("historical observer changed predictions or missed a readout")
        for i, (example, prediction, snapshot) in enumerate(
            zip(examples[start : start + count], predicted.rows, snapshots, strict=True)
        ):
            detail = snapshot.pop("context", None)
            if detail is not None:
                atomic_create_bytes(
                    output_dir / f"details/{start + i:05d}.immediate.json.gz",
                    gzip.compress(
                        canonical_json_bytes({"context": detail, "prediction": asdict(prediction)}),
                        mtime=0,
                    ),
                )
            rows.append(
                _readout(
                    example,
                    CONTEXTS[0],
                    list(prediction.action_logits),
                    **snapshot,
                    elapsed_since_activation=0.0,
                    timestamp=example.events[-1].timestamp,
                    readout_position="immediate_post_learned_content",
                    guard_context="not_installed_in_historical_recipe",
                    stop_reason=prediction.stop_reason,
                    atomic_transitions=prediction.atomic_transitions,
                    predicted_historical_terminal_class=prediction.terminal_class,
                    predicted_hypothesis_class=(
                        prediction.terminal_class if prediction.terminal_class < 4 else None
                    ),
                )
            )
    return rows


def _teacher(model, examples, config, output_dir, detailed_count):
    rows, numerators, denominators = [], Counter(), Counter()
    for start in range(0, len(examples), 128):
        group = tuple(examples[start : start + 128])
        batch = pack_training_examples(group).to(str(next(model.parameters()).device))
        with torch.no_grad():
            result = teacher_forced_unroll(model, batch)
            loss = event_flow_loss(result.loss_inputs(), config.pilot.loss_weights)
        numerators.update({k: float(v) for k, v in loss.numerators.items()})
        denominators.update({k: int(v) for k, v in loss.denominators.items()})
        for row, example in enumerate(group):
            columns = (
                (
                    batch.targets.real_step_mask[row]
                    & ((batch.targets.kind[row] == 2) | (batch.targets.action_class[row] == 4))
                )
                .nonzero()
                .flatten()
                .tolist()
            )
            if len(columns) != 1:
                raise ValueError(
                    "teacher action inventory must have exactly one real readout per example"
                )
            col = columns[0]
            step = result.steps[col]
            positive = int(batch.targets.kind[row, col]) == 2
            context = step.prediction_context if positive else step.post_context
            boundary = result.boundaries[col]
            parameters = boundary.parameters if positive else step.post_parameters
            guard_context = boundary.context if positive else step.post_context
            timestamp = float(batch.teacher_times[row, col])
            snapshot = _context_snapshot(context, row)
            # Teacher state contains disclosed registers; diagnose the actual
            # hazard head separately, only where that head was supervised.
            hazard_columns = batch.targets.validity["hazard_type"][row, :col].nonzero().flatten()
            snapshot["predicted_hypothesis_class"] = (
                int(result.steps[int(hazard_columns[-1])].composition.hazard_logits[row].argmax())
                if positive and hazard_columns.numel()
                else None
            )
            if start + row < detailed_count:
                detail = [
                    {
                        "prediction_context": _context_snapshot(
                            s.prediction_context, row, detailed=True
                        )["context"],
                        "post_context": _context_snapshot(s.post_context, row, detailed=True)[
                            "context"
                        ],
                        "post_parameters": {
                            f.name: _host(getattr(s.post_parameters, f.name)[row])
                            for f in fields(s.post_parameters)
                        },
                    }
                    for i, s in enumerate(result.steps)
                    if bool(batch.targets.real_step_mask[row, i])
                ]
                atomic_create_bytes(
                    output_dir / f"details/{start + row:05d}.teacher.json.gz",
                    gzip.compress(canonical_json_bytes({"steps": detail}), mtime=0),
                )
            rows.append(
                _readout(
                    example,
                    CONTEXTS[1],
                    step.action.class_logits[row].cpu().tolist(),
                    **snapshot,
                    **_guard(
                        parameters,
                        guard_context,
                        row,
                        crossing=step.crossings if positive else None,
                    ),
                    timestamp=timestamp,
                    elapsed_since_activation=timestamp - example.events[-1].timestamp,
                    readout_position="flowed_teacher_act" if positive else "post_composition",
                    guard_context="teacher_segment",
                    teacher_act_selected=positive,
                )
            )
    return rows, {
        k: {
            "numerator": numerators[k],
            "denominator": denominators[k],
            "mean": numerators[k] / denominators[k] if denominators[k] else None,
        }
        for k in numerators
    }


def _gradients(model, examples, config):
    # Complete timed recurrent graph on the predeclared first128, no optimization.
    model = copy.deepcopy(model).requires_grad_(True)
    batch = pack_training_examples(tuple(examples[:128])).to(str(next(model.parameters()).device))
    result = teacher_forced_unroll(model, batch)
    loss = event_flow_loss(result.loss_inputs(), config.pilot.loss_weights)
    named = tuple(model.named_parameters())
    gradients = torch.autograd.grad(loss.terms["action"], [p for _, p in named], allow_unused=True)
    norms = {}
    for (name, _), gradient in zip(named, gradients, strict=True):
        if gradient is not None and not bool(torch.isfinite(gradient).all()):
            raise ValueError("nonfinite complete-timed-unroll action gradient")
        norms[name] = (
            float(gradient.detach().cpu().double().square().sum()) if gradient is not None else 0.0
        )
    return {
        "sample_count": len(examples[:128]),
        "loss": float(loss.terms["action"].detach()),
        "action_head_l2": math.sqrt(
            sum(v for k, v in norms.items() if k.startswith("action_heads."))
        ),
        "recurrent_l2": math.sqrt(
            sum(v for k, v in norms.items() if not k.startswith("action_heads."))
        ),
        "parameter_l2": {k: math.sqrt(v) for k, v in norms.items()},
        "all_finite": True,
        "optimizer_steps": 0,
    }


def _autonomous(model, examples, bundles, identity, config, output_dir, detailed_count):
    captures = {}
    ordinal = 0

    def episode_stream():
        nonlocal ordinal
        for index, bundle in enumerate(bundles):
            ordinal = index
            yield bundle

    def observe(module, inputs, output):
        captures.setdefault(ordinal, []).append(
            _context_snapshot(inputs[0], 0, detailed=ordinal < detailed_count)
        )

    hook = model.action_heads.register_forward_hook(observe)
    try:
        evaluation = evaluate_episodes(
            model,
            identity=identity,
            config=config,
            episodes=episode_stream(),
            output_dir=output_dir / "autonomous",
            device=str(next(model.parameters()).device),
        )
    finally:
        hook.remove()
    rows = []
    for index, (example, line) in enumerate(
        zip(examples, (evaluation.output_path / "rows.jsonl").read_text().splitlines(), strict=True)
    ):
        outcome = json.loads(line)
        sidecar = json.loads((evaluation.output_path / outcome["neural_trace_ref"]).read_text())
        positions = [i for i, e in enumerate(sidecar["events"]) if e["kind"] == "act"]
        if len(positions) > 1 or len(positions) != len(captures.get(index, [])):
            raise ValueError("autonomous ACT observation inventory mismatch")
        metadata = {
            "error": outcome["error"],
            "miss_category": outcome["miss_category"],
            "neural_trace_ref": "autonomous/" + outcome["neural_trace_ref"],
            "full_trace_ref": "autonomous/" + outcome["full_trace_ref"]
            if outcome["full_trace_ref"]
            else None,
            "guard_context": "actual_autonomous_segment" if positions else "no_act",
            "readout_position": "actual_autonomous_act" if positions else "missing_act",
        }
        logits = None
        if positions:
            pos = positions[0]
            event = sidecar["events"][pos]
            causal = sidecar["causal_events"][pos]
            previous = sidecar["events"][pos - 1]
            snapshot = captures[index][0]
            detail = snapshot.pop("context", None)
            if detail is not None:
                atomic_create_bytes(
                    output_dir / f"details/{index:05d}.autonomous-act.json.gz",
                    gzip.compress(canonical_json_bytes(detail), mtime=0),
                )
            logits = dict(event["predictions"])["class_logits"]
            metadata.update(snapshot)
            verified = False
            if outcome["full_trace_ref"]:
                trajectory = json.loads(
                    gzip.decompress(
                        (evaluation.output_path / outcome["full_trace_ref"]).read_bytes()
                    )
                )
                # Anchor0 is initialization; anchor[pos] precedes event[pos].
                before = _on_device(
                    _trajectory_state(trajectory["anchors"][pos]), next(model.parameters()).device
                )
                if (
                    _state_summary(state_at(before, event["timestamp"]))[1]
                    != snapshot["continuous_state_sha256"]
                ):
                    raise ValueError("retained trajectory differs from actual ACT state")
                if (
                    sha256_bytes(
                        canonical_json_bytes(asdict(SegmentSummary.from_segment(before.segment)))
                    )
                    != event["segment_sha256"]
                ):
                    raise ValueError("retained trajectory differs from pre-ACT segment")
                verified = True
            metadata.update(
                timestamp=event["timestamp"],
                elapsed_since_activation=event["timestamp"] - example.events[-1].timestamp,
                segment_sha256=event["segment_sha256"],
                guard_targets=previous["guard_targets"],
                guard_rates=previous["guard_rates"],
                guard_crossings=[
                    None,
                    None,
                    causal["raw_predicted_delta"]
                    if causal["raw_predicted_delta"] is not None
                    else causal["delta"],
                ],
                crossing_null_means="not_selected_by_scheduler",
                predicted_hypothesis_class=causal["hypothesis_before"]["hazard_type"],
                trajectory_act_state_verified=verified,
            )
        elif sidecar["events"]:
            # A missing readout still retains the last installed boundary. This
            # is not an ACT state and never supplies a fabricated abstain logit.
            pos = len(sidecar["events"]) - 1
            if sidecar["events"][pos]["kind"] == "terminal":
                pos -= 1
            if pos >= 0:
                event = sidecar["events"][pos]
                causal = sidecar["causal_events"][pos]
                hypothesis = causal["hypothesis_after"]
                metadata.update(
                    state_sha256=causal["state_sha256"],
                    state_hash_schema="runtime-continuous-state-v1",
                    segment_sha256=event["next_segment_sha256"],
                    guard_targets=event["guard_targets"],
                    guard_rates=event["guard_rates"],
                    guard_crossings=[
                        crossing_offset_host(0.0, target, rate)
                        for target, rate in zip(
                            event["guard_targets"], event["guard_rates"], strict=True
                        )
                    ],
                    crossing_null_means="no_finite_crossing",
                    guard_context="last_installed_boundary_without_act",
                    timestamp=event["timestamp"],
                    elapsed_since_activation=event["timestamp"] - example.events[-1].timestamp,
                    predicted_hypothesis_class=hypothesis["hazard_type"] if hypothesis else None,
                    context_hypothesis_class=hypothesis["hazard_type"] if hypothesis else None,
                )
        for kind in CONTEXTS[2:]:
            rows.append(
                _readout(
                    example,
                    kind,
                    logits,
                    act_occurred=bool(positions),
                    **metadata,
                    timed_success=outcome["timed_success"] if kind == CONTEXTS[2] else None,
                )
            )
    return rows, evaluation


def _on_device(value, device):
    """Materialize retained anchors on their original execution device for exact flow."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value):
        return type(value)(
            **{f.name: _on_device(getattr(value, f.name), device) for f in fields(value)}
        )
    if isinstance(value, tuple):
        return tuple(_on_device(item, device) for item in value)
    return value


def _diagnose_authenticated(*, restored, manifest, examples, config, manifest_sha256, output_dir):
    started = time.monotonic()
    if restored.model.config != config.neural:
        raise ValueError("original model and diagnostic configuration differ")
    if output_dir.exists():
        raise ValueError("diagnostic destination must be fresh")
    parent = output_dir.absolute().parent
    while not parent.exists():
        parent = parent.parent
    required = max(64 * 1024**2, len(examples) * 2 * 1024**2)
    free = shutil.disk_usage(parent).free
    if free < required:
        raise ValueError(
            f"insufficient diagnostic disk space: {free} available, {required} required"
        )
    execution = _execution_identity()
    historical = (
        restored.descriptor.file_sha256 == HISTORICAL_WEIGHTS_SHA256
        and manifest_sha256 == HISTORICAL_MANIFEST_SHA256
    )
    count = min(128, len(examples))
    if historical and (
        len(examples) != 10_000
        or Counter(e.variant.value for e in examples[:128])
        != {"positive": 64, "safe_negative": 32, "disconnected_negative": 32}
        or [e.key.episode_index for e in examples] != list(range(10_000))
    ):
        raise ValueError("historical corpus must retain the ordered stratified prefix")
    detailed_ids = tuple(e.init.episode_public_id for e in examples[:count])
    model = restored.model.eval()
    identity = NeuralModelIdentity.from_model(
        model, source_revision=restored.descriptor.source_commit
    )
    bundles = tuple(curriculum_to_bundle(e, config=config) for e in examples)
    evaluation_identity = EvaluationIdentity(
        experiment="phase4-action-diagnostic",
        stage=examples[0].key.stage,
        split=examples[0].key.split,
        purpose="action_diagnostic",
        manifest_schema=manifest.schema_version,
        manifest_sha256=manifest_sha256,
        episodes=tuple(EpisodeBinding.from_bundle(b) for b in bundles),
        checkpoint_sha256=restored.descriptor.file_sha256,
        model_identity=identity,
        evaluation_config_canonical_json=canonical_json_bytes(config).decode(),
        execution_source_revision=execution["source_revision"],
        full_trace_public_ids=detailed_ids,
    )
    output_dir.mkdir(parents=True)
    provenance = {
        "original_weights_sha256": restored.descriptor.file_sha256,
        "original_model_state_sha256": identity.model_state_sha256,
        "original_producing_source_revision": identity.source_revision,
        "original_config_sha256": restored.config.sha256,
        "original_config_canonical_json": restored.config.canonical_json.decode(),
        "original_manifest_sha256": manifest_sha256,
        "original_manifest_schema": manifest.schema_version,
        "original_manifest_source_revision": manifest.source_revision,
        "ordered_example_hashes": [e.example_hash for e in examples],
        "ordered_projected_bindings_sha256": sha256_bytes(
            canonical_json_bytes(
                {"episodes": [e.model_dump(mode="json") for e in evaluation_identity.episodes]}
            )
        ),
        "execution": execution,
        "execution_config_sha256": sha256_bytes(canonical_json_bytes(config)),
        "execution_config_canonical_json": canonical_json_bytes(config).decode(),
        "model_seed": restored.config.config.training.model_seed,
        "generator_version": manifest.generator_version,
        "transform_version": manifest.curriculum_version,
        "runtime_projection_version": "phase4-projection-v1",
        "device": str(next(model.parameters()).device),
        "foundation_model_calls": 0,
        "optimizer_steps": 0,
        "converted_weights_sha256": None,
    }
    atomic_create_bytes(output_dir / "provenance.json", canonical_json_bytes(provenance))
    atomic_create_bytes(
        output_dir / "preflight.json",
        canonical_json_bytes(
            {
                "free_bytes": free,
                "required_bytes": required,
                "detailed_public_ids": detailed_ids,
                "selection": "fixed_first_128_ordered_entries_before_predictions",
                "every_autonomous_failure_retained": True,
            }
        ),
    )
    rows = _immediate(model, examples, output_dir, count)
    # Retain every completed context before the next potentially failing workload.
    pending = output_dir / "readouts.jsonl"
    with pending.open("xb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row) + b"\n")
    teacher_rows, losses = _teacher(model, examples, config, output_dir, count)
    with pending.open("ab") as handle:
        for row in teacher_rows:
            handle.write(canonical_json_bytes(row) + b"\n")
    rows.extend(teacher_rows)
    gradients = _gradients(model, examples, config)
    autonomous_rows, evaluation = _autonomous(
        model, examples, bundles, evaluation_identity, config, output_dir, count
    )
    rows.extend(autonomous_rows)
    with pending.open("ab") as handle:
        for row in autonomous_rows:
            handle.write(canonical_json_bytes(row) + b"\n")
    summaries = [_summarize(rows, kind) for kind in CONTEXTS]
    reproduced = (
        historical
        and summaries[0].correct_count == 6629
        and summaries[0].positive_correct_count == 1641
    )
    if historical and not reproduced:
        raise ValueError(
            "historical immediate action counts did not reproduce "
            "6629/10000 and 1641/5000; raw evidence retained"
        )
    if NeuralModelIdentity.from_model(model, source_revision=identity.source_revision) != identity:
        raise ValueError("diagnostic mutated original logical model state")
    interpretation = {
        "context_mismatch": (
            "Context differences are measured; score differences alone do not establish causation."
        ),
        "weak_classification": (
            "Inspect class margins, confusion and complete-timed-unroll action loss/gradients."
        ),
        "guard_failure": "Missing ACT and runtime miss categories remain in the full denominator.",
        "transition_mismatch": (
            "Teacher transitions and learned transitions differ by design; "
            "no implementation defect is inferred from different scores."
        ),
        "conclusion": "unresolved_mixture; no correction or action-solved claim",
        "additional_diagnostic_queries": {
            "historical_observer_control_passes": len(examples),
            "timed_teacher_examples": len(examples),
            "gradient_examples": count,
        },
        "autonomous_metrics": evaluation.metrics.model_dump(mode="json"),
    }
    artifacts = {
        str(p.relative_to(output_dir)): sha256_bytes(p.read_bytes())
        for p in sorted(output_dir.rglob("*"))
        if p.is_file() and not p.is_relative_to(output_dir / "autonomous")
    }
    artifacts["autonomous/DONE"] = sha256_bytes((output_dir / "autonomous/DONE").read_bytes())
    report = ActionDiagnosticReport(
        provenance,
        *summaries,
        gradients,
        losses,
        detailed_ids,
        True,
        reproduced,
        artifacts,
        output_dir,
        time.monotonic() - started,
        interpretation,
    )
    payload = asdict(report)
    payload.pop("output_dir")  # Locators never become portable identity.
    raw = canonical_json_bytes(payload)
    atomic_create_bytes(output_dir / "report.json", raw)
    atomic_create_bytes(
        output_dir / "DONE",
        canonical_json_bytes(
            {"report_sha256": sha256_bytes(raw), "diagnostic_only": True, "gate_eligible": False}
        ),
    )
    return report
