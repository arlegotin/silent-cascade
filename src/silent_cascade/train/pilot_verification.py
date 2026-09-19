"""Actual pilot numerical diagnostics and bounded, non-acceptance workload evidence.

``checkpoint`` means a full Phase4 training archive, never portable weights. A
checkpoint-backed report authenticates its content, not its selection; the final
gate must independently bind that hash to the selected pilot descriptor.
"""

import math
import os
import time
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field, model_validator

from silent_cascade.config import ResolvedConfig
from silent_cascade.eval.compute import NeuralComputeMeter
from silent_cascade.eventflow.checkpoint_rng import restore_rng_snapshot
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.models.content_types import ContentLossInputs
from silent_cascade.models.types import LossInputs
from silent_cascade.rng import snapshot_global_rng
from silent_cascade.train.batches import pack_training_examples
from silent_cascade.train.curriculum_data import curriculum_allocation, make_curriculum_example
from silent_cascade.train.objective import training_objective
from silent_cascade.train.pilot_config import Phase4Config
from silent_cascade.train.pilot_data import _key
from silent_cascade.train.pilot_provenance import PilotSourceIdentity
from silent_cascade.train.pilot_trainer import pilot_train_one_step
from silent_cascade.train.trainer import _check_objective
from silent_cascade.train.verification import Comparison, _compare, _tensors
from silent_cascade.validation import StrictModel

_TIMED_TERMS = (
    "guard",
    "retrieval",
    "compose_type",
    "focus",
    "hazard",
    "deadline",
    "action",
    "action_time",
    "state",
    "event_cost",
)
_COMPOSE = (
    "compose_role",
    "compose_status",
    "compose_confidence",
    "compose_append_support",
    "compose_continue_search",
)
_TIMED_REDUCTIONS = (
    "guard_active_margin",
    "guard_inactive_margin",
    "guard_time",
    "guard_race",
    "guard_final_dormancy",
    "retrieval",
    *_COMPOSE,
    "focus",
    "hazard",
    "deadline_log_delay",
    "deadline_normalized",
    "action_positive",
    "action_abstention",
    "action_time",
    "state_slow_change",
    "state_bound",
    "event_cost",
    "guard",
)
_CONTENT_TERMS = ("recall_available", "retrieval", "focus", "hazard", "log_delay", "compose_type")
_CONTENT_REDUCTIONS = ("recall_available", "retrieval", *_COMPOSE, "focus", "hazard", "log_delay")
_GROUPS = ("forward", "losses", "gradients", "parameters", "optimizer")


def _capture_dtype(group, name):
    field = name.rsplit("/", 1)[-1]
    if group == "forward":
        if field.endswith("_mask"):
            return torch.bool
        if field in {
            "kind_target",
            "action_target",
            "focus_target",
            "hazard_target",
            "retrieval_target",
            "role_target",
            "status_target",
        }:
            return torch.int64
    if group == "losses" and "/denominators/" in name:
        return torch.int64
    return torch.float32


def expected_inventory(model):
    """Derive names from model/schema declarations, independently of measured tensors.

    Forward outputs cover every tensor at the timed/content objective boundary,
    including logits, dynamics, state jumps, targets and masks. Loss inventories
    explicitly include every reduction, numerator, denominator and position.
    """
    parameters = tuple(sorted("/" + n for n, _ in model.named_parameters()))
    forward = tuple(
        sorted(
            f"/{branch}/{item.name}"
            for branch, schema in (("timed", LossInputs), ("content", ContentLossInputs))
            for item in fields(schema)
        )
    )
    losses = ["/total"]
    for branch, terms, reductions in (
        ("timed", _TIMED_TERMS, _TIMED_REDUCTIONS),
        ("content", _CONTENT_TERMS, _CONTENT_REDUCTIONS),
    ):
        losses.append(f"/{branch}/total")
        for group in ("terms", "subterms", "numerators", "denominators", "per_position"):
            losses.extend(
                f"/{branch}/{group}/{n}" for n in (terms if group == "terms" else reductions)
            )
    return {
        "forward": forward,
        "losses": tuple(sorted(losses)),
        "gradients": parameters,
        "parameters": tuple(sorted("/" + n for n in model.state_dict())),
        "optimizer": tuple(
            sorted(f"{n}/{kind}" for n in parameters for kind in ("step", "exp_avg", "exp_avg_sq"))
        ),
    }


def diagnostic_batch(config, *, stage, counter):
    """Separate debug stream; primary selects complete path-four quartets.

    Filtering is diagnostic max-trace construction, not a changed training stream.
    Counter partitions the accepted ordered stream; all original keys are retained.
    """
    if stage not in {"one_hop", "primary"} or type(counter) is not int or counter < 0:
        raise ValueError("invalid diagnostic stage/counter")
    size = config.pilot.batch_size
    selected, index = [], 0
    while len(selected) < (counter + 1) * size:
        key = _key(stage, "debug", index)
        if stage == "one_hop" or curriculum_allocation(key)[1] == 4:
            selected.append(key)
        index += 1
    return pack_training_examples(
        tuple(make_curriculum_example(config, k) for k in selected[counter * size :]),
        next_batch_counter=counter + 1,
    )


@dataclass(frozen=True)
class CapturedUpdate:
    tensors: dict
    step: object
    diagnostic_forward_compute: object


def capture_update(model, optimizer, batch, config):
    """Capture complete objective outputs, then the real trainer update.

    The extra no-grad forward is explicitly counted; hooks observe unclipped
    gradients from the real update without replacing its objective or optimizer.
    """
    rng = snapshot_global_rng()
    model.train()
    with torch.no_grad(), NeuralComputeMeter(model) as meter:
        objective = training_objective(model, batch, config.pilot)
        _check_objective(objective)
        forward = _tensors(
            {"timed": objective.timed.loss_inputs(), "content": objective.content.loss_inputs()}
        )
        losses = _tensors(
            {
                "timed": objective.timed_loss,
                "content": objective.content_loss,
                "total": objective.total,
            }
        )
    del objective
    restore_rng_snapshot(rng, restore_mps=next(model.parameters()).device.type == "mps")
    gradients, handles = {}, []
    for name, parameter in model.named_parameters():

        def capture(value, name=name):
            gradients["/" + name] = value.detach().cpu().clone()

        handles.append(parameter.register_hook(capture))
    try:
        step = pilot_train_one_step(model, optimizer, batch, config)
    finally:
        for handle in handles:
            handle.remove()
    tensors = {
        "forward": forward,
        "losses": losses,
        "gradients": gradients,
        "parameters": _tensors(model.state_dict()),
        "optimizer": _tensors({n: optimizer.state.get(p, {}) for n, p in model.named_parameters()}),
    }
    expected = expected_inventory(model)
    for group in _GROUPS:
        if tuple(sorted(tensors[group])) != expected[group]:
            raise ValueError(f"incomplete {group} inventory")
        for name, value in tensors[group].items():
            if value.dtype != _capture_dtype(group, name):
                raise ValueError(f"raw tensor dtype differs: {name}")
            if value.is_floating_point():
                if value.dtype != torch.float32:
                    raise ValueError(f"non-float32 model tensor: {name}")
                valid = torch.isfinite(value)
                if group == "forward" and name == "/timed/crossing_offsets":
                    valid |= torch.isposinf(value)
                if group == "forward" and name.endswith("/retrieval_logits"):
                    mask = tensors[group][name.replace("logits", "eligible_mask")]
                    valid |= torch.isneginf(value) & ~mask
                if not bool(valid.all()):
                    raise ValueError(f"nonfinite diagnostic tensor: {name}")
    return CapturedUpdate(tensors, step, meter.snapshot())


class UpdateComparison(StrictModel):
    mode: Literal["cpu_mps", "resume_cpu", "resume_mps"]
    expected: dict[str, tuple[str, ...]]
    forward: Comparison
    losses: Comparison
    gradients: Comparison
    parameters: Comparison
    optimizer: Comparison

    @model_validator(mode="after")
    def check(self):
        if set(self.expected) != set(_GROUPS):
            raise ValueError("comparison inventory groups differ")
        for group in _GROUPS:
            value = getattr(self, group)
            names = self.expected[group]
            if (
                not names
                or len(names) != len(set(names))
                or (
                    value.tested_names != names
                    or set(value.tensor_elements) != set(names)
                    or set(value.tensor_shapes) != set(names)
                    or value.compared != sum(value.tensor_elements.values())
                )
            ):
                raise ValueError(f"incomplete {group} inventory")
            tolerance = (
                (0.0, 0.0)
                if self.mode == "resume_cpu"
                else (
                    1e-4 if self.mode == "resume_mps" or group in {"forward", "losses"} else 1e-3,
                    1e-5,
                )
            )
            if (value.rtol, value.atol) != tolerance:
                raise ValueError("comparison tolerance changed")
        return self

    def validate_complete_inventory(self, model):
        self.check()
        if self.expected != expected_inventory(model):
            raise ValueError("model/output inventory differs from independently declared inventory")
        for group, tensors in (
            ("gradients", dict(model.named_parameters())),
            ("parameters", model.state_dict()),
        ):
            comparison = getattr(self, group)
            for name, tensor in tensors.items():
                if (
                    comparison.tensor_shapes["/" + name] != tuple(tensor.shape)
                    or comparison.tensor_elements["/" + name] != tensor.numel()
                ):
                    raise ValueError(f"{group} shape inventory differs from actual model")
        for name, tensor in model.named_parameters():
            for kind in ("step", "exp_avg", "exp_avg_sq"):
                key = f"/{name}/{kind}"
                if self.optimizer.tensor_shapes[key] != (
                    () if kind == "step" else tuple(tensor.shape)
                ) or self.optimizer.tensor_elements[key] != (
                    1 if kind == "step" else tensor.numel()
                ):
                    raise ValueError("optimizer shape inventory differs from actual model")

    @property
    def passed(self):
        self.check()
        return all(getattr(self, group).passed for group in _GROUPS)


def compare_update(left, right, *, expected, resume_device=None):
    mode = "cpu_mps" if resume_device is None else "resume_" + resume_device
    comparisons = {}
    for group in _GROUPS:
        if (
            tuple(sorted(left.tensors[group])) != expected[group]
            or tuple(sorted(right.tensors[group])) != expected[group]
        ):
            raise ValueError(f"incomplete {group} inventory")
        rtol, atol = (
            (0.0, 0.0)
            if resume_device == "cpu"
            else (1e-4 if resume_device == "mps" or group in {"forward", "losses"} else 1e-3, 1e-5)
        )
        comparisons[group] = _compare(
            left.tensors[group], right.tensors[group], rtol=rtol, atol=atol
        )
    return UpdateComparison(mode=mode, expected=expected, **comparisons)


class RuntimeComparison(StrictModel):
    episodes: int = Field(ge=1)
    decision_mismatches: int = Field(ge=0)
    score_mismatches: int = Field(ge=0)
    time_mismatches: int = Field(ge=0)
    error_episodes: int = Field(ge=0)
    rtol: Literal[1e-4] = 1e-4
    atol: Literal[1e-5] = 1e-5

    @property
    def passed(self):
        return not (
            self.decision_mismatches
            or self.score_mismatches
            or self.time_mismatches
            or self.error_episodes
        )


def compare_runtime(left, right):
    if (
        not left
        or len(left) != len(right)
        or [r["public_id"] for r in left] != [r["public_id"] for r in right]
    ):
        raise ValueError("runtime episode inventory differs")
    decisions = scores = times = errors = 0
    for a, b in zip(left, right, strict=True):
        scores += a["score"] != b["score"]
        errors += a["error"] is not None or b["error"] is not None
        if len(a["events"]) != len(b["events"]):
            decisions += 1
            continue
        for x, y in zip(a["events"], b["events"], strict=True):
            decisions += any(x[n] != y[n] for n in ("kind", "selected_record", "predicted_class"))
            times += not math.isclose(x["timestamp"], y["timestamp"], rel_tol=1e-4, abs_tol=1e-5)
            if len(x["actions"]) != len(y["actions"]):
                decisions += 1
            else:
                for ax, ay in zip(x["actions"], y["actions"], strict=True):
                    decisions += {k: v for k, v in ax.items() if k != "timestamp"} != {
                        k: v for k, v in ay.items() if k != "timestamp"
                    }
                    times += not math.isclose(
                        ax["timestamp"], ay["timestamp"], rel_tol=1e-4, abs_tol=1e-5
                    )
    return RuntimeComparison(
        episodes=len(left),
        decision_mismatches=decisions,
        score_mismatches=scores,
        time_mismatches=times,
        error_episodes=errors,
    )


def measure_operation(operation, *, device):
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in {"", "0"}:
        raise ValueError("silent MPS fallback is forbidden")
    if device not in {"cpu", "mps"}:
        raise ValueError("expected native CPU or MPS")
    if device == "mps":
        torch.mps.synchronize()
    started = time.perf_counter()
    result = operation()
    if device == "mps":
        torch.mps.synchronize()
    return result, time.perf_counter() - started


def native_devices():
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in {"", "0"}:
        raise ValueError("silent MPS fallback is forbidden")
    return (("cpu", "mps"), ()) if torch.backends.mps.is_available() else (("cpu",), ("mps",))


def load_numeric_checkpoint(path, *, config, source, device):
    """Strict full-archive load; content hashing does not assert pilot selection."""
    from silent_cascade.eventflow.neural_weights import decode_archive, read_bytes
    from silent_cascade.train.pilot_checkpoints import PilotCheckpoint, load_pilot_checkpoint

    raw = read_bytes(path)
    digest = sha256_bytes(raw)
    try:
        metadata, _ = decode_archive(raw, digest, PilotCheckpoint, "pilot_training")
    except Exception as error:
        raise ValueError(
            "checkpoint must be a valid full Phase4 training archive; "
            "portable weights are unsupported"
        ) from error
    session = load_pilot_checkpoint(
        path, expected_sha256=digest, config=config, source=source, device=device
    )
    return session, digest, metadata


def _rng_hash(snapshot):
    from silent_cascade.eventflow.checkpoint_rng import encode_rng

    metadata, tensors = encode_rng(snapshot)
    return sha256_bytes(
        canonical_json_bytes(
            {
                "metadata": metadata.model_dump(mode="json"),
                "tensors": {n: sha256_bytes(t.numpy().tobytes()) for n, t in tensors.items()},
            }
        )
    )


def _random_draws(device):
    import random

    import numpy as np

    return {
        "python": random.random(),
        "numpy": float(np.random.random()),
        "cpu": torch.rand(5).tolist(),
        "native_mps": torch.rand(5, device="mps").cpu().tolist() if device == "mps" else None,
    }


def _rng_observation(snapshot):
    from silent_cascade.eventflow.checkpoint_rng import encode_rng

    metadata, tensors = encode_rng(snapshot)
    return {
        "metadata": metadata.model_dump(mode="json"),
        "tensors": {n: t.tolist() for n, t in tensors.items()},
    }


def _decode_observed_rng(value):
    from silent_cascade.eventflow.checkpoint_rng import RngMetadata, decode_rng

    if set(value) != {"metadata", "tensors"}:
        raise ValueError("raw RNG observation differs")
    tensors = {}
    for name, values in value["tensors"].items():
        if any(type(v) is not int or not 0 <= v <= 255 for v in values):
            raise ValueError("raw RNG bytes differ")
        tensors[name] = torch.tensor(values, dtype=torch.uint8)
    return decode_rng(
        RngMetadata.model_validate_json(canonical_json_bytes(value["metadata"])), tensors
    )


def _observed_rng_hash(value):
    return _rng_hash(_decode_observed_rng(value))


def _continuation_rng(snapshot, device):
    # CPU restoration deliberately leaves opaque native state outside its claim.
    return replace(snapshot, torch_mps_state=None) if device == "cpu" else snapshot


def clone_training_state(model, optimizer, device):
    model, optimizer = deepcopy((model, optimizer))
    model.to(device)
    for state in optimizer.state.values():
        for name, value in state.items():
            state[name] = value.to("cpu" if name == "step" else device)
    return model, optimizer


class PilotResumeReport(StrictModel):
    device: Literal["cpu", "mps"]
    checkpoint_sha256: str
    archive_global_step: int = Field(ge=0, le=75000)
    next_batch_counter: int = Field(ge=1)
    next_batch_hashes: tuple[str, ...]
    restored_batch_hashes: tuple[str, ...]
    rng_before_sha256: str
    restored_rng_sha256: str
    random_draw_sha256: str
    restored_random_draw_sha256: str
    rng_after_sha256: str
    restored_rng_after_sha256: str
    comparison: UpdateComparison
    diagnostic_updates: int = Field(ge=2, le=3)
    foundation_model_calls: Literal[0] = 0

    @property
    def passed(self):
        return self.comparison.passed and (
            self.next_batch_hashes == self.restored_batch_hashes
            and self.rng_before_sha256 == self.restored_rng_sha256
            and self.random_draw_sha256 == self.restored_random_draw_sha256
            and self.rng_after_sha256 == self.restored_rng_after_sha256
        )


def measure_resume(config, *, model, optimizer, progress, source, device, output_dir):
    """Save the actual session, execute and restore its genuine next training batch.

    Diagnostic continuation may inspect step75001 after a step75000 archive, but
    never saves forged production progress or credits that update to a pilot.
    """
    from silent_cascade.train.pilot_checkpoints import save_pilot_checkpoint
    from silent_cascade.train.pilot_data import next_pilot_batch

    model, optimizer = clone_training_state(model, optimizer, device)
    observations = {"initial": _rng_observation(_continuation_rng(snapshot_global_rng(), device))}
    updates = 2
    if not optimizer.state:
        batch = next_pilot_batch(
            config.config, stage=progress.stage, batch_counter=progress.batch_counter
        ).to(device)
        bootstrap = capture_update(model, optimizer, batch, config.config)
        _save_capture(output_dir / "bootstrap", bootstrap)
        progress = progress.model_copy(
            update={
                "global_step": progress.global_step + 1,
                "batch_counter": progress.batch_counter + 1,
            }
        )
        updates += 1
    path = output_dir / "resume.safetensors"
    digest = save_pilot_checkpoint(
        path, model=model, optimizer=optimizer, progress=progress, config=config, source=source
    )
    observations["before"] = _rng_observation(_continuation_rng(snapshot_global_rng(), device))
    observations["draws"] = _random_draws(device)
    batch = next_pilot_batch(
        config.config, stage=progress.stage, batch_counter=progress.batch_counter
    ).to(device)
    expected = capture_update(model, optimizer, batch, config.config)
    observations["after"] = _rng_observation(_continuation_rng(snapshot_global_rng(), device))
    restored, _, _ = load_numeric_checkpoint(path, config=config, source=source, device=device)
    observations["restored_before"] = _rng_observation(
        _continuation_rng(snapshot_global_rng(), device)
    )
    observations["restored_draws"] = _random_draws(device)
    next_batch = next_pilot_batch(
        config.config, stage=restored.progress.stage, batch_counter=restored.progress.batch_counter
    ).to(device)
    actual = capture_update(restored.model, restored.optimizer, next_batch, config.config)
    _save_capture(output_dir / "expected", expected)
    _save_capture(output_dir / "restored", actual)
    observations["restored_after"] = _rng_observation(
        _continuation_rng(snapshot_global_rng(), device)
    )
    result = PilotResumeReport(
        device=device,
        checkpoint_sha256=digest,
        archive_global_step=progress.global_step,
        next_batch_counter=next_batch.next_batch_counter,
        next_batch_hashes=batch.example_hashes,
        restored_batch_hashes=next_batch.example_hashes,
        rng_before_sha256=_observed_rng_hash(observations["before"]),
        restored_rng_sha256=_observed_rng_hash(observations["restored_before"]),
        random_draw_sha256=sha256_bytes(canonical_json_bytes(observations["draws"])),
        restored_random_draw_sha256=sha256_bytes(
            canonical_json_bytes(observations["restored_draws"])
        ),
        rng_after_sha256=_observed_rng_hash(observations["after"]),
        restored_rng_after_sha256=_observed_rng_hash(observations["restored_after"]),
        comparison=compare_update(
            expected, actual, expected=expected_inventory(model), resume_device=device
        ),
        diagnostic_updates=updates,
    )
    from silent_cascade.train.pilot_trainer import publish_json

    publish_json(output_dir / "observations.json", observations)
    publish_json(output_dir / "resume.json", result)
    return result


def forecast_workload(**kwargs):
    from silent_cascade.train.pilot_measurement import forecast_workload as forecast

    return forecast(**kwargs)


def measure_pilot_offline(*, output_dir):
    from silent_cascade.train.pilot_offline import measure_pilot_offline as measure

    return measure(output_dir=output_dir)


class DiagnosticSubset(StrictModel):
    parent_manifest_sha256: str
    parent_count: int
    indices: tuple[int, ...]
    episode_sha256s: tuple[str, ...]
    variant_counts: dict[str, int]
    purpose: Literal["action_diagnostic"] = "action_diagnostic"

    def validate_manifest(self, manifest):
        if (
            self.parent_manifest_sha256 != sha256_bytes(canonical_json_bytes(manifest))
            or self.parent_count != manifest.count
            or self.indices != tuple(range(len(self.indices)))
            or not self.indices
            or len(self.indices) % 4
            or len(self.indices) > manifest.count
        ):
            raise ValueError("diagnostic subset parent/order mismatch")
        entries = tuple(manifest.entries[i].projected for i in self.indices)
        if self.episode_sha256s != tuple(
            e.episode_sha256 for e in entries
        ) or self.variant_counts != dict(Counter(e.variant for e in entries)):
            raise ValueError("diagnostic subset episode identity mismatch")


def declare_subset(manifest, *, count, output_path):
    from silent_cascade.train.pilot_trainer import publish_json

    if type(count) is not int or count < 4 or count > manifest.count or count % 4:
        raise ValueError("diagnostic subset must contain complete ordered quartets")
    entries = manifest.entries[:count]
    result = DiagnosticSubset(
        parent_manifest_sha256=sha256_bytes(canonical_json_bytes(manifest)),
        parent_count=manifest.count,
        indices=tuple(range(count)),
        episode_sha256s=tuple(e.projected.episode_sha256 for e in entries),
        variant_counts=dict(Counter(e.projected.variant for e in entries)),
    )
    result.validate_manifest(manifest)
    publish_json(output_path, result)
    return result


class TrainingTrial(StrictModel):
    device: Literal["cpu", "mps"]
    stage: Literal["one_hop", "primary"]
    batch_size: int = Field(ge=1, le=128)
    max_trace_steps: int = Field(ge=1, le=11)
    warmup_updates: int = Field(ge=1, le=16)
    measured_updates: int = Field(ge=1, le=32)
    update_seconds: tuple[float, ...]
    raw_step_rows: int
    initial_model_sha256: str
    final_model_sha256: str
    parameter_count: int = Field(gt=0, le=5_000_000)
    total_forward_macs: int = Field(gt=0)
    total_backward_macs: int = Field(gt=0)
    tensor_working_memory_bytes: int = Field(gt=0)
    process_highwater_bytes: int | None
    mps_driver_allocation_bytes: int | None
    foundation_model_calls: Literal[0] = 0

    @model_validator(mode="after")
    def counts(self):
        if (
            len(self.update_seconds) != self.measured_updates
            or self.raw_step_rows != self.warmup_updates + self.measured_updates
            or any(t <= 0 or not math.isfinite(t) for t in self.update_seconds)
        ):
            raise ValueError("measured update inventory differs")
        return self


class WorkloadForecast(StrictModel):
    maximum_global_updates: Literal[75000] = 75000
    scheduled_validation_boundaries: Literal[75] = 75
    autonomous_validation_episodes_upper: Literal[1500000] = 1500000
    component_validation_episodes_upper: Literal[750000] = 750000
    final_work_episodes: Literal[40640] = 40640
    replay_episode_equivalent_range: tuple[Literal[12], Literal[216]] = (12, 216)
    leakage_audit_count: Literal[4] = 4
    audit_context_seconds: float
    compute_seconds_lower: float
    compute_seconds_upper: float
    projected_retained_bytes_lower: int
    projected_retained_bytes_upper: int
    checkpoint_working_bytes: int
    required_free_bytes: int
    observed_free_bytes: int
    space_sufficient: bool
    convergence_guaranteed: Literal[False] = False
    assumptions: tuple[str, ...]
    retention_components: dict[str, int]


def measure_training_trial(*args, **kwargs):
    from silent_cascade.train.pilot_measurement import measure_training_trial as measure

    return measure(*args, **kwargs)


def authenticate_numeric_source(config, checkpoint):
    from silent_cascade.eventflow.neural_weights import decode_archive, read_bytes
    from silent_cascade.train.pilot_checkpoints import PilotCheckpoint
    from silent_cascade.train.pilot_provenance import authenticate_pilot_source
    from silent_cascade.train.provenance import git

    root = Path(__file__).resolve().parents[3]
    archived = None
    if checkpoint is not None:
        raw = read_bytes(checkpoint)
        try:
            metadata, _ = decode_archive(raw, sha256_bytes(raw), PilotCheckpoint, "pilot_training")
        except Exception as error:
            raise ValueError(
                "checkpoint must be a valid full Phase4 training archive; "
                "portable weights are unsupported"
            ) from error
        archived = metadata.source
    revision = (
        archived.source_commit if archived else git(root, "rev-parse", "HEAD").decode().strip()
    )
    source = authenticate_pilot_source(repo_root=root, source_commit=revision, config=config)
    if archived is not None:
        if archived.model_copy(update={"data_introductions": {}}) != source:
            raise ValueError("checkpoint source/config identity differs")
        # Preserve metadata to satisfy the actual loader; no data-audit claim is made.
        source = archived
    return source


def _artifact_hashes(root):
    from silent_cascade.eventflow.neural_weights import read_bytes

    return {
        str(p.relative_to(root)): sha256_bytes(read_bytes(p))
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _save_capture(path, captured):
    from safetensors.torch import save

    from silent_cascade.train.pilot_data import _publish_pilot_bytes
    from silent_cascade.train.pilot_trainer import publish_json

    _publish_pilot_bytes(
        path.with_suffix(".safetensors"),
        save(
            {
                group + name: tensor.contiguous()
                for group, values in captured.tensors.items()
                for name, tensor in values.items()
            }
        ),
    )
    publish_json(
        path.with_suffix(".json"),
        {"update": captured.step, "extra_diagnostic_forward": captured.diagnostic_forward_compute},
    )


def _evaluate_subset(config, *, model, identity, weights_sha, manifest, subset, device, output_dir):
    from silent_cascade.env.pilot import curriculum_to_bundle
    from silent_cascade.eval.artifacts import EvaluationIdentity
    from silent_cascade.eval.runner import evaluate_episodes
    from silent_cascade.train.pilot_data import _entry

    subset.validate_manifest(manifest)

    def episodes():
        for index in subset.indices:
            entry = manifest.entries[index]
            example = make_curriculum_example(config.config, entry.key)
            if _entry(example, config.config) != entry:
                raise ValueError("diagnostic subset regenerated data differs")
            yield curriculum_to_bundle(example, config=config.config)

    evaluation_identity = EvaluationIdentity(
        experiment="phase4-task10-diagnostic",
        stage=manifest.stage,
        split=manifest.split,
        purpose="action_diagnostic",
        manifest_schema=manifest.schema_version,
        manifest_sha256=subset.parent_manifest_sha256,
        episodes=tuple(manifest.entries[i].projected for i in subset.indices),
        checkpoint_sha256=weights_sha,
        model_identity=identity,
        evaluation_config_canonical_json=config.canonical_json.decode(),
        execution_source_revision=identity.source_revision,
    )
    return evaluate_episodes(
        model,
        identity=evaluation_identity,
        config=config.config,
        episodes=episodes(),
        output_dir=output_dir,
        device=device,
    )


def _runtime_rows(root, *, evidence_context=None):
    from silent_cascade.archive.readers import iter_evaluation_episodes

    evidence_context = _numeric_context(root.parent, evidence_context)
    result = []
    for episode in iter_evaluation_episodes(root, evidence_context=evidence_context):
        row, sidecar = episode.row, episode.sidecar
        result.append(
            {
                "public_id": row.public_id,
                "score": asdict(row.score),
                "error": row.error,
                "events": [
                    {
                        "kind": causal["kind"],
                        "timestamp": causal["timestamp"],
                        "selected_record": causal["selected_record_id"],
                        "predicted_class": (
                            neural["raw_action_argmax"],
                            neural["legal_action_argmax"],
                        ),
                        "actions": causal["actions"],
                    }
                    for causal, neural in zip(
                        sidecar["causal_events"], sidecar["events"], strict=True
                    )
                ],
            }
        )
    return result


class PilotNumericReport(StrictModel):
    schema_version: Literal["phase4-numeric-diagnostic-v1"] = "phase4-numeric-diagnostic-v1"
    evidence_kind: Literal["fresh_numerical_diagnostic", "checkpoint_numerical_diagnostic"]
    config_canonical_json: str
    config_sha256: str
    source: PilotSourceIdentity
    selection_verified: Literal[False] = False
    data_introductions_verified: Literal[False] = False
    original_checkpoint_sha256: str | None
    initial_model_sha256: str
    runtime_model_sha256: str
    portable_weights_sha256: str
    model_seed: Literal[11] = 11
    parameter_count: int = Field(gt=0, le=5_000_000)
    generator_version: Literal["ofd-v1"] = "ofd-v1"
    transform_version: Literal["ofd-one-hop-v1"] = "ofd-one-hop-v1"
    data_recipe_version: Literal["phase4-data-v1"] = "phase4-data-v1"
    batches: dict[str, tuple[str, ...]]
    batch_recipe_sha256s: dict[str, str]
    cpu_inventory: dict[str, UpdateComparison]
    device_parity: dict[str, UpdateComparison]
    cpu_resume: PilotResumeReport
    mps_resume: PilotResumeReport | None
    runtime_episodes: int
    runtime_subset: DiagnosticSubset
    runtime_comparison: RuntimeComparison | None
    missing_devices: tuple[Literal["mps"], ...]
    diagnostic_updates: int
    extra_diagnostic_forwards: int
    foundation_model_calls: Literal[0] = 0
    artifact_hashes: dict[str, str]

    @model_validator(mode="after")
    def validate_complete_inventory(self):
        from silent_cascade.models.event_flow import EventFlowModel
        from silent_cascade.train.pilot_config import parse_phase4_canonical

        config = parse_phase4_canonical(self.config_canonical_json)
        if (
            self.config_sha256 != sha256_bytes(self.config_canonical_json.encode())
            or self.source.config_sha256 != self.config_sha256
        ):
            raise ValueError("numeric config/source hash differs")
        if self.initial_model_sha256 != self.runtime_model_sha256:
            raise ValueError("runtime model was changed by diagnostic updates")
        if (
            self.cpu_resume.device != "cpu"
            or self.cpu_resume.comparison.mode != "resume_cpu"
            or (
                self.mps_resume is not None
                and (
                    self.mps_resume.device != "mps"
                    or self.mps_resume.comparison.mode != "resume_mps"
                )
            )
            or any(c.mode != "resume_cpu" for c in self.cpu_inventory.values())
            or any(c.mode != "cpu_mps" for c in self.device_parity.values())
        ):
            raise ValueError("resume device/comparison binding differs")
        if (
            self.runtime_episodes != (64 if config.pilot.is_production else 16)
            or len(self.runtime_subset.indices) != self.runtime_episodes
            or len(self.runtime_subset.episode_sha256s) != self.runtime_episodes
        ):
            raise ValueError("runtime subset/count inventory differs")
        if (self.original_checkpoint_sha256 is None) != (
            self.evidence_kind == "fresh_numerical_diagnostic"
        ):
            raise ValueError("numeric checkpoint evidence kind differs")
        if (
            set(self.batches) != {"one_hop", "primary"}
            or set(self.cpu_inventory) != set(self.batches)
            or set(self.batch_recipe_sha256s) != set(self.batches)
        ):
            raise ValueError("numeric stage inventory differs")
        with torch.random.fork_rng(devices=[]):
            model = EventFlowModel(config.neural)
        for comparison in (
            *self.cpu_inventory.values(),
            *self.device_parity.values(),
            self.cpu_resume.comparison,
            *((self.mps_resume.comparison,) if self.mps_resume else ()),
        ):
            comparison.validate_complete_inventory(model)
        if self.parameter_count != sum(p.numel() for p in model.parameters()):
            raise ValueError("parameter inventory count differs")
        if any(len(hashes) != config.pilot.batch_size for hashes in self.batches.values()):
            raise ValueError("numeric batch inventory differs")
        if self.missing_devices:
            if (
                self.missing_devices != ("mps",)
                or self.device_parity
                or self.mps_resume
                or self.runtime_comparison
            ):
                raise ValueError("missing native device cannot carry parity evidence")
        elif (
            set(self.device_parity) != set(self.batches)
            or self.mps_resume is None
            or self.runtime_comparison is None
        ):
            raise ValueError("native MPS evidence inventory is incomplete")
        return self

    @property
    def device_checks_passed(self):
        return (
            not self.missing_devices
            and self.cpu_resume.passed
            and self.mps_resume.passed
            and all(c.passed for c in self.device_parity.values())
            and self.runtime_comparison.passed
            and self.runtime_episodes == 64
        )


def verify_pilot_numerics(
    config: ResolvedConfig[Phase4Config], *, checkpoint: Path | None, output_dir: Path
) -> PilotNumericReport:
    """Actual recipe diagnostics; selected-checkpoint acceptance is a later binding."""
    from silent_cascade.eventflow.neural import NeuralModelIdentity
    from silent_cascade.eventflow.neural_weights import save_neural_weights
    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.rng import seed_all
    from silent_cascade.train.pilot_data import freeze_pilot_manifest
    from silent_cascade.train.pilot_state import PilotProgress
    from silent_cascade.train.pilot_trainer import publish_json
    from silent_cascade.train.trainer import make_optimizer

    source = authenticate_numeric_source(config, checkpoint)
    devices, missing = native_devices()
    if output_dir.exists():
        raise ValueError("numeric destination must be fresh; retain previous failures")
    publish_json(
        output_dir / "intent.json",
        {
            "source": source,
            "config": config.canonical_json.decode(),
            "checkpoint": str(checkpoint) if checkpoint else None,
        },
    )
    before = snapshot_global_rng()
    try:
        seed_all(11)
        if checkpoint is None:
            model = EventFlowModel(config.config.neural)
            optimizer = make_optimizer(model, config.config.pilot)
            progress, original_sha = PilotProgress(), None
            parity_rng = snapshot_global_rng()
        else:
            session, original_sha, _ = load_numeric_checkpoint(
                checkpoint, config=config, source=source, device="cpu"
            )
            model, optimizer, progress = session.model, session.optimizer, session.progress
            parity_rng = session.rng
            if parity_rng.torch_mps_state is None:
                # Available hardware cannot supply native continuation evidence
                # that is absent from the input archive.
                devices, missing = ("cpu",), ("mps",)
            restore_rng_snapshot(parity_rng, restore_mps="mps" in devices)
            from silent_cascade.train.pilot_data import _publish_pilot_bytes, _read_pilot_bytes

            _publish_pilot_bytes(
                output_dir / "input-training.safetensors", _read_pilot_bytes(checkpoint)
            )
        identity = NeuralModelIdentity.from_model(model, source_revision=source.source_commit)
        weights_sha = save_neural_weights(
            output_dir / "unchanged-weights.safetensors", model=model, identity=identity
        )
        manifest = freeze_pilot_manifest(
            config,
            stage="primary",
            output_path=output_dir / "manifest.json",
            source_commit=source.source_commit,
        )
        subset = declare_subset(
            manifest,
            count=64 if config.config.pilot.is_production else 16,
            output_path=output_dir / "runtime-subset.json",
        )
        cpu, parity, batches, recipes = {}, {}, {}, {}
        updates = forwards = 0
        for stage in ("one_hop", "primary"):
            batch = diagnostic_batch(config.config, stage=stage, counter=0)
            batches[stage] = batch.example_hashes
            recipes[stage] = publish_json(
                output_dir / f"{stage}-batch.json",
                {
                    "keys": batch.example_keys,
                    "example_hashes": batch.example_hashes,
                    "teacher_trace_steps": batch.targets.kind.shape[1],
                    "stage": stage,
                    "split": "debug",
                    "primary_path_filter": 4 if stage == "primary" else None,
                },
            )
            captured = {}
            for device in devices:
                restore_rng_snapshot(parity_rng, restore_mps=device == "mps")
                current, current_optimizer = clone_training_state(model, optimizer, device)
                captured[device] = capture_update(
                    current, current_optimizer, batch.to(device), config.config
                )
                _save_capture(output_dir / f"{stage}-{device}", captured[device])
                updates += 1
                forwards += 1
            expected = expected_inventory(model)
            cpu[stage] = compare_update(
                captured["cpu"], captured["cpu"], expected=expected, resume_device="cpu"
            )
            if "mps" in devices:
                parity[stage] = compare_update(captured["cpu"], captured["mps"], expected=expected)
        resumes, runtime = {}, {}
        for device in devices:
            restore_rng_snapshot(parity_rng, restore_mps=device == "mps")
            resumes[device] = measure_resume(
                config,
                model=model,
                optimizer=optimizer,
                progress=progress,
                source=source,
                device=device,
                output_dir=output_dir / f"resume-{device}",
            )
            updates += resumes[device].diagnostic_updates
            forwards += resumes[device].diagnostic_updates
            _evaluate_subset(
                config,
                model=model,
                identity=identity,
                weights_sha=weights_sha,
                manifest=manifest,
                subset=subset,
                device=device,
                output_dir=output_dir / f"runtime-{device}",
            )
            runtime[device] = _runtime_rows(output_dir / f"runtime-{device}")
        comparison = compare_runtime(runtime["cpu"], runtime["mps"]) if "mps" in devices else None
        if authenticate_numeric_source(config, checkpoint) != source:
            raise ValueError("source changed during numeric verification")
        report = PilotNumericReport(
            evidence_kind="fresh_numerical_diagnostic"
            if checkpoint is None
            else "checkpoint_numerical_diagnostic",
            config_canonical_json=config.canonical_json.decode(),
            config_sha256=config.sha256,
            source=source,
            original_checkpoint_sha256=original_sha,
            initial_model_sha256=identity.model_state_sha256,
            runtime_model_sha256=NeuralModelIdentity.from_model(
                model, source_revision=source.source_commit
            ).model_state_sha256,
            portable_weights_sha256=weights_sha,
            parameter_count=sum(p.numel() for p in model.parameters()),
            batches=batches,
            batch_recipe_sha256s=recipes,
            cpu_inventory=cpu,
            device_parity=parity,
            cpu_resume=resumes["cpu"],
            mps_resume=resumes.get("mps"),
            runtime_episodes=len(subset.indices),
            runtime_subset=subset,
            runtime_comparison=comparison,
            missing_devices=missing,
            diagnostic_updates=updates,
            extra_diagnostic_forwards=forwards,
            artifact_hashes=_artifact_hashes(output_dir),
        )
        digest = publish_json(output_dir / "numeric-report.json", report)
        publish_json(output_dir / "DONE", {"report_sha256": digest})
        return report
    except Exception as error:
        publish_json(
            output_dir / "failure.json", {"error_type": type(error).__name__, "error": str(error)}
        )
        raise
    finally:
        restore_rng_snapshot(before, restore_mps=before.torch_mps_state is not None)


class _NumericEvidenceContext:
    """One exhausted authenticated inventory shared by a numerical read."""

    def __init__(self, root, source):
        from silent_cascade.archive.readers import logical_root

        self.source = source
        self.run_dir = source.run_dir
        self.root = root.absolute()
        self.logical_root = logical_root(root, source)
        expected = "final/numerics"
        prefix = self.logical_root + "/"
        saw_expected = False
        self.entries_by_name = {}
        for entry in source.entries():
            saw_expected |= entry.path.startswith(expected + "/")
            if entry.path.startswith(prefix):
                self.entries_by_name[entry.path] = entry
        if saw_expected and self.logical_root != expected:
            raise ValueError("numeric evidence context root differs")

    def __getattr__(self, name):
        return getattr(self.source, name)


def _numeric_context(root, evidence_context):
    if evidence_context is None or isinstance(evidence_context, _NumericEvidenceContext):
        return evidence_context
    return _NumericEvidenceContext(root, evidence_context)


def _numeric_name(path, evidence_context):
    from silent_cascade.archive.readers import logical_root

    name = logical_root(path, evidence_context)
    if not Path(name).is_relative_to(evidence_context.logical_root):
        raise ValueError("numeric evidence path escapes root")
    return name


@contextmanager
def _numeric_path(path, *, evidence_context=None):
    if evidence_context is None:
        yield path
        return
    from silent_cascade.archive.readers import evidence_path

    name = _numeric_name(path, evidence_context)
    with evidence_path(evidence_context.run_dir, name, evidence_context=evidence_context) as local:
        yield local


def _numeric_exists(path, evidence_context):
    return path.exists() or (
        evidence_context is not None
        and _numeric_name(path, evidence_context) in evidence_context.entries_by_name
    )


def _numeric_json(path, *, evidence_context=None):
    from silent_cascade.report.pilot_artifacts import read_json

    with _numeric_path(path, evidence_context=evidence_context) as local:
        return read_json(local)


def _numeric_artifact_names(root, *, evidence_context=None):
    from silent_cascade.train.pilot_data import _check_path

    _check_path(root)
    observed = set()
    if root.is_symlink():
        raise ValueError("symbolic artifact closure")
    if root.exists():
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("symbolic artifact closure")
            if path.is_file():
                observed.add(str(path.relative_to(root)))
    if evidence_context is not None:
        prefix = _numeric_name(root, evidence_context) + "/"
        observed.update(
            name.removeprefix(prefix)
            for name in evidence_context.entries_by_name
            if name.startswith(prefix)
        )
    return observed


def _check_artifact_closure(root, hashes, required, *, report_name, evidence_context=None):
    from silent_cascade.eval.artifacts import read_evaluation_artifact
    from silent_cascade.report.pilot_artifacts import child

    observed = _numeric_artifact_names(root, evidence_context=evidence_context)
    if set(hashes) != set(required) or observed != set(required) | {report_name, "DONE"}:
        raise ValueError("artifact closure differs from required inventory")
    for name, digest in hashes.items():
        with _numeric_path(child(root, name), evidence_context=evidence_context) as local:
            raw = read_evaluation_artifact(local)
        if sha256_bytes(raw) != digest:
            raise ValueError(f"artifact integrity mismatch: {name}")


def _runtime_artifact_inventory(root, *, config, manifest, subset, weights, evidence_context=None):
    from silent_cascade.archive.readers import evaluation_header, iter_evaluation_episodes
    from silent_cascade.logging.crash_bundle import CrashBundleManifest

    evidence_context = _numeric_context(root.parent, evidence_context)
    with evaluation_header(root, evidence_context=evidence_context) as (header, _):
        pass
    rows = []
    for episode in iter_evaluation_episodes(
        root, evidence_context=evidence_context, expected_header=header
    ):
        rows.append(episode.row)
    identity = header.identity
    rows = tuple(rows)
    hashes = header.hashes
    if (
        identity.experiment != "phase4-task10-diagnostic"
        or identity.purpose != "action_diagnostic"
        or identity.manifest_sha256 != subset.parent_manifest_sha256
        or identity.episodes != tuple(manifest.entries[i].projected for i in subset.indices)
        or identity.checkpoint_sha256 != weights[0]
        or identity.model_identity != weights[1]
        or identity.evaluation_config_canonical_json.encode() != config.canonical_json
        or identity.execution_source_revision != manifest.source_commit
        or identity.stage != "primary"
        or identity.split != manifest.split
    ):
        raise ValueError("raw runtime identity differs")
    required = {"identity.json", "retention.json", "rows.jsonl", "index.json", "metrics.json"}
    if any(row.causal_trace_sha256 is not None for row in rows):
        required.add(f"crashes/weights-{identity.checkpoint_sha256}.safetensors")
    required.update(f"episodes/{i:05d}.telemetry.json" for i in range(len(rows)))
    required.update(row.neural_trace_ref for row in rows)
    required.update(row.full_trace_ref for row in rows if row.full_trace_ref is not None)
    if header.crashes:
        required.add("crashes/index.json")
        for public_id, entry in header.crashes.items():
            required.add(entry["path"])
            crash = CrashBundleManifest.model_validate_json(
                canonical_json_bytes(
                    _numeric_json(root / entry["path"], evidence_context=evidence_context)
                )
            )
            if crash.context.episode_public_id != public_id:
                raise ValueError("raw crash episode identity differs")
            reference = crash.context.checkpoint_ref
            if reference is not None:
                if reference != f"{crash.bundle_id}.safetensors":
                    raise ValueError("raw crash checkpoint reference differs")
                required.add("crashes/" + reference)
    if set(hashes) != required:
        raise ValueError("runtime artifact closure differs")
    return required | {"DONE"}


def _read_capture_operations(path, *, model, config, evidence_context=None):
    """Consume saved observations, without executing a model or training step."""
    import math

    from silent_cascade.eval.compute import NeuralComputeSnapshot
    from silent_cascade.train.trainer import StepResult

    value = _numeric_json(path.with_suffix(".json"), evidence_context=evidence_context)
    if not isinstance(value, dict) or set(value) != {"update", "extra_diagnostic_forward"}:
        raise ValueError("raw operation record schema differs")
    update = value["update"]
    if not isinstance(update, dict) or set(update) != {field.name for field in fields(StepResult)}:
        raise ValueError("raw operation update schema differs")

    def compute(raw, *, backward):
        if not isinstance(raw, dict) or set(raw) != {
            field.name for field in fields(NeuralComputeSnapshot)
        }:
            raise ValueError("raw operation compute schema differs")
        for name, observed in raw.items():
            if name in {"module_calls", "operation_estimates"}:
                valid = isinstance(observed, dict) and all(
                    isinstance(k, str) and k and type(v) is int and v >= 0
                    for k, v in observed.items()
                )
            elif name == "elapsed_seconds":
                valid = type(observed) in (int, float) and math.isfinite(observed) and observed >= 0
            elif name == "mps_peak_allocation_bytes" and observed is None:
                valid = True
            else:
                valid = type(observed) is int and observed >= 0
            if not valid:
                raise ValueError("raw operation counter type/range differs: " + name)
        if (
            raw["estimated_macs"] != raw["forward_macs"]
            or raw["parameters"] != sum(p.numel() for p in model.parameters())
            or raw["foundation_model_calls"] != 0
            or (raw["backward_macs"] > 0) is not backward
            or raw["eligible_records"] > raw["records_scored"]
        ):
            raise ValueError("raw operation compute identity/scope differs")
        return NeuralComputeSnapshot(**raw)

    if (
        update["objective_version"] != config.pilot.objective_version
        or update["auxiliary_coefficient"] != config.pilot.content_auxiliary_weight
    ):
        raise ValueError("raw operation objective identity differs")
    parsed = dict(update)
    for name in ("compute", "timed_compute", "content_compute"):
        parsed[name] = compute(update[name], backward=name == "compute")
    extra = compute(value["extra_diagnostic_forward"], backward=False)
    ignored = {"backward_macs", "elapsed_seconds", "mps_peak_allocation_bytes"}
    for name in value["extra_diagnostic_forward"].keys() - ignored:
        if update["compute"][name] != value["extra_diagnostic_forward"][name]:
            raise ValueError("raw operation extra-forward identity differs: " + name)
        left, right = update["timed_compute"][name], update["content_compute"][name]
        if name in {"module_calls", "operation_estimates"}:
            expected = dict(Counter(left) + Counter(right))
        elif name in {"parameters", "memory_bytes"}:
            expected = max(left, right)
        else:
            expected = left + right
        if update["compute"][name] != expected:
            raise ValueError("raw operation branch aggregate differs: " + name)

    def finite_numbers(observed):
        if isinstance(observed, list):
            return all(finite_numbers(v) for v in observed)
        return type(observed) in (int, float) and math.isfinite(observed)

    for name in set(update) - {"objective_version", "compute", "timed_compute", "content_compute"}:
        observed = update[name]
        if name in {"terms", "subterms", "numerators", "denominators", "per_position"}:
            valid = isinstance(observed, dict) and all(
                isinstance(k, str)
                and bool(k)
                and (isinstance(v, list) if name == "per_position" else type(v) in (int, float))
                and finite_numbers(v)
                for k, v in observed.items()
            )
        else:
            valid = type(observed) in (int, float) and math.isfinite(observed)
        if not valid:
            raise ValueError("raw operation nonfinite/invalid update field: " + name)
    if update["gradient_norm"] < 0:
        raise ValueError("raw operation negative gradient norm")
    return StepResult(**parsed), extra


def _read_capture(path, *, model, config, evidence_context=None):
    from safetensors.torch import load

    from silent_cascade.train.pilot_data import _read_pilot_bytes

    with _numeric_path(
        path.with_suffix(".safetensors"), evidence_context=evidence_context
    ) as local:
        flat = load(_read_pilot_bytes(local))
    tensors = {group: {} for group in _GROUPS}
    for name, value in flat.items():
        group, separator, key = name.partition("/")
        if group not in tensors or not separator:
            raise ValueError("raw tensor inventory differs")
        tensors[group]["/" + key] = value
    expected = expected_inventory(model)
    for group, values in tensors.items():
        if tuple(sorted(values)) != expected[group]:
            raise ValueError("raw tensor inventory differs")
        for name, value in values.items():
            if value.dtype != _capture_dtype(group, name):
                raise ValueError(f"raw tensor dtype differs: {name}")
            if value.is_floating_point():
                if value.dtype != torch.float32:
                    raise ValueError("raw tensor dtype differs")
                valid = torch.isfinite(value)
                if group == "forward" and name == "/timed/crossing_offsets":
                    valid |= torch.isposinf(value)
                if group == "forward" and name.endswith("/retrieval_logits"):
                    valid |= (
                        torch.isneginf(value) & ~values[name.replace("logits", "eligible_mask")]
                    )
                if not bool(valid.all()):
                    raise ValueError("raw tensor contains nonfinite values")
    step, extra = _read_capture_operations(
        path, model=model, config=config, evidence_context=evidence_context
    )
    result = CapturedUpdate(tensors, step, extra)
    compare_update(
        result, result, expected=expected, resume_device="cpu"
    ).validate_complete_inventory(model)
    return result


def _optimizer_groups(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    return [
        {
            **{k: v for k, v in group.items() if k != "params"},
            "params": [names[id(p)] for p in group["params"]],
        }
        for group in optimizer.param_groups
    ]


def _optimizer_tensors(model, optimizer):
    return _tensors({n: optimizer.state.get(p, {}) for n, p in model.named_parameters()})


def _exact_tensors(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[n], right[n]) for n in left)


def _bind_resume_start(
    root,
    *,
    session,
    initial,
    model,
    config,
    device,
    observations,
    evidence_context=None,
):
    """Artifact-only starting-state and bootstrap arithmetic consistency.

    Saved gradients are execution evidence, not independently regenerated here.
    Only configured AdamW/clipping arithmetic runs on a disposable CPU clone;
    no forward, training batch, event engine or persisted model update is allowed.
    """
    from silent_cascade.train.pilot_state import PilotProgress
    from silent_cascade.train.trainer import make_optimizer

    if initial is None:
        starting_model = model
        starting_optimizer = make_optimizer(starting_model, config.config.pilot)
        progress = PilotProgress()
        rng = _decode_observed_rng(observations["initial"])
    else:
        starting_model, starting_optimizer = initial.model, initial.optimizer
        progress, rng = initial.progress, initial.rng
    if device == "mps" and rng.torch_mps_state is None:
        raise ValueError("raw starting input lacks native RNG evidence")
    initial_hash = _rng_hash(_continuation_rng(rng, device))
    if initial_hash != _observed_rng_hash(
        observations["initial"]
    ) or initial_hash != _observed_rng_hash(observations["before"]):
        # The existing deterministic objective/bootstrap consumes no global RNG.
        raise ValueError("raw starting input/bootstrap RNG differs")
    expected_model, expected_optimizer = clone_training_state(
        starting_model, starting_optimizer, "cpu"
    )
    if not starting_optimizer.state:
        bootstrap = _read_capture(
            root / "bootstrap",
            model=model,
            config=config.config,
            evidence_context=evidence_context,
        )
        for name, parameter in expected_model.named_parameters():
            parameter.grad = bootstrap.tensors["gradients"]["/" + name].clone()
        torch.nn.utils.clip_grad_norm_(expected_model.parameters(), 1.0, error_if_nonfinite=True)
        expected_optimizer.step()
        expected_model.train()
        rtol, atol = (0.0, 0.0) if device == "cpu" else (1e-4, 1e-5)
        for group, actual in (
            ("parameters", _tensors(expected_model.state_dict())),
            ("optimizer", _optimizer_tensors(expected_model, expected_optimizer)),
        ):
            if not _compare(actual, bootstrap.tensors[group], rtol=rtol, atol=atol).passed:
                raise ValueError("raw bootstrap arithmetic differs from input state")
        if not _exact_tensors(
            _tensors(session.model.state_dict()), bootstrap.tensors["parameters"]
        ) or not _exact_tensors(
            _optimizer_tensors(session.model, session.optimizer), bootstrap.tensors["optimizer"]
        ):
            raise ValueError("raw bootstrap output differs from resume archive")
        progress = progress.model_copy(
            update={
                "global_step": progress.global_step + 1,
                "batch_counter": progress.batch_counter + 1,
            }
        )
    elif not _exact_tensors(
        _tensors(expected_model.state_dict()), _tensors(session.model.state_dict())
    ) or not _exact_tensors(
        _optimizer_tensors(expected_model, expected_optimizer),
        _optimizer_tensors(session.model, session.optimizer),
    ):
        raise ValueError("raw starting input model/optimizer differs")
    if (
        session.progress != progress
        or session.model.training != expected_model.training
        or canonical_json_bytes({"groups": _optimizer_groups(session.model, session.optimizer)})
        != canonical_json_bytes({"groups": _optimizer_groups(expected_model, expected_optimizer)})
    ):
        raise ValueError("raw starting input progress/mode/optimizer groups differ")


def _read_resume(
    root,
    *,
    expected_report,
    model,
    config,
    source,
    initial=None,
    evidence_context=None,
):
    from silent_cascade.train.pilot_data import next_pilot_batch

    session, digest, archive = _read_checked_archive(
        root / "resume.safetensors",
        config=config,
        source=source,
        evidence_context=evidence_context,
    )
    if archive.source != source or archive.config_json.encode() != config.canonical_json:
        raise ValueError("raw resume source/config differs")
    observations = _numeric_json(root / "observations.json", evidence_context=evidence_context)
    if set(observations) != {
        "initial",
        "before",
        "after",
        "restored_before",
        "restored_after",
        "draws",
        "restored_draws",
    }:
        raise ValueError("raw resume observation inventory differs")
    for name in ("initial", "before", "after", "restored_before", "restored_after"):
        if (_decode_observed_rng(observations[name]).torch_mps_state is not None) != (
            expected_report.device == "mps"
        ):
            raise ValueError("raw resume RNG scope differs from device")
    if _rng_hash(_continuation_rng(session.rng, expected_report.device)) != _observed_rng_hash(
        observations["before"]
    ):
        raise ValueError("raw resume RNG differs from archive")
    _bind_resume_start(
        root,
        session=session,
        initial=initial,
        model=model,
        config=config,
        device=expected_report.device,
        observations=observations,
        evidence_context=evidence_context,
    )
    batch = next_pilot_batch(
        config.config, stage=archive.progress.stage, batch_counter=archive.progress.batch_counter
    )
    captures = [
        _read_capture(
            root / name,
            model=model,
            config=config.config,
            evidence_context=evidence_context,
        )
        for name in ("expected", "restored")
    ]
    comparison = compare_update(
        *captures,
        expected=expected_inventory(model),
        resume_device=expected_report.device,
    )
    if _numeric_exists(root / "bootstrap.json", evidence_context):
        captures.append(
            _read_capture(
                root / "bootstrap",
                model=model,
                config=config.config,
                evidence_context=evidence_context,
            )
        )
    calls = sum(
        c.step.compute.foundation_model_calls + c.diagnostic_forward_compute.foundation_model_calls
        for c in captures
    )
    recomputed = expected_report.model_copy(
        update={
            "diagnostic_updates": len(captures),
            "foundation_model_calls": calls,
            "checkpoint_sha256": digest,
            "archive_global_step": archive.progress.global_step,
            "next_batch_counter": batch.next_batch_counter,
            "next_batch_hashes": batch.example_hashes,
            "restored_batch_hashes": batch.example_hashes,
            "rng_before_sha256": _observed_rng_hash(observations["before"]),
            "restored_rng_sha256": _observed_rng_hash(observations["restored_before"]),
            "rng_after_sha256": _observed_rng_hash(observations["after"]),
            "restored_rng_after_sha256": _observed_rng_hash(observations["restored_after"]),
            "random_draw_sha256": sha256_bytes(canonical_json_bytes(observations["draws"])),
            "restored_random_draw_sha256": sha256_bytes(
                canonical_json_bytes(observations["restored_draws"])
            ),
            "comparison": comparison,
        }
    )
    if recomputed != expected_report or _numeric_json(
        root / "resume.json", evidence_context=evidence_context
    ) != expected_report.model_dump(mode="json"):
        raise ValueError("raw resume summary differs")
    return len(captures), calls


def _read_checked_archive(path, *, config, source, evidence_context=None):
    """Use the full archive validator without changing the reader's global RNG."""
    before = snapshot_global_rng()
    try:
        with _numeric_path(path, evidence_context=evidence_context) as local:
            return load_numeric_checkpoint(local, config=config, source=source, device="cpu")
    except Exception as error:
        raise ValueError("raw full training archive validation failed") from error
    finally:
        restore_rng_snapshot(before, restore_mps=before.torch_mps_state is not None)


def read_numeric_report(output_dir, *, config, source_commit, evidence_context=None):

    from silent_cascade.eventflow.neural_weights import load_neural_weights, read_bytes
    from silent_cascade.report.pilot_artifacts import _decode_json
    from silent_cascade.train.pilot_data import PilotManifest

    evidence_context = _numeric_context(output_dir, evidence_context)
    with _numeric_path(
        output_dir / "numeric-report.json", evidence_context=evidence_context
    ) as local:
        raw = read_bytes(local)
    if _numeric_json(output_dir / "DONE", evidence_context=evidence_context) != {
        "report_sha256": sha256_bytes(raw)
    }:
        raise ValueError("numeric report artifact hash differs")
    report = PilotNumericReport.model_validate_json(raw)
    if (
        canonical_json_bytes(report) != raw
        or report.config_sha256 != config.sha256
        or report.config_canonical_json.encode() != config.canonical_json
    ):
        raise ValueError("numeric report config differs")
    if report.source.source_commit != source_commit:
        raise ValueError("numeric report source differs")
    required = {
        "intent.json",
        "unchanged-weights.safetensors",
        "manifest.json",
        "runtime-subset.json",
        "one_hop-batch.json",
        "primary-batch.json",
    }
    devices = ("cpu",) if report.missing_devices else ("cpu", "mps")
    for device in devices:
        required.update(
            f"{stage}-{device}.{ext}"
            for stage in ("one_hop", "primary")
            for ext in ("json", "safetensors")
        )
        required.update(
            f"resume-{device}/{name}"
            for name in (
                "resume.safetensors",
                "resume.json",
                "observations.json",
                "expected.json",
                "expected.safetensors",
                "restored.json",
                "restored.safetensors",
            )
        )
    if report.original_checkpoint_sha256 is not None:
        required.add("input-training.safetensors")
    initial = None
    resume_updates = 3
    if report.original_checkpoint_sha256 is not None:
        initial, digest, archived = _read_checked_archive(
            output_dir / "input-training.safetensors",
            config=config,
            source=report.source,
            evidence_context=evidence_context,
        )
        if digest != report.original_checkpoint_sha256:
            raise ValueError("raw input archive identity differs")
        resume_updates = 2 if archived.optimizer_names else 3
    if resume_updates == 3:
        required.update(
            f"resume-{device}/bootstrap.{ext}"
            for device in devices
            for ext in ("json", "safetensors")
        )
    if not required <= report.artifact_hashes.keys():
        raise ValueError("numeric artifact closure omits required capture")
    with _numeric_path(
        output_dir / "unchanged-weights.safetensors", evidence_context=evidence_context
    ) as local:
        weights = load_neural_weights(
            local,
            expected_sha256=report.portable_weights_sha256,
            device="cpu",
        )
    if (
        weights.identity.model_state_sha256 != report.initial_model_sha256
        or weights.identity.source_revision != source_commit
    ):
        raise ValueError("raw unchanged model identity differs")
    with _numeric_path(output_dir / "manifest.json", evidence_context=evidence_context) as local:
        manifest = PilotManifest.model_validate_json(read_bytes(local))
    if (
        manifest.config_hash != config.sha256
        or manifest.source_commit != source_commit
        or manifest.stage != "primary"
    ):
        raise ValueError("raw manifest identity differs")
    report.runtime_subset.validate_manifest(manifest)
    if _numeric_json(
        output_dir / "runtime-subset.json", evidence_context=evidence_context
    ) != report.runtime_subset.model_dump(mode="json") or report.runtime_episodes != (
        64 if config.config.pilot.is_production else 16
    ):
        raise ValueError("raw runtime subset differs")
    for device in devices:
        required.update(
            f"runtime-{device}/{name}"
            for name in _runtime_artifact_inventory(
                output_dir / f"runtime-{device}",
                config=config,
                manifest=manifest,
                subset=report.runtime_subset,
                weights=(report.portable_weights_sha256, weights.identity),
                evidence_context=evidence_context,
            )
        )
    _check_artifact_closure(
        output_dir,
        report.artifact_hashes,
        required,
        report_name="numeric-report.json",
        evidence_context=evidence_context,
    )
    if (
        report.original_checkpoint_sha256 is not None
        and archived.weights.identity != weights.identity
    ):
        raise ValueError("raw input archive identity differs")
    if (
        report.diagnostic_updates != len(devices) * (2 + resume_updates)
        or report.extra_diagnostic_forwards != (2 + resume_updates) * len(devices)
        or report.cpu_resume.diagnostic_updates != resume_updates
        or (
            report.mps_resume is not None and report.mps_resume.diagnostic_updates != resume_updates
        )
    ):
        raise ValueError("raw diagnostic operation count differs")
    observed_updates = observed_forwards = observed_calls = 0
    for stage in ("one_hop", "primary"):
        batch = diagnostic_batch(config.config, stage=stage, counter=0)
        recipe_path = output_dir / f"{stage}-batch.json"
        with _numeric_path(recipe_path, evidence_context=evidence_context) as local:
            recipe_raw = read_bytes(local)
        recipe = _decode_json(recipe_raw)
        if report.batches[stage] != batch.example_hashes or canonical_json_bytes(
            recipe
        ) != canonical_json_bytes(
            {
                "keys": [asdict(k) for k in batch.example_keys],
                "example_hashes": batch.example_hashes,
                "teacher_trace_steps": batch.targets.kind.shape[1],
                "stage": stage,
                "split": "debug",
                "primary_path_filter": 4 if stage == "primary" else None,
            }
        ):
            raise ValueError("raw diagnostic batch recipe differs")
        if report.batch_recipe_sha256s[stage] != sha256_bytes(recipe_raw):
            raise ValueError("raw batch recipe hash differs")
        captures = {
            d: _read_capture(
                output_dir / f"{stage}-{d}",
                model=weights.model,
                config=config.config,
                evidence_context=evidence_context,
            )
            for d in devices
        }
        observed_updates += len(captures)
        observed_forwards += len(captures)
        observed_calls += sum(
            c.step.compute.foundation_model_calls
            + c.diagnostic_forward_compute.foundation_model_calls
            for c in captures.values()
        )
        if (
            compare_update(
                captures["cpu"],
                captures["cpu"],
                expected=expected_inventory(weights.model),
                resume_device="cpu",
            )
            != report.cpu_inventory[stage]
        ):
            raise ValueError("raw CPU comparison summary differs")
        if (
            "mps" in devices
            and compare_update(
                captures["cpu"], captures["mps"], expected=expected_inventory(weights.model)
            )
            != report.device_parity[stage]
        ):
            raise ValueError("raw device comparison summary differs")
    for device in devices:
        count, calls = _read_resume(
            output_dir / f"resume-{device}",
            expected_report=report.cpu_resume if device == "cpu" else report.mps_resume,
            model=weights.model,
            config=config,
            source=report.source,
            initial=initial,
            evidence_context=evidence_context,
        )
        observed_updates += count
        observed_forwards += count
        observed_calls += calls
    if (observed_updates, observed_forwards, observed_calls) != (
        report.diagnostic_updates,
        report.extra_diagnostic_forwards,
        report.foundation_model_calls,
    ):
        raise ValueError("raw observed operation totals differ")
    if (
        "mps" in devices
        and compare_runtime(
            _runtime_rows(output_dir / "runtime-cpu", evidence_context=evidence_context),
            _runtime_rows(output_dir / "runtime-mps", evidence_context=evidence_context),
        )
        != report.runtime_comparison
    ):
        raise ValueError("raw runtime comparison summary differs")
    return report


class EvaluationTrial(StrictModel):
    device: Literal["cpu", "mps"]
    episodes: int = Field(ge=1)
    elapsed_seconds: float = Field(gt=0)
    mean_episode_seconds: float = Field(gt=0)
    event_counts: dict[str, int]
    miss_counts: dict[str, int]
    error_episodes: int = Field(ge=0)
    forward_macs: int = Field(gt=0)
    foundation_model_calls: Literal[0] = 0
    retained_raw_bytes: int = Field(gt=0)
    largest_episode_bytes: int = Field(gt=0)
    process_highwater_bytes: int | None
    tensor_working_memory_bytes: int = Field(ge=0)
    component_episodes: int = Field(ge=1)
    component_elapsed_seconds: float = Field(gt=0)
    component_raw_bytes: int = Field(gt=0)


class PilotThroughputReport(StrictModel):
    schema_version: Literal["phase4-throughput-diagnostic-v1"] = "phase4-throughput-diagnostic-v1"
    evidence_kind: Literal["throughput_diagnostic"] = "throughput_diagnostic"
    config_canonical_json: str
    config_sha256: str
    source: PilotSourceIdentity
    model_seed: Literal[11] = 11
    initial_model_sha256: str
    portable_weights_sha256: str
    production_workload_measured: bool
    training_trials: dict[str, TrainingTrial]
    evaluation_trials: dict[str, EvaluationTrial]
    manifest_subset: DiagnosticSubset
    numeric_report_sha256: str
    numerical_checks_passed: bool
    chosen_device: Literal["cpu", "mps"]
    device_choice_reason: str
    final_acceptance_device: Literal["cpu"] = "cpu"
    missing_devices: tuple[Literal["mps"], ...]
    checkpoint_bytes: int = Field(gt=0)
    retained_sample_bytes: int = Field(gt=0)
    destination: str
    destination_is_system_temp: bool
    forecast: WorkloadForecast
    environment: dict[str, str]
    foundation_model_calls: Literal[0] = 0
    artifact_hashes: dict[str, str]

    @model_validator(mode="after")
    def counts(self):
        from silent_cascade.train.pilot_config import parse_phase4_canonical

        config = parse_phase4_canonical(self.config_canonical_json)
        if (
            self.config_sha256 != sha256_bytes(self.config_canonical_json.encode())
            or self.source.config_sha256 != self.config_sha256
        ):
            raise ValueError("throughput config/source differs")
        if self.production_workload_measured != config.pilot.is_production:
            raise ValueError("throughput workload profile differs")
        devices = {"cpu"} if self.missing_devices else {"cpu", "mps"}
        if (
            set(self.training_trials)
            != {f"{device}/{stage}" for device in devices for stage in ("one_hop", "primary")}
            or set(self.evaluation_trials) != devices
        ):
            raise ValueError("throughput device/stage inventory differs")
        for trial in self.training_trials.values():
            expected = (128, 16, 32) if config.pilot.is_production else (8, 1, 2)
            if (trial.batch_size, trial.warmup_updates, trial.measured_updates) != expected:
                raise ValueError("throughput update recipe differs")
            if trial.initial_model_sha256 != self.initial_model_sha256:
                raise ValueError("throughput starting weights differ")
        count = 256 if config.pilot.is_production else 16
        if len(self.manifest_subset.indices) != count or any(
            t.episodes != count for t in self.evaluation_trials.values()
        ):
            raise ValueError("throughput runtime episode inventory differs")
        if self.chosen_device == "mps" and (
            self.missing_devices or not self.numerical_checks_passed
        ):
            raise ValueError("MPS fit choice requires measured numerical checks")
        return self


def profile_pilot(
    config: ResolvedConfig[Phase4Config], *, output_dir: Path
) -> PilotThroughputReport:
    from silent_cascade.train.pilot_measurement import profile_pilot as profile

    return profile(config, output_dir=output_dir)


def read_throughput_report(output_dir, *, config, source_commit):
    import json

    from silent_cascade.eventflow.neural_weights import read_bytes

    raw = read_bytes(output_dir / "throughput-report.json")
    if json.loads(read_bytes(output_dir / "DONE"))["report_sha256"] != sha256_bytes(raw):
        raise ValueError("throughput report artifact hash differs")
    report = PilotThroughputReport.model_validate_json(raw)
    if (
        canonical_json_bytes(report) != raw
        or report.config_sha256 != config.sha256
        or report.config_canonical_json.encode() != config.canonical_json
    ):
        raise ValueError("throughput report config differs")
    if report.source.source_commit != source_commit:
        raise ValueError("throughput report source differs")
    for name, digest in report.artifact_hashes.items():
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or sha256_bytes(read_bytes(output_dir / name)) != digest
        ):
            raise ValueError("throughput artifact hash differs")
    from silent_cascade.train.pilot_measurement import verify_profile_raw

    verify_profile_raw(report, output_dir, config=config, source_commit=source_commit)
    return report
