"""Executable numerical evidence; callers inspect measured mismatches, never log text."""

import json
import os
import platform
import random
import subprocess
import sys
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import torch
from pydantic import Field

from silent_cascade.config import ResolvedConfig
from silent_cascade.eventflow.checkpoint_rng import restore_rng_snapshot
from silent_cascade.models.dynamics import crossings_batch
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.rng import seed_all, snapshot_global_rng
from silent_cascade.train.batches import TrainingBatch, next_training_batch
from silent_cascade.train.checkpoints import (
    _device,
    _model_hash,
    _snapshot,
    load_training_checkpoint,
    save_training_checkpoint,
)
from silent_cascade.train.component_eval import predict_components
from silent_cascade.train.config import Phase3Config, TrainingConfig
from silent_cascade.train.objective import training_objective
from silent_cascade.train.state import TrainingError, TrainProgress
from silent_cascade.train.trainer import _check_objective, make_optimizer, train_one_step
from silent_cascade.validation import StrictModel


class Comparison(StrictModel):
    rtol: float
    atol: float
    compared: int = Field(ge=0)
    mismatches: int = Field(ge=0)
    max_absolute_error: float = Field(ge=0)
    max_relative_error: float = Field(ge=0)
    max_absolute_name: str
    max_relative_name: str
    tested_names: tuple[str, ...]
    failed_names: tuple[str, ...]
    tensor_elements: dict[str, int]

    @property
    def passed(self) -> bool:
        return self.compared > 0 and self.mismatches == 0


class DeviceParityEvidence(StrictModel):
    schema_version: Literal["phase3-device-parity-v2"] = "phase3-device-parity-v2"
    objective_version: Literal["teacher_timed_v1", "teacher_timed_plus_content_v2"]
    auxiliary_coefficient: float
    optimizer_options: dict[str, object]
    devices: tuple[Literal["cpu"], Literal["mps"]] = ("cpu", "mps")
    output_dtype: Literal["float32"] = "float32"
    python_version: str
    torch_version: str
    threads: int
    model_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    example_hashes: tuple[str, ...]
    forward: Comparison
    losses: Comparison
    gradients: Comparison
    updated_weights: Comparison
    recalled_record_mismatches: int
    action_class_mismatches: int
    active_crossings: int
    dormant_crossings: int
    empty_memory_rows: int
    foundation_model_calls: Literal[0] = 0

    @property
    def passed(self) -> bool:
        return (
            all(c.passed for c in (self.forward, self.losses, self.gradients, self.updated_weights))
            and self.recalled_record_mismatches == self.action_class_mismatches == 0
            and self.active_crossings > 0
            and self.dormant_crossings > 0
            and self.empty_memory_rows > 0
        )


class ResumeEvidence(StrictModel):
    schema_version: Literal["phase3-cpu-resume-v2"] = "phase3-cpu-resume-v2"
    objective_version: Literal["teacher_timed_v1", "teacher_timed_plus_content_v2"]
    auxiliary_coefficient: float
    optimizer_options: dict[str, object]
    device: Literal["cpu"] = "cpu"
    python_version: str
    torch_version: str
    threads: int
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resumed_step: Literal[2] = 2
    next_batch_counter: Literal[2] = 2
    example_hashes: tuple[str, ...]
    first_example_hashes: tuple[str, ...]
    parameters: Comparison
    optimizer: Comparison
    losses: Comparison
    rng_mismatches: int
    batch_hash_mismatches: int
    foundation_model_calls: Literal[0] = 0

    @property
    def passed(self) -> bool:
        return all(c.passed for c in (self.parameters, self.optimizer, self.losses)) and (
            self.rng_mismatches == self.batch_hash_mismatches == 0
        )


class MPSResumeEvidence(ResumeEvidence):
    schema_version: Literal["phase3-mps-resume-v2"] = "phase3-mps-resume-v2"
    device: Literal["mps"] = "mps"


def _tensors(value, prefix="") -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return {prefix: value.detach().cpu().clone()}
    if is_dataclass(value):
        items = ((field.name, getattr(value, field.name)) for field in fields(value))
    elif isinstance(value, Mapping):
        items = value.items()
    elif isinstance(value, (tuple, list)):
        items = enumerate(value)
    else:
        return {}
    result = {}
    for name, child in items:
        result.update(_tensors(child, f"{prefix}/{name}"))
    return result


def _compare(left, right, *, rtol, atol) -> Comparison:
    names = tuple(sorted(set(left) | set(right)))
    count = mismatches = 0
    absolute = relative = 0.0
    absolute_name = relative_name = ""
    failed = []
    for name in names:
        if name not in left or name not in right or left[name].shape != right[name].shape:
            mismatches += 1
            failed.append(name)
            continue
        a, b = left[name], right[name]
        count += a.numel()
        if a.dtype != b.dtype:
            mismatches += a.numel()
            failed.append(name)
            continue
        if a.is_floating_point():
            finite = torch.isfinite(a) & torch.isfinite(b)
            # Exactly equal signed infinity is valid for dormant guards/masked scores.
            equal = (a == b) | (finite & ((a - b).abs() <= atol + rtol * a.abs()))
            delta = (a[finite].double() - b[finite].double()).abs()
            rel = delta / a[finite].double().abs().clamp_min(torch.finfo(torch.float32).tiny)
            maximum = float(delta.max()) if delta.numel() else 0.0
            maximum_relative = float(rel.max()) if rel.numel() else 0.0
            if maximum > absolute:
                absolute, absolute_name = maximum, name
            if maximum_relative > relative:
                relative, relative_name = maximum_relative, name
        else:
            equal = a == b
        wrong = int((~equal).sum())
        mismatches += wrong
        if wrong:
            failed.append(name)
    return Comparison(
        rtol=rtol,
        atol=atol,
        compared=count,
        mismatches=mismatches,
        max_absolute_error=absolute,
        max_relative_error=relative,
        max_absolute_name=absolute_name,
        max_relative_name=relative_name,
        tested_names=names,
        failed_names=tuple(failed),
        tensor_elements={name: value.numel() for name, value in left.items()},
    )


def _parity_step(model, batch, device, training=None):
    current = deepcopy(model).to(device)
    batch = batch.to(device)
    training = training or TrainingConfig(
        profile="phase3_smoke",
        batch_size=8,
        max_steps=4,
        validation_every_steps=2,
        fixed_validation_episodes=16,
    )
    optimizer = make_optimizer(current, training)
    optimizer.zero_grad(set_to_none=True)
    objective = training_objective(current, batch, training)
    _check_objective(objective)
    if objective.total.dtype != torch.float32 or objective.total.device.type != device:
        raise TrainingError("parity output must be float32 on the requested native device")
    forward = _tensors({"timed": objective.timed, "content": objective.content})
    # Exercise true empty-memory preview/control and explicit active/dormant guards.
    empty = current.initial_context(1, device=device)
    forward.update(_tensors(current.preview_and_control(empty), "empty"))
    offsets = crossings_batch(
        torch.zeros((1, 3), device=device),
        torch.tensor([[1.5, 0.5, 1.0]], device=device),
        torch.ones((1, 3), device=device),
    )
    forward.update(_tensors(offsets, "guard_probe"))
    loss_values = _tensors(
        {"timed": objective.timed_loss, "content": objective.content_loss, "total": objective.total}
    )
    predictions = predict_components(current, batch.public)
    objective.total.backward()
    gradients = _tensors({n: p.grad for n, p in current.named_parameters() if p.grad is not None})
    torch.nn.utils.clip_grad_norm_(
        current.parameters(), training.gradient_clip_norm, error_if_nonfinite=True
    )
    optimizer.step()
    weights = _tensors(dict(current.named_parameters()))
    return (
        forward,
        loss_values,
        gradients,
        weights,
        predictions,
        (~empty.eligibility.any(dim=1)).cpu(),
        {key: value for key, value in optimizer.param_groups[0].items() if key != "params"},
    )


def measure_device_parity(
    model: EventFlowModel, batch: TrainingBatch, *, training: TrainingConfig | None = None
) -> DeviceParityEvidence:
    """Compare identical CPU-initialized weights through forward/backward/AdamW on native MPS."""
    _device("mps")
    if not isinstance(model, EventFlowModel) or not isinstance(batch, TrainingBatch):
        raise TypeError("parity requires EventFlowModel and TrainingBatch")
    training = training or TrainingConfig(
        profile="phase3_smoke",
        batch_size=8,
        max_steps=4,
        validation_every_steps=2,
        fixed_validation_episodes=16,
    )
    cpu_model = deepcopy(model).cpu()
    tensors, aliases, _ = _snapshot(cpu_model)
    cpu = _parity_step(cpu_model, batch, "cpu", training)
    mps = _parity_step(cpu_model, batch, "mps", training)
    comparisons = [
        _compare(cpu[i], mps[i], rtol=1e-4 if i < 2 else 1e-3, atol=1e-5) for i in range(4)
    ]
    crossings = [v for k, v in cpu[0].items() if k.endswith("/crossings") or k == "guard_probe"]
    return DeviceParityEvidence(
        objective_version=training.objective_version,
        auxiliary_coefficient=training.content_auxiliary_weight,
        optimizer_options=cpu[6],
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        threads=torch.get_num_threads(),
        model_state_sha256=_model_hash(tensors, aliases),
        example_hashes=batch.example_hashes,
        forward=comparisons[0],
        losses=comparisons[1],
        gradients=comparisons[2],
        updated_weights=comparisons[3],
        recalled_record_mismatches=sum(
            a.record_ids != b.record_ids for a, b in zip(cpu[4].rows, mps[4].rows, strict=True)
        ),
        action_class_mismatches=sum(
            a.action_class != b.action_class for a, b in zip(cpu[4].rows, mps[4].rows, strict=True)
        ),
        active_crossings=sum(int(torch.isfinite(v).sum()) for v in crossings),
        dormant_crossings=sum(int(torch.isposinf(v).sum()) for v in crossings),
        empty_memory_rows=int((cpu[5] & mps[5]).sum()),
    )


def _optimizer_tensors(model, optimizer):
    return _tensors(
        {name: optimizer.state[parameter] for name, parameter in model.named_parameters()}
    )


def _draws():
    return random.random(), float(np.random.random()), torch.rand(5)


def measure_cpu_resume(config: ResolvedConfig[Phase3Config], source_commit: str) -> ResumeEvidence:
    """Execute two real batches, save after one, restore and exactly repeat the second."""
    return _measure_resume(config, source_commit, "cpu")


def measure_mps_resume(
    config: ResolvedConfig[Phase3Config], source_commit: str
) -> MPSResumeEvidence:
    """Measure native two-step continuation, including actual MPS random draws."""
    return _measure_resume(config, source_commit, "mps")


def _measure_resume(config, source_commit, device):
    _device(device)
    before = snapshot_global_rng()
    rtol, atol = (0.0, 0.0) if device == "cpu" else (1e-4, 1e-5)

    def draws():
        values = _draws()
        return values if device == "cpu" else (*values, torch.rand(5, device="mps").cpu())

    try:
        seed_all(config.config.training.model_seed)
        model = EventFlowModel(config.config.neural).to(device)
        optimizer = make_optimizer(model, config.config.training)
        first = next_training_batch(config.config, stage="one_hop", batch_counter=0).to(device)
        train_one_step(model, optimizer, first, config.config)
        progress = TrainProgress(
            optimizer_step=1,
            next_batch_counter=1,
            stage="one_hop",
            train_root_seed=311,
            train_public_id_seed=331,
            patience_counter=0,
            retained_checkpoints=(),
            validation_manifest_sha256="0" * 64,
        )
        with TemporaryDirectory(prefix="silent-cascade-resume-") as directory:
            path = Path(directory).resolve()
            saved = save_training_checkpoint(
                path, model, optimizer, progress, config=config, source_commit=source_commit
            )
            expected_draws = draws()
            expected_batch = next_training_batch(
                config.config, stage="one_hop", batch_counter=1
            ).to(device)
            expected_loss = train_one_step(model, optimizer, expected_batch, config.config)
            restored = load_training_checkpoint(
                path / saved.relative_path,
                expected_config_sha256=config.sha256,
                expected_source_commit=source_commit,
                device=device,
            )
            restored.restore_rng()
            actual_draws = draws()
            actual_batch = next_training_batch(
                config.config, stage="one_hop", batch_counter=restored.progress.next_batch_counter
            ).to(device)
            actual_loss = train_one_step(
                restored.model, restored.optimizer, actual_batch, config.config
            )

            def loss_tensors(result):
                values = {"total": torch.tensor(result.loss, dtype=torch.float64)}
                for name in ("terms", "subterms", "numerators", "denominators", "per_position"):
                    values.update(
                        {
                            f"{name}/{key}": torch.tensor(value, dtype=torch.float64)
                            for key, value in getattr(result, name).items()
                        }
                    )
                values["timed/total"] = torch.tensor(result.timed_loss, dtype=torch.float64)
                if result.content_loss is not None:
                    values["content/total"] = torch.tensor(result.content_loss, dtype=torch.float64)
                return values

            evidence_type = ResumeEvidence if device == "cpu" else MPSResumeEvidence
            return evidence_type(
                objective_version=config.config.training.objective_version,
                auxiliary_coefficient=config.config.training.content_auxiliary_weight,
                optimizer_options={
                    key: value
                    for key, value in restored.optimizer.param_groups[0].items()
                    if key != "params"
                },
                python_version=platform.python_version(),
                torch_version=str(torch.__version__),
                threads=torch.get_num_threads(),
                config_sha256=config.sha256,
                source_commit=source_commit,
                checkpoint_sha256=saved.file_sha256,
                model_state_sha256=saved.model_state_sha256,
                example_hashes=actual_batch.example_hashes,
                first_example_hashes=first.example_hashes,
                parameters=_compare(
                    _tensors(dict(model.named_parameters())),
                    _tensors(dict(restored.model.named_parameters())),
                    rtol=rtol,
                    atol=atol,
                ),
                optimizer=_compare(
                    _optimizer_tensors(model, optimizer),
                    _optimizer_tensors(restored.model, restored.optimizer),
                    rtol=rtol,
                    atol=atol,
                ),
                losses=_compare(
                    loss_tensors(expected_loss), loss_tensors(actual_loss), rtol=rtol, atol=atol
                ),
                rng_mismatches=sum(
                    a != b for a, b in zip(expected_draws[:2], actual_draws[:2], strict=True)
                )
                + sum(
                    int((a != b).sum())
                    for a, b in zip(expected_draws[2:], actual_draws[2:], strict=True)
                ),
                batch_hash_mismatches=sum(
                    a != b
                    for a, b in zip(
                        expected_batch.example_hashes, actual_batch.example_hashes, strict=True
                    )
                ),
            )
    finally:
        restore_rng_snapshot(before, restore_mps=before.torch_mps_state is not None)


def measure_offline_imports(training: TrainingConfig):
    """Fresh interpreter runs one real smoke update and public evaluation with denial hooks."""
    from silent_cascade.train.evidence_types import OfflineEvidence, decode_json

    root = Path(__file__).resolve().parents[3]
    if not isinstance(training, TrainingConfig):
        raise TypeError("offline probe requires the actual validated training recipe")
    program = r"""
import importlib.abc
import json
import socket
import sys
from pathlib import Path
attempts = {'imports': 0, 'network': 0}
blocked = ('mlx', 'mlx_vlm', 'transformers', 'qwen_vl_utils')
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked or 'qwen' in fullname.lower():
            attempts['imports'] += 1
            raise RuntimeError('forbidden optional model import')
def deny(*args, **kwargs):
    attempts['network'] += 1
    raise RuntimeError('network disabled during actual training/evaluation')
sys.meta_path.insert(0, Blocker())
socket.socket.connect = deny
socket.socket.connect_ex = deny
socket.create_connection = deny
socket.getaddrinfo = deny
import torch
from silent_cascade.config import resolve_config
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.train.config import Phase3Config
from silent_cascade.train.batches import next_training_batch
from silent_cascade.train.trainer import make_optimizer, train_one_step
from silent_cascade.train.component_eval import predict_components
paths = ('configs/base.yaml', 'configs/data/primary.yaml', 'configs/model/event_flow.yaml',
         'configs/model/neural_components.yaml', sys.argv[1])
config = resolve_config(Phase3Config, tuple(Path(p) for p in paths))
torch.manual_seed(11)
model = EventFlowModel(config.config.neural)
optimizer = make_optimizer(model, config.config.training)
batch = next_training_batch(config.config, stage='one_hop', batch_counter=0)
step = train_one_step(model, optimizer, batch, config.config)
predictions = predict_components(model, batch.public)
print(json.dumps(dict(schema_version='phase3-offline-import-probe-v2',
    objective_version=step.objective_version, auxiliary_coefficient=step.auxiliary_coefficient,
    config_sha256=config.sha256,
    training_steps=int(max(s['step'].item() for s in optimizer.state.values())),
    predicted_rows=len(predictions.rows), backward_macs=step.compute.backward_macs,
    blocked_import_attempts=attempts['imports'], network_attempts=attempts['network'],
    forbidden_modules=[n for n in sys.modules if n.split('.')[0] in blocked or 'qwen' in n.lower()],
    foundation_model_calls=step.compute.foundation_model_calls)))
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", program, training.smoke_overlay_path],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "1"},
        capture_output=True,
        timeout=120,
    )
    if completed.returncode != 0 or completed.stderr:
        raise TrainingError(
            "offline import/training probe failed",
            context={
                "exit_code": completed.returncode,
                "stderr": completed.stderr.decode()[-2000:],
            },
        )
    values = decode_json(completed.stdout)
    return OfflineEvidence.model_validate_json(json.dumps(values))
