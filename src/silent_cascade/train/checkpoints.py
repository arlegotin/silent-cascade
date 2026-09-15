"""Closed safetensors training archives, explicit RNG restore and run retention.

Save/export take an existing directory, publish a content-addressed filename,
and return its descriptor. Load takes that archive file. Training archives are
distinct from paused-world checkpoints. Index ownership is the exact absolute
run directory plus source/config, data seeds and validation manifest; moving an
archive remains valid for loading, but does not transfer deletion ownership.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import re
import stat
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import Field
from safetensors import SafetensorError
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from silent_cascade.config import ResolvedConfig
from silent_cascade.errors import SilentCascadeError
from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at
from silent_cascade.eventflow.checkpoint_rng import (
    RngMetadata,
    decode_rng,
    encode_rng,
    restore_rng_snapshot,
    validate_rng_restore,
)
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.rng import RngSnapshot, snapshot_global_rng
from silent_cascade.train.config import Phase3Config, parse_phase3_canonical
from silent_cascade.train.state import CheckpointDescriptor, TrainingError, TrainProgress
from silent_cascade.validation import JsonValue, StrictModel

MAX_TRAINING_BYTES = 256 * 1024 * 1024
MAX_WEIGHTS_BYTES = 32 * 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
MAX_METADATA_DEPTH = 64
_INDEX = "checkpoint-index.json"
_OPTIONS = {
    "lr",
    "betas",
    "eps",
    "weight_decay",
    "amsgrad",
    "maximize",
    "foreach",
    "capturable",
    "differentiable",
    "fused",
    "decoupled_weight_decay",
}


class _Environment(StrictModel):
    python: str = Field(max_length=128)
    system: str = Field(max_length=256)
    machine: str = Field(max_length=128)
    torch: str = Field(max_length=128)
    numpy: str = Field(max_length=128)
    safetensors: str = Field(max_length=128)
    package: str = Field(max_length=128)
    device: Literal["cpu", "mps"]
    threads: int = Field(ge=1, le=1024)
    deterministic_algorithms: bool


class _Group(StrictModel):
    names: tuple[str, ...] = Field(min_length=1, max_length=10000)
    options: dict[str, JsonValue]


class _Metadata(StrictModel):
    schema_version: Literal["phase3-training-checkpoint-v1", "phase3-model-weights-v1"]
    config_json: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    aliases: dict[str, str]
    training: bool
    environment: _Environment
    owner_directory: str = Field(min_length=1, max_length=4096)
    progress: TrainProgress | None
    groups: tuple[_Group, ...] = Field(max_length=10000)
    optimizer_names: tuple[str, ...] = Field(max_length=10000)
    rng: RngMetadata | None


class CheckpointIndex(StrictModel):
    """Latest and best-three pointers, durable evidence pins and owned inventory."""

    schema_version: Literal["phase3-checkpoint-index-v1"] = "phase3-checkpoint-index-v1"
    owner_directory: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    train_root_seed: int
    train_public_id_seed: int
    validation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    latest: CheckpointDescriptor
    best: tuple[CheckpointDescriptor, ...] = Field(max_length=3)
    protected: tuple[CheckpointDescriptor, ...] = Field(max_length=75001)
    owned: tuple[CheckpointDescriptor, ...] = Field(max_length=75001)


@dataclass(frozen=True, slots=True)
class TrainingRestore:
    """Fresh validated state. Global generators change only through restore_rng."""

    model: EventFlowModel
    optimizer: torch.optim.AdamW
    progress: TrainProgress
    rng: RngSnapshot
    config: ResolvedConfig[Phase3Config]
    descriptor: CheckpointDescriptor
    device: str

    def restore_rng(self) -> None:
        """Validate every generator before applying a transactional RNG restore."""
        with _errors():
            restore_rng_snapshot(self.rng, restore_mps=self.device == "mps")


@contextmanager
def _errors() -> Iterator[None]:
    try:
        yield
    except TrainingError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        KeyError,
        OverflowError,
        SilentCascadeError,
        SafetensorError,
    ) as error:
        raise TrainingError(f"invalid training archive: {error}") from error


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _expected_hash(value: str, length: int) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is None:
        raise TrainingError("expected provenance binding must be an exact lowercase hash")


def _json(raw: bytes) -> dict:
    """Bound size/depth and reject duplicate keys and non-JSON numbers pre-parse."""
    if len(raw) > MAX_METADATA_BYTES:
        raise TrainingError("metadata byte limit")
    text = raw.decode("utf-8")
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > MAX_METADATA_DEPTH:
                raise TrainingError("metadata nesting limit")
        elif char in "]}":
            depth -= 1

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise TrainingError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise TrainingError(f"invalid JSON number: {value}")

    result = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise TrainingError("metadata must be an object")
    return result


def _environment(device: str) -> _Environment:
    return _Environment(
        python=platform.python_version(),
        system=platform.platform(),
        machine=platform.machine(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        safetensors=version("safetensors"),
        package=version("silent-cascade"),
        device=device,
        threads=torch.get_num_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )


def _device(device: str) -> None:
    if device not in {"cpu", "mps"} or (device == "mps" and not torch.backends.mps.is_available()):
        raise TrainingError("requested device is unavailable or unsupported")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise TrainingError("silent MPS fallback is forbidden")


def _layout(model: EventFlowModel):
    unique = {"parameter/" + k: v for k, v in model.named_parameters(remove_duplicate=True)}
    unique.update({"buffer/" + k: v for k, v in model.named_buffers(remove_duplicate=True)})
    identities = {id(value): key for key, value in unique.items()}
    aliases = {
        name: identities[id(value)] for name, value in model.state_dict(keep_vars=True).items()
    }
    return unique, aliases


def _snapshot(model: EventFlowModel):
    unique, aliases = _layout(model)
    device = next(model.parameters()).device.type
    _device(device)
    if any(
        value.device.type != device
        or value.dtype not in {torch.float32, torch.bool, torch.int64}
        or (key.startswith("parameter/") and value.dtype != torch.float32)
        for key, value in unique.items()
    ):
        raise TrainingError("model requires one device, float32 parameters and typed buffers")
    tensors = {name: value.detach().cpu().contiguous().clone() for name, value in unique.items()}
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise TrainingError("nonfinite model state")
    return tensors, aliases, device


def _model_hash(tensors: dict[str, torch.Tensor], aliases: dict[str, str]) -> str:
    digest = hashlib.sha256(canonical_json_bytes(aliases))
    for name in sorted(k for k in tensors if k.startswith(("parameter/", "buffer/"))):
        value = tensors[name]
        digest.update(
            canonical_json_bytes(
                {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
            )
        )
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _options(options: dict, config: Phase3Config) -> dict:
    if set(options) != _OPTIONS:
        raise TrainingError("unknown or missing AdamW options")
    training = config.training
    expected = {
        "lr": training.learning_rate,
        "betas": list(training.betas),
        "eps": training.epsilon,
        "weight_decay": training.weight_decay,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "decoupled_weight_decay": True,
    }
    result = dict(options)
    if isinstance(result["betas"], tuple):
        result["betas"] = list(result["betas"])
    for key, expected_value in expected.items():
        value = result[key]
        if type(value) is not type(expected_value) or value != expected_value:
            raise TrainingError(f"AdamW option differs from config: {key}")
        if key == "betas" and any(type(item) is not float for item in value):
            raise TrainingError("AdamW betas require exact floats")
    return result


def _optimizer_snapshot(model, optimizer, progress, config):
    if type(optimizer) is not torch.optim.AdamW:
        raise TrainingError("only the closed AdamW optimizer is supported")
    by_id = {id(value): key for key, value in model.named_parameters()}
    groups, names = [], []
    for group in optimizer.param_groups:
        group_names = tuple(by_id[id(value)] for value in group["params"])
        names.extend(group_names)
        groups.append(
            _Group(
                names=group_names,
                options=_options(
                    {key: value for key, value in group.items() if key != "params"}, config
                ),
            )
        )
    if len(names) != len(set(names)) or set(names) != set(by_id.values()):
        raise TrainingError("AdamW groups must own every unique model parameter exactly once")
    tensors, state_names = {}, []
    for parameter, state in optimizer.state.items():
        name = by_id[id(parameter)]
        if not state:
            continue
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise TrainingError("unexpected AdamW state")
        state_names.append(name)
        for kind, tensor in state.items():
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
                raise TrainingError("AdamW state must be float32 tensors")
            if tensor.shape != (() if kind == "step" else parameter.shape):
                raise TrainingError("AdamW state shape mismatch")
            if tensor.device.type != ("cpu" if kind == "step" else parameter.device.type):
                raise TrainingError("AdamW state device mismatch")
            tensors[f"optimizer/{name}/{kind}"] = tensor.detach().cpu().contiguous().clone()
    _validate_moments(tensors, progress.optimizer_step)
    return tuple(groups), tuple(sorted(state_names)), tensors


def _validate_moments(tensors, global_step):
    for name, value in tensors.items():
        if not name.startswith("optimizer/"):
            continue
        if not bool(torch.isfinite(value).all()):
            raise TrainingError("nonfinite AdamW state")
        if name.endswith("/step"):
            step = float(value)
            if not 1 <= step <= global_step or not step.is_integer():
                raise TrainingError("invalid AdamW step")
        elif name.endswith("/exp_avg_sq") and bool((value < 0).any()):
            raise TrainingError("negative AdamW second moment")


def _config(config: ResolvedConfig[Phase3Config]) -> Phase3Config:
    parsed = parse_phase3_canonical(config.canonical_json.decode("utf-8"))
    if parsed != config.config or _sha(config.canonical_json) != config.sha256:
        raise TrainingError("inconsistent resolved config")
    return parsed


def _progress(progress: TrainProgress, config: Phase3Config) -> None:
    TrainProgress.model_validate_json(progress.model_dump_json())
    if (progress.train_root_seed, progress.train_public_id_seed) != (
        config.training.train_root_seed,
        config.training.train_public_id_seed,
    ):
        raise TrainingError("progress data seeds differ from config")
    if progress.next_batch_counter != progress.optimizer_step:
        raise TrainingError("batch counter must describe the next completed-step batch")


def _descriptor(metadata: _Metadata, filename: str, sha: str) -> CheckpointDescriptor:
    progress = metadata.progress
    config = parse_phase3_canonical(metadata.config_json)
    return CheckpointDescriptor(
        relative_path=filename,
        file_sha256=sha,
        model_state_sha256=metadata.model_state_sha256,
        config_sha256=metadata.config_sha256,
        source_commit=metadata.source_commit,
        optimizer_step=progress.optimizer_step if progress else 0,
        stage=progress.stage if progress else config.training.curriculum_stage,
        validation_metric=(
            progress.validation_metric
            if progress.validation_metric is not None
            else progress.best_metric
            if progress.best_step == progress.optimizer_step
            else None
        )
        if progress
        else None,
        validation_composition_metric=progress.validation_composition_metric if progress else None,
    )


def _publish_bytes_at(parent: int, name: str, raw: bytes, *, replace: bool) -> None:
    """Durably publish bytes using only a previously validated directory FD.

    Archive creation uses a no-clobber hard link; only the mutable index uses
    replacement. Temporary creation, publication and cleanup never resolve the
    original directory pathname again, even if it is renamed or substituted.
    """
    if Path(name).name != name or name in {"", ".", ".."}:
        raise TrainingError("invalid publication filename")
    temporary = None
    published = False
    try:
        for _ in range(16):
            candidate = ".phase3-checkpoint-" + os.urandom(16).hex() + ".tmp"
            try:
                descriptor = os.open(
                    candidate,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent,
                )
            except FileExistsError:
                continue
            temporary = candidate
            break
        else:
            raise TrainingError("cannot create a unique publication temporary file")
        try:
            remaining = memoryview(raw)
            while remaining:
                count = os.write(descriptor, remaining)
                if count <= 0:
                    raise OSError("publication write made no progress")
                remaining = remaining[count:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if replace:
            try:
                info = os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(info.st_mode):
                    raise TrainingError("publication target must be a regular file")
            os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            temporary = None
            published = True
        else:
            try:
                os.link(
                    temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False
                )
                published = True
            except FileExistsError as existing_error:
                existing = read_archive_at(
                    parent, name, max_bytes=max(1, len(raw)), error_factory=TrainingError
                )
                if existing != raw:
                    raise TrainingError(
                        "content-addressed archive already exists with conflicting bytes"
                    ) from existing_error
            os.unlink(temporary, dir_fd=parent)
            temporary = None
        os.fsync(parent)
    except BaseException as error:
        context = {"published": published, "reason": str(error)}
        if temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent)
                os.fsync(parent)
            except OSError as cleanup_error:
                context.update(temporary_name=temporary, cleanup_error=str(cleanup_error))
        if not isinstance(error, Exception):
            error.add_note(f"checkpoint publication: {context}")
            raise
        raise TrainingError(
            "checkpoint published but durability or cleanup unconfirmed"
            if published
            else "checkpoint publication failed",
            context=context,
        ) from error


def _publish(directory: Path, metadata: _Metadata, tensors: dict, *, weights: bool):
    raw_metadata = canonical_json_bytes(metadata)
    _json(raw_metadata)
    raw = save_tensors(tensors, metadata={"training": raw_metadata.decode("utf-8")})
    if len(raw) > (MAX_WEIGHTS_BYTES if weights else MAX_TRAINING_BYTES):
        raise TrainingError("archive byte limit")
    sha = _sha(raw)
    name = f"{'weights' if weights else 'training'}-{sha}.safetensors"
    path = directory / name
    # Validate all parent components and existing output without following links.
    with archive_parent(path, error_factory=TrainingError) as (parent, basename):
        _publish_bytes_at(parent, basename, raw, replace=False)
    return _descriptor(metadata, name, sha)


def save_training_checkpoint(
    path: Path,
    model,
    optimizer,
    progress: TrainProgress,
    *,
    config: ResolvedConfig[Phase3Config],
    source_commit: str,
) -> CheckpointDescriptor:
    """Publish training-<SHA>.safetensors under the existing directory path."""
    with _errors():
        parsed = _config(config)
        _progress(progress, parsed)
        if model.config != parsed.neural:
            raise TrainingError("model config mismatch")
        tensors, aliases, device = _snapshot(model)
        groups, names, moments = _optimizer_snapshot(model, optimizer, progress, parsed)
        rng, rng_tensors = encode_rng(snapshot_global_rng())
        tensors.update(moments)
        tensors.update(rng_tensors)
        metadata = _Metadata(
            schema_version="phase3-training-checkpoint-v1",
            config_json=config.canonical_json.decode(),
            config_sha256=config.sha256,
            source_commit=source_commit,
            model_state_sha256=_model_hash(tensors, aliases),
            aliases=aliases,
            training=model.training,
            environment=_environment(device),
            owner_directory=str(path.absolute()),
            progress=progress,
            groups=groups,
            optimizer_names=names,
            rng=rng,
        )
        return _publish(path, metadata, tensors, weights=False)


def export_weights(
    path: Path, model, *, config: ResolvedConfig[Phase3Config], source_commit: str
) -> CheckpointDescriptor:
    """Export portable weights. No evaluated step/metric is asserted by this API.

    Weight exports use a separate prefix and are never eligible for pruning.
    Pin a selected training descriptor with update_checkpoint_index(protected=...)
    before exporting if its full optimizer/RNG archive is also evidence.
    """
    with _errors():
        parsed = _config(config)
        if model.config != parsed.neural:
            raise TrainingError("model config mismatch")
        tensors, aliases, device = _snapshot(model)
        metadata = _Metadata(
            schema_version="phase3-model-weights-v1",
            config_json=config.canonical_json.decode(),
            config_sha256=config.sha256,
            source_commit=source_commit,
            model_state_sha256=_model_hash(tensors, aliases),
            aliases=aliases,
            training=model.training,
            environment=_environment(device),
            owner_directory=str(path.absolute()),
            progress=None,
            groups=(),
            optimizer_names=(),
            rng=None,
        )
        return _publish(path, metadata, tensors, weights=True)


def _read(path: Path, *, weights: bool, parent: int | None = None):
    if parent is not None:
        raw = read_archive_at(
            parent,
            path.name,
            max_bytes=MAX_WEIGHTS_BYTES if weights else MAX_TRAINING_BYTES,
            error_factory=TrainingError,
        )
    else:
        with archive_parent(path, error_factory=TrainingError) as (pinned, name):
            raw = read_archive_at(
                pinned,
                name,
                max_bytes=MAX_WEIGHTS_BYTES if weights else MAX_TRAINING_BYTES,
                error_factory=TrainingError,
            )
    if len(raw) < 8:
        raise TrainingError("truncated archive")
    header_size = struct.unpack("<Q", raw[:8])[0]
    if header_size > MAX_METADATA_BYTES or header_size > len(raw) - 8:
        raise TrainingError("invalid metadata byte limit or truncated header")
    header = _json(raw[8 : 8 + header_size])
    container = header.pop("__metadata__", None)
    if (
        not isinstance(container, dict)
        or set(container) != {"training"}
        or type(container["training"]) is not str
    ):
        raise TrainingError("unknown safetensors metadata")
    values = _json(container["training"].encode())
    rng_values = values.get("rng")
    if isinstance(rng_values, dict) and (
        type(rng_values.get("numpy_has_gauss")) is not int
        or type(rng_values.get("numpy_cached_gaussian")) is not float
    ):
        raise TrainingError("RNG metadata requires exact primitives")
    metadata = _Metadata.model_validate_json(json.dumps(values))
    expected = "phase3-model-weights-v1" if weights else "phase3-training-checkpoint-v1"
    if metadata.schema_version != expected:
        raise TrainingError("wrong archive schema")
    if _sha(metadata.config_json.encode()) != metadata.config_sha256:
        raise TrainingError("config hash mismatch")
    config = parse_phase3_canonical(metadata.config_json)
    sha = _sha(raw)
    if path.name != f"{'weights' if weights else 'training'}-{sha}.safetensors":
        raise TrainingError("archive filename is not its canonical content address")
    if weights:
        if (
            metadata.progress is not None
            or metadata.rng is not None
            or metadata.groups
            or metadata.optimizer_names
        ):
            raise TrainingError("weights archive contains training state")
    else:
        if metadata.progress is None or metadata.rng is None or not metadata.groups:
            raise TrainingError("training archive missing training state")
        _progress(metadata.progress, config)
    return raw, header, metadata, config


def _validate_header(header, metadata, model, config, payload_size):
    unique, aliases = _layout(model)
    if metadata.aliases != aliases:
        raise TrainingError("model alias bindings differ")
    dtypes = {torch.float32: "F32", torch.bool: "BOOL", torch.int64: "I64"}
    expected = {key: (dtypes[value.dtype], list(value.shape)) for key, value in unique.items()}
    if metadata.progress is not None:
        names = []
        for group in metadata.groups:
            _options(group.options, config)
            names.extend(group.names)
        parameter_names = {
            key.removeprefix("parameter/") for key in unique if key.startswith("parameter/")
        }
        if len(names) != len(set(names)) or set(names) != parameter_names:
            raise TrainingError("optimizer parameter group names differ")
        if (
            len(metadata.optimizer_names) != len(set(metadata.optimizer_names))
            or not set(metadata.optimizer_names) <= parameter_names
        ):
            raise TrainingError("optimizer state parameter names differ")
        for name in metadata.optimizer_names:
            for kind in ("step", "exp_avg", "exp_avg_sq"):
                expected[f"optimizer/{name}/{kind}"] = (
                    "F32",
                    [] if kind == "step" else list(unique["parameter/" + name].shape),
                )
        rng_names = {"rng.cpu", "rng.mps"} if metadata.rng.has_mps else {"rng.cpu"}
        for name in rng_names:
            info = header.get(name)
            if not isinstance(info, dict) or not isinstance(info.get("shape"), list):
                raise TrainingError("invalid RNG header")
            shape = info["shape"]
            if len(shape) != 1 or type(shape[0]) is not int or not 0 < shape[0] <= 1024 * 1024:
                raise TrainingError("invalid RNG dimensions")
            expected[name] = ("U8", shape)
    if set(header) != set(expected):
        raise TrainingError("extra or missing tensor names")
    for name, (dtype, shape) in expected.items():
        item = header[name]
        if not isinstance(item, dict) or set(item) != {"dtype", "shape", "data_offsets"}:
            raise TrainingError("invalid tensor header")
        if (
            item["dtype"] != dtype
            or item["shape"] != shape
            or any(type(n) is not int for n in item["shape"])
        ):
            raise TrainingError("wrong tensor shape/dtype")
        offsets = item["data_offsets"]
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int or n < 0 for n in offsets)
        ):
            raise TrainingError("invalid tensor offsets")
        if (
            offsets[1] - offsets[0]
            != math.prod(shape) * {"F32": 4, "BOOL": 1, "I64": 8, "U8": 1}[dtype]
        ):
            raise TrainingError("invalid tensor byte length")
    cursor = 0
    for start, end in sorted(item["data_offsets"] for item in header.values()):
        if start != cursor or end > payload_size:
            raise TrainingError("tensor offsets overlap or leave a gap")
        cursor = end
    if cursor != payload_size:
        raise TrainingError("tensor payload length differs from header")


def _decode(
    path: Path,
    *,
    weights: bool,
    device: str,
    expected_config=None,
    expected_source=None,
    expected_sha=None,
):
    _device(device)
    raw, header, metadata, config = _read(path, weights=weights)
    if expected_config is not None and metadata.config_sha256 != expected_config:
        raise TrainingError("unexpected config hash")
    if expected_source is not None and metadata.source_commit != expected_source:
        raise TrainingError("unexpected source revision")
    if expected_sha is not None and _sha(raw) != expected_sha:
        raise TrainingError("unexpected archive hash")
    # Construction initializes parameters on CPU and consumes CPU RNG. Isolate it
    # even for successful loads: restoring archived RNG is an explicit later step.
    with torch.random.fork_rng(devices=[]):
        model = EventFlowModel(config.neural)
    _validate_header(
        header, metadata, model, config, len(raw) - 8 - struct.unpack("<Q", raw[:8])[0]
    )
    tensors = load_tensors(raw)
    if any(
        value.is_floating_point() and not bool(torch.isfinite(value).all())
        for value in tensors.values()
    ):
        raise TrainingError("nonfinite tensor payload")
    if _model_hash(tensors, metadata.aliases) != metadata.model_state_sha256:
        raise TrainingError("model state hash mismatch")
    model.load_state_dict(
        {key: tensors[value] for key, value in metadata.aliases.items()}, strict=True
    )
    model.to(device)
    model.train(metadata.training)
    return model, tensors, metadata, config, _descriptor(metadata, path.name, _sha(raw))


def load_training_checkpoint(
    path: Path, *, expected_config_sha256: str, expected_source_commit: str, device: str
) -> TrainingRestore:
    """Validate in isolation; no caller state or global RNG is replaced."""
    with _errors():
        _expected_hash(expected_config_sha256, 64)
        _expected_hash(expected_source_commit, 40)
        model, tensors, metadata, config, descriptor = _decode(
            path,
            weights=False,
            device=device,
            expected_config=expected_config_sha256,
            expected_source=expected_source_commit,
        )
        rng = decode_rng(
            metadata.rng, {key: value for key, value in tensors.items() if key.startswith("rng.")}
        )
        validate_rng_restore(rng, restore_mps=device == "mps")
        _validate_moments(tensors, metadata.progress.optimizer_step)
        parameters = dict(model.named_parameters())
        groups = []
        for group in metadata.groups:
            options = _options(group.options, config)
            options["betas"] = tuple(options["betas"])
            groups.append({"params": [parameters[name] for name in group.names], **options})
        optimizer = torch.optim.AdamW(groups)
        for name in metadata.optimizer_names:
            optimizer.state[parameters[name]] = {
                kind: tensors[f"optimizer/{name}/{kind}"]
                .to("cpu" if kind == "step" else device)
                .clone()
                for kind in ("step", "exp_avg", "exp_avg_sq")
            }
        return TrainingRestore(
            model,
            optimizer,
            metadata.progress,
            rng,
            ResolvedConfig(config, metadata.config_json.encode(), metadata.config_sha256, ()),
            descriptor,
            device,
        )


def load_weights(path: Path, *, expected_sha256: str, device: str) -> EventFlowModel:
    """Load an independently hash-verified export onto CPU or native MPS."""
    with _errors():
        _expected_hash(expected_sha256, 64)
        model, _, _, _, _ = _decode(path, weights=True, device=device, expected_sha=expected_sha256)
        return model


@dataclass(frozen=True, slots=True)
class WeightRestore:
    """One validated archive read, including its complete canonical configuration."""

    model: EventFlowModel
    config: ResolvedConfig[Phase3Config]
    descriptor: CheckpointDescriptor


def load_weight_bundle(path: Path, *, expected_sha256: str, device: str) -> WeightRestore:
    """Load safe weights and bound metadata together; no arbitrary Python state."""
    with _errors():
        _expected_hash(expected_sha256, 64)
        model, _, metadata, config, descriptor = _decode(
            path, weights=True, device=device, expected_sha=expected_sha256
        )
        return WeightRestore(
            model,
            ResolvedConfig(config, metadata.config_json.encode(), metadata.config_sha256, ()),
            descriptor,
        )


@contextmanager
def _index_lock(directory: Path):
    with archive_parent(directory / ".checkpoint-index.lock", error_factory=TrainingError) as (
        parent,
        name,
    ):
        descriptor = os.open(
            name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=parent
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise TrainingError("index lock must be a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield parent
        finally:
            os.close(descriptor)


def _identity(metadata):
    progress = metadata.progress
    if progress is None:
        raise TrainingError("index requires a training archive")
    return (
        metadata.owner_directory,
        metadata.config_sha256,
        metadata.source_commit,
        progress.train_root_seed,
        progress.train_public_id_seed,
        progress.validation_manifest_sha256,
    )


def _index_identity(index):
    return (
        index.owner_directory,
        index.config_sha256,
        index.source_commit,
        index.train_root_seed,
        index.train_public_id_seed,
        index.validation_manifest_sha256,
    )


def _owned(directory: Path, descriptor: CheckpointDescriptor, identity=None, *, parent: int):
    CheckpointDescriptor.model_validate_json(descriptor.model_dump_json())
    if descriptor.relative_path != f"training-{descriptor.file_sha256}.safetensors":
        raise TrainingError("index archive name is not canonical")
    raw, _, metadata, _ = _read(directory / descriptor.relative_path, weights=False, parent=parent)
    if descriptor != _descriptor(metadata, descriptor.relative_path, _sha(raw)):
        raise TrainingError("index descriptor disagrees with archive")
    if metadata.owner_directory != str(directory.absolute()):
        raise TrainingError("archive is not owned by this exact run directory")
    if identity is not None and _identity(metadata) != identity:
        raise TrainingError("archive is from a different run")
    return metadata


def _load_index(directory, parent):
    raw = read_archive_at(parent, _INDEX, max_bytes=MAX_METADATA_BYTES, error_factory=TrainingError)
    index = CheckpointIndex.model_validate_json(json.dumps(_json(raw)))
    if index.owner_directory != str(directory.absolute()):
        raise TrainingError("index is not owned by this directory")
    owned = {item.relative_path: item for item in index.owned}
    if len(owned) != len(index.owned):
        raise TrainingError("duplicate index ownership")
    references = (index.latest, *index.best, *index.protected)
    for item in references:
        if owned.get(item.relative_path) != item:
            raise TrainingError("index references an unowned archive")
    for items in (index.best, index.protected):
        if len(items) != len({item.relative_path for item in items}):
            raise TrainingError("duplicate index reference")
    if index.best != _best(index.owned):
        raise TrainingError("index best pointers disagree with scores")
    if any(item.optimizer_step > index.latest.optimizer_step for item in index.owned):
        raise TrainingError("index latest pointer is stale")
    for item in index.owned:
        _owned(directory, item, _index_identity(index), parent=parent)
    return index


def _best(items):
    scored = [item for item in items if item.validation_metric is not None]
    return tuple(
        sorted(
            scored,
            key=lambda item: (
                -item.validation_metric,
                -item.validation_composition_metric
                if item.validation_composition_metric is not None
                else 1.0,
                item.optimizer_step,
                item.file_sha256,
            ),
        )[:3]
    )


def load_checkpoint_index(path: Path) -> CheckpointIndex:
    """Read and verify every owned archive and retained pointer under the run lock."""
    with _errors(), _index_lock(path) as parent:
        return _load_index(path, parent)


def _write_index(parent, index):
    raw = canonical_json_bytes(index)
    _json(raw)
    _publish_bytes_at(parent, _INDEX, raw, replace=True)


def update_checkpoint_index(
    path: Path,
    descriptor: CheckpointDescriptor,
    *,
    protected: tuple[CheckpointDescriptor, ...] = (),
) -> CheckpointIndex:
    """Atomically retain latest, best three and cumulative selected/evidence pins."""
    with _errors(), _index_lock(path) as parent:
        metadata = _owned(path, descriptor, parent=parent)
        identity = _identity(metadata)
        try:
            os.stat(_INDEX, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            old = None
        else:
            old = _load_index(path, parent)
        if old is not None and (
            _index_identity(old) != identity
            or descriptor.optimizer_step < old.latest.optimizer_step
        ):
            raise TrainingError("index run mismatch or decreasing latest step")
        owned = {item.relative_path: item for item in old.owned} if old else {}
        pins = {item.relative_path: item for item in old.protected} if old else {}
        for item in (descriptor, *protected):
            _owned(path, item, identity, parent=parent)
            if item.relative_path in owned and owned[item.relative_path] != item:
                raise TrainingError("conflicting owned descriptor")
            owned[item.relative_path] = item
        pins.update({item.relative_path: item for item in protected})
        if any(item.optimizer_step > descriptor.optimizer_step for item in owned.values()):
            raise TrainingError("protected checkpoint is newer than the latest checkpoint")
        index = CheckpointIndex(
            owner_directory=identity[0],
            config_sha256=identity[1],
            source_commit=identity[2],
            train_root_seed=identity[3],
            train_public_id_seed=identity[4],
            validation_manifest_sha256=identity[5],
            latest=descriptor,
            best=_best(owned.values()),
            protected=tuple(pins.values()),
            owned=tuple(owned.values()),
        )
        _write_index(parent, index)
        return index


def prune_training_checkpoints(path: Path) -> tuple[str, ...]:
    """Remove only fully verified, indexed, unreferenced archives of this exact run.

    Publish the reduced inventory before unlinking. A crash can leave an orphan
    archive, which is deliberately not discovered or deleted by filename scans.
    """
    with _errors(), _index_lock(path) as parent:
        index = _load_index(path, parent)
        kept = {item.relative_path for item in (index.latest, *index.best, *index.protected)}
        removed = tuple(
            item.relative_path for item in index.owned if item.relative_path not in kept
        )
        if not removed:
            return ()
        # Every owned file was verified before either index publication or deletion.
        _write_index(
            parent,
            index.model_copy(
                update={"owned": tuple(item for item in index.owned if item.relative_path in kept)}
            ),
        )
        for name in removed:
            os.unlink(name, dir_fd=parent)
        os.fsync(parent)
        return removed
