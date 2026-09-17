"""Distinct Phase4 archive; validate isolated model, optimizer and RNG before restore."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from silent_cascade.config import ResolvedConfig
from silent_cascade.eventflow.checkpoint_rng import (
    RngMetadata,
    decode_rng,
    encode_rng,
    restore_rng_snapshot,
    validate_rng_restore,
)
from silent_cascade.eventflow.neural import NeuralModelIdentity
from silent_cascade.eventflow.neural_weights import (
    WeightsMetadata,
    decode_archive,
    encode_archive,
    reconstruct_weights,
    snapshot_weights,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.rng import RngSnapshot, snapshot_global_rng
from silent_cascade.train.archive_tensors import (
    optimizer_options,
    snapshot_optimizer,
    validate_moments,
)
from silent_cascade.train.checkpoints import _device, _environment, _errors
from silent_cascade.train.pilot_config import Phase4Config, parse_phase4_canonical
from silent_cascade.train.pilot_data import _publish_pilot_bytes, _read_pilot_bytes
from silent_cascade.train.pilot_provenance import PilotSourceIdentity
from silent_cascade.train.pilot_state import PilotProgress
from silent_cascade.train.state import TrainingError
from silent_cascade.validation import JsonValue, StrictModel


class OptimizerGroup(StrictModel):
    names: tuple[str, ...]
    options: dict[str, JsonValue]


class PilotCheckpoint(StrictModel):
    schema_version: Literal["phase4-pilot-training-v1"] = "phase4-pilot-training-v1"
    config_json: str
    source: PilotSourceIdentity
    weights: WeightsMetadata
    progress: PilotProgress
    groups: tuple[OptimizerGroup, ...]
    optimizer_names: tuple[str, ...]
    rng: RngMetadata
    environment: dict[str, JsonValue]
    training: bool


@dataclass(frozen=True)
class PilotTrainingSession:
    model: EventFlowModel
    optimizer: torch.optim.AdamW
    progress: PilotProgress
    rng: RngSnapshot
    device: str


def save_pilot_checkpoint(
    path: Path,
    *,
    model: EventFlowModel,
    optimizer: torch.optim.AdamW,
    progress: PilotProgress,
    config: ResolvedConfig[Phase4Config],
    source: PilotSourceIdentity,
) -> str:
    with _errors():
        _device(next(model.parameters()).device.type)
        parsed = parse_phase4_canonical(config.canonical_json.decode())
        if (
            parsed != config.config
            or source.config_sha256 != config.sha256
            or model.config != parsed.neural
            or sha256_bytes(config.canonical_json) != config.sha256
        ):
            raise TrainingError("checkpoint config differs")
        progress = PilotProgress.model_validate_json(canonical_json_bytes(progress))
        identity = NeuralModelIdentity.from_model(model, source_revision=source.source_commit)
        weights, tensors = snapshot_weights(model, identity)
        groups, names, moments = snapshot_optimizer(
            model, optimizer, parsed.pilot, progress.global_step
        )
        tensors.update(moments)
        rng, rng_tensors = encode_rng(snapshot_global_rng())
        tensors.update(rng_tensors)
        metadata = PilotCheckpoint(
            config_json=config.canonical_json.decode(),
            source=source,
            weights=weights,
            progress=progress,
            groups=tuple(OptimizerGroup(**g) for g in groups),
            optimizer_names=names,
            rng=rng,
            environment=_environment(next(model.parameters()).device.type).model_dump(),
            training=model.training,
        )
        raw = encode_archive(tensors, metadata, "pilot_training")
        _publish_pilot_bytes(path, raw)
        return sha256_bytes(raw)


def load_pilot_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    config: ResolvedConfig[Phase4Config],
    source: PilotSourceIdentity,
    device: str,
) -> PilotTrainingSession:
    with _errors():
        _device(device)
        metadata, tensors = decode_archive(
            _read_pilot_bytes(path), expected_sha256, PilotCheckpoint, "pilot_training"
        )
        parsed = parse_phase4_canonical(metadata.config_json)
        if (
            metadata.source != source
            or metadata.config_json.encode() != config.canonical_json
            or parsed != config.config
            or sha256_bytes(config.canonical_json) != source.config_sha256
            or metadata.weights.identity.source_revision != source.source_commit
            or metadata.weights.identity.model_config_json
            != canonical_json_bytes(parsed.neural).decode()
        ):
            raise TrainingError("checkpoint source/config differs from this attempt")
        weight_names = {k for k in tensors if k.startswith(("parameter/", "buffer/"))}
        model = reconstruct_weights(
            metadata.weights, {k: v for k, v in tensors.items() if k in weight_names}, device
        )
        model.requires_grad_(True).train(metadata.training)
        parameters = dict(model.named_parameters())
        names = [n for group in metadata.groups for n in group.names]
        if len(names) != len(set(names)) or set(names) != set(parameters):
            raise TrainingError("optimizer groups differ")
        if len(set(metadata.optimizer_names)) != len(metadata.optimizer_names) or not set(
            metadata.optimizer_names
        ) <= set(parameters):
            raise TrainingError("optimizer state inventory differs")
        expected = set(weight_names) | (
            {"rng.cpu", "rng.mps"} if metadata.rng.has_mps else {"rng.cpu"}
        )
        groups = []
        for group in metadata.groups:
            if canonical_json_bytes(group.options) != canonical_json_bytes(
                optimizer_options(parsed.pilot)
            ):
                raise TrainingError("optimizer options differ")
            options = dict(group.options)
            options["betas"] = tuple(options["betas"])
            groups.append({"params": [parameters[n] for n in group.names], **options})
        optimizer = torch.optim.AdamW(groups)
        for name in metadata.optimizer_names:
            state = {}
            for kind in ("step", "exp_avg", "exp_avg_sq"):
                key = f"optimizer/{name}/{kind}"
                expected.add(key)
                tensor = tensors[key]
                if tensor.dtype != torch.float32 or tensor.shape != (
                    () if kind == "step" else parameters[name].shape
                ):
                    raise TrainingError("optimizer tensor layout differs")
                state[kind] = tensor.to("cpu" if kind == "step" else device).clone()
            optimizer.state[parameters[name]] = state
        if set(tensors) != expected:
            raise TrainingError("archive tensor inventory differs")
        validate_moments(tensors, metadata.progress.global_step)
        rng = decode_rng(metadata.rng, {k: v for k, v in tensors.items() if k.startswith("rng.")})
        validate_rng_restore(rng, restore_mps=device == "mps")
        session = PilotTrainingSession(model, optimizer, metadata.progress, rng, device)
        restore_rng_snapshot(rng, restore_mps=device == "mps")
        return session
