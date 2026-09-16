"""Closed portable neural weights and bounded private archive primitives."""

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from pydantic import Field
from safetensors.torch import load, save

from silent_cascade.errors import ReplayError
from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at
from silent_cascade.eventflow.neural import NeuralModelIdentity
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.io import atomic_create_bytes
from silent_cascade.models.config import NeuralModelConfig
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.models.weights import model_layout, model_state_sha256
from silent_cascade.validation import StrictModel

MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024 * 1024


class WeightsMetadata(StrictModel):
    schema_version: Literal["phase4-neural-weights-v1"] = "phase4-neural-weights-v1"
    identity: NeuralModelIdentity
    aliases: dict[str, str] = Field(max_length=1024)


@dataclass(frozen=True, slots=True)
class NeuralWeights:
    model: EventFlowModel
    identity: NeuralModelIdentity
    sha256: str


def archive_error(field):
    return ReplayError(f"invalid neural archive: {field}", context={"field": field})


def bounded_json(raw: bytes, *, max_bytes=MAX_METADATA_BYTES):
    if not isinstance(raw, bytes) or len(raw) > max_bytes:
        raise archive_error("metadata.byte_limit")
    depth, quoted, escaped = 0, False, False
    for character in raw:
        if quoted:
            if escaped:
                escaped = False
            elif character == 92:
                escaped = True
            elif character == 34:
                quoted = False
        elif character == 34:
            quoted = True
        elif character in (91, 123):
            depth += 1
            if depth > 64:
                raise archive_error("metadata.depth")
        elif character in (93, 125):
            depth -= 1

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise archive_error("metadata.duplicate_key")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=unique)
    except (ValueError, TypeError, RecursionError) as error:
        raise archive_error("metadata.json") from error


def read_bytes(path, *, max_bytes=MAX_ARCHIVE_BYTES):
    try:
        with archive_parent(path, error_factory=archive_error) as (parent, name):
            return read_archive_at(parent, name, max_bytes=max_bytes, error_factory=archive_error)
    except OSError as error:
        raise archive_error("path") from error


def publish_bytes(path, raw):
    if len(raw) > MAX_ARCHIVE_BYTES:
        raise archive_error("archive.byte_limit")
    try:
        with archive_parent(path, error_factory=archive_error):
            atomic_create_bytes(path, raw, mode=0o600)
    except Exception as error:
        raise archive_error("publication") from error
    return sha256_bytes(raw)


def encode_archive(tensors, metadata, key):
    encoded = canonical_json_bytes(metadata)
    if (
        len(encoded) > MAX_METADATA_BYTES
        or sum(t.numel() * t.element_size() for t in tensors.values()) > MAX_ARCHIVE_BYTES
    ):
        raise archive_error("archive.byte_limit")
    raw = save(
        {name: value.detach().cpu().contiguous().clone() for name, value in tensors.items()},
        metadata={key: encoded.decode()},
    )
    if len(raw) > MAX_ARCHIVE_BYTES or struct.unpack("<Q", raw[:8])[0] > MAX_METADATA_BYTES:
        raise archive_error("archive.byte_limit")
    return raw


def decode_archive(raw, expected_sha256, schema, key):
    try:
        if len(raw) > MAX_ARCHIVE_BYTES or sha256_bytes(raw) != expected_sha256:
            raise archive_error("archive.sha256_or_limit")
        size = struct.unpack("<Q", raw[:8])[0]
        if size > MAX_METADATA_BYTES or size > len(raw) - 8:
            raise archive_error("header.byte_limit")
        header = bounded_json(raw[8 : 8 + size])
        envelope = header.get("__metadata__") if isinstance(header, dict) else None
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {key}
            or not isinstance(envelope[key], str)
        ):
            raise archive_error("envelope")
        encoded = envelope[key].encode()
        bounded_json(encoded)
        metadata = schema.model_validate_json(encoded)
        if canonical_json_bytes(metadata) != encoded:
            raise archive_error("metadata.canonical")
        return metadata, load(raw)
    except ReplayError:
        raise
    except Exception as error:
        raise archive_error("schema_or_tensors") from error


def snapshot_weights(model, identity):
    if type(model) is not EventFlowModel:
        raise archive_error("unregistered_model")
    unique, aliases = model_layout(model)
    tensors = {name: value.detach().cpu().contiguous().clone() for name, value in unique.items()}
    metadata = WeightsMetadata(identity=identity, aliases=aliases)
    if (
        canonical_json_bytes(model.config).decode() != identity.model_config_json
        or model_state_sha256(tensors, aliases) != identity.model_state_sha256
    ):
        raise archive_error("weights.identity")
    return metadata, tensors


def reconstruct_weights(metadata, tensors, device):
    """Construct only our closed architecture and never consume caller RNG."""
    try:
        metadata = WeightsMetadata.model_validate_json(canonical_json_bytes(metadata))
        if device not in {"cpu", "mps"} or (
            device == "mps" and not torch.backends.mps.is_available()
        ):
            raise archive_error("device")
        if any(
            value.device.type != "cpu" or not bool(torch.isfinite(value).all())
            for value in tensors.values()
        ):
            raise archive_error("weights.tensor")
        if model_state_sha256(tensors, metadata.aliases) != metadata.identity.model_state_sha256:
            raise archive_error("weights.logical_hash")
        config = NeuralModelConfig.model_validate_json(metadata.identity.model_config_json)
        with torch.random.fork_rng(devices=[]):
            model = EventFlowModel(config)
        layout, aliases = model_layout(model)
        if aliases != metadata.aliases or set(layout) != set(tensors):
            raise archive_error("weights.aliases_or_names")
        with torch.no_grad():
            for name, target in layout.items():
                value = tensors[name]
                if value.dtype != target.dtype or value.shape != target.shape:
                    raise archive_error("weights.layout")
                target.copy_(value)
        model.to(device).eval().requires_grad_(False)
        return model
    except ReplayError:
        raise
    except Exception as error:
        raise archive_error("weights.invalid") from error


def save_neural_weights(path: Path, *, model, identity) -> str:
    metadata, tensors = snapshot_weights(model, identity)
    reconstruct_weights(metadata, tensors, "cpu")
    return publish_bytes(path, encode_archive(tensors, metadata, "weights"))


def load_neural_weights(path: Path, *, expected_sha256: str, device: str) -> NeuralWeights:
    raw = read_bytes(path)
    metadata, tensors = decode_archive(raw, expected_sha256, WeightsMetadata, "weights")
    return NeuralWeights(
        reconstruct_weights(metadata, tensors, device), metadata.identity, sha256_bytes(raw)
    )
