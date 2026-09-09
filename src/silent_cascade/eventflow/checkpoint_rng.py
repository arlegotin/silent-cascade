"""Versioned JSON/uint8 RNG archive codecs with isolated validation."""

import math
import random
from dataclasses import replace
from typing import Literal

import numpy as np
import torch
from pydantic import Field

from silent_cascade.errors import ReplayError
from silent_cascade.rng import RngSnapshot, mps_rng_state_supported, restore_global_rng
from silent_cascade.validation import JsonValue, StrictModel


class RngMetadata(StrictModel):
    schema_version: Literal["runtime-rng-v1"] = "runtime-rng-v1"
    python_state: JsonValue
    numpy_algorithm: Literal["MT19937"]
    numpy_keys: tuple[int, ...] = Field(min_length=624, max_length=624)
    numpy_position: int = Field(ge=0, le=624)
    numpy_has_gauss: Literal[0, 1]
    numpy_cached_gaussian: float
    has_mps: bool


def _encode_tuple(value: object) -> JsonValue:
    if isinstance(value, tuple):
        return {"tuple": [_encode_tuple(item) for item in value]}
    if value is None or type(value) is int or (type(value) is float and math.isfinite(value)):
        return value
    raise ReplayError("unsupported RNG scalar")


def _decode_tuple(value: JsonValue, depth: int = 0) -> object:
    if depth > 4:
        raise ReplayError("RNG tuple nesting exceeds schema")
    if isinstance(value, dict) and set(value) == {"tuple"} and isinstance(value["tuple"], list):
        return tuple(_decode_tuple(item, depth + 1) for item in value["tuple"])
    if value is None or type(value) is int or (type(value) is float and math.isfinite(value)):
        return value
    raise ReplayError("invalid RNG tuple encoding")


def encode_rng(snapshot: RngSnapshot) -> tuple[RngMetadata, dict[str, torch.Tensor]]:
    algorithm, keys, position, gaussian, cached = snapshot.numpy_state
    metadata = RngMetadata(
        python_state=_encode_tuple(snapshot.python_state),
        numpy_algorithm=algorithm,
        numpy_keys=tuple(int(key) for key in keys),
        numpy_position=position,
        numpy_has_gauss=gaussian,
        numpy_cached_gaussian=cached,
        has_mps=snapshot.torch_mps_state is not None,
    )
    tensors = {"rng.cpu": snapshot.torch_cpu_state.detach().cpu().contiguous().clone()}
    if snapshot.torch_mps_state is not None:
        tensors["rng.mps"] = snapshot.torch_mps_state.detach().cpu().contiguous().clone()
    decode_rng(metadata, tensors)
    return metadata, tensors


def decode_rng(metadata: RngMetadata, tensors: dict[str, torch.Tensor]) -> RngSnapshot:
    """Validate CPU generators in isolation, retaining opaque archived MPS bytes."""
    try:
        if any(not 0 <= key <= 2**32 - 1 for key in metadata.numpy_keys):
            raise ValueError("invalid MT19937 key")
        required = {"rng.cpu", "rng.mps"} if metadata.has_mps else {"rng.cpu"}
        if set(tensors) != required:
            raise ValueError("RNG tensor set differs")
        for tensor in tensors.values():
            if (
                tensor.device.type != "cpu"
                or tensor.dtype is not torch.uint8
                or tensor.ndim != 1
                or not 0 < tensor.numel() <= 1024 * 1024
            ):
                raise ValueError("invalid RNG tensor")
        python_state = _decode_tuple(metadata.python_state)
        numpy_state = (
            metadata.numpy_algorithm,
            np.array(metadata.numpy_keys, dtype=np.uint32),
            metadata.numpy_position,
            metadata.numpy_has_gauss,
            metadata.numpy_cached_gaussian,
        )
        random.Random().setstate(python_state)
        np.random.RandomState().set_state(numpy_state)
        torch.Generator(device="cpu").set_state(tensors["rng.cpu"])
        return RngSnapshot(
            python_state,
            numpy_state,
            tensors["rng.cpu"].clone(),
            tensors["rng.mps"].clone() if metadata.has_mps else None,
        )
    except (ValueError, TypeError, RuntimeError, KeyError) as error:
        raise ReplayError("invalid archived RNG state") from error


def validate_rng_restore(snapshot: RngSnapshot, *, restore_mps: bool = False) -> None:
    """Validate requested generators without replacing any global RNG state."""
    encode_rng(snapshot)
    if restore_mps and (not mps_rng_state_supported() or snapshot.torch_mps_state is None):
        raise ReplayError("explicit MPS RNG restore requires an available backend and archive")
    if restore_mps:
        try:
            torch.Generator(device="mps").set_state(snapshot.torch_mps_state.clone())
        except (RuntimeError, TypeError, ValueError) as error:
            raise ReplayError("invalid archived MPS RNG state") from error


def restore_rng_snapshot(snapshot: RngSnapshot, *, restore_mps: bool = False) -> None:
    validate_rng_restore(snapshot, restore_mps=restore_mps)
    restore_global_rng(snapshot if restore_mps else replace(snapshot, torch_mps_state=None))
