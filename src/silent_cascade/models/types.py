"""Validated public tensor containers shared by neural components."""

from dataclasses import dataclass, fields

import torch

from silent_cascade.eventflow.state import ContinuousState
from silent_cascade.models.errors import NeuralError

_LATENT_CHANNELS = (
    ("z_fast", 256),
    ("z_slow", 64),
    ("drives", 8),
    ("focus_key", 64),
    ("hypothesis_latent", 64),
)
_LATENT_DIM = sum(size for _, size in _LATENT_CHANNELS)
_MAX_BATCH_SIZE = 128
_MEMORY_SLOTS = 64
_RECORD_DIM = 96


def _require_supported_device(device: torch.device, name: str) -> None:
    if device.type not in {"cpu", "mps"}:
        raise NeuralError(f"{name} must use a CPU or MPS device")


def _validate_tensor(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | None,
    name: str,
    finite: bool = False,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise NeuralError(f"{name} must be a tensor")
    _require_supported_device(value.device, name)
    if device is not None and value.device != device:
        raise NeuralError("model context tensors must use one device")
    if value.dtype is not dtype:
        raise NeuralError(f"{name} must use {dtype}")
    if tuple(value.shape) != shape:
        raise NeuralError(f"{name} must have shape {shape}")
    if finite and not bool(torch.isfinite(value).all()):
        raise NeuralError(f"{name} must contain only finite values")
    return value


def _require_batch_size(batch_size: object) -> int:
    if type(batch_size) is not int:
        raise NeuralError("batch_size must be an exact integer")
    if not 1 <= batch_size <= _MAX_BATCH_SIZE:
        raise NeuralError("batch_size must be between 1 and 128")
    return batch_size


@dataclass(frozen=True, slots=True)
class TensorWorkspace:
    """Batched latent state and guard accumulators on one device."""

    latent: torch.Tensor
    accumulators: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.latent, torch.Tensor) or self.latent.ndim != 2:
            raise NeuralError("latent must be a rank-two tensor")
        batch_size = _require_batch_size(self.latent.shape[0])
        latent = _validate_tensor(
            self.latent,
            shape=(batch_size, _LATENT_DIM),
            dtype=torch.float32,
            device=None,
            name="latent",
            finite=True,
        )
        accumulators = _validate_tensor(
            self.accumulators,
            shape=(batch_size, 3),
            dtype=torch.float32,
            device=latent.device,
            name="accumulators",
            finite=True,
        )
        if bool(((latent < -1.0) | (latent > 1.0)).any()):
            raise NeuralError("latent values must remain within [-1, 1]")
        if bool(((accumulators < 0.0) | (accumulators > 2.0)).any()):
            raise NeuralError("accumulator values must remain within [0, 2]")
        object.__setattr__(self, "latent", latent.clone())
        object.__setattr__(self, "accumulators", accumulators.clone())

    @classmethod
    def zeros(cls, batch_size: int, device: str) -> "TensorWorkspace":
        batch_size = _require_batch_size(batch_size)
        if type(device) is not str:
            raise NeuralError("device must be a string naming CPU or MPS")
        try:
            resolved = torch.device(device)
        except (RuntimeError, ValueError) as error:
            raise NeuralError("device must name CPU or MPS") from error
        _require_supported_device(resolved, "workspace")
        if resolved.type == "mps" and not torch.backends.mps.is_available():
            raise NeuralError("MPS workspace requested but MPS is unavailable")
        return cls(
            latent=torch.zeros(batch_size, _LATENT_DIM, dtype=torch.float32, device=resolved),
            accumulators=torch.zeros(batch_size, 3, dtype=torch.float32, device=resolved),
        )

    @classmethod
    def _from_functional_update(
        cls, latent: torch.Tensor, accumulators: torch.Tensor
    ) -> "TensorWorkspace":
        """Build from trusted bounded equations without device-synchronizing scans.

        Public construction remains fully validated. Analytic flow and convex
        jumps use this private boundary after validating their inputs and applying
        equations that preserve the workspace bounds by construction.
        """
        if not isinstance(latent, torch.Tensor) or latent.ndim != 2:
            raise NeuralError("functional latent must be a rank-two tensor")
        batch_size = _require_batch_size(latent.shape[0])
        latent = _validate_tensor(
            latent,
            shape=(batch_size, _LATENT_DIM),
            dtype=torch.float32,
            device=None,
            name="functional latent",
        )
        accumulators = _validate_tensor(
            accumulators,
            shape=(batch_size, 3),
            dtype=torch.float32,
            device=latent.device,
            name="functional accumulators",
        )
        instance = cls.__new__(cls)
        object.__setattr__(instance, "latent", latent)
        object.__setattr__(instance, "accumulators", accumulators)
        return instance

    @property
    def batch_size(self) -> int:
        return self.latent.shape[0]

    @property
    def device(self) -> torch.device:
        return self.latent.device

    def row(self, index: int) -> ContinuousState:
        if type(index) is not int:
            raise NeuralError("row index must be an exact integer")
        if not 0 <= index < self.batch_size:
            raise NeuralError("row index is outside the workspace batch")
        channels: dict[str, torch.Tensor] = {}
        start = 0
        for name, size in _LATENT_CHANNELS:
            channels[name] = self.latent[index, start : start + size]
            start += size
        return ContinuousState(
            **channels,
            guard_accumulators=self.accumulators[index],
        )


@dataclass(frozen=True, slots=True)
class ModelContext:
    """Public-derived tensors accepted by neural prediction components."""

    workspace: TensorWorkspace
    memory_embeddings: torch.Tensor
    eligibility: torch.Tensor
    support_mask: torch.Tensor
    active_slot_indices: torch.Tensor
    modes: torch.Tensor
    time_features: torch.Tensor
    hypothesis_features: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, TensorWorkspace):
            raise NeuralError("workspace must be a TensorWorkspace")
        batch_size = self.workspace.batch_size
        device = self.workspace.device
        specifications = {
            "memory_embeddings": ((batch_size, _MEMORY_SLOTS, _RECORD_DIM), torch.float32, True),
            "eligibility": ((batch_size, _MEMORY_SLOTS), torch.bool, False),
            "support_mask": ((batch_size, _MEMORY_SLOTS), torch.bool, False),
            "active_slot_indices": ((batch_size,), torch.int64, False),
            "modes": ((batch_size,), torch.int64, False),
            "time_features": ((batch_size, 2), torch.float32, True),
            "hypothesis_features": ((batch_size, 8), torch.float32, True),
        }
        for item in fields(self):
            if item.name == "workspace":
                continue
            shape, dtype, finite = specifications[item.name]
            value = _validate_tensor(
                getattr(self, item.name),
                shape=shape,
                dtype=dtype,
                device=device,
                name=item.name,
                finite=finite,
            )
            object.__setattr__(self, item.name, value.clone())
        if bool(((self.active_slot_indices < -1) | (self.active_slot_indices >= 64)).any()):
            raise NeuralError("active slot indices must be -1 or valid memory slots")
        if bool(((self.modes < 0) | (self.modes >= 6)).any()):
            raise NeuralError("modes must be integer indices in [0, 6)")

    @property
    def batch_size(self) -> int:
        return self.workspace.batch_size

    @property
    def device(self) -> torch.device:
        return self.workspace.device


@dataclass(frozen=True, slots=True)
class ExternalFeatures:
    """Current public FACT or ACTIVATE fields for an active row batch.

    Event kind IDs are local neural IDs: ``0`` is FACT and ``1`` is ACTIVATE.
    Padding, outcome, end, future-event, and private-target representations are
    deliberately absent from this schema.
    """

    subject_ids: torch.Tensor
    object_ids: torch.Tensor
    record_kind_ids: torch.Tensor
    hazard_ids: torch.Tensor
    provenance_ids: torch.Tensor
    record_scalar_features: torch.Tensor
    activation_entity_ids: torch.Tensor
    event_kinds: torch.Tensor
    time_features: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.subject_ids, torch.Tensor) or self.subject_ids.ndim != 1:
            raise NeuralError("external subject_ids must be a rank-one tensor")
        batch_size = _require_batch_size(self.subject_ids.shape[0])
        device = self.subject_ids.device
        specifications = {
            "subject_ids": ((batch_size,), torch.int64, False),
            "object_ids": ((batch_size,), torch.int64, False),
            "record_kind_ids": ((batch_size,), torch.int64, False),
            "hazard_ids": ((batch_size,), torch.int64, False),
            "provenance_ids": ((batch_size,), torch.int64, False),
            "record_scalar_features": ((batch_size, 6), torch.float32, True),
            "activation_entity_ids": ((batch_size,), torch.int64, False),
            "event_kinds": ((batch_size,), torch.int64, False),
            "time_features": ((batch_size, 2), torch.float32, True),
        }
        for item in fields(self):
            shape, dtype, finite = specifications[item.name]
            value = _validate_tensor(
                getattr(self, item.name),
                shape=shape,
                dtype=dtype,
                device=device,
                name=item.name,
                finite=finite,
            )
            object.__setattr__(self, item.name, value.clone())
        bounded_ids = (
            (self.subject_ids, 64, "subject IDs"),
            (self.object_ids, 64, "object IDs"),
            (self.record_kind_ids, 3, "record kind IDs"),
            (self.hazard_ids, 5, "hazard IDs"),
            (self.provenance_ids, 3, "provenance IDs"),
            (self.activation_entity_ids, 64, "activation entity IDs"),
            (self.event_kinds, 2, "external event kind IDs"),
        )
        for values, upper, name in bounded_ids:
            if bool(((values < 0) | (values >= upper)).any()):
                raise NeuralError(f"{name} must be in [0, {upper})")

    @property
    def batch_size(self) -> int:
        return self.subject_ids.shape[0]

    @property
    def device(self) -> torch.device:
        return self.subject_ids.device
