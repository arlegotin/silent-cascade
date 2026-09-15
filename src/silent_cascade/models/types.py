"""Validated public tensor containers shared by neural components."""

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING

import torch

from silent_cascade.eventflow.state import ContinuousState
from silent_cascade.models.errors import NeuralError

if TYPE_CHECKING:
    from silent_cascade.memory.tensor_store import RecordTensorBatch

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


@dataclass(frozen=True, slots=True)
class LossInputs:
    """Neutral tensors for the trace objective; never a prediction-module input.

    Predictions/targets/masks align on [B,T] internal positions, with final
    dormancy and negative action predictions attached to the last real position.
    Guard tensors end in 3, retrieval in 64, and logits in their class count.
    State tensors instead use [B,J,456] for actual external/internal jumps;
    jump_mask identifies valid positions. No field is copied or detached here.
    """

    raw_guard_targets: torch.Tensor
    crossing_offsets: torch.Tensor
    legal_guard_mask: torch.Tensor
    boundary_mask: torch.Tensor
    kind_target: torch.Tensor
    delta_target: torch.Tensor
    final_guard_targets: torch.Tensor
    final_dormancy_mask: torch.Tensor
    retrieval_logits: torch.Tensor
    retrieval_eligible_mask: torch.Tensor
    retrieval_target: torch.Tensor
    retrieval_mask: torch.Tensor
    role_logits: torch.Tensor
    role_target: torch.Tensor
    role_mask: torch.Tensor
    status_logits: torch.Tensor
    status_target: torch.Tensor
    status_mask: torch.Tensor
    confidence_logit: torch.Tensor
    confidence_target: torch.Tensor
    confidence_mask: torch.Tensor
    append_support_logit: torch.Tensor
    append_support_target: torch.Tensor
    append_support_mask: torch.Tensor
    continue_search_logit: torch.Tensor
    continue_search_target: torch.Tensor
    continue_search_mask: torch.Tensor
    next_focus_logits: torch.Tensor
    focus_target: torch.Tensor
    focus_mask: torch.Tensor
    hazard_logits: torch.Tensor
    hazard_target: torch.Tensor
    hazard_mask: torch.Tensor
    log_delay: torch.Tensor
    log_delay_target: torch.Tensor
    log_delay_mask: torch.Tensor
    normalized_deadline: torch.Tensor
    normalized_deadline_target: torch.Tensor
    normalized_deadline_mask: torch.Tensor
    action_logits: torch.Tensor
    action_target: torch.Tensor
    action_mask: torch.Tensor
    abstention_mask: torch.Tensor
    lead_fraction: torch.Tensor
    lead_target: torch.Tensor
    lead_mask: torch.Tensor
    pre_jump_latent: torch.Tensor
    post_jump_latent: torch.Tensor
    jump_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class PublicInputBatch:
    """Public observation storage; use at_observation to expose only delivered data.

    This is an input staging container, never a neural feature vector. IDs and
    absolute times are host correspondence; models consume current encoded
    external features or a ModelContext.
    """

    records: "RecordTensorBatch"
    observation_kind: torch.Tensor
    observation_slots: torch.Tensor
    activation_entities: torch.Tensor
    observation_mask: torch.Tensor
    observation_time_features: torch.Tensor
    observation_times: torch.Tensor
    initial_times: tuple[float, ...]

    def __post_init__(self) -> None:
        shape = self.observation_kind.shape
        if len(shape) != 2 or shape[0] != self.records.batch_size or not 1 <= shape[1] <= 65:
            raise NeuralError("public observations must have shape [B,O] with O <= 65")
        for name in (
            "observation_kind",
            "observation_slots",
            "activation_entities",
            "observation_mask",
            "observation_time_features",
        ):
            tensor = getattr(self, name)
            expected = (*shape, 2) if name == "observation_time_features" else shape
            dtype = (
                torch.float32
                if name == "observation_time_features"
                else torch.bool
                if name == "observation_mask"
                else torch.int64
            )
            if (
                tensor.shape != expected
                or tensor.dtype != dtype
                or tensor.device != self.records.device
            ):
                raise NeuralError(f"invalid public {name} shape, dtype, or device")
            object.__setattr__(self, name, tensor.clone())
        if (
            self.observation_times.shape != shape
            or self.observation_times.dtype != torch.float64
            or self.observation_times.device.type != "cpu"
        ):
            raise NeuralError("absolute observation times must remain CPU float64")
        if len(self.initial_times) != shape[0]:
            raise NeuralError("initial times must align with public batch rows")
        object.__setattr__(self, "observation_times", self.observation_times.clone())

    def to(self, device: str) -> "PublicInputBatch":
        from silent_cascade.memory.tensor_store import RecordTensorBatch, _resolve_device

        target = _resolve_device(device)
        records = RecordTensorBatch(
            **{
                item.name: getattr(self.records, item.name).to(target)
                if isinstance(getattr(self.records, item.name), torch.Tensor)
                else getattr(self.records, item.name)
                for item in fields(self.records)
            }
        )
        return replace(
            self,
            records=records,
            **{
                name: getattr(self, name).to(target)
                for name in (
                    "observation_kind",
                    "observation_slots",
                    "activation_entities",
                    "observation_mask",
                    "observation_time_features",
                )
            },
        )

    def permute_slots(self, permutations: torch.Tensor) -> "PublicInputBatch":
        records = self.records.permute_slots(permutations)
        inverse = torch.argsort(permutations, dim=1)
        slots = self.observation_slots
        remapped = inverse.gather(1, slots.clamp_min(0))
        return replace(
            self, records=records, observation_slots=torch.where(slots >= 0, remapped, -1)
        )

    def at_observation(self, index: int) -> "PublicInputBatch":
        """Current observation and arrived memory, with all future metadata removed."""
        from silent_cascade.memory.tensor_store import RecordTensorBatch

        if type(index) is not int or not 0 <= index < self.observation_mask.shape[1]:
            raise NeuralError("observation index is out of bounds")
        now = self.observation_times[:, index].tolist()
        arrived = (
            torch.tensor(
                [
                    [time is not None and time <= current for time in row]
                    for row, current in zip(self.records.observed_at, now, strict=True)
                ],
                dtype=torch.bool,
                device=self.records.device,
            )
            & self.records.valid_mask
        )
        arrived_host = arrived.cpu().tolist()
        values = {}
        for item in fields(self.records):
            value = getattr(self.records, item.name)
            if isinstance(value, torch.Tensor):
                mask = arrived.unsqueeze(-1) if value.ndim == 3 else arrived
                fill = False if item.name == "valid_mask" else 4 if item.name == "hazard_ids" else 0
                values[item.name] = torch.where(mask, value, fill)
            else:
                fill = () if item.name == "support_ids" else None
                values[item.name] = tuple(
                    tuple(v if valid else fill for v, valid in zip(row, flags, strict=True))
                    for row, flags in zip(value, arrived_host, strict=True)
                )
        records = RecordTensorBatch(**values)
        return replace(
            self,
            records=records,
            **{
                name: getattr(self, name)[:, index : index + 1]
                for name in (
                    "observation_kind",
                    "observation_slots",
                    "activation_entities",
                    "observation_mask",
                    "observation_time_features",
                    "observation_times",
                )
            },
        )


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

    def _updated(self, **updates: object) -> "ModelContext":
        """Trusted functional replacement for validated neural transitions.

        Internal callers preserve shapes, devices and bounds by construction.
        This avoids cloning/scanning unchanged memory at every gathered step;
        public construction still performs the complete strict validation.
        """
        names = {item.name for item in fields(self)}
        if updates.keys() - names:
            raise TypeError("unknown context fields")
        result = type(self).__new__(type(self))
        for name in names:
            object.__setattr__(result, name, updates.get(name, getattr(self, name)))
        return result

    def _gather(self, rows: torch.Tensor) -> "ModelContext":
        return self._updated(
            workspace=TensorWorkspace._from_functional_update(
                self.workspace.latent[rows], self.workspace.accumulators[rows]
            ),
            **{
                item.name: getattr(self, item.name)[rows]
                for item in fields(self)
                if item.name != "workspace"
            },
        )

    def _scatter(self, rows: torch.Tensor, source: "ModelContext") -> "ModelContext":
        return self._updated(
            workspace=TensorWorkspace._from_functional_update(
                self.workspace.latent.index_copy(0, rows, source.workspace.latent),
                self.workspace.accumulators.index_copy(0, rows, source.workspace.accumulators),
            ),
            **{
                item.name: getattr(self, item.name).index_copy(0, rows, getattr(source, item.name))
                for item in fields(self)
                if item.name != "workspace"
            },
        )

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
