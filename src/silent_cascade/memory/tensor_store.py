"""Public tensor representation of immutable runtime memory metadata."""

import math
from dataclasses import dataclass, fields

import torch

from silent_cascade.memory.store import BoundedMemory
from silent_cascade.models.errors import NeuralError
from silent_cascade.schemas import Provenance, RecordKind

_MEMORY_SLOTS = 64
_MAX_BATCH_SIZE = 128
_KIND_IDS = {RecordKind.LINK: 0, RecordKind.HAZARD: 1, RecordKind.SAFE: 2}
_PROVENANCE_IDS = {
    Provenance.PERCEIVED: 0,
    Provenance.INFERRED: 1,
    Provenance.SIMULATED: 2,
}
_MISSING_HAZARD_ID = 4


def _resolve_device(device: object) -> torch.device:
    if type(device) is not str:
        raise NeuralError("device must be a string naming CPU or MPS")
    try:
        resolved = torch.device(device)
    except (RuntimeError, ValueError) as error:
        raise NeuralError("device must name CPU or MPS") from error
    if resolved.type not in {"cpu", "mps"}:
        raise NeuralError("device must name CPU or MPS")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise NeuralError("MPS memory requested but MPS is unavailable")
    return resolved


def _require_host_rows(value: object, batch_size: int, name: str) -> tuple[tuple[object, ...], ...]:
    if not isinstance(value, tuple) or len(value) != batch_size:
        raise NeuralError(f"{name} must contain one tuple per batch row")
    if any(not isinstance(row, tuple) or len(row) != _MEMORY_SLOTS for row in value):
        raise NeuralError(f"{name} rows must contain exactly 64 slots")
    return value


@dataclass(frozen=True, slots=True)
class RecordTensorBatch:
    """Batched public record features plus host-only identity correspondence."""

    subject_ids: torch.Tensor
    object_ids: torch.Tensor
    kind_ids: torch.Tensor
    hazard_ids: torch.Tensor
    provenance_ids: torch.Tensor
    scalar_features: torch.Tensor
    valid_mask: torch.Tensor
    record_ids: tuple[tuple[int | None, ...], ...]
    support_ids: tuple[tuple[tuple[int, ...], ...], ...]
    observed_at: tuple[tuple[float | None, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.subject_ids, torch.Tensor) or self.subject_ids.ndim != 2:
            raise NeuralError("subject_ids must be a rank-two tensor")
        batch_size = self.subject_ids.shape[0]
        if not 1 <= batch_size <= _MAX_BATCH_SIZE:
            raise NeuralError("record batch size must be between 1 and 128")
        device = self.subject_ids.device
        if device.type not in {"cpu", "mps"}:
            raise NeuralError("record tensors must use a CPU or MPS device")
        specifications = {
            "subject_ids": ((batch_size, _MEMORY_SLOTS), torch.int64),
            "object_ids": ((batch_size, _MEMORY_SLOTS), torch.int64),
            "kind_ids": ((batch_size, _MEMORY_SLOTS), torch.int64),
            "hazard_ids": ((batch_size, _MEMORY_SLOTS), torch.int64),
            "provenance_ids": ((batch_size, _MEMORY_SLOTS), torch.int64),
            "scalar_features": ((batch_size, _MEMORY_SLOTS, 6), torch.float32),
            "valid_mask": ((batch_size, _MEMORY_SLOTS), torch.bool),
        }
        for item in fields(self):
            if item.name not in specifications:
                continue
            tensor = getattr(self, item.name)
            shape, dtype = specifications[item.name]
            if not isinstance(tensor, torch.Tensor):
                raise NeuralError(f"{item.name} must be a tensor")
            if tensor.device != device:
                raise NeuralError("record tensors must use one device")
            if tensor.dtype is not dtype or tuple(tensor.shape) != shape:
                raise NeuralError(f"{item.name} must have shape {shape} and dtype {dtype}")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise NeuralError(f"{item.name} must contain only finite values")
            object.__setattr__(self, item.name, tensor.clone())
        _require_host_rows(self.record_ids, batch_size, "record_ids")
        _require_host_rows(self.support_ids, batch_size, "support_ids")
        _require_host_rows(self.observed_at, batch_size, "observed_at")
        if bool(((self.subject_ids < 0) | (self.subject_ids >= 64)).any()):
            raise NeuralError("subject IDs must be in [0, 64)")
        if bool(((self.object_ids < 0) | (self.object_ids >= 64)).any()):
            raise NeuralError("object IDs must be in [0, 64)")
        if bool(((self.kind_ids < 0) | (self.kind_ids >= 3)).any()):
            raise NeuralError("kind IDs must be in [0, 3)")
        if bool(((self.hazard_ids < 0) | (self.hazard_ids >= 5)).any()):
            raise NeuralError("hazard IDs must be in [0, 5)")
        if bool(((self.provenance_ids < 0) | (self.provenance_ids >= 3)).any()):
            raise NeuralError("provenance IDs must be in [0, 3)")

    @property
    def batch_size(self) -> int:
        return self.subject_ids.shape[0]

    @property
    def device(self) -> torch.device:
        return self.subject_ids.device

    @property
    def padding_mask(self) -> torch.Tensor:
        """Mark padded slots for consumers using PyTorch padding-mask semantics."""
        return ~self.valid_mask

    def permute_slots(self, permutations: torch.Tensor) -> "RecordTensorBatch":
        """Return rows gathered by complete row-wise slot permutations."""
        expected_shape = (self.batch_size, _MEMORY_SLOTS)
        if not isinstance(permutations, torch.Tensor):
            raise NeuralError("permutations must be a tensor")
        if (
            permutations.dtype is not torch.int64
            or permutations.device != self.device
            or tuple(permutations.shape) != expected_shape
        ):
            raise NeuralError("permutations must be int64 row-wise slot permutations")
        canonical = torch.arange(_MEMORY_SLOTS, device=self.device).expand(self.batch_size, -1)
        if not bool(torch.equal(torch.sort(permutations, dim=1).values, canonical)):
            raise NeuralError("each row must be a complete slot permutation")

        tensor_values: dict[str, torch.Tensor] = {}
        for name in (
            "subject_ids",
            "object_ids",
            "kind_ids",
            "hazard_ids",
            "provenance_ids",
            "scalar_features",
            "valid_mask",
        ):
            value = getattr(self, name)
            indices = permutations
            if value.ndim == 3:
                indices = permutations.unsqueeze(-1).expand(-1, -1, value.shape[-1])
            tensor_values[name] = value.gather(1, indices)
        host_indices = permutations.detach().cpu().tolist()

        def permute_host(rows: tuple[tuple[object, ...], ...]) -> tuple[tuple[object, ...], ...]:
            return tuple(
                tuple(row[index] for index in indices)
                for row, indices in zip(rows, host_indices, strict=True)
            )

        return RecordTensorBatch(
            **tensor_values,
            record_ids=permute_host(self.record_ids),
            support_ids=permute_host(self.support_ids),
            observed_at=permute_host(self.observed_at),
        )


def pack_memory(
    memories: tuple[BoundedMemory, ...],
    initial_times: tuple[float, ...],
    *,
    device: str,
) -> RecordTensorBatch:
    """Pack immutable public memory rows without adding learned or ordinal features."""
    if not isinstance(memories, tuple) or not 1 <= len(memories) <= _MAX_BATCH_SIZE:
        raise NeuralError("memories must be a tuple with between 1 and 128 rows")
    if not isinstance(initial_times, tuple) or len(initial_times) != len(memories):
        raise NeuralError("memories and initial_times must have the same length")
    resolved_device = _resolve_device(device)
    batch_size = len(memories)
    subject_ids = torch.zeros(
        (batch_size, _MEMORY_SLOTS), dtype=torch.int64, device=resolved_device
    )
    object_ids = torch.zeros_like(subject_ids)
    kind_ids = torch.zeros_like(subject_ids)
    hazard_ids = torch.full_like(subject_ids, _MISSING_HAZARD_ID)
    provenance_ids = torch.zeros_like(subject_ids)
    scalar_features = torch.zeros(
        (batch_size, _MEMORY_SLOTS, 6), dtype=torch.float32, device=resolved_device
    )
    valid_mask = torch.zeros((batch_size, _MEMORY_SLOTS), dtype=torch.bool, device=resolved_device)
    record_rows: list[tuple[int | None, ...]] = []
    support_rows: list[tuple[tuple[int, ...], ...]] = []
    observed_rows: list[tuple[float | None, ...]] = []

    for batch_index, (memory, initial_time) in enumerate(zip(memories, initial_times, strict=True)):
        if not isinstance(memory, BoundedMemory):
            raise NeuralError("memories must contain BoundedMemory values")
        if type(initial_time) is not float or not math.isfinite(initial_time):
            raise NeuralError("initial_times must contain finite exact float values")
        row_record_ids: list[int | None] = []
        row_support_ids: list[tuple[int, ...]] = []
        row_observed_at: list[float | None] = []
        for slot_index, slot in enumerate(memory.records):
            record = slot.record
            elapsed = record.observed_at - initial_time
            if elapsed < 0.0:
                raise NeuralError("record observed_at cannot be before initial_time")
            subject_ids[batch_index, slot_index] = record.subject_id
            if record.object_id is not None:
                object_ids[batch_index, slot_index] = record.object_id
            kind_ids[batch_index, slot_index] = _KIND_IDS[record.kind]
            if record.hazard_type is not None:
                hazard_ids[batch_index, slot_index] = record.hazard_type
            provenance_ids[batch_index, slot_index] = _PROVENANCE_IDS[record.provenance]
            scalar_features[batch_index, slot_index] = torch.tensor(
                (
                    float(record.object_id is not None),
                    float(record.hazard_type is not None),
                    float(record.delay is not None),
                    math.log1p(record.delay) if record.delay is not None else 0.0,
                    math.log1p(elapsed),
                    record.confidence,
                ),
                dtype=torch.float32,
                device=resolved_device,
            )
            valid_mask[batch_index, slot_index] = True
            row_record_ids.append(record.record_id)
            row_support_ids.append(record.support_ids)
            row_observed_at.append(record.observed_at)
        padding = _MEMORY_SLOTS - len(memory.records)
        record_rows.append(tuple((*row_record_ids, *(None for _ in range(padding)))))
        support_rows.append(tuple((*row_support_ids, *(() for _ in range(padding)))))
        observed_rows.append(tuple((*row_observed_at, *(None for _ in range(padding)))))

    return RecordTensorBatch(
        subject_ids=subject_ids,
        object_ids=object_ids,
        kind_ids=kind_ids,
        hazard_ids=hazard_ids,
        provenance_ids=provenance_ids,
        scalar_features=scalar_features,
        valid_mask=valid_mask,
        record_ids=tuple(record_rows),
        support_ids=tuple(support_rows),
        observed_at=tuple(observed_rows),
    )
