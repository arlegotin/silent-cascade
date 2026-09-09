"""Exact, segment-anchored endogenous guard crossings.

The runtime path deliberately uses host ``float`` arithmetic for autonomous
crossing times.  The tensor helper is a separate differentiable training
surface and never evaluates a logarithm for a dormant element.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import torch

from silent_cascade.errors import DynamicsError
from silent_cascade.eventflow.state import (
    INTERNAL_EVENT_ID_BASE,
    ContinuousState,
    RuntimeState,
    SegmentParameters,
    require_time,
)
from silent_cascade.schemas import InternalEventKind, Mode

_GUARD_COUNT = 3
_GUARD_RATE_MIN = float(torch.tensor(1.0e-5, dtype=torch.float32).item())
_GUARD_RATE_MAX = 500.0
_GUARD_THRESHOLD = 1.0

# Index order is also the deterministic within-internal priority order.
GUARD_KIND_BY_INDEX = (
    InternalEventKind.RECALL,
    InternalEventKind.COMPOSE,
    InternalEventKind.ACT,
)

_ALLOWED_MASK_BY_MODE = {
    Mode.OBSERVING: (False, False, False),
    Mode.SEARCHING: (True, False, False),
    Mode.HAVE_MEMORY: (False, True, False),
    Mode.HOLDING_HAZARD: (False, False, True),
    Mode.QUIESCENT: (False, False, False),
    Mode.TERMINAL: (False, False, False),
}
_PRIORITY_BY_KIND = {
    InternalEventKind.ACT: 0,
    InternalEventKind.COMPOSE: 1,
    InternalEventKind.RECALL: 2,
}


@dataclass(frozen=True, slots=True)
class GuardCrossing:
    """One legal, absolute-time endogenous candidate derived from a segment."""

    event_id: int
    parent_event_id: int
    timestamp: float
    kind: InternalEventKind
    guard_index: int
    predicted_delta: float
    prediction_snapshot_sha256: str

    def __post_init__(self) -> None:
        if type(self.event_id) is not int or self.event_id < INTERNAL_EVENT_ID_BASE:
            raise DynamicsError("guard crossing requires a reserved internal event ID")
        if type(self.parent_event_id) is not int or self.parent_event_id < 0:
            raise DynamicsError("guard crossing parent ID must be a nonnegative exact integer")
        require_time(self.timestamp, "crossing timestamp")
        if self.kind not in GUARD_KIND_BY_INDEX:
            raise DynamicsError("guard crossing kind must be endogenous")
        if type(self.guard_index) is not int or not 0 <= self.guard_index < _GUARD_COUNT:
            raise DynamicsError("guard crossing index is out of bounds")
        if GUARD_KIND_BY_INDEX[self.guard_index] is not self.kind:
            raise DynamicsError("guard crossing index and kind disagree")
        if type(self.predicted_delta) is not float or not math.isfinite(self.predicted_delta):
            raise DynamicsError("guard crossing delta must be a finite host float")
        if self.predicted_delta < 0.0:
            raise DynamicsError("guard crossing delta must be nonnegative")
        if (
            not isinstance(self.prediction_snapshot_sha256, str)
            or len(self.prediction_snapshot_sha256) != 64
        ):
            raise DynamicsError("guard crossing requires a canonical segment snapshot")

    @property
    def predicted_at(self) -> float:
        """Explicit alias for callers that distinguish prediction time from event time."""
        return self.timestamp


def _validate_host_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise DynamicsError(f"{name} must be a finite host float")
    return value


def _validate_guard_scalar(a: float, asymptote: float, rate: float) -> None:
    _validate_host_float(a, "guard accumulator")
    _validate_host_float(asymptote, "guard asymptote")
    _validate_host_float(rate, "guard rate")
    if not 0.0 <= a <= 2.0:
        raise DynamicsError("guard accumulator is outside [0, 2]")
    if not 0.0 < asymptote < 2.0:
        raise DynamicsError("guard asymptote is outside (0, 2)")
    # The lower endpoint is the value representable by the float32 source
    # tensor.  A prior host representable float is illegal, rather than being
    # admitted through a broad numerical tolerance.
    if not _GUARD_RATE_MIN <= rate <= _GUARD_RATE_MAX:
        raise DynamicsError("guard rate is outside [1e-5, 500]")


def crossing_offset_host(a: float, asymptote: float, rate: float) -> float | None:
    """Return a host-float64 threshold offset, or ``None`` for a dormant guard."""
    _validate_guard_scalar(a, asymptote, rate)
    if not a < _GUARD_THRESHOLD or not asymptote > _GUARD_THRESHOLD:
        return None
    return math.log1p((_GUARD_THRESHOLD - a) / (asymptote - _GUARD_THRESHOLD)) / rate


def _validate_guard_tensors(
    accumulators: torch.Tensor, asymptotes: torch.Tensor, rates: torch.Tensor
) -> None:
    values = {
        "guard accumulators": accumulators,
        "guard asymptotes": asymptotes,
        "guard rates": rates,
    }
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise DynamicsError(f"{name} must be a tensor")
        if value.dtype is not torch.float32 or value.device.type not in {"cpu", "mps"}:
            raise DynamicsError(f"{name} must be a float32 CPU/MPS tensor")
        if value.shape != (_GUARD_COUNT,):
            raise DynamicsError(f"{name} must have shape ({_GUARD_COUNT},)")
        if not bool(torch.isfinite(value).all()):
            raise DynamicsError(f"{name} must be finite")
    if asymptotes.device != accumulators.device or rates.device != accumulators.device:
        raise DynamicsError("guard tensors must share one device")
    if bool(((accumulators < 0.0) | (accumulators > 2.0)).any()):
        raise DynamicsError("guard accumulators are outside [0, 2]")
    if bool(((asymptotes <= 0.0) | (asymptotes >= 2.0)).any()):
        raise DynamicsError("guard asymptotes are outside (0, 2)")
    lower = rates.new_tensor(1.0e-5)
    if bool(((rates < lower) | (rates > 500.0)).any()):
        raise DynamicsError("guard rates are outside [1e-5, 500]")


def crossing_offsets_tensor(
    accumulators: torch.Tensor, asymptotes: torch.Tensor, rates: torch.Tensor
) -> torch.Tensor:
    """Differentiable float32 offsets, with ``inf`` for dormant guard elements."""
    _validate_guard_tensors(accumulators, asymptotes, rates)
    result = torch.full_like(accumulators, torch.inf)
    active = (accumulators < _GUARD_THRESHOLD) & (asymptotes > _GUARD_THRESHOLD)
    if bool(active.any()):
        active_accumulators = accumulators[active]
        active_asymptotes = asymptotes[active]
        active_rates = rates[active]
        result[active] = (
            torch.log1p(
                (_GUARD_THRESHOLD - active_accumulators) / (active_asymptotes - _GUARD_THRESHOLD)
            )
            / active_rates
        )
    return result


def _frame(tag: bytes, payload: bytes) -> bytes:
    return struct.pack("!I", len(tag)) + tag + struct.pack("!Q", len(payload)) + payload


def _tensor_frame(tag: bytes, value: torch.Tensor) -> bytes:
    if value.dtype is not torch.float32:
        raise DynamicsError("prediction snapshots require float32 tensors")
    detached = value.detach().to(device="cpu").contiguous()
    dtype = str(detached.dtype).encode("ascii")
    shape = struct.pack("!I", detached.ndim) + b"".join(
        struct.pack("!q", dimension) for dimension in detached.shape
    )
    data = detached.view(torch.uint8).numpy().tobytes()
    return _frame(tag, _frame(b"dtype", dtype) + _frame(b"shape", shape) + _frame(b"bytes", data))


def _integer_frame(tag: bytes, value: int) -> bytes:
    if type(value) is not int or value < 0:
        raise DynamicsError("snapshot parent ID must be a nonnegative exact integer")
    width = max(1, (value.bit_length() + 7) // 8)
    return _frame(tag, value.to_bytes(width, byteorder="big"))


def prediction_snapshot_sha256(
    origin: ContinuousState,
    parameters: SegmentParameters,
    *,
    started_at: float,
    allowed_mode_mask: tuple[bool, bool, bool],
    parent_event_id: int,
) -> str:
    """Hash canonical guard prediction inputs without importing flow or scheduling."""
    if not isinstance(origin, ContinuousState) or not isinstance(parameters, SegmentParameters):
        raise DynamicsError("prediction snapshots require segment origin and parameters")
    require_time(started_at, "segment started_at")
    if (
        not isinstance(allowed_mode_mask, tuple)
        or len(allowed_mode_mask) != _GUARD_COUNT
        or any(type(item) is not bool for item in allowed_mode_mask)
    ):
        raise DynamicsError("allowed mode mask must be three exact booleans")
    if origin.device != parameters.guard_targets.device:
        raise DynamicsError("prediction snapshot tensors must share one device")
    encoded = b"".join(
        (
            _frame(b"silent-cascade.guard-prediction.v1", b""),
            _tensor_frame(b"origin.guard_accumulators", origin.guard_accumulators),
            _tensor_frame(b"parameters.guard_targets", parameters.guard_targets),
            _tensor_frame(b"parameters.guard_rates", parameters.guard_rates),
            _frame(b"segment.started_at.float64", struct.pack("!d", started_at)),
            _frame(b"allowed_mode_mask", bytes(allowed_mode_mask)),
            _integer_frame(b"parent_event_id", parent_event_id),
        )
    )
    return hashlib.sha256(encoded).hexdigest()


def _crossing_sort_key(crossing: GuardCrossing) -> tuple[float, int, int]:
    """Stable internal ordering; current legal modes expose at most one kind."""
    return (crossing.timestamp, _PRIORITY_BY_KIND[crossing.kind], crossing.guard_index)


def next_crossings(runtime: RuntimeState) -> list[GuardCrossing]:
    """Return legal future endogenous candidates from the immutable segment origin.

    There is intentionally no external-horizon parameter.  The queue layer owns
    external preemption.  A guard crossed before the materialized runtime time
    only remains eligible when its same-kind refractory release is still future
    (or exactly now).
    """
    if not isinstance(runtime, RuntimeState):
        raise DynamicsError("guard prediction requires RuntimeState")
    try:
        allowed_mask = _ALLOWED_MASK_BY_MODE[runtime.core.mode]
    except KeyError as error:
        raise DynamicsError("runtime mode is invalid") from error
    if not any(allowed_mask):
        return []

    origin = runtime.segment.origin.guard_accumulators
    parameters = runtime.segment.parameters
    candidates: list[GuardCrossing] = []
    for guard_index, allowed in enumerate(allowed_mask):
        if not allowed:
            continue
        offset = crossing_offset_host(
            float(origin[guard_index].item()),
            float(parameters.guard_targets[guard_index].item()),
            float(parameters.guard_rates[guard_index].item()),
        )
        if offset is None:
            continue
        mathematical_time = runtime.segment.started_at + offset
        refractory_until = runtime.core.same_kind_refractory_until[guard_index]
        if type(refractory_until) is not float or not math.isfinite(refractory_until):
            raise DynamicsError("same-kind refractory times must be finite host floats")
        timestamp = max(mathematical_time, refractory_until)
        if timestamp < runtime.time:
            continue
        kind = GUARD_KIND_BY_INDEX[guard_index]
        candidates.append(
            GuardCrossing(
                event_id=INTERNAL_EVENT_ID_BASE + runtime.core.executed_internal_events,
                parent_event_id=runtime.segment.parent_event_id,
                timestamp=timestamp,
                kind=kind,
                guard_index=guard_index,
                predicted_delta=timestamp - runtime.time,
                prediction_snapshot_sha256=runtime.segment.prediction_snapshot_sha256,
            )
        )
    return sorted(candidates, key=_crossing_sort_key)
