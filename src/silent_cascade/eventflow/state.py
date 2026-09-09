"""Validated float32 state and immutable anchors for analytic event-flow segments.

Frozen containers own clones at construction boundaries; callers must still treat
their exposed PyTorch tensors as read-only. Cloning retains autograd graphs.
Latent channels stay in [-1, 1]. Guard accumulators instead lie in [0, 2]
between events (including rounded asymptotes), and causal jumps reset them to zero.
"""

import math
import re
from dataclasses import dataclass, field, fields, replace

import torch

from silent_cascade.errors import DynamicsError, TimeOrderError
from silent_cascade.memory import BoundedMemory
from silent_cascade.schemas import Action, Hypothesis, Mode

INTERNAL_EVENT_ID_BASE = 1 << 62

_CHANNEL_DIMENSIONS = {
    "z_fast": 256,
    "z_slow": 64,
    "drives": 8,
    "focus_key": 64,
    "hypothesis_latent": 64,
}


def require_time(value: float, name: str) -> float:
    """Validate an absolute host timestamp before any device conversion."""
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise TimeOrderError(f"{name} must be a finite nonnegative host float")
    return value


def _validate_tensor(value: torch.Tensor, size: int, device: torch.device, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise DynamicsError(f"{name} must be a tensor")
    if value.device.type not in {"cpu", "mps"} or value.device != device:
        raise DynamicsError(f"{name} must use the declared CPU/MPS device")
    if value.dtype is not torch.float32:
        raise DynamicsError(f"{name} must be float32")
    if value.shape != (size,):
        raise DynamicsError(f"{name} must have shape ({size},)")
    if not bool(torch.isfinite(value).all()):
        raise DynamicsError(f"{name} must be finite")


def _require_bounds(
    value: torch.Tensor, lower: float, upper: float, name: str, *, strict: bool = False
) -> None:
    # Compare against device/dtype representations, not widened host values:
    # float32(1e-5) is slightly below the float64 literal and is a legal endpoint.
    low, high = value.new_tensor(lower), value.new_tensor(upper)
    invalid = (value <= low) | (value >= high) if strict else (value < low) | (value > high)
    if bool(invalid.any()):
        raise DynamicsError(f"{name} is outside its bounds")


@dataclass(frozen=True, slots=True)
class ContinuousChannels:
    """Five same-device tensor channels, also used for latent targets and rates."""

    z_fast: torch.Tensor
    z_slow: torch.Tensor
    drives: torch.Tensor
    focus_key: torch.Tensor
    hypothesis_latent: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.z_fast, torch.Tensor):
            raise DynamicsError("z_fast must be a tensor")
        for name, size in _CHANNEL_DIMENSIONS.items():
            value = getattr(self, name)
            _validate_tensor(value, size, self.device, name)
            object.__setattr__(self, name, value.clone())

    @property
    def device(self) -> torch.device:
        return self.z_fast.device


@dataclass(frozen=True, slots=True)
class ContinuousState(ContinuousChannels):
    guard_accumulators: torch.Tensor

    def __post_init__(self) -> None:
        ContinuousChannels.__post_init__(self)
        for name in _CHANNEL_DIMENSIONS:
            _require_bounds(getattr(self, name), -1.0, 1.0, name)
        _validate_tensor(self.guard_accumulators, 3, self.device, "guard_accumulators")
        _require_bounds(self.guard_accumulators, 0.0, 2.0, "guard_accumulators")
        object.__setattr__(self, "guard_accumulators", self.guard_accumulators.clone())


@dataclass(frozen=True, slots=True)
class SegmentParameters:
    flow_targets: ContinuousChannels
    flow_rates: ContinuousChannels
    guard_targets: torch.Tensor
    guard_rates: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.flow_targets, ContinuousChannels) or not isinstance(
            self.flow_rates, ContinuousChannels
        ):
            raise DynamicsError("flow targets and rates must be ContinuousChannels")
        device = self.flow_targets.device
        if self.flow_rates.device != device:
            raise DynamicsError("flow parameters must use one declared device")
        for name in _CHANNEL_DIMENSIONS:
            _require_bounds(getattr(self.flow_targets, name), -1.0, 1.0, f"{name} target")
            _require_bounds(getattr(self.flow_rates, name), 1e-5, 20.0, f"{name} rate")
        _validate_tensor(self.guard_targets, 3, device, "guard_targets")
        _validate_tensor(self.guard_rates, 3, device, "guard_rates")
        _require_bounds(self.guard_targets, 0.0, 2.0, "guard_targets", strict=True)
        _require_bounds(self.guard_rates, 1e-5, 500.0, "guard_rates")
        object.__setattr__(self, "flow_targets", replace(self.flow_targets))
        object.__setattr__(self, "flow_rates", replace(self.flow_rates))
        object.__setattr__(self, "guard_targets", self.guard_targets.clone())
        object.__setattr__(self, "guard_rates", self.guard_rates.clone())


@dataclass(frozen=True, slots=True)
class ComputeCounters:
    flow_evaluations: int = 0
    checkpoint_flow_evaluations: int = 0
    controller_calls: int = 0
    foundation_model_calls: int = 0
    guard_predictions: int = 0
    jump_applications: int = 0
    records_scored: int = 0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if type(value) is not int or value < 0:
                raise DynamicsError(f"{item.name} must be a nonnegative exact integer")


@dataclass(frozen=True, slots=True)
class AnalyticSegment:
    started_at: float
    origin: ContinuousState
    parameters: SegmentParameters
    parent_event_id: int
    prediction_snapshot_sha256: str

    def __post_init__(self) -> None:
        require_time(self.started_at, "started_at")
        if type(self.parent_event_id) is not int or self.parent_event_id < 0:
            raise DynamicsError("parent_event_id must be a nonnegative exact integer")
        if not isinstance(self.prediction_snapshot_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.prediction_snapshot_sha256
        ):
            raise DynamicsError("prediction_snapshot_sha256 must be a canonical SHA256 digest")
        if not isinstance(self.origin, ContinuousState) or not isinstance(
            self.parameters, SegmentParameters
        ):
            raise DynamicsError("segment requires continuous origin and parameters")
        if self.origin.device != self.parameters.flow_targets.device:
            raise DynamicsError("segment origin and parameters must use one declared device")
        object.__setattr__(self, "origin", replace(self.origin))
        object.__setattr__(self, "parameters", replace(self.parameters))


@dataclass(frozen=True, slots=True)
class RuntimeCore:
    """Public-derived state only; causal transition validation belongs to jumps."""

    continuous: ContinuousState
    mode: Mode = Mode.OBSERVING
    memory: BoundedMemory = field(default_factory=lambda: BoundedMemory(64))
    focus_node_id: int | None = None
    active_record_id: int | None = None
    support_ids: tuple[int, ...] = ()
    hypothesis: Hypothesis | None = None
    activation_time: float | None = None
    actions: tuple[Action, ...] = ()
    same_kind_refractory_until: tuple[float, float, float] = (0.0, 0.0, 0.0)
    executed_internal_events: int = 0
    consecutive_gap_clamps: int = 0
    last_event_id: int | None = None
    counters: ComputeCounters = field(default_factory=ComputeCounters)

    def __post_init__(self) -> None:
        if not isinstance(self.continuous, ContinuousState):
            raise DynamicsError("continuous must be a ContinuousState")
        object.__setattr__(self, "continuous", replace(self.continuous))


@dataclass(frozen=True, slots=True)
class RuntimeState:
    core: RuntimeCore
    segment: AnalyticSegment
    time: float

    def __post_init__(self) -> None:
        require_time(self.time, "time")
        if self.time < self.segment.started_at:
            raise TimeOrderError("runtime time cannot precede its segment origin")
        if self.core.continuous.device != self.segment.origin.device:
            raise DynamicsError("runtime and segment must use one declared device")
        object.__setattr__(self, "core", replace(self.core))
        object.__setattr__(self, "segment", replace(self.segment))


def make_initial_continuous_state(*, device: str | torch.device = "cpu") -> ContinuousState:
    """Construct zero latent channels and reset guards on the declared device."""
    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "mps"}:
        raise DynamicsError("initial state requires a CPU/MPS device")
    return ContinuousState(
        **{
            name: torch.zeros(size, dtype=torch.float32, device=resolved_device)
            for name, size in _CHANNEL_DIMENSIONS.items()
        },
        guard_accumulators=torch.zeros(3, dtype=torch.float32, device=resolved_device),
    )
