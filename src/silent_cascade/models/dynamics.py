"""Batched differentiable analytic flow and guard crossings."""

from dataclasses import dataclass, fields

import torch

from silent_cascade.models.errors import NeuralError
from silent_cascade.models.types import TensorWorkspace


def _validate_float_tensor(
    value: object,
    *,
    shape: tuple[int, ...],
    device: torch.device | None,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise NeuralError(f"{name} must be a tensor")
    if value.device.type not in {"cpu", "mps"}:
        raise NeuralError(f"{name} must use a CPU or MPS device")
    if device is not None and value.device != device:
        raise NeuralError("batched dynamics tensors must use one device")
    if value.dtype is not torch.float32 or tuple(value.shape) != shape:
        raise NeuralError(f"{name} must have shape {shape} and dtype torch.float32")
    if not bool(torch.isfinite(value).all()):
        raise NeuralError(f"{name} must contain only finite values")
    return value


def _require_bounds(
    value: torch.Tensor,
    lower: float,
    upper: float,
    name: str,
    *,
    strict: bool = False,
) -> None:
    low = value.new_tensor(lower)
    high = value.new_tensor(upper)
    invalid = (value <= low) | (value >= high) if strict else (value < low) | (value > high)
    if bool(invalid.any()):
        raise NeuralError(f"{name} is outside its bounds")


@dataclass(frozen=True, slots=True)
class BatchedSegmentParameters:
    """Fixed controller outputs for one batched analytic segment."""

    flow_targets: torch.Tensor
    flow_rates: torch.Tensor
    raw_guard_targets: torch.Tensor
    guard_targets: torch.Tensor
    guard_rates: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.flow_targets, torch.Tensor) or self.flow_targets.ndim != 2:
            raise NeuralError("flow_targets must be a rank-two tensor")
        batch_size = self.flow_targets.shape[0]
        if not 1 <= batch_size <= 128:
            raise NeuralError("parameter batch size must be between 1 and 128")
        device = self.flow_targets.device
        specifications = {
            "flow_targets": (batch_size, 456),
            "flow_rates": (batch_size, 456),
            "raw_guard_targets": (batch_size, 3),
            "guard_targets": (batch_size, 3),
            "guard_rates": (batch_size, 3),
        }
        for item in fields(self):
            value = _validate_float_tensor(
                getattr(self, item.name),
                shape=specifications[item.name],
                device=device,
                name=item.name,
            )
            object.__setattr__(self, item.name, value.clone())
        _require_bounds(self.flow_targets, -1.0, 1.0, "flow targets")
        _require_bounds(self.flow_rates, 1e-5, 20.0, "flow rates")
        _require_bounds(self.raw_guard_targets, 0.0, 2.0, "raw guard targets", strict=True)
        _require_bounds(self.guard_targets, 0.0, 2.0, "guard targets", strict=True)
        _require_bounds(self.guard_rates, 1e-5, 500.0, "guard rates")

    @property
    def batch_size(self) -> int:
        return self.flow_targets.shape[0]

    @property
    def device(self) -> torch.device:
        return self.flow_targets.device


def _flow_values(
    origin: torch.Tensor, target: torch.Tensor, rate: torch.Tensor, dt: torch.Tensor
) -> torch.Tensor:
    weight = -torch.expm1(-rate * dt[:, None])
    return origin + weight * (target - origin)


def flow_batch(
    state: TensorWorkspace,
    parameters: BatchedSegmentParameters,
    dt: torch.Tensor,
) -> TensorWorkspace:
    """Advance each row from the caller-retained segment anchor by its elapsed time."""
    if not isinstance(state, TensorWorkspace):
        raise TypeError("state must be a TensorWorkspace")
    if not isinstance(parameters, BatchedSegmentParameters):
        raise TypeError("parameters must be BatchedSegmentParameters")
    if parameters.batch_size != state.batch_size or parameters.device != state.device:
        raise NeuralError("state and segment parameters must share a batch and device")
    elapsed = _validate_float_tensor(
        dt,
        shape=(state.batch_size,),
        device=state.device,
        name="dt",
    )
    if bool((elapsed < 0.0).any()):
        raise NeuralError("dt must be nonnegative")
    return TensorWorkspace._from_functional_update(
        _flow_values(state.latent, parameters.flow_targets, parameters.flow_rates, elapsed),
        _flow_values(
            state.accumulators,
            parameters.guard_targets,
            parameters.guard_rates,
            elapsed,
        ),
    )


def _crossing_values(
    accumulators: torch.Tensor, targets: torch.Tensor, rates: torch.Tensor
) -> torch.Tensor:
    active = (accumulators < 1.0) & (targets > 1.0)
    result = torch.full_like(accumulators, torch.inf)
    if bool(active.any()):
        offsets = (
            torch.log1p((1.0 - accumulators[active]) / (targets[active] - 1.0)) / rates[active]
        )
        result = result.masked_scatter(active, offsets)
    return result


def crossings_batch(
    accumulators: torch.Tensor, targets: torch.Tensor, rates: torch.Tensor
) -> torch.Tensor:
    """Return differentiable crossing offsets with inf only for dormant entries."""
    if not isinstance(accumulators, torch.Tensor) or accumulators.ndim != 2:
        raise NeuralError("accumulators must be a rank-two tensor")
    batch_size = accumulators.shape[0]
    shape = (batch_size, 3)
    device = accumulators.device
    accumulators = _validate_float_tensor(
        accumulators, shape=shape, device=device, name="accumulators"
    )
    targets = _validate_float_tensor(targets, shape=shape, device=device, name="targets")
    rates = _validate_float_tensor(rates, shape=shape, device=device, name="rates")
    _require_bounds(accumulators, 0.0, 2.0, "accumulators")
    _require_bounds(targets, 0.0, 2.0, "targets", strict=True)
    _require_bounds(rates, 1e-5, 500.0, "rates")
    return _crossing_values(accumulators, targets, rates)
