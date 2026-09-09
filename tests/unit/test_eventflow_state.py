"""Construction boundaries for immutable, float32 event-flow state."""

from dataclasses import FrozenInstanceError, fields, replace

import pytest
import torch

from silent_cascade.errors import DynamicsError, TimeOrderError
from silent_cascade.eventflow.state import (
    AnalyticSegment,
    ComputeCounters,
    ContinuousChannels,
    ContinuousState,
    RuntimeCore,
    RuntimeState,
    SegmentParameters,
    make_initial_continuous_state,
)


def channels(value: float, *, device: str = "cpu") -> ContinuousChannels:
    return ContinuousChannels(
        z_fast=torch.full((256,), value, dtype=torch.float32, device=device),
        z_slow=torch.full((64,), value, dtype=torch.float32, device=device),
        drives=torch.full((8,), value, dtype=torch.float32, device=device),
        focus_key=torch.full((64,), value, dtype=torch.float32, device=device),
        hypothesis_latent=torch.full((64,), value, dtype=torch.float32, device=device),
    )


def parameters(*, device: str = "cpu", rate: float = 0.7) -> SegmentParameters:
    return SegmentParameters(
        flow_targets=channels(0.75, device=device),
        flow_rates=channels(rate, device=device),
        guard_targets=torch.tensor([0.5, 1.25, 1.75], dtype=torch.float32, device=device),
        guard_rates=torch.tensor([0.1, 2.0, 500.0], dtype=torch.float32, device=device),
    )


def test_initial_state_has_declared_shapes_zero_values_and_device() -> None:
    state = make_initial_continuous_state(device="cpu")
    expected = {
        "z_fast": 256,
        "z_slow": 64,
        "drives": 8,
        "guard_accumulators": 3,
        "focus_key": 64,
        "hypothesis_latent": 64,
    }
    for name, size in expected.items():
        value = getattr(state, name)
        assert value.shape == (size,)
        assert value.dtype is torch.float32
        assert value.device == torch.device("cpu")
        assert torch.count_nonzero(value) == 0
    with pytest.raises(FrozenInstanceError):
        state.drives = torch.zeros(8)


@pytest.mark.parametrize("name", [f.name for f in fields(ContinuousState)])
@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "infinity", "lower", "upper"])
def test_state_rejects_invalid_tensor_channels(name: str, fault: str) -> None:
    state = make_initial_continuous_state(device="cpu")
    value = getattr(state, name).clone()
    if fault == "shape":
        value = value.unsqueeze(0)
    elif fault == "dtype":
        value = value.to(torch.float64)
    else:
        value[0] = {
            "nan": float("nan"),
            "infinity": float("inf"),
            "lower": -0.1 if name == "guard_accumulators" else -1.1,
            "upper": 2.1 if name == "guard_accumulators" else 1.1,
        }[fault]
    with pytest.raises(DynamicsError):
        replace(state, **{name: value})


def test_state_rejects_unsupported_device_before_attempting_tensor_values() -> None:
    state = make_initial_continuous_state(device="cpu")
    with pytest.raises(DynamicsError, match="device"):
        replace(state, drives=torch.empty(8, device="meta"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_state_and_parameters_reject_mixed_cpu_mps_devices() -> None:
    state = make_initial_continuous_state(device="cpu")
    with pytest.raises(DynamicsError, match="device"):
        replace(state, drives=state.drives.to("mps"))
    with pytest.raises(DynamicsError, match="device"):
        replace(parameters(), flow_rates=channels(0.1, device="mps"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("flow_targets", -1.1),
        ("flow_targets", 1.1),
        ("flow_rates", 0.0),
        ("flow_rates", 9e-6),
        ("flow_rates", 20.1),
    ],
)
def test_segment_rejects_targets_or_rates_outside_latent_bounds(field: str, value: float) -> None:
    with pytest.raises(DynamicsError):
        replace(parameters(), **{field: channels(value)})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("guard_targets", 0.0),
        ("guard_targets", 2.0),
        ("guard_rates", 0.0),
        ("guard_rates", 9e-6),
        ("guard_rates", 500.1),
    ],
)
def test_segment_rejects_guard_parameters_outside_their_separate_bounds(
    field: str,
    value: float,
) -> None:
    with pytest.raises(DynamicsError):
        replace(parameters(), **{field: torch.full((3,), value)})


def test_rate_minimum_accepts_float32_representation_but_not_previous_float() -> None:
    minimum = torch.tensor(1e-5, dtype=torch.float32)
    assert minimum.item() < 1e-5
    accepted = replace(parameters(rate=1e-5), guard_rates=minimum.expand(3))
    assert accepted.guard_rates[0] == minimum
    below = torch.nextafter(minimum, torch.tensor(0.0))
    with pytest.raises(DynamicsError):
        replace(accepted, guard_rates=below.expand(3))
    with pytest.raises(DynamicsError):
        replace(accepted, flow_rates=channels(below.item()))


def test_construction_clones_inputs_and_preserves_autograd() -> None:
    input_tensor = torch.full((256,), 0.2, requires_grad=True)
    state = replace(make_initial_continuous_state(device="cpu"), z_fast=input_tensor)
    assert state.z_fast.data_ptr() != input_tensor.data_ptr()
    state.z_fast.sum().backward()
    torch.testing.assert_close(input_tensor.grad, torch.ones_like(input_tensor))
    core = RuntimeCore(continuous=state)
    param = parameters()
    segment = AnalyticSegment(0.0, state, param, 0, "a" * 64)
    runtime = RuntimeState(core, segment, 0.0)
    with torch.no_grad():
        input_tensor.fill_(0.8)
        state.z_fast.fill_(-0.5)
        param.flow_targets.z_fast.fill_(-0.5)
    assert runtime.core.continuous.z_fast[0].item() == pytest.approx(0.2)
    assert runtime.segment.origin.z_fast[0].item() == pytest.approx(0.2)
    assert runtime.segment.parameters.flow_targets.z_fast[0].item() == 0.75
    assert runtime.segment.origin.z_fast.data_ptr() != runtime.core.continuous.z_fast.data_ptr()


@pytest.mark.parametrize("time", [float("nan"), float("inf"), -float("inf"), -0.1])
def test_segment_rejects_invalid_absolute_start_time(time: float) -> None:
    with pytest.raises(TimeOrderError):
        AnalyticSegment(time, make_initial_continuous_state(), parameters(), 0, "a" * 64)


def test_runtime_time_must_not_precede_its_origin() -> None:
    state = make_initial_continuous_state()
    segment = AnalyticSegment(2.0, state, parameters(), 0, "a" * 64)
    with pytest.raises(TimeOrderError):
        RuntimeState(RuntimeCore(state), segment, 1.0)


@pytest.mark.parametrize("value", [-1, True, 0.1])
@pytest.mark.parametrize("field", [f.name for f in fields(ComputeCounters)])
def test_compute_counters_reject_invalid_counts(field: str, value: object) -> None:
    with pytest.raises(DynamicsError):
        ComputeCounters(**{field: value})


@pytest.mark.parametrize("bad_hash", ["", "a" * 63, "g" * 64, "A" * 64])
def test_segment_requires_a_canonical_prediction_digest(bad_hash: str) -> None:
    with pytest.raises(DynamicsError):
        AnalyticSegment(0.0, make_initial_continuous_state(), parameters(), 0, bad_hash)
