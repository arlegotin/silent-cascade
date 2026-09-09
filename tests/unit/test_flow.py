"""Independent host-float64 reference and anchored device-flow checks."""

import math
from dataclasses import fields, replace

import pytest
import torch

from silent_cascade.errors import DynamicsError, TimeOrderError
from silent_cascade.eventflow.flow import advance_to, start_segment, state_at
from silent_cascade.eventflow.state import (
    AnalyticSegment,
    ContinuousChannels,
    ContinuousState,
    RuntimeCore,
    RuntimeState,
    SegmentParameters,
    make_initial_continuous_state,
)


def parameters(*, device: str = "cpu") -> SegmentParameters:
    dimensions = {
        "z_fast": 256,
        "z_slow": 64,
        "drives": 8,
        "focus_key": 64,
        "hypothesis_latent": 64,
    }
    return SegmentParameters(
        flow_targets=ContinuousChannels(
            **{
                name: torch.full((size,), 0.75, dtype=torch.float32, device=device)
                for name, size in dimensions.items()
            }
        ),
        flow_rates=ContinuousChannels(
            **{
                name: torch.full((size,), 0.7, dtype=torch.float32, device=device)
                for name, size in dimensions.items()
            }
        ),
        guard_targets=torch.tensor([0.5, 1.25, 1.75], dtype=torch.float32, device=device),
        guard_rates=torch.tensor([0.1, 2.0, 500.0], dtype=torch.float32, device=device),
    )


def runtime(*, started_at: float = 0.0, device: str = "cpu") -> RuntimeState:
    return start_segment(
        RuntimeCore(make_initial_continuous_state(device=device)),
        parameters(device=device),
        time=started_at,
        parent_event_id=0,
        prediction_snapshot_sha256="a" * 64,
    )


def scalar_reference(origin: float, target: float, rate: float, dt: float) -> float:
    """Independent scalar math library reference, exclusively in tests."""
    return origin + (target - origin) * -math.expm1(-rate * dt)


def test_zero_advance_preserves_runtime_identity_and_counters() -> None:
    original = runtime(started_at=3.0)
    assert advance_to(original, 3.0) is original
    assert original.core.counters.flow_evaluations == 0


@pytest.mark.parametrize("target_time", [-0.1, float("nan"), float("inf"), -float("inf")])
def test_advance_rejects_negative_or_nonfinite_time(target_time: float) -> None:
    with pytest.raises(TimeOrderError):
        advance_to(runtime(), target_time)


def test_advance_cannot_rewind_materialized_state_but_snapshot_can() -> None:
    initial = runtime(started_at=10.0)
    current = advance_to(initial, 12.0)
    with pytest.raises(TimeOrderError):
        advance_to(current, 11.0)
    with pytest.raises(TimeOrderError):
        state_at(current, 9.0)
    historical = state_at(current, 11.0)
    expected = state_at(initial, 11.0)
    for f in fields(ContinuousState):
        assert torch.equal(getattr(historical, f.name), getattr(expected, f.name))


@pytest.mark.parametrize("dt", [1e-15, 1e-12, 1e-9, 0.25, 1e3, 1e6, 1e12])
def test_all_channels_match_independent_float64_reference(dt: float) -> None:
    initial = runtime()
    actual = advance_to(initial, dt).core.continuous
    param = initial.segment.parameters
    for f in fields(ContinuousState):
        origin = getattr(initial.segment.origin, f.name)
        target = (
            param.guard_targets
            if f.name == "guard_accumulators"
            else getattr(param.flow_targets, f.name)
        )
        rate = (
            param.guard_rates
            if f.name == "guard_accumulators"
            else getattr(param.flow_rates, f.name)
        )
        expected = torch.tensor(
            [
                scalar_reference(x, goal, r, dt)
                for x, goal, r in zip(origin.tolist(), target.tolist(), rate.tolist(), strict=True)
            ],
            dtype=torch.float64,
        )
        result = getattr(actual, f.name)
        assert torch.isfinite(result).all()
        assert result.dtype is torch.float32
        torch.testing.assert_close(result.to(torch.float64), expected, rtol=3e-7, atol=1e-25)


def test_materialized_pauses_preserve_bit_identical_cpu_segment_evaluation() -> None:
    initial = runtime(started_at=100.0)
    direct = advance_to(initial, 100.91)
    paused = advance_to(initial, 100.13)
    resumed = advance_to(advance_to(paused, 100.27), 100.91)
    for f in fields(ContinuousState):
        assert torch.equal(
            getattr(direct.core.continuous, f.name), getattr(resumed.core.continuous, f.name)
        )
        assert torch.equal(
            getattr(initial.segment.origin, f.name), getattr(resumed.segment.origin, f.name)
        )
    assert direct.core.counters.flow_evaluations == 1
    assert resumed.core.counters.flow_evaluations == 3
    assert resumed.core.counters.controller_calls == 0
    assert resumed.core.counters.foundation_model_calls == 0


@pytest.mark.parametrize("dt", [0.0, 0.25])
def test_snapshots_never_alias_or_mutate_runtime_and_do_not_increment_counters(dt: float) -> None:
    initial = runtime()
    snapshot = state_at(initial, dt)
    assert initial.core.counters.flow_evaluations == 0
    for f in fields(ContinuousState):
        original = getattr(initial.core.continuous, f.name)
        sampled = getattr(snapshot, f.name)
        assert original.data_ptr() != sampled.data_ptr()
        sampled.zero_()
        assert torch.count_nonzero(original) == 0
    assert initial.segment.parameters.flow_targets.z_fast[0] == 0.75


def test_nonzero_advance_owns_all_tensors_and_leaves_input_unchanged() -> None:
    initial = runtime()
    advanced = advance_to(initial, 0.25)
    for f in fields(ContinuousState):
        result = getattr(advanced.core.continuous, f.name)
        original = getattr(initial.core.continuous, f.name)
        assert result.data_ptr() != original.data_ptr()
        result.zero_()
        assert torch.count_nonzero(original) == 0
    advanced.segment.origin.z_fast.fill_(0.5)
    advanced.segment.parameters.flow_targets.z_fast.fill_(-0.5)
    assert torch.count_nonzero(initial.segment.origin.z_fast) == 0
    assert initial.segment.parameters.flow_targets.z_fast[0] == 0.75


def test_nonzero_flow_preserves_differentiable_controller_targets() -> None:
    leaf = torch.full((256,), 0.5, requires_grad=True)
    param = parameters()
    param = replace(param, flow_targets=replace(param.flow_targets, z_fast=leaf))
    initial = start_segment(
        RuntimeCore(make_initial_continuous_state()),
        param,
        time=0.0,
        parent_event_id=0,
        prediction_snapshot_sha256="a" * 64,
    )
    advanced = advance_to(initial, 0.25)
    advanced.core.continuous.z_fast.sum().backward()
    expected = -math.expm1(-float(param.flow_rates.z_fast[0]) * 0.25)
    torch.testing.assert_close(leaf.grad, torch.full_like(leaf, expected))


def test_guard_accumulators_can_pass_threshold_without_clipping() -> None:
    advanced = advance_to(runtime(), 1000.0)
    torch.testing.assert_close(
        advanced.core.continuous.guard_accumulators, torch.tensor([0.5, 1.25, 1.75])
    )


def test_start_segment_requires_the_causal_jump_to_have_reset_accumulators() -> None:
    advanced = advance_to(runtime(), 0.25)
    with pytest.raises(DynamicsError, match="reset"):
        start_segment(
            advanced.core,
            parameters(),
            time=0.25,
            parent_event_id=1,
            prediction_snapshot_sha256="b" * 64,
        )


def test_unanchored_split_flow_agrees_with_float32_semigroup_tolerance() -> None:
    initial = runtime()
    midpoint = advance_to(initial, 0.3)
    reanchored = RuntimeState(
        midpoint.core,
        AnalyticSegment(0.3, midpoint.core.continuous, initial.segment.parameters, 0, "a" * 64),
        0.3,
    )
    direct = advance_to(initial, 1.1)
    split = advance_to(reanchored, 1.1)
    for f in fields(ContinuousState):
        torch.testing.assert_close(
            getattr(direct.core.continuous, f.name),
            getattr(split.core.continuous, f.name),
            rtol=5e-7,
            atol=2e-7,
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_flow_uses_float32_on_its_declared_device() -> None:
    initial = runtime(device="mps")
    actual = advance_to(initial, 0.25)
    expected = advance_to(runtime(), 0.25)
    for f in fields(ContinuousState):
        value = getattr(actual.core.continuous, f.name)
        assert value.dtype is torch.float32
        assert value.device.type == "mps"
        torch.testing.assert_close(
            value.cpu(), getattr(expected.core.continuous, f.name), rtol=1e-6, atol=2e-7
        )
