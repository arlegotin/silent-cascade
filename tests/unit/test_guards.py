"""Independent scalar references and runtime race checks for endogenous guards."""

from __future__ import annotations

import math
from dataclasses import replace
from decimal import Decimal, getcontext
from unittest.mock import patch

import pytest
import torch

from silent_cascade.errors import DynamicsError, TimeOrderError
from silent_cascade.eventflow.flow import advance_to, start_segment
from silent_cascade.eventflow.guards import (
    GUARD_KIND_BY_INDEX,
    allowed_mode_mask,
    crossing_offset_host,
    crossing_offsets_tensor,
    next_crossings,
    prediction_snapshot_sha256,
)
from silent_cascade.eventflow.state import (
    INTERNAL_EVENT_ID_BASE,
    ContinuousChannels,
    RuntimeCore,
    RuntimeState,
    SegmentParameters,
    make_initial_continuous_state,
)
from silent_cascade.schemas import InternalEventKind, Mode


def parameters(
    *,
    targets: tuple[float, float, float] = (1.5, 1.5, 1.5),
    rates: tuple[float, float, float] = (2.0, 3.0, 4.0),
    device: str = "cpu",
) -> SegmentParameters:
    dimensions = {
        "z_fast": 256,
        "z_slow": 64,
        "drives": 8,
        "focus_key": 64,
        "hypothesis_latent": 64,
    }
    channels = ContinuousChannels(
        **{
            name: torch.zeros(size, dtype=torch.float32, device=device)
            for name, size in dimensions.items()
        }
    )
    rates_channels = ContinuousChannels(
        **{
            name: torch.full((size,), 1.0, dtype=torch.float32, device=device)
            for name, size in dimensions.items()
        }
    )
    return SegmentParameters(
        channels,
        rates_channels,
        torch.tensor(targets, dtype=torch.float32, device=device),
        torch.tensor(rates, dtype=torch.float32, device=device),
    )


def runtime(
    *,
    mode: Mode = Mode.SEARCHING,
    started_at: float = 10.0,
    targets: tuple[float, float, float] = (1.5, 1.5, 1.5),
    rates: tuple[float, float, float] = (2.0, 3.0, 4.0),
    refractory: tuple[float, float, float] = (0.0, 0.0, 0.0),
    executed: int = 0,
    parent_id: int = 17,
    device: str = "cpu",
) -> RuntimeState:
    core = RuntimeCore(
        make_initial_continuous_state(device=device),
        mode=mode,
        same_kind_refractory_until=refractory,
        executed_internal_events=executed,
    )
    params = parameters(targets=targets, rates=rates, device=device)
    digest = prediction_snapshot_sha256(
        core.continuous,
        params,
        started_at=started_at,
        allowed_mode_mask=allowed_mode_mask(mode),
        parent_event_id=parent_id,
    )
    return start_segment(
        core,
        params,
        time=started_at,
        parent_event_id=parent_id,
        prediction_snapshot_sha256=digest,
    )


def decimal_reference(a: float, asymptote: float, rate: float) -> float:
    getcontext().prec = 80
    one = Decimal(1)
    return float(
        ((one + (one - Decimal.from_float(a)) / (Decimal.from_float(asymptote) - one)).ln())
        / Decimal.from_float(rate)
    )


@pytest.mark.parametrize(
    ("a", "asymptote", "rate"),
    [
        (0.0, 1.1, 1.0e-5),
        (0.25, 1.75, 500.0),
        (0.999999, 1.0001, 17.0),
    ],
)
def test_host_crossing_matches_standalone_high_precision_reference(
    a: float, asymptote: float, rate: float
) -> None:
    expected = math.log1p((1.0 - a) / (asymptote - 1.0)) / rate
    assert crossing_offset_host(a, asymptote, rate) == expected
    assert crossing_offset_host(a, asymptote, rate) == pytest.approx(
        decimal_reference(a, asymptote, rate), rel=4e-14, abs=0.0
    )


@pytest.mark.parametrize("a, asymptote", [(0.0, 0.9), (0.0, 1.0), (1.0, 1.5), (1.5, 1.5)])
def test_host_crossing_returns_none_without_evaluating_dormant_logarithm(
    a: float, asymptote: float
) -> None:
    with patch("silent_cascade.eventflow.guards.math.log1p", side_effect=AssertionError):
        assert crossing_offset_host(a, asymptote, 1.0) is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_host_crossing_rejects_nonfinite_inputs(value: float) -> None:
    with pytest.raises(DynamicsError):
        crossing_offset_host(value, 1.5, 1.0)
    with pytest.raises(DynamicsError):
        crossing_offset_host(0.0, value, 1.0)
    with pytest.raises(DynamicsError):
        crossing_offset_host(0.0, 1.5, value)


def test_host_crossing_accepts_only_the_float32_representable_lower_rate_bound() -> None:
    legal = float(torch.tensor(1.0e-5, dtype=torch.float32).item())
    assert crossing_offset_host(0.0, 1.5, legal) is not None
    with pytest.raises(DynamicsError):
        crossing_offset_host(0.0, 1.5, math.nextafter(legal, -math.inf))


@pytest.mark.parametrize(
    ("accumulator", "asymptote", "rate"),
    [
        (-0.1, 1.5, 1.0),
        (2.1, 1.5, 1.0),
        (0.0, 0.0, 1.0),
        (0.0, -0.1, 1.0),
        (0.0, 2.0, 1.0),
        (0.0, 2.1, 1.0),
        (
            0.0,
            1.5,
            math.nextafter(float(torch.tensor(1.0e-5, dtype=torch.float32).item()), -math.inf),
        ),
        (0.0, 1.5, 500.000001),
    ],
)
def test_host_crossing_rejects_finite_values_outside_the_guard_domain(
    accumulator: float, asymptote: float, rate: float
) -> None:
    with pytest.raises(DynamicsError):
        crossing_offset_host(accumulator, asymptote, rate)


def test_tensor_crossing_masks_dormant_values_without_invalid_math_and_preserves_gradients() -> (
    None
):
    accumulators = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    targets = torch.tensor([1.5, 1.5, 0.9], dtype=torch.float32, requires_grad=True)
    rates = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32, requires_grad=True)
    result = crossing_offsets_tensor(accumulators, targets, rates)
    assert result.dtype is torch.float32
    assert result.device.type == "cpu"
    assert torch.isfinite(result[0])
    assert torch.isinf(result[1:]).all()
    result[0].backward()
    assert torch.isfinite(targets.grad[0])
    assert torch.isfinite(rates.grad[0])
    assert targets.grad[1] == 0 and targets.grad[2] == 0
    assert rates.grad[1] == 0 and rates.grad[2] == 0


@pytest.mark.parametrize(
    ("accumulators", "asymptotes", "rates"),
    [
        (
            torch.zeros(3, dtype=torch.float64),
            torch.full((3,), 1.5, dtype=torch.float32),
            torch.ones(3, dtype=torch.float32),
        ),
        (
            torch.zeros(3, dtype=torch.float32),
            torch.full((3,), 1.5, dtype=torch.float64),
            torch.ones(3, dtype=torch.float32),
        ),
        (
            torch.zeros(3, dtype=torch.float32),
            torch.full((3,), 1.5, dtype=torch.float32),
            torch.ones(3, dtype=torch.float64),
        ),
        (
            torch.zeros(2, dtype=torch.float32),
            torch.full((2,), 1.5, dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
        ),
        (
            torch.tensor([0.0, float("nan"), 0.0], dtype=torch.float32),
            torch.full((3,), 1.5, dtype=torch.float32),
            torch.ones(3, dtype=torch.float32),
        ),
        (
            torch.zeros(3, dtype=torch.float32),
            torch.tensor([1.5, float("nan"), 1.5], dtype=torch.float32),
            torch.ones(3, dtype=torch.float32),
        ),
        (
            torch.zeros(3, dtype=torch.float32),
            torch.full((3,), 1.5, dtype=torch.float32),
            torch.tensor([1.0, float("inf"), 1.0], dtype=torch.float32),
        ),
        (
            torch.tensor([-0.1, 0.0, 0.0], dtype=torch.float32),
            torch.full((3,), 1.5, dtype=torch.float32),
            torch.ones(3, dtype=torch.float32),
        ),
        (
            torch.tensor([2.1, 0.0, 0.0], dtype=torch.float32),
            torch.full((3,), 1.5, dtype=torch.float32),
            torch.ones(3, dtype=torch.float32),
        ),
        (
            torch.zeros(3, dtype=torch.float32),
            torch.tensor([0.0, 1.5, 1.5], dtype=torch.float32),
            torch.ones(3, dtype=torch.float32),
        ),
        (
            torch.zeros(3, dtype=torch.float32),
            torch.tensor([1.5, 2.0, 1.5], dtype=torch.float32),
            torch.ones(3, dtype=torch.float32),
        ),
        (
            torch.zeros(3, dtype=torch.float32),
            torch.full((3,), 1.5, dtype=torch.float32),
            torch.tensor([1.0, 500.1, 1.0], dtype=torch.float32),
        ),
        (
            torch.zeros(3, dtype=torch.float32),
            torch.full((3,), 1.5, dtype=torch.float32),
            torch.stack(
                (
                    torch.nextafter(
                        torch.tensor(1.0e-5, dtype=torch.float32),
                        torch.tensor(-math.inf, dtype=torch.float32),
                    ),
                    torch.tensor(1.0, dtype=torch.float32),
                    torch.tensor(1.0, dtype=torch.float32),
                )
            ),
        ),
    ],
)
def test_tensor_crossing_rejects_invalid_dtype_shape_finiteness_and_bounds(
    accumulators: torch.Tensor, asymptotes: torch.Tensor, rates: torch.Tensor
) -> None:
    with pytest.raises(DynamicsError):
        crossing_offsets_tensor(accumulators, asymptotes, rates)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_tensor_crossing_rejects_mixed_cpu_mps_devices() -> None:
    with pytest.raises(DynamicsError, match="share one device"):
        crossing_offsets_tensor(
            torch.zeros(3, dtype=torch.float32),
            torch.full((3,), 1.5, dtype=torch.float32, device="mps"),
            torch.ones(3, dtype=torch.float32, device="mps"),
        )


@pytest.mark.parametrize(
    ("mode", "expected_kind"),
    [
        (Mode.SEARCHING, InternalEventKind.RECALL),
        (Mode.HAVE_MEMORY, InternalEventKind.COMPOSE),
        (Mode.HOLDING_HAZARD, InternalEventKind.ACT),
    ],
)
def test_next_crossings_allows_exactly_the_mode_guard(
    mode: Mode, expected_kind: InternalEventKind
) -> None:
    crossings = next_crossings(runtime(mode=mode))
    assert [crossing.kind for crossing in crossings] == [expected_kind]
    assert crossings[0].guard_index == GUARD_KIND_BY_INDEX.index(expected_kind)


@pytest.mark.parametrize("mode", [Mode.OBSERVING, Mode.QUIESCENT, Mode.TERMINAL])
def test_next_crossings_is_dormant_in_non_eventflow_modes(mode: Mode) -> None:
    assert next_crossings(runtime(mode=mode)) == []


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (Mode.OBSERVING, (False, False, False)),
        (Mode.SEARCHING, (True, False, False)),
        (Mode.HAVE_MEMORY, (False, True, False)),
        (Mode.HOLDING_HAZARD, (False, False, True)),
        (Mode.QUIESCENT, (False, False, False)),
        (Mode.TERMINAL, (False, False, False)),
    ],
)
def test_allowed_mode_mask_is_the_exact_shared_guard_mapping(
    mode: Mode, expected: tuple[bool, bool, bool]
) -> None:
    assert allowed_mode_mask(mode) == expected


def test_allowed_mode_mask_rejects_a_non_mode_value() -> None:
    with pytest.raises(DynamicsError):
        allowed_mode_mask("searching")  # type: ignore[arg-type]


def test_next_crossing_uses_segment_origin_not_materialized_accumulator() -> None:
    initial = runtime(mode=Mode.SEARCHING, started_at=10.0, rates=(2.0, 2.0, 2.0))
    materialized = advance_to(initial, 10.1)
    crossing = next_crossings(materialized)[0]
    expected_offset = crossing_offset_host(
        0.0, float(initial.segment.parameters.guard_targets[0]), 2.0
    )
    assert crossing.timestamp == 10.0 + expected_offset
    assert crossing.predicted_delta == crossing.timestamp - materialized.time


def test_refractory_releases_an_already_mathematically_crossed_guard() -> None:
    initial = runtime(mode=Mode.SEARCHING, refractory=(12.0, 0.0, 0.0))
    current = advance_to(initial, 11.0)
    crossing = next_crossings(current)[0]
    assert crossing.timestamp == 12.0
    assert crossing.predicted_delta == 1.0


def test_missed_active_guard_crossing_raises_instead_of_silently_disappearing() -> None:
    with pytest.raises(TimeOrderError, match="predates"):
        next_crossings(advance_to(runtime(), 11.0))


def test_refractory_release_exactly_at_runtime_time_remains_eligible() -> None:
    initial = runtime(mode=Mode.SEARCHING, refractory=(11.0, 0.0, 0.0))
    crossing = next_crossings(advance_to(initial, 11.0))[0]
    assert crossing.timestamp == 11.0
    assert crossing.predicted_delta == 0.0


def test_crossing_uses_segment_parent_snapshot_and_unconsumed_internal_id() -> None:
    state = runtime(executed=4, parent_id=91)
    crossing = next_crossings(state)[0]
    assert crossing.event_id == INTERNAL_EVENT_ID_BASE + 4
    assert crossing.parent_event_id == 91
    assert crossing.prediction_snapshot_sha256 == state.segment.prediction_snapshot_sha256
    assert crossing.kind is not InternalEventKind.NOOP


def test_snapshot_hash_frames_tensor_metadata_origin_time_mask_and_parent() -> None:
    state = runtime()
    params = state.segment.parameters
    baseline = prediction_snapshot_sha256(
        state.segment.origin,
        params,
        started_at=state.segment.started_at,
        allowed_mode_mask=(True, False, False),
        parent_event_id=state.segment.parent_event_id,
    )
    assert baseline == state.segment.prediction_snapshot_sha256
    assert baseline != prediction_snapshot_sha256(
        state.segment.origin,
        params,
        started_at=state.segment.started_at + 1.0,
        allowed_mode_mask=(True, False, False),
        parent_event_id=state.segment.parent_event_id,
    )
    assert baseline != prediction_snapshot_sha256(
        state.segment.origin,
        params,
        started_at=state.segment.started_at,
        allowed_mode_mask=(False, True, False),
        parent_event_id=state.segment.parent_event_id,
    )
    assert baseline != prediction_snapshot_sha256(
        state.segment.origin,
        params,
        started_at=state.segment.started_at,
        allowed_mode_mask=(True, False, False),
        parent_event_id=state.segment.parent_event_id + 1,
    )
    changed_origin = replace(
        state.segment.origin,
        guard_accumulators=torch.tensor([0.25, 0.0, 0.0], dtype=torch.float32),
    )
    assert baseline != prediction_snapshot_sha256(
        changed_origin,
        params,
        started_at=state.segment.started_at,
        allowed_mode_mask=(True, False, False),
        parent_event_id=state.segment.parent_event_id,
    )
    changed_parameters = replace(
        params,
        guard_targets=torch.tensor([1.6, 1.5, 1.5], dtype=torch.float32),
        guard_rates=torch.tensor([2.5, 3.0, 4.0], dtype=torch.float32),
    )
    assert baseline != prediction_snapshot_sha256(
        state.segment.origin,
        changed_parameters,
        started_at=state.segment.started_at,
        allowed_mode_mask=(True, False, False),
        parent_event_id=state.segment.parent_event_id,
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_tensor_crossings_stay_float32_on_mps() -> None:
    result = crossing_offsets_tensor(
        torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device="mps"),
        torch.tensor([1.5, 1.5, 0.9], dtype=torch.float32, device="mps"),
        torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32, device="mps"),
    )
    assert result.dtype is torch.float32
    assert result.device.type == "mps"
