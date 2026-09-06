"""Shared private action-window timing derived from the active OFD contract."""

import math
from dataclasses import dataclass

from silent_cascade.env.config import OracleTimingConfig
from silent_cascade.errors import OracleError


@dataclass(frozen=True, slots=True)
class ActionWindow:
    """A target action interval whose right boundary is excluded."""

    start: float
    end: float
    target: float

    def contains(self, timestamp: float) -> bool:
        """Return whether ``timestamp`` lies in this half-open interval."""
        return self.start <= timestamp < self.end


@dataclass(frozen=True, slots=True)
class TraceTimingSchedule:
    """RNG-free non-action timing and terminal action target for one trace."""

    non_action_deltas: tuple[float, ...]
    terminal_compose_time: float
    action_target_time: float


def _require_exact_finite_float(value: object, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value):
        raise OracleError("oracle trace is temporally infeasible")
    return value


def _require_finite(value: float) -> float:
    if not math.isfinite(value):
        raise OracleError("oracle trace is temporally infeasible")
    return value


def _finite_sum(values: tuple[float, ...] | list[float]) -> float:
    try:
        return _require_finite(math.fsum(values))
    except (OverflowError, ValueError) as error:
        raise OracleError("oracle trace is temporally infeasible") from error


def build_trace_timing_schedule(
    *,
    activation_time: float,
    delay: float,
    competitive_counts: tuple[int, ...],
    jitter_normals: tuple[float, ...],
    timing: OracleTimingConfig,
) -> TraceTimingSchedule:
    """Compute the exact supervised trace schedule from pre-sampled jitter."""

    activation_time = _require_exact_finite_float(activation_time, "activation_time")
    delay = _require_exact_finite_float(delay, "delay")
    if delay <= 0.0:
        raise OracleError("oracle trace is temporally infeasible")
    if not isinstance(timing, OracleTimingConfig):
        raise TypeError("timing must be an OracleTimingConfig")
    if type(competitive_counts) is not tuple or type(jitter_normals) is not tuple:
        raise TypeError("competitive_counts and jitter_normals must be tuples")
    if not competitive_counts or len(competitive_counts) != len(jitter_normals):
        raise OracleError("oracle trace is temporally infeasible")
    for count in competitive_counts:
        if type(count) is not int:
            raise TypeError("competitive counts must be exact integers")
        if count < 0:
            raise OracleError("oracle trace is temporally infeasible")
    for normal in jitter_normals:
        _require_exact_finite_float(normal, "jitter normal")

    timing_values = (
        timing.delta_0,
        timing.delta_min,
        timing.delta_max,
        timing.jitter_log_std,
        timing.terminal_compose_fraction,
        timing.action_window_start_fraction,
        timing.action_target_fraction,
        timing.action_window_end_fraction,
    )
    for value in timing_values:
        _require_exact_finite_float(value, "timing scalar")
    if (
        timing.delta_min <= 0.0
        or not timing.delta_min <= timing.delta_0 <= timing.delta_max
        or timing.jitter_log_std < 0.0
        or not 0.0 < timing.terminal_compose_fraction < 1.0
        or not timing.terminal_compose_fraction < timing.action_window_start_fraction
        or not (
            timing.action_window_start_fraction
            < timing.action_target_fraction
            < timing.action_window_end_fraction
            <= 1.0
        )
    ):
        raise OracleError("oracle trace is temporally infeasible")

    compose_budget = _require_finite(timing.terminal_compose_fraction * delay)
    compose_deadline = _finite_sum((activation_time, compose_budget))
    provisional: list[float] = []
    total_count = len(competitive_counts)
    for index, (competitive_count, jitter_normal) in enumerate(
        zip(competitive_counts, jitter_normals, strict=True)
    ):
        elapsed = _finite_sum(provisional)
        current_time = _finite_sum((activation_time, elapsed))
        remaining_budget = _require_finite(compose_deadline - current_time)
        remaining_event_count = total_count - index
        urgency_numerator = _require_finite(remaining_event_count * timing.delta_0)
        urgency_denominator = max(remaining_budget, timing.delta_min)
        urgency = _require_finite(min(1.0, urgency_numerator / urgency_denominator))
        try:
            difficulty = _require_finite(1.0 + 0.15 * competitive_count)
        except OverflowError as error:
            raise OracleError("oracle trace is temporally infeasible") from error
        urgency_scale = _require_finite(1.0 + 0.5 * urgency)
        raw = _require_finite(timing.delta_0 * difficulty / urgency_scale)
        exponent = _require_finite(jitter_normal * timing.jitter_log_std)
        try:
            jitter_factor = math.exp(exponent)
        except OverflowError as error:
            raise OracleError("oracle trace is temporally infeasible") from error
        jittered = _require_finite(raw * jitter_factor)
        delta = _require_finite(min(timing.delta_max, max(timing.delta_min, jittered)))
        provisional.append(delta)

    total = _finite_sum(provisional)
    final_deltas: tuple[float, ...]
    if total > compose_budget:
        scale = _require_finite(compose_budget / total)
        final_deltas = tuple(_require_finite(delta * scale) for delta in provisional)
    else:
        final_deltas = tuple(provisional)
    if any(delta < timing.delta_min for delta in final_deltas):
        raise OracleError("oracle trace is temporally infeasible")

    terminal_compose_time = _finite_sum((activation_time, *final_deltas))
    action_target_time = _finite_sum((activation_time, timing.action_target_fraction * delay))
    if terminal_compose_time > compose_deadline or action_target_time <= terminal_compose_time:
        raise OracleError("oracle trace is temporally infeasible")
    return TraceTimingSchedule(final_deltas, terminal_compose_time, action_target_time)


def action_window(
    activation_time: float,
    delay: float,
    timing: OracleTimingConfig,
) -> ActionWindow:
    """Derive the configured action window from public activation and delay."""
    if not isinstance(timing, OracleTimingConfig):
        raise TypeError("timing must be an OracleTimingConfig")
    return ActionWindow(
        start=activation_time + timing.action_window_start_fraction * delay,
        end=activation_time + timing.action_window_end_fraction * delay,
        target=activation_time + timing.action_target_fraction * delay,
    )
