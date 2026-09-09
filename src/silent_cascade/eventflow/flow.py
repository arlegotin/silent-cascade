"""Pure analytic flow from fixed segment origins; no controller or event dispatch."""

from dataclasses import fields, replace

import torch

from silent_cascade.errors import DynamicsError, TimeOrderError
from silent_cascade.eventflow.state import (
    AnalyticSegment,
    ContinuousChannels,
    ContinuousState,
    RuntimeCore,
    RuntimeState,
    SegmentParameters,
    require_time,
)


def _flow_tensor(
    origin: torch.Tensor, target: torch.Tensor, rate: torch.Tensor, dt: float
) -> torch.Tensor:
    dt_tensor = origin.new_tensor(dt)
    weight = -torch.expm1(-rate * dt_tensor)
    return origin + weight * (target - origin)


def start_segment(
    core: RuntimeCore,
    parameters: SegmentParameters,
    *,
    time: float,
    parent_event_id: int,
    prediction_snapshot_sha256: str,
) -> RuntimeState:
    """Install supplied controller outputs after a causal jump has reset all guards."""
    require_time(time, "time")
    if bool(torch.count_nonzero(core.continuous.guard_accumulators)):
        raise DynamicsError("starting a segment requires reset guard accumulators")
    segment = AnalyticSegment(
        time, core.continuous, parameters, parent_event_id, prediction_snapshot_sha256
    )
    return RuntimeState(core, segment, time)


def state_at(runtime: RuntimeState, target_time: float) -> ContinuousState:
    """Return an independent snapshot within the current analytic segment.

    Historical points at or after the segment origin are valid. This pure query
    neither changes the materialized cursor nor increments compute counters.
    Tensor clones retain autograd graphs.
    """
    require_time(target_time, "target_time")
    elapsed = target_time - runtime.segment.started_at
    if elapsed < 0.0:
        raise TimeOrderError("snapshot time cannot precede the segment origin")
    origin = runtime.segment.origin
    if elapsed == 0.0:
        return replace(origin)
    parameters = runtime.segment.parameters
    return ContinuousState(
        **{
            item.name: _flow_tensor(
                getattr(origin, item.name),
                getattr(parameters.flow_targets, item.name),
                getattr(parameters.flow_rates, item.name),
                elapsed,
            )
            for item in fields(ContinuousChannels)
        },
        guard_accumulators=_flow_tensor(
            origin.guard_accumulators, parameters.guard_targets, parameters.guard_rates, elapsed
        ),
    )


def advance_to(runtime: RuntimeState, target_time: float) -> RuntimeState:
    """Materialize at an absolute host timestamp, retaining the original anchor.

    A zero-duration advance is the same object and costs zero flow evaluations.
    This runtime wrapper is separate from the ContinuousState/dt flow protocol.
    """
    require_time(target_time, "target_time")
    if target_time < runtime.time:
        raise TimeOrderError("advance time cannot precede the current runtime time")
    if target_time == runtime.time:
        return runtime
    continuous = state_at(runtime, target_time)
    counters = replace(
        runtime.core.counters, flow_evaluations=runtime.core.counters.flow_evaluations + 1
    )
    return RuntimeState(
        replace(runtime.core, continuous=continuous, counters=counters),
        runtime.segment,
        target_time,
    )
