"""Detached zero-time cognitive intervention for the Phase 5A diagnostic."""

import math
from dataclasses import asdict, dataclass, fields, replace

import torch

from silent_cascade.eval.compute import (
    NeuralComputeMeter,
    RuntimeCompute,
    aggregate_runtime_compute,
    parameter_counts,
)
from silent_cascade.eventflow.guards import allowed_mode_mask
from silent_cascade.eventflow.invariants import validate_post_jump, validate_runtime_state
from silent_cascade.eventflow.neural import NeuralEventFlowAgent
from silent_cascade.eventflow.state import ComputeCounters, RuntimeState
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.trace import SegmentSummary, tensor_sha256
from silent_cascade.schemas import InternalEvent, InternalEventKind, Mode


@dataclass(frozen=True, slots=True)
class CompressedStep:
    event_id: int
    parent_event_id: int
    kind: InternalEventKind
    executed_at: float
    predicted_crossing_at: float
    predicted_delta: float
    selected_record_id: int | None
    bypassed_refractory_until: float | None
    state_sha256: str


@dataclass(frozen=True, slots=True)
class CompressedDecision:
    final_state: RuntimeState
    predicted_act: InternalEvent | None
    steps: tuple[CompressedStep, ...]
    compute: RuntimeCompute
    stop_reason: str


class CompressedExecutionError(Exception):
    """Retain committed steps and measured work when an intervention fails."""

    def __init__(self, partial: CompressedDecision) -> None:
        super().__init__("compressed intervention failed after partial execution")
        self.partial = partial


def _counter_delta(after: ComputeCounters, before: ComputeCounters) -> ComputeCounters:
    return ComputeCounters(
        **{
            item.name: getattr(after, item.name) - getattr(before, item.name)
            for item in fields(ComputeCounters)
        }
    )


def compressed_state_sha256(state: RuntimeState) -> str:
    """Replay identity of one public-derived intervention state."""
    core = state.core
    return sha256_bytes(
        canonical_json_bytes(
            {
                "time": state.time,
                "mode": core.mode.value,
                "last_event_id": core.last_event_id,
                "executed_internal_events": core.executed_internal_events,
                "same_kind_refractory_until": core.same_kind_refractory_until,
                "active_record_id": core.active_record_id,
                "focus_node_id": core.focus_node_id,
                "support_ids": core.support_ids,
                "hypothesis": None if core.hypothesis is None else asdict(core.hypothesis),
                "memory": [asdict(slot) for slot in core.memory.records],
                "counters": asdict(core.counters),
                "segment": asdict(SegmentSummary.from_segment(state.segment)),
                "continuous": {
                    item.name: tensor_sha256(getattr(core.continuous, item.name))
                    for item in fields(core.continuous)
                },
            }
        )
    )


def _released_view(state: RuntimeState) -> tuple[RuntimeState, float | None]:
    mask = allowed_mode_mask(state.core.mode)
    if not any(mask):
        return state, None
    index = mask.index(True)
    if index == 2:
        return state, None
    original_release = state.core.same_kind_refractory_until[index]
    if original_release <= state.time:
        return state, None
    releases = tuple(
        state.time if item == index else value
        for item, value in enumerate(state.core.same_kind_refractory_until)
    )
    view = replace(state, core=replace(state.core, same_kind_refractory_until=releases))
    validate_runtime_state(view)
    return view, original_release


def run_compressed_from_activation(
    agent: NeuralEventFlowAgent, state: RuntimeState, *, transition_cap: int = 24
) -> CompressedDecision:
    """Recompute each learned guard/jump at ACTIVATE without advancing world time."""
    if not isinstance(agent, NeuralEventFlowAgent) or not isinstance(state, RuntimeState):
        raise TypeError("compressed execution requires a neural agent and public runtime state")
    if type(transition_cap) is not int or not 0 <= transition_cap <= 24:
        raise ValueError("compressed transition cap must be an integer in [0, 24]")
    validate_runtime_state(state)
    if state.core.activation_time is None or state.time != state.core.activation_time:
        raise ValueError("compressed execution requires a post-activation state at activation time")
    if state.core.mode not in {
        Mode.SEARCHING,
        Mode.HAVE_MEMORY,
        Mode.HOLDING_HAZARD,
        Mode.QUIESCENT,
    }:
        raise ValueError("compressed execution requires an activated nonterminal mode")
    current = state
    steps: list[CompressedStep] = []
    snapshots = []
    predicted_act = None
    stop_reason = "dormant"
    try:
        for _ in range(transition_cap + 1):
            view, bypass = _released_view(current)
            prediction = agent.next_internal_event(view)
            if prediction is None:
                break
            if not isinstance(prediction, InternalEvent):
                raise TypeError("learned guard must return a typed InternalEvent")
            if not math.isfinite(prediction.timestamp) or prediction.timestamp < current.time:
                raise ValueError("compressed learned guard predicted an invalid time")
            if prediction.kind is InternalEventKind.ACT:
                predicted_act = prediction
                stop_reason = "act_scheduled"
                break
            if prediction.kind not in {InternalEventKind.RECALL, InternalEventKind.COMPOSE}:
                raise ValueError("compressed learned guard predicted an unsupported kind")
            if len(steps) == transition_cap:
                stop_reason = "cap_reached"
                break
            event = InternalEvent(
                prediction.event_id,
                prediction.parent_event_id,
                current.time,
                prediction.kind,
                prediction.guard_index,
                0.0,
            )
            meter = NeuralComputeMeter(agent.model)
            try:
                with meter, torch.no_grad():
                    after, emitted = agent.on_internal(view, event)
            finally:
                snapshots.append(meter.snapshot())
            if emitted:
                raise ValueError("compressed cognitive jumps cannot emit an action")
            validate_post_jump(view, event, after)
            if after.time != state.time:
                raise ValueError("compressed cognition advanced world time")
            selected = (
                after.core.active_record_id
                if prediction.kind is InternalEventKind.RECALL
                else view.core.active_record_id
            )
            steps.append(
                CompressedStep(
                    event_id=event.event_id,
                    parent_event_id=event.parent_event_id,
                    kind=event.kind,
                    executed_at=event.timestamp,
                    predicted_crossing_at=prediction.timestamp,
                    predicted_delta=prediction.predicted_delta,
                    selected_record_id=selected,
                    bypassed_refractory_until=bypass,
                    state_sha256=compressed_state_sha256(after),
                )
            )
            current = after
    except Exception as caught:
        if not steps and not snapshots:
            raise
        raise CompressedExecutionError(
            _decision(agent, state, current, steps, snapshots, None, "dynamics_error")
        ) from caught
    return _decision(agent, state, current, steps, snapshots, predicted_act, stop_reason)


def _decision(
    agent: NeuralEventFlowAgent,
    initial: RuntimeState,
    current: RuntimeState,
    steps: list[CompressedStep],
    snapshots: list,
    predicted_act: InternalEvent | None,
    stop_reason: str,
) -> CompressedDecision:
    counters = _counter_delta(current.core.counters, initial.core.counters)
    total = parameter_counts(agent.model)
    compute = aggregate_runtime_compute(
        counters,
        tuple(snapshots),
        entity_parameters=total["entity_table"],
    ).model_copy(update={"parameters": total["total"]})
    return CompressedDecision(current, predicted_act, tuple(steps), compute, stop_reason)
