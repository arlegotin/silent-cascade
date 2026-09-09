"""Public-facts rule system for non-neural runtime engineering evidence.

Only public memory, focus and executed event ordinal determine retrieval and
flow parameters. Cognitive gaps in seconds are 0.050 + 0.001 * ((7 * record_id
+ 11 * focus_node_id + 3 * executed_internal_events) % 19). Their irregularity
exercises autonomous guard crossings; latent features make no intelligence claim.

For compressed clocks only, multiply these gaps by min(1, smallest publicly
observed hazard delay in seconds). This urgency rule carries no reachability
information. Unscaled episodes keep gaps in [0.05, 0.068]. Impossible rates or
already-past action targets fail loudly. No hidden path or external horizon is
retained. Shared jumps own transitions; the engine owns prediction/flow counts.
"""

import math
from dataclasses import fields, replace

import torch

from silent_cascade.errors import DynamicsError
from silent_cascade.eventflow.flow import start_segment
from silent_cascade.eventflow.guards import (
    allowed_mode_mask,
    next_crossings,
    prediction_snapshot_sha256,
)
from silent_cascade.eventflow.jumps import (
    ComposeDecision,
    ComposeRole,
    apply_act,
    apply_activate,
    apply_compose,
    apply_fact,
    apply_recall,
    begin_post_jump_segment,
)
from silent_cascade.eventflow.state import (
    ComputeCounters,
    ContinuousChannels,
    RuntimeCore,
    RuntimeState,
    SegmentParameters,
    make_initial_continuous_state,
)
from silent_cascade.memory import BoundedMemory, RuntimeMemoryRecord
from silent_cascade.schemas import (
    Action,
    AgentInit,
    Condition,
    ExternalEvent,
    ExternalEventKind,
    InternalEvent,
    InternalEventKind,
    Mode,
    RecordKind,
)


class ScriptedEventFlowAgent:
    """Deterministic public agent with float32 channels on CPU or MPS.

    ``compute_counters`` snapshots the latest public callback state; the engine's
    final result owns full counters, including the private terminal transition.
    """

    name = Condition.EVENT_FLOW

    def __init__(self, *, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self._counters = ComputeCounters()

    @staticmethod
    def _select(
        core: RuntimeCore, time: float
    ) -> tuple[RuntimeCore, RuntimeMemoryRecord | None, int | None]:
        if core.focus_node_id is None:
            raise DynamicsError("search requires a public focus node")
        candidates = core.memory.legal_records(subject_id=core.focus_node_id, at_time=time)
        core = replace(
            core,
            counters=replace(
                core.counters, records_scored=core.counters.records_scored + len(candidates)
            ),
        )
        if not candidates:
            return core, None, None

        def key(slot: RuntimeMemoryRecord) -> tuple[float, int]:
            return -slot.record.confidence, slot.record.record_id

        selected = min(candidates, key=key)
        rank = next(
            index
            for index, slot in enumerate(sorted(candidates, key=key), start=1)
            if slot.record.record_id == selected.record.record_id
        )
        return core, selected, rank

    @staticmethod
    def _cognitive_gap(core: RuntimeCore, record_id: int) -> float:
        delays = [
            slot.record.delay
            for slot in core.memory.records
            if slot.valid and slot.record.kind is RecordKind.HAZARD
        ]
        scale = min(1.0, min(delays)) if delays else 1.0
        ordinal = 7 * record_id + 11 * core.focus_node_id + 3 * core.executed_internal_events
        return (0.050 + 0.001 * (ordinal % 19)) * scale

    def _controller(self, core: RuntimeCore, time: float) -> tuple[RuntimeCore, SegmentParameters]:
        core = replace(
            core,
            counters=replace(core.counters, controller_calls=core.counters.controller_calls + 1),
        )
        delta = None
        guard_index = None
        if core.mode is Mode.SEARCHING:
            core, selected, _ = self._select(core, time)
            if selected is None:
                core = replace(core, mode=Mode.QUIESCENT)
            else:
                delta = self._cognitive_gap(core, selected.record.record_id)
                guard_index = 0
        elif core.mode is Mode.HAVE_MEMORY:
            if core.active_record_id is None or core.focus_node_id is None:
                raise DynamicsError("composition requires an active public record and focus")
            delta = self._cognitive_gap(core, core.active_record_id)
            guard_index = 1
        elif core.mode is Mode.HOLDING_HAZARD:
            hypothesis = core.hypothesis
            if hypothesis is None or hypothesis.deadline is None or core.activation_time is None:
                raise DynamicsError("hazard holding requires a public deadline and activation")
            delay = hypothesis.deadline - core.activation_time
            target = core.activation_time + 0.825 * delay
            if not math.isfinite(target) or target <= time:
                raise DynamicsError("action target must be strictly after current time")
            delta = target - time
            guard_index = 2

        reference = core.continuous.guard_accumulators
        guard_targets = torch.full_like(reference, 0.5)
        guard_rates = torch.ones_like(reference)
        if guard_index is not None:
            if delta <= 0.0:
                raise DynamicsError("scripted guard rate requires a positive gap")
            rate = math.log(3.0) / delta
            if not math.isfinite(rate) or not 1e-5 <= rate <= 500.0:
                raise DynamicsError("scripted guard rate is outside legal bounds")
            guard_targets[guard_index] = 1.5
            guard_rates[guard_index] = rate

        # Fixed bounded features; no trainable encoder or random/hash state.
        public_code = (
            (core.focus_node_id or 0) + 3 * len(core.memory.records) + core.executed_internal_events
        )
        targets, rates = {}, {}
        for index, item in enumerate(fields(ContinuousChannels)):
            channel = getattr(core.continuous, item.name)
            coordinate = torch.arange(channel.numel(), device=channel.device, dtype=torch.float32)
            targets[item.name] = ((coordinate + public_code + index) % 17 - 8) / 16
            rates[item.name] = torch.full_like(channel, 0.25 + 0.25 * ((public_code + index) % 7))
        return core, SegmentParameters(
            ContinuousChannels(**targets), ContinuousChannels(**rates), guard_targets, guard_rates
        )

    def _install(self, core: RuntimeCore, time: float) -> RuntimeState:
        core, parameters = self._controller(core, time)
        state = begin_post_jump_segment(core, parameters, time=time)
        self._counters = state.core.counters
        return state

    def initialize(self, init: AgentInit) -> RuntimeState:
        core = RuntimeCore(
            make_initial_continuous_state(device=self.device),
            memory=BoundedMemory(init.memory_capacity),
        )
        core, parameters = self._controller(core, init.initial_time)
        digest = prediction_snapshot_sha256(
            core.continuous,
            parameters,
            started_at=init.initial_time,
            allowed_mode_mask=allowed_mode_mask(core.mode),
            parent_event_id=0,
        )
        state = start_segment(
            core,
            parameters,
            time=init.initial_time,
            parent_event_id=0,
            prediction_snapshot_sha256=digest,
        )
        self._counters = core.counters
        return state

    def on_external(self, state: RuntimeState, event: ExternalEvent) -> RuntimeState:
        if event.kind is ExternalEventKind.FACT:
            core = apply_fact(state, event)
        elif event.kind is ExternalEventKind.ACTIVATE:
            core = apply_activate(state, event)
        else:
            raise DynamicsError("scripted agent accepts only public FACT and ACTIVATE events")
        return self._install(core, state.time)

    def next_internal_event(self, state: RuntimeState) -> InternalEvent | None:
        self._counters = state.core.counters
        candidates = next_crossings(state)
        if not candidates:
            return None
        crossing = candidates[0]
        return InternalEvent(
            crossing.event_id,
            crossing.parent_event_id,
            crossing.timestamp,
            crossing.kind,
            crossing.guard_index,
            crossing.predicted_delta,
        )

    def on_internal(
        self, state: RuntimeState, event: InternalEvent
    ) -> tuple[RuntimeState, list[Action]]:
        if event.kind is InternalEventKind.RECALL:
            core, selected, rank = self._select(state.core, state.time)
            if selected is None:
                raise DynamicsError("scheduled recall has no legal public candidate")
            core = apply_recall(
                replace(state, core=core),
                event,
                record_id=selected.record.record_id,
                selected_rank=rank,
            )
        elif event.kind is InternalEventKind.COMPOSE:
            record_id = state.core.active_record_id
            if record_id is None:
                raise DynamicsError("composition requires an active public record")
            record = state.core.memory.lookup(record_id).record
            if record.kind is RecordKind.LINK:
                decision = ComposeDecision(
                    ComposeRole.LINK,
                    next_focus_node_id=record.object_id,
                    confidence=record.confidence,
                )
            elif record.kind is RecordKind.HAZARD:
                if state.core.activation_time is None:
                    raise DynamicsError("hazard composition requires activation time")
                decision = ComposeDecision(
                    ComposeRole.HAZARD,
                    hazard_type=record.hazard_type,
                    deadline=state.core.activation_time + record.delay,
                    confidence=record.confidence,
                )
            else:
                decision = ComposeDecision(ComposeRole.SAFE, confidence=record.confidence)
            core = apply_compose(state, event, decision)
        elif event.kind is InternalEventKind.ACT:
            hypothesis = state.core.hypothesis
            if hypothesis is None or hypothesis.hazard_type is None:
                raise DynamicsError("action requires a public hazard hypothesis")
            core = apply_act(state, event, hazard_type=hypothesis.hazard_type)
        else:
            raise DynamicsError("scripted endogenous events are RECALL, COMPOSE and ACT")
        return self._install(core, state.time), list(core.actions[len(state.core.actions) :])

    def compute_counters(self) -> ComputeCounters:
        return self._counters
