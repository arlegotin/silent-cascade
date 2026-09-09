"""Pure decision-driven jumps followed by one explicit segment installation.

Callers advance the runtime to the event first, apply a jump, obtain controller
parameters from the returned core, then call ``begin_post_jump_segment`` once.
Jumps own event counts, last-event IDs, and refractory/consumption changes.
The controller caller owns controller counts; the scheduler owns gap clamps.
Private terminal handling belongs exclusively to the engine.
"""

import math
from dataclasses import dataclass, replace
from enum import StrEnum

import torch

from silent_cascade.errors import DynamicsError, ProvenanceError, TimeOrderError
from silent_cascade.eventflow.flow import start_segment
from silent_cascade.eventflow.guards import (
    GUARD_KIND_BY_INDEX,
    allowed_mode_mask,
    prediction_snapshot_sha256,
)
from silent_cascade.eventflow.state import (
    INTERNAL_EVENT_ID_BASE,
    ContinuousState,
    RuntimeCore,
    RuntimeState,
    SegmentParameters,
)
from silent_cascade.memory import (
    append_perceived_fact,
    mark_consumed,
    mark_recalled,
    require_support_ledger,
)
from silent_cascade.schemas import (
    MAX_ENTITY_ID,
    MAX_HAZARD_TYPE,
    Action,
    ActivationPayload,
    ExternalEvent,
    ExternalEventKind,
    Hypothesis,
    InternalEvent,
    InternalEventKind,
    Mode,
    Provenance,
    RecordKind,
)

_REFRACTORY = 1.0e-3
_MAX_INTERNAL_EVENTS = 64


class ComposeRole(StrEnum):
    LINK = "link"
    HAZARD = "hazard"
    SAFE = "safe"
    IRRELEVANT = "irrelevant"
    CONTRADICTORY = "contradictory"


def _require_prediction_id(value: object, name: str, maximum: int) -> None:
    if type(value) is not int or not 0 <= value <= maximum:
        raise DynamicsError(f"{name} must be an exact integer within schema bounds")


@dataclass(frozen=True, slots=True)
class ComposeDecision:
    """Agent predictions; role and answers are never inferred from record truth.

    ``append_support`` appends the active ID once to the existing ledger.
    ``continue_search`` chooses SEARCHING/QUIESCENT for nonterminal roles;
    HAZARD and SAFE always determine their own legal destination mode.
    """

    role: ComposeRole
    next_focus_node_id: int | None = None
    hazard_type: int | None = None
    deadline: float | None = None
    confidence: float = 1.0
    append_support: bool = True
    continue_search: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.role, ComposeRole):
            raise DynamicsError("composition requires a typed predicted role")
        if type(self.append_support) is not bool or type(self.continue_search) is not bool:
            raise DynamicsError("composition choices must be exact booleans")
        if type(self.confidence) is not float or not 0.0 <= self.confidence <= 1.0:
            raise DynamicsError("confidence must be a finite float in [0, 1]")
        if self.role is ComposeRole.LINK:
            _require_prediction_id(self.next_focus_node_id, "predicted focus", MAX_ENTITY_ID)
        elif self.next_focus_node_id is not None:
            raise DynamicsError("only a link decision supplies a next focus")
        if self.role is ComposeRole.HAZARD:
            _require_prediction_id(self.hazard_type, "predicted hazard", MAX_HAZARD_TYPE)
            if type(self.deadline) is not float or not math.isfinite(self.deadline):
                raise DynamicsError("predicted deadline must be a finite host float")
        elif self.hazard_type is not None or self.deadline is not None:
            raise DynamicsError("only a hazard decision supplies a hazard and deadline")


def _require_event(
    state: RuntimeState,
    event: ExternalEvent | InternalEvent,
    kind: ExternalEventKind | InternalEventKind,
    source: Mode,
) -> None:
    expected_type = ExternalEvent if isinstance(kind, ExternalEventKind) else InternalEvent
    if not isinstance(state, RuntimeState) or not isinstance(event, expected_type):
        raise DynamicsError("jump requires runtime state and the appropriate typed event")
    if event.kind is not kind or state.core.mode is not source:
        raise DynamicsError("illegal jump kind or source mode")
    if event.timestamp != state.time:
        raise TimeOrderError("jump timestamp must equal the already-advanced runtime time")
    if isinstance(event, ExternalEvent):
        if event.event_id >= INTERNAL_EVENT_ID_BASE:
            raise DynamicsError("external event ID uses the reserved internal namespace")
        if state.core.activation_time is not None:
            raise DynamicsError("FACT and ACTIVATE require a pre-activation state")
        if event.event_id == state.core.last_event_id:
            raise DynamicsError("a causal event cannot be applied twice")
        return
    if state.core.activation_time is None or state.core.activation_time > state.time:
        raise DynamicsError("endogenous events require prior activation")
    if event.event_id < INTERNAL_EVENT_ID_BASE or event.event_id == state.core.last_event_id:
        raise DynamicsError("internal event requires a fresh reserved event ID")
    if (
        event.parent_event_id != state.segment.parent_event_id
        or event.parent_event_id != state.core.last_event_id
    ):
        raise DynamicsError("internal event parent must identify the current causal segment")
    index = GUARD_KIND_BY_INDEX.index(kind)
    if event.guard_index != index or not allowed_mode_mask(state.core.mode)[index]:
        raise DynamicsError("internal event guard index or mode is illegal")
    if state.core.executed_internal_events >= _MAX_INTERNAL_EVENTS:
        raise DynamicsError("internal event limit exceeded")
    if state.time < state.core.same_kind_refractory_until[index]:
        raise DynamicsError("internal event kind is refractory")


def _impulse(reference: torch.Tensor, public_id: int, kind_code: int) -> torch.Tensor:
    """Fixed public integer arithmetic; no randomized hash or learned encoder."""
    result = torch.zeros_like(reference)
    result[(public_id * 17 + kind_code * 7) % reference.numel()] = 0.25
    result[(public_id * 13 + kind_code * 11 + 1) % reference.numel()] += 0.125
    return result


def _inject(continuous: ContinuousState, public_id: int, kind_code: int) -> ContinuousState:
    return replace(
        continuous,
        z_fast=torch.tanh(continuous.z_fast + _impulse(continuous.z_fast, public_id, kind_code)),
        z_slow=torch.tanh(
            continuous.z_slow + 0.5 * _impulse(continuous.z_slow, public_id, kind_code)
        ),
    )


def _finish(core: RuntimeCore, event: ExternalEvent | InternalEvent) -> RuntimeCore:
    refractory = core.same_kind_refractory_until
    internal_count = core.executed_internal_events
    if isinstance(event, InternalEvent):
        index = GUARD_KIND_BY_INDEX.index(event.kind)
        refractory = tuple(
            max(value, event.timestamp + _REFRACTORY) if i == index else value
            for i, value in enumerate(refractory)
        )
        internal_count += 1
    return replace(
        core,
        continuous=replace(
            core.continuous, guard_accumulators=torch.zeros_like(core.continuous.guard_accumulators)
        ),
        last_event_id=event.event_id,
        executed_internal_events=internal_count,
        same_kind_refractory_until=refractory,
        counters=replace(core.counters, jump_applications=core.counters.jump_applications + 1),
    )


def apply_fact(state: RuntimeState, event: ExternalEvent) -> RuntimeCore:
    """Store one perceived FACT and apply deterministic bounded latent impulses."""
    _require_event(state, event, ExternalEventKind.FACT, Mode.OBSERVING)
    memory = append_perceived_fact(state.core.memory, event)
    record = memory.lookup(event.event_id).record
    kind_code = (RecordKind.LINK, RecordKind.HAZARD, RecordKind.SAFE).index(record.kind)
    continuous = _inject(state.core.continuous, record.record_id + record.subject_id, kind_code)
    return _finish(replace(state.core, memory=memory, continuous=continuous), event)


def apply_activate(state: RuntimeState, event: ExternalEvent) -> RuntimeCore:
    """Initialize public focus and the activation clock exactly once."""
    _require_event(state, event, ExternalEventKind.ACTIVATE, Mode.OBSERVING)
    if not isinstance(event.payload, ActivationPayload):
        raise DynamicsError("ACTIVATE requires an activation payload")
    focus = event.payload.start_node
    continuous = _inject(state.core.continuous, focus, 3)
    continuous = replace(continuous, focus_key=torch.tanh(_impulse(continuous.focus_key, focus, 3)))
    return _finish(
        replace(
            state.core,
            mode=Mode.SEARCHING,
            continuous=continuous,
            activation_time=event.timestamp,
            focus_node_id=focus,
        ),
        event,
    )


def apply_recall(state: RuntimeState, event: InternalEvent, *, record_id: int) -> RuntimeCore:
    """Activate an explicitly selected legal record, without imposing relevance."""
    _require_event(state, event, InternalEventKind.RECALL, Mode.SEARCHING)
    slot = state.core.memory.lookup(record_id)
    if not slot.valid or slot.consumed or slot.refractory_until > state.time:
        raise DynamicsError("selected memory record is invalid, consumed, or refractory")
    memory = mark_recalled(state.core.memory, record_id, refractory_until=state.time + _REFRACTORY)
    return _finish(
        replace(
            state.core,
            mode=Mode.HAVE_MEMORY,
            memory=memory,
            active_record_id=record_id,
            continuous=_inject(state.core.continuous, record_id, 4),
        ),
        event,
    )


def apply_compose(
    state: RuntimeState, event: InternalEvent, decision: ComposeDecision
) -> RuntimeCore:
    """Consume active context and install explicit inferred predictions only."""
    _require_event(state, event, InternalEventKind.COMPOSE, Mode.HAVE_MEMORY)
    if not isinstance(decision, ComposeDecision):
        raise DynamicsError("COMPOSE requires a typed decision")
    decision = replace(decision)
    record_id = state.core.active_record_id
    if record_id is None:
        raise DynamicsError("COMPOSE requires an active record")
    slot = state.core.memory.lookup(record_id)
    if not slot.valid or slot.consumed:
        raise DynamicsError("active record is invalid or already consumed")
    supports = state.core.support_ids
    if decision.append_support and record_id not in supports:
        supports = (*supports, record_id)
    require_support_ledger(state.core.memory, supports)
    mode = Mode.SEARCHING if decision.continue_search else Mode.QUIESCENT
    focus = state.core.focus_node_id
    hypothesis = state.core.hypothesis
    continuous = _inject(state.core.continuous, record_id, 5)
    if decision.role is ComposeRole.LINK:
        focus = decision.next_focus_node_id
        continuous = replace(
            continuous, focus_key=torch.tanh(_impulse(continuous.focus_key, focus, 3))
        )
    elif decision.role in (ComposeRole.HAZARD, ComposeRole.SAFE):
        hypothesis = Hypothesis(
            decision.hazard_type,
            decision.deadline,
            decision.confidence,
            Provenance.INFERRED,
            supports,
            decision.role is ComposeRole.SAFE,
        )
        require_support_ledger(state.core.memory, hypothesis.support_ids)
        mode = Mode.QUIESCENT if hypothesis.is_safe else Mode.HOLDING_HAZARD
        predicted_class = 4 if hypothesis.is_safe else decision.hazard_type
        continuous = replace(
            continuous,
            hypothesis_latent=torch.tanh(
                _impulse(continuous.hypothesis_latent, predicted_class, 6)
            ),
        )
    else:
        continuous = replace(
            continuous,
            drives=torch.tanh(
                continuous.drives
                + (1.0 - decision.confidence) * _impulse(continuous.drives, record_id, 7)
            ),
        )
    return _finish(
        replace(
            state.core,
            mode=mode,
            continuous=continuous,
            memory=mark_consumed(state.core.memory, record_id),
            active_record_id=None,
            focus_node_id=focus,
            support_ids=supports,
            hypothesis=hypothesis,
        ),
        event,
    )


def apply_act(state: RuntimeState, event: InternalEvent, *, hazard_type: int) -> RuntimeCore:
    """Emit the explicit predicted action class, retaining its causal event ID."""
    _require_event(state, event, InternalEventKind.ACT, Mode.HOLDING_HAZARD)
    hypothesis = state.core.hypothesis
    if hypothesis is None or hypothesis.is_safe or state.core.actions:
        raise DynamicsError("ACT requires a hazard hypothesis and no previous action")
    if hypothesis.provenance is not Provenance.INFERRED:
        raise ProvenanceError("ACT requires an inferred hypothesis")
    require_support_ledger(state.core.memory, hypothesis.support_ids)
    _require_prediction_id(hazard_type, "predicted action class", MAX_HAZARD_TYPE)
    action = Action(hazard_type, event.timestamp, event.event_id)
    return _finish(
        replace(
            state.core,
            mode=Mode.QUIESCENT,
            actions=(*state.core.actions, action),
            continuous=_inject(state.core.continuous, hazard_type, 8),
        ),
        event,
    )


def begin_post_jump_segment(
    core: RuntimeCore,
    parameters: SegmentParameters,
    *,
    time: float,
) -> RuntimeState:
    """Reset guards, revalidate supplied parameters, hash and install one segment.

    ``time`` is the just-executed event's timestamp. Controller evaluation and
    its counter belong to the caller; installation does not evaluate a controller.
    """
    if not isinstance(core, RuntimeCore) or core.mode is Mode.TERMINAL:
        raise DynamicsError("post-jump installation requires a nonterminal runtime core")
    if type(core.last_event_id) is not int or core.last_event_id < 0:
        raise DynamicsError("post-jump installation requires the executed event ID")
    if not isinstance(parameters, SegmentParameters):
        raise DynamicsError("post-jump installation requires segment parameters")
    parameters = replace(parameters)
    core = replace(
        core,
        continuous=replace(
            core.continuous, guard_accumulators=torch.zeros_like(core.continuous.guard_accumulators)
        ),
    )
    snapshot = prediction_snapshot_sha256(
        core.continuous,
        parameters,
        started_at=time,
        allowed_mode_mask=allowed_mode_mask(core.mode),
        parent_event_id=core.last_event_id,
    )
    return start_segment(
        core,
        parameters,
        time=time,
        parent_event_id=core.last_event_id,
        prediction_snapshot_sha256=snapshot,
    )
