"""Fail-loud public runtime boundaries; no private truth or scoring inputs."""

import math
from dataclasses import fields, replace
from functools import wraps

import torch

from silent_cascade.errors import DynamicsError, ProvenanceError, TimeOrderError
from silent_cascade.eventflow.flow import state_at
from silent_cascade.eventflow.guards import (
    GUARD_KIND_BY_INDEX,
    allowed_mode_mask,
)
from silent_cascade.eventflow.guards import (
    prediction_snapshot_sha256 as snapshot_digest,
)
from silent_cascade.eventflow.state import (
    INTERNAL_EVENT_ID_BASE,
    ContinuousChannels,
    RuntimeCore,
    RuntimeState,
    require_time,
)
from silent_cascade.memory import require_support_ledger
from silent_cascade.schemas import (
    Action,
    Condition,
    ExternalEvent,
    ExternalEventKind,
    InternalEvent,
    InternalEventKind,
    Mode,
    Provenance,
    fact_event_to_memory_record,
)


def _require(valid: bool, invariant: str, *, time: bool = False) -> None:
    if not valid:
        error = TimeOrderError if time else DynamicsError
        raise error("runtime invariant failed", context={"invariant": invariant})


def _typed_boundary(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ProvenanceError, TypeError, ValueError, AttributeError, IndexError) as error:
            raise DynamicsError(
                "runtime structure or provenance is invalid",
                context={"invariant": "runtime_structure"},
            ) from error

    return checked


def _validate_core(core: RuntimeCore, time: float) -> None:
    _require(isinstance(core, RuntimeCore), "runtime_core")
    replace(core.continuous)
    replace(core)
    replace(core.counters)
    _require(isinstance(core.mode, Mode), "mode")
    if core.focus_node_id is not None:
        _require(type(core.focus_node_id) is int and 0 <= core.focus_node_id <= 63, "public_focus")
    if core.mode in (Mode.SEARCHING, Mode.HAVE_MEMORY):
        _require(core.focus_node_id is not None, "active_focus")
    if core.active_record_rank is not None:
        _require(core.active_record_rank <= 64, "active_record_rank_bound")
    _require(core.counters.foundation_model_calls == 0, "offline_zero_model_calls")
    for name, maximum in (("executed_internal_events", 64), ("consecutive_gap_clamps", 4)):
        value = getattr(core, name)
        _require(type(value) is int and 0 <= value <= maximum, name)
    _require(
        isinstance(core.same_kind_refractory_until, tuple)
        and len(core.same_kind_refractory_until) == 3,
        "refractory_shape",
    )
    for value in core.same_kind_refractory_until:
        require_time(value, "refractory release")
    if core.activation_time is not None:
        require_time(core.activation_time, "activation time")
        _require(core.activation_time <= time, "activation_time", time=True)
    if core.mode is Mode.OBSERVING:
        _require(core.activation_time is None, "observing_before_activation")
    elif core.mode is not Mode.TERMINAL:
        _require(core.activation_time is not None, "active_mode_requires_activation")
    if core.last_event_time is not None:
        require_time(core.last_event_time, "last event time")
        _require(core.last_event_time <= time, "last_event_time", time=True)
    _require((core.last_event_id is None) == (core.last_event_time is None), "last_event_identity")
    if core.last_event_id is not None:
        _require(type(core.last_event_id) is int and core.last_event_id >= 0, "last_event_id")
    replace(core.memory)
    for slot in core.memory.records:
        replace(slot)
        replace(slot.record)
        require_time(slot.refractory_until, "record refractory release")
        _require(slot.record.provenance is Provenance.PERCEIVED, "perceived_memory")
        _require(slot.record.support_ids == (), "perceived_support")
        _require(0.0 <= slot.record.observed_at <= time, "record_observation_time", time=True)
    require_support_ledger(core.memory, core.support_ids)
    if core.active_record_id is not None:
        _require(type(core.active_record_id) is int, "active_record_id")
        slot = core.memory.lookup(core.active_record_id)
        _require(slot.valid and not slot.consumed, "active_record_legal")
        _require(core.mode in (Mode.HAVE_MEMORY, Mode.TERMINAL), "active_record_mode")
    if core.mode is Mode.HAVE_MEMORY:
        _require(core.active_record_id is not None, "have_memory_active_record")
    if core.hypothesis is not None:
        replace(core.hypothesis)
        _require(core.hypothesis.provenance is Provenance.INFERRED, "inferred_hypothesis")
        _require(bool(core.hypothesis.support_ids), "inferred_support_nonempty")
        require_support_ledger(core.memory, core.hypothesis.support_ids)
        _require(core.hypothesis.support_ids == core.support_ids, "hypothesis_support_ledger")
    if core.mode is Mode.HOLDING_HAZARD:
        _require(
            core.hypothesis is not None and not core.hypothesis.is_safe, "holding_hazard_hypothesis"
        )
    _require(isinstance(core.actions, tuple) and len(core.actions) <= 1, "at_most_one_action")
    for action in core.actions:
        _require(isinstance(action, Action), "action_type")
        replace(action)
        _require(
            core.activation_time is not None and core.activation_time <= action.timestamp <= time,
            "action_after_activation",
            time=True,
        )
        _require(action.caused_by_event_id >= INTERNAL_EVENT_ID_BASE, "action_causal_id")
        _require(core.mode in (Mode.QUIESCENT, Mode.TERMINAL), "action_mode")


@_typed_boundary
def validate_runtime_state(state: RuntimeState) -> None:
    """Revalidate actual tensor storage, metadata and the anchored guard digest.

    Terminal states retain their preceding segment and reset guards, so their
    latent flow is checked against that segment without claiming a new origin.
    """
    _require(isinstance(state, RuntimeState), "runtime_state")
    require_time(state.time, "runtime time")
    _validate_core(state.core, state.time)
    segment = state.segment
    replace(segment.origin)
    replace(segment.parameters.flow_targets)
    replace(segment.parameters.flow_rates)
    replace(segment.parameters)
    replace(segment)
    _require(state.time >= segment.started_at, "segment_time", time=True)
    _require(state.core.continuous.device == segment.origin.device, "runtime_device")
    _require(
        not bool(torch.count_nonzero(segment.origin.guard_accumulators)), "origin_guards_reset"
    )
    if state.core.mode is not Mode.TERMINAL:
        _require(
            segment.parent_event_id
            == (state.core.last_event_id if state.core.last_event_id is not None else 0),
            "segment_parent",
        )
        if state.core.last_event_time is not None:
            _require(
                segment.started_at == state.core.last_event_time, "segment_origin_time", time=True
            )
        modes = (state.core.mode,)
    else:
        modes = tuple(mode for mode in Mode if mode is not Mode.TERMINAL)
    _require(
        any(
            segment.prediction_snapshot_sha256
            == snapshot_digest(
                segment.origin,
                segment.parameters,
                started_at=segment.started_at,
                allowed_mode_mask=allowed_mode_mask(mode),
                parent_event_id=segment.parent_event_id,
            )
            for mode in modes
        ),
        "prediction_snapshot",
    )
    expected = state_at(state, state.time)
    for item in fields(ContinuousChannels):
        _require(
            torch.equal(getattr(state.core.continuous, item.name), getattr(expected, item.name)),
            "continuous_matches_segment",
        )
    if state.core.mode is Mode.TERMINAL:
        _require(
            not bool(torch.count_nonzero(state.core.continuous.guard_accumulators)),
            "terminal_guards_reset",
        )
    else:
        _require(
            torch.equal(state.core.continuous.guard_accumulators, expected.guard_accumulators),
            "guards_match_segment",
        )


@_typed_boundary
def validate_session_boundary(
    state: RuntimeState,
    *,
    next_event: ExternalEvent | InternalEvent | None = None,
    condition: Condition = Condition.EVENT_FLOW,
    prediction_snapshot_sha256: str | None = None,
) -> None:
    """Validate a selected event or newly produced prediction at its cursor.

    A cached losing proposal must not be passed here after a pause has passed
    its timestamp: selection first decides the effective winner at the origin.
    """
    validate_runtime_state(state)
    if prediction_snapshot_sha256 is not None:
        _require(
            prediction_snapshot_sha256 == state.segment.prediction_snapshot_sha256,
            "cached_prediction_snapshot",
        )
    if next_event is None:
        return
    _require(isinstance(next_event, (ExternalEvent, InternalEvent)), "event_type")
    require_time(next_event.timestamp, "next event time")
    _require(next_event.timestamp >= state.time, "next_event_time", time=True)
    _require(state.core.mode is not Mode.TERMINAL, "no_event_after_terminal")
    replace(next_event)
    if isinstance(next_event, ExternalEvent):
        _require(next_event.event_id < INTERNAL_EVENT_ID_BASE, "external_event_id")
        if next_event.kind in (ExternalEventKind.FACT, ExternalEventKind.ACTIVATE):
            _require(
                state.core.mode is Mode.OBSERVING and state.core.activation_time is None,
                "external_before_activation",
            )
    if isinstance(next_event, InternalEvent):
        _require(
            condition is not Condition.EVENT_FLOW or next_event.kind is not InternalEventKind.NOOP,
            "eventflow_noop",
        )
        _require(next_event.kind in GUARD_KIND_BY_INDEX, "endogenous_kind")
        index = GUARD_KIND_BY_INDEX.index(next_event.kind)
        _require(
            next_event.guard_index == index and allowed_mode_mask(state.core.mode)[index],
            "prediction_mode",
        )
        _require(state.core.activation_time is not None, "prediction_after_activation")
        _require(
            next_event.event_id == INTERNAL_EVENT_ID_BASE + state.core.executed_internal_events,
            "internal_event_id",
        )
        _require(
            next_event.parent_event_id == state.segment.parent_event_id == state.core.last_event_id,
            "prediction_parent",
        )
        _require(
            type(next_event.predicted_delta) is float
            and math.isfinite(next_event.predicted_delta)
            and next_event.predicted_delta >= 0.0,
            "prediction_delta",
        )


@_typed_boundary
def validate_post_jump(
    before: RuntimeState, event: ExternalEvent | InternalEvent, after: RuntimeState | RuntimeCore
) -> None:
    """Check the declared transition and all state before trace/session commit."""
    validate_session_boundary(before, next_event=event)
    _require(before.time == event.timestamp, "jump_time", time=True)
    post = after.core if isinstance(after, RuntimeState) else after
    _validate_core(post, event.timestamp)
    terminal = isinstance(event, ExternalEvent) and event.kind in (
        ExternalEventKind.OUTCOME,
        ExternalEventKind.END,
    )
    transitions = {
        ExternalEventKind.FACT: (Mode.OBSERVING, (Mode.OBSERVING,)),
        ExternalEventKind.ACTIVATE: (Mode.OBSERVING, (Mode.SEARCHING,)),
        InternalEventKind.RECALL: (Mode.SEARCHING, (Mode.HAVE_MEMORY,)),
        InternalEventKind.COMPOSE: (
            Mode.HAVE_MEMORY,
            (Mode.SEARCHING, Mode.HOLDING_HAZARD, Mode.QUIESCENT),
        ),
        InternalEventKind.ACT: (Mode.HOLDING_HAZARD, (Mode.QUIESCENT,)),
    }
    if terminal:
        _require(
            isinstance(after, RuntimeCore) and post.mode is Mode.TERMINAL, "terminal_destination"
        )
    else:
        source, destinations = transitions[event.kind]
        _require(before.core.mode is source and post.mode in destinations, "jump_transition")
        _require(isinstance(after, RuntimeState), "post_jump_segment")
        validate_runtime_state(after)
        _require(
            after.time == after.segment.started_at == event.timestamp, "post_jump_time", time=True
        )
    _require(
        post.last_event_id == event.event_id and post.last_event_time == event.timestamp,
        "post_jump_identity",
    )
    _require(
        not bool(torch.count_nonzero(post.continuous.guard_accumulators)), "post_jump_guards_reset"
    )
    prior_records = tuple(slot.record for slot in before.core.memory.records)
    post_records = tuple(slot.record for slot in post.memory.records)
    expected_records = prior_records
    if isinstance(event, ExternalEvent) and event.kind is ExternalEventKind.FACT:
        expected_records += (fact_event_to_memory_record(event),)
    _require(post_records == expected_records, "perceived_records_preserved")
    for old, new in zip(before.core.memory.records, post.memory.records, strict=False):
        _require(
            new.refractory_until >= old.refractory_until and (not old.consumed or new.consumed),
            "memory_metadata_monotone",
        )
    _require(post.actions[: len(before.core.actions)] == before.core.actions, "actions_preserved")
    new_actions = post.actions[len(before.core.actions) :]
    if isinstance(event, InternalEvent) and event.kind is InternalEventKind.ACT:
        _require(
            len(new_actions) == 1
            and new_actions[0].timestamp == event.timestamp
            and new_actions[0].caused_by_event_id == event.event_id,
            "act_emission",
        )
    else:
        _require(not new_actions, "only_act_emits")
    internal = isinstance(event, InternalEvent)
    if isinstance(event, ExternalEvent) and event.kind is ExternalEventKind.ACTIVATE:
        _require(
            post.activation_time == event.timestamp
            and post.focus_node_id == event.payload.start_node,
            "activation_initializes_public_focus",
        )
    else:
        _require(post.activation_time == before.core.activation_time, "activation_time_preserved")
    if internal and event.kind is InternalEventKind.COMPOSE:
        active = before.core.active_record_id
        _require(
            post.active_record_id is None and post.active_record_rank is None,
            "compose_clears_active",
        )
        _require(post.memory.lookup(active).consumed, "compose_consumes_active")
        supports = before.core.support_ids
        appended = supports if active in supports else (*supports, active)
        _require(post.support_ids in (supports, appended), "compose_support_append")
    else:
        _require(post.support_ids == before.core.support_ids, "support_ledger_preserved")
    _require(
        post.executed_internal_events == before.core.executed_internal_events + int(internal),
        "internal_count_owner",
    )
    _require(
        post.counters.jump_applications == before.core.counters.jump_applications + 1, "one_jump"
    )
    _require(post.consecutive_gap_clamps == before.core.consecutive_gap_clamps, "clamp_owner")
    for item in fields(post.counters):
        _require(
            getattr(post.counters, item.name) >= getattr(before.core.counters, item.name),
            "counters_monotone",
        )
    expected_refractory = before.core.same_kind_refractory_until
    if internal:
        index = GUARD_KIND_BY_INDEX.index(event.kind)
        _require(event.timestamp >= expected_refractory[index], "same_kind_refractory")
        expected_refractory = tuple(
            max(value, event.timestamp + 1e-3) if i == index else value
            for i, value in enumerate(expected_refractory)
        )
    _require(post.same_kind_refractory_until == expected_refractory, "refractory_owner")
    if terminal:
        validate_runtime_state(replace(before, core=post))
