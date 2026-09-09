"""Mutations must fail at runtime boundaries, including mutable tensor storage."""

import json
from copy import deepcopy
from dataclasses import fields, replace

import pytest

from silent_cascade.errors import DynamicsError
from silent_cascade.eventflow.flow import advance_to
from silent_cascade.eventflow.invariants import (
    validate_post_jump,
    validate_runtime_state,
    validate_session_boundary,
)
from silent_cascade.eventflow.scripted import ScriptedEventFlowAgent
from silent_cascade.eventflow.state import ContinuousChannels
from silent_cascade.schemas import (
    Action,
    ActivationPayload,
    AgentInit,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    InternalEventKind,
    Mode,
    Provenance,
)


def trajectory():
    agent = ScriptedEventFlowAgent()
    state = agent.initialize(AgentInit("public", 64, 4, 0.0))
    for event in (
        ExternalEvent(1, 1.0, ExternalEventKind.FACT, HazardFact(0, 2, 10.0)),
        ExternalEvent(2, 2.0, ExternalEventKind.ACTIVATE, ActivationPayload(0)),
    ):
        state = agent.on_external(advance_to(state, event.timestamp), event)
    recall = agent.next_internal_event(state)
    state, _ = agent.on_internal(advance_to(state, recall.timestamp), recall)
    compose = agent.next_internal_event(state)
    before = advance_to(state, compose.timestamp)
    after, _ = agent.on_internal(before, compose)
    return before, compose, after


def test_valid_post_jump_and_materialized_boundary():
    before, event, after = trajectory()
    validate_post_jump(before, event, after)
    validate_runtime_state(after)
    validate_session_boundary(advance_to(after, after.time + 0.01))


@pytest.mark.parametrize("offset", [-1, 1, 100])
def test_live_boundary_rejects_noncanonical_internal_ordinal(offset):
    before, event, _ = trajectory()
    with pytest.raises(DynamicsError) as caught:
        validate_session_boundary(
            before, next_event=replace(event, event_id=event.event_id + offset)
        )
    assert caught.value.context["invariant"] == "internal_event_id"


def test_live_boundary_rejects_reordered_older_internal_ordinal():
    _, _, state = trajectory()
    event = ScriptedEventFlowAgent().next_internal_event(state)
    assert state.core.executed_internal_events == 2
    with pytest.raises(DynamicsError) as caught:
        validate_session_boundary(state, next_event=replace(event, event_id=event.event_id - 2))
    assert caught.value.context["invariant"] == "internal_event_id"


def corrupt(state, mutation):
    core = state.core
    if mutation == "nan_time":
        object.__setattr__(state, "time", float("nan"))
    elif mutation == "decreasing_time":
        object.__setattr__(state, "time", 0.0)
    elif mutation == "accumulator_not_reset":
        core.continuous.guard_accumulators[0] = 0.2
    elif mutation == "duplicate_memory":
        object.__setattr__(core.memory, "records", core.memory.records * 2)
    elif mutation == "changed_perceived":
        object.__setattr__(core.memory.records[0].record, "hazard_type", 3)
    elif mutation == "changed_provenance":
        object.__setattr__(core.memory.records[0].record, "provenance", Provenance.INFERRED)
    elif mutation.startswith("support_"):
        supports = {"support_empty": (), "support_unknown": (999,), "support_duplicate": (1, 1)}
        object.__setattr__(core.hypothesis, "support_ids", supports[mutation])
    elif mutation == "illegal_destination":
        object.__setattr__(core, "mode", Mode.OBSERVING)
    elif mutation == "parent_mismatch":
        object.__setattr__(state.segment, "parent_event_id", 999)
    elif mutation == "hash_mismatch":
        object.__setattr__(state.segment, "prediction_snapshot_sha256", "f" * 64)
    elif mutation == "hash_noncanonical":
        object.__setattr__(state.segment, "prediction_snapshot_sha256", "G" * 64)
    elif mutation == "action_before_activation":
        object.__setattr__(core, "actions", (Action(2, 1.0, 1 << 62),))
    elif mutation == "two_actions":
        object.__setattr__(core, "actions", (Action(2, state.time, 1 << 62),) * 2)
    elif mutation == "model_calls":
        object.__setattr__(core.counters, "foundation_model_calls", 1)
    elif mutation == "internal_count":
        object.__setattr__(core, "executed_internal_events", 65)
    elif mutation == "active_rank":
        object.__setattr__(core, "active_record_rank", 1)


@pytest.mark.parametrize(
    "mutation",
    [
        "nan_time",
        "decreasing_time",
        "accumulator_not_reset",
        "duplicate_memory",
        "changed_perceived",
        "changed_provenance",
        "support_empty",
        "support_unknown",
        "support_duplicate",
        "illegal_destination",
        "parent_mismatch",
        "hash_mismatch",
        "hash_noncanonical",
        "action_before_activation",
        "two_actions",
        "model_calls",
        "internal_count",
        "active_rank",
    ],
)
def test_post_jump_mutation_table_fails_with_bounded_typed_context(mutation):
    before, event, after = trajectory()
    # Independent snapshot avoids aliasing the previous immutable semantic record.
    after = deepcopy(after)
    corrupt(after, mutation)
    with pytest.raises(DynamicsError) as caught:
        validate_post_jump(before, event, after)
    assert caught.value.code in {"dynamics_error", "time_order_error"}
    assert len(json.dumps(caught.value.to_payload(), allow_nan=False)) < 1024


@pytest.mark.parametrize("location", ["core", "origin", "target", "rate"])
@pytest.mark.parametrize("channel", [item.name for item in fields(ContinuousChannels)])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -2.0, 21.0])
def test_each_mutable_latent_tensor_is_revalidated(location, channel, value):
    _, _, state = trajectory()
    tensors = {
        "core": state.core.continuous,
        "origin": state.segment.origin,
        "target": state.segment.parameters.flow_targets,
        "rate": state.segment.parameters.flow_rates,
    }
    getattr(tensors[location], channel)[0] = value
    with pytest.raises(DynamicsError):
        validate_runtime_state(state)


@pytest.mark.parametrize(
    "name,values",
    [
        ("guard_rates", [float("nan"), float("inf"), 0.0, -1.0, 501.0]),
        ("guard_targets", [float("nan"), float("inf"), 0.0, -1.0, 2.0]),
    ],
)
def test_mutable_guard_parameters_are_revalidated(name, values):
    for value in values:
        _, _, state = trajectory()
        getattr(state.segment.parameters, name)[0] = value
        with pytest.raises(DynamicsError):
            validate_runtime_state(state)


@pytest.mark.parametrize(
    "mutation",
    ["backward", "negative_delta", "nan_delta", "parent", "noop", "wrong_mode", "terminal"],
)
def test_next_event_boundary_rejects_illegal_predictions(mutation):
    before, event, _ = trajectory()
    event = replace(event)
    if mutation == "backward":
        object.__setattr__(event, "timestamp", before.time - 0.1)
    elif mutation in {"negative_delta", "nan_delta"}:
        object.__setattr__(
            event, "predicted_delta", -1.0 if mutation == "negative_delta" else float("nan")
        )
    elif mutation == "parent":
        object.__setattr__(event, "parent_event_id", 0)
    elif mutation == "noop":
        object.__setattr__(event, "kind", InternalEventKind.NOOP)
    elif mutation == "wrong_mode":
        object.__setattr__(event, "kind", InternalEventKind.ACT)
    else:
        object.__setattr__(before.core, "mode", Mode.TERMINAL)
    with pytest.raises(DynamicsError):
        validate_session_boundary(before, next_event=event)


def test_session_boundary_checks_prediction_snapshot_identity():
    before, event, _ = trajectory()
    with pytest.raises(DynamicsError):
        validate_session_boundary(before, next_event=event, prediction_snapshot_sha256="0" * 64)


def test_post_jump_rejects_illegal_source_even_with_valid_destination():
    before, event, after = trajectory()
    object.__setattr__(before.core, "mode", Mode.SEARCHING)
    with pytest.raises(DynamicsError):
        validate_post_jump(before, event, after)


@pytest.mark.parametrize("focus", [True, 1.0, -1, 64, "1"])
def test_public_focus_requires_an_exact_in_range_entity_id(focus):
    _, _, state = trajectory()
    object.__setattr__(state.core, "focus_node_id", focus)
    with pytest.raises(DynamicsError):
        validate_runtime_state(state)


def test_have_memory_requires_public_focus_and_bounded_rank():
    before, _, _ = trajectory()
    for field, value in (("focus_node_id", None), ("active_record_rank", 65)):
        changed = deepcopy(before)
        object.__setattr__(changed.core, field, value)
        with pytest.raises(DynamicsError):
            validate_runtime_state(changed)


@pytest.mark.parametrize("mutation", ["activation", "supports", "unconsumed", "active_record"])
def test_compose_cannot_rewrite_activation_or_omit_consumption(mutation):
    before, event, after = trajectory()
    after = deepcopy(after)
    if mutation == "activation":
        object.__setattr__(after.core, "activation_time", 1.5)
    elif mutation == "supports":
        object.__setattr__(after.core, "support_ids", ())
    elif mutation == "unconsumed":
        object.__setattr__(after.core.memory.records[0], "consumed", False)
    else:
        object.__setattr__(after.core, "active_record_id", 1)
    with pytest.raises(DynamicsError):
        validate_post_jump(before, event, after)


@pytest.mark.parametrize("kind", [ExternalEventKind.FACT, ExternalEventKind.ACTIVATE])
def test_boundary_rejects_ordinary_external_after_activation(kind):
    before, _, _ = trajectory()
    event = ExternalEvent(
        500,
        before.time,
        kind,
        HazardFact(0, 2, 10.0) if kind is ExternalEventKind.FACT else ActivationPayload(0),
    )
    with pytest.raises(DynamicsError):
        validate_session_boundary(before, next_event=event)
