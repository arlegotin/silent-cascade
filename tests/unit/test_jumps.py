"""Decision-driven jump contracts: legality, provenance, causality and installation."""

from dataclasses import fields, replace

import pytest
import torch

from silent_cascade.errors import DynamicsError, ProvenanceError, TimeOrderError
from silent_cascade.eventflow.flow import advance_to, start_segment
from silent_cascade.eventflow.guards import next_crossings, prediction_snapshot_sha256
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
    INTERNAL_EVENT_ID_BASE,
    ContinuousChannels,
    ContinuousState,
    RuntimeCore,
    RuntimeState,
    SegmentParameters,
    make_initial_continuous_state,
)
from silent_cascade.memory import BoundedMemory, append_perceived_fact
from silent_cascade.schemas import (
    ActivationPayload,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    Hypothesis,
    InternalEvent,
    InternalEventKind,
    LinkFact,
    Mode,
    Provenance,
)


def parameters(device: str = "cpu") -> SegmentParameters:
    initial = make_initial_continuous_state(device=device)
    return SegmentParameters(
        ContinuousChannels(
            **{f.name: getattr(initial, f.name) for f in fields(ContinuousChannels)}
        ),
        ContinuousChannels(
            **{
                f.name: torch.ones_like(getattr(initial, f.name))
                for f in fields(ContinuousChannels)
            }
        ),
        torch.full((3,), 1.5, device=device),
        torch.ones(3, device=device),
    )


def runtime(mode: Mode = Mode.OBSERVING, *, device: str = "cpu") -> RuntimeState:
    memory = append_perceived_fact(
        BoundedMemory(64), ExternalEvent(1, 0.0, ExternalEventKind.FACT, LinkFact(0, 1))
    )
    memory = append_perceived_fact(
        memory, ExternalEvent(2, 0.0, ExternalEventKind.FACT, HazardFact(1, 0, 5.0))
    )
    core = RuntimeCore(
        make_initial_continuous_state(device=device),
        mode=mode,
        memory=memory,
        focus_node_id=0 if mode is not Mode.OBSERVING else None,
        activation_time=0.5 if mode is not Mode.OBSERVING else None,
        active_record_id=1 if mode is Mode.HAVE_MEMORY else None,
        hypothesis=(
            Hypothesis(2, 7.5, 0.8, Provenance.INFERRED, (2,), False)
            if mode is Mode.HOLDING_HAZARD
            else None
        ),
        last_event_id=3,
    )
    return start_segment(
        core, parameters(device), time=1.0, parent_event_id=3, prediction_snapshot_sha256="a" * 64
    )


def internal(kind: InternalEventKind, *, timestamp: float = 1.0) -> InternalEvent:
    index = {
        InternalEventKind.RECALL: 0,
        InternalEventKind.COMPOSE: 1,
        InternalEventKind.ACT: 2,
        InternalEventKind.NOOP: 3,
    }[kind]
    return InternalEvent(INTERNAL_EVENT_ID_BASE, 3, timestamp, kind, index, 0.5)


def decision(role: ComposeRole = ComposeRole.LINK, **changes: object) -> ComposeDecision:
    values = dict(
        role=role,
        next_focus_node_id=17 if role is ComposeRole.LINK else None,
        hazard_type=3 if role is ComposeRole.HAZARD else None,
        deadline=9.5 if role is ComposeRole.HAZARD else None,
        confidence=0.7,
        append_support=True,
        continue_search=True,
    )
    values.update(changes)
    return ComposeDecision(**values)


def test_recall_carries_explicit_nonfirst_rank_and_compose_clears_it() -> None:
    state = runtime(Mode.SEARCHING)
    core = apply_recall(state, internal(InternalEventKind.RECALL), record_id=1, selected_rank=3)
    assert core.active_record_rank == 3
    recalled = begin_post_jump_segment(core, parameters(), time=1.0)
    event = InternalEvent(
        INTERNAL_EVENT_ID_BASE + 1, INTERNAL_EVENT_ID_BASE, 1.1, InternalEventKind.COMPOSE, 1, 0.1
    )
    composed = apply_compose(advance_to(recalled, 1.1), event, decision())
    assert composed.active_record_id is composed.active_record_rank is None


@pytest.mark.parametrize("rank", [0, -1, True, 1.0])
def test_recall_rejects_invalid_explicit_rank(rank) -> None:
    state = runtime(Mode.SEARCHING)
    with pytest.raises(DynamicsError, match="rank"):
        apply_recall(state, internal(InternalEventKind.RECALL), record_id=1, selected_rank=rank)


def test_runtime_rejects_rank_without_active_record() -> None:
    with pytest.raises(DynamicsError, match="rank"):
        replace(runtime().core, active_record_rank=2)


@pytest.mark.parametrize("mode", list(Mode))
@pytest.mark.parametrize(
    "kind,required,destination",
    [
        ("fact", Mode.OBSERVING, Mode.OBSERVING),
        ("activate", Mode.OBSERVING, Mode.SEARCHING),
        ("recall", Mode.SEARCHING, Mode.HAVE_MEMORY),
        ("compose", Mode.HAVE_MEMORY, Mode.SEARCHING),
        ("act", Mode.HOLDING_HAZARD, Mode.QUIESCENT),
    ],
)
def test_whole_source_mode_matrix(mode: Mode, kind: str, required: Mode, destination: Mode) -> None:
    state = runtime(mode)

    def apply() -> RuntimeCore:
        if kind == "fact":
            return apply_fact(state, ExternalEvent(4, 1.0, ExternalEventKind.FACT, LinkFact(5, 6)))
        if kind == "activate":
            return apply_activate(
                state, ExternalEvent(4, 1.0, ExternalEventKind.ACTIVATE, ActivationPayload(7))
            )
        if kind == "recall":
            return apply_recall(state, internal(InternalEventKind.RECALL), record_id=2)
        if kind == "compose":
            return apply_compose(state, internal(InternalEventKind.COMPOSE), decision())
        return apply_act(state, internal(InternalEventKind.ACT), hazard_type=1)

    if mode is required:
        assert apply().mode is destination
    else:
        with pytest.raises(DynamicsError):
            apply()


def test_fact_and_activation_apply_bounded_deterministic_impulses_without_mutation() -> None:
    state = runtime()
    event = ExternalEvent(4, 1.0, ExternalEventKind.FACT, LinkFact(5, 6))
    post = apply_fact(state, event)
    assert len(post.memory.records) == 3
    assert post.memory.lookup(4).record.provenance is Provenance.PERCEIVED
    for name in ("z_fast", "z_slow"):
        value = getattr(post.continuous, name)
        assert torch.count_nonzero(value) > 0
        assert bool((value.abs() <= 1.0).all())
        assert torch.equal(value, getattr(apply_fact(state, event).continuous, name))
        assert torch.count_nonzero(getattr(state.core.continuous, name)) == 0
    activated = apply_activate(
        state, ExternalEvent(4, 1.0, ExternalEventKind.ACTIVATE, ActivationPayload(7))
    )
    assert activated.activation_time == 1.0
    assert activated.focus_node_id == 7
    assert torch.count_nonzero(activated.continuous.focus_key) > 0
    assert activated.last_event_id == 4
    assert activated.counters.jump_applications == 1
    assert activated.executed_internal_events == 0


def test_recall_uses_selected_record_and_owns_refractory_and_event_accounting() -> None:
    state = runtime(Mode.SEARCHING)
    post = apply_recall(state, internal(InternalEventKind.RECALL), record_id=2)
    assert post.active_record_id == 2
    assert post.memory.lookup(2).refractory_until == 1.001
    assert post.same_kind_refractory_until == (1.001, 0.0, 0.0)
    assert post.memory.lookup(2).record is state.core.memory.lookup(2).record
    assert post.executed_internal_events == 1
    assert post.counters.jump_applications == 1
    assert post.counters.controller_calls == 0
    assert post.last_event_id == INTERNAL_EVENT_ID_BASE
    assert state.core.active_record_id is None


@pytest.mark.parametrize("role", list(ComposeRole))
@pytest.mark.parametrize("continue_search", [False, True])
def test_compose_roles_preserve_explicit_predictions_and_provenance(
    role: ComposeRole,
    continue_search: bool,
) -> None:
    state = runtime(Mode.HAVE_MEMORY)
    post = apply_compose(
        state, internal(InternalEventKind.COMPOSE), decision(role, continue_search=continue_search)
    )
    assert post.active_record_id is None
    assert post.memory.lookup(1).consumed
    assert post.memory.lookup(1).record == state.core.memory.lookup(1).record
    assert post.support_ids == (1,)
    if role is ComposeRole.LINK:
        assert post.focus_node_id == 17  # Stored target is 1; the prediction remains wrong.
    if role in (ComposeRole.HAZARD, ComposeRole.SAFE):
        assert post.hypothesis.provenance is Provenance.INFERRED
        assert post.hypothesis.support_ids == (1,)
        assert post.hypothesis.confidence == 0.7
        assert post.hypothesis.is_safe is (role is ComposeRole.SAFE)
        if role is ComposeRole.HAZARD:
            assert (post.hypothesis.hazard_type, post.hypothesis.deadline) == (3, 9.5)
        assert post.mode is (Mode.HOLDING_HAZARD if role is ComposeRole.HAZARD else Mode.QUIESCENT)
    else:
        assert post.hypothesis is None
        assert post.mode is (Mode.SEARCHING if continue_search else Mode.QUIESCENT)


def test_append_decision_is_respected_and_existing_supports_are_validated() -> None:
    state = runtime(Mode.HAVE_MEMORY)
    post = apply_compose(state, internal(InternalEventKind.COMPOSE), decision(append_support=False))
    assert post.support_ids == ()
    broken = replace(state, core=replace(state.core, support_ids=(999,)))
    with pytest.raises(ProvenanceError):
        apply_compose(broken, internal(InternalEventKind.COMPOSE), decision(ComposeRole.HAZARD))
    with pytest.raises((DynamicsError, ValueError)):
        apply_compose(
            state,
            internal(InternalEventKind.COMPOSE),
            decision(ComposeRole.HAZARD, append_support=False),
        )


def test_act_uses_predicted_class_emits_one_caused_action_and_disables_repetition() -> None:
    state = runtime(Mode.HOLDING_HAZARD)
    event = internal(InternalEventKind.ACT)
    post = apply_act(state, event, hazard_type=1)
    assert len(post.actions) == 1
    assert (
        post.actions[0].hazard_type,
        post.actions[0].timestamp,
        post.actions[0].caused_by_event_id,
    ) == (1, 1.0, event.event_id)
    assert post.mode is Mode.QUIESCENT
    again = begin_post_jump_segment(
        replace(post, mode=Mode.HOLDING_HAZARD), parameters(), time=event.timestamp
    )
    again = advance_to(again, 1.01)
    second_event = replace(
        event, event_id=event.event_id + 1, parent_event_id=event.event_id, timestamp=again.time
    )
    with pytest.raises(DynamicsError, match="no previous action"):
        apply_act(again, second_event, hazard_type=1)
    missing = replace(state, core=replace(state.core, hypothesis=None))
    with pytest.raises(DynamicsError):
        apply_act(missing, event, hazard_type=1)


@pytest.mark.parametrize("fault", ["missing", "invalid", "consumed", "refractory"])
def test_recall_rejects_illegal_selected_records(fault: str) -> None:
    state = runtime(Mode.SEARCHING)
    if fault != "missing":
        slot = state.core.memory.lookup(2)
        updates = {
            "invalid": {"valid": False},
            "consumed": {"consumed": True},
            "refractory": {"refractory_until": 2.0},
        }[fault]
        memory = replace(
            state.core.memory, records=(state.core.memory.records[0], replace(slot, **updates))
        )
        state = replace(state, core=replace(state.core, memory=memory))
    with pytest.raises(DynamicsError):
        apply_recall(
            state, internal(InternalEventKind.RECALL), record_id=999 if fault == "missing" else 2
        )


@pytest.mark.parametrize("mode", list(Mode))
def test_endogenous_noop_is_rejected_in_every_mode(mode: Mode) -> None:
    with pytest.raises(DynamicsError):
        apply_recall(runtime(mode), internal(InternalEventKind.NOOP), record_id=1)


@pytest.mark.parametrize(
    "fault", ["parent", "time", "id", "index", "activation", "limit", "refractory"]
)
def test_internal_causality_and_runtime_limits_fail_closed(fault: str) -> None:
    state = runtime(Mode.SEARCHING)
    event = internal(InternalEventKind.RECALL)
    if fault in ("parent", "time", "id", "index"):
        changes = {
            "parent": {"parent_event_id": 99},
            "time": {"timestamp": 2.0},
            "id": {"event_id": 5},
            "index": {"guard_index": 2},
        }[fault]
        event = replace(event, **changes)
    else:
        changes = {
            "activation": {"activation_time": None},
            "limit": {"executed_internal_events": 64},
            "refractory": {"same_kind_refractory_until": (2.0, 0.0, 0.0)},
        }[fault]
        state = replace(state, core=replace(state.core, **changes))
    with pytest.raises((DynamicsError, TimeOrderError)):
        apply_recall(state, event, record_id=1)


def test_compose_requires_an_active_valid_unconsumed_record() -> None:
    state = runtime(Mode.HAVE_MEMORY)
    for record_id in (None, 999):
        with pytest.raises(DynamicsError):
            apply_compose(
                replace(state, core=replace(state.core, active_record_id=record_id)),
                internal(InternalEventKind.COMPOSE),
                decision(),
            )


@pytest.mark.parametrize(
    "changes",
    [
        {"role": "link"},
        {"next_focus_node_id": 64},
        {"confidence": float("nan")},
        {"append_support": 1},
        {"continue_search": 1},
        {"hazard_type": 1},
        {"next_focus_node_id": None},
    ],
)
def test_compose_decision_rejects_malformed_prediction_schema(changes: dict) -> None:
    with pytest.raises(DynamicsError):
        decision(**changes)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_post_jump_install_resets_guards_binds_real_snapshot_and_clones_origin(device: str) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    state = runtime(Mode.SEARCHING, device=device)
    post = apply_recall(state, internal(InternalEventKind.RECALL), record_id=2)
    post = replace(
        post,
        continuous=replace(
            post.continuous, guard_accumulators=torch.tensor([0.25, 0.5, 1.0], device=device)
        ),
    )
    installed = begin_post_jump_segment(post, parameters(device), time=state.time)
    assert installed.segment.started_at == installed.time == 1.0
    assert installed.segment.parent_event_id == INTERNAL_EVENT_ID_BASE
    assert torch.count_nonzero(installed.core.continuous.guard_accumulators) == 0
    assert installed.core.continuous.guard_accumulators.dtype is torch.float32
    assert installed.segment.prediction_snapshot_sha256 == prediction_snapshot_sha256(
        installed.segment.origin,
        installed.segment.parameters,
        started_at=1.0,
        allowed_mode_mask=(False, True, False),
        parent_event_id=INTERNAL_EVENT_ID_BASE,
    )
    assert next_crossings(installed)[0].kind is InternalEventKind.COMPOSE
    assert advance_to(installed, 1.0) is installed
    assert installed.core.counters == post.counters
    for field in fields(ContinuousState):
        assert (
            getattr(installed.segment.origin, field.name).data_ptr()
            != getattr(post.continuous, field.name).data_ptr()
        )
    assert torch.count_nonzero(post.continuous.guard_accumulators) == 3


def test_install_revalidates_parameters_and_rejects_terminal_or_missing_parent() -> None:
    post = apply_activate(
        runtime(), ExternalEvent(4, 1.0, ExternalEventKind.ACTIVATE, ActivationPayload(7))
    )
    param = parameters()
    param.flow_rates.z_fast[0] = 0.0
    with pytest.raises(DynamicsError):
        begin_post_jump_segment(post, param, time=1.0)
    for changes in ({"mode": Mode.TERMINAL}, {"last_event_id": None}):
        with pytest.raises(DynamicsError):
            begin_post_jump_segment(replace(post, **changes), parameters(), time=1.0)


def test_install_rejects_relabeling_a_jump_origin_at_a_different_time() -> None:
    state = runtime(Mode.SEARCHING)
    event = internal(InternalEventKind.RECALL)
    post = apply_recall(state, event, record_id=2)
    before = post.continuous.z_fast.clone()
    with pytest.raises(TimeOrderError):
        begin_post_jump_segment(post, parameters(), time=2.0)
    assert torch.equal(post.continuous.z_fast, before)
    assert post.last_event_time == event.timestamp == 1.0
    installed = begin_post_jump_segment(post, parameters(), time=1.0)
    assert installed.segment.started_at == installed.core.last_event_time == 1.0


def test_install_requires_an_authoritative_jump_time() -> None:
    # Low-level flow fixtures may have an ID without materializing a causal jump.
    core = runtime(Mode.SEARCHING).core
    with pytest.raises(DynamicsError):
        begin_post_jump_segment(core, parameters(), time=1.0)


def test_exactly_the_64th_internal_event_is_allowed() -> None:
    state = runtime(Mode.SEARCHING)
    state = replace(state, core=replace(state.core, executed_internal_events=63))
    post = apply_recall(state, internal(InternalEventKind.RECALL), record_id=2)
    assert post.executed_internal_events == 64


@pytest.mark.parametrize("fault", ["safe", "perceived", "unknown_support", "invalid_support"])
def test_action_rejects_invalid_hypothesis_provenance_or_supports(fault: str) -> None:
    state = runtime(Mode.HOLDING_HAZARD)
    hypothesis = state.core.hypothesis
    if fault == "safe":
        hypothesis = replace(hypothesis, is_safe=True, hazard_type=None, deadline=None)
    elif fault == "perceived":
        hypothesis = replace(hypothesis, provenance=Provenance.PERCEIVED)
    elif fault == "unknown_support":
        hypothesis = replace(hypothesis, support_ids=(999,))
    else:
        memory = replace(
            state.core.memory,
            records=(
                state.core.memory.records[0],
                replace(state.core.memory.records[1], valid=False),
            ),
        )
        state = replace(state, core=replace(state.core, memory=memory))
    state = replace(state, core=replace(state.core, hypothesis=hypothesis))
    with pytest.raises((DynamicsError, ProvenanceError)):
        apply_act(state, internal(InternalEventKind.ACT), hazard_type=1)


@pytest.mark.parametrize("kind", ["fact", "activate", "recall", "compose", "act"])
def test_each_jump_resets_guards_and_installs_its_own_origin(kind: str) -> None:
    mode = {
        "fact": Mode.OBSERVING,
        "activate": Mode.OBSERVING,
        "recall": Mode.SEARCHING,
        "compose": Mode.HAVE_MEMORY,
        "act": Mode.HOLDING_HAZARD,
    }[kind]
    state = runtime(mode)
    state = replace(
        state,
        core=replace(
            state.core,
            continuous=replace(
                state.core.continuous, guard_accumulators=torch.tensor([1.0, 0.6, 0.1])
            ),
        ),
    )
    if kind == "fact":
        event = ExternalEvent(4, 1.0, ExternalEventKind.FACT, LinkFact(3, 4))
        post = apply_fact(state, event)
    elif kind == "activate":
        event = ExternalEvent(4, 1.0, ExternalEventKind.ACTIVATE, ActivationPayload(3))
        post = apply_activate(state, event)
    elif kind == "recall":
        event = internal(InternalEventKind.RECALL)
        post = apply_recall(state, event, record_id=2)
    elif kind == "compose":
        event = internal(InternalEventKind.COMPOSE)
        post = apply_compose(state, event, decision())
    else:
        event = internal(InternalEventKind.ACT)
        post = apply_act(state, event, hazard_type=1)
    assert torch.count_nonzero(post.continuous.guard_accumulators) == 0
    assert post.last_event_time == event.timestamp
    installed = begin_post_jump_segment(post, parameters(), time=state.time)
    assert installed.segment.parent_event_id == event.event_id
    assert torch.count_nonzero(installed.segment.origin.guard_accumulators) == 0
    assert torch.count_nonzero(state.core.continuous.guard_accumulators) == 3
    assert installed.core.counters.jump_applications == 1
    assert installed.core.counters.controller_calls == 0
    assert installed.core.counters.foundation_model_calls == 0
    assert installed.core.consecutive_gap_clamps == state.core.consecutive_gap_clamps
    expected_masks = {
        Mode.OBSERVING: (False, False, False),
        Mode.SEARCHING: (True, False, False),
        Mode.HAVE_MEMORY: (False, True, False),
        Mode.QUIESCENT: (False, False, False),
    }
    assert installed.segment.prediction_snapshot_sha256 == prediction_snapshot_sha256(
        installed.segment.origin,
        installed.segment.parameters,
        started_at=state.time,
        allowed_mode_mask=expected_masks[post.mode],
        parent_event_id=event.event_id,
    )


@pytest.mark.parametrize("kind", [ExternalEventKind.FACT, ExternalEventKind.ACTIVATE])
def test_external_jumps_require_matching_time_and_no_prior_activation(
    kind: ExternalEventKind,
) -> None:
    state = runtime()
    event = ExternalEvent(
        4, 2.0, kind, LinkFact(0, 1) if kind is ExternalEventKind.FACT else ActivationPayload(0)
    )
    apply = apply_fact if kind is ExternalEventKind.FACT else apply_activate
    with pytest.raises(TimeOrderError):
        apply(state, event)
    state = replace(state, core=replace(state.core, activation_time=0.5))
    with pytest.raises(DynamicsError):
        apply(state, replace(event, timestamp=1.0))
