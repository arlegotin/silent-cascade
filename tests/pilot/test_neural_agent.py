from dataclasses import replace

import pytest
import torch

from silent_cascade.eventflow.flow import state_at as continuous_at
from silent_cascade.schemas import Mode


def state_at(state, time):
    return replace(
        state, time=time, core=replace(state.core, continuous=continuous_at(state, time))
    )


def prepare(case, model):
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent

    agent = NeuralEventFlowAgent(model, identity=case.identity, device="cpu")
    state = agent.initialize(case.init)
    for event in case.events:
        state = agent.on_external(state_at(state, event.timestamp), event)
    return agent, state


def step(agent, state):
    event = agent.next_internal_event(state)
    return agent.on_internal(state_at(state, event.timestamp), event)[0]


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_act_uses_action_logits_not_hypothesis(runtime_case, controlled_model, device):
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    controlled_model.to(device)
    result = runtime_case.run(
        NeuralEventFlowAgent(controlled_model, identity=runtime_case.identity, device=device)
    )
    assert result.actions[0].hazard_type == 1
    assert result.score.timed_success is False
    assert runtime_case.diagnostics.raw_action_argmax == 4
    assert result.counters.foundation_model_calls == 0
    assert all(not context.workspace.latent.requires_grad for _, context in controlled_model.calls)


def test_wrong_memory_class_and_deadline_are_not_repaired(runtime_case, controlled_model):
    agent, state = prepare(runtime_case, controlled_model)
    state = step(agent, state)
    assert state.core.active_record_id == 30  # unreachable class 0
    state = step(agent, state)
    assert state.core.hypothesis.hazard_type == 3
    assert state.core.hypothesis.deadline == pytest.approx(5.0 + 7.0 * (1 + state.time - 5.0))
    assert state.core.hypothesis.confidence == 0.5
    assert state.core.support_ids == (30,)
    assert state.core.mode is Mode.HOLDING_HAZARD


@pytest.mark.parametrize(
    "role,mode",
    [(0, Mode.QUIESCENT), (2, Mode.QUIESCENT), (3, Mode.QUIESCENT), (4, Mode.QUIESCENT)],
)
def test_predicted_role_and_focus_authoritative(runtime_case, controlled_model, role, mode):
    controlled_model.role = role
    agent, state = prepare(runtime_case, controlled_model)
    state = step(agent, step(agent, state))
    assert state.core.mode is mode
    assert state.core.focus_node_id == (9 if role == 0 else 1)
    assert (state.core.hypothesis is not None) == (role == 2)


def test_guard_outputs_alone_change_act_time(runtime_case, controlled_model):
    agent, state = prepare(runtime_case, controlled_model)
    first = step(agent, step(agent, state))
    time_a = agent.next_internal_event(first).timestamp
    controlled_model.guard_rate *= 2
    agent, state = prepare(runtime_case, controlled_model)
    second = step(agent, step(agent, state))
    assert agent.next_internal_event(second).timestamp < time_a


def test_recall_uses_flowed_context_and_id_ties(runtime_case, controlled_model):
    controlled_model.retrieval_logits = [0.0] * 64
    runtime_case.events = (replace(runtime_case.events[0], event_id=35), *runtime_case.events[1:])
    agent, state = prepare(runtime_case, controlled_model)
    event = agent.next_internal_event(state)
    flowed = state_at(state, event.timestamp)
    state = agent.on_internal(flowed, event)[0]
    context = [c for name, c in controlled_model.calls if name == "recall"][-1]
    assert context.workspace.latent[0, 0].item() == flowed.core.continuous.z_fast[0].item()
    assert state.core.active_record_id == 20


@pytest.mark.parametrize(
    "role,status,disagreement",
    [(1, 0, False), (2, 1, False), (3, 2, False), (1, 1, True), (2, 0, True)],
)
def test_status_head_is_diagnostic_in_canonical_order(
    runtime_case, controlled_model, role, status, disagreement
):
    controlled_model.role, controlled_model.status = role, status
    agent, state = prepare(runtime_case, controlled_model)
    state = step(agent, step(agent, state))
    assert agent.diagnostic_snapshot().status_disagrees is disagreement
    assert (state.core.hypothesis is not None) == (role in (1, 2))


@pytest.mark.parametrize(
    "role,continue_search,append", [(0, True, False), (3, True, False), (4, False, True)]
)
def test_support_and_continue_are_learned(
    runtime_case, controlled_model, role, continue_search, append
):
    controlled_model.role = role
    controlled_model.continue_search = continue_search
    controlled_model.append = append
    agent, state = prepare(runtime_case, controlled_model)
    state = step(agent, step(agent, state))
    assert state.core.support_ids == ((30,) if append else ())
    assert state.core.mode is (Mode.SEARCHING if continue_search else Mode.QUIESCENT)
    assert state.core.active_record_id is None
    assert state.core.memory.lookup(30).consumed


@pytest.mark.parametrize("variant", ["safe", "disconnected"])
def test_explicit_negative_public_variants(runtime_case, controlled_model, variant):
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent

    runtime_case.events = getattr(runtime_case, variant + "_events")
    controlled_model.role = 2
    controlled_model.retrieval_logits = [0.0, 0.0, 0.0, 9.0] + [0.0] * 60
    if variant == "disconnected":
        controlled_model.dormant = True
    result = runtime_case.run(
        NeuralEventFlowAgent(controlled_model, identity=runtime_case.identity, device="cpu"),
        negative=True,
    )
    assert result.actions == ()
    assert result.score.timed_success
