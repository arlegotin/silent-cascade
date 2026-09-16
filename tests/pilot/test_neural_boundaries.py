from dataclasses import fields

import pytest
import torch

from silent_cascade.models.errors import NeuralError

from .test_neural_agent import prepare, state_at, step


def test_no_initial_forward_and_no_pre_activation_scores(runtime_case, controlled_model):
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent

    agent = NeuralEventFlowAgent(controlled_model, identity=runtime_case.identity, device="cpu")
    scores = []
    handle = controlled_model.scorer.register_forward_hook(lambda *args: scores.append(1))
    state = agent.initialize(runtime_case.init)
    assert controlled_model.calls == []
    later = state_at(state, 1.0)
    assert torch.equal(state.core.continuous.z_fast, later.core.continuous.z_fast)
    for event in runtime_case.events[:-1]:
        state = agent.on_external(state_at(state, event.timestamp), event)
        assert agent.next_internal_event(state) is None
    handle.remove()
    assert scores == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_predictions_fail_typed(runtime_case, controlled_model, bad):
    controlled_model.action_logits[0] = bad
    agent, state = prepare(runtime_case, controlled_model)
    state = step(agent, step(agent, state))
    with pytest.raises(NeuralError):
        step(agent, state)


def test_empty_recall_fails_typed(runtime_case, controlled_model):
    runtime_case.events = runtime_case.events[-1:]
    agent, state = prepare(runtime_case, controlled_model)
    with pytest.raises(NeuralError, match="empty"):
        step(agent, state)


def test_private_truth_does_not_change_prediction_prefix(runtime_case, controlled_model):
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent

    agent = NeuralEventFlowAgent(controlled_model, identity=runtime_case.identity, device="cpu")
    first = runtime_case.run(agent)
    snapshot = agent.diagnostic_snapshot()
    second = runtime_case.run(agent, negative=True)
    assert first.actions == second.actions
    assert snapshot == agent.diagnostic_snapshot()
    for item in fields(snapshot):
        assert not isinstance(getattr(snapshot, item.name), torch.Tensor)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_first_fact_after_positive_gap_matches_teacher(runtime_case, controlled_model, device):
    from silent_cascade.env.episode import PublicEpisode
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent, NeuralModelIdentity
    from silent_cascade.train.batches import pack_public_examples
    from silent_cascade.train.observations import observe_public

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    model = controlled_model.to(device)
    identity = NeuralModelIdentity.from_model(model, source_revision="a" * 40)
    agent = NeuralEventFlowAgent(model, identity=identity, device=device)
    state = agent.initialize(runtime_case.init)
    event = runtime_case.events[0]
    assert event.timestamp - runtime_case.init.initial_time == 1.0
    agent.on_external(state_at(state, event.timestamp), event)
    actual = model.calls[-1][1]
    batch = pack_public_examples((PublicEpisode(runtime_case.init, runtime_case.events),)).to(
        device
    )
    with torch.no_grad():
        expected = observe_public(model, batch)[3][0].context
    for field in fields(actual):
        left, right = getattr(actual, field.name), getattr(expected, field.name)
        if field.name == "workspace":
            torch.testing.assert_close(left.latent, right.latent, rtol=0, atol=0)
            torch.testing.assert_close(left.accumulators, right.accumulators, rtol=0, atol=0)
        else:
            torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["recall", "compose", "action", "control"])
def test_invalid_schema_and_nonfinite_heads_preserve_typed_cause(
    runtime_case, controlled_model, monkeypatch, kind
):
    agent, state = prepare(runtime_case, controlled_model)
    if kind in {"compose", "action"}:
        state = step(agent, state)
    if kind == "action":
        state = step(agent, state)
    name = {"recall": "recall_scores", "control": "preview_and_control"}.get(kind, kind)
    if kind == "control":
        original = getattr(controlled_model, name)

        def malformed(context):
            preview, parameters = original(context)
            # Deliberately corrupt an otherwise valid public output container.
            parameters.flow_targets[0, 0] = torch.nan
            return preview, parameters
    else:

        def malformed(context):
            return object()

    monkeypatch.setattr(controlled_model, name, malformed)
    with pytest.raises(NeuralError):
        step(agent, state)


def test_learned_callbacks_never_use_scripted_impulses(runtime_case, controlled_model, monkeypatch):
    from silent_cascade.eventflow import jumps
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent

    def forbidden(*args):
        pytest.fail("scripted impulse in learned policy")

    monkeypatch.setattr(jumps, "_impulse", forbidden)
    result = runtime_case.run(
        NeuralEventFlowAgent(controlled_model, identity=runtime_case.identity, device="cpu")
    )
    assert len(result.actions) == 1
    assert result.counters.controller_calls == 8
    assert result.counters.records_scored == 320
    assert result.counters.jump_applications == 9


def test_agent_import_has_no_private_or_training_dependencies():
    import subprocess
    import sys

    program = """
import importlib.abc
import sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('silent_cascade.train') or (
            fullname.startswith('silent_cascade.env.') and fullname != 'silent_cascade.env.config'
        ):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Blocker())
import silent_cascade.eventflow.neural
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_callback_accepts_protocol_keyword_arguments(runtime_case, controlled_model):
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent

    agent = NeuralEventFlowAgent(controlled_model, identity=runtime_case.identity, device="cpu")
    state = agent.initialize(init=runtime_case.init)
    event = runtime_case.events[0]
    result = agent.on_external(state=state_at(state, event.timestamp), event=event)
    assert result.core.memory.records[0].record.record_id == 10


def test_neural_diagnostics_count_module_calls_not_encoded_rows(runtime_case, controlled_model):
    agent, state = prepare(runtime_case, controlled_model)
    state = step(agent, state)
    snapshot = agent.diagnostic_snapshot()
    assert dict(snapshot.module_calls)["record_encoder"] == 2
    assert snapshot.record_rows_encoded == 128
    assert snapshot.records_scored == 128


@pytest.mark.parametrize("kind", ["recall", "compose"])
def test_nonfinite_cognitive_predictions_rejected(
    runtime_case, controlled_model, monkeypatch, kind
):
    from dataclasses import replace

    agent, state = prepare(runtime_case, controlled_model)
    if kind == "compose":
        state = step(agent, state)
    name = "recall_scores" if kind == "recall" else "compose"
    original = getattr(controlled_model, name)

    def bad(context):
        value = original(context)
        field = "raw_scores" if kind == "recall" else "normalized_deadline"
        tensor = getattr(value, field).clone()
        tensor.flatten()[0] = torch.nan
        return replace(value, **{field: tensor})

    monkeypatch.setattr(controlled_model, name, bad)
    with pytest.raises(NeuralError, match="finite"):
        step(agent, state)


def test_malformed_preview_boolean_is_rejected(runtime_case, controlled_model, monkeypatch):
    from dataclasses import replace

    agent, state = prepare(runtime_case, controlled_model)
    original = controlled_model.preview_and_control

    def malformed(context):
        preview, parameters = original(context)
        return replace(preview, has_candidate=torch.tensor([float("nan")])), parameters

    monkeypatch.setattr(controlled_model, "preview_and_control", malformed)
    with pytest.raises(NeuralError):
        step(agent, state)
