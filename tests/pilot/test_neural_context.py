"""Single-row runtime bridges and explicit learned-state metadata jumps."""

from dataclasses import fields, replace

import pytest
import torch

from silent_cascade.errors import DynamicsError
from silent_cascade.eventflow.state import ContinuousChannels, make_initial_continuous_state
from silent_cascade.models.errors import NeuralError
from silent_cascade.models.types import TensorWorkspace
from silent_cascade.schemas import ActivationPayload, ExternalEvent, ExternalEventKind, LinkFact


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_runtime_tensor_roundtrip_preserves_every_channel(device):
    from silent_cascade.eventflow.neural_context import (
        continuous_from_workspace,
        workspace_from_continuous,
    )

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    state = make_initial_continuous_state(device=device)
    state = replace(
        state,
        **{
            f.name: torch.linspace(-0.8, 0.8, getattr(state, f.name).numel(), device=device)
            for f in fields(ContinuousChannels)
        },
        guard_accumulators=torch.tensor([0.2, 0.9, 1.5], device=device),
    )
    workspace = workspace_from_continuous(state)
    restored = continuous_from_workspace(workspace)
    for field in fields(state):
        assert torch.equal(getattr(state, field.name), getattr(restored, field.name))
        assert getattr(state, field.name).data_ptr() != getattr(restored, field.name).data_ptr()
    workspace.latent.zero_()
    assert restored.z_fast.abs().sum() > 0


def test_conversion_rejects_wrong_rows_and_mutated_nan():
    from silent_cascade.eventflow.neural_context import (
        continuous_from_workspace,
        workspace_from_continuous,
    )

    with pytest.raises(NeuralError, match=r"one|single"):
        continuous_from_workspace(TensorWorkspace.zeros(2, "cpu"))
    state = make_initial_continuous_state(device="cpu")
    state.z_fast[0] = torch.nan
    with pytest.raises((NeuralError, DynamicsError), match="finite"):
        workspace_from_continuous(state)
    workspace = TensorWorkspace.zeros(1, "cpu")
    workspace.latent[0, 0] = torch.nan
    with pytest.raises((NeuralError, DynamicsError), match="finite"):
        continuous_from_workspace(workspace)


def test_device_mismatch_is_rejected(model, jump_fixtures):
    from silent_cascade.eventflow.jumps import apply_activate
    from silent_cascade.eventflow.neural_context import (
        context_from_runtime,
        workspace_from_continuous,
    )
    from silent_cascade.schemas import AgentInit

    if not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    state = jump_fixtures.runtime()
    with pytest.raises(NeuralError, match="device"):
        context_from_runtime(model.to("mps"), state, AgentInit("mismatch", 64, 4, 0.0))
    replacement = make_initial_continuous_state(device="mps")
    with pytest.raises(DynamicsError, match="device"):
        apply_activate(
            state,
            ExternalEvent(4, 1.0, ExternalEventKind.ACTIVATE, ActivationPayload(0)),
            continuous=replacement,
        )
    object.__setattr__(replacement, "drives", torch.zeros(8))
    with pytest.raises(DynamicsError, match="device"):
        workspace_from_continuous(replacement)


def test_runtime_context_uses_public_memory_legality_and_preserves_flow(model, jump_fixtures):
    import math

    from silent_cascade.eventflow.neural_context import context_from_runtime
    from silent_cascade.schemas import AgentInit, Mode

    state = jump_fixtures.runtime(Mode.HAVE_MEMORY)
    slots = state.core.memory.records
    memory = replace(state.core.memory, records=(replace(slots[0], consumed=True), slots[1]))
    core = replace(
        state.core,
        memory=memory,
        support_ids=(2,),
        continuous=replace(
            state.core.continuous,
            focus_key=torch.full((64,), 0.37),
            hypothesis_latent=torch.full((64,), -0.29),
        ),
    )
    context = context_from_runtime(model, replace(state, core=core), AgentInit("test", 64, 4, 0.0))
    assert context.eligibility[0, :2].tolist() == [False, True]
    assert context.eligibility.sum() == 1  # subject 1 differs from focus 0
    assert context.support_mask[0].nonzero().flatten().tolist() == [1]
    assert context.active_slot_indices.item() == 0
    assert context.modes.item() == 2
    assert torch.equal(context.workspace.latent[0, 328:392], core.continuous.focus_key)
    assert torch.equal(context.workspace.latent[0, 392:], core.continuous.hypothesis_latent)
    torch.testing.assert_close(
        context.time_features, torch.tensor([[math.log1p(1), math.log1p(0.5)]])
    )
    for changes in (
        {"valid": False},
        {"refractory_until": 2.0},
        {"record": replace(slots[1].record, observed_at=2.0)},
    ):
        altered = replace(
            core, memory=replace(memory, records=(memory.records[0], replace(slots[1], **changes)))
        )
        result = context_from_runtime(
            model, replace(state, core=altered), AgentInit("test", 64, 4, 0.0)
        )
        assert not result.eligibility.any()


def test_runtime_time_features_subtract_on_host_and_signed_deadline(model, jump_fixtures):
    import math

    from silent_cascade.eventflow.neural_context import context_from_runtime
    from silent_cascade.schemas import AgentInit, Mode

    state = jump_fixtures.runtime(Mode.HOLDING_HAZARD)
    now = float(2**30) + 0.75
    core = replace(
        state.core,
        activation_time=now - 0.5,
        hypothesis=replace(state.core.hypothesis, deadline=now - 0.25),
    )
    state = replace(state, core=core, time=now)
    result = context_from_runtime(model, state, AgentInit("clock", 64, 4, 0.0))
    assert result.time_features[0, 1].item() == torch.tensor(math.log1p(0.5)).item()
    assert result.hypothesis_features[0, 7].item() == torch.tensor(-math.log1p(0.25)).item()


def test_segment_conversion_is_one_row_nonaliased_and_matches_flow(model):
    from silent_cascade.eventflow.flow import start_segment, state_at
    from silent_cascade.eventflow.neural_context import segment_from_batch
    from silent_cascade.eventflow.state import RuntimeCore
    from silent_cascade.models.dynamics import flow_batch

    context = model.initial_context(1, device="cpu")
    _, parameters = model.preview_and_control(context)
    segment = segment_from_batch(parameters)
    expected = flow_batch(context.workspace, parameters, torch.tensor([0.3])).row(0)
    runtime_state = start_segment(
        RuntimeCore(context.workspace.row(0)),
        segment,
        time=0.0,
        parent_event_id=0,
        prediction_snapshot_sha256="a" * 64,
    )
    actual = state_at(runtime_state, 0.3)
    for field in fields(actual):
        torch.testing.assert_close(
            getattr(actual, field.name), getattr(expected, field.name), rtol=0, atol=0
        )
    assert segment.guard_targets.data_ptr() != parameters.guard_targets.data_ptr()
    with pytest.raises(NeuralError, match=r"one|single"):
        segment_from_batch(model.preview_and_control(model.initial_context(2, device="cpu"))[1])


@pytest.mark.parametrize(
    "kind",
    ["fact", "activate", "recall", "act", "link", "hazard", "safe", "irrelevant", "contradictory"],
)
def test_explicit_continuous_bypasses_all_scripted_impulses(monkeypatch, kind, jump_fixtures):
    from silent_cascade.eventflow import jumps
    from silent_cascade.schemas import InternalEventKind, Mode

    runtime, internal, decision = (
        jump_fixtures.runtime,
        jump_fixtures.internal,
        jump_fixtures.decision,
    )

    def forbidden(*args):
        raise AssertionError("scripted impulse")

    monkeypatch.setattr(jumps, "_impulse", forbidden)
    continuous = replace(
        make_initial_continuous_state(device="cpu"),
        z_fast=torch.full((256,), 0.3),
        guard_accumulators=torch.ones(3),
    )
    if kind in ("fact", "activate"):
        event = ExternalEvent(
            4,
            1.0,
            ExternalEventKind.FACT if kind == "fact" else ExternalEventKind.ACTIVATE,
            LinkFact(0, 2) if kind == "fact" else ActivationPayload(0),
        )

        def call(**kw):
            return getattr(jumps, "apply_" + kind)(runtime(), event, **kw)
    elif kind == "recall":

        def call(**kw):
            return jumps.apply_recall(
                runtime(Mode.SEARCHING), internal(InternalEventKind.RECALL), record_id=1, **kw
            )
    elif kind == "act":

        def call(**kw):
            return jumps.apply_act(
                runtime(Mode.HOLDING_HAZARD), internal(InternalEventKind.ACT), hazard_type=2, **kw
            )
    else:

        def call(**kw):
            return jumps.apply_compose(
                runtime(Mode.HAVE_MEMORY),
                internal(InternalEventKind.COMPOSE),
                decision(jumps.ComposeRole(kind)),
                **kw,
            )

    core = call(continuous=continuous)
    for field in fields(ContinuousChannels):
        assert torch.equal(getattr(core.continuous, field.name), getattr(continuous, field.name))
    assert not core.continuous.guard_accumulators.any()
    assert core.counters.jump_applications == 1
    with pytest.raises(AssertionError, match="scripted impulse"):
        call()
