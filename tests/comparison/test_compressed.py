"""Autonomous zero-time cognitive intervention against the real neural jumps."""

import gzip
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from silent_cascade.errors import DynamicsError
from silent_cascade.eval.comparison_data import build_manifest, diagnostic_allocations, iter_bundles
from silent_cascade.eval.comparison_runner import _run_compressed_episode, run_comparison
from silent_cascade.eval.comparison_types import BudgetLedger, ComparisonConfig, ComparisonIdentity
from silent_cascade.eventflow.compressed import run_compressed_from_activation
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.neural import NeuralEventFlowAgent, NeuralModelIdentity
from silent_cascade.eventflow.state import RuntimeState
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.trace import tensor_sha256
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.schemas import (
    ActivationPayload,
    ExternalEvent,
    ExternalEventKind,
    InternalEvent,
    InternalEventKind,
    LinkFact,
    Mode,
)
from silent_cascade.train.pilot_config import resolve_pilot_config

_PILOT_FIXTURES = Path(__file__).parents[1] / "pilot/conftest.py"
_PILOT_SPEC = importlib.util.spec_from_file_location("comparison_pilot_fixtures", _PILOT_FIXTURES)
assert _PILOT_SPEC is not None and _PILOT_SPEC.loader is not None
_PILOT_MODULE = importlib.util.module_from_spec(_PILOT_SPEC)
_PILOT_SPEC.loader.exec_module(_PILOT_MODULE)
ControlledModel = _PILOT_MODULE.ControlledModel
RuntimeCase = _PILOT_MODULE.RuntimeCase


@pytest.fixture
def activation():
    config = resolve_pilot_config("phase4_pilot").config
    torch.manual_seed(11)
    model = ControlledModel(config.neural).eval()
    case = RuntimeCase(model, config.event_flow)
    agent = NeuralEventFlowAgent(model, identity=case.identity, device="cpu")
    return model, agent, _activation_state(agent, case)


def _activation_state(agent, case):
    engine = EventEngine(case.engine_config)
    state = agent.initialize(case.init)
    for event in case.events:
        state = engine.advance_to(state, event.timestamp)
        state = agent.on_external(state, event)
    assert state.core.mode is Mode.SEARCHING
    return state


def _latent_hashes(state: RuntimeState) -> tuple[str, ...]:
    return tuple(
        tensor_sha256(getattr(state.core.continuous, name))
        for name in (
            "z_fast",
            "z_slow",
            "drives",
            "focus_key",
            "hypothesis_latent",
            "guard_accumulators",
        )
    )


def test_compressed_recomputes_choices_at_activation(activation) -> None:
    """The second recall must follow new state and consumed records, not an intact trace."""
    model, agent, state = activation
    model.role = 0  # continue with predicted LINK compositions
    model.continue_search = True
    decision = run_compressed_from_activation(agent, state, transition_cap=4)
    assert [step.kind for step in decision.steps] == [
        InternalEventKind.RECALL,
        InternalEventKind.COMPOSE,
        InternalEventKind.RECALL,
        InternalEventKind.COMPOSE,
    ]
    assert all(step.executed_at == state.time for step in decision.steps)
    assert all(len(step.state_sha256) == 64 for step in decision.steps)
    recalls = [step for step in decision.steps if step.kind is InternalEventKind.RECALL]
    assert recalls[0].selected_record_id != recalls[1].selected_record_id
    assert decision.final_state.time == state.time
    with pytest.raises(TypeError):
        run_compressed_from_activation(agent, state, intact_trace=())


def test_only_temporal_refractory_is_bypassed(activation) -> None:
    """Repeated cognition logs the exception while preserving consumed-record legality."""
    model, agent, state = activation
    model.role = 0
    model.continue_search = True
    decision = run_compressed_from_activation(agent, state, transition_cap=4)
    assert len(decision.steps) == 4
    assert any(step.bypassed_refractory_until is not None for step in decision.steps)
    selected = [
        step.selected_record_id for step in decision.steps if step.kind is InternalEventKind.RECALL
    ]
    assert len(selected) == len(set(selected))
    one_jump = run_compressed_from_activation(agent, state, transition_cap=1)
    recalled_slot = one_jump.final_state.core.memory.lookup(selected[0])
    assert recalled_slot.refractory_until > state.time
    assert all(
        slot.record.provenance.value == "perceived"
        for slot in decision.final_state.core.memory.records
    )
    invalid = replace(state, core=replace(state.core, mode=Mode.OBSERVING))
    with pytest.raises(DynamicsError, match="runtime invariant"):
        run_compressed_from_activation(agent, invalid)


def test_compressed_rejects_past_or_nonfinite_prediction(activation, monkeypatch) -> None:
    """A broken learned guard cannot schedule a negative or NaN intervention time."""
    _, agent, state = activation
    past = InternalEvent(
        1 << 62, state.core.last_event_id, state.time - 0.1, InternalEventKind.RECALL, 0, 0.0
    )
    monkeypatch.setattr(agent, "next_internal_event", lambda _state: past)
    with pytest.raises(ValueError, match="invalid time"):
        run_compressed_from_activation(agent, state)
    malformed = object.__new__(InternalEvent)
    for name, value in (
        ("event_id", 1 << 62),
        ("parent_event_id", state.core.last_event_id),
        ("timestamp", float("nan")),
        ("kind", InternalEventKind.RECALL),
        ("guard_index", 0),
        ("predicted_delta", 0.0),
    ):
        object.__setattr__(malformed, name, value)
    monkeypatch.setattr(agent, "next_internal_event", lambda _state: malformed)
    with pytest.raises(ValueError, match="invalid time"):
        run_compressed_from_activation(agent, state)


def test_compressed_cap_dormancy_and_future_act(activation) -> None:
    """A cap never grants a 25th jump, while a discovered ACT retains future time."""
    model, agent, state = activation
    model.role = 1  # HAZARD after the first recall
    decision = run_compressed_from_activation(agent, state, transition_cap=2)
    assert len(decision.steps) == 2
    assert decision.predicted_act is not None
    assert decision.predicted_act.timestamp > state.time
    assert decision.stop_reason == "act_scheduled"
    early = run_compressed_from_activation(agent, decision.final_state)
    assert early.steps == ()
    assert early.predicted_act == decision.predicted_act
    model.dormant = True
    case = RuntimeCase(model, resolve_pilot_config("phase4_pilot").config.event_flow)
    dormant_state = _activation_state(agent, case)
    dormant = run_compressed_from_activation(agent, dormant_state)
    assert dormant.steps == ()
    assert dormant.predicted_act is None
    assert dormant.stop_reason == "dormant"


def test_compressed_never_executes_a_twenty_fifth_cognitive_jump() -> None:
    """A plentiful public memory still stops at the predeclared 24-jump cap."""
    config = resolve_pilot_config("phase4_pilot").config
    torch.manual_seed(11)
    model = ControlledModel(config.neural).eval()
    model.role = 0
    model.continue_search = True
    case = RuntimeCase(model, config.event_flow)
    case.events = (
        *(
            ExternalEvent(i, float(i + 1), ExternalEventKind.FACT, LinkFact(i, i + 1))
            for i in range(30)
        ),
        ExternalEvent(30, 31.0, ExternalEventKind.ACTIVATE, ActivationPayload(0)),
    )
    agent = NeuralEventFlowAgent(model, identity=case.identity, device="cpu")
    state = _activation_state(agent, case)
    decision = run_compressed_from_activation(agent, state)
    assert len(decision.steps) == 24
    assert decision.final_state.core.executed_internal_events == 24
    assert decision.stop_reason == "cap_reached"
    assert decision.predicted_act is None

    class FinalHazardModel(ControlledModel):
        compositions = 0

        def compose(self, context):
            self.compositions += 1
            self.role = 1 if self.compositions == 12 else 0
            return super().compose(context)

    torch.manual_seed(11)
    final_model = FinalHazardModel(config.neural).eval()
    final_model.continue_search = True
    final_case = RuntimeCase(final_model, config.event_flow)
    final_case.events = case.events
    final_agent = NeuralEventFlowAgent(final_model, identity=final_case.identity, device="cpu")
    final_state = _activation_state(final_agent, final_case)
    final_decision = run_compressed_from_activation(final_agent, final_state)
    assert len(final_decision.steps) == 24
    assert final_decision.predicted_act is not None
    assert final_decision.predicted_act.timestamp > final_state.time
    assert final_decision.stop_reason == "act_scheduled"


def test_compressed_keeps_weights_and_post_activation_input_immutable(activation) -> None:
    """The intervention may mutate only its detached trajectory and agent diagnostics."""
    model, agent, state = activation
    initial_weights = NeuralModelIdentity.from_model(
        model, source_revision="a" * 40
    ).model_state_sha256
    initial_latents = _latent_hashes(state)
    initial_counters = state.core.counters
    decision = run_compressed_from_activation(agent, state, transition_cap=4)
    assert decision.final_state is not state
    assert state.core.counters == initial_counters
    assert _latent_hashes(state) == initial_latents
    assert (
        NeuralModelIdentity.from_model(model, source_revision="a" * 40).model_state_sha256
        == initial_weights
    )
    assert decision.compute.foundation_model_calls == 0


def test_compressed_runner_keeps_complete_paired_rows(tmp_path) -> None:
    """The evaluator owns scoring and counts all diagnostic episodes for compression."""
    config = ComparisonConfig()
    allocation = diagnostic_allocations()["iid"]
    first = allocation.blocks[0].model_copy(update={"episode_count": 4})
    manifest = build_manifest(config, "iid", allocation.model_copy(update={"blocks": (first,)}))
    torch.manual_seed(11)
    model = EventFlowModel(config.phase4_config.neural).eval()
    identity = ComparisonIdentity(
        condition="compressed_eventflow",
        protocol_sha256=config.protocol_sha256,
        config_sha256=config.config_sha256,
        generator_sha256=config.generator_sha256,
        manifest_sha256=sha256_bytes(canonical_json_bytes(manifest)),
        checkpoint_sha256="1" * 64,
        model_state_sha256=NeuralModelIdentity.from_model(
            model, source_revision="1" * 40
        ).model_state_sha256,
        producing_source_revision="1" * 40,
        execution_source_revision="2" * 40,
    )
    output = run_comparison(
        config=config,
        manifest=manifest,
        identity=identity,
        model=model,
        output_dir=tmp_path / "compressed",
        budget=BudgetLedger(),
    )
    with gzip.open(output / "rows.jsonl.gz", "rt") as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 4
    assert all(row["error"] is None for row in rows)
    assert all(row["result"]["end_to_end_compute"]["foundation_model_calls"] == 0 for row in rows)
    assert all(
        row["result"]["end_to_end_compute"]["forward_macs"]
        > row["result"]["post_activation_compute"]["forward_macs"]
        for row in rows
    )


def test_future_act_uses_learned_head_after_analytic_advance() -> None:
    """The scheduled ACT executes the existing learned action head at its future time."""
    config = ComparisonConfig()
    allocation = diagnostic_allocations()["iid"]
    first = allocation.blocks[0].model_copy(update={"episode_count": 4})
    manifest = build_manifest(config, "iid", allocation.model_copy(update={"blocks": (first,)}))
    bundle = next(iter_bundles(manifest, config))
    torch.manual_seed(11)
    model = ControlledModel(config.phase4_config.neural).eval()
    model.role = 1
    result = _run_compressed_episode(bundle, model=model, config=config)
    assert result.error is None
    assert result.stop_reason == "act_scheduled"
    assert len(result.actions) == 1
    assert result.actions[0].timestamp > bundle.truth.activation_time
    assert result.actions[0].timestamp < bundle.truth.private_terminal.timestamp
    assert "act" in [step.kind for step in result.steps]
    assert any(name == "action" for name, _ in model.calls)


def test_compressed_artifact_retains_refractory_bypasses() -> None:
    """The reportable row must retain each exception and mathematical crossing time."""
    config = ComparisonConfig()
    allocation = diagnostic_allocations()["iid"]
    first = allocation.blocks[0].model_copy(update={"episode_count": 4})
    manifest = build_manifest(config, "iid", allocation.model_copy(update={"blocks": (first,)}))
    bundle = next(iter_bundles(manifest, config))
    torch.manual_seed(11)
    model = ControlledModel(config.phase4_config.neural).eval()
    model.role = 0
    model.continue_search = True
    result = _run_compressed_episode(bundle, model=model, config=config)
    assert result.error is None
    bypassed = [step for step in result.steps if step.bypassed_refractory_until is not None]
    assert bypassed
    assert all(step.predicted_crossing_at > bundle.truth.activation_time for step in bypassed)


def test_compressed_failure_charges_executed_post_activation_work() -> None:
    """A failed second jump must retain measured work from the completed first jump."""

    class FailingComposeModel(ControlledModel):
        def compose(self, context):
            raise RuntimeError("controlled compose failure")

    config = ComparisonConfig()
    allocation = diagnostic_allocations()["iid"]
    first = allocation.blocks[0].model_copy(update={"episode_count": 4})
    manifest = build_manifest(config, "iid", allocation.model_copy(update={"blocks": (first,)}))
    bundle = next(iter_bundles(manifest, config))
    torch.manual_seed(11)
    model = FailingComposeModel(config.phase4_config.neural).eval()
    result = _run_compressed_episode(bundle, model=model, config=config)
    assert result.error is not None
    assert result.post_activation_compute.forward_macs > 0
    assert result.end_to_end_compute.forward_macs > result.post_activation_compute.forward_macs
