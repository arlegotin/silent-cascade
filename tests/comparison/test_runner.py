"""Private evaluator and paired result contracts."""

import gzip
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from silent_cascade.env.episode import EpisodeBundle, EpisodeVariant, PublicEpisode
from silent_cascade.env.reward import score_actions
from silent_cascade.eval.comparison_data import build_manifest, diagnostic_allocations, iter_bundles
from silent_cascade.eval.comparison_runner import (
    PublicProposal,
    arbitrate_proposal,
    run_comparison,
    run_intact_episode,
    run_ponder_episode,
    run_public_policy,
)
from silent_cascade.eval.comparison_types import (
    BudgetLedger,
    ComparisonConfig,
    ComparisonIdentity,
    ComparisonRow,
)
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.neural import NeuralEventFlowAgent, NeuralModelIdentity
from silent_cascade.eventflow.state import ComputeCounters
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.models.activation_ponder import ActivationPonderModel, matched_ponder_config
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.schemas import Action, Mode


@pytest.fixture(scope="module")
def sample():
    config = ComparisonConfig()
    allocation = diagnostic_allocations()["iid"]
    first = allocation.blocks[0].model_copy(update={"episode_count": 4})
    tiny = allocation.model_copy(update={"blocks": (first,)})
    bundles = tuple(iter_bundles(build_manifest(config, "iid", tiny), config))
    return config, bundles


def test_policy_receives_only_public_inputs(sample) -> None:
    """A public condition must not see target, recipe, or private horizon data."""
    _, bundles = sample
    negative = next(b for b in bundles if b.truth.recipe.variant is EpisodeVariant.SAFE_NEGATIVE)
    calls = []

    def spy(public: PublicEpisode) -> PublicProposal:
        assert type(public) is PublicEpisode
        assert not hasattr(public, "truth")
        calls.append(public)
        return PublicProposal(actions=())

    original = run_public_policy(negative, spy)
    changed_truth = replace(
        negative.truth,
        private_terminal=replace(
            negative.truth.private_terminal,
            timestamp=negative.truth.private_terminal.timestamp + 1.0,
        ),
        episode_delay=negative.truth.episode_delay + 1.0,
    )
    shifted = run_public_policy(EpisodeBundle(negative.public, changed_truth), spy)
    assert calls == [negative.public, negative.public]
    assert original.actions == shifted.actions == ()


def test_terminal_first_late_duplicate_and_negative_scoring(sample) -> None:
    """Public proposals are emitted only before the private terminal, then scored exactly."""
    _, bundles = sample
    positive = next(b for b in bundles if b.truth.recipe.variant is EpisodeVariant.POSITIVE)
    negative = next(b for b in bundles if b.truth.recipe.variant is EpisodeVariant.SAFE_NEGATIVE)
    t = positive.truth.private_terminal.timestamp
    hazard = positive.truth.relevant_hazard_type
    assert hazard is not None
    tied = arbitrate_proposal(positive, PublicProposal(actions=(Action(hazard, t, 99),)))
    assert tied.actions == ()
    assert not score_actions(positive.truth, tied.actions).timed_success
    late = arbitrate_proposal(
        positive, PublicProposal(actions=(Action(hazard, positive.truth.action_window_end, 99),))
    )
    assert len(late.actions) == 1
    assert not score_actions(positive.truth, late.actions).timed_success
    repeated = arbitrate_proposal(
        positive,
        PublicProposal(
            actions=(
                Action(hazard, positive.truth.action_target, 99),
                Action(hazard, positive.truth.action_target + 0.01, 100),
            )
        ),
    )
    assert score_actions(positive.truth, repeated.actions).reason == "action_count"
    assert score_actions(
        negative.truth, arbitrate_proposal(negative, PublicProposal()).actions
    ).timed_success


def test_comparison_bridge_matches_intact_engine(sample) -> None:
    """The new bridge must execute the same autonomous agent and causal engine."""
    config, bundles = sample
    bundle = bundles[0]
    torch.manual_seed(11)
    model = EventFlowModel(config.phase4_config.neural).eval()
    source = "1" * 40
    agent = NeuralEventFlowAgent(
        model, identity=NeuralModelIdentity.from_model(model, source_revision=source), device="cpu"
    )
    reference = EventEngine(config.phase4_config.event_flow).run_episode(bundle, agent)
    compared = run_intact_episode(bundle, model=model, config=config)
    assert compared.actions == reference.actions
    assert compared.end_to_end_compute.foundation_model_calls == 0
    assert compared.end_to_end_compute.forward_macs > compared.post_activation_compute.forward_macs


def test_intact_post_activation_includes_all_cognitive_modes(sample, monkeypatch) -> None:
    from silent_cascade.eval import comparison_runner

    config, bundles = sample
    model = EventFlowModel(config.phase4_config.neural).eval()
    linear = next(module for module in model.modules() if isinstance(module, torch.nn.Linear))

    class FakeEngine:
        def __init__(self, config):
            pass

        def start_episode(self, bundle, agent):
            return SimpleNamespace(
                state=SimpleNamespace(
                    core=SimpleNamespace(
                        mode=Mode.OBSERVING, counters=ComputeCounters(), actions=()
                    )
                ),
                trace=SimpleNamespace(snapshot=lambda: SimpleNamespace(events=(), sha256="0" * 64)),
                trajectory=None,
            )

        def step(self, session, agent):
            linear(torch.zeros(1, linear.in_features))
            modes = {
                Mode.OBSERVING: Mode.SEARCHING,
                Mode.SEARCHING: Mode.HAVE_MEMORY,
                Mode.HAVE_MEMORY: Mode.HOLDING_HAZARD,
                Mode.HOLDING_HAZARD: Mode.TERMINAL,
            }
            session.state.core.mode = modes[session.state.core.mode]
            return session.state.core.mode is Mode.TERMINAL

        def operational_compute_snapshot(self):
            return SimpleNamespace(causal_flow_evaluations=0)

    monkeypatch.setattr(comparison_runner, "EventEngine", FakeEngine)
    compared = run_intact_episode(bundles[0], model=model, config=config)
    assert compared.end_to_end_compute.module_calls["Linear"] == 4
    assert compared.post_activation_compute.module_calls["Linear"] == 3


def test_ponder_error_retains_executed_prefix_and_compute(sample, monkeypatch) -> None:
    _, bundles = sample
    model = ActivationPonderModel(matched_ponder_config()).eval()
    original = model.transition
    calls = 0

    def fail_after_first(encoded, hidden):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("controlled transition failure")
        output = original(encoded, hidden)
        return replace(
            output,
            heads={**output.heads, "halt": torch.full_like(output.heads["halt"], -100.0)},
        )

    monkeypatch.setattr(model, "transition", fail_after_first)
    observed = []
    result = run_ponder_episode(
        bundles[0], model=model, cap=4, trace_sink=lambda result, trace: observed.append(trace)
    )
    assert result.error is not None
    assert result.stop_reason == "dynamics_error"
    assert len(result.steps) == 1
    assert result.end_to_end_compute.forward_macs > 0
    assert result.post_activation_compute == result.end_to_end_compute
    assert observed and len(observed[0]) == 1


def test_errors_keep_episode_denominators(sample, monkeypatch, tmp_path) -> None:
    """A failed model episode still yields one scored row in the comparison."""
    from silent_cascade.eval import comparison_runner

    config, bundles = sample
    torch.manual_seed(11)
    model = EventFlowModel(config.phase4_config.neural).eval()

    def fail(*args, **kwargs):
        raise RuntimeError("controlled failure")

    monkeypatch.setattr(comparison_runner.EventEngine, "step", fail)
    result = run_intact_episode(bundles[0], model=model, config=config)
    assert result.error is not None
    assert result.actions == ()
    allocation = diagnostic_allocations()["iid"]
    first = allocation.blocks[0].model_copy(update={"episode_count": 4})
    manifest = build_manifest(config, "iid", allocation.model_copy(update={"blocks": (first,)}))
    identity = ComparisonIdentity(
        condition="intact_eventflow",
        manifest_name="iid",
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
    with pytest.raises(ValueError, match="config"):
        run_comparison(
            config=config,
            manifest=manifest,
            identity=identity.model_copy(update={"config_sha256": "0" * 64}),
            model=model,
            output_dir=tmp_path / "rejected",
            budget=BudgetLedger(),
        )
    output = run_comparison(
        config=config,
        manifest=manifest,
        identity=identity,
        model=model,
        output_dir=tmp_path / "comparison",
        budget=BudgetLedger(),
    )
    with gzip.open(output / "rows.jsonl.gz", "rt") as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 4
    assert all(row["condition"] == "intact_eventflow" for row in rows)
    assert all(row["manifest_name"] == "iid" for row in rows)
    assert all(row["protocol_sha256"] == config.protocol_sha256 for row in rows)
    assert all(row["inference_wall_seconds"] >= 0 for row in rows)
    assert all(row["error"] is not None and not row["timed_success"] for row in rows)
    assert all(not ComparisonRow.model_validate_json(json.dumps(row)).timed_success for row in rows)
