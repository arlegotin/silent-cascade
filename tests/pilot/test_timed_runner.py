import copy
import json
from dataclasses import replace

import pytest
import torch

from .test_timed_metrics import make_identity


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_autonomous_evaluation_preserves_next_training_update(
    neural_archive_case, tmp_path, device
):
    from silent_cascade.eval.runner import evaluate_episodes

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    case = neural_archive_case
    model = case.model.train().requires_grad_(True)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001)
    identity = make_identity(case, (case.bundle,), tmp_path)
    result = evaluate_episodes(
        model,
        identity=identity,
        config=case.config,
        episodes=iter([case.bundle]),
        output_dir=tmp_path / "evaluation",
        device=device,
    )
    assert result.metrics.episode_count == 1
    assert result.metrics.foundation_model_calls == 0
    assert model.training and all(p.requires_grad for p in model.parameters())
    assert all(p.device.type == "cpu" for p in model.parameters())
    for candidate, opt in ((model, optimizer), (reference, reference_optimizer)):
        loss = candidate.controller(
            candidate.initial_context(1, device="cpu"),
            candidate._active_preview(candidate.initial_context(1, device="cpu")),
        )
        loss.flow_targets.sum().backward()
        opt.step()
    assert all(
        torch.equal(a, b) for a, b in zip(model.parameters(), reference.parameters(), strict=True)
    )
    row = json.loads((result.output_path / "rows.jsonl").read_text().splitlines()[0])
    assert row["checkpoint_sha256"] == identity.checkpoint_sha256
    assert row["model_state_sha256"] == case.identity.model_state_sha256
    assert row["producing_source_revision"] == "a" * 40
    assert row["execution_source_revision"] == "b" * 40
    assert row["compute"]["parameters"] > 0
    assert (result.output_path / "DONE").is_file()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "substitution"])
def test_runner_rejects_inventory_mismatch(neural_archive_case, tmp_path, mutation):
    from silent_cascade.eval.runner import evaluate_episodes

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    episodes = [] if mutation == "missing" else [case.bundle, case.bundle]
    if mutation == "substitution":
        episodes = [
            replace(
                case.bundle,
                public=replace(
                    case.bundle.public,
                    init=replace(
                        case.bundle.public.init,
                        episode_public_id="00000000-0000-4000-8000-000000000002",
                    ),
                ),
            )
        ]
    with pytest.raises(ValueError, match="inventory"):
        evaluate_episodes(
            case.model,
            identity=identity,
            config=case.config,
            episodes=episodes,
            output_dir=tmp_path / "bad",
            device="cpu",
        )
    assert not (tmp_path / "bad/DONE").exists()


def test_enabled_mps_fallback_rejected_before_execution(neural_archive_case, tmp_path, monkeypatch):
    from silent_cascade.eval.runner import evaluate_episodes

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(ValueError, match="fallback"):
        evaluate_episodes(
            case.model,
            identity=identity,
            config=case.config,
            episodes=[case.bundle],
            output_dir=tmp_path / "bad",
            device="cpu",
        )


def test_crash_candidate_keeps_all_denominators_and_shared_weights(
    neural_archive_case, tmp_path, monkeypatch
):
    from silent_cascade.eval.runner import evaluate_episodes
    from silent_cascade.models.errors import NeuralError
    from silent_cascade.models.event_flow import EventFlowModel

    case = neural_archive_case
    second = replace(
        case.bundle,
        public=replace(
            case.bundle.public,
            init=replace(
                case.bundle.public.init, episode_public_id="00000000-0000-4000-8000-000000000002"
            ),
        ),
    )
    identity = make_identity(case, (case.bundle, second), tmp_path)
    original = EventFlowModel.compose
    calls = []

    def fail_once(self, context):
        calls.append(context)
        if len(calls) == 1:
            raise NeuralError("candidate output corrupted")
        return original(self, context)

    monkeypatch.setattr(EventFlowModel, "compose", fail_once)
    result = evaluate_episodes(
        case.model,
        identity=identity,
        config=case.config,
        episodes=[case.bundle, second],
        output_dir=tmp_path / "eval",
        device="cpu",
    )
    assert result.metrics.episode_count == 2
    assert result.metrics.error_count == 1
    assert not result.metrics.validation_valid
    assert not result.metrics.gate_passed
    rows = [
        json.loads(line) for line in (result.output_path / "rows.jsonl").read_text().splitlines()
    ]
    assert rows[0]["error"]["code"] == "dynamics_error"
    assert rows[1]["error"] is None
    assert rows[0]["full_trace_ref"]
    crashes = json.loads((result.output_path / "crashes/index.json").read_text())
    assert set(crashes) == {case.bundle.public.init.episode_public_id}
    assert len(list((result.output_path / "crashes").glob("weights-*.safetensors"))) == 1


def test_crash_publication_failure_is_not_a_completed_evaluation(
    neural_archive_case, tmp_path, monkeypatch
):
    from silent_cascade.errors import CrashBundleError
    from silent_cascade.eval.runner import evaluate_episodes
    from silent_cascade.eventflow import engine
    from silent_cascade.models.errors import NeuralError
    from silent_cascade.models.event_flow import EventFlowModel

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)

    def fail(self, context):
        raise NeuralError("bad output")

    def publication_failure(*args, **kwargs):
        raise CrashBundleError("disk unavailable")

    monkeypatch.setattr(EventFlowModel, "compose", fail)
    monkeypatch.setattr(engine, "write_crash_bundle", publication_failure)
    with pytest.raises(CrashBundleError):
        evaluate_episodes(
            case.model,
            identity=identity,
            config=case.config,
            episodes=[case.bundle],
            output_dir=tmp_path / "bad",
            device="cpu",
        )
    assert not (tmp_path / "bad/DONE").exists()


def test_evaluator_spy_observes_only_public_causal_inputs(
    neural_archive_case, tmp_path, monkeypatch
):
    import socket

    from silent_cascade.eval.runner import evaluate_episodes
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent
    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.models.types import ExternalFeatures, ModelContext
    from silent_cascade.schemas import AgentInit, ExternalEventKind

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    observed = []
    initialize = NeuralEventFlowAgent.initialize
    external = NeuralEventFlowAgent.on_external

    def public_init(self, init):
        assert type(init) is AgentInit
        observed.append("initialize")
        return initialize(self, init)

    def public_event(self, state, event):
        assert event.kind in {ExternalEventKind.FACT, ExternalEventKind.ACTIVATE}
        assert event.timestamp == state.time
        observed.append(event.event_id)
        return external(self, state, event)

    def forbid(*args, **kwargs):
        pytest.fail("offline evaluation attempted network access")

    monkeypatch.setattr(socket, "socket", forbid)
    monkeypatch.setattr(NeuralEventFlowAgent, "initialize", public_init)
    monkeypatch.setattr(NeuralEventFlowAgent, "on_external", public_event)
    for name in ("observe", "preview_and_control", "recall_scores", "compose", "action"):
        original = getattr(EventFlowModel, name)

        def spy(self, context, *args, _original=original):
            assert type(context) is ModelContext
            assert all(type(item) is ExternalFeatures for item in args)
            assert context.batch_size == 1
            return _original(self, context, *args)

        monkeypatch.setattr(EventFlowModel, name, spy)
    result = evaluate_episodes(
        case.model,
        identity=identity,
        config=case.config,
        episodes=[case.bundle],
        output_dir=tmp_path / "eval",
        device="cpu",
    )
    assert observed == ["initialize", *(e.event_id for e in case.bundle.public.events)]
    assert result.metrics.foundation_model_calls == 0


def test_runner_streams_before_requesting_next_episode_and_reuses_runtime(
    neural_archive_case, tmp_path, monkeypatch
):
    from silent_cascade.eval.runner import evaluate_episodes
    from silent_cascade.eventflow.engine import EventEngine
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent

    case = neural_archive_case
    second = replace(
        case.bundle,
        public=replace(
            case.bundle.public,
            init=replace(
                case.bundle.public.init, episode_public_id="00000000-0000-4000-8000-000000000002"
            ),
        ),
    )
    identity = make_identity(case, (case.bundle, second), tmp_path)
    constructions = []
    for runtime in (EventEngine, NeuralEventFlowAgent):
        original = runtime.__init__

        def initialize(self, *args, _original=original, **kwargs):
            constructions.append(type(self).__name__)
            _original(self, *args, **kwargs)

        monkeypatch.setattr(runtime, "__init__", initialize)

    def episodes():
        yield case.bundle
        pending = tmp_path / "eval/.rows.pending.jsonl"
        assert len(pending.read_text().splitlines()) == 1
        assert (tmp_path / "eval/episodes/00000.trajectory.json.gz").exists()
        yield second

    result = evaluate_episodes(
        case.model,
        identity=identity,
        config=case.config,
        episodes=episodes(),
        output_dir=tmp_path / "eval",
        device="cpu",
    )
    assert constructions == ["NeuralEventFlowAgent", "EventEngine"]
    assert result.metrics.episode_count == 2
