import json

import pytest

from silent_cascade.errors import DynamicsError
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.models.errors import NeuralError


def test_runtime_crash_handoff_keeps_shared_weights_until_evaluation_closes(
    neural_archive_case, tmp_path, monkeypatch
):
    from silent_cascade.eval.runner import evaluate_episodes
    from silent_cascade.models.event_flow import EventFlowModel

    from .test_pilot_archive_producer import producer_case
    from .test_timed_metrics import make_identity

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)

    def fail(*args, **kwargs):
        raise NeuralError("runtime crash")

    monkeypatch.setattr(EventFlowModel, "compose", fail)
    with producer_case(tmp_path, monkeypatch) as (producer, session, _server):
        commits = []
        handoff = producer.after_episode

        def after_episode(root, commit):
            handoff(root, commit)
            assert commit.borrowed
            assert any(entry.path.endswith(".safetensors") for entry in commit.owned)
            assert all(not (session.run_dir / entry.path).exists() for entry in commit.owned)
            assert all((session.run_dir / entry.path).exists() for entry in commit.borrowed)
            commits.append(commit)

        monkeypatch.setattr(producer, "after_episode", after_episode)
        result = evaluate_episodes(
            case.model,
            identity=identity,
            config=case.config,
            episodes=[case.bundle],
            output_dir=session.run_dir / "evaluation",
            device="cpu",
            archive_producer=producer,
            evidence_context=session,
        )
        assert result.metrics.episode_count == 1
        assert len(commits) == 1
        assert all(not (session.run_dir / entry.path).exists() for entry in commits[0].borrowed)


def test_neural_failure_stages_before_callback(neural_archive_case, tmp_path, monkeypatch):
    from silent_cascade.eventflow.neural_checkpoint import (
        load_neural_runtime_checkpoint,
        restore_neural_runtime,
    )
    from silent_cascade.hashing import sha256_bytes

    case = neural_archive_case
    engine = EventEngine(
        case.config.event_flow,
        crash_root=tmp_path,
        source_revision=case.revision,
        experiment_config_canonical_json=case.canonical,
    )
    calls = []

    def fail(context):
        calls.append(1)
        raise NeuralError("injected neural failure")

    monkeypatch.setattr(case.model, "compose", fail)
    with pytest.raises(DynamicsError) as caught:
        engine.step(case.session, case.agent)
    assert isinstance(caught.value.__cause__, NeuralError)
    assert len(calls) == 1
    bundle = json.loads(next(tmp_path.glob("*.json")).read_bytes())
    assert bundle["context"]["last_events"]
    assert len(bundle["context"]["last_events"]) <= 20
    assert case.canonical not in json.dumps(bundle)
    assert "injected neural failure" not in json.dumps(bundle)
    path = next(p for p in tmp_path.glob("*.safetensors") if not p.name.startswith("weights-"))
    artifact = load_neural_runtime_checkpoint(
        path,
        expected_sha256=sha256_bytes(path.read_bytes()),
        config=case.config.event_flow,
        source_revision=case.revision,
        device="cpu",
    )
    session, agent = restore_neural_runtime(artifact, device="cpu")
    while not case.engine.step(session, agent):
        pass
    assert session.terminal_score is not None


def test_missing_full_config_prevents_initial_callback(neural_archive_case, tmp_path, monkeypatch):
    from silent_cascade.errors import ReplayError

    case = neural_archive_case
    monkeypatch.setattr(
        case.agent,
        "initialize",
        lambda *args: pytest.fail("private configuration checked too late"),
    )
    engine = EventEngine(case.config.event_flow, crash_root=tmp_path, source_revision=case.revision)
    with pytest.raises(ReplayError):
        engine.start_episode(case.bundle, case.agent)
    assert not list(tmp_path.iterdir())


def test_crash_weights_are_reauthenticated_on_every_restore(
    neural_archive_case, tmp_path, monkeypatch
):
    from silent_cascade.errors import ReplayError
    from silent_cascade.eventflow.neural_checkpoint import (
        load_neural_runtime_checkpoint,
        restore_neural_runtime,
    )
    from silent_cascade.hashing import sha256_bytes

    case = neural_archive_case
    engine = EventEngine(
        case.config.event_flow,
        crash_root=tmp_path,
        source_revision=case.revision,
        experiment_config_canonical_json=case.canonical,
    )

    def fail(context):
        raise NeuralError("failure")

    monkeypatch.setattr(case.model, "compose", fail)
    with pytest.raises(DynamicsError):
        engine.step(case.session, case.agent)
    path = next(p for p in tmp_path.glob("*.safetensors") if not p.name.startswith("weights-"))
    artifact = load_neural_runtime_checkpoint(
        path,
        expected_sha256=sha256_bytes(path.read_bytes()),
        config=case.config.event_flow,
        source_revision=case.revision,
        device="cpu",
    )
    weights = next(tmp_path.glob("weights-*.safetensors"))
    weights.rename(weights.with_suffix(".missing"))
    with pytest.raises(ReplayError):
        restore_neural_runtime(artifact, device="cpu")


def test_callback_weight_mutation_fails_and_restores_original_weights(
    neural_archive_case, tmp_path, monkeypatch
):
    import torch

    from silent_cascade.eventflow.neural_checkpoint import (
        load_neural_runtime_checkpoint,
        restore_neural_runtime,
    )
    from silent_cascade.hashing import sha256_bytes

    case = neural_archive_case
    engine = EventEngine(
        case.config.event_flow,
        crash_root=tmp_path,
        source_revision=case.revision,
        experiment_config_canonical_json=case.canonical,
    )
    original = case.model.compose

    def mutate(context):
        result = original(context)
        with torch.no_grad():
            next(case.model.parameters()).add_(0.5)
        return result

    monkeypatch.setattr(case.model, "compose", mutate)
    with pytest.raises(DynamicsError):
        engine.step(case.session, case.agent)
    path = next(p for p in tmp_path.glob("*.safetensors") if not p.name.startswith("weights-"))
    artifact = load_neural_runtime_checkpoint(
        path,
        expected_sha256=sha256_bytes(path.read_bytes()),
        config=case.config.event_flow,
        source_revision=case.revision,
        device="cpu",
    )
    session, agent = restore_neural_runtime(artifact, device="cpu")
    assert agent.identity == case.identity
    assert session.state.core.mode.value == "have_memory"


def test_run_reuses_one_weights_blob_for_multiple_crashes(
    neural_archive_case, tmp_path, monkeypatch
):
    case = neural_archive_case
    engine = EventEngine(
        case.config.event_flow,
        crash_root=tmp_path,
        source_revision=case.revision,
        experiment_config_canonical_json=case.canonical,
    )

    def fail(context):
        raise NeuralError("fail")

    monkeypatch.setattr(case.model, "compose", fail)
    for _ in range(2):
        with pytest.raises(DynamicsError):
            engine.run_episode(case.bundle, case.agent)
    assert len(list(tmp_path.glob("weights-*.safetensors"))) == 1
    assert len(list(tmp_path.glob("*.safetensors"))) == 3
    assert len(list(tmp_path.glob("*.json"))) == 2
