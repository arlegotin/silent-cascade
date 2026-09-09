"""Exact CPU continuation from real public-facts causal boundaries."""

import json
from dataclasses import fields, replace

import pytest
import torch
from conftest import CASES, bundle_for, episode

from silent_cascade.errors import CrashBundleError, DynamicsError, ReplayError
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.scripted import ScriptedEventFlowAgent
from silent_cascade.logging.trace import tensor_sha256

REVISION = "a" * 40


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
@pytest.mark.parametrize(
    "boundary", ["initial", "flow", "fact", "activate", "recall", "link", "terminal_compose", "act"]
)
def test_checkpoint_continuation_preserves_exact_causality(
    tmp_path, config, case, boundary, monkeypatch
):
    from silent_cascade.eventflow.checkpoint import (
        load_runtime_checkpoint,
        publish_runtime_checkpoint,
        restore_runtime_session,
        snapshot_runtime,
    )

    engine, agent = EventEngine(config), ScriptedEventFlowAgent()
    bundle = bundle_for(case)
    expected = engine.run_episode(bundle, ScriptedEventFlowAgent())
    session = engine.start_episode(bundle, agent)
    if boundary == "flow":
        engine.run_until(session, agent, 0.75)
    elif boundary != "initial":
        targets = {
            "fact": 1,
            "activate": 5,
            "recall": 6,
            "link": 7,
            "terminal_compose": 5 + len(case["expected"]["grammar"]) - (case["name"] == "positive"),
            "act": 5 + len(case["expected"]["grammar"]),
        }
        for _ in range(targets[boundary]):
            engine.step(session, agent)
    # Cache an event or dormancy before taking the immutable snapshot.
    engine.next_internal_event(session, agent)
    artifact = snapshot_runtime(session, agent, config=config, source_revision=REVISION)
    path = tmp_path / "runtime.safetensors"
    publish_runtime_checkpoint(path, artifact)
    loaded = load_runtime_checkpoint(path, config=config, source_revision=REVISION)
    restored, restored_agent = restore_runtime_session(
        loaded, config=config, source_revision=REVISION
    )
    assert restored_agent.compute_counters() == agent.compute_counters()
    assert restored.trace.snapshot().sha256 == session.trace.snapshot().sha256
    assert restored.external_queue.snapshot() == session.external_queue.snapshot()
    assert restored.prediction_cache.snapshot() == session.prediction_cache.snapshot()
    assert restored.trajectory.boundary_count == session.trajectory.boundary_count
    for name in (item.name for item in fields(session.state.core.continuous)):
        assert tensor_sha256(getattr(restored.state.core.continuous, name)) == tensor_sha256(
            getattr(session.state.core.continuous, name)
        )

    def no_prediction(state):
        raise AssertionError("restored cached boundary must not call the controller")

    with monkeypatch.context() as patch:
        patch.setattr(restored_agent, "next_internal_event", no_prediction)
        engine.next_internal_event(restored, restored_agent)
        patch.setattr("time.time", lambda: 10**12)
        engine.step(restored, restored_agent)
    while not engine.step(restored, restored_agent):
        pass
    assert restored.trace.snapshot().to_payload() == expected.trace.to_payload()
    assert restored.trace.snapshot().sha256 == expected.trace.sha256
    assert restored.terminal_score == expected.score
    assert replace(restored.state.core.counters, checkpoint_flow_evaluations=0) == expected.counters
    for row in expected.trace.events:
        actual = restored.trajectory.state_at(row.timestamp)
        original = expected.trajectory.state_at(row.timestamp)
        assert all(
            torch.equal(getattr(actual, item.name), getattr(original, item.name))
            for item in fields(actual)
        )


@pytest.mark.parametrize("driver", ["step", "run_until"])
def test_real_65th_event_failure_publishes_replayable_prefailure_pair(tmp_path, config, driver):
    from silent_cascade.eventflow.checkpoint import load_runtime_checkpoint, restore_runtime_session

    engine = EventEngine(config, crash_root=tmp_path, source_revision=REVISION)
    agent = ScriptedEventFlowAgent()
    session = engine.start_episode(episode(33), agent)
    with pytest.raises(DynamicsError) as caught:
        if driver == "run_until":
            engine.run_until(session, agent, 100.0)
        else:
            while not engine.step(session, agent):
                pass
    assert caught.value.context["invariant"] == "internal_event_cap"
    manifests = list(tmp_path.glob("*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_bytes())
    ref = manifest["context"]["checkpoint_ref"]
    assert ref == manifest["bundle_id"] + ".safetensors"
    artifact = load_runtime_checkpoint(tmp_path / ref, config=config, source_revision=REVISION)
    restored, resumed_agent = restore_runtime_session(
        artifact, config=config, source_revision=REVISION
    )
    assert restored.state.core.executed_internal_events == 64
    with pytest.raises(DynamicsError) as replay_error:
        EventEngine(config).step(restored, resumed_agent)
    assert replay_error.value.context == caught.value.context


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="native MPS required")
def test_native_mps_checkpoint_preserves_bytes_and_same_device_continuation(
    tmp_path, config, monkeypatch
):
    from silent_cascade.eventflow.checkpoint import (
        load_runtime_checkpoint,
        publish_runtime_checkpoint,
        restore_runtime_session,
        snapshot_runtime,
    )

    engine, agent = EventEngine(config), ScriptedEventFlowAgent(device="mps")
    session = engine.start_episode(bundle_for(CASES[0]), agent)
    engine.run_until(session, agent, 0.15)
    artifact = snapshot_runtime(session, agent, config=config, source_revision=REVISION)
    path = tmp_path / "mps.safetensors"
    publish_runtime_checkpoint(path, artifact)
    expected_rng_draw = torch.rand(8, device="mps").cpu()
    with monkeypatch.context() as patch:
        patch.setattr(torch.backends.mps, "is_available", lambda: False)
        loaded = load_runtime_checkpoint(path, config=config, source_revision=REVISION)
        assert loaded.metadata.original_device == "mps"
        assert torch.equal(loaded.tensors["rng.mps"], artifact.tensors["rng.mps"])
        with pytest.raises(ReplayError, match="same-device"):
            restore_runtime_session(loaded, config=config, source_revision=REVISION, device="cpu")
    restored, resumed = restore_runtime_session(
        loaded, config=config, source_revision=REVISION, device="mps", restore_mps=True
    )
    assert torch.equal(torch.rand(8, device="mps").cpu(), expected_rng_draw)
    engine.step(session, agent)
    engine.step(restored, resumed)
    assert restored.trace.snapshot().sha256 == session.trace.snapshot().sha256
    assert restored.state.core.counters == session.state.core.counters


@pytest.mark.parametrize("publication_failure", ["missing_source", "writer"])
def test_crash_checkpoint_publication_failure_never_claims_missing_archive(
    tmp_path, config, monkeypatch, publication_failure
):
    import silent_cascade.eventflow.checkpoint as checkpoints

    engine = EventEngine(
        config,
        crash_root=tmp_path,
        source_revision=None if publication_failure == "missing_source" else REVISION,
    )
    agent = ScriptedEventFlowAgent()
    session = engine.start_episode(episode(33), agent)
    if publication_failure == "writer":

        def fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(checkpoints, "publish_runtime_checkpoint", fail)
    with pytest.raises(CrashBundleError) as caught:
        while not engine.step(session, agent):
            pass
    assert isinstance(caught.value.__cause__, DynamicsError)
    assert not list(tmp_path.glob("*.json"))


def test_prefailure_archive_survives_callback_tensor_corruption(tmp_path, config, monkeypatch):
    from silent_cascade.eventflow.checkpoint import load_runtime_checkpoint, restore_runtime_session

    engine = EventEngine(config, crash_root=tmp_path, source_revision=REVISION)
    agent = ScriptedEventFlowAgent()
    session = engine.start_episode(bundle_for(CASES[0]), agent)

    def corrupt(state, event):
        state.core.continuous.z_fast[0] = float("nan")
        raise DynamicsError("injected callback corruption")

    monkeypatch.setattr(agent, "on_external", corrupt)
    with pytest.raises(DynamicsError):
        engine.step(session, agent)
    manifest = json.loads(next(tmp_path.glob("*.json")).read_bytes())
    artifact = load_runtime_checkpoint(
        tmp_path / manifest["context"]["checkpoint_ref"], config=config, source_revision=REVISION
    )
    restored, resumed_agent = restore_runtime_session(
        artifact, config=config, source_revision=REVISION
    )
    assert restored.trace.snapshot().events == ()
    while not EventEngine(config).step(restored, resumed_agent):
        pass
    expected = EventEngine(config).run_episode(bundle_for(CASES[0]), ScriptedEventFlowAgent())
    assert restored.trace.snapshot().sha256 == expected.trace.sha256


def test_unregistered_subclass_crash_is_explicitly_nonreplayable(tmp_path, config):
    class UnknownAgent(ScriptedEventFlowAgent):
        def on_external(self, state, event):
            raise DynamicsError("unregistered callback")

    with pytest.raises(DynamicsError):
        EventEngine(config, crash_root=tmp_path, source_revision=REVISION).run_episode(
            bundle_for(CASES[0]), UnknownAgent()
        )
    manifest = json.loads(next(tmp_path.glob("*.json")).read_bytes())
    assert manifest["context"]["checkpoint_ref"] is None
    assert not list(tmp_path.glob("*.safetensors"))


def test_restore_preserves_cached_losing_act_inside_terminal_tie(tmp_path, config):
    """Deliberately non-OFD private scheduling fixture, never scientific evidence."""
    from silent_cascade.eventflow.checkpoint import restore_runtime_session, snapshot_runtime
    from silent_cascade.eventflow.scheduling import ExternalEventQueue

    engine, agent = EventEngine(config), ScriptedEventFlowAgent()
    bundle = bundle_for(CASES[0])
    probe = engine.start_episode(bundle, agent)
    while probe.state.core.mode.value != "holding_hazard":
        engine.step(probe, agent)
    action_time = engine.next_internal_event(probe, agent).timestamp
    terminal = replace(bundle.truth.private_terminal, timestamp=action_time + 0.5e-9)
    delay = terminal.timestamp - bundle.truth.activation_time
    truth = replace(
        bundle.truth,
        private_terminal=terminal,
        episode_delay=delay,
        action_window_start=1.0 + 0.75 * delay,
        action_window_end=1.0 + 0.9 * delay,
        action_target=1.0 + 0.825 * delay,
    )
    # This changes only the private debug schedule, not public facts, agent
    # identity, runtime invariants, or the actual analytic ACT prediction.
    probe.truth = truth
    probe.external_queue = ExternalEventQueue.from_snapshot((terminal,))
    assert engine.step(probe, agent)
    expected = probe.trace.snapshot()
    agent = ScriptedEventFlowAgent()
    session = engine.start_episode(bundle, agent)
    session.truth = truth
    session.external_queue = ExternalEventQueue.from_snapshot((*bundle.public.events, terminal))
    engine.run_until(session, agent, action_time + 0.25e-9)
    assert session.prediction_cache.snapshot().event.timestamp < session.state.time
    artifact = snapshot_runtime(session, agent, config=config, source_revision=REVISION)
    restored, resumed = restore_runtime_session(artifact, config=config, source_revision=REVISION)
    assert engine.step(restored, resumed)
    assert restored.trace.snapshot().sha256 == expected.sha256
