"""Adversarial callbacks must stop before unsafe jumps and publish safe diagnostics."""

import json
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    EpisodeBundle,
    EpisodeKey,
    EpisodeRecipe,
    EpisodeTruth,
    EpisodeVariant,
    MatchedEpisodeCoordinate,
    PublicEpisode,
)
from silent_cascade.errors import CrashBundleError, DynamicsError
from silent_cascade.eventflow.config import Phase2Config
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.scheduling import ScheduledChoice, selected_gap_clamp_streak
from silent_cascade.eventflow.scripted import ScriptedEventFlowAgent
from silent_cascade.schemas import (
    ActivationPayload,
    AgentInit,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    LinkFact,
    Mode,
)


@pytest.fixture(scope="module")
def config():
    return resolve_config(
        Phase2Config,
        [
            Path("configs/base.yaml"),
            Path("configs/data/primary.yaml"),
            Path("configs/model/event_flow.yaml"),
        ],
    ).config.event_flow


def episode(links=0):
    facts = tuple(
        ExternalEvent(i + 1, 0.0, ExternalEventKind.FACT, LinkFact(i, i + 1)) for i in range(links)
    )
    facts += (ExternalEvent(links + 1, 0.0, ExternalEventKind.FACT, HazardFact(links, 2, 100.0)),)
    return EpisodeBundle(
        PublicEpisode(
            AgentInit("public-runtime-safety", 64, 4, 0.0),
            (
                *facts,
                ExternalEvent(links + 2, 1.0, ExternalEventKind.ACTIVATE, ActivationPayload(0)),
            ),
        ),
        EpisodeTruth(
            EpisodeKey(
                "ofd-v1",
                SplitNamespace.DEBUG,
                SuiteName.IID_PRIMARY,
                998877,
                MatchedEpisodeCoordinate("matched", 0, 0),
            ),
            EpisodeRecipe(max(1, links), EpisodeVariant.POSITIVE, 0, SuiteName.IID_PRIMARY, 0),
            tuple(range(links + 1)) if links else (63, 0),
            tuple(range(1, links + 2)),
            links + 1,
            2,
            ExternalEvent(999, 101.0, ExternalEventKind.OUTCOME, None),
            1.0,
            100.0,
            76.0,
            91.0,
            83.5,
            1,
            ("PRIVATE-LABEL-SENTINEL",),
        ),
    )


def activated(config, agent=None, *, links=0):
    agent = agent or ScriptedEventFlowAgent()
    engine = EventEngine(config)
    session = engine.start_episode(episode(links), agent)
    while session.state.core.mode is Mode.OBSERVING:
        engine.step(session, agent)
    return engine, session, agent


class TinyGaps(ScriptedEventFlowAgent):
    def next_internal_event(self, state):
        event = super().next_internal_event(state)
        if event is None:
            return None
        delta = 0.00005 / (2**state.core.executed_internal_events)
        return replace(event, timestamp=state.time + delta, predicted_delta=delta)


def test_clamp_is_pure_until_selected_and_raw_prediction_is_logged(config):
    engine, session, agent = activated(config, TinyGaps())
    first = engine._next_choice(session, agent)
    again = engine._next_choice(session, agent)
    assert first == again
    assert session.state.core.consecutive_gap_clamps == 0
    assert first.was_gap_clamped
    assert first.timestamp == 1.0001
    assert first.raw_predicted_delta == 0.00005
    engine.step(session, agent)
    assert session.state.core.executed_internal_events == 1
    assert session.state.core.consecutive_gap_clamps == 1
    row = session.trace.snapshot().events[-1].to_payload()
    assert row["was_gap_clamped"] is True
    assert row["raw_predicted_delta"] == 0.00005
    assert row["timestamp"] == 1.0001


def test_four_selected_clamps_fifth_failure_and_external_reset(config):
    engine, session, agent = activated(config, TinyGaps())
    choice = engine._next_choice(session, agent)
    streak = 0
    for want in (1, 2, 3, 4):
        streak = selected_gap_clamp_streak(streak, choice)
        assert streak == want
    with pytest.raises(DynamicsError):
        selected_gap_clamp_streak(streak, choice)
    external = ScheduledChoice(ExternalEvent(1, 0.0, ExternalEventKind.FACT, LinkFact(0, 1)), 0.0)
    assert selected_gap_clamp_streak(streak, external) == 0
    assert selected_gap_clamp_streak(0, choice) == 1
    assert selected_gap_clamp_streak(streak, replace(choice, was_gap_clamped=False)) == 0


def test_fifth_selected_clamp_is_rejected_before_callback(config):
    engine, session, agent = activated(config, TinyGaps())
    session.state = replace(
        session.state, core=replace(session.state.core, consecutive_gap_clamps=4)
    )
    with pytest.raises(DynamicsError):
        engine.step(session, agent)
    assert session.state.core.executed_internal_events == 0
    assert len(session.trace.snapshot().events) == 2


def test_shrinking_alternation_hits_refractory_without_hanging(config):
    engine, session, agent = activated(config, TinyGaps(), links=3)
    engine.step(session, agent)
    engine.step(session, agent)
    with pytest.raises(DynamicsError):
        engine.step(session, agent)
    assert session.state.core.executed_internal_events == 2


@pytest.mark.parametrize("offset,allowed", [(-1e-6, False), (0.0, True)])
def test_engine_enforces_same_kind_release_before_callback(config, offset, allowed):
    class ReleaseAgent(ScriptedEventFlowAgent):
        def next_internal_event(self, state):
            event = super().next_internal_event(state)
            if event and state.core.executed_internal_events == 2:
                timestamp = state.core.same_kind_refractory_until[0] + offset
                return replace(event, timestamp=timestamp, predicted_delta=timestamp - state.time)
            if event:
                return replace(event, timestamp=state.time + 0.0002, predicted_delta=0.0002)
            return None

    engine, session, agent = activated(config, ReleaseAgent(), links=3)
    engine.step(session, agent)
    engine.step(session, agent)
    if allowed:
        engine.step(session, agent)
        assert session.state.core.executed_internal_events == 3
    else:
        with pytest.raises(DynamicsError):
            engine.step(session, agent)
        assert session.state.core.executed_internal_events == 2


def test_attempted_65th_internal_event_is_rejected_before_agent_jump(config):
    class CapAgent(ScriptedEventFlowAgent):
        attempted = 0

        def on_internal(self, state, event):
            self.attempted += 1
            return super().on_internal(state, event)

    engine, session, agent = activated(config, CapAgent(), links=33)
    for _ in range(64):
        engine.step(session, agent)
    assert session.state.core.executed_internal_events == agent.attempted == 64
    with pytest.raises(DynamicsError):
        engine.step(session, agent)
    assert session.state.core.executed_internal_events == agent.attempted == 64


@pytest.mark.parametrize("delta", [-1.0, float("nan")])
def test_invalid_predicted_delta_fails_before_flow(config, delta):
    class BadDelta(ScriptedEventFlowAgent):
        def next_internal_event(self, state):
            event = super().next_internal_event(state)
            if event:
                object.__setattr__(event, "predicted_delta", delta)
            return event

    engine, session, agent = activated(config, BadDelta())
    with pytest.raises(DynamicsError):
        engine.step(session, agent)
    assert session.state.time == 1.0


@pytest.mark.parametrize("mutation", ["continuous", "guard", "latent_parameter", "perceived"])
def test_callback_input_mutation_cannot_hide_in_new_valid_segment(config, mutation):
    class Mutator(ScriptedEventFlowAgent):
        def on_internal(self, state, event):
            if mutation == "continuous":
                state.core.continuous.z_fast[0] = 0.333
            elif mutation == "guard":
                state.segment.parameters.guard_rates[0] = 2.0
            elif mutation == "latent_parameter":
                state.segment.parameters.flow_rates.z_fast[0] = 2.0
            else:
                object.__setattr__(state.core.memory.records[0].record, "confidence", 0.25)
            return super().on_internal(state, event)

    engine, session, agent = activated(config, Mutator())
    with pytest.raises(DynamicsError):
        engine.step(session, agent)
    assert session.state.core.executed_internal_events == 0


def test_boundary_detects_in_bounds_anchored_parameter_mutation_between_steps(config):
    engine, session, agent = activated(config)
    session.state.segment.parameters.flow_targets.z_fast[0] = 0.123
    with pytest.raises(DynamicsError):
        engine.step(session, agent)


class CrashAgent(ScriptedEventFlowAgent):
    failure = DynamicsError(
        "PRIVATE-PATH-SENTINEL",
        context={
            "label": "PRIVATE-LABEL-SENTINEL",
            "window": "PRIVATE-WINDOW-SENTINEL",
            "terminal": "PRIVATE-TERMINAL-SENTINEL",
        },
    )

    def on_internal(self, state, event):
        if state.core.executed_internal_events >= 22:
            raise self.failure
        return super().on_internal(state, event)


def test_crash_bundle_is_one_atomic_private_safe_bounded_manifest(config, tmp_path):
    engine = EventEngine(config, crash_root=tmp_path, source_revision="a" * 40)
    with pytest.raises(DynamicsError) as caught:
        engine.run_episode(episode(15), CrashAgent())
    assert caught.value is CrashAgent.failure
    files = list(tmp_path.iterdir())
    assert len(files) == 1 and files[0].suffix == ".json"
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o600
    raw = files[0].read_text()
    assert all(
        sentinel not in raw for sentinel in ["PRIVATE-", str(tmp_path), "998877", "OUTCOME", "END"]
    )
    data = json.loads(raw)
    assert data["error"]["code"] == "dynamics_error"
    assert data["traceback_text"]
    assert data["context"]["episode_public_id"] == "public-runtime-safety"
    assert data["context"]["source_revision"] == "a" * 40
    assert len(data["context"]["config_sha256"]) == 64
    assert data["context"]["checkpoint_ref"] is None
    assert len(data["context"]["last_events"]) == 20


def test_crash_publication_failure_chains_original_dynamics_error(config, tmp_path):
    root = tmp_path / "not-a-directory"
    root.touch()
    with pytest.raises(CrashBundleError) as caught:
        EventEngine(config, crash_root=root).run_episode(episode(15), CrashAgent())
    assert caught.value.__cause__ is CrashAgent.failure


def test_initialization_failure_also_publishes_public_context(config, tmp_path):
    class BadInit(ScriptedEventFlowAgent):
        def initialize(self, init):
            state = super().initialize(init)
            state.core.continuous.z_fast[0] = float("nan")
            return state

    with pytest.raises(DynamicsError):
        EventEngine(config, crash_root=tmp_path).run_episode(episode(), BadInit())
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.parametrize("mutation", ["perceived", "action_history"])
def test_engine_rechecks_public_history_before_scoring(config, mutation, monkeypatch):
    import silent_cascade.eventflow.engine as module
    from silent_cascade.schemas import Action

    engine, session, agent = activated(config)
    while session.state.core.mode is not Mode.QUIESCENT:
        engine.step(session, agent)
    if mutation == "perceived":
        object.__setattr__(session.state.core.memory.records[0].record, "confidence", 0.125)
    else:
        old = session.state.core.actions[0]
        object.__setattr__(
            session.state.core, "actions", (Action(3, old.timestamp, old.caused_by_event_id),)
        )
    scored = []
    real_score = module.score_actions

    def recording_score(*args):
        scored.append(True)
        return real_score(*args)

    monkeypatch.setattr(module, "score_actions", recording_score)
    with pytest.raises(DynamicsError):
        engine.step(session, agent)
    assert scored == []


def test_new_prediction_offset_must_match_creation_cursor(config):
    class FalseOffset(ScriptedEventFlowAgent):
        def next_internal_event(self, state):
            event = super().next_internal_event(state)
            return replace(event, predicted_delta=0.0) if event else None

    engine, session, agent = activated(config, FalseOffset())
    with pytest.raises(DynamicsError):
        engine.step(session, agent)


def test_external_selection_resets_seeded_streak_in_engine(config):
    agent = ScriptedEventFlowAgent()
    engine = EventEngine(config)
    session = engine.start_episode(episode(), agent)
    session.state = replace(
        session.state, core=replace(session.state.core, consecutive_gap_clamps=4)
    )
    engine.step(session, agent)
    assert session.state.core.consecutive_gap_clamps == 0


def test_corrupt_cached_digest_is_rejected_without_silent_reprediction(config):
    engine, session, agent = activated(config)
    engine.next_internal_event(session, agent)
    cached = session.prediction_cache.snapshot()
    object.__setattr__(cached, "prediction_snapshot_sha256", "0" * 64)
    predictions = session.state.core.counters.guard_predictions
    with pytest.raises(DynamicsError):
        engine.step(session, agent)
    assert session.state.core.counters.guard_predictions == predictions


def test_wrapped_provenance_failure_preserves_traceback_and_original_cause(config, tmp_path):
    from silent_cascade.errors import ProvenanceError

    original = ProvenanceError("PRIVATE-PROVENANCE-SENTINEL")

    class BadProvenance(ScriptedEventFlowAgent):
        def on_internal(self, state, event):
            raise original

    with pytest.raises(DynamicsError) as caught:
        EventEngine(config, crash_root=tmp_path).run_episode(episode(), BadProvenance())
    assert caught.value.__cause__ is original
    data = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert data["traceback_text"]
    assert "PRIVATE-" not in json.dumps(data)
