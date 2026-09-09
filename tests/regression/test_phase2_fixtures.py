"""Hand-authored public traces; expected answers are literal fixture values."""

import json
from dataclasses import asdict
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
from silent_cascade.eventflow.checkpoint import (
    load_runtime_checkpoint,
    publish_runtime_checkpoint,
    restore_runtime_session,
    snapshot_runtime,
)
from silent_cascade.eventflow.config import Phase2Config
from silent_cascade.eventflow.engine import EpisodeResult, EventEngine
from silent_cascade.eventflow.replay import verify_replay, write_replay_artifact
from silent_cascade.eventflow.scripted import ScriptedEventFlowAgent
from silent_cascade.schemas import (
    ActivationPayload,
    AgentInit,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    LinkFact,
    Mode,
    SafeFact,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/phase2/scripted_cases.json"


@pytest.mark.parametrize("case", json.loads(FIXTURE.read_text())["cases"], ids=lambda c: c["name"])
def test_hand_authored_scripted_trace(case: dict, tmp_path: Path) -> None:
    """Wrong retrieval, composition, support, cadence or final scoring breaks this trace."""
    payloads = {"link": LinkFact, "hazard": HazardFact, "safe": SafeFact}
    facts = tuple(
        ExternalEvent(record_id, time, ExternalEventKind.FACT, payloads[kind](**payload))
        for record_id, time, kind, payload in case["facts"]
    )
    event_id, time, start = case["activation"]
    public = PublicEpisode(
        AgentInit("00000000-0000-4000-8000-000000000001", 64, 4, 0.0),
        (
            *facts,
            ExternalEvent(event_id, time, ExternalEventKind.ACTIVATE, ActivationPayload(start)),
        ),
    )
    positive = case["name"] == "positive"
    truth = EpisodeTruth(
        EpisodeKey(
            "ofd-v1",
            SplitNamespace.DEBUG,
            SuiteName.IID_PRIMARY,
            1,
            MatchedEpisodeCoordinate("matched", 0, 0),
        ),
        EpisodeRecipe(2, EpisodeVariant(case["name"]), 0, SuiteName.IID_PRIMARY, 0),
        (0, 1, 2),
        (8, 3, 5) if positive else (8, 3),
        5 if positive else None,
        2 if positive else None,
        ExternalEvent(
            10, 3.0, ExternalEventKind.OUTCOME if positive else ExternalEventKind.END, None
        ),
        1.0,
        2.0,
        2.5 if positive else None,
        2.8 if positive else None,
        2.65 if positive else None,
        0,
        (),
    )
    config = resolve_config(
        Phase2Config,
        [
            Path("configs/base.yaml"),
            Path("configs/data/primary.yaml"),
            Path("configs/model/event_flow.yaml"),
        ],
    ).config.event_flow
    agent = ScriptedEventFlowAgent()
    engine = EventEngine(config)
    session = engine.start_episode(EpisodeBundle(public, truth), agent)
    while not engine.step(session, agent):
        if session.state.core.mode is Mode.QUIESCENT:
            assert agent.next_internal_event(session.state) is None
            assert bool((session.state.segment.parameters.guard_targets <= 0.90).all())
    events = session.trace.snapshot().events
    internal = [event for event in events if event.kind in {"recall", "compose", "act"}]
    expected = case["expected"]
    assert [event.kind for event in internal] == expected["grammar"]
    assert [event.selected_record_id for event in internal] == expected["selected_ids"]
    assert [event.selected_rank for event in internal] == expected["selected_ranks"]
    assert [event.timestamp for event in internal] == pytest.approx(expected["times"], abs=1e-7)
    assert [list(event.support_after) for event in internal] == expected["support_after"]
    hypothesis = session.state.core.hypothesis
    actual = json.loads(json.dumps(asdict(hypothesis))) if hypothesis else None
    assert actual == expected["hypothesis"]
    if expected["action"] is None:
        assert session.state.core.actions == ()
    else:
        (action,) = session.state.core.actions
        assert action.hazard_type == expected["action"]["hazard_type"]
        assert action.timestamp == pytest.approx(expected["action"]["timestamp"], abs=1e-7)
        assert action.caused_by_event_id == internal[-1].event_id
    assert events[-1].timestamp == expected["terminal_time"]
    assert session.terminal_score.timed_success is expected["timed_success"]
    assert session.terminal_score.reason == expected["score_reason"]
    assert agent.compute_counters().foundation_model_calls == 0
    assert session.state.core.counters.foundation_model_calls == 0
    result = EpisodeResult(
        session.public_id,
        session.terminal_score,
        session.state.core.actions,
        session.state.core.counters,
        session.trace.snapshot(),
        session.trajectory,
    )
    artifact = write_replay_artifact(
        tmp_path / "replay.json", bundle=EpisodeBundle(public, truth), config=config, result=result
    )
    assert verify_replay(artifact).trace_sha256 == result.trace.sha256
    resumed_agent = ScriptedEventFlowAgent()
    paused = engine.start_episode(EpisodeBundle(public, truth), resumed_agent)
    engine.run_until(paused, resumed_agent, 0.75)
    checkpoint = snapshot_runtime(paused, resumed_agent, config=config, source_revision="a" * 40)
    checkpoint_path = tmp_path / "runtime.safetensors"
    publish_runtime_checkpoint(checkpoint_path, checkpoint)
    loaded = load_runtime_checkpoint(checkpoint_path, config=config, source_revision="a" * 40)
    resumed, resumed_agent = restore_runtime_session(
        loaded, config=config, source_revision="a" * 40
    )
    while not engine.step(resumed, resumed_agent):
        pass
    assert resumed.trace.snapshot().sha256 == result.trace.sha256
    assert resumed.terminal_score == result.score
