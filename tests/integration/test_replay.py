"""Private archive replay and actual trajectory queries, using hand-authored facts."""

import json
import math
import os
import stat
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

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
from silent_cascade.errors import ReplayError, TimeOrderError
from silent_cascade.eventflow.config import Phase2Config
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.scripted import ScriptedEventFlowAgent
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.trace import SegmentSummary, tensor_sha256
from silent_cascade.schemas import (
    ActivationPayload,
    AgentInit,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    LinkFact,
    SafeFact,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/phase2"
CASES = json.loads((FIXTURES / "scripted_cases.json").read_text())["cases"]
REPLAY_CASES = json.loads((FIXTURES / "replay_cases.json").read_text())["cases"]


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


def bundle_for(case):
    payloads = {"link": LinkFact, "hazard": HazardFact, "safe": SafeFact}
    facts = tuple(
        ExternalEvent(record, time, ExternalEventKind.FACT, payloads[kind](**payload))
        for record, time, kind, payload in case["facts"]
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
    return EpisodeBundle(public, truth)


def write_case(tmp_path, config, case=CASES[0]):
    from silent_cascade.eventflow.replay import write_replay_artifact

    bundle = bundle_for(case)
    result = EventEngine(config).run_episode(bundle, ScriptedEventFlowAgent())
    path = tmp_path / "replay.json"
    artifact = write_replay_artifact(path, bundle=bundle, config=config, result=result)
    return path, artifact, result


def rehash_archived_trace(payload):
    """Keep semantic-tamper fixtures internally consistent to reach row comparison."""
    online = {
        "schema_version": payload["trace"]["schema_version"],
        "events": [
            {key: value for key, value in row.items() if key != "source"}
            for row in payload["trace"]["events"]
        ],
    }
    payload["trace"]["sha256"] = sha256_bytes(canonical_json_bytes(online))


@pytest.mark.parametrize(
    "case,expected", zip(CASES, REPLAY_CASES, strict=True), ids=lambda c: c["name"]
)
def test_replay_roundtrip_is_exact_repeatable_and_private(tmp_path, config, case, expected):
    from silent_cascade.eventflow.replay import (
        load_replay_artifact,
        verify_replay,
        write_replay_artifact,
    )

    path, artifact, result = write_case(tmp_path, config, case)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    loaded = load_replay_artifact(path)
    assert loaded == artifact
    first = verify_replay(loaded)
    second = verify_replay(loaded)
    assert first == second and first.sha256 == second.sha256
    assert first.event_count == expected["event_count"]
    assert first.trace_sha256 == result.trace.sha256
    assert result.score.reason == expected["reason"]
    assert len(result.actions) == expected["action_count"]
    assert artifact.access_class == "validation_private"
    assert "truth" not in result.trace.to_payload()
    assert result.counters.foundation_model_calls == 0
    assert (
        write_replay_artifact(path, bundle=bundle_for(case), config=config, result=result)
        == artifact
    )


def test_actual_trajectory_queries_preserve_runtime_trace_and_terminal_reset(config):
    engine, agent = EventEngine(config), ScriptedEventFlowAgent()
    session = engine.start_episode(bundle_for(CASES[0]), agent)
    anchors = [(session.state.time, session.state)]
    engine.run_until(session, agent, 0.05)
    assert session.trajectory.boundary_count == 1
    while True:
        done = engine.step(session, agent)
        anchors.append((session.state.time, session.state))
        if done:
            break
    trace_hash, counters = session.trace.snapshot().sha256, session.state.core.counters
    agent_counters = agent.compute_counters()
    tensor_signatures = [
        (
            SegmentSummary.from_segment(state.segment),
            tuple(
                tensor_sha256(getattr(state.core.continuous, f.name))
                for f in fields(state.core.continuous)
            ),
        )
        for _, state in anchors
    ]
    trajectory = session.trajectory
    assert trajectory.boundary_count == len(anchors) == 14
    for time, state in anchors:
        actual = trajectory.state_at(time)
        for field in fields(actual):
            torch.testing.assert_close(
                getattr(actual, field.name),
                getattr(state.core.continuous, field.name),
                atol=0,
                rtol=0,
            )
    # Real guard approaches threshold, then resets at the actual RECALL boundary.
    recall_time = next(time for time, state in anchors if state.core.executed_internal_events == 1)
    assert float(
        trajectory.state_at(math.nextafter(recall_time, 0.0)).guard_accumulators[0]
    ) == pytest.approx(1.0, abs=2e-6)
    assert trajectory.state_at(recall_time).guard_accumulators.tolist() == [0.0, 0.0, 0.0]
    for time in (0.05, 0.15, 0.65, math.nextafter(recall_time, math.inf), 2.9, 3.0):
        trajectory.state_at(time).z_fast.zero_()
    assert trajectory.state_at(3.0).guard_accumulators.tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(TimeOrderError):
        trajectory.state_at(math.nextafter(3.0, math.inf))
    assert session.state.core.counters == counters
    assert agent.compute_counters() == agent_counters
    assert session.trace.snapshot().sha256 == trace_hash
    assert tensor_signatures == [
        (
            SegmentSummary.from_segment(state.segment),
            tuple(
                tensor_sha256(getattr(state.core.continuous, f.name))
                for f in fields(state.core.continuous)
            ),
        )
        for _, state in anchors
    ]


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("timestamp", 0.1001, "trace.events.0.timestamp"),
        ("parent_event_id", 123, "trace.events.0.parent_event_id"),
        ("event_id", 123, "trace.events.0.event_id"),
        ("selected_record_id", 999, "selected_record_id"),
        ("selected_rank", 2, "selected_rank"),
        ("support_after", [999], "support_after"),
        ("state_sha256", "0" * 64, "state_sha256"),
        ("prediction_snapshot_sha256", "0" * 64, "prediction_snapshot_sha256"),
        ("kind", "recall", "kind"),
    ],
)
def test_replay_reports_first_causal_field_mismatch(tmp_path, config, field, value, expected):
    from silent_cascade.eventflow.replay import ReplayArtifact, verify_replay

    _, artifact, _ = write_case(tmp_path, config)
    payload = artifact.model_dump(mode="json")
    index = 5 if field in {"selected_record_id", "selected_rank", "kind"} else 0
    payload["trace"]["events"][index][field] = value if field != "kind" else "compose"
    rehash_archived_trace(payload)
    payload["payload_sha256"] = sha256_bytes(
        canonical_json_bytes({k: v for k, v in payload.items() if k != "payload_sha256"})
    )
    tampered = ReplayArtifact.model_validate_json(json.dumps(payload))
    with pytest.raises(ReplayError, match=expected):
        verify_replay(tampered)


@pytest.mark.parametrize(
    "target",
    [
        "action",
        "caused_by",
        "hypothesis",
        "score",
        "trace_hash",
        "config_hash",
        "episode",
        "agent_version",
    ],
)
def test_replay_rejects_independent_envelope_tampering(tmp_path, config, target):
    from silent_cascade.eventflow.replay import load_replay_artifact, verify_replay

    path, artifact, _ = write_case(tmp_path, config)
    payload = artifact.model_dump(mode="json")
    expected = {
        "action": "actions",
        "caused_by": "caused_by_event_id",
        "hypothesis": "hypothesis_after",
        "score": "score",
        "trace_hash": "trace.sha256",
        "config_hash": "config_sha256",
        "episode": "payload_sha256",
        "agent_version": "agent_implementation",
    }[target]
    if target in {"action", "caused_by"}:
        payload["trace"]["events"][-2]["actions"][0][
            "hazard_type" if target == "action" else "caused_by_event_id"
        ] = 1
    elif target == "hypothesis":
        payload["trace"]["events"][-2]["hypothesis_after"]["hazard_type"] = 1
    elif target == "score":
        payload["expected_result"]["score"]["timed_success"] = False
    elif target == "trace_hash":
        payload["trace"]["sha256"] = "0" * 64
    elif target == "config_hash":
        payload["config_sha256"] = "0" * 64
    elif target == "episode":
        payload["episode"]["truth"]["key"]["root_seed"] = 2
    else:
        payload["agent_implementation"] = "arbitrary.import.Class"
    if target in {"action", "caused_by", "hypothesis"}:
        rehash_archived_trace(payload)
    if target != "episode":
        payload["payload_sha256"] = sha256_bytes(
            canonical_json_bytes({k: v for k, v in payload.items() if k != "payload_sha256"})
        )
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ReplayError, match=expected):
        verify_replay(load_replay_artifact(path))


@pytest.mark.parametrize(
    "kind", ["symlink", "parent_symlink", "fifo", "oversize", "unknown_field", "duplicate_key"]
)
def test_archive_reader_is_bounded_regular_strict_json(tmp_path, config, kind):
    from silent_cascade.eventflow.replay import load_replay_artifact

    path, artifact, _ = write_case(tmp_path, config)
    if kind == "symlink":
        link = tmp_path / "link.json"
        link.symlink_to(path)
        path = link
    elif kind == "parent_symlink":
        link = tmp_path / "linked"
        link.symlink_to(tmp_path, target_is_directory=True)
        path = link / path.name
    elif kind == "fifo":
        path = tmp_path / "fifo"
        os.mkfifo(path)
    elif kind == "oversize":
        with path.open("r+b") as stream:
            stream.truncate(32 * 1024 * 1024 + 1)
    elif kind == "unknown_field":
        payload = artifact.model_dump(mode="json")
        payload["plugin"] = "untrusted"
        payload["payload_sha256"] = sha256_bytes(
            canonical_json_bytes({k: v for k, v in payload.items() if k != "payload_sha256"})
        )
        path.write_bytes(canonical_json_bytes(payload))
    else:
        raw = path.read_bytes()
        path.write_bytes(b'{"schema_version":"phase2-replay-v1",' + raw[1:])
    with pytest.raises(ReplayError):
        load_replay_artifact(path)


def test_archive_publication_never_clobbers_divergent_or_symlink_target(tmp_path, config):
    from silent_cascade.eventflow.replay import write_replay_artifact

    path, _, result = write_case(tmp_path, config)
    original = path.read_bytes()
    other = EventEngine(config).run_episode(bundle_for(CASES[1]), ScriptedEventFlowAgent())
    with pytest.raises(ReplayError):
        write_replay_artifact(path, bundle=bundle_for(CASES[1]), config=config, result=other)
    assert path.read_bytes() == original
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ReplayError):
        write_replay_artifact(link, bundle=bundle_for(CASES[0]), config=config, result=result)


def test_trajectory_long_dormant_interval_has_no_sampling_work(config):
    bundle = bundle_for(CASES[2])
    bundle = replace(
        bundle,
        truth=replace(
            bundle.truth,
            episode_delay=1_000_000.0,
            private_terminal=replace(bundle.truth.private_terminal, timestamp=1_000_001.0),
        ),
    )
    result = EventEngine(config).run_episode(bundle, ScriptedEventFlowAgent())
    before = result.counters
    digest = result.trace.sha256
    state = result.trajectory.state_at(500_000.0)
    assert bool(torch.isfinite(state.z_fast).all())
    assert result.trajectory.boundary_count == 11
    assert result.counters == before
    assert result.trace.sha256 == digest


@pytest.mark.parametrize("coherent", [False, True])
def test_timing_tolerance_stops_at_preflight_or_first_exact_row_mismatch(
    tmp_path, config, monkeypatch, coherent
):
    from silent_cascade.eventflow.replay import ReplayArtifact, verify_replay

    _, artifact, _ = write_case(tmp_path, config)
    payload = artifact.model_dump(mode="json")
    payload["trace"]["events"][0]["timestamp"] += 0.5e-9
    if coherent:
        rehash_archived_trace(payload)
    payload["payload_sha256"] = sha256_bytes(
        canonical_json_bytes({k: v for k, v in payload.items() if k != "payload_sha256"})
    )
    tampered = ReplayArtifact.model_validate_json(json.dumps(payload))
    executed = []
    original_step = EventEngine.step

    def tracked_step(self, session, agent):
        terminal = original_step(self, session, agent)
        executed.append(session.state.core.last_event_id)
        return terminal

    monkeypatch.setattr(EventEngine, "step", tracked_step)
    with pytest.raises(ReplayError) as failure:
        verify_replay(tampered)
    assert executed == ([5] if coherent else [])
    assert failure.value.context["field"] == ("trace.events.0" if coherent else "trace.sha256")


def test_replay_stops_at_first_mismatch_and_config_fails_before_execution(
    tmp_path, config, monkeypatch
):
    from silent_cascade.eventflow.replay import ReplayArtifact, verify_replay

    _, artifact, _ = write_case(tmp_path, config)
    payload = artifact.model_dump(mode="json")
    payload["trace"]["events"][0]["timestamp"] += 0.001
    rehash_archived_trace(payload)
    payload["payload_sha256"] = sha256_bytes(
        canonical_json_bytes({k: v for k, v in payload.items() if k != "payload_sha256"})
    )
    tampered = ReplayArtifact.model_validate_json(json.dumps(payload))
    executed = []
    original_step = EventEngine.step

    def tracked_step(self, session, agent):
        terminal = original_step(self, session, agent)
        executed.append(session.state.core.last_event_id)
        return terminal

    monkeypatch.setattr(EventEngine, "step", tracked_step)
    with pytest.raises(ReplayError, match=r"trace\.events\.0\.timestamp"):
        verify_replay(tampered)
    assert executed == [5]
    executed.clear()
    payload["config_sha256"] = "0" * 64
    payload["payload_sha256"] = sha256_bytes(
        canonical_json_bytes({k: v for k, v in payload.items() if k != "payload_sha256"})
    )
    tampered = ReplayArtifact.model_validate_json(json.dumps(payload))
    with pytest.raises(ReplayError, match="config_sha256"):
        verify_replay(tampered)
    assert executed == []


def test_paused_original_replay_projects_only_checkpoint_diagnostics(tmp_path, config):
    from silent_cascade.eventflow.engine import EpisodeResult
    from silent_cascade.eventflow.replay import verify_replay, write_replay_artifact

    bundle = bundle_for(CASES[0])
    engine, agent = EventEngine(config), ScriptedEventFlowAgent()
    session = engine.start_episode(bundle, agent)
    engine.run_until(session, agent, 0.05)
    engine.run_until(session, agent, 1.1)
    while not engine.step(session, agent):
        pass
    result = EpisodeResult(
        session.public_id,
        session.terminal_score,
        session.state.core.actions,
        session.state.core.counters,
        session.trace.snapshot(),
        session.trajectory,
    )
    assert result.counters.checkpoint_flow_evaluations == 2
    artifact = write_replay_artifact(
        tmp_path / "paused.json", bundle=bundle, config=config, result=result
    )
    assert artifact.expected_result.counters.checkpoint_flow_evaluations == 0
    for field in fields(result.counters):
        if field.name != "checkpoint_flow_evaluations":
            assert getattr(artifact.expected_result.counters, field.name) == getattr(
                result.counters, field.name
            )
    assert verify_replay(artifact).trace_sha256 == result.trace.sha256
    assert result.counters.checkpoint_flow_evaluations == 2


@pytest.mark.parametrize(
    "field,value",
    [("source", "external"), ("checkpoint_flow_evaluations", 1), ("episode_delay", -1.0)],
)
def test_archive_rejects_invalid_source_counter_and_private_episode(tmp_path, config, field, value):
    from silent_cascade.eventflow.replay import load_replay_artifact

    path, artifact, _ = write_case(tmp_path, config)
    payload = artifact.model_dump(mode="json")
    if field == "source":
        payload["trace"]["events"][5][field] = value
    elif field == "checkpoint_flow_evaluations":
        payload["expected_result"]["counters"][field] = value
    else:
        payload["episode"]["truth"][field] = value
    payload["payload_sha256"] = sha256_bytes(
        canonical_json_bytes({k: v for k, v in payload.items() if k != "payload_sha256"})
    )
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ReplayError):
        load_replay_artifact(path)


def test_same_time_external_ties_roundtrip_the_full_tie_metadata(tmp_path, config):
    from silent_cascade.eventflow.replay import verify_replay, write_replay_artifact

    bundle = bundle_for(CASES[0])
    tied_facts = tuple(
        replace(fact, timestamp=0.1)
        for fact in sorted(bundle.public.events[:-1], key=lambda event: event.event_id)
    )
    bundle = replace(
        bundle, public=replace(bundle.public, events=(*tied_facts, bundle.public.events[-1]))
    )
    result = EventEngine(config).run_episode(bundle, ScriptedEventFlowAgent())
    assert [event.event_id for event in result.trace.events[:4]] == [1, 3, 5, 8]
    assert result.trace.events[0].tie_id is not None
    artifact = write_replay_artifact(
        tmp_path / "ties.json", bundle=bundle, config=config, result=result
    )
    assert artifact.trace.events[0].tie_id == result.trace.events[0].tie_id
    assert verify_replay(artifact).trace_sha256 == result.trace.sha256
    assert result.trajectory.state_at(0.1).guard_accumulators.tolist() == [0.0, 0.0, 0.0]
