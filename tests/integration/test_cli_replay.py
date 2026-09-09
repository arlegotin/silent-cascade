"""Public replay command boundaries; private archive contents never print."""

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime

import pytest
from conftest import CASES, bundle_for, episode
from typer.testing import CliRunner

from silent_cascade.cli import app
from silent_cascade.errors import DynamicsError
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.replay import write_replay_artifact
from silent_cascade.eventflow.scripted import ScriptedEventFlowAgent
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.logging.crash_bundle import CrashBundleManifest, CrashContext

runner = CliRunner()


def _artifact(tmp_path, config):
    result = EventEngine(config).run_episode(bundle_for(CASES[0]), ScriptedEventFlowAgent())
    path = tmp_path / "replay.json"
    artifact = write_replay_artifact(
        path, bundle=bundle_for(CASES[0]), config=config, result=result
    )
    return path, artifact, result


def test_cli_replay_emits_strict_public_success_payload(tmp_path, config) -> None:
    path, _, result = _artifact(tmp_path, config)

    completed = runner.invoke(app, ["replay", str(path), "--json"])

    assert completed.exit_code == 0, completed.output
    payload = json.loads(completed.stdout)
    assert payload == {
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "device": "cpu",
        "event_count": len(result.trace.events),
        "foundation_model_calls": 0,
        "matched": True,
        "replay_kind": "episode",
        "schema_version": "phase2-replay-cli-v1",
        "timed_success": result.score.timed_success,
        "trace_sha256": result.trace.sha256,
    }
    assert "PRIVATE" not in completed.output


def test_installed_console_script_replays_artifact(tmp_path, config) -> None:
    path, _, _ = _artifact(tmp_path, config)
    executable = shutil.which("silent-cascade")
    assert executable is not None

    completed = subprocess.run(
        [executable, "replay", str(path), "--json"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        json.loads(completed.stdout)["artifact_sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_cli_replay_maps_missing_artifact_to_stable_error_payload(tmp_path) -> None:
    completed = runner.invoke(app, ["replay", str(tmp_path / "missing.json"), "--json"])

    assert completed.exit_code == 1
    assert json.loads(completed.stderr) == {
        "code": "replay_error",
        "context": {"field": "archive.path_or_file"},
        "message": "replay mismatch: archive.path_or_file",
    }
    assert "Traceback" not in completed.output


def test_cli_replay_reproduces_one_certifiable_crash_step(tmp_path, config) -> None:
    engine = EventEngine(config, crash_root=tmp_path, source_revision="a" * 40)
    agent = ScriptedEventFlowAgent()
    session = engine.start_episode(episode(33), agent)
    with pytest.raises(DynamicsError):
        while True:
            engine.step(session, agent)

    crash = next(tmp_path.glob("*.json"))
    completed = runner.invoke(app, ["replay", str(crash), "--json"])

    assert completed.exit_code == 0, completed.output
    payload = json.loads(completed.stdout)
    assert payload["replay_kind"] == "crash"
    assert payload["matched"] is True
    assert payload["timed_success"] is None
    assert payload["failure"] == {
        "code": "dynamics_error",
        "context": {"invariant": "internal_event_cap", "replay_certifiable": True},
        "message": "runtime execution failed",
    }


def test_cli_replay_refuses_an_unclassified_crash_before_checkpoint_access(tmp_path) -> None:
    manifest = CrashBundleManifest(
        bundle_id="unclassified",
        created_at_utc=datetime(2026, 9, 9, tzinfo=UTC),
        error={
            "code": "dynamics_error",
            "message": "runtime execution failed",
            "context": {"invariant": None, "replay_certifiable": False},
        },
        traceback_text="no approved project frames",
        context=CrashContext(checkpoint_ref="missing.safetensors"),
    )
    path = tmp_path / "unclassified.json"
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    completed = runner.invoke(app, ["replay", str(path), "--json"])

    assert completed.exit_code == 1
    assert json.loads(completed.stderr)["context"] == {"field": "crash.error"}
