"""Public replay command boundaries; private archive contents never print."""

import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime

import pytest
from conftest import CASES, bundle_for, episode
from typer.testing import CliRunner

from silent_cascade.cli import app
from silent_cascade.errors import DynamicsError
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.replay import (
    MAX_REPLAY_BYTES,
    write_replay_artifact,
)
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


def _crash(tmp_path, config):
    engine = EventEngine(config, crash_root=tmp_path, source_revision="a" * 40)
    agent = ScriptedEventFlowAgent()
    session = engine.start_episode(episode(33), agent)
    with pytest.raises(DynamicsError):
        while True:
            engine.step(session, agent)
    return next(tmp_path.glob("*.json"))


def _rewrite_json(path, mutate) -> None:
    payload = json.loads(path.read_bytes())
    mutate(payload)
    path.write_bytes(canonical_json_bytes(payload) + (b"\n" if path.suffix == ".json" else b""))


def _payload_digest(payload: dict) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    return hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()


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
    crash = _crash(tmp_path, config)
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


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda payload: payload["error"].update({"code": []}), "crash.error"),
        (lambda payload: payload["error"]["context"].update({"invariant": {}}), "crash.error"),
        (
            lambda payload: payload["error"]["context"].update({"replay_certifiable": 1}),
            "crash.error",
        ),
        (lambda payload: payload["error"]["context"].pop("replay_certifiable"), "crash.error"),
        (
            lambda payload: payload["context"].update({"checkpoint_ref": "../escape.safetensors"}),
            "checkpoint_ref",
        ),
        (
            lambda payload: payload["error"].update({"code": "time_order_error"}),
            "crash.failure_identity",
        ),
        (
            lambda payload: payload["error"]["context"].update(
                {"invariant": "same_kind_refractory"}
            ),
            "crash.failure_identity",
        ),
    ],
)
def test_cli_replay_refuses_malformed_or_nonmatching_crash_diagnostics(
    tmp_path, config, mutate, field
) -> None:
    crash = _crash(tmp_path, config)
    _rewrite_json(crash, mutate)

    completed = runner.invoke(app, ["replay", str(crash), "--json"])

    assert completed.exit_code == 1
    assert json.loads(completed.stderr)["context"] == {"field": field}
    assert "Traceback" not in completed.output


def test_cli_replay_refuses_missing_crash_checkpoint(tmp_path, config) -> None:
    crash = _crash(tmp_path, config)
    next(tmp_path.glob("*.safetensors")).unlink()

    completed = runner.invoke(app, ["replay", str(crash), "--json"])

    assert completed.exit_code == 1
    assert json.loads(completed.stderr)["code"] == "replay_error"
    assert "Traceback" not in completed.output


def test_installed_crash_replay_uses_fixed_protocol_outside_repository(tmp_path, config) -> None:
    crash = _crash(tmp_path, config)
    executable = shutil.which("silent-cascade")
    assert executable is not None
    outside = tmp_path / "outside"
    outside.mkdir()

    completed = subprocess.run(
        [executable, "replay", str(crash), "--json"],
        cwd=outside,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["replay_kind"] == "crash"


@pytest.mark.parametrize("kind", ["corrupt", "mismatch", "non_validation", "oversize"])
def test_cli_replay_refuses_invalid_episode_archive_matrix(tmp_path, config, kind) -> None:
    path, _, _ = _artifact(tmp_path, config)
    if kind == "corrupt":
        path.write_bytes(b"not-json")
    elif kind == "mismatch":
        payload = json.loads(path.read_bytes())
        payload["expected_result"]["score"]["reason"] = "incorrect"
        payload["payload_sha256"] = _payload_digest(payload)
        path.write_bytes(canonical_json_bytes(payload))
    elif kind == "non_validation":
        payload = json.loads(path.read_bytes())
        payload["access_class"] = "other"
        payload["payload_sha256"] = _payload_digest(payload)
        path.write_bytes(canonical_json_bytes(payload))
    else:
        path.write_bytes(b"x" * (MAX_REPLAY_BYTES + 1))

    completed = runner.invoke(app, ["replay", str(path), "--json"])

    assert completed.exit_code == 1
    assert json.loads(completed.stderr)["code"] == "replay_error"
    assert "Traceback" not in completed.output


def test_cli_replay_refuses_symlink_and_fifo_inputs(tmp_path, config) -> None:
    path, _, _ = _artifact(tmp_path, config)
    symlink = tmp_path / "replay-link.json"
    symlink.symlink_to(path)
    fifo = tmp_path / "replay.fifo"
    os.mkfifo(fifo)

    for invalid in (symlink, fifo):
        completed = runner.invoke(app, ["replay", str(invalid), "--json"])
        assert completed.exit_code == 1
        assert json.loads(completed.stderr)["code"] == "replay_error"
        assert "Traceback" not in completed.output


def test_cli_replay_verifies_the_same_bytes_it_hashes(tmp_path, config, monkeypatch) -> None:
    path, _, result = _artifact(tmp_path, config)
    replacement_dir = tmp_path / "replacement"
    replacement_dir.mkdir()
    replacement, _, _ = _artifact(replacement_dir, config)
    initial = path.read_bytes()
    replacement_bytes = replacement.read_bytes()

    import silent_cascade.cli as cli

    real_parse = cli.parse_replay_artifact_bytes

    def parse_then_replace(raw: bytes):
        parsed = real_parse(raw)
        path.write_bytes(replacement_bytes)
        return parsed

    monkeypatch.setattr(cli, "parse_replay_artifact_bytes", parse_then_replace)
    completed = runner.invoke(app, ["replay", str(path), "--json"])

    assert completed.exit_code == 0, completed.output
    payload = json.loads(completed.stdout)
    assert payload["artifact_sha256"] == hashlib.sha256(initial).hexdigest()
    assert payload["trace_sha256"] == result.trace.sha256
