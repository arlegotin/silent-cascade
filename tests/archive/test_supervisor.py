"""Closed real child jobs and pre-spawn output admission."""

import pytest


@pytest.mark.parametrize(
    "job,job_request",
    [
        ("shell", {"command": "true"}),
        ("report", {"output_dir": "../escape"}),
        ("verify", {"repo_root": "/tmp"}),
        (
            "pilot",
            {"config_path": "configs/base.yaml", "manifest_dir": "runs/data", "device": "cpu"},
        ),
        (
            "pilot",
            {
                "config_path": "configs/train/pilot.yaml",
                "manifest_dir": "runs/data",
                "device": "cuda",
            },
        ),
    ],
)
def test_closed_job_decoder_rejects_unknown_commands_and_unsafe_paths(tmp_path, job, job_request):
    from silent_cascade.archive.supervisor import _decode_job

    with pytest.raises(ValueError):
        _decode_job(job=job, request=job_request, run_dir=tmp_path)


def test_supervisor_runs_and_joins_real_closed_report_child(tmp_path):
    from conftest import DirectoryTransport

    from silent_cascade.archive.ledger import initialize_workspace_ledger
    from silent_cascade.archive.supervisor import package_source_sha256, supervise_job
    from silent_cascade.archive.types import ArchivePolicy, JobOutputBounds
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

    policy = ArchivePolicy()
    from silent_cascade.train.pilot_offline import measure_pilot_offline

    measure_pilot_offline(output_dir=tmp_path / "spool/offline")
    run = tmp_path / "spool/offline/run"
    control = tmp_path / "metadata/control"
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    bounds = JobOutputBounds(
        job="report",
        request_sha256=sha256_bytes(canonical_json_bytes({})),
        policy_sha256=sha256_bytes(canonical_json_bytes(policy)),
        source_sha256=package_source_sha256(),
        spool_bytes=1024**2,
        cache_bytes=0,
        pinned_bytes=0,
        metadata_bytes=1024**2,
        scratch_bytes=64 * 1024**2,
        logs_bytes=64 * 1024,
        emergency_bytes=0,
    )
    result = supervise_job(
        job="report",
        request={},
        workspace_root=tmp_path,
        run_dir=run,
        control_dir=control,
        transport=DirectoryTransport(tmp_path / "scratch/remote"),
        policy=policy,
        output_bounds=bounds,
    )
    assert result == 0
    report = run / "report/report.md"
    assert report.exists()
    assert "production learning gate not established" in report.read_text()
    assert not __import__("json").loads(
        (tmp_path / ".silent-cascade-storage/workspace.json").read_bytes()
    )["reservations"]


def test_supervisor_rejects_missing_or_mismatched_bounds_before_child(tmp_path):
    from silent_cascade.archive.supervisor import supervise_job
    from silent_cascade.archive.types import ArchivePolicy

    with pytest.raises(ValueError, match="bounds"):
        supervise_job(
            job="report",
            request={},
            workspace_root=tmp_path,
            run_dir=tmp_path / "run",
            control_dir=tmp_path / "control",
            transport=None,
            policy=ArchivePolicy(),
            output_bounds=None,
        )
    assert not (tmp_path / "run").exists()


def test_supervisor_binds_existing_run_before_reservation_snapshot(tmp_path, monkeypatch):
    from silent_cascade.archive import supervisor
    from silent_cascade.archive.ledger import _StorageBudget, initialize_workspace_ledger
    from silent_cascade.archive.types import ArchivePolicy, JobOutputBounds
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

    policy = ArchivePolicy(scratch_bytes=64 * 1024, chunk_bytes=8192)
    run, control = tmp_path / "ordinary-run", tmp_path / "ordinary-control"
    run.mkdir()
    existing = run / "retained.bin"
    existing.write_bytes(bytes(range(256)) * 512)
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    bounds = JobOutputBounds(
        job="report",
        request_sha256=sha256_bytes(canonical_json_bytes({})),
        policy_sha256=sha256_bytes(canonical_json_bytes(policy)),
        source_sha256=supervisor.package_source_sha256(),
        spool_bytes=4096,
        cache_bytes=0,
        pinned_bytes=0,
        metadata_bytes=1024**2,
        scratch_bytes=32768,
        logs_bytes=16384,
        emergency_bytes=0,
    )
    spawned = []
    original_popen = supervisor.subprocess.Popen

    def inspect_admission(*args, **kwargs):
        if len(args[0]) == 2 and args[0][1] == "--version":
            return original_popen(*args, **kwargs)
        state = _StorageBudget(workspace=tmp_path, policy=policy)._state()
        (record,) = state["reservations"].values()
        assert record["before"]["spool"] >= existing.stat().st_blocks * 512
        assert record["before"]["scratch"] < 32768
        spawned.append(True)
        raise OSError("injected local spawn failure")

    monkeypatch.setattr(supervisor.subprocess, "Popen", inspect_admission)
    assert (
        supervisor.supervise_job(
            job="report",
            request={},
            workspace_root=tmp_path,
            run_dir=run,
            control_dir=control,
            transport=None,
            policy=policy,
            output_bounds=bounds,
        )
        == 75
    )
    assert spawned == [True]
    assert existing.read_bytes() == bytes(range(256)) * 512


@pytest.mark.parametrize("failure", ["admission", "corrupt", "missing", "startup", "no-space"])
def test_admission_and_startup_return_blocked_without_losing_evidence(
    tmp_path, monkeypatch, capsys, failure
):
    import json
    import os

    from silent_cascade.archive import supervisor
    from silent_cascade.archive.ledger import initialize_workspace_ledger
    from silent_cascade.archive.types import ArchivePolicy, JobOutputBounds
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

    policy = ArchivePolicy()
    run, control = tmp_path / "run", tmp_path / "control"
    run.mkdir()
    control.mkdir()
    pending = run / "pending.bin"
    pending.write_bytes(b"never remove pending evidence")
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    state = tmp_path / ".silent-cascade-storage/workspace.json"
    if failure == "corrupt":
        state.write_bytes(b"{corrupt history")
    elif failure == "missing":
        state.rename(state.with_name("retained-history.json"))
    elif failure == "startup":
        (control / "storage-state.json").write_bytes(b"{}")
    elif failure == "no-space":
        values = list(os.statvfs(tmp_path))
        values[4] = 0
        monkeypatch.setattr(os, "statvfs", lambda _: os.statvfs_result(values))
    bounds = JobOutputBounds(
        job="report",
        request_sha256=sha256_bytes(canonical_json_bytes({})),
        policy_sha256=sha256_bytes(canonical_json_bytes(policy)),
        source_sha256=supervisor.package_source_sha256(),
        spool_bytes=policy.spool_bytes + 1 if failure == "admission" else 4096,
        cache_bytes=0,
        pinned_bytes=0,
        metadata_bytes=1024**2,
        scratch_bytes=1024**2,
        logs_bytes=16384,
        emergency_bytes=0,
    )
    monkeypatch.setattr(
        supervisor.subprocess, "Popen", lambda *a, **kw: pytest.fail("child started")
    )
    assert (
        supervisor.supervise_job(
            job="report",
            request={},
            workspace_root=tmp_path,
            run_dir=run,
            control_dir=control,
            transport=None,
            policy=policy,
            output_bounds=bounds,
        )
        == 75
    )
    assert pending.read_bytes() == b"never remove pending evidence"
    message = json.loads(capsys.readouterr().err.strip())
    assert message["status"] == "storage_blocked" and "retain" in message["action"]
    if failure in {"admission", "startup"}:
        assert json.loads((control / "blocked.json").read_bytes())["status"] == "storage_blocked"
    else:
        assert not (control / "blocked.json").exists()
    if failure == "corrupt":
        assert state.read_bytes() == b"{corrupt history"
    if failure == "missing":
        assert not state.exists() and state.with_name("retained-history.json").exists()


def test_supervisor_abort_stops_nested_diagnostic_without_touching_other_processes(
    tmp_path, monkeypatch
):
    import json
    import subprocess
    import sys
    import time
    from contextlib import suppress

    import psutil
    from conftest import DirectoryTransport

    from silent_cascade.archive import supervisor
    from silent_cascade.archive.ledger import (
        StorageBlocked,
        _StorageBudget,
        initialize_workspace_ledger,
    )
    from silent_cascade.archive.types import ArchivePolicy, JobOutputBounds
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
    from silent_cascade.train.pilot_offline import offline_environment

    policy = ArchivePolicy()
    run = tmp_path / "spool/run"
    control = tmp_path / "metadata/control"
    pending = control / "transfer-scratch/pending.bin"
    pending.parent.mkdir(parents=True)
    pending.write_bytes(b"retain interrupted transfer")
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    bounds = JobOutputBounds(
        job="report",
        request_sha256=sha256_bytes(canonical_json_bytes({})),
        policy_sha256=sha256_bytes(canonical_json_bytes(policy)),
        source_sha256=supervisor.package_source_sha256(),
        spool_bytes=64 * 1024**2,
        cache_bytes=0,
        pinned_bytes=0,
        metadata_bytes=1024**2,
        scratch_bytes=64 * 1024**2,
        logs_bytes=64 * 1024,
        emergency_bytes=0,
    )
    original_popen = subprocess.Popen
    original_check = _StorageBudget.check
    direct = []
    descendants = []
    deadline = time.monotonic() + 40

    def child_fixture(command, **kwargs):
        if len(command) == 2 and command[1] == "--version":
            return original_popen(command, **kwargs)
        # Keep real _child_main, source checking, boundary and exact nested diagnostic.
        # Replace only the scientific job body to avoid a complete pilot-checks run.
        assert command[2] == "-c" and command[3].endswith("_child_main()")
        command = [*command]
        command[3] = """
import sys
from pathlib import Path
from silent_cascade.archive import supervisor as s
from silent_cascade.train.pilot_offline import measure_pilot_offline, _PROGRAM
original = s.subprocess.Popen
def observe_nested(command, **kwargs):
    process = original(command, **kwargs)
    if len(command) == 5 and command[3] == _PROGRAM:
        (Path(sys.argv[1]) / 'test-nested.pid').write_text(str(process.pid))
    return process
s.subprocess.Popen = observe_nested
s._execute_job = lambda job, request, run: measure_pilot_offline(output_dir=run / 'diagnostic')
s._child_main()
"""
        process = original_popen(command, **kwargs)
        direct.append(process)
        return process

    def exhaust_after_nested_spawn(budget):
        result = original_check(budget)
        assert time.monotonic() < deadline, "nested diagnostic did not start"
        marker = control / "test-nested.pid"
        if not descendants and marker.exists() and (pid := marker.read_text()).isdigit():
            descendants.append(psutil.Process(int(pid)))
            raise StorageBlocked("storage_blocked: injected exhaustion with active diagnostic")
        return result

    unrelated = original_popen(
        [sys.executable, "-B", "-c", "import time; time.sleep(60)"],
        env=offline_environment(tmp_path),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    monkeypatch.setattr(supervisor.subprocess, "Popen", child_fixture)
    monkeypatch.setattr(_StorageBudget, "check", exhaust_after_nested_spawn)
    try:
        result = supervisor.supervise_job(
            job="report",
            request={},
            workspace_root=tmp_path,
            run_dir=run,
            control_dir=control,
            transport=DirectoryTransport(tmp_path / "scratch/remote"),
            policy=policy,
            output_bounds=bounds,
        )
        assert result == 75
        assert len(descendants) == 1
        assert direct[0].poll() is not None
        assert unrelated.poll() is None
        assert pending.read_bytes() == b"retain interrupted transfer"
        assert (run / "diagnostic/intent.json").exists()
        assert not json.loads((tmp_path / ".silent-cascade-storage/workspace.json").read_bytes())[
            "reservations"
        ]
        with pytest.raises(psutil.NoSuchProcess):
            psutil.Process(descendants[0].pid)
    finally:
        # Test-owned cleanup also runs on RED, without broad process matching.
        for process in descendants:
            if process.is_running():
                process.kill()
                with suppress(psutil.TimeoutExpired):
                    process.wait(timeout=5)
        for process in [*direct, unrelated]:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
