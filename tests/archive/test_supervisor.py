"""Closed real child jobs and pre-spawn output admission."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_supervisor_ledger(tmp_path):
    from conftest import _authority_boundary, _hide_host_authorities

    with _hide_host_authorities(_authority_boundary(tmp_path)):
        yield


def _run_test_bound_job(workspace, monkeypatch, script):
    import os
    import sys

    from silent_cascade.archive import supervisor
    from silent_cascade.archive.ledger import initialize_workspace_ledger
    from silent_cascade.archive.types import ArchivePolicy, JobOutputBounds
    from silent_cascade.train import pilot_offline

    policy = ArchivePolicy()
    initialize_workspace_ledger(workspace_root=workspace, policy=policy, baseline=())
    original = supervisor.subprocess.Popen
    monkeypatch.setattr(
        supervisor.subprocess,
        "Popen",
        lambda _command, **kwargs: original([sys.executable, "-B", "-c", script], **kwargs),
    )
    monkeypatch.setattr(pilot_offline, "resolve_offline_git", lambda: None)
    monkeypatch.setattr(
        pilot_offline, "offline_environment", lambda *_args, **_kwargs: dict(os.environ)
    )
    bounds = JobOutputBounds(
        job="report",
        request_sha256="1" * 64,
        policy_sha256="2" * 64,
        source_sha256="3" * 64,
        spool_bytes=65536,
        cache_bytes=65536,
        pinned_bytes=0,
        metadata_bytes=1024**2,
        scratch_bytes=65536,
        logs_bytes=65536,
        emergency_bytes=0,
    )
    return supervisor._supervise_bound_job(
        job="report",
        request={},
        decoded={},
        source="3" * 64,
        workspace_root=workspace,
        run_dir=workspace / "run",
        control_dir=workspace / "control",
        transport=None,
        policy=policy,
        output_bounds=bounds,
    )


def test_supervisor_scoped_final_check_follows_cleanup_while_owned(tmp_path, monkeypatch):
    from silent_cascade.archive import supervisor
    from silent_cascade.archive.ledger import _StorageBudget

    scans, cleaned = [], []
    authenticate = _StorageBudget.retained_charge
    close = supervisor._ArchiveServer.close
    serve = supervisor._ArchiveServer.serve_once

    def full_check(budget):
        scans.append(True)
        assert budget._state()["reservations"]
        if len(scans) == 2:
            assert cleaned == [True]
        return authenticate(budget)

    def cleanup(server):
        close(server)
        cleaned.append(True)

    def repeated_requests(server):
        from silent_cascade.archive.session import _publish

        _publish(
            server.control_dir,
            "request.json",
            {
                "schema_version": "phase4-r2-request-v1",
                "run_id": server.run_id,
                "session_id": server.identity["session_id"],
                "sequence": server.sequence + 1,
                "operation": "stop",
                "policy_sha256": server.identity["policy_sha256"],
                "unit_id": None,
                "payload": {},
                "pid": server.child_pid,
            },
            max_bytes=server.policy.page_bytes,
        )
        with server.budget.reserve(_scope=server.scope, scratch=1):
            return serve(server)

    monkeypatch.setattr(_StorageBudget, "retained_charge", full_check)
    monkeypatch.setattr(supervisor._ArchiveServer, "close", cleanup)
    monkeypatch.setattr(supervisor._ArchiveServer, "serve_once", repeated_requests)
    assert _run_test_bound_job(tmp_path, monkeypatch, "import time; time.sleep(0.2)") == 0
    assert len(scans) == 2


@pytest.mark.parametrize("fault", ["final-authentication", "context-exit", "live-growth"])
def test_supervisor_cannot_accept_failed_finite_boundary(tmp_path, monkeypatch, fault):
    from silent_cascade.archive import supervisor
    from silent_cascade.archive.ledger import StorageBlocked, _StorageBudget

    scans = []
    authenticate = _StorageBudget.retained_charge
    store = _StorageBudget._store
    serve = supervisor._ArchiveServer.serve_once

    def full_check(budget):
        scans.append(True)
        if len(scans) == 2 and fault == "final-authentication":
            raise StorageBlocked("retained mutation")
        return authenticate(budget)

    def release(budget, state):
        if scans and not state["reservations"] and fault == "context-exit":
            raise OSError("reservation exit failed")
        return store(budget, state)

    def growth(server):
        if fault == "live-growth":
            (server.budget.workspace / "scratch/overflow").write_bytes(b"x" * 65537)
        return serve(server)

    monkeypatch.setattr(_StorageBudget, "retained_charge", full_check)
    monkeypatch.setattr(_StorageBudget, "_store", release)
    monkeypatch.setattr(supervisor._ArchiveServer, "serve_once", growth)
    with pytest.raises((StorageBlocked, OSError)):
        _run_test_bound_job(tmp_path, monkeypatch, "import time; time.sleep(0.2)")
    assert len(scans) == (1 if fault == "live-growth" else 2)


@pytest.mark.parametrize("stage", ["term", "kill", "probe"])
def test_supervisor_waits_for_esrch_after_transient_group_eperm(tmp_path, monkeypatch, stage):
    import signal
    from types import SimpleNamespace

    from silent_cascade.archive import supervisor
    from silent_cascade.archive.types import ArchivePolicy

    probes = []
    target = {"term": signal.SIGTERM, "kill": signal.SIGKILL, "probe": 0}[stage]

    def signal_group(pid, signum):
        assert pid == 123456
        if signum == 0:
            probes.append(signum)
            if len(probes) == 1:
                raise PermissionError("transient empty group")
            raise ProcessLookupError
        if signum == target:
            raise PermissionError("transient empty group")

    def fail_stop(*_args):
        pytest.fail("transient group EPERM prevented bounded ESRCH proof")

    monkeypatch.setattr(supervisor.os, "killpg", signal_group)
    monkeypatch.setattr(supervisor, "_supervisor_fail_stop", fail_stop)
    monkeypatch.setattr(supervisor.time, "sleep", lambda _: None)
    child = SimpleNamespace(pid=123456, wait=lambda **_: 0)
    supervisor._stop_supervised_child(child, tmp_path, ArchivePolicy())
    assert len(probes) == 2


def test_supervisor_stops_private_descendant_after_leader_exit(tmp_path, monkeypatch):
    import os
    import signal
    from contextlib import suppress

    from silent_cascade.archive import supervisor
    from silent_cascade.archive.ledger import _StorageBudget
    from silent_cascade.archive.types import ArchivePolicy

    def unexpected_fail_stop(*_args):
        raise AssertionError("owned-group cleanup failed in native process test")

    monkeypatch.setattr(supervisor, "_supervisor_fail_stop", unexpected_fail_stop)
    signal_group = supervisor.os.killpg
    stopped_groups = set()

    def signal_owned_group(pid, signum):
        assert pid not in stopped_groups, "do not signal an already retired group ID"
        try:
            return signal_group(pid, signum)
        except ProcessLookupError:
            if signum == 0:
                stopped_groups.add(pid)
            raise

    monkeypatch.setattr(supervisor.os, "killpg", signal_owned_group)

    pid_file = tmp_path / "descendant-pid"
    script = (
        "import subprocess, sys; from pathlib import Path; "
        "child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(60)'], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        f"Path({str(pid_file)!r}).write_text(str(child.pid))"
    )
    try:
        assert _run_test_bound_job(tmp_path, monkeypatch, script) == 0
        pid = int(pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert not _StorageBudget(workspace=tmp_path, policy=ArchivePolicy())._state()[
            "reservations"
        ]
    finally:
        if pid_file.exists():
            with suppress(ProcessLookupError):
                os.kill(int(pid_file.read_text()), signal.SIGKILL)


def _supervisor_cleanup_failure_parent(workspace, boundary, fault):
    import os

    from conftest import _hide_host_authorities

    from silent_cascade.archive import supervisor

    with _hide_host_authorities(boundary), pytest.MonkeyPatch.context() as patch:

        def fail_signal(_pid, _signal):
            raise PermissionError("injected owned group cleanup failure")

        if fault == "signal":
            patch.setattr(supervisor.os, "killpg", fail_signal)
        else:
            patch.setattr(
                supervisor._ArchiveServer,
                "close",
                lambda _: (_ for _ in ()).throw(OSError("cleanup")),
            )
        try:
            _run_test_bound_job(workspace, patch, "pass")
        except BaseException:
            os._exit(71)


@pytest.mark.parametrize("fault", ["signal", "server"])
def test_supervisor_cleanup_failure_preserves_owned_reservation(tmp_path, fault):
    import multiprocessing

    from conftest import _authority_boundary

    from silent_cascade.archive.ledger import _StorageBudget
    from silent_cascade.archive.types import ArchivePolicy

    parent = multiprocessing.get_context("spawn").Process(
        target=_supervisor_cleanup_failure_parent,
        args=(tmp_path, _authority_boundary(tmp_path), fault),
    )
    parent.start()
    parent.join(15)
    assert parent.exitcode == 70
    records = _StorageBudget(workspace=tmp_path, policy=ArchivePolicy())._state()["reservations"]
    assert len(records) == 1
    assert next(iter(records.values()))["pid"] == parent.pid
    parent.close()


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


@pytest.mark.parametrize(
    "failure",
    ["admission", "corrupt", "missing", "startup", "no-space", "git-timeout", "git-nonzero"],
)
def test_admission_and_startup_return_blocked_without_losing_evidence(
    tmp_path, monkeypatch, capsys, failure
):
    import json
    import os
    import subprocess

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
    elif failure in {"git-timeout", "git-nonzero"}:
        from silent_cascade.train import pilot_offline

        def failed_git_probe():
            command = ["private-executable-secret", "--version"]
            if failure == "git-timeout":
                raise subprocess.TimeoutExpired(
                    command, 5, output=b"private-output-secret", stderr=b"private-error-secret"
                )
            raise subprocess.CalledProcessError(
                1, command, output=b"private-output-secret", stderr=b"private-error-secret"
            )

        monkeypatch.setattr(pilot_offline, "resolve_offline_git", failed_git_probe)
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
    captured = capsys.readouterr()
    assert "secret" not in captured.out + captured.err
    message = json.loads(captured.err.strip())
    assert message["status"] == "storage_blocked" and "retain" in message["action"]
    if failure in {"admission", "startup", "git-timeout", "git-nonzero"}:
        blocked = (control / "blocked.json").read_bytes()
        assert json.loads(blocked)["status"] == "storage_blocked"
        assert b"secret" not in blocked
    else:
        assert not (control / "blocked.json").exists()
    if failure == "corrupt":
        assert state.read_bytes() == b"{corrupt history"
    if failure == "missing":
        assert not state.exists() and state.with_name("retained-history.json").exists()
    if failure not in {"corrupt", "missing"}:
        assert json.loads(state.read_bytes())["reservations"] == {}


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
    original_check = _StorageBudget.check_scoped
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

    def exhaust_after_nested_spawn(budget, scope):
        result = original_check(budget, scope)
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
    monkeypatch.setattr(_StorageBudget, "check_scoped", exhaust_after_nested_spawn)
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
