"""Tiny process controls; these never produce scientific diagnostic evidence."""

import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_existing_intent_blocks_launch(tmp_path, monkeypatch):
    from silent_cascade.train import pilot_offline

    (tmp_path / "intent.json").write_bytes(b"{}")

    def launch_tripwire():
        raise AssertionError("Git resolution reached before incomplete-attempt fence")

    monkeypatch.setattr(pilot_offline, "resolve_offline_git", launch_tripwire)
    with pytest.raises(ValueError, match="offline process"):
        pilot_offline.measure_pilot_offline(output_dir=tmp_path)
    assert (tmp_path / "intent.json").read_bytes() == b"{}"


def test_capture_bounds_both_pipes():
    from silent_cascade.train.pilot_offline_process import (
        OfflineProcessLimits,
        capture_offline_process,
    )

    result = capture_offline_process(
        [sys.executable, "-B", "-c", "import os; os.write(1,b'a'*65); os.write(2,b'b'*65)"],
        cwd=os.getcwd(),
        env=dict(os.environ),
        limits=OfflineProcessLimits(stdout_bytes=64, stderr_bytes=64, timeout_seconds=2),
    )
    assert result.reason == "output_limit"
    assert result.stdout == b"a" * 64
    assert len(result.stderr) <= 64
    assert result.returncode is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(result.pid, os.WNOHANG)


@pytest.mark.parametrize("descriptor", [1, 2])
def test_capture_bounds_each_pipe(descriptor):
    from silent_cascade.train.pilot_offline_process import (
        OfflineProcessLimits,
        capture_offline_process,
    )

    result = capture_offline_process(
        [sys.executable, "-B", "-c", f"import os; os.write({descriptor},b'x'*65)"],
        cwd=os.getcwd(),
        env=dict(os.environ),
        limits=OfflineProcessLimits(stdout_bytes=64, stderr_bytes=64, timeout_seconds=2),
    )
    assert result.reason == "output_limit"
    assert (result.stdout if descriptor == 1 else result.stderr) == b"x" * 64
    assert result.returncode is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(result.pid, os.WNOHANG)


@pytest.mark.parametrize("code,reason", [(0, None), (7, "nonzero_exit")])
def test_capture_preserves_exit_and_eof(code, reason):
    from silent_cascade.train.pilot_offline_process import (
        OfflineProcessLimits,
        capture_offline_process,
    )

    result = capture_offline_process(
        [sys.executable, "-B", "-c", f"import os; os.write(1,b'ok'); os._exit({code})"],
        cwd=os.getcwd(),
        env=dict(os.environ),
        limits=OfflineProcessLimits(timeout_seconds=2),
    )
    assert result.reason == reason
    assert result.returncode == code
    assert result.stdout == b"ok"
    assert result.stderr == b""
    assert result.eof
    with pytest.raises(ChildProcessError):
        os.waitpid(result.pid, os.WNOHANG)


def test_capture_timeout_joins_child():
    from silent_cascade.train.pilot_offline_process import (
        OfflineProcessLimits,
        capture_offline_process,
    )

    started = time.monotonic()
    result = capture_offline_process(
        [sys.executable, "-B", "-c", "import time; time.sleep(10)"],
        cwd=os.getcwd(),
        env=dict(os.environ),
        limits=OfflineProcessLimits(timeout_seconds=1),
    )
    assert result.reason == "timeout"
    assert result.returncode is not None
    assert time.monotonic() - started < 4
    with pytest.raises(ChildProcessError):
        os.waitpid(result.pid, os.WNOHANG)


def test_capture_launch_failure_has_no_invented_identity():
    from silent_cascade.train.pilot_offline_process import (
        OfflineProcessLimits,
        capture_offline_process,
    )

    result = capture_offline_process(
        ["/nonexistent-offline-process-control"],
        cwd=os.getcwd(),
        env=dict(os.environ),
        limits=OfflineProcessLimits(),
    )
    assert result.reason == "launch_failure"
    assert result.returncode is None
    assert result.pid is None
    assert not result.eof


def test_capture_cancellation_joins_child(monkeypatch):
    from silent_cascade.train import pilot_offline_process as process

    original = process.selectors.DefaultSelector.select
    interrupted = False

    def interrupt_once(self, timeout=None):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original(self, timeout)

    monkeypatch.setattr(process.selectors.DefaultSelector, "select", interrupt_once)
    result = process.capture_offline_process(
        [sys.executable, "-B", "-c", "import time; time.sleep(10)"],
        cwd=os.getcwd(),
        env=dict(os.environ),
        limits=process.OfflineProcessLimits(timeout_seconds=2),
    )
    assert result.reason == "cancelled"
    assert result.returncode is not None
    with pytest.raises(ChildProcessError):
        os.waitpid(result.pid, os.WNOHANG)


@pytest.mark.parametrize(
    "override",
    [
        {"stdout_bytes": 0},
        {"stderr_bytes": -1},
        {"record_bytes": True},
        {"timeout_seconds": 1.5},
        {"stdout_bytes": 262145},
        {"stderr_bytes": 262145},
        {"record_bytes": 16385},
        {"timeout_seconds": 181},
        {"unknown": 1},
    ],
)
def test_process_limits_reject_invalid_or_enlarged_values(override):
    from silent_cascade.train.pilot_offline_process import OfflineProcessLimits

    with pytest.raises((ValueError, TypeError)):
        OfflineProcessLimits(**override)


def test_reuse_rejects_incomplete_before_authentication(tmp_path, monkeypatch):
    from silent_cascade.train import pilot_checks

    output = tmp_path / "final/offline"
    output.mkdir(parents=True)
    (output / "intent.json").write_bytes(b"{}")

    def execution_tripwire(*args, **kwargs):
        raise AssertionError("execution reached before offline process fence")

    monkeypatch.setattr(pilot_checks, "authenticate_run", execution_tripwire)
    monkeypatch.setattr(pilot_checks, "_evaluate", execution_tripwire)
    monkeypatch.setattr(pilot_checks, "_continuations", execution_tripwire)
    with pytest.raises(ValueError, match="offline process"):
        pilot_checks._run_pilot_checks_owned(
            run_dir=tmp_path, config=None, output_path=tmp_path / "gate.json"
        )


def _control_intent(root):
    """Operational metadata only, with no scientific report or metrics."""
    return dict(
        schema_version="phase4-offline-process-intent-v1",
        attempt="a" * 32,
        output_root=str(root),
        bootstrap_sha256="b" * 64,
        python_executable=sys.executable,
        python_sha256="c" * 64,
        source_commit="d" * 40,
        executed_source_sha256="e" * 64,
        limits=dict(stdout_bytes=64, stderr_bytes=64, record_bytes=4096, timeout_seconds=2),
    )


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    [
        {"status": "completed", "returncode": 7},
        {"status": "completed", "reason": "timeout"},
        {"status": "completed", "eof": False},
        {"status": "completed", "offline_sha256": None},
        {"status": "failed", "reason": None},
        {"status": "failed", "reason": "launch_failure", "pid": None, "returncode": 0},
        {"status": "failed", "reason": "unknown"},
        {"stdout_bytes": True},
    ],
)
def test_terminal_record_rejects_impossible_completion(mutation):
    from silent_cascade.train.pilot_offline_process import OfflineProcessResult

    value = dict(
        schema_version="phase4-offline-process-result-v1",
        status="completed",
        intent_sha256="a" * 64,
        attempt="b" * 32,
        pid=123,
        returncode=0,
        reason=None,
        eof=True,
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_sha256=_digest(b""),
        stderr_sha256=_digest(b""),
        offline_sha256="c" * 64,
    )
    with pytest.raises(ValueError):
        OfflineProcessResult.model_validate_json(_encoded(value | mutation))


def test_marked_report_cannot_downgrade_to_legacy():
    from silent_cascade.train.pilot_offline import parse_offline_report

    # Deliberately invalid sentinel; this must never become scientific evidence.
    with pytest.raises(ValueError):
        parse_offline_report(b'{"evidence_kind":"offline_smoke_diagnostic_v2"}')


class ProcessControlContext:
    """One retained control payload lease at a time; no scientific corpus."""

    def __init__(self, run, backing, names):
        self.run_dir = run
        self.backing = backing
        self.names = names
        self.active = self.maximum = 0
        self.requests = []
        self.late_failure = False
        self.allowed = None

    def entries(self):
        for name in self.names:
            raw = (self.backing / name).read_bytes()
            yield SimpleNamespace(path=name, bytes=len(raw), sha256=_digest(raw))
        if self.late_failure:
            raise ValueError("late process inventory failure")

    @contextmanager
    def _leased(self, payload):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        self.requests.append(payload["path"])
        self.allowed = (self.backing / payload["path"]).absolute()
        try:
            yield SimpleNamespace(local_root=self.backing)
        finally:
            self.active -= 1
            self.allowed = None

    @contextmanager
    def guarded_reads(self, monkeypatch):
        entries = tuple(self.entries())
        original_open, original_path_open = os.open, Path.open
        descriptors = {}

        def authorize(path, *, directory=False):
            if path.is_relative_to(self.backing):
                assert self.active == 1 and self.allowed is not None, "read outside process lease"
                assert path == self.allowed or (directory and self.allowed.is_relative_to(path)), (
                    "undeclared process payload read"
                )

        def path_open(path, *args, **kwargs):
            authorize(path.absolute())
            return original_path_open(path, *args, **kwargs)

        def descriptor_open(path, flags, mode=0o777, *, dir_fd=None):
            target = Path(os.fsdecode(path))
            if not target.is_absolute() and dir_fd is not None:
                target = descriptors[dir_fd] / target
            target = target.absolute()
            directory = bool(flags & os.O_DIRECTORY)
            authorize(target, directory=directory)
            fd = original_open(path, flags, mode, dir_fd=dir_fd)
            if directory:
                descriptors[fd] = target
            return fd

        with monkeypatch.context() as patch:
            patch.setattr(self, "entries", lambda: iter(entries))
            patch.setattr(Path, "open", path_open)
            patch.setattr(os, "open", descriptor_open)
            yield
        assert self.active == 0


def _cold_process_controls(tmp_path, *, failed=False):
    run, backing = tmp_path / "run", tmp_path / "backing"
    output = backing / "final/offline"
    intent = _encoded(_control_intent(run / "final/offline"))
    # A terminal process control, intentionally lacking any offline.json report.
    result = _encoded(
        dict(
            schema_version="phase4-offline-process-result-v1",
            status="failed" if failed else "completed",
            intent_sha256=_digest(intent),
            attempt="a" * 32,
            pid=123,
            returncode=7 if failed else 0,
            reason="nonzero_exit" if failed else None,
            eof=True,
            stdout_bytes=2,
            stderr_bytes=0,
            stdout_sha256=_digest(b"ok"),
            stderr_sha256=_digest(b""),
            offline_sha256=None if failed else _digest(b"invalid"),
        )
    )
    payloads = (
        ("intent.json", intent),
        ("process-result.json", result),
        ("stdout.txt", b"ok"),
        ("stderr.txt", b""),
    )
    assert len(intent) <= 4096 and len(result) <= 4096
    assert sum(len(raw) for _, raw in payloads) <= 8194
    run.mkdir()
    output.mkdir(parents=True)
    for name, raw in payloads:
        (output / name).write_bytes(raw)
    names = [
        "final/offline/" + name
        for name in ("intent.json", "process-result.json", "stdout.txt", "stderr.txt")
    ]
    return run, output, ProcessControlContext(run, backing, names)


def test_cold_process_missing_report_is_unavailable_and_releases_lease(tmp_path):
    from silent_cascade.train.pilot_offline_process import read_offline_process_outcome

    run, _, context = _cold_process_controls(tmp_path)
    unavailable = []
    assert (
        read_offline_process_outcome(
            run, report=None, evidence_context=context, unavailable=unavailable
        )
        is None
    )
    assert "offline.process:offline.json" in unavailable
    assert context.maximum == 1 and context.active == 0
    assert set(context.requests) == set(context.names)


@pytest.mark.parametrize("missing", ["intent.json", "stderr.txt"])
def test_cold_process_checks_available_log_despite_missing_member(tmp_path, missing):
    from silent_cascade.train.pilot_offline_process import read_offline_process_outcome

    run, output, context = _cold_process_controls(tmp_path)
    context.names.remove("final/offline/" + missing)
    (output / "stdout.txt").write_bytes(b"NO")
    with pytest.raises(ValueError, match=r"offline process.*stdout"):
        read_offline_process_outcome(run, report=None, evidence_context=context, unavailable=[])
    assert context.maximum == 1 and context.active == 0


def test_failed_process_cannot_hide_behind_missing_report(tmp_path):
    from silent_cascade.train.pilot_offline_process import read_offline_process_outcome

    run, _, context = _cold_process_controls(tmp_path, failed=True)
    with pytest.raises(ValueError, match=r"offline process.*failed"):
        read_offline_process_outcome(run, report=None, evidence_context=context, unavailable=[])
    assert context.active == 0


def test_process_reader_exhausts_inventory_before_payloads(tmp_path):
    from silent_cascade.train.pilot_offline_process import read_offline_process_outcome

    run, _, context = _cold_process_controls(tmp_path)
    context.late_failure = True
    with pytest.raises(ValueError, match="late process inventory"):
        read_offline_process_outcome(run, report=None, evidence_context=context)
    assert context.requests == [] and context.active == 0


def test_process_reader_rejects_wrong_run_root_before_inventory(tmp_path):
    from silent_cascade.train.pilot_offline_process import read_offline_process_outcome

    run, _, context = _cold_process_controls(tmp_path)
    with pytest.raises(ValueError, match="root"):
        read_offline_process_outcome(run / "wrong", report=None, evidence_context=context)
    assert context.requests == [] and context.active == 0


def _measurement_control(monkeypatch, output, *, launch_failure=False):
    """Replace only the scientific command with a real invalid-sentinel child."""
    from silent_cascade.train import pilot_offline

    original = pilot_offline.subprocess.Popen
    launched = []
    monkeypatch.setattr(
        pilot_offline,
        "_BOUNDARY_GIT_PIN",
        pilot_offline.OfflineGitPin(sys.executable, "2.45.0", "a" * 64),
    )
    monkeypatch.setattr(
        pilot_offline,
        "_offline_source_identity",
        lambda: ("d" * 40, "e" * 64),
        raising=False,
    )

    def tiny_child(command, *args, **kwargs):
        assert command[:4] == [sys.executable, "-B", "-c", pilot_offline._PROGRAM]
        assert command[4] == str(output)
        assert (output / "intent.json").is_file()
        if launch_failure:
            raise OSError("control launch denied")
        command = [
            sys.executable,
            "-B",
            "-c",
            "import os,sys; from pathlib import Path; "
            "Path(sys.argv[1],'offline.json').write_bytes(b'invalid'); os.write(1,b'ok')",
            str(output),
        ]
        process = original(command, *args, **kwargs)
        launched.append(process)
        return process

    monkeypatch.setattr(pilot_offline.subprocess, "Popen", tiny_child)
    return pilot_offline, launched


def test_measurement_persists_failed_invalid_report_and_fences_reuse(tmp_path, monkeypatch):
    output = tmp_path / "offline"
    module, launched = _measurement_control(monkeypatch, output)
    with pytest.raises(ValueError):
        module.measure_pilot_offline(output_dir=output)
    assert (output / "process-result.json").exists(), "failed process terminal record missing"
    record = json.loads((output / "process-result.json").read_bytes())
    assert record["status"] == "failed" and record["reason"] == "invalid_report"
    assert record["returncode"] == 0
    assert record["offline_sha256"] is None
    assert (output / "stdout.txt").read_bytes() == b"ok"
    before = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="offline process"):
        module.measure_pilot_offline(output_dir=output)
    assert before == {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
    assert len(launched) == 1 and launched[0].poll() == 0
    with pytest.raises(ChildProcessError):
        os.waitpid(launched[0].pid, os.WNOHANG)


def test_measurement_records_launch_failure_without_return_code(tmp_path, monkeypatch):
    output = tmp_path / "offline"
    module, launched = _measurement_control(monkeypatch, output, launch_failure=True)
    with pytest.raises(ValueError, match="offline process"):
        module.measure_pilot_offline(output_dir=output)
    record = json.loads((output / "process-result.json").read_bytes())
    assert record["reason"] == "launch_failure"
    assert record["returncode"] is None and record["pid"] is None
    assert launched == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("bootstrap_sha256", "0" * 64),
        ("python_executable", "/other/python"),
        ("output_root", "/other/output"),
        ("python_sha256", "0" * 64),
        ("source_commit", "0" * 40),
        ("executed_source_sha256", "0" * 64),
        (
            "limits",
            dict(stdout_bytes=262145, stderr_bytes=64, record_bytes=4096, timeout_seconds=2),
        ),
    ],
)
def test_bootstrap_rejects_changed_launch_authority(tmp_path, monkeypatch, field, value):
    from silent_cascade.train import pilot_offline
    from silent_cascade.train.pilot_offline import validate_offline_launch

    monkeypatch.setattr(pilot_offline, "_offline_source_identity", lambda: ("d" * 40, "e" * 64))
    intent = _control_intent(tmp_path)
    with open(sys.executable, "rb") as executable:
        python_sha256 = hashlib.file_digest(executable, "sha256").hexdigest()
    intent.update(
        bootstrap_sha256=_digest(pilot_offline._PROGRAM.encode()), python_sha256=python_sha256
    )
    intent[field] = value
    raw = _encoded(intent)
    assert len(raw) <= 4096
    (tmp_path / "intent.json").write_bytes(raw)
    with pytest.raises(ValueError, match="offline process"):
        validate_offline_launch(tmp_path, _digest(raw))


@pytest.mark.parametrize("target", ["intent_sha256", "attempt", "stdout_bytes"])
def test_process_reader_rejects_changed_result_binding(tmp_path, target):
    from silent_cascade.train.pilot_offline_process import read_offline_process_outcome

    run, output, context = _cold_process_controls(tmp_path)
    result = json.loads((output / "process-result.json").read_bytes())
    result[target] = {"intent_sha256": "0" * 64, "attempt": "0" * 32, "stdout_bytes": 3}[target]
    raw = _encoded(result)
    assert len(raw) <= 4096
    (output / "process-result.json").write_bytes(raw)
    with pytest.raises(ValueError, match="offline process"):
        read_offline_process_outcome(run, report=None, evidence_context=context)
    assert context.active == 0


def test_collection_fences_incomplete_before_authentication(tmp_path, monkeypatch):
    from silent_cascade.train import pilot_evidence

    output = tmp_path / "final/offline"
    output.mkdir(parents=True)
    (output / "intent.json").write_bytes(b"{}")

    def execution_tripwire(*args, **kwargs):
        raise AssertionError("collection reached authentication before offline process fence")

    monkeypatch.setattr(pilot_evidence, "authenticate_run", execution_tripwire)
    with pytest.raises(ValueError, match="offline process"):
        pilot_evidence.collect_pilot_evidence(
            run_dir=tmp_path, config=None, output_path=tmp_path / "gate.json"
        )


def test_capture_cancellation_before_launch_has_no_child(monkeypatch):
    from silent_cascade.train import pilot_offline_process as module

    def cancelled(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(module.subprocess, "Popen", cancelled)
    captured = module.capture_offline_process(
        [sys.executable, "-B", "-c", "pass"],
        cwd=os.getcwd(),
        env=dict(os.environ),
        limits=module.OfflineProcessLimits(),
    )
    assert captured.reason == "cancelled"
    assert captured.pid is None and captured.returncode is None


@pytest.mark.parametrize("launch_failure", [False, True])
def test_log_publication_failure_never_publishes_completion(tmp_path, monkeypatch, launch_failure):
    from silent_cascade.train import pilot_data

    output = tmp_path / "offline"
    module, launched = _measurement_control(monkeypatch, output, launch_failure=launch_failure)
    original = pilot_data._publish_pilot_bytes

    def denied(path, raw):
        if path.name == "stdout.txt":
            raise OSError("control log publication denied")
        original(path, raw)

    monkeypatch.setattr(pilot_data, "_publish_pilot_bytes", denied)
    with pytest.raises(ValueError, match="offline process"):
        module.measure_pilot_offline(output_dir=output)
    record = json.loads((output / "process-result.json").read_bytes())
    assert record["status"] == "failed"
    assert record["reason"] == ("launch_failure" if launch_failure else "publication_failure")
    assert record["offline_sha256"] is None
    if launch_failure:
        assert launched == []
        assert record["pid"] is None and record["returncode"] is None
    else:
        assert len(launched) == 1 and launched[0].poll() == 0


def test_cold_process_uses_only_active_declared_payload_lease(tmp_path, monkeypatch):
    from silent_cascade.train.pilot_offline_process import read_offline_process_outcome

    run, output, context = _cold_process_controls(tmp_path)
    with context.guarded_reads(monkeypatch):
        assert (
            read_offline_process_outcome(run, report=None, evidence_context=context, unavailable=[])
            is None
        )
        with pytest.raises(AssertionError, match="outside process lease"):
            (output / "stdout.txt").read_bytes()
    assert context.maximum == 1 and context.active == 0


def test_advertised_process_lease_failure_is_not_unavailability(tmp_path, monkeypatch):
    from silent_cascade.train.pilot_offline_process import read_offline_process_outcome

    run, _, context = _cold_process_controls(tmp_path)

    @contextmanager
    def failed(payload):
        raise ValueError("advertised process lease failed")
        yield  # pragma: no cover

    monkeypatch.setattr(context, "_leased", failed)
    with pytest.raises(ValueError, match="advertised process lease failed"):
        read_offline_process_outcome(run, report=None, evidence_context=context, unavailable=[])
    assert context.active == 0


def test_current_process_parser_rejects_genuine_historical_v1():
    from silent_cascade.train.pilot_offline import parse_offline_report

    # Read-only historical bytes; no copied corpus and no new scientific result.
    report = Path(__file__).resolve().parents[2] / (
        ".superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-4/"
        "cover-pilot-final/test_real_smoke_workflow_reuse0/repo/runs/smoke/final/offline/offline.json"
    )
    if not report.is_file():
        pytest.skip("private historical offline report unavailable")
    raw = report.read_bytes()
    assert parse_offline_report(raw).evidence_kind == "offline_smoke_diagnostic"
    with pytest.raises(ValueError, match="offline process"):
        parse_offline_report(raw, require_process=True)


def test_fresh_nested_audit_denies_changed_launch_authority(tmp_path):
    from silent_cascade.train.pilot_offline import offline_environment
    from silent_cascade.train.pilot_offline_process import (
        OfflineProcessLimits,
        capture_offline_process,
    )

    program = r"""
import subprocess, sys
from pathlib import Path
from silent_cascade.train import pilot_offline as module
root = Path(sys.argv[1])
repo = Path(module.__file__).resolve().parents[3]
module.install_offline_boundary(workspace_root=root)
output = root / 'nested'
argv = [sys.executable, '-B', '-c', module._PROGRAM, str(output), 'a' * 64]
environment = module.offline_environment(output)
module._PENDING_OFFLINE_LAUNCH = (str(output), 'a' * 64)
mutations = [
    (argv[:4] + [str(root / 'other'), 'a' * 64], module.offline_environment(root / 'other')),
    (argv[:5] + ['b' * 64], environment),
    (argv, dict(environment, PYTHONPATH='/untrusted')),
    (argv[:3] + [module._PROGRAM + ' ', *argv[4:]], environment),
]
for command, env in mutations:
    try:
        child = subprocess.Popen(command, cwd=repo, env=env)
    except RuntimeError:
        pass
    else:
        child.wait(timeout=3)
        raise AssertionError('changed nested authority admitted')
module._PENDING_OFFLINE_LAUNCH = None
try:
    child = subprocess.Popen(argv, cwd=repo, env=environment)
except RuntimeError:
    pass
else:
    child.wait(timeout=3)
    raise AssertionError('unbound nested launch admitted')
print('denied:5')
"""
    captured = capture_offline_process(
        [sys.executable, "-B", "-c", program, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        env=offline_environment(tmp_path),
        limits=OfflineProcessLimits(stdout_bytes=256, stderr_bytes=4096, timeout_seconds=10),
    )
    assert captured.returncode == 0, captured.stderr.decode()
    assert captured.stdout == b"denied:5\n"
    assert captured.reason is None and captured.eof
    with pytest.raises(ChildProcessError):
        os.waitpid(captured.pid, os.WNOHANG)


def test_broken_offline_root_cannot_admit_new_execution(tmp_path):
    from silent_cascade.train.pilot_offline_process import require_completed_offline_process

    (tmp_path / "final").mkdir()
    (tmp_path / "final/offline").symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(ValueError, match="offline process"):
        require_completed_offline_process(tmp_path)


def test_broken_process_member_is_corruption_not_unavailability(tmp_path):
    from silent_cascade.errors import ArtifactIntegrityError
    from silent_cascade.train.pilot_offline_process import read_offline_process_outcome

    output = tmp_path / "final/offline"
    output.mkdir(parents=True)
    (output / "intent.json").symlink_to(tmp_path / "missing")
    with pytest.raises((ValueError, ArtifactIntegrityError)):
        read_offline_process_outcome(tmp_path, report=None, unavailable=[])
