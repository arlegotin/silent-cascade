"""Tiny process controls; these never produce scientific diagnostic evidence."""

import os
import sys
import time

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
