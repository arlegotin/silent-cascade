"""Bounded parent process evidence; callers reserve publication space before launch.

This is not a child filesystem quota. No scientific modules are imported here.
"""

import os
import selectors
import subprocess
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class OfflineProcessLimits:
    stdout_bytes: int = 256 * 1024
    stderr_bytes: int = 256 * 1024
    record_bytes: int = 16 * 1024
    timeout_seconds: int = 180

    def __post_init__(self):
        for name, maximum in (
            ("stdout_bytes", 262144),
            ("stderr_bytes", 262144),
            ("record_bytes", 16384),
            ("timeout_seconds", 180),
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 < value <= maximum:
                raise ValueError("invalid offline process limit: " + name)


@dataclass(frozen=True)
class CapturedOfflineProcess:
    stdout: bytes
    stderr: bytes
    pid: int | None
    returncode: int | None
    reason: str | None
    eof: bool


def _join_failed_child(process):
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=0.25)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def capture_offline_process(command, *, cwd, env, limits):
    """Drain both pipes concurrently; retain only prefixes and join the exact child."""
    if type(limits) is not OfflineProcessLimits:
        raise ValueError("invalid offline process limits")
    process = None
    selector = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    reason = None
    eof = False
    deadline = time.monotonic() + limits.timeout_seconds
    try:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, RuntimeError):
            return CapturedOfflineProcess(b"", b"", None, None, "launch_failure", False)
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map() or process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reason = "timeout"
                break
            for key, _ in selector.select(min(remaining, 0.05)):
                chunk = os.read(key.fileobj.fileno(), 4096)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffer = buffers[key.data]
                available = getattr(limits, key.data + "_bytes") - len(buffer)
                buffer.extend(chunk[:available])
                if len(chunk) > available:
                    reason = "output_limit"
            if reason:
                break
        eof = not selector.get_map()
    except (KeyboardInterrupt, SystemExit):
        reason = "cancelled"
    except (OSError, ValueError):
        reason = "capture_failure"
    finally:
        if process is not None:
            if reason or process.poll() is None:
                _join_failed_child(process)
            else:
                process.wait()
            process.stdout.close()
            process.stderr.close()
        selector.close()
    return CapturedOfflineProcess(
        bytes(buffers["stdout"]),
        bytes(buffers["stderr"]),
        process.pid,
        process.returncode,
        reason or ("nonzero_exit" if process.returncode else None),
        eof,
    )
