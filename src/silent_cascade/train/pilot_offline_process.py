"""Bounded parent process evidence; callers reserve publication space before launch.

This is not a child filesystem quota. No scientific modules are imported here.
"""

import os
import selectors
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from silent_cascade.errors import ArtifactIntegrityError
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.validation import StrictModel


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


Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Attempt = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]


class OfflineProcessIntent(StrictModel):
    schema_version: Literal["phase4-offline-process-intent-v1"]
    attempt: Attempt
    output_root: str = Field(min_length=1, max_length=4096)
    bootstrap_sha256: Hash
    python_executable: str = Field(min_length=1, max_length=4096)
    python_sha256: Hash
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    executed_source_sha256: Hash
    limits: dict[str, int]

    @model_validator(mode="after")
    def closed_limits(self):
        if set(self.limits) != {"stdout_bytes", "stderr_bytes", "record_bytes", "timeout_seconds"}:
            raise ValueError("offline process intent limits differ")
        OfflineProcessLimits(**self.limits)
        if any(
            not Path(path).is_absolute() or ".." in Path(path).parts
            for path in (self.output_root, self.python_executable)
        ):
            raise ValueError("offline process intent paths must be absolute")
        return self


class OfflineProcessResult(StrictModel):
    schema_version: Literal["phase4-offline-process-result-v1"]
    status: Literal["completed", "failed"]
    intent_sha256: Hash
    attempt: Attempt
    pid: int | None = Field(ge=1)
    returncode: int | None
    reason: (
        Literal[
            "output_limit",
            "timeout",
            "nonzero_exit",
            "launch_failure",
            "cancelled",
            "capture_failure",
            "invalid_report",
            "publication_failure",
        ]
        | None
    )
    eof: bool
    stdout_bytes: int = Field(ge=0, le=262144)
    stderr_bytes: int = Field(ge=0, le=262144)
    stdout_sha256: Hash
    stderr_sha256: Hash
    offline_sha256: Hash | None

    @model_validator(mode="after")
    def consistent_terminal_state(self):
        if self.status == "completed":
            if (
                self.returncode != 0
                or self.pid is None
                or self.reason is not None
                or not self.eof
                or self.offline_sha256 is None
            ):
                raise ValueError("offline process completion is inconsistent")
        elif self.reason is None or self.offline_sha256 is not None:
            raise ValueError("offline process failure is inconsistent")
        if (self.pid is None) != (self.returncode is None):
            raise ValueError("offline process child identity/exit differs")
        if self.reason == "launch_failure" and (self.pid is not None or self.eof):
            raise ValueError("offline process failed launch has child identity")
        if self.pid is None and self.reason not in {"launch_failure", "cancelled"}:
            raise ValueError("offline process child identity missing")
        if self.reason == "nonzero_exit" and self.returncode == 0:
            raise ValueError("offline process nonzero exit is zero")
        return self


_PROCESS_NAMES = ("intent.json", "process-result.json", "stdout.txt", "stderr.txt", "offline.json")


def _parse_record(raw, model):
    from silent_cascade.train.evidence_types import decode_json

    try:
        return model.model_validate_json(canonical_json_bytes(decode_json(raw, limit=16384)))
    except (ValueError, TypeError, ArtifactIntegrityError) as error:
        raise ValueError("invalid offline process record") from error


def read_offline_process_outcome(run_dir, *, report, evidence_context=None, unavailable=None):
    """Verify all available process members, retaining only one cold lease at a time.

    Recorded machine paths identify the original launch; cold verification can run
    on another machine or after export. Only the logical session root is current.
    """
    from silent_cascade.archive.readers import evidence_path, logical_root
    from silent_cascade.train.evidence_types import decode_json, read_bytes
    from silent_cascade.train.pilot_offline import parse_offline_report

    if evidence_context is not None and (
        run_dir is None or logical_root(run_dir, evidence_context) != "."
    ):
        raise ValueError("offline process evidence context root differs")
    if run_dir is not None and (run_dir / "final/offline").is_symlink():
        raise ValueError("offline process root cannot be a symlink")
    unavailable = [] if unavailable is None else unavailable
    prefix = "final/offline/"
    indexed = {}
    if evidence_context is not None:
        for entry in evidence_context.entries():
            if entry.path in {prefix + name for name in _PROCESS_NAMES}:
                if entry.path in indexed:
                    raise ValueError("offline process duplicate inventory member")
                indexed[entry.path] = entry

    def read(name, limit):
        logical = prefix + name
        if run_dir is None or not (
            (run_dir / logical).exists() or (run_dir / logical).is_symlink() or logical in indexed
        ):
            return None
        with evidence_path(run_dir, logical, evidence_context=evidence_context) as local:
            try:
                raw = read_bytes(local, limit=limit)
            except OSError as error:
                raise ValueError("offline process unreadable member: " + name) from error
        if logical in indexed and (
            indexed[logical].bytes != len(raw) or indexed[logical].sha256 != sha256_bytes(raw)
        ):
            raise ValueError("offline process inventory differs: " + name)
        return raw

    intent_raw = read("intent.json", 16384)
    result_raw = read("process-result.json", 16384)
    raw_report = read("offline.json", 16384)
    if raw_report is not None:
        parsed_report = parse_offline_report(raw_report)
        if report is not None and parsed_report != report:
            raise ValueError("offline process report projection differs")
        report = parsed_report
    marked = report is not None and report.evidence_kind == "offline_smoke_diagnostic_v2"
    intent = None
    if intent_raw is not None:
        legacy = decode_json(intent_raw, limit=16384) == {
            "evidence_kind": "offline_smoke_diagnostic"
        }
        if not legacy:
            intent = _parse_record(intent_raw, OfflineProcessIntent)
        elif marked or result_raw is not None:
            raise ValueError("offline process intent/report markers contradict")
    protocol = marked or intent is not None or result_raw is not None
    if not protocol:
        return None
    if report is not None and not marked:
        raise ValueError("offline process cannot use legacy diagnostic report")
    result = None if result_raw is None else _parse_record(result_raw, OfflineProcessResult)
    limits = OfflineProcessLimits(**intent.limits) if intent is not None else OfflineProcessLimits()
    for name, raw in (("intent.json", intent_raw), ("process-result.json", result_raw)):
        if raw is not None and len(raw) > limits.record_bytes:
            raise ValueError("offline process record exceeds intent limit")
        if raw is None:
            unavailable.append("offline.process:" + name)
    for name in ("stdout", "stderr"):
        bound = getattr(limits, name + "_bytes")
        raw = read(name + ".txt", bound)
        if result is not None:
            if getattr(result, name + "_bytes") > bound:
                raise ValueError("offline process log exceeds limit: " + name)
            if raw is not None and (
                len(raw) != getattr(result, name + "_bytes")
                or sha256_bytes(raw) != getattr(result, name + "_sha256")
            ):
                raise ValueError("offline process log differs: " + name)
        if raw is None:
            unavailable.append("offline.process:" + name + ".txt")
    if raw_report is None:
        unavailable.append("offline.process:offline.json")
    if intent is not None:
        digest = sha256_bytes(intent_raw)
        if result is not None and (
            result.intent_sha256 != digest or result.attempt != intent.attempt
        ):
            raise ValueError("offline process intent/attempt binding differs")
        if report is not None and (
            report.process_intent_sha256 != digest
            or report.source_commit != intent.source_commit
            or report.executed_source_sha256 != intent.executed_source_sha256
        ):
            raise ValueError("offline process report intent/source binding differs")
    if result is not None:
        if report is not None and result.intent_sha256 != report.process_intent_sha256:
            raise ValueError("offline process report intent binding differs")
        if result.status == "failed":
            raise ValueError("offline process failed: " + result.reason)
        if raw_report is not None and result.offline_sha256 != sha256_bytes(raw_report):
            raise ValueError("offline process report hash differs")
    missing = any(label.startswith("offline.process:") for label in unavailable)
    return result if result is not None and not missing else None


def require_completed_offline_process(run_dir):
    """Current execution cannot reuse partial, failed or historical-only attempts."""
    from silent_cascade.train.evidence_types import read_bytes
    from silent_cascade.train.pilot_offline import parse_offline_report

    root = run_dir / "final/offline"
    if root.is_symlink():
        raise ValueError("offline process root cannot be a symlink")
    if not root.exists():
        return False
    if not any(root.iterdir()):
        return False
    try:
        report = (
            parse_offline_report(read_bytes(root / "offline.json", limit=16384))
            if (root / "offline.json").exists()
            else None
        )
        result = read_offline_process_outcome(run_dir, report=report, unavailable=[])
        if (
            report is None
            or report.evidence_kind != "offline_smoke_diagnostic_v2"
            or result is None
        ):
            raise ValueError("missing completed outcome")
    except (ValueError, OSError, ArtifactIntegrityError) as error:
        raise ValueError("offline process attempt is not reusable") from error
    return True


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
        None if process is None else process.pid,
        None if process is None else process.returncode,
        reason or ("nonzero_exit" if process is not None and process.returncode else None),
        eof,
    )
