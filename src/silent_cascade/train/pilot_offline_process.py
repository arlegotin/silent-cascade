"""Bounded parent process evidence; callers reserve publication space before launch.

This is not a child filesystem quota. No scientific modules are imported here.
"""

import os
import re
import selectors
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, model_validator

from silent_cascade.errors import ArtifactIntegrityError
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.validation import StrictModel

if TYPE_CHECKING:
    from silent_cascade.archive.ledger import (
        _RetainedHistoryScope,
        _ScopedAdmission,
        _StorageBudget,
    )


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


class OfflineProcessIntentV1(StrictModel):
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


class OfflineProcessIntent(StrictModel):
    schema_version: Literal["phase4-offline-process-intent-v2"]
    attempt: Attempt
    output_root_sha256: Hash
    bootstrap_sha256: Hash
    python_executable_sha256: Hash
    python_sha256: Hash
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    executed_source_sha256: Hash
    limits: dict[str, int]
    write_allowance: dict[str, int] | None = None

    @model_validator(mode="after")
    def closed_limits(self):
        if set(self.limits) != {"stdout_bytes", "stderr_bytes", "record_bytes", "timeout_seconds"}:
            raise ValueError("offline process intent limits differ")
        OfflineProcessLimits(**self.limits)
        if self.write_allowance is not None:
            from silent_cascade.train.pilot_offline_writes import OfflineWriteAllowance

            if set(self.write_allowance) != {"allocated_bytes", "file_names", "directories"}:
                raise ValueError("offline process intent write allowance differs")
            OfflineWriteAllowance(**self.write_allowance)
        return self


def path_identity(kind: bytes, value: str) -> str:
    if kind not in {b"output-root", b"python-executable"}:
        raise ValueError("offline process identity kind differs")
    return sha256_bytes(b"phase4-offline-process-v2\0" + kind + b"\0" + os.fsencode(value))


class OfflineProcessResultV1(StrictModel):
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


class OfflineProcessResult(OfflineProcessResultV1):
    schema_version: Literal["phase4-offline-process-result-v2"]
    stdout_disposition: Literal["empty_public", "private_rejected"]
    stderr_disposition: Literal["empty_public", "private_rejected"]
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
            "private_output_rejected",
            "private_custody_failure",
        ]
        | None
    )

    @model_validator(mode="after")
    def private_streams_are_failures(self):
        for name in ("stdout", "stderr"):
            count = getattr(self, name + "_bytes")
            digest = getattr(self, name + "_sha256")
            if getattr(self, name + "_disposition") == "empty_public":
                if count != 0 or digest != sha256_bytes(b""):
                    raise ValueError("offline process empty stream differs")
            elif count <= 0 or self.status != "failed" or self.offline_sha256 is not None:
                raise ValueError("offline process private stream differs")
        return self


# Fixed publication sequence; no caller-supplied names or private locators.
_PUBLICATION_NAMES = (
    "binding.json",
    "intent.json",
    "stdout.raw",
    "stderr.raw",
    "stdout.txt",
    "stderr.txt",
    "process-result.json",
)
_PRIVATE_NAMES = {"binding.json", "stdout.raw", "stderr.raw"}
_BLOCK = 4096


def _category(budget, state, path):
    relative = path.relative_to(budget.workspace)
    if relative.parts[:1] == (budget.root.name,):
        return "metadata"
    for prefix, category in sorted(
        state["paths"].items(), key=lambda item: len(Path(item[0]).parts), reverse=True
    ):
        if relative.is_relative_to(prefix):
            return category
    return "scratch"


def _custody_paths(budget, output, attempt):
    if (
        output.name != "offline"
        or output.parent.name != "final"
        or output.resolve() != output
        or ".." in output.parts
        or re.fullmatch(r"[0-9a-f]{32}", attempt) is None
    ):
        raise ValueError("offline process custody root differs")
    budget.require_path(output)
    private = budget.workspace / "logs/offline-process-private" / attempt
    budget.require_path(private)
    run = output.parent.parent
    state = budget._state()
    public_forbidden = [budget.root, budget.workspace / "logs"] + [
        budget.workspace / path
        for path, category in state["paths"].items()
        if category in {"logs", "cache"}
    ]
    if any(run.is_relative_to(path) or path.is_relative_to(run) for path in public_forbidden):
        raise ValueError("offline process custody overlaps public run")
    private_forbidden = [budget.root, run] + [
        budget.workspace / path
        for path, category in state["paths"].items()
        if category in {"spool", "cache"}
    ]
    if any(
        private.is_relative_to(path) or path.is_relative_to(private) for path in private_forbidden
    ):
        raise ValueError("offline process private custody overlaps public storage")
    public_category = _category(budget, state, output)
    if any(
        _category(budget, state, output / name) != public_category
        for name in _PUBLICATION_NAMES
        if name not in _PRIVATE_NAMES
    ) or any(
        category != public_category
        and (budget.workspace / prefix).parent == output
        and re.fullmatch(r"\.pilot-[0-9a-f]{32}\.tmp", Path(prefix).name) is not None
        for prefix, category in state["paths"].items()
    ):
        # Public final and random sibling temporary must share the category
        # charged by _publication_peaks; unrelated descendants remain valid.
        raise ValueError("offline process public publication category differs")
    if _category(budget, state, private) != "logs":
        raise ValueError("offline process private custody must count as logs")
    if any(_category(budget, state, private / name) != "logs" for name in _PRIVATE_NAMES):
        raise ValueError("offline process private member category differs")
    # Frozen engineering inventories exclude this live workspace, not individual
    # filenames. check_scoped authenticates that workspace/snapshot relationship.
    return private, state


def _publication_peaks(budget, state, output, private, limits, phase=0):
    if type(phase) is not int or not 0 <= phase <= len(_PUBLICATION_NAMES):
        raise ValueError("offline process publication phase differs")
    peaks = {}
    missing_dirs = set()
    for name in _PUBLICATION_NAMES[phase:]:
        parent = private if name in _PRIVATE_NAMES else output
        category = _category(budget, state, parent / name)
        size = (
            limits.record_bytes
            if name.endswith(".json")
            else getattr(limits, name.split(".")[0] + "_bytes")
            if name.endswith(".raw")
            else 0
        )
        peaks[category] = peaks.get(category, 0) + 2 * ((size + _BLOCK - 1) // _BLOCK) * _BLOCK
        peaks[category] += 2 * _BLOCK  # final and possible temporary name
        for directory in (parent, *parent.parents):
            if directory.exists():
                break
            missing_dirs.add(directory)
    for directory in missing_dirs:
        category = _category(budget, state, directory)
        peaks[category] = peaks.get(category, 0) + _BLOCK
    category = _category(budget, state, output)
    peaks[category] = peaks.get(category, 0) + _BLOCK
    return tuple(sorted(peaks.items()))


def _check_custody_reservation(budget, retained, token, peaks):
    if budget.active_reservation != token or retained.reservation_token != token:
        raise ValueError("offline process custody reservation differs")
    if retained.owner_pid != os.getpid():
        raise ValueError("offline process custody requires its original owner")
    allocated = budget.check_scoped(retained)
    active = budget._state()["reservations"][token]
    if (active["pid"], active["create_time"]) != (retained.owner_pid, retained.owner_create_time):
        raise ValueError("offline process custody owner differs")
    for category, peak in peaks:
        remaining = active["amounts"].get(category, 0) - max(
            0, allocated[category] - active["before"][category]
        )
        if remaining < peak:
            raise ValueError("offline process custody allowance exhausted")


@dataclass(frozen=True, slots=True, repr=False)
class OfflineProcessCustody:
    budget: "_StorageBudget"
    retained: "_RetainedHistoryScope"
    token: str
    owner_pid: int
    owner_create_time: float
    workspace_device: int
    workspace_inode: int
    directory_fd: int
    directory_device: int
    directory_inode: int
    attempt_id: str
    output_root_sha256: str
    limits: OfflineProcessLimits
    category_peaks: tuple[tuple[str, int], ...]

    def __reduce_ex__(self, protocol):
        raise TypeError("offline process custody is private operational state")

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()

    def close(self):
        # Descriptor ownership ends here; retained bytes are never removed.
        os.close(self.directory_fd)


def prepare_offline_process_custody(
    *,
    budget: "_StorageBudget",
    admission: "_ScopedAdmission",
    output_dir: Path,
    limits: OfflineProcessLimits,
    attempt_id: str,
) -> OfflineProcessCustody:
    """Suballocate an already held same-owner reservation; never create a ledger."""
    from silent_cascade.archive.catalog import _pinned_directory
    from silent_cascade.archive.ledger import _ScopedAdmission, _StorageBudget
    from silent_cascade.archive.transport import _lock

    if type(budget) is not _StorageBudget or type(admission) is not _ScopedAdmission:
        raise ValueError("offline process custody admission differs")
    if type(limits) is not OfflineProcessLimits:
        raise ValueError("offline process custody limits differ")
    output = Path(output_dir).absolute()
    private, state = _custody_paths(budget, output, attempt_id)
    if private.exists() or private.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError("offline process custody attempt already exists")
    if os.statvfs(budget.workspace).f_frsize > _BLOCK:
        raise ValueError("offline process custody allocation granule unsupported")
    peaks = _publication_peaks(budget, state, output, private, limits)
    with _lock(control_dir=budget.root, relative=("workspace.lock",), shared=False, blocking=True):
        _check_custody_reservation(budget, admission.retained, admission.token, peaks)
        with _pinned_directory(budget.workspace) as workspace_fd:
            workspace_info = os.fstat(workspace_fd)
            parent = os.dup(workspace_fd)
            try:
                for part in ("logs", "offline-process-private", attempt_id):
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=parent)
                        os.fsync(parent)
                    except FileExistsError:
                        if part == attempt_id:
                            raise ValueError(
                                "offline process custody attempt already exists"
                            ) from None
                    child = os.open(
                        part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
                    )
                    os.close(parent)
                    parent = child
                    info = os.fstat(parent)
                    if info.st_dev != budget.device or (
                        part != "logs" and stat.S_IMODE(info.st_mode) != 0o700
                    ):
                        raise ValueError("offline process private directory differs")
                descriptor = os.dup(parent)
            finally:
                os.close(parent)
    return OfflineProcessCustody(
        budget,
        admission.retained,
        admission.token,
        admission.retained.owner_pid,
        admission.retained.owner_create_time,
        workspace_info.st_dev,
        workspace_info.st_ino,
        descriptor,
        info.st_dev,
        info.st_ino,
        attempt_id,
        path_identity(b"output-root", str(output)),
        limits,
        peaks,
    )


def require_offline_process_custody(custody, *, output_dir, limits=None, phase=0):
    """Check live authority before costly work and every parent publication."""
    from silent_cascade.archive.transport import _lock

    if type(custody) is not OfflineProcessCustody:
        raise ValueError("offline process custody required")
    output = Path(output_dir).absolute()
    if (
        custody.output_root_sha256 != path_identity(b"output-root", str(output))
        or (limits is not None and limits != custody.limits)
        or (custody.owner_pid, custody.owner_create_time)
        != (custody.retained.owner_pid, custody.retained.owner_create_time)
    ):
        raise ValueError("offline process custody binding differs")
    budget = custody.budget
    private, state = _custody_paths(budget, output, custody.attempt_id)
    info, current = os.fstat(custody.directory_fd), private.stat(follow_symlinks=False)
    workspace = budget.workspace.stat(follow_symlinks=False)
    if (
        (workspace.st_dev, workspace.st_ino) != (custody.workspace_device, custody.workspace_inode)
        or (info.st_dev, info.st_ino) != (custody.directory_device, custody.directory_inode)
        or (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino)
        or stat.S_IMODE(info.st_mode) != 0o700
        or set(os.listdir(custody.directory_fd)) - _PRIVATE_NAMES
        or (phase == 0 and os.listdir(custody.directory_fd))
    ):
        raise ValueError("offline process private directory changed")
    with _lock(control_dir=budget.root, relative=("workspace.lock",), shared=False, blocking=True):
        _check_custody_reservation(
            budget,
            custody.retained,
            custody.token,
            _publication_peaks(budget, state, output, private, custody.limits, phase),
        )
    return custody


def _publish_private_process_bytes(custody, name, payload):
    if name not in _PRIVATE_NAMES or type(payload) is not bytes:
        raise ValueError("offline process private member differs")
    bound = (
        custody.limits.record_bytes
        if name == "binding.json"
        else getattr(custody.limits, name.split(".")[0] + "_bytes")
    )
    if len(payload) > bound:
        raise ValueError("offline process private member exceeds bound")
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=custody.directory_fd,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("offline process private write failed")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(custody.directory_fd)


_PROCESS_NAMES = ("intent.json", "process-result.json", "stdout.txt", "stderr.txt", "offline.json")


def _parse_record(raw, model):
    from silent_cascade.train.evidence_types import decode_json

    try:
        return model.model_validate_json(canonical_json_bytes(decode_json(raw, limit=16384)))
    except (ValueError, TypeError, ArtifactIntegrityError) as error:
        raise ValueError("invalid offline process record") from error


def _parse_versioned_process_record(raw, *, intent):
    from silent_cascade.train.evidence_types import decode_json

    value = decode_json(raw, limit=16384)
    version = value.get("schema_version") if isinstance(value, dict) else None
    models = (
        {
            "phase4-offline-process-intent-v1": OfflineProcessIntentV1,
            "phase4-offline-process-intent-v2": OfflineProcessIntent,
        }
        if intent
        else {
            "phase4-offline-process-result-v1": OfflineProcessResultV1,
            "phase4-offline-process-result-v2": OfflineProcessResult,
        }
    )
    if version not in models:
        raise ValueError("offline process record version differs")
    return _parse_record(raw, models[version])


def read_offline_process_outcome(
    run_dir,
    *,
    report,
    evidence_context=None,
    unavailable=None,
    require_safe_process=False,
):
    """Verify all available process members, retaining only one cold lease at a time.

    Forensic v1 records remain readable. Current proof requires privacy-safe v2;
    neither version compares machine identities with the verifier's local paths.
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
            intent = _parse_versioned_process_record(intent_raw, intent=True)
        elif marked or result_raw is not None or require_safe_process:
            raise ValueError("offline process intent/report markers contradict")
    protocol = marked or intent is not None or result_raw is not None or require_safe_process
    if not protocol:
        return None
    if report is not None and not marked:
        raise ValueError("offline process cannot use legacy diagnostic report")
    result = (
        None if result_raw is None else _parse_versioned_process_record(result_raw, intent=False)
    )
    if require_safe_process and (
        (intent is not None and type(intent) is not OfflineProcessIntent)
        or (result is not None and type(result) is not OfflineProcessResult)
    ):
        raise ValueError("offline process privacy-safe version required")
    if (
        intent is not None
        and result is not None
        and ((type(intent) is OfflineProcessIntent) != (type(result) is OfflineProcessResult))
    ):
        raise ValueError("offline process intent/result versions differ")
    limits = OfflineProcessLimits(**intent.limits) if intent is not None else OfflineProcessLimits()
    for name, raw in (("intent.json", intent_raw), ("process-result.json", result_raw)):
        if raw is not None and len(raw) > limits.record_bytes:
            raise ValueError("offline process record exceeds intent limit")
        if raw is None:
            unavailable.append("offline.process:" + name)
    for name in ("stdout", "stderr"):
        bound = getattr(limits, name + "_bytes")
        raw = read(name + ".txt", bound)
        private = (
            type(result) is OfflineProcessResult
            and getattr(result, name + "_disposition") == "private_rejected"
        )
        if private and raw is not None:
            raise ValueError("offline process private stream has public attachment")
        if result is not None:
            if getattr(result, name + "_bytes") > bound:
                raise ValueError("offline process log exceeds limit: " + name)
            if raw is not None and (
                len(raw) != getattr(result, name + "_bytes")
                or sha256_bytes(raw) != getattr(result, name + "_sha256")
            ):
                raise ValueError("offline process log differs: " + name)
        if raw is None and not private:
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
        result = read_offline_process_outcome(
            run_dir, report=report, unavailable=[], require_safe_process=True
        )
        if (
            report is None
            or report.evidence_kind != "offline_smoke_diagnostic_v2"
            or result is None
        ):
            raise ValueError("missing completed outcome")
    except (ValueError, OSError, ArtifactIntegrityError) as error:
        raise ValueError("offline process attempt is not reusable") from error
    return True


def preflight_offline_process(run_dir, custody=None):
    """Safe reuse needs no new authority; fresh work fails before costly inputs."""
    completed = require_completed_offline_process(run_dir)
    if not completed:
        require_offline_process_custody(custody, output_dir=run_dir / "final/offline")
    return completed


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
