"""Finite parent-owned archive service and closed scientific child jobs."""

import hashlib
import json
import os
import secrets
import selectors
import signal
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import asdict
from pathlib import Path

from silent_cascade.archive.catalog import (
    _control_reader,
    _create_control_object,
    _load_manifest_bytes,
    _opened_unit,
    _safe_logical_path,
    iter_unit_files,
)
from silent_cascade.archive.ledger import (
    StorageBlocked,
    _hash_file,
    _process_identity,
    _StorageBudget,
)
from silent_cascade.archive.session import _OPERATIONS, _publish
from silent_cascade.archive.transport import (
    _cold_reader,
    _load_head,
    _lock,
    _object_key,
    _remove_control_tree,
    _unit_writer_lock,
    archive_unit,
    evict_unit,
    unit_reader_lease,
)
from silent_cascade.archive.types import ArchivePolicy, FileEntry, UnitRef
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

__all__ = ["StorageBlocked", "_process_identity"]


class _OwnerAbsent(ValueError):
    """Authenticated lookup completed without an active payload owner."""


def package_source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        name = path.relative_to(root).as_posix().encode()
        raw = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big") + name)
        digest.update(len(raw).to_bytes(8, "big") + raw)
    return digest.hexdigest()


def _decode_job(*, job: str, request: dict, run_dir: Path):
    from silent_cascade.train.pilot_data import _check_path

    fields = {
        "pilot": {"config_path", "manifest_dir", "device"},
        "checks": {"config_path"},
        "verify": {"artifact_path"},
        "report": {"output_dir"},
        "replay": {"replay_path", "weights_path"},
    }
    if job not in fields or type(request) is not dict or not set(request) <= fields[job]:
        raise ValueError("unknown closed job or request field")
    values = dict(request)
    if job == "verify":
        values.setdefault("artifact_path", "phase4-gate.json")
    if job == "report":
        values.setdefault("output_dir", "report")
    if set(values) != fields[job] or any(type(value) is not str for value in values.values()):
        raise ValueError("missing or non-string job request field")
    if "device" in values and values["device"] not in {"cpu", "mps"}:
        raise ValueError("unsupported pilot device")
    if "config_path" in values and values["config_path"] not in {
        "configs/train/pilot.yaml",
        "configs/train/pilot_smoke.yaml",
    }:
        raise ValueError("unsupported pilot config overlay")
    repo = Path(__file__).resolve().parents[3]
    decoded = {}
    for name, value in values.items():
        if name == "device":
            decoded[name] = value
            continue
        relative = _safe_logical_path(value, field="closed job path")
        path = (repo if name in {"config_path", "manifest_dir"} else run_dir.absolute()) / relative
        _check_path(path)
        decoded[name] = path
    return decoded


def _execute_job(job, request, run_dir):
    values = _decode_job(job=job, request=request, run_dir=run_dir)
    if job == "pilot":
        from silent_cascade.train.pilot_workflow import run_pilot

        run_pilot(run_dir=run_dir, **values)
    elif job == "checks":
        from silent_cascade.train.pilot_checks import run_pilot_checks
        from silent_cascade.train.pilot_workflow import resolve_pilot_path

        run_pilot_checks(
            run_dir=run_dir,
            config=resolve_pilot_path(values["config_path"]),
            output_path=run_dir / "phase4-gate.json",
        )
    elif job == "verify":
        from silent_cascade.train.pilot_evidence import verify_phase4_gate_artifact

        verify_phase4_gate_artifact(
            values["artifact_path"],
            repo_root=Path(__file__).resolve().parents[3],
            raw_run_dir=run_dir,
        )
    elif job == "report":
        from silent_cascade.report.pilot import build_pilot_report

        build_pilot_report(run_dir=run_dir, **values)
    elif job == "replay":
        from silent_cascade.eventflow.neural_replay import verify_neural_replay

        result = verify_neural_replay(values["replay_path"], weights_path=values["weights_path"])
        if not result.matched:
            raise ValueError("neural replay differs")


def _child_main():
    import resource
    import threading

    from silent_cascade.archive.session import LocalArchiveSession
    from silent_cascade.train.pilot_offline import install_offline_boundary

    def stop_child(_signal, _frame):
        # Unwind subprocess.run/Popen contexts so the permitted nested diagnostic
        # is killed and waited for before this process releases its own lifetime.
        raise SystemExit(75)

    signal.signal(signal.SIGTERM, stop_child)
    control = Path(sys.argv[1])
    job = json.loads(_control_reader(control)("job.json", 1024 * 1024))
    if package_source_sha256() != job["source_sha256"]:
        raise ValueError("executing package differs from admitted source")
    run_dir = Path(job["run_dir"])
    policy = ArchivePolicy.model_validate(job["policy"])
    resource.setrlimit(resource.RLIMIT_FSIZE, (job["file_bytes"], job["file_bytes"]))
    install_offline_boundary(workspace_root=Path(job["workspace_root"]))
    session = LocalArchiveSession(run_dir=run_dir, control_dir=control, policy=policy)
    session._request("status", {})
    finished = threading.Event()

    def watch_parent():
        while not finished.wait(1):
            try:
                session._parent_alive()
            except StorageBlocked:
                os.kill(os.getpid(), signal.SIGTERM)
                return

    watcher = threading.Thread(target=watch_parent, daemon=True)
    watcher.start()
    try:
        _execute_job(job["job"], job["request"], run_dir)
    finally:
        finished.set()
        watcher.join(timeout=2)


def _bind_storage_paths(budget, run_dir, control_dir):
    if run_dir == control_dir or run_dir.is_relative_to(control_dir):
        raise ValueError("archive controls cannot contain the scientific run")
    for path, category in (
        (run_dir, "spool"),
        (control_dir, "metadata"),
        (control_dir / "lease-cache", "cache"),
        (control_dir / "transfer-scratch", "scratch"),
        (control_dir / "quarantine/cache", "cache"),
        (control_dir / "quarantine/metadata", "metadata"),
    ):
        budget.bind(path, category=category)


def supervise_job(
    *,
    job: str,
    request: dict,
    workspace_root: Path,
    run_dir: Path,
    control_dir: Path,
    transport,
    policy: ArchivePolicy,
    output_bounds,
) -> int:
    from silent_cascade.archive.types import JobOutputBounds

    if not isinstance(output_bounds, JobOutputBounds):
        raise ValueError("explicit source-derived output bounds required")
    output_bounds = JobOutputBounds.model_validate(output_bounds.model_dump())
    decoded = _decode_job(job=job, request=request, run_dir=run_dir)
    source = package_source_sha256()
    if (
        output_bounds.job != job
        or output_bounds.request_sha256 != sha256_bytes(canonical_json_bytes(request))
        or output_bounds.policy_sha256 != sha256_bytes(canonical_json_bytes(policy))
        or output_bounds.source_sha256 != source
    ):
        raise ValueError("output bounds identity differs")
    try:
        return _supervise_bound_job(
            job=job,
            request=request,
            decoded=decoded,
            source=source,
            workspace_root=workspace_root,
            run_dir=run_dir,
            control_dir=control_dir,
            transport=transport,
            policy=policy,
            output_bounds=output_bounds,
        )
    except (RuntimeError, OSError, ValueError):
        _report_storage_blocked(
            workspace_root=workspace_root, control_dir=control_dir, policy=policy
        )
        return 75


def _report_storage_blocked(*, workspace_root, control_dir, policy):
    payload = {
        "status": "storage_blocked",
        "action": "retain pending files and last durable pilot checkpoint; inspect storage",
    }
    # stderr is the nonallocating fallback when history cannot be authenticated or
    # there is no room even for an atomic status descriptor. Never reset history.
    with suppress(OSError):
        sys.stderr.write(canonical_json_bytes(payload).decode() + "\n")
    try:
        budget = _StorageBudget(workspace=workspace_root, policy=policy)
        budget.require_path(control_dir)
        budget.bind(control_dir, category="metadata")
        missing = sum(not parent.exists() for parent in (control_dir, *control_dir.parents))
        block = max(4096, os.statvfs(workspace_root).f_frsize)
        maximum = block * (missing + 8) + 2 * len(canonical_json_bytes(payload))
        with budget.reserve(metadata=maximum):
            _publish(control_dir, "blocked.json", payload, max_bytes=policy.page_bytes)
    except (StorageBlocked, OSError, ValueError):
        # A status file must never bypass the same ledger or overwrite evidence.
        return


def _supervise_bound_job(
    *,
    job,
    request,
    decoded,
    source,
    workspace_root,
    run_dir,
    control_dir,
    transport,
    policy,
    output_bounds,
):
    from silent_cascade.archive.ledger import _CATEGORIES
    from silent_cascade.train.pilot_offline import offline_environment, resolve_offline_git

    budget = _StorageBudget(workspace=workspace_root, policy=policy)
    for path in (run_dir, control_dir):
        budget.require_path(path)
    if job == "pilot":
        budget.require_path(decoded["manifest_dir"])
    amounts = {name: getattr(output_bounds, name + "_bytes") for name in _CATEGORIES}
    with ExitStack() as ownership:
        ownership.enter_context(
            _lock(
                control_dir=control_dir, relative=("supervisor.lock",), shared=False, blocking=False
            )
        )
        _bind_storage_paths(budget, run_dir.absolute(), control_dir.absolute())
        ownership.enter_context(
            budget.reserve(admission=output_bounds.model_dump(mode="json"), **amounts)
        )
        budget.inherit_reservations = True
        run_dir.mkdir(parents=True, exist_ok=True)
        run_id = sha256_bytes(str(run_dir.absolute()).encode())
        server = _ArchiveServer(
            run_dir=run_dir,
            control_dir=control_dir,
            workspace_root=workspace_root,
            transport=transport,
            policy=policy,
            run_id=run_id,
        )
        server.budget = budget
        token = secrets.token_hex(16)
        scratch = workspace_root / "scratch" / token
        logs = workspace_root / "logs" / token
        scratch.mkdir(parents=True)
        logs.mkdir(parents=True)
        git_pin = resolve_offline_git()
        _publish(
            control_dir,
            "job.json",
            {
                "job": job,
                "request": request,
                "run_dir": str(run_dir.absolute()),
                "workspace_root": str(workspace_root.absolute()),
                "source_sha256": source,
                "policy": policy.model_dump(mode="json"),
                "file_bytes": max(amounts.values()),
            },
            max_bytes=policy.page_bytes,
        )
        command = [
            sys.executable,
            "-B",
            "-c",
            "from silent_cascade.archive.supervisor import _child_main; _child_main()",
            str(control_dir.absolute()),
        ]
        child = None
        with (
            (logs / "stdout.log").open("xb") as stdout,
            (logs / "stderr.log").open("xb") as stderr,
            selectors.DefaultSelector() as selector,
        ):
            try:
                child = subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[3],
                    env=offline_environment(scratch, git_pin=git_pin),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                selector.register(child.stdout, selectors.EVENT_READ, stdout)
                selector.register(child.stderr, selectors.EVENT_READ, stderr)
                logged = 0
                last_health = time.monotonic()
                while child.poll() is None or selector.get_map():
                    server.serve_once()
                    for key, _ in selector.select(timeout=0.05):
                        raw = os.read(
                            key.fileobj.fileno(), min(65536, amounts["logs"] - logged + 1)
                        )
                        if not raw:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        if logged + len(raw) > amounts["logs"]:
                            raise StorageBlocked("storage_blocked: admitted log maximum exhausted")
                        key.data.write(raw)
                        key.data.flush()
                        logged += len(raw)
                    budget.check()
                    if time.monotonic() - last_health >= 30:
                        _publish(
                            control_dir,
                            "health.json",
                            {
                                "session_id": server.identity["session_id"],
                                "child_pid": child.pid,
                                "child_create_time": _process_identity(child.pid),
                                "state": "running",
                            },
                            max_bytes=policy.page_bytes,
                        )
                        last_health = time.monotonic()
                return child.wait(timeout=30)
            finally:
                if child is not None and child.poll() is None:
                    # The fresh child owns this otherwise-private process group.
                    # No global process scan or unrelated process is targeted.
                    with suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        with suppress(ProcessLookupError):
                            os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=5)
                for pipe in () if child is None else (child.stdout, child.stderr):
                    if pipe is not None:
                        pipe.close()
                server.close()


class _ObservedTransport:
    """Track bounded individual operations separately from verified chunk progress."""

    def __init__(self, server):
        self.server = server

    @property
    def transport_id(self):
        return self.server.transport.transport_id

    def create(self, key, source):
        self.server._operation_started()
        return self.server.transport.create(key, source)

    def download(self, key, destination, *, max_bytes):
        self.server._operation_started()
        self.server.transport.download(key, destination, max_bytes=max_bytes)
        if "/objects/" in key and key.endswith(".bin"):
            from silent_cascade.hashing import sha256_file

            if sha256_file(destination) != Path(key).stem:
                raise ValueError("downloaded chunk hash differs")
            self.server._verified_progress()


class _ArchiveServer:
    """One finite child's file requests, one active episode, no command registry."""

    def __init__(
        self,
        *,
        run_dir: Path,
        control_dir: Path,
        workspace_root: Path,
        transport,
        policy: ArchivePolicy,
        run_id: str,
    ):
        self.run_dir = run_dir.absolute()
        self.control_dir = control_dir.absolute()
        self.policy = policy
        self.transport = transport
        self.io = _ObservedTransport(self)
        self.run_id = run_id
        self.budget = _StorageBudget(workspace=workspace_root, policy=policy)
        self.budget.require_path(self.run_dir)
        self.budget.require_path(self.control_dir)
        _bind_storage_paths(self.budget, self.run_dir, self.control_dir)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self._quarantine_staging()
        locator = {
            "workspace": str(workspace_root.absolute()),
            "run_dir": str(self.run_dir),
            "control_dir": str(self.control_dir),
            "policy_sha256": sha256_bytes(canonical_json_bytes(policy)),
        }
        locator_path = self.control_dir / "storage-state.json"
        if locator_path.exists():
            if (
                json.loads(
                    _control_reader(self.control_dir)("storage-state.json", policy.page_bytes)
                )
                != locator
            ):
                raise StorageBlocked("storage_blocked: storage locator differs")
        else:
            _create_control_object(
                self.control_dir, "storage-state.json", canonical_json_bytes(locator)
            )
        self.identity = {
            "schema_version": "phase4-r2-session-v1",
            "run_id": run_id,
            "session_id": secrets.token_hex(32),
            "run_dir": str(self.run_dir),
            "policy_sha256": locator["policy_sha256"],
            "pid": os.getpid(),
            "create_time": _process_identity(os.getpid()),
        }
        _publish(self.control_dir, "session.json", self.identity, max_bytes=policy.page_bytes)
        self.sequence = 0
        self.leases = {}
        self.stopped = False
        self.request = None
        self.progress = 0
        self.operation_time = time.monotonic()
        self.inventory_cursor = None

    def _quarantine_staging(self):
        stages = (("lease-cache", "cache"), ("lease-metadata", "metadata"))
        protocol = (
            "session.json",
            "request.json",
            "response.json",
            "progress.json",
            "job.json",
            "blocked.json",
            "health.json",
        )
        if not any(
            (self.control_dir / name).exists()
            for name in (*protocol, *(name for name, _ in stages))
        ):
            return
        with ExitStack() as locks:
            locks.enter_context(
                _lock(
                    control_dir=self.control_dir,
                    relative=("archive.lock",),
                    shared=False,
                    blocking=False,
                )
            )
            for stripe in range(64):
                locks.enter_context(
                    _lock(
                        control_dir=self.control_dir,
                        relative=("unit-stripes", f"{stripe:02d}.lock"),
                        shared=False,
                        blocking=False,
                    )
                )
            for name, category in stages:
                source = self.control_dir / name
                if not source.exists():
                    continue
                self.budget.require_path(source)
                destination = self.control_dir / "quarantine" / category / secrets.token_hex(16)
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.rename(destination)
                from silent_cascade.archive.catalog import _pinned_directory

                with _pinned_directory(destination.parent) as descriptor:
                    os.fsync(descriptor)
            previous = self.control_dir / "quarantine/metadata" / secrets.token_hex(16)
            for name in protocol:
                source = self.control_dir / name
                if source.exists():
                    self.budget.require_path(source)
                    previous.mkdir(parents=True, exist_ok=True)
                    source.rename(previous / name)
            if previous.exists():
                from silent_cascade.archive.catalog import _pinned_directory

                with _pinned_directory(previous) as descriptor:
                    os.fsync(descriptor)

    def close(self):
        if self.inventory_cursor is not None:
            self.inventory_cursor.close()
        for token in tuple(self.leases):
            self._release(token)

    def _verified_progress(self):
        self.progress += 1
        _publish(
            self.control_dir,
            "progress.json",
            {
                "session_id": self.identity["session_id"],
                "sequence": self.sequence,
                "verified": self.progress,
                "operation_time": self.operation_time,
            },
            max_bytes=self.policy.page_bytes,
        )

    def _operation_started(self):
        self.operation_time = time.monotonic()
        _publish(
            self.control_dir,
            "progress.json",
            {
                "session_id": self.identity["session_id"],
                "sequence": self.sequence,
                "verified": self.progress,
                "operation_time": self.operation_time,
            },
            max_bytes=self.policy.page_bytes,
        )

    def serve_once(self):
        try:
            raw = _control_reader(self.control_dir)("request.json", self.policy.page_bytes)
        except FileNotFoundError:
            return False
        request = json.loads(raw)
        if request.get("session_id") != self.identity["session_id"]:
            return False
        if request.get("sequence") == self.sequence:
            return False
        expected = {
            "schema_version",
            "run_id",
            "session_id",
            "sequence",
            "operation",
            "policy_sha256",
            "unit_id",
            "payload",
            "pid",
        }
        if (
            set(request) != expected
            or request["schema_version"] != "phase4-r2-request-v1"
            or type(request["sequence"]) is not int
            or request["sequence"] != self.sequence + 1
            or request["run_id"] != self.run_id
            or request["policy_sha256"] != self.identity["policy_sha256"]
            or request["operation"] not in _OPERATIONS
            or type(request["payload"]) is not dict
        ):
            raise ValueError("invalid archive request identity")
        self.sequence = request["sequence"]
        self.request = request
        response = {
            key: request[key]
            for key in ("run_id", "session_id", "sequence", "operation", "policy_sha256", "unit_id")
        }
        response["request_sha256"] = sha256_bytes(raw)
        try:
            self.budget.check()
            response.update(
                status="ok", payload=self._dispatch(request["operation"], request["payload"])
            )
            self.budget.check()
        except (OSError, ValueError, StorageBlocked) as error:
            _publish(
                self.control_dir,
                "blocked.json",
                {
                    "status": "storage_blocked",
                    "sequence": self.sequence,
                    "reason": type(error).__name__,
                    "action": "retain pending evidence; inspect local archive state before retry",
                },
                max_bytes=self.policy.page_bytes,
            )
            response.update(status="storage_blocked", payload={})
        _publish(self.control_dir, "response.json", response, max_bytes=self.policy.page_bytes)
        return True

    def _ref(self, payload):
        ref = UnitRef(**payload)
        from silent_cascade.archive.catalog import _validate_unit_ref

        _validate_unit_ref(ref)
        if self.request and self.request["unit_id"] not in {None, ref.unit_id}:
            raise ValueError("request unit differs")
        return ref

    @contextmanager
    def _metadata(self, ref):
        cold = _cold_reader(control_dir=self.control_dir, transport=self.io, run_id=self.run_id)
        raw = cold(ref.manifest_path, self.policy.page_bytes)
        manifest, _ = _load_manifest_bytes(raw, expected_unit_id=ref.unit_id)
        if (
            manifest.identity.run_id != self.run_id
            or manifest.policy_sha256 != self.identity["policy_sha256"]
        ):
            raise ValueError("unit run or policy differs")
        token = secrets.token_hex(16)
        root = self.control_dir / "lease-metadata" / token
        size = len(raw) + sum(shard.decoded_bytes for shard in manifest.inventory_shards) + 65536
        with self.budget.reserve(metadata=size):
            _create_control_object(root, ref.manifest_path, raw)
            for shard in manifest.inventory_shards:
                logical = f"units/{ref.unit_id}/{shard.path}"
                _create_control_object(root, logical, cold(logical, shard.decoded_bytes))
            for _entry in iter_unit_files(root, ref):
                pass
            head = _load_head(self.control_dir, self.run_id)
            if head is not None:
                from silent_cascade.archive.catalog import (
                    _CatalogStore,
                    _load_root,
                    _lookup,
                    verify_run_catalog_unit,
                )

                catalog = _load_root(head, scope="run", reader=cold, policy=self.policy)
                store = _CatalogStore(control_dir=self.control_dir, policy=self.policy, reader=cold)
                if _lookup(store, catalog.units, ref.unit_id, index="units") is not None:
                    verify_run_catalog_unit(
                        self.control_dir,
                        head,
                        ref,
                        policy=self.policy,
                        object_reader=cold,
                        unit_reader=cold,
                    )
                elif not (self.control_dir / ref.manifest_path).exists():
                    raise ValueError("cold unit is not in active catalog")
            elif not (self.control_dir / ref.manifest_path).exists():
                raise ValueError("cold unit lacks an active catalog")
            try:
                yield root, manifest
            finally:
                _remove_control_tree(self.control_dir, f"lease-metadata/{token}")

    def _units(self):
        from silent_cascade.archive.catalog import (
            _CatalogStore,
            _load_root,
            _lookup,
            iter_run_catalog,
        )

        head = _load_head(self.control_dir, self.run_id)
        cold = _cold_reader(control_dir=self.control_dir, transport=self.io, run_id=self.run_id)
        root = None
        store = None
        if head is not None:
            yield from iter_run_catalog(
                self.control_dir, head, policy=self.policy, object_reader=cold
            )
            root = _load_root(head, scope="run", reader=cold, policy=self.policy)
            store = _CatalogStore(control_dir=self.control_dir, policy=self.policy, reader=cold)
        units = self.control_dir / "units"
        if units.exists():
            with os.scandir(units) as directories:
                for directory in directories:
                    if not directory.is_dir(follow_symlinks=False):
                        raise ValueError("invalid resident unit metadata")
                    if (
                        store is not None
                        and _lookup(store, root.units, directory.name, index="units") is not None
                    ):
                        continue
                    raw = _control_reader(self.control_dir)(
                        f"units/{directory.name}/manifest.json", self.policy.page_bytes
                    )
                    manifest, _ = _load_manifest_bytes(raw, expected_unit_id=directory.name)
                    if manifest.identity.run_id != self.run_id:
                        raise ValueError("resident unit belongs to another run")
                    yield UnitRef(
                        directory.name,
                        manifest.kind,
                        manifest.logical_root,
                        manifest.expanded_bytes,
                        manifest.file_count,
                        f"units/{directory.name}/manifest.json",
                    )

    def _entries(self):
        from pathlib import PurePosixPath

        from silent_cascade.archive.catalog import _source_file

        for directory, directories, files in os.walk(self.run_dir, followlinks=False):
            directories[:] = [
                name for name in directories if Path(directory) / name != self.control_dir
            ]
            for name in directories:
                if (Path(directory) / name).is_symlink():
                    raise ValueError("scientific inventory contains symlink")
            for name in files:
                path = (Path(directory) / name).relative_to(self.run_dir).as_posix()
                digest = hashlib.sha256()
                with _source_file(self.run_dir, PurePosixPath(path)) as (descriptor, info):
                    while raw := os.read(descriptor, 1024 * 1024):
                        digest.update(raw)
                    yield FileEntry(path, digest.hexdigest(), info.st_size)
        for ref in self._units():
            if ref.kind == "control_snapshot":
                continue
            with self._metadata(ref) as (root, _manifest):
                for entry in iter_unit_files(root, ref):
                    if (self.run_dir / entry.path).exists():
                        _hash_file(self.run_dir, entry)
                    else:
                        yield entry

    def _find_owner(self, path):
        _safe_logical_path(path, field="evidence path")
        from silent_cascade.archive.catalog import (
            _CatalogStore,
            _load_root,
            _lookup,
            _ownership_key,
        )

        head = _load_head(self.control_dir, self.run_id)
        candidates = None
        if head is not None:
            cold = _cold_reader(control_dir=self.control_dir, transport=self.io, run_id=self.run_id)
            root = _load_root(head, scope="run", reader=cold, policy=self.policy)
            store = _CatalogStore(control_dir=self.control_dir, policy=self.policy, reader=cold)
            owned = _lookup(store, root.ownership, _ownership_key("file", path), index="ownership")
            if owned is not None:
                unit = _lookup(store, root.units, owned.unit_id, index="units")
                if unit is None:
                    raise ValueError("catalog owner lacks a unit")
                candidates = (
                    UnitRef(
                        unit.unit_id,
                        unit.kind,
                        unit.logical_root,
                        unit.expanded_bytes,
                        unit.file_count,
                        unit.manifest_path,
                    ),
                )
        if candidates is None:
            if not (self.run_dir / path).exists():
                raise _OwnerAbsent("cold evidence path has no active catalog owner")
            candidates = self._units()
            catalog_owned = False
        else:
            catalog_owned = True
        for ref in candidates:
            if ref.kind == "control_snapshot":
                continue
            with self._metadata(ref) as (root, manifest):
                for entry in iter_unit_files(root, ref):
                    if entry.path == path:
                        selected = next(
                            (
                                group.paths
                                for group in manifest.episode_groups
                                if path in group.paths
                            ),
                            None,
                        )
                        return ref, selected
        if catalog_owned:
            raise ValueError("catalog ownership is absent from its authenticated unit")
        raise _OwnerAbsent("evidence path has no authenticated owner")

    def _lease(self, ref, *, selected_paths=None):
        if ref.kind == "episode_pack" and any(
            value["ref"].kind == "episode_pack" for value in self.leases.values()
        ):
            raise StorageBlocked("storage_blocked: one episode lease is already active")
        stack = ExitStack()
        token = secrets.token_hex(16)
        cached = False
        try:
            stack.enter_context(unit_reader_lease(control_dir=self.control_dir, ref=ref))
            metadata_root, manifest = stack.enter_context(self._metadata(ref))
            if ref.kind == "episode_pack":
                if selected_paths is None:
                    if len(manifest.episode_groups) != 1:
                        raise ValueError("whole multi-episode pack cannot be leased")
                    selected_paths = manifest.episode_groups[0].paths
                if selected_paths not in tuple(group.paths for group in manifest.episode_groups):
                    raise ValueError("lease selection is not one authenticated episode")
            entries = tuple(
                entry
                for entry in iter_unit_files(metadata_root, ref)
                if selected_paths is None or entry.path in selected_paths
            )
            borrowed = tuple(
                FileEntry(entry.path, entry.sha256, entry.bytes) for entry in manifest.borrowed
            )
            for entry in borrowed:
                try:
                    owner, _ = self._find_owner(entry.path)
                except _OwnerAbsent:
                    owner = None
                if owner is not None:
                    stack.enter_context(unit_reader_lease(control_dir=self.control_dir, ref=owner))
                _hash_file(self.run_dir, entry)
            resident = all((self.run_dir / entry.path).exists() for entry in entries)
            if resident:
                local_root = self.run_dir
            else:
                from silent_cascade.archive.bundles import restore_unit
                from silent_cascade.archive.catalog import _iter_inventory_records_from_unit

                local_root = self.control_dir / "lease-cache" / token
                selected = set(entry.path for entry in entries)
                with _opened_unit(metadata_root, ref) as (unit, checked):
                    indices = sorted(
                        {
                            span.chunk_index
                            for entry in _iter_inventory_records_from_unit(unit, checked)
                            if entry.path in selected
                            for span in entry.spans
                        }
                    )

                def chunks():
                    for index in indices:
                        chunk = manifest.chunks[index]
                        path = self.control_dir / "transfer-scratch" / f"lease-{token}.chunk"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        self.io.download(
                            _object_key(self.run_id, f"objects/{chunk.sha256}.bin"),
                            path,
                            max_bytes=chunk.bytes,
                        )
                        try:
                            yield path
                            self._verified_progress()
                        finally:
                            path.unlink(missing_ok=True)

                with self.budget.reserve(
                    cache=sum(entry.bytes for entry in entries)
                    + sum(entry.bytes for entry in borrowed)
                    + 65536,
                    scratch=self.policy.chunk_bytes,
                ):
                    restore_unit(
                        control_dir=metadata_root,
                        ref=ref,
                        chunks=chunks(),
                        destination=local_root,
                        policy=self.policy,
                        selected_paths=selected_paths,
                    )
                    cached = True
                    for entry in borrowed:
                        raw = _control_reader(self.run_dir)(entry.path, entry.bytes)
                        if sha256_bytes(raw) != entry.sha256:
                            raise ValueError("borrowed dependency changed")
                        _create_control_object(local_root, entry.path, raw)
            for entry in (*entries, *borrowed):
                _hash_file(local_root, entry)
            self.leases[token] = {
                "ref": ref,
                "stack": stack,
                "root": local_root,
                "cached": cached,
                "borrowed": borrowed,
            }
            return {
                "token": token,
                "ref": asdict(ref),
                "local_root": str(local_root),
                "metadata_root": str(metadata_root),
                "selected_paths": selected_paths,
            }
        except BaseException:
            stack.close()
            # Preserve a failed restored tree; it remains charged and unexposed.
            raise

    def _release(self, token):
        value = self.leases.pop(token)
        value["stack"].close()
        if value["cached"]:
            with _unit_writer_lock(control_dir=self.control_dir, ref=value["ref"]):
                _remove_control_tree(self.control_dir, "lease-cache/" + token)
        return {}

    def _dispatch(self, operation, payload):
        if operation == "seal":
            from pydantic import TypeAdapter

            from silent_cascade.archive.session import register_episode_binding
            from silent_cascade.archive.types import EpisodeCommit

            if set(payload) == {"logical_root", "unit_ref", "commit"}:
                commit = TypeAdapter(EpisodeCommit).validate_json(
                    canonical_json_bytes(payload["commit"])
                )
                binding = register_episode_binding(
                    run_dir=self.run_dir,
                    control_dir=self.control_dir,
                    logical_root=payload["logical_root"],
                    unit_ref=self._ref(payload["unit_ref"]),
                    commit=commit,
                    transport=self.io,
                    policy=self.policy,
                )
                return {"commit_sha256": binding.commit_sha256}
            from silent_cascade.archive.catalog import seal_unit

            expected = {"logical_root", "paths", "kind", "identity", "episode_groups", "borrowed"}
            if set(payload) != expected or payload["identity"].get("run_id") != self.run_id:
                raise ValueError("invalid unit seal request")
            ref = seal_unit(
                run_dir=self.run_dir,
                control_dir=self.control_dir,
                logical_root=payload["logical_root"],
                paths=tuple(payload["paths"]),
                kind=payload["kind"],
                identity=payload["identity"],
                episode_groups=tuple(tuple(group) for group in payload["episode_groups"]),
                borrowed=tuple(FileEntry(**value) for value in payload["borrowed"]),
                policy=self.policy,
            )
            return {"ref": asdict(ref)}
        if operation == "lease":
            if set(payload) == {"ref"}:
                return self._lease(self._ref(payload["ref"]))
            if set(payload) == {"logical_root", "ordinal", "commit_sha256"}:
                from silent_cascade.archive.session import lookup_episode_binding

                binding = lookup_episode_binding(
                    control_dir=self.control_dir,
                    run_id=self.run_id,
                    transport=self.io,
                    policy=self.policy,
                    **payload,
                )
                result = self._lease(
                    binding.unit_ref,
                    selected_paths=tuple(entry.path for entry in binding.commit.owned),
                )
                result["binding"] = binding.model_dump(mode="json")
                return result
            if set(payload) == {"path"}:
                ref, selected = self._find_owner(payload["path"])
                return self._lease(ref, selected_paths=selected)
            if set(payload) == {"metadata_root"}:
                candidates = (
                    ref
                    for ref in self._units()
                    if ref.logical_root == payload["metadata_root"]
                    and ref.kind == "evaluation_metadata"
                )
                ref = next(candidates, None)
                if ref is None or next(candidates, None) is not None:
                    raise ValueError("evaluation metadata owner is missing or ambiguous")
                return self._lease(ref)
            raise ValueError("invalid lease payload")
        if operation == "release":
            if set(payload) != {"token"} or payload["token"] not in self.leases:
                raise ValueError("invalid release payload")
            return self._release(payload["token"])
        if operation == "archive":
            if set(payload) != {"ref", "evict"} or type(payload["evict"]) is not bool:
                raise ValueError("invalid archive payload")
            ref = self._ref(payload["ref"])
            pinned = {entry.path for value in self.leases.values() for entry in value["borrowed"]}
            if any(entry.path in pinned for entry in iter_unit_files(self.control_dir, ref)):
                raise StorageBlocked("storage_blocked: borrowed shared file is pinned")
            with self.budget.reserve(scratch=2 * self.policy.chunk_bytes):
                receipt = archive_unit(
                    run_dir=self.run_dir,
                    control_dir=self.control_dir,
                    ref=ref,
                    transport=self.io,
                    policy=self.policy,
                )
            if payload["evict"]:
                evict_unit(
                    run_dir=self.run_dir,
                    control_dir=self.control_dir,
                    ref=ref,
                    receipt_sha256=receipt,
                )
            return {"receipt_sha256": receipt}
        if operation == "status" and not payload:
            return {"healthy": True}
        if operation == "status" and payload in ({"inventory": "begin"}, {"inventory": "next"}):
            if payload["inventory"] == "begin":
                if self.inventory_cursor is not None:
                    self.inventory_cursor.close()
                self.inventory_cursor = self._entries()
            if self.inventory_cursor is None:
                raise ValueError("inventory cursor missing")
            entry = next(self.inventory_cursor, None)
            if entry is None:
                self.inventory_cursor.close()
                self.inventory_cursor = None
            return {"entry": None if entry is None else asdict(entry)}
        if operation == "stop" and not payload:
            self.stopped = True
            return {}
        raise ValueError("invalid closed archive operation payload")
