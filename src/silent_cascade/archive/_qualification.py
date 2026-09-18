"""Private, bounded early provider qualification; never scientific acceptance."""

import hashlib
import json
import multiprocessing
import os
import signal
import stat
import subprocess
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from silent_cascade.archive.bundles import restore_unit
from silent_cascade.archive.catalog import (
    _control_reader,
    _create_at,
    _open_child_directory,
    _opened_unit,
    _pinned_directory,
    _safe_logical_path,
    _source_file,
    _validate_member,
    iter_unit_files,
    seal_unit,
)
from silent_cascade.archive.ledger import StorageBlocked, _StorageBudget
from silent_cascade.archive.preflight import (
    _AUTHORITY,
    _allocation_unit,
    _archive_unit_bounds,
    _authority,
    _executable_digest,
)
from silent_cascade.archive.transport import (
    ObjectTransport,
    _active_receipt_authorized,
    _completed_intents,
    _fresh_download,
    _load_head,
    _local_reservation_records,
    _lock,
    _object_key,
    _opened_ledger,
    _operational_ref,
    _read_receipt,
    _read_regular,
    _regular_identity,
    _remove_fresh,
    _replace_at,
    _transport_identity,
    _verify_receipt_catalog,
    archive_operation_lock,
    archive_unit,
    unit_reader_lease,
)
from silent_cascade.archive.types import (
    ArchivePolicy,
    Count,
    EpisodeCommit,
    FileEntry,
    Hash,
    Revision,
    UnitRef,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.validation import StrictModel

_CATEGORIES = ("spool", "cache", "pinned", "metadata", "scratch", "logs", "emergency")
_REPLY_LIMIT = 16384
_EVENT_LIMIT = 4096
_EVENT_COUNT = 32
_PHASES = {"interrupt", "resume", "repeat", "debug"}
_PROBE_BYTES = 64 * 1024**2
_PROBE_SHA256 = "e04dc3863c905cbf429e51f9a248159ccbc34688763f871fd20d2dfd8964df34"
_REMOTE_STOP = 256 * 1024**2


class _LocalAllocation(StrictModel):
    spool: Count = 0
    cache: Count = 0
    pinned: Count = 0
    metadata: Count = 0
    scratch: Count = 0
    logs: Count = 0
    emergency: Count = 0
    retained_bytes: Count = 0


@dataclass(frozen=True)
class _QualificationBounds:
    local: _LocalAllocation
    remote: int
    units: tuple
    transport_calls: int


def _qualification_bounds(
    *, policy, run_id, probe, debug, debug_logical_root, block, prior, reservations, completed
):
    if type(block) is not int or block <= 0 or block % 512:
        raise ValueError("invalid qualification allocation unit")
    first = _archive_unit_bounds(
        policy=policy,
        run_id=run_id,
        logical_root="probe",
        selected=probe,
        kind="diagnostic",
        episode_groups=(),
        prior_operational_records=prior,
        local_reservations=reservations,
        completed_count=len(completed),
    )
    # The second run generation includes every index record of the first unit.
    prior_run = (
        1
        + len(probe)
        + len(
            {
                parent.as_posix()
                for entry in probe
                for parent in Path(entry.path).parents
                if parent.as_posix() != "."
            }
        )
    )
    second = _archive_unit_bounds(
        policy=policy,
        run_id=run_id,
        logical_root=debug_logical_root,
        selected=debug,
        kind="episode_pack",
        episode_groups=(tuple(item.path for item in debug),),
        prior_operational_records=first.operational_records,
        local_reservations=reservations + first.objects,
        completed_count=len(completed),
        prior_run_records=prior_run,
    )
    units = (first, second)
    extras = sum(len(raw) for _entry, _path, raw, _receipt in completed)
    remote = sum(unit.remote for unit in units) + 2 * extras
    files = (*probe, *debug)
    source = sum(((entry.bytes + block - 1) // block) * block for entry in files)
    directories = sum(len(Path(entry.path).parents) for entry in files) + 8
    source += directories * block
    # Preserve sources and both complete restores; no deduplication credit.
    metadata = sum(
        unit.inventory
        + unit.manifest
        + 2 * unit.run_catalog
        + unit.operational_catalog
        + 2 * unit.receipt
        + unit.pending
        + (unit.objects + reservations) * 32768
        + (2 * (unit.shards + unit.run_nodes + unit.operational_nodes + unit.objects + 16) + 32)
        * block
        for unit in units
    )
    metadata += 2 * extras + 4 * policy.page_bytes + 65536 + 4 * block
    # Full-sized manifests/receipts can be larger than a catalog page.
    maximum = max(
        min(sum(entry.bytes for entry in probe), policy.chunk_bytes),
        min(sum(entry.bytes for entry in debug), policy.chunk_bytes),
    )
    scratch = maximum + max(
        2
        * max(
            maximum,
            unit.manifest,
            unit.receipt,
            min(policy.page_bytes, unit.run_catalog),
            min(policy.page_bytes, unit.operational_catalog),
        )
        + unit.run_catalog
        + unit.operational_catalog
        + (2 * (unit.run_nodes + unit.operational_nodes + 2) + 12) * block
        for unit in units
    )
    logs = 32 * (((4096 + block - 1) // block) * block) + 2 * max(4096, block) + 4 * block
    # Eight operational fixed-point generations can each read the prior tree;
    # every archive object may create and read back, plus proof/index traversal.
    # A lookup can revisit all 256 hash bits despite the eight-node cache.
    calls = sum(
        32
        * 257
        * (
            unit.objects
            + unit.run_nodes
            + unit.operational_nodes
            + prior
            + reservations
            + len(completed)
            + 1
        )
        for unit in units
    )
    return _QualificationBounds(
        _LocalAllocation(spool=source, cache=source, metadata=metadata, scratch=scratch, logs=logs),
        remote,
        units,
        calls,
    )


def _debug_projection(commit: EpisodeCommit, logical_root: str):
    EpisodeCommit(**asdict(commit) | {"owned": commit.owned, "borrowed": commit.borrowed})
    root = _safe_logical_path(logical_root, field="debug logical root")
    if not commit.owned or not sum(entry.bytes for entry in commit.owned):
        raise ValueError("qualification requires a complete nonempty owned group")
    return tuple(
        (entry, _validate_member(entry.path, logical_root).relative_to(root))
        for entry in commit.owned
    )


def _copy_debug(source, destination, projection):
    for entry, relative in projection:
        target = destination / entry.path
        with (
            _source_file(source, relative) as (original, initial),
            _pinned_directory(target.parent, create=True) as directory,
        ):
            output = os.open(
                target.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
            digest, count = hashlib.sha256(), 0
            try:
                while block := os.read(original, 1024**2):
                    count += len(block)
                    if count > entry.bytes:
                        raise ValueError("debug source exceeds committed size")
                    digest.update(block)
                    view = memoryview(block)
                    while view:
                        view = view[os.write(output, view) :]
                final = os.fstat(original)
                if (
                    count != entry.bytes
                    or digest.hexdigest() != entry.sha256
                    or (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
                    != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
                ):
                    raise ValueError("debug source differs from completed commit")
                os.fsync(output)
            finally:
                os.close(output)
            os.fsync(directory)


def _bounded_record(value, limit):
    raw = canonical_json_bytes(value) + b"\n"
    if len(raw) > limit:
        raise StorageBlocked("storage_blocked: qualification evidence limit")
    return raw


def _event(root, *, phase, status, summary=None):
    if phase not in _PHASES | {"admit", "seal", "restore", "complete"}:
        raise ValueError("invalid qualification phase")
    with _pinned_directory(root, create=True) as directory:
        # Only fixed summaries are durable; transport calls aggregate in memory.
        names = os.listdir(directory)
        count = len([name for name in names if name.endswith(".json") and name != "progress.json"])
        if count >= _EVENT_COUNT:
            raise StorageBlocked("storage_blocked: qualification event count")
        value = {"sequence": count, "phase": phase, "status": status}
        if summary is not None:
            value["summary"] = summary
        raw = _bounded_record(value, _EVENT_LIMIT)
        name = "first-create.json" if status == "blocked" else f"{count:02d}-{phase}.json"
        _create_at(directory, name, raw)
    return value


def _progress(root, phase):
    with _pinned_directory(root, create=True) as directory:
        _replace_at(
            directory,
            "progress.json",
            _bounded_record({"phase": phase, "status": "waiting"}, _EVENT_LIMIT),
        )


class _ObservedTransport:
    def __init__(self, delegate, budget, phase, events, reply, gate):
        self.delegate, self.budget, self.phase = delegate, budget, phase
        self.events, self.reply, self.gate = events, reply, gate
        self.summary = {
            "creates": 0,
            "downloads": 0,
            "successes": 0,
            "conflicts": 0,
            "total_bytes": 0,
            "maximum_bytes": 0,
            "object_key_sha256": "0" * 64,
            "conflict_readback": False,
            "chain_sha256": "0" * 64,
            "local_peak": dict.fromkeys(_CATEGORIES, 0),
        }

    @property
    def transport_id(self):
        return self.delegate.transport_id

    def observe(self):
        with _lock(
            control_dir=self.budget.root,
            relative=("workspace.lock",),
            shared=False,
            blocking=True,
        ):
            measured = self.budget.check()
        for name in _CATEGORIES:
            self.summary["local_peak"][name] = max(self.summary["local_peak"][name], measured[name])

    def record(self, operation, key, size, outcome):
        key_hash = sha256_bytes(key.encode())
        value = {"operation": operation, "key_sha256": key_hash, "bytes": size, "outcome": outcome}
        self.summary["chain_sha256"] = sha256_bytes(
            bytes.fromhex(self.summary["chain_sha256"]) + canonical_json_bytes(value)
        )
        self.summary["total_bytes"] += size
        self.summary["maximum_bytes"] = max(self.summary["maximum_bytes"], size)
        if outcome == "success":
            self.summary["successes"] += 1
        return key_hash

    def create(self, key, source):
        self.observe()
        size = _regular_identity(source)[2]
        self.summary["creates"] += 1
        try:
            self.delegate.create(key, source)
        except FileExistsError:
            self.summary["conflicts"] += 1
            key_hash = self.record("create", key, size, "conflict")
            if self.summary["conflicts"] == 1:
                self.summary["object_key_sha256"] = key_hash
            raise
        else:
            key_hash = self.record("create", key, size, "success")
            if self.phase == "interrupt":
                _event(
                    self.events,
                    phase=self.phase,
                    status="blocked",
                    summary={"object_key_sha256": key_hash},
                )
                self.reply.send_bytes(
                    _bounded_record(
                        {"status": "blocked", "object_key_sha256": key_hash}, _REPLY_LIMIT
                    )
                )
                # Only SIGKILL is a passing interruption; EOF must fail closed.
                self.gate.recv_bytes(1)
                raise RuntimeError("qualification gate unexpectedly returned")
        finally:
            self.observe()

    def download(self, key, destination, *, max_bytes):
        self.observe()
        self.summary["downloads"] += 1
        try:
            self.delegate.download(key, destination, max_bytes=max_bytes)
            key_hash = self.record("download", key, max_bytes, "success")
            if key_hash == self.summary["object_key_sha256"]:
                self.summary["conflict_readback"] = True
        finally:
            self.observe()


def _archive_child(run_dir, control_dir, ref, policy, transport, phase, reply, gate):
    try:
        os.setsid()
        reply.send_bytes(_bounded_record({"status": "ready"}, _REPLY_LIMIT))
        if gate.recv_bytes(3) != b"run":
            raise ValueError("qualification child start gate differs")
        budget = _StorageBudget(workspace=control_dir.parent, policy=policy)
        observed = _ObservedTransport(
            transport, budget, phase, run_dir.parent / "events", reply, gate
        )
        receipt = archive_unit(
            run_dir=run_dir, control_dir=control_dir, ref=ref, policy=policy, transport=observed
        )
        observed.observe()
        reply.send_bytes(
            _bounded_record(
                {"status": "complete", "receipt_sha256": receipt, **observed.summary}, _REPLY_LIMIT
            )
        )
    except BaseException:
        # Never serialize exception text, provider output or source locations.
        with suppress(BaseException):
            reply.send_bytes(
                _bounded_record({"status": "failed", "type": "archive_failure"}, _REPLY_LIMIT)
            )
    finally:
        reply.close()
        gate.close()


def _fail_stop(events, phase):
    try:
        _event(events, phase=phase, status="unreaped")
    finally:
        # Do not unwind the live parent's durable reservation ownership.
        os._exit(70)


def _stop_child(child, *, owns_group, events, phase):
    if owns_group:
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
    elif child.is_alive():
        child.kill()
    child.join(60)
    if child.is_alive():
        _fail_stop(events, phase)
    if owns_group:
        deadline = time.monotonic() + 60
        while True:
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                break
            if time.monotonic() >= deadline:
                _fail_stop(events, phase)
            time.sleep(0.05)


def _spawn_archive_phase(
    *,
    run_dir,
    control_dir,
    ref,
    policy,
    transport,
    phase,
    transport_calls,
    observe_local,
):
    if phase not in _PHASES or type(transport_calls) is not int or not 0 < transport_calls < 2**63:
        raise ValueError("invalid qualification child bound")
    ctx = multiprocessing.get_context("spawn")
    reply, child_reply = ctx.Pipe(duplex=False)
    child_gate, gate = ctx.Pipe(duplex=False)
    child = ctx.Process(
        target=_archive_child,
        args=(run_dir, control_dir, ref, policy, transport, phase, child_reply, child_gate),
    )
    events = run_dir.parent / "events"
    deadline = time.monotonic() + (5460 if phase == "interrupt" else 60 + transport_calls * 5400)
    started = False
    owns_group = False
    try:
        observe_local()
        child.start()
        started = True
        child_reply.close()
        child_gate.close()
        if not reply.poll(60) or json.loads(reply.recv_bytes(_REPLY_LIMIT)) != {"status": "ready"}:
            raise StorageBlocked("storage_blocked: qualification child session handshake")
        if os.getpgid(child.pid) != child.pid:
            raise StorageBlocked("storage_blocked: qualification child session differs")
        owns_group = True
        gate.send_bytes(b"run")
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise StorageBlocked("storage_blocked: qualification child deadline")
            # Child observations own stable transfer barriers. Parent progress
            # publication shares the ledger lock with those measurements.
            with _lock(
                control_dir=control_dir.parent / ".silent-cascade-storage",
                relative=("workspace.lock",),
                shared=False,
                blocking=True,
            ):
                _progress(events, phase)
            if reply.poll(min(60, remaining)):
                value = json.loads(reply.recv_bytes(_REPLY_LIMIT))
                break
            if not child.is_alive():
                raise StorageBlocked("storage_blocked: qualification child exited without reply")
        if phase == "interrupt":
            if value.get("status") != "blocked":
                raise StorageBlocked("storage_blocked: missing qualification interruption marker")
            observe_local()
            _stop_child(child, owns_group=owns_group, events=events, phase=phase)
            if child.exitcode != -signal.SIGKILL:
                raise StorageBlocked("storage_blocked: qualification signal proof failed")
            value = {"signal": signal.SIGKILL, "object_key_sha256": value["object_key_sha256"]}
        else:
            child.join(60)
            if child.exitcode != 0 or value.get("status") != "complete":
                raise StorageBlocked("storage_blocked: qualification archive child failed")
        observe_local()
        _event(events, phase=phase, status="complete", summary=value)
        return value
    except BaseException:
        _event(events, phase=phase, status="failed")
        raise
    finally:
        if started:
            _stop_child(child, owns_group=owns_group, events=events, phase=phase)
            child.close()
        for connection in (reply, child_reply, gate, child_gate):
            connection.close()


def _probe_blocks():
    for index in range(64):
        digest = hashlib.sha256(
            b"silent-cascade-r2-qualification-v1\0" + index.to_bytes(8, "big")
        ).digest()
        yield digest * 32768


class _RemoteReservationSnapshot(StrictModel):
    transport_id: str
    accounted_bytes: Count
    reserved_bytes: Count
    operational_publication_bytes: Count
    reservation_pending: bool
    operational_pending: bool


class _UnitResult(StrictModel):
    unit_id: Hash
    receipt_sha256: Hash
    restored_inventory_sha256: Hash


class _InterruptionResult(StrictModel):
    signal: Literal[9]
    object_key_sha256: Hash
    receipt_absent: Literal[True]
    authorization_absent: Literal[True]


class _RetryResult(StrictModel):
    provider_conflict: Literal[True]
    object_key_sha256: Hash
    exact_readback: Literal[True]
    stable_receipt: Literal[True]
    stable_reserved_bytes: Literal[True]


class QualificationResult(StrictModel):
    schema_version: Literal["phase4-r2-early-qualification-v1"] = "phase4-r2-early-qualification-v1"
    source_commit: Revision
    executable_sha256: Hash
    protocol_sha256: Hash
    policy_sha256: Hash
    run_id: str
    transport_id: str
    probe_sha256: Hash
    debug_commit_sha256: Hash
    debug_review_sha256: Hash
    probe: _UnitResult
    debug: _UnitResult
    interruption: _InterruptionResult
    retry: _RetryResult
    remote_before: _RemoteReservationSnapshot
    remote_after: _RemoteReservationSnapshot
    qualification_remote_bytes: Count
    local_bound: _LocalAllocation
    local_before: _LocalAllocation
    local_peak: _LocalAllocation
    local_after: _LocalAllocation
    event_log_sha256: Hash


def _require_bounds(derived, admitted):
    if derived.remote > _REMOTE_STOP:
        raise StorageBlocked("storage_blocked: qualification remote stop bound")
    if derived.remote > admitted.remote or any(
        getattr(derived.local, name) > getattr(admitted.local, name) for name in _CATEGORIES
    ):
        raise StorageBlocked("storage_blocked: qualification grant underestimated")


def run_early_r2_qualification(
    *,
    budget: _StorageBudget,
    output_root: Path,
    debug_source_root: Path,
    debug_logical_root: str,
    debug_commit: EpisodeCommit,
    debug_review_sha256: str,
    protocol_sha256: str,
    source_commit: str,
    transport,
) -> QualificationResult:
    from silent_cascade.archive.transport import R2CliTransport

    if type(transport) is not R2CliTransport:
        raise StorageBlocked("storage_blocked: qualification requires provider transport")
    try:
        return _run_qualification(
            budget=budget,
            output_root=output_root,
            debug_source_root=debug_source_root,
            debug_logical_root=debug_logical_root,
            debug_commit=debug_commit,
            debug_review_sha256=debug_review_sha256,
            protocol_sha256=protocol_sha256,
            source_commit=source_commit,
            transport=transport,
        )
    except Exception:
        raise StorageBlocked(
            "storage_blocked: early qualification failed; preserve phase evidence"
        ) from None


def _run_qualification(
    *,
    budget,
    output_root,
    debug_source_root,
    debug_logical_root,
    debug_commit,
    debug_review_sha256,
    protocol_sha256,
    source_commit,
    transport,
):
    from pydantic import TypeAdapter

    TypeAdapter(Hash).validate_python(debug_review_sha256)
    TypeAdapter(Hash).validate_python(protocol_sha256)
    TypeAdapter(Revision).validate_python(source_commit)
    projection = _debug_projection(debug_commit, debug_logical_root)
    control = budget.workspace / "control"
    policy = budget.policy
    budget.require_path(output_root)
    if (
        output_root.exists()
        or output_root == budget.workspace
        or output_root.is_relative_to(control)
    ):
        raise ValueError("qualification output must be fresh and separately admitted")
    roots = {
        output_root: "metadata",
        output_root / "source-probe": "spool",
        output_root / "source-debug": "spool",
        output_root / "restore-probe": "cache",
        output_root / "restore-debug": "cache",
        output_root / "events": "logs",
        control: "metadata",
        control / "transfer-scratch": "scratch",
    }
    state = budget._state()
    for root, category in roots.items():
        budget.require_path(root)
        if state["paths"].get(root.relative_to(budget.workspace).as_posix()) != category:
            raise ValueError("qualification category handoff differs")
    # Require the existing engineering lock before invoking its creating helper.
    _regular_identity(budget.root / "locks/engineering.lock")
    with _lock(
        control_dir=budget.root, relative=("engineering.lock",), shared=False, blocking=True
    ):
        engineering = _authority(budget)
        if engineering is None or "pending" in engineering:
            raise ValueError("qualification requires a clean reviewed authority")
        authority = json.loads(
            _control_reader(Path(engineering["custody_root"]))(_AUTHORITY, 16384)
        )
        executable = _executable_digest()
        revision = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        if (
            authority["source_commit"] != source_commit
            or revision != source_commit
            or authority["executable_sha256"] != executable
        ):
            raise ValueError("qualification executable authority differs")
        identity = {
            "schema_version": "phase4-r2-early-qualification-input-v1",
            "source_commit": source_commit,
            "executable_sha256": executable,
            "protocol_sha256": protocol_sha256,
            "policy_sha256": sha256_bytes(canonical_json_bytes(policy)),
            "debug_commit_sha256": sha256_bytes(canonical_json_bytes(asdict(debug_commit))),
            "debug_review_sha256": debug_review_sha256,
        }
        run_id = "qualification-" + sha256_bytes(canonical_json_bytes(identity))
        before_remote = _remote_reservation_snapshot(
            control_dir=control, transport=transport, policy=policy
        )
        if before_remote.reservation_pending or before_remote.operational_pending:
            raise ValueError("qualification remote ledger has pending history")
        with (
            archive_operation_lock(control),
            _opened_ledger(control, transport_id=transport.transport_id, policy=policy) as (
                remote_state,
                _reservations,
                _objects,
            ),
        ):
            head = _operational_ref(remote_state)
            prior = 0 if head is None else head.entry_count
            reservations = len(_local_reservation_records(control))
            completed = _completed_intents(control)
            if _load_head(control, run_id) is not None:
                raise ValueError("qualification run already has catalog history")
        probe = (FileEntry("probe/probe.bin", _PROBE_SHA256, _PROBE_BYTES),)
        bound_args = dict(
            policy=policy,
            run_id=run_id,
            probe=probe,
            debug=debug_commit.owned,
            debug_logical_root=debug_logical_root,
            block=_allocation_unit(os.statvfs(budget.workspace)),
            prior=prior,
            reservations=reservations,
            completed=completed,
        )
        bound = _qualification_bounds(**bound_args)
        _require_bounds(bound, bound)
        for entry, relative in projection:
            with _source_file(debug_source_root, relative) as (descriptor, info):
                digest, size = hashlib.sha256(), 0
                while block := os.read(descriptor, 1024**2):
                    size += len(block)
                    if size > entry.bytes:
                        raise ValueError("qualification debug source exceeds commit")
                    digest.update(block)
                if (
                    size != info.st_size
                    or size != entry.bytes
                    or digest.hexdigest() != entry.sha256
                ):
                    raise ValueError("qualification debug source differs from commit")
        before = _LocalAllocation(**budget.check(), retained_bytes=budget.retained_charge())
        peak = before.model_dump()

        def observe():
            measured = budget.check()
            for name in _CATEGORIES:
                peak[name] = max(peak[name], measured[name])
            return measured

        # The only parent reservation spans writes, every child, and both restores.
        with budget.reserve(
            admission={"early_qualification": run_id},
            **{name: getattr(bound.local, name) for name in _CATEGORIES},
        ):
            with _pinned_directory(output_root, create=True):
                pass
            events = output_root / "events"
            _event(events, phase="admit", status="complete")
            source_probe, source_debug = output_root / "source-probe", output_root / "source-debug"
            target = source_probe / probe[0].path
            with _pinned_directory(target.parent, create=True) as parent:
                descriptor = os.open(
                    target.name,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=parent,
                )
                digest, size = hashlib.sha256(), 0
                try:
                    for block in _probe_blocks():
                        digest.update(block)
                        size += len(block)
                        view = memoryview(block)
                        while view:
                            view = view[os.write(descriptor, view) :]
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                os.fsync(parent)
            if size != _PROBE_BYTES or digest.hexdigest() != _PROBE_SHA256:
                raise ValueError("qualification probe recipe differs")
            _copy_debug(debug_source_root, source_debug, projection)
            observe()
            unit_identity = dict(
                run_id=run_id,
                source_commit=source_commit,
                config_sha256=protocol_sha256,
                evidence_identity_sha256=identity["debug_commit_sha256"],
                checkpoint_sha256=None,
                writer_stopped=True,
                checkpoint_committed=False,
            )
            refs = (
                seal_unit(
                    run_dir=source_probe,
                    control_dir=control,
                    logical_root="probe",
                    paths=(probe[0].path,),
                    kind="diagnostic",
                    identity=unit_identity,
                    policy=policy,
                ),
                seal_unit(
                    run_dir=source_debug,
                    control_dir=control,
                    logical_root=debug_logical_root,
                    paths=tuple(entry.path for entry in debug_commit.owned),
                    kind="episode_pack",
                    identity=unit_identity,
                    policy=policy,
                    episode_groups=(tuple(entry.path for entry in debug_commit.owned),),
                ),
            )
            for ref, expected in zip(refs, (probe, debug_commit.owned), strict=True):
                if tuple(iter_unit_files(control, ref)) != expected:
                    raise ValueError("qualification sealed inventory differs")
            exact = _qualification_bounds(
                **(
                    bound_args
                    | {
                        "probe": tuple(iter_unit_files(control, refs[0])),
                        "debug": tuple(iter_unit_files(control, refs[1])),
                    }
                )
            )
            _require_bounds(exact, bound)
            for ref, sizing in zip(refs, exact.units, strict=True):
                with _opened_unit(control, ref) as (unit, manifest):
                    from silent_cascade.archive.catalog import _read_at

                    raw = _read_at(unit, "manifest.json", max_bytes=16 * 1024**2)
                    if (
                        len(raw) > sizing.manifest
                        or sum(shard.decoded_bytes for shard in manifest.inventory_shards)
                        > sizing.inventory
                    ):
                        raise ValueError("qualification serializer exceeded admitted bound")
            observe()
            _event(events, phase="seal", status="complete")

            def phase(mode, index=0):
                result = _spawn_archive_phase(
                    run_dir=(source_probe, source_debug)[index],
                    control_dir=control,
                    ref=refs[index],
                    policy=policy,
                    transport=transport,
                    phase=mode,
                    transport_calls=bound.transport_calls,
                    observe_local=observe,
                )
                for name, amount in result.get("local_peak", {}).items():
                    peak[name] = max(peak[name], amount)
                return result

            interrupted = phase("interrupt")
            for logical in (
                f"receipts/{refs[0].unit_id}.json",
                f"active-receipts/{refs[0].unit_id}.json",
            ):
                from silent_cascade.archive.transport import _read_control_optional

                if (
                    _read_control_optional(control, logical, max_bytes=policy.metadata_bytes)
                    is not None
                ):
                    raise ValueError("qualification interruption authorized too early")
            resumed = phase("resume")
            if (
                resumed["conflicts"] < 1
                or not resumed["conflict_readback"]
                or resumed["object_key_sha256"] != interrupted["object_key_sha256"]
            ):
                raise ValueError("qualification resume lacks exact provider collision readback")
            retry_before = _remote_reservation_snapshot(
                control_dir=control, transport=transport, policy=policy
            )
            repeated = phase("repeat")
            retry_after = _remote_reservation_snapshot(
                control_dir=control, transport=transport, policy=policy
            )
            if (
                resumed["receipt_sha256"] != repeated["receipt_sha256"]
                or retry_before != retry_after
                or repeated["downloads"] == 0
            ):
                raise ValueError("qualification completed retry changed receipt or reservation")
            try:
                _restore_receipted_unit(
                    control_dir=control,
                    ref=refs[0],
                    receipt_sha256=resumed["receipt_sha256"],
                    transport=transport,
                    destination=source_probe,
                    policy=policy,
                    observe_local=observe,
                )
            except FileExistsError:
                pass
            else:
                raise ValueError("qualification existing destination was replaced")
            unit_results = []
            for index, receipt in enumerate((resumed["receipt_sha256"], None)):
                if index:
                    receipt = phase("debug", 1)["receipt_sha256"]
                destination = output_root / ("restore-probe", "restore-debug")[index] / "tree"
                entries = _restore_receipted_unit(
                    control_dir=control,
                    ref=refs[index],
                    receipt_sha256=receipt,
                    transport=transport,
                    destination=destination,
                    policy=policy,
                    observe_local=observe,
                )
                expected = (probe, debug_commit.owned)[index]
                if entries != expected:
                    raise ValueError("qualification restored inventory differs")
                # Compare retained source and restore bytes, beyond inventory identity.
                for entry in entries:
                    with (
                        _source_file((source_probe, source_debug)[index], Path(entry.path)) as (
                            left,
                            _,
                        ),
                        _source_file(destination, Path(entry.path)) as (right, _),
                    ):
                        while block := os.read(left, 1024**2):
                            if block != os.read(right, len(block)):
                                raise ValueError("qualification restored bytes differ")
                        if os.read(right, 1):
                            raise ValueError("qualification restore contains extra bytes")
                unit_results.append(
                    _UnitResult(
                        unit_id=refs[index].unit_id,
                        receipt_sha256=receipt,
                        restored_inventory_sha256=sha256_bytes(
                            canonical_json_bytes({"files": [asdict(entry) for entry in entries]})
                        ),
                    )
                )
                _event(
                    events,
                    phase="restore",
                    status="complete",
                    summary={"unit_id": refs[index].unit_id},
                )
                observe()
            remote_after = _remote_reservation_snapshot(
                control_dir=control, transport=transport, policy=policy
            )
            delta = remote_after.reserved_bytes - before_remote.reserved_bytes
            if (
                remote_after.reservation_pending
                or remote_after.operational_pending
                or remote_after.transport_id != before_remote.transport_id
                or remote_after.accounted_bytes != before_remote.accounted_bytes
                or not 0 <= delta <= bound.remote <= _REMOTE_STOP
            ):
                raise ValueError("qualification remote accounting differs")
            _event(events, phase="complete", status="complete")
            log = hashlib.sha256()
            for path in sorted(events.glob("*.json")):
                log.update(_read_regular(path, max_bytes=_EVENT_LIMIT))
            after = _LocalAllocation(**observe(), retained_bytes=before.retained_bytes)
            result = QualificationResult(
                **{key: value for key, value in identity.items() if key != "schema_version"},
                run_id=run_id,
                transport_id=transport.transport_id,
                probe_sha256=_PROBE_SHA256,
                probe=unit_results[0],
                debug=unit_results[1],
                interruption=_InterruptionResult(
                    **interrupted, receipt_absent=True, authorization_absent=True
                ),
                retry=_RetryResult(
                    provider_conflict=True,
                    object_key_sha256=interrupted["object_key_sha256"],
                    exact_readback=True,
                    stable_receipt=True,
                    stable_reserved_bytes=True,
                ),
                remote_before=before_remote,
                remote_after=remote_after,
                qualification_remote_bytes=delta,
                local_bound=bound.local,
                local_before=before,
                local_peak=_LocalAllocation(**peak),
                local_after=after,
                event_log_sha256=log.hexdigest(),
            )
            return _publish_result(output_root, result, observe, peak)


def _publish_result(root, result, observe, peak):
    # Measure the actual final inode before create-only publication. Iterate only
    # the bounded record until its own allocation and recorded maxima agree.
    stage = root / ".qualification-result.part"
    with _pinned_directory(root) as directory:
        descriptor = os.open(
            stage.name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
        )
        try:
            for _ in range(8):
                raw = _bounded_record(result, 65536)
                os.lseek(descriptor, 0, os.SEEK_SET)
                os.ftruncate(descriptor, 0)
                view = memoryview(raw)
                while view:
                    view = view[os.write(descriptor, view) :]
                os.fsync(descriptor)
                after = _LocalAllocation(
                    **observe(), retained_bytes=result.local_before.retained_bytes
                )
                updated = result.model_copy(
                    update={"local_after": after, "local_peak": _LocalAllocation(**peak)}
                )
                if _bounded_record(updated, 65536) == raw:
                    result = QualificationResult.model_validate(updated.model_dump())
                    break
                result = updated
            else:
                raise ValueError("qualification result allocation did not converge")
        finally:
            os.close(descriptor)
        identity = _regular_identity(stage)
        os.link(
            stage.name,
            "qualification-result.json",
            src_dir_fd=directory,
            dst_dir_fd=directory,
            follow_symlinks=False,
        )
        _remove_fresh(stage, expected_identity=identity)
        os.fsync(directory)
    return result


def _remote_reservation_snapshot(
    *, control_dir: Path, transport: ObjectTransport, policy: ArchivePolicy
) -> _RemoteReservationSnapshot:
    # _lock can create its hierarchy: authenticate every prerequisite first.
    with _pinned_directory(control_dir) as control:
        locks = _open_child_directory(control, "locks")
        try:
            descriptor = os.open("archive.lock", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=locks)
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("archive lock identity differs")
            finally:
                os.close(descriptor)
        finally:
            os.close(locks)
        reservations = _open_child_directory(control, "remote-reservations")
        os.close(reservations)
    transport_id = _transport_identity(transport)
    with (
        archive_operation_lock(control_dir),
        _opened_ledger(control_dir, transport_id=transport_id, policy=policy) as (
            state,
            reservations,
            _objects,
        ),
    ):

        def present(name):
            try:
                info = os.stat(name, dir_fd=reservations, follow_symlinks=False)
            except FileNotFoundError:
                return False
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("remote pending identity differs")
            return True

        return _RemoteReservationSnapshot(
            transport_id=transport_id,
            accounted_bytes=state["accounted_bytes"],
            reserved_bytes=state["reserved_bytes"],
            operational_publication_bytes=state["operational_publication_bytes"],
            reservation_pending=present("pending.json"),
            operational_pending=present("operational-pending.json"),
        )


def _restore_receipted_unit(
    *,
    control_dir: Path,
    ref: UnitRef,
    receipt_sha256: str,
    transport: ObjectTransport,
    destination: Path,
    policy: ArchivePolicy,
    observe_local: Callable[[], None],
) -> tuple[FileEntry, ...]:
    with (
        unit_reader_lease(control_dir=control_dir, ref=ref),
        _opened_unit(control_dir, ref) as (_unit, manifest),
    ):
        receipt, _raw = _read_receipt(control_dir, ref, receipt_sha256, policy.metadata_bytes)
        if (
            receipt.get("transport_id") != _transport_identity(transport)
            or receipt.get("run_id") != manifest.identity.run_id
            or ArchivePolicy.model_validate(receipt.get("policy")) != policy
            or manifest.policy_sha256 != sha256_bytes(canonical_json_bytes(policy))
        ):
            raise ValueError("qualification receipt binding differs")
        _verify_receipt_catalog(control_dir=control_dir, ref=ref, receipt=receipt, policy=policy)
        if not _active_receipt_authorized(
            control_dir=control_dir, ref=ref, receipt_sha256=receipt_sha256
        ):
            raise ValueError("qualification receipt lacks active authorization")
        expected = [
            {
                "key": _object_key(manifest.identity.run_id, f"objects/{chunk.sha256}.bin"),
                "bytes": chunk.bytes,
                "sha256": chunk.sha256,
            }
            for chunk in manifest.chunks
        ]
        records = receipt.get("objects")
        if (
            not isinstance(records, list)
            or records[: len(expected)] != expected
            or any(
                not isinstance(record, dict)
                or not isinstance(record.get("key"), str)
                or "/objects/" in record["key"]
                for record in records[len(expected) :]
            )
        ):
            raise ValueError("qualification payload sequence differs")
        # Refuse before creating scratch or issuing a payload request.
        try:
            destination.lstat()
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("qualification destination already exists")

        def chunks():
            for descriptor, record in zip(manifest.chunks, expected, strict=True):
                path = _fresh_download(control_dir)
                identity = None
                try:
                    observe_local()
                    transport.download(record["key"], path, max_bytes=descriptor.bytes)
                    identity = _regular_identity(path)
                    observe_local()
                    raw = _read_regular(path, max_bytes=descriptor.bytes)
                    if len(raw) != descriptor.bytes or sha256_bytes(raw) != descriptor.sha256:
                        raise ValueError("qualification chunk readback differs")
                    yield path
                    observe_local()
                finally:
                    if identity is not None:
                        _remove_fresh(path, expected_identity=identity)

        stream = chunks()
        try:
            restore_unit(
                control_dir=control_dir,
                ref=ref,
                chunks=stream,
                destination=destination,
                policy=policy,
            )
        finally:
            stream.close()
        observe_local()
        return tuple(iter_unit_files(control_dir, ref))
