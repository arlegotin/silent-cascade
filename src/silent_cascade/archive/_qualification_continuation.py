"""Explicit, bounded continuation of one failed storage qualification campaign."""

import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter

from silent_cascade.archive import _qualification as q
from silent_cascade.archive import preflight
from silent_cascade.archive.bundles import _read_chunk
from silent_cascade.archive.catalog import (
    _opened_unit,
    _pinned_directory,
    _source_file,
    iter_unit_files,
)
from silent_cascade.archive.ledger import StorageBlocked
from silent_cascade.archive.transport import (
    _active_receipt_authorized,
    _completed_intents,
    _load_head,
    _local_reservation_records,
    _lock,
    _object_key,
    _opened_ledger,
    _operational_ref,
    _put_verified,
    _read_receipt,
    _read_regular,
    _regular_identity,
    _transport_identity,
    _verify_receipt_catalog,
    archive_operation_lock,
)
from silent_cascade.archive.types import Count, FileEntry, Hash, Revision, UnitRef
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.validation import StrictModel

_HISTORICAL_DRIVER_SHA256 = "c43c96d9e37e154db9cae0cec66a45a9514fe6d0c74776c01e5b934824848a9f"


class _SourceWitness(StrictModel):
    source_commit: Revision
    executable_sha256: Hash
    source_authority_sha256: Hash


class _Observation(StrictModel):
    creates: Count
    downloads: Count
    successes: Count
    conflicts: Count
    total_bytes: Count
    maximum_bytes: Count
    object_key_sha256: Hash
    conflict_readback: bool
    chain_sha256: Hash
    local_peak: q._LocalAllocation


class _PhaseWitness(StrictModel):
    name: str
    sha256: Hash
    sequence: Count
    phase: Literal["admit", "seal", "interrupt", "resume"]
    status: Literal["complete", "blocked", "failed"]
    object_key_sha256: Hash | None = None
    signal: Literal[9] | None = None


class _PredecessorWitness(StrictModel):
    schema_version: Literal["phase4-qualification-predecessor-v1"] = (
        "phase4-qualification-predecessor-v1"
    )
    source: _SourceWitness
    historical_driver_sha256: Hash
    run_id: str
    policy_sha256: Hash
    protocol_sha256: Hash
    transport_id: str
    failure_proof_sha256: Hash
    recovery_proof_sha256: Hash
    probe_ref: UnitRef
    debug_ref: UnitRef
    probe_inventory_sha256: Hash
    debug_inventory_sha256: Hash
    debug_commit_sha256: Hash
    debug_review_sha256: Hash
    phases: tuple[_PhaseWitness, ...]
    progress_sha256: Hash
    interrupted_chunk_index: Literal[0]
    interrupted_chunk_bytes: Count
    interrupted_chunk_sha256: Hash
    object_key_sha256: Hash
    recovered_receipt_sha256: Hash
    receipt_absent_before_resume: Literal[True]
    authorization_absent_before_resume: Literal[True]
    historical_result_absent: Literal[True]
    resume_failure_cause: Literal["unknown"]
    old_attempt_tree_sha256: Hash


class QualificationContinuationResult(StrictModel):
    schema_version: Literal["phase4-r2-qualification-continuation-v1"] = (
        "phase4-r2-qualification-continuation-v1"
    )
    executor: _SourceWitness
    predecessor: _PredecessorWitness
    continuation_id: Hash
    run_id: str
    protocol_sha256: Hash
    policy_sha256: Hash
    transport_id: str
    probe_sha256: Hash
    debug_commit_sha256: Hash
    debug_review_sha256: Hash
    probe: q._UnitResult
    debug: q._UnitResult
    inherited_interruption: q._InterruptionResult
    retry: q._RetryResult
    collision_observation: _Observation
    repeat_observation: _Observation
    debug_observation: _Observation
    existing_destination_refused: Literal[True]
    remote_before: q._RemoteReservationSnapshot
    remote_after: q._RemoteReservationSnapshot
    remaining_remote_bound: Count
    continuation_remote_bytes: Count
    cumulative_campaign_remote_bytes: Count
    cumulative_shared_remote_bytes: Count
    local_bound: q._LocalAllocation
    local_before: q._LocalAllocation
    local_peak: q._LocalAllocation
    local_after: q._LocalAllocation
    event_log_sha256: Hash
    old_attempt_tree_sha256: Hash


def _historical_source(budget, source):
    engineering = preflight._authority(budget)
    if engineering is None or "pending" in engineering:
        raise ValueError("qualification historical anchor unavailable")
    path = (
        budget.root
        / "source-authorities"
        / source.executable_sha256
        / f"{source.source_commit}.json"
    )
    raw = _read_regular(path, max_bytes=preflight._SOURCE_AUTHORITY_LIMIT)
    record = preflight._ReviewedSourceRecord.model_validate_json(raw)
    if (
        sha256_bytes(raw) != source.source_authority_sha256
        or canonical_json_bytes(record) != raw
        or record.anchor_authority_sha256 != engineering["authority_sha256"]
        or record.source_commit != source.source_commit
        or record.executable_sha256 != source.executable_sha256
        or record.policy_sha256 != sha256_bytes(canonical_json_bytes(budget.policy))
    ):
        raise ValueError("qualification historical authority differs")
    repository = Path(__file__).absolute().parents[3]

    def git(*args, input=None):
        return subprocess.run(
            ["git", *args], cwd=repository, input=input, capture_output=True, check=True, timeout=30
        ).stdout

    tree = git("ls-tree", "-rz", "--full-tree", source.source_commit, "--", "src/silent_cascade")
    entries = []
    for row in tree.split(b"\0"):
        if not row:
            continue
        header, name = row.split(b"\t", 1)
        if not name.endswith(b".py"):
            continue
        mode, kind, blob = header.split()
        if mode not in {b"100644", b"100755"} or kind != b"blob":
            raise ValueError("qualification historical source is not regular")
        entries.append((name.decode(), blob))
    if not entries or len(entries) > 4096:
        raise ValueError("qualification historical source inventory differs")
    entries.sort()
    blobs = git("cat-file", "--batch", input=b"\n".join(blob for _, blob in entries) + b"\n")
    cursor, digest, driver = 0, hashlib.sha256(), None
    for name, blob in entries:
        end = blobs.index(b"\n", cursor)
        identity, kind, size = blobs[cursor:end].split()
        length = int(size)
        if identity != blob or kind != b"blob" or not 0 <= length <= 16 * 1024**2:
            raise ValueError("qualification historical blob differs")
        raw = blobs[end + 1 : end + 1 + length]
        cursor = end + length + 2
        if len(raw) != length or blobs[cursor - 1 : cursor] != b"\n":
            raise ValueError("qualification historical blob truncated")
        hashed = sha256_bytes(raw)
        digest.update(
            canonical_json_bytes(
                {"path": name.removeprefix("src/silent_cascade/"), "sha256": hashed}
            )
        )
        if name == "src/silent_cascade/archive/_qualification.py":
            driver = hashed
    if (
        cursor != len(blobs)
        or digest.hexdigest() != source.executable_sha256
        or driver != _HISTORICAL_DRIVER_SHA256
    ):
        raise ValueError("qualification historical executable or ordering differs")
    return driver


def _proof(path, expected_hash, keys):
    TypeAdapter(Hash).validate_python(expected_hash)
    identity = _regular_identity(path)
    raw = _read_regular(path, max_bytes=65536)
    if sha256_bytes(raw) != expected_hash or _regular_identity(path) != identity:
        raise ValueError("qualification proof hash differs")
    value = json.loads(raw)
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("qualification proof fields differ")
    if raw not in (canonical_json_bytes(value), canonical_json_bytes(value) + b"\n"):
        raise ValueError("qualification proof is not canonical")
    return value


def _file_digest(root, member, expected=None):
    with _source_file(root, member) as (descriptor, initial):
        digest, count = hashlib.sha256(), 0
        while block := os.read(descriptor, 1024**2):
            digest.update(block)
            count += len(block)
            if expected is not None and count > expected.bytes:
                raise ValueError("qualification original bytes exceed inventory")
        final = os.fstat(descriptor)
        if (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns) != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
        ) or (
            expected is not None
            and (count != expected.bytes or digest.hexdigest() != expected.sha256)
        ):
            raise ValueError("qualification original bytes differ")
        return digest.hexdigest(), count


def _tree_digest(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("qualification predecessor contains a link")
        relative = path.relative_to(root)
        if path.is_dir():
            value = {"path": relative.as_posix(), "directory": True}
        else:
            hashed, size = _file_digest(root, relative)
            value = {"path": relative.as_posix(), "sha256": hashed, "bytes": size}
        digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def _interrupted_input(budget, path, chunk):
    budget.require_path(path)
    scratch = budget.workspace / "control/transfer-scratch"
    if (
        not path.is_relative_to(scratch)
        or budget._state()["paths"].get(scratch.relative_to(budget.workspace).as_posix())
        != "scratch"
    ):
        raise ValueError("qualification interrupted input is outside charged scratch")
    with _source_file(path.parent, Path(path.name)) as (descriptor, info):
        if info.st_blocks * 512 < info.st_size:
            raise ValueError("qualification interrupted input is sparse")
        _read_chunk(path, expected_bytes=chunk.bytes, expected_sha256=chunk.sha256)
        if _regular_identity(path) != (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns):
            raise ValueError("qualification interrupted input changed")
        if os.fstat(descriptor) != info:
            # Access time can change; only immutable content identity matters.
            final = os.fstat(descriptor)
            if (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns) != (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
            ):
                raise ValueError("qualification interrupted descriptor changed")
    return _regular_identity(path)


def _predecessor(
    *,
    budget,
    transport,
    failure_proof_path,
    failure_proof_sha256,
    recovery_proof_path,
    recovery_proof_sha256,
    probe_source_root,
    debug_source_root,
    probe_ref,
    debug_ref,
    interrupted_chunk_path,
    debug_commit,
    debug_review_sha256,
    protocol_sha256,
    original_source_commit,
    original_executable_sha256,
    original_source_authority_sha256,
):
    source = _SourceWitness(
        source_commit=original_source_commit,
        executable_sha256=original_executable_sha256,
        source_authority_sha256=original_source_authority_sha256,
    )
    driver = _historical_source(budget, source)
    for digest in (debug_review_sha256, protocol_sha256):
        TypeAdapter(Hash).validate_python(digest)
    q._debug_projection(debug_commit, debug_ref.logical_root)
    identity = q._qualification_identity(
        source=source,
        protocol_sha256=protocol_sha256,
        policy=budget.policy,
        debug_commit_sha256=sha256_bytes(canonical_json_bytes(asdict(debug_commit))),
        debug_review_sha256=debug_review_sha256,
    )
    run_id = "qualification-" + sha256_bytes(canonical_json_bytes(identity))
    old = probe_source_root.parent
    if (
        probe_source_root != old / "source-probe"
        or debug_source_root != old / "source-debug"
        or (old / "qualification-result.json").exists()
    ):
        raise ValueError("qualification predecessor roots or result differ")
    for path in (probe_source_root, debug_source_root, failure_proof_path, recovery_proof_path):
        budget.require_path(path)
    manifests = []
    expected_files = (
        (FileEntry("probe/probe.bin", q._PROBE_SHA256, q._PROBE_BYTES),),
        debug_commit.owned,
    )
    for ref, root, expected, kind in zip(
        (probe_ref, debug_ref),
        (probe_source_root, debug_source_root),
        expected_files,
        ("diagnostic", "episode_pack"),
        strict=True,
    ):
        with _opened_unit(budget.workspace / "control", ref) as (_, manifest):
            if (
                manifest.kind != kind
                or manifest.identity.run_id != run_id
                or manifest.identity.source_commit != source.source_commit
                or manifest.identity.config_sha256 != protocol_sha256
                or manifest.identity.evidence_identity_sha256 != identity["debug_commit_sha256"]
                or manifest.policy_sha256 != identity["policy_sha256"]
                or not manifest.identity.writer_stopped
                or manifest.identity.checkpoint_committed
                or manifest.identity.checkpoint_sha256 is not None
                or manifest.borrowed
            ):
                raise ValueError("qualification original manifest binding differs")
            if tuple(iter_unit_files(budget.workspace / "control", ref)) != expected:
                raise ValueError("qualification original inventory differs")
            if kind == "episode_pack" and (
                len(manifest.episode_groups) != 1
                or manifest.episode_groups[0].paths != tuple(entry.path for entry in expected)
            ):
                raise ValueError("qualification original debug group differs")
            for entry in expected:
                _file_digest(root, Path(entry.path), entry)
            manifests.append(manifest)
    first = manifests[0].chunks[0]
    _interrupted_input(budget, interrupted_chunk_path, first)
    key = _object_key(run_id, f"objects/{first.sha256}.bin")
    key_hash = sha256_bytes(key.encode())
    failure = _proof(
        failure_proof_path,
        failure_proof_sha256,
        (
            "schema_version",
            "events",
            "result_absent",
            "source_authority_sha256",
            "source_commit",
            "unit",
        ),
    )
    if (
        failure["schema_version"] != "phase4-controller-qualification-failure-v1"
        or failure["result_absent"] is not True
        or failure["source_commit"] != source.source_commit
        or failure["source_authority_sha256"] != source.source_authority_sha256
        or failure["unit"] != asdict(probe_ref)
    ):
        raise ValueError("qualification failure proof binding differs")
    phases, progress = _failure_phases(failure["events"], old / "events", key_hash)
    recovery = _proof(
        recovery_proof_path,
        recovery_proof_sha256,
        (
            "schema_version",
            "status",
            "before",
            "after",
            "bounds",
            "provider_observations",
            "receipt_sha256",
            "seconds",
            "source_authority_sha256",
            "source_commit",
            "unit_id",
        ),
    )
    observed = _Observation.model_validate(recovery["provider_observations"])
    before = q._RemoteReservationSnapshot.model_validate(recovery["before"])
    after = q._RemoteReservationSnapshot.model_validate(recovery["after"])
    if (
        recovery["schema_version"] != "phase4-controller-resume-diagnostic-v1"
        or recovery["status"] != "original-unit-resume-completed-qualification-still-failed"
        or recovery["source_commit"] != source.source_commit
        or recovery["source_authority_sha256"] != source.source_authority_sha256
        or recovery["unit_id"] != probe_ref.unit_id
        or observed.conflicts < 1
        or not observed.conflict_readback
        or observed.downloads < 1
        or observed.object_key_sha256 != key_hash
        or any(
            s.transport_id != _transport_identity(transport)
            or s.reservation_pending
            or s.operational_pending
            for s in (before, after)
        )
        or before.accounted_bytes != after.accounted_bytes
        or before.reserved_bytes > after.reserved_bytes
    ):
        raise ValueError("qualification recovery proof binding differs")
    control = budget.workspace / "control"
    receipt, _ = _read_receipt(
        control, probe_ref, recovery["receipt_sha256"], budget.policy.metadata_bytes
    )
    if (
        receipt.get("transport_id") != _transport_identity(transport)
        or receipt.get("run_id") != run_id
        or receipt.get("policy") != budget.policy.model_dump()
        or not _active_receipt_authorized(
            control_dir=control, ref=probe_ref, receipt_sha256=recovery["receipt_sha256"]
        )
    ):
        raise ValueError("qualification recovered receipt binding differs")
    _verify_receipt_catalog(
        control_dir=control, ref=probe_ref, receipt=receipt, policy=budget.policy
    )
    witness = _PredecessorWitness(
        source=source,
        historical_driver_sha256=driver,
        run_id=run_id,
        policy_sha256=identity["policy_sha256"],
        protocol_sha256=protocol_sha256,
        transport_id=_transport_identity(transport),
        failure_proof_sha256=failure_proof_sha256,
        recovery_proof_sha256=recovery_proof_sha256,
        probe_ref=probe_ref,
        debug_ref=debug_ref,
        probe_inventory_sha256=manifests[0].inventory_sha256,
        debug_inventory_sha256=manifests[1].inventory_sha256,
        debug_commit_sha256=identity["debug_commit_sha256"],
        debug_review_sha256=debug_review_sha256,
        phases=phases,
        progress_sha256=progress,
        interrupted_chunk_index=0,
        interrupted_chunk_bytes=first.bytes,
        interrupted_chunk_sha256=first.sha256,
        object_key_sha256=key_hash,
        recovered_receipt_sha256=recovery["receipt_sha256"],
        receipt_absent_before_resume=True,
        authorization_absent_before_resume=True,
        historical_result_absent=True,
        resume_failure_cause="unknown",
        old_attempt_tree_sha256=_tree_digest(old),
    )
    return witness, manifests, receipt


def _continuation_bounds(
    *,
    policy,
    run_id,
    probe,
    debug,
    debug_logical_root,
    block,
    prior,
    reservations,
    completed,
    receipt_object_bytes,
):
    if type(block) is not int or block <= 0 or block % 512:
        raise ValueError("invalid continuation allocation unit")
    prior_run = (
        1
        + len(probe)
        + len({p.as_posix() for e in probe for p in Path(e.path).parents if p.as_posix() != "."})
    )
    unit = preflight._archive_unit_bounds(
        policy=policy,
        run_id=run_id,
        logical_root=debug_logical_root,
        selected=debug,
        kind="episode_pack",
        episode_groups=(tuple(e.path for e in debug),),
        prior_operational_records=prior,
        local_reservations=reservations,
        completed_count=len(completed),
        prior_run_records=prior_run,
    )

    def rounded(amount):
        return ((amount + block - 1) // block) * block

    files = (*probe, *debug)
    cache = (
        sum(rounded(e.bytes) for e in files)
        + (sum(len(Path(e.path).parents) for e in files) + 12) * block
    )
    extras = sum(len(raw) for _, _, raw, _ in completed)
    maximum = min(sum(e.bytes for e in debug), policy.chunk_bytes)
    # Sequential collision/retry/restore needs one download. Debug publication
    # additionally coexists with its payload and both complete catalog stages.
    scratch = (
        max(
            rounded(max(receipt_object_bytes, default=0)),
            rounded(min(sum(e.bytes for e in probe), policy.chunk_bytes)),
            2
            * rounded(
                max(
                    maximum,
                    unit.manifest,
                    unit.receipt,
                    min(policy.page_bytes, unit.run_catalog),
                    min(policy.page_bytes, unit.operational_catalog),
                )
            )
            + unit.run_catalog
            + unit.operational_catalog
            + (2 * (unit.run_nodes + unit.operational_nodes + 2) + 12) * block,
        )
        + 8 * block
    )
    metadata = (
        unit.inventory
        + unit.manifest
        + 2 * unit.run_catalog
        + unit.operational_catalog
        + 2 * unit.receipt
        + unit.pending
        + (unit.objects + reservations) * 32768
        + (2 * (unit.shards + unit.run_nodes + unit.operational_nodes + unit.objects + 16) + 32)
        * block
        + 2 * extras
        + 4 * policy.page_bytes
        + 2 * 65536
        + 8 * block
    )
    logs = q._EVENT_COUNT * rounded(q._EVENT_LIMIT) + 2 * max(q._EVENT_LIMIT, block) + 4 * block
    calls = (
        32
        * 257
        * (
            unit.objects
            + unit.run_nodes
            + unit.operational_nodes
            + prior
            + reservations
            + len(completed)
            + len(receipt_object_bytes)
            + 3
        )
    )
    return q._QualificationBounds(
        q._LocalAllocation(cache=cache, metadata=metadata, scratch=scratch, logs=logs),
        unit.remote + 2 * extras,
        (unit,),
        calls,
    )


def _require_campaign(snapshot, remaining, *, remote_limit=512 * 1024**2):
    if (
        snapshot.reservation_pending
        or snapshot.operational_pending
        or snapshot.reserved_bytes < snapshot.accounted_bytes
        or snapshot.reserved_bytes - snapshot.accounted_bytes + remaining > q._REMOTE_STOP
        or snapshot.reserved_bytes + remaining > 512 * 1024**2
        or snapshot.reserved_bytes + remaining > remote_limit
    ):
        raise StorageBlocked("storage_blocked: cumulative qualification allowance")


def _continue_qualification(
    *,
    budget,
    transport,
    output_root,
    failure_proof_path,
    failure_proof_sha256,
    recovery_proof_path,
    recovery_proof_sha256,
    probe_source_root,
    debug_source_root,
    probe_ref,
    debug_ref,
    interrupted_chunk_path,
    debug_commit,
    debug_review_sha256,
    protocol_sha256,
    original_source_commit,
    original_executable_sha256,
    original_source_authority_sha256,
    source_commit,
    source_authority_sha256,
):
    control, policy = budget.workspace / "control", budget.policy
    budget.require_path(output_root)
    old = probe_source_root.parent
    if (
        output_root.exists()
        or output_root == budget.workspace
        or output_root.is_relative_to(control)
        or output_root.is_relative_to(old)
        or old.is_relative_to(output_root)
    ):
        raise ValueError("qualification continuation output is not fresh and separate")
    roots = {
        output_root: "metadata",
        output_root / "events": "logs",
        output_root / "restore-probe": "cache",
        output_root / "restore-debug": "cache",
        control: "metadata",
        control / "transfer-scratch": "scratch",
    }
    state = budget._state()
    for root, category in roots.items():
        budget.require_path(root)
        if state["paths"].get(root.relative_to(budget.workspace).as_posix()) != category:
            raise ValueError("qualification continuation category handoff differs")
    _regular_identity(budget.root / "locks/engineering.lock")
    with _lock(
        control_dir=budget.root, relative=("engineering.lock",), shared=False, blocking=True
    ):
        executor = _SourceWitness(
            **asdict(
                preflight._reviewed_source(
                    budget=budget,
                    source_commit=source_commit,
                    authority_sha256=source_authority_sha256,
                )
            )
        )
        predecessor, manifests, receipt = _predecessor(
            budget=budget,
            transport=transport,
            failure_proof_path=failure_proof_path,
            failure_proof_sha256=failure_proof_sha256,
            recovery_proof_path=recovery_proof_path,
            recovery_proof_sha256=recovery_proof_sha256,
            probe_source_root=probe_source_root,
            debug_source_root=debug_source_root,
            probe_ref=probe_ref,
            debug_ref=debug_ref,
            interrupted_chunk_path=interrupted_chunk_path,
            debug_commit=debug_commit,
            debug_review_sha256=debug_review_sha256,
            protocol_sha256=protocol_sha256,
            original_source_commit=original_source_commit,
            original_executable_sha256=original_executable_sha256,
            original_source_authority_sha256=original_source_authority_sha256,
        )
        snapshot_args = dict(control_dir=control, transport=transport, policy=policy)
        before_remote = q._remote_reservation_snapshot(**snapshot_args)
        with (
            archive_operation_lock(control),
            _opened_ledger(control, transport_id=_transport_identity(transport), policy=policy) as (
                remote_state,
                _,
                _,
            ),
        ):
            head = _operational_ref(remote_state)
            prior = 0 if head is None else head.entry_count
            reservations = len(_local_reservation_records(control))
            completed = _completed_intents(control)
            if _load_head(control, predecessor.run_id) is None:
                raise ValueError("qualification original run head unavailable")
        probe = tuple(iter_unit_files(control, probe_ref))
        bound = _continuation_bounds(
            policy=policy,
            run_id=predecessor.run_id,
            probe=probe,
            debug=debug_commit.owned,
            debug_logical_root=debug_ref.logical_root,
            block=preflight._allocation_unit(os.statvfs(budget.workspace)),
            prior=prior,
            reservations=reservations,
            completed=completed,
            receipt_object_bytes=tuple(item["bytes"] for item in receipt["objects"]),
        )
        _require_campaign(before_remote, bound.remote, remote_limit=policy.remote_bytes)
        continuation_id = sha256_bytes(
            canonical_json_bytes(
                {
                    "executor": executor.model_dump(mode="json"),
                    "predecessor": predecessor.model_dump(mode="json"),
                    "output": output_root.relative_to(budget.workspace).as_posix(),
                }
            )
        )
        with budget._scoped_reservation(
            admission={"qualification_continuation": continuation_id},
            child_inheritable=True,
            **{name: getattr(bound.local, name) for name in q._CATEGORIES},
        ) as admitted:
            before = q._LocalAllocation(
                **admitted.before, retained_bytes=admitted.retained.retained_bytes
            )
            peak = before.model_dump()

            def observe():
                measured = budget.check_scoped(admitted.retained)
                for name in q._CATEGORIES:
                    peak[name] = max(peak[name], measured[name])
                return measured

            with _pinned_directory(output_root, create=True):
                pass
            events = output_root / "events"
            q._event(events, phase="admit", status="complete")
            observed = q._ObservedTransport(
                transport, budget, "resume", events, None, None, admitted.retained
            )
            chunk = manifests[0].chunks[0]
            chunk_identity = _interrupted_input(budget, interrupted_chunk_path, chunk)
            with (
                archive_operation_lock(control),
                _source_file(interrupted_chunk_path.parent, Path(interrupted_chunk_path.name)),
            ):
                _put_verified(
                    control_dir=control,
                    transport=observed,
                    key=_object_key(predecessor.run_id, f"objects/{chunk.sha256}.bin"),
                    source=interrupted_chunk_path,
                    expected_bytes=chunk.bytes,
                    expected_sha256=chunk.sha256,
                    policy=policy,
                )
            collision = _Observation.model_validate(observed.summary)
            if (
                collision.conflicts != 1
                or not collision.conflict_readback
                or collision.object_key_sha256 != predecessor.object_key_sha256
                or _regular_identity(interrupted_chunk_path) != chunk_identity
                or q._remote_reservation_snapshot(**snapshot_args) != before_remote
            ):
                raise ValueError("qualification continuation lacks original-key collision")
            q._event(events, phase="resume", status="complete", summary=collision.model_dump())

            def phase(mode, index):
                result = q._spawn_archive_phase(
                    run_dir=(probe_source_root, debug_source_root)[index],
                    control_dir=control,
                    ref=(probe_ref, debug_ref)[index],
                    policy=policy,
                    transport=transport,
                    phase=mode,
                    transport_calls=bound.transport_calls,
                    observe_local=observe,
                    scope=admitted.retained,
                    event_root=events,
                )
                observation = _Observation.model_validate(
                    {
                        key: value
                        for key, value in result.items()
                        if key not in {"status", "receipt_sha256"}
                    }
                )
                for name in q._CATEGORIES:
                    peak[name] = max(peak[name], getattr(observation.local_peak, name))
                return result["receipt_sha256"], observation

            repeated_receipt, repeated = phase("repeat", 0)
            if (
                repeated_receipt != predecessor.recovered_receipt_sha256
                or repeated.downloads == 0
                or q._remote_reservation_snapshot(**snapshot_args) != before_remote
            ):
                raise ValueError("qualification continuation completed retry differs")
            try:
                q._restore_receipted_unit(
                    control_dir=control,
                    ref=probe_ref,
                    receipt_sha256=repeated_receipt,
                    transport=transport,
                    destination=probe_source_root,
                    policy=policy,
                    observe_local=observe,
                )
            except FileExistsError:
                pass
            else:
                raise ValueError("qualification continuation replaced existing destination")
            results = []
            debug_observation = None
            for index, ref in enumerate((probe_ref, debug_ref)):
                receipt_hash = repeated_receipt
                if index:
                    receipt_hash, debug_observation = phase("debug", index)
                destination = output_root / ("restore-probe", "restore-debug")[index] / "tree"
                entries = q._restore_receipted_unit(
                    control_dir=control,
                    ref=ref,
                    receipt_sha256=receipt_hash,
                    transport=transport,
                    destination=destination,
                    policy=policy,
                    observe_local=observe,
                )
                expected = (probe, debug_commit.owned)[index]
                if entries != expected:
                    raise ValueError("qualification continuation restored inventory differs")
                for entry in expected:
                    source_root = (probe_source_root, debug_source_root)[index]
                    _file_digest(source_root, Path(entry.path), entry)
                    _file_digest(destination, Path(entry.path), entry)
                    with (
                        _source_file(source_root, Path(entry.path)) as (left, _),
                        _source_file(destination, Path(entry.path)) as (right, _),
                    ):
                        while block := os.read(left, 1024**2):
                            if block != os.read(right, len(block)):
                                raise ValueError("qualification continuation restored bytes differ")
                        if os.read(right, 1):
                            raise ValueError("qualification continuation restore has extra bytes")
                results.append(
                    q._UnitResult(
                        unit_id=ref.unit_id,
                        receipt_sha256=receipt_hash,
                        restored_inventory_sha256=sha256_bytes(
                            canonical_json_bytes({"files": [asdict(entry) for entry in entries]})
                        ),
                    )
                )
                q._event(
                    events, phase="restore", status="complete", summary={"unit_id": ref.unit_id}
                )
            after_remote = q._remote_reservation_snapshot(**snapshot_args)
            _require_campaign(after_remote, 0, remote_limit=policy.remote_bytes)
            delta = after_remote.reserved_bytes - before_remote.reserved_bytes
            if (
                not 0 <= delta <= bound.remote
                or after_remote.transport_id != before_remote.transport_id
                or after_remote.accounted_bytes != before_remote.accounted_bytes
                or _tree_digest(old) != predecessor.old_attempt_tree_sha256
            ):
                raise ValueError("qualification continuation final history differs")
            for name in q._CATEGORIES:
                peak[name] = max(peak[name], getattr(collision.local_peak, name))
            q._event(events, phase="complete", status="complete")
            log = hashlib.sha256()
            for path in sorted(events.glob("*.json")):
                log.update(_read_regular(path, max_bytes=q._EVENT_LIMIT))
            result = QualificationContinuationResult(
                executor=executor,
                predecessor=predecessor,
                continuation_id=continuation_id,
                run_id=predecessor.run_id,
                protocol_sha256=protocol_sha256,
                policy_sha256=predecessor.policy_sha256,
                transport_id=_transport_identity(transport),
                probe_sha256=q._PROBE_SHA256,
                debug_commit_sha256=predecessor.debug_commit_sha256,
                debug_review_sha256=debug_review_sha256,
                probe=results[0],
                debug=results[1],
                inherited_interruption=q._InterruptionResult(
                    signal=9,
                    object_key_sha256=predecessor.object_key_sha256,
                    receipt_absent=True,
                    authorization_absent=True,
                ),
                retry=q._RetryResult(
                    provider_conflict=True,
                    object_key_sha256=predecessor.object_key_sha256,
                    exact_readback=True,
                    stable_receipt=True,
                    stable_reserved_bytes=True,
                ),
                collision_observation=collision,
                repeat_observation=repeated,
                debug_observation=debug_observation,
                existing_destination_refused=True,
                remote_before=before_remote,
                remote_after=after_remote,
                remaining_remote_bound=bound.remote,
                continuation_remote_bytes=delta,
                cumulative_campaign_remote_bytes=after_remote.reserved_bytes
                - after_remote.accounted_bytes,
                cumulative_shared_remote_bytes=after_remote.reserved_bytes,
                local_bound=bound.local,
                local_before=before,
                local_peak=q._LocalAllocation(**peak),
                local_after=q._LocalAllocation(**observe(), retained_bytes=before.retained_bytes),
                event_log_sha256=log.hexdigest(),
                old_attempt_tree_sha256=predecessor.old_attempt_tree_sha256,
            )

            def authenticate():
                for path, expected_hash in (
                    (failure_proof_path, failure_proof_sha256),
                    (recovery_proof_path, recovery_proof_sha256),
                ):
                    _regular_identity(path)
                    if sha256_bytes(_read_regular(path, max_bytes=65536)) != expected_hash:
                        raise ValueError("qualification preserved proof changed")
                if (
                    _interrupted_input(budget, interrupted_chunk_path, chunk) != chunk_identity
                    or _tree_digest(old) != predecessor.old_attempt_tree_sha256
                ):
                    raise ValueError("qualification preserved input changed")
                return budget.check()

            return q._publish_result(
                output_root,
                result,
                observe,
                peak,
                authenticate,
                result_name="qualification-continuation-result.json",
            )


def _failure_phases(events, root: Path, key: str):
    expected = (
        ("00-admit.json", "admit", "complete", None),
        ("01-seal.json", "seal", "complete", None),
        ("first-create.json", "interrupt", "blocked", {"object_key_sha256": key}),
        ("03-interrupt.json", "interrupt", "complete", {"object_key_sha256": key, "signal": 9}),
        ("04-resume.json", "resume", "failed", None),
    )
    if type(events) is not dict or set(events) != {row[0] for row in expected} | {"progress.json"}:
        raise ValueError("qualification historical phase set differs")
    with _pinned_directory(root) as directory:
        if set(os.listdir(directory)) != set(events):
            raise ValueError("qualification historical event directory differs")
    witnesses = []
    for index, (name, phase, status, summary) in enumerate(expected):
        value = {"sequence": index, "phase": phase, "status": status}
        if summary is not None:
            value["summary"] = summary
        raw = canonical_json_bytes(value) + b"\n"
        _match_event(events[name], root / name, raw)
        witnesses.append(
            _PhaseWitness(
                name=name,
                sha256=sha256_bytes(raw),
                sequence=index,
                phase=phase,
                status=status,
                **(summary or {}),
            )
        )
    raw = canonical_json_bytes({"phase": "resume", "status": "waiting"}) + b"\n"
    _match_event(events["progress.json"], root / "progress.json", raw)
    return tuple(witnesses), sha256_bytes(raw)


def _match_event(record, path, raw):
    if (
        record != {"raw_utf8": raw.decode(), "sha256": sha256_bytes(raw)}
        or _read_regular(path, max_bytes=q._EVENT_LIMIT) != raw
    ):
        raise ValueError("qualification historical phase bytes differ")
