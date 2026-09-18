"""Private qualification compositions exercised only with small local doubles."""

import json
import os
import subprocess
import sys

import pytest
from conftest import DirectoryTransport

from silent_cascade.hashing import canonical_json_bytes, sha256_bytes


def _archive(unit, transport):
    from silent_cascade.archive.transport import archive_unit

    return archive_unit(
        run_dir=unit.root,
        control_dir=unit.control,
        ref=unit.ref,
        transport=transport,
        policy=unit.policy,
    )


def _restore(unit, transport, receipt, destination, observe=lambda: None):
    from silent_cascade.archive._qualification import _restore_receipted_unit

    return _restore_receipted_unit(
        control_dir=unit.control,
        ref=unit.ref,
        receipt_sha256=receipt,
        transport=transport,
        destination=destination,
        policy=unit.policy,
        observe_local=observe,
    )


def test_snapshot_requires_precreated_locks_without_creating_output(
    task_scratch, transport, tiny_archive_policy
):
    from silent_cascade.archive._qualification import _remote_reservation_snapshot

    control = task_scratch / "control"
    before = sorted(str(path.relative_to(control)) for path in control.rglob("*"))
    with pytest.raises(FileNotFoundError):
        _remote_reservation_snapshot(
            control_dir=control, transport=transport, policy=tiny_archive_policy
        )
    assert sorted(str(path.relative_to(control)) for path in control.rglob("*")) == before


@pytest.mark.parametrize("pending", [None, "pending.json", "operational-pending.json"])
def test_snapshot_validates_canonical_history_and_reports_pending(
    task_scratch, transport, tiny_archive_policy, pending
):
    from silent_cascade.archive._qualification import _remote_reservation_snapshot
    from silent_cascade.archive.transport import archive_operation_lock

    control = task_scratch / "control"
    with archive_operation_lock(control):
        pass
    if pending:
        (control / "remote-reservations" / pending).write_bytes(b"preserved")
    state = control / "remote-reservations/state.json"
    before = state.read_bytes()
    snapshot = _remote_reservation_snapshot(
        control_dir=control, transport=transport, policy=tiny_archive_policy
    )
    assert snapshot.reserved_bytes == 0
    assert snapshot.transport_id == transport.transport_id
    assert snapshot.reservation_pending == (pending == "pending.json")
    assert snapshot.operational_pending == (pending == "operational-pending.json")
    assert state.read_bytes() == before


def test_receipted_restore_streams_exact_inventory_and_refuses_overwrite(
    sealed_unit, transport, task_scratch
):
    from silent_cascade.archive.catalog import iter_unit_files

    receipt = _archive(sealed_unit, transport)
    destination = task_scratch / "restored"
    observations = []

    def observe():
        observations.append(
            tuple((sealed_unit.control / "transfer-scratch").glob("readback-*.part"))
        )

    assert _restore(sealed_unit, transport, receipt, destination, observe) == tuple(
        iter_unit_files(sealed_unit.control, sealed_unit.ref)
    )
    assert {
        path: (destination / path).read_bytes() for path in sealed_unit.paths
    } == sealed_unit.original_bytes()
    assert observations and max(map(len, observations)) == 1
    assert not tuple((sealed_unit.control / "transfer-scratch").glob("readback-*.part"))
    calls = transport.download_calls
    with pytest.raises(FileExistsError):
        _restore(sealed_unit, transport, receipt, destination)
    assert transport.download_calls == calls


@pytest.mark.parametrize(
    "mutation",
    [
        "transport",
        "run",
        "policy",
        "missing",
        "duplicate",
        "extra",
        "authorization",
        "hash",
        "corrupt",
    ],
)
def test_restore_rejects_foreign_or_incomplete_receipt(
    sealed_unit, transport, task_scratch, mutation
):
    receipt = _archive(sealed_unit, transport)
    leaf = sealed_unit.control / f"receipts/{sealed_unit.ref.unit_id}.json"
    value = json.loads(leaf.read_bytes())
    if mutation == "transport":
        value["transport_id"] = "foreign"
    elif mutation == "run":
        value["run_id"] = "foreign"
    elif mutation == "policy":
        value["policy"]["remote_bytes"] += 1
    elif mutation == "missing":
        value["objects"].pop(0)
    elif mutation == "duplicate":
        value["objects"].insert(0, value["objects"][0])
    elif mutation == "extra":
        value["objects"].append(value["objects"][0])
    elif mutation == "authorization":
        (sealed_unit.control / f"active-receipts/{sealed_unit.ref.unit_id}.json").unlink()
    elif mutation == "hash":
        receipt = "f" * 64
    elif mutation == "corrupt":
        transport.corrupt_downloads = True
    if mutation not in {"authorization", "hash", "corrupt"}:
        raw = canonical_json_bytes(value) + b"\n"
        leaf.write_bytes(raw)
        receipt = sha256_bytes(raw)
        authorization = sealed_unit.control / f"active-receipts/{sealed_unit.ref.unit_id}.json"
        active = json.loads(authorization.read_bytes())
        active["receipt_sha256"] = receipt
        authorization.write_bytes(canonical_json_bytes(active) + b"\n")
    destination = task_scratch / "refused"
    with pytest.raises(ValueError):
        _restore(sealed_unit, transport, receipt, destination)
    assert not destination.exists()


def test_shared_sizing_includes_long_identity_and_episode_group(tiny_archive_policy):
    from silent_cascade.archive.preflight import _archive_unit_bounds
    from silent_cascade.archive.types import FileEntry

    arguments = dict(
        policy=tiny_archive_policy,
        logical_root="pack",
        selected=(FileEntry("pack/file", "a" * 64, 17),),
        run_id="engineering-" + "a" * 64,
        kind="diagnostic",
        episode_groups=(),
        prior_operational_records=0,
        local_reservations=0,
        completed_count=0,
    )
    first = _archive_unit_bounds(**arguments)
    second = _archive_unit_bounds(
        **(
            arguments
            | {"run_id": "q" * 256, "kind": "episode_pack", "episode_groups": (("pack/file",),)}
        )
    )
    assert second.manifest > first.manifest + 180
    assert first.objects >= 4
    assert first.remote > 17 + first.manifest + first.inventory
    with pytest.raises(ValueError):
        _archive_unit_bounds(**(arguments | {"selected": (FileEntry("../escape", "a" * 64, 17),)}))
    with pytest.raises(ValueError):
        _archive_unit_bounds(**(arguments | {"prior_operational_records": -1}))


def test_probe_recipe_streams_exact_fixed_digest():
    import hashlib

    from silent_cascade.archive._qualification import _probe_blocks

    digest = hashlib.sha256()
    size = 0
    for block in _probe_blocks():
        size += len(block)
        digest.update(block)
    assert size == 67_108_864
    assert digest.hexdigest() == "e04dc3863c905cbf429e51f9a248159ccbc34688763f871fd20d2dfd8964df34"


@pytest.fixture
def process_case(task_scratch, archive_identity):
    from conftest import DirectoryTransport, SealedUnit

    from silent_cascade.archive.catalog import seal_unit
    from silent_cascade.archive.ledger import _StorageBudget, initialize_workspace_ledger
    from silent_cascade.archive.transport import initialize_remote_reservations
    from silent_cascade.archive.types import ArchivePolicy

    workspace = task_scratch / "workspace"
    policy = ArchivePolicy(chunk_bytes=64, page_bytes=4096)
    initialize_workspace_ledger(workspace_root=workspace, policy=policy, baseline=())
    budget = _StorageBudget(workspace=workspace, policy=policy)
    root = workspace / "qualification/source-probe"
    root.mkdir(parents=True)
    (root / "probe").mkdir()
    (root / "probe/probe.bin").write_bytes(bytes(range(160)))
    control = workspace / "control"
    budget.bind(control, category="metadata")
    budget.bind(control / "transfer-scratch", category="scratch")
    budget.bind(root, category="spool")
    budget.bind(root.parent / "events", category="logs")
    transport = DirectoryTransport(task_scratch / "remote")
    initialize_remote_reservations(
        control_dir=control,
        transport_id=transport.transport_id,
        accounted_bytes=11,
        accounting_evidence_sha256="a" * 64,
        policy=policy,
    )
    ref = seal_unit(
        run_dir=root,
        control_dir=control,
        logical_root="probe",
        paths=("probe/probe.bin",),
        kind="diagnostic",
        identity=archive_identity,
        policy=policy,
    )
    return budget, SealedUnit(root, control, ref, policy, ("probe/probe.bin",)), transport


def test_spawn_kill_resume_and_completed_repeat_preserve_one_history(process_case):
    import signal

    from silent_cascade.archive._qualification import (
        _remote_reservation_snapshot,
        _spawn_archive_phase,
    )

    budget, unit, transport = process_case
    arguments = dict(
        run_dir=unit.root,
        control_dir=unit.control,
        ref=unit.ref,
        policy=unit.policy,
        transport=transport,
        transport_calls=10000,
        observe_local=budget.check,
    )
    with budget.reserve(metadata=8 * 1024**2, scratch=8 * 1024**2, logs=256 * 1024):
        first = _spawn_archive_phase(**arguments, phase="interrupt")
        assert first["signal"] == signal.SIGKILL
        assert not (unit.control / f"receipts/{unit.ref.unit_id}.json").exists()
        assert not (unit.control / f"active-receipts/{unit.ref.unit_id}.json").exists()
        stranded = tuple((unit.control / "transfer-scratch").rglob("chunk-*.bin"))
        assert stranded
        resumed = _spawn_archive_phase(**arguments, phase="resume")
        assert resumed["conflicts"] >= 1 and resumed["conflict_readback"]
        snapshot = _remote_reservation_snapshot(
            control_dir=unit.control, transport=transport, policy=unit.policy
        )
        repeated = _spawn_archive_phase(**arguments, phase="repeat")
        assert repeated["receipt_sha256"] == resumed["receipt_sha256"]
        assert repeated["downloads"] > 0 and repeated["creates"] == 0
        after = _remote_reservation_snapshot(
            control_dir=unit.control, transport=transport, policy=unit.policy
        )
        assert after.reserved_bytes == snapshot.reserved_bytes > 11
        assert all(path.exists() for path in stranded)


def test_spawn_failure_is_sanitized_and_reaped(process_case):
    from silent_cascade.archive._qualification import _spawn_archive_phase
    from silent_cascade.archive.ledger import StorageBlocked

    budget, unit, transport = process_case
    transport.fail_create_call = 1
    with pytest.raises(StorageBlocked) as raised:
        _spawn_archive_phase(
            run_dir=unit.root,
            control_dir=unit.control,
            ref=unit.ref,
            policy=unit.policy,
            transport=transport,
            transport_calls=100,
            observe_local=budget.check,
            phase="resume",
        )
    assert "secret-token" not in str(raised.value)


def test_observed_transport_archive_checks_local_ledger(process_case):
    from silent_cascade.archive._qualification import _ObservedTransport

    budget, unit, transport = process_case
    observer = _ObservedTransport(
        transport, budget, "resume", unit.root.parent / "events", None, None
    )
    assert len(_archive(unit, observer)) == 64
    assert observer.summary["creates"] > 0


def test_two_unit_bound_includes_retained_copies_extra_killed_chunk_and_prior_run(
    process_case, monkeypatch
):
    from silent_cascade.archive._qualification import _qualification_bounds
    from silent_cascade.archive.catalog import iter_unit_files, seal_unit
    from silent_cascade.archive.types import FileEntry

    budget, unit, _transport = process_case
    entries = tuple(iter_unit_files(unit.control, unit.ref))
    (unit.root / "evaluation").mkdir()
    (unit.root / "evaluation/debug.bin").write_bytes(b"debug")
    debug = (FileEntry("evaluation/debug.bin", sha256_bytes(b"debug"), 5),)
    debug_ref = seal_unit(
        run_dir=unit.root,
        control_dir=unit.control,
        logical_root="evaluation",
        paths=(debug[0].path,),
        kind="episode_pack",
        policy=unit.policy,
        identity={
            "run_id": "debug-fixture",
            "source_commit": "0" * 40,
            "config_sha256": "1" * 64,
            "evidence_identity_sha256": "2" * 64,
            "checkpoint_sha256": None,
            "writer_stopped": True,
            "checkpoint_committed": False,
        },
        episode_groups=((debug[0].path,),),
    )
    bounds = _qualification_bounds(
        policy=unit.policy,
        run_id="debug-fixture",
        probe=entries,
        debug=debug,
        debug_logical_root="evaluation",
        block=4096,
        prior=7,
        reservations=3,
        completed=(),
    )
    assert bounds.local.spool >= 165 and bounds.local.cache >= 165
    assert bounds.local.scratch >= 3 * 64
    assert bounds.units[1].run_catalog > bounds.units[0].run_catalog
    assert bounds.units[1].operational_catalog > bounds.units[0].operational_catalog
    assert bounds.remote > 165
    before = budget.measure()
    peak = before.copy()
    original_fsync = os.fsync

    def measured_fsync(descriptor):
        original_fsync(descriptor)
        for name, amount in budget.measure().items():
            peak[name] = max(peak[name], amount)

    monkeypatch.setattr(os, "fsync", measured_fsync)
    with budget.reserve(**{name: getattr(bounds.local, name) for name in before}):
        _archive(unit, _transport)
        from silent_cascade.archive.transport import archive_unit

        archive_unit(
            run_dir=unit.root,
            control_dir=unit.control,
            ref=debug_ref,
            policy=unit.policy,
            transport=_transport,
        )
        after = budget.check()
    assert all(after[name] - before[name] <= getattr(bounds.local, name) for name in before)
    assert all(peak[name] - before[name] <= getattr(bounds.local, name) for name in before)


def test_projection_copies_only_exact_owned_paths_and_refuses_wrong_prefix(task_scratch):
    from silent_cascade.archive._qualification import _copy_debug, _debug_projection
    from silent_cascade.archive.types import EpisodeCommit, FileEntry

    source = task_scratch / "original"
    source.mkdir()
    (source / "owned.json").write_bytes(b"owned")
    (source / "private.json").write_bytes(b"private")
    commit = EpisodeCommit(
        "phase4-evaluation-episode-commit-v1",
        "a" * 64,
        0,
        "episode",
        "b" * 64,
        0,
        1,
        "c" * 64,
        (FileEntry("evaluation/owned.json", sha256_bytes(b"owned"), 5),),
        (FileEntry("weights.bin", "d" * 64, 900),),
    )
    with pytest.raises(ValueError):
        _debug_projection(commit, "wrong")
    projection = _debug_projection(commit, "evaluation")
    destination = task_scratch / "copy"
    _copy_debug(source, destination, projection)
    assert (destination / "evaluation/owned.json").read_bytes() == b"owned"
    assert sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    ) == ["evaluation/owned.json"]
    assert (source / "private.json").read_bytes() == b"private"


def test_evidence_limits_refuse_oversized_record_and_33rd_event(task_scratch):
    from silent_cascade.archive._qualification import _bounded_record, _event
    from silent_cascade.archive.ledger import StorageBlocked

    with pytest.raises(StorageBlocked):
        _bounded_record({"hash": "a" * 16384}, 16384)
    events = task_scratch / "events"
    for _ in range(32):
        _event(events, phase="admit", status="complete")
    with pytest.raises(StorageBlocked):
        _event(events, phase="admit", status="complete")
    assert len(tuple(events.iterdir())) == 32


def test_controller_refuses_double_without_output(process_case):
    from silent_cascade.archive._qualification import run_early_r2_qualification
    from silent_cascade.archive.ledger import StorageBlocked

    budget, unit, transport = process_case
    destination = budget.workspace / "never-created"
    with pytest.raises(StorageBlocked, match="provider transport"):
        run_early_r2_qualification(
            budget=budget,
            output_root=destination,
            debug_source_root=unit.root,
            debug_logical_root="probe",
            debug_commit=None,
            debug_review_sha256="a" * 64,
            protocol_sha256="b" * 64,
            source_commit="c" * 40,
            transport=transport,
        )
    assert not destination.exists()


def test_bound_refusal_precedes_any_output_and_undergrant_cannot_pass(process_case):
    from dataclasses import replace

    from silent_cascade.archive._qualification import _qualification_bounds, _require_bounds
    from silent_cascade.archive.catalog import iter_unit_files
    from silent_cascade.archive.ledger import StorageBlocked

    budget, unit, _transport = process_case
    entries = tuple(iter_unit_files(unit.control, unit.ref))
    bounds = _qualification_bounds(
        policy=unit.policy,
        run_id="qualification-test",
        probe=entries,
        debug=entries,
        debug_logical_root="probe",
        block=4096,
        prior=0,
        reservations=0,
        completed=(),
    )
    before = budget.measure()
    with pytest.raises(StorageBlocked, match="remote stop"):
        _require_bounds(replace(bounds, remote=256 * 1024**2 + 1), bounds)
    with pytest.raises(StorageBlocked, match="underestimated"):
        _require_bounds(
            bounds, replace(bounds, local=bounds.local.model_copy(update={"scratch": 1}))
        )
    assert budget.measure() == before


def test_restore_rejects_reordered_and_missing_remote_chunks(process_case):
    budget, unit, transport = process_case
    receipt = _archive(unit, transport)
    leaf = unit.control / f"receipts/{unit.ref.unit_id}.json"
    value = json.loads(leaf.read_bytes())
    first, second = value["objects"][:2]
    value["objects"][:2] = [second, first]
    raw = canonical_json_bytes(value) + b"\n"
    leaf.write_bytes(raw)
    altered = sha256_bytes(raw)
    authorization = unit.control / f"active-receipts/{unit.ref.unit_id}.json"
    active = json.loads(authorization.read_bytes())
    active["receipt_sha256"] = altered
    authorization.write_bytes(canonical_json_bytes(active) + b"\n")
    with pytest.raises(ValueError, match="sequence"):
        _restore(unit, transport, altered, budget.workspace / "refused")
    value["objects"][:2] = [first, second]
    leaf.write_bytes(canonical_json_bytes(value) + b"\n")
    active["receipt_sha256"] = receipt
    authorization.write_bytes(canonical_json_bytes(active) + b"\n")
    transport._path(first["key"]).unlink()
    with pytest.raises(FileNotFoundError):
        _restore(unit, transport, receipt, budget.workspace / "missing")
    assert not (budget.workspace / "missing").exists()


class _DescendantTransport(DirectoryTransport):
    def create(self, key, source):
        super().create(key, source)
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(300)"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        (self.root / "descendant-pid").write_text(str(child.pid))


def test_interruption_terminates_owned_provider_descendant(process_case):
    from silent_cascade.archive._qualification import _spawn_archive_phase

    budget, unit, transport = process_case
    # Same exact local-double identity/history, with a real owned process descendant.
    transport.__class__ = _DescendantTransport
    result = _spawn_archive_phase(
        run_dir=unit.root,
        control_dir=unit.control,
        ref=unit.ref,
        policy=unit.policy,
        transport=transport,
        phase="interrupt",
        transport_calls=1000,
        observe_local=budget.check,
    )
    assert result["signal"] == 9
    pid = int((transport.root / "descendant-pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def _unreapable_parent(workspace, policy):
    from silent_cascade.archive._qualification import _stop_child
    from silent_cascade.archive.ledger import _StorageBudget

    class Unreaped:
        def kill(self):
            pass

        def join(self, _timeout):
            pass

        def is_alive(self):
            return True

    budget = _StorageBudget(workspace=workspace, policy=policy)
    with budget.reserve(metadata=1024**2, logs=1024**2):
        _stop_child(
            Unreaped(), owns_group=False, events=workspace / "qualification/events", phase="resume"
        )


def test_unreapable_child_fail_stop_preserves_durable_reservation(process_case):
    import multiprocessing

    budget, unit, _transport = process_case
    parent = multiprocessing.get_context("spawn").Process(
        target=_unreapable_parent, args=(budget.workspace, unit.policy)
    )
    parent.start()
    parent.join(10)
    assert parent.exitcode == 70
    reservations = budget._state()["reservations"]
    assert len(reservations) == 1
    assert next(iter(reservations.values()))["pid"] == parent.pid
    assert not (unit.root.parent / "qualification-result.json").exists()
    parent.close()
