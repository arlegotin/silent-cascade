"""Private qualification compositions exercised only with small local doubles."""

import json
import os
import subprocess
import sys
from pathlib import Path

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
    failed = json.loads((unit.root.parent / "events/00-resume.json").read_bytes())
    assert failed["summary"] == {"location": "archive", "type": "permission"}
    assert "secret-token" not in json.dumps(failed)


def test_explicit_phase_events_leave_original_attempt_unchanged(process_case):
    from silent_cascade.archive._qualification import _spawn_archive_phase

    budget, unit, transport = process_case
    original = {
        p.relative_to(unit.root.parent): p.read_bytes()
        for p in unit.root.parent.rglob("*")
        if p.is_file()
    }
    events = budget.workspace / "continuation/events"
    budget.bind(events, category="logs")
    result = _spawn_archive_phase(
        run_dir=unit.root,
        control_dir=unit.control,
        ref=unit.ref,
        policy=unit.policy,
        transport=transport,
        phase="resume",
        transport_calls=10000,
        observe_local=budget.check,
        event_root=events,
    )
    assert result["downloads"] > 0
    assert (events / "00-resume.json").is_file()
    assert original == {
        p.relative_to(unit.root.parent): p.read_bytes()
        for p in unit.root.parent.rglob("*")
        if p.is_file()
    }


def test_continuation_requires_provider_without_creating_output(process_case):
    from silent_cascade.archive._qualification import continue_early_r2_qualification
    from silent_cascade.archive.ledger import StorageBlocked

    budget, unit, transport = process_case
    destination = budget.workspace / "not-created"
    with pytest.raises(StorageBlocked, match="provider"):
        continue_early_r2_qualification(
            budget=budget,
            transport=transport,
            output_root=destination,
            failure_proof_path=unit.root,
            failure_proof_sha256="a" * 64,
            recovery_proof_path=unit.root,
            recovery_proof_sha256="b" * 64,
            probe_source_root=unit.root,
            debug_source_root=unit.root,
            probe_ref=unit.ref,
            debug_ref=unit.ref,
            interrupted_chunk_path=unit.root,
            debug_commit=None,
            debug_review_sha256="c" * 64,
            protocol_sha256="d" * 64,
            original_source_commit="0" * 40,
            original_executable_sha256="e" * 64,
            original_source_authority_sha256="f" * 64,
            source_commit="1" * 40,
            source_authority_sha256="2" * 64,
        )
    assert not destination.exists()
    assert transport.create_calls == transport.download_calls == 0


def test_continuation_history_requires_exact_order_and_raw_witnesses(task_scratch):
    from silent_cascade.archive._qualification_continuation import _failure_phases

    root = task_scratch / "events"
    root.mkdir()
    key = "7" * 64
    values = [
        ("00-admit.json", {"sequence": 0, "phase": "admit", "status": "complete"}),
        ("01-seal.json", {"sequence": 1, "phase": "seal", "status": "complete"}),
        (
            "first-create.json",
            {
                "sequence": 2,
                "phase": "interrupt",
                "status": "blocked",
                "summary": {"object_key_sha256": key},
            },
        ),
        (
            "03-interrupt.json",
            {
                "sequence": 3,
                "phase": "interrupt",
                "status": "complete",
                "summary": {"object_key_sha256": key, "signal": 9},
            },
        ),
        ("04-resume.json", {"sequence": 4, "phase": "resume", "status": "failed"}),
        ("progress.json", {"phase": "resume", "status": "waiting"}),
    ]
    events = {}
    for name, value in values:
        raw = canonical_json_bytes(value) + b"\n"
        (root / name).write_bytes(raw)
        events[name] = {"raw_utf8": raw.decode(), "sha256": sha256_bytes(raw)}
    phases, progress = _failure_phases(events, root, key)
    assert len(phases) == 5 and phases[3].signal == 9
    assert progress == events["progress.json"]["sha256"]
    for name in events:
        malformed = dict(events)
        malformed.pop(name)
        with pytest.raises(ValueError):
            _failure_phases(malformed, root, key)
    for mutation in (
        {"sequence": 2, "phase": "resume", "status": "failed"},
        {"sequence": 4, "phase": "resume", "status": "complete"},
        {"sequence": 4, "phase": "resume", "status": "failed", "unexpected": True},
    ):
        malformed = dict(events)
        raw = canonical_json_bytes(mutation) + b"\n"
        malformed["04-resume.json"] = {"raw_utf8": raw.decode(), "sha256": sha256_bytes(raw)}
        (root / "04-resume.json").write_bytes(raw)
        with pytest.raises(ValueError):
            _failure_phases(malformed, root, key)
    (root / "04-resume.json").write_text(events["04-resume.json"]["raw_utf8"])
    (root / "progress.json").write_bytes(b"forged")
    with pytest.raises(ValueError):
        _failure_phases(events, root, key)
    (root / "progress.json").write_text(events["progress.json"]["raw_utf8"])
    (root / "05-resume.json").write_bytes(b'{"phase":"resume","status":"complete"}')
    with pytest.raises(ValueError):
        _failure_phases(events, root, key)


@pytest.fixture
def continuation_case(task_scratch, monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace

    from silent_cascade.archive import _qualification as q
    from silent_cascade.archive import preflight
    from silent_cascade.archive.catalog import seal_unit
    from silent_cascade.archive.transport import (
        archive_operation_lock,
        initialize_remote_reservations,
    )
    from silent_cascade.archive.types import ArchivePolicy, EpisodeCommit, FileEntry

    policy = ArchivePolicy(chunk_bytes=64, page_bytes=4096)
    custody = task_scratch / "custody"
    custody.mkdir()
    revision = preflight._git_head(Path(__file__).parents[2])
    budget = preflight.bootstrap_engineering_workspace(
        custody_root=custody,
        workspace=custody / "operational",
        policy=policy,
        source_commit=revision,
    )
    control = budget.workspace / "control"
    for path, category in ((control, "metadata"), (control / "transfer-scratch", "scratch")):
        budget.bind(path, category=category)
    transport = DirectoryTransport(task_scratch / "remote")
    initialize_remote_reservations(
        control_dir=control,
        transport_id=transport.transport_id,
        accounted_bytes=0,
        accounting_evidence_sha256="a" * 64,
        policy=policy,
    )
    with archive_operation_lock(control):
        pass
    source = SimpleNamespace(
        source_commit="5633b60cec153435c51346d4a3a73b84fa1fb8d3",
        executable_sha256="e2d3f1372c8c390fbc1ec98aa0efa008443a67924e685f831fedf5770424322f",
    )
    anchor = budget._state()["engineering"]["authority_sha256"]
    record = preflight._ReviewedSourceRecord(
        anchor_authority_sha256=anchor,
        source_commit=source.source_commit,
        executable_sha256=source.executable_sha256,
        policy_sha256=sha256_bytes(canonical_json_bytes(policy)),
        review_sha256="b" * 64,
    )
    raw = canonical_json_bytes(record)
    authority = (
        budget.root
        / "source-authorities"
        / source.executable_sha256
        / f"{source.source_commit}.json"
    )
    authority.parent.mkdir(parents=True)
    authority.write_bytes(raw)
    source.source_authority_sha256 = sha256_bytes(raw)
    entry = FileEntry("debug/data.bin", sha256_bytes(b"debug"), 5)
    commit = EpisodeCommit(
        "phase4-evaluation-episode-commit-v1",
        "1" * 64,
        0,
        "debug",
        "2" * 64,
        0,
        1,
        "3" * 64,
        (entry,),
        (),
    )
    identity = q._qualification_identity(
        source=source,
        protocol_sha256="4" * 64,
        policy=policy,
        debug_commit_sha256=sha256_bytes(canonical_json_bytes(asdict(commit))),
        debug_review_sha256="5" * 64,
    )
    run_id = "qualification-" + sha256_bytes(canonical_json_bytes(identity))
    unit_identity = dict(
        run_id=run_id,
        source_commit=source.source_commit,
        config_sha256="4" * 64,
        evidence_identity_sha256=identity["debug_commit_sha256"],
        checkpoint_sha256=None,
        writer_stopped=True,
        checkpoint_committed=False,
    )
    old = budget.workspace / "failed"
    probe_root, debug_root = old / "source-probe", old / "source-debug"
    payload = bytes(range(160))
    for root, path, data in (
        (probe_root, "probe/probe.bin", payload),
        (debug_root, entry.path, b"debug"),
    ):
        target = root / path
        target.parent.mkdir(parents=True)
        target.write_bytes(data)
        budget.bind(root, category="spool")
    budget.bind(old / "events", category="logs")
    refs = []
    for root, logical, paths, kind, groups in (
        (probe_root, "probe", ("probe/probe.bin",), "diagnostic", ()),
        (debug_root, "debug", (entry.path,), "episode_pack", ((entry.path,),)),
    ):
        refs.append(
            seal_unit(
                run_dir=root,
                control_dir=control,
                logical_root=logical,
                paths=paths,
                kind=kind,
                identity=unit_identity,
                policy=policy,
                episode_groups=groups,
            )
        )
    monkeypatch.setattr(q, "_PROBE_BYTES", len(payload))
    monkeypatch.setattr(q, "_PROBE_SHA256", sha256_bytes(payload))
    q._event(old / "events", phase="admit", status="complete")
    q._event(old / "events", phase="seal", status="complete")
    phase_args = dict(
        run_dir=probe_root,
        control_dir=control,
        ref=refs[0],
        policy=policy,
        transport=transport,
        transport_calls=10000,
        observe_local=budget.check,
    )
    interrupted = q._spawn_archive_phase(**phase_args, phase="interrupt")
    assert not (control / f"receipts/{refs[0].unit_id}.json").exists()
    q._progress(old / "events", "resume")
    q._event(old / "events", phase="resume", status="failed")
    failure = dict(
        schema_version="phase4-controller-qualification-failure-v1",
        events={
            p.name: {"raw_utf8": p.read_text(), "sha256": sha256_bytes(p.read_bytes())}
            for p in (old / "events").iterdir()
        },
        result_absent=True,
        source_authority_sha256=source.source_authority_sha256,
        source_commit=source.source_commit,
        unit=asdict(refs[0]),
    )
    failure_path = budget.workspace / "failure.json"
    failure_path.write_bytes(canonical_json_bytes(failure))
    snapshot_args = dict(control_dir=control, transport=transport, policy=policy)
    before = q._remote_reservation_snapshot(**snapshot_args)
    recovery_events = budget.workspace / "diagnostic/events"
    budget.bind(recovery_events, category="logs")
    resumed = q._spawn_archive_phase(**phase_args, phase="resume", event_root=recovery_events)
    recovery = dict(
        schema_version="phase4-controller-resume-diagnostic-v1",
        status="original-unit-resume-completed-qualification-still-failed",
        before=before.model_dump(),
        after=q._remote_reservation_snapshot(**snapshot_args).model_dump(),
        bounds={"logs": 262144, "metadata": 88466468, "scratch": 75497472},
        provider_observations={
            k: v for k, v in resumed.items() if k not in {"status", "receipt_sha256"}
        },
        receipt_sha256=resumed["receipt_sha256"],
        seconds=1.0,
        source_authority_sha256=source.source_authority_sha256,
        source_commit=source.source_commit,
        unit_id=refs[0].unit_id,
    )
    recovery_path = budget.workspace / "recovery.json"
    recovery_path.write_bytes(canonical_json_bytes(recovery))
    chunk = next((control / "transfer-scratch").rglob("chunk-00000.bin"))
    assert interrupted["signal"] == 9

    def arguments(name):
        output = budget.workspace / name
        for path, category in (
            (output, "metadata"),
            (output / "events", "logs"),
            (output / "restore-probe", "cache"),
            (output / "restore-debug", "cache"),
        ):
            budget.bind(path, category=category)
        return dict(
            budget=budget,
            transport=transport,
            output_root=output,
            failure_proof_path=failure_path,
            failure_proof_sha256=sha256_bytes(failure_path.read_bytes()),
            recovery_proof_path=recovery_path,
            recovery_proof_sha256=sha256_bytes(recovery_path.read_bytes()),
            probe_source_root=probe_root,
            debug_source_root=debug_root,
            probe_ref=refs[0],
            debug_ref=refs[1],
            interrupted_chunk_path=chunk,
            debug_commit=commit,
            debug_review_sha256="5" * 64,
            protocol_sha256="4" * 64,
            original_source_commit=source.source_commit,
            original_executable_sha256=source.executable_sha256,
            original_source_authority_sha256=source.source_authority_sha256,
            source_commit=revision,
            source_authority_sha256=anchor,
        )

    return arguments, old, payload


def test_continuation_rejects_preconditions_before_provider_or_output(continuation_case):
    from dataclasses import replace
    from types import SimpleNamespace

    from silent_cascade.archive._qualification_continuation import _continue_qualification

    arguments, old, _ = continuation_case
    inputs = arguments("refused")
    additional = [
        {"original_source_commit": "9" * 40},
        {"source_commit": "9" * 40},
        {"probe_ref": inputs["debug_ref"]},
        {"debug_ref": inputs["probe_ref"]},
        {"debug_commit": replace(inputs["debug_commit"], row_sha256="9" * 64)},
        {"transport": SimpleNamespace(transport_id="foreign")},
    ]
    for changed in additional:
        with pytest.raises((ValueError, OSError)):
            _continue_qualification(**(inputs | changed))
        assert not inputs["output_root"].exists()
    for field in (
        "failure_proof_sha256",
        "recovery_proof_sha256",
        "original_executable_sha256",
        "original_source_authority_sha256",
        "source_authority_sha256",
        "protocol_sha256",
        "debug_review_sha256",
    ):
        transport = inputs["transport"]
        before = (transport.create_calls, transport.download_calls)
        with pytest.raises((ValueError, OSError)):
            _continue_qualification(**(inputs | {field: "9" * 64}))
        assert not inputs["output_root"].exists()
        assert before == (transport.create_calls, transport.download_calls)
    original = (old / "source-probe/probe/probe.bin").read_bytes()
    (old / "source-probe/probe/probe.bin").write_bytes(b"X" * len(original))
    with pytest.raises(ValueError):
        _continue_qualification(**inputs)
    assert not inputs["output_root"].exists()


def test_continuation_completes_exact_original_inputs_and_preserves_failed_tree(
    continuation_case, monkeypatch
):
    from silent_cascade.archive._qualification_continuation import _continue_qualification

    arguments, old, payload = continuation_case
    before = {p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    inputs = arguments("continued")
    scans = []
    retained_charge = inputs["budget"].retained_charge

    def authenticated():
        scans.append(bool(inputs["budget"]._state()["reservations"]))
        return retained_charge()

    monkeypatch.setattr(inputs["budget"], "retained_charge", authenticated)
    result = _continue_qualification(**inputs)
    output = inputs["output_root"]
    assert (output / "restore-probe/tree/probe/probe.bin").read_bytes() == payload
    assert (output / "restore-debug/tree/debug/data.bin").read_bytes() == b"debug"
    assert before == {p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()}
    assert result.inherited_interruption.signal == 9
    assert result.predecessor.resume_failure_cause == "unknown"
    assert result.collision_observation.conflicts == 1
    assert result.repeat_observation.downloads > 0
    assert result.local_bound.spool == 0
    for name in ("spool", "cache", "metadata", "scratch", "logs", "pinned", "emergency"):
        assert getattr(result.local_peak, name) - getattr(result.local_before, name) <= getattr(
            result.local_bound, name
        )
    assert result.cumulative_campaign_remote_bytes == result.remote_after.reserved_bytes
    assert (output / "qualification-continuation-result.json").is_file()
    assert not (output / "qualification-result.json").exists()
    assert scans == [False, True, True]


def test_continuation_remaining_bounds_charge_history_and_every_remaining_tree(tiny_archive_policy):
    from silent_cascade.archive._qualification import _RemoteReservationSnapshot
    from silent_cascade.archive._qualification_continuation import (
        _continuation_bounds,
        _require_campaign,
    )
    from silent_cascade.archive.ledger import StorageBlocked
    from silent_cascade.archive.types import FileEntry

    bound = _continuation_bounds(
        policy=tiny_archive_policy,
        run_id="qualification-test",
        probe=(FileEntry("probe/probe.bin", "a" * 64, 160),),
        debug=(FileEntry("debug/data.bin", "b" * 64, 5),),
        debug_logical_root="debug",
        block=4096,
        prior=20,
        reservations=2,
        completed=(),
        receipt_object_bytes=(64, 800),
    )
    assert bound.local.spool == 0
    assert bound.local.cache >= 2 * 4096
    assert bound.local.scratch >= 800
    for reserved, accounted in ((256 * 1024**2, 0), (512 * 1024**2, 512 * 1024**2)):
        snapshot = _RemoteReservationSnapshot(
            transport_id="test",
            accounted_bytes=accounted,
            reserved_bytes=reserved,
            operational_publication_bytes=0,
            reservation_pending=False,
            operational_pending=False,
        )
        with pytest.raises(StorageBlocked):
            _require_campaign(snapshot, bound.remote)
    snapshot = snapshot.model_copy(update={"accounted_bytes": 0, "reserved_bytes": 0})
    with pytest.raises(StorageBlocked):
        _require_campaign(snapshot, bound.remote, remote_limit=bound.remote - 1)


def test_continuation_admission_refuses_category_normal_global_and_physical_limits(
    continuation_case, monkeypatch
):
    from types import SimpleNamespace

    from silent_cascade.archive import _qualification_continuation as c
    from silent_cascade.archive import ledger
    from silent_cascade.archive.ledger import StorageBlocked

    arguments, _, _ = continuation_case
    inputs = arguments("capacity-refused")
    budget = inputs["budget"]
    measured = budget.measure()
    initial_calls = (inputs["transport"].create_calls, inputs["transport"].download_calls)
    for boundary in (*ledger._CATEGORIES, "normal-global", "physical"):
        with monkeypatch.context() as patch:
            allocation = dict(measured)
            if boundary in ledger._CATEGORIES:
                allocation[boundary] = getattr(budget.policy, boundary + "_bytes") + (
                    -1 if boundary in {"cache", "metadata", "scratch", "logs"} else 1
                )
            elif boundary == "normal-global":
                allocation = {
                    name: getattr(budget.policy, name + "_bytes") for name in ledger._CATEGORIES
                }
                patch.setattr(budget, "retained_charge", lambda: 1)
            else:
                actual_statvfs = ledger.os.statvfs

                def volume(path, actual_statvfs=actual_statvfs):
                    actual = actual_statvfs(path)
                    return SimpleNamespace(
                        f_bavail=0, f_frsize=actual.f_frsize, f_bsize=actual.f_bsize
                    )

                patch.setattr(ledger.os, "statvfs", volume)
            patch.setattr(budget, "measure", lambda allocation=allocation: allocation)
            with pytest.raises(StorageBlocked):
                c._continue_qualification(**inputs)
        assert not inputs["output_root"].exists()
        assert not budget._state()["reservations"]
        assert initial_calls == (
            inputs["transport"].create_calls,
            inputs["transport"].download_calls,
        )


def test_continuation_is_in_exact_gate_source_inventory():
    from silent_cascade.train.pilot_evidence import REQUIRED_PACKAGE_FILES

    package = Path(__file__).parents[2] / "src/silent_cascade"
    actual = {
        "src/silent_cascade/" + p.relative_to(package).as_posix() for p in package.rglob("*.py")
    }
    assert actual == set(REQUIRED_PACKAGE_FILES)


def test_continuation_rejects_missing_new_checks_and_publication_failures(
    continuation_case, monkeypatch
):
    from silent_cascade.archive import _qualification as q
    from silent_cascade.archive import _qualification_continuation as c
    from silent_cascade.archive.ledger import StorageBlocked

    arguments, old, _ = continuation_case
    old_bytes = {p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()}

    def exercise(fault):
        inputs = arguments("fault-" + fault)
        with monkeypatch.context() as patch:
            phase = q._spawn_archive_phase
            restore = q._restore_receipted_unit
            snapshot = q._remote_reservation_snapshot
            observed_create = q._ObservedTransport.create
            stop = q._stop_child
            full = inputs["budget"].check
            removed = []

            def observed(self, key, source):
                if fault == "collision":
                    # A broken provider create which silently accepts an existing key.
                    self.summary["creates"] += 1
                    return None
                value = observed_create(self, key, source)
                return value

            def run_phase(**kwargs):
                if fault == "missing" and kwargs["phase"] == "repeat":
                    key = next(inputs["transport"].root.rglob("*.bin"))
                    removed.append((key, key.read_bytes()))
                    key.unlink()
                result = phase(**kwargs)
                if kwargs["phase"] == "repeat":
                    if fault == "downloads":
                        result["downloads"] = 0
                    if fault == "receipt":
                        result["receipt_sha256"] = "9" * 64
                    if fault in {"proof", "chunk"}:
                        path = inputs[
                            "failure_proof_path" if fault == "proof" else "interrupted_chunk_path"
                        ]
                        removed.append((path, path.read_bytes()))
                        path.write_bytes(b"changed after authentication")
                return result

            def restore_unit(**kwargs):
                if fault == "overwrite" and kwargs["destination"] == inputs["probe_source_root"]:
                    return ()
                entries = restore(**kwargs)
                if fault == "inventory":
                    return ()
                if fault == "bytes":
                    (kwargs["destination"] / entries[0].path).write_bytes(b"changed")
                return entries

            snapshots = 0

            def remote_snapshot(**kwargs):
                nonlocal snapshots
                snapshots += 1
                value = snapshot(**kwargs)
                if fault == "reservation" and snapshots == 3:
                    return value.model_copy(update={"reserved_bytes": value.reserved_bytes + 1})
                return value

            def record(self, operation, key, size, outcome):
                value = original_record(self, operation, key, size, outcome)
                return "9" * 64 if fault == "key" else value

            def cleanup(child, **kwargs):
                stop(child, **kwargs)
                if fault == "cleanup":
                    raise StorageBlocked("injected cleanup failure after reaping")

            def authenticate():
                if (
                    fault == "authentication"
                    and (inputs["output_root"] / ".qualification-result.part").exists()
                ):
                    raise StorageBlocked("injected final authentication failure")
                return full()

            original_record = q._ObservedTransport.record
            patch.setattr(q._ObservedTransport, "create", observed)
            patch.setattr(q._ObservedTransport, "record", record)
            patch.setattr(q, "_spawn_archive_phase", run_phase)
            patch.setattr(q, "_restore_receipted_unit", restore_unit)
            patch.setattr(q, "_remote_reservation_snapshot", remote_snapshot)
            patch.setattr(q, "_stop_child", cleanup)
            patch.setattr(inputs["budget"], "check", authenticate)
            try:
                with pytest.raises((ValueError, StorageBlocked)):
                    c._continue_qualification(**inputs)
            finally:
                for path, data in removed:
                    path.write_bytes(data)
        assert not (inputs["output_root"] / "qualification-continuation-result.json").exists()
        assert old_bytes == {
            p.relative_to(old): p.read_bytes() for p in old.rglob("*") if p.is_file()
        }
        assert not inputs["budget"]._state()["reservations"]

    for fault in (
        "collision",
        "key",
        "downloads",
        "receipt",
        "reservation",
        "overwrite",
        "inventory",
        "bytes",
        "cleanup",
        "authentication",
        "missing",
        "proof",
        "chunk",
    ):
        exercise(fault)


def test_observed_transport_archive_checks_local_ledger(process_case):
    from silent_cascade.archive._qualification import _ObservedTransport

    budget, unit, transport = process_case
    observer = _ObservedTransport(
        transport, budget, "resume", unit.root.parent / "events", None, None
    )
    assert len(_archive(unit, observer)) == 64
    assert observer.summary["creates"] > 0


def _scoped_only_archive_child(boundary, *args):
    from conftest import _hide_host_authorities

    from silent_cascade.archive import _qualification
    from silent_cascade.archive.ledger import _StorageBudget

    def forbidden_full_check(self):
        raise AssertionError("qualification child attempted full retained authentication")

    _StorageBudget.retained_charge = forbidden_full_check
    with _hide_host_authorities(boundary):
        _qualification._archive_child(*args)


@pytest.mark.parametrize("expired", [False, True])
def test_qualification_child_rejects_noninheritable_or_expired_scope(process_case, expired):
    from silent_cascade.archive._qualification import _spawn_archive_phase
    from silent_cascade.archive.ledger import StorageBlocked

    budget, unit, transport = process_case

    def attempt(scope):
        with pytest.raises(StorageBlocked, match="child failed"):
            _spawn_archive_phase(
                run_dir=unit.root,
                control_dir=unit.control,
                ref=unit.ref,
                policy=unit.policy,
                transport=transport,
                phase="resume",
                transport_calls=10,
                observe_local=lambda: None,
                scope=scope,
            )
        assert not (unit.control / f"receipts/{unit.ref.unit_id}.json").exists()
        assert not tuple(transport.root.rglob("*.bin"))

    with budget._scoped_reservation(
        admission={}, child_inheritable=expired, metadata=1024**2, logs=1024**2, scratch=1024**2
    ) as admitted:
        if not expired:
            attempt(admitted.retained)
    if expired:
        attempt(admitted.retained)


def _abandon_scoped_parent(workspace, policy, reply):
    from silent_cascade.archive.ledger import _StorageBudget

    budget = _StorageBudget(workspace=workspace, policy=policy)
    with budget._scoped_reservation(
        admission={}, child_inheritable=True, metadata=1024**2, logs=1024**2, scratch=1024**2
    ) as admitted:
        reply.send(admitted.retained)
        reply.close()
        os._exit(0)


def test_qualification_child_rejects_dead_scope_owner(process_case, isolated_archive_authorities):
    import multiprocessing

    from silent_cascade.archive._qualification import _spawn_archive_phase
    from silent_cascade.archive.ledger import StorageBlocked

    budget, unit, transport = process_case
    context = multiprocessing.get_context("spawn")
    reply, child_reply = context.Pipe(duplex=False)
    parent = context.Process(
        target=isolated_archive_authorities.target(
            _abandon_scoped_parent, budget.workspace, unit.policy, child_reply
        )
    )
    parent.start()
    child_reply.close()
    assert reply.poll(10)
    scope = reply.recv()
    reply.close()
    parent.join(10)
    assert parent.exitcode == 0
    assert scope.owner_pid == parent.pid
    assert scope.reservation_token in budget._state()["reservations"]
    parent.close()
    with pytest.raises(StorageBlocked, match="child failed"):
        _spawn_archive_phase(
            run_dir=unit.root,
            control_dir=unit.control,
            ref=unit.ref,
            policy=unit.policy,
            transport=transport,
            phase="resume",
            transport_calls=10,
            observe_local=lambda: None,
            scope=scope,
        )
    assert not (unit.control / f"receipts/{unit.ref.unit_id}.json").exists()


@pytest.mark.parametrize("final_failure", [False, True])
def test_qualification_scans_source_precondition_then_owned_entry_and_staged_final_boundary(
    task_scratch, monkeypatch, isolated_archive_authorities, final_failure
):
    import functools

    from silent_cascade.archive import _qualification, preflight
    from silent_cascade.archive.ledger import StorageBlocked
    from silent_cascade.archive.transport import (
        archive_operation_lock,
        initialize_remote_reservations,
    )
    from silent_cascade.archive.types import ArchivePolicy, EpisodeCommit, FileEntry

    policy = ArchivePolicy(chunk_bytes=64, page_bytes=4096)
    probe = bytes(range(160))
    custody = task_scratch / "custody"
    debug_source = custody / "debug-source"
    debug_source.mkdir(parents=True)
    (debug_source / "probe.bin").write_bytes(probe)
    revision = preflight._git_head(Path(__file__).parents[2])
    budget = preflight.bootstrap_engineering_workspace(
        custody_root=custody,
        workspace=custody / "operational",
        policy=policy,
        source_commit=revision,
    )
    control = budget.workspace / "control"
    budget.bind(control, category="metadata")
    budget.bind(control / "transfer-scratch", category="scratch")
    transport = DirectoryTransport(task_scratch / "remote")
    initialize_remote_reservations(
        control_dir=control,
        transport_id=transport.transport_id,
        accounted_bytes=11,
        accounting_evidence_sha256="a" * 64,
        policy=policy,
    )
    output = budget.workspace / "scoped-qualification"

    with archive_operation_lock(control):
        pass
    for root, category in {
        output: "metadata",
        output / "source-probe": "spool",
        output / "source-debug": "spool",
        output / "restore-probe": "cache",
        output / "restore-debug": "cache",
        output / "events": "logs",
    }.items():
        budget.bind(root, category=category)
    monkeypatch.setattr(_qualification, "_PROBE_BYTES", len(probe))
    monkeypatch.setattr(_qualification, "_PROBE_SHA256", sha256_bytes(probe))
    monkeypatch.setattr(_qualification, "_probe_blocks", lambda: iter((probe,)))
    monkeypatch.setattr(
        _qualification,
        "_archive_child",
        functools.partial(_scoped_only_archive_child, isolated_archive_authorities.boundary),
    )
    entry = FileEntry("debug/probe.bin", sha256_bytes(probe), len(probe))
    commit = EpisodeCommit(
        "phase4-evaluation-episode-commit-v1",
        "1" * 64,
        0,
        "debug",
        "2" * 64,
        0,
        1,
        "3" * 64,
        (entry,),
        (),
    )
    scans = []
    authenticate = budget.retained_charge

    def full_check():
        scans.append(bool(budget._state()["reservations"]))
        if (output / ".qualification-result.part").exists():
            assert budget._state()["reservations"]
            assert (output / ".qualification-result.part").exists()
            assert not (output / "qualification-result.json").exists()
            if final_failure:
                raise StorageBlocked("injected retained mutation")
        else:
            assert not output.exists()
        return authenticate()

    monkeypatch.setattr(budget, "retained_charge", full_check)
    arguments = dict(
        budget=budget,
        output_root=output,
        debug_source_root=debug_source,
        debug_logical_root="debug",
        debug_commit=commit,
        debug_review_sha256="4" * 64,
        protocol_sha256="5" * 64,
        source_commit=revision,
        transport=transport,
    )
    if final_failure:
        with pytest.raises(StorageBlocked, match="retained mutation"):
            _qualification._run_qualification(**arguments)
        assert not (output / "qualification-result.json").exists()
        assert (output / ".qualification-result.part").exists()
    else:
        result = _qualification._run_qualification(**arguments)
        assert result.interruption.signal == 9
        assert result.retry.exact_readback and result.retry.stable_reserved_bytes
        assert result.local_after.model_dump(exclude={"retained_bytes"}) == budget.measure()
        assert (output / "qualification-result.json").exists()
    assert len(scans) == 3
    assert sum(scans) == 2
    assert scans == [False, True, True]
    assert not budget._state()["reservations"]


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


def test_qualification_changed_source_requires_explicit_authority_before_output(
    process_case, monkeypatch
):
    from silent_cascade.archive import _qualification
    from silent_cascade.archive.transport import _lock
    from silent_cascade.archive.types import EpisodeCommit, FileEntry

    budget, unit, transport = process_case
    output = budget.workspace / "reviewed-qualification"
    for root, category in {
        output: "metadata",
        output / "source-probe": "spool",
        output / "source-debug": "spool",
        output / "restore-probe": "cache",
        output / "restore-debug": "cache",
        output / "events": "logs",
    }.items():
        budget.bind(root, category=category)
    with _lock(
        control_dir=budget.root, relative=("engineering.lock",), shared=False, blocking=True
    ):
        pass
    entry = FileEntry("probe/probe.bin", sha256_bytes(bytes(range(160))), 160)
    commit = EpisodeCommit(
        "phase4-evaluation-episode-commit-v1",
        "1" * 64,
        0,
        "debug",
        "2" * 64,
        0,
        1,
        "3" * 64,
        (entry,),
        (),
    )
    observed = {}

    def refused(**kwargs):
        observed.update(kwargs)
        raise ValueError("explicit reviewed source required")

    monkeypatch.setattr(_qualification, "_reviewed_source", refused, raising=False)
    with pytest.raises(ValueError, match="explicit reviewed source"):
        _qualification._run_qualification(
            budget=budget,
            output_root=output,
            debug_source_root=unit.root,
            debug_logical_root="probe",
            debug_commit=commit,
            debug_review_sha256="4" * 64,
            protocol_sha256="5" * 64,
            source_commit="6" * 40,
            source_authority_sha256="7" * 64,
            transport=transport,
        )
    assert observed == {
        "budget": budget,
        "source_commit": "6" * 40,
        "authority_sha256": "7" * 64,
    }
    assert not output.exists()


def test_qualification_identity_changes_with_reviewed_source_authority(tiny_archive_policy):
    from silent_cascade.archive._qualification import _qualification_identity
    from silent_cascade.archive.preflight import _ReviewedSource

    arguments = dict(
        protocol_sha256="1" * 64,
        policy=tiny_archive_policy,
        debug_commit_sha256="2" * 64,
        debug_review_sha256="3" * 64,
    )
    first = _qualification_identity(
        source=_ReviewedSource("4" * 40, "5" * 64, "6" * 64), **arguments
    )
    second = _qualification_identity(
        source=_ReviewedSource("4" * 40, "5" * 64, "7" * 64), **arguments
    )
    assert first == {
        "schema_version": "phase4-r2-early-qualification-input-v2",
        "source_commit": "4" * 40,
        "executable_sha256": "5" * 64,
        "source_authority_sha256": "6" * 64,
        "protocol_sha256": "1" * 64,
        "policy_sha256": sha256_bytes(canonical_json_bytes(tiny_archive_policy)),
        "debug_commit_sha256": "2" * 64,
        "debug_review_sha256": "3" * 64,
    }
    assert sha256_bytes(canonical_json_bytes(first)) != sha256_bytes(canonical_json_bytes(second))


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


def _unreapable_parent(workspace, policy, fault):
    from silent_cascade.archive import _qualification
    from silent_cascade.archive.ledger import _StorageBudget

    class Unreaped:
        pid = os.getpid()

        def kill(self):
            if fault == "kill":
                raise PermissionError("injected cleanup failure")

        def join(self, _timeout):
            if fault == "join":
                raise OSError("injected cleanup failure")
            if fault == "interrupt":
                raise KeyboardInterrupt

        def is_alive(self):
            return fault != "group-probe"

    def group_signal(_pid, signum):
        # Never send a real signal to the isolated test parent's own group.
        if fault == "group-kill" or (fault == "group-probe" and signum == 0):
            raise PermissionError("injected cleanup failure")

    def failed_evidence(*_args, **_kwargs):
        raise KeyboardInterrupt

    owns_group = fault in {"group-kill", "group-probe"}
    if owns_group:
        _qualification.os.killpg = group_signal
    if fault == "group-probe":
        ticks = iter((0, 61))
        _qualification.time.monotonic = lambda: next(ticks)
    if fault == "evidence":
        _qualification._event = failed_evidence

    budget = _StorageBudget(workspace=workspace, policy=policy)
    try:
        with budget.reserve(metadata=1024**2, logs=1024**2):
            _qualification._stop_child(
                Unreaped(),
                owns_group=owns_group,
                events=workspace / "qualification/events",
                phase="resume",
            )
    except BaseException:
        # Distinguish unsafe context unwinding without printing injected traces.
        os._exit(71)


@pytest.mark.parametrize(
    "fault", ["alive", "kill", "join", "group-kill", "group-probe", "interrupt", "evidence"]
)
def test_unreapable_child_fail_stop_preserves_durable_reservation(
    process_case, fault, isolated_archive_authorities
):
    import multiprocessing

    budget, unit, _transport = process_case
    parent = multiprocessing.get_context("spawn").Process(
        target=isolated_archive_authorities.target(
            _unreapable_parent, budget.workspace, unit.policy, fault
        )
    )
    parent.start()
    parent.join(10)
    assert parent.exitcode == 70
    reservations = budget._state()["reservations"]
    assert len(reservations) == 1
    assert next(iter(reservations.values()))["pid"] == parent.pid
    assert not (unit.root.parent / "qualification-result.json").exists()
    parent.close()


@pytest.mark.parametrize("stage", ["kill", "probe"])
def test_qualification_waits_for_esrch_after_transient_group_eperm(tmp_path, monkeypatch, stage):
    import signal
    from types import SimpleNamespace

    from silent_cascade.archive import _qualification

    probes = []

    def signal_group(pid, signum):
        assert pid == 123456
        if signum == 0:
            probes.append(signum)
            if len(probes) == 1:
                raise PermissionError("transient empty group")
            raise ProcessLookupError
        assert signum == signal.SIGKILL
        if stage == "kill":
            raise PermissionError("transient empty group")

    def fail_stop(*_args):
        pytest.fail("transient group EPERM prevented bounded ESRCH proof")

    monkeypatch.setattr(_qualification.os, "killpg", signal_group)
    monkeypatch.setattr(_qualification, "_fail_stop", fail_stop)
    monkeypatch.setattr(_qualification.time, "sleep", lambda _: None)
    child = SimpleNamespace(pid=123456, join=lambda _: None, is_alive=lambda: False)
    _qualification._stop_child(child, owns_group=True, events=tmp_path, phase="resume")
    assert len(probes) == 2
