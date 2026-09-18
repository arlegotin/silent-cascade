"""Offline file-only requests, authenticated leases, and authoritative inventories."""

import json
import os
import threading
import time

import psutil
import pytest


def channel_fixture(tmp_path, policy):
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

    control = tmp_path / "control"
    control.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    state = {
        "schema_version": "phase4-r2-session-v1",
        "run_id": "fixture-run",
        "session_id": "a" * 64,
        "run_dir": str(run),
        "policy_sha256": sha256_bytes(canonical_json_bytes(policy)),
        "pid": os.getpid(),
        "create_time": psutil.Process().create_time(),
    }
    (control / "session.json").write_bytes(canonical_json_bytes(state))
    return run, control, state


@pytest.mark.parametrize("stale", [False, True])
def test_local_request_rejects_another_runs_response(tmp_path, stale):
    from silent_cascade.archive.session import LocalArchiveSession
    from silent_cascade.archive.types import ArchivePolicy
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

    policy = ArchivePolicy()
    run, control, _ = channel_fixture(tmp_path, policy)
    session = LocalArchiveSession(run_dir=run, control_dir=control, policy=policy)

    def parent():
        deadline = time.monotonic() + 5
        while not (control / "request.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        raw = (control / "request.json").read_bytes()
        request = json.loads(raw)
        response = {
            "request_sha256": sha256_bytes(raw),
            "run_id": "another-run" if stale else request["run_id"],
            "session_id": request["session_id"],
            "sequence": request["sequence"],
            "operation": request["operation"],
            "policy_sha256": request["policy_sha256"],
            "unit_id": request["unit_id"],
            "status": "ok",
            "payload": {"healthy": True},
        }
        temporary = control / "response.tmp"
        temporary.write_bytes(canonical_json_bytes(response))
        temporary.rename(control / "response.json")

    thread = threading.Thread(target=parent)
    thread.start()
    try:
        if stale:
            with pytest.raises(ValueError, match=r"response.*identity"):
                session._request("status", {})
        else:
            assert session._request("status", {}) == {"healthy": True}
    finally:
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_parent_identity_mismatch_stops_before_publishing_request(tmp_path):
    from silent_cascade.archive.session import LocalArchiveSession
    from silent_cascade.archive.supervisor import StorageBlocked
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    run, control, state = channel_fixture(tmp_path, policy)
    state["create_time"] -= 100
    (control / "session.json").write_text(json.dumps(state))
    session = LocalArchiveSession(run_dir=run, control_dir=control, policy=policy)
    with pytest.raises(StorageBlocked, match="parent"):
        session._request("status", {})
    assert not (control / "request.json").exists()


def test_episode_commit_rejects_unordered_owned_and_boolean_ordinal():
    from silent_cascade.archive.types import EpisodeCommit, FileEntry

    values = dict(
        schema_version="phase4-evaluation-episode-commit-v1",
        identity_sha256="a" * 64,
        ordinal=0,
        episode_public_id="fixture-episode",
        episode_sha256="b" * 64,
        row_offset=0,
        row_bytes=1,
        row_sha256="c" * 64,
        owned=(FileEntry("eval/b.json", "d" * 64, 1), FileEntry("eval/a.json", "e" * 64, 1)),
        borrowed=(),
    )
    with pytest.raises(ValueError, match="order"):
        EpisodeCommit(**values)
    values["owned"] = tuple(reversed(values["owned"]))
    values["ordinal"] = True
    with pytest.raises(ValueError, match="ordinal"):
        EpisodeCommit(**values)


def test_episode_binding_survives_cold_catalog_and_rejects_wrong_owned_hash(tmp_path):
    import shutil
    from dataclasses import asdict, replace

    from conftest import DirectoryTransport

    from silent_cascade.archive.catalog import iter_unit_files, seal_unit
    from silent_cascade.archive.session import lookup_episode_binding, register_episode_binding
    from silent_cascade.archive.transport import initialize_remote_reservations
    from silent_cascade.archive.types import ArchivePolicy, EpisodeCommit
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

    policy = ArchivePolicy()
    run, control, _ = channel_fixture(tmp_path, policy)
    path = run / "evaluation" / "arbitrary.bin"
    path.parent.mkdir()
    path.write_bytes(b"episode evidence")
    transport = DirectoryTransport(tmp_path / "remote")
    initialize_remote_reservations(
        control_dir=control,
        transport_id=transport.transport_id,
        policy=policy,
        accounted_bytes=0,
        accounting_evidence_sha256="f" * 64,
    )
    ref = seal_unit(
        run_dir=run,
        control_dir=control,
        logical_root="evaluation",
        paths=("evaluation/arbitrary.bin",),
        kind="episode_pack",
        identity=dict(
            run_id="fixture-run",
            source_commit="0" * 40,
            config_sha256="1" * 64,
            evidence_identity_sha256="2" * 64,
            checkpoint_sha256=None,
            writer_stopped=True,
            checkpoint_committed=False,
        ),
        episode_groups=(("evaluation/arbitrary.bin",),),
        policy=policy,
    )
    commit = EpisodeCommit(
        "phase4-evaluation-episode-commit-v1",
        "2" * 64,
        17,
        "episode-17",
        "3" * 64,
        0,
        1,
        "4" * 64,
        tuple(iter_unit_files(control, ref)),
        (),
    )
    register_episode_binding(
        run_dir=run,
        control_dir=control,
        logical_root="evaluation",
        unit_ref=ref,
        commit=commit,
        transport=transport,
        policy=policy,
    )
    shutil.rmtree(control / "episode-bindings" / "catalog")
    result = lookup_episode_binding(
        control_dir=control,
        run_id="fixture-run",
        logical_root="evaluation",
        ordinal=17,
        commit_sha256=sha256_bytes(canonical_json_bytes(asdict(commit))),
        transport=transport,
        policy=policy,
    )
    assert result.commit == commit
    assert result.unit_ref == ref
    bad = replace(commit, owned=(replace(commit.owned[0], sha256="b" * 64),))
    with pytest.raises(ValueError, match="ownership"):
        register_episode_binding(
            run_dir=run,
            control_dir=control,
            logical_root="evaluation",
            unit_ref=ref,
            commit=bad,
            transport=transport,
            policy=policy,
        )


@pytest.fixture
def running_archive(tmp_path):
    from conftest import DirectoryTransport

    from silent_cascade.archive.catalog import seal_unit
    from silent_cascade.archive.ledger import initialize_workspace_ledger
    from silent_cascade.archive.supervisor import _ArchiveServer
    from silent_cascade.archive.transport import initialize_remote_reservations
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    run = tmp_path / "spool" / "run"
    control = tmp_path / "metadata" / "control"
    run.mkdir(parents=True)
    control.mkdir(parents=True)
    path = run / "evaluation" / "arbitrary.bin"
    path.parent.mkdir()
    path.write_bytes(b"episode evidence")
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    transport = DirectoryTransport(tmp_path / "scratch" / "remote")
    initialize_remote_reservations(
        control_dir=control,
        transport_id=transport.transport_id,
        policy=policy,
        accounted_bytes=0,
        accounting_evidence_sha256="f" * 64,
    )
    ref = seal_unit(
        run_dir=run,
        control_dir=control,
        logical_root="evaluation",
        paths=("evaluation/arbitrary.bin",),
        kind="episode_pack",
        identity=dict(
            run_id="fixture-run",
            source_commit="0" * 40,
            config_sha256="1" * 64,
            evidence_identity_sha256="2" * 64,
            checkpoint_sha256=None,
            writer_stopped=True,
            checkpoint_committed=False,
        ),
        episode_groups=(("evaluation/arbitrary.bin",),),
        policy=policy,
    )
    server = _ArchiveServer(
        run_dir=run,
        control_dir=control,
        workspace_root=tmp_path,
        transport=transport,
        policy=policy,
        run_id="fixture-run",
    )
    stopped = threading.Event()
    failures = []

    def serve():
        try:
            while not stopped.is_set():
                server.serve_once()
                stopped.wait(0.005)
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        yield server, ref
    finally:
        stopped.set()
        thread.join(timeout=5)
        server.close()
        assert not thread.is_alive()
        assert not failures


def test_archive_request_restores_one_authenticated_episode_and_pins_reader(running_archive):
    from dataclasses import asdict

    from silent_cascade.archive.session import LocalArchiveSession
    from silent_cascade.archive.transport import _unit_writer_lock

    server, ref = running_archive
    session = LocalArchiveSession(
        run_dir=server.run_dir, control_dir=server.control_dir, policy=server.policy
    )
    session._request("archive", {"ref": asdict(ref), "evict": True}, unit_id=ref.unit_id)
    assert not (server.run_dir / "evaluation/arbitrary.bin").exists()
    with session.lease(ref) as lease:
        assert (lease.local_root / "evaluation/arbitrary.bin").read_bytes() == b"episode evidence"
        with (
            pytest.raises(ValueError, match="lease"),
            _unit_writer_lock(control_dir=server.control_dir, ref=ref),
        ):
            pytest.fail("live evidence reader lost its pin")
        from silent_cascade.archive.ledger import StorageBlocked

        with pytest.raises(StorageBlocked), session.lease(ref):
            pytest.fail("second episode lease admitted")
    assert not lease.local_root.exists()


def test_sparse_lease_metadata_lifetime_and_corrupted_tail_are_checked(running_archive):
    import json

    from silent_cascade.archive.catalog import iter_unit_files
    from silent_cascade.archive.session import LocalArchiveSession

    server, ref = running_archive
    session = LocalArchiveSession(
        run_dir=server.run_dir, control_dir=server.control_dir, policy=server.policy
    )
    with session._leased({"path": "evaluation/arbitrary.bin"}) as lease:
        assert lease.ref == ref
        entries = tuple(iter_unit_files(lease.metadata_root, ref))
        assert [entry.path for entry in entries] == ["evaluation/arbitrary.bin"]
        manifest = json.loads((lease.metadata_root / ref.manifest_path).read_bytes())
        tail = (
            lease.metadata_root / "units" / ref.unit_id / manifest["inventory_shards"][-1]["path"]
        )
        tail.write_bytes(b"corrupt inventory tail after lease acknowledgement")
        with pytest.raises(ValueError):
            tuple(iter_unit_files(lease.metadata_root, ref))
    assert not lease.metadata_root.exists()
    assert (server.run_dir / "evaluation/arbitrary.bin").read_bytes() == b"episode evidence"
    assert not server.leases


def test_child_rechecks_files_after_success_acknowledgement(running_archive, monkeypatch):
    from silent_cascade.archive.session import LocalArchiveSession

    server, ref = running_archive
    session = LocalArchiveSession(
        run_dir=server.run_dir, control_dir=server.control_dir, policy=server.policy
    )
    original = session._request

    def replace_after_ack(operation, payload, **kwargs):
        response = original(operation, payload, **kwargs)
        if operation == "lease":
            (server.run_dir / "evaluation/arbitrary.bin").unlink()
        return response

    monkeypatch.setattr(session, "_request", replace_after_ack)
    with pytest.raises((ValueError, FileNotFoundError)), session.lease(ref):
        pytest.fail("acknowledgment exposed incomplete evidence")


def test_inventory_merges_cold_files_and_checks_external_authority(running_archive):
    from dataclasses import asdict, replace

    from silent_cascade.archive.session import LocalArchiveSession

    server, ref = running_archive
    session = LocalArchiveSession(
        run_dir=server.run_dir, control_dir=server.control_dir, policy=server.policy
    )
    original = tuple(session.entries())
    assert [entry.path for entry in original] == ["evaluation/arbitrary.bin"]
    session._request("archive", {"ref": asdict(ref), "evict": True}, unit_id=ref.unit_id)
    assert tuple(session.entries()) == original
    assert (
        session.read_record(original[0].path, expected_sha256=original[0].sha256, max_bytes=16)
        == b"episode evidence"
    )
    session.verify_inventory(iter(original))
    with pytest.raises(ValueError, match=r"inventory|hash"):
        session.verify_inventory((replace(original[0], sha256="f" * 64),))
    with pytest.raises(ValueError, match="inventory"):
        session.verify_inventory(())


def test_episode_lookup_binds_nonfilename_ordinal_and_commit(running_archive):
    from dataclasses import asdict

    from silent_cascade.archive.catalog import iter_unit_files
    from silent_cascade.archive.session import LocalArchiveSession, register_episode_binding
    from silent_cascade.archive.types import EpisodeCommit
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

    server, ref = running_archive
    commit = EpisodeCommit(
        "phase4-evaluation-episode-commit-v1",
        "2" * 64,
        17,
        "episode-17",
        "3" * 64,
        0,
        1,
        "4" * 64,
        tuple(iter_unit_files(server.control_dir, ref)),
        (),
    )
    binding = register_episode_binding(
        run_dir=server.run_dir,
        control_dir=server.control_dir,
        logical_root="evaluation",
        unit_ref=ref,
        commit=commit,
        transport=server.transport,
        policy=server.policy,
    )
    session = LocalArchiveSession(
        run_dir=server.run_dir, control_dir=server.control_dir, policy=server.policy
    )
    with session.episode(
        "evaluation", 17, commit_sha256=sha256_bytes(canonical_json_bytes(asdict(commit)))
    ) as lease:
        assert lease.ref == binding.unit_ref
        assert (lease.local_root / commit.owned[0].path).read_bytes() == b"episode evidence"
    from silent_cascade.archive.ledger import StorageBlocked

    with (
        pytest.raises(StorageBlocked),
        session.episode("evaluation", 17, commit_sha256="f" * 64),
    ):
        pytest.fail("wrong commit accepted")


def test_real_child_denies_network_while_parent_services_archive(running_archive):
    import subprocess
    import sys
    from dataclasses import asdict

    from silent_cascade.train.pilot_offline import offline_environment

    server, ref = running_archive
    code = r"""
import json, socket, subprocess, sys
from pathlib import Path
from silent_cascade.train.pilot_offline import install_offline_boundary
install_offline_boundary()
from silent_cascade.archive.session import LocalArchiveSession
from silent_cascade.archive.types import ArchivePolicy, UnitRef
run, control, policy, reference = map(json.loads, sys.argv[1:])
ref = UnitRef(**reference)
session = LocalArchiveSession(
    run_dir=Path(run), control_dir=Path(control), policy=ArchivePolicy(**policy))
for operation in (lambda: socket.getaddrinfo('example.invalid', 443),
                  lambda: subprocess.run(['aws', '--version'])):
    try:
        operation()
    except RuntimeError:
        pass
    else:
        raise AssertionError('offline escape')
session._request('archive', {'ref': reference, 'evict': True}, unit_id=ref.unit_id)
with session.lease(ref) as lease:
    assert (lease.local_root / 'evaluation/arbitrary.bin').read_bytes() == b'episode evidence'
print('verified offline child')
"""
    scratch = server.budget.workspace / "scratch" / "child"
    scratch.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            code,
            json.dumps(str(server.run_dir)),
            json.dumps(str(server.control_dir)),
            server.policy.model_dump_json(),
            json.dumps(asdict(ref)),
        ],
        env=offline_environment(scratch),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "verified offline child"
    assert server.transport.create_calls > 0


def test_seal_operation_publishes_typed_episode_binding(running_archive):
    from dataclasses import asdict

    from silent_cascade.archive.catalog import iter_unit_files
    from silent_cascade.archive.session import LocalArchiveSession
    from silent_cascade.archive.types import EpisodeCommit

    server, ref = running_archive
    commit = EpisodeCommit(
        "phase4-evaluation-episode-commit-v1",
        "2" * 64,
        8,
        "episode-8",
        "3" * 64,
        0,
        1,
        "4" * 64,
        tuple(iter_unit_files(server.control_dir, ref)),
        (),
    )
    session = LocalArchiveSession(
        run_dir=server.run_dir, control_dir=server.control_dir, policy=server.policy
    )
    response = session._request(
        "seal",
        {"logical_root": "evaluation", "unit_ref": asdict(ref), "commit": asdict(commit)},
        unit_id=ref.unit_id,
    )
    with session.episode("evaluation", 8, commit_sha256=response["commit_sha256"]) as lease:
        assert lease.ref == ref


def test_cold_episode_restores_one_group_after_local_unit_metadata_eviction(running_archive):
    import shutil
    from dataclasses import asdict

    from silent_cascade.archive.catalog import iter_unit_files, seal_unit
    from silent_cascade.archive.ledger import StorageBlocked
    from silent_cascade.archive.session import LocalArchiveSession, register_episode_binding
    from silent_cascade.archive.types import EpisodeCommit

    server, _ = running_archive
    for path in ("more/a", "more/b"):
        (server.run_dir / path).parent.mkdir(exist_ok=True)
        (server.run_dir / path).write_bytes(path.encode())
    ref = seal_unit(
        run_dir=server.run_dir,
        control_dir=server.control_dir,
        logical_root="more",
        paths=("more/a", "more/b"),
        kind="episode_pack",
        policy=server.policy,
        identity=dict(
            run_id="fixture-run",
            source_commit="0" * 40,
            config_sha256="1" * 64,
            evidence_identity_sha256="2" * 64,
            checkpoint_sha256=None,
            writer_stopped=True,
            checkpoint_committed=False,
        ),
        episode_groups=(("more/a",), ("more/b",)),
    )
    commit = EpisodeCommit(
        "phase4-evaluation-episode-commit-v1",
        "2" * 64,
        123,
        "episode-a",
        "3" * 64,
        0,
        1,
        "4" * 64,
        (next(iter_unit_files(server.control_dir, ref)),),
        (),
    )
    binding = register_episode_binding(
        run_dir=server.run_dir,
        control_dir=server.control_dir,
        logical_root="more",
        unit_ref=ref,
        commit=commit,
        transport=server.transport,
        policy=server.policy,
    )
    session = LocalArchiveSession(
        run_dir=server.run_dir, control_dir=server.control_dir, policy=server.policy
    )
    session._request("archive", {"ref": asdict(ref), "evict": True}, unit_id=ref.unit_id)
    shutil.rmtree(server.control_dir / "units" / ref.unit_id)
    with pytest.raises(StorageBlocked), session.lease(ref):
        pytest.fail("whole evaluation materialized")
    with session.episode("more", 123, commit_sha256=binding.commit_sha256) as lease:
        assert (lease.local_root / "more/a").read_bytes() == b"more/a"
        assert not (lease.local_root / "more/b").exists()


def test_per_file_owner_lookup_does_not_scan_unrelated_units(running_archive, monkeypatch):
    from dataclasses import asdict

    from silent_cascade.archive.session import LocalArchiveSession

    server, ref = running_archive
    session = LocalArchiveSession(
        run_dir=server.run_dir, control_dir=server.control_dir, policy=server.policy
    )
    session._request("archive", {"ref": asdict(ref), "evict": True}, unit_id=ref.unit_id)

    def forbidden_scan():
        raise AssertionError("per-file lookup scanned unrelated catalog units")

    monkeypatch.setattr(server, "_units", forbidden_scan)
    owner, selected = server._find_owner("evaluation/arbitrary.bin")
    assert owner == ref
    assert selected == ("evaluation/arbitrary.bin",)


def test_real_parent_death_retains_pending_request(tmp_path):
    import subprocess
    import sys

    from silent_cascade.archive.types import ArchivePolicy
    from silent_cascade.train.pilot_offline import offline_environment

    run, control, state = channel_fixture(tmp_path, ArchivePolicy())
    parent = subprocess.Popen(
        [sys.executable, "-B", "-c", "import time; time.sleep(30)"],
        env=offline_environment(tmp_path),
    )
    child = None
    try:
        state["pid"] = parent.pid
        state["create_time"] = psutil.Process(parent.pid).create_time()
        (control / "session.json").write_text(json.dumps(state))
        program = r"""
import sys
from pathlib import Path
from silent_cascade.archive.session import LocalArchiveSession
from silent_cascade.archive.ledger import StorageBlocked
from silent_cascade.archive.types import ArchivePolicy
session = LocalArchiveSession(
    run_dir=Path(sys.argv[1]), control_dir=Path(sys.argv[2]), policy=ArchivePolicy())
try:
    session._request('status', {})
except StorageBlocked:
    print('storage_blocked')
else:
    raise AssertionError('dead parent acknowledged request')
"""
        child = subprocess.Popen(
            [sys.executable, "-B", "-c", program, str(run), str(control)],
            env=offline_environment(tmp_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 5
        while not (control / "request.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        parent.terminate()
        parent.wait(timeout=5)
        stdout, stderr = child.communicate(timeout=5)
        assert child.returncode == 0, stderr.decode()
        assert stdout.strip() == b"storage_blocked"
        assert (control / "request.json").exists()
    finally:
        for process in (child, parent):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)


def test_interrupted_staging_is_quarantined_without_removing_bytes(running_archive):
    server, _ = running_archive
    stage = server.control_dir / "lease-cache/.interrupted.restore"
    stage.mkdir(parents=True)
    (stage / "pending.bin").write_bytes(b"retain after parent crash")
    server._quarantine_staging()
    assert not stage.exists()
    retained = tuple((server.control_dir / "quarantine").rglob("pending.bin"))
    assert len(retained) == 1
    assert retained[0].read_bytes() == b"retain after parent crash"


@pytest.mark.parametrize("owner_state", ["present", "absent", "corrupt"])
def test_borrowed_dependency_is_authenticated_and_blocks_its_eviction(running_archive, owner_state):
    from dataclasses import asdict

    from silent_cascade.archive.catalog import _load_root, iter_unit_files, seal_unit
    from silent_cascade.archive.ledger import StorageBlocked
    from silent_cascade.archive.session import LocalArchiveSession
    from silent_cascade.archive.transport import _cold_reader, _load_head, _object_key
    from silent_cascade.archive.types import FileEntry
    from silent_cascade.hashing import sha256_bytes

    server, _ = running_archive
    identity = dict(
        run_id="fixture-run",
        source_commit="0" * 40,
        config_sha256="1" * 64,
        evidence_identity_sha256="2" * 64,
        checkpoint_sha256=None,
        writer_stopped=True,
        checkpoint_committed=False,
    )
    (server.run_dir / "shared").mkdir()
    (server.run_dir / "shared/weights").write_bytes(b"exact shared weights")
    borrowed = (FileEntry("shared/weights", sha256_bytes(b"exact shared weights"), 20),)
    shared = None
    if owner_state != "absent":
        shared = seal_unit(
            run_dir=server.run_dir,
            control_dir=server.control_dir,
            logical_root="shared",
            paths=("shared/weights",),
            kind="diagnostic",
            identity=identity,
            policy=server.policy,
        )
        borrowed = tuple(iter_unit_files(server.control_dir, shared))
    (server.run_dir / "borrower").mkdir()
    (server.run_dir / "borrower/owned").write_bytes(b"episode")
    ref = seal_unit(
        run_dir=server.run_dir,
        control_dir=server.control_dir,
        logical_root="borrower",
        paths=("borrower/owned",),
        kind="episode_pack",
        identity=identity,
        policy=server.policy,
        episode_groups=(("borrower/owned",),),
        borrowed=borrowed,
    )
    session = LocalArchiveSession(
        run_dir=server.run_dir, control_dir=server.control_dir, policy=server.policy
    )
    if owner_state == "corrupt":
        session._request("archive", {"ref": asdict(shared), "evict": False}, unit_id=shared.unit_id)
        cold = _cold_reader(
            control_dir=server.control_dir, transport=server.io, run_id=server.run_id
        )
        root = _load_root(
            _load_head(server.control_dir, server.run_id),
            scope="run",
            reader=cold,
            policy=server.policy,
        )
        # Corrupt only ownership authority; borrower metadata and source bytes remain valid.
        local = server.control_dir / root.ownership.path
        remote = server.transport.root / _object_key(server.run_id, root.ownership.path)
        remote.write_bytes(b"corrupt authenticated owner page")
        if local.exists():
            local.write_bytes(b"corrupt authenticated owner page")
        with pytest.raises(StorageBlocked), session.lease(ref):
            pytest.fail("corrupt ownership authority was treated as an absent owner")
        return
    with session.lease(ref) as lease:
        assert (lease.local_root / "shared/weights").read_bytes() == b"exact shared weights"
        if shared is not None:
            with pytest.raises(StorageBlocked):
                session._request(
                    "archive", {"ref": asdict(shared), "evict": True}, unit_id=shared.unit_id
                )
    (server.run_dir / "shared/weights").write_bytes(b"corrupt")
    with pytest.raises(StorageBlocked), session.lease(ref):
        pytest.fail("corrupt borrowed dependency accepted")


def test_long_request_uses_each_transport_operation_deadline(tmp_path, monkeypatch):
    from silent_cascade.archive import session as module
    from silent_cascade.archive.types import ArchivePolicy
    from silent_cascade.hashing import sha256_bytes

    policy = ArchivePolicy()
    run, control, identity = channel_fixture(tmp_path, policy)
    session = module.LocalArchiveSession(run_dir=run, control_dir=control, policy=policy)
    monkeypatch.setattr(module, "_STALL_SECONDS", 0.15)

    def parent():
        while not (control / "request.json").exists():
            time.sleep(0.005)
        raw = (control / "request.json").read_bytes()
        request = json.loads(raw)
        for delay in (0, 0.12):
            time.sleep(delay)
            module._publish(
                control,
                "progress.json",
                {
                    "session_id": identity["session_id"],
                    "sequence": request["sequence"],
                    "verified": 0,
                    "operation_time": time.monotonic(),
                },
                max_bytes=policy.page_bytes,
            )
        time.sleep(0.11)
        response = {
            key: request[key]
            for key in ("run_id", "session_id", "sequence", "operation", "policy_sha256", "unit_id")
        }
        response.update(request_sha256=sha256_bytes(raw), status="ok", payload={"healthy": True})
        module._publish(control, "response.json", response, max_bytes=policy.page_bytes)

    thread = threading.Thread(target=parent)
    thread.start()
    try:
        assert session._request("status", {}) == {"healthy": True}
    finally:
        thread.join(timeout=5)


def test_restart_quarantines_previous_protocol_before_new_sequence(running_archive):
    from silent_cascade.archive.session import LocalArchiveSession
    from silent_cascade.archive.supervisor import _ArchiveServer

    server, _ = running_archive
    session = LocalArchiveSession(
        run_dir=server.run_dir, control_dir=server.control_dir, policy=server.policy
    )
    assert session._request("status", {}) == {"healthy": True}
    # No prior scientific child is live; the production entrypoint holds the
    # permanent supervisor lock across this transition.
    restarted = _ArchiveServer(
        run_dir=server.run_dir,
        control_dir=server.control_dir,
        workspace_root=server.budget.workspace,
        transport=server.transport,
        policy=server.policy,
        run_id=server.run_id,
    )
    try:
        assert not (server.control_dir / "request.json").exists()
        retained = tuple((server.control_dir / "quarantine").rglob("request.json"))
        assert len(retained) == 1
    finally:
        restarted.close()


def test_child_compares_commit_owned_hashes_with_manifest(running_archive, monkeypatch):
    from dataclasses import asdict, replace

    from silent_cascade.archive.catalog import iter_unit_files
    from silent_cascade.archive.session import LocalArchiveSession, register_episode_binding
    from silent_cascade.archive.types import EpisodeCommit
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

    server, ref = running_archive
    commit = EpisodeCommit(
        "phase4-evaluation-episode-commit-v1",
        "2" * 64,
        2,
        "episode-2",
        "3" * 64,
        0,
        1,
        "4" * 64,
        tuple(iter_unit_files(server.control_dir, ref)),
        (),
    )
    binding = register_episode_binding(
        run_dir=server.run_dir,
        control_dir=server.control_dir,
        logical_root="evaluation",
        unit_ref=ref,
        commit=commit,
        transport=server.transport,
        policy=server.policy,
    )
    bad = replace(commit, owned=(replace(commit.owned[0], sha256="f" * 64),))
    bad_sha = sha256_bytes(canonical_json_bytes(asdict(bad)))
    session = LocalArchiveSession(
        run_dir=server.run_dir, control_dir=server.control_dir, policy=server.policy
    )
    original = session._request

    def mismatched_ack(operation, payload, **kwargs):
        if operation != "lease":
            return original(operation, payload, **kwargs)
        result = original(operation, dict(payload, commit_sha256=binding.commit_sha256), **kwargs)
        result["binding"]["commit"] = asdict(bad)
        result["binding"]["commit_sha256"] = bad_sha
        return result

    monkeypatch.setattr(session, "_request", mismatched_ack)
    with (
        pytest.raises(ValueError, match="commit"),
        session.episode("evaluation", 2, commit_sha256=bad_sha),
    ):
        pytest.fail("commit hashes were silently replaced by transport hashes")
