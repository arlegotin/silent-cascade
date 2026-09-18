"""Durable producer ownership and publication cut points."""

import json
import os
from contextlib import contextmanager

import pytest

from silent_cascade.errors import DynamicsError
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.hashing import sha256_bytes
from silent_cascade.models.errors import NeuralError


@pytest.mark.parametrize("initialization", [False, True])
def test_crash_descriptor_names_actual_publication(
    neural_archive_case, tmp_path, monkeypatch, initialization
):
    case = neural_archive_case
    engine = EventEngine(
        case.config.event_flow,
        crash_root=tmp_path / "crashes",
        source_revision=case.revision,
        experiment_config_canonical_json=case.canonical,
    )

    def fail(*args):
        raise NeuralError("injected failure")

    monkeypatch.setattr(
        case.agent if initialization else case.model,
        "initialize" if initialization else "compose",
        fail,
    )
    with pytest.raises(DynamicsError):
        engine.run_episode(case.bundle, case.agent)
    published = engine.published_crash
    assert published.manifest.sha256 == sha256_bytes(published.manifest.path.read_bytes())
    manifest = json.loads(published.manifest.path.read_bytes())
    assert manifest["context"]["episode_public_id"] == case.bundle.public.init.episode_public_id
    if initialization:
        assert published.checkpoint is None
        assert published.shared_weights is None
    else:
        assert published.checkpoint.path.name == manifest["context"]["checkpoint_ref"]
        for entry in (published.checkpoint, published.shared_weights):
            assert entry.sha256 == sha256_bytes(entry.path.read_bytes())
        assert published.shared_weights.path != published.checkpoint.path


def test_pilot_bound_refuses_before_opening_destination(tmp_path, monkeypatch):
    from silent_cascade.train import pilot_data

    monkeypatch.setattr(pilot_data, "MAX_PILOT_MANIFEST_BYTES", 3)
    target = tmp_path / "absent/journal.json"
    with pytest.raises(ValueError, match="bound"):
        pilot_data._publish_pilot_bytes(target, b"four")
    assert not target.parent.exists()


@pytest.mark.parametrize("compressed", [False, True])
def test_trajectory_bound_refuses_before_opening_destination(tmp_path, monkeypatch, compressed):
    from silent_cascade.logging import neural_trace

    monkeypatch.setattr(neural_trace, "MAX_TRAJECTORY_JSON_BYTES", 4 if not compressed else 512)
    if compressed:
        monkeypatch.setattr(neural_trace.gzip, "compress", lambda *args, **kwargs: b"x" * 513)
    target = tmp_path / "absent/trace.json.gz"
    with pytest.raises(ValueError, match="bound"):
        neural_trace.write_full_neural_trace(
            target, identity_sha256="1" * 64, episode_sha256="2" * 64, trajectory=None
        )
    assert not target.parent.exists()


class DirectoryTransport:
    transport_id = "task4-local-test"

    def __init__(self, root):
        self.root = root

    def create(self, key, source):
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(source.read_bytes())

    def download(self, key, destination, *, max_bytes):
        raw = (self.root / key).read_bytes()
        if len(raw) > max_bytes:
            raise ValueError("download exceeds bound")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)


@contextmanager
def producer_case(tmp_path, monkeypatch, *, journal_records=128, stopped=False, restart=False):
    from silent_cascade.archive.ledger import initialize_workspace_ledger
    from silent_cascade.archive.producer import ArchiveProducer
    from silent_cascade.archive.session import LocalArchiveSession
    from silent_cascade.archive.supervisor import _ArchiveServer
    from silent_cascade.archive.transport import initialize_remote_reservations
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy(journal_records=journal_records)
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=restart)
    if not restart:
        initialize_workspace_ledger(workspace_root=workspace, policy=policy, baseline=())
    run, control = workspace / "run", workspace / "control"
    run.mkdir(exist_ok=restart)
    if stopped:
        from silent_cascade.hashing import canonical_json_bytes
        from silent_cascade.train.pilot_state import PilotCheckpointDescriptor
        from silent_cascade.train.pilot_workflow import pilot_ownership

        raw = b"bounded checkpoint fixture"
        digest = sha256_bytes(raw)
        descriptor = PilotCheckpointDescriptor(
            path=f"training-0-{digest}.safetensors",
            sha256=digest,
            model_state_sha256="2" * 64,
            global_step=0,
            stage="one_hop",
        )
        (run / descriptor.path).write_bytes(raw)
        (run / "checkpoint-index.json").write_bytes(
            canonical_json_bytes(
                {
                    "latest": descriptor.model_dump(mode="json"),
                    "history": [descriptor.model_dump(mode="json")],
                    "best": [],
                }
            )
        )
        with pilot_ownership(run, run_identity="a" * 64):
            pass
    transport = DirectoryTransport(tmp_path / "remote")
    if not restart:
        initialize_remote_reservations(
            control_dir=control,
            transport_id=transport.transport_id,
            policy=policy,
            accounted_bytes=0,
            accounting_evidence_sha256="f" * 64,
        )
    server = _ArchiveServer(
        run_dir=run,
        control_dir=control,
        workspace_root=workspace,
        transport=transport,
        policy=policy,
        run_id="task4",
    )
    session = LocalArchiveSession(run_dir=run, control_dir=control, policy=policy)
    server.child_pid = os.getpid()
    server.request = {"pid": os.getpid(), "unit_id": None}
    # Exercise the real parent handlers; file protocol authentication has adjacent coverage.
    monkeypatch.setattr(
        session, "_request", lambda op, payload, **kwargs: server._dispatch(op, payload)
    )
    try:
        with server.budget.reserve(
            admission={"job": "pilot"},
            spool=policy.spool_bytes - server.budget.measure()["spool"],
            metadata=policy.metadata_bytes // 2,
            scratch=policy.scratch_bytes - server.budget.measure()["scratch"],
            cache=policy.cache_bytes,
            pinned=policy.pinned_bytes,
        ):
            server.budget.inherit_reservations = True
            yield ArchiveProducer(session=session), session, server
    finally:
        server.close()


@pytest.mark.parametrize("cut", ["none", "row", "envelope", "receipt"])
def test_real_episode_handoff_and_failure_cutpoints(
    neural_archive_case, tmp_path, monkeypatch, cut
):
    from silent_cascade.eval.runner import evaluate_episodes

    from .test_timed_metrics import make_identity

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    synced_rows = []
    original_fsync = os.fsync

    def fsync(fd):
        original_fsync(fd)
        # The buffered row must have reached its file descriptor before handoff.
        if output.exists() and (output / ".rows.pending.jsonl").exists():
            raw = (output / ".rows.pending.jsonl").read_bytes()
            if raw:
                synced_rows.append(raw)

    with producer_case(tmp_path, monkeypatch) as (producer, session, _server):
        output = session.run_dir / "evaluation"
        original = session._request

        def request(operation, payload, **kwargs):
            if operation == "seal" and "commit" in payload:
                assert synced_rows and synced_rows[-1].endswith(b"\n")
                if cut == "envelope":
                    raise RuntimeError("injected envelope")
            if (
                operation == "archive"
                and payload["ref"]["kind"] == "episode_pack"
                and cut == "receipt"
            ):
                raise RuntimeError("injected receipt")
            return original(operation, payload, **kwargs)

        monkeypatch.setattr(session, "_request", request)
        monkeypatch.setattr(os, "fsync", fsync)
        if cut == "row":
            monkeypatch.setattr(
                producer,
                "after_episode",
                lambda *args: (_ for _ in ()).throw(RuntimeError("injected row")),
            )

        def evaluate():
            return evaluate_episodes(
                case.model,
                identity=identity,
                config=case.config,
                episodes=[case.bundle],
                output_dir=output,
                device="cpu",
                archive_producer=producer,
                evidence_context=session,
            )

        if cut != "none":
            with pytest.raises(RuntimeError, match="injected"):
                evaluate()
            assert not (output / "DONE").exists()
            assert (output / "episodes/00000.neural.json").exists()
            assert (output / ".rows.pending.jsonl").read_bytes().endswith(b"\n")
        else:
            result = evaluate()
            assert result.metrics.episode_count == 1
            assert not (output / "episodes/00000.neural.json").exists()
            assert "episodes/00000.neural.json" in dict(result.artifact_hashes)


def test_journal_handoff_rolls_before_any_checkpoint(tmp_path, monkeypatch):
    from silent_cascade.hashing import canonical_json_bytes

    with producer_case(tmp_path, monkeypatch, journal_records=2) as (producer, session, server):
        producer.bind_training(source_commit="b" * 40, config_sha256="a" * 64)
        head = None
        for step in range(1, 6):
            raw = canonical_json_bytes({"prior": head, "global_step": step, "kind": "update"})
            head = sha256_bytes(raw)
            path = session.run_dir / f"journal-{head}.json"
            path.write_bytes(raw)
            producer.after_journal(path.name, head)
        refs = [ref for ref in server._units() if ref.kind == "journal"]
        assert len(refs) > 1
        assert not (session.run_dir / "checkpoint-index.json").exists()
        # No record is discarded: every digest remains present in authenticated custody.
        entries = tuple(session.entries())
        assert len([entry for entry in entries if entry.path.startswith("journal-")]) == 5


@pytest.mark.parametrize("remaining", ["zero", "exhausted"])
def test_before_update_requires_remaining_admission(tmp_path, monkeypatch, remaining):
    from silent_cascade.archive.ledger import StorageBlocked

    with producer_case(tmp_path, monkeypatch) as (producer, session, server):
        state = server.budget._state()
        active = state["reservations"][server.budget.active_reservation]
        active["amounts"]["spool"] = 0 if remaining == "zero" else 64 * 1024**2 + 256 * 1024
        server.budget._store(state)
        if remaining == "exhausted":
            (session.run_dir / "retained-evidence").write_bytes(b"retain")
        with pytest.raises(StorageBlocked):
            producer.before_update(1)
        if remaining == "exhausted":
            assert (session.run_dir / "retained-evidence").read_bytes() == b"retain"


def test_real_four_update_training_preserves_scientific_state_with_cold_evidence(tmp_path):
    from pathlib import Path

    from .test_pilot_source import DATA_SETUP, checkout

    root, execute = checkout(tmp_path / "repo")
    helper = Path(__file__).parent
    execute(
        root,
        DATA_SETUP
        + f"\nhelper = {str(helper)!r}\n"
        + """
import sys
import json
from pytest import MonkeyPatch
sys.path.insert(0, helper)
from test_pilot_archive_producer import producer_case
from silent_cascade.train.pilot_trainer import run_pilot_training, verify_journal
from silent_cascade.train.pilot_workflow import pilot_ownership
whole = run_pilot_training(config, manifests=manifests, run_dir=root/'whole',
                          device='cpu', source_commit=source)
with MonkeyPatch.context() as patch:
    with producer_case(root, patch, journal_records=2) as (producer, session, server):
        complete_checkpoint = producer.after_checkpoint
        def interrupt(descriptor, progress):
            complete_checkpoint(descriptor, progress)
            if progress.global_step == 2:
                raise RuntimeError('after durable checkpoint cut')
        patch.setattr(producer, 'after_checkpoint', interrupt)
        with pilot_ownership(session.run_dir, run_identity='a' * 64):
            try:
                run_pilot_training(config, manifests=manifests, run_dir=session.run_dir,
                    device='cpu', source_commit=source, archive_producer=producer,
                    evidence_context=session)
            except RuntimeError as error:
                assert str(error) == 'after durable checkpoint cut'
            else:
                raise AssertionError('checkpoint interruption did not occur')
    with producer_case(root, patch, journal_records=2, restart=True) as (producer, session, server):
        from silent_cascade.train.pilot_workflow import _durable
        checked_source = authenticate_pilot_source(repo_root=root, source_commit=source,
                                                   config=config)
        _, introductions = verify_pilot_data(repo_root=root, source_commit=source,
                                            config=config, manifests=manifests)
        checked_source = checked_source.model_copy(update={'data_introductions': introductions})
        def recover():
            return _durable(session.run_dir, config, checked_source, 'cpu',
                            evidence_context=session)
        with pilot_ownership(session.run_dir, run_identity='a' * 64, recover=recover,
                             capture_stopped=True) as stopped:
            producer.adopt_stopped(stopped, source_commit=source, config_sha256=config.sha256)
            archived = run_pilot_training(config, manifests=manifests, run_dir=session.run_dir,
                device='cpu', source_commit=source, archive_producer=producer,
                evidence_context=session,
                resume=session.run_dir/stopped.checkpoint.path)
        assert archived.model_identity == whole.model_identity
        assert archived.progress.global_step == whole.progress.global_step == 4
        assert archived.progress.batch_counter == whole.progress.batch_counter == 4
        assert archived.latest_weights.sha256 == whole.latest_weights.sha256
        journal = verify_journal(session.run_dir, archived.progress, evidence_context=session)
        assert len(journal) == 6
        assert any(ref.kind == 'journal' for ref in server._units())
        assert len([p for p in archived.artifact_hashes if p.endswith('.neural.json')]) == 32
        assert not list(session.run_dir.rglob('*.neural.json'))
""",
        timeout=420,
    )


@pytest.mark.parametrize(
    "mode", ["cold", "owned-resident", "corrupt", "pinned", "capacity", "execution"]
)
def test_completed_evaluation_releases_shared_weights_and_can_lease_them_cold(
    neural_archive_case, tmp_path, monkeypatch, mode
):
    from dataclasses import asdict

    from silent_cascade.eval.runner import evaluate_episodes
    from silent_cascade.hashing import canonical_json_bytes

    from .test_timed_metrics import make_identity

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    with producer_case(tmp_path, monkeypatch) as (producer, session, server):
        commits = []
        real_after = producer.after_episode

        def record(root, commit):
            commits.append(commit)
            real_after(root, commit)

        monkeypatch.setattr(producer, "after_episode", record)
        evaluate_episodes(
            case.model,
            identity=identity,
            config=case.config,
            episodes=[case.bundle],
            output_dir=session.run_dir / "evaluation",
            device="cpu",
            archive_producer=producer,
            evidence_context=session,
        )
        assert commits[0].borrowed
        weights = commits[0].borrowed[0]
        assert not (session.run_dir / weights.path).exists()
        digest = sha256_bytes(canonical_json_bytes(asdict(commits[0])))
        owner, _ = server._find_owner(weights.path)
        if mode == "owned-resident":
            with session.episode("evaluation", 0, commit_sha256=digest) as lease:
                for entry in commits[0].owned:
                    target = session.run_dir / entry.path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes((lease.local_root / entry.path).read_bytes())
        if mode == "corrupt":
            from silent_cascade.archive.transport import _object_key

            with server._metadata(owner) as (_, manifest):
                path = server.transport.root / _object_key(
                    "task4", f"objects/{manifest.chunks[0].sha256}.bin"
                )
                path.write_bytes(b"corrupt weights")
            with pytest.raises(ValueError), session.episode("evaluation", 0, commit_sha256=digest):
                pass
            assert not server.leases
            return
        if mode == "capacity":
            state = server.budget._state()
            state["reservations"][server.budget.active_reservation]["amounts"]["cache"] = (
                weights.bytes
            )
            server.budget._store(state)
            with (
                pytest.raises(Exception, match="remaining admission"),
                session.episode("evaluation", 0, commit_sha256=digest),
            ):
                pass
            assert not server.leases
            return
        with session.episode("evaluation", 0, commit_sha256=digest) as lease:
            assert sha256_bytes((lease.local_root / weights.path).read_bytes()) == weights.sha256
            assert (
                len(
                    [
                        value
                        for value in server.leases.values()
                        if value["ref"].kind == "episode_pack"
                    ]
                )
                == 1
            )
            if mode == "pinned":
                with pytest.raises(Exception, match="borrowed shared file is pinned"):
                    session._request("archive", {"ref": asdict(owner), "evict": True})
        assert not server.leases
        if mode == "execution":
            (session.run_dir / "evaluation/execution.json").write_bytes(b"{}")
            producer.after_execution("evaluation")
            assert not (session.run_dir / "evaluation/DONE").exists()
            with session.metadata("evaluation") as lease:
                assert (lease.local_root / "evaluation/DONE").is_file()


@pytest.mark.parametrize("failure", ["none", "runtime", "initialization"])
def test_two_rows_preserve_local_scientific_bytes_and_shared_custody(
    neural_archive_case, tmp_path, monkeypatch, failure
):
    from dataclasses import replace
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from silent_cascade.eval.runner import evaluate_episodes
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent
    from silent_cascade.logging import crash_bundle
    from silent_cascade.models.event_flow import EventFlowModel

    from .test_timed_metrics import make_identity

    case = neural_archive_case
    second = replace(
        case.bundle,
        public=replace(
            case.bundle.public,
            init=replace(
                case.bundle.public.init, episode_public_id="00000000-0000-4000-8000-000000000002"
            ),
        ),
    )
    identity = make_identity(case, (case.bundle, second), tmp_path)

    def fail(*args):
        raise NeuralError("injected failure")

    if failure == "runtime":
        monkeypatch.setattr(EventFlowModel, "compose", fail)
    elif failure == "initialization":
        monkeypatch.setattr(NeuralEventFlowAgent, "initialize", fail)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 17, tzinfo=UTC)

    monkeypatch.setattr(crash_bundle, "datetime", FixedDatetime)
    monkeypatch.setattr(
        "silent_cascade.eventflow.neural_checkpoint.uuid4", lambda: SimpleNamespace(hex="c" * 32)
    )
    monkeypatch.setattr("silent_cascade.eval.compute.perf_counter", lambda: 1.0)
    with producer_case(tmp_path, monkeypatch) as (producer, session, _server):
        for output, selected_producer in (
            (tmp_path / "local", None),
            (session.run_dir / "evaluation", producer),
        ):
            from silent_cascade.rng import seed_all

            seed_all(11)
            ids = iter(("1" * 32, "2" * 32))
            monkeypatch.setattr(
                "silent_cascade.eventflow.engine.uuid4",
                lambda ids=ids: SimpleNamespace(hex=next(ids)),
            )
            result = evaluate_episodes(
                case.model,
                identity=identity,
                config=case.config,
                episodes=[case.bundle, second],
                output_dir=output,
                device="cpu",
                archive_producer=selected_producer,
                evidence_context=session if selected_producer else None,
            )
            if selected_producer is None:
                local_hashes = dict(result.artifact_hashes)
            else:
                assert dict(result.artifact_hashes) == local_hashes
                assert result.metrics.episode_count == 2
                assert result.metrics.error_count == (0 if failure == "none" else 2)
                assert not list(output.glob("episodes/*"))


def test_stopped_custody_is_yielded_only_after_durable_recovery(tmp_path):
    from silent_cascade.hashing import canonical_json_bytes
    from silent_cascade.train.pilot_workflow import pilot_ownership

    run = tmp_path / "run"
    run.mkdir()
    checkpoint = {
        "path": "training-0-" + "1" * 64 + ".safetensors",
        "sha256": "1" * 64,
        "model_state_sha256": "2" * 64,
        "global_step": 0,
        "stage": "one_hop",
        "rank": [],
        "eligible": False,
    }
    (run / "checkpoint-index.json").write_bytes(canonical_json_bytes({"latest": checkpoint}))
    with pilot_ownership(run, run_identity="a" * 64):
        pass
    prior = (tmp_path / ".run.owner.json").read_bytes()
    recovered = []
    with pilot_ownership(
        run, run_identity="a" * 64, recover=lambda: recovered.append(True), capture_stopped=True
    ) as custody:
        assert recovered == [True]
        assert custody.prior_owner_sha256 == sha256_bytes(prior)
        assert custody.prior_owner_canonical_json.encode() == prior
        assert custody.checkpoint.path == checkpoint["path"]


def test_stopped_unknown_bytes_are_partial_custody_not_episode_outcomes(tmp_path, monkeypatch):
    from silent_cascade.train.pilot_workflow import pilot_ownership

    with producer_case(tmp_path, monkeypatch, stopped=True) as (producer, session, server):
        attempt = session.run_dir / ("attempt-" + "1" * 32)
        attempt.mkdir()
        orphan = attempt / "orphan-crash.bin"
        orphan.write_bytes(b"unknown outcome retained")
        with pilot_ownership(
            session.run_dir, run_identity="a" * 64, recover=lambda: None, capture_stopped=True
        ) as custody:
            producer.adopt_stopped(custody, source_commit="b" * 40, config_sha256="c" * 64)
            assert not orphan.exists()
            refs = tuple(ref for ref in server._units() if ref.kind == "partial")
            assert len(refs) == 1
            with session.lease(refs[0]) as lease:
                assert (
                    lease.local_root / orphan.relative_to(session.run_dir)
                ).read_bytes() == b"unknown outcome retained"
            assert not list(session.run_dir.rglob("DONE"))
            assert not (session.control_dir / "episode-bindings/head.json").exists()


@pytest.mark.parametrize("authority", ["valid", "prior", "child"])
def test_empty_stopped_handoff_requires_parent_authority(tmp_path, monkeypatch, authority):
    from silent_cascade.train.pilot_workflow import pilot_ownership

    with producer_case(tmp_path, monkeypatch, stopped=True) as (producer, session, server):
        if authority == "prior":
            raw, device, inode = server._prior_pilot_owner
            server._prior_pilot_owner = (raw + b" ", device, inode)
        if authority == "child":
            server.child_pid = os.getpid() + 1000
        with pilot_ownership(
            session.run_dir, run_identity="a" * 64, recover=lambda: None, capture_stopped=True
        ) as custody:
            if authority == "valid":
                producer.adopt_stopped(custody, source_commit="b" * 40, config_sha256="c" * 64)
                assert producer.stopped_checkpoint == custody.checkpoint
                assert not tuple(server._units())
            else:
                with pytest.raises(ValueError, match="custody"):
                    producer.adopt_stopped(custody, source_commit="b" * 40, config_sha256="c" * 64)
                assert producer.stopped_checkpoint is None


def test_old_checkpoint_snapshot_reads_its_original_index_after_replacement(tmp_path, monkeypatch):
    from silent_cascade.hashing import canonical_json_bytes
    from silent_cascade.train.pilot_state import PilotCheckpointDescriptor, PilotProgress

    with producer_case(tmp_path, monkeypatch) as (producer, session, server):
        producer.bind_training(source_commit="b" * 40, config_sha256="c" * 64)
        original = None
        for step in (0, 1):
            raw = f"checkpoint-{step}".encode()
            digest = sha256_bytes(raw)
            descriptor = PilotCheckpointDescriptor(
                path=f"training-{step}-{digest}.safetensors",
                sha256=digest,
                model_state_sha256="2" * 64,
                global_step=step,
                stage="one_hop",
            )
            (session.run_dir / descriptor.path).write_bytes(raw)
            index = canonical_json_bytes(
                {
                    "latest": descriptor.model_dump(mode="json"),
                    "history": [descriptor.model_dump(mode="json")],
                    "best": [],
                }
            )
            (session.run_dir / "checkpoint-index.json").write_bytes(index)
            producer.after_checkpoint(
                descriptor, PilotProgress().model_copy(update={"global_step": step})
            )
            if step == 0:
                original = (
                    next(ref for ref in server._units() if ref.kind == "control_snapshot"),
                    index,
                )
        with session.lease(original[0]) as lease:
            assert (lease.local_root / "checkpoint-index.json").read_bytes() == original[1]


def test_reservation_only_publication_is_cold_discoverable_after_interruption(
    tmp_path, monkeypatch
):
    from silent_cascade.archive import transport as module
    from silent_cascade.archive.operational import lookup_remote_reservation

    with producer_case(tmp_path, monkeypatch) as (_, session, server):
        raw = b"preserved stopped control"
        source = session.control_dir / "pending-control"
        source.write_bytes(raw)
        key = module._object_key("task4", "stopped-controls/" + sha256_bytes(raw) + ".json")
        with module.archive_operation_lock(session.control_dir):
            module._put_verified(
                control_dir=session.control_dir,
                transport=server.io,
                key=key,
                source=source,
                expected_bytes=len(raw),
                expected_sha256=sha256_bytes(raw),
                policy=session.policy,
            )
            original = module._transfer_verified

            def interrupt(**kwargs):
                if "operational/" in kwargs["key"]:
                    raise OSError("operational cut")
                return original(**kwargs)

            monkeypatch.setattr(module, "_transfer_verified", interrupt)
            with pytest.raises(OSError, match="operational cut"):
                module._publish_operational_batch(
                    control_dir=session.control_dir, transport=server.io, policy=session.policy
                )
            assert source.read_bytes() == raw
            monkeypatch.setattr(module, "_transfer_verified", original)
            module._resume_operational_pending(
                control_dir=session.control_dir, transport=server.io, policy=session.policy
            )
        state = json.loads((session.control_dir / "remote-reservations/state.json").read_bytes())
        record = lookup_remote_reservation(
            session.control_dir,
            module._operational_ref(state),
            object_key=key,
            policy=session.policy,
            object_reader=module._operational_cold_reader(
                control_dir=session.control_dir, transport=server.io
            ),
        )
        assert record.sha256 == sha256_bytes(raw)
        assert not list((session.control_dir / "remote-reservations/objects").glob("*.json"))
        assert not (session.control_dir / "active-receipts").exists()


def test_episode_admission_accounts_shared_weights_in_actual_spool_category():
    from silent_cascade.archive.producer import (
        ALLOCATION_OVERHEAD,
        ArtifactOutputBounds,
        before_work_bounds,
    )

    bounds = ArtifactOutputBounds()
    assert before_work_bounds("episode")["spool"] >= (
        bounds.episode_owned + bounds.episode_row + bounds.shared_weights + ALLOCATION_OVERHEAD
    )


@pytest.mark.parametrize("cut", [None, "upload", "readback", "operational", "changed"])
def test_stopped_pending_control_is_preserved_before_slot_rotation(tmp_path, monkeypatch, cut):
    from dataclasses import asdict

    from silent_cascade.archive import transport as module
    from silent_cascade.archive.catalog import seal_unit
    from silent_cascade.archive.operational import iter_operational_records
    from silent_cascade.archive.types import EpisodeCommit, FileEntry, UnitIdentity
    from silent_cascade.hashing import canonical_json_bytes
    from silent_cascade.train.pilot_workflow import pilot_ownership

    with producer_case(tmp_path, monkeypatch, stopped=True) as (producer, session, server):
        attempt = session.run_dir / ("attempt-" + "1" * 32)
        attempt.mkdir()
        member = attempt / "owned.bin"
        member.write_bytes(b"owned bytes")
        entry = FileEntry(
            member.relative_to(session.run_dir).as_posix(),
            sha256_bytes(member.read_bytes()),
            len(member.read_bytes()),
        )
        seal_unit(
            run_dir=session.run_dir,
            control_dir=session.control_dir,
            logical_root=attempt.name,
            paths=(entry.path,),
            kind="diagnostic",
            identity=UnitIdentity(
                run_id="task4",
                source_commit="b" * 40,
                config_sha256="c" * 64,
                evidence_identity_sha256="d" * 64,
                checkpoint_sha256=None,
                writer_stopped=True,
                checkpoint_committed=False,
            ),
            policy=session.policy,
        )
        commit = EpisodeCommit(
            "phase4-evaluation-episode-commit-v1",
            "a" * 64,
            0,
            "episode",
            "e" * 64,
            0,
            1,
            "f" * 64,
            (entry,),
            (),
        )
        raw = canonical_json_bytes(asdict(commit))
        pending = session.control_dir / "episode-pending.json"
        pending.write_bytes(raw)
        original = module._put_verified

        def put(**kwargs):
            if "/stopped-controls/" in kwargs["key"] and cut == "upload":
                raise OSError("control upload cut")
            result = original(**kwargs)
            if "/stopped-controls/" in kwargs["key"] and cut == "readback":
                raise OSError("control readback cut")
            return result

        monkeypatch.setattr(module, "_put_verified", put)
        publish = module._publish_operational_batch

        def operational(**kwargs):
            if cut == "operational":
                raise OSError("control operational cut")
            result = publish(**kwargs)
            if cut == "changed":
                pending.write_bytes(b"changed control")
            return result

        monkeypatch.setattr(module, "_publish_operational_batch", operational)
        with pilot_ownership(
            session.run_dir, run_identity="a" * 64, recover=lambda: None, capture_stopped=True
        ) as custody:
            if cut:
                with pytest.raises((OSError, ValueError)):
                    producer.adopt_stopped(custody, source_commit="b" * 40, config_sha256="c" * 64)
                assert pending.read_bytes() == (b"changed control" if cut == "changed" else raw)
            else:
                producer.adopt_stopped(custody, source_commit="b" * 40, config_sha256="c" * 64)
                assert not pending.exists()
                assert not [ref for ref in server._units() if ref.kind == "partial"]
                state = json.loads(
                    (session.control_dir / "remote-reservations/state.json").read_bytes()
                )
                records = tuple(
                    iter_operational_records(
                        session.control_dir,
                        module._operational_ref(state),
                        index="reservations",
                        policy=session.policy,
                        object_reader=module._operational_cold_reader(
                            control_dir=session.control_dir, transport=server.io
                        ),
                    )
                )
                controls = [
                    record for record in records if "/stopped-controls/" in record.object_key
                ]
                assert len(controls) == 1
                preserved = (server.transport.root / controls[0].object_key).read_bytes()
                assert sha256_bytes(preserved) == controls[0].sha256
                assert json.loads(preserved)["control_canonical_json"].encode() == raw
