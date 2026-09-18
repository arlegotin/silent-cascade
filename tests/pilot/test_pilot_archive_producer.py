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
def producer_case(tmp_path, monkeypatch, *, journal_records=128):
    from silent_cascade.archive.ledger import initialize_workspace_ledger
    from silent_cascade.archive.producer import ArchiveProducer
    from silent_cascade.archive.session import LocalArchiveSession
    from silent_cascade.archive.supervisor import _ArchiveServer
    from silent_cascade.archive.transport import initialize_remote_reservations
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy(journal_records=journal_records)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    initialize_workspace_ledger(workspace_root=workspace, policy=policy, baseline=())
    run, control = workspace / "run", workspace / "control"
    run.mkdir()
    transport = DirectoryTransport(tmp_path / "remote")
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
    # Exercise the real parent handlers; file protocol authentication has adjacent coverage.
    monkeypatch.setattr(
        session, "_request", lambda op, payload, **kwargs: server._dispatch(op, payload)
    )
    try:
        with server.budget.reserve(
            admission={"job": "pilot"},
            spool=policy.spool_bytes,
            metadata=policy.metadata_bytes // 2,
            scratch=policy.scratch_bytes,
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
