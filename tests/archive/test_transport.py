import json
import os
import sys
import time
from pathlib import Path

import pytest

from silent_cascade.hashing import sha256_bytes


def _archive(sealed_unit, transport):
    from silent_cascade.archive.transport import archive_unit

    return archive_unit(
        run_dir=sealed_unit.root,
        control_dir=sealed_unit.control,
        ref=sealed_unit.ref,
        transport=transport,
        policy=sealed_unit.policy,
    )


def test_readback_corruption_never_authorizes_eviction(sealed_unit, transport):
    transport.corrupt_downloads = True
    before = sealed_unit.original_bytes()
    with pytest.raises(ValueError, match="readback"):
        _archive(sealed_unit, transport)
    assert sealed_unit.original_bytes() == before
    assert not tuple(sealed_unit.control.glob("receipts/*.json"))


def test_existing_different_object_is_not_overwritten(sealed_unit, transport):
    transport.collision_create_call = 1
    before = sealed_unit.original_bytes()
    with pytest.raises(ValueError, match=r"collision|readback"):
        _archive(sealed_unit, transport)
    assert sealed_unit.original_bytes() == before
    assert not tuple(sealed_unit.control.glob("receipts/*.json"))


@pytest.mark.parametrize("fault", ["same-content-conflict", "ambiguous-write"])
def test_create_recovery_requires_exact_readback(sealed_unit, transport, fault):
    if fault == "same-content-conflict":
        transport.conflict_create_call = 1
    else:
        transport.ambiguous_create_call = 1
    receipt_sha256 = _archive(sealed_unit, transport)
    receipts = tuple(sealed_unit.control.glob("receipts/*.json"))
    assert len(receipts) == 1
    assert sha256_bytes(receipts[0].read_bytes()) == receipt_sha256
    assert transport.download_calls >= transport.create_calls


def test_repeated_archive_revalidates_and_returns_same_receipt(sealed_unit, transport):
    first = _archive(sealed_unit, transport)
    downloads = transport.download_calls
    second = _archive(sealed_unit, transport)
    assert second == first
    assert transport.download_calls > downloads


def test_operational_publication_cut_point_recovers_without_authorizing_early(
    sealed_unit, transport
):
    transport.fail_key_contains = "operational/"
    with pytest.raises(OSError, match="keyed create failure"):
        _archive(sealed_unit, transport)
    assert (sealed_unit.control / "remote-reservations/operational-pending.json").exists()
    assert not (sealed_unit.control / f"active-receipts/{sealed_unit.ref.unit_id}.json").exists()
    transport.fail_key_contains = None
    receipt_sha256 = _archive(sealed_unit, transport)
    assert len(receipt_sha256) == 64
    assert not (sealed_unit.control / "remote-reservations/operational-pending.json").exists()


def test_operational_pending_before_counter_update_recovers(sealed_unit, transport, monkeypatch):
    from silent_cascade.archive import transport as module

    original = module._replace_at
    interrupted = False

    def interrupt_counter(parent, name, payload):
        nonlocal interrupted
        if name == "state.json" and not interrupted:
            try:
                os.stat("operational-pending.json", dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                interrupted = True
                raise OSError("injected counter interruption")
        return original(parent, name, payload)

    monkeypatch.setattr(module, "_replace_at", interrupt_counter)
    with pytest.raises(OSError, match="counter interruption"):
        _archive(sealed_unit, transport)
    monkeypatch.setattr(module, "_replace_at", original)
    receipt_sha256 = _archive(sealed_unit, transport)
    assert len(receipt_sha256) == 64


def test_receipt_cut_before_operational_staging_resumes_on_retry(
    sealed_unit, transport, monkeypatch
):
    from silent_cascade.archive import transport as module

    original = module._publish_operational_batch

    def interrupt_before_staging(**_kwargs):
        raise OSError("injected post-receipt interruption")

    monkeypatch.setattr(module, "_publish_operational_batch", interrupt_before_staging)
    with pytest.raises(OSError, match="post-receipt interruption"):
        _archive(sealed_unit, transport)
    assert tuple(sealed_unit.control.glob("receipts/*.json"))
    assert not (sealed_unit.control / "remote-reservations/operational-pending.json").exists()

    monkeypatch.setattr(module, "_publish_operational_batch", original)
    assert len(_archive(sealed_unit, transport)) == 64
    assert (sealed_unit.control / f"active-receipts/{sealed_unit.ref.unit_id}.json").exists()


def test_catalog_generation_can_extend_from_cold_authenticated_nodes(
    sealed_unit, transport, archive_identity
):
    from silent_cascade.archive.catalog import seal_unit

    _archive(sealed_unit, transport)
    for path in (sealed_unit.control / "catalog").rglob("*.json"):
        path.unlink()
    second_path = sealed_unit.root / "other/evidence.json"
    second_path.parent.mkdir()
    second_path.write_bytes(b"other evidence")
    second = seal_unit(
        run_dir=sealed_unit.root,
        control_dir=sealed_unit.control,
        logical_root="other",
        paths=("other/evidence.json",),
        kind="diagnostic",
        identity=archive_identity,
        policy=sealed_unit.policy,
    )
    receipt = _archive(
        type(
            "SecondUnit",
            (),
            {
                "root": sealed_unit.root,
                "control": sealed_unit.control,
                "ref": second,
                "policy": sealed_unit.policy,
            },
        )(),
        transport,
    )
    assert len(receipt) == 64


def test_archive_catalog_cleanup_preserves_unrelated_corpus_pages(sealed_unit, transport):
    from silent_cascade.archive.catalog import (
        iter_corpus_catalog,
        publish_corpus_catalog,
        publish_run_catalog,
    )

    run_catalog = publish_run_catalog(
        control_dir=sealed_unit.control,
        run_id="debug-fixture",
        units=(sealed_unit.ref,),
        policy=sealed_unit.policy,
    )
    corpus_catalog = publish_corpus_catalog(
        control_dir=sealed_unit.control,
        runs=(run_catalog,),
        policy=sealed_unit.policy,
    )
    corpus_root = sealed_unit.control / corpus_catalog.root_path
    before = corpus_root.read_bytes()

    _archive(sealed_unit, transport)

    assert corpus_root.read_bytes() == before
    assert tuple(
        iter_corpus_catalog(
            sealed_unit.control,
            corpus_catalog,
            policy=sealed_unit.policy,
        )
    ) == (run_catalog,)


def test_catalog_cleanup_requires_exact_bytes_and_unambiguous_ownership(task_scratch):
    from silent_cascade.archive import transport as module

    control = task_scratch / "cleanup-control"
    logical = f"catalog/nodes/{'a' * 64}.json"
    path = control / logical
    path.parent.mkdir(parents=True)
    path.write_bytes(b"different\n")
    assert not module._remove_verified_control_object(control, logical, b"verified\n")
    assert path.read_bytes() == b"different\n"

    path.write_bytes(b"verified\n")
    alias = task_scratch / "shared-alias"
    os.link(path, alias)
    assert not module._remove_verified_control_object(control, logical, b"verified\n")
    assert path.read_bytes() == b"verified\n"
    alias.unlink()

    assert module._remove_verified_control_object(control, logical, b"verified\n")
    assert not path.exists()


@pytest.mark.parametrize("transition", ["create", "readback", "catalog"])
def test_recovery_cut_points_leave_sources_and_no_receipt(sealed_unit, transport, transition):
    before = sealed_unit.original_bytes()
    if transition == "create":
        transport.fail_create_call = 1
    elif transition == "readback":
        transport.fail_download_call = 1
    else:
        transport.fail_create_call = len(sealed_unit.ref.unit_id) + 1
        transport.fail_create_call = 7
    with pytest.raises((OSError, PermissionError, ValueError)):
        _archive(sealed_unit, transport)
    assert sealed_unit.original_bytes() == before
    assert not tuple(sealed_unit.control.glob("receipts/*.json"))


def test_archive_refuses_missing_or_changed_reservation_ledger(sealed_unit, transport):
    ledger = sealed_unit.control / "remote-reservations/state.json"
    original = ledger.read_bytes()
    ledger.unlink()
    with pytest.raises(ValueError, match="reservation ledger"):
        _archive(sealed_unit, transport)
    ledger.write_bytes(original.replace(transport.transport_id.encode(), b"changed-transport"))
    with pytest.raises(ValueError, match=r"reservation ledger|transport"):
        _archive(sealed_unit, transport)


def test_reservation_bootstrap_is_create_only_and_rejects_prior_history(
    task_scratch, tiny_archive_policy
):
    from silent_cascade.archive.transport import initialize_remote_reservations

    control = task_scratch / "bootstrap"
    initialize_remote_reservations(
        control_dir=control,
        transport_id="directory:fresh",
        accounted_bytes=17,
        accounting_evidence_sha256="b" * 64,
        policy=tiny_archive_policy,
    )
    with pytest.raises((FileExistsError, ValueError), match=r"already|exists"):
        initialize_remote_reservations(
            control_dir=control,
            transport_id="directory:fresh",
            accounted_bytes=0,
            accounting_evidence_sha256="c" * 64,
            policy=tiny_archive_policy,
        )
    prior = task_scratch / "prior"
    (prior / "catalog/roots").mkdir(parents=True)
    (prior / "catalog/roots/prior.json").write_text("history")
    with pytest.raises(ValueError, match="history"):
        initialize_remote_reservations(
            control_dir=prior,
            transport_id="directory:fresh",
            accounted_bytes=0,
            accounting_evidence_sha256="d" * 64,
            policy=tiny_archive_policy,
        )


def test_remote_quota_is_reserved_before_any_write(
    task_scratch, tiny_archive_policy, archive_identity, transport
):
    from silent_cascade.archive.catalog import seal_unit

    policy = tiny_archive_policy.model_copy(update={"remote_bytes": 1})
    run = task_scratch / "quota-run"
    (run / "unit").mkdir(parents=True)
    (run / "unit/a").write_bytes(b"too-large")
    ref = seal_unit(
        run_dir=run,
        control_dir=task_scratch / "control",
        logical_root="unit",
        paths=("unit/a",),
        kind="diagnostic",
        identity=archive_identity,
        policy=policy,
    )
    with pytest.raises(ValueError, match="remote byte budget"):
        from silent_cascade.archive.transport import archive_unit

        archive_unit(
            run_dir=run,
            control_dir=task_scratch / "control",
            ref=ref,
            transport=transport,
            policy=policy,
        )
    assert transport.create_calls == 0


def test_verified_receipt_catalog_and_eviction_order(sealed_unit, transport):
    from silent_cascade.archive.transport import evict_unit

    receipt_sha256 = _archive(sealed_unit, transport)
    head = tuple((sealed_unit.control / "catalog-heads").glob("*.json"))
    assert len(head) == 1
    evict_unit(
        run_dir=sealed_unit.root,
        control_dir=sealed_unit.control,
        ref=sealed_unit.ref,
        receipt_sha256=receipt_sha256,
    )
    assert all(not (sealed_unit.root / path).exists() for path in sealed_unit.paths)
    intents = tuple((sealed_unit.control / "evictions").glob("*.json"))
    assert len(intents) == 1
    assert json.loads(intents[0].read_bytes())["completed"] is True


def test_later_catalog_generation_does_not_block_earlier_unit_eviction(
    sealed_unit, transport, archive_identity
):
    from silent_cascade.archive.catalog import seal_unit
    from silent_cascade.archive.transport import archive_unit, evict_unit

    first_receipt = _archive(sealed_unit, transport)
    second_path = sealed_unit.root / "later/evidence.json"
    second_path.parent.mkdir()
    second_path.write_bytes(b"later")
    second = seal_unit(
        run_dir=sealed_unit.root,
        control_dir=sealed_unit.control,
        logical_root="later",
        paths=("later/evidence.json",),
        kind="diagnostic",
        identity=archive_identity,
        policy=sealed_unit.policy,
    )
    archive_unit(
        run_dir=sealed_unit.root,
        control_dir=sealed_unit.control,
        ref=second,
        transport=transport,
        policy=sealed_unit.policy,
    )

    evict_unit(
        run_dir=sealed_unit.root,
        control_dir=sealed_unit.control,
        ref=sealed_unit.ref,
        receipt_sha256=first_receipt,
    )
    assert all(not (sealed_unit.root / path).exists() for path in sealed_unit.paths)
    assert second_path.read_bytes() == b"later"


def test_catalog_history_is_cold_and_earlier_eviction_uses_bounded_proof(
    sealed_unit, transport, archive_identity
):
    from silent_cascade.archive.catalog import seal_unit
    from silent_cascade.archive.transport import archive_unit, evict_unit

    first_receipt = _archive(sealed_unit, transport)
    for index in range(1):
        logical_root = f"later-{index}"
        logical_path = f"{logical_root}/evidence.json"
        path = sealed_unit.root / logical_path
        path.parent.mkdir()
        path.write_bytes(f"later-{index}".encode())
        ref = seal_unit(
            run_dir=sealed_unit.root,
            control_dir=sealed_unit.control,
            logical_root=logical_root,
            paths=(logical_path,),
            kind="diagnostic",
            identity=archive_identity,
            policy=sealed_unit.policy,
        )
        archive_unit(
            run_dir=sealed_unit.root,
            control_dir=sealed_unit.control,
            ref=ref,
            transport=transport,
            policy=sealed_unit.policy,
        )

    assert not tuple((sealed_unit.control / "catalog").rglob("*.json"))
    proof_files = tuple(
        (sealed_unit.control / f"catalog-proofs/{sealed_unit.ref.unit_id}").rglob("*.json")
    )
    assert 1 <= len(proof_files) <= 32
    downloads = transport.download_calls
    evict_unit(
        run_dir=sealed_unit.root,
        control_dir=sealed_unit.control,
        ref=sealed_unit.ref,
        receipt_sha256=first_receipt,
    )
    assert transport.download_calls == downloads


def test_stale_receipt_never_authorizes_eviction(sealed_unit, transport):
    from silent_cascade.archive.transport import evict_unit

    receipt_sha256 = _archive(sealed_unit, transport)
    changed = sealed_unit.root / sealed_unit.paths[0]
    changed.write_bytes(b"other")
    with pytest.raises(ValueError, match=r"changed|receipt"):
        evict_unit(
            run_dir=sealed_unit.root,
            control_dir=sealed_unit.control,
            ref=sealed_unit.ref,
            receipt_sha256=receipt_sha256,
        )
    assert changed.read_bytes() == b"other"


def test_catalog_proof_corruption_never_authorizes_eviction(sealed_unit, transport):
    from silent_cascade.archive.transport import evict_unit

    receipt_sha256 = _archive(sealed_unit, transport)
    proof = next(
        (sealed_unit.control / f"catalog-proofs/{sealed_unit.ref.unit_id}").rglob("*.json")
    )
    proof.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="catalog proof"):
        evict_unit(
            run_dir=sealed_unit.root,
            control_dir=sealed_unit.control,
            ref=sealed_unit.ref,
            receipt_sha256=receipt_sha256,
        )
    assert sealed_unit.original_bytes()


def test_missing_operational_authorization_blocks_eviction(sealed_unit, transport):
    from silent_cascade.archive.transport import evict_unit

    receipt_sha256 = _archive(sealed_unit, transport)
    (sealed_unit.control / f"active-receipts/{sealed_unit.ref.unit_id}.json").unlink()
    with pytest.raises(ValueError, match="operational authorization"):
        evict_unit(
            run_dir=sealed_unit.root,
            control_dir=sealed_unit.control,
            ref=sealed_unit.ref,
            receipt_sha256=receipt_sha256,
        )


def test_next_archive_colds_completed_receipt_intent_and_reservations(
    sealed_unit, transport, archive_identity
):
    from silent_cascade.archive.catalog import seal_unit
    from silent_cascade.archive.transport import archive_unit, evict_unit

    first_receipt = _archive(sealed_unit, transport)
    evict_unit(
        run_dir=sealed_unit.root,
        control_dir=sealed_unit.control,
        ref=sealed_unit.ref,
        receipt_sha256=first_receipt,
    )
    second_path = sealed_unit.root / "next/evidence.json"
    second_path.parent.mkdir()
    second_path.write_bytes(b"next")
    second = seal_unit(
        run_dir=sealed_unit.root,
        control_dir=sealed_unit.control,
        logical_root="next",
        paths=("next/evidence.json",),
        kind="diagnostic",
        identity=archive_identity,
        policy=sealed_unit.policy,
    )
    archive_unit(
        run_dir=sealed_unit.root,
        control_dir=sealed_unit.control,
        ref=second,
        transport=transport,
        policy=sealed_unit.policy,
    )
    assert not (sealed_unit.control / f"receipts/{sealed_unit.ref.unit_id}.json").exists()
    assert not (sealed_unit.control / f"evictions/{sealed_unit.ref.unit_id}.json").exists()
    assert not (sealed_unit.control / f"active-receipts/{sealed_unit.ref.unit_id}.json").exists()
    assert not (sealed_unit.control / f"catalog-proofs/{sealed_unit.ref.unit_id}").exists()
    assert not tuple((sealed_unit.control / "remote-reservations/objects").glob("*.json"))


def test_unknown_file_stops_eviction(sealed_unit, transport):
    from silent_cascade.archive.transport import evict_unit

    receipt_sha256 = _archive(sealed_unit, transport)
    unknown = sealed_unit.root / "pack/unexpected.bin"
    unknown.write_bytes(b"unknown")
    with pytest.raises(ValueError, match="unknown"):
        evict_unit(
            run_dir=sealed_unit.root,
            control_dir=sealed_unit.control,
            ref=sealed_unit.ref,
            receipt_sha256=receipt_sha256,
        )
    assert sealed_unit.original_bytes()


def test_shared_reader_lease_blocks_writer_eviction(sealed_unit, transport):
    from silent_cascade.archive.transport import evict_unit, unit_reader_lease

    receipt_sha256 = _archive(sealed_unit, transport)
    with (
        unit_reader_lease(control_dir=sealed_unit.control, ref=sealed_unit.ref),
        pytest.raises(ValueError, match=r"lease|writer lock"),
    ):
        evict_unit(
            run_dir=sealed_unit.root,
            control_dir=sealed_unit.control,
            ref=sealed_unit.ref,
            receipt_sha256=receipt_sha256,
        )
    assert sealed_unit.original_bytes()


def test_unit_lease_namespace_has_bounded_stable_stripes(task_scratch):
    from silent_cascade.archive.transport import unit_reader_lease
    from silent_cascade.archive.types import UnitRef

    control = task_scratch / "striped-locks"
    for value in range(256):
        unit_id = f"{value:064x}"
        ref = UnitRef(unit_id, "diagnostic", "unit", 1, 1, f"units/{unit_id}/manifest.json")
        with unit_reader_lease(control_dir=control, ref=ref):
            pass
    lock_files = tuple((control / "locks/unit-stripes").glob("*.lock"))
    assert 1 <= len(lock_files) <= 64
    assert len(tuple((control / "locks").rglob("*.lock"))) <= 65


def test_interrupted_eviction_resumes_only_recorded_members(sealed_unit, transport, monkeypatch):
    from silent_cascade.archive import transport as module

    receipt_sha256 = _archive(sealed_unit, transport)
    original = module._unlink_recorded_member
    calls = 0

    def interrupt_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 1:
            raise OSError("injected eviction interruption")
        return result

    monkeypatch.setattr(module, "_unlink_recorded_member", interrupt_once)
    with pytest.raises(OSError, match="interruption"):
        module.evict_unit(
            run_dir=sealed_unit.root,
            control_dir=sealed_unit.control,
            ref=sealed_unit.ref,
            receipt_sha256=receipt_sha256,
        )
    monkeypatch.setattr(module, "_unlink_recorded_member", original)
    module.evict_unit(
        run_dir=sealed_unit.root,
        control_dir=sealed_unit.control,
        ref=sealed_unit.ref,
        receipt_sha256=receipt_sha256,
    )
    assert all(not (sealed_unit.root / path).exists() for path in sealed_unit.paths)


def test_control_snapshot_sources_are_pinned(task_scratch, tiny_archive_policy, archive_identity):
    from silent_cascade.archive.catalog import seal_unit
    from silent_cascade.archive.transport import evict_unit

    run = task_scratch / "snapshot-run"
    run.mkdir()
    (run / "state.json").write_bytes(b"state")
    identity = {
        **archive_identity,
        "checkpoint_sha256": "f" * 64,
        "checkpoint_committed": True,
    }
    ref = seal_unit(
        run_dir=run,
        control_dir=task_scratch / "snapshot-control",
        logical_root=".",
        paths=("state.json",),
        kind="control_snapshot",
        identity=identity,
        policy=tiny_archive_policy,
    )
    with pytest.raises(ValueError, match="control snapshot"):
        evict_unit(
            run_dir=run,
            control_dir=task_scratch / "snapshot-control",
            ref=ref,
            receipt_sha256="0" * 64,
        )
    assert (run / "state.json").read_bytes() == b"state"


def _write_fake_aws(path: Path, body: str) -> None:
    path.write_text("import os, sys\n" + body)


def test_r2_cli_uses_closed_argv_and_scrubbed_environment(task_scratch, monkeypatch):
    from silent_cascade.archive import transport as module

    capture = task_scratch / "capture.json"
    script = task_scratch / "fake_aws.py"
    _write_fake_aws(
        script,
        "import json\n"
        "json.dump({'argv': sys.argv[1:], 'env': {k: v for k, v in os.environ.items() "
        "if k.startswith('AWS_')}}, open(os.environ['CAPTURE_PATH'], 'w'))\n",
    )
    monkeypatch.setattr(module, "_AWS_COMMAND", (sys.executable, str(script)))
    monkeypatch.setenv("CAPTURE_PATH", str(capture))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "forbidden")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "forbidden")
    monkeypatch.setenv("AWS_ENDPOINT_URL", "https://attacker.invalid")
    monkeypatch.setenv("AWS_DEBUG", "true")
    source = task_scratch / "source"
    source.write_bytes(b"payload")
    client = module.R2CliTransport(
        profile="silent-cascade-r2", bucket="silent-cascade", prefix="runs/archive"
    )
    client.create("objects/" + "1" * 64 + ".bin", source)
    recorded = json.loads(capture.read_text())
    assert recorded["argv"][:2] == ["s3api", "put-object"]
    assert "--if-none-match" in recorded["argv"]
    assert "*" in recorded["argv"]
    assert "--debug" not in recorded["argv"]
    assert recorded["env"] == {"AWS_CLI_AUTO_PROMPT": "off", "AWS_PAGER": ""}


def test_r2_download_enforces_child_write_limit_when_range_is_ignored(task_scratch, monkeypatch):
    from silent_cascade.archive import transport as module

    script = task_scratch / "oversized_aws.py"
    _write_fake_aws(
        script,
        "output = sys.argv[-1]\n"
        "descriptor = int(output.rsplit('/', 1)[1])\n"
        "os.write(descriptor, b'x' * 4096)\n"
        "print('{\"ContentLength\":4096}')\n",
    )
    monkeypatch.setattr(module, "_AWS_COMMAND", (sys.executable, str(script)))
    destination = task_scratch / "bounded.download"
    client = module.R2CliTransport(
        profile="silent-cascade-r2", bucket="silent-cascade", prefix="runs/archive"
    )
    with pytest.raises(ValueError, match=r"bound|range|size"):
        client.download("objects/" + "2" * 64 + ".bin", destination, max_bytes=32)
    assert not destination.exists()


def test_r2_download_rejects_destination_inode_replacement(task_scratch, monkeypatch):
    from silent_cascade.archive import transport as module

    script = task_scratch / "swap_aws.py"
    _write_fake_aws(
        script,
        "output = sys.argv[-1]\n"
        "os.unlink(os.environ['DESTINATION_PATH'])\n"
        "open(os.environ['DESTINATION_PATH'], 'wb').write(b'replacement')\n"
        "os.write(int(output.rsplit('/', 1)[1]), b'abc')\n"
        'print(\'{"ContentLength":3,"ContentRange":"bytes 0-2/3"}\')\n',
    )
    monkeypatch.setattr(module, "_AWS_COMMAND", (sys.executable, str(script)))
    destination = task_scratch / "replaced.download"
    monkeypatch.setenv("DESTINATION_PATH", str(destination))
    client = module.R2CliTransport(
        profile="silent-cascade-r2", bucket="silent-cascade", prefix="runs/archive"
    )
    with pytest.raises(ValueError, match="ownership"):
        client.download("objects/" + "4" * 64 + ".bin", destination, max_bytes=8)


def test_r2_download_replacement_symlink_cannot_modify_victim(task_scratch, monkeypatch):
    from silent_cascade.archive import transport as module

    victim = task_scratch / "victim"
    victim.write_bytes(b"untouched")
    script = task_scratch / "symlink_swap_aws.py"
    _write_fake_aws(
        script,
        "output = sys.argv[-1]\n"
        "os.unlink(os.environ['DESTINATION_PATH'])\n"
        "os.symlink(os.environ['VICTIM_PATH'], os.environ['DESTINATION_PATH'])\n"
        "open(output, 'wb').write(b'BAD')\n"
        'print(\'{"ContentLength":3,"ContentRange":"bytes 0-2/3"}\')\n',
    )
    monkeypatch.setattr(module, "_AWS_COMMAND", (sys.executable, str(script)))
    monkeypatch.setenv("VICTIM_PATH", str(victim))
    destination = task_scratch / "symlink-replaced.download"
    monkeypatch.setenv("DESTINATION_PATH", str(destination))
    client = module.R2CliTransport(
        profile="silent-cascade-r2", bucket="silent-cascade", prefix="runs/archive"
    )
    with pytest.raises((OSError, ValueError)):
        client.download("objects/" + "5" * 64 + ".bin", destination, max_bytes=8)
    assert victim.read_bytes() == b"untouched"


def test_r2_download_native_dev_fd_open_stays_on_displaced_inode(task_scratch, monkeypatch):
    from silent_cascade.archive import transport as module

    victim = task_scratch / "native-victim"
    victim.write_bytes(b"untouched")
    marker = task_scratch / "dev-fd-opened"
    script = task_scratch / "native_fd_aws.py"
    _write_fake_aws(
        script,
        "output = sys.argv[-1]\n"
        "os.unlink(os.environ['DESTINATION_PATH'])\n"
        "os.symlink(os.environ['VICTIM_PATH'], os.environ['DESTINATION_PATH'])\n"
        "open(output, 'wb').write(b'PINNED')\n"
        "open(os.environ['OPENED_MARKER'], 'xb').close()\n"
        'print(\'{"ContentLength":6,"ContentRange":"bytes 0-5/6"}\')\n',
    )
    monkeypatch.setattr(module, "_AWS_COMMAND", (sys.executable, str(script)))
    monkeypatch.setenv("VICTIM_PATH", str(victim))
    monkeypatch.setenv("OPENED_MARKER", str(marker))
    destination = task_scratch / "native-fd.download"
    monkeypatch.setenv("DESTINATION_PATH", str(destination))
    client = module.R2CliTransport(
        profile="silent-cascade-r2", bucket="silent-cascade", prefix="runs/archive"
    )
    with pytest.raises((OSError, ValueError)):
        client.download("objects/" + "6" * 64 + ".bin", destination, max_bytes=8)
    if not marker.exists():
        pytest.skip("managed sandbox denies child /dev/fd reopen")
    assert victim.read_bytes() == b"untouched"


def test_fresh_cleanup_does_not_unlink_replacement_occupant(task_scratch):
    from silent_cascade.archive import transport as module

    path = task_scratch / "fresh"
    path.write_bytes(b"owned")
    identity = module._regular_identity(path)
    path.unlink()
    path.write_bytes(b"replacement")
    module._remove_fresh(path, expected_identity=identity)
    assert path.read_bytes() == b"replacement"


def test_bounded_process_timeout_survives_early_pipe_close(task_scratch, monkeypatch):
    from silent_cascade.archive import transport as module

    script = task_scratch / "closed_pipes.py"
    _write_fake_aws(
        script,
        "os.close(1)\nos.close(2)\nimport time\ntime.sleep(0.3)\n",
    )
    monkeypatch.setattr(module, "_AWS_TIMEOUT_SECONDS", 0.05)
    started = time.monotonic()
    with pytest.raises(module._ProcessTimeout, match="finite timeout"):
        module._bounded_process([sys.executable, str(script)])
    assert time.monotonic() - started < 0.2


def test_r2_errors_redact_stderr_and_do_not_retry_credentials(task_scratch, monkeypatch):
    from silent_cascade.archive import transport as module

    counter = task_scratch / "count"
    script = task_scratch / "denied_aws.py"
    _write_fake_aws(
        script,
        "from pathlib import Path\n"
        "counter = Path(os.environ['COUNT_PATH'])\n"
        "counter.write_text(str(int(counter.read_text()) + 1) if counter.exists() else '1')\n"
        "sys.stderr.write('ExpiredToken super-secret-value')\n"
        "raise SystemExit(1)\n",
    )
    monkeypatch.setattr(module, "_AWS_COMMAND", (sys.executable, str(script)))
    monkeypatch.setenv("COUNT_PATH", str(counter))
    source = task_scratch / "source"
    source.write_bytes(b"payload")
    client = module.R2CliTransport(
        profile="silent-cascade-r2", bucket="silent-cascade", prefix="runs/archive"
    )
    with pytest.raises(PermissionError) as raised:
        client.create("objects/" + "3" * 64 + ".bin", source)
    assert "super-secret-value" not in str(raised.value)
    assert counter.read_text() == "1"
