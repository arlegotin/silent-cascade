"""Frozen engineering custody must consume the existing global allowance."""

import hashlib
import json
import os
import subprocess

import pytest

from silent_cascade.archive import ledger
from silent_cascade.archive.types import ArchivePolicy


def source_commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def bootstrap(root, *, policy=None):
    from silent_cascade.archive import preflight

    return preflight.bootstrap_engineering_workspace(
        custody_root=root,
        workspace=root / "operational",
        policy=policy or ArchivePolicy(),
        source_commit=source_commit(),
    )


def test_retained_history_is_charged_outside_seven_categories(tmp_path):
    old = tmp_path / "tmp/task-1"
    old.mkdir(parents=True)
    payload = old / "history.bin"
    payload.write_bytes(b"retained engineering history")
    budget = bootstrap(tmp_path)
    assert set(budget.measure()) == {
        "spool",
        "cache",
        "pinned",
        "metadata",
        "scratch",
        "logs",
        "emergency",
    }
    assert budget.retained_charge() >= payload.stat().st_blocks * 512
    assert budget.measure()["scratch"] == 0
    # Spend all normal category capacity: retained history must now block it.
    with (
        pytest.raises(ledger.StorageBlocked, match="normal"),
        budget.reserve(
            spool=budget.policy.spool_bytes,
            cache=budget.policy.cache_bytes,
            pinned=budget.policy.pinned_bytes,
            scratch=budget.policy.scratch_bytes,
            logs=budget.policy.logs_bytes,
            metadata=budget.policy.metadata_bytes - budget.measure()["metadata"],
        ),
    ):
        pytest.fail("retained output granted extra allowance")


@pytest.mark.parametrize("mutation", ["edit", "add", "swap", "missing_page"])
def test_frozen_custody_or_inventory_change_blocks_new_admission(tmp_path, mutation):
    old = tmp_path / "tmp/task-1"
    old.mkdir(parents=True)
    payload = old / "history"
    payload.write_bytes(b"original")
    budget = bootstrap(tmp_path)
    if mutation == "edit":
        payload.write_bytes(b"modified")
    elif mutation == "add":
        (old / "extra").write_bytes(b"new")
    elif mutation == "swap":
        old.rename(old.with_name("displaced"))
        old.mkdir()
        (old / "history").write_bytes(b"original")
    else:
        page = next((budget.root / "retained").glob("*.json"))
        page.rename(page.with_suffix(".missing"))
    with pytest.raises(ledger.StorageBlocked), budget.reserve(scratch=1):
        pytest.fail("stale custody authorized new output")


def test_authority_cannot_be_rebound_to_a_new_operational_allowance(tmp_path):
    from silent_cascade.archive import preflight

    budget = bootstrap(tmp_path)
    with pytest.raises((ledger.StorageBlocked, FileExistsError)):
        preflight.bootstrap_engineering_workspace(
            custody_root=tmp_path,
            workspace=tmp_path / "another",
            policy=budget.policy,
            source_commit=source_commit(),
        )
    with pytest.raises(ledger.StorageBlocked):
        ledger.initialize_workspace_ledger(
            workspace_root=tmp_path / "fixture", policy=budget.policy, baseline=()
        )
    assert not (tmp_path / "another").exists()


def test_unsupported_entries_remain_charged_and_external_alias_blocks(tmp_path):
    old = tmp_path / "tmp/task-1"
    old.mkdir(parents=True)
    (old / "sparse").touch()
    with (old / "sparse").open("r+b") as stream:
        stream.truncate(1024 * 1024)
    (old / "symlink").symlink_to("sparse")
    os.mkfifo(old / "fifo")
    budget = bootstrap(tmp_path)
    assert budget.retained_charge() >= sum(p.lstat().st_blocks * 512 for p in old.iterdir())
    outside = tmp_path.parent / (tmp_path.name + "-external")
    outside.write_bytes(b"outside")
    os.link(outside, old / "alias")
    with pytest.raises(ledger.StorageBlocked):
        budget.check()


def test_compatible_attach_preserves_baseline_reservation_and_remote_bytes(tmp_path):
    from silent_cascade.archive import preflight
    from silent_cascade.archive.types import FileEntry
    from silent_cascade.hashing import sha256_bytes

    workspace = tmp_path / "operational"
    workspace.mkdir()
    (workspace / "original").write_bytes(b"science")
    policy = ArchivePolicy()
    ledger.initialize_workspace_ledger(
        workspace_root=workspace,
        policy=policy,
        baseline=(FileEntry("original", sha256_bytes(b"science"), 7),),
    )
    budget = ledger._StorageBudget(workspace=workspace, policy=policy)
    from silent_cascade.archive.transport import initialize_remote_reservations

    initialize_remote_reservations(
        control_dir=workspace / "control",
        transport_id="prior-shared-provider",
        accounted_bytes=12345,
        accounting_evidence_sha256="c" * 64,
        policy=policy,
    )
    remote = workspace / "control/remote-reservations/state.json"
    remote_before = remote.read_bytes()
    with budget.reserve(scratch=4096):
        before = budget._state()
        preflight.bootstrap_engineering_workspace(
            custody_root=tmp_path, workspace=workspace, policy=policy, source_commit=source_commit()
        )
        after = budget._state()
        assert after["baseline"] == before["baseline"]
        assert after["reservations"] == before["reservations"]
        assert after["paths"] == before["paths"]
        assert remote.read_bytes() == remote_before
        budget.check()


def test_bootstrap_refuses_insufficient_physical_space_before_output(tmp_path, monkeypatch):
    values = list(os.statvfs(tmp_path))
    values[4] = 0
    monkeypatch.setattr(os, "statvfs", lambda _: os.statvfs_result(values))
    with pytest.raises(ledger.StorageBlocked):
        bootstrap(tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("existing", [False, True])
def test_bootstrap_metadata_peak_refusal_does_not_publish_anything(tmp_path, existing):
    policy = ArchivePolicy(metadata_bytes=16384 if existing else 4096, page_bytes=4096)
    workspace = tmp_path / "operational"
    if existing:
        ledger.initialize_workspace_ledger(workspace_root=workspace, policy=policy, baseline=())
        (workspace / ".silent-cascade-storage/prior-output").write_bytes(b"x" * 8192)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ledger.StorageBlocked, match="metadata"):
        bootstrap(tmp_path, policy=policy)
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    assert not (tmp_path / ".silent-cascade-engineering.json").exists()
    assert not (workspace / ".silent-cascade-storage/retained").exists()
    if not existing:
        assert not workspace.exists()


def test_bootstrap_cannot_spend_an_existing_metadata_owners_reservation(tmp_path):
    policy = ArchivePolicy()
    workspace = tmp_path / "operational"
    ledger.initialize_workspace_ledger(workspace_root=workspace, policy=policy, baseline=())
    budget = ledger._StorageBudget(workspace=workspace, policy=policy)
    with budget.reserve(metadata=4096):
        before = (budget.root / "workspace.json").read_bytes()
        with pytest.raises(ledger.StorageBlocked, match=r"metadata.*owned"):
            bootstrap(tmp_path, policy=policy)
        assert (budget.root / "workspace.json").read_bytes() == before
        assert not (tmp_path / ".silent-cascade-engineering.json").exists()
        assert not (budget.root / "retained").exists()


def test_linked_inventory_preserves_reverse_rows_with_two_bounded_traversals(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight

    for group in range(3):
        original = tmp_path / f"{group}-a"
        original.write_bytes(bytes([group]) * 17)
        os.link(original, tmp_path / f"{group}-b")
        os.link(original, tmp_path / f"{group}-c")
    inventory = preflight._inventory
    traversals = []

    def counted(*args, **kwargs):
        traversals.append(kwargs)
        yield from inventory(*args, **kwargs)

    monkeypatch.setattr(preflight, "_inventory", counted)
    forward = list(preflight._accounted_inventory(tmp_path, tmp_path / "operational"))
    assert len(traversals) <= 2
    traversals.clear()
    reverse = list(preflight._accounted_inventory(tmp_path, tmp_path / "operational", reverse=True))
    assert len(traversals) <= 2
    assert forward == list(reversed(reverse))
    assert sum(row["allocated"] for row in forward) == (
        tmp_path.stat().st_blocks * 512 + 3 * (tmp_path / "0-a").stat().st_blocks * 512
    )
    assert [row["path"] for row in forward if row["path"] != "." and row["allocated"]] == [
        "0-a",
        "1-a",
        "2-a",
    ]
    assert all(row["sha256"] is None for row in forward)


@pytest.mark.parametrize(
    "fault", ["external", "over_capacity", "changed_identity", "invalid_links"]
)
def test_linked_inventory_fails_closed_on_ambiguous_custody(tmp_path, monkeypatch, fault):
    from silent_cascade.archive import preflight

    for group in range(2):
        (tmp_path / f"{group}-a").write_bytes(b"linked")
        os.link(tmp_path / f"{group}-a", tmp_path / f"{group}-b")
    if fault == "external":
        os.link(tmp_path / "0-a", tmp_path.parent / (tmp_path.name + "-outside"))
    elif fault == "over_capacity":
        monkeypatch.setattr(preflight, "_LINKED_INODES_LIMIT", 1, raising=False)
    else:
        inventory = preflight._inventory

        def corrupted(*args, **kwargs):
            for row in inventory(*args, **kwargs):
                if row["path"] == "0-b":
                    row["inode" if fault == "changed_identity" else "links"] = 0
                yield row

        monkeypatch.setattr(preflight, "_inventory", corrupted)
    with pytest.raises(ledger.StorageBlocked):
        list(preflight._accounted_inventory(tmp_path, tmp_path / "operational"))


@pytest.mark.parametrize(
    ("page_bytes", "page_entries", "groups"),
    [(95, 1000, [8, 1, 1, 1, 1]), (103, 1000, [9, 2, 1]), (103, 2, [2] * 6)],
)
def test_inventory_pages_preserve_exact_byte_and_entry_boundaries(page_bytes, page_entries, groups):
    from silent_cascade.archive import preflight

    # All rows deliberately have identical seven-byte encodings.
    records = [{"n": index % 10} for index in range(12)]
    pages = list(
        preflight._pages(records, ArchivePolicy(page_bytes=page_bytes, page_entries=page_entries))
    )
    assert [len(page[2]) for page in pages] == groups
    offset, previous = 0, None
    for digest, raw, entries in pages:
        expected_entries = records[offset : offset + len(entries)]
        expected = json.dumps(
            {"entries": expected_entries, "next": previous},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        assert raw == expected
        assert digest == hashlib.sha256(expected).hexdigest()
        assert entries == expected_entries
        previous, offset = digest, offset + len(entries)


def test_inventory_pages_escape_rows_and_refuse_oversize_after_head_changes():
    from silent_cascade.archive import preflight

    records = [{"x": 'é"\\\n'}, {"x": "another"}]
    pages = list(preflight._pages(records, ArchivePolicy(page_entries=1, page_bytes=256)))
    assert pages[0][1] == b'{"entries":[{"x":"\xc3\xa9\\"\\\\\\n"}],"next":null}'
    assert pages[1][1] == (b'{"entries":[{"x":"another"}],"next":"' + pages[0][0].encode() + b'"}')
    assert list(preflight._pages([], ArchivePolicy())) == []
    with pytest.raises(ledger.StorageBlocked):
        list(preflight._pages([{"n": 0}], ArchivePolicy(page_bytes=32)))
    with pytest.raises(ledger.StorageBlocked):
        list(preflight._pages([{"n": 0}, {"n": 1}], ArchivePolicy(page_bytes=33)))


def test_inventory_page_serialization_work_is_linear_in_rows(monkeypatch):
    from silent_cascade.archive import preflight

    original = preflight.canonical_json_bytes
    work = 0

    def measured(value):
        nonlocal work
        work += len(value["entries"]) if "entries" in value else 1
        return original(value)

    monkeypatch.setattr(preflight, "canonical_json_bytes", measured)
    records = [{"n": index} for index in range(512)]
    pages = list(preflight._pages(records, ArchivePolicy(page_entries=128, page_bytes=4096)))
    assert sum(len(page[2]) for page in pages) == 512
    assert work <= 3 * len(records)


def test_bootstrap_admits_allocation_blocks_not_preferred_io_size(tmp_path, monkeypatch):
    values = list(os.statvfs(tmp_path))
    values[0], values[1] = 1024**2, 4096
    monkeypatch.setattr(os, "statvfs", lambda _: os.statvfs_result(values))
    budget = bootstrap(tmp_path, policy=ArchivePolicy(metadata_bytes=1024**2, page_bytes=4096))
    assert budget.check()["metadata"] < 65536


def test_prelude_bounds_ignore_io_hint_but_preserve_allocation_geometry(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight

    budget, _transport, candidate, review = prepared_candidate(tmp_path)
    selected = tuple(member for member in candidate.members if member.path in review.paths)
    values = list(os.statvfs(tmp_path))
    values[0], values[1] = 1024**2, 4096
    monkeypatch.setattr(os, "statvfs", lambda _: os.statvfs_result(values))
    large_io = preflight._prelude_bounds(budget, candidate, selected)
    values[0] = 4096
    small_io = preflight._prelude_bounds(budget, candidate, selected)
    assert large_io == small_io
    values[1] = 8192
    larger_allocation = preflight._prelude_bounds(budget, candidate, selected)
    assert all(larger_allocation[name] > large_io[name] for name in ("metadata", "scratch"))


@pytest.mark.parametrize("fragment", [0, -4096, 1000])
def test_bootstrap_rejects_unusable_allocation_geometry_without_output(
    tmp_path, monkeypatch, fragment
):
    values = list(os.statvfs(tmp_path))
    values[1] = fragment
    monkeypatch.setattr(os, "statvfs", lambda _: os.statvfs_result(values))
    with pytest.raises(ledger.StorageBlocked, match="allocation geometry"):
        bootstrap(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_actual_atomic_archive_peaks_fit_derived_bounds_with_shared_history(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight

    budget, transport, candidate, review = prepared_candidate(tmp_path, extra=True)
    fsync = os.fsync
    observed, before = {}, {}

    def measured_sync(descriptor):
        fsync(descriptor)
        # At file fsync the staged file is allocated; at directory fsync the
        # publication is complete. Observe real physical categories in both.
        measured = budget.measure()
        for category in ("metadata", "scratch"):
            observed[category] = max(observed[category], measured[category] - before[category])

    for index in range(2):
        if index:
            remote = json.loads(
                (budget.workspace / "control/remote-reservations/state.json").read_bytes()
            )
            assert remote["operational_head"]["entry_count"] > 0
            candidate = next(preflight.iter_engineering_candidates(budget))
            review = preflight.EngineeringContentReview(
                candidate_id=candidate.candidate_id,
                executable_sha256=candidate.executable_sha256,
                paths=("tmp/task-1/safe/second.bin",),
                producer_evidence_sha256="c" * 64,
            )
        selected = tuple(member for member in candidate.members if member.path in review.paths)
        bound = preflight._prelude_bounds(budget, candidate, selected)
        before, observed = budget.measure(), {"metadata": 0, "scratch": 0}
        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", measured_sync)
            preflight.archive_engineering_candidate(
                budget=budget, candidate=candidate, review=review, transport=transport
            )
        assert 0 < observed["metadata"] <= bound["metadata"]
        assert 0 < observed["scratch"] <= bound["scratch"]
        assert not (tmp_path / review.paths[0]).exists()


def prepared_candidate(root, *, policy=None, extra=False):
    from conftest import DirectoryTransport

    from silent_cascade.archive import preflight
    from silent_cascade.archive.transport import initialize_remote_reservations

    old = root / "tmp/task-1/safe"
    old.mkdir(parents=True)
    (old / "tensor.bin").write_bytes(b"malformed science bytes remain exact")
    (old / "owner.json").write_bytes(b'{"workspace":"/private/not-for-upload"}')
    if extra:
        (old / "second.bin").write_bytes(b"second independent engineering payload")
    mixed = root / "tmp/task-1/mixed"
    mixed.mkdir()
    (mixed / "symlink").symlink_to(old)
    budget = bootstrap(root, policy=policy)
    transport = DirectoryTransport(budget.workspace / "remote-double")
    budget.bind(transport.root, category="scratch")
    initialize_remote_reservations(
        control_dir=budget.workspace / "control",
        transport_id=transport.transport_id,
        accounted_bytes=0,
        accounting_evidence_sha256="a" * 64,
        policy=budget.policy,
    )
    budget.bind(budget.workspace / "control", category="metadata")
    budget.bind(budget.workspace / "control/transfer-scratch", category="scratch")
    candidates = tuple(preflight.iter_engineering_candidates(budget))
    assert len(candidates) == 1
    candidate = candidates[0]
    review = preflight.EngineeringContentReview(
        candidate_id=candidate.candidate_id,
        executable_sha256=candidate.executable_sha256,
        paths=("tmp/task-1/safe/tensor.bin",),
        producer_evidence_sha256="b" * 64,
    )
    return budget, transport, candidate, review


def test_safe_child_archives_exact_reviewed_bytes_and_keeps_private_sibling(tmp_path):
    from silent_cascade.archive import preflight
    from silent_cascade.archive.bundles import restore_unit

    budget, transport, candidate, review = prepared_candidate(tmp_path)
    before = budget.retained_charge()
    original = (tmp_path / review.paths[0]).read_bytes()
    private = tmp_path / "tmp/task-1/safe/owner.json"
    private_bytes = private.read_bytes()
    result = preflight.archive_engineering_candidate(
        budget=budget, candidate=candidate, review=review, transport=transport
    )
    assert not (tmp_path / review.paths[0]).exists()
    assert private.read_bytes() == private_bytes
    assert budget.retained_charge() < before
    assert (tmp_path / "tmp/task-1/mixed/symlink").is_symlink()
    from silent_cascade.archive.catalog import iter_unit_files

    files = tuple(iter_unit_files(budget.workspace / "control", result.ref))
    assert tuple(file.path for file in files) == review.paths
    # The reviewed restore API must restore the original opaque bytes.
    from silent_cascade.archive.transport import _object_key

    def fetch(descriptor):
        path = budget.workspace / "scratch" / f"restore-{descriptor.index}"
        path.parent.mkdir(exist_ok=True)
        transport.download(
            _object_key(result.run_id, f"objects/{descriptor.sha256}.bin"),
            path,
            max_bytes=descriptor.bytes,
        )
        return path

    from silent_cascade.archive.catalog import _opened_unit

    with _opened_unit(budget.workspace / "control", result.ref) as (_unit, manifest):
        chunks = tuple(manifest.chunks)
    restore_unit(
        control_dir=budget.workspace / "control",
        ref=result.ref,
        destination=budget.workspace / "restored",
        chunks=(fetch(chunk) for chunk in chunks),
        policy=budget.policy,
    )
    assert (budget.workspace / "restored" / review.paths[0]).read_bytes() == original


@pytest.mark.parametrize("fault", ["upload", "readback", "collision", "receipt"])
def test_transfer_failure_preserves_original_and_retained_charge(tmp_path, fault):
    from silent_cascade.archive import preflight

    budget, transport, candidate, review = prepared_candidate(tmp_path)
    before = budget.retained_charge()
    if fault == "upload":
        transport.fail_create_call = 1
    elif fault == "readback":
        transport.corrupt_downloads = True
    elif fault == "collision":
        transport.collision_create_call = 1
    else:
        transport.fail_key_contains = "operational/receipts/"
    with pytest.raises((OSError, ValueError)):
        preflight.archive_engineering_candidate(
            budget=budget, candidate=candidate, review=review, transport=transport
        )
    assert transport.create_calls > 0
    assert (tmp_path / review.paths[0]).exists()
    assert budget.retained_charge() == before


@pytest.mark.parametrize("fault", ["missing", "inventory", "source", "paths"])
def test_upload_requires_exact_explicit_private_content_review(tmp_path, fault):
    from silent_cascade.archive import preflight

    budget, transport, candidate, review = prepared_candidate(tmp_path)
    if fault == "missing":
        review = None
    elif fault == "inventory":
        review = review.model_copy(update={"candidate_id": "f" * 64})
    elif fault == "source":
        review = review.model_copy(update={"executable_sha256": "f" * 64})
    else:
        review = review.model_copy(update={"paths": ("tmp/task-1/outside.bin",)})
    with pytest.raises((ledger.StorageBlocked, ValueError)):
        preflight.archive_engineering_candidate(
            budget=budget, candidate=candidate, review=review, transport=transport
        )
    assert not any(path.is_file() for path in transport.root.rglob("*"))


def test_interrupted_unlink_keeps_charge_and_resumes_without_double_free(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight
    from silent_cascade.archive import transport as transfer

    budget, transport, candidate, review = prepared_candidate(tmp_path)
    before = budget.retained_charge()
    original = transfer._write_intent

    def fail_completion(control_dir, ref, intent, *, replace):
        if intent["completed"]:
            raise OSError("interrupted after unlink")
        return original(control_dir, ref, intent, replace=replace)

    monkeypatch.setattr(transfer, "_write_intent", fail_completion)
    with pytest.raises(OSError, match="interrupted"):
        preflight.archive_engineering_candidate(
            budget=budget, candidate=candidate, review=review, transport=transport
        )
    assert not (tmp_path / review.paths[0]).exists()
    assert budget.retained_charge() == before
    monkeypatch.setattr(transfer, "_write_intent", original)
    restarted = ledger._StorageBudget(workspace=budget.workspace, policy=budget.policy)
    preflight.resume_engineering_eviction(budget=restarted, transport=transport)
    assert restarted.retained_charge() < before
    assert "pending" not in restarted._state()["engineering"]


def test_bootstrap_accounts_prior_unconsumed_reservations_before_any_new_output(
    tmp_path, monkeypatch
):
    from silent_cascade.archive import preflight

    workspace = tmp_path / "operational"
    policy = ArchivePolicy()
    ledger.initialize_workspace_ledger(workspace_root=workspace, policy=policy, baseline=())
    budget = ledger._StorageBudget(workspace=workspace, policy=policy)
    volume = list(os.statvfs(tmp_path))
    volume[4] = 10 * 1024**3 // volume[1]
    monkeypatch.setattr(os, "statvfs", lambda _: os.statvfs_result(volume))
    with budget.reserve(
        spool=policy.spool_bytes,
        cache=policy.cache_bytes,
        pinned=policy.pinned_bytes,
        scratch=policy.scratch_bytes,
        logs=policy.logs_bytes,
        metadata=policy.metadata_bytes - budget.measure()["metadata"],
    ):
        before = tuple(tmp_path.iterdir())
        with pytest.raises(ledger.StorageBlocked):
            preflight.bootstrap_engineering_workspace(
                custody_root=tmp_path,
                workspace=workspace,
                policy=policy,
                source_commit=source_commit(),
            )
        assert tuple(tmp_path.iterdir()) == before
        assert budget._state()["schema_version"] == "phase4-r2-workspace-v1"


def test_eviction_resume_derives_capacity_instead_of_trusting_stored_byte_grant(
    tmp_path, monkeypatch
):
    from silent_cascade.archive import preflight
    from silent_cascade.archive import transport as transfer

    budget, transport, candidate, review = prepared_candidate(tmp_path)
    original = transfer._write_intent

    def fail_completion(control_dir, ref, intent, *, replace):
        if intent["completed"]:
            raise OSError("interrupted")
        return original(control_dir, ref, intent, replace=replace)

    monkeypatch.setattr(transfer, "_write_intent", fail_completion)
    with pytest.raises(OSError):
        preflight.archive_engineering_candidate(
            budget=budget, candidate=candidate, review=review, transport=transport
        )
    monkeypatch.setattr(transfer, "_write_intent", original)
    state = budget._state()
    state["engineering"]["pending"]["bounds"] = {"metadata": 0, "scratch": 0}
    budget._store(state)
    preflight.resume_engineering_eviction(budget=budget, transport=transport)
    assert "pending" not in budget._state()["engineering"]


def test_changed_transport_cannot_use_shared_history_allowance(tmp_path):
    from conftest import DirectoryTransport

    from silent_cascade.archive import preflight

    budget, _transport, candidate, review = prepared_candidate(tmp_path)
    other = DirectoryTransport(budget.workspace / "other-remote")
    before = budget.retained_charge()
    with pytest.raises(ValueError, match="transport"):
        preflight.archive_engineering_candidate(
            budget=budget, candidate=candidate, review=review, transport=other
        )
    assert budget.retained_charge() == before
    assert other.create_calls == 0


def test_source_changes_invalidate_review_before_output(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight

    budget, transport, candidate, review = prepared_candidate(tmp_path)
    monkeypatch.setattr(preflight, "_executable_digest", lambda: "f" * 64)
    with pytest.raises(ValueError, match="source"):
        preflight.archive_engineering_candidate(
            budget=budget, candidate=candidate, review=review, transport=transport
        )
    assert transport.create_calls == 0


def test_invented_source_revision_refused_before_bootstrap_output(tmp_path):
    from silent_cascade.archive import preflight

    with pytest.raises(ledger.StorageBlocked, match="source"):
        preflight.bootstrap_engineering_workspace(
            custody_root=tmp_path,
            workspace=tmp_path / "operational",
            policy=ArchivePolicy(),
            source_commit="0" * 40,
        )
    assert list(tmp_path.iterdir()) == []


def test_linked_sparse_and_special_candidates_stay_local_with_single_inode_charge(tmp_path):
    from silent_cascade.archive import preflight

    old = tmp_path / "tmp/task-1"
    old.mkdir(parents=True)
    (old / "safe").mkdir()
    (old / "safe/data").write_bytes(b"opaque bytes")
    for name in ("links", "sparse", "special"):
        (old / name).mkdir()
    (old / "links/a").write_bytes(b"one physical inode")
    os.link(old / "links/a", old / "links/b")
    with (old / "sparse/large").open("wb") as stream:
        stream.truncate(1024 * 1024)
    os.mkfifo(old / "special/fifo")
    budget = bootstrap(tmp_path)
    candidates = tuple(preflight.iter_engineering_candidates(budget))
    assert tuple(candidate.logical_root for candidate in candidates) == ("tmp/task-1/safe",)
    records = tuple(preflight._stored_records(budget, budget._state()["engineering"]["snapshot"]))
    linked_allocation = sum(
        row["allocated"] for row in records if row["path"].startswith("tmp/task-1/links/")
    )
    assert linked_allocation == (old / "links/a").stat().st_blocks * 512


def test_live_writer_during_inventory_invalidates_admission(tmp_path, monkeypatch):
    old = tmp_path / "tmp/task-1"
    old.mkdir(parents=True)
    payload = old / "data"
    payload.write_bytes(b"before")
    budget = bootstrap(tmp_path)
    original = os.read
    inode = payload.stat().st_ino
    changed = False

    def racing_read(descriptor, size):
        nonlocal changed
        if not changed and os.fstat(descriptor).st_ino == inode:
            changed = True
            payload.write_bytes(b"after")
        return original(descriptor, size)

    monkeypatch.setattr(os, "read", racing_read)
    with pytest.raises(ledger.StorageBlocked):
        budget.check()
    assert changed


def test_fresh_candidate_run_identity_keeps_shared_remote_history(tmp_path):
    import json

    from silent_cascade.archive import preflight

    budget, transport, candidate, review = prepared_candidate(tmp_path, extra=True)
    first = preflight.archive_engineering_candidate(
        budget=budget, candidate=candidate, review=review, transport=transport
    )
    remote_state = budget.workspace / "control/remote-reservations/state.json"
    before = json.loads(remote_state.read_bytes())["reserved_bytes"]
    candidate = next(preflight.iter_engineering_candidates(budget))
    review = preflight.EngineeringContentReview(
        candidate_id=candidate.candidate_id,
        executable_sha256=candidate.executable_sha256,
        paths=("tmp/task-1/safe/second.bin",),
        producer_evidence_sha256="c" * 64,
    )
    second = preflight.archive_engineering_candidate(
        budget=budget, candidate=candidate, review=review, transport=transport
    )
    assert first.run_id != second.run_id
    assert json.loads(remote_state.read_bytes())["reserved_bytes"] > before > 0
    assert (tmp_path / "tmp/task-1/safe/owner.json").exists()


def test_remote_quota_failure_retains_catalog_reservations_and_all_source_bytes(tmp_path):
    import json

    from silent_cascade.archive import preflight

    budget, transport, candidate, review = prepared_candidate(
        tmp_path, policy=ArchivePolicy(remote_bytes=1024)
    )
    before = budget.retained_charge()
    with pytest.raises(ValueError, match="remote byte budget"):
        preflight.archive_engineering_candidate(
            budget=budget, candidate=candidate, review=review, transport=transport
        )
    remote_state = budget.workspace / "control/remote-reservations/state.json"
    assert 0 < json.loads(remote_state.read_bytes())["reserved_bytes"] <= 1024
    assert budget.retained_charge() == before
    assert (tmp_path / review.paths[0]).exists()


def test_multi_page_frozen_inventory_authenticates_every_page(tmp_path):
    old = tmp_path / "tmp/task-1"
    old.mkdir(parents=True)
    for index in range(8):
        (old / f"file-{index}").write_bytes(f"payload-{index}".encode())
    budget = bootstrap(tmp_path, policy=ArchivePolicy(page_entries=2))
    assert budget._state()["engineering"]["snapshot"]["pages"] >= 5
    budget.check()
    (old / "file-3").write_bytes(b"changed middle page")
    with pytest.raises(ledger.StorageBlocked):
        budget.check()


@pytest.mark.parametrize("owner_alive", [False, True])
def test_dead_prelude_owner_resumes_exact_stranded_reservation(tmp_path, monkeypatch, owner_alive):
    import sys

    from silent_cascade.archive import preflight
    from silent_cascade.archive import transport as transfer

    budget, transport, candidate, review = prepared_candidate(tmp_path)
    original = transfer._write_intent
    stranded = {}

    def crash_before_completion(control_dir, ref, intent, *, replace):
        if intent["completed"]:
            stranded.update(budget._state()["reservations"])
            raise OSError("process cut after unlink")
        return original(control_dir, ref, intent, replace=replace)

    monkeypatch.setattr(transfer, "_write_intent", crash_before_completion)
    with pytest.raises(OSError, match="process cut"):
        preflight.archive_engineering_candidate(
            budget=budget, candidate=candidate, review=review, transport=transport
        )
    monkeypatch.setattr(transfer, "_write_intent", original)
    # Restore the exact reservation the exception unwinder released, with a
    # genuinely exited owner's PID/create-time. This models loss of finally.
    with subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"], stdin=subprocess.PIPE
    ) as owner:
        identity = ledger._process_identity(owner.pid)
        assert identity is not None
        owner.stdin.close()
        owner.wait(timeout=10)
    token, record = next(iter(stranded.items()))
    record["pid"], record["create_time"] = owner.pid, identity
    if owner_alive:
        record["pid"], record["create_time"] = os.getpid(), ledger._process_identity(os.getpid())
    state = budget._state()
    state["reservations"][token] = record
    budget._store(state)
    old_charge = budget.retained_charge()
    if owner_alive:
        with pytest.raises(ledger.StorageBlocked, match="live"):
            preflight.resume_engineering_eviction(budget=budget, transport=transport)
        assert budget.retained_charge() == old_charge
        assert budget._state()["reservations"][token] == record
        return
    preflight.resume_engineering_eviction(budget=budget, transport=transport)
    assert budget.retained_charge() < old_charge
    assert budget._state()["reservations"] == {}


def test_source_changed_between_review_and_seal_never_reaches_transport(tmp_path, monkeypatch):
    from silent_cascade.archive import catalog, preflight

    budget, transport, candidate, review = prepared_candidate(tmp_path)
    original = catalog.seal_unit

    def changed_source(**kwargs):
        (tmp_path / review.paths[0]).write_bytes(b"replacement bytes were never content-reviewed")
        return original(**kwargs)

    monkeypatch.setattr(catalog, "seal_unit", changed_source)
    with pytest.raises(ledger.StorageBlocked):
        preflight.archive_engineering_candidate(
            budget=budget, candidate=candidate, review=review, transport=transport
        )
    assert transport.create_calls == 0
    assert (tmp_path / review.paths[0]).exists()


def test_physical_peak_refusal_precedes_sealing_and_upload(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight

    budget, transport, candidate, review = prepared_candidate(tmp_path)
    volume = list(os.statvfs(tmp_path))
    volume[4] = (budget.policy.reserve_bytes + volume[1]) // volume[1]
    monkeypatch.setattr(os, "statvfs", lambda _: os.statvfs_result(volume))
    with pytest.raises(ledger.StorageBlocked, match="headroom"):
        preflight.archive_engineering_candidate(
            budget=budget, candidate=candidate, review=review, transport=transport
        )
    assert not (budget.workspace / "control/units").exists()
    assert transport.create_calls == 0
