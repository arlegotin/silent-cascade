"""Shared physical allocation and durable admission tests."""

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _isolate_unit_ledgers_from_an_outer_workspace_authority(monkeypatch):
    from silent_cascade.archive import preflight

    monkeypatch.setattr(preflight, "_require_single_authority", lambda _: None)
    monkeypatch.setattr(preflight, "_authority", lambda _: None)


def _inode(*, kind, blocks, device=41, links=1, inode=1):
    return SimpleNamespace(
        st_mode=kind | 0o600,
        st_blocks=blocks,
        st_dev=device,
        st_nlink=links,
        st_ino=inode,
    )


def _memory_budget(monkeypatch, *, records, walks, paths):
    from silent_cascade.archive.ledger import _StorageBudget

    workspace = Path("/virtual/workspace")
    budget = object.__new__(_StorageBudget)
    budget.workspace = workspace
    budget.device = 41
    state = {"paths": paths, "baseline": None}
    monkeypatch.setattr(budget, "_state", lambda: state)
    original_lstat = Path.lstat
    original_stat = Path.stat

    def lstat(path, *args, **kwargs):
        path = Path(path)
        if not path.is_relative_to(workspace):
            return original_lstat(path, *args, **kwargs)
        try:
            record = records[path]
        except KeyError:
            pytest.fail(f"unexpected lstat: {path}")
        return record() if callable(record) else record

    def stat_path(path, *args, **kwargs):
        path = Path(path)
        if path.is_relative_to(workspace):
            pytest.fail(f"measurement followed an entry target: {path}")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(Path, "stat", stat_path)
    monkeypatch.setattr(os, "walk", walks)
    return budget


def test_measure_counts_opaque_explicit_scratch_entries_without_following_links(monkeypatch):
    from silent_cascade.archive.ledger import StorageBlocked

    workspace = Path("/virtual/workspace")
    scratch = workspace / "scratch-zone"
    linked_directory = scratch / "linked-directory"
    records = {
        workspace: _inode(kind=stat.S_IFDIR, blocks=1),
        scratch: _inode(kind=stat.S_IFDIR, blocks=2),
        linked_directory: _inode(kind=stat.S_IFLNK, blocks=3),
        scratch / "real-directory": _inode(kind=stat.S_IFDIR, blocks=4),
        scratch / "linked-file": _inode(kind=stat.S_IFLNK, blocks=5),
        scratch / "hard-a": _inode(kind=stat.S_IFREG, blocks=6, links=2, inode=700),
        scratch / "hard-b": _inode(kind=stat.S_IFREG, blocks=6, links=2, inode=700),
    }

    def walks(root, *, followlinks):
        assert Path(root) == workspace
        assert followlinks is False
        root_directories = ["scratch-zone"]
        yield workspace, root_directories, []
        scratch_directories = ["linked-directory", "real-directory"]
        yield scratch, scratch_directories, ["linked-file", "hard-a", "hard-b"]
        if "linked-directory" in scratch_directories:
            pytest.fail("measurement retained a linked directory for traversal")
        yield scratch / "real-directory", [], []

    budget = _memory_budget(
        monkeypatch,
        records=records,
        walks=walks,
        paths={"scratch-zone": "scratch"},
    )

    measured = budget.measure()

    assert measured["scratch"] == 26 * 512
    with pytest.raises(StorageBlocked, match="symlink"):
        budget.require_path(scratch / "linked-file")


@pytest.mark.parametrize(
    ("directory", "paths"),
    [
        ("spool-zone", {"spool-zone": "spool", "scratch-zone": "scratch"}),
        ("unknown-zone", {"scratch-zone": "scratch"}),
    ],
)
def test_measure_rejects_links_outside_explicit_scratch(monkeypatch, directory, paths):
    from silent_cascade.archive.ledger import StorageBlocked

    workspace = Path("/virtual/workspace")
    root = workspace / directory
    linked_file = root / "linked-file"
    records = {
        workspace: _inode(kind=stat.S_IFDIR, blocks=1),
        root: _inode(kind=stat.S_IFDIR, blocks=2),
        linked_file: _inode(kind=stat.S_IFLNK, blocks=3),
    }

    def walks(walk_root, *, followlinks):
        assert Path(walk_root) == workspace
        assert followlinks is False
        yield workspace, [directory], []
        yield root, [], ["linked-file"]

    budget = _memory_budget(
        monkeypatch,
        records=records,
        walks=walks,
        paths=paths,
    )

    with pytest.raises(StorageBlocked, match="symlink"):
        budget.measure()


def test_measure_rejects_second_device_inside_explicit_scratch(monkeypatch):
    from silent_cascade.archive.ledger import StorageBlocked

    workspace = Path("/virtual/workspace")
    scratch = workspace / "scratch-zone"
    other_device = scratch / "other-device"
    records = {
        workspace: _inode(kind=stat.S_IFDIR, blocks=1),
        scratch: _inode(kind=stat.S_IFDIR, blocks=2),
        other_device: _inode(kind=stat.S_IFREG, blocks=3, device=42),
    }

    def walks(root, *, followlinks):
        assert Path(root) == workspace
        assert followlinks is False
        yield workspace, ["scratch-zone"], []
        yield scratch, [], ["other-device"]

    budget = _memory_budget(
        monkeypatch,
        records=records,
        walks=walks,
        paths={"scratch-zone": "scratch"},
    )

    with pytest.raises(StorageBlocked, match="second volume"):
        budget.measure()


def test_measure_rejects_stale_walk_beneath_linked_scratch_ancestor(monkeypatch):
    from silent_cascade.archive.ledger import StorageBlocked

    workspace = Path("/virtual/workspace")
    scratch = workspace / "scratch-zone"
    changed_directory = scratch / "changed-directory"
    ancestor_linked = False
    records = {
        workspace: _inode(kind=stat.S_IFDIR, blocks=1),
        scratch: _inode(kind=stat.S_IFDIR, blocks=2),
        changed_directory: lambda: _inode(
            kind=stat.S_IFLNK if ancestor_linked else stat.S_IFDIR,
            blocks=3,
        ),
    }

    def walks(root, *, followlinks):
        nonlocal ancestor_linked
        assert Path(root) == workspace
        assert followlinks is False
        yield workspace, ["scratch-zone"], []
        yield scratch, ["changed-directory"], []
        ancestor_linked = True
        yield changed_directory, [], ["outside-target"]

    budget = _memory_budget(
        monkeypatch,
        records=records,
        walks=walks,
        paths={"scratch-zone": "scratch"},
    )

    with pytest.raises(StorageBlocked, match="symlink"):
        budget.measure()


def test_storage_admission_counts_physical_scratch_and_outstanding_reservations(tmp_path):
    from silent_cascade.archive.ledger import (
        StorageBlocked,
        _StorageBudget,
        initialize_workspace_ledger,
    )
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    budget = _StorageBudget(workspace=tmp_path, policy=policy)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    payload = scratch / "attempt.bin"
    payload.write_bytes(b"retained failed attempt")
    measured = budget.measure()
    assert measured["scratch"] >= payload.stat().st_blocks * 512
    with (
        budget.reserve(scratch=policy.scratch_bytes - measured["scratch"]),
        pytest.raises(StorageBlocked, match="scratch"),
        budget.reserve(scratch=1),
    ):
        pytest.fail("category reservation bypassed")


def test_explicit_scratch_links_are_opaque_accounting_not_authorized_paths(tmp_path):
    from silent_cascade.archive.ledger import (
        StorageBlocked,
        _StorageBudget,
        initialize_workspace_ledger,
    )
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    external = tmp_path.parent / f"{tmp_path.name}-external-target"
    external.mkdir()
    (external / "unaccounted-target").write_bytes(b"must not be followed")
    linked_directory = scratch / "linked-directory"
    linked_directory.symlink_to(external, target_is_directory=True)
    linked_file = scratch / "linked-file"
    linked_file.symlink_to(external / "unaccounted-target")
    hard_a = scratch / "hard-a"
    hard_a.write_bytes(b"charge each retained name")
    hard_b = scratch / "hard-b"
    os.link(hard_a, hard_b)
    budget = _StorageBudget(workspace=tmp_path, policy=policy)

    measured = budget.measure()

    expected = (
        scratch.lstat().st_blocks
        + linked_directory.lstat().st_blocks
        + linked_file.lstat().st_blocks
        + hard_a.lstat().st_blocks
        + hard_b.lstat().st_blocks
    ) * 512
    assert measured["scratch"] == expected
    with pytest.raises(StorageBlocked, match="symlink"):
        budget.require_path(linked_directory / "unaccounted-target")
    with pytest.raises(StorageBlocked, match="symlink"):
        budget.require_path(linked_file)


def test_non_scratch_descendant_link_remains_rejected(tmp_path):
    from silent_cascade.archive.ledger import (
        StorageBlocked,
        _StorageBudget,
        initialize_workspace_ledger,
    )
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    spool = tmp_path / "spool"
    spool.mkdir()
    (spool / "linked-file").symlink_to(tmp_path.parent / "outside")
    budget = _StorageBudget(workspace=tmp_path, policy=policy)

    with pytest.raises(StorageBlocked, match="symlink"):
        budget.measure()


def test_storage_admission_checks_free_space_and_rejects_second_volume(tmp_path, monkeypatch):
    from silent_cascade.archive.ledger import (
        StorageBlocked,
        _StorageBudget,
        initialize_workspace_ledger,
    )
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    budget = _StorageBudget(workspace=tmp_path, policy=policy)
    original = os.statvfs
    volume = original(tmp_path)
    values = list(volume)
    values[4] = 1
    monkeypatch.setattr(os, "statvfs", lambda _: os.statvfs_result(values))
    with pytest.raises(StorageBlocked, match="headroom"), budget.reserve(scratch=1):
        pytest.fail("physical headroom bypassed")
    monkeypatch.setattr(os, "statvfs", original)
    with pytest.raises(StorageBlocked, match="workspace"):
        budget.require_path(tmp_path.parent / "escape")


def test_storage_admission_rejects_symlink_scratch(tmp_path):
    from silent_cascade.archive.ledger import (
        StorageBlocked,
        _StorageBudget,
        initialize_workspace_ledger,
    )
    from silent_cascade.archive.types import ArchivePolicy

    initialize_workspace_ledger(workspace_root=tmp_path, policy=ArchivePolicy(), baseline=())
    (tmp_path / "scratch").symlink_to(tmp_path.parent, target_is_directory=True)
    budget = _StorageBudget(workspace=tmp_path, policy=ArchivePolicy())
    with pytest.raises(StorageBlocked, match="symlink"):
        budget.measure()


def test_shared_ledger_cannot_reset_baseline_or_duplicate_reservations(tmp_path):
    import hashlib

    from silent_cascade.archive.ledger import (
        StorageBlocked,
        _StorageBudget,
        initialize_workspace_ledger,
    )
    from silent_cascade.archive.types import ArchivePolicy, FileEntry

    policy = ArchivePolicy()
    original = tmp_path / "old.bin"
    original.write_bytes(b"immutable original")
    baseline = (FileEntry("old.bin", hashlib.sha256(original.read_bytes()).hexdigest(), 18),)
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=baseline)
    first = _StorageBudget(workspace=tmp_path, policy=policy)
    second = _StorageBudget(workspace=tmp_path, policy=policy)
    assert first.measure()["scratch"] == 0
    with pytest.raises(FileExistsError):
        initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    with (
        first.reserve(scratch=policy.scratch_bytes),
        pytest.raises(StorageBlocked, match="scratch"),
        second.reserve(scratch=1),
    ):
        pytest.fail("a second run reset shared reservations")
    original.write_bytes(b"changed original!!")
    with pytest.raises(StorageBlocked, match="baseline"):
        first.measure()


def test_materialized_output_consumes_its_reservation_once(tmp_path):
    from silent_cascade.archive.ledger import _StorageBudget, initialize_workspace_ledger
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy(scratch_bytes=64 * 1024**2)
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    (tmp_path / "scratch").mkdir()
    budget = _StorageBudget(workspace=tmp_path, policy=policy)
    measured = budget.measure()["scratch"]
    with budget.reserve(scratch=policy.scratch_bytes - measured):
        (tmp_path / "scratch/output.bin").write_bytes(b"output")
        budget.check()
        budget.inherit_reservations = True
        with budget.reserve(scratch=1024):
            budget.check()


def test_ledger_persists_longest_category_binding_for_run_and_cache(tmp_path):
    from silent_cascade.archive.ledger import _StorageBudget, initialize_workspace_ledger
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    budget = _StorageBudget(workspace=tmp_path, policy=policy)
    budget.bind(tmp_path / "experiment", category="spool")
    budget.bind(tmp_path / "experiment/control", category="metadata")
    budget.bind(tmp_path / "experiment/control/lease-cache", category="cache")
    cached = tmp_path / "experiment/control/lease-cache/episode.bin"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached")
    restarted = _StorageBudget(workspace=tmp_path, policy=policy)
    assert restarted.measure()["cache"] >= cached.stat().st_blocks * 512


def test_category_binding_cannot_reclassify_a_live_reservation(tmp_path):
    from silent_cascade.archive.ledger import (
        StorageBlocked,
        _StorageBudget,
        initialize_workspace_ledger,
    )
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    budget = _StorageBudget(workspace=tmp_path, policy=policy)
    budget.bind(tmp_path / "run", category="spool")
    with budget.reserve(spool=4096):
        budget.bind(tmp_path / "run", category="spool")
        other = _StorageBudget(workspace=tmp_path, policy=policy)
        with pytest.raises(StorageBlocked, match="binding"):
            other.bind(tmp_path / "run/new-category", category="scratch")


@pytest.mark.parametrize("explicit_zero", [True, False])
def test_explicit_zero_admission_forbids_growth_but_omitted_category_is_not_owned(
    tmp_path, explicit_zero
):
    from silent_cascade.archive.ledger import (
        StorageBlocked,
        _StorageBudget,
        initialize_workspace_ledger,
    )
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    (tmp_path / "spool").mkdir()
    budget = _StorageBudget(workspace=tmp_path, policy=policy)
    amounts = {"spool": 0} if explicit_zero else {"scratch": 4096}
    with budget.reserve(**amounts):
        (tmp_path / "spool/output").write_bytes(b"must count these output blocks")
        if explicit_zero:
            with pytest.raises(StorageBlocked, match="spool"):
                budget.check()
        else:
            budget.check()


def test_released_nested_category_is_omitted_not_replaced_with_zero(tmp_path):
    from silent_cascade.archive.ledger import _StorageBudget, initialize_workspace_ledger
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    (tmp_path / "cache").mkdir()
    budget = _StorageBudget(workspace=tmp_path, policy=policy)
    with budget.reserve(metadata=65536):
        with budget.reserve(cache=65536):
            (tmp_path / "cache/leased-output").write_bytes(b"authenticated cached evidence")
            budget.check()
        budget.check()
        (record,) = budget._state()["reservations"].values()
        assert "cache" not in record["amounts"]
        assert budget.measure()["cache"] > 0


@pytest.mark.parametrize("interrupted", [False, True])
def test_empty_baseline_history_cannot_be_reinitialized_after_descriptor_loss(
    tmp_path, monkeypatch, interrupted
):
    from silent_cascade.archive import ledger
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    original = ledger._create_control_object
    if interrupted:

        def interrupt(root, name, raw):
            if name == "workspace.json":
                raise OSError("injected interrupted bootstrap")
            return original(root, name, raw)

        monkeypatch.setattr(ledger, "_create_control_object", interrupt)
        with pytest.raises(OSError, match="interrupted"):
            ledger.initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
        monkeypatch.setattr(ledger, "_create_control_object", original)
    else:
        ledger.initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
        budget = ledger._StorageBudget(workspace=tmp_path, policy=policy)
        context = budget.reserve(spool=4096)
        context.__enter__()
        state = tmp_path / ".silent-cascade-storage/workspace.json"
        prior = state.read_bytes()
        state.rename(state.with_name("retained-history.json"))
    try:
        with pytest.raises(ledger.StorageBlocked, match=r"bootstrap|history"):
            ledger.initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    finally:
        if not interrupted:
            # Test-only restoration exits the reservation even on a failing RED.
            assert state.with_name("retained-history.json").read_bytes() == prior
            if state.exists():
                state.rename(state.with_name("unexpected-reset.json"))
            state.with_name("retained-history.json").rename(state)
            context.__exit__(None, None, None)
