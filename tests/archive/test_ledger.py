"""Shared physical allocation and durable admission tests."""

import os
import stat
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _isolate_unit_ledgers_from_an_outer_workspace_authority(tmp_path):
    from conftest import _authority_boundary, _hide_host_authorities

    with _hide_host_authorities(_authority_boundary(tmp_path)):
        yield


def test_scoped_checks_authenticate_once_and_expire_with_owned_reservation(tmp_path, monkeypatch):
    from silent_cascade.archive.ledger import (
        StorageBlocked,
        _StorageBudget,
        initialize_workspace_ledger,
    )
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    budget = _StorageBudget(workspace=tmp_path, policy=policy)
    full = budget.retained_charge
    scans = []

    def authenticate():
        scans.append(True)
        return full()

    monkeypatch.setattr(budget, "retained_charge", authenticate)
    before = budget.measure()
    with budget._scoped_reservation(admission={"job": "test"}, scratch=65536) as admitted:
        assert admitted.before == before
        assert admitted.retained.retained_bytes == 0
        budget.inherit_reservations = True
        for _ in range(4):
            budget.check_scoped(admitted.retained)
            with budget.reserve(_scope=admitted.retained, scratch=1024):
                pass
        assert len(scans) == 1
        budget.check()
        budget.check()
        assert len(scans) == 3
        with pytest.raises(ValueError), budget.reserve(_scope=admitted.retained, scratch=-1):
            pass
        fresh = _StorageBudget(workspace=tmp_path, policy=policy)
        monkeypatch.setattr(fresh, "retained_charge", authenticate)
        fresh.check()
        assert len(scans) == 4
        with (
            pytest.raises(StorageBlocked, match="not owned"),
            fresh.reserve(_scope=admitted.retained, scratch=1),
        ):
            pass
    with pytest.raises(StorageBlocked):
        budget.check_scoped(admitted.retained)


@pytest.mark.parametrize(
    "field,value",
    [
        ("workspace", "/elsewhere"),
        ("workspace_device", -1),
        ("policy_sha256", "f" * 64),
        ("authority_sha256", "f" * 64),
        ("snapshot_sha256", "f" * 64),
        ("retained_bytes", -1),
        ("retained_bytes", 1),
        ("reservation_token", "unknown"),
        ("admission_sha256", "f" * 64),
        ("owner_pid", -1),
        ("owner_create_time", 0.0),
        ("child_inheritable", True),
    ],
)
def test_scoped_checks_reject_copied_changed_capability(tmp_path, field, value):
    from dataclasses import replace

    from silent_cascade.archive.ledger import (
        StorageBlocked,
        _StorageBudget,
        initialize_workspace_ledger,
    )
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    budget = _StorageBudget(workspace=tmp_path, policy=policy)
    with budget._scoped_reservation(admission={"job": "test"}, scratch=65536) as admitted:
        with pytest.raises(StorageBlocked):
            budget.check_scoped(replace(admitted.retained, **{field: value}))
        with pytest.raises(StorageBlocked):
            budget.check_scoped({"retained_bytes": 0})


@pytest.mark.parametrize(
    "category", ["spool", "cache", "pinned", "metadata", "scratch", "logs", "emergency"]
)
def test_scoped_checks_reject_live_admission_growth(tmp_path, category):
    from silent_cascade.archive.ledger import (
        StorageBlocked,
        _StorageBudget,
        initialize_workspace_ledger,
    )
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    budget = _StorageBudget(workspace=tmp_path, policy=policy)
    (tmp_path / category).mkdir()
    with budget._scoped_reservation(admission={}, **{category: 0}) as admitted:
        (tmp_path / category / "growth").write_bytes(b"new allocated blocks")
        with pytest.raises(StorageBlocked, match=category):
            budget.check_scoped(admitted.retained)


@pytest.mark.parametrize("mutation", ["admission", "owner", "dead", "missing"])
def test_scoped_checks_require_current_live_admission(tmp_path, monkeypatch, mutation):
    from silent_cascade.archive import ledger
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    ledger.initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    budget = ledger._StorageBudget(workspace=tmp_path, policy=policy)
    with budget._scoped_reservation(admission={"job": "exact"}, scratch=65536) as admitted:
        state = budget._state()
        record = state["reservations"][admitted.token]
        if mutation == "admission":
            record["admission"]["job"] = "different"
        elif mutation == "owner":
            record["create_time"] += 1
        elif mutation == "dead":
            monkeypatch.setattr(ledger, "_process_identity", lambda _: None)
        else:
            del state["reservations"][admitted.token]
        budget._store(state)
        with pytest.raises(ledger.StorageBlocked):
            budget.check_scoped(admitted.retained)


@pytest.mark.parametrize(
    "boundary", ["spool", "cache", "pinned", "metadata", "scratch", "logs", "emergency", "physical"]
)
def test_scoped_accounting_keeps_every_capacity_boundary(tmp_path, monkeypatch, boundary):
    from silent_cascade.archive import ledger
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    ledger.initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    budget = ledger._StorageBudget(workspace=tmp_path, policy=policy)
    with budget._scoped_reservation(admission={}) as admitted:
        measured = dict.fromkeys(ledger._CATEGORIES, 0)
        if boundary in ledger._CATEGORIES:
            measured[boundary] = getattr(policy, boundary + "_bytes") + 1
        else:
            monkeypatch.setattr(
                ledger.os, "statvfs", lambda _: SimpleNamespace(f_bavail=0, f_frsize=4096)
            )
        monkeypatch.setattr(budget, "measure", lambda: measured)
        with pytest.raises(ledger.StorageBlocked):
            budget.check_scoped(admitted.retained)


@pytest.mark.parametrize(
    "mutation", ["snapshot", "authority", "charge", "pending", "retained-file", "normal-global"]
)
def test_scoped_engineering_identity_and_final_full_authentication(tmp_path, monkeypatch, mutation):
    import subprocess
    from dataclasses import asdict, replace

    from silent_cascade.archive import ledger, preflight
    from silent_cascade.archive.types import ArchivePolicy

    frozen = tmp_path / "frozen"
    frozen.write_bytes(b"retained evidence")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    budget = preflight.bootstrap_engineering_workspace(
        custody_root=tmp_path,
        workspace=tmp_path / "operational",
        policy=ArchivePolicy(),
        source_commit=revision,
    )
    with budget._scoped_reservation(admission={}) as admitted:
        scope = admitted.retained
        assert scope.retained_bytes >= frozen.stat().st_blocks * 512
        if mutation == "normal-global":
            # All category ceilings fit policy exactly; authenticated retained
            # history pushes both normal and total allocation over the cap.
            measured = {
                name: getattr(budget.policy, name + "_bytes") for name in ledger._CATEGORIES
            }
            monkeypatch.setattr(budget, "measure", lambda: measured)
            with pytest.raises(ledger.StorageBlocked, match="normal allocation"):
                budget.check_scoped(scope)
            return
        if mutation == "retained-file":
            frozen.write_bytes(b"modified evidence")
            budget.check_scoped(scope)
            with pytest.raises(ledger.StorageBlocked, match="retained inventory"):
                budget.check()
            return
        if mutation == "authority":
            authority = tmp_path / ".silent-cascade-engineering.json"
            authority.write_bytes(authority.read_bytes() + b"\n")
        elif mutation == "charge":
            identity = asdict(scope)
            identity.pop("admission_sha256")
            identity["retained_bytes"] = 0
            scope = replace(
                scope, retained_bytes=0, admission_sha256=ledger._scope_commitment({}, identity)
            )
        else:
            state = budget._state()
            if mutation == "snapshot":
                state["engineering"]["snapshot"]["encoded_bytes"] += 1
            else:
                state["engineering"]["pending"] = {}
            budget._store(state)
        with pytest.raises(ledger.StorageBlocked):
            budget.check_scoped(scope)
    if mutation == "pending":
        with (
            pytest.raises(ledger.StorageBlocked, match="pending"),
            budget._scoped_reservation(admission={}),
        ):
            pass
        assert not budget._state()["reservations"]


def _inode(*, kind, blocks, device=41, links=1, inode=1):
    return SimpleNamespace(
        st_mode=kind | 0o600,
        st_blocks=blocks,
        st_dev=device,
        st_nlink=links,
        st_ino=inode,
    )


def _memory_budget(monkeypatch, *, records, children, paths):
    from silent_cascade.archive import ledger

    workspace = Path("/virtual/workspace")
    budget = object.__new__(ledger._StorageBudget)
    budget.workspace = workspace
    budget.device = 41
    state = {"paths": paths, "baseline": None}
    monkeypatch.setattr(budget, "_state", lambda: state)
    original_lstat = Path.lstat
    original_stat = Path.stat
    original_close = os.close
    original_fstat = os.fstat
    original_open = os.open
    original_os_stat = os.stat
    original_scandir = os.scandir
    descriptors = {9000: workspace}
    next_descriptor = 9001

    def record(path):
        try:
            value = records[Path(path)]
        except KeyError:
            pytest.fail(f"unexpected filesystem access: {path}")
        return value() if callable(value) else value

    @contextmanager
    def pinned_directory(path, *, create=False):
        assert Path(path) == workspace
        assert create is False
        yield 9000

    class Entry:
        def __init__(self, parent, name):
            self.name = name
            self._path = parent / name

        def stat(self, *, follow_symlinks=True):
            assert follow_symlinks is False
            return record(self._path)

    class Scandir:
        def __init__(self, directory):
            self._entries = iter(Entry(directory, name) for name in children.get(directory, ()))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._entries)

    def lstat(path, *args, **kwargs):
        path = Path(path)
        if not path.is_relative_to(workspace):
            return original_lstat(path, *args, **kwargs)
        return record(path)

    def path_stat(path, *args, **kwargs):
        path = Path(path)
        if path.is_relative_to(workspace):
            pytest.fail(f"measurement followed an entry target: {path}")
        return original_stat(path, *args, **kwargs)

    def fstat(descriptor):
        if descriptor not in descriptors:
            return original_fstat(descriptor)
        return record(descriptors[descriptor])

    def scandir(descriptor):
        if descriptor not in descriptors:
            return original_scandir(descriptor)
        return Scandir(descriptors[descriptor])

    def open_path(path, flags, *args, **kwargs):
        nonlocal next_descriptor
        parent = kwargs.get("dir_fd")
        if parent not in descriptors:
            return original_open(path, flags, *args, **kwargs)
        child_path = descriptors[parent] / path
        info = record(child_path)
        if not stat.S_ISDIR(info.st_mode):
            raise OSError("mocked entry is no longer a directory")
        descriptor = next_descriptor
        next_descriptor += 1
        descriptors[descriptor] = child_path
        return descriptor

    def relative_stat(path, *args, **kwargs):
        parent = kwargs.get("dir_fd")
        if parent not in descriptors:
            return original_os_stat(path, *args, **kwargs)
        assert kwargs.get("follow_symlinks") is False
        return record(descriptors[parent] / path)

    def close(descriptor):
        if descriptor in descriptors and descriptor != 9000:
            del descriptors[descriptor]
            return None
        return original_close(descriptor)

    monkeypatch.setattr(ledger, "_pinned_directory", pinned_directory)
    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(Path, "stat", path_stat)
    monkeypatch.setattr(os, "fstat", fstat)
    monkeypatch.setattr(os, "scandir", scandir)
    monkeypatch.setattr(os, "open", open_path)
    monkeypatch.setattr(os, "stat", relative_stat)
    monkeypatch.setattr(os, "close", close)
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

    budget = _memory_budget(
        monkeypatch,
        records=records,
        children={
            workspace: ["scratch-zone"],
            scratch: [
                "linked-directory",
                "real-directory",
                "linked-file",
                "hard-a",
                "hard-b",
            ],
            scratch / "real-directory": [],
        },
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

    budget = _memory_budget(
        monkeypatch,
        records=records,
        children={workspace: [directory], root: ["linked-file"]},
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

    budget = _memory_budget(
        monkeypatch,
        records=records,
        children={workspace: ["scratch-zone"], scratch: ["other-device"]},
        paths={"scratch-zone": "scratch"},
    )

    with pytest.raises(StorageBlocked, match="second volume"):
        budget.measure()


def test_measure_rejects_stale_walk_beneath_linked_scratch_ancestor(monkeypatch):
    from silent_cascade.archive.ledger import StorageBlocked

    workspace = Path("/virtual/workspace")
    scratch = workspace / "scratch-zone"
    changed_directory = scratch / "changed-directory"
    observations = 0

    def changed_info():
        nonlocal observations
        observations += 1
        return _inode(
            kind=stat.S_IFDIR if observations == 1 else stat.S_IFLNK,
            blocks=3,
        )

    records = {
        workspace: _inode(kind=stat.S_IFDIR, blocks=1),
        scratch: _inode(kind=stat.S_IFDIR, blocks=2),
        changed_directory: changed_info,
    }

    budget = _memory_budget(
        monkeypatch,
        records=records,
        children={workspace: ["scratch-zone"], scratch: ["changed-directory"]},
        paths={"scratch-zone": "scratch"},
    )

    with pytest.raises(StorageBlocked, match="changed"):
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


def test_measure_never_uses_target_following_directory_classification(tmp_path, monkeypatch):
    from silent_cascade.archive.ledger import _StorageBudget, initialize_workspace_ledger
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    external = tmp_path.parent / f"{tmp_path.name}-classification-target"
    external.mkdir()
    linked_directory = scratch / "linked-directory"
    linked_directory.symlink_to(external, target_is_directory=True)
    budget = _StorageBudget(workspace=tmp_path, policy=policy)
    real_scandir = os.scandir

    class GuardedEntry:
        def __init__(self, entry):
            self._entry = entry

        def __getattr__(self, name):
            return getattr(self._entry, name)

        def is_dir(self, *args, **kwargs):
            if self.name == linked_directory.name and kwargs.get("follow_symlinks", True):
                pytest.fail("measurement followed a symlink to classify its target")
            return self._entry.is_dir(*args, **kwargs)

        def stat(self, *args, **kwargs):
            if self.name == linked_directory.name and kwargs.get("follow_symlinks", True):
                pytest.fail("measurement followed a symlink to stat its target")
            return self._entry.stat(*args, **kwargs)

    class GuardedScandir:
        def __init__(self, directory):
            self._context = real_scandir(directory)

        def __enter__(self):
            self._entries = self._context.__enter__()
            return self

        def __exit__(self, *args):
            return self._context.__exit__(*args)

        def __iter__(self):
            return self

        def __next__(self):
            return GuardedEntry(next(self._entries))

    monkeypatch.setattr(os, "scandir", GuardedScandir)

    assert (
        budget.measure()["scratch"]
        == (scratch.lstat().st_blocks + linked_directory.lstat().st_blocks) * 512
    )


def test_measure_rejects_directory_replaced_by_link_before_traversal(tmp_path, monkeypatch):
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
    changing = scratch / "changing-directory"
    changing.mkdir()
    external = tmp_path.parent / f"{tmp_path.name}-replacement-target"
    external.mkdir()
    (external / "outside-target").write_bytes(b"must not be traversed")
    budget = _StorageBudget(workspace=tmp_path, policy=policy)
    real_islink = os.path.islink
    real_open = os.open
    real_scandir = os.scandir

    def replace_with_link():
        if changing.is_dir() and not changing.is_symlink():
            changing.rmdir()
            changing.symlink_to(external, target_is_directory=True)

    def islink(path):
        result = real_islink(path)
        if Path(path) == changing and not result:
            replace_with_link()
        return result

    def open_path(path, flags, *args, **kwargs):
        if path == changing.name and flags & os.O_DIRECTORY and kwargs.get("dir_fd") is not None:
            replace_with_link()
        return real_open(path, flags, *args, **kwargs)

    def scandir(directory):
        if not isinstance(directory, int) and Path(directory) == changing and changing.is_symlink():
            pytest.fail("measurement traversed a replacement symlink")
        return real_scandir(directory)

    monkeypatch.setattr(os.path, "islink", islink)
    monkeypatch.setattr(os, "open", open_path)
    monkeypatch.setattr(os, "scandir", scandir)

    with pytest.raises(StorageBlocked, match=r"symlink|changed"):
        budget.measure()


@pytest.mark.parametrize(
    "invalid_binding",
    [
        "",
        ".",
        "/absolute",
        "alias/",
        "alias//child",
        "alias/./child",
        "alias/../child",
        ".silent-cascade-storage",
        ".silent-cascade-storage/baseline",
        ".silent-cascade-storage-copy",
    ],
)
def test_persisted_category_bindings_must_be_safe_explicit_paths(tmp_path, invalid_binding):
    from silent_cascade.archive.ledger import (
        StorageBlocked,
        _StorageBudget,
        initialize_workspace_ledger,
    )
    from silent_cascade.archive.types import ArchivePolicy

    policy = ArchivePolicy()
    initialize_workspace_ledger(workspace_root=tmp_path, policy=policy, baseline=())
    budget = _StorageBudget(workspace=tmp_path, policy=policy)
    state = budget._state()
    state["paths"][invalid_binding] = "scratch"
    budget._store(state)

    with pytest.raises(StorageBlocked, match="invalid category paths"):
        budget._state()


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
