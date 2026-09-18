"""Shared physical allocation and durable admission tests."""

import os

import pytest


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
