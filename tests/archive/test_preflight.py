"""Frozen engineering custody must consume the existing global allowance."""

import json
import os
import subprocess
from dataclasses import replace

import pytest

from silent_cascade.archive import ledger
from silent_cascade.archive.types import ArchivePolicy
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes


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


def _set_source_identity(monkeypatch, preflight, revision, executable):
    identity = preflight._SourceIdentity(revision, executable)
    monkeypatch.setattr(preflight, "_current_source_identity", lambda: identity)
    monkeypatch.setattr(preflight, "_committed_source_identity", lambda: identity)
    return identity


def _publish_release(budget, monkeypatch, *, revision="1" * 40, executable="2" * 64):
    from silent_cascade.archive import preflight

    _set_source_identity(monkeypatch, preflight, revision, executable)
    digest = preflight.publish_reviewed_source(
        budget=budget,
        source_commit=revision,
        executable_sha256=executable,
        review_sha256="3" * 64,
    )
    return revision, executable, digest


def _source_authority_path(budget, revision, executable):
    return budget.root / "source-authorities" / executable / f"{revision}.json"


def _git_package(root):
    repository = root / "repository"
    package = repository / "src/silent_cascade"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"], cwd=repository, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repository, check=True)
    return repository, package


def test_reviewed_source_requires_explicit_hash_and_preserves_bootstrap(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight

    budget = bootstrap(tmp_path)
    engineering = budget._state()["engineering"]
    anchor_path = tmp_path / ".silent-cascade-engineering.json"
    anchor_before = anchor_path.read_bytes()
    bootstrap_source = preflight._reviewed_source(budget=budget, source_commit=source_commit())
    assert bootstrap_source.source_authority_sha256 == engineering["authority_sha256"]
    revision, executable, digest = _publish_release(budget, monkeypatch)
    with pytest.raises(ValueError, match="explicit"):
        preflight._reviewed_source(budget=budget, source_commit=revision)
    resolved = preflight._reviewed_source(
        budget=budget, source_commit=revision, authority_sha256=digest
    )
    assert (resolved.source_commit, resolved.executable_sha256) == (revision, executable)
    assert resolved.source_authority_sha256 == digest
    record_path = _source_authority_path(budget, revision, executable)
    raw = record_path.read_bytes()
    assert len(raw) <= 16_384
    assert raw == canonical_json_bytes(json.loads(raw))
    assert json.loads(raw) == {
        "schema_version": "phase4-reviewed-source-v1",
        "anchor_authority_sha256": engineering["authority_sha256"],
        "source_commit": revision,
        "executable_sha256": executable,
        "source_closure_schema": "silent-cascade-python-v1",
        "policy_sha256": sha256_bytes(canonical_json_bytes(budget.policy)),
        "review_sha256": "3" * 64,
        "failure_proof_sha256": None,
    }
    assert anchor_path.read_bytes() == anchor_before
    assert budget._state()["engineering"]["authority_sha256"] == engineering["authority_sha256"]


def test_reviewed_source_publication_is_create_only_and_idempotent(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight

    budget = bootstrap(tmp_path)
    revision, executable, first = _publish_release(budget, monkeypatch)
    second = preflight.publish_reviewed_source(
        budget=budget,
        source_commit=revision,
        executable_sha256=executable,
        review_sha256="3" * 64,
    )
    assert second == first
    before = _source_authority_path(budget, revision, executable).read_bytes()
    with pytest.raises(FileExistsError, match="different"):
        preflight.publish_reviewed_source(
            budget=budget,
            source_commit=revision,
            executable_sha256=executable,
            review_sha256="4" * 64,
        )
    assert _source_authority_path(budget, revision, executable).read_bytes() == before


@pytest.mark.parametrize(
    "fault",
    ["hash", "anchor", "policy", "revision", "digest", "noncanonical", "oversized"],
)
def test_reviewed_source_rejects_wrong_or_unbounded_record(tmp_path, monkeypatch, fault):
    from silent_cascade.archive import preflight

    budget = bootstrap(tmp_path)
    revision, executable, expected = _publish_release(budget, monkeypatch)
    path = _source_authority_path(budget, revision, executable)
    if fault == "hash":
        expected = "f" * 64
    elif fault == "noncanonical":
        path.write_bytes(path.read_bytes() + b"\n")
        expected = sha256_bytes(path.read_bytes())
    elif fault == "oversized":
        path.write_bytes(b"{" + b" " * 16_384 + b"}")
        expected = sha256_bytes(path.read_bytes())
    else:
        record = json.loads(path.read_bytes())
        field = {
            "anchor": "anchor_authority_sha256",
            "policy": "policy_sha256",
            "revision": "source_commit",
            "digest": "executable_sha256",
        }[fault]
        record[field] = "f" * (40 if fault == "revision" else 64)
        raw = canonical_json_bytes(record)
        path.write_bytes(raw)
        expected = sha256_bytes(raw)
    with pytest.raises((ValueError, ledger.StorageBlocked)):
        preflight._reviewed_source(budget=budget, source_commit=revision, authority_sha256=expected)


def test_same_digest_revisions_require_distinct_explicit_authorities(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight

    budget = bootstrap(tmp_path)
    executable = "2" * 64
    first_revision, _, first = _publish_release(
        budget, monkeypatch, revision="1" * 40, executable=executable
    )
    _set_source_identity(monkeypatch, preflight, "4" * 40, executable)
    with pytest.raises((FileNotFoundError, ValueError)):
        preflight._reviewed_source(budget=budget, source_commit="4" * 40, authority_sha256=first)
    second = preflight.publish_reviewed_source(
        budget=budget,
        source_commit="4" * 40,
        executable_sha256=executable,
        review_sha256="5" * 64,
        failure_proof_sha256="6" * 64,
    )
    assert second != first
    assert _source_authority_path(budget, first_revision, executable).exists()
    assert _source_authority_path(budget, "4" * 40, executable).exists()
    with pytest.raises(ValueError, match="explicit"):
        preflight._reviewed_source(budget=budget, source_commit="4" * 40)


def test_publication_rejects_pending_engineering_eviction_before_output(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight

    budget = bootstrap(tmp_path)
    authenticated = preflight._authority(budget)
    _set_source_identity(monkeypatch, preflight, "1" * 40, "2" * 64)
    monkeypatch.setattr(preflight, "_authority", lambda _budget: authenticated | {"pending": {}})
    with pytest.raises(ledger.StorageBlocked, match="pending"):
        preflight.publish_reviewed_source(
            budget=budget,
            source_commit="1" * 40,
            executable_sha256="2" * 64,
            review_sha256="3" * 64,
        )
    assert not (budget.root / "source-authorities").exists()


def test_publication_rejects_mismatching_source_argument_before_output(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight

    budget = bootstrap(tmp_path)
    _set_source_identity(monkeypatch, preflight, "1" * 40, "2" * 64)
    with pytest.raises(ValueError, match="arguments"):
        preflight.publish_reviewed_source(
            budget=budget,
            source_commit="3" * 40,
            executable_sha256="2" * 64,
            review_sha256="4" * 64,
        )
    assert not (budget.root / "source-authorities").exists()


@pytest.mark.parametrize("mutation", ["modified", "added", "removed"])
def test_committed_source_identity_rejects_dirty_python_closure(tmp_path, mutation):
    from silent_cascade.archive import preflight

    repository, package = _git_package(tmp_path)
    clean = preflight._committed_source_identity(repo_root=repository, package=package)
    if mutation == "modified":
        (package / "__init__.py").write_text("VALUE = 2\n")
    elif mutation == "added":
        (package / "added.py").write_text("VALUE = 2\n")
    else:
        (package / "__init__.py").unlink()
    with pytest.raises(ledger.StorageBlocked, match="committed"):
        preflight._committed_source_identity(repo_root=repository, package=package)
    assert len(clean.source_commit) == 40 and len(clean.executable_sha256) == 64


@pytest.mark.parametrize("source_read", [1, 2], ids=["initial-pass", "final-pass"])
def test_publication_rejects_python_added_during_authentication_before_output(
    tmp_path, monkeypatch, source_read
):
    from silent_cascade.archive import preflight

    repository, package = _git_package(tmp_path)
    clean = preflight._committed_source_identity(repo_root=repository, package=package)
    custody = tmp_path / "custody"
    custody.mkdir()
    budget = bootstrap(custody)
    authenticate = preflight._committed_source_identity
    read_at = preflight._read_at
    source_reads = 0

    def add_after_source_read(directory, name, *, max_bytes):
        nonlocal source_reads
        raw = read_at(directory, name, max_bytes=max_bytes)
        if name == "__init__.py":
            source_reads += 1
            if source_reads == source_read:
                (package / "added.py").write_text("VALUE = 2\n")
        return raw

    with monkeypatch.context() as authentication:
        authentication.setattr(
            preflight,
            "_committed_source_identity",
            lambda: authenticate(repo_root=repository, package=package),
        )
        authentication.setattr(preflight, "_read_at", add_after_source_read)
        with pytest.raises(ledger.StorageBlocked, match="changed during authentication"):
            preflight.publish_reviewed_source(
                budget=budget,
                source_commit=clean.source_commit,
                executable_sha256=clean.executable_sha256,
                review_sha256="3" * 64,
            )
    assert source_reads >= source_read
    assert not (budget.root / "source-authorities").exists()


def test_committed_source_identity_rejects_head_change_during_authentication(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight

    repository, package = _git_package(tmp_path)
    read_at = preflight._read_at
    changed = False

    def commit_after_source_read(directory, name, *, max_bytes):
        nonlocal changed
        raw = read_at(directory, name, max_bytes=max_bytes)
        if name == "__init__.py" and not changed:
            subprocess.run(
                ["git", "commit", "--allow-empty", "-qm", "new head"],
                cwd=repository,
                check=True,
            )
            changed = True
        return raw

    with monkeypatch.context() as authentication:
        authentication.setattr(preflight, "_read_at", commit_after_source_read)
        with pytest.raises(ledger.StorageBlocked, match="changed during authentication"):
            preflight._committed_source_identity(repo_root=repository, package=package)
    assert changed


def test_reviewed_candidate_binds_release_and_sealed_executor_identity(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight
    from silent_cascade.archive.catalog import _opened_unit

    budget, transport, bootstrap_candidate, _review = prepared_candidate(tmp_path)
    revision, executable, authority = _publish_release(budget, monkeypatch)
    candidate = next(
        preflight.iter_engineering_candidates(
            budget,
            source_commit=revision,
            source_authority_sha256=authority,
        )
    )
    assert candidate.candidate_id != bootstrap_candidate.candidate_id
    assert (candidate.source_commit, candidate.executable_sha256) == (revision, executable)
    assert candidate.source_authority_sha256 == authority
    review = preflight.EngineeringContentReview(
        candidate_id=candidate.candidate_id,
        executable_sha256=candidate.executable_sha256,
        paths=("tmp/task-1/safe/tensor.bin",),
        producer_evidence_sha256="7" * 64,
    )
    changed = replace(candidate, source_authority_sha256="f" * 64)
    with pytest.raises((ValueError, ledger.StorageBlocked)):
        preflight.archive_engineering_candidate(
            budget=budget, candidate=changed, review=review, transport=transport
        )
    assert transport.create_calls == 0
    result = preflight.archive_engineering_candidate(
        budget=budget, candidate=candidate, review=review, transport=transport
    )
    with _opened_unit(budget.workspace / "control", result.ref) as (_unit, manifest):
        assert manifest.identity.source_commit == revision


def test_reviewed_candidate_resume_requires_same_exact_source(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight
    from silent_cascade.archive import transport as transfer

    budget, transport, _bootstrap_candidate, _review = prepared_candidate(tmp_path)
    revision, executable, authority = _publish_release(budget, monkeypatch)
    candidate = next(
        preflight.iter_engineering_candidates(
            budget,
            source_commit=revision,
            source_authority_sha256=authority,
        )
    )
    review = preflight.EngineeringContentReview(
        candidate_id=candidate.candidate_id,
        executable_sha256=candidate.executable_sha256,
        paths=("tmp/task-1/safe/tensor.bin",),
        producer_evidence_sha256="8" * 64,
    )
    original = transfer._write_intent

    def interrupt(control_dir, ref, intent, *, replace):
        if intent["completed"]:
            raise OSError("interrupted reviewed eviction")
        return original(control_dir, ref, intent, replace=replace)

    monkeypatch.setattr(transfer, "_write_intent", interrupt)
    with pytest.raises(OSError, match="interrupted reviewed"):
        preflight.archive_engineering_candidate(
            budget=budget, candidate=candidate, review=review, transport=transport
        )
    pending = budget._state()["engineering"]["pending"]
    assert pending["candidate"] == {
        "candidate_id": candidate.candidate_id,
        "source_commit": revision,
        "executable_sha256": executable,
        "source_authority_sha256": authority,
        "logical_root": candidate.logical_root,
        "members": [
            {"path": item.path, "sha256": item.sha256, "bytes": item.bytes}
            for item in candidate.members
        ],
    }
    monkeypatch.setattr(transfer, "_write_intent", original)
    _set_source_identity(monkeypatch, preflight, "9" * 40, "a" * 64)
    with pytest.raises((ValueError, ledger.StorageBlocked)):
        preflight.resume_engineering_eviction(budget=budget, transport=transport)
    assert "pending" in budget._state()["engineering"]
    _set_source_identity(monkeypatch, preflight, revision, executable)
    preflight.resume_engineering_eviction(budget=budget, transport=transport)
    assert "pending" not in budget._state()["engineering"]


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
        page = next((budget.root / "retained/leaves").glob("*.json"))
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


def test_host_authority_isolation_preserves_local_and_unrelated_reads(
    tmp_path, isolated_archive_authorities
):
    from silent_cascade.archive import preflight
    from silent_cascade.archive.catalog import _pinned_directory

    host = next(parent for parent in tmp_path.parents if (parent / preflight._AUTHORITY).exists())
    with _pinned_directory(host) as directory:
        assert preflight._read_authority_at(directory) is None
        with isolated_archive_authorities.host_visible():
            assert (
                preflight._read_authority_at(directory)
                == (host / preflight._AUTHORITY).read_bytes()
            )
        assert preflight._read_authority_at(directory) is None
    ordinary = tmp_path / "ordinary.bin"
    ordinary.write_bytes(b"ordinary")
    with _pinned_directory(tmp_path) as directory:
        assert preflight._read_at(directory, ordinary.name, max_bytes=8) == b"ordinary"
    (tmp_path / preflight._AUTHORITY).write_bytes(
        canonical_json_bytes({"workspace": str(tmp_path / "owned")})
    )
    (tmp_path / "nested").mkdir()
    with pytest.raises(ledger.StorageBlocked, match="another workspace"):
        preflight._require_single_authority(tmp_path / "nested" / "candidate")
    (tmp_path / preflight._AUTHORITY).write_bytes(b"not-json")
    with pytest.raises(json.JSONDecodeError):
        preflight._require_single_authority(tmp_path / "nested" / "candidate")


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


@pytest.mark.parametrize(
    ("logical_bytes", "native_pair_bytes"),
    [
        (32 * 1024**2, 134_225_920),
        (14 * 1024**2, 62_922_752),
    ],
)
def test_prelude_native_pair_accounts_for_both_allocated_writes(
    tmp_path, monkeypatch, logical_bytes, native_pair_bytes
):
    from silent_cascade.archive import preflight
    from silent_cascade.archive.types import FileEntry

    budget, _transport, candidate, review = prepared_candidate(tmp_path)
    original = next(member for member in candidate.members if member.path in review.paths)
    selected = (FileEntry(original.path, original.sha256, logical_bytes),)
    values = list(os.statvfs(tmp_path))
    values[1] = 4096
    monkeypatch.setattr(os, "statvfs", lambda _: os.statvfs_result(values))

    bounds = preflight._prelude_bounds(budget, candidate, selected)

    assert bounds["scratch"] >= native_pair_bytes


@pytest.mark.parametrize(
    ("logical_bytes", "native_pair_bytes"),
    [
        (14 * 1024**2, 62_922_752),
        (1, 33_570_816),
    ],
)
def test_prelude_native_cold_root_accounts_for_full_page_read(
    tmp_path, monkeypatch, logical_bytes, native_pair_bytes
):
    from silent_cascade.archive import preflight
    from silent_cascade.archive.types import FileEntry

    budget, _transport, candidate, review = prepared_candidate(tmp_path)
    original = next(member for member in candidate.members if member.path in review.paths)
    selected = (FileEntry(original.path, original.sha256, logical_bytes),)
    values = list(os.statvfs(tmp_path))
    values[1] = 4096
    monkeypatch.setattr(os, "statvfs", lambda _: os.statvfs_result(values))

    bounds = preflight._prelude_bounds(budget, candidate, selected)

    assert bounds["scratch"] >= native_pair_bytes


def test_prelude_native_cold_root_refuses_when_old_pair_would_fit(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight
    from silent_cascade.archive.types import FileEntry

    budget, _transport, candidate, review = prepared_candidate(tmp_path)
    source = tmp_path / review.paths[0]
    original_bytes = source.read_bytes()
    original = next(member for member in candidate.members if member.path in review.paths)
    selected = (FileEntry(original.path, original.sha256, 14 * 1024**2),)
    values = list(os.statvfs(tmp_path))
    values[1] = 4096
    monkeypatch.setattr(os, "statvfs", lambda _: os.statvfs_result(values))
    bounds = preflight._prelude_bounds(budget, candidate, selected)
    measured = budget.measure()
    measured["scratch"] = 205_520_896
    monkeypatch.setattr(budget, "measure", lambda: dict(measured))
    yielded = False

    with (
        pytest.raises(ledger.StorageBlocked, match="scratch category exhausted"),
        preflight._engineering_reservation(budget, candidate.candidate_id, bounds),
    ):
        yielded = True

    assert not yielded
    assert source.read_bytes() == original_bytes
    assert budget._state()["reservations"] == {}


def test_prelude_native_pair_refuses_inflated_measured_scratch_before_yield(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight
    from silent_cascade.archive.types import FileEntry

    budget, _transport, candidate, review = prepared_candidate(tmp_path)
    source = tmp_path / review.paths[0]
    original_bytes = source.read_bytes()
    original = next(member for member in candidate.members if member.path in review.paths)
    selected = (FileEntry(original.path, original.sha256, 32 * 1024**2),)
    values = list(os.statvfs(tmp_path))
    values[1] = 4096
    monkeypatch.setattr(os, "statvfs", lambda _: os.statvfs_result(values))
    bounds = preflight._prelude_bounds(budget, candidate, selected)
    measured = budget.measure()
    measured["scratch"] = 199_651_328
    monkeypatch.setattr(budget, "measure", lambda: dict(measured))
    yielded = False

    with (
        pytest.raises(ledger.StorageBlocked, match="scratch category exhausted"),
        preflight._engineering_reservation(budget, candidate.candidate_id, bounds),
    ):
        yielded = True

    assert not yielded
    assert source.read_bytes() == original_bytes
    assert budget._state()["reservations"] == {}


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


def _targeted_candidate_budget(root):
    canonical = root / "tmp/task-1/canonical"
    canonical.mkdir(parents=True)
    (canonical / "one.bin").write_bytes(b"one")
    target = root / "tmp/task-2/target"
    target.mkdir(parents=True)
    (target / "a.bin").write_bytes(b"a")
    (target / "b.bin").write_bytes(b"b")
    (target.parent / "unsafe").symlink_to(target)
    unrelated = root / "tmp/task-3/unrelated"
    unrelated.mkdir(parents=True)
    (unrelated / "history.bin").write_bytes(b"history")
    return bootstrap(root)


def test_targeted_discovery_preserves_canonical_partition_and_bounds_scans(tmp_path, monkeypatch):
    from silent_cascade.archive import preflight

    budget = _targeted_candidate_budget(tmp_path)
    stored_records = preflight._stored_records
    scans = 0

    def counted_records(*args):
        nonlocal scans
        scans += 1
        yield from stored_records(*args)

    monkeypatch.setattr(preflight, "_stored_records", counted_records)
    default = tuple(preflight.iter_engineering_candidates(budget))
    default_scans = scans
    expected = next(item for item in default if item.logical_root == "tmp/task-2/target")

    scans = 0
    targeted = tuple(
        preflight.iter_engineering_candidates(budget, logical_root="tmp/task-2/target")
    )
    assert targeted == (expected,)
    assert tuple(member.path for member in targeted[0].members) == (
        "tmp/task-2/target/a.bin",
        "tmp/task-2/target/b.bin",
    )
    assert scans == 2
    assert default_scans > scans

    assert not tuple(
        preflight.iter_engineering_candidates(budget, logical_root="tmp/task-1/canonical")
    )
    for logical_root in (
        "",
        ".",
        "/tmp/task-2/target",
        "tmp/task-2/target/",
        "tmp//task-2/target",
        "tmp/task-2/../target",
        "tmp\\task-2\\target",
        "tmp/task-7/target",
        "runs/task-2/target",
    ):
        with pytest.raises(ValueError, match="logical root"):
            tuple(preflight.iter_engineering_candidates(budget, logical_root=logical_root))

    review = preflight.EngineeringContentReview(
        candidate_id=expected.candidate_id,
        executable_sha256=expected.executable_sha256,
        paths=("tmp/task-2/target/a.bin",),
        producer_evidence_sha256="d" * 64,
    )
    scans = 0
    selected = preflight._validated_review(budget, expected, review)
    assert tuple((member.path, member.sha256, member.bytes) for member in selected) == (
        ("tmp/task-2/target/a.bin", sha256_bytes(b"a"), 1),
    )
    assert scans == 2


def test_targeted_discovery_authenticates_unrelated_retained_history(tmp_path):
    from silent_cascade.archive import preflight

    budget = _targeted_candidate_budget(tmp_path)
    (tmp_path / "tmp/task-3/unrelated/history.bin").write_bytes(b"changed")
    with pytest.raises(ledger.StorageBlocked):
        tuple(preflight.iter_engineering_candidates(budget, logical_root="tmp/task-2/target"))


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
