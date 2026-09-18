"""Private inventory generations retain evidence without rewriting unaffected leaves."""

import hashlib
import json
import os

import pytest

from silent_cascade.archive.ledger import StorageBlocked
from silent_cascade.archive.preflight import _accounted_inventory
from silent_cascade.archive.types import ArchivePolicy


def references(control, snapshot):
    pages, head = [], snapshot["head"]
    while head is not None:
        page = json.loads((control / f"retained/index/{head}.json").read_bytes())
        pages.append(page["leaves"])
        head = page["next"]
    return [ref for page in reversed(pages) for ref in page]


def fixture(root, *, entries=2):
    from silent_cascade.archive import _retained

    control = root / "operational"
    control.mkdir()
    for index in range(20):
        (root / f"file-{index:02}").write_bytes(bytes([index]) * 4096)
    policy = ArchivePolicy(page_entries=entries, page_bytes=4096)
    snapshot = _retained.build_snapshot(
        _accounted_inventory(root, control), policy, control_dir=control, publish=True
    )
    return control, policy, snapshot


def test_repeated_removals_reuse_unaffected_partitions_and_keep_all_generations(tmp_path):
    from silent_cascade.archive import _retained

    control, policy, snapshot = fixture(tmp_path)
    generations = [(snapshot, tuple(_retained.read_records(control, snapshot, policy)))]
    original_layout = [
        (r["partition"], r["lower"], r["upper"], r["original_entries"])
        for r in references(control, snapshot)
    ]
    for name in ("file-00", "file-10", "file-19"):
        previous = snapshot
        before_refs = references(control, previous)
        before_files = set(control.rglob("*.json"))
        bound, objects = _retained.publication_bound(control, previous, policy, (name,))
        (tmp_path / name).unlink()
        snapshot = _retained.build_snapshot(
            _accounted_inventory(tmp_path, control),
            policy,
            control_dir=control,
            previous=previous,
            removed_paths=(name,),
            publish=True,
        )
        after_refs = references(control, snapshot)
        assert [
            (r["partition"], r["lower"], r["upper"], r["original_entries"]) for r in after_refs
        ] == original_layout
        assert snapshot["layout_sha256"] == previous["layout_sha256"]
        assert (
            sum(a["sha256"] != b["sha256"] for a, b in zip(before_refs, after_refs, strict=True))
            <= 2
        )
        after_files = set(control.rglob("*.json"))
        assert before_files <= after_files
        assert len(after_files - before_files) <= objects
        assert sum(path.stat().st_size for path in after_files - before_files) <= bound
        rows = tuple(_retained.read_records(control, snapshot, policy))
        assert rows == tuple(_accounted_inventory(tmp_path, control))
        assert tuple(_retained.read_records(control, snapshot, policy, reverse=True)) == rows[::-1]
        generations.append((snapshot, rows))
    for historical, expected in generations:
        assert tuple(_retained.read_records(control, historical, policy)) == expected


def test_empty_leaf_keeps_its_original_identity_and_dfs_order(tmp_path):
    from silent_cascade.archive import _retained

    control = tmp_path / "operational"
    control.mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "a/z").write_bytes(b"nested")
    (tmp_path / "a-foo").write_bytes(b"sibling")
    policy = ArchivePolicy(page_entries=1, page_bytes=4096)
    previous = _retained.build_snapshot(
        _accounted_inventory(tmp_path, control), policy, control_dir=control, publish=True
    )
    assert [row["path"] for row in _retained.read_records(control, previous, policy)] == [
        ".",
        "a",
        "a/z",
        "a-foo",
    ]
    (tmp_path / "a/z").unlink()
    current = _retained.build_snapshot(
        _accounted_inventory(tmp_path, control),
        policy,
        control_dir=control,
        previous=previous,
        removed_paths=("a/z",),
        publish=True,
    )
    refs = references(control, current)
    assert len(refs) == 4
    assert refs[2]["lower"] == refs[2]["upper"] == "a/z"
    assert refs[2]["original_entries"] == 1
    assert refs[2]["entries"] == refs[2]["allocated"] == 0
    assert refs[3]["sha256"] == references(control, previous)[3]["sha256"]


@pytest.mark.parametrize("mutation", ["unlisted_removal", "ordinary_change", "addition"])
def test_update_cannot_infer_authority_from_changed_live_rows(tmp_path, mutation):
    from silent_cascade.archive import _retained

    control, policy, previous = fixture(tmp_path)
    before = set(control.rglob("*.json"))
    if mutation == "unlisted_removal":
        (tmp_path / "file-00").unlink()
    elif mutation == "ordinary_change":
        (tmp_path / "file-00").write_bytes(b"unaudited replacement")
    else:
        (tmp_path / "extra").write_bytes(b"unowned")
    with pytest.raises(StorageBlocked):
        _retained.build_snapshot(
            _accounted_inventory(tmp_path, control),
            policy,
            control_dir=control,
            previous=previous,
            publish=True,
        )
    assert set(control.rglob("*.json")) == before


def rewritten_object(control, kind, value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    (control / f"retained/{kind}/{digest}.json").write_bytes(raw)
    return digest


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_index",
        "missing_leaf",
        "index_hash",
        "reorder",
        "duplicate",
        "cycle",
        "row_count",
        "byte_count",
        "allocation_count",
        "layout",
        "index_limit",
        "leaf_limit",
        "leaf_boolean_layout",
        "leaf_row_order",
        "legacy",
        "noncanonical_index",
    ],
)
def test_reader_refuses_corrupt_incomplete_or_incompatible_generations(tmp_path, mutation):
    from silent_cascade.archive import _retained

    control, policy, snapshot = fixture(tmp_path, entries=4)
    snapshot = dict(snapshot)
    index_path = control / f"retained/index/{snapshot['head']}.json"
    index = json.loads(index_path.read_bytes())
    ref = index["leaves"][-1]
    leaf_path = control / f"retained/leaves/{ref['sha256']}.json"
    if mutation == "missing_index":
        index_path.rename(index_path.with_suffix(".missing"))
    elif mutation == "missing_leaf":
        leaf_path.rename(leaf_path.with_suffix(".missing"))
    elif mutation == "index_hash":
        index_path.write_bytes(index_path.read_bytes() + b" ")
    elif mutation == "reorder":
        index["leaves"].reverse()
    elif mutation == "duplicate":
        index["leaves"][-1] = index["leaves"][0]
    elif mutation == "cycle":
        index["next"] = snapshot["head"]
    elif mutation in {"row_count", "byte_count", "allocation_count"}:
        snapshot[
            {
                "row_count": "entries",
                "byte_count": "encoded_bytes",
                "allocation_count": "allocated",
            }[mutation]
        ] += 1
    elif mutation == "layout":
        snapshot["layout_sha256"] = "f" * 64
    elif mutation == "index_limit":
        index["leaves"] *= 5
    elif mutation == "leaf_limit":
        leaf_path.write_bytes(b" " * (policy.page_bytes + 1))
    elif mutation in {"leaf_boolean_layout", "leaf_row_order"}:
        # Forge a self-consistent addressed leaf and index; rejection must not
        # merely be an obsolete digest/length check.
        ref = index["leaves"][0]
        leaf_path = control / f"retained/leaves/{ref['sha256']}.json"
        leaf = json.loads(leaf_path.read_bytes())
        if mutation == "leaf_boolean_layout":
            # original_entries in this final partial leaf is one.
            ref = index["leaves"][-1]
            leaf_path = control / f"retained/leaves/{ref['sha256']}.json"
            leaf = json.loads(leaf_path.read_bytes())
            assert ref["original_entries"] == 1
            leaf["original_entries"] = True
        else:
            leaf["entries"].reverse()
        old_size = ref["encoded_bytes"]
        ref["sha256"] = rewritten_object(control, "leaves", leaf)
        ref["encoded_bytes"] = (control / f"retained/leaves/{ref['sha256']}.json").stat().st_size
        snapshot["encoded_bytes"] += ref["encoded_bytes"] - old_size
    elif mutation == "legacy":
        snapshot.pop("schema_version")
    elif mutation == "noncanonical_index":
        raw = json.dumps(index, indent=1).encode()
        snapshot["head"] = hashlib.sha256(raw).hexdigest()
        (control / f"retained/index/{snapshot['head']}.json").write_bytes(raw)
        snapshot["encoded_bytes"] += len(raw) - index_path.stat().st_size
    if mutation in {
        "reorder",
        "duplicate",
        "cycle",
        "index_limit",
        "leaf_boolean_layout",
        "leaf_row_order",
    }:
        old_size = index_path.stat().st_size
        snapshot["head"] = rewritten_object(control, "index", index)
        snapshot["encoded_bytes"] += (
            control / f"retained/index/{snapshot['head']}.json"
        ).stat().st_size - old_size
    with pytest.raises(StorageBlocked):
        tuple(_retained.read_records(control, snapshot, policy))


def test_partition_capacity_and_empty_inventory_have_explicit_semantics(tmp_path, monkeypatch):
    from silent_cascade.archive import _retained

    empty = _retained.build_snapshot(iter(()), ArchivePolicy())
    assert empty["head"] is None
    assert empty["pages"] == empty["index_pages"] == empty["entries"] == empty["allocated"] == 0
    assert tuple(_retained.read_records(tmp_path, empty, ArchivePolicy())) == ()
    control, policy, _snapshot = fixture(tmp_path, entries=1)
    monkeypatch.setattr(_retained, "_MAX_PARTITIONS", 2)
    with pytest.raises(StorageBlocked, match="capacity"):
        _retained.initial_bound(_accounted_inventory(tmp_path, control), policy)


def test_historical_generations_are_readable_without_saved_in_memory_descriptors(tmp_path):
    from silent_cascade.archive import _retained

    control, policy, snapshot = fixture(tmp_path)
    for name in ("file-00", "file-19"):
        (tmp_path / name).unlink()
        snapshot = _retained.build_snapshot(
            _accounted_inventory(tmp_path, control),
            policy,
            control_dir=control,
            previous=snapshot,
            removed_paths=(name,),
            publish=True,
        )
    del snapshot
    descriptors = list((control / "retained/generations").glob("*.json"))
    assert len(descriptors) == 3
    counts = []
    for path in descriptors:
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == path.stem
        counts.append(len(tuple(_retained.read_records(control, json.loads(raw), policy))))
    assert sorted(counts) == [19, 20, 21]


def test_bootstrap_binds_initial_generation_and_refuses_legacy_authority(tmp_path):
    from test_preflight import bootstrap

    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

    budget = bootstrap(tmp_path)
    state = budget._state()
    assert state["engineering"]["initial_snapshot"] == state["engineering"]["snapshot"]
    authority_path = tmp_path / ".silent-cascade-engineering.json"
    authority = json.loads(authority_path.read_bytes())
    authority["schema_version"] = "phase4-engineering-authority-v1"
    raw = canonical_json_bytes(authority)
    authority_path.write_bytes(raw)
    state["engineering"]["authority_sha256"] = sha256_bytes(raw)
    budget._store(state)
    with pytest.raises(StorageBlocked, match="same-ledger transition"):
        budget.check()


def test_changed_leaf_cannot_exceed_prederived_affected_set(tmp_path, monkeypatch):
    from silent_cascade.archive import _retained

    control, policy, previous = fixture(tmp_path)
    plan = _retained._publication_plan

    def missing_root_admission(*args, **kwargs):
        encoded, objects, affected = plan(*args, **kwargs)
        return encoded, objects, affected - {0}

    monkeypatch.setattr(_retained, "_publication_plan", missing_root_admission)
    before = set(control.rglob("*.json"))
    (tmp_path / "file-10").unlink()
    with pytest.raises(StorageBlocked, match="affected set"):
        _retained.build_snapshot(
            _accounted_inventory(tmp_path, control),
            policy,
            control_dir=control,
            previous=previous,
            removed_paths=("file-10",),
            publish=True,
        )
    assert set(control.rglob("*.json")) == before


@pytest.mark.parametrize(
    "stage", ["before_leaf", "after_leaf", "after_index", "after_generation", "before_head"]
)
def test_interrupted_publication_keeps_old_charge_and_exact_pending_resume(
    tmp_path, monkeypatch, stage
):
    from test_preflight import prepared_candidate

    from silent_cascade.archive import _retained, preflight
    from silent_cascade.archive.ledger import _StorageBudget

    budget, transport, candidate, review = prepared_candidate(
        tmp_path, policy=ArchivePolicy(page_entries=2)
    )
    old = budget._state()["engineering"]["snapshot"]
    old_charge = budget.retained_charge()
    historical_files = set((budget.root / "retained").rglob("*.json"))
    publish = _retained._create_control_object
    store = _StorageBudget._store
    tripped = False

    def fail_publication(root, path, raw):
        nonlocal tripped
        kind = {
            "before_leaf": "leaves",
            "after_leaf": "leaves",
            "after_index": "index",
            "after_generation": "generations",
        }.get(stage)
        if not tripped and kind and path.startswith(f"retained/{kind}/"):
            tripped = True
            if stage != "before_leaf":
                publish(root, path, raw)
            raise RuntimeError("simulated publication interruption")
        return publish(root, path, raw)

    def fail_head(self, state):
        nonlocal tripped
        if (
            stage == "before_head"
            and not tripped
            and "pending" not in state["engineering"]
            and state["engineering"]["snapshot"] != old
        ):
            tripped = True
            raise RuntimeError("simulated publication interruption")
        return store(self, state)

    with monkeypatch.context() as patch:
        patch.setattr(_retained, "_create_control_object", fail_publication)
        patch.setattr(_StorageBudget, "_store", fail_head)
        with pytest.raises(RuntimeError, match="publication interruption"):
            preflight.archive_engineering_candidate(
                budget=budget, candidate=candidate, review=review, transport=transport
            )
    assert tripped
    state = budget._state()
    assert state["engineering"]["snapshot"] == old
    assert "pending" in state["engineering"]
    assert budget.retained_charge() == old_charge
    assert historical_files <= set((budget.root / "retained").rglob("*.json"))
    result = preflight.resume_engineering_eviction(budget=budget, transport=transport)
    assert result.ref.kind == "diagnostic"
    assert "pending" not in budget._state()["engineering"]
    assert budget.retained_charge() < old_charge
    assert historical_files <= set((budget.root / "retained").rglob("*.json"))


def index_reference(index):
    return {
        "partition": index,
        "lower": f"file-{index}",
        "upper": f"file-{index}",
        "original_entries": 1,
        "sha256": f"{index:064x}",
        "encoded_bytes": 100,
        "entries": 1,
        "allocated": 4096,
    }


@pytest.mark.parametrize(
    ("page_bytes", "page_entries", "groups"),
    [
        (349, 1000, [1, 1, 1, 1]),
        (545, 1000, [2, 2]),
        (545, 1, [1, 1, 1, 1]),
    ],
)
def test_index_wire_preserves_exact_byte_entry_and_pointer_boundaries(
    page_bytes, page_entries, groups
):
    from silent_cascade.archive import _retained

    refs = [index_reference(index) for index in range(4)]
    pages = list(
        _retained._index_pages(
            refs, ArchivePolicy(page_bytes=page_bytes, page_entries=page_entries)
        )
    )
    assert [len(json.loads(raw)["leaves"]) for _digest, raw in pages] == groups
    offset, head = 0, None
    for (digest, raw), count in zip(pages, groups, strict=True):
        expected = json.dumps(
            {
                "schema_version": "phase4-engineering-index-v2",
                "next": head,
                "first_partition": offset,
                "leaves": refs[offset : offset + count],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        assert raw == expected
        assert digest == hashlib.sha256(expected).hexdigest()
        offset, head = offset + count, digest


def test_index_escaping_empty_and_oversize_pointer_growth():
    from silent_cascade.archive import _retained

    ref = index_reference(0)
    ref["lower"] = ref["upper"] = 'é"\\\n'
    pages = list(_retained._index_pages([ref], ArchivePolicy(page_bytes=4096)))
    assert json.loads(pages[0][1])["leaves"][0] == ref
    assert b'\xc3\xa9\\"\\\\\\n' in pages[0][1]
    assert list(_retained._index_pages([], ArchivePolicy())) == []
    with pytest.raises(StorageBlocked):
        list(_retained._index_pages([index_reference(0)], ArchivePolicy(page_bytes=286)))
    with pytest.raises(StorageBlocked):
        list(
            _retained._index_pages(
                [index_reference(0), index_reference(1)], ArchivePolicy(page_bytes=287)
            )
        )


def test_index_serialization_work_stays_linear(monkeypatch):
    from silent_cascade.archive import _retained

    original, work = _retained.canonical_json_bytes, 0

    def measured(value):
        nonlocal work
        work += len(value["leaves"]) if "leaves" in value else 1
        return original(value)

    monkeypatch.setattr(_retained, "canonical_json_bytes", measured)
    refs = [index_reference(index) for index in range(512)]
    pages = list(_retained._index_pages(refs, ArchivePolicy(page_entries=128, page_bytes=4096)))
    assert sum(len(json.loads(raw)["leaves"]) for _digest, raw in pages) == 512
    assert work <= 3 * len(refs)


def test_three_archives_bound_fsync_peaks_and_many_leaf_generation_growth(tmp_path, monkeypatch):
    from conftest import DirectoryTransport
    from test_preflight import bootstrap

    from silent_cascade.archive import _retained, preflight
    from silent_cascade.archive.transport import initialize_remote_reservations

    for index in range(20):
        source = tmp_path / f"tmp/task-1/group-{index:02}/payload"
        source.parent.mkdir(parents=True)
        source.write_bytes(bytes([index]) * 32768)
    budget = bootstrap(tmp_path, policy=ArchivePolicy(page_entries=4))
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
    original = budget._state()["engineering"]["snapshot"]
    assert original["pages"] >= 10
    historical_files = set((budget.root / "retained").rglob("*.json"))
    fsync = os.fsync
    metrics = []
    for index in (0, 10, 19):
        logical = f"tmp/task-1/group-{index:02}"
        candidate = next(
            item
            for item in preflight.iter_engineering_candidates(budget)
            if item.logical_root == logical
        )
        review = preflight.EngineeringContentReview(
            candidate_id=candidate.candidate_id,
            executable_sha256=candidate.executable_sha256,
            paths=(logical + "/payload",),
            producer_evidence_sha256="b" * 64,
        )
        previous = budget._state()["engineering"]["snapshot"]
        selected = tuple(member for member in candidate.members if member.path in review.paths)
        bounds = preflight._prelude_bounds(budget, candidate, selected)
        inventory_bound, object_bound = _retained.publication_bound(
            budget.root, previous, budget.policy, review.paths
        )
        before_files = set((budget.root / "retained").rglob("*.json"))
        before, observed = budget.measure(), {"metadata": 0, "scratch": 0}

        def measure_peak(descriptor, observed=observed, before=before):
            fsync(descriptor)
            measured = budget.measure()
            for category in observed:
                observed[category] = max(observed[category], measured[category] - before[category])

        with monkeypatch.context() as patch:
            patch.setattr(os, "fsync", measure_peak)
            preflight.archive_engineering_candidate(
                budget=budget, candidate=candidate, review=review, transport=transport
            )
        assert all(0 < observed[name] <= bounds[name] for name in observed)
        current = budget._state()["engineering"]["snapshot"]
        before_refs, after_refs = (
            references(budget.root, previous),
            references(budget.root, current),
        )
        changed_leaves = sum(
            a["sha256"] != b["sha256"] for a, b in zip(before_refs, after_refs, strict=True)
        )
        assert 0 < changed_leaves <= 2
        added = set((budget.root / "retained").rglob("*.json")) - before_files
        growth = sum(path.stat().st_size for path in added)
        assert len(added) <= object_bound
        assert growth <= inventory_bound
        assert growth < original["encoded_bytes"]
        assert historical_files <= set((budget.root / "retained").rglob("*.json"))
        metrics.append(
            {
                "changed_leaves": changed_leaves,
                "new_objects": len(added),
                "inventory_growth": growth,
                "inventory_bound": inventory_bound,
                "peak": observed,
                "bounds": bounds,
            }
        )
    generations = list((budget.root / "retained/generations").glob("*.json"))
    assert len(generations) == 4
    assert sorted(
        len(
            tuple(_retained.read_records(budget.root, json.loads(path.read_bytes()), budget.policy))
        )
        for path in generations
    ) == [original["entries"] - n for n in (3, 2, 1, 0)]
    print(json.dumps({"original_leaves": original["pages"], "operations": metrics}, sort_keys=True))
