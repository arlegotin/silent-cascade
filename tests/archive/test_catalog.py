import os

import pytest
from pydantic import ValidationError


def test_unit_inventory_binds_exact_original_bytes(tmp_path):
    from silent_cascade.archive.catalog import iter_unit_files, seal_unit
    from silent_cascade.archive.types import ArchivePolicy
    from silent_cascade.hashing import sha256_bytes
    from silent_cascade.io import atomic_create_bytes

    root = tmp_path / "run"
    atomic_create_bytes(root / "unit/a.json", b'{"value":1}\n')
    ref = seal_unit(
        run_dir=root,
        control_dir=tmp_path / "control",
        logical_root="unit",
        paths=("unit/a.json",),
        kind="partial",
        identity={
            "run_id": "debug-fixture",
            "source_commit": "0" * 40,
            "config_sha256": "1" * 64,
            "evidence_identity_sha256": "2" * 64,
            "checkpoint_sha256": None,
            "writer_stopped": True,
            "checkpoint_committed": False,
        },
        policy=ArchivePolicy(),
    )
    entries = tuple(iter_unit_files(tmp_path / "control", ref))
    assert [(entry.path, entry.sha256, entry.bytes) for entry in entries] == [
        ("unit/a.json", sha256_bytes(b'{"value":1}\n'), 12)
    ]


@pytest.mark.parametrize(
    "override",
    [
        {"chunk_bytes": True},
        {"chunk_bytes": 32 * 1024**2 + 1},
        {"journal_bytes": 1, "chunk_bytes": 2},
        {"spool_bytes": 1, "episode_bytes": 2},
        {"workspace_bytes": 1},
        {"scratch_bytes": 1},
        {"page_bytes": float("inf")},
        {"page_entries": 2**63},
    ],
)
def test_archive_policy_rejects_unbounded_or_incoherent_limits(override):
    from silent_cascade.archive.types import ArchivePolicy

    with pytest.raises(ValidationError):
        ArchivePolicy(**override)


def test_unit_identity_is_strict_and_rejects_self_described_extras(archive_identity):
    from silent_cascade.archive.types import UnitIdentity

    assert UnitIdentity.model_validate(archive_identity).run_id == "debug-fixture"
    for invalid in (
        archive_identity | {"label": "trusted"},
        archive_identity | {"source_commit": "0" * 39},
        archive_identity | {"config_sha256": "g" * 64},
        archive_identity | {"writer_stopped": 1},
    ):
        with pytest.raises(ValidationError):
            UnitIdentity.model_validate(invalid)


@pytest.mark.parametrize(
    "case", ["absolute", "traversal", "duplicate", "symlink", "parent_symlink", "hardlink", "fifo"]
)
def test_seal_refuses_unsafe_or_ambiguous_inventory(
    tmp_path, tiny_archive_policy, archive_identity, case
):
    from silent_cascade.archive.catalog import seal_unit

    run = tmp_path / "run"
    (run / "unit").mkdir(parents=True)
    source = run / "unit/a"
    source.write_bytes(b"a")
    paths = ("unit/a",)
    if case == "absolute":
        paths = (str(source),)
    elif case == "traversal":
        paths = ("unit/../unit/a",)
    elif case == "duplicate":
        paths = ("unit/a", "unit/a")
    elif case == "symlink":
        link = run / "unit/link"
        link.symlink_to(source)
        paths = ("unit/link",)
    elif case == "parent_symlink":
        alias = run / "alias"
        alias.symlink_to(run / "unit", target_is_directory=True)
        paths = ("alias/a",)
    elif case == "hardlink":
        linked = run / "unit/linked"
        os.link(source, linked)
        paths = ("unit/linked",)
    elif case == "fifo":
        fifo = run / "unit/fifo"
        os.mkfifo(fifo)
        paths = ("unit/fifo",)

    with pytest.raises((OSError, ValueError, ValidationError)):
        seal_unit(
            run_dir=run,
            control_dir=tmp_path / "control",
            logical_root="unit",
            paths=paths,
            kind="partial",
            identity=archive_identity,
            policy=tiny_archive_policy,
        )


def test_inventory_pages_are_bounded_and_missing_final_page_is_rejected(
    tmp_path, tiny_archive_policy, archive_identity
):
    from silent_cascade.archive.catalog import iter_unit_files, seal_unit
    from silent_cascade.archive.types import UnitManifest

    run = tmp_path / "run"
    for index in range(8):
        path = run / f"unit/{index:02d}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes([index]))
    ref = seal_unit(
        run_dir=run,
        control_dir=tmp_path / "control",
        logical_root="unit",
        paths=tuple(f"unit/{index:02d}" for index in range(8)),
        kind="partial",
        identity=archive_identity,
        policy=tiny_archive_policy,
    )
    manifest = UnitManifest.model_validate_json(
        (tmp_path / "control" / ref.manifest_path).read_bytes()
    )
    assert len(manifest.inventory_shards) > 1
    assert all(
        shard.entries <= tiny_archive_policy.page_entries for shard in manifest.inventory_shards
    )
    assert all(
        shard.decoded_bytes <= tiny_archive_policy.page_bytes for shard in manifest.inventory_shards
    )
    (tmp_path / "control" / ref.manifest_path).parent.joinpath(
        manifest.inventory_shards[-1].path
    ).unlink()
    with pytest.raises(ValueError, match="inventory"):
        tuple(iter_unit_files(tmp_path / "control", ref))


def test_active_ownership_rejects_same_file_but_allows_disjoint_journal_segments(
    tmp_path, tiny_archive_policy, archive_identity
):
    from silent_cascade.archive.catalog import seal_unit

    run = tmp_path / "run"
    for name in ("0001", "0002"):
        path = run / "journals" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    control = tmp_path / "control"
    first = seal_unit(
        run_dir=run,
        control_dir=control,
        logical_root="journals",
        paths=("journals/0001",),
        kind="journal",
        identity=archive_identity,
        policy=tiny_archive_policy,
    )
    assert (
        seal_unit(
            run_dir=run,
            control_dir=control,
            logical_root="journals",
            paths=("journals/0001",),
            kind="journal",
            identity=archive_identity,
            policy=tiny_archive_policy,
        )
        == first
    )
    seal_unit(
        run_dir=run,
        control_dir=control,
        logical_root="journals",
        paths=("journals/0002",),
        kind="journal",
        identity=archive_identity,
        policy=tiny_archive_policy,
    )
    with pytest.raises(ValueError, match="ownership"):
        seal_unit(
            run_dir=run,
            control_dir=control,
            logical_root="journals",
            paths=("journals/0001", "journals/0002"),
            kind="journal",
            identity=archive_identity,
            policy=tiny_archive_policy,
        )


def test_active_ownership_is_scoped_by_authenticated_run_identity(
    tmp_path, tiny_archive_policy, archive_identity
):
    from silent_cascade.archive.catalog import seal_unit

    run = tmp_path / "run"
    (run / "journals").mkdir(parents=True)
    (run / "journals/0001").write_bytes(b"record")
    arguments = {
        "run_dir": run,
        "control_dir": tmp_path / "control",
        "logical_root": "journals",
        "paths": ("journals/0001",),
        "kind": "journal",
        "policy": tiny_archive_policy,
    }
    first = seal_unit(identity=archive_identity, **arguments)
    second = seal_unit(identity=archive_identity | {"run_id": "other-run"}, **arguments)
    assert first.unit_id != second.unit_id


def test_seal_canonicalizes_caller_inventory_order(tmp_path, tiny_archive_policy, archive_identity):
    from silent_cascade.archive.catalog import seal_unit

    run = tmp_path / "run"
    (run / "unit").mkdir(parents=True)
    (run / "unit/a").write_bytes(b"a")
    (run / "unit/b").write_bytes(b"b")
    arguments = {
        "run_dir": run,
        "logical_root": "unit",
        "kind": "partial",
        "identity": archive_identity,
        "policy": tiny_archive_policy,
    }
    reversed_ref = seal_unit(
        control_dir=tmp_path / "reversed", paths=("unit/b", "unit/a"), **arguments
    )
    sorted_ref = seal_unit(control_dir=tmp_path / "sorted", paths=("unit/a", "unit/b"), **arguments)
    assert reversed_ref.unit_id == sorted_ref.unit_id


def test_manifest_rejects_malformed_span_overflow_and_unknown_kind(
    tmp_path, tiny_archive_policy, archive_identity
):
    from silent_cascade.archive.catalog import seal_unit
    from silent_cascade.archive.types import ChunkSpan, InventoryEntry

    with pytest.raises(ValidationError):
        InventoryEntry(
            path="unit/a",
            sha256="0" * 64,
            bytes=2,
            spans=(ChunkSpan(chunk_index=0, offset=0, length=1),),
        )
    with pytest.raises(ValidationError):
        ChunkSpan(chunk_index=0, offset=2**63 - 1, length=1)

    run = tmp_path / "run"
    (run / "unit").mkdir(parents=True)
    (run / "unit/a").write_bytes(b"abc")
    with pytest.raises((ValueError, ValidationError)):
        seal_unit(
            run_dir=run,
            control_dir=tmp_path / "control-a",
            logical_root="unit",
            paths=("unit/a",),
            kind="user-label",
            identity=archive_identity,
            policy=tiny_archive_policy,
        )


@pytest.mark.parametrize(
    "groups",
    [
        (),
        (("pack/a",),),
        (("pack/a", "pack/b"), ("pack/b",)),
        (("pack/b", "pack/a"),),
    ],
)
def test_episode_groups_must_be_a_sorted_exact_disjoint_partition(
    tmp_path, tiny_archive_policy, archive_identity, groups
):
    from silent_cascade.archive.catalog import seal_unit

    run = tmp_path / "run"
    (run / "pack").mkdir(parents=True)
    (run / "pack/a").write_bytes(b"a")
    (run / "pack/b").write_bytes(b"b")
    with pytest.raises(ValueError, match=r"episode|partition"):
        seal_unit(
            run_dir=run,
            control_dir=tmp_path / "control",
            logical_root="pack",
            paths=("pack/a", "pack/b"),
            kind="episode_pack",
            identity=archive_identity,
            policy=tiny_archive_policy,
            episode_groups=groups,
        )


def test_borrowed_file_is_authenticated_but_not_owned(
    tmp_path, tiny_archive_policy, archive_identity
):
    from silent_cascade.archive.catalog import iter_unit_files, seal_unit
    from silent_cascade.archive.types import FileEntry, UnitManifest
    from silent_cascade.hashing import sha256_bytes

    run = tmp_path / "run"
    (run / "pack").mkdir(parents=True)
    (run / "weights").mkdir()
    (run / "pack/episode.json").write_bytes(b"episode")
    weights = b"shared-weights"
    (run / "weights/shared.safetensors").write_bytes(weights)
    ref = seal_unit(
        run_dir=run,
        control_dir=tmp_path / "control",
        logical_root="pack",
        paths=("pack/episode.json",),
        kind="episode_pack",
        identity=archive_identity,
        policy=tiny_archive_policy,
        episode_groups=(("pack/episode.json",),),
        borrowed=(
            FileEntry(
                path="weights/shared.safetensors",
                sha256=sha256_bytes(weights),
                bytes=len(weights),
            ),
        ),
    )
    manifest = UnitManifest.model_validate_json(
        (tmp_path / "control" / ref.manifest_path).read_bytes()
    )
    assert [(entry.path, entry.sha256, entry.bytes) for entry in manifest.borrowed] == [
        ("weights/shared.safetensors", sha256_bytes(weights), len(weights))
    ]
    assert [entry.path for entry in iter_unit_files(tmp_path / "control", ref)] == [
        "pack/episode.json"
    ]


@pytest.mark.parametrize("case", ["duplicate", "owned_overlap", "substituted"])
def test_borrowed_inventory_rejects_ambiguous_or_substituted_references(
    tmp_path, tiny_archive_policy, archive_identity, case
):
    from silent_cascade.archive.catalog import seal_unit
    from silent_cascade.archive.types import FileEntry
    from silent_cascade.hashing import sha256_bytes

    run = tmp_path / "run"
    (run / "unit").mkdir(parents=True)
    (run / "weights").mkdir()
    (run / "unit/a").write_bytes(b"owned")
    (run / "weights/shared").write_bytes(b"borrowed")
    borrowed = FileEntry(
        path="weights/shared", sha256=sha256_bytes(b"borrowed"), bytes=len(b"borrowed")
    )
    if case == "duplicate":
        references = (borrowed, borrowed)
    elif case == "owned_overlap":
        references = (FileEntry(path="unit/a", sha256=sha256_bytes(b"owned"), bytes=len(b"owned")),)
    else:
        references = (
            FileEntry(path="weights/shared", sha256=sha256_bytes(b"other"), bytes=len(b"other")),
        )
    with pytest.raises(ValueError, match="borrowed"):
        seal_unit(
            run_dir=run,
            control_dir=tmp_path / "control",
            logical_root="unit",
            paths=("unit/a",),
            kind="partial",
            identity=archive_identity,
            policy=tiny_archive_policy,
            borrowed=references,
        )


def test_control_snapshots_version_original_root_paths_without_active_collision(
    tmp_path, tiny_archive_policy, archive_identity
):
    from silent_cascade.archive.catalog import iter_unit_files, seal_unit

    run = tmp_path / "run"
    (run / "state").mkdir(parents=True)
    source = run / "state/head.json"
    source.write_bytes(b"version-one")
    first = seal_unit(
        run_dir=run,
        control_dir=tmp_path / "control",
        logical_root=".",
        paths=("state/head.json",),
        kind="control_snapshot",
        identity=archive_identity | {"checkpoint_sha256": "3" * 64, "checkpoint_committed": True},
        policy=tiny_archive_policy,
    )
    source.write_bytes(b"version-two")
    second = seal_unit(
        run_dir=run,
        control_dir=tmp_path / "control",
        logical_root=".",
        paths=("state/head.json",),
        kind="control_snapshot",
        identity=archive_identity | {"checkpoint_sha256": "4" * 64, "checkpoint_committed": True},
        policy=tiny_archive_policy,
    )
    assert first.unit_id != second.unit_id
    assert next(iter_unit_files(tmp_path / "control", first)).path == "state/head.json"
    assert next(iter_unit_files(tmp_path / "control", second)).path == "state/head.json"


def test_run_root_logical_root_is_limited_to_journals_and_control_snapshots(
    tmp_path, tiny_archive_policy, archive_identity
):
    from silent_cascade.archive.catalog import seal_unit

    run = tmp_path / "run"
    run.mkdir()
    (run / "journal.jsonl").write_bytes(b"{}\n")
    seal_unit(
        run_dir=run,
        control_dir=tmp_path / "journal-control",
        logical_root=".",
        paths=("journal.jsonl",),
        kind="journal",
        identity=archive_identity,
        policy=tiny_archive_policy,
    )
    with pytest.raises(ValueError, match="logical root"):
        seal_unit(
            run_dir=run,
            control_dir=tmp_path / "partial-control",
            logical_root=".",
            paths=("journal.jsonl",),
            kind="partial",
            identity=archive_identity,
            policy=tiny_archive_policy,
        )


def test_seal_rejects_control_parent_swap_before_publication(
    tmp_path, tiny_archive_policy, archive_identity, monkeypatch
):
    from silent_cascade.archive import catalog

    run = tmp_path / "run"
    (run / "unit").mkdir(parents=True)
    (run / "unit/a").write_bytes(b"a")
    control = tmp_path / "control"
    real_control = tmp_path / "real-control"
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    original_scan = catalog._scan_inventory

    def swap_after_scan(**arguments):
        result = original_scan(**arguments)
        control.rename(real_control)
        control.symlink_to(attacker, target_is_directory=True)
        return result

    monkeypatch.setattr(catalog, "_scan_inventory", swap_after_scan)
    with pytest.raises((OSError, ValueError)):
        catalog.seal_unit(
            run_dir=run,
            control_dir=control,
            logical_root="unit",
            paths=("unit/a",),
            kind="partial",
            identity=archive_identity,
            policy=tiny_archive_policy,
        )
    assert not (attacker / "units").exists()
    assert not [
        path for path in (real_control / "units").iterdir() if not path.name.startswith(".")
    ]


def _seal_catalog_unit(tmp_path, control, policy, identity, name, payload=b"x"):
    from silent_cascade.archive.catalog import seal_unit
    from silent_cascade.hashing import sha256_bytes

    run = tmp_path / f"source-{name.replace('/', '-')}-{sha256_bytes(payload)[:8]}"
    path = run / name
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return seal_unit(
        run_dir=run,
        control_dir=control,
        logical_root=name.split("/", 1)[0],
        paths=(name,),
        kind="partial",
        identity=identity,
        policy=policy,
    )


def test_run_catalog_generations_are_paged_incremental_and_cold_readable(
    tmp_path, tiny_archive_policy, archive_identity
):
    from silent_cascade.archive.catalog import iter_run_catalog, publish_run_catalog

    control = tmp_path / "control"
    first = _seal_catalog_unit(tmp_path, control, tiny_archive_policy, archive_identity, "first/a")
    second = _seal_catalog_unit(
        tmp_path, control, tiny_archive_policy, archive_identity, "second/b"
    )
    generation = publish_run_catalog(
        control_dir=control,
        run_id="debug-fixture",
        units=(first, second),
        policy=tiny_archive_policy,
    )
    assert tuple(iter_run_catalog(control, generation, policy=tiny_archive_policy)) == tuple(
        sorted((first, second), key=lambda ref: ref.unit_id)
    )

    third = _seal_catalog_unit(tmp_path, control, tiny_archive_policy, archive_identity, "third/c")
    next_generation = publish_run_catalog(
        control_dir=control,
        run_id="debug-fixture",
        units=(third,),
        previous=generation,
        policy=tiny_archive_policy,
    )
    expected = tuple(sorted((first, second, third), key=lambda ref: ref.unit_id))
    assert tuple(iter_run_catalog(control, next_generation, policy=tiny_archive_policy)) == expected
    assert (control / next_generation.root_path).stat().st_size <= tiny_archive_policy.page_bytes

    objects = {
        path.relative_to(control).as_posix(): path.read_bytes()
        for path in (control / "catalog").rglob("*")
        if path.is_file()
    }
    requests = []

    def cold_reader(path, limit):
        requests.append((path, limit))
        payload = objects[path]
        if len(payload) > limit:
            raise ValueError("cold object exceeds request limit")
        return payload

    assert (
        tuple(
            iter_run_catalog(
                control,
                next_generation,
                policy=tiny_archive_policy,
                object_reader=cold_reader,
            )
        )
        == expected
    )
    assert requests and all(limit <= tiny_archive_policy.page_bytes for _, limit in requests)


def test_run_catalog_accepts_authenticated_cold_unit_inventory(
    tmp_path, tiny_archive_policy, archive_identity
):
    from silent_cascade.archive.catalog import iter_run_catalog, publish_run_catalog

    source_control = tmp_path / "source-control"
    unit = _seal_catalog_unit(
        tmp_path,
        source_control,
        tiny_archive_policy,
        archive_identity,
        "cold/artifact",
    )
    unit_objects = {
        path.relative_to(source_control).as_posix(): path.read_bytes()
        for path in (source_control / "units").rglob("*")
        if path.is_file()
    }

    def cold_unit_reader(path, limit):
        payload = unit_objects[path]
        if len(payload) > limit:
            raise ValueError("cold unit object exceeds request limit")
        return payload

    catalog_control = tmp_path / "catalog-control"
    generation = publish_run_catalog(
        control_dir=catalog_control,
        run_id="debug-fixture",
        units=(unit,),
        policy=tiny_archive_policy,
        unit_reader=cold_unit_reader,
    )
    assert tuple(iter_run_catalog(catalog_control, generation, policy=tiny_archive_policy)) == (
        unit,
    )


def test_catalog_publication_rejects_control_directory_swap(
    tmp_path, tiny_archive_policy, archive_identity, monkeypatch
):
    import silent_cascade.archive.catalog as catalog

    control = tmp_path / "control"
    units = tuple(
        _seal_catalog_unit(
            tmp_path,
            control,
            tiny_archive_policy,
            archive_identity,
            f"unit-{index}/artifact",
        )
        for index in range(2)
    )
    original_create = catalog._create_at
    real_control = tmp_path / "real-control"
    swapped = False

    def swap_after_first_object(parent, name, payload):
        nonlocal swapped
        original_create(parent, name, payload)
        if not swapped:
            swapped = True
            control.rename(real_control)
            control.mkdir()

    monkeypatch.setattr(catalog, "_create_at", swap_after_first_object)
    with pytest.raises(ValueError, match="changed"):
        catalog.publish_run_catalog(
            control_dir=control,
            run_id="debug-fixture",
            units=units,
            policy=tiny_archive_policy,
        )
    assert not (control / "catalog" / "roots").exists()


@pytest.mark.parametrize("relation", ["exact", "ancestor", "descendant"])
def test_run_catalog_rejects_archived_ownership_overlap(
    tmp_path, tiny_archive_policy, archive_identity, relation
):
    from silent_cascade.archive.catalog import publish_run_catalog

    control = tmp_path / "control"
    original_name = "data/a" if relation != "ancestor" else "data/a/b"
    conflicting_name = {
        "exact": "data/a",
        "ancestor": "data/a",
        "descendant": "data/a/b",
    }[relation]
    first = _seal_catalog_unit(
        tmp_path, control, tiny_archive_policy, archive_identity, original_name, b"first"
    )
    generation = publish_run_catalog(
        control_dir=control,
        run_id="debug-fixture",
        units=(first,),
        policy=tiny_archive_policy,
    )
    import shutil

    shutil.rmtree((control / first.manifest_path).parent)
    conflicting = _seal_catalog_unit(
        tmp_path,
        control,
        tiny_archive_policy,
        archive_identity,
        conflicting_name,
        b"second",
    )
    with pytest.raises(ValueError, match=r"owner|ownership|owned"):
        publish_run_catalog(
            control_dir=control,
            run_id="debug-fixture",
            units=(conflicting,),
            previous=generation,
            policy=tiny_archive_policy,
        )


def test_corpus_catalog_pages_run_generations_and_authenticates_missing_tail(
    tmp_path, tiny_archive_policy, archive_identity
):
    from silent_cascade.archive.catalog import (
        iter_corpus_catalog,
        iter_run_catalog,
        publish_corpus_catalog,
        publish_run_catalog,
    )

    control = tmp_path / "control"
    run_refs = []
    for index in range(4):
        identity = archive_identity | {"run_id": f"run-{index}"}
        unit = _seal_catalog_unit(
            tmp_path,
            control,
            tiny_archive_policy,
            identity,
            f"unit-{index}/artifact",
        )
        run_refs.append(
            publish_run_catalog(
                control_dir=control,
                run_id=f"run-{index}",
                units=(unit,),
                policy=tiny_archive_policy,
            )
        )
    corpus = publish_corpus_catalog(
        control_dir=control,
        runs=tuple(run_refs),
        policy=tiny_archive_policy,
    )
    yielded = tuple(iter_corpus_catalog(control, corpus, policy=tiny_archive_policy))
    assert {ref.run_id for ref in yielded} == {f"run-{index}" for index in range(4)}
    assert [
        next(iter_run_catalog(control, ref, policy=tiny_archive_policy)).file_count
        for ref in yielded
    ] == [1, 1, 1, 1]

    from silent_cascade.archive.types import CorpusCatalogRoot

    root = CorpusCatalogRoot.model_validate_json((control / corpus.root_path).read_bytes())
    assert root.runs is not None
    (control / root.runs.path).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        tuple(iter_corpus_catalog(control, corpus, policy=tiny_archive_policy))


def test_million_descriptor_catalog_cursor_keeps_bounded_cache(tmp_path):
    import json
    import tracemalloc

    from silent_cascade.archive.catalog import iter_run_catalog
    from silent_cascade.archive.types import (
        ArchivePolicy,
        CatalogNode,
        CatalogNodeRef,
        CatalogRef,
        RunCatalogRoot,
    )
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

    total = 1_000_001
    page_entries = 1_000
    policy = ArchivePolicy()
    leaf_specs = {}
    refs = []

    def leaf_payload(start, count):
        records = []
        for index in range(start, start + count):
            unit_id = f"{index:064x}"
            records.append(
                {
                    "expanded_bytes": 0,
                    "file_count": 1,
                    "key": unit_id,
                    "kind": "partial",
                    "logical_root": "u",
                    "manifest_path": f"units/{unit_id}/manifest.json",
                    "record_type": "unit",
                    "unit_id": unit_id,
                }
            )
        value = {
            "depth": 0,
            "index": "units",
            "one": None,
            "records": records,
            "schema_version": "phase4-r2-catalog-node-v1",
            "zero": None,
        }
        return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()

    for start in range(0, total, page_entries):
        count = min(page_entries, total - start)
        payload = leaf_payload(start, count)
        digest = sha256_bytes(payload)
        path = f"catalog/nodes/{digest}.json"
        leaf_specs[path] = (start, count)
        refs.append(
            CatalogNodeRef(
                sha256=digest,
                path=path,
                entries=count,
                decoded_bytes=len(payload) + 1,
                first_key=f"{start:064x}",
                last_key=f"{start + count - 1:064x}",
            )
        )

    objects = {}
    while len(refs) > 1:
        combined = []
        for offset in range(0, len(refs), 2):
            if offset + 1 == len(refs):
                combined.append(refs[offset])
                continue
            node = CatalogNode(index="units", depth=0, zero=refs[offset], one=refs[offset + 1])
            payload = canonical_json_bytes(node)
            digest = sha256_bytes(payload)
            path = f"catalog/nodes/{digest}.json"
            objects[path] = payload + b"\n"
            combined.append(
                CatalogNodeRef(
                    sha256=digest,
                    path=path,
                    entries=refs[offset].entries + refs[offset + 1].entries,
                    decoded_bytes=len(payload) + 1,
                    first_key=refs[offset].first_key,
                    last_key=refs[offset + 1].last_key,
                )
            )
        refs = combined
    root = RunCatalogRoot(
        run_id="million-fixture",
        units=refs[0],
        ownership=None,
        unit_count=total,
        ownership_count=0,
    )
    root_payload = canonical_json_bytes(root)
    catalog_id = sha256_bytes(root_payload)
    root_path = f"catalog/roots/{catalog_id}.json"
    objects[root_path] = root_payload + b"\n"
    ref = CatalogRef(catalog_id, "run", "million-fixture", total, root_path)
    largest_request = 0

    def lazy_reader(path, limit):
        nonlocal largest_request
        largest_request = max(largest_request, limit)
        payload = leaf_payload(*leaf_specs[path]) + b"\n" if path in leaf_specs else objects[path]
        if len(payload) > limit:
            raise ValueError("synthetic page exceeds bounded request")
        return payload

    tracemalloc.start()
    count = 0
    first = last = None
    for unit in iter_run_catalog(
        tmp_path,
        ref,
        policy=policy,
        object_reader=lazy_reader,
    ):
        first = unit.unit_id if first is None else first
        last = unit.unit_id
        count += 1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert (count, first, last) == (total, f"{0:064x}", f"{total - 1:064x}")
    assert largest_request <= policy.page_bytes
    assert sum(map(len, objects.values())) < 4 * 1024**2
    assert peak < 64 * 1024**2
