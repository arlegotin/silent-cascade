"""Bounded persisted inventories preserve complete legacy/public hash maps."""

import gzip

import pytest

from silent_cascade.hashing import canonical_json_bytes, sha256_bytes


def entries(count=9):
    return [
        (f"attempt-fixture/files/{index:05d}.json", sha256_bytes(str(index).encode()))
        for index in range(count)
    ]


def test_index_roundtrip_is_streamed_byte_bounded_and_create_only(tmp_path):
    from silent_cascade.train.pilot_artifact_index import iter_artifact_index, write_artifact_index

    values = entries()
    index = write_artifact_index(
        root=tmp_path,
        prefix="attempt-fixture/index",
        entries=iter(values),
        max_rows=3,
        max_bytes=250,
    )
    assert index.entry_count == 9 and len(index.shards) >= 5
    stream = iter_artifact_index(tmp_path, index)
    assert iter(stream) is stream
    assert list(stream) == values
    expected = b"".join(canonical_json_bytes(dict(path=p, sha256=h)) + b"\n" for p, h in values)
    assert index.entries_sha256 == sha256_bytes(expected)
    assert sum(part.decompressed_bytes for part in index.shards) == len(expected)
    for shard in index.shards:
        assert (
            len(gzip.decompress((tmp_path / shard.path).read_bytes()))
            == shard.decompressed_bytes
            <= 250
        )
    assert (
        write_artifact_index(
            root=tmp_path,
            prefix="attempt-fixture/index",
            entries=iter(values),
            max_rows=3,
            max_bytes=250,
        )
        == index
    )
    with pytest.raises(ValueError, match="duplicate"):
        write_artifact_index(
            root=tmp_path,
            prefix="attempt-fixture/index",
            entries=iter(entries(8)),
            max_rows=3,
            max_bytes=250,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "hash",
        "truncated",
        "tail_count",
        "tail_bytes",
        "global_order",
        "duplicate",
        "noncanonical",
    ],
)
def test_index_rejects_corruption_through_last_shard(tmp_path, mutation):
    from silent_cascade.train.pilot_artifact_index import iter_artifact_index, write_artifact_index

    index = write_artifact_index(
        root=tmp_path, prefix="attempt-fixture/index", entries=iter(entries()), max_rows=3
    )
    last = index.shards[-1]
    path = tmp_path / last.path
    if mutation == "missing":
        path.rename(path.with_suffix(".retained"))
    elif mutation == "hash":
        path.write_bytes(b"changed")
    elif mutation == "tail_count":
        index = index.model_copy(update={"entry_count": index.entry_count - 1})
    elif mutation == "tail_bytes":
        index = index.model_copy(
            update={
                "shards": (
                    *index.shards[:-1],
                    last.model_copy(update={"decompressed_bytes": last.decompressed_bytes - 1}),
                )
            }
        )
    else:
        values = entries()[6:]
        if mutation == "truncated":
            raw = path.read_bytes()[:-6]
        elif mutation == "noncanonical":
            raw = gzip.compress(b" " + gzip.decompress(path.read_bytes()), mtime=0)
        else:
            values[0] = entries()[0] if mutation == "global_order" else entries()[5]
            raw = gzip.compress(
                b"".join(canonical_json_bytes(dict(path=p, sha256=h)) + b"\n" for p, h in values),
                mtime=0,
            )
        path.write_bytes(raw)
        index = index.model_copy(
            update={
                "shards": (
                    *index.shards[:-1],
                    last.model_copy(update={"sha256": sha256_bytes(raw)}),
                )
            }
        )
    with pytest.raises((ValueError, OSError)):
        list(iter_artifact_index(tmp_path, index))


@pytest.mark.parametrize(
    "path", ["../escape", "/absolute", "safe/../escape", "frozen/data", "./file"]
)
def test_index_refuses_unsafe_entry_paths(tmp_path, path):
    from silent_cascade.train.pilot_artifact_index import write_artifact_index

    with pytest.raises(ValueError):
        write_artifact_index(root=tmp_path, prefix="index", entries=iter([(path, "a" * 64)]))


def test_three_million_entries_fit_bounded_index_not_flat_result(tmp_path):
    from silent_cascade.train.pilot_artifact_index import write_artifact_index

    path = (
        "attempt-" + "a" * 32 + "/validation-75000-robustness/autonomous/episodes/00000.neural.json"
    )
    row_bytes = len(canonical_json_bytes({"path": path, "sha256": "b" * 64})) + 1
    flat_entry_bytes = len(canonical_json_bytes({path: "b" * 64})) - 1
    count = 3_000_000
    assert flat_entry_bytes * count > 128 * 1024**2
    assert row_bytes * 50_000 < 128 * 1024**2
    tiny = write_artifact_index(root=tmp_path, prefix="index", entries=iter(entries(1)))
    # The production-cardinality descriptor envelope needs60parts, not3M maps.
    projection = tiny.model_dump(mode="json")
    projection["entry_count"] = count
    projection["shards"] = [
        dict(
            tiny.shards[0].model_dump(mode="json"),
            path=f"attempt-fixture/index.{start:07d}.jsonl.gz",
            rows=50_000,
            decompressed_bytes=50_000 * row_bytes,
        )
        for start in range(0, count, 50_000)
    ]
    assert len(canonical_json_bytes(projection)) < 32 * 1024


def test_v2_and_unchanged_legacy_results_decode_to_same_public_map(tmp_path):
    from silent_cascade.report.pilot_artifacts import load_training_result
    from silent_cascade.train.pilot_artifact_index import training_result_payload
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_state import PilotCheckpointDescriptor, PilotProgress

    raw = b"fixture-only retained bytes, not a neural archive"
    path = tmp_path / "attempt-fixture/latest-weights.safetensors"
    path.parent.mkdir()
    path.write_bytes(raw)
    descriptor = PilotCheckpointDescriptor(
        path=path.relative_to(tmp_path).as_posix(),
        sha256=sha256_bytes(raw),
        model_state_sha256="a" * 64,
        global_step=1,
        stage="one_hop",
    )
    progress = PilotProgress(
        global_step=1, batch_counter=1, latest=descriptor, status="step_ceiling"
    )
    payload = dict(
        status=progress.status,
        progress=progress.model_dump(mode="json"),
        selected_checkpoint=None,
        selected_weights=None,
        latest_weights=descriptor.model_dump(mode="json"),
        model_identity=dict(
            model_config_json=canonical_json_bytes(
                resolve_pilot_config("phase4_smoke").config.neural
            ).decode(),
            model_state_sha256="a" * 64,
            source_revision="b" * 40,
        ),
        gate_eligible=False,
        artifact_hashes={descriptor.path: descriptor.sha256},
    )
    encoded = training_result_payload(payload, run_dir=tmp_path)
    assert encoded["schema_version"] == "phase4-training-result-v2"
    assert "artifact_hashes" not in encoded and "artifact_index" in encoded
    current = tmp_path / "training-result.json"
    current.write_bytes(canonical_json_bytes(encoded))
    assert load_training_result(tmp_path, current) == payload
    legacy = tmp_path / "legacy-result.json"
    original = canonical_json_bytes(payload)
    legacy.write_bytes(original)
    assert load_training_result(tmp_path, legacy) == payload
    assert legacy.read_bytes() == original
    assert training_result_payload(payload, run_dir=tmp_path) == encoded
