"""Compact acceptance attachments are bounded, unambiguous and score-derived."""

import gzip

import pytest


@pytest.mark.parametrize(
    "payload", [b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b'{"x":1e999}\n', b"[]\n", b"{broken}\n"]
)
def test_compact_rows_reject_ambiguous_json(tmp_path, payload):
    from silent_cascade.train.pilot_evidence_types import read_compact_rows

    path = tmp_path / "rows.jsonl.gz"
    path.write_bytes(gzip.compress(payload))
    with pytest.raises(ValueError):
        tuple(read_compact_rows(path))


def test_compact_rows_stream_limits_and_deterministic_roundtrip(tmp_path):
    from silent_cascade.train.pilot_evidence_types import read_compact_rows, write_compact_rows

    path = tmp_path / "rows.jsonl.gz"
    write_compact_rows(path, ({"index": i} for i in range(3)))
    assert tuple(read_compact_rows(path)) == tuple({"index": i} for i in range(3))
    with pytest.raises(ValueError, match="row limit"):
        tuple(read_compact_rows(path, max_rows=2))
    with pytest.raises(ValueError, match="byte limit"):
        tuple(read_compact_rows(path, max_bytes=8))
    with pytest.raises(ValueError, match="duplicate"):
        write_compact_rows(path, [{"different": True}])


def test_no_selected_checkpoint_is_not_replaced_with_latest():
    from silent_cascade.train.pilot_evidence import acceptance_failures

    failures = acceptance_failures(
        production=True,
        selected=False,
        suites={},
        repeat=False,
        pair_successes=0,
        coverage=(),
        numeric=False,
        offline=False,
    )
    assert "no_selected_checkpoint" in failures
    assert "missing_primary" in failures


def test_compact_shards_enforce_actual_bytes_preserving_order(tmp_path):
    from silent_cascade.train.pilot_evidence_types import read_compact_rows, write_compact_shards

    rows = [{"index": index, "payload": "x" * 30} for index in range(11)]
    shards = write_compact_shards(tmp_path / "rows", rows, max_bytes=140, max_rows=3)
    assert len(shards) > 3
    assert [row for part in shards for row in read_compact_rows(tmp_path / part.path)] == rows
    assert sum(part.rows for part in shards) == 11
    assert all(len(gzip.decompress((tmp_path / part.path).read_bytes())) <= 140 for part in shards)
    with pytest.raises(ValueError, match="single row"):
        write_compact_shards(tmp_path / "too-large", [{"payload": "x" * 200}], max_bytes=140)


def test_runtime_compute_cannot_be_rehashed_independently_of_observations():
    from silent_cascade.eval.compute import RuntimeCompute
    from silent_cascade.train.pilot_evidence import verify_compute_observations

    observations = [
        RuntimeCompute(forward_macs=2).model_dump(mode="json"),
        RuntimeCompute(forward_macs=5).model_dump(mode="json"),
    ]
    verify_compute_observations(RuntimeCompute(forward_macs=7), observations)
    with pytest.raises(ValueError, match="compute"):
        verify_compute_observations(RuntimeCompute(forward_macs=8), observations)


def test_compact_shard_reader_consumes_order_counts_and_late_corruption(tmp_path):
    from silent_cascade.hashing import sha256_bytes
    from silent_cascade.train.pilot_evidence import compact_attachment_records
    from silent_cascade.train.pilot_evidence_types import write_compact_shards

    rows = [{"index": index} for index in range(7)]
    shards = write_compact_shards(tmp_path / "stream", rows, max_rows=2)
    stream = compact_attachment_records(tmp_path, shards)
    assert iter(stream) is stream
    assert list(stream) == rows
    raw = gzip.compress(b'{"index":6}\n{"extra":true}\n', mtime=0)
    final = tmp_path / shards[-1].path
    final.write_bytes(raw)
    changed = (*shards[:-1], shards[-1].model_copy(update={"sha256": sha256_bytes(raw)}))
    with pytest.raises(ValueError, match="episode inventory"):
        list(compact_attachment_records(tmp_path, changed))


def test_gate_publication_refuses_file_limit_before_writing(tmp_path, monkeypatch):
    from silent_cascade.train import pilot_evidence_types as evidence

    monkeypatch.setattr(evidence, "MAX_FILE_BYTES", 50)
    with pytest.raises(ValueError, match="gate file limit"):
        evidence.publish_gate_artifact(tmp_path / "gate.json", {"payload": "x" * 100})
    assert not (tmp_path / "gate.json").exists()


def test_gate_reader_enforces_committed_file_limit(tmp_path, monkeypatch):
    from silent_cascade.train import pilot_evidence

    monkeypatch.setattr(pilot_evidence, "MAX_FILE_BYTES", 50, raising=False)
    path = tmp_path / "oversized.json"
    path.write_text('{"payload":"' + "x" * 100 + '"}')
    with pytest.raises(ValueError, match=r"archive\.byte_limit"):
        pilot_evidence.verify_phase4_gate_artifact(path, repo_root=tmp_path)
