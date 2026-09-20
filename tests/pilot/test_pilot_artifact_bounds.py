"""Real file-boundary regressions without scientific fitting or acceptance claims."""

import hashlib
from types import SimpleNamespace

import pytest

MIB = 1024**2


def _file_sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _journal_bytes(size):
    from silent_cascade.hashing import canonical_json_bytes

    empty = canonical_json_bytes({"padding": "", "prior": None})
    raw = canonical_json_bytes({"padding": "x" * (size - len(empty)), "prior": None})
    assert len(raw) == size
    return raw


def _training_with_journal(digest):
    from silent_cascade.train.pilot_state import PilotProgress

    return {
        "gate_eligible": False,
        "latest_weights": None,
        "model_identity": {},
        "progress": PilotProgress(journal_sha256=digest).model_dump(mode="json"),
        "selected_checkpoint": None,
        "selected_weights": None,
    }


def test_duplicate_publication_compares_equal_payload_above_manifest_limit(tmp_path):
    from silent_cascade.train.pilot_data import _publish_pilot_bytes

    path = tmp_path / "large.bin"
    payload = b"a" * (64 * MIB + 1)
    _publish_pilot_bytes(path, payload)
    _publish_pilot_bytes(path, payload)
    before = _file_sha256(path)
    with pytest.raises(ValueError, match="refusing overwrite"):
        _publish_pilot_bytes(path, payload[:-1] + b"b")
    assert path.stat().st_size == len(payload)
    assert _file_sha256(path) == before


def test_duplicate_publication_handles_empty_payload(tmp_path):
    from silent_cascade.train.pilot_data import _publish_pilot_bytes

    path = tmp_path / "empty.bin"
    _publish_pilot_bytes(path, b"")
    _publish_pilot_bytes(path, b"")
    assert path.read_bytes() == b""


def test_duplicate_publication_rejects_destination_larger_than_expected(tmp_path):
    from silent_cascade.train.pilot_data import _publish_pilot_bytes

    path = tmp_path / "existing.bin"
    path.write_bytes(b"ab")
    with pytest.raises(ValueError, match=r"archive\.byte_limit"):
        _publish_pilot_bytes(path, b"a")
    assert path.read_bytes() == b"ab"


def test_gate_publisher_accepts_schema_valid_payload_above_manifest_limit(tmp_path):
    from silent_cascade.hashing import canonical_json_bytes
    from silent_cascade.train.pilot_evidence_types import (
        MAX_FILE_BYTES,
        Phase4GateArtifact,
        publish_gate_artifact,
    )
    from silent_cascade.train.pilot_provenance import PilotSourceIdentity

    digest = "a" * 64
    source = PilotSourceIdentity(
        source_commit="b" * 40,
        source_files={},
        source_sha256=digest,
        plan_revision="c" * 40,
        plan_sha256=digest,
        spec_sha256=digest,
        config_sha256=digest,
    )
    # This is a serialization/publication fixture, not a scientific acceptance artifact.
    artifact = Phase4GateArtifact(
        outcome="passed",
        source=source,
        config_canonical_json="{}",
        selected_checkpoint_sha256=digest,
        selected_weights_sha256=digest,
        model_state_sha256=digest,
        training_result={"serialization_padding": "x" * (64 * MIB)},
        workload={},
        suites={},
        attachments={},
        upstream_artifact_hashes={},
        execution_evidence={},
        continuation_evidence=(),
        pair_rows=(),
        repeat_equal=True,
        numeric_passed=True,
        offline_passed=True,
        numeric_evidence=None,
        offline_evidence=None,
        local_verification=None,
        coverage=(),
        failures=(),
        raw_regeneration_command="serialization fixture only",
    )
    raw = canonical_json_bytes(artifact)
    assert 64 * MIB < len(raw) < MAX_FILE_BYTES
    path = tmp_path / "gate.json"
    publish_gate_artifact(path, artifact)
    assert path.stat().st_size == len(raw)
    assert _file_sha256(path) == hashlib.sha256(raw).hexdigest()


def test_available_training_authenticates_journal_above_metadata_limit(tmp_path):
    from silent_cascade.hashing import sha256_bytes
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_evidence import _verify_available_training

    raw = _journal_bytes(8 * MIB + 1)
    digest = sha256_bytes(raw)
    (tmp_path / f"journal-{digest}.json").write_bytes(raw)
    unavailable = []
    _verify_available_training(
        tmp_path,
        training=_training_with_journal(digest),
        config=resolve_pilot_config("phase4_smoke"),
        source=SimpleNamespace(source_commit="a" * 40),
        unavailable=unavailable,
    )
    assert unavailable == ["training.durable_journal"]


def test_available_training_rejects_changed_large_journal_hash(tmp_path):
    from silent_cascade.hashing import sha256_bytes
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_evidence import _verify_available_training

    expected = _journal_bytes(8 * MIB + 1)
    digest = sha256_bytes(expected)
    (tmp_path / f"journal-{digest}.json").write_bytes(expected[:-1] + b" ")
    with pytest.raises(ValueError, match="journal hash differs"):
        _verify_available_training(
            tmp_path,
            training=_training_with_journal(digest),
            config=resolve_pilot_config("phase4_smoke"),
            source=SimpleNamespace(source_commit="a" * 40),
            unavailable=[],
        )


def test_available_training_rejects_malformed_large_journal_json(tmp_path):
    from silent_cascade.hashing import sha256_bytes
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_evidence import _verify_available_training

    raw = b'{"padding":"' + b"x" * (8 * MIB) + b'","prior":'
    digest = sha256_bytes(raw)
    (tmp_path / f"journal-{digest}.json").write_bytes(raw)
    with pytest.raises(ValueError, match="invalid evidence JSON"):
        _verify_available_training(
            tmp_path,
            training=_training_with_journal(digest),
            config=resolve_pilot_config("phase4_smoke"),
            source=SimpleNamespace(source_commit="a" * 40),
            unavailable=[],
        )


def test_available_training_rejects_journal_above_writer_limit(tmp_path):
    from silent_cascade.errors import ArtifactIntegrityError
    from silent_cascade.hashing import sha256_bytes
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_evidence import _verify_available_training

    raw = _journal_bytes(64 * MIB + 1)
    digest = sha256_bytes(raw)
    (tmp_path / f"journal-{digest}.json").write_bytes(raw)
    with pytest.raises(ArtifactIntegrityError, match=r"archive\.byte_limit"):
        _verify_available_training(
            tmp_path,
            training=_training_with_journal(digest),
            config=resolve_pilot_config("phase4_smoke"),
            source=SimpleNamespace(source_commit="a" * 40),
            unavailable=[],
        )
