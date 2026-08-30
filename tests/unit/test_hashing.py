from pathlib import Path

import pytest

from silent_cascade.errors import ArtifactIntegrityError
from silent_cascade.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    verify_file_sha256,
)


def test_canonical_json_is_compact_and_insertion_order_independent() -> None:
    first = {"z": [2, 1], "a": {"enabled": True}}
    second = {"a": {"enabled": True}, "z": [2, 1]}
    expected = b'{"a":{"enabled":true},"z":[2,1]}'
    assert canonical_json_bytes(first) == canonical_json_bytes(second) == expected
    assert sha256_bytes(expected) == sha256_bytes(canonical_json_bytes(first))


def test_file_hash_verification_detects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original")
    expected = sha256_file(artifact)
    verify_file_sha256(artifact, expected)
    artifact.write_bytes(b"changed")
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        verify_file_sha256(artifact, expected)
