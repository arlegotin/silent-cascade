"""Canonical serialization and SHA-256 helpers."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

from silent_cascade.errors import ArtifactIntegrityError
from silent_cascade.validation import JsonValue


def canonical_json_bytes(
    value: BaseModel | Mapping[str, JsonValue],
) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return encoded.encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ArtifactIntegrityError(
            "artifact hash mismatch",
            context={"path": str(path), "expected": expected, "actual": actual},
        )
