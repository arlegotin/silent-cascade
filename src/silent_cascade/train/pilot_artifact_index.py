"""Bounded training-result serialization; public readers still materialize maps.

Index validation streams every canonical JSONL entry through the final shard.
The immutable envelope replaces a potentially multi-million-entry JSON object;
it does not change retention, archive formats, or the public training result.
"""

import gzip
import hashlib
import io
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.train.evidence_types import Hash, decode_json, read_bytes
from silent_cascade.train.pilot_evidence_types import (
    MAX_BYTES,
    MAX_FILE_BYTES,
    MAX_ROWS,
    read_compact_rows,
    write_compact_shards,
)
from silent_cascade.validation import JsonValue, StrictModel

LEGACY_KEYS = frozenset(
    {
        "status",
        "progress",
        "selected_checkpoint",
        "selected_weights",
        "latest_weights",
        "artifact_hashes",
        "model_identity",
        "gate_eligible",
    }
)


def _path(root, name):
    if (
        not isinstance(name, str)
        or not name
        or Path(name).is_absolute()
        or any(part in {"", ".", "..", "frozen", "frozen_test"} for part in name.split("/"))
    ):
        raise ValueError("unsafe artifact index path")
    return root / name


class IndexEntry(StrictModel):
    path: str
    sha256: Hash


class IndexShard(StrictModel):
    path: str
    sha256: Hash
    rows: int = Field(gt=0, le=MAX_ROWS)
    decompressed_bytes: int = Field(gt=0, le=MAX_BYTES)


class PilotArtifactIndex(StrictModel):
    schema_version: Literal["phase4-artifact-index-v1"] = "phase4-artifact-index-v1"
    entry_count: int = Field(ge=0)
    entries_sha256: Hash
    shards: tuple[IndexShard, ...]


class TrainingResultEnvelope(StrictModel):
    schema_version: Literal["phase4-training-result-v2"] = "phase4-training-result-v2"
    status: str
    progress: dict[str, JsonValue]
    selected_checkpoint: dict[str, JsonValue] | None
    selected_weights: dict[str, JsonValue] | None
    latest_weights: dict[str, JsonValue]
    model_identity: dict[str, JsonValue]
    gate_eligible: bool
    artifact_index: PilotArtifactIndex


def write_artifact_index(*, root, prefix, entries, max_rows=MAX_ROWS, max_bytes=MAX_BYTES):
    prefix_path = _path(root, prefix)
    digest = hashlib.sha256()
    count = 0

    def records():
        nonlocal count
        previous = None
        for name, hashed in entries:
            _path(root, name)
            row = IndexEntry(path=name, sha256=hashed)
            if previous is not None and name <= previous:
                raise ValueError("artifact index order or duplicate path")
            previous = name
            count += 1
            digest.update(canonical_json_bytes(row) + b"\n")
            yield row.model_dump(mode="json")

    parts = write_compact_shards(prefix_path, records(), max_rows=max_rows, max_bytes=max_bytes)
    existing = {path.name for path in prefix_path.parent.glob(prefix_path.name + ".*.jsonl.gz")}
    if existing != {part.path for part in parts}:
        raise ValueError("duplicate artifact index publication has a different tail")
    shards = []
    for part in parts:
        path = prefix_path.parent / part.path
        size = sum(len(canonical_json_bytes(record)) + 1 for record in read_compact_rows(path))
        shards.append(
            IndexShard(
                path=path.relative_to(root).as_posix(),
                sha256=part.sha256,
                rows=part.rows,
                decompressed_bytes=size,
            )
        )
    return PilotArtifactIndex(
        entry_count=count, entries_sha256=digest.hexdigest(), shards=tuple(shards)
    )


def iter_artifact_index(root, index, *, evidence_context=None):
    """Validate every shard and entry, including global order and final totals."""
    index = PilotArtifactIndex.model_validate_json(canonical_json_bytes(index))
    from silent_cascade.archive.readers import evidence_path

    previous, count = None, 0
    digest = hashlib.sha256()
    seen_shards = set()
    try:
        for part in index.shards:
            if part.path in seen_shards:
                raise ValueError("duplicate artifact index shard")
            seen_shards.add(part.path)
            _path(root, part.path)
            with evidence_path(root, part.path, evidence_context=evidence_context) as path:
                compressed = read_bytes(path, limit=MAX_FILE_BYTES)
            if sha256_bytes(compressed) != part.sha256:
                raise ValueError("artifact index shard hash differs")
            rows, size = 0, 0
            with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
                while True:
                    raw = stream.readline(MAX_BYTES - size + 1)
                    if not raw:
                        break
                    size += len(raw)
                    rows += 1
                    if size > MAX_BYTES or rows > MAX_ROWS:
                        raise ValueError("artifact index shard limit exceeded")
                    row = IndexEntry.model_validate_json(
                        canonical_json_bytes(decode_json(raw, limit=MAX_BYTES))
                    )
                    if raw != canonical_json_bytes(row) + b"\n":
                        raise ValueError("artifact index entry is not canonical JSONL")
                    _path(root, row.path)
                    if previous is not None and row.path <= previous:
                        raise ValueError("artifact index global order or duplicate path")
                    previous = row.path
                    count += 1
                    digest.update(raw)
                    yield row.path, row.sha256
            if rows != part.rows or size != part.decompressed_bytes:
                raise ValueError("artifact index shard count/bytes differ")
        if count != index.entry_count or digest.hexdigest() != index.entries_sha256:
            raise ValueError("artifact index total count/stream hash differs")
    except Exception as error:
        raise ValueError(f"invalid artifact index: {error}") from error


def training_result_payload(result, *, run_dir):
    """Uniform v2 writer; attempt and root publications bind identical shards."""
    from silent_cascade.train.trainer import _json

    if is_dataclass(result):
        values = {field.name: getattr(result, field.name) for field in fields(result)}
    else:
        values = dict(result)
    if set(values) != LEGACY_KEYS:
        raise ValueError("invalid public training result fields")
    hashes = values.pop("artifact_hashes")
    values = _json(values)
    weights = _path(run_dir, values["latest_weights"]["path"])
    relative = weights.relative_to(run_dir)
    if len(relative.parts) < 2 or not relative.parts[0].startswith("attempt-"):
        raise ValueError("training index requires attempt-local latest weights")
    prefix = (relative.parent / "artifact-index").as_posix()
    index = write_artifact_index(
        root=run_dir, prefix=prefix, entries=((p, hashes[p]) for p in sorted(hashes))
    )
    return TrainingResultEnvelope.model_validate_json(
        canonical_json_bytes(values | {"artifact_index": index.model_dump(mode="json")})
    ).model_dump(mode="json")


def expand_training_result(run_dir, payload, *, evidence_context=None):
    """Compatibility boundary: returns the historical materialized dictionary."""
    if set(payload) == LEGACY_KEYS:
        return dict(payload)
    envelope = parse_training_envelope(payload)
    value = envelope.model_dump(mode="json", exclude={"schema_version", "artifact_index"})
    value["artifact_hashes"] = dict(
        iter_artifact_index(run_dir, envelope.artifact_index, evidence_context=evidence_context)
    )
    return value


def parse_training_envelope(payload):
    if set(payload) != set(TrainingResultEnvelope.model_fields):
        raise ValueError("invalid v2 training result fields")
    envelope = TrainingResultEnvelope.model_validate_json(canonical_json_bytes(payload))
    weights = _path(Path(), envelope.latest_weights["path"])
    prefix = (weights.parent / "artifact-index").as_posix()
    offset = 0
    for part in envelope.artifact_index.shards:
        if part.path != f"{prefix}.{offset:05d}.jsonl.gz":
            raise ValueError("training index descriptor root/order differs")
        offset += part.rows
    return envelope
