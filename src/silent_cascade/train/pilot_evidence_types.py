"""Closed Phase4 envelopes; compact JSONL is limited during decompression.

Each attachment permits at most 50,000 rows and 128 MiB decompressed bytes.
Compressed files must be strictly smaller than 100 MiB. Readers reject links,
duplicate JSON keys, nonfinite numbers, trailing corruption and oversized lines.
"""

import gzip
import io
from pathlib import Path
from typing import Literal

from pydantic import Field

from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.train.evidence_types import decode_json, read_bytes
from silent_cascade.train.pilot_data import _publish_pilot_bytes
from silent_cascade.train.pilot_provenance import PilotSourceIdentity
from silent_cascade.validation import JsonValue, StrictModel

MAX_ROWS = 50_000
MAX_BYTES = 128 * 1024 * 1024
MAX_FILE_BYTES = 100 * 1024 * 1024 - 1
REQUIRED_COVERAGE = (
    "boundary:mid_flow",
    "boundary:fact",
    "boundary:activate",
    "boundary:recall",
    "boundary:compose",
    "boundary:act",
    "mode:observing",
    "mode:searching",
    "mode:have_memory",
    "mode:holding_hazard",
    "mode:quiescent",
    "mode:terminal",
    "variant:positive",
    "variant:safe_negative",
    "variant:disconnected_negative",
)


def strict_json(path, *, limit=MAX_BYTES):
    try:
        return decode_json(read_bytes(path, limit=limit), limit=limit)
    except Exception as error:
        raise ValueError(f"invalid evidence JSON: {error}") from error


def read_compact_rows(path: Path, *, max_rows=MAX_ROWS, max_bytes=MAX_BYTES):
    raw = read_bytes(path, limit=MAX_FILE_BYTES)
    total = 0
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            for index in range(max_rows + 1):
                line = stream.readline(max_bytes - total + 1)
                if not line:
                    return
                if index == max_rows:
                    raise ValueError("compact attachment row limit exceeded")
                total += len(line)
                if total > max_bytes:
                    raise ValueError("compact attachment byte limit exceeded")
                yield decode_json(line, limit=max_bytes)
    except Exception as error:
        raise ValueError(f"invalid compact evidence: {error}") from error


def write_compact_rows(path: Path, rows):
    buffer = io.BytesIO()
    total = 0
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as stream:
        for index, row in enumerate(rows):
            raw = canonical_json_bytes(row) + b"\n"
            total += len(raw)
            if index >= MAX_ROWS or total > MAX_BYTES:
                raise ValueError("compact attachment limit exceeded")
            decode_json(raw, limit=MAX_BYTES)
            stream.write(raw)
    raw = buffer.getvalue()
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("compact file limit exceeded")
    if path.exists():
        if read_bytes(path, limit=MAX_FILE_BYTES) != raw:
            raise ValueError("duplicate compact publication differs")
        return sha256_bytes(raw)
    _publish_pilot_bytes(path, raw)
    return sha256_bytes(raw)


def publish_gate_artifact(path, artifact):
    raw = canonical_json_bytes(artifact)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("gate file limit exceeded")
    _publish_pilot_bytes(path, raw)


class Attachment(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: int = Field(ge=0, le=MAX_ROWS)


class LocalVerificationReceipt(StrictModel):
    schema_version: Literal["phase4-local-verification-v1"] = "phase4-local-verification-v1"
    command: tuple[Literal["make"], Literal["verify"]] = ("make", "verify")
    verification_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    after_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    before_inventory: dict[str, str]
    after_inventory: dict[str, str]
    source_unchanged: bool
    returncode: int
    stdout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stderr_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    process_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def write_compact_shards(prefix: Path, rows, *, max_bytes=MAX_BYTES, max_rows=1000):
    """Split deterministically before either byte or row limit; never drop a row."""
    if not 0 < max_bytes <= MAX_BYTES or not 0 < max_rows <= MAX_ROWS:
        raise ValueError("invalid compact shard limits")
    result, pending, total, start = [], [], 0, 0

    def publish():
        path = prefix.parent / f"{prefix.name}.{start:05d}.jsonl.gz"
        result.append(
            Attachment(path=path.name, sha256=write_compact_rows(path, pending), rows=len(pending))
        )

    for row in rows:
        size = len(canonical_json_bytes(row)) + 1
        if size > max_bytes:
            raise ValueError("single row exceeds compact byte limit")
        if pending and (total + size > max_bytes or len(pending) == max_rows):
            publish()
            start += len(pending)
            pending, total = [], 0
        pending.append(row)
        total += size
    if pending:
        publish()
    return tuple(result)


class Phase4GateArtifact(StrictModel):
    schema_version: Literal["phase4-autonomous-gate-v1"] = "phase4-autonomous-gate-v1"
    execution_status: Literal["DONE"] = "DONE"
    outcome: Literal["passed", "failed", "debug_non_acceptance"]
    source: PilotSourceIdentity
    config_canonical_json: str
    selected_checkpoint_sha256: str | None
    selected_weights_sha256: str | None
    model_state_sha256: str | None
    training_result: dict[str, JsonValue]
    workload: dict[str, JsonValue]
    suites: dict[str, dict[str, int]]
    attachments: dict[str, tuple[Attachment, ...]]
    upstream_artifact_hashes: dict[str, str]
    execution_evidence: dict[str, JsonValue]
    continuation_evidence: tuple[dict[str, JsonValue], ...]
    pair_rows: tuple[dict[str, JsonValue], ...]
    repeat_equal: bool
    numeric_passed: bool
    offline_passed: bool
    numeric_evidence: dict[str, JsonValue] | None
    offline_evidence: dict[str, JsonValue] | None
    local_verification: dict[str, JsonValue] | None
    coverage: tuple[str, ...]
    failures: tuple[str, ...]
    foundation_model_calls: Literal[0] = 0
    raw_regeneration_command: str
    proof_scope: Literal["recorded_evidence_integrity; no fresh neural rerun"] = (
        "recorded_evidence_integrity; no fresh neural rerun"
    )


class Phase4DeliveryMap(StrictModel):
    schema_version: Literal["phase4-delivery-map-v1"] = "phase4-delivery-map-v1"
    gate: Literal["manifests/validation/phase4/autonomous-gate-v1.json"] = (
        "manifests/validation/phase4/autonomous-gate-v1.json"
    )
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    gate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
