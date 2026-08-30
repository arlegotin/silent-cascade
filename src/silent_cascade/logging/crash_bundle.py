"""Atomic, replay-oriented crash bundle manifests."""

import re
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import field_validator

from silent_cascade.errors import AtomicWriteError, CrashBundleError, SilentCascadeError
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.io import atomic_create_bytes
from silent_cascade.validation import JsonValue, StrictModel

_BUNDLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class CrashContext(StrictModel):
    seed: int | None = None
    episode_public_id: str | None = None
    checkpoint_ref: str | None = None
    config_sha256: str | None = None
    source_revision: str | None = None
    last_events: tuple[dict[str, JsonValue], ...] = ()


class CrashBundleManifest(StrictModel):
    schema_version: Literal[1] = 1
    bundle_id: str
    created_at_utc: datetime
    error: dict[str, JsonValue]
    traceback_text: str
    context: CrashContext

    @field_validator("schema_version", mode="before")
    @classmethod
    def _exact_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be exactly integer 1")
        return value


@dataclass(frozen=True, slots=True)
class CrashBundleArtifact:
    path: Path
    sha256: str


def write_crash_bundle(
    root: Path,
    *,
    error: SilentCascadeError,
    context: CrashContext,
    bundle_id: str | None = None,
    now: datetime | None = None,
) -> CrashBundleArtifact:
    resolved_id = bundle_id or uuid4().hex
    if not _BUNDLE_ID.fullmatch(resolved_id):
        raise CrashBundleError("invalid crash bundle id", context={"bundle_id": resolved_id})
    created_at = now or datetime.now(UTC)
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise CrashBundleError("crash bundle timestamp must be timezone-aware")

    bounded_context = context.model_copy(update={"last_events": context.last_events[-20:]})
    manifest = CrashBundleManifest(
        bundle_id=resolved_id,
        created_at_utc=created_at.astimezone(UTC),
        error=error.to_payload(),
        traceback_text="".join(traceback.format_exception(error)),
        context=bounded_context,
    )
    payload = canonical_json_bytes(manifest) + b"\n"
    destination = root / f"{resolved_id}.json"
    try:
        atomic_create_bytes(destination, payload, mode=0o600)
    except AtomicWriteError as write_error:
        write_context = dict(write_error.context)
        write_context.setdefault("path", str(destination))
        if write_error.message == "artifact already exists":
            raise CrashBundleError(
                "crash bundle already exists", context=write_context
            ) from write_error
        if write_context.get("published") is True:
            raise CrashBundleError(
                "crash bundle publication durability unconfirmed",
                context=write_context,
            ) from write_error
        raise CrashBundleError(
            "crash bundle publication failed",
            context={**write_context, "reason": str(write_error)},
        ) from write_error
    return CrashBundleArtifact(path=destination, sha256=sha256_bytes(payload))
