import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from silent_cascade.errors import AtomicWriteError, CrashBundleError, DynamicsError
from silent_cascade.hashing import sha256_file
from silent_cascade.logging.crash_bundle import (
    CrashBundleManifest,
    CrashContext,
    write_crash_bundle,
)


def test_crash_bundle_contains_context_and_only_last_twenty_events(
    tmp_path: Path,
) -> None:
    events = tuple({"event_id": index} for index in range(25))
    context = CrashContext(
        seed=23,
        episode_public_id="episode-public-7",
        checkpoint_ref="sha256:checkpoint",
        config_sha256="a" * 64,
        source_revision="deadbeef",
        last_events=events,
    )
    artifact = write_crash_bundle(
        tmp_path,
        error=DynamicsError("non-finite state", context={"tensor": "z_fast"}),
        context=context,
        bundle_id="bundle-001",
        now=datetime(2026, 8, 30, 12, 0, tzinfo=UTC),
    )
    payload = json.loads(artifact.path.read_text(encoding="utf-8"))
    assert artifact.sha256 == sha256_file(artifact.path)
    assert payload["error"]["code"] == "dynamics_error"
    assert "DynamicsError: non-finite state" in payload["traceback_text"]
    assert payload["context"]["seed"] == 23
    assert payload["context"]["episode_public_id"] == "episode-public-7"
    assert [event["event_id"] for event in payload["context"]["last_events"]] == list(range(5, 25))


def test_crash_bundle_refuses_duplicate_explicit_id(tmp_path: Path) -> None:
    context = CrashContext()
    error = DynamicsError("failure")
    write_crash_bundle(tmp_path, error=error, context=context, bundle_id="same-id")
    with pytest.raises(CrashBundleError, match="already exists"):
        write_crash_bundle(tmp_path, error=error, context=context, bundle_id="same-id")


def test_crash_bundle_rejects_path_like_bundle_id(tmp_path: Path) -> None:
    with pytest.raises(CrashBundleError, match="invalid crash bundle id"):
        write_crash_bundle(
            tmp_path,
            error=DynamicsError("failure"),
            context=CrashContext(),
            bundle_id="../escape",
        )


def test_crash_bundle_rejects_boolean_schema_version() -> None:
    with pytest.raises(ValidationError):
        CrashBundleManifest(
            schema_version=True,
            bundle_id="bundle-001",
            created_at_utc=datetime(2026, 8, 30, tzinfo=UTC),
            error={"code": "dynamics_error"},
            traceback_text="traceback",
            context=CrashContext(),
        )


def test_crash_bundle_distinguishes_published_durability_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "bundle-001.json"

    def fail_after_publication(*args: object, **kwargs: object) -> None:
        raise AtomicWriteError(
            "published but durability unconfirmed",
            context={"path": str(destination), "published": True},
        )

    monkeypatch.setattr(
        "silent_cascade.logging.crash_bundle.atomic_create_bytes",
        fail_after_publication,
    )
    with pytest.raises(CrashBundleError, match="durability unconfirmed") as caught:
        write_crash_bundle(
            tmp_path,
            error=DynamicsError("failure"),
            context=CrashContext(),
            bundle_id="bundle-001",
        )
    assert caught.value.context["published"] is True
    assert caught.value.context["path"] == str(destination)
