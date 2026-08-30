"""Durable atomic file publication."""

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from silent_cascade.errors import AtomicWriteError
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.validation import JsonValue


class _PreparationCleanupError(Exception):
    def __init__(self, primary: Exception, temp_path: Path, cleanup_reason: str) -> None:
        super().__init__(str(primary))
        self.primary = primary
        self.temp_path = temp_path
        self.cleanup_reason = cleanup_reason


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_temp(path: Path, data: bytes, mode: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode)
        return temp_path
    except Exception as error:
        if cleanup_reason := _cleanup_temp(temp_path):
            raise _PreparationCleanupError(error, temp_path, cleanup_reason) from error
        raise
    except BaseException as error:
        try:
            cleanup_reason = _cleanup_temp(temp_path)
        except BaseException as cleanup_error:
            cleanup_reason = str(cleanup_error)
        if cleanup_reason:
            error.add_note(f"temporary cleanup failed for {temp_path}: {cleanup_reason}")
        raise


def _cleanup_temp(path: Path) -> str | None:
    try:
        path.unlink(missing_ok=True)
    except Exception as error:
        return str(error)
    return None


def atomic_write_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    try:
        temp_path = _durable_temp(path, data, mode)
    except _PreparationCleanupError as error:
        raise AtomicWriteError(
            "atomic preparation failed",
            context={
                "path": str(path),
                "published": False,
                "reason": str(error.primary),
                "cleanup_reason": error.cleanup_reason,
                "temp_path": str(error.temp_path),
            },
        ) from error.primary
    except OSError as error:
        raise AtomicWriteError(
            "atomic preparation failed",
            context={"path": str(path), "published": False, "reason": str(error)},
        ) from error
    published = False
    try:
        os.replace(temp_path, path)
        published = True
        _fsync_directory(path.parent)
    except OSError as error:
        context = {"path": str(path), "published": published, "reason": str(error)}
        if cleanup_reason := _cleanup_temp(temp_path):
            context["cleanup_reason"] = cleanup_reason
            context["temp_path"] = str(temp_path)
        raise AtomicWriteError(
            "published but durability unconfirmed" if published else "atomic replace failed",
            context=context,
        ) from error
    if cleanup_reason := _cleanup_temp(temp_path):
        raise AtomicWriteError(
            "published but durability unconfirmed" if published else "atomic cleanup failed",
            context={
                "path": str(path),
                "published": published,
                "cleanup_reason": cleanup_reason,
                "temp_path": str(temp_path),
            },
        )


def atomic_create_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    try:
        temp_path = _durable_temp(path, data, mode)
    except _PreparationCleanupError as error:
        raise AtomicWriteError(
            "atomic preparation failed",
            context={
                "path": str(path),
                "published": False,
                "reason": str(error.primary),
                "cleanup_reason": error.cleanup_reason,
                "temp_path": str(error.temp_path),
            },
        ) from error.primary
    except OSError as error:
        raise AtomicWriteError(
            "atomic preparation failed",
            context={"path": str(path), "published": False, "reason": str(error)},
        ) from error
    published = False
    try:
        os.link(temp_path, path)
        published = True
        temp_path.unlink()
        _fsync_directory(path.parent)
    except FileExistsError as error:
        context = {"path": str(path), "published": False}
        if cleanup_reason := _cleanup_temp(temp_path):
            context["cleanup_reason"] = cleanup_reason
            context["temp_path"] = str(temp_path)
        raise AtomicWriteError(
            "artifact already exists",
            context=context,
        ) from error
    except OSError as error:
        context = {"path": str(path), "published": published, "reason": str(error)}
        if published and temp_path.exists():
            context["cleanup_reason"] = str(error)
            context["temp_path"] = str(temp_path)
        elif cleanup_reason := _cleanup_temp(temp_path):
            context["cleanup_reason"] = cleanup_reason
            context["temp_path"] = str(temp_path)
        raise AtomicWriteError(
            "published but durability unconfirmed" if published else "atomic create failed",
            context=context,
        ) from error
    if cleanup_reason := _cleanup_temp(temp_path):
        raise AtomicWriteError(
            "published but durability unconfirmed",
            context={
                "path": str(path),
                "published": published,
                "cleanup_reason": cleanup_reason,
                "temp_path": str(temp_path),
            },
        )


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o644,
) -> None:
    atomic_write_bytes(path, text.encode(encoding), mode=mode)


def atomic_write_json(
    path: Path,
    value: BaseModel | dict[str, JsonValue],
    *,
    mode: int = 0o644,
) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n", mode=mode)
