"""Durable atomic file publication."""

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from silent_cascade.errors import AtomicWriteError
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.validation import JsonValue


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
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    temp_path = _durable_temp(path, data, mode)
    try:
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except OSError as error:
        raise AtomicWriteError(
            "atomic replace failed",
            context={"path": str(path), "reason": str(error)},
        ) from error
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_create_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    temp_path = _durable_temp(path, data, mode)
    try:
        os.link(temp_path, path)
        temp_path.unlink()
        _fsync_directory(path.parent)
    except FileExistsError as error:
        raise AtomicWriteError(
            "artifact already exists",
            context={"path": str(path)},
        ) from error
    except OSError as error:
        raise AtomicWriteError(
            "atomic create failed",
            context={"path": str(path), "reason": str(error)},
        ) from error
    finally:
        temp_path.unlink(missing_ok=True)


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
