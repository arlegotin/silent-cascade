import os
from pathlib import Path

import pytest

from silent_cascade.errors import AtomicWriteError
from silent_cascade.io import atomic_create_bytes, atomic_write_bytes, atomic_write_json


def test_atomic_write_replaces_destination_with_complete_contents(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"old")
    atomic_write_bytes(destination, b"new")
    assert destination.read_bytes() == b"new"
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_atomic_write_failure_preserves_destination_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"old")

    def fail_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(AtomicWriteError, match="atomic replace failed"):
        atomic_write_bytes(destination, b"new")
    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_atomic_create_refuses_to_clobber_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "immutable.bin"
    destination.write_bytes(b"first")
    with pytest.raises(AtomicWriteError, match="already exists"):
        atomic_create_bytes(destination, b"second")
    assert destination.read_bytes() == b"first"


def test_atomic_write_json_uses_canonical_encoding(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    atomic_write_json(destination, {"z": 2, "a": 1})
    assert destination.read_bytes() == b'{"a":1,"z":2}\n'
