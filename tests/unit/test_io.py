import os
from pathlib import Path

import pytest

from silent_cascade.errors import ArtifactIntegrityError, AtomicWriteError
from silent_cascade.io import (
    atomic_create_bytes,
    atomic_write_bytes,
    atomic_write_json,
)


@pytest.mark.parametrize("payload,limit", ((b"", 0), (b"abc", 3), (b"abc", 8)))
def test_bounded_regular_read_accepts_exact_and_short_files(tmp_path, payload, limit):
    from silent_cascade import io

    path = tmp_path / "report.json"
    path.write_bytes(payload)
    read = getattr(io, "read_bounded_regular_bytes", None)
    assert callable(read), "bounded regular-file read is missing"
    assert read(path, max_bytes=limit) == payload


@pytest.mark.parametrize("race", ("oversized", "growth", "replacement"))
def test_bounded_regular_read_uses_one_opened_object_and_a_sentinel(tmp_path, monkeypatch, race):
    from silent_cascade import io

    path = tmp_path / "report.json"
    path.write_bytes(b"x" * 32 if race == "oversized" else b"abc")
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b"x" * 32)
    read = getattr(io, "read_bounded_regular_bytes", None)
    assert callable(read), "bounded regular-file read is missing"
    real_fstat, real_read = os.fstat, os.read
    consumed = 0
    changed = False

    def observe_then_change(descriptor):
        nonlocal changed
        observed = real_fstat(descriptor)
        if not changed:
            changed = True
            if race == "growth":
                path.write_bytes(b"x" * 32)
            elif race == "replacement":
                os.replace(replacement, path)
        return observed

    def counted_read(descriptor, amount):
        nonlocal consumed
        chunk = real_read(descriptor, amount)
        consumed += len(chunk)
        return chunk

    monkeypatch.setattr(os, "fstat", observe_then_change)
    monkeypatch.setattr(os, "read", counted_read)
    if race == "replacement":
        assert read(path, max_bytes=8) == b"abc"
        assert path.read_bytes() == b"x" * 32
    else:
        with pytest.raises(ArtifactIntegrityError):
            read(path, max_bytes=8)
    assert consumed <= 9


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


def test_atomic_write_preparation_failure_is_normalized_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"old")

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected preparation failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(AtomicWriteError, match="atomic preparation failed") as raised:
        atomic_write_bytes(destination, b"new")
    assert raised.value.context["published"] is False
    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_atomic_write_preparation_failure_keeps_primary_error_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"old")

    def fail_preparation(_descriptor: int) -> None:
        raise OSError("injected preparation fsync failure")

    original_unlink = Path.unlink

    def fail_temp_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.endswith(".tmp"):
            raise OSError("injected cleanup failure")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", fail_preparation)
    monkeypatch.setattr(Path, "unlink", fail_temp_unlink)
    with pytest.raises(AtomicWriteError, match="atomic preparation failed") as raised:
        atomic_write_bytes(destination, b"new")

    assert raised.value.context["published"] is False
    assert raised.value.context["reason"] == "injected preparation fsync failure"
    assert raised.value.context["cleanup_reason"] == "injected cleanup failure"
    temp_path = raised.value.context["temp_path"]
    assert isinstance(temp_path, str)
    orphan = Path(temp_path)
    assert orphan.parent == tmp_path
    assert list(tmp_path.glob(".artifact.bin.*.tmp")) == [orphan]
    assert destination.read_bytes() == b"old"


def test_atomic_create_preparation_failure_keeps_primary_error_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"

    def fail_preparation(_descriptor: int) -> None:
        raise OSError("injected preparation fsync failure")

    original_unlink = Path.unlink

    def fail_temp_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.endswith(".tmp"):
            raise OSError("injected cleanup failure")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", fail_preparation)
    monkeypatch.setattr(Path, "unlink", fail_temp_unlink)
    with pytest.raises(AtomicWriteError, match="atomic preparation failed") as raised:
        atomic_create_bytes(destination, b"new")

    assert raised.value.context["published"] is False
    assert raised.value.context["reason"] == "injected preparation fsync failure"
    assert raised.value.context["cleanup_reason"] == "injected cleanup failure"
    temp_path = raised.value.context["temp_path"]
    assert isinstance(temp_path, str)
    orphan = Path(temp_path)
    assert orphan.parent == tmp_path
    assert list(tmp_path.glob(".artifact.bin.*.tmp")) == [orphan]
    assert not destination.exists()


def test_atomic_write_non_os_preparation_failure_keeps_primary_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"old")
    primary = RuntimeError("injected preparation runtime failure")

    def fail_preparation(_path: Path, _mode: int) -> None:
        raise primary

    original_unlink = Path.unlink

    def fail_temp_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.endswith(".tmp"):
            raise PermissionError("injected cleanup failure")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(os, "chmod", fail_preparation)
    monkeypatch.setattr(Path, "unlink", fail_temp_unlink)
    with pytest.raises(AtomicWriteError, match="atomic preparation failed") as raised:
        atomic_write_bytes(destination, b"new")

    assert raised.value.context["published"] is False
    assert raised.value.context["reason"] == "injected preparation runtime failure"
    assert raised.value.context["cleanup_reason"] == "injected cleanup failure"
    temp_path = raised.value.context["temp_path"]
    assert isinstance(temp_path, str)
    orphan = Path(temp_path)
    assert orphan.parent == tmp_path
    assert list(tmp_path.glob(".artifact.bin.*.tmp")) == [orphan]
    assert destination.read_bytes() == b"old"
    assert raised.value.__cause__ is primary


def test_atomic_create_non_os_preparation_failure_keeps_primary_when_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"
    primary = TypeError("injected preparation type failure")

    def fail_preparation(_path: Path, _mode: int) -> None:
        raise primary

    original_unlink = Path.unlink

    def fail_temp_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.endswith(".tmp"):
            raise PermissionError("injected cleanup failure")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(os, "chmod", fail_preparation)
    monkeypatch.setattr(Path, "unlink", fail_temp_unlink)
    with pytest.raises(AtomicWriteError, match="atomic preparation failed") as raised:
        atomic_create_bytes(destination, b"new")

    assert raised.value.context["published"] is False
    assert raised.value.context["reason"] == "injected preparation type failure"
    assert raised.value.context["cleanup_reason"] == "injected cleanup failure"
    temp_path = raised.value.context["temp_path"]
    assert isinstance(temp_path, str)
    orphan = Path(temp_path)
    assert orphan.parent == tmp_path
    assert list(tmp_path.glob(".artifact.bin.*.tmp")) == [orphan]
    assert not destination.exists()
    assert raised.value.__cause__ is primary


def test_atomic_write_post_publication_failure_reports_uncertain_durability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"

    def fail_directory_fsync(_path: Path) -> None:
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr("silent_cascade.io._fsync_directory", fail_directory_fsync)
    with pytest.raises(AtomicWriteError, match="published but durability unconfirmed") as raised:
        atomic_write_bytes(destination, b"new")
    assert raised.value.context["published"] is True
    assert destination.read_bytes() == b"new"
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_atomic_create_post_publication_failure_reports_uncertain_durability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"

    def fail_directory_fsync(_path: Path) -> None:
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr("silent_cascade.io._fsync_directory", fail_directory_fsync)
    with pytest.raises(AtomicWriteError, match="published but durability unconfirmed") as raised:
        atomic_create_bytes(destination, b"new")
    assert raised.value.context["published"] is True
    assert destination.read_bytes() == b"new"
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_atomic_create_removes_temp_before_directory_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"
    observed: list[bool] = []
    original_fsync = __import__("silent_cascade.io", fromlist=["_fsync_directory"])._fsync_directory

    def record_directory_fsync(path: Path) -> None:
        observed.append(bool(list(path.glob(".artifact.bin.*.tmp"))))
        original_fsync(path)

    monkeypatch.setattr("silent_cascade.io._fsync_directory", record_directory_fsync)
    atomic_create_bytes(destination, b"new")
    assert observed == [False]


def test_atomic_write_cleanup_failure_preserves_replace_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"old")

    monkeypatch.setattr(os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("replace")))
    original_unlink = Path.unlink

    def fail_temp_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.endswith(".tmp"):
            raise OSError("cleanup")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temp_unlink)
    with pytest.raises(AtomicWriteError, match="atomic replace failed") as raised:
        atomic_write_bytes(destination, b"new")
    assert raised.value.context["published"] is False
    assert raised.value.context["cleanup_reason"] == "cleanup"
    assert raised.value.context["temp_path"]
    assert destination.read_bytes() == b"old"
    assert list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_atomic_create_cleanup_failure_preserves_post_link_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"
    original_unlink = Path.unlink

    def fail_temp_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.endswith(".tmp"):
            raise OSError("cleanup")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temp_unlink)
    with pytest.raises(AtomicWriteError, match="published but durability unconfirmed") as raised:
        atomic_create_bytes(destination, b"new")
    assert raised.value.context["published"] is True
    assert raised.value.context["cleanup_reason"] == "cleanup"
    assert raised.value.context["temp_path"]
    assert destination.read_bytes() == b"new"
    assert list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_atomic_create_collision_cleanup_failure_includes_temp_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"first")
    original_unlink = Path.unlink

    def fail_temp_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self.name.endswith(".tmp"):
            raise OSError("cleanup")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temp_unlink)
    with pytest.raises(AtomicWriteError, match="already exists") as raised:
        atomic_create_bytes(destination, b"second")
    assert raised.value.context["published"] is False
    assert raised.value.context["cleanup_reason"] == "cleanup"
    temp_path = raised.value.context["temp_path"]
    assert isinstance(temp_path, str)
    assert Path(temp_path).exists()
    assert destination.read_bytes() == b"first"
