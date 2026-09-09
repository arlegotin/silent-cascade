"""Shared archive descriptors preserve bounded reads and no-follow semantics."""

import os

import pytest


def failure(field):
    return ValueError(field)


@pytest.mark.parametrize("payload,limit", [(b"", 1), (b"abc", 3), (b"abc", 4)])
def test_shared_archive_reader_preserves_bytes_at_limit(tmp_path, payload, limit):
    from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at

    path = tmp_path / "artifact"
    path.write_bytes(payload)
    with archive_parent(path, error_factory=failure) as (parent, name):
        assert read_archive_at(parent, name, max_bytes=limit, error_factory=failure) == payload
    with pytest.raises(OSError):
        os.fstat(parent)


@pytest.mark.parametrize(
    "kind", ["symlink", "ancestor_symlink", "fifo", "directory", "oversize", "parent_traversal"]
)
def test_shared_archive_reader_refuses_unsafe_file_or_path(tmp_path, kind):
    from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at

    path = tmp_path / "artifact"
    path.write_bytes(b"abc")
    if kind == "symlink":
        link = tmp_path / "link"
        link.symlink_to(path)
        path = link
    elif kind == "ancestor_symlink":
        link = tmp_path / "alias"
        link.symlink_to(tmp_path, target_is_directory=True)
        path = link / path.name
    elif kind == "fifo":
        path = tmp_path / "pipe"
        os.mkfifo(path)
    elif kind == "directory":
        path = tmp_path
    elif kind == "parent_traversal":
        path = tmp_path / ".." / tmp_path.name / "artifact"
    with (
        pytest.raises((OSError, ValueError)),
        archive_parent(path, error_factory=failure) as (parent, name),
    ):
        read_archive_at(
            parent, name, max_bytes=2 if kind == "oversize" else 4, error_factory=failure
        )


def test_shared_archive_reader_detects_growth_after_size_check(tmp_path, monkeypatch):
    from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at

    path = tmp_path / "artifact"
    path.write_bytes(b"abc")
    original = os.fstat

    def grow(descriptor):
        info = original(descriptor)
        path.write_bytes(b"abcdef")
        return info

    monkeypatch.setattr(os, "fstat", grow)
    with (
        archive_parent(path, error_factory=failure) as (parent, name),
        pytest.raises(ValueError, match=r"archive\.byte_limit"),
    ):
        read_archive_at(parent, name, max_bytes=3, error_factory=failure)


@pytest.mark.parametrize("absolute", [False, True])
def test_shared_reader_accepts_only_a_leaf_name(tmp_path, absolute):
    from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at

    outside = tmp_path / "outside"
    outside.write_bytes(b"private")
    directory = tmp_path / "archive"
    directory.mkdir()
    name = str(outside) if absolute else "../outside"
    with (
        archive_parent(directory / "entry", error_factory=failure) as (parent, _),
        pytest.raises(ValueError, match=r"archive\.path"),
    ):
        read_archive_at(parent, name, max_bytes=16, error_factory=failure)
