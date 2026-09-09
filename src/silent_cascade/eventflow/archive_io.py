"""Shared Phase 2 archive descriptors; no changes to frozen Phase 1 I/O."""

import os
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

type ErrorFactory = Callable[[str], Exception]


@contextmanager
def archive_parent(path: Path, *, error_factory: ErrorFactory) -> Iterator[tuple[int, str]]:
    """Pin each path component without following links; close on every exit."""
    absolute = path.absolute()
    if ".." in absolute.parts or not absolute.name:
        raise error_factory("archive.path")
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in absolute.parts[1:-1]:
            next_descriptor = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        yield descriptor, absolute.name
    finally:
        os.close(descriptor)


def read_archive_at(
    parent: int, name: str, *, max_bytes: int, error_factory: ErrorFactory
) -> bytes:
    """Read one bounded regular file, detecting growth after the size check.

    Domain codecs adapt structural failures with error_factory and retain their
    existing OSError handling. The caller owns parent; this owns the file FD.
    """
    if type(max_bytes) is not int or max_bytes < 1:
        raise error_factory("archive.byte_limit")
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise error_factory("archive.path")
    descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise error_factory("archive.regular_file")
        if info.st_size > max_bytes:
            raise error_factory("archive.byte_limit")
        chunks, remaining = [], max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            remaining -= len(chunk)
        raise error_factory("archive.byte_limit")
    finally:
        os.close(descriptor)
