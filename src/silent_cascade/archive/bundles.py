"""Bounded chunk production and authenticated staged restoration."""

import ctypes
import errno
import hashlib
import os
import secrets
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path, PurePosixPath

from silent_cascade.archive.catalog import (
    _canonical_policy,
    _create_at,
    _iter_inventory_records_from_unit,
    _open_child_directory,
    _opened_unit,
    _pinned_directory,
    _read_at,
    _remove_tree_at,
    _safe_logical_path,
    _source_file,
    _verify_pinned_directory,
)
from silent_cascade.archive.types import ArchivePolicy, UnitManifest, UnitRef
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes


def _validate_policy_binding(manifest: UnitManifest, policy: ArchivePolicy) -> None:
    if manifest.policy_sha256 != sha256_bytes(canonical_json_bytes(policy)):
        raise ValueError("archive policy differs from the sealed unit")


def _read_exact(descriptor: int, size: int) -> bytes:
    remaining = size
    blocks = []
    while remaining:
        block = os.read(descriptor, min(65_536, remaining))
        if not block:
            raise ValueError("archive source changed after sealing")
        blocks.append(block)
        remaining -= len(block)
    return b"".join(blocks)


def iter_unit_chunks(
    *,
    run_dir: Path,
    control_dir: Path,
    ref: UnitRef,
    scratch_dir: Path,
    policy: ArchivePolicy,
) -> Iterator[Path]:
    """Rebuild and yield one authenticated content chunk at a time."""
    policy = _canonical_policy(policy)
    with _opened_unit(control_dir, ref) as (unit, manifest):
        _validate_policy_binding(manifest, policy)
        with _pinned_directory(scratch_dir, create=True) as scratch:
            work_name = f".unit-{ref.unit_id[:16]}-{secrets.token_hex(8)}"
            os.mkdir(work_name, mode=0o700, dir_fd=scratch)
            work = _open_child_directory(scratch, work_name)
            chunk_payload = bytearray()
            chunk_index = 0
            try:
                for entry in _iter_inventory_records_from_unit(unit, manifest):
                    member = _safe_logical_path(entry.path, field="archive member path")
                    digest = hashlib.sha256()
                    observed = 0
                    with _source_file(run_dir, member) as (source, initial):
                        for span in entry.spans:
                            block = _read_exact(source, span.length)
                            digest.update(block)
                            observed += len(block)
                            chunk_payload.extend(block)
                            descriptor = manifest.chunks[chunk_index]
                            if len(chunk_payload) == descriptor.bytes:
                                payload = bytes(chunk_payload)
                                if sha256_bytes(payload) != descriptor.sha256:
                                    raise ValueError("archive source changed after sealing")
                                name = f"chunk-{chunk_index:05d}.bin"
                                _create_at(work, name, payload)
                                try:
                                    yield scratch_dir / work_name / name
                                finally:
                                    os.unlink(name, dir_fd=work)
                                chunk_payload.clear()
                                chunk_index += 1
                            elif len(chunk_payload) > descriptor.bytes:
                                raise ValueError("archive span exceeds sealed chunk")
                        if os.read(source, 1):
                            raise ValueError("archive source changed after sealing")
                        final = os.fstat(source)
                        if (
                            final.st_dev,
                            final.st_ino,
                            final.st_size,
                            final.st_mtime_ns,
                        ) != (
                            initial.st_dev,
                            initial.st_ino,
                            initial.st_size,
                            initial.st_mtime_ns,
                        ):
                            raise ValueError("archive source changed after sealing")
                    if (
                        observed != entry.bytes
                        or digest.hexdigest() != entry.sha256
                        or observed != initial.st_size
                    ):
                        raise ValueError("archive source changed after sealing")
                if chunk_payload or chunk_index != len(manifest.chunks):
                    raise ValueError("archive chunk reconstruction is incomplete")
            finally:
                os.close(work)
                _remove_tree_at(scratch, work_name)


def _selection(
    *,
    unit: int,
    manifest: UnitManifest,
    selected_paths: tuple[str, ...] | None,
    policy: ArchivePolicy,
) -> tuple[set[str] | None, int]:
    if selected_paths is None:
        return None, manifest.expanded_bytes
    if not isinstance(selected_paths, tuple) or not selected_paths:
        raise ValueError("selected paths must be a nonempty tuple")
    for path in selected_paths:
        _safe_logical_path(path, field="selected archive path")
    if tuple(sorted(selected_paths)) != selected_paths or len(set(selected_paths)) != len(
        selected_paths
    ):
        raise ValueError("selected paths must be unique and sorted")
    if manifest.kind == "episode_pack":
        for group in manifest.episode_groups:
            if selected_paths == group.paths:
                return set(selected_paths), group.expanded_bytes
        raise ValueError("selected paths are not one authenticated episode ownership group")
    if manifest.kind != "diagnostic":
        raise ValueError("sparse restore is permitted only for episodes or diagnostics")
    if len(selected_paths) > policy.page_entries:
        raise ValueError("diagnostic selection exceeds its entry bound")
    wanted = set(selected_paths)
    found: set[str] = set()
    expanded = 0
    for entry in _iter_inventory_records_from_unit(unit, manifest):
        if entry.path in wanted:
            found.add(entry.path)
            expanded += entry.bytes
    if found != wanted or expanded > policy.logs_bytes:
        raise ValueError("diagnostic selection is missing or exceeds its byte bound")
    return wanted, expanded


def _read_chunk(path: Path, *, expected_bytes: int, expected_sha256: str) -> bytes:
    if not isinstance(path, Path) or path.name in {"", ".", ".."}:
        raise ValueError("invalid chunk path")
    try:
        with _pinned_directory(path.parent) as parent:
            payload = _read_at(parent, path.name, max_bytes=expected_bytes)
    except (OSError, ValueError) as error:
        raise ValueError(f"chunk read failed: {error}") from error
    if len(payload) != expected_bytes or sha256_bytes(payload) != expected_sha256:
        raise ValueError("chunk bytes or hash differ")
    return payload


def _open_output(stage: int, path: str) -> tuple[int, int]:
    member = PurePosixPath(path)
    parent = os.dup(stage)
    try:
        for component in member.parts[:-1]:
            child = _open_child_directory(parent, component, create=True)
            os.close(parent)
            parent = child
        descriptor = os.open(
            member.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
            dir_fd=parent,
        )
        return parent, descriptor
    except BaseException:
        os.close(parent)
        raise


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        view = view[written:]


def _rename_directory_noreplace(
    source_parent: int,
    source_name: str,
    destination_parent: int,
    destination_name: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = getattr(libc, "renameatx_np", None)
        flag = 0x00000004  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        function = getattr(libc, "renameat2", None)
        flag = 1  # RENAME_NOREPLACE
    else:
        function = None
        flag = 0
    if function is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace directory rename is unavailable")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = function(
        source_parent,
        os.fsencode(source_name),
        destination_parent,
        os.fsencode(destination_name),
        flag,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error, os.strerror(error), destination_name)
    raise OSError(error, os.strerror(error), destination_name)


def _restore_records(
    *,
    unit: int,
    manifest: UnitManifest,
    chunks: Iterable[Path],
    selected: set[str] | None,
    required_chunks: tuple[int, ...],
    stage: int,
) -> None:
    chunk_paths = iter(chunks)
    current_payload: bytes | None = None
    current_index: int | None = None
    required = iter(required_chunks)
    next_required = next(required, None)
    for entry in _iter_inventory_records_from_unit(unit, manifest):
        output_parent: int | None = None
        output: int | None = None
        digest = hashlib.sha256()
        written = 0
        try:
            if selected is None or entry.path in selected:
                output_parent, output = _open_output(stage, entry.path)
            if output is not None:
                for span in entry.spans:
                    if span.chunk_index != current_index:
                        if span.chunk_index != next_required:
                            raise ValueError("selected chunk stream is out of order")
                        current_index = span.chunk_index
                        next_required = next(required, None)
                        try:
                            chunk_path = next(chunk_paths)
                        except StopIteration as error:
                            raise ValueError("chunk stream is missing a chunk") from error
                        descriptor = manifest.chunks[current_index]
                        current_payload = _read_chunk(
                            chunk_path,
                            expected_bytes=descriptor.bytes,
                            expected_sha256=descriptor.sha256,
                        )
                    assert current_payload is not None
                    block = current_payload[span.offset : span.offset + span.length]
                    if len(block) != span.length:
                        raise ValueError("chunk span is truncated")
                    _write_all(output, block)
                    digest.update(block)
                    written += len(block)
            if output is not None:
                if written != entry.bytes or digest.hexdigest() != entry.sha256:
                    raise ValueError("restored file bytes or hash differ")
                os.fsync(output)
        finally:
            if output is not None:
                os.close(output)
            if output_parent is not None:
                os.fsync(output_parent)
                os.close(output_parent)
    if next_required is not None:
        raise ValueError("chunk stream ended before the authenticated tail")
    try:
        next(chunk_paths)
    except StopIteration:
        return
    raise ValueError("chunk stream contains an unexpected extra chunk")


def _required_chunk_indices(
    unit: int, manifest: UnitManifest, selected: set[str] | None
) -> tuple[int, ...]:
    if selected is None:
        return tuple(range(len(manifest.chunks)))
    required: set[int] = set()
    found: set[str] = set()
    for entry in _iter_inventory_records_from_unit(unit, manifest):
        if entry.path in selected:
            found.add(entry.path)
            required.update(span.chunk_index for span in entry.spans)
    if found != selected:
        raise ValueError("selected inventory paths differ from authenticated ownership")
    return tuple(sorted(required))


def restore_unit(
    *,
    control_dir: Path,
    ref: UnitRef,
    chunks: Iterable[Path],
    destination: Path,
    policy: ArchivePolicy,
    selected_paths: tuple[str, ...] | None = None,
) -> Path:
    """Authenticate all chunks, fsync an owned staging tree, then publish it."""
    policy = _canonical_policy(policy)
    if not destination.name or destination.name in {".", ".."} or ".." in destination.parts:
        raise ValueError("unsafe restore destination")
    with _opened_unit(control_dir, ref) as (unit, manifest):
        _validate_policy_binding(manifest, policy)
        selected, restored_bytes = _selection(
            unit=unit,
            manifest=manifest,
            selected_paths=selected_paths,
            policy=policy,
        )
        required_chunks = _required_chunk_indices(unit, manifest, selected)
        if restored_bytes + policy.chunk_bytes > policy.cache_bytes:
            raise ValueError("restore exceeds simultaneous cache reservation")
        with _pinned_directory(destination.parent, create=True) as parent:
            try:
                os.stat(destination.name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(destination)
            stage_name = f".{destination.name}.{secrets.token_hex(8)}.restore"
            os.mkdir(stage_name, mode=0o700, dir_fd=parent)
            stage = _open_child_directory(parent, stage_name)
            try:
                _restore_records(
                    unit=unit,
                    manifest=manifest,
                    chunks=chunks,
                    selected=selected,
                    required_chunks=required_chunks,
                    stage=stage,
                )
                os.fsync(stage)
                _verify_pinned_directory(destination.parent, parent)
                try:
                    os.stat(destination.name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise FileExistsError(destination)
                os.close(stage)
                stage = -1
                _rename_directory_noreplace(parent, stage_name, parent, destination.name)
                os.fsync(parent)
            except BaseException:
                if stage >= 0:
                    os.close(stage)
                    stage = -1
                _remove_tree_at(parent, stage_name)
                raise
            finally:
                if stage >= 0:
                    os.close(stage)
    return destination
