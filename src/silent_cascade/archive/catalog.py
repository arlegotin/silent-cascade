"""Immutable, paged archive inventories with exact-byte identities."""

import hashlib
import os
import secrets
import stat
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath

from silent_cascade.archive.types import (
    ArchivePolicy,
    BorrowedEntry,
    CatalogEntry,
    CatalogNode,
    CatalogNodeRef,
    CatalogRef,
    ChunkDescriptor,
    ChunkSpan,
    CorpusCatalogRoot,
    EpisodeGroup,
    FileEntry,
    InventoryEntry,
    InventoryShard,
    OwnershipCatalogEntry,
    RunCatalogEntry,
    RunCatalogRoot,
    UnitCatalogEntry,
    UnitIdentity,
    UnitManifest,
    UnitRef,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.train.evidence_types import decode_json

_KINDS = {
    "episode_pack",
    "evaluation_metadata",
    "journal",
    "control_snapshot",
    "diagnostic",
    "partial",
}
_MAX_MANIFEST_BYTES = 16 * 1024**2


def _safe_logical_path(value: str, *, field: str, allow_root_dot: bool = False) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe {field}")
    path = PurePosixPath(value)
    if value == ".":
        if allow_root_dot:
            return path
        raise ValueError(f"unsafe {field}")
    if path.is_absolute() or value.endswith("/") or "//" in value:
        raise ValueError(f"unsafe {field}")
    if any(part in {"", ".", ".."} for part in path.parts) or path.as_posix() != value:
        raise ValueError(f"unsafe {field}")
    return path


def _validate_member(path: str, logical_root: str) -> PurePosixPath:
    member = _safe_logical_path(path, field="archive member path")
    root = _safe_logical_path(logical_root, field="logical root", allow_root_dot=True)
    if logical_root == ".":
        return member
    if len(member.parts) <= len(root.parts) or member.parts[: len(root.parts)] != root.parts:
        raise ValueError("archive member lies outside logical root")
    return member


@contextmanager
def _pinned_directory(path: Path, *, create: bool = False) -> Iterator[int]:
    if ".." in path.parts:
        raise ValueError("unsafe directory path")
    absolute = path.absolute()
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                os.fsync(descriptor)
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = next_descriptor
        yield descriptor
    finally:
        os.close(descriptor)


def _open_child_directory(parent: int, name: str, *, create: bool = False) -> int:
    if PurePosixPath(name).name != name or name in {"", ".", ".."}:
        raise ValueError("unsafe directory leaf")
    try:
        return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    except FileNotFoundError:
        if not create:
            raise
        os.mkdir(name, mode=0o755, dir_fd=parent)
        os.fsync(parent)
        return os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)


def _verify_pinned_directory(path: Path, descriptor: int) -> None:
    expected = os.fstat(descriptor)
    with _pinned_directory(path) as current:
        actual = os.fstat(current)
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("archive directory path changed during publication")


def _read_at(parent: int, name: str, *, max_bytes: int) -> bytes:
    if max_bytes < 0:
        raise ValueError("invalid archive read limit")
    descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("archive object is not a regular file")
        if info.st_size > max_bytes:
            raise ValueError("archive object exceeds byte limit")
        remaining = max_bytes + 1
        chunks = []
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            remaining -= len(chunk)
        raise ValueError("archive object exceeds byte limit")
    finally:
        os.close(descriptor)


def _create_at(parent: int, name: str, payload: bytes) -> None:
    if PurePosixPath(name).name != name or name in {"", ".", ".."}:
        raise ValueError("unsafe archive leaf")
    temporary = f".{name}.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o644,
        dir_fd=parent,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        os.unlink(temporary, dir_fd=parent)
        os.fsync(parent)
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary, dir_fd=parent)
        raise


def _remove_tree_at(parent: int, name: str) -> None:
    try:
        directory = _open_child_directory(parent, name)
    except FileNotFoundError:
        return
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    _remove_tree_at(directory, entry.name)
                else:
                    os.unlink(entry.name, dir_fd=directory)
    finally:
        os.close(directory)
    os.rmdir(name, dir_fd=parent)


@contextmanager
def _source_file(run_dir: Path, member: PurePosixPath) -> Iterator[tuple[int, os.stat_result]]:
    with _pinned_directory(run_dir) as run_parent:
        parent = os.dup(run_parent)
        try:
            for component in member.parts[:-1]:
                child = _open_child_directory(parent, component)
                os.close(parent)
                parent = child
            descriptor = os.open(
                member.name,
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                dir_fd=parent,
            )
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("archive source is not a regular file")
                if info.st_nlink != 1:
                    raise ValueError("archive source has ambiguous hard-link ownership")
                yield descriptor, info
            finally:
                os.close(descriptor)
        finally:
            os.close(parent)


def _canonical_policy(policy: ArchivePolicy) -> ArchivePolicy:
    if not isinstance(policy, ArchivePolicy):
        raise TypeError("policy must be an ArchivePolicy")
    return ArchivePolicy.model_validate_json(canonical_json_bytes(policy))


def _unit_limit(kind: str, policy: ArchivePolicy) -> int:
    return {
        "episode_pack": policy.episode_bytes,
        "evaluation_metadata": policy.metadata_bytes,
        "journal": policy.journal_bytes,
        "control_snapshot": policy.pinned_bytes,
        "diagnostic": policy.logs_bytes,
        "partial": policy.episode_bytes,
    }[kind]


def _validated_borrowed(
    *,
    run_dir: Path,
    borrowed: tuple[FileEntry, ...],
    owned_paths: set[str],
    policy: ArchivePolicy,
) -> tuple[BorrowedEntry, ...]:
    if not isinstance(borrowed, tuple):
        raise ValueError("borrowed inventory must be a tuple")
    result = []
    total = 0
    for reference in borrowed:
        if not isinstance(reference, FileEntry):
            raise ValueError("borrowed inventory contains an invalid reference")
        try:
            entry = BorrowedEntry(
                path=reference.path,
                sha256=reference.sha256,
                bytes=reference.bytes,
            )
            member = _safe_logical_path(entry.path, field="borrowed archive path")
        except Exception as error:
            raise ValueError(f"invalid borrowed reference: {error}") from error
        if entry.path in owned_paths:
            raise ValueError("borrowed path overlaps owned archive inventory")
        digest = hashlib.sha256()
        observed = 0
        with _source_file(run_dir, member) as (descriptor, initial):
            if initial.st_size > policy.pinned_bytes:
                raise ValueError("borrowed file exceeds pinned admission")
            while block := os.read(descriptor, 65_536):
                digest.update(block)
                observed += len(block)
                if observed > policy.pinned_bytes:
                    raise ValueError("borrowed file exceeds pinned admission")
            final = os.fstat(descriptor)
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
                raise ValueError("borrowed file changed while sealing")
        if observed != entry.bytes or digest.hexdigest() != entry.sha256:
            raise ValueError("borrowed file differs from its authenticated reference")
        total += observed
        if total > policy.pinned_bytes:
            raise ValueError("borrowed files exceed pinned admission")
        result.append(entry)
    result_tuple = tuple(result)
    paths = tuple(entry.path for entry in result_tuple)
    if tuple(sorted(paths)) != paths or len(set(paths)) != len(paths):
        raise ValueError("borrowed paths must be unique and sorted")
    return result_tuple


def _scan_inventory(
    *,
    run_dir: Path,
    members: tuple[PurePosixPath, ...],
    stage: int,
    policy: ArchivePolicy,
) -> tuple[
    tuple[InventoryShard, ...],
    tuple[ChunkDescriptor, ...],
    str,
    int,
    dict[str, int],
    int,
]:
    chunk_digest = hashlib.sha256()
    chunk_size = 0
    chunks: list[ChunkDescriptor] = []
    inventory_digest = hashlib.sha256()
    shards: list[InventoryShard] = []
    pending: list[bytes] = []
    pending_size = 0
    shard_offset = 0
    expanded = 0
    sizes: dict[str, int] = {}
    metadata_bytes = 0

    def finish_chunk() -> None:
        nonlocal chunk_digest, chunk_size
        if chunk_size:
            chunks.append(
                ChunkDescriptor(
                    index=len(chunks), sha256=chunk_digest.hexdigest(), bytes=chunk_size
                )
            )
            chunk_digest = hashlib.sha256()
            chunk_size = 0

    def finish_shard() -> None:
        nonlocal pending, pending_size, shard_offset, metadata_bytes
        if not pending:
            return
        payload = b"".join(pending)
        if metadata_bytes + len(payload) > policy.metadata_bytes:
            raise ValueError("inventory shards exceed metadata reservation")
        name = f"inventory.{shard_offset:05d}.jsonl"
        _create_at(stage, name, payload)
        metadata_bytes += len(payload)
        shards.append(
            InventoryShard(
                path=name,
                sha256=sha256_bytes(payload),
                entries=len(pending),
                decoded_bytes=len(payload),
            )
        )
        shard_offset += len(pending)
        pending = []
        pending_size = 0

    for member in members:
        file_digest = hashlib.sha256()
        file_size = 0
        spans: list[ChunkSpan] = []
        with _source_file(run_dir, member) as (descriptor, initial):
            if initial.st_size > policy.episode_bytes:
                raise ValueError("archive source exceeds per-unit admission")
            while True:
                available = policy.chunk_bytes - chunk_size
                block = os.read(descriptor, min(65_536, available))
                if not block:
                    break
                if spans and spans[-1].chunk_index == len(chunks):
                    previous = spans[-1]
                    spans[-1] = ChunkSpan(
                        chunk_index=previous.chunk_index,
                        offset=previous.offset,
                        length=previous.length + len(block),
                    )
                else:
                    spans.append(
                        ChunkSpan(chunk_index=len(chunks), offset=chunk_size, length=len(block))
                    )
                file_digest.update(block)
                chunk_digest.update(block)
                file_size += len(block)
                expanded += len(block)
                chunk_size += len(block)
                if expanded > policy.spool_bytes:
                    raise ValueError("archive inventory exceeds spool admission")
                if chunk_size == policy.chunk_bytes:
                    finish_chunk()
            final = os.fstat(descriptor)
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
            ) or file_size != initial.st_size:
                raise ValueError("archive source changed while sealing")
        entry = InventoryEntry(
            path=member.as_posix(),
            sha256=file_digest.hexdigest(),
            bytes=file_size,
            spans=tuple(spans),
        )
        line = canonical_json_bytes(entry) + b"\n"
        if len(line) > policy.page_bytes:
            raise ValueError("single inventory entry exceeds page byte limit")
        if pending and (
            len(pending) == policy.page_entries or pending_size + len(line) > policy.page_bytes
        ):
            finish_shard()
        pending.append(line)
        pending_size += len(line)
        inventory_digest.update(line)
        sizes[entry.path] = file_size
    finish_chunk()
    finish_shard()
    return (
        tuple(shards),
        tuple(chunks),
        inventory_digest.hexdigest(),
        expanded,
        sizes,
        metadata_bytes,
    )


def _validated_groups(
    *,
    kind: str,
    episode_groups: tuple[tuple[str, ...], ...],
    members: tuple[PurePosixPath, ...],
    sizes: dict[str, int],
    policy: ArchivePolicy,
) -> tuple[EpisodeGroup, ...]:
    if kind != "episode_pack":
        if episode_groups:
            raise ValueError("episode groups are valid only for episode packs")
        return ()
    if not episode_groups or len(episode_groups) > policy.pack_episodes:
        raise ValueError("episode pack group count is invalid")
    expected = tuple(member.as_posix() for member in members)
    flattened: list[str] = []
    result = []
    for raw_group in episode_groups:
        group = tuple(raw_group)
        if not group or tuple(sorted(group)) != group or len(set(group)) != len(group):
            raise ValueError("episode group must be nonempty, unique, and sorted")
        total = 0
        for path in group:
            if path not in sizes:
                raise ValueError("episode group contains an unowned path")
            total += sizes[path]
        if total > policy.episode_bytes:
            raise ValueError("episode group exceeds episode admission")
        flattened.extend(group)
        result.append(EpisodeGroup(paths=group, expanded_bytes=total))
    if tuple(sorted(flattened)) != expected or len(flattened) != len(set(flattened)):
        raise ValueError("episode groups must exactly partition pack ownership")
    return tuple(result)


def _load_manifest_bytes(raw: bytes, *, expected_unit_id: str) -> tuple[UnitManifest, bytes]:
    try:
        payload = raw[:-1] if raw.endswith(b"\n") else raw
        decoded = decode_json(payload, limit=_MAX_MANIFEST_BYTES)
        manifest = UnitManifest.model_validate_json(canonical_json_bytes(decoded))
    except Exception as error:
        raise ValueError(f"invalid unit manifest: {error}") from error
    canonical = canonical_json_bytes(manifest)
    if raw != canonical + b"\n" or sha256_bytes(canonical) != expected_unit_id:
        raise ValueError("unit manifest is noncanonical or its identity differs")
    return manifest, raw


def _load_manifest_from_unit(unit: int, *, expected_unit_id: str) -> tuple[UnitManifest, bytes]:
    return _load_manifest_bytes(
        _read_at(unit, "manifest.json", max_bytes=_MAX_MANIFEST_BYTES),
        expected_unit_id=expected_unit_id,
    )


def _iter_inventory_records(
    manifest: UnitManifest, read_shard: Callable[[str, int], bytes]
) -> Iterator[InventoryEntry]:
    digest = hashlib.sha256()
    count = 0
    previous_path: str | None = None
    expected_chunk = 0
    expected_offset = 0
    seen_shards: set[str] = set()
    borrowed_paths = set()
    borrowed_bytes = 0
    for borrowed in manifest.borrowed:
        _safe_logical_path(borrowed.path, field="borrowed archive path")
        borrowed_paths.add(borrowed.path)
        borrowed_bytes += borrowed.bytes
    if borrowed_bytes > 1024**3:
        raise ValueError("borrowed inventory exceeds pinned ceiling")
    for shard in manifest.inventory_shards:
        if shard.path in seen_shards or shard.entries > 1000 or shard.decoded_bytes > 16 * 1024**2:
            raise ValueError("invalid inventory shard descriptor")
        seen_shards.add(shard.path)
        if shard.path != f"inventory.{count:05d}.jsonl":
            raise ValueError("inventory shard order differs")
        try:
            raw = read_shard(shard.path, shard.decoded_bytes)
        except (OSError, ValueError) as error:
            raise ValueError(f"inventory shard is unavailable: {error}") from error
        if len(raw) != shard.decoded_bytes or sha256_bytes(raw) != shard.sha256:
            raise ValueError("inventory shard bytes or hash differ")
        lines = raw.splitlines(keepends=True)
        if len(lines) != shard.entries or any(not line.endswith(b"\n") for line in lines):
            raise ValueError("inventory shard row count or final tail differs")
        for line in lines:
            try:
                decoded = decode_json(line[:-1], limit=shard.decoded_bytes)
                entry = InventoryEntry.model_validate_json(canonical_json_bytes(decoded))
            except Exception as error:
                raise ValueError(f"invalid inventory entry: {error}") from error
            if line != canonical_json_bytes(entry) + b"\n":
                raise ValueError("inventory entry is not canonical JSONL")
            _validate_member(entry.path, manifest.logical_root)
            if entry.path in borrowed_paths:
                raise ValueError("borrowed inventory overlaps owned inventory")
            if previous_path is not None and entry.path <= previous_path:
                raise ValueError("inventory paths are duplicated or out of order")
            previous_path = entry.path
            for span in entry.spans:
                if span.chunk_index >= len(manifest.chunks):
                    raise ValueError("inventory span names a missing chunk")
                chunk = manifest.chunks[span.chunk_index]
                if span.offset + span.length > chunk.bytes:
                    raise ValueError("inventory span exceeds its chunk")
                if (span.chunk_index, span.offset) != (expected_chunk, expected_offset):
                    raise ValueError("inventory spans contain a gap or overlap")
                expected_offset += span.length
                if expected_offset == chunk.bytes:
                    expected_chunk += 1
                    expected_offset = 0
            digest.update(line)
            count += 1
            yield entry
    if count != manifest.file_count or digest.hexdigest() != manifest.inventory_sha256:
        raise ValueError("inventory total count or stream hash differs")
    if expected_chunk != len(manifest.chunks) or expected_offset:
        raise ValueError("inventory does not cover the complete chunk stream")


def _iter_inventory_records_from_unit(
    unit: int, manifest: UnitManifest
) -> Iterator[InventoryEntry]:
    yield from _iter_inventory_records(
        manifest,
        lambda path, limit: _read_at(unit, path, max_bytes=limit),
    )


@contextmanager
def _opened_unit(control_dir: Path, ref: UnitRef) -> Iterator[tuple[int, UnitManifest]]:
    if (
        not isinstance(ref, UnitRef)
        or len(ref.unit_id) != 64
        or any(character not in "0123456789abcdef" for character in ref.unit_id)
        or ref.manifest_path != f"units/{ref.unit_id}/manifest.json"
    ):
        raise ValueError("invalid unit reference")
    with _pinned_directory(control_dir) as control:
        units = _open_child_directory(control, "units")
        try:
            unit = _open_child_directory(units, ref.unit_id)
            try:
                manifest, _ = _load_manifest_from_unit(unit, expected_unit_id=ref.unit_id)
                if (
                    manifest.kind != ref.kind
                    or manifest.logical_root != ref.logical_root
                    or manifest.expanded_bytes != ref.expanded_bytes
                    or manifest.file_count != ref.file_count
                ):
                    raise ValueError("unit reference differs from manifest")
                yield unit, manifest
            finally:
                os.close(unit)
        finally:
            os.close(units)


def _check_existing_ownership(
    units: int,
    *,
    candidate_id: str,
    candidate_run_id: str,
    candidate_kind: str,
    candidate_paths: set[str],
) -> None:
    if candidate_kind == "control_snapshot":
        return
    with os.scandir(units) as entries:
        for entry in entries:
            if entry.name.startswith(".pending-"):
                continue
            if not entry.is_dir(follow_symlinks=False):
                raise ValueError("unexpected object in archive unit catalog")
            if entry.name == candidate_id:
                continue
            existing = _open_child_directory(units, entry.name)
            try:
                manifest, _ = _load_manifest_from_unit(existing, expected_unit_id=entry.name)
                if (
                    manifest.kind == "control_snapshot"
                    or manifest.identity.run_id != candidate_run_id
                ):
                    continue
                for owned in _iter_inventory_records_from_unit(existing, manifest):
                    if owned.path in candidate_paths:
                        raise ValueError("duplicate active archive ownership")
            finally:
                os.close(existing)


def seal_unit(
    *,
    run_dir: Path,
    control_dir: Path,
    logical_root: str,
    paths: tuple[str, ...],
    kind: str,
    identity: dict,
    policy: ArchivePolicy,
    episode_groups: tuple[tuple[str, ...], ...] = (),
    borrowed: tuple[FileEntry, ...] = (),
) -> UnitRef:
    """Seal an exact immutable inventory without trusting a caller-provided label."""
    policy = _canonical_policy(policy)
    if kind not in _KINDS:
        raise ValueError("unknown archive unit kind")
    identity_model = UnitIdentity.model_validate(identity)
    if kind == "partial" and not identity_model.writer_stopped:
        raise ValueError("partial evidence requires a stopped writer")
    _safe_logical_path(logical_root, field="logical root", allow_root_dot=True)
    if logical_root == "." and kind not in {"journal", "control_snapshot"}:
        raise ValueError("run-root logical root is limited to journals and control snapshots")
    if kind == "control_snapshot" and (
        identity_model.checkpoint_sha256 is None or not identity_model.checkpoint_committed
    ):
        raise ValueError("control snapshot requires a committed checkpoint identity")
    if not isinstance(paths, tuple) or not paths:
        raise ValueError("archive inventory must be a nonempty tuple")
    supplied_members = tuple(_validate_member(path, logical_root) for path in paths)
    if len(set(supplied_members)) != len(supplied_members):
        raise ValueError("archive inventory paths must be unique")
    members = tuple(sorted(supplied_members))
    borrowed_entries = _validated_borrowed(
        run_dir=run_dir,
        borrowed=borrowed,
        owned_paths={member.as_posix() for member in members},
        policy=policy,
    )

    with _pinned_directory(control_dir, create=True) as control:
        units = _open_child_directory(control, "units", create=True)
        stage_name = f".pending-{secrets.token_hex(16)}"
        os.mkdir(stage_name, mode=0o755, dir_fd=units)
        stage = _open_child_directory(units, stage_name)
        try:
            shards, chunks, inventory_hash, expanded, sizes, inventory_metadata_bytes = (
                _scan_inventory(
                    run_dir=run_dir,
                    members=members,
                    stage=stage,
                    policy=policy,
                )
            )
            groups = _validated_groups(
                kind=kind,
                episode_groups=episode_groups,
                members=members,
                sizes=sizes,
                policy=policy,
            )
            if expanded > _unit_limit(kind, policy):
                raise ValueError("archive unit exceeds its kind admission")
            if kind == "episode_pack" and expanded > policy.pack_target_bytes and len(groups) != 1:
                raise ValueError("an oversized episode pack must be one admitted singleton")
            manifest = UnitManifest(
                kind=kind,
                logical_root=logical_root,
                identity=identity_model,
                policy_sha256=sha256_bytes(canonical_json_bytes(policy)),
                expanded_bytes=expanded,
                file_count=len(members),
                inventory_sha256=inventory_hash,
                inventory_shards=shards,
                chunks=chunks,
                episode_groups=groups,
                borrowed=borrowed_entries,
            )
            manifest_payload = canonical_json_bytes(manifest)
            if inventory_metadata_bytes + len(manifest_payload) + 1 > policy.metadata_bytes:
                raise ValueError("unit manifest exceeds metadata admission")
            unit_id = sha256_bytes(manifest_payload)
            candidate_paths = set(sizes)
            _check_existing_ownership(
                units,
                candidate_id=unit_id,
                candidate_run_id=identity_model.run_id,
                candidate_kind=kind,
                candidate_paths=candidate_paths,
            )
            _create_at(stage, "manifest.json", manifest_payload + b"\n")
            os.fsync(stage)
            _verify_pinned_directory(control_dir, control)
            _verify_pinned_directory(control_dir / "units", units)
            try:
                existing = _open_child_directory(units, unit_id)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                try:
                    _, existing_raw = _load_manifest_from_unit(existing, expected_unit_id=unit_id)
                    if existing_raw != manifest_payload + b"\n":
                        raise ValueError("content-addressed unit collision")
                finally:
                    os.close(existing)
                os.close(stage)
                stage = -1
                _remove_tree_at(units, stage_name)
            else:
                os.close(stage)
                stage = -1
                os.rename(stage_name, unit_id, src_dir_fd=units, dst_dir_fd=units)
                os.fsync(units)
            return UnitRef(
                unit_id=unit_id,
                kind=kind,
                logical_root=logical_root,
                expanded_bytes=expanded,
                file_count=len(members),
                manifest_path=f"units/{unit_id}/manifest.json",
            )
        except BaseException:
            if stage >= 0:
                os.close(stage)
                stage = -1
            _remove_tree_at(units, stage_name)
            raise
        finally:
            if stage >= 0:
                os.close(stage)
            os.close(units)


def iter_unit_files(control_dir: Path, ref: UnitRef) -> Iterator[FileEntry]:
    """Yield every authenticated inventory entry with bounded page state."""
    with _opened_unit(control_dir, ref) as (unit, manifest):
        for entry in _iter_inventory_records_from_unit(unit, manifest):
            yield FileEntry(path=entry.path, sha256=entry.sha256, bytes=entry.bytes)


type ObjectReader = Callable[[str, int], bytes]


def _validate_unit_ref(ref: UnitRef) -> None:
    if (
        not isinstance(ref, UnitRef)
        or type(ref.unit_id) is not str
        or len(ref.unit_id) != 64
        or any(character not in "0123456789abcdef" for character in ref.unit_id)
        or type(ref.expanded_bytes) is not int
        or ref.expanded_bytes < 0
        or type(ref.file_count) is not int
        or ref.file_count < 1
        or ref.manifest_path != f"units/{ref.unit_id}/manifest.json"
    ):
        raise ValueError("invalid unit reference")


def _control_reader(control_dir: Path) -> ObjectReader:
    def read(path: str, max_bytes: int) -> bytes:
        logical = _safe_logical_path(path, field="control object path")
        with _pinned_directory(control_dir) as root:
            parent = os.dup(root)
            try:
                for component in logical.parts[:-1]:
                    child = _open_child_directory(parent, component)
                    os.close(parent)
                    parent = child
                return _read_at(parent, logical.name, max_bytes=max_bytes)
            finally:
                os.close(parent)

    return read


def _create_control_object_at(control: int, path: str, payload: bytes) -> None:
    logical = _safe_logical_path(path, field="control object path")
    parent = os.dup(control)
    try:
        for component in logical.parts[:-1]:
            child = _open_child_directory(parent, component, create=True)
            os.close(parent)
            parent = child
        try:
            _create_at(parent, logical.name, payload)
        except FileExistsError:
            if _read_at(parent, logical.name, max_bytes=len(payload)) != payload:
                raise ValueError("content-addressed catalog object collision") from None
    finally:
        os.close(parent)


def _create_control_object(control_dir: Path, path: str, payload: bytes) -> None:
    with _pinned_directory(control_dir, create=True) as control:
        _create_control_object_at(control, path, payload)


def _external_unit(
    control_dir: Path, ref: UnitRef, reader: ObjectReader | None
) -> tuple[UnitManifest, Callable[[str, int], bytes]]:
    _validate_unit_ref(ref)
    read = reader or _control_reader(control_dir)
    manifest, _ = _load_manifest_bytes(
        read(ref.manifest_path, _MAX_MANIFEST_BYTES), expected_unit_id=ref.unit_id
    )
    if (
        manifest.kind != ref.kind
        or manifest.logical_root != ref.logical_root
        or manifest.expanded_bytes != ref.expanded_bytes
        or manifest.file_count != ref.file_count
    ):
        raise ValueError("unit reference differs from manifest")
    parent = PurePosixPath(ref.manifest_path).parent
    return manifest, lambda path, limit: read((parent / path).as_posix(), limit)


def _key_bit(key: str, depth: int) -> int:
    value = int(key[depth // 4], 16)
    return (value >> (3 - depth % 4)) & 1


def _first_differing_bit(left: str, right: str) -> int:
    for depth in range(256):
        if _key_bit(left, depth) != _key_bit(right, depth):
            return depth
    return 256


def _validate_branch_prefix(node: CatalogNode) -> None:
    if node.records:
        return
    assert node.zero is not None and node.one is not None
    endpoints = (
        node.zero.first_key,
        node.zero.last_key,
        node.one.first_key,
        node.one.last_key,
    )
    if (
        node.zero.first_key > node.zero.last_key
        or node.one.first_key > node.one.last_key
        or node.zero.last_key >= node.one.first_key
        or _first_differing_bit(node.zero.first_key, node.one.first_key) != node.depth
        or any(_key_bit(key, node.depth) != 0 for key in endpoints[:2])
        or any(_key_bit(key, node.depth) != 1 for key in endpoints[2:])
    ):
        raise ValueError("catalog branch depth or prefix differs")


class _CatalogStore:
    def __init__(
        self,
        *,
        control_dir: Path,
        policy: ArchivePolicy,
        reader: ObjectReader,
    ) -> None:
        self.control_dir = control_dir
        self.policy = policy
        self.reader = reader
        self.staged: dict[str, bytes] = {}
        self.staged_nodes: dict[str, CatalogNode] = {}
        self.staged_bytes = 0
        self.cache: OrderedDict[str, CatalogNode] = OrderedDict()

    def _ref(self, node: CatalogNode, payload: bytes) -> CatalogNodeRef:
        if node.records:
            entries = len(node.records)
            first_key, last_key = node.records[0].key, node.records[-1].key
        else:
            assert node.zero is not None and node.one is not None
            entries = node.zero.entries + node.one.entries
            first_key, last_key = node.zero.first_key, node.one.last_key
        digest = sha256_bytes(payload)
        return CatalogNodeRef(
            sha256=digest,
            path=f"catalog/nodes/{digest}.json",
            entries=entries,
            decoded_bytes=len(payload) + 1,
            first_key=first_key,
            last_key=last_key,
        )

    def stage(self, node: CatalogNode) -> CatalogNodeRef:
        if node.records and len(node.records) > self.policy.page_entries:
            raise ValueError("catalog leaf entry bound exceeded")
        payload = canonical_json_bytes(node)
        if len(payload) + 1 > self.policy.page_bytes:
            raise ValueError("catalog node byte bound exceeded")
        ref = self._ref(node, payload)
        encoded = payload + b"\n"
        previous = self.staged.get(ref.path)
        self.staged[ref.path] = encoded
        self.staged_nodes[ref.sha256] = node
        self.staged_bytes += len(encoded) - (0 if previous is None else len(previous))
        if self.staged_bytes > self.policy.metadata_bytes:
            raise ValueError("catalog staged nodes exceed metadata reservation")
        return ref

    def load(self, ref: CatalogNodeRef, *, index: str) -> CatalogNode:
        if ref.decoded_bytes > self.policy.page_bytes or ref.first_key > ref.last_key:
            raise ValueError("catalog node descriptor exceeds page bounds")
        node = self.staged_nodes.get(ref.sha256)
        if node is None:
            node = self.cache.get(ref.sha256)
        if node is None:
            raw = self.reader(ref.path, ref.decoded_bytes)
            if len(raw) != ref.decoded_bytes or len(raw) > self.policy.page_bytes:
                raise ValueError("catalog node byte count differs")
            payload = raw[:-1] if raw.endswith(b"\n") else raw
            try:
                decoded = decode_json(payload, limit=self.policy.page_bytes)
                node = CatalogNode.model_validate_json(canonical_json_bytes(decoded))
            except Exception as error:
                raise ValueError(f"invalid catalog node: {error}") from error
            if raw != canonical_json_bytes(node) + b"\n":
                raise ValueError("catalog node is not canonical JSON")
            self.cache[ref.sha256] = node
            self.cache.move_to_end(ref.sha256)
            while len(self.cache) > min(8, self.policy.page_entries):
                self.cache.popitem(last=False)
        payload = canonical_json_bytes(node)
        if node.records and len(node.records) > self.policy.page_entries:
            raise ValueError("catalog leaf entry bound exceeded")
        _validate_branch_prefix(node)
        if self._ref(node, payload) != ref or node.index != index:
            raise ValueError("catalog node descriptor or index differs")
        return node

    def publish(self, control: int) -> None:
        for path, payload in self.staged.items():
            _create_control_object_at(control, path, payload)


def _build_tree(
    records: tuple[CatalogEntry, ...],
    *,
    index: str,
    depth: int,
    store: _CatalogStore,
) -> CatalogNodeRef:
    leaf = CatalogNode(index=index, depth=min(depth, 255), records=records)
    if (
        len(records) <= store.policy.page_entries
        and len(canonical_json_bytes(leaf)) + 1 <= store.policy.page_bytes
    ):
        return store.stage(leaf)
    if len(records) == 1:
        raise ValueError("single catalog record exceeds page byte bound")
    split = depth
    while split < 256:
        zero = tuple(record for record in records if _key_bit(record.key, split) == 0)
        if zero and len(zero) != len(records):
            one = tuple(record for record in records if _key_bit(record.key, split) == 1)
            return store.stage(
                CatalogNode(
                    index=index,
                    depth=split,
                    zero=_build_tree(zero, index=index, depth=split + 1, store=store),
                    one=_build_tree(one, index=index, depth=split + 1, store=store),
                )
            )
        split += 1
    raise ValueError("catalog key collision")


def _insert_batch(
    ref: CatalogNodeRef | None,
    additions: tuple[CatalogEntry, ...],
    *,
    index: str,
    store: _CatalogStore,
    depth: int = 0,
    replace: bool = False,
) -> CatalogNodeRef | None:
    if not additions:
        return ref
    if ref is None:
        return _build_tree(additions, index=index, depth=depth, store=store)
    node = store.load(ref, index=index)
    if node.records:
        merged = {record.key: record for record in node.records}
        for record in additions:
            prior = merged.get(record.key)
            if prior is not None and prior != record and not replace:
                raise ValueError("catalog key is bound to different content")
            merged[record.key] = record
        records = tuple(sorted(merged.values(), key=lambda record: record.key))
        return _build_tree(records, index=index, depth=node.depth, store=store)
    assert node.zero is not None and node.one is not None
    combined_first = min(ref.first_key, additions[0].key)
    combined_last = max(ref.last_key, additions[-1].key)
    split = _first_differing_bit(combined_first, combined_last)
    if split < node.depth:
        zero_additions = tuple(r for r in additions if _key_bit(r.key, split) == 0)
        one_additions = tuple(r for r in additions if _key_bit(r.key, split) == 1)
        if _key_bit(ref.first_key, split) == 0:
            zero = _insert_batch(
                ref,
                zero_additions,
                index=index,
                store=store,
                depth=split + 1,
                replace=replace,
            )
            one = _build_tree(one_additions, index=index, depth=split + 1, store=store)
        else:
            zero = _build_tree(zero_additions, index=index, depth=split + 1, store=store)
            one = _insert_batch(
                ref,
                one_additions,
                index=index,
                store=store,
                depth=split + 1,
                replace=replace,
            )
        assert zero is not None and one is not None
        return store.stage(CatalogNode(index=index, depth=split, zero=zero, one=one))
    zero_additions = tuple(r for r in additions if _key_bit(r.key, node.depth) == 0)
    one_additions = tuple(r for r in additions if _key_bit(r.key, node.depth) == 1)
    zero = _insert_batch(
        node.zero,
        zero_additions,
        index=index,
        store=store,
        depth=node.depth + 1,
        replace=replace,
    )
    one = _insert_batch(
        node.one,
        one_additions,
        index=index,
        store=store,
        depth=node.depth + 1,
        replace=replace,
    )
    assert zero is not None and one is not None
    return store.stage(CatalogNode(index=index, depth=node.depth, zero=zero, one=one))


def _lookup(
    store: _CatalogStore, ref: CatalogNodeRef | None, key: str, *, index: str
) -> CatalogEntry | None:
    while ref is not None:
        node = store.load(ref, index=index)
        if node.records:
            return next((record for record in node.records if record.key == key), None)
        ref = node.one if _key_bit(key, node.depth) else node.zero
    return None


def _iter_tree(
    store: _CatalogStore, ref: CatalogNodeRef | None, *, index: str
) -> Iterator[CatalogEntry]:
    if ref is None:
        return
    pending = [ref]
    previous = None
    count = 0
    while pending:
        node = store.load(pending.pop(), index=index)
        if node.records:
            for record in node.records:
                if previous is not None and record.key <= previous:
                    raise ValueError("catalog traversal order or duplicate differs")
                previous = record.key
                count += 1
                yield record
        else:
            assert node.zero is not None and node.one is not None
            pending.extend((node.one, node.zero))
    if count != ref.entries:
        raise ValueError("catalog traversal count differs")


def _ownership_key(record_type: str, path: str) -> str:
    return sha256_bytes(f"{record_type}\0{path}".encode())


def _path_ancestors(path: str) -> Iterator[str]:
    parts = PurePosixPath(path).parts
    for size in range(1, len(parts)):
        yield "/".join(parts[:size])


def _validate_catalog_ref(ref: CatalogRef, scope: str) -> None:
    if (
        not isinstance(ref, CatalogRef)
        or ref.scope != scope
        or type(ref.catalog_id) is not str
        or len(ref.catalog_id) != 64
        or any(character not in "0123456789abcdef" for character in ref.catalog_id)
        or ref.root_path != f"catalog/roots/{ref.catalog_id}.json"
        or type(ref.entry_count) is not int
        or ref.entry_count < 0
    ):
        raise ValueError("invalid catalog reference")


def _load_root(
    ref: CatalogRef, *, scope: str, reader: ObjectReader, policy: ArchivePolicy
) -> RunCatalogRoot | CorpusCatalogRoot:
    _validate_catalog_ref(ref, scope)
    raw = reader(ref.root_path, policy.page_bytes)
    payload = raw[:-1] if raw.endswith(b"\n") else raw
    if len(raw) > policy.page_bytes or sha256_bytes(payload) != ref.catalog_id:
        raise ValueError("catalog root bytes or identity differ")
    try:
        decoded = decode_json(payload, limit=policy.page_bytes)
        model = RunCatalogRoot if scope == "run" else CorpusCatalogRoot
        root = model.model_validate_json(canonical_json_bytes(decoded))
    except Exception as error:
        raise ValueError(f"invalid catalog root: {error}") from error
    if raw != canonical_json_bytes(root) + b"\n":
        raise ValueError("catalog root is not canonical JSON")
    count = root.unit_count if isinstance(root, RunCatalogRoot) else root.run_count
    if count != ref.entry_count:
        raise ValueError("catalog reference count differs")
    if isinstance(root, RunCatalogRoot) and root.run_id != ref.run_id:
        raise ValueError("run catalog identity differs")
    return root


def _publish_generation(
    *,
    control_dir: Path,
    store: _CatalogStore,
    root: RunCatalogRoot | CorpusCatalogRoot,
    scope: str,
    run_id: str | None,
    count: int,
    policy: ArchivePolicy,
) -> CatalogRef:
    payload = canonical_json_bytes(root)
    if len(payload) + 1 > policy.page_bytes:
        raise ValueError("catalog root exceeds page byte bound")
    digest = sha256_bytes(payload)
    path = f"catalog/roots/{digest}.json"
    with _pinned_directory(control_dir, create=True) as control:
        store.publish(control)
        _verify_pinned_directory(control_dir, control)
        _create_control_object_at(control, path, payload + b"\n")
        _verify_pinned_directory(control_dir, control)
    return CatalogRef(digest, scope, run_id, count, path)


def publish_run_catalog(
    *,
    control_dir: Path,
    run_id: str,
    units: Iterable[UnitRef],
    policy: ArchivePolicy,
    previous: CatalogRef | None = None,
    object_reader: ObjectReader | None = None,
    unit_reader: ObjectReader | None = None,
) -> CatalogRef:
    """Commit a persistent run generation without materializing prior catalogs."""
    policy = _canonical_policy(policy)
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run catalog requires a run identity")
    reader = object_reader or _control_reader(control_dir)
    store = _CatalogStore(control_dir=control_dir, policy=policy, reader=reader)
    unit_root = ownership_root = None
    if previous is not None:
        prior = _load_root(previous, scope="run", reader=reader, policy=policy)
        assert isinstance(prior, RunCatalogRoot)
        if prior.run_id != run_id:
            raise ValueError("previous catalog belongs to another run")
        unit_root, ownership_root = prior.units, prior.ownership

    unit_additions: dict[str, UnitCatalogEntry] = {}
    owned: list[tuple[str, str]] = []
    buffered = 0
    for ref in units:
        manifest, read_shard = _external_unit(control_dir, ref, unit_reader)
        if manifest.identity.run_id != run_id:
            raise ValueError("unit belongs to another run catalog")
        record = UnitCatalogEntry(
            key=ref.unit_id,
            unit_id=ref.unit_id,
            kind=manifest.kind,
            logical_root=manifest.logical_root,
            expanded_bytes=manifest.expanded_bytes,
            file_count=manifest.file_count,
            manifest_path=ref.manifest_path,
        )
        existing = _lookup(store, unit_root, record.key, index="units")
        if existing is not None:
            if existing != record:
                raise ValueError("unit catalog identity collision")
            continue
        unit_additions[record.key] = record
        buffered += len(canonical_json_bytes(record))
        if manifest.kind != "control_snapshot":
            for entry in _iter_inventory_records(manifest, read_shard):
                owned.append((entry.path, ref.unit_id))
                buffered += len(entry.path) + 96
                if buffered > policy.metadata_bytes:
                    raise ValueError("catalog generation exceeds metadata reservation")

    owned.sort()
    batch_files: set[str] = set()
    batch_directories: set[str] = set()
    for path, _unit_id in owned:
        ancestors = tuple(_path_ancestors(path))
        if (
            path in batch_files
            or path in batch_directories
            or any(ancestor in batch_files for ancestor in ancestors)
        ):
            raise ValueError("catalog batch has duplicate or overlapping ownership")
        batch_files.add(path)
        batch_directories.update(ancestors)
        for ancestor in ancestors:
            if (
                _lookup(store, ownership_root, _ownership_key("file", ancestor), index="ownership")
                is not None
            ):
                raise ValueError("catalog path descends from an owned file")
        if (
            _lookup(store, ownership_root, _ownership_key("file", path), index="ownership")
            is not None
        ):
            raise ValueError("catalog path already has an owner")
        if (
            _lookup(store, ownership_root, _ownership_key("directory", path), index="ownership")
            is not None
        ):
            raise ValueError("catalog path is an ancestor of an owned file")

    ownership_additions: dict[str, OwnershipCatalogEntry] = {}
    for path, unit_id in owned:
        record = OwnershipCatalogEntry(
            record_type="file",
            key=_ownership_key("file", path),
            path=path,
            unit_id=unit_id,
        )
        ownership_additions[record.key] = record
        for ancestor in _path_ancestors(path):
            marker = OwnershipCatalogEntry(
                record_type="directory",
                key=_ownership_key("directory", ancestor),
                path=ancestor,
                unit_id=None,
            )
            ownership_additions[marker.key] = marker

    units_batch: tuple[CatalogEntry, ...] = tuple(
        sorted(unit_additions.values(), key=lambda entry: entry.key)
    )
    ownership_batch: tuple[CatalogEntry, ...] = tuple(
        sorted(ownership_additions.values(), key=lambda entry: entry.key)
    )
    unit_root = _insert_batch(unit_root, units_batch, index="units", store=store)
    ownership_root = _insert_batch(ownership_root, ownership_batch, index="ownership", store=store)
    root = RunCatalogRoot(
        run_id=run_id,
        units=unit_root,
        ownership=ownership_root,
        unit_count=0 if unit_root is None else unit_root.entries,
        ownership_count=0 if ownership_root is None else ownership_root.entries,
    )
    return _publish_generation(
        control_dir=control_dir,
        store=store,
        root=root,
        scope="run",
        run_id=run_id,
        count=root.unit_count,
        policy=policy,
    )


def iter_run_catalog(
    control_dir: Path,
    ref: CatalogRef,
    *,
    policy: ArchivePolicy,
    object_reader: ObjectReader | None = None,
) -> Iterator[UnitRef]:
    policy = _canonical_policy(policy)
    reader = object_reader or _control_reader(control_dir)
    root = _load_root(ref, scope="run", reader=reader, policy=policy)
    assert isinstance(root, RunCatalogRoot)
    store = _CatalogStore(control_dir=control_dir, policy=policy, reader=reader)
    ownership_count = 0
    for entry in _iter_tree(store, root.ownership, index="ownership"):
        if not isinstance(entry, OwnershipCatalogEntry):
            raise ValueError("run catalog yielded a non-ownership entry")
        ownership_count += 1
    if ownership_count != root.ownership_count:
        raise ValueError("run catalog ownership final count differs")
    count = 0
    for entry in _iter_tree(store, root.units, index="units"):
        if not isinstance(entry, UnitCatalogEntry):
            raise ValueError("run catalog yielded a non-unit entry")
        count += 1
        yield UnitRef(
            entry.unit_id,
            entry.kind,
            entry.logical_root,
            entry.expanded_bytes,
            entry.file_count,
            entry.manifest_path,
        )
    if count != root.unit_count:
        raise ValueError("run catalog final count differs")


def verify_run_catalog_unit(
    control_dir: Path,
    ref: CatalogRef,
    unit: UnitRef,
    *,
    policy: ArchivePolicy,
    object_reader: ObjectReader | None = None,
    unit_reader: ObjectReader | None = None,
) -> None:
    """Verify one unit and its ownership paths without traversing the run catalog."""
    policy = _canonical_policy(policy)
    reader = object_reader or _control_reader(control_dir)
    root = _load_root(ref, scope="run", reader=reader, policy=policy)
    assert isinstance(root, RunCatalogRoot)
    store = _CatalogStore(control_dir=control_dir, policy=policy, reader=reader)
    manifest, read_shard = _external_unit(control_dir, unit, unit_reader)
    expected = UnitCatalogEntry(
        key=unit.unit_id,
        unit_id=unit.unit_id,
        kind=manifest.kind,
        logical_root=manifest.logical_root,
        expanded_bytes=manifest.expanded_bytes,
        file_count=manifest.file_count,
        manifest_path=unit.manifest_path,
    )
    if (
        manifest.identity.run_id != root.run_id
        or _lookup(store, root.units, unit.unit_id, index="units") != expected
    ):
        raise ValueError("unit is not authenticated by run catalog")
    for entry in _iter_inventory_records(manifest, read_shard):
        if manifest.kind == "control_snapshot":
            if (
                manifest.identity.checkpoint_sha256 is None
                or not manifest.identity.checkpoint_committed
            ):
                raise ValueError("control snapshot lacks committed checkpoint identity")
            # Snapshots authenticate immutable versions of mutable/pinned source
            # paths. The catalog deliberately grants them no exclusive ownership.
            # Exhaust the inventory iterator so final shard/count closure still runs.
            continue
        owner = OwnershipCatalogEntry(
            record_type="file",
            key=_ownership_key("file", entry.path),
            path=entry.path,
            unit_id=unit.unit_id,
        )
        if _lookup(store, root.ownership, owner.key, index="ownership") != owner:
            raise ValueError("unit path ownership is not authenticated by run catalog")
        for ancestor in _path_ancestors(entry.path):
            directory = OwnershipCatalogEntry(
                record_type="directory",
                key=_ownership_key("directory", ancestor),
                path=ancestor,
                unit_id=None,
            )
            if _lookup(store, root.ownership, directory.key, index="ownership") != directory:
                raise ValueError("unit directory ownership is not authenticated by run catalog")


def publish_corpus_catalog(
    *,
    control_dir: Path,
    runs: Iterable[CatalogRef],
    policy: ArchivePolicy,
    previous: CatalogRef | None = None,
    object_reader: ObjectReader | None = None,
) -> CatalogRef:
    policy = _canonical_policy(policy)
    reader = object_reader or _control_reader(control_dir)
    store = _CatalogStore(control_dir=control_dir, policy=policy, reader=reader)
    run_root = None
    if previous is not None:
        prior = _load_root(previous, scope="corpus", reader=reader, policy=policy)
        assert isinstance(prior, CorpusCatalogRoot)
        run_root = prior.runs
    additions: dict[str, RunCatalogEntry] = {}
    buffered = 0
    for ref in runs:
        _validate_catalog_ref(ref, "run")
        if ref.run_id is None:
            raise ValueError("run catalog reference lacks run identity")
        _load_root(ref, scope="run", reader=reader, policy=policy)
        key = sha256_bytes(f"run\0{ref.run_id}".encode())
        record = RunCatalogEntry(
            key=key,
            run_id=ref.run_id,
            catalog_id=ref.catalog_id,
            entry_count=ref.entry_count,
            root_path=ref.root_path,
        )
        additions[key] = record
        buffered += len(canonical_json_bytes(record))
        if buffered > policy.metadata_bytes:
            raise ValueError("corpus generation exceeds metadata reservation")
    batch: tuple[CatalogEntry, ...] = tuple(sorted(additions.values(), key=lambda entry: entry.key))
    run_root = _insert_batch(run_root, batch, index="runs", store=store, replace=True)
    root = CorpusCatalogRoot(
        runs=run_root,
        run_count=0 if run_root is None else run_root.entries,
    )
    return _publish_generation(
        control_dir=control_dir,
        store=store,
        root=root,
        scope="corpus",
        run_id=None,
        count=root.run_count,
        policy=policy,
    )


def iter_corpus_catalog(
    control_dir: Path,
    ref: CatalogRef,
    *,
    policy: ArchivePolicy,
    object_reader: ObjectReader | None = None,
) -> Iterator[CatalogRef]:
    policy = _canonical_policy(policy)
    reader = object_reader or _control_reader(control_dir)
    root = _load_root(ref, scope="corpus", reader=reader, policy=policy)
    assert isinstance(root, CorpusCatalogRoot)
    store = _CatalogStore(control_dir=control_dir, policy=policy, reader=reader)
    count = 0
    for entry in _iter_tree(store, root.runs, index="runs"):
        if not isinstance(entry, RunCatalogEntry):
            raise ValueError("corpus catalog yielded a non-run entry")
        count += 1
        yield CatalogRef(
            entry.catalog_id,
            "run",
            entry.run_id,
            entry.entry_count,
            entry.root_path,
        )
    if count != root.run_count:
        raise ValueError("corpus catalog final count differs")
