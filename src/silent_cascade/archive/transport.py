"""Create-only R2 transport, verified receipts, leases, and exact eviction."""

import fcntl
import hashlib
import json
import os
import re
import resource
import secrets
import selectors
import signal
import stat
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath
from typing import Protocol

from silent_cascade.archive.bundles import iter_unit_chunks
from silent_cascade.archive.catalog import (
    _canonical_policy,
    _control_reader,
    _create_at,
    _create_control_object,
    _iter_inventory_records_from_unit,
    _open_child_directory,
    _opened_unit,
    _pinned_directory,
    _read_at,
    _remove_tree_at,
    _safe_logical_path,
    _source_file,
    publish_run_catalog,
    verify_run_catalog_unit,
)
from silent_cascade.archive.operational import (
    eviction_key,
    lookup_remote_reservation,
    publish_operational_catalog,
    receipt_key,
    reservation_key,
)
from silent_cascade.archive.types import (
    ArchivePolicy,
    CatalogRef,
    EvictionLocatorEntry,
    FileEntry,
    ReceiptLocatorEntry,
    RemoteReservationEntry,
    UnitRef,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.train.evidence_types import decode_json

_AWS_COMMAND = ("aws",)
_AWS_TIMEOUT_SECONDS = 30 * 60
_AWS_MAX_ATTEMPTS = 3
_AWS_OUTPUT_BYTES = 64 * 1024
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


class ObjectTransport(Protocol):
    @property
    def transport_id(self) -> str: ...

    def create(self, key: str, source: Path) -> None: ...

    def download(self, key: str, destination: Path, *, max_bytes: int) -> None: ...


class AmbiguousWriteError(OSError):
    """A create may have reached remote storage and requires exact readback."""


class _ProcessTimeout(TimeoutError):
    pass


def _replace_at(parent: int, name: str, payload: bytes) -> None:
    if PurePosixPath(name).name != name or name in {"", ".", ".."}:
        raise ValueError("unsafe archive state leaf")
    temporary = f".{name}.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
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
        os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
        os.fsync(parent)
    except BaseException:
        with suppress(OSError):
            os.unlink(temporary, dir_fd=parent)
        raise


def _read_json_at(parent: int, name: str, *, max_bytes: int) -> tuple[dict, bytes]:
    raw = _read_at(parent, name, max_bytes=max_bytes)
    if not raw.endswith(b"\n"):
        raise ValueError("archive state is not canonical JSON")
    try:
        decoded = decode_json(raw[:-1], limit=max_bytes)
    except Exception as error:
        raise ValueError(f"invalid archive state: {error}") from error
    if not isinstance(decoded, dict) or raw != canonical_json_bytes(decoded) + b"\n":
        raise ValueError("archive state is not canonical JSON")
    return decoded, raw


def _history_exists(control: int) -> bool:
    for name in ("receipts", "evictions", "catalog-heads"):
        try:
            directory = _open_child_directory(control, name)
        except FileNotFoundError:
            continue
        try:
            with os.scandir(directory) as entries:
                if next(entries, None) is not None:
                    return True
        finally:
            os.close(directory)
    try:
        catalog = _open_child_directory(control, "catalog")
    except FileNotFoundError:
        return False
    try:
        with os.scandir(catalog) as entries:
            return next(entries, None) is not None
    finally:
        os.close(catalog)


def initialize_remote_reservations(
    *,
    control_dir: Path,
    transport_id: str,
    accounted_bytes: int,
    accounting_evidence_sha256: str,
    policy: ArchivePolicy,
) -> None:
    """Create the remote-byte ledger from explicit, externally justified accounting."""
    policy = _canonical_policy(policy)
    if not isinstance(transport_id, str) or not transport_id or len(transport_id) > 4096:
        raise ValueError("invalid transport identity")
    if (
        type(accounted_bytes) is not int
        or accounted_bytes < 0
        or accounted_bytes > policy.remote_bytes
    ):
        raise ValueError("accounted bytes exceed remote byte budget")
    if not isinstance(accounting_evidence_sha256, str) or not _HASH_RE.fullmatch(
        accounting_evidence_sha256
    ):
        raise ValueError("invalid accounting evidence hash")
    state = {
        "schema_version": "phase4-r2-reservations-v1",
        "transport_id": transport_id,
        "accounting_evidence_sha256": accounting_evidence_sha256,
        "accounted_bytes": accounted_bytes,
        "reserved_bytes": accounted_bytes,
        "operational_publication_bytes": 0,
        "operational_head": None,
    }
    with _pinned_directory(control_dir, create=True) as control:
        if _history_exists(control):
            raise ValueError("remote reservation bootstrap refuses prior archive history")
        try:
            os.mkdir("remote-reservations", mode=0o700, dir_fd=control)
        except FileExistsError as error:
            raise FileExistsError("remote reservation ledger already exists") from error
        os.fsync(control)
        reservations = _open_child_directory(control, "remote-reservations")
        try:
            _create_at(reservations, "state.json", canonical_json_bytes(state) + b"\n")
            objects = _open_child_directory(reservations, "objects", create=True)
            os.close(objects)
        finally:
            os.close(reservations)


@contextmanager
def _lock(
    *, control_dir: Path, relative: tuple[str, ...], shared: bool, blocking: bool
) -> Iterator[None]:
    with _pinned_directory(control_dir, create=True) as control:
        parent = os.dup(control)
        try:
            for component in ("locks", *relative[:-1]):
                child = _open_child_directory(parent, component, create=True)
                os.close(parent)
                parent = child
            descriptor = os.open(
                relative[-1],
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("archive lock is not an owned regular file")
                operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                if not blocking:
                    operation |= fcntl.LOCK_NB
                try:
                    fcntl.flock(descriptor, operation)
                except BlockingIOError as error:
                    raise ValueError("active archive lease prevents writer lock") from error
                try:
                    yield
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent)


@contextmanager
def archive_operation_lock(control_dir: Path) -> Iterator[None]:
    """Serialize reservation, receipt, catalog, and eviction mutations."""
    with _lock(
        control_dir=control_dir,
        relative=("archive.lock",),
        shared=False,
        blocking=True,
    ):
        yield


@contextmanager
def unit_reader_lease(*, control_dir: Path, ref: UnitRef) -> Iterator[None]:
    """Hold a real shared lease preventing payload eviction for one unit."""
    _validate_ref(ref)
    stripe = int(ref.unit_id[:8], 16) % 64
    with _lock(
        control_dir=control_dir,
        relative=("unit-stripes", f"{stripe:02d}.lock"),
        shared=True,
        blocking=True,
    ):
        yield


@contextmanager
def _unit_writer_lock(*, control_dir: Path, ref: UnitRef) -> Iterator[None]:
    stripe = int(ref.unit_id[:8], 16) % 64
    with _lock(
        control_dir=control_dir,
        relative=("unit-stripes", f"{stripe:02d}.lock"),
        shared=False,
        blocking=False,
    ):
        yield


def _validate_ref(ref: UnitRef) -> None:
    if not isinstance(ref, UnitRef) or not _HASH_RE.fullmatch(ref.unit_id):
        raise ValueError("invalid unit reference")


def _validated_state(state: dict, *, transport_id: str, policy: ArchivePolicy) -> int:
    required = {
        "schema_version",
        "transport_id",
        "accounting_evidence_sha256",
        "accounted_bytes",
        "reserved_bytes",
        "operational_publication_bytes",
        "operational_head",
    }
    if set(state) != required or state.get("schema_version") != "phase4-r2-reservations-v1":
        raise ValueError("remote reservation ledger schema differs")
    if state.get("transport_id") != transport_id:
        raise ValueError("remote reservation ledger belongs to another transport")
    evidence = state.get("accounting_evidence_sha256")
    accounted = state.get("accounted_bytes")
    reserved = state.get("reserved_bytes")
    publication = state.get("operational_publication_bytes")
    head = state.get("operational_head")
    if (
        not isinstance(evidence, str)
        or not _HASH_RE.fullmatch(evidence)
        or type(accounted) is not int
        or type(reserved) is not int
        or accounted < 0
        or reserved < accounted
        or reserved > policy.remote_bytes
        or type(publication) is not int
        or publication < 0
        or publication > reserved
        or (head is not None and not isinstance(head, dict))
    ):
        raise ValueError("remote reservation ledger values differ")
    return reserved


@contextmanager
def _opened_ledger(
    control_dir: Path, *, transport_id: str, policy: ArchivePolicy
) -> Iterator[tuple[dict, int, int]]:
    reservations_context = _pinned_directory(control_dir / "remote-reservations")
    reservations = None
    objects = None
    try:
        reservations = reservations_context.__enter__()
        state, _ = _read_json_at(reservations, "state.json", max_bytes=16_384)
        _validated_state(state, transport_id=transport_id, policy=policy)
        objects = _open_child_directory(reservations, "objects")
    except (FileNotFoundError, OSError, ValueError) as error:
        if objects is not None:
            os.close(objects)
        if reservations is not None:
            reservations_context.__exit__(None, None, None)
        if isinstance(error, ValueError) and "transport" in str(error):
            raise
        raise ValueError("remote reservation ledger is absent or corrupt") from error
    try:
        assert reservations is not None and objects is not None
        yield state, reservations, objects
    finally:
        os.close(objects)
        reservations_context.__exit__(None, None, None)


def _recover_pending(
    *, state: dict, reservations: int, objects: int, transport_id: str, policy: ArchivePolicy
) -> None:
    try:
        pending, _ = _read_json_at(reservations, "pending.json", max_bytes=32_768)
    except FileNotFoundError:
        return
    required = {"schema_version", "transport_id", "before", "after", "reservation"}
    reservation = pending.get("reservation")
    if (
        set(pending) != required
        or pending.get("schema_version") != "phase4-r2-reservation-pending-v1"
        or pending.get("transport_id") != transport_id
        or type(pending.get("before")) is not int
        or type(pending.get("after")) is not int
        or not isinstance(reservation, dict)
    ):
        raise ValueError("remote reservation pending record is corrupt")
    before, after = pending["before"], pending["after"]
    if (
        after <= before
        or after > policy.remote_bytes
        or state["reserved_bytes"] not in {before, after}
    ):
        raise ValueError("remote reservation pending totals differ")
    if state["reserved_bytes"] == before:
        state["reserved_bytes"] = after
        _replace_at(reservations, "state.json", canonical_json_bytes(state) + b"\n")
    leaf = reservation.get("reservation_id")
    if not isinstance(leaf, str) or not _HASH_RE.fullmatch(leaf):
        raise ValueError("remote reservation identifier differs")
    payload = canonical_json_bytes(reservation) + b"\n"
    try:
        _create_at(objects, f"{leaf}.json", payload)
    except FileExistsError:
        if _read_at(objects, f"{leaf}.json", max_bytes=len(payload)) != payload:
            raise ValueError("remote reservation record collision") from None
    os.unlink("pending.json", dir_fd=reservations)
    os.fsync(reservations)


def _reserve_object(
    *,
    control_dir: Path,
    transport: ObjectTransport,
    transport_id: str,
    key: str,
    size: int,
    digest: str,
    policy: ArchivePolicy,
) -> None:
    reservation_id = reservation_key(key)
    record = {
        "schema_version": "phase4-r2-object-reservation-v1",
        "reservation_id": reservation_id,
        "transport_id": transport_id,
        "key": key,
        "bytes": size,
        "sha256": digest,
    }
    payload = canonical_json_bytes(record) + b"\n"
    with _opened_ledger(control_dir, transport_id=transport_id, policy=policy) as (
        state,
        reservations,
        objects,
    ):
        _recover_pending(
            state=state,
            reservations=reservations,
            objects=objects,
            transport_id=transport_id,
            policy=policy,
        )
        try:
            existing = _read_at(objects, f"{reservation_id}.json", max_bytes=len(payload))
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != payload:
                raise ValueError("remote object reservation collision")
            return
        if operational_ref := _operational_ref(state):
            archived = lookup_remote_reservation(
                control_dir,
                operational_ref,
                object_key=key,
                policy=policy,
                object_reader=_operational_cold_reader(
                    control_dir=control_dir,
                    transport=transport,
                ),
            )
            if archived is not None:
                if archived.bytes != size or archived.sha256 != digest:
                    raise ValueError("archived remote object reservation collision")
                return
        before = _validated_state(state, transport_id=transport_id, policy=policy)
        after = before + size
        if after > policy.remote_bytes:
            raise ValueError("remote byte budget refuses object reservation")
        pending = {
            "schema_version": "phase4-r2-reservation-pending-v1",
            "transport_id": transport_id,
            "before": before,
            "after": after,
            "reservation": record,
        }
        _create_at(reservations, "pending.json", canonical_json_bytes(pending) + b"\n")
        state["reserved_bytes"] = after
        _replace_at(reservations, "state.json", canonical_json_bytes(state) + b"\n")
        _create_at(objects, f"{reservation_id}.json", payload)
        os.unlink("pending.json", dir_fd=reservations)
        os.fsync(reservations)


def _transport_identity(transport: ObjectTransport) -> str:
    identity = getattr(transport, "transport_id", None)
    if not isinstance(identity, str) or not identity or len(identity) > 4096:
        raise ValueError("transport identity is unavailable")
    return identity


def _run_namespace(run_id: str) -> str:
    return sha256_bytes(run_id.encode())


def _object_key(run_id: str, logical: str) -> str:
    return f"runs/{_run_namespace(run_id)}/{logical}"


def _operational_object_key(logical: str) -> str:
    return f"operational/{logical}"


def _fresh_download(control_dir: Path) -> Path:
    directory = control_dir / "transfer-scratch"
    with _pinned_directory(directory, create=True):
        pass
    return directory / f"readback-{secrets.token_hex(16)}.part"


def _read_regular(path: Path, *, max_bytes: int) -> bytes:
    with _pinned_directory(path.parent) as parent:
        return _read_at(parent, path.name, max_bytes=max_bytes)


def _regular_identity(path: Path) -> tuple[int, int, int, int]:
    with _pinned_directory(path.parent) as parent:
        current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
    if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1:
        raise ValueError("fresh transfer object is not an owned regular file")
    return current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns


def _remove_fresh(path: Path, *, expected_identity: tuple[int, int, int, int]) -> None:
    with suppress(FileNotFoundError), _pinned_directory(path.parent) as parent:
        current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ) != expected_identity:
            return
        os.unlink(path.name, dir_fd=parent)
        os.fsync(parent)


def _put_verified(
    *,
    control_dir: Path,
    transport: ObjectTransport,
    key: str,
    source: Path,
    expected_bytes: int,
    expected_sha256: str,
    policy: ArchivePolicy,
) -> dict:
    transport_id = _transport_identity(transport)
    _reserve_object(
        control_dir=control_dir,
        transport=transport,
        transport_id=transport_id,
        key=key,
        size=expected_bytes,
        digest=expected_sha256,
        policy=policy,
    )
    return _transfer_verified(
        control_dir=control_dir,
        transport=transport,
        key=key,
        source=source,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )


def _transfer_verified(
    *,
    control_dir: Path,
    transport: ObjectTransport,
    key: str,
    source: Path,
    expected_bytes: int,
    expected_sha256: str,
) -> dict:
    create_error: BaseException | None = None
    try:
        transport.create(key, source)
    except (FileExistsError, AmbiguousWriteError) as error:
        create_error = error
    except BaseException:
        raise
    destination = _fresh_download(control_dir)
    destination_identity = None
    try:
        transport.download(key, destination, max_bytes=expected_bytes)
        destination_identity = _regular_identity(destination)
        raw = _read_regular(destination, max_bytes=expected_bytes)
        if len(raw) != expected_bytes or sha256_bytes(raw) != expected_sha256:
            reason = "collision" if isinstance(create_error, FileExistsError) else "readback"
            raise ValueError(f"remote {reason} bytes differ from reserved object")
    except BaseException as error:
        if isinstance(error, ValueError) and "collision" in str(error):
            raise
        raise ValueError(f"remote readback failed: {error}") from error
    finally:
        if destination_identity is not None:
            _remove_fresh(destination, expected_identity=destination_identity)
    return {"key": key, "bytes": expected_bytes, "sha256": expected_sha256}


def _catalog_head_path(run_id: str) -> str:
    return f"catalog-heads/{sha256_bytes(run_id.encode())}.json"


def _read_control_optional(control_dir: Path, path: str, *, max_bytes: int) -> bytes | None:
    try:
        return _control_reader(control_dir)(path, max_bytes)
    except FileNotFoundError:
        return None


def _catalog_ref_payload(ref: CatalogRef) -> dict:
    return {
        "catalog_id": ref.catalog_id,
        "scope": ref.scope,
        "run_id": ref.run_id,
        "entry_count": ref.entry_count,
        "root_path": ref.root_path,
    }


def _catalog_ref_from_payload(value: object) -> CatalogRef:
    if not isinstance(value, dict) or set(value) != {
        "catalog_id",
        "scope",
        "run_id",
        "entry_count",
        "root_path",
    }:
        raise ValueError("catalog reference state differs")
    try:
        return CatalogRef(
            value["catalog_id"],
            value["scope"],
            value["run_id"],
            value["entry_count"],
            value["root_path"],
        )
    except Exception as error:
        raise ValueError("catalog reference state differs") from error


def _load_head(control_dir: Path, run_id: str) -> CatalogRef | None:
    raw = _read_control_optional(control_dir, _catalog_head_path(run_id), max_bytes=16_384)
    if raw is None:
        return None
    if not raw.endswith(b"\n"):
        raise ValueError("catalog head is not canonical")
    decoded = decode_json(raw[:-1], limit=16_384)
    if raw != canonical_json_bytes(decoded) + b"\n":
        raise ValueError("catalog head is not canonical")
    return _catalog_ref_from_payload(decoded)


def _cold_reader(*, control_dir: Path, transport: ObjectTransport, run_id: str):
    def read(path: str, max_bytes: int) -> bytes:
        local = _read_control_optional(control_dir, path, max_bytes=max_bytes)
        if local is not None:
            return local
        destination = _fresh_download(control_dir)
        destination_identity = None
        try:
            transport.download(
                _object_key(run_id, path),
                destination,
                max_bytes=max_bytes,
            )
            destination_identity = _regular_identity(destination)
            return _read_regular(destination, max_bytes=max_bytes)
        finally:
            if destination_identity is not None:
                _remove_fresh(destination, expected_identity=destination_identity)

    return read


def _operational_cold_reader(*, control_dir: Path, transport: ObjectTransport):
    def read(path: str, max_bytes: int) -> bytes:
        destination = _fresh_download(control_dir)
        destination_identity = None
        try:
            transport.download(
                _operational_object_key(path),
                destination,
                max_bytes=max_bytes,
            )
            destination_identity = _regular_identity(destination)
            return _read_regular(destination, max_bytes=max_bytes)
        finally:
            if destination_identity is not None:
                _remove_fresh(destination, expected_identity=destination_identity)

    return read


def _operational_ref(state: dict) -> CatalogRef | None:
    value = state.get("operational_head")
    if value is None:
        return None
    ref = _catalog_ref_from_payload(value)
    if ref.scope != "operational" or ref.run_id is not None:
        raise ValueError("remote reservation operational head differs")
    return ref


def _stage_catalog(
    *,
    control_dir: Path,
    ref: UnitRef,
    run_id: str,
    transport: ObjectTransport,
    policy: ArchivePolicy,
) -> tuple[Path, CatalogRef]:
    scratch = control_dir / "transfer-scratch"
    with _pinned_directory(scratch, create=True) as parent:
        name = f"catalog-{secrets.token_hex(16)}"
        os.mkdir(name, mode=0o700, dir_fd=parent)
        os.fsync(parent)
    stage = scratch / name
    try:
        catalog_ref = publish_run_catalog(
            control_dir=stage,
            run_id=run_id,
            units=(ref,),
            policy=policy,
            previous=_load_head(control_dir, run_id),
            object_reader=_cold_reader(
                control_dir=control_dir,
                transport=transport,
                run_id=run_id,
            ),
            unit_reader=_control_reader(control_dir),
        )
        return stage, catalog_ref
    except BaseException:
        with _pinned_directory(scratch) as parent:
            _remove_tree_at(parent, name)
        raise


def _catalog_stage_objects(stage: Path) -> tuple[tuple[str, Path, bytes], ...]:
    result = []
    with _pinned_directory(stage) as root:
        catalog = _open_child_directory(root, "catalog")
        try:
            for group in ("nodes", "roots"):
                try:
                    directory = _open_child_directory(catalog, group)
                except FileNotFoundError:
                    continue
                try:
                    names = []
                    with os.scandir(directory) as entries:
                        for entry in entries:
                            if not entry.is_file(follow_symlinks=False) or not entry.name.endswith(
                                ".json"
                            ):
                                raise ValueError("unexpected staged catalog object")
                            names.append(entry.name)
                    for name in sorted(names):
                        raw = _read_at(directory, name, max_bytes=16 * 1024**2)
                        logical = f"catalog/{group}/{name}"
                        result.append((logical, stage / logical, raw))
                finally:
                    os.close(directory)
        finally:
            os.close(catalog)
    return tuple(result)


def _cleanup_catalog_stage(stage: Path) -> None:
    with _pinned_directory(stage.parent) as parent:
        _remove_tree_at(parent, stage.name)
        os.fsync(parent)


def _remove_control_tree(control_dir: Path, logical: str) -> None:
    path = _safe_logical_path(logical, field="control tree path")
    with _pinned_directory(control_dir) as root:
        parent = os.dup(root)
        try:
            for component in path.parts[:-1]:
                child = _open_child_directory(parent, component)
                os.close(parent)
                parent = child
            with suppress(FileNotFoundError):
                _remove_tree_at(parent, path.name)
                os.fsync(parent)
        except FileNotFoundError:
            pass
        finally:
            os.close(parent)


def _remove_verified_control_object(control_dir: Path, logical: str, expected: bytes) -> bool:
    path = _safe_logical_path(logical, field="verified catalog cleanup path")
    with _pinned_directory(control_dir) as root:
        parent = os.dup(root)
        try:
            for component in path.parts[:-1]:
                child = _open_child_directory(parent, component)
                os.close(parent)
                parent = child
            try:
                descriptor = os.open(
                    path.name,
                    os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                    dir_fd=parent,
                )
            except FileNotFoundError:
                return False
            try:
                initial = os.fstat(descriptor)
                if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
                    return False
                raw = bytearray()
                while len(raw) <= len(expected):
                    block = os.read(descriptor, min(65_536, len(expected) + 1 - len(raw)))
                    if not block:
                        break
                    raw.extend(block)
                final = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            identity = (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
            if (
                bytes(raw) != expected
                or (
                    final.st_dev,
                    final.st_ino,
                    final.st_size,
                    final.st_mtime_ns,
                )
                != identity
            ):
                return False
            current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
            ) != identity:
                return False
            os.unlink(path.name, dir_fd=parent)
            os.fsync(parent)
            return True
        except FileNotFoundError:
            return False
        finally:
            os.close(parent)


def _catalog_proof_prefix(unit_id: str, catalog_id: str) -> str:
    if not _HASH_RE.fullmatch(unit_id) or not _HASH_RE.fullmatch(catalog_id):
        raise ValueError("invalid catalog proof identity")
    return f"catalog-proofs/{unit_id}/{catalog_id}"


def _capture_catalog_proof(
    *,
    control_dir: Path,
    ref: UnitRef,
    catalog_ref: CatalogRef,
    transport: ObjectTransport,
    run_id: str,
    policy: ArchivePolicy,
) -> dict[str, bytes]:
    captured: dict[str, bytes] = {}
    remote_reader = _cold_reader(
        control_dir=control_dir,
        transport=transport,
        run_id=run_id,
    )

    def recording_reader(path: str, max_bytes: int) -> bytes:
        raw = remote_reader(path, max_bytes)
        captured[path] = raw
        return raw

    verify_run_catalog_unit(
        control_dir,
        catalog_ref,
        ref,
        policy=policy,
        object_reader=recording_reader,
        unit_reader=_control_reader(control_dir),
    )
    return captured


def _store_catalog_proof(
    *, control_dir: Path, ref: UnitRef, catalog_ref: CatalogRef, proof: dict[str, bytes]
) -> list[dict]:
    prefix = _catalog_proof_prefix(ref.unit_id, catalog_ref.catalog_id)
    descriptors = []
    for path, raw in sorted(proof.items()):
        _create_control_object(control_dir, f"{prefix}/{path}", raw)
        descriptors.append({"path": path, "bytes": len(raw), "sha256": sha256_bytes(raw)})
    if not any(record["path"] == catalog_ref.root_path for record in descriptors):
        raise ValueError("catalog proof lacks its authenticated root")
    return descriptors


def _verify_receipt_catalog(
    *, control_dir: Path, ref: UnitRef, receipt: dict, policy: ArchivePolicy
) -> CatalogRef:
    catalog_ref = _catalog_ref_from_payload(receipt.get("catalog"))
    records = receipt.get("catalog_proof")
    if not isinstance(records, list) or not records:
        raise ValueError("receipt catalog proof differs")
    descriptors: dict[str, tuple[int, str]] = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "bytes", "sha256"}
            or not isinstance(record["path"], str)
            or type(record["bytes"]) is not int
            or record["bytes"] <= 0
            or record["bytes"] > policy.page_bytes
            or not isinstance(record["sha256"], str)
            or not _HASH_RE.fullmatch(record["sha256"])
            or record["path"] in descriptors
        ):
            raise ValueError("receipt catalog proof differs")
        descriptors[record["path"]] = (record["bytes"], record["sha256"])
    prefix = _catalog_proof_prefix(ref.unit_id, catalog_ref.catalog_id)

    def proof_reader(path: str, max_bytes: int) -> bytes:
        descriptor = descriptors.get(path)
        if descriptor is None or descriptor[0] > max_bytes:
            raise ValueError("receipt catalog proof is incomplete")
        raw = _control_reader(control_dir)(f"{prefix}/{path}", descriptor[0])
        if len(raw) != descriptor[0] or sha256_bytes(raw) != descriptor[1]:
            raise ValueError("receipt catalog proof object differs")
        return raw

    verify_run_catalog_unit(
        control_dir,
        catalog_ref,
        ref,
        policy=policy,
        object_reader=proof_reader,
        unit_reader=_control_reader(control_dir),
    )
    return catalog_ref


def _publish_head(control_dir: Path, run_id: str, ref: CatalogRef) -> None:
    logical = _safe_logical_path(_catalog_head_path(run_id), field="catalog head path")
    payload = canonical_json_bytes(_catalog_ref_payload(ref)) + b"\n"
    with _pinned_directory(control_dir, create=True) as control:
        parent = os.dup(control)
        try:
            for component in logical.parts[:-1]:
                child = _open_child_directory(parent, component, create=True)
                os.close(parent)
                parent = child
            try:
                existing = _read_at(parent, logical.name, max_bytes=16_384)
            except FileNotFoundError:
                _create_at(parent, logical.name, payload)
            else:
                if existing != payload:
                    _replace_at(parent, logical.name, payload)
        finally:
            os.close(parent)


def _existing_receipt(
    *,
    control_dir: Path,
    ref: UnitRef,
    run_id: str,
    transport: ObjectTransport,
    policy: ArchivePolicy,
) -> str | None:
    logical = f"receipts/{ref.unit_id}.json"
    raw = _read_control_optional(control_dir, logical, max_bytes=policy.metadata_bytes)
    if raw is None:
        return None
    digest = sha256_bytes(raw)
    receipt, _ = _read_receipt(control_dir, ref, digest, policy.metadata_bytes)
    if (
        receipt.get("transport_id") != _transport_identity(transport)
        or receipt.get("run_id") != run_id
    ):
        raise ValueError("existing transfer receipt belongs to another transport or run")
    try:
        receipt_policy = ArchivePolicy.model_validate(receipt.get("policy"))
    except Exception as error:
        raise ValueError("existing transfer receipt policy differs") from error
    if receipt_policy != policy:
        raise ValueError("existing transfer receipt policy differs")
    objects = receipt.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("existing transfer receipt object inventory differs")
    for record in objects:
        if (
            not isinstance(record, dict)
            or set(record) != {"key", "bytes", "sha256"}
            or not isinstance(record["key"], str)
            or type(record["bytes"]) is not int
            or record["bytes"] <= 0
            or not isinstance(record["sha256"], str)
            or not _HASH_RE.fullmatch(record["sha256"])
        ):
            raise ValueError("existing transfer receipt object inventory differs")
        destination = _fresh_download(control_dir)
        destination_identity = None
        try:
            transport.download(record["key"], destination, max_bytes=record["bytes"])
            destination_identity = _regular_identity(destination)
            downloaded = _read_regular(destination, max_bytes=record["bytes"])
            if len(downloaded) != record["bytes"] or sha256_bytes(downloaded) != record["sha256"]:
                raise ValueError("existing transfer receipt readback differs")
        finally:
            if destination_identity is not None:
                _remove_fresh(destination, expected_identity=destination_identity)
    _verify_receipt_catalog(control_dir=control_dir, ref=ref, receipt=receipt, policy=policy)
    if not _active_receipt_authorized(control_dir=control_dir, ref=ref, receipt_sha256=digest):
        _publish_operational_batch(
            control_dir=control_dir,
            transport=transport,
            policy=policy,
            receipt_unit_id=ref.unit_id,
            receipt_sha256=digest,
        )
        if not _active_receipt_authorized(control_dir=control_dir, ref=ref, receipt_sha256=digest):
            raise ValueError("existing transfer receipt lacks operational authorization")
    return digest


def _stage_operational_objects(stage: Path) -> tuple[tuple[str, Path, bytes], ...]:
    result = []
    for directory, logical in (
        (stage / "catalog/nodes", "catalog/nodes"),
        (stage / "operational/roots", "operational/roots"),
    ):
        if not directory.exists():
            continue
        with _pinned_directory(directory) as parent:
            names = []
            with os.scandir(parent) as entries:
                for entry in entries:
                    if not entry.is_file(follow_symlinks=False) or not entry.name.endswith(".json"):
                        raise ValueError("unexpected staged operational object")
                    names.append(entry.name)
            for name in sorted(names):
                raw = _read_at(parent, name, max_bytes=16 * 1024**2)
                path = f"{logical}/{name}"
                result.append((_operational_object_key(path), directory / name, raw))
    return tuple(result)


def _local_reservation_records(control_dir: Path) -> tuple[RemoteReservationEntry, ...]:
    records = []
    with _pinned_directory(control_dir / "remote-reservations/objects") as objects:
        with os.scandir(objects) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            value, _ = _read_json_at(objects, name, max_bytes=32_768)
            object_key = value.get("key")
            records.append(
                RemoteReservationEntry(
                    key=reservation_key(object_key),
                    object_key=object_key,
                    bytes=value.get("bytes"),
                    sha256=value.get("sha256"),
                )
            )
    return tuple(records)


def _completed_intents(
    control_dir: Path,
) -> tuple[tuple[EvictionLocatorEntry, str, bytes, str], ...]:
    directory = control_dir / "evictions"
    if not directory.exists():
        return ()
    result = []
    with _pinned_directory(directory) as parent:
        with os.scandir(parent) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            value, raw = _read_json_at(parent, name, max_bytes=16 * 1024**2)
            if value.get("completed") is not True:
                continue
            unit_id = value.get("unit_id")
            intent_sha256 = sha256_bytes(raw)
            object_key = _operational_object_key(f"evictions/{unit_id}-{intent_sha256}.json")
            result.append(
                (
                    EvictionLocatorEntry(
                        key=eviction_key(unit_id),
                        unit_id=unit_id,
                        receipt_sha256=value.get("receipt_sha256"),
                        intent_sha256=intent_sha256,
                        object_key=object_key,
                        bytes=len(raw),
                        completed=True,
                    ),
                    f"evictions/{name}",
                    raw,
                    f"receipts/{unit_id}.json",
                )
            )
    return tuple(result)


def _remove_control_object(control_dir: Path, logical: str) -> None:
    path = _safe_logical_path(logical, field="operational cleanup path")
    with _pinned_directory(control_dir) as root:
        parent = os.dup(root)
        try:
            for component in path.parts[:-1]:
                child = _open_child_directory(parent, component)
                os.close(parent)
                parent = child
            with suppress(FileNotFoundError):
                os.unlink(path.name, dir_fd=parent)
                os.fsync(parent)
        except FileNotFoundError:
            pass
        finally:
            os.close(parent)


def _finalize_operational_pending(
    *, control_dir: Path, pending: dict, transport_id: str, policy: ArchivePolicy
) -> None:
    authorization = pending.get("authorization")
    if authorization is not None:
        payload = canonical_json_bytes(authorization) + b"\n"
        path = f"active-receipts/{authorization['unit_id']}.json"
        try:
            _create_control_object(control_dir, path, payload)
        except ValueError:
            if _control_reader(control_dir)(path, len(payload)) != payload:
                raise ValueError("active receipt authorization collision") from None
    with _opened_ledger(control_dir, transport_id=transport_id, policy=policy) as (
        state,
        reservations,
        _objects,
    ):
        if state["reserved_bytes"] != pending["after"]:
            raise ValueError("operational pending reservation total differs")
        state["operational_head"] = pending["new_head"]
        state["operational_publication_bytes"] = pending["publication_bytes"]
        _replace_at(reservations, "state.json", canonical_json_bytes(state) + b"\n")
        pending["committed"] = True
        _replace_at(
            reservations,
            "operational-pending.json",
            canonical_json_bytes(pending) + b"\n",
        )
    for logical in pending["cleanup"]:
        _remove_control_object(control_dir, logical)
    for logical in pending.get("cleanup_trees", ()):
        _remove_control_tree(control_dir, logical)
    stage = control_dir / pending["stage"]
    if stage.exists():
        _cleanup_catalog_stage(stage)
    with _opened_ledger(control_dir, transport_id=transport_id, policy=policy) as (
        _state,
        reservations,
        _objects,
    ):
        os.unlink("operational-pending.json", dir_fd=reservations)
        os.fsync(reservations)


def _resume_operational_pending(
    *,
    control_dir: Path,
    transport: ObjectTransport,
    policy: ArchivePolicy,
) -> None:
    transport_id = _transport_identity(transport)
    with _opened_ledger(control_dir, transport_id=transport_id, policy=policy) as (
        state,
        reservations,
        _objects,
    ):
        try:
            pending, _ = _read_json_at(
                reservations,
                "operational-pending.json",
                max_bytes=policy.metadata_bytes,
            )
        except FileNotFoundError:
            return
        before = pending.get("before")
        after = pending.get("after")
        records = pending.get("objects")
        total = (
            sum(record.get("bytes", -1) for record in records)
            if isinstance(records, list) and all(isinstance(record, dict) for record in records)
            else -1
        )
        if (
            pending.get("schema_version") != "phase4-r2-operational-pending-v1"
            or pending.get("transport_id") != transport_id
            or type(before) is not int
            or type(after) is not int
            or total < 1
            or after - before != total
            or state["reserved_bytes"] not in {before, after}
        ):
            raise ValueError("operational pending history is corrupt")
        if state["reserved_bytes"] == before:
            state["reserved_bytes"] = after
            _replace_at(reservations, "state.json", canonical_json_bytes(state) + b"\n")
    if pending.get("committed") is not True:
        for record in pending["objects"]:
            source = control_dir / record["source"]
            raw = _read_regular(source, max_bytes=record["bytes"])
            if len(raw) != record["bytes"] or sha256_bytes(raw) != record["sha256"]:
                raise ValueError("operational pending source differs")
            _transfer_verified(
                control_dir=control_dir,
                transport=transport,
                key=record["key"],
                source=source,
                expected_bytes=record["bytes"],
                expected_sha256=record["sha256"],
            )
    _finalize_operational_pending(
        control_dir=control_dir,
        pending=pending,
        transport_id=transport_id,
        policy=policy,
    )


def _publish_operational_batch(
    *,
    control_dir: Path,
    transport: ObjectTransport,
    policy: ArchivePolicy,
    receipt_unit_id: str | None = None,
    receipt_sha256: str | None = None,
) -> None:
    if (receipt_unit_id is None) != (receipt_sha256 is None):
        raise ValueError("operational receipt identity requires both fields")
    transport_id = _transport_identity(transport)
    receipts, extras = (), []
    if receipt_unit_id is not None:
        receipt_path = f"receipts/{receipt_unit_id}.json"
        receipt_raw = _control_reader(control_dir)(receipt_path, policy.metadata_bytes)
        receipt_object_key = _operational_object_key(
            f"receipts/{receipt_unit_id}-{receipt_sha256}.json"
        )
        receipts = (
            ReceiptLocatorEntry(
                key=receipt_key(receipt_unit_id),
                unit_id=receipt_unit_id,
                receipt_sha256=receipt_sha256,
                object_key=receipt_object_key,
                bytes=len(receipt_raw),
            ),
        )
        extras.append((receipt_object_key, receipt_path, receipt_raw))
    reservations = _local_reservation_records(control_dir)
    completed = _completed_intents(control_dir)
    evictions = tuple(item[0] for item in completed)
    extras.extend((entry.object_key, source, raw) for entry, source, raw, _ in completed)
    with _opened_ledger(control_dir, transport_id=transport_id, policy=policy) as (
        state,
        _reservations,
        _objects,
    ):
        previous = _operational_ref(state)
        previous_bytes = state["operational_publication_bytes"]
        before = state["reserved_bytes"]
    candidate = previous_bytes
    stage = None
    for _attempt in range(8):
        if stage is not None:
            _cleanup_catalog_stage(stage)
        scratch = control_dir / "transfer-scratch"
        with _pinned_directory(scratch, create=True) as parent:
            name = f"operational-{secrets.token_hex(16)}"
            os.mkdir(name, mode=0o700, dir_fd=parent)
            os.fsync(parent)
        stage = scratch / name
        new_head = publish_operational_catalog(
            control_dir=stage,
            reservations=reservations,
            receipts=receipts,
            evictions=evictions,
            publication_bytes=candidate,
            policy=policy,
            previous=previous,
            object_reader=_operational_cold_reader(
                control_dir=control_dir,
                transport=transport,
            ),
        )
        staged = _stage_operational_objects(stage)
        total = sum(len(raw) for _key, _source, raw in staged) + sum(
            len(raw) for _key, _source, raw in extras
        )
        updated = previous_bytes + total
        if updated == candidate:
            break
        candidate = updated
    else:
        raise ValueError("operational publication reservation did not converge")
    assert stage is not None
    descriptors = [
        {
            "key": key,
            "source": source.relative_to(control_dir).as_posix(),
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        }
        for key, source, raw in staged
    ]
    descriptors.extend(
        {
            "key": key,
            "source": source,
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
        }
        for key, source, raw in extras
    )
    after = before + total
    if after > policy.remote_bytes:
        raise ValueError("remote byte budget refuses operational publication")
    cleanup = [f"remote-reservations/objects/{record.key}.json" for record in reservations]
    cleanup_trees = []
    for _entry, intent_path, _raw, receipt_cleanup in completed:
        cleanup.extend((intent_path, receipt_cleanup, f"active-receipts/{_entry.unit_id}.json"))
        cleanup_trees.append(f"catalog-proofs/{_entry.unit_id}")
    authorization = (
        None
        if receipt_unit_id is None
        else {
            "schema_version": "phase4-r2-active-receipt-v1",
            "unit_id": receipt_unit_id,
            "receipt_sha256": receipt_sha256,
            "operational_head": _catalog_ref_payload(new_head),
        }
    )
    pending = {
        "schema_version": "phase4-r2-operational-pending-v1",
        "transport_id": transport_id,
        "before": before,
        "after": after,
        "publication_bytes": candidate,
        "new_head": _catalog_ref_payload(new_head),
        "stage": stage.relative_to(control_dir).as_posix(),
        "objects": descriptors,
        "cleanup": cleanup,
        "cleanup_trees": cleanup_trees,
        "authorization": authorization,
        "committed": False,
    }
    with _opened_ledger(control_dir, transport_id=transport_id, policy=policy) as (
        state,
        ledger,
        _objects,
    ):
        if state["reserved_bytes"] != before:
            raise ValueError("remote reservation total changed during operational staging")
        _create_at(
            ledger,
            "operational-pending.json",
            canonical_json_bytes(pending) + b"\n",
        )
        state["reserved_bytes"] = after
        _replace_at(ledger, "state.json", canonical_json_bytes(state) + b"\n")
    _resume_operational_pending(
        control_dir=control_dir,
        transport=transport,
        policy=policy,
    )


def archive_unit(
    *,
    run_dir: Path,
    control_dir: Path,
    ref: UnitRef,
    transport: ObjectTransport,
    policy: ArchivePolicy,
) -> str:
    """Upload, fully read back, catalog, and durably receipt one sealed unit."""
    policy = _canonical_policy(policy)
    _validate_ref(ref)
    transport_id = _transport_identity(transport)
    with archive_operation_lock(control_dir), _unit_writer_lock(control_dir=control_dir, ref=ref):
        with _opened_ledger(control_dir, transport_id=transport_id, policy=policy) as opened:
            _recover_pending(
                state=opened[0],
                reservations=opened[1],
                objects=opened[2],
                transport_id=transport_id,
                policy=policy,
            )
        _resume_operational_pending(
            control_dir=control_dir,
            transport=transport,
            policy=policy,
        )
        objects: list[dict] = []
        with _opened_unit(control_dir, ref) as (unit, manifest):
            run_id = manifest.identity.run_id
            chunks = manifest.chunks
            metadata = [
                (ref.manifest_path, _read_at(unit, "manifest.json", max_bytes=16 * 1024**2))
            ]
            for shard in manifest.inventory_shards:
                metadata.append(
                    (
                        f"units/{ref.unit_id}/{shard.path}",
                        _read_at(unit, shard.path, max_bytes=shard.decoded_bytes),
                    )
                )
        if existing := _existing_receipt(
            control_dir=control_dir,
            ref=ref,
            run_id=run_id,
            transport=transport,
            policy=policy,
        ):
            return existing
        scratch = control_dir / "transfer-scratch"
        for index, chunk_path in enumerate(
            iter_unit_chunks(
                run_dir=run_dir,
                control_dir=control_dir,
                ref=ref,
                scratch_dir=scratch,
                policy=policy,
            )
        ):
            descriptor = chunks[index]
            objects.append(
                _put_verified(
                    control_dir=control_dir,
                    transport=transport,
                    key=_object_key(run_id, f"objects/{descriptor.sha256}.bin"),
                    source=chunk_path,
                    expected_bytes=descriptor.bytes,
                    expected_sha256=descriptor.sha256,
                    policy=policy,
                )
            )
        for logical, raw in metadata:
            source = _fresh_download(control_dir)
            source_identity = None
            try:
                with _pinned_directory(source.parent) as parent:
                    _create_at(parent, source.name, raw)
                source_identity = _regular_identity(source)
                objects.append(
                    _put_verified(
                        control_dir=control_dir,
                        transport=transport,
                        key=_object_key(run_id, logical),
                        source=source,
                        expected_bytes=len(raw),
                        expected_sha256=sha256_bytes(raw),
                        policy=policy,
                    )
                )
            finally:
                if source_identity is not None:
                    _remove_fresh(source, expected_identity=source_identity)
        stage, catalog_ref = _stage_catalog(
            control_dir=control_dir,
            ref=ref,
            run_id=run_id,
            transport=transport,
            policy=policy,
        )
        catalog_proof = None
        try:
            staged = _catalog_stage_objects(stage)
            for logical, source, raw in staged:
                objects.append(
                    _put_verified(
                        control_dir=control_dir,
                        transport=transport,
                        key=_object_key(run_id, logical),
                        source=source,
                        expected_bytes=len(raw),
                        expected_sha256=sha256_bytes(raw),
                        policy=policy,
                    )
                )
            for logical, _source, raw in staged:
                _remove_verified_control_object(control_dir, logical, raw)
            catalog_proof = _capture_catalog_proof(
                control_dir=control_dir,
                ref=ref,
                catalog_ref=catalog_ref,
                transport=transport,
                run_id=run_id,
                policy=policy,
            )
            _publish_head(control_dir, run_id, catalog_ref)
        finally:
            _cleanup_catalog_stage(stage)
        assert catalog_proof is not None
        catalog_proof_descriptors = _store_catalog_proof(
            control_dir=control_dir,
            ref=ref,
            catalog_ref=catalog_ref,
            proof=catalog_proof,
        )
        receipt = {
            "schema_version": "phase4-r2-receipt-v1",
            "transport_id": transport_id,
            "unit_id": ref.unit_id,
            "run_id": run_id,
            "catalog": _catalog_ref_payload(catalog_ref),
            "catalog_proof": catalog_proof_descriptors,
            "policy": policy.model_dump(mode="json"),
            "objects": objects,
        }
        payload = canonical_json_bytes(receipt) + b"\n"
        logical = f"receipts/{ref.unit_id}.json"
        try:
            _create_control_object(control_dir, logical, payload)
        except ValueError:
            existing = _control_reader(control_dir)(logical, policy.metadata_bytes)
            if existing != payload:
                raise ValueError("existing transfer receipt differs") from None
        receipt_sha256 = sha256_bytes(payload)
        _publish_operational_batch(
            control_dir=control_dir,
            transport=transport,
            policy=policy,
            receipt_unit_id=ref.unit_id,
            receipt_sha256=receipt_sha256,
        )
        return receipt_sha256


def _read_receipt(
    control_dir: Path, ref: UnitRef, receipt_sha256: str, policy_limit: int
) -> tuple[dict, bytes]:
    raw = _control_reader(control_dir)(f"receipts/{ref.unit_id}.json", policy_limit)
    if sha256_bytes(raw) != receipt_sha256 or not raw.endswith(b"\n"):
        raise ValueError("transfer receipt hash differs")
    decoded = decode_json(raw[:-1], limit=policy_limit)
    if (
        not isinstance(decoded, dict)
        or raw != canonical_json_bytes(decoded) + b"\n"
        or decoded.get("schema_version") != "phase4-r2-receipt-v1"
        or decoded.get("unit_id") != ref.unit_id
    ):
        raise ValueError("transfer receipt differs from unit")
    return decoded, raw


def _active_receipt_authorized(*, control_dir: Path, ref: UnitRef, receipt_sha256: str) -> bool:
    authorization_raw = _read_control_optional(
        control_dir,
        f"active-receipts/{ref.unit_id}.json",
        max_bytes=16_384,
    )
    if authorization_raw is None:
        return False
    if not authorization_raw.endswith(b"\n"):
        raise ValueError("receipt operational authorization differs")
    authorization = decode_json(authorization_raw[:-1], limit=16_384)
    if (
        authorization_raw != canonical_json_bytes(authorization) + b"\n"
        or authorization.get("schema_version") != "phase4-r2-active-receipt-v1"
        or authorization.get("unit_id") != ref.unit_id
        or authorization.get("receipt_sha256") != receipt_sha256
    ):
        raise ValueError("receipt operational authorization differs")
    return True


def _inventory_snapshot(run_dir: Path, entries: tuple) -> tuple[dict, ...]:
    records = []
    for entry in entries:
        member = _safe_logical_path(entry.path, field="eviction member")
        digest = hashlib.sha256()
        with _source_file(run_dir, member) as (descriptor, initial):
            while block := os.read(descriptor, 65_536):
                digest.update(block)
            final = os.fstat(descriptor)
        if (
            digest.hexdigest() != entry.sha256
            or initial.st_size != entry.bytes
            or (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
            != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        ):
            raise ValueError("archive source changed since receipt")
        records.append(
            {
                "path": entry.path,
                "sha256": entry.sha256,
                "bytes": entry.bytes,
                "device": initial.st_dev,
                "inode": initial.st_ino,
                "mtime_ns": initial.st_mtime_ns,
            }
        )
    return tuple(records)


def _resident_files(run_dir: Path, logical_root: str) -> set[str]:
    root = _safe_logical_path(logical_root, field="logical root", allow_root_dot=True)
    start = run_dir if logical_root == "." else run_dir.joinpath(*root.parts)
    if not start.exists():
        return set()
    found = set()
    with _pinned_directory(start) as descriptor:
        pending: list[tuple[int, str]] = [(os.dup(descriptor), logical_root)]
        try:
            while pending:
                directory, prefix = pending.pop()
                try:
                    with os.scandir(directory) as entries:
                        for entry in entries:
                            relative = entry.name if prefix == "." else f"{prefix}/{entry.name}"
                            if entry.is_dir(follow_symlinks=False):
                                pending.append(
                                    (_open_child_directory(directory, entry.name), relative)
                                )
                            elif entry.is_file(follow_symlinks=False):
                                found.add(relative)
                            else:
                                raise ValueError("unknown special file stops eviction")
                finally:
                    os.close(directory)
        finally:
            for directory, _ in pending:
                os.close(directory)
    return found


def _unlink_recorded_member(run_dir: Path, record: dict) -> None:
    member = _safe_logical_path(record["path"], field="eviction member")
    try:
        with _source_file(run_dir, member) as (descriptor, initial):
            digest = hashlib.sha256()
            while block := os.read(descriptor, 65_536):
                digest.update(block)
            final = os.fstat(descriptor)
            expected = (
                record["device"],
                record["inode"],
                record["bytes"],
                record["mtime_ns"],
            )
            if (
                (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns) != expected
                or (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns) != expected
                or digest.hexdigest() != record["sha256"]
            ):
                raise ValueError("recorded eviction member changed")
        with _pinned_directory(run_dir) as root:
            parent = os.dup(root)
            try:
                for component in member.parts[:-1]:
                    child = _open_child_directory(parent, component)
                    os.close(parent)
                    parent = child
                current = os.stat(member.name, dir_fd=parent, follow_symlinks=False)
                if (
                    current.st_dev,
                    current.st_ino,
                    current.st_size,
                    current.st_mtime_ns,
                ) != expected:
                    raise ValueError("recorded eviction member changed before unlink")
                os.unlink(member.name, dir_fd=parent)
                os.fsync(parent)
            finally:
                os.close(parent)
    except FileNotFoundError:
        return


def _intent_path(ref: UnitRef) -> str:
    return f"evictions/{ref.unit_id}.json"


def _write_intent(control_dir: Path, ref: UnitRef, intent: dict, *, replace: bool) -> None:
    logical = _safe_logical_path(_intent_path(ref), field="eviction intent path")
    payload = canonical_json_bytes(intent) + b"\n"
    with _pinned_directory(control_dir, create=True) as control:
        parent = os.dup(control)
        try:
            for component in logical.parts[:-1]:
                child = _open_child_directory(parent, component, create=True)
                os.close(parent)
                parent = child
            if replace:
                _replace_at(parent, logical.name, payload)
            else:
                _create_at(parent, logical.name, payload)
        finally:
            os.close(parent)


def evict_unit(
    *,
    run_dir: Path,
    control_dir: Path,
    ref: UnitRef,
    receipt_sha256: str,
    retained: tuple[FileEntry, ...] = (),
) -> None:
    """Unlink only unchanged owned members under a durable, resumable intent."""
    _validate_ref(ref)
    if ref.kind == "control_snapshot":
        raise ValueError("control snapshot source files remain pinned")
    if not isinstance(receipt_sha256, str) or not _HASH_RE.fullmatch(receipt_sha256):
        raise ValueError("invalid transfer receipt hash")
    with archive_operation_lock(control_dir), _unit_writer_lock(control_dir=control_dir, ref=ref):
        receipt, _ = _read_receipt(control_dir, ref, receipt_sha256, 16 * 1024**2)
        if not _active_receipt_authorized(
            control_dir=control_dir,
            ref=ref,
            receipt_sha256=receipt_sha256,
        ):
            raise ValueError("receipt lacks operational authorization")
        try:
            policy = ArchivePolicy.model_validate(receipt.get("policy"))
        except Exception as error:
            raise ValueError("receipt archive policy differs") from error
        _verify_receipt_catalog(
            control_dir=control_dir,
            ref=ref,
            receipt=receipt,
            policy=policy,
        )
        intent_raw = _read_control_optional(control_dir, _intent_path(ref), max_bytes=16 * 1024**2)
        with _opened_unit(control_dir, ref) as (unit, manifest):
            entries = tuple(_iter_inventory_records_from_unit(unit, manifest))
        expected_paths = {entry.path for entry in entries}
        if type(retained) is not tuple:
            raise ValueError("retained custody must be immutable")
        previous = ""
        for entry in retained:
            if not isinstance(entry, FileEntry):
                raise ValueError("invalid retained custody entry")
            path = _safe_logical_path(entry.path, field="retained custody")
            if (
                entry.path <= previous
                or entry.path in expected_paths
                or (ref.logical_root != "." and not path.is_relative_to(ref.logical_root))
                or type(entry.bytes) is not int
                or entry.bytes < 0
                or type(entry.sha256) is not str
                or not _HASH_RE.fullmatch(entry.sha256)
            ):
                raise ValueError("retained custody ownership differs")
            previous = entry.path
        retained_records = _inventory_snapshot(run_dir, retained)
        retained_paths = {entry.path for entry in retained}
        if intent_raw is None:
            resident = _resident_files(run_dir, ref.logical_root)
            if resident != expected_paths | retained_paths:
                raise ValueError("unknown or missing files stop eviction")
            records = _inventory_snapshot(run_dir, entries)
            intent = {
                "schema_version": "phase4-r2-eviction-v1",
                "unit_id": ref.unit_id,
                "receipt_sha256": receipt_sha256,
                "members": records,
                "completed": False,
            }
            if retained:
                intent["retained"] = retained_records
            _write_intent(control_dir, ref, intent, replace=False)
        else:
            if not intent_raw.endswith(b"\n"):
                raise ValueError("eviction intent is not canonical")
            intent = decode_json(intent_raw[:-1], limit=16 * 1024**2)
            if (
                not isinstance(intent, dict)
                or intent_raw != canonical_json_bytes(intent) + b"\n"
                or intent.get("schema_version") != "phase4-r2-eviction-v1"
                or intent.get("unit_id") != ref.unit_id
                or intent.get("receipt_sha256") != receipt_sha256
                or {record.get("path") for record in intent.get("members", ())} != expected_paths
                or intent.get("retained", []) != list(retained_records)
            ):
                raise ValueError("eviction intent differs from receipt")
            resident = _resident_files(run_dir, ref.logical_root)
            if not retained_paths <= resident or not resident <= expected_paths | retained_paths:
                raise ValueError("unknown or missing retained files stop eviction")
            if intent.get("completed") is True:
                return
        for record in intent["members"]:
            _unlink_recorded_member(run_dir, record)
        intent["completed"] = True
        _write_intent(control_dir, ref, intent, replace=True)


def _validate_key(key: str) -> str:
    path = _safe_logical_path(key, field="transport object key")
    if not any(
        _HASH_RE.fullmatch(part.removesuffix(".bin").removesuffix(".json")) for part in path.parts
    ):
        raise ValueError("transport key is not content addressed")
    return path.as_posix()


def _closed_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("AWS_", "R2_", "CLOUDFLARE_"))
    }
    environment["AWS_PAGER"] = ""
    environment["AWS_CLI_AUTO_PROMPT"] = "off"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _bounded_process(
    arguments: list[str],
    *,
    pass_fds: tuple[int, ...] = (),
    file_limit: int | None = None,
) -> tuple[int, bytes, bytes]:
    def prepare() -> None:
        if file_limit is not None:
            _current_soft, current_hard = resource.getrlimit(resource.RLIMIT_FSIZE)
            hard = (
                file_limit
                if current_hard == resource.RLIM_INFINITY
                else min(file_limit, current_hard)
            )
            soft = min(file_limit, hard)
            resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
            signal.signal(signal.SIGXFSZ, signal.SIG_DFL)

    process = subprocess.Popen(
        arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_closed_environment(),
        pass_fds=pass_fds,
        preexec_fn=prepare if file_limit is not None else None,
        close_fds=True,
    )
    assert process.stdout is not None and process.stderr is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + _AWS_TIMEOUT_SECONDS

    def kill_and_reap() -> None:
        if process.poll() is None:
            process.kill()
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=1.0)

    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                kill_and_reap()
                raise _ProcessTimeout("AWS operation exceeded finite timeout")
            events = selector.select(min(remaining, 1.0))
            if not events and process.poll() is not None:
                events = [(key, selectors.EVENT_READ) for key in selector.get_map().values()]
            for key, _ in events:
                block = os.read(key.fileobj.fileno(), 16_384)
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                target = captured[key.data]
                target.extend(block)
                if len(target) > _AWS_OUTPUT_BYTES:
                    kill_and_reap()
                    raise ValueError("AWS metadata output exceeds byte bound")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            kill_and_reap()
            raise _ProcessTimeout("AWS operation exceeded finite timeout")
        try:
            return (
                process.wait(timeout=remaining),
                bytes(captured["stdout"]),
                bytes(captured["stderr"]),
            )
        except subprocess.TimeoutExpired:
            kill_and_reap()
            raise _ProcessTimeout("AWS operation exceeded finite timeout") from None
    finally:
        selector.close()
        if process.poll() is None:
            kill_and_reap()


def _failure_kind(returncode: int, stderr: bytes) -> str:
    text = stderr[:_AWS_OUTPUT_BYTES].decode("utf-8", "replace").lower()
    if "preconditionfailed" in text or "status code: 412" in text or "condition" in text:
        return "conflict"
    if any(word in text for word in ("expiredtoken", "accessdenied", "signature", "credential")):
        return "auth"
    if returncode < 0 or any(
        word in text
        for word in ("timeout", "timed out", "slowdown", "internalerror", "serviceunavailable")
    ):
        return "transient"
    return "failure"


class R2CliTransport:
    """Closed AWS CLI adapter for one fixed private bucket namespace."""

    def __init__(self, *, profile: str, bucket: str, prefix: str):
        if profile != "silent-cascade-r2":
            raise ValueError("R2 transport requires the dedicated project profile")
        if not isinstance(bucket, str) or not _BUCKET_RE.fullmatch(bucket):
            raise ValueError("invalid project bucket")
        prefix_path = _safe_logical_path(prefix, field="R2 prefix")
        self.profile = profile
        self.bucket = bucket
        self.prefix = prefix_path.as_posix()

    @property
    def transport_id(self) -> str:
        identity = sha256_bytes(
            canonical_json_bytes(
                {"profile": self.profile, "bucket": self.bucket, "prefix": self.prefix}
            )
        )
        return f"r2-cli:{identity}"

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{_validate_key(key)}"

    def create(self, key: str, source: Path) -> None:
        full_key = self._key(key)
        descriptor = os.open(source, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            initial = os.fstat(descriptor)
            if not stat.S_ISREG(initial.st_mode) or initial.st_nlink != 1:
                raise ValueError("transport source is not an owned regular file")
            arguments = [
                *_AWS_COMMAND,
                "s3api",
                "put-object",
                "--profile",
                self.profile,
                "--bucket",
                self.bucket,
                "--key",
                full_key,
                "--body",
                f"/dev/fd/{descriptor}",
                "--if-none-match",
                "*",
                "--no-cli-pager",
            ]
            ambiguous = False
            for attempt in range(_AWS_MAX_ATTEMPTS):
                try:
                    returncode, _stdout, stderr = _bounded_process(
                        arguments, pass_fds=(descriptor,)
                    )
                except _ProcessTimeout:
                    ambiguous = True
                    if attempt + 1 < _AWS_MAX_ATTEMPTS:
                        continue
                    break
                if returncode == 0:
                    final = os.fstat(descriptor)
                    if (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns) != (
                        initial.st_dev,
                        initial.st_ino,
                        initial.st_size,
                        initial.st_mtime_ns,
                    ):
                        raise ValueError("transport source changed during create")
                    return
                kind = _failure_kind(returncode, stderr)
                if kind == "conflict":
                    raise FileExistsError("remote object already exists")
                if kind == "auth":
                    raise PermissionError("R2 authentication or authorization failed")
                if kind == "transient" and attempt + 1 < _AWS_MAX_ATTEMPTS:
                    ambiguous = True
                    continue
                if kind == "transient":
                    ambiguous = True
                    break
                raise OSError("R2 create failed")
            if ambiguous:
                raise AmbiguousWriteError("R2 create completion is ambiguous")
            raise OSError("R2 create failed")
        finally:
            os.close(descriptor)

    def download(self, key: str, destination: Path, *, max_bytes: int) -> None:
        full_key = self._key(key)
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("download bound must be a positive integer")
        with _pinned_directory(destination.parent, create=True) as parent:
            descriptor = os.open(
                destination.name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent,
            )
            initial = os.fstat(descriptor)
            try:
                arguments = [
                    *_AWS_COMMAND,
                    "s3api",
                    "get-object",
                    "--profile",
                    self.profile,
                    "--bucket",
                    self.bucket,
                    "--key",
                    full_key,
                    "--range",
                    f"bytes=0-{max_bytes - 1}",
                    "--no-cli-pager",
                    f"/dev/fd/{descriptor}",
                ]
                metadata = None
                for attempt in range(_AWS_MAX_ATTEMPTS):
                    os.ftruncate(descriptor, 0)
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    try:
                        returncode, stdout, stderr = _bounded_process(
                            arguments,
                            pass_fds=(descriptor,),
                            file_limit=max_bytes,
                        )
                    except _ProcessTimeout:
                        if attempt + 1 < _AWS_MAX_ATTEMPTS:
                            continue
                        raise OSError("R2 download timed out") from None
                    if returncode == 0:
                        try:
                            metadata = json.loads(stdout)
                        except (UnicodeDecodeError, json.JSONDecodeError) as error:
                            raise ValueError("R2 range metadata is invalid") from error
                        break
                    if (
                        returncode == -signal.SIGXFSZ
                        or b"file too large" in stderr.lower()
                        or b"errno 27" in stderr.lower()
                    ):
                        raise ValueError("R2 download exceeded its local write bound")
                    kind = _failure_kind(returncode, stderr)
                    if kind == "auth":
                        raise PermissionError("R2 authentication or authorization failed")
                    if kind == "transient" and attempt + 1 < _AWS_MAX_ATTEMPTS:
                        continue
                    if kind == "transient" and returncode < 0:
                        raise ValueError("R2 download exceeded its local write bound")
                    raise OSError("R2 download failed")
                if not isinstance(metadata, dict):
                    raise OSError("R2 download failed")
                os.fsync(descriptor)
                final = os.fstat(descriptor)
                current = os.stat(destination.name, dir_fd=parent, follow_symlinks=False)
                if (current.st_dev, current.st_ino) != (initial.st_dev, initial.st_ino):
                    raise ValueError("download destination ownership changed")
                content_range = metadata.get("ContentRange")
                length = metadata.get("ContentLength")
                match = (
                    re.fullmatch(r"bytes 0-([0-9]+)/([0-9]+)", content_range)
                    if isinstance(content_range, str)
                    else None
                )
                if (
                    type(length) is not int
                    or not match
                    or int(match.group(1)) + 1 != length
                    or int(match.group(2)) != length
                    or length != final.st_size
                    or length > max_bytes
                ):
                    raise ValueError("R2 range metadata or downloaded size exceeds bound")
                os.fsync(parent)
            except BaseException:
                try:
                    current = os.stat(destination.name, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    if (current.st_dev, current.st_ino) == (initial.st_dev, initial.st_ino):
                        os.unlink(destination.name, dir_fd=parent)
                        os.fsync(parent)
                raise
            finally:
                os.close(descriptor)
