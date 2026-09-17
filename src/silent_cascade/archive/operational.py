"""Bounded authenticated indexes for archive operational history."""

from collections.abc import Iterable, Iterator
from pathlib import Path

from silent_cascade.archive.catalog import (
    ObjectReader,
    _canonical_policy,
    _CatalogStore,
    _control_reader,
    _create_control_object_at,
    _insert_batch,
    _iter_tree,
    _lookup,
    _pinned_directory,
)
from silent_cascade.archive.types import (
    ArchivePolicy,
    CatalogEntry,
    CatalogRef,
    EvictionLocatorEntry,
    OperationalCatalogRoot,
    ReceiptLocatorEntry,
    RemoteReservationEntry,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.train.evidence_types import decode_json


def reservation_key(object_key: str) -> str:
    return sha256_bytes(f"reservation\0{object_key}".encode())


def receipt_key(unit_id: str) -> str:
    return sha256_bytes(f"receipt\0{unit_id}".encode())


def eviction_key(unit_id: str) -> str:
    return sha256_bytes(f"eviction\0{unit_id}".encode())


def _validate_ref(ref: CatalogRef) -> None:
    if (
        not isinstance(ref, CatalogRef)
        or ref.scope != "operational"
        or ref.run_id is not None
        or len(ref.catalog_id) != 64
        or any(character not in "0123456789abcdef" for character in ref.catalog_id)
        or ref.root_path != f"operational/roots/{ref.catalog_id}.json"
        or type(ref.entry_count) is not int
        or ref.entry_count < 0
    ):
        raise ValueError("invalid operational catalog reference")


def _load_operational_root(
    ref: CatalogRef, *, reader: ObjectReader, policy: ArchivePolicy
) -> OperationalCatalogRoot:
    _validate_ref(ref)
    raw = reader(ref.root_path, policy.page_bytes)
    if not raw.endswith(b"\n") or len(raw) > policy.page_bytes:
        raise ValueError("operational root bytes differ")
    payload = raw[:-1]
    if sha256_bytes(payload) != ref.catalog_id:
        raise ValueError("operational root identity differs")
    try:
        decoded = decode_json(payload, limit=policy.page_bytes)
        root = OperationalCatalogRoot.model_validate_json(canonical_json_bytes(decoded))
    except Exception as error:
        raise ValueError(f"invalid operational root: {error}") from error
    if raw != canonical_json_bytes(root) + b"\n":
        raise ValueError("operational root is not canonical JSON")
    count = root.reservation_count + root.receipt_count + root.eviction_count
    if count != ref.entry_count:
        raise ValueError("operational catalog count differs")
    return root


def load_operational_root(
    control_dir: Path,
    ref: CatalogRef,
    *,
    policy: ArchivePolicy,
    object_reader: ObjectReader | None = None,
) -> OperationalCatalogRoot:
    policy = _canonical_policy(policy)
    return _load_operational_root(
        ref,
        reader=object_reader or _control_reader(control_dir),
        policy=policy,
    )


def _checked_records(records: Iterable, expected_type: type, key_function) -> tuple:
    result = tuple(records)
    for record in result:
        if not isinstance(record, expected_type) or record.key != key_function(record):
            raise ValueError("operational record key or type differs")
    return result


def publish_operational_catalog(
    *,
    control_dir: Path,
    reservations: Iterable[RemoteReservationEntry],
    receipts: Iterable[ReceiptLocatorEntry],
    evictions: Iterable[EvictionLocatorEntry],
    publication_bytes: int,
    policy: ArchivePolicy,
    previous: CatalogRef | None = None,
    object_reader: ObjectReader | None = None,
) -> CatalogRef:
    """Publish one persistent operational generation using bounded changed paths."""
    policy = _canonical_policy(policy)
    if type(publication_bytes) is not int or not 0 <= publication_bytes <= policy.remote_bytes:
        raise ValueError("invalid operational publication byte count")
    reader = object_reader or _control_reader(control_dir)
    store = _CatalogStore(control_dir=control_dir, policy=policy, reader=reader)
    reservation_root = receipt_root = eviction_root = None
    if previous is not None:
        prior = _load_operational_root(previous, reader=reader, policy=policy)
        reservation_root = prior.reservations
        receipt_root = prior.receipts
        eviction_root = prior.evictions
        if publication_bytes < prior.publication_bytes:
            raise ValueError("operational publication bytes cannot decrease")
    reservation_records = _checked_records(
        reservations,
        RemoteReservationEntry,
        lambda record: reservation_key(record.object_key),
    )
    receipt_records = _checked_records(
        receipts,
        ReceiptLocatorEntry,
        lambda record: receipt_key(record.unit_id),
    )
    eviction_records = _checked_records(
        evictions,
        EvictionLocatorEntry,
        lambda record: eviction_key(record.unit_id),
    )
    reservation_root = _insert_batch(
        reservation_root,
        tuple(sorted(reservation_records, key=lambda record: record.key)),
        index="reservations",
        store=store,
    )
    receipt_root = _insert_batch(
        receipt_root,
        tuple(sorted(receipt_records, key=lambda record: record.key)),
        index="receipts",
        store=store,
    )
    eviction_root = _insert_batch(
        eviction_root,
        tuple(sorted(eviction_records, key=lambda record: record.key)),
        index="evictions",
        store=store,
        replace=True,
    )
    root = OperationalCatalogRoot(
        reservations=reservation_root,
        receipts=receipt_root,
        evictions=eviction_root,
        reservation_count=0 if reservation_root is None else reservation_root.entries,
        receipt_count=0 if receipt_root is None else receipt_root.entries,
        eviction_count=0 if eviction_root is None else eviction_root.entries,
        publication_bytes=publication_bytes,
    )
    payload = canonical_json_bytes(root)
    if len(payload) + 1 > policy.page_bytes:
        raise ValueError("operational root exceeds page byte bound")
    digest = sha256_bytes(payload)
    path = f"operational/roots/{digest}.json"
    with _pinned_directory(control_dir, create=True) as control:
        store.publish(control)
        _create_control_object_at(control, path, payload + b"\n")
    count = root.reservation_count + root.receipt_count + root.eviction_count
    return CatalogRef(digest, "operational", None, count, path)


def _lookup_record(
    control_dir: Path,
    ref: CatalogRef,
    *,
    key: str,
    index: str,
    policy: ArchivePolicy,
    object_reader: ObjectReader | None,
) -> CatalogEntry | None:
    policy = _canonical_policy(policy)
    reader = object_reader or _control_reader(control_dir)
    root = _load_operational_root(ref, reader=reader, policy=policy)
    store = _CatalogStore(control_dir=control_dir, policy=policy, reader=reader)
    tree = {
        "reservations": root.reservations,
        "receipts": root.receipts,
        "evictions": root.evictions,
    }[index]
    return _lookup(store, tree, key, index=index)


def lookup_remote_reservation(
    control_dir: Path,
    ref: CatalogRef,
    *,
    object_key: str,
    policy: ArchivePolicy,
    object_reader: ObjectReader | None = None,
) -> RemoteReservationEntry | None:
    record = _lookup_record(
        control_dir,
        ref,
        key=reservation_key(object_key),
        index="reservations",
        policy=policy,
        object_reader=object_reader,
    )
    if record is not None and not isinstance(record, RemoteReservationEntry):
        raise ValueError("operational reservation index yielded another record type")
    return record


def lookup_receipt_locator(
    control_dir: Path,
    ref: CatalogRef,
    *,
    unit_id: str,
    policy: ArchivePolicy,
    object_reader: ObjectReader | None = None,
) -> ReceiptLocatorEntry | None:
    record = _lookup_record(
        control_dir,
        ref,
        key=receipt_key(unit_id),
        index="receipts",
        policy=policy,
        object_reader=object_reader,
    )
    if record is not None and not isinstance(record, ReceiptLocatorEntry):
        raise ValueError("operational receipt index yielded another record type")
    return record


def iter_operational_records(
    control_dir: Path,
    ref: CatalogRef,
    *,
    index: str,
    policy: ArchivePolicy,
    object_reader: ObjectReader | None = None,
) -> Iterator[CatalogEntry]:
    policy = _canonical_policy(policy)
    reader = object_reader or _control_reader(control_dir)
    root = _load_operational_root(ref, reader=reader, policy=policy)
    tree = {
        "reservations": root.reservations,
        "receipts": root.receipts,
        "evictions": root.evictions,
    }.get(index)
    if index not in {"reservations", "receipts", "evictions"}:
        raise ValueError("unknown operational index")
    store = _CatalogStore(control_dir=control_dir, policy=policy, reader=reader)
    yield from _iter_tree(store, tree, index=index)
