import hashlib

import pytest


def _reservation(number: int):
    from silent_cascade.archive.types import RemoteReservationEntry

    digest = f"{number:064x}"
    object_key = f"runs/{'a' * 64}/objects/{digest}.bin"
    return RemoteReservationEntry(
        key=hashlib.sha256(f"reservation\0{object_key}".encode()).hexdigest(),
        object_key=object_key,
        bytes=number + 1,
        sha256=digest,
    )


def test_operational_index_is_typed_incremental_and_cold_lazy(task_scratch):
    from silent_cascade.archive.operational import (
        lookup_remote_reservation,
        publish_operational_catalog,
    )
    from silent_cascade.archive.types import (
        ArchivePolicy,
        EvictionLocatorEntry,
        ReceiptLocatorEntry,
    )

    control = task_scratch / "operational"
    policy = ArchivePolicy(page_bytes=4096, page_entries=3)
    reservations = tuple(_reservation(number) for number in range(512))
    first = publish_operational_catalog(
        control_dir=control,
        reservations=reservations,
        receipts=(),
        evictions=(),
        publication_bytes=12_345,
        policy=policy,
    )
    unit_id = "f" * 64
    receipt = ReceiptLocatorEntry(
        key=hashlib.sha256(f"receipt\0{unit_id}".encode()).hexdigest(),
        unit_id=unit_id,
        receipt_sha256="1" * 64,
        object_key=f"runs/{'a' * 64}/operational/receipts/{'f' * 64}.json",
        bytes=123,
    )
    eviction = EvictionLocatorEntry(
        key=hashlib.sha256(f"eviction\0{unit_id}".encode()).hexdigest(),
        unit_id=unit_id,
        receipt_sha256="1" * 64,
        intent_sha256="2" * 64,
        object_key=f"runs/{'a' * 64}/operational/evictions/{'f' * 64}.json",
        bytes=234,
        completed=True,
    )
    second = publish_operational_catalog(
        control_dir=control,
        reservations=(),
        receipts=(receipt,),
        evictions=(eviction,),
        publication_bytes=13_579,
        policy=policy,
        previous=first,
    )
    objects = {
        path.relative_to(control).as_posix(): path.read_bytes() for path in control.rglob("*.json")
    }
    for path in control.rglob("*.json"):
        path.unlink()
    reads = []

    def cold_reader(path: str, max_bytes: int) -> bytes:
        reads.append(path)
        payload = objects[path]
        assert len(payload) <= max_bytes
        return payload

    found = lookup_remote_reservation(
        control,
        second,
        object_key=reservations[-1].object_key,
        policy=policy,
        object_reader=cold_reader,
    )
    assert found == reservations[-1]
    assert len(reads) < 40


def test_operational_index_rejects_corrupt_cold_page(task_scratch, tiny_archive_policy):
    from silent_cascade.archive.operational import (
        lookup_remote_reservation,
        publish_operational_catalog,
    )

    control = task_scratch / "corrupt-operational"
    record = _reservation(7)
    ref = publish_operational_catalog(
        control_dir=control,
        reservations=(record,),
        receipts=(),
        evictions=(),
        publication_bytes=99,
        policy=tiny_archive_policy,
    )
    objects = {
        path.relative_to(control).as_posix(): path.read_bytes() for path in control.rglob("*.json")
    }

    def corrupt_reader(path: str, max_bytes: int) -> bytes:
        payload = objects[path]
        return payload[:-2] + b"x\n"

    with pytest.raises(ValueError, match=r"identity|canonical|invalid|descriptor"):
        lookup_remote_reservation(
            control,
            ref,
            object_key=record.object_key,
            policy=tiny_archive_policy,
            object_reader=corrupt_reader,
        )
