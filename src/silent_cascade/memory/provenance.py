"""Fail-closed provenance and support-ledger checks for runtime memory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from silent_cascade.errors import DynamicsError, ProvenanceError
from silent_cascade.schemas import MemoryRecord

if TYPE_CHECKING:
    from silent_cascade.memory.store import BoundedMemory


def require_support_ledger(memory: BoundedMemory, support_ids: tuple[int, ...]) -> None:
    """Require unique support IDs that name existing valid memory records."""
    if not isinstance(support_ids, tuple):
        raise TypeError("support_ids must be a tuple")
    if len(set(support_ids)) != len(support_ids):
        raise ProvenanceError("support_ids must be unique")
    for support_id in support_ids:
        if type(support_id) is not int or support_id < 0:
            raise ProvenanceError("support_ids must be nonnegative exact integers")
        try:
            slot = memory.lookup(support_id)
        except DynamicsError as error:
            raise ProvenanceError("support_ids contain an unknown record") from error
        if not slot.valid:
            raise ProvenanceError("support_ids cannot name an invalid record")


def require_provenance_preserved(previous: MemoryRecord, current: MemoryRecord) -> None:
    """Reject record replacement that changes its immutable provenance."""
    if not isinstance(previous, MemoryRecord) or not isinstance(current, MemoryRecord):
        raise TypeError("previous and current must be MemoryRecord values")
    if previous.record_id != current.record_id:
        raise ProvenanceError("record identity must be preserved")
    if previous.provenance is not current.provenance:
        raise ProvenanceError("record provenance must be preserved")
