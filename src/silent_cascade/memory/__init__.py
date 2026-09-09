"""Bounded metadata-only runtime memory for public facts."""

from silent_cascade.memory.provenance import (
    require_provenance_preserved,
    require_support_ledger,
)
from silent_cascade.memory.store import (
    BoundedMemory,
    RuntimeMemoryRecord,
    append_perceived_fact,
    legal_records,
    mark_consumed,
    mark_recalled,
)

__all__ = [
    "BoundedMemory",
    "RuntimeMemoryRecord",
    "append_perceived_fact",
    "legal_records",
    "mark_consumed",
    "mark_recalled",
    "require_provenance_preserved",
    "require_support_ledger",
]
