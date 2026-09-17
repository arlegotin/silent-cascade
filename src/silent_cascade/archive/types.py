"""Strict, immutable schemas for bounded Phase 4 archive units."""

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from silent_cascade.validation import StrictModel

MAX_INTEGER = 2**63 - 1
Hash = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Positive = Annotated[int, Field(strict=True, gt=0, le=MAX_INTEGER)]
Count = Annotated[int, Field(strict=True, ge=0, le=MAX_INTEGER)]
UnitKind = Literal[
    "episode_pack",
    "evaluation_metadata",
    "journal",
    "control_snapshot",
    "diagnostic",
    "partial",
]


class ArchivePolicy(StrictModel):
    """Operational ceilings and coherent simultaneous-space reservations."""

    schema_version: Literal["phase4-r2-policy-v1"] = "phase4-r2-policy-v1"
    workspace_bytes: Annotated[int, Field(strict=True, gt=0, le=10 * 1024**3)] = 10 * 1024**3
    spool_bytes: Annotated[int, Field(strict=True, gt=0, le=2 * 1024**3)] = 2 * 1024**3
    cache_bytes: Annotated[int, Field(strict=True, gt=0, le=2 * 1024**3)] = 2 * 1024**3
    pinned_bytes: Annotated[int, Field(strict=True, gt=0, le=1024**3)] = 1024**3
    metadata_bytes: Annotated[int, Field(strict=True, gt=0, le=1024**3)] = 1024**3
    scratch_bytes: Annotated[int, Field(strict=True, gt=0, le=256 * 1024**2)] = 256 * 1024**2
    logs_bytes: Annotated[int, Field(strict=True, gt=0, le=256 * 1024**2)] = 256 * 1024**2
    emergency_bytes: Annotated[int, Field(strict=True, gt=0, le=1536 * 1024**2)] = 1536 * 1024**2
    reserve_bytes: Annotated[int, Field(strict=True, gt=0, le=2 * 1024**3)] = 2 * 1024**3
    episode_bytes: Annotated[int, Field(strict=True, gt=0, le=1024**3)] = 1024**3
    pack_target_bytes: Annotated[int, Field(strict=True, gt=0, le=128 * 1024**2)] = 128 * 1024**2
    pack_episodes: Annotated[int, Field(strict=True, gt=0, le=256)] = 256
    journal_bytes: Annotated[int, Field(strict=True, gt=0, le=128 * 1024**2)] = 128 * 1024**2
    journal_records: Annotated[int, Field(strict=True, gt=0, le=128)] = 128
    chunk_bytes: Annotated[int, Field(strict=True, gt=0, le=32 * 1024**2)] = 32 * 1024**2
    page_bytes: Annotated[int, Field(strict=True, gt=0, le=16 * 1024**2)] = 16 * 1024**2
    page_entries: Annotated[int, Field(strict=True, gt=0, le=1000)] = 1000
    remote_bytes: Annotated[int, Field(strict=True, gt=0, le=4_500_000_000_000)] = 4_500_000_000_000

    @model_validator(mode="after")
    def validate_reservations(self) -> "ArchivePolicy":
        if not self.chunk_bytes <= self.journal_bytes <= self.episode_bytes <= self.spool_bytes:
            raise ValueError("archive size limits are not coherently ordered")
        if self.pack_target_bytes > self.episode_bytes:
            raise ValueError("pack target exceeds the admitted episode size")
        allocated = sum(
            (
                self.spool_bytes,
                self.cache_bytes,
                self.pinned_bytes,
                self.metadata_bytes,
                self.scratch_bytes,
                self.logs_bytes,
                self.emergency_bytes,
                self.reserve_bytes,
            )
        )
        if allocated > self.workspace_bytes:
            raise ValueError("archive ledger categories exceed workspace bytes")
        if self.chunk_bytes * 2 > self.scratch_bytes:
            raise ValueError("scratch cannot hold publication and restore chunk reservations")
        if self.episode_bytes + self.pack_target_bytes + self.chunk_bytes > self.spool_bytes:
            raise ValueError("spool cannot hold the peak publication reservation")
        if self.episode_bytes + self.chunk_bytes > self.cache_bytes:
            raise ValueError("cache cannot hold the peak restore reservation")
        if self.page_bytes > self.metadata_bytes:
            raise ValueError("metadata cannot hold one decoded catalog page")
        return self


class UnitIdentity(StrictModel):
    run_id: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    source_commit: Revision
    config_sha256: Hash
    evidence_identity_sha256: Hash
    checkpoint_sha256: Hash | None
    writer_stopped: bool
    checkpoint_committed: bool


class ChunkSpan(StrictModel):
    chunk_index: Count
    offset: Count
    length: Positive

    @model_validator(mode="after")
    def validate_end(self) -> "ChunkSpan":
        if self.offset + self.length > MAX_INTEGER:
            raise ValueError("chunk span integer overflow")
        return self


class InventoryEntry(StrictModel):
    path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    sha256: Hash
    bytes: Count
    spans: tuple[ChunkSpan, ...]

    @model_validator(mode="after")
    def validate_spans(self) -> "InventoryEntry":
        if sum(span.length for span in self.spans) != self.bytes:
            raise ValueError("file spans do not equal file bytes")
        previous: tuple[int, int] | None = None
        for span in self.spans:
            position = (span.chunk_index, span.offset)
            if previous is not None and position <= previous:
                raise ValueError("file spans are not strictly ordered")
            previous = position
        return self


class InventoryShard(StrictModel):
    path: Annotated[str, StringConstraints(pattern=r"^inventory\.[0-9]{5,}\.jsonl$")]
    sha256: Hash
    entries: Annotated[int, Field(strict=True, gt=0, le=1000)]
    decoded_bytes: Annotated[int, Field(strict=True, gt=0, le=16 * 1024**2)]


class ChunkDescriptor(StrictModel):
    index: Count
    sha256: Hash
    bytes: Annotated[int, Field(strict=True, gt=0, le=32 * 1024**2)]


class BorrowedEntry(StrictModel):
    path: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    sha256: Hash
    bytes: Count


class EpisodeGroup(StrictModel):
    paths: tuple[Annotated[str, StringConstraints(min_length=1, max_length=4096)], ...] = Field(
        min_length=1
    )
    expanded_bytes: Annotated[int, Field(strict=True, gt=0, le=1024**3)]

    @model_validator(mode="after")
    def validate_order(self) -> "EpisodeGroup":
        if tuple(sorted(self.paths)) != self.paths or len(set(self.paths)) != len(self.paths):
            raise ValueError("episode group paths must be unique and sorted")
        return self


class UnitManifest(StrictModel):
    schema_version: Literal["phase4-r2-unit-v1"] = "phase4-r2-unit-v1"
    kind: UnitKind
    logical_root: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    identity: UnitIdentity
    policy_sha256: Hash
    expanded_bytes: Count
    file_count: Positive
    inventory_sha256: Hash
    inventory_shards: tuple[InventoryShard, ...] = Field(min_length=1)
    chunks: tuple[ChunkDescriptor, ...]
    episode_groups: tuple[EpisodeGroup, ...] = ()
    borrowed: tuple[BorrowedEntry, ...] = ()

    @model_validator(mode="after")
    def validate_totals(self) -> "UnitManifest":
        if sum(part.entries for part in self.inventory_shards) != self.file_count:
            raise ValueError("inventory shard count differs from file count")
        if sum(chunk.bytes for chunk in self.chunks) != self.expanded_bytes:
            raise ValueError("chunk bytes differ from expanded bytes")
        if tuple(chunk.index for chunk in self.chunks) != tuple(range(len(self.chunks))):
            raise ValueError("chunk indices are not contiguous")
        if self.kind == "episode_pack" and not self.episode_groups:
            raise ValueError("episode pack requires authenticated episode groups")
        if self.kind != "episode_pack" and self.episode_groups:
            raise ValueError("only episode packs may declare episode groups")
        borrowed_paths = tuple(entry.path for entry in self.borrowed)
        if tuple(sorted(borrowed_paths)) != borrowed_paths or len(set(borrowed_paths)) != len(
            borrowed_paths
        ):
            raise ValueError("borrowed paths must be unique and sorted")
        return self


class CatalogNodeRef(StrictModel):
    sha256: Hash
    path: Annotated[str, StringConstraints(pattern=r"^catalog/nodes/[0-9a-f]{64}\.json$")]
    entries: Positive
    decoded_bytes: Positive
    first_key: Hash
    last_key: Hash


class UnitCatalogEntry(StrictModel):
    record_type: Literal["unit"] = "unit"
    key: Hash
    unit_id: Hash
    kind: UnitKind
    logical_root: str
    expanded_bytes: Count
    file_count: Positive
    manifest_path: str


class OwnershipCatalogEntry(StrictModel):
    record_type: Literal["file", "directory"]
    key: Hash
    path: str
    unit_id: Hash | None

    @model_validator(mode="after")
    def validate_owner(self) -> "OwnershipCatalogEntry":
        if (self.record_type == "file") != (self.unit_id is not None):
            raise ValueError("only file ownership records bind a unit")
        return self


class RunCatalogEntry(StrictModel):
    record_type: Literal["run"] = "run"
    key: Hash
    run_id: str
    catalog_id: Hash
    entry_count: Count
    root_path: str


class RemoteReservationEntry(StrictModel):
    record_type: Literal["reservation"] = "reservation"
    key: Hash
    object_key: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    bytes: Positive
    sha256: Hash


class ReceiptLocatorEntry(StrictModel):
    record_type: Literal["receipt"] = "receipt"
    key: Hash
    unit_id: Hash
    receipt_sha256: Hash
    object_key: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    bytes: Positive


class EvictionLocatorEntry(StrictModel):
    record_type: Literal["eviction"] = "eviction"
    key: Hash
    unit_id: Hash
    receipt_sha256: Hash
    intent_sha256: Hash
    object_key: Annotated[str, StringConstraints(min_length=1, max_length=4096)]
    bytes: Positive
    completed: bool


type CatalogEntry = (
    UnitCatalogEntry
    | OwnershipCatalogEntry
    | RunCatalogEntry
    | RemoteReservationEntry
    | ReceiptLocatorEntry
    | EvictionLocatorEntry
)


class CatalogNode(StrictModel):
    schema_version: Literal["phase4-r2-catalog-node-v1"] = "phase4-r2-catalog-node-v1"
    index: Literal["units", "ownership", "runs", "reservations", "receipts", "evictions"]
    depth: Annotated[int, Field(strict=True, ge=0, le=255)]
    records: tuple[CatalogEntry, ...] = ()
    zero: CatalogNodeRef | None = None
    one: CatalogNodeRef | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "CatalogNode":
        leaf = bool(self.records)
        if leaf == bool(self.zero or self.one):
            raise ValueError("catalog node must be exactly one leaf or branch")
        if not leaf and (self.zero is None or self.one is None):
            raise ValueError("catalog branch requires both children")
        if leaf:
            keys = tuple(record.key for record in self.records)
            if tuple(sorted(keys)) != keys or len(set(keys)) != len(keys):
                raise ValueError("catalog leaf keys must be unique and sorted")
            expected = {
                "units": {"unit"},
                "ownership": {"file", "directory"},
                "runs": {"run"},
                "reservations": {"reservation"},
                "receipts": {"receipt"},
                "evictions": {"eviction"},
            }[self.index]
            if any(record.record_type not in expected for record in self.records):
                raise ValueError("catalog entry belongs to another index")
        return self


class RunCatalogRoot(StrictModel):
    schema_version: Literal["phase4-r2-run-catalog-v1"] = "phase4-r2-run-catalog-v1"
    run_id: str
    units: CatalogNodeRef | None
    ownership: CatalogNodeRef | None
    unit_count: Count
    ownership_count: Count

    @model_validator(mode="after")
    def validate_counts(self) -> "RunCatalogRoot":
        if self.unit_count != (0 if self.units is None else self.units.entries):
            raise ValueError("run catalog unit count differs")
        if self.ownership_count != (0 if self.ownership is None else self.ownership.entries):
            raise ValueError("run catalog ownership count differs")
        return self


class CorpusCatalogRoot(StrictModel):
    schema_version: Literal["phase4-r2-corpus-catalog-v1"] = "phase4-r2-corpus-catalog-v1"
    runs: CatalogNodeRef | None
    run_count: Count

    @model_validator(mode="after")
    def validate_count(self) -> "CorpusCatalogRoot":
        if self.run_count != (0 if self.runs is None else self.runs.entries):
            raise ValueError("corpus catalog run count differs")
        return self


class OperationalCatalogRoot(StrictModel):
    schema_version: Literal["phase4-r2-operational-catalog-v1"] = "phase4-r2-operational-catalog-v1"
    reservations: CatalogNodeRef | None
    receipts: CatalogNodeRef | None
    evictions: CatalogNodeRef | None
    reservation_count: Count
    receipt_count: Count
    eviction_count: Count
    publication_bytes: Count

    @model_validator(mode="after")
    def validate_counts(self) -> "OperationalCatalogRoot":
        expected = (
            (self.reservation_count, self.reservations),
            (self.receipt_count, self.receipts),
            (self.eviction_count, self.evictions),
        )
        if any(count != (0 if ref is None else ref.entries) for count, ref in expected):
            raise ValueError("operational catalog count differs")
        return self


@dataclass(frozen=True)
class FileEntry:
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class UnitRef:
    unit_id: str
    kind: str
    logical_root: str
    expanded_bytes: int
    file_count: int
    manifest_path: str


@dataclass(frozen=True)
class CatalogRef:
    catalog_id: str
    scope: str
    run_id: str | None
    entry_count: int
    root_path: str
