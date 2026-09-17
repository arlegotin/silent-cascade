# Phase 4 Bounded R2 Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing seed-11 Phase 4 pilot executable with bounded local
disk and verified R2 retention, without changing its scientific protocol.

**Architecture:** Preserve scientific bytes and archive committed episode evidence
during evaluation. A finite local supervisor owns R2 transfers; offline workers
use bounded metadata and single-episode leases. Streaming finalizers/readers and
paged journal/catalog/index storage retain every semantic and provenance check
inside a total 10 GiB local disk envelope, including protected headroom.

**Tech Stack:** Existing locked Python 3.12/Pydantic/pytest/Hypothesis, standard
library streaming I/O and subprocesses, existing AWS CLI v2 administration profile.
No new ML/cloud SDK dependency, filesystem mount, database or hosted automation.

**Spec:** `docs/superpowers/specs/2026-09-17-phase-4-r2-archive-design.md`, under
canonical `docs/superpowers/specs/2026-08-30-silent-cascade-design.md` v1.0.2.

**Status:** Approved for execution by the user's 2026-09-17 instruction:
“stop asking, you know phase 4 goals – proceed”. Execute autonomously with
subagents on the current branch. The production Task 12 pilot remains gated on
storage qualification and the existing scientific preflight.

**Revision:** 2. The user rejected the 64 GiB proposal and subsequently authorized
up to 10 GiB local disk. This revision replaces whole-evaluation leases with
episode streaming; the subsequent proceed instruction approves this plan.

**Inspected baseline:** `main` at `2995310` on 2026-09-17. Stay on the current
branch, commit reviewed increments locally and do not push.

## Global Constraints

- The Python package is named `silent_cascade`; the CLI executable is `silent-cascade`.
- Primary scientific execution must remain offline and record exactly zero foundation-model calls.
- No database, service, daemon, plugin backend registry, hosted execution or new ML dependency is introduced.
- Use Superpowers TDD, systematic debugging on failures, task-scoped implementation/review agents, and verification before completion.
- CI/CD is disabled. `make verify` remains the complete local quality gate.
- Keep all raw per-episode metrics, every failure/dynamics-error trace, and the existing selected-success retention policy. Never discard evidence to satisfy a budget.
- Preserve seed `11`, all four 10,000-entry validation manifests, batch `128`, validation every `1,000` global updates, total ceiling `75,000`, patience `15`, best `3` plus latest `1` checkpoints, objective and selection rules.
- Preserve all Phase 4 gates: at least 90% pilot IID timed success, at most 10% negative false-action rate, zero dynamics failures across 10,000 validation episodes, legal event sequences, delay-following, exact CPU continuation/replay and declared CPU/MPS checks.
- Do not change neural models, losses, engine scheduling, simulated times, generator semantics, counters or source-bound historical evidence. An additive engine crash-publication descriptor is allowed for storage ownership only.
- Total new local allocation plus protected headroom is at most 10 GiB; no uncounted temporary directory, second volume, test output or recovery copy.
- No frozen tests, new seeds, baseline training, Qwen, Phase 5 work or positive research claims.
- Credentials, account endpoint and machine-specific paths stay outside Git, scientific configuration, logs and archives.

## 1. Explicit parent-plan amendment

Approval of this plan authorizes these storage-only changes to the Phase 4 plan:

1. **Task 10 / Task 12 preflight:** retain the old all-local forecast and failed
   result. For explicitly archive-backed execution, require a measured bounded
   working-set certificate, local reserve, immutable remote inventory, successful
   restore/failure tests and remote byte budget instead of demanding that all
   retained bytes fit on one local disk. Keep the full-retention upper estimate
   and `1.2` safety factor visible; do not change scientific retention.
2. **Task 11 recovery:** the new opt-in archive-backed path must authenticate
   every indexed training input before new neural work, but may materialize those
   inputs sequentially. Preserve original bytes remotely and bind their catalog
   root in recovery intent. The existing eager local recovery remains unchanged.
3. **Source closure:** include this approved plan, its design, archive package and
   changed entrypoints in source authentication and independent verification.
   Historical source tuples remain historical; regenerate new-source evidence.
4. **Publication/readers:** add storage-only episode commits and streaming
   finalization. Original scientific rows/index/DONE formats, full denominators
   and every semantic check remain unchanged; no whole-evaluation cold hydration.

This plan is not approval to bypass the canonical protocol or copy old passing
receipts to a changed executable. The production pilot still begins only after
the existing Task 12 preflight and the new storage gate both pass.

## 2. Design decisions and limits

- Archive committed episodes during `attempt-*/validation-*` and
  `final/eval/<suite>` production. Keep canonical row bytes and the full 10,000-row
  scientific denominator. Close evaluation metadata only after streaming
  finalization; later validation/execution records form additional closure units.
- Episode transport commitment, evaluation DONE and checkpoint commitment are
  distinct. A receipt never promotes an incomplete evaluation or journal prefix.
- Journal units contain exact immutable `journal-<sha>.json` files, segmented at
  record boundaries by actual bytes: at most 128 MiB and 128 records. Seal/upload
  throughout training, including before durable checkpoint commitment. Bind the
  committed prefix later via the authenticated checkpoint head; preserve tails.
- Retain rolling checkpoint/control inputs locally; back them up in immutable
  checkpoint-bound snapshots. A snapshot is a recovery version, not a duplicate
  owner of a logical file in the active catalog.
- Use the design's complete 10 GiB ledger: spool 2 GiB, read cache 2 GiB, pinned
  checkpoint/control data 1 GiB, current metadata/paged indexes 1 GiB, scratch
  256 MiB, logs 256 MiB, emergency 1.5 GiB and protected free headroom 2 GiB.
  Normal allocation <=6.5 GiB; emergency allocation <=8 GiB. No extra 16 GiB
  reserve, preallocation, TMPDIR bypass or spill to another volume.
- One episode writer/reader each and one network operation; episode owned-output
  admission bound 1 GiB; chunks <=32 MiB; target packs <=128 MiB or 256 episodes,
  permitting one larger singleton within the episode bound. Pages <=1,000 entries
  and 16 MiB decoded. Prove complete artifact/atomic/crash bounds before admission.
- Archive closed metadata, catalogs, receipts, logs and output figures too. Pin
  only bounded heads/pages and active controls. Verification/test scratch and
  stopped attempts share the same new-output ledger. Preserve existing history.
- Cap new remote bytes for the run prefix at 4,500,000,000,000. This is not a
  provider billing cap, and it does not stop charges for already retained data.
- A resource cap causes `storage_blocked`, never truncation, automatic pruning,
  scientific early stopping, omitted denominators or an automatic larger budget.
- Do not add per-episode resume. Preserve current interrupted-validation behavior:
  retain partial evidence and resume the same run from its last durable checkpoint.

## 3. File map and task dependencies

All paths below are repository-relative. New modules are focused local archive
utilities, not a general storage framework. Tasks are sequential; independent
read-only reviews may run alongside tests. Only one source/test writer and one
numerical execution owner at a time.

| Task | New files | Main existing integration points |
| --- | --- | --- |
| 1 | `src/silent_cascade/archive/{__init__,types,catalog,bundles}.py`; `tests/archive/{conftest,test_catalog,test_bundles}.py` | existing safe I/O/hash/compact-index utilities, reused without weakening |
| 2 | `archive/{transport,operational}.py`; `tests/archive/{test_transport,test_operational}.py` | Task1 typed catalog nodes; existing local AWS CLI profile only |
| 3 | `archive/{session,supervisor}.py`; `tests/archive/{test_session,test_supervisor}.py` | `train/pilot_offline.py` denial hooks |
| 4 | `archive/producer.py`; `tests/pilot/test_pilot_archive_producer.py` | row publication, crash ownership descriptor, streaming finalizer, trainer/journals, provenance |
| 5 | `archive/readers.py`; `tests/pilot/test_pilot_archive_readers.py` | episode scanner, report/evidence/index/journal/recovery readers |
| 6 | `archive/{cli,preflight}.py`; `tests/archive/{test_cli,test_preflight}.py` | root CLI, Makefile, local verification inventory, setup/status docs |
| 7 | storage-gate evidence and documentation | local full verification and existing Task 12 handoff only |

Paths prefixed `archive/` in this table mean `src/silent_cascade/archive/`.
`tests/archive/conftest.py` supplies transport-only synthetic fixtures; they are
never eligible scientific evidence. Reuse the actual tiny pilot fixtures in
`tests/pilot/test_pilot_source.py` and `test_pilot_checks.py` for scientific tests.

## Task 1: Immutable catalog and bounded unit codec

**Files:** Create the Task 1 files above. Read `eventflow/archive_io.py`,
`train/pilot_artifact_index.py`, `train/pilot_evidence_types.py` and `io.py`.

**Interfaces produced:**

```python
class ArchivePolicy(StrictModel):
    schema_version: Literal["phase4-r2-policy-v1"] = "phase4-r2-policy-v1"
    workspace_bytes: int = 10 * 1024**3  # includes protected headroom
    spool_bytes: int = 2 * 1024**3
    cache_bytes: int = 2 * 1024**3
    pinned_bytes: int = 1024**3
    metadata_bytes: int = 1024**3
    scratch_bytes: int = 256 * 1024**2
    logs_bytes: int = 256 * 1024**2
    emergency_bytes: int = 1536 * 1024**2
    reserve_bytes: int = 2 * 1024**3
    episode_bytes: int = 1024**3
    pack_target_bytes: int = 128 * 1024**2
    pack_episodes: int = 256
    journal_bytes: int = 128 * 1024**2
    journal_records: int = 128
    chunk_bytes: int = 32 * 1024**2
    page_bytes: int = 16 * 1024**2
    page_entries: int = 1000
    remote_bytes: int = 4_500_000_000_000

@dataclass(frozen=True)
class FileEntry:
    path: str
    sha256: str
    bytes: int

@dataclass(frozen=True)
class UnitRef:
    unit_id: str                  # SHA-256 of canonical unit manifest
    kind: str                     # closed literals from the design
    logical_root: str
    expanded_bytes: int
    file_count: int
    manifest_path: str             # relative to control directory

def seal_unit(*, run_dir: Path, control_dir: Path, logical_root: str,
              paths: tuple[str, ...], kind: str, identity: dict,
              policy: ArchivePolicy,
              episode_groups: tuple[tuple[str, ...], ...] = (),
              borrowed: tuple[FileEntry, ...] = ()) -> UnitRef: ...
def iter_unit_files(control_dir: Path, ref: UnitRef) -> Iterator[FileEntry]: ...
def iter_unit_chunks(*, run_dir: Path, control_dir: Path, ref: UnitRef,
                     scratch_dir: Path, policy: ArchivePolicy) -> Iterator[Path]: ...
def restore_unit(*, control_dir: Path, ref: UnitRef,
                 chunks: Iterable[Path], destination: Path,
                 policy: ArchivePolicy,
                 selected_paths: tuple[str, ...] | None = None) -> Path: ...
```

`kind` accepts only `episode_pack`, `evaluation_metadata`, `journal`,
`control_snapshot`, `diagnostic`, `partial`. Identity binds run ID, source/config hashes and the relevant existing
evaluation/checkpoint identity; no bare user-provided label establishes a seal.
Define a strict `UnitIdentity` with `run_id`, forty-hex `source_commit`, sixty-four-
hex `config_sha256` and `evidence_identity_sha256`, optional `checkpoint_sha256`,
and exact booleans `writer_stopped` and `checkpoint_committed`. Reject extra keys.
The producer validates those values against its actual source/lock/evidence;
synthetic codec fixtures never establish a scientific source or acceptance gate.
Reject booleans/nonpositive values for byte/count limits, require
`chunk_bytes <= journal_bytes <= episode_bytes <= spool_bytes`; the eight ledger
categories must sum to at most `workspace_bytes`. Reject overrides above Section 2
ceilings. Small test policies must scale all categories coherently, not just the
top-level number. Validate peak simultaneous publication/restore reservations.
Use strict manifest/span/shard models in `types.py`. Separate original logical
paths from snapshot-version paths; reject duplicate active ownership, not merely
shared parent directories among disjoint journal segments.
`manifest_path` is a logical control path; Task 3 leases its bounded manifest/page
before calling codecs. `selected_paths` must be an exact authenticated episode
owned-file set or an explicit bounded diagnostic selection, never caller-supplied
unchecked spans. Full-unit restore remains subject to the same admission ledger.
For `episode_pack`, `episode_groups` is required to form an exact disjoint partition
of owned paths, with nonempty sorted groups and the declared count/episode bounds.
Authenticate it in the manifest. Do not infer episode ownership from directories:
real ordinal payloads are flat files and crash files can live elsewhere. Task 4
supplies the groups from durable episode commits. Other kinds use no episode groups.
`borrowed` authenticates unique run-relative size/hash references disjoint from
owned members; verify them while sealing, but give them no owned spans or expanded
bytes. Task 3 pins/resolves exact path/hash dependencies before exposing a lease.
`control_snapshot` ownership is versioned by `(unit_id, original path)`, outside
the active logical-file collision map; multiple snapshots may preserve different
versions of a mutable control. Do not rename scientific paths. Restore by explicit
unit reference into a fresh destination; active source controls are never evicted.
Allow canonical `logical_root='.'` only to identify the run root; member paths
still reject empty/dot/traversal components. This supports existing root journals.

Add immutable `CatalogRef` and `publish_run_catalog`/`iter_run_catalog`, plus
corresponding corpus publish/iterate APIs. Run publication accepts a previous
generation, a stream of new `UnitRef` values, and bounded cold-object and unit-
inventory readers. Use persistent content-addressed binary Merkle tries with
bounded leaves and branch nodes, not an unbounded root list of page descriptors.
Run roots bind unit references and active logical-file ownership; corpus roots
bind run references. Updating a generation must read only changed trie paths,
not download all historic inventories. Keep archived ownership after evicting
local manifests; reject exact and ancestor/descendant file-path collisions.
Control snapshots remain outside active ownership. Batch/coalesce changed nodes
before durable publication so intermediate leaf rewrites do not accumulate on
disk. Authenticate node hashes, counts, ordering and full traversal closure.
An explicit catalog root hash is sufficient to locate cold recovery metadata;
no bucket listing is needed. Old metadata remains subject to verified-remote-
receipt eviction, even when a newer root no longer references it.

- [ ] **Step 1 — RED: exact bytes and unsafe inventories.** Add this synthetic
  test, then traversal/symlink/hardlink/special-file, duplicate logical member,
  malformed span, missing final shard, nonfinite and overflow cases:

  ```python
  def test_unit_inventory_binds_exact_original_bytes(tmp_path):
      from silent_cascade.hashing import sha256_bytes
      from silent_cascade.io import atomic_create_bytes
      from silent_cascade.archive.types import ArchivePolicy
      from silent_cascade.archive.catalog import seal_unit, iter_unit_files
      root = tmp_path / "run"
      atomic_create_bytes(root / "unit/a.json", b'{"value":1}\n')
      ref = seal_unit(run_dir=root, control_dir=tmp_path / "control",
                      logical_root="unit", paths=("unit/a.json",),
                      kind="partial", identity={
                          "run_id": "debug-fixture",
                          "source_commit": "0" * 40,
                          "config_sha256": "1" * 64,
                          "evidence_identity_sha256": "2" * 64,
                          "checkpoint_sha256": None,
                          "writer_stopped": True,
                          "checkpoint_committed": False,
                      },
                      policy=ArchivePolicy())
      entries = tuple(iter_unit_files(tmp_path / "control", ref))
      assert [(e.path, e.sha256, e.bytes) for e in entries] == [
          ("unit/a.json", sha256_bytes(b'{"value":1}\n'), 12)]
  ```

  This fixture tests the transport schema and exact-byte inventory only. The
  production handoff must additionally authenticate the source and actual writer
  ownership; a self-declared dictionary never supplies that authorization.
- [ ] **Step 2 — Run RED:** `uv run pytest -q tests/archive/test_catalog.py tests/archive/test_bundles.py`.
  Confirm failure is the missing implementation, not a broken fixture.
- [ ] **Step 3 — Implement canonical inventory and streaming pack/restore.**
  Use bounded JSONL shards, exact file hashes and ordered `(chunk_sha, offset,
  length)` spans. Concatenate original bytes into at most 32 MiB chunks; no tar
  extraction, pickle, recompression or complete second bundle on disk. Two passes
  over sealed files are allowed: compute manifest, then deterministically rebuild
  one chunk at a time. Recheck file identity/hash to reject mutation between passes.
  Validate expanded totals before allocation and during every bounded write.
  Stage restore under an owned absent directory, fsync, then publish atomically.
  Pack episode commits up to the target byte/count limit; allow one admitted
  larger singleton, never split scientific evidence ownership between episodes.
  Use paged run/corpus roots and cold-readable receipt/index pages, not a giant
  local catalog. Bound decoded pages/counts and authenticate the final stream tail.
  Support restoring only selected episode file spans from a pack: no need to
  hydrate unrelated pack members, but authenticate every downloaded chunk in full.
- [ ] **Step 4 — GREEN and boundary tests:** exact round trip at small injected
  chunk sizes, split-file spans, empty regular files, byte/count limits, symlink
  parent swaps, changed source after sealing, crash before restore publication.
  Test complete inventories larger than one shard without large real allocations.
  Test missing last page, shared borrowed weights, sparse episode restore from a
  multi-episode pack, and millions of synthetic descriptors with bounded disk
  cache/cursor state. No unbounded per-receipt local files after archival.
- [ ] **Step 5 — Review and commit:** `feat: add bounded immutable artifact units`.

## Task 2: R2 transfer, readback receipts and safe eviction

**Files:** `archive/{transport,operational}.py`,
`tests/archive/{test_transport,test_operational}.py`. Narrowly extend
`archive/{types,catalog}.py` and adjacent tests for the closed operational index.
Consumes Task 1 units and safe no-follow file operations.

**Interfaces produced:**

```python
class ObjectTransport(Protocol):
    transport_id: str  # stable opaque SHA256 of the operational destination identity
    def create(self, key: str, source: Path) -> None: ...
    def download(self, key: str, destination: Path, *, max_bytes: int) -> None: ...

class R2CliTransport:
    def __init__(self, *, profile: str, bucket: str, prefix: str): ...

def initialize_remote_reservations(*, control_dir: Path, transport_id: str,
                                  accounted_bytes: int,
                                  accounting_evidence_sha256: str,
                                  policy: ArchivePolicy) -> None: ...

def archive_unit(*, run_dir: Path, control_dir: Path, ref: UnitRef,
                 transport: ObjectTransport, policy: ArchivePolicy) -> str: ...
def evict_unit(*, run_dir: Path, control_dir: Path, ref: UnitRef,
               receipt_sha256: str) -> None: ...
```

`archive_unit` returns the hash of a durable receipt only after full readback,
manifest and catalog-commit publication. The test transport is a directory-backed
implementation of the two methods, with counters/fault injection; production has
only the R2 CLI implementation. `evict_unit` requires the archive/writer locks,
zero active leases, exact current file hashes and a committed catalog generation.
Reject eviction of `control_snapshot` source files; snapshots version mutable
controls for recovery, while the active local controls remain pinned.

Remote reservation bootstrap is explicit and create-only. Bind the transport
identity, strict accounted byte count within policy, and actual accounting-
evidence hash; a supplied label/hash alone does not establish production
qualification. Missing/corrupt history never implies zero usage. Refuse automatic
reinitialization when prior remote-archive/catalog-commit/receipt/eviction history
exists; pending local units alone are not proof of remote writes. Fixtures may
attest a newly empty directory transport. Task7 supplies measured production
initialization evidence and carries qualification usage into the production
budget. A changed prefix or lost local ledger does not reset usage; catalog-only
recovery is read-only until reservation history is authenticated.

Use a bounded authenticated operational index for unique-object reservations
and receipt/eviction-intent locators, not one permanently resident file per
historical object or unit. Extend the catalog's closed typed record union; do not
introduce a generic registry, plugin system or database. Only roots and bounded
pending state remain resident; historical pages become cold only after verified
readback. Preserve Task1 catalog behavior and its adjacent regressions.

Avoid recursive self-accounting: durably reserve a finite, source-derived bound
for the complete operational-page/root/commit publication batch before any of
its network writes. Include the batch's own metadata overhead, retain its charge
on ambiguous outcomes, and settle only provably unused bytes after fully verified
publication. Do not recursively upload reservations to reserve themselves.
Missing/corrupt pending history fails closed. Test publication cut points and
large lazy histories; report the exact transaction and recovery interfaces.

- [ ] **Step 1 — RED:** add `test_readback_corruption_never_authorizes_eviction`,
  `test_existing_different_object_is_not_overwritten`, and recovery cut-point tests.
  Example corruption check using the directory-backed transport fixture:

  ```python
  def test_readback_corruption_never_authorizes_eviction(sealed_unit, transport):
      transport.corrupt_downloads = True
      before = sealed_unit.original_bytes()
      with pytest.raises(ValueError, match="readback"):
          archive_unit(run_dir=sealed_unit.root,
                       control_dir=sealed_unit.control, ref=sealed_unit.ref,
                       transport=transport, policy=sealed_unit.policy)
      assert sealed_unit.original_bytes() == before
      assert not tuple(sealed_unit.control.glob("receipts/*.json"))
  ```

  Define `sealed_unit` in Task 1's conftest from real codec output: fields
  `root`, `control`, `ref`, `policy`; `original_bytes()` returns only its exact
  inventoried files. The transport fixture writes real bytes and can corrupt
  its download copy, not the original source.
- [ ] **Step 2 — Run RED:** `uv run pytest -q tests/archive/test_transport.py`.
- [ ] **Step 3 — Implement closed AWS CLI subprocess calls.** Argument arrays,
  no shell; fixed profile/project bucket; validated prefix and content-addressed
  keys; `s3api put-object --if-none-match '*'`; fresh download files for `get-object`.
  Downloads are bounded during transfer, not only checked afterward: request a
  byte range no larger than the reserved cap and validate returned object-total
  size/range metadata plus the expected exact size/SHA. An oversized/truncated
  object fails; never issue an unbounded GET after trusting a HEAD size alone.
  Qualify the actual AWS/R2 range behavior in Task 7 before relying on it.
  Explicit finite timeouts and at most three transient retries, preserving
  ambiguous-write recovery through exact GET verification. Reject auth/collision/
  integrity failures without retrying forever. Never use recursive sync/delete,
  `--debug`, credentials in argv, ETag-as-SHA or public URLs.
  Persist unique-object byte reservations before writes; retain reservations for
  ambiguous/orphaned writes across retries/restarts, including manifest/catalog
  bytes. Do not infer a zero balance from missing local success receipts.
- [ ] **Step 4 — Implement receipt-before-eviction ordering.** Verify every
  chunk's downloaded byte count and SHA, then verify manifest and catalog commit.
  Fsync receipt and eviction intent before exact scoped unlinks; verify unchanged
  file/inode ownership immediately before deletion. Unknown files stop eviction.
  Interrupted eviction resumes only listed members; missing members are legal
  only under the same recorded eviction intent. No automatic remote deletes.
- [ ] **Step 5 — GREEN:** cover failure at every lifecycle transition, checksum
  mismatch despite successful HTTP, same-content create conflicts, stale receipt,
  credential expiry, oversized remote responses, quota refusal, concurrent lease, exact cleanup/recovery and
  redaction of stderr. Assert no AWS invocation in fixture tests.
- [ ] **Step 6 — Review and commit:** `feat: archive verified units to private R2`.

## Task 3: Local evidence leases and finite supervisor

**Files:** `archive/session.py`, `archive/supervisor.py`, corresponding tests;
extend `train/pilot_offline.py` only to share/test the existing denial boundary.

**Interfaces produced:**

```python
@dataclass(frozen=True)
class EvidenceLease:
    ref: UnitRef
    local_root: Path

class EvidenceContext(Protocol):
    def entries(self) -> Iterator[FileEntry]: ...
    def evaluation_roots(self) -> tuple[str, ...]: ...
    def lease(self, ref: UnitRef) -> ContextManager[EvidenceLease]: ...
    def metadata(self, logical_root: str) -> ContextManager[EvidenceLease]: ...
    def episode(self, logical_root: str, ordinal: int, *,
                commit_sha256: str) -> ContextManager[EvidenceLease]: ...
    def read_record(self, logical_path: str, *, expected_sha256: str,
                    max_bytes: int) -> bytes: ...
    def verify_inventory(self, entries: Iterable[FileEntry]) -> None: ...

class LocalArchiveSession:
    # Implements EvidenceContext using only files and local requests.
    def __init__(self, *, run_dir: Path, control_dir: Path,
                 policy: ArchivePolicy): ...

def supervise_job(*, job: str, request: dict, run_dir: Path,
                  control_dir: Path, transport: ObjectTransport,
                  policy: ArchivePolicy) -> int: ...
```

`entries()` merges resident original files with active catalog ownership; never
includes transport/control files in the scientific inventory. Match archive
entries against authoritative journal/index/DONE hashes, not only each other.

- [ ] **Step 1 — RED:** real local-file request/response tests, including a stale
  response from another run, response-before-complete-restore, simultaneous episode
  lease requests, writer/reader conflict and parent death. A child fixture attempts
  DNS/socket/cloud-client/AWS-subprocess access and must be denied; the parent
  transport fixture must still service explicit archive requests.
- [ ] **Step 2 — Run RED:** `uv run pytest -q tests/archive/test_session.py tests/archive/test_supervisor.py tests/pilot/test_pilot_offline.py`.
- [ ] **Step 3 — Implement strict request protocol and locks.** Atomic files
  contain run identity, sequence, operation, unit/policy hash and bounded payload.
  Allow only `seal`, `archive`, `lease`, `release`, `status`, `stop`. Hold permanent
  lock inodes; use PID plus process creation time for interrupted-owner detection.
  A lease is never published until all restored files authenticate. No silent
  trust of an acknowledgment: offline code rechecks manifest/local-file identity.
- [ ] **Step 4 — Implement bounded scheduling.** Enforce the shared ledger,
  one episode lease, bounded metadata/journal pages and pinned controls. Reserve
  full possible in-flight outputs before work; measure physical allocation and
  headroom. Include test scratch, logs, stopped attempts, staged atomic copies
  and outstanding reservations. Release never evicts a live reader. Use 30-second
  health-check waits with status updates. Each AWS operation has a 30-minute
  timeout and at most three transient attempts; a multi-chunk unit renews progress
  only after each verified chunk and may legitimately take longer. Parent death
  or exhaustion yields actionable `storage_blocked`; preserve pending files.
  Do not mistake total unit-transfer time for a stalled request. The exact pilot
  process retains/uses its existing durable checkpoint on abort.
- [ ] **Step 5 — GREEN:** real subprocess IPC; no `aws`/cloud imports in the
  child; no credential environment forwarding; no new simulated events while
  waiting; all children joined; partial staging quarantined on crash; no global
  background daemon or hidden arbitrary-command dispatcher. Small-budget tests
  must stop safely at category/global limits without allocating GiB; prove no
  whole evaluation materializes and no TMPDIR/second-volume bypass exists.
- [ ] **Step 6 — Review and commit:** `feat: isolate archive I/O from offline evidence leases`.

## Task 4: Durable episode handoff, streaming finalization and rolling journals

**Files:** Create `archive/producer.py` and
`tests/pilot/test_pilot_archive_producer.py`. Modify
`eval/{artifacts,runner}.py`, `eventflow/engine.py` (publication descriptor only),
`logging/crash_bundle.py` (descriptor types and scoped publication bound),
`logging/neural_trace.py` (existing decoded/compressed format bounds),
`train/{pilot_data,pilot_trainer,pilot_workflow,pilot_provenance,pilot_checks}.py`,
the exact source inventory in `train/pilot_evidence.py`, and
`tests/pilot/test_pilot_source.py`. Extend
`tests/pilot/{test_neural_crashes,test_pilot_artifacts,test_timed_runner}.py`.
Do not change model/loss/generator code, engine scheduling or checkpoint wire formats.

**Interfaces produced:**

```python
@dataclass(frozen=True)
class PublishedCrashFile:
    path: Path
    sha256: str

@dataclass(frozen=True)
class PublishedCrash:
    manifest: PublishedCrashFile
    checkpoint: PublishedCrashFile | None
    shared_weights: PublishedCrashFile | None

@dataclass(frozen=True)
class EpisodeCommit:
    schema_version: Literal["phase4-evaluation-episode-commit-v1"]
    identity_sha256: str
    ordinal: int
    episode_public_id: str
    episode_sha256: str
    row_offset: int
    row_bytes: int
    row_sha256: str
    owned: tuple[FileEntry, ...]
    borrowed: tuple[FileEntry, ...]

class ArchiveProducer:
    def before_update(self, global_step: int) -> None: ...
    def before_evaluation(self, logical_root: str) -> None: ...
    def before_episode(self, logical_root: str, ordinal: int) -> None: ...
    def after_episode(self, logical_root: str, commit: EpisodeCommit) -> None: ...
    def after_evaluation(self, logical_root: str) -> None: ...
    def after_validation(self, logical_root: str) -> None: ...
    def after_journal(self, logical_path: str, journal_sha256: str) -> None: ...
    def after_checkpoint(self, descriptor: PilotCheckpointDescriptor,
                         progress: PilotProgress) -> None: ...

def seal_journal_segments(*, run_dir: Path, control_dir: Path,
                          base_head: str | None, sealed_head: str,
                          policy: ArchivePolicy) -> tuple[UnitRef, ...]: ...
```

Define `EpisodeCommit` in `archive/types.py`; validate exact scalar types, sorted
ownership, row binding and borrowed references. Define `PublishedCrashFile` and
`PublishedCrash` beside existing crash publication types; convert their safe
relative paths, verified hashes and measured sizes to `FileEntry` in the producer.
The engine must not import archive code. Never add hidden truth, storage callbacks
or secrets to agent state.
Use the existing `PilotCheckpointDescriptor` and `PilotProgress` from
`train/pilot_state.py`; do not introduce alternate checkpoint schemas.

Thread optional `archive_producer=None` and `evidence_context=None` through
`run_pilot_training`, `_workflow`, `run_pilot`, `_validation`,
`evaluate_episodes`, `write_evaluation`, `_run_pilot_checks_owned` and `_evaluate`.
Only archive mode emits operational commits. Both paths keep original logical
payloads; new controls live outside the scientific artifact inventory.

- [ ] **Step 1 — RED: episode publication cut points.** Extend real evaluation
  fixtures with failure injection after artifact writes, after row fsync/before
  envelope, after envelope/before receipt, and during eviction. Assert row fsync
  precedes envelope publication. No envelope/verified receipt means no eviction;
  pending bytes remain retained and no DONE appears prematurely.
- [ ] **Step 2 — Run RED:** `uv run pytest -q tests/pilot/test_pilot_archive_producer.py`.
  Add owned-file and crash cases before implementing: success, retained failure,
  runtime crash with checkpoint, initialization crash without checkpoint, and two
  crashes borrowing the same weights. First-episode eviction must not remove
  shared weights. Confirm failures come from missing behavior, not invalid fixtures.
- [ ] **Step 3 — Implement publication ownership.** Have engine crash publication
  return/expose its actual immutable descriptor to the runner, without changing
  runtime decisions or trace fields. In `write_evaluation`, verify episode evidence,
  append the exact canonical row, flush/fsync and atomically seal its commit.
  Reserve output before neural work; `after_episode` may hand off only after the
  complete seal. Keep current row log, identity, retention and shared weights local.
  Never discover ownership by globbing crash directories.
  Enforce the existing format-reader limits before each corresponding publication:
  evaluation sidecars/controls and compressed/decoded trajectories <=128 MiB,
  neural crash JSON <=128 MiB, pilot/control files including individual journals
  <=64 MiB. Neural archives already enforce <=128 MiB. Bound telemetry and commit
  metadata explicitly from their fixed schemas. Use source-derived possible
  artifact counts, not observed sizes alone, for admission. Keep unrelated legacy
  crash publication behavior unchanged. Fail before oversized disk writes; no
  truncation or discard of already-written evidence. Add boundary tests proving
  refusal occurs before opening a publication destination.
- [ ] **Step 4 — RED/GREEN: streaming full finalization.** Evict every episode-owned
  file from a real multi-row fixture, then finalize through one-episode leases.
  Revalidate evidence, crash manifest/checkpoint/weights links and row bindings.
  Build crash index and DONE inventory from authenticated commits plus controls,
  rejecting duplicate/missing/unclassified files. Compare original rows/index/
  metrics/crash-index bytes and scientific artifact hashes with the local path.
  Only after complete coverage may DONE publish. Preserve full 10,000-row
  production cardinality; fixture size does not authorize smaller validation.
- [ ] **Step 5 — RED/GREEN: training and final-check integration.** Extend the real
  four-update debug pilot with recording callbacks and tiny policy limits.
  Handoff episode evidence during each validation, close metadata after DONE, and
  seal later validation/execution records separately. Preserve later journal
  artifact hashes even when payloads are cold. Drain pending packs between paired
  robustness/primary validations, without requiring the whole first run locally.
- [ ] **Step 6 — RED/GREEN: rolling journals and controls.** Call `after_journal`
  after each immutable journal publication, sealing bounded byte/count segments
  throughout an interval, not only after the next checkpoint. Never split records
  or rewrite bytes. Segment transport binds predecessor/end heads; subsequent
  checkpoint snapshots establish which prefix is committed. Archive closed
  catalog/receipt pages, logs and metadata with exact inventories. Bound the active
  tail/current controls. Test more than one segment before any durable checkpoint,
  complete predecessor coverage, and restart before/after checkpoint index commit.
- [ ] **Step 7 — Preserve interrupted semantics.** Pressure retains failure/partial
  evidence and stops before more work; it is not scientific early stopping.
  Authenticate stopped writers before sealing orphan files, including crashes
  without durable rows. Resume enumerates cold and local journals/partials through
  the context, preserving last-checkpoint recovery and unknown outcomes. No
  same-directory per-episode validation resume or silent change of seed.
- [ ] **Step 8 — GREEN/review/commit:** run
  `uv run pytest -q tests/pilot/test_pilot_archive_producer.py tests/pilot/test_pilot_trainer.py tests/pilot/test_pilot_workflow.py tests/pilot/test_pilot_source.py`
  plus `uv run pytest -q tests/pilot/test_neural_crashes.py tests/pilot/test_pilot_artifacts.py tests/pilot/test_timed_runner.py`. Prove unchanged
  optimizer steps, RNG, CPU outputs, schedules, counters and retention; include
  corrupted borrowed weights and insufficient admission. Review source closure
  additions including this plan/design. Commit:
  `feat: stream durably committed pilot evidence to bounded archive handoff`.

## Task 5: Complete single-episode readers, reporting and cold recovery

**Files:** Create `archive/readers.py` and
`tests/pilot/test_pilot_archive_readers.py`. Modify
`report/{pilot_artifacts,pilot}.py`, `eval/artifacts.py`,
`train/{pilot_artifact_index,pilot_trainer,pilot_workflow,pilot_evidence,pilot_checks}.py`.
Extend `tests/pilot/{test_pilot_artifact_index,test_pilot_checks,test_pilot_report,test_pilot_gate_verifier}.py`.

**Interfaces produced:**

```python
def load_evaluation_header(root: Path) -> EvaluationHeader: ...
def iter_evaluation_rows(root: Path, header: EvaluationHeader
                         ) -> Iterator[IndexedRow]: ...
def verify_evaluation_episode(root: Path, *, header: EvaluationHeader,
                              indexed_row: IndexedRow) -> VerifiedEpisode: ...
def finish_evaluation_scan(header: EvaluationHeader,
                           accumulator: EvaluationScanAccumulator) -> None: ...
def iter_journal_records(run_dir: Path, head: str | None, *,
                         evidence_context: EvidenceContext | None = None
                         ) -> Iterator[dict]: ...
def read_training_envelope(run_dir: Path, path: Path) -> TrainingResultEnvelope: ...
def verify_training_inventory(*, run_dir: Path,
                              envelope: TrainingResultEnvelope,
                              evidence_context: EvidenceContext) -> None: ...
```

Header/indexed-row/verified-episode/accumulator types live in
`report/pilot_artifacts.py` (or a focused adjacent module if necessary).
`EvaluationHeader` contains authenticated identity, retention, rows index,
metrics, crash index and original DONE file hashes, within existing file limits.
`IndexedRow` binds the parsed row and exact ordinal/offset/length/hash.
`VerifiedEpisode` carries the validated row, already-parsed neural sidecar,
optional decoded trajectory and checked crash references. The accumulator tracks
exact ordinal/cardinality, classified paths, metrics and required plot samples.
Its successful closure requires every expected member and row, not just a prefix.
`TrainingResultEnvelope` is the existing type in `train/pilot_artifact_index.py`;
the reader does not replace or silently upgrade its wire schema.

Add optional `evidence_context=None` to aggregate entrypoints `verify_journal`,
`_collect_artifact_hashes`, `_durable`, `load_training_result`,
`training_evaluation_status`, `evaluation_directories`, `authenticate_run`,
`collect_pilot_evidence`, `verify_phase4_gate_artifact`, `build_pilot_report`,
`run_pilot_checks` and `recover_pilot_checks`. Validate logical-root agreement and
thread private callers explicitly. Preserve eager `load_evaluation(Path)` using
the same scanner internally; archive callers hold metadata and then one episode
lease. Low replay/Path readers remain local and retain their integrity checks.

Let `_verify_evidence` return the sidecar it already parsed, while keeping every
existing validation. `_compact_rows` consumes verified episode data instead of
reopening files after lease release. Report figures use only already-decoded
selected positive/negative examples, or reopen an explicit selected episode lease.
No entire evaluation hydration for finalization, reporting, gate, replay or recovery.

- [ ] **Step 1 — RED:** use actual retained debug-run artifacts, not dummy metrics.
  Archive/evict episode payloads and multiple catalog/journal pages. Compare eager
  versus streaming results and assert maximum active episode leases equals one.
  Include retained failures, crash checkpoints/shared weights and abandoned tails.
- [ ] **Step 2 — Run RED:**
  `uv run pytest -q tests/pilot/test_pilot_archive_readers.py tests/pilot/test_pilot_artifact_index.py tests/pilot/test_pilot_report.py tests/pilot/test_pilot_gate_verifier.py`.
- [ ] **Step 3 — Implement header/row/episode/closure scanning.** Reuse row identity
  and semantic checks, plus exact offset/hash binding. Classify every DONE member
  once as metadata, shared input or episode-owned evidence. Check telemetry,
  optional traces, crash manifests/checkpoints/weights and their identity links.
  Reject unclassified/duplicate/missing files, reordered rows and malformed final
  index entries. Verify all original hashes while consuming leased bytes.
  Publish nothing passing until the scanner is exhausted and closure passes.
- [ ] **Step 4 — RED/GREEN: adversarial tails and partials.** Corruption in the
  final episode/page must fail despite every earlier episode succeeding. Rehash
  a transport catalog around semantically corrupt evidence and confirm rejection.
  Partial scans retain newline-complete rows and orphan crash bytes, without
  manufacturing outcomes for unterminated/missing rows. Preserve current missing/
  unavailable semantic-check reporting, including independent available failures.
- [ ] **Step 5 — Implement paged inventory and journal validation.** Stream original
  training-index shards against transport pages, authenticate every raw indexed
  file, preserve exact journal update coverage `N..1` and committed/abandoned
  bindings. Do not leave all index/catalog shards or receipts on disk. Keep the
  v2 training-result wire format; old fully local envelopes remain readable.
  Session-local scan reuse requires immutable input hashes and verifier-source
  closure; transport receipts never replace raw/semantic verification. Do not
  serialize another giant mapping or claim lazy RAM use for legacy dictionary APIs.
- [ ] **Step 6 — RED/GREEN: reports and artifact-only gates.** Compare compact
  attachments, metrics, errors and plots to eager fixtures. All figure/trace
  reads occur within leases or use verified decoded examples. Preserve
  `neural_replay='not_rerun'`; tripwire neural execution in artifact verification.
  Separately execute actual CPU replay from an episode lease plus exact weights.
- [ ] **Step 7 — Implement bounded fresh-destination recovery.** Authenticate every
  indexed input before new neural work, in episode/page succession. Pin controls
  and bind immutable originals into the destination catalog; retain no whole
  evaluation trees. Preserve original gate bytes/source/config, TOCTOU checks,
  nested-destination rejection, eager local recovery and original failed data.
  Count source/destination/staging together under the same 10 GiB ledger.
- [ ] **Step 8 — GREEN/review/commit:** repeat Step 2 tests plus actual debug cold
  resume and recovery. Assert identical CPU next batch/progress/selected tensors,
  journal coverage and scientific payloads. Test two competing readers, absent
  weights, corrupt last shard, earlier failed continuation followed by success,
  missing remote data and peak disk accounting including metadata. Commit:
  `feat: verify and recover cold pilot evidence one episode at a time`.

## Task 6: Explicit local commands and measured storage preflight

**Files:** Create `archive/{cli,preflight}.py` and matching tests. Modify
`src/silent_cascade/cli.py`, `Makefile`, source/local-verification inventories and
their tests, and `train/{pilot_checks,pilot_verification}.py` for counted test
scratch/log/capture publication. Update `docs/r2-archive-setup.md`, `docs/phase4-autonomous-eventflow.md`
and `docs/deviations.md` with implemented behavior, not future commands.

**Interfaces produced:**

```python
def assess_archive_space(*, policy: ArchivePolicy, workspace: Path,
                         bounds: ArtifactOutputBounds,
                         measurements: ArchiveMeasurements,
                         retained_upper_bytes: int) -> ArchiveSpaceReport: ...
def archive_doctor(*, operational_config: Path, output_dir: Path) -> Path: ...
```

Define strict `ArtifactOutputBounds`, `ArchiveMeasurements` and
`ArchiveSpaceReport` in `archive/types.py`. Bounds carry maximum episode/atomic/
crash/shared-weight/update/checkpoint/control-file bytes derived from serializer
limits and artifact counts, plus their source hash. Measurements carry observed
per-category allocations, concurrent reservations, qualification/test output,
physical volume/free bytes and transfer/readback times. The report is not an
arbitrary success dict: bind all bounds, actual free/allocated bytes, policy hash,
retention forecast, source, transfer/readback measurements, AWS version and each
individual pass/fail condition. Keep old `forecast_workload` output unchanged;
the archive certificate is additional evidence, not an edited historical report.
Compute exact worst-variant row/control sizes over the actual corpus before
execution: the 10,000-row `validation.json` must fit its existing 64 MiB reader,
and evaluation rows/index/DONE must fit their 128 MiB readers. Do not invent a
smaller per-row acceptance cap or shrink the denominator. For verification, route
TMPDIR/TMP/TEMP, pytest basetemp, build/cache output and recorder logs into counted
space; preserve existing receipt-reader limits and all bytes already emitted.
If safe scratch/log/capture bounds cannot be established, fail preflight or stop
with retained failure evidence, not a partial passing quality-gate receipt.

- [ ] **Step 1 — RED:** command import/argument/offline tests and resource
  arithmetic tests. Example pure sizing case:

  ```python
  def test_archive_budget_does_not_erase_total_retention(
      tmp_path, permitted_output_bounds, small_archive_measurements
  ):
      result = assess_archive_space(
          policy=ArchivePolicy(), workspace=tmp_path,
          bounds=permitted_output_bounds,
          measurements=small_archive_measurements,
          retained_upper_bytes=3_675_694_793_937)
      assert result.retained_upper_bytes == 3_675_694_793_937
      assert result.policy_workspace_bytes == 10 * 1024**3
      assert result.normal_allocation_limit_bytes == 6656 * 1024**2
      assert result.emergency_allocation_limit_bytes == 8 * 1024**3
      assert result.protected_headroom_bytes == 2 * 1024**3
  ```

  Define both fixtures from explicit toy serializer maxima, file counts and
  allocation records; they test arithmetic only, never certify production.

  Inject disk observations in dedicated tests to exercise insufficient free
  space, emergency reservation, partial-download accounting, underestimated unit
  growth, pinned-data growth and remote-budget refusal without allocating GiB.
  A whole 23.60 GB evaluation restore, missing output maximum, out-of-ledger
  test directory, or accumulating catalog/log files must fail admission. A
  footprint that fits the envelope but leaves insufficient physical free space
  must also fail; filesystem capacity never enlarges authorization.
- [ ] **Step 2 — Run RED:** `uv run pytest -q tests/archive/test_cli.py tests/archive/test_preflight.py tests/pilot/test_pilot_cli.py tests/pilot/test_pilot_source.py`.
- [ ] **Step 3 — Expose only working commands.** Add lazy `silent-cascade archive`
  subcommands `doctor`, `status`, `sync`, `restore`, `run`. `sync` consumes explicit
  sealed units; `restore` needs a pinned catalog root, an explicit episode/segment
  selector and destination. Reject full evaluation-tree hydration above the bound;
  `run --job {pilot,checks,verify,report,replay}` wraps only known existing package
  entrypoints. Local-only `status` must work without AWS installed. Network is
  explicit in help text. No generic shell runner, credential-print command,
  remote delete, sync-delete or automatically enabled archive mode.
- [ ] **Step 4 — Operational config stays separate.** Store private machine
  settings under the existing protected user configuration directory: profile,
  bucket, prefix, durable workspace and policy values. No keys in this file;
  reuse credential_process. Version only the policy schema/defaults. Add a
  `pilot-r2` Make target requiring an explicit config path; ordinary `pilot` and
  `make verify` never initiate cloud access. Scientific config hash remains based
  on the unchanged experiment config, with operational-policy hash separately bound.
- [ ] **Step 5 — Implement resource/cost reporting.** Include 1.2 times observed
  episode and pinned sizes, proven output bounds, chunk/temp/emergency/checkpoint allowances,
  actual allocated disk and protected free reserve. Remeasure original retained
  fixtures by reading them only; never relocate/delete them. Report upload and
  readback time separately from compute. Show pending/remote bytes and usage-based
  cost assumptions, but do not claim an account spending cap or convergence ETA.
- [ ] **Step 6 — GREEN and review:** fixture CLI end-to-end exercises the actual
  supervisor and source closure; ordinary imports/doctor/train/report/verify stay
  network-free. Check clean-checkout package inclusion and no optional AWS Python
  dependency. Commit: `feat: expose explicit R2 pilot supervision and storage preflight`.

## Task 7: Storage gate, real R2 verification and Task 12 handoff

**Files:** Retain raw diagnostics in the explicitly configured operational
workspace; commit a compact storage-gate report plus source/policy/catalog hashes
under `artifacts/phase4-r2-storage/`. Update setup/status/deviation docs and the
delivery navigation index. Do not mark Phase 4 Complete.

- [ ] **Step 1 — Review/freeze implementation source.** All task reviews closed,
  clean relevant worktree, no other numerical owner. Run the local source-bound
  verification recorder (`uv run python scripts/record_phase4_local_verify.py`)
  once after source is stable; it runs the complete `make verify` test obligations.
  First qualify scratch accounting under the same ledger; route local test temps
  into counted space and stop if bounds cannot be established. Never silently
  reduce test coverage or discard required failure evidence to fit. Read actual
  exit status and receipt/logs; preserve failure evidence.
- [ ] **Step 2 — Real bounded remote qualification.** Use a new isolated test
  prefix in the project bucket, never existing scientific objects. Upload a
  deterministic 64 MiB probe and one actual copied debug episode through
  the new tool; verify readback, create-only collision handling, interrupted
  transfer recovery and exact restore. Limit qualification's total new remote
  bytes to 512 MiB across qualification and the following smoke; reserve catalog
  overhead too. Count all local copies against the shared 10 GiB ledger. Never
  mutate original debug evidence. Keep test bytes as labeled diagnostic evidence;
  local copies may use the same verified-before-eviction protocol, never blind cleanup.
- [ ] **Step 3 — Actual offline archive-backed smoke.** Run the real tiny
  four-update pilot recipe with forced small transport/journal units and local
  quota pressure, then restored artifact verification, report and CPU replay.
  Demonstrate the operational parent uses R2 while the scientific child has zero
  network/import attempts and zero foundation-model calls. Retain all outputs;
  identify the run as `debug_non_acceptance`, not production evidence.
- [ ] **Step 4 — Resource proof.** Establish maxima from serializer limits/file
  counts, then measure real episode/crash/checkpoint/metadata/journal sizes and
  scale the unchanged workload. Verify one-episode concurrency and peak combined
  allocation under pressure, restart, cold report/replay and fresh-destination
  recovery. Demonstrate normal <=6.5 GiB, emergency <=8 GiB and 2 GiB protected
  headroom within the 10 GiB envelope, including retained qualification/log/test
  data and directory/catalog growth. No whole-run hydration or hidden scratch.
  The measured 1.2 factor supplements proven admission bounds, not replaces them.
  Publish total retention, resource proof, remote budget, throughput/readback and
  limitations. A missing proof is a failed gate, not permission to allocate more.
- [ ] **Step 5 — Independent integration review and commit.** Review custody of
  every adverse artifact, exact logical inventory, no sole-copy deletion, cold
  journal resume, bounded recovery, offline boundary and compatibility. Commit
  `docs: record verified bounded R2 storage gate` only if the actual gate passes;
  otherwise record the specific failure and keep Task 12 unstarted.
- [ ] **Step 6 — Resume approved Phase 4 Task 12 only after both preflights pass.**
  Generate/introduce all four validation manifests and audit reports at reviewed
  source, then run the single seed-11 pilot through `pilot-r2` on the selected
  measured device. Preserve producer/data-introduction/training commit separation.
  Stop naturally at the existing learning/step/patience gates; storage readiness
  does not promise learning success. All final CPU/MPS/replay/report obligations
  and whole-phase review remain exactly those of the parent plan.

## 4. Self-review and acceptance checklist

- [ ] No new implementation begins before explicit approval of this plan.
- [ ] Exact raw bytes/paths and complete inventories survive archival/restoration.
- [ ] Full evaluations, all journal records, partial attempts and control recovery
  snapshots are covered; no receipt substitutes for semantic verification.
- [ ] Completed units and checkpoint-committed units remain distinct.
- [ ] Local disk is bounded including transient/abandoned data, not only successes.
- [ ] The complete 10 GiB envelope includes tests/logs/metadata/atomic staging and
  protected headroom; no path requires an entire evaluation locally.
- [ ] Row fsync precedes episode commit; shared crash weights have independent
  ownership, and finalization authenticates all 10,000 rows before DONE.
- [ ] Network loss, disk pressure and missing remote evidence cannot yield a pass.
- [ ] Offline scientific child and separate operational parent are tested with
  real work, not only mocked high-level metrics.
- [ ] Existing local artifact/recovery APIs and historical evidence remain valid.
- [ ] New plan/design/executable closure is independently authenticated.
- [ ] All tests run locally; no CI/CD or hidden downloads are introduced.
- [ ] Original Phase 4 Task 12 and scientific gates remain incomplete until their
  own evidence actually passes.

**Planning self-review:** All nine sections of the companion storage design map
to Tasks 1–7. Revision 2 replaces the rejected whole-evaluation architecture;
episode commit, crash ownership, paged metadata, reader and recovery changes are explicit;
limits are not represented as retention waivers. Commands above are implementation
instructions, not claims that the archive CLI exists yet. Execution follows the
user's established subagent-driven, current-branch workflow after plan approval.
