# Phase 4 Bounded R2 Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing seed-11 Phase 4 pilot executable with bounded local
disk and verified R2 retention, without changing its scientific protocol.

**Architecture:** Keep scientific files byte-identical and archive whole sealed
evaluation units plus immutable journal segments. An explicit finite local
supervisor owns R2 transfers; offline scientific workers use authenticated local
leases. Aggregate readers visit archived units sequentially and retain all current
semantic, provenance and failure checks.

**Tech Stack:** Existing locked Python 3.12/Pydantic/pytest/Hypothesis, standard
library streaming I/O and subprocesses, existing AWS CLI v2 administration profile.
No new ML/cloud SDK dependency, filesystem mount, database or hosted automation.

**Spec:** `docs/superpowers/specs/2026-09-17-phase-4-r2-archive-design.md`, under
canonical `docs/superpowers/specs/2026-08-30-silent-cascade-design.md` v1.0.2.

**Status:** Proposed; awaiting explicit approval of this plan before source,
test, build or configuration changes. This document does not mark Task 12 started.

**Inspected baseline:** `main` at `c09faee` on 2026-09-17. Stay on the current
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
- Do not change neural models, losses, engine scheduling, simulated times, generator semantics, counters or source-bound historical evidence.
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

This plan is not approval to bypass the canonical protocol or copy old passing
receipts to a changed executable. The production pilot still begins only after
the existing Task 12 preflight and the new storage gate both pass.

## 2. Design decisions and limits

- Archive complete `attempt-*/validation-*` directories after `_validation`
  finishes its autonomous/component/validation outputs. A validation unit is a
  storage-complete unit even if not yet checkpoint-committed; preserve that
  distinction in the catalog and later journal/restart bindings.
- Final evaluation units are `final/eval/<suite>` after `DONE` and `execution.json`.
  Keep the existing full 10,000-row denominator inside each unit.
- Journal units contain exact immutable `journal-<sha>.json` files, segmented at
  record boundaries by actual bytes: at most 1 GiB and 1,000 records. A committed
  segment binds the checkpoint-authenticated journal head; stopped-owner tails
  stay explicitly uncommitted/abandoned.
- Retain rolling checkpoint/control inputs locally; back them up in immutable
  checkpoint-bound snapshots. A snapshot is a recovery version, not a duplicate
  owner of a logical file in the active catalog.
- Total workspace limit 64 GiB; one large evaluation unit limit 32 GiB; one
  journal lease up to 1 GiB; chunk limit 256 MiB; one network operation at a time.
  Count partial restores, pack/download buffers, control state and logs.
- Keep at least 16 GiB filesystem reserve plus separately derived in-flight
  output/checkpoint emergency reservation. Fail before production if this cannot
  be met on the configured volume. Prefer a durable system-volume run directory;
  the inspected repository volume has only about 15 GiB free.
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
| 2 | `archive/transport.py`; `tests/archive/test_transport.py` | existing local AWS CLI profile only |
| 3 | `archive/{session,supervisor}.py`; `tests/archive/{test_session,test_supervisor}.py` | `train/pilot_offline.py` denial hooks |
| 4 | `archive/producer.py`; `tests/pilot/test_pilot_archive_producer.py` | trainer, workflow, runner, final-check driver, provenance inventories |
| 5 | `archive/readers.py`; `tests/pilot/test_pilot_archive_readers.py` | report/evidence/index/journal/recovery aggregate layers |
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
    workspace_bytes: int = 64 * 1024**3
    evaluation_bytes: int = 32 * 1024**3
    journal_bytes: int = 1024**3
    journal_records: int = 1000
    chunk_bytes: int = 256 * 1024**2
    reserve_bytes: int = 16 * 1024**3
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
              policy: ArchivePolicy) -> UnitRef: ...
def iter_unit_files(control_dir: Path, ref: UnitRef) -> Iterator[FileEntry]: ...
def iter_unit_chunks(*, run_dir: Path, control_dir: Path, ref: UnitRef,
                     scratch_dir: Path, policy: ArchivePolicy) -> Iterator[Path]: ...
def restore_unit(*, control_dir: Path, ref: UnitRef,
                 chunks: Iterable[Path], destination: Path,
                 policy: ArchivePolicy) -> Path: ...
```

`kind` accepts only `validation`, `evaluation`, `journal`, `control_snapshot`,
`partial`. Identity binds run ID, source/config hashes and the relevant existing
evaluation/checkpoint identity; no bare user-provided label establishes a seal.
Define a strict `UnitIdentity` with `run_id`, forty-hex `source_commit`, sixty-four-
hex `config_sha256` and `evidence_identity_sha256`, optional `checkpoint_sha256`,
and exact booleans `writer_stopped` and `checkpoint_committed`. Reject extra keys.
The producer validates those values against its actual source/lock/evidence;
synthetic codec fixtures never establish a scientific source or acceptance gate.
Reject booleans/nonpositive values for byte/count limits, require
`chunk_bytes <= journal_bytes <= evaluation_bytes < workspace_bytes`, and never
permit an operational override above the production ceilings in Section 2.
Use strict manifest/span/shard models in `types.py`. Separate original logical
paths from snapshot-version paths; reject duplicate active ownership, not merely
shared parent directories among disjoint journal segments.

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
  length)` spans. Concatenate original bytes into at most 256 MiB chunks; no tar
  extraction, pickle, recompression or complete second bundle on disk. Two passes
  over sealed files are allowed: compute manifest, then deterministically rebuild
  one chunk at a time. Recheck file identity/hash to reject mutation between passes.
  Validate expanded totals before allocation and during every bounded write.
  Stage restore under an owned absent directory, fsync, then publish atomically.
- [ ] **Step 4 — GREEN and boundary tests:** exact round trip at small injected
  chunk sizes, split-file spans, empty regular files, byte/count limits, symlink
  parent swaps, changed source after sealing, crash before restore publication.
  Test complete inventories larger than one shard without large real allocations.
- [ ] **Step 5 — Review and commit:** `feat: add bounded immutable artifact units`.

## Task 2: R2 transfer, readback receipts and safe eviction

**Files:** `archive/transport.py`, `tests/archive/test_transport.py`.
Consumes Task 1 units and safe no-follow file operations.

**Interfaces produced:**

```python
class ObjectTransport(Protocol):
    def create(self, key: str, source: Path) -> None: ...
    def download(self, key: str, destination: Path) -> None: ...

class R2CliTransport:
    def __init__(self, *, profile: str, bucket: str, prefix: str): ...

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
  credential expiry, quota refusal, concurrent lease, exact cleanup/recovery and
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
  response from another run, response-before-complete-restore, simultaneous large
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
- [ ] **Step 4 — Implement bounded scheduling.** At most one materialized large
  evaluation, one journal segment and pinned control data. Account for all stages,
  not just finished files. Release never evicts a live reader. Use 30-second
  health-check waits with status updates. Each AWS operation has a 30-minute
  timeout and at most three transient attempts; a multi-chunk unit renews progress
  only after each verified chunk and may legitimately take longer. Parent death
  or exhaustion yields actionable `storage_blocked`; preserve pending files.
  Do not mistake total unit-transfer time for a stalled request. The exact pilot
  process retains/uses its existing durable checkpoint on abort.
- [ ] **Step 5 — GREEN:** real subprocess IPC; no `aws`/cloud imports in the
  child; no credential environment forwarding; no new simulated events while
  waiting; all children joined; partial staging quarantined on crash; no global
  background daemon or hidden arbitrary-command dispatcher.
- [ ] **Step 6 — Review and commit:** `feat: isolate archive I/O from offline evidence leases`.

## Task 4: Producer handoff, bounded journals and resume-safe pressure

**Files:** Create `archive/producer.py` and `tests/pilot/test_pilot_archive_producer.py`.
Modify `train/{pilot_trainer,pilot_workflow,pilot_provenance}.py`,
`eval/runner.py`, `train/pilot_checks.py`, and the exact source inventories in
`train/pilot_evidence.py` plus `tests/pilot/test_pilot_source.py`.
Do not modify model/loss/engine/generator modules or checkpoint wire format.

**Interfaces produced:**

```python
class ArchiveProducer:
    def before_update(self, global_step: int) -> None: ...
    def before_evaluation(self, logical_root: str) -> None: ...
    def before_episode(self, logical_root: str, ordinal: int) -> None: ...
    def after_validation(self, logical_root: str) -> None: ...
    def after_checkpoint(self, descriptor: PilotCheckpointDescriptor,
                         progress: PilotProgress) -> None: ...

def seal_journal_segments(*, run_dir: Path, control_dir: Path,
                          base_head: str | None, committed_head: str,
                          checkpoint: PilotCheckpointDescriptor,
                          policy: ArchivePolicy) -> tuple[UnitRef, ...]: ...
```

Thread optional `archive_producer=None` through `run_pilot_training`, `_workflow`,
`run_pilot`, `_validation`, `evaluate_episodes`, `_run_pilot_checks_owned` and
`_evaluate`. Default local behavior stays identical. A producer can perform only
storage-boundary checks/requests; it never sees or edits agent state, predictions
or private truth to decide which evidence to keep.

- [ ] **Step 1 — RED:** extend the actual four-step debug pilot fixture to record
  writer callbacks. Require preflight before the first update, handoff after each
  fully published validation, and checkpoint/journal sealing only after the real
  durable index publication. Force tiny journal segment limits and verify exact
  predecessor links, hashes, order and full coverage.
- [ ] **Step 2 — Run RED:** `uv run pytest -q tests/pilot/test_pilot_archive_producer.py tests/pilot/test_pilot_trainer.py tests/pilot/test_pilot_workflow.py`.
- [ ] **Step 3 — Implement evaluation boundaries.** Call `before_episode` before
  neural work, reserving enough space for the full legal in-flight output. The
  row writer, retention and `DONE` semantics remain unchanged. Handoff only after
  `_validation` rereads rows and publishes its sibling record. Drain robustness
  evidence before its paired primary evaluation; preserve artifact hashes in
  memory/catalog for the later validation journal. Final suite handoff occurs
  only after `execution.json`, not just `DONE`.
  Call `before_update` before each optimization step and reserve its bounded
  journal/checkpoint output; measured journal sizes are not an allocation bound.
- [ ] **Step 4 — Implement journal segmentation/control snapshots.** Never split
  a record or rewrite its bytes. Bind predecessor/final heads, ordered records,
  exact bytes, source/config/run and durable checkpoint. Multiple segments may
  cover an interval. Classify only the checkpoint prefix as committed. Back up
  pinned control snapshots before declaring an interval archived; keep active
  mutable checkpoint/index files local and exclude operational IPC from inventories.
- [ ] **Step 5 — Preserve interruption behavior.** A pressure/error exit retains
  failure/partial artifacts. Resume enumerates cold and local journals, not only
  `glob` results; stale owner recovery uses the context-aware verifier from Task 5.
  Do not treat `storage_blocked` as `early_stopping` or create a completed result.
  Tests may use a lease test double until Task 5 supplies the real cold reader.
- [ ] **Step 6 — GREEN and review:** prove no changed optimizer step, source
  budget, RNG or event schedule with a recording-only producer; prove all bytes
  remain when upload/pressure fails. Independently review source closure additions
  (including this plan/design) and preserve historical verification tuples.
- [ ] **Step 7 — Commit:** `feat: bound pilot artifact production with safe archive handoff`.

## Task 5: Complete cold-evidence verification, reporting and recovery

**Files:** Create `archive/readers.py`, `tests/pilot/test_pilot_archive_readers.py`.
Modify `report/{pilot_artifacts,pilot}.py`,
`train/{pilot_artifact_index,pilot_trainer,pilot_workflow,pilot_evidence,pilot_checks}.py`.
Extend `tests/pilot/{test_pilot_artifact_index,test_pilot_checks,test_pilot_report,test_pilot_gate_verifier}.py`.

**Interfaces produced:**

```python
def iter_journal_records(run_dir: Path, head: str | None, *,
                         evidence_context: EvidenceContext | None = None
                         ) -> Iterator[dict]: ...
def read_training_envelope(run_dir: Path, path: Path) -> TrainingResultEnvelope: ...
def verify_training_inventory(*, run_dir: Path,
                              envelope: TrainingResultEnvelope,
                              evidence_context: EvidenceContext) -> None: ...
```

Add optional `evidence_context=None` to aggregate entrypoints `verify_journal`,
`_collect_artifact_hashes`, `_durable`, `load_training_result`,
`training_evaluation_status`, `evaluation_directories`, `authenticate_run`,
`collect_pilot_evidence`, `verify_phase4_gate_artifact`, `build_pilot_report`,
`run_pilot_checks` and `recover_pilot_checks`. When supplied, it must name the same
logical run as `run_dir`/`raw_run_dir`; reject conflicting roots rather than
silently choosing one. Thread through their private callers explicitly.

Keep `load_evaluation(Path)`, `_verify_evidence` and
`verify_neural_replay(path, weights_path=...)` Path-based and semantically unchanged.
Cold aggregate paths use the small envelope/streaming inventory first; retain
legacy materialization only where the existing public result contract requires
it. Do not allocate a second serialized giant inventory or claim a lazy public
API where a dictionary remains materialized.
Archive-backed training initially requires the existing v2 training-result
envelope; older envelopes remain supported by the unchanged fully local path.
Do not silently rewrite an old result into v2 to make it eligible for archival.

- [ ] **Step 1 — RED:** use actual retained debug-run data, not fake metrics.
  Archive and evict two evaluation trees and multiple journal segments; compare
  all-local and cold-reader results while asserting only one large lease at once.
  Mutate a semantic field and consistently rehash its transport catalog: the
  unchanged semantic validator must still reject the scientific contradiction.
- [ ] **Step 2 — Run RED:** `uv run pytest -q tests/pilot/test_pilot_archive_readers.py tests/pilot/test_pilot_artifact_index.py tests/pilot/test_pilot_report.py tests/pilot/test_pilot_gate_verifier.py`.
- [ ] **Step 3 — Implement complete logical inventory and journal traversal.**
  Merge catalog entries against original authoritative index/journal/DONE hashes;
  reject extra, missing, duplicate or cross-run entries and final-shard corruption.
  Preserve exact update coverage `N..1`, predecessor hashes, committed validation
  bindings and authenticated abandoned tails. Hold one journal lease while
  reading its records; release or copy bounded record metadata before leasing
  referenced evaluations. No hash check is skipped because a file is cold.
- [ ] **Step 4 — Implement sequential report/gate reads.** Discover completed,
  failed and partial logical trees from catalog plus local inventory. Make all
  plot/trace accesses inside the corresponding lease. Recompute every denominator,
  trajectory check, raw-record digest and available numeric/continuation/offline
  semantic check. A missing remote object produces exact missing/unavailable
  evidence, never a passing archived receipt. Keep verifier return semantics
  including `neural_replay='not_rerun'`; execute replay only through the existing
  separate replay obligation with restored exact weights.
- [ ] **Step 5 — Implement opt-in bounded recovery.** Authenticate every indexed
  input in successive leases before executing any new final check. Create a fresh
  destination with pinned inputs and a separate immutable catalog binding the
  verified original units; do not accumulate all restored trees. Bind original
  gate bytes/hash, catalog root, policy and unchanged source/config in recovery
  intent. Preserve old eager recovery, nested-destination rejection, TOCTOU gate
  checks and all original failed artifacts. Missing required archive bytes must
  fail before neural execution, not refit/reselect or relabel a past result.
- [ ] **Step 6 — GREEN:** actual all-local/cold resume gives identical CPU next
  batch, progress, selected model tensors and journal coverage. Semantics and
  errors match for missing last shard, missing unrelated file with available
  contradiction, earlier failed continuation plus later success, absent weights,
  corrupted archive, partial report, transport-success/semantic-failure and two
  independent readers. Tests tripwire neural execution inside artifact verification.
- [ ] **Step 7 — Review and commit:** `feat: verify and recover archived pilot evidence in bounded leases`.

## Task 6: Explicit local commands and measured storage preflight

**Files:** Create `archive/{cli,preflight}.py` and matching tests. Modify
`src/silent_cascade/cli.py`, `Makefile`, source/local-verification inventories and
their tests. Update `docs/r2-archive-setup.md`, `docs/phase4-autonomous-eventflow.md`
and `docs/deviations.md` with implemented behavior, not future commands.

**Interfaces produced:**

```python
def assess_archive_space(*, policy: ArchivePolicy, workspace: Path,
                         measured_unit_bytes: int, measured_pinned_bytes: int,
                         emergency_bytes: int, retained_upper_bytes: int,
                         checkpoint_working_bytes: int) -> dict: ...
def archive_doctor(*, operational_config: Path, output_dir: Path) -> Path: ...
```

The returned report is a strict versioned schema, not an arbitrary success dict:
bind actual volume/free/allocated bytes, probe files, maximum unit, policy hash,
retention forecast, source, transfer/readback measurements, AWS version and each
individual pass/fail condition. Keep old `forecast_workload` output unchanged;
the archive certificate is additional evidence, not an edited historical report.

- [ ] **Step 1 — RED:** command import/argument/offline tests and resource
  arithmetic tests. Example pure sizing case:

  ```python
  def test_archive_budget_does_not_erase_total_retention(tmp_path):
      result = assess_archive_space(
          policy=ArchivePolicy(), workspace=tmp_path,
          measured_unit_bytes=23_600_210_000,
          measured_pinned_bytes=2_000_000_000,
          emergency_bytes=2_000_000_000,
          retained_upper_bytes=3_675_694_793_937,
          checkpoint_working_bytes=200_741_100)
      assert result["retained_upper_bytes"] == 3_675_694_793_937
      assert result["policy_workspace_bytes"] == 64 * 1024**3
      assert "observed_free_bytes" in result
      assert "storage_ready" in result
  ```

  Inject disk observations in dedicated tests to exercise insufficient free
  space, emergency reservation, partial-download accounting, underestimated unit
  growth, pinned-data growth and remote-budget refusal without allocating GiB.
- [ ] **Step 2 — Run RED:** `uv run pytest -q tests/archive/test_cli.py tests/archive/test_preflight.py tests/pilot/test_pilot_cli.py tests/pilot/test_pilot_source.py`.
- [ ] **Step 3 — Expose only working commands.** Add lazy `silent-cascade archive`
  subcommands `doctor`, `status`, `sync`, `restore`, `run`. `sync` consumes explicit
  sealed units; `restore` needs a pinned catalog root and explicit destination;
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
  unit and pinned working sizes, chunk/temp/emergency/checkpoint allowances,
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
  once after source is stable; it runs the complete unchanged `make verify`.
  Read actual exit status and receipt/logs; preserve failure evidence.
- [ ] **Step 2 — Real bounded remote qualification.** Use a new isolated test
  prefix in the project bucket, never existing scientific objects. Upload a
  deterministic 64 MiB probe and one actual copied debug evidence unit through
  the new tool; verify readback, create-only collision handling, interrupted
  transfer recovery and exact restore. Limit qualification's total new remote
  bytes to 512 MiB. Never mutate the original debug evidence. Retain test bytes
  as labeled diagnostic evidence until the owner chooses removal.
- [ ] **Step 3 — Actual offline archive-backed smoke.** Run the real tiny
  four-update pilot recipe with forced small transport/journal units and local
  quota pressure, then restored artifact verification, report and CPU replay.
  Demonstrate the operational parent uses R2 while the scientific child has zero
  network/import attempts and zero foundation-model calls. Retain all outputs;
  identify the run as `debug_non_acceptance`, not production evidence.
- [ ] **Step 4 — Resource proof.** Measure the real largest retained unit and
  journal/metadata/control sizes; scale the unchanged full workload explicitly.
  Verify one-large-unit concurrency and actual peak allocation with pressure and
  restart tests. A 32 GiB fit must be justified by the measured scenario plus
  safety factor and live refusal/reservation logic, not asserted as universal.
  Publish total retained forecast, bounded local need, free reserve, remote byte
  limit, throughput/readback observations and all remaining limitations.
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
- [ ] Network loss, disk pressure and missing remote evidence cannot yield a pass.
- [ ] Offline scientific child and separate operational parent are tested with
  real work, not only mocked high-level metrics.
- [ ] Existing local artifact/recovery APIs and historical evidence remain valid.
- [ ] New plan/design/executable closure is independently authenticated.
- [ ] All tests run locally; no CI/CD or hidden downloads are introduced.
- [ ] Original Phase 4 Task 12 and scientific gates remain incomplete until their
  own evidence actually passes.

**Planning self-review:** All nine sections of the companion storage design map
to Tasks 1–7. The producer, reader, journal and recovery changes are explicit;
limits are not represented as retention waivers. Commands above are implementation
instructions, not claims that the archive CLI exists yet. Execution follows the
user's established subagent-driven, current-branch workflow after plan approval.
