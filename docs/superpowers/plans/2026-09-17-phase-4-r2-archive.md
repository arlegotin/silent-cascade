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

### Live-qualification correction sequence (2026-09-18)

This is a bounded storage-only refinement of the approved Tasks 2/6/7 under the
standing autonomous-delivery instruction, not a new scientific protocol. The
first live attempt failed after confirmed create/SIGKILL/resume, at the compound
receipt-key validator. Preserve that failed result and all pending state. Keep
the original frozen SDD tree unchanged; subsequent counted coordination artifacts
live only in its existing `operational/sdd` subtree.

1. **Receipt-key contract:** RED-test both exact emitted compound key forms,
   malformed paths and real publisher integration through the production
   validator; minimally correct `archive/transport.py`; review the full fix.
2. **Reviewed source continuation:** add bounded create-only reviewed-release
   records/resolution in `archive/preflight.py`, carry the authority hash through
   `EngineeringCandidate` and the private qualification identity/result, and
   retain bootstrap compatibility. Test source/hash/anchor/policy corruption,
   revision-only changes, idempotent publication, historical preservation and
   engineering candidate/resume binding. Do not introduce a mutable authority
   pointer or change the original artifact source during transport recovery.
3. **Finite-operation accounting:** retain full `budget.check()` semantics;
   factor unchanged accounting inequalities into a private fast path using a
   freshly authenticated retained charge within one owned reservation. Wire it
   into qualification and supervisor hot paths only. Test all caps, live-owner
   and snapshot invalidation, full boundary authentication, child inheritance,
   no restart reuse, and bounded full-scan counts across repeated opportunities.
   Finish/reap writers before final authentication and success acceptance.
   Qualification requires three parent scans overall: the unchanged real
   reviewed-source precondition before ownership, scoped admission, and final
   staged-result authentication after cleanup. Assert three overall and two
   while the reservation is owned through the real source resolver; children
   require zero full scans. Supervisor still requires two overall. The previous
   count test replaced the source resolver and concealed its precondition scan.
   Keep source/custody authentication intact. The measured live-accounting
   correction precomputes validated category-prefix component tuples once per
   measurement and compares whole components in the existing longest-prefix
   order. RED-test repeated ancestor work deterministically and exact allocation
   for roots, overrides, prefix-like siblings, unknown paths and opaque scratch;
   preserve traversal, baseline hashing, caps and polling cadence.
4. **Same-ledger recovery and retry:** after source review/commit, publish the
   exact reviewed-source record, pre-admit recovery from existing descriptors,
   authenticate preserved pending evidence, and call existing `archive_unit`
   for the original probe ref/source. Log original artifact and recovery-executor
   tuples separately. Verify the pending state is resolved without changing the
   charged history. Bind a fresh qualification root and rerun create/readback,
   real interruption, collision, idempotence and restore under the same transport
  and shared qualification/smoke allowance. The old failed run stays failed.

#### Same-campaign qualification continuation refinement (2026-09-18)

The fresh `5633b60` attempt proved real create/SIGKILL, then its resume child
failed with a generic envelope. Exact-unit diagnostic resume subsequently
succeeded on unchanged source, including the original-key collision/readback.
The cause did not reproduce and is not declared fixed. Preserved scratch is now
188,784,640 bytes; another complete fresh qualification needs over103MiB more,
exceeding the unchanged256MiB category. The remaining checks fit without another
killed chunk or source copy. Implement the design's narrow continuation contract
before using composite evidence; ad-hoc helper composition is not a passing gate.

**Scope:** private `archive/_qualification.py`, an optional focused private
continuation module if it improves separation, focused archive qualification
tests, and exact executable/source inventories when a module is added. Preserve
all scientific modules, public commands, policy values and fresh-run results.

**Interfaces:** add one private `continue_early_r2_qualification` entry taking
the existing budget/transport, fresh output root, explicit predecessor/failure
and recovery proof paths plus expected hashes, original source roots and sealed
refs, reviewed debug EpisodeCommit/hash, protocol hash, and current reviewed
executor revision/authority. Use a strict distinct continuation result. The
task's reviewed interface proposal must specify its complete fields before code.
Historical source/authority is separate from the current executor; authenticate
both, never silently substitute one. Explicit fresh event-root plumbing through
parent/child helpers defaults to current fresh-run behavior for compatibility.

- [ ] Specify a bounded predecessor witness from the preserved failure phases,
  original sealed source identities and historical source authority. The resume
  stage was entered only after the historical driver checked premature receipt
  and authorization absence; prove that ordering from the authenticated source
  and exact phase witnesses. Refuse missing, forged, reordered or contradictory
  witnesses. Do not infer history from a currently completed receipt.
- [ ] RED-test wrong proof SHA, source/authority/policy/protocol/transport or
  unit binding, missing historical witness, and changed original bytes. Assert
  no provider calls or output directory on rejected preconditions.
- [ ] Add explicit event-root parameters to the existing parent/child seam.
  Test that a complete remaining-check execution leaves a digest of every old
  attempt file unchanged and puts all new phases only below the fresh root.
- [ ] Implement actual repeated collision/readback for the original interrupted
  chunk, using the existing bounded chunk/transfer/reservation validators. Reuse
  the existing child retry and receipt/streaming restore machinery. Do not copy
  a saved success dictionary or bypass any check at the fresh driver's equivalent
  interruption/retry/restore boundaries.
- [ ] RED/GREEN behavioral tests must reject no collision, a different collision
  key, no repeat downloads, changed receipt or reservations, missing remote
  bytes, overwrite of an existing restore destination, wrong inventory and
  byte-different restores. A valid local-double run restores both exact inputs
  and records the inherited interruption separately from newly executed checks;
  only real R2 can emit a provider qualification result.
- [ ] Derive remaining local and cumulative remote bounds before publication.
  Charge predecessor outputs and current shared history, complete restores,
  payload/readback coexistence, native allocated-byte overhead, catalog staging,
  metadata atomic peaks and bounded logs. No arbitrary smaller serializer limit
  or quota reset. Test refusal at each category/normal/global/physical/campaign
  limit, including previous failed usage consuming the remaining allowance.
- [ ] Preserve owned-group cleanup, fail-stop ownership on unreaped children,
  final full authentication, and create-only staged publication. Test cleanup
  failure and final-authentication failure cannot publish a result. Default
  checks remain full; scoped checks remain capability-bound.
- [ ] Add bounded non-secret failure location/type evidence at the private child
  boundary so future failures are diagnosable: no exception text, locals, paths,
  provider stdout/stderr or destination settings. Test hostile exception text
  does not escape. This does not retrospectively identify the observed failure.
- [ ] Run focused RED/GREEN and stable qualification/continuation/ledger covers
  within a separately admitted test slice, scoped lint/format/diff, commit on the
  current branch, and obtain independent task review. Freeze source, publish its
  exact reviewed authority, then execute the continuation under a fresh finite
  admission. Only its actual complete composite evidence unlocks engineering
  archival. Update status honestly; all later Task5/6/7 and pilot gates remain.

#### Continuation verification sequencing after retained test overrun (2026-09-18)

The first unchanged-source qualification/ledger cover passed122tests, but its
combined retained RED/GREEN/cover output exceeded the separately admitted8MiB
test slice. Preserve that failed admission and every output; the functional test
result is not a resource pass. Fresh ledger authentication still includes all
bytes and must satisfy the unchanged category/global/physical limits.

Before another provider call, correct the remaining-work native-allocation
allowance: an earlier real32MiB R2 readback allocated33,882,112bytes, exceeding
logical-size rounding. Use the already established conservative operational
allowance `2 * round_up(size, allocation_unit) + allocation_unit` for each new
download and each complete restored file, plus explicit concurrent debug payload,
catalog stages and directory/control overhead. Preserve sequential maxima.
This is not a universal filesystem theorem; exact logical serializer bounds,
live allocated-byte checks, refusal and fail-stop remain mandatory.

The unchanged256MiB scratch limit leaves only2,883,584bytes after the existing
retained allocation and the corrected large-readback allowance. Therefore this
ordering supersedes the immediate duplicate full-module cover above; it does
not waive Task7's complete current-source local verification before the pilot:

1. Commit the current implementation as reviewable work with its native-bound
   defect and failed test admission explicitly open; no completion/provider pass.
2. Obtain the independent task review before consuming the remaining test slice.
   Batch its findings with the known bound correction in one normal fix round.
3. Admit a separate2MiB affected-test slice, retaining all old output in the same
   ledger. Add descriptor-only native-allocation and inflated-accounting refusal
   regressions; exercise all continuation tests on the corrected frozen code.
   Use `pytest tests/archive/test_qualification.py -k 'continuation_'` with
   bytecode/cache disabled and unique preserved output roots. Capture owner PID
   and creation time before execution; measure actual growth during the run.
   No extra test run may silently consume this finite admission.
4. Review the fix and exact source. Report the earlier122-test result as pre-fix
   functional evidence only, alongside fresh affected-code results and the failed
   resource slice. This can support only the early provider qualification path;
   it is not a full current-source local-verification receipt.
5. Execute the complete native qualification continuation under newly derived
   bounds. Only a complete actual composite record allows engineering archival.
   After verified reclamation, run the full stable modules and Task7 source-bound
   `make verify` obligations before claiming final storage readiness or starting
   Phase4 Task12. No scientific gate, test manifest, quota or retention rule changes.

Each implementation slice gets RED/GREEN evidence, bounded local covering tests,
an independent task review and a meaningful current-branch commit. The scoped
briefs give exact tests/interfaces and finite output admission before dispatch.
No full `make verify`, provider call or later pilot is implied by a unit-test
pass. Full final gates remain unchanged. Do not bundle unrelated diagnostic
improvements into the deterministic validator correction.

Native cleanup refinement: diagnostics for the exact launched process group
observed EPERM after the known leader and descendant had exited, followed by
ESRCH on a bounded read-only probe. Supervisor and qualification cleanup may
continue disappearance polling after group-signal/probe PermissionError within
their existing deadlines. EPERM is never termination proof: only ESRCH permits
successful group cleanup, and persistent uncertainty must fail-stop while
retaining durable ownership. Preserve direct-child reap, other error handling
and transient/persistent regression tests; do not introduce broader signals,
process scans or longer deadlines.

#### Adversarial scratch accounting correction

The first maintenance admission's final check found two deliberately preserved
symlink-attack fixtures below the real scratch category. The old counter rejects
them before counting, preventing further admission although they occupy only
their own inode blocks. Do not delete or rewrite these fixtures, move them out
of the ledger, or weaken safe artifact I/O. This correction precedes source
continuation and performance work.

Modify only `archive/ledger.py` and focused `tests/archive/test_ledger.py` cases:

- Keep `require_path` strict: a symlink anywhere on an actual requested path is
  still an error. Explicit category roots, workspace roots, metadata controls
  and non-scratch categories retain existing rejection behavior.
- During physical measurement only, opaque descendants of an explicitly bound
  real scratch directory may contain adversarial symlinks/special files. Count
  their `lstat().st_blocks * 512` without opening/following them. Never recurse
  into a symlinked directory. Reject second-device entries. A bound scratch root
  itself may not be a symlink. Unknown default-category paths gain no exemption.
- Count scratch hard links conservatively once per name; do not subtract or
  credit deduplicated/external aliases. Actual archive/download/input APIs retain
  their single-link ownership checks. Non-scratch hard-link rejection remains.
- Add tests proving symlink inode blocks are included, targets are never read or
  walked, links to directories are not traversed, scratch-root/non-scratch links
  still fail, actual `require_path` still rejects fixture paths, and hard links
  consume their full conservative per-name charge. Existing physical/cap and
  baseline authentication tests must keep their meaning.
- Start with a zero-output RED reproduction: mock the filesystem boundary for
  real `measure()` using literal lstat/walk records and an in-memory state, not
  its result. Disable bytecode and every test cache. No tmp_path, temporary file,
  provider, real ledger write or generated report is permitted until the fixed
  counter passes a read-only check on the existing workspace. Then main obtains
  a fresh finite same-ledger admission for real-filesystem and covering tests.
- Use TDD, scoped lint/format, independent review and a separate commit. This
  deliberately distinguishes measuring opaque retained test bytes from granting
  access to an unsafe path; it is not an artifact format/codec or custody reset.

All paths below are repository-relative. New modules are focused local archive
utilities, not a general storage framework. Tasks are sequential; independent
read-only reviews may run alongside tests. Only one source/test writer and one
numerical execution owner at a time.

Execution refinement: Task5 is delivered in two independently reviewed sequential
slices: scanner/report/index/journal primitives, then remaining cold caller,
partial-evidence and fresh-recovery integration. All original Task5 requirements
remain binding. Between those slices, the narrowly scoped Task6 accounting/archive
prelude below may run when retained engineering output would prevent safe further
verification. It does not complete Task6/7 or waive their final stable-source gate.
Only one source/test writer runs at a time; existing evidence is never discarded
to fit another test run.

| Task | New files | Main existing integration points |
| --- | --- | --- |
| 1 | `src/silent_cascade/archive/{__init__,types,catalog,bundles}.py`; `tests/archive/{conftest,test_catalog,test_bundles}.py` | existing safe I/O/hash/compact-index utilities, reused without weakening |
| 2 | `archive/{transport,operational}.py`; `tests/archive/{test_transport,test_operational}.py` | Task1 typed catalog nodes; existing local AWS CLI profile only |
| 3 | `archive/{session,supervisor,ledger}.py`; `tests/archive/{test_session,test_supervisor,test_ledger}.py` | `train/pilot_offline.py` denial hooks; closed archive types/index seams; private-state ignore rule |
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

**Files:** `archive/{session,supervisor,ledger}.py`, corresponding tests;
extend `train/pilot_offline.py` only to share/test the existing denial boundary.
Narrowly extend archive types/catalog/operational/transport seams for the closed
episode-binding index, and ignore private `.silent-cascade-storage/` state.

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

def supervise_job(*, job: str, request: dict, workspace_root: Path, run_dir: Path,
                  control_dir: Path, transport: ObjectTransport,
                  policy: ArchivePolicy, output_bounds: JobOutputBounds) -> int: ...
```

`entries()` merges resident original files with active catalog ownership; never
includes transport/control files in the scientific inventory. `verify_inventory`
checks exact caller-supplied authoritative entries against authenticated transport
and resident bytes. Tasks4/5 produce those entries from journal/index/DONE and
episode-commit authorities; Task3 must not duplicate their scientific parsers or
treat transport consistency as semantic correctness.

The job set is closed. Strict per-job request models reject unknown keys,
commands, import paths and caller-supplied repository roots. The trusted source
repository is derived from the installed package. Resolve config/manifest inputs
as safe repository-relative paths and artifact/output inputs as safe run-relative
paths, rejecting traversal and symlinks through the existing safe-I/O rules.

| Job | Request fields | Existing entrypoint |
| --- | --- | --- |
| `pilot` | `config_path`, `manifest_dir`, `device` | `train.pilot_workflow.run_pilot` |
| `checks` | `config_path` | `train.pilot_checks.run_pilot_checks`; output `phase4-gate.json` |
| `verify` | `artifact_path` (default `phase4-gate.json`) | `train.pilot_evidence.verify_phase4_gate_artifact`, with trusted repo and raw run |
| `report` | `output_dir` (default `report`) | `report.pilot.build_pilot_report` |
| `replay` | `replay_path`, `weights_path` | `eventflow.neural_replay.verify_neural_replay` |

Use only the existing two pilot config overlays and `cpu`/`mps` device choices.
`verify` is artifact-only verification, not the complete local `make verify`
recorder. Task3 implements closed decoding, process isolation and the IPC runner;
Tasks4/5 thread producer/context hooks through those entrypoints. Do not claim
cold scientific execution readiness before that integration, or expose an
arbitrary-command runner to bridge the phased dependency.

Preserve required source-provenance subprocesses through a pinned trusted Git
at least version 2.45, requiring native `--no-lazy-fetch`. Resolve the real binary,
version and content identity in the operational parent before environment
scrubbing; fresh children validate and capture the immutable pin once. Nested
offline diagnostics receive the same captured pin through a dedicated sealed
environment field, not their retained scientific intent. Permit only the exact
existing provenance invocation forms, including plain `ls-files`, full-SHA
`rev-list --topo-order`, and the validated `cat-file --batch` stream. Constrain
cwd and environment, disable executable configuration features and deny all
transports. Only the pipe created for the validated batch invocation may serve
its input; do not exempt arbitrary file descriptors. Reject unsupported Git,
mutated pins/environments and extra arguments. Keep executable paths out of
scientific config, reports and archives. Task6 doctor diagnoses this prerequisite.

`JobOutputBounds` is a closed typed admission input binding job, canonical strict
request hash, policy hash, trusted executing-package source fingerprint and
per-category maximum bytes. Verify these bindings and durably reserve the complete
possible in-flight allocation before spawning a child. Missing or mismatched
bounds fail closed; there is no nominal fallback. Task6 derives scientifically
justified bounds from `ArtifactOutputBounds` and constructs this input, rather
than accepting arbitrary CLI maxima. Retain admission metadata in the durable
reservation, not a separate permanent per-job certificate history. Categories
with no output may be zero; the total must cover all actual permitted outputs.

Implement create-only `initialize_workspace_ledger` in `archive/ledger.py`.
One workspace-global descriptor lives at
`workspace_root/.silent-cascade-storage/workspace.json`, with bounded immutable
baseline pages, durable reservations and a permanent lock. A
`control_dir/storage-state.json` is only a locator binding run/control/policy to
that shared state, never a second allowance. All run/control/test/temp/cache/log
and recovery outputs must remain under the explicit workspace root on the same
device. Only explicitly authenticated unchanged pre-existing baseline entries
may be excluded from new-allocation charges; baseline pages/locks themselves are
new metadata. Missing/corrupt history or another run cannot reset usage. Task7
supplies measured production baseline evidence and counts all already-created
task/test/qualification output; taking a blanket snapshot of existing files to
declare them free is forbidden.

Move the shared `EpisodeCommit` type declaration specified in Task4 into
`archive/types.py` during Task3; Task4 still owns producing and fsyncing it.
Freeze a closed `EpisodeBinding` record with fields `schema_version` (literal
`phase4-archive-episode-binding-v1`), `run_id`, `logical_root`, `ordinal`,
`commit_sha256`, `unit_ref`, and the full typed `commit`. The digest hashes
canonical `EpisodeCommit` bytes. Require matching ordinal/evidence identity,
owned entries exactly covering one authenticated unit episode group, and
separately authenticated borrowed references. Store bindings in a bounded
authenticated paged index with cold publication, not permanent per-episode
files. Provide `register_episode_binding` as the producer/session seam: Task4
calls it only after row fsync and seal, before eviction. The binding embeds its
commit, so no filename inference or separate permanent commit filename is needed.
Registration may take explicit transport/policy and runs only in the parent
service; the producer sends typed registration through the existing `seal`
operation. Scientific child code never invokes the network transport.

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
`archive/supervisor.py` (closed producer dispatch wiring only),
`archive/transport.py` (exact retained-file custody during partial-unit eviction),
`archive/catalog.py` (targeted checkpoint-snapshot proof semantics),
`archive/session.py` (lease-scoped authenticated inventory metadata only),
`archive/bundles.py` (bounded singleton restore of non-episode owners),
the exact source inventory in `train/pilot_evidence.py`, and
`tests/pilot/test_pilot_source.py`. Extend
`tests/pilot/{test_neural_crashes,test_pilot_artifacts,test_timed_runner}.py`.
The archive fixture may add an explicit opt-in scratch-preservation flag paired
with its existing scratch-root override; keep default cleanup unchanged. Use the
flag during this plan's covering checks and charge retained fixtures to the same
workspace budget. Do not replace the fixture with an external cleanup monkeypatch.
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
                          policy: ArchivePolicy, session: LocalArchiveSession,
                          identity: UnitIdentity) -> tuple[UnitRef, ...]: ...
```

The journal helper uses the existing authenticated session to request parent-side
sealing under its ledger/locks. Supply source/run identity explicitly; do not
create another session, identity side file, or direct child `seal_unit` path.

Extend `evict_unit` narrowly with `retained: tuple[FileEntry, ...] = ()`, also
accepted by the closed archive request. This is an explicit custody snapshot of
non-evicted siblings derived from producer-authoritative controls, row log,
shared inputs, other episode commitments and authenticated catalog ownership.
Never assign ownership by scanning a directory. Require sorted unique safe paths,
no overlap with owned paths, matching logical scope, exact sizes/hashes and full
resident coverage by owned plus retained entries. Unknown, missing or changed
members still stop eviction. Bind retained custody in the durable intent and
recheck on interrupted continuation. Unlink only unchanged owned records; retain
all receipt/catalog/lock/borrower checks. Cover sibling episodes, active controls,
shared weights, forged/changed retained entries and interrupted eviction. Journal
segments sharing the run root use the same custody rule. Existing empty-retained
calls keep their strict behavior; no whole-directory deletion is permitted.

For stopped custody spanning an existing owner, finish that owner's lifecycle
using its complete authenticated inventory, not just the candidate intersection.
Expose the already-validated response metadata root as immutable
`EvidenceLease.metadata_root`, usable only inside the existing lease. Exhaust
the complete inventory there via an existing sparse path lease; do not hydrate
whole episode packs. Release the lease before eviction and suppress later
candidate groups already covered by the completed owner. No new protocol or
manifest schema is required. Cover split-owner boundaries, corrupted inventory
tails and existing lease lifetime/ref checks.

Path lookup keeps whole authenticated episode groups, but selects a singleton
for non-episode owners. The existing restore selector accepts exactly one
authenticated path for evaluation_metadata, journal, control_snapshot and partial
units, bounded by the existing policy.episode_bytes source-file ceiling. Exhaust
full inventory membership before outputs; preserve kind-specific unit bounds,
selected-plus-chunk cache checks and held remaining admission. Diagnostic subsets
retain their separate page_entries/logs_bytes limits; episode subsets must still
equal one declared group. Cover a partly cold owner whose complete payload exceeds
cache while its selected file fits, and reject missing/multiple/oversized selections.
No new schema, size constant, protocol operation or parallel restore implementation.

Before evicting finalized shared inputs, complete the supervisor's cold-borrowed
lease path. Authenticate exact shared-owner entries through bounded sparse
leases, hold their locks for the borrowing episode's lifetime, and verify/copy
from their actual lease roots rather than assuming resident run files. Expose
one complete authenticated episode root even when only its shared input is cold.
Charge source cache, copied shared bytes and atomic overhead simultaneously.
Preserve the one-public-episode limit, reject corrupt/missing owners and cycles,
and never recurse through episode packs as shared-input owners. Test cold/cold
and resident/cold combinations, live-borrower eviction refusal and safe release.

Align targeted catalog verification with checkpoint-snapshot publication:
authenticate the run root, exact unit membership/manifest and committed checkpoint
identity, and exhaust the complete snapshot inventory/shards, but do not demand
exclusive file/directory ownership for `control_snapshot`. Such snapshots version
mutable source controls and deliberately have no active path ownership. All other
kinds retain their ownership proofs; snapshot source eviction remains forbidden.
Test real archive/readback of two versions, corrupt final shards/proofs, invalid
checkpoint binding and unchanged rejection of non-snapshot ownership mismatches.

Before-work admission reuses the supervisor's held source-bound job reservation;
it must not reserve the same capacity twice. Add one strict `status` payload
variant for operation-specific, source-derived additional category maxima. The
parent validates the active admission, current allocated growth, remaining
allowance and protected physical headroom before acknowledging. If the full
next operation plus required publication overhead cannot fit, drain verified
pending packs or stop before neural work. Plain healthy status is not an admission
check. Keep the empty health request unchanged; no new protocol operation or
caller-granted capacity is introduced.

For journal units, bind canonical `JournalSegmentCommit` bytes through the
existing `UnitIdentity.evidence_identity_sha256`. Its fields are `schema_version`
(literal `phase4-journal-segment-commit-v1`), nullable SHA-256 `base_head`, SHA-256
`sealed_head` and `record_sha256s` in newest-to-oldest chain order. Derive the
descriptor from exact bounded immutable journal records, verifying filenames,
hashes, unique predecessor coverage, first record equal to `sealed_head`, final
prior equal to `base_head` and manifest file count equal to record count. Reject
gaps, cycles and extra members. The manifest authenticates original owned files;
the descriptor is reconstructible, so no extra scientific file or permanent
locator is needed. Preserve the existing canonical unit schema and old hashes.
Task5 verifies the same descriptor on read. A cold journal uses an authenticated
path lease and the existing 64 MiB pilot reader within that lease; the 16 MiB
metadata `read_record` limit remains unchanged.

Use the shared `EpisodeCommit` introduced by Task3 in `archive/types.py`; validate exact scalar types, sorted
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
  Supply a bounded immutable prior-writer/checkpoint custody handoff through the
  existing `pilot_ownership` context while its permanent exclusive lock is held.
  Validate run identity, prior owner/session status and PID creation time; denied
  inspection is not proof of death. Complete the existing durable-checkpoint
  recovery first. Do not steal, remove or reacquire a live owner's lock. Only the
  authenticated handoff permits producer partial sealing, not a supplied stopped
  boolean alone. Capturing exact safe files under a stopped attempt establishes
  transport custody, not episode ownership or a claim about unknown creators.
  Preserve pre-envelope/orphan crashes as partial bytes without inferred outcomes,
  rows, episode commits or DONE. Keep checkpoints/controls/journal authorities
  separate; unclassifiable outside-attempt files remain retained and blocking.
  Reject links, special files and escaped paths, and keep custody metadata bounded.
  The parent seal-partial variant partitions candidate custody against authenticated
  active and pending-sealed owners. Validate exact candidate hashes first; only
  authenticated owner absence permits new partial ownership. Return exhaustive,
  disjoint new and retained-owned FileEntry groups with authenticated UnitRefs and
  a rederivable new-proof digest. The child verifies the partition and inventories.
  All-owned candidates create no new partial unit/proof or deletion authority;
  existing owners may finish their already-authorized pending lifecycle.
  Preserve an interrupted episode-pending control as exact content-addressed
  stopped-controls metadata before rotating its local slot, including all-owned
  cases. Require reserved upload, verified readback and an authenticated,
  discoverable durable operational reference before exact hash-checked removal.
  Neither a successful upload nor an unreachable blob is sufficient. Do not
  manufacture an EpisodeCommit, outcome, scientific artifact or protocol operation.
  The existing operational-batch helper may accept an optional all-or-none receipt
  pair: absent receipt means no receipt object/locator or active authorization.
  Preserve schema, accounting fixed point, staged verification, recovery and exact
  cleanup under archive_operation_lock. Cover fresh cold reservation lookup and
  interrupted reservation-only publication as well as existing receipt callers.
  Even a resume with no attempt payload must authenticate prior/current writer and
  durable checkpoint before exposing stopped_checkpoint or starting neural work.
  Use the same custody variant with attempt_root=null only when entries/paths are
  empty, logical_root=null and no pending-control hash is supplied. Return an empty
  partition without manifest, catalog, proof, control or eviction publication.
  A valid non-null attempt root may also have empty candidates but gains no new
  ownership. Set stopped_checkpoint only after every required handoff succeeds.
  Test no-attempt/no-payload resume, forged/stale authority, null scope with entries
  or control, changed pending-control bytes and upload/readback/publication failures.
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
Extend `archive/{session,supervisor}.py` for the remaining fixed
verify/report/replay reader dispatch and the closed recovery-source view/import
contract below, using the already-created local session. A narrow
`archive/producer.py` custody helper is allowed when required; callers must not
mutate the producer's private resident inventory.
Extend `tests/archive/{test_session,test_supervisor}.py` for these exact interfaces.
Extend `tests/pilot/{test_pilot_artifact_index,test_pilot_checks,test_pilot_report,test_pilot_gate_verifier}.py`.
Narrowly correct `tests/pilot/test_pilot_trainer.py` byte/path-guard journal
fixtures to a real-shaped update1/validation chain with a canonical attempt ID.
Preserve exact byte boundaries and symlink/path/hash negative assertions; do not
weaken unified journal validation to accommodate the old step0/noattempt fixture.

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

Cold readers discover the original operational commitment through
`EvidenceContext.episode_commit(logical_root, ordinal) -> EpisodeCommit`.
Use a strict read-only variant of the existing `status` operation and the existing
authenticated binding-tree key `(run_id, logical_root, ordinal)`. The parent
returns the original typed binding; the child verifies run/root/ordinal and
canonical commit digest before exposing its commit. Fetch metadata only within
existing page limits. Discovery does not establish scientific validity: the
scanner must compare identity, exact row offset/length/hash and ownership with
the original DONE/index, then request the existing digest-required episode lease.
Factor the authenticated lookup, but do not make actual lease digest checks
optional. The recovery-source view uses its pinned source binding head. Test
cold discovery, wrong root/run/ordinal/digest, corrupt binding leaf and changed
binding between discovery and lease. No new tree, schema, operation or scientific
file is needed; arbitrary caller authority remains forbidden.

**Fresh-recovery authority and bounded import:** operational run IDs derive from
absolute run paths; catalogs and remote chunks bind that identity. Never relocate
source catalog heads to a new run or weaken membership checks. Use one fixed
parent-bound read-only source view on the destination's existing IPC sequence,
then re-publish verified inputs into destination-native units/catalogs. This
duplicates remote input payloads: reserve/count their transfer, readback and
retention within existing quotas, with no budget increase. Scientific file bytes,
paths, EpisodeCommit digests, source/config identity and original gate stay exact;
only operational run IDs, UnitRefs, receipts and catalog bindings change.

Implement this closed interface, not a generic cross-run resolver:

```python
# Existing checks job gains a disjoint mode="recover" request containing
# source run/control, original artifact path and optional retained-local root.
# Destination stays run_dir; ordinary checks still require config_path.
session.recovery_source() -> EvidenceContext
session.import_original(lease, *, entries, commit=None, pin=False)
```

The source view shares the same request lock/sequence, parent liveness and lease
registry. Only the admitted recovery checks job can create it. A literal
`recovery_source` payload tag authorizes existing read operations only. The import
method is a strict existing `seal` variant consuming an active parent-issued
source lease token, never arbitrary source/destination paths, remote keys or
caller-supplied run identities. Parent derives exact import membership from the
pinned original gate's envelope/index/descriptor closure with bounded shard
verification. Merely supplying a matching path/hash is not import authority.

Before the offline child starts, bind the source run/control pair, catalog and
episode-binding heads, original gate digest, policy, destination identity and
request/source admission in existing bounded operational job data. Keep source
exclusion and per-unit reader locks; verify known inactive pilot-owner identity
without rewriting or taking over that owner. Reject nested/aliased/overlapping
run/control roots, changed heads/inodes/gate/controls and out-of-workspace paths.
Source cold units require pinned catalog membership, not destination pending-unit
fallback. Cache/metadata roots live under counted destination controls, while
source locks stay in source controls. Enforce one episode lease globally across
both views, including shared borrowed inputs. No second session/server, foreign
catalog membership, new unit/schema, or general context registry is introduced.

Recovery order is binding:

1. Reserve old/new/cache/staging/readback together. Operational control creation
   may precede preflight; scientific destination creation may not.
2. Exhaust the original gate/envelope/index and every required raw input with
   complete semantic, journal and checkpoint closure. A final-page failure leaves
   the scientific destination absent and executes no neural work.
3. Recheck source authority/absence, then enter existing destination
   `pilot_ownership` using authenticated scientific identity. Never copy old owner
   files or put the old gate at destination `phase4-gate.json` for output reuse.
4. Pin only admitted required controls with original hashes. Historical root
   controls use explicit authenticated snapshot refs/checkpoints, not fabricated
   exclusive ownership of `.`; other checkpoint/weight reads stay lease-scoped.
5. Import shared primitive owners before borrowers, then one episode/page/journal/
   metadata closure at a time through existing seal/archive/readback/receipt and
   safe eviction. Preserve group/borrowed distinctions and bind unchanged
   EpisodeCommits to new destination refs. Source plus destination copies coexist
   only for the admitted current item. Preserve failed imports as counted evidence.
6. Exhaust and authenticate the destination through ordinary context membership,
   write unchanged workflow/recovery-intent schemas, then run new checks with the
   destination producer/context. Closing the source view must not break subsequent
   destination reads. Ordinary eager local recovery remains unchanged.

Test foreign refs, source-tag writes, unknown fields, stale/released tokens,
nonindexed imports, changed source heads/gate/controls, absent shared weights,
corrupt last shard, destination absence timing, no-neural preflight, new native
destination identity, byte-identical originals, one global episode lease, reader
locks and combined peak allocation. Importing old final-output trees is not
authorized merely because original gate verification could read them.

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

**Prerequisite accounting/archive slice (before remaining Task5 work if needed):**
Preserve completed engineering history remotely or locally without treating it
as pre-existing baseline. Measured retained task/test output already exceeds the
scratch category, while mixed roots contain real debug/failure evidence alongside
synthetic inputs. A new operational directory may not make those bytes disappear
from the global10GiB contract. No mounts, filesystem codec or general quota
framework is added.

**Measured inventory-lifecycle correction (2026-09-18):** the first prelude's
linked inventory pages are republished after each small diagnostic unit. Removing
one record shifts later page boundaries and hashes; retaining those generations
can use more local space than the payload eviction frees. The read-only grouping
audit found 798 positively classified files in 369 initial candidate groups,
about 2.56 GB allocated payload versus approximately 4.81 GB of six-root suffix
page churn if those groups were processed separately. These are prospective
measurements, not authenticated custody or a final archival schedule.

Correct the private retained-inventory representation before real bootstrap:
use immutable leaves with stable original page partitions and a bounded,
authenticated index. Residual publication changes only leaves containing exact
receipt-authorized removals or changed ancestor directory records, plus the
small index. Do not repack later surviving records across original leaf boundaries.
Preserve every previous leaf/index generation locally and charge it; this is
metadata reuse, not cleanup authority. Keep the existing single-unit pending
receipt, retained-sibling validation and atomic charge/reservation transition.
No multi-unit eviction transaction, generic registry or public remote-catalog
schema change is needed. Version the private representation; existing scientific
formats and source-bound historical artifacts remain unchanged. Legacy inventory
reading must be explicit and must never silently rebaseline a real authority.

Before implementation, specify the exact private leaf/index schema and finite
reader limits. RED/GREEN regressions must cover unchanged-leaf identity,
stable partitions through removals and empty leaves, exact full-residual
validation, corrupt/missing/reordered/duplicate index entries, interrupted
leaf/index/head publication, and restart with unchanged old charges. Derive
admission from the exact affected leaves and bounded index publication, including
old/new/atomic coexistence. A small many-leaf, repeated-eviction test must observe
actual publication peaks and all historical generations remaining present.
Recompute prospective full-workspace reclamation before granting real admission;
do not rely on the six-root estimate or simply assume the new representation fits.
Scope remains `archive/preflight.py`, its focused tests and source inventories;
a narrowly extracted private inventory module is allowed only if it reduces
that file's responsibilities and is included in exact source closure.

Extend `archive/{ledger,types,preflight}.py` and focused tests with page-bounded
authenticated frozen external-root charges in the same durable global ledger.
This is an accounting line, not an independent allowance or a larger existing
category. Preserve existing ledger identity, baseline, local/remote reservations
and history through a compatible versioned transition; if no real ledger exists,
bootstrap it once with old newoutput charges already present. Fixture ledgers
are test data, not independent production allowances. Count all roots, live
reservations, directories/control overhead and protected reserve; reject aliases,
overlap, changed frozen members or caller-authoritative byte totals. Keep locators
private and inspect links/sparse/special entries only for no-follow accounting,
never archive permission.

Prepare exact candidate inventories for completed old task roots only. Prefer
safe ordinary single-link dense file subtrees, leaving unsupported candidates
local and charged. Reuse existing bounded diagnostic sealing, upload/readback,
receipt and safe eviction primitives; no replacement custody codec or deletion
algorithm. Split large safe trees only through disjoint child roots or exact
retained-sibling ownership. Preserve malformed fixture bytes as opaque engineering
data, not valid scientific evidence. Do not modify original Phase4 evidence,
active fixtures, review reports or another plan workspace.

One unit's source allocation, sealed chunks, upload/readback buffers, exact
metadata/catalog growth and bounded logs must fit simultaneous global admission.
An external-root charge may decrease only after exact receipt-authorized eviction
and verified residual accounting; interrupted operations retain their charge and
resume without budget reset. Regressions cover authority/root changes, global
double spend, unsupported files, retained siblings, every failure cut point and
source+staging peak. Implementation/review use local transport doubles first.

When this slice is independently reviewed, Task7's bounded real-provider
qualification may precede the final full local gate. Its total512MiB allowance
still covers qualification plus later smoke. Only after successful descriptor,
create-only and readback qualification may eligible old engineering files be
archived and verified-before-evicted. Engineering retention is separately labeled
under the same4.5TB remote history, not hidden in qualification or a new budget.
Commit compact custody/accounting evidence, then resume the remaining Task5 work.
The full source-bound local gate, final storage gate and scientific pilot gates
remain mandatory after all implementation changes. A prelude failure leaves
evidence intact and blocks unadmitted growth; it never permits a larger quota.

**Early qualification execution refinement (2026-09-18):** implement and review
the narrow private `archive/_qualification.py` driver before the first real
engineering bootstrap, adding it only to the current executing source inventory.
Use `tests/archive/test_qualification.py` for local-double helper/process tests.
Extract only the common pure unit/catalog/receipt sizing core from
`preflight._prelude_bounds` when needed by the two qualification units. Its
private frozen result carries exact finite byte/count terms; engineering ledger,
retained-inventory, eviction and allocation-rounding work stays in the existing
caller. Include actual run-identity widths and episode groups. Focused tests must
preserve or conservatively increase engineering maxima; do not duplicate the
whole estimator or introduce a general sizing framework.
Do not construct the archive supervisor for this diagnostic: its persistent
run-directory binding would conflict with the later pilot. Do not add a public
API, CLI, provider registry, new configuration reader or new allowance.

The controller entry accepts the established workspace budget, bounded output
root, an exactly reviewed completed debug EpisodeCommit and source root, review
and protocol hashes, source commit and the fixed R2 transport. It derives the
shared control directory and policy from the existing budget. Its only new
archive compositions are a canonical locked remote-reservation snapshot and
receipt-authorized one-chunk-at-a-time restore. The lock hierarchy and remote
ledger must already be admitted and created; a snapshot may not create them.

Keep the prescribed 64 MiB probe and real copied episode. Reserve all local
sources, restores, interrupted scratch, metadata, atomic peaks and bounded logs
before publication. Use one parent-owned reservation across spawned archive
children, a real SIGKILL after the first successful provider create, an observed
provider collision on retry, exact readback, stable completed retry and fresh
no-overwrite restores. Stop at an additional 256 MiB remote qualification bound
inside the shared 512 MiB qualification-plus-smoke allowance. This stop limit is
not a serializer maximum or proof that the later smoke fits; prove that fit
separately against retained qualification usage. Reap every child before releasing
local planned capacity; all actual files and remote reservations remain charged.

Persist at most 32 bounded 4,096-byte phase summaries, a bounded progress record
and a 65,536-byte result; per-call evidence uses fixed-size counters, observed
category maxima and a rolling digest, not an unbounded event log. A success result
requires the real R2 transport; local doubles cannot issue it. Source/policy,
unit/receipt/inventory hashes, signal/collision/restore checks and local/remote
measurements are recorded without credentials, destination configuration or
machine paths. Small real files test process and restore behavior; the full fixed
probe recipe is also checked as a streamed digest without retaining duplicate
64 MiB test copies. Real provider execution still requires independent review,
positive content custody and exact shared-workspace admission.

For prospective full-verification bounds, distinguish source-bound fixture roles:
explicit acceptance/debug/failed scientific runs remain retained; mutable parser,
corruption and fake-transport test inputs and reconstructible tool state follow
their declared creator/lifetime. Passing a test is not cleanup authority. Derive
finite bounds at shared trajectory/capture/checkpoint/checkout/build publishers
using exact corpora, tensor layouts and invocation/fixture lifetimes, not observed
sizes or thousands of unrelated format maxima. Retain already-preserved roots.
Unknown output families fail admission. No arbitrary native-writer interception,
per-test cloud framework or network-enabled ordinary make target is introduced.

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
  `doctor` also validates the trusted Git >=2.45 prerequisite used by the offline
  subprocess boundary. Resolve supported installed Git without hardcoded machine
  paths; reject unsupported versions rather than weakening the boundary. Keep
  the executable pin private and report only non-sensitive capability evidence.
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
