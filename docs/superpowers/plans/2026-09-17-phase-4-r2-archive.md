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

#### Engineering native-allocation correction and exact batching (2026-09-18)

The actual same-campaign continuation on2750501 passed with typed result SHA256
`9c8e6ffa135a11ffd6bd74b26f5bbcc26659a2b8cc3237dd2e527703e6d4ab17`.
The earlier stdin-launched continuation failed at multiprocessing startup; its
separate output stays failed and retained. The successful attempt used a `-c`
launcher, repeated the provider checks, restored both inputs exactly, and passed
final authentication. This unlocks the engineering path, not final storage or
scientific acceptance. Never relabel this result as a later executor's result.

A read-only next-step audit found the same logical/native underbound in
`preflight._prelude_bounds`: generated payload and provider readback coexist,
but its pair term still uses twice the logical maximum. Correct this before
engineering transfer. This is a bounded maintenance change within the approved
storage plan; no transport, custody, public API, quota or science changes.

**Files:** modify `src/silent_cascade/archive/preflight.py` and
`tests/archive/test_preflight.py` only, plus this plan/design and operational
evidence. Keep the pure `_archive_unit_bounds` serializer unchanged.

- [ ] Add descriptor-only native-pair regression using the existing tiny
  `prepared_candidate` fixture, with a synthetic32MiB `FileEntry` passed only
  to the sizing function. At4KiB blocks the pair term must be at least134,225,920
  bytes, before unchanged catalog/directory additions. Do not write a32MiB fixture.
- [ ] Exercise the real engineering reservation with injected measured scratch
 199,651,328 and that computed bound: admission must refuse before yielding to
  sealing/provider work, preserve all source bytes, and leave no reservation.
  Also cover a14MiB descriptor and the existing two-successive-subset behavior.
- [ ] Observe RED before changing the implementation. Replace only the native
  pair calculation, retaining all existing metadata and stage accounting:

  ```python
  largest = max(
      min(expanded, policy.chunk_bytes),
      min(policy.page_bytes, max(manifest, receipt, run_catalog, operational_catalog)),
  )
  rounded = ((largest + block - 1) // block) * block
  readback = max(largest, policy.page_bytes)
  rounded_readback = ((readback + block - 1) // block) * block
  scratch = (2 * rounded + block) + (2 * rounded_readback + block)
  # Existing catalog staging and directory terms are added unchanged below.
  ```

- [ ] Run the new tests and existing allocation-geometry, actual atomic-peak,
  and fresh-subset/shared-history tests under one separately admitted finite
  slice, with unique retained roots, no bytecode/cache and recorded owner IDs.
  Size the slice before running; preserve RED/GREEN evidence and scoped quality
  checks. Complete independent task review and a meaningful current-branch commit.
- [ ] Publish the new exact reviewed-source authority. Bind prior qualification
  as historical provider evidence, not qualification on the changed source.
  The change only increases engineering admission; all final current-source
  local/resource/smoke gates remain mandatory after implementation is complete.
- [ ] Use existing `EngineeringContentReview.paths` for batches of at most14MiB
  logical data; this ceiling is not admission. Start with the largest capture
  pair (13,488,540bytes), then the other three inseparable tensor/sidecar pairs.
  Put the17portable weights into four groups of four and one singleton. Never
  split a capture pair, synthesize a candidate, or include unreviewed siblings.
- [ ] Before each batch, rediscover the authenticated current candidate, check
  exact paths/hashes against the positive content audit, remeasure allocation,
  and recompute complete native/catalog/metadata/global/physical/remote bounds.
  Candidate identity changes after eviction. Refuse if the indivisible pair
  cannot fit; reduce weight batches when needed, never increase a limit.
  Upload/readback/receipt-authorized eviction and residual authentication remain
  unchanged. Report exactly which local bytes were archived and are recoverable.

The104,817,273-byte group cannot be transferred as one unit: its corrected pair
term plus existing scratch alone exceeds256MiB. Exact batching uses already
reviewed subset/retained-sibling semantics and costs additional catalog work;
it does not reduce evidence retention or create another allowance.

**Review correction (2026-09-18):** commit5ce7b4a implemented the original
same-sized native pair and passed its six scoped tests, but that term is not a
complete bound for small units. While the generated payload remains live,
`_reserve_object` can consult the cold operational root, whose reader accepts
up to `policy.page_bytes`, independently of the current unit's smaller catalog
size. The formula above therefore reserves the larger of same-object readback
and the complete allowed cold-root read. Keep reader limits, native allowances,
catalog/directory stages and all custody checks unchanged. Add descriptor-only
14MiB and small-object regressions plus a real reservation refusal where the
old pair would fit; observe RED and then GREEN under the existing finite test
grant before provider work. The32MiB pair stays134,225,920bytes, while the14MiB
pair is62,922,752bytes at4KiB blocks, before existing stage terms.

Ruling: strengthen the approved bound rather than shrink a reader cap or bypass
admission. The cost is fewer bytes available to a batch, not weakened retention,
science, or a larger quota. The earlier passing test run and actual qualification
retain their original source bindings and are not relabeled.

Each implementation slice gets RED/GREEN evidence, bounded local covering tests,
an independent task review and a meaningful current-branch commit. The scoped
briefs give exact tests/interfaces and finite output admission before dispatch.
No full `make verify`, provider call or later pilot is implied by a unit-test
pass. Full final gates remain unchanged. Do not bundle unrelated diagnostic
improvements into the deterministic validator correction.

#### Targeted canonical engineering discovery (2026-09-18)

Actual first-batch discovery onb3a26ab reached only70eligible groups after
693.5seconds, before any provider call, admission or eviction. The recorded
process was active; interrupting the read-only operation showed repeated
`_stored_records` scans inside recursive `candidates`. A separate one-pass
diagnostic counted156,559retained records,62,000directories and2,102eligible
groups; the audited runs subtree has619members. These diagnostic counts are not
candidate, content-review, or admission authority.

Ruling: add an optional exact `logical_root` selector to existing
`iter_engineering_candidates`, and use it during `_validated_review`. The
selector only prunes unrelated traversal; it must return exactly the candidate
that default enumeration would return for that root, including all members and
the identical identity. Authenticate the full retained history, committed source
and reviewed-source authority as before. Check every ancestor's eligibility:
if an ancestor is already a canonical candidate, a requested descendant is not
a candidate. Reject unsafe paths and roots outside the existing closed `_TASKS`
namespace. Do not synthesize candidates or expose arbitrary filesystem roots.

Modify only `archive/preflight.py` and focused `tests/archive/test_preflight.py`,
plus this plan/design and operational evidence. Keep default enumeration,
identity hashes, content reviews, bounds, reservations, sibling preservation,
custody and transport semantics unchanged. No index, page cache, new persisted
state, broader authority, quota or scientific changes. This costs a small
optional API seam but avoids enumerating thousands of unrelated groups twice
per batch.

- [ ] Tiny-fixture RED tests compare targeted/default candidates, reject unsafe
  or noncanonical descendant roots, preserve rejection of corrupted unrelated
  history, and prove bounded ancestor-only traversal by counters, not timing.
- [ ] Minimal selector implementation and `_validated_review` caller threading;
  run targeted GREEN plus existing stale/forged review and sequential-subset
  coverage under one separately admitted finite test slice. Preserve all outputs.
- [ ] Independent scoped review, current-branch commit and new exact source
  authority; earlier provider qualification and interrupted discovery keep their
  original source/status. Resume the same exact batch using the real filtered
  iterator, not a diagnostic reconstruction. Full final gates remain mandatory.

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

### Task5 continuation slice ordering after verified reclamation

The completed scanner/index/journal foundation remains unchanged. Start the
remaining integration with cold evaluation-root discovery, authenticated
committed/abandoned journal status and zero-complete-row partial reporting.
Thread the existing evidence context through those report decisions; all reads
stay inside existing leases, with no new session operation or catalog schema.
Preserve eager behavior and unknown outcomes. New cold handling must reject
newline-complete rows until their full row/crash semantics are implemented in
the following slice; this limitation is not Task5 completion and cannot support
a passed gate. Do not reconstruct a whole evaluation tree or invoke neural code.

Scope the first slice to `archive/session.py`, `report/pilot_artifacts.py`,
`report/pilot.py`, focused archive-reader tests and, only if required, a compact
policy argument on the existing test producer fixture. A small canonical
identity fixture may be versioned from already retained real debug evidence;
do not generate neural weights merely to construct an identity. Tests cover
cold partial discovery/reporting with a trailing row fragment and orphan bytes,
abandoned predecessor corruption, complete-row refusal, committed coverage and
stopped-custody behavior. Use deterministic payloads and explicit producer/control
maxima to derive finite test admission, keeping remote-double files, directory
overhead and failed outputs inside the same production ledger. Historical fixture
sizes alone are not an admission proof. Get RED/GREEN, task review and a scoped
commit; all remaining original obligations continue below and in the Task5
integration brief, including complete cold rows, recovery and actual CPU equality.

The first local fixture invocation revealed that the retained engineering
authority correctly rejects a second nested synthetic workspace. The focused
test fixture may isolate only pre-existing ancestor authority directory identities
at the exact authority-read boundary, matching the established archive-test
technique. Fixture-created authorities remain visible and production checks are
unchanged. Reuse one helper across affected producer cases; do not disable the
single-authority rule or create an uncounted output location. Preserve this
setup failure separately from the later expected feature RED.

#### Next bounded slice: complete rows in abandoned cold evaluations

The first slice is implemented in `6c52d5d` and independently reviewed. Its
temporary rejection of complete cold rows is replaced only after the existing
row/trace/crash semantic checks work through authenticated reads. Add one narrow
optional byte-reader argument to `eval/artifacts.py::_verify_evidence`, retaining
the existing local default and all semantic validation. The cold partial reader
supplies a closure that checks exact inventory membership, leases one named file,
reads within existing limits and rechecks the original training digest at the
actual read. Return bytes, never deferred leased Paths. No new session operation,
EpisodeCommit, transport schema or whole-evaluation materialization is permitted.

Use the already retained real one-episode success and initialization-failure
artifacts identified in the operational preflight: payload totals 49,709 and
12,037 bytes including their shared identity. Version exact small fixtures with
decoded-byte hashes; a text encoding of existing gzip bytes is acceptable.
Do not regenerate weights, fabricate outcomes or alter original scientific bytes.
Compare eager/cold partial summaries for both cases; check retained row/error
counts, unknown unwritten outcomes, sequential leases, no invented commitment,
and rejection when evidence changes between initial inventory verification and
the semantic reread. Preserve zero-row coverage and local defaults.

Scope adds the existing `_verify_evidence` helper in `eval/artifacts.py`, its
caller in `report/pilot_artifacts.py`, focused reader tests and small fixtures.
Factor test setup locally only when needed for the repeated real producer case;
do not redesign the test harness. Use a separate finite same-ledger admission,
unique roots below the existing `operational/scratch/` binding, TDD, scoped local
checks and independent review. Broader Task5 caller/gate/recovery/CPU obligations
remain required after this slice.

#### Next bounded correction: incomplete verification cannot pass

The complete-row slice is implemented in `fb3e051` and independently reviewed.
The next existing Task5 obligation is the current gate-verification status:
`verify_phase4_gate_artifact` excludes missing raw attachments from `passed`,
but does not exclude separately recorded unavailable semantic checks. Numeric,
offline and continuation validators deliberately retain recorded comparisons
when some raw evidence is absent; these recorded facts must not imply a current
complete verification pass.

Scope: `train/pilot_evidence.py` and focused
`tests/pilot/test_pilot_gate_verifier.py` coverage. Extract only the final status
projection into a small private helper used by the real verifier, initially
preserving behavior; cover the current defect with TDD, then require:

```python
passed = outcome == "passed" and not missing and not unavailable
```

Preserve `recorded_outcome`, missing paths, sorted unique unavailable check names,
`valid=True` for internally consistent recorded evidence,
`verification_scope="recorded_evidence_integrity"`, and
`neural_replay="not_rerun"`. Do not change acceptance thresholds, artifact
schemas, recorded comparisons, validators, source identity or gate hashes.
The helper is the production status projection, not a test-only hook or a mock
full-gate success. Do not short-circuit semantic validators on unavailable data.

- [ ] Test the production status projection for each recorded outcome (`passed`,
  `failed`, `debug_non_acceptance`) crossed with missing and unavailable evidence;
  only a recorded pass with both collections empty can currently pass. Preserve
  labels and normalize unavailable names without mutating inputs.
- [ ] Show expected RED when a recorded pass has no missing attachment but has
  an unavailable semantic check; add the minimal predicate correction and GREEN.
- [ ] Exercise the real offline-evidence validator with an explicitly untrusted
  schema-valid unit declaration and no raw evidence; its returned historical
  comparison may remain true but its unavailable checks prevent status passing.
  With missing executed-source evidence and a present corrupt raw step, require
  rejection rather than allowing unavailability to hide independent corruption.
- [ ] Run existing inexpensive acceptance and unsupported-evidence tests, with
  neural execution tripwires where relevant; no costly neural fixture, numerical
  run, provider call or full-gate completion claim belongs to this unit correction.
- [ ] Use unique retained roots under `operational/scratch/`, finite same-ledger
  admission, scoped lint/format checks, a meaningful commit and independent review.

Ruling: keep this conclusion-policy correction independent of broader cold
reader threading. It closes a concrete fail-open status defect without changing
scientific evidence or claiming that a full cold gate has been verified. The
broader integration remains required and will supply end-to-end coverage.

#### Checkpoint-reader prerequisite before full cold authentication

Static integration review found that `_durable` accepts `evidence_context` but
still reads checkpoint-index.json and the full checkpoint by raw local paths;
only the subsequent journal check receives the context. Correct this narrow
prerequisite in `train/pilot_workflow.py::_durable`, with a focused new
`tests/pilot/test_pilot_cold_checkpoint.py`. Use the existing scoped reader:

```python
from silent_cascade.archive.readers import evidence_path

with evidence_path(run_dir, "checkpoint-index.json", evidence_context=evidence_context) as local:
    index = read_json(local)
# Parse the existing descriptor and keep path = child(run_dir, descriptor.path).
with evidence_path(run_dir, descriptor.path, evidence_context=evidence_context) as local:
    session = load_pilot_checkpoint(
        local, expected_sha256=descriptor.sha256, config=config, source=source, device=device
    )
# Existing journal, progress and model-identity checks remain after both leases close.
# Return the original logical path, never the released local cache path.
```

- [ ] Use an actual initialized debug EventFlowModel, its real empty AdamW state,
  `PilotProgress()` and real save/load checkpoint functions. The initial running
  zero-step state is legal and is produced by the normal trainer. Supply a
  clearly labeled unit expected-source identity, as existing checkpoint unit
  tests do; source-history authentication belongs to the later full-run test.
  No metrics, completed training envelope, terminal status or successful fit may
  be invented. This tests the checkpoint consumer, not provider publication.
- [ ] Publish a genuine checkpoint and index using existing checkpoint helpers.
  First exercise local `_durable`, then move this newly created unit fixture's
  index/checkpoint to a strict context-backed directory, preserving the bytes.
  Exercise cold `_durable`; require equal logical return paths, exact restored
  model identity/tensors, one active lease maximum and zero leases on return.
  Never move or modify any previously retained workspace artifact.
- [ ] Obtain RED on the current missing local index/checkpoint path before the
  scoped-read change. Test bad checkpoint digest and descriptor step/model
  mismatch independently; original errors must survive and all leases close.
  Do not stub checkpoint decoding, tensor reconstruction, journal verification
  or semantic success. A strict path/lease double is allowed at this unit layer.
- [ ] Preserve local defaults, return type, journal validation and all existing
  comparisons. Do not yet change authenticate_run, full trainer resume, aggregate
  collectors, supervisor dispatch or recovery authority. Those remain required.
- [ ] Bound actual checkpoint publication before writing: a test-owned guard
  checks at most4MiB and delegates to the real publication function. Historical
  same-profile zero-step archive size was3146426bytes; this observation motivates
  the guard but is not a substitute for it. No checkpoint/model fixture is
  committed; all new outputs remain in unique admitted scratch roots.
- [ ] Before each command derive its fixed file/payload/directory allowance,
  retain prior RED/GREEN output, and require measured cumulative growth plus
  that command's allowance plus administrative headroom to fit the existing
  48MiB shared status/authentication grant. No full four-update fixture is admitted
  by this subsection. Run scoped local checks and independent review, then commit.

Ruling: zero-step is sufficient only for this real checkpoint-read boundary.
It cannot substitute for completed-training authentication/reuse or later exact
CPU next-batch/resume/recovery gates. Keeping the primitive test separate avoids
fabricating terminal status to fit storage; the cost is later end-to-end work,
which remains explicit and mandatory.

#### Completed-run authentication and workflow reuse

The checkpoint primitive is implemented in `9f5047c` and independently reviewed.
Continue the existing Task5 reader contract in `train/pilot_evidence.py` and
`train/pilot_workflow.py`; add focused
`tests/pilot/test_pilot_cold_authentication.py` and a test-only helper if the
source-authenticated subprocess/bounds need separation. This is not the full
gate, recovery authority or production fit.

- [ ] Add `evidence_context=None` to `authenticate_run`. Forward it to
  `load_training_result` and `_durable`. Decode checkpoint metadata, load the
  real CPU checkpoint and load portable weights inside separate `evidence_path`
  leases. Preserve source/data authentication, RNG restoration, every identity
  comparison and eligibility rule. Return the original logical archive path;
  no released cache path may escape.

  ```python
  with evidence_path(run_dir, descriptor.path, evidence_context=evidence_context) as local:
      metadata, _ = decode_archive(
          read_bytes(local, limit=1024**3), descriptor.sha256, PilotCheckpoint, "pilot_training"
      )
  # Authenticate source/data exactly as before, then close each later tensor
  # input lease before acquiring another. The return remains archive_path.
  ```

- [ ] In `_train`, use the context's validated inventory for cold completed-root
  and `attempt-<32 lowercase hex>/result.json` discovery, unioned with legitimate
  resident candidates. Validate context/root agreement before discovery. Keep
  eager behavior when context is absent, and retain existing result/last-durable
  checks and resume-path rules. Forward context to every `load_training_result`
  and `_durable` call. A completed archived fit must return without invoking
  `run_pilot_training`; an interrupted root-result publication may be restored
  from its actual completed attempt result. No synthetic completion is allowed.
- [ ] Forward the already available workflow context to `build_pilot_report`.
  Other gate/continuation/numeric/offline callers remain the next Task5 grouping;
  do not claim they are complete or widen this change into recovery authority.
- [ ] Construct one genuine CPU `phase4_smoke` fit per RED/GREEN fixture using
  the current exact copied source and its real three-commit data-introduction
  history. Run all four updates and both 16-episode validations unchanged. Compare
  eager authentication with cold authentication and completed `_train` reuse:
  source, progress, descriptors, actual tensors, logical path and lease cleanup.
  A neural execution tripwire must catch accidental retraining during reuse.
- [ ] A strict test path context may move only newly generated fixture inputs
  into a backing directory and expose one requested file by rename at a time.
  Leave real journal records resident: authenticated cold journal segments have
  separate coverage. Checkpoint/index/weights, result/index shards and inventory
  evidence must actually be absent at their original cold paths. This is a
  reader-integration test, not provider publication, IPC or recovery evidence.
- [ ] Cover cold attempt-result discovery, wrong logical root, late inventory
  corruption, checkpoint/portable digest failures and released-lease behavior.
  Preserve genuine fixture bytes; use bounded altered control copies or return
  a deliberately wrong test input through the strict context. Never modify old
  retained artifacts, expected hashes to hide corruption, or source identities.
- [ ] Run RED before production edits. Run GREEN against a new immutable source
  checkout. For fresh controller verification, reread the same completed GREEN
  fixture using its own authenticated source rather than performing another fit
  or relabeling evidence to a newer revision. Keep all failed prefixes and logs.
  Independently review the scoped code and test-bound implementation.
- [ ] Before any checkout/test/model execution, derive and record exact root,
  file/count/payload/directory/atomic-coexistence bounds. Use the same global
  ledger: new scientific debug artifacts in spool, leased materializations in
  cache, source/Git/data/test administration in scratch, controls in metadata,
  diagnostics in logs. No nested ledger, rebinding, quota increase or old-root
  reclassification is permitted. Guard every test publication before its real
  writer, including row appends and failure output; exceedance fails the fixture
  before writing and never counts as a completed fit. Observed output sizes are
  not admission bounds. Admit RED and GREEN individually and retain both.

Ruling: a real completed smoke fixture is necessary here, unlike the initial
checkpoint prerequisite. Separate its scientific output from checkout scratch
from inception, under the existing categories and global allowance. A bounded
strict path context isolates consumer lifetime/verification behavior without
introducing another provider ledger. Full real archival/recovery and local gates
remain mandatory; the cost of this narrower test is that it cannot certify them.

#### Artifact-only continuation and offline semantic readers

Completed-run authentication/reuse is implemented in `e46e2b7` and reviewed.
Continue Task5 with `verify_continuation_records` and `verify_offline_evidence`
in `train/pilot_evidence.py`; add `tests/pilot/test_pilot_cold_semantics.py`
and one adjacent fixture helper if necessary. A narrowly scoped optional
materialized shared-weight input in `eventflow/neural_checkpoint.py` is allowed
only if needed to preserve the existing referenced-weight validation while
releasing each cold file before the next. Do not change runtime execution.

- [ ] Add optional `evidence_context=None`; reject a context/root mismatch
  before input reads. Discover expected members from the fully consumed,
  authenticated inventory plus resident files. Retain only relevant names.
  Missing unindexed members retain the existing unavailable labels. Advertised
  members with corrupt bytes, failed leases or invalid semantics raise errors;
  unavailable evidence must never hide independently available corruption.
- [ ] Consume continuation checkpoints and replay JSON within scoped reads;
  preserve all identity, trace, boundary, coverage and comparison checks.
  Materialize data, not released cache paths. Shared-weight dependencies remain
  validated against their recorded hash and metadata without hydrating a tree.
- [ ] Consume offline source, update, replay, weights and report files through
  `evidence_path`. Forward the context to the existing `load_evaluation` scanner,
  preserving its metadata and one-episode lease contract and all comparisons.

  ```python
  with evidence_path(run_dir, name, evidence_context=evidence_context) as local:
      raw = read_bytes(local, limit=MAX_BYTES)
  # Apply the existing semantic and identity comparisons to raw.
  # Absence is determined before leasing; lease errors are not absence.
  ```

- [ ] RED: compare eager and cold results from genuine retained continuation
  and offline records, preserving their original source/configuration and
  recorded `debug_non_acceptance`. Use one matching compact primary row for
  continuation. Explicitly show unavailable lists agree, not just booleans.
  Before production edits, fail on missing cold discovery/read support rather
  than merely on an unsupported keyword. Never run training, model forward,
  event execution or replay execution; install tripwires for those boundaries.
- [ ] GREEN: cover absent unindexed inputs; advertised missing/corrupt inputs;
  missing-one/corrupt-another; wrong logical root; late inventory failure;
  shared-weight dependencies; and final offline episode corruption. A strict
  context must reject reads of undeclared backing paths outside their file or
  episode lease and prove cleanup on success and failure.
- [ ] Borrow original frozen fixtures read-only. Never move, modify or delete
  them, rewrite identities, create a provider session, or copy the complete
  evidence tree. Only bounded isolated corrupted controls and test metadata may
  be written under new operational scratch. Establish their file/count/byte/
  directory bounds before execution under the same global ledger and local cap.
- [ ] Run scoped local lint/format and tests, obtain independent review and
  commit. Controller performs a fresh admitted artifact-only check. Numeric
  semantics, aggregate/check caller integration, recovery, real provider tests
  and full `make verify` remain separate required work, not implied by this slice.

Ruling: split numerical readers from continuation/offline because they have an
independently rejectable validation surface. This avoids another training fit
and keeps new writes small; the cost is no claim of complete gate validation.
Historical bytes test semantic-reader equivalence, not present-source scientific
execution or provider transport.

#### Artifact-only numerical semantic readers

After continuation/offline review, finish the numerical reader dependency group
before aggregate callers. Modify `train/pilot_verification.py` and
`train/pilot_evidence.py::verify_numeric_evidence`. Add focused
`tests/pilot/test_pilot_cold_numerics.py` and a small adjacent fixture adapter;
reuse the reviewed strict read-only fixture machinery where applicable, without
broadening production transport. Do not change numerical execution or tolerances.

- [ ] Thread optional `evidence_context=None` through `read_numeric_report`,
  `_check_artifact_closure`, `_runtime_artifact_inventory`, `_runtime_rows`,
  `_read_capture_operations`, `_read_capture`, `_bind_resume_start`,
  `_read_resume`, `_read_checked_archive` and `verify_numeric_evidence`.
  Local weight/checkpoint/JSON/tensor codecs retain their existing APIs.
- [ ] Add narrow private path/bytes/JSON adapters in pilot_verification. Derive
  names from the context run root, reject escapes before reading, and preserve
  each existing byte limit. Fully exhaust authenticated discovery, retaining
  only names under the required numerical prefix; union resident files, reject
  extras/symlinks and never turn an advertised lease/integrity failure into
  unavailable evidence. Share one discovery result within a top-level check
  rather than repeatedly scanning the full training corpus for every input.

  ```python
  @contextmanager
  def _numeric_path(path, *, evidence_context=None):
      if evidence_context is None:
          yield path
      else:
          name = logical_root(path, evidence_context)
          with evidence_path(evidence_context.run_dir, name,
                             evidence_context=evidence_context) as local:
              yield local
  ```

- [ ] Read report/DONE separately; input checkpoint and portable weights
  sequentially. Retain decoded session/digest/metadata, not released paths.
  Read manifest/subset/batch records individually. Decode capture tensors inside
  one lease and operations JSON inside the next; preserve all inventory, dtype,
  finite-value exceptions, objective and compute comparisons.
- [ ] `_read_resume` uses `_read_checked_archive`'s digest and reads observations,
  captures, optional bootstrap and summary sequentially. `_bind_resume_start`
  forwards context to bootstrap capture reads without changing its existing
  disposable CPU optimizer/clipping arithmetic, RNG checks or tolerances.
- [ ] Use real evaluation header/episode scanning for runtime inventory and
  row projections. Preserve the additional diagnostic identity, required-file
  set, crash-public-ID and canonical checkpoint-name checks. Read extra crash
  metadata in separate leases, never while holding another episode payload.
  Project already-verified sidecars without reopening released paths.
- [ ] Keep complete and partial verifier branches and every unavailable label.
  Missing unindexed capture tensors must not hide corrupt available operations
  JSON. Preserve `device_checks_passed` and missing-device results unchanged.
- [ ] RED/GREEN on the original frozen smoke `final/numerics` tree: compare
  complete eager/cold report and verifier outputs, explicitly preserving missing
  MPS and false comparison status; compare runtime-cpu rows directly; test one
  missing capture tensor, independently corrupt operations, final runtime
  corruption, substituted DONE binding, extra inventory and wrong root. Assert
  exact lease cleanup and one payload at a time under actual read guards.
- [ ] Borrow the original 83681280B tree read-only. Never copy its tensor tree,
  alter source identity, move retained files or run new diagnostics. Tripwire
  training, model forward, event and replay execution; existing artifact-only
  tensor comparisons, batch reconstruction and CPU optimizer arithmetic remain
  allowed. Admit exact small-control output bounds under the current shared
  ledger before each test stage. Preserve all failed prefixes.
- [ ] Scoped local checks, fresh controller verification, independent review
  and commit. Document limits: this CPU fixture does not establish MPS parity,
  empty-optimizer bootstrap, current-source diagnostic production, 64 production
  runtime episodes, complete gate, recovery or provider/transport integration.

Ruling: implement the actual numerical reader closure, not just context
forwarding in the outer gate. Genuine old artifacts are sufficient to test
artifact-only equivalence but never certify current-source execution. The
remaining full gate and recovery obligations stay open until independently run.

#### Artifact-only gate inputs and reuse admission

After numerical reader review, adapt the remaining independently testable gate
inputs before the complete source-bound aggregation/execution closure. This is
a bounded continuation of Task5, not approval for fresh diagnostics or training.

**Files:** modify `train/pilot_evidence.py::_verify_available_training` and
`compact_attachment_records`, and `train/pilot_checks.py` for one small reuse
admission helper used by `_run_pilot_checks_owned`. Add
`tests/pilot/test_pilot_cold_gate_inputs.py`; reuse strict historical borrowing
fixtures, with no generic archive framework or new scientific schemas.

**Interfaces:** optional `evidence_context=None` on both readers; consume
`evidence_path`, `logical_root` and existing context-aware `_durable`. Keep
existing decoded outputs, unavailable labels and RNG restoration. The small
reuse predicate consumes the existing verifier result and rejects missing raw
or semantic evidence, without promoting a failed result or requiring success.

- [ ] Write genuine eager/cold equivalence tests before reader changes. Read
  the original historical training result/config/source and compact attachment;
  do not call current-source authentication on historical bytes. Add cases for
  one missing dependency plus independently corrupt journal/checkpoint, wrong
  root, advertised lease failure, compact final count/hash, generator-close
  cleanup and export isolation. Use the actual CPU codecs and compact parser;
  tripwire training/model forward/engine/replay, never fake their outputs.

  ```python
  eager_unavailable, cold_unavailable = [], []
  _verify_available_training(backing, training=training, config=config,
                             source=source, unavailable=eager_unavailable)
  with cold.guarded_reads(monkeypatch):
      _verify_available_training(run, training=training, config=config,
                                 source=source, unavailable=cold_unavailable,
                                 evidence_context=cold)
  assert cold_unavailable == eager_unavailable == []
  assert cold.active == 0 and cold.payload_maximum == 1
  ```

- [ ] Obtain a fresh bounded test-prefix admission before imports/execution.
  Fail the real legacy cold behavior first, not just a new-keyword mismatch.
  Borrow large archives/weights unchanged; inventory exact control bytes and
  file/directory counts before writes. Preserve every test prefix. Neither
  historical measurements nor another task's reservation authorizes a new run.
- [ ] Validate exact run/context agreement before any training input reads.
  Fully exhaust discovery while retaining only descriptor paths, index and
  journal/dependency candidates. Resident files and authenticated membership
  both count as present; advertised read/integrity failures propagate.
  Decode each archive and weights file inside its own lease; preserve full
  source/model/eligibility/progress binding and RNG restoration.
- [ ] Read available journal links individually; check hashes/cycles even
  after another dependency is missing. Mark absent unindexed inputs with the
  existing unavailable labels. Run `_durable(..., evidence_context=context)`
  only when the checkpoint index, latest archive and full discovered journal
  dependency closure are available. Never treat a corrupt available dependency
  as absent or skip an independent check because another file is missing.
- [ ] Hold a compact shard lease through its hash check, parsed rows and final
  count check, closing on exhaustion, exception or explicit generator close.

  ```python
  for attachment in attachments:
      with evidence_path(root, attachment.path,
                         evidence_context=evidence_context) as local:
          if sha256_bytes(read_bytes(local, limit=MAX_BYTES)) != attachment.sha256:
              raise ValueError("compact attachment hash mismatch")
          count = 0
          for record in read_compact_rows(local):
              count += 1
              yield record
          if count != attachment.rows:
              raise ValueError("episode inventory mismatch")
  ```

- [ ] Introduce and use a small `_require_reusable_gate(verified)` predicate
  in the existing reuse branch. Cover all four missing/unavailable combinations
  without forging a complete gate. A verified `debug_non_acceptance` or failed
  recorded outcome remains reusable; this test is not full-gate evidence.

  ```python
  def _require_reusable_gate(verified):
      if (verified["missing_raw_attachments"]
              or verified["unavailable_semantic_checks"]):
          raise ValueError("missing raw or semantic evidence prevents compatible execution reuse")
  ```

- [ ] Run the bounded targeted cases, inspect all outputs, commit scoped
  source/tests/report, then obtain fresh controller checks and independent
  task review. Full `make verify`, new training, current-source aggregation,
  providers and recovery remain separately gated.

Ruling: the context belongs to `raw_run_dir`. Gate/compact/exported-index paths
belong to `artifact_path.parent`. In subsequent integrated callers, supply that
context only when both roots are equal; otherwise read the exported files
locally with their original hashes. Never substitute an identically named raw
shard for an absent exported shard. No second context or cross-run resolver is
needed for this task. Direct readers reject a mismatched supplied context.

Ruling: the later complete reuse branch must reread gate bytes within their
lease and compare their hash with `verified["artifact_sha256"]` before returning
the artifact. The small predicate does not claim to complete this broader
binding, collection, recovery or execution work. A genuinely produced
current-source debug gate is required for positive end-to-end proof; historical
source rejection is a negative test only.

#### Bounded offline-process capture and completion fence

Ruling: the source-bound full-gate test needs one actual offline diagnostic,
whose parent currently captures unlimited stdout/stderr and reuses merely
present offline.json. First deliver a separately reviewable parent-process
fence, then the closed child-writer allowance described in the operational
design. This is an in-scope Task5 execution/verification correction under the
standing approval; no scientific threshold, algorithm, RNG semantics or
foundation-model boundary changes. This slice does NOT establish a child
filesystem quota or authorize a fresh neural diagnostic.

**Files:** create private `train/pilot_offline_process.py` and
`tests/pilot/test_pilot_offline_process.py`. Modify `train/pilot_offline.py`
for its exact bootstrap, measurement and report parsing;
`train/pilot_checks.py` for pre-execution reuse admission; and
`train/pilot_evidence.py` for collection, process-evidence validation and the
new required source file. Extend existing offline tests only where their
assertions depend on the exact bootstrap/report variant. No Make/CLI changes,
CI, new archive wire format, general subprocess framework or second ledger.

**Interfaces and records:**

```python
@dataclass(frozen=True)
class OfflineProcessLimits:
    stdout_bytes: int = 256 * 1024
    stderr_bytes: int = 256 * 1024
    record_bytes: int = 16 * 1024
    timeout_seconds: int = 180

def measure_pilot_offline(*, output_dir, process_limits=None): ...

def read_offline_process_outcome(
    run_dir, *, report, evidence_context=None, unavailable=None
): ...
```

The limits must be positive plain integers no larger than these defaults
(reject bool, unknown fields and enlarged overrides). Tests lower them.
The caller must reserve their full parent publication peaks in the existing
ledger before launch; this module neither reserves nor replenishes storage.
The reader's run_dir is the existing session run root, with fixed prefix
final/offline, not a new arbitrary-root resolver.

Use strict records with these exact responsibilities:

- Immutable `intent.json`, version
  `phase4-offline-process-intent-v1`: attempt nonce, exact bootstrap hash,
  Python executable identity, source revision/digest, limits and timeout.
  This durable record precedes Popen and is the launch/incomplete fence.
- Create-only `process-result.json`, version
  `phase4-offline-process-result-v1`: completed/failed status, intent digest,
  attempt and available child identity, actual return code, closed failure
  reason, stdout/stderr byte counts and hashes. Only completed status carries
  the verified offline.json digest. Missing terminal result is incomplete.
- New diagnostic variant with evidence kind
  `offline_smoke_diagnostic_v2` and required process-intent SHA-256 binding.
  Keep the original `PilotOfflineReport` and parser for genuine historical
  evidence. Factor a parser used by measurement, collection and semantic
  verification; do not silently reinterpret old report bytes as v2.

No current-source full-gate or collection path accepts a legacy report as a
new process-completion proof. Artifact-only historical readers still retain
the original v1 interpretation. Either a v2 report marker or authenticated
v1-process intent selects the new process protocol; missing intent cannot
downgrade a marked report, and contradictory markers fail.

- [ ] **RED: bounded process primitive.** Write tiny fresh-interpreter controls
  for stdout and stderr independently and simultaneously, timeout, nonzero
  exit, launch failure and cancellation. Check captured prefixes, actual exit
  state and that the exact child is joined. Example assertions:

  ```python
  assert len(captured.stdout) <= limits.stdout_bytes
  assert len(captured.stderr) <= limits.stderr_bytes
  assert captured.reason == "output_limit"
  assert captured.returncode is not None
  assert process.poll() is not None
  ```

  A primitive's successful zero-exit control is process evidence only; no
  fabricated training result or diagnostic success is produced. Any optional
  tiny child that writes offline.json writes an explicitly invalid sentinel
  to demonstrate that file presence cannot override a failed process.
- [ ] **Obtain exact test admission before execution.** Use one fresh retained
  operational scratch prefix per RED/GREEN/controller stage, the established
  closed Python -B launcher, plugins/cacheprovider disabled and explicit
  TMP/XDG/MPL/Hypothesis/archive roots. Inventory control bytes, names and
  directories before writes. No source checkout, real model forward, training,
  replay, diagnostic, provider, copied historical corpus or full make verify.
- [ ] **Implement concurrent bounded capture.** Keep only bounded prefix
  buffers and a fixed-size read chunk. Drain both pipes without deadlock.
  On overflow/timeout/cancellation terminate and join the exact child, with a
  bounded terminate-to-kill grace. Do not use an unbounded communicate or
  capture_output fallback. Overflow is failure even if exit is zero.
  Preserve the actual return code; do not invent one for failed launch.
- [ ] **Fence launch and completion.** Validate root/limits and refuse a
  pre-existing incomplete/failed attempt before launching. Publish intent
  durably, launch only the existing exact offline bootstrap, then publish
  bounded stdout/stderr and finally the create-only terminal result.
  Completed requires exit zero, EOF on both streams, no capture failure,
  valid v2 diagnostic with exact intent/source binding, and durable log files.
  Publication failure leaves no reusable completion. Parent death naturally
  leaves intent without terminal success; no replacement/running-state loop.
  Preserve existing output and propagate a bounded actionable error.
- [ ] **Preserve bootstrap authority.** Bind the serialized intent digest to
  the permitted nested command, current program hash, executable and exact
  output root. Update the closed Popen allowlist without accepting arbitrary
  commands, altered environments, foreign roots or enlarged limits. The worker
  validates the intent before writing its v2 report. This does not grant
  child byte/name allowance; the later closed-writer slice adds that separately.
- [ ] **RED/GREEN: reuse and artifact integrity.** Before any new evaluation,
  continuation, numerics or offline execution, reject an existing new-version
  attempt lacking a valid completed outcome. Test the real reuse decision with
  execution tripwires. Collection uses the same fence. At current-source full
  gate verification require the v2 variant; do not weaken source authentication.

  ```python
  with pytest.raises(ValueError, match="offline process"):
      reuse_with_execution_tripwires(incomplete_attempt)
  assert no_new_scientific_work
  ```

  The test helper names above are local fixtures, not new production APIs.
- [ ] **Validate all available process evidence independently.** Add intent,
  terminal result and logs to the exact exhausted cold inventory. Keep one
  payload lease and validate all available records/digests despite another
  required member missing. Missing new-process records add existing-style
  unavailable labels; present failed/inconsistent records raise. Cover wrong
  intent/attempt/report/log binding, invalid status/return-code combinations,
  advertised read failure, late inventory failure, wrong root and lease cleanup.
  Corruption cannot become mere unavailability or select legacy behavior.
  V1 historical equivalence remains an artifact-only test, not current proof.
- [ ] **Bound diagnostics before launch.** Intent/result are each <=16 KiB;
  stdout/stderr each <=256 KiB by default. Charge their existing bytes plus
  full next temporary, old/new/link coexistence and directory/name cushions.
  This separately reserved parent allowance is not consumed by later child
  scientific writes. Record actual bytes/allocated peaks and preserve all
  failures. Generic 64/128 MiB format limits do not authorize larger logs.
- [ ] **GREEN/review/commit.** Run only the admitted process/reader controls
  and named legacy offline regressions, then scoped Ruff/diff checks. Commit
  source/tests and a report with exact identities, commands, timing, retained
  allocation and limits. Controller repeats cheap checks and obtains an
  independent task review. Real v2 worker/gate proof remains outstanding until
  closed child writes and the complete source-bound run allowance are reviewed.

Self-review: this slice covers bounded parent logs and process-completion
admission only. Child publication/caches, outer gate/collection cold routing,
recovery, actual current-source full gate and Task7 storage handoff remain
open. The new module must be in the explicit current source inventory.

#### Privacy correction after standalone Task A

**Goal:** Remove machine paths and arbitrary captured output from publishable
diagnostic evidence while preserving source/launch authentication and failures.

**Architecture:** Hash-only versioned launch records plus fixed private capture
custody in the existing active operational reservation. No new storage ledger,
generic redactor, inherited-authority channel or scientific behavior change.

**Tech Stack:** Python 3.12, existing process capture, strict schemas, pinned
durable file writers, existing scoped storage admission and pytest.

**Spec:** R2 design Sections 2 and 7, especially executable-path privacy and
machine-path/archive exclusions. This corrective slice supersedes the preceding
parent-process plan wherever its plaintext intent/public raw logs conflict.
Standing approval covers this reversible correction. Execute serially after
Task A source freeze/review and before production Task B, real diagnostic or
archive promotion. Existing functional tests are not privacy clearance.

##### Scope

Change pilot_offline_process.py, pilot_offline.py, pilot_evidence.py,
pilot_checks.py and bounded process tests; narrow forwarding/pre-work guards
in pilot_workflow.py/pilot_verification.py; private-subtree .gitignore rule.
No new ledger/resolver/redaction, historical/scientific changes or FM calls.

Defer inherited supervisor/FitGuard custody to open Task5 integration.
supervisor._execute_job passes archive_producer/evidence_context; _child_main
creates LocalArchiveSession and calls _request("status", {}), NOT custody.
Invent no channel. Missing authority stops fresh diagnostics before costly
work: temporarily unsupported workflow, not completed delivery.

##### Hash-only versions and readers

Strict intent-v2 replaces output_root/python_executable with
output_root_sha256/python_executable_sha256. Retain executable-content
python_sha256, attempt, bootstrap/source hashes, revision and limits.
Reject plaintext/unknown keys. Exact identity recipe:

```python
def path_identity(kind: bytes, value: str) -> str:
    return hashlib.sha256(
        b"phase4-offline-process-v2\0" + kind + b"\0" + os.fsencode(value)
    ).hexdigest()
# kind: exactly b"output-root" or b"python-executable"
```

Hash canonical absolute no-symlink output root and exact absolute argv
executable string; aliases differ. Launch/bootstrap enforce these identities;
actual argv/_PENDING/environment stay private. Artifact readers compare original
bindings, not verifier-local root/binary; retain logical evidence_context root
agreement.

Strict result-v2 retains actual pid/returncode/EOF, intent/attempt binding,
captured-prefix counts/hashes and offline hash. Add per-stream disposition:

```python
for name in ("stdout", "stderr"):
    disposition = getattr(result, name + "_disposition")
    count = getattr(result, name + "_bytes")
    digest = getattr(result, name + "_sha256")
    if disposition == "empty_public":
        if count != 0 or digest != hashlib.sha256(b"").hexdigest():
            raise ValueError("offline process empty stream differs")
    elif disposition == "private_rejected":
        if count <= 0 or result.status != "failed" or result.offline_sha256 is not None:
            raise ValueError("offline process private stream differs")
    else:
        raise ValueError("offline process stream disposition differs")
```

For an empty-public stream the reader validates a present file is exactly empty;
absence is unavailable, never completed. For private-rejected streams any
corresponding public file is corruption, including a zero-byte substitute.

Completion requires two durable empty logs, joined zero exit, EOF, no failure,
bound report. Add private_output_rejected/private_custody_failure reasons;
earlier failure stays primary. No public exception text/paths/raw bytes or
fabricated child identity.

Report-v2 binds intent-v2. Current measure/reuse/collection require all safe
versions; full gate enforces AFTER source authentication. Keep old forensic
parsers, no rewrite/reexport. Outcome reader adds require_safe_process=False;
current consumers pass True. Old present intent fails; missing safe intent
is unavailable, not downgrade. Failed/private result is fatal even without
public logs. Validate all available members; preserve inventory/one-lease/
direct-read guards. Scientific readers never hydrate private custody.

##### Same-owner custody API

Keep process module Torch-free (lazy/type-only archive imports):

```python
def prepare_offline_process_custody(
    *, budget: _StorageBudget, admission: _ScopedAdmission,
    output_dir: Path, limits: OfflineProcessLimits, attempt_id: str,
) -> OfflineProcessCustody: ...
```

OfflineProcessCustody is frozen/slots, repr=False and nonserializable. Fields:
budget (private object reference), retained (_RetainedHistoryScope), token,
owner_pid, owner_create_time, workspace_device, workspace_inode, directory_fd,
directory_device, directory_inode, attempt_id, output_root_sha256, limits,
category_peaks (immutable tuple of category/byte pairs). No path-string field
or mutable allowance dict. Closing fd never deletes evidence. Trusted-process
integrity only.

Preparation/prelaunch/publication checks use real ledger interfaces:

1. Exact _ScopedAdmission type; require budget.active_reservation ==
   admission.token == admission.retained.reservation_token. Require
   retained.owner_pid == os.getpid(); reject inherited ownership in this
   slice even when child_inheritable. budget.check_scoped(admission.retained)
   verifies live PID/create_time, policy/workspace/retained identity and
   admission commitment; returns allocated totals.
2. Under existing workspace.lock, read budget._state()["reservations"][token],
   match pid/create_time and use record["amounts"]/record["before"].
   For category c, remaining = amounts[c] -
   max(0, allocated[c] - before[c]); cover full P at preparation, only the
   still-unwritten phase peak at later checks (no double charge).
   No reserve/rebind/replenish. Sole writer, no concurrent launch; remeasure.
3. Derive ONLY budget.workspace/logs/offline-process-private/<attempt_id>,
   where attempt_id is a closed nonce component, never a path. Call
   budget.require_path on this route and output_dir; verify no-follow
   ancestor/device/inode identities and absent create-only targets.
   Use state["paths"] longest-prefix category rule as measure(): private
   route MUST already count as logs; public category must also be admitted.
   Never change bindings while a reservation is live.
4. Pin directory0700/files0600 on the same device; reject symlinks/hardlinks,
   external aliases and public/private ancestor overlap. Public root must
   have the existing canonical run/final/offline shape. The PRIVATE route must
   reject overlap with its run, budget.root controls, spool/cache bindings and
   frozen engineering custody. Public run files may occupy an already-bound,
   separately admitted spool root, as required by supervisor._bind_storage_paths
   and Task C; reject public overlap with controls, logs and cache bindings.
   Prove the fixed private route remains outside scientific/export
   and engineering membership; if boundaries are unavailable, fail admission.
   Revalidate fd/path identities before writes.

ledger.measure counts logs; preflight._inventory excludes live workspace from
engineering snapshot; engineering candidates use closed tmp/task snapshots;
scientific exports select run members. Private route stays outside membership.
Ignore **/logs/offline-process-private/; never producer/index/export/register
it. Future operational archives must exclude it or reject launch.

##### Forwarding and publication

measure_pilot_offline(..., process_limits=None, process_custody=None) validates
matching capability before public intent/Git/Popen. Missing authority raises
closed custody-required failure: no TMPDIR/arbitrary path/run-derived fallback.
Forward offline_process_custody=None through checks/recovery/verification and
workflow boundaries. Missing/stale custody fails before evaluation or fresh
complete-workflow training. Safe reuse/artifact-only reads need no new custody.
Document fresh-launch restriction; inherited wiring remains deferred.

After bounded directory preparation/source authentication, durably publish
private <=R `binding.json` receipt (attempt/intent digest/limits; no paths), then public intent
before Popen. Preserve pending/argv/env checks and worker hash validation before
Torch/report; offline boundary intact.

Persist every NONEMPTY captured prefix unchanged as private stdout.raw/
stderr.raw with create-only durable publication, including overflow/timeout/
cancellation. Never decode/print it. Discarded overflow bytes are not captured
prefixes. ANY nonempty stream fails, even exit0 or valid-looking offline.json.
Only captured empty streams may create public zero-byte .txt files.
Public count/hash/disposition contains no private locator.

Persist private bytes before terminal. Storage failure retains partial files,
publishes failed result if possible, else incomplete intent; no lossless claim.
Combined publication/launch failure keeps None identity and first failure.
Parent death leaves incomplete fence. No retry/cleanup/raw-public fallback
or completed-with-missing-private-logs state.

##### Same reservation P

Measure st_blocks*512. Keep TaskA's B=4096 admission granule:
A(x)=B*ceil(x/B), not actual allocation. Validate filesystem coverage; larger
bounds need review. 8192 is only the tiny-test cushion. Stream caps S/E,
record cap R:

P = 2*A(S) + 2*A(E) + 6*A(R) + B*(F+D+1).

Two raw files allow final+temporary coexistence; public intent/result and
private receipt each allow twice R. Empty public logs cost names only.
F=14 covers seven finals plus seven temporaries. D is exact missing directories
including fixed parents; +1 writer cushion. Defaults S=E=262144,R=16384,D=2
give P=1216512 bytes, NOT an admission.
Split P by actual category mapping, including temporary/name/directory peaks.
Public publication uses random sibling `.pilot-[0-9a-f]{32}.tmp` names.
Before custody creation, reject a public final-name binding or any binding
matching that sibling temporary namespace whose category differs from the
public output directory's category. Same-category bindings are harmless;
unrelated child subtrees retain their existing mappings. This narrow
restriction makes the public final/temporary pair's existing category forecast
valid without inventing a dynamic temporary-routing service. Recheck it at
prelaunch/publication boundaries with the existing custody validation. The
private route and its fixed members must still count as logs.
Private bytes charge existing logs; public bytes their existing category.
U counts existing bytes once; remaining reservation covers P plus separately
reserved child peaks within existing global/physical headroom checks.
Child offline.json/source/worker files remain outside P. No double reservation
or new ledger. Control harness may use8192 without changing production B.

##### Bounded TDD and handoff

Controller admits exact selectors/fresh prefix/bytes/files/dirs before tests;
no genuine worker/model/fit/evaluation/replay/provider. Assert prewrite limits.

- Tiny stdlib child emits b"/private/SECRET" on stderr and exits0: private
  stderr.raw exactly15B, public stderr.txt absent, result failed/private_rejected
  with matching digest/count and null offline hash; reuse rejects. Empty child
  proves PROCESS-only publication, never fabricated scientific metrics.
- Tripwire Git/Popen/evaluation; fresh checks with no custody must raise the
  fixed error with all tripwire counts zero.
- Independent hashes, no public sentinel; changed launch bindings rejected
  by validator/fresh-process audit.
- Stale owner/depleted budget/aliases fail before writes; account private bytes
  but exclude them from scientific/export/engineering inventories.
- Overflow/timeout/cancel prefix retention and failed-launch/publication combo;
  strict disposition/version contradictions, authenticated current rejection
  of old intent, forensic compatibility and missing-log failed result.
- Preserve cold corruption/lease/direct-read guards.

Retain failures; static checks, commit/freeze, controller repeat/privacy review.
No workflow/scientific/quota/provider/promotion proof.


##### Ordered implementation and additional exact decisions

- [ ] **RED: safe identity and publication.** Add independent domain-hash and
  strict-version tests before changing serializers/readers. Test an actual tiny
  emitted sentinel and verify its private bytes and public absence. The
  lower-level capture/publication test is process evidence, not a fabricated
  scientific report.
- [ ] **RED: custody admission.** Use tiny isolated test ledgers under the
  controller-admitted prefix. Test active same-owner scope, category exhaustion,
  stale owner, canonical root, overlap and symlink/hardlink rejection before
  any publication. These fixtures test the existing ledger; production creates
  no second ledger.
- [ ] **Implement the narrow preparation/publication path.** Derive attempt
  identity before measurement; measurement uses the capability's exact nonce.
  The only private retained names are `binding.json`, `stdout.raw` and
  `stderr.raw`. No arbitrary member argument. Preparation rejects any existing
  attempt directory and pins its new descriptor. Add explicit close/context
  support that releases descriptors only, never deletes retained evidence.
  The capability is operational-only: do not place it in config, public
  serialization, job requests or subprocess arguments.
- [ ] **Implement strict current versus forensic readers and early guards.**
  Add the exact `require_safe_process` keyword, explicit intent/result version
  dispatch, disposition checks and available-corruption checks. Forward
  `offline_process_custody` only through the listed existing call chain;
  artifact-only safe reuse needs no fresh capability. Reject fresh work lacking
  custody before training/evaluation/numerics, not after they consume time.
- [ ] **GREEN: bounded matrix and scoped static checks.** Run only the
  prospectively admitted named controls, preserve every failed prefix, and
  list full commands, source identity, actual allocation/counts and outcomes.
  Commit source/tests, freeze, then controller verification and independent
  task review. Do not run the real diagnostic, provider, source-copy workload
  or full local gate without separate complete admission.

Example independent hash assertion (no machine path is published):

```python
expected = hashlib.sha256(
    b"phase4-offline-process-v2\0output-root\0" + os.fsencode(str(output))
).hexdigest()
assert intent.output_root_sha256 == expected
assert str(output).encode() not in canonical_json_bytes(intent)
assert "output_root" not in intent.model_dump()
```

Example failure assertions after the tiny sentinel child:

```python
assert captured.stderr == b"/private/SECRET"
assert result.status == "failed"
assert result.stderr_disposition == "private_rejected"
assert result.stderr_bytes == 15
assert result.stderr_sha256 == hashlib.sha256(captured.stderr).hexdigest()
assert result.offline_sha256 is None
assert not (output / "stderr.txt").exists()
```

The test's private reader verifies the fixed `stderr.raw` bytes using the
pinned fixture custody; no private path or raw value enters a public fixture
artifact. Earlier capture failure takes precedence over privacy rejection,
but either prohibits completion. Missing or undurable private capture is a
failed/incomplete attempt, never a successful output with an attachment omitted.

##### Controller integration self-review

| Interface pair | Produced / consumed | Ruling |
| --- | --- | --- |
| Task A / privacy | Standalone writer hooks / parent-only private capture | Parent writes occur outside the diagnostic child; A remains independently testable. |
| Privacy / Task B | Strict safe intent + process custody / optional child allowance | B adds its optional allowance to safe intent-v2 and cannot accept plaintext-v1 for new launches. |
| Privacy / Task C | Split parent category peaks / held child allowance | Replace older P formula with this explicit private/public split; C and inherited custody integration remain open. |
| Privacy / cold readers | Forensic version parsing / current source-bound eligibility | Old bytes remain readable but cannot certify current execution. |
| Privacy / workflow | Explicit same-owner capability / fresh complete execution | Missing custody is an early documented restriction until inherited wiring is reviewed. |
| Scope / tests | Tiny bounded real captures and existing-ledger controls / genuine scientific run | No synthetic successful scientific report, provider proof or full gate is claimed. |

Self-review checked R2 privacy, unchanged scientific controls, current-source
authentication, cold lease/error semantics and the same global/category caps.
The remaining supervisor/FitGuard inheritance and actual full diagnostic are
explicitly open integrations, not omitted completion gates.

#### Closed offline child-writer allowance

**Goal:** Enforce an explicitly reserved, finite allowance for the existing one-update, 16-episode offline diagnostic before its known disk writes occur.

**Architecture:** Bind one child allowance into the existing parent intent. Install admission before Torch and publisher adapters before imported aliases; latch denial through bootstrap exit. Retain the existing parent ledger and FitGuard.

**Tech Stack:** Python 3.12, existing stdlib audit/descriptor boundary, existing durable publishers, pytest fresh-interpreter controls.

**Spec:** `docs/superpowers/specs/2026-09-17-phase-4-r2-archive-design.md`, the canonical Silent Cascade specification, and the recorded workload design in `operational/sdd/task5-offline-bound-design.md`. This is an in-scope Task5 slice under standing approval, not a new phase or permission to run the full diagnostic. Standalone Task A began after the parent functional review; the subsequent privacy correction must close before production Task B or diagnostic/archive promotion.

##### Constraints and ownership

- Keep the worker's actual update, all 16 evaluations, replay, reporting, device selection, global/native RNG behavior, controls, gates and zero foundation-model calls unchanged.
- Default `write_allowance=None` adds no child byte limit; it does not waive the privacy correction's parent custody requirement. Existing boundary-only callers retain their legacy behavior.
- Unknown writers fail; no descriptor/cache fallback or kernel-wide native-write claim.
- No default byte capacity or historical-size-derived success claim. Trajectory ceilings permit 16 times 128 MiB. Debug weights have 3,084,378 tensor bytes plus a header capped at 16 MiB.
- No neural worker, copied corpus, provider/network or `make verify` in primitive qualification. Full local verification/current-source gate and archival require separate admission.
- Original evidence outside `operational/` stays frozen. Use existing records/ledger and current branch; commit meaningful authorized changes. No new quota framework or archive schema.

**File responsibilities:**

- Create `src/silent_cascade/train/pilot_offline_writes.py`: stdlib-only initial imports; allowance, grammar, accounting, adapters.
- Modify `src/silent_cascade/train/pilot_offline.py`: launch handoff, early hooks, final latch checks after parent writer finishes.
- Modify `src/silent_cascade/train/pilot_offline_process.py`: optional strict intent field; retain bounded capture and legacy semantics.
- Modify `src/silent_cascade/train/pilot_evidence.py`: new `REQUIRED_PACKAGE_FILES` member; existing intent parser authenticates limits.
- Modify `tests/pilot/cold_authentication_fixture.py`: three aliases and one held offline reservation.
- Create `tests/pilot/test_pilot_offline_writes.py`; extend primitive controls in `tests/pilot/test_pilot_offline.py` and `tests/pilot/test_pilot_offline_process.py`.

##### Closed interfaces

```python
@dataclass(frozen=True)
class OfflineWriteAllowance:
    allocated_bytes: int                  # REQUIRED; no default capacity
    file_names: int = 102
    directories: int = 9

class OfflineWriteDenied(RuntimeError):
    pass

def measure_pilot_offline(
    *, output_dir, process_limits=None, process_custody=None,
    write_allowance: OfflineWriteAllowance | None = None,
): ...

def install_offline_boundary(
    *, workspace_root=None,
    write_allowance: OfflineWriteAllowance | None = None,
): ...  # preserve existing (attempts, blocked) return

def install_offline_writes(root: Path, allowance: OfflineWriteAllowance):
    ...  # return OfflineWrites; irreversible, one root per child

class OfflineWrites:
    def bind_publishers(self) -> None: ...
    def assert_clear(self) -> None: ...
    def admit_publication(self, path: Path, size: int, *, writer: str) -> None: ...

def reserve_offline_attempt(
    self, output_dir, *, write_allowance, process_limits, process_custody
):
    ...  # new FitGuard method: hold once, never replenish
```

Use positive plain integers (`type(value) is int`); `allocated_bytes <= 2**63 - 1` bounds wire integers, not granted capacity. Enforce names <=102, directories <=9. Reject unknown keys and null/bool/float/coerced values inside a present allowance, and reject nonexact API dataclass types. The optional whole allowance may be absent or None for the legacy unbounded-child mode. Tests lower limits. Caller holds bytes/names/directories before measurement; this module grants none.

Add optional `write_allowance: dict[str, int] | None = None` to the privacy-safe strict intent-v2; validate its exact three keys. Missing on genuine old intents means no child allowance in forensic parsing only; invalid present fields fail. Existing raw-byte intent SHA-256 binds limits/root/attempt/source/executable/bootstrap. Do not rewrite old records or scientific fields. Child denial remains `nonzero_exit`, with nonempty stderr in private custody and its count/hash/disposition in result-v2, never public raw stderr. No additional marker/record beyond the privacy protocol.

Production limits come only from `validate_offline_launch(root, intent_digest)`, never overriding argv/env. `_PENDING_OFFLINE_LAUNCH` stays the exact root/digest pair; `_PROGRAM` hash binds bootstrap. Changed root/intent/limits/attempt/environment/bootstrap fails. Installed allowances cannot be replaced/enlarged/rebound or authorize another child. No-allowance parents retain the exact existing fenced launch.

##### Writer-derived path grammar

Paths are relative to canonical output root. Preserve no-follow descriptor confinement: reject lexical `..`, symlinks, foreign device, nonregular files, external aliases and unsupported dir-fd forms before mutation. Count every name. Final names authorize only their listed adapter, not raw writes.

Reject pre-existing multi-link inodes unless both names are the active owned temp/final or pending/final pair; a hardlink to outside-root data must not become a writable alias. Recheck pinned parent/fd identities on mutations, not only resolved strings.

| Child family | Exact allowed names / bounded grammar | Retained maximum | Adapter |
| --- | --- | ---: | --- |
| Root outputs | `executed-source.json`, `step.json`, `weights.safetensors`, `manifest.json`, `replay.json`, `offline.json` | 6 | pilot publisher for JSON except replay; durable-temp for weights/replay |
| Evaluation controls | `run/eval/primary/{identity.json,retention.json,rows.jsonl,index.json,metrics.json,DONE}` | 6 | durable-temp; rows only from pending link |
| Episode outputs | `run/eval/primary/episodes/{ordinal}.{suffix}`; ordinal exactly `00000` through `00015`; suffix exactly `neural.json`, `trajectory.json.gz`, `telemetry.json` | 48 | durable-temp |
| Shared evaluation weights | `run/eval/primary/crashes/weights-{sha256}.safetensors`, lowercase hex64, at most one distinct digest | 1 | durable-temp |
| Crash evidence | `run/eval/primary/crashes/{uuid}.{json,safetensors}`, lowercase hex32, at most 16 distinct UUIDs; plus exact `crashes/index.json` | 33 | durable-temp |
| Report | `report/trajectories-0.svg`, `report/tables.json`, `report/report.md`, `report/report-index.json` | 4 | pilot publisher |
| Font cache | `matplotlib-cache/fontlist-v3.11.0.json`, one name | 1 | narrow text stream |

Sum: **99 child retained names**. Design's 102 included three former parent names. Parent fence now owns four: `intent.json`, `process-result.json`, `stdout.txt`, `stderr.txt`, separately reserved. Child may read validated intent but never mutate/link parent files.

Grammar sources: `_worker`; eval `_run_one`, `write_evaluation`, `_crash_index`; `stage_neural_runtime`; `EventEngine._publish_failure`; `logging/crash_bundle.py`; report enumeration starts at zero.

Only **nine directories** including root: `.`, `run`, `run/eval`, `run/eval/primary`, `run/eval/primary/episodes`, `run/eval/primary/crashes`, `report`, `matplotlib-cache`, `cache`. XDG `cache` allows no files. TMPDIR=root grants no scratch files or `matplotlib-*` fallback. Preserve synthetic EEXIST for safe publishers' existing ancestor mkdir attempts outside root; do not mutate/latch these.

Additional simultaneous child names have exact ownership:

1. **One** atomic temp: `.pilot-[0-9a-f]{32}.tmp`, or io `.{target.name}.{eight_chars}.tmp` (CPython 3.12 tempfile lowercase/digit/underscore alphabet), in the admitted target's parent. Allow only the active publisher's actual candidate, not any regex match/second temp. Failed retained temps remain counted; denial forbids new publication.
2. The exact `run/eval/primary/.rows.pending.jsonl`; allow its final link only to the exact `rows.jsonl`. Both names count during coexistence.
3. The zero-byte `matplotlib-cache/fontlist-v3.11.0.json.matplotlib-lock`.

Peak **99 + 1 + 1 + 1 = 102** also constrains failed attempts. UUID pairs share 16 stems; UUID17, weights digest2, ordinal16 or SVG2 fails even with bytes remaining. Names are ceilings, not promised artifacts.

Installed `FontManager.__version__` is `3.11.0`, independent of locked distribution `3.11.1`. Its `json_dump` writes via `open(filename,'w')`; `cbook._lock_path` uses `Path.open('xb')` then unlink. Derive grammar without importing font_manager before admission. Changed cache protocol requires reviewed closed-name update. SVG uses BytesIO; gzip/safetensors return bytes.

##### Pre-write reservation calculation

Use FitGuard's cushion: `B=4096`, `rounded(n)=((n+B-1)//B)*B`; reject unsupported filesystem allocation unit. Charge `st_blocks*512` for each child file/directory name, including hardlink duplicates. At production child startup, authenticate and exclude only the already-created `intent.json`. The other three parent-owned final names must be absent throughout child execution, are never writable by the child, and appearing unexpectedly is an authority failure. Their later parent publication occurs after the child has exited. Standalone primitive roots may omit intent; they grant no production launch authority. Child pays root/directories; parent may conservatively reserve them too.

Let `U,F,D` be current child bytes/file names/directories, `M` missing permitted parents, `s` full proposed payload length, `N` additional name allowance:

```python
projected = U + 2 * rounded(s) + B + (D + M + F + N) * B
require(F + N <= allowance.file_names)
require(D + M <= allowance.directories)
require(projected <= allowance.allocated_bytes)
```

`N=2` for durable temp/final publication, including replacement; `N=1` for stream create/growth or rows final link. Directory creation: `s=0,N=0,M=1`. Reserve parents/full incoming bytes before mkdir/open; existing destinations remain in U for replacement, idempotence and collisions. Stream `s=current_encoded_length+len(next_encoded_chunk)`; final rows-link `s=full pending size`, pending already in U. Never subtract old target before replacement or count only logical length.

Operation tokens span admission through publication/cleanup. OS hooks verify exact target/temp/fd/bytes/link without charging a second operation reserve. Pending rows coexist with episode publications: flush admitted chunks or include buffered high-water allocation in U. At most one atomic token, one row stream, one font stream/lock. Refresh inventory after operations; scans cannot retroactively admit writes.

The following was the parent-only four-public-file reserve before the privacy
correction. It remains historical design rationale, not current launch
authority. Current P must use the preceding privacy subsection's exact
private/public category split and retained receipt/raw-log peaks; Task C must
not use this older expression to authorize a launch.

Former approved lengths were record/stdout/stderr/record (defaults 16/256/256/16 KiB):

```python
sizes = (limits.record_bytes, limits.stdout_bytes,
         limits.stderr_bytes, limits.record_bytes)
P = sum(2 * rounded(n) + B for n in sizes)
P += 2 * rounded(max(sizes)) + B + (9 + 6) * B
# four retained parent names plus two conservative publication names; nine dirs
```

Using the privacy-corrected P, hold `H=allowance.allocated_bytes+P` plus
names/directories before intent, preserving each category's share. Child cannot
spend P. Bind one fresh measurement root and its separately pinned private
custody. A larger explicitly admitted P is valid; unused global headroom is not
an admission.

FitGuard stores one held root in memory. Outer admissions count non-offline spool/cache allocation plus H; do not double-count live child bytes or let unrelated writes spend H. Still validate offline inventory. Route four parent writes to P/per-record limits. No second reservation/root/replenishment; keep full hold through failures for guard lifetime. H plus retained fit/cache/outer publication peaks must fit existing FIT_BYTES/FIT_NAMES/FIT_DIRECTORIES before launch. No envelope expansion or full-gate total follows.

Task C consumes the actual `process_custody`, validated with
`require_offline_process_custody(..., phase=0)`, to obtain its immutable
`category_peaks`. It must not guess the private path or recompute the historical
four-file P. The caller first owns the global admission and prepares private
custody, then establishes this fixture hold before any public intent or child
launch. `H = write_allowance.allocated_bytes + sum(peak for _, peak in custody.category_peaks)` is
conservative inside the unchanged fixture envelope, including private bytes
whose real category remains logs. The public spool and private logs shares must
also fit the live underlying category admission; this fixture hold grants none.
If child output paths have a different effective category from the public run,
reject this fixture layout rather than allocating the whole child allowance to
the wrong category. No binding mutation, child-inherited authority or raw-private
publication hook is added by the fixture.

For the fixture directory ceiling, count the union of retained spool/cache
folders outside the held child root, required public ancestors outside that
root (whether existing or still missing), the child directory allowance, and
the prepared private-custody folders outside spool/cache. Count each shared
folder once. The child allowance includes the output root, not its run/final
ancestors. Private folders stay in logs for global accounting but are included
conservatively in FIT_DIRECTORIES, just as full private P is included in H and
parent F=14 in the name hold. Use the existing validated custody route/identity,
not a new locator or public record. These folders' bytes are already covered
by P; do not charge them a second time in H. With a wholly absent run/final
path and child directory allowance9, the simple fixture holds14 directories
(9+2 public ancestors+3 prepared private folders). Exact and one-short byte,
name and directory cases must fail before public intent or launch when short.


##### Task A: Closed adapter and tiny writer controls

**Files:** new `pilot_offline_writes.py` and `test_pilot_offline_writes.py`, with its required source inventory entry in `pilot_evidence.py`; integrate low-level boundary hooks in `pilot_offline.py` only after the parent change is stable. Source closure must not lag the new module.

- [ ] **RED — limits/grammar/latch.** Pure validation plus fresh-interpreter tests: 1–4096-byte real publications, exact calculated capacity and one byte short; invalid types/keys, unknown names/dirs, episode16, UUID17, weights2, SVG1. Preseed only admitted tiny controls before child launch. No models. Assertions:

  ```python
  from silent_cascade.train.pilot_data import _publish_pilot_bytes
  _publish_pilot_bytes(root / 'step.json', b'x')
  # Construct exact and short limits from the formula and admitted inventory.
  assert (root / 'step.json').read_bytes() == b'x'
  with pytest.raises(OfflineWriteDenied):
      guard.admit_publication(root / 'unknown.json', 1, writer='pilot')
  with pytest.raises(OfflineWriteDenied):
      guard.assert_clear()
  assert not (root / 'unknown.json').exists()
  ```

  Exercise actual patched publishers, not only admission arithmetic. Success body:

  ```python
  from silent_cascade.train.pilot_data import _publish_pilot_bytes
  from silent_cascade.io import atomic_create_bytes
  _publish_pilot_bytes(root / 'step.json', b'x')
  atomic_create_bytes(root / 'weights.safetensors', b'w')
  guard.assert_clear()
  ```

- [ ] **RED — peaks.** Tiny existing target plus incoming bytes: denial leaves old bytes/no temp; create collision reserves incoming bytes; hardlinks count twice; unknown/second temp fails; lowered names/dirs deny before creation. One intentional denial per interpreter, then check later known publication fails without clearing latch.
- [ ] **Implement adapters.** Validate canonical root and explicit allowance in the standalone adapter; Task B additionally enforces the production bootstrapped-intent binding. Install stdlib hooks before Torch/consumers, preserving os.open dir-fd confinement. Cover builtins.open, io.open (Path.open), mkdir/link/rename/unlink/truncate and fd-opening paths. Preserve original publishers:

  ```python
  def guarded_pilot(path, payload):
      self.admit_publication(path, len(payload), writer='pilot')
      with self._pilot_operation(path, payload):
          return original_pilot(path, payload)

  def guarded_temp(path, data, mode):
      self.admit_publication(path, len(data), writer='atomic')
      with self._temp_preparation(path, data):
          temporary = original_temp(path, data, mode)
      self._retain_temp_token(temporary, path, len(data))
      return temporary
  ```

  Private `OfflineWrites` methods: `_pilot_operation` owns UUID temp/parent walk/final link; `_temp_preparation` owns one mkstemp candidate/write/fsync/chmod; `_retain_temp_token` preserves ownership after helper return through caller's link/replace/fsync/unlink. Each checks grammar, parent inode/device, fd and target; no blanket 'inside publisher' bypass.

  Unlink only owned temps, linked pending rows and font lock; no retained-final deletion. Own-temp/lock cleanup remains allowed after denial without writes/reset. Audit/adapter denials latch first code (`bytes`, `names`, `directories`, `path`, `writer`, `descriptor`, `authority`); bounded details, no payload dump.

- [ ] **Exact streams.** Rows: `xb`, bytes write/tell/flush/context/close/fsync. Font: `w` UTF-8, admit encoded chunks before writing, return character count, flush admitted chunks. Lock: `xb`/close, zero bytes. Deny/latch append/update modes, seek/truncate/writelines, buffer/raw/delegation/reopen; no `__getattr__` forwarding.

  Preserve actual evaluator `os.fsync(handle.fileno())` using an opaque non-integer row token. Patched os.fsync recognizes its active token and uses the private fd. Raw open/write/ftruncate/fdopen/FileIO/dup/mmap cannot consume tokens or expose underlying streams. Wrap escape entry points so token misuse latches rather than merely raising catchable TypeError. Ordinary known-fd fsync passes through. Preserve stdout/stderr capture and exact read-only Git batch-pipe capability.

- [ ] **RED/GREEN — streams/escapes.** Real rows append/tell/flush/fsync/link; exact/over UTF-8 chunks; genuine font_manager cache import/zero-byte lock/cleanup; tiny json_dump overflow followed by assert_clear even if an exception was caught. Test buffer/raw/fdopen/dup/write/truncate/mmap, relative/dir-fd escape, symlink and external hardlink. Synthetic JSON alone cannot qualify the real font writer.
- [ ] **GREEN/review/commit.** Separately admitted named primitives and scoped Ruff/diff; cache/native side effects must be covered or denied. Record commands/allocation/denials; commit reviewed source/tests. No scientific success claim.

##### Task B: Bind launch authority and preserve parent completion

**Consumes:** privacy-corrected parent's `OfflineProcessLimits`, strict safe intent/result, private process custody, `validate_offline_launch`, `_PENDING_OFFLINE_LAUNCH`, capture/result reader. **Produces:** optional allowance-bound worker; no-allowance mode retains safe parent requirements. Forensic old parsing is not new-launch authority.

- [ ] **RED — compatibility/authority.** Old missing field parses as no allowance. Invalid/enlarged/removed field or changed root fails original digest. Tiny bootstrap sentinels are process evidence, never fabricated scientific success.
- [ ] **Wire launch.** Validate type/root/hold contract, serialize asdict(allowance) before intent digest, retain six-element command/closed environment. Bootstrap consumes validated intent:

  ```python
  intent = validate_offline_launch(sys.argv[1], sys.argv[2])
  allowance = (None if intent.write_allowance is None
               else OfflineWriteAllowance(**intent.write_allowance))
  attempts, blocked = install_offline_boundary(
      workspace_root=sys.argv[1], write_allowance=allowance)
  _worker(sys.argv[1], attempts, blocked,
          process_intent_sha256=sys.argv[2])
  if _BOUNDARY_WRITES is not None:
      _BOUNDARY_WRITES.assert_clear()
  ```

  In the actual bootstrap use `import silent_cascade.train.pilot_offline as boundary` and `boundary._BOUNDARY_WRITES` for both accesses; the snippet shows the checks, not a stale `from ... import` binding. Require the installed root/allowance to equal the validated intent before `_worker` runs and again when it validates launch.

  `_BOUNDARY_WRITES` is a once-initialized module global; read it after installation, not via stale imported value. Install stdlib admission, import Torch/network hooks, then `bind_publishers()` imports io/pilot_data and patches helpers before worker consumers. Reject late consumer aliases bound to originals. pilot_data imports Torch-dependent evaluation types: initial admission must precede that import. io atomic aliases resolve patched `_durable_temp` dynamically.

  Also assert_clear immediately before worker offline.json publication. Bootstrap postcheck catches swallowed denial/control returns. Completion still requires zero exit, drained pipes, bound scientific report and clear latch. Never substitute a report after denial.
- [ ] **GREEN — combined fence.** Caught denial then assert_clear exits nonzero; invalid preseeded offline.json cannot override failure/overflow. Simultaneous pipe overflow and timeout terminate/join child; P retains bounded logs/result despite child exhaustion. Interrupted missing result is incomplete. Changed allowance fails reuse/collection authentication. Named legacy retained.txt/descriptor controls still work with no allowance.
- [ ] **Inventory/review/commit.** Verify the required source path added with Task A; parser/bootstrap and scoped Ruff/diff; retain failed roots. No worker success claim.

##### Task C: Hold the exact child allowance in the existing fixture

**Files:** `tests/pilot/cold_authentication_fixture.py`, focused tests in `test_pilot_offline_writes.py`; one path-contract correction in `pilot_offline_process.py` and focused custody tests in `test_pilot_offline_process.py`.

**Post-privacy interface ruling:** `_custody_paths` currently rejects the public
run itself when it overlaps any spool binding. The privacy requirement concerns
PRIVATE capture membership; this public restriction also rejects the existing
supervisor's mandatory spool run. Correct that distinction in this task, after
Task B freezes the shared file. Keep the private route outside the run, controls,
all spool/cache bindings and frozen engineering custody; keep public run outside
controls/logs/cache. No binding changes under a live reservation. Do not delete
the original unadmitted-spool negative case: it still lacks a spool allowance.

- [x] **RED/GREEN — admitted public spool.** Use a tiny real isolated ledger,
  bind a fresh public run as spool before its one reservation, and reserve the
  existing privacy-derived public P in spool and private P in logs. Preparation
  and revalidation must succeed, with no raw capture inside the run. One byte
  short in the public spool share fails before custody creation. Private route
  overlapping any spool/cache binding, public cache/control/log overlap, changed
  category mappings and no admission still fail. No real diagnostic or provider.
  The correction changes only which side of the separation rule is checked;
  it does not weaken category accounting or add inherited ownership.

- [x] **RED — aliases.** Tiny undersized-guard tests for bound publishers in pilot_checks, pilot_evidence and report.pilot; no numerical execution. Patch those three aliases in installed() with existing wrapper.
- [x] **RED — hold.** Tiny retained outer file: exact H fits, one byte short fails before intent/launch (tripwire). After reserve, unrelated output cannot spend H; failed child cannot rebind/replenish; live child not double-counted; logs/result can use P.
- [x] **Implement hold.** `FitGuard.reserve_offline_attempt` validates canonical fresh spool root, no previous hold and dataclasses; checks H/names/dirs plus retained usage/missing parents; stores one hold. Extend admit to route exact parent files and preserve H on other writes. Future caller explicitly passes allowance; no FIT_BYTES increase/default capacity/full-gate fit claim or substituted report.
  Validate `process_custody` and matching limits/output before fixing H. Retain
  the public/private category split from its `category_peaks`; check the child's
  full allowance plus the public share in the run's effective category, and
  each other parent share separately. Do not count already-created private
  directories twice against remaining category capacity: compare their retained
  allocation plus the remaining phase peak to the original category hold.
- [x] **GREEN/review/commit.** Primitive hold/alias tests, source inventory and scoped static checks. Record outstanding actual worker/full-gate admission. Obtain independent task review after controller verification; the full-run admission remains separate.

Task C closed at `0aee7bd` after initial `44edf69`, independent 31 + 27
primitive cases, the single directory-union review correction, and 26 affected
cases repeated by the controller. Scoped re-review found all findings addressed
and no new Critical/Important breakage. Evidence is retained under
`operational/sdd/task5-offline-writes-c-*.md`. This closes only Task C, not
actual-worker qualification, inherited supervisor integration, Task5 or Phase4.

##### Task5 next qualification: one direct same-owner diagnostic

After Task C, qualify the existing real diagnostic once from the current
committed checkout, before attempting inherited supervisor integration. This is
a bounded execution refinement of the approved Task5 actual-worker check, not
an implementation of delegated custody or a production pilot. No source,
scientific configuration, threshold or evidence schema changes are needed.

- Use one closed-environment `python -B` operational parent, holding the existing
  global `_StorageBudget._scoped_reservation(child_inheritable=False)`. That same
  process prepares `OfflineProcessCustody` and calls `measure_pilot_offline`.
  No holder-to-child capability transfer, new ledger, checkout or synthetic
  replacement report is permitted.
- Fresh output: `operational/spool/task5-real-offline-01/final/offline`; fresh
  private attempt `d7c2cb464e9043c6a11a77e6dc1a3e51`. Neither may already exist.
  Preserve both on success or failure; never reuse this attempt.
- Pass explicit `OfflineWriteAllowance(16777216, 102, 9)` and default
  `OfflineProcessLimits` (180 seconds, 256 KiB per pipe, 16 KiB per record).
  The child performs the unchanged one-update / 16-episode CPU diagnostic,
  one replay and one report. Its pre-write adapters must remain active. The
  16 MiB cap bounds the attempt; it is not a successful-run size prediction.
- Before admission, verify canonical category bindings, source/config freeze,
  no live reservation/pending eviction, unchanged frozen allocation, and exact
  missing-directory counts. With three missing public ancestors and two private
  directories, the existing publication calculation gives public P=114688 and
  private P=1114112 bytes. Check these exact peaks before creating custody.
- Reserve spool=16891904 (child + public P), logs=2162688 (private P plus 1 MiB
  bounded operational notes/status), metadata=33562624 (two maximum 16 MiB
  control pages plus two allocation blocks), scratch=524288 (parent-only
  confined administrative temporary/cache roots), other categories zero.
  No unrelated producer runs during this reservation. Scientific child outputs
  and native font cache remain inside its own 16 MiB grammar, not this scratch
  cushion. Parent imports use bytecode suppression; no model runs in the parent.
- The last complete ledger measured 605921280 operational bytes plus
  6195077120 frozen bytes. The 53141504-byte proposed increment gives
  6854139904 bytes against the unchanged 6979321856 normal limit, preserving
  the separate 2 GiB reserve and 1.5 GiB emergency capacity. These are a planning
  calculation; the existing live ledger must authenticate and admit it again.
- Retain public intent/result and any private capture using existing semantics.
  Print only bounded status/report metadata, never raw private output. A failure
  is failed diagnostic evidence, not a scientific result. On failure investigate
  from retained evidence; no automatic retry, quota enlargement or cleanup.
- Check category deltas, source/result bindings, actual allocated bytes, zero
  network/foundation-model calls on success, and full frozen custody on owner
  release. Record runtime and outcome. This does not complete Task5/Task7,
  `make verify`, production training, learning acceptance or inherited handoff.

The controller and a read-only admission reviewer inspect the concrete API and
process graph before launch. A definite uncovered writer or missing authority
stops this one attempt before execution, without substituting a different path.

##### Diagnostic01 correction: eager optimizer cache initialization

The first actual diagnostic failed before its update: the installed Torch
optimizer lazily imports Dynamo, whose default Inductor cache calls
`tempfile.gettempdir()` and triggers a forbidden random-file probe. Preserve
that failed attempt. This is a compatibility correction within Task5, not a
scientific or quota change.

Files: `train/pilot_offline.py` (closed environment only), focused
`tests/pilot/test_pilot_offline_writes.py` / environment tests. Do not modify
Torch, the optimizer, the bootstrap, writer grammar, byte/name/directory limits,
private custody or scientific configuration.

1. RED: assert the closed environment derives `TORCHINDUCTOR_CACHE_DIR` from
   the admitted root's existing `cache` path, ignoring an ambient override.
   A fresh tiny real boundary control must construct AdamW on one scalar CPU
   parameter without training, evaluation, replay or model creation, then leave
   only the permitted empty cache directory with a clear denial latch. Confirm
   the unchanged implementation fails at the observed default-cache probe.
2. GREEN: add the fixed cache environment entry. No general temp-file exception,
   extra writable subtree, compiler invocation or disabled guard is allowed.
   Verify attempts to publish compiler/cache files or a child cache directory
   still fail and latch the existing denial.
3. Admit the exact tiny selectors and full parent/child retained bounds before
   each RED/GREEN/controller run; use the existing global ledger, closed
   launcher, fresh prefixes and existing 30-second/2048-byte-pipe tiny-control
   helper. No actual diagnostic is part of these regressions.
4. Controller repeats affected checks and scoped statics after the source
   commit; obtain independent review. Only then consider a separately admitted
   fresh Diagnostic02 under unchanged child/process limits, with exact updated
   public/private missing-directory counts. No automatic retry or failure reuse.

Diagnostic02 is the one permitted follow-up after that fix's independent
verification/review and owner release. Use fresh public
`spool/task5-real-offline-02/final/offline`, administrative
`scratch/task5-real-offline-02`, and private attempt
`6be71f5c342d4fa9a24c9d0f116a8bd2`, all inside the existing operational workspace.
Do not reuse Diagnostic01. The same 16 MiB/102 names/nine child directories and
180-second/256KiB-pipe/16KiB-record limits apply. The private parent now exists,
so one private directory is missing: P_private=1110016, P_public=114688.
Reserve spool16891904/logs2158592/metadata33562624/scratch524288, total53137408;
other increments zero. Reauthenticate these exact peaks and live totals before
launch. Use the same fixed same-owner operational parent with only these
root/attempt/peak changes; retain its exact command and result separately.
No source copy, larger limit, automatic further retry or full-pilot authority.

##### Task D1: Forward the explicit offline allowance without new authority

Prerequisite: Diagnostic02 passed at2e5b96d with unchanged16MiB/180s limits,
one genuine CPU update/16 debug evaluations/replay/report and zero network/FM
calls. This qualifies the primitive, not the inherited main workflow. A
read-only caller map confirms the next independent gap: existing high-level
calls carry process custody but drop the child write allowance.

**Design decision:** transport the caller's exact immutable allowance along the
existing custody edges. Do not infer child bytes from parent-publication P,
install a default16MiB capacity, serialize a capability or create a new hold.
This is plumbing and early argument validation only. Inherited descriptors,
production hold equality and full-workflow admission remain later work.

**Files:** `train/pilot_workflow.py`, `train/pilot_checks.py`,
`train/pilot_verification.py`, `train/pilot_offline_process.py`; focused tests
in `tests/pilot/test_pilot_offline_process.py`. No other source/test files,
supervisor/session/CLI schemas, scientific code or fixture capacities change.

1. Add optional `offline_write_allowance=None` to `_workflow`, `run_pilot`,
   `_run_pilot_checks_owned`, `run_pilot_checks`, `recover_pilot_checks`, and the
   verification module's `measure_pilot_offline` wrapper. Forward the identical
   object on every existing custody edge. At the two low-level measurement
   calls use its existing `write_allowance` keyword. No copying/coercion.
2. Extend `preflight_offline_process(run_dir, custody=None, *,
   write_allowance=None)`: if present, require exact `OfflineWriteAllowance`
   type before the existing completed-artifact/custody checks. Pass the object
   at all four existing early preflight sites (workflow, owned/public checks,
   recovery). Invalid mappings/derived dataclasses must fail before costly
   source/training/evaluation/recovery work. Keep low-level validation intact.
3. Preserve `None` compatibility and completed safe reuse without a new launch
   or live capability. Do not require a custody/allowance pair in this transport
   slice, infer a limit, or change historical forensic/read semantics. Future
   production callers must supply the explicit held allowance; these arguments
   alone do not prove that hold and must not be described as launch authority.
4. RED with tiny spies: every forwarding edge retains object identity;
   malformed/derived allowances stop before expensive tripwires; omitted/None
   behavior and completed-reuse skipping remain unchanged. Stub numerical,
   source/Git and publication operations rather than generating a new model,
   checkout, diagnostic, recovery copy or fabricated benchmark result. Use real
   existing schemas where needed; keep the tests explicitly wiring tests.
5. GREEN minimal signatures/forwarding/type check only. Review all call sites
   including recovery's final owned-check call. No new fields in workflow,
   process reports, gate, recovery intent, job JSON, session or manifests.
6. Before each RED/GREEN/controller execution, record exact selectors, cases,
   child/process count and finite output/directory/native bounds under one
   unchanged global admission and fresh retained stage. New tests need no child
   process. Mechanically bound every outer stream and timeout from the outset;
   do not repeat Diagnostic01's historical RED harness omission. No whole file
   or full local gate is admitted by this slice.
7. Freeze a scoped commit, independently repeat affected checks and statics,
   request task review, and retain any fixes/failures. Record remaining delegated
   custody/production hold/full-gate work explicitly. Do not claim Task5/Phase4
   completion or run a new genuine diagnostic just to prove argument plumbing.

##### Task D2: Admit final-evaluation control publications before writing

**Prerequisite:** D1 is reviewed at `39ecdf4`; its verification is recorded at
`8e5a62d`. This independent storage-only correction does not issue offline
custody, introduce a hold, or admit a complete supervisor/pilot run. The
whole-workflow feasibility audit continues separately. Standing approval for
this Phase4 plan covers this bounded refinement; no scientific gate changes.

**Files:** modify `src/silent_cascade/archive/producer.py` and
`src/silent_cascade/train/pilot_checks.py`; create
`tests/pilot/test_pilot_final_control_admission.py`. Do not change other source,
tests, session schemas, CLI/config, model, diagnostic or ledger code.

**Interfaces:** the existing `ArchiveProducer._before(operation)` sends the
closed `status/{before_work}` request, and `_ArchiveServer._dispatch` derives
the maximum from `before_work_bounds`. Add only fixed operation
`final_evaluation_control` and parameterless
`ArchiveProducer.before_final_evaluation_control() -> None`. It grants no
caller-selected quantity/path or new authority. No new `_OPERATIONS` member.

**Bound:** each `_publish_pilot_bytes` payload is already limited to64MiB.
Its temporary and final hardlink may coexist and per-name accounting charges
both, so preserve `2 * PILOT_BYTES + ALLOCATION_OVERHEAD` =134479872 spool
bytes per control publication. Existing run ownership guarantees the run
ancestor exists; request publication can add final/eval ancestors, and the
completion's evaluation directory already exists. The existing256KiB overhead
covers these bounded directory/name/native costs. This is a per-operation
maximum, not a new reservation or complete final-evaluation peak. Keep every
existing operation bound unchanged; source-derived complete job fit is separate.

- [ ] **Step1 — RED the closed admission behavior.** Exercise the real
  `before_work_bounds`, `ArchiveProducer._before` and server `before_work`
  dispatch with tiny explicit session/budget doubles only. Exact remaining
  134479872 permits the operation;134479871 rejects; one retained spool byte
  consumes the exact margin. Repeated requests must not create capacity. A
  response other than `{"admitted": True}` must propagate rejection.

  ```python
  @pytest.mark.parametrize("remaining, permitted", [(134479872, True), (134479871, False)])
  def test_final_control_exact_admission(remaining, permitted):
      from types import SimpleNamespace
      from silent_cascade.archive.ledger import StorageBlocked
      from silent_cascade.archive.supervisor import _ArchiveServer

      server = object.__new__(_ArchiveServer)
      active = {"before": {"spool": 0}, "amounts": {"spool": remaining},
                "admission": {"source_bound_fixture": True}}
      server.budget = SimpleNamespace(
          active_reservation="fixture",
          _state=lambda: {"reservations": {"fixture": active}},
      )
      server._check_budget = lambda: {"spool": 0}
      request = {"before_work": "final_evaluation_control"}
      if permitted:
          assert server._dispatch("status", request) == {"admitted": True}
      else:
          with pytest.raises(StorageBlocked):
              server._dispatch("status", request)
  ```

  This tests the real dispatch's remaining-capacity decision, not real authority
  issuance. No ledger is initialized by these doubles. Also prove the operation
  resolves exactly `{"spool": 134479872}` and the producer sends the closed
  operation above without numeric capacity in the request.

- [ ] **Step2 — RED publication ordering in the real `_evaluate`.** Use an
  explicitly named wiring fixture: stub source bundle/identity construction,
  neural evaluation and publishers, retaining `_evaluate`'s real branching.
  Record admission, request publication, evaluation, execution publication and
  `after_execution` in order. Denying the first admission must reach neither
  publisher nor evaluator. Denying the second must retain the request/evaluation
  observations but never publish execution or call `after_execution`.
  Success must yield `admit, request, evaluate, admit, execution, after_execution`.
  `archive_producer=None` must preserve `request, evaluate, execution` with no
  admission calls. Completed reuse keeps the existing request-publication/reuse
  validation order and does not admit/write execution again. Do not alter reuse
  behavior merely to avoid its existing request publication. No actual model,
  checkpoint, source checkout, report, recovery copy or scientific output is
  created: spies capture attempted publications in memory. Document precisely
  that these tests prove control admission/ordering, not final scientific output.

- [ ] **Step3 — Run admitted RED.** Before executing, submit exact selectors,
  case count, zero test-child count and finite allocation/name/directory/log
  bounds. Use a fresh retained
  `operational/scratch/task5-final-control-admission/red1` with closed runtime
  paths, no pytest plugins/cache/bytecode and mechanical outer150s/32768B per
  stream limits. Controller owns execution/admission. Expect unknown-operation,
  missing-method or missing-admission-order failures, not fixture/setup failures.

- [ ] **Step4 — GREEN minimal implementation.** Add the fixed mapping and
  producer method, then place the same optional producer check immediately
  before each of the two existing control `_publish` calls in `_evaluate`.

  ```python
  # before_work_bounds
  if operation == "final_evaluation_control":
      return {"spool": 2 * PILOT_BYTES + ALLOCATION_OVERHEAD}

  # ArchiveProducer
  def before_final_evaluation_control(self) -> None:
      self._before("final_evaluation_control")

  # _evaluate: once before .<name>.request.json and once before execution.json
  if archive_producer is not None:
      archive_producer.before_final_evaluation_control()
  ```

  Keep `_evaluate` identities, publication bytes, DONE/reuse checks, CPU checks,
  evaluation arguments and `after_execution` ordering otherwise identical.

- [ ] **Step5 — Verify, freeze and review.** Fresh admitted GREEN and controller
  stages retain every attempt within a proposed2MiB total task scratch ceiling;
  that ceiling is not per-stage execution approval. Run exact focused tests,
  scoped Ruff/format/diff checks and independent frozen source equality. Commit
  only the two source files and new focused test file, then independent task
  review before closing D2. No full-file existing fixtures, genuine diagnostic,
  provider, source-copy workload, full `make verify` or production run.

**Self-review:** D1's `pilot_checks.py` work is complete before D2 editing.
The two added calls consume one fixed producer interface, whose mapping feeds
the existing server dispatch. No mismatch with earlier evaluation/episode
maxima: control admission happens outside their spans. Every science/authority/
retention/local-only constraint above remains binding. Remaining continuation,
numerics, report/gate, workflow and recovery bounds are not certified by D2.

##### Primitive admission and execution instructions

Use the existing closed Python `-B` launcher, disabled external pytest plugins/cacheprovider and explicit TMP/XDG/MPL/Hypothesis/archive roots under a fresh operational scratch prefix per RED/GREEN/controller stage. Select only the named new writer tests and named parent/legacy primitive tests; never select the entire offline test file if it contains the actual diagnostic. Example inner command after its containing launcher/environment has been admitted:

```text
.venv/bin/python -B -m pytest -p no:cacheprovider tests/pilot/test_pilot_offline_writes.py -q
```

Test helper `run_writer_control(code, root, allowance)` must use existing `capture_offline_process` with `OfflineProcessLimits(stdout_bytes=2048, stderr_bytes=2048, record_bytes=4096, timeout_seconds=30)`. Pass a closed `offline_environment(root)`; create the approved root before launch. It launches a tiny `-B -c` program that installs the real boundary with the supplied allowance and performs only the requested primitive. It is a test harness, not a new permitted nested production command. Bound helper source text to 4096 bytes and each captured output to 2048 bytes. Fresh interpreters isolate irreversible hooks and latches.

Use explicit per-control stop limits: ordinary tiny cases at most 64 KiB child allocation, font import case at most 256 KiB; use 1–4096-byte scientific payloads. If a control requires more, compute its exact names/directories/payload peak first and record separate admission. Account every child root, control body, pytest output, parent logs/records, temporary, cache and native cushion; multiply by the exact planned invocation count, including RED/GREEN/controller retained failures. These ceilings bound attempts; they do not certify font import success or fit an entire suite by themselves.

Current normal headroom is about 177 MiB and scratch headroom about 10 MiB; neither is permission to consume it. If the exact retained total of all planned stages exceeds the remaining category, stage fewer named controls or complete the already-approved accounting/archive prerequisite before execution. Do not discard failing roots or silently reclassify scratch. All full-run and `make verify` admission remains outside this subsection.

##### Controller integration self-review

- [x] All 99 retained child names and three transient names are justified by actual writers; four parent files are separate; all nine directories accounted.
- [x] Pre-write limits include existing outputs, complete incoming bytes, hardlink/temp coexistence, missing directories and conservative native allocation; no unguarded row/font stream or descriptor forwarding.
- [x] Early admission precedes Torch; scientific publisher aliases bind only after adapters; first denial survives caught exceptions, cleanup and stale offline.json.
- [x] Parent process fence and cold readers bind the optional allowance without relabeling legacy evidence, weakening source authentication or changing scientific semantics.
- [x] Planned tests are tiny writer/process controls with separately admitted retained allocation. No implementation execution, tests or project imports occurred while preparing this subsection.
- [x] No claimed successful-run bound, new ledger/framework, native kernel quota, provider call or full-gate admission.

Ruling: the child installer is usable by isolated tiny controls with an explicit allowance, while only the validated existing production intent can authorize the actual diagnostic. This separates primitive testing from production launch authority without a hidden test-only bootstrap. The controller checked the installed font cache version/name, report numbering, and crash UUID/checkpoint grammar against their source. Parent process publication owns four names, but only its immutable intent exists during the child lifetime. The three tasks share interfaces in the order A -> B -> C; they must not be implemented concurrently. These are storage/process changes only; all scientific and local-only constraints above remain binding.


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

**Execution ordering ruling (2026-09-18):** the actual early qualification and
the first 13,488,540-byte engineering unit now passed, including readback,
receipt-authorized eviction and final residual accounting. Compact committed
evidence is in `docs/phase4-autonomous-eventflow.md`. Retained allocation is
6,195,077,120bytes; operational scratch is202,186,752bytes. Resume bounded Task5
slices only after their own source-derived finite admission. Do not require
all eight remaining audited batches to finish before a small integration slice
that fits: those batches remain local, charged and queued for further reclamation
when needed. Candidate identity and exact reviewed-source authority must be
refreshed before any later archival on changed source. No provider work or source
mutation may overlap an in-progress engineering eviction. The second batch was
stopped at its read-only preflight, without upload or a reservation, to preserve
that exclusion. This changes execution order, not retention or final gates.

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
