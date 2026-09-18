# Task6 prerequisite implementation report

Status: DONE_WITH_CONCERNS, ready for independent review, not provider use.
Review base: `045d234e49ac99a0c3ee4205d6677df97bb4d801`.
Controller-only setup status commit `ff874d6` is also above that base.
This implements the approved sequential prerequisite in the Task6 brief and
amendment `eaa5a73`; it does not finish Tasks5/6/7 or claim the full local gate.

## Implemented contract

- `archive/preflight.py` owns one closed engineering custody root and one direct
  operational child. `bootstrap_engineering_workspace(custody_root, workspace,
  policy, source_commit)` derives all allocations from descriptor-relative scans;
  no caller-supplied totals become allowances. The controller must declare the
  completed custody inactive before this call. The private create-only authority
  anchor binds device/inode identities, the exact operational path, six completed
  task names, actual executable digest and current checkout commit. A prior
  anchor blocks rebootstrap, including an interrupted bootstrap.
- Bootstrap scans before writing, accounts prior operational allocations and
  unconsumed reservations, authenticates bounded reverse-linked inventory pages,
  and includes the anchor, lock, directory, inventory and atomic descriptor peaks.
  An existing compatible ledger retains its baseline, reservations, bindings and
  remote history. Unknown fixture ledgers remain bytes, never real allowance.
  Initial read-only inspection found no real descriptor at repository, plan or
  plan/tmp roots. No real bootstrap was performed by this implementer.
- Ledger v2 adds a retained engineering record; `measure()` still returns the
  same seven categories. `retained_charge()` is separate from category arithmetic.
  Existing consumed-versus-remaining semantics stay intact. Non-emergency
  category allocation plus remaining reservation plus retained charge must fit
  the normal 6.5GiB default ceiling; all categories plus retained charge and the
  protected 2GiB reserve must fit 10GiB. No new capacity or emergency exemption.
- Frozen accounting is descriptor-relative/no-follow and hashes eligible dense
  single-link regular files. Unsupported entries are lstat-only, remain local and
  charged, and never become archive candidates. Linked non-directory inodes are
  charged once; aliases outside custody block. No malformed fixture owner file
  is parsed as authority. Mutable directories and eligible files are revalidated
  during scans, with exact frozen-page comparison before admission.
- `iter_engineering_candidates(budget)` discovers disjoint safe child roots only
  below `tmp/task-1`, `task-2`, `task-3`, `task-4`, `task-4-fix1`, `task-4-fix2`.
  Its bounded members are regular dense single-link files no larger than128MiB.
  Discovery is filesystem safety, NOT privacy or scientific validation.
- `EngineeringContentReview` requires the exact candidate ID, executable digest,
  ordered original paths and a producer-evidence digest. The controller must
  positively audit producer-format evidence and actual bounded metadata before
  authorizing those paths. An extension or negative substring scan is not enough.
  Private owner/config/ledger records remain exact local retained siblings. No
  bytes are sanitized and no codec/classification framework was introduced.
- `archive_engineering_candidate(budget, candidate, review, transport)` serializes
  operations under the precreated engineering lock, revalidates custody, derives
  capacity, reserves metadata/scratch and uses existing `seal_unit`, `archive_unit`
  and `evict_unit`. It compares sealed FileEntry values with reviewed values
  before any transport call, closing the review-to-seal replacement race.
  Diagnostic identity uses `engineering-<candidate_id>`, bootstrap source commit,
  review digest, frozen candidate identity, absent checkpoint hash and false
  checkpoint-committed status. It creates no scientific success/checkpoint claim.
- Only a complete verified receipt enables exact owned regular-file eviction.
  Pending state retains the old charge through partial unlink. Stable residual
  inventory publication, charge replacement and removal of the exact local
  reservation token occur together in one atomic ledger update. Directories,
  private siblings, old inventory pages and failed/ambiguous transfer history stay
  charged. `resume_engineering_eviction(budget, transport)` resumes only this
  durable pending operation, validates shared transport identity, and rederives
  bounds rather than trusting a persisted byte grant.
- A stranded admission is recoverable only for its exact candidate and a proven
  dead PID/create-time owner. The original before snapshot is retained; amount
  becomes at least consumed growth plus newly derived remaining requirement.
  Prior record bytes are retained in charged recovery metadata. Live/ambiguous
  owners fail closed. No broad stale-reservation clearing was added.

## Bounds, source authority and operational lifecycle

The frozen plan root excludes only the pinned operational child. During actual
bootstrap/operations, ALL other plan writes must stop, including reports,
review packages, Task5 fixtures and logs. They remain frozen afterward. Controller
logs and further permitted output go to reserved operational paths, including
`operational/logs`; this API does not write or silently fund harness logs.
The trusted graph is controller-declared stopped custody, the exclusive prelude
lock and stable no-follow inventory revalidation, not OS-wide writer detection.
Implementation tests/report were written before any real bootstrap.

Bootstrap source_commit is checked against actual checkout HEAD when attached,
then retained as immutable provenance. The anchor's executable_sha256 hashes the
actual package Python path/content closure, not a caller label. Candidate
discovery, content review and resume compare the current executable against the
bound source identity. Source_commit is not reinterpreted as the current HEAD on
every ledger read. Accounting remains available after later reviewed source
changes, but new archival operations cannot silently adopt changed executable
bytes. A future explicit **same-ledger source-authority transition** is remaining
work before such changed-source archival use; so is any authorized thaw/rebinding
transition. Neither exists in this slice. Rebootstrap/reset/new allowance is not
a workaround. Remaining Task5/6 output must use the same operational ledger and
reservations without changing the frozen external history.

Because ledger now lazily imports preflight, `train/pilot_evidence.py` adds
`archive/preflight.py` to STORAGE_PACKAGE_FILES and therefore the executing source
closure. The historical introduction chain is unchanged. The real source-chain
test independently compares authenticated package inventory with the required
executing inventory; it failed before this correction and passes afterward.

`_prelude_bounds` derives finite output bounds from selected FileEntry rows,
chunks/shards, bounded serializer fields, catalog record/node counts, existing
remote ledger entry_count, local reservations/completed intents, full retained
inventory repagination, fixed named control/directory closure and allocation
block size. Binary catalog tree maxima bound both run and operational catalogs.
Scratch includes catalog staging and one upload/readback pair. Metadata includes
receipt/proof/pending/eviction state, per-object reservations and atomic controls.
The original retained files remain charged concurrently. Existing 16MiB control
reader maxima are checked before output; policy/category/normal/physical checks
remain authoritative. Bootstrap includes two policy-page descriptor slots for a
compatible existing ledger, not an assumed tiny empty descriptor.

These bounds deliberately use `max(f_bsize, f_frsize,4096)`. On this APFS workspace
f_bsize is1MiB, so named-file/directory overhead can dominate a small candidate's
reservation. Full real-history bootstrap and per-candidate bounds have NOT been
qualified. If real admission fails, record its actual binding term (retained
inventory/catalog cardinality, reader cap, category cap, block overhead or
physical reserve), not a guessed spare-space grant. Repeated frozen hashing and
constant-memory ordered traversal trade speed for bounded memory; large flat
directories and linked-inode rescans can be expensive.

Qualification and engineering retention must use one profile/bucket/prefix
transport identity AND one `operational/control` remote ledger. Different run IDs
do not create fresh capacity. Engineering is labeled separately under the same
4.5TB aggregate history, never hidden in qualification's aggregate512MiB. An early
probe uses exact admitted bounds and a conservative sub-cap leaving later smoke
capacity; later smoke independently fits512MiB minus earlier retained charges.
No reset, cap increase or charge reduction is authorized here.

No wrapper or new public restore API was added. Minimal call path is bootstrap,
candidate iteration, strict positive review, then archive (or pending resume).
For real bounded download/restore, reuse supervisor `_ArchiveServer._lease`,
which already holds leases and performs bounded download followed by the reviewed
`bundles.restore_unit`; the focused roundtrip test calls that existing restore
primitive with local transport bytes. A qualification harness remains main-owned.

## Exact test command convention

Commands ran from the repository root using `.venv`, without dependency changes.
For each label below, the common shell command prefix was:

```sh
env PYTHONDONTWRITEBYTECODE=1 \
 TMPDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude/temp" \
 TMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude/temp" \
 TEMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude/temp" \
 SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude/archive" \
 SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 \
 .venv/bin/python -m pytest <selection> -q \
 --basetemp=.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude/<label> \
 -o cache_dir=.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude/cache-<label>
```

From cover-1 onward the prefix also contained `OMP_NUM_THREADS=1`.
Below, P means `tests/archive/test_preflight.py`, L means
`tests/archive/test_ledger.py`, and S means
`tests/pilot/test_pilot_source.py::test_real_introduction_chain_and_dirty_archive_helper`.
This substitution specifies commands without obscuring their unique retained
temp/cache paths. Counts/messages below are observed output, not inferred checks.

## Regression-first evidence

| Label | Selection | Observed output / reason |
| --- | --- | --- |
| red-1 | P | 9 failed; ImportError for missing preflight API, expected before implementation. |
| green-1 | P L | 21 passed in0.50s. |
| red-2 | P | 10 failed,9 passed; missing iter_engineering_candidates AttributeError. |
| green-2 | P | 17 passed,2 failed; canonical_json_bytes given a string instead of a mapping. |
| green-3 | P | 17 passed,2 failed; legitimate APFS ancestor-directory metadata changed during unlink. |
| green-4 | P | 19 passed in1.77s. |
| red-3 | P -k 'prior_unconsumed or stored_byte_grant' | 2 failed,21 deselected; bootstrap wrote before checking prior reservations, and resume trusted a fabricated zero grant. |
| green-5 | P | 21 passed,2 failed; strict JSON tuple/list decoding in resume. |
| green-6 | P | 22 passed,1 failed; logical boundary test depended on actual free disk. |
| green-7 | P -k 'prior_unconsumed or stored_byte_grant' | 2 passed,21 deselected in0.58s. |
| red-4 | P -k invented_source | 1 failed,25 deselected; DID NOT RAISE for an invented source commit. |
| green-8 | P | 29 passed in4.61s. |
| red-5 | P::test_dead_prelude_owner_resumes_exact_stranded_reservation S | 2 failed in7.79s; stranded metadata owned another admission, and current source inventory could not satisfy independent source closure. |
| green-9 | P L S | 43 passed in14.63s. |
| red-6 | P::test_source_changed_between_review_and_seal_never_reaches_transport | 1 failed in0.59s; expected zero creates, actual10. |
| green-10 | P L | 45 passed in6.73s. |
| green-11 | P L S | 46 passed in16.54s; final source, including full bootstrap descriptor peak and admitted precreated lock. |

Debugging used actual failure traces and small file/stat inspections, not broad
retries. APFS safe-directory st_nlink changed4→3 and size128→96 after unlink;
only known pending ancestors may change mutable metadata, while inode/device/mode
remain exact and all unaffected members must match. Strict persisted review JSON
uses model_validate_json. The isolated logical reservation boundary test mocks
statvfs generosity; separate physical-space tests prove real admission logic.
Transfer-fault tests assert transport was actually reached, avoiding false green
from an unrelated pre-upload exception. Final regressions also cover a live
stranded owner and physical peak refusal before sealing/upload.

## Stable affected-interface covering, then focused final covering

Controller tracked edits were frozen before `cover-1`. Its selection was exactly:

```sh
tests/archive/test_preflight.py tests/archive/test_ledger.py \
tests/archive/test_transport.py tests/archive/test_operational.py \
tests/archive/test_session.py tests/archive/test_supervisor.py \
-k 'not supervisor_runs_and_joins_real_closed_report_child and not supervisor_abort_stops_nested_diagnostic_without_touching_other_processes'
```

Observed output: `118 passed, 1 skipped, 2 deselected in21.39s`.
The skipped test was
`tests/archive/test_transport.py::test_r2_download_native_dev_fd_open_stays_on_displaced_inode`
(line797), because this managed sandbox denies the child `/dev/fd` reopen.
It is NOT a native descriptor qualification pass. The two excluded tests invoke
numerical/offline/abort-diagnostic child paths outside this narrow covering scope.
No numerical suite or full `make verify` was run.

Later self-review corrections received red-5/green-9 and red-6/green-10, then the
final focused green-11 command above. The earlier118-pass run is not represented
as a fresh broad run over those later changes. Final focused output was pristine:

```text
..............................................                           [100%]
46 passed in 16.54s
```

Final lint and formatting commands (all six source/test files):

```sh
env RUFF_CACHE_DIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude/ruff" \
 .venv/bin/ruff check src/silent_cascade/archive/ledger.py \
 src/silent_cascade/archive/types.py src/silent_cascade/archive/preflight.py \
 src/silent_cascade/train/pilot_evidence.py tests/archive/test_preflight.py \
 tests/pilot/test_pilot_source.py
# All checks passed!
env RUFF_CACHE_DIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude/ruff" \
 .venv/bin/ruff format src/silent_cascade/archive/ledger.py \
 src/silent_cascade/archive/types.py src/silent_cascade/archive/preflight.py \
 src/silent_cascade/train/pilot_evidence.py tests/archive/test_preflight.py \
 tests/pilot/test_pilot_source.py
# 6 files left unchanged
git diff --check
# exit0, no output
```

## Custody, self-review and remaining conditions

Final pre-report allocation: whole plan5,988,628KiB; this task35,292KiB, below the
6GiB warning and64MiB fixture target. Every unique basetemp/cache/archive fixture
was preserved. Only newly created synthetic test payloads were evicted by valid
double-receipt test scenarios; no old engineering/scientific evidence was removed.
One earlier Ruff format invocation omitted explicit cache routing. The controller
inspected the ordinary inactive repository `.ruff_cache` (112KiB, no symlinks)
and moved it without overwrite to
`tmp/task-6-prelude/retained-repository-ruff-cache`; inode21833522/size160 unchanged,
original path absent. Nothing was deleted/exempted; the move is reversible and
all allocation is counted. Every later Ruff invocation was explicitly routed.

Self-review corrected: prior-reservation bootstrap admission, persisted fake
capacity on resume, source-commit labeling, legitimate APFS directory changes,
dead-owner reservation restart, executable inventory closure, review/seal race,
full prior-descriptor bootstrap peak and pre-admission lock creation. Regression
coverage uses tiny real files and transport doubles, preserves private siblings,
roundtrips exact bytes, retains failed reservations, and never validates malformed
science as successful science.

Concern: preflight grew to approximately1,040 lines, largely explicit serializer
bounds, pinned inventory handling and closed interruption recovery. This was
reported to the controller; it remains one planned file, without a new general
registry/framework or unapproved split. Conservative bounds/performance require
real feasibility evidence. Independent review is still required before operation.

Remaining main-owned prerequisites: independently review this change; freeze
all non-operational plan writes; positively audit exact candidate content; derive
and record actual bootstrap/unit bounds; establish one real local/remote authority;
perform native `/dev/fd`, bounded64MiB provider upload/readback/streaming restore,
shared-counter before/after, failed/collision and process-kill qualification while
retaining stranded scratch. Same-content retry collision is sufficient; no corrupt
provider object planting is needed. Later smoke, Task5 recovery, full local gate,
source-authority transition and any thaw transition remain separate work.

Files changed: archive/ledger.py, archive/types.py, new archive/preflight.py,
train/pilot_evidence.py, new tests/archive/test_preflight.py,
tests/pilot/test_pilot_source.py, and this report. No setup document, dependencies,
CI, worktree, provider configuration or credentials were changed by this agent.

## Fix round1 — bootstrap admission and measured feasibility defects

Fix base43cb193f053f3ba8d82c2ce46a37dabb31f26092. Scope is the controller's
task-6-prelude-fix1-brief.md, independent Important finding, and three measured
feasibility defects. Status: DONE_WITH_CONCERNS pending fresh independent review.
This section supersedes the earlier I/O-size rounding and repeated-inventory
performance descriptions; earlier test evidence remains historical, not merged.

### Corrections and root causes

1. Bootstrap previously checked only normal/physical capacity before publishing
   permanent authority/pages/v2 descriptor. Metadata category denial happened
   afterward. It now separates external retained authority overhead from the
   operational metadata peak and checks existing metadata allocation plus that
   derived peak before any bootstrap publication. Any existing positive metadata
   reservation is owned by another admission and blocks bootstrap, including a
   fully consumed reservation whose ownership remains. Existing scratch-only
   reservation migration is preserved. For compatible ledgers, workspace.lock is
   held through ownership/admission checks and publication, avoiding a competing
   reservation race. Old descriptor, authority absence and page absence are
   asserted after fresh/compatible refusals. No new category or allowance.
2. `_accounted_inventory` previously rescanned the entire tree once per linked
   pathname. It now uses exactly two streaming traversals and a finite4096-inode
   linked-only table: first collect identity/count/lexicographic-first-path,
   reject external aliases/capacity/invalid links; then revalidate and emit the
   original deterministic forward or reverse rows. Only the first path receives
   physical charge, all linked members remain ineligible, and a changed/missing
   group fails closed. No all-path map, persistent cache or scratch inventory.
3. `_pages` previously serialized each growing page twice per new record. It now
   counts each canonical row once, incrementally adds comma/header bytes, and
   serializes the complete page once at publication. Exact bytes, hashes and
   boundaries are unchanged, including UTF-8/escaping, null→hashed next-pointer
   width, entry limit, exact byte limit, empty input and oversize first/later rows.
   A512-row test observed66,556 serialized row-visits before the correction;
   corrected work stays within the independent linear3N bound.
4. `_allocation_unit` now derives physical rounding from validated f_frsize,
   independent of f_bsize. It rejects nonpositive/nonintegral/out-of-range units,
   units inconsistent with512-byte stat allocation counters, and invalid available
   block counts. No fallback to a guessed unit. Both bootstrap and archival
   formulas retain every prior inode/directory/atomic/old+new/catalog term.
   [Apple's statvfs manual](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/fstatvfs.3.html)
   identifies f_frsize as allocation granularity and f_bsize as preferred I/O
   request size; on this APFS volume those differ4096 versus1,048,576.

### Exact fix-round commands and RED/GREEN

All tests used this common command, substituting the exact selection and unique
label from the table. Only fresh counted fix1 fixtures were created:

```sh
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
 TMPDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude-fix1/temp" \
 TMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude-fix1/temp" \
 TEMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude-fix1/temp" \
 SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude-fix1/archive" \
 SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 \
 .venv/bin/python -m pytest <selection> -q \
 --basetemp=.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude-fix1/<label> \
 -o cache_dir=.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude-fix1/cache-<label>
```

P/L/S retain the explicit file/node meanings defined above.

| Label | Selection | Actual output and explanation |
| --- | --- | --- |
| red-1 | P -k 'bootstrap_metadata_peak or existing_metadata_owners' |3failed,33deselected in0.56s. Fresh refusal left5new files; compatible atomic peak and metadata ownership cases DID NOT RAISE. |
| green-1 | P -k 'bootstrap_metadata_peak or existing_metadata_owners or compatible_attach' |4passed,32deselected in0.50s. |
| red-2 | P -k linked_inventory |3failed,2passed,36deselected in0.35s. Ten traversals exceeded2; capacity and invalid-link cases DID NOT RAISE. |
| green-2 | P -k 'linked_inventory or linked_sparse or unsupported_entries' |7passed,34deselected in0.52s. |
| red-3 | P -k 'inventory_pages or page_serialization' |1failed,4passed,41deselected in0.30s.66,556 serialized row-visits exceeded3*512. |
| green-3 | P -k 'inventory_pages or page_serialization or multi_page' |6passed,40deselected in0.43s. |
| red-4 | P -k 'preferred_io or allocation_geometry' |5failed,46deselected in0.87s. I/O hint incorrectly denied small bootstrap/changed archive bounds; invalid geometry lacked its own fail-closed validation. |
| green-4 | P -k 'preferred_io or allocation_geometry or bootstrap_metadata_peak or existing_metadata_owners' |8passed,43deselected in0.75s. |
| green-5 | P -k actual_atomic_archive |1passed,51deselected in10.60s. Two real small local-double archives, second with nonzero shared operational history. |
| cover-1 | P L S |65passed in27.93s, no skips/deselections/warnings. Final source-bound covering after controller freeze. |

The atomic-publication test observes actual category allocations at real file and
directory fsync boundaries, including staged and published objects. For each of
two archives it independently asserts positive observed metadata/scratch growth
does not exceed the pre-derived respective bounds. The second uses the same
remote ledger with an observed nonzero operational entry_count, not a new allowance.
Only the transport endpoint is a local double; sealing, publication, receipts,
eviction, ledger measurements and fsync remain real. This is not provider/native
descriptor qualification. No unchanged neural suites/full gate were repeated.

Final scoped commands:

```sh
env RUFF_CACHE_DIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude-fix1/ruff" \
 .venv/bin/ruff check src/silent_cascade/archive/preflight.py tests/archive/test_preflight.py
# All checks passed!
env RUFF_CACHE_DIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude-fix1/ruff" \
 .venv/bin/ruff format --check src/silent_cascade/archive/preflight.py tests/archive/test_preflight.py
# 2 files already formatted
git diff --check
# exit0, no output
```

An intermediate scoped Ruff check found one regex string needing a raw-string
prefix; corrected before final covering. No default cache was used this round.
Before the final report append, counted fix1 output8,708KiB; wholeplan5,997,512KiB,
below64MiB additional output and the6GiB warning. Earlier evidence stayed intact.

### Updated prospective formulas; not real admission

Using only the earlier measured representative3,098,274-byte tensor,132pages and
46,484,267-byte inventory bound from prelude-feasibility-prep.md, the corrected
current source formulas give the following values. No new real-history snapshot,
bootstrap, remote ledger or provider call was performed. Current executable formula
was evaluated in memory, replacing absent-ledger readers with the explicitly
listed prospective counts, exactly as the prior read-only diagnostic. Local
reservation/completed-intent counts are0 scenarios, not observed real counters.

| Prospective prior operational entries | Metadata | Scratch |
| --- | ---: | ---: |
|0 |88,029,457 |7,282,855 |
|34 |92,290,711 |8,750,193 |
|100 |100,562,557 |11,672,265 |

Bootstrap becomes81,177,387bytes aggregate output, of which81,152,811 is operational
metadata peak. With the historical retained6,132,498,432bytes, the normal inequality
would leave765,646,037bytes after bootstrap peak. These stale-input scenarios are
not executable authority; reviewer/report growth, real remote history, remaining
reservations and fresh frozen identity must be measured later. The earlier
prior34 scratch failure was specifically the incorrect1MiB I/O-size rounding;
its correction does not reset any usage or increase the256MiB cap. Existing control
reader limits and eventual cardinality limits remain unchanged.

### Remaining lifecycle concern and interface assessment (not implemented)

Controller's separate read-only grouping audit reports798 positive files in369
actual candidate groups; top10/20/50 free approximately479/625/891MB. Even the six
old roots alone have95,763rows/96pages/~31.6MB per generation; the full-plan
inventory is larger. Current reverse-linked pages include the previous page hash,
so an early changed ancestor row can change every downstream page hash. Keeping
all generations while republishing the whole chain per tiny unit can exhaust the
1GiB metadata category before useful reclamation. Geometry/performance fixes do
not solve this separate load-bearing lifecycle problem. No page deletion, quota
increase, source-authority reset or speculative catalog rewrite was added.

For the controller's requested next-interface comparison:

- **Stable-range immutable leaves plus a bounded authenticated index:** smaller
  change to the existing single-unit recovery contract. Persist original path-range
  boundaries (not shifting every1000 surviving rows), store leaf payloads without
  predecessor hashes, and atomically reference an authenticated bounded index.
  Unchanged leaves retain their hashes; only changed leaves and the small index
  become a new generation. All old pages/indexes remain present and charged.
  This mainly changes `_snapshot` publication, `_stored_records` authentication
  and page/index capacity derivation, while preserving one pending receipt,
  exact retained-sibling checks and atomic per-unit charge update. An explicit
  bounded index capacity/schema and old-generation reading policy are required;
  neither a mutable cache nor unauthenticated range hints are sufficient.
- **Bounded multi-unit reclamation followed by one residual publication:** may
  reduce generations, but changes the operational/recovery contract more widely.
  It needs an authenticated cohort of reviews/receipts, union-of-removals residual
  validation, admission covering all simultaneous controls and stranded scratch,
  and deterministic interruption recovery across completed/partial units. Current
  discovery refuses pending eviction and candidates bind the frozen snapshot, so
  this cannot be achieved merely by looping existing archive calls. Old retained
  charge must remain until the complete cohort is authenticated; no early credit.

Recommendation for a separately approved correction: investigate stable ranges
with a bounded authenticated index first. It retains the existing per-unit trust
boundary and charges rather than introducing a multi-receipt pending state machine.
Existing content-addressed create-only helpers can be reused; the public remote
catalog has different strict record schemas and should not be reclassified as a
private stat-inventory codec without explicit design. Exact changed-leaf/index
growth and representative369-group feasibility still need proof, not assumption.

Self-review: compared admission with existing reservation ownership arithmetic,
checked external authority remains retained rather than falsely categorized,
checked reverse linked rows and unchanged canonical page hashes, inspected staged
allocation observations, and reran the real source-inventory closure. Changed
files this round are only preflight.py, test_preflight.py and this appended report.
No real authority/provider/bootstrap/history eviction occurred. Independent review
and the retained-generation lifecycle correction precede real operational use.
