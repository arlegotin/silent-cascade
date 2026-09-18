# Early qualification implementation handoff

Status: DONE_WITH_CONCERNS. Implementation commit: `6090118`
(`Add bounded early R2 qualification driver`). Review BASE remains
`bb65598a64359ac74cf672583abab432621cd7af`; intervening controller documentation
commits are not implementation changes. This report is preparation for independent
review, not provider qualification, scientific acceptance, or bootstrap authority.

## Implemented contract

The private `archive/_qualification.py` implements the accepted controller entry,
including explicit `debug_logical_root`. Its exact-type R2 transport check prevents
local doubles from returning a provider result. It authenticates the source commit,
executing package digest, existing engineering authority, policy, category handoff,
precreated locks, and canonical remote ledger. It does not initialize real local or
remote history, create credentials/configuration, construct `_ArchiveServer`, expose
a CLI/public restore API, or change scientific behavior.

The driver derives two-unit bounds before reserving/output, holds one parent local
reservation across source publication, sealing, all archive children, retries,
restores, and result publication, and preserves failed actual output/remote history.
The production recipe writes its one fixed 64 MiB source directly. Debug copying
uses exactly the completed commit's owned members, strips only the explicit prefix
when opening the physical source, preserves full logical paths in the copy, checks
hash/size/source identity, and fsyncs. Borrowed/private siblings are excluded.

Canonical snapshots compose `_opened_ledger` under the existing archive lock,
require the lock/ledger hierarchy to exist, and report both pending leaves without
recovery or counter rewriting. Receipt restore composes the existing unit lease,
unit/receipt/catalog/active-authorization validators, ordered payload descriptors,
one-chunk bounded downloads, identity-safe temporary removal, and `restore_unit`.
Existing destinations refuse before payload downloads.

Archive execution uses spawn and one-way pipes. Each child establishes its own
session and completes a parent-verified process-group handshake before transfer.
The first successful delegated create publishes a bounded marker and blocks before
returning. The parent sends real SIGKILL and verifies signal exit. A fresh child
observes and rethrows the delegate's actual FileExistsError; the existing archive
code authenticates readback. A further fresh child proves stable receipt/reserved
bytes and positive revalidation downloads. Debug archive shares the same history.

Child calls aggregate counts, bounded-byte totals, local maxima and a hash chain in
memory. At most 32 permanent phase summaries (including the marker), one replaced
progress file, 16 KiB replies and a 64 KiB result are admitted. Raw exceptions,
provider output, transport configuration, keys and original paths are not persisted
in these records. Parent controller failures suppress the exception chain.

All owned child/group termination precedes reservation unwinding. The controller
explicitly approved fail-stop exit 70 if bounded kill/reap cannot prove termination:
a best-effort sanitized event is retained and the parent exits without unwinding
its durable reservation. A dedicated isolated subprocess test proves this record
remains. A real descendant-process test proves group termination covers a spawned
provider-like child. This is not a generic process manager.

The final strict result is measured while staged, converges its own allocation
record, and publishes create-only using the same inode. There is no local cleanup
or evidence eviction workflow.

## Shared sizing refinement

The controller approved the minimum extraction in `preflight.py`:

`_archive_unit_bounds(*, policy, logical_root, selected, run_id, kind,
episode_groups, prior_operational_records, local_reservations, completed_count,
prior_run_records=0) -> _ArchiveUnitBounds`.

The frozen result holds inventory, manifest, run/operational catalog, receipt,
pending, object/node/cardinality/serializer maxima, and remote-byte terms. The
original `_prelude_bounds` retains engineering ledger, retained-inventory,
eviction and allocation-rounding composition. The explicit run ID and episode
group serializers avoid the old engineering identity/diagnostic-only assumptions.
The second qualification run generation includes all first-unit ownership/unit
records, independently of the operational-head count. No deduplication credit is
used. Scratch includes the extra killed chunk, retained restores remain additive,
and filesystem `f_frsize` supplies validated allocation rounding. Call deadlines
include bounded tree depth and repeated operational generations.

On one identical synthetic engineering candidate, executing the original BASE
`_prelude_bounds` and the extracted implementation returned exactly:

```text
engineering equivalence: {'metadata': 39654585, 'scratch': 3443286}
```

The comparison compiled the original function from `git show BASE:.../preflight.py`
in memory, reused current unchanged dependencies and the same real synthetic
ledger/candidate, and asserted exact equality. No old source file was written.

The focused small two-unit case (160-byte probe, 5-byte debug member, allocation
unit 4096, prior operational count 7, local reservation count 3) derives:

```text
spool=57344 cache=57344 pinned=0 metadata=20047970
scratch=5618775 logs=155648 emergency=0 retained_bytes=0
remote=4725972 transport_calls=1907968
```

These are conservative source maxima, not observed byte usage or real-provider
admission. The test measures actual category peaks at every file/directory fsync
during two archives on one nonzero (11-byte accounted) remote history and asserts
all observed growth fits the derived category bounds. Other tests refuse a
256 MiB+1 remote bound and a deliberately underestimated scratch grant without
output growth. The real group's exact bound remains a controller runtime check.

A self-review question about `episode_pack` being two characters longer than
`diagnostic` required no edit: canonical maximum record size is 24,981 bytes
(another record dominates), versus 24,962 bytes for the worst episode-pack unit
record. Existing shared maxima therefore already cover it.

## Verification and RED/GREEN evidence

All commands ran locally. Common environment for pytest was exactly:

```sh
Q="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/early-qualification"
TMPDIR="$Q" TMP="$Q" TEMP="$Q" PYTHONDONTWRITEBYTECODE=1 \
SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$Q" \
SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 \
.venv/bin/python -m pytest <selection> -q -p no:cacheprovider \
  --basetemp="$Q/<unique-name>" <extra-options>
```

Each listed basetemp was unique and is preserved. Pytest caching was disabled;
Ruff's explicit cache is under the same counted root. No full-size probe fixture
was retained: all 64 recipe blocks were streamed directly through SHA-256,
asserting exactly 67,108,864 bytes and
`e04dc3863c905cbf429e51f9a248159ccbc34688763f871fd20d2dfd8964df34`.

| Selection / basetemp | Observed output |
| --- | --- |
| `tests/archive/test_qualification.py`; `pytest-red-01` | 14 failed in 1.36s; missing `_qualification` module, expected initial RED |
| Same; `pytest-green-01` | 14 passed in 1.22s |
| Same, `-k 'sizing or recipe' --tb=short`; `pytest-red-02` | 2 failed, 14 deselected; missing `_archive_unit_bounds` / `_probe_blocks`, expected RED |
| Same selection; `pytest-green-02` through `-04` | Strict fixture/model errors exposed actual run-ID limit 256 and EpisodeGroup required/max-size field; corrected against existing schemas |
| Same selection; `pytest-green-05` | 2 passed, 14 deselected in 0.25s |
| Same, `-k spawn --tb=short`; `pytest-red-03` | Fixture root `.` rejected for diagnostic; corrected fixture to `probe/` |
| Same selection; `pytest-red-04` | 2 failed, 16 deselected; missing spawn helper, expected RED |
| Same selection; `pytest-green-06` | 1 failed, 1 passed; missing first-create marker |
| Same, `-k observed_transport --tb=short`; `pytest-diagnose-01` | 1 failed; exact cause was canonical JSON helper accepting mapping/model, not tuple |
| Same, `-k 'spawn or observed_transport' --tb=short`; `pytest-green-07` | 3 passed, 16 deselected in 3.10s; genuine SIGKILL/resume/repeat and sanitized failure |
| Same, `-k 'two_unit or projection or evidence_limits or controller' --tb=short`; `pytest-red-05` | 3 failed (missing bounds/copy/controller), 1 passed (existing event limit) |
| Same selection; `pytest-green-08` | 4 passed, 19 deselected in 0.35s |
| Same, `-k 'bound_refusal or reordered' --tb=short`; `pytest-contract-01` | 2 passed, 23 deselected in 0.30s; added contract coverage |
| Same, `--tb=short`; `pytest-cover-01` | 25 passed in 3.48s |
| Same, `-k 'descendant or unreapable' --tb=short`; `pytest-lifetime-01` | 2 passed, 25 deselected in 0.69s |
| Same, `--tb=short`; `pytest-cover-02` | 1 failed, 26 passed in 7.16s; measurement race described below |
| Same, `--tb=short`; `pytest-cover-03` | 27 passed in 8.95s |
| `tests/archive/test_qualification.py tests/archive/test_preflight.py tests/archive/test_transport.py tests/archive/test_bundles.py tests/archive/test_ledger.py --tb=short`; `pytest-cover-final-01` | 148 passed, 1 skipped in 24.84s |
| `tests/pilot/test_pilot_source.py --tb=short -rs`; `pytest-source-final-01` | 5 passed in 12.57s after implementation commit |

The measurement-race RED was `FileNotFoundError` in `ledger.measure()` when the
parent scanned a remote-ledger atomic `.pending.json.*.tmp` while the archive child
renamed it. Root cause was observing during active child mutation, not a missing
archive output. Parent observations now occur only before spawn, at the blocked
first-create barrier and after reaping; children observe synchronously before/after
delegated calls. Parent progress replacement and child measurements coordinate
through the existing workspace lock. No ledger behavior was changed. The combined
GREEN above exercises the corrected boundary.

The sole skipped case was
`test_r2_download_native_dev_fd_open_stays_on_displaced_inode`, whose exact reason
is `managed sandbox denies child /dev/fd reopen`. The native unsandboxed descriptor
test and then the real R2 transport qualification must cover this later; a local
double does not discharge it.

Scoped Ruff check and format verification passed for all four changed source/test
files; `git diff --check` passed. Commands used
`RUFF_CACHE_DIR="$Q/ruff-cache" .venv/bin/ruff check <four-files>` and the same
prefix with `ruff format --check <four-files>`; output was `All checks passed!`
and `4 files already formatted`. Initial formatting diagnostics were corrected
with Ruff formatting and ordinary scoped edits before covering/commit.

## Files, source closure and self-review

- `src/silent_cascade/archive/_qualification.py`: private driver/compositions/models.
- `src/silent_cascade/archive/preflight.py`: approved shared pure sizing extraction.
- `src/silent_cascade/train/pilot_evidence.py`: one additive literal executing-source entry.
- `tests/archive/test_qualification.py`: 27 focused cases, local doubles/small files only.

`HISTORICAL_REQUIRED_PACKAGE_FILES` is unchanged. The private module is included in
the executing package closure and naturally in `_executable_digest`. Source tests
passed after committing the module. No package export, dependency, CLI, supervisor,
transport primitive or ledger context-manager change was made.

Self-review covered the two-unit same-run catalog bound, retry residue, ordered
receipt payload binding, preserved original logical names, child/process-group
reservation lifetime, exception sanitization, bounded evidence and the measurement
race. The accepted one-module design is 1,177 formatted lines; this is a review
concern about comprehensibility, not a reason to introduce an unapproved framework
or split. No independent review was dispatched by the implementer.

## Remaining obligations and limitations

No real provider request, profile/config/credential inspection, production bootstrap,
original debug evidence read/copy, numerical training, evidence cleanup, remote
deletion, bucket listing or scientific acceptance claim occurred. The final provider
result path was not executed with a double or a production bypass. Native descriptor
behavior and actual R2 412/readback/restore proof remain pending independent review
and controller admission/execution. The real copied-content audit, admitted output
root/category bindings, exact source authority, bootstrap and later smoke admission
remain controller-owned.

All new fixtures/caches stayed under the prescribed preserved qualification root;
after source covering/equivalence checking it occupied 36,792 KiB, below the initial 64 MiB allowance.
The complete `make verify` repository gate was not run under this task's finite
fixture allowance; the executed covering scope and native skip are explicit above.
No claim is made that retained qualification plus a later smoke fits 512 MiB:
the controller must independently establish `Q_actual + S_bound <= 536870912`.
