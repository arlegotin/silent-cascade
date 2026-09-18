# Task6 prelude: stable retained-inventory lifecycle

Status: DONE for this implementation slice, pending independent review and real
prospective feasibility. Review base: `5552cb9825afe3cecc4ed4b326a990212e2005c1`.
The controller's intervening `1e7cec9` setup-status-only commit is part of the
review package; this implementer did not edit that document. The implementation
commit containing this report is identified in the final handoff.

No provider calls, credentials, real operational bootstrap, old evidence changes,
numerical execution, dependency changes, full `make verify`, or delegated work
were performed. All archival operations below use a local directory transport
and newly created, small genuine fixture files. They are not actual archive or
full-gate qualification evidence.

## Implemented contract

The accepted contract in `task-6-prelude-lifecycle-brief.md` was followed, including
the controller's explicit removed-path authority and durable-descriptor rulings.

`archive/_retained.py` owns bounded serialization and authentication only. The
existing `preflight.py` still owns no-follow traversal, content/source custody,
admission, one-unit receipts, pending eviction, and the atomic ledger update.
No public catalog/scientific schema, codec, multi-unit transaction, registry or
second allowance was added.

Private v2 encodings:

- Leaf: `schema_version=phase4-engineering-leaf-v2`, `partition`, `lower`, `upper`,
  `original_entries`, `entries`. Original ordinal, bounds and count never change,
  including when a partition becomes empty. There is no predecessor leaf hash.
- Index: `schema_version=phase4-engineering-index-v2`, `next`, `first_partition`,
  `leaves`. Each reference has `partition`, `lower`, `upper`, `original_entries`,
  `sha256`, `encoded_bytes`, `entries`, `allocated`. Only these small index pages
  are reverse-linked; unchanged leaf identities are reused.
- Snapshot: `schema_version=phase4-engineering-snapshot-v2`, `head`, `index_pages`,
  `pages`, `layout_sha256`, `original_entries`, `entries`, `allocated`,
  `encoded_bytes`. `pages` is the immutable partition count. `encoded_bytes`
  counts the current leaves and index only, not historical files or the snapshot
  descriptor itself; physical ledger measurement counts all retained generations.
- Objects are create-only under `retained/leaves/<sha>.json`,
  `retained/index/<sha>.json`, and `retained/generations/<sha>.json`.
  The last is canonical snapshot JSON, persisted before the atomic ledger head
  change. It is not a mutable registry or independent authority. Even an
  unreferenced interrupted descriptor remains physically charged.

There are at most 4096 original partitions and 4096 index pages. Each page/leaf
respects the existing policy's finite byte and entry limits (maximum 16 MiB and
1000 entries). A bounded reference table and one logical leaf/page are used,
never an all-records map. Updates reuse the reference table in place and compute
publication bounds with a streaming reference pass. Canonical encoding, exact
keys/types (including rejecting bool-as-int), digests, byte/row/allocation totals,
contiguous ordinals, disjoint ordered ranges, leaf layout, terminating index
chain, duplicate/missing objects and policy limits are checked. Paths compare by
component tuples, with `.` the empty tuple, matching DFS rather than flat string
order. Thus `a/z` precedes sibling `a-foo` correctly.

Private APIs are `build_snapshot(records, policy, *, control_dir=None,
previous=None, removed_paths=(), publish=False)`,
`read_records(control_dir, snapshot, policy, *, reverse=False)`,
`publication_bound(control_dir, snapshot, policy, selected_paths)`, and
`initial_bound(records, policy)`. Record readers must be exhausted before their
snapshot is accepted. Historical descriptors can be read from their immutable
generation objects and passed to `read_records`; callers need not retain obsolete
ledger heads in memory.

For updates, `removed_paths` comes from the validated pending receipt, not from
inferring absent rows. The builder fully compares old and current rows, rejects
all unlisted removals/additions/ordinary-file changes, and permits only selected
ancestors' mutable stat fields while preserving path/device/inode/mode. Before
publishing any changed leaf it checks membership in the pre-derived affected
set. Both streams are exhausted before the existing atomic head/charge update.
The full outer before/after residual validation is retained. Failed partial
publications keep the old retained charge, pending receipt and reservation;
resumption consumes the same exact pending operation.

Authority v2 pins `inventory_schema=phase4-engineering-snapshot-v2`. Engineering
state retains immutable `initial_snapshot` and current `snapshot`, requiring
matching original layout/counts. V1 authority or snapshot fails closed with an
explicit same-ledger-transition-required error. There is no automatic migration,
rebaseline or reset. No real authority exists to migrate; old synthetic fixture
ledgers remain opaque retained bytes.

## Bounds and source closure

Initial leaf partitioning reserves worst textual widths for mutable ancestor
directory fields. Bootstrap's no-write initial bound includes all prospective
rows, leaf headers, bounded index references and a separate generation descriptor.
The index bound sums serialized reference sizes plus worst headers, and derives
page capacity from maximum reference width and the policy caps. This covers
changes in pointer and first-partition widths without depending on a guessed
page grouping.

Per-unit publication authenticates existing references and leaves, derives the
affected set from exact selected paths plus ancestors, and includes only those
potential new leaves plus the bounded new index and descriptor. Untouched leaf
bytes are not charged as new output. Existing physical metadata still includes
every historical generation and incomplete publication. Existing metadata,
scratch, directory, allocation-rounding, atomic replacement, pending/current
ledger and fixed-control terms remain. Bootstrap additionally accounts for the
three new private subdirectories; per-operation directory terms include them.
`snapshot.encoded_bytes` intentionally excludes descriptor size to avoid a
recursive size/hash definition.

The seven-category measure/check and reservation-consumption arithmetic are
unchanged. No quota, normal ceiling, reserve, remote budget or allowance changed.
The new module is in `STORAGE_PACKAGE_FILES` in `train/pilot_evidence.py`, hence
the current executing package/source closure and prelude executable digest.
Historical source tuples are untouched. Existing source provenance remains:
bootstrap source commit/root identities are immutable provenance, while the
current frozen executable digest is checked at live operation boundaries.
Subsequent reviewed executable changes do not authorize rebootstrap/reset; any
required same-ledger source-authority transition remains explicit future work.

The lifecycle also retains original usage restrictions: controller-declared
completed custody, exclusive prelude lock, no-follow descriptor/live inventory
validation, and frozen non-operational plan output during real operations.
Operational logs/output must use the existing operational allowance. Frozen
external history remains frozen afterward; any thaw/rebinding is an explicit
same-ledger transition. Operational qualification and history archival reuse the
same transport profile/bucket/prefix/control ledger, distinguished by run and
custody identities, never another remote allowance. Filesystem-safe discovery is
not privacy certification: source/ordered-path-bound positive content custody is
still required; private owner/ledger/config records remain local retained
siblings without byte edits.

## Reproducible commands and TDD evidence

All commands ran from the repository root with the existing `.venv`. The shared
prefix below defines the exact environment used for every pytest invocation;
`RUN` expands the command with the unique recorded basename/cache. It is a
notation for the recorded commands, not an installed script or new API.

```sh
LIFE="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude-lifecycle"
RUN() {
  label="$1"
  shift
  env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
    TMPDIR="$LIFE/temp" TMP="$LIFE/temp" TEMP="$LIFE/temp" \
    SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$LIFE/archive" \
    SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 \
    .venv/bin/python -m pytest "$@" \
    --basetemp="$LIFE/$label" -o cache_dir="$LIFE/cache-$label"
}
```

Exact runs and relevant output:

```text
RUN red-1 tests/archive/test_retained.py -q
5 failed in 0.33s
ImportError: cannot import name '_retained'
```

The first five tests required stable reuse, beginning/middle/end removals,
historical readability, empty-partition DFS semantics and refusal of unauthorised
live differences. They failed before the private implementation existed.

```text
RUN green-1 tests/archive/test_retained.py -q
5 passed in 0.57s

RUN red-2 tests/archive/test_retained.py -q
2 failed, 20 passed in 1.00s
leaf_boolean_layout: DID NOT RAISE StorageBlocked
bootstrap binding: KeyError: 'initial_snapshot'
```

This exposed Python's bool/int equality accepting an authenticated forged leaf
layout and the not-yet-integrated immutable initial snapshot. Explicit numeric
type checks and preflight v2 binding corrected both.

```text
RUN red-3 tests/archive/test_retained.py -q -k historical_generations
1 failed, 22 deselected in 0.29s
assert len(generations) == 3  # actual durable descriptors: 0
```

The test discards caller-held historical descriptors before reconstructing old
generations from disk. This established the descriptor-preservation gap; the
controller approved the exact generation-object contract before implementation.

```text
RUN green-2 tests/archive/test_retained.py tests/archive/test_preflight.py -q
75 passed in 19.11s

RUN green-3 tests/archive/test_retained.py -q -k 'affected_set or interrupted_publication'
6 passed, 23 deselected in 4.41s

RUN green-4 tests/archive/test_retained.py -q -k 'index_wire or index_escaping or serialization_work'
5 passed, 29 deselected in 0.24s
```

The interruption tests cover before/after leaf publication, after index,
after generation descriptor, and before atomic ledger head. Each asserts old
charge/head, retained pending evidence, preserved historical files and exact
same-ledger resume. The affected-set guard refuses a changed ancestor leaf
outside the admitted set before writing it. Five old private linked-page wire
tests were replaced by five equivalent v2 index tests covering exact canonical
bytes/hashes, byte/entry boundaries, escaped paths, null-to-hash pointer growth,
oversize/empty behavior and linear serialization work; this is a private version
change, not omission of coverage.

```text
RUN green-5 tests/archive/test_retained.py -q -s -k three_archives
1 failed, 34 deselected in 4.94s
StorageBlocked: frozen executable changed
```

This run was invalidated by the implementer mistakenly formatting source while
the yielded source-bound test was still running. The guard correctly refused.
Its fixtures/caches are preserved; it is not counted as a passing run or a
production defect. Later formatting completed before all test starts, and source
remained frozen through each run.

```text
RUN red-4 tests/archive/test_retained.py -q -k noncanonical_index
1 failed, 35 deselected in 0.24s
DID NOT RAISE StorageBlocked
```

A self-consistent rehashed, length-correct but noncanonical index passed JSON
parsing. `_object` now requires an exact canonical JSON object in addition to the
content digest. This also rejects duplicate-key/whitespace alternate encodings.

```text
RUN green-6 tests/archive/test_retained.py -q -s -k 'noncanonical_index or three_archives or index_wire or index_escaping or serialization_work'
7 passed, 29 deselected in 26.81s

RUN cover-1 tests/archive/test_retained.py tests/archive/test_preflight.py tests/archive/test_ledger.py tests/pilot/test_pilot_source.py::test_real_introduction_chain_and_dirty_archive_helper -q
96 passed in 53.78s
```

This final covering ran after source-freeze notice, without concurrent source
edits. It includes strict reader corruptions, existing custody/provenance/privacy/
review-to-seal/eviction/reservation/restart behavior, ledger arithmetic, and the
independent current source-closure equality/dirty-helper regression. Output was
clean; no warnings. The focused multi-operation run's deliberate JSON metrics are
the following real observations.

## Three successive archives: measured versus admitted

Fixture: twenty ordinary 32 KiB payloads in separate candidate directories,
eleven original leaves (`page_entries=4`), one operational budget and one shared
local transport/control history. Operations remove candidate 00, 10 and 19.
The test observes actual category allocation at real `os.fsync` boundaries and
calls the normal archival writer, not a fake usage calculator.

| Operation | Changed leaves | New inventory objects | New encoded bytes | Inventory byte bound | Observed metadata peak | Metadata admission | Observed scratch peak | Scratch admission |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 00 | 1 | 5 | 4,032 | 5,713 | 65,536 | 39,681,302 | 98,304 | 3,443,280 |
| 10 | 1 | 4 | 3,050 | 5,712 | 73,728 | 40,832,004 | 143,360 | 3,788,536 |
| 19 | 1 | 3 | 1,970 | 5,712 | 73,728 | 41,834,652 | 184,320 | 4,133,792 |

All values are bytes except counts. Category peaks are positive increases from
each operation's measured baseline and include old/new physical coexistence.
New object counts satisfy the derived object bounds; encoded growth is below
both its bound and the original whole-inventory size. Content-addressed index
objects which are unchanged are reused, explaining declining new-object counts.
All four durable generation descriptors remain present, hash-addressed, and
readable after completion; their row counts differ by exactly the three selected
removals. Every old leaf/index file remains. This demonstrates the requested
bounded rewrite behavior locally, not the real 134k-row admission or all 369
real candidate operations.

## Static checks, resource use and self-review

Exact final static checks (same explicit cache path):

```sh
env RUFF_CACHE_DIR="$LIFE/ruff" .venv/bin/ruff check \
  src/silent_cascade/archive/_retained.py src/silent_cascade/archive/preflight.py \
  src/silent_cascade/train/pilot_evidence.py tests/archive/test_preflight.py \
  tests/archive/test_retained.py
# All checks passed!
env RUFF_CACHE_DIR="$LIFE/ruff" .venv/bin/ruff format --check \
  src/silent_cascade/archive/_retained.py src/silent_cascade/archive/preflight.py \
  src/silent_cascade/train/pilot_evidence.py tests/archive/test_preflight.py \
  tests/archive/test_retained.py
# 5 files already formatted
git diff --check
# exit 0, no output
du -sk .superpowers/sdd/2026-09-17-phase-4-r2-archive \
  .superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-6-prelude-lifecycle
# 6023396 whole plan; 25772 lifecycle (KiB), after covering, before this report
```

Lifecycle output is about 25.2 MiB, below 64 MiB. Whole-plan allocation is below
the 6 GiB warning threshold and the 6.5 GiB normal ceiling. Earlier fixtures and
all failed-run evidence remain; no cache or evidence was deleted/exempted.
No default cache was produced in this slice. The previous original-task Ruff
cache incident remains preserved by the controller in the original counted
prelude root, as recorded in its original report.

Files changed: new `_retained.py` and `test_retained.py`; focused integration in
`preflight.py`; one executing-source inventory entry in `pilot_evidence.py`;
private-format test migration in `test_preflight.py`; this report.

Self-review checked exact affected-path authority, unchanged-leaf identity,
directory identity preservation, complete residual comparison, immutable
historical descriptors, every output-bound component, canonical strict readers,
reference-table memory bounds, v1 fail-closed behavior and current source closure.
It found and corrected the bool-layout and noncanonical-index cases with RED/GREEN
evidence, and the descriptor lifecycle gap with explicit controller approval.
The serializer is 607 lines because it owns strict schema/authentication,
bounded publication and readers; it remains the single approved private module,
not a generic tree/registry framework. No known unhandled blocker remains for
the accepted slice.

## Remaining conditions

Independent review comes next. The controller must subsequently perform the
separate read-only real-workspace v2 snapshot/bounds feasibility pass; this task
did not perform or certify it. Stable leaf reuse does not waive bootstrap normal
space, per-category admission, cumulative historic metadata, physical free-space,
content-review or retained-sibling checks. Exact future binding terms must be
reported, not replaced by caller-selected bounds or larger quotas.

Only after review and actual admission may the controller freeze real custody,
bootstrap the one allowance, and perform native/provider qualification using the
same local/remote history. Process interruption, streaming restore, transfer
before/after counters, remaining Task5 recovery and later full qualification
remain separate obligations. No claim of full Task6/Task7 or full gate completion
is made here.
