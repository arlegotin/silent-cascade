# Task 1 report: immutable catalog and bounded unit codec

## Outcome

Implemented the Phase 4 R2 archive types, exact-byte unit catalog, bounded chunk
codec/restore path, and persistent paged run/corpus catalogs. All changes are
local transport/storage code; no credentials, network transport, scientific
schemas, training behavior, or frozen core I/O were changed.

## Implementation

- Added strict archive policy, identity, manifest, inventory, span, episode,
  borrowed-file, catalog-node, catalog-root, and immutable reference types.
- Added descriptor-pinned sealing of exact regular-file bytes with traversal,
  symlink, hard-link, special-file, path-swap, duplicate ownership, admission,
  canonical-order, and final-inventory-tail checks.
- Added explicit authenticated episode groups for flat ordinal evidence and
  separately located crash evidence. Sparse episode restore accepts exactly one
  declared group; bounded diagnostic selections remain explicit.
- Added authenticated borrowed references. They are verified at seal time,
  excluded from owned spans/expanded bytes, and never restored as owned data.
- Added run-root `.` support only for journals and control snapshots. Control
  snapshots retain original logical paths and are versioned by immutable unit
  identity outside ordinary active ownership.
- Added deterministic one-chunk-at-a-time reconstruction and authenticated
  staged restoration. Restore checks every supplied chunk, including sparse
  restores, fsyncs the staged tree, rechecks pinned parents and destination
  absence, and publishes into a fresh destination.
- Added bounded content-addressed binary Merkle tries for run units, active
  ownership, and corpus runs. Roots and every node obey page limits; traversal
  authenticates hashes, descriptors, ordering, counts, and the full tail.
- Added cold object and cold unit-inventory reader seams. Incremental run
  updates read only changed trie paths, retain archived ownership after local
  unit metadata removal, and reject exact plus ancestor/descendant collisions.
- Batched additions are coalesced before publication. Only generation-reachable
  staged nodes are published, staged-byte accounting is constant-time, and the
  decoded-node cache is bounded to at most eight pages.
- Pinned an entire catalog node/root publication to one control-directory
  descriptor and reject control-path replacement before or during publication.

## TDD evidence

Initial codec RED:

```text
TMPDIR=... .venv/bin/python -m pytest -q \
  tests/archive/test_catalog.py tests/archive/test_bundles.py --basetemp=.../red
10 failed, 19 errors
```

All failures were the expected missing `silent_cascade.archive` implementation.
Focused codec work subsequently reached 29 passing tests and then 33 passing
tests as the explicit episode, borrowed, control-snapshot, root-path, canonical
order, parent-swap, and destination-race cases were added. The first broad
regression before paged catalogs was:

```text
tests/archive/test_catalog.py tests/archive/test_bundles.py \
tests/unit/test_archive_io.py tests/unit/test_io.py
77 passed in 0.60s
```

Catalog TDD correction: an executable catalog scaffold was started before the
dedicated catalog behavior tests. It was removed with `apply_patch`; the design
notes and already verified unit codec were retained. Only then were the catalog
tests added and observed RED:

```text
.venv/bin/python -m pytest -q tests/archive/test_catalog.py \
  -k 'catalog_generations or archived_ownership or corpus_catalog'
5 failed, 33 deselected
```

Those five failures were missing `CatalogRef` and run/corpus publish/iterate
APIs. After implementation, the same behavior slice was GREEN:

```text
5 passed, 33 deselected in 0.59s
```

The catalog control-directory swap regression was separately observed RED and
GREEN:

```text
RED:   1 failed in 0.31s (publication did not reject the swap)
GREEN: 1 passed in 0.41s
```

The extended catalog slice including cold unit inventories and pinned
publication passed:

```text
7 passed, 34 deselected in 0.35s
```

The synthetic scale test lazily traversed 1,000,001 unit descriptors, asserted
bounded object request size, less than 4 MiB resident synthetic metadata, and
less than 64 MiB traced cursor/cache allocation:

```text
1 passed in 198.70s (0:03:18)
```

## Final verification

No full numerical `make verify` was run; that gate is reserved for Task 7.

```text
TMPDIR=... .venv/bin/python -m pytest -q \
  tests/archive/test_catalog.py tests/archive/test_bundles.py \
  tests/unit/test_archive_io.py tests/unit/test_io.py --basetemp=.../final/pytest
86 passed in 124.69s (0:02:04)

.venv/bin/ruff check .
All checks passed!

.venv/bin/ruff format --check .
279 files already formatted

git diff --check
(exit 0, no output)
```

All test scratch and pytest base directories were placed under
`.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-1`.

## Changed files

- `src/silent_cascade/archive/__init__.py`
- `src/silent_cascade/archive/types.py`
- `src/silent_cascade/archive/catalog.py`
- `src/silent_cascade/archive/bundles.py`
- `tests/archive/conftest.py`
- `tests/archive/test_catalog.py`
- `tests/archive/test_bundles.py`
- `.superpowers/sdd/2026-09-17-phase-4-r2-archive/task-1-report.md`

## Self-review

- Re-read the amended Task 1 brief and global constraints against the public
  signatures, exact default ceilings, closed kinds, episode/borrowed/snapshot
  rulings, bounded catalog requirements, and Task 2/3 cold-reader seams.
- Checked that owned data alone contributes spans and expanded bytes; borrowed
  files and control-snapshot versioning cannot weaken ordinary ownership.
- Checked catalog generation publication order: content-addressed nodes are
  durable before the root, and all objects in a generation share one pinned
  control-directory descriptor.
- Checked that incremental ownership lookup covers exact, owned-ancestor, and
  owned-descendant conflicts without downloading prior inventories.
- Checked tests avoid large real allocations: the million-entry test synthesizes
  leaves on demand and retains only bounded internal metadata.
- Reviewed the final diff for unrelated source changes and whitespace errors.

## Concerns

No known Task 1 correctness blocker remains. Full repository numerical and
integration verification remains intentionally deferred to Task 7 as required.

## Fix round 1 — review findings for `408fc9a`

All seven Important findings in `task-1-findings.md` were reproduced before
production changes. The combined regression RED was:

```text
.venv/bin/python -m pytest -q \
  tests/archive/test_catalog.py::test_seal_rejects_combined_inventory_metadata_above_reservation \
  tests/archive/test_catalog.py::test_incremental_catalog_insert_splits_above_compressed_prefix \
  tests/archive/test_catalog.py::test_catalog_rejects_oversized_node_descriptor_before_cold_read \
  tests/archive/test_catalog.py::test_catalog_rejects_loaded_leaf_above_entry_limit \
  tests/archive/test_catalog.py::test_run_catalog_rejects_nonadjacent_same_generation_ancestor_overlap \
  tests/archive/test_catalog.py::test_run_catalog_traversal_authenticates_ownership_tree_closure \
  tests/archive/test_bundles.py::test_sparse_episode_restore_requires_one_complete_authenticated_owned_set \
  tests/archive/test_bundles.py::test_restore_atomically_refuses_destination_created_after_final_check
8 failed in 0.67s
```

The same exact regression slice after the fixes was:

```text
8 passed in 0.49s
```

Addressed findings:

1. Incremental insertion now computes a combined first differing bit and splits
   above an existing compressed subtree when needed. Loaded branches validate
   ordered child ranges, zero/one routing, and the authenticated compressed
   prefix/depth invariant.
2. Same-generation ownership validation now maintains accumulated file and
   directory-prefix sets, catching nonadjacent exact and ancestor/descendant
   conflicts before catalog publication.
3. Unit sealing accounts for inventory shard bytes cumulatively before each
   shard write, then admits the manifest only when shards plus manifest remain
   within the metadata reservation. Failed staging trees are removed.
4. Restore publication now calls descriptor-relative exclusive rename:
   `renameatx_np(..., RENAME_EXCL)` on macOS and
   `renameat2(..., RENAME_NOREPLACE)` on Linux. Missing APIs and unsupported
   filesystem responses fail closed; there is no replacing fallback. A race
   inserted after the final absence check leaves the competing destination
   intact.
5. Sparse restore derives the sorted required chunk indices only after a full
   authenticated inventory traversal. It accepts only those chunk payloads,
   verifies each downloaded chunk completely, restores selected spans, rejects
   extra/missing relevant chunks, and still rejects a missing final inventory
   shard.
6. Catalog loads reject oversized decoded-byte descriptors before invoking a
   cold reader and enforce leaf entry limits on decoded cold nodes as well as
   locally staged nodes.
7. Run-catalog iteration authenticates and exhausts the ownership tree, checks
   record types/count closure, and only then yields unit entries.

The million-descriptor fixture was changed from loosely paired branches to a
valid compressed binary trie so the test continues to measure the required
bounded cursor/cache behavior under the new invariant enforcement.

Fix-round verification:

```text
# Archive tests except the separately measured scale test
59 passed, 1 deselected in 1.06s

# 1,000,001 lazily generated descriptors, bounded cold pages/cache/allocation
1 passed in 204.59s (0:03:24)

# Existing descriptor-safe I/O regressions
.venv/bin/python -m pytest -q \
  tests/unit/test_archive_io.py tests/unit/test_io.py
33 passed in 0.19s

.venv/bin/ruff check .
All checks passed!

.venv/bin/ruff format --check .
279 files already formatted

git diff --check
(exit 0, no output)
```

All fix-round scratch and pytest base directories remained under
`.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-1`. No network, cloud,
credential, transport, scientific-schema, or frozen-I/O change was made. Full
`make verify` remains reserved for Task 7.
