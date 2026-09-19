# Task 5 cold continuation/offline semantic-reader report

## Scope and commits

- Base: `0a057e7` (`docs: scope cold continuation and offline readers`).
- Delivery commit: the commit containing this report.
- Production: `src/silent_cascade/train/pilot_evidence.py` and the narrowly
  required materialized shared-weight input in
  `src/silent_cascade/eventflow/neural_checkpoint.py`.
- Tests: `tests/pilot/test_pilot_cold_semantics.py` and its only adjacent helper,
  `tests/pilot/cold_semantics_fixture.py`.
- Report: this file. Controller-owned `operational/sdd/progress.md` and plan
  edits were deliberately left unstaged.

This slice does not integrate the context into aggregate gate callers, numerical
readers, recovery, providers, or production pilot execution. It does not claim a
full `make verify` result.

## Historical provenance

The genuine read-only fixture remains at:

`.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-4/cover-pilot-final/test_real_smoke_workflow_reuse0/repo/runs/smoke`

- Original source commit: `42b8ab0ba8a647794b08ce29327fce105e3a75f7`.
- Recorded outcome: `debug_non_acceptance`.
- `phase4-gate.json`: 1,334,489 logical bytes,
  SHA-256 `c282f0c041c5806224731a954da5c3bdd7b83c01127ca9fdf835dc5a9123b4ca`.
- `phase4-gate.primary.00000.jsonl.gz`: 242,409 logical bytes,
  SHA-256 `b281ceadbf31240dbf92267ef7249ec9f5b05677bf0cffc946d120475d649325`.
- `final/runtime/0.safetensors`: 3,157,874 logical bytes,
  SHA-256 `f2706c4cd8649708c670a9495047e9b6e1ffeddfe10440915f748e62dede7811`.
- `final/runtime/0.replay.json`: 57,958 logical bytes,
  SHA-256 `5834d35c54bdd877f6dbb9899a19753677ada6c9d8b82c6ee4991d3d8a87ba3c`.
- Genuine referenced crash runtime:
  SHA-256 `4ca929154019a89142bcc58c980fd115a755a35bd8148ed3074e6b513a580f7e`.
- Its genuine shared weights:
  SHA-256 `e102be3e3d9c04b9e939ca763d4b6ac1e8a768c62ead350668271332bbf8e094`.

No historical source/configuration/hash identity was rewritten or authenticated
against the current HEAD. The original files were never moved, modified, or
deleted. The final-episode corruption control copied exactly four bounded files;
the dangling-reference control copied one 298,016-byte checkpoint and was
reported as derived evidence, never as a new historical run.

## RED and GREEN evidence

All pytest invocations used `.venv/bin/python -B`, disabled plugin autoload and
the cache provider, isolated TMP/XDG/MPL/Hypothesis/archive-test paths, and
preserved every output prefix. The exact initial RED command was:

```bash
test ! -e "$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1" && mkdir -p "$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/tmp" "$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/xdg-cache" "$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/xdg-config" "$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/xdg-data" "$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/mpl" "$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/hypothesis" "$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/archive" && env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 TMPDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/tmp" TMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/tmp" TEMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/tmp" XDG_CACHE_HOME="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/xdg-cache" XDG_CONFIG_HOME="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/xdg-config" XDG_DATA_HOME="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/xdg-data" MPLCONFIGDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/mpl" HYPOTHESIS_STORAGE_DIRECTORY="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/hypothesis" SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/archive" .venv/bin/python -B -m pytest -q -x --tb=short -p no:cacheprovider --basetemp="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/red-1/pytest" tests/pilot/test_pilot_cold_semantics.py::test_genuine_continuation_and_offline_results_match_cold_reads
```

Result: exit 1, one failure in 2.01 seconds. Cold verification reported eight
unavailable checks while eager verification reported none. This was the intended
resident-only discovery failure, not an unsupported-keyword error. Allocation:
0 bytes, 0 files, 11 directories.

The first whole-file GREEN attempt (`green-1`, same command below with that root)
stopped after one pass on a test-only regex mismatch: one passed, one failed in
3.15 seconds; allocation 0 bytes, 0 files, 13 directories. The preserved
`green-2` attempt passed 10 tests in 5.02 seconds after the fixture correction.

The focused optional-loader RED used the same environment with the fresh
`device-red` root and this exact selector:

```bash
.venv/bin/python -B -m pytest -q -x --tb=short -p no:cacheprovider --basetemp="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/device-red/pytest" tests/pilot/test_pilot_cold_semantics.py::test_materialized_weights_reject_requested_device_mismatch
```

The root and all distinct environment directories were created before the
selector exactly as for `red-1`, with `device-red` substituted. Result: exit 1,
`DID NOT RAISE ReplayError`, one failure in 0.99 seconds. It proved that a CPU
materialized model could previously bypass a requested-device check without
executing the MPS backend. Allocation: 0 bytes, 0 files, 8 directories.

The exact final GREEN pytest suffix was:

```bash
.venv/bin/python -B -m pytest -q -x --tb=short -p no:cacheprovider --basetemp="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/final-cover/pytest" tests/pilot/test_pilot_cold_semantics.py
```

It used the same full isolated environment/root creation as `red-1`, with the
fresh `final-cover` root substituted. Result: exit 0, 11 passed in 5.10 seconds.
`final-cover` used 3,633,152 allocated bytes, 7 files and 43 directories.
All preserved Task 5 prefixes used 7,266,304 allocated bytes. No persistent test
log was written; captured diagnostics stayed below 256 KiB.

Final static checks:

```bash
.venv/bin/python -B -m ruff check --no-cache src/silent_cascade/eventflow/neural_checkpoint.py src/silent_cascade/train/pilot_evidence.py tests/pilot/cold_semantics_fixture.py tests/pilot/test_pilot_cold_semantics.py
.venv/bin/python -B -m ruff format --check --no-cache src/silent_cascade/eventflow/neural_checkpoint.py src/silent_cascade/train/pilot_evidence.py tests/pilot/cold_semantics_fixture.py tests/pilot/test_pilot_cold_semantics.py
```

Result: `All checks passed!` and `4 files already formatted`.

## Semantics and leases

- Context/root relationships are rejected through `logical_root` before
  inventory or evidence reads. `run_dir=None` with a context is rejected.
- Each inventory iterator is fully exhausted. Only expected continuation names,
  five direct offline names, and the offline evaluation prefix are retained.
- Absence is decided from resident files plus the authenticated filtered
  inventory before leasing. Advertised lease, bytes, hash, and semantic failures
  raise instead of becoming absence.
- A missing top-level continuation checkpoint cannot hide an independently
  advertised corrupt replay. A missing offline source cannot hide a corrupt
  available update. Existing unavailable labels remain exact.
- Genuine eager/cold results and unavailable lists matched exactly: both
  continuation/offline results matched and both unavailable lists were `[]`.
  The absent/unindexed control produced exactly
  `continuation.checkpoint:final/runtime/0.safetensors` and
  `continuation.replay:final/runtime/0.replay.json`.
- Offline verification passes the context run root to each `evidence_path` and
  the nested evaluation root only to the real `load_evaluation` scanner.
- The strict context observed at most one payload lease. The real scanner held
  its allowed metadata lease concurrently with at most one episode lease, for a
  maximum total depth of two. Success and every tested failure returned to zero
  active leases. The final episode was consumed and its corruption rejected.
- Referenced runtime validation decodes/releases the runtime file, then
  loads/releases its recorded shared weights, then reacquires the runtime file.
  The materialized input must match recorded archive SHA-256, identity, model
  snapshot metadata and requested tensor device. An isolated checkpoint with no
  sibling succeeded only with the valid materialized weights; the default path
  rejected the dangling reference.

## No-execution boundaries and self-review

Tripwires covered `EventFlowModel.forward`, `EventEngine.run_until`,
`EventEngine.run_episode`, neural/general replay verification, pilot training,
and general training. None fired. Decoder/model construction and tensor state
inspection occurred; no fit, forward pass, event engine, replay execution,
provider session, production pilot, or foundation-model call occurred.

Self-review checked each plan bullet and realistic mutations: resident-only
discovery, partial inventory iteration, lease failure downgraded to absence,
early return after unavailable evidence, scanner prefix-only consumption,
released cache-path reuse, shared-weight hash/identity/model/device bypass, and
hidden final-episode corruption all have focused failures. The original eager
dangling-reference behavior remains strict. No whole evidence tree was copied.

Unavailable comparisons not made in this slice: numerical tensors, aggregate
gate/caller integration, recovery, provider transport, present-source scientific
execution, MPS backend execution, and full local `make verify`. The historical
resource test skips cleanly when the private fixture is absent; such a skip is
not full integration coverage. Independent controller review and a fresh
artifact-only check remain required after this commit.
