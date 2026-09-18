# Task 5 gate-verification status correction

Date: 2026-09-19

Implementation base: `d9bd3fe`

Implementation commit: `c72bdc7` (`fix: reject incomplete pilot gate verification`)

## Scope and admission

This is only the approved bounded correction under **Next bounded correction:
incomplete verification cannot pass**. It does not complete Task 5, a cold gate,
or Phase 4.

Same-ledger holder `51477`, process `49087@1789771167.628309`, granted token
`464a2a4fabaf30a12dab71308cfa4501` for operation
`task5-status-and-authentication`. The status slice had a 4 MiB cumulative
limit and a 1 MiB limit per tiny command within the shared 48 MiB reservation.
The pre-operation ledger values supplied by the controller were scratch
`209432576`, logs `1232896`, and unchanged retained bytes `6195077120`.

Only these source/test files changed:

- `src/silent_cascade/train/pilot_evidence.py`
- `tests/pilot/test_pilot_gate_verifier.py`

All test roots are retained below
`operational/scratch/task5-gate-status/<stage>`. Bytecode and pytest cache were
disabled. `TMPDIR`, `TMP`, `TEMP`, XDG, MPL, Hypothesis, pytest basetemp, and
archive scratch were routed to the named stage; archive scratch preservation was
enabled. The expensive
`test_portable_replay_comparison_binds_actual_trace` selector was never run.

## Diagnosis and correction

`verify_phase4_gate_artifact` independently accumulated unavailable semantic
checks and returned them, but its final status predicate used only recorded
outcome and missing raw paths. Consequently, internally consistent historical
comparisons could yield a current `passed=True` even when a semantic check could
not run.

The final production projection was extracted into
`_gate_verification_status`, initially with the old predicate. After the
required RED, its only behavioral change was:

```python
passed = outcome == "passed" and not missing and not unavailable
```

The projection retains `valid=True`, recorded outcome, copied missing paths,
sorted unique unavailable names, `recorded_evidence_integrity`, and
`not_rerun`. Source, checkpoint, and artifact identities remain in the real
verifier. No validator, acceptance threshold, schema, recorded comparison, or
gate hash changed.

## Exact commands and results

### Behavior-preserving extraction

```sh
mkdir -p .superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/refactor-smoke/tmp
env TMPDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/refactor-smoke/tmp" TMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/refactor-smoke/tmp" TEMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/refactor-smoke/tmp" XDG_CACHE_HOME="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/refactor-smoke/xdg-cache" MPLCONFIGDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/refactor-smoke/mpl" HYPOTHESIS_STORAGE_DIRECTORY="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/refactor-smoke/hypothesis" SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/refactor-smoke/archive" SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q --tb=short -p no:cacheprovider --basetemp="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/refactor-smoke/basetemp" tests/pilot/test_pilot_gate_verifier.py::test_acceptance_requires_every_independent_obligation tests/pilot/test_pilot_gate_verifier.py::test_continuation_claim_cannot_be_a_cached_flag tests/pilot/test_pilot_gate_verifier.py::test_numeric_flag_without_report_is_not_evidence
```

Result: `11 passed in 0.81s`, exit 0.

### TDD RED

```sh
mkdir -p .superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/red/tmp
env TMPDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/red/tmp" TMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/red/tmp" TEMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/red/tmp" XDG_CACHE_HOME="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/red/xdg-cache" MPLCONFIGDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/red/mpl" HYPOTHESIS_STORAGE_DIRECTORY="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/red/hypothesis" SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/red/archive" SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q --tb=short -p no:cacheprovider --basetemp="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/red/basetemp" tests/pilot/test_pilot_gate_verifier.py::test_gate_verification_status_requires_complete_semantic_evidence
```

Result: `1 failed, 11 passed in 0.68s`, exit 1. The sole failure was the literal
row `passed` + no missing raw path + unavailable semantic evidence: expected
`passed=False`, actual `passed=True`.

### GREEN

```sh
mkdir -p .superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/green/tmp
env TMPDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/green/tmp" TMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/green/tmp" TEMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/green/tmp" XDG_CACHE_HOME="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/green/xdg-cache" MPLCONFIGDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/green/mpl" HYPOTHESIS_STORAGE_DIRECTORY="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/green/hypothesis" SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/green/archive" SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q --tb=short -p no:cacheprovider --basetemp="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/green/basetemp" tests/pilot/test_pilot_gate_verifier.py::test_gate_verification_status_requires_complete_semantic_evidence tests/pilot/test_pilot_gate_verifier.py::test_offline_unavailable_checks_prevent_status_pass tests/pilot/test_pilot_gate_verifier.py::test_offline_corruption_is_rejected_when_source_check_is_unavailable
```

Result: `14 passed in 0.76s`, exit 0. The real offline validator retained its
historical `True` comparison for schema-valid but untrusted declarations with no
raw evidence, populated unavailable checks, and the production projection did
not pass. A present corrupt `step.json` was rejected for a different
`backward_macs` even though executed-source evidence was missing.

### Focused stable verification

The same six exact selectors were run in both `verify` and the separately
approved `post-format-verify` root:

```text
tests/pilot/test_pilot_gate_verifier.py::test_gate_verification_status_requires_complete_semantic_evidence
tests/pilot/test_pilot_gate_verifier.py::test_offline_unavailable_checks_prevent_status_pass
tests/pilot/test_pilot_gate_verifier.py::test_offline_corruption_is_rejected_when_source_check_is_unavailable
tests/pilot/test_pilot_gate_verifier.py::test_acceptance_requires_every_independent_obligation
tests/pilot/test_pilot_gate_verifier.py::test_continuation_claim_cannot_be_a_cached_flag
tests/pilot/test_pilot_gate_verifier.py::test_numeric_flag_without_report_is_not_evidence
```

The exact command form, once with `STAGE=verify` and once with
`STAGE=post-format-verify`, was:

```sh
mkdir -p .superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/$STAGE/tmp
env TMPDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/$STAGE/tmp" TMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/$STAGE/tmp" TEMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/$STAGE/tmp" XDG_CACHE_HOME="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/$STAGE/xdg-cache" MPLCONFIGDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/$STAGE/mpl" HYPOTHESIS_STORAGE_DIRECTORY="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/$STAGE/hypothesis" SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/$STAGE/archive" SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q --tb=short -p no:cacheprovider --basetemp="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/$STAGE/basetemp" <the-six-selectors-above>
```

The executed commands used the literal stage path rather than a shell variable.
Results were `25 passed in 0.85s` for `verify` and the final fresh
`25 passed in 0.83s` for `post-format-verify`, both exit 0.

### Scoped static checks

```sh
env TMPDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/tmp" TMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/tmp" TEMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/tmp" XDG_CACHE_HOME="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/xdg-cache" MPLCONFIGDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/mpl" HYPOTHESIS_STORAGE_DIRECTORY="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/hypothesis" RUFF_CACHE_DIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/ruff-cache" SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/archive" SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check --no-cache src/silent_cascade/train/pilot_evidence.py tests/pilot/test_pilot_gate_verifier.py
env TMPDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/tmp" TMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/tmp" TEMP="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/tmp" XDG_CACHE_HOME="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/xdg-cache" MPLCONFIGDIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/mpl" HYPOTHESIS_STORAGE_DIRECTORY="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/hypothesis" RUFF_CACHE_DIR="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/ruff-cache" SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-gate-status/lint/archive" SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff format --check --no-cache src/silent_cascade/train/pilot_evidence.py tests/pilot/test_pilot_gate_verifier.py
```

The first check returned `All checks passed!`. The initial format check reported
only layout differences in both files; those exact whitespace changes were
applied with `apply_patch`. The repeated check returned `All checks passed!`, and
the final format check returned `2 files already formatted`; both exit 0.

## Allocation evidence

The pre-command bound allowed at most 64 regular files of at most 8 KiB and 128
directories at one 4 KiB block per command, with less than 4 KiB of test-owned
JSON. The actual retained allocation was much smaller:

| Stage | Allocated | Files | Directories |
| --- | ---: | ---: | ---: |
| `refactor-smoke` | 0 KiB | 0 | 2 |
| `red` | 0 KiB | 0 | 2 |
| `green` | 4 KiB | 1 | 6 |
| `verify` | 4 KiB | 1 | 6 |
| `post-format-verify` | 4 KiB | 1 | 6 |
| **Cumulative** | **12 KiB** | **3** | **22** |

Ruff ran with `--no-cache` and produced no lint output tree. No model fixture,
weights, replay, provider, source copy, or full-suite output was created.

## Self-review

- Removing `and not unavailable` recreates the observed single RED truth-table
  failure.
- The 12 literal rows cross all three recorded outcomes with missing and
  unavailable evidence. The duplicate, unsorted unavailable input has a literal
  sorted-unique expected output, and input lists are checked unchanged.
- The helper is called by the real verifier. Only status fields moved into it;
  source commit, selected weight identity, and artifact digest remain in the
  verifier.
- The no-raw offline case exercises `verify_offline_evidence`, not a fabricated
  semantic-success mapping. Its historical comparison stays true while status
  remains false.
- The corrupt-step case proves one unavailable check does not short-circuit a
  separately available corruption check.
- Existing cheap acceptance, continuation, and numeric-evidence selectors pass.
- `git diff --check`, scoped Ruff check, and scoped Ruff format check are clean.

## Limitations

This unit correction does not run the full gate verifier against a complete
artifact, the neural portable-replay fixture, any model fit/replay, the full test
suite, `make verify`, a provider, or a production pilot. It does not claim a cold
gate or Task 5 completion. Broader cold-reader threading and end-to-end Task 5
integration remain pending. Independent review, full storage reconciliation,
reservation release, and final fresh checks are owned by the controller.
