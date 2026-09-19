# Controller verification: cold numerical readers

Verified source commit d4cd56ebdc27e5ecf2a9065725d47476494aaaf5; implementer report
commit f0b3c81. This is artifact-only validation, not a new training/diagnostic run
or a complete current-source gate.

## Fresh local checks

The following cwd-relative command reproduces the executed command. During
execution the stage was expanded to its absolute path within this checkout.

```sh
stage="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-numerics/main-cover"
test ! -e "$stage" && mkdir -p "$stage/tmp" &&
env TMPDIR="$stage/tmp" TMP="$stage/tmp" TEMP="$stage/tmp" XDG_CACHE_HOME="$stage/xdg-cache" XDG_CONFIG_HOME="$stage/xdg-config" XDG_DATA_HOME="$stage/xdg-data" MPLCONFIGDIR="$stage/mpl" HYPOTHESIS_STORAGE_DIRECTORY="$stage/hypothesis" SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$stage/archive" SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -B -m pytest -q --tb=short -p no:cacheprovider --basetemp="$stage/pytest" tests/pilot/test_pilot_cold_numerics.py tests/pilot/test_pilot_cold_semantics.py::test_genuine_continuation_and_offline_results_match_cold_reads
```

Result: `9 passed in 20.36s`, exit0, no warnings. Source/test writer had finished;
no concurrent numerical command or provider process was running. The stage was
new, preserved, and admitted under owner47503 at<=5MiB/128files/64dirs with
<=256KiB diagnostics. Actual main-cover allocation3489792B; all numerical
prefixes total10469376B. This fits the existing16777216B reservation without
deleting prior controls or changing their classification.

Fresh scoped checks also passed:

```sh
.venv/bin/ruff check --no-cache src/silent_cascade/train/pilot_verification.py src/silent_cascade/train/pilot_evidence.py tests/pilot/cold_semantics_fixture.py tests/pilot/test_pilot_cold_numerics.py
.venv/bin/ruff format --check --no-cache src/silent_cascade/train/pilot_verification.py src/silent_cascade/train/pilot_evidence.py tests/pilot/cold_semantics_fixture.py tests/pilot/test_pilot_cold_numerics.py
git diff --check
```

Output: all checks passed;4files already formatted; diffcheck exit0.

The source-bound original CPU report remains missing MPS and reports false
device_checks_passed. Every cold read uses genuine old values; no historical
source was relabeled. New diagnostics, production pilot, full aggregate gate,
recovery, MPS parity and provider integration remain outside this proof.

Independent task review and final owner accounting/release are recorded in the
adjacent progress ledger; do not infer them from the successful tests alone.

## Fix-round1 fresh controller evidence

Source11de6d4176f2ccbb4e0a7c3003b7202bfb946d56/report39a6467. Fresh admission
used the same owner47503, <=1MiB/128files/64dirs plus256KiB diagnostics. Exact
covering selectors and an equivalent cwd-relative launcher:

```sh
stage="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-numerics/fix-main-1"
test ! -e "$stage" && mkdir -p "$stage/tmp" &&
env TMPDIR="$stage/tmp" TMP="$stage/tmp" TEMP="$stage/tmp" XDG_CACHE_HOME="$stage/xdg-cache" XDG_CONFIG_HOME="$stage/xdg-config" XDG_DATA_HOME="$stage/xdg-data" MPLCONFIGDIR="$stage/mpl" HYPOTHESIS_STORAGE_DIRECTORY="$stage/hypothesis" SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$stage/archive" SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -B -m pytest -q --tb=short -p no:cacheprovider --basetemp="$stage/pytest" tests/pilot/test_pilot_cold_numerics.py::test_genuine_numeric_report_matches_cold_read tests/pilot/test_pilot_cold_numerics.py::test_genuine_numeric_verifier_and_runtime_rows_match_cold_reads tests/pilot/test_pilot_cold_numerics.py::test_missing_capture_tensor_validates_available_operations tests/pilot/test_pilot_cold_numerics.py::test_missing_capture_tensor_cannot_hide_corrupt_operations tests/pilot/test_pilot_cold_numerics.py::test_partial_numeric_evidence_rejects_extra_resident_member tests/pilot/test_pilot_cold_numerics.py::test_partial_numeric_evidence_rejects_dangling_resident_symlink tests/pilot/test_pilot_cold_numerics.py::test_extra_resident_numeric_inventory_is_rejected tests/pilot/test_pilot_cold_numerics.py::test_wrong_numeric_root_and_late_inventory_failure_precede_reads tests/pilot/test_pilot_cold_semantics.py::test_genuine_continuation_and_offline_results_match_cold_reads
```

Result: `9 passed in 18.85s`, exit0, no warnings. The stage retained32768B;
all numerical prefixes now total10539008B. All originals/failed prefixes stay
retained. Three changed files passed fresh no-cache Ruff lint/format; diffcheck
passed. The original copying cases were not rerun for this narrow fix; these
selectors cover complete report/verifier behavior, common inventory rejection,
partial available/missing/corrupt semantics, both added regressions, and prior
continuation/offline equivalence.

Independent scoped re-review and final custody/accounting are separate ledger
entries. There was no new scientific execution or provider operation.
