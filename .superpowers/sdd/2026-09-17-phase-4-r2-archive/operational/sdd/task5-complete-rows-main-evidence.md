# Controller verification of complete cold partial rows

Code head fb3e051. All commands below ran locally, sequentially, under live
owner66017 with32MiB additional scratch and unique retained stage roots.
Before real commands, current slice allocation plus16MiB plus2MiB administration
remained below32MiB; strict command bound1MiB. No provider/scientific work.

## Success

```sh
stage='/Volumes/git/legotin/silent-cascade/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-complete-rows/main-success'
mkdir -p "$stage/tmp"
env TMPDIR="$stage/tmp" TMP="$stage/tmp" TEMP="$stage/tmp" XDG_CACHE_HOME="$stage/xdg-cache" MPLCONFIGDIR="$stage/mpl" HYPOTHESIS_STORAGE_DIRECTORY="$stage/hypothesis" SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$stage/archive" SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q --tb=short -p no:cacheprovider --basetemp="$stage/basetemp" 'tests/pilot/test_pilot_archive_readers.py::test_cold_partial_report_uses_authenticated_inventory[success]'
```

Result: 1 passed in3.73s;356KiB.

## Error

```sh
stage='/Volumes/git/legotin/silent-cascade/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-complete-rows/main-error'
mkdir -p "$stage/tmp"
env TMPDIR="$stage/tmp" TMP="$stage/tmp" TEMP="$stage/tmp" XDG_CACHE_HOME="$stage/xdg-cache" MPLCONFIGDIR="$stage/mpl" HYPOTHESIS_STORAGE_DIRECTORY="$stage/hypothesis" SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$stage/archive" SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q --tb=short -p no:cacheprovider --basetemp="$stage/basetemp" 'tests/pilot/test_pilot_archive_readers.py::test_cold_partial_report_uses_authenticated_inventory[initialization-error]'
```

Result: 1 passed in3.79s;316KiB.

## Strict

```sh
stage='/Volumes/git/legotin/silent-cascade/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-complete-rows/main-strict'
mkdir -p "$stage/tmp"
env TMPDIR="$stage/tmp" TMP="$stage/tmp" TEMP="$stage/tmp" XDG_CACHE_HOME="$stage/xdg-cache" MPLCONFIGDIR="$stage/mpl" HYPOTHESIS_STORAGE_DIRECTORY="$stage/hypothesis" SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$stage/archive" SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q --tb=short -p no:cacheprovider --basetemp="$stage/basetemp" tests/pilot/test_pilot_archive_readers.py::test_cold_partial_rejects_invalid_retained_evidence tests/pilot/test_pilot_archive_readers.py::test_cold_partial_rejects_invalid_newline_complete_row tests/pilot/test_pilot_archive_readers.py::test_cold_partial_rejects_unbound_predecessor tests/pilot/test_pilot_archive_readers.py::test_cold_root_training_result_is_discovered_from_authenticated_inventory
```

Result: 7 passed in0.83s;216KiB.

Scoped Ruff check, format check of3changed Python files, and git diff --check
passed. A read-only standard-library fixture decoder compared all7decoded
members byte-for-byte with the original retained success/failure artifacts;
all lengths/hashes matched. No original artifacts changed.

Full owner66017 check: scratch209432576,logs1212416,metadata52965376,
cache67153920,spool134307840,pinned0,emergency0. Retained6195077120 unchanged.
Main output888KiB; original stages2740KiB; total3628KiB. Original report prose
2760KiB was20KiB too high; table and measured per-stage allocation agree at2740.
No deletion/reset occurred. Full make verify and broader Task5 remain pending.
