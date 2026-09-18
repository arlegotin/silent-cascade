# Controller verification of cold durable-checkpoint reads

Source9f5047c. Shared holder51477 remained active; all prior outputs retained.
Fresh command admitted at5MiB with the same single <=4MiB real-checkpoint
publication guard, module-scoped reuse and directory/administrative bound.
Pre-command retained(status+durable)9474048B +5MiB+2MiB fit48MiB.

```sh
stage='/Volumes/git/legotin/silent-cascade/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-durable-reader/main-cover'
mkdir -p "$stage/tmp"
env TMPDIR="$stage/tmp" TMP="$stage/tmp" TEMP="$stage/tmp" XDG_CACHE_HOME="$stage/xdg-cache" XDG_CONFIG_HOME="$stage/xdg-config" XDG_DATA_HOME="$stage/xdg-data" MPLCONFIGDIR="$stage/mpl" HYPOTHESIS_STORAGE_DIRECTORY="$stage/hypothesis" SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$stage/archive" SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q --tb=short -p no:cacheprovider --basetemp="$stage/basetemp" tests/pilot/test_pilot_cold_checkpoint.py
```

Result:4passed in1.88s, exit0. New main-cover output3076KiB. Durable cumulative
12312KiB, plus16KiB status =12328KiB (12623872B). No outputs deleted.
Scoped no-cache Ruff check passed, format check2files alreadyformatted, and
`git diff --check` passed. Scope remains actual zero-step checkpoint consumer
behavior with labeled unit source/path context, not full run authentication,
provider eviction or a completed fit. Independent review and same-ledger full
check remain the corresponding completion gates.
