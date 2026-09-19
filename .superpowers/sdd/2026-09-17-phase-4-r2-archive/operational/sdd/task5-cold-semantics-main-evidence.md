# Controller verification: cold continuation/offline semantics

Implementation f08b161; same-ledger owner20818 remained active. Controller
admission: one fresh main-cover prefix <=8MiB, <=96files/48dirs, <=256KiB
diagnostics. Cumulative actual7266304B plus8MiB stayed below28311552B grant;
no reservation reset, extra ledger, provider call or historical modification.

Exact fresh command:

```sh
stage='/Volumes/git/legotin/silent-cascade/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-semantics/main-cover'
test ! -e "$stage" && mkdir -p "$stage/tmp" &&
env TMPDIR="$stage/tmp" TMP="$stage/tmp" TEMP="$stage/tmp" XDG_CACHE_HOME="$stage/xdg-cache" XDG_CONFIG_HOME="$stage/xdg-config" XDG_DATA_HOME="$stage/xdg-data" MPLCONFIGDIR="$stage/mpl" HYPOTHESIS_STORAGE_DIRECTORY="$stage/hypothesis" SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$stage/archive" SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -B -m pytest -q --tb=short -p no:cacheprovider --basetemp="$stage/pytest" tests/pilot/test_pilot_cold_semantics.py
```

Result: exit0, 11passed in5.10s. Real eager/cold continuation and offline semantic
results and unavailable lists matched. Final offline episode corruption,
independently corrupt evidence despite another missing input, wrong root, late
inventory error, shared-weight binding/device and missing-sibling checks passed.
The strict context enforced descriptor-relative reads and at most one payload
lease alongside metadata, with zero outstanding leases on success/error.

No training/forward/event/replay/backend execution or provider access occurred;
tripwires remained armed. This is artifact-only validation of unchanged original
source42b8ab0ba8a647794b08ce29327fce105e3a75f7 and debug_non_acceptance evidence,
not current-source scientific success or a complete Phase4 gate.

Fresh no-cache Ruff lint passed; formatcheck4filesalreadyformatted; gitdiffcheck
passed. Main-cover retained3633152B. All six prefixes together10899456B; earlier
failures preserved. Independent task review and parent full check are tracked
separately. No full makeverify, numeric semantics or recovery closure is claimed.
