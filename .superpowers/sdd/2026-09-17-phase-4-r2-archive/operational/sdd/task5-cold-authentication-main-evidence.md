# Controller verification: completed cold authentication

Implementation e46e2b7. Sole same-ledger owner80995 remained active.
Fresh verification reused the completed GREEN source fixture; it did not train
again or substitute current Git identity for that fixture's identity.
Before invocation, recursive byte comparison of current src (excluding
__pycache__), configs and docs/superpowers against GREEN checkout passed.

Additional admission: <=1MiB new outer scratch, <=64KiB repaired controls in
existing spool, <=768KiB diagnostics, one <=16MiB cache lease. These fit the
existing grant after RED/GREEN; no extra ledger, quota or baseline.

Exact fresh command:

```sh
stage='/Volumes/git/legotin/silent-cascade/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-authentication/main-cover'
test ! -e "$stage" && mkdir -p "$stage/tmp" &&
env TMPDIR="$stage/tmp" TMP="$stage/tmp" TEMP="$stage/tmp" XDG_CACHE_HOME="$stage/xdg-cache" XDG_CONFIG_HOME="$stage/xdg-config" XDG_DATA_HOME="$stage/xdg-data" MPLCONFIGDIR="$stage/mpl" HYPOTHESIS_STORAGE_DIRECTORY="$stage/hypothesis" SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$stage/archive" SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SC_COLD_SCRATCH='/Volumes/git/legotin/silent-cascade/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-authentication/green' SC_COLD_SPOOL='/Volumes/git/legotin/silent-cascade/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/spool/task5-cold-authentication/green' SC_COLD_CACHE='/Volumes/git/legotin/silent-cascade/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/cache/task5-cold-authentication/green' SC_COLD_REUSE=1 .venv/bin/python -B -m pytest -q -s --tb=short -p no:cacheprovider --basetemp="$stage/pytest" tests/pilot/test_pilot_cold_authentication.py::test_completed_fit_authentication_and_reuse_through_cold_inputs
```

Result: 1 passed in16.32s, exit0. The real completed-source fixture remained
0638d984db3e38d9eb750521b4ac0d65303e00e4, source SHA-256
2eb6613aee868285a05425829ce3bccfb9b47ce4dfacd0f87887e6e235eea879.
Its model-state hash remained
92e724e6d11ed191438b8299625aa86a514287bf309c70bf6249f704981b3922.
Restored tensors/RNG, eager/cold authentication, logical path, both completed
reuse paths under a training tripwire and all corruption/root controls passed.
Maximum one lease; zero cached files at completion. Guard peak for this reread
43110400B. GREEN spool after retained verification repairs42409984B (additional
12288B), GREEN scratch6569984B unchanged; fresh outer main-cover scratch0B.

Fresh no-cache Ruff lint passed; format check reported4files already formatted;
git diff --check passed. A full shared-ledger check passed before this command;
the post-command full check and independent task review are recorded separately.
No full make verify, provider qualification, new fit or full-gate completion
is claimed. The fixture's scientific outcome remains debug step_ceiling with
its actual adverse rows, not a Phase4 acceptance result.
