# Task5 eager optimizer cache correction

Approved scope: active R2 plan, “Diagnostic01 correction: eager optimizer cache
initialization”; failure record `task5-real-offline-01.md`. Source basef2ce734.
Use current main, Superpowers TDD/debugging/verification and apply_patch. One
source writer `/root/offline_optimizer_cache_fix`; controller owns admissions,
ledger and plan/progress. No nested subagents.

Own only `src/silent_cascade/train/pilot_offline.py` closed environment and
focused `tests/pilot/test_pilot_offline_writes.py` tests. Expected minimal fix:
derive `TORCHINDUCTOR_CACHE_DIR` from the existing `cache` child path. No Torch,
optimizer, bootstrap, allowance/grammar, custody, schema or scientific changes.

Tests: ambient-override-resistant environment mapping; real tiny scalar-CPU
AdamW construction after boundary installation, with no update/model/eval/
replay; cache remains empty and latch clear; cache files/subdirectories remain
forbidden and latch denial. Existing tiny helper: <=64KiB child,30s,2048B per
pipe,<=4096B source. Preserve original failure semantics; an expected denial may
be rethrown without its long library traceback in RED to bound test output.

Before every execution send exact selectors/cases/child count, command, finite
payload/name/directory/temp/log bounds and fresh scratch prefix. Wait for
controller admission. Entire task at most2MiB scratch including failed RED,
GREEN, statics and controller repeat. All stages remain retained. Do not infer
admission from free space. No full file/suite, actual diagnostic, source copy,
provider/network, cleanup, new ledger or category increase.

After verification commit only owned source/tests. Write exact RED/GREEN,
static and allocation evidence, full commit/hash identities, self-review and
remaining limits in `operational/sdd/task5-optimizer-cache-report.md`. Do not
stage controller documents. Controller verifies frozen source and requests
independent review before another separately admitted real diagnostic.
