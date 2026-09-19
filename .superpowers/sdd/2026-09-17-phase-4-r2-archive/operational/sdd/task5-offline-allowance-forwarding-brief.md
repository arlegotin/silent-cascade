# Task D1 — explicit allowance forwarding

Use the active R2 plan's Task D1 subsection; standing Phase4 approval covers
this in-scope execution refinement. Current main, Superpowers TDD/debugging/
verification, one source writer. Controller owns plan/progress/ledger/admission.
No subagents inside this task. Diagnostic02 is already qualified; do not repeat.

Own exactly four source files under `src/silent_cascade/train/`:
`pilot_workflow.py`, `pilot_checks.py`, `pilot_verification.py`,
`pilot_offline_process.py`; focused `tests/pilot/test_pilot_offline_process.py`.
Add optional `offline_write_allowance=None` through the six existing custody
APIs and exact-object forwarding, with `write_allowance` at low-level measure.
Existing preflight gains keyword-only optional `write_allowance` and exact-type
validation before existing completed/custody checks. None/reuse/forensics stay
compatible. No inferred/default capacity, pair requirement, report field,
delegated capability, new ledger/hold, scientific or policy change.

Map from read-only preflight:

- workflow.run_pilot -> _workflow -> checks._run_pilot_checks_owned;
- checks.run_pilot_checks -> _run_pilot_checks_owned;
- checks.recover_pilot_checks -> _run_pilot_checks_owned;
- owned checks -> offline.measure_pilot_offline;
- verification.measure_pilot_offline -> offline.measure_pilot_offline;
- current preflight sites: _workflow, owned checks, public checks, recovery.

TDD with tiny explicit spy controls only; prove identity, early invalid-type
rejection and None/completed reuse. No numerical/Git/child processes, actual
model/update/eval/replay/report, recovery copy, source checkout, provider, full
suite or make verify. Existing legacy tiny controls may be added only by exact
controller admission. Do not select the full process test file.

Before any test/project import/formatter, propose exact selectors/cases, process
count, command, finite payload/name/directory/temp/log/native bounds and a fresh
`operational/scratch/task5-offline-allowance-forwarding/<stage>` prefix. Entire
task proposed2MiB scratch including retained RED/GREEN/controller runs; policy
unchanged. Wait for explicit per-stage admission. Every outer stream/time limit
must be mechanically enforced, not just a tool token/yield setting. Retain all
attempts; no cleanup, cap increase, root reuse or second production authority.

After fresh verification, commit only five owned files. Concise report at
`operational/sdd/task5-offline-allowance-forwarding-report.md`: source/hash
identities, exact commands/results, actual per-name lstat allocation, failures,
self-review and exclusions. Controller repeats frozen checks and requests
independent review. Broader inherited supervisor/private FD and production
hold integration remain outstanding regardless of this task's outcome.
