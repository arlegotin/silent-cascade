# Task5: artifact-only continuation and offline semantic readers

Read the plan subsection `Artifact-only continuation and offline semantic
readers` in docs/superpowers/plans/2026-09-17-phase-4-r2-archive.md. That is this
task's complete requirements. Do not implement the rest of Task5 or read the
entire plan. Use Superpowers TDD, debugging, verification and self-review.
No subagents. Work on current main and commit scoped changes.

## Boundaries and interfaces

- Production: train/pilot_evidence.py::verify_continuation_records and
  ::verify_offline_evidence. Optional shared-weight materialization change in
  eventflow/neural_checkpoint.py only if necessary; first report the proposed
  interface. No archive/provider or general runtime redesign.
- Tests: tests/pilot/test_pilot_cold_semantics.py and one adjacent helper.
- `archive.readers.evidence_path` requires the context run root, not a nested
  evaluation root. `logical_root` validates that relationship. Existing
  report.pilot_artifacts.load_evaluation accepts context and uses a metadata
  lease with one active episode lease; use that real scanner, not a stub.
- Fully exhaust entries so late corruption cannot hide. Keep only relevant
  expected names; no full corpus inventory materialization.
- Decoder/model construction and tensor comparison are allowed. Model forward,
  fit, event-engine execution and replay execution are forbidden/tripwired.
- Never rewrite old source/configuration/hash provenance to the current HEAD.
  No mock full-gate pass, no manufactured completed fit, no production pilot.

## Existing genuine fixture (read-only)

Root relative to repo:
`.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-4/cover-pilot-final/test_real_smoke_workflow_reuse0/repo/runs/smoke`

Original source: 42b8ab0ba8a647794b08ce29327fce105e3a75f7.
Recorded outcome: debug_non_acceptance.
phase4-gate.json: 1334489 logical bytes.
phase4-gate.primary.00000.jsonl.gz: 242409 logical bytes.
final/runtime/0.safetensors: 3157874 logical bytes (embedded weights).
final/runtime/0.replay.json: 57958 logical bytes.
final/offline total: 11935744 allocated bytes.
Do not copy this tree. A strict read-only test context can expose original
backing only inside declared leases while guarding all actual input reads;
the context's logical run root must have no resident evidence. Model/scanner
semantics must remain real. This proves consumer lifetime, not provider/IPC.
Original files stay charged to retained custody and must never move or change.
Gate/config/source parsing is allowed; authenticate_run on current source is not.

## Storage/execution admission

Everything under `.superpowers/sdd/2026-09-17-phase-4-r2-archive/` outside
`operational/` is FROZEN. Do not run generic SDD scripts or write there.
No test/project import/numerical execution until controller admits an exact
command after receiving your proposed output bounds. Read-only inspection and
test/source edits via apply_patch are allowed (production only after RED).

Use new roots below `operational/scratch/task5-cold-semantics/` and logs below
`operational/logs/task5-cold-semantics/`. Same global ledger only. Total local
allowance10GiB; scratch limit256MiB; latest scratch235225088B leaves33210368B.
Propose at most three stages<=8MiB each plus3MiB administration (27MiB total),
prefer far less. Derive exact payload/count/directory/coexistence limits and
enforce before writes. No complete checkpoint/capture copies unless explicitly
counted. No nested ledger, deletion, reset, category reclassification or quota
increase. Preserve every failed test prefix. Controller owns reservation.

First prepare test code and send a concise test/output bound plus command;
wait for controller's admission, not user's permission. Resource-gated historical
tests must skip cleanly when this private fixture is absent in a clean checkout;
do not describe such skips as full integration coverage.

## Report

Write operational/sdd/task5-cold-semantics-report.md under this plan directory.
Include exact RED/GREEN commands, errors/counts/durations, actual allocation,
scoped files and commits, no-execution tripwires, original hashes/provenance,
lease assertions, unavailable comparisons, concerns and self-review. Full local
make verify is NOT admitted here. Return only status, commit, test summary,
concerns, report path. Independent reviewer is controller's responsibility.
