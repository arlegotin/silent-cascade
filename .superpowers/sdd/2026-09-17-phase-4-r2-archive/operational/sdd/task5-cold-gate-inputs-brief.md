# Task5 artifact-only gate inputs and reuse admission

Read `Artifact-only gate inputs and reuse admission` in
docs/superpowers/plans/2026-09-17-phase-4-r2-archive.md first; it is your complete
requirements. Also read adjacent task5-gate-inputs-preflight.md for the root and
later integration contracts. This task may start only after controller dispatch
and prior numerical reader review. No tests/imports are admitted by this brief.

Use Superpowers TDD, systematic debugging and verification. No subagents. Stay
on current main. You are the sole source/test writer when dispatched; do not
modify controller plan, progress or preflight documents. Source/test commits
must be meaningful and scoped.

## Scope and interfaces

Modify train/pilot_evidence.py::_verify_available_training and
compact_attachment_records; add and actually use the small existing-gate reuse
predicate in train/pilot_checks.py. Add focused
tests/pilot/test_pilot_cold_gate_inputs.py. Reuse the reviewed historical strict
read guards where practical; a small adapter is allowed, no general framework.

Use existing evidence_path/logical_root and context-aware _durable. Readers
retain real CPU decoding, RNG restoration and all source/model/progress and
eligibility checks. Fully exhaust inventory and compact rows, while keeping one
payload lease and bounded discovery state. Available corrupt data must fail
despite another unavailable dependency. A missing exported shard cannot fall
back to an identically named raw-run shard. This task does not finish outer
gate context forwarding, collection, recovery or full reuse binding.

## Retained evidence and storage

Original genuine historical root:
`.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-4/cover-pilot-final/test_real_smoke_workflow_reuse0/repo/runs/smoke`

Source42b8ab0ba8a647794b08ce29327fce105e3a75f7 is debug_non_acceptance with no
selected checkpoint; existing production logic uses selected or progress.latest.
Never relabel this source or claim it passed current authenticate_run/full gate.
Original compact primary shard is242409logicalB, SHA256
b281ceadbf31240dbf92267ef7249ec9f5b05677bf0cffc946d120475d649325.

Everything under this plan's .superpowers directory outside operational/ is
frozen custody: never write, move, delete or touch originals. Borrow read-only;
derive only bounded small corrupt controls. Never copy a full run/archive tree,
run a training fit or new diagnostic. Tripwire model forward/fit/event/replay;
genuine CPU artifact decoding and existing forensic comparisons are allowed.

Controller owns the only storage reservation. Before any project imports/test
execution, supply the exact selectors/command and file/dir/logical/rounded bounds
checked before writes. Large checkpoint/weights can be corrupted through a
tiny overridden invalid payload rather than copying multi-MiB archives. Use
fresh retained operational/scratch/task5-cold-gate-inputs/<stage>, refuse if
already present, keep all failures. Use .venv/bin/python -B with bytecode,
plugins/cacheprovider disabled and all TMP/XDG/MPL/Hypothesis/archive paths
explicitly scoped plus --basetemp. No uv run, nested budget, provider, manual
cleanup, reclassification, baseline reset or quota increase.

## Report

Write operational/sdd/task5-cold-gate-inputs-report.md: exact commands/results,
RED/GREEN cause, timings, actual allocation, original source/hash bindings,
missing labels, cleanup, scope, self-review and limitations. Commit only owned
source/tests/report. Full make verify is not admitted. Return status, commit,
one-line test summary and concerns. Independent reviewer comes from controller.
