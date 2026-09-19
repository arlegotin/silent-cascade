# Task5: artifact-only cold numerical semantic readers

Read the subsection `Artifact-only numerical semantic readers` in
docs/superpowers/plans/2026-09-17-phase-4-r2-archive.md first. It is this task's
complete requirements. Read the adjacent operational/sdd/task5-numeric-reader-
preflight.md for exact dependencies and measured original fixture sizes. Do not
read the entire plan or implement aggregate/recovery tasks. Use Superpowers
TDD/debugging/verification. No subagents. Work on current main, scoped commits.

## Scope

- Source: train/pilot_verification.py named reader group, plus
  train/pilot_evidence.py::verify_numeric_evidence only. No runtime execution,
  scientific tolerance or schema changes; low codecs remain local.
- Tests: tests/pilot/test_pilot_cold_numerics.py and one small adjacent helper.
  Reuse/adapt reviewed tests/pilot/cold_semantics_fixture.py where sensible:
  the new runtime also has16rows and no crashes, but its logical root differs.
  If adapting it, include the previous genuine continuation/offline equivalence
  selector in GREEN as a no-copy regression; no need to rerun its copying
  corruption cases during every iteration. Avoid another general fixture layer.
- Source/runtime/behavior predicates must still execute real artifact readers.
  Do not substitute numeric report booleans, operation comparisons, identity or
  checkpoint decoders. Return materialized values, not released cache paths.
- Permit existing forensic tensor/batch/disposable CPU optimizer arithmetic;
  forbid/tripwire model forward, training fit, event and replay execution. No
  new numeric diagnostic production, MPS backend, provider or full gate.

## Genuine read-only evidence and constraints

Fixture root:
`.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-4/cover-pilot-final/test_real_smoke_workflow_reuse0/repo/runs/smoke/final/numerics`

Original source42b8ab0ba8a647794b08ce29327fce105e3a75f7, missing_devices=["mps"].
Preserve original false comparison status. No source/hash relabeling or new
completed-run claim. It cannot establish MPS parity, empty-optimizer bootstrap,
fresh diagnostics,64production runtime episodes or complete current-source gate.

Everything under the plan's .superpowers directory outside operational/ is
FROZEN: do not move/write/delete/touch original inputs. No generic SDD scripts.
Only borrowed read-only backing plus bounded new corrupt controls is allowed.
Strict context must guard actual descriptor-relative opens and support metadata
plus exactly one payload lease. Never copy the complete83MiB numerical tree or
12MiB tensor captures. Fully consume inventory; retain only relevant prefix.

## Admission process

One source writer and one controller-owned storage reservation. No project
imports/tests before controller supplies a fresh admission; prepare tests and
send exact command/pre-write bounds first. Use .venv/bin/python -B, not uv run;
disable bytecode/plugins/cacheprovider and isolate TMP/XDG/MPL/Hypothesis paths.
Use fresh operational/scratch/task5-cold-numerics/<stage>, refuse existing,
preserve every prefix. Historical tests skip if required private input absent.

Proposed allowance, NOT yet granted:16MiB scratch total, three stages<=5MiB each
plus1MiB administration. Preflight exact13payloads3792896roundedB, one65536B
rewrite,131072B administrativefiles and<=64directories charged4KiB each gives
4251648B<5MiB. Derive actual cases before writes; original inputs stay charged
to frozen custody. Avoid copying runtime metadata/report when direct read-only
overrides suffice. First RED should be only genuine eager/cold report equality,
normally no copies and<=1MiB administration. Get a real behavior failure before
production edits, not just unknown keyword or missing fixture machinery.

No nested budget, baseline reset, quota increase, category reassignment, hidden
temporary path, provider call or cleanup deletion. Controller owns all grants.

## Report

Write operational/sdd/task5-cold-numerics-report.md in this plan directory:
exact commands/results/time/allocation, RED/GREEN behavior, real unavailable
results/missing MPS, hashes/source, lease cleanup, no-execution tripwires,
scoped files/commits, self-review and limitations. Commit only own source/tests/
report, not controller plan/progress files. Full makeverify is not admitted.
Return only DONE/concerns, commit, tests and report path; controller supplies
independent review and fresh verification after frozen commit.
