# Read-only prospective full-gate admission preflight

Read-only task5_remaining_gate_slice inspected the exact current-source smoke
fixture and writers. No tests, imports, provider calls or mutations were run.
The historical215456KiB tree is NOT a justified prospective total.

## Existing bounded pieces

- tests/pilot/cold_authentication_fixture.py:78 derives a12MiB checkout/history/
  data bound from source bytes, Git objects/indexes, directories and publication
  limits. Re-evaluate assertions against the final committed source.
- Its introduce_data at337 restricts16 data/audit outputs before publication:
  12files<=24KiB and four indexes<=256bytes.
- FitGuard at206 caps combined spool/cache64MiB,256names,48directories and
  16MiB per publication, counting pending temporary and existing output. This
  is the fit-only allowance, not another64MiB per later stage.

## Exact remaining gaps

- tests/pilot/test_pilot_workflow.py:43 uses the unbounded checkout from
  tests/neural/test_phase3_provenance.py:28, with no FitGuard installed.
- FitGuard.installed at252 omits imported _publish_pilot_bytes aliases in
  train/pilot_checks.py, train/pilot_evidence.py and report/pilot.py.
- Numerical _save_capture atpilot_verification.py:741 locally imports the
  patched pilot_data publisher and is interceptable in the same interpreter;
  its inclusion in a combined total/count allowance still needs derivation.
- io._durable_temp covers atomic evaluation/archive publication and the guard
  covers pending-row appends. Arbitrary library/cache writes are not thereby
  prospectively bounded.
- pilot_offline.py:430 launches a fresh interpreter, which inherits none of
  the monkeypatches. Path confinement is not byte/name admission. That child
  writes source records, one actual update, weights,16episode evaluation,
  replay, report/SVG and final report at478-549; Matplotlib/cache writes count.
- Offline stdout/stderr capture at439-447 has no prospective length bound;
  timeout and post-run output publication do not supply one.
- Existing64MiB pilot,128MiB evaluation and compact/gate per-file limits do
  not establish a total allocation or file/directory count bound.

## Smallest later preparation

Extend the existing test guard to the three aliases and final output families,
keeping genuine writers and fail-before-write behavior. Add an explicit,
reviewed bounded path for this one genuine offline child within the same parent
ledger, including temporaries, row appends, caches/logs and failures. Derive one
finite combined envelope including retained fit output and current lease copies.
Do not create a general quota framework or treat post-run du as admission.

## Source-authentic test strategy

Adapt gate_artifact_case (test_pilot_checks.py:46) to bounded current-source
checkout/history/data. Produce one genuine four-update CPU fit; execute all
final suites, continuations, diagnostics and actual offline child. No second
reference fit, substituted reports, old-source relabeling or omitted checks.
CPU checkpoint legitimately records missing MPS when native RNG evidence is
absent; preserve this non-acceptance result. Verify resulting gate eagerly and
through strict cold leases, test export roots and reuse with execution
tripwires, and reuse the same genuine new corpus for controller verification.
Include ordinary report outputs in the envelope if reporting is in scope.

Run cheap primitive RED tests first; generate this expensive corpus only after
its implementation/source and guard are committed. A full-gate total and run
admission remain unsupported until that bounded final-output/offline-child
slice is implemented and reviewed. Existing numerical/gate-input reader work
can continue independently without provider or fresh scientific execution.
