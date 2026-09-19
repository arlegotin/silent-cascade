### Spec Compliance

- ✅ **Spec compliant** for Task D1 over
  `a8bf9e095f99ade4e639b627eb28f30cf4d62b2f..39ecdf451090ae5c81d984f6e7a7c0c7166ac8a2`.
- The six approved custody APIs add the optional
  `offline_write_allowance=None` parameter and preserve the same object across
  every required edge: workflow entry and owned checks
  (`src/silent_cascade/train/pilot_workflow.py:400`,
  `src/silent_cascade/train/pilot_workflow.py:474`), public/owned/recovery checks
  (`src/silent_cascade/train/pilot_checks.py:344`,
  `src/silent_cascade/train/pilot_checks.py:387`,
  `src/silent_cascade/train/pilot_checks.py:441`,
  `src/silent_cascade/train/pilot_checks.py:584`), and the verification wrapper
  (`src/silent_cascade/train/pilot_verification.py:585`).
- Both terminal measurement edges translate only the keyword name, from
  `offline_write_allowance` to the existing low-level `write_allowance`; neither
  copies, coerces, or derives an allowance
  (`src/silent_cascade/train/pilot_checks.py:387`,
  `src/silent_cascade/train/pilot_verification.py:585`).
- All four existing preflight sites forward the allowance. The preflight itself
  performs exact-type validation before completed-artifact inspection and
  custody validation, while preserving both omitted/explicit `None` and safe
  completed reuse (`src/silent_cascade/train/pilot_offline_process.py:677`).
- The diff introduces no inferred/default capacity, custody/allowance pairing,
  schema or report field, authority or hold, cap, scientific, or policy change.
- ⚠️ This task-scoped review does not certify an actual diagnostic, inherited
  supervisor/private-descriptor integration, production hold equality, recovery
  copying, a full pilot, Task 5, Phase 4, the full suite, or `make verify`.

### Strengths

- The production change is mechanical and localized: optional keyword plumbing
  plus one validation guard. Existing behavior remains the default because every
  new parameter defaults to `None`.
- Identity-sensitive tests use `is` at each hop, so an accidental reconstruction
  or coercion would fail rather than merely compare equal
  (`tests/pilot/test_pilot_offline_process.py:1499`).
- The tests cover exact-type rejection for both a mapping and subclass, rejection
  before expensive work at all four preflight sites, omitted/explicit `None`,
  completed reuse without custody consumption, all public-to-owned forwarding
  paths, and both low-level measurement edges
  (`tests/pilot/test_pilot_offline_process.py:1499`).
- Recovery is appropriately limited to a wiring test with publication and costly
  operations stubbed; it does not manufacture a claim about real recovery.
- Focused unchanged-source check for the named low-level compatibility risk:
  `pilot_offline.measure_pilot_offline` already declares
  `write_allowance=None`, rejects non-exact types, and serializes `None` without
  deriving capacity (`src/silent_cascade/train/pilot_offline.py:578`,
  `src/silent_cascade/train/pilot_offline.py:595`,
  `src/silent_cascade/train/pilot_offline.py:637`). Thus explicit forwarding of
  `None` is behaviorally compatible with prior omission.
- Controller evidence records an independent frozen-source repeat of the 15
  focused cases, scoped Ruff/format/diff checks, and source equality, all with
  empty stderr and bounded capture
  (`operational/sdd/task5-offline-allowance-forwarding-controller.md:1`).

### Issues

#### Critical (Must Fix)

None.

#### Important (Should Fix)

None.

#### Minor (Nice to Have)

None.

### Verification Boundary

- I did not rerun tests, import project modules, invoke project commands, or
  inspect broader call sites. Runtime results above are the retained writer and
  controller evidence, not a new execution by this reviewer.
- No additional focused check is needed before accepting D1. Broader integration
  checks belong to their separately admitted tasks and must not be inferred from
  this verdict.

### Assessment

**Task quality:** Approved

**Reasoning:** The implementation satisfies every D1 forwarding and validation
constraint without expanding authority or policy. The focused tests exercise the
identity, ordering, `None`, and completed-reuse boundaries, and no Critical,
Important, or Minor code-quality issue is present in the reviewed diff.
