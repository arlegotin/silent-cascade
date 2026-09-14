# Task 4 Implementation Report

## Status

DONE

## Implemented

- Added `BatchedSegmentParameters` with the exact `[B,456]` latent target/rate and
  `[B,3]` raw guard target, mode-masked execution target, and guard-rate fields.
- Added vectorized analytic flow using `-expm1(-rate * dt)` and active-only guard
  crossing evaluation using the stable `log1p` expression. The flow accepts fixed
  parameters from its caller and contains no controller, queue, timer, event ID, or
  horizon behavior.
- Added a learned `670 -> 512 -> 512 -> 918` production controller (closed debug
  widths remain supported), selective LayerNorm, bounded latent/guard targets and
  rates, raw guard targets for loss, and execution targets masked from the one
  authoritative scalar `allowed_mode_mask` mapping in RECALL/COMPOSE/ACT order.
- Added the `674 -> 512 -> 912` shared event-conditioned jump with active-record
  gathering, convex bounded latent updates, and exact zero guard resets.
- Added current-event-only `ExternalFeatures`. It contains public single-record
  fields, activation entity, FACT/ACTIVATE kind, and public time features; it has no
  padding event, future fact, private label, event ID, deadline target, or oracle
  representation.
- Added the `132 -> 256 -> 256` external encoder and owned `256 -> 640` injection
  projection. It reuses the injected `RecordEncoder` entity/category tables and
  record MLP without registering a duplicate owner. External injection updates only
  fast/slow channels through a bounded convex jump and resets all guards.
- Focus projection (`32 -> 64`) and ACTIVATE focus-key installation deliberately
  remain Task 5 responsibilities of `EventFlowModel.observe`. Task 4's
  `inject_external` is only the fast/slow bounded primitive and does not preempt that
  assembler-owned behavior.
- Added optional non-owning mode-table injection to controller and shared jump,
  matching the reviewed scorer pattern so Task 5 can own exactly one shared mode
  table. Standalone modules retain normal owned tables.
- Added `TensorWorkspace._from_functional_update`, a private structural boundary
  used only after trusted analytic/convex equations. Public `TensorWorkspace`
  construction keeps all finite and bounds scans and cloning; functional updates
  avoid repeating those device-synchronizing scans. Tests replace public
  `__post_init__` with a failure to prove flow/external injection use the internal
  boundary while their public inputs remain validated.

## Public Interfaces

- `models.dynamics.BatchedSegmentParameters`
- `models.dynamics.flow_batch(state, parameters, dt) -> TensorWorkspace`
- `models.dynamics.crossings_batch(accumulators, targets, rates) -> Tensor[B,3]`
- `models.controller.FlowGuardController(config, *, mode_embedding=None)`
- `models.jump.SharedJump(config, *, mode_embedding=None)`
- `models.jump.ExternalEncoder(config, record_encoder)`
- `models.jump.inject_external(state, raw_injection) -> TensorWorkspace`
- `models.types.ExternalFeatures`

All modules remain direct imports. Existing initializers, frozen Phase 1/2 source,
configuration, lock, root CLI, workflow, and release files are unchanged.

## TDD Evidence

### RED

Command before production implementation:

```text
UV_CACHE_DIR=/tmp/silent-cascade-uv-cache uv run pytest -q \
  tests/neural/test_neural_dynamics.py tests/neural/test_controller.py \
  tests/neural/test_neural_jump.py
```

Relevant output:

```text
FFFFFFFFFFFFsFFFFFFFFFFFFFFFs                                            [100%]
27 failed, 2 skipped in 0.16s
E   ModuleNotFoundError: No module named 'silent_cascade.models.dynamics'
E   ModuleNotFoundError: No module named 'silent_cascade.models.controller'
E   ModuleNotFoundError: No module named 'silent_cascade.models.jump'
E   ImportError: cannot import name 'ExternalFeatures'
```

Every failure was expected: the tests exercised the real planned interfaces and
failed because no Task 4 production module/type existed. The RED suite already
covered scalar parity, zero/tiny/huge/split flow, active-only crossings and
gradients, invalid elapsed time, saturation, authoritative mode masks, shared
ownership, external causal inputs, resets, repeated bounded jumps, and MPS.

### GREEN

Focused CPU/sandbox run after implementation:

```text
............s...............s                                            [100%]
27 passed, 2 skipped in 0.36s
```

Scoped neural, frozen scalar flow/guards, and import-boundary regression run:

```text
...........................................................s............ [ 33%]
...........s..........s....s.......s...............s.................... [ 67%]
.s........................................s...................s.......   [100%]
205 passed, 10 skipped in 1.78s
```

Native Apple MPS run outside sandbox hardware isolation:

```text
...                                                                      [100%]
3 passed in 0.32s
```

These three tests execute controller prediction, batched flow/crossings, and both
external/internal jumps, asserting actual MPS device placement and float32 outputs.

## Files Changed

- `src/silent_cascade/models/types.py`
- `src/silent_cascade/models/dynamics.py` (new)
- `src/silent_cascade/models/controller.py` (new)
- `src/silent_cascade/models/jump.py` (new)
- `tests/neural/test_neural_dynamics.py` (new)
- `tests/neural/test_controller.py` (new)
- `tests/neural/test_neural_jump.py` (new)
- `.superpowers/sdd/2026-09-09-phase-3-neural-components/task-4-report.md` (new)

## Self-Review

- Completeness: checked every Task 4 brief item against implementation and tests.
  Controller output slicing is exactly `456+456+3+3`; context/preview input is 670;
  jump input is `570+96+8=674`; external input is `96+32+2+2=132`.
- Causality: model inputs contain only current public data. No environment, generator,
  oracle, training target, queue, scheduler, timer, or future-event code is imported.
- Numerical safety: public boundaries reject malformed/nonfinite/out-of-domain tensors;
  hot equations operate only after validation. Dormant entries never enter logarithms;
  their sole output sentinel is `inf`. Flow and convex jumps preserve bounds by
  construction; exact-zero elapsed time is value-identical. Signed-zero bit patterns
  are not part of this batched training-kernel contract; the frozen scalar runtime's
  separate zero-time identity behavior remains unchanged.
- Synchronization boundary: public `TensorWorkspace`, `ModelContext`,
  `ExternalFeatures`, and `BatchedSegmentParameters` construction retain finite/bounds
  scans; controller output construction therefore still validates its complete emitted
  segment. `flow_batch` validates finite/nonnegative `dt`, `crossings_batch` validates
  its public tensors, and jump/event inputs validate their finite/range requirements.
  Only the final trusted workspace allocation after analytic/convex equations skips
  repeated scans. Task 7 should gather/validate at its batch boundary and use functional
  workspace returns rather than publicly reconstructing a workspace/context for each
  tensor assignment; no public validation was removed here.
- Gradient safety: public constructors clone without detaching, the private functional
  boundary retains graphs, active crossing entries differentiate, dormant entries get
  zero gradient, and encoder/controller/jump gradients are finite.
- Ownership: injected mode and record encoders are plain references, so they do not
  create duplicate named parameters or state-dict registrations. Their computation
  still participates in autograd under the future Task 5 owner.
- Mutation review: changing the stable flow equation, evaluating dormant logarithms,
  omitting resets, allowing an illegal mode guard, removing raw targets, injecting all
  latent channels, bypassing shared encodings, or using additive unbounded jumps causes
  a named behavioral test to fail.
- Scope: only the allowed new Task 4 modules/tests, one needed Task 1 type extension,
  its internal functional construction boundary, and this report changed.

## Debugging Note

The first GREEN attempt reported 26 passes and one failure. Systematic tracing showed
the scalar parity fixture called frozen `start_segment` with nonzero guard accumulators,
which correctly violates the post-causal-jump reset contract. The production batch flow
was not implicated. The minimal fixture correction uses a real reset segment origin;
the parity test then passed. Initial Ruff output found only six line-length formatting
violations in new tests; canonical formatting and a repeated check resolved them.

## Concerns

None.

---

## Fix Round 1: Bounded Crossing Batches and Dormant Autograd

### Review Findings Verified

1. `crossings_batch` derived `batch_size` from the first dimension but never applied
   the shared `1..128` bound, so `(0,3)` and `(129,3)` tensors passed validation.
2. `_crossing_values` returned a newly allocated `full_like` when `active.any()` was
   false. Besides the avoidable device synchronization and host branch, this output
   had no autograd connection to accumulators, targets, or rates. A finite-selected
   empty loss therefore could not call `backward()`.
3. The original gradient tests established finite gradients independently but did not
   numerically compare them with the completed scalar row kernels.

### TDD RED

Added behavioral regressions for both out-of-range batch sizes, all-dormant
finite-selected empty-loss backward on CPU/MPS, and value/gradient parity for one
analytic flow entry and one active crossing against actual scalar `state_at` and
`crossing_offsets_tensor` rows.

Command before production correction:

```text
UV_CACHE_DIR=/tmp/silent-cascade-uv-cache uv run pytest -q \
  tests/neural/test_neural_dynamics.py \
  -k 'batch_sizes_outside or all_dormant or value_and_gradients_match'
```

Relevant output:

```text
..FFFs                                                                   [100%]
E   Failed: DID NOT RAISE <class 'silent_cascade.models.errors.NeuralError'>
E   Failed: DID NOT RAISE <class 'silent_cascade.models.errors.NeuralError'>
E   assert False  # loss.requires_grad
3 failed, 2 passed, 1 skipped, 13 deselected in 0.27s
```

The two scalar gradient-parity controls passed before the fix. The three expected
failures isolated the missing batch bound and graph-disconnected dormant branch.

### Fix

- Reject crossing batches outside `1..128` immediately after rank validation and
  before validating any peer tensors.
- Remove `bool(active.any())` and always compute the stable formula on the active
  slices. For an all-dormant batch those slices are empty, so no dormant value enters
  a logarithm, while `masked_scatter` retains the zero-gradient autograd connection.
- Corrected the original report's exact-zero wording from bit-identical to
  value-identical; no nondifferentiable signed-zero special case was introduced.

### GREEN

Focused review regressions:

```text
.....s                                                                   [100%]
5 passed, 1 skipped, 13 deselected in 0.25s
```

Complete Task 4 suite:

```text
................s.s.....s..........s                                     [100%]
32 passed, 4 skipped in 0.36s
```

Scoped neural, frozen scalar flow/guards, and import-boundary gate:

```text
........................................................s...s........... [ 32%]
................s.s..........s....s.......s...............s............. [ 65%]
........s........................................s...................s.. [ 97%]
.....                                                                    [100%]
210 passed, 11 skipped in 1.78s
```

Native Apple MPS Task 4 regressions, including all-dormant backward:

```text
....                                                                     [100%]
4 passed in 0.23s
```

Repository-wide static verification:

```text
All checks passed!
137 files already formatted
```

### Fix-Round Files Changed

- `src/silent_cascade/models/dynamics.py`
- `tests/neural/test_neural_dynamics.py`
- `.superpowers/sdd/2026-09-09-phase-3-neural-components/task-4-report.md`

### Fix-Round Self-Review

- Mutation check: removing the batch bound fails both endpoint tests; restoring the
  host conditional makes the empty loss graph-disconnected and fails CPU/MPS; changing
  either flow or crossing derivatives fails direct scalar gradient parity.
- Dormant safety: only values selected by the active mask enter division and `log1p`.
  The public result remains `inf` for every dormant entry and no loss reduces it.
- Scope: no controller, jump, type schema, frozen scalar source, initializer,
  configuration, lock, CLI, or unrelated file changed in this fix round.

### Fix-Round Concerns

None.
