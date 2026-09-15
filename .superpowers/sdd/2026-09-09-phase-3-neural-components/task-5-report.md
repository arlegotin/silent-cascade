# Task 5 Implementation Report

## Status

DONE. Prediction heads, the public EventFlow model assembly, and measured neural
compute accounting are implemented and locally verified.

## Implementation

- Added immutable neutral `ComposePredictions` and `ActionPredictions` values.
- Added `ComposeHeads` with the canonical 666-wide context-plus-active-record
  trunk and separate role, focus, hazard, delay, deadline, status, confidence,
  support, and continuation outputs. `log_delay` is nonnegative; action lead is
  a sigmoid fraction; action classes include the fifth abstain class.
- Added `ActionHeads` with the canonical 570-wide context trunk.
- Added `EventFlowModel` with exactly one owning record encoder/entity table,
  one owning 6x8 mode table, scorer, controller, shared jump, external encoder,
  32-to-64 focus projection, and the two prediction heads. All injected shared
  references are non-owning and emit no state-dict aliases.
- Froze the public methods to `initial_context`, `observe`,
  `preview_and_control`, `recall_scores`, `compose`, `action`, and `jump` with
  only the approved public tensor arguments.
- `observe` applies the external fast/slow injection, resets guards, updates
  public time features, and installs `tanh(focus_projection(entity))` only on
  ACTIVATE rows. It does not retrieve or mutate caller-owned memory metadata.
- Retrieval preview and recall gather only post-activation rows. OBSERVING rows
  receive zero/no-candidate previews or ineligible `-inf` recall logits, so no
  all-memory scorer work occurs before activation.
- Added `NeuralComputeMeter` and immutable snapshots. Linear MACs use executed
  row counts; bilinear dot MACs are explicit; padded scored slots and eligible
  records are distinct; forward and backward estimates are distinct; bias,
  SiLU, LayerNorm, sigmoid/tanh, exp/log, and embedding-byte estimates are named.
- Instrumentation discovers the non-owning `ExternalEncoder.record_encoder` and
  injected mode tables exactly once, removes all hooks on exit, preserves
  outputs/gradients/RNG, synchronizes MPS around timing, and reports unsupported
  MPS peak allocation as `None`.
- Added functional flow/jump observer seams in the Phase 3 dynamics/jump modules
  so counts attach to successful state transitions rather than encoder calls.

## Frozen Interfaces

```text
initial_context(batch_size: int, *, device: str) -> ModelContext
observe(context: ModelContext, event: ExternalFeatures) -> ModelContext
preview_and_control(context: ModelContext) -> tuple[RetrievalPreview, BatchedSegmentParameters]
recall_scores(context: ModelContext) -> RetrievalScores
compose(context: ModelContext) -> ComposePredictions
action(context: ModelContext) -> ActionPredictions
jump(context: ModelContext, event_kinds: torch.Tensor) -> TensorWorkspace
```

`ComposePredictions` fields, in order: `role_logits`, `next_focus_logits`,
`hazard_logits`, `log_delay`, `normalized_deadline`, `status_logits`,
`confidence_logit`, `append_support_logit`, `continue_search_logit`.

`ActionPredictions` fields, in order: `class_logits`, `lead_fraction`.

## Parameter Count

Production profile, unique trainable parameters:

```text
total        2,781,042
entity_table     2,048
non_entity   2,778,994
```

The identity-deduplicated total equals entity plus non-entity and is inside the
approved 1,000,000–5,000,000 envelope.

## TDD Evidence

Initial RED, before production files existed:

```text
uv run pytest -q tests/neural/test_heads.py tests/neural/test_event_flow_model.py tests/neural/test_neural_compute.py
17 failed in 0.17s
ModuleNotFoundError for silent_cascade.models.heads, models.event_flow, and eval.compute
```

The failures were expected because the Task 5 interfaces did not yet exist.

Accounting refinement RED 1:

```text
uv run pytest -q tests/neural/test_neural_compute.py::test_external_encoding_without_injection_is_not_a_jump
1 failed in 0.08s
expected jump_applications == 0, observed 1
```

The counter was attached to encoding rather than the successful state change.

Accounting refinement RED 2, after pinning nested encoder work:

```text
uv run pytest -q tests/neural/test_neural_compute.py::test_external_encoding_without_injection_is_not_a_jump
1 failed in 0.08s
expected 298880 forward MACs, observed 263168
```

The exact 35,712-MAC deficit was the executed `90 -> 192 -> 96` network reached
through the external encoder's non-owning record-encoder reference.

Final focused GREEN:

```text
uv run pytest -q tests/neural/test_heads.py tests/neural/test_event_flow_model.py tests/neural/test_neural_compute.py
20 passed in 0.41s
```

## Broader Verification

```text
uv run pytest -q tests/neural tests/regression/test_import_boundaries.py tests/unit/test_flow.py tests/unit/test_guards.py tests/unit/test_eventflow_protocols.py tests/unit/test_eventflow_state.py
332 passed in 2.53s
```

```text
uv run ruff check <nine Task 5 production/test paths>
All checks passed!

uv run ruff format --check <nine Task 5 production/test paths>
9 files already formatted
```

`git diff --check` produced no output. The initial sandbox-limited interpreter
check hid MPS and was incorrectly reported as host unavailability. An approved
`uv run` check reports `mps_available=True`, and the native-MPS coverage passes.
PyTorch still exposes no true peak-allocation API, so `None` remains the correct
peak result. Per the phase instruction, the 16-minute repository-wide
`make verify` was intentionally deferred.

## Files Changed

- `src/silent_cascade/models/heads.py`
- `src/silent_cascade/models/event_flow.py`
- `src/silent_cascade/eval/__init__.py`
- `src/silent_cascade/eval/compute.py`
- `src/silent_cascade/models/dynamics.py`
- `src/silent_cascade/models/jump.py`
- `src/silent_cascade/memory/retrieval.py` (functional accounting seam)
- `tests/neural/test_heads.py`
- `tests/neural/test_event_flow_model.py`
- `tests/neural/test_neural_compute.py`

## Self-Review

- Confirmed every required output shape and gradient path, fifth abstain class,
  bounded predictions, shared-table identity, one state-dict owner, ACTIVATE
  focus installation, no observation retrieval, mixed-batch executed-row
  scoring, separate preview/recall work, exact linear/bilinear counts, hook
  removal, immutable snapshots, and unchanged output/gradient/RNG behavior.
- Confirmed external record-network work is measured despite bypassing
  `RecordEncoder.forward`.
- No Phase 1/2, initializer, dependency lock, root CLI, workflow, or canonical
  specification file was changed.

## Concerns

No correctness concerns. Platform MPS peak APIs expose current allocation but
not a true peak in this installed PyTorch, so the snapshot deliberately reports
`None`, as required, instead of fabricating zero or mislabeling an endpoint.

## Fix Round 1 — Review Findings

### Corrections

- Retain every handle returned by `Tensor.register_hook` and remove it in the
  meter's unconditional exit cleanup. A retained output now has an empty actual
  tensor hook mapping after the context exits; backward outside the context does
  not change the meter, while a later measured backward counts normally.
- Forward hooks now retain only public eligibility/mode tensor references and
  integer shape metadata. After MPS synchronization and after stopping elapsed
  timing, exit copies those tensors to CPU, performs reductions there, freezes
  integer totals, and releases the references. A Torch dispatch trace proves
  measured and unmeasured forward calls execute the same tensor operations.
- Added metadata-only functional seams for external/shared jump sigmoid+tanh,
  ACTIVATE focus tanh, and retrieval-preview softmax/logsumexp. Removed the old
  SharedJump module estimate so the internal jump remains single-counted.
- Added a durable complete assembled-model MPS regression covering observe,
  preview/controller, recall, compose, action, jump, flow, and backward.

### TDD RED

```text
uv run pytest -q <five focused review tests>
4 failed, 1 passed in 1.07s
```

Observed failures were the retained tensor hook, five extra measured dispatch
operations including `aten.sum`, missing external/focus sigmoid-tanh keys, and
missing retrieval softmax/logsumexp keys. The single-count SharedJump baseline
passed before the seam change.

### Incremental GREEN

```text
tensor-hook lifecycle/backward tests: 3 passed in 0.06s
metadata-only dispatch/count tests:   3 passed in 0.52s
functional nonlinear count tests:    3 passed in 0.10s
native assembled MPS regression:     1 passed in 1.21s
```

The approved-context device check was:

```text
uv run python -c "import torch; print(f'mps_available={torch.backends.mps.is_available()}')"
mps_available=True
```

### Fix-Round Verification

```text
uv run pytest -q tests/neural/test_heads.py tests/neural/test_event_flow_model.py tests/neural/test_neural_compute.py
26 passed in 0.90s

uv run pytest -q tests/neural tests/regression/test_import_boundaries.py tests/unit/test_flow.py tests/unit/test_guards.py tests/unit/test_eventflow_protocols.py tests/unit/test_eventflow_state.py
338 passed in 2.72s

uv run ruff check <ten scoped production/test paths>
All checks passed!

uv run ruff format --check <ten scoped production/test paths>
10 files already formatted
```

The assembled native-MPS run produced float32 finite outputs and gradients,
counted 256 scored records and zero foundation-model calls, retained the exact
`2,781,042 / 2,048 / 2,778,994` parameter split, and reported peak allocation
as `None` because the installed PyTorch has no true MPS peak API.
