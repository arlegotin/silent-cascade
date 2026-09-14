# Task 3 Implementation Report

## Status

DONE

## Implemented

- Added canonical `context_features(...)` assembly in the required order: 456 latent,
  8 mode, 96 mean-pooled support, 2 public time, and 8 hypothesis features. Support
  pooling uses `support_mask` independently of retrieval eligibility, so consumed support
  records remain represented; empty support rows remain exact zero.
- Added the fixed learned retrieval scorer: `496 -> 256` SiLU query, `96 -> 256`
  bias-free record projection, scaled bilinear score, and `352 -> 256 -> 1` SiLU pair
  score. The debug profile uses its closed configured width without changing interfaces.
- Retrieval legality is exactly the supplied public `ModelContext.eligibility`; no subject
  equality, oracle reachability, record ID, or slot index enters scoring.
- Added safe, differentiable 100-wide previews with top score, top-two margin, entropy,
  eligible fraction, and soft top-4 record pooling. Preview gathers nonempty rows before
  softmax, produces exact zero for empty rows, gives zero margin/entropy for one candidate,
  and includes every record tied at the fourth-score boundary.
- Added host-only selected-record decoding with exact-score argmax and lowest record ID as
  the deterministic tie breaker. Learned modules never receive IDs.
- Preview calls `forward(context)` on every invocation and retains no cache or mutable state.
  A deterministic flowed-query test proves both selected ID and pooled preview change after
  public workspace state changes.
- Added optional non-owning mode-embedding injection. Standalone scorers own their table;
  Task 5 can instead own one shared mode table and inject it without duplicate parameter or
  state-dict registration.
- Added the scoped public-only `model_context` fixture built from Task 1/2 constructors and
  hand-authored public memories. It has zero workspace, declared modes, public time
  features, and no support/hypothesis state.

## Public Interfaces

- `models.common.context_features(context, mode_embedding) -> Tensor[B,570]`
- `memory.retrieval.RetrievalScores(raw_scores, masked_logits, eligible_mask)`
- `memory.retrieval.RetrievalPreview(features, has_candidate)`
- `memory.retrieval.RetrievalScorer(config, *, mode_embedding=None)`
- `RetrievalScorer.forward(context) -> RetrievalScores`
- `RetrievalScorer.preview(context) -> RetrievalPreview`
- `memory.retrieval.select_record_ids(scores, record_ids) -> tuple[int | None, ...]`

New modules remain direct imports. No existing initializer, frozen Phase 1/2 source,
configuration, lock, CLI, workflow, or release file was changed.

## TDD Evidence

### RED

Command:

```text
UV_CACHE_DIR=/tmp/silent-cascade-uv-cache uv run pytest -q tests/neural/test_retrieval.py
```

Relevant output before production implementation:

```text
FFFFFFFFFFs                                                              [100%]
E   ModuleNotFoundError: No module named 'silent_cascade.models.common'
E   ModuleNotFoundError: No module named 'silent_cascade.memory.retrieval'
10 failed, 1 skipped in 0.11s
```

The failure was expected: all tests constructed their real public-memory fixtures and then
failed only because the two planned production modules did not yet exist. The RED suite
already covered empty/one-candidate rows, fourth-boundary ties, wrong-subject eligibility,
consumed support pooling, host tie decoding, preview immutability, flowed-state selection,
single shared mode ownership, gradients, and permutation behavior.

### GREEN

Initial focused pass:

```text
UV_CACHE_DIR=/tmp/silent-cascade-uv-cache uv run pytest -q tests/neural/test_retrieval.py
..........s                                                              [100%]
10 passed, 1 skipped in 0.30s
```

The single skip was the MPS parameter under sandboxed hardware visibility.

Final scoped neural and import-boundary gate:

```text
UV_CACHE_DIR=/tmp/silent-cascade-uv-cache uv run pytest -q tests/neural tests/regression/test_import_boundaries.py
......................................................s................. [ 72%]
....s...............s.......                                             [100%]
97 passed, 3 skipped in 1.65s
```

All three skips are hardware-conditioned cases under sandbox visibility. The Task 3 MPS
case was run separately with native host access:

```text
UV_CACHE_DIR=/tmp/silent-cascade-uv-cache uv run pytest -q 'tests/neural/test_retrieval.py::test_encoder_and_scorer_have_finite_gradients_and_permutation_equivariance[mps]'
.                                                                        [100%]
1 passed in 1.93s
```

Repository-wide lint and formatting:

```text
UV_CACHE_DIR=/tmp/silent-cascade-uv-cache uv run ruff check .
All checks passed!
UV_CACHE_DIR=/tmp/silent-cascade-uv-cache uv run ruff format --check .
131 files already formatted
```

Per task instruction, the 16-minute `make verify` gate was left for the later phase task.

## Files Changed

- `src/silent_cascade/models/common.py` (new)
- `src/silent_cascade/memory/retrieval.py` (new)
- `tests/neural/test_retrieval.py` (new)
- `tests/neural/conftest.py` (scoped fixture extension)
- `.superpowers/sdd/2026-09-09-phase-3-neural-components/task-3-report.md` (new)

## Self-Review

- Completeness: checked every Task 3 acceptance item and both public output shapes. Empty
  rows never enter softmax; one-candidate metrics are finite; fourth-boundary ties are all
  pooled; the current workspace is re-scored for every preview.
- Mutation review: semantic masking, eligibility-dependent support pooling, slot-sensitive
  tie pooling, arbitrary slot tie decoding, stale preview reuse, zero preview for live rows,
  and missing scorer/encoder gradients each cause a named test to fail.
- Numerical review: raw scores remain finite for validated finite inputs/parameters;
  `-inf` is confined to illegal masked logits; entropy is evaluated only on live rows and
  avoids multiplying invalid `-inf` logits; empty output rows are initialized as zero.
- Ownership review: external mode tables are deliberately unregistered inside the scorer,
  while standalone mode tables are normal registered submodules. This gives Task 5 one
  explicit owner without duplicate trainable tables.
- Boundary review: neither new production module imports environment, generator, oracle,
  training, runtime memory, or host metadata implementations. The import-boundary
  regression remains green.
- Scope review: no frozen source, initializer, configuration, lock, CLI, CI, or unrelated
  working-tree file was touched.

## Debugging Note

The first targeted Ruff check reproduced one E501 line-length violation in the new host
decoder; after the minimal line wrap, `ruff format --check` then identified the two new
files as not yet canonically formatted. Root cause was local formatter state, not behavior.
Running the formatter on those two files and repeating the exact checks produced the clean
repository-wide result recorded above.

## Concerns

None.
