# Phase 1 Task 7 Implementer Report

- Plan base revision: `98b49f9`
- Implementation commit: `ae5fe1e9f6d5a5c57d0f7e6858d7a960fbdecfed`
- Commit subject: `feat: generate matched OFD cohorts`

## RED / GREEN evidence

RED was captured with `uv run pytest -q tests/unit/test_generator.py` before
the construction API existed. All five tests failed for the expected reason:
`generate_matched_cohort` and `regenerate_matched_episode` were absent.

GREEN checks after the implementation:

- `uv run pytest -q tests/unit/test_generator_spec.py tests/unit/test_generator.py tests/unit/test_oracle.py` — 106 passed.
- `uv run ruff check src/silent_cascade/env/generator.py tests/unit/test_generator.py` — passed.
- `uv run ruff format --check src/silent_cascade/env/generator.py tests/unit/test_generator.py` — passed.
- `make verify` — Ruff passed, 368 tests passed, doctor passed, and local source/wheel builds succeeded.

## Matching audit

Each accepted cohort uses one cohort-scoped template and one deterministic
permutation of `[positive, positive, safe_negative, disconnected_negative]`.
All members share the template’s path topology, distractor count, delay,
observation-gap sequence, and activation gap. Each member independently
relabels all 64 entities, assigns the two hazard classes, and permutes its
facts. Every member has two hazards and one safe fact; only reachability
changes the terminal result. FACT IDs are assigned after presentation as
`0..N-1`, with ACTIVATE at `N` and the private terminal at `N+1`.

## Privacy and scorer audit

Public episodes contain only initialization, FACT events, and final ACTIVATE.
Private terminal truth, retry data, seed coordinates, and paths remain in
`EpisodeTruth`. Public IDs are allocated only after all four local shape checks
accept the cohort, and derive from the accepted attempt. Tests independently
solve each public episode with the oracle and verify it against private truth.
Exhaustion exposes only an opaque cohort SHA-256, attempt count, and aggregate
rejection reasons.

## RNG audit

The cohort uses the existing cohort-scoped LABEL, TEMPLATE, STRUCTURE, and
TIMESTAMPS domains (`member_index=-1`). NODE_PERMUTATION, TERMINALS, and
PRESENTATION use member indices `0..3`; their 16 recorded tokens are unique.
No global RNG is used.

## Deviations

No scientific-protocol deviations. `EpisodeTruth` validates its stored
`requested_path_length` against the number of nodes in `relevant_node_path`,
whereas `CohortRequest`/`CohortTemplate` use the configured number of LINK
edges. The generator therefore stores `request.requested_path_length + 1` in
`EpisodeRecipe`, matching the existing episode and oracle conventions.
