# Phase 1 Task 16 Implementer Report

## RED / GREEN evidence

- RED: `uv run pytest -q tests/integration/test_phase1_services.py::test_inspection_request_requires_exactly_one_selector`
  failed with the expected missing `ConfigSelection` / `InspectEpisodeRequest`
  imports.
- RED: `uv run pytest -q tests/integration/test_phase1_services.py::test_freeze_with_test_dependencies_publishes_immutable_report`
  failed with the expected absent `Phase1ServiceDependencies` contract.
- RED: `uv run pytest -q tests/integration/test_phase1_services.py::test_debug_inspection_can_include_oracle_without_private_truth`
  failed with the expected absent `inspect_episode` service.
- RED: `uv run pytest -q tests/integration/test_phase1_services.py::test_oracle_evaluation_streams_bound_test_allocation`
  failed with the expected absent `evaluate_oracle` service.
- GREEN: `env UV_CACHE_DIR=/private/tmp/silent-cascade-task16-uv-cache uv run pytest -q tests/integration/test_phase1_services.py`
  passed: **8 passed**.
- GREEN: `env UV_CACHE_DIR=/private/tmp/silent-cascade-task16-uv-cache uv run ruff check src/silent_cascade/env/services.py tests/integration/test_phase1_services.py`
  passed with no findings; `ruff format` left the final files formatted.

## Production-boundary rulings

- Task 16 constructs audit examples only from the dependency-bound manifest
  regenerator or exact independent allocation iterator. It never accepts a
  caller-provided bundle sequence, and it hashes canonical source order before
  creating the separate `LeakageAuditEvidenceAnchor` capability.
- The production dependency object is constructed once from the frozen
  validation and Phase 1 gate allocations. Debug/test dependencies must use
  `test-` DEBUG allocations; their manifests can only be DEBUG, while production
  requires VALIDATION and exactly 10,000 episodes.
- Service reports publish canonical report bytes only. Dynamic `created` state
  is returned outside the persisted payload, so identical reuse is byte-stable;
  different existing payloads fail closed.
- Oracle evaluation regenerates and independently verifies every episode,
  derives exact per-stratum denominators and corpus digest in source order,
  performs separate exact positive/negative binomial diagnostics, and verifies
  declared paired clock transforms. No per-ID labels, oracle selections, seed
  fields, or private rejection details enter published reports.
- The leakage service binds provenance to the resolved config, exact audit
  seeds, source descriptor, source-manifest digest, clock manifest, and count
  matrix before invoking the Task 15 engine. It uses a caller-owned outer
  workspace and Task 15's two-pass cleanup; corruption/incomplete input raises
  before publication.

## Full verification status

- A direct complete-suite attempt under the normal sandbox first failed before
  project code ran because its reproducibility test could not write
  `.git/worktrees`; the same test proceeded after narrowly scoped local
  permission.
- The temporary task-local uv cache then lacked the offline `hatchling` build
  dependency for the sdist regression test. The existing uv cache was used for
  later local runs. The harness truncated long full-suite output after roughly
  40% rather than returning a terminal total, so this report does **not** claim
  a complete `make verify` result yet.

## Commit evidence

- `f48b1b3` — `feat: orchestrate Phase 1 data services`.

## Fix round 1 (in progress)

- Reviewer RED root causes reproduced from `task-16-review.md`: oracle evaluation
  did not call independent invariants; raw clock `OracleSolution` equality
  rejected valid time scaling; caller-created `production_mode=True`
  dependencies were accepted; manifest execution reused stale provenance.
- Interim GREEN: `env UV_CACHE_DIR=/private/tmp/silent-cascade-task16-uv-cache uv run pytest -q tests/integration/test_phase1_services.py` — 8 passed.
- Interim GREEN: scoped Ruff check and formatting are clean after adding
  dependency entry guards, independent per-episode/matched-cohort validation,
  normalized clock decision/window comparison, current manifest provenance
  collection, and the final `phase1_analysis` provenance scope.
- The required full reviewer acceptance matrix and full local `make verify` are
  still pending; this round is not a completion claim.
