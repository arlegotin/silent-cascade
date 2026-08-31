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

## Fix round 3: complete RED / GREEN evidence

### Formatting and manifest-clock authentication

- RED: the mandatory scoped formatter reported that
  `tests/integration/test_phase1_services.py` would be reformatted; the inherited
  `make verify` therefore exited during lint before tests, doctor, or builds.
- GREEN: after formatting, scoped Ruff check and format verification both
  passed. Commit `212f105` records only that gate restoration.
- RED: the first real manifest-clock fixture reached the independent invariant
  validator as an unscaled child and failed with the expected FACT/activation
  gap invariant error. After separating parent validation, an adversarial
  second regenerator returned a valid child derived from the wrong parent and
  pytest failed exactly with `Failed: DID NOT RAISE <class 'ValueError'>`.
- GREEN: Task 16 now reconstructs authenticated parents from each manifest
  entry's parent public ID/hash, regenerates the child independently, checks
  exact transform/order/counts, and compares normalized decision/window
  evidence. The focused manifest-clock set passed `3 passed, 9 deselected in
  1.15s`; commit `48b8556` records the fix.

### Independent leakage authority and executable source modes

- RED: the independent-authority test first failed with
  `Phase1ServiceDependencies.for_test() got an unexpected keyword argument
  'build_audit_anchor'`.
- GREEN: Task 16 now requires a separate immutable anchor-authority capability,
  calls it independently from source construction, and never copies source
  authentication into provenance. A poison test substitutes authentication
  from a second genuinely generated source while the anchor remains pinned;
  the real Task 15 boundary rejects it and publishes nothing.
- RED: the real matched-source path failed with `manifest audit provenance
  changed during source binding` when current execution commit/audit seeds
  correctly differed from immutable embedded source provenance.
- GREEN: current execution provenance is now distinct from immutable source
  provenance while authenticating the exact generator/config/allocation/root
  identity. Matched sources derive positive `0.1x` and `10x` clock pairs from
  every authenticated base parent. Both matched and independent 12-episode
  sources execute the real Task 16-to-Task 15 call, preserve complete
  scientific-failure reports, refuse corrupted authentication before
  publication, and clean the workspace. Commit `f0ca233` records this boundary.

### Acceptance matrix and allocation recipe authentication

- GREEN checkpoint before further expansion:
  `uv run pytest -q tests/integration/test_phase1_services.py` passed `24 passed
  in 5.27s`; scoped Ruff check and format verification passed. Commit `05fb48b`
  preserves that coherent acceptance slice.
- RED: an independently valid bundle generated with root seed 42 was accepted
  for the bound root-41 allocation. The focused regression failed exactly with
  `Failed: DID NOT RAISE <class 'ValueError'>`.
- GREEN: every base and clock-parent bundle is now authenticated against its
  exact allocation request: namespace, suite, root seed, episode/quartet
  coordinate, requested path, allocated variant, unscaled recipe, and derived
  public ID. Manifest regeneration is independently bound to entry ID/hash,
  accepted attempt, coordinate, suite, and path. The same checks protect oracle
  and both leakage source modes. Task 16 passed `26 passed in 5.22s`; commit
  `f164d55` records this fix.
- GREEN: the complete approved Step 1/5/9 matrix passed `35 passed in 5.85s`.
  It covers sealed production identity and the deferred 10,000 adapter spy;
  exact 8-16 test bounds and output-tree separation; all selector/source
  request errors; freeze reuse/divergence/privacy; authorized validation and
  renamed frozen access; current complete provenance and zero model calls;
  per-episode, cohort, allocation-recipe, suite-policy, denominator, rejection,
  and separate exact-random checks; allocation and manifest clocks; immutable
  publication reuse/divergence; both leakage modes and pass/scientific-failure/
  corruption behavior; workspace cleanup; typed counterfactual schemas and
  audit seeds; and the shared Task 2 oracle/leakage/reproducibility digest.
  Commits `8b3b8c1` and `5f27637` record the matrix and its bounded-resource
  publication checks. All real matched/allocation Task 15 calls remain; only
  duplicate audit reruns were replaced by direct immutable-report publication
  checks.

### Final local verification

- `uv run ruff check src/silent_cascade/env/services.py tests/integration/test_phase1_services.py`
  passed: `All checks passed!`.
- `uv run ruff format --check src/silent_cascade/env/services.py tests/integration/test_phase1_services.py`
  passed: `2 files already formatted`.
- `uv run pytest -q tests/integration/test_phase1_services.py tests/unit/test_manifest.py tests/unit/test_oracle.py tests/unit/test_leakage.py`
  passed: `285 passed in 198.01s`.
- The first two complete `make verify` attempts each reached `745 passed` but
  the existing full-profile leakage RSS test crossed its process ceiling after
  full-suite allocator retention. The same test passed alone (`1 passed in
  1.67s`) and directly after the Task 16 matrix (`36 passed in 6.31s`), ruling
  out a deterministic product failure and direct Task 16 leak. The ceiling was
  not changed. Removing only duplicate Task 15 executions from publication
  reuse checks preserved every real scientific path and bounded suite memory.
- Final `make verify` passed with normal local repository/cache access:
  repository Ruff passed, all 60 files were formatted, pytest passed `746
  passed in 211.98s`, `silent-cascade doctor` reported Overall PASS, and
  `uv build` produced both `dist/silent_cascade-0.1.0.tar.gz` and
  `dist/silent_cascade-0.1.0-py3-none-any.whl`.

## Fix round 3 commits

- `212f105` — `style: restore Task 16 formatting gate`
- `48b8556` — `fix: authenticate manifest clock evidence`
- `f0ca233` — `fix: separate leakage source authority`
- `05fb48b` — `test: exercise Task 16 evidence boundaries`
- `f164d55` — `fix: authenticate oracle source recipes`
- `8b3b8c1` — `test: complete Task 16 acceptance matrix`
- `5f27637` — `test: bound Task 16 acceptance resources`

## Fix round 4: sealed manifest source boundary

- RED: `uv run pytest -q tests/integration/test_phase1_services.py -k
  'default_production_manifest_services_refuse or
  injected_manifest_services_refuse'` failed `10 failed, 35 deselected in
  1.86s`. All six sealed-production cases reached regeneration for a valid
  DEBUG/`test-` manifest, a wrong-identity manifest, or a short VALIDATION
  manifest; all four injected cases reached work for a cross-allocation DEBUG
  or VALIDATION-access manifest. This reproduced the review's exact P1-6 root
  cause: manifests authenticated only their own declarations and were never
  bound to the service dependency's allocation/access authority.
- GREEN: one shared manifest boundary now requires sealed production sources to
  be the exact 10,000-entry VALIDATION allocation recipe, including access
  class, generation mode, allocation ID, namespace, suite, count, canonical
  cohort coordinates, and path denominators. Injected services require DEBUG
  and exact matched or independent dependency-bound allocation coordinates.
  Oracle and leakage validate before work, leakage validates again at source
  binding, and both reload, revalidate, and compare the source digest
  immediately before publication.
- GREEN: the ten negative regressions passed `10 passed, 35 deselected in
  1.56s`; the complete Task 16 integration suite passed `46 passed in 5.31s`;
  scoped Ruff check passed and both changed files were already formatted.
- The production-positive regression constructs the exact metadata-only
  10,000-entry validation recipe and passes the boundary without regenerating
  deferred Task 18 episodes. Existing matched DEBUG and independent clock
  fixtures remain accepted only after their allocation IDs were bound to their
  injected dependencies.
- `ce65f68` — `fix: seal production manifest sources`.

### Round 4 final local verification

- Scoped Ruff check passed with `All checks passed!`; scoped format verification
  passed with `2 files already formatted`.
- The Task 16 and upstream manifest/oracle/leakage suites passed `296 passed in
  183.64s`.
- `make verify` passed under normal local access: repository Ruff passed, all 60
  files were formatted, pytest passed `757 passed in 205.12s`,
  `silent-cascade doctor` reported Overall PASS, and `uv build` produced both
  `dist/silent_cascade-0.1.0.tar.gz` and
  `dist/silent_cascade-0.1.0-py3-none-any.whl`.
