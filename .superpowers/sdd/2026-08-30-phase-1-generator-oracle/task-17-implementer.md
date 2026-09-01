# Phase 1 Task 17 Implementer Report

## RED evidence

- `env UV_CACHE_DIR=/private/tmp/silent-cascade-task17-uv-cache uv run pytest
  -q tests/integration/test_cli_phase1.py
  tests/integration/test_phase1_gate_verifier.py
  tests/integration/test_cli_doctor.py
  tests/integration/test_phase0_repository.py
  tests/regression/test_import_boundaries.py` exited during collection with
  `ModuleNotFoundError: No module named 'scripts'` at the missing
  `verify_phase1_gate_artifacts` import. No production files had been edited.
- After the verifier slice reached GREEN, the same matrix failed `18` tests
  and passed `21`. The failures were the expected missing four CLI groups and
  service adapter symbols plus unchanged Make/documentation assertions.

## GREEN checkpoints

- Gate verifier: `14 passed in 6.75s`; scoped Ruff check passed and both files
  were formatted. Commit `863b16d` records the read-only verifier and its
  strict missing/tamper/report/provenance/call-count/counterfactual/denominator/
  corpus failure matrix.
- CLI/privacy: `26 passed in 4.42s`; scoped Ruff check passed and all four files
  were formatted. Commit `508ddad` records the four nested command adapters,
  exact request/mode/seed mapping, stable output/error handling, AST import
  guards, and public-projection spy.

## Final verification

- The complete focused Task 17 matrix passed: `88 passed in 15.33s`.
- `make smoke` passed: `47 passed in 10.35s`.
- Scoped Ruff check reported `All checks passed!`; scoped Ruff format check
  reported `7 files already formatted`.
- The first full `make verify` attempt used an isolated UV cache under the
  managed sandbox. Ruff passed and pytest reached `788 passed, 1 skipped`; its
  only two failures were environmental: the sandbox denied a test's temporary
  `.git/worktrees` write, and the isolated offline cache did not contain the
  Hatchling build dependency.
- The identical full local `make verify` rerun with repository-metadata and
  normal dependency-cache access passed: Ruff check passed, Ruff format
  reported `63 files already formatted`, pytest reported
  `791 passed in 410.30s`, doctor reported `Overall PASS`, and both the source
  distribution and wheel built successfully.
- Installed-command help exposed exactly `doctor`, `data`, `episode`, `oracle`,
  and `leakage`; the nested help surfaces exposed only `freeze`, `inspect`,
  `evaluate`, and `audit`, respectively.
- The final range audit from `b37331f` through `db099fa` contained only the
  eleven Task 17 files, no `.github`/workflow file, and no Task 18 artifact.
  `git diff --check` passed. Privacy coverage remained in the full green suite:
  AST oracle-import guards, blocked optional-model imports, public-projection
  truth-access spying, and the existing renamed frozen-manifest refusal.

## Fix Round 1

### Root-cause and RED evidence

- Real installed-command probes against the canonical 10,000-entry validation
  recipe reproduced Rich tracebacks for out-of-bounds index and unknown UUID
  inspection. A real `_publish_report` no-clobber conflict reproduced the full
  `FileExistsError` to `AtomicWriteError` to `ValueError` chain. The shared
  cause was the adapter catching `SilentCascadeError` while documented Task 16
  domain and integrity refusals cross the public service seam as `ValueError`.
- Coordinated canonical five-artifact mutations reproduced three verifier
  bypasses: validation sample size one with empty reproducibility schedules,
  a contradictory leakage audit authority, and a rehashed 10,000-entry
  validation manifest with every path length set to one. Every bypass returned
  a result with `passed=True` before the corrections.
- Verifier TDD RED: the new review matrix reported
  `19 failed, 10 passed, 14 deselected in 15.22s`. The failures were the exact
  missing schedule, leakage-authority, and validation-recipe checks; the ten
  passing cases were fields already rejected by strict report/manifest models.
- CLI TDD RED: the new adapter/progress matrix reported
  `12 failed, 16 deselected in 2.63s`. The failures were the three selector or
  config refusals, the real immutable publication conflict, and all six
  human/JSON progress paths plus the two prior success-output expectations.

### Corrective commits and GREEN evidence

- Commit `6760f7f` requires both reproducibility reports to declare the exact
  sample size 1,000, mode tuple, chunks `(1,3,7)`, hash seeds `(0,1)`, zero
  mismatches, and complete source-entry counts. It reconstructs the sealed
  full-gate leakage descriptor from exact allocation/config/generator identity
  and binds the audit authority's profile, allocation hash, descriptor hash,
  denominators, clock counts, and episode count. It also checks every ordered
  validation coordinate/member/path against the exact 834/833/833 allocation
  and experiment version without generation or network access. The new matrix
  passed `29` tests and the complete verifier file passed
  `43 passed in 25.66s`; scoped Ruff and formatting passed.
- Commit `e5afdf7` narrowly normalizes service `ValueError` refusals into the
  private-safe `phase1_command_error` payload while leaving programmer
  `TypeError` and `RuntimeError` visible. Freeze, oracle, and leakage emit exact
  deterministic started/completed progress lines to stderr in human and JSON
  modes; stdout remains the single final result. The targeted matrix passed
  `12` tests, the complete CLI/doctor matrix passed `31 passed in 4.84s`, and a
  real installed out-of-bounds probe exited one with one typed JSON error and
  no traceback.

### Final Round 1 verification

- Complete focused Task 17 matrix: `127 passed in 37.35s`.
- `make smoke`: `86 passed in 28.32s`.
- Scoped Ruff check: `All checks passed!`; scoped Ruff format:
  `7 files already formatted`.
- Full local `make verify`: Ruff check passed, Ruff format reported
  `63 files already formatted`, pytest reported `830 passed in 457.18s`,
  doctor reported `Overall PASS`, and both the source distribution and wheel
  built successfully.
