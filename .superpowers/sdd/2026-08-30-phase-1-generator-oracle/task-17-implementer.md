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
