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

## Fix Round 2

### Root-cause and RED evidence

- A read-only counterexample replaced only independent reproducibility's
  `source_payload_sha256` with an arbitrary digest and still returned
  `passed=True`. A second counterexample recomputed every validation and
  independent manifest/report/descriptor digest around coordinated alternate
  root/public seed pairs and also returned `passed=True`. The verifier did not
  authenticate the Task 14 independent source descriptor or the predeclared
  Phase 1 evidence seeds.
- The verifier TDD matrix covered an arbitrary source hash, all seven mutable
  descriptor coordinates (the eighth field is a strict literal), and coordinated
  validation-only, independent-only, and combined seed replacement. Before
  implementation it reported
  `11 failed, 43 deselected in 7.79s`.
- An adapter-boundary probe that raised `ValueError("programmer bug")` was
  converted into `phase1_command_error`, while equivalent `TypeError` and
  `RuntimeError` values propagated. After adding all six documented Task 16
  selector/config/publication refusals plus exact-match near-misses, the CLI
  TDD matrix reported `3 failed, 8 passed, 25 deselected in 2.24s`. The three
  failures were the arbitrary `ValueError`, selector trailing-space near-miss,
  and publication appended-detail near-miss.

### Corrective commits and GREEN evidence

- Commit `5664d8e` reconstructs `IndependentSourceDescriptor` from the literal
  Phase 1 gate allocation identity and the report's authenticated namespace,
  root/public-seed fingerprint, config, and generator provenance, then requires
  its canonical SHA-256. It also pins validation seeds
  `2026083001/2026083002` and independent seeds
  `2026083011/2026083012` through their exact root values, raw validation public
  seed, and literal public-seed fingerprints. The new 11-case matrix passed,
  the complete verifier file reported `54 passed in 28.34s`, and scoped Ruff,
  formatting, and diff checks passed.
- Commit `4235c7a` centralizes an immutable exact-match set containing only the
  six documented Task 16 selector/config/publication refusal messages. Only a
  `ValueError` whose complete message is in that set receives the stable
  private-safe exit-one payload; every other `ValueError` is re-raised unchanged,
  and `TypeError`/`RuntimeError` remain outside the catch boundary. The targeted
  matrix reported `12 passed, 24 deselected in 2.53s`; complete CLI/doctor tests
  reported `39 passed in 4.56s`; scoped Ruff, formatting, and diff checks passed.

### Final Round 2 verification

- Complete focused Task 17 matrix: `146 passed in 42.03s`.
- `make smoke`: `105 passed in 32.83s`.
- Scoped Ruff check: `All checks passed!`; scoped Ruff format:
  `7 files already formatted`.
- Full local `make verify`: Ruff check passed, Ruff format reported
  `63 files already formatted`, pytest reported `849 passed in 499.45s`,
  doctor reported `Overall PASS`, and both
  `dist/silent_cascade-0.1.0.tar.gz` and
  `dist/silent_cascade-0.1.0-py3-none-any.whl` built successfully.
- The range from `f95303e` through `4235c7a` contains only the verifier, CLI,
  and their two integration-test files (`232` insertions, `7` deletions), with
  no Make target, CI/CD, or Task 18 change. `git diff --check` passed. The full
  focused and local gates retained the AST import/privacy boundaries, public
  inspection protections, frozen-access refusal, read-only/no-regeneration
  verifier contract, deterministic stderr progress, and four-command surface.

## Fix Round 3

### Root-cause and RED evidence

- `data freeze --output` published `ManifestFreezeReport` bytes through the
  generic report writer even though every downstream consumer requires the
  canonical Task 13 `EpisodeManifest` envelope. The focused load-after-freeze
  test failed once in `2.88s` with 11 envelope validation errors; the actual
  CLI freeze-to-load path independently failed once in `3.03s` at the same
  boundary.
- Manifest leakage mode had no executable production pairing: the CLI admitted
  `manifest + phase1-gate`, while the service minted a TEST authentication for
  all 10,000 validation rows. The new selector-helper test failed at collection
  with the missing `_manifest_test_profile_entries` import, and the production
  metadata test then failed because the descriptor declared 10,000 rather than
  the frozen 1,200-example TEST source.
- Routine service refusals crossed the adapter as a mixture of raw `ValueError`
  and typed errors, forcing the CLI to maintain a finite string allowlist. The
  CLI RED matrix reported `8 failed, 21 passed, 14 deselected in 2.40s`, covering
  the unsupported profile pairing, every former allowlisted programmer
  `ValueError`, and publication typing. The service refusal inventory reported
  `21 failed, 40 deselected in 3.50s`; the additional oracle/leakage
  config/provenance matrix reported `4 failed in 2.46s`.

### Corrective commits and GREEN evidence

- Commit `80023fd` publishes the verified manifest with Task 13
  `publish_manifest`, maps `ManifestPublication` into the freeze result, and
  keeps the human/JSON result as the summary plus publication metadata. Exact
  identical reuse remains immutable (`created=false`), while divergent files
  and directory targets fail through typed manifest/artifact boundaries without
  changing bytes or modification time.
- The same commit makes `manifest + TEST` the supported audit pairing. For the
  exact production validation manifest it independently hashes and ranks whole
  matched cohorts with audit seed `2026083091` and the full canonical manifest
  digest, selects 100 cohorts at each path 2/3/4, and emits their 1,200 examples
  in original manifest order. The descriptor retains the full-manifest digest;
  source and clock hashes, TEST denominators (`400` per path), both positive
  clock strata, and the anchor authenticate the selected source. Injected DEBUG
  manifests retain all of their complete cohorts. Task 15 normalizes only the
  validation audit stratum label to its declared `iid_primary` TEST authority;
  source bytes and frozen controls are unchanged.
- Commit `e67a45f` replaces the CLI message allowlist with the principled typed
  boundary: routine user/artifact/config/provenance/episode refusals are
  `SilentCascadeError` subclasses at the services that own them, the adapter
  catches only `SilentCascadeError`, and arbitrary `ValueError`, `TypeError`,
  and `RuntimeError` remain visible. Stable typed stderr deliberately discards
  diagnostic context, so private paths or sentinels cannot cross the public CLI
  boundary. Typer still owns invalid mode/profile requests and exits two.
- The core freeze/subset matrix passed `5 passed in 5.99s`; the complete service
  file passed `61 passed in 18.92s`; the complete CLI file passed
  `45 passed in 3.64s`; and their combined post-format run passed
  `106 passed in 18.70s`.

### Final Round 3 verification

- Task 16/17 plus manifest, leakage, oracle, config, reproducibility, fixture,
  and import/privacy upstream suites passed `548 passed in 479.82s`.
- `make smoke` passed `114 passed in 37.51s`.
- Repository-wide Ruff check reported `All checks passed!`; Ruff format check
  reported `63 files already formatted`.
- Full local `make verify`: Ruff check passed, Ruff format reported
  `63 files already formatted`, pytest reported
  `873 passed in 519.17s`, doctor reported `Overall PASS`, and both
  `dist/silent_cascade-0.1.0.tar.gz` and
  `dist/silent_cascade-0.1.0-py3-none-any.whl` built successfully.
