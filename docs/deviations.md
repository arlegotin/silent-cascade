# Approved Deviations

This append-only log records necessary, explicitly approved departures from the canonical Silent Cascade design.

## Current status

## 2026-08-31 — Task 15 pre-freeze leakage regularization correction

The global leakage-probe L2 penalty was corrected from `1.0` to `0.03` before
Task 15 evidence was frozen. The originally specified mean class-balanced
cross-entropy plus `0.5 * 1.0 * ||W||^2` could represent, but would not fit,
the required three contiguous manifest-rank control blocks at the unchanged
`0.95` power gate. An exact 1,200-row synthetic known-answer test, using the
frozen train-only standardization and unpenalized multinomial intercept, gave
balanced accuracies `0.745833`, `0.837500`, `0.912500`, `0.970833`,
`0.995833`, `0.995833`, and `0.995833` for the predeclared penalty sweep
`1.0`, `0.3`, `0.1`, `0.03`, `0.01`, `0.003`, and `0.001`. The largest value
meeting the frozen gate was therefore selected. The `0.03` penalty applies
identically to clean, shuffled, permutation, and positive-control probes; no
control-only fit or threshold change is permitted.

## 2026-08-30 — Plan version 1.0.2

CI/CD was removed by explicit project-owner direction. Silent Cascade now uses
only local verification: `make verify` runs linting, formatting checks, tests,
the environment doctor, and the package build in the current workspace. No
hosted workflow, deployment pipeline, release pipeline, or third-party
automation is configured. This operational policy does not alter the frozen
scientific controls, metrics, acceptance gates, or zero-foundation-call primary
path.

## 2026-08-30 — Plan version 1.0.1

The optional `qwen` dependency range `huggingface-hub>=0.34,<1` was changed to
`huggingface-hub>=1,<2`. The original range was unsatisfiable because
`mlx-vlm>=0.6.17` depends on `mlx-audio>=0.4.3`, which requires
`huggingface-hub>=1`. With the corrected range, the universal dependency graph
resolved successfully and was locked. This packaging correction does not affect
primary scientific execution, which remains offline and records zero foundation-model calls.
