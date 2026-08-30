# Approved Deviations

This append-only log records necessary, explicitly approved departures from the canonical Silent Cascade design.

## Current status

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
