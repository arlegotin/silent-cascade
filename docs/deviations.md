# Approved Deviations

This append-only log records necessary, explicitly approved departures from the canonical Silent Cascade design.

## Current status

## 2026-08-30 — Plan version 1.0.1

The optional `qwen` dependency range `huggingface-hub>=0.34,<1` was changed to
`huggingface-hub>=1,<2`. The original range was unsatisfiable because
`mlx-vlm>=0.6.17` depends on `mlx-audio>=0.4.3`, which requires
`huggingface-hub>=1`. With the corrected range, the universal dependency graph
resolved successfully and was locked. This packaging correction does not affect
primary scientific execution, which remains offline and records zero foundation-model calls.
