# Silent Cascade Delivery Index

The canonical requirements live in the [design specification](superpowers/specs/2026-08-30-silent-cascade-design.md). This file is navigation only and never duplicates or overrides that specification.

| Phase | Plan | Gate |
|---|---|---|
| 0 — Bootstrap | [Phase 0 implementation plan](superpowers/plans/2026-08-30-phase-0-bootstrap.md) | `uv sync --locked --group dev`, `make verify`, and `uv run silent-cascade doctor` |
| 1 — Generator and oracle | [Phase 1 implementation plan](superpowers/plans/2026-08-30-phase-1-generator-oracle.md) | Complete at collector source `7b5f33c20b6842fdbf19928f5032fa0d22ddbf14`: validation `84926a3217b27b04d7ed3e41038447533636aea06bcdc85b787f148f3d737e07`; oracle `2f5ffa58bde0e6690bc0d673b2f42a5beaa7aa650a077512a8e3d0578d56c66c`; leakage `2446db0d57b0641ec4988dd8dd1cd66e16f5b65ee4b13fbdb1769ee50834e72b`; matched reproducibility `4909d2981691605101c324c553c087623ec81193eda8e926e11b009bcae4460a`; independent reproducibility `3b50e4ad71646ec882288c039a3569ea60b5e6e1bd1ac0bd854a2c5eb9f8c37a`. The v3 cross-artifact verifier and local `make verify` passed with zero foundation-model calls; this is generator/oracle engineering evidence, not learned-model or benchmark evidence. |
| 2 — Flow and event engine | [Approved Phase 2 implementation plan](superpowers/plans/2026-09-09-phase-2-flow-event-engine.md) | Complete at engine source `33fb8ed105046cb4fdb7be61eac50ed391415b7c` with gate artifact `c05b2f78680b14e5b8fd81597489f34ec68a1dc2d1114e62daa95d7428694c42`: the public-facts-only scripted EventFlow condition achieved 10,000/10,000 timed successes; numeric, long-silence, Zeno, checkpoint/resume, CPU replay, local `make verify`, and the independent artifact verifier passed with zero foundation-model calls. This is non-neural runtime engineering evidence, not learned-model or benchmark evidence. |
| 3 — Neural components | Governed by a separate plan after the Phase 2 gate | Phase 2 must pass first |
| 4 — Autonomous EventFlow | Governed by a separate plan after the Phase 3 gate | Phase 3 must pass first |
| 5 — Strong baselines | Governed by a separate plan after the Phase 4 gate | Phase 4 must pass first |
| 6A — Freeze preparation | Governed by a separate plan after the Phase 5 gate | Phase 5 must pass first |
| 6B — Final runs | Governed by a separate plan after the explicit freeze checkpoint | Frozen protocol required |
| 7 — Optional showcase | Governed by a separate optional plan after Phase 6 | Scientific release path required |
