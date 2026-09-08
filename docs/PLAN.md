# Silent Cascade Delivery Index

The canonical requirements live in the [design specification](superpowers/specs/2026-08-30-silent-cascade-design.md). This file is navigation only and never duplicates or overrides that specification.

| Phase | Plan | Gate |
|---|---|---|
| 0 — Bootstrap | [Phase 0 implementation plan](superpowers/plans/2026-08-30-phase-0-bootstrap.md) | `uv sync --locked --group dev`, `make verify`, and `uv run silent-cascade doctor` |
| 1 — Generator and oracle | [Phase 1 implementation plan](superpowers/plans/2026-08-30-phase-1-generator-oracle.md) | Complete at collector source `bee142bd08a8b7b5621ae65551280ebdeed6c1b6`: validation `a3ca60f825ef1ec0ccc7a9b770a1eeccfb1d2ad48e0311e02bc5a1c99b2a3dcc`; oracle `86687de1e39cad77d258877fd9668032c5dc58abd4d42048aa343479a2c8dee2`; leakage `b9edadc7147818e32e701cefc78eac5ee391e76f39c0c5e59291a62d99c62771`; matched reproducibility `409ec745522a1c78a43b68fae83a0dae2f4e3311756925893080624451d2b829`; independent reproducibility `6e5268472204bb747734403cd84c1e46960ba285b7ed35eccae5a9da61814c8e`. The v2 cross-artifact verifier and local `make verify` passed with zero foundation-model calls; this is generator/oracle engineering evidence, not learned-model or benchmark evidence. |
| 2 — Flow and event engine | Governed by a separate plan after the Phase 1 gate | Phase 1 must pass first |
| 3 — Neural components | Governed by a separate plan after the Phase 2 gate | Phase 2 must pass first |
| 4 — Autonomous EventFlow | Governed by a separate plan after the Phase 3 gate | Phase 3 must pass first |
| 5 — Strong baselines | Governed by a separate plan after the Phase 4 gate | Phase 4 must pass first |
| 6A — Freeze preparation | Governed by a separate plan after the Phase 5 gate | Phase 5 must pass first |
| 6B — Final runs | Governed by a separate plan after the explicit freeze checkpoint | Frozen protocol required |
| 7 — Optional showcase | Governed by a separate optional plan after Phase 6 | Scientific release path required |
