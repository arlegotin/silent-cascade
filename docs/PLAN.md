# Silent Cascade Delivery Index

The canonical requirements live in the [design specification](superpowers/specs/2026-08-30-silent-cascade-design.md). This file is navigation only and never duplicates or overrides that specification.

| Phase | Plan | Gate |
|---|---|---|
| 0 — Bootstrap | [Phase 0 implementation plan](superpowers/plans/2026-08-30-phase-0-bootstrap.md) | `uv sync --locked --group dev`, `make verify`, and `uv run silent-cascade doctor` |
| 1 — Generator and oracle | [Phase 1 implementation plan](superpowers/plans/2026-08-30-phase-1-generator-oracle.md) | Completed generator/oracle engineering evidence (not learned-model or benchmark evidence), including 100,000 independent-recipe generated/validated episodes, from source `0b00b970020856593c567a5ea8c5a9587659847e`: validation `aa8024724e74e219cdc06678948ceb1002a7aba8bcd247b6a6e5477b38035d69`, oracle `ca0e2bd25802d9b8bbfab58e6e3b57cff622bae8c4ca7dc9035d2852f5f34a1b`, leakage `7b1b1679399b6cac5ad9a3e8da46b074eb6749d3d027925e72b490965b16ad0c`, matched reproducibility `0338c442869c4e83dc327e7f1c4d1ec693b8e2993b51d7309ee3b2873dccb16c`, and independent reproducibility `acf7799cee4c216d24acec5a6600b9c00cc973a5693d7fc9ad16a4599dff803f` (SHA-256); local `make verify` |
| 2 — Flow and event engine | Governed by a separate plan after the Phase 1 gate | Phase 1 must pass first |
| 3 — Neural components | Governed by a separate plan after the Phase 2 gate | Phase 2 must pass first |
| 4 — Autonomous EventFlow | Governed by a separate plan after the Phase 3 gate | Phase 3 must pass first |
| 5 — Strong baselines | Governed by a separate plan after the Phase 4 gate | Phase 4 must pass first |
| 6A — Freeze preparation | Governed by a separate plan after the Phase 5 gate | Phase 5 must pass first |
| 6B — Final runs | Governed by a separate plan after the explicit freeze checkpoint | Frozen protocol required |
| 7 — Optional showcase | Governed by a separate optional plan after Phase 6 | Scientific release path required |
