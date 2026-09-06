# Silent Cascade Delivery Index

The canonical requirements live in the [design specification](superpowers/specs/2026-08-30-silent-cascade-design.md). This file is navigation only and never duplicates or overrides that specification.

| Phase | Plan | Gate |
|---|---|---|
| 0 — Bootstrap | [Phase 0 implementation plan](superpowers/plans/2026-08-30-phase-0-bootstrap.md) | `uv sync --locked --group dev`, `make verify`, and `uv run silent-cascade doctor` |
| 1 — Generator and oracle | [Phase 1 implementation plan](superpowers/plans/2026-08-30-phase-1-generator-oracle.md) | Completed generator/oracle engineering evidence (not learned-model or benchmark evidence), including 100,000 independent-recipe generated/validated episodes, from source `c8486ae2423beb3f48e62675f90d76de733eecb3`: validation `2cccf61bc630e46288d49d0e14784e7d3fa780215b9d894fe2c28d06e97477c3`, oracle `139a807691d0c3a1516d8d2458576ff85ff982c46248169942cd37b335738560`, leakage `7b2ed5ad6f292ac917fa36300cbd6d1e16c8cdfb976e59a16ff6c2d70c5462d7`, matched reproducibility `a097d0df5035041ea94e7dcd7bfdef011819fa509f5cb343ea987fd267a24d89`, and independent reproducibility `51dc521d92a3333f45702c98ff0eeb4b17fc2eddbee4e66fb9011b3022d44497` (SHA-256); local `make verify` |
| 2 — Flow and event engine | Governed by a separate plan after the Phase 1 gate | Phase 1 must pass first |
| 3 — Neural components | Governed by a separate plan after the Phase 2 gate | Phase 2 must pass first |
| 4 — Autonomous EventFlow | Governed by a separate plan after the Phase 3 gate | Phase 3 must pass first |
| 5 — Strong baselines | Governed by a separate plan after the Phase 4 gate | Phase 4 must pass first |
| 6A — Freeze preparation | Governed by a separate plan after the Phase 5 gate | Phase 5 must pass first |
| 6B — Final runs | Governed by a separate plan after the explicit freeze checkpoint | Frozen protocol required |
| 7 — Optional showcase | Governed by a separate optional plan after Phase 6 | Scientific release path required |
