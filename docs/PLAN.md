# Silent Cascade Delivery Index

The canonical requirements live in the [design specification](superpowers/specs/2026-08-30-silent-cascade-design.md). This file is navigation only and never duplicates or overrides that specification.

| Phase | Plan | Gate |
|---|---|---|
| 0 — Bootstrap | [Phase 0 implementation plan](superpowers/plans/2026-08-30-phase-0-bootstrap.md) | `uv sync --locked --group dev`, `make verify`, and `uv run silent-cascade doctor` |
| 1 — Generator and oracle | [Phase 1 implementation plan](superpowers/plans/2026-08-30-phase-1-generator-oracle.md) | Completed generator/oracle engineering evidence (not learned-model or benchmark evidence), including 100,000 independent-recipe generated/validated episodes, from source `f43499a4f1bee8665e02c10a139ad236d9fa0322`: validation `dde1ad2b5db9df2d8dec48238be65510844aa00306cc05bc4fa7f3b26afc153a`, oracle `0e552391b2e04e8caacecd5797ad50cde52ce5a1b1202304c5046eec247ff9f3`, leakage `e060c33954d085e0cbda22077715665fe2e3ed8ba6a07cf6c203ca85a6088c02`, matched reproducibility `5c34beea37b5fb7e5c49b01b043762449568c23969526b6fa58589125093bab7`, and independent reproducibility `6463a11144653a4abb3e569e420ff37b47782d518ba0102d3f29f85174b91657` (SHA-256); local `make verify` |
| 2 — Flow and event engine | Governed by a separate plan after the Phase 1 gate | Phase 1 must pass first |
| 3 — Neural components | Governed by a separate plan after the Phase 2 gate | Phase 2 must pass first |
| 4 — Autonomous EventFlow | Governed by a separate plan after the Phase 3 gate | Phase 3 must pass first |
| 5 — Strong baselines | Governed by a separate plan after the Phase 4 gate | Phase 4 must pass first |
| 6A — Freeze preparation | Governed by a separate plan after the Phase 5 gate | Phase 5 must pass first |
| 6B — Final runs | Governed by a separate plan after the explicit freeze checkpoint | Frozen protocol required |
| 7 — Optional showcase | Governed by a separate optional plan after Phase 6 | Scientific release path required |
