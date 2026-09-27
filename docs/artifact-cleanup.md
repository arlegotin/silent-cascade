# Artifact cleanup after project sunset

**Cleanup date:** 2026-09-28

**Scope:** Generated Silent Cascade evidence, local environments, and temporary test copies in the repository and in verified project-specific external paths.

The [sunset decision](project-sunset.md) ended further scientific execution after the adverse Phase 5A comparison. The raw runs and validation artifacts were subsequently removed to recover disk space. Source code, the canonical design and phase plans, compact reports, and small receipts remain. Those retained reports are historical summaries: their raw inputs are no longer present for independent replay or artifact re-verification. The final Phase 5A repository-wide `make verify` gate was interrupted before this cleanup and is not claimed as passed.

## Evidence retained in writing

- [Phase 4](phase4-autonomous-eventflow.md) passed its unchanged one-seed engineering gate at source `3b132253512f02c0ed9f2d774fefce3036019d91`. Its selected portable weights had SHA-256 `bc2850deb2bacb64d336750c5e824390e4a8eb45a3144467e3a61af84c49285b`; the canonical gate had SHA-256 `6353acb150fc5f46d10215e5b4206f08744818184cce3933f6d5b33e0eb75ab1`. Primary, two-hop, and robustness timed successes were 9,994/10,000, 9,991/10,000, and 9,966/10,000. The [compact Phase 4 report](../reports/phase4-pilot-v1/report.md) remains.
- [Phase 5A](phase5a-comparative-pilot.md) compared that EventFlow checkpoint with one independently trained activation-time ponderer. EventFlow scored 512/512 IID and 504/512 depth; the ponderer scored 512/512 on both at caps 12–24 while using about 13.3× and 13.8× less measured forward MACs at cap 12. Compressed execution of EventFlow's own weights scored 319/512 IID and 289/512 depth. The [compact comparison reports](../reports/phase5a-v1/milestone-b/report.md) remain.
- The Phase 5A [audit receipt](../reports/phase5a-v1/verification/receipt.json) recorded authentication of 7,168 corrected condition rows, recounting of 120,000 primary validation rows, and exact replay of 224 selected decisions and semantic traces. These checks were performed before raw deletion; they cannot be rerun from the cleaned checkout.

## Removed data

The table reports allocated disk space measured before deletion. File counts include file symlinks; directory counts are reported separately below. The temporary checkout set was included only where the files identified the `silent_cascade` package. Shared caches and unrelated temporary projects were excluded.

| Location and contents | Files | Allocated space |
| --- | ---: | ---: |
| Repository: `runs/`, validation manifests and gates, old Superpowers test artifacts, `.venv/`, `dist/`, and local caches | 42,737 | 5.49 GiB |
| Project Library store: `~/Library/Application Support/silent-cascade/` (Phase 3–5A runs and execution logs) | 712,621 | 46.47 GiB |
| Verified Silent Cascade checkouts under `/private/tmp/` | 54,691 | 5.62 GiB |
| Project pytest runs 102–104 under the user's macOS temporary directory | 1,571 | 1.51 GiB |
| Small Phase 5A logs and temporary files under `/private/tmp/` | 26 | <0.01 GiB |
| **Total** | **811,646** | **59.09 GiB** |

All 52 inventoried deletion roots were absent after cleanup. The set also contained 33,844 directories and 374 symlink entries. The repository directory fell from 6.11 GiB to 0.62 GiB. Free space increased by approximately 5.52 GiB on the Git volume and 54.04 GiB on the Data volume; volume readings can also reflect filesystem accounting and concurrent activity.

Normal Git history was not rewritten: earlier commits still contain copies of tracked validation artifacts, and `.git/` continues to occupy disk space. The present revision intentionally omits those artifacts. Repository delivery tests that require the deleted gate and manifests cannot pass in this archival checkout. Restoring tracked files from historical commits or regenerating and rerunning the scientific program would consume space and time again.
