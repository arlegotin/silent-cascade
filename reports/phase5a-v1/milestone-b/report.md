# Phase 5A comparative pilot — Milestone B

Single-seed exploratory comparison only; no converged-baseline certification, final Phase 5 gate, or confirmatory scientific-support claim.

Training updates: 12000; last stage: robustness; selected at update 12000 by primary validation cap 24.

| Corpus | Condition | Timed success | Errors | Mean forward MACs |
| --- | --- | ---: | ---: | ---: |
| iid | intact_eventflow | 512/512 | 0 | 195508138 |
| iid | activation_ponder_cap4 | 371/512 | 0 | 13853987 |
| iid | activation_ponder_cap8 | 512/512 | 0 | 14739326 |
| iid | activation_ponder_cap12 | 512/512 | 0 | 14739326 |
| iid | activation_ponder_cap16 | 512/512 | 0 | 14739326 |
| iid | activation_ponder_cap24 | 512/512 | 0 | 14739326 |
| depth | intact_eventflow | 504/512 | 0 | 333427600 |
| depth | activation_ponder_cap4 | 39/512 | 0 | 14755136 |
| depth | activation_ponder_cap8 | 387/512 | 0 | 23524208 |
| depth | activation_ponder_cap12 | 512/512 | 0 | 24198752 |
| depth | activation_ponder_cap16 | 512/512 | 0 | 24198752 |
| depth | activation_ponder_cap24 | 512/512 | 0 | 24198752 |

Nearest measured IID compute cap: none within 10%.
Same cap overlaps on depth: False.
