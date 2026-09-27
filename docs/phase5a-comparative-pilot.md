# Phase 5A comparative pilot

Status on 2026-09-27: **Milestone A complete; Milestone B admitted; fixed DEBUG competence complete; competitive training pending.** This is an exploratory DEBUG comparison. It does not pass the full Phase 5 gate or establish competitive architectural merit.

## Frozen inputs and checks

- The evaluator used the accepted Phase 4 seed-11 EventFlow weights at update 12,000 (`bc2850deb2bacb64d336750c5e824390e4a8eb45a3144467e3a61af84c49285b`) and model state `2c5e5b2a0cb3749404ec3f2a5b70dc20c00fd8a2560a4035f577311fa375e30f`. The frozen Phase 4 gate, historical local verification receipt, training-result binding, producer revision, and portable tensors were authenticated without comparing the current Phase 5A package to the old producer tree.
- The intact bridge exactly reproduced the actions, scorer results, and error status of 32 preselected retained primary rows.
- The fresh [IID](../manifests/validation/phase5a-v1/iid.json) and [depth](../manifests/validation/phase5a-v1/depth.json) manifests contain 512 episodes each. Every episode passed generator invariants and the public oracle. Their public IDs had zero overlap with 1,536,000 accepted Phase 4 training exposures, 40,000 retained validation IDs, and 64 fixed debug IDs; the [namespace audit](../manifests/validation/phase5a-v1/namespace-audit.json) records that check. Manifest hashes were committed at `5640f57` before any fresh comparison.
- The evaluator revision was `d51a1fd`. Both conditions ran offline on CPU with exactly zero recorded foundation model calls and no dynamics errors. Complete per-episode rows, retained full traces, identities, and cumulative budgets are kept in the local run directory. The compact [Milestone A report](../reports/phase5a-v1/milestone-a/report.md), [decision](../reports/phase5a-v1/milestone-a/decision.json), and [artifact index](../reports/phase5a-v1/milestone-a/artifact-index.json) are versioned.

## Milestone A results

| Corpus | Intact EventFlow | Compressed EventFlow | Paired compressed minus intact |
| --- | ---: | ---: | ---: |
| IID, depths 2–4 | 512/512 (100.0%) | 319/512 (62.3%) | −37.7 percentage points; 95% interval −40.4 to −34.8 |
| Depth transfer, depths 5–8 | 504/512 (98.4%) | 289/512 (56.4%) | −42.0 percentage points; 95% interval −44.1 to −39.6 |

The intervals are **single-seed exploratory episode bootstrap intervals**, with 10,000 replicates, fixed analysis seed 8009, and sampling within depth/variant strata. They are not five-seed hierarchical intervals or confirmatory tests. The report retains integer numerators, variant and depth breakdowns, all actions and failure rows, end-to-end and post-activation compute, and the fixed 32-row timing subsets.

Intact EventFlow passed the predeclared B admission heuristics: IID at least 461/512 and depth at least 384/512, with complete valid pairs and no offline or dynamics failure. Compression's score had no role in admission. The same weights perform much worse when cognitive transitions are compressed to activation time. That observation establishes an execution-regime dependence for this checkpoint on these diagnostic samples. A model trained for compressed execution could behave differently; the competitive ponderer has not yet been trained or measured.

The four comparison units consumed 708.5 cumulative scientific seconds and retained 208.2 MB of reported evidence. The larger source preparation, checkpoint authentication, local tests, and reporting work is separate. No time or storage target extension was needed. Full traces were retained for the first 32 successful episodes and every unsuccessful episode in each condition/corpus. The local run directory is `/Users/artemlegotin/Library/Application Support/silent-cascade/runs/phase5a-v1` on the larger Data volume. That path is a local artifact location, not a portable dependency of the versioned compact report.

## Fixed DEBUG competence checkpoint

The independent seed-11 activation ponderer used recurrent width 800, with 2,855,044 trainable parameters (2,048 entity-table parameters); accepted EventFlow has 2,781,042 parameters. Its separate, nonpromotable 64-example DEBUG fit reached **64/64 autonomous timed successes at cap 24 after 600 physical updates**, below the fixed 1,000-update ceiling. The set had 16 one-hop, 16 two-hop, and 32 primary examples with complete quartets and fixed roots `(7963, 7993)`. The final rows were independently rescored against reconstructed private truth; all cognitive timestamps equaled public activation time, and all rows recorded zero foundation-model calls. Exact CPU checkpoint continuation and a controlled nine-transition execution were verified in the local tests. The fit consumed 64.55 scientific seconds and retained 424.6 MB on the Data volume. The [compact competence receipt](../reports/phase5a-v1/competence/receipt.json) binds the dataset, final evaluation, checkpoint, and cumulative budget hashes. This DEBUG success establishes execution readiness only; none of its weights or examples are promoted to competitive training.

## Local reproduction

Set the same offline environment used for the pilot (`UV_OFFLINE=1`, `UV_FROZEN=1`, `UV_NO_SYNC=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `PYTORCH_ENABLE_MPS_FALLBACK=0`). With access to the retained Phase 4 archive, run `scripts/run_phase5a.py` commands in order: `prepare`, `checkpoint-check`, `compatibility-check`, `evaluate-a`, then `report`. The exact arguments are in the [approved plan](superpowers/plans/2026-09-27-phase-5a-comparative-pilot.md#7-commands-to-deliver-during-implementation). Repeated compatible evaluations reuse immutable completed units or resume verified episode shards; they do not create new seeds.

Milestone B remains governed by the same approved plan. Its measured budget admission, fresh single-seed training, primary-validation selection, and all five cap evaluations are required before any statement about a trained competitor. Full Phase 5 still requires the later baseline and fairness work specified in the canonical design.
