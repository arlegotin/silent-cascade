# Phase 4 autonomous EventFlow: action-context diagnostic

## Scope and status

The 2026-09-17 Task 7 diagnostic used the unchanged, previously trained Phase 3
one-hop weights and the original 10,000-example corpus. It reproduced the
historical immediate five-way score and completed all four readouts offline on
CPU, with zero optimizer steps and zero foundation-model calls.

This is **diagnostic, non-acceptance evidence** (`purpose=action_diagnostic`,
`gate_eligible=false`, `gate_passed=false`). It is not a fresh-seed, multi-stage
Phase 4 pilot, does not complete Phase 4, and does not establish that the action
problem is solved. No production fit or model/loss amendment was performed.

## Results and denominators

Every context retains the same ordered 10,000 examples: 5,000 positives, 2,500
safe negatives and 2,500 disconnected negatives. The historical and teacher
contexts are readouts, not autonomous ACT events.

| Context | Positive correct / 5,000 | Safe correct / 2,500 | Disconnected correct / 2,500 | Total correct / 10,000 | Timed successes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical immediate five-way | 1,641 | 2,500 | 2,488 | 6,629 | Not applicable |
| Teacher-context five-way | 5,000 | 2,500 | 2,500 | 10,000 | Not applicable |
| Autonomous legal four-shield | 5,000 | 2,500 | 2,498 | 9,998 | 9,998 / 10,000 |
| Raw five-way at the same autonomous ACT states | 5,000 | 0 | 0 | 5,000 | Not applicable |

For autonomous legal decoding, a negative is correct only with no action and no
error. There were 5,002 committed ACT events and 4,998 episodes without ACT.
No positive lacked an ACT. All 5,000 positive actions had the correct class and
fell in the unchanged `[t0 + 0.75*d, t0 + 0.90*d)` window; their measured delay
fractions ranged from 0.7539281189 to 0.8740574166 (mean 0.8095034692).

The raw-at-ACT row retains a 10,000-episode denominator but only **5,002 available
readouts**: 5,000 correct and two incorrect. The other 4,998 rows have null
logits/choices, not manufactured correct abstentions. Thus 5,000/10,000 is not
directly comparable to timed success; conditional readout accuracy is
5,000/5,002. Raw and legal argmax agreed at every actual ACT, including both
failures. Masking away abstention therefore does not explain this run's results.

Both failures were disconnected negatives, each taking shield 0 with no runtime
exception: public IDs `3d9587bc-2947-4c39-8492-f9bd90e9f8f9` (index 2806) and
`422b866b-396c-4335-bb74-a951fd59aec6` (index 9149). Their raw five-way readouts
also selected 0, with true-abstention margins -15.9818935394 and -16.2020535469.
Both complete trajectories are retained. The negative false-action rate is
2/5,000 (0.04%); runtime/dynamics errors and uncommitted action attempts were zero.

The report retains aggregate and per-variant confusion matrices. Rows are true
classes 0–4; columns are predicted classes 0–4 followed by missing readout.
Historical immediate confusion is:

```text
619    9   81   25   499  0
 29  289  110    0   808  0
  0    0  206    0  1031  0
  0    0    0  527   767  0
  5    0    3    4  4988  0
```

Teacher confusion is diagonal `(1233, 1236, 1237, 1294, 5000)` with no missing
readouts. Both autonomous confusion matrices have the first four diagonal
counts `(1233, 1236, 1237, 1294)` and final row `(2, 0, 0, 0, 0, 4998)`.
The legal summary separately credits successful negative no-action behavior;
the raw summary does not.

## Context, loss and gradient evidence

The immediate recipe performs learned atomic content transitions without
advancing world time; it has no installed runtime segment or actual ACT. The
teacher diagnostic uses the complete timed unroll, selecting positive flowed
ACT contexts and negative post-composition contexts, never padded predictions.
Its teacher-populated `context_hypothesis_class` is separate from the actual
composition head's `predicted_hypothesis_class`; inapplicable predictions are null.
The autonomous branch uses only public callbacks and learned guard/event
execution, not teacher timing, records, labels or scoring windows.

| Positive readout context | Mean five-way cross-entropy | Mean true-class margin |
| --- | ---: | ---: |
| Historical immediate | 4.7941925446 | -4.0878747364 |
| Teacher ACT | 4.0316332841e-7 | 15.8847272346 |
| Actual autonomous ACT | 3.9900673428e-7 | 15.8900564972 |

Of the 5,000 historical positives, 3,105 selected abstention and 254 selected a
wrong shield. The historical terminal hypothesis was correct for 4,980 positives;
3,339 still had an incorrect action despite that correct hypothesis. This is
evidence of a context-sensitive readout, not merely incorrect upstream hazard
classification. The full timed objective supervises action, whereas the content
auxiliary does not supply an action term. Teacher and autonomous execution also
change timing and state trajectories, so these scores do not isolate one causal
mechanism or prove a transition defect.

Original float32 masked-loss reductions over all examples include:

| Reduction | Denominator | Mean |
| --- | ---: | ---: |
| Positive action | 5,000 | 4.0035232514e-7 |
| Negative abstention | 5,000 | 5.5742257445e-8 |
| Action time | 5,000 | 6.5146804936e-7 |
| Guard group | 50,000 | 0.0037829141 |
| Guard time | 40,000 | 0.0047281440 |
| Final guard dormancy | 10,000 | 1.9945542488e-6 |
| Hazard | 5,000 | 2.4282945291e-5 |
| Retrieval | 17,500 | 2.7496323754e-5 |

All 22 reduction numerators/denominators are retained in `report.json`.
`guard_race` has no eligible comparisons: numerator 0, denominator 0, mean null,
not an observed zero loss. Scalar readout CE is recomputed from stored logits;
near-zero differences from float32 masked training reductions are not different
examples or substituted objectives.

The preselected first 128 examples (64 positive, 32 safe, 32 disconnected) give
complete-timed-unroll action loss 2.1513545789e-7, action-head gradient L2
5.3731957985e-6 and remaining recurrent-graph gradient L2 3.7897942923e-6.
All gradients are finite; 47 of 71 parameter gradient norms are nonzero.
This establishes an existing differentiable action path, not adequate future
training or a reason to change the loss. No optimization step was taken.

There were no positive dormant/missing-ACT failures. All 4,998 no-ACT negatives
had no terminal hypothesis and no finite ACT crossing at the last installed
boundary. The two unwanted negative ACTs remain failures. The evidence supports
context sensitivity and does not show a blanket inability to classify at ACT,
a masking-only repair, or a measured transition implementation mismatch. Causal
attribution remains an unresolved mixture; no speculative correction is inferred.

## Provenance, retention and verification

Original weight producer: `844a89cd04405139ca670e1b269ce78f2ae9a168`.
Actual diagnostic executor: `cccbb29f5dbc44ff3c2f666ddd8f241fed60a3ef`, clean at
execution. Subsequent documentation commits do not replace this producing source.

| Identity | SHA-256 |
| --- | --- |
| Original weight file | `ba6f929f1e5bfa9cf00b8c31b948f35582dc6662a23203db02c9e5f17ee93e3c` |
| Original corpus manifest | `2e1d367c6ac072d682afd241f280b5ff729491ca523c2106a4bb176be308d6d2` |
| Original logical model state | `c3661889be31ced19edeb1b89b90c8a1f0f0a5479077acb95bc5e485754b97aa` |
| Original canonical configuration | `cc8f910cff1438fe7ea581b5d58ad964a3823559916fa4a7a6d625208018f85e` |
| Diagnostic canonical configuration | `62b892f07b07b8b943121a7084d2492e33bb4e295e5dc311f50f9bed23b57729` |
| Diagnostic source-file map | `1556fbf9342ec7e02ce69ac2564a252d3b1e437bc3440ae08cd4dc29c54b3ab1` |
| Ordered projected bindings | `1bdf25014e523822997329436fb2e034c20ccb7f0551808ef9880d624d908b28` |
| `report.json` | `9dd53a0e6b009c9869355b0c651c7f62fa04e047f79179517c4b8c1c1d9f4db4` |
| `readouts.jsonl` (40,000 rows) | `0e73f29f3ae2ba5689dd16e8cdf7cef3ddf5c4d3e6d421521f6e268fda892062` |
| `autonomous/rows.jsonl` (10,000 rows) | `98db276e0eb8cc412347407032709d1fc72b570f812e510f9d94e3b66aefe0f9` |
| `autonomous/DONE` | `194120a9effcdb8b77824fb52c6b8e5d44c7a9fbe53d59fb7dd7d92d79359741` |
| Top-level `DONE` | `1c9de5c7202ee4f87f52008d9935d2268d15e5ec62530d3d9af03809e51689ac` |

The original manifest remains `phase3-component-manifest-v1`, not a forged pilot
manifest. Generator `ofd-v1`, curriculum `ofd-one-hop-v1`, runtime projection
`phase4-projection-v1`, model seed 11 and all ordered example hashes are bound in
provenance. Original validation root/public-ID seeds remain 313/337; the
execution profile is `phase4_pilot`, but this does not make the historical
corpus pilot-eligible.
Actual original files were authenticated, not retrained or converted.

The fixed prefix produced 128 immediate details, 128 teacher trajectories and
64 ACT-context details. Autonomous retention contains 130 full trajectories:
the fixed 128 plus both later failures. All 66 retained ACT states match flow
from their authentic pre-event anchors and segment identities. Guard summaries
refer to the pre-ACT segment, not the observer's post-event installed segment.
Historical predictions matched exactly with the read-only observer on and off.
The extra 10,000 historical control queries, 10,000 teacher examples and
128 gradient examples are additional diagnostic work, not original inference.

Post-run checks verified every one of 324 top-level artifact hashes, all 20,136
autonomous artifact hashes, all 120 executing source files, ordered inventories,
confusions, paired raw/legal context values and failure retention. Independent
raw action/window rescoring also found zero persisted-score mismatches.
The complete local artifact tree contains 20,462 files and 659,832,162 logical
bytes (682 MiB allocated). Preflight found 140,928,544,768 free bytes against a
20,971,520,000-byte reserve. The process exited 0: 3,265.57 seconds wall,
3,230.47 user, 24.99 system; report workload time was 3,205.33112079103 seconds,
excluding initial original-file authentication. These are diagnostic timings,
not a speed benchmark.

Local verification before the final narrow corrections passed 3,225 tests,
doctor diagnostics and source/wheel builds. The final corrected code passed
49 native diagnostic/boundary/transition tests plus scoped lint/format/diff
checks. The earlier full gate is not relabeled as a full gate on the corrected
source. The interrupted `5c3f5a8` run remains separately preserved as
NON-ACCEPTANCE; none of its rows were reused in this completed run.

## Reproduction

Use the original executor checkout, the accepted existing weight archive and a
fresh output directory. Supply local locators externally; never commit personal
paths or overwrite either earlier run. No fitting command is involved:

```sh
OMP_NUM_THREADS=1 UV_OFFLINE=1 UV_CACHE_DIR="$task7_cache_dir" \
  uv run python scripts/diagnose_phase4_actions.py \
  --weights "$task7_original_weights" \
  --manifest manifests/validation/phase3/one-hop-10000-content-v2-authenticated.json \
  --expected-weights-sha256 ba6f929f1e5bfa9cf00b8c31b948f35582dc6662a23203db02c9e5f17ee93e3c \
  --expected-manifest-sha256 2e1d367c6ac072d682afd241f280b5ff729491ca523c2106a4bb176be308d6d2 \
  --profile phase4_pilot --device cpu --output "$task7_fresh_output"
```

Large raw artifacts stay in owner-local storage, identified by the portable
hashes above. A fresh run records its own timing-dependent report/artifact hashes;
do not assume byte equality or relabel its executing source as this run's source.
