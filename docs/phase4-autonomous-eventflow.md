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

## Local pilot commands and initial data setup

The pilot uses only seed 11, offline execution and zero foundation-model calls.
`make pilot` uses the closed production profile; `make pilot-smoke` uses the
16-example debug profile and stops after four optimizer updates. Set
`PILOT_RUN_DIR`, `PILOT_MANIFEST_DIR` and `PILOT_DEVICE` externally as needed;
their defaults are `runs/phase4-pilot-v1/event_flow/11/pilot`,
`manifests/validation/phase4` and `cpu`. The smoke target appends `-smoke` to the
run and manifest directories to keep the two profiles distinct.

Start from committed, compatible source. The first workflow invocation produces
the missing `one_hop.json`, `two_hop.json`, `primary.json`, `robustness.json` and
`audits/<stage>/report.json` inputs, then stops with a **commit-data precondition**.
The raw audit auxiliary manifest/index remain under `audit-raw/`; introduce only
the four canonical stage manifests and the four exact audit reports. Review and
commit those eight files, then create a distinct compatible training revision
(an explicit empty commit is sufficient). Run the same command again. No command
auto-commits Git state. Debug data preserve this introduction chain while their
small-corpus audits explicitly remain non-acceptance. Production statistical
audit acceptance is required before fitting; no training command reruns probes
to replace failed or missing evidence.

```sh
make pilot-smoke
# After reviewing and explicitly introducing the eight printed input paths:
git commit --allow-empty -m "Record compatible pilot training revision"
make pilot-smoke
```

Production command forms, once their stated inputs exist:

```sh
uv run silent-cascade data freeze --pilot-stage primary --config configs/train/pilot.yaml --output manifests/validation/phase4/primary.json
uv run silent-cascade train --config configs/train/pilot.yaml --manifest-dir manifests/validation/phase4 --run-dir runs/phase4-pilot-v1/event_flow/11/pilot --seed 11 --device cpu
uv run silent-cascade evaluate --checkpoint runs/phase4-pilot-v1/event_flow/11/pilot/selected.json --manifest manifests/validation/phase4/primary.json --output runs/phase4-pilot-v1/event_flow/11/pilot/eval/primary --device cpu
uv run silent-cascade report build --pilot --run-dir runs/phase4-pilot-v1/event_flow/11/pilot --output reports/phase4-pilot-v1
```

`selected.json` binds an eligible checkpoint by hash; debug or unsuccessful runs
have latest diagnostic weights and cannot publish accepted selection. `train
--resume PATH` accepts the run's exact latest durable training archive. The full
workflow verifies and reuses completed steps, recovers interrupted training from
its verified durable checkpoint, and refuses incompatible directories with a
new-run-path instruction. Ownership records bind PID, absolute process creation
time and run identity; no process is killed to clear ownership.

Neural replay uses `silent-cascade replay ARTIFACT --weights WEIGHTS --json`.
Scripted replay remains available without loading neural training modules.
Reports read retained artifacts, verify hashes and reconstruct integer
denominators; missing or corrupt evidence fails explicitly. Timelines and guard
plots use retained trajectories. Debug completion is reported separately from an
unmet production gate. Failed production learning produces an adverse report and
nonzero workflow status; the workflow never starts another seed or Phase 5.
These commands never access frozen tests. `make verify` remains the complete
local quality gate and never launches production fitting or pilot statistical
audits. Device/resource acceptance and the final Phase 4 evidence gate remain
separate later tasks.

### Final checks and bounded evidence publication

The shared package final-check driver is also available through
`uv run python scripts/check_phase4_pilot.py --config configs/train/pilot.yaml --run-dir RUN --output GATE`.
It never repeats fitting. It authenticates the original selected full archive,
executes the prescribed final CPU corpora/repeat/delay pairs and actual CPU
continuation/replays, then invokes the existing selected-archive numerical and
offline diagnostics. Missing selection and adverse final checks stay adverse;
debug execution remains non-acceptance. A learning-eligible result is not a
passing final gate, and the full production workflow returns nonzero if either
condition is unmet.

`uv run python scripts/verify_phase4_gate_artifact.py --artifact GATE --repo-root REPO --raw-run-dir RUN`
performs artifact-only verification. Without raw attachments it explicitly lists
missing coverage and proves recorded-evidence integrity, not a fresh neural
rerun. Collection, verification and reporting do not launch model workloads.

Recovery is a separate, explicit execution mode. In a checkout with the original
gate's authenticated executable/config/plan/spec bytes, set `PHASE4_RECOVERY_DEST`
to an absent fresh run path and `PHASE4_RETAINED_RUN` to a same-layout retained
input mirror, then execute the gate's `raw_regeneration_command`. For direct CLI
execution, also set `PHASE4_ORIGINAL_GATE` and `PHASE4_ORIGINAL_RUN` to the actual
gate and original raw run paths:

```sh
UV_OFFLINE=1 UV_FROZEN=1 OMP_NUM_THREADS=1 uv run python scripts/check_phase4_pilot.py \
  --recover-from "${PHASE4_ORIGINAL_GATE:?set the original gate path}" \
  --run-dir "${PHASE4_ORIGINAL_RUN:?set the original raw run path}" \
  --destination "${PHASE4_RECOVERY_DEST:?set an absent fresh destination}" \
  --retained-run-dir "${PHASE4_RETAINED_RUN:?set retained training input mirror}"
```

The CLI validates these paths; it never treats an existing output as a recovery
destination. The retained-mirror flag may be omitted when every training input
still exists in the original run. Every original training-result/index entry and
selected/latest archive/weight input must be restored byte-for-byte and hash
verified; missing inputs are reported by exact path and never trigger refitting.
Only final evaluations, continuation/replay and numerical/offline diagnostics are
rerun into the fresh destination through the shared driver. Original adverse data
remain untouched, and a recovery intent binds source/config/weights and input
inventories. Recovery does not imply a passing gate. Portable verification checks
each available raw evidence family independently and lists both missing paths and
`unavailable_semantic_checks`; unrelated missing files never excuse contradictions
in present archives, replay, numerical operations or evaluation rows.

New training results uniformly use `phase4-training-result-v2`, preserving the
seven non-inventory fields and replacing the flattened artifact map with a
`phase4-artifact-index-v1` descriptor. Its canonical JSONL path/hash stream is
globally ordered and hash-bound, with deterministic attempt-local gzip shards.
Each shard permits at most 50,000 rows and 128 MiB decoded bytes and must remain
below 100 MiB on disk. All entries are retained, including adverse/abandoned
evidence; no historical result or archive is rewritten. The exact legacy flat
result remains readable. Public `PilotTrainingResult.artifact_hashes` and
`load_training_result` still materialize dictionaries and therefore retain that
memory cost; only persisted serialization and index validation are streamed.
The gate copies identical index shards and binds the original result bytes
instead of embedding a second full training map. Final compact causal rows
are independently byte-bounded and streamed through every shard.

The required local quality run is recorded once with
`UV_OFFLINE=1 UV_FROZEN=1 OMP_NUM_THREADS=1 uv run python scripts/record_phase4_local_verify.py`.
This wrapper runs the existing `make verify` command and retains its actual
exit status, stdout/stderr and source inventories under
`artifacts/phase4-local-verification/VERIFICATION_COMMIT/`. Failed or incomplete
attempts are create-only evidence. A source-compatible passing receipt is
required for final acceptance; discovery uses Git ancestry, never filesystem
timestamps, and a newer failure cannot fall back to an older pass. Later
data-only commits may reuse unchanged executable/test/build/config/plan/spec
inputs. Gate publication and its subsequent independent verification are not
inputs to their own receipt. Phase4 remains incomplete pending actual Task12
acceptance and its final delivery checks; the historical storage-headroom and
native parameter-parity failures are not waived by these software interfaces.

## Existing fixed64 engineering diagnostic

Task 8's existing seed-11 debug run at source
`1853840629d26136feb0b5ee70a293beca50ae1c` first met its separate fixed64 criterion
after 525 updates: 64/64 complete chains, 59/64 timed successes, 0/32 negative
false actions and zero errors at that boundary. Positives had 27/32 timed
successes, all 32 classes correct and five early actions. Four earlier dynamics
failures remain in the retained history. These are training-exposed debug
results (`gate_eligible=False`), not held-out generalization or production
readiness. The prior wrapper recorded 560.85 seconds and exited 1 after the
application completed because macOS denied its clockrate query; no peak-RSS
claim follows. Task 9 does not rerun this diagnostic.

The preserved artifact root is
`runs/phase4-overfit-debug-v1/task8-1853840-seed11-20260917`.
Its raw `result.json` has SHA-256
`90bded2f293113c07943117d125a07cc89e5783ee026213fbedf5b07981deffb`;
final portable weights hash
`0c5d590ce816f7b29cda9f8c026fc100bb05f9711e8cf5e0374938d84f19812e`;
ordered debug data hash
`982a8660db4a318a93e77a91cd1d3abdcf4eb5cd84dfe576ae7ef953bc7ad9b9`;
canonical configuration hash
`f9b1fa70f4bbb075fef516b491125ff975cdb36301639741a44ad1f92956069b`.
The recipe uses roots 449/457, fixed indices 0–63, generator `ofd-v1` and
one-hop transform `ofd-one-hop-v1` inside `phase4-data-v1`.

## Task 10 numerical and workload diagnostics

`verify_pilot_numerics(config, checkpoint=..., output_dir=...)` checks the actual
combined objective on one-hop and maximum-four-hop debug batches. It enumerates
every timed/content objective-boundary tensor, loss reduction, named gradient,
updated state tensor and AdamW moment independently. CPU continuation is exact;
native MPS continuation uses the approved tolerances and exact RNG/next-batch
identity. Missing native MPS is reported as missing evidence.

The optional checkpoint must be a full `phase4-pilot-training-v1` archive.
Portable-only weights cannot establish optimizer/RNG continuation and are
rejected. Its original file hash and initial logical model hash remain distinct
from diagnostic updates. Runtime comparison uses untouched exported weights on
an explicitly addressed balanced subset of a real manifest. A supplied archive
is not proof of selection: Task 11 must bind its hash to the selected descriptor.
Artifact readers derive required capture/runtime inventories and recompute numeric
and throughput summaries from retained raw evidence; merely rehashing a changed
summary does not validate it. Original full archives, resume RNG observations and
final diagnostic weights remain separate from the untouched inference export.

Checkpoint-backed native continuation restores the validated archive's MPS RNG;
an archive without native RNG supplies missing native evidence even when MPS
hardware is available. CPU continuation binds Python, NumPy and Torch CPU RNG
exactly; opaque native bytes retained in the input are outside that CPU claim.
Native continuation additionally binds the archived MPS state and observed draws.
Readers bind starting model, optimizer tensors/groups, mode, complete progress
and scoped RNG to the original archive. An empty optimizer triggers a separately
retained real bootstrap update. Its reader checks configured AdamW/clipping
arithmetic on disposable tensors and the legal progress/RNG transition, without
running a forward, training batch or event engine. This establishes arithmetic
consistency of retained execution evidence, not independent regeneration of its
saved gradients. The extra bootstrap diagnostic forward is counted explicitly.

New retained trajectories declare their actual CPU/MPS origin in schema v2.
Portable CPU reading permits the existing forward tolerance only when recomputing
positive-duration terminal flow from an MPS origin. Original float32 values,
causal hashes, guards, decisions and timestamps remain unchanged. Live state and
checkpoint validation stay exact; legacy v1 trajectories retain exact CPU checks.

`profile_pilot(config, output_dir=...)` measures native, synchronized full updates
and the actual scalar engine, retaining warmup/update rows and every evaluation
failure. The production recipe uses batch 128, 16 warmups and 32 measured updates
for each one-hop/four-hop device trial, plus 256 manifest-addressed autonomous
episodes per device. The smoke recipe is explicitly diagnostic and smaller.
Reports include component-evaluation costs, event/error distributions, process
RSS high-water and separate tensor-memory estimates, measured retained bytes,
and a conditional compute/storage range. Final acceptance/replay stays on CPU.

### Actual production-dimensional diagnostic (2026-09-17)

The reviewed executor `64446327b537565f9fb7dccd9182d0938b655f62` completed one
`profile_pilot(resolve_pilot_config("phase4_pilot"), output_dir=...)` call in
4,450.753695 seconds wall (09:10:58.814154–10:25:09.567849 UTC), exit 0 and empty
stderr. This was fresh seed-11 diagnostic work, **not a production fit** or the
existing fixed64 model. No checkpoint was supplied, selection/data-introduction
verification remain false, and foundation-model calls were exactly zero.

All four batch-128 trials retained 16 warmup and 32 measured updates: 192 actual
updates, including 64 warmups. One-hop and primary traces had maximum lengths
5 and 11 respectively, with primary exercising four hops. Numerical diagnostics
added 10 actual updates and 10 separately counted diagnostic forwards; none of
these weights count toward pilot training. The 2,781,042-parameter starting model
and untouched portable runtime export were identical across devices.

The complete numerical inventories contain 79 forward tensors, 147 loss tensors,
71 gradients, 71 updated parameter tensors and 213 optimizer tensors per recipe.
One-hop passed every group. Primary forward/loss/gradient/optimizer groups passed,
but **one updated parameter element failed** the fixed `rtol=1e-3, atol=1e-5`:
`controller.network.0.weight[167,449]` was 0.00021155402646400034 on CPU and
0.00022310856729745865 on MPS. Absolute error 0.000011554540833458304 exceeded
the allowed 0.00001022310856729746. The failure is retained, with no rerun or
tolerance amendment. Exact CPU resume and native MPS resume passed all groups
and scoped RNG/next-batch checks. No device evidence was missing.

The 64 paired numerical episodes had zero decision, score, timestamp or error
mismatches. Both devices took no ACT: 32 positives failed with `no_action` and
32 negatives succeeded. Throughput used another 256 executions per device
(128 positive, 64 safe negative, 64 disconnected negative), again no ACT,
128 `no_action` failures, 128 successes and zero errors. Each device recorded
256 activations, 3,077 facts, 67 recalls, 18 compositions and 256 terminal events;
episodes had 7–24 events (mean 14.3515625). These are short, untrained cascades,
not evidence of competent validation or learned multi-hop performance.

Both subsets are explicit prefixes (0–63 and 0–255) of the same authenticated
10,000-entry primary manifest, not smaller replacement acceptance corpora.
There were 640 scalar episode executions in total, not 640 unique examples;
the prefixes overlap. Component timing added 256 actual examples per device.

| Observed operation | CPU, one thread | Native MPS |
| --- | ---: | ---: |
| One-hop full update, mean seconds | 0.799317 | 3.146011 |
| Primary full update, mean seconds | 1.854461 | 7.845695 |
| 256 scalar episodes, seconds | 82.567770 | 2,815.954506 |
| Scalar episodes/second | 3.100483 | 0.090911 |
| 256 component examples, seconds | 0.238568 | 2.637973 |

The recorded choice is **CPU: native MPS numerical checks failed**. CPU was also
faster in these observed operations. This is not an uncontended benchmark:
process snapshots showed a VM around 101–300% CPU, desktop activity and later
external Python/Rust work around 99% CPU. No unrelated process was stopped.
Native MPS fallback was disabled; Python 3.12.1, Torch 2.13.0 and one Torch/OMP
thread were used offline. Final acceptance and exact replay remain on CPU.

Process-lifetime RSS high-water was 2,866,954,240 bytes at CPU trials and
2,943,336,448 at MPS trials, not isolated per-trial peaks. Reported tensor working
estimates were 3,145,728 bytes for training and 24,576 for runtime; these are not
total memory. MPS current driver allocation was 3,239,788,544/3,264,626,688 bytes
after one-hop/primary training, not a peak or energy measurement.

The conditional compute projection is 58,042.55–2,874,015.20 seconds
(16.12 hours–33.26 days), **not a convergence promise or competent-validation
ETA**. It includes the optimistic 4,000-update path and the 75,000-update ceiling,
up to 75 boundaries/1.5 million autonomous/750,000 component validations, and
40,640 final episode executions (40,000 repeat/two-hop/robustness, 512 delay-pair,
128 device-smoke). Replay is a coverage-derived estimate of 12–216 episode
equivalents, approximately 3.87–378.73 seconds under the same assumptions;
Task 11 must collect actual coverage. Four historical 6,376.053914-second audits
are context, not newly measured four-stage audits. The upper scenario scales
runtime/storage by 5.4363 for 64 internal events plus observed ordinary events;
future cascades, failure modes and host load can exceed this scenario. Direct
10,000-row scaling of the observed CPU sample alone is 3,225.30 seconds.

The final retained tree has 2,197 files and 955,360,236 logical bytes; the forecast
captured 955,046,884 bytes before its own report/DONE. All failures remain.
Projected retained storage is 30,322,463,395–3,675,694,793,937 bytes, including
journals, component rows, manifests and explicit estimated audit allowances.
The required `ceil(1.2 * upper_retention) + checkpoint_working_space` is
**4,411,034,493,825 bytes**, including 200,741,100 checkpoint-working bytes,
against only **16,358,002,688 observed free bytes**. Storage headroom failed.
No production run was started; nothing was deleted, relocated or pruned to fit.
Remeasure at the actual stable retained destination before production.

Raw evidence remains under the ignored stable local path
`runs/phase4-task10-measurements/6444632-seed11-20260917/profile`, not system temp.
Both strict readers authenticated required inventories and recomputed summaries
from saved tensors/runtime rows without retraining (430 numeric artifacts and
2,195 throughput artifacts, the latter including the numeric subtree).

| Measured identity | SHA-256 |
| --- | --- |
| Numeric report | `955b011e178f1222d5fcf4709e4c2e07b6fb44ed8f18481a8dc23fc8c63c12df` |
| Throughput report | `4a9cd9885d68afcfaa4cb6dd9ee15ad783f4a6a3cd5d6339f5115de473793419` |
| Source-file closure (129 files) | `19c3f9b1ed4b16f19daf4dcf244268ec9179ffb6331486a8804d0aca5a3cd72c` |
| Canonical configuration | `62b892f07b07b8b943121a7084d2492e33bb4e295e5dc311f50f9bed23b57729` |
| Primary manifest | `76558bbf83fac06c25d590fbd00aab9d2384fe8a61011473bdea1519df9e77a7` |
| Initial/runtime logical model | `5053ec7272f59e0d65c16e14c007c8ddf9c31c40a713269589efcb78f904502d` |
| Untouched portable weights | `22ccef645265b2e00c757051be393e2b41605c39dfb82ebcacbd74367a5483af` |

The earlier native full `make verify` passed 3,365 tests, doctor and package builds
on `1be49c37`; the reviewed checkpoint-evidence correction then passed 288 native
covering tests with no skips, repository lint/format and package builds on the
measured source. This is explicitly pre-fix full versus post-fix covering evidence,
not a relabeled full suite on the later commit. Subsequent documentation does not
replace the measured source identity. Selected-checkpoint checks, the production
headroom gate and final Phase 4 acceptance remain unestablished. No throughput or
debug report satisfies a training, accuracy or final-weight gate.

## Task 11 recorded local software verification (2026-09-17)

After the scoped evidence/recovery and canonical-filename reviews, the unchanged
source `f9fa6ed955797971f75c53a1a0370fb9b3e52cae` passed the actual local gate:

```sh
UV_OFFLINE=1 UV_FROZEN=1 OMP_NUM_THREADS=1 uv run python scripts/record_phase4_local_verify.py
```

The recorder ran `make verify` once, with real native MPS available and fallback
disabled. Lint passed and all 272 Python files were already formatted. The three
isolated pytest processes passed 3,169 main tests (7,452.20 seconds), 93 service
tests (17.20 seconds), and 216 leakage tests (247.93 seconds): **3,478 passed,
zero skips and zero reported warnings**. Doctor passed CPU, MPS availability,
fallback, RNG, numeric-smoke and writable-path checks; both the source distribution
and wheel built successfully. Timings reflect a shared host, not an idle benchmark.

The create-only [receipt](../artifacts/phase4-local-verification/f9fa6ed955797971f75c53a1a0370fb9b3e52cae/receipt.json),
intent, raw stdout/stderr and process result retain the exact execution evidence.
Receipt SHA-256 is
`df3771b1b6596245669a9f03d57a360faa361e98c9bbf04b9bb7780d05418b12`.
All 292 tracked execution/test/build/config/approved-plan/spec inputs matched
before and after execution and the committed source; artifact-only discovery
authenticated the receipt and bound log hashes. This later documentation/evidence
record does not relabel the producing source or require a duplicate test run.

The real tiny Task 11 fixture retained both one-hop/primary native tensor
inventories and 16 MPS runtime trajectories. The explicit omitted-MPS-tensor
regression ran, rather than skipping. Its gate remained `debug_non_acceptance`;
the additional fresh-destination recovery fixture was intentionally CPU-only.
These are software/evidence-integration checks, not selected-weight production
acceptance, convergence or throughput evidence. No production fit, new full
profile, production audit or Task 12 execution was started. **Phase 4 remains
incomplete and production remains storage-blocked** under the unchanged Task 10
headroom result above; its historical native parameter-parity failure is retained.

## R2 provisioning update (2026-09-17)

The owner authorized a private remote archive as a response to the local storage
limit. The dedicated `silent-cascade-artifacts` R2 bucket and bucket-only local
credentials are now provisioned. A tiny upload/download passed byte and SHA-256
comparison; its test object was removed, and access to an unrelated bucket was
denied. See [R2 archive setup](r2-archive-setup.md) for the verified scope and local
credential handling.

No experiment evidence was uploaded or deleted. The current trainer and evidence
readers have not acquired remote-storage support, and the historical preflight
has not been relaxed. The owner approved the
[bounded R2 implementation plan](superpowers/plans/2026-09-17-phase-4-r2-archive.md)
on 2026-09-17 with a 10 GiB total local ceiling, including protected headroom.
As of 2026-09-18, the bounded bundle/catalog and transport tasks are reviewed
through `de88d0b`; supervisor, scientific integration and real-provider
qualification work remain outstanding. **Task 12 remains unstarted** until
the reviewed storage integration demonstrates bounded local space, verified
archival and recovery while preserving offline scientific execution and every
retention gate.
