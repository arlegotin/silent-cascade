# Local neural components

## Final review hardening and historical evidence — 2026-09-15

Whole-phase review found that production fit/resume and standalone evaluation
checked matching source labels without authenticating the code executing those
commands. The shared production boundary now checks the actual loaded checkout,
clean source closure, latest effective approved plan and exact committed recipe
before corpus generation, model loading, RNG restoration or output creation.
Archive configurations reconstruct that recipe when their portable metadata has
no overlay paths. Durable checkpoints and final evaluation publication recheck
execution identity. Debug smoke profiles remain portable outside Git and are
ineligible for production acceptance. Fit success, evaluation success and raw
evaluation envelopes are version 2 and explicitly label `publication` as `debug`
or `production`; checkpoint archives and numeric evidence formats are unchanged.
Run and attempt creation now uses pinned
no-follow directory descriptors; rejected symlink paths leave their targets unchanged.

The independently authenticated historical source
`061fd4079f30956b3f51c706bbb374eb5ae0c2db` remains supported: its corpus
`one-hop-10000-content-v2.json` has SHA-256
`2d36a77d8ea1a0b361553a321bd4af9405539e310c1bc432d2036ab4c17eb412`,
and its gate `component-gate-content-v2.json` has SHA-256
`4090494bd7b3fc072a4654a3ed5c9eaf6a5d65afe7cf9402394f791f8191a0e0`.
Both artifacts and the earlier rejected run are immutable historical evidence.
The interface fixes change their literal execution closure, so those artifacts
cannot supply current-source acceptance or be relabeled as newly executed.

The delivery map now designates
`manifests/validation/phase3/one-hop-10000-content-v2-authenticated.json` and
`manifests/validation/phase3/component-gate-content-v2-authenticated.json`.
Phase 3 is In progress pending the scoped rereview and fresh source-bound run.
That run must use the unchanged `teacher_timed_plus_content_v2` recipe, coefficient
1.0, epsilon 1e-6, seed 11 and the identical 10,000 ordered examples. All original
numeric tolerances, acceptance gates and scientific claim boundaries remain fixed.
The designated final-source production corpus and acceptance fit remain pending;
the final-source native/debug checks are prerequisites only. An isolated mutation
diagnostic accidentally crossed its checkpoint sentinel and completed 16 updates
before termination. Its full files were preserved, the batch sentinel was corrected,
and none of that diagnostic is acceptance evidence or a resumed production run.

Phase 3 provides a teacher-forced trainer and an unassisted **untimed** component
evaluator. The evaluator checks learned retrieval, composition and stopping with
at most two RECALL/COMPOSE pairs. It does not demonstrate autonomous world-time
scheduling or timed OFD success. Root `silent-cascade train` belongs to Phase 4;
the separate module below exposes two complete commands.

Before the first source freeze, a documented optimizer correction set AdamW epsilon to
`1e-7` in the original profiles and on both devices. At the original `1e-8`,
10 first-step weights failed native parity through amplification of nearly
cancelling gradients. The original fixture now passes unchanged tolerances.
This is a real optimizer change that can damp small-gradient learning; see the
[versioned rationale](deviations.md#2026-09-15--pre-freeze-phase-3-adamw-stabilization).
The unsupported pre-stabilization `1e-8` configuration remains rejected.
An intermediate `1e-6` trial passed numerical parity but reached only 63/64
chains at the unchanged 1,000-step learning limit; that failed attempt is
retained in the development record. The original B8 `1e-7` candidate passed native
parity and reaches 64/64 content chains at step 900 of the unchanged learning
fixture. Its secondary untimed action accuracy is only 32/64; this does not
establish timed competence.
The accepted corrected production component result is recorded below. Its
claim boundary remains untimed component engineering evidence.

## Rejected one-hop production attempt

The first production attempt is preserved, without relabeling it as acceptance,
under [its versioned attempt index](../manifests/validation/phase3/attempts/one-hop-v1/attempt.json).
It ran the declared CPU configuration unchanged: AdamW learning rate `3e-4`,
weight decay `1e-4`, betas `0.9/0.999`, epsilon `1e-7`, batch 128, clip norm
1.0, model seed 11, train seeds 311/331, validation seeds 313/337, validation
every 1,000 updates, a 75,000-update ceiling, and patience 15. It stopped
naturally at update 25,000 with `early_stopping`; the maximum recorded gradient
norm was 68.28138732910156. All 25 scheduled validations were non-passing:

| Update | Recall | Composition | Complete chain |
| ---: | ---: | ---: | ---: |
| 1,000 | 17,411/17,500 | 14,937/17,500 | 7,462/10,000 |
| 2,000 | 17,432/17,500 | 15,171/17,500 | 7,694/10,000 |
| 3,000 | 17,439/17,500 | 15,475/17,500 | 7,997/10,000 |
| 4,000 | 17,451/17,500 | 15,386/17,500 | 7,905/10,000 |
| 5,000 | 17,495/17,500 | 15,806/17,500 | 8,308/10,000 |
| 6,000 | 17,137/17,500 | 15,002/17,500 | 7,856/10,000 |
| 7,000 | 17,492/17,500 | 15,613/17,500 | 8,115/10,000 |
| 8,000 | 17,491/17,500 | 15,384/17,500 | 7,887/10,000 |
| 9,000 | 17,493/17,500 | 15,233/17,500 | 7,734/10,000 |
| 10,000 | 17,364/17,500 | 15,788/17,500 | 8,423/10,000 |
| 11,000 | 17,445/17,500 | 15,760/17,500 | 8,305/10,000 |
| 12,000 | 17,357/17,500 | 15,574/17,500 | 8,212/10,000 |
| 13,000 | 17,445/17,500 | 15,636/17,500 | 8,189/10,000 |
| 14,000 | 16,094/17,500 | 13,165/17,500 | 7,070/10,000 |
| 15,000 | 15,423/17,500 | 11,943/17,500 | 6,520/10,000 |
| 16,000 | 15,815/17,500 | 12,583/17,500 | 6,767/10,000 |
| 17,000 | 15,850/17,500 | 12,480/17,500 | 6,631/10,000 |
| 18,000 | 15,825/17,500 | 12,160/17,500 | 6,340/10,000 |
| 19,000 | 14,761/17,500 | 10,356/17,500 | 5,595/10,000 |
| 20,000 | 17,033/17,500 | 14,074/17,500 | 7,041/10,000 |
| 21,000 | 16,508/17,500 | 13,147/17,500 | 6,647/10,000 |
| 22,000 | 17,179/17,500 | 14,242/17,500 | 7,063/10,000 |
| 23,000 | 17,285/17,500 | 14,052/17,500 | 6,768/10,000 |
| 24,000 | 17,099/17,500 | 13,768/17,500 | 6,682/10,000 |
| 25,000 | 17,258/17,500 | 13,720/17,500 | 6,489/10,000 |

The deterministic rule selected update 10,000. Its category-level complete
chains were positive 4,908/5,000, safe 2,453/2,500, and disconnected
1,062/2,500; secondary untimed action accuracy was 4,775/10,000. Two complete
CPU evaluations of the exported weights produced identical ordered predictions,
raw decisions, and metrics across all 10,000 rows, so repetition confirmed the
failure rather than repairing it. Fit, both evaluations, and collection exited
0; these process outcomes do not imply acceptance.

The independent artifact verifier exited 1 because native production-batch
CPU/MPS parity had three updated-weight mismatches in 2,781,042 compared values,
while forward values, losses, raw gradients, CPU exact resume, and MPS resume
passed their unchanged tolerances. Bounded diagnosis reproduced all three
coordinates. Opposing loss contributions near `1e-3` cancel to raw gradients
near `1e-7`; CPU/MPS differences in the guard loss group dominate the residual,
and the observed AdamW moments predict the updates within
`1.5664267209725136e-9`. This is numerical near-cancellation, not evidence of an
incorrect AdamW formula. The specific guard subterm or backend operation remains
unidentified, and the numeric gate remains failed.

Read-only outcome classification found both unwanted extra terminal retrievals
and later premature null stops. In a bounded 18-case discriminator, swapping
only explicit clock features changed zero continuation decisions; all nine
preselected failures remained wrong with the untimed latent state and correct
with the teacher latent state. This supports a latent-context exposure
hypothesis for those cases, but it does not identify a unique latent channel,
prove a model-code defect, or establish a repair. The archived artifact remains
`passed=false`; at that rejection, Phase 3 remained in progress.

## Accepted corrected content-v2 component gate

The corrected production run used the reviewed executable source
`061fd4079f30956b3f51c706bbb374eb5ae0c2db`, approved plan revision
`048a612e02f520e1f4c0cb6574b379ea13fe72bf`, and byte-identical 10,000-example
one-hop corpus rebound to `teacher_timed_plus_content_v2`. The new manifest is
`manifests/validation/phase3/one-hop-10000-content-v2.json`, SHA-256
`2d36a77d8ea1a0b361553a321bd4af9405539e310c1bc432d2036ab4c17eb412`,
introduced at commit `35b83bb9d603c185cb341ec637ab430e5fe6cdfa`. It contains 5,000 positive,
2,500 safe-negative and 2,500 disconnected-negative examples, with no duplicate
validation keys or public IDs and no overlap with the corresponding 10,000-index
training probe. The configuration SHA-256 is
`cc8f910cff1438fe7ea581b5d58ad964a3823559916fa4a7a6d625208018f85e`.

One CPU run with one OpenMP thread trained naturally for 8,000 updates and
stopped at `component_gate`, selecting update 8,000 with patience counter zero.
It executed the combined objective on every update, reached maximum recorded
gradient norm 31.499881744384766, and made zero foundation-model calls. The
selected full training archive is
`5e00b39b86339bf266e8ae988fb91bd6a81ce367448bb5be09a9db88a1a473c7`;
its model-state SHA-256 is
`c3661889be31ced19edeb1b89b90c8a1f0f0a5479077acb95bc5e485754b97aa`.
The portable selected weights have SHA-256
`1237654c0549fecca7eb9dac6f5ce700c3533a508b91e09511dbcbffa268c5e9`.
Their descriptor's update zero is the portable-format convention; the model
state is the selected update-8,000 state. Training result, checkpoint-index and
journal SHA-256 values are respectively
`abe3bcf7351f0e44145715e84bf80733386bf01ed7ab9ff7b6bb8d1f94900934`,
`c7f7965ab08dc8949ccb0923946c04be80a54dffff6a17258458766ccb94eee4`
and `9ccbc9a3569ab5d127ecab2861aabdafe7b05e90166223852dd9196857feb46c`.

All scheduled validations are retained; only the natural terminal validation
passed all three strict greater-than-99-percent content gates:

| Update | Recall | Composition | Complete chain | Gate |
| ---: | ---: | ---: | ---: | :---: |
| 1,000 | 17,500/17,500 | 15,070/17,500 | 7,570/10,000 | fail |
| 2,000 | 17,114/17,500 | 14,825/17,500 | 7,711/10,000 | fail |
| 3,000 | 17,225/17,500 | 15,864/17,500 | 8,639/10,000 | fail |
| 4,000 | 17,412/17,500 | 16,911/17,500 | 9,499/10,000 | fail |
| 5,000 | 17,479/17,500 | 17,266/17,500 | 9,787/10,000 | fail |
| 6,000 | 17,474/17,500 | 17,361/17,500 | 9,887/10,000 | fail |
| 7,000 | 17,494/17,500 | 17,369/17,500 | 9,875/10,000 | fail |
| 8,000 | 17,475/17,500 | 17,395/17,500 | 9,920/10,000 | pass |

The final exact rates are 99.85714285714286% recall, 99.4% composition and
99.2% complete chains. The 80 remaining chain failures are retained. By variant,
complete chains were positive 4,980/5,000, safe 2,495/2,500, and disconnected
2,445/2,500. Their recall counts were 9,980/10,000, 4,995/5,000 and 2,500/2,500;
composition counts were 9,960/10,000, 4,990/5,000 and 2,445/2,500. There were
no invalid predictions or caps. Secondary untimed action accuracy was only
6,629/10,000 (66.29%): positive 1,641/5,000, safe 2,500/2,500, disconnected
2,488/2,500. Action accuracy is not one of the Phase 3 content gates and is not
evidence of timed action competence.

Two complete CPU evaluations of the exported weights each reproduced all
10,000 ordered predictions and all metrics exactly. Their artifact SHA-256
values are
`fc3170e54fc4b6c18bfa0c418a8d7754f4f5a2cdd003ce6a3c66c20ca9e29928`
and `46f2062dcf1f6695bc06fa9b1c60fff2846017667cb376a9f0279846d916d3a1`;
the common ordered-prediction SHA-256 is
`2777a18254d700a11f6b6d2119fe22ed9e94f9ddbcd44fc5da3a6a0e9645cbe5`.
After excluding measured compute latency, the complete artifacts match. The
configured gate repeats all 10,000 predictions with zero mismatches.

The production unassisted collector evaluated 2,781,042 parameters in 79
batches (78 of 128 examples and one of 16), recording 488,500,694,528 forward
MACs, 3,145,728 maximum tensor bytes and zero foundation-model calls. Its
8.680945412488654 measured evaluation seconds are diagnostic only: the Phase 3
gate is explicitly untimed and is not a speed benchmark.

Native production-batch CPU/MPS parity used float32, one thread and all 128
examples. Forward comparison covered 44,475,393 elements across 840 named,
shape-inventoried tensors; loss comparison covered 23,539 elements across 147;
gradients and updated weights each covered 2,781,042 elements across all 71
parameters. Every mismatch count was zero. Forward/loss tolerance was
`rtol=1e-4, atol=1e-5`; gradient/updated-weight tolerance was
`rtol=1e-3, atol=1e-5`. The largest absolute forward, loss, gradient and update
differences were 0.00030231475830078125, 0.00006103515625,
0.000041365623474121094 and 0.000009017996490001678. Discrete recall and action
mismatches were zero. The evidence exercised 205 active crossings, 1,718 dormant
crossings and one genuine empty-memory row.

Exact CPU resume compared 2,781,042 parameter, 5,562,155 optimizer and 23,539
loss elements across complete 71/213/147-name and shape inventories; all numeric,
RNG and next-batch mismatch counts were zero. Native MPS resume used the same
complete inventories and tolerances `rtol=1e-4, atol=1e-5`; all mismatch counts
were zero. Its largest nonzero optimizer difference was
5.820766091346741e-11 absolute and 0.00010658708164570453 relative. The matching
same-recipe offline smoke performed one real training update and predicted eight
public rows while denying optional imports and network access; observed blocked
imports, network attempts, forbidden modules and foundation-model calls were all
zero.

The committed production v3 gate is
`manifests/validation/phase3/component-gate-content-v2.json`, SHA-256
`4090494bd7b3fc072a4654a3ed5c9eaf6a5d65afe7cf9402394f791f8191a0e0`.
Its full tensor name, shape and element inventories, raw rows, bounded traces,
three replay samples, compute records and adverse outcomes are part of the
artifact. The independent verifier returned `valid=true`, `passed=true`,
`publication=production`, 10,000 episodes and zero foundation-model calls.

The complete run can be reproduced locally and offline in an isolated checkout
starting from the reviewed source, with absent manifest/gate output targets and
a new owner-only run directory. The freeze and collector publishers use
exclusive no-clobber creation: do not aim them at the committed canonical files
in a current checkout. To audit this historical delivery, use its compatible
source checkout and committed manifest/gate, authenticate their hashes, and run
only the independent verifier at the end of this block. The final-hardening
delivery has a separate designation and cannot reuse this gate as current-source
acceptance. These are the historically executed producer
command forms; set the two local storage variables without committing their
machine-specific values:

```sh
phase3_source_commit=061fd4079f30956b3f51c706bbb374eb5ae0c2db
phase3_plan_revision=048a612e02f520e1f4c0cb6574b379ea13fe72bf
phase3_run_dir='<new owner-only local run directory>'
phase3_weights_sha256=1237654c0549fecca7eb9dac6f5ce700c3533a508b91e09511dbcbffa268c5e9
phase3_weights_path="$phase3_run_dir/weights-$phase3_weights_sha256.safetensors"

OMP_NUM_THREADS=1 UV_OFFLINE=1 uv run --offline python scripts/check_phase3_components.py freeze-validation \
  --config configs/base.yaml --config configs/data/primary.yaml \
  --config configs/model/event_flow.yaml --config configs/model/neural_components.yaml \
  --config configs/train/one_hop_content_v2.yaml \
  --expected-source-commit "$phase3_source_commit" \
  --expected-plan-base-revision "$phase3_plan_revision" \
  --output manifests/validation/phase3/one-hop-10000-content-v2.json
OMP_NUM_THREADS=1 UV_OFFLINE=1 uv run --offline python -m silent_cascade.train fit \
  --config configs/base.yaml --config configs/data/primary.yaml \
  --config configs/model/event_flow.yaml --config configs/model/neural_components.yaml \
  --config configs/train/one_hop_content_v2.yaml \
  --validation-manifest manifests/validation/phase3/one-hop-10000-content-v2.json \
  --run-dir "$phase3_run_dir" --expected-source-commit "$phase3_source_commit" \
  --device cpu
OMP_NUM_THREADS=1 UV_OFFLINE=1 uv run --offline python -m silent_cascade.train evaluate-components \
  --weights "$phase3_weights_path" --expected-checkpoint-sha256 "$phase3_weights_sha256" \
  --manifest manifests/validation/phase3/one-hop-10000-content-v2.json \
  --output "$phase3_run_dir/cpu-evaluation-1.json"
OMP_NUM_THREADS=1 UV_OFFLINE=1 uv run --offline python -m silent_cascade.train evaluate-components \
  --weights "$phase3_weights_path" --expected-checkpoint-sha256 "$phase3_weights_sha256" \
  --manifest manifests/validation/phase3/one-hop-10000-content-v2.json \
  --output "$phase3_run_dir/cpu-evaluation-2.json"
OMP_NUM_THREADS=1 UV_OFFLINE=1 uv run --offline python scripts/check_phase3_components.py collect \
  --config configs/base.yaml --config configs/data/primary.yaml \
  --config configs/model/event_flow.yaml --config configs/model/neural_components.yaml \
  --config configs/train/one_hop_content_v2.yaml \
  --manifest manifests/validation/phase3/one-hop-10000-content-v2.json \
  --weights "$phase3_weights_path" --expected-checkpoint-sha256 "$phase3_weights_sha256" \
  --training-run "$phase3_run_dir" --expected-source-commit "$phase3_source_commit" \
  --expected-plan-base-revision "$phase3_plan_revision" \
  --output manifests/validation/phase3/component-gate-content-v2.json
OMP_NUM_THREADS=1 UV_OFFLINE=1 uv run --offline python scripts/verify_phase3_gate_artifact.py \
  --artifact manifests/validation/phase3/component-gate-content-v2.json \
  --manifest manifests/validation/phase3/one-hop-10000-content-v2.json \
  --weights "$phase3_weights_path" --expected-source-commit "$phase3_source_commit"
```

This acceptance establishes teacher-forced training plus unassisted, untimed
retrieval, composition and stopping on the source-bound one-hop component corpus.
It does not establish autonomous world-time scheduling, autonomous cascade
operation, broader held-out generalization, or a comparative benchmark. Those
claims remain outside Phase 3; Phase 4 still owns autonomous EventFlow work.

## Training

The corrected recipe is explicitly named `teacher_timed_plus_content_v2`.
Its only profiles are `phase3_one_hop_content_v2` and `phase3_smoke_content_v2`,
with their original workloads and epsilon `1e-6`. The original `one_hop.yaml`
and `smoke.yaml` bytes and their `teacher_timed_v1`, epsilon `1e-7` identity
remain supported for historical configuration and checkpoint decoding. A
checkpoint cannot resume across recipe/configuration identities.

The shared `train.objective.training_objective` always computes the complete
original timed objective. For v2 it additionally computes the reviewed fresh
public observation graph and teacher-forced untimed content path, then sums
both actual losses with fixed auxiliary coefficient `1.0`. The profile derives
the objective identity and coefficient; neither is an independent setting.
There is one backward, clip and AdamW update per original batch counter.
Step diagnostics retain both `timed/` and `content/` reduction namespaces,
branch losses, and branch forward compute. The outer compute snapshot already
includes both graphs and backward; branch snapshots must not be added to it.
The public evaluator and every scientific gate remain unchanged.

The corrected fixed64 engineering regression reached 64/64 full chains and
112/112 required recalls/compositions at update 175 within its unchanged
1,000-update cap. All three variants passed (32 positive, 16 safe, 16
disconnected); secondary untimed action accuracy remained 32/64. A preceding
recording-helper attempt stopped at update 25 due to a snapshot unpack error;
that partial record is retained separately and has no learning-gate conclusion.
Configured B8/B128 native parity and CPU/MPS resume prerequisites passed before
the corrected learning run. These engineering results establish no timed or
held-out production acceptance.

Supply every configuration layer explicitly and in order. Paths can be absolute;
installed wheels contain the Python modules but do not discover configuration
files from the checkout or the current directory.

```sh
uv run python -m silent_cascade.train fit \
  --config /path/to/checkout/configs/base.yaml \
  --config /path/to/checkout/configs/data/primary.yaml \
  --config /path/to/checkout/configs/model/event_flow.yaml \
  --config /path/to/checkout/configs/model/neural_components.yaml \
  --config /path/to/checkout/configs/train/smoke_content_v2.yaml \
  --validation-manifest /path/to/debug-validation-manifest.json \
  --run-dir /path/to/new-run \
  --expected-source-commit SOURCE_SHA \
  --device cpu
```

`--config` repeats and preserves order. `--set key=value` uses the existing strict
configuration override parser; unknown fields, incompatible profiles and changed
manifest bindings fail. The smoke profile runs four updates, validates every two
updates and requires the matching 16-row debug manifest. The production
`configs/train/one_hop_content_v2.yaml` profile requires its source/configuration-bound
10,000-row production manifest and uses the approved 75,000-step ceiling,
1,000-step validation interval and 15-validation patience. That workload is an
explicit phase acceptance run, never part of local smoke.

The manifest API is `train.curriculum_data.ComponentManifest.from_examples`;
its entries preserve exact configuration, source, split, seeds, ordered keys,
accepted generation attempts and example hashes. The CLI requires a previously
prepared manifest. Frozen-test and unsupported-stage inputs fail closed.

`--device cpu|mps` selects actual placement without changing the manifest's
canonical configuration hash. Omit it to follow configured device preference.
Explicit unavailable MPS fails. `PYTORCH_ENABLE_MPS_FALLBACK=1` fails even when
CPU is selected. Training is float32 and offline with zero foundation-model
calls. Optional foundation-model libraries are not required.

Resume the same command and run directory with `--resume` pointing to the latest
archive named in `checkpoint-index.json`. Checkpoints are durable at step zero,
scheduled validation and final stop. A crash may replay up to 1,000 production
updates or two smoke updates, including a crash during validation/publication.
The archive binds a hash-linked journal; resume verifies the logged files before
continuing. Interrupted attempt logs remain available. Existing nonempty run
directories require explicit resume.

The selected full training archive is preserved beside portable weights. Its
descriptor records the selected optimizer step. A portable weight descriptor's
step zero is a format convention and does not mean the model is untrained.

## Component evaluation

```sh
uv run python -m silent_cascade.train evaluate-components \
  --weights /path/to/run/weights-HASH.safetensors \
  --expected-checkpoint-sha256 CHECKPOINT_SHA256 \
  --manifest /path/to/bound-validation-manifest.json \
  --output /path/to/new-component-evaluation.json
```

Evaluation uses CPU and reads the complete canonical configuration from the safe,
hash-verified weight archive. Production authenticates the executing checkout and
committed recipe before model construction or corpus regeneration, and rechecks
before publication. Debug profiles remain portable outside Git and explicitly
ineligible for acceptance. All predictions are completed before labels are
scored. Missing/wrong predictions remain in the metric denominators. Output
parents must exist and must not contain symlinks; publication uses an open parent
directory descriptor and refuses an existing file. No pickle or arbitrary Python
state is loaded.

## JSON contracts

All command successes and errors are one JSON object on stdout. Exit 0 means the
command completed, not that component accuracy passed acceptance. Exit 1 denotes
an execution/configuration error and exit 2 a command usage error. `--help` remains
human-readable. The strict Pydantic schemas are in `train.cli`:

| Schema | Contents |
| --- | --- |
| `phase3-fit-result-v2` (`FitSuccess`) | `status=ok`, `publication=debug` or `production`, `foundation_model_calls=0`, `result`: progress, stop reason, latest/selected/weights descriptors, validation history, actual device, selection rule and checkpoint cadence |
| `phase3-evaluate-result-v2` (`EvaluationSuccess`) | `status=ok`, `publication=debug` or `production`, output path/SHA-256, checkpoint SHA-256, episode count, zero foundation-model calls |
| `phase3-component-evaluation-v2` (`EvaluationArtifact`, output file) | `publication=debug` or `production`, source/configuration/checkpoint/manifest hashes, seed and curriculum version, raw predictions, full metric numerators/denominators and per-row outcomes, batch compute records, `timed=false`, `device=cpu` |
| `phase3-cli-error-v1` (`ErrorResponse`) | `status=error`, `error` containing stable typed `code`, `message` and JSON `context` |

Typical error codes are `usage_error`, `configuration_error`, `training_error`
`provenance_error` and `neural_error`. Full structured schemas can be inspected with each class's
`model_json_schema()`; unknown fields and invalid primitive types are rejected.

## Local checks and numerical evidence

`make smoke` includes the small training CLI and import smoke. `make verify`
retains its three pytest processes, then runs the doctor and package build
locally. No hosted automation is used. The fixed-training-set tiny overfit test
is an engineering regression, not held-out acceptance evidence.

`train.verification.measure_device_parity(model, batch, training=config.training)` executes identical
CPU-initialized weights on CPU and native MPS through the complete teacher graph,
every named output/loss, backward and an AdamW update. It also tests actual
empty-memory preview/control and active/dormant crossings. Forward/loss tolerances
are `rtol=1e-4, atol=1e-5`; gradients/updated weights use `rtol=1e-3, atol=1e-5`.
Discrete record/action choices must match. Evidence records tested names,
mismatch counts, maximum absolute/relative errors, model/example hashes, device,
versions and threads. Relative error is measured against the CPU magnitude with
a float32-tiny denominator floor; near-zero values can have large relative error
while meeting the unchanged combined absolute/relative criterion. Nonfinite
differences count as failures; matching dormant positive infinities are allowed.

`measure_cpu_resume(resolved_config, source_commit)` performs a real update,
safe save/load, and the next update both continuously and after restoring RNG.
Parameters, AdamW state, losses, random draws and next-batch hashes must agree
exactly. Its evidence includes the actual archive hash. Both APIs expose a
derived `passed` property; collectors must use primitive results, not infer a
pass from a test log. A native MPS failure remains a failed gate; a sandbox skip
does not mean native MPS is absent.

New `phase3-component-gate-v3` evidence records the derived objective identity,
actual optimizer options, both numeric contexts, the complete contiguous batch
counter sequence, and full raw step diagnostics for counters zero and one.
Every logged step is checked against the fixed objective during collection;
the independent verifier reconstructs sample reduction arithmetic and branch
compute accounting. Version 3 parity/resume records carry actual tensor shapes;
the verifier derives exact names, ranks, dimensions and element counts from the
authenticated architecture and regenerated public/teacher trace structure,
including every AdamW slot and both resume reductions. Verification constructs
no neural model. Event-cost positions are checked at every padded boundary and
weighted by that boundary's legal-guard count to reconstruct their numerator.
The offline subprocess accepts the validated recipe and executes
the matching v1 or v2 smoke overlay, one real update and eight public rows under
network and optional-import denial.

The source-bound [delivery map](../manifests/validation/phase3/delivery.json)
designates only the corrected 10,000-row manifest and corrected gate paths.
Neither file is created by this integration. Missing delivery evidence requires
Phase 3 to remain in progress; present evidence must independently pass and
match exact Complete metadata. A legacy designated gate cannot bypass this
check. The failed v1 archive remains unchanged in its attempt directory.
To reproduce its rejection, use the verifier and source revision
`5a96b673f67a634d268f000a240e0de0f3dca034` named by that archive, with its original
`one-hop-10000.json` manifest and recorded weights. Earlier shape-less v2 debug
evidence likewise uses its named verifier/source; v3 does not reinterpret old
bytes as the expanded contract. Evidence schema v3 does not rename or alter the
fixed `teacher_timed_plus_content_v2` training recipe.
