# Local neural components

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
No production acceptance result is claimed.

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
`passed=false`; Phase 3 remains in progress.

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
hash-verified weight archive. It regenerates the exact bound corpus without
assuming a checkout directory. All predictions are completed before labels are
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
| `phase3-fit-result-v1` (`FitSuccess`) | `status=ok`, `foundation_model_calls=0`, `result`: progress, stop reason, latest/selected/weights descriptors, validation history, actual device, selection rule and checkpoint cadence |
| `phase3-evaluate-result-v1` (`EvaluationSuccess`) | `status=ok`, output path/SHA-256, checkpoint SHA-256, episode count, zero foundation-model calls |
| `phase3-component-evaluation-v1` (`EvaluationArtifact`, output file) | Source/configuration/checkpoint/manifest hashes, seed and curriculum version, raw predictions, full metric numerators/denominators and per-row outcomes, batch compute records, `timed=false`, `device=cpu` |
| `phase3-cli-error-v1` (`ErrorResponse`) | `status=error`, `error` containing stable typed `code`, `message` and JSON `context` |

Typical error codes are `usage_error`, `configuration_error`, `training_error`
and `neural_error`. Full structured schemas can be inspected with each class's
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

New `phase3-component-gate-v2` evidence records the derived objective identity,
actual optimizer options, both numeric contexts, the complete contiguous batch
counter sequence, and full raw step diagnostics for counters zero and one.
Every logged step is checked against the fixed objective during collection;
the independent verifier reconstructs sample reduction arithmetic and branch
compute accounting. Numeric comparison records also reconcile each tensor's
element count. The offline subprocess accepts the validated recipe and executes
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
`one-hop-10000.json` manifest and recorded weights; the v2 schema does not
reinterpret the old bytes as new evidence.
