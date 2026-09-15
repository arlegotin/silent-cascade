# Local neural components

Phase 3 provides a teacher-forced trainer and an unassisted **untimed** component
evaluator. The evaluator checks learned retrieval, composition and stopping with
at most two RECALL/COMPOSE pairs. It does not demonstrate autonomous world-time
scheduling or timed OFD success. Root `silent-cascade train` belongs to Phase 4;
the separate module below exposes two complete commands.

Before source freeze, a documented optimizer correction sets AdamW epsilon to
`1e-7` globally in both profiles and on both devices. At the original `1e-8`,
10 first-step weights failed native parity through amplification of nearly
cancelling gradients. The original fixture now passes unchanged tolerances.
This is a real optimizer change that can damp small-gradient learning; see the
[versioned rationale](deviations.md#2026-09-15--pre-freeze-phase-3-adamw-stabilization).
Old configuration archives are rejected and learning evidence must be regenerated.
An intermediate `1e-6` trial passed numerical parity but reached only 63/64
chains at the unchanged 1,000-step learning limit; that failed attempt is
retained in the development record. The final `1e-7` candidate passes native
parity and reaches 64/64 content chains at step 900 of the unchanged learning
fixture. Its secondary untimed action accuracy is only 32/64; this does not
establish timed competence.
No production acceptance result is claimed.

## Training

Supply every configuration layer explicitly and in order. Paths can be absolute;
installed wheels contain the Python modules but do not discover configuration
files from the checkout or the current directory.

```sh
uv run python -m silent_cascade.train fit \
  --config /path/to/checkout/configs/base.yaml \
  --config /path/to/checkout/configs/data/primary.yaml \
  --config /path/to/checkout/configs/model/event_flow.yaml \
  --config /path/to/checkout/configs/model/neural_components.yaml \
  --config /path/to/checkout/configs/train/smoke.yaml \
  --validation-manifest /path/to/debug-validation-manifest.json \
  --run-dir /path/to/new-run \
  --expected-source-commit SOURCE_SHA \
  --device cpu
```

`--config` repeats and preserves order. `--set key=value` uses the existing strict
configuration override parser; unknown fields, incompatible profiles and changed
manifest bindings fail. The smoke profile runs four updates, validates every two
updates and requires the matching 16-row debug manifest. The production
`configs/train/one_hop.yaml` profile requires its source/configuration-bound
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

`train.verification.measure_device_parity(model, batch)` executes identical
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
