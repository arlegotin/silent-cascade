# Phase 3 Neural Components Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the bounded, trainable EventFlow component stack and reproducible teacher-forced training path, then demonstrate greater-than-99% one-hop retrieval/composition on a fixed, label-isolated component-validation corpus.

**Architecture:** Add a batched differentiable neural workspace alongside the completed scalar event engine, sharing its dimensions, equations, public schemas, and numerical contract without changing its frozen source. Separate public model inputs from privileged training targets; use the same encoders, scorer, controller, jumps, and heads during training and unassisted component validation. Phase 4, not this plan, connects those components to autonomous world-time guard scheduling and the full OFD pilot.

**Tech Stack:** Existing locked Python 3.12, PyTorch float32 on CPU/Apple MPS, NumPy, Pydantic, safetensors, Typer, pytest/Hypothesis, and uv; local `make verify` only.

**Spec:** `docs/superpowers/specs/2026-08-30-silent-cascade-design.md`, approved version 1.0.2, especially Sections 5, 8, 9.4, 13, 17 Phase 3, and 18.

**Status:** User-approved; completion tracked in docs/PLAN.md. Execution approved on 2026-09-14 ("proceed, use subagents"). This document is not an implementation-completion claim.

**Repository inspected:** `main` at `b4d65a60ee412b87870ce5e17d08b75f60709b4c`, with a clean worktree before planning.

## Global Constraints

- The Python package is named `silent_cascade`; the CLI executable is `silent-cascade`.
- Work on the current branch and make meaningful commits. Use Superpowers TDD, task reviews, systematic debugging for failures, and verification before completion.
- CI/CD is disabled. Run linting, formatting, tests, diagnostics, and package builds locally; `make verify` is the complete local quality gate.
- Primary scientific execution is offline and records exactly zero foundation-model calls. No optional MLX/Qwen import, download, hidden text reasoning, network request, or telemetry.
- At most 5,000,000 total trainable parameters, **including** the shared entity table; report its subtotal separately. Target approximately 1–3 million, not artificially padded parameters.
- At most 64 primary memory records and 64 EventFlow internal events. Fixed-grid controls retain their separately declared 25,000-opportunity ceiling; they are not implemented here.
- State dimensions: `z_fast=256`, `z_slow=64`, `drives=8`, guards `3`, focus key `64`, hypothesis latent `64`. Model floating tensors are float32 on CPU or MPS; index tensors may be int64 and masks boolean.
- Absolute timestamps and archived world-time metadata are host float64/Python floats. Never create float64 MPS tensors. Eager execution only: no mixed precision, `torch.compile`, silent MPS fallback, or Transformer in EventFlow v1.
- Latent bounds `[-1,1]`; latent rates `[1e-5,20]`; guard targets `(0,2)`; guard rates `[1e-5,500]`; threshold `1`; active/inactive margins `1.10/0.90`.
- Preserve the completed engine's `1e-4` minimum gap, `1e-3` same-kind refractory, `1e-9` tie tolerance, fifth consecutive clamp failure, and attempted-65th-event failure.
- Agent-facing/model code receives only public-derived features, never a generator seed, episode coordinate, oracle path, label, terminal horizon, action window, or future external event. `models` and `memory` cannot import environment or training implementations.
- Relevant retrieval, relevance, focus, terminal interpretation, confidence, support append, halting, action class, and timing must be learned. Schema checks may enforce legality; they must not replace a wrong prediction with an answer from a record or oracle.
- No deliberate retrieval/composition before activation. Observation jumps may encode the currently delivered fact and update persistent state; they cannot use an all-memory retrieval preview to solve the graph before activation.
- Training is teacher-forced; evaluation choices are not. Never describe teacher-forced loss, scripted behavior, or the component gate as autonomous timed OFD success.
- Training defaults: AdamW, learning rate `3e-4`, weight decay `1e-4`, batch size `128`, gradient norm clip `1.0`, step ceiling `75000`, validation interval `1000`, fixed validation episodes `10000`, patience `15`, best checkpoints `3`, latest checkpoint `1`.
- Preserve final training seeds `[11,23,37,53,71]`; this phase runs only the predeclared engineering seed `11`, not the five-seed scientific protocol.
- Every reported component number comes from raw rows bound to source, canonical configuration, data/transform versions, validation-manifest hash, model seed, and checkpoint SHA-256. Publish no dummy values or unimplemented commands.
- No frozen-test manifests or OOD-depth/extreme-clock training. Baselines, interventions, pilot timed success, final statistics, scientific claims, and Qwen remain outside this phase.

---

## 1. Starting State and Compatibility Boundary

Read these existing implementations before their dependent task; do not infer APIs from the original design's pseudocode:

| Existing implementation | Actual contract | Phase 3 use |
|---|---|---|
| `config.resolve_config(model_type, yaml_paths, *, set_overrides=())` | Strict deterministic merge, canonical JSON bytes and SHA | New `Phase3Config` and new overlays |
| `eventflow.config.Phase2Config` | Frozen data/runtime configuration | Compose into Phase 3 config without modifying it |
| `eventflow.state.ContinuousState`, `SegmentParameters` | Single-row float32 validated tensors; constructors clone while retaining autograd | Row-level equivalence references and export targets |
| `eventflow.flow.state_at(runtime, target_time)` | Analytic snapshot from the segment anchor | Verify batched flow against actual scalar implementation |
| `eventflow.guards.crossing_offsets_tensor(a,A,rate)` | Shape `(3,)`, differentiable active entries; dormant entries `inf` | Reference for a new batched implementation, not a shape-compatible batch API |
| `eventflow.jumps.ComposeDecision` | Explicit predicted role/focus/class/deadline/support/continue choices | Define matching neural outputs; do not call its scripted latent injections in neural training |
| `memory.store.BoundedMemory` | Exactly 64 capacity; immutable record metadata; no embeddings; primary overflow raises | Keep metadata authoritative; add a separate tensor packing/encoding layer |
| `memory.store.legal_records` | Optional exact subject/kind filtering | **Do not** use subject filtering to solve learned retrieval |
| `env.generator.IndependentEpisodeRequest` | Rejects path length 1; primary suites hard-code paths 2–4 | Separate versioned one-hop curriculum transform, not a weakened primary generator |
| `env.oracle.solve_public_episode`, `build_oracle_trace` | Public graph solution and privileged timed teacher trace | Training-target construction only |
| `eventflow.checkpoint_rng` | Safe JSON/uint8 RNG encoding and transactional restoration | Reuse for training checkpoints, without changing runtime archive schemas |
| `eventflow.archive_io` | Bounded regular-file reads through validated parents | Reuse safe reads with a training-specific byte limit |
| `eventflow.checkpoint/replay` | Closed scripted-agent registry and exact CPU continuation | Remain unchanged; do not pretend they support learned weights |
| `Makefile` | Three pytest processes isolate Phase 1 resource tests | Preserve isolation; add small neural tests, not full training, to local verification |

### Preserve existing evidence, not merely its filenames

The Phase 2 verifier requires every path in `PHASE2_ENGINE_SOURCE_PATHS` to have
the same bytes at its source and current HEAD. That tuple includes the root CLI,
shared schemas/errors/config, all existing `eventflow` files, existing memory
files, `pyproject.toml`, and `uv.lock`. Editing any of these while retaining the
old complete row would invalidate accepted evidence.

This phase therefore adds new modules and overlays. Keep all paths in that
tuple, the canonical spec, both prior phase plans, and all existing validation
artifacts byte-identical. Do not extend the historical tuple to new modules or
make its verifier less strict. New `models`, `train`, and `eval` initializers
must not be imported by existing initializers. New memory modules are imported
directly, without modifying `memory/__init__.py`.

Preserved Phase 2 source: `33fb8ed105046cb4fdb7be61eac50ed391415b7c`.
Preserved artifact: `manifests/validation/v1/phase2-engine-gate.json`, SHA-256
`c05b2f78680b14e5b8fd81597489f34ec68a1dc2d1114e62daa95d7428694c42`.
Keep the complete Phase 1 and Phase 2 delivery rows exactly as inspected.

The temporary phase-scoped entry point is `uv run python -m silent_cascade.train`.
Do not add `silent-cascade train` or change the existing executable in this phase.
Phase 4 must explicitly plan the historical-provenance transition when it
integrates learned runtime/CLI support; this is a deferred adapter, not a stub.

## 2. Design Decisions Resolved Before Execution

1. **Batched neural core alongside the scalar engine.** Training each row through
   validated scalar snapshots would impose repeated device synchronization and
   Python overhead. Batched functions use the same closed-form equations and
   are checked against scalar reference rows. Cost: two representations with
   explicit parity tests; no replacement of the accepted runtime.
2. **New one-hop curriculum version.** Transform independently generated canonical
   two-link episodes into one-link public episodes. Do not monkeypatch suite
   tables, use unchecked dataclass construction, or modify Phase 1 generation.
   Cost: a separately authenticated curriculum transform, not a primary suite.
3. **Unassisted component validation, not timed-runtime validation.** The Phase 3
   gate rolls out learned retrieval/composition/halting choices without teacher
   records or focus corrections, in a bounded untimed content harness. It does
   not test endogenous world-time scheduling. Cost: Phase 4 still must establish
   the two-hop/autonomous timing and pilot gates independently.
4. **No semantic retrieval mask.** Only arrival, validity, consumption,
   refractory, and schema legality can remove candidates. Subject equality and
   terminal relevance remain learned. Cost: a harder but meaningful retrieval
   gate; exact-match lookup is not an acceptable fallback.
5. **Training archives are distinct from runtime archives.** Safetensors stores
   weights, AdamW moments, and RNG bytes; strict metadata stores named parameter
   groups, progress, and hashes. Cost: a small training-specific codec, without
   claiming learned mid-world checkpoint support.
6. **All-memory preview starts at activation.** During observation there is no
   scorer call; the controller receives a zero preview and dormant legal guards.
   Cost: this explicitly resolves the general “every post-event preview” wording
   against the more specific prohibition on pre-activation deliberation.
7. **A dormant predicted correct guard is still trainable.** Its timing/race
   losses are masked until it has a valid crossing; the active-margin loss always
   supplies gradient. No fake finite crossing or differentiable clamp may be used
   as an autonomous prediction. Cost: timing learning follows guard activation.
8. **Component metrics have explicit denominators.** Require retrieval,
   composition, and complete one-hop content-chain accuracy each strictly above
   99%; timing regression is reported separately. Cost: these are demanding
   component gates, not a replacement or relaxation of timed episode success.

## 3. File Map and Task Order

Configuration, script, test, and documentation paths are repository-relative.
Production module shorthand `models/`, `train/`, `memory/`, `eval/`, `eventflow/`,
and `env/` always expands beneath `src/silent_cascade/`; for example,
`models/types.py` means `src/silent_cascade/models/types.py`. Existing source
files not listed as new remain unchanged. New tests may share fixtures in
`tests/neural/conftest.py`; add no imports to the existing global conftest.

| Task | New production files | Responsibility |
|---|---|---|
| 1 | `models/{__init__,config,types,errors}.py`, `train/{__init__,config,state}.py`; `configs/model/neural_components.yaml`, `configs/train/{smoke,one_hop}.yaml` | Typed configuration, batch contracts, progress |
| 2 | `memory/{tensor_store,encoder,eviction}.py` | Public tensor memory, shared embeddings, explicit stress eviction |
| 3 | `memory/retrieval.py`, `models/common.py` | Learned scorer, safe preview, context assembly |
| 4 | `models/{dynamics,controller,jump}.py` | Batched flow/guards and bounded learned jumps |
| 5 | `models/{heads,event_flow}.py`, `eval/{__init__,compute}.py` | Prediction heads, composed model, exact compute accounting |
| 6 | `train/{curriculum_data,traces,batches}.py` | Versioned curriculum, oracle targets, deterministic batches |
| 7 | `train/unroll.py` | Predict-before-teacher full differentiable unroll |
| 8 | `models/losses.py` | Masked objectives and diagnostic reductions |
| 9 | `train/checkpoints.py` | Safe exact training continuation and weight export |
| 10 | `train/{component_eval,curriculum,trainer}.py` | Unassisted component gate, optimizer loop, stage policy |
| 11 | `train/{__main__,cli,verification}.py`; `docs/phase3-neural-components.md` | Working local interface and device/import/package integration |
| 12 | `train/{provenance,evidence_types,evidence}.py`, `scripts/{check_phase3_components,verify_phase3_gate_artifact}.py` | Source-bound raw evidence and independent verifier |
| 13 | `manifests/validation/phase3/one-hop-10000.json`, `manifests/validation/phase3/component-gate.json` | Actual fixed corpus, training, evaluation, evidence commit |

Execution is sequential by task number. Component implementations may be handed
to fresh subagents; only independent read-only reviews run in parallel. Each
task gets a spec/quality review before its consumers begin. The full plan has
one final whole-phase review; source fixes invalidate new Phase 3 evidence and
require recollection from a new reviewed source, not patched completion claims.

## 4. Shared Tensor and Learning Contract

### Dimensions and inputs

`B` is batch size (1–128); `M=64`; `D=456` is concatenated latent width in order
`z_fast, z_slow, drives, focus_key, hypothesis_latent`. Guards are separate.
No learned input contains a record ID, memory index, episode ID, path length,
remaining hops, private deadline, teacher delta, or curriculum stage ID.

`RecordTensorBatch` holds categorical IDs/missing masks, host record IDs and
timestamps for bookkeeping, and public scalar features. Encode a record from:

- subject embedding 32; object embedding 32 (zero for absent object);
- kind embedding 8; hazard embedding 8 (four classes plus a missing category);
- provenance embedding 4;
- six float32 scalars: object-present, hazard-present, delay-present,
  `log1p(delay)` or zero, `log1p(observed_at - initial_time)`, confidence.

Total record encoder input is `90`; output is `96`. The shared entity table is
`64 x 32`; absent values are masked, not additional semantic entities. The same
table encodes activation entities. Record IDs remain host side solely for
selection/support/trace correspondence. Never encode their numeric values.

`TensorWorkspace` has `latent[B,456]` and `accumulators[B,3]`.
`ModelContext` adds memory embeddings `[B,64,96]`, eligibility `[B,64]`,
support mask `[B,64]`, active slot indices `[B]` (`-1` absent), modes `[B]`,
public time features `[B,2]`, and hypothesis features `[B,8]`.

Time features are `log1p(time - initial_time)` and
`log1p(max(time - activation_time,0))`, the latter zero before activation.
Hypothesis features are presence, four-class one-hot, safe flag, confidence,
and signed `log1p(abs(predicted_deadline - time))`; use zeros for absent fields.
The deadline here is a **previously predicted** register value (teacher value
only after the corresponding training jump), never scorer truth.

Mode embedding is `6 x 8`. Support summary is the mean of all support embeddings
(including consumed supporting records), or zeros. Context width is therefore
`456 + 8 + 96 + 2 + 8 = 570`.

### Network shape

| Module | Fixed default dimensions |
|---|---|
| Record encoder | `90 -> 192 -> 96`, SiLU |
| External encoder | `132 -> 256 -> 256`; input record 96, activation entity 32, two event-kind indicators, time 2 |
| External injection | `256 -> 640`, split into target/gate for fast+slow 320 |
| Focus projection | `32 -> 64`, tanh |
| Retrieval query | `(64+256+64+8+8+96)=496 -> 256`, SiLU |
| Bilinear term | query 256 dot projected record `96 -> 256`, divide by `sqrt(256)` |
| Pair scorer | concatenated query/record `352 -> 256 -> 1`, SiLU, added to bilinear term |
| Controller | context+preview `670 -> 512 -> 512 -> 918` |
| Shared internal jump | context+active record+event embedding `674 -> 512 -> 912` |
| Composition trunk | context+active record `666 -> 256`, separate output heads |
| Action trunk | context `570 -> 256`, class 5 and lead-fraction 1 |

Preview width is `100`: top score, top-two margin, entropy, eligible count / 64,
and a 96-vector soft top-4 pool. Zero eligible records gives four zeros and a
zero pool; one record gives margin/entropy zero. Stable ties are resolved using
record IDs only by the host selection adapter, never by learned positional
features. For preview pooling, include all records tied at the fourth-score
boundary and softmax over that selected set; do not let an arbitrary slot-order
top-k break invariance. Preview pooling/metrics must be invariant under a slot
permutation within documented float32 reduction tolerance.

Use selective LayerNorm after hidden linear layers in controller, jump,
composition, and action trunks; no BatchNorm, dropout, recurrent text, or
normalization across memory slots. Count actual parameters, including shared
tables once. The stated widths are expected to be near 2.8M; the measured count,
not this estimate, determines acceptance.

The controller's `918` outputs split into latent targets/rates `456+456`, guard
targets/rates `3+3`. Targets use tanh; rates use the specification's clamped
softplus. Guard targets use `2*sigmoid(raw)` clamped to
`[torch.finfo(float32).eps, 2-torch.finfo(float32).eps]` to preserve open bounds
under finite-precision saturation. Store both raw bounded targets and the
mode-masked execution targets; loss sees raw targets, scheduling sees masked
targets. OBSERVING/QUIESCENT/TERMINAL execution targets are dormant.

Jumps split `912` outputs into target/gate for all latent channels:
`next = (1-sigmoid(gate))*previous + sigmoid(gate)*tanh(target)`.
External FACT injections update fast/slow only. ACTIVATE also installs the
learned focus projection. LINK composition installs the predicted/teacher-next
entity's focus projection. Terminal composition updates hypothesis latent via
the learned jump, not a fixed hazard-class impulse. All causal jumps reset
accumulators to zero. No controller call occurs during analytic flow.

### Composition, action, and teacher contract

Composition outputs: role logits 5 in `ComposeRole` order; next-focus logits 64;
hazard logits 4; nonnegative `log1p(delay)` prediction; normalized-deadline
prediction; status logits `[hazard,safe,null]`; confidence logit; support-append
logit; continue-search logit. Action outputs: class logits `[0,1,2,3,abstain]`
and sigmoid lead fraction. Threshold boolean decisions at sigmoid `>=0.5`.

Normalize deadline as `(deadline - activation_time) / (1 + elapsed_since_activation)`;
the denominator is public and available to every condition. Supervise delay and
normalized deadline separately under their masks. The action lead target is
`0.175`; action time is ultimately scheduled by the learned ACT guard, **not**
hard-coded `0.825 * observed_delay` or a post-hoc snapped timestamp.

Training labels live only in `train` dataclasses. They include step kind/delta,
selected record ID, focus, role, hazard, delay/deadline, status, confidence,
support/continue decisions, and masks. A model call receives only `ModelContext`
or current external features, never the batch/target container.

---

### Task 1: Strict Neural Configuration and Public Batch Contracts

**Files:** Create the Task 1 files in the map; create
`tests/neural/{conftest.py,test_config.py,test_types.py}`. No existing source edits.

**Interfaces:**

- `models.config.NeuralModelConfig`: strict defaults for the exact dimensions,
  widths, top-k 4, and parameter ceiling above; no arbitrary architecture plugin.
  An explicit `architecture_profile=debug` may use hidden widths record 96,
  external 64, query 64, controller 128, jump 128, and heads 64. State/entity/
  record-output dimensions remain fixed. Production requires the exact Section 4
  widths; only the fixed-64-example overfit regression uses this debug profile.
  Scorer concatenation/dot dimensions derive from the query width, not a second
  hard-coded production-only value.
- `train.config.Phase3Config(Phase2Config)`: adds `neural` and `training` sections;
  consume with the existing generic `resolve_config`.
  `parse_phase3_canonical(raw: str) -> Phase3Config` validates bounded JSON,
  restores the seven known Phase 2 integer-key allocation maps in the new codec,
  performs strict model validation, and requires an exact canonical byte round trip.
  Do not modify the existing Phase 2 parser. Test this before checkpoint work.
- `models.config.LossWeights` defines all ten combined coefficients from Task 8;
  the production defaults are exact, and recorded pilot overrides are bounded
  to one-quarter through four times each default. Store it at
  `Phase3Config.training.loss_weights`.
- `models.errors.NeuralError(SilentCascadeError)` and
  `train.state.TrainingError(SilentCascadeError)` provide bounded typed failures.
- `TensorWorkspace.zeros(batch_size: int, device: str) -> TensorWorkspace`;
  `TensorWorkspace.row(index: int) -> ContinuousState` preserves autograd and
  provides the single-row parity boundary.
- `ModelContext` uses the exact fields/shapes from Section 4. Construction
  validates shapes/dtypes/devices/finite values, but hot unroll updates avoid
  scalar device synchronization at every tensor assignment.
- `TrainProgress`: schema version, optimizer step, next batch counter, stage,
  data root/public-ID seeds, best metric/step, patience counter, retained
  checkpoint descriptors, validation-manifest SHA; never a model input.

- [ ] **Step 1: Write failing configuration/shape tests.** Include unknown keys,
  bool-for-int, CPU float64, malformed masks, mixed-device tensors, batch 129,
  and coordinated attempts to enlarge frozen runtime limits.

```python
def test_phase3_config_adds_neural_fields_without_changing_phase2(neural_config):
    assert neural_config.neural.entity_dim == 32
    assert neural_config.neural.record_dim == 96
    assert neural_config.limits.max_trainable_parameters == 5_000_000
    assert neural_config.training.max_steps == 75_000

def test_workspace_row_is_independent_and_differentiable():
    from silent_cascade.models.types import TensorWorkspace
    batch = TensorWorkspace.zeros(2, "cpu")
    batch.latent.requires_grad_()
    row = batch.row(1)
    row.z_fast.sum().backward()
    assert batch.latent.grad[1, :256].eq(1).all()
    assert batch.latent.grad[0].eq(0).all()
```

Define `neural_config` in the new scoped conftest using real YAML resolution;
smoke overrides only workload values, never shape/science constants.

- [ ] **Step 2: Run** `uv run pytest -q tests/neural/test_config.py tests/neural/test_types.py`.
  Expected RED: missing new modules/contracts, not a dependency or syntax error.
- [ ] **Step 3: Implement strict contracts and complete YAML overlays.**

```yaml
# configs/train/one_hop.yaml
training:
  profile: phase3_one_hop
  optimizer: adamw
  learning_rate: 3.0e-4
  weight_decay: 1.0e-4
  batch_size: 128
  gradient_clip_norm: 1.0
  max_steps: 75000
  validation_every_steps: 1000
  fixed_validation_episodes: 10000
  early_stop_patience_validations: 15
  checkpoint_keep_best: 3
  checkpoint_keep_latest: 1
  model_seed: 11
  train_root_seed: 311
  train_public_id_seed: 331
  validation_root_seed: 313
  validation_public_id_seed: 337
  curriculum_version: ofd-one-hop-v1
```

The smoke overlay uses profile `phase3_smoke`, batch 8, steps 4, validation
interval 2, validation size 16, and the same architecture/default losses.
Record its separate config hash; it can never satisfy the production gate.
Loss coefficients from Task 8 belong to `training.loss_weights` with
the exact specification defaults. Configure AdamW with default betas
`(0.9,0.999)`, epsilon `1e-7`, `foreach=False`, `fused=False`, no scheduler.

Pre-freeze correction (2026-09-15, standing approval): the original plan selected
epsilon `1e-8`. A real fixed-seed CPU/MPS first-step comparison passed every
forward/loss/gradient and discrete-choice check but failed ten updated weights.
Opposing objective gradients near `1e-3` cancelled to about `1e-7`; norm clipping
put the net values near epsilon, and AdamW amplified backend rounding into a
maximum weight difference near `5.48e-5`. Independent scalar analysis reproduced
the difference. This is not a claim of incorrect backend arithmetic.

Use `1e-7` consistently in both profiles, both devices, real training, parity,
archives and resume. This is a deliberate optimizer change, not an equivalent
formula or test-only workaround: it lowers worst-case fresh-step sensitivity
`learning_rate / epsilon` from 30000 to 3000, while potentially slowing genuine
small-gradient learning. The value is fixed before retesting, not selected from
an undisclosed passing-value sweep. Keep every tolerance, fixture, loss, seed,
budget and accuracy threshold unchanged. Regenerate configuration hashes and
reject old-configuration resume; do not reuse previous debug learning evidence
as evidence for this optimizer. Canonical specification Section 8.6 does not
fix epsilon and remains byte-identical. See `docs/deviations.md` for the ruling.

The first stabilization candidate, `1e-6`, passed the unchanged numerical and
resume checks but failed the fixed-64 learning gate at the unchanged 1000-step
limit (63/64 chains). A saved-model probe localized the remaining error to
learned fast-state exposure under the deliberately zero-world-time content
harness; memory, masks, mode and discrete decisions before that error agreed.
It is not corrected with teacher state or a different evaluator. Declare one
intermediate candidate, `1e-7`, before retesting: its worst-case update
sensitivity is 10x below the original, with less small-gradient damping than
`1e-6`. Preserve the complete three-configuration development history. If this
candidate fails either unchanged gate, stop this bounded optimizer adjustment
and reassess; do not continue an automatic sweep or raise the step limit.

- [ ] **Step 4: Run the tests to GREEN**, plus existing Phase 2 config tests and
  `uv run ruff check .`; prove original resolved Phase 2 SHA remains
  `29e4afac2118bdac5a7a01896167805f1fd1f7379a13e5b21566fef2c34fd546`.
- [ ] **Step 5: Commit** only new configuration/types/tests:
  `git commit -m "feat: define Phase 3 neural training contracts"`.

### Task 2: Public Tensor Memory, Encoders, and Explicit Stress Eviction

**Files:** Create `memory/{tensor_store,encoder,eviction}.py` and
`tests/neural/{test_tensor_memory,test_encoder,test_eviction}.py`.

**Interfaces:**

- `pack_memory(memories: tuple[BoundedMemory,...], initial_times: tuple[float,...], *, device: str) -> RecordTensorBatch`.
- `RecordTensorBatch.permute_slots(permutations: torch.Tensor) -> RecordTensorBatch`
  permutes tensors and host-ID correspondence together.
- `RecordEncoder(config: NeuralModelConfig)` owns shared categorical tables;
  `forward(records: RecordTensorBatch) -> torch.Tensor[B,64,96]`;
  `encode_entities(ids: torch.Tensor) -> torch.Tensor[...,32]` reuses its entity table.
- `append_with_stress_eviction(memory, event, *, protected_ids: frozenset[int]) -> EvictionResult`
  returns `memory` and optional `evicted_record_id`; existing primary append is unchanged.

- [ ] **Step 1: Write failing tests** for all record kinds, absent fields, time
  origin, padding, permutation equivariance, shared entity gradients, no ID/index
  features, and no source mutation. Include capacity and every eviction tie key.

```python
def test_record_ids_do_not_enter_embeddings(public_memories, neural_config):
    from dataclasses import replace
    from silent_cascade.memory.encoder import RecordEncoder
    from silent_cascade.memory.tensor_store import pack_memory
    memory = public_memories[0]
    renamed = replace(memory, records=tuple(
        replace(slot, record=replace(slot.record, record_id=slot.record.record_id + 100))
        for slot in memory.records
    ))
    encoder = RecordEncoder(neural_config.neural)
    left = encoder(pack_memory((memory,), (0.0,), device="cpu"))
    right = encoder(pack_memory((renamed,), (0.0,), device="cpu"))
    torch.testing.assert_close(left, right, rtol=0, atol=0)
```

Create the local `public_memories` fixture from hand-authored `ExternalEvent`
FACT values and existing `append_perceived_fact`; it calls no oracle/generator.
Import torch explicitly in every test module that uses it.

- [ ] **Step 2: Run** `uv run pytest -q tests/neural/test_tensor_memory.py tests/neural/test_encoder.py tests/neural/test_eviction.py` and record RED.
- [ ] **Step 3: Implement** the Section 4 encoder and padding mask. Do not store
  learned embeddings in immutable `MemoryRecord`, cache detached training
  embeddings across optimizer updates, or use ordinal IDs as features.

Eviction sorts eligible stored records by
`(record.confidence, record.observed_at, record.record_id)` ascending. Protect
the active record and every active support ID passed by the caller. Validate
the protected set first; if all records are protected, raise `DynamicsError`
without mutation. Duplicate incoming IDs, illegal provenance, and malformed
records fail before choosing a victim. A tensor cache must rebuild/invalidate
the evicted slot; no stale embedding or support reference survives. Primary
`append_perceived_fact` still raises on overflow and is never redirected here.

Effective protection is the union of caller-protected IDs and every record ID
referenced by a stored record's `support_ids`. Reject existing dangling support
IDs before mutation. If this leaves no eligible victim, raise the same controlled
memory-exhaustion error. Never remove or rewrite immutable support history just
to make an eviction possible; perceived-only primary behavior remains unchanged.

Use this private selection kernel after validating incoming/protected IDs;
`records` contains the existing stored-slot objects, not neural embeddings:

```python
def _eviction_id(records, protected_ids):
    candidates = [slot.record for slot in records
                  if slot.record.record_id not in protected_ids]
    if not candidates:
        raise DynamicsError("No unprotected memory record can be evicted")
    return min(candidates, key=lambda record: (
        record.confidence, record.observed_at, record.record_id
    )).record_id
```

- [ ] **Step 4: Run to GREEN**, including `tests/unit/test_runtime_memory.py` and
  finite encoder gradients on CPU/available MPS.
- [ ] **Step 5: Commit** `git commit -m "feat: add neural memory representation"`.

### Task 3: Learned Retrieval and Non-Mutating Preview

**Files:** Create `memory/retrieval.py`, `models/common.py`, and
`tests/neural/test_retrieval.py`; extend only the new scoped fixtures.

**Interfaces:**

- `context_features(context: ModelContext, mode_embedding: nn.Embedding) -> Tensor[B,570]`.
- `RetrievalScorer(config)` has `forward(context) -> RetrievalScores` and
  `preview(context) -> RetrievalPreview` using the **same** scorer.
- `RetrievalScores`: raw scores `[B,64]`, legal masked logits, eligible mask;
  `RetrievalPreview`: features `[B,100]`, has-candidate boolean `[B]`.
- `select_record_ids(scores, record_ids) -> tuple[int|None,...]` uses argmax with
  lowest record ID for exact ties; no learned module consumes that ID.

- [ ] **Step 1: Write RED tests**, including a high-scoring wrong-subject record
  that remains eligible, all-masked and one-candidate rows, ties, consumed
  support pooling, preview immutability, and a flowed query changing selection.

```python
def test_preview_is_finite_for_empty_memory(model_context, neural_config):
    from dataclasses import replace
    from silent_cascade.memory.retrieval import RetrievalScorer
    context = replace(model_context, eligibility=torch.zeros_like(model_context.eligibility))
    preview = RetrievalScorer(neural_config.neural).preview(context)
    assert torch.isfinite(preview.features).all()
    assert preview.features.eq(0).all()
    assert not preview.has_candidate.any()
```

`model_context` is a batch of hand-authored public memories with zero workspace,
declared modes, no supports/hypothesis, and public time features. Implement it
in the scoped conftest using Task 1/2 constructors; do not create privileged
fixtures inside production model code.

- [ ] **Step 2: Run** `uv run pytest -q tests/neural/test_retrieval.py` and record RED.
- [ ] **Step 3: Implement** the query/bilinear/pair network and soft top-4 preview.
  Gather nonempty rows before softmax; never softmax a row of all `-inf` and
  subsequently multiply NaN by zero. Mask invalid entries before top-k/entropy.
  Do not use `subject_id == focus` or oracle reachability as a legality mask.
  Do not mutate active slots, support IDs, refractory metadata, or state in a
  preview. Keep selected-record decoding separate from scoring.

The pool kernel includes boundary ties and never normalizes an empty row;
test both its permutation behavior and its encoder/scorer gradients:

```python
def _preview_pool(scores, embeddings, eligible):
    live = eligible.any(dim=1)
    output = embeddings.new_zeros((embeddings.shape[0], embeddings.shape[2]))
    if not live.any():
        return output
    logits = scores[live].masked_fill(~eligible[live], -torch.inf)
    boundary = logits.topk(k=4, dim=1).values[:, -1:]
    selected = eligible[live] & (logits >= boundary)
    weights = logits.masked_fill(~selected, -torch.inf).softmax(dim=1)
    pooled = (weights.unsqueeze(-1) * embeddings[live]).sum(dim=1)
    return output.index_copy(0, live.nonzero(as_tuple=True)[0], pooled)
```
- [ ] **Step 4: Run to GREEN**, including memory permutation and scorer gradient
  tests. Re-score after changing flowed state; prove no stale-preview reuse.
- [ ] **Step 5: Commit** `git commit -m "feat: learn retrieval and guard previews"`.

### Task 4: Batched Analytic Dynamics and Learned Bounded Jumps

**Files:** Create `models/{dynamics,controller,jump}.py` and
`tests/neural/{test_neural_dynamics,test_controller,test_neural_jump}.py`.

**Interfaces:**

- `BatchedSegmentParameters`: `flow_targets`/`flow_rates` `[B,456]`,
  `raw_guard_targets`/`guard_targets` `[B,3]`, and `guard_rates` `[B,3]`.
- `flow_batch(state: TensorWorkspace, parameters, dt: Tensor[B]) -> TensorWorkspace`.
- `crossings_batch(a: Tensor[B,3], targets, rates) -> Tensor[B,3]` uses `inf`
  only as the explicit dormant sentinel, never in a loss reduction.
- `FlowGuardController(config).forward(context, preview) -> BatchedSegmentParameters`.
- `SharedJump(config).forward(context, event_kinds: Tensor[B]) -> TensorWorkspace`.
- `ExternalEncoder(config, record_encoder)` and `inject_external` implement
  Section 4's current-fact/activation encoding without memory retrieval.

- [ ] **Step 1: Write RED tests** for formula/gradient equivalence to scalar rows,
  zero/tiny/huge/split elapsed time, dormant and near-threshold guards, saturated
  controller logits, reset guards, masked modes, and bounded repeated jumps.

```python
def test_batch_crossings_match_completed_scalar_kernel():
    from silent_cascade.eventflow.guards import crossing_offsets_tensor
    from silent_cascade.models.dynamics import crossings_batch
    a = torch.zeros((2, 3), dtype=torch.float32)
    targets = torch.tensor([[1.2, 0.9, 1.5], [0.5, 1.0, 1.1]])
    rates = torch.tensor([[2.0, 1.0, 0.1], [1.0, 1.0, 500.0]])
    expected = torch.stack([crossing_offsets_tensor(*row)
                            for row in zip(a, targets, rates, strict=True)])
    torch.testing.assert_close(crossings_batch(a, targets, rates), expected)
```

- [ ] **Step 2: Run** `uv run pytest -q tests/neural/test_neural_dynamics.py tests/neural/test_controller.py tests/neural/test_neural_jump.py` and record RED.
- [ ] **Step 3: Implement** vectorized `origin + (-expm1(-rate*dt))*(target-origin)`.
  The caller retains anchors; `flow_batch` does not secretly call a controller.
  Zero elapsed is exact identity per row. Validate dt finite/nonnegative before
  execution. Evaluate crossing logarithms only on active entries, with the
  stable `log1p((1-a)/(A-1))/rate` expression. Cross-check one row's materialized
  state through actual `ContinuousState`, `SegmentParameters`, and `state_at`.

Use these tensor kernels after boundary validation; broadcasting `dt[:,None]`
is intentional, and the crossing kernel never evaluates dormant logarithms:

```python
def _flow_values(origin, target, rate, dt):
    weight = -torch.expm1(-rate * dt[:, None])
    return origin + weight * (target - origin)

def _crossing_values(accumulators, targets, rates):
    active = (accumulators < 1) & (targets > 1)
    result = torch.full_like(accumulators, torch.inf)
    offsets = torch.log1p(
        (1 - accumulators[active]) / (targets[active] - 1)
    ) / rates[active]
    return result.masked_scatter(active, offsets)
```

No fabricated runtime event IDs or checkpoint hashes are needed to train.
The batched implementation does not implement an event queue, external horizon,
timer, or polling loop. Keep tensor validation outside hot inner equations,
while checking finite outputs/losses before every optimizer step.

- [ ] **Step 4: Run to GREEN** on CPU and available MPS, and existing scalar flow
  and guard tests unchanged. Assert actual float32 MPS outputs, not just mocked
  device arguments.
- [ ] **Step 5: Commit** `git commit -m "feat: add differentiable EventFlow dynamics"`.

### Task 5: Prediction Heads, Model Assembly, and Compute Accounting

**Files:** Create `models/{heads,event_flow}.py`, `eval/{__init__,compute}.py`, and
`tests/neural/{test_heads,test_event_flow_model,test_neural_compute}.py`.

**Interfaces:**

- `ComposeHeads.forward(context) -> ComposePredictions` and
  `ActionHeads.forward(context) -> ActionPredictions`, exact fields in Section 4.
- `EventFlowModel(config: NeuralModelConfig)` owns one record encoder, one
  shared scorer, controller, jump, mode table, external encoder, focus
  projection, and heads. Expose `observe`, `preview_and_control`, `recall_scores`,
  `compose`, `action`, `jump`, and `initial_context` methods. Each method consumes
  only the explicitly public tensor inputs defined above.
- `NeuralComputeMeter` is a context-managed observer, with immutable snapshots:
  module calls, row-level transitions, records scored, estimated MACs, flow
  evaluations, jump applications, opportunities, parameters, memory bytes,
  foundation-model calls, elapsed time, optional MPS peak allocation.
- `parameter_counts(model) -> dict[str,int]` returns total, entity table, and
  non-entity subtotal without double-counting shared parameters.

Freeze the model method signatures before consumers are written:

```text
initial_context(batch_size: int, *, device: str) -> ModelContext
observe(context: ModelContext, event: ExternalFeatures) -> ModelContext
preview_and_control(context: ModelContext) -> tuple[RetrievalPreview, BatchedSegmentParameters]
recall_scores(context: ModelContext) -> RetrievalScores
compose(context: ModelContext) -> ComposePredictions
action(context: ModelContext) -> ActionPredictions
jump(context: ModelContext, event_kinds: torch.Tensor) -> TensorWorkspace
```

`ExternalFeatures` is a neutral Task 1 type holding only the current event's
record fields/activation entity, kind, and public time features for each active
row; padding and private events cannot be represented as a semantic event.
The assembler's `observe` performs encoding/injection and returns updated public
context, but the caller separately requests the next preview/controller exactly
once. Learned shared table aliases must have one owning module; if state-dict
aliases are emitted, checkpoint metadata must bind and validate their equality.

- [ ] **Step 1: Write RED tests**, including all head shapes/gradients, parameter
  sharing, class-abstain output, reference MACs, preview plus actual recall
  counting twice, and no hook duplication after repeated instrumentation.

```python
def test_linear_mac_accounting():
    from silent_cascade.eval.compute import NeuralComputeMeter
    layer = torch.nn.Linear(3, 5)
    with NeuralComputeMeter(layer) as meter:
        layer(torch.ones(2, 4, 3))
    assert meter.snapshot().estimated_macs == 2 * 4 * 3 * 5

def test_model_budget(neural_config):
    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.eval.compute import parameter_counts
    counts = parameter_counts(EventFlowModel(neural_config.neural))
    assert counts["entity_table"] == 64 * 32
    assert 1_000_000 <= counts["total"] <= 5_000_000
    assert counts["total"] == counts["entity_table"] + counts["non_entity"]
```

- [ ] **Step 2: Run** `uv run pytest -q tests/neural/test_heads.py tests/neural/test_event_flow_model.py tests/neural/test_neural_compute.py` and record RED.
- [ ] **Step 3: Implement** prediction-only heads and exact shared ownership.
  One MAC is one multiply-accumulate; linear cost is input rows times input
  width times output width, bilinear/scorer dot products are counted explicitly.
  Bias additions/SiLU/LayerNorm/exp/log operations are separately named estimates,
  not falsely called zero total compute. Embedding lookups count calls/bytes,
  not dense MACs. Count encoded/scored padded slots if actually executed;
  report eligible-record counts separately. Record training backward work
  separately from inference forward estimates.

The linear hook uses this exact MAC kernel; instrument non-linear/module-level
work separately rather than folding it into this number:

```python
def _linear_macs(module: torch.nn.Linear, inputs: tuple[torch.Tensor, ...]) -> int:
    values = inputs[0]
    rows = values.numel() // module.in_features
    return rows * module.in_features * module.out_features
```

The meter must not alter outputs, gradients, RNG, state, or scheduling. It
syncs MPS before/after wall timing and reports unavailable peak allocation as
null, never fabricated zero. No energy claim. Repeated contexts remove hooks.

Eligibility/mode counts may be reduced on CPU after the synchronized timer ends,
but their call-time identity must be protected without adding GPU work inside
hooks. Capture each observed tensor's mutation version as host metadata and
reject in-place mutation before publishing a snapshot. Reject metadata whose
mutation version cannot be tracked; never publish silently stale counts. The
planned functional context replacements remain supported. On such a failure,
remove every hook and refuse a valid snapshot; unsafe out-of-band storage writes
are outside the supported tensor-input contract.

- [ ] **Step 4: Run to GREEN**, demonstrate that separate preview/recall calls
  incur separate scorer work and that observation invokes no retrieval.
- [ ] **Step 5: Commit** `git commit -m "feat: assemble trainable EventFlow components"`.

### Task 6: Versioned One-Hop Curriculum, Teacher Targets, and Online Batches

**Files:** Create `train/{curriculum_data,traces,batches}.py` and
`tests/neural/{test_curriculum_data,test_teacher_traces,test_batches}.py`.

**Interfaces:**

- `CurriculumKey`: version, split (`train`, `validation`, or `debug`), data root
  seed, public-ID seed, episode index, stage (`one_hop`, `two_hop`, `primary`,
  or `robustness`); this is training-private metadata.
- `make_curriculum_example(config: Phase3Config, key: CurriculumKey) -> CurriculumExample`.
- `CurriculumExample`: public `AgentInit`, tuple of public FACT/ACTIVATE events,
  private oracle solution/trace, private terminal horizon, parent recipe/hash,
  transform identity, and canonical example hash. It is **not** a model input.
- `build_teacher_trace(example: CurriculumExample) -> TeacherTrace`.
- `next_training_batch(config: Phase3Config, *, stage: str, batch_counter: int) -> TrainingBatch`.
- `TrainingBatch`: padded public observations, public record fields/IDs, host
  float64 times, separate `TeacherTargets`, per-operation masks, private example
  keys/hashes, and next batch counter. `to(device)` converts floating **features
  and relative deltas** to float32; absolute host metadata stays on CPU.
- `PublicInputBatch` in `models.types` contains only the public observation
  fields, public masks, and host public time/record-ID correspondence. A training
  batch produces this projection without targets or private keys.
- `TeacherTargets` has named `[B,T]` tensors `kind`, `delta`, `selected_slot`,
  `focus`, `role`, `hazard_type`, `log_delay`, `normalized_deadline`, `status`,
  `confidence`, `append_support`, `continue_search`, `action_class`, `action_lead`,
  plus a same-shaped validity mask per target and real-step/final-dormancy masks.
  Float targets/deltas are float32 on the chosen device; absolute teacher times
  are distinct host metadata. Invalid integer padding uses `-1` and is never
  evaluated by CE.
- `ComponentTarget` is a training-private immutable ordered record/content
  target sequence and terminal class/status; it has no model methods.
- `ComponentCorpus` holds immutable public examples and separate target tuples,
  with `public_batch(start: int, count: int) -> PublicInputBatch` and private
  scoring slices. A target replacement cannot change the public projection.
- `ComponentManifest` and bounded `load_component_manifest(path, config)` belong
  in `train.curriculum_data` now, so the Task 10 trainer has no forward dependency
  on Task 12 evidence code. The schema binds keys, versions, public/example
  hashes, seeds, source/plan revisions, and counts. Task 12 adds source-authenticated
  production publication; Task 6/10 tests use explicitly debug manifests.

- [ ] **Step 1: Write RED tests** for repeatability, split/domain separation,
  one-hop topology, exact terminal counts, 0–4 nuisance links, permutations,
  target labels, no padding loss, no future observation exposure, and no global
  RNG consumption from data generation.

```python
def test_online_batch_is_counter_addressable(neural_config):
    from silent_cascade.train.batches import next_training_batch
    first = next_training_batch(neural_config, stage="one_hop", batch_counter=7)
    repeat = next_training_batch(neural_config, stage="one_hop", batch_counter=7)
    following = next_training_batch(neural_config, stage="one_hop", batch_counter=8)
    assert first.example_hashes == repeat.example_hashes
    assert first.example_hashes != following.example_hashes
    assert first.next_batch_counter == 8

def test_one_hop_has_no_primary_generator_override(neural_config):
    from silent_cascade.env.config import SuiteName
    from silent_cascade.env.generator import suite_spec
    assert suite_spec(SuiteName.IID_PRIMARY).path_lengths == (2, 3, 4)
    assert neural_config.data.train_path_lengths == (2, 3, 4)
```

- [ ] **Step 2: Run** `uv run pytest -q tests/neural/test_curriculum_data.py tests/neural/test_teacher_traces.py tests/neural/test_batches.py` and record RED.
- [ ] **Step 3: Implement the explicit curriculum procedure.**

For `ofd-one-hop-v1`:

1. Derive local NumPy generators from SHA-256 of canonical
   `("silent-cascade/phase3/data/v1", stage, split, root_seed, episode_index,
   stream_name, attempt)`; use distinct `parent`, `subset`, `presentation`,
   `timestamps`, `trace_jitter`, and `augmentation` domains. Do not add values
   to the frozen shared `SeedStream` enum. Persist keys and accepted attempts.
2. Allocate balanced variant quartets using existing
   `allocate_independent_variants(AllocationLabelKey(...))`; construct valid
   `IndependentEpisodeRequest` values for path length 2 with explicit member
   identity. Use `TRAIN`, `VALIDATION`, or `DEBUG` namespaces, never `FROZEN`.
   Parent generation uses the unchanged primary configuration and generator.
3. Solve the parent's public graph in this training-only module. Retain the
   last relevant LINK; activate its source. Retain exactly the parent's two
   HAZARD records and one SAFE record. Remove the preceding relevant LINK.
4. Sample a desired nuisance-link count uniformly from `{0,1,2,3,4}` independent
   of variant. Select that many original off-path links without replacement.
   If the parent has too few, regenerate the parent using the next bounded
   attempt (maximum 1000), preserving desired count and variant. Do not lower
   the desired count to fit the parent. Check no selected nuisance source is
   reachable from the new activation node and no alternate terminal is reached.
5. Reshuffle retained facts and reassign presentation record IDs from zero.
   Resample observation gaps independently from `[0.1,8]` log-uniform; add one
   independent final activation gap. IDs and timestamps must not retain the
   oracle filtering order. Create a new opaque UUID from a separate HMAC public
   ID domain and public-ID seed; no private coordinate bits enter model features.
6. Preserve terminal delays/class assignments. The private terminal horizon is
   the new activation time plus the parent's matched delay. Build a fresh
   `PublicEpisode`, solve it afresh, and call existing `build_oracle_trace` with
   the real private terminal and a fresh local trace-jitter generator. Do not
   forge a primary `EpisodeRecipe(path_length=1)` or mutate parent truth.
7. Validate exactly one reachable link and the intended hazard/safe/disconnected
   terminal, terminal counts `2/1`, capacity, unique trace, feasible timing,
   independent presentation, and zero model calls. Hash public events, private
   target recipe, parent hash, and transform version in separate domains.

Use canonical, domain-separated seed material rather than process-randomized
Python `hash()`. Import `hashlib`, `json`, and NumPy in this data module:

```python
def _phase3_seed(stage, split, root_seed, episode_index, stream_name, attempt):
    material = json.dumps(
        ["silent-cascade/phase3/data/v1", stage, split, root_seed,
         episode_index, stream_name, attempt],
        ensure_ascii=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest(), "big")
```

Pass all six explicit key fields to `_phase3_seed`, then construct a local
`np.random.default_rng(seed)` for each named stream; no module-global generator.

For `two_hop` and `primary`, use unchanged independent primary generation with
requested paths 2 and 2–4 respectively. For `robustness`, use only paths 2–4 and
paired time factors drawn log-uniformly from `[0.5,2]`, rescaling observations,
delays, teacher intervals, and targets together. An independent fact-order
permutation followed by independently regenerated legal timestamps supplies
order/jitter augmentation. Rebuild the teacher trace after transformation;
never sort relevant facts into path order. These stages are constructible and
unit-tested here; promotion/execution beyond one hop remains Phase 4-gated.

Teacher content is built by walking the oracle trace in `train`, with explicit
pre/post modes, teacher active IDs, support masks, focus, hazard, safe/null,
delay, confidence, append, and continue labels. For a disconnected ending,
supervise abstention and dormant post-trace guards; never append a fake RECALL,
COMPOSE, ACT, or NOOP to represent the absence of an event. Terminal HAZARD/SAFE
and disconnected LINK composition have explicit target masks.

Optional composition hard-negative supervision is a separate training-only
auxiliary row: copy a teacher context, substitute one deterministic eligible
wrong-subject record, label it irrelevant, append=false, and never apply that
auxiliary prediction to the main trace. Add hand-authored contradictory-context
fixtures to exercise that output/loss; do not add contradictory worlds to the
primary training distribution or claim contradiction competence in this phase.

Pad observations/traces to the current batch maxima, not an unbounded transcript.
Primary trace length is at most 11 for four links plus terminal and ACT; one hop
is at most 5. Keep separate masks for real observations, real trace steps,
retrieval, each composition field, action, and final dormancy. Remap teacher
record IDs to slots after every independent memory-tensor permutation.

- [ ] **Step 4: Run to GREEN**, including a 1000-example construction check across
  all three variants and all supported curriculum stages, plus exact batch
  equality under changed global RNG state and out-of-order batch requests.
- [ ] **Step 5: Commit** `git commit -m "feat: add deterministic neural curriculum data"`.

### Task 7: Predict-Before-Teacher Batched Unroll

**Files:** Create `train/unroll.py` and `tests/neural/test_unroll.py`.

**Interfaces:**

- `teacher_forced_unroll(model: EventFlowModel, batch: TrainingBatch) -> UnrollResult`.
- `UnrollResult` stores boundary guard predictions, per-operation outputs,
  applicable target/mask references for the loss reducer, differentiable states,
  and per-module compute; it is training-private and never passed back to model calls.
  Its `loss_inputs() -> LossInputs` method projects only the neutral tensor
  fields defined in Task 8, without copying/detaching their computation graph.
- The unroll uses the existing batch/encoder/controller/jump interfaces only;
  no runtime engine, oracle call, checkpoint hash, or scorer truth is needed
  once a `TrainingBatch` has been constructed.

- [ ] **Step 1: Write RED tests** for causal input isolation, correct prediction
  order, complete cross-step gradient retention, padding, masked quiescence,
  scorer re-evaluation, teacher-record remapping, and fresh graphs after updates.

```python
def test_future_teacher_choices_cannot_change_earlier_predictions(
    model, teacher_batch
):
    from dataclasses import replace
    from silent_cascade.train.unroll import teacher_forced_unroll
    hazard = teacher_batch.targets.hazard_type.clone()
    hazard[:, 3] = (hazard[:, 3] + 1) % 4
    altered = replace(teacher_batch, targets=replace(teacher_batch.targets, hazard_type=hazard))
    first = teacher_forced_unroll(model, teacher_batch)
    second = teacher_forced_unroll(model, altered)
    torch.testing.assert_close(first.boundaries[0].raw_guard_targets,
                               second.boundaries[0].raw_guard_targets, rtol=0, atol=0)
```

Use a five-position positive one-hop fixture for this test. Its only mutation
is the future HAZARD target; no production target-injection API is added.

- [ ] **Step 2: Run** `uv run pytest -q tests/neural/test_unroll.py` and record RED.
- [ ] **Step 3: Implement the complete sequence**, with no input/target aliasing:

```text
zero workspace + empty visible memory
for each actual public observation position:
    flow each active row from its previous public event time
    reveal/encode only this FACT or ACTIVATE, then apply external jump
    reset accumulators; store the newly emitted segment parameters
    before ACTIVATE: zero preview, no retrieval call, all execution guards dormant
    at ACTIVATE: enable one post-event preview and learned guard controller
for each actual teacher internal position:
    retain predictions from the preceding boundary (before teacher disclosure)
    compute differentiable crossings from that boundary's reset accumulators
    advance analytically by teacher delta using those fixed parameters
    score retrieval / composition / action from the flowed current context
    record all predicted outputs before applying teacher choices
    apply teacher-selected record/role/focus/support to discrete training state
    apply the shared learned jump to the continuous state; reset guards
    recompute preview/controller once for every row that actually just jumped
evaluate final no-event/abstain targets without creating an internal event
```

The teacher event type may select which supervised head is evaluated, but must
not be an input to the earlier guard controller or record selector. A jump may
use its actual/teacher event kind **after** prediction. Teacher dt is passed only
to analytic advancement/loss targets; it is never a controller feature.

For a batch with mixed padding/quiescence, gather active rows before neural
calls and scatter functional outputs back. Do not detach recurrent state or
cache an encoder graph across optimizer steps. Do not in-place overwrite a
tensor needed by an earlier backward pass. An external prefix may not encode,
pool, or score a future fact; future input mutation tests must prove this.

Targets/teacher context must not be loaded by any `models` import. Record
teacher-forced diagnostics under that exact label; this result is not
validation accuracy.
The real jump that first enters QUIESCENT still receives its one post-event
controller evaluation. Thereafter that row is inactive. Final dormancy loss
uses the retained output of this boundary, not an extra fake controller tick.

- [ ] **Step 4: Run to GREEN**, including backward from the final action loss to
  an early observation encoder and early jump parameter, complete length-4
  traces, and numerical equality between independently unrolled rows and batches.
- [ ] **Step 5: Commit** `git commit -m "feat: unroll differentiable cognitive traces"`.

### Task 8: Masked Multi-Task Losses and Finite-Gradient Diagnostics

**Files:** Create `models/losses.py` and
`tests/neural/{test_losses,test_gradients}.py`.

**Interfaces:**

- `LossInputs` is a neutral tensor-only prediction/target/mask structure in
  `models.types`; `train.unroll` converts its private result into this structure.
  Loss computation may consume targets; prediction modules never consume them.
- `event_flow_loss(inputs: LossInputs, weights: LossWeights) -> LossBreakdown`.
- `LossBreakdown`: scalar differentiable `total`, named term values, raw
  numerator/denominator pairs, and per-position diagnostics. Detached logging
  views are produced only after the complete objective is constructed.

- [ ] **Step 1: Write RED tests** with exact small hand-calculated losses,
  correct-guard dormancy, all-null batches, empty masks, invalid padding values,
  masked gradients, race losers, and every required nonzero gradient path.

```python
def test_masked_mean_does_not_evaluate_invalid_padding():
    from silent_cascade.models.losses import masked_mean
    values = torch.tensor([2.0, float("nan")], requires_grad=True)
    result = masked_mean(values, torch.tensor([True, False]))
    assert result.item() == 2.0
    result.backward()
    torch.testing.assert_close(values.grad, torch.tensor([1.0, 0.0]))
```

Define `masked_mean(values, mask)` by selecting valid entries **before**
reduction. For an empty selection, return the sum of the empty selected tensor
(a graph-connected zero), not `values.sum()*0`. For CE/log terms, gather valid
rows **before evaluating the loss**, so invalid labels or `inf` are never fed
into an operation whose result is merely masked afterward.

- [ ] **Step 2: Run** `uv run pytest -q tests/neural/test_losses.py tests/neural/test_gradients.py` and record RED.
- [ ] **Step 3: Implement these exact starting terms.**

| Group | Definition and mask |
|---|---|
| Guard | Sum active-margin, inactive-margin, Smooth L1 log-crossing error, and mean `relu(0.05 + correct_delta - wrong_valid_delta)`; only actual finite active crossings enter temporal terms |
| Final dormancy | Mean `relu(A - 0.90)` for all three raw guard targets at genuine post-trace no-event states |
| Retrieval | CE over legally eligible slots at RECALL, teacher slot as target; all-masked rows cannot carry a retrieval target |
| Compose type | CE over five roles; add status CE, confidence BCE, support-append BCE, and continue-search BCE, averaging those five auxiliary/type terms equally |
| Focus | CE over 64 entities only for LINK decisions |
| Hazard | CE over four classes only for HAZARD decisions |
| Deadline | Mean of Smooth L1 on `log1p(delay)` and on the explicitly normalized deadline, only HAZARD decisions |
| Action | CE over four classes plus abstain: correct class at ACT, abstain at genuine safe/null final states; average positive/negative groups equally when both exist |
| Action time | Smooth L1 on lead fraction, target `0.175`, only positive ACT |
| State | Mean squared slow-state jump change plus state-bound violation penalty; no reward for motion and no required nonzero trajectory |
| Event cost | Mean `relu(A-1)` over legal raw guards at real active boundaries; differentiable proxy only, actual event/compute counters remain separate |

The combined weights are exactly guard `1`, retrieval `1`, compose type `1`,
focus `0.5`, hazard `1`, deadline `0.25`, action `1`, action time `0.25`, state
`0.05`, event cost `0.001`. Include final dormancy inside the guard group, with
equal per-boundary weighting; do not let padded trace length change a term's
denominator. Report every subterm separately so an aggregate cannot conceal
failed retrieval or a dormant timing head.

Implement the reduction and final weighting directly; `terms` contains all ten
named group tensors and the strict `LossWeights` has exactly those field names:

```python
def masked_mean(values, mask):
    selected = values[mask]
    return selected.mean() if selected.numel() else selected.sum()

def _weighted_total(terms, weights):
    names = ("guard", "retrieval", "compose_type", "focus", "hazard",
             "deadline", "action", "action_time", "state", "event_cost")
    return torch.stack([terms[name] * getattr(weights, name) for name in names]).sum()
```

Only validation-driven pilot tuning within factor four of each combined
coefficient is permitted, with every attempt recorded. This phase starts with
the exact defaults and does not silently tune after observing the acceptance
run. No RL, REINFORCE, straight-through fabricated event time, or differentiation
through autonomous discrete trajectories.

- [ ] **Step 4: Run to GREEN** with length-1/2/4 traces, all three variants, and
  a two-step optimization regression proving fresh autograd graphs. On any
  nonfinite real loss or gradient, the trainer raises `TrainingError` before
  `optimizer.step` and preserves the offending batch key/checkpoint reference; never `nan_to_num`
  the failure into a normal training result.
  Model-side validation raises `NeuralError`; it must not import `train.state`
  merely to use the trainer's error class.
- [ ] **Step 5: Commit** `git commit -m "feat: train EventFlow with masked trace losses"`.

### Task 9: Safe Training Checkpoints and Exact Resume

**Files:** Create `train/checkpoints.py` and
`tests/neural/{test_training_checkpoint,test_training_resume}.py`.

**Interfaces:**

- `save_training_checkpoint(path: Path, model, optimizer, progress: TrainProgress, *, config: ResolvedConfig[Phase3Config], source_commit: str) -> CheckpointDescriptor`.
- `load_training_checkpoint(path: Path, *, expected_config_sha256: str, expected_source_commit: str, device: str) -> TrainingRestore`.
- `TrainingRestore` contains fresh validated model/optimizer/progress/RNG data;
  `restore_rng()` is an explicit transactional operation after all validation.
- `export_weights(path: Path, model, *, config, source_commit: str) -> CheckpointDescriptor`;
  `load_weights(path, *, expected_sha256: str, device: str) -> EventFlowModel`.
- A descriptor binds relative archive filename, file SHA, model-state SHA,
  config/source, step, stage, and validation score (if actually evaluated).

- [ ] **Step 1: Write RED tests** for uninterrupted versus resumed CPU steps,
  next-batch equality, named shared parameters, optimizer moments/step, empty
  optimizer state before the first step, wrong source/config/schema, truncation,
  oversized/deep metadata, wrong tensor shapes/dtypes, symlinks, preexisting
  output, and failed load leaving global RNG/caller state untouched.

```python
def test_checkpoint_resume_reproduces_next_cpu_step(training_pair, tmp_path):
    uninterrupted, resumed = training_pair.after_identical_first_step(tmp_path)
    left = uninterrupted.step_once()
    right = resumed.step_once()
    assert left.batch_hashes == right.batch_hashes
    torch.testing.assert_close(left.loss, right.loss, rtol=0, atol=0)
    for a, b in zip(uninterrupted.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
```

The `training_pair` fixture is a test-only two-step optimizer harness using
Tasks 6–8, not the not-yet-created full trainer. It must perform a real save,
load, RNG restoration, and second AdamW step.

- [ ] **Step 2: Run** `uv run pytest -q tests/neural/test_training_checkpoint.py tests/neural/test_training_resume.py` and record RED.
- [ ] **Step 3: Implement a closed safetensors format**, schema
  `phase3-training-checkpoint-v1`. Maximum full archive size `256 MiB`, maximum
  JSON metadata `8 MiB`; reject before allocating/loading tensors. Weight-only
  archives use `phase3-model-weights-v1` and maximum `32 MiB`.

Flatten model tensors by unique parameter/state name; AdamW tensors are mapped
to those stable names, never arbitrary Python optimizer-object IDs. Metadata
contains exact parameter-group names/options, step, stage, data seeds/counter,
complete canonical config/hash, full source revision, package/device metadata,
validation values, and RNG metadata. Store Python/NumPy states with the existing
safe RNG codec; CPU/MPS RNG bytes are uint8 tensors. No pickle, `torch.save`,
arbitrary import/class registry, or callable metadata. Reject extra/missing
tensors, aliases with conflicting payloads, malformed exact primitives, and
nonfinite weights/moments. AdamW's non-capturable scalar step may remain CPU
float32 even when moments/parameters use MPS; this is not float64 fallback.

Snapshot unique named parameters as detached owned CPU storage; serialize
buffers similarly and bind every state-dict alias separately in metadata:

```python
def _model_tensor_snapshot(model):
    return {
        "parameter/" + name: parameter.detach().cpu().contiguous().clone()
        for name, parameter in model.named_parameters(remove_duplicate=True)
    }
```

Use `safetensors.torch.save` to obtain bytes, then the validated atomic
no-clobber writer; never let a convenience serializer overwrite a run archive.
Checkpoint loading validates names/shapes/dtypes/alias bindings before copying
into the newly constructed model and reconstructing named AdamW state.

Use atomic no-clobber publication and content-addressed checkpoint names.
An index atomically points to the latest and best three archives only after
hash verification. Reuse an identical existing archive; conflicting bytes
fail. Prune only verified, unreferenced archives owned by this exact run;
never delete a selected/exported evidence checkpoint. Validate all restoration
objects in isolation before restoring RNG or changing caller-owned state.

CPU resume is bit-identical on the same recorded environment. For MPS, restore
on MPS and test next-batch equality plus next-loss tolerance `rtol=1e-4,
atol=1e-5`; do not claim bitwise MPS identity. Weight-only export is CPU portable
and contains every parameter needed for independent component evaluation.
Training checkpoints do not replace paused-world runtime checkpoints.

- [ ] **Step 4: Run to GREEN**, including available-MPS save/load and CPU loading
  of exported MPS weights; keep all existing runtime checkpoint tests unchanged.
- [ ] **Step 5: Commit** `git commit -m "feat: checkpoint neural training safely"`.

### Task 10: Unassisted Component Evaluation and Bounded Training Loop

**Files:** Create `train/{component_eval,curriculum,trainer}.py` and
`tests/neural/{test_component_eval,test_curriculum,test_trainer,test_tiny_overfit}.py`.
Extend the new `train/curriculum_data.py` recipe schema/loader with the minimal
production-versus-debug checks needed by this complete driver. Task 12 retains
final source-closure and Git-introduction authentication.

**Interfaces:**

- `predict_components(model: EventFlowModel, public_inputs: PublicInputBatch) -> ComponentPredictions`:
  public tensors/events only, no examples, targets, manifest coordinates, or oracle imports.
  `ComponentPredictions.to_primitive()` returns its ordered immutable prediction
  records as JSON-compatible primitives, including raw decisions and stop flags.
- `score_components(predictions, targets) -> ComponentMetrics`: called only
  after prediction; private labels stay in the scorer.
- `evaluate_components(model, corpus: ComponentCorpus, *, batch_size: int) -> ComponentEvaluation`:
  explicit prediction then scoring; returns raw rows and derived metrics.
- `train_one_step(model, optimizer, batch, config) -> StepResult`: zero gradients,
  unroll, loss, finite check, backward, finite-gradient check, clip, step.
- `run_training(config: ResolvedConfig[Phase3Config], *, validation_manifest: Path, run_dir: Path, source_commit: str, resume: Path|None = None, device: str|None = None) -> TrainingRunResult`.
  Explicit placement validates availability and records the actual device without
  changing the manifest-bound canonical configuration; `None` uses its preferences.
- `CurriculumController.next_stage(evidence)` implements ordered promotion and
  rejects unsupported/unevidenced promotions; no validation label enters a model.

- [ ] **Step 1: Write RED tests** for label isolation, errors counted as failures,
  bounded stopping, seed/counter resume, best/latest selection, early stopping,
  exact step ceiling, finite-gradient failure, and a real fixed-64-example overfit.

```python
def test_component_predictions_do_not_depend_on_scorer_labels(model, component_corpus):
    from dataclasses import replace
    from silent_cascade.train.component_eval import predict_components
    public = component_corpus.public_batch(0, 8)
    first = predict_components(model, public)
    component_corpus = replace(component_corpus, targets=tuple(reversed(component_corpus.targets)))
    second = predict_components(model, component_corpus.public_batch(0, 8))
    assert first.to_primitive() == second.to_primitive()
```

Production corpus/model APIs must not offer a target-injection shortcut.
Also block `env.oracle` and private environment imports inside a
fresh prediction subprocess after public features are serialized.

- [ ] **Step 2: Run** `uv run pytest -q tests/neural/test_component_eval.py tests/neural/test_curriculum.py tests/neural/test_trainer.py tests/neural/test_tiny_overfit.py` and record RED.
- [ ] **Step 3: Implement the unassisted component harness.**

Process public observations chronologically with actual external flow and
learned observation injections, then hold world time at activation. Starting
from SEARCHING, obtain the learned preview/controller: a dormant RECALL guard
may stop the content process. Otherwise choose a record from learned scores,
apply the learned RECALL jump, predict composition, and apply the predicted
role/focus/support/continue decision and shared jump. At subsequent SEARCHING
states repeat the learned preview/dormancy decision. No teacher-selected record,
teacher focus, teacher role, teacher delta, or known path length enters this loop.

The content harness has a fixed maximum of **four atomic RECALL/COMPOSE
transitions** (two pairs), sufficient for one LINK plus a terminal. At HAZARD or
SAFE, stop after the predicted composition; at a predicted null/dormant state,
stop without a fake event. If the budget ends without a resolved prediction,
record a failed/capped component prediction; never silently count it as abstention.
Wrong or malformed predicted content is a scored component failure with its
raw decisions retained, not a corrected teacher transition. Internal tensor
NaNs, invalid device/dtype, or corrupted metadata still raise a typed error.

This harness deliberately omits world-time crossing/refractory execution and
does **not** instantiate `AgentCondition`, `EventEngine`, or the Phase 5
Ponder-at-Activation comparator. It reports `evaluation_mode=unassisted_content`,
never `timed_success`. Its bounded content loop is not the experimental
EventFlow clock. Phase 4 must independently test that clock with learned weights.

Score these exact metrics after the entire prediction is available:

- Required-recall accuracy: number of required oracle record positions correctly
  predicted, divided by required oracle RECALL positions. Missing predictions
  are wrong; extra selections make complete-chain success false.
- Required-composition accuracy: at each required COMPOSE position jointly
  correct record, role, LINK focus or terminal class/status, support append,
  and legal continuation/stop. Missing/wrong/extra content never reduces the
  denominator. Float delay/deadline accuracy is reported as regression error,
  not folded into an undefined exact-float classification metric.
- Complete-chain accuracy: correct entire ordered record/content chain and
  final hazard class, safe status, or disconnected/null result, with no extra
  operations or cap/failure. Divide by all 10,000 episodes.
- Report positive/safe/disconnected counts and accuracies separately, plus
  action-head accuracy, delay/deadline errors, guard-margin/dormancy diagnostics,
  compute counters, and explicit untimed status.

All three headline component accuracies must be **strictly greater than 0.99**.
For 10,000 complete-chain rows this means at least 9,901 correct, not 9,900.
No denominator conditions on successful retrieval or convenient stopping.
For a disconnected final LINK, either a learned explicit stop or a learned
post-compose preview-dormancy stop is semantically correct; do not penalize the
second solely for disagreeing with the teacher's continue bit. Stopping before
a required record is always wrong. Training continue targets are true while
another oracle record remains and false at a disconnected ending; terminal
roles determine their own mode and mask out the unused continue target.

The trainer generates online batches by counter, uses the exact default AdamW
settings, and records every loss term and per-position diagnostic. Validation
uses `model.eval()`/`no_grad()` and the unassisted harness only; training loss
cannot select a checkpoint. Select highest complete-chain accuracy, then highest
composition accuracy, then earliest step. Keep best three and latest one.
Stop at 75,000 steps or 15 validations without improvement, or at the first
scheduled validation where all three one-hop gates pass. Record the rule and
stop reason. A failed gate stays failed; do not raise the ceiling or reduce the
validation set to manufacture completion.

The optimizer body operates on the complete trace graph in this order; the
caller has already generated the counter-addressed batch and zeroed gradients:

```python
unroll = teacher_forced_unroll(model, batch)
loss = event_flow_loss(unroll.loss_inputs(), config.training.loss_weights)
if not torch.isfinite(loss.total):
    raise TrainingError("Nonfinite training loss")
loss.total.backward()
for name, parameter in model.named_parameters():
    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
        raise TrainingError(f"Nonfinite gradient in {name}")
torch.nn.utils.clip_grad_norm_(
    model.parameters(), config.training.gradient_clip_norm, error_if_nonfinite=True
)
optimizer.step()
```

Use a single deterministic data producer initially (`num_workers=0` if using
a DataLoader); advance the durable next-batch counter only with a completed
step checkpoint. On failure, archive the current batch key and prior checkpoint
reference without disguising the failed batch as consumed.

Checkpoint at initial step zero, each scheduled validation, and final stop.
Only checkpoint-committed journals/counters are durable consumed progress;
retain interrupted segment journals under distinct attempt IDs and replay
deterministically from the preceding checkpoint. A crash during validation or
publication may require replaying up to 1,000 optimizer steps (two in smoke).
This cadence avoids a full archive write per update without changing the
validation interval, step ceiling, or failure semantics.

Both approved profiles use the complete driver. Pull forward only the recipe
schema/loader support required for exact production/debug profile, count,
split, seed, config/source binding, regeneration and variant-allocation checks.
Do not add a runtime placeholder refusing one-hop merely because Task 12 has
not yet run. Task 12 supplies the final historical source authentication and
may tighten these new loader/driver boundaries. No production corpus is
published and no production training is executed before Task 13.

Curriculum policy records oracle/bootstrap evidence and one-hop completion.
Promotion to two-hop requires this component gate; promotion from two-hop to
primary requires the later 10,000-episode **autonomous** zero-dynamics-failure
gate; robustness follows primary. In this phase `run_training` supports only
one-hop and smoke execution and refuses higher stages with an explicit phase
boundary error. Data construction for later stages remains tested, not executed.

The tiny-overfit regression trains the actual small-config model on 64 fixed
debug examples, with a separate fixed limit of 1000 steps, and requires at least
99% unassisted complete-chain accuracy on that same training set. Label it an
overfit engineering regression, not held-out evidence. Use narrowed hidden
widths only in that explicitly debug profile; production model widths stay fixed.

- [ ] **Step 4: Run to GREEN**, including an interrupted/resumed real optimizer
  run and tests that a changed validation manifest/source/config prevents reuse.
- [ ] **Step 5: Commit** `git commit -m "feat: train and validate one-hop neural components"`.

### Task 11: Local Training Interface, Device Parity, and Import Boundaries

**Files:** Create `train/{cli,__main__,verification}.py`,
`docs/phase3-neural-components.md`, and
`tests/neural/{test_training_cli,test_device_parity,test_training_imports,test_training_package}.py`;
update new Phase 3 `train/config.py`, both `configs/train/{smoke,one_hop}.yaml`
and neural config/checkpoint/resume fixtures for the documented global epsilon
correction;
modify `Makefile` and `tests/integration/test_phase0_repository.py` only where
needed for actual local smoke/test commands. Existing root CLI and frozen source
remain unchanged.

Resolve the inherited pre-freeze numerical/accounting integration findings in
the new Phase 3 code with focused RED/GREEN tests: reject NaN/negative-infinite
real crossing inputs in `models/losses.py` while preserving valid dormant
positive infinity, and record executed functional confidence/focus/hypothesis
operations in `train/component_eval.py`. Valid-input objectives and every
numeric tolerance remain unchanged; retain the findings and verification
history for the final whole-phase review.

Apply the pre-freeze optimizer correction documented under Task 1, using TDD:
first assert both resolved profiles and the actual optimizer factory use
`epsilon=1e-7`, and assert explicit old `epsilon=1e-8` and `epsilon=1e-6` inputs
are rejected. Record
RED against the old configuration before changing its strict Literal and both
YAML values. Update checkpoint/resume fixtures to the same global value; preserve
all corruption checks. Re-run the original fixed-seed native parity test without
changing its batch, weights or comparisons, exact CPU/native MPS resume, the
fixed-64 tiny-overfit gate and the full local gate. Save new canonical hashes
and retain the original failing measurements. Later learned comparators must
share the declared optimizer stabilization, never receive a device-specific or
numerically weaker setup. No production run may consume old-configuration
checkpoints or claim the old debug learning result under the new hash.

**Interfaces:** `python -m silent_cascade.train` exposes exactly:

```text
fit --config PATH --validation-manifest PATH --run-dir PATH --expected-source-commit SHA [--resume PATH] [--device cpu|mps]
evaluate-components --weights PATH --expected-checkpoint-sha256 SHA --manifest PATH --output PATH
```

`--config` is repeatable and preserves order; `--set` uses the existing strict
override syntax. Both commands fail on unsupported profile/stage, frozen-test
input, missing/mismatched source/config/checkpoint, existing conflicting output,
unavailable explicitly requested MPS, or enabled fallback. JSON success/error
outputs use strict schemas and stable typed messages. No placeholder subcommand.

`train.verification` exposes
`measure_device_parity(model, batch) -> DeviceParityEvidence` and
`measure_cpu_resume(config, source_commit: str) -> ResumeEvidence` for the later
collector. They execute actual forward/backward/AdamW and save/load comparisons,
returning primitive maximum errors, equality/mismatch counts, tested hashes,
devices, versions, and tolerances. The tests exercise these same production
checks; the collector never infers numerical success from a copied log string.

- [ ] **Step 1: Write RED tests** for both real commands, resume, installed-wheel
  execution from unrelated CWD, absent optional packages, no network, and
  CPU/MPS parity of outputs, losses, gradients, and one optimizer step.

```python
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_neural_one_step_cpu_mps_agree(parity_run):
    cpu, mps = parity_run(seed=11)
    torch.testing.assert_close(cpu.loss, mps.loss.cpu(), rtol=1e-4, atol=1e-5)
    assert cpu.recalled_record_ids == mps.recalled_record_ids
    assert cpu.predicted_action_classes == mps.predicted_action_classes
    assert mps.output_dtype is torch.float32
    assert mps.output_device.type == "mps"
```

The parity fixture loads identical CPU-initialized weights onto both devices,
uses the same fixed public batch, verifies every named forward output and
loss term with `rtol=1e-4, atol=1e-5`, and gradients/updated weights with
`rtol=1e-3, atol=1e-5`. Include active/dormant guards, empty memory, and all
record/terminal types. Record maximum absolute/relative deviations. Do not
loosen these tolerances after viewing acceptance results; diagnose discrepancies.

- [ ] **Step 2: Run** `uv run pytest -q tests/neural/test_training_cli.py tests/neural/test_device_parity.py tests/neural/test_training_imports.py tests/neural/test_training_package.py` and record RED.
- [ ] **Step 3: Implement working adapters and documentation.** `__main__` imports
  only the new training CLI. New model/memory modules import neither `train`
  nor any `env` implementation (neutral runtime/schema/config imports are
  allowed only where explicitly needed). Loss-target types are neutral tensor
  data, not an import route into an oracle. A fresh subprocess blocks optional
  model libraries and socket access during actual training/evaluation smoke.

`train.cli` owns a Typer instance named `app` and the two complete command
implementations above. Keep `src/silent_cascade/train/__main__.py` this small:

```python
from silent_cascade.train.cli import app

if __name__ == "__main__":
    app()
```

Do not change the locked dependencies, root CLI, existing initializers, or
runtime doctor to make this work. Native MPS verification is local; a sandbox
that hides MPS is not proof the machine lacks it. Run the actual native gate
with the environment's permitted local access and no silent fallback.

Keep `make verify`'s three test-process structure. New `tests/neural` tests are
automatically collected by the main pytest process. If the tiny-overfit/device
tests materially contaminate the Phase 1 memory-sensitive processes, preserve
their existing separate children; do not relax resource ceilings or skip tests.
Extend `make smoke` with the small training CLI/import smoke only, not the
75,000-step trainer or 10,000-example acceptance evaluation. Retain the existing
set of Make targets; the explicit phase acceptance commands remain scripts.

Document complete config paths, output schemas, untimed component limitations,
checkpoint resume, and that root `silent-cascade train` belongs to Phase 4.
Installed commands require explicit external config paths; they must not assume
the current directory is this checkout. No README benchmark claim is added.

- [ ] **Step 4: Run to GREEN**, then native local `make verify`,
  `uv run silent-cascade doctor`, and `uv build`; capture actual exits.
  Assert original Phase 1/2 artifacts and source closure still verify.
- [ ] **Step 5: Commit** `git commit -m "feat: expose local neural training workflow"`.

### Task 12: Source-Bound Component Evidence and Independent Verification

**Files:** Create `train/{provenance,evidence_types,evidence}.py`,
`scripts/check_phase3_components.py`, `scripts/verify_phase3_gate_artifact.py`, and
`tests/neural/{test_phase3_provenance,test_phase3_evidence,test_phase3_verifier}.py`;
update new `train/verification.py` and `tests/neural/test_device_parity.py` to
derive empty-memory coverage from actual probe eligibility and expose measured
native MPS resume as part of evidence integration;
align the new `train/curriculum_data.py` manifest-read constant with this task's
32 MiB manifest limit (the trainer already imports the same constant);
modify `tests/integration/test_phase0_repository.py` for a new Phase 3 delivery
state check, without altering Phase 1/2 accepted rows or tests.

**Interfaces:**

- `Phase3EvidenceProvenance` binds full source revision, approved plan revision,
  spec/plan hashes, a literal sorted source closure, config, parent generator and
  transform versions, manifest hash, seeds, Python/Torch/device metadata,
  checkpoint hashes, and exact zero foundation-model calls.
- `freeze_component_manifest(config: ResolvedConfig[Phase3Config], *, source_commit: str, plan_revision: str, output_path: Path) -> ComponentManifest`: immutable recipe for
  10,000 independent transformed validation examples, exactly
  5,000 positive / 2,500 safe / 2,500 disconnected, hashes/coordinates/attempts
  sufficient to regenerate every public input and private scoring target.
- `collect_component_gate(config: ResolvedConfig[Phase3Config], *, manifest_path: Path, weights_path: Path, expected_checkpoint_sha256: str, training_run: Path, source_commit: str, plan_revision: str, output_path: Path) -> ComponentGateReport` evaluates real exported
  weights on all fixed rows, captures native parity/resume checks and raw
  predictions, and derives pass state. No injected success arrays or synthetic
  production-report shortcut in the collector.
- `verify_phase3_gate_artifact` independently rederives all metrics and
  authenticates historical inputs without training; its exact keyword-only
  signature is specified below in this task.
- `measure_mps_resume(config: ResolvedConfig[Phase3Config], source_commit: str)
  -> MPSResumeEvidence` in `train/verification.py` performs the existing native
  two-update checkpoint/RNG continuation check and returns primitive measured
  comparisons under `phase3-mps-resume-v1`. Preserve the existing CPU API/schema
  and its exact comparisons. Shared report schemas live in the data-only
  `train/evidence_types.py`, with no producer metric, pass, or hash-chain code.

- [ ] **Step 1: Write RED tests** for debug-size real collection, production/debug
  separation, source closure (including new ancestor initializers), seed/manifest
  binding, checkpoint mismatch, raw-denominator arithmetic, wrong/extra/missing
  predictions, omitted variants, changed thresholds, source mutation, deep JSON,
  no-clobber publication, and preserved prior-phase evidence.

Include the inherited Task 11 coverage-hardening regression before source
freeze: a deliberately nonempty replacement for the isolated empty-memory
probe must not yield positive `empty_memory_rows` or a passing coverage gate.
Derive the paired-device count from actual probe eligibility, not literal `1`;
preserve the existing genuine empty fixture, all numerical comparisons and
the public evidence schema. The current Task 11 probe was independently
verified genuinely empty, so its recorded result remains valid. This small
integration correction changes no model/training computation; its cost is one
additional focused evidence regression. Retain its review history for the final
whole-phase audit rather than adding a separate Task 11 fix loop.

```python
def test_independent_component_verifier_rejects_changed_success_count(
    debug_component_gate, tmp_path
):
    import json
    from runpy import run_path
    from silent_cascade.errors import ArtifactIntegrityError
    verify = run_path("scripts/verify_phase3_gate_artifact.py")["verify_phase3_gate_artifact"]
    raw = debug_component_gate.report.model_dump(mode="json")
    raw["summary"]["complete_chain_correct"] += 1
    path = tmp_path / "corrupted.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError):
        verify(artifact_path=path, manifest_path=debug_component_gate.manifest_path,
               repo_root=debug_component_gate.repo_root, allow_debug=True)
```

The debug fixture holds a real 16-example report, manifest path, and clean
temporary source repository. Production serialization uses validated producer
models; this test corrupts raw JSON bytes and exercises the public independent
verifier, with no production mutation helper.

- [ ] **Step 2: Run** `uv run pytest -q tests/neural/test_phase3_provenance.py tests/neural/test_phase3_evidence.py tests/neural/test_phase3_verifier.py` and record RED.
- [ ] **Step 3: Implement the complete evidence boundary.**

Use a literal Phase 3 source tuple that includes the actually executed new
modules/scripts/configs and all transitive local dependencies/initializers,
without editing historical Phase 1/2 tuples. Authenticate the executing package
against the named clean checkout, not a different clean repository. The new
canonical config hash includes frozen base/data/runtime plus new model/train
overlays. Serialize canonical integer-key allocation maps correctly; do not
repeat Phase 2's JSON-string/integer-key trap or call a producer loss helper in
the independent verifier.

Define `phase3-component-manifest-v1`, `phase3-component-gate-v1`, and
`phase3-component-verification-v1`. Production gates require exactly 10,000 rows;
debug gates use 16 and are explicitly non-acceptance. Bound manifests/gates at
32 MiB and metadata at 8 MiB; parse exact primitive types and reject duplicate
keys, nonfinite JSON numbers, noncanonical hashes, wrong schemas, or recursion
overflow with typed errors.

The minimal Task 10 manifest loader initially used an 8,000,000-byte bound.
Set its shared `MAX_COMPONENT_MANIFEST_BYTES` to `32 * 1024 * 1024` and test
the manifest read boundary. This aligns existing new-phase consumers with the
declared contract; it does not increase the separate metadata bound or change
generator/transform semantics. Cost: a larger, still strictly bounded accepted
manifest input, with no historical Phase 1/2 IO helper change.

Each raw episode row contains its public ID/example hash, corpus position,
private target record/role/focus/class/status sequence, actual predicted
sequence, required/correct recall/composition counts, final content outcome,
cap/invalid-prediction flags, named regression errors, neural compute counters,
and model/config/source/data identity. The predictions must be recorded before
scoring; every failed example stays in the denominator. Store final-state or
output hashes for three distinct replay samples, one per variant. Raw rows and
their framed ordered hash chain, not copied aggregate values, determine totals.
The top-level `summary` contains required/correct recall counts, required/correct
composition counts, `complete_chain_correct`, total episode count, per-variant
counts, cap/invalid counts, and derived ratios. The verifier reconstructs every
field from rows; redundant stored pass flags are checked, never trusted.

The independent verifier shares closed **data-only** schemas/safe reads but not
producer metric arithmetic, pass validators, or hash-chain implementation. It
checks all counts/denominators, strict `>99%` thresholds, exact variant totals,
checkpoint/source/config/manifest hashes, finite gradient/numeric evidence,
and zero foundation calls. Selected weights remain in the explicitly selected
local run directory, outside version control; raw artifact
verification binds their SHA, while a separate execution check loads those
weights and reproduces all raw component decisions on CPU. Do not claim that
arithmetic/hash verification alone proves the training computation occurred.

The run directory may be on a persistent local data volume rather than beneath
the checkout. Accept an explicit owned canonical real directory, retain all
descriptor-relative archive checks, and reject symlink traversal. This is an
execution-location choice, not a new storage backend. Commit portable run IDs,
relative archive names and hashes; keep the absolute machine-specific location
in ignored local execution records. Optional `--weights` supplies the physical
archive path for verification. Do not infer or silently fall back to a different
run location. Source/configuration/corpus remain repository-bound.

After independently reconstructing correct/required counts from the recorded
sequences, use exact integer arithmetic for each acceptance threshold:

```python
def _above_99_percent(correct: int, required: int) -> bool:
    if not 0 <= correct <= required or required == 0:
        return False
    return 100 * correct > 99 * required
```

Strict primitive validation rejects booleans before this calculation. Keep the
verifier's arithmetic local; do not import this kernel from the producer.

The collector adapter exposes real modes `freeze-validation` and `collect`:

```text
check_phase3_components.py freeze-validation --config PATH (repeatable)
    --expected-source-commit SHA --expected-plan-base-revision SHA --output PATH
check_phase3_components.py collect --config PATH (repeatable) --manifest PATH
    --weights PATH --expected-checkpoint-sha256 SHA --training-run PATH
    --expected-source-commit SHA --expected-plan-base-revision SHA --output PATH
verify_phase3_gate_artifact.py --artifact PATH --manifest PATH
    [--weights PATH] [--expected-source-commit SHA]
```

`--weights` on the verifier checks archive byte identity/schema; it does not
invoke the producer or model. The separate `evaluate-components` command is the
actual execution reproduction path. `collect` checks training-run progress,
source/config, validation selection history, checkpoint identity, real numeric
test measurements, and exact CPU resume evidence before publication.

Collect actual native MPS resume measurements as well: save after the first
real optimizer update, compare the uninterrupted and restored second update,
and require the existing native test's `rtol=1e-4, atol=1e-5` for named losses,
parameters, and AdamW state. Next-batch hashes and Python/NumPy/Torch CPU/MPS
random draws must match exactly; restore the caller's RNG afterward. Record
source/config/archive identity and native device/version/thread metadata.
Unavailable MPS cannot satisfy production acceptance. The old passing test log
is not a replacement for these report measurements. Focused tests cover this
adapter without weakening or removing the existing continuation regression.
This pre-freeze integration correction adds one data-only schema module and a
measured adapter for an already-required check; its cost is a slightly larger
evidence/test surface, not a new scientific requirement or changed tolerance.

The public Python verifier signature is
`verify_phase3_gate_artifact(*, artifact_path: Path, manifest_path: Path,
repo_root: Path, weights_path: Path|None = None, expected_source_commit:
str|None = None, allow_debug: bool = False) -> Phase3VerificationResult`.
CLI production verification never enables `allow_debug`.

Validation manifest creation precedes training and fixes keys/order/denominators.
The source may acquire this new data artifact and later evidence/docs commits,
but its executable closure must remain byte-identical. Freeze refuses overwrite;
after a necessary reviewed source correction use a new experiment/run identity
and regenerate incompatible artifacts, never append to a completed run.

The new corpus does not exist at the earlier implementation source commit.
Authenticate its introduction with a separate `validation_manifest_commit`
(a descendant of the reviewed source and ancestor of the evidence commit),
and compare its regular Git blob and file hash there. Do not require a manifest
blob at a commit predating its creation. The manifest records its creator source;
the training command separately receives its reviewed model source explicitly.

The repository delivery regression accepts either (a) Phase 3 proposed/in
progress and no acceptance gate, or (b) a regular verified gate plus exact
source/artifact/checkpoint-bound completion text. A present corpus alone is not
completion. Pin the Phase 2 artifact SHA/source/complete row, analogous to the
existing frozen Phase 1 baseline, so coordinated substitutions cannot pass.

- [ ] **Step 4: Run to GREEN**, including a verifier test that makes every producer
  arithmetic helper fail if invoked. Run full native `make verify` and both old
  phase artifact verifiers. Source review must finish before Task 13 execution.
- [ ] **Step 5: Commit** `git commit -m "test: define source-bound neural component gate"`.

### Task 13: Execute, Verify, and Commit the Actual Phase 3 Gate

**Files:** Create the two Task 13 manifest/gate files; modify only the Phase 3
delivery row in `docs/PLAN.md` and append measured results to
`docs/phase3-neural-components.md`. Model checkpoints and full training logs
stay in the explicitly selected local run directory, outside version control;
do not commit normal checkpoints or machine-specific storage paths.

This task changes **no implementation, tests, configuration, spec, or approved
plan**. A source correction returns to a reviewed source task and invalidates
incompatible Phase 3 evidence. Preserve all Phase 1/2 artifacts and source.

- [ ] **Step 1: Verify and capture a clean reviewed source.**

```bash
make verify
git status --short
phase3_source_commit="$(git rev-parse HEAD)"
phase3_plan_revision="$(git log -1 --format=%H -- docs/superpowers/plans/2026-09-09-phase-3-neural-components.md)"
git merge-base --is-ancestor "$phase3_plan_revision" "$phase3_source_commit"
test -z "$(git status --porcelain)"
phase3_run_dir="$(uv run --offline python -c 'from pathlib import Path; import shutil; root = Path.home() / "Library" / "Application Support"; assert shutil.disk_usage(root).free >= 20 * 1024**3, "Phase 3 requires 20 GiB of free artifact space"; print(root / "silent-cascade" / "runs" / "phase3-components" / "event_flow" / "11" / "one-hop-v1")')"
```

The repository volume has only about 5.4 GiB free at the preflight. One actual
disposable production-shaped step serializes to about 184 KiB; all 75,000 raw
step logs can require about 13 GiB before validation/checkpoint artifacts. A
lossless gzip probe still projected about 4.3 GiB for steps alone. Use the
explicit persistent main-volume path above (about 214 GiB free at preflight),
with private owner-only directories and at least 20 GiB free. Preserve all raw
logs, the full workload and existing storage format. Do not delete unrelated
files, use a symlink, or rely on early convergence to fit the smaller volume.
Recheck available space during execution. Cost: artifacts live separately from
the source checkout and require their explicit path for backup/replay.

At execution start, after actual user approval, update this plan's status to
`User-approved; completion tracked in docs/PLAN.md` and commit that approval
record before implementing source. Do not invent approval in this planning turn.

- [ ] **Step 2: Create the immutable one-hop validation corpus before training.**

```bash
uv run --offline python scripts/check_phase3_components.py freeze-validation \
  --config configs/base.yaml --config configs/data/primary.yaml \
  --config configs/model/event_flow.yaml --config configs/model/neural_components.yaml \
  --config configs/train/one_hop.yaml \
  --expected-source-commit "$phase3_source_commit" \
  --expected-plan-base-revision "$phase3_plan_revision" \
  --output manifests/validation/phase3/one-hop-10000.json
git add manifests/validation/phase3/one-hop-10000.json
git commit -m "test: freeze one-hop component validation corpus"
```

Assert exactly 10,000 regenerated rows, fixed 5000/2500/2500 allocation, and no
train/validation key/public-ID overlap. This is a curriculum validation corpus,
not final frozen-test data and not a replacement for the Phase 1 manifest.

- [ ] **Step 3: Train one model seed locally with the declared workload.**

```bash
OMP_NUM_THREADS=1 uv run --offline python -m silent_cascade.train fit \
  --config configs/base.yaml --config configs/data/primary.yaml \
  --config configs/model/event_flow.yaml --config configs/model/neural_components.yaml \
  --config configs/train/one_hop.yaml \
  --validation-manifest manifests/validation/phase3/one-hop-10000.json \
  --expected-source-commit "$phase3_source_commit" \
  --run-dir "$phase3_run_dir" --device cpu
```

Use the measured faster CPU placement with one Torch CPU thread, and recheck
throughput briefly on the final reviewed source before the full run. Independent
production-model/B128 measurements during Task 10 were approximately 0.25s per
full CPU optimizer step versus 1.28s on native MPS, excluding generation,
validation and checkpoint writing. This execution choice overrides the ordinary
available-MPS preference without changing model/data/optimizer configuration.
Keep native MPS parity/resume gates mandatory and forbid silent fallback; use
the same explicit thread setting when resuming the CPU training checkpoint.
Write a source-
and config-bound run manifest, train metrics, validation raw metrics, checkpoint
index, and resume state. Checkpoint identity uses the captured reviewed source
even when HEAD additionally contains the immutable corpus commit; verify exact
closure equality, not merely ancestry. A run interrupted by reboot resumes from
its validated latest checkpoint and next batch counter, never silently starts
over or selects a different validation corpus.

On failure to exceed all three thresholds, publish an honest non-passing
engineering run summary and continue only with documented in-scope diagnosis.
Do not mark Phase 3 complete, start Phase 4, or weaken its gate. Architecture or
loss changes require reviewed source/config changes and a separately identified
run with its attempted configuration recorded.

- [ ] **Step 4: Export and evaluate the selected checkpoint on CPU.**

The successful run manifest records exact `selected_weights_path` and
`selected_weights_sha256`. Read those validated fields into task-specific shell
variables `phase3_weights_path` and `phase3_weights_sha256`; validate the path is
an owned regular archive beneath the run directory before using it. Do not use
globs, “latest file”, or a checkpoint selected from a test result.

```bash
uv run --offline python -m silent_cascade.train evaluate-components \
  --weights "$phase3_weights_path" \
  --expected-checkpoint-sha256 "$phase3_weights_sha256" \
  --manifest manifests/validation/phase3/one-hop-10000.json \
  --output "$phase3_run_dir/cpu-evaluation.json"
```

The CPU result must independently exceed all three thresholds. Run it twice
through distinct no-clobber output paths and compare ordered prediction hashes,
raw decisions, and metrics exactly. Require all 10,000 rows and all failures,
not just three favorable examples. Also execute the real CPU resume and native
CPU/MPS parity checks and retain named measurements/commands/exit statuses.

- [ ] **Step 5: Collect and independently verify the artifact.**

```bash
uv run --offline python scripts/check_phase3_components.py collect \
  --config configs/base.yaml --config configs/data/primary.yaml \
  --config configs/model/event_flow.yaml --config configs/model/neural_components.yaml \
  --config configs/train/one_hop.yaml \
  --manifest manifests/validation/phase3/one-hop-10000.json \
  --weights "$phase3_weights_path" --expected-checkpoint-sha256 "$phase3_weights_sha256" \
  --training-run "$phase3_run_dir" \
  --expected-source-commit "$phase3_source_commit" \
  --expected-plan-base-revision "$phase3_plan_revision" \
  --output manifests/validation/phase3/component-gate.json
uv run --offline python scripts/verify_phase3_gate_artifact.py \
  --artifact manifests/validation/phase3/component-gate.json \
  --manifest manifests/validation/phase3/one-hop-10000.json \
  --weights "$phase3_weights_path" --expected-source-commit "$phase3_source_commit"
```

Artifact pass requires measured parameter count <=5M including entity table,
strict >99% retrieval/composition/complete-chain accuracy, finite training
gradients, passed CPU/MPS parity on this M3 Max, exact CPU resume, identical
repeated CPU predictions, import/offline boundary checks, and exactly zero
foundation-model calls. Missing or unsupported native MPS evidence on this
target is not a pass. Compact raw rows are retained for every episode; full
latent traces obey the 500-sample-plus-failures bound.

- [ ] **Step 6: Update documentation from verified values and commit evidence.**

Record source/artifact/checkpoint hashes, all three raw numerator/denominator
pairs, every variant, actual parameter count, numeric tolerances/results,
training stop reason, failures, and exact reproduction commands. The Phase 3
delivery row must state **neural component engineering evidence, not autonomous
timed OFD or comparative benchmark evidence**. The detailed document explains
the one-hop transform, unassisted untimed harness, teacher-forced training, and
remaining Phase 4 obligations. Do not change either earlier delivery row.

```bash
make verify
git diff --check
git add manifests/validation/phase3/component-gate.json docs/PLAN.md docs/phase3-neural-components.md
git commit -m "test: freeze verified Phase 3 component evidence"
make verify
uv run --offline python scripts/verify_phase3_gate_artifact.py \
  --artifact manifests/validation/phase3/component-gate.json \
  --manifest manifests/validation/phase3/one-hop-10000.json \
  --weights "$phase3_weights_path" --expected-source-commit "$phase3_source_commit"
git status --short
```

Capture actual process exits and durable logs, with one live process per
expensive command. Do not duplicate a run after an output timeout. Final
independent review covers specification compliance, public/teacher information
boundaries, numerical/autograd behavior, evidence integrity, and every deferred
task finding. Correct source findings through TDD/re-review and regenerate
affected Phase 3 evidence; never leave a stale artifact marked complete.

## 5. Phase 3 Completion Checklist

- [ ] Existing Phase 1/2 source closures, artifacts, complete rows, and replay
  fixtures remain unchanged and independently verify.
- [ ] Every new production module has a direct tested consumer; there are no
  placeholder agents, unused framework hooks, documented stubs, or hosted workflows.
- [ ] Actual model size is <=5M including the shared `64x32` entity table.
- [ ] Memory encoding and scores are permutation-equivariant; preview is
  invariant, non-mutating, finite for empty memory, and counted on each call.
- [ ] No exact-subject retrieval filter, hidden path, future fact, teacher timing,
  label, stage ID, or private horizon enters a prediction call.
- [ ] Batched flow/guards agree with the completed scalar reference on CPU/MPS,
  gradients stay finite, state remains bounded, and dormant math is never evaluated.
- [ ] Training retains gradients across complete traces; padding/quiescence are
  excluded before invalid operations; no fake NOOP or detached hidden recurrence.
- [ ] Teacher-forced diagnostics are clearly separated from unassisted component
  accuracy and from the not-yet-implemented autonomous timed benchmark.
- [ ] Real fixed-64-example overfit passes; real 10,000-example component
  validation exceeds all three 99% gates with full, unconditional denominators.
- [ ] Native CPU/MPS one-step parity, CPU exact resume, MPS resume tolerance,
  safe weight export, and repeated full CPU component predictions pass.
- [ ] Every loss/compute count, selected checkpoint, attempted configuration,
  validation row, and failure is bound to reproducible source/config/data identity.
- [ ] Full local `make verify`, doctor, build, installed-package smoke, offline
  checks, independent artifact verification, and final reviews pass.
- [ ] Worktree clean; meaningful commits on the current branch; zero foundation
  calls; no Phase 4 implementation or frozen-test generation started.

## 6. Scope Coverage and Explicit Deferrals

| Specification requirement | Ownership |
|---|---|
| 5.2–5.5 dimensions, bounded flow/guards | Existing Phase 2; batched neural realization Tasks 1/4 |
| 5.7–5.9 typed learned jumps and no oracle scaffolding | Tasks 3–5/7/10; actual event-engine integration Phase 4 |
| 5.10–5.12 tensor memory, eviction, retrieval/preview, model budget | Tasks 1–5 |
| 8.1–8.4 teacher traces, timing, losses | Tasks 6–8 |
| 8.5 curriculum | Tasks 6/10; autonomous two-hop and later promotion gates Phase 4 |
| 8.6–8.7 optimizer, counter-based batches, training checkpoints | Tasks 6/9/10 |
| 9.4 compute instrumentation | Task 5; competitive matching and complete baselines Phase 5 |
| 12.1/12.3/12.5 neural numerics, memory edge cases, training resume | Tasks 2/4/9/11 |
| 13 CPU/MPS/dependency/resource policy | All tasks, verified Tasks 5/11–13 |
| 15 working training workflow | Task 11 phase module; root CLI integration Phase 4 |
| 16 provenance/raw artifacts | Tasks 9/12/13; scientific report generator Phase 6 |
| 17 Phase 3 gate and 18 neural tests | Tasks 8–13 |
| 17 Phase 4 timed pilot and zero-failure two-hop runtime | Separate next approved plan |
| 8.8, 9 baseline schedules, 10–11 scientific statistics/causality | Phases 5–6; no substitution by component results |
| 19–20 showcase/Qwen/public result assets | Phase 7; no optional model in this phase |

Phase 4 must consume the actual trained-core APIs and separately plan neural
runtime-state/embedding persistence, learned guard scheduling, engine adapter,
learned checkpoint/replay registration, trace/crash integration, the root CLI
and historical-evidence transition, autonomous two-hop gate, and one-seed IID
pilot. Neither a high teacher-forced score nor this untimed one-hop component
gate pre-satisfies any of those requirements.

## 7. Task 13 Corrective Return — 2026-09-15

This is the approved plan's source-correction path, exercised under the user's
standing approval for reversible, in-scope plan/configuration corrections. Task
13 is **not complete**. Its first production run naturally stopped at 25,000
updates with no passing validation. The selected step 10,000 yielded recall
17,364/17,500, composition 15,788/17,500, and complete content chains
8,423/10,000. Two exported-weight CPU evaluations reproduced every prediction.
The collector wrote `passed=false`; the independent verifier exited 1 because
three production-B128 updated weights failed the unchanged CPU/MPS tolerance.
This is rejected engineering evidence, not an accepted negative gate.

Source corrections must be separately reviewed before another Task 13 fit.
Keep all scientific thresholds, validation examples, public evaluator semantics,
teacher-timed EventFlow behavior, update ceilings, prior-phase pins and the
canonical specification unchanged. No Phase 4 work is authorized by this return.

### Return A: Preserve the Actual Failed Attempt

**Files:** Relocate only the generated, untracked
`manifests/validation/phase3/component-gate.json` to
`manifests/validation/phase3/attempts/one-hop-v1/component-gate.json`; create
`manifests/validation/phase3/attempts/one-hop-v1/attempt.json`; append measured
history to `docs/phase3-neural-components.md` and `docs/deviations.md`.
Do not change source, tests, configurations, the original validation manifest,
or either prior-phase delivery row. Leave Phase 3 In progress.

**Interfaces:** This is archival data, not a new acceptance schema or verifier.
The attempt index binds the exact artifact, original manifest, source/config,
selected checkpoint and actual process exits. Its consumers are the history
document and subsequent delivery review; it must not claim `valid=true`.

- [ ] **Step 1:** Before touching the artifact, verify it is an owned regular
  non-symlink file of 17,635,868 bytes with SHA-256
  `b4e04bb7b742f2bd3564e7538150d16dacb50382fdd840d6afe23f80b77396c2`.
  Verify the exact archive destination does not exist and its parent chain has
  no symlinks. Preserve the original corpus at its current committed path.
- [ ] **Step 2:** Move that one generated file recoverably to the archive path,
  without modifying its bytes or embedded provenance. Recheck the same length
  and SHA-256 at the destination and absence at the former canonical path.
  No run log, checkpoint, diagnostic or corpus is deleted or overwritten.
- [ ] **Step 3:** Write the small attempt index through `apply_patch`, with
  `status: rejected`, `phase_complete: false`, source
  `5a96b673f67a634d268f000a240e0de0f3dca034`, config
  `26b21c1acd79e6c47375f03c4e68da24fadddde9ae150454bbd3db7f50237041`,
  corpus SHA-256
  `5f57796f0798e9b494e47663f25e89c5fa0ebb33b79bac7601b77e4b969a2e32`,
  selected weights SHA-256
  `26bb303619d694f6a8329cc7ff06dc796410193d788f5f3fc73813ad7cea73d9`,
  selected training step 10,000, completed updates 25,000, stop reason
  `early_stopping`, all three raw metric pairs, and verifier exit 1/error
  `artifact_integrity_error: numerical comparison failed or empty`.
  Record fit/evaluation/evaluation-repeat/collector exits 0 separately from
  acceptance. Include portable repository-relative paths and logical run ID
  `phase3-components/event_flow/11/one-hop-v1`; no machine-specific paths.
- [ ] **Step 4:** Document all 25 nonpassing scheduled validations, per-variant
  selected results, secondary untimed action 4,775/10,000, repeat equality,
  original optimizer history, and the measured diagnostic limits. Explicitly
  distinguish the latent-context exposure hypothesis from a proven model-code
  defect, and numerical near-cancellation from an incorrect AdamW formula.
- [ ] **Step 5:** Run the existing repository delivery tests locally and check
  exact archive/corpus hashes and `git diff --check`. Do not change the test
  that requires any designated acceptance artifact to pass independently.
  Commit only the archive/index and truthful history docs with message
  `docs: preserve rejected Phase 3 component attempt`.

The archival ruling changes location, not evidence: the original canonical
filename denotes acceptance to the repository checker and cannot honestly hold
this rejected attempt. The cost is a separate historical path that reproduction
instructions must name explicitly. A corrected run will use distinct config,
manifest, gate and run identities; never overwrite or relabel these v1 bytes.

### Return B: Predeclare the Fixed Numerical Candidate

**Scope:** A disposable local diagnostic, not another production fit and not
accepted parity evidence. The prior native-B128 diagnosis reproduced all three
updated-weight failures and traced them to near-cancelled guard gradients
amplified by AdamW. It did not demonstrate an incorrect optimizer formula.

Reassess **only epsilon `1e-6`**, previously tested at B8, on the original timed
objective with the production architecture, seed 11 and original counter-0
batch of 128. Keep lr `3e-4`, weight decay `1e-4`, betas `(0.9,0.999)`,
clip norm 1, `foreach=False`, `fused=False`, native float32 and all comparison
tolerances unchanged. This is one named candidate, not a sweep or a fourth
epsilon. Its previous 63/64 fixed-set learning failure remains recorded.

**Files:** Only an ignored diagnostic helper/report/result inside this plan's
SDD workspace. No source/config/test/manifest/checkpoint/run bytes change.

- [ ] **Step 1:** Authenticate the same 68 source files, original model state
  `5053ec7272f59e0d65c16e14c007c8ddf9c31c40a713269589efcb78f904502d`,
  and all 128 ordered batch hashes against the retained failed evidence.
- [ ] **Step 2:** Reuse the exact previously reproduced timed parity operation.
  Its only controlled experimental change is constructing the actual AdamW
  optimizer with epsilon `1e-6` on both CPU and MPS. Preserve and restore any
  narrowly scoped observational wrapper; do not monkeypatch tensor results,
  tolerance arithmetic, loss, batch or seed. Record that this disposable probe
  does **not** use the current production config's epsilon `1e-7` and cannot
  stand in for later source-bound configured training-path parity.
- [ ] **Step 3:** Execute exactly one fresh CPU/MPS first-step comparison.
  Retain every raw comparison and process exit, optimizer settings, initial
  model/batch/source identity and discrepancy coordinates. No long fit,
  checkpoint selection, historical artifact mutation or network use.
- [ ] **Step 4:** If any unchanged comparison fails, stop this candidate and
  return to bounded guard-subterm/backend investigation. Do not pick another
  epsilon. If all pass, carry this evidence into the reviewed corrected-recipe
  implementation; actual combined-objective B128/B8 parity and fixed-64
  learning must still pass before production training.

This ruling revisits the earlier bounded adjustment after independent
latent-exposure and production-B128 diagnoses. It does not erase either prior
failure. Cost if wrong: small-gradient learning can slow and the corrected
objective can introduce new cancellations; both require fresh unchanged gates.

### Return C: Expose the Actual Untimed Content Context During Training

**Goal:** Test the diagnosed latent-context exposure mismatch without replacing
the timed EventFlow objective or changing the public component evaluator.

**Design ruling:** Every corrected optimizer update uses the original batch
twice: the unchanged full teacher-timed unroll and a fresh, differentiable
teacher-forced content unroll with no internal world-time advance. Sum their
losses with fixed auxiliary coefficient 1.0, then backward/clip/update once.
Alternating objectives would halve timed updates under the same ceiling;
adding a new no-match architecture is not justified by the current evidence.
Cost if wrong: the second unroll adds real compute and its gradients may compete
with timed learning. No convergence claim follows from choosing this design.

Return C implements the independently testable auxiliary unroll/loss only;
Return D supplies its real training/evidence consumer before another release
gate. Do not wire it into an incomplete production recipe in this subtask.

**Files:** Create `src/silent_cascade/models/content_types.py`,
`src/silent_cascade/models/content_loss.py`, and
`src/silent_cascade/train/content_unroll.py`; create
`tests/neural/test_content_loss.py` and `tests/neural/test_content_unroll.py`.
Keep `event_flow_loss`, `teacher_forced_unroll`, `predict_components`, scoring,
model architecture, generator and schemas unchanged. Reuse existing functional
context/observation APIs. Do not call the timed `_apply_teacher`, which injects
teacher clocks and normalized deadlines.

**Interfaces:**

```python
# models/content_types.py; neutral tensors, no train/env imports
@dataclass(frozen=True, slots=True)
class ContentLossInputs:
    boundary_mask: torch.Tensor          # bool[B,T], content only
    recall_mask: torch.Tensor            # bool[B,T]
    raw_recall_target: torch.Tensor      # float32[B,T], raw controller A
    retrieval_logits: torch.Tensor      # float32[B,T,64]
    retrieval_eligible_mask: torch.Tensor  # bool[B,T,64]
    retrieval_target: torch.Tensor       # int64[B,T]
    role_logits: torch.Tensor            # float32[B,T,5]
    status_logits: torch.Tensor          # float32[B,T,3]
    confidence_logit: torch.Tensor       # float32[B,T]
    append_support_logit: torch.Tensor   # float32[B,T]
    continue_search_logit: torch.Tensor  # float32[B,T]
    next_focus_logits: torch.Tensor      # float32[B,T,64]
    hazard_logits: torch.Tensor          # float32[B,T,4]
    log_delay: torch.Tensor              # float32[B,T]
    # Same-shaped *_target and bool *_mask fields for role, status,
    # confidence, append_support, continue_search, focus, hazard, log_delay.
    # Integer labels: role/status/focus/hazard. Other targets: float32.

# models/content_loss.py
def content_loss(inputs: ContentLossInputs) -> LossBreakdown: ...

# train/content_unroll.py
@dataclass(frozen=True, slots=True)
class ContentOperation:
    prediction_context: ModelContext
    post_context: ModelContext
    row_mask: torch.Tensor
    kind: torch.Tensor
    raw_recall_target: torch.Tensor
    retrieval: RetrievalScores
    composition: ComposePredictions

@dataclass(frozen=True, slots=True)
class ContentUnrollResult:
    observations: tuple[Boundary, ...]
    steps: tuple[ContentOperation, ...]
    final_context: ModelContext
    targets: TeacherTargets
    compute: NeuralComputeSnapshot
    diagnostic_label: str = "teacher-forced-untimed-content"
    def loss_inputs(self) -> ContentLossInputs: ...

def teacher_forced_content_unroll(
    model: EventFlowModel, batch: TrainingBatch
) -> ContentUnrollResult: ...
```

The comment-expanded target fields above are an exact Cartesian contract, not
an optional dictionary: `role_target`, `status_target`, `confidence_target`,
`append_support_target`, `continue_search_target`, `focus_target`,
`hazard_target`, `log_delay_target`, and the eight corresponding `_mask` fields.
All inputs use the existing bounded B<=128 and T<=11 layout; ACT/padding rows
are excluded. The one-hop workload contains at most four content transitions.

- [ ] **Step 1 — RED, loss:** Write hand-derived mixed-role fixtures using real
  tensor reducers. For one active RECALL target A=0.8, assert the availability
  numerator is 0.3 and denominator 1. For equal two-class eligible retrieval
  logits, assert CE `log(2)`. Reject an ineligible target even if its logit is
  large; verify invalid padding is selected away before arithmetic. Independently
  assert every term's numerator, denominator, mask and per-position shape.
  Empty masks must yield graph-connected zero, not NaN or detached constants.

  ```python
  loss = content_loss(fixture)
  assert loss.terms["recall_available"].item() == pytest.approx(0.3)
  assert loss.denominators["recall_available"].item() == 1
  assert loss.terms["retrieval"].item() == pytest.approx(math.log(2))
  loss.total.backward()
  assert fixture.raw_recall_target.grad[0, 0].item() == pytest.approx(-1.0)
  ```

- [ ] **Step 2 — RED, trajectory:** Use the existing real seeded model,
  curriculum examples and packer. Hook observation flow separately from
  internal computation: external observations retain their actual flow; every
  internal `prediction_context.time_features` equals that row's post-activation
  features. Assert no internal flow/crossing/ACT evaluation. Changing a future
  teacher record/label cannot change an earlier prediction. Check all three
  variants, ragged content lengths, no calls for quiescent/padded rows, terminal
  exclusion, correct active-record lifetime and consumed-record eligibility.

  ```python
  result = teacher_forced_content_unroll(model, batch)
  activation = result.observations[-1].context.time_features
  for step in result.steps:
      torch.testing.assert_close(
          step.prediction_context.time_features[step.row_mask],
          activation[step.row_mask], rtol=0, atol=0,
      )
  # Capture actual calls, not a mock returning the desired trajectory.
  assert internal_flow_calls == 0
  assert action_calls == 0
  ```

- [ ] **Step 3 — Run RED:** Run only the two new focused test modules with
  `OMP_NUM_THREADS=1 UV_OFFLINE=1 uv run --offline pytest -q`. Record the actual
  missing-behavior failures before implementation; fix fixture errors first.

- [ ] **Step 4 — Minimal implementation:** Start from a fresh `observe_public`
  graph. Before each required RECALL, call `preview_and_control`, retain its raw
  A and score retrieval before disclosing the teacher-selected slot. Install
  that slot/HAVE_MEMORY only after prediction, apply the existing RECALL jump,
  then predict COMPOSE immediately with no intervening flow/controller tick.
  After prediction, apply teacher content/focus/support/continuation state,
  retain active record through the COMPOSE jump, clear it afterward and mask
  it consumed. Use functional gather/scatter; preserve the full gradient graph.
  Keep actual public observation time but fixed activation-time internal clocks.
  Teacher hazard delay may enter a terminal hypothesis only after prediction,
  as an activation-relative delay, never the teacher-timed normalized deadline.
  No further auxiliary prediction uses a terminal timing register.

  Auxiliary loss is exactly:

  ```python
  total = (
      recall_available + retrieval + compose_type
      + 0.5 * focus + hazard + 0.125 * log_delay
  )
  ```

  `recall_available = mean(relu(1.10 - raw_A_recall))` over required RECALLs.
  Retrieval is eligible-slot CE. `compose_type` equally averages the existing
  five masked role/status/confidence/append/continue subterm means. Focus and
  hazard are their masked CE; delay is masked Smooth L1 on `log1p(delay)`.
  The 0.125 delay coefficient retains the original 0.25 two-component deadline
  mean's delay contribution. Do not add timed crossing/race, normalized-deadline,
  ACT, inactive-guard, final-dormancy, state or event-cost terms to this branch.
  They remain in the original timed objective. Do not fabricate a SEARCHING
  no-match target: the diagnosed explicit disconnected continue=false target
  is the minimal supported supervision, and legal public dormancy stays valid.

- [ ] **Step 5 — GREEN and interface checks:** Run the two new tests and existing
  unroll/loss/gradient/import-boundary tests. Demonstrate gradients reach the
  early encoder and RECALL/COMPOSE jumps, and two successive backward/update
  cycles use fresh graphs. Compare captured pre-prediction contexts against the
  public evaluator on the *same public slot order* when prior learned choices
  agree with teacher choices. Test slot permutation separately. Do not compare
  distinct pooling orders as if bitwise equality were guaranteed. Confirm no
  producer model API acquires a teacher/private input and no public evaluation
  is patched to return a correct answer. Record exact forward meter counters;
  duplicated public observations are real counted compute.
- [ ] **Step 6 — Commit and independent review:** Commit with
  `feat: train content decisions in immediate latent contexts`. The review
  checks the auxiliary contract and unchanged timed/public paths. Return D,
  not this unit task, owns source-bound production integration and its gates.

### Return D: Integrate the Versioned Recipe, Numerics and Evidence

**Files:** Modify `train/config.py`, `train/trainer.py`, `train/verification.py`,
the profile-dispatch checks in `train/curriculum_data.py` and `train/checkpoints.py`,
`train/provenance.py`, `train/evidence.py`, `train/evidence_types.py`,
`scripts/verify_phase3_gate_artifact.py`, and their focused neural tests;
create `train/objective.py`, `configs/train/one_hop_content_v2.yaml`,
`configs/train/smoke_content_v2.yaml`,
`manifests/validation/phase3/delivery.json`, and
`tests/neural/test_training_objective.py`. Update `train/cli.py`, checkpoint
tests, package/import tests and `scripts/check_phase3_components.py` only where
needed by the changed contracts. Modify the Phase 3 delivery check/tests in
`tests/integration/test_phase0_repository.py` without changing Phase 1/2 pins.
Document the fixed recipe and adverse history in `docs/phase3-neural-components.md`
and `docs/deviations.md`. Preserve the old two training overlays byte-for-byte.

**Recipe identity:** Add exactly `phase3_one_hop_content_v2` and
`phase3_smoke_content_v2` profiles with the original respective workloads
`(128,75000,1000,10000)` and `(8,4,2,16)`. New overlays explicitly set epsilon
`1e-6`. Existing v1 profiles retain epsilon `1e-7`; reject cross-version epsilon
combinations and all unapproved values. The profile itself defines the objective
version and fixed auxiliary coefficient, exposed as read-only derived properties
`objective_version` (`teacher_timed_v1` or `teacher_timed_plus_content_v2`) and
`content_auxiliary_weight` (0.0 or 1.0). This preserves old canonical config bytes
without a new default field silently changing historical hashes. Record the
derived identity/coefficient explicitly in new step/evidence metadata and bind
them to the canonical profile; they are not independently tunable options.
No new seed, loss-weight sweep, schedule, update allowance or runtime agent.
Use explicit profile-family properties for production/debug and curriculum
stage where shared consumers need them; do not let a new production profile
fall through an old `else smoke` branch. Preserve the actual data transform,
seed namespaces, canonical serialization and checkpoint ownership semantics.

**Interfaces:**

```python
# train/objective.py
@dataclass(frozen=True, slots=True)
class TrainingObjective:
    total: torch.Tensor
    timed: UnrollResult
    timed_loss: LossBreakdown
    content: ContentUnrollResult | None
    content_loss: LossBreakdown | None
    objective_version: str
    auxiliary_coefficient: float

def training_objective(
    model: EventFlowModel, batch: TrainingBatch, training: TrainingConfig
) -> TrainingObjective: ...

# verification.py: optional argument preserves explicit old fixtures;
# every new configured evidence caller MUST pass its actual training config.
def measure_device_parity(
    model: EventFlowModel, batch: TrainingBatch,
    *, training: TrainingConfig | None = None,
) -> DeviceParityEvidence: ...
```

The shared objective builder always computes the unchanged timed branch.
It computes content only for v2; `total = timed_loss.total +
training.content_auxiliary_weight * content_loss.total`. Its real consumers
are `train_one_step` and configured device parity. Do not create a second
approximate training implementation inside verification. Preserve actual
forward/loss tensors for both branches, all gradients and updated parameters.
V1 reproduction remains explicit, not silently upgraded to v2. Old archives
must still canonical-decode; source-bound historical execution uses its named
source revision. Cross-recipe resume must reject before changing model/RNG.

- [ ] **Step 1 — RED, training:** Tests assert the v2 optimizer sees both losses,
  one original batch counter, exactly one backward/clip/update, and unchanged
  timed term values. `StepResult` exposes both namespaces (`timed/...`,
  `content/...`), scalar totals, raw reduction diagnostics, objective identity
  and per-branch forward/combined compute. All tensor-valued states and selected
  losses/gradients must be finite. No fake all-zero auxiliary groups.

  ```python
  objective = training_objective(model, batch, corrected.training)
  torch.testing.assert_close(
      objective.total,
      objective.timed_loss.total + objective.content_loss.total,
      rtol=0, atol=0,
  )
  result = train_one_step(model, optimizer, batch, corrected)
  assert result.objective_version == "teacher_timed_plus_content_v2"
  assert result.auxiliary_coefficient == 1.0
  assert result.compute.foundation_model_calls == 0
  ```

  Also test v1 canonical JSON round trips byte-for-byte, v2 rejects wrong
  epsilon/profile pairings, and actual optimizer options survive checkpoint
  save/load. A v1 checkpoint cannot resume a v2 run or mutate state on rejection.
- [ ] **Step 2 — RED, evidence:** Tampering with objective version/coefficient,
  missing either branch's actual numeric tensor names or step diagnostics,
  mismatched configured optimizer, incomplete counters, omitted counter-0/1
  batches, raw row totals or source closure must fail independently. Existing
  tamper and executing-source tests remain. Include Return C's three modules,
  objective module, two new overlays and fixed delivery map in the literal
  authenticated source closure. Producer arithmetic is not verifier authority.
- [ ] **Step 3 — RED, delivery:** The fixed map declares only:

  ```json
  {
    "schema_version": "phase3-delivery-map-v1",
    "recipe": "teacher_timed_plus_content_v2",
    "manifest": "manifests/validation/phase3/one-hop-10000-content-v2.json",
    "gate": "manifests/validation/phase3/component-gate-content-v2.json"
  }
  ```

  This is an immutable source-bound path/recipe mapping, not a results file.
  Hashes are bound by exact Complete metadata and independent gate/manifest
  verification when present. Test missing gate + In progress, unearned Complete,
  present nonpass gate + In progress, nonpassing verification, mismatched map
  recipe, path escapes, symlinks, conflicting old/new designated gates and
  coordinated prior-phase substitution. A failed archive never qualifies as
  delivery. Presence of the old canonical `component-gate.json` must not bypass
  validation merely because the map points elsewhere. Keep legacy test fixtures
  explicit; do not broadly exempt rejected artifacts from the delivery check.
- [ ] **Step 4 — Run RED and implement:** Run focused changed tests first.
  Integrate the fixed recipe and shared builder. Preserve full timed loss
  construction and public prediction code. Prefix v2 diagnostics without key
  collisions; CPU/MPS resume comparisons include both objective contexts.
  Use one outer `NeuralComputeMeter` across both forward graphs and backward;
  branch snapshots describe their own forwards and must not be summed again
  into the already inclusive outer total. Every duplicate observation/scorer/
  controller/jump/flow operation counts. Keep `next_batch_counter ==
  optimizer_step`, existing journal ordering, atomic publication and retention.

  New gate evidence must identify the v2 objective explicitly and validate both
  contexts; version its schema where its shape changes rather than pretending
  old bytes have new fields. Preserve the original failed v1 JSON exactly and
  name its original verifier/source in archival reproduction instructions.
  Update exact overlay validation and production-profile tests to distinguish
  the two reviewed recipes without accepting arbitrary paths or profiles.
  `measure_offline_imports` currently embeds `configs/train/smoke.yaml` in its
  real subprocess. Give it an explicit validated recipe/config input so the
  new gate probes `smoke_content_v2.yaml` and reports the combined objective;
  its independent verifier must derive that exact same-family smoke identity.
  Retain network/import denial and actual one-step/eight-public-row execution.
- [ ] **Step 5 — Actual native numerical prerequisites:** After focused tests
  pass, run native CPU/MPS float32 parity for the original B8 fixture and
  corrected-config B8/B128 counter-0 batches with the production architecture
  and seed 11. Preserve the Return B original-timed epsilon1e-6 result separately.
  Configured v2 parity must execute the **actual combined training objective**
and epsilon1e-6; an old timed-only smoke helper is insufficient. Keep all raw
  tensor/choice comparisons, genuine empty memory and active/dormant guards.
  Use rtol1e-4/atol1e-5 forward/loss, rtol1e-3/atol1e-5 gradients/weights,
  exact CPU resume and rtol1e-4/atol1e-5 MPS resume. Any failure stops the
  candidate before a long fit; diagnose it, do not change tolerance or epsilon.
- [ ] **Step 6 — Actual learning/local prerequisites:** Run the existing fixed64
  overfit workload under the corrected recipe with the unchanged 1,000-update
  cap and >=0.99 full-chain assertion (therefore 64/64), retaining all variants,
  recall/composition counts and secondary action results. No extra pilot or
  hyperparameter search is predeclared. Run local `make verify`, historical
  Phase1/2 verifiers, installed-package/offline checks and fresh source-bound
  smoke evidence. Resolve the previously deferred overwrite-regression Minor
  in `test_phase3_provenance.py` here if its surface changes: demonstrate the
  actual no-clobber collision after the first manifest is committed, or directly
  test the publisher and verify unchanged bytes, not merely a dirty-tree error.
  Report actual per-step wall time, metadata bytes and free-space projection
  for the new recipe; the original throughput/storage estimates are not valid
  for two graphs. Do not start another fit before these prerequisites pass.
- [ ] **Step 7 — Commit and review:** Commit the source/config/evidence correction
  with `fix: align component training and verify the configured objective`.
  Obtain independent spec/quality review of the full correction, including
  historical compatibility, actual native evidence and unchanged scientific
  gates. Corrections use TDD and scoped re-review. Capture this reviewed commit
  and effective plan revision as the next Task13 source identities.

### Return E: Resume Task 13 with New Identities Only

After Returns A–D and reviews, repeat Task13's full evidence workflow with:

| Item | Corrected identity |
| --- | --- |
| Training overlay | `configs/train/one_hop_content_v2.yaml` |
| Smoke overlay | `configs/train/smoke_content_v2.yaml` |
| Corpus | `manifests/validation/phase3/one-hop-10000-content-v2.json` |
| Acceptance artifact | `manifests/validation/phase3/component-gate-content-v2.json` |
| Logical run ID | `phase3-components/event_flow/11/one-hop-content-v2` |

Reissue the source/config-bound corpus metadata before training, and prove its
10,000 ordered keys, public IDs, public/example hashes, exact variant allocation
and content labels equal the original corpus. Commit it before fit. Keep the
original corpus bytes and rejected attempt untouched. A new metadata envelope
is not a new selection of validation examples.

Fit from fresh seed11 weights, B128, at most75,000 total updates, validation
every1,000 on all10,000 rows, patience15 and the original checkpoint-selection
rules. Keep every attempted validation and all raw logs. The extra content
branch does not grant additional updates. One live fit only, persistent
main-volume storage with measured sufficient space, no hidden model/network.
On completion, export selected weights, repeat full public CPU evaluation
twice, collect actual configured native parity/resume/offline evidence and run
independent verification. Archive any new rejected attempt with exact bytes;
do not place it in a designated delivery path while claiming In progress.

Only a valid independently passing artifact plus fresh local checks permits
the exact Complete row. Then finish Task13's documentation/commit, independent
task review and the final whole-phase review from `daa729d`. The conclusion
remains untimed neural-component engineering evidence, never autonomous timed
OFD success. Any source change after this freeze invalidates incompatible v2
evidence and requires another explicit reviewed source-return ruling.

**Spec/fairness boundary:** The auxiliary uses only the already authorized
oracle content supervision, never private model inputs. The original real-time
teacher trace remains trained in full on every update, satisfying the timing
contract; it is not replaced with an activation-time cognitive experiment.
Carry the extra exposure and measured training compute into the Phase5 baseline
plan so every recurrent comparator receives equivalent content supervision.
This is a versioned component-training correction, not factor-four tuning of
the original loss coefficients, a scientific protocol change or a new claim.

## 8. Final Review Hardening — 2026-09-15

This is the single collective fix wave from the whole-phase review of
`daa729d..4829bbcaaca64725d62ef06db6606b464fe353d6`, under the user's standing
approval for reversible in-scope plan corrections. It addresses Important I1
(production execution-source authentication) and Minor M1 (directory creation
before symlink rejection). It does not change the canonical specification,
model, objective, optimizer, corpus examples, numeric tolerances or acceptance
gates. Preserve the full review and both historical production attempts.

### Design and evidence ruling

Use a shared production preflight at the callable training boundary and before
standalone evaluation constructs a model. Read bounded manifest/archive metadata,
authenticate the actual loaded checkout and its declared source closure, and
validate canonical configuration before generating the corpus or creating output.
Both initial fit and resume go through the same boundary. Reuse the existing
`authenticate_source` mechanism; do not build a competing trust-anchor system.
Debug profiles remain explicitly non-acceptance and continue to run from an
installed wheel outside Git. Low-level tensor/weight loading is not itself a
publication claim and does not require every unit-test model to live in Git.

Create run/attempt directories only through validated no-follow directory
descriptors. Preserve the existing archive codec and its protections. An invalid
leaf symlink or symlinked ancestor must be rejected before its target is modified.

The retained `061fd4079f30956b3f51c706bbb374eb5ae0c2db` run was independently
authenticated at launch and collection; its result is supported, not erased or
relabeled by these interface findings. Nevertheless, correcting files in the
literal execution closure makes its gate incompatible with current-source
acceptance. Preserve that gate and corpus at their existing paths and designate
a separately identified final-source corpus/gate. Do not weaken closure checks
or transfer historical training to a new source label. After source fixes and
the one scoped rereview pass, execute one fresh unchanged-recipe run for final
acceptance. This run regenerates current-source evidence; rejected-input tests
themselves require no long training. Cost: additional local compute and storage,
but no altered validation selection, mathematical training or scientific gate.

### Files and interfaces

- Create `src/silent_cascade/train/execution.py` for shared production-source
  preflight, and `src/silent_cascade/train/run_directory.py` for no-follow
  directory preparation/attempt creation.
- Modify only the necessary Phase3 portions of `train/trainer.py`,
  `train/cli.py`, `train/provenance.py`, and, if a metadata-only public accessor
  is needed, `train/checkpoints.py`. Reuse `_read`/bounded archive validation;
  do not change tensor/archive formats or checkpoint selection.
- Add new source paths to both literal source-closure inventories, including
  `scripts/verify_phase3_gate_artifact.py`'s independent inventory. Do not remove
  existing paths or weaken byte/origin/ancestry checks.
- Update `manifests/validation/phase3/delivery.json` and its exact integration
  expectations for the new designated paths below. Existing historical gates
  cannot substitute for the new designated gate.
- Add `tests/neural/test_production_execution.py` and
  `tests/neural/test_run_directory.py`; extend the existing CLI, trainer,
  provenance, checkpoint, package and delivery tests only as required.
- Update `docs/PLAN.md`'s Phase3 row to In progress while the new designated
  gate is absent; preserve Phase1/2 rows. Document the findings and immutable
  historical result in `docs/phase3-neural-components.md` and `docs/deviations.md`.
- No changes to any frozen Phase1/2 source, specification, prior phase plans,
  historical corpus/gate bytes, model/loss/configuration values, or CI/CD.

Required shared interface:

```python
def authenticate_production_execution(
    config: ResolvedConfig[Phase3Config],
    *,
    manifest: ComponentManifest,
    source_commit: str,
) -> None:
    """Reject misattributed production execution; debug is non-acceptance."""

def prepare_run_directory(path: Path, *, resume: bool) -> Path:
    """Validate/create a real run directory without following symlinks."""

def create_attempt_directory(path: Path) -> Path:
    """Create one unique attempt through a validated directory descriptor."""
```

Extract metadata-only manifest validation from `_manifest_corpus` without
changing its schema/key/seed/profile checks or the corpus it builds. Production
authentication uses the manifest's source/plan identities and the actual package
root, validates loaded local dependency origins, and compares the canonical
configuration with the exact committed recipe. Nonempty configuration source
paths must be the exact ordered overlays. A configuration decoded from a checked
weights archive may have empty source paths; reconstruct and compare the committed
recipe instead of trusting absent paths. Existing checkpoint/index/source fields
retain the verified identity; a new archive format is not required merely to
repeat that field. Recheck execution identity before durable production checkpoint
or final evaluation publication so source changes during a run fail loudly.

For standalone evaluation, verify bounded weights metadata and expected archive
hash before authentication, then call the normal checked model loader with that
same hash. Do not construct a model or regenerate a corpus first. A second bounded
archive read is acceptable; do not add a new codec to avoid it. Debug/package
outputs must remain unmistakably ineligible for production acceptance.

### Ordered fix and validation steps

- [ ] **Step 1 — Reproduce I1 and M1 without a production fit.** Use the real
  isolated-checkout/child-process helpers in `test_phase3_provenance.py`. A
  production manifest, archive and source label must agree while the executing
  source is changed, stale or loaded from a foreign package. Exercise CLI fit,
  callable fit, both resume paths and evaluation. Assert typed rejection before
  corpus generation, model construction, RNG restoration or any output creation.
  Use downstream sentinels to stop immediately if the old implementation crosses
  that boundary; do not run 75,000 updates to demonstrate a missing preflight.
  Test dirty and separately committed source changes and foreign dependency origins.

  The directory regression must observe the real target tree, not only an error:

  ```python
  def test_symlink_run_rejection_has_no_target_side_effects(tmp_path, smoke_config):
      from silent_cascade.train.trainer import run_training
      from silent_cascade.train.state import TrainingError
      from .test_trainer import _manifest

      target = tmp_path / "target"
      target.mkdir()
      link = tmp_path / "run-link"
      link.symlink_to(target, target_is_directory=True)
      manifest = _manifest(smoke_config, tmp_path / "manifest.json")
      before = tuple(target.iterdir())
      with pytest.raises(TrainingError):
          run_training(smoke_config, validation_manifest=manifest, run_dir=link,
                       source_commit="a" * 40, device="cpu")
      assert tuple(target.iterdir()) == before
  ```

  Reuse/import the existing `smoke_config` fixture explicitly in the new test
  module. Add the ancestor variant `run-link/new/run`, asserting that neither
  intermediate nor attempt directories appear in the target. Cover a legitimate
  empty real run, fresh missing real parents, and resume of an existing real run.
- [ ] **Step 2 — Apply the minimal source and directory fixes.** All production
  producer routes call the checked boundary; debug remains non-acceptance and
  both installed-wheel smoke profiles still work. No model-math, corpus, loss,
  optimizer, budget or tolerance changes. Directory operations use `O_NOFOLLOW`,
  `O_DIRECTORY` and descriptor-relative creation, close descriptors on every exit,
  and preserve the existing explicit-resume/nonempty-run behavior. Unique attempt
  naming must not consume the Python/NumPy/Torch training RNG streams.
- [ ] **Step 3 — Refresh delivery identity honestly.** Set the new fixed map
  below and an In progress row; retain both older corpora/gates unchanged. Extend
  delivery regressions for missing/new/conflicting gates and forged Complete rows.
  No archived or merely asserted success may satisfy the new designation.
- [ ] **Step 4 — Verify and commit the source fix.** Run focused RED/GREEN and
  covering CLI/trainer/checkpoint/provenance/package/import/delivery regressions,
  local Ruff and the full local `make verify`. Commit this plan clarification
  with the source fix and truthful delivery state. On the clean committed source,
  run a real same-recipe debug smoke plus native production-B128 parity/CPU+MPS
  resume and the independent complete-inventory checks, using unchanged tolerances.
  Retain actual commands/exits/hashes. These are prerequisites, not a new fit.
- [ ] **Step 5 — One scoped rereview, then release execution.** The same final
  reviewer verifies I1, M1 and fix-introduced breakage against the full fix diff,
  including the new source/data boundary. Do not launch the final production run
  before that review passes. Later generated evidence and final local exits remain
  explicit pending conditions resolved from actual artifacts by the controller;
  no second broad review or second independent source-fix wave is authorized here.
- [ ] **Step 6 — Regenerate final current-source acceptance.** Capture the reviewed
  source and latest effective primary-plan revision, reissue exactly the same
  10,000 ordered examples under the new source envelope, verify equality with
  both historical corpora, and commit only that manifest before fitting. Use
  fresh seed11, the unchanged content-v2 recipe and the exact Task13 workload,
  CPU OMP1, one live process, persistent owner-only storage and no silent restart.
  Run to its natural stopping rule. Retain every validation and adverse outcome.
  Export the actual selected weights; perform both complete CPU evaluations,
  actual configured native evidence collection and independent verification.
  Never relabel or resume either completed historical run under the new source.
- [ ] **Step 7 — Finish delivery from actual evidence.** Only a valid passing
  new gate permits the exact Complete row. Append the full new history/variant/
  action/compute/numeric/source/config/manifest/weights data and reproduction
  commands while preserving all historical results. Run Task13's precommit and
  postcommit local gates and final independent verifier; commit only compatible
  artifacts/docs. The controller checks every remaining scoped-review condition,
  preserves all coordination/diagnostic evidence, and performs the prescribed
  final cleanup. Do not start Phase4 or claim autonomous timed OFD success.

### Final-source identities (recipe and schemas unchanged)

| Item | Designated identity |
| --- | --- |
| Training overlay | `configs/train/one_hop_content_v2.yaml` |
| Smoke overlay | `configs/train/smoke_content_v2.yaml` |
| Corpus | `manifests/validation/phase3/one-hop-10000-content-v2-authenticated.json` |
| Acceptance artifact | `manifests/validation/phase3/component-gate-content-v2-authenticated.json` |
| Logical run | `phase3-components/event_flow/11/one-hop-content-v2-authenticated` |
| Execution-log namespace | `phase3-task13-final-hardening` |

The recipe stays `teacher_timed_plus_content_v2`, coefficient1.0 and epsilon1e-6;
the component/numeric evidence envelopes remain v3 unless their actual primitive
schema changes. Source/plan/manifest/run identities change; historical bytes do
not. The new corpus is not a new draw or validation-selection opportunity.
