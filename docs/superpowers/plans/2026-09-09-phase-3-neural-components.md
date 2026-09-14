# Phase 3 Neural Components Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the bounded, trainable EventFlow component stack and reproducible teacher-forced training path, then demonstrate greater-than-99% one-hop retrieval/composition on a fixed, label-isolated component-validation corpus.

**Architecture:** Add a batched differentiable neural workspace alongside the completed scalar event engine, sharing its dimensions, equations, public schemas, and numerical contract without changing its frozen source. Separate public model inputs from privileged training targets; use the same encoders, scorer, controller, jumps, and heads during training and unassisted component validation. Phase 4, not this plan, connects those components to autonomous world-time guard scheduling and the full OFD pilot.

**Tech Stack:** Existing locked Python 3.12, PyTorch float32 on CPU/Apple MPS, NumPy, Pydantic, safetensors, Typer, pytest/Hypothesis, and uv; local `make verify` only.

**Spec:** `docs/superpowers/specs/2026-08-30-silent-cascade-design.md`, approved version 1.0.2, especially Sections 5, 8, 9.4, 13, 17 Phase 3, and 18.

**Status:** User-approved on 2026-09-14 ("proceed, use subagents"); completion tracked in `docs/PLAN.md`. This document is not an implementation-completion claim.

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
| 12 | `train/{provenance,evidence}.py`, `scripts/{check_phase3_components,verify_phase3_gate_artifact}.py` | Source-bound raw evidence and independent verifier |
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
`(0.9,0.999)`, epsilon `1e-8`, `foreach=False`, `fused=False`, no scheduler.

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
    context = replace(model_context, eligible=torch.zeros_like(model_context.eligible))
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
- `run_training(config: ResolvedConfig[Phase3Config], *, validation_manifest: Path, run_dir: Path, source_commit: str, resume: Path|None = None) -> TrainingRunResult`.
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
modify `Makefile` and `tests/integration/test_phase0_repository.py` only where
needed for actual local smoke/test commands. Existing root CLI and frozen source
remain unchanged.

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

**Files:** Create `train/{provenance,evidence}.py`,
`scripts/check_phase3_components.py`, `scripts/verify_phase3_gate_artifact.py`, and
`tests/neural/{test_phase3_provenance,test_phase3_evidence,test_phase3_verifier}.py`;
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

- [ ] **Step 1: Write RED tests** for debug-size real collection, production/debug
  separation, source closure (including new ancestor initializers), seed/manifest
  binding, checkpoint mismatch, raw-denominator arithmetic, wrong/extra/missing
  predictions, omitted variants, changed thresholds, source mutation, deep JSON,
  no-clobber publication, and preserved prior-phase evidence.

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
and zero foundation calls. Selected weights remain under `runs/`; raw artifact
verification binds their SHA, while a separate execution check loads those
weights and reproduces all raw component decisions on CPU. Do not claim that
arithmetic/hash verification alone proves the training computation occurred.

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
stay in ignored `runs/phase3-components/`; do not commit normal checkpoints.

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
```

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
uv run --offline python -m silent_cascade.train fit \
  --config configs/base.yaml --config configs/data/primary.yaml \
  --config configs/model/event_flow.yaml --config configs/model/neural_components.yaml \
  --config configs/train/one_hop.yaml \
  --validation-manifest manifests/validation/phase3/one-hop-10000.json \
  --expected-source-commit "$phase3_source_commit" \
  --run-dir runs/phase3-components/event_flow/11/one-hop-v1
```

Prefer available MPS with no fallback; preserve the CPU path. Write a source-
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
  --output runs/phase3-components/event_flow/11/one-hop-v1/cpu-evaluation.json
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
  --training-run runs/phase3-components/event_flow/11/one-hop-v1 \
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
