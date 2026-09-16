# Phase 4 Autonomous EventFlow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the learned neural components to the real event engine and deliver a reproducible, one-seed, autonomously timed OFD pilot that meets the unchanged Phase 4 gates.

**Architecture:** Reuse the scalar causal scheduler and batched EventFlow model through a public-state-only adapter and shared learned transition functions. Keep teacher supervision in training, put all runtime decisions and event times under learned heads/guards, and validate the complete action path rather than substituting component accuracy. Preserve earlier evidence at its original source while giving every new training run, runtime checkpoint, evaluation and report its own authenticated provenance.

**Tech Stack:** Existing locked Python 3.12, float32 PyTorch CPU/Apple MPS, NumPy, Pydantic, safetensors, Typer, pytest/Hypothesis and uv. Local `make verify`; no new dependency is planned.

**Spec:** `docs/superpowers/specs/2026-08-30-silent-cascade-design.md`, approved v1.0.2; especially Sections 4–8, 9.4, 12–16, 17 Phase 4 and 18. The canonical specification remains unchanged.

**Status:** User-approved; completion tracked in docs/PLAN.md. Approved on 2026-09-16 by the user's explicit instruction, "execute with subagents". Execute on the current branch with task-scoped implementation and review agents.

**Repository inspected:** Clean `main` at `ea713cdce4107e709985d9333050a947db6561ab` on 2026-09-16. Keep the current branch. This is a phase-scoped plan, not authorization for Phases 5–7.

## Global Constraints

- The Python package is named `silent_cascade`; the CLI executable is `silent-cascade`.
- Use Superpowers TDD, systematic debugging for failures, task-scoped reviews, one final whole-phase review, and verification before completion. Make meaningful local commits; do not push, merge or create a different worktree by default.
- CI/CD is disabled. Run linting, formatting, tests, diagnostics, and package builds locally. `make verify` is the complete local quality gate.
- Primary scientific execution must remain offline and record exactly zero foundation-model calls. No Qwen/MLX imports, downloads, network, telemetry or hidden natural-language reasoning.
- At most 5,000,000 total trainable parameters, including the entity table; 64 primary memory records; 64 EventFlow internal events. Preserve the separate 25,000-opportunity ceiling for future scheduled controls.
- Keep `z_fast=256`, `z_slow=64`, drives `8`, focus key `64`, hypothesis latent `64`, guards `3`, and 96-dimensional record embeddings. Use float32 CPU/MPS model tensors and host float64 absolute times; no float64 MPS tensors, mixed precision, `torch.compile`, silent MPS fallback or EventFlow Transformer.
- Preserve exact exponential flow, latent bounds `[-1,1]`, latent rates `[1e-5,20]`, guard targets `(0,2)`, guard rates `[1e-5,500]`, threshold `1`, active/inactive margins `1.10/0.90`.
- Preserve minimum internal gap `1e-4`, same-kind refractory `1e-3`, near-tie tolerance `1e-9`, failure on the fifth consecutive clamp, and failure on an attempted 65th internal event. No periodic fallback, invented cognitive tick or EventFlow `NOOP`.
- Preserve tie order: terminal, ordinary external event, ACT, COMPOSE, RECALL, baseline NOOP. Time is nondecreasing; equal-time events retain the declared ordering and unique IDs.
- The agent receives only `AgentInit`, public FACT/ACTIVATE callbacks and public-derived runtime state. Never pass a bundle, target, generator coordinate, seed, private terminal timestamp, future queue, path, label or scoring window to it.
- The engine alone compares internal predictions with the next external event and privately handles OUTCOME/END. A private outcome is not an agent callback.
- No deliberate retrieval/composition before activation. No exact-subject retrieval mask, copied oracle focus/hazard, hand-coded path traversal, or scripted action scheduling inside the learned condition.
- A positive succeeds only with exactly one correct-class action in `[t0 + 0.75*d, t0 + 0.90*d)`; a negative succeeds only with no action. Do not substitute reward, conditional accuracy, class accuracy or teacher forcing for timed success.
- Training remains supervised and teacher-forced. Do not differentiate through autonomous discrete rollouts, introduce RL or implement a foundation-model solver.
- Preserve AdamW defaults: learning rate `3e-4`, weight decay `1e-4`, batch `128`, clip norm `1.0`, total optimizer-step ceiling `75000`, validation interval `1000`, fixed validation count `10000`, patience `15`, best checkpoints `3`, latest `1`. Carry the tested epsilon `1e-6`, betas `(0.9,0.999)`, `foreach=False`, `fused=False`, no scheduler.
- The pilot uses model seed `11` only. Final seeds `[11,23,37,53,71]`, strong baselines, compute-matched comparisons, full interventions/statistics, frozen tests and scientific-support claims belong to later phases.
- Every new number must bind raw rows, code revision, canonical config, generator/transform version, manifest, model seed and actual checkpoint hash. Fail loudly on NaNs, invalid dynamics, corruption and replay mismatch; never omit a failed episode from its denominator.
- The final Phase 4 gate is **at least 90% pilot IID timed success, at most 10% negative false-action rate, zero dynamics failures across 10,000 validation episodes, legal event sequences, actions following swapped delays, and zero foundation-model calls**. Additional engineering checks below do not weaken or replace these gates.

---

## 1. Starting Evidence and the Action Question

### 1.1 Accepted baseline

| Item | Immutable identity |
| --- | --- |
| Phase 2 source | `33fb8ed105046cb4fdb7be61eac50ed391415b7c` |
| Phase 2 gate | `c05b2f78680b14e5b8fd81597489f34ec68a1dc2d1114e62daa95d7428694c42` |
| Phase 3 source/effective plan | `844a89cd04405139ca670e1b269ce78f2ae9a168` |
| Phase 3 corpus | `2e1d367c6ac072d682afd241f280b5ff729491ca523c2106a4bb176be308d6d2` |
| Phase 3 accepted gate | `5c87928eb70fa5c8fc0b338522fb3ec08a7c216938b701b0887d3f68b2247e4e` |
| Phase 3 portable weights | `ba6f929f1e5bfa9cf00b8c31b948f35582dc6662a23203db02c9e5f17ee93e3c` |
| Historical verification checkout | `ea713cdce4107e709985d9333050a947db6561ab` |

The Phase 3 run achieved 9,920/10,000 complete **one-hop, untimed** chains, not a
timed cascade. Its five-class action readout achieved 6,629/10,000: positive
1,641/5,000; safe 2,500/2,500; disconnected 2,488/2,500. All 80 chain failures,
the original rejected attempt and both later accepted runs remain preserved.
See `docs/phase3-neural-components.md`, accepted authenticated gate section.

### 1.2 What must be diagnosed, not assumed

Inspection establishes a context difference, not its causal explanation:

- `train/component_eval.py` calls `model.action(context)` immediately after its
  untimed content loop and scores five-way argmax, including abstain.
- `train/unroll.py` supervises positive actions at the later teacher ACT time,
  after continuous flow; negative abstention is supervised after terminal/null
  composition.
- `train/content_unroll.py` and `models/content_loss.py` add retrieval/composition
  supervision, but deliberately contain no untimed action loss.

Do not predeclare that the action head is broken, that timing will cure it, or
that copying the hypothesis class is a fix. Task 7 records confusion matrices,
guard availability, and action predictions at the relevant contexts. An action
repair is permitted only after a reproducible defect is isolated and covered by
a failing test. A new loss or architecture is not silently invented by this plan.

### 1.3 Inspected APIs and hazards

| Existing code | Actual interface / consequence |
| --- | --- |
| `EventEngine` | `start_episode`, `step`, `run_until`, `run_episode`; callbacks receive `RuntimeState`, never an external horizon |
| `RuntimeCore` / `RuntimeState` | Own continuous state, segment origin, memory, support, hypothesis, action history and authoritative engine counters |
| `eventflow/jumps.py` | Typed metadata legality is reusable, but current jumps also inject scripted integer-derived latent impulses; learned execution must bypass those impulses |
| `eventflow/guards.py` | `next_crossings(state)` already performs host-time analytic crossing/refractory decisions |
| `EventFlowModel` | `observe`, `preview_and_control`, `recall_scores`, `compose`, `action`, `jump`; preserve these component owners and shared parameter aliases |
| `ModelContext` / `TensorWorkspace` | Batched public tensors; latent width 456, separate three accumulators; use functional updates and `torch.no_grad()`, not unversioned inference tensors |
| `memory/tensor_store.py` | `pack_memory(memories, initial_times, device=...)` supplies record-slot correspondence; filter validity/consumption/refractory separately |
| `train/curriculum_data.py` | Already implements one-hop, two-hop, primary and `[0.5,2]` robustness transforms, but its config check is Phase3-specific |
| `train/batches.py` / `traces.py` | `pack_training_examples` accepts curriculum examples up to primary depth; max trace is 11 positions for four links plus terminal/action |
| `train/trainer.py` | Actual driver, selection and stop conditions are one-hop-only; merely changing a stage string does not implement a pilot |
| `train/checkpoints.py` | Safe weights/AdamW/RNG archives authenticate a closed Phase3 configuration; do not serialize a Phase4 run under that identity |
| `eventflow/checkpoint.py` / `replay.py` | Closed `scripted-event-flow-v1` registry; neural weights and private agent metadata are not supported yet |
| `logging/trace.py` | Existing causal chain and analytic trajectory; detailed neural scores need separately bound observations, not polling events |
| `scripts/verify_phase{2,3}_gate_artifact.py` | Require historical source closure to match the executing checkout; root CLI/runtime edits necessitate explicit historical verification |
| `Makefile` | Three isolated pytest processes; preserve their isolation, add only working pilot targets |

## 2. Chosen Approach and Scope

### Alternatives considered

1. **Chosen: adapter over the existing engine, shared learned transitions.** This
   preserves tested queue/flow/tie semantics and exposes runtime/teacher mismatches
   with row-equivalence tests. Cost: explicit tensor/scalar conversion and a
   carefully versioned historical-evidence transition.
2. A second vectorized neural event engine could increase validation throughput,
   but would duplicate scheduling, replay and failure semantics. Defer it until
   measured profiling justifies a separately reviewed equivalence-preserving change.
3. Copying the terminal hypothesis into an action or scheduling at `0.825*d`
   would bypass the learned decisions being tested. Do not implement this option.

### Resolved implementation decisions

1. **One authoritative causal state.** Keep all latent/guard/semantic history in
   `RuntimeState`. Agent-owned data is limited to immutable model weights/config,
   the current public `AgentInit`, and diagnostic counters. Reconstruct public
   context from state without replaying the transcript. Initially re-encode the
   bounded memory at each needed callback and count it; no uncheckpointed cache.
   Before the first public FACT, use a neutral analytic segment whose latent
   targets equal the initial state and whose guards are dormant. Do not call a
   learned controller before the first event. The existing teacher observer
   starts its learned trajectory at that first FACT; preserve this shared
   initialization contract instead of introducing an untrained runtime-only
   latent flow. Elapsed time still enters the public FACT time features.
2. **Shared transition order.** Extract public tensor transition operations from
   teacher unrolls and reuse them at runtime. Prediction happens before applying
   the decoded/teacher decision; mode/support/hypothesis features enter the jump
   in the same order. Preserve Phase3 forward behavior under its existing recipe.
3. **Learned ACT timing only.** Runtime ACT time comes exclusively from the ACT
   guard crossing. The action lead head stays supervised/diagnostic; it cannot
   schedule a second action, move a crossing or clip into a private window.
4. **Explicit action legality.** At an actual ACT event in HOLDING_HAZARD, choose
   argmax over the four SHIELD logits; abstention is represented by not crossing
   ACT. Keep all five raw logits and the unmasked argmax in diagnostics. Apply the
   same legality convention to later recurrent controls. This is a declared
   decoder contract, not an improvement to the historical five-way 66.29% score.
   Do not copy the hypothesis class, mask wrong shield classes, or choose using
   private truth. Tests force disagreement between hypothesis and action heads.
5. **Predicted deadline.** Decode the existing normalized-deadline output as
   `activation_time + predicted_value * (1 + elapsed_since_activation)` using
   host floats. Do not read the active record's delay to correct the prediction.
   Log the separately predicted log-delay regression. A past prediction is not
   moved into the future; invalid/nonfinite schema values fail loudly.
6. **Fresh pilot training.** The accepted Phase3 weights are for diagnosis, not
   free uncounted pretraining. Train the pilot from seed 11 with a single 75,000
   total-update ceiling across all curriculum stages. Reuse architecture/code,
   not old optimizer progress or configuration identity. No fresh seed retries.
7. **Keep the existing objective initially.** Use the full teacher-timed loss
   plus the fixed coefficient-1.0 untimed-content auxiliary and epsilon `1e-6`.
   Do not add action copying, straight-through autonomous training, a new head,
   or an action auxiliary without an evidence-backed versioned plan amendment.
   Any validation-only coefficient adjustment must stay inside the canonical
   factor-four envelope and retain every attempted configuration/result.
8. **Historical evidence stays historical.** Preserve original source closures,
   plans, specs, artifacts, delivery-map bytes and completed rows. Verify their
   original code in a pinned disposable local checkout. New Phase4 results must
   authenticate their actual executing source; no relaxed legacy verifier flag.
9. **Errors are observable.** A bad but legal class/focus/early stop is an ordinary
   scored failure. NaN, illegal state, missing required support, malformed output,
   cap/Zeno, time reversal or corruption aborts that episode with a crash bundle.
   A validation runner may continue with other independent episodes while marking
   the checkpoint invalid; it must raise/report a failed validation at the end.
   It never resumes the corrupted episode or turns its error into abstention.
10. **Paused-world CPU publication.** Generate accepted pilot traces on CPU with
    frozen weights. Require exact CPU resume/replay; native MPS remains a real
    numerical/resume/behavior smoke gate. Do not promise bitwise MPS trajectories.
11. **One legal terminal interpretation.** The learned COMPOSE role selects the
    typed LINK/HAZARD/SAFE/irrelevant/contradictory interpretation; predicted
    `continue_search` selects continuation where the role permits it. Use that
    same decision to build typed registers and tensor hypothesis features. The
    auxiliary status head remains supervised and its disagreement is logged,
    not secretly used to repair the role or create a second conflicting state.
    Wrong finite semantic choices are scored failures, not oracle-corrected facts.

### Explicit deferrals

No one-shot/ponder/fixed-grid competitors, compressed-time intervention, full
causal/probe suite, OOD scientific evaluation, five-seed bootstrap, frozen-test
generation/tag, full research report, Qwen, public showcase, hosted automation,
or generic framework. A small artifact-only pilot report and narrow delay-swap
engineering check are in scope; neither supports a continuity-advantage claim.

## 3. Phase 4 Data, Training and Acceptance Contract

### Data identities

Use `phase4-pilot-v1` as experiment version and `phase4-data-v1` as the new recipe
envelope. Preserve generator `ofd-v1` and existing `ofd-one-hop-v1` transform
bytes/results. Pilot `CurriculumKey` values use train roots `431/433` (generation /
opaque public-ID key), validation `439/443`, debug `449/457`. The existing tuple
hash includes split, stage and index; different roots keep Phase3 streams separate.
Do not feed these keys to the model. Record the legacy transform identity honestly
inside the Phase4 envelope rather than relabeling its RNG implementation.

Each production validation manifest contains exactly 10,000 fixed entries:
5,000 positive, 2,500 safe, 2,500 disconnected. Freeze one manifest per stage
(`one_hop`, `two_hop`, `primary`, `robustness`) before the first pilot fit; keep
generation, public and target hashes and ordering. Primary uses paths 2–4,
0–12 distractor links and delays `[8,64]`; robustness uses the same depths plus
the already defined order/timestamp transforms and paired scale `[0.5,2]`.
Two-hop uses path length 2. One-hop retains its explicit curriculum transform,
not a changed primary generator. No `manifests/frozen/` access is needed.

Private curriculum-to-engine projection belongs in `env/pilot.py`: construct
ordinary validated `EpisodeTruth`/`EpisodeBundle` values from the transformed
public episode, solution and private horizon. Rebuild requested depth, support
IDs, terminal identity and action window for that public projection; do not pass
stale parent truth or bypass constructors. Record arbitrary robustness scaling
in the Phase4 recipe envelope, not the canonical extreme-clock suite fields.
Neither the agent nor model may import this private projection module.

### Curriculum and selection

| Stage | Training | Promotion / completion evidence |
| --- | --- | --- |
| One-hop | Fresh seed-11 model; existing combined objective | On its fixed 10,000 corpus, required recall, composition and full-chain accuracy each strictly >99%; report five-way action and timing diagnostics separately |
| Two-hop | Same optimizer/model, no counter reset | 10,000 fully autonomous episodes, legal sequences, zero dynamics failures; require >=90% timed success and <=10% negative false actions as a non-vacuous engineering promotion gate |
| Primary | Paths 2–4, full primary nuisance/delay range | 10,000 autonomous IID episodes meeting the unchanged Phase4 pilot success/error gates |
| Robustness | Same depths, order/timestamp/`[0.5,2]` augmentation | At least 1,000 updates in this stage; evaluate both fixed robustness and unscaled primary corpora at each scheduled validation; both must meet the pilot success/error gates |

Validation remains every 1,000 **global** updates. Stage promotions happen only
at those boundaries; no optimizer/LR reset. Reset patience only on a legitimate
stage promotion. The total step ceiling stays 75,000, not 75,000 per stage.
For one-hop selection use the existing chain/composition/earlier-step order.
For a timed stage, rank eligible checkpoints by timed success, then lower negative
false-action rate, then earlier global step; a checkpoint with any dynamics error
is ineligible. Final selection is from robustness-stage checkpoints passing both
its and the primary gate, ranked by primary success, primary false actions,
robustness success, then earlier step. Do not select using delay-swap results.

Retain best three and latest one full training checkpoints. Store every validation
row, stage-promotion certificate, journal and pruned checkpoint descriptor/hash;
intermediate states can be regenerated from the recorded training stream. Do not
delete adverse diagnostics or failed-run artifacts as checkpoint pruning.

Natural stops: successful robustness completion, global ceiling, or 15 scheduled
validations without improvement in the current stage. Record a failed gate for
unsuccessful stops. Do not automatically add steps, start new seeds, or redefine
success. Resuming an interrupted run continues the same counters and stopping
rule; it is not another attempt budget.

### Final measured gate

The **same selected checkpoint** must produce:

- Complete 10,000-row primary and two-hop CPU runs: timed success >=9,000/10,000,
  negative false actions <=500/5,000, zero dynamics/provenance/cap/illegal-sequence
  failures. Report positives, safe and disconnected separately, including every
  numerator/denominator; abstention-only cannot pass.
- Complete 10,000-row robustness CPU run meeting the same thresholds. This is
  in-range augmentation evidence, not the later `0.1x/10x` scientific test.
- A second complete primary CPU evaluation with identical ordered decisions,
  event identities/timestamps, semantic trace hashes, actions and scores. Ignore
  only explicitly excluded wall-time/allocation telemetry in equality checks.
- 256 predeclared positive primary graphs (first in immutable manifest order),
  each evaluated with **both** publicly observed hazard delays set to `12.0` and
  `48.0`, respectively. Keep graph, class, fact order and observation times fixed;
  regenerate private outcome/scoring truth only in the environment. The windows
  are disjoint. Require at least 231/256 pairs with correct actions inside both
  windows; report all pair deltas and normalized times. Missing actions count as
  failed pairs. This is the concrete engineering definition of following swapped
  delays; it cannot be used for tuning or checkpoint selection.
- Exact CPU continuation and replay at mid-flow and after FACT, ACTIVATE, RECALL,
  COMPOSE and ACT; exact repeated replay hashes. Cover each legal mode and all
  three outcome variants using real selected-weight trajectories.
- Native MPS/CPU checks on the actual recipe and complete named tensor inventories:
  forward/loss `rtol=1e-4, atol=1e-5`; gradients/updates `rtol=1e-3, atol=1e-5`;
  MPS training resume `rtol=1e-4, atol=1e-5`, exact RNG/next-batch identity.
  Also compare event kinds, selected records, predicted classes and final scores
  on a predeclared 64-episode balanced runtime smoke subset, with host event-time
  `rtol=1e-4, atol=1e-5`. Device-induced decision differences are failures, not
  tolerated simply because an aggregate score agrees.
- Real offline/import checks and compute counters, <=5M parameters, all ceilings,
  authenticated weights/source/config/data, artifact-only report generation,
  full local `make verify`, and independent gate verification.

These are pilot engineering acceptance criteria, not Section 10.11/10.12
scientific support. The later 100,000 mixed autonomous-episode gate is not claimed
by these 10,000-row suites.

## 4. File Map and Execution Order

Paths below are repository-relative. Each task has an independently reviewable
deliverable; execute tasks in order. Under SDD use fresh task implementers and
task-scoped spec/quality reviews. Do not run multiple model fits or competing
source writers concurrently. Independent read-only work may overlap verification.

| Task | Main new files | Existing files deliberately changed |
| --- | --- | --- |
| 1 | `src/silent_cascade/train/pilot_config.py`, `scripts/verify_historical_delivery.py`, `manifests/validation/historical-delivery.json` | delivery tests, new pilot config overlays |
| 2 | `src/silent_cascade/models/transitions.py`, `src/silent_cascade/eventflow/neural_context.py` | tensor teacher transitions and explicit-continuous jump boundary |
| 3 | `src/silent_cascade/eventflow/neural.py` | runtime error/diagnostic boundary where needed |
| 4 | `src/silent_cascade/eventflow/{checkpoint_state,neural_checkpoint,neural_replay}.py` | legacy codec extraction, engine closed crash dispatch |
| 5 | `src/silent_cascade/eval/{runner,metrics,artifacts}.py`, `src/silent_cascade/logging/neural_trace.py` | compute instrumentation |
| 6 | `src/silent_cascade/env/pilot.py`, `src/silent_cascade/train/pilot_data.py`, `src/silent_cascade/eval/pilot_audit.py` | generalize curriculum config boundary; reuse shortcut-probe mathematics |
| 7 | `src/silent_cascade/eval/action_diagnostics.py`, `scripts/diagnose_phase4_actions.py` | no model/loss change by default |
| 8 | `src/silent_cascade/train/{pilot_state,pilot_checkpoints,pilot_trainer,pilot_provenance,archive_tensors}.py` | narrowly extract common archive/optimization contracts |
| 9 | `src/silent_cascade/train/pilot_cli.py`, `src/silent_cascade/report/{__init__,pilot}.py`, `scripts/run_pilot.py`, `docs/phase4-autonomous-eventflow.md` | root CLI, Makefile, command/package tests |
| 10 | `src/silent_cascade/train/pilot_verification.py` | new local numerical/offline/throughput tests |
| 11 | `src/silent_cascade/train/{pilot_evidence_types,pilot_evidence}.py`, `scripts/{check_phase4_pilot,verify_phase4_gate_artifact}.py` | finish source inventory, Phase4 delivery guard/tests |
| 12 | `manifests/validation/phase4/` stage manifests, gate and delivery map; generated pilot report | Phase4 delivery row and measured documentation only |

Tests live under new `tests/pilot/`, with integration coverage in the existing
integration/regression directories. Do not add imports from training/private
environment code into model, memory or agent modules.

---

### Task 1: Preserve Historical Acceptance and Define Pilot Configuration

**Files:**

- Create: `scripts/verify_historical_delivery.py`, `manifests/validation/historical-delivery.json`.
- Create: `src/silent_cascade/train/pilot_config.py`, `configs/train/pilot.yaml`, `configs/train/pilot_smoke.yaml`.
- Create: `tests/pilot/test_historical_delivery.py`, `tests/pilot/test_pilot_config.py`.
- Modify: `tests/integration/test_phase0_repository.py`; update real-repository legacy-verifier integration callers to use the explicit historical route, while retaining direct verifier rejection tests.
- Document: `docs/deviations.md` (historical vs current-source verification, no relaxed scientific gate).

**Interfaces:**

- `PilotTrainingConfig(StrictModel)`: profile `phase4_pilot|phase4_smoke`, fixed
  optimizer/workload/seed fields from Sections 2–3, `loss_weights: LossWeights`,
  `objective_version='teacher_timed_plus_content_v2'`,
  `content_auxiliary_weight=1.0`, and `is_production` true only for `phase4_pilot`.
- `Phase4Config(Phase2Config)`: `neural: NeuralModelConfig`, `pilot: PilotTrainingConfig`; strict unknown-key rejection and cross-limit validation.
- `resolve_pilot_config(profile: str) -> ResolvedConfig[Phase4Config]`: existing base/data/event-flow/neural overlays followed by the matching pilot overlay.
- `verify_historical_delivery(*, repo_root: Path, phase: int) -> dict`: only phases 1, 2, 3; no arbitrary command, revision or artifact selector.

- [ ] **Step 1: Add failing configuration and historical-substitution tests.**

```python
def test_pilot_budget_is_global_and_component_profile_is_rejected():
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from pydantic import ValidationError
    import pytest

    resolved = resolve_pilot_config("phase4_pilot")
    assert resolved.config.pilot.max_steps == 75_000
    assert resolved.config.pilot.model_seed == 11
    values = resolved.config.model_dump()
    values["pilot"]["profile"] = "phase3_one_hop_content_v2"
    with pytest.raises(ValidationError):
        type(resolved.config).model_validate(values)
```

Historical tests must mutate a pinned artifact and its claimed hash together,
substitute the checkout revision, use symlinked parents, and inject the current
package into the child search path. All must fail before running a verifier.

- [ ] **Step 2: Run RED.**

Run `uv run pytest -q tests/pilot/test_pilot_config.py tests/pilot/test_historical_delivery.py`.
Expected: missing implementation or explicit unmet pin assertions, not a missing dependency.

- [ ] **Step 3: Implement closed historical execution and config.**

Pin checkout `ea713cdce4107e709985d9333050a947db6561ab` plus exact Phase1/2/3
artifact/source/verifier hashes obtained from its regular Git blobs. Compare
current historical artifacts/maps/completed rows against those pins first.
Use a fresh local clone with `--no-local --no-hardlinks`, no remote contact or
checkout changes in the working repository. Checkout the exact pin detached in
that disposable clone. Run the **pinned original** verifier scripts there with
the existing Python environment, an isolated import path rooted at that clone,
offline variables and bounded JSON output. Authenticate loaded package origins.
Do not run the old training/collector or download dependencies. Return
`verification_scope='historical'`, source, checkout, artifact and verifier hashes
alongside the original verifier result. Cache only a hash-complete receipt; a
changed current artifact invalidates reuse. Keep direct old verifiers strict:
after source changes, invoking them against the live tree must still reject it.

Smoke config: debug architecture, batch 8, max steps 4, validation every 2,
16 debug entries; never acceptance. Production config has the exact Section 3
workload and data roots. Both use epsilon `1e-6`. No mutation of existing Phase3
config files, schemas, artifact bytes or completed index rows.

```python
profile_paths = {
    "phase4_pilot": "configs/train/pilot.yaml",
    "phase4_smoke": "configs/train/pilot_smoke.yaml",
}
if profile not in profile_paths:
    raise ValueError("unsupported Phase4 profile")
```

- [ ] **Step 4: Run GREEN and the historical commands locally.**

```bash
uv run pytest -q tests/pilot/test_pilot_config.py tests/pilot/test_historical_delivery.py tests/integration/test_phase0_repository.py
uv run python scripts/verify_historical_delivery.py --phase 1
uv run python scripts/verify_historical_delivery.py --phase 2
uv run python scripts/verify_historical_delivery.py --phase 3
```

Expected: exact historical acceptance, explicit scope and no current-source claim.

- [ ] **Step 5: Commit:** `feat: preserve historical gates and define autonomous pilot contracts`.

### Task 2: Share Learned Transitions and Bridge Tensor/Runtime State

**Files:**

- Create: `src/silent_cascade/models/transitions.py`, `src/silent_cascade/eventflow/neural_context.py`.
- Modify: `src/silent_cascade/train/{unroll,content_unroll}.py`, `src/silent_cascade/eventflow/jumps.py`.
- Modify: `src/silent_cascade/train/provenance.py`, `tests/neural/test_phase3_provenance.py` to authenticate the new shared dependency in current-source execution.
- Create: `tests/pilot/{conftest,test_neural_context,test_transition_equivalence}.py`.

**Interfaces:**

- `workspace_from_continuous(state: ContinuousState) -> TensorWorkspace` and `continuous_from_workspace(workspace: TensorWorkspace) -> ContinuousState`, exactly one runtime row.
- `context_from_runtime(model: EventFlowModel, state: RuntimeState, init: AgentInit) -> ModelContext`.
- `segment_from_batch(parameters: BatchedSegmentParameters) -> SegmentParameters`.
- `RecallTransition` / `ComposeTransition`: frozen public tensor decisions, including selected slots, mode, support and decoded hypothesis/focus fields; no teacher batch or episode.
- `apply_recall_context(model, context, decision) -> ModelContext`, `apply_compose_context(model, context, decision) -> ModelContext`, `apply_action_context(model, context) -> ModelContext`.
- Existing `apply_fact`, `apply_activate`, `apply_recall`, `apply_compose`, `apply_act` gain keyword-only `continuous: ContinuousState | None = None`. `None` preserves original scripted behavior byte-for-byte; an explicit value bypasses **all** scripted impulses but retains every metadata/invariant check.

- [ ] **Step 1: Write conversion, no-impulse and prediction-before-disclosure tests.**

```python
def test_runtime_tensor_roundtrip_preserves_every_channel():
    import torch
    from dataclasses import fields
    from silent_cascade.eventflow.state import make_initial_continuous_state
    from silent_cascade.eventflow.neural_context import (
        workspace_from_continuous, continuous_from_workspace,
    )

    state = make_initial_continuous_state(device="cpu")
    restored = continuous_from_workspace(workspace_from_continuous(state))
    for field in fields(state):
        assert torch.equal(getattr(state, field.name), getattr(restored, field.name))
```

Add nonzero bounded tensors, device mismatch, wrong row count, NaNs and aliased
storage tests. Monkeypatch `_impulse` to raise: explicit learned-continuous jumps
must still work; the scripted path must still call it. Capture existing teacher
pre/post contexts before extraction and require identical outputs/gradients after
extraction on fixed one-/two-/four-hop batches.

- [ ] **Step 2: Run RED:** `uv run pytest -q tests/pilot/test_neural_context.py tests/pilot/test_transition_equivalence.py`.

- [ ] **Step 3: Implement public conversion and shared transition order.**

Concatenate channels in the actual `456 = 256+64+8+64+64` order. Repack and encode
only records already in `state.core.memory`; no transcript reconstruction or
future slots. Eligibility is valid, arrived, not consumed and not refractory,
never subject equality. Resolve support and active slot by immutable record IDs.
Use host differences before float32 time features. Hypothesis time is signed
`log1p(abs(predicted_deadline-now))`; absent/safe deadline contributes zero.

```python
elapsed = 0.0 if core.activation_time is None else now - core.activation_time
time_features = (math.log1p(now - init.initial_time), math.log1p(elapsed))
remaining = 0.0 if core.hypothesis is None or core.hypothesis.is_safe else (
    core.hypothesis.deadline - now
)
signed_log_remaining = math.copysign(math.log1p(abs(remaining)), remaining)
```

Do not rebuild focus or hypothesis latent from typed registers between events;
those channels retain their analytic flow. Only ACTIVATE/predicted LINK COMPOSE
may explicitly inject a learned focus embedding. Shared functions accept public
decisions; the teacher adapter constructs them only after prediction, whereas
runtime constructs them from model outputs. Reset guards after every jump.
Keep record-level consumed/refractory updates and event counters in typed jumps.

Add `models/transitions.py` to the current `PHASE3_SOURCE_PATHS` literal closure
when teacher unrolls begin importing it. Authenticate any other newly introduced
transitive dependency as well. Original closures at their historical commits and
all accepted artifact/map bytes remain unchanged. Add a regression proving that
an uncommitted transition-helper mutation is rejected by current-source
authentication; do not relax any old verifier or reinterpret historical hashes.

- [ ] **Step 4: Run GREEN:** new tests plus `tests/neural/test_unroll.py`, `test_content_unroll.py`, `test_gradients.py`, `tests/unit/test_jumps.py`, and existing scripted fixtures. Include CPU row-equivalence and native MPS when available.
- [ ] **Step 5: Commit:** `feat: share learned transitions with the event runtime`.

### Task 3: Implement the Autonomous Learned Agent

**Files:**

- Create: `src/silent_cascade/eventflow/neural.py`.
- Create: `tests/pilot/{test_neural_agent,test_neural_boundaries,test_neural_scheduler}.py`.
- Modify: typed runtime diagnostics only where a new neural failure identity is needed.

**Interfaces:**

- `NeuralModelIdentity`: frozen model config canonical JSON, weight SHA, source SHA; no data truth.
- `NeuralEventFlowAgent(model: EventFlowModel, *, identity: NeuralModelIdentity, device: str)` implements the existing `AgentCondition`; implementation ID `neural-event-flow-v1`.
- `diagnostic_snapshot()` returns immutable per-callback module/score/prediction data; it cannot create events or expose private truth.

- [ ] **Step 1: Add adversarial learned-choice tests.**

In `tests/pilot/conftest.py`, provide explicit public fixtures: LINK(1,2),
HAZARD(2,3,24), unreachable HAZARD(4,0,24), SAFE(5), ACTIVATE(1), plus negative
variants. `controlled_model` is a deterministic test double for the real neural
module interfaces, configured by logits, not an oracle-driven production agent.

```python
def test_act_uses_action_logits_not_hypothesis(runtime_case, controlled_model):
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent
    controlled_model.set_hazard_class(3)
    controlled_model.set_action_logits([0.0, 8.0, 0.0, 0.0, 10.0])
    result = runtime_case.run(NeuralEventFlowAgent(
        controlled_model, identity=runtime_case.identity, device="cpu"
    ))
    assert result.actions[0].hazard_type == 1
    assert result.score.timed_success is False
    assert runtime_case.diagnostics.raw_action_argmax == 4
```

Define `runtime_case` in the fixture module to hold the public-only controlled
policy inputs, private test bundle/engine, and captured diagnostics; only its
`run` method owns the private bundle. Test wrongly selected memories/focus/class
are never corrected; changing guard outputs alone changes timestamps.

- [ ] **Step 2: Run RED:** `uv run pytest -q tests/pilot/test_neural_agent.py tests/pilot/test_neural_boundaries.py tests/pilot/test_neural_scheduler.py`.

- [ ] **Step 3: Implement callbacks under `torch.no_grad()`.**

Initialize bounded state and public `AgentInit` with a neutral analytic segment
(latent targets equal the initial state, valid positive rates, dormant guards).
Install learned controller parameters once after each public/internal jump,
starting with the first FACT; no initial neural forward. Add a positive first-
observation-gap regression proving that the first post-FACT context matches the
unchanged teacher observer. During OBSERVING use
the model's zero-preview path; no scorer calls. On FACT, encode/inject the
delivered record and store identical perceived metadata. ACTIVATE establishes
focus/activation time through `model.observe`.

`next_internal_event` calls existing `next_crossings(state)` only: no neural
forward, horizon, jump or latent mutation. RECALL re-scores **at the flowed event
state**, resolves argmax deterministically (record ID tie-break only), applies the
learned recall jump and typed metadata. COMPOSE uses predicted role/focus/class/
deadline/confidence/support/continue values and shared tensor transitions. Never
derive relevance or targets from the selected record's kind/subject/object.
ACT requires HOLDING_HAZARD, applies the declared four-class legal decoder,
emits once at `event.timestamp`, and becomes QUIESCENT.

Decode role indices `(LINK, HAZARD, SAFE, IRRELEVANT, CONTRADICTORY)` and threshold
support/continue logits at zero; use sigmoid confidence. Predicted role is
authoritative for the typed hypothesis, status-head disagreement is diagnostic.
HAZARD enters HOLDING_HAZARD; SAFE enters QUIESCENT; other roles use the predicted
continue decision. LINK updates the focus with predicted argmax entity. Do not
turn irrelevant/contradictory finite content into an oracle-selected valid link.
Learned guard dormancy handles disconnected SEARCHING states without a fake jump.

```python
predictions = model.action(context)
raw_choice = int(predictions.class_logits[0].argmax().item())
shield = int(predictions.class_logits[0, :4].argmax().item())
# Preserve raw_choice in diagnostics; only a guard-selected ACT reaches here.
```

Structural guards may mask illegal modes/empty slots, never knowledge of the
answer. Empty learned recall, invalid schema or nonfinite predictions raise typed
errors; finite wrong predictions remain wrong. Preserve engine-owned counters.

- [ ] **Step 4: Run GREEN and spy audits.** Verify no pre-activation deliberation,
no model/agent import of oracle/training/private env, dormant direct advance,
external preemption, ties, clamps, same-kind refractory, cap and nonmutating
arbitrary-time queries. Deliberately provide inconsistent private truth in a test
owner while keeping public callbacks identical; agent prediction prefixes must
remain identical until the engine privately ends each run.
- [ ] **Step 5: Commit:** `feat: execute learned guards and cognitive jumps autonomously`.

### Task 4: Add Neural Runtime Checkpoints, Crash Recovery and Replay

**Files:**

- Create: `src/silent_cascade/eventflow/{checkpoint_state,neural_checkpoint,neural_replay}.py`.
- Modify: `src/silent_cascade/eventflow/{checkpoint,engine}.py` (narrow codec extraction and closed failure-boundary dispatch).
- Modify: `tests/regression/test_import_boundaries.py` with exact private
  checkpoint/replay-module exceptions; never exempt the learned agent or model.
- Create: `tests/pilot/{test_neural_checkpoint,test_neural_replay,test_neural_crashes}.py`.

**Interfaces:**

- `NeuralRuntimeCheckpoint`: validated immutable archive metadata and tensor
  payload described below; `NeuralReplayComparison`: exact reconstructed trace,
  decision/action/score comparison, weights hash and mismatch details.
- `snapshot_neural_runtime(session: RuntimeSession, agent: NeuralEventFlowAgent,
  *, config: EventFlowConfig, source_revision: str) -> NeuralRuntimeCheckpoint`.
- `load_neural_runtime_checkpoint(path: Path, *, expected_sha256: str,
  config: EventFlowConfig, source_revision: str,
  device: str) -> NeuralRuntimeCheckpoint`.
- `restore_neural_runtime(artifact, *, device, restore_rng=True) -> tuple[RuntimeSession, NeuralEventFlowAgent]`.
- `write_neural_replay(path, *, bundle, result, identity, config, weights) -> str` and `verify_neural_replay(path, *, weights_path: Path) -> NeuralReplayComparison`.
- New schemas `phase4-neural-runtime-v1` and `phase4-neural-replay-v1`; keep original scripted schemas/bytes unchanged.

- [ ] **Step 1: Write RED pause/resume and corruption tests.**

```python
def test_midflow_neural_resume_is_exact(neural_session_case):
    case = neural_session_case
    uninterrupted = case.finish_fresh()
    resumed = case.finish_after_pause_and_restore("between_recall_and_compose")
    assert resumed.actions == uninterrupted.actions
    assert resumed.score == uninterrupted.score
    assert resumed.trace.sha256 == uninterrupted.trace.sha256
```

The fixture must run the actual engine with a deterministic bounded model and
exercise the safe archive API, not serialize a fake result. Add corruption of
weights, aliases, segment origin/rates, queue, cached prediction, provenance,
config/source, support, initial public time, counters and tensor shapes. Invalid
archives must leave caller model/RNG/files unchanged.

- [ ] **Step 2: Run RED:** `uv run pytest -q tests/pilot/test_neural_checkpoint.py tests/pilot/test_neural_replay.py tests/pilot/test_neural_crashes.py`.

- [ ] **Step 3: Extract only shared engine-state encoding/validation.**

Keep closed explicit dispatch for scripted versus neural implementations; no
registry/plugin system. Neural archives include exact architecture/weights and
aliases, public init, continuous/segment state, all typed state/memory/refractory,
queue/cache, paused cursor, trace/trajectory anchors, authoritative and neural
counters, Python/NumPy/Torch/MPS RNG and full source/config/weight identities.
They are private artifacts; no private fields are passed back to agent callbacks.
No controller/scorer forward is needed to restore a pending segment.
The private codec may import engine/episode types through the narrowly enumerated
boundary exception; agent/model/context modules may not import that codec.
Take engine configuration as `EventFlowConfig`, model configuration from the
identity, and full experiment canonical bytes from the artifact envelope. Avoid
making the neural agent import a training configuration or checkpoint module.

```python
artifact = load_neural_runtime_checkpoint(
    path, expected_sha256=expected_sha256, config=engine_config,
    source_revision=source_revision, device="cpu",
)
session, agent = restore_neural_runtime(artifact, device="cpu", restore_rng=True)
```

Use safetensors plus bounded strict JSON, no pickle. Reuse validated no-follow
parent access/atomic creation; neural runtime archive limit 128 MiB, JSON metadata
16 MiB, with limits tested before allocation. Validate everything before restoring
global RNG. Extract common codec helpers rather than duplicating the full legacy
checkpoint implementation. Legacy scripted archives and tests must retain exact
semantics and default schema.

Extend the engine failure boundary to stage neural state before untrusted
callbacks and attach a neural checkpoint plus the last 20 sanitized events to
its crash bundle. Catch `NeuralError` at the adapter boundary and convert to a
typed `DynamicsError` preserving the cause. Publication failure remains an error.
Do not rerun the failed callback while writing the bundle. Replay uses actual
weights/engine decisions, not an intact trace's selected records or event sequence.

Staging is a lightweight runtime/RNG snapshot plus tensor-version checks, not
serialization of all model weights on every event. Authenticate frozen weights
once per evaluator and fail on their mutation. Crash bundles may reference one
hash-named regular weights blob shared within that run; authenticate it on every
restore and fail if missing. Standalone runtime checkpoints contain the weights
for portability. Never duplicate a multi-megabyte model in every event log or
silently omit a failure to save storage. Include reference resolution and model
mutation in the corruption tests.

- [ ] **Step 4: Run GREEN:** neural archive tests plus existing checkpoint,
scripted replay, RNG, crash, archive-security and pause tests. Require exact CPU
continuation; preserve same-device restoration and separate MPS tolerances.
- [ ] **Step 5: Commit:** `feat: checkpoint and replay learned event-flow trajectories`.

### Task 5: Evaluate Timed Episodes and Retain Complete Raw Evidence

**Files:**

- Create: `src/silent_cascade/eval/{runner,metrics,artifacts}.py`, `src/silent_cascade/logging/neural_trace.py`.
- Modify: `src/silent_cascade/eval/compute.py` only for tested runtime aggregation.
- Create: `tests/pilot/{test_timed_runner,test_timed_metrics,test_neural_accounting,test_pilot_artifacts}.py`.

**Interfaces:**

- `TimedEpisodeRow`: public ID, episode/weights/config/source hashes, actions,
  score fields, event counts/hashes, public predictions, error/crash reference,
  compute, and `evaluation_mode='autonomous_timed'`.
- `EvaluationIdentity`: immutable experiment/stage/split, manifest hash, ordered
  episode-hash inventory, `NeuralModelIdentity` and evaluation configuration.
- `PilotMetrics`: integer outcome/error denominators, derived rates and explicit
  pilot-gate result; `PilotEvaluation`: identity, metrics, output path and artifact
  hashes. These are strict, serializable records, not live models or sessions.
- `evaluate_episodes(model: EventFlowModel, *, identity: EvaluationIdentity,
  config: Phase4Config, episodes: Iterable[EpisodeBundle], output_dir: Path,
  device: str) -> PilotEvaluation`.
- `summarize_timed_rows(rows: Sequence[TimedEpisodeRow]) -> PilotMetrics`.
- `write_evaluation(*, identity: EvaluationIdentity,
  rows: Iterable[TimedEpisodeRow], output_dir: Path) -> PilotEvaluation` atomically
  publishes rows/index/hashes; `DONE` only for a fully completed, integrity-checked
  execution, with separate `gate_passed`.

- [ ] **Step 1: Add denominator and half-open boundary tests.**

```python
def test_failed_episodes_are_not_removed_from_denominators(scored_rows):
    from silent_cascade.eval.metrics import summarize_timed_rows
    # Fixture: positive success, positive miss, negative abstention, dynamics error.
    metrics = summarize_timed_rows(scored_rows)
    assert metrics.episode_count == 4
    assert metrics.timed_success_count == 2
    assert metrics.dynamics_error_count == 1
    assert metrics.gate_passed is False
```

Construct `scored_rows` with real `EpisodeScore` outcomes and a typed error row.
Add exact window-left success/right failure, wrong class, zero/two actions,
negative false action, duplicate/missing/public-ID substitution and unsupported
manifest tests. No survivor-only metric or average of unequal-sized batches.

- [ ] **Step 2: Run RED:** `uv run pytest -q tests/pilot/test_timed_runner.py tests/pilot/test_timed_metrics.py tests/pilot/test_neural_accounting.py tests/pilot/test_pilot_artifacts.py`.

- [ ] **Step 3: Implement autonomous evaluation and observability.**

The runner owns private episodes; construct the agent with weights/config only
and call the real engine. Score from actual emitted actions using `score_actions`.
Classify positive misses as no action / wrong class / premature / late / multiple
actions / dynamics error, retaining all raw facts needed for post-run scoring.
Never send metric feedback to the agent during an episode.

```python
episode_count = len(rows)
success_count = sum(row.error is None and row.timed_success for row in rows)
error_count = sum(row.error is not None for row in rows)
negative_count = sum(not row.is_positive for row in rows)
false_action_count = sum(not row.is_positive and bool(row.actions) for row in rows)
```

Keep compact event summaries and metrics for every episode. Retain at most 500
predeclared stratified full traces plus **every** failure, dynamics error and
delay-swap/report example. Stream completed rows and release trajectories instead
of holding 10,000 full tensor histories in RAM. Neural sidecars bind each event ID
and parent/segment hash to raw guard predictions, all valid recall scores/ranks,
compose/action predictions and compute deltas. They are observers, not callbacks
with authority to mutate state. Wall-clock telemetry is excluded from semantic
hashes, while decisions, scalar times and model/state hashes are included.

Preserve engine flow/jump counters and neural module/MAC counters separately,
then reconcile without double counting: a scalar analytic flow is counted once,
a preview scorer and later real recall are two scored passes, all memory
re-encodings are real work. Pauses/checkpoint serialization/replay costs have
separate counters. No `opportunity` ticks are added to EventFlow.

Validation errors produce error rows/crash bundles and a failing final validation
status. Numerical training corruption stops training immediately. Model candidate
validation errors can be recorded while subsequent independent episodes finish;
the trainer may continue learning from its valid optimizer state but cannot
promote/select that candidate or report the invalid evaluation as passed.

- [ ] **Step 4: Run GREEN:** runner/metric/accounting/artifact tests plus offline
spy tests proving no teacher choices, future observations or private horizon reach
agent/model calls. Verify reporting can reconstruct scores from persisted rows.
- [ ] **Step 5: Commit:** `feat: measure autonomous timed outcomes and causal compute`.

### Task 6: Build Honest Multi-Stage Pilot Data and Immutable Manifests

**Files:**

- Create: `src/silent_cascade/env/pilot.py`, `src/silent_cascade/train/pilot_data.py`, `src/silent_cascade/eval/pilot_audit.py`.
- Modify: `src/silent_cascade/train/curriculum_data.py` only to accept the shared
  `Phase1Config` base used by both Phase3Config and Phase4Config.
- Modify: `src/silent_cascade/env/leakage.py` for a narrow public shortcut-kernel
  wrapper; do not change the frozen Phase1 profile, feature schema or mathematics.
- Create: `tests/pilot/{test_pilot_projection,test_pilot_manifests,test_pilot_batches,test_pilot_audit}.py`.
- Extend: `tests/regression/test_import_boundaries.py` with the new private module.

**Interfaces:**

- `curriculum_to_bundle(example: CurriculumExample, *, config: Phase4Config)
  -> EpisodeBundle`: private, validated engine projection in `env/pilot.py`.
- `delay_pair(bundle: EpisodeBundle, *, delays: tuple[float, float])
  -> tuple[EpisodeBundle, EpisodeBundle]`: change both public hazard delays and
  corresponding private scoring truth; never modify the input bundle.
- `PilotManifest`: strict `phase4-data-v1` schema with experiment, stage, split,
  config/source/generator/transform identity, exact ordered curriculum keys,
  public/target/projected-episode hashes and allocation inventory.
- `freeze_pilot_manifest(config: ResolvedConfig[Phase4Config], *, stage: str,
  output_path: Path, source_commit: str) -> PilotManifest`.
- `load_pilot_manifest(path: Path, *, config: ResolvedConfig[Phase4Config])
  -> PilotManifest`; `iter_pilot_examples(manifest: PilotManifest, *,
  config: Phase4Config) -> Iterator[CurriculumExample]`.
- `next_pilot_batch(config: Phase4Config, *, stage: str, batch_counter: int)
  -> TrainingBatch`: uses the existing `pack_training_examples` and returns the
  next counter, never a mutable global RNG stream.
- `pilot_component_corpus(manifest: PilotManifest, *, config: Phase4Config)
  -> ComponentCorpus`: only one-hop; build existing `ComponentTarget` values with
  `build_teacher_trace`/`component_target`, never a forged Phase3 manifest.
- `audit_pilot_manifest(manifest: PilotManifest, *, config: Phase4Config,
  output_dir: Path) -> PilotAuditReport`: private oracle/structural/shortcut
  evidence with a distinct `phase4-pilot-audit-v1` identity and complete hashes.
- `audit_public_shortcuts(examples: Sequence[AuditExample], *,
  config: LeakageAuditConfig, profile: LeakageAuditProfileConfig,
  corpus_hash: str) -> tuple[ShortcutProbeResult, ...]`: new neutral wrapper in
  `env/leakage.py` reusing existing feature extraction, split/probe/permutation
  kernels; cannot publish or certify a Phase1 gate artifact.

- [ ] **Step 1: Write transform, identity and projection tests.**

```python
def test_projection_matches_transformed_public_truth(pilot_examples, pilot_config):
    from silent_cascade.env.oracle import solve_public_episode
    from silent_cascade.env.pilot import curriculum_to_bundle
    for example in pilot_examples:
        bundle = curriculum_to_bundle(example, config=pilot_config)
        assert bundle.public == example.public
        assert solve_public_episode(bundle.public) == example.solution
        assert bundle.truth.private_terminal == example.terminal_horizon
        assert bundle.truth.recipe.requested_path_length == len(example.solution.link_record_ids)
```

The `pilot_examples` fixture covers all four stages and three variants, including
robustness scales below/above 1. Add deterministic resume of next batch, distinct
train/validation/debug roots, count matching, hash corruption, duplicate keys,
symlinks, frozen-test namespace refusal and existing-manifest overwrite refusal.
Capture the current Phase3 example hashes before generalizing its type check;
those hashes must remain exactly the same afterward.

- [ ] **Step 2: Run RED:** `uv run pytest -q tests/pilot/test_pilot_projection.py tests/pilot/test_pilot_manifests.py tests/pilot/test_pilot_batches.py tests/pilot/test_pilot_audit.py tests/regression/test_import_boundaries.py`.

- [ ] **Step 3: Implement projection without falsifying generator identity.**

Recover the exact original parent using `curriculum_allocation`, the saved
`accepted_attempt` and the existing `IndependentEpisodeRequest` coordinate:

```python
root, path, variant = curriculum_allocation(example.key)
request = IndependentEpisodeRequest(
    SplitNamespace(example.key.split), SuiteName.IID_PRIMARY, root,
    example.key.episode_index * MAX_PARENT_ATTEMPTS + example.accepted_attempt,
    path, variant, example.key.episode_index // 4, example.key.episode_index % 4,
)
parent = generate_independent_episode(config, request, example.key.public_id_seed)
if episode_sha256(parent) != example.parent_hash:
    raise EpisodeInvariantError("pilot parent identity mismatch")
```

Use its coordinate as the declared **parent** coordinate. Rebuild the projected
recipe/truth with the transformed public solution, actual requested depth,
reindexed support IDs, nuisance count, private horizon and half-open window.
The new manifest explicitly binds parent identity plus curriculum transform and
projected hash; never claim that the parent key alone regenerates the projection.
For robustness, keep the ordinary IID recipe's `clock_scale=1.0`, put the actual
factor in the Phase4 envelope, and scale its oracle timing parameters consistently
with the existing transform. Do not misuse the frozen `0.1x/10x` suite schema.

Validate constructor invariants, oracle solution and temporal feasibility for
every projection, including disconnected negatives. Use the normal oracle solely
in this private producer/validation layer. Runtime imports may not reach it.

Generate four fixed production validation manifests before training. Reuse the
existing independently allocated balanced quartets, not matched positive/negative
public copies. Keep the root/public-ID key pairs in Section 3; shuffle/order and
delay marginals stay label-independent. Debug manifests use their own namespace
and 16 entries. Enforce a 64 MiB manifest read bound, strict JSON, no-follow paths,
canonical hashes and atomic create-only publication. Identical bytes may be
reused; a changed recipe requires a new identity/path, not overwrite.

For delay pairs, verify both hazard records are changed, the graph and all public
timestamps except the private outcome stay fixed, and windows are rebuilt from
the reachable terminal. Preserve parent/swap hashes in the evaluation identity.

Audit the transformed pilot data under its own profile, not by passing a one-hop
projection off as a canonical Phase1 suite. `PilotAuditReport` binds all 10,000
projected examples, exact variant/record counts, oracle 100%, feasible unique
traces, original parent regeneration and public-ID/key separation. Reuse the
existing 27 shortcut probes (three tasks by nine feature groups), grouped 80/20
split with allocation quartets kept together, 4,999 permutations, Holm alpha
`0.01`, L2 `0.03`, max optimizer iterations `500`, existing numerical tolerances
and minimum 200 held-out examples per class. The neutral wrapper uses the same
feature masking, solver and label-permutation mathematics as Phase1; extraction
regressions require unchanged Phase1 results. Statistical failures block fitting.

The Phase4 wrapper must validate the new manifest/projection itself before calling
that mathematical kernel; bypassing the old full-source authenticator is not a
shortcut to claiming a Phase1 audit. Require a named injected-feature positive
control (>=95% balanced accuracy and significant corrected result) and a shuffled-
label negative control. All injected arrays are diagnostic copies, never training
examples. Debug audit output is explicitly non-acceptance. This is a pilot check
in addition to, not a replacement for, historical Phase1 generator validation.

- [ ] **Step 4: Run GREEN:** data tests, existing Phase3 recipe regressions,
oracle 100% on generated debug projections, and shortcut probes on pilot
validation data using the existing leakage thresholds. Hashes match after a
fresh load and a changed batch counter changes only the addressed examples.
- [ ] **Step 5: Commit:** `feat: freeze reproducible autonomous pilot curricula`.

### Task 7: Explain the Action Readout Before Expensive Training

**Files:**

- Create: `src/silent_cascade/eval/action_diagnostics.py`, `scripts/diagnose_phase4_actions.py`.
- Create: `tests/pilot/test_action_diagnostics.py`.
- Document findings: `docs/phase4-autonomous-eventflow.md`, `docs/deviations.md` only for an actual correction.

**Interfaces:**

- `ActionDiagnosticReport`: source/config/weights/corpus hashes, context-labeled
  raw prediction rows, denominators and confusion matrices; explicitly diagnostic,
  never a timed pilot gate.
- `diagnose_actions(model: EventFlowModel, *, examples: Sequence[CurriculumExample],
  identity: NeuralModelIdentity, config: Phase4Config,
  output_dir: Path) -> ActionDiagnosticReport`.
- Consume the Task 3 learned agent and Task 5 autonomous evaluator. Private
  teacher contexts stay inside this diagnostic/training module, not agent code.
  Use Task 6's validated private curriculum projection for the autonomous branch.

- [ ] **Step 1: Write context-separation and denominator tests.**

```python
def test_missing_autonomous_act_remains_a_failure(diagnostic_case):
    report = diagnostic_case.run_with_dormant_act()
    assert report.autonomous.positive_count == 1
    assert report.autonomous.action_count == 0
    assert report.autonomous.missing_action_count == 1
    assert report.autonomous.timed_success_count == 0
    assert report.teacher_timed.label == "teacher_context_diagnostic"
```

Build `diagnostic_case` in this test module with the Task 3 controlled model and
one private positive curriculum example. Test a deliberately wrong shield head,
an abstain-dominant head and a model with different immediate/future predictions.
Assert that teacher time/record labels never enter the autonomous branch. A
perfect teacher-context result cannot set an autonomous success field.

- [ ] **Step 2: Run RED:** `uv run pytest -q tests/pilot/test_action_diagnostics.py`.

- [ ] **Step 3: Implement the four explicitly labeled readouts.**

For the same frozen Phase3 weights and original 10,000-example corpus, collect:

1. Historical immediate post-composition five-way action, reproducing 6,629/10,000
   and 1,641/5,000 positives under the original context recipe.
2. Teacher-timed ACT-context five-way action, using the full timed unroll.
3. Fully autonomous guard/event execution, including dormant/no-action/error
   cases and actual four-shield decoding at ACT.
4. The unmasked five-way argmax **at those same autonomous ACT states**, without
   changing their actions or counting episodes without ACT as correct abstentions.

```python
raw_choice = int(action_logits.argmax(dim=-1).item())
shield_choice = int(action_logits[..., :4].argmax(dim=-1).item())
row = {
    "context_kind": context_kind,
    "raw_five_way_choice": raw_choice,
    "legal_act_shield_choice": shield_choice,
    "logits": action_logits.detach().cpu().tolist(),
}
```

Each row also stores elapsed time, state/segment hash, predicted hypothesis class,
guard target/rate/crossing, true class for post-run scoring, and whether ACT
actually occurred. Include per-variant confusion, class margins and loss/gradient
diagnostics. Use a fixed first 128 stratified corpus entries for detailed state
trajectories; retain compact counts for all 10,000.

Load the historical weights by their accepted digest, preserving both their
training identity and the current diagnostic implementation identity. Use the
existing local artifact; if missing, report that exact dependency rather than
silently retraining Phase3 or claiming reproduction. Do not run an optimization
step in this diagnostic. Record whether the evidence supports a context mismatch,
weak classification, guard failure, transition mismatch or an unresolved mixture.
Different scores alone do not prove a cause.

If a concrete implementation defect is found, use systematic debugging, add the
smallest failing regression and fix it in its owning task. Preserve all previous
diagnostic rows. A new loss/head requires a versioned, evidence-backed amendment
within the standing approval and unchanged scientific constraints; no speculative
"force the correct class" repair. The five-way historical score stays unchanged.

- [ ] **Step 4: Run GREEN:** diagnostic tests, finite nonzero action gradients
through the complete timed unroll, and teacher/runtime boundary tests. Publish the
diagnostic before the production fit; no claim that the action problem is solved.
- [ ] **Step 5: Commit:** `feat: diagnose action decisions across timed contexts`.

### Task 8: Train the Real Curriculum with Resumable Stage Gates

**Files:**

- Create: `src/silent_cascade/train/{pilot_state,pilot_checkpoints,pilot_trainer,pilot_provenance,archive_tensors}.py`.
- Modify: `src/silent_cascade/train/{objective,checkpoints}.py` only for shared
  objective typing and safe tensor/archive helpers; preserve Phase3 wire format.
- Create: `tests/pilot/{test_pilot_curriculum,test_pilot_trainer,test_pilot_checkpoints,test_pilot_source}.py`.

**Interfaces:**

- `ObjectiveRecipe(Protocol)` in `train/objective.py`: `loss_weights: LossWeights`,
  `content_auxiliary_weight: float`, `objective_version: str`. Both existing
  TrainingConfig and new PilotTrainingConfig satisfy it; the objective's arithmetic
  remains unchanged.
- `PilotProgress`: global optimizer step, global data batch counter, stage,
  stage-start step, patience, selected/latest checkpoint identities, promotion
  certificates and immutable manifest hashes. No stage-relative budget reset.
- `PilotTrainingResult`: status/natural stop, progress, selected checkpoint
  descriptor, all journal/validation/checkpoint artifact hashes and model identity.
- `authenticate_pilot_source(*, repo_root: Path, source_commit: str,
  config: ResolvedConfig[Phase4Config]) -> PilotSourceIdentity` in
  `pilot_provenance.py`; record approved plan/spec, regular Git blobs, loaded
  module origins, production config and immutable data introductions. Task 11
  completes its explicit evidence/report inventory before a production run.
- `save_pilot_checkpoint(path: Path, *, model: EventFlowModel,
  optimizer: torch.optim.AdamW, progress: PilotProgress,
  config: ResolvedConfig[Phase4Config], source: PilotSourceIdentity) -> str`.
- `load_pilot_checkpoint(path: Path, *, expected_sha256: str,
  config: ResolvedConfig[Phase4Config], source: PilotSourceIdentity,
  device: str) -> PilotTrainingSession`; the session owns model, optimizer,
  progress and restored RNG state.
- `run_pilot_training(config: ResolvedConfig[Phase4Config], *,
  manifests: Mapping[str, Path], run_dir: Path, device: str, source_commit: str,
  resume: Path | None = None) -> PilotTrainingResult`.

- [ ] **Step 1: Write promotion, global-budget and real resume tests.**

```python
def test_stage_promotion_does_not_reset_budget(pilot_training_case):
    before = pilot_training_case.at_step(2000, stage="one_hop")
    after = pilot_training_case.promote_with_valid_rows(before)
    assert after.stage == "two_hop"
    assert after.global_step == 2000
    assert after.batch_counter == before.batch_counter
    assert after.stage_start_step == 2000
    assert after.patience == 0
```

Define `pilot_training_case` with hashed, complete synthetic validation fixtures
for policy tests, explicitly not scientific artifacts. Separate integration tests
must use actual neural updates and autonomous runner outputs. Reject fabricated
promotion booleans, missing rows, any dynamics error, wrong stage/checkpoint/hash,
stale primary results during robustness, and a 75,001st update. Exercise patience,
tie selection, no passing final checkpoint, best-three/latest-one retention,
interruption before/after a durable checkpoint and exact CPU next update.

- [ ] **Step 2: Run RED:** `uv run pytest -q tests/pilot/test_pilot_curriculum.py tests/pilot/test_pilot_trainer.py tests/pilot/test_pilot_checkpoints.py tests/pilot/test_pilot_source.py`.

- [ ] **Step 3: Implement one objective, one optimizer and one global budget.**

The actual optimization step uses the existing complete timed-plus-content graph:

```python
optimizer.zero_grad(set_to_none=True)
objective = training_objective(model, batch, config.config.pilot)
if not torch.isfinite(objective.total):
    raise TrainingError("nonfinite pilot objective")
objective.total.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
optimizer.step()
```

Check all updated parameters and AdamW tensors for finiteness before publishing
the step. Log every component loss by trace position, gradient diagnostics,
parameter count, batch hash, stage, counters and update number. Do not detach the
teacher trajectory or reveal teacher choices before predicting the current step.

Use Task 6 batches and the exact Section 3 curriculum. Bootstrap by verifying the
oracle/leakage evidence, then fresh seed-11 one-hop training. One-hop promotion
uses the existing content evaluator on the fixed pilot corpus. Later promotions
consume Task 5 raw autonomous evaluations, not a copied component metric. Evaluate
both primary and robustness on the **same current weights** during robustness.
Do not replace this with "primary_complete=True" supplied by a caller.

Use a distinct `phase4-pilot-training-v1` archive. Factor only reusable safe
safetensors/optimizer/RNG reading into `archive_tensors.py`; leave the Phase3
schema, config parser and authenticity rules intact. Bind actual tensor inventories
and shared aliases, optimizer groups, CPU/MPS RNG, full canonical Phase4 config,
source identity, global progress and promotion hashes. Restore transactionally:
validation must finish before changing global RNG or a live model.

Publish durable checkpoints at validation/promotion/final boundaries, with an
initial checkpoint and append-only metric journal. An unexpected restart resumes
the last durable checkpoint and deterministically regenerates subsequent batches;
mark any abandoned journal tail as uncommitted work instead of counting duplicate
updates as new evidence. Preserve failures and source identities. A changed source
or config is a different attempt, not an in-place resume.

Authenticate executing source/config before loading training data and again at
durable publication. Accept only approved source/plan identities, no environment
variable to bypass checks. Debug profiles can run short tests but cannot generate
a production promotion or accepted gate. Only Task 12 starts production training.

- [ ] **Step 4: Run GREEN:** trainer/curriculum/checkpoint tests, exact two-step
CPU interrupted/uninterrupted update comparison, original Phase3 checkpoint and
objective regressions. A tiny 64-example debug overfit run is diagnostic: show
actual class/timing/guard losses and autonomous progress, never substitute it for
the 10,000-episode promotion. Failure to learn triggers debugging before Task 12.
- [ ] **Step 5: Commit:** `feat: train resumable autonomous EventFlow curricula`.

### Task 9: Expose Working Local Commands and an Artifact-Only Pilot Report

**Files:**

- Create: `src/silent_cascade/train/pilot_cli.py`, `src/silent_cascade/report/{__init__,pilot}.py`, `scripts/run_pilot.py`.
- Create/update: `docs/phase4-autonomous-eventflow.md`, reader-facing command status in `README.md`.
- Modify: `src/silent_cascade/cli.py`, `Makefile` and their existing command/package tests.
- Create: `tests/pilot/{test_pilot_cli,test_pilot_report,test_pilot_workflow}.py`.

**Interfaces:**

- `register_pilot_commands(app: typer.Typer) -> None`: register working train,
  evaluate and a new report group with its build command. Extend the existing
  root `freeze` callback with `--pilot-stage` dispatch to Task 6's producer; do
  not register a second command named `data freeze`. Extend existing replay with
  closed schema dispatch for neural files, preserving its scripted path.
- `build_pilot_report(*, run_dir: Path, output_dir: Path) -> Path`: read verified
  raw artifacts only; return generated `report.md`.
- `run_pilot(*, config_path: Path, manifest_dir: Path, run_dir: Path,
  device: str) -> PilotTrainingResult`: preflight identity, immutable validation
  data/oracle/leakage, train/resume, selected-weight evaluation and pilot report.
  Task 11 adds final evidence collection through its implemented API.

- [ ] **Step 1: Write real CLI dispatch, reuse and report fixture tests.**

```python
def test_report_does_not_import_or_call_training(pilot_artifacts, monkeypatch):
    from silent_cascade.report.pilot import build_pilot_report
    import silent_cascade.train.pilot_trainer as trainer
    def forbidden(*args, **kwargs):
        raise AssertionError("reporting must not train")
    monkeypatch.setattr(trainer, "run_pilot_training", forbidden)
    report = build_pilot_report(
        run_dir=pilot_artifacts.run_dir, output_dir=pilot_artifacts.report_dir,
    )
    assert report.is_file()
    assert "Autonomous timed pilot" in report.read_text()
```

Create `pilot_artifacts` with real fixture evaluations and explicit fixture-only
identity. Check numeric regeneration against raw rows, error/missing artifact
refusal, malicious hashes/paths, negative conclusion and no canned metrics/plots.
CLI tests invoke the smoke trainer and real miniature evaluator, not mocks claiming
the command worked. Preserve all existing Phase1/2/3 commands and help behavior.

- [ ] **Step 2: Run RED:** `uv run pytest -q tests/pilot/test_pilot_cli.py tests/pilot/test_pilot_report.py tests/pilot/test_pilot_workflow.py`.

- [ ] **Step 3: Wire only commands with working implementations.**

Implement these exact future command forms; output/help must distinguish pilot
validation from frozen tests and reject unsupported condition/config combinations:

```bash
uv run silent-cascade data freeze --pilot-stage primary --config configs/train/pilot.yaml --output manifests/validation/phase4/primary.json
uv run silent-cascade train --config configs/train/pilot.yaml --manifest-dir manifests/validation/phase4 --run-dir runs/phase4-pilot-v1/event_flow/11/pilot --seed 11 --device cpu
uv run silent-cascade evaluate --checkpoint runs/phase4-pilot-v1/event_flow/11/pilot/selected.json --manifest manifests/validation/phase4/primary.json --output runs/phase4-pilot-v1/event_flow/11/pilot/eval/primary --device cpu
uv run silent-cascade report build --pilot --run-dir runs/phase4-pilot-v1/event_flow/11/pilot --output reports/phase4-pilot-v1
```

`selected.json` is a hash-bound descriptor published by the trainer, not an alias
to whichever file happens to be newest. `train --resume PATH` accepts a specific
verified checkpoint. `--config` resolves only the strict supported overlay recipe;
record explicit device/seed choices in run metadata and reject seed disagreement.
Do not create no-op commands for later phases, interventions or generic reporting.

Add `make pilot` for the full local one-seed workflow and `make pilot-smoke` for a
16-example debug workflow. Expose `PILOT_RUN_DIR` and `PILOT_DEVICE`, defaulting to
the ignored example run path and CPU; resolve paths without writing private paths
into committed artifacts. Keep the current isolated test processes. A pilot
target never calls final `freeze-tests` or accesses `manifests/frozen/`.
Smoke runs stop at four updates and test finite optimization, artifact creation,
real evaluation/replay and reporting. Their execution status is distinct from an
unmet production learning gate; they cannot promote a production curriculum or
publish accepted evidence. Production fitting requires the four validation
manifests already introduced in Git, as Task 12 specifies. If absent, create the
requested validation files and stop before fitting with an explicit commit-data
precondition, rather than silently training on an unrecorded freeze.

The orchestrator must refuse concurrent writers via an ownership record containing
PID, process start identity and run identity. An existing live matching process is
reported, not duplicated. A stale record after a reboot is recoverable only after
checking process identity and the last durable checkpoint. Hash-compatible finished
steps are reused; incompatible directories fail with a new-run-path instruction,
never overwrite. A failed learning gate produces a complete adverse report and
nonzero overall status; it does not run Phase5 or start another seed.

Generate small tables for class/variant/timed success, errors, event counts,
compute and action diagnostics; positive/negative timelines and timing-error/guard
plots from actual retained trajectories. Label all output pilot validation evidence,
including training exposure and selection. No baseline/continuity/novelty win.
Check each rendered table back against its raw integer denominators and hashes.

- [ ] **Step 4: Run GREEN:** real CLI smoke, report fixtures, idempotent restart
tests and local package build. All documented commands must work when their stated
input artifacts exist; missing inputs fail explicitly rather than fabricate data.
- [ ] **Step 5: Commit:** `feat: expose local autonomous pilot workflow and report`.

### Task 10: Verify Real Device Behavior and Measure the Workload

**Files:**

- Create: `src/silent_cascade/train/pilot_verification.py`.
- Create: `tests/pilot/{test_pilot_numerics,test_pilot_offline,test_pilot_throughput}.py`.
- Record measured preflight: `docs/phase4-autonomous-eventflow.md`.

**Interfaces:**

- `verify_pilot_numerics(config: ResolvedConfig[Phase4Config], *,
  checkpoint: Path | None, output_dir: Path) -> PilotNumericReport`: complete
  named output/loss/gradient/update inventories, source/config/weights/batch hashes,
  CPU and native-MPS resume results, tolerances and missing-device status.
- `profile_pilot(config: ResolvedConfig[Phase4Config], *, output_dir: Path)
  -> PilotThroughputReport`: measured train/update and real scalar-runner evaluation
  throughput, synchronized device timing, memory/disk forecast and chosen device.
- Both strict reports carry `evidence_kind`; throughput/debug measurements cannot
  satisfy training, accuracy or final-checkpoint device gates.

- [ ] **Step 1: Write inventory completeness and offline tests.**

```python
def test_missing_gradient_inventory_is_not_device_parity(numeric_report_case):
    report = numeric_report_case.with_gradient_removed("action_head")
    with pytest.raises(ValueError, match="inventory"):
        report.validate_complete_inventory()
```

Define `numeric_report_case` from an actually enumerated model state/parameter
inventory, not one sentinel tensor. Test CPU exact resume, unavailable MPS as
explicit missing evidence, changed tolerance/config/source rejection and behavior
disagreement even when aggregate accuracy matches. Block sockets/download helpers
and optional package imports around tiny training, evaluation, replay and report;
assert their foundation-model call counters are exactly zero.

- [ ] **Step 2: Run RED:** `uv run pytest -q tests/pilot/test_pilot_numerics.py tests/pilot/test_pilot_offline.py tests/pilot/test_pilot_throughput.py`. Native MPS evidence is collected locally on this machine.

- [ ] **Step 3: Implement synchronized, bounded actual-work measurements.**

Use the actual batch-128 combined objective for both one-hop and primary four-hop
max-trace batches. Start device comparisons from the same weights, batch, optimizer
and RNG state. Check **every** loss component, named parameter gradient, updated
parameter and optimizer state using Section 3 tolerances; missing entries fail.
Verify same-device MPS checkpoint continuation and exact CPU continuation. Later
repeat with the selected pilot weights; initial smoke is not final-weight evidence.

```python
if device == "mps":
    torch.mps.synchronize()
started = time.perf_counter()
result = measured_operation()
if device == "mps":
    torch.mps.synchronize()
elapsed_seconds = time.perf_counter() - started
```

Here `measured_operation` is a no-argument callable for the real full training
update or complete scalar-engine evaluation, not a surrogate GEMM. For each CPU
(`OMP_NUM_THREADS=1`) and native MPS trial, use 16 warmup and 32 measured updates
in a distinct debug run, plus 256 manifest-addressed autonomous evaluation episodes.
Count these diagnostic operations and do not credit their weights to the pilot.
Do not enable silent MPS fallback. If CPU is faster, keep CPU; use MPS for the fit
only if the measured full workload is faster and numerical checks pass. CPU remains
the final acceptance/replay device regardless.

Report compute-only projection for the chosen device using measured update cost,
10,000-episode validation cost, scheduled validation count up to 75,000 global
steps, dual primary/robustness validations and final repeat/pair/replay work. Show
the range rather than promising convergence or a deadline. Measure retained raw
artifact bytes from the sample; require free space above 1.2 times projected
retention plus checkpoint working space. Do not discard failures to fit a budget.

If scalar validation is prohibitively slow, profile conversion/scoring/serialization
separately. An optimization must have decision, timestamp, counter and trace
equivalence tests before use. Do not silently substitute a second event engine or
reduce corpus sizes/frequency. Record a reviewed plan adjustment if needed.

- [ ] **Step 4: Run GREEN:** local CPU/MPS checks, complete inventories and offline
tests; run `make verify` before production preflight. Publish real measured compute
expectations. No concurrent training job should exist when the pilot starts.
- [ ] **Step 5: Commit:** `test: verify autonomous pilot numerics and resource use`.

### Task 11: Authenticate the End-to-End Gate Independently

**Files:**

- Create: `src/silent_cascade/train/{pilot_evidence_types,pilot_evidence}.py`.
- Create: `scripts/check_phase4_pilot.py`, `scripts/verify_phase4_gate_artifact.py`.
- Modify: `src/silent_cascade/train/pilot_provenance.py`, `scripts/run_pilot.py`,
  `tests/integration/test_phase0_repository.py` for the actual Phase4 delivery guard.
- Create: `tests/pilot/{test_pilot_evidence,test_pilot_gate_verifier,test_pilot_delivery}.py`.

**Interfaces:**

- `Phase4GateArtifact`: closed `phase4-autonomous-gate-v1` schema with source,
  approved plan/spec/config/data/weights hashes; complete expected workload;
  original integer metrics, repetition/pair/replay/device/import/compute evidence;
  upstream artifact hashes, failures, zero foundation calls and explicit outcome.
- `collect_pilot_evidence(*, run_dir: Path, config: ResolvedConfig[Phase4Config],
  output_path: Path) -> Phase4GateArtifact`: authenticates actual raw execution
  artifacts; cannot manufacture outcomes from a config or cached success flag.
- `verify_phase4_gate_artifact(artifact_path: Path, *, repo_root: Path,
  raw_run_dir: Path | None = None) -> dict`: independently validates source,
  introduction history, recomputed aggregates and all gate conditions. With raw
  data present, rehash and recompute every referenced row; portable verification
  must say explicitly which raw attachments were unavailable and cannot claim a
  full reproduction without them.
- `check_phase4_pilot.py`: CLI entry to complete selected-weight evaluation,
  repeat, delay pairs, runtime checkpoint/replay, final numerics and collector;
  uses actual Task 5/10 APIs, does not rerun training.

- [ ] **Step 1: Write adversarial collector/verifier tests.**

```python
def test_rehashed_gate_cannot_omit_a_failed_episode(gate_artifact_case):
    artifact = gate_artifact_case.remove_failure_and_rehash()
    with pytest.raises(ValueError, match="episode inventory"):
        gate_artifact_case.verify(artifact)
```

Create `gate_artifact_case` as a small unmistakably fixture-only execution; the
production verifier must reject its profile/counts as real acceptance. Test
coordinated row/summary/hash substitution, stale weights, wrong introduction commit,
foreign loaded package, omitted source file, modified approved plan, source change
mid-run, wrong variant denominators, conditional-only success, untested MPS tensors,
missing CPU replay, hidden failed delay pairs and a debug run relabeled production.

- [ ] **Step 2: Run RED:** `uv run pytest -q tests/pilot/test_pilot_evidence.py tests/pilot/test_pilot_gate_verifier.py tests/pilot/test_pilot_delivery.py`. Preserve historical Phase1/2/3 rejection tests, not only their passing pinned-checkout path.

- [ ] **Step 3: Implement a complete source and artifact chain.**

Finalize the literal source inventory covering the actual engine/jumps/context,
model/memory/flow/heads, objective/curriculum/optimization/checkpoint, configs,
evaluator/metrics/projection, report, CLI, collector/verifier and approved plan/spec.
The independent verifier derives/checks the required inventory independently of
the producer's "passed" flag. Authenticate regular tracked bytes and actual loaded
module origins; reject dirty relevant source and symlinks. Require the effective
approved plan at or before the source commit and immutable manifest introduction
before the first optimizer update. No accepted-result rehashing at a new source.

```python
if observed_episode_ids != expected_episode_ids:
    raise ValueError("episode inventory mismatch")
successes = sum(row.timed_success for row in verified_rows)
if successes != artifact.primary.timed_success_count:
    raise ValueError("primary success aggregate mismatch")
```

Recompute timing/class/negative outcomes from raw actions plus regenerated private
episode truth, never trust the saved success boolean alone. Verify ordering,
denominators, all errors, exact repeated primary traces, all 256 delay pairs,
actual selected-checkpoint replay and numerical tensor inventories. Pair success
requires both actions, both classes and both half-open windows. All claimed model
results must name the same selected weights; earlier-stage promotion checkpoints
are separate and cannot be substituted for final two-hop/primary evidence.

Store compact per-episode rows and replay identities as hashed attachments;
streaming gzip/JSON must have documented decompressed size/row limits and reject
duplicate keys/nonfinite values. Commit modest manifests/summary/gate/report files,
including the compressed acceptance rows needed to recompute every published
count and repeat/pair comparison. Do not commit large raw latent traces or normal
checkpoints. Set compact-row limits to 50,000 rows and 128 MiB decompressed per
attachment; no committed individual file may reach 100 MiB. List SHA-256 and
regeneration commands for external raw attachments. Portable verification proves
recorded-evidence integrity, not a fresh neural rerun; actual replay status must
name its run/weights or explicitly be not rerun. `DONE` means execution completion,
while gate status can still be failed. Never collapse these two concepts.

Add the Phase4 delivery-map schema and guard now, but leave `docs/PLAN.md` incomplete
until Task 12 has real passing evidence. The guard must reject a premature Complete
row, a missing gate or mismatched source/weights/hash. Existing complete rows and
original artifact bytes stay pinned through Task 1's historical verifier.

- [ ] **Step 4: Run GREEN:** all tamper tests and a complete tiny debug pipeline
whose gate is correctly labeled non-acceptance. Run full local `make verify` and
task-scoped spec/quality review. Resolve source changes before production freeze.
- [ ] **Step 5: Commit:** `feat: authenticate autonomous pilot acceptance evidence`.

### Task 12: Run the One-Seed Pilot and Publish the Actual Outcome

**Files:**

- Generate: `manifests/validation/phase4/{one_hop,two_hop,primary,robustness}.json`.
- Generate: `manifests/validation/phase4/autonomous-gate-v1.json`,
  `manifests/validation/phase4/delivery.json`,
  `manifests/validation/phase4/compact/` acceptance rows, `reports/phase4-pilot-v1/`.
- Update from evidence: `docs/phase4-autonomous-eventflow.md`, `docs/PLAN.md`,
  `docs/deviations.md` if needed and the command/result status in `README.md`.
- Retain uncommitted large raw artifacts/checkpoints in the configured run directory.

**Interfaces:** Consume the actual config/data/trainer/evaluator/replay/report/
verification commands implemented by Tasks 1–11. No new learning algorithm or
acceptance criterion is introduced during this task.

- [ ] **Step 1: Verify the executable preflight before starting the long run.**

Check clean relevant source, approved effective plan, all prior task reviews,
historical acceptance, Task 7 action diagnosis, local `make verify`, native smoke,
resource forecast and no other active compute owner. Capture all stdout/stderr,
exit codes, environment, source/config and available hardware. A completed
Phase3 worker name is not evidence that an old process is still running; inspect
actual process/run ownership before resuming or starting anything.

Commit the complete reviewed executable source first. Generate **all four**
production validation manifests from it, run their oracle/shortcut audits, then
commit them separately without changing their bytes. Record both source and
manifest-introduction commits. This is a pilot validation freeze, not the Phase6
architecture/test freeze and not the `experiment-v1-freeze` tag.

- [ ] **Step 2: Start or resume exactly one authenticated pilot.**

```bash
OMP_NUM_THREADS=1 make pilot PILOT_DEVICE=cpu PILOT_RUN_DIR=runs/phase4-pilot-v1/event_flow/11/pilot
```

Use `PILOT_DEVICE=mps` only if Task 10 selected it from real measurements. Record
the choice before fitting, and keep acceptance on CPU. Fresh initial seed is 11;
global step, batch counter and curriculum stage begin at zero/one-hop. All stages
share the 75,000-step ceiling and declared stop rules. Do not preload the old
Phase3 component checkpoint into this production run.

Monitor durable progress and process health, not just an idle agent status. After
an interruption/reboot, inspect the run owner and checkpoint hash, then resume the
same run only if the previous process is gone. Never duplicate active fits, delete
the last checkpoint, restart a fresh seed silently or pretend a stopped process
completed. Keep user updates concise while the real compute is running.

- [ ] **Step 3: Evaluate the selected frozen weights and collect real evidence.**

```bash
uv run python scripts/check_phase4_pilot.py --run-dir runs/phase4-pilot-v1/event_flow/11/pilot --output manifests/validation/phase4/autonomous-gate-v1.json
uv run python scripts/verify_phase4_gate_artifact.py --artifact manifests/validation/phase4/autonomous-gate-v1.json --raw-run-dir runs/phase4-pilot-v1/event_flow/11/pilot
uv run silent-cascade report build --pilot --run-dir runs/phase4-pilot-v1/event_flow/11/pilot --output reports/phase4-pilot-v1
```

Run the complete Section 3 acceptance workload: final primary, two-hop and
robustness; exact primary repeat; 256 fixed delay pairs; real paused-world
checkpoint/replay; selected-weight native CPU/MPS checks; offline, compute and
identity checks. Reuse only byte-verified compatible completed steps. Checkpoint
selection precedes these final analyses and cannot be changed after a bad delay
swap or device result.

If training naturally stops below its gate, collect its failed outcome, losses,
errors and diagnostic report; commands return a nonzero gate status. Diagnose a
reproducible implementation defect before any fix. A source/config change must be
committed, reviewed and recorded as a new attempt identity, with invalidated
evidence regenerated; no old pass is carried across changed executable code.
An inconclusive/negative learning result is honest progress, not Phase4 completion.
Do not quietly expand compute, lower gates, select on final diagnostics or proceed
to strong-baseline training while this phase gate is unsatisfied.

- [ ] **Step 4: Verify the delivery and run final whole-phase review.**

```bash
make verify
git diff --check
uv run python scripts/verify_phase4_gate_artifact.py --artifact manifests/validation/phase4/autonomous-gate-v1.json --raw-run-dir runs/phase4-pilot-v1/event_flow/11/pilot
```

Review the final diff against the canonical spec and this plan. Resolve real review
findings through TDD and rerun every affected verification; relevant source changes
require new source-bound evidence. Confirm no unfinished required commands, fake
plots/values, hidden network use, optional model imports or CI/CD were added.

Only a passing gate permits the Phase4 delivery row to say Complete, with actual
source, gate and selected-weight hashes and **autonomous pilot engineering** claim
boundary. Preserve every negative diagnostic and all prior phases' historical
rows. If the gate fails, document the observed failure and remaining work instead.

- [ ] **Step 5: Commit the meaningful evidence and documentation:** use
`docs: record verified autonomous EventFlow pilot` only if verified; otherwise
`docs: record autonomous pilot results and remaining gate failures`.
After the commit, confirm a clean intended worktree and recheck the delivery guard.
Do not push, create a result-release tag, or begin Phase5 without its own plan.

## 5. Verification Commands and Execution Boundaries

The commands inside Tasks 1–12 are **implementation instructions**, not claims
that Phase4 commands already exist today. During execution run targeted RED/GREEN
tests after every task and task-scoped reviews before accepting it. The complete
local quality gate remains:

```bash
uv sync --locked --group dev
uv run silent-cascade doctor
make verify
```

Resolve Python/dependencies from the committed lockfile; do not update them just
to execute this plan. Use local tooling only. Production data/training/evaluation
must work with network disabled. A report reads artifacts and cannot start a fit.
The pilot orchestrator is the single process owner; independent review agents may
inspect artifacts or code but must not launch competing training jobs.

For the implementation handoff, prefer the user's established **subagent-driven
execution on the current branch**: one fresh implementer per task, followed by
spec and quality reviews. Sequential dependencies in the file map remain real;
"use subagents" is not permission to have multiple writers edit shared modules
simultaneously. No subagent execution is started by the act of writing this plan.

## 6. Completion Checklist and Specification Coverage

- [ ] Historical Phase1/2/3 acceptance is immutable and explicitly verified at its
  pinned source; current-source validation is not weakened.
- [ ] Learned runtime uses analytic guard crossings and shared neural transitions;
  no scripted impulses, private truth, fixed ticks or action copying.
- [ ] The old 66.29% readout is explained by measured context-specific diagnostics,
  or remaining uncertainty is stated honestly; real timed performance is reported.
- [ ] Four immutable validation curricula, fresh one-seed global-budget training,
  promotion/selection/stop/resume and every attempted result are authenticated.
- [ ] The same selected checkpoint meets every final Section 3 gate with complete
  denominators, CPU repeat/replay, delay-following and native device evidence.
- [ ] All intended CLI/Make workflows run locally; no CI/CD, Qwen or network enters
  the scientific path; foundation-model call count is exactly zero.
- [ ] Every published pilot value/plot comes from raw artifacts with hashes; failed
  examples and negative diagnostics remain visible.
- [ ] Full local verification, independent artifact verification and final review
  pass before the delivery row changes to Complete.

| Canonical requirement | Phase4 coverage / deliberate later boundary |
| --- | --- |
| Sections 4–7: OFD, public/private separation, typed dynamics and memory | Tasks 2–6; existing generator/engine semantics preserved |
| Section 8: supervised traces, autonomous evaluation, curriculum, checkpoints | Tasks 6–8; existing objective retained, timed rollout measured |
| Section 9.4: complete compute accounting | Tasks 3 and 5; competitive matching/controls stay Phase5 |
| Sections 10–11: frozen tests, scientific support, full causal/statistical analysis | Not claimed; Phase4 uses validation engineering gates and narrow delay swaps only |
| Section 12: numerical, scheduler, memory, shortcuts, checkpoint and offline integrity | Tasks 2–8 and 10–11, plus existing regression/property coverage |
| Sections 13–16: bounded local resources, commands, hashed artifacts, reporting | Tasks 1, 4–5 and 8–11; small pilot report, not final scientific report |
| Section 17 Phase4: autonomous guards/recall/compose/action, quiescence, pilot gate | Tasks 3, 5, 8 and 12; >=90% IID, <=10% false actions, zero dynamics failures |
| Section 18: unit/property/integration/resume/replay/import tests | Task-scoped RED/GREEN tests and final local `make verify` |
| Sections 19–24: optional showcase, public claims, release, research update | Deferred to their phase plans; no Qwen, novelty or comparative claim here |

**Planning self-review:** Completed against the canonical Phase4 scope and current
interfaces, including task dependencies, immutable historical evidence, public/
private boundaries, action decoding, global training budget and unchanged pilot
gates. The user subsequently approved execution in the task thread on 2026-09-16.
Approval does not establish any implementation, training or acceptance result.

**Execution clarification (2026-09-16):** A read-only fixed-seed reproduction
confirmed that the originally proposed pre-first-FACT controller call would
produce a runtime-only latent change: at first gap `0.5731144887975146`, the
maximum absolute pre-FACT latent was `0.28444260358810425`, versus `0.0` in
`train/observations.py`. Initialization is therefore explicitly neutral until
the first public event, matching the existing teacher path. This correction is
authorized by the standing in-scope plan-adjustment policy; it changes neither
historical training behavior nor post-activation dynamics or acceptance gates.

**Source-closure clarification (2026-09-16):** Task 2 introduces an executable
dependency into the legacy teacher path. Current-source authentication must grow
its literal closure to cover that dependency. Preserving historical closure
bytes means preserving their original Git revisions and artifacts, not leaving
new current execution dependencies unauthenticated. This additive correction
changes no historical result, schema, training arithmetic or acceptance gate.
