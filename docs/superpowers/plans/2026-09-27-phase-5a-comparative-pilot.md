# Phase 5A Comparative Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended for this bounded, sequential plan) or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Obtain an inexpensive, reproducible development comparison of the accepted EventFlow checkpoint, its compressed-time intervention, and, if the first results justify the cost, one separately trained activation ponderer.

**Architecture:** Preserve the accepted Phase 4 execution and artifact formats. Add a small comparison driver with public-input-only condition adapters, paired diagnostic manifests, measured compute, and artifact-derived reports. Execute the checkpoint-only milestone before implementing and training the competitive ponderer; its architecture, losses, selection rule, and resource limits are fixed by this plan before viewing diagnostic results.

**Tech Stack:** Existing locked Python 3.12, float32 PyTorch on CPU, NumPy, Pydantic, safetensors, pytest, and uv. No new dependency, network service, hosted automation, or foundation model.

**Spec:** [Canonical design v1.0.2](../specs/2026-08-30-silent-cascade-design.md), especially Sections 4, 8, 9.1–9.5, 10.1/10.3/10.9/10.11/10.12, 11.1, 16, and Phase 5 in Section 17. The canonical specification is unchanged; the smaller corpora and resource stops below belong to a separate exploratory pilot, not replacement acceptance gates.

**Status:** DRAFT — awaiting explicit user approval. The user authorized writing this plan and waiting on 2026-09-27. No implementation or experiment execution is authorized by that instruction.

**Starting checkout:** `main` at `f16e4d1f039bd5eaaa672c8f7158ac90bbf9b81b`. Two existing untracked `manifests/validation/phase4*/audit-raw/` trees are retained evidence. Work on the current branch, with meaningful local commits.

## Global Constraints

- Package `silent_cascade`; executable `silent-cascade`. Primary scientific execution stays offline with exactly zero foundation-model calls.
- CPU is the pilot execution device; float32 model tensors and host float64 absolute times. Disable MPS fallback. Do not turn this pilot into a new CPU/MPS replication campaign.
- At most 5,000,000 trainable parameters, including embeddings; 64 memory records; 64 internal events for intact EventFlow. Preserve its flow, guard, mode, provenance, and replay rules.
- Preserve the positive metric: exactly one correct-class action in `[t0 + 0.75*d, t0 + 0.90*d)`; negatives require no action. Errors remain in the denominator.
- Conditions receive only public initialization, delivered FACT/ACTIVATE events, and state derived from those inputs. Private outcome time, graph truth, generator keys, labels, teacher choices, and scoring windows remain outside inference.
- No deliberate recall/composition before activation, hand-coded graph traversal in a learned policy, exact-subject retrieval mask, or copying teacher answers at inference.
- Ponderer budgets are exactly `{4, 8, 12, 16, 24}`; training samples them uniformly; deterministic halting uses probability `>= 0.5`; maximum 24 recurrent transitions.
- Competitive recurrent parameter count must be within ±10% of EventFlow. Count actual trained parameters, all retrieval scores, recurrent transitions, readouts, halting work, and initialization.
- Canonical training remains AdamW, learning rate `3e-4`, weight decay `1e-4`, batch 128, gradient clip 1.0, ceiling 75,000 updates, validation every 1,000 updates on 10,000 examples, patience 15, best three checkpoints plus latest. Preserve the recorded epsilon `1e-6`, betas `(0.9, 0.999)`, `foreach=False`, `fused=False`, and no scheduler.
- The additional execution budget below can stop a pilot before canonical training completion. Such a stop is explicitly resource-limited, cannot establish baseline inferiority, and cannot pass the full Phase 5 gate.
- No final frozen-test generation/access, extra scientific seed, OOD training, hyperparameter search, one-shot/scheduled controls, full causal study, final scientific claim, or Phase 6/7 work in this plan.
- Use TDD for implementation, systematic debugging for unexpected behavior, the applicable execution/review skills, and local `make verify` before software delivery. No CI/CD. Record failures and material rulings in the execution ledger.

## Review Focus

1. Compressed execution accidentally changes weights, consumes an intact trace, advances cognition time, or disables record legality: Task 3 tests compare state/weight hashes and adversarial choices.
2. Private terminal information or oracle labels reach a condition through a convenient evaluator object: Task 2 tests use a spy condition and paired private-terminal changes.
3. Cheap stopping, selective failures, or an undertrained ponderer produce an apparent EventFlow win: Tasks 4 and 7 test resource-stop statuses, complete denominators, and restricted conclusions.
4. Budget, observation initialization, GRU, retrieval, or halting work is omitted from compute: Tasks 2 and 5 test known small operations and all executed branches.
5. Diagnostic OOD results influence baseline design, selection, or continuation: Tasks 1, 6, and 7 pin the protocol hash and restrict model selection to the existing IID validation corpus.

---

## 1. Scientific Questions and Delivery Boundary

**Milestone A — existing checkpoint, no optimization:**

1. Does the accepted EventFlow model solve fresh IID episodes and unseen path lengths 5–8?
2. Does the same checkpoint still solve them when its autonomous recall/composition work occurs at activation time?

**Milestone B — one competitive model, conditional:**

3. Can a separately trained adaptive ponderer achieve similar timed success with overlapping measured compute?

The outcomes may be promising, negative, or inconclusive. Successful implementation means trustworthy execution and reporting, not an EventFlow victory. Phase 5 remains incomplete after this plan: both one-shot sizes, scheduled controls, other ablations, and the full Phase 5 fairness gates still require subsequent work.

The generated final report must explicitly distinguish same-checkpoint causal dependence from competitive architectural merit. A compressed-time loss can reflect weights trained for a different regime; a poorly trained competitor cannot establish an architectural advantage.

### 1.1 Starting checkpoint and provenance

| Object | Required identity |
| --- | --- |
| Accepted training source | `3b132253512f02c0ed9f2d774fefce3036019d91` |
| Canonical Phase 4 gate | `manifests/validation/phase4/autonomous-gate-v1.json` |
| Gate SHA-256 | `6353acb150fc5f46d10215e5b4206f08744818184cce3933f6d5b33e0eb75ab1` |
| Selected portable weights SHA-256 | `bc2850deb2bacb64d336750c5e824390e4a8eb45a3144467e3a61af84c49285b` |
| Logical model-state SHA-256 | `2c5e5b2a0cb3749404ec3f2a5b70dc20c00fd8a2560a4035f577311fa375e30f` |
| Selected update / seed | 12,000 / 11 |
| Reference parameter count | 2,781,042; rederive from the authenticated model |

Resolve the retained run through an explicit `--phase4-run-dir` argument, never a committed machine-specific path. Authenticate the delivery, selected descriptor, portable bytes, logical state, and recorded local-verification receipt. Bind the old producer revision and the new executing revision separately. Do not require the old receipt to certify new code, rewrite historical evidence, or repeat Phase 4 fitting/audits/full raw verification as a routine preflight.

Authenticate the historical receipt against its producer Git blobs and bound logs. The current-tree `discover_local_verification` check intentionally rejects changed execution inputs; do not relax that validator or present a historical receipt as verification of the new comparison code.

### 1.2 What is already reusable

- `eventflow/neural.py`: `NeuralEventFlowAgent`, including public observation and learned jump execution.
- `eventflow/engine.py`: `EventEngine`, private terminal ownership, analytic advancement, intact execution.
- `eventflow/guards.py`: host crossing calculation and legal candidate ordering.
- `env/generator.py`: `IndependentEpisodeRequest` and `generate_independent_episode`.
- `env/reward.py`: `score_actions`; `train/curriculum_data.py` and `train/batches.py`: teacher-only data and packing.
- `eval/compute.py`: `NeuralComputeMeter`, parameter counts, operation estimates; `io.py`, hashing, safetensors, and existing checkpoint primitives.

The existing `eval/runner.py`, `TimedEpisodeRow`, and `RuntimeCompute` describe the closed Phase 4 pilot. Do not relax their seed/purpose/opportunity restrictions to disguise new results as old pilot evidence. Use new comparison records that reuse scoring and counters where compatible.

## 2. Fixed Data and Analysis Protocol

### 2.1 Diagnostic manifests

Create a versioned `phase5a-diagnostic-v1` manifest schema under `manifests/validation/phase5a-v1/`; its generator split is **DEBUG**, purpose `exploratory_comparison`, and gate eligibility is always false. Do not change `ManifestSizesConfig` or create anything under `manifests/frozen/`.

| Manifest | Episodes / path allocation | Root / public-ID seed | Purpose |
| --- | --- | --- | --- |
| `iid.json` | 512: depths 2/3/4 have 172/172/168 | 7919 / 7927 | Fresh diagnostic IID |
| `depth.json` | 512: 128 each at depths 5/6/7/8 | 7933 / 7937 | Exploratory depth transfer |

Use the canonical IID and OOD-depth suite recipes, including OOD-depth delay range `[16,128]`. Each depth allocation uses complete independently generated allocation quartets with 50% positives, 25% safe negatives, and 25% disconnected negatives; use `allocate_independent_variants`, not index-encoded labels. Opaque public IDs must not expose seeds or labels. Store exact recipes and episode hashes in evaluator-owned manifests; never supply them to policies.

Generate the manifests once, commit their identities before evaluating either condition, and reject overwrite or incompatible regeneration. Verify uniqueness and disjointness against pilot train/validation/debug identities, and between the two new manifests. Existing generator invariant checks apply to every episode; run the oracle over all 1,024 episodes and require 100%. This small corpus does not replace the accepted statistical leakage audit or certify a new production generator.

For wiring and timing, select a deterministic 32-episode stratified prefix from each manifest before seeing results. These rows are part of the main corpora; reuse identical verified outputs rather than count them twice.

### 2.2 Ponderer fitting and selection data

Train on the existing Phase 4 train namespace `(431,433)` and curriculum recipes. Reproduce the accepted first 12,000-update exposure schedule: one-hop updates 1–9,000, two-hop 9,001–10,000, primary 10,001–11,000, robustness 11,001–12,000. Training uses the same counter-addressed batch IDs and public examples at corresponding steps. This exposure-matched pilot schedule is fixed before baseline results; it is not an assertion that the ponderer needs the same curriculum duration.

Use the existing corrected 10,000-episode **primary validation** manifest for checkpoint selection at each 1,000-update boundary. All cap evaluations used for selection are autonomous. Rank checkpoints at cap 24 by timed success, then lower negative false actions, then earlier update; never rank by the new diagnostic IID/depth results. Keep the canonical patience rule, even if the pilot budget expires first.

The new depth corpus is never used for optimization, checkpoint/budget selection, loss/width changes, or a second attempt selected for better results. Any subsequent model revision informed by these diagnostics must explicitly retain their development exposure and use a separately approved follow-up plan.

### 2.3 Measurements and interpretation

For every condition, manifest, and ponder cap retain all episode outcomes, positive/negative denominators, error categories, selected actions/timestamps, halt/cap reason, and per-episode compute. Report timed success by variant and depth; operations, records scored, actual recurrent/jump work, parameter count, and synchronized wall time. Keep setup/report/verification time separate from inference time. Include all preactivation encoding work in end-to-end compute and show post-activation work separately.

Retain compact rows and event summaries for every episode, full traces for 32 deterministically selected successful episodes per condition/corpus plus every failure/dynamics error, and the config/environment/checkpoint identities needed for CPU replay. Stream compressed shards and hash them at completion; never build one unbounded JSON artifact. If required retention exceeds the resource cap, stop with incomplete coverage rather than dropping failure traces.

Report paired differences with 10,000 episode-bootstrap replicates, fixed analysis seed 8009, preserving the predeclared depth/variant strata. Label intervals **single-seed exploratory episode intervals**, not the five-seed hierarchical intervals or confirmatory tests in Section 10.8. No Holm-adjusted scientific-support declaration or final equivalence claim follows.

Compare intact EventFlow with each of the five ponder caps; show all points. For a nearest-compute comparison, choose from measured cap points by mean inference MAC distance on diagnostic IID, require relative distance <=10%, break ties toward smaller cap, and apply that chosen cap to depth without reselection. Explicitly state when depth compute does not overlap within 10%. Never extrapolate, interpolate an unmeasured operating point, or equate parameter matching with compute matching.

Use the canonical distinctions in interpretation:

- Compression matching intact behavior weakens a timing-specific interpretation on this diagnostic sample.
- Compression harming behavior establishes a dependence of these weights on the execution regime; it does not establish superiority over a trained ponderer.
- A competent ponderer matching/beating EventFlow is useful adverse evidence.
- A resource-limited or non-competent ponderer yields an inconclusive competitive comparison, even if its numerical score is lower.

## 3. Budgets and Automatic Decision Points

These are new pilot resource limits, not changes to the final scientific gates.

| Resource | Limit |
| --- | --- |
| Milestone A model optimization | Exactly zero steps |
| Milestone A scientific execution | 2 hours, including preparation/evaluation/reporting |
| Milestone B scientific execution | 6 hours, including competence fit, profiling, training, validation, and comparison |
| Ponderer scientific seed | 11 only; fresh independent initialization |
| Ponderer main-trajectory updates | At most 12,000, also subject to the measured budget below |
| Tiny-set competence fit | Separate debug weights; at most 1,000 updates; never promoted |
| New retained artifacts for both milestones | 10 GiB; require an additional 2 GiB working reserve before starting each unit |
| Full local quality gate | One `make verify` on final source; its engineering cost is separate from the scientific limits |

Time caps use elapsed execution time and persist across interruption/resume. Stop at a safe episode/update boundary, allowing at most one in-flight unit to finish. Enforce storage before starting a unit; keep all already written evidence. Do not delete adverse evidence or suppress required failure traces to fit. No automatic extra seeds, hyperparameter sweeps, budget increases, or source-change restarts of a learned run. Charge actual scientific work across attempts to the same ledger.

**A -> B rule:** Proceed automatically only if A completed both paired corpora, every episode has a valid result, no dynamics/integrity/offline check failed, intact IID timed success is >=90%, and intact depth timed success is >=75%. These are resource-allocation heuristics for this pilot, not scientific acceptance thresholds. Compression's score does not decide whether to train the competitor. Otherwise finish the A report, mark B deferred with the measured reason, and stop this plan's scientific execution; do not repair generalization by tuning on depth.

**B admission:** First pass the tiny-set competence tests in Task 5. Profile the first 100 main-trajectory updates without restarting/discarding them, plus the 64 fixed timing rows at cap 24. Record actual data-generation, update, evaluation, checkpoint, and artifact costs. The first 100 updates are part of the sole seed-11 trajectory.

Also measure the later curriculum shapes on disposable model/optimizer copies using three forward/backward/update trials per stage, restoring scientific model/optimizer/RNG/batch state exactly. These are counted profiling operations, not additional training examples on the main trajectory. Reserve worst-cap runtime for the final cap sweep; include generation and failure-trace storage rather than extrapolating only the cheap one-hop update.

Calculate the largest total update count `N` in `{1000,2000,...,12000}` that fits the remaining six-hour/10-GiB allowance with a 2x safety factor. Include every planned 10,000-row validation boundary, final evaluation of all five caps on all 1,024 diagnostic episodes, checkpoint writes, report generation, and Task 7's 16-episode replays for each condition/cap. Publish the calculation and `N` before advancing past update 100. If no candidate fits, publish `budget_insufficient` and the existing diagnostic weights. Never shorten a completed evaluation denominator or treat the cost estimate as a runtime guarantee.

This exposure-matched schedule may leave the budgeted ponderer in the one-hop stage. Record the last completed stage prominently; such a result cannot serve as a competent primary/OOD baseline. It is an honest cost/learning diagnostic, not a reason to change the agreed curriculum or silently grant a larger budget.

Use a final status of `completed_exploratory`, `inconclusive_budget`, `inconclusive_training`, `failed_engineering`, or `deferred_after_a`, together with separate execution completeness. A resource-limited checkpoint may be compared descriptively but is not certified a converged strong baseline.

## 4. Condition Designs

### 4.1 Intact EventFlow

Load the exact accepted portable weights in eval mode. Use the unchanged `NeuralEventFlowAgent` and `EventEngine` on new evaluator-owned episodes. A small bridge converts the resulting actions, runtime diagnostics, and compute into the comparison schema. Check that running this bridge on 32 preselected existing primary episodes reproduces their retained selected-checkpoint actions and success decisions before interpreting new data.

### 4.2 Compressed-time EventFlow

Implement a dedicated public-state-only intervention function. Its input is the same post-ACTIVATE runtime state and frozen model as intact execution; it cannot accept a bundle, intact trajectory, teacher trace, future queue, or private horizon.

1. Query current learned crossings, recomputed by the existing controller after every jump. Rank candidates by the canonical crossing offset/tie policy, disregarding only same-kind temporal refractory for compressed cognitive execution.
2. If the next candidate is RECALL/COMPOSE, execute that learned atomic jump at activation time with `dt=0`. Use the existing neural agent/jump functions and typed state. Preserve record-level consumption/refractory, memory masks, modes, provenance, and all other legality checks.
3. Accommodate the specified same-kind refractory exception through an explicitly logged temporary execution view. Record original release times and each bypass. Validate the transition against that declared view; never add an unrestricted bypass flag to normal EventFlow.
4. Do not pass zero-time microsteps through the ordinary minimum-gap normalizer: they are the explicit compressed intervention, not ordinary endogenous crossings. Log the mathematical crossing offset separately from the actual zero-time execution timestamp. Use increasing event/parent IDs and causal step order.
5. After at most 24 cognitive jumps, query once for ACT or dormancy without executing a 25th cognitive jump. If ACT is next, retain its newly predicted future timestamp; no additional recall/composition is permitted. If further cognition is needed, record `cap_reached` and no action. Distinguish this legitimate intervention cap from an illegal event/dynamics failure.
6. The private evaluator arbitrates the returned scheduled ACT against the private terminal with the existing terminal-first tie rule. If ACT wins, advance analytically to its predicted time and execute the existing learned action head/jump normally. Never clamp its prediction to the correct window. If terminal wins, the action is not emitted.

A narrow detached intervention runner can use `EventEngine.advance_to`, existing jump validation, and scheduling helpers without mutating an active session behind its history anchors. It produces its own comparison trace; it must not claim that a compressed trajectory is an ordinary Phase 4 replay artifact. Do not modify the normal engine, ordinary checkpoint format, or guard legality to make the intervention work.

### 4.3 Competitive activation ponderer

Train a separate `ActivationPonderModel` from seed 11; do not initialize it from the accepted EventFlow weights. It receives the same complete public memory and activation information, while owning its own learned latent state. Storage before ACTIVATE performs no graph traversal or recurrent pondering.

- Reuse the existing public record tensor schema and `RecordEncoder` architecture (96-dimensional embeddings). Record order is randomized without a slot-position feature; padding alone masks access. Every recurrent step may access all valid records, including previously read records, plus a learned null-read option.
- Initialize hidden state from the mean valid record embedding, shared activation-entity embedding, and the two public time features through a learned linear projection and `tanh`. Retain the public activation entity/time throughout; no hidden truth enters initialization.
- Each shared recurrent step computes a width-256 SiLU retrieval query from hidden state plus activation/time features, scores linearly projected record keys by a scaled dot product, and selects one record or null. Deterministic ties use stable public record ID, not slot index. A single `GRUCell` updates hidden state from the selected 96-dimensional embedding, activation embedding, and time features.
- Readouts use a shared width-256 SiLU trunk: record role (5), next focus (64), hazard class (4), log-delay, normalized deadline, safe/null status (3), confidence, support-append and continue-search logits, action class (four shields plus abstain), nonnegative action offset, and halt logit. These are learned predictions supervised by the same usable oracle content labels. Do not impose EventFlow's RECALL/COMPOSE alternation, mode gate, guard mechanism, or consumed-record restriction on this competitor.
- Choose recurrent width once from `{640,672,704,736,768,800}` by closest total parameter count to authenticated EventFlow, subject to ±10% and the 5M ceiling; break ties toward smaller width. This rule is fixed before A results and uses parameter arithmetic only; derive and record the exact width before the first ponderer fit. All counted parameters must participate in real forward paths; no padding/dummy parameters. Record the exact count and entity-table subtotal. If none qualifies, treat the proposed architecture as unimplementable within this plan and report it before ponderer scientific execution.
- One GRU update is one recurrent transition; count separate scorer/readout/halt computations and MACs as well. At inference evaluate all caps `{4,8,12,16,24}` using the same selected checkpoint. Halt at sigmoid >=0.5 or the cap. From the final hidden state choose the five-way action; abstain emits nothing, otherwise schedule `t0 + softplus(offset)` with no access to the terminal/window.
- During teacher-forced training align one transition to one oracle record interpretation; disconnected path termination supplies the null-read target. Score retrieval before revealing the teacher-selected record for the update. At inference use only predicted reads. Teacher choices never enter runtime.
- Use masked CE for retrieval/role/focus/hazard/status/action, BCE for confidence/support/continue/halt, and Smooth L1 for log-delay, normalized deadline, and normalized action offset. Reuse existing content-label validity and independently masked reductions; an absent target contributes no fabricated example. At activation the normalized deadline label is relative delay; normalize the positive action-offset loss by the teacher delay, with target `0.825`. This teacher value is a loss target only. Retain negative abstention/false-action supervision. Use canonical weights: retrieval/compose-type/hazard/action 1.0, focus 0.5, deadline/action-time 0.25, state-bound 0.05; halt BCE 1.0; event cost 0.001. As in the current content objective, compose-type averages its role/status/confidence/support/continue subterms. The deadline group averages the two applicable delay representations. The event-cost term is the mean sum of survival probabilities `prod(1 - sigmoid(halt_before_step))` through the sampled cap. Halt targets become true at the oracle terminal/no-path point. Sample the cap uniformly each training example, mask all later steps, and do not label cap exhaustion as a successful learned halt. Final labels may supervise the cap-ending readout; unused oracle future states are never injected.
- No autonomous-rollout gradient, RL, architecture/loss sweep, or extra model seed. A training failure remains a recorded result.

The width-selection rule and architecture/loss description above are committed before A results. Implementation corrections must fix demonstrable engineering defects and retain their history; changing the scientific mechanism after seeing results requires a follow-up plan.

## 5. File Responsibilities and Interfaces

New files are phase-scoped and share the existing low-level primitives. No plugin system or broad evaluation-framework rewrite is needed.

| File | Responsibility |
| --- | --- |
| `configs/eval/phase5a.yaml` | Exact diagnostic protocol, budgets, identities, and condition settings |
| `src/silent_cascade/eval/comparison_types.py` | `ComparisonConfig`, `ComparisonManifest`, `ComparisonIdentity`, `ComparisonRow`, `ConditionResult`, `BudgetLedger`, status enums |
| `src/silent_cascade/eval/comparison_data.py` | Fixed manifest generation, reconstruction, invariants, public projections |
| `src/silent_cascade/eval/comparison_runner.py` | Private terminal/scoring ownership, paired condition execution, measured accounting, bounded artifact writes |
| `src/silent_cascade/eventflow/compressed.py` | Public-only compressed function and intervention-step records |
| `src/silent_cascade/eval/comparison_protocol.py` | Admission decisions, resource accounting, nearest-compute selection, paired analysis |
| `src/silent_cascade/models/activation_ponder.py` | Separate model, strict model config, predictions/state, width selection |
| `src/silent_cascade/train/activation_ponder.py` | Teacher-only batch projection, losses, bounded training, selected/latest checkpoint handling |
| `src/silent_cascade/report/comparison.py` | Hash verification and report generation from stored rows, without model execution |
| `scripts/run_phase5a.py` | Explicit `prepare`, `checkpoint-check`, `ponder`, and `report` entrypoints |
| `docs/phase5a-comparative-pilot.md` | Executed results, budgets, limits, and reproduction commands |
| `docs/PLAN.md` | Phase 5A draft/execution/result navigation; full Phase 5 stays pending |

Modify `eval/compute.py` only as needed to count actual new functional operations/GRU work while preserving Phase 4 counters and schema. Test files live under `tests/comparison/`, with any shared-counter regression in `tests/neural/test_neural_compute.py`. Name each new schema `phase5a-...-v1`; legacy validators remain closed.

Common types used by tasks:

- `ComparisonIdentity`: experiment/version, condition/intervention/cap, producer and executor revisions, dirty status, config/generator/manifest hashes, checkpoint/logical-state hashes, seed/device, protocol hash, and exploratory purpose.
- `ConditionResult`: emitted actions, stop reason, ordered public execution steps, compute, and an optional typed error; no private truth field.
- `ComparisonRow`: immutable episode ID/hash, identity hash, `ConditionResult`, scorer-owned outcome/variant/depth, and artifact references. Recompute scoring on load.
- `BudgetLedger`: cumulative elapsed scientific time, update counts, retained bytes, attempts, committed boundaries, and finish reason. Resume cannot reset it.
- `ComparisonManifest`: ordered evaluator-owned recipes and hashes with fixed allocation; `iter_bundles(manifest, config)` reconstructs them locally.

## 6. Implementation Tasks

### Task 1: Pin the diagnostic protocol and data

**Files:** Create config, `comparison_types.py`, `comparison_data.py`, `tests/comparison/__init__.py`, `tests/comparison/test_data.py`, `tests/comparison/test_protocol.py`.

**Interfaces:** `prepare_manifests(config: ComparisonConfig, *, output_dir: Path) -> dict[str, ComparisonManifest]`; `iter_bundles(manifest: ComparisonManifest, config: ComparisonConfig) -> Iterator[EpisodeBundle]`. Consume the existing generator and scorer/oracle APIs; expose only `bundle.public` to later policy adapters.

- [ ] Write `test_diagnostic_allocations_and_identity`: assert IID depth counts `{2:172,3:172,4:168}`, depth counts `{5:128,6:128,7:128,8:128}`, each corpus variants `256/128/128`, DEBUG split, 1,024 distinct opaque IDs, deterministic hashes, and `gate_eligible is False`.
- [ ] Add `test_reject_frozen_overwrite_or_protocol_mutation` and `test_every_diagnostic_episode_satisfies_invariants_and_oracle`; assert frozen destinations/changed protocol fail and every generated score is successful. Use small fixtures for ordinary tests; the real 1,024-row audit belongs to Task 4.
- [ ] Run `uv run pytest -q tests/comparison/test_data.py tests/comparison/test_protocol.py`; confirm failure because the new contract does not exist.
- [ ] Implement the strict types, exact allocation, reconstruction, atomic publication, and protocol hashing; include legacy evidence identities and budget constants.
- [ ] Rerun the tests, inspect zero failures, and commit `feat: add bounded comparison protocol and manifests`.

### Task 2: Implement paired evaluation and complete compute accounting

**Files:** Create `comparison_runner.py`, `tests/comparison/test_runner.py`, `tests/comparison/test_compute.py`; modify `eval/compute.py` only for shared operation coverage.

**Interfaces:** `run_comparison(*, config: ComparisonConfig, manifest: ComparisonManifest, identity: ComparisonIdentity, model: torch.nn.Module, output_dir: Path, budget: BudgetLedger) -> Path`; private evaluator helper `run_intact_episode(bundle: EpisodeBundle, *, model: EventFlowModel, config: ComparisonConfig) -> ConditionResult`. Dispatch only the three closed condition identities in this plan and validate the model type/cap before execution. The evaluator owns the bundle and passes only initialization/public callbacks to the intact agent, public post-activation state to compression, or `PublicEpisode` to the ponderer. Public policies return proposals/steps; the private evaluator arbitrates actual emission before constructing `ConditionResult`. No private terminal is captured by a policy closure.

- [ ] Write `test_policy_receives_only_public_inputs`: a spy rejects bundles/keys/targets; changing only the private terminal leaves preterminal policy choices unchanged. Add tests for terminal-first ties, late scheduled actions, duplicate actions, and negative abstention using the existing scorer.
- [ ] Add `test_comparison_bridge_matches_intact_engine` and `test_errors_keep_episode_denominators`; compare actions/decisions on fixed fixtures and retain one row for every failed episode.
- [ ] Add `test_compute_includes_prefix_and_all_executed_calls`: verify small known linear/retrieval operations, no free initialization, and zero foundation-model calls. Assert legacy `RuntimeCompute(opportunities=1)` remains invalid.
- [ ] Run `uv run pytest -q tests/comparison/test_runner.py tests/comparison/test_compute.py`; confirm the expected initial failures.
- [ ] Implement the evaluator-owned terminal adapter, intact bridge, new rows, compact streaming artifacts, and operation accounting. Separate full end-to-end and post-activation counters; report both.
- [ ] Rerun these tests plus `tests/pilot/test_timed_metrics.py` and `tests/pilot/test_neural_accounting.py`; commit `feat: evaluate paired exploratory conditions with measured compute`.

### Task 3: Implement the exact compressed-time intervention

**Files:** Create `eventflow/compressed.py`, `tests/comparison/test_compressed.py`; extend the new comparison runner only.

**Interfaces:** `run_compressed_from_activation(agent: NeuralEventFlowAgent, state: RuntimeState, *, transition_cap: int = 24) -> CompressedDecision`. Define `CompressedDecision` in this module: final public runtime state, optional predicted ACT, ordered steps/bypasses, compute, and stop reason. The input state is detached from any live engine session.

- [ ] Write `test_compressed_recomputes_choices_at_activation`: a controlled learned-agent fixture changes its preferred record after a jump; assert choices follow recomputed predictions, all cognitive timestamps equal activation, and no intact trace can be supplied.
- [ ] Add `test_only_temporal_refractory_is_bypassed`: alternating jumps execute at `dt=0`, every bypass is logged, consumed/refractory records remain ineligible, invalid modes/provenance still fail, and ordinary EventFlow behavior is unchanged.
- [ ] Add `test_compressed_cap_dormancy_and_future_act`: no 25th cognitive jump; ACT discovered after the 24th jump is allowed, its future timestamp is preserved, terminal wins ties, and absent ACT remains an ordinary unsuccessful/abstaining outcome as scored. Include early ACT, no candidate, NaN, and predicted-past-time cases.
- [ ] Add `test_compressed_keeps_weights_and_post_activation_input_immutable`; compare exact weight/input hashes and ensure the future ACT uses the learned head after the proper analytic advance.
- [ ] Run `uv run pytest -q tests/comparison/test_compressed.py`; confirm failures, implement Section 4.2, and rerun with `tests/pilot/test_neural_agent.py`, `tests/pilot/test_neural_scheduler.py`, and `tests/pilot/test_transition_equivalence.py`.
- [ ] Commit `feat: add autonomous compressed-time EventFlow diagnostic`.

### Task 4: Deliver and execute Milestone A

**Files:** Create `comparison_protocol.py`, `report/comparison.py`, `scripts/run_phase5a.py`, `tests/comparison/test_report.py`, `tests/comparison/test_workflow.py`, and the results guide. Update `docs/PLAN.md` with execution status.

**Interfaces:** `decide_after_a(rows: Sequence[ComparisonRow], *, protocol_sha256: str) -> MilestoneDecision`; `build_comparison_report(run_dir: Path, output_dir: Path) -> Path`. `MilestoneDecision` stores measured denominators, engineering/resource status, and the exact Section 3 decision.

- [ ] Write `test_a_admission_is_predeclared`: complete valid fixtures at IID 90%/depth 75% admit B; lower scores, missing rows, dynamics failure, protocol mismatch, or offline failure defer it. Compression score alone never affects admission.
- [ ] Write `test_report_is_artifact_only_and_never_certifies_phase5`: reject corruption/missing pairs, reconstruct integer numerators, preserve failed rows, and never emit a final scientific-support claim. Add budget/resume tests proving caps persist and partial corpora are not reported as full results.
- [ ] Run the new report/workflow tests, confirm failures, implement admission/report/CLI, and make them pass. Commit the executable source and protocol before data generation.
- [ ] Execute `prepare`, authenticate legacy weights, audit all 1,024 new episodes, publish the manifests, and commit their hashes before the comparison. Record generator time; stop if A's resource limit is reached.
- [ ] Execute the retained 32-row intact compatibility check and fixed timing subsets. Then run intact/compressed on both complete corpora within the same A budget, retaining sample and failure traces.
- [ ] Build the A report and immutable decision from artifacts; commit compact results and documentation. If B is deferred, proceed directly to Task 7 with that outcome. Do not implement Tasks 5–6 speculatively before this checkpoint.

### Task 5: Implement and establish ponderer competence

**Files:** Create `models/activation_ponder.py`, `train/activation_ponder.py`, `tests/comparison/test_ponder_model.py`, `tests/comparison/test_ponder_training.py`; extend `eval/compute.py` and its existing neural compute tests for GRU accounting.

**Interfaces:** `ActivationPonderModel(config: PonderModelConfig)`; `ponder_public(model: ActivationPonderModel, public: PublicEpisode, *, cap: int) -> PonderDecision`; `project_ponder_batch(batch: TrainingBatch, *, caps: torch.Tensor) -> PonderBatch`; `ponder_loss(model: ActivationPonderModel, batch: PonderBatch) -> PonderLoss`. Define `PonderModelConfig` and `PonderDecision` in the model file, `PonderBatch` and `PonderLoss` in the training file. `PonderDecision` contains an optional proposed future action, public recurrent steps, stop reason, and compute; actual emission and scoring belong to the private evaluator.

- [ ] Write `test_ponder_width_is_parameter_matched_without_dummy_weights`, `test_ponder_reads_complete_memory_and_is_slot_permutation_invariant`, and `test_ponder_halts_or_caps_without_advancing_world_time`. Assert all five caps, `>=0.5` threshold, full-memory access, one GRU update per transition, and action offset independent of private truth.
- [ ] Write `test_teacher_projection_has_no_target_in_prediction_inputs`, `test_budget_masks_future_teacher_steps`, and `test_all_loss_terms_and_shared_steps_receive_finite_gradients`; cover positives, safe negatives, disconnected/null reads, and insufficient caps.
- [ ] Add `test_gru_macs_match_hand_calculated_toy_cell` and branch-count checks for retrieval/null/early-halt/cap paths. Do not count one composite step as zero or omit repeated shared-module calls.
- [ ] Run these tests, confirm the expected initial failures, implement the fixed architecture/training projection/loss, and rerun them together with `tests/neural/test_neural_compute.py` and import-boundary regressions.
- [ ] Execute a separate fixed 64-episode DEBUG competence fit with roots `(7963,7993)`, seed 11, at most 1,000 updates, and no promotion. Use 16 one-hop, 16 two-hop, and 32 primary examples, balanced by complete quartets. Require 64/64 autonomous timed successes at cap 24, no errors, no world-time advance during cognition, and exact CPU checkpoint continuation on this fixed set. A controlled long-chain fixture must also prove the runtime can execute nine learned-style transitions before scheduling; this structural fixture is not evidence of learned OOD success.
- [ ] If competence fails within the budget, investigate engineering failures systematically, retain the result, and finish with `inconclusive_training` if learning remains insufficient. No loss/architecture sweep or new seed. On success commit `feat: add parameter-matched adaptive activation ponderer`.

### Task 6: Run the bounded single-seed competitive pilot

**Files:** Extend `train/activation_ponder.py`, `comparison_protocol.py`, CLI and `tests/comparison/test_ponder_training.py`, `tests/comparison/test_workflow.py`.

**Interfaces:** `fit_ponder(*, config: ComparisonConfig, run_dir: Path, budget: BudgetLedger, resume: Path | None = None) -> PonderTrainingResult`; `admit_ponder_budget(profile: CostProfile, budget: BudgetLedger) -> TrainingAdmission`. Define result/profile/admission records in `comparison_types.py`; bind selected/latest descriptors, counters, selection evidence, projected cost, and stop reason.

- [ ] Write `test_budget_admission_counts_full_validation_and_final_caps`: the 2x estimate includes every 10,000-row validation and all 5,120 final episode executions; no fitting N yields `budget_insufficient`. Add `test_resume_preserves_next_batch_optimizer_and_consumed_budget`.
- [ ] Add `test_selection_uses_only_primary_validation_and_fixed_cap24`; a superior depth/IID diagnostic result cannot change checkpoint rank, update budget, or training configuration. Add corrupt/incompatible checkpoint and dirty-source tests.
- [ ] Run these tests, confirm failures, implement deterministic checkpoints, latest/selection handling, resource stops, and the Phase 4 exposure schedule, then rerun all comparison tests and commit the training source.
- [ ] Start the one fresh seed-11 trajectory; profile its first 100 counted updates and the fixed cap-24 timing rows. Publish the admission calculation, architecture parameter count, and selected `N`. Do not reset initialization/optimizer after profiling.
- [ ] Continue only through the admitted boundaries, running complete primary validations at 1,000-step intervals and preserving failure evidence. On interruption resume the exact durable CPU checkpoint and remaining budget. On resource exhaustion publish the appropriate inconclusive status.
- [ ] Select by the fixed primary-validation rule. Evaluate the selected checkpoint at all five caps on both diagnostic corpora only after selection. If the resource budget cannot complete them, retain partial evidence without silently narrowing the planned comparison.
- [ ] Build the paired report, including training exposure/budget asymmetry, competence result, every cap, missing compute overlap, and all adverse outcomes. Commit compact result artifacts and the guide; do not start another seed or baseline family.

### Task 7: Verify and deliver the completed scope

**Files:** Results guide, compact manifests/reports, `docs/PLAN.md`; any targeted fixes require TDD and a documented root cause.

- [ ] Run `uv run pytest -q tests/comparison` and relevant existing regression tests for every touched shared module; use the full local gate below as the final software check.
- [ ] Request one independent whole-change code review using `superpowers:requesting-code-review`; focus on the five Review Focus items, precise compressed semantics, fairness, and resource bounds. Fix substantiated findings and record their validation.
- [ ] Run **`make verify` locally on the final source**. Retain real output, exit status, and revision. This gate may take roughly the historical 97 minutes or longer; it is not hidden inside the scientific budget. A failed required check blocks a software-complete claim.
- [ ] Independently reload the small comparison artifact inventory, verify hashes/identities and recompute scores, paired denominators, counters, selected checkpoint rank, admission decision, and report values. Reexecute a deterministic 16-episode sample per executed condition/cap on CPU, with identical decisions and semantic traces; charge this scientific replay to the relevant milestone's budget and reserve it in profiling.
- [ ] Confirm the accepted Phase 4 artifacts remain untouched, no final-test manifests exist from this work, and zero foundation-model calls are recorded throughout. Do not rerun the full historical Phase 4 verifier merely because new phase-scoped source exists.
- [ ] Commit the final outcome and reproduction commands. Mark Phase 5A complete only for its actually executed scope; if B was deferred/inconclusive say so explicitly. Keep full Phase 5 and Phase 6 pending. Summarize measured cost, limitations, and the evidence-based next research decision.

## 7. Commands to Deliver During Implementation

These commands are proposed interfaces, not currently implemented commands. All execution requires approval of this plan first.

```sh
UV_OFFLINE=1 UV_FROZEN=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  uv run python scripts/run_phase5a.py prepare \
  --config configs/eval/phase5a.yaml --run-dir "$PHASE5A_RUN_DIR"

UV_OFFLINE=1 UV_FROZEN=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  uv run python scripts/run_phase5a.py checkpoint-check \
  --config configs/eval/phase5a.yaml --run-dir "$PHASE5A_RUN_DIR" \
  --phase4-run-dir "$PHASE4_RETAINED_RUN"

UV_OFFLINE=1 UV_FROZEN=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  uv run python scripts/run_phase5a.py ponder \
  --config configs/eval/phase5a.yaml --run-dir "$PHASE5A_RUN_DIR"

UV_OFFLINE=1 UV_FROZEN=1 uv run python scripts/run_phase5a.py report \
  --run-dir "$PHASE5A_RUN_DIR" --output reports/phase5a-v1
```

`ponder` verifies the committed A decision and refuses execution if B was deferred. Repeated commands authenticate/reuse completed compatible units or resume their exact checkpoint; they do not overwrite results or grant fresh budgets. `report` loads artifacts only. CLI errors distinguish scientific resource/learning outcomes from integrity/implementation failures.

## 8. Approval and Execution Handoff

Only this document and the plan index are being changed during plan preparation. No model, configuration, source, test, build file, or scientific artifact is implemented/generated at this stage.

Recommend native execution with `superpowers:executing-plans`, sequential task commits, and one independent final review: the tasks share tight runtime/data interfaces, and the first milestone can avoid unnecessary baseline work. An alternative execution method may be specified with approval.

**Wait for the user's explicit approval of this written plan before Task 1.** Approval covers both milestones subject to the fixed admission and resource rules; a deferral or inconclusive result is a valid stopping point, not permission to expand the experiment.
