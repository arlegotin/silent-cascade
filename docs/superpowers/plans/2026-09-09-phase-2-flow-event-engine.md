# Silent Cascade Phase 2 Flow and Event Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the complete non-neural hybrid runtime for Silent
Cascade: bounded persistent state, stable closed-form flow, exact endogenous
guard crossings, typed jumps, a private causal event queue, fail-loud dynamics
invariants, arbitrary-time snapshots, safe checkpoints, deterministic replay,
and a public-facts-only scripted agent that solves the fixed 10,000-episode
validation manifest without importing or receiving oracle truth.

**Architecture:** Represent each inter-event interval as an immutable analytic
segment anchored at its last causal event. The agent-visible `RuntimeState`
contains only public-derived memory, bounded continuous tensors, modes,
hypotheses, actions, and the current segment; the engine-private
`RuntimeSession` separately owns the terminal queue, scorer truth, trace
recorder, and cached next crossing. Agents emit post-jump segment parameters
and predict internal events from those parameters without seeing an external
horizon. The engine privately races one cached endogenous prediction against
the external heap, applies the declared tie policy, and invalidates the
prediction after every causal event. A deterministic scripted condition drives
the same guard and jump machinery from public facts solely as an engineering
oracle for the runtime—not as a learned baseline or scientific result.

**Tech Stack:** Python 3.12, PyTorch float32 on CPU and capability-gated Apple
MPS, host Python float64 timestamps, NumPy, strict Pydantic v2, frozen
dataclasses, `safetensors`, Typer, SHA-256, pytest, Hypothesis, Ruff, and `uv`.

**Spec:** `docs/superpowers/specs/2026-08-30-silent-cascade-design.md`

**Plan status:** **User-approved; execution in progress.**

## Execution Preconditions and Superpowers Workflow

Before changing source, tests, configuration, manifests, documentation other
than this proposed plan/index link, or build files:

1. Confirm explicit approval of this exact Phase 2 plan in the task thread.
2. Confirm Phase 1 remains complete in `docs/PLAN.md`, all five Phase 1
   artifacts verify, this plan is committed, and `git status --short` is empty.
   Record `git rev-parse HEAD` as `phase2_plan_base_revision`.
3. Work on the current branch and make the task-level commits named below.
4. Invoke `superpowers:subagent-driven-development`. Give each implementation
   task to a fresh worker, then run specification-compliance and code-quality
   reviews before accepting that task.
5. Invoke `superpowers:test-driven-development` for Tasks 1–13. Preserve the
   stated RED/GREEN order; do not write implementation before observing the
   named test fail for the intended reason.
6. On any failure or unexpected behavior, invoke
   `superpowers:systematic-debugging` before changing code.
7. Invoke `superpowers:requesting-code-review` at every task boundary and
   resolve findings before the task commit.
8. Task 14 is evidence collection only. Freeze the reviewed source commit
   before running it; any later source correction invalidates and regenerates
   the complete Phase 2 evidence artifact.
9. Invoke `superpowers:verification-before-completion` before any completion
   claim and `superpowers:finishing-a-development-branch` after the committed
   Phase 2 gate passes.

Repository standing approval covers routine safe, reversible implementation
rulings. Do not ask permission questions when the specification, tests, or
repository evidence determines the best answer.

## Global Constraints

- Distribution: `silent-cascade`; package: `silent_cascade`; executable:
  `silent-cascade`; Python: `>=3.12,<3.13`.
- Work only on the current branch. Make one meaningful commit per accepted
  task; do not batch unrelated tasks into one commit.
- All verification is local. Do not create `.github/workflows`, hosted checks,
  deployment/release automation, badges for hosted checks, or CI/CD language.
  `make verify` remains the complete routine local quality gate.
- Primary and Phase 2 gate execution is offline and records exactly zero
  foundation-model calls. Core imports may not load MLX, `mlx-vlm`, or Qwen.
- Phase 1 evidence is immutable historical evidence. Do not regenerate, edit,
  bless, or relabel any Phase 1 artifact while implementing Phase 2.
- Do not create anything beneath `manifests/frozen/`; final test freezing is
  Phase 6-only.
- The agent receives only `AgentInit`, chronological `FACT` events, and one
  `ACTIVATE`. It never receives `EpisodeBundle`, `EpisodeTruth`, the private
  terminal, next external timestamp, action window, variant, path, generator
  seed, manifest coordinate, accepted attempt, or oracle choice.
- `OUTCOME` and `END` remain engine-private. The engine scores at the private
  terminal and then sets `TERMINAL`; neither terminal kind is delivered to
  `AgentCondition.on_external`.
- EventFlow endogenous kinds are exactly `RECALL`, `COMPOSE`, and `ACT`.
  `NOOP` is rejected for `Condition.EVENT_FLOW`; no `THINK`, polling fallback,
  timer prompt, or fixed cognitive grid enters Phase 2.
- State tensors are float32 on CPU/MPS. Absolute timestamps, event comparisons,
  serialized times, and autonomous crossing calculations are host float64.
  Never allocate a float64 MPS tensor.
- Exact limits: state bounds `[-1, 1]`, latent flow rates `[1e-5, 20]`, guard
  rates `[1e-5, 500]`, guard threshold `1`, minimum internal gap `1e-4`,
  same-kind refractory `1e-3`, near-tie tolerance `1e-9`, primary memory
  capacity `64`, and at most `64` executed EventFlow internal events.
- A fifth consecutive selected sub-minimum prediction raises `DynamicsError`.
  Exactly 64 internal events are permitted; an attempted 65th event raises
  `DynamicsError`. An earlier external event breaks a clamp streak and
  invalidates the cached internal prediction.
- A dormant guard (`A <= 1`) never evaluates the crossing logarithm. When all
  guards are dormant, the engine advances directly to the next external event.
- Every causal jump advances continuous state exactly to its timestamp, applies
  exactly one legal transition, resets all three accumulators to zero, installs
  one new segment, and records the event and prediction snapshot. The private
  terminal is the only event that ends without installing a next segment.
- Do not add a neural encoder/controller, learned retrieval, tensor memory
  embeddings, deterministic eviction, training loop/checkpoint, baseline,
  causal intervention, report generator, or optional language shell. Those
  belong to later phase plans.
- The Phase 2 scripted condition is a transparent public-facts rule system. Its
  10,000/10,000 result validates engine semantics only and must always be
  labelled “non-neural runtime engineering evidence,” never benchmark or model
  evidence.

## Frozen Phase 2 Design Rulings

### Segment-anchored analytic state

Every segment retains its post-jump origin time, origin tensors, bounded
targets/rates, three guard targets/rates, parent event ID, and prediction hash.
`advance_to` evaluates from that immutable origin rather than repeatedly
integrating from the last observation. This has three consequences:

1. ordinary flow and `state_at` use the exact closed form;
2. a mid-flow checkpoint may materialize a current snapshot without changing
   the causal segment or rescheduling the next event; and
3. resumed CPU execution reaches the same later causal boundary from the same
   anchor, avoiding split-step rounding changes in event decisions and hashes.

A pause cursor is not an event, does not reset guards, does not invoke the
controller, does not enter the causal trace, and does not consume a new event
ID. Checkpoint-only materialization is counted separately from causal compute.

### Public runtime versus private session

Use two sibling containers:

```text
RuntimeState (agent-visible)       RuntimeSession (engine-private)
----------------------------       -------------------------------
continuous state + segment         EpisodeTruth/private terminal
mode + focus node                  external heap
typed bounded memory               cached internal prediction/dormancy
active record + supports           private trace recorder
hypothesis + action history        score/result and crash context
refractory + safety counters       checkpoint cursor
```

`RuntimeState` contains no external queue, terminal horizon, private label, or
manifest coordinate. The engine passes only this state to
`next_internal_event`; the latest v1.0.2 protocol intentionally has no
`external_horizon` argument.

### Exact event IDs, parentage, and ties

Keep generator-owned external IDs unchanged. Allocate executed endogenous IDs
from a disjoint constant namespace, `INTERNAL_EVENT_ID_BASE = 1 << 62`, plus
the zero-based executed-internal ordinal. A preempted prediction does not
consume an ordinal. Its parent is the causal event after which its segment was
installed.

The engine compares one external heap head with one cached internal prediction.
Times within `1e-9` are treated as a tie and resolved in this order:

```text
private terminal -> ordinary external -> ACT -> COMPOSE -> RECALL -> NOOP
```

The internal prediction is never inserted into the external heap. Therefore a
higher-priority later-within-tolerance external event can preempt it without
leaving an older queued internal event behind. The discarded prediction is
recomputed from the post-external segment. Every tie records both original
timestamps, the tolerance, candidates, and winner.

### Minimal Phase 2 memory

Implement the typed metadata portion of `memory/store.py` now because `FACT`,
`RECALL`, and `COMPOSE` require it. It stores at most 64 immutable
`MemoryRecord` values plus validity, consumed, and refractory metadata. Primary
overflow fails loudly. Phase 3 adds the `[64, 96]` tensor embeddings, learned
scorer, and deterministic memory-pressure eviction; Phase 2 must not pre-empt
that work or silently evict primary facts.

### Scripted condition boundary

`ScriptedEventFlowAgent` may inspect only its `RuntimeState`. In `SEARCHING`,
it selects a legal, non-consumed record whose public subject equals the current
focus, with deterministic `(confidence desc, record_id asc)` tie-breaking. It
uses public record fields to construct a correct typed composition and uses the
standing objective’s public action-target fraction `0.825` to time `ACT` after
a reachable hazard is composed. It does not import any `silent_cascade.env`
module, solve from `EpisodeTruth`, consume oracle traces, or receive a terminal
horizon.

RECALL and COMPOSE gaps are irregular deterministic functions of public record
ID, kind, focus, and event ordinal in `[0.05, 0.068]`. For each desired gap the
scripted controller emits `A=1.5` and
`lambda=log(3)/gap`, so the common analytic guard—not a queued timer—causes the
event. All other guards remain dormant. The action guard targets the midpoint
`t0 + 0.825 * delay`; safe and disconnected states are fully dormant.

### Runtime checkpoints versus training checkpoints

Phase 2 checkpoints are safe immutable runtime snapshots. Store metadata and
CPU-copied tensors in one `safetensors` file with canonical JSON metadata,
per-tensor hashes, source/config/agent identifiers, queue, cached prediction,
trace prefix, and Python/NumPy/Torch RNG states. Do not use pickle or
`torch.save`. Phase 3 training checkpoints extend this container with model and
optimizer tensors; Phase 2 does not implement optimizer semantics.

### Trace and replay privacy

The live causal recorder is engine-private and gives the agent no access to its
contents. A sanitized online event summary represents the private terminal as
`terminal` with no payload or label. An explicitly marked post-run validation
replay artifact may contain `EpisodeArtifact`/scorer truth so the run can be
regenerated and scored. `silent-cascade replay` consumes that private artifact,
runs on CPU, and compares kinds, parents, timestamps, selected records,
hypotheses, supports, action, metric, state hashes, and final trace hash.

## Phase 2 File Map

| Path | Responsibility |
|---|---|
| `configs/model/event_flow.yaml` | Exact state dimensions, flow/guard bounds, event limits, tie policy, and scripted validation settings. |
| `src/silent_cascade/eventflow/__init__.py` | Side-effect-free runtime package boundary. |
| `src/silent_cascade/eventflow/config.py` | Strict Phase 2 config and cross-checks against Phase 0/1 ceilings. |
| `src/silent_cascade/eventflow/protocols.py` | `AgentCondition` and `ClosedFormFlow` protocols with no external-horizon leak. |
| `src/silent_cascade/eventflow/state.py` | Continuous channels, anchored segments, runtime core/state, safety metadata, counters, and public-derived state construction. |
| `src/silent_cascade/eventflow/flow.py` | Stable float32 closed-form flow, segment advance, and non-mutating state snapshots. |
| `src/silent_cascade/eventflow/guards.py` | Stable host-float64 crossings, differentiable float32 crossings, dormancy, mode masks, refractory release, and prediction hashes. |
| `src/silent_cascade/memory/__init__.py` | Side-effect-free memory package boundary. |
| `src/silent_cascade/memory/store.py` | Bounded immutable record metadata, legality masks, consumed state, and record refractory. |
| `src/silent_cascade/memory/provenance.py` | Runtime provenance immutability and support-ledger validation. |
| `src/silent_cascade/eventflow/jumps.py` | Typed FACT/ACTIVATE/RECALL/COMPOSE/ACT jumps and explicit composition decisions. |
| `src/silent_cascade/eventflow/scheduling.py` | Private external heap, tie ranking, cached prediction, endogenous ID allocation, and preemption. |
| `src/silent_cascade/eventflow/invariants.py` | Post-jump/state/queue/action/parentage/safety checks. |
| `src/silent_cascade/eventflow/engine.py` | Private runtime session, finite causal loop, callback seam, terminal dispatch, scoring, pause/resume cursor, and result. |
| `src/silent_cascade/eventflow/scripted.py` | Public-facts-only guard-driven scripted engineering condition. |
| `src/silent_cascade/eventflow/checkpoint.py` | Safe single-file runtime checkpoint codec and RNG/state restoration. |
| `src/silent_cascade/eventflow/replay.py` | Strict replay artifact, deterministic CPU rerun, and mismatch diagnostics. |
| `src/silent_cascade/logging/trace.py` | Causal event/segment summaries, bounded crash projection, trace hash, and post-run trajectory query. |
| `src/silent_cascade/doctor.py` | Doctor smoke routed through production Phase 2 flow/guard functions. |
| `src/silent_cascade/cli.py` | Thin `silent-cascade replay` adapter; existing Phase 1 commands remain unchanged. |
| `scripts/check_phase2_engine.py` | Explicit 10,000-episode CPU gate collector; not part of routine `make verify`. |
| `scripts/verify_phase2_gate_artifact.py` | Bounded independent arithmetic/hash/history verifier for the committed gate artifact. |
| `tests/fixtures/phase2/` | Hand-authored flow, guard, event, replay, and checkpoint regressions. |
| `tests/unit/test_phase2_config.py` | Exact config values/types/cross-limit checks. |
| `tests/unit/test_eventflow_state.py` | Tensor shapes/dtypes/devices, clone/alias safety, and anchored segment construction. |
| `tests/unit/test_flow.py` | Identity, bounds, semigroup, huge/tiny time, float32 device, and reference accuracy. |
| `tests/unit/test_guards.py` | Dormant/active crossings, threshold edges, rates, masks, refractory, and gradients. |
| `tests/unit/test_runtime_memory.py` | Capacity, masks, consumption, refractory, supports, and provenance. |
| `tests/unit/test_jumps.py` | Legal/illegal modes, bounded injections, hypotheses, support ledgers, and actions. |
| `tests/unit/test_scheduling.py` | Queue privacy, event IDs, tie order, preemption, caching, gap clamping, and caps. |
| `tests/unit/test_runtime_invariants.py` | Every post-jump fail-loud invariant and sanitized crash context. |
| `tests/unit/test_trace.py` | Deterministic summaries/hashes, bounded projections, and arbitrary-time snapshots. |
| `tests/unit/test_checkpoint.py` | Safe tensor/metadata/RNG codec, corruption/mismatch refusal, and no-clobber writes. |
| `tests/integration/test_event_engine.py` | Full callback seam, private terminal dispatch, positive/negative scoring, and dormancy. |
| `tests/integration/test_scripted_agent.py` | Public-only scripted positive/safe/disconnected cascades and delay swaps. |
| `tests/integration/test_dynamics_failures.py` | Zeno-like clamps, event cap, time reversal, crash bundles, and external streak reset. |
| `tests/integration/test_checkpoint_resume.py` | Mid-flow and per-kind paused-world continuation identity. |
| `tests/integration/test_replay.py` | CPU trace replay, tamper detection, and repeatable trace hash. |
| `tests/integration/test_cli_replay.py` | Installed/root CLI replay success and typed failure output. |
| `tests/integration/test_phase2_gate.py` | Small collector/verifier profile, artifact tampering, and two-state delivery index. |
| `tests/property/test_eventflow_properties.py` | Random flow/guard/time/event/memory properties. |
| `tests/regression/test_import_boundaries.py` | Agent/public/private/oracle/Qwen import and callback boundaries. |
| `tests/regression/test_phase2_fixtures.py` | Exact hand-authored trace/checkpoint/replay regressions. |
| `manifests/validation/v1/phase2-engine-gate.json` | Final source-bound non-neural Phase 2 engineering evidence. |
| `docs/PLAN.md` | Proposed plan link, then exact source/hash-bound completion state. |
| `README.md` | Truthful Phase 2 runtime/replay status with no learned claim. |

## Dependency Direction

```text
config + errors + hashing + io + rng + schemas
                         │
                         ▼
               eventflow.config/state
                  │       │       │
                  ▼       ▼       ▼
                flow    guards  memory.store/provenance
                  │       │       │
                  └───┬───┴───┬───┘
                      ▼       ▼
                    jumps  scheduling
                      │       │
                      └───┬───┘
                          ▼
                invariants + logging.trace
                          │
                          ▼
        env.episode/reward -> eventflow.engine
                          │
                 scripted/checkpoint/replay
                          │
                          ▼
                         cli
```

`eventflow.engine`, `eventflow.checkpoint`, and `eventflow.replay` are the only
runtime modules allowed to consume the private `EpisodeBundle` or post-run
private artifact. `eventflow.scripted`, `eventflow.flow`, `eventflow.guards`,
`eventflow.jumps`, `eventflow.state`, and every `memory` module may not import
`env.oracle`, `env.generator`, `env.invariants`, `env.services`,
`EpisodeTruth`, or `EpisodeBundle`.

## Coverage Matrix

| Canonical requirement | Owning task(s) | Evidence |
|---|---|---|
| §5.2 state dimensions and exact event kinds | 1–2 | Strict config/state/schema tests |
| §5.3 stable closed-form flow and semigroup | 2 | CPU reference, huge/tiny-`dt`, property, and MPS tests |
| §5.4 analytic endogenous guards | 3 | Scalar/tensor threshold and autonomous scheduling tests |
| §5.5 dormancy, margins, refractory, Zeno limits | 3, 9 | Dormancy, clamp streak, cap, and crash tests |
| §5.6 exact tie order | 6 | Complete equal/near-equal tie matrix |
| §5.7 legal modes and observing mask | 5, 7, 9 | Transition matrix and callback integration tests |
| §5.8 typed jumps | 5, 8 | Jump unit tests and full scripted traces |
| §5.9 oracle/scaffolding boundary | 7–8, 12 | Static imports and spy callbacks |
| §5.10 bounded typed memory | 4 | Capacity/mask/refractory/provenance tests; embeddings/eviction deferred |
| §6.1 finite causal event loop | 6–9 | Queue/engine/preemption/dormancy tests |
| §6.2 IDs, parents, and prediction snapshots | 3, 6, 10 | Exact trace and replay checks |
| §6.3 non-mutating arbitrary-time state | 2, 10 | Alias checks and trajectory fixtures |
| §6.4 paused-world checkpoint semantics | 11 | Mid-flow/per-kind exact continuation tests |
| §6.5 CPU replay | 10–12 | Library and installed CLI replay tests |
| §6.6 post-jump invariants/crash bundles | 9 | Mutation matrix and bounded last-20 event tests |
| §7 agent protocols/private terminal | 1, 7 | Signature and sentinel spy tests |
| §12.1–12.2 numerical/scheduler stress | 2–3, 6, 9, 12 | Unit/property/MPS/Zeno/long-silence gates |
| §12.5 checkpoint/process behavior | 10–11 | Corruption, repeat replay, and wall-pause tests |
| §15.2 `replay` command | 12 | CliRunner and installed executable tests |
| §17 Phase 2 gate | 13–14 | Source-bound 10,000-episode artifact and independent verifier |

---

### Task 1: Strict Phase 2 Configuration and Runtime Protocols

**Files:**

- Create: `configs/model/event_flow.yaml`
- Create: `src/silent_cascade/eventflow/__init__.py`
- Create: `src/silent_cascade/eventflow/config.py`
- Create: `src/silent_cascade/eventflow/protocols.py`
- Create: `tests/unit/test_phase2_config.py`
- Create: `tests/unit/test_eventflow_protocols.py`
- Modify: `docs/PLAN.md`

**Interfaces:**

- Consumes: `Phase1Config`, `ProjectConfig`, `AgentInit`, `ExternalEvent`,
  `InternalEvent`, `Action`, `Condition`, and `StrictModel`.
- Produces: `Phase2Config`, `EventFlowConfig`, `StateDimensionsConfig`,
  `FlowDynamicsConfig`, `GuardDynamicsConfig`, `ScriptedAgentConfig`,
  `AgentCondition`, and `ClosedFormFlow`.

- [ ] **Step 1: Mark Phase 2 in progress only after approval**

Replace the proposed Phase 2 gate cell in `docs/PLAN.md` with exactly:

```text
In progress under the approved Phase 2 flow-and-event-engine plan. No Phase 2 acceptance artifact may exist until Tasks 1–13 are committed, reviewed, and locally verified.
```

Do not touch the completed Phase 1 row.

- [ ] **Step 2: Write failing strict-config and protocol tests**

The config test must resolve base, primary data, and EventFlow layers and assert
all exact values:

```python
def test_event_flow_config_resolves_exact_phase2_contract() -> None:
    resolved = resolve_config(
        Phase2Config,
        [
            Path("configs/base.yaml"),
            Path("configs/data/primary.yaml"),
            Path("configs/model/event_flow.yaml"),
        ],
    )
    cfg = resolved.config.event_flow
    assert cfg.dimensions.model_dump() == {
        "z_fast": 256,
        "z_slow": 64,
        "drives": 8,
        "guard_accumulators": 3,
        "focus_key": 64,
        "hypothesis_latent": 64,
    }
    assert (cfg.flow.rate_min, cfg.flow.rate_max) == (1.0e-5, 20.0)
    assert cfg.flow.state_min == -1.0
    assert cfg.flow.state_max == 1.0
    assert cfg.guards.threshold == 1.0
    assert (cfg.guards.rate_min, cfg.guards.rate_max) == (1.0e-5, 500.0)
    assert cfg.guards.active_margin == 1.10
    assert cfg.guards.inactive_margin == 0.90
    assert cfg.guards.minimum_internal_gap == 1.0e-4
    assert cfg.guards.same_kind_refractory == 1.0e-3
    assert cfg.guards.near_tie_tolerance == 1.0e-9
    assert cfg.guards.maximum_consecutive_gap_clamps == 4
    assert cfg.max_internal_events == 64
```

Also reject unknown keys, bool-as-int, float strings, float64 runtime dtype,
three-versus-four guard accumulators, an EventFlow event cap different from the
Phase 0/1 ceiling, a memory cap above 64, and an action-target fraction that
does not equal the data timing contract.

Protocol tests must inspect the exact method names and prove there is no
external horizon parameter:

```python
def test_agent_protocol_does_not_receive_external_horizon() -> None:
    parameters = inspect.signature(AgentCondition.next_internal_event).parameters
    assert tuple(parameters) == ("self", "state")
    assert "external_horizon" not in parameters
```

- [ ] **Step 3: Run the focused tests and confirm RED**

```bash
uv run pytest -q \
  tests/unit/test_phase2_config.py \
  tests/unit/test_eventflow_protocols.py
```

Expected: collection fails because the Phase 2 package/config/protocols do not
exist.

- [ ] **Step 4: Implement strict configuration**

Use exact Pydantic models with `extra="forbid"` inherited from `StrictModel`.
`Phase2Config` extends `Phase1Config` and cross-validates duplicate limits:

```python
class Phase2Config(Phase1Config):
    event_flow: EventFlowConfig

    @model_validator(mode="after")
    def validate_phase2_limits(self) -> Self:
        if self.event_flow.max_internal_events != self.limits.max_eventflow_events:
            raise ValueError("EventFlow event ceilings must match")
        if self.data.primary_memory_capacity != self.limits.primary_memory_records:
            raise ValueError("memory ceilings must match")
        if (
            self.event_flow.scripted.action_target_fraction
            != self.data.oracle_timing.action_target_fraction
        ):
            raise ValueError("scripted target must match the standing OFD objective")
        return self
```

The YAML contains only scientific/runtime values, not machine-specific paths
or evidence roots.

- [ ] **Step 5: Implement runtime protocols**

Use `typing.Protocol` and forward type imports guarded by `TYPE_CHECKING`:

```python
class AgentCondition(Protocol):
    name: Condition

    def initialize(self, init: AgentInit) -> RuntimeState: ...
    def on_external(self, state: RuntimeState, event: ExternalEvent) -> RuntimeState: ...
    def next_internal_event(self, state: RuntimeState) -> InternalEvent | None: ...
    def on_internal(
        self, state: RuntimeState, event: InternalEvent
    ) -> tuple[RuntimeState, list[Action]]: ...
    def compute_counters(self) -> ComputeCounters: ...


class ClosedFormFlow(Protocol):
    def advance(self, state: ContinuousState, dt: float) -> ContinuousState: ...
    def next_crossings(self, state: ContinuousState) -> list[GuardCrossing]: ...
```

Do not add a callback that reveals a session, bundle, queue, horizon, trace, or
score.

- [ ] **Step 6: Run GREEN verification and commit**

```bash
uv run pytest -q \
  tests/unit/test_phase2_config.py \
  tests/unit/test_eventflow_protocols.py \
  tests/unit/test_config.py \
  tests/unit/test_phase1_config.py
uv run ruff check src/silent_cascade/eventflow tests/unit/test_phase2_config.py \
  tests/unit/test_eventflow_protocols.py
uv run ruff format --check src/silent_cascade/eventflow \
  tests/unit/test_phase2_config.py tests/unit/test_eventflow_protocols.py
git diff --check
git add configs/model/event_flow.yaml src/silent_cascade/eventflow \
  tests/unit/test_phase2_config.py tests/unit/test_eventflow_protocols.py docs/PLAN.md
git commit -m "feat: define Phase 2 runtime contracts"
```

Expected: focused config/protocol and unchanged Phase 0/1 config tests pass.

---

### Task 2: Anchored Continuous State and Stable Closed-Form Flow

**Files:**

- Create: `src/silent_cascade/eventflow/state.py`
- Create: `src/silent_cascade/eventflow/flow.py`
- Create: `tests/unit/test_eventflow_state.py`
- Create: `tests/unit/test_flow.py`
- Create: `tests/property/test_eventflow_properties.py`

**Interfaces:**

- Produces: `ContinuousChannels`, `ContinuousState`, `SegmentParameters`,
  `AnalyticSegment`, `ComputeCounters`, `make_initial_continuous_state`,
  `start_segment`, `advance_to`, and `state_at`.
- `flow.py` imports no environment, generator, oracle, memory, trace, or CLI
  module.

- [ ] **Step 1: Write failing shape, dtype, identity, and reference tests**

Create an independent float64 scalar reference in the test module only. Cover:

- exact shapes `(256,)`, `(64,)`, `(8,)`, `(3,)`, `(64,)`, `(64,)`;
- all production tensors float32 on one declared device;
- invalid shape, dtype, device mixture, nonfinite value, target outside
  `[-1,1]`, or rate outside its bound fails loudly;
- `dt=0` returns the same state object and does not increment the flow counter;
- negative time raises `TimeOrderError`;
- direct and split flow agree within declared float32 tolerance;
- anchored direct evaluation reaches bit-identical CPU tensors after a
  materialized mid-segment snapshot;
- `dt in {1e3, 1e6, 1e12}` approaches target without overflow;
- `dt in {1e-15, 1e-12, 1e-9}` is finite and agrees with the reference; and
- the input state and tensors are not mutated or aliased by a nonzero advance.

Representative assertions:

```python
advanced = advance_to(runtime, runtime.time + 0.25)
weight = -torch.expm1(
    -runtime.segment.parameters.flow_rates.z_fast
    * runtime.core.continuous.z_fast.new_tensor(0.25)
)
expected = runtime.segment.origin.z_fast + weight * (
    runtime.segment.parameters.flow_targets.z_fast - runtime.segment.origin.z_fast
)
torch.testing.assert_close(advanced.core.continuous.z_fast, expected)
assert runtime.core.continuous.z_fast.data_ptr() != advanced.core.continuous.z_fast.data_ptr()
```

- [ ] **Step 2: Run tests and confirm RED**

```bash
uv run pytest -q \
  tests/unit/test_eventflow_state.py \
  tests/unit/test_flow.py \
  tests/property/test_eventflow_properties.py
```

Expected: missing state/flow types and functions.

- [ ] **Step 3: Implement immutable state containers**

Use frozen, slotted dataclasses whose tensor values are defensively cloned at
public construction boundaries. The segment holds the immutable origin, not
only the current snapshot:

```python
@dataclass(frozen=True, slots=True)
class AnalyticSegment:
    started_at: float
    origin: ContinuousState
    parameters: SegmentParameters
    parent_event_id: int
    prediction_snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class RuntimeState:
    core: RuntimeCore
    segment: AnalyticSegment
    time: float
```

Task 5 fills `RuntimeCore`'s discrete fields; in this task define its continuous
minimum without importing future modules. Prefer one stable final field layout
over a temporary compatibility shim.

- [ ] **Step 4: Implement stable analytic advance**

Use only float32 device operations for production tensors:

```python
def _flow_tensor(
    origin: torch.Tensor,
    target: torch.Tensor,
    rate: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    dt_tensor = origin.new_tensor(dt)
    weight = -torch.expm1(-rate * dt_tensor)
    return origin + weight * (target - origin)
```

Calculate `elapsed = target_time - segment.started_at` as a Python float, then
evaluate every channel and accumulator from the segment origin. Do not invoke a
controller in `advance_to`; only `start_segment` installs controller outputs.
At zero elapsed, preserve identity. Validate finite host times before making a
tensor.

- [ ] **Step 5: Add property coverage**

Use bounded Hypothesis strategies, no filtering on outcomes, for semigroup,
bounds, finite huge-time behavior, monotonic host time, and non-mutation. Cap
examples so routine local verification remains fast.

- [ ] **Step 6: Run GREEN verification and commit**

```bash
uv run pytest -q \
  tests/unit/test_eventflow_state.py \
  tests/unit/test_flow.py \
  tests/property/test_eventflow_properties.py
uv run ruff check src/silent_cascade/eventflow tests/unit/test_eventflow_state.py \
  tests/unit/test_flow.py tests/property/test_eventflow_properties.py
uv run ruff format --check src/silent_cascade/eventflow \
  tests/unit/test_eventflow_state.py tests/unit/test_flow.py \
  tests/property/test_eventflow_properties.py
git diff --check
git add src/silent_cascade/eventflow/state.py \
  src/silent_cascade/eventflow/flow.py \
  tests/unit/test_eventflow_state.py tests/unit/test_flow.py \
  tests/property/test_eventflow_properties.py
git commit -m "feat: implement anchored analytic flow"
```

---

### Task 3: Exact Endogenous Guard Crossings and Dormancy

**Files:**

- Create: `src/silent_cascade/eventflow/guards.py`
- Create: `tests/unit/test_guards.py`
- Modify: `tests/property/test_eventflow_properties.py`

**Interfaces:**

- Produces: `GuardCrossing`, `GUARD_KIND_BY_INDEX`,
  `crossing_offset_host`, `crossing_offsets_tensor`, `next_crossings`, and
  `prediction_snapshot_sha256`.
- Consumes only current segment/state/config; accepts no external horizon.

- [ ] **Step 1: Write failing scalar and tensor crossing tests**

Cover `A < 1`, `A == 1`, `A > 1`, `a0 == 0`, `a0 -> 1`, `a0 >= 1`, rates at
both bounds, invalid/nonfinite values, and threshold accuracy against an
80-digit `decimal.Decimal` reference. Patch `math.log1p` with a spy and prove it
is never called for dormant guards.

Test the stable host formula exactly:

```python
expected = math.log1p((1.0 - a) / (asymptote - 1.0)) / rate
assert crossing_offset_host(a, asymptote, rate) == expected
```

The differentiable float32 function must evaluate only active masked elements,
produce `inf` for dormant guards, propagate finite gradients through active
values, and stay float32 on CPU/MPS.

- [ ] **Step 2: Write failing race, mask, and refractory tests**

Assert:

- `OBSERVING`, `QUIESCENT`, and `TERMINAL` yield no crossings;
- `SEARCHING` permits only RECALL, `HAVE_MEMORY` only COMPOSE, and
  `HOLDING_HAZARD` only ACT;
- crossings within tolerance use `ACT > COMPOSE > RECALL` priority;
- an already-crossed guard held by same-kind refractory fires at
  `refractory_until`, not never and not before release;
- predicted absolute time and `predicted_delta` use host float64;
- parent ID and prediction hash come from the segment; and
- `InternalEventKind.NOOP` is never produced.

- [ ] **Step 3: Run tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_guards.py \
  tests/property/test_eventflow_properties.py
```

Expected: guard module and crossings are missing.

- [ ] **Step 4: Implement guarded masked math**

Validate `a < 1 and A > 1` before evaluating:

```python
def crossing_offset_host(a: float, asymptote: float, rate: float) -> float | None:
    _validate_guard_scalar(a, asymptote, rate)
    if not a < 1.0 or not asymptote > 1.0:
        return None
    return math.log1p((1.0 - a) / (asymptote - 1.0)) / rate
```

For tensor training math, allocate an `inf` result and index only the active
mask before `torch.log1p`; never rely on `torch.where` around an invalid branch.

- [ ] **Step 5: Implement absolute race construction**

Calculate crossings from the segment origin, then require the resulting
absolute time to be no earlier than current runtime time. Apply mode masks and
same-kind release after the mathematical crossing. Deterministically derive the
internal ID from `INTERNAL_EVENT_ID_BASE + executed_internal_count`; do not
consume it until execution.

Hash a canonical snapshot of float32 guard target/rate bytes, origin
accumulator bytes, segment start, allowed-mode mask, and parent ID. Include
dtype/shape framing so concatenation is unambiguous.

- [ ] **Step 6: Run GREEN verification and commit**

```bash
uv run pytest -q tests/unit/test_guards.py \
  tests/unit/test_flow.py tests/property/test_eventflow_properties.py
uv run ruff check src/silent_cascade/eventflow/guards.py \
  tests/unit/test_guards.py tests/property/test_eventflow_properties.py
uv run ruff format --check src/silent_cascade/eventflow/guards.py \
  tests/unit/test_guards.py tests/property/test_eventflow_properties.py
git diff --check
git add src/silent_cascade/eventflow/guards.py tests/unit/test_guards.py \
  tests/property/test_eventflow_properties.py
git commit -m "feat: implement exact endogenous guards"
```

---

### Task 4: Bounded Typed Runtime Memory and Provenance

**Files:**

- Create: `src/silent_cascade/memory/__init__.py`
- Create: `src/silent_cascade/memory/store.py`
- Create: `src/silent_cascade/memory/provenance.py`
- Create: `tests/unit/test_runtime_memory.py`
- Modify: `tests/property/test_eventflow_properties.py`
- Modify: `tests/regression/test_import_boundaries.py`

**Interfaces:**

- Produces: `RuntimeMemoryRecord`, `BoundedMemory`, `append_perceived_fact`,
  `mark_recalled`, `mark_consumed`, `legal_records`, `require_support_ledger`,
  and `require_provenance_preserved`.
- Consumes immutable `MemoryRecord` and public `FACT`; introduces no embedding,
  scorer, learned ranking, or eviction.

- [ ] **Step 1: Write failing memory/provenance tests**

Cover:

- deterministic FACT-to-record conversion and presentation-order record IDs;
- capacity exactly 64, duplicate ID rejection, and fail-loud 65th insertion;
- invalid, consumed, illegal-subject, and refractory records masked;
- recall sets but never shortens `refractory_until`;
- compose consumption cannot be undone;
- lookup of a missing ID fails loudly;
- records and `PERCEIVED` provenance are immutable;
- inferred hypotheses require unique existing support IDs;
- a support record cannot refer to an invalid/unknown record; and
- attempted `INFERRED`/`SIMULATED -> PERCEIVED` escalation is rejected.

Do not assert eviction behavior; Phase 2 overflow is an error.

- [ ] **Step 2: Extend static import tests before implementation**

Parse every new `memory` module with the existing AST helper and reject imports
of `env.oracle`, `env.generator`, `env.invariants`, `env.services`, or any
optional foundation package.

- [ ] **Step 3: Run tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_runtime_memory.py \
  tests/property/test_eventflow_properties.py \
  tests/regression/test_import_boundaries.py
```

Expected: missing memory package/types; the import regression must already be
capable of scanning the future directory.

- [ ] **Step 4: Implement metadata-only bounded memory**

Use an immutable tuple in record-arrival order and a separate immutable runtime
wrapper:

```python
@dataclass(frozen=True, slots=True)
class RuntimeMemoryRecord:
    record: MemoryRecord
    refractory_until: float = 0.0
    valid: bool = True
    consumed: bool = False


@dataclass(frozen=True, slots=True)
class BoundedMemory:
    capacity: int
    records: tuple[RuntimeMemoryRecord, ...] = ()
```

Return new wrappers for every metadata update. Never mutate the underlying
`MemoryRecord`. `legal_records` applies explicit kind/subject/consumed/
refractory masks but does not choose a winner.

- [ ] **Step 5: Add randomized state-machine properties**

Generate legal append/recall/consume sequences and prove IDs remain unique,
support IDs remain valid, metadata transitions are monotone, and capacity is
never exceeded without a typed error.

- [ ] **Step 6: Run GREEN verification and commit**

```bash
uv run pytest -q tests/unit/test_runtime_memory.py \
  tests/property/test_eventflow_properties.py \
  tests/regression/test_import_boundaries.py
uv run ruff check src/silent_cascade/memory tests/unit/test_runtime_memory.py \
  tests/property/test_eventflow_properties.py tests/regression/test_import_boundaries.py
uv run ruff format --check src/silent_cascade/memory \
  tests/unit/test_runtime_memory.py tests/property/test_eventflow_properties.py \
  tests/regression/test_import_boundaries.py
git diff --check
git add src/silent_cascade/memory tests/unit/test_runtime_memory.py \
  tests/property/test_eventflow_properties.py tests/regression/test_import_boundaries.py
git commit -m "feat: add bounded runtime memory"
```

---

### Task 5: Typed Jumps, Modes, and Segment Reparameterization

**Files:**

- Modify: `src/silent_cascade/eventflow/state.py`
- Create: `src/silent_cascade/eventflow/jumps.py`
- Create: `tests/unit/test_jumps.py`
- Modify: `tests/property/test_eventflow_properties.py`

**Interfaces:**

- Produces: final `RuntimeCore`/`RuntimeState`, `ComposeRole`,
  `ComposeDecision`, `apply_fact`, `apply_activate`, `apply_recall`,
  `apply_compose`, `apply_act`, and `begin_post_jump_segment`.
- Jump functions consume only agent-visible events/state and explicit agent
  decisions. They do not inspect hidden graph truth or select relevant memory.

- [ ] **Step 1: Write the failing transition matrix**

Test every legal source/event/destination and representative illegal edges:

| Event | Required source | Legal destination |
|---|---|---|
| FACT | OBSERVING | OBSERVING |
| ACTIVATE | OBSERVING | SEARCHING |
| RECALL | SEARCHING | HAVE_MEMORY |
| COMPOSE link | HAVE_MEMORY | SEARCHING or QUIESCENT |
| COMPOSE hazard | HAVE_MEMORY | HOLDING_HAZARD |
| COMPOSE safe | HAVE_MEMORY | QUIESCENT |
| COMPOSE irrelevant/contradictory | HAVE_MEMORY | SEARCHING or QUIESCENT |
| ACT | HOLDING_HAZARD | QUIESCENT |
| private terminal | any post-activation nonterminal mode | TERMINAL |

Also prove FACT after activation, double activation, COMPOSE without an active
record, ACT without a hazard, action before activation, a second action,
endogenous NOOP, and any endogenous event from OBSERVING/QUIESCENT/TERMINAL
raise `DynamicsError`.

- [ ] **Step 2: Write failing state-effect tests**

Assert:

- FACT writes one `PERCEIVED` record and applies bounded deterministic
  injections to `z_fast`/`z_slow` while staying OBSERVING;
- ACTIVATE stores activation time/current focus, sets a deterministic focus
  key, and enters SEARCHING;
- RECALL copies one explicitly selected record ID into the active slot, marks
  record and same-kind refractory, and enters HAVE_MEMORY;
- link composition uses an explicit predicted next focus and append decision;
- hazard composition builds an `INFERRED` typed hypothesis with deadline,
  confidence, and nonempty supports;
- safe composition builds a safe inferred hypothesis and quiesces;
- irrelevant/contradictory decisions cannot fabricate `PERCEIVED` truth;
- ACT emits exactly one action caused by its internal event and disables later
  ACT; and
- every nonterminal jump resets all accumulators to exact float32 zero and
  installs a segment whose parent is the just-executed event.

Use counterfactual predicted focus/class values in tests to prove the common
jump layer does not silently read `MemoryRecord.object_id`/`hazard_type` as the
answer. It validates schema and provenance, while scoring later determines
correctness.

- [ ] **Step 3: Run tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_jumps.py \
  tests/property/test_eventflow_properties.py
```

Expected: jump types/functions and the final discrete runtime layout are
missing.

- [ ] **Step 4: Finalize the runtime core**

Use one immutable discrete core:

```python
@dataclass(frozen=True, slots=True)
class RuntimeCore:
    continuous: ContinuousState
    mode: Mode
    memory: BoundedMemory
    focus_node_id: int | None
    active_record_id: int | None
    support_ids: tuple[int, ...]
    hypothesis: Hypothesis | None
    activation_time: float | None
    actions: tuple[Action, ...]
    same_kind_refractory_until: tuple[float, float, float]
    executed_internal_events: int
    consecutive_gap_clamps: int
    last_event_id: int | None
    last_event_time: float | None
    counters: ComputeCounters
```

No field may contain an episode variant, truth object, terminal timestamp,
window, external queue, seed, coordinate, or oracle trace.

Approved Task 5 review ruling: `last_event_time` records the timestamp of the
already-executed causal event, alongside its ID. It is an optional finite,
nonnegative host float, with no invented causal history in low-level flow
fixtures. Every public jump stamps it; the installer requires it and rejects a
supplied time that differs. This preserves the materialized jump origin's time
without introducing private future timing or changing the scientific protocol.

- [ ] **Step 5: Implement decision-driven jumps**

`ComposeDecision` carries the agent's predicted role/focus/hazard/deadline/
confidence/append choice. Common jump code checks types, mode, active-record
existence, support validity, and provenance, but never corrects a prediction
using private truth.

Use deterministic fixed public-feature impulses only for the scripted harness:
construct zero float32 vectors on the state's device, add bounded values at
indices derived arithmetically from public kind/entity/record IDs, and apply
`torch.tanh`. Do not use Python's randomized `hash()`.

- [ ] **Step 6: Implement one segment install point**

`begin_post_jump_segment` is the only way to turn a post-jump core into a
runnable `RuntimeState`. It must:

1. set all accumulators to exact zero;
2. validate bounded targets and positive rates;
3. require `time == core.last_event_time` and set `started_at` to that
   already-executed event timestamp; reject missing, invalid, or mismatched times;
4. clone the post-jump continuous origin;
5. bind `parent_event_id` to the executed event; and
6. derive the prediction snapshot hash from the newly installed parameters.

`dt=0` between tied events uses the existing segment without another install;
the next causal jump then installs exactly one replacement segment.

- [ ] **Step 7: Run GREEN verification and commit**

```bash
uv run pytest -q tests/unit/test_jumps.py tests/unit/test_runtime_memory.py \
  tests/unit/test_eventflow_state.py tests/property/test_eventflow_properties.py
uv run ruff check src/silent_cascade/eventflow/state.py \
  src/silent_cascade/eventflow/jumps.py tests/unit/test_jumps.py
uv run ruff format --check src/silent_cascade/eventflow/state.py \
  src/silent_cascade/eventflow/jumps.py tests/unit/test_jumps.py
git diff --check
git add src/silent_cascade/eventflow/state.py \
  src/silent_cascade/eventflow/jumps.py tests/unit/test_jumps.py \
  tests/property/test_eventflow_properties.py
git commit -m "feat: implement typed event-flow jumps"
```

---

### Task 6: Private Event Queue, Tie Policy, and Causal Trace Contracts

**Files:**

- Create: `src/silent_cascade/eventflow/scheduling.py`
- Create: `src/silent_cascade/logging/trace.py`
- Modify: `src/silent_cascade/logging/__init__.py`
- Create: `tests/unit/test_scheduling.py`
- Create: `tests/unit/test_trace.py`
- Modify: `tests/property/test_eventflow_properties.py`

**Interfaces:**

- Produces: `ExternalEventQueue`, `PredictionCache`, `ScheduledChoice`,
  `TieResolution`, `choose_next_event`, `CausalEventSummary`,
  `SegmentSummary`, `CausalTrace`, `TraceRecorder`, and
  `sanitized_crash_events`.

- [ ] **Step 1: Write failing external-queue privacy tests**

Build the queue only from `PublicEpisode.events` plus the engine-supplied
private terminal. Assert chronological heap behavior, unique IDs, terminal
presence, and no mutation of the episode. Do not expose iteration, peek payload,
or terminal fields through any object passed to an agent. A queue snapshot is
checkpoint-private.

Test an earlier external fact/activation preempts a cached internal event; the
cache is invalid after the external jump and the agent is called once at the
new post-event boundary. A noncausal pause preserves the cache.

- [ ] **Step 2: Write the complete failing tie matrix**

Parameterize equal and `±0.5e-9` pairs for all relevant kinds. Assert:

```python
EXPECTED_PRIORITY = (
    "terminal",
    "external",
    InternalEventKind.ACT,
    InternalEventKind.COMPOSE,
    InternalEventKind.RECALL,
    InternalEventKind.NOOP,
)
```

At `1.1e-9`, ordinary timestamp order wins. Ties must record original times,
candidate identities, winner, and tolerance. Terminal-before-ACT and
external-before-every-internal are mandatory. EventFlow NOOP is rejected even
when it would lose the tie.

- [ ] **Step 3: Write failing trace tests**

Trace tests must cover:

- one record per executed causal event, no pause/snapshot records;
- unique event IDs and exact parent IDs;
- nondecreasing timestamps with same-time order matching policy;
- pre/post modes, delta, selected record/rank, support/hypothesis changes,
  actions, state norm/hash, counter delta, prediction snapshot, and tie ID;
- online terminal kind serialized only as `terminal` with `payload=None`;
- last-events crash projection bounded to 20; and
- exact canonical trace hash stable across dict/hash seed/order changes.

- [ ] **Step 4: Run tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_scheduling.py tests/unit/test_trace.py \
  tests/property/test_eventflow_properties.py
```

Expected: scheduling/trace modules are absent.

- [ ] **Step 5: Implement queue and selection**

Use `heapq` with `(timestamp, external_priority, event_id, event)` tuples.
Cache either one `InternalEvent` or an explicit dormant sentinel, both bound to
the segment prediction hash and parent ID. Never call
`next_internal_event` twice for the same unchanged segment.

`choose_next_event` compares original host floats. A selected external event
removes only that heap item. A selected internal event never came from the heap.
After any executed event, invalidate the cache; after a checkpoint cursor move,
retain it.

- [ ] **Step 6: Implement deterministic trace summaries**

Hash tensor values with explicit dtype/shape framing after copying contiguous
bytes to CPU. Keep full tensors out of JSON traces. Store sufficient causal
fields for rerun comparison and separate the sanitized live summary from the
authorized private replay envelope introduced in Task 12.

- [ ] **Step 7: Run GREEN verification and commit**

```bash
uv run pytest -q tests/unit/test_scheduling.py tests/unit/test_trace.py \
  tests/unit/test_guards.py tests/property/test_eventflow_properties.py
uv run ruff check src/silent_cascade/eventflow/scheduling.py \
  src/silent_cascade/logging tests/unit/test_scheduling.py tests/unit/test_trace.py
uv run ruff format --check src/silent_cascade/eventflow/scheduling.py \
  src/silent_cascade/logging tests/unit/test_scheduling.py tests/unit/test_trace.py
git diff --check
git add src/silent_cascade/eventflow/scheduling.py \
  src/silent_cascade/logging tests/unit/test_scheduling.py \
  tests/unit/test_trace.py tests/property/test_eventflow_properties.py
git commit -m "feat: add causal scheduling and traces"
```

---

### Task 7: Event Engine, Private Terminal Dispatch, and Callback Isolation

**Files:**

- Create: `src/silent_cascade/eventflow/engine.py`
- Create: `tests/integration/test_event_engine.py`
- Modify: `tests/regression/test_import_boundaries.py`
- Modify: `tests/property/test_eventflow_properties.py`

**Interfaces:**

- Produces: `RuntimeSession`, `EpisodeResult`, and `EventEngine` methods
  `start_episode`, `next_internal_event`, `step`, `run_until`, `run_episode`,
  and `advance_to`.
- Consumes the complete `EpisodeBundle` only inside `eventflow.engine`; passes
  only public initialization/events and `RuntimeState` to the agent.

- [ ] **Step 1: Write a sentinel spy agent before the engine**

The spy records every callback argument and accesses every dataclass field it
receives. Construct a bundle whose private fields contain unique sentinels.
Assert:

- `initialize` receives the identical `bundle.public.init` only;
- `on_external` receives FACT/ACTIVATE only, never OUTCOME/END;
- `next_internal_event` receives only `RuntimeState`, no horizon/session;
- `on_internal` receives only state and its own internal event;
- callback state contains none of the private sentinel values or fields;
- the engine alone calls `score_actions(bundle.truth, actions)`; and
- a private terminal ends execution even when the agent is dormant.

Extend AST import checks so every agent-facing module is forbidden from
importing private environment modules. Permit `eventflow.engine` to import only
`env.episode` and `env.reward`, never `env.oracle` or `env.generator`.

- [ ] **Step 2: Write failing finite-loop integration tests**

Use tiny deterministic test agents to cover:

- all-dormant direct advance to the next external event;
- internal-before-external execution;
- external preemption and rescheduling;
- outcome/END terminal scoring and `TERMINAL` mode;
- ACT exactly tied with outcome loses;
- positive action-window half-open scoring;
- no callback after terminal;
- one cached guard calculation per causal boundary; and
- no fixed-grid opportunity or repeated polling during long silence.

- [ ] **Step 3: Run tests and confirm RED**

```bash
uv run pytest -q tests/integration/test_event_engine.py \
  tests/regression/test_import_boundaries.py \
  tests/property/test_eventflow_properties.py
```

Expected: the engine/session/result types do not exist.

- [ ] **Step 4: Implement the engine-private session**

`RuntimeSession` must never be passed to the agent:

```python
@dataclass(slots=True)
class RuntimeSession:
    public_id: str
    truth: EpisodeTruth
    state: RuntimeState
    external_queue: ExternalEventQueue
    prediction_cache: PredictionCache
    trace: TraceRecorder
    pause_cursor: float
    terminal_score: EpisodeScore | None = None
```

The session may carry private truth because it is engine-private. It must not be
returned as the public `EpisodeResult` or appear in an online trace/crash event.

- [ ] **Step 5: Implement one-event stepping**

Use a finite method, not an unbounded polling task:

```python
def step(self, session: RuntimeSession, agent: AgentCondition) -> bool:
    internal = self._cached_or_compute_internal(session, agent)
    external = session.external_queue.peek()
    chosen = choose_next_event(external, internal, self.config.guards)
    session.state = advance_to(session.state, chosen.timestamp)
    before = session.state
    if chosen.is_private_terminal:
        session.terminal_score = score_actions(session.truth, before.core.actions)
        session.state = self._terminalize(before, chosen)
    elif chosen.is_external:
        session.state = agent.on_external(before, chosen.external_event)
    else:
        session.state, emitted = agent.on_internal(before, chosen.internal_event)
        self._require_emitted_matches_state(session.state, emitted)
    self._record_and_invalidate(session, chosen, before)
    return session.state.core.mode is Mode.TERMINAL
```

The actual implementation must dispatch and trace exceptions safely and check
invariants in Task 9; this step establishes the callback/private seam.

- [ ] **Step 6: Implement `run_until` without creating an event**

Process every causal event with timestamp `<= pause_time` according to the tie
policy. If the next event is later, materialize current continuous state from
the unchanged segment anchor at `pause_time`, set only the session pause cursor,
preserve cached prediction/dormancy, and return. Refuse a pause before current
time or after a terminal session. Wall-clock time is never consulted.

- [ ] **Step 7: Run GREEN verification and commit**

```bash
uv run pytest -q tests/integration/test_event_engine.py \
  tests/unit/test_scheduling.py tests/unit/test_trace.py \
  tests/regression/test_import_boundaries.py \
  tests/property/test_eventflow_properties.py
uv run ruff check src/silent_cascade/eventflow/engine.py \
  tests/integration/test_event_engine.py tests/regression/test_import_boundaries.py
uv run ruff format --check src/silent_cascade/eventflow/engine.py \
  tests/integration/test_event_engine.py tests/regression/test_import_boundaries.py
git diff --check
git add src/silent_cascade/eventflow/engine.py \
  tests/integration/test_event_engine.py tests/regression/test_import_boundaries.py \
  tests/property/test_eventflow_properties.py
git commit -m "feat: implement private causal event engine"
```

---

### Task 8: Public-Facts-Only Scripted Guard-Driven Agent

**Files:**

- Create: `src/silent_cascade/eventflow/scripted.py`
- Create: `tests/integration/test_scripted_agent.py`
- Modify: `tests/regression/test_import_boundaries.py`
- Create: `tests/fixtures/phase2/scripted_cases.json`
- Create: `tests/regression/test_phase2_fixtures.py`

**Interfaces:**

- Produces: `ScriptedEventFlowAgent`, deterministic public-memory selection,
  public composition decisions, and scripted segment parameters.
- The module imports no `silent_cascade.env.*` package and carries no oracle
  dependency, even in tests or type-only imports.

- [ ] **Step 1: Write failing hand-authored trace fixtures**

Commit at least one positive, safe-negative, and disconnected-negative public
episode with expected endogenous grammar, selected public record IDs, support
ledger, hypothesis, action, and final timed outcome. Derive the expected values
by hand in the fixture; do not call `env.oracle` to create them.

Expected positive grammar for `L` links:

```text
(RECALL, COMPOSE) * L
RECALL terminal
COMPOSE terminal
ACT
```

Safe stops after terminal COMPOSE. Disconnected traverses its candidate links
and becomes dormant with no fabricated terminal hypothesis or action.

- [ ] **Step 2: Write failing generated-episode integration tests**

Using Phase 1 generators only in the test harness, not the agent, cover all
three variants, path lengths 2–4, short and long delays, distractors, shuffled
fact order, and clock-scaled pairs. Assert:

- no endogenous event before activation;
- irregular gaps in `[0.05, 0.068]` are produced by guard crossings;
- positive actions are exactly one/correct/in-window;
- negatives emit no action;
- selected records form a public subject/focus chain;
- swapping public terminal delays moves action time accordingly; and
- `compute_counters().foundation_model_calls == 0`.

An AST/source test must fail if `scripted.py` imports any `env` module or
contains the strings `oracle`, `EpisodeTruth`, or `EpisodeBundle` as symbols.

- [ ] **Step 3: Run tests and confirm RED**

```bash
uv run pytest -q tests/integration/test_scripted_agent.py \
  tests/regression/test_phase2_fixtures.py \
  tests/regression/test_import_boundaries.py
```

Expected: scripted module and Phase 2 fixtures are missing.

- [ ] **Step 4: Implement deterministic public retrieval and composition**

For SEARCHING:

```python
candidates = memory.legal_records(
    subject_id=state.core.focus_node_id,
    at_time=state.time,
)
selected = min(candidates, key=lambda slot: (-slot.record.confidence, slot.record.record_id))
```

If there is no candidate, return a quiescent rescheduled state with all guards
dormant. At RECALL crossing, recompute this choice from current public-derived
state, record its deterministic rank, and pass its ID explicitly to
`apply_recall`. At COMPOSE, explicitly create a `ComposeDecision` from the
active public record. Do not cache a hidden path.

- [ ] **Step 5: Implement scripted segment parameters**

Set all guard targets to a dormant value at or below `0.90`, then enable only
the mode-legal guard with `A=1.5`. For public-derived desired gap `delta`:

```python
guard_rate = math.log(3.0) / delta
```

Use a stable documented integer formula over record ID/focus/event ordinal so
RECALL/COMPOSE gaps fall in `[0.05, 0.068]` and are not a constant cadence.
For HOLDING_HAZARD, derive `delay = hypothesis.deadline - activation_time` and
schedule the ACT guard for `activation_time + 0.825 * delay`. If that time is
not strictly after current time, fail loudly; do not act immediately as a
fallback.

Use deterministic public-feature flow targets and rates within bounds. The
scripted agent validates engine mechanics; no claim depends on its latent
features being intelligent.

- [ ] **Step 6: Run GREEN verification and commit**

```bash
uv run pytest -q tests/integration/test_scripted_agent.py \
  tests/regression/test_phase2_fixtures.py \
  tests/regression/test_import_boundaries.py \
  tests/integration/test_event_engine.py
uv run ruff check src/silent_cascade/eventflow/scripted.py \
  tests/integration/test_scripted_agent.py tests/regression/test_phase2_fixtures.py
uv run ruff format --check src/silent_cascade/eventflow/scripted.py \
  tests/integration/test_scripted_agent.py tests/regression/test_phase2_fixtures.py
git diff --check
git add src/silent_cascade/eventflow/scripted.py \
  tests/integration/test_scripted_agent.py tests/fixtures/phase2/scripted_cases.json \
  tests/regression/test_phase2_fixtures.py tests/regression/test_import_boundaries.py
git commit -m "feat: add public-only scripted EventFlow"
```

---

### Task 9: Post-Jump Invariants, Event Caps, Zeno Protection, and Crash Bundles

**Files:**

- Create: `src/silent_cascade/eventflow/invariants.py`
- Modify: `src/silent_cascade/eventflow/engine.py`
- Modify: `src/silent_cascade/eventflow/scheduling.py`
- Modify: `src/silent_cascade/logging/crash_bundle.py`
- Create: `tests/unit/test_runtime_invariants.py`
- Create: `tests/integration/test_dynamics_failures.py`
- Modify: `tests/property/test_eventflow_properties.py`

**Interfaces:**

- Produces: `validate_runtime_state`, `validate_post_jump`,
  `validate_session_boundary`, gap-clamp normalization, event-cap enforcement,
  and engine crash-bundle publication.

- [ ] **Step 1: Write a mutation-complete invariant test table**

Start from one valid post-jump state and independently mutate every required
invariant:

```text
nonfinite/decreasing time
nonfinite or out-of-bound continuous tensor
nonpositive/out-of-range latent or guard rate
accumulator not reset after jump
duplicate memory ID
changed PERCEIVED record/provenance
unknown/empty/duplicate inferred support
illegal source/destination mode
action before activation or more than one action
endogenous event after TERMINAL
next event earlier than current time
parent event/prediction hash mismatch
EventFlow NOOP
```

Every mutation must raise `DynamicsError` or `TimeOrderError` with a stable code
and bounded JSON context; never an `AssertionError`, NaN continuation, or hang.

- [ ] **Step 2: Write failing Zeno/cap/refractory tests**

Use malicious test agents to produce:

- geometrically shrinking alternating crossings;
- five consecutive sub-`1e-4` selected predictions;
- four clamps followed by an ordinary external event and then another clamp;
- a 65th internal event;
- same-kind events before and exactly at refractory release; and
- a negative/NaN predicted delta.

The first four selected sub-minimum events are clamped and logged; the fifth
raises. An earlier external event resets the streak. The 65th attempted event
raises before its jump, so no result can contain more than 64 executed internal
events.

- [ ] **Step 3: Write failing crash-bundle tests**

Configure a temporary crash root, trigger a dynamics failure, and assert one
mode-0600 atomic bundle containing public ID, config/source/checkpoint
references where known, typed error, traceback, and the last at most 20
sanitized causal summaries. Scan it for private label/path/window/terminal
payload sentinels. A bundle publication failure must surface as
`CrashBundleError`, not hide the original failure; chain the original error.

- [ ] **Step 4: Run tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_runtime_invariants.py \
  tests/integration/test_dynamics_failures.py \
  tests/property/test_eventflow_properties.py
```

Expected: invariant module and safety enforcement are missing.

- [ ] **Step 5: Enforce invariants after every causal jump**

The engine validates the pre-event boundary, advances, dispatches one jump,
validates the declared transition and complete post-jump state, then records.
Terminal dispatch validates state/action history before scoring and again after
terminalization. Never catch and continue from a dynamics failure.

- [ ] **Step 6: Implement selected-event clamp/cap semantics**

Do not mutate state merely because a prediction was inspected. Attach a
`was_gap_clamped` flag to the scheduled choice; increment the streak only if
that internal event is selected. Reset on a nonclamped selected internal or any
ordinary external event. Before an internal jump, reject when
`executed_internal_events == max_internal_events`.

- [ ] **Step 7: Run GREEN verification and commit**

```bash
uv run pytest -q tests/unit/test_runtime_invariants.py \
  tests/integration/test_dynamics_failures.py tests/integration/test_event_engine.py \
  tests/integration/test_scripted_agent.py tests/property/test_eventflow_properties.py
uv run ruff check src/silent_cascade/eventflow src/silent_cascade/logging/crash_bundle.py \
  tests/unit/test_runtime_invariants.py tests/integration/test_dynamics_failures.py
uv run ruff format --check src/silent_cascade/eventflow \
  src/silent_cascade/logging/crash_bundle.py tests/unit/test_runtime_invariants.py \
  tests/integration/test_dynamics_failures.py
git diff --check
git add src/silent_cascade/eventflow src/silent_cascade/logging/crash_bundle.py \
  tests/unit/test_runtime_invariants.py tests/integration/test_dynamics_failures.py \
  tests/property/test_eventflow_properties.py
git commit -m "feat: enforce fail-loud event dynamics"
```

---

### Task 10: Non-Mutating Trajectory Queries and Deterministic Trace Replay Core

**Files:**

- Modify: `src/silent_cascade/logging/trace.py`
- Create: `src/silent_cascade/eventflow/replay.py`
- Create: `tests/integration/test_replay.py`
- Modify: `tests/unit/test_trace.py`
- Modify: `tests/regression/test_phase2_fixtures.py`
- Create: `tests/fixtures/phase2/replay_cases.json`

**Interfaces:**

- Produces: `Trajectory`, `Trajectory.state_at`, `ReplayArtifact`,
  `ReplayComparison`, `write_replay_artifact`, `load_replay_artifact`, and
  `verify_replay`.
- Replay is CPU-only and reconstructs the scripted condition from an exact
  versioned config; later phases extend the agent/checkpoint registry.

- [ ] **Step 1: Write failing arbitrary-time trajectory tests**

Retain one post-event segment snapshot for every causal boundary. Query times:

- at episode start;
- strictly between external facts;
- immediately before and after internal events;
- exactly at a same-time tie (return the last post-jump state in tie order);
- in a long dormant interval; and
- at/after terminal (return terminal state only at the exact terminal time and
  reject a later time).

Assert the query does not mutate any runtime/trace tensor, increment counters,
insert a trace row, invoke an agent/controller, or alter the trace hash. A
query strictly between boundaries evaluates the latest preceding segment from
its immutable anchor.

- [ ] **Step 2: Write failing replay and tamper tests**

For each hand-authored fixture, run once, write a private validation replay
artifact, then rerun on CPU and compare:

```text
event kind/source/ID/parent
timestamp within CPU tolerance
prediction snapshot hash
selected memory ID and rank
hypothesis and support IDs
action and caused-by ID
final EpisodeScore
state hash at each causal boundary
full trace hash
```

Replay the same artifact twice and require identical comparison output/hash.
Independently tamper one timestamp, parent, selected record, support, action,
score, state hash, trace hash, config hash, episode byte, and agent version;
each must raise `ReplayError` naming the first mismatch without continuing.

- [ ] **Step 3: Run tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_trace.py \
  tests/integration/test_replay.py tests/regression/test_phase2_fixtures.py
```

Expected: trajectory and replay contracts are missing.

- [ ] **Step 4: Implement trajectory lookup**

Keep causal segment snapshots in stable event order. For equal timestamps,
select the last post-event snapshot according to recorded tie order. Clone all
returned tensors. Reject queries before initial time, in an unrepresented gap,
or after terminal; do not silently extrapolate across a jump.

- [ ] **Step 5: Implement a strict private replay artifact**

Use a Pydantic envelope with an internal payload SHA-256:

```python
class ReplayArtifact(StrictModel):
    schema_version: Literal["phase2-replay-v1"]
    access_class: Literal["validation_private"]
    episode: EpisodeArtifact
    config_canonical_json: str
    config_sha256: str
    condition: Literal[Condition.EVENT_FLOW]
    agent_implementation: Literal["scripted-event-flow-v1"]
    expected_result: EpisodeResultArtifact
    trace: CausalTraceArtifact
    payload_sha256: str
```

The payload hash excludes only itself and is recomputed before any rerun. Use
`atomic_create_bytes`; an existing divergent artifact is an error. The replay
factory is an exact closed mapping for the scripted v1 agent, not a plugin
loader or arbitrary import path.

- [ ] **Step 6: Run GREEN verification and commit**

```bash
uv run pytest -q tests/unit/test_trace.py tests/integration/test_replay.py \
  tests/regression/test_phase2_fixtures.py tests/integration/test_scripted_agent.py
uv run ruff check src/silent_cascade/logging/trace.py \
  src/silent_cascade/eventflow/replay.py tests/integration/test_replay.py
uv run ruff format --check src/silent_cascade/logging/trace.py \
  src/silent_cascade/eventflow/replay.py tests/integration/test_replay.py
git diff --check
git add src/silent_cascade/logging/trace.py \
  src/silent_cascade/eventflow/replay.py tests/unit/test_trace.py \
  tests/integration/test_replay.py tests/regression/test_phase2_fixtures.py \
  tests/fixtures/phase2/replay_cases.json
git commit -m "feat: add deterministic causal replay"
```

---

### Task 11: Safe Runtime Checkpoints and Exact Paused-World Resume

**Files:**

- Create: `src/silent_cascade/eventflow/checkpoint.py`
- Modify: `src/silent_cascade/eventflow/engine.py`
- Modify: `src/silent_cascade/eventflow/scripted.py`
- Modify: `src/silent_cascade/rng.py`
- Create: `tests/unit/test_checkpoint.py`
- Create: `tests/integration/test_checkpoint_resume.py`
- Modify: `tests/regression/test_phase2_fixtures.py`

**Interfaces:**

- Produces: `RuntimeCheckpointMetadata`, `RuntimeCheckpointArtifact`,
  `snapshot_runtime`, `publish_runtime_checkpoint`,
  `load_runtime_checkpoint`, `restore_runtime_session`, and explicit safe RNG
  JSON/tensor codecs.
- The container is one immutable `.safetensors` file; no pickle execution or
  arbitrary class import is permitted.

- [ ] **Step 1: Write failing safe-codec tests**

Round-trip:

- every continuous origin/current/target/rate/accumulator tensor;
- runtime mode, focus, active record, support ledger, hypothesis, actions,
  memory validity/consumption/refractory, safety counters, and causal counters;
- private external heap including terminal;
- cached internal prediction or cached dormancy;
- trace prefix and exact trace hash;
- Python, NumPy, Torch CPU, and capability-gated available MPS RNG state;
- engine/config/source/condition/agent/checkpoint schema IDs; and
- simulated cursor and paused-world semantics.

Inspect the file header and source to prove neither `pickle`, `torch.save`, nor
dynamic imports are used.

- [ ] **Step 2: Write failing corruption/refusal tests**

Reject changed schema version, config hash, source/agent identity, missing or
extra tensor, dtype/shape mismatch, per-tensor hash mismatch, truncated file,
nonfinite restored state, corrupt queue/trace, noncanonical metadata, and an
existing destination. A failed load/restore must leave caller RNGs and agent
counters unchanged.

- [ ] **Step 3: Write failing continuation tests**

For positive, safe, and disconnected fixtures, checkpoint:

- during a long flow segment;
- immediately after FACT;
- immediately after ACTIVATE;
- after RECALL;
- after COMPOSE link;
- after COMPOSE terminal; and
- after ACT.

Compare uninterrupted versus restored CPU continuation for remaining event
kinds, IDs, parents, timestamps, memory choices, hypotheses, supports, action,
score, causal counters, and final trace hash. The checkpoint-only materialized
flow counter may differ only in its separately named diagnostic field.

Pause the host process using a patched wall clock or an actual very short test
delay and prove simulated state/outcome is unchanged. Never sleep in the
implementation.

- [ ] **Step 4: Run tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_checkpoint.py \
  tests/integration/test_checkpoint_resume.py \
  tests/regression/test_phase2_fixtures.py
```

Expected: checkpoint codec and restore entry points are missing.

- [ ] **Step 5: Implement canonical safe RNG codecs**

Encode Python random state recursively as versioned JSON tuples, NumPy legacy
state as algorithm/key-array/position/Gaussian fields, and Torch RNG states as
named uint8 tensors. Validate into isolated RNG objects/generators before
transactionally replacing globals. Preserve an archived MPS RNG tensor even
when loading for CPU replay; only require an available MPS backend when the
caller explicitly requests MPS restoration.

- [ ] **Step 6: Implement one-file safetensors checkpoints**

Use `safetensors.torch.save(tensors, metadata=...)` to obtain bytes, then
`atomic_create_bytes`. Move cloned tensors to contiguous CPU storage before
serialization and record original device. Canonical metadata includes a hash
of every tensor framed by name/dtype/shape/bytes and a hash of the metadata
payload excluding its self-hash. Loader checks all hashes before constructing
any runtime object.

Do not embed absolute workspace paths. Store source revision/config hashes and
opaque checkpoint IDs.

- [ ] **Step 7: Preserve causal anchors and prediction cache**

Checkpointing may update only the session cursor/current materialized snapshot
and the separate checkpoint-flow diagnostic counter. It must retain segment
origin and cached prediction/dormancy. Restore must not invoke
`next_internal_event` until the cached boundary is consumed or invalidated by a
causal event.

When the engine is configured with a crash root, snapshot the pre-failure
session into a sibling immutable `.safetensors` file before publishing the JSON
crash bundle. Store only its safe relative filename in
`CrashContext.checkpoint_ref`. The two artifacts share a random bundle ID. If
checkpoint publication fails, surface `CrashBundleError` chained from the
original dynamics failure; never publish a JSON bundle that claims a missing
checkpoint is replayable.

- [ ] **Step 8: Run GREEN verification and commit**

```bash
uv run pytest -q tests/unit/test_checkpoint.py \
  tests/integration/test_checkpoint_resume.py tests/integration/test_replay.py \
  tests/regression/test_phase2_fixtures.py tests/unit/test_rng.py
uv run ruff check src/silent_cascade/eventflow/checkpoint.py \
  src/silent_cascade/eventflow/engine.py src/silent_cascade/rng.py \
  tests/unit/test_checkpoint.py tests/integration/test_checkpoint_resume.py
uv run ruff format --check src/silent_cascade/eventflow/checkpoint.py \
  src/silent_cascade/eventflow/engine.py src/silent_cascade/rng.py \
  tests/unit/test_checkpoint.py tests/integration/test_checkpoint_resume.py
git diff --check
git add src/silent_cascade/eventflow/checkpoint.py \
  src/silent_cascade/eventflow/engine.py src/silent_cascade/eventflow/scripted.py \
  src/silent_cascade/rng.py tests/unit/test_checkpoint.py \
  tests/integration/test_checkpoint_resume.py tests/regression/test_phase2_fixtures.py
git commit -m "feat: add exact runtime checkpoints"
```

---

### Task 12: Replay CLI, Production Doctor Math, Platform Checks, and Local Workflow

**Files:**

- Modify: `src/silent_cascade/cli.py`
- Modify: `src/silent_cascade/doctor.py`
- Modify: `README.md`
- Modify: `Makefile`
- Create: `tests/integration/test_cli_replay.py`
- Modify: `tests/integration/test_cli_doctor.py`
- Modify: `tests/unit/test_doctor.py`
- Modify: `tests/integration/test_phase0_repository.py`
- Modify: `tests/integration/test_sdist_contents.py`
- Modify: `tests/regression/test_import_boundaries.py`
- Modify: `tests/property/test_eventflow_properties.py`

**Interfaces:**

- Adds root command: `silent-cascade replay ARTIFACT [--json]`.
- Routes doctor numeric smoke through `eventflow.flow` and
  `eventflow.guards`, deleting the duplicated provisional formulas.
- Keeps `make verify` local-only and all existing Phase 1 commands stable.

- [ ] **Step 1: Write failing CLI tests**

Use `CliRunner` and the installed console script to replay a fixture artifact.
JSON success output must be strict and contain schema, artifact/trace hashes,
event count, timed result, device=`cpu`, and `foundation_model_calls=0`.
Human output is concise. Corruption, mismatch, missing file, symlink/FIFO,
oversize artifact, or non-validation access class returns nonzero with the
stable `ReplayError` payload and no traceback by default.

Also pass a crash-bundle fixture that names a sibling pre-failure runtime
checkpoint. Replay must restore the scripted session on CPU, execute one step,
and reproduce the recorded typed failure. A missing/escaping checkpoint
reference or a different error code/context fails closed.

Update the root-command expectation from `{doctor}` to `{doctor, replay}`;
nested Phase 1 command groups remain intact.

- [ ] **Step 2: Write failing doctor/platform tests**

Patch production flow/guard functions and prove `_numeric_smoke` calls them.
Delete acceptance of a locally duplicated formula. On MPS-capable hosts, test
float32 flow/guard behavior against CPU within documented absolute tolerance,
assert all tensors stay on MPS, synchronize before timing, and assert the
process has not enabled `PYTORCH_ENABLE_MPS_FALLBACK`. On non-MPS hosts, skip
only MPS comparisons; CPU coverage is mandatory.

Add static tests rejecting float64 tensor literals/conversions in production
EventFlow modules and rejecting fixed-grid constructs (`arange`, interval-loop
configuration, sleep/poll calls) in `eventflow.engine`, `guards`, and
`scripted`.

- [ ] **Step 3: Write failing repository/workflow tests**

Assert:

- no `.github/workflows` content;
- `make verify` still expands only local commands and includes all routine
  Phase 2 unit/property/integration/regression tests through `pytest`;
- the explicit 10,000 gate collector is not run by routine `make verify`;
- source and wheel imports expose replay and eventflow without MLX/Qwen;
- sdist/wheel contain every new package module and config documentation needed
  by installed commands; and
- README calls Phase 2 non-neural engineering evidence and makes no learned or
  consciousness claim.

- [ ] **Step 4: Run tests and confirm RED**

```bash
uv run pytest -q tests/integration/test_cli_replay.py \
  tests/integration/test_cli_doctor.py tests/unit/test_doctor.py \
  tests/integration/test_phase0_repository.py \
  tests/integration/test_sdist_contents.py \
  tests/regression/test_import_boundaries.py \
  tests/property/test_eventflow_properties.py
```

Expected: replay command is absent; doctor still owns provisional math; Phase 2
workflow/docs assertions fail.

- [ ] **Step 5: Implement the thin replay adapter**

The CLI resolves a bounded regular file, calls `verify_replay`, renders the
strict result, and maps only `SilentCascadeError` to stable output. It does not
accept an arbitrary class/module name, download anything, or expose private
episode content.

Closed schema dispatch accepts `phase2-replay-v1` and the existing crash-bundle
schema only. For a crash bundle, resolve `checkpoint_ref` as a regular sibling
filename with no absolute path or `..`, load the scripted checkpoint on CPU,
execute exactly one step, and require the same typed error code and canonical
context. It must not dynamically import an agent or continue past the
reproduced failure.

- [ ] **Step 6: Route doctor through production math**

Construct fixed float32 state/guard inputs on the chosen device and call the
actual flow/guard APIs. Compare outputs after copying scalars to CPU. Keep all
existing host/RNG/path/Qwen-opt-in diagnostics and exact zero foundation calls.

- [ ] **Step 7: Update local-only workflow and docs**

Do not add a hosted workflow or a Phase 2 gate Make target. Routine
`make verify` remains lint + partitioned tests + doctor + package build. Add
Phase 2 replay tests to `smoke`; update the exact Make-target regression only
for targets that actually exist. README documents the verified non-neural
runtime and `silent-cascade replay`, while still stating that no learned
benchmark result exists.

- [ ] **Step 8: Run complete local verification and commit**

```bash
uv sync --locked --group dev
make verify
uv run silent-cascade doctor
uv build
git diff --check
git status --short
git add src/silent_cascade/cli.py src/silent_cascade/doctor.py README.md Makefile \
  tests/integration/test_cli_replay.py tests/integration/test_cli_doctor.py \
  tests/unit/test_doctor.py tests/integration/test_phase0_repository.py \
  tests/integration/test_sdist_contents.py tests/regression/test_import_boundaries.py \
  tests/property/test_eventflow_properties.py
git commit -m "feat: expose local EventFlow replay"
```

Expected: the full local gate passes; no MPS fallback, optional model import,
network access, or hosted automation appears.

---

### Task 13: Source-Bound Phase 2 Gate Collector and Independent Verifier

**Files:**

- Modify: `src/silent_cascade/provenance.py`
- Create: `src/silent_cascade/eventflow/evidence.py`
- Modify: `src/silent_cascade/cli.py`
- Create: `scripts/check_phase2_engine.py`
- Create: `scripts/verify_phase2_gate_artifact.py`
- Create: `tests/integration/test_phase2_gate.py`
- Modify: `tests/integration/test_cli_replay.py`
- Modify: `tests/unit/test_provenance.py`
- Modify: `tests/integration/test_phase0_repository.py`
- Modify: `tests/integration/test_sdist_contents.py`

**Interfaces:**

- Produces: `PHASE2_ENGINE_SOURCE_PATHS`, `Phase2EvidenceProvenance`,
  `collect_phase2_evidence_provenance`, strict
  `Phase2EngineGateReport`/`phase2-engine-gate-v1`, a small test profile, the
  exact 10,000 profile, embedded-sample CLI replay, and a bounded standalone
  verifier.
- Reads the already committed Phase 1 validation manifest and regenerates each
  entry through its authenticated recipe. It never creates test/frozen data.

- [ ] **Step 1: Freeze the exact evidence schema in failing tests**

The report must carry primitive counts plus derived summaries:

```python
class Phase2EngineGateReport(StrictModel):
    schema_version: Literal["phase2-engine-gate-v1"]
    access_class: Literal["validation_private"]
    provenance: Phase2EvidenceProvenance
    validation_manifest_payload_sha256: str
    validation_manifest_file_sha256: str
    config_sha256: str
    generator_version: Literal["ofd-v1"]
    condition: Literal["scripted_event_flow"]
    agent_implementation: Literal["scripted-event-flow-v1"]
    requested_episode_count: Literal[10_000]
    completed_episode_count: int
    positive_count: int
    safe_negative_count: int
    disconnected_negative_count: int
    timed_success_count: int
    false_action_count: int
    dynamics_failure_count: int
    replay_failure_count: int
    internal_event_count: int
    maximum_episode_internal_events: int
    minimum_internal_gap: float | None
    gap_clamp_count: int
    event_cap_failure_count: int
    time_reversal_count: int
    provenance_failure_count: int
    trace_chain_sha256: str
    selected_replay_samples: tuple[ReplayArtifact, ...]
    selected_replay_trace_hashes: tuple[str, ...]
    foundation_model_calls: Literal[0]
    passed: bool
```

Add exact variant counts `5000/2500/2500`. Derive `passed` from
`completed=timed_success=10_000`, every failure/clamp/cap/reversal/provenance/
replay count zero, max internal events `<=64`, a positive finite minimum gap
`>=1e-4`, exactly three replay samples (one per variant), all trace hashes
unique, clean source provenance, and zero foundation calls.

The small test profile must use a temporary debug manifest and a separate
schema/profile flag; it may never emit a production-passing 10,000 report.
The three embedded replay samples are environment-private validation evidence;
their separately listed hashes must equal their nested trace hashes in
positive, safe-negative, disconnected-negative order.

Define the small-path output as a distinct
`phase2-engine-gate-debug-v1` model with `access_class="debug"` and a positive
multiple-of-four requested count. It exercises the same streaming core but
cannot validate or deserialize as `Phase2EngineGateReport`.

- [ ] **Step 2: Write failing collector/verifier tests**

Cover:

- successful 12- or 16-episode debug collection;
- a failing episode, missing variant, wrong denominator, false action,
  dynamics error, trace replay mismatch, nonzero clamp, and nonzero foundation
  call all fail the report;
- bounded regular-file reads and refusal of FIFO/symlink/oversize data;
- config/manifest/source/plan/hash mismatches;
- coordinated edits to totals versus primitive per-variant counts;
- report `passed=true` substitution;
- dirty source or uncommitted plan;
- output no-clobber and atomic publication; and
- independent verifier does not call collector helpers or rerun episodes by
  default.

- [ ] **Step 3: Freeze the two-state delivery-index regression**

Before any Phase 2 artifact exists, update the repository test to allow exactly
two states:

1. the exact in-progress cell from Task 1 and no artifact; or
2. one regular artifact whose byte SHA-256 and common source commit appear in
   the exact completion cell defined in Task 14.

A partial artifact, symlink, stale hash, wrong schema, dirty provenance, or
completed cell without the artifact fails. The regression must preserve and
independently validate the already-completed Phase 1 row.

- [ ] **Step 4: Run tests and confirm RED**

```bash
uv run pytest -q tests/integration/test_phase2_gate.py \
  tests/unit/test_provenance.py tests/integration/test_phase0_repository.py \
  tests/integration/test_sdist_contents.py
```

Expected: Phase 2 provenance, collector, verifier, and delivery-state logic do
not exist.

- [ ] **Step 5: Implement exact Phase 2 source provenance**

Define a literal sorted tuple containing every source/config/script that can
change generation, runtime execution, scoring, tracing, checkpoint/replay, or
gate arithmetic. Include executed ancestor `__init__.py` files. Keep existing
Phase 1 source tuples and evidence models byte-compatible; add a sibling Phase
2 provenance model rather than repurposing Phase 1 fields.

Authenticate from regular Git blobs at the named full source commit:

```text
phase2_plan_base_revision
source_commit and clean/dirty state
canonical Phase 2 source-path tuple and historical source-tree SHA-256
canonical spec and approved plan blob SHA-256
config canonical SHA-256
validation manifest payload/file SHA-256
Python/Torch/platform metadata
```

- [ ] **Step 6: Implement streaming collection**

For each manifest entry in canonical order:

1. regenerate and hash-check the bundle;
2. run a new CPU `ScriptedEventFlowAgent` autonomously through `EventEngine`;
3. score only inside the engine at the private terminal;
4. update exact variant/result/dynamics counters;
5. hash the canonical per-episode summary into an ordered framed trace chain;
6. retain one replay artifact candidate per variant; and
7. immediately rerun those three selected candidates twice through CPU replay
   before publication.

Do not retain 10,000 full traces in memory. Keep compact event summaries only
for the current episode and three replay samples. Any exception exits nonzero,
writes a crash bundle, and publishes no passing artifact.

- [ ] **Step 7: Implement the standalone verifier independently**

The verifier parses bounded bytes, recomputes schema arithmetic/pass state,
validates all hashes and exact counts, resolves full Git revisions/ancestry,
recomputes historical plan/spec/source hashes from regular blobs, verifies the
Phase 1 validation manifest envelope/hash, and checks the trace-chain/replay
sample witnesses serialized in the report. It may trust the committed
per-episode trace-chain digest without rerunning all 10,000 episodes; document
that coherent replacement of artifact and all Git trust anchors is out of
scope.

- [ ] **Step 8: Add strict embedded-sample CLI replay**

Extend the replay loader with a closed schema dispatch for
`phase2-engine-gate-v1`. Require `--sample-index` in `0..2` for that artifact,
select only its nested `ReplayArtifact`, and run the same CPU verifier. A
standalone `phase2-replay-v1` artifact rejects `--sample-index`; a gate artifact
without the selector, with a debug schema, or with inconsistent nested/listed
hashes fails. Add these cases to `test_cli_replay.py` and
`test_phase2_gate.py`.

- [ ] **Step 9: Run focused and full verification, review, then commit source**

```bash
uv run pytest -q tests/integration/test_phase2_gate.py \
  tests/unit/test_provenance.py tests/integration/test_phase0_repository.py \
  tests/integration/test_sdist_contents.py
uv run python scripts/check_phase2_engine.py --help
uv run python scripts/verify_phase2_gate_artifact.py --help
make verify
git diff --check
git status --short
```

Run separate specification-compliance and adversarial code-quality reviews.
Resolve every finding and rerun the commands. Then:

```bash
git add src/silent_cascade/provenance.py \
  src/silent_cascade/eventflow/evidence.py src/silent_cascade/cli.py \
  scripts/check_phase2_engine.py scripts/verify_phase2_gate_artifact.py \
  tests/integration/test_phase2_gate.py tests/unit/test_provenance.py \
  tests/integration/test_cli_replay.py \
  tests/integration/test_phase0_repository.py \
  tests/integration/test_sdist_contents.py
git commit -m "test: define Phase 2 engine gate"
```

This exact reviewed commit, or a reviewed fix descendant made before evidence
collection, becomes the candidate `phase2_evidence_source_commit`.

---

### Task 14: Collect, Verify, and Commit the 10,000-Episode Phase 2 Gate

**Files:**

- Create: `manifests/validation/v1/phase2-engine-gate.json`
- Modify: `docs/PLAN.md`

**This task changes no implementation, test, config, fixture, verifier, or
collector source.** Any need to change one returns to Task 13, creates a new
reviewed source commit, and invalidates every partial Phase 2 artifact.

- [ ] **Step 1: Capture a clean immutable collector source**

```bash
make verify
git status --short
phase2_evidence_source_commit="$(git rev-parse HEAD)"
phase2_plan_base_revision="$(git log -1 --format=%H -- \
  docs/superpowers/plans/2026-09-09-phase-2-flow-event-engine.md)"
test -z "$(git status --porcelain)"
```

Expected: local verification passes and status is empty. Record both full
40-hex revisions. Verify the plan revision is an ancestor of the source commit.

- [ ] **Step 2: Collect all 10,000 validation executions locally on CPU**

```bash
uv run python scripts/check_phase2_engine.py \
  --config configs/base.yaml \
  --data-config configs/data/primary.yaml \
  --event-flow-config configs/model/event_flow.yaml \
  --manifest manifests/validation/v1/ofd-primary-10000.json \
  --expected-source-commit "${phase2_evidence_source_commit}" \
  --expected-plan-base-revision "${phase2_plan_base_revision}" \
  --output manifests/validation/v1/phase2-engine-gate.json
```

Expected exact gate:

```text
completed episodes             10000
positive/safe/disconnected     5000 / 2500 / 2500
timed successes                10000
false actions                  0
dynamics/replay failures       0 / 0
gap clamps/cap/time reversal   0 / 0 / 0
maximum internal events        <= 64
minimum internal gap           >= 1e-4
selected replay samples        3 distinct hashes, twice verified on CPU
foundation-model calls         0
passed                         true
```

This command must make no network request and must not open frozen tests.

- [ ] **Step 3: Verify the artifact boundary**

```bash
uv run python scripts/verify_phase2_gate_artifact.py \
  --artifact manifests/validation/v1/phase2-engine-gate.json \
  --manifest manifests/validation/v1/ofd-primary-10000.json \
  --expected-source-commit "${phase2_evidence_source_commit}" \
  --expected-plan-base-revision "${phase2_plan_base_revision}"
uv run silent-cascade replay \
  manifests/validation/v1/phase2-engine-gate.json --sample-index 0 --json
uv run silent-cascade replay \
  manifests/validation/v1/phase2-engine-gate.json --sample-index 1 --json
uv run silent-cascade replay \
  manifests/validation/v1/phase2-engine-gate.json --sample-index 2 --json
git diff --check
```

The CLI must explicitly recognize the gate report's embedded private replay
samples; ordinary replay artifacts need no sample selector. Each replay runs on
CPU and matches its recorded trace/action/score/hash.

- [ ] **Step 4: Update only the Phase 2 delivery cell**

Calculate the artifact file SHA-256 and replace the in-progress Phase 2 cell
with exactly:

```text
Complete at engine source `<phase2_evidence_source_commit>` with gate artifact `<artifact_file_sha256>`: the public-facts-only scripted EventFlow condition achieved 10,000/10,000 timed successes; numeric, long-silence, Zeno, checkpoint/resume, CPU replay, local `make verify`, and the independent artifact verifier passed with zero foundation-model calls. This is non-neural runtime engineering evidence, not learned-model or benchmark evidence.
```

Do not alter the Phase 1 cell and do not add a scientific claim.

- [ ] **Step 5: Run the completed-state gate and commit evidence**

```bash
uv run python scripts/verify_phase2_gate_artifact.py \
  --artifact manifests/validation/v1/phase2-engine-gate.json \
  --manifest manifests/validation/v1/ofd-primary-10000.json \
  --expected-source-commit "${phase2_evidence_source_commit}" \
  --expected-plan-base-revision "${phase2_plan_base_revision}"
make verify
uv run silent-cascade doctor
uv build
git diff --check
git add manifests/validation/v1/phase2-engine-gate.json docs/PLAN.md
git commit -m "test: freeze Phase 2 engine evidence"
```

- [ ] **Step 6: Verify the committed Phase 2 boundary**

```bash
make verify
uv run python scripts/verify_phase2_gate_artifact.py \
  --artifact manifests/validation/v1/phase2-engine-gate.json \
  --manifest manifests/validation/v1/ofd-primary-10000.json
git status --short
git log --reverse --oneline "${phase2_plan_base_revision}..HEAD"
```

Expected: all checks pass, status is empty, the artifact still names the
reviewed Task 13 source commit rather than the later evidence/docs commit, and
history contains the meaningful Phase 2 task commits.

Run fresh independent specification-compliance, scientific-boundary, and
adversarial code-quality audits over the committed result. If an audit finds a
source defect, remove the invalid Phase 2 gate artifact in the corrective
source commit, return the delivery cell to in-progress, fix/test/review, and
rerun all of Task 14. Never patch source after evidence and leave the earlier
artifact marked complete.

## Phase 2 Completion Boundary

Phase 2 is complete only when all of the following are simultaneously true:

1. all exact state dimensions, float32 device rules, host-float64 time rules,
   flow/guard bounds, event/memory limits, and tie tolerances are strict config;
2. the flow is stable for zero, tiny, ordinary, huge, and split time, remains
   bounded/finite, and passes CPU plus available-MPS comparison;
3. dormant guards do not evaluate invalid math, active crossings reach the
   threshold within tolerance, and the standard EventFlow race accepts no
   external horizon;
4. state persists across public observations and analytic silence without a
   fixed tick, polling fallback, timer prompt, or hidden semantic observation;
5. the engine-private queue owns OUTCOME/END and scorer truth, while every spy
   callback receives only allowed public-derived objects;
6. all declared tie, preemption, parent, prediction-cache, ID, refractory,
   clamp, cap, and terminal semantics are deterministic and trace-visible;
7. every post-jump invariant fails loudly with a bounded actionable crash
   bundle; Zeno-like sequences cannot hang or silently continue;
8. typed memory/provenance/support rules work at primary capacity without
   learned retrieval or silent eviction;
9. arbitrary-time state queries are analytic, non-mutating, and noncausal;
10. checkpoints at mid-flow and after every event kind restore identical CPU
    causal continuation under paused-world semantics;
11. replay detects every declared mismatch and reproduces publication fixture
    decisions/traces twice with stable hashes;
12. the public-facts-only scripted condition imports no environment oracle or
    private bundle/truth type and autonomously solves all 10,000 committed
    validation entries with zero false actions/dynamics failures;
13. local `make verify`, doctor, package builds, MPS capability gate, import
    boundaries, and the standalone artifact verifier pass;
14. foundation-model calls are exactly zero and no optional model package is
    loaded; and
15. the committed delivery index binds one regular Phase 2 gate artifact to
    the reviewed source commit/file hash and labels it non-neural engineering
    evidence rather than learned benchmark evidence.

Do not begin Phase 3 from this plan. After this gate, write and explicitly
approve a separate Phase 3 neural-components plan against the canonical spec
and the actual completed Phase 2 APIs.

## Explicitly Deferred Scope

- **Phase 3:** 96-dimensional tensor record embeddings, entity table, learned
  retrieval/query/scoring, neural flow/guard controller, shared jump trunk,
  compose/action heads, losses, teacher-forced batches, compute hooks, and
  training checkpoints.
- **Phase 4:** learned autonomous EventFlow execution, primary curriculum,
  exposure-bias diagnostics, one-seed pilot, and 10,000 learned validation
  dynamics gate.
- **Phase 5:** one-shot, activation-ponder, compressed-time, flow-only,
  fixed/periodic/random-time/frozen controls, compute matching, and causal
  interventions.
- **Phase 6:** final frozen manifests, five seeds, confirmatory statistics,
  aggregate/report generation, publication traces, and scientific claims.
- **Phase 7:** Qwen, natural-language parsing/narration, demos, and public
  showcase assets.
- **Later stress ownership:** deterministic eviction is Phase 3; branching,
  cycles, contradiction policy learning, and intervention semantics are
  evaluated only after their owning learned/runtime mechanisms exist.

## Final Self-Review Checklist for the Implementer

- [ ] Every behavior change in Tasks 1–13 began with the specified failing
  test and retained a visible RED/GREEN record in the task review.
- [ ] No agent callback receives an external horizon, queue, truth, terminal,
  score, trace recorder, generator seed, manifest coordinate, or window.
- [ ] OUTCOME and END are never passed to `on_external` and are sanitized from
  online/crash summaries.
- [ ] `eventflow.scripted` imports no environment module and uses no oracle
  symbol, trace, hidden path, or answer.
- [ ] EventFlow emits only RECALL/COMPOSE/ACT and never NOOP/THINK.
- [ ] OBSERVING masks every endogenous guard; QUIESCENT and TERMINAL are truly
  dormant with no periodic fallback.
- [ ] Absolute time/crossing comparisons are host float64; production tensors
  are float32 and MPS never receives float64.
- [ ] Flow uses `-torch.expm1`, evaluates from immutable segment anchors, and
  leaves `dt=0` as identity without a controller call.
- [ ] Dormant guards never evaluate their logarithm; active runtime crossings
  use `math.log1p` on detached host scalars.
- [ ] Same-kind refractory delays eligibility rather than deleting a valid
  later crossing.
- [ ] The engine caches exactly one crossing/dormancy result per causal
  boundary and invalidates it after every causal event or external preemption,
  but not after a checkpoint cursor move.
- [ ] Tie records prove terminal > external > ACT > COMPOSE > RECALL > NOOP for
  equal and near-equal times.
- [ ] A selected fifth consecutive sub-minimum prediction and an attempted 65th
  internal event raise `DynamicsError` with crash evidence.
- [ ] Common compose jumps apply explicit predictions and never auto-correct
  them from memory or hidden truth.
- [ ] Primary memory never exceeds 64 or silently evicts; Phase 3 eviction is
  still absent.
- [ ] Inferred hypotheses have nonempty valid immutable supports; PERCEIVED
  provenance never changes.
- [ ] Arbitrary-time queries clone outputs and create no event, callback,
  counter increment, or trace mutation.
- [ ] Checkpoints use one hashed safetensors container, no pickle/`torch.save`,
  and preserve segment anchors, cached prediction, queue, trace, counters, and
  safe RNG state.
- [ ] CPU continuation after every declared checkpoint position has identical
  causal trace/action/score/hash to uninterrupted execution.
- [ ] Replaying the same artifact twice produces the same trace hash, while
  every declared corruption causes a typed mismatch.
- [ ] Doctor invokes production flow/guard code and MPS comparisons synchronize
  explicitly without fallback.
- [ ] `make verify` is local-only, no hosted workflow exists, and the 10,000
  collector remains an explicit acceptance command rather than a routine test.
- [ ] Phase 1 artifacts/hashes/row remain unchanged and `manifests/frozen/`
  remains absent.
- [ ] The gate artifact authenticates exact config, manifest, spec, approved
  plan, source paths, source commit, counts, trace chain, and three replay
  samples.
- [ ] All 10,000 scripted runs succeed; every dynamics/replay/clamp/cap/time/
  provenance failure count is zero; foundation-model calls are exactly zero.
- [ ] README, delivery index, and final response call the result non-neural
  runtime engineering evidence only.
- [ ] Final worktree is clean and every meaningful Phase 2 task/evidence commit
  is present on the current branch.
