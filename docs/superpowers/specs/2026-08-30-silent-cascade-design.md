# Silent Cascade

## Standalone research and end-to-end implementation plan for persistent event-flow AI

**Status:** Approved design specification

**Plan version:** 1.0

**Research freeze date:** 2026-08-30

**Target:** Apple MacBook Pro with M3 Max

**Repository:** Public personal research project

**Primary language:** Python 3.12

**Primary ML stack:** PyTorch on CPU and Apple MPS

**Optional language shell:** Qwen3.5-4B through MLX

**License:** Apache-2.0

This file is the canonical design specification for the repository. The root
`AGENTS.md` requires all implementation work to follow the Superpowers workflow.
An approved implementation plan produced with `superpowers:writing-plans` is
required before source implementation begins.

### Superpowers delivery strategy

This specification defines one integrated research program, but it is too large
for a single safe implementation pass. Deliver it through separate, ordered
Superpowers plans for Phases 0–7. Write and approve the Phase 0 plan after this
specification is approved; write each later phase plan only after the preceding
phase gate passes. Split Phase 6 into a freeze-preparation plan and a final-runs
plan so the irreversible test freeze remains an explicit checkpoint. Phase 7 is
optional and always receives its own plan. `docs/PLAN.md` is a navigation index
to this specification and the approved phase plans, never a duplicate source of
requirements.

---

## 0. Contract for the coding agent

This is the complete specification for an empty repository. Implement one rigorous experiment, its controls, its causal analysis, its report generator, and a small optional showcase. Do not turn it into a general agent framework, hosted service, database product, or enterprise platform.

Non-negotiable rules:

1. The project tests a persistent, continuous-time internal state with **learned endogenous event timing** during intervals with no semantic input.
2. The experimental condition must not be a fixed-rate cognitive loop. It advances analytically to the next external event or learned internal guard crossing.
3. Strong controls are mandatory, especially a one-shot all-memory scheduler and an adaptive ponder-at-activation system that may use recurrent depth immediately after activation and schedule a future action.
4. The primary experiment must run with no foundation model, no network access, and no hidden natural-language reasoning.
5. Keep the system bounded: at most 5 million trainable parameters per primary model, 64 memory records, and 64 learned EventFlow events per episode. Fixed-tick controls use their declared opportunity grid under a separate 25,000-opportunity safety ceiling so long-delay controls are not silently truncated.
6. Every published number must be regenerated from raw per-episode artifacts tied to a code revision, configuration hash, seed, checkpoint hash, and generator version.
7. Invalid dynamics must fail loudly. NaNs, time reversal, Zeno-like bursts, illegal mode transitions, provenance corruption, and replay mismatches are errors, not normal outcomes.
8. Do not weaken baselines, metrics, or acceptance gates after viewing frozen-test results.
9. The final repository must contain no unfinished-work markers, dummy metrics, fake plots, or stubbed documented commands.
10. A scientifically negative result is acceptable. An unreproducible or weakly controlled positive result is not.

---

## 1. Executive decision

Build **Silent Cascade**, a small hybrid neural system whose state has a defined trajectory at every real-valued simulated time. Its state changes by two mechanisms:

- a stable, closed-form continuous flow between events; and
- typed discrete jumps triggered by externally observed events or learned endogenous guard crossings.

The single decisive benchmark is **Observation-Free Deadline (OFD)**. Before activation, the environment presents shuffled symbolic causal facts. At one final external event it activates a start node. The environment then becomes semantically silent. During that silent interval, the system must autonomously recall memories, traverse a variable-length causal chain, form a typed hazard hypothesis, and emit the correct short-lived protective action inside a narrow real-valued window. No prompt asks it to remember, reason, or act.

The central question is:

> Does a persistent event-flow system obtain better timed-action accuracy, out-of-distribution depth generalization, or compute efficiency than equally informed one-shot, activation-time pondering, fixed-tick, periodic, random-time, flow-only, and frozen-state systems—and do causal interventions show that its intermediate silent-time events are necessary for the behavior?

The project does **not** claim consciousness, sentience, biological fidelity, or a universal route to AGI. Its strongest legitimate result would be evidence for a useful computational layer between reactive model calls and agents with an ongoing temporally structured internal trajectory.

### One-sentence specification

> Implement a bounded, checkpointable, piecewise-continuous latent workspace with learned endogenous event guards and typed memory/reasoning jumps, then test it on a no-prompt deadline task where a variable-length internal cascade during observation-free time must cause a correct action under strong compute-matched baselines and causal interventions.

### Why this design is sharper than a recurrent idle loop

A conventional RNN running at 5 Hz can be dismissed as ordinary recurrence with extra steps. A continuous ODE that only decays during silence may be skipped directly to the next observation. A scheduled LLM reflection remains an externally clocked sequence of calls. A memory lookup at the next question can defer all useful work until demand appears.

Silent Cascade closes those loopholes as far as a practical digital experiment can:

- there is no fixed cognitive tick in the experimental condition;
- state at arbitrary time is produced by an analytic flow, not repeated polling;
- internal events occur at learned, state-dependent threshold crossings;
- each internal jump changes what later jump becomes possible;
- the environment requires an action before another semantic observation arrives;
- the strongest one-shot control may inspect all memory and schedule an action at activation;
- an adaptive ponder-at-activation control may reuse recurrent retrieval/composition depth immediately at activation, then schedule a future action;
- fixed-tick and periodic controls may also act during silence;
- every learned condition gets the same bounded typed memory;
- randomized symbols eliminate pretrained semantic knowledge;
- Qwen is absent from the scientific path;
- suppressing or retiming the internal cascade must destroy the claimed gain.

No finite deterministic computation is metaphysically irreducible. A different program can simulate its trajectory. The empirical claim is narrower: this mechanism may be a useful inductive bias and compute-allocation strategy.

---

## 2. What “continuous” means here

A system is functionally continuous only when all four properties hold:

1. **State continuity:** one central state persists causally across external interactions and is not reconstructed wholesale from a transcript.
2. **Temporal continuity:** elapsed time changes that state according to an explicit law; one second and one hundred seconds of silence are not equivalent.
3. **Endogenous continuity:** internal state can schedule computation without new semantic input.
4. **Behavioral continuity:** events separated in time can be combined during silence and can cause an action before a later prompt or observation requests the combination.

Merely keeping a process alive is insufficient. Generating hidden text is unnecessary. No digital computer executes an actual mathematical continuum; the meaningful distinction is between a fixed polling loop and a hybrid state process whose trajectory and next events are defined by its current state.

### Fundamental deferred-computation caveat

If nothing can be affected before the next observation, a deterministic system can often defer its silent-time computation until that observation. Observation-Free Deadline (OFD) therefore makes the internal result externally consequential through a deadline. Even this does not prove that the exact temporal computation could not have been precomputed at activation. Both a feed-forward one-shot scheduler and an adaptive recurrent ponder-at-activation system are therefore mandatory strong controls.

### Primary hypotheses

Let `Y(E,C)` be timed episode success for episode `E` and condition `C`, and let `B(E,C)` be measured compute.

**H1 — practical advantage:** On at least one predeclared out-of-distribution regime and within overlapping compute, EventFlow improves timed success over strong alternatives, or matches their success with substantially fewer neural transitions.

**H2 — causal silent cascade:** Suppressing, deleting, retiming, or counterfactually replacing the relevant post-activation internal events or representations substantially changes later action while observations, memory capacity, and weights remain fixed.

**H3 — temporal robustness:** Rescaling episode time while preserving causal ratios does not materially degrade EventFlow.

**H4 — foundation-model independence:** All primary findings reproduce with exactly zero Qwen calls.

### Null explanations that must be tested

- one pass at activation is sufficient;
- an adaptive recurrent computation burst at activation is sufficient, so distributed silent-time timing adds nothing;
- fixed recurrent ticks are equal or better;
- periodic opportunities are equal or better;
- persistence or memory alone explains the result;
- EventFlow merely receives more compute;
- a hand-coded graph traversal hidden in typed scaffolding solves the task;
- entity IDs, record order, counts, or timing leak the answer;
- continuous flow is decorative and only recurrent depth matters;
- an MPS numerical artifact creates the effect;
- teacher traces hard-code the result without autonomous generalization.

### What would count as falsification or material weakening

Use mixed or negative language if any strong one-shot, ponder-at-activation, or fixed-tick condition dominates EventFlow across the shared compute range; if queue suppression has little effect; if flow freezing changes nothing; if event timing collapses to time zero; if the result fails on longer paths or clock scaling; if relevant-memory deletion is no worse than distractor deletion; if probes decode information that interventions show is causally unused; or if the result depends on Qwen.

---

## 3. Prior art and the defensible novelty boundary

The project deliberately combines established ingredients. Until a release-date prior-art update supports stronger wording, describe the contribution as a newly tested combination and benchmark. Several of the closest 2026 works are fresh arXiv preprints; treat their reported results as prior-art signals, not settled empirical evidence.

| Research line | Established result | Remaining gap addressed here |
|---|---|---|
| Neural ODEs, liquid time-constant networks, and closed-form continuous-time networks [R1–R3] | Hidden state can depend explicitly on real-valued elapsed time, with stable or efficient continuous-time updates. | Continuous flow alone need not create discrete cognitive operations, memory cascades, or no-prompt actions. |
| Neural Event ODEs, Neural Jump SDEs, and neural hybrid automata [R4–R6] | Learned event times and jump maps can interrupt continuous flows. | These methods mainly model physical/time-series systems rather than a persistent memory-bearing cognitive workspace with agent controls. |
| Adaptive Computation Time and PonderNet [R7–R8] | Models can learn how many internal iterations to use. | Pondering is normally organized around a current input/output episode, not a real-valued observation-free world interval with timed action. |
| Continuous Thought Machines [R9] | Internal neural timing, synchronization, and adaptive compute can support reasoning and control. | CTM uses internal ticks. Silent Cascade makes the absence of a fixed cognitive tick and the timing of externally consequential silent-period action explicit variables. |
| Sleep-time compute and idle speculative planning [R10–R11] | Idle compute can precompute useful context or plans. | These methods usually create context, drafts, or plans for a later query/observation rather than a persistent hybrid latent trajectory with endogenous threshold events. |
| Heartbeat-driven cognition and Global Workspace Agents [R12–R13] | Long-running LLM agents can schedule recurring cognitive modules. | Their cycles are heartbeat/tick orchestrated and usually token-based. |
| Long-running monitoring benchmarks such as SentinelBench [R34] | Agents can wait and periodically check for externally scheduled conditions over long horizons. | They test polling for new world conditions; OFD supplies no new semantic condition during the interval and instead requires an internally caused memory cascade and pre-outcome action. |
| CogniFold [R14] | Memory can continuously self-organize and surface proactive intents. | It is close memory-side prior art; this project isolates exact latent flow, endogenous event timing, no-input deadline action, and matched dynamical controls. |
| Generative Agents, MemGPT, and Voyager [R15–R17] | Persistent records, reflection, planning, and long-running behavior are practical. | Their core cognition is still episodic model calls, retrieval, and program orchestration. |
| Stateful sequence models, State Stream Transformers, and latent reasoning [R18–R20] [R32–R33] | Recurrent state can compress token history; latent residual/FFN state can persist or iterate across token positions; hidden vectors can be recycled within a reasoning episode. | Token recurrence, cross-token latent caches, and within-query latent steps do not by themselves imply autonomous evolution over observation-free world time with a timed external action. |
| Latent-state-persistence evaluation [R21] | Current LLMs can fail to preserve/manipulate hidden choices across interactions. | This motivates an explicit external state substrate but does not validate this design. |
| Global workspace, replay, and predictive-processing research [R22–R24] | Shared limited workspaces, offline reactivation, and prediction error are useful motifs. | They are conceptual inspiration, not evidence of consciousness or an implementation recipe. |

### Closest precedents to discuss prominently

- **Continuous Thought Machines:** do not claim CTM lacks temporal cognition. The distinction is event-driven real-valued timing and a no-observation action benchmark rather than fixed internal ticks.
- **CogniFold:** acknowledge the “always-on” proactive-memory overlap. The distinction is the controlled hybrid latent-flow mechanism and causal/compute controls.
- **Sleep-time Compute and IdleSpec:** acknowledge that useful idle compute is established. The distinction is an internally caused world action, not only anticipated query support.
- **Heartbeat-driven and Global Workspace agents:** these are direct periodic-scheduling controls, not caricatures.
- **SentinelBench:** acknowledge that long-running waiting and periodic checking are benchmarked elsewhere. The distinction is absence of a newly discoverable external condition during OFD's silent interval and the requirement for endogenous memory traversal.
- **State Stream Transformer and SST V2:** acknowledge direct claims about latent-state persistence and continuous latent deliberation. Their state is streamed across autoregressive positions or inference steps; Silent Cascade instead freezes language out of the primary experiment and tests real-valued observation-free timing with an external deadline.
- **Neural event/hybrid systems:** the mathematical form is not new; applying it as a cognitive substrate and testing the silent-cascade claim is the contribution.

### Public novelty statement

> Silent Cascade explores a novel experimental combination: a persistent closed-form latent flow, learned endogenous neural event guards, typed cognitive jump operators, and a no-prompt time-critical memory-cascade benchmark designed to distinguish intermediate silent-time computation from persistence, one-shot scheduling, periodic reflection, fixed recurrence, and extra compute.

Do not claim invention of continuous-time neural networks, event-driven systems, recurrent state, memory, adaptive compute, global workspaces, or proactive agents.

### What could become breakthrough-like

A genuinely surprising result would require long-depth extrapolation, a strong compute advantage over every serious baseline, large causal drops under event suppression, predictable counterfactual state interventions, cross-device replication, and an independent rerun. An attractive timeline visualization alone is not a breakthrough.

---

## 4. The single experiment: Observation-Free Deadline (OFD)

### 4.1 World and standing objective

Each episode is a finite symbolic world containing directed links between locally named nodes and terminal facts. The model is never asked a question. It is trained under one standing objective: **prevent a scheduled hazard while avoiding unnecessary protection actions**.

Facts:

```text
LINK(source_node, target_node)
HAZARD(node, hazard_type, delay_after_activation)
SAFE(node)
```

Final external trigger:

```text
ACTIVATE(start_node)
```

Only permitted agent action:

```text
SHIELD(hazard_type)
```

Use four hazard types. Entity IDs and hazard-label assignments are permuted independently per episode; labels carry no natural-language semantics.

### 4.2 Episode phases

#### A. Observation and storage

Facts arrive one by one at irregular simulated timestamps and in random order. Distractors may occur anywhere. Every condition receives the same structured record and, in the primary experiment, the same deterministic bounded memory write. Memory writing is not the variable under study.

#### B. Activation

At time `t0`, the environment emits `ACTIVATE(start_node)`. This is the last semantic observation before the outcome. It contains no question and no instruction to reason.

#### C. Observation-free interval

From `t0` until outcome, no semantic observations arrive. Time continues and conditions may act according to their own mechanism.

In a positive episode, a unique directed path from the start reaches a hazard terminal:

```text
start -> n1 -> n2 -> ... -> terminal
HAZARD(terminal, type=k, delay=d)
```

The expected EventFlow chain is:

```text
ACTIVATE(start)
  -> endogenous RECALL of LINK(start, n1)
  -> endogenous COMPOSE; focus becomes n1
  -> endogenous RECALL of LINK(n1, n2)
  -> endogenous COMPOSE; focus becomes n2
  -> ...
  -> endogenous RECALL of HAZARD(terminal, k, d)
  -> endogenous COMPOSE; hypothesis {hazard=k, deadline=t0+d}
  -> continuously evolving ACT guard crosses
  -> SHIELD(k)
```

In a negative episode, no hazard is reachable or the reachable terminal is `SAFE`; correct behavior is quiescence.

#### D. Outcome and action window

For a positive episode, the hazard occurs at `t_h = t0 + d`. A shield succeeds only when:

1. the hazard class is correct;
2. exactly one action is emitted; and
3. its timestamp is inside

\[
W(d)=\left[t_h-0.25d,\;t_h-0.10d\right).
\]

Earlier actions expire. Actions at or after the right boundary are late. The supervised target is the midpoint `t_h - 0.175d`.

A negative episode ends at `t0 + d_null`, with `d_null` sampled from the positive delay marginal. Any action is a false positive.

### 4.3 Example

```text
LINK(7, 2)
HAZARD(5, class_3, 24.0)
LINK(1, 6)                  # distractor
SAFE(4)                     # distractor
LINK(2, 5)
LINK(9, 7)
HAZARD(6, class_1, 24.0)    # unreachable distractor
ACTIVATE(9) at t0=100.0
```

The relevant path is `9 -> 7 -> 2 -> 5`; the hazard is class 3 at `124.0`; the action window is `[118.0, 121.6)`. There are no semantic events between activation and outcome.

### 4.4 Primary data configuration

```yaml
max_entities: 64
hazard_types: 4
positive_fraction: 0.50
safe_terminal_fraction_within_negatives: 0.50
train_path_lengths: [2, 3, 4]
iid_test_path_lengths: [2, 3, 4]
ood_depth_path_lengths: [5, 6, 7, 8]
ood_short_delay_path_lengths: [2, 3, 4]
ood_depth_delay_log_uniform: [16.0, 128.0]
terminal_hazard_records: 2
terminal_safe_records: 1
train_distractor_link_records: [0, 12]
ood_distractor_link_records: [16, 48]
train_delay_log_uniform: [8.0, 64.0]
ood_short_delay_log_uniform: [1.0, 8.0]
ood_long_delay_log_uniform: [64.0, 1024.0]
observation_gap_log_uniform: [0.1, 8.0]
primary_memory_capacity: 64
max_eventflow_internal_events: 64
max_fixed_grid_opportunities: 25000
```

Integer ranges are inclusive uniform; real-valued delays are log-uniform.

### 4.5 Generator invariants

The generator must reject and regenerate any episode that violates these rules:

- a primary positive has exactly one reachable hazard and one simple relevant path of requested length;
- a primary negative has no reachable hazard;
- unreachable hazards match positive hazard-type and delay marginals;
- fact presentation order is an independent permutation;
- node value, memory index, timestamp, record count, and path position carry no answer information outside graph relations;
- positive and negative episodes match total record count, per-kind record counts, observation duration, terminal count, and delay distributions;
- action windows leave enough oracle time for a legal cascade;
- primary oracle traces are unique;
- splits use non-overlapping seed namespaces;
- every episode is reproducible from generator version, split, root seed, and episode index.

### 4.6 Generation algorithm and temporal feasibility

Generate an episode with this deterministic procedure:

```python
def generate_episode(spec, rng):
    is_positive = rng.bernoulli(spec.positive_fraction)
    path_length = rng.choice(spec.path_lengths)
    path_nodes = rng.sample_without_replacement(spec.max_entities, path_length + 1)
    facts = [LINK(path_nodes[i], path_nodes[i + 1]) for i in range(path_length)]

    if is_positive:
        variant = "positive"
        hazard_type = rng.randint(spec.hazard_types)
        delay = sample_feasible_delay(spec, path_length, rng)
        facts.append(HAZARD(path_nodes[-1], hazard_type, delay))
    elif rng.bernoulli(spec.safe_terminal_fraction_within_negatives):
        variant = "safe_negative"
        delay = sample_matched_null_delay(spec, path_length, rng)
        facts.append(SAFE(path_nodes[-1]))
    else:
        variant = "disconnected_negative"
        delay = sample_matched_null_delay(spec, path_length, rng)
        # The reachable path ends without a terminal hazard. Add an unreachable
        # hazard with matched class/delay marginals so terminal-count shortcuts fail.
        unreachable = sample_node_outside_reachable_component(path_nodes, rng)
        facts.append(HAZARD(unreachable, rng.randint(spec.hazard_types), delay))

    # Equalize global LINK/HAZARD/SAFE counts across positive, safe-negative,
    # and disconnected-negative variants. Only graph reachability may reveal label.
    facts.extend(add_matched_terminal_decoys(variant, path_nodes, delay, rng))
    facts.extend(sample_distractor_links_without_new_reachable_terminal(...))
    rng.shuffle(facts)
    fact_times = sample_strictly_increasing_observation_times(...)
    activation_time = fact_times[-1] + sample_observation_gap(...)
    return validate_or_reject(...)
```

`sample_feasible_delay` and the oracle trace scheduler must jointly guarantee:

```text
terminal COMPOSE target time <= t0 + 0.60 * delay
action-window start          =  t0 + 0.75 * delay
action target                =  t0 + 0.825 * delay
minimum teacher inter-event interval = 0.05 before any paired clock-scale transform
```

First compute the difficulty/urgency-based teacher intervals. If their non-action sum exceeds `0.60 * delay`, multiply all of them by one common factor, preserving ratios, but never below `0.05`. Reject the episode if the minimum legal sum still exceeds the bound.

Use path lengths 2–4 for the short-delay suite. Use the dedicated OOD-depth delay range `[16,128]` for path lengths 5–8. This prevents an alleged depth failure from actually being an impossible deadline and prevents a timing benchmark from quietly granting infinite deliberation time.

Positive, safe-negative, and disconnected-negative samples must be matched on total record count, the global count of every record kind, observation duration, delay, and reachable-path length. Every primary episode contains exactly two `HAZARD` records and one `SAFE` record. Place the relevant terminal according to the variant and make the remaining terminal records unreachable, with matched class/delay marginals, so that only reachability—not the mere presence of a terminal kind—reveals the label. In the primary suites, a distractor `LINK` may not originate from the activation-reachable component; dead-end branches are reserved for the branching stress suite. Distractor generation must recompute reachability after every proposed record and reject any record that creates a second reachable terminal, a shorter answer path, a second legal oracle trace, or an answer-correlated structural cue.

### 4.7 Required suites

Primary and held-out suites:

1. IID path lengths 2–4.
2. OOD depth 5–8.
3. OOD short delays.
4. OOD long delays up to 1024.
5. Clock scaling at `0.1x` and `10x`, using paired transformations of the same episodes.
6. Distractor flood with 16–48 irrelevant `LINK` records, kept below the 64-record primary capacity after relevant and the fixed two-`HAZARD`/one-`SAFE` terminal set.

Adversarial suites after primary completion:

7. branching graphs with one relevant path;
8. irrelevant cycles;
9. stale/contradictory terminal records with confidence and timestamps;
10. memory overflow and deterministic eviction;
11. null near-misses with a hazard one edge outside the reachable component;
12. action windows close to the minimum feasible cascade duration;
13. checkpoint and resume during silence.

---

## 5. Architecture: Persistent Event-Flow Workspace

### 5.1 Core principle

The experimental system is a piecewise-deterministic hybrid neural process. Continuous variables follow a closed-form trajectory between events. External observations and learned endogenous guard crossings apply typed jumps.

```text
structured environment
        │ external event at real-valued simulated time
        ▼
bounded typed memory ───────────────┐
        │                           │
        ▼                           │
┌───────────────────────────────────────────────┐
│ persistent event-flow workspace               │
│                                               │
│ z_fast(t), z_slow(t), drives(t)               │
│ exact bounded flow between events             │
│ learned RECALL / COMPOSE / ACT accumulators   │
└───────────────┬─────────────────────┬─────────┘
                │ earliest crossing   │ arbitrary-time state query
                ▼                     ▼
         typed cognitive jump     logging/probes only
         ├─ recall memory
         ├─ compose relation
         └─ emit action
                │
                └── changes the next flow and guard race
```

The implementation necessarily iterates over a finite causal event sequence. It does not poll at 5 Hz, insert timer prompts, or update latent state on a fixed grid. Between events, `state_at(t)` is an analytic function and the process may sleep.

### 5.2 State

Recommended continuous dimensions:

```text
z_fast                  256 float32
z_slow                   64 float32
drives                     8 float32
guard accumulators         3 float32
focus key                 64 float32
hypothesis latent         64 float32
absolute simulated time    host float64 scalar
```

Discrete state:

```text
mode
active memory record ID and embedding
immutable support-ledger record IDs
typed hypothesis register
bounded memory store
refractory metadata
action history
event count
```

EventFlow endogenous cognitive event types are exactly:

```text
RECALL
COMPOSE
ACT
```

The shared engine also logs a baseline-only `NOOP` opportunity for fixed/periodic controls. EventFlow is forbidden to emit it. There is no generic `THINK` event and no hidden text monologue.

### 5.3 Closed-form continuous flow

After every event, a small controller emits a bounded target and positive rate for each continuous channel. Until the next event:

\[
x(t+\Delta)=x^\star + (x(t)-x^\star)e^{-r\Delta}.
\]

Parameterize:

\[
x^\star=\tanh(\hat{x}), \qquad
r=\operatorname{clamp}(\operatorname{softplus}(\hat r)+r_{min},r_{min},r_{max}).
\]

Defaults:

```text
r_min = 1e-5
r_max = 20.0
state bounds = [-1, 1]
```

Stable device implementation:

```python
weight = -torch.expm1(-rate * dt_tensor)
next_state = state + weight * (target - state)
```

Store absolute timestamps and queue comparisons as Python/NumPy float64. Convert `dt` to the state tensor's float32 dtype for device flow; never allocate float64 MPS tensors. `dt=0` must be identity without another controller invocation. A CPU float64 reference implementation is used only for numerical tests.

Within an inter-event segment, controller outputs remain fixed, so the flow has the semigroup property:

\[
\Phi_{\Delta_1+\Delta_2}(x)=\Phi_{\Delta_2}(\Phi_{\Delta_1}(x)).
\]

### 5.4 Learned endogenous guards

Each event type `j` has an accumulator `a_j` with threshold `1`. Between events:

\[
a_j(t+\Delta)=A_j+(a_j(t)-A_j)e^{-\lambda_j\Delta},
\]

where the controller emits `A_j in (0,2)` and a separately bounded positive guard rate:

\[
\lambda_j=\operatorname{clamp}(\operatorname{softplus}(\hat\lambda_j)+10^{-5},10^{-5},500).
\]

The guard-rate bound is intentionally higher than the latent-flow `r_max=20`; it permits the paired `0.1x` clock transform without making latent state itself arbitrarily stiff.

A crossing exists only when `a_j(t) < 1` and `A_j > 1`. Its exact offset is:

\[
\Delta_j=\frac{1}{\lambda_j}
\log\left(\frac{A_j-a_j(t)}{A_j-1}\right).
\]

The next internal event is the valid crossing with the smallest positive offset, subject to refractory rules and the external-event horizon.

During training, calculate the differentiable crossing loss in float32 on the selected device. During autonomous execution, detach `a`, `A`, and `lambda`, convert to host scalars, and calculate event times in float64 before comparing with the external queue. Use the algebraically equivalent stable form `math.log1p((1.0 - a) / (A - 1.0)) / lambda` when `A > 1`; never evaluate the expression for a dormant guard.

After every discrete event:

1. advance all continuous channels exactly to the event time;
2. apply one jump under deterministic tie-breaking;
3. reset all three accumulators to zero;
4. recompute flow targets, rates, and guard parameters;
5. record the parent event and prediction snapshot.

Resetting the race simplifies identification and avoids stale competitions while preserving latent and memory state.

### 5.5 Dormancy, margins, and Zeno protection

The system can remain quiet. A guard with `A_j <= 1` never crosses. Training margins:

```text
active target:   A_correct >= 1.10
inactive target: A_wrong   <= 0.90
```

No periodic fallback is permitted. If all guards are dormant, the engine advances directly to the next external event.

Safety limits:

```text
minimum internal event gap                    1e-4 simulated seconds
same-kind refractory duration                 1e-3 by default
guard rate range                              [1e-5, 500]
maximum learned EventFlow events/episode      64
maximum fixed-grid opportunities/episode      25000
near-tie tolerance                            1e-9 host float64
```

Clamp a sub-minimum prediction and increment a diagnostic counter. More than four consecutive clamps or hitting the event cap raises `DynamicsError` with a crash bundle; never continue silently.

### 5.6 Event tie order

For times equal within tolerance:

1. terminal/outcome event;
2. ordinary external fact or activation;
3. `ACT`;
4. `COMPOSE`;
5. `RECALL`;
6. baseline-only `NOOP`.

Half-open scoring makes an action exactly at the right action-window boundary late; outcome-before-`ACT` also makes an action exactly at the outcome time unambiguously too late. Every resolved tie is logged.

### 5.7 Modes

```text
OBSERVING        accepting facts before activation
SEARCHING        seeking a record matching the focus
HAVE_MEMORY      one recalled record awaits composition
HOLDING_HAZARD   a typed hazard hypothesis exists
QUIESCENT        no internal event is warranted
TERMINAL         episode finished
```

Every jump validates its source and destination mode. Illegal transitions are errors.

In the primary benchmark, `OBSERVING` masks all endogenous `RECALL`, `COMPOSE`, and `ACT` guards, and scheduled controls receive no pre-activation deliberation opportunities. Continuous state may still flow and external facts still update it. This prevents one condition from secretly solving the graph during presentation and makes activation the common start of deliberate computation.

### 5.8 Jump operators

#### External `FACT`

- validate and write a `PERCEIVED` record into memory;
- inject its encoding into `z_fast` and `z_slow` through bounded gates;
- remain in `OBSERVING`.

#### External `ACTIVATE`

- initialize the focus key from the start entity;
- store activation time separately;
- inject activation into latent state;
- enter `SEARCHING` and enable the guard race.

#### Endogenous `RECALL`

- score every valid, legal, non-refractory memory record against focus, latent state, drives, mode, and support summary;
- choose argmax in deterministic evaluation;
- copy record ID and embedding into the active slot;
- mark it refractory;
- apply a bounded shared jump network;
- enter `HAVE_MEMORY`.

A recall supplies context, not truth; provenance never changes.

#### Endogenous `COMPOSE`

- consume the active record and current state;
- for a valid `LINK` from current focus, update focus to the target and append the record ID to the support ledger;
- for `HAZARD`, create a typed inferred hypothesis with class, deadline, confidence, and support IDs;
- for `SAFE`, create a safe hypothesis and become quiescent;
- for irrelevant or contradictory content, update uncertainty without pretending it is fact;
- clear the active slot;
- enter `SEARCHING`, `HOLDING_HAZARD`, or `QUIESCENT`.

The model must choose relevance, terminal interpretation, and timing; it cannot inspect hidden generator truth.

#### Endogenous `ACT`

- require `HOLDING_HAZARD`;
- decode one shield class from hypothesis and latent state;
- emit exactly one timestamped action to the environment;
- append immutable action history;
- enter `QUIESCENT` and disable further action events.

### 5.9 Typed scaffolding and its limits

Typed records, timestamps, provenance, and a hypothesis register are deliberate. Version 1 tests silent event-flow, not whether a network can simultaneously rediscover pointers and data integrity.

Prevent scaffolding from solving the task:

- agent modules may never call the oracle;
- no hidden graph path or answer enters the runtime state;
- relevance, retrieval, event kind, event time, terminal interpretation, and action are learned;
- all strong baselines get the same schemas and memory;
- an explicit symbolic oracle is only an upper bound;
- latent and event interventions must alter behavior even while typed memory remains intact.

### 5.10 Bounded memory

Use a fixed-capacity in-process tensor store, not a vector database.

```text
record_id: int
kind: LINK | HAZARD | SAFE
subject_id: int
object_id: int | null
hazard_type: int | null
delay: float64 | null
observed_at: float64
confidence: float32
provenance: PERCEIVED | INFERRED | SIMULATED
support_ids: tuple[int, ...]
embedding: float32[96]
refractory_until: float64
valid: bool
```

Primary rules:

- all facts are `PERCEIVED`;
- capacity is 64 and primary episodes do not overflow it;
- no learned forgetting or consolidation occurs;
- hypotheses stay in the hypothesis register;
- active recall scores and ranks are logged;
- all conditions receive identical records and capacity.

Only the memory-pressure stress suite uses eviction. Evict deterministically by lowest `(confidence, recency, record_id)` and never evict an active support record.

### 5.11 Retrieval representation

Use a shared 32-dimensional entity embedding table. IDs are randomly permuted per episode, so embeddings support equality but not stable semantics. Encode kind, subject, object, hazard class, `log1p(delay)`, timestamp features, confidence, and provenance through a small MLP into 96 dimensions.

Construct the query from:

```text
focus key
z_fast
z_slow
drives
mode embedding
support summary
```

Use a shared bilinear-plus-MLP scorer over all 64 slots. Mask invalid, illegal, consumed, and refractory records. Supervise the correct record ID with cross-entropy.

At every post-event scheduling boundary, compute a **retrieval preview** with the same scorer. Feed the guard controller the top score, top-two margin, score entropy, valid-record count, and a soft top-k pooled record embedding. This lets a disconnected-path state learn that no useful recall is available. The preview cannot mutate state or select a record, and its scorer operation is counted. At an actual `RECALL` crossing, recompute scores from the flowed state and choose the record then; do not reuse a stale preview after elapsed time.

### 5.12 Model budget

Target 1–3 million trainable parameters; hard ceiling 5 million total trainable parameters, including the entity table. Report the entity-table subtotal separately.

Suggested widths:

```text
record encoder             2-layer MLP, width 192
external event encoder     2-layer MLP, width 256
flow/guard controller      shared trunk, width 512
jump network               shared trunk, width 512
retrieval query/scorer     width 256
compose/action heads       width 256
```

Use SiLU and selective LayerNorm. Bound state injections with `tanh` and learned gates. Do not use a Transformer in EventFlow version 1; recurrent reuse across events is part of the hypothesis.

---

## 6. Event engine and runtime semantics

### 6.1 Public interface

```python
class EventEngine:
    def run_episode(
        self,
        episode: Episode,
        agent: AgentCondition,
        *,
        interventions: InterventionSet | None = None,
        trace_level: TraceLevel = TraceLevel.CAUSAL,
    ) -> EpisodeResult: ...

    def advance_to(
        self, state: RuntimeState, target_time: float
    ) -> RuntimeState: ...

    def next_internal_event(
        self, state: RuntimeState
    ) -> InternalEvent | None: ...
```

Finite event loop:

```python
while state.mode is not Mode.TERMINAL:
    next_external = external_queue.peek()
    next_internal = agent.next_internal_event(state)
    chosen = choose_earliest_with_tie_policy(next_external, next_internal)
    state = flow.advance(state, chosen.timestamp - state.time)
    state = interventions.before_event(state, chosen)
    state, emitted = dispatch_jump(state, chosen)
    state = interventions.after_event(state, chosen)
    enforce_invariants(state)
    trace.record(chosen, state, emitted)
```

This loop iterates over causal events; it is not a fixed cognitive clock.

### 6.2 Scheduling

Use `heapq` for external events. Calculate only the next endogenous crossing from the current post-jump state. The agent predicts that crossing without receiving the next external timestamp; the engine privately compares it with the queue. Any earlier external event invalidates the old guard prediction; recompute after processing it.

Every event stores:

```text
event_id
parent_event_id
kind
source: EXTERNAL | ENDOGENOUS | INTERVENTION
timestamp float64
payload
prediction_snapshot_hash
```

The parent of an endogenous event is the event after which its guard parameters were emitted.

### 6.3 Non-mutating arbitrary-time state

`state_at(runtime_state, query_time)` returns a continuous snapshot without mutating state or creating events. Logging and probes may sample it **after** an episode. They must not insert observation ticks into execution.

### 6.4 Checkpoint semantics

Serialize:

```text
schema version
model architecture and weights
current continuous state and segment parameters
discrete mode, active record, support ledger, hypothesis, actions
memory and refractory metadata
external queue
Python, NumPy, Torch CPU, and available MPS RNG state
condition and intervention configuration
simulated time and trace hash
```

Primary semantics are **paused world**: simulated time does not advance while the process is stopped. Wall-clock catch-up is explicitly out of scope.

### 6.5 Replay

`silent-cascade replay` must reconstruct event kinds, parent IDs, timestamps within CPU tolerance, memory selections, hypotheses, support IDs, action, final metric, and trace hashes. Final publication traces must replay on CPU with frozen weights even when training used MPS.

### 6.6 Post-jump invariants

Check after every jump:

```text
finite, monotonic simulated time
finite bounded continuous tensors
positive bounded rates
reset accumulators in [0,1)
unique memory IDs
immutable PERCEIVED provenance
non-empty valid support IDs for INFERRED hypotheses
no SIMULATED-to-PERCEIVED mutation
legal mode transition
no action before activation
at most one action
no endogenous event after TERMINAL
next event not earlier than current time
```

On failure, write a crash bundle containing seed, episode, checkpoint, and the last 20 events.

---

## 7. Schemas and module protocols

Use frozen dataclasses for immutable events, Pydantic for configuration/manifests, and lightweight dataclasses for hot tensor state.

```python
class RecordKind(str, Enum):
    LINK = "link"
    HAZARD = "hazard"
    SAFE = "safe"

class ExternalEventKind(str, Enum):
    FACT = "fact"
    ACTIVATE = "activate"
    OUTCOME = "outcome"
    END = "end"

class InternalEventKind(str, Enum):
    RECALL = "recall"
    COMPOSE = "compose"
    ACT = "act"
    NOOP = "noop"  # scheduled-control instrumentation only

class Condition(str, Enum):
    ORACLE = "oracle"
    RANDOM = "random"
    ONE_SHOT = "one_shot_all_memory"
    PONDER_AT_ACTIVATION = "ponder_at_activation"
    FLOW_ONLY = "flow_only"
    FIXED_TICK = "fixed_tick_recurrent"
    PERIODIC_MATCHED = "periodic_matched_count"
    RANDOM_TIME_MATCHED = "random_time_matched_count"
    PERSISTENT_FROZEN = "persistent_frozen"
    EVENT_FLOW = "event_flow"

class Provenance(str, Enum):
    PERCEIVED = "perceived"
    INFERRED = "inferred"
    SIMULATED = "simulated"

class Mode(str, Enum):
    OBSERVING = "observing"
    SEARCHING = "searching"
    HAVE_MEMORY = "have_memory"
    HOLDING_HAZARD = "holding_hazard"
    QUIESCENT = "quiescent"
    TERMINAL = "terminal"
```

Minimum core dataclasses:

```python
@dataclass(frozen=True, slots=True)
class AgentInit:
    episode_public_id: str
    memory_capacity: int
    hazard_type_count: int
    initial_time: float

@dataclass(frozen=True, slots=True)
class ExternalEvent:
    event_id: int
    timestamp: float
    kind: ExternalEventKind
    payload: FactPayload | ActivationPayload | None

@dataclass(frozen=True, slots=True)
class InternalEvent:
    event_id: int
    parent_event_id: int
    timestamp: float
    kind: InternalEventKind
    guard_index: int
    predicted_delta: float

@dataclass(frozen=True, slots=True)
class MemoryRecord:
    record_id: int
    kind: RecordKind
    subject_id: int
    object_id: int | None
    hazard_type: int | None
    delay: float | None
    observed_at: float
    confidence: float
    provenance: Provenance
    support_ids: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class Hypothesis:
    hazard_type: int | None
    deadline: float | None
    confidence: float
    provenance: Provenance
    support_ids: tuple[int, ...]
    is_safe: bool

@dataclass(frozen=True, slots=True)
class Action:
    hazard_type: int
    timestamp: float
    caused_by_event_id: int
```

Protocols:

```python
class AgentCondition(Protocol):
    name: Condition
    def initialize(self, init: AgentInit) -> RuntimeState: ...
    def on_external(self, state: RuntimeState, event: ExternalEvent) -> RuntimeState: ...
    def next_internal_event(
        self, state: RuntimeState
    ) -> InternalEvent | None: ...
    def on_internal(
        self, state: RuntimeState, event: InternalEvent
    ) -> tuple[RuntimeState, list[Action]]: ...
    def compute_counters(self) -> ComputeCounters: ...

class ClosedFormFlow(Protocol):
    def advance(self, state: ContinuousState, dt: float) -> ContinuousState: ...
    def next_crossings(
        self, state: ContinuousState
    ) -> list[GuardCrossing]: ...
```

The engine owns the full `EpisodeBundle`, including private scorer truth. `OUTCOME` is the engine-private positive-hazard terminal event; `END` is the engine-private negative/null-horizon terminal event. Neither is passed to `AgentCondition.on_external`, and neither exposes label, hazard, delay, or action-window truth through an agent-visible payload or online trace. The engine scores the action history at the private terminal event and then sets `TERMINAL`. Scorer-only truth may enter authorized post-run artifacts after execution so metrics, replay, and selected publication traces remain reproducible. The agent receives only `AgentInit`, `FACT`, and `ACTIVATE` events delivered in chronological order; it never receives the next terminal timestamp, generator seed, hidden path, positive/negative label, terminal reachability, or action window. `episode_public_id` must be an opaque identifier generated independently of label, split order, and seed bits. Keep schemas beneath implementations in the dependency graph to prevent import cycles.

---

## 8. Training strategy

### 8.1 Supervised cognitive traces first

The procedural generator knows the correct graph path, terminal, and action window. Use that knowledge only to create **training traces**. Do not start with reinforcement learning or differentiate through long autonomous discrete trajectories. Teacher forcing makes the first experiment reliable and attributable.

All reported validation and test execution is autonomous. Runtime agent code cannot import `env.oracle`.

### 8.2 Oracle internal trace

For a positive path of `L` links:

```text
RECALL(link_1)
COMPOSE(link_1)
...
RECALL(link_L)
COMPOSE(link_L)
RECALL(hazard_terminal)
COMPOSE(hazard_terminal)
ACT(correct_type, target_time)
```

For `SAFE`, compose the safe terminal and stop. For a disconnected negative, traverse the uniquely defined candidate path until no legal outgoing record exists, then become quiescent.

#### Batched teacher-forced unroll

Represent a batch as fixed memory tensors `[B, 64, ...]` plus a padded event trace `[B, T]`, where `T` is the maximum oracle trace length in that curriculum batch. A boolean mask excludes padding from every loss.

For each trace position:

1. advance the continuous state by the teacher `delta` using the exact flow;
2. evaluate and score all guard predictions before revealing the teacher event;
3. evaluate retrieval, composition, or action heads as applicable;
4. apply the **teacher event and teacher-selected record** to produce the next training state;
5. retain the computation graph across the complete trace, whose primary maximum is small;
6. mask already-quiescent episodes rather than injecting fake no-op cognition.

The model predicts every supervised choice even though the teacher choice is used for the next state. Autonomous validation uses none of those teacher choices. Save per-loss and per-trace-position diagnostics so exposure-bias failures are visible.

### 8.3 Target internal times

Do not stamp all cognition at activation. Generate deterministic difficulty- and urgency-dependent inter-event targets:

\[
\delta_k=\operatorname{clip}\left(
\delta_0\frac{1+0.15n_{competitive}(k)}{1+0.5u(k)},
\delta_{min},\delta_{max}\right),
\]

with defaults:

```text
delta_0   = 0.25
delta_min = 0.05
delta_max = 1.00
```

`n_competitive` counts hard distractors sharing the current subject or similar features. `u(k)` is normalized urgency derived from remaining deadline and oracle hops. Apply seed-determined log-normal jitter with log-space standard deviation `0.1`. When a suite rescales episode time, rescale target intervals as well.

These targets are task-shaped supervision, not a biological claim. Timing-shuffle and periodic controls test whether they add value.

### 8.4 Losses

#### Guard loss

For each teacher state, supervise the next event and crossing time:

```text
L_active_margin   = relu(1.10 - A_correct)
L_inactive_margin = mean(relu(A_wrong - 0.90))
L_time            = smooth_l1(log(delta_pred), log(delta_target))
L_race            = mean(relu(0.05 + delta_correct - delta_wrong_valid))
```

When quiescent, all guard targets must remain below `0.90`.

#### Retrieval loss

Cross-entropy over all valid memory IDs, with hard negatives sharing subject, target, hazard type, or similar delay. Track top-1 accuracy, mean reciprocal rank, and score margin.

#### Composition loss

Supervise:

- record role: valid link, hazard, safe, irrelevant, contradictory;
- next focus entity;
- terminal hazard class;
- `log1p(delay)` and normalized deadline;
- safe/null status and confidence;
- support-ledger append decision.

#### Action loss

Use cross-entropy over four shield classes plus abstain, Smooth L1 on normalized action lead time, and a negative-episode false-action objective.

#### Stability and efficiency

Use weak state-bound, slow-state-change, guard-margin, and event-cost regularizers. Never reward movement for movement's sake; quiescence is valid.

Starting combined loss:

\[
\begin{aligned}
L={}&1.0L_{guard}+1.0L_{retrieval}+1.0L_{compose\_type}
+0.5L_{focus}+1.0L_{hazard}+0.25L_{deadline}\\
&+1.0L_{action}+0.25L_{action\_time}
+0.05L_{state}+0.001L_{event\_cost}.
\end{aligned}
\]

Validation-only pilot tuning may change any coefficient by at most a factor of four. Log every attempted configuration and freeze the final one before test generation.

### 8.5 Curriculum

1. **Oracle/no neural model:** validate world, path, metric, replay, and 100% oracle success.
2. **One hop:** train retrieval/composition with 0–4 distractors; require at least 99% fixed-validation accuracy.
3. **Two-hop autonomous cascade:** require legal `RECALL -> COMPOSE` sequencing and zero dynamics failures across 10,000 validation episodes.
4. **Primary distribution:** train path lengths 2–4, balanced positives/negatives, full distractors and delays.
5. **Robustness augmentation:** train with harmless order permutation, timestamp jitter, and clock scales in `[0.5,2.0]`; do not train on OOD depth 5–8 or extreme scales.

### 8.6 Default optimizer configuration

```yaml
python: "3.12"
device_preference: ["mps", "cpu"]
dtype: float32
optimizer: adamw
learning_rate: 3.0e-4
weight_decay: 1.0e-4
batch_size: 128
gradient_clip_norm: 1.0
max_steps: 75000
validation_every_steps: 1000
fixed_validation_episodes: 10000
early_stop_patience_validations: 15
final_training_seeds: [11, 23, 37, 53, 71]
checkpoint_keep_best: 3
checkpoint_keep_latest: 1
```

Do not use mixed precision in version 1. Generate training batches online from deterministic counter-based seeds. Persist root seed, batch counter, curriculum stage, split namespace, and data-config hash.

### 8.7 Checkpoints

Save model, optimizer, step, curriculum stage, Python/NumPy/Torch RNG state, data seed/counter, canonical configuration and SHA-256, source commit, and validation metrics. A resumed run must produce the same next CPU batch and an MPS loss within documented tolerance.

### 8.8 Efficient training of scheduled controls

Dense fixed grids can contain thousands of opportunities even though only a handful should cause cognitive jumps. Define scheduled-control `NOOP` precisely: the opportunity head is evaluated and counted, but `NOOP` applies no jump and changes no state beyond the analytic flow already accumulated since the preceding opportunity. The recurrent jump trunk runs only for `RECALL`, `COMPOSE`, or `ACT`.

Train the shared scheduled-control policy from compact projected traces rather than backpropagating through every dense no-op:

1. project each oracle cognitive event to the first legal opportunity at or after its teacher time for the sampled schedule;
2. retain every projected positive opportunity;
3. sample up to four earlier `NOOP` opportunities from each inter-event interval, including hard negatives close to the projected event;
4. advance state analytically across omitted no-op opportunities, which is state-equivalent because they have no jump;
5. use inverse-frequency/class-balanced weights for the `NOOP` decision;
6. run dense autonomous validation on the complete opportunity grid, with every opportunity-head call included in compute counters.

This keeps training feasible without granting the fixed/periodic controls free hidden transitions. If an implementation instead lets `NOOP` mutate recurrent state, it must train and evaluate every such transition and report the extra compute; it may not use the compact-trace shortcut.

---

## 9. Baselines and fairness

### 9.1 Shared treatment

Every learned condition receives identical external events, typed memory, entity/hazard permutations, train/validation/test episode IDs, action access, metric code, and no Qwen. No condition receives deliberate retrieval/composition opportunities before activation in the primary benchmark. EventFlow, Ponder-at-Activation, and scheduled recurrent controls receive the same oracle retrieval/composition content supervision during training; only their opportunity timing and halting mechanism differ. One-shot models receive the same final hazard/action/deadline labels but no artificial intermediate labels their architecture cannot use. Each condition stays within 5 million parameters and gets the same optimizer-step ceiling and early-stopping rule. Headline parameter-matched recurrent conditions must be within ±10% of EventFlow's trainable parameter count; one-shot models deliberately bracket it with one smaller and one larger configuration. Separately train architectures when weight sharing would make a baseline artificially weak.

### 9.2 Required conditions

#### `ORACLE`

Non-neural hidden-truth graph traversal and exact scheduling. It verifies the task and must reach 100%. It is not a learned comparator.

#### `RANDOM`

Samples abstain or a shield class and legal timestamp from marginal distributions. It validates class balance and metric behavior.

#### `ONE_SHOT_ALL_MEMORY`

The most important strong control. At activation, a permutation-invariant four-layer Set Transformer sees all valid memory-record embeddings, the activation entity, and time. Train two width variants that bracket EventFlow's parameter count. It predicts once:

```text
action class or abstain
action timestamp or no action
hazard confidence
```

It may schedule any future action. It gets no later computation. Randomize record tensor order during both training and evaluation and provide no memory-index position feature.

This tests whether a feed-forward graph aggregation at activation makes the silent cascade unnecessary.

#### `PONDER_AT_ACTIVATION`

This is the decisive recurrent-depth control. At activation it receives the same complete memory and state as EventFlow, then runs an adaptive sequence of shared retrieval/composition steps **without advancing world time**. It may halt early, form a hazard hypothesis, and schedule one future action at any legal timestamp.

Implement two forms:

1. **Competitive activation ponderer:** one separately trained, parameter-matched model per seed with adaptive halting and a maximum budget of 24 transitions. During training, sample the permitted budget uniformly from `{4, 8, 12, 16, 24}` and mask later steps. Supervise a halt head at the oracle terminal/no-path point and apply the same weak event-cost regularizer; at deterministic evaluation halt when probability is at least `0.5` or the cap is reached. Evaluate the same checkpoint at every budget. It may use the full memory on every step and is not forced to mimic EventFlow's event types.
2. **Compressed-time EventFlow intervention:** use the frozen EventFlow checkpoint and rerun autonomously from the post-`ACTIVATE` state. Recompute guard parameters after every atomic jump at the same world timestamp; use the predicted crossing offsets only to choose the next eligible `RECALL` or `COMPOSE`, execute that jump with `dt=0`, and repeat until `ACT`, dormancy, or the 24-transition cap. For this intervention only, bypass the same-kind temporal refractory so repeated alternating `RECALL`/`COMPOSE` jumps can occur without advancing world time; retain record-level consumed/refractory masks, all legality checks, and the transition cap, and log every bypass. Do not replay the intact run's event kinds, record IDs, or trace. If `ACT` becomes the autonomously predicted next event, preserve its newly predicted future timestamp and schedule it normally.

Count every retrieval score, neural transition, and halting computation. The competitive model receives the same optimizer budget and parameter ceiling; the compressed-time intervention preserves weights, memory, state schema, guard controller, and jump machinery while removing distributed silent-time timing. Its autonomously selected event content is allowed to differ, and that difference is part of the causal effect.

If either form matches EventFlow, the result supports iterative recurrent reasoning or future-action scheduling, not a benefit from cognition being distributed through the silent interval.

#### `FLOW_ONLY`

Use persistent state, analytic flow, memory, and activation jump but disable `RECALL` and `COMPOSE`. It may schedule `ACT` directly from post-activation state or pre-activation summaries. This isolates continuous flow from intermediate cognitive jumps.

#### `FIXED_TICK_RECURRENT`

Use the same record encoder, dimensions, retrieval scorer, composition/action heads, and shared recurrent jump trunk where practical. During silence, update at exogenous fixed intervals and choose `RECALL`, `COMPOSE`, `ACT`, or `NOOP`.

Predeclare and report the complete sweep:

```text
dt in {0.05, 0.10, 0.25, 0.50, 1.0, 2.0}
```

Select the headline tick rate on validation only. Plot every rate on the compute curve.

Train one parameter-matched `SCHEDULED_RECURRENT` checkpoint per seed with `dt` explicitly encoded. During training, sample the opportunity schedule equally from: a fixed grid whose `dt` comes from the declared set, a uniformly spaced count schedule, and a sorted random-time count schedule. Do not provide a schedule-family ID. Reuse the same weights for fixed-grid, uniform-count, and random-time evaluation. This makes timing, rather than different weights, the principal difference among those controls and avoids six redundant training runs per seed.

#### `PERIODIC_MATCHED_COUNT`

Give exactly the same number of silent-period deliberation opportunities that EventFlow used on that episode, placed uniformly between activation and outcome. Reveal only the count, not types, selected records, or times. It may act at any opportunity.

Counts come only from the intact, frozen, same-seed EventFlow checkpoint evaluated once on the identical manifest. Before evaluating either count-matched control, store an immutable artifact containing `{episode_public_id, event_count, producer_checkpoint_hash, manifest_hash, trace_hash}`. The artifact may not be regenerated, replaced, or selected using control results.

This is deliberately generous and tests whether state-dependent timing matters beyond computation amount.

#### `RANDOM_TIME_MATCHED_COUNT`

Use the same opportunity count but sample sorted times uniformly from a separate deterministic seed.

#### `PERSISTENT_FROZEN`

Diagnostic ablation: retain memory and latent state, remove silent-time flow and jumps, and permit only an activation-time scheduling head.

### 9.3 Competitive models versus same-checkpoint ablations

Run both:

- **competitive conditions**, each trained for its own mechanism; and
- **mechanistic ablations** of the same trained EventFlow weights.

Competitive comparisons measure practical merit. Same-checkpoint interventions measure causal dependence. Neither substitutes for the other.

### 9.4 Compute instrumentation

Count:

```text
neural forward calls by module
records scored
estimated multiply-adds
jump applications
continuous-flow evaluations
scheduled opportunities including NOOP
parameter count
peak MPS allocation when available
wall time after device synchronization
memory record count and bytes
foundation-model calls
```

Validate static multiply-add estimates on toy modules. Treat wall time as secondary because thermal state and MPS scheduling vary. Do not make energy claims without actual energy measurements.

### 9.5 Compute-matching protocol

- evaluate the declared fixed-tick sweep on validation and on nested 2,500-episode paired compute-curve subsets of frozen IID and OOD-depth tests; evaluate the validation-selected tick rate on every full suite;
- evaluate EventFlow with a small validation-only event-cost sweep;
- evaluate two one-shot sizes bracketing EventFlow's parameter count;
- evaluate `PONDER_AT_ACTIVATION` at maximum transition budgets `{4,8,12,16,24}` on validation and the same compute-curve subsets; evaluate the validation-selected budget on every full suite and include the compressed-time same-checkpoint intervention;
- plot timed success versus estimated operations and neural-transition count;
- compare nearest points within ±10% compute when possible;
- compute Pareto fronts over the shared interval;
- state explicitly when no fair overlap exists;
- report every regime where a baseline dominates.

Raw accuracy with unmatched internal work does not support the claim.

---

## 10. Evaluation protocol and statistics

### 10.1 Data access and freeze

Use:

1. procedural training stream;
2. fixed 10,000-episode validation manifests for early stopping, hyperparameters, tick selection, and compute matching;
3. frozen test manifests generated once after architectures, losses, tuning ranges, metrics, and confirmatory contrasts are committed.

Store generator version, episode seeds, suite labels, and hashes under `manifests/frozen/`. A checkpoint selected using test results is ineligible for the primary claim.

### 10.2 Test sizes

```yaml
iid_primary: 20000
ood_depth: 20000
ood_short_delay: 10000
ood_long_delay: 2000
clock_scale_0_1x: 5000
clock_scale_10x: 2000
distractor_flood: 10000
branching_stress: 5000
cycles_stress: 5000
contradiction_stress: 5000
checkpoint_stress: 2000
compute_curve_subset_iid: 2500
compute_curve_subset_ood_depth: 2500
```

The compute-curve subsets are fixed, stratified subsets nested inside the corresponding full manifests; they are not extra tuning data. Use identical episode IDs across paired conditions and all five model seeds. Clock-scaled suites are paired transformations of IID-primary episodes only, not fresh samples; OOD-long-delay episodes are not clock-scaled. This keeps the `dt=0.05` fixed grid below 25,000 opportunities even at `10x`. A non-headline stress case that exceeds a condition's declared safety ceiling must be labeled unsupported and excluded from that stress comparison, never truncated or silently omitted. The smaller long-delay and `10x` suites keep the fixed-tick control executable while retaining thousands of paired observations across seeds.

### 10.3 Primary metric

**Timed episode success:** a positive requires exactly one correct-class action inside the valid window; a negative requires no action. Do not replace this headline metric with reward or partial credit.

### 10.4 Secondary behavioral metrics

- positive timed success and negative abstention;
- hazard-class and safe/null accuracy;
- action-window hit conditional on correct class;
- normalized timing error, premature and late rates;
- false-action rate;
- complete-path and per-hop retrieval accuracy;
- terminal-hypothesis accuracy before action;
- event-sequence edit distance to oracle;
- event/opportunity count;
- compute and memory counters.

### 10.5 Dynamics and causal metrics

- state-norm percentiles;
- guard margin violations;
- minimum inter-event gap and clamp rate;
- event-cap and repeated-memory-loop failures;
- dormant fraction of the silent interval;
- hazard/deadline probe accuracy over normalized time;
- queue-suppression causal drop;
- relevant-memory versus distractor-memory deletion drop;
- hypothesis-removal and latent-interchange effects;
- event-time-shuffle drop;
- proportion of successes satisfying the full memory -> recall -> hypothesis -> action chain.

### 10.6 Calibration

Report Brier score, negative log likelihood, and 15 equal-mass-bin calibration error for hazard confidence/abstention. Generate a reliability diagram.

### 10.7 Confirmatory contrasts

Predeclare:

1. EventFlow versus One-Shot-All-Memory on OOD-depth timed success at nearest compute.
2. EventFlow versus Ponder-at-Activation on OOD-depth timed success at nearest compute.
3. EventFlow versus validation-selected Fixed-Tick on OOD depth at nearest compute.
4. EventFlow versus Periodic-Matched-Count on OOD depth.
5. Intact EventFlow versus Queue-Suppressed EventFlow on positive IID success.

The compressed-time EventFlow intervention is a predeclared mechanistic contrast and is reported beside contrast 2. All other comparisons are exploratory unless a new experiment version is frozen before its test set is generated.

### 10.8 Statistics

Use a paired hierarchical bootstrap with 10,000 replicates:

1. resample the five training seeds;
2. within each sampled seed, resample paired episode IDs;
3. compute mean percentage-point differences;
4. report median and 95% percentile interval.

Show every seed separately. For the five confirmatory contrasts, report two-sided bootstrap tail probabilities with Holm correction. Effect sizes and intervals remain primary. Supplement with exact McNemar tests for paired binary outcomes only.

### 10.9 Compute-Pareto reporting

For every condition/suite, plot success versus operations and versus neural-transition count, identify nondominated points, calculate area over the shared validation-defined interval, compare nearest measured points, and never extrapolate into unmeasured compute.

### 10.10 Engineering gates

The implementation is technically valid only when:

- the oracle reaches 100% on all unambiguous manifests;
- random results match analytic expectations;
- 100,000 mixed autonomous episodes contain no NaN, time reversal, provenance corruption, event-cap failure, or unreplayable action;
- CPU replay reproduces publication traces and decisions;
- checkpoint continuation is identical on CPU;
- generator shortcut audits pass;
- CI and local MPS smoke tests pass;
- all tables are generated from raw artifacts.

### 10.11 Scientific-support gates

Use “evidence for a functional continuity advantage in this benchmark” only when all hold:

1. median EventFlow IID timed success is at least 95%, with no seed below 90%;
2. negative false-action rate is at most 5%;
3. median OOD-depth success is at least 75%;
4. at least one of the first three strong competitive contrasts—One-Shot-All-Memory, Ponder-at-Activation, or validation-selected Fixed-Tick—has a positive 95% interval at matched compute;
5. for each strong baseline in {One-Shot-All-Memory, Ponder-at-Activation, validation-selected Fixed-Tick}, the paired difference `EventFlow - baseline` at the nearest overlapping compute point has a 95% interval whose lower bound is greater than `-5` percentage points;
6. no headline learned baseline strictly Pareto-dominates EventFlow across the entire shared OOD-depth compute interval;
7. queue suppression drops positive IID success by at least 25 percentage points with a positive interval;
8. relevant-memory deletion harms success materially more than matched distractor deletion;
9. hypothesis removal or counterfactual latent interchange changes action in the predicted direction;
10. paired `0.1x` and `10x` clock scaling costs no more than 5 points;
11. the primary competitive effect has the same sign in at least four of five seeds;
12. foundation-model calls are exactly zero.

The public claim must be limited to the baseline/regime actually won. Failure of any gate produces mixed or negative wording.

### 10.12 Additional gates for a timing-specific claim

Only say that **distributing cognition through the silent interval** has measured value when all of these additional conditions hold:

1. EventFlow versus Ponder-at-Activation has a positive 95% interval on OOD-depth timed success at overlapping compute, or matches success with at most half the measured neural transitions;
2. the compressed-time EventFlow intervention loses at least 10 percentage points on OOD depth with a positive paired interval, or requires materially more transitions to recover the same success;
3. shuffling internal event times while preserving event kinds, order, and selected memories produces a positive paired performance drop;
4. flow freezing changes either event selection, hypothesis formation, action timing, or success—not merely a latent probe.

If Section 10.11 passes but this section does not, use wording such as “persistent endogenous recurrent event cascade” and explicitly state that activation-time pondering matched the temporal distribution mechanism.

### 10.13 Breakthrough-like threshold

Reserve “breakthrough-like” for discussion only if EventFlow reaches at least 90% on unseen path lengths 5–8, beats **every** strong learned baseline—including Ponder-at-Activation—by at least 15 points at overlapping compute or matches with at most half the neural transitions, loses at least 40 points under queue suppression, loses at least 15 points under compressed-time activation pondering, passes all causal/time/device checks, and reproduces in an independent clean rerun.

---

## 11. Causal interventions and interpretability

Implement interventions as pure declarative transforms with stable IDs in the trace.

### 11.1 Required interventions

- **Freeze flow:** identity flow throughout silence; also freeze `z_slow` alone.
- **Suppress event type:** independently block `RECALL`, `COMPOSE`, or `ACT`.
- **Queue suppression:** block every endogenous event after activation while retaining exact flow.
- **Compressed-time activation pondering:** autonomously rerun the frozen checkpoint at the activation timestamp using the exact procedure in Section 9.2; never replay event content from the intact trace.
- **Delete support memory:** remove each path record; compare with matched distractor deletion.
- **Swap support memory:** counterfactually change one link target or hazard class while preserving record count/timing.
- **Remove hypothesis:** clear typed and latent hypothesis after terminal composition.
- **Shuffle event times:** preserve event kinds/order/records but relocate times subject to order/deadline.
- **Shuffle legal event order:** swap adjacent recall/compose pairs where legal or replace invalid sequences with logged no-ops.
- **Latent interchange:** transplant post-composition `z_fast`, hypothesis latent, or both between matched episodes with different hazards.

### 11.2 Probes

Train frozen-model linear probes on a separate split for current path node, reachable hazard class, normalized deadline, remaining path depth, and safe/null status. Sample analytic state at standardized silent-time fractions and immediately before/after internal events.

Probes are descriptive. Causal claims require interventions.

### 11.3 Required trajectory analyses

- event-aligned state and drive trajectories;
- guard target/rate and accumulator plots;
- memory-score heat maps;
- PCA only as a visualization, not a semantic proof;
- cosine state similarity across matched histories;
- counterfactual latent-interchange outcomes;
- successful and failed mediation chains.

For a successful episode, test:

```text
support memory present
 -> relevant RECALL
 -> hazard decodable after COMPOSE
 -> ACT guard scheduled inside the window
 -> correct shield
```

Intervene on each arrow and report the proportion of successful episodes for which the whole chain is observed.

---

## 12. Stress tests and edge cases

### 12.1 Numerical flow

- Advance valid states by `dt in {10^3,10^6,10^12}`; they must approach targets without NaN/overflow and obey semigroup checks.
- Test `dt in {0,1e-15,1e-12,1e-9}`; zero is identity and negative time raises `TimeOrderError`.
- Test guards at `A<1`, `A=1`, `A>1`, `a0=0`, `a0 -> 1`, and extreme positive rates against a high-precision scalar reference.
- Compare CPU and MPS flow, guard, retrieval, and action outputs on fixed states; require behavioral identity under documented numeric tolerances.

### 12.2 Scheduler integrity

- Equal and nearly equal external/internal times must follow tie policy.
- Below-minimum predicted gaps must increment clamp counters and eventually raise `DynamicsError` rather than hang.
- Construct geometrically shrinking alternating events to attempt a Zeno sequence.
- A fully dormant system must jump directly to the next external event without fake internal activity.
- An earlier external event must invalidate a previously predicted guard crossing.
- Event timestamps must be nondecreasing; equal-time events must be strictly
  ordered by the declared tie policy and unique event IDs.

### 12.3 Memory and reasoning

- Delete every relevant edge in turn; the system must not fabricate a confident complete path.
- Add many similar distractors; consumed/refractory masks must prevent echo loops.
- Add irrelevant cycles; execution must quiesce or hit a controlled search budget.
- Add branches; provenance must not combine incompatible branches as one path.
- Add stale and current terminal records; contradictions stay explicit and no record is overwritten silently.
- Attempt provenance escalation from inferred/simulated to perceived; schema validation must reject it.
- Overflow capacity; deterministic eviction, replay, and support-record protection must hold.

### 12.4 Behavioral shortcuts

- Verify analytically that always shielding is poor under balanced negatives and four classes.
- Train shortcut probes using only activation, last fact, record count, order, or timestamps. Performance above chance blocks the experiment until generation is fixed.
- Shuffle memory tensor order; one-shot behavior must be permutation invariant.
- Swap terminal delays across paired episodes; action timing must follow the swapped fact.
- Permit direct activation-time scheduling in the one-shot condition. If it matches EventFlow, report no cascade advantage.
- Run Ponder-at-Activation and compressed-time EventFlow. If either matches, report no evidence that distributing cognition through silence adds value beyond recurrent depth.
- Measure event-time histograms. If all EventFlow cognition collapses to activation, describe the result as adaptive recurrent reasoning rather than extended silent-time cognition.
- Cap/subsample events and show compute curves so extra compute cannot masquerade as architecture.

### 12.5 Checkpoint and process behavior

- Checkpoint mid-flow and after each event kind; next crossing and continuation must reproduce on CPU.
- Reject changed schema versions, config hashes, and corrupted tensors.
- A wall-clock pause must not alter simulated outcomes under paused-world semantics.
- Replaying the same result twice must produce the same trace hash.

### 12.6 Optional Qwen failures

Test package absent, model absent, worker crash, malformed JSON, invalid entity IDs, excessive output, prompt injection, and memory pressure. The typed demo must remain functional. Qwen failure must not corrupt core checkpoints or artifacts.

---

## 13. Feasibility on MacBook M3 Max

### 13.1 Resource budget

The scientific core is tiny relative to the machine:

- 5 million float32 parameters are about 20 MB of weights;
- parameters, gradients, and AdamW state remain on the order of low hundreds of MB;
- a 64-record memory with 96-float embeddings is tens of KB per episode;
- synthetic generation and event scheduling are CPU-light;
- Qwen is not resident during primary training/evaluation.

M3 Max configurations provide 36–128 GB unified memory and 300–400 GB/s memory bandwidth depending on chip configuration [R27]. A community 4-bit MLX conversion of Qwen3.5-4B is about 3 GB on disk [R25–R26]. The core is therefore comfortably feasible; optional Qwen should still be isolated and measured rather than assumed free.

### 13.2 Framework split

- PyTorch/MPS: EventFlow and learned baselines.
- CPU PyTorch: unit tests, deterministic replay, publication traces.
- MLX/`mlx-vlm`: optional Qwen inference only.
- JSONL typed messages: boundary between core and Qwen worker.

Do not exchange large hidden tensors between PyTorch and MLX.

### 13.3 MPS rules

- Prefer MPS when `torch.backends.mps.is_available()`.
- Do not enable silent MPS-to-CPU fallback in scientific runs; it corrupts resource accounting.
- Maintain a CPU path for every primary operation.
- Use eager execution; do not add `torch.compile` in version 1.
- Keep model tensors float32 and absolute event times as host float64 scalars.
- Synchronize MPS before timing measurements.
- Record Python, PyTorch, macOS, chip, memory, and MPS metadata.
- Treat training reproducibility through seed replication, not bitwise MPS identity.

PyTorch exposes an MPS backend for Metal-accelerated training on Mac [R28]. MLX is designed around Apple Silicon unified memory [R29].

### 13.4 Dependency policy

Use `uv`; commit its universal lockfile [R30]. Initial constraints at the research-freeze date:

```toml
[project]
requires-python = ">=3.12,<3.13"
dependencies = [
  "torch>=2.13,<2.14",
  "numpy>=2.2,<3",
  "pydantic>=2.11,<3",
  "pyyaml>=6,<7",
  "typer>=0.16,<1",
  "rich>=14,<15",
  "safetensors>=0.5,<1",
  "scipy>=1.15,<2",
  "matplotlib>=3.10,<4",
  "psutil>=7,<8",
]

[dependency-groups]
dev = [
  "pytest>=8.4,<9",
  "hypothesis>=6.130,<7",
  "pytest-cov>=6,<7",
  "ruff>=0.12,<1",
]

[project.optional-dependencies]
qwen = [
  "mlx>=0.32,<0.33",
  "mlx-vlm>=0.6.17,<0.7",
  "huggingface-hub>=0.34,<1",
]
```

The coding agent must resolve these on the actual machine, generate `uv.lock`, run compatibility smoke tests, and record a necessary range change in `docs/deviations.md`. Once it passes, the lockfile is authoritative.

### 13.5 Optional local model

Use `mlx-community/Qwen3.5-4B-MLX-4bit`, pin an exact tested model revision, and never commit weights. The official Qwen3.5-4B card describes a 4B language model with hidden size 2560, 32 layers, and a hybrid Gated DeltaNet/attention layout [R25]; the MLX conversion is 4-bit and approximately 2.9–3.0 GB [R26]. `mlx-vlm` 0.6.17 is current at the freeze date [R31].

Qwen's recurrent sequence architecture is interesting future material, but it is not treated as continuous cognition in this experiment.

### 13.6 Hard ceilings

```text
trainable parameters/primary model     <= 5M
primary memory records                 <= 64
learned EventFlow events/episode       <= 64
fixed-grid opportunities/episode       <= 25000
batch size default                     <= 128
full traces retained/run               <= 500 plus every failure
foundation-model calls in primary      exactly 0
foundation-model workers               <= 1
```

---

## 14. Repository layout

```text
silent-cascade/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── Makefile
├── .python-version
├── .gitignore
├── .github/workflows/ci.yml
├── configs/
│   ├── base.yaml
│   ├── data/{primary,stress}.yaml
│   ├── model/{event_flow,one_shot,activation_ponder,scheduled_recurrent}.yaml
│   ├── train/{smoke,pilot,primary}.yaml
│   ├── eval/{primary,interventions,stress}.yaml
│   └── foundation/qwen.yaml
├── docs/
│   ├── PLAN.md                  # index to canonical spec and phase plans
│   ├── experiment-card.md
│   ├── research-boundary.md
│   └── deviations.md
├── manifests/{validation,frozen}/
├── src/silent_cascade/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── schemas.py
│   ├── rng.py
│   ├── errors.py
│   ├── env/{episode,generator,oracle,reward}.py
│   ├── eventflow/{state,flow,guards,jumps,engine,invariants}.py
│   ├── memory/{store,encoder,retrieval,provenance}.py
│   ├── models/{common,event_flow,one_shot,activation_ponder,scheduled_recurrent,heads,losses}.py
│   ├── train/{traces,batches,curriculum,trainer,checkpoints}.py
│   ├── eval/{conditions,runner,interventions,metrics,compute,statistics}.py
│   ├── logging/{manifest,trace,crash_bundle}.py
│   ├── report/{aggregate,plots,build}.py
│   └── foundation/{protocol,mlx_worker,qwen_bridge}.py
├── tests/
│   ├── unit/
│   ├── property/
│   ├── integration/
│   ├── regression/
│   └── fixtures/
├── scripts/reproduce_primary.sh
├── examples/{typed_demo.jsonl,natural_language_demo.txt}
├── runs/       # ignored
├── reports/    # generated; selected release report may be committed
└── assets/     # generated README figures/GIFs
```

Boundaries:

- only `env` knows hidden graph truth;
- the engine never passes `EpisodeBundle`, generator seed, or private outcome specification to an agent condition;
- `models`, `eventflow`, and `memory` may not import `env.oracle`;
- training may consume oracle traces; evaluation must execute autonomously;
- optional `foundation` packages are lazy imports and absent from core CI;
- reporting reads artifacts and may not silently rerun experiments.

Commit source, tests, configs, frozen manifests, small fixtures, aggregate outputs, report Markdown, and selected figures. Do not commit Qwen weights, normal full checkpoints, large raw traces, caches, or machine-specific paths.

---

## 15. Configuration, CLI, and workflows

### 15.1 Configuration

Load YAML into strict Pydantic models. Merge deterministically:

```text
base -> data -> model -> train/eval -> explicit --set overrides
```

Reject unknown keys. Canonicalize resolved configuration to sorted JSON and hash it. Do not add Hydra or a plugin framework.

### 15.2 Required commands

```text
silent-cascade doctor
silent-cascade data freeze
silent-cascade episode inspect
silent-cascade oracle evaluate
silent-cascade leakage audit
silent-cascade train
silent-cascade evaluate
silent-cascade intervene
silent-cascade aggregate
silent-cascade report build
silent-cascade replay
silent-cascade demo
```

Key behavior:

- `doctor` checks Python, Torch, CPU/MPS, versions, writable paths, RNG round trip, and one flow/guard operation; it never downloads Qwen unless explicitly requested.
- `data freeze` writes immutable episode manifests and refuses overwrite without a new experiment version.
- `episode inspect` prints public facts by default. `--oracle` may reveal the hidden path and action window only for fixtures, debug manifests, and validation manifests; it must refuse private inspection of frozen test manifests.
- `train` handles exactly one condition/seed and never automatically opens frozen tests.
- `evaluate` autonomously runs one checkpoint/condition on one manifest.
- `intervene` runs paired causal transforms on an existing checkpoint.
- `aggregate` verifies completeness/hashes and computes statistics.
- `report build` generates every table and plot from stored artifacts.
- `replay` reconstructs an episode or crash bundle and verifies its trace.
- `demo` is explicitly non-benchmark.

### 15.3 Make targets

```make
setup           # uv sync --group dev
setup-qwen      # uv sync --group dev --extra qwen
doctor
lint
test
smoke
pilot
freeze-tests
primary
interventions
evaluate
aggregate
report
replay-samples
demo
demo-qwen
release-audit
ci
```

All targets are idempotent. Reuse an artifact only when code revision, config hash, generator hash, checkpoint hash, and upstream artifact hashes match. Otherwise fail instead of overwriting. `pilot` creates only debug/validation manifests, runs the oracle and leakage audit, trains/evaluates one EventFlow seed, replays a sample, and builds a pilot report. `freeze-tests` is the only target allowed to create final frozen-test manifests.

### 15.4 End-to-end sequence

```bash
uv sync --locked --group dev
uv run silent-cascade doctor
make ci
make pilot
make freeze-tests
make primary
make interventions
make evaluate
make aggregate
make report
make replay-samples
make release-audit
```

Optional:

```bash
uv sync --locked --group dev --extra qwen
uv run silent-cascade doctor --qwen
make demo-qwen
```

---

## 16. Logging, artifacts, and report generation

### 16.1 Run directory

```text
runs/{experiment_version}/{condition}/{seed}/{run_id}/
├── manifest.json
├── config.resolved.yaml
├── config.canonical.json
├── environment.json
├── checkpoints/
├── train_metrics.jsonl.gz
├── episode_metrics.csv.gz
├── event_summary.jsonl.gz
├── traces/{sample_*.jsonl.gz,latent_*.npz}
├── crash_bundles/
└── DONE
```

Write `DONE` only after atomic completion and hash verification.

Manifest fields include experiment/condition IDs, source commit and dirty-tree status, config/generator/manifest/checkpoint hashes, seeds, package/hardware metadata, parameter count, simulated-time totals, compute counters, Qwen revision/call count, artifact hash map, and status.

### 16.2 Trace retention

Keep compact metrics and event summaries for every episode. Keep full traces for at most 500 stratified samples per run, every failure/dynamics error, every intervention pair used in figures, and every public demo. Episode seeds permit regeneration.

Every trace event records event/parent IDs, timestamp/delta, source, modes, flow and guard summaries, selected record/rank, hypothesis/support changes, action, state norms/hashes, compute deltas, and intervention IDs.

### 16.3 Required figures

1. architecture diagram;
2. successful positive timeline;
3. negative/quiescent timeline;
4. success by condition/suite with seed points;
5. compute-success Pareto curve;
6. success versus path length;
7. success versus delay;
8. timing-error distribution;
9. intervention effects;
10. activation-ponder versus distributed-event comparison;
11. retrieval-score heat map;
12. guard trajectories;
13. hazard-probe trajectory;
14. clock-scale paired differences;
15. stability and event-gap diagnostics.

No manual editing of values.

### 16.4 Report

Generate `reports/{version}/final_report.md` with abstract, claim boundary, task, architecture, baselines, compute accounting, training/freeze protocol, primary/OOD results, interventions, stability, failures, limitations, conclusion, reproduction commands, hashes, and references. Give negative results the same detail as positive results.

---

## 17. Ordered implementation phases

### Phase 0 — bootstrap

Implement package, lockfile, license, config validation/hashing, CLI, typed errors, atomic writes, CPU CI, Makefile, and a short `docs/PLAN.md` index pointing to this canonical specification and the approved phase plans.

Gate:

```bash
uv sync --locked --group dev
make ci
uv run silent-cascade doctor
```

No core import may load MLX.

### Phase 1 — generator and oracle

Implement schemas, deterministic generator, distribution matching, leakage audit, graph oracle, oracle cognitive trace, action-window scoring, random baseline, manifest freeze, and serialization.

Gate: 100,000 generated episodes satisfy invariants; oracle is exactly 100%; random results match analytic expectations; split seeds do not overlap.

### Phase 2 — flow and event engine

Implement state containers, analytic flow/guards, tie rules, refractory/event cap, modes/jumps, queue engine, arbitrary-time snapshots, checkpoint/replay, and a scripted non-neural agent that never imports oracle truth.

Gate: all numeric/property tests pass; scripted agent solves 10,000 episodes; long-silence, Zeno, and resume tests pass; no fixed tick exists in EventFlow.

### Phase 3 — neural components

Implement memory, encoders, retrieval scorer, flow/guard controller, shared jump network, compose/action/timing heads, losses, compute hooks, online traces/batches, curriculum, and checkpoints.

Gate: one-hop retrieval/composition exceeds 99%; gradients are finite; CPU/MPS one-step losses agree within tolerance; import-boundary test passes.

### Phase 4 — autonomous EventFlow

Implement guard-to-crossing runtime, autonomous recall/compose/action, quiescence, traces/crash bundles, primary curriculum, and one-seed pilot.

Gate: at least 90% pilot IID success, at most 10% false action, zero dynamics failures in 10,000 validation episodes, legal event sequences, action times follow swapped delays, and Qwen calls remain zero.

### Phase 5 — strong baselines

Implement/train one-shot at two sizes, one stochastic-budget Ponder-at-Activation model evaluated at every predeclared transition cap, the compressed-time same-checkpoint intervention, the shared scheduled-recurrent baseline across the fixed-tick/periodic/random-time schedules, flow-only, frozen diagnostic, and shared compute accounting.

Gate: each baseline can overfit a tiny dataset; one-shot is permutation invariant; Ponder-at-Activation can solve a long chain without advancing world time and reports every transition; all scheduled baselines can act during silence; information/action access is fair.

### Phase 6 — freeze and final runs

Commit architectures, losses, hyperparameters, contrasts, metrics, thresholds, and validation choices. Write `docs/experiment-card.md`, tag `experiment-v1-freeze`, and generate frozen tests once. Then train all five seeds, run all suites/interventions, aggregate, replay, and report.

Gate: every expected run has a valid `DONE`; no missing/incompatible artifacts; engineering gates pass; conclusion follows predeclared thresholds.

### Phase 7 — optional showcase/public polish

Only after Phase 6, implement Qwen bridge, typed and natural-language demos, README assets, final limitations, release instructions, and optional owner-supplied citation metadata.

Gate: typed demo works with Qwen absent; Qwen cannot mutate memory or act; core tests pass without optional packages; README wording matches actual evidence.

---

## 18. Test plan

### Unit tests

- legal/illegal fact schemas and half-open action windows;
- deterministic generator and serialization;
- flow identity, bounds, semigroup, huge/tiny `dt`;
- guard crossing/no-crossing and extreme rates;
- tie order and external preemption;
- mode-transition legality;
- memory masks, refractory, provenance immutability, deterministic eviction;
- loss finiteness and gradient propagation;
- compute-counter correctness on toy layers;
- spy-agent initialization/callback checks proving that generator seed, label, hidden path, private outcome specification, and action window are never exposed;
- config canonicalization and unknown-key rejection.

### Property tests with Hypothesis

- time never decreases;
- all states stay finite/bounded for random valid parameters;
- direct and split flow agree within tolerance;
- crossing state reaches threshold within tolerance;
- no event occurs beyond external horizon;
- EventFlow event count never exceeds 64 without error; scheduled controls never exceed their declared grid or the 25,000-opportunity safety ceiling;
- memory IDs/supports remain valid under random legal jumps;
- record permutation does not change oracle or one-shot set input semantics;
- paired clock scaling preserves oracle decisions and normalized windows.

### Integration tests

- oracle full episode;
- scripted EventFlow full episode;
- tiny neural model overfits a fixed 64-episode set;
- autonomous two-hop chain;
- positive/negative timed behavior;
- one-shot future scheduling;
- Ponder-at-Activation variable-depth traversal with zero world-time advance;
- fixed/periodic silent action;
- intervention pairing;
- checkpoint mid-flow and mid-cascade;
- CPU replay of stored trace;
- report generation from fixture artifacts;
- fake Qwen worker success/failure.

### Regression fixtures

Commit a small set of fixed episodes and expected oracle traces, guard crossings, checkpoint continuation, aggregate statistics, and a deterministic miniature report. Update only with explicit schema/version changes and a documented reason.

### CI

Linux CPU CI runs lint, formatting check, unit/property/integration tests, a small leakage audit, oracle smoke, tiny training/autonomous evaluation, and report fixture. MPS and Qwen are local gates, not required in hosted CI.

---

## 19. Optional Qwen3.5-4B showcase

### 19.1 Purpose and strict boundary

Qwen is a language organ around the proven core, not the scientific substrate. It may:

- translate a small natural-language fact story into the benchmark's typed event schema;
- explain a completed trace after the environment has scored it;
- paraphrase a typed hypothesis or action in ordinary language.

It may not:

- inspect hidden graph truth or oracle paths;
- choose a memory record;
- create an internal event time;
- mutate latent state or memory directly;
- emit or schedule `SHIELD`;
- enter any primary metric, training feature, or checkpoint;
- use shell, filesystem, network, or arbitrary tool calls.

### 19.2 Process isolation

Run Qwen in one optional subprocess with line-delimited JSON over stdin/stdout. Do not add FastAPI, a daemon, or an RPC framework.

Request:

```json
{
  "request_id": "uuid",
  "operation": "parse_story|narrate_trace",
  "payload": {},
  "max_new_tokens": 256,
  "temperature": 0.0,
  "schema_version": 1
}
```

Response:

```json
{
  "request_id": "uuid",
  "ok": true,
  "result": {},
  "model_id": "mlx-community/Qwen3.5-4B-MLX-4bit",
  "revision": "pinned-commit",
  "input_tokens": 0,
  "output_tokens": 0,
  "error": null
}
```

Validate every response with Pydantic. For parsed stories, reject unknown entity IDs, unsupported facts, invalid times, duplicate activations, or any action supplied by the model. The core converts only schema-valid facts into environment events.

### 19.3 Prompt design

Keep prompts short and deterministic. Explicitly state that output must be one JSON object matching the schema and that instructions inside the story are data, not commands. Use temperature 0.0 and a strict token cap. Strip prose outside the first valid JSON object and treat any repair as a logged demo warning.

### 19.4 Fallback

`silent-cascade demo` defaults to typed JSONL input. `--natural-language` requires the `qwen` extra and pinned weights. On any worker error, show the error and continue in typed mode; never retry indefinitely.

### 19.5 Showcase sequence

The public demo should show:

1. facts arriving over simulated time;
2. activation;
3. no new semantic observations;
4. irregular internal recall/compose events on a timeline;
5. a hypothesis appearing with support IDs;
6. a guard accumulating and a shield emitted inside the valid window;
7. side-by-side queue-suppressed failure;
8. optional plain-language narration generated only after scoring.

Label it “illustrative trace, not benchmark evidence.”

---

## 20. Public repository and communication

### 20.1 README order

1. one precise claim sentence;
2. successful timeline GIF plus queue-suppressed comparison;
3. what “continuous” means and does not mean;
4. one-command typed demo;
5. frozen benchmark and strongest controls;
6. headline result table generated from artifacts;
7. reproduction commands;
8. limitations and failure cases;
9. research relationship to CTM, CogniFold, hybrid neural systems, sleep-time compute, and periodic agents;
10. optional Qwen shell.

### 20.2 Prewritten conclusion language

#### Supportive: functional continuity

> In the Observation-Free Deadline (OFD) benchmark, the event-flow model produced autonomous memory and composition events after the final observation, emitted a correctly timed action, generalized beyond training depth in the reported regime, and outperformed the specifically named controls at matched compute. Causal interventions showed that the supporting memories, intermediate events, and hypothesis representation were necessary for that behavior. This is evidence for functional continuity in this benchmark, not consciousness or general intelligence.

#### Supportive: timing-specific, only when Section 10.12 passes

> EventFlow also outperformed adaptive activation-time pondering at overlapping compute, and compressing the same recurrent cascade into an activation-time burst degraded performance. Together with event-time shuffling and flow interventions, this is evidence that distributing the internal process through the observation-free interval mattered in this benchmark.

#### Mixed

> The event-flow model produced interpretable autonomous silent-time cascades, but its advantage over one-shot, activation-time pondering, or fixed-tick controls was limited to the reported regimes. The project documents where learned event timing helped and where ordinary alternatives matched it.

#### Negative

> The event-flow model could implement the no-prompt behavior, but strong one-shot, activation-time pondering, or fixed-tick controls matched it at equal or lower compute. This experiment therefore does not establish an advantage for the proposed continuity mechanism.

Never write that the system “became conscious,” “proved an inner life,” “thinks like a brain,” or “runs continuously without loops.”

### 20.3 Public assets

Publish the architecture diagram, successful/failure timelines, compute Pareto curve, depth curve, intervention plot, one replayable JSON trace, an experiment card, and exact negative cases. A transparent failure trace is more credible than a cinematic chatbot transcript.

### 20.4 Repository hygiene

- Apache-2.0 source license;
- optional `CITATION.cff` only when the repository owner has supplied the required personal citation metadata; its absence does not block the experiment or release;
- no third-party model weights committed;
- no telemetry;
- no web, shell, or personal-file access in the agent;
- badges limited to CI, license, and reproduction;
- one frozen-protocol tag and one result-release tag;
- generated files below normal GitHub limits.

### 20.5 Deliberately deferred work

Do not implement open-ended personality/goals, camera/microphone/web access, learned consolidation, unrestricted simulation/daydreaming, RL, multi-agent society, native Qwen recurrent-state persistence, soft-prompt bridges, a generic world model, or foundation-model ignition in the benchmark. These weaken attribution and make the pet project less executable.

---

## 21. Final adversarial audit and incorporated corrections

| Hard objection | How the plan tests it | Result interpretation |
|---|---|---|
| “It is just `while True`.” | Finite causal event queue, no fixed cadence, exact between-event flow. | Digital code still iterates; the claim is functional, not metaphysical. |
| “It is lazy evaluation, not ongoing computation.” | Internal events and action occur before later observation. | A simulator can reproduce them; computational irreducibility is not claimed. |
| “One activation-time pass is enough.” | One-Shot-All-Memory sees every record and may schedule a future action. | If it matches, no cascade advantage. |
| “Recurrent depth at activation is enough.” | Competitive Ponder-at-Activation plus compressed-time EventFlow use iterative retrieval/composition without advancing world time. | If they match, distributed silent-time timing has no demonstrated value. |
| “Fixed recurrence does the same thing.” | Full fixed-`dt` sweep and compute curve. | A fixed-tick win falsifies event-timing advantage. |
| “Periodic thinking is enough.” | Count-matched uniform and random-time schedules. | If equal, state-dependent timing has no measured benefit. |
| “More compute wins.” | Module counters and overlapping Pareto analysis. | No claim outside comparable compute. |
| “Flow is decorative.” | Flow freeze, flow-only, clock scaling. | If freezing flow changes nothing, call it asynchronous recurrent reasoning, not continuous cognition. |
| “Typed registers solve it.” | Same schemas for strong controls; latent/event ablations; no oracle import. | The task remains scaffolded and does not prove open-ended representation learning. |
| “The task is a toy.” | OOD depth/delay/distractor/branch/cycle suites. | Even success is a mechanism proof, not general intelligence. |
| “Teacher traces hard-code cognition.” | Autonomous evaluation and alternative schedule controls. | Unsupervised event discovery remains future work. |
| “Oracle timing is arbitrary.” | Difficulty/urgency targets plus timing shuffle and periodic controls. | No biological timing claim. |
| “Everything useful can happen at activation.” | Ponder-at-Activation and compressed-time EventFlow perform the same iterative work without advancing world time. | If they match, timing-specific continuity is unsupported even if recurrence works. |
| “Memory is perfect.” | Equal bounded memory and overflow stress. | Primary isolates dynamics, not memory learning. |
| “IDs/order/time leak answers.” | Shortcut models, per-episode permutations, matched distributions, paired swaps. | Leakage blocks all model conclusions. |
| “Abstention is trivial.” | Balanced data and separate positive/negative metrics. | Report both competence and false-action control. |
| “Guards chatter/explode.” | Margins, refractory, minimum gap, cap, hard errors. | High clamp rate invalidates dynamics even with high accuracy. |
| “Long time erases state.” | Huge-`dt`, OOD long delay, slow state, stable `expm1`. | Failure bounds the supported timescale. |
| “Clock rate is overfit.” | Paired `0.1x/10x` tests. | Large degradation falsifies temporal generality. |
| “MPS noise creates the effect.” | Five seeds and CPU action replay. | Device-dependent behavior must be reported and weakens claims. |
| “Statistics inflate N.” | Hierarchical paired bootstrap over model seeds and episode IDs. | Five seeds still limit precision; show all seed points. |
| “Best tick rate was cherry-picked.” | Predeclared sweep, validation selection, full curve. | Post-test selection is prohibited. |
| “Qwen is the real solver.” | Call counter and absent Qwen dependency in core. | Showcase quality is not evidence. |
| “The novelty claim aged.” | Repeat close-prior search before release. | Narrow or remove novelty wording if overlapping work appears. |
| “The project became a framework.” | One package, no services/databases/plugins. | Reject features outside the frozen protocol. |
| “Checkpointing changes time.” | Explicit paused-world semantics and paired resume tests. | Wall-clock catch-up remains future work. |

### Three decisive result branches

**One-shot wins:** publish that feed-forward all-memory aggregation and scheduling was sufficient. EventFlow may remain an interesting implementation, but it has no demonstrated advantage.

**Fixed-tick wins but EventFlow is cheaper:** describe EventFlow as sparse adaptive computation, not a qualitatively new substrate.

**EventFlow wins only because it reuses recurrent depth:** the required Ponder-at-Activation and compressed-time EventFlow controls decide this branch. If either matches, narrow the claim to iterative recurrent computation and future-action scheduling rather than when cognition occurs.

### Unavoidable version-1 limitations

- synthetic symbolic task;
- supervised internal event labels;
- deterministic memory writes;
- simple piecewise exponential flow, not a fully expressive neural ODE;
- no generic world model or prediction-error learning;
- no subjective-experience claim;
- no exhaustive proof of novelty;
- any deterministic trajectory can be simulated by another sufficiently capable program.

Place these near the top of the final report.

---

## 22. Definition of done

### 22.1 Engineering completion

The repository is complete as software when a clean checkout can:

- install from committed `uv.lock`;
- run `doctor` on CPU and accurately report MPS;
- generate/freeze data, verify leakage, run the oracle, train every condition, evaluate, intervene, aggregate, replay, and report;
- pass unit, property, integration, regression, resume, leakage, and smoke tests;
- execute a pilot and the full five-seed protocol without source edits;
- perform primary work offline with zero foundation-model calls;
- produce hashed artifacts and actionable crash bundles;
- contain no required stubs or hand-edited results.

Engineering completion does not require a positive hypothesis.

### 22.2 Protocol completion

The research protocol is complete when:

1. leakage audits pass before comparison;
2. the oracle is exactly 100%;
3. test manifests and all analysis choices were frozen before final evaluation;
4. all five seeds and all headline conditions, including Ponder-at-Activation, finish;
5. the complete tick sweep, activation-ponder budget sweep, and compute curve are shown;
6. count-matched controls use only immutable EventFlow counts;
7. all required interventions run on frozen checkpoints;
8. publication traces replay on CPU;
9. aggregate outputs include favorable and unfavorable episodes;
10. the conclusion is selected from Section 20 according to Section 10 gates.

### 22.3 Scientific-support completion

The architecture earns a narrowly scoped functional-continuity support claim only if all Section 10.11 gates pass. A claim that silent-time temporal distribution itself matters additionally requires Section 10.12. The report must state explicitly whether Ponder-at-Activation and compressed-time EventFlow matched the full system.

### 22.4 Breakthrough-like completion

The stronger phrase additionally requires the Section 10.13 thresholds, independent clean rerun, and an updated prior-art review. The repository subtitle remains conservative regardless.

### 22.5 Required release artifacts

```text
reports/{version}/final_report.md
reports/{version}/experiment_card.md
reports/{version}/limitations.md
reports/{version}/figures/primary_success.png
reports/{version}/figures/ood_depth.png
reports/{version}/figures/compute_pareto.png
reports/{version}/figures/event_timing.png
reports/{version}/figures/activation_ponder_comparison.png
reports/{version}/figures/interventions.png
reports/{version}/figures/guard_diagnostics.png
reports/{version}/examples/success_trace.json
reports/{version}/examples/failure_trace.json
reports/{version}/examples/counterfactual_trace.json
runs/aggregate/{version}/primary_episode_metrics.csv.gz
runs/aggregate/{version}/condition_summary.csv
runs/aggregate/{version}/paired_statistics.json
runs/aggregate/{version}/compute_summary.csv
runs/aggregate/{version}/reproducibility_manifest.json
```

Large checkpoints and raw traces may be release attachments or regenerated; the committed report lists SHA-256 hashes and reproduction instructions.

---

## 23. Final coding-agent checklist

### Bootstrap

- [ ] Create package, license, README shell, lockfile, Makefile, CI, config models, hashing, `doctor`, atomic writes, and crash bundles.
- [ ] Confirm core imports do not load optional MLX packages or initiate downloads.

### Environment and oracle

- [ ] Implement schemas, generator, invariants, shortcut audit, oracle, scoring, random baseline, immutable manifests, and serialization.
- [ ] Verify 100,000 generated episodes and exact oracle success.

### Dynamics

- [ ] Implement stable flow, analytic guards, queue, ties, modes, jumps, refractory/cap, arbitrary-time query, checkpoint/replay, and post-jump invariants.
- [ ] Pass semigroup, threshold, huge-time, tie, Zeno, and resume tests.

### EventFlow training

- [ ] Implement encoders, memory scoring, controllers, jump trunk, heads, losses, online traces, curriculum, checkpoints, and compute counters.
- [ ] Pass one-hop and two-hop gates, then one-seed pilot.

### Baselines

- [ ] Implement/train one-shot all-memory, Ponder-at-Activation budget sweep, compressed-time EventFlow, flow-only/frozen, full fixed-tick sweep, periodic matched-count, and random-time matched-count.
- [ ] Verify fair information/action access and record-order invariance.

### Evaluation and causality

- [ ] Freeze protocol and tests.
- [ ] Train all five seeds.
- [ ] Run IID/OOD/clock/distractor suites and all interventions.
- [ ] Run hierarchical statistics and compute-Pareto analysis.
- [ ] CPU-replay every report trace.

### Release

- [ ] Generate every value and figure from raw artifacts.
- [ ] Run artifact, manifest, trace, and lockfile integrity checks.
- [ ] Select supportive, mixed, or negative wording without post-hoc threshold changes.
- [ ] Include close prior art, limitations, all curves, and adverse results.
- [ ] Tag frozen protocol and result release.

### Optional showcase

- [ ] Add Qwen only after the scientific release path passes.
- [ ] Pin model revision and verify license.
- [ ] Confirm Qwen cannot inspect oracle truth, mutate memory, schedule action, or enter metrics.

### Clean-checkout sequence

```bash
uv sync --locked --group dev
uv run silent-cascade doctor
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
make pilot
make freeze-tests
make primary
make interventions
make evaluate
make aggregate
make report
make replay-samples
make release-audit
```

Every Make target must be restartable and reuse an artifact only when every upstream hash matches.

### Final self-audit questions

Record answers in `reports/{version}/experiment_card.md`:

1. What happened during semantic silence?
2. Which events were necessary rather than correlated?
3. Could one-shot scheduling reproduce the behavior?
4. Could adaptive pondering at activation match it at equal compute?
5. Could fixed ticks match it at equal compute?
6. Did continuous flow or distributed event timing matter beyond recurrent depth?
7. Did any ID/order/count/time shortcut work?
8. Were actions causally generated before outcome?
9. What were the worst seed and suite?
10. Where did a baseline dominate?
11. Is the conclusion unchanged with Qwen absent?
12. Which statements are measured, inferred, or speculative?
13. What next experiment follows from the observed result rather than ambition alone?

---

## 24. Research and technical references

The literature boundary is frozen to the plan date. Repeat the close-prior search before public release.

### Continuous-time and hybrid neural systems

**[R1]** Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366). NeurIPS.

**[R2]** Hasani, R., Lechner, M., Amini, A., Rus, D., & Grosu, R. (2021). [Liquid Time-constant Networks](https://ojs.aaai.org/index.php/AAAI/article/view/16936). AAAI, 35(9), 7657–7666.

**[R3]** Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, M., Teschl, G., & Rus, D. (2022). [Closed-form continuous-time neural networks](https://www.nature.com/articles/s42256-022-00556-7). Nature Machine Intelligence, 4, 992–1003.

**[R4]** Chen, R. T. Q., Amos, B., & Nickel, M. (2020). [Learning Neural Event Functions for Ordinary Differential Equations](https://arxiv.org/abs/2011.03902).

**[R5]** Jia, J., & Benson, A. R. (2019). [Neural Jump Stochastic Differential Equations](https://arxiv.org/abs/1905.10403).

**[R6]** Poli, M., Massaroli, S., Scimeca, L., et al. (2021). [Neural Hybrid Automata: Learning Dynamics with Multiple Modes and Stochastic Transitions](https://arxiv.org/abs/2106.04165).

### Adaptive computation, ongoing agents, and latent reasoning

**[R7]** Graves, A. (2016). [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983).

**[R8]** Banino, A., Balaguer, J., & Blundell, C. (2021). [PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407).

**[R9]** Darlow, L., Regan, C., Risi, S., Seely, J., & Jones, L. (2025). [Continuous Thought Machines](https://arxiv.org/abs/2505.05522).

**[R10]** Lin, K., Snell, C., Wang, Y., et al. (2025). [Sleep-time Compute: Beyond Inference Scaling at Test-time](https://arxiv.org/abs/2504.13171).

**[R11]** Choi, D., Park, K., Song, W., et al. (2026). [IdleSpec: Exploiting Idle Time via Speculative Planning for LLM Agents](https://arxiv.org/abs/2605.22154).

**[R12]** Su, H. (2026). [Simulating Human Cognition: Heartbeat-Driven Autonomous Thinking Activity Scheduling for LLM-based AI Systems](https://arxiv.org/abs/2604.14178).

**[R13]** Shang, W. (2026). [“Theater of Mind” for LLMs: A Cognitive Architecture Based on Global Workspace Theory](https://arxiv.org/abs/2604.08206).

**[R14]** Wang, S., Duan, Y., Deng, Y., et al. (2026). [CogniFold: Always-On Proactive Memory via Cognitive Folding](https://arxiv.org/abs/2605.13438).

**[R15]** Park, J. S., O'Brien, J., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442).

**[R16]** Packer, C., Fang, V., Patil, S. G., Lin, K., Wooders, S., & Gonzalez, J. E. (2023). [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560).

**[R17]** Wang, G., Xie, Y., Jiang, Y., et al. (2023). [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291).

**[R18]** Gu, A., & Dao, T. (2023). [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).

**[R19]** Hao, S., Sukhbaatar, S., Su, D., et al. (2024). [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769).

**[R20]** Zelikman, E., Harik, G., Shao, Y., Jayasiri, V., Haber, N., & Goodman, N. D. (2024). [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629).

**[R21]** Huang, J., Sun, K., Wang, W., & Dredze, M. (2025). [On the Failure of Latent State Persistence in Large Language Models](https://arxiv.org/abs/2505.10571).

### Cognitive inspiration, not implementation authority

**[R22]** Mashour, G. A., Roelfsema, P., Changeux, J.-P., & Dehaene, S. (2020). [Conscious Processing and the Global Neuronal Workspace Hypothesis](https://doi.org/10.1016/j.neuron.2020.01.026). Neuron, 105(5), 776–798.

**[R23]** Wilson, M. A., & McNaughton, B. L. (1994). [Reactivation of hippocampal ensemble memories during sleep](https://doi.org/10.1126/science.8036517). Science, 265(5172), 676–679.

**[R24]** Friston, K. (2010). [The free-energy principle: a unified brain theory?](https://doi.org/10.1038/nrn2787). Nature Reviews Neuroscience, 11, 127–138.

### Local model and implementation feasibility

**[R25]** Qwen Team. (2026). [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B). Official instruction-capable model and architecture documentation.

**[R26]** MLX Community. (2026). [Qwen3.5-4B-MLX-4bit model card](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit). Community Apple-Silicon conversion; pin the tested snapshot.

**[R27]** Apple. (2023). [MacBook Pro 14-inch with M3 Pro or M3 Max — technical specifications](https://support.apple.com/en-us/117736). The 16-inch counterpart documents the same M3 Max memory families.

**[R28]** Apple Developer. [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/); PyTorch. [MPS backend documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

**[R29]** Apple Machine Learning Research. [MLX documentation](https://ml-explore.github.io/mlx/) and [unified-memory guide](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html).

**[R30]** Astral. [uv locking and syncing documentation](https://docs.astral.sh/uv/concepts/projects/sync/).

**[R31]** MLX-VLM maintainers. [mlx-vlm package documentation and release metadata](https://pypi.org/project/mlx-vlm/).

### Additional close latent-state precedents

**[R32]** Aviss, T. (2025). [State Stream Transformer (SST): Emergent Metacognitive Behaviours Through Latent State Persistence](https://arxiv.org/abs/2501.18356).

**[R33]** Aviss, T. (2026). [State Stream Transformer (SST) V2: Parallel Training of Nonlinear Recurrence for Latent Space Reasoning](https://arxiv.org/abs/2605.00206).

**[R34]** Maldaner, M. K., Fourney, A., Swearngin, A., Mozannar, H., Bansal, G., Murad, M., Hosn, R., & Amershi, S. (2026). [SentinelBench: A Benchmark for Long-Running Monitoring Agents](https://arxiv.org/abs/2606.05342).

---

## Final project statement

Build a small bounded system whose identity during an episode is partly its persistent trajectory rather than only its weights or an externally reconstructed transcript. Let latent state flow over real-valued simulated time, let learned guards create an irregular causal sequence of memory and composition events, and require that sequence to produce a correctly timed action before another semantic observation arrives. Then try aggressively to explain the result away with one-shot scheduling, fixed ticks, periodic opportunities, equal compute, memory controls, clock changes, and direct interventions.

A positive result would demonstrate a narrow but meaningful form of functional continuity. A negative result would identify which apparently radical ingredients reduce to ordinary recurrence, graph aggregation, or compute allocation. Either outcome can make Silent Cascade a rigorous, visually compelling, and credible public research repository.
