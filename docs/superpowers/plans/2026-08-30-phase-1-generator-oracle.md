# Silent Cascade Phase 1 Generator and Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete offline Observation-Free Deadline data layer:
strict public/private schemas, deterministic matched-validation and
independent claim-data generation, independent invariant and oracle analysis,
timed oracle traces, exact scoring, a diagnostic random baseline, leakage
audits, immutable validation manifests, and the four Phase 1 CLI surfaces.

**Architecture:** Keep agent-visible episode data structurally separate from
environment-private truth. Use one explicitly matched recipe for validation
diagnostics and a second episode-local recipe for acceptance and future frozen
claim data. Generate both from domain-separated counter seeds, validate them
with an implementation independent from the graph oracle, and store only
versioned recipes plus hashes in immutable manifests. Expose all work through
thin Typer adapters over testable services; no event engine, model, memory
runtime, or final frozen-test data enters this phase.

**Tech Stack:** Python 3.12, NumPy PCG64DXSM, SciPy, strict Pydantic v2,
frozen dataclasses, Typer, PyYAML, SHA-256/HMAC-SHA-256, pytest, Hypothesis,
Ruff, and `uv`.

**Spec:** `docs/superpowers/specs/2026-08-30-silent-cascade-design.md`

**Plan status:** **Approved and in execution.** The final whole-phase audits
invalidated the previous Task 18 artifacts and require the five corrective
implementation passes in this plan before Task 18 is rerun. The canonical
design and this Phase 1 plan are approved. The OFD task, distributions,
scientific settings, denominators, controls, thresholds, frozen seeds, resource
ceilings, and zero-foundation-model-call rule are unchanged. The corrections do
change pre-completion generator construction/rejection semantics, private
coordinate/schema fields, and evidence schemas; every earlier artifact is
therefore invalid and has no compatibility path.

## Execution Preconditions and Superpowers Workflow

Before changing source, tests, configuration, manifests, documentation, or
build files:

1. Confirm explicit approval of this exact Phase 1 plan in the task thread.
2. Confirm this plan and `docs/PLAN.md` are committed and `git status --short`
   is empty. Record `git rev-parse HEAD` as `plan_base_revision`; every Phase 1
   evidence report records it. The plan author commits these documents before
   handoff, so execution must not create an untracked planning precondition.
3. Work on the current branch, as required by `AGENTS.md`, and make the
   task-level commits named below.
4. Invoke `superpowers:subagent-driven-development` or
   `superpowers:executing-plans`.
5. Invoke `superpowers:test-driven-development` for every implementation task
   (Tasks 1–17) and preserve the stated RED/GREEN sequence. Task 18 is an
   evidence-only acceptance run: it changes no implementation and therefore
   verifies frozen gates rather than introducing behavior through TDD.
6. On any failure or unexpected behavior, invoke
   `superpowers:systematic-debugging` before changing the implementation.
7. Request and receive code review at every task boundary.
8. Invoke `superpowers:verification-before-completion` before every completion
   claim and `superpowers:finishing-a-development-branch` after the final gate.

Repository standing approval covers safe, reversible implementation rulings.
Do not ask routine permission questions when repository evidence identifies the
best answer.

## Global Constraints

- Distribution name: `silent-cascade`; package: `silent_cascade`; executable:
  `silent-cascade`; Python: `>=3.12,<3.13`.
- The primary path remains offline and records exactly zero foundation-model
  calls. No Phase 1 command downloads a model or accesses the network.
- CI/CD remains disabled. Run every lint, test, doctor, gate, and package build
  locally; `make verify` remains the complete routine quality gate.
- Use `generator_version = "ofd-v1"` and episode/manifest
  `schema_version = 1`. This retention is permitted only because Phase 1 has
  not completed or released: the correction is a pre-completion replacement,
  every artifact made by the earlier construction is invalid, and there is no
  compatibility/migration promise. Evidence reports use the exact versioned
  literals declared by their owning tasks: Phase 1 reproducibility and oracle
  evaluation remain v2; leakage is v3; and final gate verification is v3 after
  Correction Pass 5. Any post-completion change to generator construction,
  rejection, private coordinates, or episode/manifest schema semantics requires
  a new version and a deviations entry.
- Use `max_entities=64`, four hazard types, primary memory capacity `64`, an
  EventFlow event ceiling of `64`, and a scheduled-opportunity ceiling of
  `25,000` in the primary data contract.
- Primary variants are exactly 50% positive, 25% safe-negative, and 25%
  disconnected-negative in validation and gate corpora.
- Every primary episode contains exactly two `HAZARD` records, one `SAFE`
  record, and a requested-length activation path.
- Store timestamps as finite Python float64 values. Reject booleans wherever an
  exact integer is required. Reject NaN, infinity, negative delays, time
  reversal, duplicate IDs, and unsupported schema versions.
- Public episode data may contain only an opaque public ID, `AgentInit`,
  chronological `FACT` events, and one final `ACTIVATE` event.
- Public data, default inspection, agent-facing errors, and online logs must not
  contain generator roots, episode indices, retry counts, variants, labels,
  hidden paths, reachability, terminal time, action window, scorer truth, or
  oracle record choices. Validation manifests are explicitly environment-private
  recipe artifacts: they may bind an opaque public ID to deterministic
  regeneration coordinates, but the engine must never pass a manifest or those
  coordinates to an agent. Aggregate CLI/report output may record roots,
  namespaces, and variant counts, but never per-ID labels, paths, or selections.
- Assign fact event/record IDs only after the independently sampled presentation
  permutation. Construction-order IDs are forbidden.
- `OUTCOME` and `END` are private terminal events. They are stored only in the
  private bundle and never exposed through the public episode.
- Generator invariants and the oracle must implement reachability separately.
  `env.generator` and `env.invariants` may not import `env.oracle`; `env.oracle`
  may not import `env.generator` or `env.invariants`.
- Final files under `manifests/frozen/` remain Phase 6-only. Phase 1 creates a
  fixed validation manifest and tests frozen-access refusal using temporary
  fixtures.
- No Hydra, plugin framework, database, service, telemetry, generic agent
  framework, neural model, event engine, checkpoint system, replay command, or
  Qwen integration enters Phase 1.
- The Phase 1 100,000-episode gate uses the same independent-nuisance recipe
  that Phase 6 may instantiate for frozen tests. Phase 6 may choose new roots
  and exact suite sizes, but may not introduce a new generator algorithm.
- A Phase 1 100,000-episode result means “generated with the independent recipe
  and independently validated.” It is not the later autonomous-dynamics
  engineering gate and is not benchmark evidence.

## Frozen Phase 1 Design Rulings

### Public/private separation

Use three sibling representations:

1. `PublicEpisode`: only agent-visible initialization and observations.
2. `EpisodeTruth`: variant, generator recipe, hidden path, private terminal,
   delay/window, rejection diagnostics, and scorer truth.
3. `EpisodeBundle`: a private wrapper containing the public episode and truth.

Do not put private fields on `PublicEpisode`, even with underscores, optional
values, or serialization exclusions. Oracle traces are separate artifacts and
never embedded in evaluation manifest entries.

### Matched validation and independent claim-data generation

The fixed Phase 1 validation manifest is generated in four-episode matched
cohorts. A domain-separated label stream permutes this fixed cohort composition:

```text
positive, positive, safe_negative, disconnected_negative
```

All four members share one canonical unlabeled LINK topology, requested path
length, episode delay, distractor count, observation-gap template, and
activation gap. Each member applies an independent bijective node relabeling,
hazard-record assignment, fact permutation, and public ID. This makes the
LINK-only component/degree signature equal by construction rather than by
retry, gives exact nuisance matching, and preserves independent presentation.

The Phase 1 acceptance gate and all future Phase 6 frozen test manifests use
`generate_independent_episode`. Variant counts are fixed by allocation quartets
containing `[positive, positive, safe_negative, disconnected_negative]` in a
domain-separated permutation, but every episode independently samples its
LINK topology, delay, distractor count, observation gaps, activation gap,
terminal classes/placement, node relabeling, and presentation from episode-local
streams. A quartet is a fixed-design label-allocation block only; it shares no
nuisance draw or RNG token. This preserves exact aggregate strata while making
the episode ID the inferential unit.

Online training generation is deferred to Phase 3. Rejection retries rebuild
the complete matched cohort for validation, but retry only the rejected episode
for independent generation. Phase 6 may instantiate the already-tested
independent recipe with frozen roots and final suite sizes; it may not first
introduce a new generator. Reusing matched validation cohorts in frozen tests
is forbidden unless a versioned specification change replaces every episode
bootstrap and paired subset analysis with cohort-cluster inference.

### Terminal layout and delay law

For every primary episode:

| Variant | Activation-path terminal | Unreachable terminals |
|---|---|---|
| positive | one `HAZARD` | one `HAZARD`, one `SAFE` |
| safe-negative | one `SAFE` | two `HAZARD` |
| disconnected-negative | none | two `HAZARD`, one `SAFE` |

Both hazard records carry the same episode delay. In matched validation, their
hazard classes are sampled once as a two-item cohort multiset from `0..3` and
every member independently permutes the assignments. In independent mode, the
two-item multiset and its assignment are sampled only from that episode's
terminal stream.
The relevant positive class is whichever class is placed at the
activation-path terminal. Thus terminal counts and the complete hazard
class/delay multiset match exactly, while record/class assignment remains
independently permuted. The public delay is usable by the diagnostic random
baseline without revealing reachability.

### Counter seeds and public IDs

Semantic RNG seeds use a typed key. `member_index=-1` is required for the
cohort-level `label`, `template`, `structure`, and `timestamps` streams.
`node_permutation`, `terminals`, `presentation`, `trace_jitter`, and
`random_baseline` require a member index in `0..3`:

```python
class SeedStream(str, Enum):
    LABEL = "label"
    TEMPLATE = "template"
    STRUCTURE = "structure"
    NODE_PERMUTATION = "node_permutation"
    TERMINALS = "terminals"
    PRESENTATION = "presentation"
    TIMESTAMPS = "timestamps"
    TRACE_JITTER = "trace_jitter"
    RANDOM_BASELINE = "random_baseline"


@dataclass(frozen=True, slots=True)
class CounterSeedKey:
    generator_version: Literal["ofd-v1"]
    split_namespace: SplitNamespace
    suite: SuiteName
    root_seed: int                 # 0 <= value < 2**128; bool rejected
    cohort_index: int              # value >= 0; bool rejected
    member_index: int              # scope checked against stream; -1 or 0..3
    stream: SeedStream
    attempt: int                   # 0 <= value < 1000; bool rejected
```

Independent episodes never reuse this cohort key. They use a second typed key
whose topology/timing streams are episode-local:

```python
@dataclass(frozen=True, slots=True)
class IndependentCounterSeedKey:
    generator_version: Literal["ofd-v1"]
    split_namespace: SplitNamespace
    suite: SuiteName
    root_seed: int                 # 0 <= value < 2**128; bool rejected
    episode_index: int             # value >= 0; bool rejected
    stream: SeedStream             # every stream except LABEL
    attempt: int                   # 0 <= value < 1000; bool rejected


@dataclass(frozen=True, slots=True)
class AllocationLabelKey:
    generator_version: Literal["ofd-v1"]
    split_namespace: SplitNamespace
    suite: SuiteName
    root_seed: int
    requested_path_length: int
    allocation_quartet_index: int
```

Independent allocation identity is explicit and stable even when an episode
block begins at an index not divisible by four. Both
`IndependentEpisodeRequest` and `IndependentEpisodeCoordinate` carry
`quartet_member_index: int` in `0..3`. The assigned variant is exactly
`allocate_independent_variants(AllocationLabelKey(...))[quartet_member_index]`.
No implementation may reconstruct that member with `episode_index % 4`, a
rank, `list.index(variant)`, or an assumed production block boundary.

`AllocationLabelKey` selects only a permutation of `[P,P,S,D]`; no episode
generator consumes it. Each `IndependentCounterSeedKey` serializes the same way
as the cohort key with domain
`silent-cascade/ofd-v1/independent-counter-seed/v1`, `episode_index` replacing
`cohort_index/member_index`, and no `LABEL` stream. Its known-answer vector for
`(phase1_gate, iid_primary, root=41, episode=7, presentation, attempt=0)` is
digest
`b93ed757b3430cc59003146a0afeead8f43d3919db05872185a7b504de1d90bf`
and integer seed `246233469291832394751030201046666242776`.

The serialized payload is:

```python
sha256(canonical_json_bytes({
    "domain": "silent-cascade/ofd-v1/counter-seed/v1",
    "generator_version": generator_version,
    "split_namespace": split_namespace,
    "suite": suite,
    "root_seed": root_seed,
    "cohort_index": cohort_index,
    "member_index": member_index,
    "stream": stream,
    "attempt": attempt,
}))
```

Use the first 16 digest bytes as the unsigned PCG64DXSM seed. Keep the complete
digest as the collision-audit token. Stable stream names are:

```text
label, template, structure, node_permutation, terminals, presentation, timestamps,
trace_jitter, random_baseline
```

No generator path uses Python `hash()`, process-global RNG state, unordered set
iteration, or a sequential cross-episode stream.

The known-answer vector for
`(validation, iid_primary, root=41, cohort=7, member=2,
presentation, attempt=0)` is digest
`14dfc485480dfb97aa21f99f8cb152ffe09e7b183e9a0bffbe424b28df52f8e2`
and integer seed `27746428027079338998295180152501523199`.

Public IDs use a separate 128-bit `public_id_seed`, never `root_seed`. Encode
the HMAC key as
`b"silent-cascade/ofd-v1/public-id-key/v1\0" + seed.to_bytes(16, "big")`.
For counters `0..3`, HMAC the canonical JSON object containing domain
`silent-cascade/ofd-v1/id-pool/v1`, generator version, split, suite, cohort,
attempt, and counter. Set RFC 4122 version-4 and variant bits in the first 16
digest bytes. Sort the four candidates by a second HMAC over the same context,
domain `silent-cascade/ofd-v1/id-assignment/v1`, and the canonical UUID string;
the resulting order maps to member indices `0..3`. Reject pool/global
collisions. For `(seed=91, validation, iid_primary, cohort=7, attempt=0)`, the
assigned IDs are exactly:

```python
@dataclass(frozen=True, slots=True)
class PublicIdBatchKey:
    generator_version: Literal["ofd-v1"]
    split_namespace: SplitNamespace
    suite: SuiteName
    public_id_seed: int
    cohort_index: int
    accepted_attempt: int
```

The exact public API is
`allocate_public_ids(key: PublicIdBatchKey) -> tuple[str, str, str, str]`.
Validate both integer fields as exact non-boolean integers in the same ranges
as their counter-seed equivalents. Allocate only after all four semantic draws
pass generator-local checks. Bind the IDs into final bundles, then run the
independent invariant analyzer; any disagreement is a fatal implementation
error, never a sampling retry. Persist `accepted_attempt` and require it to
equal `EpisodeTruth.rejection_count` during regeneration.

```text
eb78e7f5-c0bb-4b38-88f3-5d9d9024fc32
bd8da50f-9499-4c1f-a23e-09ceeeb68e53
91800f56-56f7-4815-8db1-8e4c7c892aa2
8a4f6ea2-4a78-490b-bf96-d007436f1e5c
```

Store a raw public-ID seed only in environment-private manifests and the final
environment-private Phase 1 acceptance evidence that must rederive IDs.
Ordinary aggregate provenance exposes only
`sha256(b"silent-cascade/ofd-v1/public-id-seed-fingerprint/v1\0" +
public_id_seed.to_bytes(16, "big"))`; for seed `91`, the fingerprint is
`3acaee04f18b5609e8d4bdd6a5ab1ee2e24e8143bb1b33a3de05204cb195d980`.
Leakage tests cover public ID and manifest-order prediction.

The final Phase 1 acceptance artifacts bind the exact raw seeds needed for
independent ID rederivation. Matched validation uses `2026083002`
with fingerprint
`0454fca622eb08379a5d88ecbe0a5ef70f6a15e9ca7d333acecf840f2df38802`;
the independent gate and its clock children use `2026083012` with fingerprint
`f21ac562825bfd96e875eedf90c8ba5ed09883ebe2c18449843b03acebec778a`.
These raw seeds remain environment-side artifact evidence and are never passed
to an agent.

Independent public IDs use the same HMAC key framing but a single typed key and
domain, with no batch assignment:

```python
@dataclass(frozen=True, slots=True)
class IndependentPublicIdKey:
    generator_version: Literal["ofd-v1"]
    split_namespace: SplitNamespace
    suite: SuiteName
    public_id_seed: int
    episode_index: int
    accepted_attempt: int
```

`allocate_independent_public_id` HMACs canonical JSON containing domain
`silent-cascade/ofd-v1/independent-id/v1`, generator version, split, suite,
episode index, and `attempt=accepted_attempt`, then sets the RFC 4122 version-4
and variant bits. For `(seed=91, phase1_gate, iid_primary, episode=7,
attempt=0)`, the ID is
`8d1d0713-5f5e-4c49-a6a1-f3620133d2d7`. Matched and independent public-ID
tokens share the same global collision set during a corpus build.

### Independent correctness checks

`env.invariants` recomputes generator constraints from public facts and private
truth. `env.oracle` separately parses only public facts to derive the path and
terminal. Deliberate duplication is required: a shared reachability helper
would allow one common bug to certify itself.

Mutation tests corrupt facts and truth independently. At least one independent
checker must reject every corruption, and oracle evaluation must fail rather
than choose arbitrarily on duplicate or branching primary traces.

### Random baseline law

The Phase 1 random baseline is a public-view-only diagnostic, not a competitive
learned condition:

- abstain with probability `0.5`;
- otherwise select each hazard type with probability `0.125`;
- when acting, read the common public hazard delay and sample uniformly from
  `[t0 + 0.75d, t0 + 0.90d)`.

Conditional success is `0.125` on positives and `0.5` on negatives. With the
balanced primary mixture, analytic timed success is exactly `0.3125`, but the
pooled success count is not binomial because the two strata have different
probabilities. The 100,000-episode gate therefore checks the 50,000 positives
against `Binomial(50_000, 0.125)` and the 50,000 negatives against
`Binomial(50_000, 0.5)` separately. Each stratum must have a two-sided exact
binomial `p >= 0.001` and an absolute rate error no greater than `0.01`.
The pooled `0.3125` rate is reported as a derived diagnostic only, so errors
cannot cancel across strata. Changing the seed, margins, or test after seeing
the result is prohibited.

### Immutable manifest semantics

A manifest is one canonical JSON envelope containing a strict payload and the
SHA-256 of that payload. The file SHA-256 is computed externally and reported;
the file never embeds its own hash. Publication uses `atomic_create_bytes`.

If a path already exists, load and verify it. Return an unchanged verified
artifact only when its canonical bytes match exactly. Any difference is a hard
`ManifestError` and requires a new experiment version. There is no force or
overwrite flag. Access class is hashed content, so renaming or copying a frozen
manifest cannot enable oracle inspection.

### Acceptance allocation

The Phase 1 gate contains exactly 100,000 fresh independently generated base
episodes, excluding clock transforms:

| Suite | Episodes | Per path length | Allocation quartets |
|---|---:|---|---:|
| IID primary | 24,000 | 8,000 each for 2, 3, 4 | 6,000 |
| OOD depth | 16,000 | 4,000 each for 5, 6, 7, 8 | 4,000 |
| OOD short delay | 24,000 | 8,000 each for 2, 3, 4 | 6,000 |
| OOD long delay | 12,000 | 4,000 each for 2, 3, 4 | 3,000 |
| Distractor flood | 24,000 | 8,000 each for 2, 3, 4 | 6,000 |

Every allocation quartet assigns two positives, one safe-negative, and one
disconnected-negative, but its four episodes share no nuisance draw. Every
independent block contains a positive multiple of four episodes; quartets are
formed from consecutive local offsets `0..3` within that block, regardless of
the block's absolute first episode index. Iteration is lexicographic by the
table's suite order, requested path length, and local episode ordinal. Episode
indices are monotonic within each suite, so chunking cannot change a key. The
first episode index for each path block in the Phase 1 gate is the sum of
earlier same-suite block sizes: IID/short/flood use `0, 8000, 16000`; OOD depth
uses `0, 4000, 8000, 12000`; OOD long uses `0, 4000, 8000`. Those boundaries
belong to the Phase 1 gate allocation, not to generic independent or `FROZEN`
invariant semantics. Validation remains matched and uses cohort starts
`0, 834, 1667`.

Within the 24,000 IID gate episodes, select the first `1668`, `1668`, and
`1664` episodes inside path-length blocks 2, 3, and 4 (5,000 parents) for
`0.1x`. The first `668`, `668`, and `664` of those same blocks (2,000 nested
parents) also receive `10x`. These multiples of four preserve exact variant
strata. The parent is never regenerated under a clock-suite key:
`EpisodeKey` continues to identify the unscaled IID semantic source, while
`EpisodeRecipe.evaluation_suite`, `clock_scale`, parent ID, and parent hash
identify the transformed artifact. Clock children get separate opaque public
IDs. Transforms are paired evidence and excluded from the 100,000 fresh-episode
denominator.

Both modes use explicit requests rather than allowing a generator to randomly
choose a required stratum:

```python
@dataclass(frozen=True, slots=True)
class CohortRequest:
    split_namespace: SplitNamespace
    suite: SuiteName
    root_seed: int
    cohort_index: int
    requested_path_length: int


@dataclass(frozen=True, slots=True)
class IndependentEpisodeRequest:
    split_namespace: SplitNamespace
    suite: SuiteName
    root_seed: int
    episode_index: int
    requested_path_length: int
    variant: EpisodeVariant
    allocation_quartet_index: int
    quartet_member_index: int       # exact 0..3 position in this quartet
```

Exact signatures are
`iter_validation_requests(root_seed: int) -> Iterator[CohortRequest]`,
`iter_phase1_gate_requests(root_seed: int) ->
Iterator[IndependentEpisodeRequest]`,
`generate_matched_cohort(config: Phase1Config, request: CohortRequest,
public_id_seed: int) -> GenerationCohort`, and
`generate_independent_episode(config: Phase1Config, request:
IndependentEpisodeRequest, public_id_seed: int) -> EpisodeBundle`.

The validation iterator emits 834, 833, and 833 cohorts for lengths 2, 3, and
4 respectively, for exactly 10,000 matched episodes. The gate iterator emits
the 100,000 independent requests in the frozen table. For each consecutive
four local offsets inside a suite/path block, `AllocationLabelKey` chooses one
of the 12 unique permutations of `[P,P,S,D]`; the request stores both the exact
`quartet_member_index` and the variant at that position. The generator checks
that identity but never derives or resamples it, and neither value is exposed
publicly. A generic non-aligned DEBUG block and a future 20,000-episode
canonical-total `FROZEN` allocation use these same semantics without adding a
generator algorithm or importing Phase 1 gate block boundaries.
Tasks 6–9 contain both complete algorithms. There is no Phase 1 Bernoulli
training sampler or batching API. Phase 3 consumes oracle traces and may wrap
this frozen independent primitive for online batches without changing its
episode semantics.

### Label-blind trace feasibility

The difficulty/urgency/jitter/clamp/common-scale calculation is one neutral,
pure timing primitive in `env.timing`; it accepts only activation time, delay,
the ordered competitive-record counts, an already drawn jitter vector, and
`OracleTimingConfig`. It has no graph, record, variant, truth, generator, or
oracle dependency:

```python
@dataclass(frozen=True, slots=True)
class TraceTimingSchedule:
    non_action_deltas: tuple[float, ...]
    terminal_compose_time: float
    action_target_time: float


def build_trace_timing_schedule(
    *,
    activation_time: float,
    delay: float,
    competitive_counts: tuple[int, ...],
    jitter_normals: tuple[float, ...],
    timing: OracleTimingConfig,
) -> TraceTimingSchedule: ...
```

The primitive implements Section 8.3 exactly. It requires equal nonempty count
and jitter tuples, exact non-boolean nonnegative integer counts, and finite
`activation_time`, `delay`, every timing scalar, every jitter normal, and every
intermediate/result value. It consumes the provided `jitter_normals` directly;
it has no RNG argument, import, callback, or hidden draw. For step `k` it uses
`math.exp(jitter_normals[k] * timing.jitter_log_std)`, accumulates elapsed
intervals with `math.fsum`, and derives `current_time` with
`math.fsum((activation_time, elapsed))`. It also uses `math.fsum` for the
pre-scale total and terminal composition timestamp. Every final non-action
delta must be at least `delta_min`, terminal composition must be no later than
`activation_time + terminal_compose_fraction * delay`, and
`action_target_time == activation_time + action_target_fraction * delay`
strictly after terminal composition. It raises
`OracleError("oracle trace is temporally infeasible")` otherwise. Oracle,
generator, and the independent invariant analyzer each derive their own
relevant records and competitive counts before calling it; generator and
invariants do not import `env.oracle`.

Before accepting any primary attempt, label selection is the last retryable
construction step. For each nuisance draw/member, the generator constructs the
complete positive, safe-negative, and disconnected-negative counterfactual
family from exactly the same template/structure, node permutation, terminal
class/delay multiset, timestamps, presentation-order source, and reinitialized
member-local `TRACE_JITTER` stream. It runs every retry-causing semantic,
shape, leakage-prevention, and exact trace-timing check on all three candidate
variants. Only if all three pass may it select the request/allocation-assigned
label, materialize that candidate, and proceed to public-ID allocation.
Independent failure retries only that episode; matched failure of any candidate
for any of the four member draws rejects the cohort once and retries the whole
cohort. A post-ID disagreement from Task 8's independently derived invariant
analyzer is a fatal implementation defect and is never converted to sampling
rejection. The independent invariant validator reconstructs the assigned
member's exact schedule from public facts, explicit quartet provenance, and a
rederived authenticated jitter stream. These checks freeze OOD-short
regressions at gate episode indexes `16219` (safe-negative) and `16500`
(positive) and exhaustively
require all 24,000 OOD-short Phase 1 gate traces to build after deterministic
rejection/resampling.

`TRACE_JITTER` is construction evidence because it affects acceptance. The
accepted-attempt token order is exactly cohort-scoped `LABEL`, `TEMPLATE`,
`STRUCTURE`, `TIMESTAMPS`, then for each matched member `0..3` its
`NODE_PERMUTATION`, `TERMINALS`, `PRESENTATION`, `TRACE_JITTER` tokens (20 per
matched cohort); independent order is `TEMPLATE`, `STRUCTURE`, `TIMESTAMPS`,
`NODE_PERMUTATION`, `TERMINALS`, `PRESENTATION`, `TRACE_JITTER` (7 per episode).

## Phase 1 File Map

| Path | Responsibility |
|---|---|
| `configs/data/primary.yaml` | Exact primary distributions, suites, timing constants, retry bound, and gate allocation. |
| `configs/data/stress.yaml` | Exact adversarial generation settings; no runtime/event-engine behavior. |
| `src/silent_cascade/errors.py` | Typed Phase 1 schema, generation, oracle, scoring, leakage, and manifest errors. |
| `src/silent_cascade/provenance.py` | Shared evidence provenance, construction-namespace witnesses, and exact scientific source-tree fingerprints. |
| `src/silent_cascade/rng.py` | Domain-separated counter digest and local PCG64DXSM construction. |
| `src/silent_cascade/schemas.py` | Agent-visible enums, fact/event payloads, records, hypotheses, and actions. |
| `src/silent_cascade/env/__init__.py` | Side-effect-free environment package boundary. |
| `src/silent_cascade/env/config.py` | Strict Phase 1 data, suite, timing, stress, and gate configuration models. |
| `src/silent_cascade/env/episode.py` | Public episode, private truth/bundle, recipes, canonical episode/corpus hashing, and clock transforms. |
| `src/silent_cascade/env/invariants.py` | Independent generator-invariant analysis and mutation-safe validation. |
| `src/silent_cascade/env/generator.py` | Matched-validation, independent claim-data, and stress-suite generation with bounded deterministic rejection. |
| `src/silent_cascade/env/oracle.py` | Facts-derived graph oracle, unique cognitive trace, and target timing. |
| `src/silent_cascade/env/timing.py` | Pure action-window and exact trace-schedule timing primitives shared without graph logic. |
| `src/silent_cascade/env/reward.py` | Half-open scoring, diagnostic random policy, and analytic expectation. |
| `src/silent_cascade/env/leakage.py` | Exact construction checks, shortcut probes, positive controls, and streaming audit. |
| `src/silent_cascade/env/services.py` | Freeze, inspect, oracle-evaluate, leakage-audit, and gate orchestration. |
| `src/silent_cascade/env/reproducibility.py` | Order/chunk/process regeneration comparison and reproducibility reports. |
| `src/silent_cascade/logging/manifest.py` | Strict manifest/envelope models, hash verification, no-clobber publication, and regeneration. |
| `src/silent_cascade/cli.py` | Thin nested Typer adapters and stable JSON/human error rendering. |
| `scripts/check_phase1_reproducibility.py` | Explicit full acceptance harness; never part of routine `make verify`. |
| `scripts/verify_phase1_gate_artifacts.py` | Local cross-artifact count, provenance, counterfactual, and corpus-hash verifier. |
| `tests/fixtures/phase1/` | Hand-authored public facts and expected private oracle/scoring results. |
| `tests/unit/test_phase1_config.py` | Exact data config and cross-limit validation. |
| `tests/unit/test_schemas.py` | Strict public schema and privacy validation. |
| `tests/unit/test_episode.py` | Public projection, canonical serialization, and clock transforms. |
| `tests/unit/test_counter_rng.py` | Counter-seed determinism, isolation, and collision checks. |
| `tests/unit/test_reward.py` | Action-window scoring and analytic random baseline. |
| `tests/unit/test_oracle.py` | Independent positive/safe/disconnected oracle and timing tests. |
| `tests/unit/test_generator_spec.py` | Exact suite/allocation/template construction. |
| `tests/unit/test_generator.py` | Matched cohort construction, retries, IDs, and regeneration. |
| `tests/unit/test_independent_generator.py` | Episode-local claim-data construction, exact allocation, retries, IDs, and regeneration. |
| `tests/unit/test_invariants.py` | Independent facts/truth mutation rejection. |
| `tests/unit/test_clock_scaling.py` | Paired source-key, time, and normalized-window semantics. |
| `tests/unit/test_stress_graphs.py` | Structural adversarial generation and policies. |
| `tests/unit/test_stress_terminals.py` | Contradiction, feasibility-boundary, and checkpoint metadata. |
| `tests/unit/test_leakage.py` | Clean audit, detector power, and injected-leak failures. |
| `tests/unit/test_manifest.py` | Manifest hash, immutable reuse/refusal, corruption, and access class. |
| `tests/unit/test_provenance.py` | Shared evidence fields, exact source path sets, framing, and dirty-source checks. |
| `tests/property/test_generator_properties.py` | Broad invariant, order, regeneration, and scale properties. |
| `tests/integration/test_cli_phase1.py` | All four nested CLI surfaces and failure semantics. |
| `tests/integration/test_phase1_services.py` | Freeze/regenerate/inspect/evaluate/audit service path. |
| `tests/integration/test_phase1_reproducibility.py` | Small process/order/chunk reproducibility profile. |
| `tests/integration/test_phase1_reproducibility_script.py` | Real adapter/fresh-process reproducibility boundary. |
| `tests/integration/test_phase1_trace_feasibility.py` | Exact-index and exhaustive 24,000-row OOD-short trace-feasibility gate. |
| `tests/integration/test_phase1_gate_verifier.py` | Cross-artifact provenance, count, counterfactual, and corpus-hash gate. |
| `tests/regression/test_phase1_fixtures.py` | Exact hand-authored episode/oracle trace regression. |
| `tests/regression/test_import_boundaries.py` | Public/private and oracle import boundaries. |
| `manifests/validation/v1/` | One fixed 10,000-episode validation manifest and hashed Phase 1 gate reports. |
| `README.md` | Truthful Phase 1 command/status documentation with no benchmark claim. |
| `docs/PLAN.md` | Phase 1 plan link and local completion gate. |

## Dependency Direction

```text
validation + errors + hashing + io + config + rng
                      │
                      ▼
                  schemas
                      │
                      ▼
              env.config + env.episode
                 │       │       │       │
                 ▼       ▼       ▼       ▼
        env.generator env.invariants env.oracle env.reward
                 │       │       │       │
                 └───────┴───┬───┴───────┘
                             ▼
                       env.leakage
                             │
env.episode + hashing + io ──┤
              │              │
              ▼              │
      logging.manifest ──────┘
                             ▼
                       env.services
                             ▼
                            cli
```

The diagram expresses data flow, not permission to share reachability code.
`env.generator`, `env.invariants`, and `env.oracle` keep independent graph
implementations and have regression-enforced import restrictions.
`provenance` imports only strict base models, hashing helpers, and the
suite/split enums; generator/oracle code never imports report or manifest
modules. Evidence consumers may import `provenance` without reversing the graph.

## Coverage Matrix

| Canonical requirement | Owning task(s) | Evidence |
|---|---|---|
| §4.1–4.3 facts, activation, silence boundary, half-open outcome | 2, 4 | Strict schemas and boundary scoring tests |
| §4.4 exact primary values | 1 | Strict YAML/config equality tests |
| §4.5 generator invariants and matching | 6–9 | Independent validation, matched/episode-local generation, and mutation tests |
| §4.6 feasibility and terminal construction | 5, 7, 9, 12 | Hand oracle timing tests and both generator-mode checks |
| §4.7 primary/OOD/clock/distractor suites | 6, 9, 10 | Independent suite and paired-scale property tests |
| §4.7 adversarial data suites | 11, 12 | Episode-local stress generation smoke/property tests |
| §7 immutable schemas and private truth boundary | 2 | Sentinel projection/serialization tests |
| §8.2–8.3 oracle traces and timing | 5 | Exact trace-length, support, time, and compression tests |
| §9 `ORACLE` and `RANDOM` diagnostics | 4, 5, 16 | Exact oracle score and stratified random contract |
| §10.1 validation/freeze policy | 13, 14 | Hashed access class and immutable manifest tests |
| §10.3 timed episode success | 4 | Exact scorer boundary matrix |
| §12.4 shortcut audits | 15 | Clean and injected-leak corpora in matched and independent modes |
| §15.1 strict layered config/hash | 1 | Base-plus-data resolution tests |
| §15.2 Phase 1 CLI commands | 17 | CliRunner and installed-command integration tests |
| §17 Phase 1 gate | 18 | 100,000 independent-recipe streaming report and exact denominators |
| §18 unit/property/integration/regression tests | 1–18 | Local `make verify` plus explicit gate commands |

---

### Task 1: Strict Phase 1 Errors and Data Configuration

**Files:**

- Create: `configs/data/primary.yaml`
- Create: `configs/data/stress.yaml`
- Create: `src/silent_cascade/env/__init__.py`
- Create: `src/silent_cascade/env/config.py`
- Modify: `src/silent_cascade/errors.py`
- Create: `tests/unit/test_phase1_config.py`
- Modify: `tests/unit/test_errors.py`

**Interfaces:**

- Consumes: `ProjectConfig`, `resolve_config`, `StrictModel`, and
  `SilentCascadeError` from Phase 0.
- Produces: `Phase1Config`, `OFDDataConfig`, `OracleTimingConfig`,
  `StressDataConfig`, `Phase1GateAllocation`, `SuiteName`, `SplitNamespace`,
  `EpisodeError`, `GenerationError`,
  `EpisodeInvariantError`, `OracleError`, `ScoringError`, `LeakageError`,
  `ManifestError`, and `ManifestAccessError`.

- [ ] **Step 1: Write failing strict-config and typed-error tests**

Create tests that resolve `configs/base.yaml` followed by
`configs/data/primary.yaml`, assert every frozen value, reject unknown keys and
bool-as-int values, and reject a data memory/event/opportunity ceiling above the
Phase 0 limit:

```python
def test_primary_data_config_resolves_exact_frozen_values() -> None:
    resolved = resolve_config(
        Phase1Config,
        [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
    )
    data = resolved.config.data
    assert data.generator_version == "ofd-v1"
    assert data.max_entities == 64
    assert data.hazard_types == 4
    assert data.positive_fraction == 0.5
    assert data.safe_terminal_fraction_within_negatives == 0.5
    assert data.train_path_lengths == (2, 3, 4)
    assert data.ood_depth_path_lengths == (5, 6, 7, 8)
    assert data.train_delay_log_uniform == (8.0, 64.0)
    assert data.ood_depth_delay_log_uniform == (16.0, 128.0)
    assert data.ood_short_delay_log_uniform == (1.0, 8.0)
    assert data.ood_long_delay_log_uniform == (64.0, 1024.0)
    assert data.train_distractor_link_records == (0, 12)
    assert data.ood_distractor_link_records == (16, 48)
    assert data.terminal_hazard_records == 2
    assert data.terminal_safe_records == 1
    assert data.manifest_sizes.validation == 10_000
    assert data.manifest_sizes.compute_curve_subset_iid == 2_500
    assert data.oracle_timing.delta_min == 0.05
    assert data.oracle_timing.action_target_fraction == 0.825


def test_phase1_config_rejects_data_limit_above_project_limit(tmp_path: Path) -> None:
    overlay = tmp_path / "data.yaml"
    overlay.write_text("data:\n  primary_memory_capacity: 65\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(
            Phase1Config,
            [Path("configs/base.yaml"), Path("configs/data/primary.yaml"), overlay],
        )
```

Add parameterized error-payload assertions for every new error code.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
uv run pytest -q tests/unit/test_phase1_config.py tests/unit/test_errors.py
```

Expected: collection fails because the Phase 1 config and error types do not
exist.

- [ ] **Step 3: Add the typed error hierarchy**

Append these classes without changing Phase 0 payload behavior:

```python
class EpisodeError(SilentCascadeError):
    code = "episode_error"


class EpisodeInvariantError(EpisodeError):
    code = "episode_invariant_error"


class GenerationError(EpisodeError):
    code = "generation_error"


class OracleError(EpisodeError):
    code = "oracle_error"


class ScoringError(EpisodeError):
    code = "scoring_error"


class LeakageError(EpisodeError):
    code = "leakage_error"


class ManifestError(ArtifactError):
    code = "manifest_error"


class ManifestAccessError(ManifestError):
    code = "manifest_access_error"
```

- [ ] **Step 4: Implement exact data configuration models**

Define string enums for all suites/splits and strict nested models. Convert YAML
lists to tuples before strict validation. Use Pydantic v2
`@model_validator(mode="after")` on `Phase1Config` to compare the duplicated
data ceilings with `LimitsConfig`:

```python
class SuiteName(str, Enum):
    VALIDATION = "validation"
    IID_PRIMARY = "iid_primary"
    OOD_DEPTH = "ood_depth"
    OOD_SHORT_DELAY = "ood_short_delay"
    OOD_LONG_DELAY = "ood_long_delay"
    DISTRACTOR_FLOOD = "distractor_flood"
    CLOCK_SCALE_0_1X = "clock_scale_0_1x"
    CLOCK_SCALE_10X = "clock_scale_10x"
    BRANCHING_STRESS = "branching_stress"
    CYCLES_STRESS = "cycles_stress"
    CONTRADICTION_STRESS = "contradiction_stress"
    MEMORY_OVERFLOW_STRESS = "memory_overflow_stress"
    NULL_NEAR_MISS_STRESS = "null_near_miss_stress"
    MINIMUM_DURATION_STRESS = "minimum_duration_stress"
    CHECKPOINT_STRESS = "checkpoint_stress"


class SplitNamespace(str, Enum):
    TRAIN = "train"
    DEBUG = "debug"
    VALIDATION = "validation"
    PHASE1_GATE = "phase1_gate"
    FROZEN = "frozen"


class OracleTimingConfig(StrictModel):
    delta_0: float = Field(default=0.25, gt=0.0)
    delta_min: float = Field(default=0.05, gt=0.0)
    delta_max: float = Field(default=1.0, gt=0.0)
    jitter_log_std: float = Field(default=0.1, ge=0.0)
    terminal_compose_fraction: float = Field(default=0.60, gt=0.0, lt=1.0)
    action_window_start_fraction: float = Field(default=0.75, gt=0.0, lt=1.0)
    action_target_fraction: float = Field(default=0.825, gt=0.0, lt=1.0)
    action_window_end_fraction: float = Field(default=0.90, gt=0.0, le=1.0)


class ManifestSizesConfig(StrictModel):
    validation: Literal[10_000] = 10_000
    iid_primary: Literal[20_000] = 20_000
    ood_depth: Literal[20_000] = 20_000
    ood_short_delay: Literal[10_000] = 10_000
    ood_long_delay: Literal[2_000] = 2_000
    clock_scale_0_1x: Literal[5_000] = 5_000
    clock_scale_10x: Literal[2_000] = 2_000
    distractor_flood: Literal[10_000] = 10_000
    branching_stress: Literal[5_000] = 5_000
    cycles_stress: Literal[5_000] = 5_000
    contradiction_stress: Literal[5_000] = 5_000
    checkpoint_stress: Literal[2_000] = 2_000
    compute_curve_subset_iid: Literal[2_500] = 2_500
    compute_curve_subset_ood_depth: Literal[2_500] = 2_500


class Phase1GateAllocation(StrictModel):
    iid_primary: dict[Literal[2, 3, 4], int]
    ood_depth: dict[Literal[5, 6, 7, 8], int]
    ood_short_delay: dict[Literal[2, 3, 4], int]
    ood_long_delay: dict[Literal[2, 3, 4], int]
    distractor_flood: dict[Literal[2, 3, 4], int]
    clock_parent_episodes: dict[Literal[2, 3, 4], int]
    clock_10x_episodes: dict[Literal[2, 3, 4], int]

    @model_validator(mode="after")
    def validate_exact_allocation(self) -> Self:
        expected = {
            "iid_primary": {2: 8_000, 3: 8_000, 4: 8_000},
            "ood_depth": {5: 4_000, 6: 4_000, 7: 4_000, 8: 4_000},
            "ood_short_delay": {2: 8_000, 3: 8_000, 4: 8_000},
            "ood_long_delay": {2: 4_000, 3: 4_000, 4: 4_000},
            "distractor_flood": {2: 8_000, 3: 8_000, 4: 8_000},
            "clock_parent_episodes": {2: 1_668, 3: 1_668, 4: 1_664},
            "clock_10x_episodes": {2: 668, 3: 668, 4: 664},
        }
        actual = {name: getattr(self, name) for name in expected}
        if actual != expected:
            raise ValueError("phase1_gate must equal the frozen allocation")
        return self


class LeakageAuditProfileConfig(StrictModel):
    episode_count: int = Field(gt=0, multiple_of=4)
    permutation_replicates: int = Field(gt=0)
    positive_control_episode_count: int = Field(gt=0, multiple_of=20)
    positive_control_permutation_replicates: int = Field(gt=0)
    minimum_test_examples_per_class: int = Field(gt=0)
    enforce_clean_statistical_gate: bool


class LeakageAuditConfig(StrictModel):
    schema_version: Literal["leakage-v1"] = "leakage-v1"
    audit_seed: Literal[2026083091] = 2026083091
    positive_control_seed: Literal[2026083092] = 2026083092
    train_fraction: Literal[0.8] = 0.8
    alpha: Literal[0.01] = 0.01
    l2_penalty: Literal[0.03] = 0.03
    optimizer_max_iterations: Literal[500] = 500
    optimizer_gradient_tolerance: Literal[1.0e-8] = 1.0e-8
    optimizer_function_tolerance: Literal[1.0e-12] = 1.0e-12
    positive_control_min_balanced_accuracy: Literal[0.95] = 0.95
    feature_batch_size: int = Field(default=4096, ge=1, le=4096)
    permutation_batch_size: int = Field(default=64, ge=1, le=64)
    max_feature_store_bytes: int = Field(default=800_000_000, gt=0, le=800_000_000)
    max_resident_working_bytes: int = Field(default=512_000_000, gt=0, le=512_000_000)
    test: LeakageAuditProfileConfig
    phase1_gate: LeakageAuditProfileConfig


class OFDDataConfig(StrictModel):
    generator_version: Literal["ofd-v1"] = "ofd-v1"
    max_entities: Literal[64] = 64
    hazard_types: Literal[4] = 4
    positive_fraction: Literal[0.5] = 0.5
    safe_terminal_fraction_within_negatives: Literal[0.5] = 0.5
    train_path_lengths: tuple[int, ...]
    iid_test_path_lengths: tuple[int, ...]
    ood_depth_path_lengths: tuple[int, ...]
    ood_short_delay_path_lengths: tuple[int, ...]
    ood_depth_delay_log_uniform: tuple[float, float]
    terminal_hazard_records: Literal[2] = 2
    terminal_safe_records: Literal[1] = 1
    train_distractor_link_records: tuple[int, int]
    ood_distractor_link_records: tuple[int, int]
    train_delay_log_uniform: tuple[float, float]
    ood_short_delay_log_uniform: tuple[float, float]
    ood_long_delay_log_uniform: tuple[float, float]
    observation_gap_log_uniform: tuple[float, float]
    primary_memory_capacity: Literal[64] = 64
    max_eventflow_internal_events: Literal[64] = 64
    max_fixed_grid_opportunities: Literal[25_000] = 25_000
    max_generation_attempts: int = Field(default=1_000, ge=1, le=1_000)
    manifest_sizes: ManifestSizesConfig
    oracle_timing: OracleTimingConfig
    phase1_gate: Phase1GateAllocation
    leakage_audit: LeakageAuditConfig


class StressDataConfig(StrictModel):
    branching_records: tuple[int, int]
    irrelevant_cycle_length: tuple[int, int]
    contradiction_records: tuple[int, int]
    overflow_record_count: tuple[int, int]
    near_miss_missing_edges: Literal[1] = 1
    minimum_duration_epsilon: float = Field(default=1.0e-6, gt=0.0)
    minimum_duration_search_upper: Literal[64.0] = 64.0
    minimum_duration_monotonic_grid_points: Literal[257] = 257
    branching_freeze_episodes: Literal[5_000] = 5_000
    cycles_freeze_episodes: Literal[5_000] = 5_000
    contradiction_freeze_episodes: Literal[5_000] = 5_000
    memory_overflow_freeze_episodes: Literal[5_000] = 5_000
    null_near_miss_freeze_episodes: Literal[5_000] = 5_000
    minimum_duration_freeze_episodes: Literal[5_000] = 5_000
    checkpoint_freeze_episodes: Literal[2_000] = 2_000


class Phase1Config(ProjectConfig):
    data: OFDDataConfig
    stress: StressDataConfig | None = None

    @model_validator(mode="after")
    def validate_phase1_limits(self) -> Self:
        if self.data.primary_memory_capacity > self.limits.primary_memory_records:
            raise ValueError("data memory capacity exceeds project limit")
        if self.data.max_eventflow_internal_events > self.limits.max_eventflow_events:
            raise ValueError("data event ceiling exceeds project limit")
        if self.data.max_fixed_grid_opportunities > self.limits.max_fixed_grid_opportunities:
            raise ValueError("data opportunity ceiling exceeds project limit")
        return self
```

Add `@field_validator("train_path_lengths", "iid_test_path_lengths",
"ood_depth_path_lengths", "ood_short_delay_path_lengths",
"ood_depth_delay_log_uniform", "train_distractor_link_records",
"ood_distractor_link_records", "train_delay_log_uniform",
"ood_short_delay_log_uniform", "ood_long_delay_log_uniform",
"observation_gap_log_uniform", mode="before")`; accept YAML lists there and
return tuples. Shared helpers reject booleans,
nonfinite values, unsorted/duplicate path tuples, and non-increasing two-item
ranges. `OracleTimingConfig` has a model validator enforcing the four timing
inequalities listed below. Tests call each helper with both valid boundaries
and one invalid value, so none can be a no-op.

The validator must enforce:

```text
delta_min <= delta_0 <= delta_max
terminal_compose_fraction < action_window_start_fraction
action_window_start_fraction < action_target_fraction < action_window_end_fraction
primary_memory_capacity <= limits.primary_memory_records
max_eventflow_internal_events <= limits.max_eventflow_events
max_fixed_grid_opportunities <= limits.max_fixed_grid_opportunities
max_generation_attempts in 1..1000
all ranges are two finite increasing endpoints
all path-length tuples are sorted, unique, and positive
```

- [ ] **Step 5: Write the exact YAML layers**

`configs/data/primary.yaml` contains the Section 4.4 values plus:

```yaml
data:
  generator_version: ofd-v1
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
  max_generation_attempts: 1000
  manifest_sizes:
    validation: 10000
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
  oracle_timing:
    delta_0: 0.25
    delta_min: 0.05
    delta_max: 1.0
    jitter_log_std: 0.1
    terminal_compose_fraction: 0.60
    action_window_start_fraction: 0.75
    action_target_fraction: 0.825
    action_window_end_fraction: 0.90
  phase1_gate:
    iid_primary: {2: 8000, 3: 8000, 4: 8000}
    ood_depth: {5: 4000, 6: 4000, 7: 4000, 8: 4000}
    ood_short_delay: {2: 8000, 3: 8000, 4: 8000}
    ood_long_delay: {2: 4000, 3: 4000, 4: 4000}
    distractor_flood: {2: 8000, 3: 8000, 4: 8000}
    clock_parent_episodes: {2: 1668, 3: 1668, 4: 1664}
    clock_10x_episodes: {2: 668, 3: 668, 4: 664}
  leakage_audit:
    schema_version: leakage-v1
    audit_seed: 2026083091
    positive_control_seed: 2026083092
    train_fraction: 0.80
    alpha: 0.01
    l2_penalty: 0.03
    optimizer_max_iterations: 500
    optimizer_gradient_tolerance: 1.0e-8
    optimizer_function_tolerance: 1.0e-12
    positive_control_min_balanced_accuracy: 0.95
    feature_batch_size: 4096
    permutation_batch_size: 64
    max_feature_store_bytes: 800000000
    max_resident_working_bytes: 512000000
    test:
      episode_count: 1200
      permutation_replicates: 199
      positive_control_episode_count: 1200
      positive_control_permutation_replicates: 199
      minimum_test_examples_per_class: 8
      enforce_clean_statistical_gate: false
    phase1_gate:
      episode_count: 100000
      permutation_replicates: 4999
      positive_control_episode_count: 8000
      positive_control_permutation_replicates: 4999
      minimum_test_examples_per_class: 200
      enforce_clean_statistical_gate: true
```

`configs/data/stress.yaml` supplies strict generation ranges and declared future
freeze sizes:

```yaml
stress:
  branching_records: [2, 6]
  irrelevant_cycle_length: [2, 6]
  contradiction_records: [2, 4]
  overflow_record_count: [65, 80]
  near_miss_missing_edges: 1
  minimum_duration_epsilon: 1.0e-6
  minimum_duration_search_upper: 64.0
  minimum_duration_monotonic_grid_points: 257
  branching_freeze_episodes: 5000
  cycles_freeze_episodes: 5000
  contradiction_freeze_episodes: 5000
  memory_overflow_freeze_episodes: 5000
  null_near_miss_freeze_episodes: 5000
  minimum_duration_freeze_episodes: 5000
  checkpoint_freeze_episodes: 2000
```

The three 5,000 counts absent from the canonical test-size table are Phase 1
defaults only. Phase 6 must freeze them explicitly before final generation.

- [ ] **Step 6: Run GREEN checks**

Run:

```bash
uv run pytest -q tests/unit/test_phase1_config.py tests/unit/test_errors.py
uv run ruff check src/silent_cascade/env/config.py src/silent_cascade/errors.py tests/unit/test_phase1_config.py
uv run ruff format --check src/silent_cascade/env/config.py src/silent_cascade/errors.py tests/unit/test_phase1_config.py
```

Expected: all focused tests and Ruff checks pass.

- [ ] **Step 7: Commit**

```bash
git add configs/data/primary.yaml configs/data/stress.yaml \
  src/silent_cascade/env/__init__.py src/silent_cascade/env/config.py \
  src/silent_cascade/errors.py tests/unit/test_phase1_config.py \
  tests/unit/test_errors.py
git commit -m "feat: define Phase 1 data configuration"
```

---

### Task 2: Public Schemas, Private Episode Bundles, and Canonical Serialization

**Files:**

- Create: `src/silent_cascade/schemas.py`
- Create: `src/silent_cascade/env/episode.py`
- Create: `tests/unit/test_schemas.py`
- Create: `tests/unit/test_episode.py`

**Interfaces:**

- Consumes: Task 1 enums/config and Phase 0 canonical hashing.
- Produces: `RecordKind`, `ExternalEventKind`, `InternalEventKind`, `Condition`,
  `Provenance`, `Mode`, fact payloads, `AgentInit`, `ExternalEvent`,
  `MatchedEpisodeCoordinate`,
  `IndependentEpisodeCoordinate`, `EpisodeCoordinate`,
  `InternalEvent`, `MemoryRecord`, `Hypothesis`, `Action`, `PublicEpisode`,
  `EpisodeVariant`, `EpisodeKey`, `EpisodeRecipe`, `EpisodeTruth`,
  `StressMetadata`, `EpisodeBundle`, `fact_event_to_memory_record`,
  `PublicEpisodeArtifact`, `EpisodeArtifact`, `public_projection`,
  `canonical_episode_bytes`, `episode_sha256`, `episode_from_bytes`,
  `CorpusDigestEntry`, `CorpusHashBuilder`, and `corpus_sha256`.

- [ ] **Step 1: Write failing schema and privacy tests**

Cover exact enum values, bool-as-int rejection, invalid payload combinations,
finite times, immutable tuples, and the public sentinel boundary:

```python
def test_public_projection_has_no_private_truth_fields(positive_bundle: EpisodeBundle) -> None:
    public = public_projection(positive_bundle)
    dumped = repr(public)
    for forbidden in (
        "root_seed",
        "episode_index",
        "positive",
        "relevant_record_ids",
        "terminal_time",
        "action_window",
        "rejection",
    ):
        assert forbidden not in dumped
    assert {event.kind for event in public.events} <= {
        ExternalEventKind.FACT,
        ExternalEventKind.ACTIVATE,
    }


@pytest.mark.parametrize("invalid", [True, -1, 1.5])
def test_link_fact_rejects_non_exact_nonnegative_entity_ids(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        LinkFact(source_node=invalid, target_node=2, confidence=1.0)
```

Add a sentinel whose hidden values are distinctive strings/numbers and assert
they do not occur in `PublicEpisode`, public primitive serialization, or a
public exception payload.

- [ ] **Step 2: Run the focused tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_schemas.py tests/unit/test_episode.py
```

Expected: collection fails because the schema and episode modules do not exist.

- [ ] **Step 3: Implement strict frozen public dataclasses**

Define the complete public enum vocabulary and core records here rather than
inventing any additional kind:

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
    NOOP = "noop"


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


class EpisodeVariant(str, Enum):
    POSITIVE = "positive"
    SAFE_NEGATIVE = "safe_negative"
    DISCONNECTED_NEGATIVE = "disconnected_negative"
```

Round-trip every `EpisodeVariant` through private canonical serialization and
reject unknown spellings. It is environment-private and never appears in a
`PublicEpisodeArtifact` or public error payload.

Add fact payloads with strict `__post_init__` checks:

```python
@dataclass(frozen=True, slots=True)
class LinkFact:
    source_node: int
    target_node: int
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class HazardFact:
    node: int
    hazard_type: int
    delay: float
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class SafeFact:
    node: int
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class ActivationPayload:
    start_node: int


type FactPayload = LinkFact | HazardFact | SafeFact
type ExternalPayload = FactPayload | ActivationPayload | None
```

`ExternalEvent` validates these exact pairings:

```text
FACT      -> LinkFact | HazardFact | SafeFact
ACTIVATE  -> ActivationPayload
OUTCOME   -> None
END       -> None
```

Implement these exact public data fields; validation rejects nonfinite time,
confidence outside `[0,1]`, negative/excess entity or hazard IDs, and an event
kind/payload mismatch:

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
    payload: ExternalPayload


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

Do not define `AgentCondition` or `ClosedFormFlow` until their referenced
runtime types exist in Phase 2. `MemoryRecord` remains the immutable semantic
record; later runtime slot tensors are a different type. In Phase 1, every
`FACT` event converts
one-to-one to one `MemoryRecord`, and `record_id == event_id`; activation and
private terminal event IDs never enter the record namespace:

```python
def fact_event_to_memory_record(event: ExternalEvent) -> MemoryRecord:
    if event.kind is not ExternalEventKind.FACT:
        raise EpisodeInvariantError("only FACT events become memory records")
    payload = event.payload
    if isinstance(payload, LinkFact):
        kind, subject, object_id, hazard_type, delay = (
            RecordKind.LINK, payload.source_node, payload.target_node, None, None
        )
    elif isinstance(payload, HazardFact):
        kind, subject, object_id, hazard_type, delay = (
            RecordKind.HAZARD, payload.node, None, payload.hazard_type, payload.delay
        )
    elif isinstance(payload, SafeFact):
        kind, subject, object_id, hazard_type, delay = (
            RecordKind.SAFE, payload.node, None, None, None
        )
    else:
        raise EpisodeInvariantError("FACT has an invalid payload")
    return MemoryRecord(
        record_id=event.event_id,
        kind=kind,
        subject_id=subject,
        object_id=object_id,
        hazard_type=hazard_type,
        delay=delay,
        observed_at=event.timestamp,
        confidence=payload.confidence,
        provenance=Provenance.PERCEIVED,
        support_ids=(),
    )
```

All oracle `selected_record_id`, terminal record IDs, and support IDs therefore
refer to the originating FACT event ID. Test this mapping for every fact kind
and reject duplicate public event IDs.

- [ ] **Step 4: Implement structurally separate episode types**

Use:

```python
@dataclass(frozen=True, slots=True)
class PublicEpisode:
    init: AgentInit
    events: tuple[ExternalEvent, ...]


@dataclass(frozen=True, slots=True)
class MatchedEpisodeCoordinate:
    mode: Literal["matched"]
    cohort_index: int
    member_index: int


@dataclass(frozen=True, slots=True)
class IndependentEpisodeCoordinate:
    mode: Literal["independent"]
    episode_index: int
    allocation_quartet_index: int
    quartet_member_index: int


type EpisodeCoordinate = MatchedEpisodeCoordinate | IndependentEpisodeCoordinate


@dataclass(frozen=True, slots=True)
class EpisodeKey:
    generator_version: str
    split_namespace: SplitNamespace
    suite: SuiteName
    root_seed: int
    coordinate: EpisodeCoordinate


@dataclass(frozen=True, slots=True)
class EpisodeRecipe:
    requested_path_length: int
    variant: EpisodeVariant
    distractor_link_count: int
    evaluation_suite: SuiteName
    accepted_attempt: int
    clock_scale: float = 1.0
    parent_public_id: str | None = None
    parent_episode_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class StressMetadata:
    over_capacity_record_count: int | None = None
    proposed_checkpoint_pause_time: float | None = None
    minimum_feasible_delay: float | None = None


@dataclass(frozen=True, slots=True)
class EpisodeTruth:
    key: EpisodeKey
    recipe: EpisodeRecipe
    relevant_node_path: tuple[int, ...]
    relevant_record_ids: tuple[int, ...]
    terminal_record_id: int | None
    relevant_hazard_type: int | None
    private_terminal: ExternalEvent
    activation_time: float
    episode_delay: float
    action_window_start: float | None
    action_window_end: float | None
    action_target: float | None
    rejection_count: int
    rejection_reasons: tuple[str, ...]
    stress_metadata: StressMetadata | None = None


@dataclass(frozen=True, slots=True)
class EpisodeBundle:
    public: PublicEpisode
    truth: EpisodeTruth
```

Validate chronological public events, exactly one final activation, private
terminal kind/timestamp, discriminated coordinate bounds (including exact
non-boolean `quartet_member_index` in `0..3`), recipe consistency,
and the absence of
private terminal events from `PublicEpisode.events`. Validate that stress
metadata is absent for primary suites, that over-capacity count is present only
for `MEMORY_OVERFLOW_STRESS`, and that a pause time is finite and strictly
between activation and the private terminal only for `CHECKPOINT_STRESS`.
`minimum_feasible_delay` is finite/positive and present only for
`MINIMUM_DURATION_STRESS`; all three fields remain private serialization.

- [ ] **Step 5: Add strict canonical episode artifacts**

Create private `StrictModel` artifact types in `env.episode`, with explicit
list-to-tuple validators. Implement lossless conversion functions rather than
passing dataclasses to `canonical_json_bytes`:

```python
class PublicEpisodeArtifact(StrictModel):
    schema_version: Literal[1] = 1
    generator_version: Literal["ofd-v1"] = "ofd-v1"
    init: AgentInit
    events: tuple[ExternalEvent, ...]

    @classmethod
    def from_public(cls, public: PublicEpisode) -> Self:
        return cls(init=public.init, events=public.events)

    def to_public(self) -> PublicEpisode:
        return PublicEpisode(init=self.init, events=self.events)


class EpisodeArtifact(StrictModel):
    schema_version: Literal[1] = 1
    generator_version: Literal["ofd-v1"] = "ofd-v1"
    public: PublicEpisodeArtifact
    truth: EpisodeTruth

    @classmethod
    def from_bundle(cls, bundle: EpisodeBundle) -> Self:
        return cls(
            public=PublicEpisodeArtifact.from_public(bundle.public),
            truth=bundle.truth,
        )

    def to_bundle(self) -> EpisodeBundle:
        return EpisodeBundle(public=self.public.to_public(), truth=self.truth)


def canonical_episode_bytes(bundle: EpisodeBundle) -> bytes:
    artifact = EpisodeArtifact.from_bundle(bundle)
    return canonical_json_bytes(artifact)


def episode_sha256(bundle: EpisodeBundle) -> str:
    return sha256_bytes(canonical_episode_bytes(bundle))


def episode_from_bytes(payload: bytes) -> EpisodeBundle:
    artifact = EpisodeArtifact.model_validate_json(payload)
    return artifact.to_bundle()
```

The serialized private artifact begins with `schema_version=1` and
`generator_version="ofd-v1"`. Add tests for byte-identical round trips, unknown
fields, altered event order, duplicate IDs, and nonfinite values.

Define one corpus digest algorithm here and reuse it unchanged in Tasks 14–16:

```python
@dataclass(frozen=True, slots=True)
class CorpusDigestEntry:
    episode_public_id: str
    episode_sha256: str


class CorpusHashBuilder:
    def __init__(self, expected_count: int) -> None: ...
    def add(self, entry: CorpusDigestEntry) -> None: ...
    def finalize(self) -> str: ...


def corpus_sha256(
    entries: Iterable[CorpusDigestEntry], *, expected_count: int
) -> str: ...
```

Initialize SHA-256 with `b"silent-cascade/corpus/v1\0"` followed by the
expected count as an unsigned eight-byte big-endian integer. For each entry in
canonical source order, append its zero-based ordinal as unsigned eight-byte
big-endian, the UTF-8 public-ID length as unsigned four-byte big-endian, the
public-ID bytes, and the 32 raw bytes decoded from the lowercase 64-hex episode
digest. Reject invalid counts/digests/IDs, duplicate public IDs, too many/few
entries, add-after-finalize, or a second finalize. The two-entry known-answer
fixture uses exact valid RFC 4122 v4 IDs
`00000000-0000-4000-8000-000000000001` and
`00000000-0000-4000-8000-000000000002`, with digests `"11" * 32` and
`"22" * 32`; its corpus hash is
`d38af712baf94c88101b00aabfba0178e9e5957d8faf1ea58f6faa83d37dd41f`.

“Canonical source order” means manifest entry order for a manifest source and
declared allocation/request order for an allocation source, regardless of the
actual generation traversal or chunking order. Reproducibility modes collect
results by coordinate and feed them back in that canonical order. For the
Phase 1 independent gate, the denominator and digest contain exactly the
100,000 base episodes; derived `0.1x`/`10x` clock counterfactual children are
excluded. A future manifest digest includes every entry actually present in
that manifest, including clock children. Positive-control overlays and other
counterfactuals have their own explicitly named hashes and never alter the
clean base-corpus digest.

- [ ] **Step 6: Run GREEN checks**

```bash
uv run pytest -q tests/unit/test_schemas.py tests/unit/test_episode.py
uv run ruff check src/silent_cascade/schemas.py src/silent_cascade/env/episode.py tests/unit/test_schemas.py tests/unit/test_episode.py
uv run ruff format --check src/silent_cascade/schemas.py src/silent_cascade/env/episode.py tests/unit/test_schemas.py tests/unit/test_episode.py
```

Expected: focused tests and Ruff pass.

- [ ] **Step 7: Commit**

```bash
git add src/silent_cascade/schemas.py src/silent_cascade/env/episode.py \
  tests/unit/test_schemas.py tests/unit/test_episode.py
git commit -m "feat: add private-safe OFD episode schemas"
```

---

### Task 3: Domain-Separated Counter RNG and Public IDs

**Files:**

- Modify: `src/silent_cascade/rng.py`
- Create: `tests/unit/test_counter_rng.py`

**Interfaces:**

- Consumes: Task 1 suite/split enums and Phase 0 hashing/RNG snapshots.
- Produces: `SeedStream`, `CounterSeedKey`, `CounterSeed`,
  `derive_counter_seed`, `local_generator`, `IndependentCounterSeedKey`,
  `derive_independent_counter_seed`,
  `independent_local_generator`, `AllocationLabelKey`,
  `allocate_independent_variants`, `PublicIdBatchKey`,
  `allocate_public_ids`, `IndependentPublicIdKey`, and
  `allocate_independent_public_id`.

- [ ] **Step 1: Write failing determinism and known-answer tests**

```python
def test_counter_rng_is_order_independent_and_does_not_touch_globals() -> None:
    before = snapshot_global_rng()
    keys = [counter_seed_key(index) for index in range(20)]
    forward = {item.cohort_index: local_generator(item).integers(0, 2**63) for item in keys}
    reverse = {
        item.cohort_index: local_generator(item).integers(0, 2**63)
        for item in reversed(keys)
    }
    assert forward == reverse
    assert_rng_snapshots_equal(snapshot_global_rng(), before)
```

`assert_rng_snapshots_equal` must compare Python RNG state structurally,
NumPy arrays with `numpy.array_equal`, Torch tensors with `torch.equal`, and the
optional MPS state only when MPS is available; dataclass equality is not valid
for tensor-bearing snapshots.

Also assert all four frozen cohort/independent seed and ID known-answer vectors,
stream/member-scope rejection,
bool/range rejection, UUID version/variant bits, seed-key JSON byte equality,
and identical results in a subprocess with `PYTHONHASHSEED=0` and `1`.

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_counter_rng.py
```

Expected: imports fail because the counter RNG/public-ID APIs do not exist.

- [ ] **Step 3: Implement counter-seed derivation**

```python
@dataclass(frozen=True, slots=True)
class CounterSeed:
    token: str
    seed: int


COHORT_SCOPED_STREAMS = frozenset(
    {SeedStream.LABEL, SeedStream.TEMPLATE, SeedStream.STRUCTURE, SeedStream.TIMESTAMPS}
)


def validate_seed_key_scope(key: CounterSeedKey) -> None:
    if type(key.member_index) is not int:
        raise ValueError("member_index must be an exact integer")
    for name, value, lower, upper in (
        ("root_seed", key.root_seed, 0, 2**128),
        ("cohort_index", key.cohort_index, 0, None),
        ("attempt", key.attempt, 0, 1_000),
    ):
        if type(value) is not int or value < lower or (upper is not None and value >= upper):
            raise ValueError(f"{name} is outside its exact integer range")
    cohort_scoped = key.stream in COHORT_SCOPED_STREAMS
    if (cohort_scoped and key.member_index != -1) or (
        not cohort_scoped and key.member_index not in range(4)
    ):
        raise ValueError("member_index does not match stream scope")


def seed_key_primitive(key: CounterSeedKey) -> dict[str, JsonValue]:
    return {
        "generator_version": key.generator_version,
        "split_namespace": key.split_namespace.value,
        "suite": key.suite.value,
        "root_seed": key.root_seed,
        "cohort_index": key.cohort_index,
        "member_index": key.member_index,
        "stream": key.stream.value,
        "attempt": key.attempt,
    }


def derive_counter_seed(key: CounterSeedKey) -> CounterSeed:
    validate_seed_key_scope(key)
    payload = canonical_json_bytes(
        {
            "domain": "silent-cascade/ofd-v1/counter-seed/v1",
            **seed_key_primitive(key),
        }
    )
    digest = hashlib.sha256(payload).digest()
    return CounterSeed(token=digest.hex(), seed=int.from_bytes(digest[:16], "big"))


def local_generator(key: CounterSeedKey) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(derive_counter_seed(key).seed))


def validate_public_id_key(key: PublicIdBatchKey) -> None:
    for name, value, upper in (
        ("public_id_seed", key.public_id_seed, 2**128),
        ("cohort_index", key.cohort_index, None),
        ("accepted_attempt", key.accepted_attempt, 1_000),
    ):
        if type(value) is not int or value < 0 or (upper is not None and value >= upper):
            raise ValueError(f"{name} is outside its exact integer range")


def allocate_public_ids(key: PublicIdBatchKey) -> tuple[str, str, str, str]:
    validate_public_id_key(key)
    hmac_key = (
        b"silent-cascade/ofd-v1/public-id-key/v1\0"
        + key.public_id_seed.to_bytes(16, "big")
    )
    context = {
        "generator_version": key.generator_version,
        "split_namespace": key.split_namespace.value,
        "suite": key.suite.value,
        "cohort_index": key.cohort_index,
        "attempt": key.accepted_attempt,
    }
    candidates: list[str] = []
    for counter in range(4):
        message = canonical_json_bytes(
            {"domain": "silent-cascade/ofd-v1/id-pool/v1", **context, "counter": counter}
        )
        raw = bytearray(hmac.new(hmac_key, message, hashlib.sha256).digest()[:16])
        raw[6] = (raw[6] & 0x0F) | 0x40
        raw[8] = (raw[8] & 0x3F) | 0x80
        candidates.append(str(uuid.UUID(bytes=bytes(raw))))
    if len(set(candidates)) != 4:
        raise GenerationError("public ID collision")
    assigned = sorted(
        candidates,
        key=lambda public_id: (
            hmac.new(
                hmac_key,
                canonical_json_bytes(
                    {
                        "domain": "silent-cascade/ofd-v1/id-assignment/v1",
                        **context,
                        "public_id": public_id,
                    }
                ),
                hashlib.sha256,
            ).digest(),
            public_id,
        ),
    )
    return assigned[0], assigned[1], assigned[2], assigned[3]
```

`seed_key_primitive` returns exactly the eight non-domain fields shown above
and normalizes each enum to `.value`; the enclosing domain-framed
`derive_counter_seed` payload contains nine fields after adding `domain`.
Neither function accepts arbitrary mappings. `validate_seed_key_scope`
enforces the integer bounds and cohort/member stream rule from the frozen
ruling. Implement `PublicIdBatchKey` with generator version, split, suite,
128-bit public-ID seed, cohort index, and attempt, and
`allocate_public_ids(key: PublicIdBatchKey) -> tuple[str, str, str, str]` with
the byte framing/HMAC algorithm and both known-answer vectors in the frozen
ruling. Add a focused 10,000-token unit collision test. Unit tests exercise
bounded samples only. Task 18 derives the complete namespace evidence from
accepted draws and requires exactly 50,000 matched tokens plus 700,000
independent tokens, and 10,000 matched base IDs plus 100,000 independent base
IDs and 7,000 independent clock IDs: 750,000 construction tokens and 117,000
public IDs with zero within-source or cross-split collisions. Coordinate tuples
are not seed tokens and may not substitute for these actual derived values.

Implement the independent path beside—not by overloading—the cohort path.
`derive_independent_counter_seed` uses the exact independent domain and payload
from the frozen ruling, rejects `SeedStream.LABEL`, and
`independent_local_generator` constructs a local PCG64DXSM instance. For each
`AllocationLabelKey`, hash the canonical key under domain
`silent-cascade/ofd-v1/allocation-label/v1`, use the first 128 bits as a local
PCG64DXSM seed, and select uniformly from the 12 distinct permutations of
`[P,P,S,D]` in lexicographic enum-value order. The result is exactly four
variants and does not consume an episode stream.

`allocate_independent_public_id` implements the single-ID HMAC algorithm and
known-answer vector in the frozen ruling. Validate its exact non-boolean ranges
and test that its domain cannot collide with any matched-ID token in a focused
10,000+10,000 token sample.

- [ ] **Step 4: Run focused GREEN checks**

```bash
uv run pytest -q tests/unit/test_counter_rng.py tests/unit/test_rng.py
uv run ruff check src/silent_cascade/rng.py tests/unit/test_counter_rng.py
uv run ruff format --check src/silent_cascade/rng.py tests/unit/test_counter_rng.py
```

Expected: new and existing RNG tests pass; Ruff is clean.

- [ ] **Step 5: Commit**

```bash
git add src/silent_cascade/rng.py tests/unit/test_counter_rng.py
git commit -m "feat: add domain-separated episode RNG"
```

---

### Task 4: Half-Open Scoring and Public-Only Random Baseline

**Files:**

- Create: `src/silent_cascade/env/reward.py`
- Create: `tests/unit/test_reward.py`

**Interfaces:**

- Consumes: Task 1 timing config and Task 2 public/private episode/action types.
- Produces: `ActionWindow`, `EpisodeScore`, `action_window`, `score_actions`,
  `random_baseline_actions`, positive expectation `0.125`, negative expectation
  `0.5`, and balanced pooled diagnostic `0.3125`.

- [ ] **Step 1: Write failing boundary and policy tests**

```python
def test_action_window_is_half_open() -> None:
    window = action_window(100.0, 24.0, OracleTimingConfig())
    assert (window.start, window.end, window.target) == (118.0, 121.6, 119.8)
    assert window.contains(window.start)
    assert window.contains(math.nextafter(window.end, -math.inf))
    assert not window.contains(window.end)


@pytest.mark.parametrize(
    "actions",
    [(), (wrong_class_action(),), (early_action(),), (late_action(),),
     (correct_action(), correct_action())],
)
def test_positive_requires_one_correct_in_window_action(actions: tuple[Action, ...]) -> None:
    assert not score_actions(positive_truth(), actions).timed_success
```

Cover negative abstention/action, pre-activation and nonfinite action times,
exact-right-boundary/outcome-time rejection, random abstention/class/timestamp
frequencies from a fixed seed, public-only input typing, and analytic
always-shield success `0.5 * 0.25 = 0.125`.

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_reward.py
```

Expected: import fails because `env.reward` does not exist.

- [ ] **Step 3: Implement exact scoring and diagnostic random policy**

```python
@dataclass(frozen=True, slots=True)
class ActionWindow:
    start: float
    end: float
    target: float

    def contains(self, timestamp: float) -> bool:
        return self.start <= timestamp < self.end


@dataclass(frozen=True, slots=True)
class EpisodeScore:
    timed_success: bool
    is_positive: bool
    action_count: int
    correct_class: bool | None
    in_window: bool | None
    false_action: bool
    reason: str


def action_window(
    activation_time: float,
    delay: float,
    timing: OracleTimingConfig,
) -> ActionWindow:
    return ActionWindow(
        start=activation_time + timing.action_window_start_fraction * delay,
        end=activation_time + timing.action_window_end_fraction * delay,
        target=activation_time + timing.action_target_fraction * delay,
    )


def score_actions(truth: EpisodeTruth, actions: Sequence[Action]) -> EpisodeScore:
    if any(not math.isfinite(action.timestamp) for action in actions):
        raise ScoringError("action timestamp must be finite")
    if any(action.timestamp < truth.activation_time for action in actions):
        raise ScoringError("action precedes activation")
    if truth.recipe.variant is not EpisodeVariant.POSITIVE:
        return EpisodeScore(
            timed_success=not actions,
            is_positive=False,
            action_count=len(actions),
            correct_class=None,
            in_window=None,
            false_action=bool(actions),
            reason="negative_abstention" if not actions else "negative_false_action",
        )
    if len(actions) != 1:
        return EpisodeScore(
            timed_success=False,
            is_positive=True,
            action_count=len(actions),
            correct_class=None,
            in_window=None,
            false_action=False,
            reason="action_count",
        )
    action = actions[0]
    if (
        truth.relevant_hazard_type is None
        or truth.action_window_start is None
        or truth.action_window_end is None
    ):
        raise ScoringError("positive episode is missing scorer truth")
    correct_class = action.hazard_type == truth.relevant_hazard_type
    in_window = truth.action_window_start <= action.timestamp < truth.action_window_end
    return EpisodeScore(
        timed_success=correct_class and in_window,
        is_positive=True,
        action_count=1,
        correct_class=correct_class,
        in_window=in_window,
        false_action=False,
        reason="success" if correct_class and in_window else "incorrect",
    )
```

Implement the public-only diagnostic policy exactly:

```python
def random_baseline_actions(
    public: PublicEpisode,
    timing: OracleTimingConfig,
    rng: np.random.Generator,
) -> tuple[Action, ...]:
    activation = public.events[-1]
    if activation.kind is not ExternalEventKind.ACTIVATE:
        raise ScoringError("public episode has no final activation")
    delays = [
        event.payload.delay
        for event in public.events
        if event.kind is ExternalEventKind.FACT
        and isinstance(event.payload, HazardFact)
    ]
    if len(delays) != 2 or delays[0] != delays[1]:
        raise ScoringError("random baseline requires two equal public hazard delays")
    if float(rng.random()) < 0.5:
        return ()
    window = action_window(activation.timestamp, delays[0], timing)
    return (
        Action(
            hazard_type=int(rng.integers(0, public.init.hazard_type_count)),
            timestamp=float(rng.uniform(window.start, window.end)),
            caused_by_event_id=activation.event_id,
        ),
    )
```

`random_baseline_actions(public, timing, rng)` obtains activation time and
verifies the two public hazard delays are equal. It calls `action_window` with
the supplied `OracleTimingConfig`, samples abstention/class and a uniform
normalized point in the public window. A sampled action uses the public
activation event ID as `caused_by_event_id`, because it is scheduled directly
at activation. The function never accepts `EpisodeTruth`. No timing fraction is
repeated as a numeric literal outside `configs/data/primary.yaml` and the
`OracleTimingConfig` defaults.

- [ ] **Step 4: Run GREEN checks**

```bash
uv run pytest -q tests/unit/test_reward.py
uv run ruff check src/silent_cascade/env/reward.py tests/unit/test_reward.py
uv run ruff format --check src/silent_cascade/env/reward.py tests/unit/test_reward.py
```

Expected: all scoring/policy tests pass; Ruff is clean.

- [ ] **Step 5: Commit**

```bash
git add src/silent_cascade/env/reward.py tests/unit/test_reward.py
git commit -m "feat: score OFD actions and random policy"
```

---

### Task 5: Independent Facts-Derived Oracle and Cognitive Timing

**Files:**

- Modify: `src/silent_cascade/env/timing.py`
- Create: `src/silent_cascade/env/oracle.py`
- Create: `tests/fixtures/phase1/oracle_cases.json`
- Modify: `tests/unit/test_reward.py`
- Create: `tests/unit/test_oracle.py`
- Create: `tests/regression/test_phase1_fixtures.py`

**Interfaces:**

- Consumes: public events, private terminal horizon for trace timing, local
  `trace_jitter` RNG, and Task 4 scoring.
- Produces: `TraceTimingSchedule`, `build_trace_timing_schedule`,
  `OraclePolicy`, `OracleTerminalKind`, `OracleSolution`, `OracleTraceStep`,
  `OracleTrace`, `solve_public_episode`, `build_oracle_trace`,
  `scale_oracle_trace`, `verify_oracle_truth`, and `oracle_actions`.

- [ ] **Step 1: Hand-author three independent fixtures and failing tests**

The fixture file contains one case per primary variant. The positive case uses
the canonical example:

```json
{
  "name": "positive_length_3",
  "facts": [
    ["link", 7, 2],
    ["hazard", 5, 3, 24.0],
    ["link", 1, 6],
    ["safe", 4],
    ["link", 2, 5],
    ["link", 9, 7],
    ["hazard", 6, 1, 24.0]
  ],
  "activation_node": 9,
  "activation_time": 100.0,
  "terminal_time": 124.0,
  "expected_path": [9, 7, 2, 5],
  "expected_terminal": "hazard",
  "expected_hazard_type": 3,
  "expected_window": [118.0, 121.6],
  "expected_target": 119.8,
  "expected_event_count": 9
}
```

Hand-author a length-2 safe case with eight total public facts and a length-2
disconnected case with no reachable terminal. Do not generate this fixture with
production generator or oracle code.

Tests assert:

```python
@pytest.mark.parametrize(
    ("path_length", "variant", "expected_events"),
    [(3, "positive", 9), (2, "safe_negative", 6), (2, "disconnected_negative", 4)],
)
def test_oracle_trace_lengths(path_length: int, variant: str, expected_events: int) -> None:
    case = load_fixture_case(variant)
    solution = solve_public_episode(case.public)
    trace = build_oracle_trace(
        case.public,
        solution,
        case.private_terminal,
        OracleTimingConfig(),
        local_generator(case.trace_seed_key),
    )
    assert len(trace.steps) == expected_events
```

Add tests for duplicate equivalent links, two reachable terminals, branching
primary paths, cycles, corrupt facts, corrupt truth, and record permutation.

- [ ] **Step 2: Run oracle tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_oracle.py tests/regression/test_phase1_fixtures.py
```

Expected: imports fail because `env.oracle` does not exist.

- [ ] **Step 3: Implement independent public graph solving**

`solve_public_episode(public: PublicEpisode, policy: OraclePolicy =
OraclePolicy.PRIMARY) -> OracleSolution` must:

1. index fact records without importing generator/invariant helpers;
2. start only from the activation payload;
3. follow exactly one legal outgoing link in primary data;
4. reject a repeated node, duplicate equivalent record, branch, or multiple
   reachable terminals with `OracleError`;
5. stop on a reachable hazard, reachable safe, or no outgoing link;
6. return ordered node and record IDs derived solely from public facts.

Use:

```python
class OraclePolicy(str, Enum):
    PRIMARY = "primary"
    BRANCHING = "branching"
    CONTRADICTION = "contradiction"


class OracleTerminalKind(str, Enum):
    HAZARD = "hazard"
    SAFE = "safe"
    DISCONNECTED = "disconnected"


@dataclass(frozen=True, slots=True)
class OracleTraceStep:
    trace_step_id: int
    parent_trace_step_id: int | None
    kind: InternalEventKind
    timestamp: float
    delta: float
    selected_record_id: int | None
    focus_before: int | None
    focus_after: int | None


@dataclass(frozen=True, slots=True)
class OracleSolution:
    terminal_kind: OracleTerminalKind
    node_path: tuple[int, ...]
    link_record_ids: tuple[int, ...]
    terminal_record_id: int | None
    hazard_type: int | None
    public_delay: float | None
    superseded_terminal_record_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class OracleTrace:
    solution: OracleSolution
    steps: tuple[OracleTraceStep, ...]
    actions: tuple[Action, ...]
```

`OraclePolicy.PRIMARY` is the default and enforces the primary unique-chain
contract. Tasks 11–12 add suite-specific policies without changing primary
semantics. `PublicEpisode` deliberately contains no suite metadata. Authorized
services select the policy from the requested/manifest evaluation suite using
this frozen mapping, then pass it explicitly:

```text
BRANCHING_STRESS                         -> BRANCHING
CONTRADICTION_STRESS                    -> CONTRADICTION
all primary/clock/other Phase 1 suites  -> PRIMARY
```

Task 5 unit tests call `PRIMARY` directly. Tasks 11 and 12 add direct tests for
`BRANCHING` and `CONTRADICTION` when those semantics are implemented. Service
tests prove that the mapping uses authorized suite metadata and never
`EpisodeTruth` or a public feature.

`verify_oracle_truth` compares the independently derived solution with private
truth and raises on any mismatch. Never repair either side.

- [ ] **Step 4: Implement exact oracle timing**

`build_oracle_trace` has this exact signature so competitive-record counts and
activation time come from public facts while all protocol fractions come from
the resolved timing configuration:

`build_oracle_trace(public: PublicEpisode, solution: OracleSolution,
terminal_horizon: ExternalEvent, timing: OracleTimingConfig,
rng: np.random.Generator) -> OracleTrace`.

The oracle independently selects records and computes the ordered
`competitive_counts`; it draws exactly one finite standard-normal jitter value
per non-action step from the supplied RNG, stores those values in an immutable
tuple, and passes that tuple to the frozen neutral
`build_trace_timing_schedule` API rather than reimplementing the equations.
The primitive never receives or calls an RNG. It first rejects every boolean,
wrong primitive type, non-finite input/timing scalar/jitter, unequal or empty
input sequence, negative competitive count, or non-positive delay. Inside that
pure primitive, for each non-action step `k`, compute:

```python
compose_deadline = math.fsum(
    (activation_time, timing.terminal_compose_fraction * delay)
)
elapsed = math.fsum(provisional_deltas)
current_time = math.fsum((activation_time, elapsed))
remaining_budget = compose_deadline - current_time
urgency = min(
    1.0,
    remaining_event_count * timing.delta_0
    / max(remaining_budget, timing.delta_min),
)
raw = timing.delta_0 * (1.0 + 0.15 * competitive_count) / (1.0 + 0.5 * urgency)
jittered = raw * math.exp(jitter_normals[k] * timing.jitter_log_std)
delta = min(timing.delta_max, max(timing.delta_min, jittered))
```

`remaining_event_count` includes the current non-action step. `current_time` is
the preceding step timestamp, or activation time for the first step;
`provisional_deltas` is the exact tuple of already computed unscaled deltas.
Validate every intermediate above with `math.isfinite` before using it; convert
`math.exp` overflow into the same typed infeasibility error rather than leaking
`OverflowError`.

A competitor is a non-teacher record of the same kind matching at least one
non-null selected-record field among subject, object, hazard type, or
`floor(log2(delay))`.

Use `math.fsum(provisional_deltas)` for the pre-scale total. If the sum of
non-action intervals exceeds
`timing.terminal_compose_fraction * delay`, multiply every interval by the
single factor `(timing.terminal_compose_fraction * delay) / total`. If any
scaled interval would be below `timing.delta_min`, raise
`OracleError("oracle trace is temporally infeasible")`.
Do not clamp individual intervals after common scaling because that would alter
their ratios. Use `math.fsum((activation_time, *final_deltas))` for terminal
composition, reject any non-finite scale/delta/timestamp, and derive the action
target with `math.fsum((activation_time,
timing.action_target_fraction * delay))`.

The primitive additionally validates its complete output: each final delta is
at least `delta_min`, terminal composition is no later than the configured
fraction, and the exact action target follows terminal composition. The
generator and invariant analyzer later derive their own selected records and
competitor counts and call the same math primitive without importing the
oracle.

Trace contract:

```text
positive:              2 * (L + 1) non-action steps, then ACT at
                       t0 + timing.action_target_fraction * d
safe-negative:         2 * (L + 1) non-action steps, no ACT
disconnected-negative: 2 * L non-action steps, no ACT
```

For a positive trace, the ACT delta is exactly
`(t0 + timing.action_target_fraction * d) - terminal_compose_timestamp`.

Each `RECALL`/`COMPOSE` step records its selected record ID and focus before and
after. Trace step IDs are zero-based and parent-linked in their own artifact
namespace. `ACT` records no memory selection and produces exactly one action
whose `caused_by_event_id` is the ACT trace step ID.
`scale_oracle_trace(trace, factor)` multiplies every trace timestamp/delta and
action timestamp, preserving kinds/order/record IDs; the pre-transform trace
continues to enforce `timing.delta_min`.

Add one non-default `OracleTimingConfig` test and require trace compression,
action target, scoring window, and minimum-duration generation to follow every
override. No timing algorithm may reintroduce the default numeric literals.

- [ ] **Step 5: Prove scoring and permutation invariance**

Score oracle actions for every fixture and assert exact 100% success. Reassign
fact event IDs after at least 100 deterministic permutations, remap expected
support by fact identity, and assert the terminal decision, hazard, delay, and
normalized action window remain unchanged.

- [ ] **Step 6: Run GREEN checks**

```bash
uv run pytest -q tests/unit/test_oracle.py tests/regression/test_phase1_fixtures.py tests/unit/test_reward.py
uv run ruff check src/silent_cascade/env/timing.py src/silent_cascade/env/oracle.py tests/unit/test_oracle.py tests/unit/test_reward.py tests/regression/test_phase1_fixtures.py
uv run ruff format --check src/silent_cascade/env/timing.py src/silent_cascade/env/oracle.py tests/unit/test_oracle.py tests/unit/test_reward.py tests/regression/test_phase1_fixtures.py
```

Expected: all oracle, fixture, and scoring tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/silent_cascade/env/timing.py src/silent_cascade/env/oracle.py \
  tests/fixtures/phase1/oracle_cases.json tests/unit/test_oracle.py \
  tests/unit/test_reward.py tests/regression/test_phase1_fixtures.py
git commit -m "feat: add independent OFD graph oracle"
```

---

### Task 6: Suite Specifications, Matched Templates, and Independent Allocation

**Files:**

- Create: `src/silent_cascade/env/generator.py`
- Create: `tests/unit/test_generator_spec.py`

**Interfaces:**

- Consumes: Tasks 1–5 config, types, local streams, and oracle timing contract.
- Produces: `SuiteSpec`, `CohortBlock`, `CohortAllocation`, `EpisodeBlock`,
  `ClockEpisodeBlock`, `IndependentAllocation`, `VALIDATION_ALLOCATION`,
  `PHASE1_GATE_ALLOCATION`, `CohortRequest`, `IndependentEpisodeRequest`,
  `CohortTemplate`, `suite_spec`, `iter_cohort_requests`,
  `iter_independent_requests`, `iter_phase1_gate_requests`, and
  `sample_cohort_template`, `validate_validation_allocation`, and
  `validate_phase1_gate_allocation`.

- [ ] **Step 1: Write failing suite/allocation/template tests**

```python
def test_phase1_gate_allocation_has_exact_frozen_blocks() -> None:
    requests = tuple(iter_independent_requests(PHASE1_GATE_ALLOCATION, root_seed=41))
    assert len(requests) == 100_000
    assert sum(request.suite is SuiteName.IID_PRIMARY for request in requests) == 24_000
    assert Counter((request.suite, request.requested_path_length) for request in requests) == {
        (SuiteName.IID_PRIMARY, 2): 8_000,
        (SuiteName.IID_PRIMARY, 3): 8_000,
        (SuiteName.IID_PRIMARY, 4): 8_000,
        (SuiteName.OOD_DEPTH, 5): 4_000,
        (SuiteName.OOD_DEPTH, 6): 4_000,
        (SuiteName.OOD_DEPTH, 7): 4_000,
        (SuiteName.OOD_DEPTH, 8): 4_000,
        (SuiteName.OOD_SHORT_DELAY, 2): 8_000,
        (SuiteName.OOD_SHORT_DELAY, 3): 8_000,
        (SuiteName.OOD_SHORT_DELAY, 4): 8_000,
        (SuiteName.OOD_LONG_DELAY, 2): 4_000,
        (SuiteName.OOD_LONG_DELAY, 3): 4_000,
        (SuiteName.OOD_LONG_DELAY, 4): 4_000,
        (SuiteName.DISTRACTOR_FLOOD, 2): 8_000,
        (SuiteName.DISTRACTOR_FLOOD, 3): 8_000,
        (SuiteName.DISTRACTOR_FLOOD, 4): 8_000,
    }
    assert all(
        Counter(item.variant for item in requests[offset : offset + 4])
        == Counter({EpisodeVariant.POSITIVE: 2,
                    EpisodeVariant.SAFE_NEGATIVE: 1,
                    EpisodeVariant.DISCONNECTED_NEGATIVE: 1})
        for offset in range(0, len(requests), 4)
    )


def test_template_contains_one_shared_unlabelled_link_topology(config: Phase1Config) -> None:
    request = cohort_request(SuiteName.DISTRACTOR_FLOOD, path_length=4)
    template = sample_cohort_template(config, request, attempt=0)
    assert template.requested_path_length == 4
    assert 16 <= len(template.distractor_edges) <= 48
    assert not set(template.relevant_nodes) & set(template.distractor_source_nodes)
```

Also assert the exact validation cohort starts `0,834,1667`, independent gate
episode starts, clock nested counts, exact quartet composition, suite axes,
inclusive integer/log-uniform real sampling, stable iteration under chunking,
one shared validation timestamp-gap template, and one validation two-class
hazard multiset. Construct valid 8- and 16-episode `DEBUG` allocations,
including a block whose first episode index is nonzero and not divisible by
four, and prove the generic iterators emit explicit member indices `0..3` and
accept them. Construct a generic `FROZEN` allocation with the canonical 20,000
total episodes but deliberately different block boundaries and prove structural
validation accepts it without importing the Phase 1 gate layout. Separately
prove the two production allocation validators reject those test allocations
and accept only their corresponding exact frozen constant.

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_generator_spec.py
```

Expected: import fails because `env.generator` does not exist.

- [ ] **Step 3: Implement suite selection, allocations, and templates**

`suite_spec` changes only the declared factor:

```text
VALIDATION/IID: path 2..4, delay 8..64, distractors 0..12
OOD_DEPTH:      path 5..8, delay 16..128, distractors 0..12
OOD_SHORT:      path 2..4, delay 1..8, distractors 0..12
OOD_LONG:       path 2..4, delay 64..1024, distractors 0..12
DISTRACTOR:     path 2..4, delay 8..64, distractors 16..48
```

Draw real ranges log-uniformly and integer ranges inclusively. A cohort template
contains the complete canonical unlabeled LINK topology—not merely its count—
plus requested path length, common episode delay, fact-gap sequence, activation
gap, and a two-item hazard-class multiset:

```python
class CohortBlock(StrictModel):
    suite: SuiteName
    requested_path_length: int
    first_cohort_index: int
    cohort_count: int


class EpisodeBlock(StrictModel):
    suite: SuiteName
    requested_path_length: int
    first_episode_index: int
    episode_count: int


class ClockEpisodeBlock(StrictModel):
    requested_path_length: int
    source_first_episode_index: int
    scale_0_1x_episode_count: int
    scale_10x_episode_count: int


class CohortAllocation(StrictModel):
    allocation_id: str
    split_namespace: SplitNamespace
    blocks: tuple[CohortBlock, ...]


class IndependentAllocation(StrictModel):
    allocation_id: str
    split_namespace: SplitNamespace
    blocks: tuple[EpisodeBlock, ...]
    clock_blocks: tuple[ClockEpisodeBlock, ...] = ()


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    path_lengths: tuple[int, ...]
    delay_log_uniform: tuple[float, float]
    distractor_link_records: tuple[int, int]


@dataclass(frozen=True, slots=True)
class CohortTemplate:
    requested_path_length: int
    relevant_nodes: tuple[int, ...]
    relevant_edges: tuple[tuple[int, int], ...]
    distractor_edges: tuple[tuple[int, int], ...]
    unreachable_terminal_nodes: tuple[int, int, int]
    episode_delay: float
    fact_gap_sequence: tuple[float, ...]
    activation_gap: float
    hazard_class_multiset: tuple[int, int]
```

Construct `VALIDATION_ALLOCATION` with blocks `(VALIDATION,2,0,834)`,
`(VALIDATION,3,834,833)`, and `(VALIDATION,4,1667,833)`. Construct
`PHASE1_GATE_ALLOCATION` with these exact episode
`(suite,path,first,count)` rows:

```text
IID_PRIMARY:        (2,0,8000), (3,8000,8000), (4,16000,8000)
OOD_DEPTH:          (5,0,4000), (6,4000,4000), (7,8000,4000), (8,12000,4000)
OOD_SHORT_DELAY:    (2,0,8000), (3,8000,8000), (4,16000,8000)
OOD_LONG_DELAY:     (2,0,4000), (3,4000,4000), (4,8000,4000)
DISTRACTOR_FLOOD:   (2,0,8000), (3,8000,8000), (4,16000,8000)
clock IID parents:  (2,0,1668,668), (3,8000,1668,668), (4,16000,1664,664)
```

The Pydantic allocation models are structurally reusable: they reject
overlapping same-suite indices, unsupported paths, nonpositive counts,
independent counts not divisible by four, and non-nested clock counts, but do
not hard-code a corpus total or require a block's first episode index to be
divisible by four. `validate_validation_allocation(allocation,
config)` and `validate_phase1_gate_allocation(allocation, config)` are separate
production guards. The former requires `VALIDATION_ALLOCATION` to contain
exactly 2,500 matched cohorts; the latter requires
`PHASE1_GATE_ALLOCATION` to contain exactly 100,000 independent episodes with
the frozen IDs, namespaces, blocks, clock subsets, and config values above.
Task 14/16 tests may inject 8–16-episode allocations with
`split_namespace=DEBUG` and an `allocation_id` beginning `test-`. Generic
iterators, generators, and pure DEBUG artifact builders accept them; only
production-facing wrappers call the frozen guards. Test allocations may
produce DEBUG artifacts under caller-owned temporary paths but may not produce
`VALIDATION` or `FROZEN_TEST` artifacts or reach the installed CLI. Production
dependencies call the appropriate frozen validator before generation and again
before publication.

```python
def iter_cohort_requests(
    allocation: CohortAllocation,
    root_seed: int,
) -> Iterator[CohortRequest]:
    for block in allocation.blocks:
        for offset in range(block.cohort_count):
            yield CohortRequest(
                split_namespace=allocation.split_namespace,
                suite=block.suite,
                root_seed=root_seed,
                cohort_index=block.first_cohort_index + offset,
                requested_path_length=block.requested_path_length,
            )
```

`iter_independent_requests` walks `EpisodeBlock`s identically at episode
granularity. Within each consecutive group of four local offsets inside one
block, it derives one `AllocationLabelKey`, assigns the selected `[P,P,S,D]`
permutation, and emits four requests with the same private
`allocation_quartet_index` and explicit `quartet_member_index` values `0..3`.
The variant is exactly `variants[quartet_member_index]`. A quartet never crosses
a suite/path block boundary, and quartet indices remain unique across the
allocation. Neither absolute episode-index alignment nor the assigned variant
is used to recover member identity. It is a fixed
stratified design: neither request order nor allocation labels are public, and
each request later receives disjoint episode-local nuisance streams.
`iter_phase1_gate_requests(root_seed)` is a thin exact wrapper around
`iter_independent_requests(PHASE1_GATE_ALLOCATION, root_seed)`; Phase 6 uses the
generic iterator with its own predeclared `IndependentAllocation`.

`sample_cohort_template` uses only cohort-scoped streams. It rejects any
distractor whose source is in the relevant component, any duplicate/cycle or
path shortcut, and any topology that cannot fit the fixed terminal records
within 64 facts. Member-local streams are not read in this task.

- [ ] **Step 4: Run GREEN checks**

```bash
uv run pytest -q tests/unit/test_generator_spec.py
uv run ruff check src/silent_cascade/env/generator.py tests/unit/test_generator_spec.py
uv run ruff format --check src/silent_cascade/env/generator.py tests/unit/test_generator_spec.py
```

- [ ] **Step 5: Commit**

```bash
git add src/silent_cascade/env/generator.py tests/unit/test_generator_spec.py
git commit -m "feat: define OFD suite allocations"
```

---

### Task 7: Matched Primary Cohort Construction

**Files:**

- Modify: `src/silent_cascade/env/generator.py`
- Create: `tests/unit/test_generator.py`

**Interfaces:**

- Consumes: `CohortRequest`/`CohortTemplate`, Task 2 episode types, and Task 3
  member streams/public IDs.
- Produces: `GenerationCohort`, `RejectionDiagnostic`,
  `generate_matched_cohort`, and `regenerate_matched_episode`.

```python
@dataclass(frozen=True, slots=True)
class RejectionDiagnostic:
    reason: str
    count: int


@dataclass(frozen=True, slots=True)
class GenerationCohort:
    request: CohortRequest
    accepted_attempt: int
    episodes: tuple[EpisodeBundle, EpisodeBundle, EpisodeBundle, EpisodeBundle]
    seed_tokens: tuple[str, ...]
    rejections: tuple[RejectionDiagnostic, ...]
```

`seed_tokens` is the exact 20-token accepted-attempt sequence frozen under
label-blind trace feasibility, including one `TRACE_JITTER` token per member.

Exact signatures are `generate_matched_cohort(config: Phase1Config, request:
CohortRequest, public_id_seed: int) -> GenerationCohort` and
`regenerate_matched_episode(config: Phase1Config, request: CohortRequest,
public_id_seed: int, expected_public_id: str, expected_accepted_attempt: int) ->
EpisodeBundle`.

- [ ] **Step 1: Write failing construction/rejection tests**

For one cohort, assert the exact 2/1/1 variant composition, shared delay/path/
distractor count/timestamp gaps, equal LINK topology signatures after removing
node labels, two hazards plus one safe, independent node/presentation
permutations, contiguous post-permutation FACT IDs, and public IDs allocated
from the accepted attempt. Spy on all four member draws and prove each evaluates
the complete positive/safe/disconnected counterfactual family with identical
member-local nuisance values and a reinitialized, byte-identical
`TRACE_JITTER` initial state before selecting its assigned label; each candidate
draws exactly its own required step count, so differently sized traces share an
identical jitter prefix rather than an artificially padded vector. Inject a
failure in
every retry-causing semantic, shape, leakage-prevention, and timing check for an
unassigned candidate and require the whole cohort—not one member—to retry
exactly once. Exhaustion at 1,000 raises
`GenerationError` with only an opaque cohort hash and aggregate reasons.

```python
def test_matched_cohort_has_exact_primary_composition(config: Phase1Config) -> None:
    cohort = generate_matched_cohort(config, cohort_request(index=17), public_id_seed=91)
    assert Counter(bundle.truth.recipe.variant for bundle in cohort.episodes) == {
        EpisodeVariant.POSITIVE: 2,
        EpisodeVariant.SAFE_NEGATIVE: 1,
        EpisodeVariant.DISCONNECTED_NEGATIVE: 1,
    }
    assert {bundle.truth.recipe.accepted_attempt for bundle in cohort.episodes} == {
        cohort.accepted_attempt
    }
```

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_generator.py
```

Expected: construction APIs are absent.

- [ ] **Step 3: Implement bounded cohort construction**

For each attempt up to `max_generation_attempts`:

1. sample one unlabeled cohort template, shared delay/path/distractor/timestamp
   nuisances, and the allocation-only permutation of `[P,P,S,D]` without
   materializing a selected-label episode;
2. for each member, sample once its bijection over all 64 entity IDs,
   terminal-class/delay multiset and presentation permutation source from that
   member's streams; retain the `TRACE_JITTER` key, not a variant-sized draw;
3. for that member draw, construct complete positive, safe-negative, and
   disconnected-negative candidates from the same sampled nuisance values;
   reinitialize the same `TRACE_JITTER` stream state for each candidate and draw
   exactly that candidate's finite step-count vector, so shared positions are
   identical and one candidate cannot advance another's stream;
4. for each of those three candidates, place exactly two hazards and one safe,
   apply the same node/presentation permutations, assign provisional contiguous
   post-presentation FACT IDs, bind the same strictly increasing gap template,
   and derive private terminal/window truth;
5. run the complete generator-local retryable acceptance pipeline on every
   candidate: record/terminal counts, capacity and schema shape, unique
   reachability/path/terminal semantics, absence of shortcut/answer-correlated
   structure, timestamp/order rules, and the exact jittered trace schedule;
6. if any candidate for any member fails, append exactly one stable rejection
   reason for this cohort attempt, discard the entire four-member attempt, and
   continue; never count one rejection per failing member/candidate;
7. only after all twelve candidates pass, use the frozen label permutation to
   select one candidate per member, set `accepted_attempt=attempt`, and discard
   the unselected counterfactual materializations;
8. allocate the four public IDs from the accepted attempt, bind the immutable
   selected bundles, and return the cohort.

Task 8 subsequently inserts the independent episode/tuple validation call
after step 8. It is deliberately not imported in the Task 7 commit before the
module exists. Any later invariant disagreement is fatal, not another retry.

`regenerate_matched_episode` reconstructs the whole cohort from attempt zero,
requires the persisted accepted attempt and requested public ID to match, then
selects exactly one member. No online Bernoulli training API is exported;
Task 9 adds the separately named independent primitive used by gate/frozen
allocations.

Every bundle's rejection-reason sequence contains one stable reason for each
rejected cohort construction draw represented by its `rejection_count`; a
failed cohort attempt contributes one reason even if several member/candidate
checks fail. Do not store only distinct reason keys. On exhaustion, raise `GenerationError` with an opaque SHA-256 of the private
cohort key, attempt count, and aggregated rejection reasons. Its serializable
context never contains root seed, split, index, or variant. Do not expose richer
internal diagnostics through a public episode callback or inspection result.

- [ ] **Step 4: Run GREEN checks**

```bash
uv run pytest -q tests/unit/test_generator_spec.py tests/unit/test_generator.py tests/unit/test_oracle.py
uv run ruff check src/silent_cascade/env/generator.py tests/unit/test_generator.py
uv run ruff format --check src/silent_cascade/env/generator.py tests/unit/test_generator.py
```

- [ ] **Step 5: Commit**

```bash
git add src/silent_cascade/env/generator.py tests/unit/test_generator.py
git commit -m "feat: generate matched OFD cohorts"
```

---

### Task 8: Independent Invariants and Generator Properties

**Files:**

- Create: `src/silent_cascade/env/invariants.py`
- Modify: `src/silent_cascade/env/generator.py`
- Create: `tests/unit/test_invariants.py`
- Create: `tests/property/test_generator_properties.py`
- Modify: `tests/regression/test_import_boundaries.py`

**Interfaces:**

- Consumes: Task 2 episode artifacts and tuples of Task 7 bundles, without
  importing `env.generator`, its graph helpers, or the oracle.
- Produces: `InvariantReport`, `validate_episode_invariants`, and
  `validate_cohort_invariants`; Task 7 generator calls them only after this
  task introduces them.

```python
@dataclass(frozen=True, slots=True)
class InvariantReport:
    valid: bool
    fact_count: int
    link_count: int
    hazard_count: int
    safe_count: int
    reachable_node_count: int
    reachable_terminal_count: int
    requested_path_length: int
    check_ids: tuple[str, ...]
```

Exact signatures are `validate_episode_invariants(bundle: EpisodeBundle,
config: Phase1Config, strict: bool = True) -> InvariantReport` and
`validate_cohort_invariants(episodes: tuple[EpisodeBundle, EpisodeBundle,
EpisodeBundle, EpisodeBundle], config: Phase1Config,
strict: bool = True) -> tuple[InvariantReport, InvariantReport,
InvariantReport, InvariantReport]`.

- [ ] **Step 1: Write failing mutation/property/import tests**

Add one mutation per independent check: reachable second terminal, shortcut,
branch, wrong requested length, duplicate fact, reachable-source distractor,
bad terminal counts, unequal delays, capacity overflow in primary, corrupted
truth path/window/private terminal, nonmonotonic time, invalid IDs, and trace
ambiguity. Each must raise `EpisodeInvariantError`. Hypothesis samples primary
suites/seeds and asserts finite bounds, exact matching, order/chunk/global-RNG
independence, and no more than 64 facts. Add the import tests below before
production edits.

```python
def test_generator_and_invariants_do_not_import_oracle() -> None:
    assert "silent_cascade.env.oracle" not in imported_modules("silent_cascade.env.generator")
    assert "silent_cascade.env.oracle" not in imported_modules("silent_cascade.env.invariants")
    assert "silent_cascade.env.generator" not in imported_modules("silent_cascade.env.invariants")
```

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_invariants.py tests/property/test_generator_properties.py tests/regression/test_import_boundaries.py
```

Expected: invariants imports/checks fail.

- [ ] **Step 3: Implement an independent invariant analyzer**

`env.invariants` parses facts independently from both generator and oracle. It
recomputes reachable nodes, simple paths, terminal reachability, record counts,
path length, distractor legality, timing, memory capacity, private truth/window
consistency, and unique record-level trace. It returns a frozen aggregate
report and raises on the first invalid episode in strict mode.

For independent bundles it authenticates the allocation label from the
explicit `(suite, requested_path_length, allocation_quartet_index,
quartet_member_index)` tuple and the `AllocationLabelKey`. It never uses
`episode_index % 4`, an assigned-variant rank, or Phase 1 gate block boundaries.
Allocation-owning services—not generic episode invariants—authenticate whether
an episode index belongs to a declared block. It also independently reconstructs
the assigned episode's selected records, competitor counts, exact jitter stream,
and final trace schedule through the neutral timing primitive.

`validate_cohort_invariants` additionally requires exact shared path length,
delay, distractor count, observation duration/gap template, activation gap,
hazard class/delay multiset, total/per-kind counts, and LINK-only
entity/component/degree signature across the four variants. It also verifies
that presentation and node permutations came from member-local stream tokens,
not construction order.

After generator-local semantic checks pass, the generator allocates IDs, binds
complete bundles, and calls both validators. Invariant disagreement raises
immediately and is never counted as a rejected random draw. Because the cohort
validator accepts only the tuple type from `env.episode`, `env.invariants`
never imports `GenerationCohort` or `env.generator`; dependency direction is
one-way. The oracle-side import regression additionally requires that
`env.oracle` imports neither generator nor invariants.

- [ ] **Step 4: Run GREEN checks**

```bash
uv run pytest -q tests/unit/test_invariants.py tests/unit/test_generator.py tests/property/test_generator_properties.py tests/regression/test_import_boundaries.py
uv run ruff check src/silent_cascade/env/generator.py src/silent_cascade/env/invariants.py tests/unit/test_invariants.py tests/property/test_generator_properties.py tests/regression/test_import_boundaries.py
uv run ruff format --check src/silent_cascade/env/generator.py src/silent_cascade/env/invariants.py tests/unit/test_invariants.py tests/property/test_generator_properties.py tests/regression/test_import_boundaries.py
```

- [ ] **Step 5: Commit**

```bash
git add src/silent_cascade/env/generator.py src/silent_cascade/env/invariants.py \
  tests/unit/test_invariants.py tests/property/test_generator_properties.py \
  tests/regression/test_import_boundaries.py
git commit -m "feat: validate OFD generation independently"
```

---

### Task 9: Independent Claim-Data Episode Generation

**Files:**

- Modify: `src/silent_cascade/env/generator.py`
- Modify: `src/silent_cascade/env/invariants.py`
- Create: `tests/unit/test_independent_generator.py`
- Modify: `tests/property/test_generator_properties.py`

**Interfaces:**

- Consumes: Task 2 independent coordinates, Task 3 independent seed/ID paths,
  Task 6 `IndependentAllocation`, and Task 8 episode invariants.
- Produces: `generate_independent_episode`,
  `regenerate_independent_episode`, and `independent_seed_tokens`. These are the
  only semantic episode-generation primitives Phase 6 may use for frozen
  primary claim data.

Exact signatures are:

```python
def generate_independent_episode(
    config: Phase1Config,
    request: IndependentEpisodeRequest,
    public_id_seed: int,
) -> EpisodeBundle: ...


def regenerate_independent_episode(
    config: Phase1Config,
    request: IndependentEpisodeRequest,
    public_id_seed: int,
    expected_public_id: str,
    expected_accepted_attempt: int,
) -> EpisodeBundle: ...


def independent_seed_tokens(
    request: IndependentEpisodeRequest,
    accepted_attempt: int,
) -> tuple[str, ...]: ...
```

`independent_seed_tokens` returns exactly seven tokens in the frozen order,
including the accepted attempt's `TRACE_JITTER` token.

- [ ] **Step 1: Write failing independent-allocation and construction tests**

Assert the exact 100,000-request suite/path/variant allocation, monotonic
episode indices, explicit quartet member indices, quartet label composition,
and clock-parent counts. For each
generated episode assert exact terminal counts, requested reachability/path,
fact capacity, post-permutation IDs, private/public separation, oracle truth,
and independent invariants. Record every RNG key consumed by four requests in
one allocation quartet and require that all nuisance keys contain their own
episode index, no counter token is shared, and no cohort-scoped sampler is
called.

Force attempts `0..2` invalid for one request and assert that only that request
retries; the other three quartet members retain attempt zero and identical
hashes. Assert exact regeneration by accepted attempt/public ID, failure on a
mismatch, global-RNG isolation, forward/reverse/chunks `1,3,7`, and a small
fresh-process check under `PYTHONHASHSEED=0/1`.
For every independent nuisance draw, spy on all three counterfactual candidates
and require identical template/structure, node permutation, terminal
class/delay multiset, timestamps, presentation-order source, and reinitialized
`TRACE_JITTER` state/common prefix. A retryable semantic, shape,
leakage-prevention, or timing failure in an unassigned candidate must reject
that episode attempt
before the requested label is selected. A post-ID invariant mismatch must raise
immediately and must not increment the rejection sequence.

```python
def test_independent_quartet_shares_labels_but_no_nuisance_seed(
    config: Phase1Config,
) -> None:
    requests = tuple(first_gate_quartet(root_seed=41))
    assert Counter(item.variant for item in requests) == {
        EpisodeVariant.POSITIVE: 2,
        EpisodeVariant.SAFE_NEGATIVE: 1,
        EpisodeVariant.DISCONNECTED_NEGATIVE: 1,
    }
    token_sets = [
        set(independent_seed_tokens(item, accepted_attempt=0))
        for item in requests
    ]
    assert all(left.isdisjoint(right)
               for index, left in enumerate(token_sets)
               for right in token_sets[index + 1 :])
```

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_independent_generator.py tests/property/test_generator_properties.py
```

Expected: the independent construction/regeneration APIs are absent.

- [ ] **Step 3: Implement bounded episode-local construction**

For each request and attempt up to `max_generation_attempts`:

1. authenticate the request's explicit member and variant against
   `allocate_independent_variants(AllocationLabelKey(...))`, then derive only
   `IndependentCounterSeedKey`s containing that episode index; retain the
   authenticated assigned label outside candidate construction and every
   retryable acceptance predicate until step 7;
2. sample once the suite-specific delay, distractor count, complete unlabeled
   LINK topology, observation/activation gaps, 64-entity bijection,
   terminal-class/delay multiset and presentation permutation source; retain
   the exact `TRACE_JITTER` key;
3. construct complete positive, safe-negative, and disconnected-negative
   candidates from those identical nuisance values, reinitializing the same
   `TRACE_JITTER` stream for each candidate and drawing its exact finite trace
   length (with identical shared prefix), and applying the same node and
   presentation permutations;
4. for each candidate, place exactly two hazards and one safe, assign
   provisional contiguous post-presentation FACT/activation/private-terminal
   IDs, and derive its private scorer truth independently;
5. before selecting the requested label, run every retry-causing
   generator-local semantic, shape, leakage-prevention, and timing check on all
   three candidates: exact counts/capacity/schema, unique reachability/path,
   absence of a second terminal/shortcut/ambiguity/answer-correlated cue,
   timestamp/order rules, independently derived record/competitor inputs, and
   `build_trace_timing_schedule`;
6. if any candidate fails, append exactly one stable rejection reason for this
   episode attempt, discard only this episode attempt, and increment its
   attempt; do not allocate an ID or expose which counterfactual failed;
7. only after all three candidates pass, select the request-assigned variant,
   set `accepted_attempt=attempt`, and discard the other materializations;
8. allocate the single public ID only after the selected draw passes, construct
   an `IndependentEpisodeCoordinate` carrying `quartet_member_index`, and run
   Task 8's independent invariant analyzer. An invariant disagreement is a code
   defect and raises immediately rather than becoming another sampling retry.

The variant and member position come only from `IndependentEpisodeRequest` and
must match the allocation-label permutation; generation must never infer,
rank-recover, or resample them. No nuisance seed contains
`allocation_quartet_index`, so
quartets create exact labels but no shared template, timing, class, or
presentation variable. An exhausted request raises a privacy-safe
`GenerationError` containing an opaque request hash and aggregate rejection
reasons, not the root, episode index, or variant.

- [ ] **Step 4: Prove future-frozen recipe equivalence**

Add a property test that clones independent requests from `PHASE1_GATE` to
`FROZEN` while holding all other typed fields explicit and verifies that the
same code path, invariant set, serialization schema, and regeneration API are
used; namespace separation must change seed/ID tokens. There is no
`generate_frozen_episode` function. Phase 6 may construct an
`IndependentAllocation` with its predeclared suite counts and frozen roots,
then call this primitive only. Add a 20,000-episode-total `FROZEN` allocation
with different valid block boundaries and prove invariants authenticate the
explicit quartet/member/variant relationship without consulting the Phase 1
gate block table.

- [ ] **Step 5: Run GREEN checks**

```bash
uv run pytest -q tests/unit/test_independent_generator.py tests/unit/test_invariants.py tests/property/test_generator_properties.py
uv run ruff check src/silent_cascade/env/generator.py src/silent_cascade/env/invariants.py tests/unit/test_independent_generator.py tests/property/test_generator_properties.py
uv run ruff format --check src/silent_cascade/env/generator.py src/silent_cascade/env/invariants.py tests/unit/test_independent_generator.py tests/property/test_generator_properties.py
```

Expected: exact allocation, episode-local key, invariant, regeneration,
order/chunk/process, and Ruff checks pass.

- [ ] **Step 6: Commit**

```bash
git add src/silent_cascade/env/generator.py src/silent_cascade/env/invariants.py \
  tests/unit/test_independent_generator.py tests/property/test_generator_properties.py
git commit -m "feat: generate independent OFD claim episodes"
```

---

### Task 10: Paired Clock Transforms and Source-Key Semantics

**Files:**

- Modify: `src/silent_cascade/env/episode.py`
- Modify: `src/silent_cascade/env/oracle.py`
- Create: `tests/unit/test_clock_scaling.py`
- Modify: `tests/property/test_generator_properties.py`

**Interfaces:**

- Consumes: Task 2 source key/recipe, Task 5 `scale_oracle_trace`, and the Task
  6 clock block selection.
- Produces: `scale_episode_time` and immutable parent-linkage integration;
  verifies and reuses Task 5's generic `scale_oracle_trace` implementation.

- [ ] **Step 1: Write failing key/time/decision tests**

Require only IID/unscaled parents, exact suite-to-factor mapping, preserved
source key, new child public ID, parent ID/hash, every absolute timestamp and
delay multiplied by the factor, unchanged graph/order/decision, identical
normalized windows, and rejection of a clock child as another parent.

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_clock_scaling.py
```

Expected: scaling APIs lack the new semantics.

- [ ] **Step 3: Implement paired clock transforms**

`scale_episode_time(bundle, target_suite, paired_public_id)` accepts only
`CLOCK_SCALE_0_1X` or `CLOCK_SCALE_10X` and maps them to factors `0.1` and
`10.0`. The source recipe must be unscaled `IID_PRIMARY`; otherwise raise
`EpisodeInvariantError`. Multiply fact timestamps, activation time, fact
delays, private terminal time, and window/target relative to the zero-time
origin. Preserve `EpisodeTruth.key` as the unscaled semantic source key, graph
facts, record order, variant, support, and normalized window ratios. The child
recipe sets its evaluation suite, scale, parent public ID, and parent episode
SHA-256; the child public ID is the supplied independently derived paired ID.
Transform the separate oracle trace with `scale_oracle_trace`. The source
bundle remains immutable, and manifest regeneration verifies the parent before
the child. Gate/frozen clock children derive that ID with
`IndependentPublicIdKey` using the target clock suite and the parent's episode
index/accepted attempt, so the ID namespace is disjoint without resampling any
semantic fact.

- [ ] **Step 4: Run GREEN checks**

```bash
uv run pytest -q tests/unit/test_clock_scaling.py tests/unit/test_episode.py tests/unit/test_oracle.py tests/property/test_generator_properties.py
uv run ruff check src/silent_cascade/env/episode.py src/silent_cascade/env/oracle.py tests/unit/test_clock_scaling.py tests/property/test_generator_properties.py
uv run ruff format --check src/silent_cascade/env/episode.py src/silent_cascade/env/oracle.py tests/unit/test_clock_scaling.py tests/property/test_generator_properties.py
```

Expected: clock, episode, oracle, and property tests pass; Ruff is clean.

- [ ] **Step 5: Commit**

```bash
git add src/silent_cascade/env/episode.py src/silent_cascade/env/oracle.py \
  tests/unit/test_clock_scaling.py tests/property/test_generator_properties.py
git commit -m "feat: add paired OFD clock transforms"
```

---

### Task 11: Structural Adversarial Data Suites

**Files:**

- Modify: `src/silent_cascade/env/generator.py`
- Modify: `src/silent_cascade/env/invariants.py`
- Modify: `src/silent_cascade/env/oracle.py`
- Create: `tests/unit/test_stress_graphs.py`
- Modify: `tests/unit/test_oracle.py`
- Modify: `tests/property/test_generator_properties.py`

**Interfaces:**

- Consumes: Task 9's episode-local primitive and `StressDataConfig`.
- Produces: `generate_stress_episode`/`regenerate_stress_episode` for branching,
  irrelevant-cycle, memory-overflow, and null-near-miss
  generation/validation plus facts-derived branching/cycle oracle policies.
  Runtime stress behavior and eviction remain deferred.

Exact signatures are:

```python
def generate_stress_episode(
    config: Phase1Config,
    request: IndependentEpisodeRequest,
    public_id_seed: int,
) -> EpisodeBundle: ...


def regenerate_stress_episode(
    config: Phase1Config,
    request: IndependentEpisodeRequest,
    public_id_seed: int,
    expected_public_id: str,
    expected_accepted_attempt: int,
) -> EpisodeBundle: ...
```

- [ ] **Step 1: Write failing structural-suite tests**

Each suite test asserts its intended difference and every unchanged primary
invariant:

```python
def test_irrelevant_cycle_is_outside_activation_reachability(config: Phase1Config) -> None:
    bundle = generate_stress_episode(
        config, stress_request(SuiteName.CYCLES_STRESS), public_id_seed=91
    )
    cycle_nodes = find_directed_cycles(bundle.public)
    assert cycle_nodes
    assert all(node not in activation_reachable_nodes(bundle.public) for node in cycle_nodes)


def test_overflow_metadata_is_private_and_eviction_is_not_run(config: Phase1Config) -> None:
    bundle = generate_stress_episode(
        config,
        stress_request(SuiteName.MEMORY_OVERFLOW_STRESS),
        public_id_seed=91,
    )
    assert 65 <= len(fact_events(bundle.public)) <= 80
    assert bundle.truth.stress_metadata.over_capacity_record_count == len(fact_events(bundle.public))
    assert "over_capacity" not in public_primitive(bundle)
```

- [ ] **Step 2: Run tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_stress_graphs.py
```

Expected: suite selection raises unsupported-suite errors.

- [ ] **Step 3: Implement structural stress transforms**

- `BRANCHING_STRESS`: add 2–6 activation-reachable dead-end branches while
  retaining exactly one hazard-reaching path.
- `CYCLES_STRESS`: add a 2–6 node directed cycle whose every source remains
  outside the activation-reachable component.
- `MEMORY_OVERFLOW_STRESS`: generate 65–80 valid records and mark the recipe
  over-capacity; do not implement eviction.
- `NULL_NEAR_MISS_STRESS`: end a negative activation path at a node with no
  terminal and place a hazard on one distinct node that would require exactly
  one missing edge to reach.
Every stress request has an `IndependentEpisodeCoordinate` and uses disjoint
episode-local nuisance streams. Stress generation extends the same bounded
retry/regeneration path as Task 9; no stress cohort or shared nuisance template
is exported.
Stress validators explicitly distinguish intended violations from invalid
episodes. Primary-suite validators continue rejecting every stress-only shape.
`OraclePolicy.BRANCHING` enumerates simple activation-rooted paths and requires
exactly one reachable terminal solution while allowing dead-end branches.
Cycles are ignored only when every cycle node is outside activation
reachability. Policy selection comes from authorized suite metadata, never
private truth; ambiguity always raises `OracleError`.

- [ ] **Step 4: Run GREEN checks**

```bash
uv run pytest -q tests/unit/test_stress_graphs.py tests/unit/test_oracle.py tests/property/test_generator_properties.py
uv run ruff check src/silent_cascade/env/generator.py src/silent_cascade/env/invariants.py src/silent_cascade/env/oracle.py tests/unit/test_stress_graphs.py tests/unit/test_oracle.py tests/property/test_generator_properties.py
uv run ruff format --check src/silent_cascade/env/generator.py src/silent_cascade/env/invariants.py src/silent_cascade/env/oracle.py tests/unit/test_stress_graphs.py tests/unit/test_oracle.py tests/property/test_generator_properties.py
```

Expected: stress and existing primary properties pass.

- [ ] **Step 5: Commit**

```bash
git add src/silent_cascade/env/generator.py src/silent_cascade/env/invariants.py \
  src/silent_cascade/env/oracle.py tests/unit/test_stress_graphs.py \
  tests/unit/test_oracle.py tests/property/test_generator_properties.py
git commit -m "feat: add structural OFD stress suites"
```

---

### Task 12: Terminal and Timing Adversarial Suites

**Files:**

- Modify: `src/silent_cascade/env/episode.py`
- Modify: `src/silent_cascade/env/generator.py`
- Modify: `src/silent_cascade/env/invariants.py`
- Modify: `src/silent_cascade/env/oracle.py`
- Create: `tests/unit/test_stress_terminals.py`
- Modify: `tests/unit/test_episode.py`
- Modify: `tests/unit/test_oracle.py`

**Interfaces:**

- Consumes: Task 2 private stress metadata, Task 5 interval scheduler, and valid
  Task 11 stress dispatch.
- Produces: contradiction, minimum-duration, and checkpoint-scenario data;
  `OraclePolicy.CONTRADICTION`; and exact private stress diagnostics.

- [ ] **Step 1: Write failing terminal/timing tests**

```python
def test_minimum_duration_is_exactly_above_reference_boundary(config: Phase1Config) -> None:
    bundle = generate_stress_episode(
        config,
        stress_request(SuiteName.MINIMUM_DURATION_STRESS),
        public_id_seed=91,
    )
    boundary = bundle.truth.stress_metadata.minimum_feasible_delay
    delay = bundle.truth.episode_delay
    assert member_is_feasible(bundle, delay)
    assert not member_is_feasible(bundle, math.nextafter(boundary, -math.inf))
    assert delay - boundary >= config.stress.minimum_duration_epsilon
    assert delay - boundary <= config.stress.minimum_duration_epsilon + 4 * math.ulp(delay)
```

Also test 2–4 explicit stale/current terminal records, lexicographic current
selection, superseded record IDs in `OracleSolution`, ambiguity refusal, a
checkpoint pause strictly between oracle events, private serialization round
trip, and absence of every stress diagnostic from public projection/default
inspection.

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_stress_terminals.py tests/unit/test_episode.py tests/unit/test_oracle.py
```

Expected: terminal/timing stress suites are unsupported.

- [ ] **Step 3: Implement contradiction and checkpoint scenarios**

- `CONTRADICTION_STRESS` adds 2–4 terminal records for one node with distinct
  public times/confidences. `OraclePolicy.CONTRADICTION` groups records by node,
  selects the greatest `(observed_at, confidence, event_id)`, and returns every
  other record ID in `superseded_terminal_record_ids`; it never overwrites a
  fact.
- `CHECKPOINT_STRESS` uses a valid long-delay episode and stores one pause time
  at the midpoint of a deterministically selected adjacent oracle-event pair.
  It creates no checkpoint and performs no resume in Phase 1.

- [ ] **Step 4: Implement the exact minimum-feasible-delay search**

Pre-sample one complete episode-local jitter vector independent of candidate
delay. Recompute the ordinary urgency/difficulty raw interval vector `q(d)` and
define:

```text
budget_required(d) = timing.delta_min * sum(q(d)) / min(q(d))
slack(d) = timing.terminal_compose_fraction * d - budget_required(d)
```

Start at the largest theoretical floor
`N * delta_min / terminal_compose_fraction`; double an upper bound until the
episode is feasible, refusing above configured `64.0`. Evaluate 257 fixed
grid points and reject the draw if its feasibility predicate flips from true
back to false. Then bisect float64 values until
`math.nextafter(lo, math.inf) == hi`. Set the boundary to `hi` and the shared
episode delay to `math.nextafter(hi + epsilon, math.inf)`. Rebuild its trace and
require all deltas at least `timing.delta_min`, terminal composition by
`t0 + timing.terminal_compose_fraction * d`, and ACT at
`t0 + timing.action_target_fraction * d`; otherwise reject the draw. Store `hi`
only in private
stress metadata.

- [ ] **Step 5: Run GREEN checks**

```bash
uv run pytest -q tests/unit/test_stress_terminals.py tests/unit/test_episode.py tests/unit/test_oracle.py
uv run ruff check src/silent_cascade/env/episode.py src/silent_cascade/env/generator.py src/silent_cascade/env/invariants.py src/silent_cascade/env/oracle.py tests/unit/test_stress_terminals.py tests/unit/test_episode.py tests/unit/test_oracle.py
uv run ruff format --check src/silent_cascade/env/episode.py src/silent_cascade/env/generator.py src/silent_cascade/env/invariants.py src/silent_cascade/env/oracle.py tests/unit/test_stress_terminals.py tests/unit/test_episode.py tests/unit/test_oracle.py
```

- [ ] **Step 6: Commit**

```bash
git add src/silent_cascade/env/episode.py src/silent_cascade/env/generator.py \
  src/silent_cascade/env/invariants.py src/silent_cascade/env/oracle.py \
  tests/unit/test_stress_terminals.py tests/unit/test_episode.py \
  tests/unit/test_oracle.py
git commit -m "feat: add terminal and timing stress suites"
```

---

### Task 13: Immutable Manifest Schema and Publication

**Files:**

- Create: `src/silent_cascade/provenance.py`
- Create: `src/silent_cascade/logging/manifest.py`
- Create: `tests/unit/test_provenance.py`
- Create: `tests/unit/test_manifest.py`

**Interfaces:**

- Consumes: Phase 0 hashing/no-clobber I/O, Task 2 artifact identities, source
  revision, resolved configuration hash, and independent public ID seed.
- Produces from `provenance`: `SourceTreeFingerprint`,
  `EvidenceProvenance`, `AcceptedAttemptRun`,
  `ConstructionNamespaceEvidence`, `GENERATOR_SOURCE_PATHS`,
  `PHASE1_ANALYSIS_SOURCE_PATHS`, `TASK14_ANALYSIS_SOURCE_PATHS`,
  `source_tree_sha256`, `EvidenceProvenanceCollector`,
  `collect_evidence_provenance`, and `collect_final_phase1_provenance`.
- Produces from `logging.manifest`: `ManifestAccessClass`,
  `ManifestCoordinate`, `EpisodeManifestEntry`, `EpisodeManifest`, `ManifestEnvelope`,
  `ManifestPublication`, `publish_manifest`, `load_manifest`, and
  `verify_manifest`.

- [ ] **Step 1: Write failing manifest and concurrency tests**

Cover canonical round trips, strict schema rejection, payload tampering,
per-episode hash mismatch, duplicate IDs/recipes, wrong config/generator
version, immutable reuse, divergent collision, and access control:

```python
def test_identical_manifest_is_verified_without_rewrite(tmp_path: Path, manifest: EpisodeManifest) -> None:
    path = tmp_path / "validation.json"
    first = publish_manifest(path, manifest)
    before = path.stat().st_mtime_ns
    second = publish_manifest(path, manifest)
    assert first.file_sha256 == second.file_sha256
    assert not second.created
    assert path.stat().st_mtime_ns == before


def test_divergent_existing_manifest_requires_new_version(tmp_path: Path, manifest: EpisodeManifest) -> None:
    path = tmp_path / "validation.json"
    publish_manifest(path, manifest)
    with pytest.raises(ManifestError, match="different immutable manifest already exists"):
        publish_manifest(path, replace_manifest_root_seed(manifest, 99))
```

Use two spawned processes racing to publish the same and then divergent bytes.
Exactly one divergent payload may win; readers must observe a complete verified
envelope, never a mixture.

- [ ] **Step 2: Run tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_provenance.py tests/unit/test_manifest.py
```

Expected: import fails because the manifest module does not exist.

- [ ] **Step 3: Define strict manifest content**

```python
class ManifestAccessClass(str, Enum):
    FIXTURE = "fixture"
    DEBUG = "debug"
    VALIDATION = "validation"
    FROZEN_TEST = "frozen_test"


HexDigest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class AcceptedAttemptRun(StrictModel):
    first_draw_index: int = Field(ge=0)
    draw_count: int = Field(gt=0)
    accepted_attempt: int = Field(ge=0, lt=1_000)


class ConstructionNamespaceEvidence(StrictModel):
    schema_version: Literal["construction-namespace-evidence-v1"]
    generation_mode: Literal["matched", "independent"]
    public_id_seed: int = Field(ge=0, lt=2**128)
    accepted_draw_count: int = Field(gt=0)
    accepted_attempt_runs: tuple[AcceptedAttemptRun, ...]
    rejected_draw_count: int = Field(ge=0)
    generation_attempt_count: int = Field(gt=0)
    seed_token_count: int = Field(gt=0)
    seed_token_sequence_sha256: HexDigest
    seed_token_collision_count: Literal[0]
    base_public_id_count: int = Field(gt=0)
    clock_public_id_count: int = Field(ge=0)
    total_public_id_count: int = Field(gt=0)
    public_id_sequence_sha256: HexDigest
    public_id_collision_count: Literal[0]


class SourceTreeFingerprint(StrictModel):
    frame_version: Literal["sc-source-tree-v1"]
    scope: Literal[
        "generator", "phase1_analysis", "phase1_task14_analysis"
    ]
    paths: tuple[str, ...]
    sha256: HexDigest


class EvidenceProvenance(StrictModel):
    schema_version: Literal["phase1-evidence-provenance-v1"]
    plan_base_revision: str = Field(min_length=7)
    source_commit: str = Field(min_length=7)
    source_dirty: bool
    generator_version: Literal["ofd-v1"]
    generation_mode: Literal["matched", "independent"]
    allocation_id: str = Field(min_length=1)
    split_namespace: SplitNamespace
    config_sha256: HexDigest
    generator_source: SourceTreeFingerprint
    analysis_source: SourceTreeFingerprint
    root_seed: int
    public_id_seed_sha256: HexDigest
    analysis_seeds: dict[str, int] = Field(default_factory=dict)
    foundation_model_calls: Literal[0] = 0


class MatchedManifestCoordinate(StrictModel):
    mode: Literal["matched"] = "matched"
    cohort_index: int = Field(ge=0)
    member_index: int = Field(ge=0, le=3)


class IndependentManifestCoordinate(StrictModel):
    mode: Literal["independent"] = "independent"
    episode_index: int = Field(ge=0)
    allocation_quartet_index: int = Field(ge=0)
    quartet_member_index: int = Field(ge=0, le=3)


type ManifestCoordinate = Annotated[
    MatchedManifestCoordinate | IndependentManifestCoordinate,
    Field(discriminator="mode"),
]


class EpisodeManifestEntry(StrictModel):
    episode_public_id: str
    split_namespace: SplitNamespace
    suite: SuiteName
    coordinate: ManifestCoordinate
    requested_path_length: int
    accepted_attempt: int
    episode_sha256: HexDigest
    subset_memberships: tuple[str, ...] = ()
    parent_public_id: str | None = None
    parent_episode_sha256: HexDigest | None = None
    clock_scale: float = 1.0


class EpisodeManifest(StrictModel):
    schema_version: Literal[1]
    experiment_version: str
    access_class: ManifestAccessClass
    provenance: EvidenceProvenance
    suite: SuiteName
    # Environment-private regeneration secret; hash must match provenance.
    public_id_seed: int
    episode_count: int
    entries: tuple[EpisodeManifestEntry, ...]


class ManifestEnvelope(StrictModel):
    payload: EpisodeManifest
    payload_sha256: HexDigest


class ManifestPublication(StrictModel):
    path: str
    created: bool
    payload_sha256: HexDigest
    file_sha256: HexDigest
```

`AcceptedAttemptRun` is the unique run-length encoding of every accepted draw
in canonical source order. A matched draw is one whole accepted cohort, not
one member, so validation contains exactly 2,500 accepted draws; an independent
draw is one episode, so the Phase 1 gate contains exactly 100,000. Runs start at
zero, are contiguous and exactly cover `accepted_draw_count`; adjacent runs may
not carry the same accepted attempt, and overlaps/gaps/zero-length runs are
invalid. `ConstructionNamespaceEvidence` derives total IDs as base plus clock
IDs, validates the exact non-boolean raw `public_id_seed` against
`EvidenceProvenance.public_id_seed_sha256`, and rejects any noncanonical attempt
run, count mismatch, collision, wrong generation-mode count law, or non-exact
primitive type. Derive `rejected_draw_count` as
`sum(run.draw_count * run.accepted_attempt for run in accepted_attempt_runs)`
and require `generation_attempt_count == accepted_draw_count +
rejected_draw_count`; neither counter is trusted serialized arithmetic.

Build ordered namespace digests with explicit length framing and distinct
domains `silent-cascade/ofd-v1/construction-token-sequence/v1` and
`silent-cascade/ofd-v1/public-id-sequence/v1`. Matched token order is canonical
cohort order followed by each cohort's exact 20-token stream order; independent
token order is allocation-request order followed by each request's exact
7-token order. Public-ID order is every base source ID in canonical source
order, then clock suite order and parent order. Evidence builders consume
actual regenerated bundle coordinates, accepted attempts, the exact raw
public-ID seed, derived tokens, and emitted IDs—not configured counts or
coordinate surrogates. Use fixed-width
token/UUID storage and a sorted adjacent-duplicate scan so the complete audit
remains within its frozen memory ceiling.

Do not include wall-clock creation time in the hashed payload. Validate count,
order, coordinate/ID/hash uniqueness, paired-parent references, and exact
access-class rules. Validate the raw environment-private `public_id_seed`
against `provenance.public_id_seed_sha256`. A validation manifest requires
`provenance.generation_mode="matched"` and only matched coordinates; a
frozen-test manifest requires `"independent"` and only independent coordinates.
`DEBUG` additionally requires `split_namespace=DEBUG`, an allocation ID that
starts with `test-`, and a caller-selected DEBUG access class. `VALIDATION`
requires `split_namespace=VALIDATION`, allocation ID `validation-v1`, matched
generation, and the canonical validation builder. No generic builder may label
a test allocation as `VALIDATION` or `FROZEN_TEST`.
`provenance.allocation_id`, split namespace, root seed, plan/source revisions,
generator version, config/source hashes, and zero-call counter are mandatory.
For an independent entry, regenerate the private assigned variant from
`(root_seed, split_namespace, suite, requested_path_length,
allocation_quartet_index, quartet_member_index)` using Task 6's exact
`AllocationLabelKey`; never infer member position from `episode_index % 4`.
Require every independent manifest base quartet to contain member indices
exactly `{0,1,2,3}` and reject any independent base-entry set that contains an
incomplete quartet. Clock children are authenticated through their base parent
and do not form separate allocation quartets. This makes a nonzero or non-aligned block independently
regenerable without persisting a per-public-ID label or importing an external
allocation.
`subset_memberships`
lets Phase 6 encode the two nested compute-curve
subsets without creating duplicate tuning episodes; validation entries leave it
empty.

Freeze these exact sorted path sets; production evidence may not discover files
with a glob:

```python
GENERATOR_SOURCE_PATHS = (
    "src/silent_cascade/__init__.py",
    "src/silent_cascade/config.py",
    "src/silent_cascade/env/__init__.py",
    "src/silent_cascade/env/config.py",
    "src/silent_cascade/env/episode.py",
    "src/silent_cascade/env/generator.py",
    "src/silent_cascade/env/invariants.py",
    "src/silent_cascade/env/oracle.py",
    "src/silent_cascade/env/timing.py",
    "src/silent_cascade/errors.py",
    "src/silent_cascade/hashing.py",
    "src/silent_cascade/rng.py",
    "src/silent_cascade/schemas.py",
    "src/silent_cascade/validation.py",
)

PHASE1_ANALYSIS_SOURCE_PATHS = tuple(sorted((*GENERATOR_SOURCE_PATHS,
    "scripts/check_phase1_reproducibility.py",
    "scripts/verify_phase1_gate_artifacts.py",
    "src/silent_cascade/provenance.py",
    "src/silent_cascade/env/reward.py",
    "src/silent_cascade/env/leakage.py",
    "src/silent_cascade/env/reproducibility.py",
    "src/silent_cascade/env/services.py",
    "src/silent_cascade/io.py",
    "src/silent_cascade/logging/__init__.py",
    "src/silent_cascade/logging/manifest.py",
)))

TASK14_ANALYSIS_SOURCE_PATHS = tuple(sorted((*GENERATOR_SOURCE_PATHS,
    "scripts/check_phase1_reproducibility.py",
    "src/silent_cascade/env/leakage.py",
    "src/silent_cascade/env/reproducibility.py",
    "src/silent_cascade/env/reward.py",
    "src/silent_cascade/env/services.py",
    "src/silent_cascade/io.py",
    "src/silent_cascade/logging/__init__.py",
    "src/silent_cascade/logging/manifest.py",
    "src/silent_cascade/provenance.py",
)))
```

The generator tuple is already lexicographically sorted; the two analysis
tuples sort their explicit unions. Tests freeze all three tuples with literal exact-tuple
assertions: generator, final Phase 1 analysis, and historical Task 14 analysis.
A sortedness or generator-superset assertion is not a substitute for any of
those three literal expectations, and expected analysis tuples must not be
constructed from `GENERATOR_SOURCE_PATHS`, `sorted`, or the production constants
under test.

All three scopes must be closed over every explicit project-local import
reachable from a scoped file and over every package initializer Python executes
while importing those files. The closure test must parse `ast.Import` and
`ast.ImportFrom`, resolve both absolute and relative imports against the source
module's package, map local module candidates beneath both
`src/silent_cascade/` and `scripts/`, and add each existing ancestor
`__init__.py` for every scoped or imported package module. Parameterize the
closure check over `GENERATOR_SOURCE_PATHS`, `PHASE1_ANALYSIS_SOURCE_PATHS`,
and `TASK14_ANALYSIS_SOURCE_PATHS`; fail with the exact missing paths when any
derived local module or executed initializer is absent from the applicable
scope. The `phase1_task14_analysis` label records when that scope was
introduced; it does not preserve an obsolete dependency snapshot. Because the
current scoped `env/services.py` explicitly imports `env/leakage.py`, the Task
14 scope fingerprints the current leakage module and must continue to close
over any future local imports of its scoped files. `src/silent_cascade/cli.py`
is an adapter and is excluded from the scientific content fingerprint because
all successful evidence inputs and outputs are independently authenticated by
the services and the final verifier. No executed package initializer may be
omitted because it is empty, side-effect-free, or currently believed to be
non-semantic: its future contents can change import-time scientific behavior,
so the initializer itself remains fingerprinted.

Start hashing with
`b"silent-cascade/source-tree/v1\0"`, append the file count
as four-byte big-endian, then frame each item as
`len(path_utf8).to_bytes(4, "big") || path_utf8 ||
len(content).to_bytes(8, "big") || content`. Reject missing, duplicate,
symlinked, non-regular, root-escaping, or unexpected paths. The known-answer fixture
`{"a.py": b"x", "bb.py": b"yz"}` hashes to
`481adae060acaa85ecfc996ec2b847b52624ea432c5b4f7e91c5462f3fc41c49`.
Test that path/content pairs which collide under naive concatenation have
different framed hashes.

Compute `source_dirty` from changes to the selected fingerprint paths, resolved
YAML inputs, `pyproject.toml`, and `uv.lock`. Tests, docs, and generated output
artifacts do not alter this scientific-source flag. Task 18 still requires an
entirely clean worktree before producing evidence. This lets a second identical
publication verify its own output without treating the first artifact as a
scientific source change. Mutation regressions must build a committed
disposable repository containing all three exact scopes, collect generator,
final-analysis, and Task-14-analysis provenance, and first assert every
collector reports `source_dirty is False`. Parameterize a one-file mutation
over `src/silent_cascade/__init__.py`,
`src/silent_cascade/env/__init__.py`,
`src/silent_cascade/logging/__init__.py`, and
`src/silent_cascade/env/timing.py`, plus
`src/silent_cascade/env/leakage.py`. After each mutation, assert applicable
collectors report `source_dirty is True`; root/environment initializer and
timing mutations change the generator plus both analysis fingerprints, while
logging-initializer and leakage-module mutations change the final and Task 14
analysis fingerprints and leave the generator fingerprint unchanged. A dirty
result without the corresponding fingerprint change does not satisfy this
test.

Use this exact collector API:

```python
PHASE1_PLAN_PATH = Path(
    "docs/superpowers/plans/2026-08-30-phase-1-generator-oracle.md"
)


class EvidenceProvenanceCollector(Protocol):
    def __call__(
        self,
        resolved: ResolvedConfig[Phase1Config],
        *,
        repo_root: Path,
        generation_mode: Literal["matched", "independent"],
        allocation_id: str,
        split_namespace: SplitNamespace,
        root_seed: int,
        public_id_seed: int,
        analysis_seeds: Mapping[str, int],
    ) -> EvidenceProvenance: ...


def collect_evidence_provenance(
    resolved: ResolvedConfig[Phase1Config],
    *,
    repo_root: Path,
    generation_mode: Literal["matched", "independent"],
    allocation_id: str,
    split_namespace: SplitNamespace,
    root_seed: int,
    public_id_seed: int,
    analysis_seeds: Mapping[str, int],
    analysis_scope: Literal[
        "phase1_analysis", "phase1_task14_analysis"
    ] = "phase1_task14_analysis",
) -> EvidenceProvenance: ...


def collect_final_phase1_provenance(
    resolved: ResolvedConfig[Phase1Config],
    *,
    repo_root: Path,
    generation_mode: Literal["matched", "independent"],
    allocation_id: str,
    split_namespace: SplitNamespace,
    root_seed: int,
    public_id_seed: int,
    analysis_seeds: Mapping[str, int],
) -> EvidenceProvenance: ...
```

Resolve `source_commit` with `git rev-parse --verify HEAD^{commit}`. Require it
to be a full lowercase commit ID and an ancestor of the current `HEAD` under
`git merge-base --is-ancestor`; arbitrary, missing, abbreviated, unrelated, or
future revisions fail. Resolve `plan_base_revision` historically with
`git log -1 --format=%H <source_commit> -- <PHASE1_PLAN_PATH>`: it is the exact
latest commit that touched the approved plan as of the evidence source commit,
not a caller-supplied label and not the current worktree result. Fail if the
plan did not exist at that revision.

Fingerprint evidence content from Git objects at `source_commit`, never from a
possibly coordinated current worktree. For every exact v1 generator and
selected analysis path, use read-only `git ls-tree` to require one regular
`100644` or `100755` blob and `git cat-file blob
<source_commit>:<path>` to obtain its bytes. Reject absent paths, trees,
submodules, symlinks (`120000`), non-blob modes/types, duplicate paths, or any
hash mismatch. Do not checkout a revision or mutate the worktree. Set
`config_sha256=resolved.sha256` and retain current-worktree dirty coverage over
exactly `resolved.source_paths`, all selected fingerprint paths,
`pyproject.toml`, and `uv.lock`; Task 18 additionally requires a wholly clean
tree. Hash the public-ID seed through Task 3's domain-separated helper. The raw
seed is forbidden from agent inputs, public episode data, public manifests, and
public reports. The explicit exception is the environment-private,
agent-inaccessible final Phase 1 acceptance evidence reports whose verifier
contract must rederive every public ID; those reports are private acceptance
evidence, not public artifacts. Sort and copy `analysis_seeds`, enforce zero
foundation calls, and reject a missing source path rather than shrinking a
scope.

Task 13 unit tests create a temporary Git repository containing every frozen
path and prove exact commit resolution/ancestry, historical plan-base lookup,
Git-blob fingerprints, dirty state, config, seed, and zero-call behavior. They
mutate the worktree after committing and prove historical bytes still
authenticate while dirty state changes; missing/unrelated commits and
blob/tree/symlink substitutions fail without a checkout. The generic collector defaults to the exact historical
Task 14 scope; `collect_final_phase1_provenance` selects the exact final Phase 1
scope. Because `PHASE1_ANALYSIS_SOURCE_PATHS` deliberately names later modules
and both acceptance scripts, production final-scope provenance collection is
first exercised after Task 17 creates the final path. Tasks 14–17 tests inject
a collector and never weaken the production missing-path error.

- [ ] **Step 4: Implement no-clobber publication and verified loading**

Build canonical envelope bytes, call `atomic_create_bytes`, and read back the
published file before reporting success. On `artifact already exists`, compare
the verified existing bytes with the candidate. Preserve Phase 0
`published`, `cleanup_reason`, and `temp_path` context when wrapping atomic
errors.

`load_manifest` verifies payload hash before returning. Keep
`logging.manifest` independent of generator/oracle modules.

An access decision uses `manifest.access_class`, not filesystem path:

```python
def require_oracle_inspection_allowed(manifest: EpisodeManifest) -> None:
    if manifest.access_class is ManifestAccessClass.FROZEN_TEST:
        raise ManifestAccessError("oracle inspection is forbidden for frozen test data")
```

- [ ] **Step 5: Run GREEN checks**

```bash
uv run pytest -q tests/unit/test_provenance.py tests/unit/test_manifest.py
uv run ruff check src/silent_cascade/provenance.py src/silent_cascade/logging/manifest.py tests/unit/test_provenance.py tests/unit/test_manifest.py
uv run ruff format --check src/silent_cascade/provenance.py src/silent_cascade/logging/manifest.py tests/unit/test_provenance.py tests/unit/test_manifest.py
```

Expected: manifest hash, access, race, source-frame, and no-clobber tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/silent_cascade/provenance.py src/silent_cascade/logging/manifest.py \
  tests/unit/test_provenance.py tests/unit/test_manifest.py
git commit -m "feat: publish immutable episode manifests"
```

---

### Task 14: Validation Manifest Services and Reproducibility Harness

**Files:**

- Create: `src/silent_cascade/env/services.py`
- Create: `src/silent_cascade/env/reproducibility.py`
- Create: `scripts/check_phase1_reproducibility.py`
- Create: `tests/integration/test_phase1_services.py`
- Create: `tests/integration/test_phase1_reproducibility.py`

**Interfaces:**

- Consumes: Task 2 shared corpus hashing, Task 6 allocations, Tasks 7/9
  regeneration, Task 10 clock transforms, and Task 13 manifests.
- Produces: `build_cohort_manifest`, `build_validation_manifest`,
  `regenerate_entry`, `ReproducibilityRequest`, `ReproducibilityReport`, and
  `check_reproducibility`; pure
  `select_manifest_reproducibility_sample`,
  `select_independent_reproducibility_sample`,
  `manifest_sample_membership_sha256`, and
  `independent_sample_membership_sha256` helpers.

- [ ] **Step 1: Write failing small-service and process-order tests**

Use an injected clean `EvidenceProvenance` fixture and an 8- or 16-episode test
allocation. Assert build/regenerate hashes, accepted-attempt/public-ID match,
dirty production-source refusal, forward/reverse/chunks `1,3,7`, fresh process
with `PYTHONHASHSEED=0/1`, and atomic report refusal on a mismatch. The test
profile exercises both a matched manifest and a 16-episode independent
allocation; it never generates 10,000 episodes. Assert manifest-mode sample
membership, the independent worker payload round trip, injected provenance
reaching the report unchanged, and provenance/allocation/config/root/seed
mismatches failing before generation. Test the pure production quota planner's
exact 62/63 split without generating the 100,000 source episodes, and prove
production dependencies reject both small allocations. Include an independent
DEBUG block whose first episode index is nonzero and not divisible by four;
round-trip every member and prove regeneration uses the persisted
`quartet_member_index`, not `episode_index % 4`.

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
uv run pytest -q tests/integration/test_phase1_services.py tests/integration/test_phase1_reproducibility.py
```

Expected: services and reproducibility modules are absent.

- [ ] **Step 3: Implement pure manifest build/regeneration services**

```python
class ManifestReproducibilitySource(StrictModel):
    mode: Literal["manifest"] = "manifest"
    manifest_path: Path


class IndependentAllocationReproducibilitySource(StrictModel):
    mode: Literal["independent_allocation"] = "independent_allocation"
    allocation_id: str = Field(min_length=1)
    root_seed: int
    public_id_seed: int


class IndependentSourceDescriptor(StrictModel):
    schema_version: Literal["phase1-independent-source-v1"]
    allocation_id: str
    allocation_sha256: HexDigest
    split_namespace: SplitNamespace
    root_seed: int
    public_id_seed_sha256: HexDigest
    config_sha256: HexDigest
    generator_source_sha256: HexDigest


type ReproducibilitySource = Annotated[
    ManifestReproducibilitySource | IndependentAllocationReproducibilitySource,
    Field(discriminator="mode"),
]


class ReproducibilityRequest(StrictModel):
    source: ReproducibilitySource
    sample_size: int = Field(gt=0)
    chunk_sizes: tuple[int, ...]
    python_hash_seeds: tuple[int, ...]
    verify_all_source_entries: bool
    output_path: Path | None = None


class ReproducibilityReport(StrictModel):
    schema_version: Literal["phase1-reproducibility-v2"]
    source_mode: Literal["manifest", "independent_allocation"]
    source_payload_sha256: HexDigest
    sample_size: int
    verified_source_entries: int
    modes: tuple[str, ...]
    chunk_sizes: tuple[int, ...]
    python_hash_seeds: tuple[int, ...]
    reference_corpus_sha256: HexDigest
    sample_membership_sha256: HexDigest
    mismatch_count: Literal[0]
    namespace_evidence: ConstructionNamespaceEvidence
    provenance: EvidenceProvenance
    passed: bool


class FreshProcessRunner(Protocol):
    def __call__(
        self, work_order_bytes: bytes, python_hash_seed: int
    ) -> bytes: ...


@dataclass(frozen=True, slots=True)
class ReproducibilityDependencies:
    production_mode: bool
    independent_allocation: IndependentAllocation
    collect_provenance: EvidenceProvenanceCollector
    load_verified_manifest: Callable[[Path], EpisodeManifest]
    regenerate_manifest_entry: Callable[..., EpisodeBundle]
    iter_independent: Callable[..., Iterator[IndependentEpisodeRequest]]
    generate_independent: Callable[..., EpisodeBundle]
    run_fresh_process: FreshProcessRunner
```

`build_cohort_manifest` is the reusable pure builder. It accepts a structurally
valid matched allocation and an explicit access class restricted to `DEBUG` or
`VALIDATION`. `DEBUG` requires a DEBUG namespace and `test-` allocation ID;
`VALIDATION` requires the frozen validation guard. `build_validation_manifest`
is the production wrapper: it hardcodes `VALIDATION_ALLOCATION`, streams it,
retains entries only, computes exact 5,000/2,500/2,500 counts, emits only
`VALIDATION`, and refuses dirty production provenance. `regenerate_entry`
dispatches on the entry coordinate. It
reconstructs the complete source cohort for a matched entry or exactly one
episode for an independent entry, requires accepted attempt/public ID/hash
equality, then applies/verifies a clock parent before a child when present.
Independent regeneration reconstructs `IndependentEpisodeRequest` from the
entry's suite/path/index/quartet/member fields plus authenticated provenance;
it neither guesses block alignment nor receives a hidden allocation.
Phase 1's builder never creates `FROZEN_TEST` data, but this regeneration path
is tested against temporary independent frozen-access fixtures.

Exact signatures are:

```python
def build_cohort_manifest(
    config: Phase1Config,
    allocation: CohortAllocation,
    provenance: EvidenceProvenance,
    root_seed: int,
    public_id_seed: int,
    *,
    access_class: Literal[
        ManifestAccessClass.DEBUG, ManifestAccessClass.VALIDATION
    ],
) -> EpisodeManifest: ...


def build_validation_manifest(
    config: Phase1Config,
    provenance: EvidenceProvenance,
    root_seed: int,
    public_id_seed: int,
) -> EpisodeManifest: ...


def regenerate_entry(
    config: Phase1Config,
    manifest: EpisodeManifest,
    entry: EpisodeManifestEntry,
) -> EpisodeBundle: ...
```

Tests pass a structurally valid 8–16 episode DEBUG allocation to
`build_cohort_manifest`; production services bind only
`build_validation_manifest`. Both builders validate `root_seed` and the
fingerprint of `public_id_seed` against the supplied provenance before
generating anything.

- [ ] **Step 4: Implement the reusable fresh-process harness**

`check_reproducibility` establishes the manifest-order or frozen independent
allocation-order hash reference, checks forward/reverse/one-at-a-time and every
chunk size in-process, then invokes the same module in a fresh Python process
with an explicit `PYTHONHASHSEED` for each declared value. Subprocesses emit one
canonical result object; malformed output or any hash difference is a hard
failure. For an independent-allocation source it streams every request in the
dependency-bound allocation once (100,000 in production, 16 in the declared
test fixture), verifies every accepted hash/ID/token, and applies the
order/chunk/subprocess matrix to the deterministic audit-hash-selected
`sample_size` requests. Before spawning, the parent writes a private temporary
work order containing only the selected public regeneration coordinates,
accepted-attempt tokens, canonical config, and the raw public-ID seed needed to
reconstruct those coordinates. The subprocess does not enumerate or choose an
allocation, deletes the work order in a `finally` block, and returns only its
canonical digest object. The script is a thin argparse adapter that builds
`ReproducibilityRequest`, publishes only a fully passing report via Task 13
no-clobber I/O, and exits nonzero without a final artifact otherwise.
Its exact library signature is `check_reproducibility(request:
ReproducibilityRequest, resolved: ResolvedConfig[Phase1Config], *,
deps: ReproducibilityDependencies = PRODUCTION_REPRODUCIBILITY_DEPENDENCIES) ->
ReproducibilityReport`; generation receives `resolved.config`, while provenance
receives the complete resolved object so YAML source paths remain covered.

Every reference and comparison corpus digest uses Task 2's exact
`corpus_sha256` helper. Execution order may be forward, reverse, or chunked, but
results are keyed by source coordinate and returned to canonical source order
before hashing. `reference_corpus_sha256` is therefore the manifest-entry
digest for manifest mode and the 100,000-base-episode allocation digest for the
production independent mode; paired clock children are not added to the latter.

The source `allocation_id` must equal `deps.independent_allocation.allocation_id`.
Production dependencies additionally require the canonical Phase 1 allocation,
namespace, exact 100,000 count, and `phase1-independent-gate-v1` ID. Test
dependencies set `production_mode=False`, bind a structurally valid 16-episode
`DEBUG` allocation whose ID starts with `test-`, supply a clean injected
`EvidenceProvenance`, and use the same real subprocess entry point. The library
calls `collect_provenance` before generating an episode. In manifest mode it
must first load and cryptographically verify the manifest, because generation
mode, allocation, split, root, and public-ID-seed fingerprint are authenticated
manifest fields needed as collector inputs; it then collects current
provenance and compares it with the embedded provenance before regenerating any
entry. Independent mode collects directly from the dependency-bound allocation
and request, then requires generation mode, allocation ID, split, root seed,
public-ID-seed fingerprint, config hash, generator-source hash, and zero
foundation calls to match before iteration. The returned report embeds that
exact validated provenance. The production script always binds
`PRODUCTION_REPRODUCIBILITY_DEPENDENCIES` and refuses `DEBUG`/`test-` sources.

For manifest mode, `source_payload_sha256` is the verified manifest payload
hash. For independent mode, it is SHA-256 of canonical JSON bytes for
`IndependentSourceDescriptor`; no raw public-ID seed enters the descriptor.
`allocation_sha256` is the hash of the complete canonical allocation model, so
reusing an ID with different blocks cannot alias a source. The known-answer
descriptor with root `41`, public-ID-seed fingerprint `"0" * 64`, config hash
`"1" * 64`, generator hash `"2" * 64`, and allocation hash `"3" * 64` is
`02bfb26fe2cd5d84bfcea48ef60175e5a31cc6b28e59958dc5ec0ae0ac32cb8b`.

For manifest mode, select `min(request.sample_size, manifest.episode_count)`
entries with the same allocation-derived rule below, using nonempty
suite/path strata in their first manifest occurrence order. Rank entries with
domain `silent-cascade/ofd-v1/repro-manifest-sample-rank/v1`, the verified
manifest payload hash, suite, path length, coordinate, and public ID. The
manifest sample-membership object uses domain
`silent-cascade/ofd-v1/repro-manifest-sample-membership/v1` and includes the
ordered public IDs plus coordinates. This value populates the same
`sample_membership_sha256` report field.

For independent mode, select
`min(request.sample_size, source_episode_count)` requests deterministically.
Let `strata` be the nonempty allocation blocks in declared order, then compute
`per_stratum, remainder = divmod(selected_count, len(strata))`; take
`per_stratum` from every stratum and one additional request from each of the
first `remainder` strata. Reject a request if any stratum lacks its quota.
Within each stratum, rank requests by SHA-256 of canonical JSON containing domain
`silent-cascade/ofd-v1/repro-sample-rank/v1`, source payload hash, suite,
requested path length, and episode index; break a digest tie by episode index.
The production request freezes `sample_size=1_000`; the 16-episode test request
freezes `sample_size=16` and therefore selects every entry. Production retains
the frozen 62-per-stratum plus one for the first eight strata rule. The rank
known answer for the dummy descriptor above and
`(iid_primary, path=2, episode=7)` is
`9f0db840b7386858370339a548c8b785b5560be30c39968d2c427a587e2fb80b`.
Hash selected coordinates in frozen stratum order and then rank order as one
canonical object with domain
`silent-cascade/ofd-v1/repro-sample-membership/v1`, the source payload hash, and
a `coordinates` list of `{suite, requested_path_length, episode_index}`. For the
ordered two-item
fixture `[(iid_primary,2,7),(ood_depth,5,3)]`, the membership hash is
`f18653443059e3af94996faf6cd201d3f2e51f502572b2c52fac7ad06509c8f5`.
Persist that value as `sample_membership_sha256` and test all three vectors in
fresh processes.

Expose the sample selection and membership hashing algorithms through the four
pure helpers named in this task. `check_reproducibility` calls those helpers
directly. The final verifier separately invokes them against the verified
validation manifest and the exact Phase 1 allocation/descriptor, then requires
the recomputed digests to equal each report. Editing a stored report digest—even
with coordinated edits to that report's other membership fields—cannot alter
the independently selected membership derived from the verified source; this
is a bounded cross-check, not a claim against replacement of every trust
anchor. Remove the unused request from the
entry helper: its exact signature is `_entry_for(bundle: EpisodeBundle) ->
CorpusDigestEntry`, and every caller passes only the regenerated bundle. Retain
a real matched-manifest fresh-process regression rather than satisfying this
gate through an injected fake runner.

Every report includes namespace evidence built from the actual regeneration
pass. Matched validation records raw public-ID seed `2026083002`, exactly 2,500
accepted cohort draws, 50,000 ordered construction tokens, 10,000 base IDs, and
zero clock IDs. The independent gate records raw public-ID seed `2026083012`,
exactly 100,000 accepted episode draws, 700,000 ordered construction tokens,
100,000 base IDs, and 7,000 clock IDs. Attempt runs and ordered sequence hashes
must describe the actual bundles; all collision counts are exact zero. Strict
report validation checks each raw seed's provenance fingerprint. The matched
validation reproducibility report is the canonical matched namespace summary;
the oracle, leakage, and independent-reproducibility reports must independently
construct byte-for-byte equal independent namespace summaries from the same
100,000 accepted draws and 7,000 derived clock IDs.

- [ ] **Step 5: Run GREEN checks**

```bash
uv run pytest -q tests/integration/test_phase1_services.py tests/integration/test_phase1_reproducibility.py
uv run ruff check src/silent_cascade/env/services.py src/silent_cascade/env/reproducibility.py scripts/check_phase1_reproducibility.py tests/integration/test_phase1_services.py tests/integration/test_phase1_reproducibility.py
uv run ruff format --check src/silent_cascade/env/services.py src/silent_cascade/env/reproducibility.py scripts/check_phase1_reproducibility.py tests/integration/test_phase1_services.py tests/integration/test_phase1_reproducibility.py
```

- [ ] **Step 6: Commit**

```bash
git add src/silent_cascade/env/services.py src/silent_cascade/env/reproducibility.py \
  scripts/check_phase1_reproducibility.py tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_reproducibility.py
git commit -m "feat: build reproducible validation manifests"
```

---

### Task 15: Leakage Audit with Positive Controls

**Files:**

- Create: `src/silent_cascade/env/leakage.py`
- Create: `tests/unit/test_leakage.py`
- Modify: `tests/property/test_generator_properties.py`

**Interfaces:**

- Consumes: generated private bundles only inside the authorized auditor.
- Produces: `LeakageAuditProfileName`, `ShortcutTask`,
  `ShortcutFeatureGroup`, `AuditSeedKey`, `ShortcutFeatureSet`,
  `ShortcutProbeResult`, `PositiveControlResult`, `AuditExample`,
  `AuditSourceDescriptor`, `ReiterableAuditSource`, `InMemoryAuditSource`,
  `NamedLeakInjector`, `CounterfactualCheckId`,
  `CounterfactualPairResult`, `CounterfactualCheckResult`, `LeakageReport`,
  `extract_shortcut_features`, `audit_leakage`, and streaming detector state.
  All corpus hashes consume Task 2's shared digest framing.

Exact signatures are `extract_shortcut_features(example: AuditExample,
corpus_size: int) -> ShortcutFeatureSet` and
`audit_leakage(source: ReiterableAuditSource, config: LeakageAuditConfig,
profile: LeakageAuditProfileName, provenance: EvidenceProvenance, workspace:
Path, positive_control: str | None = None) -> LeakageReport`.

Use these exact task/group and report schemas:

```python
class LeakageAuditProfileName(str, Enum):
    TEST = "test"
    PHASE1_GATE = "phase1_gate"


class ShortcutTask(str, Enum):
    POSITIVE_BINARY = "positive_binary"
    VARIANT_THREE_WAY = "variant_three_way"
    POSITIVE_HAZARD_CLASS = "positive_hazard_class"


class ShortcutFeatureGroup(str, Enum):
    ID_POSITION = "id_position"
    ACTIVATION_NODE = "activation_node"
    FIRST_LAST_FACT = "first_last_fact"
    COUNTS = "counts"
    ORDER_RECORD_IDS = "order_record_ids"
    TIMES = "times"
    TERMINAL_MULTISET = "terminal_multiset"
    LINK_TOPOLOGY = "link_topology"
    COMBINED = "combined"


@dataclass(frozen=True, slots=True)
class AuditSeedKey:
    schema_version: Literal["leakage-v1"]
    audit_seed: int
    corpus_hash: HexDigest
    task: ShortcutTask
    replicate_index: int
    suite: SuiteName
    requested_path_length: int
    randomization_block_index: int
    episode_position: int


@dataclass(frozen=True, slots=True)
class ShortcutFeatureSet:
    schema_version: Literal["leakage-features-v1"]
    episode_public_id: str
    generation_mode: Literal["matched", "independent"]
    audit_group_id: str
    suite: SuiteName
    requested_path_length: int
    vectors: Mapping[ShortcutFeatureGroup, np.ndarray]


class OrderedSha256Pack(StrictModel):
    schema_version: Literal["ordered-sha256-pack-v1"]
    encoding: Literal["base64-concatenated-sha256-v1"]
    item_count: int = Field(gt=0, le=100_000)
    payload_base64: str
    payload_sha256: HexDigest


class ShortcutProbeResult(StrictModel):
    task: ShortcutTask
    feature_group: ShortcutFeatureGroup
    feature_dimension: int = Field(gt=0)
    train_examples: int = Field(gt=0)
    test_examples: int = Field(gt=0)
    train_class_counts: dict[str, int]
    test_class_counts: dict[str, int]
    test_confusion_counts: dict[str, dict[str, int]]
    permutation_exceedance_count: int = Field(ge=0)
    permutation_replicate_count: int = Field(gt=0)
    raw_accuracy: float = Field(ge=0.0, le=1.0)
    balanced_accuracy: float = Field(ge=0.0, le=1.0)
    balanced_chance: float = Field(ge=0.0, le=1.0)
    raw_permutation_p: float = Field(ge=0.0, le=1.0)
    holm_adjusted_p: float = Field(ge=0.0, le=1.0)
    optimizer_iterations: int = Field(ge=0, le=500)
    optimizer_converged: bool
    passed: bool


class PositiveControlResult(StrictModel):
    control_id: str
    target_task: ShortcutTask
    expected_detector_id: str
    observed_detector_ids: tuple[str, ...]
    base_subset_corpus_sha256: HexDigest
    base_subset_membership_sha256: HexDigest
    injected_corpus_sha256: HexDigest
    injected_episode_sha256s: OrderedSha256Pack
    split_membership_sha256: HexDigest
    balanced_accuracy: float | None = Field(ge=0.0, le=1.0)
    holm_adjusted_p: float | None = Field(ge=0.0, le=1.0)
    probes: tuple[ShortcutProbeResult, ...]
    passed: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class AuditExample:
    bundle: EpisodeBundle
    manifest_rank: int
    generation_mode: Literal["matched", "independent"]
    randomization_block_index: int
    episode_position: int
    quartet_member_index: int


class AuditSourceDescriptor(StrictModel):
    schema_version: Literal["leakage-source-v1"]
    generation_mode: Literal["matched", "independent"]
    allocation_id: str
    allocation_or_manifest_sha256: HexDigest
    split_namespace: SplitNamespace
    root_seed: int
    public_id_seed_sha256: HexDigest
    config_sha256: HexDigest
    generator_source_sha256: HexDigest
    episode_count: int = Field(gt=0, multiple_of=4)


class ReiterableAuditSource(Protocol):
    @property
    def descriptor(self) -> AuditSourceDescriptor: ...

    @property
    def episode_count(self) -> int: ...

    def iter_examples(self) -> Iterator[AuditExample]: ...


@dataclass(frozen=True, slots=True)
class InMemoryAuditSource:
    descriptor: AuditSourceDescriptor
    examples: tuple[AuditExample, ...]

    @property
    def episode_count(self) -> int:
        return len(self.examples)

    def iter_examples(self) -> Iterator[AuditExample]:
        return iter(self.examples)


@dataclass(frozen=True, slots=True)
class NamedLeakInjector:
    control_id: str
    target_task: ShortcutTask
    expected_detector_id: str
    apply: Callable[[ReiterableAuditSource], ReiterableAuditSource]


class CounterfactualCheckId(str, Enum):
    TERMINAL_DELAY_SWAP = "terminal_delay_swap"
    PRESENTATION_PERMUTATION = "presentation_permutation"
    PAIRED_CLOCK_SCALE = "paired_clock_scale"


class CounterfactualPairResult(StrictModel):
    pair_key_sha256: HexDigest
    decision_mismatch: bool
    temporal_mismatch: bool


class CounterfactualCheckResult(StrictModel):
    check_id: CounterfactualCheckId
    checked_pairs: int = Field(gt=0)
    decision_mismatch_count: int = Field(ge=0)
    temporal_mismatch_count: int = Field(ge=0)
    result_payload_sha256: HexDigest
    passed: bool


class LeakageConstructionStatistics(StrictModel):
    schema_version: Literal["leakage-construction-statistics-v1"]
    source_episode_sha256s: OrderedSha256Pack
    clock_child_episode_sha256s: OrderedSha256Pack
    invariant_verified_count: int = Field(ge=0)
    feature_row_count: int = Field(ge=0)
    finite_feature_row_count: int = Field(ge=0)
    second_pass_verified_count: int = Field(ge=0)
    second_pass_match_count: int = Field(ge=0)
    first_pass_source_manifest_sha256: HexDigest
    second_pass_source_manifest_sha256: HexDigest


class LeakageMembershipEvidence(StrictModel):
    schema_version: Literal["leakage-membership-evidence-v1"]
    split_membership_sha256: HexDigest
    train_membership_sha256: HexDigest
    test_membership_sha256: HexDigest
    train_episode_count: int = Field(gt=0)
    test_episode_count: int = Field(gt=0)


class LeakageReport(StrictModel):
    schema_version: Literal["leakage-report-v3"]
    provenance: EvidenceProvenance
    namespace_evidence: ConstructionNamespaceEvidence
    construction_statistics: LeakageConstructionStatistics
    membership_evidence: LeakageMembershipEvidence
    generation_mode: Literal["matched", "independent"]
    profile: LeakageAuditProfileName
    corpus_hash: HexDigest
    feature_schema_hash: HexDigest
    split_membership_hash: HexDigest
    train_membership_hash: HexDigest
    test_membership_hash: HexDigest
    episode_count: int
    randomization_block_count: int
    suite_path_denominators: dict[str, int]
    construction_check_ids: tuple[str, ...]
    construction_checks: dict[str, bool]
    probes: tuple[ShortcutProbeResult, ...]
    label_shuffled_probes: tuple[ShortcutProbeResult, ...]
    positive_controls: tuple[PositiveControlResult, ...]
    counterfactual_checks: tuple[CounterfactualCheckResult, ...]
    label_shuffled_control_passed: bool
    passed: bool
```

This is the exact v3 extension of the v2 leakage evidence schema: no v2
evidence field may disappear during migration. In particular, the three
top-level membership hashes and all construction check fields remain as
redundant cross-checks rather than being replaced by their typed evidence.
`PositiveControlResult.probes`, `LeakageReport.construction_check_ids`, and
`LeakageReport.label_shuffled_probes` remain mandatory typed tuples, while
`namespace_evidence` remains mandatory. The displayed `Field` bounds are carried over
verbatim: every feature/workload count is positive, every probability or
accuracy is in `[0,1]`, and optimizer iterations are in `[0,500]`; optional
positive-control summary probabilities, when present, use the same `[0,1]`
bounds. A positive control's workload remains its complete ordered
nine-feature `probes` family, so each contained probe retains those exact
dimension/count/metric/iteration bounds. Port the existing v1 exact-primitive,
task-derived chance, non-empty/nonnegative class-count and count-sum,
phase-profile workload, feature-dimension, permutation-grid, convergence,
Holm, detector-identity, summary-metric, construction-check, label-shuffled
pass, and outer-pass validators without weakening or replacing them. Every
probe must additionally validate an exact square test confusion matrix whose
row sums equal `test_class_counts`, derive raw and mean-per-class balanced
accuracy from its diagonal, and derive the add-one permutation p-value from
`permutation_exceedance_count` and `permutation_replicate_count`. The Phase 1
clean, label-shuffled, and positive-control probes require exactly 4,999
replicates. The producer and standalone verifier implement these derivations
independently. They must
still rederive the complete ordered 27 clean probes, complete ordered 27
label-shuffled probes, and every positive control's complete ordered
nine-feature probe family. Validate that
`counterfactual_checks` contains each enum value exactly once in enum order. A
check passes only with zero decision and temporal mismatches, and the overall
report passes only when all three checks pass. A scientific-failure report
retains nonzero counts and `passed=false`; corruption or a missing check raises
instead of producing an incomplete report.

`OrderedSha256Pack` is not a free-form compressed blob. Decode
`payload_base64` with strict standard padded Base64, require canonical
round-trip encoding, require exactly `32 * item_count` decoded bytes, split it
into the declared ordered SHA-256 values, and require `payload_sha256` to equal
the SHA-256 of those raw concatenated bytes. The source pack contains exactly
100,000 episode digests in canonical source order for the Phase 1 profile. The
clock-child pack contains exactly 7,000 child episode digests: first all 5,000
`0.1x` children in increasing authenticated parent-manifest-rank order, then
all 2,000 `10x` children in increasing authenticated parent-manifest-rank
order. Every one of the nine positive controls contains exactly 8,000 injected
episode digests in its canonical selected-source order. These are exactly
eleven packs: one base source, one clock child, and nine positive-control
injected packs. Reject a duplicate, missing, reordered, truncated, overlong,
noncanonical, or digest-inconsistent pack. The complete canonical
`leakage-report-v3` JSON, including all eleven packs, must be no more than
`16 * 1024 * 1024` bytes in the producer, immutable publisher, loader, and
standalone verifier.

`LeakageConstructionStatistics` is produced by the two real streaming passes.
Its source pack and the independently reconstructed public IDs/coordinates
must reproduce the report corpus hash and the v2 source-manifest hash. Its
clock-child pack combines with the source pack, reconstructed parent/child
public IDs, exact scale values, and authenticated parent ranks to reconstruct
the complete canonical
`silent-cascade/ofd-v1/leakage-clock-pair-manifest/v2` payload and digest. That
digest is checked at production time against both the live
`AuditSourceAuthentication.clock_pair_manifest_sha256` and the serialized v2
`LeakageAuditEvidenceAnchor.clock_pair_manifest_sha256`. The standalone
verifier has no live `AuditSourceAuthentication` object, so it independently
reconstructs the digest from the two packs and frozen recipe and checks the
serialized v2 anchor (plus a redundant serialized report field only if one is
introduced explicitly). No stored clock-manifest hash is accepted without
this reconstruction. Its
invariant, feature, finite-feature, second-pass, and second-pass-match counts
must all equal `episode_count`; both source-manifest hashes must be equal.
Derive the six ordered construction checks exactly: `provenance` from the
authenticated descriptor/anchor/source manifest, `public_ids` and
`seed_tokens` from `ConstructionNamespaceEvidence`, `invariants` from the
verified count, `finite_features` from the exact feature counts, and
`two_pass_identity` from the second-pass count/hash equality. Do not accept a
self-reported boolean that contradicts these primitives.

`LeakageMembershipEvidence` duplicates the existing top-level split, train,
and test hashes plus exact counts. Reconstruct it from the frozen audit seed,
corpus hash, canonical suite/path/block groups, and complete ordered public-ID
stream. Require the main train/test sets to be disjoint, their union to equal
the full source, and their groupwise split to have the frozen 80/20 counts.
For each clean and label-shuffled probe, derive its workload independently from
the main membership and task filter: positive-binary and variant-three-way use
the full split, while positive-hazard-class uses only positive members of each
split. Compare that task-filtered train/test count with the probe's exact
workload; never require the main 80,000/20,000 counts to equal a hazard probe.

For each positive control, independently select the balanced 8,000-item source
subset, derive and compare its membership and corpus hash, derive its frozen
groupwise split and split-membership hash, reconstruct the control-specific
injected public IDs, and combine them with its packed injected digests to
reproduce `injected_corpus_sha256`. Each control probe's workload is derived
from that named control's own 8,000-item subset and groupwise split, then
filtered for its target task; the hazard-class control again uses positives
only. Frozen denominators and thresholds do not change.

`base_subset_membership_sha256` is SHA-256 of canonical UTF-8 JSON with exactly
these keys and structure:

```json
{
  "domain": "silent-cascade/ofd-v1/pc-base-subset-membership/v1",
  "item_count": 2,
  "members": [
    {
      "episode_public_id": "00000000-0000-4000-8000-000000000001",
      "source_index": 1
    },
    {
      "episode_public_id": "00000000-0000-4000-8000-000000000002",
      "source_index": 3
    }
  ],
  "source_episode_count": 4
}
```

The real payload has `source_episode_count=100000`, `item_count=8000`, and one
member per selected source row. `members` is ordered by strictly increasing,
unique zero-based `source_index`, and each public ID must equal the source row
at that index. `item_count` is the explicit length frame and must equal the
array length; no sorting after construction is permitted. The literal
two-member known-answer digest above is
`d9abb33f9eeb7d56d585590565a586cb0c32f955948aee98dfafee0fb1e92bf1`.
Producer and verifier tests compute that literal independently, reject a wrong
domain/key/count/index/order/public ID, and do not call one another's helper.
The producer and verifier may share only versioned schemas and domain strings,
not membership, metric, aggregation, or counterfactual builder functions.

These bounded packs are raw per-episode digest evidence for arithmetic, not
the full public episode payloads, optimizer predictions, or permutation
vectors. The artifact and final report must preserve the explicit limitation:
an attacker able to replace all artifacts, packs, historical source, and trust
anchors coherently cannot be detected without regenerating the corpus and
rerunning the fits. The local verifier proves every relationship derivable
from the frozen inputs and serialized primitives; it does not claim universal
tamper detection.

Frame each pair key as SHA-256 of canonical JSON:

```json
{
  "domain": "silent-cascade/ofd-v1/counterfactual-pair/v1",
  "check_id": "terminal_delay_swap",
  "source_public_ids": [
    "00000000-0000-4000-8000-000000000001",
    "00000000-0000-4000-8000-000000000002"
  ],
  "transform": "swap_terminal_delay"
}
```

Use transform tags `swap_terminal_delay`, `permute_presentation`,
`scale_0_1x`, and `scale_10x`. Delay IDs retain left/right canonical pairing
order; presentation has one source ID; clock has one parent ID plus its scale
tag. The pair-key known answer above is
`a72ebffcd5bedcb5da8932a6bacdf307994dddeb8951e8976638b9c1c1e961ca`.

Hash a check's result as canonical JSON with exactly this shape:

```json
{
  "domain": "silent-cascade/ofd-v1/counterfactual-check/v1",
  "check_id": "terminal_delay_swap",
  "pairs": [
    {
      "pair_key_sha256": "a72ebffcd5bedcb5da8932a6bacdf307994dddeb8951e8976638b9c1c1e961ca",
      "decision_mismatch": false,
      "temporal_mismatch": false
    }
  ]
}
```

Pairs are ordered by canonical source coordinate; the clock check orders all
`0.1x` pairs before all `10x` pairs and uses parent source order within a scale.
Derive `checked_pairs`, both mismatch counts, and `passed` from this exact list
before discarding it. The result-payload known answer above is
`28b15afe82e20b9e8db3c2f5bf4402e454858bfdaf7ce9663001d5aac99f1372`.

- [ ] **Step 1: Write failing construction, feature, and stream tests**

The detector family covers binary positive/negative, three-way variant, and
positive hazard class. Predeclared feature groups are:

```text
public ID and manifest position
activation ID and individual node-value summaries
first/last fact kind and fields
total and per-kind counts
presentation positions and record IDs
timestamp gaps, duration, activation gap
hazard delay/class multisets
entity/component/degree summaries from the `LINK` subgraph, not rooted at activation
all shortcut features combined
```

Exclude explicit activation-rooted reachability/path features and any feature
that joins terminal kind to graph position, because that join is the intended
graph relation the benchmark requires. Link-only topology summaries remain in
the audit.

Test both generation modes, every fixed feature dimension/value, forbidden
feature absence, first/second-pass identity, memmap cleanup, and deliberate
feature-store/RSS ceiling failures. These tests do not mention the optimizer,
permutations, injectors, or counterfactuals yet.

- [ ] **Step 2: Run tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_leakage.py
```

Expected: import fails because the construction/feature stream does not exist.

- [ ] **Step 3: Implement exact construction audits**

Fail immediately on missing strata, mismatched kind/terminal counts, duplicate
public IDs, actual derived construction-token collisions, invalid invariants,
nonfinite features, or
insufficient sample size. In matched mode, require exact within-cohort nuisance
equality. In independent mode, instead require disjoint episode-local nuisance
tokens, absence of any cohort-scoped generator call, exact aggregate variant
strata, and the independent coordinate/recipe for every episode; nuisance
values are not required to equal or differ by value because independent draws
may coincide by chance. Report generation mode plus variant/path/suite
denominators explicitly. Require `report.generation_mode ==
report.provenance.generation_mode` and validate source/allocation/config/audit
seed provenance before reading the first episode. Specifically compare
`source.descriptor` with the supplied `EvidenceProvenance` for generation mode,
allocation ID, split, root, public-ID-seed fingerprint, config hash, generator
source hash, and episode count; validate the descriptor's manifest/allocation
hash when the source is constructed. Any mismatch fails before
`iter_examples()` is called. Leakage provenance requires
`analysis_seeds={"audit_seed": config.audit_seed,
"positive_control_seed": config.positive_control_seed}` exactly; other Phase 1
reports use only their explicitly named analysis seeds or an empty mapping.
The `seed_tokens` construction check derives the exact ordered 20-token matched
or 7-token independent sequence from each accepted attempt, stores fixed-width
tokens, sorts a bounded copy for adjacent duplicate detection, and hashes the
canonical ordered sequence. It may not substitute episode coordinates,
requests, configured counts, or any self-reported boolean for actual counter
digest tokens. Oracle and leakage acceptance paths independently compute the
same actual token and public-ID sets and fail on any collision before reporting
success.

- [ ] **Step 4: Implement fixed feature encoding**

Extract float32 arrays with these immutable dimensions and vocabularies;
convert only the optimizer input to float64:

```text
ID_POSITION (17)
  UUID bytes / 255 (16), manifest rank / max(N-1, 1) (1)
ACTIVATION_NODE (256)
  activation one-hot (64), subject frequency (64), non-null object frequency
  (64), all-node mention frequency (64)
FIRST_LAST_FACT (278)
  for first and last: kind one-hot (3), subject one-hot (64), object one-hot
  including None (65), hazard-present (1), hazard class one-hot (4),
  delay-present (1), log1p(delay)-or-zero (1)
COUNTS (4)
  total/LINK/HAZARD/SAFE divided by 64
ORDER_RECORD_IDS (768)
  64 slots x [valid, kind one-hot(3), normalized subject, object-present,
  normalized object, hazard-present, normalized class, delay-present,
  log1p(delay), normalized record ID]
TIMES (194)
  64 slots x [valid, log1p(time-initial), log1p(previous-gap)] plus
  log1p(observation-duration) and log1p(activation-gap)
TERMINAL_MULTISET (10)
  four-bin hazard histogram / 2, two sorted log1p delays, two sorted normalized
  hazard classes, normalized hazard/safe counts
LINK_TOPOLOGY (33)
  normalized node/edge/weak-component/SCC counts and cycle flag (5), five-bin
  in/out-degree histograms (10), eight-bin weak-component size histogram (8),
  min/max/mean/std for in/out degree (8), source/sink fractions (2)
COMBINED (1560)
  concatenation of the preceding eight groups in enum order
```

Missing padded numeric fields are zero and presence flags disambiguate them.
For `POSITIVE_HAZARD_CLASS`, remove every hazard-class identity channel before
fitting; retaining it would make the target legitimately visible in the public
terminal multiset. Forbidden inputs are private truth, seed/key fields, rooted
reachability, activation-rooted path length, terminal-to-graph joins, and any
cross-feature that performs the intended graph relation.

The full profile is a bounded two-pass stream, never an in-memory bundle tuple:

1. pass one iterates `source.iter_examples()` in batches of 4,096, performs
   construction/invariant checks, updates Task 2's `CorpusHashBuilder` in exact
   source order, and writes
   only the 1,560-column `COMBINED` float32 vector plus compact labels/group
   metadata to workspace-local `numpy.memmap` files;
2. pass two regenerates the source, verifies the same ordered public IDs and
   episode hashes, and prepares counterfactual/positive-control streams. Probe
   groups are column views into `COMBINED`, never duplicate stores;
3. fitting reads batches/views from the memmap. Permutation statistics process
   at most 64 replicates per batch and retain only running exceedance counts,
   not an episode-by-replicate matrix;
4. before allocation, compute the exact expected store bytes and fail above
   `max_feature_store_bytes=800_000_000`. Track process RSS with `psutil` at
   every batch boundary and fail above
   `max_resident_working_bytes=512_000_000`;
5. create files only beneath the caller-owned temporary workspace, flush
   completed stores, and remove them on success or failure. Only hashed reports
   remain.

At 100,000 episodes the sole dense store is exactly 624,000,000 feature bytes,
plus bounded metadata below the 800 MB ceiling. Unit tests use an injected
32–128-example source and smaller ceilings to prove byte/RSS failures,
second-pass mismatch detection, cleanup, and absence of retained bundles.

- [ ] **Step 5: Run construction/feature GREEN checks**

```bash
uv run pytest -q tests/unit/test_leakage.py -k "construction or feature or stream"
```

Expected: only the first vertical slice is green; probe/control/counterfactual
tests have not been written.

- [ ] **Step 6: Write failing probe-core tests**

Add deterministic group split, standardization, optimizer convergence/failure,
class balance, permutation, Holm, label-shuffled, and matched/independent mode
tests. Use 32–128 examples and 19 permutations in unit tests; assert exact
membership hashes and non-materialization of a replicate matrix.

- [ ] **Step 7: Run probe tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_leakage.py -k "probe or split or permutation or optimizer"
```

Expected: the probe engine APIs/results are absent while Step 5 remains green.

- [ ] **Step 8: Implement deterministic group-held-out probes**

For each `(suite, requested_path_length)` stratum, hash canonical
`(leakage-v1, audit_seed, corpus_hash, generation_mode, suite, path_length,
randomization_block_index)`, sort whole audit groups by full digest, and place
exactly the first 80% in train. A matched group is a shared-template validation
cohort; an independent group is a label-allocation quartet whose nuisance
draws remain episode-local. Never split either group during this shortcut
audit, and require every full-gate stratum's group count divisible by five.
This conservative audit grouping does not redefine the independent episode as
the later scientific bootstrap unit. Hash the memberships before fitting.

Standardize only continuous columns with train-set population mean/std; map a
zero-variance column to zero and do not standardize one-hot/presence columns.
Fit multinomial logistic regression with zero initialization, an unpenalized
intercept, and objective `mean class-balanced cross-entropy +
0.5 * 0.03 * ||W||^2`, using `scipy.optimize.minimize(method="L-BFGS-B")`, at
most 500 iterations, gradient tolerance `1e-8`, and function tolerance
`1e-12`. Prediction ties choose the lowest enum value. There is no tuning.

Fit once, then stream held-out permutations without refitting. Binary uses one
of the six unique `P,P,N,N` assignments inside each audit group; three-way uses
one of the twelve `P,P,S,D` assignments. For the hazard-class task,
independently swap the two public hazard classes for each positive episode whose
two-class multiset is distinct; duplicate-class episodes remain fixed and do not
contribute permutation variance. This preserves every episode/group class
multiset and never assigns an impossible target. Each choice comes from
SHA-256 of the canonical `AuditSeedKey`; never materialize the full replicate
matrix.

Use balanced accuracy with chance `1/K` and:

```python
p_value = (1 + sum(value >= observed for value in permuted_values)) / (replicates + 1)
```

Apply Holm correction across the complete applicable task/group family. A
clean full-gate probe fails only when balanced accuracy exceeds `1/K` and
adjusted `p < 0.01`. Missing classes/strata, insufficient test examples,
nonfinite data, optimizer failure, or a failed positive control is a hard
failure. The `test` profile uses 1,200 matched-validation episodes/199
permutations only to test mechanics and control power: exactly the first 100
IID cohorts at each path length 2, 3, and 4 after audit-hash ordering. Its
`enforce_clean_statistical_gate=false` means it
cannot claim absence of leakage. The acceptance profile uses all 100,000
independent gate episodes/4,999 permutations. Its nine positive controls use a
whole-allocation-quartet 8,000-episode subset fixed before injection and 4,999
permutations: take 125 audit-hash-ranked quartets from each of the 16 suite/path
strata. This yields exactly 2,000 independent-nuisance allocation quartets
while preserving exact `[P,P,S,D]` allocation and binary balance, and every
stratum remains divisible by five for an exact 80/20 group split. The held-out
control set therefore has 1,600 episodes,
including 800 positives. `PC_HAZARD_LAYOUT_BY_CLASS` assigns its injected
hazard targets by frozen group rank and positive position so both train and
test have exactly equal four-class counts; the held-out set has 200 per class
and satisfies the shared minimum without an exception. The complete
confirmatory family has at most
`3 tasks * 9 feature groups = 27` tests. Therefore the minimum attainable raw
permutation value is `1/5000` and even the worst Holm multiplier gives
`27/5000 = 0.0054 < 0.01`; the gate must reject a configuration whose
replicate count cannot attain its declared family-wise threshold.

Positive-control membership is a two-stage frozen operation. First select the
profile's clean base episodes and hash their ordered clean subset. The full
profile uses the 8,000 episodes above; the test profile uses its declared 1,200
matched episodes. Within each
stratum, rank whole quartets with canonical
`H("silent-cascade/ofd-v1/pc-split/v1", positive_control_seed,
clean_subset_corpus_hash, suite, path, group_id)` and freeze the exact 80/20
memberships. Only then apply one isolated injector overlay. Every control reuses
those same memberships; never include the injected corpus hash in the split
key. Compute both clean-subset and injected hashes with Task 2's helper in the
unchanged selected-source order. Record both the clean base-subset hash and injected corpus hash plus the
membership hash in `PositiveControlResult`. For the hazard control, cycle the
injected targets separately over frozen positive positions in train and test,
yielding exactly 800 and 200 examples per class respectively.
The test profile uses the same separate cycling and therefore has 120 train and
30 held-out positives per hazard class, above its eight-example mechanical
floor.

- [ ] **Step 9: Run probe-core GREEN checks**

```bash
uv run pytest -q tests/unit/test_leakage.py -k "probe or split or permutation or optimizer or shuffled"
```

Expected: probe core and earlier construction/feature slices are green.

- [ ] **Step 10: Write failing named-control tests**

Add one positive-control source wrapper for each injected leak: record count,
timestamp, terminal order, node range, record ID, public ID, common delay,
hazard-class layout, and manifest order. Every injected source must fail while
the label-shuffled clean source passes. The expected detector ID is the exact
`task:feature_group` pair, not merely `COMBINED`, so each feature family proves
that it can detect its named leak.

```python
@pytest.mark.parametrize(
    "injector",
    [
        leak_record_count,
        leak_timestamp,
        leak_terminal_order,
        leak_node_range,
        leak_record_id,
        leak_public_id,
        leak_delay_layout,
        leak_hazard_layout,
        leak_manifest_order,
    ],
)
def test_leakage_positive_controls_are_detected(
    clean_source, injector, config, provenance, tmp_path
) -> None:
    report = audit_leakage(
        injector.apply(clean_source),
        config.data.leakage_audit,
        LeakageAuditProfileName.TEST,
        provenance,
        tmp_path,
        positive_control=injector.control_id,
    )
    assert not report.passed
    control = next(item for item in report.positive_controls
                   if item.control_id == injector.control_id)
    assert control.target_task is injector.target_task
    assert injector.expected_detector_id in control.observed_detector_ids
    assert control.passed
```

- [ ] **Step 11: Run named-control tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_leakage.py -k "positive_control"
```

- [ ] **Step 12: Implement named positive controls**

Each injector may read private labels only inside positive-control mode and
must write a deterministic, disjoint code. Overlapping distribution shifts are
not valid controls for the universal `0.95` power threshold. Implement these
exact codes and targeted detectors:

```text
control                         target task             disjoint encoding -> detector
PC_COUNT_BY_LABEL               POSITIVE_BINARY         total records 48 vs 56 -> COUNTS
PC_ACTIVATION_GAP_BY_LABEL      POSITIVE_BINARY         gap 1.0 vs 4.0 -> TIMES
PC_TERMINAL_ORDER_BY_VARIANT    VARIANT_THREE_WAY       absolute slots 0..2 HHS/HSH/SHH -> ORDER_RECORD_IDS
PC_ACTIVATION_ID_BY_LABEL       POSITIVE_BINARY         start node 0 vs 63 via full bijection -> ACTIVATION_NODE
PC_RECORD_ID_BY_VARIANT         VARIANT_THREE_WAY       ID bands 0/128/256 + local rank -> ORDER_RECORD_IDS
PC_PUBLIC_ID_BY_LABEL           POSITIVE_BINARY         UUID first byte 0 vs 255, unique suffix -> ID_POSITION
PC_DELAY_BY_LABEL               POSITIVE_BINARY         both terminal delays 1.0 vs 1024.0 -> TERMINAL_MULTISET
PC_HAZARD_LAYOUT_BY_CLASS       POSITIVE_HAZARD_CLASS   SAFE offset 0/1/2/3 in absolute slots 0..3 -> ORDER_RECORD_IDS
PC_MANIFEST_ORDER_BY_VARIANT    VARIANT_THREE_WAY       contiguous P/S/D rank blocks -> ID_POSITION
```

For `PC_COUNT`, rewrite only unreachable padding LINKs to reach the exact total;
never remove a relevant record. For `PC_TERMINAL_ORDER`, reserve absolute
presentation slots `0..2` for the terminal records before writing the
variant-specific `HHS`/`HSH`/`SHH` code. For `PC_HAZARD_LAYOUT`, ensure one
unreachable sentinel LINK exists, reserve absolute presentation slots `0..3`
for the two hazards, the SAFE record, and that sentinel, and place SAFE at the
zero-based hazard-class offset. Fill the other three slots in stable
`HAZARD`, `HAZARD`, `LINK` order, and rewrite the unreachable hazard class to
`(target + 1) % 4` so every positive is permutation-active. This remains
perfectly decodable after the hazard-class identity
channels are removed from the hazard-class probe. For record-ID bands, IDs may
be construction-invalid by design and are admitted only by the explicit
positive-control path. When forcing the public-ID first byte, derive the other
15 bytes from the unique corpus episode position under a control-specific
domain so injected IDs cannot collide.

Construction-invalid controls continue to feature extraction only under an
explicit positive-control mode, while retaining the expected construction
failure ID. Each injector's preflight test proves its targeted feature is
disjoint on both train and test groups before fitting. On the full profile, the
exact targeted task/group detector must reach balanced accuracy at least
`0.95` with Holm-adjusted `p < 0.01`. The label-shuffled negative
control group-permutes train/test labels independently, refits, and must reject
no probe at the frozen seed. In the 199-permutation test profile, named-control
`passed` requires the expected detector, balanced accuracy at least `0.95`, and
raw `p <= 0.05`; it deliberately makes no Holm-corrected leakage claim.

- [ ] **Step 13: Run named-control GREEN checks**

```bash
uv run pytest -q tests/unit/test_leakage.py -k "positive_control or shuffled"
```

- [ ] **Step 14: Write failing counterfactual audit tests**

Add delay-swap, presentation-permutation, and paired-clock cases independently;
first confirm each fails because its typed audit result is absent. Also require
the report validator to reject a missing/duplicate check ID and require the
three result payload hashes to be deterministic. Freeze both pair/result
known-answer hashes declared above.

- [ ] **Step 15: Run counterfactual tests and confirm RED**

```bash
uv run pytest -q tests/unit/test_leakage.py -k "counterfactual or clock"
```

- [ ] **Step 16: Implement terminal-delay and clock counterfactual audits**

Swap the common public terminal delay across paired episodes and require oracle
action target/window to follow the swapped fact. Permute record presentation
and require oracle semantics to remain stable. Apply paired `0.1x`/`10x`
transforms and require unchanged decision plus identical normalized windows.
Populate the three exact `CounterfactualCheckResult` entries and their canonical
aggregate payload hashes; any mismatch makes that entry and the report fail.
On the full independent gate, pair all 50,000 positives exactly once within
their suite/path stratum in canonical source order (25,000 delay-swap pairs),
check one independently keyed presentation permutation for each of the 100,000
base episodes, and check all declared 5,000 `0.1x` plus 2,000 `10x`
parent/child pairs. The test profile checks every available fixture pair and
asserts nonzero denominators; it does not substitute those small denominators
for the frozen full-gate counts.

- [ ] **Step 17: Run complete GREEN checks**

```bash
uv run pytest -q tests/unit/test_leakage.py tests/property/test_generator_properties.py
uv run ruff check src/silent_cascade/env/leakage.py tests/unit/test_leakage.py tests/property/test_generator_properties.py
uv run ruff format --check src/silent_cascade/env/leakage.py tests/unit/test_leakage.py tests/property/test_generator_properties.py
```

Expected: clean and shuffled controls pass, every injected leak fails, and Ruff
is clean.

- [ ] **Step 18: Commit**

```bash
git add src/silent_cascade/env/leakage.py tests/unit/test_leakage.py \
  tests/property/test_generator_properties.py
git commit -m "feat: audit OFD generator shortcuts"
```

---

### Task 16: Phase 1 Service Orchestration and Evidence Reports

**Files:**

- Modify: `src/silent_cascade/env/services.py`
- Modify: `tests/integration/test_phase1_services.py`

**Interfaces:**

- Consumes: all Phase 1 APIs through Task 15, including Task 2's shared corpus
  digest.
- Produces `ConfigSelection`, discriminated `CorpusSource`, the four request
  models, `ArtifactPublication`, `ManifestFreezeReport`,
  `EpisodeInspectionReport`, `ExactRandomCheck`, `OracleEvaluationReport`, the
  three result wrappers, and complete service implementations for freeze,
  inspect, oracle evaluation, and leakage audit.

Use exact request/source models; seed options exist only for allocation mode:

```python
@dataclass(frozen=True, slots=True)
class Phase1ServiceDependencies:
    production_mode: bool
    validation_allocation: CohortAllocation
    independent_allocation: IndependentAllocation
    collect_provenance: EvidenceProvenanceCollector
    build_manifest: Callable[..., EpisodeManifest]
    regenerate_manifest_entry: Callable[..., EpisodeBundle]
    iter_independent: Callable[..., Iterator[IndependentEpisodeRequest]]
    generate_independent: Callable[..., EpisodeBundle]
    build_audit_source: Callable[..., ReiterableAuditSource]
    create_audit_workspace: Callable[[], ContextManager[Path]]
class ConfigSelection(StrictModel):
    base_path: Path = Path("configs/base.yaml")
    data_path: Path = Path("configs/data/primary.yaml")
    set_overrides: tuple[str, ...] = ()


class ManifestCorpusSource(StrictModel):
    mode: Literal["manifest"] = "manifest"
    manifest_path: Path


class Phase1GateCorpusSource(StrictModel):
    mode: Literal["phase1_gate"] = "phase1_gate"
    allocation_id: str = Field(min_length=1)
    root_seed: int
    public_id_seed: int


type CorpusSource = Annotated[
    ManifestCorpusSource | Phase1GateCorpusSource,
    Field(discriminator="mode"),
]


class FreezeValidationRequest(StrictModel):
    config: ConfigSelection
    output_path: Path
    episode_count: int = Field(default=10_000, gt=0, multiple_of=4)
    root_seed: int
    public_id_seed: int


class InspectEpisodeRequest(StrictModel):
    config: ConfigSelection
    manifest_path: Path
    episode_public_id: UUID | None = None
    entry_index: int | None = Field(default=None, ge=0)
    include_oracle: bool = False

    @model_validator(mode="after")
    def require_one_selector(self) -> Self:
        if (self.episode_public_id is None) == (self.entry_index is None):
            raise ValueError("select exactly one of episode_public_id or entry_index")
        return self


class OracleEvaluationRequest(StrictModel):
    config: ConfigSelection
    source: CorpusSource
    output_path: Path | None = None


class LeakageAuditRequest(StrictModel):
    config: ConfigSelection
    source: CorpusSource
    profile: LeakageAuditProfileName
    output_path: Path | None = None
```

The persisted report excludes invocation-dependent publication state:

```python
class ArtifactPublication(StrictModel):
    path: str
    created: bool
    payload_sha256: HexDigest
    file_sha256: HexDigest


class ExactRandomCheck(StrictModel):
    successes: int = Field(ge=0)
    total: int = Field(gt=0)
    observed_rate: float = Field(ge=0.0, le=1.0)
    expected_rate: Literal[0.125, 0.5]
    absolute_error: float = Field(ge=0.0, le=1.0)
    exact_binomial_p: float = Field(ge=0.0, le=1.0)
    passed: bool


class OracleInspection(StrictModel):
    terminal_kind: OracleTerminalKind
    node_path: tuple[int, ...]
    support_record_ids: tuple[int, ...]
    hazard_type: int | None
    action_window_start: float | None
    action_window_end: float | None
    action_target: float | None


class EpisodeInspectionReport(StrictModel):
    schema_version: Literal["episode-inspection-report-v1"]
    manifest_payload_sha256: HexDigest
    access_class: ManifestAccessClass
    entry_index: int
    episode_public_id: str
    public_episode: PublicEpisodeArtifact
    oracle: OracleInspection | None = None


class ManifestFreezeReport(StrictModel):
    schema_version: Literal["manifest-freeze-report-v1"]
    manifest_payload_sha256: HexDigest
    provenance: EvidenceProvenance
    access_class: Literal[
        ManifestAccessClass.DEBUG, ManifestAccessClass.VALIDATION
    ]
    episode_count: int = Field(gt=0, multiple_of=4)
    cohort_count: int = Field(gt=0)
    positive_count: int = Field(gt=0)
    safe_negative_count: int = Field(gt=0)
    disconnected_negative_count: int = Field(gt=0)


class OracleEvaluationReport(StrictModel):
    schema_version: Literal["oracle-evaluation-report-v2"]
    provenance: EvidenceProvenance
    namespace_evidence: ConstructionNamespaceEvidence
    source_mode: Literal["manifest", "phase1_gate"]
    requested_episode_count: int
    verified_episode_count: int
    positive_count: int
    safe_negative_count: int
    disconnected_negative_count: int
    suite_path_denominators: dict[str, int]
    invariant_failures: Literal[0]
    oracle_ambiguities: Literal[0]
    seed_token_collisions: Literal[0]
    public_id_collisions: Literal[0]
    oracle_successes: int
    oracle_failures: Literal[0]
    random_positive: ExactRandomCheck
    random_negative: ExactRandomCheck
    random_pooled_observed_rate: float
    random_pooled_expected_rate: Literal[0.3125]
    clock_0_1x_episode_count: int
    clock_10x_episode_count: int
    clock_decision_mismatches: Literal[0]
    rejection_reason_counts: dict[str, int]
    generation_attempt_count: int
    rejected_draw_count: int
    rejected_draw_rate: float
    corpus_sha256: HexDigest
    passed: bool


class ManifestFreezeResult(StrictModel):
    report: ManifestFreezeReport
    publication: ArtifactPublication


class OracleEvaluationResult(StrictModel):
    report: OracleEvaluationReport
    publication: ArtifactPublication | None


class LeakageAuditResult(StrictModel):
    report: LeakageReport
    publication: ArtifactPublication | None
```

Both report models are derived evidence, not permissive storage bags.
`ExactRandomCheck` rejects booleans/coercions through `StrictModel`, requires
`successes <= total` and finite fields, and recomputes exactly:

```python
observed_rate = successes / total
absolute_error = abs(observed_rate - expected_rate)
exact_binomial_p = scipy.stats.binomtest(
    successes, total, expected_rate, alternative="two-sided"
).pvalue
passed = exact_binomial_p >= 0.001 and absolute_error <= 0.01
```

The serialized values must equal those recomputed values. The oracle report
likewise derives and cross-checks: requested and verified episode counts; the
exact variant sum; the suite/path denominator sum; oracle success/failure sum;
positive and negative random totals; pooled random observed and expected rates;
nonnegative clock, rejection, and generation counters;
`generation_attempt_count == namespace_evidence.accepted_draw_count +
rejected_draw_count`; `rejected_draw_rate == rejected_draw_count /
generation_attempt_count`; exact sum of rejection-reason counts; namespace
generation mode/count/token/ID laws; equality of the top-level rejection and
attempt counters to the namespace fields; and outer `passed` from all invariant/
oracle/collision/clock/random conditions. For an independent source, an
accepted draw is one episode; for a matched source, it is one complete cohort,
so a 10,000-episode validation source has 2,500 accepted draws rather than
10,000. Every rejected independent episode attempt or matched cohort attempt
appends one reason to the private rejection-reason sequence; a matched failure
is recorded once per cohort attempt, never once per member/candidate. Storing
only distinct reason names is invalid. The standalone verifier recomputes every
relationship derivable from primitive fields and verified artifacts, and
cross-checks redundant evidence across the five-file boundary. It does not
claim to detect an arbitrary coordinated rewrite that replaces all mutually
consistent evidence and its authenticated source history.

Production `freeze_validation` rejects any report tuple other than
`(episode_count, cohort_count, positive, safe, disconnected) =
(10_000, 2_500, 5_000, 2_500, 2_500)`. The integer schema permits small pure
service tests; it is not permission to publish a smaller validation manifest.
It also requires `request.episode_count == 10_000` and
`access_class=VALIDATION`. Injected non-production dependencies set the count to
their exact 8–16 episode allocation size and may return only `DEBUG`; a
production dependency returning DEBUG or a test dependency returning
VALIDATION is a hard error.

The service requires `oracle=None` unless inspection was authorized by access
class. No private generator seed/key/rejection field appears in this output
type. `EpisodeInspectionReport` is a user-facing view, not an evidence
artifact, and intentionally omits `EvidenceProvenance` so default inspection
cannot reveal roots. Freeze/oracle/leakage/reproducibility artifacts all embed
the shared provenance model.

- [ ] **Step 1: Write failing dependency, freeze, and inspection tests**

Use injected 8–16 episode allocations in integration tests. Cover dependency
production/test separation, request mode errors, immutable freeze
reuse/divergence, public inspection sentinel privacy, authorized validation
oracle inspection, frozen-access refusal after copy/rename, provenance, and
zero foundation calls. Assert saved bytes exclude
`ArtifactPublication.created` and every per-public-ID label/path. The production
10,000-count adapter is tested with a spy asserting it selects only
`PRODUCTION_DEPENDENCIES`; its real full-size execution is Task 18, not routine
`make verify`.

- [ ] **Step 2: Run service tests and confirm RED**

```bash
uv run pytest -q tests/integration/test_phase1_services.py
```

Expected: dependency/freeze/inspection behaviors are absent.

- [ ] **Step 3: Implement dependencies, freeze, and inspection**

Services accept strict request models and return strict report models with
these exact signatures:

```text
freeze_validation(FreezeValidationRequest, *, deps=PRODUCTION_DEPENDENCIES) -> ManifestFreezeResult
inspect_episode(InspectEpisodeRequest, *, deps=PRODUCTION_DEPENDENCIES) -> EpisodeInspectionReport
evaluate_oracle(OracleEvaluationRequest, *, deps=PRODUCTION_DEPENDENCIES) -> OracleEvaluationResult
run_leakage_audit(LeakageAuditRequest, *, deps=PRODUCTION_DEPENDENCIES) -> LeakageAuditResult
```

Construct `PRODUCTION_DEPENDENCIES` once from the exact functions and frozen
allocations named above. When `production_mode=True`, every service asserts the
canonical allocation identity/counts before work and before publication.
Tests inject an 8–16-episode structurally valid dependency set and may never
route that set through CLI or publish it beneath `manifests/validation/`.
For allocation-mode requests, require `source.allocation_id ==
deps.independent_allocation.allocation_id` before collection or iteration.
Production additionally requires `phase1-independent-gate-v1`; the installed
CLI's `--allocation phase1-gate` is only a selector token and always constructs
that exact production ID. Test services use their dependency-bound `test-` ID.
The dependency-level `build_manifest` callable has the normalized four-argument
shape `(config, provenance, root_seed, public_id_seed)`. Production binds
`build_validation_manifest`; tests bind a closure over
`build_cohort_manifest(..., access_class=DEBUG)` and their explicit test
allocation.

Resolve `ConfigSelection` once to `ResolvedConfig[Phase1Config]` and retain that
object through the service call. Pass `resolved.config` to generators/scorers
and the complete `resolved` object to `EvidenceProvenanceCollector`; no service
reconstructs a provenance object from a plain config or drops the YAML source
paths.

Implement the exact dependency object, production count guards,
`freeze_validation`, and `inspect_episode`. When `--output` is supplied,
publish the canonical report with the same
verified no-clobber semantics as manifests: identical bytes are a no-op and
different existing bytes fail. Serialize only the inner report; return
the dynamic `ArtifactPublication.created` field outside the saved bytes so a
second identical invocation remains byte-identical.

- [ ] **Step 4: Run freeze/inspection GREEN checks**

```bash
uv run pytest -q tests/integration/test_phase1_services.py -k "dependencies or freeze or inspect or access"
```

- [ ] **Step 5: Write failing oracle-evaluation service tests**

Add incomplete denominator, suite-to-`OraclePolicy`, separate random strata,
clock pair, aggregate rejection-rate, report reuse/divergence, and provenance
cases against injected 8–16 episode sources.

- [ ] **Step 6: Run oracle service tests and confirm RED**

```bash
uv run pytest -q tests/integration/test_phase1_services.py -k "oracle or random or denominator or clock"
```

- [ ] **Step 7: Implement oracle evaluation**

`evaluate_oracle` validates each regenerated episode independently, derives its
denominator from unique successfully deserialized public IDs, scores oracle and
random actions, and fails on any incomplete denominator or disagreement.
For every base episode it reconstructs the authenticated `TRACE_JITTER` stream,
calls `build_oracle_trace`, validates the complete timing contract, and scores
`trace.actions`; it may not fabricate a midpoint action directly from truth or
the timing configuration. For every declared clock child it constructs the
paired scaled episode and `scale_oracle_trace`, then compares the complete
parent/child decisions, normalized timing, actions, and scores.
It computes `random_positive` with
`scipy.stats.binomtest(k, positive_count, 0.125, alternative="two-sided")` and
`random_negative` with
`scipy.stats.binomtest(k, safe_negative_count + disconnected_negative_count,
0.5, alternative="two-sided")`; each passes
only when `p >= 0.001` and absolute rate error is at most `0.01`. It never runs
a pooled binomial test.

Stream the run and retain aggregate counters, rolling corpus hash, bounded
diagnostic examples, and rejection summaries rather than bundles/traces.
Derive actual construction tokens and public IDs from every accepted source
draw in canonical order, using exactly 20 tokens per matched cohort or 7 per
independent episode, and fail on a within-source collision. Rejection reason
sequences contain one entry per rejected draw and must aggregate exactly to
`rejected_draw_count`; matched mode counts one rejected cohort attempt even
when multiple member/candidate checks fail, while independent mode counts one
rejected episode attempt. The accepted-attempt runs independently derive that
count and require `generation_attempt_count == accepted_draw_count +
rejected_draw_count`.
The rolling `corpus_sha256` is Task 2's exact hash in canonical source order.
For allocation mode it covers only the requested 100,000 base episodes; the
reported paired clock checks are derived counterfactuals and are excluded.
Select `OraclePolicy` only through Task 5's authorized suite mapping. Success
reports expose shared evidence provenance and aggregate counts, never episode
indices, variants attached to public IDs, or per-episode oracle selections.

- [ ] **Step 8: Run oracle-evaluation GREEN checks**

```bash
uv run pytest -q tests/integration/test_phase1_services.py -k "oracle or random or denominator or clock"
```

- [ ] **Step 9: Write failing leakage-service tests**

Add matched/independent source construction, caller-owned temporary workspace,
two-pass cleanup, pass/failure publication, provenance/audit seeds, and report
reuse/divergence cases with a small injected audit configuration. Require the
three typed counterfactual checks and assert that oracle, leakage, and
reproducibility fixture reports over the same source share Task 2's corpus
known-answer digest.

- [ ] **Step 10: Run leakage-service tests and confirm RED**

```bash
uv run pytest -q tests/integration/test_phase1_services.py -k "leakage or audit_workspace"
```

- [ ] **Step 11: Implement leakage service orchestration**

`run_leakage_audit` builds a reiterable source, creates one bounded workspace,
passes exact `EvidenceProvenance` into Task 15, and publishes only a complete
passing or complete failing report. Always clean feature stores; preserve the
report on a scientific failure but publish nothing on corruption/incomplete
input.

- [ ] **Step 12: Run complete GREEN checks**

```bash
uv run pytest -q tests/integration/test_phase1_services.py tests/unit/test_manifest.py tests/unit/test_oracle.py tests/unit/test_leakage.py
uv run ruff check src/silent_cascade/env/services.py tests/integration/test_phase1_services.py
uv run ruff format --check src/silent_cascade/env/services.py tests/integration/test_phase1_services.py
```

- [ ] **Step 13: Commit**

```bash
git add src/silent_cascade/env/services.py tests/integration/test_phase1_services.py
git commit -m "feat: orchestrate Phase 1 data services"
```

---

### Task 17: CLI, Privacy Regressions, Local Smoke, and Documentation

**Files:**

- Modify: `src/silent_cascade/cli.py`
- Modify: `README.md`
- Modify: `docs/PLAN.md`
- Modify: `Makefile`
- Create: `scripts/verify_phase1_gate_artifacts.py`
- Create: `tests/integration/test_cli_phase1.py`
- Create: `tests/integration/test_phase1_gate_verifier.py`
- Modify: `tests/integration/test_cli_doctor.py`
- Modify: `tests/integration/test_phase0_repository.py`
- Modify: `tests/regression/test_import_boundaries.py`

**Interfaces:**

- Consumes: Task 16 request/result services, report schemas, and Phase 0 error
  renderer.
- Produces: the four nested Phase 1 command surfaces with stable human/JSON
  output, `verify_phase1_gate_artifacts`, stronger import/privacy regressions,
  local smoke coverage, and truthful Phase 1 documentation.

- [ ] **Step 1: Write failing command, gate-verifier, and privacy tests**

```python
def test_cli_exposes_only_implemented_phase1_commands() -> None:
    assert {command.name for command in app.registered_commands} == {"doctor"}
    assert {group.name for group in app.registered_groups} == {
        "data", "episode", "oracle", "leakage"
    }
    assert subcommands(data_app) == {"freeze"}
    assert subcommands(episode_app) == {"inspect"}
    assert subcommands(oracle_app) == {"evaluate"}
    assert subcommands(leakage_app) == {"audit"}
```

Cover JSON success as exactly one stdout object, stable stderr error/exit 1,
Typer usage/mode exit 2, no traceback/private sentinel, entry-index and UUID
selectors, frozen access refusal after rename, conditional seed options, and
absence of unimplemented commands. CLI tests monkeypatch only the service
callables at the adapter boundary; they assert the exact production request
objects and never replace `PRODUCTION_DEPENDENCIES` inside service tests. Thus
routine CLI coverage does not generate 10,000/100,000 episodes; Task 18 is the
real installed-command acceptance run.

Add failing verifier tests with tiny strict fixture artifacts. Require refusal
for a missing artifact, tampered envelope, failed inner report, source/config/
plan/generator provenance disagreement, nonzero foundation calls, absent or
failed counterfactual check, wrong denominator, and unequal independent-gate
corpus hashes. A complete consistent fixture set returns one canonical success
object. Also mutate exact primitive report fields, rejection arithmetic,
accepted-attempt runs, token/ID counts and hashes, sample-membership digests,
analysis scope, source commit ancestry, historical plan base, and Git blob mode
or content; each isolated mutation and every coordinated inconsistency that is
rederivable from raw artifacts, frozen seeds/configuration/allocation,
authenticated Git history, or another report must fail through the same
five-file API. No test may assert the impossible guarantee that an arbitrary
attacker-controlled, fully rewritten, internally consistent evidence set and
its source history is detectable without an external trust anchor.

- [ ] **Step 2: Run CLI tests and confirm RED**

```bash
uv run pytest -q tests/integration/test_cli_phase1.py tests/integration/test_phase1_gate_verifier.py tests/integration/test_cli_doctor.py tests/integration/test_phase0_repository.py tests/regression/test_import_boundaries.py
```

Expected: Phase 1 groups and the gate verifier are absent, so command,
verifier, and repository assertions fail.

- [ ] **Step 3: Add thin nested Typer adapters**

Register four `Typer(no_args_is_help=True)` groups. Use shared options:

```text
--config configs/base.yaml
--data-config configs/data/primary.yaml
--set dotted.key=value            (repeatable)
--json
```

Command-specific interfaces:

```text
data freeze
  --output PATH --root-seed INT --public-id-seed INT

episode inspect
  MANIFEST (--episode-id UUID | --entry-index INT) [--oracle]

oracle evaluate
  [--manifest PATH | --allocation phase1-gate]
  [--root-seed INT --public-id-seed INT] [--output PATH]

leakage audit
  [--manifest PATH | --allocation phase1-gate]
  [--root-seed INT --public-id-seed INT]
  --profile [test|phase1-gate] [--output PATH]
```

Reject mutually inconsistent mode options. Manifest mode forbids root/public-ID
seed options and uses the verified private recipe metadata in the manifest;
allocation mode requires both. `data freeze` is validation-only and always
uses the configured literal 10,000 size. Allocation leakage requires the full
gate profile. Long-running progress goes to stderr. JSON stdout contains one
final object. Catch `SilentCascadeError`, emit `error.to_payload()` to stderr,
and exit 1 without a traceback.

- [ ] **Step 4: Strengthen import/privacy regressions**

Use AST imports rather than runtime import side effects. Permit `env.services`
to call oracle, but forbid `env.generator`, `env.invariants`, and every future
module under `eventflow`, `memory`, `models`, and `eval.conditions` from
importing it. Assert core imports still do not load MLX/Qwen.

Add a public-projection spy test now; the full callback spy enters Phase 2 with
the engine.

- [ ] **Step 5: Implement the local gate verifier and update documentation**

`verify_phase1_gate_artifacts` accepts the five exact artifact paths. After
Correction Pass 5 it loads oracle and both reproducibility reports through
their strict v2 loaders, leakage through its strict v3 loader, and checks
frozen validation/gate counts,
every derived inner `passed` state, all three leakage counterfactual IDs, common
plan/source/config/generator provenance, zero foundation calls, and exact
equality of the oracle/leakage/independent-reproducibility corpus SHA-256. Its
result schema is exactly `phase1-gate-verification-v3`; stale v1 oracle,
stale v1/v2 leakage, stale v1 reproducibility, or stale v1/v2 outer
gate-verification artifacts are rejected with no compatibility path.

The verifier uses read-only Git object access to authenticate each exact full
`source_commit`, require it to be an ancestor of current `HEAD`, derive the
historical plan base, and recompute the framed generator and final-analysis
hashes from regular blobs at that commit. It requires every evidence artifact's
`analysis_source.scope == "phase1_analysis"`; a coordinated Task-14 scope
downgrade, nonexistent revision, symlink/tree substitution, or current-
worktree-only hash is invalid. It independently recomputes both reproducibility
sample memberships through Task 14's pure functions and independently derives
matched and independent accepted-attempt runs, construction-token sequences,
base/clock public-ID sequences, and collision checks. It requires the exact raw
matched validation public-ID seed `2026083002` and independent gate public-ID
seed `2026083012`, recomputes their provenance fingerprints, and rederives every
public ID from the authenticated coordinate, accepted attempt, and applicable
raw seed. The oracle, leakage, and independent-reproducibility namespace
summaries must be exactly equal; the matched validation reproducibility summary
is independently derived from the verified manifest. The combined gate is
exactly 750,000 tokens and 117,000 IDs, all collision-free. It recomputes every
OracleEvaluationReport v2 relationship even for unchecked model copies and
every derivable/cross-artifact relationship at the public five-file boundary.

For leakage v3, the verifier additionally decodes and authenticates the
100,000-item source digest pack, the 7,000-item clock-child digest pack, and all
nine 8,000-item injected digest packs; reconstructs source/corpus,
source-manifest, the complete v2 clock-pair manifest, main split/train/test, and
each positive-control subset/split/injected hash; independently derives every
confusion-based accuracy, permutation add-one p-value, Holm value, probe and
detector pass, all six construction checks, and all three counterfactual result
hashes; and rejects canonical leakage bytes above 16 MiB. These checks bind the
published derivations to the producer's packed raw digests and frozen recipes;
they do not claim recovery if every raw trust anchor and every redundant digest
is replaced consistently.

The verifier never checks out a revision, regenerates episode semantics,
rewrites artifacts, or accesses the network. Its script adapter emits one
canonical v3 JSON result on success and a stable typed error on stderr with exit
1 on failure.

```python
class Phase1GateVerificationResult(StrictModel):
    schema_version: Literal["phase1-gate-verification-v3"]
    validation_episode_count: Literal[10_000]
    independent_episode_count: Literal[100_000]
    matched_accepted_draw_count: Literal[2_500]
    independent_accepted_draw_count: Literal[100_000]
    matched_public_id_seed: Literal[2026083002]
    independent_public_id_seed: Literal[2026083012]
    construction_token_count: Literal[750_000]
    public_id_count: Literal[117_000]
    validation_manifest_payload_sha256: HexDigest
    independent_corpus_sha256: HexDigest
    foundation_model_calls: Literal[0]
    passed: Literal[True]
```

Keep the Make target set unchanged. Expand `smoke` to include the Phase 1 CLI
smoke, fixture oracle, gate-verifier fixture, and import-boundary tests. Keep
`verify` local-only.

Update README status to Phase 1 data/oracle infrastructure, explicitly state
that no learned benchmark result exists, and list only implemented commands.
Update `docs/PLAN.md` Phase 1 row to link this plan and use this gate:

```text
fixed matched-validation manifest + 100,000 independent-recipe
generated/validated episodes + oracle 100% + random analytic check +
leakage/reproducibility audits + make verify
```

- [ ] **Step 6: Run GREEN checks**

```bash
uv run pytest -q tests/integration/test_cli_phase1.py tests/integration/test_phase1_gate_verifier.py tests/integration/test_phase1_services.py tests/integration/test_cli_doctor.py tests/integration/test_phase0_repository.py tests/regression/test_import_boundaries.py
make smoke
uv run ruff check src/silent_cascade/cli.py scripts/verify_phase1_gate_artifacts.py tests/integration/test_cli_phase1.py tests/integration/test_phase1_gate_verifier.py tests/integration/test_cli_doctor.py tests/integration/test_phase0_repository.py tests/regression/test_import_boundaries.py
uv run ruff format --check src/silent_cascade/cli.py scripts/verify_phase1_gate_artifacts.py tests/integration/test_cli_phase1.py tests/integration/test_phase1_gate_verifier.py tests/integration/test_cli_doctor.py tests/integration/test_phase0_repository.py tests/regression/test_import_boundaries.py
```

Expected: CLI/integration/import tests, local smoke, and Ruff pass.

- [ ] **Step 7: Commit**

```bash
git add src/silent_cascade/cli.py README.md docs/PLAN.md Makefile \
  scripts/verify_phase1_gate_artifacts.py \
  tests/integration/test_cli_phase1.py \
  tests/integration/test_phase1_gate_verifier.py \
  tests/integration/test_cli_doctor.py tests/integration/test_phase0_repository.py \
  tests/regression/test_import_boundaries.py
git commit -m "feat: expose Phase 1 data and oracle commands"
```

---

## Required Final-Audit Correction Passes

The executions of Tasks 1–18 produced well-formed but insufficiently
authenticated evidence. The whole-phase audits invalidated all five artifacts.
Execute these five passes in order from the approved current-branch history,
using TDD and the exact commit subjects below. After each pass, obtain an
independent specification review and an independent code-quality review. Both
must pass before the next pass begins; correct a finding in a separate
meaningful fix commit and repeat both reviews. No pass may weaken a frozen
scientific setting, seed, denominator, control, threshold, resource ceiling,
or zero-foundation-model-call requirement.

### Correction Pass 1: Authenticate Independent Quartet Coordinates

**Files:**

- Modify: `src/silent_cascade/env/episode.py`
- Modify: `src/silent_cascade/env/generator.py`
- Modify: `src/silent_cascade/env/invariants.py`
- Modify: `src/silent_cascade/env/services.py`
- Modify: `src/silent_cascade/env/reproducibility.py`
- Modify: `src/silent_cascade/logging/manifest.py`
- Modify: `tests/unit/test_episode.py`
- Modify: `tests/unit/test_generator_spec.py`
- Modify: `tests/unit/test_independent_generator.py`
- Modify: `tests/unit/test_invariants.py`
- Modify: `tests/unit/test_manifest.py`
- Modify: `tests/unit/test_clock_scaling.py`
- Modify: `tests/unit/test_leakage.py`
- Modify: `tests/unit/test_stress_graphs.py`
- Modify: `tests/unit/test_stress_terminals.py`
- Modify: `tests/property/test_generator_properties.py`
- Modify: `tests/integration/test_phase1_services.py`
- Modify: `tests/integration/test_phase1_reproducibility.py`

**Interfaces:**

- Adds exact `quartet_member_index: int` in `0..3` to
  `IndependentEpisodeRequest` and `IndependentEpisodeCoordinate`.
- Preserves `IndependentManifestCoordinate.quartet_member_index` and makes it a
  lossless copy of the source coordinate rather than a reconstructed value.
- Makes generic independent invariants authenticate only explicit
  member/label identity; the allocation-owning service authenticates block
  membership.

- [ ] **Step 1: Write exact RED regressions**

Add strict bool/range/round-trip tests for both new fields. Add a DEBUG block
starting at episode index `5` with eight entries and assert emitted member
indices are `(0,1,2,3,0,1,2,3)`, every variant equals the allocation permutation
at that explicit member, manifest/service/reproducibility round trips preserve
it, and mutations to member, quartet, or variant fail. Add a structurally valid
`FROZEN` allocation totaling 20,000 episodes with non-Phase-1 block boundaries
and require generic generation/invariants to accept it. Add AST guards against
`episode_index % 4` and `list.index(variant)` in generator, invariant, manifest,
service, and reproducibility code. In `test_clock_scaling.py`,
`test_leakage.py`, `test_stress_graphs.py`, and `test_stress_terminals.py`, add
round-trip/mutation regressions proving every derived clock, audit, and stress
coordinate preserves the explicit quartet member and never reconstructs it.

- [ ] **Step 2: Run RED tests**

```bash
uv run pytest -q tests/unit/test_episode.py tests/unit/test_generator_spec.py \
  tests/unit/test_independent_generator.py tests/unit/test_invariants.py \
  tests/unit/test_manifest.py tests/unit/test_clock_scaling.py \
  tests/unit/test_leakage.py tests/unit/test_stress_graphs.py \
  tests/unit/test_stress_terminals.py tests/property/test_generator_properties.py \
  tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_reproducibility.py
```

Expected: explicit-member construction/round-trip/mutation tests fail against
the implicit coordinate contract.

- [ ] **Step 3: Implement the explicit identity path**

Validate exact non-boolean member indices in both frozen dataclasses.
`iter_independent_requests` forms each quartet from local block offsets and
stores `quartet_member_index`; it binds `variant =
allocate_independent_variants(label_key)[quartet_member_index]`. Propagate the
member through primary/stress generation, `EpisodeKey`, private artifacts,
manifest entries, service regeneration, fresh work orders, and reproducibility
keys. Generic invariants recompute only the label permutation from the explicit
quartet/member fields. Remove the Phase 1 gate block table and all hard-coded
PHASE1_GATE/FROZEN boundary logic from `env.invariants`; production services
validate their bound allocations and episode-index membership.

- [ ] **Step 4: Run GREEN and static checks**

```bash
uv run pytest -q tests/unit/test_episode.py tests/unit/test_generator_spec.py \
  tests/unit/test_independent_generator.py tests/unit/test_invariants.py \
  tests/unit/test_manifest.py tests/unit/test_clock_scaling.py \
  tests/unit/test_leakage.py tests/unit/test_stress_graphs.py \
  tests/unit/test_stress_terminals.py tests/property/test_generator_properties.py \
  tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_reproducibility.py
uv run ruff check src/silent_cascade/env/episode.py \
  src/silent_cascade/env/generator.py src/silent_cascade/env/invariants.py \
  src/silent_cascade/env/services.py \
  src/silent_cascade/env/reproducibility.py \
  src/silent_cascade/logging/manifest.py tests/unit/test_episode.py \
  tests/unit/test_generator_spec.py tests/unit/test_independent_generator.py \
  tests/unit/test_invariants.py tests/unit/test_manifest.py \
  tests/unit/test_clock_scaling.py tests/unit/test_leakage.py \
  tests/unit/test_stress_graphs.py tests/unit/test_stress_terminals.py \
  tests/property/test_generator_properties.py \
  tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_reproducibility.py
```

Expected: every explicit-member and generic-allocation regression passes; Ruff
is clean.

- [ ] **Step 5: Commit and independently review**

```bash
git add src/silent_cascade/env/episode.py src/silent_cascade/env/generator.py \
  src/silent_cascade/env/invariants.py src/silent_cascade/env/services.py \
  src/silent_cascade/env/reproducibility.py \
  src/silent_cascade/logging/manifest.py tests/unit/test_episode.py \
  tests/unit/test_generator_spec.py tests/unit/test_independent_generator.py \
  tests/unit/test_invariants.py tests/unit/test_manifest.py \
  tests/unit/test_clock_scaling.py tests/unit/test_leakage.py \
  tests/unit/test_stress_graphs.py tests/unit/test_stress_terminals.py \
  tests/property/test_generator_properties.py \
  tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_reproducibility.py
git commit -m "fix: authenticate independent quartet coordinates"
```

### Correction Pass 2: Enforce Feasible Oracle Trace Schedules

**Files:**

- Modify: `src/silent_cascade/env/timing.py`
- Modify: `src/silent_cascade/env/oracle.py`
- Modify: `src/silent_cascade/env/generator.py`
- Modify: `src/silent_cascade/env/invariants.py`
- Modify: `src/silent_cascade/env/services.py`
- Modify: `tests/unit/test_oracle.py`
- Modify: `tests/unit/test_generator.py`
- Modify: `tests/unit/test_independent_generator.py`
- Modify: `tests/unit/test_invariants.py`
- Modify: `tests/unit/test_counter_rng.py`
- Create: `tests/integration/test_phase1_trace_feasibility.py`
- Modify: `tests/integration/test_phase1_services.py`

**Interfaces:**

- Produces the neutral `TraceTimingSchedule` and
  `build_trace_timing_schedule` contract frozen above.
- Changes construction token counts to exactly 20 per accepted matched cohort
  and 7 per accepted independent episode.
- Makes generator acceptance label-blind and oracle evaluation consume complete
  authenticated traces.

- [ ] **Step 1: Write timing and acceptance RED tests**

Freeze pure timing known answers for difficulty, urgency, the exact
`math.exp(jitter_normals[k] * jitter_log_std)` transformation, `math.fsum`
accumulation, clamp, common scaling, infeasibility, and action-after-compose.
Reject NaN/infinity in every input and intermediate and spy that the pure
primitive has no RNG import/call. Spy separately on oracle, generator, and
invariant record/competitor derivation so none delegates graph reasoning to
another. Force one unassigned counterfactual variant infeasible while the
assigned variant is feasible and require the independent draw to reject; for a
matched member, force an unassigned candidate to fail each semantic, shape,
leakage-prevention, and timing check in turn and require one whole-cohort retry
per failed cohort attempt. Assert token
order/count `20` and `7` including `TRACE_JITTER`. Add exact gate-root
regressions for OOD-short episode `16219` (safe-negative) and `16500`
(positive), and a streaming integration test that constructs all 24,000
OOD-short gate requests and successfully builds every exact oracle trace.
Assert oracle evaluation calls `build_oracle_trace`, scores `trace.actions`, and
builds/scales clock traces rather than synthesizing a target action.

- [ ] **Step 2: Run RED tests**

```bash
uv run pytest -q tests/unit/test_oracle.py tests/unit/test_generator.py \
  tests/unit/test_independent_generator.py tests/unit/test_invariants.py \
  tests/unit/test_counter_rng.py \
  tests/integration/test_phase1_trace_feasibility.py \
  tests/integration/test_phase1_services.py
```

Expected: neutral scheduling, label-blind rejection, jitter-token, exact-index,
and trace-scoring regressions fail against the previous implementation.

- [ ] **Step 3: Implement exact shared timing math and independent derivations**

Move only the timing equations into `env.timing`. Oracle, generator, and
invariants independently derive ordered selected records and competitor counts,
draw/rederive the exact finite jitter vector, and call the pure primitive. The
primitive accepts that vector, uses `math.exp(jitter_normals[k] *
jitter_log_std)` and stable `math.fsum` accumulation, and performs no RNG work.
Before an independent attempt can be accepted, construct all three label
counterfactuals from its unchanged nuisance draw and reinitialized
`TRACE_JITTER` stream; reject unless all three candidates pass every retryable
semantic, shape, leakage-prevention, and timing condition. For every matched
member draw, do the same three-candidate check with identical member streams;
reject the complete cohort once unless all twelve candidates pass, then select
the allocation-assigned labels. Include `TRACE_JITTER` in the accepted-draw
token APIs in the frozen order.
Change oracle evaluation to build the authenticated base trace, score its
actions, construct paired clock traces, and compare complete scaled outcomes.

- [ ] **Step 4: Run GREEN and local checks**

```bash
uv run pytest -q tests/unit/test_oracle.py tests/unit/test_generator.py \
  tests/unit/test_independent_generator.py tests/unit/test_invariants.py \
  tests/unit/test_counter_rng.py \
  tests/integration/test_phase1_trace_feasibility.py \
  tests/integration/test_phase1_services.py
uv run ruff check src/silent_cascade/env/timing.py \
  src/silent_cascade/env/oracle.py src/silent_cascade/env/generator.py \
  src/silent_cascade/env/invariants.py src/silent_cascade/env/services.py \
  tests/unit/test_oracle.py tests/unit/test_generator.py \
  tests/unit/test_independent_generator.py tests/unit/test_invariants.py \
  tests/unit/test_counter_rng.py \
  tests/integration/test_phase1_trace_feasibility.py \
  tests/integration/test_phase1_services.py
```

Expected: both exact OOD-short regressions and all 24,000 streamed schedules
pass after deterministic retry; focused tests and Ruff are clean.

- [ ] **Step 5: Commit and independently review**

```bash
git add src/silent_cascade/env/timing.py src/silent_cascade/env/oracle.py \
  src/silent_cascade/env/generator.py src/silent_cascade/env/invariants.py \
  src/silent_cascade/env/services.py tests/unit/test_oracle.py \
  tests/unit/test_generator.py tests/unit/test_independent_generator.py \
  tests/unit/test_invariants.py tests/unit/test_counter_rng.py \
  tests/integration/test_phase1_trace_feasibility.py \
  tests/integration/test_phase1_services.py
git commit -m "fix: enforce feasible oracle trace schedules"
```

### Correction Pass 3: Authenticate Construction Namespaces

**Files:**

- Modify: `src/silent_cascade/provenance.py`
- Modify: `src/silent_cascade/env/generator.py`
- Modify: `src/silent_cascade/env/leakage.py`
- Modify: `src/silent_cascade/env/services.py`
- Modify: `src/silent_cascade/env/reproducibility.py`
- Modify: `scripts/verify_phase1_gate_artifacts.py`
- Modify: `tests/unit/test_provenance.py`
- Modify: `tests/unit/test_leakage.py`
- Modify: `tests/unit/test_generator.py`
- Modify: `tests/unit/test_independent_generator.py`
- Modify: `tests/integration/test_phase1_services.py`
- Modify: `tests/integration/test_phase1_reproducibility.py`
- Modify: `tests/integration/test_phase1_reproducibility_script.py`
- Modify: `tests/integration/test_phase1_gate_verifier.py`

**Interfaces:**

- Produces strict `AcceptedAttemptRun`, `ConstructionNamespaceEvidence`,
  `accepted_attempt_runs`, `construction_token_sequence_sha256`, and
  `public_id_sequence_sha256` under the already fingerprinted provenance
  module.
- Bumps `ReproducibilityReport` to `phase1-reproducibility-v2` and
  `LeakageReport` to `leakage-report-v2`; stale v1 artifacts are invalid.
- Migrates `verify_phase1_gate_artifacts.py` and its fixtures only far enough to
  parse and authenticate those two v2 inner reports. The outer
  `Phase1GateVerificationResult` deliberately remains
  `phase1-gate-verification-v1` in this independently green pass.
- Preserves every current leakage evidence field and validator while adding
  `namespace_evidence`; specifically, `PositiveControlResult.probes`,
  `LeakageReport.construction_check_ids`, and
  `LeakageReport.label_shuffled_probes` remain mandatory.

- [ ] **Step 1: Write namespace-evidence RED tests**

Test exact primitive types, raw public-ID seed/fingerprint binding,
empty/gapped/overlapping attempt runs, adjacent equal runs, incomplete
coverage, wrong count sums, invalid digests, and nonzero collisions. Freeze
domain-separated ordered-sequence known answers and prove order changes the
hash. Mutate actual token or public-ID derivation while leaving
coordinates/counts unchanged and require oracle, leakage, and reproducibility
paths to fail. Test canonical run compression and verify that accepted retries
appear exactly once per construction draw: 2,500 matched cohort draws versus
100,000 independent episode draws. A failed matched attempt contributes one
rejection reason regardless of how many member counterfactuals fail. Test
`rejected_draw_count == sum(draw_count * accepted_attempt)` and
`generation_attempt_count == accepted_draw_count + rejected_draw_count`. Test
inner leakage/reproducibility schema-v1 report rejection and exact fixture
migration to v2. Assert the v2 leakage schema and strict validators retain the
complete ordered 27 clean probes, 27 label-shuffled probes,
`construction_check_ids`, and all nine
positive controls with their complete nine-feature `probes` families. Update
the standalone-verifier fixtures and parsing assertions to consume
`ReproducibilityReport` and `LeakageReport` v2 while asserting its own result
still has the exact outer literal `phase1-gate-verification-v1`; do not add the
Pass 4 historical-Git, independent membership, raw-seed ID rederivation,
derived-oracle, or outer-v2 acceptance checks here.

- [ ] **Step 2: Run RED tests**

```bash
uv run pytest -q tests/unit/test_provenance.py tests/unit/test_leakage.py \
  tests/unit/test_generator.py tests/unit/test_independent_generator.py \
  tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_reproducibility.py \
  tests/integration/test_phase1_reproducibility_script.py \
  tests/integration/test_phase1_gate_verifier.py
```

Expected: v2 schema, attempt-run, actual-token/ID, and ordered namespace tests
fail while existing semantic generation remains unchanged.

- [ ] **Step 3: Implement streaming actual namespace evidence**

Implement the two strict models and helpers in `provenance.py`. Store each
SHA-256 token as 32 fixed bytes and each UUID as 16 fixed bytes; preserve the
canonical stream for framed hashing, sort a bounded copy, and scan adjacent
values for collisions. Build run-length encoded attempts from actual
regenerated cohorts/episodes and bind the exact raw public-ID seed plus its
provenance fingerprint. Add the coherent namespace evidence object to oracle,
leakage, matched reproducibility, and independent reproducibility reports and
derive it from the same bundles used for each corpus. Oracle and leakage
independently derive actual tokens and IDs and fail on collisions; leakage's
`seed_tokens` construction check no longer accepts coordinate tuples. Include
clock IDs in the independent public-ID evidence after base IDs in exact clock
suite/parent order. Preserve every v1 leakage evidence field and all existing
strict all-nine/27-probe/derived-pass validators while migrating the outer
leakage-report schema to v2; there is no compatibility loader. Migrate the
standalone verifier and its fixtures at the same time so they consume the v2
leakage and reproducibility reports and validate their new namespace evidence.
Keep `Phase1GateVerificationResult` and its emitted literal at
`phase1-gate-verification-v1`; Pass 3 may not pre-implement Pass 4's final
cross-artifact hardening or outer-schema bump.

- [ ] **Step 4: Run GREEN and resource checks**

```bash
uv run pytest -q tests/unit/test_provenance.py tests/unit/test_leakage.py \
  tests/unit/test_generator.py tests/unit/test_independent_generator.py \
  tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_reproducibility.py \
  tests/integration/test_phase1_reproducibility_script.py \
  tests/integration/test_phase1_gate_verifier.py
uv run ruff check src/silent_cascade/provenance.py \
  src/silent_cascade/env/generator.py src/silent_cascade/env/leakage.py \
  src/silent_cascade/env/services.py \
  src/silent_cascade/env/reproducibility.py \
  scripts/verify_phase1_gate_artifacts.py tests/unit/test_provenance.py \
  tests/unit/test_leakage.py tests/unit/test_generator.py \
  tests/unit/test_independent_generator.py \
  tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_reproducibility.py \
  tests/integration/test_phase1_reproducibility_script.py \
  tests/integration/test_phase1_gate_verifier.py
```

Expected: exact schema and namespace tests pass within the unchanged memory
ceilings; Ruff is clean.

- [ ] **Step 5: Commit and independently review**

```bash
git add src/silent_cascade/provenance.py src/silent_cascade/env/generator.py \
  src/silent_cascade/env/leakage.py src/silent_cascade/env/services.py \
  src/silent_cascade/env/reproducibility.py \
  scripts/verify_phase1_gate_artifacts.py tests/unit/test_provenance.py \
  tests/unit/test_leakage.py tests/unit/test_generator.py \
  tests/unit/test_independent_generator.py \
  tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_reproducibility.py \
  tests/integration/test_phase1_reproducibility_script.py \
  tests/integration/test_phase1_gate_verifier.py
git commit -m "fix: authenticate construction namespaces"
```

### Correction Pass 4: Harden Phase 1 Evidence Verification

**Files:**

- Modify: `src/silent_cascade/provenance.py`
- Modify: `src/silent_cascade/env/generator.py`
- Modify: `src/silent_cascade/env/services.py`
- Modify: `src/silent_cascade/env/reproducibility.py`
- Modify: `scripts/check_phase1_reproducibility.py`
- Modify: `scripts/verify_phase1_gate_artifacts.py`
- Modify: `tests/unit/test_provenance.py`
- Modify: `tests/integration/test_phase1_services.py`
- Modify: `tests/integration/test_phase1_reproducibility.py`
- Modify: `tests/integration/test_phase1_reproducibility_script.py`
- Modify: `tests/integration/test_phase1_gate_verifier.py`
- Delete: `manifests/validation/v1/ofd-primary-10000.json`
- Delete: `manifests/validation/v1/phase1-oracle-gate.json`
- Delete: `manifests/validation/v1/phase1-leakage-gate.json`
- Delete: `manifests/validation/v1/phase1-validation-reproducibility.json`
- Delete: `manifests/validation/v1/phase1-independent-reproducibility-gate.json`

**Interfaces:**

- Bumps `OracleEvaluationReport` to `oracle-evaluation-report-v2` and
  `Phase1GateVerificationResult` to `phase1-gate-verification-v2`.
- Starts from Pass 3's independently green verifier, which already consumes
  v2 leakage/reproducibility inputs while emitting outer v1. This pass alone
  adds final cross-artifact checks and bumps that outer result to v2; do not
  duplicate the inner-report migration owned by Pass 3.
- Exposes the four pure reproducibility selection/membership helpers declared in
  Task 14 and changes `_entry_for` to accept only `EpisodeBundle`.
- Adds read-only historical Git-blob authentication to final provenance and the
  five-artifact verifier.

- [ ] **Step 1: Write report, membership, and Git-authentication RED tests**

Mutate every `ExactRandomCheck` derived value/type/bound and each oracle outer
count, sum, rate, reason, collision, clock, and pass field in isolation. Add
coordinated mutations for every relationship independently derivable from raw
artifact content, frozen configuration/allocation, exact seeds, authenticated
Git source, or another artifact, and require strict model or verifier rejection;
do not claim detection of an arbitrary rewrite that consistently replaces all
of those authorities.
Expose the pure sample helpers, freeze their known answers, and coordinate-edit
stored membership hashes to prove the verifier independently recomputes them.
Build oracle, leakage, and independent-reproducibility fixtures over one source
and require exact namespace-object equality; mutate one accepted-attempt run,
raw seed, token/ID digest, or count in any single report and require rejection.
Build the matched validation reproducibility fixture with 2,500 cohort draws and
prove the verifier independently reconstructs its run/count equations.
Create disposable Git histories that cover: exact source commit; descendant
HEAD; unrelated/missing revision; plan changed before versus after source;
worktree content diverging from committed blobs; missing path; and regular,
executable, symlink, tree, and submodule modes. Require final analysis scope and
reject Task-14 scope. Retain a real matched fresh-process regression. Assert
`_entry_for(bundle)` and remove all two-argument callers.

- [ ] **Step 2: Run RED tests**

```bash
uv run pytest -q tests/unit/test_provenance.py \
  tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_reproducibility.py \
  tests/integration/test_phase1_reproducibility_script.py \
  tests/integration/test_phase1_gate_verifier.py
```

Expected: derived-report, independent-membership, historical-Git, final-scope,
v2 gate, and one-argument entry tests fail.

- [ ] **Step 3: Implement fail-closed v2 evidence verification**

Add strict derived validators to `ExactRandomCheck` and
`OracleEvaluationReport`, expand rejection reason sequences one item per draw,
and migrate report construction to v2 without dropping any current leakage
field or validator. Enforce `generation_attempt_count ==
namespace_evidence.accepted_draw_count + rejected_draw_count`; count matched
acceptance/rejection per cohort draw and independent acceptance/rejection per
episode draw. Expose and reuse the pure selection and membership helpers; make
the final verifier recompute both exact memberships from the verified manifest
and frozen allocation/descriptor. Remove the unused request from `_entry_for`
and keep the true subprocess path.

Resolve full Git commits and ancestry, derive the plan base as of source commit,
read every exact generator/final-analysis file from regular Git blobs, and
recompute framed hashes without a checkout. In the final verifier require
`phase1_analysis`, all exact v2 schemas, both namespace summaries, combined
750,000-token/117,000-ID collision-free counts, every oracle derived
relationship, both sample memberships, common provenance, corpus equality, and
zero foundation calls. Require raw matched public-ID seed `2026083002` and raw
independent public-ID seed `2026083012`, verify each provenance fingerprint,
regenerate every base/clock ID from its authenticated coordinate, accepted
attempt, and exact seed, and compare the complete ordered ID sequences/hashes.
Require byte-for-byte equality of the independent namespace summaries in the
oracle, leakage, and independent reproducibility reports; independently
rederive the matched validation reproducibility summary. Reject every stale v1
artifact without migration. This historical pass deleted the then-current
invalid artifacts before its review. Correction Pass 5's prelude separately
deletes the later five artifacts rejected by the terminal audit; no executor
may reuse either set.

- [ ] **Step 4: Run complete local verification**

```bash
uv run pytest -q tests/unit/test_provenance.py \
  tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_reproducibility.py \
  tests/integration/test_phase1_reproducibility_script.py \
  tests/integration/test_phase1_gate_verifier.py
uv run ruff check src/silent_cascade/provenance.py \
  src/silent_cascade/env/generator.py src/silent_cascade/env/services.py \
  src/silent_cascade/env/reproducibility.py \
  scripts/check_phase1_reproducibility.py \
  scripts/verify_phase1_gate_artifacts.py tests/unit/test_provenance.py \
  tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_reproducibility.py \
  tests/integration/test_phase1_reproducibility_script.py \
  tests/integration/test_phase1_gate_verifier.py
uv run ruff format --check src scripts tests
make verify
```

Expected: all focused adversarial mutations and the full local quality gate
pass; no hosted automation or foundation-model call occurs.

- [ ] **Step 5: Commit and independently review**

```bash
git add src/silent_cascade/provenance.py src/silent_cascade/env/generator.py \
  src/silent_cascade/env/services.py \
  src/silent_cascade/env/reproducibility.py \
  scripts/check_phase1_reproducibility.py \
  scripts/verify_phase1_gate_artifacts.py tests/unit/test_provenance.py \
  tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_reproducibility.py \
  tests/integration/test_phase1_reproducibility_script.py \
  tests/integration/test_phase1_gate_verifier.py
git commit -m "fix: harden Phase 1 evidence verification"
```

The Pass 4 commit included deletion of that pass's five invalid Task 18
artifacts. Its reviewed descendant was the collector for the subsequently
rejected run; it is historical input to Correction Pass 5, not Phase 1
completion evidence.

### Correction Pass 5: Close Leakage Evidence and Quartet Cardinality

The terminal audits of collector
`bee142bd08a8b7b5621ae65551280ebdeed6c1b6`, evidence commit
`4aa6eca5d25e3c6dac879850b2a0557bbe84b54e`, and fixture-only descendant
`2203b68a4c6b67f26b9aae1bdc7d1ab00f8a6e6e` found three remaining
pre-completion failures: leakage discarded the explicit independent quartet
member; probe statistics and leakage memberships were not independently
derivable; and independent manifests accepted more than four rows per
quartet. Execute the following prelude and three TDD subpasses in order. Each
subpass must be independently green and reviewed before the next. The five
Task 18 artifacts remain absent throughout.

#### Pass 5 Prelude: Invalidate the incomplete evidence

**Files:**

- Modify: `docs/superpowers/plans/2026-08-30-phase-1-generator-oracle.md`
- Modify: `docs/deviations.md`
- Modify: `docs/PLAN.md`
- Modify: `tests/integration/test_phase0_repository.py`
- Delete: `manifests/validation/v1/ofd-primary-10000.json`
- Delete: `manifests/validation/v1/phase1-oracle-gate.json`
- Delete: `manifests/validation/v1/phase1-leakage-gate.json`
- Delete: `manifests/validation/v1/phase1-validation-reproducibility.json`
- Delete: `manifests/validation/v1/phase1-independent-reproducibility-gate.json`

- [ ] **Step 1: Write and run the status RED**

Change the repository regression first so the artifact-absent branch requires
the exact in-progress Phase 1 state, no stale collector/evidence/fix hashes, the
zero-foundation-call boundary, and the
generator/oracle-engineering-not-benchmark boundary. The immediately following
plan-review amendment freezes its completed branch before collection.

```bash
uv run pytest -q \
  tests/integration/test_phase0_repository.py::test_phase0_plan_index_is_wired_to_frozen_inputs
```

Expected: FAIL because the delivery index still claims completion.

- [ ] **Step 2: Record the ruling and delete only the five artifacts**

Use `apply_patch` to mark only Phase 1 in progress, append the approved
deviation, amend this plan, and delete the exact five files. Do not touch an
analysis source, configuration, Makefile, workflow, or any other artifact.

- [ ] **Step 3: Run local verification and commit**

```bash
uv run pytest -q tests/integration/test_phase0_repository.py
make verify
git diff --check
git status --short
git add docs/superpowers/plans/2026-08-30-phase-1-generator-oracle.md \
  docs/deviations.md docs/PLAN.md tests/integration/test_phase0_repository.py \
  manifests/validation/v1/ofd-primary-10000.json \
  manifests/validation/v1/phase1-oracle-gate.json \
  manifests/validation/v1/phase1-leakage-gate.json \
  manifests/validation/v1/phase1-validation-reproducibility.json \
  manifests/validation/v1/phase1-independent-reproducibility-gate.json
git commit -m "test: invalidate incomplete Phase 1 evidence"
```

Expected: the focused test and complete local gate pass; the commit contains
exactly the four modified text/test paths and five deletions. It changes no
`PHASE1_ANALYSIS_SOURCE_PATHS` file.

#### Pass 5 plan-review amendment: Freeze the delivery-index transition

This documentation/test-only amendment executes after artifact invalidation
and before Pass 5A. It changes no analysis source and creates no evidence.

**Files:**

- Modify: `docs/superpowers/plans/2026-08-30-phase-1-generator-oracle.md`
- Modify: `docs/deviations.md`
- Modify: `tests/integration/test_phase0_repository.py`

- [ ] **Step 1: Write the two-state regression RED**

Replace the temporary artifact-absent-only assertion with a small test-only
validator and literal temporary fixtures for exactly two legal repository
states:

1. in progress: the Phase 1 gate cell equals the exact approved in-progress
   sentence and all five Task 18 artifact paths are absent; or
2. complete: all five paths are regular files, the Phase 1 gate cell exactly
   matches the v3 completion format, each named SHA-256 equals the corresponding
   file bytes, and all five cheaply parsed provenance `source_commit` values
   equal the non-invalidated collector source printed in the cell.

Reject mixed presence, missing/extra completion metadata, a mismatched file
digest, different source commits, any invalidated source revision, and an
in-progress sentence while artifacts exist. This regression hashes bytes and
reads the already-public provenance field only; it must not import or duplicate
the scientific verifier. The completed-state fixture uses temporary tiny JSON
files and never restores an invalid real artifact.

```bash
uv run pytest -q \
  tests/integration/test_phase0_repository.py::test_phase1_delivery_state_contract
```

Expected: FAIL against the old artifact-absent-only regression because its
completed and mixed-state behavior is absent.

- [ ] **Step 2: Implement the test-only validator and run GREEN**

Apply the validator to both the temporary known-answer states and the real
repository. The real repository must select the in-progress branch before
collection; after Task 18 Step 8 it selects the complete branch without a
fixture/source edit.

```bash
uv run pytest -q tests/integration/test_phase0_repository.py
make verify
git diff --check
```

Expected: both exact legal states pass, every hybrid/stale state fails in the
temporary cases, the real repository remains in progress with all five paths
absent, and the complete local gate passes.

- [ ] **Step 3: Commit the plan-review amendment**

```bash
git add docs/superpowers/plans/2026-08-30-phase-1-generator-oracle.md \
  docs/deviations.md tests/integration/test_phase0_repository.py
git commit -m "test: harden Phase 1 evidence transition"
```

Expected: one docs/test-only commit before Pass 5A. Task 18 Step 8 may modify
only `docs/PLAN.md` and the five evidence artifacts; Step 9 requires no
post-evidence regression or fixture commit.

#### Pass 5A: Authenticate leakage quartet structure

**Files:**

- Modify: `src/silent_cascade/env/leakage.py`
- Modify: `src/silent_cascade/env/services.py`
- Modify: `src/silent_cascade/logging/manifest.py`
- Modify: `src/silent_cascade/provenance.py`
- Modify: `tests/unit/test_leakage.py`
- Modify: `tests/unit/test_manifest.py`
- Modify: `tests/unit/test_provenance.py`
- Modify: `tests/property/test_generator_properties.py`
- Modify: `tests/integration/test_phase1_services.py`
- Modify: `tests/integration/test_phase1_gate_verifier.py`

**Interfaces:**

- Adds mandatory exact `quartet_member_index: int` in `0..3` to keyword-only
  `AuditExample` and `_StoredExample`. Matched examples store the matched
  member index rather than `None`; independent examples store the independent
  coordinate's explicit member. No default or inferred value is permitted.
- Moves source/clock manifest hash domains to
  `silent-cascade/ofd-v1/leakage-source-manifest/v2` and
  `silent-cascade/ofd-v1/leakage-clock-pair-manifest/v2`, internal source
  authentication to `leakage-source-auth-v2`, and the leakage provenance
  anchor to `phase1-leakage-audit-anchor-v2`.
- Migrates the standalone-verifier fixture's
  `LeakageAuditEvidenceAnchor` literal to
  `phase1-leakage-audit-anchor-v2` in this pass. Leakage remains
  `leakage-report-v2` and the outer verifier remains
  `phase1-gate-verification-v2` until their owning Pass 5B/5C migrations.
- Tightens independent manifest validation without changing public manifest
  schema 1 or the independent fact that each member may have a different
  accepted attempt.

- [ ] **Step 1: Write coordinate/cardinality RED tests**

Use real generated bundles and allocation keys. Require a quartet at absolute
episode indices `5,6,7,8`, with explicit members `0,1,2,3`, to pass both the
audit and manifest boundaries. Independently require rejection of: member /
coordinate disagreement; member / allocation-label disagreement; a duplicate
member with another absolute index; five or eight rows whose member set is
still `{0,1,2,3}`; nonconsecutive local indices; member order that does not
match the locally sorted indices; and mixed requested path lengths. Keep a
case whose four accepted attempts differ and require it to pass.

Add source- and clock-manifest known answers proving that changing only the
explicit member changes the digest. Exercise every service adapter: independent
base source, independent clock parent, independent manifest clock parent,
matched source, and matched clock parent. Add an AST regression scoped to
`_validate_audit_coordinate` and `_validate_independent_quartets` that rejects
`ast.Mod`, floor division used to recover a member, and calls to `divmod`.
In the gate-verifier fixture, require the v1 anchor literal to fail and the v2
anchor to pass while asserting that leakage and outer report versions are still
v2. This RED/GREEN changes only the fixture anchor required by the production
anchor migration; it must not pre-implement either later schema bump.

- [ ] **Step 2: Run RED tests**

```bash
uv run pytest -q tests/unit/test_manifest.py \
  tests/property/test_generator_properties.py
uv run pytest -q tests/integration/test_phase1_services.py
uv run pytest -q tests/unit/test_leakage.py
uv run pytest -q tests/integration/test_phase1_gate_verifier.py
```

Expected: the missing audit field, discarded service value, modulo inference,
source-hash omission, and permissive manifest cardinality tests fail for their
named reasons; the verifier fixture fails only because it still carries the v1
anchor.

- [ ] **Step 3: Implement lossless member/cardinality authentication**

Populate the explicit field at every constructor. Include it in both
authenticated manifest payloads. In `_validate_audit_coordinate`, compare the
field with the private coordinate and independently derive the expected
variant as:

```python
allocate_independent_variants(
    AllocationLabelKey(
        "ofd-v1",
        descriptor.split_namespace,
        row.suite,
        descriptor.root_seed,
        row.path_length,
        row.block,
    )
)[row.quartet_member_index]
```

`_validate_independent_quartets(rows, descriptor)` groups by explicit block,
requires exactly four rows and one member each, sorts by absolute episode
index, and requires pairs `(base + 0, 0)` through `(base + 3, 3)`, common
suite/path/group identity, and the exact member-to-variant allocation. It never
requires `base % 4 == 0`.

`EpisodeManifest._validate_coordinates` applies the same allocation-neutral
four-row/member/consecutive/common-path rule to independent entries. Existing
manifest-level suite/namespace validation remains. Do not require equal
accepted attempts across independent members.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest -q tests/unit/test_manifest.py tests/unit/test_provenance.py \
  tests/property/test_generator_properties.py
uv run pytest -q tests/integration/test_phase1_services.py
uv run pytest -q tests/unit/test_leakage.py
uv run pytest -q tests/integration/test_phase1_gate_verifier.py
uv run ruff check src/silent_cascade/env/leakage.py \
  src/silent_cascade/env/services.py src/silent_cascade/logging/manifest.py \
  src/silent_cascade/provenance.py tests/unit/test_leakage.py \
  tests/unit/test_manifest.py tests/unit/test_provenance.py \
  tests/property/test_generator_properties.py \
  tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_gate_verifier.py
uv run ruff format --check src tests
make verify
git diff --check
git add src/silent_cascade/env/leakage.py src/silent_cascade/env/services.py \
  src/silent_cascade/logging/manifest.py src/silent_cascade/provenance.py \
  tests/unit/test_leakage.py tests/unit/test_manifest.py \
  tests/unit/test_provenance.py tests/property/test_generator_properties.py \
  tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_gate_verifier.py
git commit -m "fix: authenticate leakage quartet structure"
```

Expected: all focused and local gates pass with the evidence paths absent.
Obtain independent specification and quality reviews; fix and repeat both
before Pass 5B.

#### Pass 5B: Publish strict leakage sufficient evidence

**Files:**

- Modify: `src/silent_cascade/env/leakage.py`
- Modify: `src/silent_cascade/env/services.py`
- Modify: `scripts/verify_phase1_gate_artifacts.py`
- Modify: `tests/unit/test_leakage.py`
- Modify: `tests/integration/test_phase1_services.py`
- Modify: `tests/integration/test_phase1_gate_verifier.py`

**Interfaces:**

- Migrates only leakage to `leakage-report-v3`, preserving every v2 field and
  adding `OrderedSha256Pack`, `LeakageConstructionStatistics`, and
  `LeakageMembershipEvidence`.
- Adds `test_confusion_counts`, `permutation_exceedance_count`, and
  `permutation_replicate_count` to every `ShortcutProbeResult`.
- Adds `base_subset_membership_sha256` and one 8,000-item
  `injected_episode_sha256s` pack to every positive control.
- Adds one canonical 7,000-item `clock_child_episode_sha256s` pack to
  `LeakageConstructionStatistics`; together with the source pack it makes the
  v2 clock-pair manifest independently reconstructable.
- Teaches the standalone verifier to parse and validate the complete v3
  schema, but deliberately retains outer
  `phase1-gate-verification-v2` until Pass 5C completes every independent
  derivation.

- [ ] **Step 1: Write sufficient-evidence RED tests**

Use this unequal-class known answer so raw and balanced accuracy cannot be
interchanged:

```text
test class counts       0=10, 1=20
diagonal correct        0=8,  1=10
raw accuracy            18/30 = 0.6
balanced accuracy       (8/10 + 10/20)/2 = 0.65
permutation exceedances 4 of 99
add-one p-value         5/100 = 0.05
```

Reject a wrong/missing confusion row or prediction column, boolean/negative/
noninteger count, row-sum mismatch, total mismatch, diagonal metric mismatch,
exceedance above replicate count, raw-p mismatch, Holm mismatch, probe pass
mismatch, detector-summary mismatch, and outer pass mismatch. Mutate the real
prior clean-probe balanced accuracy from approximately `0.49245` to `0.1` and
require schema rejection. Cover all 27 clean, 27 shuffled, and 81 control
probes, including exact 4,999-replicate Phase 1 evidence.

Add strict pack known answers and mutations for Base64 alphabet/padding,
decoded length, item count, order, payload digest, truncation, duplicate/missing
items, and the 100,000-item bound. Assert exactly one 100,000-item source pack,
one 7,000-item clock-child pack in exact `0.1x`-then-`10x` parent order, and nine
exact 8,000-item control packs. Mutate the clock-child pack alone and the live
`AuditSourceAuthentication.clock_pair_manifest_sha256` alone or the serialized
v2 anchor alone and require producer rejection. Reject a canonical leakage
artifact larger than 16 MiB at
construction/publication/loading boundaries. Reject stale `leakage-report-v2`.

Freeze the literal two-member
`base_subset_membership_sha256=d9abb33f9eeb7d56d585590565a586cb0c32f955948aee98dfafee0fb1e92bf1`
known answer defined in Task 15. Producer and verifier tests construct its
canonical payload independently and reject wrong item/source counts, missing
or extra keys, duplicate/nonmonotonic source indices, reordered items, and a
public ID that does not match its indexed source row. Add exact workload tests
showing that full-source clean/shuffled binary and three-way probes use the main
80/20 membership, hazard probes use only its positive members, and each control
uses its own 8,000-item groupwise split followed by its target-task filter.

- [ ] **Step 2: Run RED tests**

```bash
uv run pytest -q tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_gate_verifier.py
uv run pytest -q tests/unit/test_leakage.py
```

Expected: missing sufficient-statistic/pack/schema fields and underived metric
mutations fail for their named reasons.

- [ ] **Step 3: Implement bounded v3 production evidence**

Build each confusion matrix from held-out true/predicted labels. Derive raw
accuracy from the diagonal total and balanced accuracy from the mean diagonal
recall using one stable integer-statistics function; do not retain a separate
NumPy-derived value. Store the exact exceedance and replicate counts and derive
`(1 + exceedance) / (1 + replicates)`. Recompute Holm values, probe pass,
observed detector IDs, positive-control summaries, and outer pass. Implement
the same arithmetic separately in the standalone verifier.

Encode each pack as canonical standard padded Base64 of concatenated 32-byte
digests, with exact decoded length, ordered count, and payload SHA-256. The
100,000 source digests are collected during the authenticated first pass; the
7,000 clock-child digests are collected in exact `0.1x` then `10x`
authenticated parent order; each positive-control pack contains its 8,000
injected digests in selected source order. Enforce the whole-report 16 MiB
bound before atomic publication and on load.

Populate construction statistics from actual pass counters, first/second
source-manifest hashes, and the clock-child pack. Populate membership evidence
without removing the three v2 top-level fields. Keep the six existing check IDs
and booleans, but derive them from namespace, provenance,
invariant/feature/finite/second-pass counts, and equal authenticated hashes.
Derive every probe workload from its task-filtered membership rather than
copying the main split count.

Before publication, the producer must combine the base source and clock-child
packs with the live frozen source recipe to reconstruct the complete v2 clock
manifest payload. It must require that reconstructed digest to equal both the
live `AuditSourceAuthentication.clock_pair_manifest_sha256` and the serialized
v2 `LeakageAuditEvidenceAnchor.clock_pair_manifest_sha256`. This live check is
a producer responsibility and is not an input invented for the later
standalone verifier.

- [ ] **Step 4: Run transitional GREEN and commit**

```bash
uv run pytest -q tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_gate_verifier.py
uv run pytest -q tests/unit/test_leakage.py
uv run ruff check src/silent_cascade/env/leakage.py \
  src/silent_cascade/env/services.py scripts/verify_phase1_gate_artifacts.py \
  tests/unit/test_leakage.py tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_gate_verifier.py
uv run ruff format --check src scripts tests
make verify
git diff --check
git add src/silent_cascade/env/leakage.py src/silent_cascade/env/services.py \
  scripts/verify_phase1_gate_artifacts.py tests/unit/test_leakage.py \
  tests/integration/test_phase1_services.py \
  tests/integration/test_phase1_gate_verifier.py
git commit -m "fix: authenticate leakage sufficient statistics"
```

Expected: the independently green transitional verifier consumes only leakage
v3 and still emits outer v2. Evidence paths remain absent. Obtain both reviews
and resolve every finding before Pass 5C.

#### Pass 5C: Independently rederive the leakage gate

**Files:**

- Modify: `scripts/verify_phase1_gate_artifacts.py`
- Modify: `tests/integration/test_phase1_gate_verifier.py`

**Interfaces:**

- Retains the complete ordered independent requests, accepted attempts, base
  public IDs, clock public IDs, and allocation-derived labels reconstructed by
  the v2 verifier.
- Independently rederives every v3 leakage hash/count/check without importing
  a producer aggregation, split, positive-control, metric, pack, or
  counterfactual builder.
- This pass alone migrates `Phase1GateVerificationResult` to
  `phase1-gate-verification-v3`.

- [ ] **Step 1: Write verifier RED mutations and known answers**

Use a literal nonaligned-quartet fixture and hand-computed ordered hashes.
Require rejection when only one of these values changes while its strict SHA
shape remains valid: source digest pack item/order/payload; clock-child digest
pack item/order/payload; corpus hash; source-manifest hash; serialized v2
anchor clock-pair-manifest hash; split/train/test membership hash; membership
count; one positive-control subset membership, base subset corpus, split
membership, injected digest, injected corpus, or pack order; one of the six
construction booleans/counters; or any counterfactual result payload hash.
Require the 100,000 reconstructed train/test IDs to be disjoint and complete,
and check every task-filtered clean/shuffled/control workload. Require the old
outer v2 result expectation to fail.

- [ ] **Step 2: Run RED tests**

```bash
uv run pytest -q tests/integration/test_phase1_gate_verifier.py
```

Expected: membership, construction, positive-control, counterfactual, and
outer-v3 tests fail while the transitional v3 loader remains green.

- [ ] **Step 3: Implement independent full rederivation**

From the allocation, roots, raw public-ID seed, canonical accepted-attempt
runs, and explicit member coordinates, reconstruct the ordered 100,000 base
IDs and 7,000 clock IDs. Decode the source digest pack and independently
recompute the corpus hash and source-manifest v2 hash, including manifest rank,
generation mode, block, absolute episode position, explicit quartet member,
public ID, and episode digest.

Decode the 7,000-item clock-child pack in its exact `0.1x`-then-`10x`
authenticated parent order. Combine it with source-pack parent digests,
reconstructed parent/child public IDs, parent manifest ranks, and scale values
to rebuild every canonical v2 clock-pair-manifest row and the complete framed
payload. Independently hash that payload and require equality with the
serialized v2 provenance anchor. The standalone verifier does not receive the
producer's live `AuditSourceAuthentication` object. A redundant serialized
report hash may be compared only if the schema explicitly carries one; neither
such a field nor the anchor is a substitute for the child pack and independent
reconstruction.

Reproduce the frozen main group ranking and 80/20 split to derive the exact
split/train/test payload hashes and counts. For each clean and shuffled task,
apply its exact task filter before checking probe workloads; the hazard task
uses only positive members. Reproduce the balanced positive-control subset,
its canonical source-index/public-ID membership payload, and groupwise split.
For all nine named injectors, independently derive the injected public IDs from
allocation-derived labels, combine them with the control's ordered packed
digests, and derive its subset, split, injected-corpus hash, and task-filtered
probe workloads. Recompute the six construction checks from primitives.

Recreate the three exact counterfactual result streams without episode
regeneration: presentation uses every base ID in source order; delay swap pairs
allocation-derived positives within suite/path in their source order; clock
uses the reconstructed parents in exact `0.1x` then `10x` order. Derive each
pair key and result payload hash from the frozen recipe. Because the final gate
accepts only exact zero mismatch counts, it reconstructs the canonical
all-false primitive mismatch stream and rejects any nonzero report before hash
comparison; a future schema that admits nonzero evidence would have to publish
the per-pair mismatch flags. A zero mismatch count does not authorize a
placeholder result hash.

Require canonical leakage bytes at or below 16 MiB, all v3 internal
relationships, common historical provenance, the existing 750,000-token and
117,000-ID namespace gates, oracle/reproducibility v2, and exactly zero
foundation-model calls. Emit only `phase1-gate-verification-v3`.

- [ ] **Step 4: Run final GREEN and commit**

```bash
uv run pytest -q tests/integration/test_phase1_gate_verifier.py
uv run ruff check scripts/verify_phase1_gate_artifacts.py \
  tests/integration/test_phase1_gate_verifier.py
uv run ruff format --check scripts tests
make verify
git diff --check
git add scripts/verify_phase1_gate_artifacts.py \
  tests/integration/test_phase1_gate_verifier.py
git commit -m "fix: rederive leakage gate evidence"
```

Expected: every literal known answer and adversarial mutation passes, the full
local gate is green, and the five evidence paths remain absent.

#### Pass 5 final collector-source review

Obtain fresh independent science/specification and code-quality reviews over
the complete Pass 5 range. They must inspect the exact production code, not
only test summaries, and must reproduce the `5,6,7,8` nonaligned quartet,
wrong-member/wrong-label failures, eight-row manifest failure, metric mutation,
membership mutation, control-pack mutation, and counterfactual-hash mutation.
They must also confirm all 135 probe records are derivable, all eleven digest
packs are canonical/bounded, the source plus clock-child packs reconstruct the
authenticated v2 clock-pair manifest, the raw-trust-anchor limitation is
stated, and no scientific setting changed. If either review finds an issue,
commit the smallest TDD fix and repeat both reviews.

After both reviews pass, run:

```bash
make verify
git diff --check
git status --short
collector_head="$(git rev-parse HEAD)"
evidence_source_commit="$(git log -1 --format=%H "${collector_head}" -- \
  scripts/check_phase1_reproducibility.py \
  scripts/verify_phase1_gate_artifacts.py \
  src/silent_cascade/provenance.py src/silent_cascade/env/reward.py \
  src/silent_cascade/env/leakage.py src/silent_cascade/env/reproducibility.py \
  src/silent_cascade/env/services.py src/silent_cascade/io.py \
  src/silent_cascade/logging/__init__.py \
  src/silent_cascade/logging/manifest.py \
  src/silent_cascade/__init__.py src/silent_cascade/config.py \
  src/silent_cascade/env/__init__.py \
  src/silent_cascade/env/config.py src/silent_cascade/env/episode.py \
  src/silent_cascade/env/generator.py src/silent_cascade/env/invariants.py \
  src/silent_cascade/env/oracle.py src/silent_cascade/env/timing.py \
  src/silent_cascade/errors.py src/silent_cascade/hashing.py \
  src/silent_cascade/rng.py src/silent_cascade/schemas.py \
  src/silent_cascade/validation.py)"
git merge-base --is-ancestor "${evidence_source_commit}" "${collector_head}"
```

Record both hashes. The current clean HEAD may be a test-only descendant, but
`evidence_source_commit` must be the final independently reviewed commit that
touches the exact final-analysis scope. No Task 18 Step 2–7 command may change
HEAD or any analysis-source byte. Every report must record that exact source
commit, and no post-evidence source or fixture correction is allowed.

---

### Task 18: Fixed Validation Manifest and 100,000-Episode Independent Gate

**Files:**

- Create: `manifests/validation/v1/ofd-primary-10000.json`
- Create: `manifests/validation/v1/phase1-oracle-gate.json`
- Create: `manifests/validation/v1/phase1-leakage-gate.json`
- Create: `manifests/validation/v1/phase1-validation-reproducibility.json`
- Create: `manifests/validation/v1/phase1-independent-reproducibility-gate.json`
- Modify: `docs/PLAN.md`

**Interfaces:**

- Consumes: the exact clean committed and independently reviewed Correction
  Pass 5 collector state (whose history already deletes the five invalid
  artifacts) and every prior Phase 1 API. Its final-analysis source revision,
  resolved by the reviewed collector step, must be recorded as `source_commit`
  in every regenerated artifact; no later analysis-source commit may intervene.
- Produces: immutable validation/gate artifacts tied to source/config/generator
  hashes and the evidence required to start a Phase 2 plan.

Run Steps 1–7 in one shell session so the captured path array, collector
revision, evidence-source revision, and guard function remain in scope. Invoke
the guard exactly where shown; it checks committed history and also compares
the current index/worktree bytes with `collector_head`, so an uncommitted edit
to any final-analysis path aborts collection.

- [ ] **Step 1: Verify the implementation revision is clean**

```bash
git status --short
git diff --check
phase1_analysis_paths=(
  scripts/check_phase1_reproducibility.py
  scripts/verify_phase1_gate_artifacts.py
  src/silent_cascade/provenance.py
  src/silent_cascade/env/reward.py
  src/silent_cascade/env/leakage.py
  src/silent_cascade/env/reproducibility.py
  src/silent_cascade/env/services.py
  src/silent_cascade/io.py
  src/silent_cascade/logging/__init__.py
  src/silent_cascade/logging/manifest.py
  src/silent_cascade/__init__.py
  src/silent_cascade/config.py
  src/silent_cascade/env/__init__.py
  src/silent_cascade/env/config.py
  src/silent_cascade/env/episode.py
  src/silent_cascade/env/generator.py
  src/silent_cascade/env/invariants.py
  src/silent_cascade/env/oracle.py
  src/silent_cascade/env/timing.py
  src/silent_cascade/errors.py
  src/silent_cascade/hashing.py
  src/silent_cascade/rng.py
  src/silent_cascade/schemas.py
  src/silent_cascade/validation.py
)
collector_head="$(git rev-parse HEAD)"
evidence_source_commit="$(git log -1 --format=%H "${collector_head}" -- \
  "${phase1_analysis_paths[@]}")"
assert_phase1_analysis_immutable() {
  if ! {
    test "$(git rev-parse HEAD)" = "${collector_head}" &&
      test "$(git log -1 --format=%H "${collector_head}" -- \
        "${phase1_analysis_paths[@]}")" = "${evidence_source_commit}" &&
      git merge-base --is-ancestor \
        "${evidence_source_commit}" "${collector_head}" &&
      git diff --quiet "${evidence_source_commit}" "${collector_head}" -- \
        "${phase1_analysis_paths[@]}" &&
      git diff --cached --quiet "${collector_head}" -- \
        "${phase1_analysis_paths[@]}" &&
      git diff --quiet "${collector_head}" -- \
        "${phase1_analysis_paths[@]}"
  }; then
    echo "Phase 1 analysis source changed; aborting evidence collection" >&2
    exit 1
  fi
}
assert_phase1_analysis_immutable
uv sync --locked --group dev
make verify
assert_phase1_analysis_immutable
```

Expected: status/diff checks print nothing; locked sync, lint, all tests, doctor,
and both package builds pass locally. Record the full
`collector_head` and `evidence_source_commit`; require the latter to be the
independently reviewed final-analysis revision from Pass 5 and the former to be
its clean reviewed descendant whose tree omits the stale artifacts. Every Step
2–7 collector must run without changing `HEAD` or analysis bytes and must emit
that exact full source hash.

- [ ] **Step 2: Create the fixed validation manifest once**

Run from the exact clean current `collector_head` captured in Step 1. Do not
check out `evidence_source_commit` when it is an earlier reviewed
analysis-source revision; a test-only reviewed descendant is a valid collector
HEAD. Immediately before and after the command, require current HEAD to remain
`collector_head` and recompute the exact final-analysis path fingerprint/last
touch to prove all analysis bytes and `evidence_source_commit` are unchanged.
The manifest and every later report must record `evidence_source_commit`, not
the test-only collector descendant:

```bash
assert_phase1_analysis_immutable
uv run silent-cascade data freeze \
  --config configs/base.yaml \
  --data-config configs/data/primary.yaml \
  --output manifests/validation/v1/ofd-primary-10000.json \
  --root-seed 2026083001 \
  --public-id-seed 2026083002 \
  --json
assert_phase1_analysis_immutable
```

Expected: exit 0, 10,000 unique entries, 5,000 positive, 2,500 safe-negative,
2,500 disconnected-negative, verified hashes, and zero foundation calls.

Run the same collector once more with identical arguments, guarding both sides.
Expected: verified no-op, `created=false`, unchanged bytes/mtime:

```bash
assert_phase1_analysis_immutable
uv run silent-cascade data freeze \
  --config configs/base.yaml \
  --data-config configs/data/primary.yaml \
  --output manifests/validation/v1/ofd-primary-10000.json \
  --root-seed 2026083001 \
  --public-id-seed 2026083002 \
  --json
assert_phase1_analysis_immutable
```

Change the root seed while retaining the path and require the refusal rather
than allowing the shell to treat it as an optional diagnostic. Expected: exit
1 with `manifest_error` and unchanged bytes:

```bash
assert_phase1_analysis_immutable
if uv run silent-cascade data freeze \
  --config configs/base.yaml \
  --data-config configs/data/primary.yaml \
  --output manifests/validation/v1/ofd-primary-10000.json \
  --root-seed 2026083003 \
  --public-id-seed 2026083002 \
  --json; then
  echo "expected divergent existing manifest refusal" >&2
  exit 1
fi
assert_phase1_analysis_immutable
```

- [ ] **Step 3: Inspect public and authorized oracle views**

Select the first deterministic manifest entry without manual ID substitution:

```bash
assert_phase1_analysis_immutable
uv run silent-cascade episode inspect \
  manifests/validation/v1/ofd-primary-10000.json \
  --entry-index 0 \
  --json

uv run silent-cascade episode inspect \
  manifests/validation/v1/ofd-primary-10000.json \
  --entry-index 0 \
  --oracle \
  --json
assert_phase1_analysis_immutable
```

Expected: public output contains facts/activation only; authorized validation
oracle output contains path/window/support. The public output contains none of
the private sentinel field names from Task 2.

- [ ] **Step 4: Run the exact streaming oracle/random gate**

```bash
assert_phase1_analysis_immutable
uv run silent-cascade oracle evaluate \
  --config configs/base.yaml \
  --data-config configs/data/primary.yaml \
  --allocation phase1-gate \
  --root-seed 2026083011 \
  --public-id-seed 2026083012 \
  --output manifests/validation/v1/phase1-oracle-gate.json \
  --json
assert_phase1_analysis_immutable
```

Expected report:

```text
generation mode                           independent
generator version                         ofd-v1
plan/source revisions                     recorded / clean
requested episodes                         100000
unique accepted/deserialized denominator   100000
positive / safe / disconnected             50000 / 25000 / 25000
invariant failures                         0
oracle ambiguities                         0
actual construction-token collisions      0
public-ID collisions                       0
accepted construction draws                100000
raw public-ID seed                          2026083012
actual construction tokens                 700000
base / clock public IDs                    100000 / 7000
rejected draws / generation attempts       recorded / recorded
rejected draw rate                         recorded
oracle timed success                       100000 / 100000
random analytic success                    0.3125
random positive successes / total          recorded / 50000
random positive exact-binomial p-value      >= 0.001
random positive absolute rate error         <= 0.01
random negative successes / total          recorded / 50000
random negative exact-binomial p-value      >= 0.001
random negative absolute rate error         <= 0.01
foundation-model calls                     0
```

The report also records suite/path denominators, rejection reasons/rates,
exact `generation_attempt_count == 100000 + rejected_draw_count`, source
commit/dirty state, config hash, generator version/source hash, corpus hash,
the raw public-ID seed, and its fingerprint. It uses
`oracle-evaluation-report-v2`, embeds complete independent
`ConstructionNamespaceEvidence`, builds and
scores the authenticated trace for every base episode, constructs and compares
paired clock traces, and requires all 24,000 OOD-short traces—including exact
indexes `16219` and `16500`—to be feasible after deterministic rejection.

- [ ] **Step 5: Run the exact streaming leakage gate**

```bash
assert_phase1_analysis_immutable
uv run silent-cascade leakage audit \
  --config configs/base.yaml \
  --data-config configs/data/primary.yaml \
  --allocation phase1-gate \
  --root-seed 2026083011 \
  --public-id-seed 2026083012 \
  --profile phase1-gate \
  --output manifests/validation/v1/phase1-leakage-gate.json \
  --json
assert_phase1_analysis_immutable
```

Expected: `generation_mode=independent`, `schema_version=leakage-report-v3`;
every exact construction check, held-out probe, actual 700,000-token and
107,000-ID namespace check,
terminal-delay-swap, presentation-permutation, and paired-clock counterfactual,
label-shuffled control, and injected-leak positive control passes. The report
contains all three typed counterfactual result IDs exactly once, all mandatory
v1-carried evidence (`construction_check_ids`, the complete 27-probe clean and
label-shuffled families, and every positive control's nine-feature `probes`),
and complete independent `ConstructionNamespaceEvidence` with raw seed
`2026083012`. All 135 probes contain strict confusion and permutation
sufficient statistics; the source has one 100,000-digest ordered pack and the
clock suite has one ordered 7,000-child-digest pack, while the nine controls
each have one 8,000-digest injected pack: eleven packs total. The report contains
derived `LeakageConstructionStatistics` and `LeakageMembershipEvidence`,
reconstructable source/corpus/main-split/control hashes and a reconstructable
complete v2 clock-pair-manifest hash, exact v2 source/clock authentication
domains and anchor, and canonical bytes no larger than 16 MiB. Any
detector failure, missing stratum/check, underpowered test, pack/statistic/hash
mismatch, or oversize artifact exits nonzero and blocks Phase 1.

- [ ] **Step 6: Run and save deterministic generation-order checks**

First verify the matched validation artifact:

```bash
assert_phase1_analysis_immutable
uv run python scripts/check_phase1_reproducibility.py \
  --config configs/base.yaml \
  --data-config configs/data/primary.yaml \
  --manifest manifests/validation/v1/ofd-primary-10000.json \
  --sample-size 1000 \
  --chunk-size 1 \
  --chunk-size 3 \
  --chunk-size 7 \
  --python-hash-seed 0 \
  --python-hash-seed 1 \
  --verify-all-source-entries \
  --output manifests/validation/v1/phase1-validation-reproducibility.json
assert_phase1_analysis_immutable
```

The harness runs forward, reverse, one-at-a-time, every declared chunk size,
and a fresh subprocess for every declared `PYTHONHASHSEED`. It requires the
same ordered episode hashes/corpus hash in all modes, verifies all 10,000
manifest entries, and uses no shell interpolation or network access.

Then gate the independent recipe that Phase 6 will instantiate:

```bash
assert_phase1_analysis_immutable
uv run python scripts/check_phase1_reproducibility.py \
  --config configs/base.yaml \
  --data-config configs/data/primary.yaml \
  --allocation phase1-gate \
  --root-seed 2026083011 \
  --public-id-seed 2026083012 \
  --sample-size 1000 \
  --chunk-size 1 \
  --chunk-size 3 \
  --chunk-size 7 \
  --python-hash-seed 0 \
  --python-hash-seed 1 \
  --verify-all-source-entries \
  --output manifests/validation/v1/phase1-independent-reproducibility-gate.json
assert_phase1_analysis_immutable
```

Expected: all 100,000 independent requests regenerate with identical IDs,
accepted attempts, episode hashes, exact 7-token construction sequences, and
aggregate corpus hash; `phase1-reproducibility-v2` records canonical attempt
runs, 700,000 tokens, 100,000 base IDs, and 7,000 clock IDs; the
selected 1,000 also match in every order/chunk/fresh-process mode. Any mismatch
in either run exits nonzero and publishes no report.

Require the oracle, leakage, and independent-reproducibility reports for the
shared Phase 1 gate source to contain the identical Task 2 corpus SHA-256 and
byte-for-byte identical independently built namespace evidence: raw seed
`2026083012`, 100,000 accepted episode draws, canonical attempt runs, 700,000
tokens, 100,000 base IDs, and 7,000 clock IDs. A cross-report mismatch is an
acceptance failure even when each report passes in isolation. The matched v2
report records raw seed `2026083002`, 2,500 accepted cohort draws, canonical
cohort-attempt runs, 50,000 tokens, and 10,000 base IDs. For both modes,
generation attempts equal accepted draws plus rejected draws and each rejected
matched cohort attempt is counted once, not per member. Both reports'
sample-membership digests are independently recomputable through the frozen
pure helper APIs.

- [ ] **Step 7: Run pre-freeze evidence verification**

```bash
assert_phase1_analysis_immutable
uv run python scripts/verify_phase1_gate_artifacts.py \
  --validation manifests/validation/v1/ofd-primary-10000.json \
  --oracle manifests/validation/v1/phase1-oracle-gate.json \
  --leakage manifests/validation/v1/phase1-leakage-gate.json \
  --validation-reproducibility manifests/validation/v1/phase1-validation-reproducibility.json \
  --independent-reproducibility manifests/validation/v1/phase1-independent-reproducibility-gate.json
assert_phase1_analysis_immutable
uv sync --locked --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest -q tests/property tests/regression tests/integration/test_cli_phase1.py tests/integration/test_phase1_services.py
uv run silent-cascade doctor
uv build
git diff --check
assert_phase1_analysis_immutable
```

Expected: the cross-artifact gate verifier, locked sync, lint, formatting,
doctor, source/wheel build, focused property/regression/integration tests, and
diff check pass locally. Do not run `make verify` in this transient state: all
five evidence files now exist while the delivery index is intentionally still
in progress, so its pre-frozen two-state regression must fail until Step 8
atomically supplies exact completion metadata. The verifier emits
`phase1-gate-verification-v3`, authenticates historical Git blobs and the
historical plan base, requires final `phase1_analysis` scope, independently
recomputes both sample memberships and both namespace summaries, and confirms
the combined 750,000 tokens and 117,000 IDs have zero collisions. It requires
raw public-ID seeds `2026083002` and `2026083012`, verifies their fingerprints,
rederives every base/clock ID from the authenticated coordinate and accepted
attempt, reconstructs every v3 leakage metric/hash/check and all three
counterfactual result hashes, and requires every artifact's full
`source_commit` to equal the Step 1 `evidence_source_commit`.

- [ ] **Step 8: Record the completed Phase 1 gate and commit artifacts**

Update the Phase 1 row in `docs/PLAN.md` from its planned gate to the exact
artifact hashes and the Step 1 `evidence_source_commit` recorded by all five
artifacts. This later evidence/docs commit is not itself the collector source
revision. Do not add benchmark claims. Do not make any post-evidence source,
fixture, schema, or verifier correction in this commit; an audit finding
invalidates all five artifacts and starts a new correction/regeneration cycle.

Use exactly this completion-cell grammar so the already-committed two-state
repository regression validates the byte hashes and common source commit:

```text
Complete at collector source `<evidence_source_commit>`: validation `<file sha256>`; oracle `<file sha256>`; leakage `<file sha256>`; matched reproducibility `<file sha256>`; independent reproducibility `<file sha256>`. The v3 cross-artifact verifier and local `make verify` passed with zero foundation-model calls; this is generator/oracle engineering evidence, not learned-model or benchmark evidence.
```

```bash
make verify
git diff --check
git add manifests/validation/v1/ofd-primary-10000.json \
  manifests/validation/v1/phase1-oracle-gate.json \
  manifests/validation/v1/phase1-leakage-gate.json \
  manifests/validation/v1/phase1-validation-reproducibility.json \
  manifests/validation/v1/phase1-independent-reproducibility-gate.json \
  docs/PLAN.md
git commit -m "test: freeze Phase 1 validation evidence"
```

- [ ] **Step 9: Verify the committed Phase 1 state**

```bash
make verify
git status --short
phase1_plan_base_revision="$(git log -1 --format=%H -- \
  docs/superpowers/plans/2026-08-30-phase-1-generator-oracle.md)"
git rev-list --count "${phase1_plan_base_revision}..HEAD"
git log --reverse --oneline "${phase1_plan_base_revision}..HEAD"
```

Expected: the complete local gate passes, status is clean, and history contains
the reviewed Phase 1 commits. The resolved value must match
`plan_base_revision` in every evidence report; report the actual commit count
rather than assuming one. The five evidence artifacts must still name the full
Step 1 `evidence_source_commit` (the reviewed Pass 5 final-analysis source
revision), not the later Task 18 evidence commit or whichever commit is current
during replay. Run fresh independent science/specification and code-quality
audits over the committed five-artifact boundary. Any finding invalidates all
five artifacts; no post-evidence patch may preserve or bless them.

## Phase 1 Completion Boundary

Phase 1 is complete only when all of the following are simultaneously true:

1. strict public projection contains no private scorer/generator/oracle truth;
2. all primary and declared stress data generators pass their independent
   invariants;
3. the hand-authored oracle fixtures and generated corpus agree exactly; every
   accepted matched member and independent episode passes all retry-causing
   semantic, shape, leakage-prevention, and exact timing checks for its complete
   positive/safe/disconnected counterfactual family before label selection; and
   all 24,000 OOD-short gate traces build and score through their authenticated
   jitter;
4. the fixed 10,000 validation manifest is immutable and fully regenerable;
5. exactly 100,000 unique accepted episodes from the episode-local independent
   recipe pass independent invariants;
6. oracle timed success is exactly `100000/100000`;
7. positive and negative random strata pass their separate frozen exact-binomial
   and absolute-error checks; pooled expectation `0.3125` is diagnostic only;
8. all clean leakage probes pass and every injected leak is detected; all 135
   probe workloads are checked against their task-filtered memberships, the
   positive-hazard-class workload uses only positive members, all eleven digest
   packs are exact/canonical/bounded, and the source plus clock-child packs
   independently reconstruct the authenticated v2 clock-pair manifest;
9. namespace evidence derives exactly 50,000 matched plus 700,000 independent
   construction tokens and 10,000 matched plus 107,000 independent public IDs,
   binds exact raw public-ID seeds `2026083002` and `2026083012` to their
   fingerprints, records 2,500 matched cohort draws and 100,000 independent
   episode draws, derives attempts as accepted plus rejected draws, and all
   750,000 tokens and 117,000 IDs have zero collisions;
10. `make verify` passes locally with zero foundation-model calls;
11. matched-validation and independent-allocation v2 reproducibility reports
    both pass, including all 100,000 independent source entries, canonical
    accepted-attempt runs, independently recomputed sample memberships, and
    namespace evidence coherent with the oracle/leakage reports;
12. no final frozen-test manifest exists and Phase 6 needs no new episode
    generator implementation; and
13. oracle and both reproducibility reports remain strict v2, leakage is strict
    `leakage-report-v3`, and the final verifier is strict
    `phase1-gate-verification-v3`; every report and verifier derives its own
    pass state and rejects every inconsistency derivable from raw artifacts,
    exact frozen seeds/configuration/allocation, authenticated Git history, or
    redundant cross-artifact evidence, without claiming detection after
    consistent replacement of all raw trust anchors;
14. every evidence source commit, historical plan base, generator hash, and
    final-analysis hash is authenticated from regular Git blobs without a
    checkout, and every artifact uses `phase1_analysis`; and
15. reports call this generator/oracle engineering evidence, not model or
    benchmark evidence.

Do not begin Phase 2 from this plan. After this gate, write and explicitly
approve a separate Phase 2 flow-and-event-engine plan against the canonical
specification.

## Explicitly Deferred Scope

- **Phase 2:** continuous flow/guards, runtime state, event queue, ties,
  refractory/event caps, jumps, arbitrary-time query, terminal dispatch,
  scripted agent, checkpoints, replay, and full spy-agent callbacks.
- **Phase 3:** tensor memory/eviction, encoders, retrieval, neural controllers,
  losses, teacher-forced batches consuming the oracle traces, and training
  checkpoints.
- **Phase 4:** autonomous EventFlow, pilot training, and autonomous dynamics
  failure gates.
- **Phase 5:** learned one-shot/ponder/fixed/periodic/random-time baselines,
  compute matching, and interventions. Phase 1 `RANDOM` remains diagnostic.
- **Phase 6:** instantiate Task 9's already-gated independent recipe into final
  `manifests/frozen/`, then freeze tag, model seeds, evaluation, statistics,
  aggregation, reports, scientific claims, and release artifacts. It may add
  allocation manifests, not a new semantic generator.
- **Phase 7:** Qwen, natural-language parsing/narration, and showcase assets.

## Final Self-Review Checklist for the Implementer

- [ ] Every changed behavior was introduced by a failing test.
- [ ] Generator and oracle reachability code are independent and import-guarded.
- [ ] Public output was sentinel-scanned for every private field/value.
- [ ] Record IDs are assigned only after presentation permutation.
- [ ] Cohort rejection cannot alter variant matching.
- [ ] Independent rejection cannot alter any other allocation-quartet member.
- [ ] Independent quartet member identity is explicit through request,
  coordinate, artifact, manifest, service, invariant, and reproducibility
  paths; no absolute-index/rank reconstruction remains.
- [ ] Nonaligned independent quartet indices such as `5,6,7,8` validate from
  explicit members `0,1,2,3`; no modulo/divmod inference exists.
- [ ] Every independent manifest quartet contains exactly four consecutive
  entries, each explicit member exactly once, with common suite/path/block
  identity and independently permitted accepted attempts.
- [ ] Phase 1 leakage splits/permutations keep whole matched cohorts or
  independent label-allocation quartets together as appropriate.
- [ ] Shared LINK topology is sampled once per cohort and only relabeled.
- [ ] Every gate episode uses an independent coordinate and episode-local
  nuisance streams; no allocation quartet shares a template or seed token.
- [ ] Phase 6 frozen episodes can be created only through Task 9's tested
  independent primitive, never the shared-template validation path.
- [ ] Both hazard delays are identical in every primary episode.
- [ ] The random baseline accepts only `PublicEpisode`.
- [ ] The oracle accepts only public facts for graph solving.
- [ ] Generator acceptance is label-blind across all three primary variants,
  for every matched member and independent nuisance draw; every retry-causing
  semantic/shape/leakage/timing check precedes label selection, and oracle/
  generator/invariants independently derive inputs to the same RNG-free pure
  trace-timing math.
- [ ] Manifest access class, not path, controls private inspection.
- [ ] Existing identical manifests are verified without rewriting.
- [ ] Divergent existing manifests cannot be forced or overwritten.
- [ ] The 100,000 independent gate streams and reports its actual verified
  denominator, and all entries pass the independent reproducibility gate.
- [ ] Both v2 reproducibility reports contain canonical accepted-attempt runs
  and actual ordered token/ID namespace evidence; the final verifier rederives
  all 750,000 tokens and 117,000 IDs from exact raw public-ID seeds
  `2026083002`/`2026083012`, coordinates, and accepted attempts, and rederives
  both sample memberships.
- [ ] `PositiveControlResult.probes`, `LeakageReport.construction_check_ids`,
  `LeakageReport.label_shuffled_probes`, all nine positive controls, and both
  complete ordered 27-probe families survive the v3 migration; all 135 probes
  publish strict confusion and permutation sufficient statistics from which
  metrics, Holm values, detector summaries, and pass states are independently
  rederived.
- [ ] The canonical leakage report is at most 16 MiB and carries one ordered
  100,000-source-digest pack, one ordered 7,000-clock-child-digest pack, and
  nine ordered 8,000-injected-digest packs; source/corpus/source-manifest,
  complete v2 clock-pair manifest, main task-filtered split/train/test, every
  positive-control subset/split/injected hash and task-filtered workload, all
  six construction checks, and all three counterfactual result hashes are
  reconstructed independently.
- [ ] Oracle/random arithmetic and pass flags are derived under strict exact
  schemas and independently recomputed at the five-artifact boundary.
- [ ] Evidence revisions, historical plan base, and both source fingerprints
  are recomputed from regular Git blobs at the authenticated source commit.
- [ ] The invalid artifacts tied to collector `bee142b`, evidence `4aa6eca`,
  and post-evidence fix `2203b68` remain deleted; the reviewed Pass 5
  final-analysis revision is the exact `source_commit` for every regenerated
  Task 18 artifact.
- [ ] No post-evidence source, fixture, schema, or verifier commit exists; an
  audit finding after regeneration restarted the correction cycle instead of
  preserving the evidence.
- [ ] Random positive and negative strata pass separately; no pooled binomial
  law is used.
- [ ] No final frozen tests, neural/runtime code, hosted automation, or Qwen code
  entered the phase.
- [ ] The final worktree is clean and every committed artifact hash verifies.
