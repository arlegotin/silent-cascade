# Approved Deviations

This append-only log records necessary, explicitly approved departures from the canonical Silent Cascade design.

## Current status

## 2026-09-15 — Phase 3 evidence integration boundaries

Task 12 inspection found that native MPS checkpoint continuation was covered by
an actual regression test but had no primitive measurement API for the report
collector. Under standing approval, expose that existing two-update check as
`measure_mps_resume` and retain its unchanged loss/parameter/AdamW tolerances
(`rtol=1e-4, atol=1e-5`) and exact RNG/batch checks. CPU continuation remains
exact and keeps its existing API/schema. This prevents a copied passing-test
message from substituting for source-bound native measurements. Cost if wrong:
one additional verification adapter and regression surface; no model, optimizer,
data, scientific gate or numerical tolerance changes.

The closed report schemas are assigned a small data-only module,
`train/evidence_types.py`, shared by collector and independent verifier. Metric
arithmetic, pass decisions and hash-chain implementations remain independent.
Cost if wrong: one additional module to maintain and authenticate; the separation
keeps producer execution out of the verifier's schema imports. Both corrections
are recorded before production corpus creation or training.

## 2026-09-15 — Pre-freeze Phase 3 AdamW stabilization

Standing approval for reversible configuration and plan corrections is applied
to a Phase 3 implementation-plan choice, not a relaxation of the canonical
scientific protocol. Specification Section 8.6 fixes AdamW, learning rate,
weight decay, float32, clipping and budgets, but does not specify epsilon. The
canonical specification and accepted Phase 1/2 evidence remain unchanged.

The original `epsilon=1e-8` failed the fixed native CPU/MPS first-update gate on
ten weights (maximum absolute difference about `5.48e-5`), despite passing the
forward, loss, raw-gradient and discrete-choice comparisons. At controller
weight `[371,295]`, raw gradients were approximately `-4.265e-7` and `-2.026e-7`
after cancellation of opposing contributions near `1e-3`. A global norm near
`35.0978` placed the clipped gradients near epsilon. Independent analysis of
the actual fresh-step AdamW equation predicted a difference of `5.4776e-5`.
The Metal-preference diagnostic did not fix the mismatch. No production
component corpus, final training run or frozen test was used for this decision.

Ruling: declare `epsilon=1e-6` globally in Phase 3 training and smoke profiles,
on both CPU and MPS, including parity, checkpoint and resume paths. This lowers
the worst-case first-step sensitivity bound `learning_rate / epsilon` by 100x.
It is a real optimizer hyperparameter change, not algebraically equivalent code
or a device-specific test workaround. Cost if wrong: genuine small-gradient
updates are damped and convergence may worsen; the unchanged tiny-overfit,
component-accuracy and training-budget gates must detect that. Later comparable
learned conditions must share this declared setting.

The value is fixed before retesting, with no hidden passing-value sweep. Keep
the same numerical tolerances, seeded parity fixture, model, losses, learning
rate, decay, clipping, seeds, budgets and scientific thresholds. New canonical
configuration hashes must bind all new runs; old optimizer archives cannot be
resumed under the revision. Preserve the original failure and regenerate all
affected debug/acceptance evidence. This entry records a correction to verify,
not a claim that the revised optimizer already passes.

Initial verification outcome: with the original seed/batch and comparisons,
`1e-6` has zero CPU/MPS mismatches and a maximum updated-weight difference of
`2.8312206268310547e-6`; exact CPU resume also passes. However, the unchanged
fixed-64 debug overfit reaches only 63/64 complete chains by step 1000, with
111/112 required recalls and compositions correct. Thus this candidate does
not meet the complete engineering gate. These failed learning results remain
part of the development record; no production acceptance is claimed.

Bounded follow-up ruling: evaluate exactly one intermediate global candidate,
`epsilon=1e-7`, before any production freeze. The retained `1e-6` model's sole
failed example selected an unreachable hazard instead of the correct SAFE on
its second recall. Public memory embeddings, masks, active slot, mode and
hypothesis fields matched the teacher path exactly before that choice; the
different flowed fast state explained the changed ranking. Diagnostic-only
state substitution restored the correct ranking, but no teacher state or
timing is added to the actual unassisted evaluator. The observed issue is a
learning/exposure tradeoff, not a discovered pointer or memory-mask defect.

`1e-7` has a worst-case fresh-step sensitivity bound of 3000, still 10x below
the original, while damping genuine small-gradient updates less than `1e-6`.
Cost if wrong: it may still fail numerical parity or the unchanged learning
gate. This is the final candidate in this bounded adjustment, declared before
testing; failure requires reassessment, not an automatic sweep, altered fixture
or higher step budget. Apply it consistently to real training, both devices
and profiles, parity, checkpoints/resume and later comparable conditions.
Regenerate hashes/evidence and reject both older epsilon configurations. The
previous failed attempts remain recorded, and all other settings, thresholds
and canonical specification bytes remain unchanged.

Final-candidate engineering outcome at source
`f6a52a6b148577e80b1020ad70a2eaf75b0acae0`: `1e-7` passes the unchanged native
numerical comparisons and exact CPU resume. The fixed-64 content regression
passes at step 900 with 64/64 chains and 112/112 recalls/compositions. Its
secondary untimed action accuracy remains 32/64; this is not evidence of timed
OFD competence. Full local `OMP_NUM_THREADS=1 UV_OFFLINE=1 make verify` exits
zero with 2472, 93 and 216 passing tests in its three processes, followed by a
passing doctor and both package builds. Independent task review approves the
implementation. Production 10,000-example component acceptance remains separate
and unexecuted at this point.

Final configuration hashes: smoke
`4a2d2946047216eb5cb209a4e88ed7eff2be962f0496c612d59d906f8bebbba7`;
one-hop production
`26b21c1acd79e6c47375f03c4e68da24fadddde9ae150454bbd3db7f50237041`.

## 2026-09-08 — Final Phase 1 leakage-evidence closure

The terminal science and code audits rejected the five artifacts collected at
source `bee142bd08a8b7b5621ae65551280ebdeed6c1b6`, committed by
`4aa6eca5d25e3c6dac879850b2a0557bbe84b54e`, and followed by fixture-only
commit `2203b68a4c6b67f26b9aae1bdc7d1ab00f8a6e6e`. Three independently
reproduced pre-completion gaps remain:

1. the leakage source discarded the explicit independent
   `quartet_member_index`, authenticated only quartet/absolute episode
   position, and inferred quartet shape from `position % 4`; it therefore
   rejects the valid allocation-neutral quartet at absolute indices
   `5,6,7,8` with explicit members `0,1,2,3` while failing to authenticate the
   member-to-label relation;
2. the leakage report stores raw accuracy, balanced accuracy, permutation
   p-values, Holm values, and pass flags without the confusion counts or
   permutation exceedance counts needed to derive them, and the final verifier
   checks the three split/train/test membership fields only as SHA-256-shaped
   strings; coordinated or even isolated metric and membership substitutions
   can therefore pass; and
3. the independent manifest validator groups quartet members into a set but
   does not require exactly four rows, consecutive local episode indices, or a
   common requested path. Eight distinct episode indices carrying members
   `0,1,2,3,0,1,2,3` can validate as one quartet.

The listed artifacts are numerically credible outputs of their named run, but
they are not sufficient Phase 1 acceptance evidence. They are deleted before
any new analysis-source edit, the delivery index returns to in progress, and
all five artifacts must be regenerated from one later clean reviewed collector
revision. No artifact or computation from the prior leakage run may be reused.

Correction Pass 5 makes the independent member a mandatory exact field through
`AuditExample`, stored leakage rows, service adapters, source manifests, clock
parents, and validation. Member-to-variant identity is rederived from the
allocation-label key. Quartet validation uses the explicit member and a local
sorted four-row relation; it never uses modulo, division, absolute alignment,
or variant rank. Independent manifests require exactly four rows, members
`0..3` once each, consecutive local episode indices paired with those members,
and one requested path per quartet. Per-member accepted attempts remain
independent and may differ.

The leakage source-manifest and clock-manifest hash domains, internal source
authentication, and leakage audit anchor move to v2 because their digest
payloads now include explicit member identity. The public leakage report moves
to `leakage-report-v3` while preserving every v2 field. Every one of the 27
clean probes, 27 label-shuffled probes, and 81 positive-control probes carries
a strict test confusion matrix plus exact permutation exceedance and replicate
counts. Raw and balanced accuracy, add-one permutation p-value, Holm value,
detector identity, and probe/control/report pass state are derived from those
primitives by both the producer and an independently implemented verifier.

Version 3 additionally carries bounded canonical `OrderedSha256Pack` values:
one ordered 100,000-item pack for source episode digests, one ordered 7,000-item
clock-child pack, and one ordered 8,000-item injected-digest pack for each of
the nine positive controls. Each pack is a canonical padded-base64 encoding of
concatenated 32-byte SHA-256 values with an exact count and payload digest; the
whole canonical leakage artifact is capped at 16 MiB. Typed
`LeakageConstructionStatistics` and
`LeakageMembershipEvidence` retain the raw counts and redundant hashes needed
to rederive the six construction checks, independent corpus and source-manifest
hashes, split/train/test membership hashes, and every positive-control subset,
split, and injected-corpus hash. The source and clock-child packs additionally
reconstruct the complete v2 clock-pair manifest and bind its digest to source
authentication and the v2 anchor during production. The final verifier has no
live source-authentication object: it reconstructs the complete ordered
public-ID and coordinate stream from frozen allocation, seeds, and
accepted-attempt runs, independently rebuilds the clock manifest from the
serialized packs and frozen recipe, and compares that digest with the
serialized v2 anchor. It performs those derivations without calling the
producer helpers and also independently reconstructs all three counterfactual
result payload hashes from their canonical pair keys and primitive mismatch
counts.

The transitional report/schema pass remains independently green by teaching
the verifier to consume leakage v3 while its own result is still
`phase1-gate-verification-v2`. The final verifier pass adds the independent
rederivations and alone moves the outer result to
`phase1-gate-verification-v3`. Generator `ofd-v1`, episode/manifest schema 1,
`oracle-evaluation-report-v2`, `phase1-reproducibility-v2`, and all frozen
seeds, denominators, probes, controls, thresholds, optimizer settings,
permutation counts, and resource ceilings remain unchanged.

The packed digests are bounded raw per-episode evidence for report arithmetic,
not the full episode payloads or predictions. As before, a party able to replace
all artifacts, packed values, historical source, and trust anchors coherently
can defeat an offline verifier that intentionally does not regenerate the
100,000 episodes or rerun the statistical fits. The report and final text must
state this trust-anchor limitation rather than claim universal tamper
detection. Primary execution remains local and offline with exactly zero
foundation-model calls.

A pre-implementation re-review tightened this correction without changing a
scientific setting. Leakage v3 now carries an eleventh bounded digest pack for
the 7,000 clock children in exact `0.1x`-then-`10x` authenticated parent order.
The producer must combine it with the 100,000-item source pack to rebuild the
complete v2 clock-pair-manifest payload and compare its hash with both live
source authentication and the serialized v2 leakage anchor before publication.
The standalone verifier independently performs the same reconstruction but,
because it has no live producer object, compares the digest with the serialized
v2 anchor. This closes the prior gap in which child public IDs were
reconstructable but child episode digests were not.

The same review made the Task 18 byte-immutability rule executable rather than
prose-only. Steps 1–7 now retain one exact final-analysis path array and call a
guard that prints a concise error and exits the collector shell on any failed
condition. The guard runs before and after each collector and rejects changed
committed history, staged bytes, and net worktree bytes relative to the captured
clean `collector_head` on any final-analysis path. Evidence commands still run
from that captured head and reports still record the last reviewed
analysis-source revision.

The same review made the Pass 5A anchor-fixture migration explicit, froze the
delivery-index regression before collection to accept only a wholly absent
in-progress state or a wholly present byte-hash/source-bound completed state,
and fully specified the positive-control base-subset membership digest. Task
workload validation is task-filtered: hazard-class probes use only positives,
including within each named 8,000-item control subset. Task 18 collectors run
from the captured clean `collector_head`; their reports continue to record the
last reviewed analysis-source revision, `evidence_source_commit`. These are
evidence-authentication and workflow corrections only. Frozen denominators,
thresholds, seeds, schemas outside the already declared leakage/anchor/gate
migrations, local-only execution, and zero foundation-model calls remain
unchanged.

## 2026-09-06 — Final Phase 1 evidence-authentication audit correction

The final whole-phase scientific and adversarial code audits rejected the five
artifacts generated from source `c8486ae2423beb3f48e62675f90d76de733eecb3`.
Seven pre-completion gaps allowed apparently self-consistent evidence to omit
or reconstruct facts that the acceptance boundary must authenticate directly:

1. independent quartet member identity was reconstructed from absolute block
   position instead of being explicit in the request and coordinate;
2. primary generation checked only a minimum timing proxy rather than every
   assigned/counterfactual trace under the exact jittered schedule, and the
   oracle gate synthesized an action instead of scoring a built trace;
3. collision evidence used coordinate surrogates and counts rather than actual
   accepted-draw construction tokens and emitted public IDs;
4. oracle/random report fields and outer pass state were not all independently
   derived from exact primitive evidence;
5. provenance authenticated current-worktree bytes without rederiving source
   and plan history from the named Git commit;
6. reproducibility sample-membership digests were trusted rather than
   independently recomputed, and one worker helper retained an unused request
   argument; and
7. report schema literals did not distinguish the expanded acceptance
   contracts from the invalid v1 evidence.

The corrective ruling is to execute four ordered, independently reviewed TDD
passes. Independent requests and coordinates carry exact
`quartet_member_index`; generic invariants authenticate its allocation-label
relation while allocation-owning services authenticate block membership. A
neutral pure timing primitive owns the exact difficulty/urgency/jitter/clamp/
common-scale calculation without RNG and uses the provided finite jitter values
through `math.exp` plus stable `math.fsum` accumulation. For every matched and
independent member nuisance draw, acceptance checks the complete positive,
safe-negative, and disconnected-negative counterfactual family under identical
member streams, including reinitialized `TRACE_JITTER`, and runs every
retry-causing semantic, shape, leakage-prevention, and timing check before
selecting the allocation-assigned label. A matched failure rejects one whole
cohort attempt and contributes one rejection reason; an independent failure
rejects only that episode attempt. Post-ID invariant disagreement is fatal and
never a retry. The oracle gate builds/scores actual base and paired-clock
traces. Accepted construction evidence includes `TRACE_JITTER`, yielding
exactly 20 tokens per matched cohort and 7 per independent episode.

Strict canonical attempt runs and actual ordered namespace digests must record
2,500 matched cohort draws / 50,000 tokens / 10,000 base IDs under raw public-ID
seed `2026083002`, and 100,000 independent episode draws / 700,000 tokens /
100,000 base plus 7,000 clock IDs under raw seed `2026083012`. Every namespace
object binds its raw seed to the provenance fingerprint
(`0454fca622eb08379a5d88ecbe0a5ef70f6a15e9ca7d333acecf840f2df38802`
matched and
`f21ac562825bfd96e875eedf90c8ba5ed09883ebe2c18449843b03acebec778a`
independent). Oracle, leakage, and
independent reproducibility must independently emit equal independent summaries;
validation reproducibility owns the matched summary. For either mode,
generation attempts equal accepted draws plus rejected draws. The final
verifier rederives all IDs from raw seeds, coordinates, and accepted attempts,
rederives both summaries, and requires all combined 750,000 tokens and 117,000
IDs to be collision-free. Oracle/random fields and pass state become derived v2
evidence. Historical Git authentication resolves full commits, ancestry, the
plan base as of the source commit, and framed generator/final-analysis hashes
from regular Git blobs without checkout. Pure sample-selection and membership
helpers let the verifier recompute both report memberships; the unused worker
parameter is removed and real matched subprocess coverage stays. This verifies
every derivable and redundant cross-artifact relationship; it does not promise
to detect replacement of all mutually consistent artifacts and trust anchors.

Those raw public-ID seeds are forbidden from agent inputs, public episode data,
public manifests, and public reports. Their only report-level exception is the
environment-private, agent-inaccessible final Phase 1 acceptance evidence used
by the local verifier for exact ID rederivation; that evidence is not public.
The ordered implementation remains independently green: Pass 3 migrates the
leakage and reproducibility reports plus the standalone verifier/fixtures that
consume them to inner v2 while the verifier result stays outer v1. Pass 4 alone
adds final cross-artifact hardening and bumps the outer verifier result to v2.

The exact new outer schemas are `phase1-reproducibility-v2`,
`oracle-evaluation-report-v2`, `leakage-report-v2`, and
`phase1-gate-verification-v2`; v2 leakage preserves
`PositiveControlResult.probes`, `construction_check_ids`,
`label_shuffled_probes`, every existing bounded metric/count/iteration field,
and all existing exact-primitive, workload, probability, all-nine/27-probe,
and derived-pass validators. Stale v1
acceptance artifacts have no compatibility path. Generator version `ofd-v1`
and episode/manifest schema 1 are retained solely as a pre-completion
replacement: Phase 1 has not released, construction/rejection semantics and
private coordinate/schema fields do change, and every earlier artifact is
invalid. The OFD task and public distributions do not change. The reviewed
Pass 4 commit must include deletion of all five stale artifacts so the exact
clean final reviewed Pass 4 tip (including any re-reviewed fix descendants) is
also the Task 18 collector HEAD; all evidence is then regenerated in
full—including the all-nine leakage gate—from that hash. This
ruling changes no frozen seed, denominator, control, threshold, resource
ceiling, or exact zero-foundation-model-call requirement.

## 2026-09-05 — Phase 1 executed-package initializer provenance closure

Adversarial review of the explicit-import repair found that Python executes
`src/silent_cascade/__init__.py` and `src/silent_cascade/env/__init__.py` while
loading generator modules, and also executes
`src/silent_cascade/logging/__init__.py` on the manifest-backed analysis path,
but those initializers remained outside the corresponding fingerprints. The
generator scope must include the root and environment initializers; final Phase
1 and historical Task 14 analysis scopes must additionally include the logging
initializer. Closure tests must resolve absolute and relative imports beneath
both local module roots, derive ancestor initializers, enforce literal exact
tuples for all three scopes, and start initializer/timing mutation regressions
from an asserted clean provenance baseline.

The first plan correction still treated `phase1_task14_analysis` as a
historical dependency snapshot, even though its scoped current
`src/silent_cascade/env/services.py` explicitly imports
`src/silent_cascade/env/leakage.py`. The Task 14 scope must therefore include
the current leakage module as well. Its historical label records introduction
time, not permission to omit a presently executed dependency; closure and
mutation tests must prove that leakage changes affect both analysis
fingerprints while leaving the generator fingerprint unchanged.

All five existing Phase 1 evidence artifacts remain invalid acceptance
evidence. After a new source-and-test correction passes full local verification
and independent review, every artifact must be regenerated from that one clean
reviewed source revision before Phase 1 can be marked complete. This closes an
import-time provenance hole without changing the OFD data, controls,
thresholds, seeds, metrics, acceptance gates, or exact zero-foundation-model-
call requirement.

## 2026-09-05 — Phase 1 explicit-import provenance closure

The final whole-Phase-1 code audit found that the exact generator and analysis
source fingerprints omitted `src/silent_cascade/env/timing.py`, even though it
defines the action-window semantics used by generation, validation, scoring,
leakage, scaling, and services. A complete explicit-import closure audit also
found `src/silent_cascade/errors.py` omitted despite its fail-closed publication
role. Both paths are now required in the generator scope and, through it, the
analysis scope; tests must enforce complete explicit project-local import
closure and prove that a timing-only mutation dirties source and changes both
fingerprints. The CLI remains an excluded adapter because services and the
final verifier independently authenticate every successful evidence input and
output. Executed package initializers are part of the applicable scientific
scope regardless of their current contents; the initializer-specific correction
above supersedes the earlier exclusion rationale.

The five existing Phase 1 artifacts are stale acceptance evidence. After the
source and test repair passes full local verification and review, all five must
be regenerated from one new clean source revision before Phase 1 can be marked
complete. This strengthens provenance without changing the OFD task, controls,
thresholds, seeds, data, metrics, acceptance gates, or zero-foundation-model-call
requirement.

## 2026-09-02 — Task 18 complete leakage-report boundary

Mutation review found that the standalone verifier still accepted
schema-bypassed top-level report values with coercion-compatible types, list
substitutions for ordered evidence families, and reordered or duplicated
counterfactual evidence. It also found that the canonically sorted
construction-check mapping could prove membership but not the engine's declared
execution order. The verifier now checks one strict outer-report type boundary,
requires exact ordered counterfactual result types and identities, and validates
SHA-256 syntax before interpreting evidence. The report now serializes an
explicit ordered construction-check ID tuple derived from the engine's declared
sequence and cross-checks it against the exact sorted boolean-map key set. This
adds authenticated report structure without changing an audit, fit, threshold,
sample size, replicate count, seed, control, or failure-publication rule.

## 2026-09-02 — Task 18 exact in-memory evidence types and schema identity

Adversarial review of the independent verifier found that unchecked in-memory
model copies could exploit Python equality between booleans and numbers or
strings and string enums, even though strict JSON parsing rejected the same
values. Report and verifier acceptance now authenticate exact enum, integer,
float, boolean, tuple, mapping-key, and class-count value types before using
numeric or equality checks. The same review found that arbitrary all-true
construction-check names and a forged feature-schema hash were accepted. The
Phase 1 report and independent verifier now require the exact six construction
check IDs and independently derive the feature hash from the frozen schema,
dimensions, and ordered feature groups. This hardens the existing evidence
contract without changing the audit algorithm or any scientific threshold.

## 2026-09-02 — Task 18 complete leakage-evidence authentication

Adversarial review found four remaining fail-open seams in the Phase 1 leakage
acceptance boundary: only the nested six-field profile was checked before
source access; nested positive-control convergence was not authenticated;
probe workload, dimension, optimizer-iteration, and permutation-grid evidence
was not pinned; and the complete label-shuffled probe family was discarded
behind a boolean. The Phase 1 engine now checks every consumed audit setting,
including the frozen seeds, thresholds, optimizer controls, batch sizes, and
800 MB/512 MB ceilings, before source access. Probe schemas and contextual
validators bind exact task dimensions, 80/20 workloads, complete class maps,
the 500-iteration ceiling, convergence, and the 1/5000 permutation grid. The
report now serializes the ordered 27-probe label-shuffled family, and both the
report and independent verifier recompute its Holm values and pass state. This
adds required publication evidence and fail-closed validation without changing
any production threshold, scientific sample size, or failure-report behavior.

## 2026-09-02 — Task 18 leakage statistical-evidence consistency

Mutation review found that the artifact verifier trusted serialized balanced
chance and Holm-adjusted values, and that probability/accuracy fields admitted
values outside `[0,1]`. Leakage probe schemas now bound every accuracy, chance,
and p-value; balanced chance is derived exactly from its binary, three-way, or
four-way task. Both the report and the independently implemented artifact
verifier recompute Holm adjustment and pass state across the complete ordered
27-probe clean family. Each positive-control summary now includes its complete
ordered nine-feature probe family, allowing report and verifier to recompute
Holm values, observed detector IDs, summary metrics, and pass state instead of
trusting stored booleans. TEST controls retain their approved raw-p detection
rule; the Phase 1 profile additionally requires familywise adjusted p below
0.01. This adds authenticated evidence fields and consistency checks without
changing a fit, null, threshold, seed, sample, or replicate count.

## 2026-09-02 — Task 18 exact leakage acceptance-profile binding

Review of the all-nine scheduler correction found that coordinated `--set`
overrides could still replace the Phase 1 leakage profile with smaller values
whose p-value resolution happened to meet the alpha threshold. The production
profile now has specialized literal fields for exactly 100,000 clean episodes,
4,999 clean permutations, 8,000 positive-control episodes, 4,999 control
permutations, a 200-example per-class minimum, and enabled clean statistical
enforcement. The audit engine redundantly compares that complete profile before
source authentication, so even an unchecked model copy fails before generation.
The report and artifact verifier also require the exact ordered three-task by
nine-feature clean-probe family; vacuous or partial clean evidence cannot pass.
These checks freeze already approved values and add no new scientific parameter.

## 2026-09-02 — Task 18 all-nine positive-control publication correction

Independent review found that the first Task 18 leakage artifact contained an
empty positive-control family. The exact production command supplied a clean
source, while the audit engine scheduled a positive control only for an
internal pre-injected source; the verifier and derived report pass state also
accepted that absence. Fixture-scale coverage cannot replace the binding Phase
1 acceptance run. The `phase1-gate` audit now schedules all nine frozen named
injectors in declared order against the shared 8,000-episode subset with 4,999
permutations, requires the exact complete passing family in its derived report
state, and the five-artifact verifier checks its identities, power,
significance, and shared subset/split witnesses. TEST-profile single-control
reports retain their existing non-publishable semantics. No injector,
threshold, sample size, replicate count, seed, or resource ceiling changes.

The earlier five artifacts are invalid acceptance evidence. After this repair
passes full local verification and independent review, only those five files
are removed and all Task 18 evidence is regenerated from one new clean source
revision; no mixed-revision evidence is retained. Phase 1 remains incomplete
until that regeneration and its terminal review pass.

## 2026-09-02 — Task 18 service resource-test process isolation

The final committed Task 18 local gate exposed a second suite-order resource
baseline: the complete non-leakage pytest child can retain more than 512 MB
before the real leakage service integration test starts. All four service
parametrizations pass with the unchanged production 512 MB guard in a fresh
process. `make test` therefore runs the main suite, the complete Phase 1
services module, and the complete leakage unit module in three pytest children,
with the latter two excluded only from the main child. All 906 tests still run;
no production code, resource ceiling, scientific profile, or assertion changes.
The harness-only correction follows the six-path evidence commit and is outside
the frozen generator and `phase1_analysis` source fingerprints, so the reviewed
`f43499a` artifact provenance and evidence bytes remain unchanged.

## 2026-09-02 — Task 18 validation experiment-version guard correction

The first exact matched reproducibility command exposed a pre-evidence adapter
guard mismatch: the canonical base configuration and exact freeze command
produce `experiment_version=v1`, while the newly added reproducibility and
cross-artifact guards expected the unconfigured value
`ofd-primary-validation-v1`. Integration fixtures had duplicated that stale
literal and masked the production incompatibility. Both guards and their
canonical fixtures now bind the existing configured value `v1`; access class,
allocation, seeds, ordered 10,000-entry recipe, provenance, and every scientific
gate remain unchanged. The failed command published no reproducibility report.
All artifacts generated under the earlier source revision are discarded only
after this correction passes full local verification and independent review,
then every Task 18 evidence artifact is regenerated from one clean source
revision so no mixed-revision evidence is retained.

The correction's full local gate also exposed that unrelated integration-test
fixtures can leave the shared pytest process above the audit's absolute 512 MB
RSS ceiling before the leakage tests start. The complete leakage module passes
with the unchanged production guard in a fresh process, and no individual
suspect integration module reproduces the breach; the full inherited
integration baseline does. `make test` therefore runs all non-leakage tests and
then the complete leakage module in a second pytest process. No test is skipped,
and the production code, 512 MB ceiling, audit profiles, and assertions are
unchanged; process isolation gives the resource test a defined baseline.

## 2026-09-01 — Task 18 reproducibility adapter and command-matrix correction

Before any Phase 1 evidence was generated, the Task 18 reproducibility commands
were found to be unreachable through the production script adapter: the
underlying Task 14 library supported both authenticated manifest and independent
allocation sources, while the adapter exposed only allocation mode and no
request-matrix options. The adapter was restored to the approved two-source
contract with mutually exclusive source selection, conditional allocation
seeds, exact request forwarding, stable typed failures, and one verified
no-clobber publication path. Task 18's stale chunk examples `(1,31,257)` were
corrected to the already reviewed and verifier-enforced production matrix
`(1,3,7)` with hash seeds `(0,1)`. This correction changes no generator,
allocation, sample size, scientific threshold, corpus seed, or zero-model-call
requirement; all evidence is generated only after the repair is committed and
independently reviewed.

## 2026-08-31 — Task 15 pre-freeze leakage regularization correction

The global leakage-probe L2 penalty was corrected from `1.0` to `0.03` before
Task 15 evidence was frozen. The originally specified mean class-balanced
cross-entropy plus `0.5 * 1.0 * ||W||^2` could represent, but would not fit,
the required three contiguous manifest-rank control blocks at the unchanged
`0.95` power gate. An exact 1,200-row synthetic known-answer test, using the
frozen train-only standardization and unpenalized multinomial intercept, gave
balanced accuracies `0.745833`, `0.837500`, `0.912500`, `0.970833`,
`0.995833`, `0.995833`, and `0.995833` for the predeclared penalty sweep
`1.0`, `0.3`, `0.1`, `0.03`, `0.01`, `0.003`, and `0.001`. The largest value
meeting the frozen gate was therefore selected. The `0.03` penalty applies
identically to clean, shuffled, permutation, and positive-control probes; no
control-only fit or threshold change is permitted.

## 2026-08-30 — Plan version 1.0.2

CI/CD was removed by explicit project-owner direction. Silent Cascade now uses
only local verification: `make verify` runs linting, formatting checks, tests,
the environment doctor, and the package build in the current workspace. No
hosted workflow, deployment pipeline, release pipeline, or third-party
automation is configured. This operational policy does not alter the frozen
scientific controls, metrics, acceptance gates, or zero-foundation-call primary
path.

## 2026-08-30 — Plan version 1.0.1

The optional `qwen` dependency range `huggingface-hub>=0.34,<1` was changed to
`huggingface-hub>=1,<2`. The original range was unsatisfiable because
`mlx-vlm>=0.6.17` depends on `mlx-audio>=0.4.3`, which requires
`huggingface-hub>=1`. With the corrected range, the universal dependency graph
resolved successfully and was locked. This packaging correction does not affect
primary scientific execution, which remains offline and records zero foundation-model calls.
