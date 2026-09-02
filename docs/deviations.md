# Approved Deviations

This append-only log records necessary, explicitly approved departures from the canonical Silent Cascade design.

## Current status

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
