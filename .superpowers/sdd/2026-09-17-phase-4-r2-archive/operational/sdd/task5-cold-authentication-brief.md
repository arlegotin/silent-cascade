# Task 5: completed-run cold authentication

Implement the approved plan's subsection "Completed-run authentication and
workflow reuse" in docs/superpowers/plans/2026-09-17-phase-4-r2-archive.md.
Read that subsection, Task5 interfaces/global constraints, and the R2 spec.
Current milestone: durable primitive9f5047c, reviewed with four passing checks;
milestoneb80e309. Work on main; no worktree/branch/push/CI/CD. Use Superpowers
TDD/debugging/verification. Sole source/test writer. No worker subagents.

Own only train/pilot_evidence.py, train/pilot_workflow.py, the focused new
tests/pilot/test_pilot_cold_authentication.py and a focused test helper if needed.
Controller owns plans/progress. Do not touch frozen historical .superpowers
files outside operational/. Never delete/relabel old evidence or create a new
storage ledger. No provider calls, full gate, MPS training, production fit or
recovery authority in this slice.

First construct tests/helpers, then STOP BEFORE EXECUTION and report complete
write-path guard coverage, fixed request/root/count/byte bounds and exact RED
command. Main must admit the command in the existing ledger before any pytest,
source checkout/data generation/model instantiation. No production edits before
expected RED. An initial signature mismatch alone is not meaningful RED: the
genuine local fit/authentication must succeed, then the cold call must expose
the missing integration. Optional signature compatibility in test setup may
exercise old authenticate_run without context; do not bypass real decoders.

Required production behavior is in plan. Preserve all authenticity/semantic
checks. authenticate_run consumes context-scoped checkpoint metadata, checkpoint,
weights, result/index and durable journal inputs. Return logical checkpoint path.
_train discovers root/attempt result via validated inventory, unions resident
candidates, validates context/root, forwards every consumer. Completed reuse
must not train. Workflow report gets context. No broader caller claim.

## Real fixture, not a fabricated successful run

Use exact currently copied source and real Git producer/data/training revisions
as tests/pilot/test_pilot_source.py and tests/neural/test_phase3_provenance.py.
Execute genuine phase4_smoke CPU4updates and validations at2/4 (16episodes each).
No source identity mock or rewritten completion status; retain real progress,
model tensors and all adverse outcomes. One fit per RED/GREEN fixture, not two
reference fits. All output including failed prefixes retained. Main fresh check
can reread the immutable GREEN fixture with that fixture's own source; don't
force a third fit just to retest unchanged code.

After eager authentication/reuse, move only NEW fixture inputs into backing;
strict context renames requested file into lease root and back in finally.
One active lease max; real journals may stay resident (cold journal protocol
is separately covered). Need actual cold checkpoint/index/portable weights,
training envelope/index shards and inventory evidence. Compare real eager/cold
results, logical path, source, progress, model identity/tensors and RNG where
contract preserves it. Completed and attempt-result reuse use neural execution
tripwire. Cover wrong context root, final-index corruption, bad checkpoint and
portable bytes/hash, restoration of originals and lease cleanup. Use small
altered controls, not copies of full weights/checkpoints or changed expected
hashes hiding corruption. No real provider/IPC completion claim.

## Guarded fixture admission design

No command admitted yet. Previous owner51477 released successfully. Existing
global ledger workspace is .superpowers/sdd/2026-09-17-phase-4-r2-archive/operational.
Current scratch222056448/cap268435456B, spool134307840B, cache67153920B,
metadata52965376B, retained6195077120B. New outputs assigned from inception:
scientific run/backing in existing spool; cold single-file lease in existing
cache; checkout/Git/manifests/audits/temp/pytest in scratch; bounded logs in logs.
No nested producer_case ledger. Root strings supplied by controller, no outside
temp/cache/bytecode. Keep categories exact through rename and final restoration.

Read-only training audit established fixed maxima:4updates+4updatejournals,
2validations+2validationjournals,32episodes,2componentreports,2validationrecords,
3checkpoints at0/2/4 (6payloadwrites incl pending->final copy),3indexreplacements,
3exportedweights+2runtimeweights,32each sidecars/trajectories/telemetry,up to32
crashpairs+2crashindexes,1artifactindexshard+1attemptresult. Smoke cannot promote
past one_hop because content gate requires production. <=195originalnames;
allow256guarded names including temporaries/rootresult; base10directories.
Model771090float32params/71tensors+18boolbytes: weight body3084378B,
AdamW6169004B,trainedcheckpoint9253382B before RNG/header. CPU may capture MPS
RNG. These are static derivations, not runtime success claims.

Natural production maxima DO NOT prove64MiBfit: individual trace/crash formats
allow128MiB and failure messages64MiB. A TEST-OWNED pre-write aggregate guard
may enforce <=64MiB live new fit allocation; it must fail the test BEFORE any
violating write and preserve the prefix. Do not reduce production limits or
present a budget-aborted fit as completed. Use actual filesystem block rounding,
charged current st_blocks, next complete temp/coexistence and directory/name
allowance. Pendingcheckpoint and old index remain charged during copy/replace.
Avoid cumulative-all-bytes-ever counter (normal checkpoint retention would make
it unnecessarily loose); never manually prune to fit. Preserve normal retention.

Exhaustive write audit must cover imported _publish_pilot_bytes aliases in
trainer/checkpoints/workflow/pilot_evidence_types; trainer _publish_bytes_at;
atomic publishers used by evaluation/neuralweights/trajectories/crashes; and
direct .rows.pending.jsonl Path.open('xb') handle.write appends. Atomic temp/final
hardlinks may transiently coexist; account conservatively and do not call parent
ledger during temporary hardlink state. Count directories/temporary names and
bounded outer diagnostics before writes too. Real publishers/decoders stay real.
New small damaged control copies need their own bounded count/payload allowance.
Read-only checkout agent will provide separate finite scratch derivation; await
controller integration of that bound before executing. No general quota framework.

Report exact commands, RED failure, GREEN results, source/fixture identities,
all retained allocation, guard coverage/limits and narrow claims. Commit scoped
implementation/report after verification; independent review follows.
