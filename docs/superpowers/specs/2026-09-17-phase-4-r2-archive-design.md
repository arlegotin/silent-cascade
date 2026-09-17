# Phase 4 bounded R2 archive design

**Status:** Proposed storage-only amendment; implementation requires approval of
the companion phase-scoped implementation plan. The user's request to proceed
authorizes preparing this design, not a claim that archive support already works.

**Scope:** Unblock Task 12 of the approved Phase 4 plan without changing the
experiment, evidence retention, learning recipe, device checks or acceptance gates.

**Parent specification:** [Silent Cascade v1.0.2](2026-08-30-silent-cascade-design.md).
**Parent plan:** [Phase 4](../plans/2026-09-16-phase-4-autonomous-eventflow.md).
**Inspected checkout:** `c09faee` on `main`, 2026-09-17.

## 1. Problem and evidence

R2 provisioning is complete; integration is not. The private Standard bucket,
bucket-only local credentials and byte-exact round trip are documented in
[R2 setup](../../r2-archive-setup.md). No experiment data has been uploaded.

The historical Task 10 forecast requires 4,411,034,493,825 local bytes. Preserve
that failed preflight and its original source. The forecast is a conservative
scenario, not 4.41 TB already produced or an immutable lower bound on working disk.
Its upper retained-data estimate is about 3.68 TB, including approximately
37.61 GB of update journals. Archiving only episode trajectories is insufficient.

The inspected machine has about 15 GiB free on the repository volume and 109 GiB
on the system volume. These are observations, not reserved capacity. Recheck them
before execution. Use an explicitly configured, durable system-volume workspace;
do not move the repository, use a symlink/mount, or use system temporary storage
as the only retained copy of scientific evidence.

Relevant existing boundaries:

- `eval/artifacts.py:write_evaluation` seals an entire evaluation with `DONE`
  after complete row/evidence validation. Per-episode row appends are not an
  independent durable commit protocol. Preserve these scientific file formats.
- `train/pilot_trainer.py:_validation` rereads rows and publishes a validation
  record. Robustness and primary validations run sequentially before the next
  durable training checkpoint. Permit handoff between those two evaluations.
- `verify_journal`, `_collect_artifact_hashes`, `load_training_result`, report
  discovery and the final gate verifier currently require all raw paths locally.
  A background upload script alone cannot fix this requirement safely.
- Existing recovery materializes all training inputs into a fresh destination.
  Add an explicit bounded archive-backed recovery path; keep the existing local
  path and preserve the requirement to authenticate every indexed input.

## 2. Alternatives and decision

1. **Chosen: immutable evaluation units and journal segments with local leases.**
   Keep existing scientific bytes. Archive sealed units in small transport chunks;
   restore one large evaluation and one small journal segment at a time. This
   needs explicit producer, discovery and aggregate-reader integration, but leaves
   event dynamics and per-evaluation semantic validators intact.
2. Per-episode cold storage can use less disk, but requires new episode commit,
   partial-evaluation recovery and per-file fetch semantics. It is a larger change
   to scientific publication; defer it unless measured unit sizes invalidate the
   chosen working-space bound.
3. Mounting R2 as a filesystem or running a blind directory sync does not preserve
   the repository's local atomic-write, no-follow, durability and replay contracts.
   Do not implement it.

No database, service, daemon, plugin backend registry, hosted execution or new ML
dependency is introduced. A finite local supervisor uses the already installed
AWS CLI for R2. Tests use a local transport double, not real account credentials.

## 3. Process boundary

```text
offline scientific child -> sealed local unit -> finite local archive supervisor
       ^                        |                         |
       | local lease            | hash-verified receipt   | R2 S3 requests
       +---- restored bytes <---+-------------------------+ private bucket
```

The scientific child performs the same train/evaluate/check/report/replay work.
It has no AWS profile, endpoint or secret in its scientific configuration, no
cloud client import, and no network access. Archive waits do not change simulated
time, RNG state, event counts, optimizer-step budgets or selection decisions.

The supervisor is explicitly network-enabled operational tooling. It exchanges
only bounded, atomic local-file requests with the child and holds a separate
single-writer archive lock. Requests bind a run identity, monotonically numbered
request ID, operation and manifest hash; responses must match all four. No
arbitrary shell commands, HTTP endpoints or remotely supplied executable code.
Archive processing has its own I/O/timing counters, separate from neural compute.

Ordinary commands remain offline. Without a supervisor, an evicted input produces
an actionable hydration-required error, never an implicit network call. An
explicit `archive run` wrapper supplies the operational supervisor. It supports
only the closed pilot/checks/verify/report/replay jobs and joins all children on
exit. It is not a persistent system service.

## 4. Bounded working set

Initial operational limits, measured and enforced independently of ML config:

| Allocation | Limit |
| --- | --- |
| Total workspace, including partial/staging/control files | 64 GiB |
| One materialized evaluation/validation unit | 32 GiB |
| One journal segment | 1 GiB or 1,000 records, whichever comes first |
| One transport chunk | 256 MiB |
| Concurrent transport operations | 1 |
| Concurrent large evaluation leases/writers | 1 |
| Filesystem free-space reserve | 16 GiB, plus checkpoint/emergency reservation |
| New remote bytes under this run prefix | 4,500,000,000,000 bytes |

The measured scenario implies approximately 23.60 GB per 10,000-episode unit;
its 1.2 safety factor fits 32 GiB. This must be re-established with actual
serialization and file-allocation measurements. These numbers are operational
limits, never permission to truncate an evaluation, drop a failure, reduce a
denominator or change the experimental event ceiling.

Reserve space before starting a unit and recheck before each episode. Use actual
allocated bytes as well as logical sizes; include temporary chunk/download files,
partial restores, metadata, checkpoints, abandoned attempts and logs. Reserve
enough emergency space for the maximum permitted in-flight episode/crash output
and checkpoint publication, derived from existing serialization limits. If that
reservation cannot be proved to fit, preflight fails before training.

Drain the first robustness validation before starting its paired primary run.
Seal immutable journal segments at record/byte boundaries so 37.61 GB of journals
does not remain pinned locally. Mutable checkpoint index, active journal tail,
latest/best/selected full checkpoints, needed weights, catalog roots and compact
manifest/index shards remain local and are counted. They are also backed up in
immutable, checkpoint-bound control snapshots; active control files are not evicted.

If a budget, transfer or disk check fails: stop issuing new work, retain every
written byte and report `storage_blocked`. Use the existing last-durable-checkpoint
recovery semantics after an interruption, retaining abandoned journals/partial
evaluations. Do not invent validation resume or silently rerun in a new seed.
For a live process, bounded waits between safe boundaries preserve current state;
supervisor failure does not grant permission to delete its pending output.

The remote byte limit is a client-side stop, not a Cloudflare spending cap. R2
continues charging for retained storage until its owner removes it. Do not add
automatic deletion/lifecycle policies or expand the budget automatically.

## 5. Archive format and lifecycle

Use an additive `phase4-r2-archive-v1` envelope outside the logical scientific
tree. Preserve every original relative path, file byte, SHA-256 and source/config/
checkpoint/generator/manifest identity. Reject absolute paths, traversal, duplicate
or overlapping logical member ownership, links, special files, unexpected members
and changed files. Disjoint journal segments may share the logical run root.

Unit kinds are closed: completed validation/evaluation, immutable journal segment,
checkpoint-bound control snapshot, and stopped-owner partial evidence. Partial
evidence can be archived only after the writer is confirmed stopped; it is never
relabeled `DONE`, complete or scientifically passing. Recovered partial evidence
keeps its original path and restart/abandonment bindings.

File inventories use sorted, bounded canonical JSONL shards with counts and stream
hashes. Each file entry specifies path, bytes, SHA-256 and ordered byte spans in
transport chunks. Chunks concatenate original bytes, with no lossy transformation
or second full-size bundle on disk. Pack/read at bounded buffer sizes. Existing
compressed trajectories remain byte-identical. Validate spans, totals, order,
missing tails and integer bounds before allocating/restoring.

Object keys are content-addressed within a run-specific prefix. Upload using
create-only conditional writes. An existing object is reusable only after its
downloaded bytes match the expected size and SHA-256; a collision fails. ETags,
metadata and successful HTTP status are not SHA-256 verification.

Reserve remote-byte budget durably before each new object write, including
manifests, catalogs and ambiguous/failed attempts. A retry of the same verified
object does not consume a second unique-object reservation. Never reset usage on
restart or ignore orphaned/unconfirmed writes to remain under the configured cap.

Lifecycle:

1. The offline owner seals a closed inventory under its lock.
2. The supervisor uploads each chunk and reads it back completely for comparison.
3. It uploads/verifies the unit manifest and immutable catalog-generation commit.
4. It fsyncs a local receipt binding all identities and the verified catalog root.
5. Only then may it evict the exact unchanged files listed in the unit, with no
   active lease. Record eviction intent and completion; recover partial eviction
   idempotently from the remote unit. Unlisted files are never removed.

No remote scientific objects are deleted by normal operation. Upload completion
without readback, a stale receipt, a missing catalog commit, credential expiry or
an interrupted request leaves the local copy intact. A local receipt proves a
past verified transfer, not permanent remote availability or scientific correctness.

Pin catalog commits, selected checkpoint descriptors and reproduction locators in
release evidence. Recovery from loss of local control state must work from an
explicit catalog root hash plus authorized credentials, not an unbounded bucket
listing or a mutable `latest` object.

## 6. Reader, resume and recovery semantics

Introduce a local-only `EvidenceContext` for logical discovery and unit leases.
The context inventories local and archived paths, rejects mismatching duplicates,
and yields real local paths after full chunk/file authentication. Keep existing
per-evaluation validators and replay routines operating on those local paths.

Aggregate journal, training-result, gate and report readers accept this context.
They must visit every expected unit, including failed and abandoned units. Keep
bounded index iteration; do not duplicate the multi-million-entry training index
in a second serialized mapping. Existing materialized public result dictionaries
may remain for compatibility, with their memory cost reported honestly.

`verify_journal` walks all predecessor links, checks every update and validation
binding, and validates referenced artifacts through successive leases. Preserve
the exact `[global_step, ..., 1]` update coverage and rejection of extra/cyclic/
corrupt records. Do not hydrate all historic evaluations at once.

Gate verification still recomputes raw hashes, row-derived metrics, trajectory
semantics, continuation and numeric/offline evidence. It remains artifact-only:
do not add neural execution to the verifier. Actual CPU replay is the existing
separate execution obligation, performed with exact restored inputs/weights.
Missing remote bytes remain missing evidence, with the same unavailable-semantic-
check reporting; cached archive receipts cannot make the gate pass.

For fresh-destination recovery, the archive-backed path authenticates every
indexed training input before any new neural work. Copy pinned control inputs and
bind verified immutable remote units to the destination's separate catalog; read
each unit fully in bounded succession. This changes *simultaneous materialization*,
not required input coverage. Keep old local recovery unchanged, preserve original
artifacts, and reject overlapping destinations and changed gate identities.

## 7. Provenance, compatibility and security

- Archive policy, source, manifests, transfer receipts and control snapshots have
  separate hashes. They do not masquerade as unchanged historical scientific data.
- Include the approved amendment and new executable files in both pilot source
  authentication and the independent gate verifier's exact inventories.
- Keep old fully local artifacts readable. Never rewrite historical gate files,
  source closures, failed diagnostics or the old storage forecast.
- Keep credentials/endpoints and machine-specific paths outside Git and archives.
  Use only `silent-cascade-r2`, the project bucket and the selected run prefix.
- Do not fetch credentials in ordinary imports or in the scientific child. Strip
  inherited cloud credential variables from child environments and test attempted
  socket/DNS/cloud-client/external-AWS use. The parent alone invokes AWS CLI.
- No new dependencies or lockfile changes are planned. AWS CLI is an explicit
  local administration prerequisite, not a core Python dependency.

## 8. Acceptance and execution boundary

Required before changing production preflight: local TDD/failure-injection tests,
exact all-local versus archive-backed semantic equivalence, real bounded R2
round-trip/conflict/recovery checks, a fresh local `make verify` receipt, measured
working-space and transfer behavior, reviewed source and complete raw evidence.

Test interruption before upload, during upload/readback, before/after catalog
commit, during eviction/restore and during a producer wait. Test corrupt/missing
remote chunks, swapped manifests, symlink attacks, budget exhaustion and expired
credentials. No such failure may delete the sole verified copy or yield a pass.

The new preflight reports separately: unchanged total-retention forecast, remote
capacity/budget, maximum local working set, minimum free headroom and transfer time.
It does not simply replace the old 4.41 TB number with 64 GiB. If measurements do
not support this design's limits, report that blocker before production.

After implementation and verification, resume the existing Task 12 sequence:
introduced manifests/audits, one seed-11 run, all final checks, honest result and
whole-phase review. No Phase 5 work or positive research claim is authorized here.

## 9. Technical references checked for this design

R2 documents conditional `PutObject`, `GetObject` and its S3 compatibility limits;
the design still requires real conflict/readback tests on this account before use.
[Cloudflare S3 compatibility](https://developers.cloudflare.com/r2/api/s3/api/).
The 256 MiB chunk bound is well below the documented single-part upload limit.
[Cloudflare limits](https://developers.cloudflare.com/r2/platform/limits/).
Storage and operation charges remain usage-based.
[Cloudflare pricing](https://developers.cloudflare.com/r2/pricing/).

**Self-review:** Scientific behavior and old evidence are unchanged; journals,
partial evidence, control snapshots, reader completeness and bounded recovery
are included. The 64 GiB design is conditional on measured preflight, not a claim
of an implemented or tested solution.
