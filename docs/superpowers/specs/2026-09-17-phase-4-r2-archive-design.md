# Phase 4 bounded R2 archive design

**Status:** Approved storage-only amendment. The user approved execution of the
companion plan on 2026-09-17 with “stop asking, you know phase 4 goals – proceed”.
Implementation/qualification remains work to perform, not an existing capability.

**Scope:** Unblock Task 12 of the approved Phase 4 plan without changing the
experiment, evidence retention, learning recipe, device checks or acceptance gates.

**Parent specification:** [Silent Cascade v1.0.2](2026-08-30-silent-cascade-design.md).
**Parent plan:** [Phase 4](../plans/2026-09-16-phase-4-autonomous-eventflow.md).
**Revision:** 2, superseding the rejected 64 GiB whole-evaluation proposal.
The user authorized up to **10 GiB of local disk** on 2026-09-17. This is a ceiling,
not a reservation. The subsequent proceed instruction approved implementation;
it does not claim that archive support already works.
**Inspected checkout:** `2995310` on `main`, 2026-09-17.

## 1. Problem and evidence

R2 provisioning is complete; integration is not. The private Standard bucket,
bucket-only local credentials and byte-exact round trip are documented in
[R2 setup](../../r2-archive-setup.md). No experiment data has been uploaded.

The historical Task 10 forecast requires 4,411,034,493,825 local bytes. Preserve
that failed preflight and its original source. The forecast is a conservative
scenario, not 4.41 TB already produced or an immutable lower bound on working disk.
Its upper retained-data estimate is about 3.68 TB, including approximately
37.61 GB of update journals. Archiving only episode trajectories is insufficient.

Available filesystem space is not permission to consume it. Use one configured
durable workspace; no repository relocation, mount or silent spill to another
volume. Existing repository/environment/history remains untouched and is not new
allocation. All new task output, including test/qualification scratch, logs and
abandoned attempts, counts even if it must live beneath the repository. Recheck
physical free space before execution; temporary directories are not exempt.

Relevant existing boundaries:

- `eval/artifacts.py:write_evaluation` flushes rows but fsyncs only at the end;
  evidence verification and crash discovery are also deferred. Add a storage-only
  episode commit protocol while preserving scientific bytes and the denominator.
- `eventflow/engine.py` publishes crash checkpoints/manifests. Several crashes
  can share a weights blob, so directory scans do not establish safe ownership.
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

1. **Chosen: committed episodes, small packs and streaming readers.** Archive
   during evaluation and restore one episode at a time. Add durable episode
   commits, explicit crash ownership, paged catalogs and streaming finalization;
   preserve original scientific bytes and semantic checks.
2. Whole-evaluation leasing is rejected: the measured scenario is approximately
   23.60 GB per 10,000-episode unit, already above the authorized local envelope.
   Lowering the old setting without changing publication/readers is unsafe.
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

The offline subprocess boundary requires a trusted Git version at least 2.45,
with native `--no-lazy-fetch` support. The operational parent resolves and pins
the real executable, version and content identity before launching a child.
The fresh child validates and captures that immutable pin once; nested offline
diagnostics inherit the captured pin through a dedicated sealed environment
field, never through scientific diagnostic intent or published configuration.
Only exact existing provenance command forms are allowed, with a sealed
environment, trusted working directory, all transports denied and executable
configuration features disabled. Unsupported Git fails closed; there is no
fallback to older Apple Git. Executable paths remain private operational state,
not archive or report content. The archive doctor diagnoses this prerequisite.

The `verify` job means existing artifact-only gate verification, not the local
quality-gate recorder. Each job has a strict closed request schema and a fixed
package entrypoint; callers cannot select a command, module or repository root.
Before child launch, require an explicit typed output bound tied to that closed
request, policy and executing package bytes. The preflight derives its category
maxima from source-defined output limits; callers cannot grant arbitrary extra
space. Persist the admission binding in the shared durable reservation rather
than maintaining another per-job certificate history. Missing bounds block launch.
Scientific schema authorities remain in the producer/readers. The lease layer
checks their exact expected inventories against transport and resident bytes.
An authenticated paged episode binding embeds the canonical typed episode commit,
its digest, ordinal, logical root, run identity and containing unit reference.
Lease selection validates that binding and one exact authenticated episode group;
it never infers ownership from filenames. Bindings can become cold after verified
publication and do not accumulate as permanent per-episode local files.

## 4. Bounded working set: at most 10 GiB

An explicit workspace root owns one durable global budget/baseline descriptor
and permanent lock under `.silent-cascade-storage/`. Per-run control descriptors
only locate that state; a new run or recovery destination does not gain another
allowance. All new outputs, including tests, caches and temporary files, stay on
the same device inside that root. Exclude only authenticated unchanged pre-existing
baseline entries, never a blanket snapshot that makes already-generated task or
qualification output free. Missing/corrupt accounting fails closed. Baseline
pages and operational state are themselves charged metadata.

Initial operational limits, measured and enforced independently of ML config:

| Allocation | Limit |
| --- | --- |
| Pending producer evidence / unuploaded spool | 2 GiB |
| Restored episode evidence cache | 2 GiB |
| Active checkpoints, weights and mutable controls | 1 GiB |
| Current metadata and paged catalog/index cache | 1 GiB |
| Transfer and atomic-publication scratch | 256 MiB |
| New logs and operational diagnostics | 256 MiB |
| Emergency in-flight failure/checkpoint allowance | 1.5 GiB |
| Protected free headroom, not ordinary allocation | 2 GiB |
| **Total allocation plus protected headroom** | **10 GiB** |

Normal allocation is at most 6.5 GiB; emergency use may reach 8 GiB with 2 GiB
protected. No additional 16 GiB reserve or preallocation is required. Categories
are shared across workers/tests/recovery, not separate allowances per process.
Account for actual allocated bytes, directory metadata, atomic copies, partial
downloads and outstanding write reservations. Both the ledger and physical-free-
space checks must pass. Logs/tests cannot bypass accounting through TMPDIR.

Adversarial test fixtures below an explicitly bound real scratch directory are
opaque accounting data, not authorized I/O paths. Measurement counts symlink
inode blocks without following targets or descending symlinked directories,
and conservatively counts hard-linked scratch files once per name. Root and
non-scratch path protections remain strict; actual artifact reads/writes still
reject unsafe links and ambiguous ownership. This permits preserving attack-test
evidence without an exception to the byte budget or permission to access its
targets. Unknown paths do not gain this explicit-scratch treatment.

Additional bounds: one episode writer, one episode reader and one network operation
at a time; chunks at most 32 MiB; target packs at most 128 MiB or 256 episode commits,
with a larger single episode allowed only within a 1 GiB admission bound; journal
segments at most 128 MiB or 128 records, never splitting a record. Catalog pages
have at most 1,000 entries and 16 MiB decoded bytes. New remote bytes remain capped
at 4,500,000,000,000, including operational metadata and ambiguous writes.

Reserve full possible output before each episode/update. There is no established
aggregate per-episode bound today: derive it from permitted artifact counts and
serializer limits, including crash/atomic overhead and shared weights. Compare
real serialization with the unchanged 1.2 measurement safety factor as well.
If admission/emergency allocations cannot cover every permitted output, preflight
fails before production. Never truncate traces, reduce denominators, lower event
ceilings or drop failures to fit. Apply the same rule to verification scratch.

Only current rows/metadata, mutable controls and required rolling checkpoints/
weights stay pinned. Archive closed metadata, journal segments, figures and logs
as they grow. Keep bounded catalog heads/pages, not every receipt/index shard,
local. Back up checkpoint-bound control snapshots. Immutable journals can be
transported before checkpoint commitment, but only a durable checkpoint head marks
the committed prefix. Stopped attempts still count until safely archived.

If a budget, transfer or disk check fails: stop issuing new work, retain every
written byte and report `storage_blocked`. Use the existing last-durable-checkpoint
recovery semantics after an interruption, retaining abandoned journals/partial
evaluations. Do not invent validation resume or silently rerun in a new seed.
For a live process, bounded waits between safe boundaries preserve current state;
supervisor failure does not grant permission to delete its pending output.

Before each operation, verify that the supervisor's existing source-bound job
reservation still has room for the complete permitted output and publication
overhead. Use a strict local status/admission exchange, not a second reservation
of the same capacity or a health-only acknowledgment. Drain verified pending
packs or stop before further neural work when the remaining allowance is too
small; all materialized bytes and protected physical headroom stay charged.

The remote byte limit is a client-side stop, not a Cloudflare spending cap. R2
continues charging for retained storage until its owner removes it. Do not add
automatic deletion/lifecycle policies or expand the budget automatically.

## 5. Archive format and lifecycle

Use an additive `phase4-r2-archive-v1` envelope outside the logical scientific
tree. Preserve every original relative path, file byte, SHA-256 and source/config/
checkpoint/generator/manifest identity. Reject absolute paths, traversal, duplicate
or overlapping logical member ownership, links, special files, unexpected members
and changed files. Disjoint journal segments may share the logical run root.

Unit kinds are closed: episode pack, evaluation metadata, immutable journal segment,
checkpoint-bound control snapshot, completed diagnostic output and stopped-owner
partial evidence. Separate owned files from borrowed shared weights; active
borrowers pin their inputs. Partial evidence is sealed only after its writer is
confirmed stopped, never relabeled `DONE` or passing. Include crash files without
a durable row, preserving bytes while leaving outcome unknown. Preserve restart
and abandonment bindings.

Journal units bind a reconstructible `phase4-journal-segment-commit-v1` descriptor
through their existing evidence-identity digest: predecessor head, sealed end
head and ordered record hashes. Verify the complete exact raw-record chain and
member count when sealing and reading. This adds no scientific artifact or
permanent local locator and preserves the canonical unit schema and old hashes.

### Durable episode publication

1. Reserve output before neural work. The engine exposes an additive
   `PublishedCrash` descriptor: manifest, optional runtime checkpoint and shared
   weights with hashes. Do not infer ownership by before/after directory scans.
   This changes instrumentation only, not event scheduling.
2. Publish original neural sidecar, retained trajectory, telemetry and any crash
   files; verify evidence. Append the canonical row to `.rows.pending.jsonl`, then
   flush and **fsync the row before** publishing its transport commit.
3. Atomically create `phase4-evaluation-episode-commit-v1` outside the scientific
   tree, binding identity, ordinal/public ID, row offset/length/hash, exact owned
   path/size/hash inventory and shared input references. Without it no live-writer
   eviction is legal. Shared crash weights remain pinned through finalization;
   they are never owned by the first failed episode.
   Transport manifests authenticate explicit episode-owned path groups; directory
   names cannot establish ownership because ordinal files are flat and crash
   files can live in another directory.
4. Pack committed episodes, upload/readback/receipt, then evict exact unchanged
   owned payloads. Keep active rows/controls. Backpressure is between episodes,
   not a change to neural computation.
   Where units share a logical directory, supply an explicit producer-authoritative
   retained-file custody snapshot for controls, shared inputs and sibling evidence.
   Verify full resident coverage, exact retained hashes and disjoint ownership;
   bind the snapshot to the eviction intent and recheck after interruption.
   Unknown or changed members stop eviction; retained files are never deletion
   targets. A directory scan cannot establish scientific ownership.
5. Finalization leases each episode in turn, revalidates evidence/crash links,
   builds the original crash index and derives the full logical inventory from
   authenticated ownership plus controls, not resident `rglob`. Publish original
   rows/index/metrics formats. `DONE` requires the full denominator and every
   original semantic check. Later validation/execution files get separate closure.

An episode transport commit is not evaluation `DONE` or a training checkpoint.
Do not split 10,000-row evaluations into smaller scientific evaluations. Keep the
original metadata reader limits and prove that full control files fit them.

File inventories use sorted, bounded canonical JSONL shards with counts and stream
hashes. Small run/corpus roots reference bounded pages rather than every file;
closed catalog/receipt pages are themselves cold-readable. Validate decoded sizes,
counts and the complete stream hash, including the last page. Each file entry
specifies path, bytes, SHA-256 and ordered byte spans in
transport chunks. Chunks concatenate original bytes, with no lossy transformation
or second full-size bundle on disk. Pack/read at bounded buffer sizes. Existing
compressed trajectories remain byte-identical. Validate spans, totals, order,
missing tails and integer bounds before allocating/restoring.

Run/corpus catalog generations use bounded content-addressed Merkle nodes. Run
roots retain both unit references and active file ownership after local payload
or manifest eviction; control snapshots have separate versioned ownership.
Cold readers authenticate pages through explicit root hashes, without bucket
listing. Incremental publication touches only changed trie paths and coalesces
intermediate node rewrites before writing, keeping metadata working space bounded.
Exact and ancestor/descendant file ownership collisions remain errors across
archived and resident units. Becoming unreachable from the newest generation
does not by itself authorize deletion of historical catalog evidence.

Object keys are content-addressed within a run-specific prefix. Upload using
create-only conditional writes. An existing object is reusable only after its
downloaded bytes match the expected size and SHA-256; a collision fails. ETags,
metadata and successful HTTP status are not SHA-256 verification.
Bound downloads during transfer with qualified range/total-size checks, not only
after saving the file; an oversized remote object cannot consume unlimited scratch.

Reserve remote-byte budget durably before each new object write, including
manifests, catalogs and ambiguous/failed attempts. A retry of the same verified
object does not consume a second unique-object reservation. Never reset usage on
restart or ignore orphaned/unconfirmed writes to remain under the configured cap.

Initialize this reservation history explicitly from measured accounting evidence,
using a create-only record bound to the operational destination identity. Missing
or corrupt history is not a fresh zero balance. Qualification bytes carry into
production accounting; changing a prefix does not reset usage. Catalog-only
evidence recovery cannot authorize new remote writes until reservation history
is authenticated. Pending local units do not themselves imply remote usage.

Keep reservation and receipt/eviction-intent history in a bounded authenticated
operational index with closed typed records. Only its roots and bounded pending
transactions stay resident; verified historical pages are cold-readable. No
unbounded per-object/per-unit local metadata or general database is introduced.
Before publishing an operational metadata batch, durably reserve a finite
source-derived bound for every page, root and commit, including the batch's own
overhead. This local pending reservation does not trigger recursive reservation
uploads. Retain ambiguous charges and settle only provably unused bytes after
fully verified publication. Missing/corrupt pending history fails closed, and
catalog-only recovery cannot reset remote usage.

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

Introduce a local-only `EvidenceContext` for paged discovery, metadata and single-
episode leases. It yields authenticated sparse real local trees, not links. Add
an evaluation-header/indexed-row scanner plus episode verifier, reusing existing
semantic checks and returning parsed evidence once. Preserve eager
`load_evaluation(Path)` compatibility.

Hold bounded corpus metadata; validate each original row offset/length/hash and
identity, lease its sidecar/telemetry/trajectory/crash files, verify semantics,
consume compact metrics/selected decoded plot examples, then release. Shared
weights remain available. At closure classify every `DONE` member exactly once
as metadata/shared/episode evidence; reject missing/extra/duplicate/unclassified
paths. Exhaust the scan before a passing gate/report: corruption in episode 10,000
invalidates the result. Partial scans preserve newline-complete/unknown-outcome
rules and inspect orphan crash files too.

Aggregate journal, training-result, gate and report readers accept this context.
They must visit every expected unit, including failed and abandoned units. Keep
bounded index iteration, paging journals/catalogs too; leaving every metadata
shard local would merely move the storage failure. Do not duplicate the giant
training index on disk. Compatibility APIs may materialize dictionaries in RAM;
report that honestly. Any semantic-validation cache must be session-local and
keyed by immutable inputs and verifier source, never a persisted transport receipt.

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
each episode/page fully in bounded succession. No cold path restores a whole
evaluation tree. This changes *simultaneous materialization*,
not required input coverage. Keep old local recovery unchanged, preserve original
artifacts, and reject overlapping destinations and changed gate identities.

Operational run IDs are path-bound. Fresh recovery therefore uses one fixed
parent-bound read-only source view on the existing destination session, followed
by bounded re-publication into destination-native units/catalogs; it does not
relocate catalog heads or authorize foreign members. Preserve scientific bytes,
paths, EpisodeCommit digests and source/config identity. The duplicated remote
inputs, source lease, current destination copy, metadata and readback all count
against existing limits. Complete original-input authentication precedes creation
of the scientific destination; complete native destination authentication precedes
new neural work. Pin source heads/gate/owner authority and reject changes. There
is one IPC sequence, one lease registry and one episode lease across both views.
Imports require an active authenticated source lease and parent-derived membership
in the original recovery input stream. The source view cannot write or evict.
No new manifest schema, generic multi-run resolver or extra budget is introduced.

## 7. Provenance, compatibility and security

Completed engineering/test history created during this plan stays charged to the
same global10GiB envelope, even outside the active operational directory. Frozen
external-root charges are authenticated accounting, not baseline exclusions or
additional allowances. Existing categories and protected reserve stay unchanged.
An independently reviewed early archival prelude may qualify the provider and
archive eligible ordinary regular-file history before remaining large verification
runs. Use existing diagnostic custody/readback/eviction; unsupported sparse/link/
special fixtures remain local. No mount or filesystem codec is added. Qualification
keeps its512MiB total, engineering bytes share the existing4.5TB remote budget,
and all final local/scientific gates still run after implementation is complete.

The private frozen-engineering inventory must also remain space-efficient across
small evictions. Use stable immutable leaf partitions and a bounded authenticated
index so unchanged records do not cause a new whole-inventory generation. Only
receipt-authorized removed members and their revalidated ancestor records may
change; surviving rows retain their original leaf partitions. Keep old leaves
and indexes charged and readable. This does not authorize metadata deletion,
batch eviction, rebaselining or changes to scientific/public catalog schemas.
Admission includes changed leaves, index publication and atomic coexistence;
the single-unit pending receipt remains the recovery and charge-update boundary.

Early provider qualification uses a narrow private driver, implemented and
reviewed before the first real bootstrap. It composes existing remote-ledger,
receipt and streaming-restore validators rather than constructing the pilot
supervisor or introducing another control history. One parent reservation spans
the real interrupted child upload, collision/readback retry and restored copies.
The fixed 64 MiB probe and one positively reviewed copied debug episode have a
256 MiB additional remote stop limit within the unchanged 512 MiB combined
qualification/smoke allowance; neither number substitutes for source-derived
output bounds. Bounded diagnostic records contain hashes and measured counters,
not credentials, destination settings or private paths. Actual provider checks
and later smoke are distinct gates, and local doubles cannot certify R2 behavior.

### Bounded qualification continuation after an interrupted attempt

A failed qualification remains failed. If its immutable, source-bound evidence
already proves the real successful provider create followed by SIGKILL, a
separately identified continuation may finish the same qualification campaign.
This is a private storage-only recovery path, not permission to manufacture a
normal fresh-run result or reuse a scientific pass. A new complete qualification
would otherwise retain another interrupted chunk and can exhaust the unchanged
256 MiB scratch ceiling even though the remaining checks fit.

The continuation uses a distinct versioned result type and fresh evidence and
restore roots. It must authenticate exact predecessor proof hashes, historical
reviewed-source authority, original probe/debug manifests and inventories,
policy/protocol, transport identity, and the actual interruption object's hash.
It must establish historical premature-receipt/authorization absence from the
preserved execution evidence and the authenticated historical driver's ordering;
today's completed receipt is not evidence that it was absent before interruption.
Missing or ambiguous historical evidence refuses continuation. Record explicitly
which checks are inherited and which execute now. The cause of an unexplained
failed resume remains unresolved even if a later retry succeeds.

Reexecute provider collision and exact readback against the interrupted object's
original key, using the verified original payload and existing reservation
history. Then require a completed child retry with actual downloads, identical
receipt and unchanged remote reservations; reject restore to an existing target;
archive/read back the original sealed debug unit; restore both units one chunk
at a time to fresh targets; verify exact inventories and source/restore bytes.
The fixed 64 MiB probe and reviewed copied debug episode remain mandatory.
No inherited successful status can substitute for these remaining operations.

Source directories in the failed attempt are read-only. Pass explicit fresh
event destinations through parent and child helpers rather than deriving them
from the old source root. One newly admitted owner and explicit child capability
span all continuation writes and process cleanup. Use source-derived remaining-
work bounds: old interrupted chunks/sources are already charged, no new killed
chunk or duplicate source copy is required, and complete restored trees still
count. Preserve every predecessor, recovery and continuation byte. The 256 MiB
additional campaign stop and 512 MiB combined qualification/smoke allowance cover
all attempts cumulatively, including failures, not a new allowance per attempt.

Final full authentication follows writer cleanup and precedes create-only result
publication. Failed continuation produces no pass. An independently reviewed,
actually executed composite record satisfying every check above can satisfy the
early provider gate for engineering archival; it is not the final stable-source
storage gate, local verification gate, smoke, or Phase 4 scientific acceptance.
Keep the existing fresh-run interface/result unchanged. Do not add automatic
resume, a public CLI, another ledger, or a general workflow engine.

Native allocated bytes can exceed rounded logical bytes: a preserved32MiB
readback allocated33,882,112bytes on the target filesystem. Remaining-work
admission must include the established conservative `2 * rounded(size) + block`
native-write allowance for downloads and full restored files, alongside exact
logical serializer/coexistence bounds and live allocation checks. Do not describe
this operational cushion as a universal filesystem allocation proof.

Apply the same allowance to engineering archival's coexisting generated payload
and provider readback. With `native(n) = 2 * rounded(n) + block`, the pair term
is `native(largest) + native(max(largest, policy.page_bytes))`: the generated
payload can coexist with a cold operational-root read accepting the full page
limit, not merely a same-sized payload readback. Retain existing catalog and
directory additions and reader limits. A complete group that no
longer fits must use the existing exact reviewed-path subset interface, not a
smaller allowance. For the first audited group use at most14MiB logical batches,
keep every tensor/capture sidecar pair inseparable, and start with the largest
pair. This ceiling is only a grouping rule; freshly authenticated candidate,
exact positive content review and full native/category/global/physical/remote
admission remain mandatory for every batch. Refresh candidate identity after
each receipt-authorized eviction and preserve all unselected siblings. A later
reviewed correction retains the earlier qualification's actual executor tuple;
it cannot promote that historical result to new source. Final current-source
local/resource/smoke gates remain prerequisites for the scientific pilot.

If retained implementation tests exhaust an internal slice, preserve and report
the failed admission separately from functional outcomes. A reviewed early
provider continuation may use prior full-module functional evidence plus fresh
admitted coverage of every amended path, with that distinction explicit; it
cannot claim a full current-source local gate. Independent review and actual
provider checks remain mandatory before engineering archival. Complete stable
module coverage and the unchanged Task7 source-bound local verification follow
verified reclamation and remain prerequisites to final storage readiness and the
scientific pilot. This changes verification order only, not scientific acceptance,
local/remote ceilings, retained evidence, or the required final test coverage.

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

### Reviewed maintenance after the first bootstrap

The first live qualification exposed an adapter contract error: the publisher's
compound receipt/eviction keys were rejected by the R2 key validator. Preserve
those already-recorded keys and the failed run; correct the validator, never
rename pending objects or reset the remote ledger. Recover the original pending
publication with the existing `archive_unit` path under bounded admission. Its
manifest retains its original source identity; recovery evidence names the
distinct reviewed executor. A new qualification uses a fresh output/run identity
on the same workspace, transport namespace and charged remote history.

Bootstrap source identity is immutable historical provenance, not a permanent
ban on reviewed maintenance. After independent review, a controller may publish
one create-only, at-most-16-KiB reviewed-source record for an exact committed
revision and actual package digest. Store it in the counted local ledger at
`source-authorities/<executable_sha256>/<source_commit>.json`. It binds the
bootstrap authority hash, policy hash, exact revision/digest, closure schema,
review evidence hash and optional failed-operation evidence hash. Callers supply
its content hash explicitly; no mutable "latest" pointer, authority replacement,
budget reset or automatic approval of changed source is permitted. The unchanged
bootstrap tuple remains usable without such a record. Independent review is a
controller obligation; a hash-shaped string alone does not prove that review.

Qualification identity and engineering candidate identity retain the exact
reviewed-source hash. Recheck it before sealing or resuming engineering work.
Old records and source tuples are never promoted to a later executable. Reject
dirty/uncommitted package code, mismatched source/hash/policy/anchor, absent or
changed records, and source adoption while an engineering eviction is pending.
Different reviewed releases may coexist without rewriting earlier evidence.

### Bounded retained-history verification cadence

Live qualification also showed repeated full scans of approximately 6.2 GB of
frozen history around each provider operation. The default `budget.check()` must
continue fully authenticating retained history. A finite owned qualification or
supervisor job may instead authenticate that history at entry, use its unchanged
charged amount for frequent live accounting, and fully authenticate again after
all owned writers are stopped and before accepting success. Live checks still
measure actual operational allocation, reservation growth, every category,
normal/global caps, root identity and protected physical headroom. A pending
engineering eviction cannot use this optimization because it changes custody.

Qualification performs three parent full scans overall: the real reviewed-source
precondition before reservation ownership, scoped admission, and final staged-result
authentication after cleanup. Two of those scans are inside the owned operation;
children perform zero full scans. Supervisor jobs require two overall. These fixed
counts are independent of provider-call and chunk counts. Preserve the fully
authenticated source precondition; the earlier two-overall qualification test
incorrectly hid it by substituting the source resolver.

Each live measurement prepares validated category-prefix component tuples once
and matches whole path components in longest-prefix order. This avoids repeated
ancestor reconstruction without caching measured allocation or reducing checks.
Exact roots, nested overrides and opaque scratch descendants retain their existing
classification and descriptor-safety rules.

Any reusable accounting snapshot is process/operation-scoped, bound to the live
reservation and exact authority/snapshot identity, and cannot become a persistent
cache or a restart grant. A qualification child may inherit only that explicit
live parent's scope; a new independent process defaults to full authentication.
Full verification failures prevent successful qualification/job acceptance. The
tradeoff is explicit: unexpected edits to otherwise frozen history may be
detected at the finite boundary rather than every polling iteration. They never
become baseline exclusions or uncharged bytes. This storage-only cadence change
does not alter scientific observations, events, metrics, artifacts or gates.

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
It does not simply replace the old 4.41 TB number with 10 GiB. Demonstrate the
combined allocation, headroom, qualification/recovery/log accounting and absence
of hidden scratch. If measurements do not support the limits, report the blocker.

After implementation and verification, resume the existing Task 12 sequence:
introduced manifests/audits, one seed-11 run, all final checks, honest result and
whole-phase review. No Phase 5 work or positive research claim is authorized here.

## 9. Reference and review boundary

Previously inspected operational references remain implementation starting points;
this revision makes no new provider-capability or price claim. Qualify actual
account behavior before use.
[Cloudflare S3 compatibility](https://developers.cloudflare.com/r2/api/s3/api/).
[Cloudflare limits](https://developers.cloudflare.com/r2/platform/limits/).
[Cloudflare pricing](https://developers.cloudflare.com/r2/pricing/).

**Self-review:** Scientific behavior and old evidence are unchanged; journals,
partial evidence, control snapshots, reader completeness and bounded recovery
are included. The 64 GiB design is withdrawn. The user authorized the 10 GiB cap,
but episode publication, streaming readers and measured feasibility remain to be
implemented and verified under the revised plan.
