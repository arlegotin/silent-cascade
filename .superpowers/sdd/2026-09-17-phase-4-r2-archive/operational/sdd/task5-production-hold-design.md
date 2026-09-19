# Task5 production offline hold proposal

Status: design only at frozen source `39ecdf4` (archive seams unchanged from
`2e5b96d`). This proposal adds no launch
authority, descriptor inheritance, reservation, quota, diagnostic, or full-pilot
admission. D1's optional `OfflineWriteAllowance` transport remains only data
plumbing: carrying the object does not prove that bytes are held.

## Decision

Do not implement a fixed-hold-only field yet. First prove that every writer in
the complete pilot/checks workflow has a source-derived closed boundary and that
the resulting simultaneous category maxima plus the offline hold fit the
unchanged policy. An unused forever-held field would encode the wrong eventual
lifecycle: once the diagnostic consumes `C`, subtracting unchanged `C` again
would defeat exact-fit post-offline gate/report admission.

Final evaluation controls,
continuations, numerics, report generation, gate collection, workflow controls,
and recovery import contain writers outside the current closed `before_work`
boundaries. Their source-derived whole-span maxima and explicit calls must be
settled first. The first implementable code slice should close final-evaluation
request/completion controls only; it is useful independently and adds no hold or
authority. Only after whole-workflow feasibility is proven should one coherent
hold-lifecycle/delegated-descriptor plan change supervisor admission.

## Existing reservation and admission seam

`supervise_job` authenticates `JobOutputBounds` against the exact job request,
policy and package source (`archive/supervisor.py:214-228`).
`_supervise_bound_job` binds run/control categories and opens one
`_scoped_reservation` containing the seven category amounts before spawning the
child (`:285-320`). The parent owns that reservation for the whole child lifetime.

The child sends only closed session operations. `ArchiveProducer._before` sends
`status: {"before_work": <fixed-name>}` (`archive/producer.py:649-650`). The
parent maps the fixed name through `before_work_bounds`, measures the live
reservation, and requires the next operation maximum to fit the remaining
category amount (`archive/supervisor.py:1412-1431`). Checks occur before and
after every request (`:859-875`). This is the correct hold integration seam; no
new ledger API or session operation is needed.

## Target fixed hold contract (deferred)

After whole-workflow feasibility is proven, add one strict local-admission model
in `archive/types.py`, embedded optionally
in `JobOutputBounds`:

```python
class OfflineJobHold(StrictModel):
    schema_version: Literal["phase4-offline-job-hold-v1"]
    process_limits: dict[str, int]
    write_allowance: dict[str, int]

class JobOutputBounds(StrictModel):
    ...
    offline_hold: OfflineJobHold | None = None
```

The model accepts exactly the four `OfflineProcessLimits` keys and exactly the
three `OfflineWriteAllowance` keys, reconstructs the exact dataclasses for
validation, and rejects bool/coercion, unknown/missing keys and enlarged process
limits. It contains no path, descriptor, attempt, PID, token, category amount or
caller-selected private locator. Absence preserves current jobs and grants
nothing. Presence is permitted only for `pilot` and `checks`; it still grants
nothing.

In `archive/supervisor.py`, a pure `_offline_hold_categories(spec)` derives a
conservative category hold from the validated values. With block `B=4096` and
`rounded(n)=ceil(n/B)*B`:

```text
public_P = 4*rounded(record_bytes) + 12*B
private_P = 2*(rounded(record_bytes)+rounded(stdout_bytes)+rounded(stderr_bytes)) + 8*B
spool hold = write_allowance.allocated_bytes + public_P
logs hold = private_P
```

The formula covers the four public names, three private names, atomic/final name
peaks, the public category cushion, and conservatively three missing public and
two missing private ancestors. For the current explicit 16 MiB / 102 names /
9 directories and default 16 KiB / 256 KiB / 256 KiB records/streams, it is:

```text
public_P  =   114,688
private_P = 1,114,112
spool     = 16,891,904
logs      =  1,114,112
total     = 18,006,016
```

Here `private_P` uses `8*B`, not `9*B`: three private final/temporary name pairs
and the two private directories missing in the admitted fresh-root state. These
are the exact current `_publication_peaks` values for that state. The
pre-launch calculation must not be reduced using an optimistic later phase.

Before child spawn, require the active reservation's original `spool` and `logs`
amounts to be at least these holds. This comparison does not add capacity. Pass
the immutable category tuple into `_ArchiveServer`; do not publish it in
`job.json`, `session.json`, a response, or scientific evidence.

Change the existing `before_work` check only as follows:

```python
required = next_operation_maximum + fixed_hold.get(category, 0)
if required > remaining:
    raise StorageBlocked(...)
```

Check all held categories even when the operation has no maximum in that
category. Thus an `update` cannot consume spool while supervisor log growth has
already consumed the protected logs share. Repeated checks never decrement the
pre-launch hold. `_check_budget` remains authoritative for live allocation.

The hold model does not use name/directory counts as global grants. Those counts
remain confined to the future fresh child root and are enforced only by the
intent-bound child writer.

## Required hold lifecycle

The eventual invariant cannot be `remaining >= next maximum + unchanged C + P`
for the whole job. Child scientific files consume `C`, while `intent.json` and
the process publications consume `P`; adding the original hold again after that
growth double-counts retained bytes. Phase-based subtraction is also unsafe:
the low-level parent uses publication phase 2 both before nested launch and
after capture while publishing stdout, so phase 2 does not prove that the child
was joined. The child inventory deliberately excludes parent-owned intent.

The later combined hold/descriptor design therefore needs three explicit states:

1. `held`: ordinary producer checks preserve complete `C + P`; phase values do
   not change it.
2. `executing`: entered only by the original supervisor for the one bound
   attempt immediately before nested launch. Reject all ordinary `before_work`
   requests while executing. The ledger charges actual output growth and the
   existing custody validator protects remaining publication `P`; do not also
   demand unchanged `C` from remaining bytes.
3. `terminal`: entered only after the exact nested child is joined and its exact
   create-only terminal result is durably authenticated after all required
   public/private publications. Retained growth remains charged; only unused
   future allowance ceases to be held. Failure flags, EOF, phase, reuse, or a
   missing result never cause this transition. Interruption aborts the job and
   keeps capacity unavailable until reservation close.

This serialization is the minimal safe approach. A dynamic formula such as
`aggregate hold - owned-root growth` could permit concurrent work, but would
need pinned no-follow inventories, an authenticated baseline, separation of
parent `P` from child `C`, and proof that arbitrary spool/log writes cannot earn
hold credit. That is a broader ownership mechanism and is not recommended.

## Existing closed maxima

`archive/producer.py:185-223` derives these only from fixed source constants:
`E=128 MiB`, `P=64 MiB`, telemetry `128`, commit `64 KiB`, allocation overhead
`256 KiB`.

| Closed name | Spool maximum | Metadata maximum | Covered writer span |
| --- | ---: | ---: | --- |
| `update` | 67,371,008 | 0 | one journal record |
| `episode` | 671,350,912 | 327,680 | episode payload, row, shared weights, commit |
| `evaluation` | 268,697,600 | 0 | final evaluation controls after evaluation begins |
| `validation_autonomous` | 1,812,201,600 | 327,680 | validation controls and one live episode peak |
| `validation_autonomous_weights` | 1,946,419,328 | 327,680 | above plus weights |
| `validation_one_hop` | 1,879,310,464 | 327,680 | one-hop extra control |
| `validation_one_hop_weights` | 2,013,528,192 | 327,680 | above plus weights |
| `checkpoint` | 268,697,600 | 0 | checkpoint pair |

These maxima remain unchanged. The small slice only requires `maximum + hold`
to fit; it must not fold the hold into each maximum or treat a successful check
as a consumable token.

## Uncovered writer spans

The following prevent production descriptor issuance even after the fixed-hold
slice:

| Span | Current writes before/outside a closed boundary | Required explicit boundary |
| --- | --- | --- |
| Workflow control | permanent owner/start controls and `workflow.json` around `_workflow` | fixed workflow-start/final-control maxima |
| Final suite evaluation | `.request.json` is written before `evaluate_episodes`; `execution.json` follows it | request and completion controls around each fixed `SUITES` member; existing evaluation/episode maxima stay separate |
| Continuation | runtime intent, checkpoint/replay pairs and final continuation JSON; failed comparisons do not advance `covered`, so `REQUIRED_COVERAGE` length 15 is not a record bound | first derive a finite bound from primary episode/event/candidate limits and retained mismatch behavior, then admit the whole span before `_continuations` |
| Numerics | intent, weights, manifest/subset, per-device captures, resume/runtime evaluations, report/DONE or failure; per-device runtime evaluation has no `ArchiveProducer` | first derive the simultaneous per-device/raw-row peak and prove it fits local spool, then add a boundary before `verify_pilot_numerics` |
| Report | one SVG per discovered evaluation plus tables, markdown and index | maximum tied to the authenticated fixed evaluation inventory before `build_pilot_report` |
| Gate collection | compact attachment shards, copied training-index shards and final gate | bound from fixed `SUITES`, `MAX_ROWS`, `MAX_FILE_BYTES`, and authenticated index cardinality before collection writes |
| Recovery | restored indexed inputs plus workflow/recovery intent before owned checks | closed recovery-import bound authenticated from the original artifact index before destination creation |

Candidate maxima must be derived without importing project code during planning:
fixed counts and byte ceilings are literals in `pilot_evidence_types.py`,
`archive/producer.py`, `pilot_checks.py`, `pilot_verification.py`, report writers,
configs and the authenticated training artifact index. Current feasibility is
not established. In particular, continuation mismatch retention invalidates a
simple 15-record multiplier, and multiplying numerical per-device evaluation
serializer maxima by all rows may exceed local spool. Actual logical sizes are
not admission. A later implementation plan must write down simultaneous peaks,
atomic coexistence, directory allocation and category routing for each span;
summing serializer limits alone is insufficient and an over-cap result must stop
the full-pilot design rather than weaken scientific work or enlarge policy.

Use explicit producer methods, not a caller-supplied generic name:
`before_workflow_control`, `before_final_evaluation_control`,
`before_continuation`, `before_numerics`, `before_report`,
`before_gate_collection`, and, only in a future supervised recovery path,
`before_recovery_import`. Each method maps internally to one new fixed
`before_work_bounds` case. No new `_OPERATIONS` member is required.

## Ordered delivery

### Next bounded slice: read-only whole-workflow feasibility

Do not change source yet. Produce a source-derived table for every uncovered
span with exact file/name counts, maximum payloads, atomic coexistence,
directories and category routing. Resolve continuation mismatch cardinality and
numerics per-device runtime retention explicitly. Combine the simultaneous
peaks with existing `before_work_bounds`, supervisor scratch/log/control maxima
and the deferred offline hold, then compare the result with unchanged category,
normal and physical limits. This audit must conclude either `fits` with complete
arithmetic or `does not fit`; observed historical sizes are not a substitute.

If it does not fit, stop. Do not reduce episode/device/control coverage, release
evidence, increase caps or treat serial phases as nonoverlapping without source
proof.

### First implementable code slice after a successful audit

Close only final-evaluation control gaps: add explicit fixed producer methods and
`before_work_bounds` cases immediately before `.request.json` and
`execution.json`, with exact/one-short spy controls proving denial before either
write. Keep the existing evaluation/episode maxima separate. This slice is
independent of allowance transport, hold state and descriptors and is useful
even if later hold work changes.

Then close the remaining audited spans in phase-scoped plans. After every writer
is covered, add one source-bound factory for complete pilot/checks
`JobOutputBounds` and re-prove the composite exact fit.

### Hold lifecycle and delegated FD

Only after that proof should one coherent supervisor slice add the strict hold
field, category derivation, `held/executing/terminal` lifecycle, exact/one-short
tests, and attempt/root/source/policy/parent/direct-child bindings. Descriptor
issuance must remain after pre-launch hold validation; ordinary producer work is
forbidden while executing; terminal transition requires joined/durable evidence.
Recovery remains separate unless a supervised recovery job is designed.

D1 transport may then carry the matching allowance, but the parent's live hold
and lifecycle validation remain the authority.

## Negative requirements

- Reject category amounts one byte short before spawning the scientific child.
- Reject hold fields on `verify`, `report`, or `replay` jobs.
- Reject changed request/source/policy identity, malformed numeric types and
  unknown hold fields.
- Never accept hold bytes from a session payload or let the child choose an
  operation maximum, path, category, attempt or private locator.
- Never transition out of `held` or `executing` after a phase, failure, reuse
  observation or repeated request; only the authenticated `terminal` condition
  may end the future allowance hold.
- Missing hold means no future delegated fresh launch; it is not an implicit
  unbounded allowance.
- Do not claim that existing producer boundaries cover the uncovered spans or
  that this proposal admits a full pilot.
