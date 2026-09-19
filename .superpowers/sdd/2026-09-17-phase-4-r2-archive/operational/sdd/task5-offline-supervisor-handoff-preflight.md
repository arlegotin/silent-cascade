# Task5 offline supervisor handoff — read-only preflight

Source basee323533. Prepared beside Task C by
`/root/offline_supervisor_handoff_map`; controller checked the cited custody,
supervisor, session request/status and ledger ownership seams. This is an
interface map, not a design, implementation approval, test, or full-run admission.
Task C does not implement inherited ownership; its reviewed scope remains the
fixture-held child/parent allowance and public-spool path correction.

## Existing interfaces

| Producer / consumer | Existing state | Remaining connection |
| --- | --- | --- |
| supervise_job / _supervise_bound_job | Closed job/request, source, run/control/workspace, policy and JobOutputBounds bind admission; supervisor.py:205,221,279 | _decode_job:63 admits no offline custody field or locator. |
| Parent ledger / archive server | _ScopedAdmission(token,before,retained); retained has owner PID/create-time, reservation/admission/workspace/policy/retained commitments and child_inheritable; ledger.py:34,49,452,537 | Supervisor uses child_inheritable=False and keeps the active budget/admission in its own process; supervisor.py:308. |
| Parent job.json / _child_main | job,request,run_dir,workspace_root,source_sha256,policy,file_bytes; supervisor.py:330 | No process attempt/limits, private FD, budget or retained admission is passed. Child checks package source, sets file limit and installs boundary; :159-167. |
| Server session.json / LocalArchiveSession | schema,run/session IDs,run path,policy hash,parent PID/create-time; session.py:239 | Identity/liveness is not offline process custody or storage ownership. |
| Session request / serve_once | Exclusive requests.lock, sequence, closed operation, identities/policy, child PID and payload; response binds request hash and same identity fields; session.py:447, supervisor.py:817 | Empty status returns only healthy=True (:1390). No existing custody request/response. |
| _child_main / _execute_job / pilot or checks | session plus ArchiveProducer and evidence_context forwarded; supervisor.py:102,108,119,182 | No offline_process_custody reaches run_pilot/run_pilot_checks. |

The parent holds supervisor.lock and its scoped reservation throughout child
lifetime (supervisor.py:301,372). Session waits check live parent PID/create-time
(session.py:438); the server checks its budget before and after request dispatch.
Those guarantees do not by themselves transfer the private capability.

`check_scoped` checks the original live owner and permits a different current PID
only for an explicitly inheritable retained scope (ledger.py:374-409). The
separate offline custody validator additionally demands its original same owner
and active reservation (pilot_offline_process.py:292-305). Merely flipping the
ledger flag would not satisfy that contract. `OfflineProcessCustody` contains
private object references and a pinned directory descriptor and is explicitly
nonserializable; it cannot simply be added to the current job JSON.

The plan explicitly defers this integration (plan:2027,2146) and requires
missing authority to fail before costly fresh work. Its separately planned
recovery-source view is read-only on the same session/sequence/lease registry
(plan:1268-1283); that tag is not offline custody and supplies no launch authority.

## Consequence for the next scoped plan refinement

The reservation owner is the supervisor; fresh diagnostic measurement would
run in a separately exec'd child. A future, explicit, reviewed handoff must
resolve that ownership/lifetime boundary and forward the resulting capability
through the existing pilot/checks interfaces. No new channel, permission,
inherited-capability implementation, category/cap change, or success claim is
authorized or supplied by this map. Actual diagnostic/full-gate and recovery
qualification remain outstanding after Task C.

Controller also checked the separate allowance forwarding seam:
`run_pilot_checks` currently calls measurement with custody but no
`write_allowance` (train/pilot_checks.py:392). The primitive default is
explicitly no child byte cap. Future production integration must pass the exact
held child allowance as well as custody; possession of parent custody alone
does not activate Task B's child limit. This is already required by Task C's
future-caller contract, not implemented by the current fixture-only task.

No files beyond this controller-written note/progress changed for the preflight;
no tests, project imports, child probes, provider/network or scientific work ran.

## Read-only design comparison after Task C primitives

`/root/offline_handoff_design` inspected the concrete ownership seams without
executing or modifying code. Its recommendation is a distinct delegated handle
with an inherited pinned private-directory descriptor; this is analysis, not
implementation approval. The original same-owner custody must remain strict.

| Approach | Concrete tradeoff |
| --- | --- |
| Supervisor computes diagnostic before pilot/checks | Not a drop-in: a nonempty fresh run enters durable recovery in pilot_workflow.py:328, and recovery requires full original closure plus destination ownership before new neural work. |
| Child asks supervisor to run diagnostic at the existing measurement point | Preserves same-owner custody but adds another child process group, cancellation and continuous budget/liveness supervision while the request blocks. Existing cleanup targets the pilot group. |
| Keep nested diagnostic launch, issue a distinct child-scoped handle | Preserves existing launch order/process-group cleanup, but requires explicit descriptor delegation and parent validation, not serialization of OfflineProcessCustody. |

Any delegated design needs source-bound exact process limits and child allowance
inside the existing global admission; aggregate category amounts alone are not
a child allowance. Supervisor retains its original custody/reservation. A fixed
descriptor is inherited only by the direct pilot/checks child, made
noninheritable immediately, and excluded from the nested scientific worker.
The handle cannot expose a ledger mutation API or caller-selected private path.

Issuance must bind the admitted job/root/attempt, source/policy/admission,
parent and direct-child PID/create-time, workspace/descriptor identities, limits
and allowance. A strict existing-session status variant could revalidate only
the issued attempt/publication phase in the original owner; healthy status or
its response alone is not authority. Exact delegated-type validation and
explicit custody-plus-allowance forwarding would be required at every consumer.
Private raw bytes never travel in IPC or public evidence.

The production hold must also prevent intervening ordinary producers from
spending future diagnostic capacity. Task C proves fixture arithmetic only;
it is not a production hold. No second reservation, new ledger, general resolver,
cap increase or relaxed frozen-history/private-export boundary follows.

Falsifying controls should cover missing/wrong/stale descriptor and owner,
direct-child binding, exact/short held capacity after unrelated writes, genuine
tiny nested private-output failure, interruption with joined child group,
duplicate attempts and safe reuse with no descriptor or new launch. All require
separate finite admission. A versioned, explicit Task5 refinement must settle
these interfaces before any inherited implementation is dispatched.

### Existing production admission seam

Follow-up analysis found no need for a new ledger API: ArchiveProducer._before
sends a closed operation name to the current status handler, which derives
source-bound maxima. A future fixed-hold comparison can require each next
operation peak plus the remaining offline share to fit the existing remaining
reservation. It must also protect the private logs share against supervisor
stdout/stderr writes. The hold is not a replenishable per-operation token.

Existing update/validation/evaluation boundaries do not cover every later
writer. Final evaluation request/execution controls, whole continuation and
numerics diagnostics, report/cache writes, gate attachments, workflow startup/
final controls and recovery imports need explicitly covered spans or fixed
additional closed operations. The update bound is journal-sized, not authority
for all these outputs. Whole-span maxima must be derived before admission;
source-sized serializers alone do not establish a combined peak.

This is feasibility analysis only. No production hold, delegated descriptor,
status variant or new phase boundary is implemented or approved by this note.
