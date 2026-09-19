# Task5 parent-process privacy corrective brief

Read this first: approved plan correction5298eff, implementing R2spec98–107/528.
This is the follow-up to the Important privacy finding in the parent process
slice. Start source/test work only when the controller explicitly dispatches it
after TaskA review closes. Single source writer; current branch. No subagents.
Use Superpowers debugging/TDD/verification; preserve original evidence outside
operational. No providers, real scientific worker, model/fit/evaluation/replay,
copied corpus, full gate or make verify in primitive qualification.

Own only files named in Scope below; extend focused parent-process tests and
narrow early-guard caller tests. No TaskB serialized child allowance, TaskC hold,
supervisor inherited channel or generic quota service in this slice. The active
TaskA writer/reviewer may still make reviewed corrections before dispatch; use
the controller's supplied final source base. Do not change algorithm/RNG/gates.

Controller owns live reservation and exact execution admission. Before tests,
project imports or child launch, submit exact selectors, number of tiny children,
raw/logical/allocated bound, names/dirs/temp peaks and fresh operational scratch
prefix. Keep all failed artifacts. Fixtures are tiny isolated instances testing
the existing ledger, physically charged under the one controller reservation;
production never creates another ledger. Do not inspect real credentials or
reuse an old diagnostic prefix. Keep real scope history untouched.

Write only operational/sdd/task5-offline-privacy-report.md for your delivery
report. Include exact commands/results, source identities, retained allocation,
nonempty-stream and prelaunch-tripwire evidence, limitations, and source/test
commits. Do not stage controller docs. Independent review belongs to controller.

#### Privacy correction after standalone Task A

**Goal:** Remove machine paths and arbitrary captured output from publishable
diagnostic evidence while preserving source/launch authentication and failures.

**Architecture:** Hash-only versioned launch records plus fixed private capture
custody in the existing active operational reservation. No new storage ledger,
generic redactor, inherited-authority channel or scientific behavior change.

**Tech Stack:** Python 3.12, existing process capture, strict schemas, pinned
durable file writers, existing scoped storage admission and pytest.

**Spec:** R2 design Sections 2 and 7, especially executable-path privacy and
machine-path/archive exclusions. This corrective slice supersedes the preceding
parent-process plan wherever its plaintext intent/public raw logs conflict.
Standing approval covers this reversible correction. Execute serially after
Task A source freeze/review and before production Task B, real diagnostic or
archive promotion. Existing functional tests are not privacy clearance.

##### Scope

Change pilot_offline_process.py, pilot_offline.py, pilot_evidence.py,
pilot_checks.py and bounded process tests; narrow forwarding/pre-work guards
in pilot_workflow.py/pilot_verification.py; private-subtree .gitignore rule.
No new ledger/resolver/redaction, historical/scientific changes or FM calls.

Defer inherited supervisor/FitGuard custody to open Task5 integration.
supervisor._execute_job passes archive_producer/evidence_context; _child_main
creates LocalArchiveSession and calls _request("status", {}), NOT custody.
Invent no channel. Missing authority stops fresh diagnostics before costly
work: temporarily unsupported workflow, not completed delivery.

##### Hash-only versions and readers

Strict intent-v2 replaces output_root/python_executable with
output_root_sha256/python_executable_sha256. Retain executable-content
python_sha256, attempt, bootstrap/source hashes, revision and limits.
Reject plaintext/unknown keys. Exact identity recipe:

```python
def path_identity(kind: bytes, value: str) -> str:
    return hashlib.sha256(
        b"phase4-offline-process-v2\0" + kind + b"\0" + os.fsencode(value)
    ).hexdigest()
# kind: exactly b"output-root" or b"python-executable"
```

Hash canonical absolute no-symlink output root and exact absolute argv
executable string; aliases differ. Launch/bootstrap enforce these identities;
actual argv/_PENDING/environment stay private. Artifact readers compare original
bindings, not verifier-local root/binary; retain logical evidence_context root
agreement.

Strict result-v2 retains actual pid/returncode/EOF, intent/attempt binding,
captured-prefix counts/hashes and offline hash. Add per-stream disposition:

```python
for name in ("stdout", "stderr"):
    disposition = getattr(result, name + "_disposition")
    count = getattr(result, name + "_bytes")
    digest = getattr(result, name + "_sha256")
    if disposition == "empty_public":
        if count != 0 or digest != hashlib.sha256(b"").hexdigest():
            raise ValueError("offline process empty stream differs")
    elif disposition == "private_rejected":
        if count <= 0 or result.status != "failed" or result.offline_sha256 is not None:
            raise ValueError("offline process private stream differs")
    else:
        raise ValueError("offline process stream disposition differs")
```

For an empty-public stream the reader validates a present file is exactly empty;
absence is unavailable, never completed. For private-rejected streams any
corresponding public file is corruption, including a zero-byte substitute.

Completion requires two durable empty logs, joined zero exit, EOF, no failure,
bound report. Add private_output_rejected/private_custody_failure reasons;
earlier failure stays primary. No public exception text/paths/raw bytes or
fabricated child identity.

Report-v2 binds intent-v2. Current measure/reuse/collection require all safe
versions; full gate enforces AFTER source authentication. Keep old forensic
parsers, no rewrite/reexport. Outcome reader adds require_safe_process=False;
current consumers pass True. Old present intent fails; missing safe intent
is unavailable, not downgrade. Failed/private result is fatal even without
public logs. Validate all available members; preserve inventory/one-lease/
direct-read guards. Scientific readers never hydrate private custody.

##### Same-owner custody API

Keep process module Torch-free (lazy/type-only archive imports):

```python
def prepare_offline_process_custody(
    *, budget: _StorageBudget, admission: _ScopedAdmission,
    output_dir: Path, limits: OfflineProcessLimits, attempt_id: str,
) -> OfflineProcessCustody: ...
```

OfflineProcessCustody is frozen/slots, repr=False and nonserializable. Fields:
budget (private object reference), retained (_RetainedHistoryScope), token,
owner_pid, owner_create_time, workspace_device, workspace_inode, directory_fd,
directory_device, directory_inode, attempt_id, output_root_sha256, limits,
category_peaks (immutable tuple of category/byte pairs). No path-string field
or mutable allowance dict. Closing fd never deletes evidence. Trusted-process
integrity only.

Preparation/prelaunch/publication checks use real ledger interfaces:

1. Exact _ScopedAdmission type; require budget.active_reservation ==
   admission.token == admission.retained.reservation_token. Require
   retained.owner_pid == os.getpid(); reject inherited ownership in this
   slice even when child_inheritable. budget.check_scoped(admission.retained)
   verifies live PID/create_time, policy/workspace/retained identity and
   admission commitment; returns allocated totals.
2. Under existing workspace.lock, read budget._state()["reservations"][token],
   match pid/create_time and use record["amounts"]/record["before"].
   For category c, remaining = amounts[c] -
   max(0, allocated[c] - before[c]); cover full P at preparation, only the
   still-unwritten phase peak at later checks (no double charge).
   No reserve/rebind/replenish. Sole writer, no concurrent launch; remeasure.
3. Derive ONLY budget.workspace/logs/offline-process-private/<attempt_id>,
   where attempt_id is a closed nonce component, never a path. Call
   budget.require_path on this route and output_dir; verify no-follow
   ancestor/device/inode identities and absent create-only targets.
   Use state["paths"] longest-prefix category rule as measure(): private
   route MUST already count as logs; public category must also be admitted.
   Never change bindings while a reservation is live.
4. Pin directory0700/files0600 on the same device; reject symlinks/hardlinks,
   external aliases and public/private ancestor overlap. Public root must
   have the existing canonical run/final/offline shape. Reject overlap with
   its run, budget.root controls, spool/cache bindings and frozen engineering
   custody. Prove the fixed private route remains outside scientific/export
   and engineering membership; if boundaries are unavailable, fail admission.
   Revalidate fd/path identities before writes.

ledger.measure counts logs; preflight._inventory excludes live workspace from
engineering snapshot; engineering candidates use closed tmp/task snapshots;
scientific exports select run members. Private route stays outside membership.
Ignore **/logs/offline-process-private/; never producer/index/export/register
it. Future operational archives must exclude it or reject launch.

##### Forwarding and publication

measure_pilot_offline(..., process_limits=None, process_custody=None) validates
matching capability before public intent/Git/Popen. Missing authority raises
closed custody-required failure: no TMPDIR/arbitrary path/run-derived fallback.
Forward offline_process_custody=None through checks/recovery/verification and
workflow boundaries. Missing/stale custody fails before evaluation or fresh
complete-workflow training. Safe reuse/artifact-only reads need no new custody.
Document fresh-launch restriction; inherited wiring remains deferred.

After bounded directory preparation/source authentication, durably publish
private <=R `binding.json` receipt (attempt/intent digest/limits; no paths), then public intent
before Popen. Preserve pending/argv/env checks and worker hash validation before
Torch/report; offline boundary intact.

Persist every NONEMPTY captured prefix unchanged as private stdout.raw/
stderr.raw with create-only durable publication, including overflow/timeout/
cancellation. Never decode/print it. Discarded overflow bytes are not captured
prefixes. ANY nonempty stream fails, even exit0 or valid-looking offline.json.
Only captured empty streams may create public zero-byte .txt files.
Public count/hash/disposition contains no private locator.

Persist private bytes before terminal. Storage failure retains partial files,
publishes failed result if possible, else incomplete intent; no lossless claim.
Combined publication/launch failure keeps None identity and first failure.
Parent death leaves incomplete fence. No retry/cleanup/raw-public fallback
or completed-with-missing-private-logs state.

##### Same reservation P

Measure st_blocks*512. Keep TaskA's B=4096 admission granule:
A(x)=B*ceil(x/B), not actual allocation. Validate filesystem coverage; larger
bounds need review. 8192 is only the tiny-test cushion. Stream caps S/E,
record cap R:

P = 2*A(S) + 2*A(E) + 6*A(R) + B*(F+D+1).

Two raw files allow final+temporary coexistence; public intent/result and
private receipt each allow twice R. Empty public logs cost names only.
F=14 covers seven finals plus seven temporaries. D is exact missing directories
including fixed parents; +1 writer cushion. Defaults S=E=262144,R=16384,D=2
give P=1216512 bytes, NOT an admission.
Split P by actual category mapping, including temporary/name/directory peaks.
Private bytes charge existing logs; public bytes their existing category.
U counts existing bytes once; remaining reservation covers P plus separately
reserved child peaks within existing global/physical headroom checks.
Child offline.json/source/worker files remain outside P. No double reservation
or new ledger. Control harness may use8192 without changing production B.

##### Bounded TDD and handoff

Controller admits exact selectors/fresh prefix/bytes/files/dirs before tests;
no genuine worker/model/fit/evaluation/replay/provider. Assert prewrite limits.

- Tiny stdlib child emits b"/private/SECRET" on stderr and exits0: private
  stderr.raw exactly15B, public stderr.txt absent, result failed/private_rejected
  with matching digest/count and null offline hash; reuse rejects. Empty child
  proves PROCESS-only publication, never fabricated scientific metrics.
- Tripwire Git/Popen/evaluation; fresh checks with no custody must raise the
  fixed error with all tripwire counts zero.
- Independent hashes, no public sentinel; changed launch bindings rejected
  by validator/fresh-process audit.
- Stale owner/depleted budget/aliases fail before writes; account private bytes
  but exclude them from scientific/export/engineering inventories.
- Overflow/timeout/cancel prefix retention and failed-launch/publication combo;
  strict disposition/version contradictions, authenticated current rejection
  of old intent, forensic compatibility and missing-log failed result.
- Preserve cold corruption/lease/direct-read guards.

Retain failures; static checks, commit/freeze, controller repeat/privacy review.
No workflow/scientific/quota/provider/promotion proof.


##### Ordered implementation and additional exact decisions

- [ ] **RED: safe identity and publication.** Add independent domain-hash and
  strict-version tests before changing serializers/readers. Test an actual tiny
  emitted sentinel and verify its private bytes and public absence. The
  lower-level capture/publication test is process evidence, not a fabricated
  scientific report.
- [ ] **RED: custody admission.** Use tiny isolated test ledgers under the
  controller-admitted prefix. Test active same-owner scope, category exhaustion,
  stale owner, canonical root, overlap and symlink/hardlink rejection before
  any publication. These fixtures test the existing ledger; production creates
  no second ledger.
- [ ] **Implement the narrow preparation/publication path.** Derive attempt
  identity before measurement; measurement uses the capability's exact nonce.
  The only private retained names are `binding.json`, `stdout.raw` and
  `stderr.raw`. No arbitrary member argument. Preparation rejects any existing
  attempt directory and pins its new descriptor. Add explicit close/context
  support that releases descriptors only, never deletes retained evidence.
  The capability is operational-only: do not place it in config, public
  serialization, job requests or subprocess arguments.
- [ ] **Implement strict current versus forensic readers and early guards.**
  Add the exact `require_safe_process` keyword, explicit intent/result version
  dispatch, disposition checks and available-corruption checks. Forward
  `offline_process_custody` only through the listed existing call chain;
  artifact-only safe reuse needs no fresh capability. Reject fresh work lacking
  custody before training/evaluation/numerics, not after they consume time.
- [ ] **GREEN: bounded matrix and scoped static checks.** Run only the
  prospectively admitted named controls, preserve every failed prefix, and
  list full commands, source identity, actual allocation/counts and outcomes.
  Commit source/tests, freeze, then controller verification and independent
  task review. Do not run the real diagnostic, provider, source-copy workload
  or full local gate without separate complete admission.

Example independent hash assertion (no machine path is published):

```python
expected = hashlib.sha256(
    b"phase4-offline-process-v2\0output-root\0" + os.fsencode(str(output))
).hexdigest()
assert intent.output_root_sha256 == expected
assert str(output).encode() not in canonical_json_bytes(intent)
assert "output_root" not in intent.model_dump()
```

Example failure assertions after the tiny sentinel child:

```python
assert captured.stderr == b"/private/SECRET"
assert result.status == "failed"
assert result.stderr_disposition == "private_rejected"
assert result.stderr_bytes == 15
assert result.stderr_sha256 == hashlib.sha256(captured.stderr).hexdigest()
assert result.offline_sha256 is None
assert not (output / "stderr.txt").exists()
```

The test's private reader verifies the fixed `stderr.raw` bytes using the
pinned fixture custody; no private path or raw value enters a public fixture
artifact. Earlier capture failure takes precedence over privacy rejection,
but either prohibits completion. Missing or undurable private capture is a
failed/incomplete attempt, never a successful output with an attachment omitted.

##### Controller integration self-review

| Interface pair | Produced / consumed | Ruling |
| --- | --- | --- |
| Task A / privacy | Standalone writer hooks / parent-only private capture | Parent writes occur outside the diagnostic child; A remains independently testable. |
| Privacy / Task B | Strict safe intent + process custody / optional child allowance | B adds its optional allowance to safe intent-v2 and cannot accept plaintext-v1 for new launches. |
| Privacy / Task C | Split parent category peaks / held child allowance | Replace older P formula with this explicit private/public split; C and inherited custody integration remain open. |
| Privacy / cold readers | Forensic version parsing / current source-bound eligibility | Old bytes remain readable but cannot certify current execution. |
| Privacy / workflow | Explicit same-owner capability / fresh complete execution | Missing custody is an early documented restriction until inherited wiring is reviewed. |
| Scope / tests | Tiny bounded real captures and existing-ledger controls / genuine scientific run | No synthetic successful scientific report, provider proof or full gate is claimed. |

Self-review checked R2 privacy, unchanged scientific controls, current-source
authentication, cold lease/error semantics and the same global/category caps.
The remaining supervisor/FitGuard inheritance and actual full diagnostic are
explicitly open integrations, not omitted completion gates.
