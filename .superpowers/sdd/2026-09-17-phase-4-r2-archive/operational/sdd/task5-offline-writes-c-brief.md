# Task5 fixture child/parent hold (Task C) brief

Read this first: the approved Task C requirements below are not test execution
admission. Wait for the controller's reviewed Task B base. Task A and privacy
are complete; do not reimplement their primitives. Work on current main with
Superpowers TDD/debugging/verification, apply_patch and scoped commits. No
subagents. Controller owns ledger/docs/reservations; one source/test writer.

Own tests/pilot/cold_authentication_fixture.py and focused hold/alias tests in
tests/pilot/test_pilot_offline_writes.py. Also own only the narrow public-spool
separation correction in src/silent_cascade/train/pilot_offline_process.py and
its focused custody tests in tests/pilot/test_pilot_offline_process.py. Do not
change child grammar/adapters, launch/bootstrap, scientific code, source
inventories, archive/session protocols, policy limits or inherited authority.

Use existing custody.category_peaks; preserve process_custody in every measured
launch. The old abbreviated measure signature below never removes that argument.
Private preparation already has a global owner; the fixture's hold is not a new
reservation or permission to relabel bytes. Include full H in existing FIT_BYTES
64MiB/FIT_NAMES256/FIT_DIRECTORIES48, conservatively including private P even
though its real category remains logs. No default child capacity. Do not invoke
create_checkout, run_fixture, a real diagnostic, model, fit, eval/replay, provider,
full gate or make verify. Private writes remain in the existing parent custody
code, not a new fixture publisher or export member.

Before any tests/project imports/children, send exact selectors, child count,
payload/name/directory/temporary peaks, logical and allocated bound, time/output
limits and a fresh operational scratch prefix. Retain all attempts. No cleanup,
new production ledger, overlapping owner or admission from observed headroom.
Tiny isolated ledgers are test-only and their physical cost counts in the single
controller reservation. Source reading and focused test writing may precede it.

Full report:
.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/sdd/task5-offline-writes-c-report.md
Record exact RED/GREEN commands and outputs, actual allocation, hashes, source
commits, self-review and limits. Do not stage controller docs. Return only short
status/commits/test summary/concerns; controller verifies/reviews independently.

## Approved extract

##### Closed interfaces

```python
@dataclass(frozen=True)
class OfflineWriteAllowance:
    allocated_bytes: int                  # REQUIRED; no default capacity
    file_names: int = 102
    directories: int = 9

class OfflineWriteDenied(RuntimeError):
    pass

def measure_pilot_offline(
    *, output_dir, process_limits=None, process_custody=None,
    write_allowance: OfflineWriteAllowance | None = None,
): ...

def install_offline_boundary(
    *, workspace_root=None,
    write_allowance: OfflineWriteAllowance | None = None,
): ...  # preserve existing (attempts, blocked) return

def install_offline_writes(root: Path, allowance: OfflineWriteAllowance):
    ...  # return OfflineWrites; irreversible, one root per child

class OfflineWrites:
    def bind_publishers(self) -> None: ...
    def assert_clear(self) -> None: ...
    def admit_publication(self, path: Path, size: int, *, writer: str) -> None: ...

def reserve_offline_attempt(
    self, output_dir, *, write_allowance, process_limits, process_custody
):
    ...  # new FitGuard method: hold once, never replenish
```

Use positive plain integers (`type(value) is int`); `allocated_bytes <= 2**63 - 1` bounds wire integers, not granted capacity. Enforce names <=102, directories <=9. Reject unknown keys and null/bool/float/coerced values inside a present allowance, and reject nonexact API dataclass types. The optional whole allowance may be absent or None for the legacy unbounded-child mode. Tests lower limits. Caller holds bytes/names/directories before measurement; this module grants none.

Add optional `write_allowance: dict[str, int] | None = None` to the privacy-safe strict intent-v2; validate its exact three keys. Missing on genuine old intents means no child allowance in forensic parsing only; invalid present fields fail. Existing raw-byte intent SHA-256 binds limits/root/attempt/source/executable/bootstrap. Do not rewrite old records or scientific fields. Child denial remains `nonzero_exit`, with nonempty stderr in private custody and its count/hash/disposition in result-v2, never public raw stderr. No additional marker/record beyond the privacy protocol.

Production limits come only from `validate_offline_launch(root, intent_digest)`, never overriding argv/env. `_PENDING_OFFLINE_LAUNCH` stays the exact root/digest pair; `_PROGRAM` hash binds bootstrap. Changed root/intent/limits/attempt/environment/bootstrap fails. Installed allowances cannot be replaced/enlarged/rebound or authorize another child. No-allowance parents retain the exact existing fenced launch.

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
Public publication uses random sibling `.pilot-[0-9a-f]{32}.tmp` names.
Before custody creation, reject a public final-name binding or any binding
matching that sibling temporary namespace whose category differs from the
public output directory's category. Same-category bindings are harmless;
unrelated child subtrees retain their existing mappings. This narrow
restriction makes the public final/temporary pair's existing category forecast
valid without inventing a dynamic temporary-routing service. Recheck it at
prelaunch/publication boundaries with the existing custody validation. The
private route and its fixed members must still count as logs.
Private bytes charge existing logs; public bytes their existing category.
U counts existing bytes once; remaining reservation covers P plus separately
reserved child peaks within existing global/physical headroom checks.
Child offline.json/source/worker files remain outside P. No double reservation
or new ledger. Control harness may use8192 without changing production B.

##### Pre-write reservation calculation

Use FitGuard's cushion: `B=4096`, `rounded(n)=((n+B-1)//B)*B`; reject unsupported filesystem allocation unit. Charge `st_blocks*512` for each child file/directory name, including hardlink duplicates. At production child startup, authenticate and exclude only the already-created `intent.json`. The other three parent-owned final names must be absent throughout child execution, are never writable by the child, and appearing unexpectedly is an authority failure. Their later parent publication occurs after the child has exited. Standalone primitive roots may omit intent; they grant no production launch authority. Child pays root/directories; parent may conservatively reserve them too.

Let `U,F,D` be current child bytes/file names/directories, `M` missing permitted parents, `s` full proposed payload length, `N` additional name allowance:

```python
projected = U + 2 * rounded(s) + B + (D + M + F + N) * B
require(F + N <= allowance.file_names)
require(D + M <= allowance.directories)
require(projected <= allowance.allocated_bytes)
```

`N=2` for durable temp/final publication, including replacement; `N=1` for stream create/growth or rows final link. Directory creation: `s=0,N=0,M=1`. Reserve parents/full incoming bytes before mkdir/open; existing destinations remain in U for replacement, idempotence and collisions. Stream `s=current_encoded_length+len(next_encoded_chunk)`; final rows-link `s=full pending size`, pending already in U. Never subtract old target before replacement or count only logical length.

Operation tokens span admission through publication/cleanup. OS hooks verify exact target/temp/fd/bytes/link without charging a second operation reserve. Pending rows coexist with episode publications: flush admitted chunks or include buffered high-water allocation in U. At most one atomic token, one row stream, one font stream/lock. Refresh inventory after operations; scans cannot retroactively admit writes.

The following was the parent-only four-public-file reserve before the privacy
correction. It remains historical design rationale, not current launch
authority. Current P must use the preceding privacy subsection's exact
private/public category split and retained receipt/raw-log peaks; Task C must
not use this older expression to authorize a launch.

Former approved lengths were record/stdout/stderr/record (defaults 16/256/256/16 KiB):

```python
sizes = (limits.record_bytes, limits.stdout_bytes,
         limits.stderr_bytes, limits.record_bytes)
P = sum(2 * rounded(n) + B for n in sizes)
P += 2 * rounded(max(sizes)) + B + (9 + 6) * B
# four retained parent names plus two conservative publication names; nine dirs
```

Using the privacy-corrected P, hold `H=allowance.allocated_bytes+P` plus
names/directories before intent, preserving each category's share. Child cannot
spend P. Bind one fresh measurement root and its separately pinned private
custody. A larger explicitly admitted P is valid; unused global headroom is not
an admission.

FitGuard stores one held root in memory. Outer admissions count non-offline spool/cache allocation plus H; do not double-count live child bytes or let unrelated writes spend H. Still validate offline inventory. Route four parent writes to P/per-record limits. No second reservation/root/replenishment; keep full hold through failures for guard lifetime. H plus retained fit/cache/outer publication peaks must fit existing FIT_BYTES/FIT_NAMES/FIT_DIRECTORIES before launch. No envelope expansion or full-gate total follows.

Task C consumes the actual `process_custody`, validated with
`require_offline_process_custody(..., phase=0)`, to obtain its immutable
`category_peaks`. It must not guess the private path or recompute the historical
four-file P. The caller first owns the global admission and prepares private
custody, then establishes this fixture hold before any public intent or child
launch. `H = write_allowance.allocated_bytes + sum(peak for _, peak in custody.category_peaks)` is
conservative inside the unchanged fixture envelope, including private bytes
whose real category remains logs. The public spool and private logs shares must
also fit the live underlying category admission; this fixture hold grants none.
If child output paths have a different effective category from the public run,
reject this fixture layout rather than allocating the whole child allowance to
the wrong category. No binding mutation, child-inherited authority or raw-private
publication hook is added by the fixture.

##### Task C: Hold the exact child allowance in the existing fixture

**Files:** `tests/pilot/cold_authentication_fixture.py`, focused tests in `test_pilot_offline_writes.py`; one path-contract correction in `pilot_offline_process.py` and focused custody tests in `test_pilot_offline_process.py`.

**Post-privacy interface ruling:** `_custody_paths` currently rejects the public
run itself when it overlaps any spool binding. The privacy requirement concerns
PRIVATE capture membership; this public restriction also rejects the existing
supervisor's mandatory spool run. Correct that distinction in this task, after
Task B freezes the shared file. Keep the private route outside the run, controls,
all spool/cache bindings and frozen engineering custody; keep public run outside
controls/logs/cache. No binding changes under a live reservation. Do not delete
the original unadmitted-spool negative case: it still lacks a spool allowance.

- [ ] **RED/GREEN — admitted public spool.** Use a tiny real isolated ledger,
  bind a fresh public run as spool before its one reservation, and reserve the
  existing privacy-derived public P in spool and private P in logs. Preparation
  and revalidation must succeed, with no raw capture inside the run. One byte
  short in the public spool share fails before custody creation. Private route
  overlapping any spool/cache binding, public cache/control/log overlap, changed
  category mappings and no admission still fail. No real diagnostic or provider.
  The correction changes only which side of the separation rule is checked;
  it does not weaken category accounting or add inherited ownership.

- [ ] **RED — aliases.** Tiny undersized-guard tests for bound publishers in pilot_checks, pilot_evidence and report.pilot; no numerical execution. Patch those three aliases in installed() with existing wrapper.
- [ ] **RED — hold.** Tiny retained outer file: exact H fits, one byte short fails before intent/launch (tripwire). After reserve, unrelated output cannot spend H; failed child cannot rebind/replenish; live child not double-counted; logs/result can use P.
- [ ] **Implement hold.** `FitGuard.reserve_offline_attempt` validates canonical fresh spool root, no previous hold and dataclasses; checks H/names/dirs plus retained usage/missing parents; stores one hold. Extend admit to route exact parent files and preserve H on other writes. Future caller explicitly passes allowance; no FIT_BYTES increase/default capacity/full-gate fit claim or substituted report.
  Validate `process_custody` and matching limits/output before fixing H. Retain
  the public/private category split from its `category_peaks`; check the child's
  full allowance plus the public share in the run's effective category, and
  each other parent share separately. Do not count already-created private
  directories twice against remaining category capacity: compare their retained
  allocation plus the remaining phase peak to the original category hold.
- [ ] **GREEN/review/commit.** Primitive hold/alias tests, source inventory and scoped static checks. Record outstanding actual worker/full-gate admission. Obtain independent task review after controller verification; the full-run admission remains separate.
