# Task5 child-write launch binding (Task B) brief

Read this first: this is the approved Task B extract, not execution admission.
Do not start until the controller supplies the privacy-corrected, reviewed base.
Task A is complete at 07e9394. Do not redispatch or reimplement its adapter.
The privacy correction is separately reviewed before this task begins.

Use Superpowers TDD/debugging/verification and the current branch. No subagents.
Only one source/test writer. Keep original engineering custody outside
operational immutable. No providers, real scientific worker, model/fit/evaluation/
replay, checkout copy, full gate or make verify in primitive qualification.
No scientific algorithm, RNG, gates, source authentication or storage-cap changes.

## Exact task ownership

Own pilot_offline.py and pilot_offline_process.py launch/schema integration and
focused primitive tests in test_pilot_offline_process.py,
test_pilot_offline_writes.py and named cases of test_pilot_offline.py.
Verify the existing pilot_evidence.py package inventory entry added in Task A;
do not create another entry or restructure that verifier.
The standalone adapter and cold_authentication_fixture.py hold implementation
are not this task. Report a concrete interface defect to the controller rather
than expanding those neighboring tasks.

Preserve the privacy correction's process_custody parameter when adding
write_allowance to measure_pilot_offline. The abbreviated old signature in the
extract below does not remove custody. Keep safe intent-v2/result-v2 and
report-v2 bindings; version1 stays forensic-only. A missing child allowance is
not a missing parent-custody waiver.

The controller owns the one live operational reservation. Before each test/
project import/child launch, send exact selectors, child count, names/directories/
temporary peaks, logical/allocated bound, and a fresh operational scratch prefix.
Tiny isolated ledger controls are charged physically under that reservation;
never create a replacement production ledger or enlarge allowances.
Retain all failed attempts. No repeated or unadmitted test prefix.

Delivery report:
operational/sdd/task5-offline-writes-b-report.md beneath this plan workspace.
Record exact commands, failure-before-fix evidence, final results, source hashes,
retained allocation, limitations and scoped source/test commits. Do not stage
controller-owned documentation. Return a short status/commit/test/concern summary.
Controller performs fresh verification and the independent task review.

Carry-forward review item: task5-offline-privacy-review.md records missing
private-prefix publication controls for overflow/cancellation. Task B's combined
fence overlaps those paths; include bounded controls when wiring the child
allowance, without manufacturing a successful scientific report.

## Approved requirement extract

The general responsibility list and hold discussion describe interfaces with
A/C; only the Task B ownership above is dispatched. The historical parent P
formula remains explicitly non-authoritative, as stated in the approved text.
##### Constraints and ownership

- Keep the worker's actual update, all 16 evaluations, replay, reporting, device selection, global/native RNG behavior, controls, gates and zero foundation-model calls unchanged.
- Default `write_allowance=None` adds no child byte limit; it does not waive the privacy correction's parent custody requirement. Existing boundary-only callers retain their legacy behavior.
- Unknown writers fail; no descriptor/cache fallback or kernel-wide native-write claim.
- No default byte capacity or historical-size-derived success claim. Trajectory ceilings permit 16 times 128 MiB. Debug weights have 3,084,378 tensor bytes plus a header capped at 16 MiB.
- No neural worker, copied corpus, provider/network or `make verify` in primitive qualification. Full local verification/current-source gate and archival require separate admission.
- Original evidence outside `operational/` stays frozen. Use existing records/ledger and current branch; commit meaningful authorized changes. No new quota framework or archive schema.

**File responsibilities:**

- Create `src/silent_cascade/train/pilot_offline_writes.py`: stdlib-only initial imports; allowance, grammar, accounting, adapters.
- Modify `src/silent_cascade/train/pilot_offline.py`: launch handoff, early hooks, final latch checks after parent writer finishes.
- Modify `src/silent_cascade/train/pilot_offline_process.py`: optional strict intent field; retain bounded capture and legacy semantics.
- Modify `src/silent_cascade/train/pilot_evidence.py`: new `REQUIRED_PACKAGE_FILES` member; existing intent parser authenticates limits.
- Modify `tests/pilot/cold_authentication_fixture.py`: three aliases and one held offline reservation.
- Create `tests/pilot/test_pilot_offline_writes.py`; extend primitive controls in `tests/pilot/test_pilot_offline.py` and `tests/pilot/test_pilot_offline_process.py`.

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
    *, output_dir, process_limits=None,
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

def reserve_offline_attempt(self, output_dir, *, write_allowance, process_limits):
    ...  # new FitGuard method: hold once, never replenish
```

Use positive plain integers (`type(value) is int`); `allocated_bytes <= 2**63 - 1` bounds wire integers, not granted capacity. Enforce names <=102, directories <=9. Reject unknown keys and null/bool/float/coerced values inside a present allowance, and reject nonexact API dataclass types. The optional whole allowance may be absent or None for the legacy unbounded-child mode. Tests lower limits. Caller holds bytes/names/directories before measurement; this module grants none.

Add optional `write_allowance: dict[str, int] | None = None` to the privacy-safe strict intent-v2; validate its exact three keys. Missing on genuine old intents means no child allowance in forensic parsing only; invalid present fields fail. Existing raw-byte intent SHA-256 binds limits/root/attempt/source/executable/bootstrap. Do not rewrite old records or scientific fields. Child denial remains `nonzero_exit`, with nonempty stderr in private custody and its count/hash/disposition in result-v2, never public raw stderr. No additional marker/record beyond the privacy protocol.

Production limits come only from `validate_offline_launch(root, intent_digest)`, never overriding argv/env. `_PENDING_OFFLINE_LAUNCH` stays the exact root/digest pair; `_PROGRAM` hash binds bootstrap. Changed root/intent/limits/attempt/environment/bootstrap fails. Installed allowances cannot be replaced/enlarged/rebound or authorize another child. No-allowance parents retain the exact existing fenced launch.

##### Writer-derived path grammar

Paths are relative to canonical output root. Preserve no-follow descriptor confinement: reject lexical `..`, symlinks, foreign device, nonregular files, external aliases and unsupported dir-fd forms before mutation. Count every name. Final names authorize only their listed adapter, not raw writes.

Reject pre-existing multi-link inodes unless both names are the active owned temp/final or pending/final pair; a hardlink to outside-root data must not become a writable alias. Recheck pinned parent/fd identities on mutations, not only resolved strings.

| Child family | Exact allowed names / bounded grammar | Retained maximum | Adapter |
| --- | --- | ---: | --- |
| Root outputs | `executed-source.json`, `step.json`, `weights.safetensors`, `manifest.json`, `replay.json`, `offline.json` | 6 | pilot publisher for JSON except replay; durable-temp for weights/replay |
| Evaluation controls | `run/eval/primary/{identity.json,retention.json,rows.jsonl,index.json,metrics.json,DONE}` | 6 | durable-temp; rows only from pending link |
| Episode outputs | `run/eval/primary/episodes/{ordinal}.{suffix}`; ordinal exactly `00000` through `00015`; suffix exactly `neural.json`, `trajectory.json.gz`, `telemetry.json` | 48 | durable-temp |
| Shared evaluation weights | `run/eval/primary/crashes/weights-{sha256}.safetensors`, lowercase hex64, at most one distinct digest | 1 | durable-temp |
| Crash evidence | `run/eval/primary/crashes/{uuid}.{json,safetensors}`, lowercase hex32, at most 16 distinct UUIDs; plus exact `crashes/index.json` | 33 | durable-temp |
| Report | `report/trajectories-0.svg`, `report/tables.json`, `report/report.md`, `report/report-index.json` | 4 | pilot publisher |
| Font cache | `matplotlib-cache/fontlist-v3.11.0.json`, one name | 1 | narrow text stream |

Sum: **99 child retained names**. Design's 102 included three former parent names. Parent fence now owns four: `intent.json`, `process-result.json`, `stdout.txt`, `stderr.txt`, separately reserved. Child may read validated intent but never mutate/link parent files.

Grammar sources: `_worker`; eval `_run_one`, `write_evaluation`, `_crash_index`; `stage_neural_runtime`; `EventEngine._publish_failure`; `logging/crash_bundle.py`; report enumeration starts at zero.

Only **nine directories** including root: `.`, `run`, `run/eval`, `run/eval/primary`, `run/eval/primary/episodes`, `run/eval/primary/crashes`, `report`, `matplotlib-cache`, `cache`. XDG `cache` allows no files. TMPDIR=root grants no scratch files or `matplotlib-*` fallback. Preserve synthetic EEXIST for safe publishers' existing ancestor mkdir attempts outside root; do not mutate/latch these.

Additional simultaneous child names have exact ownership:

1. **One** atomic temp: `.pilot-[0-9a-f]{32}.tmp`, or io `.{target.name}.{eight_chars}.tmp` (CPython 3.12 tempfile lowercase/digit/underscore alphabet), in the admitted target's parent. Allow only the active publisher's actual candidate, not any regex match/second temp. Failed retained temps remain counted; denial forbids new publication.
2. The exact `run/eval/primary/.rows.pending.jsonl`; allow its final link only to the exact `rows.jsonl`. Both names count during coexistence.
3. The zero-byte `matplotlib-cache/fontlist-v3.11.0.json.matplotlib-lock`.

Peak **99 + 1 + 1 + 1 = 102** also constrains failed attempts. UUID pairs share 16 stems; UUID17, weights digest2, ordinal16 or SVG2 fails even with bytes remaining. Names are ceilings, not promised artifacts.

Installed `FontManager.__version__` is `3.11.0`, independent of locked distribution `3.11.1`. Its `json_dump` writes via `open(filename,'w')`; `cbook._lock_path` uses `Path.open('xb')` then unlink. Derive grammar without importing font_manager before admission. Changed cache protocol requires reviewed closed-name update. SVG uses BytesIO; gzip/safetensors return bytes.

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
##### Task B: Bind launch authority and preserve parent completion

**Consumes:** privacy-corrected parent's `OfflineProcessLimits`, strict safe intent/result, private process custody, `validate_offline_launch`, `_PENDING_OFFLINE_LAUNCH`, capture/result reader. **Produces:** optional allowance-bound worker; no-allowance mode retains safe parent requirements. Forensic old parsing is not new-launch authority.

- [ ] **RED — compatibility/authority.** Old missing field parses as no allowance. Invalid/enlarged/removed field or changed root fails original digest. Tiny bootstrap sentinels are process evidence, never fabricated scientific success.
- [ ] **Wire launch.** Validate type/root/hold contract, serialize asdict(allowance) before intent digest, retain six-element command/closed environment. Bootstrap consumes validated intent:

  ```python
  intent = validate_offline_launch(sys.argv[1], sys.argv[2])
  allowance = (None if intent.write_allowance is None
               else OfflineWriteAllowance(**intent.write_allowance))
  attempts, blocked = install_offline_boundary(
      workspace_root=sys.argv[1], write_allowance=allowance)
  _worker(sys.argv[1], attempts, blocked,
          process_intent_sha256=sys.argv[2])
  if _BOUNDARY_WRITES is not None:
      _BOUNDARY_WRITES.assert_clear()
  ```

  In the actual bootstrap use `import silent_cascade.train.pilot_offline as boundary` and `boundary._BOUNDARY_WRITES` for both accesses; the snippet shows the checks, not a stale `from ... import` binding. Require the installed root/allowance to equal the validated intent before `_worker` runs and again when it validates launch.

  `_BOUNDARY_WRITES` is a once-initialized module global; read it after installation, not via stale imported value. Install stdlib admission, import Torch/network hooks, then `bind_publishers()` imports io/pilot_data and patches helpers before worker consumers. Reject late consumer aliases bound to originals. pilot_data imports Torch-dependent evaluation types: initial admission must precede that import. io atomic aliases resolve patched `_durable_temp` dynamically.

  Also assert_clear immediately before worker offline.json publication. Bootstrap postcheck catches swallowed denial/control returns. Completion still requires zero exit, drained pipes, bound scientific report and clear latch. Never substitute a report after denial.
- [ ] **GREEN — combined fence.** Caught denial then assert_clear exits nonzero; invalid preseeded offline.json cannot override failure/overflow. Simultaneous pipe overflow and timeout terminate/join child; P retains bounded logs/result despite child exhaustion. Interrupted missing result is incomplete. Changed allowance fails reuse/collection authentication. Named legacy retained.txt/descriptor controls still work with no allowance.
- [ ] **Inventory/review/commit.** Verify the required source path added with Task A; parser/bootstrap and scoped Ruff/diff; retain failed roots. No worker success claim.

##### Primitive admission and execution instructions

Use the existing closed Python `-B` launcher, disabled external pytest plugins/cacheprovider and explicit TMP/XDG/MPL/Hypothesis/archive roots under a fresh operational scratch prefix per RED/GREEN/controller stage. Select only the named new writer tests and named parent/legacy primitive tests; never select the entire offline test file if it contains the actual diagnostic. Example inner command after its containing launcher/environment has been admitted:

```text
.venv/bin/python -B -m pytest -p no:cacheprovider tests/pilot/test_pilot_offline_writes.py -q
```

Test helper `run_writer_control(code, root, allowance)` must use existing `capture_offline_process` with `OfflineProcessLimits(stdout_bytes=2048, stderr_bytes=2048, record_bytes=4096, timeout_seconds=30)`. Pass a closed `offline_environment(root)`; create the approved root before launch. It launches a tiny `-B -c` program that installs the real boundary with the supplied allowance and performs only the requested primitive. It is a test harness, not a new permitted nested production command. Bound helper source text to 4096 bytes and each captured output to 2048 bytes. Fresh interpreters isolate irreversible hooks and latches.

Use explicit per-control stop limits: ordinary tiny cases at most 64 KiB child allocation, font import case at most 256 KiB; use 1–4096-byte scientific payloads. If a control requires more, compute its exact names/directories/payload peak first and record separate admission. Account every child root, control body, pytest output, parent logs/records, temporary, cache and native cushion; multiply by the exact planned invocation count, including RED/GREEN/controller retained failures. These ceilings bound attempts; they do not certify font import success or fit an entire suite by themselves.

Current normal headroom is about 177 MiB and scratch headroom about 10 MiB; neither is permission to consume it. If the exact retained total of all planned stages exceeds the remaining category, stage fewer named controls or complete the already-approved accounting/archive prerequisite before execution. Do not discard failing roots or silently reclassify scratch. All full-run and `make verify` admission remains outside this subsection.
