# Task5 child writes A: closed adapter and tiny controls

Read this first; this is the Task A requirements extract from the approved
R2 plan, with controller rulings. Use Superpowers TDD, systematic debugging
for unexpected behavior and verification before commits. No subagents.
Current main branch only. One source/test writer.

Own new train/pilot_offline_writes.py, tests/pilot/test_pilot_offline_writes.py,
the minimal low-level boundary hook integration needed by Task A in
train/pilot_offline.py, and the new required source inventory entry in
train/pilot_evidence.py. The current parent-process changes were reviewed
and frozen at70aace8. Do not change parent capture/records/reuse semantics,
the scientific worker, schemas, algorithms, config, RNG or thresholds.

Task A implements standalone early admission plus publisher/stream controls.
Do not implement Task B's serialized production allowance/worker launch
handoff or Task C's fixture hold in this dispatch. The standalone installer
validates explicit root/limits; the later production path must bind these to
the authenticated parent intent. Tiny controls may call install_offline_writes,
then the existing install_offline_boundary, then bind_publishers before
consumer imports; resolve any minimal hook coordination against those stages.
The new source inventory entry belongs in this commit so source closure never
lags the new module.

Controller owns the plan/progress and all reservations. Before any tests,
project imports or child launch submit exact selectors, control sizes, file/
directory/temp counts and allocated peak. An admitted reservation is not
blanket command admission. Fresh retained prefixes only under
operational/scratch/task5-offline-writes-a/. All original SDD evidence outside
operational is immutable. Never write/remove/move it, use cleanup scripts,
touch provider credentials, run network/real workers/model/fit/replay or copy a
corpus. Actual child allowance installation is qualified by tiny controls,
not a fabricated successful diagnostic. No real make verify or fullrun.

Write the report only to operational/sdd/task5-offline-writes-a-report.md.
Include exact commands, RED/GREEN distinctions, bytes/names/dirs and any denied
native/cache path; meaningfully commit scoped source/tests. Main dispatches
independent reviewer. Do not expand into a quota framework or change scientific
semantics to make a writer fit.

**Goal:** Enforce an explicitly reserved, finite allowance for the existing one-update, 16-episode offline diagnostic before its known disk writes occur.

**Architecture:** Bind one child allowance into the existing parent intent. Install admission before Torch and publisher adapters before imported aliases; latch denial through bootstrap exit. Retain the existing parent ledger and FitGuard.

**Tech Stack:** Python 3.12, existing stdlib audit/descriptor boundary, existing durable publishers, pytest fresh-interpreter controls.

**Spec:** `docs/superpowers/specs/2026-09-17-phase-4-r2-archive-design.md`, the canonical Silent Cascade specification, and the recorded workload design in `operational/sdd/task5-offline-bound-design.md`. This is the next in-scope Task5 slice under standing approval, not a new phase or permission to run the full diagnostic. Begin only after the parent-process slice's independent review closes.

##### Constraints and ownership

- Keep the worker's actual update, all 16 evaluations, replay, reporting, device selection, global/native RNG behavior, controls, gates and zero foundation-model calls unchanged.
- Default `write_allowance=None` retains the existing process-fenced diagnostic without adding a child byte limit. Existing boundary-only callers retain their legacy behavior.
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

Add optional `write_allowance: dict[str, int] | None = None` to the existing strict intent; validate its exact three keys. Missing on genuine old intents means no child allowance; invalid present fields fail. Existing raw-byte intent SHA-256 binds limits/root/attempt/source/executable/bootstrap. Do not rewrite old records or scientific fields. Existing `nonzero_exit` result plus bounded stderr records denial; no new marker/record.

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

For parent reserve P, approved lengths are record/stdout/stderr/record (defaults 16/256/256/16 KiB). Conservative bound:

```python
sizes = (limits.record_bytes, limits.stdout_bytes,
         limits.stderr_bytes, limits.record_bytes)
P = sum(2 * rounded(n) + B for n in sizes)
P += 2 * rounded(max(sizes)) + B + (9 + 6) * B
# four retained parent names plus two conservative publication names; nine dirs
```

This over-reserves final/temp/link peak; a larger approved P is valid. Hold `H=allowance.allocated_bytes+P` plus names/directories before intent. Child cannot spend P. Bind one fresh measurement root.

FitGuard stores one held root in memory. Outer admissions count non-offline spool/cache allocation plus H; do not double-count live child bytes or let unrelated writes spend H. Still validate offline inventory. Route four parent writes to P/per-record limits. No second reservation/root/replenishment; keep full hold through failures for guard lifetime. H plus retained fit/cache/outer publication peaks must fit existing FIT_BYTES/FIT_NAMES/FIT_DIRECTORIES before launch. No envelope expansion or full-gate total follows.

##### Task A: Closed adapter and tiny writer controls

**Files:** new `pilot_offline_writes.py` and `test_pilot_offline_writes.py`, with its required source inventory entry in `pilot_evidence.py`; integrate low-level boundary hooks in `pilot_offline.py` only after the parent change is stable. Source closure must not lag the new module.

- [ ] **RED — limits/grammar/latch.** Pure validation plus fresh-interpreter tests: 1–4096-byte real publications, exact calculated capacity and one byte short; invalid types/keys, unknown names/dirs, episode16, UUID17, weights2, SVG1. Preseed only admitted tiny controls before child launch. No models. Assertions:

  ```python
  from silent_cascade.train.pilot_data import _publish_pilot_bytes
  _publish_pilot_bytes(root / 'step.json', b'x')
  # Construct exact and short limits from the formula and admitted inventory.
  assert (root / 'step.json').read_bytes() == b'x'
  with pytest.raises(OfflineWriteDenied):
      guard.admit_publication(root / 'unknown.json', 1, writer='pilot')
  with pytest.raises(OfflineWriteDenied):
      guard.assert_clear()
  assert not (root / 'unknown.json').exists()
  ```

  Exercise actual patched publishers, not only admission arithmetic. Success body:

  ```python
  from silent_cascade.train.pilot_data import _publish_pilot_bytes
  from silent_cascade.io import atomic_create_bytes
  _publish_pilot_bytes(root / 'step.json', b'x')
  atomic_create_bytes(root / 'weights.safetensors', b'w')
  guard.assert_clear()
  ```

- [ ] **RED — peaks.** Tiny existing target plus incoming bytes: denial leaves old bytes/no temp; create collision reserves incoming bytes; hardlinks count twice; unknown/second temp fails; lowered names/dirs deny before creation. One intentional denial per interpreter, then check later known publication fails without clearing latch.
- [ ] **Implement adapters.** Validate canonical root and explicit allowance in the standalone adapter; Task B additionally enforces the production bootstrapped-intent binding. Install stdlib hooks before Torch/consumers, preserving os.open dir-fd confinement. Cover builtins.open, io.open (Path.open), mkdir/link/rename/unlink/truncate and fd-opening paths. Preserve original publishers:

  ```python
  def guarded_pilot(path, payload):
      self.admit_publication(path, len(payload), writer='pilot')
      with self._pilot_operation(path, payload):
          return original_pilot(path, payload)

  def guarded_temp(path, data, mode):
      self.admit_publication(path, len(data), writer='atomic')
      with self._temp_preparation(path, data):
          temporary = original_temp(path, data, mode)
      self._retain_temp_token(temporary, path, len(data))
      return temporary
  ```

  Private `OfflineWrites` methods: `_pilot_operation` owns UUID temp/parent walk/final link; `_temp_preparation` owns one mkstemp candidate/write/fsync/chmod; `_retain_temp_token` preserves ownership after helper return through caller's link/replace/fsync/unlink. Each checks grammar, parent inode/device, fd and target; no blanket 'inside publisher' bypass.

  Unlink only owned temps, linked pending rows and font lock; no retained-final deletion. Own-temp/lock cleanup remains allowed after denial without writes/reset. Audit/adapter denials latch first code (`bytes`, `names`, `directories`, `path`, `writer`, `descriptor`, `authority`); bounded details, no payload dump.

- [ ] **Exact streams.** Rows: `xb`, bytes write/tell/flush/context/close/fsync. Font: `w` UTF-8, admit encoded chunks before writing, return character count, flush admitted chunks. Lock: `xb`/close, zero bytes. Deny/latch append/update modes, seek/truncate/writelines, buffer/raw/delegation/reopen; no `__getattr__` forwarding.

  Preserve actual evaluator `os.fsync(handle.fileno())` using an opaque non-integer row token. Patched os.fsync recognizes its active token and uses the private fd. Raw open/write/ftruncate/fdopen/FileIO/dup/mmap cannot consume tokens or expose underlying streams. Wrap escape entry points so token misuse latches rather than merely raising catchable TypeError. Ordinary known-fd fsync passes through. Preserve stdout/stderr capture and exact read-only Git batch-pipe capability.

- [ ] **RED/GREEN — streams/escapes.** Real rows append/tell/flush/fsync/link; exact/over UTF-8 chunks; genuine font_manager cache import/zero-byte lock/cleanup; tiny json_dump overflow followed by assert_clear even if an exception was caught. Test buffer/raw/fdopen/dup/write/truncate/mmap, relative/dir-fd escape, symlink and external hardlink. Synthetic JSON alone cannot qualify the real font writer.
- [ ] **GREEN/review/commit.** Separately admitted named primitives and scoped Ruff/diff; cache/native side effects must be covered or denied. Record commands/allocation/denials; commit reviewed source/tests. No scientific success claim.


##### Primitive admission and execution instructions

Use the existing closed Python `-B` launcher, disabled external pytest plugins/cacheprovider and explicit TMP/XDG/MPL/Hypothesis/archive roots under a fresh operational scratch prefix per RED/GREEN/controller stage. Select only the named new writer tests and named parent/legacy primitive tests; never select the entire offline test file if it contains the actual diagnostic. Example inner command after its containing launcher/environment has been admitted:

```text
.venv/bin/python -B -m pytest -p no:cacheprovider tests/pilot/test_pilot_offline_writes.py -q
```

Test helper `run_writer_control(code, root, allowance)` must use existing `capture_offline_process` with `OfflineProcessLimits(stdout_bytes=2048, stderr_bytes=2048, record_bytes=4096, timeout_seconds=30)`. Pass a closed `offline_environment(root)`; create the approved root before launch. It launches a tiny `-B -c` program that installs the real boundary with the supplied allowance and performs only the requested primitive. It is a test harness, not a new permitted nested production command. Bound helper source text to 4096 bytes and each captured output to 2048 bytes. Fresh interpreters isolate irreversible hooks and latches.

Use explicit per-control stop limits: ordinary tiny cases at most 64 KiB child allocation, font import case at most 256 KiB; use 1–4096-byte scientific payloads. If a control requires more, compute its exact names/directories/payload peak first and record separate admission. Account every child root, control body, pytest output, parent logs/records, temporary, cache and native cushion; multiply by the exact planned invocation count, including RED/GREEN/controller retained failures. These ceilings bound attempts; they do not certify font import success or fit an entire suite by themselves.

Current normal headroom is about 177 MiB and scratch headroom about 10 MiB; neither is permission to consume it. If the exact retained total of all planned stages exceeds the remaining category, stage fewer named controls or complete the already-approved accounting/archive prerequisite before execution. Do not discard failing roots or silently reclassify scratch. All full-run and `make verify` admission remains outside this subsection.


