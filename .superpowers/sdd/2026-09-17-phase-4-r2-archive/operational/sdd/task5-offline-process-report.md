# Task5 offline process implementation report

Status: implemented and source frozen for controller verification and independent review.
This completes the bounded parent-process slice only, not Task5 as a whole.

Source commits: `32452b4` (bounded capture, existing-attempt fence, current source inventory)
and `70aace8` (launch/completion binding, semantic readers, reuse and control tests).
Base: `47395b7`. Current branch: main. No subagents dispatched.

## Scope and interfaces

Added private `train/pilot_offline_process.py`: strict positive plain-integer
`OfflineProcessLimits`, bounded concurrent stdout/stderr capture, strict immutable
intent/result schemas, portable `read_offline_process_outcome`, and an execution
reuse fence. Defaults are stdout/stderr262144B each, record16384B, timeout180s;
enlarged or unknown overrides and bool/nonintegral limits fail.

Capture holds only bounded prefixes plus a4096B read chunk. Overflow, timeout,
cancellation and launch failure retain the actual child identity/return code when
available. Termination has a0.25s grace before kill; the exact child is waited.
Successful primitive controls are process evidence only, never scientific reports.
Tests verify reaping with `os.waitpid` raising ChildProcessError.

`pilot_offline.py` now publishes immutable intent before Popen and create-only
terminal result after durable logs and report validation. The v2 report binds the
intent hash, source revision/digest, executable and exact bootstrap. The nested
Popen audit checks the active parent tuple, root, environment and program; worker
startup validates before the boundary imports Torch and validates again before its
v2 report. Parent rereads intent before accepting a report. Publication failure,
failed capture or malformed report cannot publish a reusable completed outcome.
Cancellation before Popen has no invented pid or return code. A log-publication
failure preserves an earlier launch failure, avoiding an impossible terminal record.

`pilot_checks.py` fences a pre-existing attempt before authentication/evaluation/
continuation/numerics. Collection applies the same fence. Collection and the
full-gate verifier require the v2 parser variant; full-gate enforcement follows
actual source/config/workload authentication. Standalone artifact verification
preserves genuine v1 interpretation and invokes the new process reader when the
v2 marker or process record selects the protocol. Present contradictory markers,
failed outcomes, changed bindings, corrupted logs and symlinks cannot downgrade
to historical behavior or become mere unavailability.

The cold reader exhausts inventory before payload reads, validates independently
available members despite missing required members, and releases each payload
lease before the next. Recorded launch-machine paths are not compared with the
current verifier's machine; the context's logical session root is checked.
Broken root links fail reuse; broken member links reach the bounded no-follow
reader and become a bounded ValueError with the original OSError as its cause.
Lease-acquisition errors are not caught by that adapter.

New source file is included in the explicit current package inventory. No
scientific algorithm, RNG action, threshold, foundation-model counter or
historical evidence bytes were changed. Existing scientific worker execution
remains unchanged apart from the new process evidence binding.

## Verification and TDD evidence

All executions used fresh retained prefixes below
`operational/scratch/task5-offline-process/`, closed Python -B launchers,
PYTHONDONTWRITEBYTECODE=1, plugins/cacheprovider disabled, explicit TMP/XDG/MPL/
Hypothesis/archive paths, and --noconftest. The latter avoids pilot fixture
imports, not every transitive Torch module import. No model, fit, evaluation,
replay or provider execution occurred. Tiny invalid-sentinel children replaced
only the scientific Popen command in parent-failure controls.

| Stage | Result | pytest / wall seconds | Logical B | Allocated B | Files / links / dirs |
| --- | --- | --- | ---: | ---: | --- |
| red1 | 2 failed | 1.43 / 1.982 | 1371 | 8192 | 3 / 1 / 8 |
| red2 | 17 failed, 1 deselected | 0.07 / 0.291 | 7777 | 8192 | 2 / 0 / 6 |
| green1 | 18 passed | 1.26 / 1.501 | 101 | 8192 | 3 / 1 / 8 |
| red3 | 16 failed, 18 deselected | 2.07 / 2.725 | 9752 | 16384 | 3 / 6 / 16 |
| green2 | 16 passed, 18 deselected | 2.10 / 2.725 | 7999 | 81920 | 27 / 6 / 40 |
| red4 | 9 failed, 37 deselected | 1.57 / 2.150 | 6262 | 24576 | 7 / 3 / 18 |
| green3 | 9 passed, 37 deselected | 1.57 / 2.150 | 7844 | 57344 | 18 / 3 / 18 |
| red5 | 4 failed, 3 passed, 46 deselected | 2.24 / 2.880 | 7905 | 49152 | 17 / 4 / 24 |
| final1 | 58 passed | 8.25 / 8.792 | 24903 | 233472 | 74 / 20 / 90 |
| red6 | 2 failed, 54 deselected | 0.24 / 0.482 | 1143 | 4096 | 2 / 4 / 12 |
| green4 | 12 passed, 1 failed, 43 deselected | 1.53 / 2.120 | 16051 | 139264 | 46 / 12 / 67 |
| green5 | 2 passed, 54 deselected | 1.52 / 2.069 | 113 | 4096 | 2 / 4 / 12 |

All stderr.log files are empty. Retained total:634880 allocated bytes,
91221 regular-file logical bytes. No stage was removed or reused.
Native allocated measurement sums lstat.st_blocks*512 for directories, files
and links; conservative prospective name/directory cushions remain separate.

Actual behavioral REDs:

- red1: existing intent reached Git-resolution tripwire; missing capture module
  was separately recorded as new-API RED.
- red2:17 absent capture API controls, explicitly new-API RED, not measured old
  behavior. Limits, independent/simultaneous overflow, zero/nonzero exits,
  timeout, cancellation, failed launch and joined children then passed green1.
- red3: reuse reached authentication before its fence.15 other failures were
  missing new record/parser/reader APIs.
- red4: invalid-sentinel exit0 left no terminal record; launch denial leaked
  OSError.7 bootstrap validator cases were missing-API RED.
- red5: collection reached authentication, pre-Popen cancellation dereferenced
  None, combined launch/log failure left no terminal after schema rejection;
  current-process parser keyword was absent.3 existing controls already passed.
- red6: both broken-path cases incorrectly returned without raising.
- green4: broken member correctly failed no-follow open, but the adapter leaked
  raw ELOOP; this was not unavailability or skipped corruption detection.
  The narrow bounded-read OSError normalization passed green5.

FINAL1 ran58 tests together before the final symlink correction. GREEN4/GREEN5
cover the changed reader/fence. The implementer does not claim all60 final cases
ran together; that frozen-source repeat belongs to the controller.

Exact stage selectors after the shared launcher flags:

- red1: the two nodes `test_existing_intent_blocks_launch` and
  `test_capture_bounds_both_pipes`.
- red2: file plus `-k 'not existing_intent'` at the then18-case collection.
- green1: entire then18-case process file.
- red3/green2: `-k 'reuse_rejects or terminal_record or marked_report or cold_process or failed_process or process_reader'`
  at the then34-case collection (16 selected).
- red4/green3: `-k 'measurement_persists or measurement_records or bootstrap_rejects'`
  at the then46-case collection (9 selected).
- red5: `-k 'collection_fences or cancellation_before_launch or log_publication_failure or uses_only_active or advertised_process or current_process_parser'`
  at the then53-case collection (7 selected).
- final1: whole then54-case process file plus the four exact legacy nodes below.
- red6/green5: `-k 'broken_offline_root or broken_process_member'`
  at56-case collection (2 selected).
- green4: `-k 'broken_offline_root or broken_process_member or cold_process or failed_process or process_reader or advertised_process'`
  at56-case collection (13 selected).

Fresh scoped Ruff lint, Ruff format --check and git diff --check all passed
on the final five source/test files before commit.

## Accounting exception and bounds

Controller sole reservation holder23159, token
`deff627ffe458e5573ed87bf7504534a`, carried4MiB task scratch;
this module did not create or replenish a ledger/reservation. Stage allowances
were obtained independently before every execution.

RED3's directory forecast failed:16 actual directories exceeded the declared12,
because pytest created tmp_path fixtures before each body reached its missing API.
Its128KiB allocation bound was not exceeded. This was reported immediately and
recorded by the controller as a failed forecast, never retroactively approved.
All later forecasts included per-parameter fixture directories and transient names.

Stage allocated allowances: red1/red2/green1/red3 128KiB;
green2/red4/green3/red5 1MiB; final1 3MiB;
red6/green5 256KiB; green4 1.5MiB.
Each launcher capped stdout/stderr separately at16KiB before retained writes.
Cold controls assert each JSON<=4096B and per-case payload<=8194B before writes.
The fixed sentinel child writes only invalid offline.json7B and stdout2B.
The nested negative audit control has256B stdout/4096B stderr and10s timeout.
Counted-workspace legacy control writes two6B files. Other selected legacy
controls perform module imports and closed read-only Git commands.

Final frozen collection is56 new process cases +4 exact legacy cases =60.
A fresh controller prefix fits the existing3MiB envelope:
<256KiB logical, <=144 file/link/temporary names and <=128 directories.
The exact directory derivation is6 stage/environment +1 pytest +32 fixture
directories +44 cold-control directories +4 reuse/collection directories +
4 parent-output directories +1 legacy allowed directory +3 new symlink-inner
directories =95. The conservative allowance is
8192*(144+128)+256KiB <3MiB. Existing retained task bytes must still be counted
before controller admission.

## Reproducible controller launcher

Run from the repository root only after controller admission. The code below
matches FINAL1's launcher and selectors, with a fresh `main1` prefix and a
portable expression for the same PYTHONPATH. The file now contains56 tests, so
the same selectors collect60. The wrapper prints pytest's actual exit field;
the wrapper's own successful exit is not the pytest verdict.

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd(); stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-process/main1'
assert not stage.exists(); assert 256*1024+8192*(144+128)<=3*1048576
stage.mkdir(parents=True)
for name in ('tmp','xdg','mpl','hypothesis','archive'): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/'tmp'),TMP=str(stage/'tmp'),TEMP=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'xdg'),MPLCONFIGDIR=str(stage/'mpl'),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'),OMP_NUM_THREADS='1')
args=[str(root/'.venv/bin/python'),'-B','-m','pytest','-p','no:cacheprovider','--noconftest','-q','--tb=short','--basetemp',str(stage/'pytest'),'tests/pilot/test_pilot_offline_process.py','tests/pilot/test_pilot_offline.py::test_offline_git_pin_requires_no_lazy_fetch_version','tests/pilot/test_pilot_offline.py::test_fresh_offline_boundary_denies_cloud_and_subprocess_escape','tests/pilot/test_pilot_offline.py::test_offline_child_cannot_write_outside_counted_workspace','tests/pilot/test_pilot_offline.py::test_git_provenance_denies_helper_configuration_and_keeps_closed_reads']
start=time.monotonic(); p=subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE); sel=selectors.DefaultSelector(); buffers={}
for name,s in [('stdout',p.stdout),('stderr',p.stderr)]: sel.register(s,selectors.EVENT_READ,name); buffers[name]=bytearray()
try:
 while sel.get_map():
  for key,_ in sel.select(1):
   chunk=os.read(key.fileobj.fileno(),4096)
   if not chunk: sel.unregister(key.fileobj); continue
   if len(buffers[key.data])+len(chunk)>16384: p.kill(); raise RuntimeError('output bound')
   buffers[key.data].extend(chunk)
  if time.monotonic()-start>30: p.kill(); raise RuntimeError('timeout')
finally:
 p.wait(); sel.close()
 for name,payload in buffers.items():
  with (stage/(name+'.log')).open('xb') as f: f.write(payload)
for name,payload in buffers.items(): print(name+':\n'+payload.decode())
items=list(stage.rglob('*')); files=[p for p in items if p.is_file() and not p.is_symlink()]
print(dict(exit=p.returncode,seconds=round(time.monotonic()-start,3),logical=sum(p.stat().st_size for p in files),allocated=sum(p.lstat().st_blocks*512 for p in [stage,*items]),files=len(files),dirs=sum(p.is_dir() and not p.is_symlink() for p in [stage,*items])))
PY
```

Forbidden legacy selectors, deliberately never collected:

- `test_only_exact_nested_offline_diagnostic_is_admitted` (ends in the real worker).
- `test_offline_probe_runs_training_evaluation_replay_and_artifact_report`.

The historical v1 control reads only the existing837B offline.json under the
frozen Task4 smoke run. It does not copy, regenerate, relabel or mutate evidence.
The fresh nested audit child tests changed root, digest, environment, bootstrap
program and absent pending authority. No nested diagnostic is executed.

## Remaining limits and handoff

Real v2 worker proof, closed child publication/cache/name allowance, the complete
source-bound diagnostic/full-gate envelope and full local `make verify` remain
pending separate admission. This slice supplies no child filesystem quota,
kernel-wide syscall enforcement, successful training metrics or current-source
full-gate result. Outer cold gate/collection routing and recovery remain as
scoped by the approved plan. Independent task review and controller all60 repeat
are pending; no hosted checks, workflows or providers were used.

Owned files: the new process module and test file, existing pilot_offline.py,
pilot_checks.py and pilot_evidence.py, plus this report. Controller plan/progress/
preflights were not edited by this implementer. Retained failures remain available
for review.

