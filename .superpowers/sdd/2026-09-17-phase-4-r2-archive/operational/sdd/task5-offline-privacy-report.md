# Task5 offline privacy correction report

Status: DONE_WITH_CONCERNS; source/tests frozen for controller repeat and independent review.
Source commit71eabfeb062f0525dc739491f893f6458c78659f:
fix(pilot): keep offline process paths and captures private.
Base89e987a. Controller owns reservation61737. No subagents.
Scope is the integrated task5-offline-privacy-brief.md, not TaskB/C or inherited
supervisor/FitGuard routing. No genuine worker/model/fit/evaluation/replay/provider,
source/corpus copy, full gate or make verify.

## Current evidence

| Stage | Result | Seconds pytest/wall | Logical B | Allocated B | Files/links/dirs |
| --- | --- | --- | ---: | ---: | --- |
| red1 |3 failed|1.54/2.141|4633|20480|7/3/13|
| red2 |12 failed|2.66/3.265|14170|49152|20/4/48|
| green1 |15 passed|2.68/3.249|6895|65536|28/7/63|
| final1 |98 passed|12.40/13.135|55012|507904|183/36/306|

stderr empty in all stages; retained643072 allocated bytes/80710 logical bytes. RED1 also created the task parent
directory (one additional directory,0 allocated bytes). All prefixes retained.
RED1: actual missing-custody Git tripwire and public raw sentinel exposure;
separate new intent-v2 schema RED. Hash-only intent implementation began AFTER
RED1. RED2: nine missing-custody-API cases, old result schema rejecting v2,
and two actual checks/workflow expensive-boundary tripwires. No false RED from
renaming the old result symbol: the original public alias remained during RED2.
GREEN1 exercised a15B sentinel, private0600 retention, public stdout absence,
empty public stderr and failed result-v2, with original child joined.

Scoped Ruff initially found unused uuid plus five nested-with style findings.
The mechanical edit briefly misindented the fixture after a decorator; Ruff
caught the syntax error before any test/import. Corrected the fixture only.
After FINAL1, fresh scoped Ruff check and format --check plus git diff --check
passed; no scratch output from static
checks. Seven owned Python files only; controller docs untouched.

## FINAL1 admitted envelope and actual accounting

Frozen test file94 cases + four exact legacy nodes in launcher =98.
24 isolated tiny-ledger fixtures,19 cold process fixtures,26 other tmp_path
fixtures =69 fixture dirs. Directory estimate:7 stage/env/pytest +69 fixtures
+122 ledger inner dirs +76 cold inner dirs +8 other inner dirs =282;
cap384 includes102-directory cushion. Actual306 includes24 ledger locks
subdirectories omitted from the core282 estimate; all stayed within the cap. File estimate183 regular names plus
<=72 fixture/broken-path links, with320 total names covering65 extra/current
temporary names. Retained production temporary cleanup is its normal publisher
operation; no stage/evidence cleanup is performed.

Ledger pages capped2048B; measured-process intent/result/receipt capped1024B;
cold prewrite asserts intent<=1536/result<=1024 and logs<=2B (corruption3B).
Other controlled records remain<=4096B. Child invalid reports7B and sentinels15B.
256KiB logical cushion covers these payloads, ledger replacement peaks and two
16KiB launcher logs. Native measured bytes use st_blocks*512. Prospective bound
uses verified f_frsize<=4096 and4096 per-name/directory rounding, not8192:
256KiB+4096*(320+384)=3MiB. Prior135168B+3MiB fits existing held4MiB.

17 Python control children:7 capture primitives,2 legacy measurement/publication
children,1 nested negative audit,1 private sentinel,3 private failure children,
3 legacy boundary children. Failed-launch/cancellation-before-Popen attempts
are separate controls. Timeout primitive1s; private timeout2s; nested negative
audit256/4096B and10s. The three UNCHANGED legacy boundary tests use their
original subprocess.run timeout30s, not the new bounded capture API; their
fixed scripts print at most512B expected stdout and no stderr, and write only
two6B allowed files. Allow <=32 trusted read-only Git subprocesses for pin and
existing provenance forms, no materialization or external helpers.
Outer qualification timeout60s; all launch logs <=16KiB. Read exactly the
existing837B historical v1 report, unchanged; no new scientific success report.

Forbidden selectors:
- test_only_exact_nested_offline_diagnostic_is_admitted
- test_offline_probe_runs_training_evaluation_replay_and_artifact_report

## Reproducible final launcher

Run only after controller admission. For independent repeat replace only
the fresh final1 prefix with main1. Wrapper exit is not pytest exit; inspect
the printed exit field. AST collection forecast is checked before writes.

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import ast, math, os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-privacy/final1'
assert not stage.exists()
assert os.statvfs(root).f_frsize <= 4096
assert 256*1024+4096*(320+384)==3*1048576
tree=ast.parse((root/'tests/pilot/test_pilot_offline_process.py').read_text())
cases=0
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name.startswith('test_'):
  parameters=[d for d in node.decorator_list if isinstance(d,ast.Call) and isinstance(d.func,ast.Attribute) and d.func.attr=='parametrize']
  assert all(isinstance(d.args[1],(ast.List,ast.Tuple)) for d in parameters)
  dimensions=[len(d.args[1].elts) for d in parameters]
  cases+=math.prod(dimensions)
assert cases==94, cases
historical=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-4/cover-pilot-final/test_real_smoke_workflow_reuse0/repo/runs/smoke/final/offline/offline.json'
assert historical.stat().st_size==837
stage.mkdir(parents=True)
for name in ('tmp','xdg','mpl','hypothesis','archive'): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/'tmp'),TMP=str(stage/'tmp'),TEMP=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'xdg'),XDG_CONFIG_HOME=str(stage/'xdg'),XDG_DATA_HOME=str(stage/'xdg'),PYTHONHASHSEED='0',SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH='1',MPLCONFIGDIR=str(stage/'mpl'),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'),OMP_NUM_THREADS='1')
args=[str(root/'.venv/bin/python'),'-B','-m','pytest','-p','no:cacheprovider','--noconftest','-q','--tb=short','--basetemp',str(stage/'pytest'),'tests/pilot/test_pilot_offline_process.py','tests/pilot/test_pilot_offline.py::test_offline_git_pin_requires_no_lazy_fetch_version','tests/pilot/test_pilot_offline.py::test_fresh_offline_boundary_denies_cloud_and_subprocess_escape','tests/pilot/test_pilot_offline.py::test_offline_child_cannot_write_outside_counted_workspace','tests/pilot/test_pilot_offline.py::test_git_provenance_denies_helper_configuration_and_keeps_closed_reads']
start=time.monotonic(); p=subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE); sel=selectors.DefaultSelector(); buffers={}
for name,s in [('stdout',p.stdout),('stderr',p.stderr)]: sel.register(s,selectors.EVENT_READ,name); buffers[name]=bytearray()
try:
 while sel.get_map():
  for key,_ in sel.select(0.2):
   chunk=os.read(key.fileobj.fileno(),4096)
   if not chunk: sel.unregister(key.fileobj); continue
   if len(buffers[key.data])+len(chunk)>16384: p.kill(); raise RuntimeError('output bound')
   buffers[key.data].extend(chunk)
  if time.monotonic()-start>60: p.kill(); raise RuntimeError('timeout')
finally:
 p.wait(); sel.close()
 for name,payload in buffers.items():
  with (stage/(name+'.log')).open('xb') as f: f.write(payload)
for name,payload in buffers.items(): print(name+':\n'+payload.decode())
items=list(stage.rglob('*')); files=[q for q in items if q.is_file() and not q.is_symlink()]
counts=dict(exit=p.returncode,seconds=round(time.monotonic()-start,3),logical=sum(q.stat().st_size for q in files),allocated=sum(q.lstat().st_blocks*512 for q in [stage,*items]),files=len(files),links=sum(q.is_symlink() for q in items),dirs=sum(q.is_dir() and not q.is_symlink() for q in [stage,*items]))
print(counts)
assert counts['logical']<256*1024 and counts['allocated']<=3*1048576
assert counts['files']+counts['links']<=320 and counts['dirs']<=384
PY
```

### Prelaunch AST forecast correction

The first FINAL1 launcher stopped before stage.mkdir, project imports, tests or
children: ast.literal_eval rejected a parameter value containing an ast.BinOp
(the test's literal hash string multiplication). No final1 directory or output
was created; retained135168B unchanged. Corrected forecast counts AST list/tuple
members without interpreting their values. Same prefix remains unused;
Controller confirmed prefix absence and re-admitted the corrected invocation.

## Implementation and qualified limits

Hash-only intent-v2 and strict result-v2 dispositions replace new plaintext-path
records. Executable-content/source/bootstrap hashes and exact private launch
argv/pending/environment checks remain. Public streams are empty only;
nonempty prefixes stay unchanged under fixed operational logs custody, and
make the attempt failed/nonreusable. Capture's byte/timeout/join behavior is
unchanged. Forensic V1 classes/readers stay explicit; the old terminal controls
now first validate a positive V1 baseline before exercising corrupt variants.

Same-owner preparation consumes the existing _ScopedAdmission, validates
check_scoped ownership/commitment and remaining category amounts under the
existing workspace lock, pins new private custody and never reserves/rebinds.
Only binding.json/stdout.raw/stderr.raw are retained privately. Direct
create-only0600 publication retains partial failures rather than deleting them;
the admitted final+temporary P remains conservative. Descriptor context close
does not delete evidence. Capability is frozen/private-repr and rejected by
normal public JSON/pickle serialization; no public locator is emitted.

Current collection/reuse require safe records. Full-gate safe eligibility
continues after actual source authentication; standalone forensic reading
retains default historical compatibility. Failed/private available outcomes
remain fatal despite missing logs; independently available corruption is checked
with exhaustive inventory and one cold lease. Logical context-root checks and
bounded read error/lease distinctions remain unchanged.

Checks, recovery, run/workflow and verification wrapper forward only the narrow
custody parameter. New launches lacking it fail before Git or expensive work.
Inherited supervisor/FitGuard authority remains unimplemented and explicitly
unsupported for fresh diagnostics. Training-only paths were not changed.
No successful scientific report, model execution, provider behavior, full gate,
TaskB allowance, TaskC hold or complete workflow is qualified by these controls.

### Explicit boundary qualification concern

The98 controls exercise private placement, public absence and logs counting,
but do NOT invoke a scientific exporter or engineering candidate generator.
The exclusion argument is source-grounded:
- archive/producer.py:226, _entry, derives members relative to the scientific
  run; ArchiveProducer.track_file at778 passes self.run_dir into it.
- train/pilot_evidence.py:1503 collects only descendants of run/final/offline.
- archive/preflight.py:146, _inventory, skips the direct live workspace at160.
- archive/preflight.py:764, iter_engineering_candidates, consumes the
  authenticated stored engineering snapshot, not arbitrary live workspace files.
- train/pilot_offline_process.py, _custody_paths, derives the fixed logs route
  and rejects public-run/control/spool/cache overlap.
These are paths under src/silent_cascade. No end-to-end export or future
operational-archive reclassification proof is claimed. The controller's
independent reviewer must judge this boundary against the brief; source remains
frozen unless a finding is routed.

## Exact earlier stage commands

These are the commands actually used before FINAL1's environment additions.
Each wrapper prints the pytest exit field; wrapper success alone is not GREEN.
No retained prefix is reused.

### RED1
```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-privacy/red1'
assert not stage.exists()
assert 48*1024+8192*(20+24)<=512*1024
stage.mkdir(parents=True)
for name in ('tmp','xdg','mpl','hypothesis','archive'): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/'tmp'),TMP=str(stage/'tmp'),TEMP=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'xdg'),MPLCONFIGDIR=str(stage/'mpl'),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'),OMP_NUM_THREADS='1')
names=['test_privacy_requires_custody_before_git','test_privacy_intent_hashes_replace_paths','test_privacy_nonempty_capture_never_enters_public_logs']
args=[str(root/'.venv/bin/python'),'-B','-m','pytest','-p','no:cacheprovider','--noconftest','-q','--tb=short','--basetemp',str(stage/'pytest'),*['tests/pilot/test_pilot_offline_process.py::'+name for name in names]]
start=time.monotonic(); p=subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE); sel=selectors.DefaultSelector(); buffers={}
for name,s in [('stdout',p.stdout),('stderr',p.stderr)]: sel.register(s,selectors.EVENT_READ,name); buffers[name]=bytearray()
try:
 while sel.get_map():
  for key,_ in sel.select(0.2):
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
items=list(stage.rglob('*')); files=[q for q in items if q.is_file() and not q.is_symlink()]
counts=dict(exit=p.returncode,seconds=round(time.monotonic()-start,3),logical=sum(q.stat().st_size for q in files),allocated=sum(q.lstat().st_blocks*512 for q in [stage,*items]),files=len(files),links=sum(q.is_symlink() for q in items),dirs=sum(q.is_dir() and not q.is_symlink() for q in [stage,*items]))
print(counts)
assert counts['logical']<48*1024 and counts['allocated']<=512*1024
assert counts['files']+counts['links']<=20 and counts['dirs']<=24
PY
```

### RED2
```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-privacy/red2'
assert not stage.exists()
assert 96*1024+8192*(96+96)<=2*1048576
stage.mkdir(parents=True)
for name in ('tmp','xdg','mpl','hypothesis','archive'): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/'tmp'),TMP=str(stage/'tmp'),TEMP=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'xdg'),MPLCONFIGDIR=str(stage/'mpl'),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'),OMP_NUM_THREADS='1')
names=['test_privacy_custody_pins_counted_private_directory','test_privacy_custody_rejects_unadmitted_targets','test_privacy_terminal_disposition_is_strict','test_privacy_callers_stop_before_expensive_work']
args=[str(root/'.venv/bin/python'),'-B','-m','pytest','-p','no:cacheprovider','--noconftest','-q','--tb=short','--basetemp',str(stage/'pytest'),*['tests/pilot/test_pilot_offline_process.py::'+name for name in names]]
start=time.monotonic(); p=subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE); sel=selectors.DefaultSelector(); buffers={}
for name,s in [('stdout',p.stdout),('stderr',p.stderr)]: sel.register(s,selectors.EVENT_READ,name); buffers[name]=bytearray()
try:
 while sel.get_map():
  for key,_ in sel.select(0.2):
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
items=list(stage.rglob('*')); files=[q for q in items if q.is_file() and not q.is_symlink()]
counts=dict(exit=p.returncode,seconds=round(time.monotonic()-start,3),logical=sum(q.stat().st_size for q in files),allocated=sum(q.lstat().st_blocks*512 for q in [stage,*items]),files=len(files),links=sum(q.is_symlink() for q in items),dirs=sum(q.is_dir() and not q.is_symlink() for q in [stage,*items]))
print(counts)
assert counts['logical']<96*1024 and counts['allocated']<=2*1048576
assert counts['files']+counts['links']<=96 and counts['dirs']<=96
PY
```

### GREEN1
```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-privacy/green1'
assert not stage.exists()
assert 128*1024+8192*(96+96)<=2*1048576
stage.mkdir(parents=True)
for name in ('tmp','xdg','mpl','hypothesis','archive'): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/'tmp'),TMP=str(stage/'tmp'),TEMP=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'xdg'),MPLCONFIGDIR=str(stage/'mpl'),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'),OMP_NUM_THREADS='1')
names=['test_privacy_requires_custody_before_git','test_privacy_intent_hashes_replace_paths','test_privacy_nonempty_capture_never_enters_public_logs','test_privacy_custody_pins_counted_private_directory','test_privacy_custody_rejects_unadmitted_targets','test_privacy_terminal_disposition_is_strict','test_privacy_callers_stop_before_expensive_work']
args=[str(root/'.venv/bin/python'),'-B','-m','pytest','-p','no:cacheprovider','--noconftest','-q','--tb=short','--basetemp',str(stage/'pytest'),*['tests/pilot/test_pilot_offline_process.py::'+name for name in names]]
start=time.monotonic(); p=subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE); sel=selectors.DefaultSelector(); buffers={}
for name,s in [('stdout',p.stdout),('stderr',p.stderr)]: sel.register(s,selectors.EVENT_READ,name); buffers[name]=bytearray()
try:
 while sel.get_map():
  for key,_ in sel.select(0.2):
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
items=list(stage.rglob('*')); files=[q for q in items if q.is_file() and not q.is_symlink()]
counts=dict(exit=p.returncode,seconds=round(time.monotonic()-start,3),logical=sum(q.stat().st_size for q in files),allocated=sum(q.lstat().st_blocks*512 for q in [stage,*items]),files=len(files),links=sum(q.is_symlink() for q in items),dirs=sum(q.is_dir() and not q.is_symlink() for q in [stage,*items]))
print(counts)
assert counts['logical']<128*1024 and counts['allocated']<=2*1048576
assert counts['files']+counts['links']<=96 and counts['dirs']<=96
PY
```

## Static checks and handoff

The exact seven-file list for both Ruff invocations was:
src/silent_cascade/train/pilot_offline_process.py,
src/silent_cascade/train/pilot_offline.py,
src/silent_cascade/train/pilot_evidence.py,
src/silent_cascade/train/pilot_checks.py,
src/silent_cascade/train/pilot_workflow.py,
src/silent_cascade/train/pilot_verification.py,
tests/pilot/test_pilot_offline_process.py.

Commands: .venv/bin/ruff check --no-cache <seven files>;
.venv/bin/ruff format --check --no-cache <seven files>; git diff --check.
All returned0 after FINAL1, before source commit. No semantic source edits
followed FINAL1. Git index writes initially hit the sandbox's read-only .git
boundary; scoped escalation then staged/committed only the eight owned files.
Controller progress and other docs were not staged. This report is a separate
report-only handoff; controller repeat/review is independent evidence.
