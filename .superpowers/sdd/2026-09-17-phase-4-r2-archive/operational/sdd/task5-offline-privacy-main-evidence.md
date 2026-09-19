# Controller verification — Task5 parent privacy correction

Source base89e987a; frozen source/test commit71eabfe. All eight changed
source/test/ignore files matched that commit before and after execution.
This is privacy/process primitive verification, not the real offline diagnostic,
a full local gate, inherited supervisor/FitGuard routing, R2 promotion or Phase4
completion.

## Fresh MAIN1 result

- Exact98 cases:94 process module cases plus the four explicitly named safe
  legacy controls in the command below.
- pytest exit0,98 passed in12.11s; wrapper wall12.859s; stderr empty.
-54976 logical bytes,507904 allocated bytes,183 files,36 symlinks,306 dirs.
- Freshmain1 admitted <=3MiB/<256KiB/320names/384dirs/60s and16KiB per
  launcher log. Every asserted ceiling passed.
-17 tiny Python controls and up to32 trusted read-only Git calls; no real
  worker/model/fit/evaluation/replay/provider execution or copied corpus.
- Prior stages643072B plusmain1=1150976 allocated bytes retained; no deletion.
- Source/function coverage includes hash-only launch binding, private captures,
  strict current versus forensic readers, early caller guards, bounded capture
  failure paths and same-owner ledger custody. Existing export membership
  boundaries are source-reviewed, not a newly executed end-to-end export.

## Exact command executed

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import ast, math, os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-privacy/main1'
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

## Static and source equality checks

Scoped Ruff check --no-cache: All checks passed.
Scoped Ruff format --check --no-cache:7 files already formatted.
git diff --check and git diff --exit-code71eabfe over the eight task files:exit0.

```sh
.venv/bin/ruff check --no-cache src/silent_cascade/train/pilot_checks.py src/silent_cascade/train/pilot_evidence.py src/silent_cascade/train/pilot_offline.py src/silent_cascade/train/pilot_offline_process.py src/silent_cascade/train/pilot_verification.py src/silent_cascade/train/pilot_workflow.py tests/pilot/test_pilot_offline_process.py
.venv/bin/ruff format --check --no-cache src/silent_cascade/train/pilot_checks.py src/silent_cascade/train/pilot_evidence.py src/silent_cascade/train/pilot_offline.py src/silent_cascade/train/pilot_offline_process.py src/silent_cascade/train/pilot_verification.py src/silent_cascade/train/pilot_workflow.py tests/pilot/test_pilot_offline_process.py
git diff --check
git diff --exit-code 71eabfe -- .gitignore src/silent_cascade/train/pilot_checks.py src/silent_cascade/train/pilot_evidence.py src/silent_cascade/train/pilot_offline.py src/silent_cascade/train/pilot_offline_process.py src/silent_cascade/train/pilot_verification.py src/silent_cascade/train/pilot_workflow.py tests/pilot/test_pilot_offline_process.py
```

## File SHA-256

```text
abbafd45509930c6d0d323828e333f09c1789817c7f2e2c6d617c2748145e9cb  .gitignore
9341d80b2aefdd9f245532b3464a4219eedf60ff90f3b794072db2c4d8eb304f  src/silent_cascade/train/pilot_checks.py
528b25bc25ee495dbbbe72616255af46ff542d7d73b2770d0d0585c289d02fec  src/silent_cascade/train/pilot_evidence.py
a21e08c6a8f2f493b1b8a0d4bef620edbacdcd83ab974e47b9e8dba38e91cb72  src/silent_cascade/train/pilot_offline.py
a92da53dfef17343f356252250ed4e4e2395cfe2d46f28cfdf0d8483ab110db2  src/silent_cascade/train/pilot_offline_process.py
9efb5456150dca1319e01a287870cb9537994207ab64f3c7802348fbf36c86c8  src/silent_cascade/train/pilot_verification.py
cbb55bcb2ab8bf105130e31d70f7f84976c5101a2e30d2bff16466b53b439471  src/silent_cascade/train/pilot_workflow.py
09f6cf232f538c8c4e441a6821beaba548d5cbbd304a0fb711ffd11ae66aa70b  tests/pilot/test_pilot_offline_process.py
```

Independent task review and full owner/frozen-custody accounting are separate
remaining checks. Original evidence outside operational remains immutable.

## Existing membership boundaries (read-only source check)

Named risk: private raw captures accidentally entering an existing engineering
or scientific export. The controller inspected the existing boundary functions,
not an exporter execution:

- archive/preflight.py:146 `_inventory` skips the live workspace child at its
  root before visiting any contained path (line160).
- archive/preflight.py:837 `iter_engineering_candidates` accepts only closed
  tmp/task roots and its default traversal starts at those roots (line857).
- archive/producer.py:226 `_entry` requires a supplied file to be relative to
  the scientific run; track_file at778 uses that exact run root.
- The privacy task places capture files in the separate fixed workspace logs
  subtree and rejects run/private overlap. Its scientific readers request no
  raw captures or private receipts.

These static boundaries support present membership exclusion, not permission
for a future generic operational export. Such an exporter must independently
exclude private custody or reject launch. No R2/export end-to-end test was run.

Owner61737 full check passed after MAIN1: scratch260026368,logs2301952,
spool219099136,cache67153920,metadata52965376,pinned/emergency0. Original
frozen retained bytes remain6195077120 under the unchanged policy. Owner stays
live for review/fixes; this check does not close Task5 or Phase4.

## FIX1-MAIN independent repeat — source6754bd8

32 passed in5.31s; pytest exit0, wall5.890s, stderr empty. Fresh retained
fix1-main:29543 logical /278528 allocated bytes,111 files/12 links/218 dirs.
All admitted bounds passed; total privacy scratch1720320 allocated bytes.
No full110-case run, genuine diagnostic, provider, or Phase4 claim.

Exact independent command:

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-privacy/fix1-main'
assert not stage.exists() and not stage.is_symlink()
assert os.statvfs(root).f_frsize <= 4096
assert 192*1024+4096*(256+320)<=2560*1024
stage.mkdir(parents=True)
for name in ('tmp','xdg','mpl','hypothesis','archive'): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/'tmp'),TMP=str(stage/'tmp'),TEMP=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'xdg'),MPLCONFIGDIR=str(stage/'mpl'),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'),OMP_NUM_THREADS='1',XDG_CONFIG_HOME=str(stage/'xdg'),XDG_DATA_HOME=str(stage/'xdg'),PYTHONHASHSEED='0',SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH='1')
names=["test_privacy_custody_rejects_split_publication_categories_before_writes","test_privacy_custody_preserves_compatible_publication_categories","test_privacy_custody_rechecks_publication_categories","test_privacy_custody_pins_counted_private_directory","test_privacy_custody_rejects_unadmitted_targets","test_privacy_prepared_custody_cannot_survive_authority_change","test_measurement_persists_failed_invalid_report_and_fences_reuse","test_measurement_records_launch_failure_without_return_code","test_log_publication_failure_never_publishes_completion","test_privacy_nonempty_capture_never_enters_public_logs","test_privacy_nonempty_failure_retains_closed_outcome"]
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
assert counts['logical']<192*1024 and counts['allocated']<=2560*1024
assert counts['files']+counts['links']<=256 and counts['dirs']<=320
PY
```

Scoped Ruff check --no-cache passed; format --check --no-cache reports2 files
already formatted. git diff --check and source equality to6754bd8 passed.

```text
e20eaa75f86423dbb5e9811cb69a35a95865232f21422bc39ada0e9f38fd2abd  src/silent_cascade/train/pilot_offline_process.py
971a4f6b8e164bd9f0f3746ae916ffb4db5938194a086167668be8298f926189  tests/pilot/test_pilot_offline_process.py
```

Owner61737 full custody/accounting check launched after repeat. Same reviewer
will inspect only the open Important and new breakage in the fix diff.
