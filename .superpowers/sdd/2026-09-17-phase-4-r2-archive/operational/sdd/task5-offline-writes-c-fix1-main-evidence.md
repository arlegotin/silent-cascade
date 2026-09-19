# Task C FIX1 controller verification

Frozen source0aee7bdc3b99bf8cad7aec51ae5a12075c3ff75d; fix changes only fixture/writer tests. Four source/test files match HEAD; process path source remains unchanged from44edf69.

| Fresh prefix | Result | Wall seconds | Logical B | Allocated B | Files / links / dirs |
| --- | --- | ---: | ---: | ---: | --- |
| main-fix1 |26passed2.41s|2.948|23513|122880|50 /16 /156|
| main-fix1-static |fourcommands0|<1|45|8192|8 /0 /1|

Pytest/wrapper0; stdout99B/stderr0. Four-file Ruff check/format check, diff check and commit equality0. Fixed-file hashes match the implementer report. Current C retained allocation1163264B; all RED/failure stages remain. One Important directory-union finding is under scoped re-review; no actual offline worker, model training/evaluation/replay, fullgate, make verify or provider ran. Original31+27qualification at44edf69 remains recorded; this is a26-case fix repeat, not84unique tests.

## Exact independently executed commands

### task5CFix1MainCommand

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-c/main-fix1'
assert not stage.exists() and not stage.is_symlink()
assert os.statvfs(root).f_frsize <= 4096
assert 262144+8192*(96+188)+4096<=2621440
stage.mkdir(parents=True)
for name in ('tmp','xdg','mpl','hypothesis','archive'): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/'tmp'),TMP=str(stage/'tmp'),TEMP=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'xdg'),MPLCONFIGDIR=str(stage/'mpl'),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'),OMP_NUM_THREADS='1',XDG_CONFIG_HOME=str(stage/'xdg'),XDG_DATA_HOME=str(stage/'xdg'),PYTHONHASHSEED='0',SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH='1')
names=["tests/pilot/test_pilot_offline_writes.py::test_fit_guard_counts_public_ancestors_outside_spool","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_binds_existing_publisher_aliases","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_reserves_exact_offline_hold_before_publication","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_real_parent_publication_creates_fresh_held_root","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_offline_hold_cannot_be_rebound_or_replenished","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_offline_hold_rejects_short_inventory","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_live_child_bytes_are_not_double_counted","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_unrelated_output_cannot_spend_offline_hold","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_routes_parent_publications_through_held_process_peak","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_rejects_split_child_categories","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_validates_exact_offline_hold_bindings","tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_rejects_oversized_publisher_before_any_output","tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_counts_existing_and_pending_checkpoint_before_publication","tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_stops_pending_row_append_before_handle_write","tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_rejection_remains_blocked_after_caller_catches_error","tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_refuses_name_or_directory_growth"]
args=[str(root/'.venv/bin/python'),'-B','-m','pytest','-p','no:cacheprovider','--noconftest','-q','--tb=short','--basetemp',str(stage/'pytest'),*names]
start=time.monotonic(); p=subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE); sel=selectors.DefaultSelector(); buffers={}
for name,s in [('stdout',p.stdout),('stderr',p.stderr)]: sel.register(s,selectors.EVENT_READ,name); buffers[name]=bytearray()
try:
 while sel.get_map():
  for key,_ in sel.select(0.2):
   chunk=os.read(key.fileobj.fileno(),4096)
   if not chunk: sel.unregister(key.fileobj); continue
   if len(buffers[key.data])+len(chunk)>32768: p.kill(); raise RuntimeError('output bound')
   buffers[key.data].extend(chunk)
  if time.monotonic()-start>120: p.kill(); raise RuntimeError('timeout')
finally:
 p.wait(); sel.close()
 for name,payload in buffers.items():
  with (stage/(name+'.log')).open('xb') as f: f.write(payload)
for name,payload in buffers.items(): print(name+':\n'+payload.decode())
items=list(stage.rglob('*')); files=[q for q in items if q.is_file() and not q.is_symlink()]
counts=dict(exit=p.returncode,seconds=round(time.monotonic()-start,3),logical=sum(q.stat().st_size for q in files),allocated=sum(q.lstat().st_blocks*512 for q in [stage,*items]),files=len(files),links=sum(q.is_symlink() for q in items),dirs=sum(q.is_dir() and not q.is_symlink() for q in [stage,*items]))
print(counts)
assert counts['logical']<262144 and counts['allocated']<=2621440
assert counts['files']+counts['links']<=96 and counts['dirs']<=188
assert p.returncode == 0
PY
```

### task5CFix1MainStaticCommand

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-c/main-fix1-static'
assert not stage.exists() and not stage.is_symlink()
assert 128*1024+8192*(16+24)+4096<=512*1024
stage.mkdir(parents=True)
files=["src/silent_cascade/train/pilot_offline_process.py","tests/pilot/cold_authentication_fixture.py","tests/pilot/test_pilot_offline_process.py","tests/pilot/test_pilot_offline_writes.py"]
commands=[['.venv/bin/ruff','check','--no-cache',*files],['.venv/bin/ruff','format','--check','--no-cache',*files],['git','diff','--check'],['git','diff','--exit-code','0aee7bd','--',*files]]
statuses=[]
for index,args in enumerate(commands):
 start=time.monotonic(); p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE); sel=selectors.DefaultSelector(); buffers={}
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
   with (stage/(str(index)+'-'+name+'.log')).open('xb') as f: f.write(payload)
 statuses.append(p.returncode); print(args,p.returncode)
 for name,payload in buffers.items(): print(name+':'+payload.decode())
items=list(stage.rglob('*')); files=[q for q in items if q.is_file() and not q.is_symlink()]
counts=dict(logical=sum(q.stat().st_size for q in files),allocated=sum(q.lstat().st_blocks*512 for q in [stage,*items]),files=len(files),dirs=sum(q.is_dir() for q in [stage,*items]))
print(counts)
assert counts['logical']<128*1024 and counts['allocated']<=512*1024
assert counts['files']<=16 and counts['dirs']<=24 and statuses==[0,0,0,0]
PY
```
