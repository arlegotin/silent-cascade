# Task C controller verification

Source freeze44edf6947aa6cf59d75c2bb31e8ae1a8bfaa9ee0; four owned files match this commit. Same owner96335, task hold4MiB scratch, unchanged10GiB policy.

| Fresh prefix | Result | Wall seconds | Logical B | Allocated B | Files / links / dirs |
| --- | --- | ---: | ---: | ---: | --- |
| main-custody |31passed0.98s|1.180|17421|139264|66 /10 /169|
| main-hold |27passed6.80s|7.347|21397|126976|47 /20 /137|
| main-static |fourcommands0|<1|45|8192|8 /0 /1|

All pytest/wrapper commands exited0. Both pytest stdout99B/stderr0. Static Ruff check/format check, diff check and exact four-file equality all exit0. The31+27 cases are disjoint. No genuine offline worker, fit, evaluation, replay, checkout, provider/network, full gate or make verify ran. Public/private custody paths and fixture-held reservation/writers are qualified only; Task5,Task7,Phase4 remain incomplete.

Current task allocation880640B includes every RED/failure/qualification stage. Frozen historical custody was not modified. Final owner check/release and independent task review are recorded separately in progress.md.

## Exact independently executed commands

Each ran once under its own prior admission with retained fresh output roots.

### task5CMainCustodyCommand

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-c/main-custody'
assert not stage.exists() and not stage.is_symlink()
assert os.statvfs(root).f_frsize <= 4096
assert 262144+8192*(96+200)+4096<=2752512
stage.mkdir(parents=True)
for name in ('tmp','xdg','mpl','hypothesis','archive'): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/'tmp'),TMP=str(stage/'tmp'),TEMP=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'xdg'),MPLCONFIGDIR=str(stage/'mpl'),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'),OMP_NUM_THREADS='1',XDG_CONFIG_HOME=str(stage/'xdg'),XDG_DATA_HOME=str(stage/'xdg'),PYTHONHASHSEED='0',SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH='1')
names=["tests/pilot/test_pilot_offline_process.py::test_privacy_custody_accepts_admitted_public_spool","tests/pilot/test_pilot_offline_process.py::test_privacy_custody_rejects_public_spool_one_byte_short","tests/pilot/test_pilot_offline_process.py::test_privacy_custody_rejects_fixed_logs_even_with_spool_subbinding","tests/pilot/test_pilot_offline_process.py::test_privacy_custody_rejects_private_storage_overlap","tests/pilot/test_pilot_offline_process.py::test_privacy_custody_rejects_split_publication_categories_before_writes","tests/pilot/test_pilot_offline_process.py::test_privacy_custody_preserves_compatible_publication_categories","tests/pilot/test_pilot_offline_process.py::test_privacy_custody_rechecks_publication_categories","tests/pilot/test_pilot_offline_process.py::test_privacy_custody_pins_counted_private_directory","tests/pilot/test_pilot_offline_process.py::test_privacy_custody_rejects_unadmitted_targets","tests/pilot/test_pilot_offline_process.py::test_privacy_prepared_custody_cannot_survive_authority_change"]
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
assert counts['logical']<262144 and counts['allocated']<=2752512
assert counts['files']+counts['links']<=96 and counts['dirs']<=200
assert p.returncode == 0
PY
```

### task5CMainHoldCommand

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-c/main-hold'
assert not stage.exists() and not stage.is_symlink()
assert os.statvfs(root).f_frsize <= 4096
assert 524288+8192*(128+220)+4096<=3407872
stage.mkdir(parents=True)
for name in ('tmp','xdg','mpl','hypothesis','archive'): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/'tmp'),TMP=str(stage/'tmp'),TEMP=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'xdg'),MPLCONFIGDIR=str(stage/'mpl'),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'),OMP_NUM_THREADS='1',XDG_CONFIG_HOME=str(stage/'xdg'),XDG_DATA_HOME=str(stage/'xdg'),PYTHONHASHSEED='0',SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH='1')
names=["tests/pilot/test_pilot_offline_writes.py::test_fit_guard_binds_existing_publisher_aliases","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_reserves_exact_offline_hold_before_publication","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_real_parent_publication_creates_fresh_held_root","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_offline_hold_cannot_be_rebound_or_replenished","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_offline_hold_rejects_short_inventory","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_live_child_bytes_are_not_double_counted","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_unrelated_output_cannot_spend_offline_hold","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_routes_parent_publications_through_held_process_peak","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_rejects_split_child_categories","tests/pilot/test_pilot_offline_writes.py::test_fit_guard_validates_exact_offline_hold_bindings","tests/pilot/test_pilot_offline_writes.py::test_install_cannot_replace_or_enlarge_allowance","tests/pilot/test_pilot_offline_writes.py::test_real_publishers_exact_capacity","tests/pilot/test_pilot_offline_writes.py::test_rows_stream_and_final_link","tests/pilot/test_pilot_offline_writes.py::test_atomic_replace_and_create_preserve_original_behavior","tests/pilot/test_pilot_offline_writes.py::test_closed_git_batch_pipe_still_works","tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_rejects_oversized_publisher_before_any_output","tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_counts_existing_and_pending_checkpoint_before_publication","tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_stops_pending_row_append_before_handle_write","tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_rejection_remains_blocked_after_caller_catches_error","tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_refuses_name_or_directory_growth"]
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
  if time.monotonic()-start>180: p.kill(); raise RuntimeError('timeout')
finally:
 p.wait(); sel.close()
 for name,payload in buffers.items():
  with (stage/(name+'.log')).open('xb') as f: f.write(payload)
for name,payload in buffers.items(): print(name+':\n'+payload.decode())
items=list(stage.rglob('*')); files=[q for q in items if q.is_file() and not q.is_symlink()]
counts=dict(exit=p.returncode,seconds=round(time.monotonic()-start,3),logical=sum(q.stat().st_size for q in files),allocated=sum(q.lstat().st_blocks*512 for q in [stage,*items]),files=len(files),links=sum(q.is_symlink() for q in items),dirs=sum(q.is_dir() and not q.is_symlink() for q in [stage,*items]))
print(counts)
assert counts['logical']<524288 and counts['allocated']<=3407872
assert counts['files']+counts['links']<=128 and counts['dirs']<=220
assert p.returncode == 0
PY
```

### task5CMainStaticCommand

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-c/main-static'
assert not stage.exists() and not stage.is_symlink()
assert 128*1024+8192*(16+24)+4096<=512*1024
stage.mkdir(parents=True)
files=["src/silent_cascade/train/pilot_offline_process.py","tests/pilot/cold_authentication_fixture.py","tests/pilot/test_pilot_offline_process.py","tests/pilot/test_pilot_offline_writes.py"]
commands=[['.venv/bin/ruff','check','--no-cache',*files],['.venv/bin/ruff','format','--check','--no-cache',*files],['git','diff','--check'],['git','diff','--exit-code','44edf69','--',*files]]
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
