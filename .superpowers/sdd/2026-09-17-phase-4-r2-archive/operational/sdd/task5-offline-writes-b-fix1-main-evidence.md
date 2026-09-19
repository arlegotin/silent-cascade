# Task B FIX1 controller verification

Source/test freeze cc05f1fd97a258f04aa4d9262d337fca1078bf17.
Test SHA256184b69d47e0a2b1113a20de067f65ad4f47d90745a3d35f73f124b9600684eac.
Production source unchanged from9f0c0b3; owned4file Git equality exit0.

Fresh independent prefixes beneath operational/scratch/task5-offline-writes-b:

| Prefix | Result | Wall | Logical B | Allocated B | Files / links / dirs |
| --- | --- | ---: | ---: | ---: | --- |
| main-fix1-combined | 4passed20.52s | 21.106s | 9134 | 102400 | 34 / 1 / 47 |
| main-fix1-legacy | 7passed4.69s | 5.265s | 14503 | 163840 | 57 / 3 / 77 |
| main-fix1-static | four commands exit0 | <1s | 45 | 8192 | 8 / 0 / 1 |

All stdout bounded, all stderr empty; pytest and wrappers exit0. Whole11
covering tests independently repeated in two partitions to preserve the same
4MiB task hold. Static4file Ruff check/format-check, diffcheck, and equality
againstcc05f1f all passed. All attempts retained; final Task B allocation2322432B.
Combined controls require real observed denial before their competing failure,
failed public results, exact private captured prefixes/hashes, invalid report
preservation without successful authentication, and already-reaped children.

No genuine worker, fullgate, make verify, ML fit/eval/replay or provider ran.
Reviewer accepted FIX1 with no new breakage; initial40+whole64 source evidence
remains in task5-offline-writes-b-main-evidence.md at9f0c0b3. The postfix run is
scoped11, not a fresh rerun of all prior tests. Final custody closeout is recorded
in progress.md after holder29448 completes its final check.

## Exact independently executed commands

Each admitted before execution, once, HEAD unchanged throughout identity tests.

### task5BMainFix1CombinedCommand

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-b/main-fix1-combined'
assert not stage.exists() and not stage.is_symlink()
assert os.statvfs(root).f_frsize <= 4096
assert 128*1024+8192*(96+128)+4096<=2*1024*1024
stage.mkdir(parents=True)
for name in ('tmp','xdg','mpl','hypothesis','archive'): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/'tmp'),TMP=str(stage/'tmp'),TEMP=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'xdg'),MPLCONFIGDIR=str(stage/'mpl'),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'),OMP_NUM_THREADS='1',XDG_CONFIG_HOME=str(stage/'xdg'),XDG_DATA_HOME=str(stage/'xdg'),PYTHONHASHSEED='0',SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH='1')
names=["tests/pilot/test_pilot_offline_process.py::test_measurement_combines_write_exhaustion_with_parent_failure_custody"]
args=[str(root/'.venv/bin/python'),'-B','-m','pytest','-p','no:cacheprovider','--noconftest','-q','--tb=short','--basetemp',str(stage/'pytest'),*names]
start=time.monotonic(); p=subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE); sel=selectors.DefaultSelector(); buffers={}
for name,s in [('stdout',p.stdout),('stderr',p.stderr)]: sel.register(s,selectors.EVENT_READ,name); buffers[name]=bytearray()
try:
 while sel.get_map():
  for key,_ in sel.select(0.2):
   chunk=os.read(key.fileobj.fileno(),4096)
   if not chunk: sel.unregister(key.fileobj); continue
   if len(buffers[key.data])+len(chunk)>16384: p.kill(); raise RuntimeError('output bound')
   buffers[key.data].extend(chunk)
  if time.monotonic()-start>90: p.kill(); raise RuntimeError('timeout')
finally:
 p.wait(); sel.close()
 for name,payload in buffers.items():
  with (stage/(name+'.log')).open('xb') as f: f.write(payload)
for name,payload in buffers.items(): print(name+':\n'+payload.decode())
items=list(stage.rglob('*')); files=[q for q in items if q.is_file() and not q.is_symlink()]
counts=dict(exit=p.returncode,seconds=round(time.monotonic()-start,3),logical=sum(q.stat().st_size for q in files),allocated=sum(q.lstat().st_blocks*512 for q in [stage,*items]),files=len(files),links=sum(q.is_symlink() for q in items),dirs=sum(q.is_dir() and not q.is_symlink() for q in [stage,*items]))
print(counts)
assert counts['logical']<128*1024 and counts['allocated']<=2*1024*1024
assert counts['files']+counts['links']<=96 and counts['dirs']<=128
assert p.returncode == 0
PY
```

### task5BMainFix1LegacyCommand

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-b/main-fix1-legacy'
assert not stage.exists() and not stage.is_symlink()
assert os.statvfs(root).f_frsize <= 4096
assert 128*1024+8192*(80+104)+4096<=1792*1024
stage.mkdir(parents=True)
for name in ('tmp','xdg','mpl','hypothesis','archive'): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/'tmp'),TMP=str(stage/'tmp'),TEMP=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'xdg'),MPLCONFIGDIR=str(stage/'mpl'),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'),OMP_NUM_THREADS='1',XDG_CONFIG_HOME=str(stage/'xdg'),XDG_DATA_HOME=str(stage/'xdg'),PYTHONHASHSEED='0',SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH='1')
names=["tests/pilot/test_pilot_offline_process.py::test_measurement_serializes_exact_write_allowance_before_launch","tests/pilot/test_pilot_offline_process.py::test_measurement_persists_failed_invalid_report_and_fences_reuse","tests/pilot/test_pilot_offline_process.py::test_privacy_nonempty_failure_retains_closed_outcome"]
args=[str(root/'.venv/bin/python'),'-B','-m','pytest','-p','no:cacheprovider','--noconftest','-q','--tb=short','--basetemp',str(stage/'pytest'),*names]
start=time.monotonic(); p=subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE); sel=selectors.DefaultSelector(); buffers={}
for name,s in [('stdout',p.stdout),('stderr',p.stderr)]: sel.register(s,selectors.EVENT_READ,name); buffers[name]=bytearray()
try:
 while sel.get_map():
  for key,_ in sel.select(0.2):
   chunk=os.read(key.fileobj.fileno(),4096)
   if not chunk: sel.unregister(key.fileobj); continue
   if len(buffers[key.data])+len(chunk)>16384: p.kill(); raise RuntimeError('output bound')
   buffers[key.data].extend(chunk)
  if time.monotonic()-start>90: p.kill(); raise RuntimeError('timeout')
finally:
 p.wait(); sel.close()
 for name,payload in buffers.items():
  with (stage/(name+'.log')).open('xb') as f: f.write(payload)
for name,payload in buffers.items(): print(name+':\n'+payload.decode())
items=list(stage.rglob('*')); files=[q for q in items if q.is_file() and not q.is_symlink()]
counts=dict(exit=p.returncode,seconds=round(time.monotonic()-start,3),logical=sum(q.stat().st_size for q in files),allocated=sum(q.lstat().st_blocks*512 for q in [stage,*items]),files=len(files),links=sum(q.is_symlink() for q in items),dirs=sum(q.is_dir() and not q.is_symlink() for q in [stage,*items]))
print(counts)
assert counts['logical']<128*1024 and counts['allocated']<=1792*1024
assert counts['files']+counts['links']<=80 and counts['dirs']<=104
assert p.returncode == 0
PY
```

### task5BMainFix1StaticCommand

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-b/main-fix1-static'
assert not stage.exists() and not stage.is_symlink()
assert 128*1024+8192*(16+24)+4096<=512*1024
stage.mkdir(parents=True)
files=["src/silent_cascade/train/pilot_offline.py","src/silent_cascade/train/pilot_offline_process.py","tests/pilot/test_pilot_offline_process.py","tests/pilot/test_pilot_offline_writes.py"]
commands=[['.venv/bin/ruff','check','--no-cache',*files],['.venv/bin/ruff','format','--check','--no-cache',*files],['git','diff','--check'],['git','diff','--exit-code','cc05f1f','--',*files]]
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
