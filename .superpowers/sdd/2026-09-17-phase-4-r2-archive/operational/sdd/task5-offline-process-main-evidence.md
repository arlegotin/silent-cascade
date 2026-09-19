# Task5 offline-process controller verification

Source frozen at `70aace8` (base `47395b7`); report-only `985a3a7` does not change source.
The sole holder23159/token deff627ffe458e5573ed87bf7504534a remained active.
No providers, scientific worker, training, evaluation or replay were executed.

## Fresh final check

Prospective main1 admission: retained task634880B plus next3MiB <= held4MiB.
Stage ceiling3MiB/<256KiB logical/144file-link-temp names/128dirs; logs16KiB
each. Root used the worker's closed launcher with additional PYTHONHASHSEED=0,
PRESERVE_SCRATCH=1 and XDG_CONFIG_HOME/XDG_DATA_HOME directed at its existing
xdg directory. No additional directory or scientific behavior was introduced.

Actual:60passed in8.00s, wall8.629s, pytestexit0, stderr empty.
Retained main1:24882logical/233472allocated/74regularfiles/24links/95dirs.
All thirteen task stages remain:868352allocated/278regularfiles/88links/415dirs.
No failed prefix was removed or reused. RED3's original directory forecast
failure remains recorded in the implementation report, not retroactively passed.

stdout SHA256:47e4f25f9c431101ccc1ff0750b4644658347975274d295270a16547cff99b89
stderr SHA256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

Fresh scoped Ruff lint and format checks passed on pilot_checks.py,
pilot_evidence.py, pilot_offline.py, pilot_offline_process.py and the new process
tests. `git diff --check` passed; `git diff --exit-code 70aace8 -- src tests` confirmed
unchanged source/tests.

## Executed command

This prefix already exists and must never be reused. A future repetition needs
a new explicit prefix/admission; the code below records the command already run.

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd(); stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-process/main1'
assert not stage.exists(); assert 256*1024+8192*(144+128)<=3*1048576
stage.mkdir(parents=True)
for name in ('tmp','xdg','mpl','hypothesis','archive'): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/'tmp'),TMP=str(stage/'tmp'),TEMP=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'xdg'),XDG_CONFIG_HOME=str(stage/'xdg'),XDG_DATA_HOME=str(stage/'xdg'),MPLCONFIGDIR=str(stage/'mpl'),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'),OMP_NUM_THREADS='1')
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

## Limits

Independent review approved with no blocking findings; see the review record.
Full local make verify, actual v2 worker,
child-write allowance, outer cold integration/recovery, storage handoff and
the Phase4 production pilot remain incomplete. These controls are process and
artifact integrity evidence, not learning or timing-advantage evidence.
