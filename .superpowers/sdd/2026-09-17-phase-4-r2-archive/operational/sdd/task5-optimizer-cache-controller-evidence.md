# Optimizer cache correction — controller execution evidence

Task owner81184/PTY20632/token14b87188832c94b90df668388f03ca65. Same global
ledger,2MiB task scratch; no provider, full diagnostic, training update or
scientific gate. Every stage retained. RED outer-limit enforcement deviation
is recorded in progress/report; no such enforcement claim is made for RED.

## GREEN1

Working diff: one closed-environment line +47 test lines. Exact four selectors
expand to five cases, including the pre-existing closed Git batch control.
Pytest5passed4.86s/wall5.032s; Ruffcheck0(0.061s),formatcheck0(0.014s),
scopeddiffcheck0(0.010s). Wrapper0. Raw output in scratch/task5-optimizer-cache/
green1/{0,1,2,3}-{stdout,stderr}.log. All stderr empty. Actual143logical/
12288allocated bytes,8files4links19dirs. Cumulativetask24576allocated.

The wrapper enforces32768bytes per stream and150seconds per command; fresh
prefix, plugins/cache disabled, closed environment, confined runtime roots.
Admission1310720allocated including full64KiB allowance per tinychild, complete
output retention, metadata and directory/name cushions. No fullfile selection.

Exact command (executed once):

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/Volumes/git/legotin/silent-cascade/src .venv/bin/python -B -u -c 'import json,os,pathlib,selectors,subprocess,time
root=pathlib.Path.cwd()
stage=root/".superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-optimizer-cache/green1"
assert not stage.exists() and not stage.is_symlink()
stage.mkdir()
for name in ("tmp","xdg","mpl","hypothesis","archive"): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/"tmp"),TMP=str(stage/"tmp"),TEMP=str(stage/"tmp"),XDG_CACHE_HOME=str(stage/"xdg"),XDG_CONFIG_HOME=str(stage/"xdg"),XDG_DATA_HOME=str(stage/"xdg"),MPLCONFIGDIR=str(stage/"mpl"),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/"hypothesis"),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/"archive"),OMP_NUM_THREADS="1")
nodes=["tests/pilot/test_pilot_offline_writes.py::test_offline_environment_derives_inductor_cache_and_ignores_ambient","tests/pilot/test_pilot_offline_writes.py::test_real_scalar_adamw_initializes_only_empty_cache_root","tests/pilot/test_pilot_offline_writes.py::test_optimizer_cache_artifact_denials_latch","tests/pilot/test_pilot_offline_writes.py::test_closed_git_batch_pipe_still_works"]
files=["src/silent_cascade/train/pilot_offline.py","tests/pilot/test_pilot_offline_writes.py"]
commands=[[str(root/".venv/bin/python"),"-B","-m","pytest","--noconftest","-p","no:cacheprovider","-q","--tb=short","--basetemp",str(stage/"pytest"),*nodes],[str(root/".venv/bin/ruff"),"check","--no-cache",*files],[str(root/".venv/bin/ruff"),"format","--check","--no-cache",*files],["git","diff","--check", "--", *files]]

statuses=[]
for index,args in enumerate(commands):
 started=time.monotonic()
 p=subprocess.Popen(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 selector=selectors.DefaultSelector(); buffers={}
 for name,stream in (("stdout",p.stdout),("stderr",p.stderr)):
  selector.register(stream,selectors.EVENT_READ,name); buffers[name]=bytearray()
 try:
  while selector.get_map():
   for key,_ in selector.select(0.2):
    chunk=os.read(key.fileobj.fileno(),4096)
    if not chunk: selector.unregister(key.fileobj); continue
    if len(buffers[key.data])+len(chunk)>32768:
     p.kill(); raise RuntimeError("outer output bound")
    buffers[key.data].extend(chunk)
   if time.monotonic()-started>150:
    p.kill(); raise RuntimeError("outer timeout")
 finally:
  p.wait(); selector.close()
  for name,payload in buffers.items():
   with (stage/(str(index)+"-"+name+".log")).open("xb") as stream: stream.write(payload)
 print(json.dumps(dict(command=args,exit=p.returncode,seconds=round(time.monotonic()-started,3))),flush=True)
 for name,payload in buffers.items(): print(name+":\n"+payload.decode(errors="replace"),flush=True)
 statuses.append(p.returncode)
 if p.returncode: break
paths=[stage,*stage.rglob("*")]
infos=[p.lstat() for p in paths]
counts=dict(allocated=sum(s.st_blocks*512 for s in infos),logical=sum(p.stat().st_size for p in paths if p.is_file() and not p.is_symlink()),files=sum(p.is_file() and not p.is_symlink() for p in paths),links=sum(p.is_symlink() for p in paths),dirs=sum(p.is_dir() and not p.is_symlink() for p in paths))
print(json.dumps(dict(inventory=counts,statuses=statuses)),flush=True)
assert counts["allocated"]<=1310720 and counts["logical"]<524288
assert counts["files"]+counts["links"]<=32 and counts["dirs"]<=48
assert len(statuses)==len(commands) and not any(statuses)
'
```
