# Independent frozen-source verification

Source `13fef3b8be474b7cd77768c5e8ff7b459e611130`; both owned file hashes
match the writer report. Same task owner and2MiB scratch allowance. Fresh main
prefix, four selectors/five cases, four tiny64KiB children; no actual diagnostic
or scientific model/update/evaluation. Retained raw logs at
scratch/task5-optimizer-cache/main/{0,1,2,3,4}-{stdout,stderr}.log.

Results: pytest5passed4.79s/wall4.962s; Ruffcheck0(0.057s),formatcheck0(0.015s),
scopeddiff0(0.007s),sourceequality0(0.007s); wrapper0. Allstderr0;
143logical/12288allocated bytes,10files4links19dirs. Cumulativetask36864bytes.
The controller read the complete writer report and both RED/GREEN/frozen
test output. RED mechanical outer-limit deviation is retained, not erased.

Exact bounded command (once):

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/Volumes/git/legotin/silent-cascade/src .venv/bin/python -B -u -c 'import json,os,pathlib,selectors,subprocess,time
root=pathlib.Path.cwd()
stage=root/".superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-optimizer-cache/main"
assert not stage.exists() and not stage.is_symlink()
stage.mkdir()
for name in ("tmp","xdg","mpl","hypothesis","archive"): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/"tmp"),TMP=str(stage/"tmp"),TEMP=str(stage/"tmp"),XDG_CACHE_HOME=str(stage/"xdg"),XDG_CONFIG_HOME=str(stage/"xdg"),XDG_DATA_HOME=str(stage/"xdg"),MPLCONFIGDIR=str(stage/"mpl"),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/"hypothesis"),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/"archive"),OMP_NUM_THREADS="1")
nodes=["tests/pilot/test_pilot_offline_writes.py::test_offline_environment_derives_inductor_cache_and_ignores_ambient","tests/pilot/test_pilot_offline_writes.py::test_real_scalar_adamw_initializes_only_empty_cache_root","tests/pilot/test_pilot_offline_writes.py::test_optimizer_cache_artifact_denials_latch","tests/pilot/test_pilot_offline_writes.py::test_closed_git_batch_pipe_still_works"]
files=["src/silent_cascade/train/pilot_offline.py","tests/pilot/test_pilot_offline_writes.py"]
commands=[[str(root/".venv/bin/python"),"-B","-m","pytest","--noconftest","-p","no:cacheprovider","-q","--tb=short","--basetemp",str(stage/"pytest"),*nodes],[str(root/".venv/bin/ruff"),"check","--no-cache",*files],[str(root/".venv/bin/ruff"),"format","--check","--no-cache",*files],["git","diff","--check", "--", *files]]
commands.append(["git","diff","--exit-code","13fef3b8be474b7cd77768c5e8ff7b459e611130","--",*files])
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
