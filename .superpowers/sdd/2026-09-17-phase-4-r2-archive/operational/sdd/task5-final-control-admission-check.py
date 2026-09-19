import json,os,pathlib,selectors,subprocess,sys,time
root=pathlib.Path.cwd()
label=sys.argv[1]
assert label in {"red1","green1","green2","main1"}
stage=root/".superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-final-control-admission"/label
assert not stage.exists() and not stage.is_symlink()
stage.mkdir(parents=True)
for name in ("tmp","xdg","mpl","hypothesis","archive"): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/"tmp"),TMP=str(stage/"tmp"),TEMP=str(stage/"tmp"),XDG_CACHE_HOME=str(stage/"xdg"),XDG_CONFIG_HOME=str(stage/"xdg"),XDG_DATA_HOME=str(stage/"xdg"),MPLCONFIGDIR=str(stage/"mpl"),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/"hypothesis"),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/"archive"),OMP_NUM_THREADS="1")
nodes=["tests/pilot/test_pilot_final_control_admission.py::test_final_control_bound_is_fixed","tests/pilot/test_pilot_final_control_admission.py::test_final_control_dispatch_uses_exact_remaining_capacity","tests/pilot/test_pilot_final_control_admission.py::test_final_control_producer_sends_only_closed_operation","tests/pilot/test_pilot_final_control_admission.py::test_final_control_admission_orders_real_evaluate_branching"]
files=["src/silent_cascade/archive/producer.py","src/silent_cascade/train/pilot_checks.py","tests/pilot/test_pilot_final_control_admission.py"]
commands=[[str(root/".venv/bin/python"),"-B","-m","pytest","--noconftest","-p","no:cacheprovider","-q","--tb=short","--basetemp",str(stage/"pytest"),*nodes],[str(root/".venv/bin/ruff"),"check","--no-cache",*files],[str(root/".venv/bin/ruff"),"format","--check","--no-cache",*files],["git","diff","--check", "--", *files]]
if label=="red1": commands=commands[:1]
if label in {"green1","green2"}: commands.insert(0,[str(root/".venv/bin/ruff"),"format","--no-cache",*files])
if label=="main1":
 assert len(sys.argv)==3
 commands.append(["git","diff","--exit-code",sys.argv[2],"--",*files])
statuses=[]; total_output=0
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
    total_output+=len(chunk)
    if len(buffers[key.data])+len(chunk)>32768 or total_output>65536:
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
assert counts["allocated"]<=524288 and counts["logical"]<524288
assert counts["files"]+counts["links"]<=(8 if label=="red1" else 20) and counts["dirs"]<=32
assert statuses==[1] if label=="red1" else len(statuses)==len(commands) and not any(statuses)
