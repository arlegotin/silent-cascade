# Task5 direct offline diagnostic 02

Status: **passed genuine standalone offline diagnostic**, not a Phase4 pilot
or learning-support gate. The owner released/exited0 after full frozen-history
verification. Source fix13fef3b had passed independent review and five focused
cases/statics twice before this separately admitted execution.

Reason for this fresh attempt: Diagnostic01 reproduced a specific ordinary
AdamW initialization failure; the one-line closed-cache mapping now passes the
real boundary regression while forbidden cache files/directories remain denied.
All previous evidence remains unchanged. The active plan specifies this
bounded follow-up under standing approval; it is not the production pilot.

Changed from Diagnostic01: fresh public/admin roots and private attempt; the
private parent already exists. P_private1110016/P_public114688; child16777216;
reservation spool16891904/logs2158592/metadata33562624/scratch524288, other0,
total53137408. Same child102names/9dirs,180s/256KiBpipe/16KiBrecord limits.
The parent asserts current peaks and absence before preparation, awaits one
`run`, then closes custody and performs the full frozen check before release.

Latest completed task-owner check: spool219123712/cache67153920/metadata52965376/
scratch264118272/logs2695168/pinned0/emergency0; frozen6195077120 intact. This
is preflight evidence, not a substitute for the new ledger admission.

## Fixed parent code

Use the same closed shell environment as Diagnostic01, `python -B -u -c`, and
the exact frozen HEAD as its sole argument. No scientific model or optimizer
executes in the parent; no private raw bytes printed. The worker remains unchanged
except for the reviewed environment correction.

```python
import hashlib,json,os,stat,sys,time
from pathlib import Path
import psutil
from silent_cascade.archive.ledger import _StorageBudget
from silent_cascade.archive.types import ArchivePolicy
from silent_cascade.train.pilot_offline_process import OfflineProcessLimits,prepare_offline_process_custody,_custody_paths,_publication_peaks
from silent_cascade.train.pilot_offline_writes import OfflineWriteAllowance
from silent_cascade.train.pilot_offline import measure_pilot_offline
def emit(value):
 raw=json.dumps(value,sort_keys=True)
 assert len(raw.encode())<=8192
 print(raw,flush=True)
root=Path.cwd()
workspace=root/".superpowers/sdd/2026-09-17-phase-4-r2-archive/operational"
output=workspace/"spool/task5-real-offline-02/final/offline"
admin=workspace/"scratch/task5-real-offline-02"
attempt="6be71f5c342d4fa9a24c9d0f116a8bd2"
expected_revision=sys.argv[1]
assert len(expected_revision)==40
limits=OfflineProcessLimits()
allowance=OfflineWriteAllowance(16777216,102,9)
amounts=dict(spool=16891904,cache=0,pinned=0,metadata=33562624,scratch=524288,logs=2158592,emergency=0)
budget=_StorageBudget(workspace=workspace,policy=ArchivePolicy())
state=budget._state()
assert not state["reservations"] and "pending" not in state["engineering"]
assert state["engineering"]["snapshot"]["allocated"]==6195077120
assert not output.parent.parent.exists() and not admin.exists()
private,state=_custody_paths(budget,output,attempt)
assert not private.exists() and private.parent.is_dir()
assert dict(_publication_peaks(budget,state,output,private,limits))==dict(logs=1110016,spool=114688)
assert os.statvfs(workspace).f_frsize<=4096
assert "torch" not in sys.modules
owner=psutil.Process(os.getpid())
emit(dict(phase="pre-admission",pid=owner.pid,create_time=owner.create_time(),expected_revision=expected_revision,amounts=amounts))
failed=True
with budget._scoped_reservation(admission=dict(operation="task5-real-offline-02",source_commit=expected_revision,child_bytes=16777216,child_names=102,child_directories=9,timeout_seconds=180,attempt=attempt,scope="one genuine same-owner offline diagnostic; no inherited authority/provider/fullgate/productionpilot"),child_inheritable=False,**amounts) as admission:
 emit(dict(phase="admitted",token=admission.token,before=admission.before))
 admin.mkdir()
 for name in ("tmp","cache","mpl"): (admin/name).mkdir()
 for name in ("TMPDIR","TMP","TEMP"): os.environ[name]=str(admin/"tmp")
 for name in ("XDG_CACHE_HOME","XDG_CONFIG_HOME","XDG_DATA_HOME"): os.environ[name]=str(admin/"cache")
 os.environ["MPLCONFIGDIR"]=str(admin/"mpl")
 with prepare_offline_process_custody(budget=budget,admission=admission,output_dir=output,limits=limits,attempt_id=attempt) as custody:
  assert dict(custody.category_peaks)==dict(logs=1110016,spool=114688)
  current=budget.check_scoped(admission.retained)
  active=budget._state()["reservations"][admission.token]
  for category,needed in dict(spool=16891904,logs=1110016).items():
   remaining=active["amounts"][category]-max(0,current[category]-active["before"][category])
   # Created private directories are already retained and covered by original P.
   remaining_needed=dict(_publication_peaks(budget,budget._state(),output,private,limits))[category]
   if category=="spool": remaining_needed+=allowance.allocated_bytes
   assert remaining>=remaining_needed
  emit(dict(phase="ready",category_peaks=dict(custody.category_peaks),allocated=current))
  for line in sys.stdin:
   if line.strip()!="run": break
   started=time.monotonic()
   try:
    report=measure_pilot_offline(output_dir=output,process_limits=limits,process_custody=custody,write_allowance=allowance)
    assert report.source_commit==expected_revision
    assert report.foundation_model_calls==0 and report.network_attempts==0
    emit(dict(phase="diagnostic-completed",seconds=round(time.monotonic()-started,3),report=report.model_dump()))
    failed=False
   except Exception as error:
    emit(dict(phase="diagnostic-failed",seconds=round(time.monotonic()-started,3),error_type=type(error).__name__))
   break
  result_path=output/"process-result.json"
  if result_path.exists():
   raw=result_path.read_bytes()
   assert len(raw)<=limits.record_bytes
   result=json.loads(raw)
   emit(dict(phase="public-process-result",sha256=hashlib.sha256(raw).hexdigest(),result=result))
  def inventory(path):
   paths=[path,*path.rglob("*")]
   infos=[p.lstat() for p in paths]
   assert all(s.st_dev==budget.device and not stat.S_ISLNK(s.st_mode) for s in infos)
   return dict(allocated=sum(s.st_blocks*512 for s in infos),logical=sum(s.st_size for s in infos if stat.S_ISREG(s.st_mode)),files=sum(stat.S_ISREG(s.st_mode) for s in infos),directories=sum(stat.S_ISDIR(s.st_mode) for s in infos))
  emit(dict(phase="retained-inventories",public=inventory(output.parent.parent) if output.parent.parent.exists() else None,private=inventory(private),admin=inventory(admin)))
 emit(dict(phase="final-check-start"))
 emit(dict(phase="final-check",allocated=budget.check()))
emit(dict(phase="released",diagnostic_passed=not failed))
sys.exit(1 if failed else 0)
```

## Observed result

Executed exactly once at `2e5b96dbc8a436ff039a1efb5836f2c7236c2e36`.
Owner PID82437/create-time1789799877.814541; token
`977357be9c5347bd99de470b4afa4018`; worker PID82514. Before categories matched
the preflight table. No second owner, checkout, provider or broad gate ran.

Measurement completed in **60.250 seconds**; worker exit0, EOF, no stdout or
stderr. The genuine worker performed one CPU training update,16 autonomous
debug evaluations, one verified neural replay and one report build. Its report
records backwardMACs337667328, foundation-model calls0, network attempts0,
optional-import attempts0, forbidden modules empty. These are engineering
diagnostic results, not model-accuracy or scientific-support results.

Authenticated identities/hashes:

- Process intent: `fd9120b5c3d00ec71119ce61ef5a484a9d1402445ae1b4911cdb66f98b443422`
- Process result: `a54d85c3f947642dcff98b4e0348701752261759d0157d76bf7850336260f593`
- Offline report: `7aa16ec2c77ac40e83a2dbdf820587eac93a9730240deedfb7c6ba6979dfcb41`
- Executed source: `e264f35374335ce2f4c93c2e9d8f214b4c0b41476325353927fdfd1e4762fd24`
- Config: `f9b1fa70f4bbb075fef516b491125ff975cdb36301639741a44ad1f92956069b`
- Manifest: `09c049b27f26bc4b13e89085ffed602ec199d9e63a3826c8540480036e567975`
- Weights: `afd82f1dd49ecb053125ef1eb369f8abeb71496566847a6dde30c440204bf541`
- Replay: `c266e0f87b98e1f271695dca891e7a2b5dde2b99acc9ac7e102f721f5e2c1515`
- Generated report: `d9cf6947fce30185dbd69e4a051f918e624ead4c61380d770b9146a4368be13f`

Actual retained public run:11939840allocated/11779480logical bytes,70files/
11directories (includes run/final ancestors outside the nine-directory child
grammar). Private binding:4096allocated/227logical,1file/1directory. Parent
administrative root:0allocated,0files/4dirs. Limits remained unchanged; no
failure/root reuse or cleanup occurred.

Final global categories: spool231063552/cache67153920/metadata52965376/
scratch264118272/logs2699264/pinned0/emergency0. Full frozen6195077120 check
passed, then owner exited0 and released the reservation. The controller also
independently read the public report and checked the raw intent/result/report/
source/weights/replay/generated-report SHA-256 values without neural reruns.

Still outstanding: explicit allowance forwarding into the main workflow,
same-owner versus inherited supervisor integration, full Task5/Task7 gates,
`make verify` and the production Phase4 pilot/learning acceptance.
