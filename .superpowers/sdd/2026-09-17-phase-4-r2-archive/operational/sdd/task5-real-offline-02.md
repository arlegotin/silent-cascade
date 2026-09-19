# Task5 direct offline diagnostic 02

Status: prepared, not launched. Optimizer-cache review passed with no required
fixes; its owner released/exited0 after full frozen verification. New exact
same-owner global admission remains required. Source fix13fef3b passed five
focused cases and statics twice, with frozen-source equality.

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
