# Task5 direct offline diagnostic 01

Status: preflight; not launched.

Read-only admission reviewer `/root/offline_diagnostic_admission_review`
confirmed the public/private arithmetic and conservative outer reservation,
current empty reservation set, absent private root, 4096-byte granule, same
device, trusted Git 2.49.0, and no definite fixed-path launch blocker. It did
not execute the diagnostic or certify success within the limits. Parent
scratch/status bounds are cooperative, not kernel quotas; the admitted parent
is fixed and creates only four empty administrative directories. The actual
child has the reviewed irreversible pre-write adapters. No source/module
import, scientific computation or provider operation ran during this review.

Controller corrected the review's subprocess count: the worker revalidates
again before its final report. The successful fixed API path has seven trusted
local Git calls (parent: two version + one revision; child: four revisions),
one worker and one owner: nine processes total, at most three concurrently.
No additional application/control children are admitted. The fixed parent
status code is 5089 bytes, emits at most nine 8192-byte JSON lines, and keeps
child streams only in the existing private-capture path. Runtime duration of
the worker excludes the slower full frozen-history admission/release checks.

Task C is complete at source `0aee7bd`; closure/evidence commit `636c9f5`.
The next bounded check is specified in the active R2 plan's “one direct
same-owner diagnostic” subsection. Existing user approval of Phase4/Task5 and
standing approval cover this reversible, in-scope qualification refinement.
No new implementation or scientific-protocol change is proposed.

The parent owns the original scoped reservation and private custody for their
entire lifetime. `child_inheritable=False`; no capability is passed to another
operational process. The existing diagnostic itself starts its fenced worker.
No source copy, test suite, provider call, archive publication/eviction, full
gate, production pilot or inherited-supervisor implementation is included.

Prospective increments: spool16891904, logs2162688, metadata33562624,
scratch524288; total53141504 bytes. Exact fresh roots/attempt, public/private P,
child and process caps, failure retention, and source freeze are in the plan.
Admission must fail before launch if current bindings, retained counts or
publication peaks do not fit those fixed amounts.

The controller has read the measurement, custody, publication-peaks, existing
ledger ownership and worker paths. A separate read-only reviewer is checking
the same launch and accounting, including the parent import/write surface.

## Exact operational parent

The closed shell environment is `PATH=/opt/homebrew/bin:/usr/bin:/bin`,
`LANG=C.UTF-8`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`,
`PYTHONPATH=<checkout>/src`, TMP/TEMP/XDG/MPL initially the existing counted
operational scratch directory. Run `.venv/bin/python -B -u -c <code> <frozen
HEAD>` once with a PTY. The controller sends exactly `run` after `ready` and
retains the bounded status output. No private raw bytes are printed.

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
output=workspace/"spool/task5-real-offline-01/final/offline"
admin=workspace/"scratch/task5-real-offline-01"
attempt="d7c2cb464e9043c6a11a77e6dc1a3e51"
expected_revision=sys.argv[1]
assert len(expected_revision)==40
limits=OfflineProcessLimits()
allowance=OfflineWriteAllowance(16777216,102,9)
amounts=dict(spool=16891904,cache=0,pinned=0,metadata=33562624,scratch=524288,logs=2162688,emergency=0)
budget=_StorageBudget(workspace=workspace,policy=ArchivePolicy())
state=budget._state()
assert not state["reservations"] and "pending" not in state["engineering"]
assert state["engineering"]["snapshot"]["allocated"]==6195077120
assert not output.parent.parent.exists() and not admin.exists()
private,state=_custody_paths(budget,output,attempt)
assert not private.exists() and not private.parent.exists()
assert dict(_publication_peaks(budget,state,output,private,limits))==dict(logs=1114112,spool=114688)
assert os.statvfs(workspace).f_frsize<=4096
assert "torch" not in sys.modules
owner=psutil.Process(os.getpid())
emit(dict(phase="pre-admission",pid=owner.pid,create_time=owner.create_time(),expected_revision=expected_revision,amounts=amounts))
failed=True
with budget._scoped_reservation(admission=dict(operation="task5-real-offline-01",source_commit=expected_revision,child_bytes=16777216,child_names=102,child_directories=9,timeout_seconds=180,attempt=attempt,scope="one genuine same-owner offline diagnostic; no inherited authority/provider/fullgate/productionpilot"),child_inheritable=False,**amounts) as admission:
 emit(dict(phase="admitted",token=admission.token,before=admission.before))
 admin.mkdir()
 for name in ("tmp","cache","mpl"): (admin/name).mkdir()
 for name in ("TMPDIR","TMP","TEMP"): os.environ[name]=str(admin/"tmp")
 for name in ("XDG_CACHE_HOME","XDG_CONFIG_HOME","XDG_DATA_HOME"): os.environ[name]=str(admin/"cache")
 os.environ["MPLCONFIGDIR"]=str(admin/"mpl")
 with prepare_offline_process_custody(budget=budget,admission=admission,output_dir=output,limits=limits,attempt_id=attempt) as custody:
  assert dict(custody.category_peaks)==dict(logs=1114112,spool=114688)
  current=budget.check_scoped(admission.retained)
  active=budget._state()["reservations"][admission.token]
  for category,needed in dict(spool=16891904,logs=1114112).items():
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
