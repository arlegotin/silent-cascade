# Task B controller verification

Source/test frozen at9f0c0b3c96494ddc595e184e2b724a772a2eef4e.
Verification in progress; not a completion or full-gate claim.

MAIN-PROCESS independently repeats the exact40 expanded process/privacy/legacy
cases from the implementer report. Fresh operational scratch prefix
`task5-offline-writes-b/main-process`; existing holder29448/token1d4f93d8.
Passed40/40 in12.30s/wall12.927,pytest and wrapper exit0. Stdout100B/stderr0;
30709logical/299008allocated,96files/18links/142dirs. All admitted bounds pass.
Whole64case writer coverage is independently repeated in two disjoint
partitions28+36, due to unchanged4MiB task admission; neither is a worker run.
Writer repeat and final four-file static results follow; review remains pending.

## Controller repeat results at frozen source

- MAIN-WRITERS1:28 passed in17.69s/wall17.941; stdout99B/stderr0;
  124logical/65536allocated,17files/14links/50dirs; pytest/wrapper0.
- MAIN-WRITERS2:36 passed in51.30s/wall51.540; stdout100B/stderr0;
  47563logical/126976allocated,40files/13links/129dirs; pytest/wrapper0.
- MAIN-STATIC: scoped4file Ruff check/format-check, diffcheck and equality to
  9f0c0b3 all exit0.45logical/8192allocated,8files/1dir.
- Combined retained task1404928allocated; all fresh prefixes retained. These
  tests cover40 process/legacy + entire64writer file (one shared case overlaps).
  Not full make verify, genuine worker, full gate, learning result or release.
- Holder fullcheck passed during repeat: scratch261980160/logs2441216,
  spool219099136/cache67153920/metadata52965376, pinned/emergency0;
  frozen retained6195077120. Final custody check still required after review.

## Exact independent commands

Commands were each admitted first in progress.md and executed once in order.
No changed science or storage limits. Source mutation remained stopped.

### task5WritesBMainProcessCommand

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-b/main-process'
assert not stage.exists() and not stage.is_symlink()
assert os.statvfs(root).f_frsize <= 4096
assert 192*1024+8192*(160+192)+4096<=3*1024*1024
stage.mkdir(parents=True)
for name in ('tmp','xdg','mpl','hypothesis','archive'): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/'tmp'),TMP=str(stage/'tmp'),TEMP=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'xdg'),MPLCONFIGDIR=str(stage/'mpl'),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'),OMP_NUM_THREADS='1',XDG_CONFIG_HOME=str(stage/'xdg'),XDG_DATA_HOME=str(stage/'xdg'),PYTHONHASHSEED='0',SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH='1')
names=["tests/pilot/test_pilot_offline_process.py::test_safe_intent_write_allowance_is_optional_and_strict","tests/pilot/test_pilot_offline_process.py::test_measurement_serializes_exact_write_allowance_before_launch","tests/pilot/test_pilot_offline_process.py::test_measurement_rejects_nonexact_write_allowance_before_launch","tests/pilot/test_pilot_offline_process.py::test_bootstrap_postcheck_exposes_caught_write_denial","tests/pilot/test_pilot_offline_process.py::test_bootstrap_requires_installed_allowance_before_worker","tests/pilot/test_pilot_offline_process.py::test_bootstrap_digest_binds_write_allowance","tests/pilot/test_pilot_offline_process.py::test_process_reader_rejects_changed_intent_allowance","tests/pilot/test_pilot_offline_process.py::test_privacy_intent_hashes_replace_paths","tests/pilot/test_pilot_offline_process.py::test_privacy_bootstrap_accepts_bound_hash_identities","tests/pilot/test_pilot_offline_process.py::test_privacy_nonempty_failure_retains_closed_outcome","tests/pilot/test_pilot_offline_process.py::test_capture_bounds_both_pipes","tests/pilot/test_pilot_offline_process.py::test_capture_timeout_joins_child","tests/pilot/test_pilot_offline_process.py::test_fresh_nested_audit_denies_changed_launch_authority","tests/pilot/test_pilot_offline_process.py::test_bootstrap_rejects_changed_launch_authority","tests/pilot/test_pilot_offline_process.py::test_reuse_rejects_incomplete_before_authentication","tests/pilot/test_pilot_offline_process.py::test_collection_fences_incomplete_before_authentication","tests/pilot/test_pilot_offline_process.py::test_measurement_persists_failed_invalid_report_and_fences_reuse","tests/pilot/test_pilot_offline_process.py::test_log_publication_failure_never_publishes_completion","tests/pilot/test_pilot_offline_writes.py::test_real_publishers_exact_capacity","tests/pilot/test_pilot_offline.py::test_offline_child_cannot_write_outside_counted_workspace"]
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
  if time.monotonic()-start>240: p.kill(); raise RuntimeError('timeout')
finally:
 p.wait(); sel.close()
 for name,payload in buffers.items():
  with (stage/(name+'.log')).open('xb') as f: f.write(payload)
for name,payload in buffers.items(): print(name+':\n'+payload.decode())
items=list(stage.rglob('*')); files=[q for q in items if q.is_file() and not q.is_symlink()]
counts=dict(exit=p.returncode,seconds=round(time.monotonic()-start,3),logical=sum(q.stat().st_size for q in files),allocated=sum(q.lstat().st_blocks*512 for q in [stage,*items]),files=len(files),links=sum(q.is_symlink() for q in items),dirs=sum(q.is_dir() and not q.is_symlink() for q in [stage,*items]))
print(counts)
assert counts['logical']<192*1024 and counts['allocated']<=3*1024*1024
assert counts['files']+counts['links']<=160 and counts['dirs']<=192
assert p.returncode == 0
PY
```

### task5BMainWriters1Command

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-b/main-writers1'
assert not stage.exists() and not stage.is_symlink()
assert os.statvfs(root).f_frsize <= 4096
assert 128*1024+8192*(64+120)+4096<=2*1024*1024
stage.mkdir(parents=True)
for name in ('tmp','xdg','mpl','hypothesis','archive'): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/'tmp'),TMP=str(stage/'tmp'),TEMP=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'xdg'),MPLCONFIGDIR=str(stage/'mpl'),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'),OMP_NUM_THREADS='1',XDG_CONFIG_HOME=str(stage/'xdg'),XDG_DATA_HOME=str(stage/'xdg'),PYTHONHASHSEED='0',SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH='1')
names=["tests/pilot/test_pilot_offline_writes.py::test_rows_substitution_is_denied_before_final_link","tests/pilot/test_pilot_offline_writes.py::test_temp_substitution_is_denied_before_chmod","tests/pilot/test_pilot_offline_writes.py::test_allowance_rejects_invalid_values","tests/pilot/test_pilot_offline_writes.py::test_install_requires_exact_allowance_type","tests/pilot/test_pilot_offline_writes.py::test_install_cannot_replace_or_enlarge_allowance","tests/pilot/test_pilot_offline_writes.py::test_late_consumer_binding_is_denied","tests/pilot/test_pilot_offline_writes.py::test_real_publishers_exact_capacity","tests/pilot/test_pilot_offline_writes.py::test_rows_stream_and_final_link","tests/pilot/test_pilot_offline_writes.py::test_closed_git_batch_pipe_still_works","tests/pilot/test_pilot_offline_writes.py::test_transitive_evaluation_publisher_keeps_token_lifetime","tests/pilot/test_pilot_offline_writes.py::test_publication_one_byte_short_has_no_temp_or_mutation","tests/pilot/test_pilot_offline_writes.py::test_create_collision_reserves_incoming_bytes","tests/pilot/test_pilot_offline_writes.py::test_atomic_replace_and_create_preserve_original_behavior","tests/pilot/test_pilot_offline_writes.py::test_second_atomic_temp_denied_and_owned_cleanup_allowed","tests/pilot/test_pilot_offline_writes.py::test_pending_and_final_hardlink_both_charge_allocation"]
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
  if time.monotonic()-start>1500: p.kill(); raise RuntimeError('timeout')
finally:
 p.wait(); sel.close()
 for name,payload in buffers.items():
  with (stage/(name+'.log')).open('xb') as f: f.write(payload)
for name,payload in buffers.items(): print(name+':\n'+payload.decode())
items=list(stage.rglob('*')); files=[q for q in items if q.is_file() and not q.is_symlink()]
counts=dict(exit=p.returncode,seconds=round(time.monotonic()-start,3),logical=sum(q.stat().st_size for q in files),allocated=sum(q.lstat().st_blocks*512 for q in [stage,*items]),files=len(files),links=sum(q.is_symlink() for q in items),dirs=sum(q.is_dir() and not q.is_symlink() for q in [stage,*items]))
print(counts)
assert counts['logical']<128*1024 and counts['allocated']<=2*1024*1024
assert counts['files']+counts['links']<=64 and counts['dirs']<=120
assert p.returncode == 0
PY
```

### task5BMainWriters2Command

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-b/main-writers2'
assert not stage.exists() and not stage.is_symlink()
assert os.statvfs(root).f_frsize <= 4096
assert 500*1024+8192*(96+160)+4096<=2560*1024
stage.mkdir(parents=True)
for name in ('tmp','xdg','mpl','hypothesis','archive'): (stage/name).mkdir()
env=dict(os.environ,TMPDIR=str(stage/'tmp'),TMP=str(stage/'tmp'),TEMP=str(stage/'tmp'),XDG_CACHE_HOME=str(stage/'xdg'),MPLCONFIGDIR=str(stage/'mpl'),HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'),OMP_NUM_THREADS='1',XDG_CONFIG_HOME=str(stage/'xdg'),XDG_DATA_HOME=str(stage/'xdg'),PYTHONHASHSEED='0',SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH='1')
names=["tests/pilot/test_pilot_offline_writes.py::test_closed_names_latch_before_publication","tests/pilot/test_pilot_offline_writes.py::test_lowered_inventory_ceiling_denies_before_creation","tests/pilot/test_pilot_offline_writes.py::test_second_weights_digest_denied","tests/pilot/test_pilot_offline_writes.py::test_seventeenth_crash_stem_denied","tests/pilot/test_pilot_offline_writes.py::test_last_episode_ordinal_and_crash_pair_are_admitted","tests/pilot/test_pilot_offline_writes.py::test_rows_descriptor_and_stream_escapes_latch","tests/pilot/test_pilot_offline_writes.py::test_unowned_writes_and_paths_latch","tests/pilot/test_pilot_offline_writes.py::test_external_hardlink_fails_at_installation","tests/pilot/test_pilot_offline_writes.py::test_preexisting_symlink_fails_at_installation","tests/pilot/test_pilot_offline_writes.py::test_real_font_manager_cache_and_lock","tests/pilot/test_pilot_offline_writes.py::test_real_font_json_dump_overflow_latches_and_cleans_lock","tests/pilot/test_pilot_offline_writes.py::test_utf8_stream_admits_encoded_length"]
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
  if time.monotonic()-start>1500: p.kill(); raise RuntimeError('timeout')
finally:
 p.wait(); sel.close()
 for name,payload in buffers.items():
  with (stage/(name+'.log')).open('xb') as f: f.write(payload)
for name,payload in buffers.items(): print(name+':\n'+payload.decode())
items=list(stage.rglob('*')); files=[q for q in items if q.is_file() and not q.is_symlink()]
counts=dict(exit=p.returncode,seconds=round(time.monotonic()-start,3),logical=sum(q.stat().st_size for q in files),allocated=sum(q.lstat().st_blocks*512 for q in [stage,*items]),files=len(files),links=sum(q.is_symlink() for q in items),dirs=sum(q.is_dir() and not q.is_symlink() for q in [stage,*items]))
print(counts)
assert counts['logical']<500*1024 and counts['allocated']<=2560*1024
assert counts['files']+counts['links']<=96 and counts['dirs']<=160
assert p.returncode == 0
PY
```

### task5BMainStaticCommand

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root=pathlib.Path.cwd()
stage=root/'.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-b/main-static'
assert not stage.exists() and not stage.is_symlink()
assert 128*1024+8192*(16+24)+4096<=512*1024
stage.mkdir(parents=True)
files=["src/silent_cascade/train/pilot_offline.py","src/silent_cascade/train/pilot_offline_process.py","tests/pilot/test_pilot_offline_process.py","tests/pilot/test_pilot_offline_writes.py"]
commands=[['.venv/bin/ruff','check','--no-cache',*files],['.venv/bin/ruff','format','--check','--no-cache',*files],['git','diff','--check'],['git','diff','--exit-code','9f0c0b3','--',*files]]
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
