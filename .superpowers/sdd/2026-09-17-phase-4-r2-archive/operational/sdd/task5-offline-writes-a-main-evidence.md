# Task5 standalone writer controller evidence

Initial source freeze:14abc50; base3820408; reviewed identity fix07e9394.
Standalone TaskA review is complete; subsequent production work remains open.
No production worker, fit, evaluation,
replay, providers or archived diagnostic was executed. All prefixes are retained.

## Fresh static verification

The following four files match commit14abc50 exactly; scoped Ruff lint and
format --check both exit0; git diff --check exits0.

- src/silent_cascade/train/pilot_offline_writes.py
- src/silent_cascade/train/pilot_offline.py
- src/silent_cascade/train/pilot_evidence.py
- tests/pilot/test_pilot_offline_writes.py

## Matrix and current results

MAIN1:33cases passed23.91s; wall24.165s, pytestexit0, empty stderr.
111logical/24576allocated bytes,7files/7links/56dirs.
MAIN2:remaining29new-module+4legacy cases passed42.71s; wall42.933s,
pytestexit0, empty stderr;47585logical/155648allocated bytes,47files/22links/
121dirs. All66cases passed on14abc50; no scientific result is inferred.
Both groups together cover all62new plus4legacy cases on the frozen source.
Review package:review-task5-offline-writes-a.diff (52291chars).

All12prefixes, including failures, retain503808allocated bytes. Fresh stdout
SHA256 values (both stderr files are empty):

- MAIN1:6b3efa832566f19f5eee2b3f9226f12635d33e89ac37d09be3a2db9955568275
- MAIN2:186ec14a196c965fbe4deb9bd30de424877a557a54bf432ba7ce411a91c3cf39

At the initial66-case checkpoint, independent review and full accounting were
pending. Their subsequent outcomes and the scoped fix are recorded below.

Subsequent owner fullcheck passed: scratch258756608/logs2117632/spool219099136/
cache67153920/metadata52965376,pinned/emergency0. Frozen retained history
6195077120 remains unchanged. Review is investigating identity checks; this
full test result is not a claim that review findings are resolved.

Admission MAIN1:3MiB/<256KiB logical/160names/180dirs/21children/690s.
Admission MAIN2:3.5MiB/<512KiB logical/180names/200dirs/31children/990s.
Child limits64KiB except exact crash128KiB/font256KiB cases. Source4096B,
pipes2048B each for new controls; parent log files16KiB each. All within owner
65407's existing4MiB task increment; no second reservation/category change.

## Exact executed commands

These used prefixes must never be reused. A repeat requires separately admitted
fresh stage names. Outer shell exit alone is not pytest success; inspect its
recorded child exit and all stderr.


## Fix round1 controller verification

Source07e9394 changes only two pre-mutation identity checks and two bounded
real-inode substitution regressions. Fresh six-case covering run passed7.39s
(wall7.630), pytestexit0/stderrempty. Stage111logical/45056allocated bytes,
12files/6links/28dirs. All15prefixes retain622592allocated bytes. Scoped
twofile Ruff lint/format, diff check and source equality to07e9394 passed.
Earlier66case run remains evidence for14abc50; this is amended-path coverage,
not a claim of a new complete68case or full project run.

Post-fix full ownercheck passed: scratch258875392/logs2162688/spool219099136/
cache67153920/metadata52965376,pinned/emergency0; frozen retained6195077120
unchanged. Independent scoped re-review then approved both corrected checks,
with no new findings. Release of owner65407 was requested after that review;
the final release record belongs in the operational ledger.

Exact executed command (usedprefix; never reuse):

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root = pathlib.Path.cwd()
stage = root / '.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-a/fix1-main'
assert not stage.exists()
stage.mkdir(parents=True)
for name in ('tmp', 'xdg', 'mpl', 'hypothesis', 'archive'):
    (stage / name).mkdir()
env = dict(os.environ, TMPDIR=str(stage/'tmp'), TMP=str(stage/'tmp'),
    TEMP=str(stage/'tmp'), XDG_CACHE_HOME=str(stage/'xdg'),
    XDG_CONFIG_HOME=str(stage/'xdg'), XDG_DATA_HOME=str(stage/'xdg'),
    MPLCONFIGDIR=str(stage/'mpl'), HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),
    SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'), OMP_NUM_THREADS='1')
selected = ["tests/pilot/test_pilot_offline_writes.py::test_rows_substitution_is_denied_before_final_link","tests/pilot/test_pilot_offline_writes.py::test_temp_substitution_is_denied_before_chmod","tests/pilot/test_pilot_offline_writes.py::test_rows_stream_and_final_link","tests/pilot/test_pilot_offline_writes.py::test_pending_and_final_hardlink_both_charge_allocation","tests/pilot/test_pilot_offline_writes.py::test_atomic_replace_and_create_preserve_original_behavior","tests/pilot/test_pilot_offline_writes.py::test_second_atomic_temp_denied_and_owned_cleanup_allowed"]
args = [str(root/'.venv/bin/python'), '-B', '-m', 'pytest', '-p', 'no:cacheprovider',
    '--noconftest', '-q', '--tb=short', '--basetemp', str(stage/'pytest'), *selected]
start = time.monotonic()
p = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sel = selectors.DefaultSelector()
buffers = {}
for name, stream in [('stdout', p.stdout), ('stderr', p.stderr)]:
    sel.register(stream, selectors.EVENT_READ, name)
    buffers[name] = bytearray()
try:
    while sel.get_map():
        for key, _ in sel.select(1):
            chunk = os.read(key.fileobj.fileno(), 4096)
            if not chunk:
                sel.unregister(key.fileobj)
                continue
            if len(buffers[key.data]) + len(chunk) > 16384:
                p.kill()
                raise RuntimeError('output bound')
            buffers[key.data].extend(chunk)
        if time.monotonic() - start > 240:
            p.kill()
            raise RuntimeError('timeout')
finally:
    p.wait()
    sel.close()
    for name, payload in buffers.items():
        with (stage/(name+'.log')).open('xb') as stream:
            stream.write(payload)
for name, payload in buffers.items():
    print(name + ':\n' + payload.decode())
items = list(stage.rglob('*'))
files = [q for q in items if q.is_file() and not q.is_symlink()]
print(dict(exit=p.returncode, seconds=round(time.monotonic()-start, 3),
    logical=sum(q.stat().st_size for q in files),
    allocated=sum(q.lstat().st_blocks*512 for q in [stage, *items]), files=len(files),
    links=sum(q.is_symlink() for q in items),
    dirs=sum(q.is_dir() and not q.is_symlink() for q in [stage, *items])))
PY
```

### MAIN1

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root = pathlib.Path.cwd()
stage = root / '.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-a/main1'
assert not stage.exists()
stage.mkdir(parents=True)
for name in ('tmp', 'xdg', 'mpl', 'hypothesis', 'archive'):
    (stage / name).mkdir()
env = dict(os.environ, TMPDIR=str(stage/'tmp'), TMP=str(stage/'tmp'),
    TEMP=str(stage/'tmp'), XDG_CACHE_HOME=str(stage/'xdg'),
    XDG_CONFIG_HOME=str(stage/'xdg'), XDG_DATA_HOME=str(stage/'xdg'),
    MPLCONFIGDIR=str(stage/'mpl'), HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),
    SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'), OMP_NUM_THREADS='1')
selected = ["tests/pilot/test_pilot_offline_writes.py::test_allowance_rejects_invalid_values","tests/pilot/test_pilot_offline_writes.py::test_transitive_evaluation_publisher_keeps_token_lifetime","tests/pilot/test_pilot_offline_writes.py::test_publication_one_byte_short_has_no_temp_or_mutation","tests/pilot/test_pilot_offline_writes.py::test_create_collision_reserves_incoming_bytes","tests/pilot/test_pilot_offline_writes.py::test_closed_names_latch_before_publication","tests/pilot/test_pilot_offline_writes.py::test_lowered_inventory_ceiling_denies_before_creation","tests/pilot/test_pilot_offline_writes.py::test_second_weights_digest_denied","tests/pilot/test_pilot_offline_writes.py::test_unowned_writes_and_paths_latch"]
args = [str(root/'.venv/bin/python'), '-B', '-m', 'pytest', '-p', 'no:cacheprovider',
    '--noconftest', '-q', '--tb=short', '--basetemp', str(stage/'pytest'), *selected]
start = time.monotonic()
p = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sel = selectors.DefaultSelector()
buffers = {}
for name, stream in [('stdout', p.stdout), ('stderr', p.stderr)]:
    sel.register(stream, selectors.EVENT_READ, name)
    buffers[name] = bytearray()
try:
    while sel.get_map():
        for key, _ in sel.select(1):
            chunk = os.read(key.fileobj.fileno(), 4096)
            if not chunk:
                sel.unregister(key.fileobj)
                continue
            if len(buffers[key.data]) + len(chunk) > 16384:
                p.kill()
                raise RuntimeError('output bound')
            buffers[key.data].extend(chunk)
        if time.monotonic() - start > 690:
            p.kill()
            raise RuntimeError('timeout')
finally:
    p.wait()
    sel.close()
    for name, payload in buffers.items():
        with (stage/(name+'.log')).open('xb') as stream:
            stream.write(payload)
for name, payload in buffers.items():
    print(name + ':\n' + payload.decode())
items = list(stage.rglob('*'))
files = [q for q in items if q.is_file() and not q.is_symlink()]
print(dict(exit=p.returncode, seconds=round(time.monotonic()-start, 3),
    logical=sum(q.stat().st_size for q in files),
    allocated=sum(q.lstat().st_blocks*512 for q in [stage, *items]), files=len(files),
    links=sum(q.is_symlink() for q in items),
    dirs=sum(q.is_dir() and not q.is_symlink() for q in [stage, *items])))
PY
```

### MAIN2

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root = pathlib.Path.cwd()
stage = root / '.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-a/main2'
assert not stage.exists()
stage.mkdir(parents=True)
for name in ('tmp', 'xdg', 'mpl', 'hypothesis', 'archive'):
    (stage / name).mkdir()
env = dict(os.environ, TMPDIR=str(stage/'tmp'), TMP=str(stage/'tmp'),
    TEMP=str(stage/'tmp'), XDG_CACHE_HOME=str(stage/'xdg'),
    XDG_CONFIG_HOME=str(stage/'xdg'), XDG_DATA_HOME=str(stage/'xdg'),
    MPLCONFIGDIR=str(stage/'mpl'), HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),
    SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'), OMP_NUM_THREADS='1')
selected = ["tests/pilot/test_pilot_offline_writes.py::test_install_requires_exact_allowance_type","tests/pilot/test_pilot_offline_writes.py::test_install_cannot_replace_or_enlarge_allowance","tests/pilot/test_pilot_offline_writes.py::test_late_consumer_binding_is_denied","tests/pilot/test_pilot_offline_writes.py::test_real_publishers_exact_capacity","tests/pilot/test_pilot_offline_writes.py::test_rows_stream_and_final_link","tests/pilot/test_pilot_offline_writes.py::test_closed_git_batch_pipe_still_works","tests/pilot/test_pilot_offline_writes.py::test_atomic_replace_and_create_preserve_original_behavior","tests/pilot/test_pilot_offline_writes.py::test_second_atomic_temp_denied_and_owned_cleanup_allowed","tests/pilot/test_pilot_offline_writes.py::test_pending_and_final_hardlink_both_charge_allocation","tests/pilot/test_pilot_offline_writes.py::test_seventeenth_crash_stem_denied","tests/pilot/test_pilot_offline_writes.py::test_last_episode_ordinal_and_crash_pair_are_admitted","tests/pilot/test_pilot_offline_writes.py::test_rows_descriptor_and_stream_escapes_latch","tests/pilot/test_pilot_offline_writes.py::test_external_hardlink_fails_at_installation","tests/pilot/test_pilot_offline_writes.py::test_preexisting_symlink_fails_at_installation","tests/pilot/test_pilot_offline_writes.py::test_real_font_manager_cache_and_lock","tests/pilot/test_pilot_offline_writes.py::test_real_font_json_dump_overflow_latches_and_cleans_lock","tests/pilot/test_pilot_offline_writes.py::test_utf8_stream_admits_encoded_length","tests/pilot/test_pilot_offline.py::test_offline_git_pin_requires_no_lazy_fetch_version","tests/pilot/test_pilot_offline.py::test_fresh_offline_boundary_denies_cloud_and_subprocess_escape","tests/pilot/test_pilot_offline.py::test_offline_child_cannot_write_outside_counted_workspace","tests/pilot/test_pilot_offline.py::test_git_provenance_denies_helper_configuration_and_keeps_closed_reads"]
args = [str(root/'.venv/bin/python'), '-B', '-m', 'pytest', '-p', 'no:cacheprovider',
    '--noconftest', '-q', '--tb=short', '--basetemp', str(stage/'pytest'), *selected]
start = time.monotonic()
p = subprocess.Popen(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
sel = selectors.DefaultSelector()
buffers = {}
for name, stream in [('stdout', p.stdout), ('stderr', p.stderr)]:
    sel.register(stream, selectors.EVENT_READ, name)
    buffers[name] = bytearray()
try:
    while sel.get_map():
        for key, _ in sel.select(1):
            chunk = os.read(key.fileobj.fileno(), 4096)
            if not chunk:
                sel.unregister(key.fileobj)
                continue
            if len(buffers[key.data]) + len(chunk) > 16384:
                p.kill()
                raise RuntimeError('output bound')
            buffers[key.data].extend(chunk)
        if time.monotonic() - start > 990:
            p.kill()
            raise RuntimeError('timeout')
finally:
    p.wait()
    sel.close()
    for name, payload in buffers.items():
        with (stage/(name+'.log')).open('xb') as stream:
            stream.write(payload)
for name, payload in buffers.items():
    print(name + ':\n' + payload.decode())
items = list(stage.rglob('*'))
files = [q for q in items if q.is_file() and not q.is_symlink()]
print(dict(exit=p.returncode, seconds=round(time.monotonic()-start, 3),
    logical=sum(q.stat().st_size for q in files),
    allocated=sum(q.lstat().st_blocks*512 for q in [stage, *items]), files=len(files),
    links=sum(q.is_symlink() for q in items),
    dirs=sum(q.is_dir() and not q.is_symlink() for q in [stage, *items])))
PY
```
