# Task5 standalone child writer controls (Task A)

Task A implementation frozen at `14abc50e9b033032ec9379163db2a7ee527c13c6`.
Final targeted check:16 passed. Independent review and controller execution of
the complete frozen primitive matrix remain outstanding. No scientific
execution, production launch binding, FitGuard hold, full local quality gate
or archival success is claimed. Controller owns the reservation and review.
Scope is the standalone adapter, boundary coordination, source inventory entry
and named tiny tests.

Base: `3820408`; parent process slice `70aace8`. Holder65407 was admitted by
the controller with 4 MiB task scratch, metadata33562624 B and logs2 MiB.
All prefixes below are retained and must never be reused or removed.

## Completed test stages

All stages use `.venv/bin/python -B -m pytest -p no:cacheprovider
--noconftest -q --tb=short --basetemp <prefix>/pytest`, external plugins disabled,
closed environment, bounded stdout/stderr, and the existing bounded process
capture for tiny fresh interpreters. No tests import or run the scientific
worker. Boundary installation imports Torch solely to retain its existing
network hooks; no model/update/evaluation/replay is constructed.

Each of the first four stages selected these exact tests:

- `tests/pilot/test_pilot_offline_writes.py::test_allowance_rejects_invalid_values[allocated_bytes-True]`
- `tests/pilot/test_pilot_offline_writes.py::test_real_publishers_exact_capacity`
- `tests/pilot/test_pilot_offline_writes.py::test_rows_stream_and_final_link`

Prefixes are under `.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-a/`.

| Stage | Result | Logical B | Allocated B | Regular files | Symlinks | Dirs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| red1 | 3 missing-API failures | 2918 | 4096 | 2 | 2 | 11 |
| green1 | 2 passed, 1 failed | 3082 | 12288 | 4 | 2 | 14 |
| green2 | 2 passed, 1 failed | 1335 | 16384 | 5 | 2 | 14 |
| green3 | 3 passed, 2.86 s | 104 | 16384 | 5 | 2 | 14 |
| controls1 | 29 passed, 3 behavioral RED, 22.61 s | 4719 | 24576 | 6 | 7 | 54 |
| controls2 | 32 passed, 22.61 s | 111 | 24576 | 7 | 7 | 54 |
| streams1 | 18 passed, 1 behavioral RED, 21.01 s | 10606 | 77824 | 36 | 8 | 87 |
| font1 | 3 passed, 12.63 s | 39348 | 45056 | 4 | 3 | 15 |
| additions1 | 5 passed, 2 behavioral RED, 6.59 s | 1464 | 20480 | 6 | 7 | 25 |
| final1 | 16 passed, 24.77 s | 39370 | 81920 | 13 | 16 | 45 |

Each of the first four stages was separately admitted at768 KiB, logical<160 KiB,
32 file/link/temp names,40 directories and120 seconds. Two child controls
maximum, each≤64 KiB, source≤4096 B, captured stdout/stderr≤2048 B each,
record limit4096 B, child deadline30 seconds. Outer stdout/stderr≤16384 B
each before writing two fresh `.log` files. Stage infrastructure includes five
environment directories, pytest base, per-test fixtures, current symlinks and
pytest lock ownership. Actual cumulative allocation after green3 is49152 B.
Wrapper shell exit0 does not imply pytest success; child pytest exit codes are
respectively1,1,1,0. Outer stderr logs were empty.

RED1 establishes missing API failures, not behavioral quota failures.
GREEN1 exposed the real stdlib `mkstemp` use of `O_RDWR|O_CREAT|O_EXCL`;
the adapter initially expected the pilot publisher's `O_WRONLY`. Inspection
of CPython3.12 `_mkstemp_inner` established the cause. The correction accepts
that access mode only for the active original durable-temp caller/candidate.
GREEN2 exposed the original atomic-create helper's two cleanup calls; the
token was released at first unlink. The correction retains ownership through
the complete original atomic create/write call, including final idempotent
cleanup. Original publisher bodies and payload semantics stay intact.

`controls1` proved two additional failures: evaluator aliases imported through
`pilot_data` escaped the outer token-lifetime wrapper, and the legacy boundary
rejected lexical/dir-fd `..` before the new denial latched. Durable adapters now
bind before the transitive evaluator import; the active guard validates paths
before the legacy boundary. `controls2` passed all32 cases. Original dir-fd
arguments now remain attached to link/rename mutations.

`streams1` passed descriptor escape, hardlink duplicate allocation, second-temp
denial/owned cleanup, UUID17 and UTF-8 controls. Its one RED was the real Git
batch pipe: `subprocess._communicate` uses `os.write` after `Popen` returns.
The correction retains only live stdin handles created in the existing exact
Git batch window, checking descriptor identity and closed state. `font1` and
`final1` confirmed that read-only command capability.

`additions1` exposed both late consumer imports being accepted before binding.
Binding now rejects preloaded `pilot_data` and project modules holding original
durable publisher aliases. `final1` passed those two controls together with
real publishers, row streams, both font cases and the four legacy checks.

Genuine `font_manager` import retained26961 B JSON (28672 allocated), using
`MPL_IGNORE_SYSTEM_FONTS=1`. A separate64 KiB allowance stopped its actual
`json_dump` at12288 B (12288 allocated), latched `bytes`, and cleaned the actual
zero-byte Matplotlib lock. Both were repeated successfully in `final1`. There
was no font subprocess/native fallback or unknown cache-path admission.
UTF-8 controls independently admitted exactly4096 encoded bytes, returned2048
characters, and rejected the next two encoded bytes before writing.

All ten prefixes remain. A final read-only no-follow stat inventory recorded
323584 allocated B,88 regular-file names,56 symlinks and334 directories including
the task container. Per-stage logical totals sum103057 B. Hardlink names count
separately. No stage was removed, replaced or silently reclassified. Every
stage fit the held4 MiB prospective reservation using its separately admitted
ceiling plus all previously retained actual bytes, never reusing one ceiling
as a total across executions.

## Later stage selectors and admission

The names below are appended to `tests/pilot/test_pilot_offline_writes.py::`.
Parameter expansion is included in the recorded case counts.

`controls1` and `controls2` selected exactly:

```text
test_allowance_rejects_invalid_values
test_transitive_evaluation_publisher_keeps_token_lifetime
test_publication_one_byte_short_has_no_temp_or_mutation
test_create_collision_reserves_incoming_bytes
test_closed_names_latch_before_publication
test_lowered_inventory_ceiling_denies_before_creation
test_second_weights_digest_denied
test_unowned_writes_and_paths_latch
```

At those stages the closed-name test had8 cases; the frozen version has9.
Each original stage:32 cases/20 children,≤3 MiB/<256 KiB logical,
160 file/link/temp names,180 directories,660-second batch. Each child≤64 KiB.
The conservative byte formula was256 KiB +8192*(160+180)=3047424 B≤3 MiB.

`streams1` selected exactly:

```text
test_rows_descriptor_and_stream_escapes_latch
test_atomic_replace_and_create_preserve_original_behavior
test_second_atomic_temp_denied_and_owned_cleanup_allowed
test_pending_and_final_hardlink_both_charge_allocation
test_seventeenth_crash_stem_denied
test_external_hardlink_fails_at_installation
test_utf8_stream_admits_encoded_length
test_closed_git_batch_pipe_still_works
```

19 children,18 at≤64 KiB and the16-preseed-zero-file crash control at≤128 KiB.
Stage≤3 MiB/<256 KiB/160 names/180 directories,630-second batch.
The hardlink test charged both pending and final names before a later tiny
publication; external hardlink and symlink controls stayed in their test fixture.
Git read one existing package `__init__.py` blob into memory, never copied source
or a corpus to disk.

`font1` selected the Git batch test plus
`test_real_font_manager_cache_and_lock` and
`test_real_font_json_dump_overflow_latches_and_cleans_lock`:3 children at64,
256 and64 KiB. Stage≤1 MiB/<448 KiB/32 names/40 directories,120-second batch;
448 KiB +8192*(32+40)=1 MiB. No native fallback was authorized.

`additions1` selected exactly:

```text
test_install_requires_exact_allowance_type
test_install_cannot_replace_or_enlarge_allowance
test_late_consumer_binding_is_denied
test_preexisting_symlink_fails_at_installation
test_last_episode_ordinal_and_crash_pair_are_admitted
test_closed_names_latch_before_publication[weights.safetensors-pilot]
```

7 cases/6 children≤64 KiB; the prospective8/7 arithmetic overcount was corrected
in the controller message before execution. Stage≤1.5 MiB/<128 KiB/80 names/
90 directories,270-second batch;128 KiB +8192*(80+90)=1523712 B≤1.5 MiB.

`final1` selected the exact executable list in the launcher below.16 cases/
14 children, one at256 KiB and the other13 at≤64 KiB, including the finite
legacy bodies. Stage≤3 MiB/<256 KiB/160 names/180 directories,480 seconds.
Legacy subprocess capture retains its existing30-second timeout; new helpers
use the exact2048/2048/4096-byte capture limits and30-second deadline. All
outer logs were capped at16384 B per stream before disk publication. Stage
fixtures, current links, pytest lock and five environment directories are
included in these name/directory ceilings; the launcher creates no source files.

## Frozen complete new matrix:62 cases

All functions below are in `tests/pilot/test_pilot_offline_writes.py`. The two
groups match the controller's final selection. They cover every frozen new
test without selecting a scientific diagnostic. Group1 contains33 new cases;
group2 contains29 new cases plus the four legacy tests in the launcher:
66 total controller cases,62 new plus4 legacy. These counts are not reports
of a completed controller execution.

| Group | Function | Cases |
| --- | --- | ---: |
| A | test_allowance_rejects_invalid_values | 12 |
| B | test_install_requires_exact_allowance_type | 1 |
| B | test_install_cannot_replace_or_enlarge_allowance | 1 |
| B | test_late_consumer_binding_is_denied | 2 |
| B | test_real_publishers_exact_capacity | 1 |
| B | test_rows_stream_and_final_link | 1 |
| A | test_transitive_evaluation_publisher_keeps_token_lifetime | 1 |
| A | test_publication_one_byte_short_has_no_temp_or_mutation | 2 |
| A | test_create_collision_reserves_incoming_bytes | 1 |
| A | test_closed_names_latch_before_publication | 9 |
| A | test_lowered_inventory_ceiling_denies_before_creation | 2 |
| A | test_second_weights_digest_denied | 1 |
| A | test_unowned_writes_and_paths_latch | 5 |
| B | test_closed_git_batch_pipe_still_works | 1 |
| B | test_atomic_replace_and_create_preserve_original_behavior | 1 |
| B | test_second_atomic_temp_denied_and_owned_cleanup_allowed | 1 |
| B | test_pending_and_final_hardlink_both_charge_allocation | 1 |
| B | test_seventeenth_crash_stem_denied | 1 |
| B | test_last_episode_ordinal_and_crash_pair_are_admitted | 1 |
| B | test_rows_descriptor_and_stream_escapes_latch | 11 |
| B | test_external_hardlink_fails_at_installation | 1 |
| B | test_preexisting_symlink_fails_at_installation | 1 |
| B | test_real_font_manager_cache_and_lock | 1 |
| B | test_real_font_json_dump_overflow_latches_and_cleans_lock | 1 |
| B | test_utf8_stream_admits_encoded_length | 2 |

Group A/main1 has33 cases/21 children plus12 pure validations. Group B/main2
has29 new cases/28 children plus1 pure validation; adding the four legacy
controls gives33 cases/31 children. Group B includes the128 KiB crash and256 KiB
font controls. Controller prospective ceilings are3 MiB/<256 KiB/160 names/
180 directories for main1 and3.5 MiB/<512 KiB/180 names/200 directories for
main2; main1 actuals must be measured before main2 admission. Ordinary children
are≤64 KiB; these are per-attempt ceilings, not a group reservation.

## Source identities and checks

Commit `14abc50e9b033032ec9379163db2a7ee527c13c6` contains only these four files:

```text
9e397b81beb504d8d022f62ad344ef6e25e2ecc09f3577ef3258ab82e43b397e  src/silent_cascade/train/pilot_offline_writes.py
9e961ed6d1a6f75fff2d2d38cfa109ee5c1c30b46a78841eb23e0b32c1ea230b  tests/pilot/test_pilot_offline_writes.py
50a809e529339a65ab5334a95ad82dda4789a756cbbd53dd43e8c87e43f963af  src/silent_cascade/train/pilot_offline.py
92eebeb19f0e82de2c4d9fe365b8b39431e942763fa5ca286af25f375691c7d6  src/silent_cascade/train/pilot_evidence.py
```

Before committing, `ruff check --no-cache` and `ruff format --no-cache --check`
passed on exactly those four files, and `git diff --check` passed. Git index
write required the standard sandbox escalation; scoped commit succeeded. No
controller documents were staged. This report remains controller-owned evidence
for recording/review. Source is frozen after the listed checks and commit.

Standalone installation is irreversible and validates an explicit root and
exact allowance dataclass. The guard is a closed Python writer/descriptor
adapter for this workload. It does not claim a kernel-wide arbitrary native
syscall quota or protection against arbitrary Python introspection/patching of
its private state. Serialized production allowance authentication, parent
reservation binding and FitGuard hold are Tasks B/C, deliberately absent here.
Scientific algorithms, controls, gates and RNG code were not modified.

## Reproducible bounded final launcher

This reproduces the exact `final1` selectors and480-second/16 KiB outer bounds.
The original used prefix was `final1`; the code uses a fresh prospective
`controller-recheck1` so it cannot overwrite that evidence. Obtain controller
admission for the new prefix before executing. Earlier stages used the same
launcher structure with their exact selector lists and deadlines above.

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 \
  PYTHONPATH="$PWD/src" .venv/bin/python -B - <<'PY'
import os, pathlib, selectors, subprocess, time
root = pathlib.Path.cwd()
stage_name = 'controller-recheck1'
names = [
    'test_install_requires_exact_allowance_type',
    'test_install_cannot_replace_or_enlarge_allowance',
    'test_late_consumer_binding_is_denied',
    'test_preexisting_symlink_fails_at_installation',
    'test_last_episode_ordinal_and_crash_pair_are_admitted',
    'test_closed_names_latch_before_publication[weights.safetensors-pilot]',
    'test_real_publishers_exact_capacity', 'test_rows_stream_and_final_link',
    'test_closed_git_batch_pipe_still_works',
    'test_real_font_manager_cache_and_lock',
    'test_real_font_json_dump_overflow_latches_and_cleans_lock',
]
legacy = [
    'test_offline_git_pin_requires_no_lazy_fetch_version',
    'test_fresh_offline_boundary_denies_cloud_and_subprocess_escape',
    'test_offline_child_cannot_write_outside_counted_workspace',
    'test_git_provenance_denies_helper_configuration_and_keeps_closed_reads',
]
selected = ['tests/pilot/test_pilot_offline_writes.py::' + name for name in names]
selected += ['tests/pilot/test_pilot_offline.py::' + name for name in legacy]
stage = root / '.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-a' / stage_name
assert not stage.exists()
stage.mkdir(parents=True)
for name in ('tmp', 'xdg', 'mpl', 'hypothesis', 'archive'):
    (stage / name).mkdir()
env = dict(os.environ, TMPDIR=str(stage/'tmp'), TMP=str(stage/'tmp'),
    TEMP=str(stage/'tmp'), XDG_CACHE_HOME=str(stage/'xdg'),
    XDG_CONFIG_HOME=str(stage/'xdg'), XDG_DATA_HOME=str(stage/'xdg'),
    MPLCONFIGDIR=str(stage/'mpl'), HYPOTHESIS_STORAGE_DIRECTORY=str(stage/'hypothesis'),
    SILENT_CASCADE_ARCHIVE_CACHE=str(stage/'archive'), OMP_NUM_THREADS='1')
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
        if time.monotonic() - start > 480:
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

Child environment comes from the existing `offline_environment(root)`, including
the pinned read-only Git command capability and TMP/XDG/MPL roots inside each
child. Allocation includes each retained directory/file/link name's
`st_blocks*512`; the admission formula adds4096 B rounding and metadata cushion.
Child capacity is an explicitly supplied limit, never a granted reservation.
