# Task5 child-write launch binding (Task B) report

Status: source/test delivery complete at
`9f0c0b3c96494ddc595e184e2b724a772a2eef4e`. Controller repeat and independent
review remain outstanding. This report is not staged in the scoped source/test
commit.

## Delivered boundary

- The privacy-safe intent-v2 accepts an optional exact three-key
  `write_allowance`. Missing old fields parse as `None` for forensic reading;
  fresh launch validation requires the field to be present, including explicit
  `null` for unbounded-child mode.
- `measure_pilot_offline` preserves mandatory `process_custody`, rejects
  nonexact allowance dataclass types, and serializes the caller-supplied
  allowance before computing the existing raw intent digest.
- The six-element child command and closed environment are unchanged. The
  bootstrap validates the intent, constructs the exact allowance, installs the
  writer boundary before Torch/consumer imports, binds publishers, verifies the
  installed root/allowance before calling the worker, and performs a final
  latched-denial check.
- Worker launch validation requires the installed root and allowance, repeats
  that check before `offline.json`, and checks the writer latch immediately
  before final publication. A caught denial therefore still exits nonzero.
- The no-allowance mode retains the prior network/import/output fence and its
  legacy retained-file/descriptor behavior.
- Overflow and cancellation controls retain bounded raw stderr only in private
  custody, publish only count/hash/disposition publicly, join the child, and do
  not accept an invalid preseeded report as completion.

No FitGuard hold/category authority was created or inferred. Task C owns H and
category-share establishment. No spool/cache routing, scientific algorithm,
RNG, controls, gates, storage caps, source authentication, actual neural worker,
provider/network operation, copied corpus, full gate, or `make verify` was run
or changed.

## Exact launcher and commands

Every executable stage used this outer shell command, with the exact stage
prefix and Python body parameters recorded below:

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH="$PWD/src" .venv/bin/python -B -
```

The stdin launcher asserted that its stage did not exist, created only the
stage and `tmp`, `xdg-cache`, `xdg-config`, `xdg-data`, `mpl`, `hypothesis`, and
`archive`, and did not precreate pytest's basetemp. Its child environment set
`TMPDIR`, `TMP`, `TEMP`, the three XDG roots, `MPLCONFIGDIR`,
`HYPOTHESIS_STORAGE_DIRECTORY`, `SILENT_CASCADE_ARCHIVE_TEST_SCRATCH`,
`SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1`, `OMP_NUM_THREADS=1`, and
`MPL_IGNORE_SYSTEM_FONTS=1` to that fresh stage. It launched each exact argv
below with `subprocess.Popen`, drained stdout/stderr concurrently with
`selectors.DefaultSelector`, capped each command pipe at 16 KiB, enforced the
listed wall limit, terminated/killed and joined on a bound, and wrote each
stream once with `xb`. It then measured the retained tree with `lstat`, printed
per-command return codes and byte counts, and asserted the stage's logical,
allocated, name, and directory limits.

All pytest commands had the exact fixed prefix:

```text
.venv/bin/python -B -m pytest -p no:cacheprovider --noconftest -q
--tb=short --basetemp <fresh-stage>/pytest
```

RED selected these exact node IDs:

```text
tests/pilot/test_pilot_offline_process.py::test_safe_intent_write_allowance_is_optional_and_strict
tests/pilot/test_pilot_offline_process.py::test_measurement_serializes_exact_write_allowance_before_launch
tests/pilot/test_pilot_offline_process.py::test_measurement_rejects_nonexact_write_allowance_before_launch
tests/pilot/test_pilot_offline_process.py::test_bootstrap_postcheck_exposes_caught_write_denial
tests/pilot/test_pilot_offline_process.py::test_bootstrap_digest_binds_write_allowance
tests/pilot/test_pilot_offline_process.py::test_process_reader_rejects_changed_intent_allowance
tests/pilot/test_pilot_offline_writes.py::test_real_publishers_exact_capacity
```

Focused RED2 selected exactly:

```text
tests/pilot/test_pilot_offline_process.py::test_bootstrap_requires_installed_allowance_before_worker
```

GREEN selected the RED nodes, focused RED2, and exactly:

```text
tests/pilot/test_pilot_offline_process.py::test_privacy_intent_hashes_replace_paths
tests/pilot/test_pilot_offline_process.py::test_privacy_bootstrap_accepts_bound_hash_identities
tests/pilot/test_pilot_offline_process.py::test_privacy_nonempty_failure_retains_closed_outcome[output_limit]
tests/pilot/test_pilot_offline_process.py::test_privacy_nonempty_failure_retains_closed_outcome[cancelled]
tests/pilot/test_pilot_offline_process.py::test_capture_bounds_both_pipes
tests/pilot/test_pilot_offline_process.py::test_capture_timeout_joins_child
tests/pilot/test_pilot_offline_process.py::test_fresh_nested_audit_denies_changed_launch_authority
tests/pilot/test_pilot_offline.py::test_offline_child_cannot_write_outside_counted_workspace
```

Cover-process selected the following exact 40-case set:

```text
tests/pilot/test_pilot_offline_process.py::test_safe_intent_write_allowance_is_optional_and_strict
tests/pilot/test_pilot_offline_process.py::test_measurement_serializes_exact_write_allowance_before_launch
tests/pilot/test_pilot_offline_process.py::test_measurement_rejects_nonexact_write_allowance_before_launch
tests/pilot/test_pilot_offline_process.py::test_bootstrap_postcheck_exposes_caught_write_denial
tests/pilot/test_pilot_offline_process.py::test_bootstrap_requires_installed_allowance_before_worker
tests/pilot/test_pilot_offline_process.py::test_bootstrap_digest_binds_write_allowance
tests/pilot/test_pilot_offline_process.py::test_process_reader_rejects_changed_intent_allowance
tests/pilot/test_pilot_offline_process.py::test_privacy_intent_hashes_replace_paths
tests/pilot/test_pilot_offline_process.py::test_privacy_bootstrap_accepts_bound_hash_identities
tests/pilot/test_pilot_offline_process.py::test_privacy_nonempty_failure_retains_closed_outcome
tests/pilot/test_pilot_offline_process.py::test_capture_bounds_both_pipes
tests/pilot/test_pilot_offline_process.py::test_capture_timeout_joins_child
tests/pilot/test_pilot_offline_process.py::test_fresh_nested_audit_denies_changed_launch_authority
tests/pilot/test_pilot_offline_process.py::test_bootstrap_rejects_changed_launch_authority
tests/pilot/test_pilot_offline_process.py::test_reuse_rejects_incomplete_before_authentication
tests/pilot/test_pilot_offline_process.py::test_collection_fences_incomplete_before_authentication
tests/pilot/test_pilot_offline_process.py::test_measurement_persists_failed_invalid_report_and_fences_reuse
tests/pilot/test_pilot_offline_process.py::test_log_publication_failure_never_publishes_completion
tests/pilot/test_pilot_offline_writes.py::test_real_publishers_exact_capacity
tests/pilot/test_pilot_offline.py::test_offline_child_cannot_write_outside_counted_workspace
```

Both full writer stages used the exact pytest selector:

```text
tests/pilot/test_pilot_offline_writes.py
```

This expands to 64 cases and 51 tiny Python controls. Each full writer stage
also ran these exact commands on the four owned files:

```text
.venv/bin/ruff check --no-cache <four-owned-files>
.venv/bin/ruff format --check --no-cache <four-owned-files>
git diff --check -- <four-owned-files>
```

The final static stage ran those same three commands. The four owned files were
`pilot_offline.py`, `pilot_offline_process.py`,
`test_pilot_offline_process.py`, and `test_pilot_offline_writes.py` at their
repository paths.

## RED, GREEN, and retained accounting

All prefixes are beneath
`.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-b/`
and remain retained. Inventory logical bytes are distinct from individual log
byte counts.

| Stage/prefix | Result | Wall | stdout/stderr logs | Logical | Allocated | Files+links | Dirs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `red-20260919-1d4f93d8` | pytest exit 1; 11 failed, 5 passed | 3.113 s | 7625 / 0 B | 13906 B | 53248 B | 23 | 42 |
| `red2-installed-20260919-1d4f93d8` | pytest exit 1; 1 failed | 2.228 s | 1375 / 0 B | 2139 B | 8192 B | 4 | 10 |
| `green-20260919-1d4f93d8` | pytest exit 0; 25 passed | 9.602 s | 99 / 0 B | 13579 B | 135168 B | 56 | 71 |
| `cover-process-20260919-1d4f93d8` | 40 passed; static check exposed formatting only | 13.241 s | per-command logs retained | 34789 B | 307200 B | 120 | 144 |
| `cover-writers-20260919-1d4f93d8` | 63 passed, 1 helper failure; format check failed | 60.454 s | per-command logs retained | 49523 B | 196608 B | 88 | 174 |
| `cover-writers2-20260919-1d4f93d8` | 64 passed; check/diff passed; one blank-line format failure | 64.337 s | pytest 110/0 B | 47849 B | 196608 B | 88 | 174 |
| `cover-static-20260919-1d4f93d8` | all three static commands exit 0 | 0.116 s | 45 / 0 B total | 45 B | 8192 B | 6 | 8 |

Total retained Task B evidence: 161830 logical bytes, 905216 allocated
bytes, 385 regular/symlink names, and 623 directories. No prefix was reused,
deleted, or reclassified.

The first RED failed for the intended missing schema/API seams: seven absent
intent-field failures, two missing measurement keyword failures, one missing
boundary keyword failure, and the bootstrap stopping at the old strict schema.
Five digest/reader cases already passed via existing authentication. RED2 then
isolated the remaining phase defect: after a control removed the installed
guard, bootstrap reached the worker tripwire. The explicit post-install check
made that path fail before the worker.

Cover-process's behavior was fully green; its static failure identified one
precedence-parenthesis and formatting issue. The first full writer run found a
real test-harness regression: the retained second-install control still needed
`install_offline_writes` in the generated child's imports. Restoring that
already-loaded symbol enabled the existing `_ACTIVE` denial and did not add an
artifact family. The second full writer run passed all 64 cases. Its only
remaining failure was a missing blank line; the current source then passed the
fresh final Ruff/format/diff stage.

## Final verification evidence

- Focused current behavior: 25 passed.
- Expanded process/privacy/legacy cover: 40 passed.
- Entire shared writer helper file: 64 passed in 64.03 seconds.
- Current four-file Ruff check: exit 0, `All checks passed!`.
- Current four-file Ruff format check: exit 0, `4 files already formatted`.
- Current four-file `git diff --check`: exit 0.
- `pilot_evidence.py` already contains
  `src/silent_cascade/train/pilot_offline_writes.py` in the required package
  inventory; Task B did not edit or restructure that verifier.

The final full writer envelope was source-derived from the closed literal
families and existing per-control limits, not granted as historical-size-based
child capacity. It retained the actual 128 KiB 16-crash-stem control and 256
KiB font control; other controls kept their exact lower allowances.

## Final file identities and commit

```text
614ed444d6b1b9a76ace78f91897b7d0817a4aa50bfac1a37aa6d59feba05559  src/silent_cascade/train/pilot_offline.py
b899f2edfade6a3a431ce60d99dd7760d367729dadfa5bd595c669a5194278cf  src/silent_cascade/train/pilot_offline_process.py
517cb7ad23c4198f2bb1b8ad31b3746c8b7ff7dfe790612a96b4c58ceb12c916  src/silent_cascade/train/pilot_offline_writes.py
528b25bc25ee495dbbbe72616255af46ff542d7d73b2770d0d0585c289d02fec  src/silent_cascade/train/pilot_evidence.py
68a1e8fdfedfc1ef128eefd0bc9d7821cae75efeb97a62674d77e1e50c890490  tests/pilot/test_pilot_offline_process.py
d9428def6cfd407dad23a36807efa2ce0edfbe000ca485c701b42c72d814425a  tests/pilot/test_pilot_offline_writes.py
```

Scoped source/test commit:

```text
9f0c0b3c96494ddc595e184e2b724a772a2eef4e
feat(pilot): bind offline child write allowance
```

Only the four owned source/test files were staged in that commit. Controller
changes to progress, the Task C brief, and the approved plan were preserved and
left unstaged. This report is likewise left unstaged for controller handling.

## Limitations and handoff

- This is primitive launch-binding qualification only. It does not claim a
  real scientific report, actual neural-worker success, full source gate,
  `make verify`, archive success, or release readiness.
- Default `write_allowance=None` is explicit child-unbounded mode only; it does
  not waive mandatory parent process custody.
- Task B can authenticate the once-installed root/exact allowance but cannot
  see or claim Task C's held H/category shares. The later integration must
  establish that authority without bypassing the current public/private or
  spool/cache custody rules.
- All failed roots and logs remain private operational evidence. No raw private
  stderr was placed in a public process attachment by these controls.
- Controller fresh repeat and independent Task B review are still required.

## FIX1 combined-fence qualification

Independent review of source commit `9f0c0b3` found a qualification gap rather
than a production defect: the guarded-denial controls stopped at direct
capture, while the parent custody controls substituted an unguarded child and
passed no write allowance. FIX1 changes only
`tests/pilot/test_pilot_offline_process.py`. It adds a real
`measure_pilot_offline` + real custody + exact allowance control whose launch
shim substitutes only a tiny worker and then executes the production
`boundary._PROGRAM` bootstrap.

The worker validates the installed launch binding, publishes the seven-byte
invalid `offline.json`, then attempts a one-byte `step.json` after consuming
the exact two-name allowance. It catches `OfflineWriteDenied`, writes a
`DENIED:` marker to stderr, and either returns or sleeps for the competing
termination control. The four parameter cases prove:

- bootstrap postcheck turns a caught denial into `nonzero_exit`;
- a 65-byte marked stderr write becomes a 64-byte `output_limit` prefix;
- the marked denial occurs before a 15-second parent timeout of a 60-second
  sentinel sleep;
- cancellation is injected only after the selector has delivered the marked
  stderr event;
- every public result is failed, has `offline_sha256=null`, and contains only
  the private stream count/hash/disposition;
- the invalid `offline.json` remains an invalid sentinel and never becomes
  completion evidence; and
- each exact child has already been joined (`waitpid(..., WNOHANG)` raises
  `ChildProcessError`).

No artificial RED was manufactured: the review explicitly identified missing
coverage, not broken production behavior. The first focused qualification was
therefore expected to pass the unchanged production implementation.

The exact new selector, expanding to four cases, was:

```text
tests/pilot/test_pilot_offline_process.py::test_measurement_combines_write_exhaustion_with_parent_failure_custody
```

The focused inner command was the common pytest prefix from this report plus
that selector. The covering inner command was the same prefix plus these exact
selectors, expanding to 11 cases:

```text
tests/pilot/test_pilot_offline_process.py::test_measurement_combines_write_exhaustion_with_parent_failure_custody
tests/pilot/test_pilot_offline_process.py::test_measurement_serializes_exact_write_allowance_before_launch
tests/pilot/test_pilot_offline_process.py::test_measurement_persists_failed_invalid_report_and_fences_reuse
tests/pilot/test_pilot_offline_process.py::test_privacy_nonempty_failure_retains_closed_outcome
```

Each covering stage then ran these exact commands:

```text
.venv/bin/ruff check --no-cache tests/pilot/test_pilot_offline_process.py
.venv/bin/ruff format --check --no-cache tests/pilot/test_pilot_offline_process.py
git diff --check -- tests/pilot/test_pilot_offline_process.py
```

The outer launcher and environment were exactly the closed launcher already
recorded above. FIX1 used at most four tiny Python measurement children and 14
trusted read-only Git children for the new cases; the covering run added seven
existing tiny Python controls. The substituted source was at most 4096 bytes,
scientific sentinel payloads were seven bytes plus a refused one byte, stdout
was capped at 64 bytes, stderr at 4096 bytes for the postcheck case and 64
bytes for the other cases, records at 1024 bytes, and process timeout at 15
seconds. No real scientific worker, provider, network, copied fixture, full
suite, or full gate ran.

| Stage/prefix | Result | Wall | stdout/stderr logs | Logical | Allocated | Files+links | Dirs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fix1-combined-20260919-1d4f93d8` | 4 passed in 20.59 s | 21.154 s | 99 / 0 B | 9382 B | 102400 B | 35 | 49 |
| `fix1-cover-20260919-1d4f93d8` | commands green; wrapper exit 1 because 99 names exceeded forecast 96 | under 90 s | 144 B total across eight logs | 24480 B | 270336 B | 99 | 119 |
| `fix1-cover2-20260919-1d4f93d8` | 11 passed in 23.33 s; all three statics exit 0 | 23.938 s | 144 B total across eight logs | 24495 B | 270336 B | 99 | 119 |

The first covering prefix is retained as an honest accounting failure. Its
pytest, Ruff, format, and diff commands all exited zero, but the wrapper exited
one on the admitted 96-name limit. COVER2 repeated the identical commands
under the corrected 112-name/136-directory envelope and passed. Its exact raw
logs are `pytest.stdout.log`, `pytest.stderr.log`,
`ruff-check.stdout.log`, `ruff-check.stderr.log`,
`ruff-format.stdout.log`, `ruff-format.stderr.log`,
`git-diff.stdout.log`, and `git-diff.stderr.log` directly under the prefix.

The retained-name and directory forecast is source-derived for these exact 11
cases. Eight wrapper logs consume eight names. Four pytest `current` links
consume four. The four combined controls retain eight files each: two ledger
files, private binding/stderr, and public intent/result/empty-stdout/invalid
offline. The serialization and invalid-report controls also retain eight each.
Four existing nonempty-failure modes retain eight each, while the injected
private-write failure retains seven because `stderr.raw` is refused. Thus the
exact family count is `8 + 4 + 4*8 + 2*8 + 4*8 + 7 = 99` names, with 13 spare
in the corrected limit. Directories are the stage, seven closed environment
roots, pytest root, and ten per retained case (test root, workspace, three
private-log levels, two ledger levels, and three public-run levels):
`1 + 7 + 1 + 11*10 = 119`, with 17 spare. Final process-public and private
files replace their atomic temporaries, so no temporary name survives; the
112-name ceiling still covers all 13 spare names if an admitted atomic
temporary is observed at inventory time.

Across all Task B owner prefixes recorded in this report, retained evidence is
now 220187 logical bytes, 1548288 allocated bytes, 618 regular/symlink names,
and 910 directories. Including controller/reviewer prefixes, the shared Task B
holder stood at 2048000 allocated bytes after COVER2, below its 4 MiB hold.

FIX1 test-only commit:

```text
cc05f1fd97a258f04aa4d9262d337fca1078bf17
test(pilot): qualify combined offline fences
184b69d47e0a2b1113a20de067f65ad4f47d90745a3d35f73f124b9600684eac  tests/pilot/test_pilot_offline_process.py
```

Only the test file was staged in this commit. Controller-owned progress and
this operational report remained unstaged for controller handling.
