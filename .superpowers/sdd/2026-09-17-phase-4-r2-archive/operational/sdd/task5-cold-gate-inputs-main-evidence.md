# Controller verification: Task 5 cold gate inputs

Source/test commit: `b417c4b14aa286cd0cec63990d354d84ea7e9b2e`.
Worker report commit: `8934f22`.
Independent review is recorded separately; this is executed local evidence.

## Fresh focused verification

Under the already-held owner13703 reservation, admitted a fresh absent
`operational/scratch/task5-cold-gate-inputs/main-cover` stage with a 2 MiB
rounded / 1 MiB logical / 96-file / 96-directory prospective bound. The
previous retained task total was 212,992 allocated bytes, so this stage and
previous failures fit the unchanged 4 MiB task allowance.

Used the closed launcher in task5-cold-gate-inputs-report.md, with that fresh
stage and these exact selectors:

```sh
tests/pilot/test_pilot_cold_gate_inputs.py tests/pilot/test_pilot_evidence.py
```

Result: **39 passed in 2.89 seconds**, wall 5.14, user 3.32, system 0.92.
No new fit, forward, event execution, replay, diagnostic, full gate, provider
or historical payload copy occurred. The genuine readers used the original
historical debug artifacts under the same tripwires and lease guards.

Main stage: 30 regular files, 66 directories, 3,393 logical bytes,
122,880 allocated bytes, zero symlinks.
All task prefixes including every prior failure: 81 regular files,
263 directories, 23,434 logical bytes, 335,872 allocated bytes.
The root stage has no separate pytest.log; command output was retained by
the existing tool log. No prefix was reused or removed.

Fresh local static verification:

```sh
.venv/bin/ruff check --no-cache src/silent_cascade/train/pilot_evidence.py src/silent_cascade/train/pilot_checks.py tests/pilot/test_pilot_cold_gate_inputs.py
.venv/bin/ruff format --check --no-cache src/silent_cascade/train/pilot_evidence.py src/silent_cascade/train/pilot_checks.py tests/pilot/test_pilot_cold_gate_inputs.py
git diff --check
```

All checks passed; three files already formatted; no whitespace errors.
Full `make verify` was not admitted. This does not complete the outer gate,
collection, recovery, R2 handoff or Phase 4 production pilot.

## Fresh verification after integrity review fix

The independent review found a real partial-availability gap: a missing
journal dependency skipped checkpoint-index validation. The original worker
added four regressions, all observed failing, then extracted the existing
journal-record validator and independently validated the available index.
Controller review also found a newly allowed null latest descriptor; a fifth
observed RED confirmed it, and unconditional typed parsing restored the actual
producer/durable schema. See the worker report for every retained RED stage.

Source/test fix: `dc9d54685c4f8814ff6f23a47486c4a088b6c5a5`.
Fix report: `3302d29`.

Controller admitted a fresh absent
`operational/scratch/task5-cold-gate-inputs/fix-main-1` under owner13703,
at <=2 MiB rounded / 1 MiB logical / 128 files / 128 directories. The
prior inclusive task total was 569,344 bytes; prospective growth therefore
fit the unchanged 4 MiB reservation. Used the same closed launcher with:

```sh
tests/pilot/test_pilot_cold_gate_inputs.py tests/pilot/test_pilot_evidence.py tests/pilot/test_pilot_trainer.py::test_journal_rejects_malformed_record_shape
```

Result: **47 passed in 2.95 seconds**, wall 5.19, user 3.44, system 0.94.
Only the named tiny malformed-journal trainer selector ran, not its module's
large-file or training tests. The four touched source/test files passed
fresh Ruff lint and format checks; git diff --check passed.

Fresh fix stage: 37 files, 83 directories, 6,786 logical bytes,
151,552 allocated bytes, zero symlinks. All inclusive task prefixes:
175 files, 500 directories, 52,470 logical bytes, 720,896 allocated bytes.
All prior failure and success prefixes remain intact. Source and protocol
limits are unchanged; scoped re-review and final owner accounting follow.
