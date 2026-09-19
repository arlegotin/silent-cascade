# Task D2 — final evaluation control admission

Status: scoped implementation committed; controller frozen repeat and independent
review remain outside this writer handoff.

## Identity and scope

- Base: `0e92793029bd7d4392df1f7e90a64984c7c6e5c3`.
- Commit: `6ba9e687497ef31030977c30eaba46fa1c44570d`
  (`fix(pilot): admit final evaluation controls`).
- `src/silent_cascade/archive/producer.py`: SHA-256
  `d050db23cfe320b98700234da882f1050c0b13324831d765e3c47398d0526247`.
- `src/silent_cascade/train/pilot_checks.py`: SHA-256
  `172205b2db6a458621346d9a9e65605aea51e5c26ac0723499c54f16f0a41fd7`.
- `tests/pilot/test_pilot_final_control_admission.py`: SHA-256
  `84851a889b25e478b5edf2e9c8361d181fa3a52f5bb8d5c4d4290afa6a20dd0d`.

The implementation adds only the fixed `final_evaluation_control` spool bound
of `2 * PILOT_BYTES + ALLOCATION_OVERHEAD` (134,479,872 bytes), the
parameterless producer method that sends that exact closed operation, and
optional checks immediately before the final request and execution publishers.
The commit contains exactly the two owned source files and one focused test
file; controller progress/admission documents were not staged.

## Bounded commands and admission

The exact driver is retained at
`operational/sdd/task5-final-control-admission-check.py`, SHA-256
`c34e604af080a206c86c79bcb0c7a2aab218889cd116ada69bc3b446aba2b00e`.
It ran these exact inner commands, with stage-specific insertion/truncation:

1. `.venv/bin/ruff format --no-cache src/silent_cascade/archive/producer.py src/silent_cascade/train/pilot_checks.py tests/pilot/test_pilot_final_control_admission.py`
   (GREEN stages only).
2. `.venv/bin/python -B -m pytest --noconftest -p no:cacheprovider -q --tb=short --basetemp <fresh-stage>/pytest` followed by the four exact nodes
   `test_final_control_bound_is_fixed`,
   `test_final_control_dispatch_uses_exact_remaining_capacity`,
   `test_final_control_producer_sends_only_closed_operation`, and
   `test_final_control_admission_orders_real_evaluate_branching` from the new
   test file (11 expanded cases).
3. `.venv/bin/ruff check --no-cache` on the same three files.
4. `.venv/bin/ruff format --check --no-cache` on the same three files.
5. `git diff --check --` on the same three files.

RED ran command 2 only. Each command had a mechanical 150-second timeout,
32,768-byte limit per stream and 65,536-byte aggregate output limit. Each fresh
stage was limited to 524,288 allocated/logical bytes and 32 directories; RED
allowed eight file/link names and GREEN allowed 20. TMP/XDG/MPL/Hypothesis and
archive roots were closed below the stage. The tests used no child process,
model, evaluator execution, provider, network, or generated scientific output.

## RED

Prefix: `operational/scratch/task5-final-control-admission/red1`. Pytest exited
1 with ten intended failures and the no-archive branch passing: four unknown
operation failures, two missing producer-method failures, and four missing
admission/order/rejection failures. It took 1.75 seconds (2.172 seconds wrapper
wall). `0-stdout.log` is 6,255 bytes, SHA-256
`63d1ae0278423c1da39f1c3152862040d67197fd6efa1c595c43fe9fc1cc1fe3`;
`0-stderr.log` is empty, SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
Actual lstat inventory: 8,192 allocated bytes, two regular files, one symlink
and 12 directories. The stage was retained.

## GREEN1 and mechanical correction

Prefix: `operational/scratch/task5-final-control-admission/green1`. Ruff format
reformatted the new test; pytest then passed all 11 cases in 1.12 seconds
(1.487 seconds wrapper wall). Scoped Ruff check correctly stopped the stage on
I001 because one extra blank remained between the import block and module
constant. No behavioral test failed. Actual retained inventory: 12,288
allocated / 587 logical bytes, six regular files, one symlink and 12
directories; every stderr log is empty. Only that blank line was removed.

## GREEN2 and scoped statics

Prefix: `operational/scratch/task5-final-control-admission/green2`. All five
commands exited 0: three files unchanged by formatting; 11 cases passed in
1.04 seconds (1.409 seconds wrapper wall); Ruff check reported all checks
passed; Ruff format check reported three files already formatted; scoped diff
check emitted no output. Actual retained inventory: 16,384 allocated / 167
logical bytes, ten regular files, one symlink and 12 directories; every stderr
log is empty. Total retained task-stage allocation is 36,864 bytes. No attempt
was removed or reused.

## Self-review and exclusions

- Exact-capacity dispatch admits with no active hold mutation; one byte short
  and nonzero retained capacity reject. Producer response authentication is
  exercised for both admit and reject.
- Request admission occurs before request publication, including completed
  reuse. Execution admission occurs after evaluation and immediately before
  execution publication. First/second denials prevent the corresponding
  publisher; `archive_producer=None` preserves the existing path.
- Tests are wiring-only spies. The evaluator and publishers do not create
  destination data or synthetic scientific results; the final filesystem
  assertion uses the original `Path.exists` implementation.
- No archive-free authority, hold, schema, report field, CLI, scientific
  protocol, capacity inference, delegation, or completed-reuse behavior was
  changed. No full-file/suite, `make verify`, real evaluation, model execution,
  provider/network access, source copy, or child process was run.

