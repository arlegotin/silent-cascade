# Task 5 cold numerical semantic-reader report

## Scope and immutable evidence

Implemented only the artifact reader group in
`train/pilot_verification.py` and `train/pilot_evidence.py::verify_numeric_evidence`,
with the focused cold-numerics test and the reviewed adjacent historical fixture.
No diagnostic production, training fit, model forward, replay/event execution,
provider call, MPS execution, recovery, aggregate gate, tolerance, or wire-schema
change was made.

The borrowed source remained read-only at:

`.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-4/cover-pilot-final/test_real_smoke_workflow_reuse0/repo/runs/smoke/final/numerics`

- Original execution source: `42b8ab0ba8a647794b08ce29327fce105e3a75f7`.
- Numeric report binding: `80ac180c9c4a86ab611e0d495415aa81b54bf3f2f1866f0fa2051f7c6821bec7`.
- Original input checkpoint: `e55caab7431984fdb47204eff898af4c04f4aa91adaaf2cf60efb9f74703c357`.
- Portable/runtime weights: `e102be3e3d9c04b9e939ca763d4b6ac1e8a768c62ead350668271332bbf8e094`.
- Recorded result remains `missing_devices=["mps"]`, 16 runtime CPU rows,
  and `device_checks_passed=False`. No source/hash relabeling or completed-run
  claim was introduced.

## Reader result

The numerical reader now exhausts authenticated discovery once per top-level
read while retaining only entries below the requested numerical prefix. It
unions resident and authenticated cold inventory, rejects wrong roots, extras,
symlinks and integrity failures, and leases one payload at a time. Report and
DONE, input archive and portable weights, capture tensor and operations JSON,
resume observations/captures/summary, and closure members are read in separate
bounded leases. Decoded values and archive digests survive lease release; cache
paths do not.

Runtime evidence uses the real evaluation header/index/episode scanner. The
diagnostic required-file set is independently reconstructed from verified rows
and crash references and compared with the authenticated DONE inventory. Resume
verification reuses the archive decoder's digest rather than retaining or
rereading raw archive bytes. Existing disposable CPU optimizer/clipping
arithmetic is unchanged.

The complete numeric verifier returns the original `False` result with no
unavailable labels. With only `one_hop-cpu.safetensors` absent, the partial path
returns `False` and exactly:

```text
numeric.complete_cross_record_validation
numeric.capture_tensors:one_hop-cpu
```

The available operations JSON is still semantically validated; independently
corrupt operations raise rather than being converted to unavailable evidence.

## Admission, commands, results, and allocation

Controller reservation: owner `47503`, token
`3b0bba158666a2c4f9fb6b7fe2a4d1c4`, scratch allowance 16,777,216 bytes.
Every stage first ran `test ! -e <absolute-stage>` and then created distinct
`tmp`, `xdg-cache`, `xdg-config`, `xdg-data`, `mpl`, `hypothesis`, `archive`, and
`pytest` children. Prefixes were preserved; no cleanup deletion occurred.

Every pytest command used this exact launcher, with `STAGE` replaced by the
absolute stage shown below and `SELECTORS` replaced by the exact table entry:

```text
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONHASHSEED=0 TMPDIR=STAGE/tmp TMP=STAGE/tmp TEMP=STAGE/tmp XDG_CACHE_HOME=STAGE/xdg-cache XDG_CONFIG_HOME=STAGE/xdg-config XDG_DATA_HOME=STAGE/xdg-data MPLCONFIGDIR=STAGE/mpl HYPOTHESIS_STORAGE_DIRECTORY=STAGE/hypothesis SILENT_CASCADE_ARCHIVE_TEST_SCRATCH=STAGE/archive .venv/bin/python -B -m pytest --basetemp=STAGE/pytest -q -x --tb=short -p no:cacheprovider SELECTORS
```

The absolute stage prefix was
`/Volumes/git/legotin/silent-cascade/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-cold-numerics/`.

| Stage | Exact selectors | Result | Allocated/files/dirs |
| --- | --- | --- | --- |
| `red-1` | `tests/pilot/test_pilot_cold_numerics.py::test_genuine_numeric_report_matches_cold_read` | Genuine RED: eager completed; cold failed at absent logical report, 1 failed in 4.40s | 0 B / 0 / 12 |
| `green-1` | prior selector plus `tests/pilot/test_pilot_cold_semantics.py::test_genuine_continuation_and_offline_results_match_cold_reads` | 2 passed in 9.49s | 0 B / 0 / 14 |
| `red-2` | `tests/pilot/test_pilot_cold_numerics.py::test_genuine_numeric_verifier_and_runtime_rows_match_cold_reads` | Setup RED only: incorrect fixture checkpoint binding, 1 failed in 0.84s; not counted as behavior RED | 0 B / 0 / 10 |
| `red-2b` | same verifier/runtime selector | Genuine RED: eager verifier/rows completed; cold runtime DONE was absent, 1 failed in 4.84s | 0 B / 0 / 12 |
| `green-2` | verifier/runtime selector plus prior continuation/offline selector | 2 passed in 13.22s | 0 B / 0 / 14 |
| `red-3-partial` | `::test_missing_capture_tensor_validates_available_operations` and `::test_missing_capture_tensor_cannot_hide_corrupt_operations` in the cold-numerics file | Genuine RED: available cold operations were mislabeled unavailable, 1 failed in 0.94s | 0 B / 0 / 11 |
| `green-3` | full `tests/pilot/test_pilot_cold_numerics.py` plus prior continuation/offline selector | 7 passed, then fixture guard setup failure in 18.03s; production behavior not implicated | 3,489,792 B / 12 / 43 |
| `green-3b` | same full matrix | 9 passed in 20.67s | 3,489,792 B / 12 / 45 |
| `green-4-raw` | genuine report selector plus prior continuation/offline selector | 2 passed in 10.16s after restoring exact raw batch-recipe hash binding | 0 B / 0 / 14 |

The full control stage guarded all sizes before writes: four final episode
members at no more than 3,366,912 rounded bytes, one operations rewrite at no
more than 65,536 rounded bytes, six runtime metadata files at 81,907 logical /
94,208 rounded bytes, one 4,096-byte extra-inventory control, 131,072 bytes of
administration, and at most 64 directories charged at 4 KiB. The prospective
total was 3,923,968 bytes, below 5 MiB. Actual retained Task 5 scratch is
6,979,584 bytes (6,816 KiB), entirely in preserved `green-3` and `green-3b`
controls; all other stages allocated zero bytes.

No complete 83 MiB numerical tree or 12 MiB tensor capture was copied.

## Read guards, cleanup, and execution tripwires

The strict fixture guarded both `Path.open` and descriptor-relative `os.open`.
Cold success, unavailable and failure paths all ended with
`payload_active == 0` and `active == 0`; maximum simultaneous payload leases was
one. Advertised missing/corrupt members raised, late inventory failure occurred
before reads, and wrong-root requests produced no lease requests.

Tripwires rejected calls to `EventFlowModel.forward`, `EventEngine.run_until`,
`EventEngine.run_episode`, neural replay, causal replay, pilot training, and
generic training. All genuine reader tests passed with those tripwires active.

## Static checks and self-review

Commands and results after implementation:

```text
git diff --check
# exit 0

.venv/bin/ruff check --no-cache src/silent_cascade/train/pilot_verification.py src/silent_cascade/train/pilot_evidence.py tests/pilot/cold_semantics_fixture.py tests/pilot/test_pilot_cold_numerics.py
# All checks passed!

.venv/bin/ruff format --check --no-cache src/silent_cascade/train/pilot_verification.py src/silent_cascade/train/pilot_evidence.py tests/pilot/cold_semantics_fixture.py tests/pilot/test_pilot_cold_numerics.py
# 4 files already formatted
```

Self-review confirmed filtered discovery rather than retention of the whole
training inventory, independent runtime closure derivation, raw-byte recipe hash
preservation, sequential lease lifetime, no exported cache path, unchanged
tolerances/codecs, original false/MPS status, and complete/partial verifier
behavior. Full `make verify` was not admitted.

## Scoped files and commits

Implementation commit: `d4cd56ebdc27e5ecf2a9065725d47476494aaaf5`
(`feat: verify cold numerical evidence`). It contains only:

- `src/silent_cascade/train/pilot_verification.py`
- `src/silent_cascade/train/pilot_evidence.py`
- `tests/pilot/cold_semantics_fixture.py`
- `tests/pilot/test_pilot_cold_numerics.py`

This report is committed separately. Controller-owned public-plan and progress
changes were neither staged nor committed by this writer.

## Limitations

The historical CPU smoke evidence does not establish MPS parity, empty-optimizer
bootstrap, current-source diagnostic production, 64 production runtime episodes,
a complete current-source gate, recovery, or provider/transport integration.
Primary scientific execution remains offline with exactly zero foundation-model
calls; no new scientific execution occurred in this task.
