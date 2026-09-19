# Task 5 artifact-only gate inputs and reuse admission

## Scope and result

Implemented only the approved bounded gate-input slice:

- `pilot_evidence._verify_available_training` now accepts an optional evidence
  context, rejects a mismatched or missing logical run root before discovery,
  fully exhausts authenticated discovery, leases/decodes genuine CPU training
  archives and portable weights one at a time, validates the complete available
  journal chain and each available dependency independently, restores RNG state,
  and calls the existing context-aware `_durable` only for a complete closure.
- Discovery retains only descriptors, `checkpoint-index.json`, journal names,
  and—on a second exhausted inventory pass—the exact dependency names learned
  from the authenticated journal records. It does not retain every `attempt-*`
  descendant.
- `compact_attachment_records` keeps one shard lease through hash validation,
  complete row iteration, and the final row-count check. Exhaustion, error and
  explicit generator close release it. A context for a different root is
  rejected, so a missing export shard cannot fall back to a raw-run namesake.
- `pilot_checks._require_reusable_gate` rejects either raw or semantic
  incompleteness and is used by the existing reuse branch. It deliberately does
  not require `passed=True`; recorded `debug_non_acceptance` and `failed`
  outcomes remain reusable when verification is otherwise complete.

Source/test commit:
`b417c4b14aa286cd0cec63990d354d84ea7e9b2e`
(`feat(pilot): authenticate cold gate inputs`).

## Evidence identity and custody

All genuine evidence was borrowed read-only from:

`.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-4/cover-pilot-final/test_real_smoke_workflow_reuse0/repo/runs/smoke`

- Historical source revision:
  `42b8ab0ba8a647794b08ce29327fce105e3a75f7`.
- Recorded outcome: `debug_non_acceptance`.
- `selected_checkpoint` is absent. The tested archive is the actual
  `progress.latest`,
  `training-4-e55caab7431984fdb47204eff898af4c04f4aa91adaaf2cf60efb9f74703c357.safetensors`.
- Latest archive SHA-256:
  `e55caab7431984fdb47204eff898af4c04f4aa91adaaf2cf60efb9f74703c357`.
- Latest portable weights SHA-256:
  `e102be3e3d9c04b9e939ca763d4b6ac1e8a768c62ead350668271332bbf8e094`.
- Journal head:
  `be8b80f90dd825b1addfdc520b15ee0ea88c0da0b9a16879f189789d270e53da`;
  the genuine chain has six records (four update, two validation) and 117
  journal dependency entries.
- Genuine primary compact shard: 242,409 logical bytes, 16 rows, SHA-256
  `b281ceadbf31240dbf92267ef7249ec9f5b05677bf0cffc946d120475d649325`.

The strict test adapter generated only a typed journal manifest and inventory
from the exact original six journal bytes/hashes. Its `JournalSegmentCommit`
matches the original chain. This generated metadata is not an original remote
receipt, archive transport proof, or evidence of a newly executed run. The
original files were never written, moved, linked, copied or deleted.

No current-source `authenticate_run`, full gate, training fit, diagnostic,
provider, model forward, event execution, replay, recovery or collection ran.
Tripwires prohibited those execution boundaries. Genuine CPU artifact decoding
and existing forensic comparisons did run.

## TDD and command evidence

Every pytest invocation used this closed launcher, with a fresh absent `stage`:

```sh
env TMPDIR="$stage/tmp" TMP="$stage/tmp" TEMP="$stage/tmp" \
  XDG_CACHE_HOME="$stage/xdg-cache" XDG_CONFIG_HOME="$stage/xdg-config" \
  XDG_DATA_HOME="$stage/xdg-data" MPLCONFIGDIR="$stage/mpl" \
  HYPOTHESIS_STORAGE_DIRECTORY="$stage/hypothesis" \
  SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$stage/archive" \
  SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONHASHSEED=0 \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /usr/bin/time -p .venv/bin/python -B -m pytest -q --tb=short \
  -p no:cacheprovider --basetemp="$stage/pytest" SELECTORS
```

The stage was refused if already present; all stages and failures remain
preserved. Exact selectors/results:

| Stage | `SELECTORS` | Result and cause | pytest / wall seconds | Actual files / dirs / logical / allocated |
|---|---|---|---|---|
| `red-training-equivalence` | `tests/pilot/test_pilot_cold_gate_inputs.py::test_genuine_training_inputs_match_cold_reads` | RED: cold returned `training.archive:latest`, `training.weights:latest_weights`, `training.durable_journal`; eager returned no unavailable labels | 1 failed in 1.97 / 4.18 | 1 / 12 / 797 / 4,096 |
| `green-training-equivalence` | same | Expected adapter-contract failure after production reached `_durable`: reviewed strict fake lacked the real journal lease `ref` required by `iter_journal_records` | 1 failed in 2.24 / 4.41 | 1 / 12 / 1,185 / 4,096 |
| `green-training-adapter` | same | GREEN after adding bounded typed metadata for the genuine journal commitment; no semantic-reader bypass | 1 passed in 2.43 / 4.51 | 3 / 15 / 2,561 / 12,288 |
| `red-compact-reuse-root` | `'tests/pilot/test_pilot_cold_gate_inputs.py::test_training_context_requires_exact_run_root_before_reads[none]' tests/pilot/test_pilot_cold_gate_inputs.py::test_genuine_compact_attachment_matches_cold_read tests/pilot/test_pilot_cold_gate_inputs.py::test_reusable_gate_requires_complete_recorded_evidence` | RED: root agreement was skipped; cold compact resolved an absent logical file (not a keyword mismatch); reuse predicate was a separately classified new-API import failure | 6 failed in 0.97 / 2.96 | 3 / 16 / 6,754 / 16,384 |
| `green-compact-reuse-root` | prior three selectors plus `tests/pilot/test_pilot_cold_gate_inputs.py::test_genuine_training_inputs_match_cold_reads` | GREEN, including changed exact two-pass training discovery | 7 passed in 2.51 / 4.62 | 3 / 19 / 2,561 / 12,288 |
| `adverse-matrix` | `tests/pilot/test_pilot_cold_gate_inputs.py` | GREEN: complete adverse and cleanup matrix | 19 passed in 2.64 / 4.59 | 9 / 50 / 2,662 / 36,864 |
| `focused-cover` | `tests/pilot/test_pilot_cold_gate_inputs.py tests/pilot/test_pilot_evidence.py` | Final focused new plus existing eager regression coverage | 39 passed in 2.95 / 5.05 | 31 / 72 / 3,521 / 126,976 |

The focused-cover CPU times were user 3.32 seconds and system 0.90 seconds.
Across every retained Task 5 gate-input stage: 51 files, 197 directories,
20,041 logical bytes and 212,992 allocated bytes, below the accepted 4 MiB
prospective rounded envelope. The largest generated metadata members were the
1,416-byte inventory and 1,018-byte manifest. Corrupt controls were 18, 3, 3,
37 and 37 bytes; the invalid reuse-branch gate was 2 bytes. The fixture policy
enforced a 64 KiB metadata and 16 KiB page limit before sealing.

Fresh post-test static and diff checks:

```sh
.venv/bin/ruff check --no-cache \
  src/silent_cascade/train/pilot_evidence.py \
  src/silent_cascade/train/pilot_checks.py \
  tests/pilot/test_pilot_cold_gate_inputs.py
.venv/bin/ruff format --check --no-cache \
  src/silent_cascade/train/pilot_evidence.py \
  src/silent_cascade/train/pilot_checks.py \
  tests/pilot/test_pilot_cold_gate_inputs.py
git diff --check
```

Results: `All checks passed!`, all three files already formatted, and no diff
whitespace errors.

## Coverage and self-review

Focused tests exercise genuine eager/cold equivalence, exact historical source
labels, actual CPU codecs, RNG restoration, one-payload maximum, fully exhausted
inventory, wrong and absent context roots before reads, late inventory failure,
advertised lease failure, legacy missing labels, and a missing dependency paired
independently with a corrupt available checkpoint, journal, and journal artifact.

Compact tests exercise genuine 242,409-byte shard equivalence, final hash and
row-count failures, exhaustion/error/explicit-close cleanup, and export
isolation. Reuse tests cover all four missing/unavailable combinations, both
nonpassing recorded outcomes, and actual use in the existing reuse branch.

Self-review found no writes outside the owned source/test/report and admitted
fresh operational stages. The controller-owned `progress.md` was neither staged
nor modified by this worker. No generic framework or scientific schema was
added. Existing eager callers retain their default behavior.

## Deliberate limitations

This slice does not forward context through the outer aggregate verifier, bind
the final reuse reread/hash comparison, integrate collection/execution callers,
or implement recovery. It does not claim that the historical source passed the
current full gate. The historical artifact remains a negative/current-source
non-acceptance reference with no selected checkpoint. A genuinely produced
current-source gate is still required for positive end-to-end evidence.

Full `make verify` was not admitted. No cleanup occurred: all RED, adapter-failure
and GREEN prefixes are retained for controller accounting and independent review.
