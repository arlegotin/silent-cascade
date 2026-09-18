# Completed-run cold authentication slice

Implemented the approved completed-run authentication/workflow reuse slice in
`train/pilot_evidence.py` and `train/pilot_workflow.py`. `authenticate_run` threads
the evidence context through training-result/index and durable journal checking,
then reads metadata, CPU checkpoint tensors and portable weights in separate
single-file leases. Its returned checkpoint path remains logical. `_train`
validates the context root, exhausts the validated inventory while retaining only
root/valid attempt-result candidates, unions resident candidates, and reuses the
real completed result without training. The workflow report receives its context.
Source/data checks, result/durable comparisons, eligibility and RNG restoration
remain intact.

## Verification and preserved outcomes

- Original guard checks: `5 passed, 1 deselected in 0.68s`.
- Sticky rejection regression RED: `1 failed in 0.71s`, expected
  `DID NOT RAISE AssertionError` when a caller caught the first rejection and
  retried a smaller write. Its written prefix remains retained.
- Sticky guard GREEN: `6 passed, 1 deselected in 0.67s`.
- Numerical RED: `1 failed in 28.44s`. Genuine four-update CPU fitting, both
  validations, eager authentication and completed eager reuse succeeded. The
  subsequent cold call failed with `FileNotFoundError: training-result.json`.
  This was a real missing-reader failure, not a keyword/signature mismatch.
- Numerical GREEN: `1 passed in 36.07s`. Eager/cold source, progress, descriptors,
  loaded manifests, logical checkpoint path, model identity and every portable
  model tensor agree. Python/NumPy/CPU Torch RNG is unchanged by authentication.
  Completed-root and completed-attempt reuse pass with a training tripwire.
  Foreign roots, last index-shard corruption, last inventory-member corruption,
  checkpoint corruption and portable-weight corruption are rejected. Original
  bytes are restored and the lease cache is empty after success/failure.
- Scoped Ruff lint and formatting and `git diff --check` passed. No full
  `make verify` was run. The real-fit test explicitly skips without admitted
  `SC_COLD_*` roots; an ordinary suite run does not reproduce this numerical case.

Each numerical fixture ran exactly four updates and validations at steps 2/4,
with 16 episodes each. Both retained outcomes are `step_ceiling`, not scientific
acceptance. Each validation had eight timed successes; step 2 had zero error
rows and step 4 had one error row. All 32 rows in each fixture record zero
foundation-model calls. Adverse outcomes and all failed prefixes remain intact.

| Identity | RED | GREEN |
| --- | --- | --- |
| Copied-source producer commit | `3551dce003f88baec7dc298056707b9c549fa8ac` | `80387ce353def197aad4cd25ad1f3acd8383541f` |
| Training commit | `3feb90269ba2c11a1848bb2bb62636845e3963fc` | `0638d984db3e38d9eb750521b4ac0d65303e00e4` |
| Authenticated source SHA-256 | `e89e7bdd847d52f67d987ed4ae303379a66b8526e876a95bbcdb1b55729e2c95` | `2eb6613aee868285a05425829ce3bccfb9b47ce4dfacd0f87887e6e235eea879` |

Both actual model-state hashes are
`92e724e6d11ed191438b8299625aa86a514287bf309c70bf6249f704981b3922`.
Each checkout has its own actual source/data-introduction/training Git history;
no source identity or completion status was substituted.

## Bounds and write coverage

All executions were admitted by controller owner `80995` in the existing shared
ledger. No additional ledger or provider operation was created. Output roots
are beneath this operational directory:

- `scratch/task5-cold-authentication/{red,green}`: exact source checkout/Git/data,
  isolated temporary paths and pytest administration.
- `spool/task5-cold-authentication/{red,green}`: scientific run, renamed backing
  inputs and bounded altered/repaired controls.
- `cache/task5-cold-authentication/{red,green}`: one renamed input lease.
- `logs/task5-cold-authentication/`: retained bounded command diagnostics.

Checkout admission recomputed 177 source files and a 10,788,864-byte upper bound
before creating the checkout, below the 12 MiB checkout allowance plus 2 MiB
outer administration. It checks regular dense single-link input files, exact
bytes/modes, 4 KiB filesystem units, tree/header/compression bounds, loose Git
objects, index/lock/metadata, and directory allowances. Git uses an empty
test-owned template, SHA-1, fixed local identity/dates, disabled hooks/signing/
automatic GC and disabled system/global configuration. The data publisher guard
admits exactly 16 fixed paths/calls: eight manifest copies and four audit reports
at most 24 KiB each, and four indexes at most 256 bytes each.

The test-only scientific guard checks before each publication/append:
current `st_blocks`, complete next temporary allocation, old-index and pending-
checkpoint coexistence, missing directories and temporary/final name allowances.
It uses the established `2 * rounded(size) + block` native-write cushion, not an
assertion of a universal filesystem allocation proof. Limits are 64 MiB live
aggregate, 16 MiB per input/cache lease, 256 live file names and 48 directories
across run/backing/cache. The recorded GREEN maximum prospective reservation
was 58,544,128 bytes. First rejection is sticky; later writes remain blocked even
if a scientific consumer catches the initial exception, and success requires a
clean guard. A budget-aborted prefix cannot be reported as a completed fit.

Coverage includes all imported `_publish_pilot_bytes` aliases in pilot data,
trainer, checkpoints, workflow and evidence types; trainer `_publish_bytes_at`;
the shared atomic `_durable_temp` used by evaluation, neural weights, trajectories
and crashes; direct `.rows.pending.jsonl` open/write; its final hardlink; and
directory creation. Production publisher/decoder logic and production limits
remain unchanged. Ordinary checkpoint retention stays enabled; no manual pruning
or evidence deletion was performed. The parent ledger was not measured during
temporary hardlink coexistence.

The strict test context moves only new fixture inputs into backing, then renames
one requested file into/out of the lease tree in `finally`. Real journals stay
resident. The actual checkpoint/index/weights, training envelopes/index shards
and inventory evidence were absent at their original cold paths. Maximum active
leases was exactly one; the context issues no provider or IPC qualification.

Measured retained allocation immediately after GREEN:

| Output | Scratch bytes | Spool bytes | Cache bytes | Log bytes |
| --- | ---: | ---: | ---: | ---: |
| Original guards | 12,288 | 0 | 0 | 4,096 |
| Sticky guard RED | 4,096 | 0 | 0 | 4,096 |
| Sticky guard GREEN | 12,288 | 0 | 0 | 4,096 |
| Numerical RED | 6,569,984 | 42,381,312 | 0 | 4,096 |
| Numerical GREEN | 6,569,984 | 42,397,696 | 0 | 4,096 |

These are measured fixture/log allocations, not the full parent ledger. Parent
metadata, report publication and later controller verification remain separately
charged in that same ledger. RED source checkout and failed evidence were not
reused as GREEN scientific input. A controller may independently rerun the same
GREEN fixture with its own copied source and `SC_COLD_REUSE=1`, avoiding a third
fit; that controller verification is not claimed here.

## Exact command forms

Commands ran from the repository root with `.venv/bin/python -B -m pytest`,
`PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `-q -s
-p no:cacheprovider --tb=short`. No pytest cache or bytecode was enabled.
The operational prefix below is the literal relative location expanded to its
absolute repository path at invocation:

```sh
SC_OPERATIONAL="$PWD/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational"
```

For guard executions, all six environment variables `TMPDIR`, `TMP`, `TEMP`,
`XDG_CACHE_HOME`, `MPLCONFIGDIR`, `HYPOTHESIS_STORAGE_DIRECTORY` named
`$SC_OPERATIONAL/scratch/task5-cold-authentication/<case>/tmp`.
`--basetemp` named that same `<case>/pytest`. Cases and exact selectors:

```text
guards:
  tests/pilot/test_pilot_cold_authentication.py -k fit_guard
guards-sticky-red:
  tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_rejection_remains_blocked_after_caller_catches_error
guards-sticky-green:
  tests/pilot/test_pilot_cold_authentication.py -k fit_guard
```

Each command redirected stdout/stderr to
`$SC_OPERATIONAL/logs/task5-cold-authentication/<case>.log`, with a 128 KiB
admitted log bound and 1 MiB admitted scratch bound per guard case.

Numerical commands used `<case>` equal to `red` then `green`; each had the
following exact environment/path/selector assignment (shell variables below
expand to the actual absolute values used):

```sh
SC_SCRATCH="$SC_OPERATIONAL/scratch/task5-cold-authentication/<case>"
env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  SC_COLD_SCRATCH="$SC_SCRATCH" \
  SC_COLD_SPOOL="$SC_OPERATIONAL/spool/task5-cold-authentication/<case>" \
  SC_COLD_CACHE="$SC_OPERATIONAL/cache/task5-cold-authentication/<case>" \
  TMPDIR="$SC_SCRATCH/temp" TMP="$SC_SCRATCH/temp" TEMP="$SC_SCRATCH/temp" \
  XDG_CACHE_HOME="$SC_SCRATCH/temp" MPLCONFIGDIR="$SC_SCRATCH/temp" \
  HYPOTHESIS_STORAGE_DIRECTORY="$SC_SCRATCH/temp" \
  .venv/bin/python -B -m pytest -q -s -p no:cacheprovider --tb=short \
  tests/pilot/test_pilot_cold_authentication.py::test_completed_fit_authentication_and_reuse_through_cold_inputs \
  --basetemp="$SC_SCRATCH/pytest" \
  > "$SC_OPERATIONAL/logs/task5-cold-authentication/<case>.log" 2>&1
```

The outer subprocess timeout was 600 seconds. Bounded child exception diagnostics
retain at most 24 stack frames plus 4 KiB of exception text; captured output is
checked against 256 KiB before pytest prints it. Numerical logs were admitted at
768 KiB each. No provider, full gate, production fit, MPS training, recovery
authority or broader caller-completion claim follows from these checks. Task 5,
Task 7 and Phase 4 remain incomplete; independent review follows this slice.
