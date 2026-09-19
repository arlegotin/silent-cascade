# Task 5 optimizer-cache compatibility correction

Status: implemented and committed; controller repeat and independent review remain
outside this writer handoff. No Diagnostic02, training update, model construction,
evaluation, replay, compiler execution, provider access, network access, full suite,
or `make verify` ran.

## Source and scope

- Historical diagnostic source: `f2ce7341f659a4113f4d27b27a17edcb98ab5516`.
- Previously reviewed Task C: `0aee7bd` (not changed or repeated).
- Immediate implementation base: `4d1d2bb9807a7dad02f56e63ead965454354044d`.
- Correction commit: `13fef3b8be474b7cd77768c5e8ff7b459e611130`
  (`fix(pilot): pin offline optimizer cache`).
- Owned files only:
  - `src/silent_cascade/train/pilot_offline.py`, SHA-256
    `e7ff2d965e7d78c3a9bb3060ce17dbd761fd42043a7605a7cc007b38603b92a1`.
  - `tests/pilot/test_pilot_offline_writes.py`, SHA-256
    `b1237e130fe11a57220d127f3bad57c23ce34ce6eb9722c7779ff7df96390ec6`.

The production change is one closed-environment entry:
`TORCHINDUCTOR_CACHE_DIR=str(scratch / "cache")`. It is derived after
`scratch.absolute()`, does not inherit an ambient override, and reuses the
existing admitted `cache` directory. There is no grammar, byte/name/directory
cap, public API, bootstrap, Torch/library, optimizer, or scientific change.

## Root cause and hypothesis

The retained Diagnostic01 evidence showed AdamW construction lazily importing
Torch Dynamo. Without an explicit Inductor cache root, installed Torch called
`tempfile.gettempdir()`, whose writability probe attempted a random file. The
existing offline writer boundary correctly denied and latched that write before
the first optimizer update. Installed Torch's explicit-cache branch creates the
cache root, while `DiskDynamoStore` construction itself does not create compiler
artifacts.

The tested hypothesis was that fixing the explicit cache root would permit only
the already-admitted empty directory, avoid the tempfile probe, and preserve
the existing denial latch for a cache file and a nested cache directory.

## RED

Fresh prefix:
`.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-optimizer-cache/red1`.
The exact executed launcher is retained byte-for-byte in `red1/command.txt`
(SHA-256 `cf7528ef6ba20c619390c9df91585381c808948fb13fc38919b6174a635ad983`).
It selected exactly these three nodes, expanding to four cases:

1. `test_offline_environment_derives_inductor_cache_and_ignores_ambient`
2. `test_real_scalar_adamw_initializes_only_empty_cache_root`
3. `test_optimizer_cache_artifact_denials_latch` (`file`, `directory`)

There were three sequential tiny Python controls, each with the existing
`OfflineWriteAllowance(65536)` defaulting to 102 names and nine directories,
and the existing 30-second/2048-byte-per-pipe capture. Each child program was
bounded by the helper's 4096-byte assertion. The outer admission was 768 KiB:
three 64 KiB child ceilings, 4096 bytes parent/pytest metadata, an 8192-byte
atomic/native cushion per 16 names plus 40 directories, and two 32768-byte
outer streams.

Result: exit 1, wall 3.981 seconds, pytest `2 failed, 2 passed in 3.87s`.
The environment test failed with the expected missing-key `KeyError`. The real
scalar CPU AdamW control (one scalar, `foreach=False`, `fused=False`) reached the
observed default-cache probe and produced the intentionally shortened terminal
`RuntimeError: offline write denied: writer`; captured child stderr was 141
bytes. Both negative controls passed: direct cache-file creation denied
`writer`, nested cache-directory creation denied `path`, and each denial latched.

Full stdout is retained at `red1/stdout.txt` (6453 logical bytes, SHA-256
`dc479c34c238270e60d42e1a26044f37c75d91e17d2518bed87e486a3b955842`);
stderr is the empty `red1/stderr.txt` (SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).

RED enforcement deviation: the executed launcher did not mechanically enforce
the admitted 32768-byte outer stream limit or 150-second outer timeout.
`exec_command` used an 8000-token output setting and a 30-second yield, neither
of which is that byte cap or a process timeout. The observed run nevertheless
completed in 3.981 seconds with 6453-byte stdout and zero-byte stderr, inside
both admitted ceilings. RED was not rerun; this deviation is preserved here
and in controller evidence.

Post-log RED lstat inventory: 27 names total: 21 directories, three pytest
symlinks, and three regular files. Every directory and symlink had zero
allocated blocks. `command.txt` had 8 blocks/4096 allocated bytes,
`stdout.txt` had 16 blocks/8192 allocated bytes, and `stderr.txt` had zero.
Total retained RED allocation is 12,288 bytes. The failed AdamW root contains
no cache directory; each negative-control root contains only an empty cache
directory. Nothing was deleted or reused.

## GREEN and scoped statics

The controller executed the bounded GREEN wrapper once under fresh `green1`.
Its exact full launcher, environment, four command argument arrays, output-cap
loop, timeout loop, and inventory assertions are retained verbatim in
`operational/sdd/task5-optimizer-cache-controller-evidence.md` (SHA-256
`6034f610dcafe3053fac08a9f9d2eca6ac3b971706fbbea4af601ad7c51f01b2`).
The wrapper enforced 32768 bytes per output stream and 150 seconds per command,
disabled plugin autoload/cacheprovider, used confined TMP/XDG/MPL/Hypothesis/
archive roots, and retained every output under `green1`.

Exact results:

- The three new nodes plus existing
  `test_closed_git_batch_pipe_still_works` expanded to five cases:
  `5 passed in 4.86s`; command wall 5.032 seconds; exit 0.
- `.venv/bin/ruff check --no-cache` on the two owned files: exit 0,
  `All checks passed!`, 0.061 seconds.
- `.venv/bin/ruff format --check --no-cache` on the two owned files: exit 0,
  `2 files already formatted`, 0.014 seconds.
- `git diff --check --` on the two owned files: exit 0, no output,
  0.010 seconds.
- Wrapper exit 0.

GREEN lstat inventory: 31 names total: eight regular log files, four pytest
symlinks, and 19 directories. All four stderr logs and `3-stdout.log` were
zero bytes/zero blocks. `0-stdout.log` was 98 logical bytes/4096 allocated,
`1-stdout.log` was 19/4096, and `2-stdout.log` was 26/4096. All directories
and symlinks had zero allocated blocks. Total GREEN is 143 logical and 12,288
allocated bytes. Cumulative retained task scratch is 24,576 allocated bytes.

## Failed operational attempts

- Expected RED failures are recorded above; no RED retry occurred.
- The first exact `git add`/`git commit` attempt was sandbox-blocked before an
  index lock could be created: `fatal: Unable to create '.git/index.lock':
  Operation not permitted`. The same two-file-only command was retried with
  approved Git escalation and created commit `13fef3b8be474b7cd77768c5e8ff7b459e611130`.
- No other implementation, test, formatter, or commit failure occurred.

## Self-review

- Removing the one environment entry makes both intended positive regressions
  fail for the observed reasons; the negative controls already pass and prove
  unchanged latching.
- The real control installs the irreversible boundary before constructing the
  real optimizer, creates one scalar CPU tensor only, and performs no optimizer
  step or compiler action. On GREEN, the only child-side addition is the empty
  admitted cache directory.
- The cache-file and nested-directory attempts are independently isolated in
  fresh interpreters because the first denial is intentionally irreversible.
- The existing closed Git batch control passes with the augmented trusted
  environment, covering exact environment equality used by the boundary.
- The two-file commit contains one production line and focused tests only.
  Controller ledger/docs dirt was neither staged nor committed.
- Diagnostic01 evidence and all RED/GREEN roots remain retained. No allowance
  was enlarged and no successful diagnostic or scientific claim is made.
