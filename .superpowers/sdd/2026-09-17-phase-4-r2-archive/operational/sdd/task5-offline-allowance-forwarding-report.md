# Task D1 — explicit offline allowance forwarding

Status: scoped implementation committed; controller frozen repeat and independent
review remain outside this writer handoff.

## Identity and scope

- Base: `a8bf9e095f99ade4e639b627eb28f30cf4d62b2f`.
- Commit: `39ecdf451090ae5c81d984f6e7a7c0c7166ac8a2`
  (`feat(pilot): forward offline write allowance`).
- `pilot_workflow.py`: SHA-256
  `e025ce07488472d15ca7fba04d75241ca2f1acf037915d6d582179360accbee3`.
- `pilot_checks.py`: SHA-256
  `2456a3cbfcf1b1c05d4b348496102f914925e304ba3540af41d896df351c4213`.
- `pilot_verification.py`: SHA-256
  `28a8ad0f4f16f5fb3db3a8c20705c8bd1404c713e82bcb7766fb381325b64259`.
- `pilot_offline_process.py`: SHA-256
  `12998aab01a755efdc8fb2262b87732523fdae2b17a16fe81c378ae47b07717a`.
- `test_pilot_offline_process.py`: SHA-256
  `e827c0e0aa6a24d927be9098fe0cd75464b5d3f83ba299aeb86e06cde6fb3c7b`.

The source change adds optional `offline_write_allowance=None` to the six
approved custody APIs and forwards the identical object through every existing
edge. The two low-level measurement edges use `write_allowance`. All four
preflight sites pass the object. `preflight_offline_process` validates an exact
`OfflineWriteAllowance` type before completed-artifact inspection and existing
custody checks.

## Bounded command and admission

The exact RED/GREEN driver is retained at
`operational/sdd/task5-offline-allowance-forwarding-check.py`, SHA-256
`87abbd1dacc24300d8f002abc06b4157e3c2abd15191598d737f062cddc30366`.
It records the full closed environment and exact command arrays. Pytest selected
nine named nodes in `test_pilot_offline_process.py`, expanding to 15 cases;
the whole file was not selected. Tests used spies only: zero child, Git, neural,
model, training, evaluation, replay, report, recovery-copy, or provider process.

Each fresh stage was admitted for at most 655,360 bytes, 24 file/link names
(RED used the tighter 16-name bound), and 40 directories. The driver enforces
150 seconds per command, 32,768 bytes per stream, 65,536 bytes total output,
and retains every stdout/stderr stream. TMP/XDG/MPL/Hypothesis/archive roots
were confined under the fresh stage. The entire task retains 24,576 allocated
bytes against its unchanged 2 MiB task allowance.

## RED

Prefix: `operational/scratch/task5-offline-allowance-forwarding/red1`.
Pytest exited 1 after 2.04 seconds (2.466 seconds wrapper wall): 14 expected
`TypeError` failures for the missing approved keywords and one passing omitted-
default case. There were no other failures. Full stdout is `red1/0-stdout.log`
(6,830 logical bytes; SHA-256
`df02f34d16a948bd6a3a8b626386b5c1db3856e72bbb1940b8d449ecde565833`);
stderr is empty (SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).

Actual post-run lstat inventory: two regular files, nine pytest symlinks and 22
directories. Only `0-stdout.log` allocated blocks: 16 blocks / 8,192 bytes.
`0-stderr.log`, every symlink and every directory had zero allocated blocks.
No attempt was removed or reused.

## GREEN and scoped statics

Prefix: `operational/scratch/task5-offline-allowance-forwarding/green1`.
The bounded driver ran these five commands in order:

1. Scoped Ruff format: exit 0; one test file reformatted, four files unchanged.
2. The same nine pytest selectors: exit 0; `15 passed in 1.42s`
   (1.841 seconds wrapper wall).
3. Scoped Ruff check: exit 0; `All checks passed!`.
4. Scoped Ruff format check: exit 0; `5 files already formatted`.
5. Scoped `git diff --check`: exit 0; no output.

All five stderr logs and `4-stdout.log` are empty. The nonempty stdout logs are:

- `0-stdout.log`: 43 logical / 4,096 allocated bytes, SHA-256
  `be228cc22ccf7a5e14a1482327c7e542160d0dd5947a71e13b821729bde1746f`.
- `1-stdout.log`: 99 / 4,096, SHA-256
  `01541be033cc71e93c6debb066dab44cf43cfc2f8ae0f5fc3a73774bc065ac0c`.
- `2-stdout.log`: 19 / 4,096, SHA-256
  `82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18`.
- `3-stdout.log`: 26 / 4,096, SHA-256
  `f114d83b30c5c657ee43a847a45bafccaea255d82673feb87c26d252c61e29c1`.

Actual GREEN lstat inventory: ten regular files, nine pytest symlinks and 22
directories; 187 logical and 16,384 allocated bytes. Every empty log, symlink
and directory had zero allocated blocks.

## Self-review and exclusions

- Tests use `is` at every transport edge, so copying/coercion fails the suite.
- Mapping and derived-dataclass inputs fail before reuse inspection or caller
  tripwires. Omitted and explicit `None` preserve existing behavior; completed
  safe reuse skips custody without treating the allowance as launch authority.
- Public checks and recovery reach the final owned-check edge. The recovery
  case is intentionally wiring-only: gate/envelope/decode/config/ownership,
  publication and authentication are stubbed; no destination or synthetic
  scientific output is created. It does not claim to validate recovery copying.
- No allowance is inferred or defaulted, no custody/allowance pair is required,
  and no capability, hold, report field, schema, CLI, manifest, session,
  scientific setting or policy changes.
- Diagnostic02 was not repeated. There was no actual offline diagnostic,
  numerical/Git execution, provider/network access, full-file test, full suite,
  or `make verify`.
- Delegated custody/private descriptors, production hold equality, inherited
  supervisor integration and the full local/scientific gates remain outstanding.

No failed implementation or GREEN command occurred beyond the intended RED.
The commit contains only the five owned source/test files; controller progress,
ledger and admission documents were not staged.
