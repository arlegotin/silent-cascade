# Task5 complete rows in abandoned cold evaluations

Date: 2026-09-19
Implementation base: `1dbdaaa`
Scope: the approved next bounded slice only

## Result and source scope

The cold partial reader now accepts newline-complete rows only after the same
row, neural sidecar, full trajectory, and crash checks used for local evidence.
`eval/artifacts.py::_verify_evidence` accepts an optional
`raw_reader(reference) -> bytes`; its default remains the existing bounded local
read. Every semantic check following the raw reads is unchanged.

`report/pilot_artifacts.py::load_abandoned_evaluation` supplies a closure using
the original training-inventory snapshot. It checks exact name membership,
consumes the named file inside `evidence_path`, and checks its original digest
on every read, including the semantic reread after initial inventory validation.
The closure returns bytes, never a deferred leased Path. The temporary blanket
complete-row refusal is removed. No session operation, transport schema,
EpisodeCommit, or whole-evaluation materialization was added.

The focused reader tests parameterize the original real producer case across
zero rows, a retained successful row, and a retained initialization failure.
They compare the local default-reader summary with the real cold report and
literal counts: retained rows 0/1/1; retained errors 0/0/1; unknown episodes
1/0/0; trailing bytes 16 in every case. All remain abandoned_incomplete, with no
DONE or episode-binding commitment. Maximum active evidence leases remains one,
and no leases remain after each read. The previous refusal test is renamed
`test_cold_partial_rejects_invalid_newline_complete_row` and requires actual
row-schema rejection rather than blanket rejection of supported valid rows.

## Exact retained fixtures

`tests/fixtures/pilot/retained_partial_evaluations.json` stores original text
as JSON strings and original gzip bytes as base64. Decoding asserts every
recorded byte count and SHA-256. The common identity reuses the existing
canonical identity fixture (7,111 decoded bytes; SHA-256
`26583e594139ab8be066ad7144f0f1e0fc3586d5fdc133838626bfdce985e647`).
No weights, neural data, outcomes, or gzip streams were regenerated.

Source roots under the frozen plan workspace, read only:

- success: `tmp/task-5/milestone-cover-1/test_serialized_rows_reconstru0/eval`
- initialization error: `tmp/task-5/milestone-cover-1/test_zero_event_failure_requir0/eval`

The controller independently checked the seven original hashes and absence of
machine-specific paths before admission. This implementation read those bytes
without modifying their frozen sources. Payload totals including shared identity
are 49,709 and 12,037 bytes. The real partial fixture publishes the unchanged
retained row prefix as .rows.pending.jsonl and adds a separate 16-byte unfinished
tail. The initialization failure deliberately omits the finished crash index to
represent pre-index publication, preserving its real manifest and row reference.

| Case | Logical fixture member | Decoded bytes | SHA-256 |
| --- | --- | ---: | --- |
| success | rows.jsonl | 3730 | `f9a3192be3ec3dd385eb5410760e16cc5c5809e731c8e85638b17ebad8ab2e49` |
| success | episodes/00000.neural.json | 35989 | `f7440d7aaca8564d27b61703353c394d36fd4ed7ccd3d799018915200c231a11` |
| success | episodes/00000.trajectory.json.gz | 2879 | `fa62216bb743e9887f236980f2ed4e36a44479adc9c67f761af74bc763534581` |
| initialization-error | rows.jsonl | 3140 | `ec7d4e3d2251ed9e4466e168d8684320439a97a079551bcd47a116de97a5b379` |
| initialization-error | episodes/00000.neural.json | 921 | `5e7154207ef9af6a9ab6d81a297de51c5b3d3c7310b32cf4a395ca6d1efd3e30` |
| initialization-error | episodes/00000.trajectory.json.gz | 204 | `5b26f0ed8910f51748f8a30bdb5c359f43fc82a89c63f7de0883ebd74c1fda71` |
| initialization-error | crashes/68e9c3ddb4954b43897ea6416d94ceca.json | 661 | `66d121247e91a87b892c527a73e6931c254c1a02049ecdfc7a8c9b1756de2909` |

## Admission, retention, and commands

The controller admitted 32 MiB additional scratch under the existing same-ledger
opaque scratch binding, owner 66017 / PID 47779 at 1789770209.048225. Historical
allocation remained charged (scratch 205,717,504 bytes, logs 1,105,920 bytes before
the grant). No baseline, quota, prefix, or reservation reset occurred.

Every stage is a unique child of:
`/Volumes/git/legotin/silent-cascade/.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-complete-rows`.

Each real producer command was admitted separately at 16 MiB: an 8 MiB workspace,
4 MiB fake remote, less than 512 rounded entries, and administrative headroom.
Before each command, actual cumulative retained allocation plus 16 MiB plus
2 MiB administration remained below 32 MiB. The fixture makes four publications
(two journal units, one stopped partial unit, one control snapshot). Stable
real cases measured 79 files, at most 58 directories, and 46 remote files,
within the conservative fewer-than-160 files/directories estimate.

The first strict group was admitted at 512 KiB plus 256 KiB administration.
Its measured 19 files and 33 directories include pytest/environment directories;
the initial 30-directory estimate omitted three administrative directories.
Actual allocation was 160 KiB, within the admitted total. The stable strict
group was separately admitted at 1 MiB and measured 32 files/44 directories,
216 KiB. All failed and successful output remains retained.

All pytest invocations used the following exact argument/environment form.
`stage` is the absolute root above plus the stage name in the table; the
selectors are listed immediately below the table. Each invocation first ran
`mkdir -p "$stage/tmp"`. No other caches, temporary roots, or default basetemp
were used.

```sh
env TMPDIR="$stage/tmp" TMP="$stage/tmp" TEMP="$stage/tmp" \
  XDG_CACHE_HOME="$stage/xdg-cache" MPLCONFIGDIR="$stage/mpl" \
  HYPOTHESIS_STORAGE_DIRECTORY="$stage/hypothesis" \
  SILENT_CASCADE_ARCHIVE_TEST_SCRATCH="$stage/archive" \
  SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q --tb=short -p no:cacheprovider \
  --basetemp="$stage/basetemp" <selectors>
```

| Stage | Selectors | Result | Allocated bytes |
| --- | --- | --- | ---: |
| red-success | A | 1 failed, 3.63s; expected explicit cold-row refusal | 352256 |
| green-success | A | 1 passed, 3.68s | 364544 |
| green-initialization-error | B | 1 passed, 3.87s | 323584 |
| green-zero | C | 1 passed, 3.09s | 319488 |
| red-digest | D | 1 failed, 0.36s; expected DID NOT RAISE | 53248 |
| green-strict | E F | 5 passed, 0.28s | 163840 |
| stable-success | A | 1 passed, 3.75s | 364544 |
| stable-initialization-error | B | 1 passed, 3.88s | 323584 |
| stable-zero | C | 1 passed, 3.19s | 319488 |
| stable-strict | E F G H | 7 passed, 0.86s | 221184 |

Exact selectors (quote each parameterized selector as shown):

```text
A: 'tests/pilot/test_pilot_archive_readers.py::test_cold_partial_report_uses_authenticated_inventory[success]'
B: 'tests/pilot/test_pilot_archive_readers.py::test_cold_partial_report_uses_authenticated_inventory[initialization-error]'
C: 'tests/pilot/test_pilot_archive_readers.py::test_cold_partial_report_uses_authenticated_inventory[zero]'
D: 'tests/pilot/test_pilot_archive_readers.py::test_cold_partial_rejects_invalid_retained_evidence[digest-reread-artifact integrity mismatch]'
E: tests/pilot/test_pilot_archive_readers.py::test_cold_partial_rejects_invalid_retained_evidence
F: tests/pilot/test_pilot_archive_readers.py::test_cold_partial_rejects_invalid_newline_complete_row
G: tests/pilot/test_pilot_archive_readers.py::test_cold_partial_rejects_unbound_predecessor
H: tests/pilot/test_pilot_archive_readers.py::test_cold_root_training_result_is_discovered_from_authenticated_inventory
```

Total retained scratch: 2,805,760 allocated bytes (2,740 KiB), measured with
`du -sk` and independently summed filesystem blocks. No test process identity
was exposed by the synchronous runner; all commands exited normally except the
two expected RED commands.

Independent review corrected the original total's arithmetic overstatement of
20 KiB. The per-stage table is unchanged; no retained output was deleted.

The first feature test failed at the explicit refusal before production edits.
The second RED was a targeted mutation check: temporarily replace the closure's
original-digest condition with false, run selector D, observe acceptance of
altered evidence, and immediately restore the condition before GREEN. The
test changes only the staged sidecar after the initial inventory read, adds
whitespace, and also substitutes its row digest. Thus the row's own hash check
cannot mask a missing original-training-digest check. Fixture originals remain
unchanged.

Other adversarial cases rehash a sidecar whose observation count is wrong,
reference a non-inventory path alias, and remove a required crash manifest.
These still reject at the existing semantic, exact-membership, and crash checks.
The stable cover ran after the final Python formatting correction.

Scoped static commands:

```sh
.venv/bin/ruff check --no-cache src/silent_cascade/eval/artifacts.py src/silent_cascade/report/pilot_artifacts.py tests/pilot/test_pilot_archive_readers.py
.venv/bin/ruff format --check --no-cache src/silent_cascade/eval/artifacts.py src/silent_cascade/report/pilot_artifacts.py tests/pilot/test_pilot_archive_readers.py
git diff --check
```

All passed. Initial Ruff inspection found one 105-character test line; it was
split mechanically before stable verification.

## Self-review and limitations

Reviewed the complete source/test diff: local default reads and every semantic
check are preserved; the cold closure captures the original exact inventory;
reads remain bounded and sequential; incomplete status and unparsed tails remain;
row identity, extra/conflicting rows, crash identity/source/checkpoint/index checks
remain in place. The tests use real producer custody/publication for success/error
and zero rows. Strict contexts only isolate adverse evidence changes. The existing
cold scanner, journal foundation, and session API are unchanged.

This is one completed implementation slice pending controller review, not complete
Task5 or Phase4. No make verify, full suite, neural execution, CPU replay/resume,
recovery-source authentication, provider qualification, network operation, or
scientific run was performed. Aggregate/check/gate caller threading, cold recovery,
CPU equality, and the remaining approved local gates remain separate obligations.
The controller performs independent review after the scoped commit.
