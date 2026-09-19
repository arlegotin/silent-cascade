# Task5 fixture-held offline writes (Task C) report

Status: source and focused test delivery, including the round-one directory
union correction, are complete at
`0aee7bdc3b99bf8cad7aec51ae5a12075c3ff75d`. Controller verification of the
initial delivery is complete; FIX1 controller repeat and re-review remain
separate. This report is deliberately not staged in a scoped source/test
commit.

## Delivered boundary

- `_custody_paths` now permits an admitted public run in spool while retaining
  the fixed control and `workspace/logs` exclusions and all configured
  logs/cache exclusions. The private route remains outside the public run,
  control root, and every spool/cache binding. Existing category-split and
  reservation checks remain in force.
- `FitGuard.installed()` now wraps the three already-bound publisher aliases in
  `pilot_checks`, `pilot_evidence`, and `report.pilot` with the existing
  publisher wrapper.
- `FitGuard.reserve_offline_attempt` accepts only the exact allowance, limits,
  and custody dataclasses; binds one canonical fresh spool root; validates the
  supplied custody at phase zero; and uses its actual immutable
  `category_peaks`. It holds
  `H = allowance.allocated_bytes + sum(category_peaks)` before intent or child
  creation, without creating a reservation, changing a binding, or granting
  child authority.
- The live category check charges retained category growth plus the remaining
  process peak and assigns the child allowance only to the public run's actual
  category. A descendant mapped to another category is rejected.
- The fixture hold includes the child allowance, the fourteen parent process
  names, missing public ancestors, and prepared private directories. In the
  focused fixture this is exactly 22 names and 14 directories: child 8/9,
  parent names 14, public ancestors 2, private directories 3.
- Outer admissions preserve the complete hold, exclude the held child tree
  from double counting while still validating its inventory, and cannot spend
  the hold. The four exact parent publications are checked against their
  process phases and record limits. A hold cannot be rebound or replenished.
- No default capacity, new ledger, new reservation, binding mutation,
  inherited authority, public private-capture hook, provider, network path,
  scientific algorithm, source inventory member, or full-gate behavior was
  added.

## Exact execution form

Every test stage used a fresh prefix below
`.superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/scratch/task5-offline-writes-c/`
and this closed outer launcher:

```sh
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 \
  PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH="$PWD/src" .venv/bin/python -B -
```

The launcher rejected an existing stage, created only its stage-local `tmp`,
XDG cache/config/data, matplotlib, Hypothesis, archive, and later pytest
basetemp paths, then started the exact command with `subprocess.Popen`. The
child environment set `TMPDIR`, `TMP`, `TEMP`, all three XDG roots,
`MPLCONFIGDIR`, `HYPOTHESIS_STORAGE_DIRECTORY`,
`SILENT_CASCADE_ARCHIVE_TEST_SCRATCH`,
`SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH=1`, `OMP_NUM_THREADS=1`, and
`MPL_IGNORE_SYSTEM_FONTS=1` to the fresh stage. It drained stdout and stderr
concurrently, enforced the admitted pipe and wall limits, joined every child,
wrote each log once with exclusive creation, and measured every retained path
with `lstat` and `st_blocks * 512`.

The exact pytest prefix was:

```text
.venv/bin/python -B -m pytest -p no:cacheprovider --noconftest -q
--tb=short --basetemp <fresh-stage>/pytest
```

RED1 and GREEN1 appended these exact selectors:

```text
tests/pilot/test_pilot_offline_process.py::test_privacy_custody_accepts_admitted_public_spool
tests/pilot/test_pilot_offline_process.py::test_privacy_custody_rejects_public_spool_one_byte_short
```

RED2 and GREEN2 appended:

```text
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_binds_existing_publisher_aliases
```

This expands to the `pilot_checks`, `pilot_evidence`, and `report.pilot`
parameter cases. Each target parent was created before guard installation, so
the missing alias wrapper—not a directory hook—caused RED.

RED3 and GREEN3 appended these exact eight selectors, expanding to 12 cases:

```text
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_reserves_exact_offline_hold_before_publication
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_offline_hold_cannot_be_rebound_or_replenished
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_offline_hold_rejects_short_inventory
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_live_child_bytes_are_not_double_counted
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_unrelated_output_cannot_spend_offline_hold
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_routes_parent_publications_through_held_process_peak
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_rejects_split_child_categories
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_validates_exact_offline_hold_bindings
```

The integration characterization appended exactly:

```text
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_real_parent_publication_creates_fresh_held_root
```

COVER-A appended these exact ten nodes, expanding to 31 cases:

```text
tests/pilot/test_pilot_offline_process.py::test_privacy_custody_accepts_admitted_public_spool
tests/pilot/test_pilot_offline_process.py::test_privacy_custody_rejects_public_spool_one_byte_short
tests/pilot/test_pilot_offline_process.py::test_privacy_custody_rejects_fixed_logs_even_with_spool_subbinding
tests/pilot/test_pilot_offline_process.py::test_privacy_custody_rejects_private_storage_overlap
tests/pilot/test_pilot_offline_process.py::test_privacy_custody_rejects_split_publication_categories_before_writes
tests/pilot/test_pilot_offline_process.py::test_privacy_custody_preserves_compatible_publication_categories
tests/pilot/test_pilot_offline_process.py::test_privacy_custody_rechecks_publication_categories
tests/pilot/test_pilot_offline_process.py::test_privacy_custody_pins_counted_private_directory
tests/pilot/test_pilot_offline_process.py::test_privacy_custody_rejects_unadmitted_targets
tests/pilot/test_pilot_offline_process.py::test_privacy_prepared_custody_cannot_survive_authority_change
```

COVER-B and COVER-B2 appended these exact 20 nodes, expanding to 27 cases:

```text
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_binds_existing_publisher_aliases
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_reserves_exact_offline_hold_before_publication
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_real_parent_publication_creates_fresh_held_root
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_offline_hold_cannot_be_rebound_or_replenished
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_offline_hold_rejects_short_inventory
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_live_child_bytes_are_not_double_counted
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_unrelated_output_cannot_spend_offline_hold
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_routes_parent_publications_through_held_process_peak
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_rejects_split_child_categories
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_validates_exact_offline_hold_bindings
tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_rejects_oversized_publisher_before_any_output
tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_counts_existing_and_pending_checkpoint_before_publication
tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_stops_pending_row_append_before_handle_write
tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_rejection_remains_blocked_after_caller_catches_error
tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_refuses_name_or_directory_growth
tests/pilot/test_pilot_offline_writes.py::test_install_cannot_replace_or_enlarge_allowance
tests/pilot/test_pilot_offline_writes.py::test_real_publishers_exact_capacity
tests/pilot/test_pilot_offline_writes.py::test_rows_stream_and_final_link
tests/pilot/test_pilot_offline_writes.py::test_atomic_replace_and_create_preserve_original_behavior
tests/pilot/test_pilot_offline_writes.py::test_closed_git_batch_pipe_still_works
```

The four post-pytest COVER-B2 commands were exactly:

```text
.venv/bin/ruff check --no-cache src/silent_cascade/train/pilot_offline_process.py tests/pilot/cold_authentication_fixture.py tests/pilot/test_pilot_offline_process.py tests/pilot/test_pilot_offline_writes.py
.venv/bin/ruff format --check --no-cache src/silent_cascade/train/pilot_offline_process.py tests/pilot/cold_authentication_fixture.py tests/pilot/test_pilot_offline_process.py tests/pilot/test_pilot_offline_writes.py
git diff --check -- src/silent_cascade/train/pilot_offline_process.py tests/pilot/cold_authentication_fixture.py tests/pilot/test_pilot_offline_process.py tests/pilot/test_pilot_offline_writes.py
rg -n -F -e '"src/silent_cascade/train/pilot_offline_process.py",' -e '"src/silent_cascade/train/pilot_offline_writes.py",' src/silent_cascade/train/pilot_evidence.py
```

The wording “five commands” counts pytest plus these four post-pytest commands.
COVER-B initially also ran `ruff format --no-cache` over the same four owned
files before pytest, for six total commands. COVER-B2 did not repeat that
mutating formatter.

## RED/GREEN results and retained accounting

All prefixes and logs remain retained. Earlier stage “names” counts combine
regular and symlink names. COVER-B inventories show them separately. The
failed COVER-B allocation uses the controller's per-name `lstat` measurement;
`du` deduplicated a hardlink block and was not used.

| Stage/prefix | Exact result | Wall | stdout/stderr | Logical | Allocated | Names | Dirs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `red1-public-spool-ba5e7530` | pytest 1; 2 failed as expected: blanket spool overlap, short case did not reach allowance | 0.529 s | 2457/0 B | 3935 B | 12288 B | 8 | 17 |
| `green1-public-spool-ba5e7530` | pytest 0; 2 passed in 0.29 s | 0.489 s | 98/0 B | 1584 B | 12288 B | 8 | 20 |
| `red2-alias-ba5e7530` | pytest 1; 3 `DID NOT RAISE` failures proving bound aliases bypassed the wrapper | 1.921 s | 1605/0 B | 1792 B | 16384 B | 6 | 15 |
| `green2-alias-ba5e7530` | pytest 0; 3 passed in 1.31 s | 1.711 s | 98/0 B | 284 B | 4096 B | 3 | 15 |
| `red3-hold-ba5e7530` | pytest 1; 12 missing-method failures | 0.860 s | 5612/0 B | 13592 B | 57344 B | 34 | 93 |
| `green3-hold-ba5e7530` | pytest 0; 12 passed in 0.73 s | 0.929 s | 99/0 B | 14265 B | 73728 B | 41 | 100 |
| `red4-real-parent-ba5e7530` | characterization passed unchanged: 1 passed in 1.52 s | 1.949 s | 98/0 B | 837 B | 16384 B | 7 | 19 |
| `cover-custody-ba5e7530` | pytest 0; 31 passed in 0.99 s | 1.184 s | 99/0 B | 19619 B | 139264 B | 76 | 171 |
| `cover-hold-static-ba5e7530` | formatter 0; 27 passed in 6.77 s; Ruff check 1 on two RUF043 test literals; later commands intentionally not run | bounded under 180 s/command | retained per command | 22867 B | 135168 B | 51 files + 20 links | 139 |
| `cover-hold-static2-ba5e7530` | 27 passed in 6.72 s; Ruff check/format, diff check, and inventory all 0 | 7.379 s total | pytest 99/0 B; static stderr empty | 21797 B | 139264 B | 55 files + 20 links | 139 |

Total retained implementer evidence is 100572 logical bytes and 606208
allocated bytes, with 329 regular/symlink names and 728 directories. Every
failed attempt remains present; no prefix was reused or cleaned.

After source freeze, the controller independently repeated the same disjoint
31 custody and 27 hold cases: MAIN-CUSTODY passed 31 in 0.98 seconds and
retained 139264 allocated bytes; MAIN-HOLD passed 27 in 6.80 seconds and
retained 126976 allocated bytes. A fresh MAIN-STATIC ran Ruff check, Ruff
format check, diff check, and exact equality to `44edf69`; all four commands
returned zero and retained 8192 allocated bytes. Thus all Task C attempts,
including failures and independent repeat, retain 880640 allocated bytes. The
controller's literal commands and measurements are recorded once in
`task5-offline-writes-c-main-evidence.md`.

The admitted envelopes were conservative source-derived maxima, not measured
headroom: RED/GREEN1 and RED/GREEN2 used 1536 KiB allocated, 256 KiB logical,
64 names, 64 directories, one simultaneous temporary, 16 KiB per pipe, and
120 seconds. RED/GREEN3 used 3 MiB, 512 KiB logical, 128 names, 160
directories, one temporary, 32 KiB per pipe, and 120 seconds. RED4 used 768
KiB, 128 KiB logical, 32 names, 40 directories, one temporary, 16 KiB per
pipe, and 120 seconds. COVER-A used 2752512 allocated bytes, 256 KiB logical,
96 names, 200 directories, one temporary, 32 KiB per pipe, and 120 seconds.
COVER-B/B2 used 3407872 allocated bytes, 512 KiB logical, 128 names, 220
directories, one temporary, 32 KiB per main-command pipe, and 180 seconds per
command. Its five tiny Python controls used 4096 bytes of child code, 2048-byte
child pipes, 30 seconds, and five 65536-byte child allowances; the Git control
added one closed read-only Git child. No application worker was started.

## Debugging and self-review

- RED1 failed for the intended old public-spool rejection. The one-byte-short
  assertion required the real `allowance exhausted` reason, preventing a false
  green from the blanket path error. GREEN reached that category accounting
  check.
- RED2 used precreated parents and empty-target assertions. All three failures
  were therefore the missing alias bindings, not the general `Path.mkdir`
  hook. GREEN used only the existing wrapper.
- RED3 first incorporated the reviewed directory-union ruling and the byte,
  name, and directory one-short cases. Every case failed because the hold API
  was absent. GREEN passed the exact same 12 cases.
- A concern that installed directory hooks would block a real held-root parent
  publication was tested with the actual descriptor-based publisher. The
  characterization passed unchanged, proving that the publisher uses its
  descriptor path rather than the patched `Path.mkdir`; no broad hook
  exception was added.
- COVER-B exposed only two raw-regex Ruff diagnostics in new tests. Only those
  two literals were made raw. COVER-B2 reran the same 27 cases and every static
  check from a fresh prefix.
- Self-review confirmed the public correction retains both the fixed logs root
  and configured logs/cache mappings, private custody retains spool/cache and
  control separation, the original unadmitted-spool negative remains, category
  peaks come from the actual custody, child split-category layouts fail,
  parent helper limits from Task B are unchanged, and held private directories
  are not charged a second byte peak.
- The four-file staged diff was reviewed and `git diff --cached --check`
  returned zero before commit. Controller-owned progress, brief, and plan
  changes were left unstaged.

## Initial delivery identities and source commit

The task started from reviewed base `e323533`; Task A source/fix commits already
present were `14abc50` and `07e9394`, while Task B source/fix commits were
`9f0c0b3` and `cc05f1f`. Final owned-file hashes:

```text
23da2a3d513412c529e8b8615398154c314f73ebe39a055f16c505d415a83ccf  src/silent_cascade/train/pilot_offline_process.py
a1381527123e5d92928359e61c443df655c404dd87138ece16636acfa07e8f90  tests/pilot/cold_authentication_fixture.py
041eae31468331119f7792d8aed75aa8f1f4af15c2d34ee057858a5d3f2211f4  tests/pilot/test_pilot_offline_process.py
905aabb4ab901799280a9fb8411208823325b85f55b6b873749ec3613a444e27  tests/pilot/test_pilot_offline_writes.py
```

Scoped commit:

```text
44edf6947aa6cf59d75c2bb31e8ae1a8bfaa9ee0
feat(pilot): reserve offline child and parent writes
```

Exactly the four owned source/test files were committed. This report and all
controller documents remain outside that commit.

## Limits and handoff

- This is primitive fixture/custody qualification. It did not call
  `create_checkout`, `run_fixture`, a real diagnostic, neural worker, fit,
  evaluation, replay, provider, network, full gate, or `make verify`.
- No completed scientific result, archive acceptance, full-run fit, or release
  readiness is claimed.
- The future production caller must explicitly supply the allowance and actual
  process custody after obtaining the existing global admission. This task did
  not implement or claim that supervisor handoff.
- The real actual-worker/full-gate integration remains separately admitted
  work. The controller's fresh repeat is complete; independent review remains
  the acceptance authority.

## FIX1: complete public-ancestor directory union

Round-one review found that the initial directory hold selected missing public
ancestors only when they were below `FitGuard.spool`. The valid layout
`FitGuard(output.parent, cache)` therefore counted `final` but omitted `run`.
It admitted 13 directories even though the brief requires the 14-path union:
child allowance 9, public ancestors 2, and prepared private directories 3.

The regression names the observable break directly. Its two fresh-ledger
layouts make both public ancestors wholly missing or precreate both ancestors
so `final` is simultaneously an existing retained spool root and a required
public ancestor. Each layout has a 13-directory rejection and a 14-directory
acceptance case. This independently covers missing versus existing ancestors
and deduplication of a shared path.

FIX1-RED used the report's closed launcher and exact pytest prefix with fresh
prefix `fix1-red-ancestor-union-ba5e7530` and this one selector, expanding to
four cases:

```text
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_counts_public_ancestors_outside_spool
```

The exact result was `FF..`: both 13-directory cases failed with `DID NOT
RAISE`, while both 14-directory controls passed. Pytest reported 2 failed and
2 passed in 0.46 seconds; wrapper wall was 0.656 seconds. The retained prefix
has 3334 logical bytes, 20480 allocated bytes, 10 regular files, one link, and
41 directories; stdout was bounded and stderr empty. Its admitted maxima were
1152 KiB allocated, 128 KiB logical, 48 names, 64 directories, one temporary,
16 KiB per pipe, 8192 bytes of launcher code, and 120 seconds. It launched one
pytest and no application/control child.

The correction now constructs required public ancestors from every output
parent below the custody workspace, independent of existence and spool
boundary. It inventories spool/cache with a set of concrete paths, so
overlapping roots count one shared directory while distinct hardlink names
remain distinct paths. The held directory paths are the set union of retained
spool/cache directories, required public ancestors, and prepared private
directories; the child allowance is added separately. Later outer accounting
does not count those directory paths a second time, but a fixed
reservation-time cushion preserves their existing 4 KiB-per-directory byte
charge. Initially missing public ancestors remain covered by P rather than
being charged twice. No cap, category, binding, process custody, or ledger
behavior changed.

FIX1-GREEN/COVER used fresh prefix
`fix1-cover-ancestor-union-ba5e7530`. Its exact pytest command appended these
16 nodes to the common prefix, expanding to 26 cases:

```text
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_counts_public_ancestors_outside_spool
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_binds_existing_publisher_aliases
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_reserves_exact_offline_hold_before_publication
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_real_parent_publication_creates_fresh_held_root
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_offline_hold_cannot_be_rebound_or_replenished
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_offline_hold_rejects_short_inventory
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_live_child_bytes_are_not_double_counted
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_unrelated_output_cannot_spend_offline_hold
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_routes_parent_publications_through_held_process_peak
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_rejects_split_child_categories
tests/pilot/test_pilot_offline_writes.py::test_fit_guard_validates_exact_offline_hold_bindings
tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_rejects_oversized_publisher_before_any_output
tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_counts_existing_and_pending_checkpoint_before_publication
tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_stops_pending_row_append_before_handle_write
tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_rejection_remains_blocked_after_caller_catches_error
tests/pilot/test_pilot_cold_authentication.py::test_fit_guard_refuses_name_or_directory_growth
```

The same launcher then ran these exact three post-pytest commands:

```text
.venv/bin/ruff check --no-cache tests/pilot/cold_authentication_fixture.py tests/pilot/test_pilot_offline_writes.py
.venv/bin/ruff format --check --no-cache tests/pilot/cold_authentication_fixture.py tests/pilot/test_pilot_offline_writes.py
git diff --check -- tests/pilot/cold_authentication_fixture.py tests/pilot/test_pilot_offline_writes.py
```

All four commands exited zero: 26 tests passed in 2.62 seconds, Ruff reported
`All checks passed!`, format check reported `2 files already formatted`, and
the scoped diff check was empty. Wrapper wall was 3.251 seconds. The prefix
retains 23983 logical bytes, 131072 allocated bytes, 56 regular files, 16
links, and 158 directories, with empty stderr. Its admitted maxima were 2.5
MiB allocated, 256 KiB logical, 96 names, 188 directories, one temporary, 32
KiB per pipe, 8192 bytes of launcher code, and 120 seconds per command. It ran
one pytest and no application/control child. Task A primitive writer children
were intentionally excluded because neither their source nor their fixture
path changed.

All retained Task C attempts through implementer FIX1 now occupy 1032192
allocated bytes. No prefix was reused or removed. FIX1 did not run an actual
worker, diagnostic, fit, evaluation, replay, checkout, provider/network, full
gate, or `make verify`.

Current changed-file identities and scoped FIX1 commit:

```text
e97b55925a4bd46e8f2129de5e38112c9d9efcdd74200ce83e83720d61d76760  tests/pilot/cold_authentication_fixture.py
4f38c34f9fa286578ac4907f13d03dbc3362d7b360ba5cd0fcae76cb1c4108f7  tests/pilot/test_pilot_offline_writes.py

0aee7bdc3b99bf8cad7aec51ae5a12075c3ff75d
fix(pilot): count held offline parent directories
```

Exactly those two files were staged in FIX1. Controller-owned progress, brief,
and plan changes and this report remained unstaged.
