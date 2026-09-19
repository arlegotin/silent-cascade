# Task5 read-only next-caller preflight

Read-only task5_remaining_gate_slice reviewed the actual remaining callers.
This records dependency analysis, not source-edit or numerical-run admission.
The numerical reader slice is still the sole active source writer.

## Next independently testable grouping

Use the existing `_verify_available_training` and `compact_attachment_records`
in train/pilot_evidence.py, plus a small production reuse-status check used by
train/pilot_checks.py::_run_pilot_checks_owned. Then integrate full verification,
collection and execution callers as one source-bound closure. Do not call the
old historical gate a positive current-source aggregate proof.

`_verify_available_training` must validate context/run-root agreement before
reads; discover resident plus authenticated checkpoint/weights/index/journal
inputs; decode real CPU archives individually and preserve RNG restoration,
eligibility, model/source and progress checks. Walk available journal links
individually; another missing dependency may not hide cycle/hash corruption.
Call the existing context-aware `_durable` only when its actual dependency
closure is available. Missing unindexed input retains existing unavailable
labels, while an advertised unreadable/corrupt input raises.

`compact_attachment_records` keeps a shard lease through hash validation,
complete row consumption and final count validation. Exhaustion, error or
generator close must release it. No eager list of all shards/rows.

## Root and reuse contracts

The evidence context belongs to raw_run_dir, not arbitrary exports. Validate
their exact agreement. Gate JSON, compact shards and exported index shards are
rooted at artifact_path.parent. If that parent equals the context run root,
use the context. Otherwise retain local export reads and original digest
checks. Never satisfy an absent exported shard from an identically named
raw-run member. Cold exports under another prefix need separate routing and
are not silently supported by this bounded task.

Existing gate reuse rejects either missing_raw_attachments or
unavailable_semantic_checks. It does not require passed=True: fully verified
debug_non_acceptance and failed artifacts are valid recorded outcomes. Read
the gate again under its own lease and compare its digest with the verifier's
artifact_sha256 before returning it.

## Subsequent integrated closure

Verifier: lease initial/final gate bytes, compact shards, indexes, raw hashes,
training-result comparisons and every semantic helper; exhaust indexes and
episode tails; retain missing/unavailable status projection and independent
corruption checks.

Collection: context-aware authentication, cold suite/control and partial-prefix
discovery, leased continuation/execution/DONE/numeric/offline/result/index reads,
complete offline subtree hashing, exact export-local attachment/index writes,
and final source-authentication recheck.

Execution: `_evaluate` detects cold DONE and partial prefixes and reuses through
the real scanner; `_continuations` releases an episode before a portable-weight
lease/replay operation and fully consumes its scan even when coverage is met.
Forward context through `_run_pilot_checks_owned` and run ownership/recovery;
reacquire a logical checkpoint around its complete numerical consumer. The
external owner file remains local, while workflow.json is leased.

## Evidence and limits

Safe artifact-only cases: genuine compact shard eager/cold equality, final
count/hash corruption and generator-close cleanup; export isolation; real
available training archive/weights/journal equivalence, missing labels and
independent corruption; reuse predicate combinations preserving nonpassing
recorded outcomes. Historical source-closure rejection is a negative test only.

The original42b8 source closure does not contain today's archive/readers.py or
archive/preflight.py. Its 215456KiB smoke tree is a historical measurement, not
a prospective storage bound. A genuine current-source gate needs separately
admitted production of debug evidence with exact code/history bindings, likely
after additional verified archival. The active16MiB reader allowance does not
authorize copying that tree, new training/diagnostics, a full gate or providers.
