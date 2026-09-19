# Task review: cold gate inputs, round 0

Independent reviewer: task5_cold_gate_inputs_review, sol/high, read-only.
Scope: `430520b..8934f22`; source diff matches `b417c4b`.
No Critical or Minor findings. One Important finding remains open.

## Important: partial availability hides invalid checkpoint index

`pilot_evidence.py::_verify_available_training` only tests whether
`checkpoint-index.json` is available. Its lease, parsing and latest-descriptor
comparison happen inside `_durable`, skipped when a journal member or artifact
is missing. Thus an available malformed/mismatching index or advertised index
lease failure can be downgraded to durable-journal unavailability.

Independently validate the available index before complete-closure testing.
Retain the complete `_durable` call. Also apply the existing per-record journal
schema checks to available records when durable closure cannot run.

Add missing-journal-dependency plus corrupt/mismatching-index and failing-index-
lease regressions; add an available rehashed invalid-journal-schema control.
The previous matrix covered checkpoint, journal hash and artifact corruption,
but not these partial-index/schema cases.

## Controller ruling

Confirmed against `pilot_workflow.py::_durable` and the partial reader.
Fix round 1/5 returns to the original implementer under owner13703. A narrow
extraction of the existing journal-record schema check in `archive/readers.py`
is permitted to avoid duplicated validation; no new schema or runtime changes.
No new execution begins without a separately bounded tiny RED stage.

The reviewer otherwise confirms genuine codecs, root isolation, compact lease
lifetime, RNG restoration and correct nonpassing-outcome reuse. The 39 passing
tests are valid but insufficient for this uncovered integrity case.
