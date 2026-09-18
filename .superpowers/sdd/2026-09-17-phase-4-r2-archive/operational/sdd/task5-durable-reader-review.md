# Durable checkpoint reader: independent task review

Reviewed implementation `9f5047c` against the scoped prerequisite in the
approved R2 plan and `task5-durable-reader-brief.md`.
Reviewer: `task5_durable_reader_review` (read-only, independent).

Spec: PASS. Quality: APPROVE. No severity findings.

The reviewer verified sequential index/checkpoint leases, decode inside each
lease, unchanged post-release journal/progress/model checks, logical return
paths, and local defaults. Coverage uses genuine model/AdamW/checkpoint data,
a pre-publication 4 MiB guard, strict single-lease lifetime checks, restored
tensor equality and independent digest/step/model failures. The supplied
source identity is explicitly unit-only, not completed-run authentication.

The reviewer did not execute tests, inspect provider state or validate the
global ledger. The controller independently ran the four scoped cases:
4 passed in 1.88 seconds, plus clean scoped Ruff/format/diff checks; see
`task5-durable-main-evidence.md`. The subsequent same-ledger full check passed:
spool 134307840, cache 67153920, pinned 0, metadata 52965376,
scratch 222056448, logs 1355776, emergency 0 bytes; retained history unchanged
at 6195077120 bytes. No provider or completed fit ran in this slice.

Full Task 5 authentication, caller propagation, recovery, CPU equivalence and
full local gates remain pending. This review does not approve those unrun paths.
