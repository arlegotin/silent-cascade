# Independent review: cold continuation/offline semantic readers

Reviewer: task5_cold_semantics_review, gpt-5.6-sol/high, read-only.
Range:0a057e7..f08b161. Inputs: scoped brief, implementer report and full-U10
review package. No project imports, tests, providers, writes or worker subagents.

Spec compliance: PASS. Task quality: APPROVED.
Critical:0. Important:0. Minor:0.

Verified scoped optional contexts, early logical-root validation, exhausted
filtered inventories, existing unavailable labels and fail-closed advertised
lease/corruption behavior. Continuation inputs are read sequentially; materialized
weights preserve archive hash, identity, model snapshot and requested-device
checks. Offline direct inputs use evidence_path and the real evaluation scanner
receives its nested root/context. Genuine tests cover eager/cold equivalence,
independent corruption, final-episode closure, lease cleanup and shared weights.
Execution tripwires and prospective corrupt-control bounds were reviewed.

Named outside-diff checks: neural_weights.read_bytes preserves bounded archive
reads during checkpoint predecode; archive.readers.evidence_path preserves eager
resident behavior and scoped cold access. No broader repository crawl.

Cannot-verify items and controller disposition:

- Final global accounting/protected reserve/same-device placement: owner20818
  full check passed after fresh controller tests, scratch246124544/logs1572864,
  other categories unchanged; release check remains separately recorded.
- Test results not rerun by reviewer: controller independently ran all11cases
  in5.10s plus fresh no-cache Ruff/format/diff checks; exact retained command in
  task5-cold-semantics-main-evidence.md.
- Numeric/aggregate/recovery/provider/full makeverify remain later obligations,
  explicitly not claimed by this independently completed reader slice.
