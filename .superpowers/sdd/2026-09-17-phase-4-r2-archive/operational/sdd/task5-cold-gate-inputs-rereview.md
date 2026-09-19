# Scoped re-review: cold gate inputs, fix round 1

Original reviewer task5_cold_gate_inputs_review, sol/high, read-only.
Fix package exactly matches `8934f22..dc9d546`.

- Original Important finding: addressed. Available index is independently
  leased, parsed as a real descriptor and compared before partial closure
  decisions. Advertised index lease failures propagate.
- The partial journal path now shares the unchanged real record validator
  with iter_journal_records.
- The controller-identified null latest bypass is rejected unconditionally.
- Five focused regressions cover malformed/mismatched index, index lease
  failure, rehashed invalid journal schema and null latest.
- No new Critical or Important findings. Fix round ready.

Controller fresh frozen-source verification: 47 passed in 2.95 seconds;
four-file Ruff lint/format and diff checks passed.
Full make verify, outer gate context forwarding, final reread/hash binding,
collection, recovery and current-source full-gate execution remain outstanding.
The later process-capture plan subsection is not part of this review.
