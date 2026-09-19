# Privacy task review — 71eabfe

Reviewer: /root/task5_offline_privacy_review (astra/high), read-only.
Base89e987a..71eabfe; source diff87011chars.
Spec: issues found. Quality: needs fixes.

## Important (open)

Temporary publication peaks can be charged to the wrong category.
pilot_offline_process.py:249–258 charges both final and temporary payload/name
allowances to the final filename's category. _publish_pilot_bytes creates a
random sibling .pilot-<nonce>.tmp (pilot_data.py:162). The ledger permits exact
descendant bindings and applies the longest matching prefix (ledger.py:208,251).

Supported example: existing offline directory classified scratch, with
process-result.json separately bound metadata before reservation. At phase6,
R=16384, forecast puts both record copies in metadata and only4096 writer cushion
in scratch. Temporary record belongs to scratch; its A(R) and temporary-name
peak are missing there. A metadata overestimate cannot satisfy scratch's
category-specific reservation. Compute categories independently, or reject
ambiguous public descendant bindings before custody creation; cover insufficient
temporary-category allowance with a pre-publication rejection control.

## Minor (deferred)

Private publication controls cover nonzero exit, timeout and private-write
rejection, but not retained private prefixes after overflow/cancellation
(test_pilot_offline_process.py:1160). Capture preserves in-memory prefixes
(pilot_offline_process.py:719,727); bounded publication controls would complete
the requested matrix. Carry into TaskB's overlapping combined-fence tests and
final whole-branch review; not part of the Important fix loop.

## Positive checks and cross-task scope

Hash-only intent/result types, private prefix persistence before terminal,
source/launch authentication, same-owner reservation checks, current versus
forensic readers, fatal failed outcomes and early caller guards are supported
by the diff and fresh98-case evidence.

Reviewer named-risk source checks support current export exclusion:
producer.py:229,779 requires run-relative members; preflight.py:160 excludes
live workspace and :791,837,857 selects closed stored task snapshots. This is
not an end-to-end export or future operational-export qualification.

Inherited supervisor/FitGuard routing, complete scientific execution/all16
evaluations, and future operational exports remain unverified open integration.
No tests, imports, provider/scientific execution, edits or Git operations were
performed by the reviewer. Root retains the same scope limitations.

## Scoped FIX1 re-review — 6754bd8

Same reviewer: Important ADDRESSED; spec compliant, task quality Approved.
No new Critical, Important or Minor issue in the two-file fix. Shared path
validation at process.py233/351/418 rejects differing final/temp categories
before creation and during revalidation. Tests at909/929/958 cover all four
finals, the one-byte temporary admission, compatible mappings and phases0/6.
Reviewer checked worker RED/GREEN and controller32-case evidence, empty stderr,
allocation/static/source equality; no tests or execution in re-review.

The prior Minor is carried to TaskB/final review. Root resolution of cross-task
items: existing export exclusion is source-verified above; a future generic
operational export requires separate qualification. Inherited custody/FitGuard
and actual all16-evaluation diagnostic are explicitly downstream requirements,
not standalone privacy completion. The spool-route contract remains in the
active ledger for integration before any real launch.
