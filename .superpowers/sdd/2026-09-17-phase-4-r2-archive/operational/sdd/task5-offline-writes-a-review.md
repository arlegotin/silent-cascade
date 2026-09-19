# Task A standalone writer independent review

Reviewer task5_offline_writes_a_review, astra/high, isolated read-only.
Range3820408..14abc50; supplied52291char diff and implementer/controller reports.
Spec compliance: issues found. Task quality: needs fixes.
No Critical or Minor findings; one Important finding covers two mutation paths.

## Important: verify ownership before rows link and temporary chmod

src/silent_cascade/train/pilot_offline_writes.py:615 and :677: Closed rows
publication and temporary-file chmod do not verify the recorded file identity
before mutation. The rows branch checks only path, stream membership, and
closed state; _chmod checks the temporary pathname and inventory. _inventory
validates regular-file shape and links but does not compare these files against
their owned stream/operation identities. Consequently, replacing either pathname
with another singly linked regular inode after creation permits linking
substituted rows into rows.jsonl, or changing permissions on an unowned
replacement temporary file. Later detection during cleanup does not satisfy
pre-mutation admission. Compare the live inode with the recorded stream/operation
identity before both mutations, latch authority on mismatch, and add bounded
substitution controls proving no link/chmod occurs. Atomic publication already
performs the relevant identity comparison at :625.

## Evidence and controller ruling

The reviewer confirmed strict limits, sticky denial, conservative accounting,
original publisher bodies and actual writer/stream/font controls. Focused
unchanged-code checks covered durable/pilot publisher lifetime, evaluator rows
close/link sequence, durable-temp callers, boundary descriptor coordination and
the source inventory member. No reruns, mutations, imports or providers.

Controller accepts the Important finding as a direct violation of the brief's
pre-mutation identity requirement. Fix round1/5 resumes the original implementer;
only these two paths and bounded regression controls are in scope. Later B/C,
privacy, scientific/fullgate work remains open. The66-test frozen result is
valid for covered behavior, not evidence these missing cases passed.
Shared owner65407 fullcheck passed with all503808 task bytes retained and
frozen retained history6195077120 unchanged; no production workflow claim.

## Scoped fix1 re-review

Same reviewer approved07e9394: both missing identity checks ADDRESSED, spec
compliant and task quality approved. Rows device/inode is compared before link
at:617; temporary device/inode before chmod at:685. Both mismatches latch
authority. Genuine bounded substitution controls retain displaced originals,
prove no forbidden mutation and exercise sticky denial without production
test seams. No new Critical/Important/Minor breakage or out-of-scope findings.
Reviewer read only fix diff/appended evidence; no executions or mutations.

Controller resolves accounting: full post-fix ownercheck passed with622592
allocated task bytes retained, frozen history6195077120 unchanged, and no
category/global cap change. Baseline66cases apply to14abc50; fresh six covering
cases apply to07e9394, not a complete68case/full-project run. Later privacy,
production launch binding, FitGuard hold, scientific equivalence/current gate,
full local verification and storage handoff remain explicit open integrations.
Task A complete as a standalone slice; Task5/Task7/Phase4 remain incomplete.
