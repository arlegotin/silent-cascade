# Task B initial independent review

Reviewer /root/task5_offline_writes_b_review; source9f0c0b3.

### Spec Compliance

- ❌ Issues found: Task B’s required combined-fence qualification is incomplete; see Important finding below.
- ⚠️ Cannot verify from this diff: Task A’s unchanged allocation/confinement implementation and required-package inventory, or Task C’s held H/category shares. The report identifies those boundaries explicitly (task5-offline-writes-b-report.md:191, :229). They require separate integration evidence.

### Strengths

- Strict allowance validation delegates to the existing allowance contract; production launch rejects forensic intents missing the field (pilot_offline_process.py:87; pilot_offline.py:546).
- Bootstrap reads the module’s live guard, verifies its root and allowance before the worker, and checks the denial latch after return. The worker also checks immediately before publishing offline.json (pilot_offline.py:482, :532, :838).
- New tests exercise actual fresh interpreters for swallowed denial and missing installed guard, including joined-child assertions (test_pilot_offline_process.py:583, :634).
- Measurement retains parent custody and serializes the explicit allowance before the intent digest; scientific changes visible in this diff are limited to authority/latch checks (pilot_offline.py:577, :747, :836).

### Issues

#### Critical

None identified.

#### Important

**Missing required combined-fence control.** tests/pilot/test_pilot_offline_process.py:1472 exercises overflow/cancellation through _measurement_control (:482), which replaces the bootstrap with an unguarded raw-writing child; measurement receives no write_allowance. Conversely, the guarded caught-denial test (:583) ends at direct capture and never exercises the parent’s custody/publication path. Existing overflow (:29) and timeout (:91) controls are also separate, unguarded launches. Consequently, these tests do not demonstrate that parent custody can retain bounded logs and publish the failed result after actual child allowance exhaustion, including competing output/timeout conditions—the explicit requirement in task5-offline-writes-b-brief.md:209. Add bounded controls using real measurement/custody, the real allowance/bootstrap, and a tiny substituted worker; assert joined children, privacy-safe failed results, retained private logs, and rejection of any invalid report. This is a required qualification gap, not an established production-code failure.

#### Minor

None identified.

### Assessment

**Task quality: Needs fixes.** The launch/schema implementation is coherent, but the mandated interaction between child exhaustion and parent completion remains untested.

- Review checks: read supplied diff; inspected cutoff _measurement_control and existing capture/exact-capacity controls specifically for combined coverage. No tests/imports/child probes/network/mutations.
- Evidence available during review:40/40 process tests; independent writer/static still pending in that read revision.

## FIX1 scoped re-review — source cc05f1f

Missing required combined-fence control — ADDRESSED. Real bootstrap plus tiny
worker triggers actual denial (test_pilot_offline_process.py:1551); real
measurement/private custody/allowance cover denial, overflow, timeout and
cancellation (:1612,:1648), and assert failed result, private hash/count/
disposition, invalid report sentinel and joined child (:1664,:1674,:1685).
New breakage: none. Optional limits preserves defaults (:1110).
Out-of-scope observations: none. Reviewer read fix diff/report and retained
COVER2 logs (11passed23.33s, Ruff/format0, stderr0), no execution/mutations.
Verdict: all findings addressed, no new Critical/Important breakage.
