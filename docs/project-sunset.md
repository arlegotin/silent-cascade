# Silent Cascade project sunset

**Decision date:** 2026-09-27

**Status:** Scientific development is stopped after the Phase 5A exploratory comparison. The remaining Phase 5 baseline program and Phases 6–7 will not be pursued under the current project.

## Why we stopped

Silent Cascade set out to test whether EventFlow's learned events during observation-free time improve timed-action accuracy, depth transfer, or compute efficiency over strong alternatives. The [Phase 4 autonomous pilot](phase4-autonomous-eventflow.md) succeeded as an engineering milestone: its one-seed EventFlow checkpoint passed the unchanged Phase 4 gate. That milestone did not compare EventFlow with a trained baseline.

The [Phase 5A comparison](phase5a-comparative-pilot.md) provided that first direct diagnostic. On fresh 512-episode corpora, intact EventFlow succeeded on **512/512 IID** and **504/512 depth-transfer** episodes. A separately trained, similarly sized activation-time ponderer succeeded on **512/512 in both corpora** at caps 12, 16, and 24. At cap 12, it used about **13.3 times fewer measured forward MACs on IID** and **13.8 times fewer on depth transfer**. The ponderer's eight-episode depth edge is small; the large compute gap is the clearer adverse result for EventFlow's proposed practical advantage.

Compressing the same EventFlow weights to activation-time execution lowered their scores to **319/512 IID** and **289/512 depth transfer**. This shows that the trained checkpoint depends on its event-timing regime. It does not show that this regime is better than a separately trained alternative; the ponderer matched or slightly exceeded EventFlow while doing much less measured neural work.

Given that evidence, we are sunsetting the project rather than spending the much larger budget required for the full Phase 5 program and later frozen tests. This is a resource and research-priority decision: the current benchmark has not shown the advantage the project was designed to seek.

## What this decision does and does not establish

- Phase 4 remains an accepted autonomous-pilot engineering result. The sunset does not revise its gate, weights, or recorded outcome.
- Phase 5A is **single-seed exploratory evidence** on DEBUG corpora. It is not a five-seed comparison, a compute-matched estimate, or a frozen final-test result. None of the measured ponderer caps fell within the predeclared ±10% IID compute-matching window.
- The results are a strong negative signal for an EventFlow advantage **on this task under the tested conditions**. They do not prove that timed event models are useless in every setting. Possible value in unpredictable streaming inputs, online control, or inspectable temporal state remains untested here.
- The [canonical specification](superpowers/specs/2026-08-30-silent-cascade-design.md) and its scientific gates remain historical requirements, not claims of completion. No gate or claim boundary has been relaxed to turn this pilot into a final positive or negative result.

## Where work stopped

The Phase 5A scientific runs and [independent artifact audit](../reports/phase5a-v1/verification/receipt.json) were completed; their compact reports and receipt remain. The audit authenticated corrected comparison rows, recounted primary validation, and replayed selected episodes before the raw inputs were removed. Full Phase 5 controls, additional seeds, causal interventions, and frozen final tests were not run.

The repository-wide local `make verify` gate for the final Phase 5A source **was not completed**. An earlier run exposed a stale historical delivery-test fixture; that fixture was corrected and its focused tests passed. The subsequent full rerun was interrupted at the user's request while broad tests were still in progress. No Phase 5A-wide `make verify` pass is claimed.

The code, plans, compact reports, and adverse results remain as a historical research record. The [artifact cleanup](artifact-cleanup.md) removed raw runs and validation evidence, so exact replay from this checkout is no longer available. Any future revival would need a new, specific use case and a fresh approved plan; it should not assume that EventFlow has earned a performance or efficiency claim.
