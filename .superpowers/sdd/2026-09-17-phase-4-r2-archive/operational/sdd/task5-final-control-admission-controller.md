# D2 controller verification

Source freeze: `6ba9e687497ef31030977c30eaba46fa1c44570d`.
Source and writer-report hashes independently match the handoff. The retained
check driver contains the exact four selectors (11 expanded cases), three
scoped files and per-command150s/32768B-stream/65536B-aggregate limits.

MAIN1 used `task5-final-control-admission-check.py main1` with the source freeze
above, bytecode/plugins/cache disabled and fresh counted runtime roots. Pytest
passed11 in1.35s (1.755s commandwall); Ruff check, format check, scoped diff check
and scoped source equality all exited0. No formatter in MAIN1; stderr empty.
Its logs occupy12288allocated/144logical bytes,10files/1link/12directories.
All four retained stages total49152allocated bytes. RED10expectedfailures and
GREEN1 lint failure remain intact beside GREEN2 and MAIN1.

Independent reviewer: `/root/offline_optimizer_cache_review`. The full report
is `task5-final-control-admission-review.md`; spec PASS, quality Approved,
no Critical/Important/Minor findings. Its limitations are real and remain open:
whole-workflow storage, inherited authority, full pilot and `make verify`.

Owner85648/PTY8892 token262fcaa5b6dd9b1488dfd587d3789b89 released/exited0 after
full frozen-history authentication (6195077120 unchanged). Final allocation:
spool231063552/cache67153920/metadata52965376/scratch264204288/logs2854912,
pinned/emergency0. No provider calls, scientific runs, deletion or cap changes.

Read-only feasibility notes were reviewed and corrected: numerical serializer
maxima describe permitted bounds, not measured disk usage; atomic hardlink
coexistence is charged per name. Cold-reader and remaining producer gaps are
recorded in the sibling numerics/final-span notes. D2 is complete; Task5,
Task7 and Phase4 are not.
