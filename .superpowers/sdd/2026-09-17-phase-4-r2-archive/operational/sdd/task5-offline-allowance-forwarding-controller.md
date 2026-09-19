# D1 controller verification

Source freeze: `39ecdf451090ae5c81d984f6e7a7c0c7166ac8a2`.
Owner PTY49338/tokenc87d6ee040bddf9c9c9b012345d1e0f8 retained throughout.

Exact launcher (append `red1`, `green1`, or
`main1 39ecdf451090ae5c81d984f6e7a7c0c7166ac8a2`):

```text
env -i PATH=/opt/homebrew/bin:/usr/bin:/bin LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=/Volumes/git/legotin/silent-cascade/src .venv/bin/python -B -u .superpowers/sdd/2026-09-17-phase-4-r2-archive/operational/sdd/task5-offline-allowance-forwarding-check.py
```

The retained driver contains the exact nine selectors and five owned files;
all stages enforce150s per command,32768B per stream,65536B cumulative output,
closed runtime roots and no pytest plugins/cache/bytecode. No test children,
neural training, recovery copies, provider calls or genuine diagnostic rerun.
Streams remain in each fresh scratch/task5-offline-allowance-forwarding stage.

| Stage | Test result | Pytest command wall | Allocated | Logical | Files/links/dirs |
| --- | --- | --- | --- | --- | --- |
| red1 | 14 expected missing-keyword failures,1 pass;2.04s |2.466s|8192|6830|2/9/22|
| green1 |15 passed;1.42s|1.841s|16384|187|10/9/22|
| main1 |15 passed;1.41s|1.863s|12288|144|10/9/22|

GREEN scoped formatter changed test layout only. Ruff check, format check and
diff check all exited0. Frozen main repeat additionally ran exact scoped source
equality against the commit above (exit0); every command exited0, stderr empty.
Total retained stage allocation36864B; no failed root removed or reused.

Controller read the full test draft, source diff and all RED failures. Recovery
is expressly a wiring test with costly operations stubbed: this does not qualify
real recovery copies or production authority. Main1 is independent execution,
not a broader suite. Full make verify, production pilot, Task5/Phase4 remain open.
