# Task5 remaining final spans — controller source audit

Base: `0e92793029bd7d4392df1f7e90a64984c7c6e5c3`; D2 changes only two
control-publication admission points. Read-only source analysis under owner8892.
No tests, imports, model execution, provider calls or new execution authority.

## Result

A complete bounded/cold final workflow is not established. This is not a
measurement that the real run needs a huge disk: serializer ceilings are
conservative permitted maxima, not predicted output sizes. Do not present their
products as expected physical usage. The genuine standalone diagnostic already
used about12MiB; that observation alone cannot certify other producers.

## Explicit context is lost by remaining high-level callers

`archive/readers.py:evidence_path` uses a lease only when an explicit context
exists and the original path is absent; there is no implicit global fallback.
Existing `authenticate_run`, `compact_evaluation`, `compact_attachment_records`,
`read_numeric_report` and `load_evaluation` have context-aware reader paths.
However, at this source:

- `pilot_checks._run_pilot_checks_owned` accepts context but omits it when calling
  `authenticate_run`, `_continuations`, and `collect_pilot_evidence`.
- Public checks likewise authenticate without context. D1 intentionally only
  forwarded offline allowance, not these unrelated data readers.
- `_continuations` has no context parameter; it calls `load_evaluation` and
  reads sidecars directly under `final/eval/primary`.
- `collect_pilot_evidence` has no context parameter and calls `authenticate_run`,
  `compact_evaluation`, attachment readers and numerical readers without it;
  it also directly reads DONE/execution/training-index files and tests local
  existence. The cold-aware leaf implementations do not fix these callers.

Therefore archived episode eviction can make a required final read fail, or
local-existence selection can miss remote evidence. A future scoped propagation
task must retain exact logical paths, authenticated hashes and semantic checks;
never serialize a cache lease path as scientific provenance. Do not hydrate
whole corpora to work around the missing context. Diagnose with tiny cold-path
tripwires before changing any caller.

## Continuation cardinality

`pilot_checks._continuations` uses the15 `REQUIRED_COVERAGE` labels to decide
which candidates add evidence. It updates `covered` only when `matched` is true.
A mismatched comparison is appended and processing continues; thus15 is not an
unconditional record/checkpoint bound. For a supplied manifest, a conservative
finite expression is:

```text
checkpoint records <= sum(1 + len(uninterrupted.trace.events) for every eligible episode)
replay files       <= number of eligible episodes
controls           = runtime/intent.json + final/continuation.json
```

The extra1 is the possible mid-flow candidate. Each emitted record gets its own
runtime checkpoint; each participating episode gets at most one replay file.
`neural_weights.py` limits an archive to128MiB and metadata to16MiB. Multiplying
the archive ceiling by the candidate count is not a viable2GiB whole-span proof.
Even the successful-coverage15-record argument requires explicit matched-path
reasoning, shared-weight/runtime tensor shape bounds and simultaneous atomic
allocation. No repeated-failure outputs may be silently discarded.

The canonical contract says replay mismatches are errors, not normal outcomes.
Whether this loop should terminate on its first inconsistent comparison is a
separate debugging/RED decision, not a storage permission to shrink evidence or
change scientific outcomes. No behavior correction is made by this audit.

## Gate attachments and report

Production final suites contain four10000-row corpora and two256-row delay
corpora (40512rows). `write_compact_shards` splits at1000rows or128MiB raw bytes,
whichever comes first. Ten shards per10000-row suite is not an upper bound when
the byte threshold wins. In the worst permitted row partition there is one
shard per row. Its compressed ceiling is100MiB-1, but actual publication also
passes the64MiB pilot publisher limit. Every shard is retained locally by the
current collector, followed by copied training-index shards and the gate.
These format bounds alone cannot certify total fit. Compact rows include
causal events and compute observations; a tighter shape-derived limit or
streamed authenticated attachment archival must preserve all40512rows/checks.

`build_pilot_report` already consumes explicit episode leases. It produces one
SVG for each completed authenticated evaluation root, plus tables.json,
report.md and report-index.json. Per-corpus plotting has at most two retained
examples,20histogram bins and three guard curves; plotting does not contain
all10000episode traces. The root inventory includes committed/stopped histories,
so use its authenticated cardinality, not an assumed fixed validation count.
Every publication uses the64MiB pilot format limit with temporary/final
coexistence. Matplotlib font-cache writes must remain in the counted runtime
root. Source-sized SVG/JSON bounds or phased artifact archival are needed;
neither a sample file size nor64MiB times an guessed root count is sufficient.

## Delivery implications

D2's request/execution prewrite correction is independently valid. Do not add
unused forever-held admission fields or issue a delegated descriptor before
the remaining spans and cold-reader call chain are executable. Pair the
numerics-specific audit with this map, choose the smallest storage-only fixes,
then write exact phase-scoped tests/admissions. Preserve all scientific gates,
all raw failures, the10GiB total contract and local-only verification. An
overlarge conservative bound is a design problem to resolve, not a reason to
reduce test/model coverage or claim that observed output consumed that bound.
