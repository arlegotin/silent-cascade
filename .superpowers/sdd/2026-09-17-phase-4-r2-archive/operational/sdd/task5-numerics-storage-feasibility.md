# Task5 numerics storage feasibility

Status: read-only source analysis at base `0e92793`. D2 concurrently changes only
producer/checks control calls; this note does not treat those edits as storage
closure. It adds no authority, cap, schema, hold, diagnostic, or source change.

## Conclusion

Current `verify_pilot_numerics` cannot be admitted against the 2 GiB spool cap
from its declared serializer maxima. With CPU and MPS, its non-runtime families
alone have a 2,240 MiB coarse maximum for a fresh model, before either
64-episode runtime evaluation. A completed runtime evaluation has a
16.875 GiB + 8 KiB maximum with no episode errors and a 33 GiB + 8 KiB maximum
when every episode retains crash evidence. `_evaluate_subset` passes no
`ArchiveProducer`, so all of those files remain resident.

The shapes strongly suggest much smaller ordinary files, but current source has
no closed, enforced conversion from those shapes to a canonical serialized-byte
maximum. Historical size therefore cannot establish admission. The smallest
defensible storage-only route is explicit archive-as-you-go through the existing
producer/session: seal and evict fixed capture/resume groups after use, stream
each runtime evaluation through per-episode archive ownership, read it through
the explicit evidence context, then evict its metadata. All retained bytes stay
authenticated in cold archive. This is necessary but not yet end-to-end
sufficient because downstream checks currently drop the explicit context.

## Source constants and notation

- Spool policy ceiling: `ArchivePolicy.spool_bytes <= 2 * 1024**3` in
  `archive/types.py`.
- Pilot JSON, copied checkpoint, capture and training-checkpoint publication:
  64 MiB in `train/pilot_data.py`.
- Neural weights, evaluation files, neural crash manifest/checkpoint, decoded
  trajectory and compressed trajectory: 128 MiB in
  `eventflow/neural_weights.py`, `eval/artifacts.py`,
  `logging/crash_bundle.py`, and `logging/neural_trace.py`.
- Production runtime subset: exactly 64 episodes
  (`verify_pilot_numerics` and `PilotNumericReport.validate_complete_inventory`).
- Telemetry: at most 128 bytes per episode (`eval/runner.py:_run_one`).

Let `d=1` for CPU-only or `d=2` for CPU+MPS, `q=1` when an input checkpoint is
present, and `b=1` when the starting optimizer is empty. Fresh execution has
`q=0,b=1`; a checkpoint may have `b=0` or `b=1` according to
`PilotCheckpoint.optimizer_names`.

## Exact output families and cardinalities

All paths below are under the fresh numerics root. Counts are source-fixed; byte
figures are publication ceilings, not predicted sizes.

| Family | File count | Per-file ceiling |
| --- | ---: | ---: |
| `intent.json` | 1 | 64 MiB |
| `input-training.safetensors` | `q` | 64 MiB |
| `unchanged-weights.safetensors` | 1 | 128 MiB |
| `manifest.json`, `runtime-subset.json` | 2 | 64 MiB |
| `one_hop-batch.json`, `primary-batch.json` | 2 | 64 MiB |
| `{stage}-{device}.{json,safetensors}` | `4d` | 64 MiB |
| base resume family per device | `7d` | 64 MiB |
| optional `resume-{device}/bootstrap.{json,safetensors}` | `2bd` | 64 MiB |
| runtime evaluation per device, no episode errors | `199d` | mixed; below |
| extra runtime crash members for `e` errored episodes/device | `2e + 1[e>0]` | 128 MiB each |
| successful `numeric-report.json`, `DONE` | 2 | 64 MiB |
| escaping-exception `failure.json` | 1 instead of the final pair | 64 MiB |

The seven base resume files are `resume.safetensors`, `observations.json`,
`resume.json`, and the `expected` and `restored` JSON/safetensors pairs. An
absent checkpoint always bootstraps because the newly created optimizer has no
state. A present full checkpoint omits bootstrap only when it has optimizer
names.

For each device, the 199-file no-error runtime inventory is:

- `identity.json`, `retention.json`;
- one shared `crashes/weights-<sha>.safetensors` (staged even if no episode
  fails);
- 64 neural sidecars, 64 full trajectories, and 64 telemetry files;
- the single-inode row log, `index.json`, `metrics.json`, and `DONE`.

Every runtime episode is retained in full: purpose is `action_diagnostic`, and
`retained_public_ids()` returns all 64 entries because the inventory is below
500. Each errored episode adds one crash manifest and one runtime checkpoint;
the first error also adds `crashes/index.json`. Thus all 64 errored episodes
produce 328 files/device. A represented `DynamicsError` is an episode outcome,
not an escaping verifier failure; evaluation may still reach `DONE` with failed
metrics.

The total successful file counts are:

| Hardware/input | No episode errors | All runtime episodes errored |
| --- | ---: | ---: |
| CPU, fresh | 220 | 349 |
| CPU, checkpoint with optimizer | 219 | 348 |
| CPU, checkpoint without optimizer | 221 | 350 |
| CPU+MPS, fresh | 432 | 690 |
| CPU+MPS, checkpoint with optimizer | 429 | 687 |
| CPU+MPS, checkpoint without optimizer | 433 | 691 |

An exception after all otherwise retainable families replaces the two final
success files with one `failure.json`, so the corresponding maximum completed
file count is one smaller. Earlier exceptions retain only a prefix plus the
failure file and any one in-flight temporary name.

The numerics root creates at most `1 + 4d` persistent directories: the root,
one resume root/device, and runtime, episodes, and crashes roots/device. Existing
run/final ancestors are outside this count.

## Coarse byte accounting

Ignoring runtime evaluations but including a successful report/DONE, the exact
sum of current file-format ceilings is

```text
576 MiB + 704 MiB*d + 128 MiB*b*d + 64 MiB*q
```

| Hardware/input | Coarse non-runtime maximum | Relation to 2 GiB |
| --- | ---: | ---: |
| CPU, fresh | 1,408 MiB | 640 MiB below, before runtime |
| CPU, checkpoint with optimizer | 1,344 MiB | 704 MiB below, before runtime |
| CPU, checkpoint without optimizer | 1,472 MiB | 576 MiB below, before runtime |
| CPU+MPS, fresh | 2,240 MiB | 192 MiB over |
| CPU+MPS, checkpoint with optimizer | 2,048 MiB | exact ceiling; allocation overhead makes it fail |
| CPU+MPS, checkpoint without optimizer | 2,304 MiB | 256 MiB over |

Before the final report/DONE, fresh CPU+MPS output is already 2,112 MiB by
format maxima. This disproves admission based only on current per-file limits.

One completed runtime tree with no episode errors has 135 files that may each
reach 128 MiB (two controls, shared weights, 128 episode payloads, row log,
index, metrics, DONE) plus 64 telemetry files:

```text
135 * 128 MiB + 64 * 128 bytes = 16.875 GiB + 8 KiB
```

If all episodes error, 128 crash files and one crash index add 16.125 GiB:

```text
264 * 128 MiB + 64 * 128 bytes = 33 GiB + 8 KiB per device
```

CPU and MPS run serially but their completed trees coexist, so serial execution
does not reduce retained occupancy. The row pending/final names are hard links
to one inode; the completed tables count only the quiescent final name after the
pending name is removed. Their simultaneous ledger peak is larger, as below.

## Atomic coexistence and failure retention

Writers are sequential. `_publish_pilot_bytes` and `atomic_create_bytes` use a
temporary name and final hard link to the same physical inode. Physical blocks
are shared, but `_StorageBudget.measure` has no inode-dedup set: it walks names
and adds `st_blocks * 512` for each directory entry. During temporary/final
coexistence the reservation is therefore charged the full payload twice. The
quiescent completed-payload tables above must be augmented by one current
publication payload, up to 128 MiB, plus block rounding and directory
allocation. Pilot-only publications add at most 64 MiB; evaluation and archive
publications can add 128 MiB.

`.rows.pending.jsonl` likewise grows in place, is hard-linked to `rows.jsonl`,
then has the pending name removed. Its final-link interval adds another charge
equal to the complete row log, up to 128 MiB, even though the quiescent table
contains one row payload. Safetensors encoding and trajectory gzip remain in
memory and add no disk scratch.

An atomic durability or cleanup failure may retain a temporary entry. That is
not mere name overhead: every later measurement charges its full `st_blocks`.
If the final name also exists, both entries remain charged; if only the
temporary exists, the partial-failure inventory still gains one bounded payload.
Publishing the subsequent `failure.json` can then add its own temporary/final
overlap of up to 64 MiB. Thus failure admission must cover the retained partial
payload(s) plus the next bounded failure-control publication, rather than only
the completed-family sum plus a directory entry.

In particular, the checkpoint-with-optimizer CPU+MPS case cannot use its
apparent exact 2 GiB fit, and every other row in the table needs the applicable
up-to-128-MiB operational overlap in addition to names/directories. The tables
remain valid only as quiescent completed-payload sums. `verify_pilot_numerics`
keeps the fresh root on every exception and performs no rollback.

## Why shapes do not yet prove a smaller bound

There is useful structural evidence:

- production uses exactly 64 manifest-bound episodes/device;
- event-flow permits 64 internal events, memory has 64 slots, and checkpoint /
  replay schemas expect at most 131 trajectory anchors / 130 events;
- each trajectory anchor carries 1,836 fixed-shape float values
  (`459 current + 459 origin + 456 targets + 456 rates + 6 guard values`),
  plus bounded causal metadata;
- each neural observation is tied to one committed event, recall inventory is
  at most 64 slots, and model outputs have fixed production dimensions;
- model parameters are capped at five million and archives use fixed dtypes.

Those facts could support materially smaller source-derived maxima, but no
writer currently calculates and enforces their canonical JSON character count,
gzip worst case, safetensors header, per-event metadata, actions, or crash
checkpoint total. `PublicEpisode` and the trace writer themselves rely on
generated inputs and the 128 MiB byte guard rather than one smaller serialized
ceiling. The only presently defensible byte bound is therefore the coarse
format bound above. A future tighter admission proof must state the complete
formula and test exact/one-byte-over serialization; multiplying observed sizes
or tensor payloads alone is insufficient.

## Smallest storage-only route

Do not lower episode count, omit CPU/MPS, drop full trajectories, discard
mismatches/failures, or enlarge spool. Keep the current scientific computation
and move authenticated completed files from spool to the existing cold archive:

1. Give `verify_pilot_numerics` the existing `ArchiveProducer` and explicit
   evidence context. This transports no new authority.
2. Add fixed, numerics-specific producer groups only: setup files; each
   stage/device capture pair; each resume-device family; each runtime evaluation;
   and final report controls. Seal/evict a group only after its in-memory result
   and digest inventory are complete. No caller-selected generic paths.
3. Before each runtime evaluation, reuse the conservative existing
   `validation_autonomous_weights` admission maximum (1,946,419,328 bytes,
   below 2 GiB), then pass the producer into `evaluate_episodes`. Its existing
   episode commits archive/evict sidecar, trajectory and crash ownership after
   every row.
4. Preserve the returned `PilotEvaluation.artifact_hashes`; call
   `_runtime_rows` with the explicit producer session so evicted episode files
   are leased from archive. After rows are materialized in memory, use one
   purpose/root-bound numeric method to archive and evict that evaluation's
   metadata before starting the next device.
5. Build `PilotNumericReport.artifact_hashes` from the authenticated group and
   evaluation inventories, not `_artifact_hashes(output_dir).rglob()`. Publish
   and seal report/DONE last. On an exception, seal the bounded partial group
   and failure control; never delete prior evidence.

Runtime streaming alone is not a complete fix: CPU+MPS non-runtime format
maxima already exceed spool. Fixed setup/capture/resume groups must also be
evicted progressively unless a separate source-derived serialization proof
first establishes a lower bound. This route preserves retention and mismatch
evidence byte-for-byte; only residency changes.

## Definite end-to-end blockers outside this storage slice

Even after numerics is cold-aware, the current checks chain cannot consume all
cold evidence: `_run_pilot_checks_owned` calls `authenticate_run` without its
available `evidence_context`; `collect_pilot_evidence` has no context parameter
and calls compact/read helpers without one; `_continuations` directly loads
evaluation and sidecar paths. Readers support explicit leases and have no
implicit fallback. Full-pilot admission must remain denied until those caller
gaps and the exact fixed-group maxima are closed and the composite reservation
is re-proved under unchanged caps.
