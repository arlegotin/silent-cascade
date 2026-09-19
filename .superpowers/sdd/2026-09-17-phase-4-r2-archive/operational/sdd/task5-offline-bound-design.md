# Task 5 ruling: closed offline diagnostic output allowance

Read-only architecture review by task5_offline_bound_design (astra/high),
checked against pilot_offline, io, pilot_data, report/pilot and the existing
test-only FitGuard. This records a design, not a numerical/provider admission.

Ruling: bound the one known offline worker's publication paths and pipes,
not a general filesystem quota subsystem. Current output lengths are available
in memory before scientific publication. Reserve a single child suballowance
within the parent's existing reservation; never replenish it or create a
second ledger. Keep genuine training, evaluation, replay, reporting and RNG
semantics unchanged. An undersized attempt must fail before publication and
cannot establish a successful diagnostic.

## Writer closure

Scientific writes funnel through pilot_data._publish_pilot_bytes,
io._durable_temp, and the exact .rows.pending.jsonl stream/final link.
Install guards before their consumers bind imported aliases. Cover existing
targets, incoming temporaries, hardlink coexistence, directories and file names
with the same conservative native-allocation cushion as FitGuard.

The additional identified library writer is Matplotlib's fontlist-v*.json
text stream and its zero-byte .matplotlib-lock. Rendered SVG already uses
BytesIO. Safetensors save and gzip.compress return bytes, not disk paths.
Do not permit arbitrary writable streams or raw descriptor escape via a
font/row proxy. Existing descriptor-relative confinement stays in force.

Closed retained file families for the unchanged 16-episode worker:

| Family | Maximum names |
| --- | ---: |
| Root scientific outputs | 6 |
| Parent intent/stdout/stderr | 3 |
| Evaluation controls | 6 |
| Episode neural/trajectory/telemetry | 48 |
| Shared evaluation weights, also present without a crash | 1 |
| Crash manifest/checkpoint pairs plus crash index | 33 |
| One SVG, tables, Markdown and report index | 4 |
| Matplotlib font cache | 1 |

Total 102, plus separately admitted atomic temporary, row-link coexistence,
font lock and parent failure-record names. Validate exact path grammars and
family cardinalities, not an unbounded wildcard. Scientific directory closure:
root, run, run/eval, run/eval/primary, its episodes/crashes, report and
matplotlib-cache. The existing environment also points XDG_CACHE_HOME at cache;
allowing that directory never permits unknown cache files. Exact final limits
must include it if created, and be checked before writes.

## Process boundary

Activate path/name/cache admission before the boundary imports Torch, then
bind the publisher wrappers before scientific consumers import their aliases.
The optional allowance is closed numeric data bound to this exact output
root and the parent's launch. Preserve legacy no-allowance behavior. Reject
changed nested bootstrap root or enlarged/replaced serialized limits.

Drain stdout and stderr concurrently into separately bounded buffers. Overflow
or timeout terminates and joins the exact child, persists only bounded
diagnostic prefixes plus a compact failure record, and raises. Reserve these
parent diagnostics outside the exhaustible child allowance before launch.
Latch the first denied write even if code catches its exception. Existing
offline.json cannot override a blocked attempt or make it reusable.

## What the numbers prove

The fixed debug model has 771,090 float32 parameters and 18 bool-buffer bytes:
3,084,378 tensor bytes before the safetensors header. Current generic header
limit is 16 MiB, so its format bound is 19,861,602 bytes per model archive.
Telemetry is already limited to 128 bytes per episode. Existing trajectory
format ceilings permit 16 times 128 MiB, so they cannot prove that a complete
diagnostic fits the current 177.6 MiB normal headroom.

Exact next-publication admission proves a bounded failed attempt, not that
the full worker will complete. A successful-run total, full-gate integration
and any needed verified archival remain separate obligations.

No exact project-visible native MPS cache-write path was found in the read-only
review. CPU global RNG setup can initialize native MPS RNG; retain this
platform caveat and existing native-allocation cushion. Do not claim
kernel-wide arbitrary-syscall enforcement or alter RNG semantics to hide it.

## Implementation/test boundary

Use private train/pilot_offline_writes.py for the closed allowance/adapter,
pilot_offline.py for bootstrap, bounded pipes and final failure admission,
and the current-source inventory in pilot_evidence.py for the new module.
Extend cold_authentication_fixture.py only with the omitted publisher aliases
and a parent-held suballowance for the exact offline measurement. No API
credentials, new archive schema, generic resolver, service or quota engine.

Tiny fresh-interpreter controls must test exact/over-budget writes, replacement
and hardlink peaks, row append, genuine font-cache text/lock, unknown names and
directories, path/dir-fd/symlink escapes, swallowed denial, simultaneous pipe
overflow, timeout/join, stale completion, and changed bootstrap authority.
No full neural corpus is needed for these primitive checks. All test prefixes
remain retained under a separately admitted allocation.
