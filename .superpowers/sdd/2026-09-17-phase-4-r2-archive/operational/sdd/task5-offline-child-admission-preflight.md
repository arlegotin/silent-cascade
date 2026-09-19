# Read-only design preflight: one bounded offline diagnostic child

This is a recommendation from task5_remaining_gate_slice, not execution
admission or a new general quota subsystem. The gate-input writer remains the
only active implementation task.

## Alternatives and recommendation

- Outer monkeypatches do not propagate into the fresh _PROGRAM interpreter
  (pilot_offline.py:422-440).
- Per-process file-size limits alone do not bound total allocation or count;
  parent polling only detects growth after writes.
- A child-installed allowance covering this exact diagnostic's known writers,
  backed by the existing parent reservation, is the smallest suitable approach.
  Keep the ledger in the parent; no second ledger, arbitrary-root authority,
  mount, service or quota framework.

Candidate interfaces:

```python
measure_pilot_offline(*, output_dir, write_allowance=None)
install_offline_boundary(*, workspace_root=None, write_limits=None)
```

The allowance denotes an already-held parent reservation for the exact output
root. Transfer only closed byte/file/directory and log/failure ceilings, no
credentials. Ordinary eager behavior remains unchanged when absent; a bounded
integration test must require it. These signatures are a design candidate,
not implemented APIs or permission to select an unsupported total.

## Required enforcement points

1. Activate admission before the boundary's Torch import (pilot_offline.py:414).
   Update its exact nested-child bootstrap allowlist (:109,371-399). Reject a
   changed root, malformed limits or enlarged nested allowance.
2. Guard the underlying pilot_data._publish_pilot_bytes and io._durable_temp,
   keeping genuine writers. Before temporary publication reserve new payload,
   existing target, temporary/final hardlink coexistence, names and directories.
3. Cover pending-row encoded appends (eval/artifacts.py:365-385) before writes.
   Extend descriptor-aware boundary operations (pilot_offline.py:282-369) for
   new names/directories, link/rename/truncate. Latch admission failure so a
   caught error cannot permit further scientific output.
4. TMPDIR/MPLCONFIGDIR/XDG_CACHE_HOME already lie inside output_root (:141-146),
   but caches need byte limits, not only confinement. Cover actual binary/text
   stream growth, append/seek/truncate and descriptor writes. Reject unsupported
   writable opens; do not disable reporting or model execution to avoid output.
5. Replace unbounded parent capture_output (:439-447) with concurrent bounded
   stdout/stderr draining. Admit retained bytes before writing. On overflow,
   terminate/join the exact child, preserve bounded prefixes and fail; never
   silently truncate into success. Include later log-publication peaks.
6. Hold failure-diagnostic reserve outside the child's exhaustible allowance.
   Persist the blocked attempt without discarding prior output. Existing
   offline.json cannot promote a blocked/terminated attempt into reusable success.

## Exact later tests

Extend tests/pilot/test_pilot_offline.py with fresh interpreters: oversized real
pilot/atomic publication; existing target plus replacement/link peak; row/cache
binary/text append and descriptor/truncate; name/directory limits and symlink/
dir-fd escapes; caught-error continuation; simultaneous stdout/stderr overflow
and timeout with bounded retained diagnostics/no live child; exact nested
bootstrap with changed-root/limit rejection. Ultimately run one genuine worker
with its update,16episodes,replay,report and unchanged semantic verification
(pilot_offline.py:478-549), under a separately justified total allowance.

## Limits and unresolved admission

Python writer/audit hooks do not prove a kernel-wide limit for arbitrary native
syscalls or writable mappings. They are usable only after this exact worker's
writer/cache paths are identified, covered and tested. An unidentified native
writer remains a blocker; OS/filesystem quotas would be a materially larger
change and are not authorized by this preflight. No full-gate byte total follows
from these recommendations. The next implementation plan must choose justified
ceilings and pre-write accounting for the actual bounded workload, including
source/history/data, retained fit, diagnostics, child, reports and lease copies.
