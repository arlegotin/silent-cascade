# Task5 durable-checkpoint reader prerequisite

Read first the subsection `Checkpoint-reader prerequisite before full cold
authentication` in docs/superpowers/plans/2026-09-17-phase-4-r2-archive.md; it is
the exact scoped requirements. Main supplies its committed BASE and admits you
only after the prior status correction passes independent review.

Files: src/silent_cascade/train/pilot_workflow.py::_durable and a new focused
tests/pilot/test_pilot_cold_checkpoint.py. Scope is only the initial checkpoint
read boundary, not complete run authentication or recovery.

_durable currently accepts evidence_context but uses local paths for index and
checkpoint. Reuse archive.readers.evidence_path in two sequential with blocks,
decode index/checkpoint while leased, then close both before unchanged journal,
progress and model checks. Preserve its returned original logical checkpoint
path; callers must reacquire at later consumption, not receive released cache
paths. No new production API/helper/schema/session operation.

Follow TDD using a real initialized phase4_smoke model, real AdamW/progress and
real save/load checkpoint functions, with supplied labeled unit source identity
like existing checkpoint tests. Do not replace decoding/model reconstruction/
journal checks with mocks or fabricate a completed result. A strict path lease
context double is allowed; it must enforce exact requested paths, max1 active
lease and release on success/error. Compare eager with cold using the same new
fixture bytes moved inside its unique test root, plus actual tensor identity.
No old retained artifact may be moved, relabeled or changed. Error cases cover
wrong digest and descriptor step/model mismatch; avoid broad checkpoint suite.

The observed original initial checkpoint is3146426bytes; future/current output
must be enforced <=4MiB before real publication by a test-owned size guard that
delegates to the actual writer. No production limit changes. Bound all fixture
copies and unique adversarial bytes; do not assume model parameter ceilings
equal whole-run bytes. Checkpoint metadata contains unit source identity, not
a claimed authenticated full training run. Zero-step remains running and has
no journal; completed authenticate_run/_train/next-batch gates remain later work.

Budget: same owner51477/PID49087@1789771167.628309/
token464a2a4fabaf30a12dab71308cfa4501. Total additional scratch50331648bytes
relative to original209432576; prior status outputs16KiB remain charged.
This is not blanket test admission: propose exact selector/count, payload and
directory/rounding/admin bound to main BEFORE each new command family. All
roots under operational/scratch/task5-durable-reader/<unique-stage>. Set every
TMP/TEMP/TMPDIR/XDG/MPL/Hypothesis/archive/pytest directory there; disable bytecode
and caches; preserve each earlier output. Main performs full check after writes
stop. No new allowance, baseline, provider, deletion or source copy to fit.

Current branch, meaningful scoped commit, apply_patch, TDD/debugging/verification,
no subagents. Read helpers as needed; don't read entire huge plans or re-audit
completed scanner work. No full suite, four-update fit, source checkout/data
audits, MPS, neural episode execution, external network/Qwen or production pilot.
Model initialization and actual checkpoint encoding/restoration are in scope.

After GREEN, run scoped no-cache Ruff/format and covering checks, report exact
commands/results/byte allocations and limitations to
operational/sdd/task5-durable-reader-report.md. Ask main internally if a new
scope/format is actually needed; routine best-judgment decisions need no user
question. Main handles independent review. Commit only your files/report and
STOP writing before returning concise status/commit/test summary/report path.
