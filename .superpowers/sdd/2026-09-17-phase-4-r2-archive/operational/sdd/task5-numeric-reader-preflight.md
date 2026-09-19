# Read-only preflight: later cold numerical readers

Read-only task5_remaining_gate_slice reviewed pilot_verification.py and
verify_numeric_evidence. This is dependency analysis, not implementation or
execution admission. Continuation/offline is the only current writer.

## Scope

Thread evidence_context through read_numeric_report, _check_artifact_closure,
_runtime_artifact_inventory, _runtime_rows, _read_capture_operations,
_read_capture, _bind_resume_start, _read_resume, and _read_checked_archive;
also through pilot_evidence.verify_numeric_evidence. The last archive reader
is necessary in both complete and partially available evidence branches.
Keep local numeric-checkpoint, weight, JSON and tensor codecs unchanged.

## Dependency and lease design

Private numeric path/JSON/bytes helpers can derive run-relative names from
context.run_dir, validate ancestry and wrap evidence_path; they must preserve
existing per-reader limits and yield no released cache path. Availability is
resident or inventory-advertised, with fully exhausted authenticated discovery.
Advertised corrupt/missing inputs raise rather than become unavailable.

Read report and DONE separately; preserve canonical/hash/config/source checks.
Load the input archive inside a lease and retain session/digest/metadata only;
release before portable weights. Read manifests/subsets/batches individually.
Decode capture tensors, release, then read operation JSON; retain dtype,
inventory, finite-value exceptions, objective and compute comparisons.
_read_resume uses its archive reader's digest rather than a second retained raw
read; observations, expected/restored captures, bootstrap and summary are
sequential. _bind_resume_start forwards context to bootstrap capture reads;
its existing disposable CPU optimizer/clipping arithmetic remains unchanged.

Runtime inventory uses the real header/episode scanner while retaining its
additional diagnostic identity, required-file-set, crash-public-ID and canonical
checkpoint-name checks. Additional crash metadata reads use separate leases,
outside episode leases. Runtime rows project already-verified parsed sidecars.
Closure compares required names against resident plus authenticated cold inputs,
rejects extras/symlinks, exhausts discovery and hashes one file at a time.
verify_numeric_evidence preserves complete/partial paths, every unavailable
label and report.device_checks_passed, including recorded failure/absence.

## Candidate evidence (not yet admitted)

Read-only root:
`.superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-4/cover-pilot-final/test_real_smoke_workflow_reuse0/repo/runs/smoke/final/numerics`

Original allocated tree83681280B, source42b8ab0ba8a647794b08ce29327fce105e3a75f7,
73payload members, four diagnostic updates and missing_devices=["mps"].
Borrow unchanged original backing; create only bounded small corruption copies.
Never relabel source, copy the tensor tree or execute new scientific work.

Candidate tests: complete eager/cold report and verifier equality; explicit
runtime-cpu row equality (single-device report skips two-device comparison);
one missing capture tensor with genuine operations JSON still validated;
independently corrupt operations despite another unavailable capture;
final runtime corruption, substituted DONE, extra inventory and wrong root;
lease lifetime/cleanup under guarded actual I/O. Tripwire training, forward,
event and replay execution, but permit existing artifact-only arithmetic.

This fixture does not establish MPS captures/parity, empty-optimizer bootstrap,
fresh diagnostic production, production64runtime episodes, current-source closed
gate or provider/transport integration. Seek separate genuine evidence or later
admitted runs for those obligations. A versioned plan subsection, concrete
pre-write bounds and controller admission are still required before tests.

## Exact later fixture measurement

Read-only resource preflight found16runtime-cpu episodes, zero error rows and
zero crash manifests/checkpoints. The crashes directory contains shared weights
only. Runtime DONE binds54members (5metadata+48episode+1weights), plus DONE itself.
Ordinal15 public ID:0ee52d49-497a-47ba-a385-c6c69805046f.

| Runtime ordinal15 member | Logical bytes | 4KiB-rounded bytes |
| --- | ---: | ---: |
| episodes/00015.neural.json | 61851 | 65536 |
| episodes/00015.telemetry.json | 39 | 4096 |
| episodes/00015.trajectory.json.gz | 192778 | 196608 |
| crashes/weights-e102be3e3d9c04b9e939ca763d4b6ac1e8a768c62ead350668271332bbf8e094.safetensors | 3098274 | 3100672 |
| Total | 3352942 | 3366912 |

Small overrides: one_hop-cpu.json21337logical/24576rounded; numerical DONE84/
4096; runtime-cpu/DONE5515/8192. Exact4fileepisode+operationsJSON+numericDONE
=3395584rounded. Including all6runtime metadata files adds81907logical/
94208rounded, total3489792. Including numeric-report.json299824/303104 gives
13files/3792896rounded. All other inputs remain borrowed read-only.

Ruling: use a prospective5MiB stage rather than the tight4MiB proposal. Up to
3792896known payload+65536rewrite+131072administrativefiles+64dirs*4096
=4251648B, leaving991232B below5MiB. Three retained stages plus1MiB general
administration fit16MiB. At scratch246124544, a16MiB reservation fits the
268435456 category ceiling. Actual pre-write guards/counts and a fresh same-
ledger admission are still required; this measurement is not permission to run.
