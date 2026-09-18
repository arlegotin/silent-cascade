# Remaining aggregate context integration: bounded preflight

Read-only agent task5_aggregate_preflight at fb3e051/d1e509d. No tests, project
imports, providers, writes or subagents. This is preparation, not execution.

Smallest safe next integration grouping: cold training authentication and
workflow result reuse, before full collection/gate closure. Existing readers
suffice; no recovery-source redesign is needed for that grouping.

- authenticate_run (pilot_evidence.py:336–414): add context to result/index,
  checkpoint and weights consumption. Decode inside existing evidence_path
  leases; return materialized values and a logical run checkpoint locator,
  never a released cache Path.
- _durable (pilot_workflow.py:238–247): despite accepting context, currently
  opens checkpoint-index.json and checkpoint directly; only journal receives it.
- _train (:293–310): forward context for result/recovered-attempt discovery and
  second _durable call; use authenticated inventory for cold existence/discovery.
- workflow report caller (:443): forward available context.

Later aggregate grouping still needs checks authentication/gate reuse, returned
checkpoint reacquisition at numerical execution (:376), continuation row/trace/
weights runtime reads, collection suite/continuation discovery and controls,
numeric/offline groups and inventory, gate attachment/raw-run root separation,
and all semantic helpers. Do not apply one context blindly to two different
roots or hold every input in one broad ExitStack. Acquire checkpoint/weights
separately from episode iteration; reuse existing bounded scoped readers.

Meaningful training coverage can reuse real four-update factory at
tests/pilot/test_pilot_archive_producer.py:347 (4updates,6journalrecords,
32sidecars) without duplicate reference fit. Compare eager/cold progress,
descriptors/tensors and all leases released; completed _train must reuse with
execution tripwire. Corrupt final index/checkpoint must reject. Full gate
coverage later can reuse gate_artifact_case from test_pilot_checks.py:45,
which includes16primary/16repeat plus other suites/continuations/numerics/
offline output. No complete small checked-in training/gate fixture was found.
Existing exact identity/partial fixtures are insufficient for full gate claims.
No byte estimate is justified by episode counts: measure retained exact inputs,
derive finite output/cache/publication bounds, admit before any numerical run.

## Checkpoint prerequisite refinement

Followup static review confirms an actual zero-step checkpoint is legal for
_durable: initial PilotProgress has step/batch0, no journal and running status;
empty optimizer state is supported and zero-update journal coverage is empty.
The real trainer itself publishes this initial checkpoint. It is NOT a truthful
completed training result: validate_training_result rejects running, and the
fixed debug profile reaches its step ceiling at4, not0. Never relabel status
to save test space. Therefore a bounded initial-checkpoint read-path test can
advance _durable only; full authenticate_run/_train reuse still needs real
completed training and later finite admission.

Main read-only size observations from frozen Task4 files (no copies or imports):
green-training-1/test_real_four_update_training0/repo/workspace/run/
training-0-7bf0e773f8056ad5045a069b678440b92da17c2111183009a7fabb1b7bca5cc3.safetensors:
3146426logicalbytes/6152filesystemblocks.
red-training/test_real_four_update_training0/repo/whole/
training-4-fac9a99c6177238f6a91551f67f81793e7166ab6f13b74c80554f0eba2f93d48.safetensors:
9342174logicalbytes/18248blocks; its matching latest portable weights3098274bytes/
6056blocks. Paths are relative to frozen tmp/task-4. These observations are
fixture-sizing evidence, not authorization to relabel/rewrite old checkpoint
source identity or a bound for all future output. A focused new zero-step unit
case can enforce a4MiB pre-publication byte cap using the actual writer behind
a test-owned size guard, without mocks of model/checkpoint/semantic success.
