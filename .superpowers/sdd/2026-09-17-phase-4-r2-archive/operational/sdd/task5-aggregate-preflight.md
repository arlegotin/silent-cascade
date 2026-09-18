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
