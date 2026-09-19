"""Artifact-only gate inputs through bounded cold evidence leases."""

import gzip
import inspect
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from .cold_semantics_fixture import HistoricalContext, historical_root
from .test_pilot_cold_semantics import install_no_execution_tripwires


class HistoricalTrainingContext(HistoricalContext):
    """Add genuine journal-unit metadata to the reviewed strict file leases."""

    def __init__(self, *args, journal_ref, journal_metadata, journal_paths, **kwargs):
        super().__init__(*args, **kwargs)
        self.journal_ref = journal_ref
        self.journal_metadata = journal_metadata
        self.journal_paths = frozenset(journal_paths)
        self.lease_failure = None

    @contextmanager
    def _leased(self, payload):
        if payload["path"] == self.lease_failure:
            raise ValueError("advertised training lease failure")
        with super()._leased(payload) as lease:
            if payload["path"] in self.journal_paths:
                yield SimpleNamespace(
                    local_root=lease.local_root,
                    ref=self.journal_ref,
                    metadata_root=self.journal_metadata,
                )
            else:
                yield lease


@pytest.fixture(scope="module")
def historical_training_case(tmp_path_factory):
    backing = historical_root()
    if backing is None:
        pytest.skip("requires private retained Task 4 smoke evidence")

    from silent_cascade.archive.catalog import seal_unit
    from silent_cascade.archive.producer import JournalSegmentCommit
    from silent_cascade.archive.types import ArchivePolicy
    from silent_cascade.config import ResolvedConfig
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
    from silent_cascade.train.pilot_config import parse_phase4_canonical
    from silent_cascade.train.pilot_evidence_types import read_compact_rows
    from silent_cascade.train.pilot_provenance import PilotSourceIdentity

    gate = json.loads((backing / "phase4-gate.json").read_bytes())
    canonical = gate["config_canonical_json"].encode()
    config = ResolvedConfig(
        config=parse_phase4_canonical(canonical.decode()),
        canonical_json=canonical,
        sha256=sha256_bytes(canonical),
        source_paths=(),
    )
    source = PilotSourceIdentity.model_validate(gate["source"])
    training = gate["training_result"]
    index = training["artifact_index"]
    inventory = {}
    for shard in index["shards"]:
        assert sha256_bytes((backing / shard["path"]).read_bytes()) == shard["sha256"]
        rows = tuple(read_compact_rows(backing / shard["path"]))
        assert len(rows) == shard["rows"]
        inventory.update((row["path"], row["sha256"]) for row in rows)

    cursor = training["progress"]["journal_sha256"]
    journal_paths = []
    journal_dependencies = []
    record_sha256s = []
    while cursor is not None:
        name = f"journal-{cursor}.json"
        record = json.loads((backing / name).read_bytes())
        journal_paths.append(name)
        journal_dependencies.extend(record.get("artifacts", {}).items())
        record_sha256s.append(cursor)
        cursor = record["prior"]
    commitment = JournalSegmentCommit(
        base_head=cursor,
        sealed_head=record_sha256s[0],
        record_sha256s=tuple(record_sha256s),
    )
    assert len(journal_paths) == len(record_sha256s) == 6
    journal_metadata = tmp_path_factory.mktemp("training-journal-catalog")
    journal_ref = seal_unit(
        run_dir=backing,
        control_dir=journal_metadata,
        logical_root=".",
        paths=tuple(journal_paths),
        kind="journal",
        identity=dict(
            run_id="historical-training-inputs",
            source_commit=source.source_commit,
            config_sha256=config.sha256,
            evidence_identity_sha256=sha256_bytes(canonical_json_bytes(commitment)),
            checkpoint_sha256=training["progress"]["latest"]["sha256"],
            writer_stopped=True,
            checkpoint_committed=True,
        ),
        policy=ArchivePolicy(metadata_bytes=64 * 1024, page_bytes=16 * 1024),
    )

    assert source.source_commit == "42b8ab0ba8a647794b08ce29327fce105e3a75f7"
    assert gate["outcome"] == "debug_non_acceptance"
    assert training["selected_checkpoint"] is None
    assert training["progress"]["latest"] is not None
    assert index["entry_count"] == len(inventory) == 129
    return dict(
        backing=backing,
        gate=gate,
        config=config,
        source=source,
        training=training,
        inventory=inventory,
        journal_metadata=journal_metadata,
        journal_paths=tuple(journal_paths),
        journal_dependencies=tuple(journal_dependencies),
        journal_ref=journal_ref,
    )


def cold_kwargs(function, context):
    return (
        {"evidence_context": context}
        if "evidence_context" in inspect.signature(function).parameters
        else {}
    )


def training_context(case, tmp_path):
    run = tmp_path / "logical-run"
    run.mkdir()
    context = HistoricalTrainingContext(
        run,
        case["backing"],
        case["inventory"],
        journal_ref=case["journal_ref"],
        journal_metadata=case["journal_metadata"],
        journal_paths=case["journal_paths"],
    )
    return run, context


def assert_rng_equal(before, after):
    import numpy as np
    import torch

    assert before.python_state == after.python_state
    assert before.numpy_state[0] == after.numpy_state[0]
    np.testing.assert_array_equal(before.numpy_state[1], after.numpy_state[1])
    assert before.numpy_state[2:] == after.numpy_state[2:]
    assert torch.equal(before.torch_cpu_state, after.torch_cpu_state)
    if before.torch_mps_state is not None:
        assert after.torch_mps_state is not None
        assert torch.equal(before.torch_mps_state, after.torch_mps_state)


def test_genuine_training_inputs_match_cold_reads(historical_training_case, tmp_path, monkeypatch):
    from silent_cascade.rng import snapshot_global_rng
    from silent_cascade.train.pilot_evidence import _verify_available_training

    case = historical_training_case
    install_no_execution_tripwires(monkeypatch)
    eager_unavailable = []
    before = snapshot_global_rng()
    _verify_available_training(
        case["backing"],
        training=case["training"],
        config=case["config"],
        source=case["source"],
        unavailable=eager_unavailable,
    )
    assert_rng_equal(before, snapshot_global_rng())

    run, context = training_context(case, tmp_path)
    cold_unavailable = []
    before = snapshot_global_rng()
    with context.guarded_reads(monkeypatch):
        _verify_available_training(
            run,
            training=case["training"],
            config=case["config"],
            source=case["source"],
            unavailable=cold_unavailable,
            **cold_kwargs(_verify_available_training, context),
        )
    assert_rng_equal(before, snapshot_global_rng())

    assert cold_unavailable == eager_unavailable == []
    assert context.payload_maximum == 1 and context.active == 0


@pytest.mark.parametrize("target", ["none", "nested"])
def test_training_context_requires_exact_run_root_before_reads(
    historical_training_case, tmp_path, target
):
    from silent_cascade.train.pilot_evidence import _verify_available_training

    case = historical_training_case
    run, context = training_context(case, tmp_path)
    supplied = None if target == "none" else run / "nested"
    with pytest.raises(ValueError, match="root"):
        _verify_available_training(
            supplied,
            training=case["training"],
            config=case["config"],
            source=case["source"],
            unavailable=[],
            evidence_context=context,
        )
    assert context.requests == []


def test_late_training_inventory_failure_precedes_payload_reads(historical_training_case, tmp_path):
    from silent_cascade.train.pilot_evidence import _verify_available_training

    case = historical_training_case
    run, context = training_context(case, tmp_path)
    context.late_failure = ValueError("late training inventory failure")
    with pytest.raises(ValueError, match="late training inventory"):
        _verify_available_training(
            run,
            training=case["training"],
            config=case["config"],
            source=case["source"],
            unavailable=[],
            evidence_context=context,
        )
    assert context.requests == []


def test_advertised_training_lease_failure_is_not_unavailable(
    historical_training_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_evidence import _verify_available_training

    case = historical_training_case
    run, context = training_context(case, tmp_path)
    context.lease_failure = case["training"]["progress"]["latest"]["path"]
    install_no_execution_tripwires(monkeypatch)
    with (
        context.guarded_reads(monkeypatch),
        pytest.raises(ValueError, match="advertised training lease"),
    ):
        _verify_available_training(
            run,
            training=case["training"],
            config=case["config"],
            source=case["source"],
            unavailable=[],
            evidence_context=context,
        )
    assert context.active == 0


def test_missing_weight_cannot_hide_corrupt_available_checkpoint(
    historical_training_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_evidence import _verify_available_training
    from silent_cascade.train.state import TrainingError

    case = historical_training_case
    run, context = training_context(case, tmp_path)
    checkpoint = case["training"]["progress"]["latest"]["path"]
    weight = case["training"]["latest_weights"]["path"]
    context.hidden.add(weight)
    control_root = tmp_path / "corrupt-checkpoint"
    control = control_root / checkpoint
    control.parent.mkdir(parents=True)
    control.write_bytes(b"invalid checkpoint")
    assert control.stat().st_size < 64
    context.overrides[checkpoint] = control_root
    install_no_execution_tripwires(monkeypatch)
    with context.guarded_reads(monkeypatch), pytest.raises(TrainingError, match="archive"):
        _verify_available_training(
            run,
            training=case["training"],
            config=case["config"],
            source=case["source"],
            unavailable=[],
            evidence_context=context,
        )
    assert "file:" + checkpoint in context.requests
    assert "file:" + weight not in context.requests
    assert context.payload_maximum == 1 and context.active == 0


def test_missing_weight_cannot_hide_corrupt_available_journal(
    historical_training_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_evidence import _verify_available_training

    case = historical_training_case
    run, context = training_context(case, tmp_path)
    weight = case["training"]["latest_weights"]["path"]
    journal = case["journal_paths"][0]
    context.hidden.add(weight)
    control_root = tmp_path / "corrupt-journal"
    control = control_root / journal
    control.parent.mkdir(parents=True)
    control.write_bytes(b"{}\n")
    context.overrides[journal] = control_root
    install_no_execution_tripwires(monkeypatch)
    with context.guarded_reads(monkeypatch), pytest.raises(ValueError, match="journal hash"):
        _verify_available_training(
            run,
            training=case["training"],
            config=case["config"],
            source=case["source"],
            unavailable=[],
            evidence_context=context,
        )
    assert "file:" + journal in context.requests
    assert context.payload_maximum == 1 and context.active == 0


def test_missing_weight_cannot_hide_corrupt_available_journal_artifact(
    historical_training_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_evidence import _verify_available_training

    case = historical_training_case
    run, context = training_context(case, tmp_path)
    weight = case["training"]["latest_weights"]["path"]
    artifact = next(
        name for name, _ in case["journal_dependencies"] if name.endswith("/validation.json")
    )
    context.hidden.add(weight)
    control_root = tmp_path / "corrupt-journal-artifact"
    control = control_root / artifact
    control.parent.mkdir(parents=True)
    control.write_bytes(b"{}\n")
    context.overrides[artifact] = control_root
    install_no_execution_tripwires(monkeypatch)
    with context.guarded_reads(monkeypatch), pytest.raises(ValueError, match="journal artifact"):
        _verify_available_training(
            run,
            training=case["training"],
            config=case["config"],
            source=case["source"],
            unavailable=[],
            evidence_context=context,
        )
    assert "file:" + artifact in context.requests
    assert context.payload_maximum == 1 and context.active == 0


def test_missing_unindexed_training_inputs_keep_existing_labels(
    historical_training_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_evidence import _verify_available_training

    case = historical_training_case
    run, context = training_context(case, tmp_path)
    checkpoint = case["training"]["progress"]["latest"]["path"]
    weight = case["training"]["latest_weights"]["path"]
    context.hidden.update((checkpoint, weight, case["journal_paths"][0]))
    unavailable = []
    install_no_execution_tripwires(monkeypatch)
    with context.guarded_reads(monkeypatch):
        _verify_available_training(
            run,
            training=case["training"],
            config=case["config"],
            source=case["source"],
            unavailable=unavailable,
            evidence_context=context,
        )
    assert unavailable == [
        "training.archive:latest",
        "training.weights:latest_weights",
        "training.durable_journal",
    ]
    assert context.active == 0


def compact_context(case, tmp_path):
    attachment = case["gate"]["attachments"]["primary"][0]
    run = tmp_path / "compact-root"
    run.mkdir()
    context = HistoricalContext(run, case["backing"], {attachment["path"]: attachment["sha256"]})
    return run, context, attachment


def attachment_model(value):
    from silent_cascade.train.pilot_evidence_types import Attachment

    return Attachment.model_validate(value)


def test_genuine_compact_attachment_matches_cold_read(
    historical_training_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_evidence import compact_attachment_records

    case = historical_training_case
    run, context, value = compact_context(case, tmp_path)
    attachment = attachment_model(value)
    eager = tuple(compact_attachment_records(case["backing"], (attachment,)))
    with context.guarded_reads(monkeypatch):
        cold = tuple(
            compact_attachment_records(
                run,
                (attachment,),
                **cold_kwargs(compact_attachment_records, context),
            )
        )
    assert cold == eager
    assert len(cold) == attachment.rows == 16
    assert attachment.sha256 == "b281ceadbf31240dbf92267ef7249ec9f5b05677bf0cffc946d120475d649325"
    assert context.payload_maximum == 1 and context.active == 0


@pytest.mark.parametrize("damage", ["hash", "count"])
def test_compact_attachment_final_validation_releases_lease(
    historical_training_case, tmp_path, monkeypatch, damage
):
    from silent_cascade.hashing import sha256_bytes
    from silent_cascade.train.pilot_evidence import compact_attachment_records

    case = historical_training_case
    run, context, value = compact_context(case, tmp_path)
    raw = gzip.compress(b'{"control":true}\n', mtime=0)
    assert len(raw) <= 4096
    control_root = tmp_path / "compact-control"
    control = control_root / value["path"]
    control.parent.mkdir(parents=True)
    control.write_bytes(raw)
    context.overrides[value["path"]] = control_root
    changed = dict(value)
    if damage == "count":
        changed.update(sha256=sha256_bytes(raw), rows=2)
    error = "hash" if damage == "hash" else "inventory"
    with context.guarded_reads(monkeypatch), pytest.raises(ValueError, match=error):
        tuple(
            compact_attachment_records(
                run,
                (attachment_model(changed),),
                evidence_context=context,
            )
        )
    assert context.payload_maximum == 1 and context.active == 0


def test_compact_attachment_generator_close_releases_lease(
    historical_training_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_evidence import compact_attachment_records

    case = historical_training_case
    run, context, value = compact_context(case, tmp_path)
    with context.guarded_reads(monkeypatch):
        stream = compact_attachment_records(
            run, (attachment_model(value),), evidence_context=context
        )
        assert next(stream)["row"]["public_id"]
        assert context.active == context.payload_active == 1
        stream.close()
        assert context.active == context.payload_active == 0


def test_compact_export_cannot_fall_back_to_raw_run_shard(historical_training_case, tmp_path):
    from silent_cascade.train.pilot_evidence import compact_attachment_records

    case = historical_training_case
    raw_run, context, value = compact_context(case, tmp_path)
    export = tmp_path / "export"
    export.mkdir()
    assert raw_run != export and not (export / value["path"]).exists()
    with pytest.raises(ValueError, match="root"):
        tuple(
            compact_attachment_records(export, (attachment_model(value),), evidence_context=context)
        )
    assert context.requests == []


@pytest.mark.parametrize(
    ("missing", "unavailable", "allowed"),
    [
        ([], [], True),
        (["raw"], [], False),
        ([], ["semantic"], False),
        (["raw"], ["semantic"], False),
    ],
)
def test_reusable_gate_requires_complete_recorded_evidence(missing, unavailable, allowed):
    from silent_cascade.train.pilot_checks import _require_reusable_gate

    verified = dict(
        passed=False,
        recorded_outcome="debug_non_acceptance",
        missing_raw_attachments=missing,
        unavailable_semantic_checks=unavailable,
    )
    if allowed:
        for outcome in ("debug_non_acceptance", "failed"):
            verified["recorded_outcome"] = outcome
            assert _require_reusable_gate(verified) is None
    else:
        with pytest.raises(ValueError, match="raw or semantic"):
            _require_reusable_gate(verified)


def test_existing_gate_branch_uses_reuse_admission(tmp_path, monkeypatch):
    from silent_cascade.train import pilot_checks

    output = tmp_path / "existing-gate.json"
    output.write_bytes(b"{}")
    monkeypatch.setattr(
        pilot_checks,
        "authenticate_run",
        lambda *args, **kwargs: (tmp_path, None, None, None, None, None),
    )
    monkeypatch.setattr(
        pilot_checks,
        "verify_phase4_gate_artifact",
        lambda *args, **kwargs: dict(
            missing_raw_attachments=[],
            unavailable_semantic_checks=["offline.mps"],
        ),
    )
    with pytest.raises(ValueError, match="raw or semantic"):
        pilot_checks._run_pilot_checks_owned(
            run_dir=tmp_path / "run", config=object(), output_path=output
        )
