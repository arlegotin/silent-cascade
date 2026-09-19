"""Artifact-only continuation and offline semantic readers through cold leases."""

import inspect
import json
from dataclasses import replace

import pytest

from .cold_semantics_fixture import (
    HistoricalContext,
    historical_root,
    isolated_referenced_checkpoint,
    referenced_checkpoint,
    relevant_inventory,
    sha256,
)


@pytest.fixture(scope="module")
def historical_case():
    backing = historical_root()
    if backing is None:
        pytest.skip("requires private retained Task 4 smoke evidence")

    from silent_cascade.config import ResolvedConfig
    from silent_cascade.hashing import sha256_bytes
    from silent_cascade.train.pilot_config import parse_phase4_canonical
    from silent_cascade.train.pilot_evidence_types import read_compact_rows
    from silent_cascade.train.pilot_provenance import PilotSourceIdentity

    gate_raw = (backing / "phase4-gate.json").read_bytes()
    gate = json.loads(gate_raw)
    canonical = gate["config_canonical_json"].encode()
    config = ResolvedConfig(
        config=parse_phase4_canonical(canonical.decode()),
        canonical_json=canonical,
        sha256=sha256_bytes(canonical),
        source_paths=(),
    )
    source = PilotSourceIdentity.model_validate(gate["source"])
    record = gate["continuation_evidence"][0]
    rows = tuple(read_compact_rows(backing / gate["attachments"]["primary"][0]["path"]))
    row = next(value for value in rows if value["row"]["public_id"] == record["public_id"])
    assert source.source_commit == "42b8ab0ba8a647794b08ce29327fce105e3a75f7"
    assert gate["outcome"] == "debug_non_acceptance"
    assert (backing / "phase4-gate.json").stat().st_size == 1_334_489
    assert (backing / "phase4-gate.primary.00000.jsonl.gz").stat().st_size == 242_409
    assert (backing / "final/runtime/0.safetensors").stat().st_size == 3_157_874
    assert (backing / "final/runtime/0.replay.json").stat().st_size == 57_958
    return dict(
        backing=backing,
        gate=gate,
        config=config,
        source=source,
        record=record,
        row=row,
    )


def context_for(case, tmp_path):
    run = tmp_path / "logical-run"
    run.mkdir()
    return run, HistoricalContext(run, case["backing"], relevant_inventory(case["gate"]))


def install_no_execution_tripwires(monkeypatch):
    from silent_cascade.eventflow import neural_replay, replay
    from silent_cascade.eventflow.engine import EventEngine
    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.train import pilot_trainer, trainer

    def forbidden(*args, **kwargs):
        raise AssertionError("artifact-only reader crossed an execution boundary")

    for owner, name in (
        (EventFlowModel, "forward"),
        (EventEngine, "run_until"),
        (EventEngine, "run_episode"),
        (neural_replay, "verify_neural_replay"),
        (replay, "verify_replay"),
        (pilot_trainer, "run_pilot_training"),
        (trainer, "run_training"),
    ):
        monkeypatch.setattr(owner, name, forbidden)


def cold_kwargs(function, context):
    return (
        {"evidence_context": context}
        if "evidence_context" in inspect.signature(function).parameters
        else {}
    )


def test_genuine_continuation_and_offline_results_match_cold_reads(
    historical_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_evidence import (
        verify_continuation_records,
        verify_offline_evidence,
    )

    case = historical_case
    install_no_execution_tripwires(monkeypatch)
    eager_unavailable = []
    eager_continuation = verify_continuation_records(
        [case["record"]],
        rows=[case["row"]],
        config=case["config"],
        source=case["source"],
        weights_sha=case["record"]["weights_sha256"],
        run_dir=case["backing"],
        unavailable=eager_unavailable,
    )
    eager_offline = verify_offline_evidence(
        case["gate"]["offline_evidence"],
        source=case["source"],
        run_dir=case["backing"],
        unavailable=eager_unavailable,
    )

    run, context = context_for(case, tmp_path)
    cold_unavailable = []
    with context.guarded_reads(monkeypatch):
        cold_continuation = verify_continuation_records(
            [case["record"]],
            rows=[case["row"]],
            config=case["config"],
            source=case["source"],
            weights_sha=case["record"]["weights_sha256"],
            run_dir=run,
            unavailable=cold_unavailable,
            **cold_kwargs(verify_continuation_records, context),
        )
        cold_offline = verify_offline_evidence(
            case["gate"]["offline_evidence"],
            source=case["source"],
            run_dir=run,
            unavailable=cold_unavailable,
            **cold_kwargs(verify_offline_evidence, context),
        )

    assert (cold_continuation, cold_offline) == (eager_continuation, eager_offline)
    assert cold_unavailable == eager_unavailable == []
    assert case["gate"]["outcome"] == "debug_non_acceptance"
    assert context.maximum == 2 and context.payload_maximum == 1 and context.active == 0
    assert "file:final/runtime/0.safetensors" in context.requests
    assert "file:final/runtime/0.replay.json" in context.requests
    assert f"metadata:{context.evaluation_roots()[0]}" in context.requests
    assert f"episode:{context.evaluation_roots()[0]}:15" in context.requests


def test_context_rejects_undeclared_historical_read(historical_case, tmp_path, monkeypatch):
    _, context = context_for(historical_case, tmp_path)
    with context.guarded_reads(monkeypatch), pytest.raises(AssertionError, match=r"outside.*lease"):
        (historical_case["backing"] / "workflow.json").read_bytes()


def test_wrong_logical_root_is_rejected_before_inventory_or_reads(
    historical_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_evidence import verify_continuation_records

    case = historical_case
    run, context = context_for(case, tmp_path)
    with context.guarded_reads(monkeypatch), pytest.raises(ValueError, match="root"):
        verify_continuation_records(
            [case["record"]],
            rows=[case["row"]],
            config=case["config"],
            source=case["source"],
            weights_sha=case["record"]["weights_sha256"],
            run_dir=run / "nested",
            **cold_kwargs(verify_continuation_records, context),
        )
    assert context.requests == []


def test_late_inventory_failure_is_not_hidden(historical_case, tmp_path, monkeypatch):
    from silent_cascade.train.pilot_evidence import verify_continuation_records

    case = historical_case
    run, context = context_for(case, tmp_path)
    context.late_failure = ValueError("late inventory failure")
    with context.guarded_reads(monkeypatch), pytest.raises(ValueError, match="late inventory"):
        verify_continuation_records(
            [case["record"]],
            rows=[case["row"]],
            config=case["config"],
            source=case["source"],
            weights_sha=case["record"]["weights_sha256"],
            run_dir=run,
            **cold_kwargs(verify_continuation_records, context),
        )
    assert context.requests == []


def test_absent_unindexed_continuation_inputs_remain_unavailable(historical_case, tmp_path):
    from silent_cascade.train.pilot_evidence import verify_continuation_records

    case = historical_case
    run, context = context_for(case, tmp_path)
    context.hidden.update((case["record"]["checkpoint_path"], case["record"]["replay_path"]))
    unavailable = []
    coverage = verify_continuation_records(
        [case["record"]],
        rows=[case["row"]],
        config=case["config"],
        source=case["source"],
        weights_sha=case["record"]["weights_sha256"],
        run_dir=run,
        unavailable=unavailable,
        **cold_kwargs(verify_continuation_records, context),
    )
    assert coverage == tuple(case["record"]["coverage"])
    assert unavailable == [
        "continuation.checkpoint:" + case["record"]["checkpoint_path"],
        "continuation.replay:" + case["record"]["replay_path"],
    ]
    assert context.requests == []


def test_missing_checkpoint_cannot_hide_advertised_corrupt_replay(
    historical_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_evidence import verify_continuation_records

    case = historical_case
    run, context = context_for(case, tmp_path)
    checkpoint = case["record"]["checkpoint_path"]
    replay = case["record"]["replay_path"]
    context.hidden.add(checkpoint)
    control = tmp_path / "corrupt" / replay
    control.parent.mkdir(parents=True)
    control.write_bytes(b"corrupt")
    context.overrides[replay] = tmp_path / "corrupt"
    unavailable = []
    with context.guarded_reads(monkeypatch), pytest.raises(ValueError, match="replay hash"):
        verify_continuation_records(
            [case["record"]],
            rows=[case["row"]],
            config=case["config"],
            source=case["source"],
            weights_sha=case["record"]["weights_sha256"],
            run_dir=run,
            unavailable=unavailable,
            **cold_kwargs(verify_continuation_records, context),
        )
    assert unavailable == ["continuation.checkpoint:" + checkpoint]
    assert context.requests == ["file:" + replay]
    assert context.active == 0


def test_advertised_missing_member_is_a_lease_error_not_absence(
    historical_case, tmp_path, monkeypatch
):
    from silent_cascade.errors import ReplayError
    from silent_cascade.train.pilot_evidence import verify_continuation_records

    case = historical_case
    run, context = context_for(case, tmp_path)
    checkpoint = case["record"]["checkpoint_path"]
    context.overrides[checkpoint] = tmp_path / "empty"
    unavailable = []
    with context.guarded_reads(monkeypatch), pytest.raises(ReplayError):
        verify_continuation_records(
            [case["record"]],
            rows=[case["row"]],
            config=case["config"],
            source=case["source"],
            weights_sha=case["record"]["weights_sha256"],
            run_dir=run,
            unavailable=unavailable,
            **cold_kwargs(verify_continuation_records, context),
        )
    assert unavailable == []
    assert context.requests == ["file:" + checkpoint]
    assert context.active == 0


def test_missing_offline_source_cannot_hide_corrupt_available_update(
    historical_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_evidence import verify_offline_evidence

    case = historical_case
    run, context = context_for(case, tmp_path)
    source_name = "final/offline/executed-source.json"
    step_name = "final/offline/step.json"
    context.hidden.add(source_name)
    control_root = tmp_path / "corrupt-step"
    control = control_root / step_name
    raw = json.dumps(
        {
            "compute": {
                "backward_macs": case["gate"]["offline_evidence"]["backward_macs"] + 1,
                "foundation_model_calls": 0,
            }
        }
    ).encode()
    assert len(raw) <= 256
    control.parent.mkdir(parents=True)
    control.write_bytes(raw)
    context.overrides[step_name] = control_root
    unavailable = []
    with context.guarded_reads(monkeypatch), pytest.raises(ValueError, match="update/count"):
        verify_offline_evidence(
            case["gate"]["offline_evidence"],
            source=case["source"],
            run_dir=run,
            unavailable=unavailable,
            **cold_kwargs(verify_offline_evidence, context),
        )
    assert unavailable == ["offline.executed_source"]
    assert context.requests == ["file:" + step_name]
    assert context.active == 0


def test_offline_scanner_rejects_corruption_in_final_episode(
    historical_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_evidence import verify_offline_evidence

    case = historical_case
    run, context = context_for(case, tmp_path)
    context.corrupt_final_episode(tmp_path / "final-control")
    install_no_execution_tripwires(monkeypatch)
    with context.guarded_reads(monkeypatch), pytest.raises(ValueError, match="integrity mismatch"):
        verify_offline_evidence(
            case["gate"]["offline_evidence"],
            source=case["source"],
            run_dir=run,
            **cold_kwargs(verify_offline_evidence, context),
        )
    assert f"episode:{context.evaluation_roots()[0]}:15" in context.requests
    assert context.active == 0


def test_genuine_referenced_runtime_accepts_only_bound_materialized_weights(
    historical_case, tmp_path, monkeypatch
):
    from silent_cascade.errors import ReplayError
    from silent_cascade.eventflow.neural_checkpoint import load_neural_runtime_checkpoint
    from silent_cascade.eventflow.neural_weights import load_neural_weights

    install_no_execution_tripwires(monkeypatch)
    if "materialized_weights" not in inspect.signature(load_neural_runtime_checkpoint).parameters:
        pytest.fail("runtime loader lacks bounded materialized shared-weight input")
    checkpoint_path, weights_path = referenced_checkpoint(historical_case["backing"])
    weights_sha = weights_path.name.removeprefix("weights-").removesuffix(".safetensors")
    weights = load_neural_weights(weights_path, expected_sha256=weights_sha, device="cpu")
    checkpoint_sha = sha256(checkpoint_path)
    eager = load_neural_runtime_checkpoint(
        checkpoint_path,
        expected_sha256=checkpoint_sha,
        config=historical_case["config"].config.event_flow,
        source_revision=historical_case["source"].source_commit,
        device="cpu",
    )
    materialized = load_neural_runtime_checkpoint(
        checkpoint_path,
        expected_sha256=checkpoint_sha,
        config=historical_case["config"].config.event_flow,
        source_revision=historical_case["source"].source_commit,
        device="cpu",
        materialized_weights=weights,
    )
    assert materialized.metadata == eager.metadata
    with pytest.raises(ReplayError, match="referenced weights"):
        load_neural_runtime_checkpoint(
            checkpoint_path,
            expected_sha256=checkpoint_sha,
            config=historical_case["config"].config.event_flow,
            source_revision=historical_case["source"].source_commit,
            device="cpu",
            materialized_weights=replace(weights, sha256="0" * 64),
        )
    isolated = isolated_referenced_checkpoint(
        historical_case["backing"], tmp_path / "checkpoint-without-weights"
    )
    with pytest.raises(ReplayError):
        load_neural_runtime_checkpoint(
            isolated,
            expected_sha256=checkpoint_sha,
            config=historical_case["config"].config.event_flow,
            source_revision=historical_case["source"].source_commit,
            device="cpu",
        )
    isolated_materialized = load_neural_runtime_checkpoint(
        isolated,
        expected_sha256=checkpoint_sha,
        config=historical_case["config"].config.event_flow,
        source_revision=historical_case["source"].source_commit,
        device="cpu",
        materialized_weights=weights,
    )
    assert isolated_materialized.metadata == eager.metadata


def test_materialized_weights_reject_requested_device_mismatch(historical_case, monkeypatch):
    from silent_cascade.errors import ReplayError
    from silent_cascade.eventflow.neural_checkpoint import (
        _validated_weights,
        load_neural_runtime_checkpoint,
    )
    from silent_cascade.eventflow.neural_weights import load_neural_weights

    install_no_execution_tripwires(monkeypatch)
    checkpoint_path, weights_path = referenced_checkpoint(historical_case["backing"])
    weights_sha = weights_path.name.removeprefix("weights-").removesuffix(".safetensors")
    weights = load_neural_weights(weights_path, expected_sha256=weights_sha, device="cpu")
    checkpoint = load_neural_runtime_checkpoint(
        checkpoint_path,
        expected_sha256=sha256(checkpoint_path),
        config=historical_case["config"].config.event_flow,
        source_revision=historical_case["source"].source_commit,
        device="cpu",
    )
    with pytest.raises(ReplayError, match="device"):
        _validated_weights(checkpoint, "mps", weights)
