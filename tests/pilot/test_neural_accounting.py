import json

from .test_timed_metrics import make_identity


def test_runtime_accounting_and_sidecars(neural_archive_case, tmp_path):
    from silent_cascade.eval.runner import evaluate_episodes

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    result = evaluate_episodes(
        case.model,
        identity=identity,
        config=case.config,
        episodes=[case.bundle],
        output_dir=tmp_path / "eval",
        device="cpu",
    )
    row = json.loads((result.output_path / "rows.jsonl").read_text().splitlines()[0])
    compute = row["compute"]
    assert compute["flow_evaluations"] == compute["engine"]["flow_evaluations"]
    assert compute["neural_flow_evaluations"] == 0
    assert compute["opportunities"] == 0
    assert compute["records_scored"] == compute["engine"]["records_scored"]
    assert compute["module_calls"]["RecordEncoder"] > 0
    assert compute["record_rows_encoded"] >= 64 * compute["module_calls"]["RecordEncoder"]
    assert compute["forward_macs"] > 0
    sidecar = json.loads((result.output_path / row["neural_trace_ref"]).read_text())
    recalls = [e for e in sidecar["events"] if e["kind"] == "recall"]
    assert recalls
    for event in recalls:
        valid = [score for score in event["recall_scores"] if score["eligible"]]
        assert valid and min(score["rank"] for score in valid) == 1
        assert event["segment_sha256"] and event["prediction_snapshot_sha256"]
        assert event["compute"]["forward_macs"] > 0
    terminal = sidecar["events"][-1]
    assert terminal["kind"] == "terminal"
    assert terminal["predictions"] == [] and terminal["guard_targets"] == []
    # RECALL does a real scorer pass, then another preview pass; no logger rescoring.
    assert recalls[0]["compute"]["module_calls"]["RetrievalScorer"] == 2
    assert recalls[0]["compute"]["records_scored"] == 128


def test_wall_clock_telemetry_does_not_change_semantic_hashes(neural_archive_case, tmp_path):
    from silent_cascade.eval.runner import evaluate_episodes

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    results = [
        evaluate_episodes(
            case.model,
            identity=identity,
            config=case.config,
            episodes=[case.bundle],
            output_dir=tmp_path / f"run{n}",
            device="cpu",
        )
        for n in range(2)
    ]
    rows = [json.loads((r.output_path / "rows.jsonl").read_text()) for r in results]
    assert rows[0] == rows[1]
    assert (
        dict(results[0].artifact_hashes)["episodes/00000.trajectory.json.gz"]
        == dict(results[1].artifact_hashes)["episodes/00000.trajectory.json.gz"]
    )


def test_post_flow_callback_failure_retains_performed_flow(
    neural_archive_case, tmp_path, monkeypatch
):
    from silent_cascade.eval.runner import evaluate_episodes
    from silent_cascade.eventflow import engine
    from silent_cascade.models.errors import NeuralError
    from silent_cascade.models.event_flow import EventFlowModel

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    actual_flows = []
    original_flow = engine.state_at

    def measured_flow(state, time):
        actual_flows.append((state.time, time))
        return original_flow(state, time)

    def fail(self, context):
        raise NeuralError("failure after engine materialized the causal flow")

    monkeypatch.setattr(engine, "state_at", measured_flow)
    monkeypatch.setattr(EventFlowModel, "compose", fail)
    result = evaluate_episodes(
        case.model,
        identity=identity,
        config=case.config,
        episodes=[case.bundle],
        output_dir=tmp_path / "eval",
        device="cpu",
    )
    row = json.loads((result.output_path / "rows.jsonl").read_text())
    assert row["error"]["code"] == "dynamics_error"
    assert row["compute"]["engine"]["flow_evaluations"] < len(actual_flows)
    assert row["compute"]["flow_evaluations"] == len(actual_flows)


def test_operational_snapshot_counts_real_flow_and_is_independent_of_commits(neural_archive_case):
    from dataclasses import FrozenInstanceError

    import pytest

    case = neural_archive_case
    engine = case.engine
    initial = engine.operational_compute_snapshot()
    state = case.agent.initialize(case.bundle.public.init)
    assert engine.advance_to(state, 0.0) is state
    assert engine.operational_compute_snapshot() == initial
    advanced = engine.advance_to(state, 0.5)
    after = engine.operational_compute_snapshot()
    assert after.causal_flow_evaluations - initial.causal_flow_evaluations == 1
    assert advanced.core.counters.flow_evaluations == 1
    assert state.core.counters.flow_evaluations == 0
    with pytest.raises(FrozenInstanceError):
        after.causal_flow_evaluations = 0
    result = engine.run_episode(case.bundle, case.agent)
    assert engine.operational_compute_snapshot().causal_flow_evaluations == (
        after.causal_flow_evaluations + result.counters.flow_evaluations
    )
