import pytest

from silent_cascade.errors import ReplayError


def test_replay_executes_portable_weights(neural_archive_case, tmp_path):
    from silent_cascade.eventflow.neural_replay import verify_neural_replay, write_neural_replay
    from silent_cascade.eventflow.neural_weights import load_neural_weights, save_neural_weights

    case = neural_archive_case
    weights = tmp_path / "weights.safetensors"
    digest = save_neural_weights(weights, model=case.model, identity=case.identity)
    assert (
        load_neural_weights(weights, expected_sha256=digest, device="cpu").identity == case.identity
    )
    result = case.engine.run_episode(case.bundle, case.agent)
    path = tmp_path / "replay.json"
    write_neural_replay(
        path,
        bundle=case.bundle,
        result=result,
        identity=case.identity,
        config=case.config.event_flow,
        weights=weights,
        source_revision=case.revision,
        experiment_config_canonical_json=case.canonical,
    )
    comparison = verify_neural_replay(path, weights_path=weights)
    assert comparison.matched
    assert comparison.trace_sha256 == result.trace.sha256
    assert comparison.weights_sha256 == digest
    weights.write_bytes(b"corrupt")
    with pytest.raises(ReplayError):
        verify_neural_replay(path, weights_path=weights)


def test_rehashed_decision_forgery_is_detected_by_real_execution(neural_archive_case, tmp_path):
    import json

    from silent_cascade.eventflow.neural_replay import verify_neural_replay, write_neural_replay
    from silent_cascade.eventflow.neural_weights import save_neural_weights
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

    case = neural_archive_case
    weights, path = tmp_path / "weights.safetensors", tmp_path / "replay.json"
    save_neural_weights(weights, model=case.model, identity=case.identity)
    result = case.engine.run_episode(case.bundle, case.agent)
    write_neural_replay(
        path,
        bundle=case.bundle,
        result=result,
        identity=case.identity,
        config=case.config.event_flow,
        weights=weights,
        source_revision=case.revision,
        experiment_config_canonical_json=case.canonical,
    )
    payload = json.loads(path.read_bytes())
    row = next(row for row in payload["trace"]["events"] if row["kind"] == "recall")
    row["selected_record_id"] = 999
    payload["trace"]["sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "events": [
                    {k: v for k, v in row.items() if k != "source"}
                    for row in payload["trace"]["events"]
                ],
            }
        )
    )
    payload["payload_sha256"] = sha256_bytes(
        canonical_json_bytes({k: v for k, v in payload.items() if k != "payload_sha256"})
    )
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ReplayError, match="mismatch: trace"):
        verify_neural_replay(path, weights_path=weights)


def test_replay_runtime_failure_is_a_typed_mismatch(neural_archive_case, tmp_path, monkeypatch):
    from silent_cascade.eventflow.neural_replay import verify_neural_replay, write_neural_replay
    from silent_cascade.eventflow.neural_weights import save_neural_weights
    from silent_cascade.models.errors import NeuralError
    from silent_cascade.models.event_flow import EventFlowModel

    case = neural_archive_case
    weights, path = tmp_path / "weights.safetensors", tmp_path / "replay.json"
    save_neural_weights(weights, model=case.model, identity=case.identity)
    result = case.engine.run_episode(case.bundle, case.agent)
    write_neural_replay(
        path,
        bundle=case.bundle,
        result=result,
        identity=case.identity,
        config=case.config.event_flow,
        weights=weights,
        source_revision=case.revision,
        experiment_config_canonical_json=case.canonical,
    )

    def fail(*args):
        raise NeuralError("injected runtime failure")

    monkeypatch.setattr(EventFlowModel, "compose", fail)
    with pytest.raises(ReplayError, match="runtime"):
        verify_neural_replay(path, weights_path=weights)
