"""Real tiny pilot execution under network and optional-model denial hooks."""

import pytest


@pytest.fixture
def verification():
    from silent_cascade.train import pilot_verification

    return pilot_verification


def test_offline_probe_runs_training_evaluation_replay_and_artifact_report(verification, tmp_path):
    result = verification.measure_pilot_offline(output_dir=tmp_path / "offline")
    assert result.training_updates == 1
    assert result.evaluation_episodes == 16
    assert result.replayed_episodes == 1
    assert result.artifact_reports == 1
    assert result.backward_macs > 0
    assert result.foundation_model_calls == 0
    assert result.network_attempts == result.optional_import_attempts == 0
    assert result.forbidden_modules == ()
