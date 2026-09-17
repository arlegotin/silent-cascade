"""Measured operation timing and honest bounded forecasts."""

import pytest


@pytest.fixture
def verification():
    from silent_cascade.train import pilot_verification

    return pilot_verification


def test_synchronized_timer_measures_the_actual_callable(verification):
    import time

    values = []

    def operation():
        values.append(sum(range(1000)))
        return "actual-result"

    started = time.perf_counter()
    result, seconds = verification.measure_operation(operation, device="cpu")
    assert result == "actual-result" and values == [499500]
    assert 0 < seconds <= time.perf_counter() - started


def test_forecast_counts_validation_and_final_work_without_promising_convergence(verification):
    result = verification.forecast_workload(
        update_seconds=(1.0, 2.0),
        evaluation_seconds_per_episode=(0.1, 0.2),
        component_seconds_per_episode=0.05,
        retained_bytes_per_episode=(1000, 2000),
        checkpoint_bytes=100,
        manifest_bytes=10,
        free_bytes=10_000_000_000,
    )
    assert result.maximum_global_updates == 75000
    assert result.scheduled_validation_boundaries == 75
    assert result.autonomous_validation_episodes_upper == 1_500_000
    assert result.component_validation_episodes_upper == 750_000
    assert result.final_work_episodes > 30000
    assert result.leakage_audit_count == 4
    assert result.compute_seconds_upper > 75000 * 2 + 1_500_000 * 0.2
    assert result.required_free_bytes > 1.2 * result.projected_retained_bytes_upper
    assert result.space_sufficient
    assert not result.convergence_guaranteed


def test_space_forecast_keeps_failures_even_when_destination_is_too_small(verification):
    result = verification.forecast_workload(
        update_seconds=(1.0, 2.0),
        evaluation_seconds_per_episode=(0.1, 0.2),
        component_seconds_per_episode=0.05,
        retained_bytes_per_episode=(1000, 2000),
        checkpoint_bytes=100,
        manifest_bytes=10,
        free_bytes=1,
    )
    assert not result.space_sufficient
    assert result.projected_retained_bytes_upper > 1_000_000_000


def test_forecast_accounts_for_journals_components_audits_and_diagnostic_retention(verification):
    base = dict(
        update_seconds=(1.0, 2.0),
        evaluation_seconds_per_episode=(0.1, 0.2),
        component_seconds_per_episode=0.05,
        retained_bytes_per_episode=(1000, 2000),
        checkpoint_bytes=100,
        manifest_bytes=10,
        free_bytes=10_000_000_000,
    )
    plain = verification.forecast_workload(**base)
    complete = verification.forecast_workload(
        **base,
        journal_bytes_per_update=100,
        component_bytes_per_episode=10,
        diagnostic_retained_bytes=500,
    )
    assert (
        complete.projected_retained_bytes_upper - plain.projected_retained_bytes_upper
        == 75000 * 100 + 750000 * 10 + 500
    )
    assert complete.retention_components["audit_artifact_allowance"] == 40


def test_actual_update_measurement_counts_warmup_and_measured_work(verification, tmp_path):
    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.train.pilot_config import resolve_pilot_config

    config = resolve_pilot_config("phase4_smoke")
    model = EventFlowModel(config.config.neural)
    trial = verification.measure_training_trial(
        config,
        model=model,
        stage="primary",
        device="cpu",
        output_dir=tmp_path / "trial",
        warmup_updates=1,
        measured_updates=2,
    )
    assert trial.warmup_updates == 1 and trial.measured_updates == 2
    assert len(trial.update_seconds) == 2
    assert trial.total_forward_macs > 0 and trial.total_backward_macs > 0
    assert trial.parameter_count == sum(p.numel() for p in model.parameters())
    assert trial.foundation_model_calls == 0
    assert trial.max_trace_steps == 11 and trial.batch_size == 8
    assert trial.raw_step_rows == 3


def test_diagnostic_subset_authenticates_real_parent_manifest(verification, tmp_path):
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_data import freeze_pilot_manifest

    config = resolve_pilot_config("phase4_smoke")
    manifest = freeze_pilot_manifest(
        config, stage="primary", output_path=tmp_path / "manifest.json", source_commit="a" * 40
    )
    subset = verification.declare_subset(manifest, count=8, output_path=tmp_path / "subset.json")
    assert subset.indices == tuple(range(8))
    assert subset.parent_count == 16
    assert subset.variant_counts == {"positive": 4, "safe_negative": 2, "disconnected_negative": 2}
    assert subset.purpose == "action_diagnostic"
    subset.validate_manifest(manifest)
    import pytest

    with pytest.raises(ValueError, match="subset"):
        subset.model_copy(update={"indices": (1, 0, 2, 3, 4, 5, 6, 7)}).validate_manifest(manifest)


def test_public_profile_uses_real_work_and_keeps_cpu_without_mps(
    verification, tmp_path, monkeypatch
):
    import torch

    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_provenance import PilotSourceIdentity

    config = resolve_pilot_config("phase4_smoke")
    source = PilotSourceIdentity(
        source_commit="a" * 40,
        source_files={"fixture": "b" * 64},
        source_sha256="c" * 64,
        plan_revision="d" * 40,
        plan_sha256="e" * 64,
        spec_sha256="f" * 64,
        config_sha256=config.sha256,
    )
    monkeypatch.setattr(
        verification, "authenticate_numeric_source", lambda config, checkpoint: source
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    result = verification.profile_pilot(config, output_dir=tmp_path / "profile")
    assert result.evidence_kind == "throughput_diagnostic"
    assert result.chosen_device == "cpu" and result.final_acceptance_device == "cpu"
    assert result.evaluation_trials["cpu"].episodes == 16
    assert result.evaluation_trials["cpu"].retained_raw_bytes > 0
    assert result.evaluation_trials["cpu"].forward_macs > 0
    assert result.training_trials.keys() == {"cpu/one_hop", "cpu/primary"}
    assert result.production_workload_measured is False
    assert result.foundation_model_calls == 0
    assert result.manifest_subset.parent_count == 16
    assert result.forecast.observed_free_bytes > 0
    assert (
        verification.read_throughput_report(
            tmp_path / "profile", config=config, source_commit=source.source_commit
        )
        == result
    )
    import json

    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

    root = tmp_path / "profile"
    original = (root / "throughput-report.json").read_bytes()
    for mutation in ("counter", "timing", "events", "forecast", "choice", "omitted_update"):
        payload = json.loads(original)
        removed = None
        if mutation == "counter":
            payload["training_trials"]["cpu/primary"]["total_forward_macs"] += 1
        elif mutation == "timing":
            payload["training_trials"]["cpu/primary"]["update_seconds"][0] *= 2
        elif mutation == "events":
            payload["evaluation_trials"]["cpu"]["forward_macs"] += 1
        elif mutation == "forecast":
            payload["forecast"]["compute_seconds_upper"] *= 2
        elif mutation == "choice":
            payload["device_choice_reason"] = "unmeasured choice"
        else:
            path = root / "cpu/primary/update-001.json"
            removed = path, path.read_bytes()
            path.unlink()
            payload["artifact_hashes"].pop("cpu/primary/update-001.json")
        raw = canonical_json_bytes(payload)
        (root / "throughput-report.json").write_bytes(raw)
        (root / "DONE").write_bytes(canonical_json_bytes({"report_sha256": sha256_bytes(raw)}))
        with pytest.raises(ValueError, match=r"raw|summary|closure|forecast|choice"):
            verification.read_throughput_report(
                root, config=config, source_commit=source.source_commit
            )
        if removed:
            removed[0].write_bytes(removed[1])
        (root / "throughput-report.json").write_bytes(original)
        (root / "DONE").write_bytes(canonical_json_bytes({"report_sha256": sha256_bytes(original)}))
