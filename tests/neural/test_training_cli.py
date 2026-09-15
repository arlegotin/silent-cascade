"""Exercise the complete temporary training commands and their failure envelopes."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from silent_cascade.config import resolve_config
from silent_cascade.train.config import Phase3Config

from .test_trainer import _manifest

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = tuple(
    ROOT / path
    for path in (
        "configs/base.yaml",
        "configs/data/primary.yaml",
        "configs/model/event_flow.yaml",
        "configs/model/neural_components.yaml",
        "configs/train/smoke.yaml",
    )
)


@pytest.fixture
def cli_inputs(tmp_path):
    config = resolve_config(Phase3Config, CONFIGS)
    manifest = _manifest(config, tmp_path / "manifest.json")
    return config, manifest


def fit_args(manifest, directory):
    return [
        "fit",
        *(part for path in CONFIGS for part in ("--config", str(path))),
        "--validation-manifest",
        str(manifest),
        "--run-dir",
        str(directory),
        "--expected-source-commit",
        "a" * 40,
        "--device",
        "cpu",
    ]


def test_real_fit_resume_and_evaluate(cli_inputs, tmp_path):
    from silent_cascade.train.cli import EvaluationArtifact, EvaluationSuccess, FitSuccess, app

    config, manifest = cli_inputs
    runner = CliRunner()
    directory = tmp_path / "run"
    fit = runner.invoke(app, fit_args(manifest, directory))
    assert fit.exit_code == 0, fit.output
    result = FitSuccess.model_validate_json(fit.stdout)
    assert result.result.progress.optimizer_step == 4
    assert result.result.weights.config_sha256 == config.sha256
    conflict = runner.invoke(app, fit_args(manifest, directory))
    assert conflict.exit_code == 1
    assert json.loads(conflict.stdout)["error"]["code"] == "training_error"
    resume = runner.invoke(
        app,
        [
            *fit_args(manifest, directory),
            "--resume",
            str(directory / result.result.latest.relative_path),
        ],
    )
    assert resume.exit_code == 0, resume.output
    assert FitSuccess.model_validate_json(resume.stdout).result.progress.optimizer_step == 4
    output = tmp_path / "evaluation.json"
    args = [
        "evaluate-components",
        "--weights",
        str(directory / result.result.weights.relative_path),
        "--expected-checkpoint-sha256",
        result.result.weights.file_sha256,
        "--manifest",
        str(manifest),
        "--output",
        str(output),
    ]
    evaluation = runner.invoke(app, args)
    assert evaluation.exit_code == 0, evaluation.output
    summary = EvaluationSuccess.model_validate_json(evaluation.stdout)
    artifact = EvaluationArtifact.model_validate_json(output.read_bytes())
    assert artifact.metrics.episode_count == 16
    assert artifact.timed is False
    assert artifact.foundation_model_calls == 0
    assert summary.checkpoint_sha256 == result.result.weights.file_sha256
    conflict = runner.invoke(app, args)
    assert conflict.exit_code == 1
    assert json.loads(conflict.stdout)["error"]["code"] == "training_error"


@pytest.mark.parametrize(
    "case",
    ["missing_config", "override", "source", "manifest", "profile", "frozen", "fallback", "device"],
)
def test_preflight_errors_are_typed_json_and_create_no_run(cli_inputs, tmp_path, monkeypatch, case):
    from silent_cascade.train.cli import ErrorResponse, app

    _, manifest = cli_inputs
    directory = tmp_path / "run"
    args = fit_args(manifest, directory)
    if case == "missing_config":
        args[2] = str(tmp_path / "missing.yaml")
    elif case == "override":
        args += ["--set", "training.unknown=1"]
    elif case == "source":
        args[args.index("--expected-source-commit") + 1] = "b" * 40
    elif case in {"manifest", "frozen"}:
        payload = json.loads(manifest.read_bytes())
        if case == "manifest":
            payload["config_hash"] = "b" * 64
        else:
            payload["entries"][0]["key"]["split"] = "frozen_test"
        manifest.write_text(json.dumps(payload))
    elif case == "profile":
        args += ["--set", "training.profile=unsupported"]
    elif case == "fallback":
        monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    elif case == "device":
        monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
        args[-1] = "mps"
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    error = ErrorResponse.model_validate_json(result.stdout)
    assert error.error.code in {"configuration_error", "training_error"}
    assert error.error.message
    assert not directory.exists()


@pytest.mark.parametrize("args", [["fit"], ["unknown"], ["fit", "--device", "cuda"]])
def test_parser_failures_are_json(args):
    from silent_cascade.train.cli import ErrorResponse, app

    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    assert ErrorResponse.model_validate_json(result.stdout).error.code == "usage_error"


def test_cli_contains_only_completed_commands():
    from typer.main import get_command

    from silent_cascade.train.cli import app

    assert set(get_command(app).commands) == {"fit", "evaluate-components"}


def test_evaluation_rejects_wrong_archive_and_manifest_bindings(cli_inputs, tmp_path, monkeypatch):
    import torch

    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.train.checkpoints import export_weights
    from silent_cascade.train.cli import ErrorResponse, app

    config, manifest = cli_inputs
    torch.manual_seed(11)
    saved = export_weights(
        tmp_path, EventFlowModel(config.config.neural), config=config, source_commit="a" * 40
    )
    output = tmp_path / "evaluation.json"
    args = [
        "evaluate-components",
        "--weights",
        str(tmp_path / saved.relative_path),
        "--expected-checkpoint-sha256",
        saved.file_sha256,
        "--manifest",
        str(manifest),
        "--output",
        str(output),
    ]
    original = manifest.read_bytes()
    for case in ("missing", "hash", "source", "config", "stage", "frozen", "fallback"):
        invocation = args.copy()
        manifest.write_bytes(original)
        if case == "missing":
            invocation[2] = str(tmp_path / "missing.safetensors")
        elif case == "hash":
            invocation[4] = "b" * 64
        elif case == "fallback":
            monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        else:
            payload = json.loads(original)
            if case in {"source", "config"}:
                payload["source_revision" if case == "source" else "config_hash"] = "b" * (
                    40 if case == "source" else 64
                )
            else:
                payload["entries"][0]["key"]["stage" if case == "stage" else "split"] = (
                    "two_hop" if case == "stage" else "frozen_test"
                )
            manifest.write_text(json.dumps(payload))
        result = CliRunner().invoke(app, invocation)
        assert result.exit_code == 1, (case, result.output)
        assert ErrorResponse.model_validate_json(result.stdout).error.code == "training_error"
        assert not output.exists()


def test_ordered_config_layers_and_strict_set_override(tmp_path):
    from silent_cascade.train.cli import FitSuccess, app

    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text("paths:\n  reports: first\n")
    second.write_text("paths:\n  reports: second\n")
    config = resolve_config(
        Phase3Config, (*CONFIGS, first, second), set_overrides=("paths.runs=custom-runs",)
    )
    manifest = _manifest(config, tmp_path / "manifest.json")
    result = CliRunner().invoke(
        app,
        [
            *fit_args(manifest, tmp_path / "run"),
            "--config",
            str(first),
            "--config",
            str(second),
            "--set",
            "paths.runs=custom-runs",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        FitSuccess.model_validate_json(result.stdout).result.weights.config_sha256 == config.sha256
    )
