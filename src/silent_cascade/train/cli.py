"""Complete offline Phase 3 commands, separate from the frozen root executable."""

from dataclasses import replace
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import Field, ValidationError
from typer import _click as click
from typer.core import TyperGroup

from silent_cascade.config import resolve_config
from silent_cascade.errors import SilentCascadeError
from silent_cascade.eval.compute import NeuralComputeSnapshot
from silent_cascade.train.checkpoints import (
    _device,
    _expected_hash,
    load_weight_bundle,
    read_weight_metadata,
)
from silent_cascade.train.component_eval import (
    ComponentMetrics,
    ComponentPrediction,
    evaluate_components,
)
from silent_cascade.train.config import Phase3Config
from silent_cascade.train.execution import authenticate_production_execution
from silent_cascade.train.state import TrainingError
from silent_cascade.train.trainer import (
    TrainingRunResult,
    _manifest_metadata,
    _write_json,
    run_training,
)
from silent_cascade.validation import JsonValue, StrictModel


class ErrorDetail(StrictModel):
    code: str
    message: str
    context: dict[str, JsonValue] = Field(default_factory=dict)


class ErrorResponse(StrictModel):
    schema_version: Literal["phase3-cli-error-v1"] = "phase3-cli-error-v1"
    status: Literal["error"] = "error"
    error: ErrorDetail


class FitSuccess(StrictModel):
    schema_version: Literal["phase3-fit-result-v2"] = "phase3-fit-result-v2"
    status: Literal["ok"] = "ok"
    publication: Literal["debug", "production"]
    foundation_model_calls: Literal[0] = 0
    result: TrainingRunResult


class EvaluationArtifact(StrictModel):
    schema_version: Literal["phase3-component-evaluation-v2"] = "phase3-component-evaluation-v2"
    publication: Literal["debug", "production"]
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_seed: Literal[11] = 11
    curriculum_version: Literal["ofd-one-hop-v1"] = "ofd-one-hop-v1"
    evaluation_mode: Literal["unassisted_content"] = "unassisted_content"
    timed: Literal[False] = False
    device: Literal["cpu"] = "cpu"
    foundation_model_calls: Literal[0] = 0
    predictions: tuple[ComponentPrediction, ...]
    metrics: ComponentMetrics
    compute: tuple[NeuralComputeSnapshot, ...]


class EvaluationSuccess(StrictModel):
    schema_version: Literal["phase3-evaluate-result-v2"] = "phase3-evaluate-result-v2"
    status: Literal["ok"] = "ok"
    publication: Literal["debug", "production"]
    output: str
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    episode_count: int = Field(ge=1, le=10_000)
    foundation_model_calls: Literal[0] = 0


class _JsonGroup(TyperGroup):
    """Keep parser and execution failures in the same strict JSON envelope."""

    def main(self, *args, standalone_mode=True, **kwargs):
        try:
            return super().main(*args, standalone_mode=False, **kwargs)
        except click.ClickException as error:
            payload = ErrorDetail(code="usage_error", message=error.format_message())
            code = error.exit_code
        except SilentCascadeError as error:
            payload = ErrorDetail(**error.to_payload())
            code = 1
        except (OSError, ValueError, TypeError, RuntimeError, ValidationError) as error:
            payload = ErrorDetail(code="training_error", message=str(error))
            code = 1
        typer.echo(ErrorResponse(error=payload).model_dump_json())
        if standalone_mode:
            raise SystemExit(code)
        raise typer.Exit(code)


app = typer.Typer(
    cls=_JsonGroup,
    no_args_is_help=True,
    add_completion=False,
    help="Offline Phase 3 neural training and untimed component evaluation.",
    pretty_exceptions_enable=False,
)


@app.command()
def fit(
    config: Annotated[list[Path], typer.Option("--config", help="Ordered explicit YAML layers.")],
    validation_manifest: Annotated[Path, typer.Option("--validation-manifest")],
    run_dir: Annotated[Path, typer.Option("--run-dir")],
    expected_source_commit: Annotated[str, typer.Option("--expected-source-commit")],
    resume: Annotated[Path | None, typer.Option("--resume")] = None,
    device: Annotated[str | None, typer.Option("--device")] = None,
    set_overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
) -> None:
    """Fit the manifest-bound smoke or one-hop profile, optionally resuming."""
    _expected_hash(expected_source_commit, 40)
    # Validate fallback and explicit placement before creating any output.
    _device(device or "cpu")
    resolved = resolve_config(Phase3Config, config, set_overrides=set_overrides or ())
    result = run_training(
        resolved,
        validation_manifest=validation_manifest,
        run_dir=run_dir,
        source_commit=expected_source_commit,
        resume=resume,
        device=device,
    )
    typer.echo(
        FitSuccess(
            result=result,
            publication="production" if resolved.config.training.is_production else "debug",
        ).model_dump_json()
    )


@app.command("evaluate-components")
def evaluate(
    weights: Annotated[Path, typer.Option("--weights")],
    expected_checkpoint_sha256: Annotated[str, typer.Option("--expected-checkpoint-sha256")],
    manifest: Annotated[Path, typer.Option("--manifest")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Regenerate the bound corpus and publish raw unassisted CPU predictions."""
    _device("cpu")
    if output.exists() or output.is_symlink():
        raise TrainingError("Evaluation output already exists")
    config, source = read_weight_metadata(weights, expected_sha256=expected_checkpoint_sha256)
    metadata, manifest_hash = _manifest_metadata(config, manifest, source)
    authenticate_production_execution(config, manifest=metadata, source_commit=source)
    restored = load_weight_bundle(weights, expected_sha256=expected_checkpoint_sha256, device="cpu")
    corpus = metadata.build_corpus(config.config)
    evaluation = evaluate_components(
        restored.model, corpus, batch_size=restored.config.config.training.batch_size
    )
    artifact = EvaluationArtifact(
        publication=metadata.publication,
        source_commit=restored.descriptor.source_commit,
        config_sha256=restored.config.sha256,
        checkpoint_sha256=expected_checkpoint_sha256,
        manifest_sha256=manifest_hash,
        predictions=evaluation.predictions.rows,
        metrics=evaluation.metrics,
        compute=tuple(
            replace(
                counter,
                module_calls=dict(counter.module_calls),
                operation_estimates=dict(counter.operation_estimates),
            )
            for counter in evaluation.compute
        ),
    )
    authenticate_production_execution(config, manifest=metadata, source_commit=source)
    digest = _write_json(output, artifact)
    typer.echo(
        EvaluationSuccess(
            publication=metadata.publication,
            output=str(output),
            output_sha256=digest,
            checkpoint_sha256=expected_checkpoint_sha256,
            episode_count=evaluation.metrics.episode_count,
        ).model_dump_json()
    )
