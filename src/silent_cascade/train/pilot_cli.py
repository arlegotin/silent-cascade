"""Lightweight root command registration; neural imports occur only in callbacks."""

import json
from pathlib import Path
from typing import Annotated

import typer

from silent_cascade.errors import SilentCascadeError


def train_command(
    config: Annotated[Path, typer.Option("--config")],
    manifest_dir: Annotated[Path, typer.Option("--manifest-dir")],
    run_dir: Annotated[Path, typer.Option("--run-dir")],
    seed: Annotated[int, typer.Option("--seed")] = 11,
    device: Annotated[str, typer.Option("--device")] = "cpu",
    resume: Annotated[Path | None, typer.Option("--resume")] = None,
) -> None:
    """Fit the seed-11 pilot using introduced validation data (never frozen tests)."""
    from silent_cascade.train.pilot_workflow import train_pilot

    try:
        result = train_pilot(
            config_path=config,
            manifest_dir=manifest_dir,
            run_dir=run_dir,
            seed=seed,
            device=device,
            resume=resume,
        )
    except (ValueError, OSError, RuntimeError, SilentCascadeError) as error:
        typer.echo(f"Pilot training refused: {error}", err=True)
        raise typer.Exit(1) from None
    typer.echo(
        f"Pilot execution: {result.status}; updates={result.progress.global_step}; "
        f"production learning gate={'passed' if result.gate_eligible else 'unmet'}"
    )
    if (
        json.loads(result.model_identity.model_config_json)["architecture_profile"] == "production"
        and not result.gate_eligible
    ):
        raise typer.Exit(1)


def evaluate_command(
    checkpoint: Annotated[Path, typer.Option("--checkpoint")],
    manifest: Annotated[Path, typer.Option("--manifest")],
    output: Annotated[Path, typer.Option("--output")],
    device: Annotated[str, typer.Option("--device")] = "cpu",
) -> None:
    """Evaluate concrete portable pilot weights on introduced validation/debug data."""
    from silent_cascade.train.pilot_workflow import evaluate_pilot

    try:
        result = evaluate_pilot(
            checkpoint=checkpoint, manifest=manifest, output=output, device=device
        )
    except (ValueError, OSError, RuntimeError, SilentCascadeError) as error:
        typer.echo(f"Pilot evaluation refused: {error}", err=True)
        raise typer.Exit(1) from None
    typer.echo(
        f"Pilot validation: {result.timed_success_count}/{result.episode_count} timed; "
        f"errors={result.error_count}; production learning gate="
        f"{'passed' if result.gate_passed else 'unmet'}"
    )
    if result.error_count:
        raise typer.Exit(1)


def report_command(
    run_dir: Annotated[Path, typer.Option("--run-dir")],
    output: Annotated[Path, typer.Option("--output")],
    pilot: Annotated[bool, typer.Option("--pilot")] = False,
) -> None:
    """Build an artifact-only pilot validation report."""
    if not pilot:
        raise typer.BadParameter("only --pilot reporting is implemented")
    from silent_cascade.report.pilot import build_pilot_report

    try:
        path = build_pilot_report(run_dir=run_dir, output_dir=output)
    except (ValueError, OSError, RuntimeError, SilentCascadeError) as error:
        typer.echo(f"Pilot report artifact refused: {error}", err=True)
        raise typer.Exit(1) from None
    typer.echo(str(path))


def register_pilot_commands(app: typer.Typer) -> None:
    app.command("train")(train_command)
    app.command("evaluate")(evaluate_command)
    report = typer.Typer(no_args_is_help=True)
    report.command("build")(report_command)
    app.add_typer(report, name="report")
