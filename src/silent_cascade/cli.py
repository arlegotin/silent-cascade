"""Silent Cascade command-line interface."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from silent_cascade.doctor import DoctorReport, run_doctor

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def root() -> None:
    """Silent Cascade research tooling."""


def _render_doctor(report: DoctorReport) -> None:
    table = Table(title="Silent Cascade doctor")
    table.add_column("Check")
    table.add_column("Result")
    table.add_row("Python 3.12", "PASS" if report.python_supported else "FAIL")
    table.add_row("Torch CPU", "PASS" if report.torch_cpu_ok else "FAIL")
    table.add_row("MPS", "available" if report.mps.available else "unavailable")
    table.add_row(
        "MPS fallback disabled",
        "PASS" if not report.mps_fallback_enabled else "FAIL",
    )
    rng_ok = report.rng.python_ok and report.rng.numpy_ok and report.rng.torch_cpu_ok
    table.add_row("RNG round trip", "PASS" if rng_ok else "FAIL")
    table.add_row(
        "Flow/guard numeric smoke",
        "PASS" if report.numeric.flow_ok and report.numeric.guard_ok else "FAIL",
    )
    table.add_row(
        "Writable paths",
        "PASS" if all(item.writable for item in report.writable_paths) else "FAIL",
    )
    if report.qwen_extra_requested:
        table.add_row(
            "Qwen extra",
            "PASS" if report.qwen_extra_installed else "FAIL",
        )
    table.add_row("Overall", "PASS" if report.ok else "FAIL")
    Console().print(table)


@app.command("doctor")
def doctor_command(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    qwen: Annotated[bool, typer.Option("--qwen")] = False,
    config: Annotated[str, typer.Option("--config")] = "configs/base.yaml",
) -> None:
    config_path = None if config == "-" else Path(config)
    report = run_doctor(
        config_path=config_path,
        writable_paths=[Path("runs"), Path("reports"), Path("manifests")],
        check_qwen=qwen,
    )
    if json_output:
        typer.echo(json.dumps(report.model_dump(mode="json"), sort_keys=True))
    else:
        _render_doctor(report)
    if not report.ok:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
