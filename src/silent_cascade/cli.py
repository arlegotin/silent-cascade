"""Silent Cascade command-line interface."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated
from uuid import UUID

import typer
from rich.console import Console
from rich.table import Table

from silent_cascade.doctor import DoctorReport, run_doctor
from silent_cascade.env.leakage import LeakageAuditProfileName
from silent_cascade.env.services import (
    ConfigSelection,
    FreezeValidationRequest,
    InspectEpisodeRequest,
    LeakageAuditRequest,
    ManifestCorpusSource,
    OracleEvaluationRequest,
    Phase1GateCorpusSource,
    evaluate_oracle,
    freeze_validation,
    inspect_episode,
    run_leakage_audit,
)
from silent_cascade.errors import SilentCascadeError
from silent_cascade.validation import StrictModel

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True)
episode_app = typer.Typer(no_args_is_help=True)
oracle_app = typer.Typer(no_args_is_help=True)
leakage_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(episode_app, name="episode")
app.add_typer(oracle_app, name="oracle")
app.add_typer(leakage_app, name="leakage")

_PHASE1_ALLOCATION_SELECTOR = "phase1-gate"
_PHASE1_ALLOCATION_ID = "phase1-independent-gate-v1"


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


def _config_selection(
    config: Path,
    data_config: Path,
    set_overrides: list[str] | None,
) -> ConfigSelection:
    return ConfigSelection(
        base_path=config,
        data_path=data_config,
        set_overrides=tuple(set_overrides or ()),
    )


def _seed(value: int, *, option: str) -> int:
    if not 0 <= value < 2**128:
        raise typer.BadParameter(
            "seed must be an unsigned 128-bit integer",
            param_hint=option,
        )
    return value


def _corpus_source(
    *,
    manifest: Path | None,
    allocation: str | None,
    root_seed: int | None,
    public_id_seed: int | None,
) -> ManifestCorpusSource | Phase1GateCorpusSource:
    if (manifest is None) == (allocation is None):
        raise typer.BadParameter(
            "select exactly one of --manifest or --allocation",
            param_hint="--manifest/--allocation",
        )
    if manifest is not None:
        if root_seed is not None or public_id_seed is not None:
            raise typer.BadParameter(
                "manifest mode forbids --root-seed and --public-id-seed",
                param_hint="--manifest",
            )
        return ManifestCorpusSource(manifest_path=manifest)
    assert allocation is not None
    if allocation != _PHASE1_ALLOCATION_SELECTOR:
        raise typer.BadParameter(
            "only the canonical phase1-gate allocation is implemented",
            param_hint="--allocation",
        )
    if root_seed is None or public_id_seed is None:
        raise typer.BadParameter(
            "allocation mode requires --root-seed and --public-id-seed",
            param_hint="--allocation",
        )
    return Phase1GateCorpusSource(
        allocation_id=_PHASE1_ALLOCATION_ID,
        root_seed=_seed(root_seed, option="--root-seed"),
        public_id_seed=_seed(public_id_seed, option="--public-id-seed"),
    )


def _render_phase1_result(result: StrictModel, *, json_output: bool, title: str) -> None:
    payload = result.model_dump(mode="json")
    if json_output:
        typer.echo(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return
    typer.echo(title)
    typer.echo(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _invoke_phase1[RequestT](
    operation: Callable[[RequestT], StrictModel],
    request: RequestT,
    *,
    json_output: bool,
    title: str,
) -> None:
    try:
        result = operation(request)
    except SilentCascadeError as error:
        typer.echo(
            json.dumps(
                error.to_payload(),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None
    _render_phase1_result(result, json_output=json_output, title=title)
    report = getattr(result, "report", None)
    if getattr(report, "passed", True) is False:
        raise typer.Exit(code=1)


@data_app.command("freeze")
def data_freeze_command(
    output: Annotated[Path, typer.Option("--output")],
    root_seed: Annotated[int, typer.Option("--root-seed")],
    public_id_seed: Annotated[int, typer.Option("--public-id-seed")],
    config: Annotated[Path, typer.Option("--config")] = Path("configs/base.yaml"),
    data_config: Annotated[Path, typer.Option("--data-config")] = Path("configs/data/primary.yaml"),
    set_overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Freeze the fixed matched-validation manifest."""
    request = FreezeValidationRequest(
        config=_config_selection(config, data_config, set_overrides),
        output_path=output,
        episode_count=10_000,
        root_seed=_seed(root_seed, option="--root-seed"),
        public_id_seed=_seed(public_id_seed, option="--public-id-seed"),
    )
    _invoke_phase1(
        freeze_validation,
        request,
        json_output=json_output,
        title="Phase 1 validation freeze",
    )


@episode_app.command("inspect")
def episode_inspect_command(
    manifest: Annotated[Path, typer.Argument()],
    episode_id: Annotated[UUID | None, typer.Option("--episode-id")] = None,
    entry_index: Annotated[int | None, typer.Option("--entry-index")] = None,
    include_oracle: Annotated[bool, typer.Option("--oracle")] = False,
    config: Annotated[Path, typer.Option("--config")] = Path("configs/base.yaml"),
    data_config: Annotated[Path, typer.Option("--data-config")] = Path("configs/data/primary.yaml"),
    set_overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Inspect one public manifest episode, optionally with authorized oracle data."""
    if (episode_id is None) == (entry_index is None):
        raise typer.BadParameter(
            "select exactly one of --episode-id or --entry-index",
            param_hint="--episode-id/--entry-index",
        )
    if entry_index is not None and entry_index < 0:
        raise typer.BadParameter(
            "entry index must be nonnegative",
            param_hint="--entry-index",
        )
    request = InspectEpisodeRequest(
        config=_config_selection(config, data_config, set_overrides),
        manifest_path=manifest,
        episode_public_id=episode_id,
        entry_index=entry_index,
        include_oracle=include_oracle,
    )
    _invoke_phase1(
        inspect_episode,
        request,
        json_output=json_output,
        title="Phase 1 episode inspection",
    )


@oracle_app.command("evaluate")
def oracle_evaluate_command(
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    allocation: Annotated[str | None, typer.Option("--allocation")] = None,
    root_seed: Annotated[int | None, typer.Option("--root-seed")] = None,
    public_id_seed: Annotated[int | None, typer.Option("--public-id-seed")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    config: Annotated[Path, typer.Option("--config")] = Path("configs/base.yaml"),
    data_config: Annotated[Path, typer.Option("--data-config")] = Path("configs/data/primary.yaml"),
    set_overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Evaluate the facts-derived oracle and analytic random diagnostic."""
    request = OracleEvaluationRequest(
        config=_config_selection(config, data_config, set_overrides),
        source=_corpus_source(
            manifest=manifest,
            allocation=allocation,
            root_seed=root_seed,
            public_id_seed=public_id_seed,
        ),
        output_path=output,
    )
    _invoke_phase1(
        evaluate_oracle,
        request,
        json_output=json_output,
        title="Phase 1 oracle evaluation",
    )


@leakage_app.command("audit")
def leakage_audit_command(
    profile: Annotated[str, typer.Option("--profile")],
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    allocation: Annotated[str | None, typer.Option("--allocation")] = None,
    root_seed: Annotated[int | None, typer.Option("--root-seed")] = None,
    public_id_seed: Annotated[int | None, typer.Option("--public-id-seed")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    config: Annotated[Path, typer.Option("--config")] = Path("configs/base.yaml"),
    data_config: Annotated[Path, typer.Option("--data-config")] = Path("configs/data/primary.yaml"),
    set_overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Run the fixed leakage audit over a verified corpus source."""
    profiles = {
        "test": LeakageAuditProfileName.TEST,
        "phase1-gate": LeakageAuditProfileName.PHASE1_GATE,
    }
    if profile not in profiles:
        raise typer.BadParameter(
            "profile must be test or phase1-gate",
            param_hint="--profile",
        )
    source = _corpus_source(
        manifest=manifest,
        allocation=allocation,
        root_seed=root_seed,
        public_id_seed=public_id_seed,
    )
    selected_profile = profiles[profile]
    if isinstance(source, Phase1GateCorpusSource) and (
        selected_profile is not LeakageAuditProfileName.PHASE1_GATE
    ):
        raise typer.BadParameter(
            "allocation mode requires the phase1-gate profile",
            param_hint="--profile",
        )
    request = LeakageAuditRequest(
        config=_config_selection(config, data_config, set_overrides),
        source=source,
        profile=selected_profile,
        output_path=output,
    )
    _invoke_phase1(
        run_leakage_audit,
        request,
        json_output=json_output,
        title="Phase 1 leakage audit",
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
