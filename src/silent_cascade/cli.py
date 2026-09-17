"""Silent Cascade command-line interface."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal
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
from silent_cascade.errors import ReplayError, SilentCascadeError
from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at
from silent_cascade.eventflow.checkpoint import (
    load_runtime_checkpoint,
    restore_runtime_session,
)
from silent_cascade.eventflow.config import EventFlowConfig
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.replay import (
    MAX_REPLAY_BYTES,
    parse_replay_artifact_bytes,
    verify_replay,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.crash_bundle import CrashBundleManifest
from silent_cascade.logging.runtime_diagnostics import (
    RuntimeDiagnosticIdentity,
    runtime_diagnostic_identity,
)
from silent_cascade.train.pilot_cli import register_pilot_commands
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
register_pilot_commands(app)

_PHASE1_ALLOCATION_SELECTOR = "phase1-gate"
_PHASE1_ALLOCATION_ID = "phase1-independent-gate-v1"


class EpisodeReplayReport(StrictModel):
    schema_version: Literal["phase2-replay-cli-v1"] = "phase2-replay-cli-v1"
    replay_kind: Literal["episode"] = "episode"
    matched: Literal[True] = True
    artifact_sha256: str
    trace_sha256: str
    event_count: int
    timed_success: bool
    device: Literal["cpu"] = "cpu"
    foundation_model_calls: Literal[0] = 0


class CrashReplayReport(StrictModel):
    schema_version: Literal["phase2-replay-cli-v1"] = "phase2-replay-cli-v1"
    replay_kind: Literal["crash"] = "crash"
    matched: Literal[True] = True
    artifact_sha256: str
    checkpoint_sha256: str
    trace_sha256: str
    event_count: int
    timed_success: None = None
    failure: dict[str, object]
    device: Literal["cpu"] = "cpu"
    foundation_model_calls: Literal[0] = 0


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


def _replay_error(field: str) -> ReplayError:
    return ReplayError(f"replay mismatch: {field}", context={"field": field})


def _read_replay_input(path: Path) -> bytes:
    try:
        with archive_parent(path, error_factory=_replay_error) as (parent, name):
            return read_archive_at(
                parent, name, max_bytes=MAX_REPLAY_BYTES, error_factory=_replay_error
            )
    except OSError as error:
        raise _replay_error("archive.path_or_file") from error


def _closed_schema(raw: bytes) -> str | int:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise _replay_error("archive.json") from error
    if not isinstance(payload, dict) or "schema_version" not in payload:
        raise _replay_error("archive.schema_version")
    schema = payload["schema_version"]
    if schema in ("phase2-replay-v1", "phase2-engine-gate-v1", "phase4-neural-replay-v1") or (
        type(schema) is int and schema == 1
    ):
        return schema
    raise _replay_error("archive.schema_version")


def _runtime_config() -> EventFlowConfig:
    """Return the closed Phase 2 runtime protocol without caller-CWD inputs."""
    return EventFlowConfig.model_validate(
        {
            "dimensions": {
                "z_fast": 256,
                "z_slow": 64,
                "drives": 8,
                "guard_accumulators": 3,
                "focus_key": 64,
                "hypothesis_latent": 64,
            },
            "flow": {"rate_min": 1.0e-5, "rate_max": 20.0, "state_min": -1.0, "state_max": 1.0},
            "guards": {
                "threshold": 1.0,
                "rate_min": 1.0e-5,
                "rate_max": 500.0,
                "active_margin": 1.10,
                "inactive_margin": 0.90,
                "minimum_internal_gap": 1.0e-4,
                "same_kind_refractory": 1.0e-3,
                "near_tie_tolerance": 1.0e-9,
                "maximum_consecutive_gap_clamps": 4,
            },
            "scripted": {"action_target_fraction": 0.825},
            "max_internal_events": 64,
        }
    )


def _crash_checkpoint_name(value: object) -> str:
    if type(value) is not str or Path(value).name != value or value in {"", ".", ".."}:
        raise _replay_error("checkpoint_ref")
    return value


def _recognized_failure(payload: object) -> RuntimeDiagnosticIdentity:
    if not isinstance(payload, dict):
        raise _replay_error("crash.error")
    context = payload.get("context")
    if not isinstance(context, dict):
        raise _replay_error("crash.error")
    code, invariant, certifiable = (
        payload.get("code"),
        context.get("invariant"),
        context.get("replay_certifiable"),
    )
    if type(code) is not str or type(invariant) is not str or certifiable is not True:
        raise _replay_error("crash.error")
    identity = RuntimeDiagnosticIdentity(code, invariant)
    if not identity.replay_certifiable or payload != identity.to_payload():
        raise _replay_error("crash.error")
    return identity


def _replay_crash(path: Path, raw: bytes) -> CrashReplayReport:
    try:
        manifest = CrashBundleManifest.model_validate_json(raw)
    except ValueError as error:
        raise _replay_error("crash.schema") from error
    if raw != canonical_json_bytes(manifest) + b"\n":
        raise _replay_error("crash.canonical_json")
    original = _recognized_failure(manifest.error)
    checkpoint_name = _crash_checkpoint_name(manifest.context.checkpoint_ref)
    if manifest.context.source_revision is None or manifest.context.config_sha256 is None:
        raise _replay_error("crash.context")
    config = _runtime_config()
    if sha256_bytes(canonical_json_bytes(config)) != manifest.context.config_sha256:
        raise _replay_error("crash.config_sha256")
    checkpoint_path = path.parent / checkpoint_name
    try:
        artifact = load_runtime_checkpoint(
            checkpoint_path,
            config=config,
            source_revision=manifest.context.source_revision,
        )
        session, agent = restore_runtime_session(
            artifact,
            config=config,
            source_revision=manifest.context.source_revision,
            device="cpu",
        )
        try:
            EventEngine(config).step(session, agent)
        except SilentCascadeError as error:
            reproduced = runtime_diagnostic_identity(error)
        else:
            raise _replay_error("crash.failure_not_reproduced")
    except ReplayError:
        raise
    except SilentCascadeError as error:
        raise _replay_error("checkpoint") from error
    if not reproduced.replay_certifiable or reproduced.to_payload() != original.to_payload():
        raise _replay_error("crash.failure_identity")
    if artifact.sha256 is None:
        raise _replay_error("checkpoint.sha256")
    return CrashReplayReport(
        artifact_sha256=sha256_bytes(raw),
        checkpoint_sha256=artifact.sha256,
        trace_sha256=artifact.metadata.trace.sha256,
        event_count=len(artifact.metadata.trace.events),
        failure=reproduced.to_payload(),
    )


def _render_replay_error(error: SilentCascadeError) -> None:
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


@app.command("replay")
def replay_command(
    artifact: Annotated[Path, typer.Argument()],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    sample_index: Annotated[int | None, typer.Option("--sample-index")] = None,
    weights: Annotated[Path | None, typer.Option("--weights")] = None,
) -> None:
    """Verify scripted CPU replay or neural replay on its recorded device."""
    try:
        raw = _read_replay_input(artifact)
        schema = _closed_schema(raw)
        if schema == "phase4-neural-replay-v1":
            from silent_cascade.eventflow.neural_replay import verify_neural_replay

            if weights is None or sample_index is not None:
                raise _replay_error("neural.weights_or_sample_index")
            neural_report = verify_neural_replay(artifact, weights_path=weights)
            public_report = neural_report.model_dump(mode="json", exclude={"trace", "result"})
            typer.echo(
                json.dumps(public_report, allow_nan=False, sort_keys=True)
                if json_output
                else "Neural replay matched"
            )
            return
        if weights is not None:
            raise _replay_error("scripted.unexpected_weights")
        if schema == "phase2-engine-gate-v1":
            from silent_cascade.eventflow.evidence import parse_phase2_gate_bytes

            if sample_index is None or not 0 <= sample_index <= 2:
                raise _replay_error("archive.sample_index")
            try:
                gate = parse_phase2_gate_bytes(raw)
            except SilentCascadeError as error:
                raise _replay_error("archive.gate") from error
            loaded = gate.selected_replay_samples[sample_index]
        else:
            if sample_index is not None:
                raise _replay_error("archive.sample_index")
            loaded = parse_replay_artifact_bytes(raw) if schema == "phase2-replay-v1" else None
        if loaded is not None:
            comparison = verify_replay(loaded)
            report: EpisodeReplayReport | CrashReplayReport = EpisodeReplayReport(
                artifact_sha256=sha256_bytes(raw),
                trace_sha256=comparison.trace_sha256,
                event_count=comparison.event_count,
                timed_success=loaded.expected_result.score.timed_success,
            )
        else:
            report = _replay_crash(artifact, raw)
    except SilentCascadeError as error:
        _render_replay_error(error)
        raise typer.Exit(code=1) from None
    if json_output:
        typer.echo(
            json.dumps(
                report.model_dump(mode="json"),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    elif isinstance(report, CrashReplayReport):
        typer.echo("Replay matched: failure reproduced")
    else:
        typer.echo("Replay matched: episode result verified")


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


def _render_phase1_error(error: SilentCascadeError) -> None:
    typer.echo(
        json.dumps(
            {"code": error.code, "context": {}, "message": error.message},
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        err=True,
    )


def _emit_phase1_progress(operation: str, state: str) -> None:
    typer.echo(f"phase1-progress: {operation} {state}", err=True)


def _invoke_phase1[RequestT](
    operation: Callable[[RequestT], StrictModel],
    request: RequestT,
    *,
    json_output: bool,
    title: str,
    progress_operation: str | None = None,
) -> None:
    if progress_operation is not None:
        _emit_phase1_progress(progress_operation, "started")
    try:
        result = operation(request)
    except SilentCascadeError as error:
        _render_phase1_error(error)
        raise typer.Exit(code=1) from None
    if progress_operation is not None:
        _emit_phase1_progress(progress_operation, "completed")
    _render_phase1_result(result, json_output=json_output, title=title)
    report = getattr(result, "report", None)
    if getattr(report, "passed", True) is False:
        raise typer.Exit(code=1)


@data_app.command("freeze")
def data_freeze_command(
    output: Annotated[Path, typer.Option("--output")],
    root_seed: Annotated[int | None, typer.Option("--root-seed")] = None,
    public_id_seed: Annotated[int | None, typer.Option("--public-id-seed")] = None,
    config: Annotated[Path, typer.Option("--config")] = Path("configs/base.yaml"),
    data_config: Annotated[Path, typer.Option("--data-config")] = Path("configs/data/primary.yaml"),
    set_overrides: Annotated[list[str] | None, typer.Option("--set")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    pilot_stage: Annotated[str | None, typer.Option("--pilot-stage")] = None,
) -> None:
    """Freeze the fixed matched-validation manifest."""
    if pilot_stage is not None:
        from silent_cascade.train.pilot_workflow import freeze_pilot

        if (
            root_seed is not None
            or public_id_seed is not None
            or set_overrides
            or data_config != Path("configs/data/primary.yaml")
        ):
            raise typer.BadParameter("pilot seed/data/config overrides are forbidden")
        try:
            manifest = freeze_pilot(config_path=config, stage=pilot_stage, output=output)
        except (ValueError, OSError) as error:
            typer.echo(f"Pilot validation freeze refused: {error}", err=True)
            raise typer.Exit(1) from None
        typer.echo(
            manifest.model_dump_json()
            if json_output
            else f"Pilot {manifest.split} freeze: {manifest.count} entries; "
            "commit-data precondition applies"
        )
        return
    if root_seed is None or public_id_seed is None:
        raise typer.BadParameter(
            "--root-seed and --public-id-seed are required outside --pilot-stage"
        )
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
        progress_operation="data.freeze",
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
        progress_operation="oracle.evaluate",
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
    if isinstance(source, ManifestCorpusSource) and (
        selected_profile is not LeakageAuditProfileName.TEST
    ):
        raise typer.BadParameter(
            "manifest mode requires the test profile",
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
        progress_operation="leakage.audit",
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
