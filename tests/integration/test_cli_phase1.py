"""Task 17 contracts for the four implemented Phase 1 CLI adapters."""

import json
from pathlib import Path
from uuid import UUID

import pytest
from typer.testing import CliRunner

import silent_cascade.cli as cli
from silent_cascade.env.leakage import LeakageAuditProfileName
from silent_cascade.env.services import (
    ConfigSelection,
    FreezeValidationRequest,
    InspectEpisodeRequest,
    LeakageAuditRequest,
    ManifestCorpusSource,
    OracleEvaluationRequest,
    Phase1GateCorpusSource,
)
from silent_cascade.errors import ManifestAccessError
from silent_cascade.validation import StrictModel

runner = CliRunner()


class _AdapterPayload(StrictModel):
    operation: str
    passed: bool = True


class _AdapterResult(StrictModel):
    report: _AdapterPayload


def _subcommands(group: object) -> set[str | None]:
    return {command.name for command in group.registered_commands}  # type: ignore[attr-defined]


def test_cli_exposes_only_implemented_phase1_commands() -> None:
    """Registering a placeholder command would advertise unimplemented behavior."""
    assert {command.name for command in cli.app.registered_commands} == {"doctor"}
    assert {group.name for group in cli.app.registered_groups} == {
        "data",
        "episode",
        "oracle",
        "leakage",
    }
    assert _subcommands(cli.data_app) == {"freeze"}
    assert _subcommands(cli.episode_app) == {"inspect"}
    assert _subcommands(cli.oracle_app) == {"evaluate"}
    assert _subcommands(cli.leakage_app) == {"audit"}


def test_data_freeze_json_builds_the_exact_production_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing option mapping would freeze evidence from different resolved inputs."""
    observed: list[FreezeValidationRequest] = []

    def freeze(request: FreezeValidationRequest) -> _AdapterResult:
        observed.append(request)
        return _AdapterResult(report=_AdapterPayload(operation="data.freeze"))

    monkeypatch.setattr(cli, "freeze_validation", freeze)
    result = runner.invoke(
        cli.app,
        [
            "data",
            "freeze",
            "--config",
            "base.yaml",
            "--data-config",
            "primary.yaml",
            "--set",
            "runtime.device=cpu",
            "--set",
            "logging.level=debug",
            "--output",
            "validation.json",
            "--root-seed",
            "41",
            "--public-id-seed",
            "91",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"report": {"operation": "data.freeze", "passed": True}}
    assert result.stdout.count("\n") == 1
    assert result.stderr == ""
    assert observed == [
        FreezeValidationRequest(
            config=ConfigSelection(
                base_path=Path("base.yaml"),
                data_path=Path("primary.yaml"),
                set_overrides=("runtime.device=cpu", "logging.level=debug"),
            ),
            output_path=Path("validation.json"),
            episode_count=10_000,
            root_seed=41,
            public_id_seed=91,
        )
    ]


@pytest.mark.parametrize(
    ("selector", "episode_id", "entry_index"),
    [
        (
            ["--episode-id", "00000000-0000-4000-8000-000000000001"],
            UUID("00000000-0000-4000-8000-000000000001"),
            None,
        ),
        (["--entry-index", "7"], None, 7),
    ],
)
def test_episode_inspect_supports_exactly_one_uuid_or_index_selector(
    monkeypatch: pytest.MonkeyPatch,
    selector: list[str],
    episode_id: UUID | None,
    entry_index: int | None,
) -> None:
    """Dropping either selector would make stable private manifest lookup impossible."""
    observed: list[InspectEpisodeRequest] = []

    def inspect(request: InspectEpisodeRequest) -> _AdapterPayload:
        observed.append(request)
        return _AdapterPayload(operation="episode.inspect")

    monkeypatch.setattr(cli, "inspect_episode", inspect)
    result = runner.invoke(
        cli.app,
        ["episode", "inspect", "renamed.json", *selector, "--oracle", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["operation"] == "episode.inspect"
    assert observed == [
        InspectEpisodeRequest(
            config=ConfigSelection(),
            manifest_path=Path("renamed.json"),
            episode_public_id=episode_id,
            entry_index=entry_index,
            include_oracle=True,
        )
    ]


def test_oracle_allocation_mode_builds_the_frozen_internal_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public selector must not become a caller-controlled allocation identity."""
    observed: list[OracleEvaluationRequest] = []

    def evaluate(request: OracleEvaluationRequest) -> _AdapterResult:
        observed.append(request)
        return _AdapterResult(report=_AdapterPayload(operation="oracle.evaluate"))

    monkeypatch.setattr(cli, "evaluate_oracle", evaluate)
    result = runner.invoke(
        cli.app,
        [
            "oracle",
            "evaluate",
            "--allocation",
            "phase1-gate",
            "--root-seed",
            "41",
            "--public-id-seed",
            "91",
            "--output",
            "oracle.json",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed == [
        OracleEvaluationRequest(
            config=ConfigSelection(),
            source=Phase1GateCorpusSource(
                allocation_id="phase1-independent-gate-v1",
                root_seed=41,
                public_id_seed=91,
            ),
            output_path=Path("oracle.json"),
        )
    ]


def test_leakage_manifest_mode_builds_the_exact_profile_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI spelling must map to the strict internal full-gate enum."""
    observed: list[LeakageAuditRequest] = []

    def audit(request: LeakageAuditRequest) -> _AdapterResult:
        observed.append(request)
        return _AdapterResult(report=_AdapterPayload(operation="leakage.audit"))

    monkeypatch.setattr(cli, "run_leakage_audit", audit)
    result = runner.invoke(
        cli.app,
        [
            "leakage",
            "audit",
            "--manifest",
            "validation.json",
            "--profile",
            "phase1-gate",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed == [
        LeakageAuditRequest(
            config=ConfigSelection(),
            source=ManifestCorpusSource(manifest_path=Path("validation.json")),
            profile=LeakageAuditProfileName.PHASE1_GATE,
        )
    ]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["episode", "inspect", "a.json"], "exactly one"),
        (
            [
                "episode",
                "inspect",
                "a.json",
                "--episode-id",
                "00000000-0000-4000-8000-000000000001",
                "--entry-index",
                "0",
            ],
            "exactly one",
        ),
        (
            ["episode", "inspect", "a.json", "--entry-index", "-1"],
            "nonnegative",
        ),
        (["oracle", "evaluate"], "exactly one"),
        (
            [
                "oracle",
                "evaluate",
                "--manifest",
                "a.json",
                "--allocation",
                "phase1-gate",
            ],
            "exactly one",
        ),
        (
            ["oracle", "evaluate", "--manifest", "a.json", "--root-seed", "41"],
            "forbids",
        ),
        (
            ["oracle", "evaluate", "--allocation", "phase1-gate"],
            "requires",
        ),
        (
            [
                "oracle",
                "evaluate",
                "--allocation",
                "other",
                "--root-seed",
                "41",
                "--public-id-seed",
                "91",
            ],
            "phase1-gate",
        ),
        (
            [
                "data",
                "freeze",
                "--output",
                "a.json",
                "--root-seed",
                "-1",
                "--public-id-seed",
                "91",
            ],
            "128-bit",
        ),
        (
            [
                "leakage",
                "audit",
                "--allocation",
                "phase1-gate",
                "--root-seed",
                "41",
                "--public-id-seed",
                "91",
                "--profile",
                "test",
            ],
            "phase1-gate profile",
        ),
    ],
)
def test_invalid_mode_and_seed_combinations_are_typer_usage_errors(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    message: str,
) -> None:
    """Invalid modes must stop at the adapter before any production service runs."""
    monkeypatch.setattr(cli, "freeze_validation", lambda request: pytest.fail("service called"))
    monkeypatch.setattr(cli, "inspect_episode", lambda request: pytest.fail("service called"))
    monkeypatch.setattr(cli, "evaluate_oracle", lambda request: pytest.fail("service called"))
    monkeypatch.setattr(cli, "run_leakage_audit", lambda request: pytest.fail("service called"))

    result = runner.invoke(cli.app, arguments)

    assert result.exit_code == 2
    assert all(token in result.stderr for token in message.split())
    assert "Traceback" not in result.stderr


def test_silent_cascade_error_is_one_stable_stderr_object_without_private_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An access refusal must never become a traceback or echo private recipe values."""
    private_sentinel = "PRIVATE-ROOT-SEED-998877"

    def inspect(request: InspectEpisodeRequest) -> _AdapterPayload:
        del request
        raise ManifestAccessError("oracle inspection is forbidden for frozen test data")

    monkeypatch.setattr(cli, "inspect_episode", inspect)
    result = runner.invoke(
        cli.app,
        ["episode", "inspect", "renamed.json", "--entry-index", "0", "--oracle"],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "code": "manifest_access_error",
        "message": "oracle inspection is forbidden for frozen test data",
        "context": {},
    }
    assert result.stderr.count("\n") == 1
    assert "Traceback" not in result.stderr
    assert private_sentinel not in result.stderr


def test_human_output_is_stable_and_failed_scientific_report_exits_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Human mode still reports complete evidence and propagates a failed gate status."""

    def evaluate(request: OracleEvaluationRequest) -> _AdapterResult:
        del request
        return _AdapterResult(report=_AdapterPayload(operation="oracle.evaluate", passed=False))

    monkeypatch.setattr(cli, "evaluate_oracle", evaluate)
    result = runner.invoke(cli.app, ["oracle", "evaluate", "--manifest", "a.json"])

    assert result.exit_code == 1
    assert result.stderr == ""
    assert result.stdout == (
        "Phase 1 oracle evaluation\n"
        "{\n"
        '  "report": {\n'
        '    "operation": "oracle.evaluate",\n'
        '    "passed": false\n'
        "  }\n"
        "}\n"
    )
