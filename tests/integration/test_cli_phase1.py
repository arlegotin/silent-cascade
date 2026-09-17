"""Task 17 contracts for the four implemented Phase 1 CLI adapters."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
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
    _publish_report,
)
from silent_cascade.errors import ArtifactError, ConfigurationError, ManifestAccessError
from silent_cascade.validation import StrictModel

runner = CliRunner()


class _AdapterPayload(StrictModel):
    operation: str
    passed: bool = True


class _AdapterResult(StrictModel):
    report: _AdapterPayload


def _subcommands(group: object) -> set[str | None]:
    return {command.name for command in group.registered_commands}  # type: ignore[attr-defined]


def test_cli_keeps_phase1_groups_with_implemented_pilot_commands() -> None:
    """Root registration advertises working commands and no future placeholders."""
    assert {command.name for command in cli.app.registered_commands} == {
        "doctor",
        "replay",
        "train",
        "evaluate",
    }
    assert {group.name for group in cli.app.registered_groups} == {
        "data",
        "episode",
        "oracle",
        "leakage",
        "report",
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
    assert result.stderr == (
        "phase1-progress: data.freeze started\nphase1-progress: data.freeze completed\n"
    )
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


def test_cli_freeze_manifest_is_loadable_and_consumed_by_inspect_and_oracle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The command's declared output must feed its two downstream manifest adapters."""
    from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
    from silent_cascade.env.generator import CohortAllocation, CohortBlock
    from silent_cascade.env.services import (
        Phase1ServiceDependencies,
        build_cohort_manifest,
        evaluate_oracle,
        freeze_validation,
        inspect_episode,
    )
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
    from silent_cascade.logging.manifest import ManifestAccessClass, load_manifest
    from silent_cascade.provenance import (
        GENERATOR_SOURCE_PATHS,
        PHASE1_ANALYSIS_SOURCE_PATHS,
        EvidenceProvenance,
        SourceTreeFingerprint,
        public_id_seed_sha256,
    )

    allocation = CohortAllocation(
        allocation_id="test-cli-validation-v1",
        split_namespace=SplitNamespace.DEBUG,
        blocks=tuple(
            CohortBlock(
                suite=SuiteName.IID_PRIMARY,
                requested_path_length=path,
                first_cohort_index=path - 2,
                cohort_count=1,
            )
            for path in (2, 3, 4)
        ),
    )

    def collect(resolved: object, **kwargs: object) -> EvidenceProvenance:
        return EvidenceProvenance(
            schema_version="phase1-evidence-provenance-v1",
            plan_base_revision="c" * 40,
            source_commit="d" * 40,
            source_dirty=False,
            generator_version="ofd-v1",
            generation_mode=kwargs["generation_mode"],  # type: ignore[arg-type]
            allocation_id=kwargs["allocation_id"],  # type: ignore[arg-type]
            split_namespace=kwargs["split_namespace"],  # type: ignore[arg-type]
            config_sha256=resolved.sha256,  # type: ignore[attr-defined]
            generator_source=SourceTreeFingerprint(
                frame_version="sc-source-tree-v1",
                scope="generator",
                paths=GENERATOR_SOURCE_PATHS,
                sha256="a" * 64,
            ),
            analysis_source=SourceTreeFingerprint(
                frame_version="sc-source-tree-v1",
                scope="phase1_analysis",
                paths=PHASE1_ANALYSIS_SOURCE_PATHS,
                sha256="b" * 64,
            ),
            root_seed=kwargs["root_seed"],  # type: ignore[arg-type]
            public_id_seed_sha256=public_id_seed_sha256(kwargs["public_id_seed"]),  # type: ignore[arg-type]
            analysis_seeds=kwargs["analysis_seeds"],  # type: ignore[arg-type]
        )

    def build(
        config: Phase1Config,
        provenance: EvidenceProvenance,
        root_seed: int,
        public_id_seed: int,
    ):
        return build_cohort_manifest(
            config,
            allocation,
            provenance,
            root_seed,
            public_id_seed,
            access_class=ManifestAccessClass.DEBUG,
        )

    deps = Phase1ServiceDependencies.for_test(
        validation_allocation=allocation,
        collect_provenance=collect,
        build_manifest=build,
    )
    output = tmp_path / "cli-validation.json"
    monkeypatch.setattr(
        cli,
        "freeze_validation",
        lambda request: freeze_validation(
            request.model_copy(update={"episode_count": 12}),
            deps=deps,
        ),
    )
    frozen = runner.invoke(
        cli.app,
        [
            "data",
            "freeze",
            "--output",
            str(output),
            "--root-seed",
            "41",
            "--public-id-seed",
            "91",
            "--json",
        ],
    )

    assert frozen.exit_code == 0, frozen.output
    loaded = load_manifest(output)
    frozen_payload = json.loads(frozen.stdout)
    assert loaded.episode_count == 12
    assert frozen_payload["report"]["manifest_payload_sha256"] == sha256_bytes(
        canonical_json_bytes(loaded)
    )

    monkeypatch.setattr(
        cli,
        "inspect_episode",
        lambda request: inspect_episode(request, deps=deps),
    )
    inspected = runner.invoke(
        cli.app,
        ["episode", "inspect", str(output), "--entry-index", "0", "--json"],
    )
    assert inspected.exit_code == 0, inspected.output
    assert json.loads(inspected.stdout)["episode_public_id"] == loaded.entries[0].episode_public_id

    monkeypatch.setattr(
        cli,
        "evaluate_oracle",
        lambda request: evaluate_oracle(request, deps=deps),
    )
    evaluated = runner.invoke(
        cli.app,
        ["oracle", "evaluate", "--manifest", str(output), "--json"],
    )
    oracle_payload = json.loads(evaluated.stdout)
    assert evaluated.exit_code == 1
    assert oracle_payload["report"]["requested_episode_count"] == 12
    assert oracle_payload["report"]["verified_episode_count"] == 12


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
            "test",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert observed == [
        LeakageAuditRequest(
            config=ConfigSelection(),
            source=ManifestCorpusSource(manifest_path=Path("validation.json")),
            profile=LeakageAuditProfileName.TEST,
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
        (
            [
                "leakage",
                "audit",
                "--manifest",
                "validation.json",
                "--profile",
                "phase1-gate",
            ],
            "manifest mode requires the test profile",
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


def test_typed_phase1_error_context_is_removed_at_the_private_cli_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed filesystem/config diagnostics must not echo private paths or values."""
    private_sentinel = "PRIVATE-PUBLICATION-PATH-998877"

    def inspect(request: InspectEpisodeRequest) -> _AdapterPayload:
        del request
        raise ArtifactError(
            "immutable report publication failed",
            context={"path": private_sentinel, "reason": private_sentinel},
        )

    monkeypatch.setattr(cli, "inspect_episode", inspect)
    result = runner.invoke(
        cli.app,
        ["episode", "inspect", "renamed.json", "--entry-index", "0", "--json"],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "code": "artifact_error",
        "message": "immutable report publication failed",
        "context": {},
    }
    assert private_sentinel not in result.stderr


@pytest.mark.parametrize(
    ("selector", "service_error"),
    [
        (
            ["--entry-index", "10000"],
            ManifestAccessError("inspection entry index is outside the manifest"),
        ),
        (
            ["--episode-id", "ffffffff-ffff-4fff-bfff-ffffffffffff"],
            ManifestAccessError("inspection public ID is not in the manifest"),
        ),
        (
            ["--entry-index", "0"],
            ConfigurationError("manifest config does not match resolved inspection config"),
        ),
        (
            ["--entry-index", "0"],
            ArtifactError("immutable report publication failed"),
        ),
        (
            ["--entry-index", "0"],
            ArtifactError("different immutable report already exists"),
        ),
        (
            ["--entry-index", "0"],
            ArtifactError("published report differs from candidate"),
        ),
    ],
)
def test_expected_task16_service_refusals_are_one_private_safe_typed_error(
    monkeypatch: pytest.MonkeyPatch,
    selector: list[str],
    service_error: ArtifactError | ConfigurationError | ManifestAccessError,
) -> None:
    """Routine selector/config refusals must not expose service stack frames or values."""
    private_sentinel = "PRIVATE-ROOT-SEED-998877"

    def inspect(request: InspectEpisodeRequest) -> _AdapterPayload:
        del request
        raise service_error

    monkeypatch.setattr(cli, "inspect_episode", inspect)
    result = runner.invoke(cli.app, ["episode", "inspect", "renamed.json", *selector, "--json"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "code": service_error.code,
        "message": service_error.message,
        "context": {},
    }
    assert result.stderr.count("\n") == 1
    assert "Traceback" not in result.stderr
    assert private_sentinel not in result.stderr


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(ValueError("programmer bug"), id="unrecognized-value-error"),
        pytest.param(
            ValueError("inspection entry index is outside the manifest"),
            id="former-selector-allowlist-value",
        ),
        pytest.param(
            ValueError("inspection public ID is not in the manifest"),
            id="former-uuid-allowlist-value",
        ),
        pytest.param(
            ValueError("manifest config does not match resolved inspection config"),
            id="former-config-allowlist-value",
        ),
        pytest.param(
            ValueError("immutable report publication failed"),
            id="former-publication-allowlist-value",
        ),
        pytest.param(
            ValueError("different immutable report already exists"),
            id="former-conflict-allowlist-value",
        ),
        pytest.param(
            ValueError("published report differs from candidate"),
            id="former-post-publication-allowlist-value",
        ),
        pytest.param(
            ValueError("inspection entry index is outside the manifest "),
            id="selector-near-miss",
        ),
        pytest.param(
            ValueError("different immutable report already exists: detail"),
            id="publication-near-miss",
        ),
        pytest.param(TypeError("programmer type bug"), id="type-error"),
        pytest.param(RuntimeError("programmer runtime bug"), id="runtime-error"),
    ],
)
def test_unexpected_service_failures_remain_visible_to_the_cli_runtime(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    """Only the exact finite Task 16 refusal contract may become a generic exit-one error."""

    def inspect(request: InspectEpisodeRequest) -> _AdapterPayload:
        del request
        raise error

    monkeypatch.setattr(cli, "inspect_episode", inspect)
    result = runner.invoke(
        cli.app,
        ["episode", "inspect", "renamed.json", "--entry-index", "0", "--json"],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == ""
    assert result.exception is error


def test_immutable_publication_conflict_is_typed_after_progress_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-clobber publication is expected domain refusal, not a programmer traceback."""

    def conflict(request: FreezeValidationRequest) -> _AdapterResult:
        del request
        with TemporaryDirectory(prefix="silent-cascade-cli-conflict-") as raw:
            path = Path(raw) / "report.json"
            _publish_report(path, ConfigSelection())
            _publish_report(path, ConfigSelection(base_path=Path("different.yaml")))
        raise AssertionError("publication conflict did not fail")

    monkeypatch.setattr(cli, "freeze_validation", conflict)
    result = runner.invoke(
        cli.app,
        [
            "data",
            "freeze",
            "--output",
            "unused.json",
            "--root-seed",
            "41",
            "--public-id-seed",
            "91",
            "--json",
        ],
    )

    lines = result.stderr.splitlines()
    assert result.exit_code == 1
    assert result.stdout == ""
    assert lines[0] == "phase1-progress: data.freeze started"
    assert json.loads(lines[1]) == {
        "code": "artifact_error",
        "message": "different immutable report already exists",
        "context": {},
    }
    assert len(lines) == 2
    assert "Traceback" not in result.stderr
    assert "different.yaml" not in result.stderr
    assert "silent-cascade-cli-conflict" not in result.stderr


@pytest.mark.parametrize("json_output", [False, True], ids=["human", "json"])
@pytest.mark.parametrize(
    ("service_name", "arguments", "operation", "title"),
    [
        (
            "freeze_validation",
            [
                "data",
                "freeze",
                "--output",
                "validation.json",
                "--root-seed",
                "41",
                "--public-id-seed",
                "91",
            ],
            "data.freeze",
            "Phase 1 validation freeze",
        ),
        (
            "evaluate_oracle",
            ["oracle", "evaluate", "--manifest", "validation.json"],
            "oracle.evaluate",
            "Phase 1 oracle evaluation",
        ),
        (
            "run_leakage_audit",
            [
                "leakage",
                "audit",
                "--manifest",
                "validation.json",
                "--profile",
                "test",
            ],
            "leakage.audit",
            "Phase 1 leakage audit",
        ),
    ],
)
def test_long_running_commands_emit_deterministic_stderr_progress_only(
    monkeypatch: pytest.MonkeyPatch,
    service_name: str,
    arguments: list[str],
    operation: str,
    title: str,
    json_output: bool,
) -> None:
    """Dropping stderr updates would leave production-scale work silent."""

    def complete(request: object) -> _AdapterResult:
        del request
        return _AdapterResult(report=_AdapterPayload(operation=operation))

    monkeypatch.setattr(cli, service_name, complete)
    result = runner.invoke(cli.app, [*arguments, *(["--json"] if json_output else [])])

    assert result.exit_code == 0, result.output
    assert result.stderr == (
        f"phase1-progress: {operation} started\nphase1-progress: {operation} completed\n"
    )
    if json_output:
        assert json.loads(result.stdout) == {"report": {"operation": operation, "passed": True}}
        assert result.stdout.count("\n") == 1
    else:
        assert result.stdout.startswith(f"{title}\n")
        payload = json.loads(result.stdout.removeprefix(f"{title}\n"))
        assert payload["report"]["operation"] == operation


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
    assert result.stderr == (
        "phase1-progress: oracle.evaluate started\nphase1-progress: oracle.evaluate completed\n"
    )
    assert result.stdout == (
        "Phase 1 oracle evaluation\n"
        "{\n"
        '  "report": {\n'
        '    "operation": "oracle.evaluate",\n'
        '    "passed": false\n'
        "  }\n"
        "}\n"
    )
