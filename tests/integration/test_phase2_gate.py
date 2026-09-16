"""Source-bound gate contracts; all executions here use twelve debug episodes."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from runpy import run_path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.generator import CohortAllocation, CohortBlock
from silent_cascade.env.services import build_cohort_manifest
from silent_cascade.errors import ArtifactIntegrityError, SilentCascadeError
from silent_cascade.eventflow.config import Phase2Config
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.manifest import ManifestAccessClass, publish_manifest
from silent_cascade.provenance import (
    GENERATOR_SOURCE_PATHS,
    PHASE1_ANALYSIS_SOURCE_PATHS,
    EvidenceProvenance,
    SourceTreeFingerprint,
    public_id_seed_sha256,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("entrypoint", ["public", "verifier"])
def test_gate_parsers_deep_json_is_typed_before_execution(tmp_path, monkeypatch, entrypoint):
    from silent_cascade.eventflow.engine import EventEngine

    raw = b"[" * 1500 + b"0" + b"]" * 1500
    path = tmp_path / "deep.json"
    path.write_bytes(raw)

    def forbidden(*args, **kwargs):
        pytest.fail("malformed gate reached episode execution")

    monkeypatch.setattr(EventEngine, "start_episode", forbidden)
    verifier = run_path(str(ROOT / "scripts/verify_phase2_gate_artifact.py"))
    with pytest.raises(ArtifactIntegrityError):
        if entrypoint == "public":
            gate_api().parse_phase2_gate_bytes(raw)
        else:
            verifier["verify_phase2_gate_artifact"](artifact_path=path, repo_root=ROOT)


def test_standalone_verifier_deep_json_has_no_traceback(tmp_path):
    path = tmp_path / "deep.json"
    path.write_bytes(b"[" * 1500 + b"0" + b"]" * 1500)
    completed = subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "scripts/verify_phase2_gate_artifact.py"),
            "--artifact",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert completed.stdout == "Phase 2 artifact verification failed\n"
    assert completed.stderr == ""


def gate_api():
    assert importlib.util.find_spec("silent_cascade.eventflow.evidence") is not None, (
        "Phase 2 source-bound evidence collector is missing"
    )
    from silent_cascade.eventflow import evidence

    return evidence


def git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture(scope="module")
def debug_gate(tmp_path_factory):
    evidence = gate_api()
    from silent_cascade.eventflow.provenance import (
        PHASE2_ENGINE_SOURCE_PATHS,
        PHASE2_PLAN_PATH,
        SPEC_PATH,
    )

    folder = tmp_path_factory.mktemp("phase2-gate")
    repo = folder / "repo"
    repo.mkdir()
    for relative in (*PHASE2_ENGINE_SOURCE_PATHS, PHASE2_PLAN_PATH, SPEC_PATH):
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Gate Test")
    git(repo, "config", "user.email", "gate@example.invalid")
    config_paths = [
        repo / path
        for path in (
            "configs/base.yaml",
            "configs/data/primary.yaml",
            "configs/model/event_flow.yaml",
        )
    ]
    resolved = resolve_config(Phase2Config, config_paths)
    phase1 = resolve_config(Phase1Config, config_paths[:2])
    source = EvidenceProvenance(
        schema_version="phase1-evidence-provenance-v1",
        plan_base_revision="a" * 40,
        source_commit="b" * 40,
        source_dirty=False,
        generator_version="ofd-v1",
        generation_mode="matched",
        allocation_id="test-phase2-v1",
        split_namespace=SplitNamespace.DEBUG,
        config_sha256=phase1.sha256,
        generator_source=SourceTreeFingerprint(
            frame_version="sc-source-tree-v1",
            scope="generator",
            paths=GENERATOR_SOURCE_PATHS,
            sha256="c" * 64,
        ),
        analysis_source=SourceTreeFingerprint(
            frame_version="sc-source-tree-v1",
            scope="phase1_analysis",
            paths=PHASE1_ANALYSIS_SOURCE_PATHS,
            sha256="d" * 64,
        ),
        root_seed=41,
        public_id_seed_sha256=public_id_seed_sha256(91),
    )
    manifest = build_cohort_manifest(
        phase1.config,
        CohortAllocation(
            allocation_id="test-phase2-v1",
            split_namespace=SplitNamespace.DEBUG,
            blocks=tuple(
                CohortBlock(
                    suite=SuiteName.IID_PRIMARY,
                    requested_path_length=depth,
                    first_cohort_index=depth - 2,
                    cohort_count=1,
                )
                for depth in (2, 3, 4)
            ),
        ),
        source,
        41,
        91,
        access_class=ManifestAccessClass.DEBUG,
    )
    manifest_path = repo / "debug-manifest.json"
    publish_manifest(manifest_path, manifest)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "source and debug recipes")
    commit = git(repo, "rev-parse", "HEAD")
    output = folder / "gate.json"
    options = dict(
        repo_root=repo,
        resolved=resolved,
        manifest_path=manifest_path,
        expected_source_commit=commit,
        expected_plan_base_revision=commit,
        profile="debug",
        requested_episode_count=12,
    )
    report = evidence.collect_phase2_gate(output_path=output, **options)
    return dict(repo=repo, output=output, report=report, options=options, manifest=manifest)


def verifier():
    path = ROOT / "scripts/verify_phase2_gate_artifact.py"
    assert path.is_file(), "independent Phase 2 verifier is missing"
    return run_path(str(path))["verify_phase2_gate_artifact"]


def verify_debug(fixture, artifact=None):
    return verifier()(
        artifact_path=artifact or fixture["output"],
        repo_root=fixture["repo"],
        manifest_path=fixture["options"]["manifest_path"],
        allow_debug=True,
        expected_source_commit=fixture["options"]["expected_source_commit"],
        expected_plan_base_revision=fixture["options"]["expected_plan_base_revision"],
    )


def test_twelve_debug_episodes_bind_real_variants_rows_and_replays(debug_gate):
    report = debug_gate["report"]
    assert report.schema_version == "phase2-engine-gate-debug-v1"
    assert report.access_class == "debug"
    assert report.passed
    assert report.completed_episode_count == report.timed_success_count == 12
    assert (
        report.positive_count,
        report.safe_negative_count,
        report.disconnected_negative_count,
    ) == (6, 3, 3)
    assert len(report.episode_witnesses) == 12
    assert len({row.trace_sha256 for row in report.episode_witnesses}) == 12
    assert len(report.selected_replay_samples) == 3
    assert report.selected_replay_trace_hashes == tuple(
        sample.trace.sha256 for sample in report.selected_replay_samples
    )
    assert report.minimum_internal_gap >= 1e-4
    assert report.maximum_episode_internal_events <= 64
    assert report.foundation_model_calls == 0
    assert verify_debug(debug_gate).passed
    with pytest.raises(ValueError):
        gate_api().Phase2EngineGateReport.model_validate_json(debug_gate["output"].read_bytes())


@pytest.mark.parametrize(
    "field,value",
    [
        ("completed_episode_count", 11),
        ("timed_success_count", 11),
        ("positive_count", 5),
        ("false_action_count", 1),
        ("dynamics_failure_count", 1),
        ("replay_failure_count", 1),
        ("gap_clamp_count", 1),
        ("foundation_model_calls", 1),
        ("config_sha256", "f" * 64),
        ("trace_chain_sha256", "f" * 64),
        ("passed", False),
        ("requested_episode_count", 16),
    ],
)
def test_primitive_aggregate_and_hash_tampering_is_rejected(debug_gate, tmp_path, field, value):
    payload = json.loads(debug_gate["output"].read_bytes())
    payload[field] = value
    target = tmp_path / "tamper.json"
    target.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(SilentCascadeError):
        verify_debug(debug_gate, target)


def test_coordinated_totals_and_variants_cannot_override_raw_rows(debug_gate, tmp_path):
    payload = json.loads(debug_gate["output"].read_bytes())
    payload.update(positive_count=8, safe_negative_count=2, disconnected_negative_count=2)
    target = tmp_path / "totals.json"
    target.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(SilentCascadeError):
        verify_debug(debug_gate, target)


@pytest.mark.parametrize("mutation", ["missing", "order", "entry", "sample", "runtime"])
def test_manifest_and_sample_witnesses_are_bound(debug_gate, tmp_path, mutation):
    payload = json.loads(debug_gate["output"].read_bytes())
    if mutation == "missing":
        payload["episode_witnesses"].pop()
    elif mutation == "order":
        payload["episode_witnesses"].reverse()
    elif mutation == "entry":
        payload["episode_witnesses"][0]["episode_sha256"] = "f" * 64
    elif mutation == "sample":
        payload["selected_replay_trace_hashes"][0] = "f" * 64
    else:
        payload["selected_replay_samples"][0]["config_sha256"] = payload["config_sha256"]
    target = tmp_path / "witness.json"
    target.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(SilentCascadeError):
        verify_debug(debug_gate, target)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "oversize"])
def test_verifier_refuses_unbounded_or_nonregular_inputs(debug_gate, tmp_path, kind):
    target = tmp_path / "unsafe.json"
    if kind == "symlink":
        target.symlink_to(debug_gate["output"])
    elif kind == "fifo":
        os.mkfifo(target)
    else:
        with target.open("wb") as stream:
            stream.truncate(gate_api().MAX_GATE_BYTES + 1)
    with pytest.raises(SilentCascadeError):
        verify_debug(debug_gate, target)


def test_standalone_verifier_does_not_execute_collector_or_episodes(debug_gate, monkeypatch):
    from silent_cascade.eventflow import provenance, replay
    from silent_cascade.eventflow.engine import EventEngine

    def forbidden(*args, **kwargs):
        pytest.fail("independent verifier executed producer or episode machinery")

    monkeypatch.setattr(gate_api(), "collect_phase2_gate", forbidden)
    monkeypatch.setattr(provenance, "collect_phase2_evidence_provenance", forbidden)
    monkeypatch.setattr(EventEngine, "run_episode", forbidden)
    monkeypatch.setattr(replay, "verify_replay", forbidden)
    assert verify_debug(debug_gate).passed


def test_collector_refuses_dirty_source_and_uncommitted_plan(debug_gate, tmp_path):
    repo = debug_gate["repo"]
    for relative in (
        "src/silent_cascade/eventflow/flow.py",
        "src/silent_cascade/eventflow/checkpoint_state.py",
        "docs/superpowers/plans/2026-09-09-phase-2-flow-event-engine.md",
    ):
        path = repo / relative
        saved = path.read_bytes()
        try:
            path.write_bytes(saved + b"\n")
            with pytest.raises(SilentCascadeError):
                gate_api().collect_phase2_gate(
                    output_path=tmp_path / "refused.json", **debug_gate["options"]
                )
            assert not (tmp_path / "refused.json").exists()
        finally:
            path.write_bytes(saved)


def test_collector_never_clobbers_existing_output(debug_gate):
    before = debug_gate["output"].read_bytes()
    with pytest.raises(SilentCascadeError):
        gate_api().collect_phase2_gate(output_path=debug_gate["output"], **debug_gate["options"])
    assert debug_gate["output"].read_bytes() == before


def test_chain_has_independent_length_framing(debug_gate):
    rows = debug_gate["report"].episode_witnesses
    digest = hashlib.sha256(b"silent-cascade/phase2/episode-chain/v1\0")
    for index, row in enumerate(rows):
        raw = json.dumps(
            row.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        digest.update(index.to_bytes(8, "big"))
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    assert digest.hexdigest() == debug_gate["report"].trace_chain_sha256


def test_collector_exception_writes_crash_and_no_gate(debug_gate, tmp_path, monkeypatch):
    from silent_cascade.errors import DynamicsError
    from silent_cascade.eventflow.engine import EventEngine

    def fail_episode(*args, **kwargs):
        raise DynamicsError("injected dynamics failure")

    monkeypatch.setattr(EventEngine, "run_episode", fail_episode)
    output = tmp_path / "failed.json"
    with pytest.raises(SilentCascadeError):
        gate_api().collect_phase2_gate(output_path=output, **debug_gate["options"])
    assert not output.exists()
    assert list((tmp_path / "failed.json.crashes").glob("*.json"))


@pytest.mark.parametrize("script", ["check_phase2_engine.py", "verify_phase2_gate_artifact.py"])
def test_phase2_scripts_have_working_offline_help(script):
    result = subprocess.run(
        ["uv", "run", "--offline", "python", str(ROOT / "scripts" / script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--expected-source-commit" in result.stdout
    assert "--expected-plan-base-revision" in result.stdout


@pytest.fixture(scope="module")
def synthetic_cli_gate(debug_gate, tmp_path_factory):
    """Routing fixture only: three real validation replays, 9,997 synthetic rows.

    Never authenticates as 10,000-execution evidence and never enters manifests.
    Real full-production acceptance is deliberately reserved for Task 14.
    """
    from silent_cascade.env.services import regenerate_entry
    from silent_cascade.eventflow.engine import EventEngine
    from silent_cascade.eventflow.replay import write_replay_artifact
    from silent_cascade.eventflow.scripted import ScriptedEventFlowAgent
    from silent_cascade.logging.manifest import load_manifest

    evidence = gate_api()
    directory = tmp_path_factory.mktemp("synthetic-cli-only")
    manifest_path = ROOT / "manifests/validation/v1/ofd-primary-10000.json"
    manifest = load_manifest(manifest_path)
    cfg = debug_gate["options"]["resolved"].config
    selected = {}
    first_variants = {}
    for index, entry in enumerate(manifest.entries[:4]):
        bundle = regenerate_entry(cfg, manifest, entry)
        variant = bundle.truth.recipe.variant.value
        first_variants[index] = variant
        if variant not in selected:
            result = EventEngine(cfg.event_flow).run_episode(bundle, ScriptedEventFlowAgent())
            sample = write_replay_artifact(
                directory / f"{variant}.json", bundle=bundle, config=cfg.event_flow, result=result
            )
            selected[variant] = (index, sample)
    payload = json.loads(debug_gate["output"].read_bytes())
    payload.update(
        schema_version="phase2-engine-gate-v1",
        access_class="validation_private",
        profile="production",
        requested_episode_count=10_000,
        validation_manifest_file_sha256=sha256_bytes(manifest_path.read_bytes()),
        validation_manifest_payload_sha256=sha256_bytes(canonical_json_bytes(manifest)),
    )
    for field in ("validation_manifest_file_sha256", "validation_manifest_payload_sha256"):
        payload["provenance"][field] = payload[field]
    templates = {row["variant"]: row for row in payload["episode_witnesses"]}
    rows = []
    default_variants = ("positive", "positive", "safe_negative", "disconnected_negative")
    for index, entry in enumerate(manifest.entries):
        variant = first_variants[index] if index < 4 else default_variants[index % 4]
        row = dict(templates[variant])
        row.update(
            entry_index=index,
            episode_public_id=entry.episode_public_id,
            episode_sha256=entry.episode_sha256,
            manifest_entry_sha256=sha256_bytes(canonical_json_bytes(entry)),
            coordinate=entry.coordinate.model_dump(mode="json"),
            trace_sha256=sha256_bytes(f"synthetic-only-{index}".encode()),
        )
        selected_index, sample = selected[variant]
        if selected_index == index:
            internal = [event for event in sample.trace.events if event.source == "internal"]
            row.update(
                trace_sha256=sample.trace.sha256,
                score=sample.expected_result.model_dump(mode="json")["score"],
                internal_event_count=len(internal),
                minimum_internal_gap=min(event.delta for event in internal),
            )
        rows.append(evidence.EpisodeGateWitness.model_validate_json(canonical_json_bytes(row)))
    rows = tuple(rows)
    payload.update(evidence._totals(rows))
    payload["episode_witnesses"] = [row.model_dump(mode="json") for row in rows]
    payload["trace_chain_sha256"] = evidence._chain(rows)
    payload["selected_replay_samples"] = [
        selected[v][1].model_dump(mode="json") for v in evidence.VARIANTS
    ]
    payload["selected_replay_trace_hashes"] = [
        selected[v][1].trace.sha256 for v in evidence.VARIANTS
    ]
    raw = canonical_json_bytes(payload)
    evidence.Phase2EngineGateReport.model_validate_json(raw)
    path = directory / "synthetic-gate-routing-only.json"
    path.write_bytes(raw)
    return path


@pytest.mark.parametrize("index", [0, 1, 2])
def test_cli_selects_exact_real_sample_from_synthetic_production_envelope(
    synthetic_cli_gate, index
):
    from typer.testing import CliRunner

    from silent_cascade.cli import app

    raw = synthetic_cli_gate.read_bytes()
    sample = json.loads(raw)["selected_replay_samples"][index]
    completed = CliRunner().invoke(
        app, ["replay", str(synthetic_cli_gate), "--sample-index", str(index), "--json"]
    )
    assert completed.exit_code == 0, completed.output
    result = json.loads(completed.stdout)
    assert result["artifact_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["trace_sha256"] == sample["trace"]["sha256"]
    assert result["timed_success"] is True
    assert result["foundation_model_calls"] == 0


def test_cli_gate_requires_selector_and_rejects_debug(debug_gate, synthetic_cli_gate):
    from typer.testing import CliRunner

    from silent_cascade.cli import app

    cases = [
        [str(synthetic_cli_gate)],
        [str(synthetic_cli_gate), "--sample-index", "-1"],
        [str(synthetic_cli_gate), "--sample-index", "3"],
        [str(debug_gate["output"]), "--sample-index", "0"],
    ]
    for arguments in cases:
        result = CliRunner().invoke(app, ["replay", *arguments, "--json"])
        assert result.exit_code != 0
        assert json.loads(result.stderr)["code"] == "replay_error"
        assert "Traceback" not in result.stderr


def test_verifier_debug_opt_in_is_closed(debug_gate):
    with pytest.raises(SilentCascadeError):
        verifier()(
            artifact_path=debug_gate["output"],
            repo_root=debug_gate["repo"],
            manifest_path=debug_gate["options"]["manifest_path"],
        )


@pytest.mark.parametrize(
    "field", ["source_tree_sha256", "approved_plan_sha256", "specification_sha256"]
)
def test_verifier_authenticates_each_historical_hash(debug_gate, tmp_path, field):
    payload = json.loads(debug_gate["output"].read_bytes())
    payload["provenance"][field] = "f" * 64
    path = tmp_path / "historical.json"
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(SilentCascadeError):
        verify_debug(debug_gate, path)


def test_clean_other_checkout_cannot_claim_to_be_the_executing_source(debug_gate, tmp_path):
    other = tmp_path / "other"
    subprocess.run(["git", "clone", "-q", str(debug_gate["repo"]), str(other)], check=True)
    git(other, "config", "user.name", "Gate Test")
    git(other, "config", "user.email", "gate@example.invalid")
    source = other / "src/silent_cascade/eventflow/flow.py"
    source.write_bytes(source.read_bytes() + b"\n# another implementation\n")
    git(other, "add", ".")
    git(other, "commit", "-qm", "different runtime")
    options = {
        **debug_gate["options"],
        "repo_root": other,
        "manifest_path": other / "debug-manifest.json",
        "expected_source_commit": git(other, "rev-parse", "HEAD"),
    }
    output = tmp_path / "other.json"
    with pytest.raises(SilentCascadeError):
        gate_api().collect_phase2_gate(output_path=output, **options)
    assert not output.exists()


@pytest.mark.parametrize(
    "failure", ["score", "dynamics", "replay", "clamp", "cap", "reversal", "provenance"]
)
def test_failed_raw_episode_cannot_be_blessed_by_passed_true(debug_gate, tmp_path, failure):
    payload = json.loads(debug_gate["output"].read_bytes())
    sample_ids = {
        sample["expected_result"]["public_id"] for sample in payload["selected_replay_samples"]
    }
    row = next(
        row
        for row in payload["episode_witnesses"]
        if row["variant"] == "positive" and row["episode_public_id"] not in sample_ids
    )
    if failure == "score":
        row["score"].update(timed_success=False, correct_class=False, reason="incorrect")
    else:
        field = {
            "dynamics": "dynamics_failure",
            "replay": "replay_failure",
            "clamp": "gap_clamp_count",
            "cap": "event_cap_failure",
            "reversal": "time_reversal_count",
            "provenance": "provenance_failure",
        }[failure]
        row[field] = 1 if failure in {"clamp", "reversal"} else True
    evidence = gate_api()
    rows = tuple(
        evidence.EpisodeGateWitness.model_validate_json(canonical_json_bytes(row))
        for row in payload["episode_witnesses"]
    )
    payload.update(evidence._totals(rows), trace_chain_sha256=evidence._chain(rows), passed=False)
    failed = evidence.Phase2EngineGateDebugReport.model_validate_json(canonical_json_bytes(payload))
    assert failed.passed is False
    path = tmp_path / "failed-report.json"
    path.write_bytes(canonical_json_bytes(payload))
    assert verify_debug(debug_gate, path).passed is False
    payload["passed"] = True
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(SilentCascadeError):
        verify_debug(debug_gate, path)


def test_nonzero_foundation_call_is_never_published(debug_gate, tmp_path, monkeypatch):
    from dataclasses import replace

    from silent_cascade.eventflow.engine import EventEngine

    original = EventEngine.run_episode

    def add_call(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        return replace(result, counters=replace(result.counters, foundation_model_calls=1))

    monkeypatch.setattr(EventEngine, "run_episode", add_call)
    output = tmp_path / "calls.json"
    with pytest.raises(SilentCascadeError):
        gate_api().collect_phase2_gate(output_path=output, **debug_gate["options"])
    assert not output.exists()
    assert list((tmp_path / "calls.json.crashes").glob("*.json"))


def test_cli_uses_one_bounded_gate_snapshot_for_selection_and_identity(
    synthetic_cli_gate, monkeypatch
):
    from typer.testing import CliRunner

    from silent_cascade import cli

    original = cli._read_replay_input
    snapshot = synthetic_cli_gate.read_bytes()

    def replace_after_read(path):
        raw = original(path)
        path.write_bytes(b"{}")
        return raw

    monkeypatch.setattr(cli, "_read_replay_input", replace_after_read)
    try:
        result = CliRunner().invoke(
            cli.app, ["replay", str(synthetic_cli_gate), "--sample-index", "0", "--json"]
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["artifact_sha256"] == hashlib.sha256(snapshot).hexdigest()
    finally:
        synthetic_cli_gate.write_bytes(snapshot)


def test_failed_publication_durability_removes_only_new_gate(debug_gate, tmp_path, monkeypatch):
    evidence = gate_api()
    original = evidence.os.fsync
    output = tmp_path / "durability.json"
    failed_once = False

    def fail_gate_directory_once(descriptor):
        nonlocal failed_once
        observed = evidence.os.fstat(descriptor)
        if observed.st_ino == tmp_path.stat().st_ino and not failed_once:
            failed_once = True
            raise OSError("injected gate directory fsync failure")
        original(descriptor)

    monkeypatch.setattr(evidence.os, "fsync", fail_gate_directory_once)
    with pytest.raises(SilentCascadeError):
        gate_api().collect_phase2_gate(output_path=output, **debug_gate["options"])
    assert not output.exists(), "a nonzero collection exit must leave no passing gate"
    assert list((tmp_path / "durability.json.crashes").glob("*.json"))
    assert not list(tmp_path.glob(".durability.json.*.tmp"))


@pytest.mark.parametrize("kind", ["different", "identical", "symlink", "unlink_failure"])
def test_gate_rollback_preserves_replacements_and_reports_cleanup_failure(
    tmp_path, monkeypatch, kind
):
    evidence = gate_api()
    output = tmp_path / "publication.json"
    candidate = b"candidate"
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(candidate if kind == "identical" else b"other data")
    original_fsync = evidence.os.fsync
    failed_once = False

    def interrupted_durability(descriptor):
        nonlocal failed_once
        if evidence.os.fstat(descriptor).st_ino != tmp_path.stat().st_ino or failed_once:
            return original_fsync(descriptor)
        failed_once = True
        if kind != "unlink_failure":
            if kind == "symlink":
                output.unlink()
                output.symlink_to(replacement)
            else:
                assert output.stat().st_ino != replacement.stat().st_ino
                replacement.replace(output)
        raise OSError("published but durability unconfirmed")

    monkeypatch.setattr(evidence.os, "fsync", interrupted_durability)
    if kind == "unlink_failure":
        original_unlink = evidence.os.unlink

        def fail_owned_leaf(path, **kwargs):
            if path == output.name and "dir_fd" in kwargs:
                raise OSError("injected rollback refusal")
            return original_unlink(path, **kwargs)

        monkeypatch.setattr(evidence.os, "unlink", fail_owned_leaf)
    with pytest.raises(SilentCascadeError, match="residual output") as caught:
        evidence._publish_gate(output, candidate)
    assert caught.value.context["rollback_failed"] is True
    assert output.read_bytes() == (
        candidate if kind in {"identical", "unlink_failure"} else b"other data"
    )
    if replacement.exists():
        assert replacement.read_bytes() == b"other data"


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_no_clobber_preexisting_output_never_enters_rollback(tmp_path, kind):
    evidence = gate_api()
    output = tmp_path / "existing.json"
    if kind == "symlink":
        target = tmp_path / "target.json"
        target.write_bytes(b"old")
        output.symlink_to(target)
    else:
        output.write_bytes(b"old")
    with pytest.raises(SilentCascadeError):
        evidence._publish_gate(output, b"new")
    assert output.read_bytes() == b"old"
    assert output.is_symlink() == (kind == "symlink")


@pytest.mark.parametrize(
    "failure", ["identity", "identity_unavailable", "file_fsync", "link", "temp_unlink"]
)
def test_owned_publication_preparation_and_temp_cleanup_failures(tmp_path, monkeypatch, failure):
    import stat

    evidence = gate_api()
    output = tmp_path / "owned.json"
    failed_once = False
    operation = {
        "identity": "fstat",
        "identity_unavailable": "fstat",
        "file_fsync": "fsync",
        "link": "link",
        "temp_unlink": "unlink",
    }[failure]
    original = getattr(evidence.os, operation)
    original_fstat = evidence.os.fstat

    def fail_once(*args, **kwargs):
        nonlocal failed_once
        relevant = (
            stat.S_ISREG(original_fstat(args[0]).st_mode)
            if failure in {"identity", "identity_unavailable", "file_fsync"}
            else str(args[0]).endswith(".tmp")
        )
        if relevant and (not failed_once or failure == "identity_unavailable"):
            failed_once = True
            raise OSError("injected owned publication failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(evidence.os, operation, fail_once)
    with pytest.raises(SilentCascadeError) as caught:
        evidence._publish_gate(output, b"owned")
    assert failed_once
    assert not output.exists()
    # An unavailable original identity cannot justify deleting even a temp leaf.
    temporary = list(tmp_path.glob(".owned.json.*.tmp"))
    assert len(temporary) == (1 if failure == "identity_unavailable" else 0)
    if failure == "identity_unavailable":
        assert caught.value.context["rollback_failed"] is True
        assert "residual output" in str(caught.value)


@pytest.mark.parametrize("relationship", ["completed", "gap", "sample_identity"])
def test_data_schema_does_not_replace_independent_relationship_checks(
    debug_gate, tmp_path, relationship
):
    payload = json.loads(debug_gate["output"].read_bytes())
    row = payload["episode_witnesses"][0]
    if relationship == "completed":
        row["completed"] = False
        message = "completed/score relationship"
    elif relationship == "gap":
        row["internal_event_count"] = 0
        message = "internal event/gap relationship"
    else:
        sample = payload["selected_replay_samples"][0]
        sample["expected_result"]["public_id"] = next(
            witness["episode_public_id"]
            for witness in payload["episode_witnesses"]
            if witness["variant"] == "positive"
            and witness["episode_public_id"] != sample["expected_result"]["public_id"]
        )
        sample["payload_sha256"] = hashlib.sha256(
            canonical_json_bytes(
                {key: value for key, value in sample.items() if key != "payload_sha256"}
            )
        ).hexdigest()
        message = "sample public/result identity"
    raw = canonical_json_bytes(payload)
    gate_api().Phase2EngineGateDebugData.model_validate_json(raw)
    with pytest.raises((ValueError, SilentCascadeError)):
        gate_api().Phase2EngineGateDebugReport.model_validate_json(raw)
    output = tmp_path / "relationship.json"
    output.write_bytes(raw)
    with pytest.raises(SilentCascadeError, match=message):
        verify_debug(debug_gate, output)


def test_cli_rejects_inconsistent_nested_and_listed_gate_hashes(synthetic_cli_gate, tmp_path):
    from typer.testing import CliRunner

    from silent_cascade.cli import app

    payload = json.loads(synthetic_cli_gate.read_bytes())
    payload["selected_replay_trace_hashes"][0] = "f" * 64
    output = tmp_path / "bad-sample.json"
    output.write_bytes(canonical_json_bytes(payload))
    result = CliRunner().invoke(app, ["replay", str(output), "--sample-index", "0", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stderr)["code"] == "replay_error"


def test_model_and_verifier_reject_noncanonical_config_shapes(debug_gate, tmp_path):
    for config in (
        [],
        {},
        {"data": None},
        {"data": {"phase1_gate": {"iid_primary": {"02": 8000}}}},
    ):
        payload = json.loads(debug_gate["output"].read_bytes())
        payload["provenance"]["config_canonical_json"] = json.dumps(config)
        output = tmp_path / "bad-config.json"
        output.write_bytes(canonical_json_bytes(payload))
        with pytest.raises(SilentCascadeError):
            verify_debug(debug_gate, output)


def test_verifier_derives_rows_independently_of_producer_arithmetic(
    debug_gate, monkeypatch, tmp_path
):
    from silent_cascade.eventflow import provenance, replay

    payload = json.loads(debug_gate["output"].read_bytes())

    def forbidden(*args, **kwargs):
        pytest.fail("independent arithmetic called a producer summary helper")

    monkeypatch.setattr(gate_api(), "_totals", forbidden)
    monkeypatch.setattr(gate_api(), "_passes", forbidden)
    monkeypatch.setattr(gate_api(), "_chain", forbidden)
    monkeypatch.setattr(gate_api(), "parse_replay_artifact_bytes", forbidden)
    monkeypatch.setattr(gate_api(), "parse_gate_config", forbidden)
    monkeypatch.setattr(provenance, "parse_gate_config", forbidden)
    monkeypatch.setattr(replay, "parse_replay_artifact_bytes", forbidden)
    assert verify_debug(debug_gate).passed
    payload["positive_count"] = 8
    payload["safe_negative_count"] = payload["disconnected_negative_count"] = 2
    output = tmp_path / "coordinated-counts.json"
    output.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(SilentCascadeError):
        verify_debug(debug_gate, output)


def test_cli_rejects_malformed_full_config_with_typed_error(synthetic_cli_gate, tmp_path):
    from typer.testing import CliRunner

    from silent_cascade.cli import app

    payload = json.loads(synthetic_cli_gate.read_bytes())
    payload["provenance"]["config_canonical_json"] = "{}"
    output = tmp_path / "config-shape.json"
    output.write_bytes(canonical_json_bytes(payload))
    result = CliRunner().invoke(app, ["replay", str(output), "--sample-index", "0", "--json"])
    assert result.exit_code == 1
    assert json.loads(result.stderr)["code"] == "replay_error"


def test_independent_verifier_reconstructs_historical_yaml_configuration(debug_gate, monkeypatch):
    module = run_path(str(ROOT / "scripts/verify_phase2_gate_artifact.py"))
    assert "_historical_config" in module, "historical YAML-to-config binding is missing"
    derive = module["_historical_config"]
    source = debug_gate["options"]["expected_source_commit"]
    assert derive(debug_gate["repo"], source) == debug_gate["options"]["resolved"].canonical_json
    original = module["_blob"]

    def changed_yaml(root, revision, path):
        raw = original(root, revision, path)
        if path == "configs/base.yaml":
            return raw.replace(b"primary_offline: true", b"primary_offline: false")
        return raw

    monkeypatch.setitem(derive.__globals__, "_blob", changed_yaml)
    with pytest.raises(SilentCascadeError):
        derive(debug_gate["repo"], source)
