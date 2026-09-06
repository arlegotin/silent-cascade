"""Installed Task 18 reproducibility-adapter contracts."""

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from silent_cascade.env import reproducibility
from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.reproducibility import (
    IndependentAllocationReproducibilitySource,
    ManifestReproducibilitySource,
    ReproducibilityReport,
)
from silent_cascade.errors import ArtifactIntegrityError, ManifestAccessError
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.logging.manifest import (
    EpisodeManifest,
    EpisodeManifestEntry,
    ManifestAccessClass,
    MatchedManifestCoordinate,
)
from silent_cascade.provenance import (
    GENERATOR_SOURCE_PATHS,
    PHASE1_ANALYSIS_SOURCE_PATHS,
    AcceptedAttemptRun,
    ConstructionNamespaceEvidence,
    EvidenceProvenance,
    SourceTreeFingerprint,
    public_id_seed_sha256,
)

_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "check_phase1_reproducibility_script",
    Path("scripts/check_phase1_reproducibility.py"),
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
reproducibility_script = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(reproducibility_script)


def _provenance(*, generation_mode: str) -> EvidenceProvenance:
    public_id_seed = 2026083002 if generation_mode == "matched" else 2026083012
    return EvidenceProvenance(
        schema_version="phase1-evidence-provenance-v1",
        plan_base_revision="a" * 40,
        source_commit="b" * 40,
        source_dirty=False,
        generator_version="ofd-v1",
        generation_mode=generation_mode,
        allocation_id=(
            "validation-v1" if generation_mode == "matched" else "phase1-independent-gate-v1"
        ),
        split_namespace=(
            SplitNamespace.VALIDATION
            if generation_mode == "matched"
            else SplitNamespace.PHASE1_GATE
        ),
        config_sha256="c" * 64,
        generator_source=SourceTreeFingerprint(
            frame_version="sc-source-tree-v1",
            scope="generator",
            paths=GENERATOR_SOURCE_PATHS,
            sha256="d" * 64,
        ),
        analysis_source=SourceTreeFingerprint(
            frame_version="sc-source-tree-v1",
            scope="phase1_analysis",
            paths=PHASE1_ANALYSIS_SOURCE_PATHS,
            sha256="e" * 64,
        ),
        root_seed=41,
        public_id_seed_sha256=public_id_seed_sha256(public_id_seed),
    )


def _namespace(*, generation_mode: str) -> ConstructionNamespaceEvidence:
    matched = generation_mode == "matched"
    draw_count = 2_500 if matched else 100_000
    return ConstructionNamespaceEvidence(
        schema_version="construction-namespace-evidence-v1",
        generation_mode=generation_mode,
        public_id_seed=2026083002 if matched else 2026083012,
        accepted_draw_count=draw_count,
        accepted_attempt_runs=(
            AcceptedAttemptRun(first_draw_index=0, draw_count=draw_count, accepted_attempt=0),
        ),
        rejected_draw_count=0,
        generation_attempt_count=draw_count,
        seed_token_count=draw_count * (20 if matched else 7),
        seed_token_sequence_sha256="4" * 64,
        seed_token_collision_count=0,
        base_public_id_count=draw_count * (4 if matched else 1),
        clock_public_id_count=0 if matched else 7_000,
        total_public_id_count=draw_count * (4 if matched else 1) + (0 if matched else 7_000),
        public_id_sequence_sha256="5" * 64,
        public_id_collision_count=0,
    )


def _report(*, source_mode: str) -> ReproducibilityReport:
    generation_mode = "matched" if source_mode == "manifest" else "independent"
    return ReproducibilityReport(
        schema_version="phase1-reproducibility-v2",
        source_mode=source_mode,
        source_payload_sha256="1" * 64,
        sample_size=1_000,
        verified_source_entries=(10_000 if source_mode == "manifest" else 100_000),
        modes=("forward", "reverse", "chunked", "fresh_process"),
        chunk_sizes=(1, 3, 7),
        python_hash_seeds=(0, 1),
        reference_corpus_sha256="2" * 64,
        sample_membership_sha256="3" * 64,
        mismatch_count=0,
        namespace_evidence=_namespace(generation_mode=generation_mode),
        provenance=_provenance(generation_mode=generation_mode),
        passed=True,
    )


def _canonical_validation_manifest() -> EpisodeManifest:
    provenance = _provenance(generation_mode="matched").model_copy(
        update={
            "root_seed": 2026083001,
            "public_id_seed_sha256": (
                "0454fca622eb08379a5d88ecbe0a5ef70f6a15e9ca7d333acecf840f2df38802"
            ),
        }
    )
    path_lengths = (2,) * 834 + (3,) * 833 + (4,) * 833
    entries = tuple(
        EpisodeManifestEntry.model_construct(
            episode_public_id=f"{cohort_index:08d}-0000-4000-8000-{member_index:012d}",
            split_namespace=SplitNamespace.VALIDATION,
            suite=SuiteName.VALIDATION,
            coordinate=MatchedManifestCoordinate(
                cohort_index=cohort_index,
                member_index=member_index,
            ),
            requested_path_length=path_length,
            accepted_attempt=0,
            episode_sha256="1" * 64,
            subset_memberships=(),
            parent_public_id=None,
            parent_episode_sha256=None,
            clock_scale=1.0,
        )
        for cohort_index, path_length in enumerate(path_lengths)
        for member_index in range(4)
    )
    return EpisodeManifest.model_construct(
        schema_version=1,
        experiment_version="v1",
        access_class=ManifestAccessClass.VALIDATION,
        provenance=provenance,
        suite=SuiteName.VALIDATION,
        public_id_seed=2026083002,
        episode_count=10_000,
        entries=entries,
    )


def test_production_manifest_boundary_accepts_the_canonical_source() -> None:
    reproducibility._require_production_manifest_source(_canonical_validation_manifest())


def test_manifest_command_builds_exact_task18_request_and_reuses_identical_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Removing manifest selection, matrix forwarding, or no-clobber reuse breaks Task 18."""
    requests = []
    report = _report(source_mode="manifest")
    monkeypatch.setattr(
        reproducibility_script,
        "check_reproducibility",
        lambda request, _resolved: requests.append(request) or report,
    )
    output = tmp_path / "validation-reproducibility.json"
    arguments = [
        "--manifest",
        "manifests/validation/v1/ofd-primary-10000.json",
        "--sample-size",
        "1000",
        "--chunk-size",
        "1",
        "--chunk-size",
        "3",
        "--chunk-size",
        "7",
        "--python-hash-seed",
        "0",
        "--python-hash-seed",
        "1",
        "--verify-all-source-entries",
        "--output",
        str(output),
    ]

    first = CliRunner().invoke(reproducibility_script.app, arguments)
    before = output.stat().st_mtime_ns
    second = CliRunner().invoke(reproducibility_script.app, arguments)

    assert first.exit_code == second.exit_code == 0
    assert first.stderr == second.stderr == ""
    assert json.loads(first.stdout) == json.loads(second.stdout) == report.model_dump(mode="json")
    assert output.read_bytes() == canonical_json_bytes(report)
    assert output.stat().st_mtime_ns == before
    assert len(requests) == 2
    assert all(isinstance(item.source, ManifestReproducibilitySource) for item in requests)
    assert all(item.sample_size == 1_000 for item in requests)
    assert all(item.chunk_sizes == (1, 3, 7) for item in requests)
    assert all(item.python_hash_seeds == (0, 1) for item in requests)
    assert all(item.verify_all_source_entries for item in requests)


def test_allocation_command_builds_exact_task18_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping either fixed seed or the canonical allocation identity breaks gate provenance."""
    requests = []
    report = _report(source_mode="independent_allocation")
    monkeypatch.setattr(
        reproducibility_script,
        "check_reproducibility",
        lambda request, _resolved: requests.append(request) or report,
    )

    result = CliRunner().invoke(
        reproducibility_script.app,
        [
            "--allocation",
            "phase1-gate",
            "--root-seed",
            "2026083011",
            "--public-id-seed",
            "2026083012",
            "--sample-size",
            "1000",
            "--chunk-size",
            "1",
            "--chunk-size",
            "3",
            "--chunk-size",
            "7",
            "--python-hash-seed",
            "0",
            "--python-hash-seed",
            "1",
            "--verify-all-source-entries",
        ],
    )

    assert result.exit_code == 0
    assert len(requests) == 1
    source = requests[0].source
    assert isinstance(source, IndependentAllocationReproducibilitySource)
    assert source.allocation_id == "phase1-independent-gate-v1"
    assert (source.root_seed, source.public_id_seed) == (2026083011, 2026083012)


def test_divergent_existing_report_is_refused_without_rewrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Overwriting a different report would destroy immutable evidence."""
    report = _report(source_mode="manifest")
    monkeypatch.setattr(
        reproducibility_script,
        "check_reproducibility",
        lambda *_args: report,
    )
    output = tmp_path / "validation-reproducibility.json"
    output.write_bytes(b"different")
    before = output.stat().st_mtime_ns

    result = CliRunner().invoke(
        reproducibility_script.app,
        [
            "--manifest",
            "validation.json",
            "--verify-all-source-entries",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert json.loads(result.stderr)["code"] == "artifact_integrity_error"
    assert output.read_bytes() == b"different"
    assert output.stat().st_mtime_ns == before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: manifest.model_copy(update={"access_class": ManifestAccessClass.DEBUG}),
        lambda manifest: manifest.model_copy(update={"access_class": ManifestAccessClass.FIXTURE}),
        lambda manifest: manifest.model_copy(
            update={"access_class": ManifestAccessClass.FROZEN_TEST}
        ),
        lambda manifest: manifest.model_copy(
            update={"episode_count": 9_999, "entries": manifest.entries[:-1]}
        ),
        lambda manifest: manifest.model_copy(update={"experiment_version": "wrong-v1"}),
        lambda manifest: manifest.model_copy(
            update={"provenance": manifest.provenance.model_copy(update={"root_seed": 41})}
        ),
        lambda manifest: manifest.model_copy(update={"public_id_seed": 91}),
        lambda manifest: manifest.model_copy(
            update={
                "entries": (
                    manifest.entries[0].model_copy(update={"requested_path_length": 4}),
                    *manifest.entries[1:],
                )
            }
        ),
    ],
)
def test_production_manifest_boundary_rejects_noncanonical_sources(
    mutation: Callable[[EpisodeManifest], EpisodeManifest],
) -> None:
    """Relaxing access, count, or ordered recipe admits a different corpus."""
    manifest = mutation(_canonical_validation_manifest())

    with pytest.raises(ManifestAccessError, match="canonical validation"):
        reproducibility._require_production_manifest_source(manifest)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--manifest", "validation.json", "--sample-size", "999"],
        ["--manifest", "validation.json", "--chunk-size", "1"],
        ["--manifest", "validation.json", "--python-hash-seed", "0"],
        ["--manifest", "validation.json"],
    ],
)
def test_production_matrix_must_be_complete_before_library_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    arguments: list[str],
) -> None:
    """An altered or partial production matrix must not generate or publish evidence."""
    calls = 0

    def never_check(*_args):
        nonlocal calls
        calls += 1
        return _report(source_mode="manifest")

    monkeypatch.setattr(reproducibility_script, "check_reproducibility", never_check)
    output = tmp_path / "report.json"
    result = CliRunner().invoke(
        reproducibility_script.app,
        [*arguments, "--output", str(output)],
    )

    assert result.exit_code == 2
    assert calls == 0
    assert not output.exists()


def test_failed_report_is_refused_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Publishing passed=false would poison the immutable Task 18 output path."""
    monkeypatch.setattr(
        reproducibility_script,
        "check_reproducibility",
        lambda *_args: _report(source_mode="manifest").model_copy(update={"passed": False}),
    )
    output = tmp_path / "report.json"

    result = CliRunner().invoke(
        reproducibility_script.app,
        [
            "--manifest",
            "validation.json",
            "--verify-all-source-entries",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.stderr)["code"] == "artifact_integrity_error"
    assert result.stdout == ""
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report.model_copy(update={"source_mode": "independent_allocation"}),
        lambda report: report.model_copy(update={"sample_size": 999}),
        lambda report: report.model_copy(update={"verified_source_entries": 9_999}),
        lambda report: report.model_copy(update={"modes": ("forward",)}),
        lambda report: report.model_copy(update={"chunk_sizes": (1,)}),
        lambda report: report.model_copy(update={"python_hash_seeds": (0,)}),
    ],
)
def test_inexact_report_is_refused_before_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: Callable[[ReproducibilityReport], ReproducibilityReport],
) -> None:
    """Every claimed production denominator and matrix dimension is authenticated."""
    monkeypatch.setattr(
        reproducibility_script,
        "check_reproducibility",
        lambda *_args: mutation(_report(source_mode="manifest")),
    )
    output = tmp_path / "report.json"

    result = CliRunner().invoke(
        reproducibility_script.app,
        [
            "--manifest",
            "validation.json",
            "--verify-all-source-entries",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert json.loads(result.stderr)["code"] == "artifact_integrity_error"
    assert result.stdout == ""
    assert not output.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--manifest", "validation.json", "--allocation", "phase1-gate"],
        ["--manifest", "validation.json", "--root-seed", "41"],
        ["--allocation", "phase1-gate", "--root-seed", "41"],
    ],
)
def test_source_selection_rejects_ambiguous_or_incomplete_modes(arguments: list[str]) -> None:
    """Accepting ambiguous sources or partial seed coordinates can authenticate the wrong corpus."""
    result = CliRunner().invoke(reproducibility_script.app, arguments)

    assert result.exit_code == 2
    assert result.exception is not None


def test_scientific_failure_is_one_stable_typed_stderr_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Letting an expected integrity refusal escape exposes a traceback and unstable internals."""
    monkeypatch.setattr(
        reproducibility_script,
        "check_reproducibility",
        lambda *_args: (_ for _ in ()).throw(ArtifactIntegrityError("reproducibility mismatch")),
    )

    result = CliRunner().invoke(
        reproducibility_script.app,
        ["--manifest", "validation.json", "--verify-all-source-entries"],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "code": "artifact_integrity_error",
        "context": {},
        "message": "reproducibility mismatch",
    }
    assert "Traceback" not in result.stderr


def test_programmer_value_error_remains_visible_at_the_acceptance_script_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Masking an arbitrary ValueError hides defects during the expensive gate."""
    monkeypatch.setattr(
        reproducibility_script,
        "check_reproducibility",
        lambda *_args: (_ for _ in ()).throw(ValueError("PRIVATE-GATE-DETAIL")),
    )

    result = CliRunner().invoke(
        reproducibility_script.app,
        ["--manifest", "validation.json", "--verify-all-source-entries"],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == ""
    assert isinstance(result.exception, ValueError)
    assert str(result.exception) == "PRIVATE-GATE-DETAIL"
