"""Installed Task 18 reproducibility-adapter contracts."""

import importlib.util
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from silent_cascade.env.config import SplitNamespace
from silent_cascade.env.reproducibility import (
    IndependentAllocationReproducibilitySource,
    ManifestReproducibilitySource,
    ReproducibilityReport,
)
from silent_cascade.errors import ArtifactIntegrityError
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.provenance import (
    GENERATOR_SOURCE_PATHS,
    PHASE1_ANALYSIS_SOURCE_PATHS,
    EvidenceProvenance,
    SourceTreeFingerprint,
)

_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "check_phase1_reproducibility_script",
    Path("scripts/check_phase1_reproducibility.py"),
)
assert _SCRIPT_SPEC is not None and _SCRIPT_SPEC.loader is not None
reproducibility_script = importlib.util.module_from_spec(_SCRIPT_SPEC)
_SCRIPT_SPEC.loader.exec_module(reproducibility_script)


def _provenance(*, generation_mode: str) -> EvidenceProvenance:
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
        public_id_seed_sha256="f" * 64,
    )


def _report(*, source_mode: str) -> ReproducibilityReport:
    return ReproducibilityReport(
        schema_version="phase1-reproducibility-v1",
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
        provenance=_provenance(
            generation_mode="matched" if source_mode == "manifest" else "independent"
        ),
        passed=True,
    )


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
        ["--manifest", "validation.json", "--output", str(output)],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert json.loads(result.stderr)["code"] == "artifact_integrity_error"
    assert output.read_bytes() == b"different"
    assert output.stat().st_mtime_ns == before


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
        ["--manifest", "validation.json"],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "code": "artifact_integrity_error",
        "context": {},
        "message": "reproducibility mismatch",
    }
    assert "Traceback" not in result.stderr


def test_library_value_error_is_normalized_at_the_acceptance_script_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expected library gate refusals must not expose a traceback from the script."""
    monkeypatch.setattr(
        reproducibility_script,
        "check_reproducibility",
        lambda *_args: (_ for _ in ()).throw(ValueError("PRIVATE-GATE-DETAIL")),
    )

    result = CliRunner().invoke(
        reproducibility_script.app,
        ["--manifest", "validation.json"],
    )

    assert result.exit_code == 1
    assert result.stdout == ""
    assert json.loads(result.stderr) == {
        "code": "artifact_integrity_error",
        "context": {},
        "message": "reproducibility verification failed",
    }
    assert "PRIVATE-GATE-DETAIL" not in result.stderr
    assert "Traceback" not in result.stderr
