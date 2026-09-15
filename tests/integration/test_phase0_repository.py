import hashlib
import json
import re
import subprocess
from pathlib import Path
from runpy import run_path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_PHASE1_IN_PROGRESS_GATE = (
    "In progress: final Phase 1 evidence correction. The prior artifacts are "
    "invalid; the ordered Correction Pass 5, complete Task 18 regeneration, v3 "
    "cross-artifact verification, final audits, and local `make verify` must pass "
    "with zero foundation-model calls. Any eventual result is generator/oracle "
    "engineering evidence, not learned-model or benchmark evidence."
)
_PHASE1_EVIDENCE_RELATIVE_PATHS = {
    "validation": Path("manifests/validation/v1/ofd-primary-10000.json"),
    "oracle": Path("manifests/validation/v1/phase1-oracle-gate.json"),
    "leakage": Path("manifests/validation/v1/phase1-leakage-gate.json"),
    "matched_reproducibility": Path(
        "manifests/validation/v1/phase1-validation-reproducibility.json"
    ),
    "independent_reproducibility": Path(
        "manifests/validation/v1/phase1-independent-reproducibility-gate.json"
    ),
}
_PHASE1_COMPLETE_GATE = re.compile(
    r"Complete at collector source `(?P<source>[0-9a-f]{40})`: "
    r"validation `(?P<validation>[0-9a-f]{64})`; "
    r"oracle `(?P<oracle>[0-9a-f]{64})`; "
    r"leakage `(?P<leakage>[0-9a-f]{64})`; "
    r"matched reproducibility `(?P<matched_reproducibility>[0-9a-f]{64})`; "
    r"independent reproducibility "
    r"`(?P<independent_reproducibility>[0-9a-f]{64})`\. "
    r"The v3 cross-artifact verifier and local `make verify` passed with zero "
    r"foundation-model calls; this is generator/oracle engineering evidence, "
    r"not learned-model or benchmark evidence\."
)
_INVALIDATED_PHASE1_REVISIONS = {
    "bee142bd08a8b7b5621ae65551280ebdeed6c1b6",
    "4aa6eca5d25e3c6dac879850b2a0557bbe84b54e",
    "2203b68a4c6b67f26b9aae1bdc7d1ab00f8a6e6e",
}
# Copied from docs/PLAN.md at immutable evidence commit
# 84b1b87bb9194aeca68637967c4af3d3c5164168. The legacy helper independently
# hashes all five files and checks their common source against this exact cell.
_PHASE1_FROZEN_COMPLETE_GATE = (
    "Complete at collector source `7b5f33c20b6842fdbf19928f5032fa0d22ddbf14`: "
    "validation `84926a3217b27b04d7ed3e41038447533636aea06bcdc85b787f148f3d737e07`; "
    "oracle `2f5ffa58bde0e6690bc0d673b2f42a5beaa7aa650a077512a8e3d0578d56c66c`; "
    "leakage `2446db0d57b0641ec4988dd8dd1cd66e16f5b65ee4b13fbdb1769ee50834e72b`; "
    "matched reproducibility "
    "`4909d2981691605101c324c553c087623ec81193eda8e926e11b009bcae4460a`; "
    "independent reproducibility "
    "`3b50e4ad71646ec882288c039a3569ea60b5e6e1bd1ac0bd854a2c5eb9f8c37a`. "
    "The v3 cross-artifact verifier and local `make verify` passed with zero "
    "foundation-model calls; this is generator/oracle engineering evidence, "
    "not learned-model or benchmark evidence."
)

_PHASE2_IN_PROGRESS_GATE = (
    "In progress under the approved Phase 2 flow-and-event-engine plan. "
    "No Phase 2 acceptance artifact may exist until Tasks 1\u201313 are committed, "
    "reviewed, and locally verified."
)
_PHASE2_COMPLETE_GATE = re.compile(
    r"Complete at engine source `(?P<source>[0-9a-f]{40})` with gate artifact "
    r"`(?P<artifact>[0-9a-f]{64})`: the public-facts-only scripted EventFlow condition "
    r"achieved 10,000/10,000 timed successes; numeric, long-silence, Zeno, "
    r"checkpoint/resume, CPU replay, local `make verify`, and the independent artifact "
    r"verifier passed with zero foundation-model calls\. This is non-neural runtime "
    r"engineering evidence, not learned-model or benchmark evidence\."
)


def _assert_phase2_delivery_state(root: Path, plan_index: str) -> None:
    rows = [line for line in plan_index.splitlines() if line.startswith("| 2 —")]
    assert len(rows) == 1
    cell = rows[0].rsplit("|", maxsplit=2)[1].strip()
    path = root / "manifests/validation/v1/phase2-engine-gate.json"
    if not path.exists() and not path.is_symlink():
        assert cell == _PHASE2_IN_PROGRESS_GATE
        return
    assert path.is_file() and not path.is_symlink(), "Phase 2 evidence must be a regular file"
    match = _PHASE2_COMPLETE_GATE.fullmatch(cell)
    assert match is not None, "present Phase 2 evidence requires exact completion metadata"
    from silent_cascade.errors import SilentCascadeError

    verify = run_path(str(ROOT / "scripts/verify_phase2_gate_artifact.py"))[
        "verify_phase2_gate_artifact"
    ]
    try:
        result = verify(
            artifact_path=path, repo_root=root, expected_source_commit=match.group("source")
        )
    except SilentCascadeError as error:
        raise AssertionError("Phase 2 delivery artifact fails independent verification") from error
    assert result.passed and result.profile == "production"
    assert result.artifact_file_sha256 == match.group("artifact")
    assert result.source_commit == match.group("source")


def _assert_phase1_delivery_state(root: Path, plan_index: str) -> None:
    phase1_rows = [line for line in plan_index.splitlines() if line.startswith("| 1 —")]
    assert len(phase1_rows) == 1
    phase1_gate = phase1_rows[0].rsplit("|", maxsplit=2)[1].strip()
    evidence_paths = {
        label: root / relative_path
        for label, relative_path in _PHASE1_EVIDENCE_RELATIVE_PATHS.items()
    }
    evidence_presence = {
        label: path.exists() or path.is_symlink() for label, path in evidence_paths.items()
    }

    if not any(evidence_presence.values()):
        assert phase1_gate == _PHASE1_IN_PROGRESS_GATE
        assert not (_INVALIDATED_PHASE1_REVISIONS & set(re.findall(r"[0-9a-f]{40}", phase1_gate)))
        return

    assert all(evidence_presence.values()), "Phase 1 evidence must be wholly absent or present"
    assert all(path.is_file() and not path.is_symlink() for path in evidence_paths.values()), (
        "Phase 1 evidence paths must be regular files"
    )
    match = _PHASE1_COMPLETE_GATE.fullmatch(phase1_gate)
    assert match is not None, "present Phase 1 evidence requires exact completion metadata"
    expected_hashes = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in evidence_paths.items()
    }
    assert {key: value for key, value in match.groupdict().items() if key != "source"} == (
        expected_hashes
    )
    source_commits = set()
    for path in evidence_paths.values():
        artifact = json.loads(path.read_bytes())
        payload = artifact.get("payload", artifact)
        source_commits.add(payload["provenance"]["source_commit"])
    assert source_commits == {match.group("source")}
    assert match.group("source") not in _INVALIDATED_PHASE1_REVISIONS


def _assert_phase1_frozen_baseline(root: Path, plan_index: str) -> None:
    _assert_phase1_delivery_state(root, plan_index)
    row = next(line for line in plan_index.splitlines() if line.startswith("| 1 —"))
    cell = row.rsplit("|", maxsplit=2)[1].strip()
    assert cell == _PHASE1_FROZEN_COMPLETE_GATE, "completed Phase 1 baseline must remain frozen"


def test_phase0_repository_exposes_only_working_targets_and_commands() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([A-Za-z0-9][A-Za-z0-9_-]*):(?:\s|$)", makefile, re.MULTILINE))
    assert targets == {"setup", "doctor", "lint", "test", "smoke", "verify"}

    dry_run = subprocess.run(
        ["make", "-n", "verify"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    assert "uv run ruff check ." in dry_run.stdout
    regular_tests = (
        "uv run pytest -q --ignore=tests/integration/test_phase1_services.py "
        "--ignore=tests/unit/test_leakage.py"
    )
    service_tests = "uv run pytest -q tests/integration/test_phase1_services.py"
    leakage_tests = "uv run pytest -q tests/unit/test_leakage.py"
    assert regular_tests in dry_run.stdout
    assert service_tests in dry_run.stdout
    assert leakage_tests in dry_run.stdout
    assert dry_run.stdout.index(regular_tests) < dry_run.stdout.index(service_tests)
    assert dry_run.stdout.index(service_tests) < dry_run.stdout.index(leakage_tests)
    assert "uv run silent-cascade doctor" in dry_run.stdout
    assert "uv build" in dry_run.stdout
    assert "check_phase2_engine.py" not in dry_run.stdout

    smoke = subprocess.run(
        ["make", "-n", "smoke"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert smoke.returncode == 0, smoke.stderr
    assert "tests/integration/test_cli_phase1.py" in smoke.stdout
    assert "tests/integration/test_phase1_gate_verifier.py" in smoke.stdout
    assert "tests/regression/test_phase1_fixtures.py" in smoke.stdout
    assert "tests/regression/test_import_boundaries.py" in smoke.stdout
    assert "tests/integration/test_cli_replay.py" in smoke.stdout
    assert "tests/neural/test_training_cli.py" in smoke.stdout
    assert "tests/neural/test_training_imports.py" in smoke.stdout

    workflows = ROOT / ".github" / "workflows"
    assert not workflows.exists() or not any(workflows.iterdir())

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "reports no learned benchmark results" in readme
    assert "silent-cascade doctor" in readme
    assert "silent-cascade replay" in readme
    assert "non-neural runtime engineering evidence" in readme
    assert "silent-cascade data freeze" in readme
    assert "silent-cascade episode inspect" in readme
    assert "silent-cascade oracle evaluate" in readme
    assert "silent-cascade leakage audit" in readme
    for forbidden in (
        "became conscious",
        "proved an inner life",
        "thinks like a brain",
    ):
        assert forbidden not in readme.lower()


def test_phase0_plan_index_is_wired_to_frozen_inputs() -> None:
    plan_index = (ROOT / "docs/PLAN.md").read_text(encoding="utf-8")
    assert "superpowers/specs/2026-08-30-silent-cascade-design.md" in plan_index
    assert "superpowers/plans/2026-08-30-phase-0-bootstrap.md" in plan_index
    assert "superpowers/plans/2026-08-30-phase-1-generator-oracle.md" in plan_index
    _assert_phase1_frozen_baseline(ROOT, plan_index)
    assert "_assert_phase2_delivery_state" in globals(), "Phase 2 two-state regression is missing"
    _assert_phase2_delivery_state(ROOT, plan_index)


def test_phase2_delivery_state_refuses_partial_or_unearned_completion(tmp_path: Path) -> None:
    assert "_assert_phase2_delivery_state" in globals(), "Phase 2 two-state regression is missing"
    in_progress = (
        "In progress under the approved Phase 2 flow-and-event-engine plan. "
        "No Phase 2 acceptance artifact may exist until Tasks 1\u201313 are committed, "
        "reviewed, and locally verified."
    )
    plan = f"| 2 — Flow and event engine | plan | {in_progress} |"
    _assert_phase2_delivery_state(tmp_path, plan)
    with pytest.raises(AssertionError):
        _assert_phase2_delivery_state(tmp_path, plan.replace(in_progress, "Complete"))
    path = tmp_path / "manifests/validation/v1/phase2-engine-gate.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    with pytest.raises(AssertionError):
        _assert_phase2_delivery_state(tmp_path, plan)
    with pytest.raises(AssertionError):
        _assert_phase2_delivery_state(tmp_path, plan.replace(in_progress, "Complete"))
    path.unlink()
    path.symlink_to(tmp_path / "missing.json")
    with pytest.raises(AssertionError):
        _assert_phase2_delivery_state(tmp_path, plan)


def test_phase1_delivery_state_contract(tmp_path: Path) -> None:
    source_commit = "a" * 40
    evidence_paths = {
        label: tmp_path / relative_path
        for label, relative_path in _PHASE1_EVIDENCE_RELATIVE_PATHS.items()
    }
    for label, path in evidence_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"artifact": label, "provenance": {"source_commit": source_commit}}
        path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
    evidence_hashes = {
        label: hashlib.sha256(path.read_bytes()).hexdigest()
        for label, path in evidence_paths.items()
    }
    complete_gate = (
        f"Complete at collector source `{source_commit}`: "
        f"validation `{evidence_hashes['validation']}`; "
        f"oracle `{evidence_hashes['oracle']}`; "
        f"leakage `{evidence_hashes['leakage']}`; "
        "matched reproducibility "
        f"`{evidence_hashes['matched_reproducibility']}`; "
        "independent reproducibility "
        f"`{evidence_hashes['independent_reproducibility']}`. "
        "The v3 cross-artifact verifier and local `make verify` passed with zero "
        "foundation-model calls; this is generator/oracle engineering evidence, "
        "not learned-model or benchmark evidence."
    )
    complete_plan = f"| 1 — Generator and oracle | plan | {complete_gate} |"
    _assert_phase1_delivery_state(tmp_path, complete_plan)

    evidence_paths["oracle"].write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError):
        _assert_phase1_delivery_state(tmp_path, complete_plan)

    evidence_paths["oracle"].unlink()
    with pytest.raises(AssertionError, match="wholly absent or present"):
        _assert_phase1_delivery_state(tmp_path, complete_plan)

    for path in evidence_paths.values():
        path.unlink(missing_ok=True)
    in_progress_plan = "| 1 — Generator and oracle | plan | " + _PHASE1_IN_PROGRESS_GATE + " |"
    _assert_phase1_delivery_state(tmp_path, in_progress_plan)
    with pytest.raises(AssertionError):
        _assert_phase1_delivery_state(tmp_path, complete_plan)


def _trusted_phase1_plan_index() -> str:
    return subprocess.run(
        ["git", "show", "84b1b87bb9194aeca68637967c4af3d3c5164168:docs/PLAN.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_phase2_frozen_phase1_baseline_accepts_the_completed_historical_evidence() -> None:
    trusted = _trusted_phase1_plan_index()
    _assert_phase1_frozen_baseline(ROOT, trusted)
    current = (ROOT / "docs/PLAN.md").read_text(encoding="utf-8")
    assert next(line for line in current.splitlines() if line.startswith("| 1 —")) == next(
        line for line in trusted.splitlines() if line.startswith("| 1 —")
    )


def test_phase2_frozen_phase1_baseline_rejects_deletion_and_historical_in_progress(
    tmp_path: Path,
) -> None:
    index = "| 1 — Generator and oracle | plan | " + _PHASE1_IN_PROGRESS_GATE + " |"
    _assert_phase1_delivery_state(tmp_path, index)  # Still valid for the historical helper.
    with pytest.raises(AssertionError):
        _assert_phase1_frozen_baseline(tmp_path, index)


@pytest.mark.parametrize("alter_source", [False, True])
def test_phase2_frozen_phase1_baseline_rejects_coherently_rehashed_replacements(
    tmp_path: Path,
    alter_source: bool,
) -> None:
    index = _trusted_phase1_plan_index()
    cell = (
        next(line for line in index.splitlines() if line.startswith("| 1 —"))
        .rsplit("|", maxsplit=2)[1]
        .strip()
    )
    metadata = _PHASE1_COMPLETE_GATE.fullmatch(cell).groupdict()
    source = "a" * 40 if alter_source else metadata["source"]
    rewritten_cell = cell.replace(metadata["source"], source)
    for label, relative in _PHASE1_EVIDENCE_RELATIVE_PATHS.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"replacement": label, "provenance": {"source_commit": source}},
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        rewritten_cell = rewritten_cell.replace(
            metadata[label], hashlib.sha256(path.read_bytes()).hexdigest()
        )
    rewritten_index = index.replace(cell, rewritten_cell)
    _assert_phase1_delivery_state(tmp_path, rewritten_index)
    with pytest.raises(AssertionError):
        _assert_phase1_frozen_baseline(tmp_path, rewritten_index)


_PHASE2_FROZEN_SOURCE = "33fb8ed105046cb4fdb7be61eac50ed391415b7c"
_PHASE2_FROZEN_ARTIFACT = "c05b2f78680b14e5b8fd81597489f34ec68a1dc2d1114e62daa95d7428694c42"
_PHASE3_COMPLETE_GATE = re.compile(
    r"Complete at component source `(?P<source>[0-9a-f]{40})` with gate artifact "
    r"`(?P<artifact>[0-9a-f]{64})` and selected weights `(?P<weights>[0-9a-f]{64})`: "
    r"source-bound independent verification and local `make verify` passed with zero "
    r"foundation-model calls\. This is untimed neural component engineering evidence, "
    r"not autonomous cascade or benchmark evidence\."
)


def _assert_phase2_frozen_baseline(root, plan_index):
    rows = [line for line in plan_index.splitlines() if line.startswith("| 2 —")]
    assert len(rows) == 1
    cell = rows[0].rsplit("|", maxsplit=2)[1].strip()
    match = _PHASE2_COMPLETE_GATE.fullmatch(cell)
    assert match is not None
    assert match.group("source") == _PHASE2_FROZEN_SOURCE
    assert match.group("artifact") == _PHASE2_FROZEN_ARTIFACT
    path = root / "manifests/validation/v1/phase2-engine-gate.json"
    assert path.is_file() and not path.is_symlink()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _PHASE2_FROZEN_ARTIFACT
    _assert_phase2_delivery_state(root, plan_index)


def _assert_phase3_delivery_state(root, plan_index):
    rows = [line for line in plan_index.splitlines() if line.startswith("| 3 —")]
    assert len(rows) == 1
    cell = rows[0].rsplit("|", maxsplit=2)[1].strip()
    path = root / "manifests/validation/phase3/component-gate.json"
    if not path.exists() and not path.is_symlink():
        assert re.match(r"^(?:Proposed|In progress)(?:[;:.]|$)", cell)
        return
    assert path.is_file() and not path.is_symlink()
    match = _PHASE3_COMPLETE_GATE.fullmatch(cell)
    assert match is not None, "present Phase 3 gate requires exact completion metadata"
    from silent_cascade.errors import SilentCascadeError

    verify = run_path(str(ROOT / "scripts/verify_phase3_gate_artifact.py"))[
        "verify_phase3_gate_artifact"
    ]
    try:
        result = verify(
            artifact_path=path,
            manifest_path=root / "manifests/validation/phase3/one-hop-10000.json",
            repo_root=root,
            expected_source_commit=match.group("source"),
        )
    except SilentCascadeError as error:
        raise AssertionError("Phase 3 gate fails independent verification") from error
    assert result.valid and result.passed and result.publication == "production"
    assert result.artifact_file_sha256 == match.group("artifact")
    assert result.checkpoint_sha256 == match.group("weights")


def test_phase3_repository_delivery_preserves_completed_prior_phases():
    index = (ROOT / "docs/PLAN.md").read_text()
    assert "_assert_phase2_frozen_baseline" in globals(), "Phase 2 evidence pin is missing"
    assert "_assert_phase3_delivery_state" in globals(), "Phase 3 delivery check is missing"
    _assert_phase2_frozen_baseline(ROOT, index)
    _assert_phase3_delivery_state(ROOT, index)


def test_phase3_corpus_is_not_completion_and_present_gate_cannot_be_unearned(tmp_path):
    assert "_assert_phase3_delivery_state" in globals(), "Phase 3 delivery check is missing"
    index = "| 3 — Neural components | plan | In progress; corpus only. |"
    directory = tmp_path / "manifests/validation/phase3"
    directory.mkdir(parents=True)
    (directory / "one-hop-10000.json").write_text("{}")
    _assert_phase3_delivery_state(tmp_path, index)
    with pytest.raises(AssertionError):
        _assert_phase3_delivery_state(
            tmp_path, index.replace("In progress; corpus only.", "Complete.")
        )
    (directory / "component-gate.json").write_text("{}")
    with pytest.raises(AssertionError):
        _assert_phase3_delivery_state(tmp_path, index)


def test_phase3_pin_rejects_coordinated_phase2_substitution(tmp_path):
    assert "_assert_phase2_frozen_baseline" in globals(), "Phase 2 evidence pin is missing"
    index = (ROOT / "docs/PLAN.md").read_text()
    artifact = tmp_path / "manifests/validation/v1/phase2-engine-gate.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"coordinated":"replacement"}')
    old_hash = "c05b2f78680b14e5b8fd81597489f34ec68a1dc2d1114e62daa95d7428694c42"
    rewritten = index.replace(old_hash, hashlib.sha256(artifact.read_bytes()).hexdigest())
    with pytest.raises(AssertionError):
        _assert_phase2_frozen_baseline(tmp_path, rewritten)
