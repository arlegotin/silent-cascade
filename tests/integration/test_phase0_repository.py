import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


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
    regular_tests = "uv run pytest -q --ignore=tests/unit/test_leakage.py"
    leakage_tests = "uv run pytest -q tests/unit/test_leakage.py"
    assert regular_tests in dry_run.stdout
    assert leakage_tests in dry_run.stdout
    assert dry_run.stdout.index(regular_tests) < dry_run.stdout.index(leakage_tests)
    assert "uv run silent-cascade doctor" in dry_run.stdout
    assert "uv build" in dry_run.stdout

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

    workflows = ROOT / ".github" / "workflows"
    assert not workflows.exists() or not any(workflows.iterdir())

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "reports no learned benchmark results" in readme
    assert "silent-cascade doctor" in readme
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
    assert "100,000 independent-recipe" in plan_index
