import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_phase0_repository_exposes_only_working_targets_and_commands() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([A-Za-z0-9][A-Za-z0-9_-]*):(?:\s|$)", makefile, re.MULTILINE))
    assert targets == {"setup", "doctor", "lint", "test", "smoke", "ci"}

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "reports no benchmark results" in readme
    assert "silent-cascade doctor" in readme
    for forbidden in (
        "became conscious",
        "proved an inner life",
        "thinks like a brain",
    ):
        assert forbidden not in readme.lower()


def test_phase0_plan_index_and_cpu_ci_are_wired_to_frozen_inputs() -> None:
    plan_index = (ROOT / "docs/PLAN.md").read_text(encoding="utf-8")
    assert "superpowers/specs/2026-08-30-silent-cascade-design.md" in plan_index
    assert "superpowers/plans/2026-08-30-phase-0-bootstrap.md" in plan_index

    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "uv sync --locked --group dev" in workflow
    assert "make ci" in workflow
    assert "uv build" in workflow
    assert "--extra qwen" not in workflow
