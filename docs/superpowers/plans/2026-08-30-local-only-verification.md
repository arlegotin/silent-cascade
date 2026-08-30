# Local-Only Verification Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove CI/CD from Silent Cascade and make every quality, test, doctor, and package-build gate explicitly local-only.

**Architecture:** Delete the hosted workflow and replace the public `ci` Make target with a local `verify` target. Enforce that repository shape with an integration test, then align agent governance, public documentation, the canonical design, the Phase 0 plan, and the deviations log with the permanent local-only policy.

**Tech Stack:** GNU Make, Python 3.12, pytest, Ruff, `uv`, Git, Markdown, GitHub repository layout.

**Spec:** `docs/superpowers/specs/2026-08-30-silent-cascade-design.md`

## Global Constraints

- Work on the current branch and make meaningful commits.
- Do not add, retain, or invoke hosted CI/CD workflows, deployment pipelines, release pipelines, or third-party automation.
- All linting, formatting, tests, environment diagnostics, and package builds run locally.
- Preserve the offline primary path and exactly zero foundation-model calls.
- Do not weaken any scientific control, metric, acceptance gate, or claim boundary.
- Keep the only implemented CLI command as `silent-cascade doctor`.
- Use test-driven development for the repository automation change.
- Update canonical plan version `1.0.1` to `1.0.2` and record the approved policy change in `docs/deviations.md`.

---

### Task 1: Replace Hosted CI/CD with a Local Verification Gate

**Files:**

- Modify: `tests/integration/test_phase0_repository.py`
- Modify: `Makefile`
- Delete: `.github/workflows/ci.yml`
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/PLAN.md`
- Modify: `docs/deviations.md`
- Modify: `docs/superpowers/specs/2026-08-30-silent-cascade-design.md`
- Modify: `docs/superpowers/plans/2026-08-30-local-only-verification.md`
- Modify: `docs/superpowers/plans/2026-08-30-phase-0-bootstrap.md`

**Interfaces:**

- Consumes: existing local `lint`, `test`, `doctor`, and `uv build` commands.
- Produces: `make verify`, which runs lint, tests, doctor, and the package build locally; a repository with no hosted workflow; and a permanent agent/documentation policy forbidding CI/CD until the user explicitly reverses it.

- [ ] **Step 1: Change the repository contract test first**

Replace the workflow-positive assertions with local-only behavior checks. The resulting test module must retain the existing README/plan-link checks and include this executable Make assertion:

```python
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_phase0_repository_exposes_only_working_targets_and_commands() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(
        re.findall(r"^([A-Za-z0-9][A-Za-z0-9_-]*):(?:\s|$)", makefile, re.MULTILINE)
    )
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
    assert "uv run pytest -q" in dry_run.stdout
    assert "uv run silent-cascade doctor" in dry_run.stdout
    assert "uv build" in dry_run.stdout

    workflows = ROOT / ".github" / "workflows"
    assert not workflows.exists() or not any(workflows.iterdir())

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "reports no benchmark results" in readme
    assert "silent-cascade doctor" in readme
    for forbidden in (
        "became conscious",
        "proved an inner life",
        "thinks like a brain",
    ):
        assert forbidden not in readme.lower()
```

Rename the second test to `test_phase0_plan_index_is_wired_to_frozen_inputs` and leave only the two plan-link assertions; prose is reviewed directly rather than tested by string matching.

- [ ] **Step 2: Run the repository test and observe RED**

Run:

```bash
env UV_CACHE_DIR=/tmp/silent-cascade-uv-cache \
  uv run pytest tests/integration/test_phase0_repository.py -q
```

Expected: failure because the target set still contains `ci`, `make verify` does not exist, and `.github/workflows/ci.yml` remains present.

- [ ] **Step 3: Implement the local-only Make surface and remove the workflow**

Replace `Makefile` with:

```make
.PHONY: setup doctor lint test smoke verify

setup:
	uv sync --locked --group dev

doctor:
	uv run silent-cascade doctor

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

smoke:
	uv run pytest -q tests/integration/test_cli_doctor.py tests/regression/test_import_boundaries.py

verify: lint test doctor
	uv build
```

Delete `.github/workflows/ci.yml`. Do not replace it with another workflow, hook, scheduled job, deployment file, or hosted service.

- [ ] **Step 4: Make the agent policy permanent**

Add this section to `AGENTS.md` immediately before `## Silent Cascade contract`:

```markdown
## Local-only verification policy

- Do not add or restore CI/CD workflows, hosted build/test automation,
  deployment pipelines, release pipelines, or third-party automation unless the
  user explicitly reverses this policy.
- Run linting, formatting, tests, doctor diagnostics, and package builds only
  in the local workspace. Use `make verify` as the complete local quality gate.
- Do not describe a hosted check as required or available when documenting the
  project. Keep repository instructions and phase plans aligned with this
  local-only policy.
```

- [ ] **Step 5: Update public and canonical documentation**

Apply all of these exact policy changes:

1. `README.md`: replace `make ci` with `make verify`; state that no CI/CD is configured and all checks/builds run locally.
2. `docs/PLAN.md`: make the Phase 0 gate `uv sync --locked --group dev`, `make verify`, and `uv run silent-cascade doctor`.
3. `docs/deviations.md`: add a `2026-08-30 — Plan version 1.0.2` entry recording the user-approved removal of CI/CD and the local `make verify` replacement.
4. Canonical design: bump `Plan version` to `1.0.2`; add a non-negotiable local-only automation rule; remove `.github/workflows/ci.yml` from the tree; replace every active CI/hosted-workflow requirement with local verification; rename the CI test-plan subsection to `Local verification`; remove the CI badge allowance.
5. Phase 0 plan: replace CPU/Linux CI goals, file-map entries, coverage evidence, Task 8 workflow steps, commands, and completion gates with local verification; change the Task 8 commit example to `chore: add local Phase 0 quality gate`.

After editing, this command must print nothing across the maintained product and
Phase 0 documents. This migration plan is intentionally excluded because it
records the removed path and the RED state that preceded the change.

```bash
rg -n -i \
  'make ci|\.github/workflows|linux cpu ci|hosted ci|core ci|cpu ci|github actions' \
  AGENTS.md README.md Makefile docs/PLAN.md docs/deviations.md \
  docs/superpowers/specs/2026-08-30-silent-cascade-design.md \
  docs/superpowers/plans/2026-08-30-phase-0-bootstrap.md tests
```

Explicit statements that CI/CD is disabled are allowed and required.

- [ ] **Step 6: Run focused GREEN checks**

Run:

```bash
env UV_CACHE_DIR=/tmp/silent-cascade-uv-cache \
  uv run pytest tests/integration/test_phase0_repository.py -q
env UV_CACHE_DIR=/tmp/silent-cascade-uv-cache uv run ruff check .
env UV_CACHE_DIR=/tmp/silent-cascade-uv-cache uv run ruff format --check .
```

Expected: repository contract passes and Ruff is clean.

- [ ] **Step 7: Run the complete local gate**

Run locally:

```bash
env UV_CACHE_DIR=/tmp/silent-cascade-uv-cache uv sync --locked --group dev
env UV_CACHE_DIR=/tmp/silent-cascade-uv-cache make verify
env UV_CACHE_DIR=/tmp/silent-cascade-uv-cache \
  uv run pytest tests/regression/test_import_boundaries.py \
  tests/integration/test_sdist_contents.py -q
git diff --check
git status --short
```

Expected: locked sync, lint, formatting, all tests, doctor, package build, import isolation, and sdist-content checks pass locally; the final worktree contains only the intended unstaged implementation before commit.

- [ ] **Step 8: Commit the permanent policy change**

```bash
git add AGENTS.md Makefile README.md docs/PLAN.md docs/deviations.md \
  docs/superpowers/specs/2026-08-30-silent-cascade-design.md \
  docs/superpowers/plans/2026-08-30-local-only-verification.md \
  docs/superpowers/plans/2026-08-30-phase-0-bootstrap.md \
  tests/integration/test_phase0_repository.py
git add -u .github/workflows/ci.yml
git commit -m "chore: make project verification local-only"
```

Verify the committed state with `make verify` and `git status --short`.
