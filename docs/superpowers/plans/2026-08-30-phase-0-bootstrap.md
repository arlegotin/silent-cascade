# Silent Cascade Phase 0 Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the reproducible, offline-safe Python foundation required to start Silent Cascade: a locked package, strict configuration and hashing, durable artifact primitives, deterministic RNG state, crash bundles, a real environment doctor, and CPU CI.

**Architecture:** Keep `silent_cascade` shallow and dependency-directed: shared validation and typed errors at the bottom, atomic I/O and hashing above them, then configuration/RNG/crash services, with Typer as a thin adapter over a testable doctor service. Phase 0 exposes only the working `doctor` command; experiment data, hybrid dynamics, models, training, evaluation, and reporting stay outside this plan.

**Tech Stack:** Python 3.12, `uv`, PyTorch float32 on CPU with local Apple MPS capability and numeric checks, NumPy, Pydantic v2, PyYAML, Typer, Rich, psutil, pytest, Hypothesis, pytest-cov, and Ruff.

**Spec:** `docs/superpowers/specs/2026-08-30-silent-cascade-design.md`

**Plan status:** **Draft — execution may begin only after the user explicitly approves this exact plan in this task thread; choosing an execution mode is not approval.** Written at the user's request on 2026-08-30.

## Execution Preconditions and Superpowers Workflow

Before changing any source, test, build, configuration, workflow, or release file:

1. Confirm that the user explicitly approved this exact plan in the task thread.
2. Invoke `superpowers:using-superpowers`, then invoke `superpowers:using-git-worktrees` and follow it to establish an isolated implementation workspace.
3. Invoke the approved execution skill: `superpowers:subagent-driven-development` for same-session task dispatch, or `superpowers:executing-plans` for checkpointed batch execution.
4. Invoke `superpowers:test-driven-development` for every task and preserve each stated red/green sequence.
5. On any failure or unexpected behavior—not only dependency resolution failures—stop implementation and invoke `superpowers:systematic-debugging` before proposing or applying a fix.
6. Use `superpowers:requesting-code-review` at the review checkpoints required by the selected execution skill, and invoke `superpowers:receiving-code-review` before applying review feedback.
7. Before any completion claim, invoke `superpowers:verification-before-completion` and run the fresh verification command specified by this plan. After every task and final review is complete, invoke `superpowers:finishing-a-development-branch` to present the integration options.

The implementation worker must not infer approval from this document's existence, its Git commit, or a choice of execution mode.

## Global Constraints

- The distribution name is `silent-cascade`, the import package is `silent_cascade`, and the executable is `silent-cascade`.
- Require Python `>=3.12,<3.13`; `.python-version` contains exactly `3.12`.
- Use the dependency ranges frozen in spec Section 13.4. Do not loosen a range after a resolution failure without explicit user approval and a concrete entry in `docs/deviations.md`.
- Generate and commit a universal `uv.lock`; every verification after locking uses `uv sync --locked --group dev`.
- Core imports and CPU CI must work without the `qwen` extra and must not import `mlx`, `mlx_vlm`, or `huggingface_hub`.
- Primary scientific execution remains offline with exactly zero foundation-model calls. Phase 0 performs no model download and no network call at runtime.
- Do not set or enable `PYTORCH_ENABLE_MPS_FALLBACK`; report a truthy inherited value as a doctor failure.
- CPU is mandatory. Missing MPS is a reported capability state, not a failure. MPS smoke work runs only when both built and available.
- Keep tensors float32. Store timestamps and host-side comparisons as Python/NumPy float64 when those concepts enter future phases.
- Enforce the hard ceilings in the bootstrap configuration: 5,000,000 total trainable parameters, 64 primary records, 64 EventFlow events, 25,000 fixed-grid opportunities, batch size 128, 500 retained full traces plus failures, zero primary foundation calls, and one foundation worker.
- Load YAML into strict Pydantic models, reject unknown keys, merge `base -> selected layer -> explicit --set overrides`, and hash canonical sorted compact JSON including defaults.
- Use no Hydra, plugin framework, service, database, telemetry, generic agent runtime, shell integration, or hidden natural-language reasoning.
- Do not add placeholder CLI commands or Make targets. Phase 0 exposes only behavior implemented and tested in this plan.
- Do not claim benchmark results, consciousness, sentience, biological fidelity, or general intelligence.
- Use Apache-2.0. Do not commit caches, virtual environments, model weights, normal checkpoints, raw run artifacts, or machine-specific paths.

## Phase 0 File Map

| Path | Responsibility |
|---|---|
| `.python-version` | Select Python 3.12 for `uv`. |
| `.gitignore` | Exclude local environments, caches, generated runs/reports/assets, checkpoints, and model weights. |
| `LICENSE` | Unmodified Apache License 2.0 text. |
| `pyproject.toml` | Package metadata, frozen dependency ranges, console script, build backend, Ruff/pytest/coverage settings. |
| `uv.lock` | Universal authoritative dependency lock generated by `uv lock`. |
| `README.md` | Conservative Phase 0 project status, claim boundary, and only commands that work. |
| `Makefile` | Idempotent working `setup`, `doctor`, `lint`, `test`, `smoke`, and `ci` targets. |
| `.github/workflows/ci.yml` | Linux CPU locked install, lint, tests, package build, and doctor. |
| `configs/base.yaml` | Strict bootstrap defaults and hard ceilings. |
| `docs/PLAN.md` | Navigation index to the canonical spec and phase-scoped plans. |
| `docs/deviations.md` | Append-only record of approved departures from frozen requirements; initially records none. |
| `docs/research-boundary.md` | Public claim and scope boundary for the scientific project. |
| `src/silent_cascade/__init__.py` | Package version only; no eager subsystem imports. |
| `src/silent_cascade/validation.py` | Shared strict Pydantic base and recursive JSON value types. |
| `src/silent_cascade/errors.py` | Stable typed error hierarchy and serializable error payloads. |
| `src/silent_cascade/hashing.py` | Canonical JSON and SHA-256 byte/file helpers. |
| `src/silent_cascade/io.py` | Durable atomic replace and race-safe no-clobber file creation. |
| `src/silent_cascade/config.py` | Bootstrap schema, YAML layering, dotted overrides, canonical resolved configuration. |
| `src/silent_cascade/rng.py` | Python/NumPy/Torch CPU and optional MPS RNG capture, restore, seeding, and round-trip check. |
| `src/silent_cascade/logging/__init__.py` | Logging package boundary without side effects. |
| `src/silent_cascade/logging/crash_bundle.py` | Strict crash manifest and atomic bundle publication. |
| `src/silent_cascade/doctor.py` | Testable environment/capability diagnostics and isolated continuous-time numeric smoke. |
| `src/silent_cascade/cli.py` | Thin Typer adapter exposing `doctor`. |
| `tests/unit/` | Focused unit tests for every Phase 0 service. |
| `tests/integration/test_cli_doctor.py` | Installed console-script and JSON report smoke. |
| `tests/integration/test_phase0_repository.py` | Repository docs, Make targets, and CI boundary contract. |
| `tests/regression/test_import_boundaries.py` | Fresh-process proof that core imports do not load optional MLX packages. |

## Dependency Direction

```text
validation
    └── errors
        ├── hashing
        └── io
             └── crash_bundle

validation + errors + hashing
    └── config

validation + errors
    └── rng

config + rng + io
    └── doctor
         └── cli
```

`silent_cascade.__init__` imports none of these modules eagerly. `cli` may import `doctor`; no core module imports `foundation` or an optional Qwen dependency.

## Explicitly Deferred Scope

- **Phase 1:** OFD schemas, generator, oracle, scoring, leakage audit, immutable manifests, data configs, and their CLI commands.
- **Phase 2:** production flow/guard modules, state containers, event queue, tie semantics, jumps, invariants, checkpoint/replay, and the `replay` command.
- **Phases 3–4:** memory, neural models, training/checkpoints, run directories, autonomous EventFlow, traces, and pilot execution.
- **Phase 5:** strong baselines, scheduled controls, interventions, and compute accounting.
- **Phase 6:** frozen tests, full evaluation, aggregation, statistics, reports, figures, experiment card, and release tags.
- **Phase 7:** MLX/Qwen bridge, model revision, demos, public assets, and optional citation metadata.

The Phase 0 doctor performs an isolated, deterministic evaluation of the exact exponential-flow and analytic guard formulas to prove that the installed CPU/MPS numeric stack supports them. Phase 2 replaces that isolated probe with calls to the production flow/guard primitives when those modules exist; Phase 0 does not create an event engine or runtime state.

## Phase 0 Coverage Matrix

| Spec requirement | Owning task(s) | Phase 0 evidence |
|---|---|---|
| §0 fail-loud behavior and no unfinished command surfaces | 2, 6, 7, 8 | Typed errors, crash manifest, real doctor, repository-contract test |
| §13.2–13.4 CPU/MPS split, Python 3.12, frozen dependencies, universal lock | 1, 5, 7, 8 | Locked sync, RNG/device report, Linux CPU CI, local MPS check |
| §13.6 hard ceilings and zero primary foundation calls | 4, 7 | Strict bounded defaults, doctor configuration hash/report |
| §14 repository and import boundaries | 1–8 | Focused package layout and fresh-process optional-import regression |
| §15.1 strict layered config and canonical hash | 4 | Unknown-key, precedence, override, default, and hash tests |
| §15.2 real `doctor` with versions, devices, paths, RNG, flow/guard smoke | 5, 7 | Service, CLI JSON schema, installed console-script test |
| §15.3 idempotent working Make targets | 8 | Repository-contract test and `make ci` |
| §16.1 atomic completion and artifact hashes | 3, 6 | Replace/no-clobber tests, SHA-256 verification, crash publication |
| §17 Phase 0 gate | 8 | Locked sync, `make ci`, doctor, import boundary, wheel build |
| §18 config/import/integration tests and Linux CPU CI | 2–8 | Unit, integration, regression suites and GitHub workflow |
| §20 conservative public communication and Apache-2.0 | 1, 8 | Exact license, README, research boundary |
| §22–23 clean checkout, no MLX core import, reproducible bootstrap | 1, 7, 8 | Lockfile, clean import subprocess, final committed-state gate |

Self-review found no Phase 0 requirement without an owning task. Requirements assigned to Phases 1–7 are listed in Explicitly Deferred Scope and receive no placeholder implementation here.

---

### Task 1: Reproducible Python Package Skeleton

**Files:**

- Create: `.python-version`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `uv.lock` (generated)
- Create: `docs/deviations.md`
- Create: `src/silent_cascade/__init__.py`
- Create: `tests/unit/test_package.py`

**Interfaces:**

- Consumes: the approved spec and this plan only.
- Produces: installable distribution `silent-cascade==0.1.0`, import package `silent_cascade`, `silent_cascade.__version__: str`, and the locked Python environment used by every remaining task.

- [ ] **Step 1: Create the packaging and repository metadata**

Create `.python-version`:

```text
3.12
```

Create `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling>=1.27,<2"]
build-backend = "hatchling.build"

[project]
name = "silent-cascade"
version = "0.1.0"
description = "Persistent event-flow AI research on the Observation-Free Deadline benchmark"
readme = "README.md"
requires-python = ">=3.12,<3.13"
license = "Apache-2.0"
dependencies = [
  "torch>=2.13,<2.14",
  "numpy>=2.2,<3",
  "pydantic>=2.11,<3",
  "pyyaml>=6,<7",
  "typer>=0.16,<1",
  "rich>=14,<15",
  "safetensors>=0.5,<1",
  "scipy>=1.15,<2",
  "matplotlib>=3.10,<4",
  "psutil>=7,<8",
]

[dependency-groups]
dev = [
  "pytest>=8.4,<9",
  "hypothesis>=6.130,<7",
  "pytest-cov>=6,<7",
  "ruff>=0.12,<1",
]

[project.optional-dependencies]
qwen = [
  "mlx>=0.32,<0.33",
  "mlx-vlm>=0.6.17,<0.7",
  "huggingface-hub>=1,<2",
]

[tool.hatch.build.targets.wheel]
packages = ["src/silent_cascade"]

[tool.pytest.ini_options]
minversion = "8.4"
testpaths = ["tests"]
addopts = ["--strict-config", "--strict-markers"]

[tool.coverage.run]
branch = true
source = ["silent_cascade"]

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["B", "E", "F", "I", "RUF", "SIM", "UP"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

Create a deliberately shallow `src/silent_cascade/__init__.py` containing only a package docstring. Create `README.md` with this truthful initial content:

```markdown
# Silent Cascade

Silent Cascade is a public research project testing persistent, learned event-flow computation on the Observation-Free Deadline benchmark.

The repository is in Phase 0 bootstrap. It reports no benchmark results yet. The canonical design is [`docs/superpowers/specs/2026-08-30-silent-cascade-design.md`](docs/superpowers/specs/2026-08-30-silent-cascade-design.md).

The project tests a bounded computational mechanism. It makes no claim about consciousness, sentience, biological fidelity, or general intelligence.
```

Create `docs/deviations.md`:

```markdown
# Approved Deviations

This append-only log records necessary, explicitly approved departures from the canonical Silent Cascade design.

## Current status

No deviations are recorded for plan version 1.0.
```

Create `.gitignore` with, at minimum:

```gitignore
.DS_Store
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.hypothesis/
.coverage
htmlcov/
dist/
build/
*.egg-info/
runs/
reports/
assets/
*.safetensors
*.ckpt
*.pt
*.pth
*.mlx
```

Copy Appendix A of this plan byte-for-byte into `LICENSE`. Do not abbreviate it or add project-specific restrictions.

- [ ] **Step 2: Write the failing package metadata test**

Create `tests/unit/test_package.py`:

```python
from importlib.metadata import version

from silent_cascade import __version__


def test_public_version_matches_distribution_metadata() -> None:
    assert __version__ == version("silent-cascade") == "0.1.0"
```

- [ ] **Step 3: Resolve and install the frozen dependency graph**

Run:

```bash
uv lock
uv sync --locked --group dev
uv run python -c 'import sys; assert sys.version_info[:2] == (3, 12)'
```

Expected: all commands exit 0; `uv.lock` is generated; the interpreter assertion prints nothing.

If `uv lock` cannot resolve the frozen ranges for Python 3.12 on the supported platforms, stop this task and invoke `superpowers:systematic-debugging`. Preserve the resolver output. Do not change a range until the user approves the exact old range, new range, incompatibility, and deviation entry.

- [ ] **Step 4: Run the package test to verify it fails**

Run:

```bash
uv run pytest tests/unit/test_package.py -q
```

Expected: collection fails with `ImportError: cannot import name '__version__' from 'silent_cascade'`.

- [ ] **Step 5: Implement the package version without eager imports**

Replace `src/silent_cascade/__init__.py` with:

```python
"""Silent Cascade research package."""

from importlib.metadata import version

__version__ = version("silent-cascade")

__all__ = ["__version__"]
```

- [ ] **Step 6: Verify the package and wheel**

Run:

```bash
uv run pytest tests/unit/test_package.py -q
uv build
```

Expected: `1 passed`; both sdist and wheel are created under ignored `dist/`.

- [ ] **Step 7: Commit the package skeleton**

```bash
git add .python-version .gitignore LICENSE README.md pyproject.toml uv.lock \
  docs/deviations.md src/silent_cascade/__init__.py tests/unit/test_package.py
git commit -m "chore: bootstrap Python package"
```

---

### Task 2: Shared Strict Validation and Typed Errors

**Files:**

- Create: `src/silent_cascade/validation.py`
- Create: `src/silent_cascade/errors.py`
- Create: `tests/unit/test_validation.py`
- Create: `tests/unit/test_errors.py`

**Interfaces:**

- Consumes: the package environment from Task 1.
- Produces: `JsonValue`, `StrictModel`, and the stable exception classes `SilentCascadeError`, `ConfigurationError`, `ArtifactError`, `AtomicWriteError`, `ArtifactIntegrityError`, `CrashBundleError`, `DoctorError`, `DynamicsError`, `TimeOrderError`, `ProvenanceError`, and `ReplayError`.

- [ ] **Step 1: Write failing strict-model tests**

Create `tests/unit/test_validation.py`:

```python
import math

import pytest
from pydantic import ValidationError

from silent_cascade.validation import StrictModel


class ExampleModel(StrictModel):
    count: int
    value: float


def test_strict_model_rejects_unknown_fields_and_coercion() -> None:
    with pytest.raises(ValidationError):
        ExampleModel.model_validate({"count": 1, "value": 2.0, "extra": True})
    with pytest.raises(ValidationError):
        ExampleModel.model_validate({"count": "1", "value": 2.0})


def test_strict_model_is_frozen_and_rejects_nonfinite_values() -> None:
    model = ExampleModel(count=1, value=2.0)
    with pytest.raises(ValidationError):
        model.count = 2
    with pytest.raises(ValidationError):
        ExampleModel(count=1, value=math.nan)
```

- [ ] **Step 2: Write failing typed-error tests**

Create `tests/unit/test_errors.py`:

```python
from silent_cascade.errors import (
    ArtifactError,
    AtomicWriteError,
    DynamicsError,
    SilentCascadeError,
    TimeOrderError,
)


def test_typed_error_payload_has_stable_code_message_and_context() -> None:
    error = AtomicWriteError("replace failed", context={"path": "artifact.json"})
    assert error.to_payload() == {
        "code": "atomic_write_error",
        "message": "replace failed",
        "context": {"path": "artifact.json"},
    }


def test_error_hierarchy_supports_precise_and_family_catches() -> None:
    assert issubclass(AtomicWriteError, ArtifactError)
    assert issubclass(ArtifactError, SilentCascadeError)
    assert issubclass(TimeOrderError, DynamicsError)
```

- [ ] **Step 3: Run the focused tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_validation.py tests/unit/test_errors.py -q
```

Expected: collection fails because `silent_cascade.validation` and `silent_cascade.errors` do not exist.

- [ ] **Step 4: Implement the shared strict model**

Create `src/silent_cascade/validation.py`:

```python
"""Strict shared validation types."""

from pydantic import BaseModel, ConfigDict

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class StrictModel(BaseModel):
    """Base model for immutable, non-coercing repository contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )
```

- [ ] **Step 5: Implement the typed error hierarchy**

Create `src/silent_cascade/errors.py`:

```python
"""Stable typed errors for fail-loud execution."""

from collections.abc import Mapping
from typing import ClassVar

from silent_cascade.validation import JsonValue


class SilentCascadeError(Exception):
    code: ClassVar[str] = "silent_cascade_error"

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = dict(context or {})

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }


class ConfigurationError(SilentCascadeError):
    code = "configuration_error"


class ArtifactError(SilentCascadeError):
    code = "artifact_error"


class AtomicWriteError(ArtifactError):
    code = "atomic_write_error"


class ArtifactIntegrityError(ArtifactError):
    code = "artifact_integrity_error"


class CrashBundleError(ArtifactError):
    code = "crash_bundle_error"


class DoctorError(SilentCascadeError):
    code = "doctor_error"


class DynamicsError(SilentCascadeError):
    code = "dynamics_error"


class TimeOrderError(DynamicsError):
    code = "time_order_error"


class ProvenanceError(SilentCascadeError):
    code = "provenance_error"


class ReplayError(SilentCascadeError):
    code = "replay_error"
```

- [ ] **Step 6: Run focused and package tests**

Run:

```bash
uv run pytest tests/unit/test_validation.py tests/unit/test_errors.py \
  tests/unit/test_package.py -q
```

Expected: `5 passed`.

- [ ] **Step 7: Commit the validation foundation**

```bash
git add src/silent_cascade/validation.py src/silent_cascade/errors.py \
  tests/unit/test_validation.py tests/unit/test_errors.py
git commit -m "feat: add strict validation and typed errors"
```

---

### Task 3: Canonical Hashing and Durable Atomic I/O

**Files:**

- Create: `src/silent_cascade/hashing.py`
- Create: `src/silent_cascade/io.py`
- Create: `tests/unit/test_hashing.py`
- Create: `tests/unit/test_io.py`

**Interfaces:**

- Consumes: `StrictModel`, `JsonValue`, `AtomicWriteError`, and `ArtifactIntegrityError` from Task 2.
- Produces: `canonical_json_bytes(value) -> bytes`, `sha256_bytes(payload) -> str`, `sha256_file(path) -> str`, `verify_file_sha256(path, expected) -> None`, `atomic_write_bytes`, `atomic_write_text`, `atomic_write_json`, and no-clobber `atomic_create_bytes`.

- [ ] **Step 1: Write failing canonical-hash tests**

Create `tests/unit/test_hashing.py`:

```python
from pathlib import Path

import pytest

from silent_cascade.errors import ArtifactIntegrityError
from silent_cascade.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    verify_file_sha256,
)


def test_canonical_json_is_compact_and_insertion_order_independent() -> None:
    first = {"z": [2, 1], "a": {"enabled": True}}
    second = {"a": {"enabled": True}, "z": [2, 1]}
    expected = b'{"a":{"enabled":true},"z":[2,1]}'
    assert canonical_json_bytes(first) == canonical_json_bytes(second) == expected
    assert sha256_bytes(expected) == sha256_bytes(canonical_json_bytes(first))


def test_file_hash_verification_detects_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original")
    expected = sha256_file(artifact)
    verify_file_sha256(artifact, expected)
    artifact.write_bytes(b"changed")
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        verify_file_sha256(artifact, expected)
```

- [ ] **Step 2: Write failing atomic-I/O tests**

Create `tests/unit/test_io.py`:

```python
import os
from pathlib import Path

import pytest

from silent_cascade.errors import AtomicWriteError
from silent_cascade.io import atomic_create_bytes, atomic_write_bytes, atomic_write_json


def test_atomic_write_replaces_destination_with_complete_contents(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"old")
    atomic_write_bytes(destination, b"new")
    assert destination.read_bytes() == b"new"
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_atomic_write_failure_preserves_destination_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"old")

    def fail_replace(
        source: str | os.PathLike[str], target: str | os.PathLike[str]
    ) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(AtomicWriteError, match="atomic replace failed"):
        atomic_write_bytes(destination, b"new")
    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob(".artifact.bin.*.tmp"))


def test_atomic_create_refuses_to_clobber_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "immutable.bin"
    destination.write_bytes(b"first")
    with pytest.raises(AtomicWriteError, match="already exists"):
        atomic_create_bytes(destination, b"second")
    assert destination.read_bytes() == b"first"


def test_atomic_write_json_uses_canonical_encoding(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    atomic_write_json(destination, {"z": 2, "a": 1})
    assert destination.read_bytes() == b'{"a":1,"z":2}\n'
```

- [ ] **Step 3: Run the focused tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_hashing.py tests/unit/test_io.py -q
```

Expected: collection fails because `silent_cascade.hashing` and `silent_cascade.io` do not exist.

- [ ] **Step 4: Implement canonical JSON and SHA-256 helpers**

Create `src/silent_cascade/hashing.py`:

```python
"""Canonical serialization and SHA-256 helpers."""

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel

from silent_cascade.errors import ArtifactIntegrityError
from silent_cascade.validation import JsonValue


def canonical_json_bytes(
    value: BaseModel | Mapping[str, JsonValue],
) -> bytes:
    payload = (
        value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    )
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return encoded.encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ArtifactIntegrityError(
            "artifact hash mismatch",
            context={"path": str(path), "expected": expected, "actual": actual},
        )
```

- [ ] **Step 5: Implement durable atomic replace and no-clobber creation**

Create `src/silent_cascade/io.py` with these public signatures and implementation rules:

```python
"""Durable atomic file publication."""

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel

from silent_cascade.errors import AtomicWriteError
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.validation import JsonValue


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_temp(path: Path, data: bytes, mode: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode)
        return temp_path
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    temp_path = _durable_temp(path, data, mode)
    try:
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except OSError as error:
        raise AtomicWriteError(
            "atomic replace failed",
            context={"path": str(path), "reason": str(error)},
        ) from error
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_create_bytes(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    temp_path = _durable_temp(path, data, mode)
    try:
        os.link(temp_path, path)
        temp_path.unlink()
        _fsync_directory(path.parent)
    except FileExistsError as error:
        raise AtomicWriteError(
            "artifact already exists",
            context={"path": str(path)},
        ) from error
    except OSError as error:
        raise AtomicWriteError(
            "atomic create failed",
            context={"path": str(path), "reason": str(error)},
        ) from error
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o644,
) -> None:
    atomic_write_bytes(path, text.encode(encoding), mode=mode)


def atomic_write_json(
    path: Path,
    value: BaseModel | dict[str, JsonValue],
    *,
    mode: int = 0o644,
) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n", mode=mode)
```

Do not implement no-clobber creation as `exists()` followed by `replace()`: that sequence is race-prone. `os.link` publishes the fsynced inode only when the destination is absent on the supported macOS/Linux filesystems.

- [ ] **Step 6: Run focused and full unit tests**

Run:

```bash
uv run pytest tests/unit/test_hashing.py tests/unit/test_io.py -q
uv run pytest tests/unit -q
```

Expected: the focused file reports `6 passed`; the unit suite reports `11 passed`.

- [ ] **Step 7: Commit the artifact primitives**

```bash
git add src/silent_cascade/hashing.py src/silent_cascade/io.py \
  tests/unit/test_hashing.py tests/unit/test_io.py
git commit -m "feat: add atomic artifact primitives"
```

---

### Task 4: Strict Layered Configuration and Configuration Hashes

**Files:**

- Create: `configs/base.yaml`
- Create: `src/silent_cascade/config.py`
- Create: `tests/unit/test_config.py`

**Interfaces:**

- Consumes: `StrictModel`, `JsonValue`, `ConfigurationError`, `canonical_json_bytes`, and `sha256_bytes`.
- Produces: `RuntimeConfig`, `LimitsConfig`, `PathsConfig`, `ProjectConfig`, generic `ResolvedConfig[TConfig]`, `parse_set_override(expression)`, and `resolve_config(model_type, yaml_paths, set_overrides=())`.

- [ ] **Step 1: Write failing layer, override, and hash tests**

Create `tests/unit/test_config.py`:

```python
from pathlib import Path

import pytest

from silent_cascade.config import ProjectConfig, resolve_config
from silent_cascade.errors import ConfigurationError


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_resolve_config_deep_merges_in_declared_order_then_applies_overrides(
    tmp_path: Path,
) -> None:
    base = write_yaml(
        tmp_path / "base.yaml",
        """
schema_version: 1
experiment_version: v1
limits:
  batch_size: 128
  primary_memory_records: 64
""",
    )
    layer = write_yaml(
        tmp_path / "layer.yaml",
        """
limits:
  batch_size: 64
""",
    )
    resolved = resolve_config(
        ProjectConfig,
        [base, layer],
        set_overrides=["limits.batch_size=32"],
    )
    assert resolved.config.limits.batch_size == 32
    assert resolved.config.limits.primary_memory_records == 64
    assert resolved.source_paths == (base, layer)


def test_resolve_config_rejects_unknown_keys_and_nonfinite_numbers(
    tmp_path: Path,
) -> None:
    unknown = write_yaml(
        tmp_path / "unknown.yaml",
        "schema_version: 1\nexperiment_version: v1\nunknown: true\n",
    )
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(ProjectConfig, [unknown])

    nonfinite = write_yaml(
        tmp_path / "nonfinite.yaml",
        "schema_version: 1\nexperiment_version: v1\nlimits:\n  batch_size: .nan\n",
    )
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(ProjectConfig, [nonfinite])


def test_equivalent_configs_have_identical_canonical_bytes_and_hash(
    tmp_path: Path,
) -> None:
    first = write_yaml(
        tmp_path / "first.yaml",
        "schema_version: 1\nexperiment_version: v1\nruntime:\n  dtype: float32\n",
    )
    second = write_yaml(
        tmp_path / "second.yaml",
        "runtime:\n  dtype: float32\nexperiment_version: v1\nschema_version: 1\n",
    )
    resolved_first = resolve_config(ProjectConfig, [first])
    resolved_second = resolve_config(ProjectConfig, [second])
    assert resolved_first.canonical_json == resolved_second.canonical_json
    assert resolved_first.sha256 == resolved_second.sha256
    assert b'"max_trainable_parameters":5000000' in resolved_first.canonical_json
    assert b'"retain_all_failure_traces":true' in resolved_first.canonical_json


@pytest.mark.parametrize(
    "override",
    ["", "missing_equals", ".leading=1", "trailing.=1", "double..dot=1"],
)
def test_invalid_set_override_raises_typed_error(tmp_path: Path, override: str) -> None:
    base = write_yaml(
        tmp_path / "base.yaml",
        "schema_version: 1\nexperiment_version: v1\n",
    )
    with pytest.raises(ConfigurationError, match="invalid --set override"):
        resolve_config(ProjectConfig, [base], set_overrides=[override])


def test_committed_paths_must_be_relative(tmp_path: Path) -> None:
    config = write_yaml(
        tmp_path / "absolute.yaml",
        "schema_version: 1\nexperiment_version: v1\npaths:\n  runs: /tmp/runs\n",
    )
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(ProjectConfig, [config])


def test_runtime_device_preference_must_include_cpu(tmp_path: Path) -> None:
    config = write_yaml(
        tmp_path / "mps-only.yaml",
        "schema_version: 1\nexperiment_version: v1\nruntime:\n"
        "  device_preference: [mps]\n",
    )
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(ProjectConfig, [config])
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_config.py -q
```

Expected: collection fails because `silent_cascade.config` does not exist.

- [ ] **Step 3: Implement the strict bootstrap schema**

Create the models in `src/silent_cascade/config.py`:

```python
"""Strict layered configuration with canonical hashes."""

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, field_validator

from silent_cascade.errors import ConfigurationError
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.validation import JsonValue, StrictModel


class RuntimeConfig(StrictModel):
    device_preference: tuple[Literal["mps", "cpu"], ...] = ("mps", "cpu")
    dtype: Literal["float32"] = "float32"
    primary_offline: Literal[True] = True
    allow_mps_fallback: Literal[False] = False
    primary_foundation_model_calls: Literal[0] = 0

    @field_validator("device_preference", mode="before")
    @classmethod
    def tuple_from_yaml_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("device_preference")
    @classmethod
    def require_cpu_path(
        cls, value: tuple[Literal["mps", "cpu"], ...]
    ) -> tuple[Literal["mps", "cpu"], ...]:
        if "cpu" not in value:
            raise ValueError("device_preference must include the mandatory CPU path")
        if len(value) != len(set(value)):
            raise ValueError("device_preference may not contain duplicates")
        return value


class LimitsConfig(StrictModel):
    max_trainable_parameters: int = Field(default=5_000_000, ge=1, le=5_000_000)
    primary_memory_records: int = Field(default=64, ge=1, le=64)
    max_eventflow_events: int = Field(default=64, ge=1, le=64)
    max_fixed_grid_opportunities: int = Field(default=25_000, ge=1, le=25_000)
    batch_size: int = Field(default=128, ge=1, le=128)
    retained_full_traces: int = Field(default=500, ge=0, le=500)
    retain_all_failure_traces: Literal[True] = True
    foundation_model_workers: int = Field(default=1, ge=0, le=1)


class PathsConfig(StrictModel):
    runs: str = "runs"
    reports: str = "reports"
    manifests: str = "manifests"

    @field_validator("runs", "reports", "manifests")
    @classmethod
    def relative_repository_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "repository path must be relative and may not contain '..'"
            )
        return value


class ProjectConfig(StrictModel):
    schema_version: Literal[1] = 1
    experiment_version: str = Field(default="v1", pattern=r"^[a-z0-9][a-z0-9._-]*$")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


@dataclass(frozen=True, slots=True)
class ResolvedConfig[TConfig: StrictModel]:
    config: TConfig
    canonical_json: bytes
    sha256: str
    source_paths: tuple[Path, ...]
```

- [ ] **Step 4: Implement deterministic merge, override parsing, and validation**

Continue `src/silent_cascade/config.py` with:

```python

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


def _read_yaml_mapping(path: Path) -> dict[str, JsonValue]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(
            "configuration load failed",
            context={"path": str(path), "reason": str(error)},
        ) from error
    if loaded is None:
        return {}
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise ConfigurationError(
            "configuration root must be a string-keyed mapping",
            context={"path": str(path)},
        )
    return loaded


def _deep_merge(
    base: Mapping[str, JsonValue], overlay: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    merged = deepcopy(dict(base))
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def parse_set_override(expression: str) -> tuple[tuple[str, ...], JsonValue]:
    if "=" not in expression:
        raise ConfigurationError(
            "invalid --set override",
            context={"expression": expression},
        )
    dotted_key, raw_value = expression.split("=", 1)
    path = tuple(dotted_key.split("."))
    if not path or any(not segment or not _KEY.fullmatch(segment) for segment in path):
        raise ConfigurationError(
            "invalid --set override",
            context={"expression": expression},
        )
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as error:
        raise ConfigurationError(
            "invalid --set override",
            context={"expression": expression, "reason": str(error)},
        ) from error
    return path, value


def _apply_override(
    target: dict[str, JsonValue], path: tuple[str, ...], value: JsonValue
) -> None:
    cursor = target
    for segment in path[:-1]:
        child = cursor.setdefault(segment, {})
        if not isinstance(child, dict):
            raise ConfigurationError(
                "override traverses a non-mapping value",
                context={"path": ".".join(path)},
            )
        cursor = child
    cursor[path[-1]] = value


def resolve_config[TConfig: StrictModel](
    model_type: type[TConfig],
    yaml_paths: Sequence[Path],
    *,
    set_overrides: Sequence[str] = (),
) -> ResolvedConfig[TConfig]:
    merged: dict[str, JsonValue] = {}
    for path in yaml_paths:
        merged = _deep_merge(merged, _read_yaml_mapping(path))
    for expression in set_overrides:
        key_path, value = parse_set_override(expression)
        _apply_override(merged, key_path, value)
    try:
        config = model_type.model_validate(merged)
    except ValidationError as error:
        raise ConfigurationError(
            "configuration validation failed",
            context={"details": str(error)},
        ) from error
    canonical = canonical_json_bytes(config)
    return ResolvedConfig(
        config=config,
        canonical_json=canonical,
        sha256=sha256_bytes(canonical),
        source_paths=tuple(yaml_paths),
    )
```

The implementation must validate once, after all layers and overrides are applied. Lists and scalar values replace wholesale; only mappings merge recursively.

- [ ] **Step 5: Add the committed bootstrap configuration**

Create `configs/base.yaml`:

```yaml
schema_version: 1
experiment_version: v1
runtime:
  device_preference: [mps, cpu]
  dtype: float32
  primary_offline: true
  allow_mps_fallback: false
  primary_foundation_model_calls: 0
limits:
  max_trainable_parameters: 5000000
  primary_memory_records: 64
  max_eventflow_events: 64
  max_fixed_grid_opportunities: 25000
  batch_size: 128
  retained_full_traces: 500
  retain_all_failure_traces: true
  foundation_model_workers: 1
paths:
  runs: runs
  reports: reports
  manifests: manifests
```

- [ ] **Step 6: Run focused and full unit tests**

Run:

```bash
uv run pytest tests/unit/test_config.py -q
uv run pytest tests/unit -q
```

Expected: config tests report `10 passed`; the full unit suite reports `21 passed`.

- [ ] **Step 7: Commit strict configuration**

```bash
git add configs/base.yaml src/silent_cascade/config.py tests/unit/test_config.py
git commit -m "feat: add strict configuration resolution"
```

---

### Task 5: Reproducible Global RNG State

**Files:**

- Create: `src/silent_cascade/rng.py`
- Create: `tests/unit/test_rng.py`

**Interfaces:**

- Consumes: `StrictModel` and `DoctorError`.
- Produces: immutable `RngSnapshot`, serializable `RngRoundTripReport`, `mps_rng_state_supported()`, `seed_all(seed)`, `snapshot_global_rng()`, `restore_global_rng(snapshot)`, and `verify_rng_round_trip(include_mps)`.

- [ ] **Step 1: Write failing CPU and capability tests**

Create `tests/unit/test_rng.py`:

```python
import random

import numpy as np
import pytest
import torch

from silent_cascade.rng import (
    restore_global_rng,
    seed_all,
    snapshot_global_rng,
    verify_rng_round_trip,
)


def test_rng_snapshot_restore_repeats_python_numpy_and_torch_cpu_draws() -> None:
    outer = snapshot_global_rng()
    try:
        seed_all(1729)
        checkpoint = snapshot_global_rng()
        first = (random.random(), np.random.random(), torch.rand(4))
        restore_global_rng(checkpoint)
        second = (random.random(), np.random.random(), torch.rand(4))
        assert first[0] == second[0]
        assert first[1] == second[1]
        assert torch.equal(first[2], second[2])
    finally:
        restore_global_rng(outer)


def test_rng_round_trip_reports_mps_unchecked_when_not_requested() -> None:
    report = verify_rng_round_trip(include_mps=False)
    assert report.python_ok
    assert report.numpy_ok
    assert report.torch_cpu_ok
    assert not report.torch_mps_checked
    assert report.torch_mps_ok is None


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is a local-only capability gate",
)
def test_rng_snapshot_restore_repeats_mps_draws_when_available() -> None:
    report = verify_rng_round_trip(include_mps=True)
    assert report.torch_mps_checked
    assert isinstance(report.torch_mps_ok, bool)
```

- [ ] **Step 2: Run the RNG tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_rng.py -q
```

Expected: collection fails because `silent_cascade.rng` does not exist.

- [ ] **Step 3: Implement RNG snapshot, restore, and seeding**

Create `src/silent_cascade/rng.py`:

```python
"""Reproducible global RNG capture for Python, NumPy, Torch CPU, and MPS."""

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from silent_cascade.errors import DoctorError
from silent_cascade.validation import StrictModel


@dataclass(frozen=True, slots=True)
class RngSnapshot:
    python_state: object
    numpy_state: tuple[Any, ...]
    torch_cpu_state: torch.Tensor
    torch_mps_state: torch.Tensor | None


class RngRoundTripReport(StrictModel):
    python_ok: bool
    numpy_ok: bool
    torch_cpu_ok: bool
    torch_mps_checked: bool
    torch_mps_ok: bool | None


def mps_rng_state_supported() -> bool:
    return bool(
        torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and hasattr(torch.mps, "set_rng_state")
    )


def seed_all(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if mps_rng_state_supported() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def snapshot_global_rng() -> RngSnapshot:
    mps_state = torch.mps.get_rng_state().clone() if mps_rng_state_supported() else None
    return RngSnapshot(
        python_state=copy.deepcopy(random.getstate()),
        numpy_state=copy.deepcopy(np.random.get_state()),
        torch_cpu_state=torch.get_rng_state().clone(),
        torch_mps_state=mps_state,
    )


def restore_global_rng(snapshot: RngSnapshot) -> None:
    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_state)
    torch.set_rng_state(snapshot.torch_cpu_state.clone())
    if snapshot.torch_mps_state is not None:
        if not mps_rng_state_supported():
            raise DoctorError("MPS RNG state cannot be restored on this runtime")
        torch.mps.set_rng_state(snapshot.torch_mps_state.clone())
```

- [ ] **Step 4: Implement the non-mutating RNG round-trip diagnostic**

Continue `src/silent_cascade/rng.py`:

```python

def verify_rng_round_trip(*, include_mps: bool) -> RngRoundTripReport:
    outer = snapshot_global_rng()
    try:
        seed_all(0x5A17)
        checkpoint = snapshot_global_rng()
        first_python = random.random()
        first_numpy = float(np.random.random())
        first_cpu = torch.rand(8)
        first_mps = (
            torch.rand(8, device="mps").cpu()
            if include_mps and mps_rng_state_supported()
            else None
        )

        restore_global_rng(checkpoint)
        second_python = random.random()
        second_numpy = float(np.random.random())
        second_cpu = torch.rand(8)
        second_mps = (
            torch.rand(8, device="mps").cpu()
            if include_mps and mps_rng_state_supported()
            else None
        )
        if include_mps and torch.backends.mps.is_available():
            torch.mps.synchronize()

        mps_checked = include_mps and mps_rng_state_supported()
        mps_ok = (
            torch.equal(first_mps, second_mps)
            if first_mps is not None and second_mps is not None
            else None
        )
        return RngRoundTripReport(
            python_ok=first_python == second_python,
            numpy_ok=first_numpy == second_numpy,
            torch_cpu_ok=torch.equal(first_cpu, second_cpu),
            torch_mps_checked=mps_checked,
            torch_mps_ok=mps_ok,
        )
    finally:
        restore_global_rng(outer)
```

The diagnostic must restore the caller's original RNG state in `finally`, including when a check raises.

- [ ] **Step 5: Run focused and full unit tests**

Run:

```bash
uv run pytest tests/unit/test_rng.py -q
uv run pytest tests/unit -q
```

Expected when MPS is unavailable: `2 passed, 1 skipped` for the focused file. Expected on the target M3 Max when MPS is available: `3 passed`; if that runtime reports MPS unavailable, retain the single capability skip and diagnose the reported state rather than failing merely for absence. The MPS test proves that state capture/restore can be exercised and reports whether the repeated draw was bitwise equal; bitwise equality is a non-fatal capability diagnostic, not the MPS reproducibility contract. Later scientific runs establish MPS reproducibility through seed replication. The full suite has the same conditional skip and no failures.

- [ ] **Step 6: Commit RNG state support**

```bash
git add src/silent_cascade/rng.py tests/unit/test_rng.py
git commit -m "feat: add reproducible RNG state"
```

---

### Task 6: Atomic Crash Bundle Manifests

**Files:**

- Create: `src/silent_cascade/logging/__init__.py`
- Create: `src/silent_cascade/logging/crash_bundle.py`
- Create: `tests/unit/test_crash_bundle.py`

**Interfaces:**

- Consumes: `StrictModel`, `JsonValue`, `SilentCascadeError`, `CrashBundleError`, `atomic_create_bytes`, `canonical_json_bytes`, and `sha256_bytes`.
- Produces: strict `CrashContext`, strict `CrashBundleManifest`, immutable `CrashBundleArtifact`, and `write_crash_bundle(root, error, context, bundle_id=None, now=None) -> CrashBundleArtifact`.

- [ ] **Step 1: Write failing crash-bundle tests**

Create `tests/unit/test_crash_bundle.py`:

```python
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from silent_cascade.errors import CrashBundleError, DynamicsError
from silent_cascade.hashing import sha256_file
from silent_cascade.logging.crash_bundle import CrashContext, write_crash_bundle


def test_crash_bundle_contains_context_and_only_last_twenty_events(
    tmp_path: Path,
) -> None:
    events = tuple({"event_id": index} for index in range(25))
    context = CrashContext(
        seed=23,
        episode_public_id="episode-public-7",
        checkpoint_ref="sha256:checkpoint",
        config_sha256="a" * 64,
        source_revision="deadbeef",
        last_events=events,
    )
    artifact = write_crash_bundle(
        tmp_path,
        error=DynamicsError("non-finite state", context={"tensor": "z_fast"}),
        context=context,
        bundle_id="bundle-001",
        now=datetime(2026, 8, 30, 12, 0, tzinfo=UTC),
    )
    payload = json.loads(artifact.path.read_text(encoding="utf-8"))
    assert artifact.sha256 == sha256_file(artifact.path)
    assert payload["error"]["code"] == "dynamics_error"
    assert "DynamicsError: non-finite state" in payload["traceback_text"]
    assert payload["context"]["seed"] == 23
    assert payload["context"]["episode_public_id"] == "episode-public-7"
    assert [event["event_id"] for event in payload["context"]["last_events"]] == list(
        range(5, 25)
    )


def test_crash_bundle_refuses_duplicate_explicit_id(tmp_path: Path) -> None:
    context = CrashContext()
    error = DynamicsError("failure")
    write_crash_bundle(tmp_path, error=error, context=context, bundle_id="same-id")
    with pytest.raises(CrashBundleError, match="already exists"):
        write_crash_bundle(tmp_path, error=error, context=context, bundle_id="same-id")


def test_crash_bundle_rejects_path_like_bundle_id(tmp_path: Path) -> None:
    with pytest.raises(CrashBundleError, match="invalid crash bundle id"):
        write_crash_bundle(
            tmp_path,
            error=DynamicsError("failure"),
            context=CrashContext(),
            bundle_id="../escape",
        )
```

- [ ] **Step 2: Run the crash-bundle tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_crash_bundle.py -q
```

Expected: collection fails because `silent_cascade.logging.crash_bundle` does not exist.

- [ ] **Step 3: Implement strict crash schemas**

Create an empty, side-effect-free `src/silent_cascade/logging/__init__.py` and create `src/silent_cascade/logging/crash_bundle.py`:

```python
"""Atomic, replay-oriented crash bundle manifests."""

import re
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from silent_cascade.errors import (
    AtomicWriteError,
    CrashBundleError,
    SilentCascadeError,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.io import atomic_create_bytes
from silent_cascade.validation import JsonValue, StrictModel

_BUNDLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class CrashContext(StrictModel):
    seed: int | None = None
    episode_public_id: str | None = None
    checkpoint_ref: str | None = None
    config_sha256: str | None = None
    source_revision: str | None = None
    last_events: tuple[dict[str, JsonValue], ...] = ()


class CrashBundleManifest(StrictModel):
    schema_version: Literal[1] = 1
    bundle_id: str
    created_at_utc: datetime
    error: dict[str, JsonValue]
    traceback_text: str
    context: CrashContext


@dataclass(frozen=True, slots=True)
class CrashBundleArtifact:
    path: Path
    sha256: str
```

- [ ] **Step 4: Implement bounded context and atomic publication**

Continue `src/silent_cascade/logging/crash_bundle.py`:

```python

def write_crash_bundle(
    root: Path,
    *,
    error: SilentCascadeError,
    context: CrashContext,
    bundle_id: str | None = None,
    now: datetime | None = None,
) -> CrashBundleArtifact:
    resolved_id = bundle_id or uuid4().hex
    if not _BUNDLE_ID.fullmatch(resolved_id):
        raise CrashBundleError(
            "invalid crash bundle id",
            context={"bundle_id": resolved_id},
        )
    created_at = now or datetime.now(UTC)
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise CrashBundleError("crash bundle timestamp must be timezone-aware")

    bounded_context = context.model_copy(
        update={"last_events": context.last_events[-20:]},
    )
    manifest = CrashBundleManifest(
        bundle_id=resolved_id,
        created_at_utc=created_at.astimezone(UTC),
        error=error.to_payload(),
        traceback_text="".join(traceback.format_exception(error)),
        context=bounded_context,
    )
    payload = canonical_json_bytes(manifest) + b"\n"
    destination = root / f"{resolved_id}.json"
    try:
        atomic_create_bytes(destination, payload, mode=0o600)
    except AtomicWriteError as write_error:
        if destination.exists():
            raise CrashBundleError(
                "crash bundle already exists",
                context={"path": str(destination)},
            ) from write_error
        raise CrashBundleError(
            "crash bundle publication failed",
            context={"path": str(destination), "reason": str(write_error)},
        ) from write_error
    return CrashBundleArtifact(path=destination, sha256=sha256_bytes(payload))
```

The bundle is a single atomically published manifest in Phase 0. Its context already carries stable checkpoint/config references and the final 20 event summaries; Phase 4 may add referenced tensor/trace artifacts without changing this manifest's existing fields.

- [ ] **Step 5: Run focused and full unit tests**

Run:

```bash
uv run pytest tests/unit/test_crash_bundle.py -q
uv run pytest tests/unit -q
```

Expected: crash-bundle tests report `3 passed`; the full suite has no failures and at most the one MPS skip.

- [ ] **Step 6: Commit crash bundle support**

```bash
git add src/silent_cascade/logging/__init__.py \
  src/silent_cascade/logging/crash_bundle.py tests/unit/test_crash_bundle.py
git commit -m "feat: add crash bundle manifests"
```

---

### Task 7: Environment Doctor, CLI, and Optional-Import Boundary

**Files:**

- Create: `src/silent_cascade/doctor.py`
- Create: `src/silent_cascade/cli.py`
- Modify: `pyproject.toml`
- Create: `tests/unit/test_doctor.py`
- Create: `tests/integration/test_cli_doctor.py`
- Create: `tests/regression/test_import_boundaries.py`

**Interfaces:**

- Consumes: `StrictModel`, `ProjectConfig`, `resolve_config`, RNG diagnostics, and the installed Torch/NumPy/Pydantic environment.
- Produces: `MpsReport`, `WritablePathReport`, `NumericSmokeReport`, `HostReport`, `DoctorReport`, `run_doctor(...)`, Typer `app`, callable `main()`, and executable `silent-cascade doctor [--json] [--qwen]`.

- [ ] **Step 1: Write failing doctor service tests**

Create `tests/unit/test_doctor.py`:

```python
import importlib.util
import sys
from pathlib import Path

import torch

from silent_cascade.doctor import run_doctor


def test_run_doctor_reports_cpu_rng_numeric_and_mps_capability(tmp_path: Path) -> None:
    report = run_doctor(
        config_path=Path("configs/base.yaml"),
        writable_paths=[tmp_path / "runs", tmp_path / "reports"],
    )
    assert report.python_supported
    assert report.torch_cpu_ok
    assert report.numeric.flow_ok
    assert report.numeric.guard_ok
    assert report.rng.python_ok
    assert report.rng.numpy_ok
    assert report.rng.torch_cpu_ok
    assert report.primary_foundation_model_calls == 0
    assert report.mps.built == torch.backends.mps.is_built()
    assert report.mps.available == torch.backends.mps.is_available()
    if report.mps.available:
        assert report.mps.flow_ok
        assert report.mps.guard_ok
        assert report.mps.rng_round_trip_checked == report.rng.torch_mps_checked
        assert report.mps.rng_round_trip_exact == report.rng.torch_mps_ok
    assert report.host.total_memory_bytes > 0
    assert report.host.system
    assert report.host.machine
    assert all(item.writable for item in report.writable_paths)
    assert report.ok


def test_doctor_fails_when_silent_mps_fallback_is_enabled(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    report = run_doctor(
        config_path=Path("configs/base.yaml"),
        writable_paths=[tmp_path],
    )
    assert report.mps_fallback_enabled
    assert not report.ok


def test_qwen_check_discovers_packages_without_importing_them(
    tmp_path: Path, monkeypatch
) -> None:
    requested: list[str] = []
    real_find_spec = importlib.util.find_spec

    def recording_find_spec(name: str):
        requested.append(name)
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", recording_find_spec)
    report = run_doctor(
        config_path=Path("configs/base.yaml"),
        writable_paths=[tmp_path],
        check_qwen=True,
    )
    assert requested == ["mlx", "mlx_vlm", "huggingface_hub"]
    assert not {"mlx", "mlx_vlm", "huggingface_hub"}.intersection(sys.modules)
    assert report.qwen_extra_requested
```

- [ ] **Step 2: Write failing installed CLI and import-boundary tests**

Create `tests/integration/test_cli_doctor.py`:

```python
import json
import shutil
import subprocess

from typer.testing import CliRunner

from silent_cascade.cli import app

runner = CliRunner()


def test_cli_doctor_json_matches_report_schema(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--json", "--config", "-"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["torch_cpu_ok"] is True
    assert payload["numeric"]["flow_ok"] is True


def test_installed_console_script_runs_doctor(tmp_path) -> None:
    executable = shutil.which("silent-cascade")
    assert executable is not None
    completed = subprocess.run(
        [executable, "doctor", "--json", "--config", "-"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["ok"] is True


def test_cli_has_no_unimplemented_commands() -> None:
    assert {command.name for command in app.registered_commands} == {"doctor"}
```

The CLI convention `--config -` means “use the strict in-code `ProjectConfig` defaults.” A normal repository invocation defaults to `configs/base.yaml`.

Create `tests/regression/test_import_boundaries.py`:

```python
import subprocess
import sys


def test_core_imports_do_not_load_optional_qwen_modules() -> None:
    program = r"""
import importlib.abc
import sys

blocked = {"mlx", "mlx_vlm", "huggingface_hub"}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError(f"optional import attempted: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
import silent_cascade
import silent_cascade.config
import silent_cascade.rng
import silent_cascade.io
import silent_cascade.doctor
import silent_cascade.cli
assert not blocked.intersection(sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run:

```bash
uv run pytest tests/unit/test_doctor.py tests/integration/test_cli_doctor.py \
  tests/regression/test_import_boundaries.py -q
```

Expected: collection fails because `silent_cascade.doctor` and `silent_cascade.cli` do not exist.

- [ ] **Step 4: Implement strict doctor report models and numeric smoke**

Create the following foundation in `src/silent_cascade/doctor.py`:

```python
"""Offline environment diagnostics for Silent Cascade."""

import importlib.metadata
import importlib.util
import math
import os
import platform
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import psutil
import torch

from silent_cascade.config import ProjectConfig, resolve_config
from silent_cascade.rng import (
    RngRoundTripReport,
    mps_rng_state_supported,
    verify_rng_round_trip,
)
from silent_cascade.validation import StrictModel


class MpsReport(StrictModel):
    built: bool
    available: bool
    flow_ok: bool | None
    guard_ok: bool | None
    rng_state_supported: bool
    rng_round_trip_checked: bool
    rng_round_trip_exact: bool | None


class WritablePathReport(StrictModel):
    path: str
    writable: bool
    error: str | None = None


class NumericSmokeReport(StrictModel):
    device: Literal["cpu", "mps"]
    flow_ok: bool
    guard_ok: bool


class HostReport(StrictModel):
    system: str
    release: str
    machine: str
    processor: str
    total_memory_bytes: int


class DoctorReport(StrictModel):
    ok: bool
    python_version: str
    python_supported: bool
    package_versions: dict[str, str]
    torch_cpu_ok: bool
    cpu_count_logical: int | None
    cpu_count_physical: int | None
    host: HostReport
    mps_fallback_enabled: bool
    mps: MpsReport
    writable_paths: tuple[WritablePathReport, ...]
    rng: RngRoundTripReport
    numeric: NumericSmokeReport
    config_sha256: str
    primary_foundation_model_calls: Literal[0]
    qwen_extra_requested: bool
    qwen_extra_installed: bool | None


def _numeric_smoke(device: Literal["cpu", "mps"]) -> NumericSmokeReport:
    state = torch.tensor([0.0], dtype=torch.float32, device=device)
    target = torch.tensor([0.5], dtype=torch.float32, device=device)
    rate = torch.tensor([2.0], dtype=torch.float32, device=device)
    dt = torch.tensor(0.25, dtype=torch.float32, device=device)
    weight = -torch.expm1(-rate * dt)
    flowed = state + weight * (target - state)
    expected = 0.5 * (1.0 - math.exp(-0.5))
    flow_ok = math.isclose(float(flowed.cpu().item()), expected, rel_tol=1e-6)

    accumulator = torch.tensor([0.0], dtype=torch.float32, device=device)
    asymptote = torch.tensor([1.5], dtype=torch.float32, device=device)
    guard_rate = torch.tensor([2.0], dtype=torch.float32, device=device)
    delta = torch.log1p((1.0 - accumulator) / (asymptote - 1.0)) / guard_rate
    crossed = asymptote + (accumulator - asymptote) * torch.exp(-guard_rate * delta)
    guard_ok = math.isclose(float(crossed.cpu().item()), 1.0, rel_tol=0.0, abs_tol=1e-6)
    return NumericSmokeReport(device=device, flow_ok=flow_ok, guard_ok=guard_ok)
```

- [ ] **Step 5: Implement capability, path, config, and Qwen discovery checks**

Continue `src/silent_cascade/doctor.py`:

```python

_CORE_PACKAGES = (
    "torch",
    "numpy",
    "pydantic",
    "pyyaml",
    "typer",
    "rich",
    "safetensors",
    "scipy",
    "matplotlib",
    "psutil",
)
_QWEN_PACKAGES = ("mlx", "mlx_vlm", "huggingface_hub")


def _truthy_environment(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _writable_path_report(path: Path) -> WritablePathReport:
    try:
        path.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".doctor-", dir=path)
        os.close(descriptor)
        Path(temporary).unlink()
        return WritablePathReport(path=str(path), writable=True)
    except OSError as error:
        return WritablePathReport(path=str(path), writable=False, error=str(error))


def _qwen_installed() -> bool:
    discoveries = [importlib.util.find_spec(name) for name in _QWEN_PACKAGES]
    return all(discovery is not None for discovery in discoveries)


def run_doctor(
    *,
    config_path: Path | None,
    writable_paths: Sequence[Path],
    check_qwen: bool = False,
) -> DoctorReport:
    resolved = (
        resolve_config(ProjectConfig, [config_path])
        if config_path is not None
        else resolve_config(ProjectConfig, [])
    )
    python_supported = platform.python_version_tuple()[:2] == ("3", "12")
    cpu_probe = torch.tensor([1.0], dtype=torch.float32) + 1.0
    torch_cpu_ok = cpu_probe.dtype == torch.float32 and cpu_probe.item() == 2.0
    fallback_enabled = _truthy_environment("PYTORCH_ENABLE_MPS_FALLBACK")

    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    mps_numeric: NumericSmokeReport | None = None
    if mps_available:
        mps_numeric = _numeric_smoke("mps")
        torch.mps.synchronize()
    mps_rng_supported = mps_rng_state_supported()

    rng_report = verify_rng_round_trip(include_mps=mps_available)
    numeric_report = _numeric_smoke("cpu")
    path_reports = tuple(_writable_path_report(path) for path in writable_paths)
    qwen_installed = _qwen_installed() if check_qwen else None
    package_versions = {
        name: importlib.metadata.version(name) for name in _CORE_PACKAGES
    }

    required_ok = all(
        (
            python_supported,
            torch_cpu_ok,
            not fallback_enabled,
            numeric_report.flow_ok,
            numeric_report.guard_ok,
            rng_report.python_ok,
            rng_report.numpy_ok,
            rng_report.torch_cpu_ok,
            all(item.writable for item in path_reports),
            (not mps_available)
            or (
                mps_numeric is not None and mps_numeric.flow_ok and mps_numeric.guard_ok
            ),
            (not check_qwen) or bool(qwen_installed),
        )
    )
    return DoctorReport(
        ok=required_ok,
        python_version=platform.python_version(),
        python_supported=python_supported,
        package_versions=package_versions,
        torch_cpu_ok=torch_cpu_ok,
        cpu_count_logical=psutil.cpu_count(logical=True),
        cpu_count_physical=psutil.cpu_count(logical=False),
        host=HostReport(
            system=platform.system(),
            release=platform.release(),
            machine=platform.machine(),
            processor=platform.processor(),
            total_memory_bytes=psutil.virtual_memory().total,
        ),
        mps_fallback_enabled=fallback_enabled,
        mps=MpsReport(
            built=mps_built,
            available=mps_available,
            flow_ok=mps_numeric.flow_ok if mps_numeric is not None else None,
            guard_ok=mps_numeric.guard_ok if mps_numeric is not None else None,
            rng_state_supported=mps_rng_supported,
            rng_round_trip_checked=rng_report.torch_mps_checked,
            rng_round_trip_exact=rng_report.torch_mps_ok,
        ),
        writable_paths=path_reports,
        rng=rng_report,
        numeric=numeric_report,
        config_sha256=resolved.sha256,
        primary_foundation_model_calls=(
            resolved.config.runtime.primary_foundation_model_calls
        ),
        qwen_extra_requested=check_qwen,
        qwen_extra_installed=qwen_installed,
    )
```

`run_doctor` must not import or download an optional package. Discovery is metadata-only. Unavailable MPS is successful. Available MPS must pass both float32 tensor flow and guard probes. MPS RNG-state support and exact round-trip equality remain visible, non-fatal capability diagnostics; scientific MPS reproducibility is assessed through replicated seeds rather than bitwise identity.

- [ ] **Step 6: Implement the thin Typer adapter**

Create `src/silent_cascade/cli.py`:

```python
"""Silent Cascade command-line interface."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from silent_cascade.doctor import DoctorReport, run_doctor

app = typer.Typer(no_args_is_help=True, add_completion=False)


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
```

Add the console script to `pyproject.toml` immediately after `[project]` metadata:

```toml
[project.scripts]
silent-cascade = "silent_cascade.cli:main"
```

- [ ] **Step 7: Run focused tests, then the installed command**

Run:

```bash
uv sync --locked --group dev
uv run pytest tests/unit/test_doctor.py tests/integration/test_cli_doctor.py \
  tests/regression/test_import_boundaries.py -q
uv run silent-cascade doctor --json
```

Expected without the Qwen extra: all focused tests pass; the doctor exits 0 and emits one JSON object with `"ok": true`. MPS fields reflect the current host. Do not run `--qwen` as a Phase 0 gate unless the optional extra was explicitly installed.

- [ ] **Step 8: Run the complete test suite**

Run:

```bash
uv run pytest -q
```

Expected: no failures. The MPS-only RNG test has one skip whenever MPS is unavailable and no skip on the target M3 Max only when its runtime reports MPS available.

- [ ] **Step 9: Commit the doctor and CLI**

```bash
git add pyproject.toml src/silent_cascade/doctor.py src/silent_cascade/cli.py \
  tests/unit/test_doctor.py tests/integration/test_cli_doctor.py \
  tests/regression/test_import_boundaries.py
git commit -m "feat: add offline environment doctor"
```

---

### Task 8: Make Targets, Linux CPU CI, and Protocol-Facing Documentation

**Files:**

- Create: `Makefile`
- Create: `.github/workflows/ci.yml`
- Modify: `README.md`
- Create: `docs/PLAN.md`
- Create: `docs/research-boundary.md`
- Create: `tests/integration/test_phase0_repository.py`

**Interfaces:**

- Consumes: every Phase 0 command and test from Tasks 1–7.
- Produces: working `make setup`, `make doctor`, `make lint`, `make test`, `make smoke`, and `make ci`; Linux CPU CI; conservative public documentation; and the complete Phase 0 gate.

- [ ] **Step 1: Write the failing repository-contract test**

Create `tests/integration/test_phase0_repository.py`:

```python
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_phase0_repository_exposes_only_working_targets_and_commands() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(
        re.findall(r"^([A-Za-z0-9][A-Za-z0-9_-]*):(?:\s|$)", makefile, re.MULTILINE)
    )
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
```

- [ ] **Step 2: Run the repository-contract test to verify it fails**

Run:

```bash
uv run pytest tests/integration/test_phase0_repository.py -q
```

Expected: both tests fail because the Makefile, plan index, and CI workflow do not exist and the README does not yet document the working doctor.

- [ ] **Step 3: Add only the working Phase 0 Make targets**

Create `Makefile` with literal tab-indented recipes:

```make
.PHONY: setup doctor lint test smoke ci

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

ci: lint test doctor
```

Do not add a target named in a deferred phase. Each additional target enters the Makefile in the phase that supplies its real implementation.

- [ ] **Step 4: Add locked Linux CPU CI**

Create `.github/workflows/ci.yml`:

```yaml
name: ci

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  cpu:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          version: "0.11.29"
          python-version: "3.12"
          enable-cache: true
      - name: Install locked core and development dependencies
        run: uv sync --locked --group dev
      - name: Run Phase 0 quality gate
        run: make ci
      - name: Build source and wheel artifacts
        run: uv build
```

Core CI installs no optional extra. It treats unavailable MPS as a reported capability, not an error.

- [ ] **Step 5: Publish only truthful Phase 0 documentation**

Replace `README.md` with:

````markdown
# Silent Cascade

Silent Cascade is a public research project testing persistent, learned event-flow computation on the Observation-Free Deadline benchmark.

The repository is in Phase 0 bootstrap and reports no benchmark results. The project tests a bounded computational mechanism; it makes no claim about consciousness, sentience, biological fidelity, or general intelligence.

## Current working path

```bash
uv sync --locked --group dev
uv run silent-cascade doctor
make ci
```

The primary path is offline at runtime and uses exactly zero foundation-model calls. Qwen is optional, isolated, and outside the scientific path.

## Protocol

- [Canonical design specification](docs/superpowers/specs/2026-08-30-silent-cascade-design.md)
- [Implementation-plan index](docs/PLAN.md)
- [Research and claim boundary](docs/research-boundary.md)
- [Approved deviations](docs/deviations.md)

Only the Phase 0 `doctor` command is implemented. Commands for data generation, training, evaluation, intervention, replay, reporting, and demos enter the repository with their owning implementation phase.
````

When placing this nested Markdown in the real README, use a normal fenced `bash` block; do not copy the enclosing plan fence.

Create `docs/PLAN.md`:

```markdown
# Silent Cascade Delivery Index

The canonical requirements live in the [design specification](superpowers/specs/2026-08-30-silent-cascade-design.md). This file is navigation only and never duplicates or overrides that specification.

| Phase | Plan | Gate |
|---|---|---|
| 0 — Bootstrap | [Phase 0 implementation plan](superpowers/plans/2026-08-30-phase-0-bootstrap.md) | `uv sync --locked --group dev`, `make ci`, and `uv run silent-cascade doctor` |
| 1 — Generator and oracle | Governed by a separate plan after the Phase 0 gate | Phase 0 must pass first |
| 2 — Flow and event engine | Governed by a separate plan after the Phase 1 gate | Phase 1 must pass first |
| 3 — Neural components | Governed by a separate plan after the Phase 2 gate | Phase 2 must pass first |
| 4 — Autonomous EventFlow | Governed by a separate plan after the Phase 3 gate | Phase 3 must pass first |
| 5 — Strong baselines | Governed by a separate plan after the Phase 4 gate | Phase 4 must pass first |
| 6A — Freeze preparation | Governed by a separate plan after the Phase 5 gate | Phase 5 must pass first |
| 6B — Final runs | Governed by a separate plan after the explicit freeze checkpoint | Frozen protocol required |
| 7 — Optional showcase | Governed by a separate optional plan after Phase 6 | Scientific release path required |
```

Create `docs/research-boundary.md`:

```markdown
# Research Boundary

Silent Cascade tests whether a bounded, persistent hybrid neural process with learned endogenous event timing is a useful computational mechanism on one synthetic benchmark.

The primary experiment contains no foundation-model calls, network access, or hidden natural-language reasoning. Qwen is an optional post-research language shell and cannot choose memories, create event times, mutate core state, schedule actions, or enter metrics.

A positive result can support only the predeclared benchmark-specific language in the canonical specification. A negative result is scientifically valid. The project does not test or claim consciousness, sentience, inner experience, biological fidelity, or general intelligence.

Version 1 deliberately uses supervised event traces, deterministic typed memory writes, a synthetic symbolic task, and a simple piecewise-exponential flow. Those choices isolate the event-flow mechanism and bound every conclusion.
```

- [ ] **Step 6: Run focused tests and formatting checks**

Run:

```bash
uv run pytest tests/integration/test_phase0_repository.py -q
uv run ruff check .
uv run ruff format --check .
```

Expected: repository-contract tests report `2 passed`; Ruff exits 0 twice.

- [ ] **Step 7: Run the complete Phase 0 gate before committing**

Run:

```bash
uv sync --locked --group dev
make ci
uv run silent-cascade doctor --json
uv run pytest tests/regression/test_import_boundaries.py -q
uv build
```

Expected: locked sync exits 0; lint and formatting are clean; the complete test suite has no failures; doctor exits 0 with `"ok": true`; import-boundary regression reports `1 passed`; sdist and wheel build successfully. Any host that reports MPS unavailable has one expected capability skip; the target M3 Max local gate has no MPS skip only when its runtime reports MPS available.

- [ ] **Step 8: Commit automation and protocol-facing documentation**

```bash
git add Makefile .github/workflows/ci.yml README.md docs/PLAN.md \
  docs/research-boundary.md tests/integration/test_phase0_repository.py
git commit -m "ci: add Phase 0 quality gate"
```

- [ ] **Step 9: Verify the committed Phase 0 state is clean and reproducible**

Run:

```bash
git log --format=%s HEAD
uv sync --locked --group dev
make ci
uv run silent-cascade doctor
git status --short
```

Expected: the log contains each of the eight task commit subjects named by this plan; review/fix commits are allowed. The locked sync, complete CPU/MPS-appropriate gate, and human-readable doctor all exit 0. The final `git status --short` prints nothing, proving those verification commands left the implementation workspace clean.

## Phase 0 Completion Boundary

Phase 0 is complete only when Task 8 Step 9 has fresh passing output and `superpowers:verification-before-completion` confirms it. Do not begin Phase 1 implementation from this plan. After the Phase 0 gate, write and explicitly approve a separate Phase 1 generator-and-oracle plan against the same canonical specification.

## Appendix A: Exact `LICENSE` Content

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
