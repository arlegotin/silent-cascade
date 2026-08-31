"""Fail-closed Phase 1 evidence provenance collection."""

import hashlib
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, field_validator

from silent_cascade.config import ResolvedConfig
from silent_cascade.env.config import Phase1Config, SplitNamespace
from silent_cascade.errors import ProvenanceError
from silent_cascade.validation import StrictModel

GENERATOR_SOURCE_PATHS = (
    "src/silent_cascade/config.py",
    "src/silent_cascade/env/config.py",
    "src/silent_cascade/env/episode.py",
    "src/silent_cascade/env/generator.py",
    "src/silent_cascade/env/invariants.py",
    "src/silent_cascade/env/oracle.py",
    "src/silent_cascade/hashing.py",
    "src/silent_cascade/rng.py",
    "src/silent_cascade/schemas.py",
    "src/silent_cascade/validation.py",
)
PHASE1_ANALYSIS_SOURCE_PATHS = tuple(
    sorted(
        (
            *GENERATOR_SOURCE_PATHS,
            "scripts/check_phase1_reproducibility.py",
            "scripts/verify_phase1_gate_artifacts.py",
            "src/silent_cascade/provenance.py",
            "src/silent_cascade/env/reward.py",
            "src/silent_cascade/env/leakage.py",
            "src/silent_cascade/env/reproducibility.py",
            "src/silent_cascade/env/services.py",
            "src/silent_cascade/io.py",
            "src/silent_cascade/logging/manifest.py",
        )
    )
)
PHASE1_PLAN_PATH = Path("docs/superpowers/plans/2026-08-30-phase-1-generator-oracle.md")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_FRAME = b"silent-cascade/source-tree/v1\0"
_PUBLIC_ID_SEED_FRAME = b"silent-cascade/ofd-v1/public-id-seed-fingerprint/v1\0"


class SourceTreeFingerprint(StrictModel):
    frame_version: Literal["sc-source-tree-v1"]
    scope: Literal["generator", "phase1_analysis"]
    paths: tuple[str, ...]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("paths")
    @classmethod
    def require_sorted_unique_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or tuple(sorted(value)) != value or len(set(value)) != len(value):
            raise ValueError("source paths must be nonempty, sorted, and unique")
        return value


class EvidenceProvenance(StrictModel):
    schema_version: Literal["phase1-evidence-provenance-v1"]
    plan_base_revision: str = Field(min_length=7)
    source_commit: str = Field(min_length=7)
    source_dirty: bool
    generator_version: Literal["ofd-v1"]
    generation_mode: Literal["matched", "independent"]
    allocation_id: str = Field(min_length=1)
    split_namespace: SplitNamespace
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_source: SourceTreeFingerprint
    analysis_source: SourceTreeFingerprint
    root_seed: int
    public_id_seed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_seeds: dict[str, int] = Field(default_factory=dict)
    foundation_model_calls: Literal[0] = 0

    @field_validator("plan_base_revision", "source_commit")
    @classmethod
    def require_full_commit_id(cls, value: str) -> str:
        if _COMMIT_PATTERN.fullmatch(value) is None:
            raise ValueError("revision must be a full lowercase commit ID")
        return value

    @field_validator("root_seed")
    @classmethod
    def require_seed_range(cls, value: int) -> int:
        if type(value) is not int or not 0 <= value < 2**128:
            raise ValueError("root_seed must be an exact 128-bit unsigned integer")
        return value

    @field_validator("analysis_seeds")
    @classmethod
    def require_exact_analysis_seeds(cls, value: dict[str, int]) -> dict[str, int]:
        if any(not key or type(seed) is not int for key, seed in value.items()):
            raise ValueError("analysis_seeds must have nonempty keys and exact integer values")
        return dict(sorted(value.items()))


class EvidenceProvenanceCollector(Protocol):
    def __call__(
        self,
        resolved: ResolvedConfig[Phase1Config],
        *,
        repo_root: Path,
        generation_mode: Literal["matched", "independent"],
        allocation_id: str,
        split_namespace: SplitNamespace,
        root_seed: int,
        public_id_seed: int,
        analysis_seeds: Mapping[str, int],
    ) -> EvidenceProvenance: ...


def _safe_relative_path(repo_root: Path, path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts or not path:
        raise ProvenanceError("source path escapes repository root", context={"path": path})
    current = repo_root
    for part in candidate.parts:
        current /= part
        if current.is_symlink():
            raise ProvenanceError("source path contains a symlink", context={"path": path})
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(repo_root.resolve(strict=True))
    except FileNotFoundError as error:
        raise ProvenanceError("source path is missing", context={"path": path}) from error
    except ValueError as error:
        raise ProvenanceError(
            "source path escapes repository root", context={"path": path}
        ) from error
    if not resolved.is_file():
        raise ProvenanceError("source path is not a regular file", context={"path": path})
    return resolved


def source_tree_sha256(repo_root: Path, paths: tuple[str, ...]) -> str:
    """Hash an explicit source list with unambiguous path/content framing."""
    if not isinstance(repo_root, Path) or not isinstance(paths, tuple):
        raise TypeError("repo_root must be a Path and paths must be a tuple")
    if not paths or len(set(paths)) != len(paths):
        raise ProvenanceError("source paths must be nonempty and unique")
    digest = hashlib.sha256()
    digest.update(_SOURCE_FRAME)
    digest.update(len(paths).to_bytes(4, "big"))
    for path in paths:
        if not isinstance(path, str):
            raise ProvenanceError("source path must be a string")
        content = _safe_relative_path(repo_root, path).read_bytes()
        encoded_path = path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(4, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def public_id_seed_sha256(seed: int) -> str:
    if type(seed) is not int or not 0 <= seed < 2**128:
        raise ProvenanceError("public ID seed must be an exact 128-bit unsigned integer")
    return hashlib.sha256(_PUBLIC_ID_SEED_FRAME + seed.to_bytes(16, "big")).hexdigest()


def _git(repo_root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProvenanceError(
            "git provenance resolution failed", context={"reason": str(error)}
        ) from error
    return result.stdout.strip()


def _relative_tracked_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(repo_root.resolve(strict=True)).as_posix()
    except (FileNotFoundError, ValueError) as error:
        raise ProvenanceError(
            "dirty-state source path is missing or escapes repository", context={"path": str(path)}
        ) from error


def _source_dirty(repo_root: Path, source_paths: tuple[Path, ...]) -> bool:
    candidates = set(GENERATOR_SOURCE_PATHS) | set(PHASE1_ANALYSIS_SOURCE_PATHS)
    candidates.update({"pyproject.toml", "uv.lock"})
    candidates.update(_relative_tracked_path(repo_root, path) for path in source_paths)
    for path in candidates:
        _safe_relative_path(repo_root, path)
    output = _git(
        repo_root, "status", "--porcelain=v1", "--untracked-files=all", "--", *sorted(candidates)
    )
    return bool(output)


def collect_evidence_provenance(
    resolved: ResolvedConfig[Phase1Config],
    *,
    repo_root: Path,
    generation_mode: Literal["matched", "independent"],
    allocation_id: str,
    split_namespace: SplitNamespace,
    root_seed: int,
    public_id_seed: int,
    analysis_seeds: Mapping[str, int],
) -> EvidenceProvenance:
    """Collect immutable scientific inputs without broad source discovery."""
    if not isinstance(resolved, ResolvedConfig) or not isinstance(repo_root, Path):
        raise TypeError("resolved must be ResolvedConfig and repo_root must be a Path")
    if not isinstance(split_namespace, SplitNamespace):
        raise ProvenanceError("split_namespace must be a SplitNamespace")
    if generation_mode not in ("matched", "independent") or not allocation_id:
        raise ProvenanceError("generation mode and allocation ID are required")
    if resolved.config.runtime.primary_foundation_model_calls != 0:
        raise ProvenanceError("primary execution must record zero foundation-model calls")
    source_commit = _git(repo_root, "rev-parse", "--verify", "HEAD")
    plan_base_revision = _git(repo_root, "log", "-1", "--format=%H", "--", str(PHASE1_PLAN_PATH))
    if (
        _COMMIT_PATTERN.fullmatch(source_commit) is None
        or _COMMIT_PATTERN.fullmatch(plan_base_revision) is None
    ):
        raise ProvenanceError(
            "repository must contain committed source and approved plan revisions"
        )
    return EvidenceProvenance(
        schema_version="phase1-evidence-provenance-v1",
        plan_base_revision=plan_base_revision,
        source_commit=source_commit,
        source_dirty=_source_dirty(repo_root, resolved.source_paths),
        generator_version="ofd-v1",
        generation_mode=generation_mode,
        allocation_id=allocation_id,
        split_namespace=split_namespace,
        config_sha256=resolved.sha256,
        generator_source=SourceTreeFingerprint(
            frame_version="sc-source-tree-v1",
            scope="generator",
            paths=GENERATOR_SOURCE_PATHS,
            sha256=source_tree_sha256(repo_root, GENERATOR_SOURCE_PATHS),
        ),
        analysis_source=SourceTreeFingerprint(
            frame_version="sc-source-tree-v1",
            scope="phase1_analysis",
            paths=PHASE1_ANALYSIS_SOURCE_PATHS,
            sha256=source_tree_sha256(repo_root, PHASE1_ANALYSIS_SOURCE_PATHS),
        ),
        root_seed=root_seed,
        public_id_seed_sha256=public_id_seed_sha256(public_id_seed),
        analysis_seeds=dict(analysis_seeds),
        foundation_model_calls=resolved.config.runtime.primary_foundation_model_calls,
    )
