"""Fail-closed Phase 1 evidence provenance collection."""

import hashlib
import re
import subprocess
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import Field, field_validator, model_validator

from silent_cascade.config import ResolvedConfig
from silent_cascade.env.config import Phase1Config, SplitNamespace
from silent_cascade.errors import ProvenanceError
from silent_cascade.validation import StrictModel

GENERATOR_SOURCE_PATHS = (
    "src/silent_cascade/__init__.py",
    "src/silent_cascade/config.py",
    "src/silent_cascade/env/__init__.py",
    "src/silent_cascade/env/config.py",
    "src/silent_cascade/env/episode.py",
    "src/silent_cascade/env/generator.py",
    "src/silent_cascade/env/invariants.py",
    "src/silent_cascade/env/oracle.py",
    "src/silent_cascade/env/timing.py",
    "src/silent_cascade/errors.py",
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
            "src/silent_cascade/logging/__init__.py",
            "src/silent_cascade/logging/manifest.py",
        )
    )
)
# The Task 14 label records when this scope was introduced. It remains closed
# over every current local import of its scoped modules.
TASK14_ANALYSIS_SOURCE_PATHS = tuple(
    sorted(
        (
            *GENERATOR_SOURCE_PATHS,
            "scripts/check_phase1_reproducibility.py",
            "src/silent_cascade/env/leakage.py",
            "src/silent_cascade/env/reproducibility.py",
            "src/silent_cascade/env/reward.py",
            "src/silent_cascade/env/services.py",
            "src/silent_cascade/io.py",
            "src/silent_cascade/logging/__init__.py",
            "src/silent_cascade/logging/manifest.py",
            "src/silent_cascade/provenance.py",
        )
    )
)
PHASE1_PLAN_PATH = Path("docs/superpowers/plans/2026-08-30-phase-1-generator-oracle.md")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_FRAME = b"silent-cascade/source-tree/v1\0"
_PUBLIC_ID_SEED_FRAME = b"silent-cascade/ofd-v1/public-id-seed-fingerprint/v1\0"
_CONSTRUCTION_TOKEN_SEQUENCE_DOMAIN = b"silent-cascade/ofd-v1/construction-token-sequence/v1"
_PUBLIC_ID_SEQUENCE_DOMAIN = b"silent-cascade/ofd-v1/public-id-sequence/v1"
_MAX_ACCEPTED_DRAWS = 100_000


class AcceptedAttemptRun(StrictModel):
    first_draw_index: int = Field(ge=0)
    draw_count: int = Field(gt=0)
    accepted_attempt: int = Field(ge=0, lt=1_000)


class ConstructionNamespaceEvidence(StrictModel):
    schema_version: Literal["construction-namespace-evidence-v1"]
    generation_mode: Literal["matched", "independent"]
    public_id_seed: int = Field(ge=0, lt=2**128)
    accepted_draw_count: int = Field(gt=0, le=_MAX_ACCEPTED_DRAWS)
    accepted_attempt_runs: tuple[AcceptedAttemptRun, ...]
    rejected_draw_count: int = Field(ge=0)
    generation_attempt_count: int = Field(gt=0)
    seed_token_count: int = Field(gt=0)
    seed_token_sequence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed_token_collision_count: Literal[0]
    base_public_id_count: int = Field(gt=0)
    clock_public_id_count: int = Field(ge=0)
    total_public_id_count: int = Field(gt=0)
    public_id_sequence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_id_collision_count: Literal[0]

    @model_validator(mode="after")
    def require_derived_namespace_counts(self) -> "ConstructionNamespaceEvidence":
        next_index = 0
        previous_attempt: int | None = None
        rejected = 0
        for run in self.accepted_attempt_runs:
            if run.first_draw_index != next_index or run.accepted_attempt == previous_attempt:
                raise ValueError("accepted attempt runs must be contiguous and canonical")
            next_index += run.draw_count
            rejected += run.draw_count * run.accepted_attempt
            previous_attempt = run.accepted_attempt
        if not self.accepted_attempt_runs or next_index != self.accepted_draw_count:
            raise ValueError("accepted attempt runs must exactly cover accepted draws")
        if self.rejected_draw_count != rejected:
            raise ValueError("rejected draw count must be derived from accepted attempts")
        if self.generation_attempt_count != self.accepted_draw_count + rejected:
            raise ValueError("generation attempt count must include accepted and rejected draws")
        tokens_per_draw = 20 if self.generation_mode == "matched" else 7
        base_ids_per_draw = 4 if self.generation_mode == "matched" else 1
        if self.seed_token_count != self.accepted_draw_count * tokens_per_draw:
            raise ValueError("construction token count law does not match generation mode")
        if self.base_public_id_count != self.accepted_draw_count * base_ids_per_draw:
            raise ValueError("base public ID count law does not match generation mode")
        if self.total_public_id_count != (self.base_public_id_count + self.clock_public_id_count):
            raise ValueError("total public ID count must equal base plus clock IDs")
        return self


class SourceTreeFingerprint(StrictModel):
    frame_version: Literal["sc-source-tree-v1"]
    scope: Literal["generator", "phase1_analysis", "phase1_task14_analysis"]
    paths: tuple[str, ...]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("paths")
    @classmethod
    def require_sorted_unique_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or tuple(sorted(value)) != value or len(set(value)) != len(value):
            raise ValueError("source paths must be nonempty, sorted, and unique")
        return value

    @model_validator(mode="after")
    def require_exact_declared_scope(self) -> "SourceTreeFingerprint":
        expected_paths = {
            "generator": GENERATOR_SOURCE_PATHS,
            "phase1_analysis": PHASE1_ANALYSIS_SOURCE_PATHS,
            "phase1_task14_analysis": TASK14_ANALYSIS_SOURCE_PATHS,
        }[self.scope]
        if self.paths != expected_paths:
            raise ValueError("source fingerprint paths must equal the exact frozen scope")
        return self


class LeakageAuditEvidenceAnchor(StrictModel):
    """Independently supplied identity for one publishable leakage experiment."""

    schema_version: Literal["phase1-leakage-audit-anchor-v1"]
    profile: Literal["test", "phase1_gate"]
    allocation_id: str = Field(min_length=1)
    allocation_or_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    descriptor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    suite_path_denominators: dict[str, int]
    clock_pair_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    clock_scale_pair_counts: dict[Literal["scale_0_1x", "scale_10x"], int]
    episode_count: int = Field(gt=0, multiple_of=4)

    @model_validator(mode="after")
    def require_complete_profile(self) -> "LeakageAuditEvidenceAnchor":
        if (
            not self.suite_path_denominators
            or any(
                not key or type(value) is not int or value <= 0
                for key, value in self.suite_path_denominators.items()
            )
            or set(self.clock_scale_pair_counts) != {"scale_0_1x", "scale_10x"}
            or any(
                type(value) is not int or value <= 0
                for value in self.clock_scale_pair_counts.values()
            )
        ):
            raise ValueError("leakage audit anchor counts must be complete positive integers")
        return self


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
    leakage_audit: LeakageAuditEvidenceAnchor | None = None
    foundation_model_calls: Literal[0] = 0

    @field_validator("foundation_model_calls", mode="before")
    @classmethod
    def require_exact_zero_foundation_calls(cls, value: object) -> object:
        if type(value) is not int or value != 0:
            raise ValueError("foundation_model_calls must be an exact integer zero")
        return value

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

    @model_validator(mode="after")
    def require_exact_source_fingerprints(self) -> "EvidenceProvenance":
        if (
            self.generator_source.scope != "generator"
            or self.generator_source.paths != GENERATOR_SOURCE_PATHS
            or self.analysis_source.scope not in {"phase1_analysis", "phase1_task14_analysis"}
            or self.analysis_source.paths
            != {
                "phase1_analysis": PHASE1_ANALYSIS_SOURCE_PATHS,
                "phase1_task14_analysis": TASK14_ANALYSIS_SOURCE_PATHS,
            }[self.analysis_source.scope]
        ):
            raise ValueError("evidence provenance requires exact frozen source fingerprints")
        return self


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
    if not paths or len(set(paths)) != len(paths) or tuple(sorted(paths)) != paths:
        raise ProvenanceError("source paths must be nonempty, unique, and sorted")
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


def accepted_attempt_runs(attempts: Iterable[int]) -> tuple[AcceptedAttemptRun, ...]:
    """Return the unique run-length encoding of accepted draws in source order."""
    if not isinstance(attempts, Iterable):
        raise TypeError("attempts must be iterable")
    result: list[AcceptedAttemptRun] = []
    first = 0
    previous: int | None = None
    count = 0
    for draw_index, attempt in enumerate(attempts):
        if type(attempt) is not int or attempt < 0:
            raise ValueError("accepted attempt must be a nonnegative exact integer")
        if attempt >= 1_000:
            raise ValueError("accepted attempt must be below 1000")
        if draw_index >= _MAX_ACCEPTED_DRAWS:
            raise ValueError("accepted draw count exceeds the bounded Phase 1 corpus")
        if previous is None:
            first = draw_index
            previous = attempt
            count = 1
        elif attempt == previous:
            count += 1
        else:
            result.append(
                AcceptedAttemptRun(
                    first_draw_index=first,
                    draw_count=count,
                    accepted_attempt=previous,
                )
            )
            first = draw_index
            previous = attempt
            count = 1
    if previous is None:
        raise ValueError("accepted attempts must be nonempty")
    result.append(
        AcceptedAttemptRun(
            first_draw_index=first,
            draw_count=count,
            accepted_attempt=previous,
        )
    )
    return tuple(result)


def _token_bytes(value: object) -> bytes:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("construction tokens must be lowercase 32-byte hexadecimal values")
    return bytes.fromhex(value)


def _public_id_bytes(value: object) -> bytes:
    if type(value) is not str:
        raise ValueError("public IDs must be canonical UUID strings")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as error:
        raise ValueError("public IDs must be canonical UUID strings") from error
    if str(parsed) != value:
        raise ValueError("public IDs must be canonical UUID strings")
    return parsed.bytes


def _framed_sequence_sha256(domain: bytes, values: Iterable[bytes], count: int) -> str:
    digest = hashlib.sha256()
    digest.update(len(domain).to_bytes(4, "big"))
    digest.update(domain)
    digest.update(count.to_bytes(8, "big"))
    seen = 0
    for value in values:
        digest.update(len(value).to_bytes(4, "big"))
        digest.update(value)
        seen += 1
    if seen != count:
        raise ValueError("sequence count does not match its framed values")
    return digest.hexdigest()


def construction_token_sequence_sha256(tokens: Iterable[str]) -> str:
    """Hash an ordered construction-token sequence with explicit framing."""
    values = tuple(_token_bytes(token) for token in tokens)
    return _framed_sequence_sha256(_CONSTRUCTION_TOKEN_SEQUENCE_DOMAIN, values, len(values))


def public_id_sequence_sha256(public_ids: Iterable[str]) -> str:
    """Hash an ordered public-ID sequence with explicit framing."""
    values = tuple(_public_id_bytes(public_id) for public_id in public_ids)
    return _framed_sequence_sha256(_PUBLIC_ID_SEQUENCE_DOMAIN, values, len(values))


def _sequence_digest(domain: bytes, count: int) -> Any:
    digest = hashlib.sha256()
    digest.update(len(domain).to_bytes(4, "big"))
    digest.update(domain)
    digest.update(count.to_bytes(8, "big"))
    return digest


class ConstructionNamespaceBuilder:
    """Stream one bounded corpus into fixed-width collision evidence."""

    def __init__(
        self,
        *,
        generation_mode: Literal["matched", "independent"],
        public_id_seed: int,
        accepted_draw_count: int,
        clock_public_id_count: int,
    ) -> None:
        if generation_mode not in ("matched", "independent"):
            raise ValueError("generation mode must be matched or independent")
        if type(public_id_seed) is not int or not 0 <= public_id_seed < 2**128:
            raise ProvenanceError("public ID seed must be an exact 128-bit unsigned integer")
        if (
            type(accepted_draw_count) is not int
            or not 0 < accepted_draw_count <= _MAX_ACCEPTED_DRAWS
            or type(clock_public_id_count) is not int
            or clock_public_id_count < 0
        ):
            raise ValueError("namespace corpus counts are outside their bounded ranges")
        self._generation_mode = generation_mode
        self._public_id_seed = public_id_seed
        self._accepted_draw_count = accepted_draw_count
        self._tokens_per_draw = 20 if generation_mode == "matched" else 7
        self._base_ids_per_draw = 4 if generation_mode == "matched" else 1
        self._token_count = accepted_draw_count * self._tokens_per_draw
        self._base_id_count = accepted_draw_count * self._base_ids_per_draw
        self._clock_id_count = clock_public_id_count
        self._total_id_count = self._base_id_count + clock_public_id_count
        self._attempts = np.empty(accepted_draw_count, dtype=np.uint16)
        self._tokens = np.empty(self._token_count, dtype="V32")
        self._public_ids = np.empty(self._total_id_count, dtype="V16")
        self._token_digest = _sequence_digest(
            _CONSTRUCTION_TOKEN_SEQUENCE_DOMAIN, self._token_count
        )
        self._public_id_digest = _sequence_digest(_PUBLIC_ID_SEQUENCE_DOMAIN, self._total_id_count)
        self._draw_position = 0
        self._token_position = 0
        self._public_id_position = 0
        self._finalized = False

    def _append_token(self, token: str) -> None:
        if self._token_position >= self._token_count:
            raise ValueError("construction token sequence exceeds its expected count")
        raw = _token_bytes(token)
        self._tokens[self._token_position] = raw
        self._token_digest.update((32).to_bytes(4, "big"))
        self._token_digest.update(raw)
        self._token_position += 1

    def _append_public_id(self, public_id: str) -> None:
        if self._public_id_position >= self._total_id_count:
            raise ValueError("public ID sequence exceeds its expected count")
        raw = _public_id_bytes(public_id)
        self._public_ids[self._public_id_position] = raw
        self._public_id_digest.update((16).to_bytes(4, "big"))
        self._public_id_digest.update(raw)
        self._public_id_position += 1

    def add_draw(
        self,
        accepted_attempt: int,
        seed_tokens: Sequence[str],
        base_public_ids: Sequence[str],
    ) -> None:
        if self._finalized:
            raise RuntimeError("cannot add to finalized namespace evidence")
        if self._draw_position >= self._accepted_draw_count:
            raise ValueError("accepted draw sequence exceeds its expected count")
        if type(accepted_attempt) is not int or not 0 <= accepted_attempt < 1_000:
            raise ValueError("accepted attempt must be an exact integer below 1000")
        if len(seed_tokens) != self._tokens_per_draw:
            raise ValueError("construction token count law does not match generation mode")
        if len(base_public_ids) != self._base_ids_per_draw:
            raise ValueError("base public ID count law does not match generation mode")
        self._attempts[self._draw_position] = accepted_attempt
        for token in seed_tokens:
            self._append_token(token)
        for public_id in base_public_ids:
            self._append_public_id(public_id)
        self._draw_position += 1

    def add_clock_public_id(self, public_id: str) -> None:
        if self._finalized:
            raise RuntimeError("cannot add to finalized namespace evidence")
        if self._public_id_position < self._base_id_count:
            raise ValueError("clock public IDs must follow every base public ID")
        self._append_public_id(public_id)

    def finalize(self) -> ConstructionNamespaceEvidence:
        if self._finalized:
            raise RuntimeError("namespace evidence has already been finalized")
        self._finalized = True
        if (
            self._draw_position != self._accepted_draw_count
            or self._token_position != self._token_count
            or self._public_id_position != self._total_id_count
        ):
            raise ValueError("construction namespace evidence is incomplete")
        sorted_tokens = np.sort(self._tokens.copy())
        token_collisions = int(np.count_nonzero(sorted_tokens[1:] == sorted_tokens[:-1]))
        sorted_public_ids = np.sort(self._public_ids.copy())
        public_id_collisions = int(
            np.count_nonzero(sorted_public_ids[1:] == sorted_public_ids[:-1])
        )
        if token_collisions:
            raise ProvenanceError(
                "construction token collision",
                context={"collision_count": token_collisions},
            )
        if public_id_collisions:
            raise ProvenanceError(
                "public ID collision",
                context={"collision_count": public_id_collisions},
            )
        runs = accepted_attempt_runs(int(value) for value in self._attempts)
        rejected_count = sum(run.draw_count * run.accepted_attempt for run in runs)
        return ConstructionNamespaceEvidence(
            schema_version="construction-namespace-evidence-v1",
            generation_mode=self._generation_mode,
            public_id_seed=self._public_id_seed,
            accepted_draw_count=self._accepted_draw_count,
            accepted_attempt_runs=runs,
            rejected_draw_count=rejected_count,
            generation_attempt_count=self._accepted_draw_count + rejected_count,
            seed_token_count=self._token_count,
            seed_token_sequence_sha256=self._token_digest.hexdigest(),
            seed_token_collision_count=0,
            base_public_id_count=self._base_id_count,
            clock_public_id_count=self._clock_id_count,
            total_public_id_count=self._total_id_count,
            public_id_sequence_sha256=self._public_id_digest.hexdigest(),
            public_id_collision_count=0,
        )


def build_construction_namespace_evidence(
    *,
    generation_mode: Literal["matched", "independent"],
    public_id_seed: int,
    accepted_attempts: Iterable[int],
    seed_tokens: Iterable[str],
    base_public_ids: Iterable[str],
    clock_public_ids: Iterable[str],
) -> ConstructionNamespaceEvidence:
    """Authenticate actual draw attempts, tokens, and IDs under bounded storage."""
    attempt_values = tuple(accepted_attempts)
    base_identifiers = tuple(base_public_ids)
    clock_identifiers = tuple(clock_public_ids)
    builder = ConstructionNamespaceBuilder(
        generation_mode=generation_mode,
        public_id_seed=public_id_seed,
        accepted_draw_count=len(attempt_values),
        clock_public_id_count=len(clock_identifiers),
    )
    tokens_per_draw = 20 if generation_mode == "matched" else 7
    base_ids_per_draw = 4 if generation_mode == "matched" else 1
    if len(base_identifiers) != len(attempt_values) * base_ids_per_draw:
        raise ProvenanceError("base public ID count does not match accepted draws")
    token_iterator = iter(seed_tokens)
    for index, attempt in enumerate(attempt_values):
        tokens = tuple(next(token_iterator, None) for _ in range(tokens_per_draw))
        if any(token is None for token in tokens):
            raise ValueError("construction token sequence is incomplete")
        first_id = index * base_ids_per_draw
        builder.add_draw(
            attempt,
            tokens,  # type: ignore[arg-type]
            base_identifiers[first_id : first_id + base_ids_per_draw],
        )
    if next(token_iterator, None) is not None:
        raise ValueError("construction token sequence exceeds its expected count")
    for public_id in clock_identifiers:
        builder.add_clock_public_id(public_id)
    return builder.finalize()


def require_namespace_evidence_provenance(
    evidence: ConstructionNamespaceEvidence,
    provenance: EvidenceProvenance,
) -> None:
    """Require namespace evidence to bind the same raw seed and generation mode."""
    if not isinstance(evidence, ConstructionNamespaceEvidence) or not isinstance(
        provenance, EvidenceProvenance
    ):
        raise TypeError("namespace evidence and provenance must be strict models")
    if (
        evidence.generation_mode != provenance.generation_mode
        or public_id_seed_sha256(evidence.public_id_seed) != provenance.public_id_seed_sha256
    ):
        raise ValueError("construction namespace evidence does not match provenance")


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


def _source_dirty(
    repo_root: Path, source_paths: tuple[Path, ...], analysis_paths: tuple[str, ...]
) -> bool:
    candidates = set(GENERATOR_SOURCE_PATHS) | set(analysis_paths)
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
    analysis_scope: Literal["phase1_analysis", "phase1_task14_analysis"] = "phase1_task14_analysis",
) -> EvidenceProvenance:
    """Collect immutable scientific inputs without broad source discovery."""
    if not isinstance(resolved, ResolvedConfig) or not isinstance(repo_root, Path):
        raise TypeError("resolved must be ResolvedConfig and repo_root must be a Path")
    if not isinstance(split_namespace, SplitNamespace):
        raise ProvenanceError("split_namespace must be a SplitNamespace")
    if generation_mode not in ("matched", "independent") or not allocation_id:
        raise ProvenanceError("generation mode and allocation ID are required")
    analysis_paths = {
        "phase1_analysis": PHASE1_ANALYSIS_SOURCE_PATHS,
        "phase1_task14_analysis": TASK14_ANALYSIS_SOURCE_PATHS,
    }[analysis_scope]
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
        source_dirty=_source_dirty(repo_root, resolved.source_paths, analysis_paths),
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
            scope=analysis_scope,
            paths=analysis_paths,
            sha256=source_tree_sha256(repo_root, analysis_paths),
        ),
        root_seed=root_seed,
        public_id_seed_sha256=public_id_seed_sha256(public_id_seed),
        analysis_seeds=dict(analysis_seeds),
        foundation_model_calls=resolved.config.runtime.primary_foundation_model_calls,
    )


def collect_final_phase1_provenance(
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
    """Collect provenance against the complete final Phase 1 analysis source scope."""
    return collect_evidence_provenance(
        resolved,
        repo_root=repo_root,
        generation_mode=generation_mode,
        allocation_id=allocation_id,
        split_namespace=split_namespace,
        root_seed=root_seed,
        public_id_seed=public_id_seed,
        analysis_seeds=analysis_seeds,
        analysis_scope="phase1_analysis",
    )
