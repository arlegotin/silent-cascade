"""Exact Phase 2 evidence source closure, separate from immutable Phase 1 scopes."""

import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path
from typing import Literal, Self

import torch
from pydantic import Field, field_validator, model_validator

from silent_cascade.config import ResolvedConfig, resolve_config
from silent_cascade.errors import ProvenanceError
from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at
from silent_cascade.eventflow.config import Phase2Config
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.validation import StrictModel

PHASE2_PLAN_PATH = "docs/superpowers/plans/2026-09-09-phase-2-flow-event-engine.md"
SPEC_PATH = "docs/superpowers/specs/2026-08-30-silent-cascade-design.md"
VALIDATION_MANIFEST_PATH = "manifests/validation/v1/ofd-primary-10000.json"
VALIDATION_FILE_SHA256 = "84926a3217b27b04d7ed3e41038447533636aea06bcdc85b787f148f3d737e07"
PHASE2_CONFIG_SHA256 = "29e4afac2118bdac5a7a01896167805f1fd1f7379a13e5b21566fef2c34fd546"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_SOURCE_BLOB_BYTES = 16 * 1024 * 1024

# Deliberately literal: ancestor initializers and transitive Phase 1 services
# are executed dependencies too. Never expand the frozen Phase 1 tuples.
PHASE2_ENGINE_SOURCE_PATHS = (
    "configs/base.yaml",
    "configs/data/primary.yaml",
    "configs/model/event_flow.yaml",
    "pyproject.toml",
    "scripts/check_phase2_engine.py",
    "scripts/verify_phase2_gate_artifact.py",
    "src/silent_cascade/__init__.py",
    "src/silent_cascade/cli.py",
    "src/silent_cascade/config.py",
    "src/silent_cascade/doctor.py",
    "src/silent_cascade/env/__init__.py",
    "src/silent_cascade/env/config.py",
    "src/silent_cascade/env/episode.py",
    "src/silent_cascade/env/generator.py",
    "src/silent_cascade/env/invariants.py",
    "src/silent_cascade/env/leakage.py",
    "src/silent_cascade/env/oracle.py",
    "src/silent_cascade/env/reproducibility.py",
    "src/silent_cascade/env/reward.py",
    "src/silent_cascade/env/services.py",
    "src/silent_cascade/env/timing.py",
    "src/silent_cascade/errors.py",
    "src/silent_cascade/eventflow/__init__.py",
    "src/silent_cascade/eventflow/archive_io.py",
    "src/silent_cascade/eventflow/checkpoint.py",
    "src/silent_cascade/eventflow/checkpoint_rng.py",
    "src/silent_cascade/eventflow/checkpoint_state.py",
    "src/silent_cascade/eventflow/config.py",
    "src/silent_cascade/eventflow/engine.py",
    "src/silent_cascade/eventflow/evidence.py",
    "src/silent_cascade/eventflow/flow.py",
    "src/silent_cascade/eventflow/guards.py",
    "src/silent_cascade/eventflow/invariants.py",
    "src/silent_cascade/eventflow/jumps.py",
    "src/silent_cascade/eventflow/protocols.py",
    "src/silent_cascade/eventflow/provenance.py",
    "src/silent_cascade/eventflow/replay.py",
    "src/silent_cascade/eventflow/scheduling.py",
    "src/silent_cascade/eventflow/scripted.py",
    "src/silent_cascade/eventflow/state.py",
    "src/silent_cascade/hashing.py",
    "src/silent_cascade/io.py",
    "src/silent_cascade/logging/__init__.py",
    "src/silent_cascade/logging/crash_bundle.py",
    "src/silent_cascade/logging/manifest.py",
    "src/silent_cascade/logging/runtime_diagnostics.py",
    "src/silent_cascade/logging/trace.py",
    "src/silent_cascade/memory/__init__.py",
    "src/silent_cascade/memory/provenance.py",
    "src/silent_cascade/memory/store.py",
    "src/silent_cascade/provenance.py",
    "src/silent_cascade/rng.py",
    "src/silent_cascade/schemas.py",
    "src/silent_cascade/train/__init__.py",
    "src/silent_cascade/train/pilot_cli.py",
    "src/silent_cascade/validation.py",
    "uv.lock",
)


def parse_gate_config(raw: str) -> Phase2Config:
    """Restore the frozen integer allocation keys lost at the JSON boundary.

    This is a Phase 2 archive codec, not a change to Phase 1 configuration or
    hashing. Exact canonical bytes are still required after strict validation.
    """
    values = json.loads(raw)
    if not isinstance(values, dict) or not isinstance(values.get("data"), dict):
        raise ValueError("full gate configuration requires a data object")
    expected = {
        "iid_primary",
        "ood_depth",
        "ood_short_delay",
        "ood_long_delay",
        "distractor_flood",
        "clock_parent_episodes",
        "clock_10x_episodes",
    }
    if not isinstance(values["data"].get("phase1_gate"), dict):
        raise ValueError("full gate configuration requires frozen allocation mappings")
    allocation = values["data"]["phase1_gate"]
    if set(allocation) != expected or any(
        not isinstance(counts, dict) for counts in allocation.values()
    ):
        raise ValueError("full gate configuration requires the exact known allocation mappings")
    for name, counts in allocation.items():
        if isinstance(counts, dict):
            if any(not key.isdecimal() or str(int(key)) != key for key in counts):
                raise ValueError("allocation key must be a canonical integer")
            allocation[name] = {int(key): value for key, value in counts.items()}
    config = Phase2Config.model_validate(values)
    if canonical_json_bytes(config).decode() != raw:
        raise ValueError("full gate configuration must be canonical")
    return config


class Phase2EvidenceProvenanceData(StrictModel):
    """Strict serialized fields, without producer trust/configuration derivation."""

    schema_version: Literal["phase2-evidence-provenance-v1"]
    phase2_plan_base_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_dirty: bool
    source_paths: tuple[str, ...]
    source_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    specification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approved_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_canonical_json: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_manifest_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_manifest_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    python_version: str = Field(pattern=r"^3\.12\.[0-9]+$")
    torch_version: str = Field(min_length=1, max_length=128)
    platform: str = Field(min_length=1, max_length=512)
    device: Literal["cpu"]
    foundation_model_calls: Literal[0]

    @field_validator("foundation_model_calls", mode="before")
    @classmethod
    def exact_zero(cls, value: object) -> object:
        if type(value) is not int or value != 0:
            raise ValueError("foundation_model_calls must be an exact integer zero")
        return value


class Phase2EvidenceProvenance(Phase2EvidenceProvenanceData):
    @model_validator(mode="after")
    def bind_closed_sources_and_config(self) -> Self:
        if self.source_paths != PHASE2_ENGINE_SOURCE_PATHS:
            raise ValueError("Phase 2 source paths must equal the exact frozen scope")
        config = parse_gate_config(self.config_canonical_json)
        raw = canonical_json_bytes(config)
        if raw.decode() != self.config_canonical_json or sha256_bytes(raw) != self.config_sha256:
            raise ValueError("Phase 2 canonical configuration hash mismatch")
        if self.config_sha256 != PHASE2_CONFIG_SHA256:
            raise ValueError("gate requires the fixed full Phase 2 configuration")
        return self


def _git(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ProvenanceError("Phase 2 Git trust anchor is invalid") from error


def regular_git_blob(repo: Path, revision: str, path: str) -> bytes:
    """Read an immutable, size-bounded, regular historical Git blob."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ProvenanceError("source revision must be a full lowercase commit ID")
    if not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise ProvenanceError("source path must be repository relative")
    entry = _git(repo, "ls-tree", "-z", revision, "--", path)
    try:
        metadata, actual = entry.rstrip(b"\0").split(b"\t")
        mode, kind, oid = metadata.split()
    except ValueError as error:
        raise ProvenanceError("historical source blob is missing") from error
    if actual != path.encode() or mode not in {b"100644", b"100755"} or kind != b"blob":
        raise ProvenanceError("historical source must be a regular Git blob")
    if int(_git(repo, "cat-file", "-s", oid.decode())) > MAX_SOURCE_BLOB_BYTES:
        raise ProvenanceError("historical source blob exceeds byte limit")
    return _git(repo, "cat-file", "blob", oid.decode())


def _source_digest(repo: Path, revision: str) -> str:
    digest = hashlib.sha256(b"silent-cascade/phase2/source-tree/v1\0")
    digest.update(len(PHASE2_ENGINE_SOURCE_PATHS).to_bytes(4, "big"))
    for path in PHASE2_ENGINE_SOURCE_PATHS:
        raw = regular_git_blob(repo, revision, path)
        encoded = path.encode()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def collect_phase2_evidence_provenance(
    resolved: ResolvedConfig[Phase2Config],
    *,
    repo_root: Path,
    manifest_path: Path,
    manifest_raw: bytes,
    manifest_payload_sha256: str,
    expected_source_commit: str,
    expected_plan_base_revision: str,
) -> Phase2EvidenceProvenance:
    """Authenticate the executing checkout before and after streaming collection."""
    for revision in (expected_source_commit, expected_plan_base_revision):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ProvenanceError("expected revision must be a full lowercase commit ID")
        if (
            _git(repo_root, "rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()
            != revision
        ):
            raise ProvenanceError("expected revision did not resolve exactly")
    if _git(repo_root, "rev-parse", "HEAD").decode().strip() != expected_source_commit:
        raise ProvenanceError("collector HEAD differs from expected source commit")
    if _git(repo_root, "status", "--porcelain", "--untracked-files=normal"):
        raise ProvenanceError("Phase 2 collection requires clean source and committed plan")
    _git(
        repo_root,
        "merge-base",
        "--is-ancestor",
        expected_plan_base_revision,
        expected_source_commit,
    )
    latest_plan = (
        _git(repo_root, "log", "-1", "--format=%H", expected_source_commit, "--", PHASE2_PLAN_PATH)
        .decode()
        .strip()
    )
    if latest_plan != expected_plan_base_revision:
        raise ProvenanceError("plan base must be the latest committed approved plan")
    plan = regular_git_blob(repo_root, expected_source_commit, PHASE2_PLAN_PATH)
    if b"User-approved; completion tracked in docs/PLAN.md" not in plan:
        raise ProvenanceError("approved plan header is not normalized")
    if plan != regular_git_blob(repo_root, expected_plan_base_revision, PHASE2_PLAN_PATH):
        raise ProvenanceError("approved plan differs from the plan base")
    try:
        relative = manifest_path.absolute().relative_to(repo_root.absolute()).as_posix()
    except ValueError as error:
        raise ProvenanceError("manifest must be committed inside the source repository") from error
    if regular_git_blob(repo_root, expected_source_commit, relative) != manifest_raw:
        raise ProvenanceError("manifest differs from its historical Git blob")
    # Bind the importing installation too: a caller cannot name a different
    # clean checkout as the source of already imported Python implementation.
    executing_root = Path(__file__).absolute().parents[3]
    source_roots = {repo_root.absolute(), executing_root}
    # Match disk bytes even for assume-unchanged/skip-worktree Git flags.
    for path in (*PHASE2_ENGINE_SOURCE_PATHS, PHASE2_PLAN_PATH, SPEC_PATH):
        historical = regular_git_blob(repo_root, expected_source_commit, path)
        for root in source_roots:
            try:
                with archive_parent(root / path, error_factory=ProvenanceError) as (parent, name):
                    raw = read_archive_at(
                        parent, name, max_bytes=MAX_SOURCE_BLOB_BYTES, error_factory=ProvenanceError
                    )
            except OSError as error:
                raise ProvenanceError("source path is not a readable regular file") from error
            if raw != historical:
                raise ProvenanceError("executing source differs from historical Git blob")
    if canonical_json_bytes(resolved.config) != resolved.canonical_json:
        raise ProvenanceError("resolved configuration is inconsistent")
    checkout_config = resolve_config(
        Phase2Config,
        [
            repo_root / path
            for path in (
                "configs/base.yaml",
                "configs/data/primary.yaml",
                "configs/model/event_flow.yaml",
            )
        ],
    )
    if checkout_config.canonical_json != resolved.canonical_json:
        raise ProvenanceError("resolved configuration differs from authenticated YAML inputs")
    return Phase2EvidenceProvenance(
        schema_version="phase2-evidence-provenance-v1",
        phase2_plan_base_revision=expected_plan_base_revision,
        source_commit=expected_source_commit,
        source_dirty=False,
        source_paths=PHASE2_ENGINE_SOURCE_PATHS,
        source_tree_sha256=_source_digest(repo_root, expected_source_commit),
        specification_sha256=sha256_bytes(
            regular_git_blob(repo_root, expected_source_commit, SPEC_PATH)
        ),
        approved_plan_sha256=sha256_bytes(plan),
        config_canonical_json=resolved.canonical_json.decode(),
        config_sha256=resolved.sha256,
        validation_manifest_payload_sha256=manifest_payload_sha256,
        validation_manifest_file_sha256=sha256_bytes(manifest_raw),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        platform=platform.platform(),
        device="cpu",
        foundation_model_calls=0,
    )
