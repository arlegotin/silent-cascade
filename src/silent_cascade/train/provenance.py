"""Reviewed executable closure and immutable transformed-corpus provenance."""

import hashlib
import platform
import re
import subprocess
import sys
from pathlib import Path

import torch

from silent_cascade.config import ResolvedConfig, resolve_config
from silent_cascade.errors import ProvenanceError
from silent_cascade.eventflow.archive_io import archive_parent
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.train.checkpoints import _publish_bytes_at
from silent_cascade.train.config import Phase3Config
from silent_cascade.train.curriculum_data import (
    ComponentManifest,
    CurriculumKey,
    make_curriculum_example,
)
from silent_cascade.train.evidence_types import (
    MAX_ARTIFACT_BYTES,
    Phase3EvidenceProvenance,
    read_bytes,
)

PLAN_PATH = "docs/superpowers/plans/2026-09-09-phase-3-neural-components.md"
SPEC_PATH = "docs/superpowers/specs/2026-08-30-silent-cascade-design.md"
PHASE3_SOURCE_PATHS = (
    "configs/base.yaml",
    "configs/data/primary.yaml",
    "configs/model/event_flow.yaml",
    "configs/model/neural_components.yaml",
    "configs/train/one_hop.yaml",
    "configs/train/one_hop_content_v2.yaml",
    "configs/train/smoke.yaml",
    "configs/train/smoke_content_v2.yaml",
    "manifests/validation/phase3/delivery.json",
    "pyproject.toml",
    "scripts/check_phase3_components.py",
    "scripts/verify_phase3_gate_artifact.py",
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
    "src/silent_cascade/eval/__init__.py",
    "src/silent_cascade/eval/compute.py",
    "src/silent_cascade/eventflow/__init__.py",
    "src/silent_cascade/eventflow/archive_io.py",
    "src/silent_cascade/eventflow/checkpoint_rng.py",
    "src/silent_cascade/eventflow/config.py",
    "src/silent_cascade/eventflow/guards.py",
    "src/silent_cascade/eventflow/protocols.py",
    "src/silent_cascade/eventflow/state.py",
    "src/silent_cascade/hashing.py",
    "src/silent_cascade/memory/__init__.py",
    "src/silent_cascade/memory/encoder.py",
    "src/silent_cascade/memory/provenance.py",
    "src/silent_cascade/memory/retrieval.py",
    "src/silent_cascade/memory/store.py",
    "src/silent_cascade/memory/tensor_store.py",
    "src/silent_cascade/models/__init__.py",
    "src/silent_cascade/models/common.py",
    "src/silent_cascade/models/config.py",
    "src/silent_cascade/models/content_loss.py",
    "src/silent_cascade/models/content_types.py",
    "src/silent_cascade/models/controller.py",
    "src/silent_cascade/models/dynamics.py",
    "src/silent_cascade/models/errors.py",
    "src/silent_cascade/models/event_flow.py",
    "src/silent_cascade/models/heads.py",
    "src/silent_cascade/models/jump.py",
    "src/silent_cascade/models/losses.py",
    "src/silent_cascade/models/types.py",
    "src/silent_cascade/rng.py",
    "src/silent_cascade/schemas.py",
    "src/silent_cascade/train/__init__.py",
    "src/silent_cascade/train/__main__.py",
    "src/silent_cascade/train/batches.py",
    "src/silent_cascade/train/checkpoints.py",
    "src/silent_cascade/train/cli.py",
    "src/silent_cascade/train/component_eval.py",
    "src/silent_cascade/train/config.py",
    "src/silent_cascade/train/content_unroll.py",
    "src/silent_cascade/train/curriculum.py",
    "src/silent_cascade/train/curriculum_data.py",
    "src/silent_cascade/train/evidence.py",
    "src/silent_cascade/train/evidence_types.py",
    "src/silent_cascade/train/numeric_inventory.py",
    "src/silent_cascade/train/objective.py",
    "src/silent_cascade/train/observations.py",
    "src/silent_cascade/train/provenance.py",
    "src/silent_cascade/train/state.py",
    "src/silent_cascade/train/traces.py",
    "src/silent_cascade/train/trainer.py",
    "src/silent_cascade/train/unroll.py",
    "src/silent_cascade/train/verification.py",
    "src/silent_cascade/validation.py",
    "uv.lock",
)


def git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise ProvenanceError("invalid Git trust anchor") from error


def relative(root: Path, path: Path) -> str:
    try:
        value = path.absolute().relative_to(root.absolute()).as_posix()
        if ".." in Path(value).parts or value == ".":
            raise ValueError("unsafe relative path")
        return value
    except ValueError as error:
        raise ProvenanceError("artifact must be inside the named repository") from error


def regular_blob(root: Path, revision: str, path: str, limit=MAX_ARTIFACT_BYTES) -> bytes:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ProvenanceError("revision must be an exact lowercase commit SHA")
    if Path(path).is_absolute() or ".." in Path(path).parts:
        raise ProvenanceError("invalid Git path")
    try:
        meta, actual = git(root, "ls-tree", "-z", revision, "--", path).rstrip(b"\0").split(b"\t")
        mode, kind, oid = meta.split()
        if actual.decode() != path or mode not in (b"100644", b"100755") or kind != b"blob":
            raise ValueError("not a regular blob")
        if int(git(root, "cat-file", "-s", oid.decode())) > limit:
            raise ValueError("Git blob byte limit exceeded")
        return git(root, "cat-file", "blob", oid.decode())
    except (ValueError, UnicodeError) as error:
        raise ProvenanceError("missing or invalid regular Git blob") from error


def source_digest(root, source):
    digest = hashlib.sha256(b"silent-cascade/phase3/source-tree/v1\0")
    for name in PHASE3_SOURCE_PATHS:
        data, encoded = regular_blob(root, source, name), name.encode()
        digest.update(len(encoded).to_bytes(4, "big") + encoded)
        digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def authenticate_source(root: Path, source: str, plan: str) -> str:
    """Require the actual package checkout, clean history and unchanged execution closure."""
    root = root.resolve()
    if Path(__file__).resolve() != root / "src/silent_cascade/train/provenance.py":
        raise ProvenanceError("executing package differs from the named checkout")
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("silent_cascade") and filename:
            path = Path(filename).resolve()
            if not path.is_relative_to(root / "src/silent_cascade"):
                raise ProvenanceError("executing local dependency comes from another checkout")
    for revision in (source, plan):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ProvenanceError("revision must be a full commit SHA")
        if git(root, "rev-parse", f"{revision}^{{commit}}").decode().strip() != revision:
            raise ProvenanceError("revision does not resolve exactly")
    head = git(root, "rev-parse", "HEAD").decode().strip()
    git(root, "merge-base", "--is-ancestor", source, head)
    git(root, "merge-base", "--is-ancestor", plan, source)
    if git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise ProvenanceError("collection requires a clean checkout")
    if git(root, "log", "-1", "--format=%H", source, "--", PLAN_PATH).decode().strip() != plan:
        raise ProvenanceError("plan revision must be the latest effective approved plan")
    if b"User-approved; completion tracked in docs/PLAN.md" not in regular_blob(
        root, plan, PLAN_PATH
    ):
        raise ProvenanceError("plan lacks the approved header")
    for name in (*PHASE3_SOURCE_PATHS, SPEC_PATH, PLAN_PATH):
        expected = regular_blob(root, source, name)
        if expected != read_bytes(root / name, MAX_ARTIFACT_BYTES) or expected != regular_blob(
            root, head, name
        ):
            raise ProvenanceError(f"reviewed source changed: {name}")
    return head


def validate_config(config, root):
    if not isinstance(config, ResolvedConfig) or not isinstance(config.config, Phase3Config):
        raise ProvenanceError("expected resolved Phase3Config")
    expected = (
        "configs/base.yaml",
        "configs/data/primary.yaml",
        "configs/model/event_flow.yaml",
        "configs/model/neural_components.yaml",
        config.config.training.overlay_path,
    )
    if tuple(relative(root, path) for path in config.source_paths) != expected:
        raise ProvenanceError("gate requires the exact ordered source configuration overlays")
    if (
        resolve_config(Phase3Config, tuple(root / name for name in expected)).sha256
        != config.sha256
    ):
        raise ProvenanceError("configuration differs from committed overlays")
    return expected


def publish(path: Path, value) -> None:
    raw = canonical_json_bytes(value)
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ProvenanceError("artifact byte limit exceeded")
    try:
        with archive_parent(path, error_factory=ProvenanceError) as (parent, name):
            _publish_bytes_at(parent, name, raw, replace=False)
    except Exception as error:
        if isinstance(error, ProvenanceError):
            raise
        raise ProvenanceError("immutable artifact publication failed") from error


def freeze_component_manifest(
    config, *, source_commit: str, plan_revision: str, output_path: Path
) -> ComponentManifest:
    root = Path(__file__).resolve().parents[3]
    relative(root, output_path)
    authenticate_source(root, source_commit, plan_revision)
    validate_config(config, root)
    production = config.config.training.is_production
    examples = tuple(
        make_curriculum_example(
            config.config,
            CurriculumKey(
                "ofd-one-hop-v1", "validation" if production else "debug", 313, 337, i, "one_hop"
            ),
        )
        for i in range(10000 if production else 16)
    )
    manifest = ComponentManifest.from_examples(
        examples, config.config, source_revision=source_commit, plan_revision=plan_revision
    )
    manifest = ComponentManifest.model_validate_json(
        canonical_json_bytes(
            {
                **manifest.model_dump(mode="json"),
                "publication": "production" if production else "debug",
            }
        )
    )
    authenticate_source(root, source_commit, plan_revision)
    publish(output_path, manifest)
    return manifest


def collect_provenance(config, root, manifest_path, source, plan):
    head = authenticate_source(root, source, plan)
    paths = validate_config(config, root)
    name = relative(root, manifest_path)
    intro = git(root, "log", "-1", "--format=%H", "--", name).decode().strip()
    git(root, "merge-base", "--is-ancestor", source, intro)
    git(root, "merge-base", "--is-ancestor", intro, head)
    raw = read_bytes(manifest_path, MAX_ARTIFACT_BYTES)
    if raw != regular_blob(root, intro, name):
        raise ProvenanceError("manifest differs from its introduction blob")
    return Phase3EvidenceProvenance(
        source_commit=source,
        plan_revision=plan,
        evidence_base_revision=head,
        source_paths=PHASE3_SOURCE_PATHS,
        source_tree_sha256=source_digest(root, source),
        specification_sha256=hashlib.sha256(regular_blob(root, source, SPEC_PATH)).hexdigest(),
        approved_plan_sha256=hashlib.sha256(regular_blob(root, source, PLAN_PATH)).hexdigest(),
        config_paths=paths,
        config_canonical_json=config.canonical_json.decode(),
        config_sha256=config.sha256,
        validation_manifest_path=name,
        validation_manifest_commit=intro,
        validation_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        threads=torch.get_num_threads(),
    )
