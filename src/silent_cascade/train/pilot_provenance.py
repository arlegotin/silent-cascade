"""Regular Git source closure and separately introduced immutable pilot evidence."""

import re
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from pydantic import Field

from silent_cascade.config import ResolvedConfig, resolve_config
from silent_cascade.env.episode import CorpusDigestEntry, corpus_sha256
from silent_cascade.eval.pilot_audit import PilotAuditReport
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.train.pilot_config import Phase4Config
from silent_cascade.train.pilot_data import (
    Hash,
    _read_pilot_bytes,
    iter_pilot_examples,
    load_pilot_manifest,
)
from silent_cascade.train.provenance import git, regular_blob, relative
from silent_cascade.validation import StrictModel

PLAN = "docs/superpowers/plans/2026-09-16-phase-4-autonomous-eventflow.md"
SPEC = "docs/superpowers/specs/2026-08-30-silent-cascade-design.md"
R2_PLAN = "docs/superpowers/plans/2026-09-17-phase-4-r2-archive.md"
R2_SPEC = "docs/superpowers/specs/2026-09-17-phase-4-r2-archive-design.md"
PILOT_ENTRYPOINTS = (
    "scripts/run_pilot.py",
    "scripts/check_phase4_pilot.py",
    "scripts/verify_phase4_gate_artifact.py",
    "scripts/record_phase4_local_verify.py",
)


def _config_paths(root, config):
    paths = (
        "configs/base.yaml",
        "configs/data/primary.yaml",
        "configs/model/event_flow.yaml",
        "configs/model/neural_components.yaml",
        "configs/train/pilot.yaml"
        if config.config.pilot.is_production
        else "configs/train/pilot_smoke.yaml",
    )
    if tuple(relative(root, p) for p in config.source_paths) != paths:
        raise ValueError("pilot overlay inventory differs from closed profile")
    return paths


class PilotSourceIdentity(StrictModel):
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_files: dict[str, Hash]
    source_sha256: Hash
    plan_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    plan_sha256: Hash
    spec_sha256: Hash
    config_sha256: Hash
    data_introductions: dict[str, dict[str, str]] = Field(default_factory=dict)


def _package_paths(root, revision):
    return tuple(
        path
        for path in git(root, "ls-tree", "-r", "--name-only", revision, "--", "src/silent_cascade")
        .decode()
        .splitlines()
        if path.endswith(".py")
    )


def _blobs(root, revision, paths):
    """One bounded Git batch, preserving regular-file and exact-path checks."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("expected exact source commit")
    paths = tuple(sorted(set(paths)))
    if any(Path(p).is_absolute() or ".." in Path(p).parts for p in paths):
        raise ValueError("unsafe source path")
    entries = {}
    for entry in git(root, "ls-tree", "-rz", revision).split(b"\0"):
        if not entry:
            continue
        meta, name = entry.split(b"\t", 1)
        mode, kind, oid = meta.split()
        name = name.decode()
        if name in paths:
            if mode not in (b"100644", b"100755") or kind != b"blob":
                raise ValueError("source requires regular Git blobs")
            entries[name] = oid
    if set(entries) != set(paths):
        raise ValueError("source inventory has missing regular blobs")
    raw = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=root,
        input=b"\n".join(entries[p] for p in paths) + b"\n",
        capture_output=True,
        check=True,
        timeout=30,
    ).stdout
    cursor, result = 0, {}
    for path in paths:
        end = raw.index(b"\n", cursor)
        oid, kind, size_raw = raw[cursor:end].split()
        size = int(size_raw)
        if oid != entries[path] or kind != b"blob" or not 0 <= size <= 64 * 1024 * 1024:
            raise ValueError("invalid source blob batch")
        cursor = end + 1
        result[path] = raw[cursor : cursor + size]
        cursor += size + 1
    if cursor != len(raw):
        raise ValueError("source blob batch length differs")
    return result


def authenticate_pilot_source(
    *, repo_root: Path, source_commit: str, config: ResolvedConfig[Phase4Config]
) -> PilotSourceIdentity:
    root = repo_root.resolve()
    if Path(__file__).resolve() != root / "src/silent_cascade/train/pilot_provenance.py":
        raise ValueError("executing pilot package differs from repository")
    head = git(root, "rev-parse", "HEAD").decode().strip()
    git(root, "merge-base", "--is-ancestor", source_commit, head)
    package = _package_paths(root, source_commit)
    actual = tuple(sorted(relative(root, p) for p in (root / "src/silent_cascade").rglob("*.py")))
    if package != actual:
        raise ValueError("executing package inventory differs from committed source")
    configs = _config_paths(root, config)
    resolved = resolve_config(Phase4Config, tuple(root / p for p in configs))
    if (
        resolved.sha256 != config.sha256
        or resolved.canonical_json != config.canonical_json
        or canonical_json_bytes(config.config) != config.canonical_json
    ):
        raise ValueError("pilot config differs from source overlays")
    paths = (
        *package,
        *PILOT_ENTRYPOINTS,
        *configs,
        "configs/train/pilot.yaml",
        "pyproject.toml",
        "uv.lock",
        "Makefile",
        ".python-version",
        PLAN,
        SPEC,
        R2_PLAN,
        R2_SPEC,
    )
    source_blobs = _blobs(root, source_commit, paths)
    head_blobs = source_blobs if head == source_commit else _blobs(root, head, paths)
    inventory = {}
    for name, blob in source_blobs.items():
        if blob != _read_pilot_bytes(root / name) or blob != head_blobs[name]:
            raise ValueError(f"executing source differs from committed blob: {name}")
        inventory[name] = sha256_bytes(blob)
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("silent_cascade") and filename:
            path = Path(filename)
            if path.resolve() != path.absolute() or relative(root, path) not in package:
                raise ValueError("loaded pilot module differs from authenticated closure")
    plan = source_blobs[PLAN]
    if b"User-approved; completion tracked in docs/PLAN.md" not in plan:
        raise ValueError("pilot plan is not approved")
    revision = git(root, "log", "-1", "--format=%H", source_commit, "--", PLAN).decode().strip()
    return PilotSourceIdentity(
        source_commit=source_commit,
        source_files=inventory,
        source_sha256=sha256_bytes(canonical_json_bytes(inventory)),
        plan_revision=revision,
        plan_sha256=sha256_bytes(plan),
        spec_sha256=inventory[SPEC],
        config_sha256=config.sha256,
    )


def _introduction(root, source, producer, path, raw):
    name = relative(root, path)
    if raw != regular_blob(root, source, name):
        raise ValueError("pilot evidence is substituted or uncommitted")
    introduction = git(root, "log", "-1", "--format=%H", source, "--", name).decode().strip()
    if introduction in {producer, source}:
        raise ValueError("producer, evidence introduction and training revisions must be distinct")
    git(root, "merge-base", "--is-ancestor", producer, introduction)
    git(root, "merge-base", "--is-ancestor", introduction, source)
    if regular_blob(root, introduction, name) != raw:
        raise ValueError("evidence introduction differs")
    return {
        "path": name,
        "producer": producer,
        "introduction": introduction,
        "sha256": sha256_bytes(raw),
    }


def verify_pilot_data(*, repo_root, source_commit, config, manifests):
    configs = _config_paths(repo_root, config)
    if set(manifests) != {"one_hop", "two_hop", "primary", "robustness"}:
        raise ValueError("all four fixed pilot manifests and audits are required")
    loaded, introductions = {}, {}
    producer_inventory = {}
    for stage, path in manifests.items():
        manifest = load_pilot_manifest(path, config=config)
        if manifest.stage != stage:
            raise ValueError("pilot manifest stage mismatch")
        audit_path = path.parent / "audits" / stage / "report.json"
        raw = _read_pilot_bytes(audit_path)
        audit = PilotAuditReport.model_validate_json(raw)
        if canonical_json_bytes(audit) != raw:
            raise ValueError("audit report must be canonical")
        producer = audit.source_commit
        expected_paths = _package_paths(repo_root, producer)
        compatible = (
            *expected_paths,
            PLAN,
            SPEC,
            *configs,
        )
        if producer not in producer_inventory:
            blobs = _blobs(repo_root, producer, compatible)
            training = _blobs(repo_root, source_commit, compatible)
            if blobs != training:
                raise ValueError("audit executable/config/plan is incompatible")
            producer_inventory[producer] = {
                p.removeprefix("src/silent_cascade/"): sha256_bytes(blobs[p])
                for p in expected_paths
            }
        expected = producer_inventory[producer]
        if audit.executing_source_files != expected:
            raise ValueError("audit executing source inventory differs from producer Git blobs")
        projection = tuple(e.projected.episode_sha256 for e in manifest.entries)
        corpus = corpus_sha256(
            (
                CorpusDigestEntry(e.projected.public_id, h)
                for e, h in zip(manifest.entries, projection, strict=True)
            ),
            expected_count=manifest.count,
        )
        oracle = tuple(
            sha256_bytes(canonical_json_bytes(asdict(e.oracle_trace)))
            for e in iter_pilot_examples(manifest, config=config.config)
        )
        if (
            audit.manifest_sha256 != sha256_bytes(_read_pilot_bytes(path))
            or audit.config_sha256 != config.sha256
            or audit.source_commit != manifest.source_commit
            or audit.projected_episode_sha256s != projection
            or audit.oracle_trace_sha256s != oracle
            or audit.corpus_sha256 != corpus
            or audit.audit_config_sha256
            != sha256_bytes(canonical_json_bytes(config.config.data.leakage_audit))
            or (config.config.pilot.is_production and not audit.acceptance)
        ):
            raise ValueError("pilot audit report does not authenticate acceptance/data")
        introductions[stage + "/manifest"] = _introduction(
            repo_root, source_commit, producer, path, _read_pilot_bytes(path)
        )
        introductions[stage + "/audit"] = _introduction(
            repo_root, source_commit, producer, audit_path, raw
        )
        loaded[stage] = manifest
    return loaded, introductions
