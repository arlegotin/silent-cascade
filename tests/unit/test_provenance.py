"""Contracts for immutable Phase 1 evidence provenance."""

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace
from silent_cascade.errors import ProvenanceError
from silent_cascade.provenance import (
    GENERATOR_SOURCE_PATHS,
    PHASE1_ANALYSIS_SOURCE_PATHS,
    collect_evidence_provenance,
    source_tree_sha256,
)


def test_source_tree_hash_uses_unambiguous_length_framing(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_bytes(b"x")
    (tmp_path / "bb.py").write_bytes(b"yz")

    assert source_tree_sha256(tmp_path, ("a.py", "bb.py")) == (
        "481adae060acaa85ecfc996ec2b847b52624ea432c5b4f7e91c5462f3fc41c49"
    )


def test_source_tree_hash_distinguishes_naively_colliding_path_content_pairs(
    tmp_path: Path,
) -> None:
    (tmp_path / "a").write_bytes(b"bc")
    (tmp_path / "ab").write_bytes(b"c")

    assert source_tree_sha256(tmp_path, ("a",)) != source_tree_sha256(tmp_path, ("ab",))


@pytest.mark.parametrize(
    "paths",
    [
        ("missing.py",),
        ("a.py", "a.py"),
        ("../outside.py",),
        ("absolute.py", "/tmp/outside.py"),
    ],
)
def test_source_tree_hash_rejects_any_invalid_scope_path(
    tmp_path: Path, paths: tuple[str, ...]
) -> None:
    (tmp_path / "a.py").write_text("x", encoding="utf-8")

    with pytest.raises(ProvenanceError):
        source_tree_sha256(tmp_path, paths)


def test_source_tree_hash_rejects_symlinks_and_non_regular_files(tmp_path: Path) -> None:
    (tmp_path / "regular.py").write_text("x", encoding="utf-8")
    (tmp_path / "linked.py").symlink_to(tmp_path / "regular.py")
    (tmp_path / "directory.py").mkdir()

    with pytest.raises(ProvenanceError, match="symlink"):
        source_tree_sha256(tmp_path, ("linked.py",))
    with pytest.raises(ProvenanceError, match="regular"):
        source_tree_sha256(tmp_path, ("directory.py",))


def test_frozen_source_path_sets_are_exact_and_sorted() -> None:
    assert GENERATOR_SOURCE_PATHS == (
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
    assert tuple(sorted(PHASE1_ANALYSIS_SOURCE_PATHS)) == PHASE1_ANALYSIS_SOURCE_PATHS
    assert set(GENERATOR_SOURCE_PATHS) < set(PHASE1_ANALYSIS_SOURCE_PATHS)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _committed_provenance_repository(repo: Path) -> Path:
    for relative in (*PHASE1_ANALYSIS_SOURCE_PATHS, "pyproject.toml", "uv.lock"):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"contents for {relative}\n", encoding="utf-8")
    plan = repo / "docs/superpowers/plans/2026-08-30-phase-1-generator-oracle.md"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text("approved task plan\n", encoding="utf-8")
    resolved_input = repo / "configs/resolved.yaml"
    resolved_input.parent.mkdir(parents=True, exist_ok=True)
    resolved_input.write_text("runtime: {}\n", encoding="utf-8")
    (repo / "README.md").write_text("documentation\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Test Runner")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "initial scientific inputs")
    return resolved_input


def test_collector_records_committed_source_config_seed_and_scoped_dirty_state(
    tmp_path: Path,
) -> None:
    resolved_input = _committed_provenance_repository(tmp_path)
    resolved = replace(
        resolve_config(
            Phase1Config,
            [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
        ),
        source_paths=(resolved_input,),
    )

    collected = collect_evidence_provenance(
        resolved,
        repo_root=tmp_path,
        generation_mode="matched",
        allocation_id="validation-v1",
        split_namespace=SplitNamespace.VALIDATION,
        root_seed=17,
        public_id_seed=91,
        analysis_seeds={"z": 9, "a": 1},
    )

    assert collected.plan_base_revision == _git(
        tmp_path,
        "log",
        "-1",
        "--format=%H",
        "--",
        "docs/superpowers/plans/2026-08-30-phase-1-generator-oracle.md",
    )
    assert collected.source_commit == _git(tmp_path, "rev-parse", "--verify", "HEAD")
    assert not collected.source_dirty
    assert collected.config_sha256 == resolved.sha256
    assert collected.public_id_seed_sha256 == (
        "3acaee04f18b5609e8d4bdd6a5ab1ee2e24e8143bb1b33a3de05204cb195d980"
    )
    assert collected.analysis_seeds == {"a": 1, "z": 9}
    assert collected.foundation_model_calls == 0

    (tmp_path / "README.md").write_text("edited documentation\n", encoding="utf-8")
    assert not collect_evidence_provenance(
        resolved,
        repo_root=tmp_path,
        generation_mode="matched",
        allocation_id="validation-v1",
        split_namespace=SplitNamespace.VALIDATION,
        root_seed=17,
        public_id_seed=91,
        analysis_seeds={},
    ).source_dirty

    (tmp_path / GENERATOR_SOURCE_PATHS[0]).write_text("changed\n", encoding="utf-8")
    assert collect_evidence_provenance(
        resolved,
        repo_root=tmp_path,
        generation_mode="matched",
        allocation_id="validation-v1",
        split_namespace=SplitNamespace.VALIDATION,
        root_seed=17,
        public_id_seed=91,
        analysis_seeds={},
    ).source_dirty
