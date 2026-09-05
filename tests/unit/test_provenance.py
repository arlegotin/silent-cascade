"""Contracts for immutable Phase 1 evidence provenance."""

import ast
import json
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
    TASK14_ANALYSIS_SOURCE_PATHS,
    EvidenceProvenance,
    SourceTreeFingerprint,
    collect_evidence_provenance,
    collect_final_phase1_provenance,
    source_tree_sha256,
)

EXPECTED_GENERATOR_SOURCE_PATHS = (
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
EXPECTED_PHASE1_ANALYSIS_SOURCE_PATHS = (
    "scripts/check_phase1_reproducibility.py",
    "scripts/verify_phase1_gate_artifacts.py",
    "src/silent_cascade/__init__.py",
    "src/silent_cascade/config.py",
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
    "src/silent_cascade/hashing.py",
    "src/silent_cascade/io.py",
    "src/silent_cascade/logging/__init__.py",
    "src/silent_cascade/logging/manifest.py",
    "src/silent_cascade/provenance.py",
    "src/silent_cascade/rng.py",
    "src/silent_cascade/schemas.py",
    "src/silent_cascade/validation.py",
)
EXPECTED_TASK14_ANALYSIS_SOURCE_PATHS = (
    "scripts/check_phase1_reproducibility.py",
    "src/silent_cascade/__init__.py",
    "src/silent_cascade/config.py",
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
    "src/silent_cascade/hashing.py",
    "src/silent_cascade/io.py",
    "src/silent_cascade/logging/__init__.py",
    "src/silent_cascade/logging/manifest.py",
    "src/silent_cascade/provenance.py",
    "src/silent_cascade/rng.py",
    "src/silent_cascade/schemas.py",
    "src/silent_cascade/validation.py",
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


def test_source_tree_hash_rejects_unsorted_paths_before_hashing(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_bytes(b"x")
    (tmp_path / "b.py").write_bytes(b"y")

    with pytest.raises(ProvenanceError, match="sorted"):
        source_tree_sha256(tmp_path, ("b.py", "a.py"))


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


def test_generator_source_paths_match_the_literal_approved_scope() -> None:
    assert GENERATOR_SOURCE_PATHS == EXPECTED_GENERATOR_SOURCE_PATHS


def test_final_phase1_source_paths_match_the_literal_approved_scope() -> None:
    assert PHASE1_ANALYSIS_SOURCE_PATHS == EXPECTED_PHASE1_ANALYSIS_SOURCE_PATHS


def test_task14_source_paths_match_the_literal_approved_scope() -> None:
    assert TASK14_ANALYSIS_SOURCE_PATHS == EXPECTED_TASK14_ANALYSIS_SOURCE_PATHS


def _source_package_parts(source_path: str) -> tuple[str, ...]:
    path = Path(source_path)
    if path.parts[:2] == ("src", "silent_cascade"):
        module_parts = path.with_suffix("").parts[1:]
    elif path.parts and path.parts[0] == "scripts":
        module_parts = path.with_suffix("").parts
    else:
        return ()
    if module_parts[-1] == "__init__":
        return module_parts[:-1]
    return module_parts[:-1]


def _module_names_from_import(source_path: str, node: ast.Import | ast.ImportFrom) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}

    if node.level:
        package_parts = _source_package_parts(source_path)
        retained_count = len(package_parts) - (node.level - 1)
        if retained_count <= 0:
            return set()
        base_parts = package_parts[:retained_count]
        module_parts = (*base_parts, *(node.module.split(".") if node.module else ()))
        module = ".".join(module_parts)
    elif node.module:
        module = node.module
    else:
        return set()
    return {module, *(f"{module}.{alias.name}" for alias in node.names)}


def _existing_local_module_paths(repo_root: Path, module: str) -> set[str]:
    candidates: tuple[Path, ...]
    if module == "silent_cascade" or module.startswith("silent_cascade."):
        base = Path("src", *module.split("."))
        candidates = (base.with_suffix(".py"), base / "__init__.py")
    elif module == "scripts" or module.startswith("scripts."):
        base = Path(*module.split("."))
        candidates = (base.with_suffix(".py"), base / "__init__.py")
    else:
        base = Path("scripts", *module.split("."))
        candidates = (base.with_suffix(".py"), base / "__init__.py")
    return {candidate.as_posix() for candidate in candidates if (repo_root / candidate).is_file()}


def _existing_ancestor_initializers(repo_root: Path, source_path: str) -> set[str]:
    path = Path(source_path)
    if path.parts[:1] == ("src",):
        root_depth = 1
    elif path.parts[:1] == ("scripts",):
        root_depth = 0
    else:
        return set()
    return {
        initializer.as_posix()
        for depth in range(root_depth + 1, len(path.parts))
        if (initializer := Path(*path.parts[:depth], "__init__.py")) != path
        and (repo_root / initializer).is_file()
    } | ({path.as_posix()} if path.name == "__init__.py" else set())


def _scientific_source_closure(repo_root: Path, declared_paths: tuple[str, ...]) -> set[str]:
    closure = set(declared_paths)
    pending = list(declared_paths)
    while pending:
        source_path = pending.pop()
        discovered = _existing_ancestor_initializers(repo_root, source_path)
        tree = ast.parse((repo_root / source_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for module in _module_names_from_import(source_path, node):
                    discovered.update(_existing_local_module_paths(repo_root, module))
        for path in tuple(discovered):
            discovered.update(_existing_ancestor_initializers(repo_root, path))
        new_paths = discovered - closure
        closure.update(new_paths)
        pending.extend(new_paths)
    return closure


def _missing_scientific_source_paths(
    repo_root: Path, declared_paths: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        sorted(_scientific_source_closure(repo_root, declared_paths) - set(declared_paths))
    )


@pytest.mark.parametrize(
    ("source", "expected_missing"),
    [
        pytest.param(
            "from . import timing\nfrom .timing import action_window\n",
            (
                "src/silent_cascade/__init__.py",
                "src/silent_cascade/env/__init__.py",
                "src/silent_cascade/env/timing.py",
            ),
            id="relative-package-and-module-imports",
        ),
        pytest.param(
            "from silent_cascade.env.timing import action_window\n",
            (
                "src/silent_cascade/__init__.py",
                "src/silent_cascade/env/__init__.py",
                "src/silent_cascade/env/timing.py",
            ),
            id="absolute-local-import",
        ),
    ],
)
def test_import_closure_resolves_relative_absolute_and_ancestor_modules(
    tmp_path: Path, source: str, expected_missing: tuple[str, ...]
) -> None:
    for relative in (
        "src/silent_cascade/__init__.py",
        "src/silent_cascade/env/__init__.py",
        "src/silent_cascade/env/timing.py",
        "src/silent_cascade/env/probe.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source if path.name == "probe.py" else "", encoding="utf-8")

    assert (
        _missing_scientific_source_paths(tmp_path, ("src/silent_cascade/env/probe.py",))
        == expected_missing
    )


def test_import_closure_resolves_local_script_imports_and_initializer(
    tmp_path: Path,
) -> None:
    for relative, contents in (
        ("scripts/__init__.py", ""),
        ("scripts/helper.py", ""),
        ("scripts/probe.py", "import helper\n"),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")

    assert _missing_scientific_source_paths(tmp_path, ("scripts/probe.py",)) == (
        "scripts/__init__.py",
        "scripts/helper.py",
    )


@pytest.mark.parametrize(
    "declared_paths",
    [GENERATOR_SOURCE_PATHS, PHASE1_ANALYSIS_SOURCE_PATHS, TASK14_ANALYSIS_SOURCE_PATHS],
    ids=["generator", "phase1_analysis", "phase1_task14_analysis"],
)
def test_scientific_source_scopes_include_explicit_project_import_closure(
    declared_paths: tuple[str, ...],
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    missing = _missing_scientific_source_paths(repo_root, declared_paths)

    assert not missing, f"scientific source scope omits exact local dependencies: {missing}"


@pytest.mark.parametrize(
    ("scope", "paths"),
    [
        ("generator", GENERATOR_SOURCE_PATHS[:-1]),
        ("generator", (*GENERATOR_SOURCE_PATHS, "unexpected.py")),
        ("generator", PHASE1_ANALYSIS_SOURCE_PATHS),
        ("phase1_analysis", GENERATOR_SOURCE_PATHS),
        ("generator", tuple(reversed(GENERATOR_SOURCE_PATHS))),
    ],
)
def test_source_tree_fingerprint_rejects_any_noncanonical_declared_scope(
    scope: str, paths: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError, match=r"(sorted|exact frozen)"):
        SourceTreeFingerprint(
            frame_version="sc-source-tree-v1",
            scope=scope,  # type: ignore[arg-type]
            paths=paths,
            sha256="a" * 64,
        )


def _evidence_payload() -> dict[str, object]:
    return {
        "schema_version": "phase1-evidence-provenance-v1",
        "plan_base_revision": "b" * 40,
        "source_commit": "c" * 40,
        "source_dirty": False,
        "generator_version": "ofd-v1",
        "generation_mode": "matched",
        "allocation_id": "validation-v1",
        "split_namespace": "validation",
        "config_sha256": "d" * 64,
        "generator_source": {
            "frame_version": "sc-source-tree-v1",
            "scope": "generator",
            "paths": GENERATOR_SOURCE_PATHS,
            "sha256": "a" * 64,
        },
        "analysis_source": {
            "frame_version": "sc-source-tree-v1",
            "scope": "phase1_analysis",
            "paths": PHASE1_ANALYSIS_SOURCE_PATHS,
            "sha256": "a" * 64,
        },
        "root_seed": 17,
        "public_id_seed_sha256": "e" * 64,
        "foundation_model_calls": 0,
    }


@pytest.mark.parametrize("invalid", [False, 0.0, "0"])
def test_evidence_provenance_rejects_non_exact_zero_foundation_counter_json(
    invalid: object,
) -> None:
    payload = _evidence_payload()
    payload["foundation_model_calls"] = invalid

    with pytest.raises(ValueError, match="exact integer zero"):
        EvidenceProvenance.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("invalid", [False, 0.0, "0"])
def test_evidence_provenance_rejects_non_exact_zero_foundation_counter_constructor(
    invalid: object,
) -> None:
    payload = _evidence_payload()
    payload["split_namespace"] = SplitNamespace.VALIDATION
    payload["foundation_model_calls"] = invalid

    with pytest.raises(ValueError, match="exact integer zero"):
        EvidenceProvenance.model_validate(payload)


def test_evidence_provenance_defensively_rejects_swapped_fingerprint_models() -> None:
    payload = _evidence_payload()
    payload["generator_source"], payload["analysis_source"] = (
        payload["analysis_source"],
        payload["generator_source"],
    )

    with pytest.raises(ValueError, match="exact frozen"):
        EvidenceProvenance.model_validate_json(json.dumps(payload))


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _committed_provenance_repository(repo: Path) -> Path:
    for relative in (*EXPECTED_PHASE1_ANALYSIS_SOURCE_PATHS, "pyproject.toml", "uv.lock"):
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


def _collect_final_and_task14_provenance(
    resolved: object, repo_root: Path
) -> tuple[EvidenceProvenance, EvidenceProvenance]:
    common = {
        "repo_root": repo_root,
        "generation_mode": "independent",
        "allocation_id": "phase1-gate-v1",
        "split_namespace": SplitNamespace.PHASE1_GATE,
        "root_seed": 17,
        "public_id_seed": 91,
        "analysis_seeds": {},
    }
    final = collect_final_phase1_provenance(resolved, **common)  # type: ignore[arg-type]
    task14 = collect_evidence_provenance(
        resolved,
        analysis_scope="phase1_task14_analysis",
        **common,  # type: ignore[arg-type]
    )
    return final, task14


@pytest.mark.parametrize(
    ("mutated_path", "expected_changes"),
    [
        pytest.param("src/silent_cascade/__init__.py", (True, True, True), id="root-initializer"),
        pytest.param(
            "src/silent_cascade/env/__init__.py",
            (True, True, True),
            id="env-initializer",
        ),
        pytest.param(
            "src/silent_cascade/logging/__init__.py",
            (False, True, True),
            id="logging-initializer",
        ),
        pytest.param("src/silent_cascade/env/timing.py", (True, True, True), id="timing-module"),
        pytest.param(
            "src/silent_cascade/env/leakage.py",
            (False, True, True),
            id="leakage-module",
        ),
    ],
)
def test_scoped_source_mutation_dirties_and_changes_exact_applicable_fingerprints(
    tmp_path: Path,
    mutated_path: str,
    expected_changes: tuple[bool, bool, bool],
) -> None:
    resolved_input = _committed_provenance_repository(tmp_path)
    resolved = replace(
        resolve_config(
            Phase1Config,
            [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
        ),
        source_paths=(resolved_input,),
    )
    final_before, task14_before = _collect_final_and_task14_provenance(resolved, tmp_path)
    assert not final_before.source_dirty
    assert not task14_before.source_dirty

    path = tmp_path / mutated_path
    path.write_text(f"changed contents for {mutated_path}\n", encoding="utf-8")
    final_after, task14_after = _collect_final_and_task14_provenance(resolved, tmp_path)

    assert final_after.source_dirty
    assert task14_after.source_dirty
    actual_changes = (
        final_after.generator_source.sha256 != final_before.generator_source.sha256,
        final_after.analysis_source.sha256 != final_before.analysis_source.sha256,
        task14_after.analysis_source.sha256 != task14_before.analysis_source.sha256,
    )
    assert actual_changes == expected_changes
