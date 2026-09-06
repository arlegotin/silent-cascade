"""Contracts for immutable Phase 1 evidence provenance."""

import ast
import json
import subprocess
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

import silent_cascade.provenance as provenance_module
from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace
from silent_cascade.errors import ProvenanceError
from silent_cascade.provenance import (
    GENERATOR_SOURCE_PATHS,
    PHASE1_ANALYSIS_SOURCE_PATHS,
    TASK14_ANALYSIS_SOURCE_PATHS,
    AcceptedAttemptRun,
    ConstructionNamespaceBuilder,
    ConstructionNamespaceEvidence,
    EvidenceProvenance,
    SourceTreeFingerprint,
    accepted_attempt_runs,
    build_construction_namespace_evidence,
    collect_evidence_provenance,
    collect_final_phase1_provenance,
    construction_token_sequence_sha256,
    public_id_sequence_sha256,
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


def _namespace_payload() -> dict[str, object]:
    return {
        "schema_version": "construction-namespace-evidence-v1",
        "generation_mode": "matched",
        "public_id_seed": 91,
        "accepted_draw_count": 1,
        "accepted_attempt_runs": ({"first_draw_index": 0, "draw_count": 1, "accepted_attempt": 0},),
        "rejected_draw_count": 0,
        "generation_attempt_count": 1,
        "seed_token_count": 20,
        "seed_token_sequence_sha256": "a" * 64,
        "seed_token_collision_count": 0,
        "base_public_id_count": 4,
        "clock_public_id_count": 0,
        "total_public_id_count": 4,
        "public_id_sequence_sha256": "b" * 64,
        "public_id_collision_count": 0,
    }


def test_namespace_sequence_hashes_use_framed_ordered_fixed_width_values() -> None:
    """Changing order or accepting malformed-width values must break namespace identity."""
    tokens = ("00" * 32, "11" * 32)
    identifiers = (
        "00000000-0000-4000-8000-000000000001",
        "00000000-0000-4000-8000-000000000002",
    )

    assert construction_token_sequence_sha256(tokens) == (
        "71e6ce23089f79f685f0af45dbaafb74fe57b0e5d45805be7f2aaf3d637f51c1"
    )
    assert public_id_sequence_sha256(identifiers) == (
        "3040c9f8aa83d45f3228d98277fed044b898630042eb643d459b88316d1010f7"
    )
    assert construction_token_sequence_sha256(tuple(reversed(tokens))) != (
        construction_token_sequence_sha256(tokens)
    )
    assert public_id_sequence_sha256(tuple(reversed(identifiers))) != (
        public_id_sequence_sha256(identifiers)
    )
    with pytest.raises(ValueError, match="32-byte"):
        construction_token_sequence_sha256(("00",))
    with pytest.raises(ValueError, match="UUID"):
        public_id_sequence_sha256(("not-a-uuid",))


def test_accepted_attempt_runs_are_unique_canonical_run_length_encoding() -> None:
    """Splitting an equal run or losing a draw must be rejected as noncanonical evidence."""
    assert accepted_attempt_runs((0, 0, 2, 2, 2, 1)) == (
        AcceptedAttemptRun(first_draw_index=0, draw_count=2, accepted_attempt=0),
        AcceptedAttemptRun(first_draw_index=2, draw_count=3, accepted_attempt=2),
        AcceptedAttemptRun(first_draw_index=5, draw_count=1, accepted_attempt=1),
    )
    with pytest.raises(ValueError, match="nonempty"):
        accepted_attempt_runs(())
    with pytest.raises(ValueError, match="exact integer"):
        accepted_attempt_runs((False,))
    with pytest.raises(ValueError, match="below 1000"):
        accepted_attempt_runs((1_000,))


@pytest.mark.parametrize(
    "runs",
    [
        (),
        ({"first_draw_index": 1, "draw_count": 1, "accepted_attempt": 0},),
        (
            {"first_draw_index": 0, "draw_count": 1, "accepted_attempt": 0},
            {"first_draw_index": 2, "draw_count": 1, "accepted_attempt": 1},
        ),
        (
            {"first_draw_index": 0, "draw_count": 2, "accepted_attempt": 0},
            {"first_draw_index": 1, "draw_count": 1, "accepted_attempt": 1},
        ),
        (
            {"first_draw_index": 0, "draw_count": 1, "accepted_attempt": 0},
            {"first_draw_index": 1, "draw_count": 1, "accepted_attempt": 0},
        ),
    ],
    ids=("empty", "nonzero-start", "gap", "overlap", "adjacent-equal"),
)
def test_namespace_evidence_rejects_noncanonical_attempt_run_coverage(
    runs: tuple[dict[str, int], ...],
) -> None:
    payload = _namespace_payload()
    payload["accepted_attempt_runs"] = runs

    with pytest.raises(ValueError, match="attempt runs"):
        ConstructionNamespaceEvidence.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("accepted_draw_count", False, "integer"),
        ("public_id_seed", -1, "greater than or equal"),
        ("seed_token_sequence_sha256", "0" * 63, "pattern"),
        ("public_id_sequence_sha256", "g" * 64, "pattern"),
        ("seed_token_collision_count", 1, "literal"),
        ("public_id_collision_count", 1, "literal"),
        ("seed_token_count", 19, "count law"),
        ("base_public_id_count", 3, "count law"),
        ("total_public_id_count", 5, "total"),
        ("rejected_draw_count", 1, "rejected"),
        ("generation_attempt_count", 2, "generation"),
    ],
)
def test_namespace_evidence_rejects_corrupt_primitives_and_derived_counts(
    field: str, value: object, message: str
) -> None:
    payload = _namespace_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        ConstructionNamespaceEvidence.model_validate(payload)


def test_namespace_builder_scans_actual_token_and_id_values_for_collisions() -> None:
    """Coordinate uniqueness must not hide a reused derived token or emitted public ID."""
    tokens = tuple(f"{index:064x}" for index in range(20))
    identifiers = tuple(str(uuid.UUID(int=index + 1)) for index in range(4))
    evidence = build_construction_namespace_evidence(
        generation_mode="matched",
        public_id_seed=91,
        accepted_attempts=(0,),
        seed_tokens=tokens,
        base_public_ids=identifiers,
        clock_public_ids=(),
    )

    assert evidence.accepted_attempt_runs == (
        AcceptedAttemptRun(first_draw_index=0, draw_count=1, accepted_attempt=0),
    )
    assert evidence.public_id_seed == 91
    assert evidence.seed_token_count == 20
    assert evidence.total_public_id_count == 4
    with pytest.raises(ProvenanceError, match="construction token collision"):
        build_construction_namespace_evidence(
            generation_mode="matched",
            public_id_seed=91,
            accepted_attempts=(0,),
            seed_tokens=(*tokens[:-1], tokens[0]),
            base_public_ids=identifiers,
            clock_public_ids=(),
        )
    with pytest.raises(ProvenanceError, match="public ID collision"):
        build_construction_namespace_evidence(
            generation_mode="matched",
            public_id_seed=91,
            accepted_attempts=(0,),
            seed_tokens=tokens,
            base_public_ids=(*identifiers[:-1], identifiers[0]),
            clock_public_ids=(),
        )


def test_namespace_builder_counts_retries_once_per_construction_draw() -> None:
    """Matched member count must not multiply one cohort retry into four rejections."""
    evidence = build_construction_namespace_evidence(
        generation_mode="matched",
        public_id_seed=91,
        accepted_attempts=(0, 2, 2),
        seed_tokens=tuple(f"{index:064x}" for index in range(60)),
        base_public_ids=tuple(str(uuid.UUID(int=index + 1)) for index in range(12)),
        clock_public_ids=(),
    )

    assert evidence.accepted_draw_count == 3
    assert evidence.rejected_draw_count == 4
    assert evidence.generation_attempt_count == 7


def test_namespace_evidence_rejects_more_than_two_clock_children_per_base_episode() -> None:
    """A serialized clock count cannot exceed the paired-suite child relationship."""
    payload = _namespace_payload()
    payload["clock_public_id_count"] = 9
    payload["total_public_id_count"] = 13

    with pytest.raises(ValueError, match="clock public ID count"):
        ConstructionNamespaceEvidence.model_validate(payload)


def test_namespace_builder_rejects_huge_clock_count_before_numpy_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed clock counts must fail before any attempt/token/UUID buffer allocation."""
    import silent_cascade.provenance as provenance

    def prohibited_allocation(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("malformed count reached NumPy allocation")

    monkeypatch.setattr(provenance.np, "empty", prohibited_allocation)
    with pytest.raises(ValueError, match="clock public ID count"):
        ConstructionNamespaceBuilder(
            generation_mode="independent",
            public_id_seed=91,
            accepted_draw_count=1,
            clock_public_id_count=1_000_000_000,
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


def test_final_provenance_authenticates_historical_git_blobs_and_plan_ancestry(
    tmp_path: Path,
) -> None:
    """Worktree bytes and later plan edits cannot rewrite committed evidence."""
    resolved_input = _committed_provenance_repository(tmp_path)
    resolved = replace(
        resolve_config(
            Phase1Config,
            [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
        ),
        source_paths=(resolved_input,),
    )
    evidence = collect_final_phase1_provenance(
        resolved,
        repo_root=tmp_path,
        generation_mode="matched",
        allocation_id="validation-v1",
        split_namespace=SplitNamespace.VALIDATION,
        root_seed=17,
        public_id_seed=91,
        analysis_seeds={},
    )
    assert hasattr(provenance_module, "source_tree_sha256_at_revision")
    assert hasattr(provenance_module, "authenticate_final_phase1_provenance")
    source_tree_sha256_at_revision = provenance_module.source_tree_sha256_at_revision
    authenticate_final_phase1_provenance = provenance_module.authenticate_final_phase1_provenance

    assert (
        source_tree_sha256_at_revision(tmp_path, evidence.source_commit, GENERATOR_SOURCE_PATHS)
        == evidence.generator_source.sha256
    )
    authenticate_final_phase1_provenance(evidence, repo_root=tmp_path)

    (tmp_path / GENERATOR_SOURCE_PATHS[0]).write_text("uncommitted divergence\n")
    plan = tmp_path / "docs/superpowers/plans/2026-08-30-phase-1-generator-oracle.md"
    plan.write_text("later approved plan edit\n", encoding="utf-8")
    _git(tmp_path, "add", str(plan.relative_to(tmp_path)))
    _git(tmp_path, "commit", "-qm", "later plan edit")

    authenticate_final_phase1_provenance(evidence, repo_root=tmp_path)
    assert (
        source_tree_sha256_at_revision(tmp_path, evidence.source_commit, GENERATOR_SOURCE_PATHS)
        == evidence.generator_source.sha256
    )

    with pytest.raises(ProvenanceError, match="commit"):
        authenticate_final_phase1_provenance(
            evidence.model_copy(update={"source_commit": "f" * 40}),
            repo_root=tmp_path,
        )
    with pytest.raises(ProvenanceError, match="final Phase 1"):
        authenticate_final_phase1_provenance(
            evidence.model_copy(
                update={
                    "analysis_source": SourceTreeFingerprint(
                        frame_version="sc-source-tree-v1",
                        scope="phase1_task14_analysis",
                        paths=TASK14_ANALYSIS_SOURCE_PATHS,
                        sha256=evidence.analysis_source.sha256,
                    )
                }
            ),
            repo_root=tmp_path,
        )


def test_historical_source_hash_rejects_non_regular_git_modes(tmp_path: Path) -> None:
    """A symlink blob must never satisfy an authenticated scientific source path."""
    resolved_input = _committed_provenance_repository(tmp_path)
    del resolved_input
    target = tmp_path / GENERATOR_SOURCE_PATHS[0]
    target.unlink()
    target.symlink_to("config.py")
    _git(tmp_path, "add", GENERATOR_SOURCE_PATHS[0])
    _git(tmp_path, "commit", "-qm", "replace source with symlink")
    revision = _git(tmp_path, "rev-parse", "HEAD")
    assert hasattr(provenance_module, "source_tree_sha256_at_revision")
    source_tree_sha256_at_revision = provenance_module.source_tree_sha256_at_revision

    with pytest.raises(ProvenanceError, match="regular Git blob"):
        source_tree_sha256_at_revision(tmp_path, revision, GENERATOR_SOURCE_PATHS)


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
def test_scoped_source_mutation_dirties_without_rewriting_historical_fingerprints(
    tmp_path: Path,
    mutated_path: str,
    expected_changes: tuple[bool, bool, bool],
) -> None:
    del expected_changes
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
    assert actual_changes == (False, False, False)
