"""Contracts for immutable, environment-private episode manifests."""

import json
from multiprocessing import get_context
from pathlib import Path

import pytest

from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.errors import ManifestAccessError, ManifestError
from silent_cascade.logging.manifest import (
    EpisodeManifest,
    EpisodeManifestEntry,
    ManifestAccessClass,
    ManifestEnvelope,
    MatchedManifestCoordinate,
    load_manifest,
    publish_manifest,
    require_oracle_inspection_allowed,
    verify_manifest,
)
from silent_cascade.provenance import EvidenceProvenance, SourceTreeFingerprint

_DIGEST = "a" * 64


def _publish_in_child(path: str, payload: dict[str, object], result_queue: object) -> None:
    """Publish through a fresh interpreter so atomic create is genuinely contested."""
    from silent_cascade.errors import ManifestError
    from silent_cascade.logging.manifest import EpisodeManifest, publish_manifest

    try:
        manifest = EpisodeManifest.model_validate(payload)
        published = publish_manifest(Path(path), manifest)
        result_queue.put(("published", published.created))  # type: ignore[union-attr]
    except ManifestError:
        result_queue.put(("collision", False))  # type: ignore[union-attr]


def _provenance(*, split: SplitNamespace = SplitNamespace.VALIDATION) -> EvidenceProvenance:
    fingerprint = SourceTreeFingerprint(
        frame_version="sc-source-tree-v1",
        scope="generator",
        paths=("src/silent_cascade/env/generator.py",),
        sha256=_DIGEST,
    )
    analysis = fingerprint.model_copy(update={"scope": "phase1_analysis"})
    return EvidenceProvenance(
        schema_version="phase1-evidence-provenance-v1",
        plan_base_revision="b" * 40,
        source_commit="c" * 40,
        source_dirty=False,
        generator_version="ofd-v1",
        generation_mode="matched",
        allocation_id="validation-v1",
        split_namespace=split,
        config_sha256="d" * 64,
        generator_source=fingerprint,
        analysis_source=analysis,
        root_seed=17,
        public_id_seed_sha256="3acaee04f18b5609e8d4bdd6a5ab1ee2e24e8143bb1b33a3de05204cb195d980",
    )


@pytest.fixture
def manifest() -> EpisodeManifest:
    entries = tuple(
        EpisodeManifestEntry(
            episode_public_id=f"00000000-0000-4000-8000-00000000000{member}",
            split_namespace=SplitNamespace.VALIDATION,
            suite=SuiteName.VALIDATION,
            coordinate=MatchedManifestCoordinate(cohort_index=4, member_index=member),
            requested_path_length=2,
            accepted_attempt=0,
            episode_sha256=f"{member:x}" * 64,
        )
        for member in range(4)
    )
    return EpisodeManifest(
        schema_version=1,
        experiment_version="v1",
        access_class=ManifestAccessClass.VALIDATION,
        provenance=_provenance(),
        suite=SuiteName.VALIDATION,
        public_id_seed=91,
        episode_count=4,
        entries=entries,
    )


def test_manifest_round_trip_is_canonical_and_verified(
    tmp_path: Path, manifest: EpisodeManifest
) -> None:
    path = tmp_path / "validation.json"

    published = publish_manifest(path, manifest)

    assert published.created
    assert load_manifest(path) == manifest
    envelope = ManifestEnvelope.model_validate_json(path.read_bytes())
    assert envelope.payload_sha256 == published.payload_sha256
    assert verify_manifest(envelope) == manifest


def test_identical_manifest_is_verified_without_rewrite(
    tmp_path: Path, manifest: EpisodeManifest
) -> None:
    path = tmp_path / "validation.json"
    first = publish_manifest(path, manifest)
    before = path.stat().st_mtime_ns

    second = publish_manifest(path, manifest)

    assert first.file_sha256 == second.file_sha256
    assert not second.created
    assert path.stat().st_mtime_ns == before


def test_divergent_existing_manifest_requires_new_version(
    tmp_path: Path, manifest: EpisodeManifest
) -> None:
    path = tmp_path / "validation.json"
    publish_manifest(path, manifest)
    provenance = manifest.provenance.model_dump()
    provenance["public_id_seed_sha256"] = (
        "68a2c5b8283857fd0e511fef3b67123544062c2da7605f3201db2a23df147cdc"
    )
    divergent = EpisodeManifest.model_validate(
        {**manifest.model_dump(), "public_id_seed": 99, "provenance": provenance}
    )

    with pytest.raises(ManifestError, match="different immutable manifest already exists"):
        publish_manifest(path, divergent)


def test_spawned_publishers_never_create_partial_or_divergent_manifests(
    tmp_path: Path, manifest: EpisodeManifest
) -> None:
    path = tmp_path / "validation.json"
    context = get_context("spawn")
    result_queue = context.Queue()
    first = context.Process(
        target=_publish_in_child, args=(str(path), manifest.model_dump(), result_queue)
    )
    second = context.Process(
        target=_publish_in_child, args=(str(path), manifest.model_dump(), result_queue)
    )
    first.start()
    second.start()
    first.join(timeout=15)
    second.join(timeout=15)

    assert first.exitcode == 0
    assert second.exitcode == 0
    assert sorted(result_queue.get(timeout=2) for _ in range(2)) == [
        ("published", False),
        ("published", True),
    ]
    assert load_manifest(path) == manifest


def test_spawned_divergent_publishers_allow_exactly_one_complete_winner(
    tmp_path: Path, manifest: EpisodeManifest
) -> None:
    path = tmp_path / "validation.json"
    provenance = manifest.provenance.model_dump()
    provenance["public_id_seed_sha256"] = (
        "68a2c5b8283857fd0e511fef3b67123544062c2da7605f3201db2a23df147cdc"
    )
    divergent = EpisodeManifest.model_validate(
        {**manifest.model_dump(), "public_id_seed": 99, "provenance": provenance}
    )
    context = get_context("spawn")
    result_queue = context.Queue()
    first = context.Process(
        target=_publish_in_child, args=(str(path), manifest.model_dump(), result_queue)
    )
    second = context.Process(
        target=_publish_in_child, args=(str(path), divergent.model_dump(), result_queue)
    )
    first.start()
    second.start()
    first.join(timeout=15)
    second.join(timeout=15)

    assert first.exitcode == 0
    assert second.exitcode == 0
    assert sorted(result_queue.get(timeout=2) for _ in range(2)) == [
        ("collision", False),
        ("published", True),
    ]
    assert load_manifest(path) in (manifest, divergent)


@pytest.mark.parametrize("mutation", ["payload_hash", "episode_hash", "unknown_field"])
def test_loader_rejects_tampered_or_non_strict_envelopes(
    tmp_path: Path, manifest: EpisodeManifest, mutation: str
) -> None:
    path = tmp_path / "validation.json"
    publish_manifest(path, manifest)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "payload_hash":
        raw["payload_sha256"] = "f" * 64
    elif mutation == "episode_hash":
        raw["payload"]["entries"][0]["episode_sha256"] = "f" * 64
    else:
        raw["unexpected"] = True
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ManifestError):
        load_manifest(path)


def test_manifest_rejects_duplicate_public_ids_and_recipes(manifest: EpisodeManifest) -> None:
    duplicate_id = manifest.entries[1].model_copy(
        update={"episode_public_id": manifest.entries[0].episode_public_id}
    )
    with pytest.raises(ValueError, match="public IDs"):
        EpisodeManifest.model_validate(
            {
                **manifest.model_dump(),
                "entries": (manifest.entries[0], duplicate_id, *manifest.entries[2:]),
            }
        )

    duplicate_recipe = manifest.entries[1].model_copy(
        update={"coordinate": manifest.entries[0].coordinate}
    )
    with pytest.raises(ValueError, match="coordinates"):
        EpisodeManifest.model_validate(
            {
                **manifest.model_dump(),
                "entries": (manifest.entries[0], duplicate_recipe, *manifest.entries[2:]),
            }
        )


def test_entry_rejects_non_hex_paired_parent_digest() -> None:
    with pytest.raises(ValueError, match="parent"):
        EpisodeManifestEntry(
            episode_public_id="00000000-0000-4000-8000-000000000010",
            split_namespace=SplitNamespace.FROZEN,
            suite=SuiteName.CLOCK_SCALE_0_1X,
            coordinate=MatchedManifestCoordinate(cohort_index=1, member_index=0),
            requested_path_length=2,
            accepted_attempt=0,
            episode_sha256="1" * 64,
            parent_public_id="00000000-0000-4000-8000-000000000011",
            parent_episode_sha256="x" * 64,
            clock_scale=0.1,
        )


def test_validation_manifest_rejects_wrong_namespace_mode_and_membership(
    manifest: EpisodeManifest,
) -> None:
    wrong_namespace = _provenance(split=SplitNamespace.DEBUG)
    with pytest.raises(ValueError, match="validation"):
        EpisodeManifest.model_validate({**manifest.model_dump(), "provenance": wrong_namespace})

    independent = _provenance().model_copy(update={"generation_mode": "independent"})
    with pytest.raises(ValueError, match="coordinate mode"):
        EpisodeManifest.model_validate({**manifest.model_dump(), "provenance": independent})

    nonempty_membership = manifest.entries[0].model_copy(
        update={"subset_memberships": ("curve-1",)}
    )
    with pytest.raises(ValueError, match="validation manifests"):
        EpisodeManifest.model_validate(
            {**manifest.model_dump(), "entries": (nonempty_membership, *manifest.entries[1:])}
        )


def test_frozen_manifest_denies_oracle_inspection_by_access_class(
    manifest: EpisodeManifest,
) -> None:
    frozen = manifest.model_copy(update={"access_class": ManifestAccessClass.FROZEN_TEST})

    with pytest.raises(ManifestAccessError, match="forbidden"):
        require_oracle_inspection_allowed(frozen)
