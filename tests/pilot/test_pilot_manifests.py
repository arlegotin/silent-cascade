import json

import pytest

from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.train.pilot_config import resolve_pilot_config


@pytest.mark.parametrize("stage", ["one_hop", "two_hop", "primary", "robustness"])
def test_manifest_roundtrip_authenticates_ordered_projected_examples(tmp_path, stage):
    from silent_cascade.train.pilot_data import (
        freeze_pilot_manifest,
        iter_pilot_examples,
        load_pilot_manifest,
        pilot_component_corpus,
    )

    resolved = resolve_pilot_config("phase4_smoke")
    path = tmp_path / f"{stage}.json"
    manifest = freeze_pilot_manifest(
        resolved, stage=stage, output_path=path, source_commit="a" * 40
    )
    assert manifest.schema_version == "phase4-data-v1"
    assert manifest.experiment == "phase4-pilot-v1"
    assert manifest.count == 16
    assert manifest.split == "debug"
    assert manifest.variant_counts == {
        "positive": 8,
        "safe_negative": 4,
        "disconnected_negative": 4,
    }
    assert load_pilot_manifest(path, config=resolved) == manifest
    assert (
        freeze_pilot_manifest(resolved, stage=stage, output_path=path, source_commit="a" * 40)
        == manifest
    )
    examples = tuple(iter_pilot_examples(manifest, config=resolved.config))
    assert [e.example_hash for e in examples] == [e.example_hash for e in manifest.entries]
    assert {e.key.root_seed for e in manifest.entries} == {449}
    assert {e.key.public_id_seed for e in manifest.entries} == {457}
    assert all(e.projected.transform is None for e in manifest.entries)
    assert all(e.parent_public_id != e.projected.public_id for e in manifest.entries)
    if stage == "one_hop":
        corpus = pilot_component_corpus(manifest, config=resolved.config)
        assert corpus.public_examples == tuple(e.public for e in examples)
        assert len(corpus.targets) == 16
        assert {t.terminal_class for t in corpus.targets}.issubset(set(range(5)))
    else:
        with pytest.raises(ValueError, match=r"one.hop"):
            pilot_component_corpus(manifest, config=resolved.config)


@pytest.mark.parametrize(
    "corruption", ["hash", "duplicate", "count", "root", "unknown", "duplicate_json"]
)
def test_manifest_fails_closed_on_corruption(tmp_path, corruption):
    from silent_cascade.train.pilot_data import freeze_pilot_manifest, load_pilot_manifest

    resolved = resolve_pilot_config("phase4_smoke")
    path = tmp_path / "manifest.json"
    freeze_pilot_manifest(resolved, stage="one_hop", output_path=path, source_commit="a" * 40)
    values = json.loads(path.read_bytes())
    if corruption == "hash":
        values["entries"][0]["target_hash"] = "f" * 64
    elif corruption == "duplicate":
        values["entries"][1] = values["entries"][0]
    elif corruption == "count":
        values["count"] = 12
    elif corruption == "root":
        values["entries"][0]["key"]["root_seed"] = 313
    elif corruption == "unknown":
        values["unexpected"] = True
    payload = canonical_json_bytes(values)
    if corruption == "duplicate_json":
        payload = b'{"count":16,' + payload[1:]
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        load_pilot_manifest(path, config=resolved)


def test_manifest_refuses_overwrite_symlinks_frozen_and_oversize(tmp_path):
    from silent_cascade.train.pilot_data import freeze_pilot_manifest, load_pilot_manifest

    resolved = resolve_pilot_config("phase4_smoke")
    path = tmp_path / "manifest.json"
    freeze_pilot_manifest(resolved, stage="one_hop", output_path=path, source_commit="a" * 40)
    saved = path.read_bytes()
    with pytest.raises(ValueError, match=r"exist|overwrite"):
        freeze_pilot_manifest(resolved, stage="primary", output_path=path, source_commit="a" * 40)
    assert path.read_bytes() == saved
    link = tmp_path / "link.json"
    link.symlink_to(path)
    directory_link = tmp_path / "directory"
    directory_link.symlink_to(tmp_path, target_is_directory=True)
    for target in (link, directory_link / path.name, tmp_path / "manifests/frozen/pilot.json"):
        with pytest.raises((ValueError, OSError)):
            load_pilot_manifest(target, config=resolved)
        with pytest.raises((ValueError, OSError)):
            freeze_pilot_manifest(
                resolved, stage="one_hop", output_path=target, source_commit="a" * 40
            )
    large = tmp_path / "large.json"
    with large.open("wb") as handle:
        handle.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="byte_limit"):
        load_pilot_manifest(large, config=resolved)


def test_production_identity_cannot_accept_debug_inventory(tmp_path):
    from silent_cascade.train.pilot_data import freeze_pilot_manifest, iter_pilot_examples

    debug = resolve_pilot_config("phase4_smoke")
    production = resolve_pilot_config("phase4_pilot")
    manifest = freeze_pilot_manifest(
        debug, stage="one_hop", output_path=tmp_path / "debug.json", source_commit="a" * 40
    )
    forged = manifest.model_copy(
        update={
            "split": "validation",
            "config_hash": production.sha256,
            "config_canonical_json": production.canonical_json.decode(),
        }
    )
    with pytest.raises(ValueError, match="count"):
        tuple(iter_pilot_examples(forged, config=production.config))


def test_train_validation_debug_keys_generate_disjoint_public_ids():
    from silent_cascade.train.curriculum_data import CurriculumKey, make_curriculum_example

    config = resolve_pilot_config("phase4_smoke").config
    ids = []
    for split, root, opaque in (("train", 431, 433), ("validation", 439, 443), ("debug", 449, 457)):
        for stage in ("one_hop", "two_hop", "primary", "robustness"):
            example = make_curriculum_example(
                config, CurriculumKey("ofd-one-hop-v1", split, root, opaque, 0, stage)
            )
            ids.append(example.public.init.episode_public_id)
    assert len(set(ids)) == 12


def test_manifest_bindings_are_valid_normal_and_delay_evaluation_inputs(tmp_path):
    from silent_cascade.env.episode import episode_sha256
    from silent_cascade.env.pilot import curriculum_to_bundle, delay_pair
    from silent_cascade.eval.artifacts import DelayTransform, EpisodeBinding, EvaluationIdentity
    from silent_cascade.eventflow.neural import NeuralModelIdentity
    from silent_cascade.hashing import sha256_bytes
    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.train.pilot_data import freeze_pilot_manifest, iter_pilot_examples

    resolved = resolve_pilot_config("phase4_smoke")
    manifest = freeze_pilot_manifest(
        resolved, stage="primary", output_path=tmp_path / "m.json", source_commit="a" * 40
    )
    identity = NeuralModelIdentity.from_model(
        EventFlowModel(resolved.config.neural), source_revision="a" * 40
    )
    fields = dict(
        experiment="phase4-pilot-v1",
        stage="primary",
        split="debug",
        manifest_schema=manifest.schema_version,
        manifest_sha256=sha256_bytes(canonical_json_bytes(manifest)),
        checkpoint_sha256="b" * 64,
        model_identity=identity,
        evaluation_config_canonical_json=resolved.canonical_json.decode(),
        execution_source_revision="a" * 40,
    )
    ordinary = EvaluationIdentity(
        **fields, purpose="debug", episodes=tuple(e.projected for e in manifest.entries)
    )
    assert len(ordinary.episodes) == 16
    parent = curriculum_to_bundle(
        next(iter_pilot_examples(manifest, config=resolved.config)), config=resolved.config
    )
    child = delay_pair(parent, delays=(16.0, 48.0))[1]
    binding = EpisodeBinding.from_bundle(
        child,
        transform=DelayTransform(
            parent_public_id=parent.public.init.episode_public_id,
            parent_episode_sha256=episode_sha256(parent),
            version="phase4-delay-pair-v1",
            parameters_canonical_json=canonical_json_bytes({"delay": 48.0}).decode(),
        ),
    )
    paired = EvaluationIdentity(
        **fields,
        purpose="delay_swap",
        parent_manifest_sha256=ordinary.manifest_sha256,
        episodes=(binding,),
    )
    assert paired.episodes[0].episode_sha256 == episode_sha256(child)
    assert paired.episodes[0].transform.parent_episode_sha256 == episode_sha256(parent)
    assert paired.episodes[0].scoring_truth_sha256 != ordinary.episodes[0].scoring_truth_sha256
