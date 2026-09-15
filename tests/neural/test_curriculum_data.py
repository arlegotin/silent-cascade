from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from silent_cascade.env.episode import EpisodeVariant
from silent_cascade.schemas import HazardFact, LinkFact, SafeFact


def key(index=0, stage="one_hop", split="debug"):
    from silent_cascade.train.curriculum_data import CurriculumKey

    return CurriculumKey("ofd-one-hop-v1", split, 311, 331, index, stage)


def test_repeatability_domains_and_global_rng(neural_config):
    from silent_cascade.train.curriculum_data import make_curriculum_example

    np.random.seed(71)
    expected = np.random.get_state()
    example = make_curriculum_example(neural_config, key())
    actual = np.random.get_state()
    assert np.array_equal(expected[1], actual[1]) and expected[2:] == actual[2:]
    np.random.seed(8)
    assert example == make_curriculum_example(neural_config, key())
    assert (
        len(
            {
                make_curriculum_example(neural_config, key(split=s)).example_hash
                for s in ("debug", "train", "validation")
            }
        )
        == 3
    )
    other_id = make_curriculum_example(neural_config, replace(key(), public_id_seed=337))
    assert example.public.events == other_id.public.events
    assert example.public.init.episode_public_id != other_id.public.init.episode_public_id


def test_one_hop_topology_counts_permutations_and_balance(neural_config):
    from silent_cascade.train.curriculum_data import make_curriculum_example

    examples = [make_curriculum_example(neural_config, key(i)) for i in range(80)]
    assert Counter(e.variant for e in examples) == {
        EpisodeVariant.POSITIVE: 40,
        EpisodeVariant.SAFE_NEGATIVE: 20,
        EpisodeVariant.DISCONNECTED_NEGATIVE: 20,
    }
    assert {e.nuisance_link_count for e in examples} == set(range(5))
    assert len({e.solution.link_record_ids[0] for e in examples}) > 1
    for example in examples:
        facts = example.public.events[:-1]
        assert len(example.solution.link_record_ids) == 1
        assert sum(isinstance(e.payload, HazardFact) for e in facts) == 2
        assert sum(isinstance(e.payload, SafeFact) for e in facts) == 1
        assert (
            sum(isinstance(e.payload, LinkFact) for e in facts) == 1 + example.nuisance_link_count
        )
        assert [e.event_id for e in example.public.events] == list(range(len(facts) + 1))
        assert len(example.oracle_trace.steps) <= 5
        assert example.foundation_model_calls == 0
        assert example.parent_recipe.requested_path_length == 2


@pytest.mark.parametrize("stage", ["two_hop", "primary", "robustness"])
def test_all_stages_constructible(neural_config, stage):
    from silent_cascade.train.curriculum_data import make_curriculum_example

    for i in range(8):
        example = make_curriculum_example(neural_config, key(i, stage))
        assert len(example.solution.link_record_ids) in ((2,) if stage == "two_hop" else (2, 3, 4))
        assert len(example.oracle_trace.steps) <= 11
        assert 0.5 <= example.time_factor <= 2.0
        assert all(
            s.timestamp < example.terminal_horizon.timestamp for s in example.oracle_trace.steps
        )
        if stage == "robustness":
            for original, scaled in zip(
                example.paired_unscaled_trace.steps, example.oracle_trace.steps, strict=True
            ):
                assert scaled.delta == pytest.approx(
                    original.delta * example.time_factor, rel=1e-10
                )
                assert scaled.timestamp == pytest.approx(
                    original.timestamp * example.time_factor, rel=1e-10
                )
            for original, scaled in zip(
                example.paired_unscaled_public.events, example.public.events, strict=True
            ):
                assert scaled.timestamp == pytest.approx(original.timestamp * example.time_factor)
                if isinstance(original.payload, HazardFact):
                    assert scaled.payload.delay == pytest.approx(
                        original.payload.delay * example.time_factor
                    )


def test_fixed_validation_quartets_are_exactly_balanced():
    from silent_cascade.train.curriculum_data import curriculum_allocation

    counts = Counter(
        curriculum_allocation(replace(key(i, split="validation"), root_seed=313))[2]
        for i in range(10_000)
    )
    assert counts == {
        EpisodeVariant.POSITIVE: 5000,
        EpisodeVariant.SAFE_NEGATIVE: 2500,
        EpisodeVariant.DISCONNECTED_NEGATIVE: 2500,
    }


@pytest.mark.parametrize(
    "changes",
    [
        dict(split="frozen"),
        dict(stage="ood_depth"),
        dict(root_seed=True),
        dict(episode_index=-1),
        dict(public_id_seed=2**128),
        dict(version="future"),
    ],
)
def test_private_key_rejects_unsupported_scope(changes):
    with pytest.raises(ValueError):
        replace(key(), **changes)


def test_one_hop_has_no_primary_generator_override(neural_config):
    from silent_cascade.env.config import SuiteName
    from silent_cascade.env.generator import suite_spec

    assert suite_spec(SuiteName.IID_PRIMARY).path_lengths == (2, 3, 4)
    assert neural_config.data.train_path_lengths == (2, 3, 4)


def test_debug_manifest_round_trip_tampering_and_target_isolation(neural_config, tmp_path):
    from silent_cascade.train.curriculum_data import (
        ComponentCorpus,
        ComponentManifest,
        load_component_manifest,
        make_curriculum_example,
    )

    examples = tuple(make_curriculum_example(neural_config, key(i)) for i in range(4))
    manifest = ComponentManifest.from_examples(
        examples,
        neural_config,
        source_revision="a" * 40,
        plan_revision="b" * 40,
    )
    path = tmp_path / "debug.json"
    path.write_text(manifest.model_dump_json())
    loaded = load_component_manifest(path, neural_config)
    assert loaded == manifest
    corpus = loaded.build_corpus(neural_config)
    assert isinstance(corpus, ComponentCorpus)
    assert corpus.scoring_slice(1, 2) == corpus.targets[1:3]
    with pytest.raises(ValueError, match="slice"):
        corpus.public_batch(3, 2)
    for target in corpus.targets:
        assert tuple(content.record_id for content in target.content) == target.record_ids
        assert target.content[0].role == 0
        assert target.content[0].focus is not None
        assert target.content[0].confidence == 1.0
    changed = replace(corpus, targets=tuple(replace(t, terminal_class=4) for t in corpus.targets))
    assert corpus.public_examples == changed.public_examples
    assert (
        corpus.public_batch(0, 4).records.record_ids
        == changed.public_batch(0, 4).records.record_ids
    )
    data = manifest.model_dump(mode="json")
    data["entries"][0]["example_hash"] = "0" * 64
    import json

    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        load_component_manifest(path, neural_config)
    actual_path = tmp_path / "real.json"
    actual_path.write_text(manifest.model_dump_json())
    path.unlink()
    path.symlink_to(actual_path)
    with pytest.raises(OSError):
        load_component_manifest(path, neural_config)


def test_manifest_reader_is_bounded(neural_config, tmp_path):
    from silent_cascade.train.curriculum_data import (
        MAX_COMPONENT_MANIFEST_BYTES,
        load_component_manifest,
    )

    path = tmp_path / "oversized.json"
    with path.open("wb") as stream:
        stream.truncate(MAX_COMPONENT_MANIFEST_BYTES + 1)
    with pytest.raises(ValueError, match="byte_limit"):
        load_component_manifest(path, neural_config)
