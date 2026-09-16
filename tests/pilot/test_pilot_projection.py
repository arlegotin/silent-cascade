from dataclasses import replace

import pytest

from silent_cascade.env.episode import EpisodeVariant, episode_sha256
from silent_cascade.env.oracle import solve_public_episode, verify_oracle_truth
from silent_cascade.errors import EpisodeInvariantError
from silent_cascade.schemas import HazardFact
from silent_cascade.train.curriculum_data import CurriculumKey, make_curriculum_example
from silent_cascade.train.pilot_config import resolve_pilot_config


@pytest.mark.parametrize("stage", ["one_hop", "two_hop", "primary", "robustness"])
def test_projection_matches_transformed_public_truth(stage):
    from silent_cascade.env.pilot import curriculum_to_bundle

    config = resolve_pilot_config("phase4_smoke").config
    variants, factors = set(), []
    for index in range(16):
        example = make_curriculum_example(
            config, CurriculumKey("ofd-one-hop-v1", "debug", 449, 457, index, stage)
        )
        bundle = curriculum_to_bundle(example, config=config)
        assert bundle.public == example.public
        assert solve_public_episode(bundle.public) == example.solution
        verify_oracle_truth(example.solution, bundle.truth)
        assert bundle.truth.private_terminal == example.terminal_horizon
        assert bundle.truth.recipe.requested_path_length == len(example.solution.link_record_ids)
        assert bundle.truth.recipe.distractor_link_count == example.nuisance_link_count
        assert bundle.truth.key.coordinate.episode_index == index * 1000 + example.accepted_attempt
        assert bundle.truth.recipe.clock_scale == 1.0
        assert bundle.truth.recipe.oracle_timing.delta_min == (
            config.data.oracle_timing.delta_min * example.time_factor
        )
        variants.add(example.variant)
        factors.append(example.time_factor)
    assert variants == set(EpisodeVariant)
    if stage == "robustness":
        assert min(factors) < 1.0 < max(factors)


def test_projection_rejects_forged_parent_and_trace_and_accepts_historical_roots():
    from silent_cascade.env.pilot import curriculum_to_bundle

    config = resolve_pilot_config("phase4_smoke").config
    example = make_curriculum_example(
        config, CurriculumKey("ofd-one-hop-v1", "validation", 313, 337, 0, "one_hop")
    )
    assert curriculum_to_bundle(example, config=config).public == example.public
    for bad in (replace(example, parent_hash="a" * 64), replace(example, target_hash="b" * 64)):
        with pytest.raises(EpisodeInvariantError, match=r"identity|hash"):
            curriculum_to_bundle(bad, config=config)


def test_shared_accepted_parent_regeneration_authenticates_parent_hash():
    from silent_cascade.env.pilot import _accepted_parent

    config = resolve_pilot_config("phase4_smoke").config
    example = make_curriculum_example(
        config, CurriculumKey("ofd-one-hop-v1", "debug", 449, 457, 0, "robustness")
    )
    parent = _accepted_parent(example, config=config)
    assert episode_sha256(parent) == example.parent_hash
    assert parent.truth.key.coordinate.episode_index == example.accepted_attempt
    with pytest.raises(EpisodeInvariantError, match="parent identity"):
        _accepted_parent(replace(example, parent_hash="a" * 64), config=config)


@pytest.mark.parametrize("stage", ["one_hop", "two_hop", "primary", "robustness"])
def test_delay_pairs_change_both_hazards_and_scoring_only(stage):
    from silent_cascade.env.pilot import curriculum_to_bundle, delay_pair

    config = resolve_pilot_config("phase4_smoke").config
    for index in range(4):
        example = make_curriculum_example(
            config, CurriculumKey("ofd-one-hop-v1", "debug", 449, 457, index, stage)
        )
        parent = curriculum_to_bundle(example, config=config)
        original = episode_sha256(parent)
        children = delay_pair(parent, delays=(16.0, 48.0))
        assert episode_sha256(parent) == original
        assert episode_sha256(children[0]) != episode_sha256(children[1])
        for child, delay in zip(children, (16.0, 48.0), strict=True):
            assert child.public.init == parent.public.init
            assert child.truth.private_terminal.timestamp == child.truth.activation_time + delay
            hazards = []
            for before, after in zip(parent.public.events, child.public.events, strict=True):
                assert (before.event_id, before.timestamp, before.kind) == (
                    after.event_id,
                    after.timestamp,
                    after.kind,
                )
                if isinstance(after.payload, HazardFact):
                    hazards.append(after.payload.delay)
                    assert replace(after.payload, delay=before.payload.delay) == before.payload
                else:
                    assert before == after
            assert hazards == [delay, delay]
            verify_oracle_truth(solve_public_episode(child.public), child.truth)
            if example.variant is EpisodeVariant.POSITIVE:
                assert child.truth.action_window_start == child.truth.activation_time + 0.75 * delay
                assert child.truth.action_window_end == child.truth.activation_time + 0.90 * delay
            else:
                assert child.truth.action_window_start is None


def test_phase3_example_hashes_are_preserved(neural_config):
    expected = {
        "one_hop": "eb7bf9c2cd63fcbe84394f0d13e985bcfd62b4602090af4ae3ff4ca763986c01",
        "two_hop": "49946872f58b5d490059c9ea39bd4455ce3935435509177d53fabc3cf58daacd",
        "primary": "e5984e399f1ea4d325d7410b298b2bad98cb6a4d37606cd66c804fa6c92973c6",
        "robustness": "5f80b17b6d0a18953a652b29a4beeb7d7faa5f27c668cae83cd6f1d553d828db",
    }
    for stage, digest in expected.items():
        key = CurriculumKey("ofd-one-hop-v1", "validation", 313, 337, 0, stage)
        assert make_curriculum_example(neural_config, key).example_hash == digest
