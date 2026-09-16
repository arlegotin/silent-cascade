import pytest
import torch

from silent_cascade.train.pilot_config import resolve_pilot_config


@pytest.mark.parametrize("stage", ["one_hop", "two_hop", "primary", "robustness"])
def test_counter_resume_is_exact_and_stages_are_separate(stage):
    from silent_cascade.train.pilot_data import next_pilot_batch

    config = resolve_pilot_config("phase4_smoke").config
    first = next_pilot_batch(config, stage=stage, batch_counter=0)
    resumed = next_pilot_batch(config, stage=stage, batch_counter=first.next_batch_counter)
    replay = next_pilot_batch(config, stage=stage, batch_counter=1)
    assert first.next_batch_counter == 1
    assert resumed.next_batch_counter == 2
    assert resumed.example_hashes == replay.example_hashes
    assert not set(first.example_hashes) & set(resumed.example_hashes)
    assert {key.root_seed for key in first.example_keys} == {431}
    assert {key.public_id_seed for key in first.example_keys} == {433}
    assert torch.equal(resumed.targets.kind, replay.targets.kind)
    assert torch.equal(resumed.teacher_times, replay.teacher_times)


@pytest.mark.parametrize("counter", [-1, True, 0.5])
def test_batch_counter_is_exact_nonnegative(counter):
    from silent_cascade.train.pilot_data import next_pilot_batch

    with pytest.raises((TypeError, ValueError)):
        next_pilot_batch(
            resolve_pilot_config("phase4_smoke").config, stage="primary", batch_counter=counter
        )
