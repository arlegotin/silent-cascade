from dataclasses import fields

import numpy as np
import torch


def test_online_batch_is_counter_addressable(neural_config):
    from silent_cascade.train.batches import next_training_batch

    first = next_training_batch(neural_config, stage="one_hop", batch_counter=7)
    following = next_training_batch(neural_config, stage="one_hop", batch_counter=8)
    np.random.seed(912)
    torch.manual_seed(888)
    repeat = next_training_batch(neural_config, stage="one_hop", batch_counter=7)
    assert first.example_hashes == repeat.example_hashes
    assert first.example_hashes != following.example_hashes
    assert first.next_batch_counter == 8
    for field in fields(first.targets):
        a, b = getattr(first.targets, field.name), getattr(repeat.targets, field.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
    for field in fields(first.public):
        a, b = getattr(first.public, field.name), getattr(repeat.public, field.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
    for field in fields(first.public.records):
        a, b = getattr(first.public.records, field.name), getattr(repeat.public.records, field.name)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        else:
            assert a == b
    for name in first.targets.validity:
        assert torch.equal(first.targets.validity[name], repeat.targets.validity[name])


def test_batch_masks_slots_projection_and_cpu_metadata(neural_config):
    from silent_cascade.train.batches import next_training_batch

    batch = next_training_batch(neural_config, stage="one_hop", batch_counter=0).to("cpu")
    public = batch.public_inputs()
    assert not hasattr(public, "targets") and not hasattr(public, "example_keys")
    assert public.observation_mask.shape[1] == max(len(e.public.events) for e in batch.examples)
    targets = batch.targets
    assert torch.equal(batch.operation_masks["action"], targets.kind == 2)
    assert torch.equal(batch.operation_masks["abstention"], targets.action_class == 4)
    assert targets.kind.shape[1] == 5
    for name, mask in targets.validity.items():
        assert mask.dtype is torch.bool and mask.shape == targets.kind.shape
        assert not (mask & ~targets.real_step_mask).any()
        tensor = getattr(targets, name)
        if tensor.dtype is torch.int64:
            assert (tensor[~mask] == -1).all()
        else:
            assert tensor.dtype is torch.float32
    for row, trace in enumerate(batch.teacher_traces):
        for col, step in enumerate(trace.steps):
            if step.kind.value == "recall":
                slot = int(targets.selected_slot[row, col])
                assert public.records.record_ids[row][slot] == step.selected_record_id
    assert batch.teacher_times.dtype is torch.float64 and batch.teacher_times.device.type == "cpu"
    for i in range(public.observation_mask.shape[1]):
        prefix = public.at_observation(i)
        for row in range(public.records.batch_size):
            now = float(public.observation_times[row, i])
            for slot, arrived in enumerate(prefix.records.valid_mask[row].tolist()):
                if arrived:
                    assert prefix.records.observed_at[row][slot] <= now
                else:
                    assert prefix.records.record_ids[row][slot] is None
                    assert prefix.records.observed_at[row][slot] is None
                    assert (prefix.records.scalar_features[row, slot] == 0).all()
        assert prefix.observation_mask.shape[1] == 1
    permutation = torch.stack([torch.arange(63, -1, -1)] * public.records.batch_size)
    permuted = batch.permute_slots(permutation)
    for row, trace in enumerate(batch.teacher_traces):
        for col, step in enumerate(trace.steps):
            if step.kind.value == "recall":
                slot = int(permuted.targets.selected_slot[row, col])
                assert (
                    permuted.public_inputs().records.record_ids[row][slot]
                    == step.selected_record_id
                )


def test_batch_generation_preserves_all_global_rng(neural_config):
    import random

    from silent_cascade.train.batches import next_training_batch

    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    next_training_batch(neural_config, stage="one_hop", batch_counter=2)
    assert random.getstate() == python_state
    actual = np.random.get_state()
    assert np.array_equal(numpy_state[1], actual[1]) and numpy_state[2:] == actual[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_target_padding_has_zero_loss_gradient(neural_config):
    from silent_cascade.train.batches import next_training_batch

    targets = next_training_batch(neural_config, stage="one_hop", batch_counter=0).targets
    logits = torch.zeros((*targets.kind.shape, 3), requires_grad=True)
    mask = targets.validity["kind"]
    torch.nn.functional.cross_entropy(logits[mask], targets.kind[mask]).backward()
    assert not logits.grad[~mask].any()
    assert logits.grad[mask].abs().sum() > 0


def test_final_dormancy_cannot_mark_padding(neural_config):
    from dataclasses import replace

    import pytest

    from silent_cascade.train.batches import next_training_batch

    targets = next_training_batch(neural_config, stage="one_hop", batch_counter=0).targets
    with pytest.raises(ValueError, match="dormancy"):
        replace(targets, final_dormancy_mask=~targets.real_step_mask)


def test_native_mps_keeps_absolute_metadata_on_cpu(neural_config):
    import pytest

    from silent_cascade.train.batches import next_training_batch

    if not torch.backends.mps.is_available():
        pytest.skip("MPS not exposed in this process; native gate is a separate invocation")
    batch = next_training_batch(neural_config, stage="one_hop", batch_counter=1).to("mps")
    assert batch.targets.delta.device.type == "mps"
    assert batch.targets.delta.dtype == torch.float32
    assert batch.teacher_times.device.type == "cpu"
    assert batch.public.observation_times.dtype == torch.float64
    assert batch.public.observation_times.device.type == "cpu"
    assert batch.public.at_observation(0).records.scalar_features.device.type == "mps"
