"""Behavioral tests for the shared public record encoder."""

from dataclasses import replace

import pytest
import torch


def test_encoder_shape_padding_and_slot_permutation_equivariance(
    public_memories, neural_config
) -> None:
    from silent_cascade.memory.encoder import RecordEncoder
    from silent_cascade.memory.tensor_store import pack_memory

    records = pack_memory(public_memories, (0.0, 9.0), device="cpu")
    permutations = torch.arange(64).repeat(2, 1)
    permutations[0, :3] = torch.tensor([2, 0, 1])
    encoder = RecordEncoder(neural_config.neural)

    encoded = encoder(records)
    encoded_permuted = encoder(records.permute_slots(permutations))

    assert encoded.shape == (2, 64, 96)
    assert not bool(encoded[:, 3:].any())
    torch.testing.assert_close(
        encoded_permuted,
        encoded.gather(1, permutations.unsqueeze(-1).expand(-1, -1, 96)),
    )


def test_record_ids_and_slot_indices_do_not_enter_embeddings(
    public_memories, neural_config
) -> None:
    from silent_cascade.memory.encoder import RecordEncoder
    from silent_cascade.memory.tensor_store import pack_memory

    memory = public_memories[0]
    renamed = replace(
        memory,
        records=tuple(
            replace(slot, record=replace(slot.record, record_id=slot.record.record_id + 100))
            for slot in memory.records
        ),
    )
    encoder = RecordEncoder(neural_config.neural)
    left = encoder(pack_memory((memory,), (0.0,), device="cpu"))
    right = encoder(pack_memory((renamed,), (0.0,), device="cpu"))

    torch.testing.assert_close(left, right, rtol=0, atol=0)
    shifted = torch.arange(64).roll(7).unsqueeze(0)
    shifted_records = pack_memory((memory,), (0.0,), device="cpu").permute_slots(shifted)
    torch.testing.assert_close(
        encoder(shifted_records), left.gather(1, shifted.unsqueeze(-1).expand(-1, -1, 96))
    )


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_shared_entity_table_receives_finite_record_and_activation_gradients(
    public_memories, neural_config, device
) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    from silent_cascade.memory.encoder import RecordEncoder
    from silent_cascade.memory.tensor_store import pack_memory

    encoder = RecordEncoder(neural_config.neural).to(device)
    records = pack_memory((public_memories[0],), (0.0,), device=device)
    record_loss = encoder(records).sum()
    activation_loss = encoder.encode_entities(torch.tensor([1, 2], device=device)).sum()
    (record_loss + activation_loss).backward()

    gradient = encoder.entity_embedding.weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert bool(gradient[1].any())
    assert bool(gradient[2].any())


def test_absent_object_is_masked_from_entity_embedding(public_memories, neural_config) -> None:
    from silent_cascade.memory.encoder import RecordEncoder
    from silent_cascade.memory.tensor_store import pack_memory

    encoder = RecordEncoder(neural_config.neural)
    records = pack_memory((public_memories[0],), (0.0,), device="cpu")
    loss = encoder(records)[0, 1:3].sum()
    loss.backward()

    gradient = encoder.entity_embedding.weight.grad
    assert gradient is not None
    assert not bool(gradient[0].any())
