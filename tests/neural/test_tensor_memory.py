"""Behavioral tests for public metadata-to-tensor memory packing."""

from dataclasses import replace

import pytest
import torch

from silent_cascade.models.errors import NeuralError


def test_pack_memory_encodes_public_fields_time_origin_and_padding(public_memories) -> None:
    from silent_cascade.memory.tensor_store import pack_memory

    records = pack_memory(public_memories, (0.5, 9.0), device="cpu")

    assert records.subject_ids.shape == (2, 64)
    assert records.subject_ids.dtype is torch.int64
    assert records.scalar_features.dtype is torch.float32
    assert records.valid_mask.dtype is torch.bool
    assert records.record_ids[0][:4] == (11, 12, 13, None)
    assert records.support_ids[0][:4] == ((), (), (), ())
    assert records.observed_at[0][:4] == (1.0, 2.5, 4.0, None)
    assert records.kind_ids[0, :3].tolist() == [0, 1, 2]
    assert records.object_ids[0, :3].tolist() == [2, 0, 0]
    assert records.hazard_ids[0, :3].tolist() == [4, 3, 4]
    torch.testing.assert_close(
        records.scalar_features[0, :3],
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, torch.log1p(torch.tensor(0.5)).item(), 0.8],
                [
                    0.0,
                    1.0,
                    1.0,
                    torch.log1p(torch.tensor(4.0)).item(),
                    torch.log1p(torch.tensor(2.0)).item(),
                    0.7,
                ],
                [0.0, 0.0, 0.0, 0.0, torch.log1p(torch.tensor(3.5)).item(), 0.6],
            ],
            dtype=torch.float32,
        ),
    )
    assert records.valid_mask[0, :4].tolist() == [True, True, True, False]
    assert records.padding_mask[0, :4].tolist() == [False, False, False, True]
    assert not bool(records.valid_mask[0, 3:].any())
    assert not bool(records.scalar_features[0, 3:].any())


def test_pack_memory_rejects_bad_batch_time_or_device(public_memories) -> None:
    from silent_cascade.memory.tensor_store import pack_memory

    with pytest.raises(NeuralError, match="same length"):
        pack_memory(public_memories, (0.0,), device="cpu")
    with pytest.raises(NeuralError, match="exact float"):
        pack_memory(public_memories, (0, 0.0), device="cpu")  # type: ignore[arg-type]
    with pytest.raises(NeuralError, match="before initial_time"):
        pack_memory(public_memories, (2.0, 9.0), device="cpu")
    with pytest.raises(NeuralError, match="CPU or MPS"):
        pack_memory(public_memories, (0.0, 0.0), device="meta")


def test_slot_permutation_moves_tensors_and_host_correspondence_together(
    public_memories,
) -> None:
    from silent_cascade.memory.tensor_store import pack_memory

    records = pack_memory(public_memories, (0.0, 9.0), device="cpu")
    permutations = torch.arange(64).repeat(2, 1)
    permutations[0, :3] = torch.tensor([2, 0, 1])
    permuted = records.permute_slots(permutations)

    assert permuted.record_ids[0][:3] == (13, 11, 12)
    assert permuted.observed_at[0][:3] == (4.0, 1.0, 2.5)
    assert permuted.subject_ids[0, :3].tolist() == [3, 1, 2]
    assert records.record_ids[0][:3] == (11, 12, 13)
    with pytest.raises(NeuralError, match="permutation"):
        records.permute_slots(torch.zeros((2, 64), dtype=torch.int64))


def test_packing_does_not_mutate_runtime_memory(public_memories) -> None:
    from silent_cascade.memory.tensor_store import pack_memory

    before = public_memories
    packed = pack_memory(public_memories, (0.0, 9.0), device="cpu")
    renamed = replace(
        public_memories[0],
        records=tuple(
            replace(slot, record=replace(slot.record, record_id=slot.record.record_id + 100))
            for slot in public_memories[0].records
        ),
    )

    assert public_memories == before
    assert packed.record_ids[0][0] == 11
    assert renamed.records[0].record.record_id == 111
