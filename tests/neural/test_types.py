from dataclasses import fields

import pytest
import torch

from silent_cascade.models.errors import NeuralError
from silent_cascade.models.types import ModelContext, TensorWorkspace


def valid_context(batch_size: int = 2, device: str = "cpu") -> ModelContext:
    return ModelContext(
        workspace=TensorWorkspace.zeros(batch_size, device),
        memory_embeddings=torch.zeros(batch_size, 64, 96, dtype=torch.float32, device=device),
        eligibility=torch.zeros(batch_size, 64, dtype=torch.bool, device=device),
        support_mask=torch.zeros(batch_size, 64, dtype=torch.bool, device=device),
        active_slot_indices=torch.full((batch_size,), -1, dtype=torch.int64, device=device),
        modes=torch.zeros(batch_size, dtype=torch.int64, device=device),
        time_features=torch.zeros(batch_size, 2, dtype=torch.float32, device=device),
        hypothesis_features=torch.zeros(batch_size, 8, dtype=torch.float32, device=device),
    )


def test_workspace_zeros_has_exact_batch_contract():
    workspace = TensorWorkspace.zeros(2, "cpu")
    assert workspace.latent.shape == (2, 456)
    assert workspace.accumulators.shape == (2, 3)
    assert workspace.latent.dtype is torch.float32
    assert workspace.accumulators.dtype is torch.float32
    with pytest.raises(NeuralError, match="batch_size"):
        TensorWorkspace.zeros(129, "cpu")
    with pytest.raises(NeuralError, match="exact integer"):
        TensorWorkspace.zeros(True, "cpu")


def test_workspace_row_is_independent_and_differentiable():
    batch = TensorWorkspace.zeros(2, "cpu")
    batch.latent.requires_grad_()
    row = batch.row(1)
    row.z_fast.sum().backward()
    assert batch.latent.grad[1, :256].eq(1).all()
    assert batch.latent.grad[0].eq(0).all()


def test_workspace_rejects_float64_nonfinite_and_bad_rows():
    with pytest.raises(NeuralError, match="float32"):
        TensorWorkspace(torch.zeros(2, 456, dtype=torch.float64), torch.zeros(2, 3))
    latent = torch.zeros(2, 456)
    latent[0, 0] = torch.nan
    with pytest.raises(NeuralError, match="finite"):
        TensorWorkspace(latent, torch.zeros(2, 3))
    with pytest.raises(NeuralError, match="row index"):
        TensorWorkspace.zeros(2, "cpu").row(2)


def test_model_context_exposes_only_public_section4_fields():
    context = valid_context()
    assert tuple(field.name for field in fields(context)) == (
        "workspace",
        "memory_embeddings",
        "eligibility",
        "support_mask",
        "active_slot_indices",
        "modes",
        "time_features",
        "hypothesis_features",
    )
    assert context.batch_size == 2
    assert context.device == torch.device("cpu")


@pytest.mark.parametrize(
    "mutation",
    [
        {"memory_embeddings": torch.zeros(2, 64, 96, dtype=torch.float64)},
        {"eligibility": torch.zeros(2, 64, dtype=torch.float32)},
        {"support_mask": torch.zeros(2, 63, dtype=torch.bool)},
        {"active_slot_indices": torch.zeros(2, dtype=torch.int32)},
        {"modes": torch.full((2,), 6, dtype=torch.int64)},
        {"time_features": torch.full((2, 2), float("inf"))},
        {"hypothesis_features": torch.zeros(2, 7)},
    ],
)
def test_model_context_rejects_malformed_public_tensors(mutation):
    values = {field.name: getattr(valid_context(), field.name) for field in fields(ModelContext)}
    values.update(mutation)
    with pytest.raises(NeuralError):
        ModelContext(**values)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_model_context_rejects_mixed_device_tensors():
    values = {field.name: getattr(valid_context(), field.name) for field in fields(ModelContext)}
    values["memory_embeddings"] = torch.zeros(2, 64, 96, device="mps")
    with pytest.raises(NeuralError, match="one device"):
        ModelContext(**values)
