"""Behavioral tests for bounded learned flow and guard control."""

import pytest
import torch
from torch import nn


def test_controller_outputs_exact_batched_contract_and_gradients(
    model_context, neural_config
) -> None:
    from silent_cascade.memory.retrieval import RetrievalScorer
    from silent_cascade.models.controller import FlowGuardController

    controller = FlowGuardController(neural_config.neural)
    preview = RetrievalScorer(neural_config.neural).preview(model_context)
    output = controller(model_context, preview)
    assert output.flow_targets.shape == output.flow_rates.shape == (2, 456)
    assert output.raw_guard_targets.shape == output.guard_targets.shape == (2, 3)
    assert output.guard_rates.shape == (2, 3)
    assert output.flow_targets.abs().le(1.0).all()
    assert output.flow_rates.ge(1e-5).all() and output.flow_rates.le(20.0).all()
    assert output.raw_guard_targets.gt(0.0).all() and output.raw_guard_targets.lt(2.0).all()
    assert output.guard_rates.ge(1e-5).all() and output.guard_rates.le(500.0).all()
    (
        output.flow_targets.sum() + output.flow_rates.sum() + output.raw_guard_targets.sum()
    ).backward()
    gradients = [parameter.grad for parameter in controller.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_controller_masks_execution_guards_by_authoritative_mode_order(neural_config) -> None:
    from silent_cascade.eventflow.guards import allowed_mode_mask
    from silent_cascade.memory.retrieval import RetrievalPreview
    from silent_cascade.models.controller import FlowGuardController
    from silent_cascade.models.types import ModelContext, TensorWorkspace
    from silent_cascade.schemas import Mode

    modes = tuple(Mode)
    context = ModelContext(
        workspace=TensorWorkspace.zeros(6, "cpu"),
        memory_embeddings=torch.zeros(6, 64, 96),
        eligibility=torch.zeros(6, 64, dtype=torch.bool),
        support_mask=torch.zeros(6, 64, dtype=torch.bool),
        active_slot_indices=torch.full((6,), -1, dtype=torch.int64),
        modes=torch.arange(6, dtype=torch.int64),
        time_features=torch.zeros(6, 2),
        hypothesis_features=torch.zeros(6, 8),
    )
    preview = RetrievalPreview(torch.zeros(6, 100), torch.zeros(6, dtype=torch.bool))
    output = FlowGuardController(neural_config.neural)(context, preview)
    for row, mode in enumerate(modes):
        mask = torch.tensor(allowed_mode_mask(mode))
        assert output.guard_targets[row, mask].equal(output.raw_guard_targets[row, mask])
        assert output.guard_targets[row, ~mask].lt(1.0).all()


def test_saturated_controller_logits_remain_finite_and_strictly_bounded(
    model_context, neural_config
) -> None:
    from silent_cascade.memory.retrieval import RetrievalPreview
    from silent_cascade.models.controller import FlowGuardController

    controller = FlowGuardController(neural_config.neural)
    with torch.no_grad():
        for parameter in controller.parameters():
            parameter.zero_()
        output_layer = controller.network[-1]
        output_layer.bias.copy_(
            torch.cat(
                (
                    torch.full((456,), 1e6),
                    torch.full((456,), -1e6),
                    torch.tensor([1e6, -1e6, 1e6, 1e6, -1e6, 1e6]),
                )
            )
        )
    preview = RetrievalPreview(torch.zeros(2, 100), torch.zeros(2, dtype=torch.bool))
    output = controller(model_context, preview)
    for value in (
        output.flow_targets,
        output.flow_rates,
        output.raw_guard_targets,
        output.guard_targets,
        output.guard_rates,
    ):
        assert torch.isfinite(value).all()
    assert output.raw_guard_targets.gt(0.0).all()
    assert output.raw_guard_targets.lt(2.0).all()


def test_controller_injected_mode_table_has_one_parameter_owner(
    model_context, neural_config
) -> None:
    from silent_cascade.memory.retrieval import RetrievalPreview
    from silent_cascade.models.controller import FlowGuardController

    table = nn.Embedding(6, 8)
    controller = FlowGuardController(neural_config.neural, mode_embedding=table)
    preview = RetrievalPreview(torch.zeros(2, 100), torch.zeros(2, dtype=torch.bool))
    assert controller.mode_embedding is table
    assert all("mode_embedding" not in name for name, _ in controller.named_parameters())
    assert controller(model_context, preview).flow_targets.shape == (2, 456)


def test_controller_rejects_preview_from_wrong_batch(model_context, neural_config) -> None:
    from silent_cascade.memory.retrieval import RetrievalPreview
    from silent_cascade.models.controller import FlowGuardController
    from silent_cascade.models.errors import NeuralError

    preview = RetrievalPreview(torch.zeros(1, 100), torch.zeros(1, dtype=torch.bool))
    with pytest.raises(NeuralError):
        FlowGuardController(neural_config.neural)(model_context, preview)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_controller_outputs_are_native_float32_mps(neural_config) -> None:
    from silent_cascade.memory.retrieval import RetrievalPreview
    from silent_cascade.models.controller import FlowGuardController
    from silent_cascade.models.types import ModelContext, TensorWorkspace

    context = ModelContext(
        workspace=TensorWorkspace.zeros(2, "mps"),
        memory_embeddings=torch.zeros(2, 64, 96, device="mps"),
        eligibility=torch.zeros(2, 64, dtype=torch.bool, device="mps"),
        support_mask=torch.zeros(2, 64, dtype=torch.bool, device="mps"),
        active_slot_indices=torch.full((2,), -1, dtype=torch.int64, device="mps"),
        modes=torch.tensor([1, 4], dtype=torch.int64, device="mps"),
        time_features=torch.zeros(2, 2, device="mps"),
        hypothesis_features=torch.zeros(2, 8, device="mps"),
    )
    preview = RetrievalPreview(
        torch.zeros(2, 100, device="mps"), torch.zeros(2, dtype=torch.bool, device="mps")
    )
    output = FlowGuardController(neural_config.neural).to("mps")(context, preview)
    for value in (
        output.flow_targets,
        output.flow_rates,
        output.raw_guard_targets,
        output.guard_targets,
        output.guard_rates,
    ):
        assert value.device.type == "mps"
        assert value.dtype is torch.float32
