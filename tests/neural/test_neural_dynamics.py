"""Parity and boundary tests for batched analytic EventFlow dynamics."""

from dataclasses import fields

import pytest
import torch

from silent_cascade.models.errors import NeuralError
from silent_cascade.models.types import TensorWorkspace


def _parameters(batch_size: int = 2, *, device: str = "cpu"):
    from silent_cascade.models.dynamics import BatchedSegmentParameters

    return BatchedSegmentParameters(
        flow_targets=torch.full((batch_size, 456), 0.75, device=device),
        flow_rates=torch.full((batch_size, 456), 0.7, device=device),
        raw_guard_targets=torch.tensor(
            [[0.5, 1.25, 1.75], [1.1, 0.9, 1.5]], dtype=torch.float32, device=device
        ),
        guard_targets=torch.tensor(
            [[0.5, 1.25, 1.75], [1.1, 0.9, 1.5]], dtype=torch.float32, device=device
        ),
        guard_rates=torch.tensor(
            [[0.1, 2.0, 500.0], [1e-5, 1.0, 20.0]], dtype=torch.float32, device=device
        ),
    )


def _scalar_parameters(parameters, row: int):
    from silent_cascade.eventflow.state import ContinuousChannels, SegmentParameters

    starts = (0, 256, 320, 328, 392)
    names = ("z_fast", "z_slow", "drives", "focus_key", "hypothesis_latent")
    ends = (256, 320, 328, 392, 456)
    return SegmentParameters(
        flow_targets=ContinuousChannels(
            **{
                name: parameters.flow_targets[row, start:end]
                for name, start, end in zip(names, starts, ends, strict=True)
            }
        ),
        flow_rates=ContinuousChannels(
            **{
                name: parameters.flow_rates[row, start:end]
                for name, start, end in zip(names, starts, ends, strict=True)
            }
        ),
        guard_targets=parameters.guard_targets[row],
        guard_rates=parameters.guard_rates[row],
    )


def test_batch_flow_matches_actual_anchored_scalar_runtime_row() -> None:
    from silent_cascade.eventflow.flow import start_segment, state_at
    from silent_cascade.eventflow.state import RuntimeCore
    from silent_cascade.models.dynamics import flow_batch

    latent = torch.linspace(-0.8, 0.6, 912, dtype=torch.float32).reshape(2, 456)
    # A real scalar segment begins immediately after the causal reset.
    accumulators = torch.zeros((2, 3), dtype=torch.float32)
    workspace = TensorWorkspace(latent, accumulators)
    parameters = _parameters()
    actual = flow_batch(workspace, parameters, torch.tensor([0.25, 2.0]))
    scalar = start_segment(
        RuntimeCore(workspace.row(1)),
        _scalar_parameters(parameters, 1),
        time=11.0,
        parent_event_id=7,
        prediction_snapshot_sha256="a" * 64,
    )
    expected = state_at(scalar, 13.0)

    expected_latent = torch.cat(
        [
            getattr(expected, item.name)
            for item in fields(expected)
            if item.name != "guard_accumulators"
        ]
    )
    torch.testing.assert_close(actual.latent[1], expected_latent)
    torch.testing.assert_close(actual.accumulators[1], expected.guard_accumulators)


@pytest.mark.parametrize("dt", [0.0, 1e-12, 1e6])
def test_batch_flow_handles_zero_tiny_and_huge_elapsed_time(dt: float) -> None:
    from silent_cascade.models.dynamics import flow_batch

    state = TensorWorkspace(
        torch.full((2, 456), -0.25), torch.tensor([[0.0, 0.2, 0.4], [0.1, 0.3, 0.5]])
    )
    actual = flow_batch(state, _parameters(), torch.full((2,), dt))
    assert torch.isfinite(actual.latent).all()
    assert torch.isfinite(actual.accumulators).all()
    if dt == 0.0:
        assert torch.equal(actual.latent, state.latent)
        assert torch.equal(actual.accumulators, state.accumulators)
    elif dt == 1e6:
        torch.testing.assert_close(actual.latent, torch.full_like(actual.latent, 0.75))


def test_reanchored_split_flow_has_float32_semigroup_equivalence() -> None:
    from silent_cascade.models.dynamics import flow_batch

    state = TensorWorkspace(torch.full((2, 456), -0.4), torch.zeros(2, 3))
    parameters = _parameters()
    direct = flow_batch(state, parameters, torch.full((2,), 1.1))
    split = flow_batch(
        flow_batch(state, parameters, torch.full((2,), 0.3)),
        parameters,
        torch.full((2,), 0.8),
    )
    torch.testing.assert_close(split.latent, direct.latent, rtol=5e-7, atol=2e-7)
    torch.testing.assert_close(split.accumulators, direct.accumulators, rtol=5e-7, atol=2e-7)


def test_batch_flow_preserves_controller_gradients() -> None:
    from silent_cascade.models.dynamics import BatchedSegmentParameters, flow_batch

    targets = torch.full((2, 456), 0.5, requires_grad=True)
    rates = torch.full((2, 456), 0.7, requires_grad=True)
    guard_targets = torch.full((2, 3), 1.5, requires_grad=True)
    guard_rates = torch.full((2, 3), 2.0, requires_grad=True)
    parameters = BatchedSegmentParameters(targets, rates, guard_targets, guard_targets, guard_rates)
    result = flow_batch(TensorWorkspace.zeros(2, "cpu"), parameters, torch.tensor([0.25, 1.0]))
    (result.latent.sum() + result.accumulators.sum()).backward()
    for tensor in (targets, rates, guard_targets, guard_rates):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "dt",
    [
        torch.tensor([0.0, -0.1]),
        torch.tensor([0.0, float("nan")]),
        torch.zeros(2, dtype=torch.float64),
    ],
)
def test_batch_flow_rejects_invalid_elapsed_tensors(dt: torch.Tensor) -> None:
    from silent_cascade.models.dynamics import flow_batch

    with pytest.raises(NeuralError):
        flow_batch(TensorWorkspace.zeros(2, "cpu"), _parameters(), dt)


def test_batch_crossings_match_completed_scalar_kernel() -> None:
    from silent_cascade.eventflow.guards import crossing_offsets_tensor
    from silent_cascade.models.dynamics import crossings_batch

    accumulators = torch.zeros((2, 3), dtype=torch.float32)
    targets = torch.tensor([[1.2, 0.9, 1.5], [0.5, 1.0, 1.1]])
    rates = torch.tensor([[2.0, 1.0, 0.1], [1.0, 1.0, 500.0]])
    expected = torch.stack(
        [crossing_offsets_tensor(*row) for row in zip(accumulators, targets, rates, strict=True)]
    )
    torch.testing.assert_close(crossings_batch(accumulators, targets, rates), expected)


def test_batch_crossings_only_differentiate_active_near_threshold_entries() -> None:
    from silent_cascade.models.dynamics import crossings_batch

    accumulators = torch.tensor([[0.999999, 1.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[1.0001, 1.5, 0.9]], requires_grad=True)
    rates = torch.tensor([[17.0, 3.0, 4.0]], requires_grad=True)
    offsets = crossings_batch(accumulators, targets, rates)
    assert torch.isfinite(offsets[0, 0])
    assert torch.isinf(offsets[0, 1:]).all()
    offsets[0, 0].backward()
    for tensor in (accumulators, targets, rates):
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad[0, 0] != 0
        assert tensor.grad[0, 1:].eq(0).all()


def test_flow_uses_trusted_functional_workspace_boundary(monkeypatch) -> None:
    from silent_cascade.models.dynamics import flow_batch

    state = TensorWorkspace.zeros(2, "cpu")
    parameters = _parameters()

    def reject_public_revalidation(self) -> None:
        raise AssertionError("hot functional updates must not invoke public constructor validation")

    monkeypatch.setattr(TensorWorkspace, "__post_init__", reject_public_revalidation)
    result = flow_batch(state, parameters, torch.tensor([0.1, 0.2]))
    assert result.latent.shape == (2, 456)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_batched_dynamics_are_native_float32_mps() -> None:
    from silent_cascade.models.dynamics import crossings_batch, flow_batch

    state = TensorWorkspace.zeros(2, "mps")
    parameters = _parameters(device="mps")
    flowed = flow_batch(state, parameters, torch.tensor([0.1, 0.2], device="mps"))
    offsets = crossings_batch(flowed.accumulators, parameters.guard_targets, parameters.guard_rates)
    assert flowed.latent.device.type == offsets.device.type == "mps"
    assert flowed.latent.dtype is offsets.dtype is torch.float32
