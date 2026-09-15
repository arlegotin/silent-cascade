"""Behavioral tests for non-invasive neural compute accounting."""

from dataclasses import FrozenInstanceError

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class _OperationTrace(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.operations: list[object] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.append(func)
        return func(*args, **(kwargs or {}))


def test_linear_mac_accounting() -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter

    layer = torch.nn.Linear(3, 5)
    with NeuralComputeMeter(layer) as meter:
        layer(torch.ones(2, 4, 3))
    snapshot = meter.snapshot()
    assert snapshot.estimated_macs == 2 * 4 * 3 * 5
    assert snapshot.forward_macs == 2 * 4 * 3 * 5
    assert snapshot.backward_macs == 0
    assert snapshot.operation_estimates["linear_bias_adds"] == 2 * 4 * 5


def test_training_backward_is_reported_separately() -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter

    layer = torch.nn.Linear(3, 5)
    values = torch.ones(2, 3, requires_grad=True)
    with NeuralComputeMeter(layer) as meter:
        layer(values).sum().backward()
    snapshot = meter.snapshot()
    assert snapshot.forward_macs == 2 * 3 * 5
    assert snapshot.backward_macs == 2 * (2 * 3 * 5)
    assert snapshot.estimated_macs == snapshot.forward_macs


def test_preview_plus_actual_recall_counts_executed_scorer_work_twice(
    model_context, neural_config
) -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter
    from silent_cascade.models.event_flow import EventFlowModel

    model = EventFlowModel(neural_config.neural)
    active = model_context
    with NeuralComputeMeter(model) as once:
        model.recall_scores(active)
    with NeuralComputeMeter(model) as twice:
        model.preview_and_control(active)
        model.recall_scores(active)
    first = once.snapshot()
    second = twice.snapshot()
    assert first.records_scored == 2 * 64
    assert second.records_scored == 2 * first.records_scored
    assert second.eligible_records == 2 * first.eligible_records
    assert second.module_calls["RetrievalScorer"] == 2
    assert second.operation_estimates["bilinear_dot_macs"] == 2 * 2 * 64 * 256


def test_repeated_instrumentation_does_not_duplicate_or_leave_hooks() -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter

    layer = torch.nn.Linear(3, 5)
    for _ in range(2):
        with NeuralComputeMeter(layer) as meter:
            layer(torch.ones(2, 3))
        assert meter.snapshot().estimated_macs == 2 * 3 * 5
        assert not layer._forward_hooks
        assert not layer._forward_pre_hooks
        assert not layer._backward_hooks


def test_retained_output_loses_meter_tensor_hook_on_context_exit() -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter

    layer = torch.nn.Linear(3, 5)
    values = torch.ones(2, 3, requires_grad=True)
    with NeuralComputeMeter(layer) as first:
        retained = layer(values)
        assert retained._backward_hooks
    assert not retained._backward_hooks
    retained.sum().backward()
    assert first.snapshot().backward_macs == 0

    layer.zero_grad(set_to_none=True)
    values.grad = None
    with NeuralComputeMeter(layer) as second:
        layer(values).sum().backward()
    assert second.snapshot().backward_macs == 2 * (2 * 3 * 5)


def test_forward_hooks_add_no_tensor_operations_and_freeze_public_counts(
    model_context, neural_config
) -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter
    from silent_cascade.models.event_flow import EventFlowModel

    model = EventFlowModel(neural_config.neural)
    with torch.no_grad(), _OperationTrace() as plain:
        model.preview_and_control(model_context)
    with NeuralComputeMeter(model) as meter, torch.no_grad(), _OperationTrace() as measured:
        model.preview_and_control(model_context)
    assert measured.operations == plain.operations
    expected_eligible = int(model_context.eligibility.sum())
    snapshot = meter.snapshot()
    assert snapshot.eligible_records == expected_eligible
    assert snapshot.opportunities == 2
    model_context.eligibility.zero_()
    model_context.modes.zero_()
    assert meter.snapshot().eligible_records == expected_eligible
    assert meter.snapshot().opportunities == 2


def test_meter_preserves_outputs_gradients_and_rng() -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter

    torch.manual_seed(101)
    plain = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.SiLU())
    measured = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.SiLU())
    measured.load_state_dict(plain.state_dict())
    plain_input = torch.randn(2, 3, requires_grad=True)
    measured_input = plain_input.detach().clone().requires_grad_()
    rng_before = torch.random.get_rng_state().clone()
    plain_output = plain(plain_input)
    plain_output.sum().backward()
    with NeuralComputeMeter(measured):
        measured_output = measured(measured_input)
        measured_output.sum().backward()
    torch.testing.assert_close(measured_output, plain_output)
    torch.testing.assert_close(measured_input.grad, plain_input.grad)
    for left, right in zip(plain.parameters(), measured.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_snapshot_is_immutable_and_reports_named_nonlinear_work() -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter

    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4), torch.nn.SiLU())
    with NeuralComputeMeter(module) as meter:
        module(torch.ones(2, 3))
    snapshot = meter.snapshot()
    assert snapshot.operation_estimates["layer_norm_ops"] > 0
    assert snapshot.operation_estimates["silu_ops"] > 0
    assert snapshot.mps_peak_allocation_bytes is None
    with pytest.raises(TypeError):
        snapshot.module_calls["Linear"] = 99
    with pytest.raises(FrozenInstanceError):
        snapshot.records_scored = 99


def test_batched_flow_reports_executed_rows(neural_config) -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter
    from silent_cascade.models.dynamics import BatchedSegmentParameters, flow_batch
    from silent_cascade.models.types import TensorWorkspace

    workspace = TensorWorkspace.zeros(2, "cpu")
    parameters = BatchedSegmentParameters(
        flow_targets=torch.zeros(2, 456),
        flow_rates=torch.ones(2, 456),
        raw_guard_targets=torch.full((2, 3), 0.5),
        guard_targets=torch.full((2, 3), 0.5),
        guard_rates=torch.ones(2, 3),
    )
    with NeuralComputeMeter(torch.nn.Identity()) as meter:
        flow_batch(workspace, parameters, torch.ones(2))
    snapshot = meter.snapshot()
    assert snapshot.flow_evaluations == 2
    assert snapshot.row_transitions == 2
    assert snapshot.operation_estimates["exp_log_ops"] == 2 * (456 + 3)


def test_external_encoding_without_injection_is_not_a_jump(neural_config) -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter
    from silent_cascade.memory.encoder import RecordEncoder
    from silent_cascade.models.jump import ExternalEncoder
    from silent_cascade.models.types import ExternalFeatures

    config = neural_config.neural
    encoder = ExternalEncoder(config, RecordEncoder(config))
    event = ExternalFeatures(
        subject_ids=torch.tensor([1]),
        object_ids=torch.tensor([2]),
        record_kind_ids=torch.tensor([0]),
        hazard_ids=torch.tensor([4]),
        provenance_ids=torch.tensor([0]),
        record_scalar_features=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.2, 0.8]]),
        activation_entity_ids=torch.tensor([0]),
        event_kinds=torch.tensor([0]),
        time_features=torch.tensor([[0.2, 0.0]]),
    )
    with NeuralComputeMeter(encoder) as meter:
        encoder(event)
    snapshot = meter.snapshot()
    assert snapshot.jump_applications == 0
    expected_linear_macs = (
        90 * config.record_hidden_dim
        + config.record_hidden_dim * 96
        + 132 * config.external_hidden_dim
        + config.external_hidden_dim * 256
        + 256 * 640
    )
    assert snapshot.forward_macs == expected_linear_macs
    assert snapshot.operation_estimates["embedding_output_bytes"] > 0


def test_external_injection_and_activation_focus_count_functional_nonlinears(
    neural_config,
) -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter
    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.models.types import ExternalFeatures

    model = EventFlowModel(neural_config.neural)
    context = model.initial_context(2, device="cpu")
    event = ExternalFeatures(
        subject_ids=torch.tensor([1, 2]),
        object_ids=torch.tensor([2, 3]),
        record_kind_ids=torch.tensor([0, 0]),
        hazard_ids=torch.tensor([4, 4]),
        provenance_ids=torch.tensor([0, 0]),
        record_scalar_features=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.2, 0.8]] * 2),
        activation_entity_ids=torch.tensor([0, 4]),
        event_kinds=torch.tensor([0, 1]),
        time_features=torch.tensor([[0.2, 0.0], [0.3, 0.0]]),
    )
    with NeuralComputeMeter(model) as meter:
        model.observe(context, event)
    estimates = meter.snapshot().operation_estimates
    assert estimates["sigmoid_ops"] == 2 * 320
    assert estimates["tanh_ops"] == 2 * 320 + 64


def test_retrieval_preview_counts_executed_softmax_and_logsumexp_slots(
    model_context, neural_config
) -> None:
    from dataclasses import replace

    from silent_cascade.eval.compute import NeuralComputeMeter
    from silent_cascade.models.event_flow import EventFlowModel

    eligibility = model_context.eligibility.clone()
    eligibility[1] = False
    context = replace(model_context, eligibility=eligibility)
    model = EventFlowModel(neural_config.neural)
    with NeuralComputeMeter(model) as meter:
        model.preview_and_control(context)
    estimates = meter.snapshot().operation_estimates
    assert estimates["softmax_ops"] == 2 * 64
    assert estimates["logsumexp_ops"] == 64


def test_shared_jump_functional_nonlinears_are_not_double_counted(
    model_context, neural_config
) -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter
    from silent_cascade.models.event_flow import EventFlowModel

    model = EventFlowModel(neural_config.neural)
    with NeuralComputeMeter(model) as meter:
        model.jump(model_context, torch.tensor([0, 2]))
    estimates = meter.snapshot().operation_estimates
    assert estimates["sigmoid_ops"] == 2 * 456
    assert estimates["tanh_ops"] == 2 * 456


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_meter_synchronizes_real_mps_without_fabricating_peak() -> None:
    from silent_cascade.eval.compute import NeuralComputeMeter

    layer = torch.nn.Linear(3, 5).to("mps")
    with NeuralComputeMeter(layer) as meter:
        output = layer(torch.ones(2, 3, device="mps"))
    assert output.device.type == "mps"
    assert meter.snapshot().elapsed_seconds >= 0.0
    assert meter.snapshot().mps_peak_allocation_bytes is None
