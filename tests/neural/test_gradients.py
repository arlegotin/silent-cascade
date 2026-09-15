import pytest
import torch

from silent_cascade.models.config import LossWeights, NeuralModelConfig
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.train.batches import pack_training_examples
from silent_cascade.train.curriculum_data import CurriculumKey, make_curriculum_example


def _debug_model_config(neural_config) -> NeuralModelConfig:
    values = neural_config.neural.model_dump()
    values.update(
        architecture_profile="debug",
        record_hidden_dim=96,
        external_hidden_dim=64,
        query_dim=64,
        controller_hidden_dim=128,
        jump_hidden_dim=128,
        head_hidden_dim=64,
    )
    return NeuralModelConfig.model_validate(values)


def _examples_for_variants_and_depths(neural_config):
    wanted = {
        (variant, depth)
        for variant in ("positive", "safe_negative", "disconnected_negative")
        for depth in (1, 2, 4)
    }
    found = {}
    for stage in ("one_hop", "primary"):
        for index in range(80):
            example = make_curriculum_example(
                neural_config,
                CurriculumKey("ofd-one-hop-v1", "debug", 311, 331, index, stage),
            )
            key = (example.variant.value, len(example.solution.link_record_ids))
            if key in wanted and key not in found:
                found[key] = example
    assert set(found) == wanted
    return tuple(found[key] for key in sorted(wanted))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_real_unroll_loss_is_finite_and_backpropagates_every_component(neural_config, device):
    from silent_cascade.models.losses import event_flow_loss
    from silent_cascade.train.unroll import teacher_forced_unroll

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS is not exposed")
    torch.manual_seed(11)
    model = EventFlowModel(_debug_model_config(neural_config)).to(device)
    examples = _examples_for_variants_and_depths(neural_config)
    inputs = teacher_forced_unroll(model, pack_training_examples(examples).to(device)).loss_inputs()
    result = event_flow_loss(inputs, LossWeights())
    assert result.total.dtype is torch.float32
    assert result.total.device.type == device
    assert torch.isfinite(result.total)
    result.total.backward()
    paths = {
        "record encoder": model.record_encoder.network[0].weight,
        "external encoder": model.external_encoder.network[0].weight,
        "retrieval scorer": model.scorer.query_network[0].weight,
        "controller": model.controller.network[-1].weight,
        "shared jump": model.shared_jump.network[0].weight,
        "focus projection": model.focus_projection.weight,
        "compose role": model.compose_heads.role.weight,
        "compose status": model.compose_heads.status.weight,
        "compose confidence": model.compose_heads.confidence.weight,
        "compose support": model.compose_heads.append_support.weight,
        "compose continue": model.compose_heads.continue_search.weight,
        "compose focus": model.compose_heads.next_focus.weight,
        "compose hazard": model.compose_heads.hazard.weight,
        "compose delay": model.compose_heads.delay.weight,
        "compose deadline": model.compose_heads.deadline.weight,
        "action class": model.action_heads.classifier.weight,
        "action time": model.action_heads.lead.weight,
    }
    for name, parameter in paths.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_two_full_loss_optimizer_steps_build_fresh_graphs(neural_config):
    from silent_cascade.models.losses import event_flow_loss
    from silent_cascade.train.unroll import teacher_forced_unroll

    torch.manual_seed(11)
    model = EventFlowModel(_debug_model_config(neural_config))
    examples = _examples_for_variants_and_depths(neural_config)
    batch = pack_training_examples(examples)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, eps=1e-7)
    totals = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = event_flow_loss(teacher_forced_unroll(model, batch).loss_inputs(), LossWeights())
        result.total.backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        optimizer.step()
        totals.append(result.total.detach().item())
    assert all(math_value == math_value for math_value in totals)
    assert totals[0] != totals[1]
