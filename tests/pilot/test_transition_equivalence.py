"""Compare complete trajectories and gradients to the pre-extraction source."""

import copy
import subprocess
import sys
import types
from collections.abc import Mapping
from dataclasses import fields, is_dataclass

import pytest
import torch

from silent_cascade.train.batches import pack_training_examples
from silent_cascade.train.curriculum_data import CurriculumKey, make_curriculum_example

BASE = "b3debbc20600bba8dc87f57666f1edc9a904ac29"


def compare(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif is_dataclass(left):
        for field in fields(left):
            # Measurements of wall time and allocator peaks are not model outputs.
            if field.name in ("elapsed_seconds", "mps_peak_allocation_bytes"):
                continue
            compare(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            compare(a, b)
    elif isinstance(left, Mapping):
        assert left.keys() == right.keys()
        for key in left:
            compare(left[key], right[key])
    else:
        assert left == right


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize(
    "module_name,function",
    [
        ("unroll", "teacher_forced_unroll"),
        ("content_unroll", "teacher_forced_content_unroll"),
    ],
)
def test_pre_extraction_outputs_and_gradients(model, neural_config, device, module_name, function):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    source = subprocess.check_output(
        [
            "git",
            "show",
            f"{BASE}:src/silent_cascade/train/{module_name}.py",
        ],
        text=True,
    )
    legacy = types.ModuleType(f"_legacy_{module_name}")
    sys.modules[legacy.__name__] = legacy
    exec(compile(source, f"{BASE}/{module_name}.py", "exec"), legacy.__dict__)
    from importlib import import_module

    current = import_module(f"silent_cascade.train.{module_name}")
    examples = tuple(
        make_curriculum_example(
            neural_config, CurriculumKey("ofd-one-hop-v1", "debug", 311, 331, i, bucket)
        )
        for bucket in ("one_hop", "two_hop", "primary")
        for i in range(4)
    )
    batch = pack_training_examples(examples).to(device)
    model = model.to(device)
    old_model = copy.deepcopy(model)
    expected = getattr(legacy, function)(old_model, batch)
    actual = getattr(current, function)(model, batch)
    compare(actual, expected)
    compare(actual.loss_inputs(), expected.loss_inputs())
    from silent_cascade.models.content_loss import content_loss
    from silent_cascade.models.losses import event_flow_loss

    def loss(result):
        inputs = result.loss_inputs()
        return (
            event_flow_loss(inputs, neural_config.training.loss_weights)
            if module_name == "unroll"
            else content_loss(inputs)
        )

    compare(loss(actual), loss(expected))
    for result in (expected, actual):
        objective = sum(
            step.post_context.workspace.latent.square().sum()
            + step.prediction_context.workspace.latent.square().sum()
            for step in result.steps
        )
        (objective + loss(result).total).backward()
    for (name, parameter), (_, old) in zip(
        model.named_parameters(), old_model.named_parameters(), strict=True
    ):
        if old.grad is None:
            assert parameter.grad is None, name
        else:
            # MPS atomic embedding reductions vary even for unchanged old-vs-old;
            # see the Phase 4 Task 2 numerical deviation. CPU remains bitwise exact.
            rtol, atol = (1e-5, 1e-6) if device == "mps" else (0, 0)
            torch.testing.assert_close(parameter.grad, old.grad, rtol=rtol, atol=atol, msg=name)


def test_public_transition_boundary_exists():
    from silent_cascade.models import transitions

    assert all(
        hasattr(transitions, name)
        for name in (
            "RecallTransition",
            "ComposeTransition",
            "apply_recall_context",
            "apply_compose_context",
            "apply_action_context",
        )
    )


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_public_wrappers_preserve_row_decisions_and_reset_guards(model, device):
    from silent_cascade.models.transitions import (
        ComposeTransition,
        RecallTransition,
        apply_action_context,
        apply_compose_context,
        apply_recall_context,
    )
    from silent_cascade.models.types import TensorWorkspace

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")
    model = model.to(device)
    context = model.initial_context(2, device=device)
    context = context._updated(
        workspace=TensorWorkspace(
            torch.full((2, 456), 0.2, device=device), torch.ones((2, 3), device=device)
        ),
        eligibility=torch.ones_like(context.eligibility),
    )
    recall = apply_recall_context(
        model,
        context,
        RecallTransition(torch.tensor([1, 3], device=device), torch.full_like(context.modes, 2)),
    )
    assert recall.active_slot_indices.tolist() == [1, 3]
    assert not recall.workspace.accumulators.any()
    support = context.support_mask.clone()
    support[:, 1] = True
    decision = ComposeTransition(
        support,
        torch.tensor([1, 3], device=device),
        torch.ones_like(context.hypothesis_features),
        torch.tensor([5, 7], device=device),
        torch.tensor([True, False], device=device),
    )
    composed = apply_compose_context(model, recall, decision)
    assert composed.active_slot_indices.tolist() == [-1, -1]
    assert torch.equal(composed.eligibility, context.eligibility)
    assert torch.equal(composed.support_mask, support)
    assert not composed.workspace.accumulators.any()
    focus = torch.tanh(
        model.focus_projection(model.record_encoder.encode_entities(decision.focus_ids[:1]))
    )
    torch.testing.assert_close(composed.workspace.latent[:1, 328:392], focus, rtol=0, atol=0)
    action = apply_action_context(model, composed)
    assert action.modes.tolist() == [4, 4]
    assert not action.workspace.accumulators.any()
    for row in range(2):
        single = apply_action_context(model, composed._gather(torch.tensor([row], device=device)))
        torch.testing.assert_close(
            action.workspace.latent[row], single.workspace.latent[0], atol=3e-6, rtol=3e-5
        )
