"""Behavioral tests for prediction-only composition and action heads."""

from dataclasses import fields

import torch
from torch import nn


def test_compose_heads_emit_every_prediction_with_finite_gradients(
    model_context, neural_config
) -> None:
    from silent_cascade.models.heads import ComposeHeads

    heads = ComposeHeads(neural_config.neural)
    predictions = heads(model_context)
    assert predictions.role_logits.shape == (2, 5)
    assert predictions.next_focus_logits.shape == (2, 64)
    assert predictions.hazard_logits.shape == (2, 4)
    assert predictions.log_delay.shape == (2,)
    assert predictions.normalized_deadline.shape == (2,)
    assert predictions.status_logits.shape == (2, 3)
    assert predictions.confidence_logit.shape == (2,)
    assert predictions.append_support_logit.shape == (2,)
    assert predictions.continue_search_logit.shape == (2,)
    assert predictions.log_delay.ge(0.0).all()
    loss = sum(getattr(predictions, field.name).sum() for field in fields(predictions))
    loss.backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in heads.parameters()
    )


def test_action_heads_include_abstain_and_bounded_lead_fraction(
    model_context, neural_config
) -> None:
    from silent_cascade.models.heads import ActionHeads

    heads = ActionHeads(neural_config.neural)
    predictions = heads(model_context)
    assert predictions.class_logits.shape == (2, 5)
    assert predictions.lead_fraction.shape == (2,)
    assert predictions.lead_fraction.ge(0.0).all()
    assert predictions.lead_fraction.le(1.0).all()
    (predictions.class_logits.sum() + predictions.lead_fraction.sum()).backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in heads.parameters()
    )


def test_heads_use_one_externally_owned_mode_table(model_context, neural_config) -> None:
    from silent_cascade.models.heads import ActionHeads, ComposeHeads

    table = nn.Embedding(6, 8)
    compose = ComposeHeads(neural_config.neural, mode_embedding=table)
    action = ActionHeads(neural_config.neural, mode_embedding=table)
    assert compose.mode_embedding is action.mode_embedding is table
    assert all("mode_embedding" not in name for name, _ in compose.named_parameters())
    assert all("mode_embedding" not in name for name, _ in action.named_parameters())
    assert compose(model_context).role_logits.shape == (2, 5)
    assert action(model_context).class_logits.shape == (2, 5)
