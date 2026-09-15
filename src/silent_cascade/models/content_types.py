"""Neutral tensors for the auxiliary immediate-context content objective."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ContentLossInputs:
    """Graph-preserving [B,T] storage; boundary masks exclude ACT and padding."""

    boundary_mask: torch.Tensor
    recall_mask: torch.Tensor
    raw_recall_target: torch.Tensor
    retrieval_logits: torch.Tensor
    retrieval_eligible_mask: torch.Tensor
    retrieval_target: torch.Tensor
    role_logits: torch.Tensor
    status_logits: torch.Tensor
    confidence_logit: torch.Tensor
    append_support_logit: torch.Tensor
    continue_search_logit: torch.Tensor
    next_focus_logits: torch.Tensor
    hazard_logits: torch.Tensor
    log_delay: torch.Tensor
    role_target: torch.Tensor
    status_target: torch.Tensor
    confidence_target: torch.Tensor
    append_support_target: torch.Tensor
    continue_search_target: torch.Tensor
    focus_target: torch.Tensor
    hazard_target: torch.Tensor
    log_delay_target: torch.Tensor
    role_mask: torch.Tensor
    status_mask: torch.Tensor
    confidence_mask: torch.Tensor
    append_support_mask: torch.Tensor
    continue_search_mask: torch.Tensor
    focus_mask: torch.Tensor
    hazard_mask: torch.Tensor
    log_delay_mask: torch.Tensor
