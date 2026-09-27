"""Measured work is the basis of fair compute comparisons."""

import pytest
import torch
from pydantic import ValidationError

from silent_cascade.eval.compute import (
    NeuralComputeMeter,
    RuntimeCompute,
    _record_ponder_retrieval,
)


def test_compute_includes_prefix_and_all_executed_calls() -> None:
    """Each invocation contributes its actual matrix work, including initialization."""
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    with NeuralComputeMeter(model) as meter:
        model(torch.ones(2, 2))
        model(torch.zeros(2, 2))
    snapshot = meter.snapshot()
    assert snapshot.forward_macs == 36
    assert snapshot.module_calls["Linear"] == 4
    assert snapshot.foundation_model_calls == 0
    with pytest.raises(ValidationError):
        RuntimeCompute(opportunities=1)


def test_ponder_records_scored_counts_executed_masked_slots_and_null() -> None:
    model = torch.nn.Linear(2, 2)
    valid = torch.tensor([[True, False], [False, False]])
    with NeuralComputeMeter(model) as meter:
        _record_ponder_retrieval(valid, query_dim=2)
    snapshot = meter.snapshot()
    assert snapshot.records_scored == 6
    assert snapshot.eligible_records == 1
    assert snapshot.forward_macs == 12
