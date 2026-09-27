"""Public-only, parameter-matched adaptive activation competitor."""

import torch

from silent_cascade.eval.comparison_data import build_manifest, diagnostic_allocations, iter_bundles
from silent_cascade.eval.comparison_types import ComparisonConfig
from silent_cascade.eval.compute import parameter_counts
from silent_cascade.eval.ponder_policy import ponder_public
from silent_cascade.models.activation_ponder import (
    ActivationPonderModel,
    PonderModelConfig,
    matched_ponder_config,
)
from silent_cascade.train.batches import pack_public_examples


def _public():
    config = ComparisonConfig()
    allocation = diagnostic_allocations()["iid"]
    first = allocation.blocks[0].model_copy(update={"episode_count": 4})
    manifest = build_manifest(config, "iid", allocation.model_copy(update={"blocks": (first,)}))
    return next(iter_bundles(manifest, config)).public


def test_ponder_width_is_parameter_matched_without_dummy_weights() -> None:
    torch.manual_seed(11)
    config = matched_ponder_config()
    assert config.record_config == ComparisonConfig().phase4_config.neural
    model = ActivationPonderModel(config)
    counted = parameter_counts(model)
    assert config.width == 800
    assert counted["total"] <= 5_000_000
    assert abs(counted["total"] - 2_781_042) / 2_781_042 <= 0.10
    assert counted["entity_table"] == 2048
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert not any(
        "dummy" in name or "padding_parameter" in name for name, _ in model.named_parameters()
    )


def test_ponder_reads_complete_memory_and_is_slot_permutation_invariant() -> None:
    torch.manual_seed(11)
    model = ActivationPonderModel(PonderModelConfig(width=800)).eval()
    public = pack_public_examples((_public(),))
    permuted = public.permute_slots(torch.randperm(64).unsqueeze(0))
    with torch.no_grad():
        state = model.encode_public(public)
        moved = model.encode_public(permuted)
        first = model.transition(state, state.hidden)
        changed = model.transition(moved, moved.hidden)
        second = model.transition(state, first.hidden)
    assert first.retrieval_logits.shape == (1, 65)
    assert first.selected_record_ids == changed.selected_record_ids
    assert torch.allclose(first.hidden, changed.hidden, atol=1e-5)
    assert int(state.valid_mask.sum()) > 1
    assert second.retrieval_logits[0, :64][state.valid_mask[0]].numel() == int(
        state.valid_mask.sum()
    )


def test_ponder_halts_or_caps_without_advancing_world_time() -> None:
    torch.manual_seed(11)
    model = ActivationPonderModel(PonderModelConfig(width=800)).eval()
    public = _public()
    with torch.no_grad():
        model.heads["halt"].weight.zero_()
        model.heads["halt"].bias.fill_(20.0)
    early = ponder_public(model, public, cap=24)
    assert early.stop_reason == "halt"
    assert len(early.steps) == 1
    assert early.compute.foundation_model_calls == 0
    assert early.compute.neural_jump_applications == 1
    assert early.compute.records_scored > 0
    assert early.compute.module_calls["SiLU"] >= 3
    with torch.no_grad():
        model.heads["halt"].bias.fill_(-20.0)
    for cap in (4, 8, 12, 16, 24):
        result = ponder_public(model, public, cap=cap)
        assert result.stop_reason == "cap_reached"
        assert len(result.steps) == cap
        assert all(step.cognitive_timestamp == public.events[-1].timestamp for step in result.steps)
        assert result.action is None or result.action.timestamp >= public.events[-1].timestamp
        assert result.compute.neural_jump_applications == cap
        assert result.compute.records_scored == cap * early.compute.records_scored
    assert len(ponder_public(model, public, cap=24).steps) >= 9


def test_null_read_is_a_real_recurrent_transition(monkeypatch) -> None:
    torch.manual_seed(11)
    model = ActivationPonderModel(PonderModelConfig(width=800)).eval()
    monkeypatch.setattr(
        model,
        "_stable_selection",
        lambda logits, record_ids: torch.full(
            (logits.shape[0],), 64, dtype=torch.int64, device=logits.device
        ),
    )
    with torch.no_grad():
        model.heads["halt"].weight.zero_()
        model.heads["halt"].bias.fill_(20.0)
    result = ponder_public(model, _public(), cap=4)
    assert result.steps[0].selected_record_id is None
    assert result.compute.neural_jump_applications == 1
    assert result.compute.module_calls["GRUCell"] == 1
