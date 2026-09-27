"""Masked teacher projections and finite shared recurrent supervision."""

import torch

from silent_cascade.eval.comparison_data import build_manifest, diagnostic_allocations, iter_bundles
from silent_cascade.eval.comparison_types import ComparisonConfig
from silent_cascade.eval.compute import RuntimeCompute
from silent_cascade.models.activation_ponder import (
    ActivationPonderModel,
    PonderDecision,
    PonderModelConfig,
)
from silent_cascade.schemas import Action
from silent_cascade.train.activation_ponder import (
    build_fixed_competence_set,
    evaluate_competence,
    load_debug_checkpoint,
    ponder_loss,
    project_ponder_batch,
    save_debug_checkpoint,
)
from silent_cascade.train.batches import pack_training_examples
from silent_cascade.train.curriculum_data import make_curriculum_example
from silent_cascade.train.pilot_config import resolve_pilot_config
from silent_cascade.train.pilot_data import _key


def _batch():
    config = resolve_pilot_config("phase4_pilot").config
    examples = tuple(
        make_curriculum_example(config, _key(stage, "debug", index))
        for stage, index in (("one_hop", 0), ("two_hop", 1), ("primary", 2), ("primary", 3))
    )
    return pack_training_examples(examples)


def test_teacher_projection_has_no_target_in_prediction_inputs() -> None:
    torch.manual_seed(11)
    model = ActivationPonderModel(PonderModelConfig(width=800))
    projected = project_ponder_batch(_batch(), caps=torch.tensor([24, 24, 24, 24]))
    assert projected.public is not None
    assert projected.teacher_slots.shape == projected.step_mask.shape
    assert projected.public.records.valid_mask.shape == (4, 64)
    changed = projected.teacher_slots.clone()
    changed[:, 0] = 64
    with torch.no_grad():
        original = model.forward_teacher(
            projected.public, projected.teacher_slots, projected.step_mask
        )
        altered = model.forward_teacher(projected.public, changed, projected.step_mask)
    assert torch.allclose(original.retrieval_logits[:, 0], altered.retrieval_logits[:, 0])


def test_budget_masks_future_teacher_steps() -> None:
    batch = _batch()
    short = project_ponder_batch(batch, caps=torch.tensor([1, 1, 1, 1]))
    full = project_ponder_batch(batch, caps=torch.tensor([24, 24, 24, 24]))
    assert torch.equal(short.step_mask.sum(dim=1), torch.ones(4, dtype=torch.int64))
    assert not bool(short.step_mask[:, 1:].any())
    assert bool(full.step_mask[:, 1:].any())
    assert not bool((short.validity["action_class"] & ~short.step_mask).any())
    assert not bool((short.halt_target[:, 0] & (full.step_mask.sum(dim=1) > 1)).any())


def test_event_cost_accounts_for_survival_through_sampled_cap() -> None:
    torch.manual_seed(11)
    model = ActivationPonderModel(PonderModelConfig(width=800))
    batch = _batch()
    short = project_ponder_batch(batch, caps=torch.tensor([4, 4, 4, 4]))
    long = project_ponder_batch(batch, caps=torch.tensor([24, 24, 24, 24]))
    short_loss = ponder_loss(model, short)
    long_loss = ponder_loss(model, long)
    assert long.rollout_mask.shape[1] == 24
    assert long_loss.terms["event_cost"] > short_loss.terms["event_cost"]


def test_all_loss_terms_and_shared_steps_receive_finite_gradients() -> None:
    torch.manual_seed(11)
    model = ActivationPonderModel(PonderModelConfig(width=800))
    projected = project_ponder_batch(_batch(), caps=torch.tensor([4, 8, 12, 24]))
    result = ponder_loss(model, projected)
    assert torch.isfinite(result.total)
    assert result.terms["retrieval"].isfinite()
    assert result.terms["halt"].isfinite()
    assert result.terms["event_cost"].isfinite()
    assert result.terms["action"].isfinite()
    result.total.backward()
    assert model.gru.weight_hh.grad is not None
    assert torch.isfinite(model.gru.weight_hh.grad).all()
    assert model.gru.weight_hh.grad.abs().sum() > 0
    assert model.record_encoder.network[0].weight.grad is not None
    assert torch.isfinite(model.record_encoder.network[0].weight.grad).all()


def test_fixed_competence_set_uses_balanced_independent_debug_roots() -> None:
    config = resolve_pilot_config("phase4_pilot").config
    examples, identity = build_fixed_competence_set(config)
    assert len(examples) == 64
    assert [
        sum(e.key.stage == stage for e in examples) for stage in ("one_hop", "two_hop", "primary")
    ] == [16, 16, 32]
    assert all((e.key.root_seed, e.key.public_id_seed) == (7963, 7993) for e in examples)
    assert len({e.public.init.episode_public_id for e in examples}) == 64
    assert [
        sum(e.solution.terminal_kind == kind for e in examples)
        for kind in ("hazard", "safe", "disconnected")
    ] == [32, 16, 16]
    assert len(identity) == 64


def test_debug_checkpoint_preserves_next_update_and_rng(tmp_path) -> None:
    torch.manual_seed(11)
    original = ActivationPonderModel(PonderModelConfig(width=800))
    optimizer = torch.optim.AdamW(original.parameters(), lr=3e-4)

    def update(model, optim):
        optim.zero_grad(set_to_none=True)
        noise = torch.rand(())
        loss = model.heads["halt"].weight.square().mean() + noise * model.null_key.square().mean()
        loss.backward()
        optim.step()

    update(original, optimizer)
    path = tmp_path / "debug-0001.pt"
    save_debug_checkpoint(path, original, optimizer, step=1, dataset_sha256="a" * 64)
    update(original, optimizer)
    expected = {name: tensor.detach().clone() for name, tensor in original.state_dict().items()}
    resumed = ActivationPonderModel(PonderModelConfig(width=800))
    restored_optimizer = torch.optim.AdamW(resumed.parameters(), lr=3e-4)
    assert (
        load_debug_checkpoint(path, resumed, restored_optimizer, expected_dataset_sha256="a" * 64)
        == 1
    )
    update(resumed, restored_optimizer)
    assert all(torch.equal(expected[name], tensor) for name, tensor in resumed.state_dict().items())


def test_competence_evaluator_keeps_private_terminal_out_of_policy(monkeypatch) -> None:
    from silent_cascade.train import activation_ponder

    config = ComparisonConfig()
    allocation = diagnostic_allocations()["iid"]
    first = allocation.blocks[0].model_copy(update={"episode_count": 4})
    manifest = build_manifest(config, "iid", allocation.model_copy(update={"blocks": (first,)}))
    bundles = tuple(iter_bundles(manifest, config))
    seen = []

    def tied_action(model, public, *, cap):
        assert not hasattr(public, "truth")
        seen.append(public.init.episode_public_id)
        truth = next(bundle.truth for bundle in bundles if bundle.public == public)
        return PonderDecision(
            action=Action(0, truth.private_terminal.timestamp, public.events[-1].event_id),
            steps=(),
            stop_reason="halt",
            compute=RuntimeCompute(),
        )

    monkeypatch.setattr(activation_ponder, "ponder_public", tied_action)
    model = ActivationPonderModel(PonderModelConfig(width=800))
    rows = evaluate_competence(model, bundles, cap=24)
    assert len(rows) == len(seen) == 4
    assert all(row["actions"] == [] for row in rows)
