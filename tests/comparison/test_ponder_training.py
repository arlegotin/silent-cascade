"""Masked teacher projections and finite shared recurrent supervision."""

import pytest
import torch

from silent_cascade.eval.comparison_data import build_manifest, diagnostic_allocations, iter_bundles
from silent_cascade.eval.comparison_protocol import (
    admit_ponder_budget,
    nearest_ponder_cap,
    ponder_checkpoint_rank,
)
from silent_cascade.eval.comparison_types import BudgetLedger, ComparisonConfig, CostProfile
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
    load_main_checkpoint,
    main_training_stage,
    ponder_loss,
    project_ponder_batch,
    require_clean_main_source,
    save_debug_checkpoint,
    save_main_checkpoint,
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


def test_budget_admission_counts_full_validation_and_final_caps() -> None:
    profile = CostProfile(
        update_seconds_by_stage={
            name: 0.5 for name in ("one_hop", "two_hop", "primary", "robustness")
        },
        validation_episode_seconds=0.01,
        final_episode_seconds=0.2,
        replay_episode_seconds=0.2,
        checkpoint_seconds=2.0,
        report_seconds=10.0,
        retained_bytes_per_update_boundary=1_000_000,
        retained_bytes_per_validation_episode=100,
        retained_bytes_per_final_episode=100,
    )
    admitted = admit_ponder_budget(profile, BudgetLedger())
    assert admitted.chosen_updates == 12_000
    estimate = admitted.estimates[12_000]
    assert estimate.validation_episodes == 120_000
    assert estimate.final_episodes == 5_120
    assert estimate.replay_episodes >= 16 * (4 + 10)
    assert estimate.projected_seconds >= 2 * (6_000 + 1_200 + 1_024 + 10)
    too_costly = profile.model_copy(update={"validation_episode_seconds": 100.0})
    decision = admit_ponder_budget(too_costly, BudgetLedger())
    assert decision.chosen_updates is None
    assert decision.status == "extension_required"


def test_selection_uses_only_primary_validation_and_fixed_cap24() -> None:
    one = ponder_checkpoint_rank(
        update=1000, cap=24, primary_successes=9000, negative_false_actions=10
    )
    two = ponder_checkpoint_rank(
        update=2000, cap=24, primary_successes=9000, negative_false_actions=9
    )
    assert two > one
    assert (
        ponder_checkpoint_rank(
            update=3000, cap=24, primary_successes=9000, negative_false_actions=9
        )
        < two
    )
    with pytest.raises(ValueError):
        ponder_checkpoint_rank(update=1000, cap=8, primary_successes=9000, negative_false_actions=0)


def test_nearest_compute_uses_measured_iid_only_and_ten_percent_overlap() -> None:
    assert nearest_ponder_cap(100.0, {4: 40.0, 8: 95.0, 12: 105.0, 16: 200.0, 24: 300.0}) == 8
    assert nearest_ponder_cap(100.0, {4: 40.0, 8: 80.0, 12: 120.0, 16: 200.0, 24: 300.0}) is None


def test_main_checkpoint_preserves_next_batch_optimizer_and_consumed_budget(tmp_path) -> None:
    torch.manual_seed(11)
    model = ActivationPonderModel(PonderModelConfig(width=800))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    budget = BudgetLedger(updates=1, attempts=1, elapsed_scientific_seconds=3.0)

    def update(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        noise = torch.rand(())
        loss = model.heads["halt"].weight.square().mean() + noise * model.null_key.square().mean()
        loss.backward()
        optimizer.step()

    update(model, optimizer)
    checkpoint = tmp_path / "main-0001.pt"
    save_main_checkpoint(
        checkpoint,
        model,
        optimizer,
        step=1,
        budget=budget,
        config_sha256="a" * 64,
        source_revision="b" * 40,
    )
    update(model, optimizer)
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    restored = ActivationPonderModel(PonderModelConfig(width=800))
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    step, loaded_budget = load_main_checkpoint(
        checkpoint, restored, restored_optimizer, config_sha256="a" * 64, source_revision="b" * 40
    )
    assert step == 1 and loaded_budget == budget
    update(restored, restored_optimizer)
    assert all(torch.equal(expected[name], value) for name, value in restored.state_dict().items())
    with pytest.raises(ValueError):
        load_main_checkpoint(
            checkpoint,
            restored,
            restored_optimizer,
            config_sha256="a" * 64,
            source_revision="c" * 40,
        )
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
    with pytest.raises(ValueError):
        load_main_checkpoint(
            checkpoint,
            restored,
            restored_optimizer,
            config_sha256="a" * 64,
            source_revision="b" * 40,
        )


def test_main_stage_schedule_and_dirty_source_rejection(monkeypatch) -> None:
    assert [main_training_stage(n) for n in (1, 9000, 9001, 10000, 10001, 11000, 11001, 12000)] == [
        "one_hop",
        "one_hop",
        "two_hop",
        "two_hop",
        "primary",
        "primary",
        "robustness",
        "robustness",
    ]
    with pytest.raises(ValueError):
        main_training_stage(12001)
    from silent_cascade.train import activation_ponder

    monkeypatch.setattr(
        activation_ponder.subprocess, "check_output", lambda *a, **k: "b" * 40 + "\n"
    )
    monkeypatch.setattr(activation_ponder.subprocess, "call", lambda *a, **k: 1)
    with pytest.raises(ValueError, match="uncommitted"):
        require_clean_main_source()
