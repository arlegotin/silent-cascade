"""The configured optimizer must train both actual prediction contexts exactly once."""

from copy import copy, deepcopy
from pathlib import Path

import pytest
import torch

from silent_cascade.config import resolve_config
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.models.losses import event_flow_loss
from silent_cascade.train.batches import next_training_batch
from silent_cascade.train.config import Phase3Config, TrainingConfig, parse_phase3_canonical
from silent_cascade.train.trainer import make_optimizer, train_one_step
from silent_cascade.train.unroll import teacher_forced_unroll

from .test_training_cli import CONFIGS


def corrected_config(profile="smoke"):
    return resolve_config(
        Phase3Config, (*CONFIGS[:-1], Path(f"configs/train/{profile}_content_v2.yaml"))
    )


@pytest.mark.parametrize("profile", ["smoke", "one_hop"])
def test_recipe_profiles_preserve_historical_bytes_and_bind_optimizer(profile):
    old = resolve_config(Phase3Config, (*CONFIGS[:-1], Path(f"configs/train/{profile}.yaml")))
    assert (
        canonical_json_bytes(parse_phase3_canonical(old.canonical_json.decode()))
        == old.canonical_json
    )
    assert old.config.training.objective_version == "teacher_timed_v1"
    assert old.config.training.content_auxiliary_weight == 0.0
    assert "objective_version" not in old.config.training.model_dump()
    corrected = corrected_config(profile)
    assert corrected.config.training.objective_version == "teacher_timed_plus_content_v2"
    assert corrected.config.training.content_auxiliary_weight == 1.0
    assert corrected.config.training.is_production is (profile == "one_hop")
    assert corrected.config.training.curriculum_stage == (
        "one_hop" if profile == "one_hop" else "smoke"
    )
    optimizer = make_optimizer(EventFlowModel(corrected.config.neural), corrected.config.training)
    assert optimizer.param_groups[0]["eps"] == 1e-6
    for config, epsilon in ((old, 1e-6), (corrected, 1e-7), (corrected, 1e-8)):
        with pytest.raises(ValueError):
            TrainingConfig.model_validate(
                {**config.config.training.model_dump(), "epsilon": epsilon}
            )
    with pytest.raises(ValueError):
        TrainingConfig.model_validate(
            {**corrected.config.training.model_dump(), "content_auxiliary_weight": 2.0}
        )


def test_combined_objective_preserves_timed_loss_and_one_adamw_update():
    from silent_cascade.train.objective import training_objective

    config = corrected_config().config
    torch.manual_seed(11)
    model = EventFlowModel(config.neural)
    batch = next_training_batch(config, stage="one_hop", batch_counter=0)
    objective = training_objective(model, batch, config.training)
    timed = event_flow_loss(
        teacher_forced_unroll(model, batch).loss_inputs(), config.training.loss_weights
    )
    torch.testing.assert_close(objective.timed_loss.total, timed.total, rtol=0, atol=0)
    for key in timed.terms:
        torch.testing.assert_close(
            objective.timed_loss.terms[key], timed.terms[key], rtol=0, atol=0
        )
    assert objective.content_loss is not None
    torch.testing.assert_close(
        objective.total, timed.total + objective.content_loss.total, rtol=0, atol=0
    )
    assert all(value.item() > 0 for value in objective.content_loss.terms.values())
    expected = deepcopy(model)
    expected_optimizer = make_optimizer(expected, config.training)
    manual = training_objective(expected, batch, config.training)
    manual.total.backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), 1.0, error_if_nonfinite=True)
    expected_optimizer.step()
    optimizer = make_optimizer(model, config.training)
    result = train_one_step(model, optimizer, batch, config)
    for left, right in zip(model.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert {state["step"].item() for state in optimizer.state.values()} == {1}
    assert batch.next_batch_counter == 1
    assert result.objective_version == "teacher_timed_plus_content_v2"
    assert result.auxiliary_coefficient == 1.0
    assert (
        result.loss == result.timed_loss + result.content_loss
        or abs(result.loss - result.timed_loss - result.content_loss) < 1e-4
    )
    for name in ("terms", "subterms", "numerators", "denominators", "per_position"):
        values = getattr(result, name)
        assert any(key.startswith("timed/") for key in values)
        assert any(key.startswith("content/") for key in values)
    assert result.compute.foundation_model_calls == 0
    assert (
        result.compute.forward_macs
        == result.timed_compute.forward_macs + result.content_compute.forward_macs
    )
    assert result.compute.backward_macs > 0
    assert result.content_compute.backward_macs == result.timed_compute.backward_macs == 0


def test_legacy_objective_has_only_the_original_timed_graph():
    from silent_cascade.train.objective import training_objective

    config = resolve_config(Phase3Config, CONFIGS).config
    model = EventFlowModel(config.neural)
    batch = next_training_batch(config, stage="one_hop", batch_counter=0)
    objective = training_objective(model, batch, config.training)
    assert objective.content is objective.content_loss is None
    assert objective.total is objective.timed_loss.total


def test_cross_recipe_checkpoint_rejects_before_rng_mutation(tmp_path):
    from silent_cascade.train.checkpoints import load_training_checkpoint, save_training_checkpoint
    from silent_cascade.train.state import TrainingError, TrainProgress

    config = resolve_config(Phase3Config, CONFIGS)
    model = EventFlowModel(config.config.neural)
    optimizer = make_optimizer(model, config.config.training)
    batch = next_training_batch(config.config, stage="one_hop", batch_counter=0)
    train_one_step(model, optimizer, batch, config.config)
    progress = TrainProgress(
        optimizer_step=1,
        next_batch_counter=1,
        stage="smoke",
        train_root_seed=311,
        train_public_id_seed=331,
        patience_counter=0,
        retained_checkpoints=(),
        validation_manifest_sha256="0" * 64,
    )
    saved = save_training_checkpoint(
        tmp_path, model, optimizer, progress, config=config, source_commit="a" * 40
    )
    rng = torch.get_rng_state().clone()
    with pytest.raises(TrainingError):
        load_training_checkpoint(
            tmp_path / saved.relative_path,
            expected_config_sha256=corrected_config().sha256,
            expected_source_commit="a" * 40,
            device="cpu",
        )
    assert torch.equal(rng, torch.get_rng_state())


def test_configured_parity_executes_both_graphs_and_actual_optimizer():
    from silent_cascade.train.verification import _parity_step

    config = corrected_config().config
    torch.manual_seed(11)
    model = EventFlowModel(config.neural)
    batch = next_training_batch(config, stage="one_hop", batch_counter=0)
    forward, losses, gradients, weights, *_ = _parity_step(model, batch, "cpu", config.training)
    assert "/timed/final_context/workspace/latent" in forward
    assert "/content/final_context/workspace/latent" in forward
    assert "/timed/total" in losses and "/content/total" in losses and "/total" in losses
    expected = deepcopy(model)
    optimizer = make_optimizer(expected, config.training)
    train_one_step(expected, optimizer, batch, config)
    for name, parameter in expected.named_parameters():
        torch.testing.assert_close(weights["/" + name], parameter, rtol=0, atol=0)
    assert gradients and all(torch.isfinite(value).all() for value in gradients.values())


def test_content_state_corruption_stops_before_optimizer_update(monkeypatch):
    from dataclasses import replace

    from silent_cascade.train import objective
    from silent_cascade.train.state import TrainingError

    config = corrected_config().config
    model = EventFlowModel(config.neural)
    optimizer = make_optimizer(model, config.training)
    batch = next_training_batch(config, stage="one_hop", batch_counter=0)
    original = objective.teacher_forced_content_unroll

    def corrupt(*args):
        content = original(*args)
        context = copy(content.final_context)
        workspace = copy(context.workspace)
        object.__setattr__(workspace, "latent", workspace.latent * torch.nan)
        object.__setattr__(context, "workspace", workspace)
        return replace(content, final_context=context)

    monkeypatch.setattr(objective, "teacher_forced_content_unroll", corrupt)
    with pytest.raises(TrainingError, match="Nonfinite"):
        train_one_step(model, optimizer, batch, config)
    assert not optimizer.state


def test_corrected_optimizer_checkpoint_and_portable_stage_survive(tmp_path):
    from silent_cascade.train.checkpoints import (
        export_weights,
        load_training_checkpoint,
        save_training_checkpoint,
    )
    from silent_cascade.train.state import TrainProgress

    config = corrected_config()
    model = EventFlowModel(config.config.neural)
    optimizer = make_optimizer(model, config.config.training)
    batch = next_training_batch(config.config, stage="one_hop", batch_counter=0)
    train_one_step(model, optimizer, batch, config.config)
    progress = TrainProgress(
        optimizer_step=1,
        next_batch_counter=1,
        stage="smoke",
        train_root_seed=311,
        train_public_id_seed=331,
        patience_counter=0,
        retained_checkpoints=(),
        validation_manifest_sha256="0" * 64,
    )
    saved = save_training_checkpoint(
        tmp_path, model, optimizer, progress, config=config, source_commit="a" * 40
    )
    restored = load_training_checkpoint(
        tmp_path / saved.relative_path,
        expected_config_sha256=config.sha256,
        expected_source_commit="a" * 40,
        device="cpu",
    )
    assert {k: v for k, v in restored.optimizer.param_groups[0].items() if k != "params"} == {
        k: v for k, v in optimizer.param_groups[0].items() if k != "params"
    }
    assert restored.optimizer.param_groups[0]["eps"] == 1e-6
    weights = export_weights(tmp_path, model, config=config, source_commit="a" * 40)
    assert weights.stage == "smoke"
