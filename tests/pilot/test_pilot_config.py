from pathlib import Path

import pytest
from pydantic import ValidationError


def test_pilot_budget_is_global_and_component_profile_is_rejected():
    from silent_cascade.train.pilot_config import resolve_pilot_config

    resolved = resolve_pilot_config("phase4_pilot")
    assert resolved.config.pilot.max_steps == 75_000
    assert resolved.config.pilot.model_seed == 11
    values = resolved.config.model_dump()
    values["pilot"]["profile"] = "phase3_one_hop_content_v2"
    with pytest.raises(ValidationError):
        type(resolved.config).model_validate(values)


def test_production_pilot_resolves_the_fixed_optimizer_workload_and_roots():
    from silent_cascade.train.pilot_config import resolve_pilot_config

    resolved = resolve_pilot_config("phase4_pilot")
    pilot = resolved.config.pilot
    assert resolved.source_paths == (
        Path("configs/base.yaml"),
        Path("configs/data/primary.yaml"),
        Path("configs/model/event_flow.yaml"),
        Path("configs/model/neural_components.yaml"),
        Path("configs/train/pilot.yaml"),
    )
    assert resolved.config.experiment_version == "phase4-pilot-v1"
    assert resolved.config.neural.architecture_profile == "production"
    assert (
        pilot.profile,
        pilot.optimizer,
        pilot.learning_rate,
        pilot.weight_decay,
        pilot.betas,
        pilot.epsilon,
        pilot.foreach,
        pilot.fused,
        pilot.scheduler,
    ) == (
        "phase4_pilot",
        "adamw",
        3.0e-4,
        1.0e-4,
        (0.9, 0.999),
        1.0e-6,
        False,
        False,
        "none",
    )
    assert (
        pilot.batch_size,
        pilot.gradient_clip_norm,
        pilot.max_steps,
        pilot.validation_every_steps,
        pilot.fixed_validation_episodes,
        pilot.early_stop_patience_validations,
        pilot.checkpoint_keep_best,
        pilot.checkpoint_keep_latest,
    ) == (128, 1.0, 75_000, 1_000, 10_000, 15, 3, 1)
    assert (
        pilot.train_root_seed,
        pilot.train_public_id_seed,
        pilot.validation_root_seed,
        pilot.validation_public_id_seed,
        pilot.debug_root_seed,
        pilot.debug_public_id_seed,
    ) == (431, 433, 439, 443, 449, 457)
    assert pilot.data_recipe_version == "phase4-data-v1"
    assert pilot.one_hop_transform_version == "ofd-one-hop-v1"
    assert pilot.objective_version == "teacher_timed_plus_content_v2"
    assert pilot.content_auxiliary_weight == 1.0
    assert pilot.is_production is True


def test_smoke_pilot_is_debug_bounded_and_never_production():
    from silent_cascade.train.pilot_config import resolve_pilot_config

    resolved = resolve_pilot_config("phase4_smoke")
    pilot = resolved.config.pilot
    assert resolved.source_paths[-1] == Path("configs/train/pilot_smoke.yaml")
    assert resolved.config.neural.architecture_profile == "debug"
    assert (
        pilot.batch_size,
        pilot.max_steps,
        pilot.validation_every_steps,
        pilot.fixed_validation_episodes,
    ) == (8, 4, 2, 16)
    assert pilot.epsilon == 1.0e-6
    assert pilot.objective_version == "teacher_timed_plus_content_v2"
    assert pilot.content_auxiliary_weight == 1.0
    assert pilot.is_production is False


def test_phase4_config_rejects_unknown_keys_and_cross_limit_substitution():
    from silent_cascade.train.pilot_config import resolve_pilot_config

    resolved = resolve_pilot_config("phase4_pilot")
    values = resolved.config.model_dump()
    values["pilot"]["unexpected"] = "ignored"
    with pytest.raises(ValidationError):
        type(resolved.config).model_validate(values)

    values = resolved.config.model_dump()
    values["limits"]["batch_size"] = 64
    with pytest.raises(ValidationError, match="batch ceilings must match"):
        type(resolved.config).model_validate(values)


def test_unknown_phase4_profile_has_no_fallback_overlay():
    from silent_cascade.train.pilot_config import resolve_pilot_config

    with pytest.raises(ValueError, match="unsupported Phase4 profile"):
        resolve_pilot_config("phase3_one_hop_content_v2")
