import json
from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.errors import ConfigurationError, SilentCascadeError
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.models.config import LossWeights, NeuralModelConfig
from silent_cascade.models.errors import NeuralError
from silent_cascade.train.config import (
    MAX_PHASE3_CONFIG_BYTES,
    MAX_PHASE3_CONFIG_DEPTH,
    Phase3Config,
    TrainingConfig,
    parse_phase3_canonical,
)
from silent_cascade.train.state import CheckpointDescriptor, TrainingError, TrainProgress

PHASE3_CONFIG_PATHS = (
    Path("configs/base.yaml"),
    Path("configs/data/primary.yaml"),
    Path("configs/model/event_flow.yaml"),
    Path("configs/model/neural_components.yaml"),
    Path("configs/train/one_hop.yaml"),
)


class FloatSubtype(float):
    pass


@pytest.mark.parametrize("profile", ["smoke", "one_hop"])
def test_profiles_and_real_optimizer_use_global_stabilized_epsilon(profile):
    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.train.trainer import make_optimizer

    resolved = resolve_config(
        Phase3Config, (*PHASE3_CONFIG_PATHS[:-1], Path(f"configs/train/{profile}.yaml"))
    )
    optimizer = make_optimizer(EventFlowModel(resolved.config.neural), resolved.config.training)
    assert resolved.config.training.epsilon == 1e-7
    assert all(group["eps"] == 1e-7 for group in optimizer.param_groups)


@pytest.mark.parametrize("profile", ["smoke", "one_hop"])
@pytest.mark.parametrize("epsilon", ["1.0e-8", "1.0e-6"])
def test_pre_stabilization_optimizer_configuration_is_rejected(profile, epsilon):
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(
            Phase3Config,
            (*PHASE3_CONFIG_PATHS[:-1], Path(f"configs/train/{profile}.yaml")),
            set_overrides=(f"training.epsilon={epsilon}",),
        )


def test_phase3_config_adds_neural_fields_without_changing_phase2(neural_config):
    assert neural_config.neural.entity_dim == 32
    assert neural_config.neural.record_dim == 96
    assert neural_config.neural.latent_dim == 456
    assert neural_config.neural.context_dim == 570
    assert neural_config.neural.preview_dim == 100
    assert neural_config.neural.retrieval_query_input_dim == 496
    assert neural_config.neural.retrieval_pair_input_dim == 352
    assert neural_config.neural.top_k == 4
    assert neural_config.limits.max_trainable_parameters == 5_000_000
    assert neural_config.training.max_steps == 75_000


def test_production_and_debug_profiles_are_closed_architectures(neural_config):
    assert neural_config.neural.model_dump() == {
        "architecture_profile": "production",
        "max_trainable_parameters": 5_000_000,
        "memory_slots": 64,
        "max_batch_size": 128,
        "entity_count": 64,
        "entity_dim": 32,
        "kind_dim": 8,
        "hazard_dim": 8,
        "provenance_dim": 4,
        "record_scalar_dim": 6,
        "record_input_dim": 90,
        "record_hidden_dim": 192,
        "record_dim": 96,
        "external_input_dim": 132,
        "external_hidden_dim": 256,
        "external_dim": 256,
        "mode_count": 6,
        "mode_dim": 8,
        "time_feature_dim": 2,
        "hypothesis_feature_dim": 8,
        "retrieval_query_input_dim": 496,
        "query_dim": 256,
        "top_k": 4,
        "controller_hidden_dim": 512,
        "jump_hidden_dim": 512,
        "head_hidden_dim": 256,
    }

    debug = neural_config.neural.model_dump()
    debug.update(
        architecture_profile="debug",
        record_hidden_dim=96,
        external_hidden_dim=64,
        query_dim=64,
        controller_hidden_dim=128,
        jump_hidden_dim=128,
        head_hidden_dim=64,
    )
    debug_config = NeuralModelConfig.model_validate(debug)
    assert debug_config.retrieval_pair_input_dim == 160
    assert debug_config.bilinear_record_output_dim == 64

    debug["record_hidden_dim"] = 97
    with pytest.raises(ValueError):
        NeuralModelConfig.model_validate(debug)


def test_phase3_config_rejects_unknown_keys_and_bool_for_int(tmp_path: Path):
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("neural:\n  arbitrary_architecture_plugin: dotted.path\n")
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(Phase3Config, [*PHASE3_CONFIG_PATHS, unknown])

    boolean = tmp_path / "boolean.yaml"
    boolean.write_text("training:\n  batch_size: true\n")
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(Phase3Config, [*PHASE3_CONFIG_PATHS, boolean])

    integer_float = tmp_path / "integer-float.yaml"
    integer_float.write_text("training:\n  gradient_clip_norm: 1\n")
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(Phase3Config, [*PHASE3_CONFIG_PATHS, integer_float])


@pytest.mark.parametrize(
    "field",
    [
        "max_trainable_parameters",
        "memory_slots",
        "max_batch_size",
        "entity_count",
        "entity_dim",
        "kind_dim",
        "hazard_dim",
        "provenance_dim",
        "record_scalar_dim",
        "record_input_dim",
        "record_hidden_dim",
        "record_dim",
        "external_input_dim",
        "external_hidden_dim",
        "external_dim",
        "mode_count",
        "mode_dim",
        "time_feature_dim",
        "hypothesis_feature_dim",
        "retrieval_query_input_dim",
        "query_dim",
        "top_k",
        "controller_hidden_dim",
        "jump_hidden_dim",
        "head_hidden_dim",
    ],
)
def test_phase3_yaml_rejects_float_for_every_integer_neural_literal(neural_config, field):
    value = getattr(neural_config.neural, field)
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(
            Phase3Config,
            PHASE3_CONFIG_PATHS,
            set_overrides=[f"neural.{field}={value}.0"],
        )


@pytest.mark.parametrize(
    ("field", "wrong_yaml_value"),
    [
        ("foreach", "0"),
        ("fused", "0"),
        ("early_stop_patience_validations", "15.0"),
        ("checkpoint_keep_best", "3.0"),
        ("checkpoint_keep_latest", "1.0"),
        ("model_seed", "11.0"),
        ("train_root_seed", "311.0"),
        ("train_public_id_seed", "331.0"),
        ("validation_root_seed", "313.0"),
        ("validation_public_id_seed", "337.0"),
    ],
)
def test_phase3_yaml_rejects_wrong_types_for_training_literals(field, wrong_yaml_value):
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(
            Phase3Config,
            PHASE3_CONFIG_PATHS,
            set_overrides=[f"training.{field}={wrong_yaml_value}"],
        )


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("learning_rate", FloatSubtype(3.0e-4)),
        ("weight_decay", FloatSubtype(1.0e-4)),
        ("epsilon", FloatSubtype(1.0e-7)),
        ("gradient_clip_norm", FloatSubtype(1.0)),
    ],
)
def test_training_float_literals_require_exact_float_inputs(neural_config, field, wrong_value):
    values = neural_config.training.model_dump()
    values[field] = wrong_value
    with pytest.raises(ValueError, match="exact float"):
        TrainingConfig.model_validate(values)


def test_adamw_beta_elements_require_exact_float_inputs(neural_config):
    values = neural_config.training.model_dump()
    values["betas"] = (FloatSubtype(0.9), FloatSubtype(0.999))
    with pytest.raises(ValueError, match=r"beta.*exact float"):
        TrainingConfig.model_validate(values)


def test_phase3_config_rejects_coordinated_runtime_limit_enlargement(tmp_path: Path):
    overlay = tmp_path / "enlarge.yaml"
    overlay.write_text(
        """limits:
  primary_memory_records: 65
  max_eventflow_events: 65
  batch_size: 129
data:
  primary_memory_capacity: 65
  max_eventflow_internal_events: 65
event_flow:
  max_internal_events: 65
neural:
  memory_slots: 65
  max_batch_size: 129
"""
    )
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(Phase3Config, [*PHASE3_CONFIG_PATHS, overlay])


def test_loss_weights_are_exactly_named_and_pilot_bounded():
    weights = LossWeights()
    assert weights.model_dump() == {
        "guard": 1.0,
        "retrieval": 1.0,
        "compose_type": 1.0,
        "focus": 0.5,
        "hazard": 1.0,
        "deadline": 0.25,
        "action": 1.0,
        "action_time": 0.25,
        "state": 0.05,
        "event_cost": 0.001,
    }
    LossWeights(guard=0.25, event_cost=0.004)
    with pytest.raises(ValueError):
        LossWeights(guard=0.249)
    with pytest.raises(ValueError):
        LossWeights(event_cost=0.00401)
    with pytest.raises(ValueError):
        LossWeights(action=1)


def test_smoke_overlay_changes_only_bounded_workload(neural_config):
    smoke = resolve_config(
        Phase3Config,
        [*PHASE3_CONFIG_PATHS[:-1], Path("configs/train/smoke.yaml")],
    )
    assert smoke.config.training.profile == "phase3_smoke"
    assert (
        smoke.config.training.batch_size,
        smoke.config.training.max_steps,
        smoke.config.training.validation_every_steps,
        smoke.config.training.fixed_validation_episodes,
    ) == (8, 4, 2, 16)
    assert smoke.config.neural == neural_config.neural
    assert smoke.config.training.loss_weights == neural_config.training.loss_weights


def test_phase3_canonical_parser_restores_integer_allocation_keys(resolved_neural_config):
    raw = resolved_neural_config.canonical_json.decode("utf-8")
    parsed = parse_phase3_canonical(raw)
    assert set(parsed.data.phase1_gate.iid_primary) == {2, 3, 4}
    assert canonical_json_bytes(parsed).decode("utf-8") == raw


def test_phase3_canonical_parser_rejects_noncanonical_and_unknown_allocation_maps(
    resolved_neural_config,
):
    raw = resolved_neural_config.canonical_json.decode("utf-8")
    with pytest.raises(ValueError, match="canonical"):
        parse_phase3_canonical(raw + "\n")

    values = json.loads(raw)
    values["data"]["phase1_gate"]["unexpected"] = {"2": 1}
    with pytest.raises(ValueError, match="exact known allocation"):
        parse_phase3_canonical(canonical_json_bytes(values).decode("utf-8"))


def test_phase3_canonical_parser_rejects_oversized_or_deep_json_before_validation():
    oversized = "{" + (" " * MAX_PHASE3_CONFIG_BYTES) + "}"
    with pytest.raises(ValueError, match="byte limit"):
        parse_phase3_canonical(oversized)

    deep = ("[" * (MAX_PHASE3_CONFIG_DEPTH + 1)) + "0" + ("]" * (MAX_PHASE3_CONFIG_DEPTH + 1))
    with pytest.raises(ValueError, match="nesting limit"):
        parse_phase3_canonical(deep)


def test_training_progress_and_descriptor_are_strict_bounded_contracts():
    descriptor = CheckpointDescriptor(
        relative_path="checkpoints/step-12.safetensors",
        file_sha256="1" * 64,
        model_state_sha256="2" * 64,
        config_sha256="3" * 64,
        source_commit="4" * 40,
        optimizer_step=12,
        stage="one_hop",
        validation_metric=0.75,
    )
    progress = TrainProgress(
        optimizer_step=12,
        next_batch_counter=12,
        stage="one_hop",
        train_root_seed=311,
        train_public_id_seed=331,
        best_metric=0.75,
        best_step=12,
        patience_counter=0,
        retained_checkpoints=(descriptor,),
        validation_manifest_sha256="5" * 64,
    )
    assert progress.retained_checkpoints == (descriptor,)
    with pytest.raises(ValueError):
        TrainProgress.model_validate({**progress.model_dump(), "optimizer_step": True})
    with pytest.raises(ValueError, match="relative"):
        CheckpointDescriptor.model_validate(
            {**descriptor.model_dump(), "relative_path": "../escape.safetensors"}
        )
    with pytest.raises(ValueError, match="exact float"):
        TrainProgress.model_validate({**progress.model_dump(), "best_metric": 1})
    with pytest.raises(ValueError, match="exact float"):
        CheckpointDescriptor.model_validate({**descriptor.model_dump(), "validation_metric": 1})


def test_phase3_errors_have_stable_family_and_codes():
    assert issubclass(NeuralError, SilentCascadeError)
    assert issubclass(TrainingError, SilentCascadeError)
    assert NeuralError("bad tensor").to_payload()["code"] == "neural_error"
    assert TrainingError("bad step").to_payload()["code"] == "training_error"
