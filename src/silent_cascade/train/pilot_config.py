"""Closed Phase 4 pilot configuration profiles."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from silent_cascade.config import ResolvedConfig, resolve_config
from silent_cascade.eventflow.config import Phase2Config
from silent_cascade.models.config import LossWeights, NeuralModelConfig
from silent_cascade.validation import StrictModel


class PilotTrainingConfig(StrictModel):
    """Exact optimizer, workload, and data identities for the Phase 4 pilot."""

    profile: Literal["phase4_pilot", "phase4_smoke"]
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: Literal[3.0e-4] = 3.0e-4
    weight_decay: Literal[1.0e-4] = 1.0e-4
    betas: tuple[Literal[0.9], Literal[0.999]] = (0.9, 0.999)
    epsilon: Literal[1.0e-6] = 1.0e-6
    foreach: Literal[False] = False
    fused: Literal[False] = False
    scheduler: Literal["none"] = "none"
    batch_size: int = Field(ge=1, le=128)
    gradient_clip_norm: Literal[1.0] = 1.0
    max_steps: int = Field(ge=1, le=75_000)
    validation_every_steps: int = Field(ge=1, le=1_000)
    fixed_validation_episodes: int = Field(ge=1, le=10_000)
    early_stop_patience_validations: Literal[15] = 15
    checkpoint_keep_best: Literal[3] = 3
    checkpoint_keep_latest: Literal[1] = 1
    model_seed: Literal[11] = 11
    train_root_seed: Literal[431] = 431
    train_public_id_seed: Literal[433] = 433
    validation_root_seed: Literal[439] = 439
    validation_public_id_seed: Literal[443] = 443
    debug_root_seed: Literal[449] = 449
    debug_public_id_seed: Literal[457] = 457
    data_recipe_version: Literal["phase4-data-v1"] = "phase4-data-v1"
    one_hop_transform_version: Literal["ofd-one-hop-v1"] = "ofd-one-hop-v1"
    objective_version: Literal["teacher_timed_plus_content_v2"] = "teacher_timed_plus_content_v2"
    content_auxiliary_weight: Literal[1.0] = 1.0
    loss_weights: LossWeights = Field(default_factory=LossWeights)

    @field_validator("betas", mode="before")
    @classmethod
    def restore_yaml_tuple(cls, value: object) -> object:
        value = tuple(value) if isinstance(value, list) else value
        if isinstance(value, tuple) and any(type(item) is not float for item in value):
            raise ValueError("each AdamW beta must be an exact float")
        return value

    @field_validator("foreach", "fused", mode="before")
    @classmethod
    def require_exact_boolean_type(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("AdamW execution flags must be exact booleans")
        return value

    @field_validator(
        "batch_size",
        "max_steps",
        "validation_every_steps",
        "fixed_validation_episodes",
        "early_stop_patience_validations",
        "checkpoint_keep_best",
        "checkpoint_keep_latest",
        "model_seed",
        "train_root_seed",
        "train_public_id_seed",
        "validation_root_seed",
        "validation_public_id_seed",
        "debug_root_seed",
        "debug_public_id_seed",
        mode="before",
    )
    @classmethod
    def require_exact_integer_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("pilot integer must be an exact int")
        return value

    @field_validator(
        "learning_rate",
        "weight_decay",
        "epsilon",
        "gradient_clip_norm",
        "content_auxiliary_weight",
        mode="before",
    )
    @classmethod
    def require_exact_float_type(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("pilot scalar must be an exact float")
        return value

    @model_validator(mode="after")
    def validate_profile_workload(self) -> Self:
        workload = (
            self.batch_size,
            self.max_steps,
            self.validation_every_steps,
            self.fixed_validation_episodes,
        )
        expected = (128, 75_000, 1_000, 10_000) if self.is_production else (8, 4, 2, 16)
        if workload != expected:
            raise ValueError(f"{self.profile} workload must equal its approved bounded profile")
        return self

    @property
    def is_production(self) -> bool:
        return self.profile == "phase4_pilot"


class Phase4Config(Phase2Config):
    neural: NeuralModelConfig
    pilot: PilotTrainingConfig

    @model_validator(mode="after")
    def validate_phase4_limits(self) -> Self:
        if self.neural.max_trainable_parameters != self.limits.max_trainable_parameters:
            raise ValueError("neural and project parameter ceilings must match")
        if self.neural.memory_slots != self.limits.primary_memory_records:
            raise ValueError("neural and project memory ceilings must match")
        if self.neural.max_batch_size != self.limits.batch_size:
            raise ValueError("neural and project batch ceilings must match")
        if self.pilot.batch_size > self.neural.max_batch_size:
            raise ValueError("pilot batch exceeds the neural batch ceiling")
        expected_architecture = "production" if self.pilot.is_production else "debug"
        if self.neural.architecture_profile != expected_architecture:
            raise ValueError("pilot profile and neural architecture must match")
        return self


def resolve_pilot_config(profile: str) -> ResolvedConfig[Phase4Config]:
    """Resolve one of the two closed Phase 4 overlays."""

    profile_paths = {
        "phase4_pilot": "configs/train/pilot.yaml",
        "phase4_smoke": "configs/train/pilot_smoke.yaml",
    }
    if profile not in profile_paths:
        raise ValueError("unsupported Phase4 profile")
    paths = (
        Path("configs/base.yaml"),
        Path("configs/data/primary.yaml"),
        Path("configs/model/event_flow.yaml"),
        Path("configs/model/neural_components.yaml"),
        Path(profile_paths[profile]),
    )
    return resolve_config(Phase4Config, paths)
