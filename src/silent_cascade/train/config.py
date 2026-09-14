"""Strict Phase 3 training configuration and canonical JSON codec."""

import json
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from silent_cascade.eventflow.config import Phase2Config
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.models.config import LossWeights, NeuralModelConfig
from silent_cascade.validation import StrictModel

MAX_PHASE3_CONFIG_BYTES = 1024 * 1024
MAX_PHASE3_CONFIG_DEPTH = 64

_ALLOCATION_NAMES = {
    "iid_primary",
    "ood_depth",
    "ood_short_delay",
    "ood_long_delay",
    "distractor_flood",
    "clock_parent_episodes",
    "clock_10x_episodes",
}


class TrainingConfig(StrictModel):
    """Bounded AdamW workload for the supported Phase 3 execution profiles."""

    profile: Literal["phase3_one_hop", "phase3_smoke"]
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: Literal[3.0e-4] = 3.0e-4
    weight_decay: Literal[1.0e-4] = 1.0e-4
    betas: tuple[Literal[0.9], Literal[0.999]] = (0.9, 0.999)
    epsilon: Literal[1.0e-8] = 1.0e-8
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
    train_root_seed: Literal[311] = 311
    train_public_id_seed: Literal[331] = 331
    validation_root_seed: Literal[313] = 313
    validation_public_id_seed: Literal[337] = 337
    curriculum_version: Literal["ofd-one-hop-v1"] = "ofd-one-hop-v1"
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
        "early_stop_patience_validations",
        "checkpoint_keep_best",
        "checkpoint_keep_latest",
        "model_seed",
        "train_root_seed",
        "train_public_id_seed",
        "validation_root_seed",
        "validation_public_id_seed",
        mode="before",
    )
    @classmethod
    def require_exact_integer_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("training constant must be an exact integer")
        return value

    @field_validator(
        "learning_rate",
        "weight_decay",
        "epsilon",
        "gradient_clip_norm",
        mode="before",
    )
    @classmethod
    def require_exact_float_type(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("optimizer scalar must be an exact float")
        return value

    @model_validator(mode="after")
    def validate_profile_workload(self) -> Self:
        workload = (
            self.batch_size,
            self.max_steps,
            self.validation_every_steps,
            self.fixed_validation_episodes,
        )
        expected = {
            "phase3_one_hop": (128, 75_000, 1_000, 10_000),
            "phase3_smoke": (8, 4, 2, 16),
        }[self.profile]
        if workload != expected:
            raise ValueError(f"{self.profile} workload must equal its approved bounded profile")
        return self


class Phase3Config(Phase2Config):
    neural: NeuralModelConfig
    training: TrainingConfig

    @model_validator(mode="after")
    def validate_neural_limits(self) -> Self:
        if self.neural.max_trainable_parameters != self.limits.max_trainable_parameters:
            raise ValueError("neural and project parameter ceilings must match")
        if self.neural.memory_slots != self.limits.primary_memory_records:
            raise ValueError("neural and project memory ceilings must match")
        if self.neural.max_batch_size != self.limits.batch_size:
            raise ValueError("neural and project batch ceilings must match")
        if self.training.batch_size > self.neural.max_batch_size:
            raise ValueError("training batch exceeds the neural batch ceiling")
        return self


def _require_bounded_json(raw: str) -> dict[str, object]:
    if type(raw) is not str:
        raise ValueError("full Phase 3 configuration must be a JSON string")
    try:
        encoded = raw.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("full Phase 3 configuration must be valid UTF-8") from error
    if len(encoded) > MAX_PHASE3_CONFIG_BYTES:
        raise ValueError("full Phase 3 configuration exceeds the byte limit")

    depth = 0
    in_string = False
    escaped = False
    for character in raw:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_PHASE3_CONFIG_DEPTH:
                raise ValueError("full Phase 3 configuration exceeds the nesting limit")
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise ValueError("full Phase 3 configuration is malformed JSON")
    try:
        values = json.loads(raw)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("full Phase 3 configuration is malformed JSON") from error
    if not isinstance(values, dict):
        raise ValueError("full Phase 3 configuration must be a JSON object")
    return values


def parse_phase3_canonical(raw: str) -> Phase3Config:
    """Validate bounded canonical JSON while restoring frozen integer map keys."""

    values = _require_bounded_json(raw)
    data = values.get("data")
    if not isinstance(data, dict):
        raise ValueError("full Phase 3 configuration requires a data object")
    allocation = data.get("phase1_gate")
    if not isinstance(allocation, dict) or set(allocation) != _ALLOCATION_NAMES:
        raise ValueError("full Phase 3 configuration requires the exact known allocation mappings")
    if any(not isinstance(counts, dict) for counts in allocation.values()):
        raise ValueError("full Phase 3 configuration requires the exact known allocation mappings")
    for name, counts in allocation.items():
        assert isinstance(counts, dict)
        restored: dict[int, object] = {}
        for key, value in counts.items():
            if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
                raise ValueError("allocation key must be a canonical integer")
            restored[int(key)] = value
        allocation[name] = restored
    config = Phase3Config.model_validate(values)
    if canonical_json_bytes(config).decode("utf-8") != raw:
        raise ValueError("full Phase 3 configuration must be canonical")
    return config
