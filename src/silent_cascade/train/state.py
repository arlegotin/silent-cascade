"""Strict serializable progress contracts for Phase 3 training."""

from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from silent_cascade.errors import SilentCascadeError
from silent_cascade.validation import StrictModel

type TrainingStage = Literal["smoke", "one_hop", "two_hop", "primary", "robustness"]


class TrainingError(SilentCascadeError):
    """A bounded training or checkpoint contract was violated."""

    code = "training_error"


class CheckpointDescriptor(StrictModel):
    """Content and provenance identity of a retained training archive."""

    schema_version: Literal["phase3-checkpoint-descriptor-v1"] = "phase3-checkpoint-descriptor-v1"
    relative_path: str = Field(min_length=1, max_length=512)
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    optimizer_step: int = Field(ge=0, le=75_000)
    stage: TrainingStage
    validation_metric: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("relative_path")
    @classmethod
    def require_safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("checkpoint path must be a safe relative path")
        if "\\" in value:
            raise ValueError("checkpoint path must use portable relative separators")
        return value

    @field_validator("validation_metric", mode="before")
    @classmethod
    def require_exact_optional_float(cls, value: object) -> object:
        if value is not None and type(value) is not float:
            raise ValueError("validation metric must be an exact float when present")
        return value


class TrainProgress(StrictModel):
    """Durable private counters and selection state; never a model input."""

    schema_version: Literal["phase3-train-progress-v1"] = "phase3-train-progress-v1"
    optimizer_step: int = Field(ge=0, le=75_000)
    next_batch_counter: int = Field(ge=0, le=75_000)
    stage: TrainingStage
    train_root_seed: int = Field(ge=0, le=(1 << 63) - 1)
    train_public_id_seed: int = Field(ge=0, le=(1 << 63) - 1)
    best_metric: float | None = Field(default=None, ge=0.0, le=1.0)
    best_step: int | None = Field(default=None, ge=0, le=75_000)
    patience_counter: int = Field(ge=0, le=15)
    retained_checkpoints: tuple[CheckpointDescriptor, ...] = Field(max_length=4)
    validation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("best_metric", mode="before")
    @classmethod
    def require_exact_optional_float(cls, value: object) -> object:
        if value is not None and type(value) is not float:
            raise ValueError("best metric must be an exact float when present")
        return value

    @model_validator(mode="after")
    def validate_selection_state(self) -> Self:
        if (self.best_metric is None) != (self.best_step is None):
            raise ValueError("best metric and best step must either both be present or both absent")
        if self.best_step is not None and self.best_step > self.optimizer_step:
            raise ValueError("best step cannot exceed the current optimizer step")
        paths = [descriptor.relative_path for descriptor in self.retained_checkpoints]
        if len(paths) != len(set(paths)):
            raise ValueError("retained checkpoint paths must be unique")
        return self
