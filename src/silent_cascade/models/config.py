"""Closed, strictly validated dimensions for the Phase 3 neural model."""

from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from silent_cascade.validation import StrictModel

_NEURAL_INTEGER_FIELDS = (
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
)


class LossWeights(StrictModel):
    """Combined loss coefficients with the approved fourfold pilot envelope."""

    guard: float = Field(default=1.0, ge=0.25, le=4.0)
    retrieval: float = Field(default=1.0, ge=0.25, le=4.0)
    compose_type: float = Field(default=1.0, ge=0.25, le=4.0)
    focus: float = Field(default=0.5, ge=0.125, le=2.0)
    hazard: float = Field(default=1.0, ge=0.25, le=4.0)
    deadline: float = Field(default=0.25, ge=0.0625, le=1.0)
    action: float = Field(default=1.0, ge=0.25, le=4.0)
    action_time: float = Field(default=0.25, ge=0.0625, le=1.0)
    state: float = Field(default=0.05, ge=0.0125, le=0.2)
    event_cost: float = Field(default=0.001, ge=0.00025, le=0.004)

    @field_validator("*", mode="before")
    @classmethod
    def require_exact_float_type(cls, value: object) -> object:
        if type(value) is not float:
            raise ValueError("loss weight must be an exact float")
        return value


class NeuralModelConfig(StrictModel):
    """One of the two approved EventFlow neural architectures.

    Interface dimensions stay fixed in both profiles. The debug profile only
    narrows hidden computation used by the fixed-example overfit regression.
    """

    architecture_profile: Literal["production", "debug"] = "production"
    max_trainable_parameters: Literal[5_000_000] = 5_000_000
    memory_slots: Literal[64] = 64
    max_batch_size: Literal[128] = 128
    entity_count: Literal[64] = 64
    entity_dim: Literal[32] = 32
    kind_dim: Literal[8] = 8
    hazard_dim: Literal[8] = 8
    provenance_dim: Literal[4] = 4
    record_scalar_dim: Literal[6] = 6
    record_input_dim: Literal[90] = 90
    record_hidden_dim: Literal[192, 96] = 192
    record_dim: Literal[96] = 96
    external_input_dim: Literal[132] = 132
    external_hidden_dim: Literal[256, 64] = 256
    external_dim: Literal[256] = 256
    mode_count: Literal[6] = 6
    mode_dim: Literal[8] = 8
    time_feature_dim: Literal[2] = 2
    hypothesis_feature_dim: Literal[8] = 8
    retrieval_query_input_dim: Literal[496] = 496
    query_dim: Literal[256, 64] = 256
    top_k: Literal[4] = 4
    controller_hidden_dim: Literal[512, 128] = 512
    jump_hidden_dim: Literal[512, 128] = 512
    head_hidden_dim: Literal[256, 64] = 256

    @field_validator(*_NEURAL_INTEGER_FIELDS, mode="before")
    @classmethod
    def require_exact_integer_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("neural dimension must be an exact integer")
        return value

    @model_validator(mode="after")
    def validate_closed_profile(self) -> Self:
        widths = (
            self.record_hidden_dim,
            self.external_hidden_dim,
            self.query_dim,
            self.controller_hidden_dim,
            self.jump_hidden_dim,
            self.head_hidden_dim,
        )
        expected = {
            "production": (192, 256, 256, 512, 512, 256),
            "debug": (96, 64, 64, 128, 128, 64),
        }[self.architecture_profile]
        if widths != expected:
            raise ValueError(f"{self.architecture_profile} architecture widths must be exact")
        return self

    @property
    def latent_dim(self) -> int:
        return 256 + 64 + 8 + 64 + 64

    @property
    def context_dim(self) -> int:
        return (
            self.latent_dim
            + self.mode_dim
            + self.record_dim
            + self.time_feature_dim
            + self.hypothesis_feature_dim
        )

    @property
    def preview_dim(self) -> int:
        return 4 + self.record_dim

    @property
    def retrieval_pair_input_dim(self) -> int:
        return self.query_dim + self.record_dim

    @property
    def bilinear_record_output_dim(self) -> int:
        return self.query_dim

    @property
    def controller_input_dim(self) -> int:
        return self.context_dim + self.preview_dim

    @property
    def controller_output_dim(self) -> int:
        return 2 * self.latent_dim + 6

    @property
    def external_injection_output_dim(self) -> int:
        return 2 * (256 + 64)

    @property
    def jump_input_dim(self) -> int:
        return self.context_dim + self.record_dim + self.kind_dim

    @property
    def jump_output_dim(self) -> int:
        return 2 * self.latent_dim

    @property
    def composition_input_dim(self) -> int:
        return self.context_dim + self.record_dim

    @property
    def action_input_dim(self) -> int:
        return self.context_dim
