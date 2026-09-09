"""Strict configuration for the Phase 2 flow and event engine."""

from typing import Self

from pydantic import Field, field_validator, model_validator

from silent_cascade.env.config import Phase1Config
from silent_cascade.validation import StrictModel


def _require_exact_numeric_type(value: object, expected_type: type[int] | type[float]) -> object:
    if type(value) is not expected_type:
        raise ValueError(f"value must be an exact {expected_type.__name__}")
    return value


class StateDimensionsConfig(StrictModel):
    z_fast: int = Field(gt=0)
    z_slow: int = Field(gt=0)
    drives: int = Field(gt=0)
    guard_accumulators: int = Field(gt=0)
    focus_key: int = Field(gt=0)
    hypothesis_latent: int = Field(gt=0)

    @field_validator(
        "z_fast",
        "z_slow",
        "drives",
        "guard_accumulators",
        "focus_key",
        "hypothesis_latent",
        mode="before",
    )
    @classmethod
    def require_exact_integer_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, int)

    @field_validator("guard_accumulators")
    @classmethod
    def require_three_guard_accumulators(cls, value: int) -> int:
        if value != 3:
            raise ValueError("guard_accumulators must equal 3")
        return value


class FlowDynamicsConfig(StrictModel):
    rate_min: float = Field(gt=0.0)
    rate_max: float = Field(gt=0.0)
    state_min: float
    state_max: float

    @field_validator("rate_min", "rate_max", "state_min", "state_max", mode="before")
    @classmethod
    def require_exact_float_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, float)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.rate_min > self.rate_max:
            raise ValueError("flow rate bounds must be ordered")
        if self.state_min >= self.state_max:
            raise ValueError("flow state bounds must be strictly ordered")
        return self


class GuardDynamicsConfig(StrictModel):
    threshold: float = Field(gt=0.0)
    rate_min: float = Field(gt=0.0)
    rate_max: float = Field(gt=0.0)
    active_margin: float = Field(gt=0.0)
    inactive_margin: float = Field(ge=0.0)
    minimum_internal_gap: float = Field(gt=0.0)
    same_kind_refractory: float = Field(gt=0.0)
    near_tie_tolerance: float = Field(ge=0.0)
    maximum_consecutive_gap_clamps: int = Field(ge=1)

    @field_validator(
        "threshold",
        "rate_min",
        "rate_max",
        "active_margin",
        "inactive_margin",
        "minimum_internal_gap",
        "same_kind_refractory",
        "near_tie_tolerance",
        mode="before",
    )
    @classmethod
    def require_exact_float_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, float)

    @field_validator("maximum_consecutive_gap_clamps", mode="before")
    @classmethod
    def require_exact_integer_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, int)

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.rate_min > self.rate_max:
            raise ValueError("guard rate bounds must be ordered")
        if not self.inactive_margin < self.threshold < self.active_margin:
            raise ValueError("guard margins must straddle the threshold")
        return self


class ScriptedAgentConfig(StrictModel):
    action_target_fraction: float = Field(gt=0.0, lt=1.0)

    @field_validator("action_target_fraction", mode="before")
    @classmethod
    def require_exact_float_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, float)


class EventFlowConfig(StrictModel):
    dimensions: StateDimensionsConfig
    flow: FlowDynamicsConfig
    guards: GuardDynamicsConfig
    scripted: ScriptedAgentConfig
    max_internal_events: int = Field(ge=1, le=64)

    @field_validator("max_internal_events", mode="before")
    @classmethod
    def require_exact_integer_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, int)


class Phase2Config(Phase1Config):
    event_flow: EventFlowConfig

    @model_validator(mode="after")
    def validate_phase2_limits(self) -> Self:
        if self.event_flow.max_internal_events != self.limits.max_eventflow_events:
            raise ValueError("EventFlow event ceilings must match")
        if self.data.primary_memory_capacity != self.limits.primary_memory_records:
            raise ValueError("memory ceilings must match")
        if (
            self.event_flow.scripted.action_target_fraction
            != self.data.oracle_timing.action_target_fraction
        ):
            raise ValueError("scripted target must match the standing OFD objective")
        return self
