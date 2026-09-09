"""Strict configuration for the Phase 2 flow and event engine."""

from typing import Literal, Self

from pydantic import field_validator, model_validator

from silent_cascade.env.config import Phase1Config
from silent_cascade.validation import StrictModel


def _require_exact_numeric_type(value: object, expected_type: type[int] | type[float]) -> object:
    if type(value) is not expected_type:
        raise ValueError(f"value must be an exact {expected_type.__name__}")
    return value


class StateDimensionsConfig(StrictModel):
    z_fast: Literal[256]
    z_slow: Literal[64]
    drives: Literal[8]
    guard_accumulators: Literal[3]
    focus_key: Literal[64]
    hypothesis_latent: Literal[64]

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


class FlowDynamicsConfig(StrictModel):
    rate_min: Literal[1.0e-5]
    rate_max: Literal[20.0]
    state_min: Literal[-1.0]
    state_max: Literal[1.0]

    @field_validator("rate_min", "rate_max", "state_min", "state_max", mode="before")
    @classmethod
    def require_exact_float_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, float)


class GuardDynamicsConfig(StrictModel):
    threshold: Literal[1.0]
    rate_min: Literal[1.0e-5]
    rate_max: Literal[500.0]
    active_margin: Literal[1.10]
    inactive_margin: Literal[0.90]
    minimum_internal_gap: Literal[1.0e-4]
    same_kind_refractory: Literal[1.0e-3]
    near_tie_tolerance: Literal[1.0e-9]
    maximum_consecutive_gap_clamps: Literal[4]

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


class ScriptedAgentConfig(StrictModel):
    action_target_fraction: Literal[0.825]

    @field_validator("action_target_fraction", mode="before")
    @classmethod
    def require_exact_float_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, float)


class EventFlowConfig(StrictModel):
    dimensions: StateDimensionsConfig
    flow: FlowDynamicsConfig
    guards: GuardDynamicsConfig
    scripted: ScriptedAgentConfig
    max_internal_events: Literal[64]

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
