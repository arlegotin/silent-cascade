"""Strict, frozen Phase 1 OFD data configuration."""

import math
from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from silent_cascade.config import ProjectConfig
from silent_cascade.validation import StrictModel


class SuiteName(StrEnum):
    VALIDATION = "validation"
    IID_PRIMARY = "iid_primary"
    OOD_DEPTH = "ood_depth"
    OOD_SHORT_DELAY = "ood_short_delay"
    OOD_LONG_DELAY = "ood_long_delay"
    DISTRACTOR_FLOOD = "distractor_flood"
    CLOCK_SCALE_0_1X = "clock_scale_0_1x"
    CLOCK_SCALE_10X = "clock_scale_10x"
    BRANCHING_STRESS = "branching_stress"
    CYCLES_STRESS = "cycles_stress"
    CONTRADICTION_STRESS = "contradiction_stress"
    MEMORY_OVERFLOW_STRESS = "memory_overflow_stress"
    NULL_NEAR_MISS_STRESS = "null_near_miss_stress"
    MINIMUM_DURATION_STRESS = "minimum_duration_stress"
    CHECKPOINT_STRESS = "checkpoint_stress"


class SplitNamespace(StrEnum):
    TRAIN = "train"
    DEBUG = "debug"
    VALIDATION = "validation"
    PHASE1_GATE = "phase1_gate"
    FROZEN = "frozen"


def _yaml_list_to_tuple(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


def _require_exact_numeric_type(value: object, expected_type: type[int] | type[float]) -> object:
    if type(value) is not expected_type:
        raise ValueError(f"value must be an exact {expected_type.__name__}")
    return value


def _strict_path_lengths(value: object) -> tuple[int, ...]:
    value = _yaml_list_to_tuple(value)
    if not isinstance(value, tuple):
        raise ValueError("path lengths must be a tuple or YAML list")
    if not value or any(type(item) is not int or item <= 0 for item in value):
        raise ValueError("path lengths must contain only positive integers")
    if tuple(sorted(value)) != value or len(set(value)) != len(value):
        raise ValueError("path lengths must be sorted and unique")
    return value


def _strict_increasing_range(value: object) -> tuple[int | float, int | float]:
    value = _yaml_list_to_tuple(value)
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("range must have exactly two endpoints")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError("range endpoints must be finite numbers and not booleans")
    if value[0] >= value[1]:
        raise ValueError("range endpoints must be strictly increasing")
    return value


class OracleTimingConfig(StrictModel):
    delta_0: float = Field(default=0.25, gt=0.0)
    delta_min: float = Field(default=0.05, gt=0.0)
    delta_max: float = Field(default=1.0, gt=0.0)
    jitter_log_std: float = Field(default=0.1, ge=0.0)
    terminal_compose_fraction: float = Field(default=0.60, gt=0.0, lt=1.0)
    action_window_start_fraction: float = Field(default=0.75, gt=0.0, lt=1.0)
    action_target_fraction: float = Field(default=0.825, gt=0.0, lt=1.0)
    action_window_end_fraction: float = Field(default=0.90, gt=0.0, le=1.0)

    @field_validator(
        "delta_0",
        "delta_min",
        "delta_max",
        "jitter_log_std",
        "terminal_compose_fraction",
        "action_window_start_fraction",
        "action_target_fraction",
        "action_window_end_fraction",
        mode="before",
    )
    @classmethod
    def require_exact_float_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, float)

    @model_validator(mode="after")
    def validate_timing_order(self) -> Self:
        if not self.delta_min <= self.delta_0 <= self.delta_max:
            raise ValueError("delta_min <= delta_0 <= delta_max is required")
        if self.terminal_compose_fraction >= self.action_window_start_fraction:
            raise ValueError("terminal composition must precede the action window")
        if not (
            self.action_window_start_fraction
            < self.action_target_fraction
            < self.action_window_end_fraction
        ):
            raise ValueError("action target must be strictly inside the action window")
        return self


class ManifestSizesConfig(StrictModel):
    validation: Literal[10_000] = 10_000
    iid_primary: Literal[20_000] = 20_000
    ood_depth: Literal[20_000] = 20_000
    ood_short_delay: Literal[10_000] = 10_000
    ood_long_delay: Literal[2_000] = 2_000
    clock_scale_0_1x: Literal[5_000] = 5_000
    clock_scale_10x: Literal[2_000] = 2_000
    distractor_flood: Literal[10_000] = 10_000
    branching_stress: Literal[5_000] = 5_000
    cycles_stress: Literal[5_000] = 5_000
    contradiction_stress: Literal[5_000] = 5_000
    checkpoint_stress: Literal[2_000] = 2_000
    compute_curve_subset_iid: Literal[2_500] = 2_500
    compute_curve_subset_ood_depth: Literal[2_500] = 2_500

    @field_validator(
        "validation",
        "iid_primary",
        "ood_depth",
        "ood_short_delay",
        "ood_long_delay",
        "clock_scale_0_1x",
        "clock_scale_10x",
        "distractor_flood",
        "branching_stress",
        "cycles_stress",
        "contradiction_stress",
        "checkpoint_stress",
        "compute_curve_subset_iid",
        "compute_curve_subset_ood_depth",
        mode="before",
    )
    @classmethod
    def require_exact_integer_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, int)


class Phase1GateAllocation(StrictModel):
    iid_primary: dict[Literal[2, 3, 4], int]
    ood_depth: dict[Literal[5, 6, 7, 8], int]
    ood_short_delay: dict[Literal[2, 3, 4], int]
    ood_long_delay: dict[Literal[2, 3, 4], int]
    distractor_flood: dict[Literal[2, 3, 4], int]
    clock_parent_episodes: dict[Literal[2, 3, 4], int]
    clock_10x_episodes: dict[Literal[2, 3, 4], int]

    @model_validator(mode="after")
    def validate_exact_allocation(self) -> Self:
        expected = {
            "iid_primary": {2: 8_000, 3: 8_000, 4: 8_000},
            "ood_depth": {5: 4_000, 6: 4_000, 7: 4_000, 8: 4_000},
            "ood_short_delay": {2: 8_000, 3: 8_000, 4: 8_000},
            "ood_long_delay": {2: 4_000, 3: 4_000, 4: 4_000},
            "distractor_flood": {2: 8_000, 3: 8_000, 4: 8_000},
            "clock_parent_episodes": {2: 1_668, 3: 1_668, 4: 1_664},
            "clock_10x_episodes": {2: 668, 3: 668, 4: 664},
        }
        actual = {name: getattr(self, name) for name in expected}
        if actual != expected:
            raise ValueError("phase1_gate must equal the frozen allocation")
        return self


class LeakageAuditProfileConfig(StrictModel):
    episode_count: int = Field(gt=0, multiple_of=4)
    permutation_replicates: int = Field(gt=0)
    positive_control_episode_count: int = Field(gt=0, multiple_of=20)
    positive_control_permutation_replicates: int = Field(gt=0)
    minimum_test_examples_per_class: int = Field(gt=0)
    enforce_clean_statistical_gate: bool


class LeakageAuditConfig(StrictModel):
    schema_version: Literal["leakage-v1"] = "leakage-v1"
    audit_seed: Literal[2026083091] = 2026083091
    positive_control_seed: Literal[2026083092] = 2026083092
    train_fraction: Literal[0.8] = 0.8
    alpha: Literal[0.01] = 0.01
    l2_penalty: Literal[1.0] = 1.0
    optimizer_max_iterations: Literal[500] = 500
    optimizer_gradient_tolerance: Literal[1.0e-8] = 1.0e-8
    optimizer_function_tolerance: Literal[1.0e-12] = 1.0e-12
    positive_control_min_balanced_accuracy: Literal[0.95] = 0.95
    feature_batch_size: int = Field(default=4096, ge=1, le=4096)
    permutation_batch_size: int = Field(default=64, ge=1, le=64)
    max_feature_store_bytes: int = Field(default=800_000_000, gt=0, le=800_000_000)
    max_resident_working_bytes: int = Field(default=512_000_000, gt=0, le=512_000_000)
    test: LeakageAuditProfileConfig
    phase1_gate: LeakageAuditProfileConfig

    @field_validator(
        "audit_seed",
        "positive_control_seed",
        "optimizer_max_iterations",
        mode="before",
    )
    @classmethod
    def require_exact_integer_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, int)

    @field_validator(
        "train_fraction",
        "alpha",
        "l2_penalty",
        "optimizer_gradient_tolerance",
        "optimizer_function_tolerance",
        "positive_control_min_balanced_accuracy",
        mode="before",
    )
    @classmethod
    def require_exact_float_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, float)


class OFDDataConfig(StrictModel):
    generator_version: Literal["ofd-v1"] = "ofd-v1"
    max_entities: Literal[64] = 64
    hazard_types: Literal[4] = 4
    positive_fraction: Literal[0.5] = 0.5
    safe_terminal_fraction_within_negatives: Literal[0.5] = 0.5
    train_path_lengths: tuple[int, ...]
    iid_test_path_lengths: tuple[int, ...]
    ood_depth_path_lengths: tuple[int, ...]
    ood_short_delay_path_lengths: tuple[int, ...]
    ood_depth_delay_log_uniform: tuple[float, float]
    terminal_hazard_records: Literal[2] = 2
    terminal_safe_records: Literal[1] = 1
    train_distractor_link_records: tuple[int, int]
    ood_distractor_link_records: tuple[int, int]
    train_delay_log_uniform: tuple[float, float]
    ood_short_delay_log_uniform: tuple[float, float]
    ood_long_delay_log_uniform: tuple[float, float]
    observation_gap_log_uniform: tuple[float, float]
    primary_memory_capacity: Literal[64] = 64
    max_eventflow_internal_events: Literal[64] = 64
    max_fixed_grid_opportunities: Literal[25_000] = 25_000
    max_generation_attempts: int = Field(default=1_000, ge=1, le=1_000)
    manifest_sizes: ManifestSizesConfig
    oracle_timing: OracleTimingConfig
    phase1_gate: Phase1GateAllocation
    leakage_audit: LeakageAuditConfig

    @field_validator(
        "max_entities",
        "hazard_types",
        "terminal_hazard_records",
        "terminal_safe_records",
        "primary_memory_capacity",
        "max_eventflow_internal_events",
        "max_fixed_grid_opportunities",
        mode="before",
    )
    @classmethod
    def require_exact_integer_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, int)

    @field_validator(
        "positive_fraction",
        "safe_terminal_fraction_within_negatives",
        mode="before",
    )
    @classmethod
    def require_exact_float_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, float)

    @field_validator(
        "train_path_lengths",
        "iid_test_path_lengths",
        "ood_depth_path_lengths",
        "ood_short_delay_path_lengths",
        mode="before",
    )
    @classmethod
    def validate_path_lengths(cls, value: object) -> tuple[int, ...]:
        return _strict_path_lengths(value)

    @field_validator(
        "ood_depth_delay_log_uniform",
        "train_distractor_link_records",
        "ood_distractor_link_records",
        "train_delay_log_uniform",
        "ood_short_delay_log_uniform",
        "ood_long_delay_log_uniform",
        "observation_gap_log_uniform",
        mode="before",
    )
    @classmethod
    def validate_ranges(cls, value: object) -> tuple[int | float, int | float]:
        return _strict_increasing_range(value)


class StressDataConfig(StrictModel):
    branching_records: tuple[int, int]
    irrelevant_cycle_length: tuple[int, int]
    contradiction_records: tuple[int, int]
    overflow_record_count: tuple[int, int]
    near_miss_missing_edges: Literal[1] = 1
    minimum_duration_epsilon: float = Field(default=1.0e-6, gt=0.0)
    minimum_duration_search_upper: Literal[64.0] = 64.0
    minimum_duration_monotonic_grid_points: Literal[257] = 257
    branching_freeze_episodes: Literal[5_000] = 5_000
    cycles_freeze_episodes: Literal[5_000] = 5_000
    contradiction_freeze_episodes: Literal[5_000] = 5_000
    memory_overflow_freeze_episodes: Literal[5_000] = 5_000
    null_near_miss_freeze_episodes: Literal[5_000] = 5_000
    minimum_duration_freeze_episodes: Literal[5_000] = 5_000
    checkpoint_freeze_episodes: Literal[2_000] = 2_000

    @field_validator(
        "near_miss_missing_edges",
        "minimum_duration_monotonic_grid_points",
        "branching_freeze_episodes",
        "cycles_freeze_episodes",
        "contradiction_freeze_episodes",
        "memory_overflow_freeze_episodes",
        "null_near_miss_freeze_episodes",
        "minimum_duration_freeze_episodes",
        "checkpoint_freeze_episodes",
        mode="before",
    )
    @classmethod
    def require_exact_integer_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, int)

    @field_validator("minimum_duration_search_upper", mode="before")
    @classmethod
    def require_exact_float_type(cls, value: object) -> object:
        return _require_exact_numeric_type(value, float)

    @field_validator(
        "branching_records",
        "irrelevant_cycle_length",
        "contradiction_records",
        "overflow_record_count",
        mode="before",
    )
    @classmethod
    def validate_ranges(cls, value: object) -> tuple[int | float, int | float]:
        return _strict_increasing_range(value)


class Phase1Config(ProjectConfig):
    data: OFDDataConfig
    stress: StressDataConfig | None = None

    @model_validator(mode="after")
    def validate_phase1_limits(self) -> Self:
        if self.data.primary_memory_capacity > self.limits.primary_memory_records:
            raise ValueError("data memory capacity exceeds project limit")
        if self.data.max_eventflow_internal_events > self.limits.max_eventflow_events:
            raise ValueError("data event ceiling exceeds project limit")
        if self.data.max_fixed_grid_opportunities > self.limits.max_fixed_grid_opportunities:
            raise ValueError("data opportunity ceiling exceeds project limit")
        return self
