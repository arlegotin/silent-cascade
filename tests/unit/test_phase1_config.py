import math
from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import OracleTimingConfig, Phase1Config, SplitNamespace, SuiteName
from silent_cascade.errors import ConfigurationError


def resolve_primary(*overlays: Path):
    return resolve_config(
        Phase1Config,
        [Path("configs/base.yaml"), Path("configs/data/primary.yaml"), *overlays],
    )


def write_overlay(tmp_path: Path, text: str) -> Path:
    overlay = tmp_path / "data.yaml"
    overlay.write_text(text, encoding="utf-8")
    return overlay


def nested_boolean_overlay(dotted_path: str) -> str:
    sections = dotted_path.split(".")
    lines = [f"{'  ' * depth}{section}:" for depth, section in enumerate(sections[:-1])]
    lines.append(f"{'  ' * (len(sections) - 1)}{sections[-1]}: true")
    return "\n".join(lines) + "\n"


def test_primary_data_config_resolves_exact_frozen_values() -> None:
    resolved = resolve_primary()
    data = resolved.config.data

    assert data.generator_version == "ofd-v1"
    assert data.max_entities == 64
    assert data.hazard_types == 4
    assert data.positive_fraction == 0.5
    assert data.safe_terminal_fraction_within_negatives == 0.5
    assert data.train_path_lengths == (2, 3, 4)
    assert data.iid_test_path_lengths == (2, 3, 4)
    assert data.ood_depth_path_lengths == (5, 6, 7, 8)
    assert data.ood_short_delay_path_lengths == (2, 3, 4)
    assert data.train_delay_log_uniform == (8.0, 64.0)
    assert data.ood_depth_delay_log_uniform == (16.0, 128.0)
    assert data.ood_short_delay_log_uniform == (1.0, 8.0)
    assert data.ood_long_delay_log_uniform == (64.0, 1024.0)
    assert data.observation_gap_log_uniform == (0.1, 8.0)
    assert data.train_distractor_link_records == (0, 12)
    assert data.ood_distractor_link_records == (16, 48)
    assert data.terminal_hazard_records == 2
    assert data.terminal_safe_records == 1
    assert data.primary_memory_capacity == 64
    assert data.max_eventflow_internal_events == 64
    assert data.max_fixed_grid_opportunities == 25_000
    assert data.max_generation_attempts == 1_000
    assert data.manifest_sizes.model_dump() == {
        "validation": 10_000,
        "iid_primary": 20_000,
        "ood_depth": 20_000,
        "ood_short_delay": 10_000,
        "ood_long_delay": 2_000,
        "clock_scale_0_1x": 5_000,
        "clock_scale_10x": 2_000,
        "distractor_flood": 10_000,
        "branching_stress": 5_000,
        "cycles_stress": 5_000,
        "contradiction_stress": 5_000,
        "checkpoint_stress": 2_000,
        "compute_curve_subset_iid": 2_500,
        "compute_curve_subset_ood_depth": 2_500,
    }
    assert data.oracle_timing.model_dump() == {
        "delta_0": 0.25,
        "delta_min": 0.05,
        "delta_max": 1.0,
        "jitter_log_std": 0.1,
        "terminal_compose_fraction": 0.60,
        "action_window_start_fraction": 0.75,
        "action_target_fraction": 0.825,
        "action_window_end_fraction": 0.90,
    }
    assert data.phase1_gate.model_dump() == {
        "iid_primary": {2: 8_000, 3: 8_000, 4: 8_000},
        "ood_depth": {5: 4_000, 6: 4_000, 7: 4_000, 8: 4_000},
        "ood_short_delay": {2: 8_000, 3: 8_000, 4: 8_000},
        "ood_long_delay": {2: 4_000, 3: 4_000, 4: 4_000},
        "distractor_flood": {2: 8_000, 3: 8_000, 4: 8_000},
        "clock_parent_episodes": {2: 1_668, 3: 1_668, 4: 1_664},
        "clock_10x_episodes": {2: 668, 3: 668, 4: 664},
    }
    assert data.leakage_audit.model_dump() == {
        "schema_version": "leakage-v1",
        "audit_seed": 2_026_083_091,
        "positive_control_seed": 2_026_083_092,
        "train_fraction": 0.80,
        "alpha": 0.01,
        "l2_penalty": 0.03,
        "optimizer_max_iterations": 500,
        "optimizer_gradient_tolerance": 1.0e-8,
        "optimizer_function_tolerance": 1.0e-12,
        "positive_control_min_balanced_accuracy": 0.95,
        "feature_batch_size": 4_096,
        "permutation_batch_size": 64,
        "max_feature_store_bytes": 800_000_000,
        "max_resident_working_bytes": 512_000_000,
        "test": {
            "episode_count": 1_200,
            "permutation_replicates": 199,
            "positive_control_episode_count": 1_200,
            "positive_control_permutation_replicates": 199,
            "minimum_test_examples_per_class": 8,
            "enforce_clean_statistical_gate": False,
        },
        "phase1_gate": {
            "episode_count": 100_000,
            "permutation_replicates": 4_999,
            "positive_control_episode_count": 8_000,
            "positive_control_permutation_replicates": 4_999,
            "minimum_test_examples_per_class": 200,
            "enforce_clean_statistical_gate": True,
        },
    }


_ORACLE_TIMING_FLOAT_FIELDS = (
    "delta_0",
    "delta_min",
    "delta_max",
    "jitter_log_std",
    "terminal_compose_fraction",
    "action_window_start_fraction",
    "action_target_fraction",
    "action_window_end_fraction",
)


@pytest.mark.parametrize("field", _ORACLE_TIMING_FLOAT_FIELDS)
def test_oracle_timing_config_accepts_each_exact_float_field(field: str) -> None:
    """Changing exact floats to reject their own type must fail this constructor contract."""
    values = OracleTimingConfig().model_dump()
    values[field] = float(values[field])

    timing = OracleTimingConfig(**values)

    assert type(getattr(timing, field)) is float


@pytest.mark.parametrize("field", _ORACLE_TIMING_FLOAT_FIELDS)
@pytest.mark.parametrize(
    ("invalid", "error"),
    (
        (1, "exact float"),
        (True, "exact float"),
        ("0.5", "exact float"),
        (math.nan, "finite number"),
        (math.inf, "finite number"),
    ),
)
def test_oracle_timing_config_rejects_non_exact_float_fields(
    field: str,
    invalid: object,
    error: str,
) -> None:
    """Changing timing validation to coerce JSON-like values must fail this contract."""
    values = OracleTimingConfig().model_dump()
    values[field] = invalid

    with pytest.raises(ValueError, match=error):
        OracleTimingConfig(**values)


def test_phase1_config_resolves_stress_ranges_as_tuples() -> None:
    resolved = resolve_config(
        Phase1Config,
        [
            Path("configs/base.yaml"),
            Path("configs/data/primary.yaml"),
            Path("configs/data/stress.yaml"),
        ],
    )

    assert resolved.config.stress is not None
    assert resolved.config.stress.model_dump() == {
        "branching_records": (2, 6),
        "irrelevant_cycle_length": (2, 6),
        "contradiction_records": (2, 4),
        "overflow_record_count": (65, 80),
        "near_miss_missing_edges": 1,
        "minimum_duration_epsilon": 1.0e-6,
        "minimum_duration_search_upper": 64.0,
        "minimum_duration_monotonic_grid_points": 257,
        "branching_freeze_episodes": 5_000,
        "cycles_freeze_episodes": 5_000,
        "contradiction_freeze_episodes": 5_000,
        "memory_overflow_freeze_episodes": 5_000,
        "null_near_miss_freeze_episodes": 5_000,
        "minimum_duration_freeze_episodes": 5_000,
        "checkpoint_freeze_episodes": 2_000,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("primary_memory_capacity", 65),
        ("max_eventflow_internal_events", 65),
        ("max_fixed_grid_opportunities", 25_001),
    ],
)
def test_phase1_config_rejects_data_limit_above_project_limit(
    tmp_path: Path, field: str, value: int
) -> None:
    overlay = write_overlay(tmp_path, f"data:\n  {field}: {value}\n")

    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_primary(overlay)


@pytest.mark.parametrize(
    ("limit_field", "limit_value", "failure"),
    [
        ("primary_memory_records", 63, "data memory capacity exceeds project limit"),
        ("max_eventflow_events", 63, "data event ceiling exceeds project limit"),
        ("max_fixed_grid_opportunities", 24_999, "data opportunity ceiling exceeds project limit"),
    ],
)
def test_phase1_config_cross_checks_frozen_data_ceilings_against_parent_limits(
    tmp_path: Path, limit_field: str, limit_value: int, failure: str
) -> None:
    overlay = write_overlay(tmp_path, f"limits:\n  {limit_field}: {limit_value}\n")

    with pytest.raises(ConfigurationError, match="configuration validation failed") as raised:
        resolve_primary(overlay)

    assert failure in raised.value.context["details"]


@pytest.mark.parametrize(
    "overlay_text",
    [
        "data:\n  unexpected: 1\n",
        "data:\n  max_generation_attempts: true\n",
        "data:\n  primary_memory_capacity: true\n",
        "data:\n  train_path_lengths: [2, true, 4]\n",
    ],
)
def test_phase1_config_rejects_unknown_keys_and_bool_as_int(
    tmp_path: Path, overlay_text: str
) -> None:
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_primary(write_overlay(tmp_path, overlay_text))


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("episode_count", "99996"),
        ("permutation_replicates", "4998"),
        ("positive_control_episode_count", "7980"),
        ("positive_control_permutation_replicates", "4998"),
        ("minimum_test_examples_per_class", "199"),
        ("enforce_clean_statistical_gate", "false"),
    ],
)
def test_phase1_config_rejects_noncanonical_leakage_gate_profile(
    field: str,
    replacement: str,
) -> None:
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(
            Phase1Config,
            [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
            set_overrides=(f"data.leakage_audit.phase1_gate.{field}={replacement}",),
        )


@pytest.mark.parametrize(
    "field",
    [
        "data.max_entities",
        "data.hazard_types",
        "data.positive_fraction",
        "data.safe_terminal_fraction_within_negatives",
        "data.terminal_hazard_records",
        "data.terminal_safe_records",
        "data.primary_memory_capacity",
        "data.max_eventflow_internal_events",
        "data.max_fixed_grid_opportunities",
        "data.manifest_sizes.validation",
        "data.manifest_sizes.iid_primary",
        "data.manifest_sizes.ood_depth",
        "data.manifest_sizes.ood_short_delay",
        "data.manifest_sizes.ood_long_delay",
        "data.manifest_sizes.clock_scale_0_1x",
        "data.manifest_sizes.clock_scale_10x",
        "data.manifest_sizes.distractor_flood",
        "data.manifest_sizes.branching_stress",
        "data.manifest_sizes.cycles_stress",
        "data.manifest_sizes.contradiction_stress",
        "data.manifest_sizes.checkpoint_stress",
        "data.manifest_sizes.compute_curve_subset_iid",
        "data.manifest_sizes.compute_curve_subset_ood_depth",
        "data.leakage_audit.audit_seed",
        "data.leakage_audit.positive_control_seed",
        "data.leakage_audit.train_fraction",
        "data.leakage_audit.alpha",
        "data.leakage_audit.l2_penalty",
        "data.leakage_audit.optimizer_max_iterations",
        "data.leakage_audit.optimizer_gradient_tolerance",
        "data.leakage_audit.optimizer_function_tolerance",
        "data.leakage_audit.positive_control_min_balanced_accuracy",
        "stress.near_miss_missing_edges",
        "stress.minimum_duration_search_upper",
        "stress.minimum_duration_monotonic_grid_points",
        "stress.branching_freeze_episodes",
        "stress.cycles_freeze_episodes",
        "stress.contradiction_freeze_episodes",
        "stress.memory_overflow_freeze_episodes",
        "stress.null_near_miss_freeze_episodes",
        "stress.minimum_duration_freeze_episodes",
        "stress.checkpoint_freeze_episodes",
    ],
)
def test_phase1_config_rejects_booleans_for_all_frozen_numeric_fields(
    tmp_path: Path, field: str
) -> None:
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(
            Phase1Config,
            [
                Path("configs/base.yaml"),
                Path("configs/data/primary.yaml"),
                Path("configs/data/stress.yaml"),
                write_overlay(tmp_path, nested_boolean_overlay(field)),
            ],
        )


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("train_path_lengths", "[2, 3, 3]"),
        ("iid_test_path_lengths", "[3, 2, 4]"),
        ("ood_depth_path_lengths", "[5, 6, 0]"),
        ("ood_short_delay_path_lengths", "[2, true, 4]"),
        ("ood_depth_delay_log_uniform", "[128.0, 16.0]"),
        ("train_distractor_link_records", "[12, 12]"),
        ("ood_distractor_link_records", "[16, .inf]"),
        ("train_delay_log_uniform", "[8.0, 8.0]"),
        ("ood_short_delay_log_uniform", "[.nan, 8.0]"),
        ("ood_long_delay_log_uniform", "[1024.0, 64.0]"),
        ("observation_gap_log_uniform", "[8.0, 0.1]"),
    ],
)
def test_phase1_config_rejects_invalid_path_and_range_tuples(
    tmp_path: Path, field: str, replacement: str
) -> None:
    overlay = write_overlay(tmp_path, f"data:\n  {field}: {replacement}\n")

    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_primary(overlay)


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("delta_0", "0.01"),
        ("terminal_compose_fraction", "0.80"),
        ("action_target_fraction", "0.95"),
    ],
)
def test_phase1_config_rejects_invalid_oracle_timing_order(
    tmp_path: Path, field: str, replacement: str
) -> None:
    overlay = write_overlay(tmp_path, f"data:\n  oracle_timing:\n    {field}: {replacement}\n")

    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_primary(overlay)


def test_phase1_config_rejects_nonfrozen_gate_allocation(tmp_path: Path) -> None:
    overlay = write_overlay(tmp_path, "data:\n  phase1_gate:\n    iid_primary: {2: 7999}\n")

    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_primary(overlay)


def test_phase1_suite_and_split_names_are_frozen_strings() -> None:
    assert {suite.value for suite in SuiteName} == {
        "validation",
        "iid_primary",
        "ood_depth",
        "ood_short_delay",
        "ood_long_delay",
        "distractor_flood",
        "clock_scale_0_1x",
        "clock_scale_10x",
        "branching_stress",
        "cycles_stress",
        "contradiction_stress",
        "memory_overflow_stress",
        "null_near_miss_stress",
        "minimum_duration_stress",
        "checkpoint_stress",
    }
    assert {split.value for split in SplitNamespace} == {
        "train",
        "debug",
        "validation",
        "phase1_gate",
        "frozen",
    }
