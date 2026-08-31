from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
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
    assert data.train_distractor_link_records == (0, 12)
    assert data.ood_distractor_link_records == (16, 48)
    assert data.terminal_hazard_records == 2
    assert data.terminal_safe_records == 1
    assert data.manifest_sizes.validation == 10_000
    assert data.manifest_sizes.compute_curve_subset_iid == 2_500
    assert data.oracle_timing.delta_min == 0.05
    assert data.oracle_timing.action_target_fraction == 0.825


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
    assert resolved.config.stress.branching_records == (2, 6)
    assert resolved.config.stress.irrelevant_cycle_length == (2, 6)
    assert resolved.config.stress.contradiction_records == (2, 4)
    assert resolved.config.stress.overflow_record_count == (65, 80)


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
