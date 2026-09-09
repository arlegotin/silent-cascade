from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.errors import ConfigurationError
from silent_cascade.eventflow.config import EventFlowConfig, Phase2Config, ScriptedAgentConfig


def resolve_event_flow(*overlays: Path):
    return resolve_config(
        Phase2Config,
        [
            Path("configs/base.yaml"),
            Path("configs/data/primary.yaml"),
            Path("configs/model/event_flow.yaml"),
            *overlays,
        ],
    )


def write_overlay(tmp_path: Path, text: str) -> Path:
    overlay = tmp_path / "event-flow-overlay.yaml"
    overlay.write_text(text, encoding="utf-8")
    return overlay


def test_event_flow_config_resolves_exact_phase2_contract() -> None:
    resolved = resolve_config(
        Phase2Config,
        [
            Path("configs/base.yaml"),
            Path("configs/data/primary.yaml"),
            Path("configs/model/event_flow.yaml"),
        ],
    )
    cfg = resolved.config.event_flow
    assert cfg.dimensions.model_dump() == {
        "z_fast": 256,
        "z_slow": 64,
        "drives": 8,
        "guard_accumulators": 3,
        "focus_key": 64,
        "hypothesis_latent": 64,
    }
    assert (cfg.flow.rate_min, cfg.flow.rate_max) == (1.0e-5, 20.0)
    assert cfg.flow.state_min == -1.0
    assert cfg.flow.state_max == 1.0
    assert cfg.guards.threshold == 1.0
    assert (cfg.guards.rate_min, cfg.guards.rate_max) == (1.0e-5, 500.0)
    assert cfg.guards.active_margin == 1.10
    assert cfg.guards.inactive_margin == 0.90
    assert cfg.guards.minimum_internal_gap == 1.0e-4
    assert cfg.guards.same_kind_refractory == 1.0e-3
    assert cfg.guards.near_tie_tolerance == 1.0e-9
    assert cfg.guards.maximum_consecutive_gap_clamps == 4
    assert cfg.scripted.action_target_fraction == 0.825
    assert cfg.max_internal_events == 64


@pytest.mark.parametrize(
    ("dotted_path", "mutated_value"),
    [
        ("dimensions.z_fast", "257"),
        ("dimensions.z_slow", "65"),
        ("dimensions.drives", "9"),
        ("dimensions.guard_accumulators", "4"),
        ("dimensions.focus_key", "65"),
        ("dimensions.hypothesis_latent", "65"),
        ("flow.rate_min", "2.0e-5"),
        ("flow.rate_max", "21.0"),
        ("flow.state_min", "-2.0"),
        ("flow.state_max", "2.0"),
        ("guards.threshold", "2.0"),
        ("guards.rate_min", "2.0e-5"),
        ("guards.rate_max", "501.0"),
        ("guards.active_margin", "1.11"),
        ("guards.inactive_margin", "0.89"),
        ("guards.minimum_internal_gap", "2.0e-4"),
        ("guards.same_kind_refractory", "2.0e-3"),
        ("guards.near_tie_tolerance", "2.0e-9"),
        ("guards.maximum_consecutive_gap_clamps", "5"),
        ("scripted.action_target_fraction", "0.826"),
    ],
)
def test_phase2_config_rejects_each_frozen_numerical_mutation(
    dotted_path: str, mutated_value: str
) -> None:
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(
            Phase2Config,
            [
                Path("configs/base.yaml"),
                Path("configs/data/primary.yaml"),
                Path("configs/model/event_flow.yaml"),
            ],
            set_overrides=[f"event_flow.{dotted_path}={mutated_value}"],
        )


def test_event_flow_config_itself_requires_the_exact_phase2_event_cap() -> None:
    values = resolve_event_flow().config.event_flow.model_dump()
    values["max_internal_events"] = 63

    with pytest.raises(ValueError):
        EventFlowConfig.model_validate(values)


def test_guard_threshold_and_active_margin_cannot_drift_together() -> None:
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(
            Phase2Config,
            [
                Path("configs/base.yaml"),
                Path("configs/data/primary.yaml"),
                Path("configs/model/event_flow.yaml"),
            ],
            set_overrides=[
                "event_flow.guards.threshold=2.0",
                "event_flow.guards.active_margin=2.1",
            ],
        )


def test_scripted_config_itself_requires_the_exact_action_target() -> None:
    with pytest.raises(ValueError):
        ScriptedAgentConfig(action_target_fraction=0.826)


def test_phase2_config_rejects_coupled_event_ceiling_drift() -> None:
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(
            Phase2Config,
            [
                Path("configs/base.yaml"),
                Path("configs/data/primary.yaml"),
                Path("configs/model/event_flow.yaml"),
            ],
            set_overrides=[
                "limits.max_eventflow_events=63",
                "event_flow.max_internal_events=63",
            ],
        )


@pytest.mark.parametrize(
    ("overlay_text", "error"),
    [
        ("event_flow:\n  unknown_key: 1\n", "Extra inputs are not permitted"),
        ("event_flow:\n  dimensions:\n    z_fast: true\n", "exact int"),
        ("event_flow:\n  flow:\n    rate_min: '0.00001'\n", "exact float"),
        ("runtime:\n  dtype: float64\n", "Input should be 'float32'"),
        (
            "event_flow:\n  dimensions:\n    guard_accumulators: 4\n",
            "guard_accumulators",
        ),
        (
            "event_flow:\n  max_internal_events: 63\n",
            "max_internal_events",
        ),
        ("data:\n  primary_memory_capacity: 65\n", "Input should be 64"),
        (
            "event_flow:\n  scripted:\n    action_target_fraction: 0.826\n",
            "action_target_fraction",
        ),
    ],
)
def test_phase2_config_rejects_contract_drift(
    tmp_path: Path, overlay_text: str, error: str
) -> None:
    overlay = write_overlay(tmp_path, overlay_text)

    with pytest.raises(ConfigurationError, match="configuration validation failed") as raised:
        resolve_event_flow(overlay)

    assert error in raised.value.context["details"]
