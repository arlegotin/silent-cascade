from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.errors import ConfigurationError
from silent_cascade.eventflow.config import Phase2Config


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
    ("overlay_text", "error"),
    [
        ("event_flow:\n  unknown_key: 1\n", "Extra inputs are not permitted"),
        ("event_flow:\n  dimensions:\n    z_fast: true\n", "exact int"),
        ("event_flow:\n  flow:\n    rate_min: '0.00001'\n", "exact float"),
        ("runtime:\n  dtype: float64\n", "Input should be 'float32'"),
        (
            "event_flow:\n  dimensions:\n    guard_accumulators: 4\n",
            "guard_accumulators must equal 3",
        ),
        (
            "event_flow:\n  max_internal_events: 63\n",
            "EventFlow event ceilings must match",
        ),
        ("data:\n  primary_memory_capacity: 65\n", "Input should be 64"),
        (
            "event_flow:\n  scripted:\n    action_target_fraction: 0.826\n",
            "scripted target must match the standing OFD objective",
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
