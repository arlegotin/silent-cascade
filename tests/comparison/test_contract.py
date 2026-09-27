"""Consumer-facing Phase 5A identity contract."""


def test_phase5a_config_cannot_certify_a_frozen_gate() -> None:
    """An exploratory comparison must not be represented as gate evidence."""
    from silent_cascade.eval.comparison_types import ComparisonConfig

    config = ComparisonConfig()
    assert config.purpose == "exploratory_comparison"
    assert config.gate_eligible is False
