"""The evaluator exposes the approved intact bridge."""


def test_intact_bridge_is_callable() -> None:
    """Removing the bridge prevents an authenticated checkpoint comparison."""
    from silent_cascade.eval.comparison_runner import run_intact_episode

    assert callable(run_intact_episode)
