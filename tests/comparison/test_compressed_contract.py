"""Compressed intervention is a distinct public-state operation."""


def test_compressed_intervention_entrypoint_exists() -> None:
    """Removing the intervention prevents the checkpoint-only comparison."""
    from silent_cascade.eventflow.compressed import run_compressed_from_activation

    assert callable(run_compressed_from_activation)
