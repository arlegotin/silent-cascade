"""Fixed-64 debug overfit engineering regression, never held-out evidence."""

from pathlib import Path

from silent_cascade.config import resolve_config
from silent_cascade.train.config import Phase3Config


def test_real_fixed_64_overfit():
    from silent_cascade.train.trainer import run_tiny_overfit

    config = resolve_config(
        Phase3Config,
        tuple(
            Path(p)
            for p in (
                "configs/base.yaml",
                "configs/data/primary.yaml",
                "configs/model/event_flow.yaml",
                "configs/model/neural_components.yaml",
                "configs/train/smoke_content_v2.yaml",
            )
        ),
    )
    result = run_tiny_overfit(config.config, device="cpu")
    print(result)
    assert result.episodes == 64
    assert result.steps <= 1000
    assert result.complete_chain_accuracy >= 0.99
    assert result.evidence_kind == "overfit_engineering_regression"
