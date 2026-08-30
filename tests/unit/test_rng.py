import random

import numpy as np
import pytest
import torch

from silent_cascade.rng import (
    restore_global_rng,
    seed_all,
    snapshot_global_rng,
    verify_rng_round_trip,
)


def test_rng_snapshot_restore_repeats_python_numpy_and_torch_cpu_draws() -> None:
    outer = snapshot_global_rng()
    try:
        seed_all(1729)
        checkpoint = snapshot_global_rng()
        first = (random.random(), np.random.random(), torch.rand(4))
        restore_global_rng(checkpoint)
        second = (random.random(), np.random.random(), torch.rand(4))
        assert first[0] == second[0]
        assert first[1] == second[1]
        assert torch.equal(first[2], second[2])
    finally:
        restore_global_rng(outer)


def test_rng_round_trip_reports_mps_unchecked_when_not_requested() -> None:
    report = verify_rng_round_trip(include_mps=False)
    assert report.python_ok
    assert report.numpy_ok
    assert report.torch_cpu_ok
    assert not report.torch_mps_checked
    assert report.torch_mps_ok is None


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is a local-only capability gate",
)
def test_rng_snapshot_restore_repeats_mps_draws_when_available() -> None:
    report = verify_rng_round_trip(include_mps=True)
    assert report.torch_mps_checked
    assert isinstance(report.torch_mps_ok, bool)
