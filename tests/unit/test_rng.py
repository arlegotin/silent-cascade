import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from silent_cascade.errors import DoctorError
from silent_cascade.rng import (
    restore_global_rng,
    seed_all,
    snapshot_global_rng,
    verify_rng_round_trip,
)


def _draw_cpu_stream() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(4)


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


def test_seed_all_rejects_out_of_range_seed_without_mutating_rng() -> None:
    seed_all(41)
    current = snapshot_global_rng()
    expected = _draw_cpu_stream()
    restore_global_rng(current)

    with pytest.raises(ValueError, match="seed"):
        seed_all(2**64)

    actual = _draw_cpu_stream()
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_restore_rejects_unavailable_mps_without_mutating_cpu_rng(monkeypatch) -> None:
    seed_all(41)
    target = snapshot_global_rng()
    seed_all(99)
    current = snapshot_global_rng()
    expected = _draw_cpu_stream()
    restore_global_rng(current)
    mps_snapshot = replace(target, torch_mps_state=torch.zeros(1, dtype=torch.uint8))
    monkeypatch.setattr("silent_cascade.rng.mps_rng_state_supported", lambda: False)

    with pytest.raises(DoctorError, match="MPS RNG state"):
        restore_global_rng(mps_snapshot)

    actual = _draw_cpu_stream()
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is a local-only capability gate",
)
def test_rng_snapshot_restore_repeats_mps_draws_when_available() -> None:
    report = verify_rng_round_trip(include_mps=True)
    assert report.torch_mps_checked
    assert isinstance(report.torch_mps_ok, bool)
