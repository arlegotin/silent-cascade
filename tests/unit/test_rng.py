import random
from dataclasses import replace

import numpy as np
import pytest
import torch

import silent_cascade.rng as rng
from silent_cascade.errors import DoctorError
from silent_cascade.rng import (
    RngSnapshot,
    mps_rng_state_supported,
    restore_global_rng,
    seed_all,
    snapshot_global_rng,
    verify_rng_round_trip,
)


def _draw_cpu_stream() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(4)


def _assert_global_rng_matches(snapshot: RngSnapshot) -> None:
    python_state = random.getstate()
    assert python_state == snapshot.python_state

    numpy_state = np.random.get_state()
    assert numpy_state[0] == snapshot.numpy_state[0]
    assert np.array_equal(numpy_state[1], snapshot.numpy_state[1])
    assert numpy_state[2:] == snapshot.numpy_state[2:]

    assert torch.equal(torch.get_rng_state(), snapshot.torch_cpu_state)
    if snapshot.torch_mps_state is not None:
        assert torch.equal(torch.mps.get_rng_state(), snapshot.torch_mps_state)


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


@pytest.mark.parametrize(
    ("invalid_seed", "exception_type"),
    [(-1, ValueError), (True, TypeError), (2**64, ValueError)],
)
def test_seed_all_rejects_invalid_seed_without_mutating_rng(
    invalid_seed: object, exception_type: type[Exception]
) -> None:
    outer = snapshot_global_rng()
    try:
        seed_all(41)
        current = snapshot_global_rng()
        expected = _draw_cpu_stream()
        restore_global_rng(current)
        mps_before = torch.mps.get_rng_state().clone() if mps_rng_state_supported() else None

        with pytest.raises(exception_type, match="seed"):
            seed_all(invalid_seed)

        actual = _draw_cpu_stream()
        assert actual[0] == expected[0]
        assert actual[1] == expected[1]
        assert torch.equal(actual[2], expected[2])
        if mps_before is not None:
            assert torch.equal(torch.mps.get_rng_state(), mps_before)
    finally:
        restore_global_rng(outer)


@pytest.mark.parametrize("seed", [0, 2**64 - 1])
def test_seed_all_accepts_exact_endpoint_seeds(seed: int) -> None:
    outer = snapshot_global_rng()
    try:
        seed_all(seed)
    finally:
        restore_global_rng(outer)


def test_restore_rejects_unavailable_mps_without_mutating_cpu_rng(monkeypatch) -> None:
    outer = snapshot_global_rng()
    try:
        seed_all(41)
        target = snapshot_global_rng()
        seed_all(99)
        current = snapshot_global_rng()
        expected = _draw_cpu_stream()
        restore_global_rng(current)
        mps_snapshot = replace(target, torch_mps_state=torch.zeros(1, dtype=torch.uint8))
        with monkeypatch.context() as patcher:
            patcher.setattr("silent_cascade.rng.mps_rng_state_supported", lambda: False)

            with pytest.raises(DoctorError, match="MPS RNG state"):
                restore_global_rng(mps_snapshot)

            actual = _draw_cpu_stream()
            assert actual[0] == expected[0]
            assert actual[1] == expected[1]
            assert torch.equal(actual[2], expected[2])
    finally:
        restore_global_rng(outer)


def test_seed_all_rolls_back_every_stream_when_torch_seed_fails(monkeypatch) -> None:
    outer = snapshot_global_rng()
    try:
        seed_all(41)
        current = snapshot_global_rng()
        original_manual_seed = torch.manual_seed

        def fail_after_seeding(seed: int) -> torch.Generator:
            original_manual_seed(seed)
            raise RuntimeError("injected torch seed failure")

        monkeypatch.setattr(rng.torch, "manual_seed", fail_after_seeding)

        with pytest.raises(RuntimeError, match="injected torch seed failure"):
            seed_all(99)

        _assert_global_rng_matches(current)
    finally:
        restore_global_rng(outer)


def test_seed_all_rolls_back_simulated_mps_stream_when_mps_seed_fails(monkeypatch) -> None:
    outer = snapshot_global_rng()
    simulated_mps_state = torch.tensor([41], dtype=torch.uint8)
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(rng, "mps_rng_state_supported", lambda: True)
            patcher.setattr(rng.torch.mps, "get_rng_state", lambda: simulated_mps_state.clone())
            patcher.setattr(
                rng.torch.mps,
                "set_rng_state",
                lambda state: simulated_mps_state.copy_(state),
            )

            def fail_after_mutating_mps(seed: int) -> None:
                simulated_mps_state.fill_(seed % 256)
                raise RuntimeError("injected MPS seed failure")

            patcher.setattr(rng.torch.mps, "manual_seed", fail_after_mutating_mps)
            current = snapshot_global_rng()

            with pytest.raises(RuntimeError, match="injected MPS seed failure"):
                seed_all(99)

            _assert_global_rng_matches(current)
    finally:
        restore_global_rng(outer)


def test_seed_all_reports_primary_and_rollback_failures(monkeypatch) -> None:
    outer = snapshot_global_rng()
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(
                rng.torch,
                "manual_seed",
                lambda seed: (_ for _ in ()).throw(RuntimeError("injected seed failure")),
            )
            patcher.setattr(
                rng.random,
                "setstate",
                lambda state: (_ for _ in ()).throw(RuntimeError("injected rollback failure")),
            )

            with pytest.raises(
                DoctorError,
                match="seed global RNGs failed and RNG rollback also failed",
            ) as exc:
                seed_all(99)

        assert exc.value.context["original_error"] == "injected seed failure"
        assert exc.value.context["rollback_errors"] == [
            "Python: RuntimeError: injected rollback failure"
        ]
    finally:
        restore_global_rng(outer)


def test_restore_rejects_malformed_numpy_state_without_mutating_any_stream() -> None:
    outer = snapshot_global_rng()
    try:
        seed_all(41)
        target = snapshot_global_rng()
        seed_all(99)
        current = snapshot_global_rng()
        malformed = replace(target, numpy_state=("not a NumPy RNG state",))

        with pytest.raises((IndexError, TypeError, ValueError)):
            restore_global_rng(malformed)

        _assert_global_rng_matches(current)
    finally:
        restore_global_rng(outer)


def test_restore_rejects_malformed_torch_cpu_state_without_mutating_any_stream() -> None:
    outer = snapshot_global_rng()
    try:
        seed_all(41)
        target = snapshot_global_rng()
        seed_all(99)
        current = snapshot_global_rng()
        malformed = replace(target, torch_cpu_state=torch.empty(0, dtype=torch.uint8))

        with pytest.raises(RuntimeError, match="RNG state"):
            restore_global_rng(malformed)

        _assert_global_rng_matches(current)
    finally:
        restore_global_rng(outer)


def test_restore_rolls_back_every_stream_when_simulated_mps_setter_fails(monkeypatch) -> None:
    outer = snapshot_global_rng()
    simulated_mps_state = torch.tensor([0], dtype=torch.uint8)
    set_calls = 0
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(rng, "mps_rng_state_supported", lambda: True)
            patcher.setattr(rng.torch.mps, "get_rng_state", lambda: simulated_mps_state.clone())

            def manual_seed(seed: int) -> None:
                simulated_mps_state.fill_(seed % 256)

            def fail_once_after_mutating(state: torch.Tensor) -> None:
                nonlocal set_calls
                set_calls += 1
                simulated_mps_state.copy_(state)
                if set_calls == 1:
                    raise RuntimeError("injected MPS setter failure")

            patcher.setattr(rng.torch.mps, "manual_seed", manual_seed)
            patcher.setattr(rng.torch.mps, "set_rng_state", fail_once_after_mutating)
            seed_all(41)
            target = snapshot_global_rng()
            seed_all(99)
            current = snapshot_global_rng()

            with pytest.raises(RuntimeError, match="injected MPS setter failure"):
                restore_global_rng(target)

            _assert_global_rng_matches(current)
    finally:
        restore_global_rng(outer)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is a local-only capability gate",
)
def test_rng_snapshot_restore_repeats_mps_draws_when_available() -> None:
    report = verify_rng_round_trip(include_mps=True)
    assert report.torch_mps_checked
    assert isinstance(report.torch_mps_ok, bool)
