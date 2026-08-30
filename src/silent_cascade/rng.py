"""Reproducible global RNG capture for Python, NumPy, Torch CPU, and MPS."""

import copy
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from silent_cascade.errors import DoctorError
from silent_cascade.validation import StrictModel


@dataclass(frozen=True, slots=True)
class RngSnapshot:
    python_state: object
    numpy_state: tuple[Any, ...]
    torch_cpu_state: torch.Tensor
    torch_mps_state: torch.Tensor | None


class RngRoundTripReport(StrictModel):
    python_ok: bool
    numpy_ok: bool
    torch_cpu_ok: bool
    torch_mps_checked: bool
    torch_mps_ok: bool | None


def mps_rng_state_supported() -> bool:
    return bool(
        torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and hasattr(torch.mps, "set_rng_state")
    )


def seed_all(seed: int) -> None:
    if type(seed) is not int:
        raise TypeError("seed must be an int")
    if not 0 <= seed <= 2**64 - 1:
        raise ValueError("seed must be in range 0..2**64-1")

    def apply_seed() -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        # torch.manual_seed also seeds the MPS default generator when present.
        torch.manual_seed(seed)

    _apply_transactionally("seed global RNGs", apply_seed)


def snapshot_global_rng() -> RngSnapshot:
    mps_state = torch.mps.get_rng_state().clone() if mps_rng_state_supported() else None
    return RngSnapshot(
        python_state=copy.deepcopy(random.getstate()),
        numpy_state=copy.deepcopy(np.random.get_state()),
        torch_cpu_state=torch.get_rng_state().clone(),
        torch_mps_state=mps_state,
    )


def restore_global_rng(snapshot: RngSnapshot) -> None:
    _prevalidate_snapshot(snapshot)
    _apply_transactionally("restore global RNGs", lambda: _apply_snapshot(snapshot))


def _prevalidate_snapshot(snapshot: RngSnapshot) -> None:
    if snapshot.torch_mps_state is not None and not mps_rng_state_supported():
        raise DoctorError("MPS RNG state cannot be restored on this runtime")

    isolated_python = random.Random()
    isolated_python.setstate(copy.deepcopy(snapshot.python_state))

    isolated_numpy = np.random.RandomState()
    isolated_numpy.set_state(copy.deepcopy(snapshot.numpy_state))

    isolated_torch_cpu = torch.Generator(device="cpu")
    isolated_torch_cpu.set_state(snapshot.torch_cpu_state.clone())


def _apply_snapshot(snapshot: RngSnapshot) -> None:
    random.setstate(copy.deepcopy(snapshot.python_state))
    np.random.set_state(copy.deepcopy(snapshot.numpy_state))
    torch.set_rng_state(snapshot.torch_cpu_state.clone())
    if snapshot.torch_mps_state is not None:
        torch.mps.set_rng_state(snapshot.torch_mps_state.clone())


def _apply_transactionally(operation: str, mutation: Callable[[], None]) -> None:
    caller_snapshot = snapshot_global_rng()
    try:
        mutation()
    except Exception as original_error:
        rollback_errors = _rollback_global_rng(caller_snapshot)
        if rollback_errors:
            rollback_detail = "; ".join(rollback_errors)
            raise DoctorError(
                f"{operation} failed and RNG rollback also failed: "
                f"{type(original_error).__name__}: {original_error}; {rollback_detail}",
                context={
                    "operation": operation,
                    "original_error_type": type(original_error).__name__,
                    "original_error": str(original_error),
                    "rollback_errors": rollback_errors,
                },
            ) from original_error
        raise


def _rollback_global_rng(snapshot: RngSnapshot) -> list[str]:
    """Best-effort rollback that never re-enters the public restore path."""

    rollback_errors: list[str] = []
    for backend, restore in (
        ("Python", lambda: random.setstate(copy.deepcopy(snapshot.python_state))),
        ("NumPy", lambda: np.random.set_state(copy.deepcopy(snapshot.numpy_state))),
        ("Torch CPU", lambda: torch.set_rng_state(snapshot.torch_cpu_state.clone())),
    ):
        try:
            restore()
        except Exception as error:
            rollback_errors.append(f"{backend}: {type(error).__name__}: {error}")

    if snapshot.torch_mps_state is not None:
        try:
            if not mps_rng_state_supported():
                raise DoctorError("MPS RNG state cannot be restored on this runtime")
            torch.mps.set_rng_state(snapshot.torch_mps_state.clone())
        except Exception as error:
            rollback_errors.append(f"MPS: {type(error).__name__}: {error}")

    return rollback_errors


def verify_rng_round_trip(*, include_mps: bool) -> RngRoundTripReport:
    outer = snapshot_global_rng()
    try:
        seed_all(0x5A17)
        checkpoint = snapshot_global_rng()
        first_python = random.random()
        first_numpy = float(np.random.random())
        first_cpu = torch.rand(8)
        first_mps = (
            torch.rand(8, device="mps").cpu() if include_mps and mps_rng_state_supported() else None
        )

        restore_global_rng(checkpoint)
        second_python = random.random()
        second_numpy = float(np.random.random())
        second_cpu = torch.rand(8)
        second_mps = (
            torch.rand(8, device="mps").cpu() if include_mps and mps_rng_state_supported() else None
        )
        if include_mps and torch.backends.mps.is_available():
            torch.mps.synchronize()

        mps_checked = include_mps and mps_rng_state_supported()
        mps_ok = (
            torch.equal(first_mps, second_mps)
            if first_mps is not None and second_mps is not None
            else None
        )
        return RngRoundTripReport(
            python_ok=first_python == second_python,
            numpy_ok=first_numpy == second_numpy,
            torch_cpu_ok=torch.equal(first_cpu, second_cpu),
            torch_mps_checked=mps_checked,
            torch_mps_ok=mps_ok,
        )
    finally:
        restore_global_rng(outer)
