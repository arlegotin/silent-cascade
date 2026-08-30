"""Reproducible global RNG capture for Python, NumPy, Torch CPU, and MPS."""

import copy
import random
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
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if mps_rng_state_supported() and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)


def snapshot_global_rng() -> RngSnapshot:
    mps_state = torch.mps.get_rng_state().clone() if mps_rng_state_supported() else None
    return RngSnapshot(
        python_state=copy.deepcopy(random.getstate()),
        numpy_state=copy.deepcopy(np.random.get_state()),
        torch_cpu_state=torch.get_rng_state().clone(),
        torch_mps_state=mps_state,
    )


def restore_global_rng(snapshot: RngSnapshot) -> None:
    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_state)
    torch.set_rng_state(snapshot.torch_cpu_state.clone())
    if snapshot.torch_mps_state is not None:
        if not mps_rng_state_supported():
            raise DoctorError("MPS RNG state cannot be restored on this runtime")
        torch.mps.set_rng_state(snapshot.torch_mps_state.clone())


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
