"""Offline environment diagnostics for Silent Cascade."""

import importlib.metadata
import importlib.util
import math
import os
import platform
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import psutil
import torch
from pydantic import field_validator

from silent_cascade.config import ProjectConfig, resolve_config
from silent_cascade.rng import (
    RngRoundTripReport,
    mps_rng_state_supported,
    verify_rng_round_trip,
)
from silent_cascade.validation import StrictModel


class MpsReport(StrictModel):
    built: bool
    available: bool
    flow_ok: bool | None
    guard_ok: bool | None
    rng_state_supported: bool
    rng_round_trip_checked: bool
    rng_round_trip_exact: bool | None


class WritablePathReport(StrictModel):
    path: str
    writable: bool
    error: str | None = None


class NumericSmokeReport(StrictModel):
    device: Literal["cpu", "mps"]
    flow_ok: bool
    guard_ok: bool


class HostReport(StrictModel):
    system: str
    release: str
    machine: str
    processor: str
    total_memory_bytes: int


class DoctorReport(StrictModel):
    ok: bool
    python_version: str
    python_supported: bool
    package_versions: dict[str, str]
    torch_cpu_ok: bool
    cpu_count_logical: int | None
    cpu_count_physical: int | None
    host: HostReport
    mps_fallback_enabled: bool
    mps: MpsReport
    writable_paths: tuple[WritablePathReport, ...]
    rng: RngRoundTripReport
    numeric: NumericSmokeReport
    config_sha256: str
    primary_foundation_model_calls: Literal[0]
    qwen_extra_requested: bool
    qwen_extra_installed: bool | None

    @field_validator("primary_foundation_model_calls", mode="before")
    @classmethod
    def require_exact_integer_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("value must be an exact int")
        return value


def _numeric_smoke(device: Literal["cpu", "mps"]) -> NumericSmokeReport:
    state = torch.tensor([0.0], dtype=torch.float32, device=device)
    target = torch.tensor([0.5], dtype=torch.float32, device=device)
    rate = torch.tensor([2.0], dtype=torch.float32, device=device)
    dt = torch.tensor(0.25, dtype=torch.float32, device=device)
    weight = -torch.expm1(-rate * dt)
    flowed = state + weight * (target - state)
    expected = 0.5 * (1.0 - math.exp(-0.5))
    flow_ok = math.isclose(float(flowed.cpu().item()), expected, rel_tol=1e-6)

    accumulator = torch.tensor([0.0], dtype=torch.float32, device=device)
    asymptote = torch.tensor([1.5], dtype=torch.float32, device=device)
    guard_rate = torch.tensor([2.0], dtype=torch.float32, device=device)
    delta = torch.log1p((1.0 - accumulator) / (asymptote - 1.0)) / guard_rate
    crossed = asymptote + (accumulator - asymptote) * torch.exp(-guard_rate * delta)
    guard_ok = math.isclose(float(crossed.cpu().item()), 1.0, rel_tol=0.0, abs_tol=1e-6)
    return NumericSmokeReport(device=device, flow_ok=flow_ok, guard_ok=guard_ok)


_CORE_PACKAGES = (
    "torch",
    "numpy",
    "pydantic",
    "pyyaml",
    "typer",
    "rich",
    "safetensors",
    "scipy",
    "matplotlib",
    "psutil",
)
_QWEN_PACKAGES = ("mlx", "mlx_vlm", "huggingface_hub")


def _truthy_environment(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _writable_path_report(path: Path) -> WritablePathReport:
    try:
        path.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".doctor-", dir=path)
        os.close(descriptor)
        Path(temporary).unlink()
        return WritablePathReport(path=str(path), writable=True)
    except OSError as error:
        return WritablePathReport(path=str(path), writable=False, error=str(error))


def _qwen_installed() -> bool:
    discoveries = [importlib.util.find_spec(name) for name in _QWEN_PACKAGES]
    return all(discovery is not None for discovery in discoveries)


def run_doctor(
    *,
    config_path: Path | None,
    writable_paths: Sequence[Path],
    check_qwen: bool = False,
) -> DoctorReport:
    resolved = (
        resolve_config(ProjectConfig, [config_path])
        if config_path is not None
        else resolve_config(ProjectConfig, [])
    )
    python_supported = platform.python_version_tuple()[:2] == ("3", "12")
    cpu_probe = torch.tensor([1.0], dtype=torch.float32) + 1.0
    torch_cpu_ok = cpu_probe.dtype == torch.float32 and cpu_probe.item() == 2.0
    fallback_enabled = _truthy_environment("PYTORCH_ENABLE_MPS_FALLBACK")

    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available()
    mps_numeric: NumericSmokeReport | None = None
    if mps_available:
        mps_numeric = _numeric_smoke("mps")
        torch.mps.synchronize()
    mps_rng_supported = mps_rng_state_supported()

    rng_report = verify_rng_round_trip(include_mps=mps_available)
    numeric_report = _numeric_smoke("cpu")
    path_reports = tuple(_writable_path_report(path) for path in writable_paths)
    qwen_installed = _qwen_installed() if check_qwen else None
    package_versions = {name: importlib.metadata.version(name) for name in _CORE_PACKAGES}

    required_ok = all(
        (
            python_supported,
            torch_cpu_ok,
            not fallback_enabled,
            numeric_report.flow_ok,
            numeric_report.guard_ok,
            rng_report.python_ok,
            rng_report.numpy_ok,
            rng_report.torch_cpu_ok,
            all(item.writable for item in path_reports),
            (not mps_available)
            or (mps_numeric is not None and mps_numeric.flow_ok and mps_numeric.guard_ok),
            (not check_qwen) or bool(qwen_installed),
        )
    )
    return DoctorReport(
        ok=required_ok,
        python_version=platform.python_version(),
        python_supported=python_supported,
        package_versions=package_versions,
        torch_cpu_ok=torch_cpu_ok,
        cpu_count_logical=psutil.cpu_count(logical=True),
        cpu_count_physical=psutil.cpu_count(logical=False),
        host=HostReport(
            system=platform.system(),
            release=platform.release(),
            machine=platform.machine(),
            processor=platform.processor(),
            total_memory_bytes=psutil.virtual_memory().total,
        ),
        mps_fallback_enabled=fallback_enabled,
        mps=MpsReport(
            built=mps_built,
            available=mps_available,
            flow_ok=mps_numeric.flow_ok if mps_numeric is not None else None,
            guard_ok=mps_numeric.guard_ok if mps_numeric is not None else None,
            rng_state_supported=mps_rng_supported,
            rng_round_trip_checked=rng_report.torch_mps_checked,
            rng_round_trip_exact=rng_report.torch_mps_ok,
        ),
        writable_paths=path_reports,
        rng=rng_report,
        numeric=numeric_report,
        config_sha256=resolved.sha256,
        primary_foundation_model_calls=resolved.config.runtime.primary_foundation_model_calls,
        qwen_extra_requested=check_qwen,
        qwen_extra_installed=qwen_installed,
    )
