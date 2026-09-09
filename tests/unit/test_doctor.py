import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from silent_cascade import doctor
from silent_cascade.doctor import DoctorReport, run_doctor


def test_run_doctor_reports_cpu_rng_numeric_and_mps_capability(tmp_path: Path) -> None:
    report = run_doctor(
        config_path=Path("configs/base.yaml"),
        writable_paths=[tmp_path / "runs", tmp_path / "reports"],
    )
    assert report.python_supported
    assert report.torch_cpu_ok
    assert report.numeric.flow_ok
    assert report.numeric.guard_ok
    assert report.rng.python_ok
    assert report.rng.numpy_ok
    assert report.rng.torch_cpu_ok
    assert report.primary_foundation_model_calls == 0
    assert report.mps.built == torch.backends.mps.is_built()
    assert report.mps.available == torch.backends.mps.is_available()
    if report.mps.available:
        assert report.mps.flow_ok
        assert report.mps.guard_ok
        assert report.mps.rng_round_trip_checked == report.rng.torch_mps_checked
        assert report.mps.rng_round_trip_exact == report.rng.torch_mps_ok
    assert report.host.total_memory_bytes > 0
    assert report.host.system
    assert report.host.machine
    assert all(item.writable for item in report.writable_paths)
    assert report.ok


def test_doctor_fails_when_silent_mps_fallback_is_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    report = run_doctor(
        config_path=Path("configs/base.yaml"),
        writable_paths=[tmp_path],
    )
    assert report.mps_fallback_enabled
    assert not report.ok


def test_qwen_check_discovers_packages_without_importing_them(tmp_path: Path, monkeypatch) -> None:
    requested: list[str] = []
    real_find_spec = importlib.util.find_spec

    def recording_find_spec(name: str):
        requested.append(name)
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", recording_find_spec)
    report = run_doctor(
        config_path=Path("configs/base.yaml"),
        writable_paths=[tmp_path],
        check_qwen=True,
    )
    assert requested == ["mlx", "mlx_vlm", "huggingface_hub"]
    assert not {"mlx", "mlx_vlm", "huggingface_hub"}.intersection(sys.modules)
    assert report.qwen_extra_requested


def test_doctor_report_rejects_boolean_foundation_call_count() -> None:
    with pytest.raises(ValueError):
        DoctorReport.model_validate(
            {
                "ok": True,
                "python_version": "3.12.0",
                "python_supported": True,
                "package_versions": {},
                "torch_cpu_ok": True,
                "cpu_count_logical": 1,
                "cpu_count_physical": 1,
                "host": {
                    "system": "test",
                    "release": "test",
                    "machine": "test",
                    "processor": "test",
                    "total_memory_bytes": 1,
                },
                "mps_fallback_enabled": False,
                "mps": {
                    "built": False,
                    "available": False,
                    "flow_ok": None,
                    "guard_ok": None,
                    "rng_state_supported": False,
                    "rng_round_trip_checked": False,
                    "rng_round_trip_exact": None,
                },
                "writable_paths": (),
                "rng": {
                    "python_ok": True,
                    "numpy_ok": True,
                    "torch_cpu_ok": True,
                    "torch_mps_checked": False,
                    "torch_mps_ok": None,
                },
                "numeric": {"device": "cpu", "flow_ok": True, "guard_ok": True},
                "config_sha256": "0" * 64,
                "primary_foundation_model_calls": False,
                "qwen_extra_requested": False,
                "qwen_extra_installed": None,
            }
        )


def test_numeric_smoke_uses_production_flow_and_guard_operations(monkeypatch) -> None:
    calls: list[str] = []
    real_state_at = doctor.flow.state_at
    real_crossing_offsets = doctor.guards.crossing_offsets_tensor

    def state_at(*args, **kwargs):
        calls.append("flow")
        return real_state_at(*args, **kwargs)

    def crossing_offsets(*args, **kwargs):
        calls.append("guards")
        return real_crossing_offsets(*args, **kwargs)

    monkeypatch.setattr(doctor.flow, "state_at", state_at)
    monkeypatch.setattr(doctor.guards, "crossing_offsets_tensor", crossing_offsets)

    report = doctor._numeric_smoke("cpu")

    assert report.flow_ok and report.guard_ok
    assert calls == ["flow", "guards"]


def test_mps_numeric_smoke_uses_float32_production_tensors_when_available(
    tmp_path: Path, monkeypatch
) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("native MPS is unavailable")
    observed_devices: list[str] = []
    synchronized: list[bool] = []
    real_flow = doctor.flow.state_at
    real_guards = doctor.guards.crossing_offsets_tensor

    def state_at(runtime, target_time):
        observed_devices.append(runtime.core.continuous.device.type)
        return real_flow(runtime, target_time)

    def crossing_offsets(accumulators, asymptotes, rates):
        observed_devices.extend(
            (accumulators.device.type, asymptotes.device.type, rates.device.type)
        )
        return real_guards(accumulators, asymptotes, rates)

    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(doctor.flow, "state_at", state_at)
    monkeypatch.setattr(doctor.guards, "crossing_offsets_tensor", crossing_offsets)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: synchronized.append(True))

    cpu = doctor._numeric_smoke("cpu")
    report = run_doctor(config_path=Path("configs/base.yaml"), writable_paths=[tmp_path])

    assert cpu.flow_ok and cpu.guard_ok
    assert report.mps.flow_ok and report.mps.guard_ok
    assert observed_devices.count("mps") == 4
    assert synchronized
    assert not report.mps_fallback_enabled
