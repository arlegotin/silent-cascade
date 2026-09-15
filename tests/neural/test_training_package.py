"""A locally built wheel runs both commands from an unrelated directory."""

import os
import subprocess
import sys
import zipfile

import pytest

from .test_training_cli import CONFIGS, ROOT
from .test_training_imports import OFFLINE_SCRIPT


@pytest.mark.parametrize("profile", ["smoke", "smoke_content_v2"])
def test_installed_wheel_executes_training_offline_outside_checkout(tmp_path, profile):
    build = subprocess.run(
        ["uv", "build", "--wheel", "--offline", "--out-dir", str(tmp_path / "dist")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert build.returncode == 0, build.stderr
    stage = tmp_path / "installed"
    with zipfile.ZipFile(next((tmp_path / "dist").glob("*.whl"))) as wheel:
        wheel.extractall(stage)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    # -S disables the checkout's editable .pth; dependencies are provided explicitly.
    dependencies = [p for p in sys.path if p.endswith("site-packages")]
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(stage), *dependencies]),
        "OMP_NUM_THREADS": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            OFFLINE_SCRIPT,
            str(stage),
            str(unrelated),
            *map(str, (*CONFIGS[:-1], ROOT / f"configs/train/{profile}.yaml")),
        ],
        cwd=unrelated,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    help_result = subprocess.run(
        [sys.executable, "-S", "-m", "silent_cascade.train", "--help"],
        cwd=unrelated,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "evaluate-components" in help_result.stdout
