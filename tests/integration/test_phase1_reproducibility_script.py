"""The production reproducibility CLI exposes no mutable execution matrix."""

import subprocess
import sys


def test_production_script_help_locks_the_execution_matrix() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_phase1_reproducibility.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--sample-size" not in result.stdout
    assert "--chunk-size" not in result.stdout
    assert "--python-hash-seed" not in result.stdout
