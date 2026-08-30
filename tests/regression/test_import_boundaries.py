import subprocess
import sys


def test_core_imports_do_not_load_optional_qwen_modules() -> None:
    program = r"""
import importlib.abc
import sys

blocked = {"mlx", "mlx_vlm", "huggingface_hub"}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError(f"optional import attempted: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
import silent_cascade
import silent_cascade.config
import silent_cascade.rng
import silent_cascade.io
import silent_cascade.doctor
import silent_cascade.cli
assert not blocked.intersection(sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
