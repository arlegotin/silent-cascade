import subprocess
import sys


def imported_modules(module_name: str) -> set[str]:
    program = f"""
import sys
import {module_name}
print('\\n'.join(sorted(sys.modules)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    return set(completed.stdout.splitlines())


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


def test_generator_and_invariants_do_not_import_oracle() -> None:
    assert "silent_cascade.env.oracle" not in imported_modules("silent_cascade.env.generator")
    assert "silent_cascade.env.oracle" not in imported_modules("silent_cascade.env.invariants")
    assert "silent_cascade.env.generator" not in imported_modules("silent_cascade.env.invariants")


def test_oracle_does_not_import_generator_or_invariants() -> None:
    modules = imported_modules("silent_cascade.env.oracle")
    assert "silent_cascade.env.generator" not in modules
    assert "silent_cascade.env.invariants" not in modules
