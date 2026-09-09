import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_source_distribution_excludes_internal_sdd_data(tmp_path: Path) -> None:
    sentinel = ROOT / ".superpowers" / "sdd" / "sdist-regression-sentinel.txt"
    output_dir = tmp_path / "dist"
    output_dir.mkdir()

    try:
        sentinel.write_text("must never ship\n", encoding="utf-8")
        subprocess.run(
            ["uv", "build", "--sdist", "--out-dir", str(output_dir), "--offline"],
            cwd=ROOT,
            check=True,
        )

        archives = list(output_dir.glob("*.tar.gz"))
        assert len(archives) == 1
        with tarfile.open(archives[0], mode="r:gz") as archive:
            members = archive.getnames()

        assert not any("/.superpowers/" in f"/{member}/" for member in members)
        assert not any(member.endswith("sdist-regression-sentinel.txt") for member in members)
    finally:
        sentinel.unlink(missing_ok=True)


def test_distributions_include_phase2_replay_runtime_modules(tmp_path: Path) -> None:
    output_dir = tmp_path / "dist"
    output_dir.mkdir()
    subprocess.run(
        ["uv", "build", "--out-dir", str(output_dir), "--offline"],
        cwd=ROOT,
        check=True,
    )

    with tarfile.open(next(output_dir.glob("*.tar.gz")), mode="r:gz") as archive:
        sdist_members = set(archive.getnames())
    with zipfile.ZipFile(next(output_dir.glob("*.whl"))) as archive:
        wheel_members = set(archive.namelist())
    for module in ("archive_io.py", "checkpoint.py", "replay.py", "engine.py"):
        assert any(
            member.endswith(f"silent_cascade/eventflow/{module}") for member in sdist_members
        )
        assert f"silent_cascade/eventflow/{module}" in wheel_members
    assert any(member.endswith("/README.md") for member in sdist_members)


def test_extracted_wheel_imports_replay_without_optional_modules_or_editable_source(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "dist"
    output_dir.mkdir()
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(output_dir), "--offline"],
        cwd=ROOT,
        check=True,
    )
    stage = tmp_path / "wheel-stage"
    with zipfile.ZipFile(next(output_dir.glob("*.whl"))) as archive:
        archive.extractall(stage)
    program = r"""
import importlib.abc
import os
import pathlib
import sys

blocked = {"mlx", "mlx_vlm", "huggingface_hub"}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError(f"optional import attempted: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
import silent_cascade
import silent_cascade.cli
import silent_cascade.eventflow.archive_io
import silent_cascade.eventflow.replay

assert not blocked.intersection(sys.modules)
stage = pathlib.Path(os.environ["SILENT_CASCADE_WHEEL_STAGE"]).resolve()
assert pathlib.Path(silent_cascade.__file__).resolve().is_relative_to(stage)
assert pathlib.Path(silent_cascade.cli.__file__).resolve().is_relative_to(stage)
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(stage),
        "SILENT_CASCADE_WHEEL_STAGE": str(stage),
    }
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
