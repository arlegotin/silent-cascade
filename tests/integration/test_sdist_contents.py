import subprocess
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
