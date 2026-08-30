import subprocess
import tarfile
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
