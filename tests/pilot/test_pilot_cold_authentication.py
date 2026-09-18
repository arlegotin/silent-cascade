"""Real completed smoke fits must remain authentic through one-file cold leases."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_fit_guard_rejects_oversized_publisher_before_any_output(tmp_path):
    from silent_cascade.train import pilot_data

    from .cold_authentication_fixture import MIB, FitGuard

    run = tmp_path / "run"
    guard = FitGuard(tmp_path, tmp_path / "cache")
    with guard.installed(run), pytest.raises(AssertionError, match="single-file"):
        # A virtual payload supplies its length; it must never reach the writer.
        class Oversized:
            def __len__(self):
                return 16 * MIB + 1

        pilot_data._publish_pilot_bytes(run / "oversized", Oversized())
    assert not run.exists()


def test_fit_guard_counts_existing_and_pending_checkpoint_before_publication(tmp_path, monkeypatch):
    from silent_cascade.train import pilot_data

    from . import cold_authentication_fixture as fixture

    run = tmp_path / "run"
    run.mkdir()
    old = run / "checkpoint-index.json"
    pending = run / "checkpoint.pending"
    old.write_bytes(b"o" * 4096)
    pending.write_bytes(b"p" * 4096)
    guard = fixture.FitGuard(tmp_path, tmp_path / "cache")
    current = fixture.allocation(tmp_path)[0]
    monkeypatch.setattr(fixture, "FIT_BYTES", current + 4096)
    with guard.installed(run), pytest.raises(AssertionError, match="before write"):
        pilot_data._publish_pilot_bytes(run / "checkpoint-final", b"f" * 4096)
    assert old.read_bytes() == b"o" * 4096
    assert pending.read_bytes() == b"p" * 4096
    assert not (run / "checkpoint-final").exists()
    assert sorted(path.name for path in run.iterdir()) == [
        "checkpoint-index.json",
        "checkpoint.pending",
    ]


def test_fit_guard_stops_pending_row_append_before_handle_write(tmp_path, monkeypatch):
    from . import cold_authentication_fixture as fixture

    run = tmp_path / "run"
    run.mkdir()
    path = run / ".rows.pending.jsonl"
    guard = fixture.FitGuard(tmp_path, tmp_path / "cache")
    with guard.installed(run), path.open("xb") as stream:
        stream.write(b"original\n")
        monkeypatch.setattr(fixture, "FIT_BYTES", fixture.allocation(tmp_path)[0])
        with pytest.raises(AssertionError, match="before write"):
            stream.write(b"rejected\n")
    assert path.read_bytes() == b"original\n"


def test_fit_guard_rejection_remains_blocked_after_caller_catches_error(tmp_path, monkeypatch):
    from silent_cascade.train import pilot_data

    from . import cold_authentication_fixture as fixture

    guard = fixture.FitGuard(tmp_path, tmp_path / "cache")
    with guard.installed(tmp_path / "run"):
        monkeypatch.setattr(fixture, "FIT_BYTES", 0)
        with pytest.raises(AssertionError, match="before write"):
            pilot_data._publish_pilot_bytes(tmp_path / "run/first", b"too much")
        monkeypatch.setattr(fixture, "FIT_BYTES", 64 * fixture.MIB)
        with pytest.raises(AssertionError, match="remains blocked"):
            pilot_data._publish_pilot_bytes(tmp_path / "run/smaller", b"x")
    assert guard.blocked is not None
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("limit", ["FIT_NAMES", "FIT_DIRECTORIES"])
def test_fit_guard_refuses_name_or_directory_growth(tmp_path, monkeypatch, limit):
    from silent_cascade.train import pilot_data

    from . import cold_authentication_fixture as fixture

    guard = fixture.FitGuard(tmp_path, tmp_path / "cache")
    monkeypatch.setattr(fixture, limit, 1)
    with guard.installed(tmp_path / "run"), pytest.raises(AssertionError, match="bound"):
        pilot_data._publish_pilot_bytes(tmp_path / "run/new/artifact", b"small")
    assert not (tmp_path / "run").exists()


def test_completed_fit_authentication_and_reuse_through_cold_inputs():
    # The controller admits these roots before this test creates any output.
    if "SC_COLD_SCRATCH" not in os.environ:
        pytest.skip("requires explicitly admitted retained cold-authentication roots")
    from .cold_authentication_fixture import closed_environment, create_checkout

    scratch = Path(os.environ["SC_COLD_SCRATCH"]).absolute()
    spool = Path(os.environ["SC_COLD_SPOOL"]).absolute()
    cache = Path(os.environ["SC_COLD_CACHE"]).absolute()
    reuse = os.environ.get("SC_COLD_REUSE") == "1"
    root = scratch / "repo" if reuse else create_checkout(scratch)
    environment = closed_environment(scratch) | {"PYTHONPATH": str(root / "src")}
    helper = Path(__file__).with_name("cold_authentication_fixture.py")
    completed = subprocess.run(
        [sys.executable, "-B", str(helper), str(spool), str(cache), "reuse" if reuse else "fit"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    # Subprocess output is fixed small diagnostics; no tensors or trace payloads.
    assert len(completed.stdout.encode()) + len(completed.stderr.encode()) <= 256 * 1024
    print(completed.stdout, end="")
    assert completed.returncode == 0, completed.stdout + completed.stderr
