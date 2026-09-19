"""Fresh, tiny real writers; no training, evaluation, replay or provider calls."""

import json
import os
import stat
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest


def run_writer_control(code, root, allowance=65536, *, installation_denial=None, before_binding=""):
    from silent_cascade.train.pilot_offline import offline_environment
    from silent_cascade.train.pilot_offline_process import (
        OfflineProcessLimits,
        capture_offline_process,
    )

    root.mkdir(exist_ok=True)
    program = (
        """
import json, os, sys
from pathlib import Path
from contextlib import contextmanager
from silent_cascade.train.pilot_offline_writes import (
    OfflineWriteAllowance, OfflineWriteDenied, install_offline_writes,
)
from silent_cascade.train import pilot_offline as boundary
root = Path(sys.argv[1])
attempts, blocked = boundary.install_offline_boundary(
    workspace_root=root,
    write_allowance=OfflineWriteAllowance(**json.loads(sys.argv[2])),
)
guard = boundary._BOUNDARY_WRITES
assert guard is boundary._BOUNDARY_WRITES
from silent_cascade.train.pilot_data import _publish_pilot_bytes
from silent_cascade.io import atomic_create_bytes, atomic_write_bytes
@contextmanager
def denied(reason=None):
    try:
        yield
    except OfflineWriteDenied as error:
        if reason is not None:
            assert str(error) == 'offline write denied: ' + reason, str(error)
    else:
        raise AssertionError('write was not denied')
    try:
        guard.assert_clear()
    except OfflineWriteDenied:
        pass
    else:
        raise AssertionError('denial did not latch')
    try:
        _publish_pilot_bytes(root / 'offline.json', b'x')
    except OfflineWriteDenied:
        pass
    else:
        raise AssertionError('latched publisher wrote')
"""
        + code
    )
    program = program.replace(
        "attempts, blocked = boundary.install_offline_boundary(",
        before_binding + "\nattempts, blocked = boundary.install_offline_boundary(",
    )
    assert len(program.encode()) <= 4096
    limits = allowance if isinstance(allowance, dict) else dict(allocated_bytes=allowance)
    captured = capture_offline_process(
        [sys.executable, "-B", "-c", program, str(root), json.dumps(limits)],
        cwd=Path(__file__).resolve().parents[2],
        env=offline_environment(root),
        limits=OfflineProcessLimits(
            stdout_bytes=2048, stderr_bytes=2048, record_bytes=4096, timeout_seconds=30
        ),
    )
    if installation_denial is not None:
        assert captured.reason == "nonzero_exit"
        assert (
            "OfflineWriteDenied: offline write denied: " + installation_denial
        ).encode() in captured.stderr
        return
    assert captured.reason is None, captured.stderr.decode(errors="replace")
    assert captured.returncode == 0
    assert captured.eof


@contextmanager
def _held_fixture_attempt(tmp_path, *, extra_spool=0, bindings=()):
    from silent_cascade.train.pilot_offline_writes import OfflineWriteAllowance

    from . import cold_authentication_fixture as fixture
    from .test_pilot_offline_process import _prepare_privacy, _privacy_ledger

    allowance = OfflineWriteAllowance(65536, file_names=8, directories=9)
    with (
        _privacy_ledger(
            tmp_path,
            logs=0,
            metadata=0,
            scratch=0,
            spool=0,
            run_category="spool",
            privacy_spool_short=0,
            privacy_spool_extra=allowance.allocated_bytes + extra_spool,
            bindings=bindings,
        ) as (budget, admission, output),
        _prepare_privacy(budget, admission, output) as custody,
    ):
        guard = fixture.FitGuard(budget.workspace / "run", budget.workspace / "cache")
        hold = allowance.allocated_bytes + sum(peak for _, peak in custody.category_peaks)
        yield fixture, guard, custody, output, allowance, hold


@contextmanager
def _replace_owned_inode(root, *, rows):
    """A bounded external actor preserves then replaces one stage-local inode."""
    replacement, displaced = root.parent / "replacement", root.parent / "displaced"
    replacement.write_bytes(b"y")
    replacement.chmod(0o600)
    stopped, changed = threading.Event(), threading.Event()
    errors = []

    def substitute():
        try:
            deadline = time.monotonic() + 10
            while not stopped.is_set() and time.monotonic() < deadline:
                if rows:
                    targets = [root / "run/eval/primary/.rows.pending.jsonl"]
                    ready = (root / "step.json").exists()
                else:
                    targets = list(root.glob(".weights.safetensors.*.tmp"))
                    ready = len(targets) == 1 and stat.S_IMODE(targets[0].stat().st_mode) == 0o400
                if ready:
                    target = targets[0]
                    before = target.stat()
                    assert before.st_size == 1
                    os.link(target, displaced)
                    os.replace(replacement, target)
                    assert target.stat().st_ino != before.st_ino
                    changed.set()
                    return
                stopped.wait(0.01)
        except BaseException as error:
            errors.append(error)

    actor = threading.Thread(target=substitute)
    actor.start()
    try:
        yield
    finally:
        stopped.set()
        actor.join(timeout=10)
        assert not actor.is_alive()
        if errors:
            raise errors[0]
        assert changed.is_set(), "bounded inode replacement did not occur"


def test_rows_substitution_is_denied_before_final_link(tmp_path):
    root = tmp_path / "child"
    root.mkdir()
    with _replace_owned_inode(root, rows=True):
        run_writer_control(
            """
import time
pending = root / 'run/eval/primary/.rows.pending.jsonl'
pending.parent.mkdir(parents=True)
with pending.open('xb') as stream:
    stream.write(b'x')
_publish_pilot_bytes(root / 'step.json', b'x')
deadline = time.monotonic() + 5
while pending.read_bytes() != b'y' and time.monotonic() < deadline:
    time.sleep(0.01)
assert pending.read_bytes() == b'y'
with denied('authority'):
    os.link(pending, pending.parent / 'rows.jsonl')
assert not (pending.parent / 'rows.jsonl').exists()
""",
            root,
        )
    assert not (root / "run/eval/primary/rows.jsonl").exists()
    assert (tmp_path / "displaced").read_bytes() == b"x"


def test_temp_substitution_is_denied_before_chmod(tmp_path):
    root = tmp_path / "child"
    root.mkdir()
    with _replace_owned_inode(root, rows=False):
        run_writer_control(
            """
import stat, time
from silent_cascade.io import _durable_temp
temporary = _durable_temp(root / 'weights.safetensors', b'x', 0o400)
deadline = time.monotonic() + 5
while temporary.read_bytes() != b'y' and time.monotonic() < deadline:
    time.sleep(0.01)
assert temporary.read_bytes() == b'y'
with denied('authority'):
    os.chmod(temporary, 0o644)
assert stat.S_IMODE(temporary.stat().st_mode) == 0o600
""",
            root,
        )
    (temporary,) = root.glob(".weights.safetensors.*.tmp")
    assert stat.S_IMODE(temporary.stat().st_mode) == 0o600
    assert (tmp_path / "displaced").read_bytes() == b"x"


@pytest.mark.parametrize(
    "key,value",
    [
        ("allocated_bytes", True),
        ("allocated_bytes", None),
        ("allocated_bytes", 0),
        ("allocated_bytes", -1),
        ("allocated_bytes", 1.5),
        ("allocated_bytes", "4096"),
        ("allocated_bytes", 2**63),
        ("file_names", 0),
        ("file_names", 103),
        ("directories", 0),
        ("directories", 10),
        ("unknown", 1),
    ],
)
def test_allowance_rejects_invalid_values(key, value):
    import importlib.util

    assert importlib.util.find_spec("silent_cascade.train.pilot_offline_writes") is not None
    from silent_cascade.train.pilot_offline_writes import OfflineWriteAllowance

    with pytest.raises((TypeError, ValueError)):
        OfflineWriteAllowance(**(dict(allocated_bytes=65536) | {key: value}))


def test_install_requires_exact_allowance_type(tmp_path):
    from silent_cascade.train.pilot_offline_writes import (
        OfflineWriteAllowance,
        install_offline_writes,
    )

    class DerivedAllowance(OfflineWriteAllowance):
        pass

    with pytest.raises(ValueError, match="invalid offline write allowance"):
        install_offline_writes(tmp_path, DerivedAllowance(65536))
    assert not list(tmp_path.iterdir())


def test_install_cannot_replace_or_enlarge_allowance(tmp_path):
    run_writer_control(
        """
with denied('authority'):
    install_offline_writes(root, OfflineWriteAllowance(131072))
assert not list(root.iterdir())
""",
        tmp_path / "child",
    )


@pytest.mark.parametrize(
    "module_name",
    [
        "silent_cascade.train.pilot_checks",
        "silent_cascade.train.pilot_evidence",
        "silent_cascade.report.pilot",
    ],
)
def test_fit_guard_binds_existing_publisher_aliases(tmp_path, monkeypatch, module_name):
    from . import cold_authentication_fixture as fixture

    publisher = __import__(module_name, fromlist=["_publish_pilot_bytes"])
    run = tmp_path / "run"
    run.mkdir()
    guard = fixture.FitGuard(tmp_path, tmp_path / "cache")
    monkeypatch.setattr(fixture, "FIT_BYTES", 0)
    with guard.installed(run), pytest.raises(AssertionError, match="before write"):
        publisher._publish_pilot_bytes(run / "blocked.json", b"x")
    assert list(run.iterdir()) == []


@pytest.mark.parametrize("short", [0, 1])
def test_fit_guard_reserves_exact_offline_hold_before_publication(tmp_path, monkeypatch, short):
    with _held_fixture_attempt(tmp_path) as (
        fixture,
        guard,
        custody,
        output,
        allowance,
        hold,
    ):
        guard.cache.mkdir()
        (guard.cache / "retained").write_bytes(b"x")
        used, files, directories = fixture.allocation(guard.cache)
        exact = hold + used + (files + directories) * fixture.BLOCK
        monkeypatch.setattr(fixture, "FIT_BYTES", exact - short)
        monkeypatch.setattr(fixture, "FIT_NAMES", allowance.file_names + 14 + files)
        # Child directories start at offline; two missing public ancestors and
        # the three prepared private-custody directories remain parent-held.
        monkeypatch.setattr(
            fixture,
            "FIT_DIRECTORIES",
            allowance.directories + 5 + directories,
        )
        if short:
            with pytest.raises(AssertionError, match="before write"):
                guard.reserve_offline_attempt(
                    output,
                    write_allowance=allowance,
                    process_limits=custody.limits,
                    process_custody=custody,
                )
        else:
            guard.reserve_offline_attempt(
                output,
                write_allowance=allowance,
                process_limits=custody.limits,
                process_custody=custody,
            )
            assert guard.maximum == exact
        assert not output.exists()
        assert os.listdir(custody.directory_fd) == []


@pytest.mark.parametrize("existing", [False, True], ids=["missing", "existing-shared"])
@pytest.mark.parametrize("directory_limit, accepted", [(13, False), (14, True)])
def test_fit_guard_counts_public_ancestors_outside_spool(
    tmp_path, monkeypatch, existing, directory_limit, accepted
):
    with _held_fixture_attempt(tmp_path) as (
        fixture,
        original_guard,
        custody,
        output,
        allowance,
        hold,
    ):
        if existing:
            output.parent.mkdir(parents=True)
        assert output.parent.exists() is existing
        assert output.parent.parent.exists() is existing

        guard = fixture.FitGuard(output.parent, original_guard.cache)
        used, files, directories = fixture.allocation(guard.spool)
        exact = hold + used + (files + directories) * fixture.BLOCK
        monkeypatch.setattr(fixture, "FIT_BYTES", exact)
        monkeypatch.setattr(fixture, "FIT_NAMES", allowance.file_names + 14 + files)
        monkeypatch.setattr(fixture, "FIT_DIRECTORIES", directory_limit)

        if accepted:
            guard.reserve_offline_attempt(
                output,
                write_allowance=allowance,
                process_limits=custody.limits,
                process_custody=custody,
            )
            assert guard.maximum == exact
            assert guard.blocked is None
        else:
            with pytest.raises(AssertionError, match="directory bound"):
                guard.reserve_offline_attempt(
                    output,
                    write_allowance=allowance,
                    process_limits=custody.limits,
                    process_custody=custody,
                )
        assert not output.exists()
        assert os.listdir(custody.directory_fd) == []


def test_fit_guard_real_parent_publication_creates_fresh_held_root(tmp_path, monkeypatch):
    from silent_cascade.train import pilot_data
    from silent_cascade.train import pilot_offline_process as process

    with _held_fixture_attempt(tmp_path) as (
        fixture,
        guard,
        custody,
        output,
        allowance,
        hold,
    ):
        monkeypatch.setattr(fixture, "FIT_BYTES", hold)
        monkeypatch.setattr(fixture, "FIT_NAMES", allowance.file_names + 14)
        monkeypatch.setattr(fixture, "FIT_DIRECTORIES", allowance.directories + 5)
        guard.reserve_offline_attempt(
            output,
            write_allowance=allowance,
            process_limits=custody.limits,
            process_custody=custody,
        )
        process._publish_private_process_bytes(custody, "binding.json", b"x")
        with guard.installed(guard.spool):
            pilot_data._publish_pilot_bytes(output / "intent.json", b"x")
        assert (output / "intent.json").read_bytes() == b"x"
        assert guard.blocked is None


def test_fit_guard_offline_hold_cannot_be_rebound_or_replenished(tmp_path, monkeypatch):
    with _held_fixture_attempt(tmp_path) as (
        fixture,
        guard,
        custody,
        output,
        allowance,
        hold,
    ):
        monkeypatch.setattr(fixture, "FIT_BYTES", hold * 2)
        guard.reserve_offline_attempt(
            output,
            write_allowance=allowance,
            process_limits=custody.limits,
            process_custody=custody,
        )
        with pytest.raises(AssertionError, match="already held"):
            guard.reserve_offline_attempt(
                output,
                write_allowance=allowance,
                process_limits=custody.limits,
                process_custody=custody,
            )
        assert guard.maximum == hold


@pytest.mark.parametrize("short", ["names", "directories"])
def test_fit_guard_offline_hold_rejects_short_inventory(tmp_path, monkeypatch, short):
    with _held_fixture_attempt(tmp_path) as (
        fixture,
        guard,
        custody,
        output,
        allowance,
        hold,
    ):
        monkeypatch.setattr(fixture, "FIT_BYTES", hold)
        monkeypatch.setattr(
            fixture,
            "FIT_NAMES",
            allowance.file_names + 14 - (short == "names"),
        )
        monkeypatch.setattr(
            fixture,
            "FIT_DIRECTORIES",
            allowance.directories + 5 - (short == "directories"),
        )
        with pytest.raises(AssertionError, match=r"file name bound|directory bound"):
            guard.reserve_offline_attempt(
                output,
                write_allowance=allowance,
                process_limits=custody.limits,
                process_custody=custody,
            )
        assert not output.exists()
        assert os.listdir(custody.directory_fd) == []


def test_fit_guard_live_child_bytes_are_not_double_counted(tmp_path, monkeypatch):
    with _held_fixture_attempt(tmp_path, extra_spool=32768) as (
        fixture,
        guard,
        custody,
        output,
        allowance,
        hold,
    ):
        monkeypatch.setattr(fixture, "FIT_BYTES", hold + 32768)
        guard.reserve_offline_attempt(
            output,
            write_allowance=allowance,
            process_limits=custody.limits,
            process_custody=custody,
        )
        output.mkdir(parents=True)
        (output / "step.json").write_bytes(b"x" * 4096)
        guard.admit(guard.spool / "outer.json", 1)
        assert guard.blocked is None


def test_fit_guard_unrelated_output_cannot_spend_offline_hold(tmp_path, monkeypatch):
    with _held_fixture_attempt(tmp_path, extra_spool=24576) as (
        fixture,
        guard,
        custody,
        output,
        allowance,
        hold,
    ):
        monkeypatch.setattr(fixture, "FIT_BYTES", hold + 24576)
        guard.reserve_offline_attempt(
            output,
            write_allowance=allowance,
            process_limits=custody.limits,
            process_custody=custody,
        )
        first = guard.spool / "outer.json"
        guard.admit(first, 1)
        first.parent.mkdir(parents=True)
        first.write_bytes(b"x")
        with pytest.raises(AssertionError, match="before write"):
            guard.admit(guard.spool / "second.json", 1)
        assert not (guard.spool / "second.json").exists()


def test_fit_guard_routes_parent_publications_through_held_process_peak(tmp_path, monkeypatch):
    from silent_cascade.train import pilot_offline_process as process

    with _held_fixture_attempt(tmp_path) as (
        fixture,
        guard,
        custody,
        output,
        allowance,
        hold,
    ):
        monkeypatch.setattr(fixture, "FIT_BYTES", hold)
        guard.reserve_offline_attempt(
            output,
            write_allowance=allowance,
            process_limits=custody.limits,
            process_custody=custody,
        )
        process._publish_private_process_bytes(custody, "binding.json", b"x")
        for name, size in (
            ("intent.json", custody.limits.record_bytes),
            ("stdout.txt", 0),
            ("stderr.txt", 0),
            ("process-result.json", custody.limits.record_bytes),
        ):
            guard.admit(output / name, size)
            output.mkdir(parents=True, exist_ok=True)
            (output / name).write_bytes(b"x" * size)
        assert guard.blocked is None


def test_fit_guard_rejects_split_child_categories(tmp_path, monkeypatch):
    with _held_fixture_attempt(
        tmp_path,
        bindings=(("run/final/offline/child", "scratch"),),
    ) as (fixture, guard, custody, output, allowance, hold):
        monkeypatch.setattr(fixture, "FIT_BYTES", hold * 2)
        with pytest.raises(AssertionError, match="child category"):
            guard.reserve_offline_attempt(
                output,
                write_allowance=allowance,
                process_limits=custody.limits,
                process_custody=custody,
            )


@pytest.mark.parametrize("invalid", ["allowance", "limits", "output"])
def test_fit_guard_validates_exact_offline_hold_bindings(tmp_path, monkeypatch, invalid):
    from silent_cascade.train.pilot_offline_process import OfflineProcessLimits
    from silent_cascade.train.pilot_offline_writes import OfflineWriteAllowance

    class DerivedAllowance(OfflineWriteAllowance):
        pass

    class DerivedLimits(OfflineProcessLimits):
        pass

    with _held_fixture_attempt(tmp_path) as (
        fixture,
        guard,
        custody,
        output,
        allowance,
        hold,
    ):
        monkeypatch.setattr(fixture, "FIT_BYTES", hold * 2)
        selected_allowance = (
            DerivedAllowance(**allowance.__dict__) if invalid == "allowance" else allowance
        )
        selected_limits = (
            DerivedLimits(**custody.limits.__dict__) if invalid == "limits" else custody.limits
        )
        selected_output = output.with_name("different") if invalid == "output" else output
        with pytest.raises((AssertionError, ValueError), match=r"offline|custody|root"):
            guard.reserve_offline_attempt(
                selected_output,
                write_allowance=selected_allowance,
                process_limits=selected_limits,
                process_custody=custody,
            )


@pytest.mark.parametrize(
    "module", ["silent_cascade.eval.artifacts", "silent_cascade.train.pilot_data"]
)
def test_late_consumer_binding_is_denied(tmp_path, module):
    run_writer_control(
        "", tmp_path / "child", installation_denial="authority", before_binding=f"import {module}"
    )


def test_real_publishers_exact_capacity(tmp_path):
    root = tmp_path / "child"
    root.mkdir()
    # Empty root, one incoming block, temp + final names, one directory.
    capacity = root.stat().st_blocks * 512 + 6 * 4096
    run_writer_control(
        """
_publish_pilot_bytes(root / 'step.json', b'x')
assert (root / 'step.json').read_bytes() == b'x'
guard.assert_clear()
""",
        root,
        capacity,
    )


def test_rows_stream_and_final_link(tmp_path):
    run_writer_control(
        """
pending = root / 'run/eval/primary/.rows.pending.jsonl'
pending.parent.mkdir(parents=True)
with pending.open('xb') as stream:
    assert stream.write(b'{}\\n') == 3
    assert stream.tell() == 3
    stream.flush()
    os.fsync(stream.fileno())
    atomic_create_bytes(pending.parent / 'identity.json', b'{}')
os.link(pending, pending.parent / 'rows.jsonl')
pending.unlink()
assert (pending.parent / 'rows.jsonl').read_bytes() == b'{}\\n'
guard.assert_clear()
""",
        tmp_path / "child",
    )


def test_closed_git_batch_pipe_still_works(tmp_path):
    run_writer_control(
        """
import subprocess
from silent_cascade.train import pilot_offline
repo = Path(pilot_offline.__file__).resolve().parents[3]
completed = subprocess.run(['git', 'cat-file', '--batch'], cwd=repo,
    input=b'HEAD:src/silent_cascade/__init__.py\\n', capture_output=True,
    check=True, timeout=5)
assert b' blob ' in completed.stdout.split(b'\\n', 1)[0]
assert len(completed.stdout) < 2048
guard.assert_clear()
""",
        tmp_path / "child",
    )


def test_transitive_evaluation_publisher_keeps_token_lifetime(tmp_path):
    run_writer_control(
        """
from silent_cascade.eval.artifacts import publish_evaluation_bytes
publish_evaluation_bytes(root / 'run/eval/primary/identity.json', b'{}')
publish_evaluation_bytes(root / 'run/eval/primary/retention.json', b'{}')
guard.assert_clear()
""",
        tmp_path / "child",
    )


@pytest.mark.parametrize("existing", [False, True])
def test_publication_one_byte_short_has_no_temp_or_mutation(tmp_path, existing):
    root = tmp_path / "child"
    root.mkdir()
    if existing:
        (root / "weights.safetensors").write_bytes(b"old")
    # Existing bytes remain charged, with two full incoming publication names.
    used = sum(p.stat().st_blocks * 512 for p in (root, *root.iterdir()))
    capacity = used + (7 if existing else 6) * 4096 - 1
    run_writer_control(
        """
with denied('bytes'):
    atomic_write_bytes(root / 'weights.safetensors', b'x' * 4096)
assert not any(p.name.endswith('.tmp') for p in root.iterdir())
""",
        root,
        capacity,
    )
    assert (
        ((root / "weights.safetensors").read_bytes() == b"old")
        if existing
        else not (root / "weights.safetensors").exists()
    )


def test_create_collision_reserves_incoming_bytes(tmp_path):
    root = tmp_path / "child"
    root.mkdir()
    (root / "step.json").write_bytes(b"old")
    used = sum(p.stat().st_blocks * 512 for p in (root, *root.iterdir()))
    run_writer_control(
        """
with denied('bytes'):
    _publish_pilot_bytes(root / 'step.json', b'old')
assert (root / 'step.json').read_bytes() == b'old'
assert not any(p.name.endswith('.tmp') for p in root.iterdir())
""",
        root,
        used + 7 * 4096 - 1,
    )


def test_atomic_replace_and_create_preserve_original_behavior(tmp_path):
    run_writer_control(
        """
atomic_create_bytes(root / 'weights.safetensors', b'a')
atomic_write_bytes(root / 'weights.safetensors', b'b')
_publish_pilot_bytes(root / 'step.json', b'x')
_publish_pilot_bytes(root / 'step.json', b'x')
assert (root / 'weights.safetensors').read_bytes() == b'b'
guard.assert_clear()
""",
        tmp_path / "child",
    )


def test_second_atomic_temp_denied_and_owned_cleanup_allowed(tmp_path):
    run_writer_control(
        """
import tempfile
from silent_cascade.io import _durable_temp, _cleanup_temp
temporary = _durable_temp(root / 'weights.safetensors', b'x', 0o644)
with denied('writer'):
    tempfile.mkstemp(prefix='.weights.safetensors.', suffix='.tmp', dir=root)
assert _cleanup_temp(temporary) is None
assert not list(root.iterdir())
""",
        tmp_path / "child",
    )


def test_pending_and_final_hardlink_both_charge_allocation(tmp_path):
    root = tmp_path / "child"
    parent = root / "run/eval/primary"
    parent.mkdir(parents=True)
    used = sum(p.stat().st_blocks * 512 for p in (root, root / "run", root / "run/eval", parent))
    run_writer_control(
        """
pending = root / 'run/eval/primary/.rows.pending.jsonl'
final = pending.parent / 'rows.jsonl'
with pending.open('xb') as stream:
    stream.write(b'x')
os.link(pending, final)
assert pending.stat().st_ino == final.stat().st_ino
with denied('bytes'):
    _publish_pilot_bytes(root / 'step.json', b'x')
pending.unlink()
assert final.read_bytes() == b'x'
""",
        root,
        used + 13 * 4096 - 1,
    )


@pytest.mark.parametrize(
    "relative,writer",
    [
        ("unknown.json", "pilot"),
        ("report/trajectories-1.svg", "pilot"),
        ("run/eval/primary/episodes/00016.neural.json", "atomic"),
        ("run/eval/primary/episodes/0000.neural.json", "atomic"),
        ("cache/unknown", "atomic"),
        ("matplotlib-other/unknown", "pilot"),
        ("stdout.txt", "pilot"),
        ("intent.json", "pilot"),
        ("weights.safetensors", "pilot"),
    ],
)
def test_closed_names_latch_before_publication(tmp_path, relative, writer):
    run_writer_control(
        f"""
with denied():
    guard.admit_publication(root / {relative!r}, 1, writer={writer!r})
assert not (root / {relative!r}).exists()
""",
        tmp_path / "child",
    )


@pytest.mark.parametrize("which", ["names", "directories"])
def test_lowered_inventory_ceiling_denies_before_creation(tmp_path, which):
    limits = dict(
        allocated_bytes=65536, **({"file_names": 1} if which == "names" else {"directories": 1})
    )
    run_writer_control(
        f"""
with denied({which!r}):
    _publish_pilot_bytes(root / {"step.json" if which == "names" else "report/report.md"!r}, b'x')
assert not list(root.iterdir())
""",
        tmp_path / "child",
        limits,
    )


def test_second_weights_digest_denied(tmp_path):
    run_writer_control(
        """
parent = root / 'run/eval/primary/crashes'
atomic_create_bytes(parent / ('weights-' + 'a' * 64 + '.safetensors'), b'x')
with denied('path'):
    atomic_create_bytes(parent / ('weights-' + 'b' * 64 + '.safetensors'), b'x')
assert len(list(parent.iterdir())) == 1
""",
        tmp_path / "child",
    )


def test_seventeenth_crash_stem_denied(tmp_path):
    root = tmp_path / "child"
    parent = root / "run/eval/primary/crashes"
    parent.mkdir(parents=True)
    for number in range(16):
        (parent / f"{number:032x}.json").touch()
    run_writer_control(
        """
with denied('path'):
    atomic_create_bytes(root / ('run/eval/primary/crashes/' + 'f' * 32 + '.json'), b'x')
""",
        root,
        131072,
    )


def test_last_episode_ordinal_and_crash_pair_are_admitted(tmp_path):
    run_writer_control(
        """
atomic_create_bytes(root / 'run/eval/primary/episodes/00015.neural.json', b'x')
crashes = root / 'run/eval/primary/crashes'
atomic_create_bytes(crashes / ('a' * 32 + '.json'), b'x')
atomic_create_bytes(crashes / ('a' * 32 + '.safetensors'), b'x')
guard.assert_clear()
assert len(list(crashes.iterdir())) == 2
""",
        tmp_path / "child",
    )


@pytest.mark.parametrize(
    "escape",
    [
        "buffer",
        "raw",
        "fdopen",
        "dup",
        "write",
        "truncate",
        "mmap",
        "fileio",
        "seek",
        "writelines",
        "reopen",
    ],
)
def test_rows_descriptor_and_stream_escapes_latch(tmp_path, escape):
    actions = {
        "buffer": "stream.buffer",
        "raw": "stream.raw",
        "fdopen": "os.fdopen(stream.fileno(), 'wb')",
        "dup": "os.dup(stream.fileno())",
        "write": "os.write(stream.fileno(), b'x')",
        "truncate": "os.ftruncate(stream.fileno(), 0)",
        "mmap": "mmap.mmap(stream.fileno(), 1)",
        "fileio": "io.FileIO(stream.fileno(), 'wb')",
        "seek": "stream.seek(0)",
        "writelines": "stream.writelines([b'x'])",
        "reopen": "pending.open('ab')",
    }
    run_writer_control(
        f"""
import io, mmap
pending = root / 'run/eval/primary/.rows.pending.jsonl'
pending.parent.mkdir(parents=True)
with pending.open('xb') as stream:
    stream.write(b'x')
    with denied():
        {actions[escape]}
assert pending.read_bytes() == b'x'
""",
        tmp_path / "child",
    )


@pytest.mark.parametrize("escape", ["relative", "dirfd", "raw", "symlink", "temp"])
def test_unowned_writes_and_paths_latch(tmp_path, escape):
    actions = {
        "relative": "os.open(root / '../escape', os.O_WRONLY | os.O_CREAT, 0o600)",
        "dirfd": "os.open('../escape', os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=fd)",
        "raw": "(root / 'step.json').write_bytes(b'x')",
        "symlink": "os.symlink('step.json', root / 'manifest.json')",
        "temp": "(root / ('.pilot-' + 'a' * 32 + '.tmp')).write_bytes(b'x')",
    }
    run_writer_control(
        f"""
fd = os.open(root, os.O_RDONLY)
try:
    with denied():
        {actions[escape]}
finally:
    os.close(fd)
assert not list(root.iterdir())
""",
        tmp_path / "child",
    )


def test_external_hardlink_fails_at_installation(tmp_path):
    import os

    root = tmp_path / "child"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"old")
    os.link(outside, root / "step.json")
    run_writer_control("", root, installation_denial="authority")
    assert outside.read_bytes() == b"old"


def test_preexisting_symlink_fails_at_installation(tmp_path):
    root = tmp_path / "child"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"old")
    (root / "step.json").symlink_to(outside)
    run_writer_control("", root, installation_denial="path")
    assert outside.read_bytes() == b"old"


def test_real_font_manager_cache_and_lock(tmp_path):
    run_writer_control(
        """
from matplotlib import font_manager
cache = root / 'matplotlib-cache/fontlist-v3.11.0.json'
assert font_manager.FontManager.__version__ == '3.11.0'
assert cache.is_file() and 0 < cache.stat().st_size < 65536
assert not cache.with_name(cache.name + '.matplotlib-lock').exists()
guard.assert_clear()
""",
        tmp_path / "child",
        262144,
    )


def test_real_font_json_dump_overflow_latches_and_cleans_lock(tmp_path):
    run_writer_control(
        """
with denied('bytes'):
    from matplotlib import font_manager
cache = root / 'matplotlib-cache/fontlist-v3.11.0.json'
assert cache.is_file()
assert 0 < cache.stat().st_size < 16384
assert not cache.with_name(cache.name + '.matplotlib-lock').exists()
""",
        tmp_path / "child",
        65536,
    )


@pytest.mark.parametrize("overflow", [False, True])
def test_utf8_stream_admits_encoded_length(tmp_path, overflow):
    root = tmp_path / "child"
    parent = root / "matplotlib-cache"
    parent.mkdir(parents=True)
    used = sum(p.stat().st_blocks * 512 for p in (root, parent))
    # Growth reserves one extra name even for an existing zero-byte stream.
    run_writer_control(
        f"""
cache = root / 'matplotlib-cache/fontlist-v3.11.0.json'
with cache.open('w', encoding='utf-8') as stream:
    assert stream.write('é' * 2048) == 2048
    {'with denied("bytes"):' if overflow else "if True:"}
        {'stream.write("é")' if overflow else "stream.flush()"}
assert cache.read_bytes() == ('é' * 2048).encode()
""",
        root,
        used + 7 * 4096,
    )
