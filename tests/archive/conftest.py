import functools
import hashlib
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

_ENGINEERING_AUTHORITY = ".silent-cascade-engineering.json"
_ENGINEERING_AUTHORITY_LIMIT = 16_384


def _authority_boundary(case_root: Path) -> tuple[tuple[int, int], ...]:
    """Capture only pre-existing ancestor authorities, before a case creates any."""
    identities = []
    for parent in case_root.parents:
        try:
            (parent / _ENGINEERING_AUTHORITY).lstat()
        except FileNotFoundError:
            continue
        info = parent.stat()
        identities.append((info.st_dev, info.st_ino))
    return tuple(identities)


@contextmanager
def _hide_host_authorities(boundary: tuple[tuple[int, int], ...]):
    """Hide captured host custody only at the exact production read boundary."""
    from silent_cascade.archive import preflight

    original = preflight._read_at
    hidden = set(boundary)

    def isolated(parent, name, *, max_bytes):
        info = os.fstat(parent)
        if (
            name == _ENGINEERING_AUTHORITY
            and max_bytes == _ENGINEERING_AUTHORITY_LIMIT
            and (info.st_dev, info.st_ino) in hidden
        ):
            raise FileNotFoundError(name)
        return original(parent, name, max_bytes=max_bytes)

    isolated.__wrapped__ = original
    preflight._read_at = isolated
    try:
        yield
    finally:
        preflight._read_at = original


def _isolated_spawn_target(boundary, target, args):
    with _hide_host_authorities(boundary):
        target(*args)


def _isolated_archive_child(boundary, *args):
    with _hide_host_authorities(boundary):
        from silent_cascade.archive._qualification import _archive_child

        _archive_child(*args)


@dataclass(frozen=True)
class AuthorityIsolation:
    boundary: tuple[tuple[int, int], ...]

    def target(self, callable_, *args):
        return functools.partial(_isolated_spawn_target, self.boundary, callable_, args)

    @contextmanager
    def host_visible(self):
        from silent_cascade.archive import preflight

        isolated = preflight._read_at
        original = isolated.__wrapped__
        preflight._read_at = original
        try:
            yield
        finally:
            preflight._read_at = isolated


@pytest.fixture(autouse=True)
def isolated_archive_authorities(request, monkeypatch):
    """Keep synthetic authorities independent from the live parent custody root."""
    if request.module.__name__ not in {"test_preflight", "test_qualification"}:
        yield None
        return
    case_root = request.getfixturevalue("tmp_path")
    boundary = _authority_boundary(case_root)
    isolation = AuthorityIsolation(boundary)
    from silent_cascade.archive import _qualification

    with _hide_host_authorities(boundary):
        monkeypatch.setattr(
            _qualification,
            "_archive_child",
            functools.partial(_isolated_archive_child, boundary),
        )
        yield isolation


@pytest.fixture
def tiny_archive_policy():
    from silent_cascade.archive.types import ArchivePolicy

    return ArchivePolicy(
        workspace_bytes=40_000,
        spool_bytes=4_096,
        cache_bytes=4_096,
        pinned_bytes=1_024,
        metadata_bytes=16_384,
        scratch_bytes=1_024,
        logs_bytes=512,
        emergency_bytes=2_048,
        reserve_bytes=4_096,
        episode_bytes=2_048,
        pack_target_bytes=512,
        pack_episodes=4,
        journal_bytes=256,
        journal_records=4,
        chunk_bytes=64,
        page_bytes=1_024,
        page_entries=3,
        remote_bytes=100_000,
    )


@pytest.fixture
def archive_identity() -> dict[str, object]:
    return {
        "run_id": "debug-fixture",
        "source_commit": "0" * 40,
        "config_sha256": "1" * 64,
        "evidence_identity_sha256": "2" * 64,
        "checkpoint_sha256": None,
        "writer_stopped": True,
        "checkpoint_committed": False,
    }


@dataclass(frozen=True)
class SealedUnit:
    root: Path
    control: Path
    ref: object
    policy: object
    paths: tuple[str, ...]

    def original_bytes(self) -> dict[str, bytes]:
        return {path: (self.root / path).read_bytes() for path in self.paths}


class DirectoryTransport:
    """Real-file transport double with deterministic transfer fault injection."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True)
        self.create_calls = 0
        self.download_calls = 0
        self.corrupt_downloads = False
        self.fail_create_call: int | None = None
        self.fail_download_call: int | None = None
        self.ambiguous_create_call: int | None = None
        self.conflict_create_call: int | None = None
        self.collision_create_call: int | None = None
        self.fail_key_contains: str | None = None
        self.failed_key_once = False

    @property
    def transport_id(self) -> str:
        return "directory:" + hashlib.sha256(str(self.root).encode()).hexdigest()

    def _path(self, key: str) -> Path:
        if not key or key.startswith("/") or ".." in Path(key).parts:
            raise ValueError("unsafe transport key")
        return self.root / key

    def create(self, key: str, source: Path) -> None:
        self.create_calls += 1
        if (
            self.fail_key_contains is not None
            and self.fail_key_contains in key
            and not self.failed_key_once
        ):
            self.failed_key_once = True
            raise OSError("injected keyed create failure")
        if self.fail_create_call == self.create_calls:
            raise PermissionError("credential expired secret-token")
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.collision_create_call == self.create_calls:
            destination.write_bytes(b"different-object")
            raise FileExistsError(key)
        try:
            with destination.open("xb") as output, source.open("rb") as input_file:
                shutil.copyfileobj(input_file, output)
        except FileExistsError:
            raise
        if self.conflict_create_call == self.create_calls:
            raise FileExistsError(key)
        if self.ambiguous_create_call == self.create_calls:
            from silent_cascade.archive.transport import AmbiguousWriteError

            raise AmbiguousWriteError("ambiguous write")

    def download(self, key: str, destination: Path, *, max_bytes: int) -> None:
        self.download_calls += 1
        if self.fail_download_call == self.download_calls:
            raise OSError("injected download failure")
        payload = self._path(key).read_bytes()
        if len(payload) > max_bytes:
            raise ValueError("remote object exceeds byte limit")
        if self.corrupt_downloads and payload:
            payload = bytes([payload[0] ^ 1]) + payload[1:]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)


@pytest.fixture
def task_scratch():
    scratch_parent = Path(
        os.environ.get(
            "SILENT_CASCADE_ARCHIVE_TEST_SCRATCH",
            Path(__file__).parents[2] / ".superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-2",
        )
    )
    scratch_parent.mkdir(parents=True, exist_ok=True)
    root = Path(
        tempfile.mkdtemp(
            prefix="case-",
            dir=scratch_parent,
        )
    )
    try:
        yield root
    finally:
        if not (
            os.environ.get("SILENT_CASCADE_ARCHIVE_TEST_SCRATCH")
            and os.environ.get("SILENT_CASCADE_ARCHIVE_TEST_PRESERVE_SCRATCH") == "1"
        ):
            shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def transport(task_scratch, tiny_archive_policy):
    from silent_cascade.archive.transport import initialize_remote_reservations

    instance = DirectoryTransport(task_scratch / "remote")
    initialize_remote_reservations(
        control_dir=task_scratch / "control",
        transport_id=instance.transport_id,
        accounted_bytes=0,
        accounting_evidence_sha256="a" * 64,
        policy=tiny_archive_policy,
    )
    return instance


@pytest.fixture
def sealed_unit(task_scratch, tiny_archive_policy, archive_identity):
    from silent_cascade.archive.catalog import seal_unit

    policy = tiny_archive_policy.model_copy(
        update={"workspace_bytes": 60_000, "metadata_bytes": 32_768, "page_bytes": 4_096}
    )
    run = task_scratch / "run"
    paths = (
        "pack/crashes/crash-00000.json",
        "pack/episodes/00000.neural.json",
        "pack/episodes/00000.trajectory.json.gz",
    )
    payloads = (b"crash", b"neural", b"trajectory")
    for path, payload in zip(paths, payloads, strict=True):
        destination = run / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    control = task_scratch / "control"
    ref = seal_unit(
        run_dir=run,
        control_dir=control,
        logical_root="pack",
        paths=paths,
        kind="episode_pack",
        identity=archive_identity,
        policy=policy,
        episode_groups=(paths,),
    )
    return SealedUnit(run, control, ref, policy, paths)
