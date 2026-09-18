"""Closed engineering custody for one frozen plan and one operational workspace.

The controller freezes all other plan writes before bootstrap and keeps them
frozen afterward. Only the pinned operational child may grow under the ordinary
ledger's reservations. There is no thaw, rebaseline, caller byte credit, or
arbitrary root registration API. Fixture owner files are opaque retained bytes.
"""

import hashlib
import json
import os
import re
import stat
import subprocess
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter

from silent_cascade.archive import _retained
from silent_cascade.archive.catalog import (
    _control_reader,
    _create_at,
    _create_control_object,
    _open_child_directory,
    _pinned_directory,
    _read_at,
    _verify_pinned_directory,
)
from silent_cascade.archive.ledger import StorageBlocked, _process_identity, _StorageBudget
from silent_cascade.archive.transport import _lock
from silent_cascade.archive.types import (
    ArchivePolicy,
    EngineeringContentReview,
    FileEntry,
    Hash,
    Revision,
    UnitRef,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.validation import StrictModel

_AUTHORITY = ".silent-cascade-engineering.json"
_TASKS = ("task-1", "task-2", "task-3", "task-4", "task-4-fix1", "task-4-fix2")
_REGULAR_LIMIT = 128 * 1024**2
_LINKED_INODES_LIMIT = 4096
_SOURCE_AUTHORITY_LIMIT = 16 * 1024
_SOURCE_CLOSURE_SCHEMA = "silent-cascade-python-v1"


class _ReviewedSourceRecord(StrictModel):
    schema_version: Literal["phase4-reviewed-source-v1"] = "phase4-reviewed-source-v1"
    anchor_authority_sha256: Hash
    source_commit: Revision
    executable_sha256: Hash
    source_closure_schema: Literal["silent-cascade-python-v1"] = _SOURCE_CLOSURE_SCHEMA
    policy_sha256: Hash
    review_sha256: Hash
    failure_proof_sha256: Hash | None = None


@dataclass(frozen=True)
class _SourceIdentity:
    source_commit: str
    executable_sha256: str


@dataclass(frozen=True)
class _ReviewedSource:
    source_commit: str
    executable_sha256: str
    source_authority_sha256: str


def _allocation_unit(volume):
    # statvfs.f_frsize is the allocation unit. On Darwin f_bsize is only
    # preferred I/O length (statfs.f_iosize), not physical allocation rounding.
    unit = volume.f_frsize
    if (
        type(unit) is not int
        or not 0 < unit <= 2**63 - 1
        or unit % 512
        or type(volume.f_bavail) is not int
        or not 0 <= volume.f_bavail <= volume.f_blocks
    ):
        raise StorageBlocked("storage_blocked: unusable allocation geometry")
    return unit


def _read_authority_at(directory: int) -> bytes | None:
    try:
        return _read_at(directory, _AUTHORITY, max_bytes=16384)
    except FileNotFoundError:
        return None


def _require_single_authority(workspace: Path) -> None:
    for parent in workspace.parents:
        with _pinned_directory(parent) as directory:
            raw = _read_authority_at(directory)
        if raw is None:
            continue
        authority = json.loads(raw)
        if authority.get("workspace") != str(workspace):
            raise StorageBlocked("storage_blocked: another workspace owns this custody allowance")


def _stat_record(path: str, info: os.stat_result) -> dict:
    return {
        "path": path,
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": info.st_mode,
        "links": info.st_nlink,
        "bytes": info.st_size,
        "allocated": info.st_blocks * 512,
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def _names(directory, *, reverse=False):
    """Ordered traversal with constant directory-entry memory, including huge roots."""
    previous = None
    while True:
        following = None
        with os.scandir(directory) as entries:
            for entry in entries:
                beyond = previous is None or (
                    entry.name < previous if reverse else entry.name > previous
                )
                closer = following is None or (
                    entry.name > following if reverse else entry.name < following
                )
                if beyond and closer:
                    following = entry.name
        if following is None:
            return
        yield following
        previous = following


def _inventory(custody_root: Path, workspace: Path, *, hashes=True, reverse=False):
    """Descriptor-relative accounting; unsupported files are never opened."""
    excluded = workspace.name
    with _pinned_directory(custody_root) as root:
        device = os.fstat(root).st_dev

        def visit(directory, prefix):
            initial = os.fstat(directory)
            record = _stat_record(prefix, initial)
            record["sha256"] = None
            directory_record = record
            if not reverse:
                yield directory_record
            for name in _names(directory, reverse=reverse):
                if prefix == "." and name == excluded:
                    continue
                path = name if prefix == "." else f"{prefix}/{name}"
                if len(path.encode()) > 4096:
                    raise StorageBlocked("storage_blocked: retained path exceeds finite inventory")
                info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if info.st_dev != device:
                    raise StorageBlocked("storage_blocked: retained root crosses device")
                if stat.S_ISDIR(info.st_mode):
                    child = _open_child_directory(directory, name)
                    try:
                        opened = os.fstat(child)
                        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                            raise StorageBlocked("storage_blocked: retained directory changed")
                        yield from visit(child, path)
                    finally:
                        os.close(child)
                    continue
                record = _stat_record(path, info)
                record["sha256"] = None
                if (
                    hashes
                    and stat.S_ISREG(info.st_mode)
                    and info.st_nlink == 1
                    and info.st_size <= _REGULAR_LIMIT
                    and info.st_blocks * 512 >= info.st_size
                ):
                    descriptor = os.open(
                        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
                    )
                    try:
                        if _stat_record(path, os.fstat(descriptor)) != _stat_record(path, info):
                            raise StorageBlocked("storage_blocked: retained file changed")
                        digest = hashlib.sha256()
                        while block := os.read(descriptor, 65536):
                            digest.update(block)
                        record["sha256"] = digest.hexdigest()
                        if _stat_record(path, os.fstat(descriptor)) != _stat_record(path, info):
                            raise StorageBlocked("storage_blocked: live retained writer")
                    finally:
                        os.close(descriptor)
                yield record
            if _stat_record(prefix, os.fstat(directory)) != _stat_record(prefix, initial):
                raise StorageBlocked("storage_blocked: live retained directory writer")
            if reverse:
                yield directory_record

        yield from visit(root, ".")
        _verify_pinned_directory(custody_root, root)


def _accounted_inventory(custody_root, workspace, *, reverse=False):
    # Store only linked inode identities, never an all-path inventory. Two
    # traversals suffice even for many aliases; the second authenticates the
    # same inode metadata and emits the original forward/reverse row order.
    linked = {}
    for record in _inventory(custody_root, workspace, hashes=False):
        if stat.S_ISDIR(record["mode"]):
            continue
        if type(record["links"]) is not int or record["links"] < 1:
            raise StorageBlocked("storage_blocked: malformed retained inode links")
        if record["links"] == 1:
            continue
        key = record["device"], record["inode"]
        identity = {name: value for name, value in record.items() if name != "path"}
        if key not in linked:
            if len(linked) == _LINKED_INODES_LIMIT:
                raise StorageBlocked("storage_blocked: linked inode inventory exceeds capacity")
            linked[key] = {"first": record["path"], "identity": identity, "count": 0, "seen": 0}
        entry = linked[key]
        if identity != entry["identity"]:
            raise StorageBlocked("storage_blocked: retained linked inode changed")
        entry["count"] += 1
        entry["first"] = min(entry["first"], record["path"])
    if any(entry["count"] != entry["identity"]["links"] for entry in linked.values()):
        raise StorageBlocked("storage_blocked: retained inode has external aliases")
    for record in _inventory(custody_root, workspace, reverse=reverse):
        key = record["device"], record["inode"]
        if not stat.S_ISDIR(record["mode"]) and (record["links"] != 1 or key in linked):
            entry = linked.get(key)
            identity = {name: value for name, value in record.items() if name != "path"}
            if entry is None or identity != entry["identity"]:
                raise StorageBlocked("storage_blocked: retained linked inode changed")
            entry["seen"] += 1
            if record["path"] != entry["first"]:
                record["allocated"] = 0
        yield record
    if any(entry["seen"] != entry["count"] for entry in linked.values()):
        raise StorageBlocked("storage_blocked: retained linked aliases changed")


def _snapshot(
    custody_root,
    workspace,
    policy,
    *,
    publish=None,
    control_dir=None,
    previous=None,
    removed_paths=(),
):
    return _retained.build_snapshot(
        _accounted_inventory(custody_root, workspace),
        policy,
        control_dir=publish if publish is not None else control_dir,
        previous=previous,
        removed_paths=removed_paths,
        publish=publish is not None,
    )


def _inventory_output_bound(records, policy):
    return _retained.initial_bound(records, policy)


def _authority(budget):
    state = budget._state()
    engineering = state.get("engineering")
    if engineering is None:
        _require_single_authority(budget.workspace)
        for parent in budget.workspace.parents:
            with _pinned_directory(parent) as directory:
                raw = _read_authority_at(directory)
            if raw is not None:
                raise StorageBlocked("storage_blocked: engineering bootstrap is incomplete")
        return None
    custody = Path(engineering["custody_root"])
    with _pinned_directory(custody) as root, _pinned_directory(budget.workspace) as workspace:
        raw = _read_at(root, _AUTHORITY, max_bytes=16384)
        authority = json.loads(raw)
        if (
            authority.get("schema_version") != "phase4-engineering-authority-v2"
            or authority.get("inventory_schema") != "phase4-engineering-snapshot-v2"
        ):
            raise StorageBlocked(
                "storage_blocked: legacy engineering authority requires "
                "explicit same-ledger transition"
            )
        if (
            sha256_bytes(raw) != engineering["authority_sha256"]
            or authority["workspace"] != str(budget.workspace)
            or authority["custody_device"] != os.fstat(root).st_dev
            or authority["custody_inode"] != os.fstat(root).st_ino
            or authority["workspace_device"] != os.fstat(workspace).st_dev
            or authority["workspace_inode"] != os.fstat(workspace).st_ino
        ):
            raise StorageBlocked("storage_blocked: engineering authority or root identity changed")
    initial, current = engineering.get("initial_snapshot"), engineering.get("snapshot")
    _retained._descriptor(initial, budget.policy)
    _retained._descriptor(current, budget.policy)
    if any(
        initial[name] != current[name] for name in ("pages", "layout_sha256", "original_entries")
    ):
        raise StorageBlocked("storage_blocked: engineering original partition layout changed")
    return engineering


def _retained_charge(budget):
    try:
        engineering = _authority(budget)
        if engineering is None:
            return 0
        expected = engineering["snapshot"]
        if "pending" in engineering:
            current = _validate_pending_residual(budget, engineering)
            return max(expected["allocated"], current)
        observed = _snapshot(
            Path(engineering["custody_root"]),
            budget.workspace,
            budget.policy,
            control_dir=budget.root,
            previous=expected,
        )
        if observed != expected:
            raise StorageBlocked("storage_blocked: frozen engineering inventory changed")
        return observed["allocated"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise StorageBlocked("storage_blocked: missing or invalid engineering inventory") from error


def bootstrap_engineering_workspace(
    *, custody_root: Path, workspace: Path, policy: ArchivePolicy, source_commit: str
) -> _StorageBudget:
    """Bind one frozen plan to its sole live child, or attach a compatible ledger.

    This is a controller-only operation after source review and writer freeze.
    Bootstrap admits its own finite inventory output before creating any object.
    Existing ledgers retain every baseline, reservation, binding and remote byte.
    Interrupted bootstrap remains fail-closed; it is never silently retried/reset.
    """
    from silent_cascade.archive.ledger import initialize_workspace_ledger

    custody_root, workspace = custody_root.absolute(), workspace.absolute()
    policy = ArchivePolicy.model_validate(policy.model_dump())
    if workspace.parent != custody_root or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("engineering workspace must be one direct child with exact source commit")
    revision = (
        subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=Path(__file__).absolute().parents[3],
            check=True,
            capture_output=True,
            timeout=10,
        )
        .stdout.decode()
        .strip()
    )
    if revision != source_commit:
        raise StorageBlocked(
            "storage_blocked: engineering source revision differs from executable checkout"
        )
    with _pinned_directory(custody_root) as root, ExitStack() as admission:
        try:
            os.stat(_AUTHORITY, dir_fd=root, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise StorageBlocked("storage_blocked: prior engineering authority/history exists")
        _require_single_authority(workspace)
        initial = _snapshot(custody_root, workspace, policy)
        volume = os.statvfs(custody_root)
        block = _allocation_unit(volume)

        def bootstrap_records():
            yield from _accounted_inventory(custody_root, workspace)
            anchor = _stat_record(_AUTHORITY, os.fstat(root))
            anchor["sha256"] = "f" * 64
            yield anchor

        inventory_bytes, inventory_pages = _inventory_output_bound(bootstrap_records(), policy)
        bootstrap_bytes = inventory_bytes + inventory_pages * 2 * block
        # Authority's existing reader cap; full compatible descriptor and its
        # atomic replacement use the policy cap, even for a large prior ledger.
        # Seven directories, four control leaves and two atomic directory slots.
        # Leaves/index/generations are separate private retained subdirectories.
        bootstrap_bytes += 16384 + policy.page_bytes * 2 + 13 * block
        # The external authority leaf and its parent remain retained custody;
        # all other bootstrap output belongs to the existing metadata category.
        metadata_peak = bootstrap_bytes - 16384 - 2 * block
        prior = None
        if workspace.exists():
            prior = _StorageBudget(workspace=workspace, policy=policy)
            admission.enter_context(
                _lock(
                    control_dir=prior.root,
                    relative=("workspace.lock",),
                    shared=False,
                    blocking=True,
                )
            )
            prior.check()
        operational, remaining, metadata = 0, 0, 0
        if prior is not None:
            allocated = prior.measure()
            operational = sum(allocated.values())
            metadata = allocated["metadata"]
            for reservation in prior._state()["reservations"].values():
                if reservation["amounts"].get("metadata", 0):
                    raise StorageBlocked(
                        "storage_blocked: bootstrap metadata capacity is owned by another admission"
                    )
                remaining += sum(
                    max(0, amount - max(0, allocated[name] - reservation["before"][name]))
                    for name, amount in reservation["amounts"].items()
                )
        if metadata + metadata_peak > policy.metadata_bytes:
            raise StorageBlocked("storage_blocked: bootstrap metadata peak exceeds category")
        operational += remaining
        if (
            initial["allocated"] + operational + bootstrap_bytes
            > (policy.workspace_bytes - policy.reserve_bytes - policy.emergency_bytes)
            or volume.f_bavail * volume.f_frsize
            < bootstrap_bytes + remaining + policy.reserve_bytes
        ):
            raise StorageBlocked("storage_blocked: bootstrap normal/physical space unavailable")
        if _snapshot(custody_root, workspace, policy) != initial:
            raise StorageBlocked("storage_blocked: retained writer changed bootstrap inventory")
        workspace.mkdir(mode=0o700, exist_ok=True)
        with _pinned_directory(workspace) as opened:
            root_info, work_info = os.fstat(root), os.fstat(opened)
            if work_info.st_dev != root_info.st_dev:
                raise StorageBlocked("storage_blocked: engineering workspace crosses device")
            authority = {
                "schema_version": "phase4-engineering-authority-v2",
                "inventory_schema": "phase4-engineering-snapshot-v2",
                "workspace": str(workspace),
                "source_commit": source_commit,
                "custody_device": root_info.st_dev,
                "custody_inode": root_info.st_ino,
                "workspace_device": work_info.st_dev,
                "workspace_inode": work_info.st_ino,
                "completed_tasks": list(_TASKS),
                "executable_sha256": _executable_digest(),
            }
            raw = canonical_json_bytes(authority)
            _create_at(root, _AUTHORITY, raw)
            authority_descriptor = os.open(_AUTHORITY, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root)
            try:
                os.fchmod(authority_descriptor, 0o600)
                os.fsync(authority_descriptor)
            finally:
                os.close(authority_descriptor)
        if prior is None:
            initialize_workspace_ledger(workspace_root=workspace, policy=policy, baseline=())
        budget = _StorageBudget(workspace=workspace, policy=policy)
        # Precreate the closed operation lock within this admitted bootstrap.
        with _lock(
            control_dir=budget.root, relative=("engineering.lock",), shared=False, blocking=True
        ):
            pass
        with (
            nullcontext()
            if prior is not None
            else _lock(
                control_dir=budget.root, relative=("workspace.lock",), shared=False, blocking=True
            )
        ):
            state = budget._state()
            snapshot = _snapshot(custody_root, workspace, policy, publish=budget.root)
            if snapshot != _snapshot(
                custody_root, workspace, policy, control_dir=budget.root, previous=snapshot
            ):
                raise StorageBlocked("storage_blocked: bootstrap inventory changed")
            state["schema_version"] = "phase4-r2-workspace-v2"
            state["engineering"] = {
                "custody_root": str(custody_root),
                "authority_sha256": sha256_bytes(raw),
                "initial_snapshot": snapshot,
                "snapshot": snapshot,
            }
            budget._store(state)
        budget.check()
        return budget


def _package_digest(package: Path, paths: tuple[Path, ...] | None = None) -> str:
    paths = tuple(sorted(package.rglob("*.py"))) if paths is None else paths
    digest = hashlib.sha256()
    for path in paths:
        with _pinned_directory(path.parent) as directory:
            raw = _read_at(directory, path.name, max_bytes=16 * 1024**2)
        digest.update(
            canonical_json_bytes(
                {"path": path.relative_to(package).as_posix(), "sha256": sha256_bytes(raw)}
            )
        )
    return digest.hexdigest()


def _executable_digest(*, package: Path | None = None):
    """Hash the actual loaded package source closure, independently of caller labels."""
    package = Path(__file__).absolute().parents[1] if package is None else package.absolute()
    return _package_digest(package)


def _git_head(repo_root: Path) -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise StorageBlocked("storage_blocked: source revision is unavailable") from error
    return TypeAdapter(Revision).validate_python(revision)


def _current_source_identity() -> _SourceIdentity:
    return _SourceIdentity(
        _git_head(Path(__file__).absolute().parents[3]),
        _executable_digest(),
    )


def _committed_source_identity(
    *, repo_root: Path | None = None, package: Path | None = None
) -> _SourceIdentity:
    """Authenticate the exact working package against regular blobs at Git HEAD."""
    repo_root = Path(__file__).absolute().parents[3] if repo_root is None else repo_root.absolute()
    package = Path(__file__).absolute().parents[1] if package is None else package.absolute()
    try:
        prefix = package.relative_to(repo_root).as_posix()
    except ValueError as error:
        raise StorageBlocked("storage_blocked: package escapes source repository") from error
    revision = _git_head(repo_root)
    try:
        raw_tree = subprocess.run(
            ["git", "ls-tree", "-r", "-z", revision, "--", prefix],
            cwd=repo_root,
            check=True,
            capture_output=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise StorageBlocked(
            "storage_blocked: committed source inventory is unavailable"
        ) from error
    committed = set()
    try:
        for entry in (item for item in raw_tree.split(b"\0") if item):
            metadata, encoded = entry.split(b"\t", 1)
            mode, object_type, _object_id = metadata.decode("ascii").split(" ", 2)
            path = encoded.decode("utf-8")
            if path.endswith(".py"):
                if mode not in {"100644", "100755"} or object_type != "blob":
                    raise ValueError("package source is not a regular Git blob")
                committed.add(path)
    except (UnicodeDecodeError, ValueError) as error:
        raise StorageBlocked("storage_blocked: committed source inventory is malformed") from error
    actual = tuple(sorted(package.rglob("*.py")))
    if {path.relative_to(repo_root).as_posix() for path in actual} != committed:
        raise StorageBlocked("storage_blocked: package source is not exactly committed")
    digest = hashlib.sha256()
    for path in actual:
        relative = path.relative_to(repo_root).as_posix()
        try:
            expected = subprocess.run(
                ["git", "cat-file", "blob", f"{revision}:{relative}"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                timeout=30,
            ).stdout
            with _pinned_directory(path.parent) as directory:
                observed = _read_at(directory, path.name, max_bytes=16 * 1024**2)
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            raise StorageBlocked(
                "storage_blocked: committed package source is unavailable"
            ) from error
        if observed != expected:
            raise StorageBlocked("storage_blocked: package source is not exactly committed")
        digest.update(
            canonical_json_bytes(
                {"path": path.relative_to(package).as_posix(), "sha256": sha256_bytes(observed)}
            )
        )
    executable = digest.hexdigest()
    final_actual = tuple(sorted(package.rglob("*.py")))
    final_executable = _package_digest(package, final_actual)
    post_hash_actual = tuple(sorted(package.rglob("*.py")))
    if (
        final_actual != actual
        or final_executable != executable
        or post_hash_actual != final_actual
        or _git_head(repo_root) != revision
    ):
        raise StorageBlocked("storage_blocked: package source changed during authentication")
    return _SourceIdentity(revision, executable)


def _resolve_reviewed_source(
    *, budget, source_commit: str, authority_sha256: str | None, allow_pending: bool
) -> _ReviewedSource:
    source_commit = TypeAdapter(Revision).validate_python(source_commit)
    if authority_sha256 is not None:
        authority_sha256 = TypeAdapter(Hash).validate_python(authority_sha256)
    budget.check()
    engineering = _authority(budget)
    if engineering is None or ("pending" in engineering and not allow_pending):
        raise StorageBlocked("storage_blocked: clean engineering authority is unavailable")
    anchor_raw = _control_reader(Path(engineering["custody_root"]))(_AUTHORITY, 16384)
    anchor = json.loads(anchor_raw)
    current = _current_source_identity()
    if current.source_commit != source_commit:
        raise ValueError("source revision differs from executable checkout")
    if (
        source_commit == anchor["source_commit"]
        and current.executable_sha256 == anchor["executable_sha256"]
    ):
        if authority_sha256 not in {None, engineering["authority_sha256"]}:
            raise ValueError("bootstrap source authority hash differs")
        return _ReviewedSource(
            source_commit, current.executable_sha256, engineering["authority_sha256"]
        )
    if authority_sha256 == engineering["authority_sha256"]:
        raise ValueError("source differs from immutable bootstrap authority")
    if authority_sha256 is None:
        raise ValueError("changed source requires an explicit reviewed-source hash")
    path = budget.root / "source-authorities" / current.executable_sha256 / f"{source_commit}.json"
    with _pinned_directory(path.parent) as directory:
        raw = _read_at(directory, path.name, max_bytes=_SOURCE_AUTHORITY_LIMIT)
    if sha256_bytes(raw) != authority_sha256:
        raise ValueError("reviewed-source content hash differs")
    try:
        record = _ReviewedSourceRecord.model_validate_json(raw)
    except Exception as error:
        raise ValueError("reviewed-source record is invalid") from error
    if canonical_json_bytes(record) != raw:
        raise ValueError("reviewed-source record is not canonical")
    if (
        record.anchor_authority_sha256 != engineering["authority_sha256"]
        or record.source_commit != source_commit
        or record.executable_sha256 != current.executable_sha256
        or record.policy_sha256 != sha256_bytes(canonical_json_bytes(budget.policy))
    ):
        raise ValueError("reviewed-source binding differs")
    return _ReviewedSource(source_commit, current.executable_sha256, authority_sha256)


def _reviewed_source(
    *, budget, source_commit: str, authority_sha256: str | None = None
) -> _ReviewedSource:
    return _resolve_reviewed_source(
        budget=budget,
        source_commit=source_commit,
        authority_sha256=authority_sha256,
        allow_pending=False,
    )


def publish_reviewed_source(
    *,
    budget,
    source_commit: str,
    executable_sha256: str,
    review_sha256: str,
    failure_proof_sha256: str | None = None,
) -> str:
    """Publish one bounded exact release authority; never select or replace one."""
    source_commit = TypeAdapter(Revision).validate_python(source_commit)
    executable_sha256 = TypeAdapter(Hash).validate_python(executable_sha256)
    review_sha256 = TypeAdapter(Hash).validate_python(review_sha256)
    if failure_proof_sha256 is not None:
        failure_proof_sha256 = TypeAdapter(Hash).validate_python(failure_proof_sha256)
    committed = _committed_source_identity()
    if committed != _SourceIdentity(source_commit, executable_sha256):
        raise ValueError("reviewed source arguments differ from committed package source")
    with _lock(
        control_dir=budget.root, relative=("engineering.lock",), shared=False, blocking=True
    ):
        engineering = _authority(budget)
        if engineering is None or "pending" in engineering:
            raise StorageBlocked(
                "storage_blocked: pending engineering eviction blocks source review"
            )
        budget.check()
        engineering = _authority(budget)
        if engineering is None or "pending" in engineering:
            raise StorageBlocked(
                "storage_blocked: pending engineering eviction blocks source review"
            )
        anchor_raw = _control_reader(Path(engineering["custody_root"]))(_AUTHORITY, 16384)
        if sha256_bytes(anchor_raw) != engineering["authority_sha256"]:
            raise StorageBlocked("storage_blocked: engineering anchor changed")
        record = _ReviewedSourceRecord(
            anchor_authority_sha256=engineering["authority_sha256"],
            source_commit=source_commit,
            executable_sha256=executable_sha256,
            policy_sha256=sha256_bytes(canonical_json_bytes(budget.policy)),
            review_sha256=review_sha256,
            failure_proof_sha256=failure_proof_sha256,
        )
        raw = canonical_json_bytes(record)
        if len(raw) > _SOURCE_AUTHORITY_LIMIT:
            raise StorageBlocked("storage_blocked: reviewed-source record exceeds limit")
        digest = sha256_bytes(raw)
        block = _allocation_unit(os.statvfs(budget.workspace))
        allocation = ((len(raw) + block - 1) // block) * block + 4 * block
        parent = budget.root / "source-authorities" / executable_sha256
        with budget.reserve(admission={"reviewed_source": digest}, metadata=allocation):
            with _pinned_directory(parent, create=True) as directory:
                try:
                    existing = _read_at(
                        directory, f"{source_commit}.json", max_bytes=_SOURCE_AUTHORITY_LIMIT
                    )
                except FileNotFoundError:
                    try:
                        _create_at(directory, f"{source_commit}.json", raw)
                    except FileExistsError:
                        existing = _read_at(
                            directory,
                            f"{source_commit}.json",
                            max_bytes=_SOURCE_AUTHORITY_LIMIT,
                        )
                    else:
                        existing = raw
                if existing != raw:
                    raise FileExistsError("different reviewed-source record already exists")
                reread = _read_at(
                    directory, f"{source_commit}.json", max_bytes=_SOURCE_AUTHORITY_LIMIT
                )
                if reread != raw or sha256_bytes(reread) != digest:
                    raise ValueError("reviewed-source publication readback differs")
            budget.check()
        return digest


def _stored_records(budget, snapshot):
    yield from _retained.read_records(budget.root, snapshot, budget.policy, reverse=True)


@dataclass(frozen=True)
class EngineeringCandidate:
    candidate_id: str
    source_commit: str
    executable_sha256: str
    source_authority_sha256: str
    logical_root: str
    members: tuple[FileEntry, ...]


def iter_engineering_candidates(
    budget, *, source_commit: str | None = None, source_authority_sha256: str | None = None
):
    """Yield filesystem-safe children only; this does NOT authorize their upload.

    Positive privacy/producer-format custody is separately required for the exact
    selected files. All unselected ordinary siblings remain authenticated/local.
    Mixed special/sparse/linked trees are searched only for disjoint safe children.
    """
    budget.check()
    engineering = _authority(budget)
    if engineering is None or "pending" in engineering:
        raise StorageBlocked("storage_blocked: frozen engineering authority is unavailable")
    authority = json.loads(_control_reader(Path(engineering["custody_root"]))(_AUTHORITY, 16384))
    source_commit = authority["source_commit"] if source_commit is None else source_commit
    source = _reviewed_source(
        budget=budget,
        source_commit=source_commit,
        authority_sha256=source_authority_sha256,
    )

    def candidates(prefix):
        members, eligible, exists = [], True, False
        for record in _stored_records(budget, engineering["snapshot"]):
            if record["path"] == prefix:
                exists = stat.S_ISDIR(record["mode"])
            if not record["path"].startswith(prefix + "/"):
                continue
            if stat.S_ISDIR(record["mode"]):
                continue
            if record["sha256"] is None or len(members) == budget.policy.page_entries:
                eligible = False
            elif eligible:
                members.append(FileEntry(record["path"], record["sha256"], record["bytes"]))
        if exists and eligible and members:
            members = tuple(sorted(members, key=lambda item: item.path))
            identity = {
                "anchor_authority": engineering["authority_sha256"],
                "inventory": engineering["snapshot"]["head"],
                "root": prefix,
                "source_commit": source.source_commit,
                "executable": source.executable_sha256,
                "source_authority": source.source_authority_sha256,
                "members": [asdict(member) for member in members],
            }
            yield EngineeringCandidate(
                sha256_bytes(canonical_json_bytes(identity)),
                source.source_commit,
                source.executable_sha256,
                source.source_authority_sha256,
                prefix,
                members,
            )
        elif exists:
            # Retained records are already paged; never materialize a directory list.
            for record in _stored_records(budget, engineering["snapshot"]):
                if (
                    stat.S_ISDIR(record["mode"])
                    and Path(record["path"]).parent.as_posix() == prefix
                ):
                    yield from candidates(record["path"])

    for task in _TASKS:
        yield from candidates(f"tmp/{task}")


def _validated_review(budget, candidate, review):
    if not isinstance(candidate, EngineeringCandidate) or not isinstance(
        review, EngineeringContentReview
    ):
        raise ValueError("exact engineering content review is required")
    review = EngineeringContentReview.model_validate(review.model_dump())
    source = _reviewed_source(
        budget=budget,
        source_commit=candidate.source_commit,
        authority_sha256=candidate.source_authority_sha256,
    )
    if (
        review.candidate_id != candidate.candidate_id
        or review.executable_sha256 != candidate.executable_sha256
        or candidate.executable_sha256 != source.executable_sha256
        or candidate.source_authority_sha256 != source.source_authority_sha256
        or not set(review.paths) <= {member.path for member in candidate.members}
    ):
        raise ValueError("content review inventory/source/paths differ")
    if not any(
        actual == candidate
        for actual in iter_engineering_candidates(
            budget,
            source_commit=candidate.source_commit,
            source_authority_sha256=candidate.source_authority_sha256,
        )
    ):
        raise StorageBlocked("storage_blocked: stale or forged engineering candidate")
    selected = tuple(member for member in candidate.members if member.path in review.paths)
    if sum(member.bytes for member in selected) > budget.policy.logs_bytes:
        raise StorageBlocked("storage_blocked: diagnostic unit exceeds existing log-kind limit")
    return selected


def _validate_pending_residual(budget, engineering):
    pending = engineering["pending"]
    removed = set(pending["paths"])
    changed_directories = {"."}
    for path in removed:
        changed_directories.update(parent.as_posix() for parent in Path(path).parents)
    actual = iter(
        _accounted_inventory(Path(engineering["custody_root"]), budget.workspace, reverse=True)
    )
    current = next(actual, None)
    allocated = 0
    for expected in _stored_records(budget, engineering["snapshot"]):
        if expected["path"] in removed and (current is None or current["path"] != expected["path"]):
            continue
        if current is None:
            raise StorageBlocked("storage_blocked: unrecorded retained removal")
        if expected["path"] in changed_directories and stat.S_ISDIR(expected["mode"]):
            if any(current[key] != expected[key] for key in ("path", "device", "inode", "mode")):
                raise StorageBlocked("storage_blocked: retained directory identity changed")
        elif current != expected:
            raise StorageBlocked("storage_blocked: retained residual changed outside eviction")
        allocated += current["allocated"]
        current = next(actual, None)
    if current is not None:
        raise StorageBlocked("storage_blocked: unrecorded retained addition")
    return allocated


@dataclass(frozen=True)
class EngineeringArchiveResult:
    ref: UnitRef
    receipt_sha256: str
    run_id: str


@dataclass(frozen=True)
class _ArchiveUnitBounds:
    inventory: int
    manifest: int
    run_catalog: int
    operational_catalog: int
    receipt: int
    pending: int
    objects: int
    run_nodes: int
    operational_nodes: int
    operational_records: int
    shards: int
    record_bytes: int
    node_bytes: int
    remote: int


def _archive_unit_bounds(
    *,
    policy,
    logical_root,
    selected,
    run_id,
    kind,
    episode_groups,
    prior_operational_records,
    local_reservations,
    completed_count,
    prior_run_records=0,
):
    """Finite serializer/cardinality bound, including both catalog generations.

    Binary catalog trees contain at most 2N-1 nodes. Changed-path publication
    stages no more than a complete tree; each record occurs in one leaf. Use the
    authenticated prior operational count, not a new prefix's empty run index.
    All field/string maxima below come from current strict schemas and readers.
    """
    from silent_cascade.archive.catalog import _validate_member
    from silent_cascade.archive.types import MAX_INTEGER, EpisodeGroup, UnitIdentity

    policy = ArchivePolicy.model_validate(policy.model_dump())
    UnitIdentity(
        run_id=run_id,
        source_commit="f" * 40,
        config_sha256="f" * 64,
        evidence_identity_sha256="f" * 64,
        checkpoint_sha256=None,
        writer_stopped=True,
        checkpoint_committed=False,
    )
    if kind not in {"diagnostic", "episode_pack"} or not selected:
        raise ValueError("invalid bounded archive unit")
    if any(
        type(value) is not int or not 0 <= value <= MAX_INTEGER
        for value in (
            prior_operational_records,
            local_reservations,
            completed_count,
            prior_run_records,
        )
    ):
        raise ValueError("invalid bounded archive cardinality")
    for item in selected:
        _validate_member(item.path, logical_root)
        if type(item.bytes) is not int or not 0 <= item.bytes <= MAX_INTEGER:
            raise ValueError("invalid bounded archive member size")
    paths = tuple(item.path for item in selected)
    if paths != tuple(sorted(set(paths))):
        raise ValueError("invalid bounded archive member order")
    groups = tuple(EpisodeGroup(paths=group, expanded_bytes=1024**3) for group in episode_groups)
    if (kind == "episode_pack") != bool(groups) or (
        groups and tuple(sorted(path for group in groups for path in group.paths)) != paths
    ):
        raise ValueError("invalid bounded archive episode groups")
    expanded = sum(member.bytes for member in selected)
    if expanded > MAX_INTEGER:
        raise ValueError("bounded archive size overflow")
    chunks = (expanded + policy.chunk_bytes - 1) // policy.chunk_bytes
    ancestors = {
        parent.as_posix()
        for member in selected
        for parent in Path(member.path).parents
        if parent.as_posix() != "."
    }
    run_records = prior_run_records + 1 + len(selected) + len(ancestors)
    run_nodes = 2 * run_records + 1  # two indexes and their root (conservative).
    integer = 2**63 - 1
    hash_value = "f" * 64
    # An inventory row has at most chunks+1 spans; exact logical paths are finite.
    span = {"chunk_index": integer, "offset": integer, "length": integer}
    inventory = sum(
        len(
            canonical_json_bytes(
                {
                    "path": item.path,
                    "sha256": hash_value,
                    "bytes": integer,
                    "spans": [span]
                    * ((item.bytes + policy.chunk_bytes - 1) // policy.chunk_bytes + 1),
                }
            )
        )
        + 1
        for item in selected
    )
    # A shard per member is an upper bound even when page byte limits split early.
    shards = len(selected)
    manifest = (
        len(
            canonical_json_bytes(
                {
                    "schema_version": "phase4-r2-unit-v1",
                    "kind": kind,
                    "logical_root": logical_root,
                    "identity": {
                        "run_id": run_id,
                        "source_commit": "f" * 40,
                        "config_sha256": hash_value,
                        "evidence_identity_sha256": hash_value,
                        "checkpoint_sha256": None,
                        "writer_stopped": True,
                        "checkpoint_committed": False,
                    },
                    "policy_sha256": hash_value,
                    "expanded_bytes": integer,
                    "file_count": integer,
                    "inventory_sha256": hash_value,
                    "inventory_shards": [
                        {
                            "path": "inventory.9223372036854775807.jsonl",
                            "sha256": hash_value,
                            "entries": integer,
                            "decoded_bytes": integer,
                        }
                    ]
                    * shards,
                    "chunks": [{"index": integer, "sha256": hash_value, "bytes": integer}] * chunks,
                    "episode_groups": [group.model_dump(mode="json") for group in groups],
                    "borrowed": [],
                }
            )
        )
        + 1
    )
    # UTF-8 JSON can escape every allowed 4096-character string to six bytes.
    max_path = "\u0001" * 4096
    record_bytes = max(
        len(canonical_json_bytes(record)) + 1
        for record in (
            {
                "record_type": "reservation",
                "key": hash_value,
                "object_key": max_path,
                "bytes": integer,
                "sha256": hash_value,
            },
            {
                "record_type": "eviction",
                "key": hash_value,
                "unit_id": hash_value,
                "receipt_sha256": hash_value,
                "intent_sha256": hash_value,
                "object_key": max_path,
                "bytes": integer,
                "completed": True,
            },
            {
                "record_type": "unit",
                "key": hash_value,
                "unit_id": hash_value,
                "kind": "diagnostic",
                "logical_root": max_path,
                "expanded_bytes": integer,
                "file_count": integer,
                "manifest_path": "units/" + hash_value + "/manifest.json",
            },
        )
    )
    # Every node is a leaf header plus records, or a branch with two bounded refs.
    ref = {
        "sha256": hash_value,
        "path": "catalog/nodes/" + hash_value + ".json",
        "entries": integer,
        "decoded_bytes": integer,
        "first_key": hash_value,
        "last_key": hash_value,
    }
    node_bytes = (
        len(
            canonical_json_bytes(
                {
                    "schema_version": "phase4-r2-catalog-node-v1",
                    "index": "reservations",
                    "depth": 255,
                    "records": [],
                    "zero": ref,
                    "one": ref,
                }
            )
        )
        + 1
    )
    run_catalog = run_nodes * node_bytes + run_records * record_bytes
    objects = chunks + shards + 1 + run_nodes
    operational_records = (
        prior_operational_records + local_reservations + objects + 1 + completed_count
    )
    operational_nodes = 2 * operational_records + 3
    operational_catalog = operational_nodes * node_bytes + operational_records * record_bytes
    # Transfer object descriptors, receipt proof, pending publication and cleanup
    # paths all use bounded keys/paths; allow each such field its schema maximum.
    object_descriptor = (
        len(
            canonical_json_bytes(
                {
                    "key": max_path,
                    "source": max_path,
                    "bytes": integer,
                    "sha256": hash_value,
                }
            )
        )
        + 1
    )
    receipt = objects * object_descriptor + run_nodes * (object_descriptor + node_bytes)
    receipt += len(canonical_json_bytes(policy)) + node_bytes
    pending = (operational_nodes + completed_count + 1) * object_descriptor
    pending += (local_reservations + objects + 4 * completed_count) * len(
        canonical_json_bytes({"path": max_path})
    )
    pending += 4 * node_bytes
    if max(manifest, receipt, pending) > 16 * 1024**2:
        raise StorageBlocked(
            "storage_blocked: derived prelude controls exceed existing reader limits"
        )
    remote = expanded + inventory + manifest + run_catalog + operational_catalog + receipt
    return _ArchiveUnitBounds(
        inventory,
        manifest,
        run_catalog,
        operational_catalog,
        receipt,
        pending,
        objects,
        run_nodes,
        operational_nodes,
        operational_records,
        shards,
        record_bytes,
        node_bytes,
        remote,
    )


def _prelude_bounds(budget, candidate, selected):
    from silent_cascade.archive.transport import (
        _completed_intents,
        _local_reservation_records,
        _opened_ledger,
    )

    policy = budget.policy
    control = budget.workspace / "control"
    transport_id = json.loads(_control_reader(control)("remote-reservations/state.json", 32768))[
        "transport_id"
    ]
    with _opened_ledger(control, transport_id=transport_id, policy=policy) as opened:
        remote_state = opened[0]
        prior = (
            0
            if remote_state["operational_head"] is None
            else remote_state["operational_head"]["entry_count"]
        )
    local_reservations = len(_local_reservation_records(control))
    completed = _completed_intents(control)
    sizing = _archive_unit_bounds(
        policy=policy,
        logical_root=candidate.logical_root,
        selected=selected,
        run_id="engineering-" + "f" * 64,
        kind="diagnostic",
        episode_groups=(),
        prior_operational_records=prior,
        local_reservations=local_reservations,
        completed_count=len(completed),
    )
    inventory, manifest = sizing.inventory, sizing.manifest
    run_catalog, operational_catalog = sizing.run_catalog, sizing.operational_catalog
    receipt, pending, objects = sizing.receipt, sizing.pending, sizing.objects
    run_nodes, operational_nodes = sizing.run_nodes, sizing.operational_nodes
    node_bytes, shards = sizing.node_bytes, sizing.shards
    expanded = sum(member.bytes for member in selected)
    integer, hash_value = 2**63 - 1, "f" * 64
    eviction = (
        sum(
            len(
                canonical_json_bytes(
                    {
                        "path": member.path,
                        "sha256": hash_value,
                        "bytes": integer,
                        "device": integer,
                        "inode": integer,
                        "mtime_ns": integer,
                    }
                )
            )
            + 1
            for member in candidate.members
        )
        + node_bytes
    )
    inventory_snapshot = budget._state()["engineering"]["snapshot"]
    new_inventory, retained_pages = _retained.publication_bound(
        budget.root, inventory_snapshot, policy, tuple(member.path for member in selected)
    )
    fixed_controls = 2 * policy.page_bytes  # ledger/pending state and their atomic replacement
    metadata = inventory + manifest + 2 * run_catalog + operational_catalog
    metadata += 2 * receipt + pending + 2 * eviction + new_inventory + fixed_controls
    metadata += (objects + local_reservations) * 32768  # remote reservation reader maximum
    metadata += sum(len(raw) for _entry, _path, raw, _receipt in completed)
    metadata += sum(
        len(canonical_json_bytes(record))
        for record in budget._state()["reservations"].values()
        if record["admission"] == {"engineering_candidate": candidate.candidate_id}
    )
    volume = os.statvfs(budget.workspace)
    block = _allocation_unit(volume)
    # Source-visible fixed leaves/directory closure; variable addressed leaves above.
    fixed_files = (
        "workspace.json",
        "workspace.lock",
        "engineering.lock",
        "archive.lock",
        "stripe.lock",
        "manifest.json",
        "state.json",
        "pending.json",
        "operational-pending.json",
        "active-receipt.json",
        "receipt.json",
        "eviction.json",
        "run-head.json",
        "recovered-admission.json",
    )
    fixed_directories = (
        "control",
        "locks",
        "unit-stripes",
        "units",
        "unit",
        "pending-unit",
        "remote-reservations",
        "reservation-objects",
        "receipts",
        "evictions",
        "active-receipts",
        "run-heads",
        "catalog-proofs",
        "proof-unit",
        "proof-generation",
        "proof-catalog",
        "proof-nodes",
        "proof-roots",
        "retained",
        "retained/leaves",
        "retained/index",
        "retained/generations",
        "engineering-recovery",
    )
    file_count = shards + run_nodes + objects + retained_pages + len(fixed_files)
    metadata += (file_count * 2 + len(fixed_directories) + 1) * block
    largest = max(
        min(expanded, policy.chunk_bytes),
        min(policy.page_bytes, max(manifest, receipt, run_catalog, operational_catalog)),
    )
    rounded = ((largest + block - 1) // block) * block
    scratch = 2 * (2 * rounded + block)
    # Catalog staging shares scratch with the one upload/readback buffer.
    scratch += run_catalog + operational_catalog
    staging_directories = (
        "transfer-scratch",
        "unit",
        "run-stage",
        "run-catalog",
        "run-nodes",
        "run-roots",
        "operational-stage",
        "operational",
        "operational-nodes",
        "operational-roots",
    )
    scratch += (2 * (run_nodes + operational_nodes + 2) + len(staging_directories)) * block
    return {"metadata": metadata, "scratch": scratch}


def archive_engineering_candidate(*, budget, candidate, review, transport):
    """Archive one content-reviewed diagnostic unit on the existing remote ledger."""
    from silent_cascade.archive.catalog import iter_unit_files, seal_unit
    from silent_cascade.archive.transport import archive_unit

    with _lock(
        control_dir=budget.root, relative=("engineering.lock",), shared=False, blocking=True
    ):
        selected = _validated_review(budget, candidate, review)
        bounds = _prelude_bounds(budget, candidate, selected)
        control = budget.workspace / "control"
        budget.bind(control, category="metadata")
        budget.bind(control / "transfer-scratch", category="scratch")
        engineering = _authority(budget)
        run_id = "engineering-" + candidate.candidate_id
        custody = Path(engineering["custody_root"])
        with _engineering_reservation(budget, candidate.candidate_id, bounds) as token:
            source = _reviewed_source(
                budget=budget,
                source_commit=candidate.source_commit,
                authority_sha256=candidate.source_authority_sha256,
            )
            ref = seal_unit(
                run_dir=custody,
                control_dir=control,
                logical_root=candidate.logical_root,
                paths=review.paths,
                kind="diagnostic",
                policy=budget.policy,
                identity={
                    "run_id": run_id,
                    "source_commit": source.source_commit,
                    "config_sha256": sha256_bytes(canonical_json_bytes(review)),
                    "evidence_identity_sha256": candidate.candidate_id,
                    "checkpoint_sha256": None,
                    "writer_stopped": True,
                    "checkpoint_committed": False,
                },
            )
            if tuple(iter_unit_files(control, ref)) != selected:
                raise StorageBlocked(
                    "storage_blocked: sealed files differ from reviewed frozen bytes"
                )
            receipt = archive_unit(
                run_dir=custody,
                control_dir=control,
                ref=ref,
                transport=transport,
                policy=budget.policy,
            )
            budget.check()
            pending = {
                "ref": asdict(ref),
                "receipt_sha256": receipt,
                "run_id": run_id,
                "paths": list(review.paths),
                "retained": [
                    asdict(member)
                    for member in candidate.members
                    if member.path not in review.paths
                ],
                "candidate": asdict(candidate),
                "review": review.model_dump(mode="json"),
                "reservation_token": token,
            }
            with _lock(
                control_dir=budget.root, relative=("workspace.lock",), shared=False, blocking=True
            ):
                state = budget._state()
                state["engineering"]["pending"] = pending
                budget._store(state)
            return _finish_engineering_eviction(budget)


def _finish_engineering_eviction(budget):
    from silent_cascade.archive.transport import evict_unit

    engineering = _authority(budget)
    pending = engineering["pending"]
    ref = UnitRef(**pending["ref"])
    budget.check()
    custody = Path(engineering["custody_root"])
    evict_unit(
        run_dir=custody,
        control_dir=budget.workspace / "control",
        ref=ref,
        receipt_sha256=pending["receipt_sha256"],
        retained=tuple(FileEntry(**entry) for entry in pending["retained"]),
    )
    _validate_pending_residual(budget, engineering)
    with _lock(control_dir=budget.root, relative=("workspace.lock",), shared=False, blocking=True):
        state = budget._state()
        snapshot = _snapshot(
            custody,
            budget.workspace,
            budget.policy,
            publish=budget.root,
            previous=engineering["snapshot"],
            removed_paths=tuple(pending["paths"]),
        )
        if snapshot != _snapshot(
            custody, budget.workspace, budget.policy, control_dir=budget.root, previous=snapshot
        ):
            raise StorageBlocked("storage_blocked: residual changed while publishing")
        _validate_pending_residual(budget, engineering)
        state["engineering"]["snapshot"] = snapshot
        del state["engineering"]["pending"]
        del state["reservations"][pending["reservation_token"]]
        budget._store(state)
    budget.check()
    return EngineeringArchiveResult(ref, pending["receipt_sha256"], pending["run_id"])


def resume_engineering_eviction(*, budget, transport):
    """Resume only the same authenticated pending eviction, retaining its old charge."""
    from silent_cascade.archive.transport import _opened_ledger, _transport_identity

    with _lock(
        control_dir=budget.root, relative=("engineering.lock",), shared=False, blocking=True
    ):
        engineering = _authority(budget)
        if engineering is None or "pending" not in engineering:
            raise StorageBlocked("storage_blocked: no engineering eviction to resume")
        with _opened_ledger(
            budget.workspace / "control",
            transport_id=_transport_identity(transport),
            policy=budget.policy,
        ):
            pass
        pending = engineering["pending"]
        value = pending["candidate"]
        candidate = EngineeringCandidate(
            candidate_id=value["candidate_id"],
            source_commit=value["source_commit"],
            executable_sha256=value["executable_sha256"],
            source_authority_sha256=value["source_authority_sha256"],
            logical_root=value["logical_root"],
            members=tuple(FileEntry(**entry) for entry in value["members"]),
        )
        review = EngineeringContentReview.model_validate_json(
            canonical_json_bytes(pending["review"])
        )
        source = _resolve_reviewed_source(
            budget=budget,
            source_commit=candidate.source_commit,
            authority_sha256=candidate.source_authority_sha256,
            allow_pending=True,
        )
        if (
            candidate.executable_sha256 != source.executable_sha256
            or candidate.source_authority_sha256 != source.source_authority_sha256
            or review.candidate_id != candidate.candidate_id
            or review.executable_sha256 != candidate.executable_sha256
            or tuple(pending["paths"]) != review.paths
        ):
            raise StorageBlocked("storage_blocked: pending engineering source or review changed")
        selected = tuple(member for member in candidate.members if member.path in review.paths)
        bounds = _prelude_bounds(budget, candidate, selected)
        with _engineering_reservation(budget, candidate.candidate_id, bounds) as token:
            with _lock(
                control_dir=budget.root, relative=("workspace.lock",), shared=False, blocking=True
            ):
                state = budget._state()
                state["engineering"]["pending"]["reservation_token"] = token
                budget._store(state)
            return _finish_engineering_eviction(budget)


@contextmanager
def _engineering_reservation(budget, candidate_id, bounds):
    """Reenter only this closed candidate's dead admission, retaining its arithmetic."""
    if budget.active_reservation is not None:
        raise StorageBlocked("storage_blocked: prelude requires its own isolated reservation")
    admission = {"engineering_candidate": candidate_id}
    with _lock(control_dir=budget.root, relative=("workspace.lock",), shared=False, blocking=True):
        state = budget._state()
        matches = [
            (token, record)
            for token, record in state["reservations"].items()
            if record["admission"] == admission
        ]
        if len(matches) > 1:
            raise StorageBlocked("storage_blocked: ambiguous engineering admission history")
        if matches:
            token, record = matches[0]
            if _process_identity(record["pid"]) == record["create_time"]:
                raise StorageBlocked("storage_blocked: engineering admission owner remains live")
            prior = canonical_json_bytes(record)
            previous_owner = record["pid"], record["create_time"]
            allocated = budget.measure()
            for name, amount in bounds.items():
                growth = max(0, allocated[name] - record["before"][name])
                record["amounts"][name] = max(record["amounts"].get(name, 0), growth + amount)
            record["pid"], record["create_time"] = os.getpid(), _process_identity(os.getpid())
            budget._store(state)
            try:
                budget.check()
            except BaseException:
                state["reservations"][token] = json.loads(prior)
                budget._store(state)
                raise
    if not matches:
        with budget.reserve(admission=admission, **bounds) as token:
            yield token
        return
    budget.active_reservation = token
    try:
        _create_control_object(
            budget.root, f"engineering-recovery/{sha256_bytes(prior)}.json", prior
        )
        yield token
    finally:
        budget.active_reservation = None
        with _lock(
            control_dir=budget.root, relative=("workspace.lock",), shared=False, blocking=True
        ):
            state = budget._state()
            if token in state["reservations"]:
                # An exception leaves every extended amount and physical byte
                # charged. The known old dead owner permits an explicit retry.
                state["reservations"][token]["pid"] = previous_owner[0]
                state["reservations"][token]["create_time"] = previous_owner[1]
                budget._store(state)
