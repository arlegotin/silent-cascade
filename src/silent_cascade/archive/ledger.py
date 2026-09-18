"""Shared durable physical-space admission and immutable baseline accounting."""

import hashlib
import json
import os
import secrets
import stat
from contextlib import contextmanager
from pathlib import Path

import psutil

from silent_cascade.archive.catalog import (
    _control_reader,
    _create_control_object,
    _pinned_directory,
    _safe_logical_path,
    _source_file,
)
from silent_cascade.archive.transport import _lock, _replace_at
from silent_cascade.archive.types import ArchivePolicy, FileEntry
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes


class StorageBlocked(RuntimeError):
    """Evidence is retained; admission needs space or a healthy archive parent."""


_CATEGORIES = ("spool", "cache", "pinned", "metadata", "scratch", "logs", "emergency")
_STATE_DIRECTORY = ".silent-cascade-storage"


def _process_identity(pid: int) -> float | None:
    try:
        process = psutil.Process(pid)
        return None if process.status() == psutil.STATUS_ZOMBIE else process.create_time()
    except psutil.NoSuchProcess:
        return None
    except (psutil.AccessDenied, PermissionError) as error:
        raise StorageBlocked("storage_blocked: cannot inspect archive owner") from error


def _hash_file(root: Path, entry: FileEntry) -> int:
    digest = hashlib.sha256()
    member = _safe_logical_path(entry.path, field="authenticated file")
    with _source_file(root, member) as (descriptor, info):
        if info.st_size != entry.bytes:
            raise ValueError("authenticated file size differs")
        while data := os.read(descriptor, 1024 * 1024):
            digest.update(data)
        if digest.hexdigest() != entry.sha256:
            raise ValueError("authenticated file hash differs")
        return info.st_blocks * 512


def initialize_workspace_ledger(*, workspace_root: Path, policy: ArchivePolicy, baseline):
    """Create one durable shared ledger; baseline is explicit, exact and immutable."""
    workspace_root = workspace_root.absolute()
    from silent_cascade.archive.preflight import _require_single_authority

    _require_single_authority(workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    root = workspace_root / _STATE_DIRECTORY
    policy = ArchivePolicy.model_validate(policy.model_dump())
    # The directory itself is the create-only bootstrap/history sentinel, even
    # before a first baseline page or descriptor exists. Never reuse an old lock.
    with _pinned_directory(workspace_root) as descriptor:
        try:
            os.mkdir(_STATE_DIRECTORY, mode=0o700, dir_fd=descriptor)
        except FileExistsError as error:
            if (root / "workspace.json").exists():
                raise FileExistsError("workspace ledger already exists") from error
            raise StorageBlocked(
                "storage_blocked: prior bootstrap/history requires explicit recovery"
            ) from error
        os.fsync(descriptor)
    with _lock(control_dir=root, relative=("workspace.lock",), shared=False, blocking=True):
        if (root / "workspace.json").exists():
            raise FileExistsError("workspace ledger already exists")
        if (root / "baseline").exists():
            raise StorageBlocked("storage_blocked: interrupted ledger bootstrap requires recovery")
        head = None
        page = []
        previous = ""

        def publish():
            nonlocal head, page
            raw = canonical_json_bytes({"next": head, "entries": page})
            if len(raw) > policy.page_bytes:
                raise StorageBlocked("storage_blocked: baseline page exceeds policy")
            digest = sha256_bytes(raw)
            _create_control_object(root, f"baseline/{digest}.json", raw)
            head = digest
            page = []

        for entry in baseline:
            if (
                not isinstance(entry, FileEntry)
                or entry.path <= previous
                or entry.path.startswith(_STATE_DIRECTORY + "/")
            ):
                raise ValueError("baseline must be ordered unique scientific files")
            _hash_file(workspace_root, entry)
            previous = entry.path
            record = {"path": entry.path, "sha256": entry.sha256, "bytes": entry.bytes}
            if page and (
                len(page) >= policy.page_entries
                or len(canonical_json_bytes({"next": head, "entries": [*page, record]}))
                > policy.page_bytes
            ):
                publish()
            page.append(record)
        if page:
            publish()
        payload = {
            "schema_version": "phase4-r2-workspace-v1",
            "workspace": str(workspace_root),
            "device": workspace_root.stat().st_dev,
            "policy_sha256": sha256_bytes(canonical_json_bytes(policy)),
            "baseline": head,
            "reservations": {},
            "paths": {name: name for name in _CATEGORIES},
        }
        _create_control_object(root, "workspace.json", canonical_json_bytes(payload))


class _StorageBudget:
    """Count every inode below one workspace, including unknown stopped output."""

    def __init__(self, *, workspace: Path, policy: ArchivePolicy):
        self.workspace = workspace.absolute()
        self.policy = ArchivePolicy.model_validate(policy.model_dump())
        self.device = self.workspace.stat().st_dev
        self.root = self.workspace / _STATE_DIRECTORY
        self.active_reservation = None
        self.inherit_reservations = False
        self._state()

    def _state(self):
        state = json.loads(_control_reader(self.root)("workspace.json", self.policy.page_bytes))
        expected_keys = {
            "schema_version",
            "workspace",
            "device",
            "policy_sha256",
            "baseline",
            "reservations",
            "paths",
        }
        if state.get("schema_version") == "phase4-r2-workspace-v2":
            expected_keys.add("engineering")
        if set(state) != expected_keys or (
            state["schema_version"] not in {"phase4-r2-workspace-v1", "phase4-r2-workspace-v2"}
            or state["workspace"] != str(self.workspace)
            or state["device"] != self.device
            or state["policy_sha256"] != sha256_bytes(canonical_json_bytes(self.policy))
            or not isinstance(state["reservations"], dict)
        ):
            raise StorageBlocked("storage_blocked: workspace ledger identity differs")
        if type(state["paths"]) is not dict or any(
            category not in _CATEGORIES for category in state["paths"].values()
        ):
            raise StorageBlocked("storage_blocked: invalid category paths")
        try:
            for value in state["paths"]:
                _safe_logical_path(value, field="category root")
                if value.startswith(_STATE_DIRECTORY):
                    raise ValueError("reserved category root")
        except (TypeError, ValueError) as error:
            raise StorageBlocked("storage_blocked: invalid category paths") from error
        return state

    def retained_charge(self) -> int:
        """Authenticate frozen engineering bytes without changing category semantics."""
        from silent_cascade.archive.preflight import _retained_charge

        return _retained_charge(self)

    def bind(self, path: Path, *, category: str):
        self.require_path(path)
        relative = path.absolute().relative_to(self.workspace).as_posix()
        _safe_logical_path(relative, field="category root")
        if category not in _CATEGORIES or relative.startswith(_STATE_DIRECTORY):
            raise ValueError("invalid workspace category binding")
        with _lock(
            control_dir=self.root, relative=("workspace.lock",), shared=False, blocking=True
        ):
            state = self._state()
            if relative in state["paths"] and state["paths"][relative] != category:
                raise StorageBlocked("storage_blocked: category binding cannot be reassigned")
            if state["paths"].get(relative) == category:
                return
            if state["reservations"]:
                raise StorageBlocked("storage_blocked: category binding has live reservations")
            state["paths"][relative] = category
            self._store(state)

    def _store(self, state):
        raw = canonical_json_bytes(state)
        if len(raw) > self.policy.page_bytes:
            raise StorageBlocked("storage_blocked: reservation metadata limit")
        with _pinned_directory(self.root) as descriptor:
            _replace_at(descriptor, "workspace.json", raw)

    def require_path(self, path: Path) -> None:
        path = path.absolute()
        if ".." in path.parts or not path.is_relative_to(self.workspace):
            raise StorageBlocked("storage_blocked: output escapes counted workspace")
        for parent in (path, *path.parents):
            if not parent.is_relative_to(self.workspace):
                break
            try:
                info = parent.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(info.st_mode) or info.st_dev != self.device:
                raise StorageBlocked("storage_blocked: symlink or second volume in workspace")

    def measure(self) -> dict[str, int]:
        allocated = dict.fromkeys(_CATEGORIES, 0)
        state = self._state()
        bindings = sorted(
            state["paths"].items(), key=lambda pair: len(Path(pair[0]).parts), reverse=True
        )

        def binding_for(relative):
            if relative.parts and relative.parts[0] == _STATE_DIRECTORY:
                return None, "metadata"
            return next(
                (
                    (Path(prefix), category)
                    for prefix, category in bindings
                    if relative.is_relative_to(prefix)
                ),
                (None, "scratch"),
            )

        def category_for(relative):
            return binding_for(relative)[1]

        def classify(info, relative):
            prefix, category = binding_for(relative)
            if info.st_dev != self.device:
                raise StorageBlocked("storage_blocked: symlink or second volume in workspace")
            opaque = prefix is not None and category == "scratch" and relative != prefix
            if not opaque and stat.S_ISLNK(info.st_mode):
                raise StorageBlocked("storage_blocked: symlink or second volume in workspace")
            if not opaque and stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                raise StorageBlocked("storage_blocked: ambiguous hard-linked allocation")
            return category if relative.parts else "metadata"

        def scan_directory(descriptor, relative):
            directory_info = os.fstat(descriptor)
            category = classify(directory_info, relative)
            if not stat.S_ISDIR(directory_info.st_mode):
                raise StorageBlocked("storage_blocked: workspace changed during measurement")
            allocated[category] += directory_info.st_blocks * 512
            with os.scandir(descriptor) as entries:
                for entry in entries:
                    child_relative = relative / entry.name if relative.parts else Path(entry.name)
                    info = entry.stat(follow_symlinks=False)
                    category = classify(info, child_relative)
                    if not stat.S_ISDIR(info.st_mode):
                        allocated[category] += info.st_blocks * 512
                        continue
                    try:
                        child = os.open(
                            entry.name,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor,
                        )
                    except OSError as error:
                        raise StorageBlocked(
                            "storage_blocked: workspace changed during measurement"
                        ) from error
                    try:
                        opened = os.fstat(child)
                        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                            raise StorageBlocked(
                                "storage_blocked: workspace changed during measurement"
                            )
                        scan_directory(child, child_relative)
                        current = os.stat(entry.name, dir_fd=descriptor, follow_symlinks=False)
                        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
                            raise StorageBlocked(
                                "storage_blocked: workspace changed during measurement"
                            )
                    finally:
                        os.close(child)

        try:
            with _pinned_directory(self.workspace) as workspace:
                workspace_info = os.fstat(workspace)
                scan_directory(workspace, Path())
                with _pinned_directory(self.workspace) as current:
                    current_info = os.fstat(current)
                if (current_info.st_dev, current_info.st_ino) != (
                    workspace_info.st_dev,
                    workspace_info.st_ino,
                ):
                    raise StorageBlocked("storage_blocked: workspace changed during measurement")
        except OSError as error:
            raise StorageBlocked("storage_blocked: workspace changed during measurement") from error
        head = self._state()["baseline"]
        while head is not None:
            raw = _control_reader(self.root)(f"baseline/{head}.json", self.policy.page_bytes)
            if sha256_bytes(raw) != head:
                raise StorageBlocked("storage_blocked: corrupt baseline page")
            page = json.loads(raw)
            if set(page) != {"next", "entries"} or len(page["entries"]) > self.policy.page_entries:
                raise StorageBlocked("storage_blocked: malformed baseline page")
            for value in page["entries"]:
                entry = FileEntry(**value)
                try:
                    blocks = _hash_file(self.workspace, entry)
                except (OSError, ValueError) as error:
                    raise StorageBlocked("storage_blocked: immutable baseline differs") from error
                allocated[category_for(Path(entry.path))] -= blocks
            head = page["next"]
        return allocated

    def check(self) -> dict[str, int]:
        allocated = self.measure()
        retained = self.retained_charge()
        reservations = dict.fromkeys(_CATEGORIES, 0)
        for record in self._state()["reservations"].values():
            for name, amount in record["amounts"].items():
                if name not in reservations or type(amount) is not int or amount < 0:
                    raise StorageBlocked("storage_blocked: corrupt reservation")
                growth = max(0, allocated[name] - record["before"][name])
                if growth > amount:
                    raise StorageBlocked(
                        f"storage_blocked: {name} output exceeded admitted maximum"
                    )
                reservations[name] += max(0, amount - growth)
        for name, value in allocated.items():
            if value + reservations[name] > getattr(self.policy, f"{name}_bytes"):
                raise StorageBlocked(
                    f"storage_blocked: {name} category exhausted; retain pending files"
                )
        reserved = sum(reservations.values())
        normal = sum(allocated.values()) + reserved - allocated["emergency"]
        normal -= reservations["emergency"]
        if normal + retained > (
            self.policy.workspace_bytes - self.policy.reserve_bytes - self.policy.emergency_bytes
        ):
            raise StorageBlocked("storage_blocked: normal allocation exhausted by retained history")
        if (
            sum(allocated.values()) + reserved + retained + self.policy.reserve_bytes
            > self.policy.workspace_bytes
        ):
            raise StorageBlocked(
                "storage_blocked: global allocation exhausted; retain pending files"
            )
        volume = os.statvfs(self.workspace)
        if volume.f_bavail * volume.f_frsize < reserved + self.policy.reserve_bytes:
            raise StorageBlocked("storage_blocked: protected physical headroom unavailable")
        return allocated

    @contextmanager
    def reserve(self, *, admission=None, **amounts: int):
        if any(
            name not in _CATEGORIES or type(value) is not int or value < 0
            for name, value in amounts.items()
        ):
            raise ValueError("invalid storage reservation")
        if self.active_reservation is not None and self.inherit_reservations:
            active = self._state()["reservations"][self.active_reservation]
            if any(value > active["amounts"].get(name, 0) for name, value in amounts.items()):
                raise StorageBlocked("storage_blocked: nested output exceeds job admission")
            self.check()
            yield self.active_reservation
            return
        if self.active_reservation is not None:
            token = self.active_reservation
            with _lock(
                control_dir=self.root, relative=("workspace.lock",), shared=False, blocking=True
            ):
                state = self._state()
                active = state["reservations"][token]
                prior = dict(active["amounts"])
                for name, value in amounts.items():
                    active["amounts"][name] = active["amounts"].get(name, 0) + value
                self._store(state)
                try:
                    self.check()
                except BaseException:
                    active["amounts"] = prior
                    self._store(state)
                    raise
            try:
                yield token
            finally:
                with _lock(
                    control_dir=self.root, relative=("workspace.lock",), shared=False, blocking=True
                ):
                    state = self._state()
                    for name, value in amounts.items():
                        state["reservations"][token]["amounts"][name] -= value
                        if name not in prior and state["reservations"][token]["amounts"][name] == 0:
                            del state["reservations"][token]["amounts"][name]
                    self._store(state)
            return
        token = secrets.token_hex(16)
        with _lock(
            control_dir=self.root, relative=("workspace.lock",), shared=False, blocking=True
        ):
            state = self._state()
            for record in state["reservations"].values():
                for name, value in amounts.items():
                    if value and record["amounts"].get(name, 0):
                        raise StorageBlocked(
                            f"storage_blocked: {name} capacity is owned by another admission"
                        )
            state["reservations"][token] = {
                "pid": os.getpid(),
                "create_time": _process_identity(os.getpid()),
                "amounts": amounts,
                "before": self.measure(),
                "admission": admission,
            }
            self._store(state)
            try:
                self.check()
            except BaseException:
                del state["reservations"][token]
                self._store(state)
                raise
        try:
            previous_reservation = self.active_reservation
            self.active_reservation = token
            yield token
        finally:
            self.active_reservation = previous_reservation
            with _lock(
                control_dir=self.root, relative=("workspace.lock",), shared=False, blocking=True
            ):
                state = self._state()
                # A closed engineering completion may atomically release its
                # own reservation together with the verified residual charge.
                state["reservations"].pop(token, None)
                self._store(state)
