"""Shared durable physical-space admission and immutable baseline accounting."""

import hashlib
import json
import os
import secrets
import stat
from contextlib import contextmanager
from dataclasses import asdict, dataclass
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


@dataclass(frozen=True, slots=True)
class _RetainedHistoryScope:
    workspace: str
    workspace_device: int
    policy_sha256: str
    authority_sha256: str | None
    snapshot_sha256: str | None
    retained_bytes: int
    reservation_token: str
    admission_sha256: str
    owner_pid: int
    owner_create_time: float
    child_inheritable: bool


@dataclass(frozen=True, slots=True)
class _ScopedAdmission:
    token: str
    before: dict[str, int]
    retained: _RetainedHistoryScope


def _scope_commitment(admission, identity):
    # An accidental-copy integrity binding inside the trusted fixed process graph,
    # not a signature or a defense against arbitrary in-process code.
    return sha256_bytes(canonical_json_bytes({"admission": admission, "scope": identity}))


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
        return self._check_accounting(retained_bytes=self.retained_charge())

    def _retained_identity(self):
        from silent_cascade.archive.preflight import _authority

        engineering = _authority(self)
        if engineering is not None and "pending" in engineering:
            raise StorageBlocked("storage_blocked: pending engineering eviction forbids scope")
        return {
            "workspace": str(self.workspace),
            "workspace_device": self.workspace.stat().st_dev,
            "policy_sha256": sha256_bytes(canonical_json_bytes(self.policy)),
            "authority_sha256": None if engineering is None else engineering["authority_sha256"],
            "snapshot_sha256": None
            if engineering is None
            else sha256_bytes(canonical_json_bytes(engineering["snapshot"])),
            "retained_bytes": 0 if engineering is None else engineering["snapshot"]["allocated"],
        }

    def check_scoped(self, scope) -> dict[str, int]:
        try:
            if type(scope) is not _RetainedHistoryScope:
                raise ValueError("invalid scope type")
            identity = asdict(scope)
            commitment = identity.pop("admission_sha256")
            for name in ("workspace", "policy_sha256", "reservation_token", "admission_sha256"):
                if type(getattr(scope, name)) is not str:
                    raise ValueError("invalid scope text")
            for name in ("authority_sha256", "snapshot_sha256"):
                if getattr(scope, name) is not None and type(getattr(scope, name)) is not str:
                    raise ValueError("invalid scope digest")
            for name in ("workspace_device", "retained_bytes", "owner_pid"):
                if type(getattr(scope, name)) is not int or getattr(scope, name) < 0:
                    raise ValueError("invalid scope number")
            if (
                type(scope.owner_create_time) is not float
                or type(scope.child_inheritable) is not bool
            ):
                raise ValueError("invalid scope owner")
            if any(identity[name] != value for name, value in self._retained_identity().items()):
                raise ValueError("retained identity changed")
            if scope.workspace_device != self.device:
                raise ValueError("workspace device changed")
            record = self._state()["reservations"][scope.reservation_token]
            if (
                record["pid"] != scope.owner_pid
                or record["create_time"] != scope.owner_create_time
                or _process_identity(scope.owner_pid) != scope.owner_create_time
                or (os.getpid() != scope.owner_pid and not scope.child_inheritable)
                or commitment != _scope_commitment(record["admission"], identity)
            ):
                raise ValueError("scope admission or owner changed")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise StorageBlocked(
                "storage_blocked: retained-history scope is invalid or expired"
            ) from error
        return self._check_accounting(retained_bytes=scope.retained_bytes)

    def _check_accounting(self, *, retained_bytes: int) -> dict[str, int]:
        allocated = self.measure()
        retained = retained_bytes
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
    def _scoped_reservation(self, *, admission, child_inheritable=False, **amounts):
        if self.active_reservation is not None or type(child_inheritable) is not bool:
            raise StorageBlocked("storage_blocked: scoped admission requires a fresh owner")
        with self._reservation(
            admission=admission, child_inheritable=child_inheritable, **amounts
        ) as admitted:
            yield admitted

    @contextmanager
    def reserve(self, *, admission=None, _scope=None, **amounts: int):
        with self._reservation(admission=admission, _scope=_scope, **amounts) as token:
            yield token

    @contextmanager
    def _reservation(self, *, admission=None, _scope=None, child_inheritable=None, **amounts):
        if any(
            name not in _CATEGORIES or type(value) is not int or value < 0
            for name, value in amounts.items()
        ):
            raise ValueError("invalid storage reservation")
        if _scope is not None:
            self.check_scoped(_scope)
            if (
                self.active_reservation != _scope.reservation_token
                or _scope.owner_pid != os.getpid()
            ):
                raise StorageBlocked("storage_blocked: nested reservation is not owned")
        check = self.check if _scope is None else lambda: self.check_scoped(_scope)
        if self.active_reservation is not None and self.inherit_reservations:
            active = self._state()["reservations"][self.active_reservation]
            if any(value > active["amounts"].get(name, 0) for name, value in amounts.items()):
                raise StorageBlocked("storage_blocked: nested output exceeds job admission")
            check()
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
                    check()
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
                retained_identity = None if child_inheritable is None else self._retained_identity()
                self.check()
                if retained_identity is not None:
                    record = state["reservations"][token]
                    identity = retained_identity | {
                        "reservation_token": token,
                        "owner_pid": record["pid"],
                        "owner_create_time": record["create_time"],
                        "child_inheritable": child_inheritable,
                    }
                    scope = _RetainedHistoryScope(
                        **identity,
                        admission_sha256=_scope_commitment(record["admission"], identity),
                    )
                    self.check_scoped(scope)
                    admitted = _ScopedAdmission(token, dict(record["before"]), scope)
            except BaseException:
                del state["reservations"][token]
                self._store(state)
                raise
        try:
            previous_reservation = self.active_reservation
            self.active_reservation = token
            yield token if child_inheritable is None else admitted
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
