"""Closed Python writer admission for the offline diagnostic's known artifacts.

Install in a fresh child before consumer imports. This is an allowance supplied
by the caller, not a reservation service or a native syscall quota.
"""

import builtins
import io
import os
import re
import stat
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

_BLOCK = 4096
_DIRECTORIES = frozenset(
    (
        ".",
        "run",
        "run/eval",
        "run/eval/primary",
        "run/eval/primary/episodes",
        "run/eval/primary/crashes",
        "report",
        "matplotlib-cache",
        "cache",
    )
)
_PILOT = frozenset(
    (
        "executed-source.json",
        "step.json",
        "manifest.json",
        "offline.json",
        "report/trajectories-0.svg",
        "report/tables.json",
        "report/report.md",
        "report/report-index.json",
    )
)
_ATOMIC = frozenset(
    (
        "weights.safetensors",
        "replay.json",
        "run/eval/primary/identity.json",
        "run/eval/primary/retention.json",
        "run/eval/primary/index.json",
        "run/eval/primary/metrics.json",
        "run/eval/primary/DONE",
        "run/eval/primary/crashes/index.json",
    )
)
_ROWS = "run/eval/primary/.rows.pending.jsonl"
_ROWS_FINAL = "run/eval/primary/rows.jsonl"
_FONT = "matplotlib-cache/fontlist-v3.11.0.json"
_LOCK = _FONT + ".matplotlib-lock"
_PARENT = frozenset(("intent.json", "process-result.json", "stdout.txt", "stderr.txt"))
_ACTIVE = None


@dataclass(frozen=True)
class OfflineWriteAllowance:
    allocated_bytes: int
    file_names: int = 102
    directories: int = 9

    def __post_init__(self):
        for value, maximum in (
            (self.allocated_bytes, 2**63 - 1),
            (self.file_names, 102),
            (self.directories, 9),
        ):
            if type(value) is not int or not 0 < value <= maximum:
                raise ValueError("invalid offline write allowance")


class OfflineWriteDenied(RuntimeError):
    """A first denial remains fatal even when a library catches the exception."""


def _rounded(size):
    return ((size + _BLOCK - 1) // _BLOCK) * _BLOCK


class _Descriptor:
    __slots__ = ("stream",)

    def __init__(self, stream):
        self.stream = stream

    def __index__(self):
        self.stream._guard._deny("descriptor")


class _Stream:
    """Only the operations used by the actual rows/font/publisher writers."""

    __slots__ = ("_guard", "_handle", "_identity", "_kind", "_length", "_path", "_token")

    def __init__(self, guard, handle, path, kind):
        self._guard, self._handle, self._path, self._kind = guard, handle, path, kind
        self._length = 0
        self._token = _Descriptor(self)
        self._identity = guard._identity(os.fstat(handle.fileno()))

    def _check(self):
        self._guard.assert_clear()
        if self._handle.closed:
            raise ValueError("I/O operation on closed file")
        self._guard._check_stream(self)

    def write(self, chunk):
        self._check()
        guard = self._guard
        if self._kind == "font":
            if type(chunk) is not str:
                guard._deny("writer")
            payload = chunk.encode("utf-8")
        else:
            if type(chunk) is not bytes:
                guard._deny("writer")
            payload = chunk
        if self._kind == "atomic":
            operation = guard._operation
            if operation is None or payload != operation["payload"] or self._length:
                guard._deny("writer")
        elif self._kind == "lock":
            guard._deny("writer")
        else:
            guard._reserve(self._path, self._length + len(payload), names=1)
        written = self._handle.write(payload)
        self._handle.flush()
        self._length += written
        return len(chunk) if self._kind == "font" else written

    def tell(self):
        self._check()
        return self._length

    def flush(self):
        self._check()
        self._handle.flush()

    def fileno(self):
        self._check()
        return self._token

    def close(self):
        self._handle.close()

    @property
    def closed(self):
        return self._handle.closed

    @property
    def buffer(self):
        self._guard._deny("descriptor")

    @property
    def raw(self):
        self._guard._deny("descriptor")

    def seek(self, *args, **kwargs):
        self._guard._deny("writer")

    truncate = seek
    writelines = seek
    detach = seek

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()


class OfflineWrites:
    def __init__(self, root, allowance):
        self.root, self.allowance = root, allowance
        self._denial = None
        self._pins = {}
        self._operation = None
        self._streams = {}
        self._intent = None
        self._bound = False
        self._thread = threading.get_ident()
        self._permit = threading.local()
        self._git_batch_pipe = False
        self._git_streams = []
        self._original = {}
        self._inventory(initial=True)

    def _deny(self, code):
        if self._denial is None:
            self._denial = code
        raise OfflineWriteDenied("offline write denied: " + self._denial)

    def assert_clear(self):
        if self._denial is not None:
            self._deny(self._denial)
        self._path(self.root)
        self._inventory()

    @staticmethod
    def _identity(info):
        return info.st_dev, info.st_ino

    def _descriptor_path(self, fd):
        if type(fd) is not int:
            self._deny("descriptor")
        if sys.platform == "darwin":
            import fcntl

            path = Path(os.fsdecode(fcntl.fcntl(fd, 50, bytes(1024)).split(b"\0", 1)[0]))
        else:
            path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if self._identity(os.fstat(fd)) != self._identity(path.lstat()):
            self._deny("authority")
        return path

    def _path(self, value, directory_fd=None, *, ancestor=False):
        if isinstance(value, (int, _Descriptor)):
            self._deny("descriptor")
        try:
            path = Path(os.fsdecode(value))
        except (TypeError, ValueError):
            self._deny("path")
        if ".." in path.parts:
            self._deny("path")
        if directory_fd not in (None, -1):
            if path.is_absolute() or len(path.parts) != 1:
                self._deny("descriptor")
            path = self._descriptor_path(directory_fd) / path
        path = path.absolute()
        if ancestor and self.root.is_relative_to(path) and path.is_dir():
            return path
        if not path.is_relative_to(self.root) or path.resolve() != path:
            self._deny("path")
        for parent, identity in self._pins.items():
            try:
                info = parent.lstat()
            except OSError:
                self._deny("authority")
            if not stat.S_ISDIR(info.st_mode) or self._identity(info) != identity:
                self._deny("authority")
        return path

    def _family(self, relative):
        if relative in _PILOT:
            return "pilot"
        if relative in _ATOMIC:
            return "atomic"
        if re.fullmatch(
            r"run/eval/primary/episodes/000(?:0[0-9]|1[0-5])"
            r"\.(?:neural\.json|trajectory\.json\.gz|telemetry\.json)",
            relative,
        ):
            return "atomic"
        if re.fullmatch(
            r"run/eval/primary/crashes/(?:weights-[0-9a-f]{64}\.safetensors|"
            r"[0-9a-f]{32}\.(?:json|safetensors))",
            relative,
        ):
            return "atomic"
        if relative in (_ROWS, _ROWS_FINAL, _FONT, _LOCK):
            return "stream"
        self._deny("authority" if relative in _PARENT else "path")

    def _inventory(self, *, initial=False):
        entries = [(self.root, self.root.lstat())]
        for path, info in entries:
            if stat.S_ISDIR(info.st_mode):
                relative = str(path.relative_to(self.root))
                if relative not in _DIRECTORIES:
                    self._deny("path")
                identity = self._identity(info)
                if self._pins and info.st_dev != self._pins[self.root][0]:
                    self._deny("path")
                if path in self._pins and self._pins[path] != identity:
                    self._deny("authority")
                self._pins[path] = identity
                with os.scandir(path) as children:
                    entries.extend(
                        (Path(child.path), child.stat(follow_symlinks=False)) for child in children
                    )
        files, dirs, used = {}, 0, 0
        weights, crashes = set(), set()
        for path, info in entries:
            relative = str(path.relative_to(self.root))
            if info.st_dev != self._pins[self.root][0]:
                self._deny("path")
            if stat.S_ISDIR(info.st_mode):
                dirs += 1
            elif stat.S_ISREG(info.st_mode):
                if relative == "intent.json":
                    identity = (self._identity(info), info.st_size, info.st_mtime_ns)
                    if initial and info.st_nlink == 1:
                        self._intent = identity
                    if identity != self._intent:
                        self._deny("authority")
                    continue
                if self._operation is None or path != self._operation.get("temp"):
                    self._family(relative)
                files[path] = info
                if "/crashes/weights-" in relative:
                    weights.add(path.name)
                elif re.fullmatch(r"[0-9a-f]{32}\.(?:json|safetensors)", path.name):
                    crashes.add(path.stem)
            else:
                self._deny("path")
            used += info.st_blocks * 512
        if self._intent is not None and not (self.root / "intent.json").exists():
            self._deny("authority")
        if len(weights) > 1 or len(crashes) > 16:
            self._deny("path")
        for info in files.values():
            if info.st_nlink != 1:
                aliases = {
                    p for p, other in files.items() if self._identity(other) == self._identity(info)
                }
                owned = {self.root / _ROWS, self.root / _ROWS_FINAL}
                if self._operation is not None:
                    owned_atomic = {self._operation.get("temp"), self._operation["target"]}
                else:
                    owned_atomic = set()
                if (
                    info.st_nlink != 2
                    or len(aliases) != 2
                    or not (
                        (aliases == owned and _ROWS in self._streams) or aliases == owned_atomic
                    )
                ):
                    self._deny("authority")
        if len(files) > self.allowance.file_names:
            self._deny("names")
        if dirs > self.allowance.directories:
            self._deny("directories")
        if used > self.allowance.allocated_bytes:
            self._deny("bytes")
        return used, files, dirs

    def _reserve(self, path, size, *, names, missing=None):
        self.assert_clear()
        if threading.get_ident() != self._thread:
            self._deny("authority")
        if type(size) is not int or size < 0:
            self._deny("bytes")
        path = self._path(path)
        used, files, dirs = self._inventory()
        if missing is None:
            parents = [
                parent
                for parent in path.parents
                if parent.is_relative_to(self.root) and not parent.exists()
            ]
            if any(str(parent.relative_to(self.root)) not in _DIRECTORIES for parent in parents):
                self._deny("path")
            missing = len(parents)
        if len(files) + names > self.allowance.file_names:
            self._deny("names")
        if dirs + missing > self.allowance.directories:
            self._deny("directories")
        projected = (
            used + 2 * _rounded(size) + _BLOCK + (dirs + missing + len(files) + names) * _BLOCK
        )
        if projected > self.allowance.allocated_bytes:
            self._deny("bytes")

    def admit_publication(self, path, size, *, writer):
        self.assert_clear()
        path = self._path(path)
        if self._family(str(path.relative_to(self.root))) != writer or writer not in (
            "pilot",
            "atomic",
        ):
            self._deny("writer")
        if self._operation is not None:
            self._deny("writer")
        _, files, _ = self._inventory()
        if "/crashes/weights-" in str(path) and any(
            p.parent == path.parent and p.name.startswith("weights-") and p != path for p in files
        ):
            self._deny("path")
        if re.fullmatch(r"[0-9a-f]{32}\.(?:json|safetensors)", path.name):
            stems = {
                p.stem
                for p in files
                if p.parent == path.parent
                and re.fullmatch(r"[0-9a-f]{32}\.(?:json|safetensors)", p.name)
            }
            if path.stem not in stems and len(stems) >= 16:
                self._deny("path")
        self._reserve(path, size, names=2)

    @contextmanager
    def _event(self, name):
        previous = getattr(self._permit, "event", None)
        self._permit.event = name
        try:
            yield
        finally:
            self._permit.event = previous

    def _check_stream(self, stream):
        self._path(stream._path)
        info = os.fstat(stream._handle.fileno())
        if (
            self._identity(info) != stream._identity
            or self._identity(stream._path.lstat()) != stream._identity
            or info.st_nlink != 1
        ):
            self._deny("authority")

    def bind_publishers(self):
        self.assert_clear()
        if self._bound:
            self._deny("authority")
        from silent_cascade import io as durable

        originals = (durable._durable_temp, durable.atomic_create_bytes, durable.atomic_write_bytes)
        if "silent_cascade.train.pilot_data" in sys.modules or any(
            name.startswith("silent_cascade.")
            and name != "silent_cascade.io"
            and module is not None
            and any(value is writer for value in vars(module).values() for writer in originals)
            for name, module in tuple(sys.modules.items())
        ):
            self._deny("authority")
        original_temp = durable._durable_temp
        self._temp_code = tempfile._mkstemp_inner.__code__

        def guarded_pilot(path, payload):
            self.admit_publication(path, len(payload), writer="pilot")
            self._operation = dict(target=self._path(path), payload=payload, kind="pilot")
            try:
                return original_pilot(path, payload)
            finally:
                if (
                    self._operation is not None
                    and not self._operation.get("temp", self.root).exists()
                ):
                    self._operation = None

        def guarded_temp(path, data, mode):
            self.admit_publication(path, len(data), writer="atomic")
            self._operation = dict(target=self._path(path), payload=data, kind="atomic")
            return original_temp(path, data, mode)

        def finish_publication(original):
            def guarded(path, data, *, mode=0o644):
                try:
                    return original(path, data, mode=mode)
                finally:
                    if (
                        self._operation is not None
                        and not self._operation.get("temp", self.root).exists()
                    ):
                        self._operation = None

            return guarded

        durable._durable_temp = guarded_temp
        durable.atomic_create_bytes = finish_publication(durable.atomic_create_bytes)
        durable.atomic_write_bytes = finish_publication(durable.atomic_write_bytes)
        # pilot_data imports evaluator consumers: bind durable aliases first.
        from silent_cascade.train import pilot_data

        original_pilot = pilot_data._publish_pilot_bytes
        self._pilot_code = original_pilot.__code__
        pilot_data._publish_pilot_bytes = guarded_pilot
        self._bound = True

    def _open_fd(self, path, flags, mode=0o777, *, dir_fd=None):
        writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        if not writing:
            if isinstance(path, _Descriptor):
                self._deny("descriptor")
            return self._original["open"](path, flags, mode, dir_fd=dir_fd)
        self.assert_clear()
        target = self._path(path, dir_fd)
        operation = self._operation
        caller = sys._getframe(1).f_code
        # The legacy boundary wraps os.open once. Inspect only that known shim.
        if caller.co_name == "bounded_open" and caller.co_filename.endswith("/pilot_offline.py"):
            caller = sys._getframe(2).f_code
        if operation is None or "temp" in operation:
            self._deny("writer")
        if operation["kind"] == "pilot":
            valid = caller is self._pilot_code and re.fullmatch(
                r"\.pilot-[0-9a-f]{32}\.tmp", target.name
            )
        else:
            valid = caller is self._temp_code and re.fullmatch(
                r"\." + re.escape(operation["target"].name) + r"\.[a-z0-9_]{8}\.tmp", target.name
            )
        access = os.O_WRONLY if operation["kind"] == "pilot" else os.O_RDWR
        required = access | os.O_CREAT | os.O_EXCL
        if (
            not valid
            or target.parent != operation["target"].parent
            or flags & required != required
            or flags & (os.O_TRUNC | os.O_APPEND)
        ):
            self._deny("writer")
        self._inventory()
        operation["temp"] = target
        with self._event("open"):
            fd = self._original["open"](path, flags | os.O_NOFOLLOW, mode, dir_fd=dir_fd)
        operation["fd"] = fd
        operation["identity"] = self._identity(os.fstat(fd))
        return fd

    def _open_stream(
        self,
        file,
        mode="r",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    ):
        if isinstance(file, _Descriptor):
            self._deny("descriptor")
        if not any(flag in mode for flag in "wax+"):
            return self._original["builtin_open"](
                file, mode, buffering, encoding, errors, newline, closefd, opener
            )
        self.assert_clear()
        if type(file) is int:
            operation = self._operation
            if (
                operation is not None
                and file == operation.get("fd")
                and mode == "wb"
                and "stream" not in operation
            ):
                if self._identity(os.fstat(file)) != operation["identity"]:
                    self._deny("authority")
                with self._event("open"):
                    handle = self._original["builtin_open"](file, "wb")
                stream = _Stream(self, handle, operation["temp"], "atomic")
                operation["stream"] = stream
                return stream
            if self._git_batch_pipe and stat.S_ISFIFO(os.fstat(file).st_mode):
                with self._event("open"):
                    handle = self._original["builtin_open"](file, mode, buffering)
                self._git_streams = [item for item in self._git_streams if not item[0].closed]
                self._git_streams.append((handle, self._identity(os.fstat(file))))
                return handle
            self._deny("descriptor")
        path = self._path(file)
        relative = str(path.relative_to(self.root))
        kind = {_ROWS: "rows", _FONT: "font", _LOCK: "lock"}.get(relative)
        if (
            kind is None
            or relative in self._streams
            or mode != ("w" if kind == "font" else "xb")
            or buffering != -1
            or encoding not in (None, "utf-8", "UTF-8")
            or errors is not None
            or newline is not None
            or not closefd
            or opener is not None
        ):
            self._deny("writer")
        if path.exists():
            self._deny("writer")
        self._reserve(path, 0, names=1)
        with self._event("open"):
            handle = self._original["builtin_open"](path, "xb")
        stream = _Stream(self, handle, path, kind)
        self._streams[relative] = stream
        return stream

    def _mkdir(self, path, mode=0o777, *, dir_fd=None):
        target = self._path(path, dir_fd, ancestor=True)
        if self.root.is_relative_to(target) and target.exists():
            raise FileExistsError(target)
        self.assert_clear()
        if str(target.relative_to(self.root)) not in _DIRECTORIES:
            self._deny("path")
        if self._operation is not None and not self._operation["target"].parent.is_relative_to(
            target
        ):
            self._deny("writer")
        if not target.exists() and self._operation is None:
            self._reserve(target, 0, names=0, missing=1)
        else:
            self._inventory()
        with self._event("os.mkdir"):
            result = self._original["mkdir"](path, mode, dir_fd=dir_fd)
        self._pins[target] = self._identity(target.lstat())
        return result

    def _link_or_rename(
        self, name, source, destination, *, src_dir_fd=None, dst_dir_fd=None, follow_symlinks=True
    ):
        self.assert_clear()
        original_source, original_destination = source, destination
        source = self._path(source, src_dir_fd)
        destination = self._path(destination, dst_dir_fd)
        operation = self._operation
        if (
            name == "link"
            and source == self.root / _ROWS
            and destination == self.root / _ROWS_FINAL
            and _ROWS in self._streams
            and self._streams[_ROWS].closed
        ):
            self._reserve(destination, source.stat().st_size, names=1)
        elif (
            operation is None
            or source != operation.get("temp")
            or destination != operation["target"]
        ):
            self._deny("writer")
        else:
            self._inventory()
            if self._identity(source.lstat()) != operation["identity"]:
                self._deny("authority")
            if source.stat().st_size != len(operation["payload"]):
                self._deny("writer")
        event = "os.link" if name == "link" else "os.rename"
        with self._event(event):
            if name == "link":
                return self._original[name](
                    original_source,
                    original_destination,
                    src_dir_fd=src_dir_fd,
                    dst_dir_fd=dst_dir_fd,
                    follow_symlinks=False,
                )
            return self._original[name](
                original_source, original_destination, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd
            )

    def _unlink(self, path, *, dir_fd=None):
        target = self._path(path, dir_fd)
        operation = self._operation
        relative = str(target.relative_to(self.root))
        owned_temp = operation is not None and target == operation.get("temp")
        owned_lock = relative == _LOCK and _LOCK in self._streams
        owned_rows = (
            relative == _ROWS
            and _ROWS in self._streams
            and (self.root / _ROWS_FINAL).exists()
            and self._identity(target.stat()) == self._identity((self.root / _ROWS_FINAL).stat())
        )
        if not (owned_temp or owned_lock or owned_rows):
            self._deny("writer")
        if target.exists():
            identity = operation["identity"] if owned_temp else self._streams[relative]._identity
            if self._identity(target.lstat()) != identity:
                self._deny("authority")
        with self._event("os.remove"):
            result = self._original["unlink"](path, dir_fd=dir_fd)
        if owned_lock:
            del self._streams[_LOCK]
        return result

    def _fsync(self, fd):
        if isinstance(fd, _Descriptor):
            stream = fd.stream
            if stream._guard is not self:
                self._deny("descriptor")
            stream._check()
            return self._original["fsync"](stream._handle.fileno())
        return self._original["fsync"](fd)

    def _chmod(self, path, mode, *, dir_fd=None, follow_symlinks=True):
        self.assert_clear()
        target = self._path(path, dir_fd)
        if self._operation is None or target != self._operation.get("temp"):
            self._deny("writer")
        self._inventory()
        with self._event("os.chmod"):
            return self._original["chmod"](
                path, mode, dir_fd=dir_fd, follow_symlinks=follow_symlinks
            )

    def _audit(self, event, args):
        writing = event == "open" and args[2] & (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        )
        if (
            writing
            or event
            in {
                "os.mkdir",
                "os.remove",
                "os.rmdir",
                "os.rename",
                "os.link",
                "os.symlink",
                "os.truncate",
                "os.chmod",
                "os.chown",
            }
        ) and getattr(self._permit, "event", None) != event:
            self._deny("writer")

    def _install(self):
        import _io
        import mmap

        for name in ("open", "mkdir", "link", "rename", "replace", "unlink", "fsync", "chmod"):
            self._original[name] = getattr(os, name)
        self._original["builtin_open"] = builtins.open
        os.open, os.mkdir, os.unlink, os.remove = (
            self._open_fd,
            self._mkdir,
            self._unlink,
            self._unlink,
        )
        os.fsync, os.chmod = self._fsync, self._chmod
        os.link = lambda src, dst, **kw: self._link_or_rename("link", src, dst, **kw)
        os.rename = lambda src, dst, **kw: self._link_or_rename("rename", src, dst, **kw)
        os.replace = lambda src, dst, **kw: self._link_or_rename("replace", src, dst, **kw)
        builtins.open = io.open = _io.open = self._open_stream
        os.fdopen = self._open_stream

        def denied_descriptor(*args, **kwargs):
            self._deny("descriptor")

        def file_io(file, mode="r", *args, **kwargs):
            if isinstance(file, _Descriptor) or any(flag in mode for flag in "wax+"):
                self._deny("descriptor")
            return original_fileio(file, mode, *args, **kwargs)

        original_fileio = io.FileIO
        io.FileIO = _io.FileIO = file_io
        for name in (
            "write",
            "writev",
            "pwrite",
            "ftruncate",
            "truncate",
            "dup",
            "dup2",
            "sendfile",
            "copy_file_range",
            "fchmod",
            "fchown",
        ):
            if hasattr(os, name):
                original = getattr(os, name)
                if name == "write":

                    def write(fd, data, original=original):
                        if type(fd) is int and fd in (1, 2):
                            return original(fd, data)
                        if type(fd) is int and any(
                            not handle.closed
                            and handle.fileno() == fd
                            and self._identity(os.fstat(fd)) == identity
                            for handle, identity in self._git_streams
                        ):
                            return original(fd, data)
                        self._deny("descriptor")

                    os.write = write
                else:
                    setattr(os, name, denied_descriptor)
        original_mmap = mmap.mmap

        def mapped(fd, *args, **kwargs):
            if isinstance(fd, _Descriptor) or (
                fd != -1 and kwargs.get("access") != mmap.ACCESS_READ
            ):
                self._deny("descriptor")
            return original_mmap(fd, *args, **kwargs)

        mmap.mmap = mapped
        sys.addaudithook(self._audit)


def active_offline_writes():
    """Boundary coordination only; installation itself never grants launch authority."""
    return _ACTIVE


def install_offline_writes(root: Path, allowance: OfflineWriteAllowance):
    global _ACTIVE
    if _ACTIVE is not None:
        _ACTIVE._deny("authority")
    if type(allowance) is not OfflineWriteAllowance:
        raise ValueError("invalid offline write allowance")
    OfflineWriteAllowance(allowance.allocated_bytes, allowance.file_names, allowance.directories)
    root = Path(root)
    if (
        not root.is_absolute()
        or ".." in root.parts
        or root.resolve(strict=True) != root
        or not root.is_dir()
    ):
        raise ValueError("offline write root must be a canonical existing directory")
    if os.statvfs(root).f_frsize != _BLOCK:
        raise ValueError("unsupported offline allocation unit")
    guard = OfflineWrites(root, allowance)
    guard._install()
    _ACTIVE = guard
    return guard
