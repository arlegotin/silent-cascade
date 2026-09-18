"""File-only offline evidence requests and independently authenticated leases."""

import json
import os
import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from silent_cascade.archive.catalog import _control_reader, _pinned_directory
from silent_cascade.archive.transport import _lock, _replace_at
from silent_cascade.archive.types import ArchivePolicy, FileEntry, UnitRef
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

_OPERATIONS = {"seal", "archive", "lease", "release", "status", "stop"}
_HEALTH_SECONDS = 30
_STALL_SECONDS = 30 * 60 * 3 + _HEALTH_SECONDS


def _episode_key(run_id: str, logical_root: str, ordinal: int) -> str:
    return sha256_bytes(
        canonical_json_bytes({"run_id": run_id, "logical_root": logical_root, "ordinal": ordinal})
    )


def _binding_root(control_dir, run_id, policy, reader):
    try:
        head = json.loads(
            _control_reader(control_dir)("episode-bindings/head.json", policy.page_bytes)
        )
    except FileNotFoundError:
        return None
    from silent_cascade.archive.types import CatalogNodeRef

    if set(head) != {"run_id", "root_sha256"} or head["run_id"] != run_id:
        raise ValueError("episode binding root identity differs")
    raw = reader(f"roots/{head['root_sha256']}.json", policy.page_bytes)
    if sha256_bytes(raw) != head["root_sha256"]:
        raise ValueError("episode binding root hash differs")
    root = json.loads(raw)
    if (
        set(root) != {"schema_version", "run_id", "episodes"}
        or root["schema_version"] != "phase4-r2-episodes-v1"
        or root["run_id"] != run_id
    ):
        raise ValueError("invalid episode binding root")
    return CatalogNodeRef.model_validate(root["episodes"])


def register_episode_binding(
    *,
    run_dir: Path,
    control_dir: Path,
    logical_root: str,
    unit_ref: UnitRef,
    commit,
    transport,
    policy: ArchivePolicy,
):
    """Parent-only durable exact-commit publication, before payload eviction."""
    from silent_cascade.archive.catalog import (
        _CatalogStore,
        _insert_batch,
        _opened_unit,
        _pinned_directory,
        iter_unit_files,
    )
    from silent_cascade.archive.ledger import _hash_file
    from silent_cascade.archive.transport import (
        _cold_reader,
        _object_key,
        _put_verified,
        archive_operation_lock,
    )
    from silent_cascade.archive.types import EpisodeBinding, EpisodeBindingEntry

    with archive_operation_lock(control_dir):
        with _opened_unit(control_dir, unit_ref) as (_, manifest):
            owned = tuple(iter_unit_files(control_dir, unit_ref))
            paths = tuple(entry.path for entry in commit.owned)
            if (
                manifest.kind != "episode_pack"
                or logical_root != manifest.logical_root
                or commit.identity_sha256 != manifest.identity.evidence_identity_sha256
                or not any(group.paths == paths for group in manifest.episode_groups)
                or tuple(entry for entry in owned if entry.path in paths) != commit.owned
            ):
                raise ValueError("episode commit ownership differs from authenticated group")
            if any(
                entry
                not in tuple(
                    FileEntry(value.path, value.sha256, value.bytes) for value in manifest.borrowed
                )
                for entry in commit.borrowed
            ):
                raise ValueError("episode borrowed ownership differs")
            for entry in commit.borrowed:
                _hash_file(run_dir, entry)
            binding = EpisodeBinding(
                run_id=manifest.identity.run_id,
                logical_root=logical_root,
                ordinal=commit.ordinal,
                commit_sha256=sha256_bytes(canonical_json_bytes(asdict(commit))),
                unit_ref=unit_ref,
                commit=commit,
            )
        cold = _cold_reader(control_dir=control_dir, transport=transport, run_id=binding.run_id)

        def reader(path, maximum):
            return cold("episode-bindings/" + path, maximum)

        root = _binding_root(control_dir, binding.run_id, policy, reader)
        store = _CatalogStore(
            control_dir=control_dir / "episode-bindings", policy=policy, reader=reader
        )
        entry = EpisodeBindingEntry(
            key=_episode_key(binding.run_id, logical_root, commit.ordinal), binding=binding
        )
        root = _insert_batch(root, (entry,), index="episodes", store=store)
        raw_root = canonical_json_bytes(
            {
                "schema_version": "phase4-r2-episodes-v1",
                "run_id": binding.run_id,
                "episodes": root.model_dump(mode="json"),
            }
        )
        digest = sha256_bytes(raw_root)
        if len(raw_root) > policy.page_bytes:
            raise ValueError("episode binding root exceeds page budget")
        from silent_cascade.archive.catalog import _create_control_object

        with _pinned_directory(control_dir / "episode-bindings", create=True) as descriptor:
            store.publish(descriptor)
        _create_control_object(control_dir, f"episode-bindings/roots/{digest}.json", raw_root)
        objects = {**store.staged, f"roots/{digest}.json": raw_root}
        for logical, raw in objects.items():
            _put_verified(
                control_dir=control_dir,
                transport=transport,
                key=_object_key(binding.run_id, "episode-bindings/" + logical),
                source=control_dir / "episode-bindings" / logical,
                expected_bytes=len(raw),
                expected_sha256=sha256_bytes(raw),
                policy=policy,
            )
        _publish(
            control_dir / "episode-bindings",
            "head.json",
            {"run_id": binding.run_id, "root_sha256": digest},
            max_bytes=policy.page_bytes,
        )
        from silent_cascade.archive.transport import _remove_verified_control_object

        for logical, raw in objects.items():
            _remove_verified_control_object(control_dir, "episode-bindings/" + logical, raw)
        return binding


def lookup_episode_binding(
    *,
    control_dir: Path,
    run_id: str,
    logical_root: str,
    ordinal: int,
    commit_sha256: str,
    transport,
    policy: ArchivePolicy,
):
    binding = _read_episode_binding(
        control_dir=control_dir,
        run_id=run_id,
        logical_root=logical_root,
        ordinal=ordinal,
        transport=transport,
        policy=policy,
    )
    if binding.commit_sha256 != commit_sha256:
        raise ValueError("episode binding absent or commit identity differs")
    return binding


def _read_episode_binding(*, control_dir, run_id, logical_root, ordinal, transport, policy):
    from silent_cascade.archive.catalog import _CatalogStore, _lookup
    from silent_cascade.archive.transport import _cold_reader
    from silent_cascade.archive.types import EpisodeBindingEntry

    cold = _cold_reader(control_dir=control_dir, transport=transport, run_id=run_id)

    def reader(path, maximum):
        return cold("episode-bindings/" + path, maximum)

    root = _binding_root(control_dir, run_id, policy, reader)
    store = _CatalogStore(
        control_dir=control_dir / "episode-bindings", policy=policy, reader=reader
    )
    entry = _lookup(store, root, _episode_key(run_id, logical_root, ordinal), index="episodes")
    if not isinstance(entry, EpisodeBindingEntry) or (
        entry.binding.run_id,
        entry.binding.logical_root,
        entry.binding.ordinal,
    ) != (run_id, logical_root, ordinal):
        raise ValueError("episode binding absent or commit identity differs")
    return entry.binding


@dataclass(frozen=True)
class EvidenceLease:
    """Authenticated payload and inventory roots, valid only inside their lease."""

    ref: UnitRef
    local_root: Path
    metadata_root: Path


class EvidenceContext(Protocol):
    def entries(self) -> Iterator[FileEntry]: ...
    def evaluation_roots(self) -> tuple[str, ...]: ...
    def lease(self, ref: UnitRef) -> AbstractContextManager[EvidenceLease]: ...
    def metadata(self, logical_root: str) -> AbstractContextManager[EvidenceLease]: ...
    def episode_commit(self, logical_root: str, ordinal: int): ...
    def episode(
        self, logical_root: str, ordinal: int, *, commit_sha256: str
    ) -> AbstractContextManager[EvidenceLease]: ...
    def read_record(self, logical_path: str, *, expected_sha256: str, max_bytes: int) -> bytes: ...
    def verify_inventory(self, entries: Iterable[FileEntry]) -> None: ...


def _publish(root: Path, name: str, value, *, max_bytes: int):
    raw = canonical_json_bytes(value)
    if len(raw) > max_bytes:
        raise ValueError("archive protocol payload exceeds bounded page")
    with _pinned_directory(root, create=True) as descriptor:
        _replace_at(descriptor, name, raw)


class LocalArchiveSession:
    def __init__(self, *, run_dir: Path, control_dir: Path, policy: ArchivePolicy):
        self.run_dir = run_dir.absolute()
        self.control_dir = control_dir.absolute()
        self.policy = ArchivePolicy.model_validate(policy.model_dump())
        self._reader = _control_reader(self.control_dir)
        self._identity = json.loads(self._reader("session.json", self.policy.page_bytes))
        expected = {
            "schema_version",
            "run_id",
            "session_id",
            "run_dir",
            "policy_sha256",
            "pid",
            "create_time",
        }
        if set(self._identity) != expected or (
            self._identity["schema_version"] != "phase4-r2-session-v1"
            or self._identity["run_dir"] != str(self.run_dir)
            or self._identity["policy_sha256"] != sha256_bytes(canonical_json_bytes(self.policy))
        ):
            raise ValueError("archive session identity differs")

    @contextmanager
    def lease(self, ref: UnitRef):
        with self._leased({"ref": asdict(ref)}, ref=ref) as lease:
            yield lease

    @contextmanager
    def _leased(self, payload, *, ref=None):
        from silent_cascade.archive.catalog import _opened_unit, iter_unit_files
        from silent_cascade.archive.ledger import _hash_file
        from silent_cascade.archive.transport import unit_reader_lease

        response = self._request("lease", payload, unit_id=None if ref is None else ref.unit_id)
        try:
            base_keys = {"token", "ref", "local_root", "metadata_root", "selected_paths"}
            if set(response) not in (base_keys, base_keys | {"binding"}) or (
                ref is not None and UnitRef(**response["ref"]) != ref
            ):
                raise ValueError("lease response unit identity differs")
            ref = UnitRef(**response["ref"])
            local_root = Path(response["local_root"])
            metadata_root = Path(response["metadata_root"])
            if local_root != self.run_dir and not local_root.is_relative_to(
                self.control_dir / "lease-cache"
            ):
                raise ValueError("lease root escapes counted cache")
            if not metadata_root.is_relative_to(self.control_dir / "lease-metadata"):
                raise ValueError("lease metadata escapes bounded cache")
            with (
                unit_reader_lease(control_dir=self.control_dir, ref=ref),
                _opened_unit(metadata_root, ref) as (_, manifest),
            ):
                if (
                    manifest.policy_sha256 != self._identity["policy_sha256"]
                    or manifest.identity.run_id != self._identity["run_id"]
                ):
                    raise ValueError("lease manifest identity differs")
                selected = response["selected_paths"]
                if ref.kind == "episode_pack" and tuple(selected) not in tuple(
                    group.paths for group in manifest.episode_groups
                ):
                    raise ValueError("lease response is not one episode group")
                if "commit_sha256" in payload:
                    from silent_cascade.archive.types import EpisodeBinding

                    binding = EpisodeBinding.model_validate_json(
                        canonical_json_bytes(response["binding"])
                    )
                    owned = tuple(
                        entry
                        for entry in iter_unit_files(metadata_root, ref)
                        if entry.path in selected
                    )
                    borrowed = tuple(
                        FileEntry(entry.path, entry.sha256, entry.bytes)
                        for entry in manifest.borrowed
                    )
                    if (
                        binding.commit_sha256 != payload["commit_sha256"]
                        or binding.ordinal != payload["ordinal"]
                        or binding.logical_root != payload["logical_root"]
                        or binding.unit_ref != ref
                        or binding.commit.identity_sha256
                        != manifest.identity.evidence_identity_sha256
                        or binding.commit.owned != owned
                        or any(entry not in borrowed for entry in binding.commit.borrowed)
                    ):
                        raise ValueError("lease episode commit differs")
                for entry in iter_unit_files(metadata_root, ref):
                    if selected is None or entry.path in selected:
                        _hash_file(local_root, entry)
                for entry in manifest.borrowed:
                    _hash_file(local_root, FileEntry(entry.path, entry.sha256, entry.bytes))
                yield EvidenceLease(ref, local_root, metadata_root)
        finally:
            self._request("release", {"token": response["token"]}, unit_id=ref.unit_id)

    def metadata(self, logical_root: str):
        return self._leased({"metadata_root": logical_root})

    def episode_commit(self, logical_root: str, ordinal: int):
        from silent_cascade.archive.catalog import _safe_logical_path
        from silent_cascade.archive.types import EpisodeBinding

        _safe_logical_path(logical_root, field="evaluation root")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("invalid episode ordinal")
        response = self._request(
            "status", {"episode_commit": {"logical_root": logical_root, "ordinal": ordinal}}
        )
        if set(response) != {"binding"}:
            raise ValueError("invalid episode discovery response")
        binding = EpisodeBinding.model_validate_json(canonical_json_bytes(response["binding"]))
        if (binding.run_id, binding.logical_root, binding.ordinal) != (
            self._identity["run_id"],
            logical_root,
            ordinal,
        ):
            raise ValueError("episode discovery identity differs")
        return binding.commit

    def episode(self, logical_root: str, ordinal: int, *, commit_sha256: str):
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("invalid episode ordinal")
        return self._leased(
            {"logical_root": logical_root, "ordinal": ordinal, "commit_sha256": commit_sha256}
        )

    def entries(self):
        action = "begin"
        while True:
            result = self._request("status", {"inventory": action})
            if set(result) != {"entry"}:
                raise ValueError("invalid inventory response")
            if result["entry"] is None:
                return
            yield FileEntry(**result["entry"])
            action = "next"

    def evaluation_roots(self):
        return tuple(
            sorted(
                {
                    str(Path(entry.path).parent)
                    for entry in self.entries()
                    if Path(entry.path).name == "DONE"
                }
            )
        )

    def read_record(self, logical_path: str, *, expected_sha256: str, max_bytes: int):
        from silent_cascade.archive.catalog import _safe_logical_path

        _safe_logical_path(logical_path, field="evidence record")
        if type(max_bytes) is not int or not 0 <= max_bytes <= self.policy.page_bytes:
            raise ValueError("record exceeds bounded metadata page")

        def read(root):
            raw = _control_reader(root)(logical_path, max_bytes)
            if sha256_bytes(raw) != expected_sha256:
                raise ValueError("record authoritative hash differs")
            return raw

        try:
            return read(self.run_dir)
        except FileNotFoundError:
            with self._leased({"path": logical_path}) as lease:
                return read(lease.local_root)

    def verify_inventory(self, entries):
        from silent_cascade.archive.ledger import _hash_file

        expected_count = expected_sum = 0
        for entry in entries:
            if not isinstance(entry, FileEntry):
                raise ValueError("invalid authoritative inventory entry")
            try:
                _hash_file(self.run_dir, entry)
            except FileNotFoundError:
                with self._leased({"path": entry.path}) as lease:
                    _hash_file(lease.local_root, entry)
            expected_sum += int(sha256_bytes(canonical_json_bytes(asdict(entry))), 16)
            expected_count += 1
        actual_sum = actual_count = 0
        for entry in self.entries():
            actual_sum += int(sha256_bytes(canonical_json_bytes(asdict(entry))), 16)
            actual_count += 1
        if (expected_count, expected_sum) != (actual_count, actual_sum):
            raise ValueError("authoritative inventory closure differs")

    def _parent_alive(self):
        from silent_cascade.archive.supervisor import StorageBlocked, _process_identity

        if _process_identity(self._identity["pid"]) != self._identity["create_time"]:
            raise StorageBlocked(
                "storage_blocked: archive parent stopped; "
                "preserve pending files and durable checkpoint"
            )

    def _request(self, operation: str, payload: dict, *, unit_id: str | None = None) -> dict:
        from silent_cascade.archive.supervisor import StorageBlocked

        if operation not in _OPERATIONS or type(payload) is not dict:
            raise ValueError("unknown archive request operation")
        with _lock(
            control_dir=self.control_dir, relative=("requests.lock",), shared=False, blocking=False
        ):
            self._parent_alive()
            try:
                previous = json.loads(self._reader("request.json", self.policy.page_bytes))
                sequence = previous["sequence"] + 1
            except FileNotFoundError:
                sequence = 1
            request = {
                "schema_version": "phase4-r2-request-v1",
                "run_id": self._identity["run_id"],
                "session_id": self._identity["session_id"],
                "sequence": sequence,
                "operation": operation,
                "policy_sha256": self._identity["policy_sha256"],
                "unit_id": unit_id,
                "payload": payload,
                "pid": os.getpid(),
            }
            _publish(self.control_dir, "request.json", request, max_bytes=self.policy.page_bytes)
            expected_hash = sha256_bytes(canonical_json_bytes(request))
            started = progress_time = time.monotonic()
            progress_value = None
            while True:
                self._parent_alive()
                try:
                    response = json.loads(self._reader("response.json", self.policy.page_bytes))
                except FileNotFoundError:
                    response = None
                if response is not None and response.get("sequence") == sequence:
                    expected_keys = {
                        "request_sha256",
                        "run_id",
                        "session_id",
                        "sequence",
                        "operation",
                        "policy_sha256",
                        "unit_id",
                        "status",
                        "payload",
                    }
                    if (
                        set(response) != expected_keys
                        or response["request_sha256"] != expected_hash
                        or any(
                            response[key] != request[key]
                            for key in (
                                "run_id",
                                "session_id",
                                "sequence",
                                "operation",
                                "policy_sha256",
                                "unit_id",
                            )
                        )
                    ):
                        raise ValueError("archive response identity differs from request")
                    if response["status"] == "storage_blocked":
                        raise StorageBlocked(
                            "storage_blocked: archive request failed; "
                            "inspect retained parent status"
                        )
                    if response["status"] != "ok" or type(response["payload"]) is not dict:
                        raise ValueError("invalid archive response status")
                    return response["payload"]
                try:
                    progress = json.loads(self._reader("progress.json", self.policy.page_bytes))
                except FileNotFoundError:
                    progress = None
                if (
                    progress is not None
                    and progress.get("session_id") == request["session_id"]
                    and progress.get("sequence") == sequence
                    and progress.get("verified") != progress_value
                ):
                    progress_value = progress["verified"]
                    progress_time = time.monotonic()
                now = time.monotonic()
                operation_time = progress_time
                if (
                    progress is not None
                    and progress.get("session_id") == request["session_id"]
                    and progress.get("sequence") == sequence
                    and type(progress.get("operation_time")) in {int, float}
                    and progress_time <= progress["operation_time"] <= now
                ):
                    operation_time = progress["operation_time"]
                if now - started >= _HEALTH_SECONDS:
                    print(
                        "archive waiting: parent alive; scientific execution paused",
                        file=sys.stderr,
                        flush=True,
                    )
                    started = now
                if now - operation_time > _STALL_SECONDS:
                    raise StorageBlocked(
                        "storage_blocked: archive operation made no verified progress; "
                        "preserve pending files"
                    )
                time.sleep(0.05)
