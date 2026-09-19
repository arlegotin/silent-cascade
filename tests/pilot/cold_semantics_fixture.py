"""Read-only historical evidence context for cold semantic-reader tests."""

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

MIB = 1024**2
FIXTURE_RELATIVE = Path(
    ".superpowers/sdd/2026-09-17-phase-4-r2-archive/tmp/task-4/"
    "cover-pilot-final/test_real_smoke_workflow_reuse0/repo/runs/smoke"
)
OFFLINE_ROOT = "final/offline/run/eval/primary"
NUMERIC_RUNTIME_ROOT = "final/numerics/runtime-cpu"


def historical_root():
    root = Path(__file__).resolve().parents[2] / FIXTURE_RELATIVE
    return root if root.is_dir() else None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def allocated(root):
    return sum(path.lstat().st_blocks * 512 for path in (root, *root.rglob("*")))


class HistoricalContext:
    """Expose original files only during an exact file/metadata/episode lease."""

    def __init__(self, run_dir, backing, entries, *, evaluation_root=OFFLINE_ROOT):
        from silent_cascade.archive.types import FileEntry

        self.run_dir = run_dir.absolute()
        self.backing = backing.absolute()
        self._entries = tuple(
            FileEntry(name, digest, (self.backing / name).stat().st_size)
            for name, digest in sorted(entries.items())
        )
        self._entry_names = frozenset(entry.path for entry in self._entries)
        self.active = 0
        self.maximum = 0
        self.payload_active = 0
        self.payload_maximum = 0
        self.requests = []
        self.hidden = set()
        self.overrides = {}
        self.late_failure = None
        self._allowed = frozenset()
        self.evaluation_root = evaluation_root
        self._commits = self._episode_commits(evaluation_root)

    def entries(self):
        for entry in self._entries:
            if entry.path not in self.hidden:
                yield entry
        if self.late_failure is not None:
            raise self.late_failure

    def evaluation_roots(self):
        return (self.evaluation_root,)

    @contextmanager
    def _scope(self, kind, names, root=None):
        payload = kind.startswith(("file:", "episode:"))
        metadata = kind.startswith("metadata:")
        assert payload or metadata
        if metadata:
            assert self.active == 0, "metadata leases cannot overlap"
        else:
            assert self.payload_active == 0, "semantic readers must release the prior payload"
        root = self.backing if root is None else Path(root).absolute()
        previous = self._allowed
        self.active = 1
        if metadata:
            self.active = 1
        else:
            self.active += int(bool(previous))
            self.payload_active = 1
            self.payload_maximum = max(self.payload_maximum, self.payload_active)
        self.maximum = max(self.maximum, self.active)
        self._allowed = previous | frozenset((root / name).absolute() for name in names)
        self.requests.append(kind)
        try:
            yield root
        finally:
            self._allowed = previous
            if payload:
                self.payload_active = 0
            self.active = int(bool(previous))

    @contextmanager
    def _leased(self, payload):
        assert set(payload) == {"path"}
        name = payload["path"]
        assert name in self._entry_names and name not in self.hidden
        root = self.overrides.get(name, self.backing)
        if root == self.backing:
            assert (root / name).is_file()
        with self._scope("file:" + name, (name,), root) as local_root:
            yield SimpleNamespace(local_root=local_root)

    @contextmanager
    def metadata(self, logical_root):
        assert logical_root == self.evaluation_root
        names = tuple(
            f"{logical_root}/{name}"
            for name in (
                "DONE",
                "identity.json",
                "retention.json",
                "rows.jsonl",
                "index.json",
                "metrics.json",
            )
        )
        ref = SimpleNamespace(logical_root=logical_root, kind="evaluation_metadata")
        root = self.overrides.get(("metadata", logical_root), self.backing)
        with self._scope("metadata:" + logical_root, names, root) as local_root:
            yield SimpleNamespace(local_root=local_root, ref=ref)

    def episode_commit(self, logical_root, ordinal):
        assert logical_root == self.evaluation_root
        return self._commits[ordinal]

    @contextmanager
    def episode(self, logical_root, ordinal, *, commit_sha256):
        from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

        commit = self.episode_commit(logical_root, ordinal)
        assert commit_sha256 == sha256_bytes(canonical_json_bytes(asdict(commit)))
        names = tuple(entry.path for entry in (*commit.owned, *commit.borrowed))
        key = (logical_root, ordinal)
        root = self.overrides.get(key, self.backing)
        ref = SimpleNamespace(logical_root=logical_root)
        with self._scope(f"episode:{logical_root}:{ordinal}", names, root) as local_root:
            yield SimpleNamespace(local_root=local_root, ref=ref)

    def _episode_commits(self, logical_root):
        from silent_cascade.archive.types import EpisodeCommit, FileEntry

        root = self.backing / logical_root
        done = json.loads((root / "DONE").read_bytes())
        hashes = done["artifact_hashes"]
        identity = json.loads((root / "identity.json").read_bytes())
        index = json.loads((root / "index.json").read_bytes())["rows"]
        rows = (root / "rows.jsonl").read_bytes().splitlines(keepends=True)
        assert len(rows) == len(index) == len(identity["episodes"]) == 16
        assert "crashes/index.json" not in hashes
        weights = f"crashes/weights-{identity['checkpoint_sha256']}.safetensors"
        result = []
        for ordinal, (raw, indexed, binding) in enumerate(
            zip(rows, index, identity["episodes"], strict=True)
        ):
            row = json.loads(raw)
            owned = {
                row["neural_trace_ref"],
                row["full_trace_ref"],
                f"episodes/{ordinal:05d}.telemetry.json",
            }
            owned.discard(None)
            borrowed = {weights} if weights in hashes and row["causal_trace_sha256"] else set()

            def entries(names):
                return tuple(
                    FileEntry(f"{logical_root}/{name}", hashes[name], (root / name).stat().st_size)
                    for name in sorted(names)
                )

            result.append(
                EpisodeCommit(
                    schema_version="phase4-evaluation-episode-commit-v1",
                    identity_sha256=done["identity_sha256"],
                    ordinal=ordinal,
                    episode_public_id=binding["public_id"],
                    episode_sha256=binding["episode_sha256"],
                    row_offset=indexed["offset"],
                    row_bytes=indexed["bytes"],
                    row_sha256=indexed["sha256"],
                    owned=entries(owned),
                    borrowed=entries(borrowed),
                )
            )
        return tuple(result)

    @contextmanager
    def guarded_reads(self, monkeypatch):
        """Guard descriptor-relative opens as well as Path convenience reads."""
        original_open = os.open
        original_path_open = Path.open
        descriptors = {}

        def resolve(path, dir_fd):
            candidate = Path(path)
            if candidate.is_absolute():
                return candidate.absolute()
            if dir_fd is None:
                return candidate.absolute()
            parent = descriptors.get(dir_fd)
            return None if parent is None else (parent / candidate).absolute()

        def authorize(candidate, flags):
            if candidate is None:
                return
            backing = self.backing
            if candidate == backing or candidate.is_relative_to(backing):
                assert self.active > 0, f"historical read outside lease: {candidate}"
                directory = bool(flags & getattr(os, "O_DIRECTORY", 0))
                allowed = candidate in self._allowed or any(
                    path.is_relative_to(candidate) for path in self._allowed
                )
                assert (directory and allowed) or candidate in self._allowed, (
                    "undeclared historical read",
                    candidate,
                )
            elif backing.is_relative_to(candidate):
                assert flags & getattr(os, "O_DIRECTORY", 0)

        def guarded_open(path, flags, mode=0o777, *, dir_fd=None):
            candidate = resolve(path, dir_fd)
            authorize(candidate, flags)
            descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
            if candidate is not None and flags & getattr(os, "O_DIRECTORY", 0):
                descriptors[descriptor] = candidate
            return descriptor

        def guarded_path_open(path, *args, **kwargs):
            candidate = path.absolute()
            if candidate == self.backing or candidate.is_relative_to(self.backing):
                assert self.active > 0 and candidate in self._allowed, (
                    "historical Path read outside declared lease",
                    candidate,
                )
            return original_path_open(path, *args, **kwargs)

        monkeypatch.setattr(os, "open", guarded_open)
        monkeypatch.setattr(Path, "open", guarded_path_open)
        try:
            yield
        finally:
            assert self.active == 0
            assert self.payload_active == 0

    def corrupt_final_episode(self, control_root):
        """Copy only one bounded episode lease and alter its final telemetry."""
        assert not control_root.exists()
        commit = self._commits[-1]
        entries = (*commit.owned, *commit.borrowed)
        outputs = tuple(control_root / entry.path for entry in entries)
        assert len(entries) == len(outputs) == len(set(outputs)) <= 4
        assert all((self.backing / entry.path).stat().st_size == entry.bytes for entry in entries)
        directories = {
            parent
            for output in outputs
            for parent in output.relative_to(control_root).parents
            if parent != Path(".")
        }
        logical = sum(entry.bytes for entry in entries) + 1
        allocated_bound = (
            sum(
                ((entry.bytes + int("telemetry" in entry.path) + 4095) // 4096) * 4096
                for entry in entries
            )
            + (len(directories) + 1) * 4096
        )
        assert logical <= 5 * MIB
        assert len(directories) <= 8 and allocated_bound <= 6 * MIB
        for entry in entries:
            target = control_root / entry.path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.backing / entry.path, target)
        target = control_root / f"{self.evaluation_root}/episodes/00015.telemetry.json"
        target.write_bytes(target.read_bytes() + b" ")
        assert allocated(control_root) <= 6 * MIB
        assert sum(path.is_file() for path in control_root.rglob("*")) <= 4
        assert sum(path.is_dir() for path in control_root.rglob("*")) <= 8
        self.overrides[(self.evaluation_root, 15)] = control_root.absolute()


def relevant_inventory(gate):
    names = set()
    for record in gate["continuation_evidence"]:
        names.update((record["checkpoint_path"], record["replay_path"]))
        names.add(
            str(
                Path(record["checkpoint_path"]).parent
                / f"weights-{record['weights_sha256']}.safetensors"
            )
        )
    names.update(
        {
            "final/offline/executed-source.json",
            "final/offline/step.json",
            "final/offline/replay.json",
            "final/offline/weights.safetensors",
            "final/offline/report/report.md",
        }
    )
    names.update(
        name for name in gate["upstream_artifact_hashes"] if name.startswith(OFFLINE_ROOT + "/")
    )
    inventory = gate["upstream_artifact_hashes"]
    return {name: inventory[name] for name in names if name in inventory}


def numeric_inventory(gate):
    prefix = "final/numerics/"
    return {
        name: digest
        for name, digest in gate["upstream_artifact_hashes"].items()
        if name.startswith(prefix)
    }


def referenced_checkpoint(backing):
    root = next(backing.glob("attempt-*/validation-4-one_hop/autonomous/crashes"))
    checkpoint = next(
        path for path in root.glob("*.safetensors") if not path.name.startswith("weights-")
    )
    weights = next(root.glob("weights-*.safetensors"))
    assert checkpoint.stat().st_size < 512 * 1024
    assert weights.stat().st_size < 4 * MIB
    return checkpoint, weights


def isolated_referenced_checkpoint(backing, control_root):
    checkpoint, _ = referenced_checkpoint(backing)
    size = checkpoint.stat().st_size
    assert not control_root.exists() and size <= 512 * 1024
    assert ((size + 4095) // 4096) * 4096 + 4096 <= 520 * 1024
    control_root.mkdir()
    target = control_root / checkpoint.name
    shutil.copyfile(checkpoint, target)
    assert target.stat().st_size == size and sha256(target) == sha256(checkpoint)
    return target
