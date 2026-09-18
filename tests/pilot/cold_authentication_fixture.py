"""Test-only bounded real-source fit and strict path leases; no provider authority."""

import hashlib
import inspect
import json
import os
import shutil
import stat
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

BLOCK = 4096
MIB = 1024**2
FIT_BYTES = 64 * MIB
FIT_NAMES = 256
FIT_DIRECTORIES = 48
STAGES = ("one_hop", "two_hop", "primary", "robustness")


def rounded(size):
    return ((size + BLOCK - 1) // BLOCK) * BLOCK


def allocation(root):
    """Count every name, including transient hardlinks, without following links."""
    if not root.exists():
        return 0, 0, 0
    total = files = directories = 0
    for path in (root, *root.rglob("*")):
        info = path.lstat()
        assert stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode), path
        total += info.st_blocks * 512
        files += stat.S_ISREG(info.st_mode)
        directories += stat.S_ISDIR(info.st_mode)
    return total, files, directories


def closed_environment(scratch):
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GIT_", "AWS_", "R2_", "CLOUDFLARE_"))
    }
    return environment | {
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_AUTHOR_DATE": "2026-09-19T00:00:00+0000",
        "GIT_COMMITTER_DATE": "2026-09-19T00:00:00+0000",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "TMPDIR": str(scratch / "temp"),
        "TMP": str(scratch / "temp"),
        "TEMP": str(scratch / "temp"),
        "XDG_CACHE_HOME": str(scratch / "temp"),
        "MPLCONFIGDIR": str(scratch / "temp"),
    }


def git(root, *args):
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        env=closed_environment(root.parent),
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert len(result.stdout) + len(result.stderr) < 65536
    return result.stdout.strip()


def create_checkout(scratch):
    """Same exact copy closure as test_phase3_provenance, with a finite Git bound."""
    original = Path(__file__).resolve().parents[2]
    files = []
    directories = {Path(".")}
    for name in ("src", "configs", "scripts", "docs/superpowers"):
        for path in (original / name, *(original / name).rglob("*")):
            if "__pycache__" in path.parts:
                continue
            info = path.lstat()
            relative = path.relative_to(original)
            if stat.S_ISDIR(info.st_mode):
                directories.add(relative)
            else:
                assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1, path
                assert info.st_blocks * 512 >= info.st_size, path
                files.append(relative)
    files.extend(
        map(Path, ("pyproject.toml", "uv.lock", ".gitignore", "Makefile", ".python-version"))
    )
    mapping = Path("manifests/validation/phase3/delivery.json")
    if (original / mapping).exists():
        files.append(mapping)
    inventory = {}
    for relative in files:
        path = original / relative
        info = path.lstat()
        assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1
        assert info.st_blocks * 512 >= info.st_size
        inventory[relative] = (
            info.st_size,
            stat.S_IMODE(info.st_mode),
            hashlib.sha256(path.read_bytes()).digest(),
        )
        directories.update(relative.parents)
    assert os.statvfs(original).f_frsize == BLOCK
    trees = {parent for name in files for parent in name.parents}
    for directory in trees:
        children = {
            name.parts[len(directory.parts)] if directory != Path(".") else name.parts[0]
            for name in files
            if name.is_relative_to(directory)
        }
        # Six mode bytes, space, name, NUL and the SHA-1 object ID.
        body = sum(28 + len(name.encode()) for name in children)
        raw = body + 6 + len(str(body))
        assert raw + (raw >> 12) + (raw >> 14) + (raw >> 25) + 13 <= BLOCK

    def blob_bound(size):
        raw = size + 6 + len(str(size))
        compressed = raw + (raw >> 12) + (raw >> 14) + (raw >> 25) + 13
        return rounded(compressed)

    final_paths = files + [Path("pilot-data") / f"{stage}.json" for stage in STAGES]
    final_paths += [
        Path("pilot-data/audits") / stage / suffix
        for stage in STAGES
        for suffix in ("manifest.json", "report.json", "index.json")
    ]
    index_bound = 32 + sum(
        ((62 + len(str(name).encode()) + 1 + 7) // 8) * 8 for name in final_paths
    )
    assert index_bound <= 32768
    object_count = len(files) + len(trees) + 1 + 16 + 7 + 2
    directory_bound = len(directories) + 6 + 1 + 12 + object_count
    bound = (
        sum(rounded(value[0]) for value in inventory.values())
        + sum(blob_bound(value[0]) for value in inventory.values())
        + (len(trees) + 7 + 3) * BLOCK
        + 12 * blob_bound(24576)
        + 4 * blob_bound(256)
        + 12 * 24576
        + 4 * BLOCK
        + 24576
        + 131072
        + directory_bound * BLOCK
    )
    assert bound <= 12 * MIB, ("checkout admission exceeded", bound)
    assert not (scratch / "repo").exists()
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "temp").mkdir(exist_ok=True)
    template = scratch / "empty-template"
    template.mkdir()
    root = scratch / "repo"
    root.mkdir()
    for directory in sorted(directories, key=lambda path: len(path.parts)):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for relative, expected in inventory.items():
        source = original / relative
        assert (
            source.stat().st_size,
            stat.S_IMODE(source.stat().st_mode),
            hashlib.sha256(source.read_bytes()).digest(),
        ) == expected
        shutil.copy(source, root / relative)
        assert hashlib.sha256((root / relative).read_bytes()).digest() == expected[2]
    git(
        root,
        "init",
        "-q",
        "--object-format=sha1",
        "--initial-branch=main",
        f"--template={template}",
    )
    for key, value in (
        ("user.email", "test@example.invalid"),
        ("user.name", "Local test"),
        ("core.hooksPath", os.devnull),
        ("commit.gpgSign", "false"),
        ("gc.auto", "0"),
        ("core.logAllRefUpdates", "false"),
    ):
        git(root, "config", key, value)
    git(root, "add", ".")
    git(root, "commit", "-qm", "reviewed source")
    assert allocation(scratch)[0] <= 12 * MIB
    print(
        json.dumps(
            {
                "checkout_bound": bound,
                "source_files": len(files),
                "source_commit": git(root, "rev-parse", "HEAD"),
            }
        )
    )
    return root


class FitGuard:
    """Reserve full next publication against current live names, never total history."""

    def __init__(self, spool, cache):
        self.spool, self.cache = spool, cache
        self.maximum = 0
        self.calls = {}
        self.blocked = None

    def admit(self, path, size, *, temporary=True):
        assert self.blocked is None, "fixture guard remains blocked: " + str(self.blocked)
        try:
            self._admit(path, size, temporary=temporary)
        except AssertionError as error:
            self.blocked = str(error)
            raise

    def _admit(self, path, size, *, temporary):
        path = path.absolute()
        assert path.is_relative_to(self.spool) or path.is_relative_to(self.cache), path
        assert size <= 16 * MIB, "fixture input exceeds admitted single-file lease"
        used = files = directories = 0
        for root in (self.spool, self.cache):
            values = allocation(root)
            used += values[0]
            files += values[1]
            directories += values[2]
        missing = sum(
            not parent.exists()
            for parent in path.parents
            if parent.is_relative_to(self.spool) or parent.is_relative_to(self.cache)
        )
        # Count every current allocation, the complete next temp (old target stays),
        # missing directories, and temp/final directory entries. A hardlink gets
        # charged twice conservatively, without calling the parent ledger here.
        names = 2 if temporary else 1
        projected = (
            used + 2 * rounded(size) + BLOCK + (directories + missing + files + names) * BLOCK
        )
        assert files + names <= FIT_NAMES, "fixture file name bound"
        assert directories + missing <= FIT_DIRECTORIES, "fixture directory bound"
        assert projected <= FIT_BYTES, ("fixture live allocation blocked before write", projected)
        self.maximum = max(self.maximum, projected)
        self.calls[path.suffix] = self.calls.get(path.suffix, 0) + 1

    @contextmanager
    def installed(self, run):
        from contextlib import ExitStack

        from silent_cascade import io
        from silent_cascade.train import (
            pilot_checkpoints,
            pilot_data,
            pilot_evidence_types,
            pilot_trainer,
            pilot_workflow,
        )

        def publication(original):
            def guarded(path, raw, *args, **kwargs):
                self.admit(path, len(raw))
                return original(path, raw, *args, **kwargs)

            return guarded

        original_open, original_mkdir, original_link = Path.open, Path.mkdir, os.link

        def linked(source, destination, *args, **kwargs):
            if isinstance(source, Path) and source.name == ".rows.pending.jsonl":
                self.admit(Path(destination), source.stat().st_size)
            return original_link(source, destination, *args, **kwargs)

        def mkdir(path, *args, **kwargs):
            if not path.exists() and (
                path.is_relative_to(self.spool) or path.is_relative_to(self.cache)
            ):
                self.admit(path / "directory-allowance", 0, temporary=False)
            return original_mkdir(path, *args, **kwargs)

        def opened(path, mode="r", *args, **kwargs):
            if "x" in mode and path.name == ".rows.pending.jsonl":
                self.admit(path, 0)
                handle = original_open(path, mode, *args, **kwargs)
                guard = self

                class Rows:
                    def __enter__(self):
                        return self

                    def __exit__(self, *exc):
                        return handle.__exit__(*exc)

                    def __getattr__(self, name):
                        return getattr(handle, name)

                    def write(self, raw):
                        guard.admit(path, handle.tell() + len(raw))
                        count = handle.write(raw)
                        handle.flush()
                        return count

                return Rows()
            return original_open(path, mode, *args, **kwargs)

        original_at = pilot_trainer._publish_bytes_at

        def at(parent, name, raw, **kwargs):
            # This trainer always publishes final checkpoints/index at its run FD.
            assert os.fstat(parent).st_ino == run.stat().st_ino
            self.admit(run / name, len(raw))
            return original_at(parent, name, raw, **kwargs)

        with ExitStack() as stack:
            for module in (
                pilot_data,
                pilot_trainer,
                pilot_checkpoints,
                pilot_workflow,
                pilot_evidence_types,
            ):
                stack.enter_context(
                    patch.object(
                        module, "_publish_pilot_bytes", publication(module._publish_pilot_bytes)
                    )
                )
            stack.enter_context(patch.object(pilot_trainer, "_publish_bytes_at", at))
            stack.enter_context(patch.object(io, "_durable_temp", publication(io._durable_temp)))
            stack.enter_context(patch.object(Path, "open", opened))
            stack.enter_context(patch.object(Path, "mkdir", mkdir))
            stack.enter_context(patch.object(os, "link", linked))
            yield


def introduce_data(root, config):
    from contextlib import ExitStack

    from silent_cascade.eval import pilot_audit
    from silent_cascade.train import pilot_data
    from silent_cascade.train.pilot_provenance import authenticate_pilot_source

    producer = git(root, "rev-parse", "HEAD")
    authenticate_pilot_source(repo_root=root, source_commit=producer, config=config)
    expected = {root / "pilot-data" / f"{stage}.json": 24576 for stage in STAGES}
    expected.update(
        {
            root / "pilot-data/audits" / stage / name: (256 if name == "index.json" else 24576)
            for stage in STAGES
            for name in ("manifest.json", "report.json", "index.json")
        }
    )
    seen = set()
    original = pilot_data._publish_pilot_bytes

    def publish(path, raw):
        assert path in expected and path not in seen and len(raw) <= expected[path], path
        assert allocation(root.parent)[0] + 2 * rounded(len(raw)) + 16 * BLOCK <= 12 * MIB
        seen.add(path)
        original(path, raw)
        assert allocation(root.parent)[0] <= 12 * MIB

    with ExitStack() as stack:
        for module in (pilot_data, pilot_audit):
            stack.enter_context(patch.object(module, "_publish_pilot_bytes", publish))
        for stage in STAGES:
            manifest = pilot_data.freeze_pilot_manifest(
                config,
                stage=stage,
                output_path=root / "pilot-data" / f"{stage}.json",
                source_commit=producer,
            )
            pilot_audit.audit_pilot_manifest(
                manifest, config=config.config, output_dir=root / "pilot-data/audits" / stage
            )
    assert seen == set(expected)
    git(root, "add", "pilot-data")
    git(root, "commit", "-qm", "introduce immutable debug data")
    git(root, "commit", "--allow-empty", "-qm", "training attempt")
    assert allocation(root.parent)[0] <= 12 * MIB


class StrictContext:
    """Only one named new fixture input exists in the lease tree at any time."""

    def __init__(self, run, backing, cache, guard):
        from silent_cascade.archive.types import FileEntry

        self.run_dir, self.backing, self.cache, self.guard = run, backing, cache, guard
        self.inventory = tuple(
            FileEntry(
                path.relative_to(run).as_posix(),
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_size,
            )
            for path in sorted(run.rglob("*"))
            if path.is_file()
        )
        self.active = self.maximum = 0
        self.requests = []
        self.hidden = set()
        self.override = {}

    def entries(self):
        for entry in self.inventory:
            if entry.path not in self.hidden:
                yield entry

    def cool(self):
        for entry in self.inventory:
            if entry.path.startswith("journal-"):
                continue
            target = self.backing / entry.path
            self.guard.admit(target, 0, temporary=False)
            target.parent.mkdir(parents=True, exist_ok=True)
            (self.run_dir / entry.path).rename(target)

    @contextmanager
    def _leased(self, payload):
        assert set(payload) == {"path"} and self.active == 0
        name = payload["path"]
        assert name in {entry.path for entry in self.entries()}
        original = self.override.get(name, self.backing / name)
        assert original.is_file() and not (self.run_dir / name).exists()
        local = self.cache / name
        self.guard.admit(local, 0, temporary=False)
        local.parent.mkdir(parents=True, exist_ok=True)
        assert allocation(self.cache)[0] + original.stat().st_blocks * 512 + 48 * BLOCK <= 16 * MIB
        original.rename(local)
        self.active = 1
        self.maximum = max(self.maximum, self.active)
        self.requests.append(name)
        try:
            yield SimpleNamespace(local_root=self.cache)
        finally:
            local.rename(original)
            self.active = 0
            assert not local.exists()

    def restore(self):
        for entry in self.inventory:
            source = self.backing / entry.path
            target = self.run_dir / entry.path
            if source.exists():
                if target.exists():
                    assert target.read_bytes() == source.read_bytes()
                    duplicate = self.backing / "recovered-root-result.json"
                    assert not duplicate.exists()
                    target.rename(duplicate)
                source.rename(target)
            assert target.stat().st_size == entry.bytes
            assert hashlib.sha256(target.read_bytes()).hexdigest() == entry.sha256
        assert self.active == 0 and not any(path.is_file() for path in self.cache.rglob("*"))


def assert_rng_equal(before, after):
    import numpy as np
    import torch

    assert before.python_state == after.python_state
    assert before.numpy_state[0] == after.numpy_state[0]
    np.testing.assert_array_equal(before.numpy_state[1], after.numpy_state[1])
    assert before.numpy_state[2:] == after.numpy_state[2:]
    assert torch.equal(before.torch_cpu_state, after.torch_cpu_state)


def run_fixture(spool, cache, reuse):
    import torch

    from silent_cascade.errors import SilentCascadeError
    from silent_cascade.rng import snapshot_global_rng
    from silent_cascade.train import pilot_workflow
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_evidence import authenticate_run
    from silent_cascade.train.pilot_provenance import authenticate_pilot_source, verify_pilot_data

    root, run = Path.cwd(), spool / "run"
    config = resolve_pilot_config("phase4_smoke")
    assert config.config.pilot.max_steps == 4 and config.config.pilot.validation_every_steps == 2
    if not reuse:
        introduce_data(root, config)
    source_commit = git(root, "rev-parse", "HEAD")
    manifests = {stage: root / "pilot-data" / f"{stage}.json" for stage in STAGES}
    loaded, introductions = verify_pilot_data(
        repo_root=root, source_commit=source_commit, config=config, manifests=manifests
    )
    source = authenticate_pilot_source(
        repo_root=root, source_commit=source_commit, config=config
    ).model_copy(update={"data_introductions": introductions})
    guard = FitGuard(spool, cache)
    with guard.installed(run):
        spool.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)
        if not reuse:
            fitted = pilot_workflow._train(
                config, manifests, run, "cpu", source_commit, source, None
            )
            assert guard.blocked is None, guard.blocked
            assert fitted.progress.global_step == 4 and fitted.status == "step_ceiling"
        before = snapshot_global_rng()
        eager = authenticate_run(run, config, repo_root=root)
        assert_rng_equal(before, snapshot_global_rng())
        assert eager[1] == source and eager[3] == loaded
        assert eager[2].progress.global_step == 4 and eager[2].progress.status == "step_ceiling"
        validations = sorted(run.glob("attempt-*/validation-*/validation.json"))
        assert len(validations) == 2
        assert sorted(json.loads(path.read_bytes())["global_step"] for path in validations) == [
            2,
            4,
        ]
        rows = list(run.glob("attempt-*/validation-*/autonomous/rows.jsonl"))
        assert len(rows) == 2 and all(len(path.read_bytes().splitlines()) == 16 for path in rows)
        assert all(
            json.loads(raw)["compute"]["foundation_model_calls"] == 0
            for path in rows
            for raw in path.read_bytes().splitlines()
        )
        assert eager[4] is not None and eager[5] == run / eager[2].progress.latest.path
        print(
            json.dumps(
                {
                    "completed_fit": True,
                    "source_commit": source_commit,
                    "source_sha256": source.source_sha256,
                    "global_step": 4,
                    "validation_episodes": 32,
                    "status": eager[2].status,
                    "model_state_sha256": eager[4].identity.model_state_sha256,
                }
            ),
            flush=True,
        )

        def tripwire(*args, **kwargs):
            raise AssertionError("completed reuse attempted neural training")

        context = StrictContext(run, spool / "backing", cache, guard)
        with patch.object(pilot_workflow, "run_pilot_training", tripwire):
            assert (
                pilot_workflow._train(config, manifests, run, "cpu", source_commit, source, None)
                == eager[2]
            )
            context.cool()
            try:
                kwargs = {"repo_root": root}
                # RED must reach the real cold reader, not stop at a new keyword.
                if "evidence_context" in inspect.signature(authenticate_run).parameters:
                    kwargs["evidence_context"] = context
                before = snapshot_global_rng()
                cold = authenticate_run(run, config, **kwargs)
                assert_rng_equal(before, snapshot_global_rng())
                assert cold[:4] == eager[:4] and cold[5] == eager[5]
                assert cold[4].identity == eager[4].identity
                for name, tensor in eager[4].model.state_dict().items():
                    assert torch.equal(tensor, cold[4].model.state_dict()[name]), name
                assert context.maximum == 1 and context.active == 0
                reused = pilot_workflow._train(
                    config,
                    manifests,
                    run,
                    "cpu",
                    source_commit,
                    source,
                    None,
                    evidence_context=context,
                )
                assert reused == eager[2]
                context.hidden.add("training-result.json")
                reused = pilot_workflow._train(
                    config,
                    manifests,
                    run,
                    "cpu",
                    source_commit,
                    source,
                    None,
                    evidence_context=context,
                )
                assert reused == eager[2]
                context.hidden.clear()
                # Retain the newly repaired publication and return its logical path
                # to cold state so later negative controls exercise result leases.
                repaired = run / "training-result.json"
                assert (
                    repaired.read_bytes() == (context.backing / "training-result.json").read_bytes()
                )
                prefix = "verification-repaired-" if reuse else "repaired-"
                repaired_target = spool / (prefix + "result.json")
                assert not repaired_target.exists()
                repaired.rename(repaired_target)
                for path in run.glob("attempt-*/artifact-index.*.jsonl.gz"):
                    assert (
                        path.read_bytes() == (context.backing / path.relative_to(run)).read_bytes()
                    )
                    repaired_target = spool / (prefix + path.name)
                    assert not repaired_target.exists()
                    path.rename(repaired_target)
                wrong = run / "wrong-root"
                for invoke in (
                    lambda: authenticate_run(
                        wrong, config, repo_root=root, evidence_context=context
                    ),
                    lambda: pilot_workflow._train(
                        config,
                        manifests,
                        wrong,
                        "cpu",
                        source_commit,
                        source,
                        None,
                        evidence_context=context,
                    ),
                ):
                    try:
                        invoke()
                    except ValueError as error:
                        assert "root" in str(error)
                    else:
                        raise AssertionError("foreign logical root accepted")
                envelope = json.loads((context.backing / "training-result.json").read_bytes())
                shard = envelope["artifact_index"]["shards"][-1]["path"]
                last_artifact = max(eager[2].artifact_hashes)
                control = spool / "damaged-control"
                if control.exists():
                    assert control.read_bytes() == b"damaged"
                else:
                    guard.admit(control, 7, temporary=False)
                    control.write_bytes(b"damaged")
                for name in (
                    shard,
                    last_artifact,
                    eager[2].progress.latest.path,
                    eager[2].latest_weights.path,
                ):
                    context.override[name] = control
                    start = len(context.requests)
                    try:
                        authenticate_run(run, config, repo_root=root, evidence_context=context)
                    except (ValueError, SilentCascadeError):
                        assert name in context.requests[start:]
                    else:
                        raise AssertionError("corrupted cold input accepted: " + name)
                    finally:
                        context.override.clear()
                    assert context.active == 0 and control.read_bytes() == b"damaged"
                assert cold[2] == eager[2]
            finally:
                context.restore()
        assert guard.blocked is None, guard.blocked
        print(
            json.dumps(
                {
                    "cold_authentication": "passed",
                    "single_lease_maximum": context.maximum,
                    "guard_peak_reserved_bytes": guard.maximum,
                    "spool_allocated": allocation(spool)[0],
                    "cache_allocated": allocation(cache)[0],
                    "scratch_allocated": allocation(root.parent)[0],
                    "guard_calls": guard.calls,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    import traceback

    try:
        run_fixture(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "reuse")
    except Exception as error:
        # Keep failed diagnostics finite without removing any scientific prefix.
        print(
            "".join(traceback.format_list(traceback.extract_tb(error.__traceback__)[-24:])),
            file=sys.stderr,
        )
        print(type(error).__name__ + ": " + str(error)[:4096], file=sys.stderr)
        raise SystemExit(1) from None
