"""Fresh-interpreter network/import denial around actual train/eval/replay/report."""

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from silent_cascade.validation import StrictModel


class PilotOfflineReport(StrictModel):
    evidence_kind: Literal["offline_smoke_diagnostic"] = "offline_smoke_diagnostic"
    training_updates: Literal[1] = 1
    evaluation_episodes: Literal[16] = 16
    replayed_episodes: Literal[1] = 1
    artifact_reports: Literal[1] = 1
    backward_macs: int = Field(gt=0)
    network_attempts: Literal[0] = 0
    optional_import_attempts: Literal[0] = 0
    forbidden_modules: tuple[str, ...]
    foundation_model_calls: Literal[0] = 0
    config_sha256: str
    source_commit: str
    executed_source_sha256: str
    weights_sha256: str
    manifest_sha256: str
    replay_sha256: str
    artifact_report_sha256: str


@dataclass(frozen=True)
class OfflineGitPin:
    executable: str
    version: str
    sha256: str


_BOUNDARY_GIT_PIN: OfflineGitPin | None = None
_GIT_PIN_ENV = "SILENT_CASCADE_OFFLINE_GIT_PIN"


def _git_identity(path):
    resolved = Path(path).resolve(strict=True)
    if str(resolved) != path or not resolved.is_file():
        raise RuntimeError("offline Git pin is not a real executable")
    with resolved.open("rb") as source:
        info = os.fstat(source.fileno())
        if info.st_uid not in {0, os.getuid()} or info.st_mode & 0o022 or not info.st_mode & 0o111:
            raise RuntimeError("offline Git executable ownership/mode is untrusted")
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return digest


def resolve_offline_git() -> OfflineGitPin:
    """Operational parent prerequisite: local Git >=2.45, no install or network."""
    candidate = shutil.which("git")
    if candidate is None:
        raise RuntimeError("offline provenance requires trusted Git >=2.45")
    path = str(Path(candidate).resolve(strict=True))
    digest = _git_identity(path)
    completed = subprocess.run(
        [path, "--version"],
        capture_output=True,
        timeout=5,
        check=True,
        env={
            "PATH": os.defpath,
            "LANG": "C",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
        },
    )
    match = re.fullmatch(rb"git version ([0-9]+\.[0-9]+\.[0-9]+)(?:[^\n]*)\n?", completed.stdout)
    if match is None or len(completed.stdout) > 256:
        raise RuntimeError("offline provenance requires trusted Git >=2.45")
    version = match[1].decode()
    if tuple(map(int, version.split(".")[:2])) < (2, 45):
        raise RuntimeError("offline provenance requires trusted Git >=2.45")
    return OfflineGitPin(path, version, digest)


def _decode_git_pin(raw):
    if type(raw) is not str or len(raw) > 4096:
        raise RuntimeError("invalid offline Git pin")
    value = json.loads(raw)
    if type(value) is not dict or set(value) != {"executable", "version", "sha256"}:
        raise RuntimeError("invalid offline Git pin")
    if (
        any(type(item) is not str for item in value.values())
        or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", value["version"])
        or tuple(map(int, value["version"].split(".")[:2])) < (2, 45)
        or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
        or _git_identity(value["executable"]) != value["sha256"]
    ):
        raise RuntimeError("offline Git executable pin differs")
    return OfflineGitPin(**value)


def offline_environment(scratch: Path, *, git_pin: OfflineGitPin | None = None) -> dict[str, str]:
    """Closed child environment; every runtime cache stays in counted scratch."""
    scratch = scratch.absolute()
    pin = git_pin or _BOUNDARY_GIT_PIN or resolve_offline_git()
    return {
        key: value
        for key, value in {
            "PATH": os.defpath,
            _GIT_PIN_ENV: json.dumps(asdict(pin), sort_keys=True, separators=(",", ":")),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONCOERCECLOCALE": "0",
            **(
                {"__CF_USER_TEXT_ENCODING": f"0x{os.getuid():X}:0x0:0x0"}
                if sys.platform == "darwin"
                else {}
            ),
            "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_ALLOW_PROTOCOL": "",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_LITERAL_PATHSPECS": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_PAGER": "",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.fsmonitor",
            "GIT_CONFIG_VALUE_0": "false",
            "TMPDIR": str(scratch),
            "TMP": str(scratch),
            "TEMP": str(scratch),
            "MPLCONFIGDIR": str(scratch / "matplotlib-cache"),
            "MPL_IGNORE_SYSTEM_FONTS": "1",
            "XDG_CACHE_HOME": str(scratch / "cache"),
        }.items()
    }


_GIT_PREFIX = (
    "--no-pager",
    "--no-replace-objects",
    "--no-lazy-fetch",
    "--no-optional-locks",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "log.showSignature=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "submodule.recurse=false",
    "-c",
    "status.submoduleSummary=false",
)
_GIT_LOG_OPTIONS = ("--no-ext-diff", "--no-textconv", "--no-show-signature")


def _closed_git_arguments(arguments):
    """Only the local object/history forms used by existing provenance readers."""
    values = tuple(arguments)
    if not values or any(type(value) is not str for value in values):
        raise RuntimeError("offline child forbids external command")

    def revision(value):
        return bool(re.fullmatch(r"(?:HEAD|[0-9a-f]{40})(?:\^\{commit\})?", value))

    def path(value):
        return bool(value) and not value.startswith(("/", "-")) and ".." not in Path(value).parts

    command, *tail = values
    admitted = False
    if command == "rev-parse":
        tail = tail[1:] if tail[:1] == ["--verify"] else tail
        admitted = len(tail) == 1 and revision(tail[0])
    elif command == "merge-base":
        admitted = len(tail) == 3 and tail[0] == "--is-ancestor" and all(map(revision, tail[1:]))
    elif command == "ls-files":
        admitted = not tail
    elif command == "rev-list":
        admitted = len(tail) == 2 and tail[0] == "--topo-order" and revision(tail[1])
    elif command == "status":
        admitted = tail == ["--porcelain", "--untracked-files=normal"] or (
            tail[:3] == ["--porcelain=v1", "--untracked-files=all", "--"]
            and len(tail) > 3
            and all(map(path, tail[3:]))
        )
    elif command == "cat-file":
        admitted = tail == ["--batch"]
        if len(tail) == 2 and tail[0] in {"-s", "blob"}:
            oid, separator, name = tail[1].partition(":")
            admitted = bool(re.fullmatch(r"[0-9a-f]{40}", oid)) and (
                not separator or (tail[0] == "blob" and path(name))
            )
    elif command == "ls-tree":
        if tail[:2] == ["-r", "--name-only"]:
            tail = tail[2:]
        elif tail[:1] in (["-z"], ["-rz"]):
            tail = tail[1:]
        else:
            tail = []
        admitted = (
            bool(tail)
            and revision(tail[0])
            and (
                len(tail) == 1
                or (tail[1:2] == ["--"] and len(tail) > 2 and all(map(path, tail[2:])))
            )
        )
    elif command == "log" and tail[:2] == ["-1", "--format=%H"]:
        tail = tail[2:]
        if tail and revision(tail[0]):
            tail = tail[1:]
        admitted = tail[:1] == ["--"] and len(tail) > 1 and all(map(path, tail[1:]))
    if not admitted:
        raise RuntimeError("offline child forbids external command")
    if command == "status":
        return (*values[:1], "--ignore-submodules=none", *values[1:])
    return values[:1] + (_GIT_LOG_OPTIONS if command == "log" else ()) + values[1:]


def install_offline_boundary(*, workspace_root: Path | None = None):
    """Install irreversible child-only denial hooks, retaining read-only Git."""
    import importlib.abc
    import socket
    import threading
    import urllib.request

    global _BOUNDARY_GIT_PIN
    pin = _decode_git_pin(os.environ.get(_GIT_PIN_ENV))
    _BOUNDARY_GIT_PIN = pin
    git_prefix = (pin.executable, *_GIT_PREFIX)
    workspace = None if workspace_root is None else Path(workspace_root).resolve(strict=True)
    workspace_device = None if workspace is None else workspace.stat().st_dev
    descriptor_open = threading.local()
    trusted_repo = Path(__file__).resolve().parents[3]
    trusted_environment = offline_environment(Path(os.environ["TMPDIR"]), git_pin=pin)
    original_popen = subprocess.Popen

    def closed_popen(args, *positional, **kwargs):
        if isinstance(args, (list, tuple)) and args and args[0] in {"git", pin.executable}:
            if (
                positional
                or kwargs.get("shell", False)
                or kwargs.get("executable") not in {None, "git", pin.executable}
                or Path(kwargs.get("cwd") or Path.cwd()) != trusted_repo
                or dict(kwargs.get("env") if kwargs.get("env") is not None else os.environ)
                != trusted_environment
            ):
                raise RuntimeError("offline child forbids untrusted Git context")
            arguments = _closed_git_arguments(args[1:])
            args = (*git_prefix, *arguments)
            kwargs.update(
                executable=pin.executable, cwd=trusted_repo, env=dict(trusted_environment)
            )
            descriptor_open.git_batch_pipe = (
                arguments == ("cat-file", "--batch") and kwargs.get("stdin") == subprocess.PIPE
            )
        try:
            return original_popen(args, *positional, **kwargs)
        finally:
            descriptor_open.git_batch_pipe = False

    def descriptor_path(descriptor):
        if sys.platform == "darwin":
            import fcntl

            return Path(os.fsdecode(fcntl.fcntl(descriptor, 50, bytes(1024)).split(b"\0", 1)[0]))
        return Path(os.readlink(f"/proc/self/fd/{descriptor}"))

    def check_write(path, directory_fd=None, *, mkdir=False):
        if workspace is None:
            return
        if isinstance(path, int):
            if getattr(descriptor_open, "git_batch_pipe", False) and stat.S_ISFIFO(
                os.fstat(path).st_mode
            ):
                return
            target = descriptor_path(path)
        else:
            target = Path(os.fsdecode(path))
            if not target.is_absolute() and directory_fd not in {None, -1}:
                target = descriptor_path(directory_fd) / target
        target = target.absolute()
        resolved = target.resolve()
        if not resolved.is_relative_to(workspace):
            if mkdir and workspace.is_relative_to(resolved) and resolved.is_dir():
                # Existing safe publishers attempt mkdir while walking from /.
                # Report EEXIST without issuing an out-of-workspace mutation.
                raise FileExistsError(target)
            raise RuntimeError("offline output escapes counted workspace")
        for parent in (target, *target.parents):
            if parent.exists():
                if parent.is_symlink() or parent.stat().st_dev != workspace_device:
                    raise RuntimeError("offline output uses symlink or second volume")
                break

    original_open = os.open

    def bounded_open(path, flags, mode=0o777, *, dir_fd=None):
        writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        if writing:
            check_write(path, dir_fd)
        descriptor_open.validated = bool(writing)
        try:
            return original_open(path, flags, mode, dir_fd=dir_fd)
        finally:
            descriptor_open.validated = False

    attempts = {"network": 0, "imports": 0}
    blocked = (
        "mlx",
        "mlx_vlm",
        "transformers",
        "qwen_vl_utils",
        "huggingface_hub",
        "openai",
        "boto3",
        "botocore",
        "s3fs",
        "aiobotocore",
        "awscli",
        "google",
        "azure",
    )
    for name in tuple(os.environ):
        if name.startswith(("AWS_", "R2_", "CLOUDFLARE_", "GOOGLE_", "AZURE_", "OPENAI_")):
            del os.environ[name]

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in blocked or "qwen" in fullname.lower():
                attempts["imports"] += 1
                raise RuntimeError("forbidden optional model/cloud import")

    def deny(*args, **kwargs):
        attempts["network"] += 1
        raise RuntimeError("offline diagnostic forbids network/downloads")

    def audit(event, args):
        if event == "open" and not getattr(descriptor_open, "validated", False):
            path, _mode, flags = args
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                check_write(path)
        if event in {"os.mkdir", "os.remove", "os.rmdir"}:
            check_write(
                args[0], args[2] if event == "os.mkdir" else args[1], mkdir=event == "os.mkdir"
            )
        if event in {"os.rename", "os.link"}:
            check_write(args[0], args[2])
            check_write(args[1], args[3])
        if event == "os.symlink" and workspace is not None:
            raise RuntimeError("offline outputs cannot contain symlinks")
        if event == "os.truncate":
            check_write(args[0])
        if event.startswith("socket.") and event != "socket.__new__":
            deny()
        if event in {"os.system", "os.exec", "os.posix_spawn", "os.spawn"}:
            raise RuntimeError("offline child forbids external command")
        if event == "subprocess.Popen":
            executable, argv, _cwd, _env = args
            if (
                workspace is not None
                and executable == sys.executable
                and isinstance(argv, (list, tuple))
                and len(argv) == 5
                and list(argv[:4]) == [sys.executable, "-B", "-c", _PROGRAM]
                and type(argv[4]) is str
                and Path(argv[4]).is_absolute()
                and _cwd is not None
                and Path(_cwd) == Path(__file__).resolve().parents[3]
                and _env == offline_environment(Path(argv[4]))
            ):
                check_write(argv[4])
                return
            if (
                executable != pin.executable
                or not isinstance(argv, (list, tuple))
                or tuple(argv[: len(git_prefix)]) != git_prefix
                or _cwd != trusted_repo
                or _env != trusted_environment
            ):
                raise RuntimeError("offline child permits only read-only Git provenance")
            arguments = tuple(argv[len(git_prefix) :])
            original_arguments = arguments
            if arguments[:1] == ("log",) and arguments[1:4] == _GIT_LOG_OPTIONS:
                original_arguments = arguments[:1] + arguments[4:]
            if arguments[:2] == ("status", "--ignore-submodules=none"):
                original_arguments = arguments[:1] + arguments[2:]
            if _closed_git_arguments(original_arguments) != arguments:
                raise RuntimeError("offline child forbids external command")

    sys.meta_path.insert(0, Blocker())
    sys.addaudithook(audit)
    subprocess.Popen = closed_popen
    os.open = bounded_open
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    socket.getaddrinfo = deny
    socket.socket.sendto = deny
    urllib.request.urlopen = deny
    urllib.request.urlretrieve = deny
    import torch

    torch.hub.download_url_to_file = deny
    torch.hub.load_state_dict_from_url = deny
    return attempts, blocked


_PROGRAM = r"""
import sys
from silent_cascade.train.pilot_offline import _worker, install_offline_boundary
attempts, blocked = install_offline_boundary(workspace_root=sys.argv[1])
_worker(sys.argv[1], attempts, blocked)
"""


def measure_pilot_offline(*, output_dir):
    from silent_cascade.train.pilot_data import _publish_pilot_bytes, _read_pilot_bytes

    root = Path(__file__).resolve().parents[3]
    git_pin = _BOUNDARY_GIT_PIN or resolve_offline_git()
    # Persist intent before the child starts; all outputs/errors survive failures.
    _publish_pilot_bytes(
        output_dir / "intent.json", b'{"evidence_kind":"offline_smoke_diagnostic"}'
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _PROGRAM, str(output_dir.absolute())],
        cwd=root,
        env=offline_environment(output_dir, git_pin=git_pin),
        capture_output=True,
        timeout=180,
    )
    _publish_pilot_bytes(output_dir / "stdout.txt", completed.stdout)
    _publish_pilot_bytes(output_dir / "stderr.txt", completed.stderr)
    if completed.returncode:
        raise ValueError(
            f"offline probe failed exit{completed.returncode}: {completed.stderr.decode()[-3000:]}"
        )
    return PilotOfflineReport.model_validate_json(_read_pilot_bytes(output_dir / "offline.json"))


def _worker(directory, attempts, blocked):
    import torch

    from silent_cascade.env.pilot import curriculum_to_bundle
    from silent_cascade.eval.artifacts import EvaluationIdentity
    from silent_cascade.eval.runner import evaluate_episodes
    from silent_cascade.eventflow.engine import EventEngine
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent, NeuralModelIdentity
    from silent_cascade.eventflow.neural_replay import verify_neural_replay, write_neural_replay
    from silent_cascade.eventflow.neural_weights import save_neural_weights
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.report.pilot import build_pilot_report
    from silent_cascade.rng import seed_all
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_data import (
        freeze_pilot_manifest,
        iter_pilot_examples,
        next_pilot_batch,
    )
    from silent_cascade.train.pilot_trainer import pilot_train_one_step, publish_json
    from silent_cascade.train.trainer import make_optimizer

    output_dir = Path(directory)
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    # Exact loaded file bytes are included even in a precommit unit-test invocation.
    source = {
        str(p.relative_to(root)): sha256_bytes(p.read_bytes())
        for p in sorted((root / "src/silent_cascade").rglob("*.py"))
    }
    publish_json(output_dir / "executed-source.json", source)
    config = resolve_pilot_config("phase4_smoke")
    seed_all(11)
    model = EventFlowModel(config.config.neural)
    step = pilot_train_one_step(
        model,
        make_optimizer(model, config.config.pilot),
        next_pilot_batch(config.config, stage="one_hop", batch_counter=0),
        config.config,
    )
    publish_json(output_dir / "step.json", step)
    model_identity = NeuralModelIdentity.from_model(model, source_revision=revision)
    weights = output_dir / "weights.safetensors"
    weights_sha = save_neural_weights(weights, model=model, identity=model_identity)
    manifest = freeze_pilot_manifest(
        config, stage="primary", output_path=output_dir / "manifest.json", source_commit=revision
    )
    bundles = tuple(
        curriculum_to_bundle(e, config=config.config)
        for e in iter_pilot_examples(manifest, config=config.config)
    )
    identity = EvaluationIdentity(
        experiment="phase4-offline-diagnostic",
        stage="primary",
        split="debug",
        purpose="debug",
        manifest_schema=manifest.schema_version,
        manifest_sha256=sha256_bytes(canonical_json_bytes(manifest)),
        episodes=tuple(e.projected for e in manifest.entries),
        checkpoint_sha256=weights_sha,
        model_identity=model_identity,
        evaluation_config_canonical_json=config.canonical_json.decode(),
        execution_source_revision=revision,
    )
    evaluation = evaluate_episodes(
        model,
        identity=identity,
        config=config.config,
        episodes=bundles,
        output_dir=output_dir / "run/eval/primary",
        device="cpu",
    )
    agent = NeuralEventFlowAgent(model, identity=model_identity, device="cpu")
    engine = EventEngine(config.config.event_flow)
    replay_path = output_dir / "replay.json"
    with torch.no_grad():
        result = engine.run_episode(bundles[0], agent)
    write_neural_replay(
        replay_path,
        bundle=bundles[0],
        result=result,
        identity=model_identity,
        config=config.config.event_flow,
        weights=weights,
        source_revision=revision,
        experiment_config_canonical_json=config.canonical_json.decode(),
    )
    replay = verify_neural_replay(replay_path, weights_path=weights)
    report = build_pilot_report(run_dir=output_dir / "run", output_dir=output_dir / "report")
    forbidden = tuple(n for n in sys.modules if n.split(".")[0] in blocked or "qwen" in n.lower())
    if forbidden or not replay.matched:
        raise ValueError("offline replay/import probe failed")
    publish_json(
        output_dir / "offline.json",
        PilotOfflineReport(
            backward_macs=step.compute.backward_macs,
            network_attempts=attempts["network"],
            optional_import_attempts=attempts["imports"],
            forbidden_modules=forbidden,
            foundation_model_calls=step.compute.foundation_model_calls
            + evaluation.metrics.foundation_model_calls
            + result.counters.foundation_model_calls
            + replay.result.counters.foundation_model_calls,
            config_sha256=config.sha256,
            source_commit=revision,
            executed_source_sha256=sha256_bytes(canonical_json_bytes(source)),
            weights_sha256=weights_sha,
            manifest_sha256=identity.manifest_sha256,
            replay_sha256=sha256_bytes(replay_path.read_bytes()),
            artifact_report_sha256=sha256_bytes(report.read_bytes()),
        ),
    )
