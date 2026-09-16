"""Verify completed Phase 1-3 deliveries in their authenticated historical checkout."""

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

from silent_cascade.errors import ArtifactIntegrityError, SilentCascadeError
from silent_cascade.hashing import canonical_json_bytes

_CHECKOUT = "ea713cdce4107e709985d9333050a947db6561ab"
_MANIFEST = "manifests/validation/historical-delivery.json"
_MANIFEST_CANONICAL_SHA256 = "8f3ffbf619ec4fa5a8a82c3645953458463def9d433f9f41dbb5ce4759b59deb"
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
_MAX_CHILD_OUTPUT_BYTES = 1024 * 1024
_CHILD_TIMEOUT_SECONDS = 600
_RECEIPTS: dict[tuple[object, ...], bytes] = {}

_CHILD = r"""
import json
import os
import runpy
import sys
from pathlib import Path

request = json.loads(os.environ["SILENT_CASCADE_HISTORICAL_REQUEST"])
root = Path(request["root"]).resolve()
source_root = (root / "src").resolve()
sys.path.insert(0, str(source_root))

def check_origins():
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name != "silent_cascade" and not name.startswith("silent_cascade."):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            raise RuntimeError(f"loaded package module lacks an origin: {name}")
        origin = Path(filename).resolve()
        if not origin.is_relative_to(source_root):
            raise RuntimeError(f"loaded package origin escapes historical checkout: {name}")
        package_paths = getattr(module, "__path__", None)
        if package_paths is not None and any(
            not Path(path).resolve().is_relative_to(source_root) for path in package_paths
        ):
            raise RuntimeError(f"loaded package search path escapes historical checkout: {name}")
        origins[name] = str(origin.relative_to(root))
    return dict(sorted(origins.items()))

try:
    namespace = runpy.run_path(str(root / request["verifier_path"]))
    check_origins()
    kwargs = {
        key: (root / value if key.endswith("_path") or key == "repo_root" else value)
        for key, value in request["kwargs"].items()
    }
    result = namespace[request["function"]](**kwargs)
    origins = check_origins()
    values = result.model_dump(mode="json")
    print(json.dumps({"ok": True, "origins": origins, "result": values}, sort_keys=True,
                     separators=(",", ":"), ensure_ascii=False, allow_nan=False))
except BaseException as error:
    print(json.dumps({"ok": False, "error_type": type(error).__name__,
                      "message": str(error)[:2000]}, sort_keys=True, separators=(",", ":")))
    raise SystemExit(1)
"""


def _fail(message: str, **context: str | int) -> ArtifactIntegrityError:
    return ArtifactIntegrityError(message, context=context)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in ("", ".", "..") for part in path.parts)
        or "\\" in value
    ):
        raise _fail("unsafe historical relative path", path=value)
    return path


def _regular_bytes(repo_root: Path, relative: str, *, max_bytes: int) -> bytes:
    path = repo_root / _relative_path(relative)
    cursor = repo_root
    for part in _relative_path(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise _fail("historical path has a symlinked parent", path=relative)
    try:
        metadata = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
            raise _fail("historical input must be a bounded regular file", path=relative)
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            raw = os.read(descriptor, max_bytes + 1)
        finally:
            os.close(descriptor)
    except ArtifactIntegrityError:
        raise
    except OSError as error:
        raise _fail("historical input must be a bounded regular file", path=relative) from error
    if len(raw) != metadata.st_size or len(raw) > max_bytes:
        raise _fail("historical input changed while reading", path=relative)
    return raw


def _run_git(repo_root: Path, *arguments: str, max_bytes: int = 1024 * 1024) -> bytes:
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            env=environment,
            check=True,
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise _fail("historical Git authentication failed") from error
    if completed.stderr or len(completed.stdout) > max_bytes:
        raise _fail("historical Git output is not bounded and quiet")
    return completed.stdout


def _regular_git_blob(repo_root: Path, revision: str, relative: str) -> bytes:
    path = _relative_path(relative).as_posix()
    entry = _run_git(repo_root, "ls-tree", "-z", revision, "--", path)
    try:
        header, found = entry.rstrip(b"\0").split(b"\t")
        mode, kind, object_id = header.split()
    except ValueError as error:
        raise _fail("missing historical regular Git blob", path=relative) from error
    if found.decode() != path or mode not in (b"100644", b"100755") or kind != b"blob":
        raise _fail("historical pin is not a regular Git blob", path=relative)
    try:
        size = int(_run_git(repo_root, "cat-file", "-s", object_id.decode()).strip())
    except ValueError as error:
        raise _fail("invalid historical Git blob size", path=relative) from error
    if size > _MAX_ARTIFACT_BYTES:
        raise _fail("historical Git blob exceeds byte limit", path=relative)
    raw = _run_git(
        repo_root,
        "cat-file",
        "blob",
        object_id.decode(),
        max_bytes=_MAX_ARTIFACT_BYTES,
    )
    if len(raw) != size:
        raise _fail("historical Git blob size changed", path=relative)
    return raw


def _load_manifest(repo_root: Path) -> tuple[dict[str, object], str]:
    raw = _regular_bytes(repo_root, _MANIFEST, max_bytes=_MAX_MANIFEST_BYTES)
    try:
        values = json.loads(raw)
        canonical = json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise _fail("historical delivery manifest is invalid") from error
    if _sha256(canonical) != _MANIFEST_CANONICAL_SHA256:
        raise _fail("historical delivery manifest differs from fixed pins")
    if not isinstance(values, dict) or values.get("checkout_commit") != _CHECKOUT:
        raise _fail("historical delivery manifest differs from fixed pins")
    return values, _sha256(raw)


def _completion_row(repo_root: Path, phase: int) -> bytes:
    plan = _regular_bytes(repo_root, "docs/PLAN.md", max_bytes=_MAX_MANIFEST_BYTES)
    prefix = f"| {phase} —".encode()
    rows = [line for line in plan.splitlines() if line.startswith(prefix)]
    if len(rows) != 1:
        raise _fail("historical completion row is missing or ambiguous", phase=phase)
    return rows[0]


def _preflight(repo_root: Path, phase: int) -> tuple[dict[str, object], tuple[object, ...]]:
    if type(phase) is not int or phase not in (1, 2, 3):
        raise ValueError("historical phase must be exactly 1, 2, or 3")
    if os.environ.get("PYTHONPATH"):
        raise _fail("PYTHONPATH injection is forbidden for historical verification")
    manifest, manifest_file_hash = _load_manifest(repo_root)
    phases = manifest.get("phases")
    if not isinstance(phases, dict) or set(phases) != {"1", "2", "3"}:
        raise _fail("historical delivery manifest differs from fixed pins")
    selected = phases.get(str(phase))
    if not isinstance(selected, dict):
        raise _fail("historical delivery manifest differs from fixed pins")
    artifact_hashes = selected.get("artifact_hashes")
    verifier = selected.get("verifier")
    if not isinstance(artifact_hashes, dict) or not isinstance(verifier, dict):
        raise _fail("historical delivery manifest differs from fixed pins")
    pinned_paths = {**artifact_hashes, verifier.get("path"): verifier.get("sha256")}
    if not all(
        isinstance(path, str) and isinstance(digest, str) for path, digest in pinned_paths.items()
    ):
        raise _fail("historical delivery manifest differs from fixed pins")
    live_hashes: dict[str, str] = {}
    for relative, expected_hash in artifact_hashes.items():
        raw = _regular_bytes(repo_root, relative, max_bytes=_MAX_ARTIFACT_BYTES)
        actual_hash = _sha256(raw)
        if actual_hash != expected_hash:
            raise _fail("current historical artifact differs from fixed pin", path=relative)
        live_hashes[relative] = actual_hash
    completion_hash = _sha256(_completion_row(repo_root, phase))
    if completion_hash != selected.get("completion_row_sha256"):
        raise _fail("current historical completion row differs from fixed pin", phase=phase)
    resolved = (
        _run_git(repo_root, "rev-parse", "--verify", f"{_CHECKOUT}^{{commit}}").decode().strip()
    )
    if resolved != _CHECKOUT:
        raise _fail("historical checkout revision substitution")
    for relative, expected_hash in pinned_paths.items():
        raw = _regular_git_blob(repo_root, _CHECKOUT, relative)
        if _sha256(raw) != expected_hash:
            raise _fail("historical Git blob differs from fixed pin", path=relative)
    receipt_key: tuple[object, ...] = (
        phase,
        _CHECKOUT,
        manifest_file_hash,
        tuple(sorted(live_hashes.items())),
        completion_hash,
        verifier["sha256"],
        tuple(sorted(selected["source_hashes"].items())),
    )
    return selected, receipt_key


def _request(phase: int, clone: Path, selected: dict[str, object]) -> dict[str, object]:
    artifact_hashes = selected["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    verifier = selected["verifier"]
    assert isinstance(verifier, dict)
    if phase == 1:
        kwargs = {
            "validation_path": "manifests/validation/v1/ofd-primary-10000.json",
            "oracle_path": "manifests/validation/v1/phase1-oracle-gate.json",
            "leakage_path": "manifests/validation/v1/phase1-leakage-gate.json",
            "validation_reproducibility_path": (
                "manifests/validation/v1/phase1-validation-reproducibility.json"
            ),
            "independent_reproducibility_path": (
                "manifests/validation/v1/phase1-independent-reproducibility-gate.json"
            ),
        }
        function = "verify_phase1_gate_artifacts"
    elif phase == 2:
        kwargs = {
            "artifact_path": "manifests/validation/v1/phase2-engine-gate.json",
            "repo_root": ".",
            "expected_source_commit": selected["source_commit"],
        }
        function = "verify_phase2_gate_artifact"
    else:
        kwargs = {
            "artifact_path": (
                "manifests/validation/phase3/component-gate-content-v2-authenticated.json"
            ),
            "manifest_path": (
                "manifests/validation/phase3/one-hop-10000-content-v2-authenticated.json"
            ),
            "repo_root": ".",
            "expected_source_commit": selected["source_commit"],
        }
        function = "verify_phase3_gate_artifact"
    return {
        "root": str(clone),
        "verifier_path": verifier["path"],
        "function": function,
        "kwargs": kwargs,
    }


def _run_pinned_verifier(
    repo_root: Path, phase: int, selected: dict[str, object]
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="silent-cascade-historical-") as directory:
        clone = Path(directory) / "checkout"
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        environment.update(
            {
                "ALL_PROXY": "",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_TERMINAL_PROMPT": "0",
                "HF_HUB_OFFLINE": "1",
                "HTTP_PROXY": "",
                "HTTPS_PROXY": "",
                "NO_PROXY": "*",
                "PYTHONNOUSERSITE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "WANDB_MODE": "offline",
            }
        )
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--no-local",
                    "--no-hardlinks",
                    "--no-checkout",
                    "--no-tags",
                    str(repo_root),
                    str(clone),
                ],
                env=environment,
                check=True,
                capture_output=True,
                timeout=300,
            )
            subprocess.run(
                ["git", "checkout", "--detach", _CHECKOUT],
                cwd=clone,
                env=environment,
                check=True,
                capture_output=True,
                timeout=120,
            )
            subprocess.run(
                ["git", "remote", "remove", "origin"],
                cwd=clone,
                env=environment,
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise _fail("failed to create the pinned disposable checkout") from error
        head = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=clone,
                env=environment,
                check=True,
                capture_output=True,
                timeout=30,
            )
            .stdout.decode()
            .strip()
        )
        if head != _CHECKOUT:
            raise _fail("disposable checkout revision substitution")
        request = _request(phase, clone, selected)
        environment["SILENT_CASCADE_HISTORICAL_REQUEST"] = json.dumps(
            request, sort_keys=True, separators=(",", ":")
        )
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _CHILD],
                cwd=clone,
                env=environment,
                check=False,
                capture_output=True,
                timeout=_CHILD_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise _fail("pinned historical verifier execution failed") from error
        if (
            len(completed.stdout) > _MAX_CHILD_OUTPUT_BYTES
            or len(completed.stderr) > _MAX_CHILD_OUTPUT_BYTES
        ):
            raise _fail("pinned historical verifier output exceeded its byte limit")
        if completed.stderr:
            raise _fail("pinned historical verifier emitted unexpected stderr")
        try:
            payload = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _fail("pinned historical verifier returned invalid JSON") from error
        if completed.returncode != 0 or payload.get("ok") is not True:
            raise _fail(
                "pinned historical verifier rejected the delivery",
                detail=str(payload.get("message", "verification failed"))[:2000],
            )
        result = payload.get("result")
        origins = payload.get("origins")
        if not isinstance(result, dict) or not isinstance(origins, dict) or not origins:
            raise _fail("pinned historical verifier omitted authenticated output")
        if result.get("passed") is not True or result.get("foundation_model_calls") != 0:
            raise _fail("pinned historical verifier did not return accepted zero-call evidence")
        if result.get("source_commit") not in (None, selected["source_commit"]):
            raise _fail("pinned historical verifier returned the wrong source revision")
        return result


def verify_historical_delivery(*, repo_root: Path, phase: int) -> dict[str, object]:
    """Run one fixed legacy verifier in the fixed historical checkout."""

    root = repo_root.resolve(strict=True)
    selected, receipt_key = _preflight(root, phase)
    cached = _RECEIPTS.get(receipt_key)
    if cached is not None:
        return json.loads(cached)
    original_result = _run_pinned_verifier(root, phase, selected)
    verifier = selected["verifier"]
    assert isinstance(verifier, dict)
    result: dict[str, object] = {
        "schema_version": "historical-delivery-verification-v1",
        "verification_scope": "historical",
        "phase": phase,
        "checkout_commit": _CHECKOUT,
        "source_commit": selected["source_commit"],
        "source_hashes": selected["source_hashes"],
        "artifact_hashes": selected["artifact_hashes"],
        "verifier": verifier,
        "original_result": original_result,
        "original_result_sha256": _sha256(canonical_json_bytes(original_result)),
        "foundation_model_calls": 0,
    }
    receipt = canonical_json_bytes(result)
    _RECEIPTS[receipt_key] = receipt
    return json.loads(receipt)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, type=int, choices=(1, 2, 3))
    arguments = parser.parse_args()
    try:
        result = verify_historical_delivery(repo_root=Path.cwd(), phase=arguments.phase)
    except (OSError, SilentCascadeError, ValueError) as error:
        payload = (
            error.to_payload()
            if isinstance(error, SilentCascadeError)
            else {"code": "historical_delivery_error", "message": str(error), "context": {}}
        )
        sys.stderr.buffer.write(canonical_json_bytes(payload) + b"\n")
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
