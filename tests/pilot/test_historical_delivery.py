import hashlib
import json
import os
from pathlib import Path
from runpy import run_path

import pytest

from silent_cascade.errors import ArtifactIntegrityError

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/verify_historical_delivery.py"
MANIFEST = ROOT / "manifests/validation/historical-delivery.json"


def _module():
    return run_path(str(SCRIPT))


def _write_manifest(root: Path, payload: dict) -> None:
    path = root / "manifests/validation/historical-delivery.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")


@pytest.mark.parametrize("phase", [1, 2, 3])
def test_historical_delivery_runs_the_pinned_original_verifier(phase):
    result = _module()["verify_historical_delivery"](repo_root=ROOT, phase=phase)
    expected = json.loads(MANIFEST.read_bytes())["phases"][str(phase)]
    assert result["schema_version"] == "historical-delivery-verification-v1"
    assert result["verification_scope"] == "historical"
    assert result["phase"] == phase
    assert result["checkout_commit"] == "ea713cdce4107e709985d9333050a947db6561ab"
    assert result["source_commit"] == expected["source_commit"]
    assert result["source_hashes"] == expected["source_hashes"]
    assert result["artifact_hashes"] == expected["artifact_hashes"]
    assert result["verifier"] == expected["verifier"]
    assert result["original_result"]["passed"] is True
    assert result["foundation_model_calls"] == 0


def _sentinel_module(monkeypatch):
    namespace = _module()

    def verifier_must_not_run(*_args, **_kwargs):
        raise AssertionError("pinned verifier ran before historical preflight completed")

    monkeypatch.setitem(namespace, "_run_pinned_verifier", verifier_must_not_run)
    return namespace


def test_coordinated_artifact_and_claimed_hash_substitution_fails_before_verifier(
    tmp_path: Path, monkeypatch
):
    payload = json.loads(MANIFEST.read_bytes())
    relative = "manifests/validation/v1/phase2-engine-gate.json"
    replacement = b'{"coordinated":"replacement"}'
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(replacement)
    payload["phases"]["2"]["artifact_hashes"][relative] = hashlib.sha256(replacement).hexdigest()
    _write_manifest(tmp_path, payload)

    with pytest.raises(ArtifactIntegrityError, match="historical delivery manifest"):
        _sentinel_module(monkeypatch)["verify_historical_delivery"](repo_root=tmp_path, phase=2)


def test_checkout_revision_substitution_fails_before_verifier(tmp_path: Path, monkeypatch):
    payload = json.loads(MANIFEST.read_bytes())
    payload["checkout_commit"] = "0" * 40
    _write_manifest(tmp_path, payload)

    with pytest.raises(ArtifactIntegrityError, match="historical delivery manifest"):
        _sentinel_module(monkeypatch)["verify_historical_delivery"](repo_root=tmp_path, phase=1)


def test_symlinked_artifact_parent_fails_before_verifier(tmp_path: Path, monkeypatch):
    payload = json.loads(MANIFEST.read_bytes())
    _write_manifest(tmp_path, payload)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "manifests/validation/v1").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        _sentinel_module(monkeypatch)["verify_historical_delivery"](repo_root=tmp_path, phase=1)


def test_current_package_search_path_injection_fails_before_verifier(monkeypatch):
    namespace = _sentinel_module(monkeypatch)
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src") + os.pathsep + "/tmp/untrusted")

    with pytest.raises(ArtifactIntegrityError, match="PYTHONPATH"):
        namespace["verify_historical_delivery"](repo_root=ROOT, phase=3)


def test_manifest_pins_are_exact_regular_blobs_from_the_fixed_checkout():
    payload = json.loads(MANIFEST.read_bytes())
    assert payload["checkout_commit"] == "ea713cdce4107e709985d9333050a947db6561ab"
    assert set(payload["phases"]) == {"1", "2", "3"}
    for phase in payload["phases"].values():
        for relative, expected_hash in {
            **phase["artifact_hashes"],
            phase["verifier"]["path"]: phase["verifier"]["sha256"],
        }.items():
            entry = os.fsdecode(
                __import__("subprocess")
                .run(
                    ["git", "ls-tree", payload["checkout_commit"], "--", relative],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                )
                .stdout
            )
            assert entry.startswith("100644 blob ") or entry.startswith("100755 blob ")
            raw = (
                __import__("subprocess")
                .run(
                    ["git", "show", f"{payload['checkout_commit']}:{relative}"],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                )
                .stdout
            )
            assert hashlib.sha256(raw).hexdigest() == expected_hash
