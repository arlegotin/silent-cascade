"""Independent arithmetic, trust history and hostile primitive regressions."""

import json
from pathlib import Path
from runpy import run_path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def verifier():
    return run_path(str(ROOT / "scripts/verify_phase3_gate_artifact.py"))[
        "verify_phase3_gate_artifact"
    ]


@pytest.mark.parametrize(
    "correct,required,expected",
    [
        (99, 100, False),
        (9900, 10000, False),
        (9901, 10000, True),
        (0, 0, False),
        (True, 1, False),
        (100, 100, True),
    ],
)
def test_strict_threshold_uses_integer_arithmetic(correct, required, expected):
    namespace = run_path(str(ROOT / "scripts/verify_phase3_gate_artifact.py"))
    assert namespace["_above_99_percent"](correct, required) is expected


def test_independent_verifier_accepts_real_debug_and_refuses_acceptance(debug_component_gate):
    from silent_cascade.errors import ArtifactIntegrityError

    gate = debug_component_gate
    result = verifier()(
        artifact_path=gate / "component-gate.json",
        manifest_path=gate / "component-manifest.json",
        repo_root=gate,
        allow_debug=True,
    )
    assert result.valid and not result.passed and result.episode_count == 16
    with pytest.raises(ArtifactIntegrityError):
        verifier()(
            artifact_path=gate / "component-gate.json",
            manifest_path=gate / "component-manifest.json",
            repo_root=gate,
        )


@pytest.mark.parametrize(
    "kind",
    [
        "count",
        "bool",
        "ratio",
        "threshold",
        "missing",
        "extra",
        "target",
        "prediction",
        "variant",
        "chain",
        "source",
        "config",
        "numeric",
        "foundation",
        "parity_batch",
        "resume_batch",
        "history_denominator",
        "history_stop",
    ],
)
def test_independent_verifier_rejects_raw_tampering(debug_component_gate, tmp_path, kind):
    from silent_cascade.errors import ArtifactIntegrityError

    gate = debug_component_gate
    raw = json.loads((gate / "component-gate.json").read_bytes())
    if kind == "count":
        raw["summary"]["complete_chain_correct"] += 1
    elif kind == "bool":
        raw["summary"]["complete_chain_correct"] = False
    elif kind == "ratio":
        raw["summary"]["complete_chain_accuracy"] = 1.0
    elif kind == "threshold":
        raw["threshold_percent"] = 98
    elif kind == "missing":
        raw["rows"].pop()
    elif kind == "extra":
        raw["rows"].append(raw["rows"][0])
    elif kind == "target":
        raw["rows"][0]["target"]["record_ids"][0] += 1
    elif kind == "prediction":
        raw["rows"][0]["prediction"]["record_ids"] = [999]
    elif kind == "variant":
        raw["rows"][0]["variant"] = "omitted"
    elif kind == "chain":
        raw["rows_sha256"] = "0" * 64
    elif kind == "source":
        raw["provenance"]["source_commit"] = "0" * 40
    elif kind == "config":
        raw["provenance"]["config_sha256"] = "0" * 64
    elif kind == "numeric":
        raw["numeric"]["parity"]["gradients"]["mismatches"] = 1
    elif kind == "foundation":
        raw["rows"][0]["foundation_model_calls"] = 1
    elif kind == "parity_batch":
        raw["numeric"]["parity"]["example_hashes"][0] = "0" * 64
    elif kind == "resume_batch":
        raw["numeric"]["cpu_resume"]["example_hashes"][0] = "0" * 64
    elif kind == "history_denominator":
        for point in raw["training"]["validation_history"]:
            point["required_recall_count"] *= 2
            point["correct_recall_count"] *= 2
    elif kind == "history_stop":
        raw["training"]["stop_reason"] = "component_gate"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ArtifactIntegrityError):
        verifier()(
            artifact_path=path,
            manifest_path=gate / "component-manifest.json",
            repo_root=gate,
            allow_debug=True,
        )


def test_independent_verifier_never_calls_producer_arithmetic(debug_component_gate, monkeypatch):
    from silent_cascade.train import component_eval, evidence

    def forbidden(*args, **kwargs):
        raise AssertionError("producer arithmetic invoked by verifier")

    for name in ("score_components", "evaluate_components"):
        monkeypatch.setattr(component_eval, name, forbidden)
    for name in ("score_row", "summarize_rows", "row_chain", "collect_component_gate"):
        monkeypatch.setattr(evidence, name, forbidden)
    gate = debug_component_gate
    assert verifier()(
        artifact_path=gate / "component-gate.json",
        manifest_path=gate / "component-manifest.json",
        repo_root=gate,
        allow_debug=True,
    ).valid


def test_optional_weight_check_accepts_real_typed_buffers_and_rejects_coherent_payload_change(
    tmp_path,
):
    import hashlib

    from silent_cascade.config import resolve_config
    from silent_cascade.errors import ArtifactIntegrityError
    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.train.checkpoints import export_weights
    from silent_cascade.train.config import Phase3Config

    from .test_training_cli import CONFIGS

    config = resolve_config(Phase3Config, CONFIGS)
    descriptor = export_weights(
        tmp_path, EventFlowModel(config.config.neural), config=config, source_commit="a" * 40
    )
    path = tmp_path / descriptor.relative_path
    raw = {
        "training": {"weights": descriptor.model_dump(mode="json")},
        "provenance": {"config_canonical_json": config.canonical_json.decode()},
    }
    check = run_path(str(ROOT / "scripts/verify_phase3_gate_artifact.py"))["_weights"]
    check(path, raw)
    damaged = bytearray(path.read_bytes())
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    raw["training"]["weights"]["file_sha256"] = hashlib.sha256(damaged).hexdigest()
    with pytest.raises(ArtifactIntegrityError):
        check(path, raw)


def test_verifier_byte_check_imports_no_producer_or_learned_computation(debug_component_gate):
    from .test_phase3_provenance import execute

    execute(
        debug_component_gate,
        """
import importlib.abc, json, runpy, sys
from pathlib import Path
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'silent_cascade.train.evidence', 'silent_cascade.train.component_eval',
                        'silent_cascade.train.verification', 'silent_cascade.models.event_flow',
                        'silent_cascade.models.losses', 'silent_cascade.train.trainer'}:
            raise AssertionError('independent verifier imported producer/model execution')
sys.meta_path.insert(0, Blocker())
root = Path.cwd()
report = json.loads((root/'component-gate.json').read_bytes())
weights = root/report['training']['run_path']/report['training']['weights']['relative_path']
verify = runpy.run_path('scripts/verify_phase3_gate_artifact.py')['verify_phase3_gate_artifact']
result = verify(artifact_path=root/'component-gate.json',
    manifest_path=root/'component-manifest.json', repo_root=root,
    weights_path=weights, allow_debug=True)
assert result.valid and not result.passed
""",
    )
