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


def verify_in_checkout(gate, artifact=None, *, allow_debug=True, reject=False, prelude=""):
    """Execute real verification using the package in the authenticated fixture checkout."""
    from .test_phase3_provenance import execute

    output = execute(
        gate,
        f"""
import json, runpy
from pathlib import Path
from silent_cascade.errors import ArtifactIntegrityError
{prelude}
verify = runpy.run_path('scripts/verify_phase3_gate_artifact.py')['verify_phase3_gate_artifact']
try:
    result = verify(artifact_path=Path({str(artifact or gate / "component-gate.json")!r}),
        manifest_path=Path.cwd()/'component-manifest.json', repo_root=Path.cwd(),
        allow_debug={allow_debug!r})
except ArtifactIntegrityError:
    assert {reject!r}, 'valid real artifact rejected'
    print('{{}}')
else:
    assert not {reject!r}, 'tampered artifact accepted'
    print(result.model_dump_json())
""",
    )
    return json.loads(output)


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
    gate = debug_component_gate
    result = verify_in_checkout(gate)
    assert result["valid"] and not result["passed"] and result["episode_count"] == 16
    verify_in_checkout(gate, allow_debug=False, reject=True)


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
        "compute_records",
        "compute_forward",
        "compute_flow",
        "compute_jumps",
        "compute_transitions",
        "compute_opportunities",
        "compute_eligibility",
        "compute_bilinear",
        "compute_memory",
        "compute_module_calls",
        "objective",
        "coefficient",
        "optimizer",
        "missing_timed_numeric",
        "missing_content_numeric",
        "missing_step",
        "step_loss",
        "step_counter",
        "counter_zero",
        "offline_objective",
        "raw_row_total",
        "auxiliary_denominator",
        "numeric_counter",
        "missing_forward_context",
    ],
)
def test_independent_verifier_rejects_raw_tampering(debug_component_gate, tmp_path, kind):
    gate = debug_component_gate
    raw = json.loads((gate / "component-gate.json").read_bytes())
    if kind == "raw_row_total":
        raw["training"]["diagnostic_samples"][0]["result"]["per_position"][
            "content/retrieval"
        ].pop()
    elif kind == "auxiliary_denominator":
        result = raw["training"]["diagnostic_samples"][0]["result"]
        result["denominators"]["content/retrieval"] *= 2
        result["numerators"]["content/retrieval"] *= 2
    elif kind == "numeric_counter":
        raw["numeric"]["parity"]["forward"]["compared"] -= 1
    elif kind == "missing_forward_context":
        data = raw["numeric"]["parity"]["forward"]
        key = "/content/final_context/workspace/latent"
        data["tested_names"].remove(key)
        data["compared"] -= data["tensor_elements"].pop(key)
    elif kind == "objective":
        raw["objective_version"] = "teacher_timed_v1"
    elif kind == "coefficient":
        raw["auxiliary_coefficient"] = 0.0
    elif kind == "optimizer":
        raw["numeric"]["parity"]["optimizer_options"]["eps"] = 1e-7
    elif kind in ("missing_timed_numeric", "missing_content_numeric"):
        prefix = "/timed/" if kind == "missing_timed_numeric" else "/content/"
        data = raw["numeric"]["parity"]["forward"]
        data["tested_names"] = [
            name for name in data["tested_names"] if not name.startswith(prefix)
        ]
    elif kind == "missing_step":
        raw["training"]["diagnostic_samples"].pop()
    elif kind == "step_loss":
        raw["training"]["diagnostic_samples"][0]["result"]["terms"].pop("content/retrieval")
    elif kind == "step_counter":
        raw["training"]["step_counters"].pop()
    elif kind == "counter_zero":
        raw["numeric"]["cpu_resume"]["first_example_hashes"] = []
    elif kind == "offline_objective":
        raw["offline"]["objective_version"] = "teacher_timed_v1"
    elif kind == "count":
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
    elif kind.startswith("compute_"):
        field = {
            "compute_records": "records_scored",
            "compute_forward": "forward_macs",
            "compute_flow": "flow_evaluations",
            "compute_jumps": "jump_applications",
            "compute_transitions": "row_transitions",
            "compute_opportunities": "opportunities",
            "compute_eligibility": "eligible_records",
            "compute_memory": "memory_bytes",
        }.get(kind)
        if field:
            raw["compute"][0][field] += 1
        elif kind == "compute_bilinear":
            raw["compute"][0]["operation_estimates"]["bilinear_dot_macs"] += 1
        else:
            raw["compute"][0]["module_calls"]["RetrievalScorer"] += 1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(raw))
    verify_in_checkout(gate, path, reject=True)


def test_independent_verifier_never_calls_producer_arithmetic(debug_component_gate):
    assert verify_in_checkout(
        debug_component_gate,
        prelude="""
from silent_cascade.train import component_eval, evidence
def forbidden(*args, **kwargs):
    raise AssertionError('producer arithmetic invoked by verifier')
for name in ('score_components', 'evaluate_components'):
    setattr(component_eval, name, forbidden)
for name in ('score_row', 'summarize_rows', 'row_chain', 'collect_component_gate'):
    setattr(evidence, name, forbidden)
""",
    )["valid"]


@pytest.mark.parametrize("stale", [False, True])
def test_verifier_rejects_foreign_executing_package_before_reconstruction(
    debug_component_gate, tmp_path, stale
):
    import shutil

    from .test_phase3_provenance import execute

    foreign = tmp_path / "foreign-src"
    shutil.copytree(
        debug_component_gate / "src", foreign, ignore=shutil.ignore_patterns("__pycache__")
    )
    if stale:
        path = foreign / "silent_cascade/train/curriculum_data.py"
        path.write_text(path.read_text() + "\n# unrelated stale package revision\n")
    execute(
        debug_component_gate,
        f"""
import sys
sys.path.insert(0, {str(foreign)!r})
import runpy
from pathlib import Path
from silent_cascade.errors import ArtifactIntegrityError
from silent_cascade.train import curriculum_data
assert str(curriculum_data.__file__).startswith({str(foreign)!r})
def forbidden(*args, **kwargs):
    raise AssertionError('foreign reconstruction executed before identity rejection')
curriculum_data.make_curriculum_example = forbidden
verify = runpy.run_path('scripts/verify_phase3_gate_artifact.py')['verify_phase3_gate_artifact']
try:
    verify(artifact_path=Path.cwd()/'component-gate.json',
           manifest_path=Path.cwd()/'component-manifest.json', repo_root=Path.cwd(),
           allow_debug=True)
except ArtifactIntegrityError as error:
    assert 'executing' in str(error), str(error)
else:
    raise AssertionError('foreign executing package authenticated')
""",
    )


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
