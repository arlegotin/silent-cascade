"""Real debug collection never becomes production acceptance."""

import json


def test_gate_records_every_real_failure_and_measured_numeric_evidence(debug_component_gate):
    from silent_cascade.train.evidence_types import ComponentGateReport

    gate = debug_component_gate
    report = ComponentGateReport.model_validate_json((gate / "component-gate.json").read_bytes())
    assert report.publication == "debug" and not report.passed
    assert len(report.rows) == report.summary.episode_count == 16
    assert report.summary.variant_counts == {"positive": 8, "safe": 4, "disconnected": 4}
    assert (
        sum(row.required_recall_count for row in report.rows)
        == report.summary.required_recall_count
    )
    assert report.summary.complete_chain_correct < 16  # Real four-step training is not overfit.
    assert report.numeric.cpu_resume.resumed_step == 2
    assert report.numeric.mps_resume.device == "mps"
    assert report.numeric.parity.empty_memory_rows > 0
    assert len({sample.position for sample in report.replay_samples}) == 3
    assert report.training.selected.optimizer_step > 0
    assert report.training.weights.optimizer_step == 0
    assert report.offline.training_steps == 1
    assert report.offline.predicted_rows == 8
    assert report.offline.blocked_import_attempts == report.offline.network_attempts == 0
    assert (
        report.objective_version
        == report.offline.objective_version
        == "teacher_timed_plus_content_v2"
    )
    assert report.training.step_counters == (0, 1, 2, 3)
    assert [sample.step for sample in report.training.diagnostic_samples] == [1, 2]


def test_collect_rejects_checkpoint_identity_before_publication(debug_component_gate):
    from .test_phase3_provenance import execute

    execute(
        debug_component_gate,
        """
import json
from pathlib import Path
from silent_cascade.train.config import parse_phase3_canonical
from silent_cascade.config import ResolvedConfig
from silent_cascade.train.evidence import collect_component_gate
from silent_cascade.errors import SilentCascadeError
root = Path.cwd()
raw = json.loads((root/'component-gate.json').read_bytes())
p = raw['provenance']
from silent_cascade.config import resolve_config
config = resolve_config(type(parse_phase3_canonical(p['config_canonical_json'])),
 tuple(Path(v) for v in p['config_paths']))
try:
    collect_component_gate(config, manifest_path=root/'component-manifest.json',
      weights_path=root/raw['training']['run_path']/raw['training']['weights']['relative_path'],
      expected_checkpoint_sha256='0'*64, training_run=root/raw['training']['run_path'],
      source_commit=p['source_commit'], plan_revision=p['plan_revision'],
      output_path=root/'bad.json')
except SilentCascadeError:
    pass
else:
    raise AssertionError('wrong checkpoint accepted')
assert not (root/'bad.json').exists()
""",
    )


def test_full_row_serialization_has_capacity_for_production_bound(debug_component_gate):
    raw = json.loads((debug_component_gate / "component-gate.json").read_bytes())
    # Size stress only; no synthetic production report is passed to the collector/verifier.
    largest = dict(max(raw["rows"], key=lambda row: len(json.dumps(row))))
    largest["prediction"] = dict(largest["prediction"])
    # A trained row can contain two full decisions even when smoke predictions halt early.
    target = next(row["target"] for row in raw["rows"] if len(row["target"]["content"]) == 2)
    largest["prediction"]["record_ids"] = target["record_ids"]
    largest["prediction"]["content"] = [
        {
            **content,
            "focus": 63,
            "hazard_type": 3,
            "status": 2,
            "log_delay": -0.12345678901234567,
            "normalized_deadline": 0.12345678901234567,
            "confidence": 0.12345678901234567,
            "continue_search": True,
        }
        for content in target["content"]
    ]
    bound = len(json.dumps(largest, separators=(",", ":")).encode()) * 10_000
    metadata = len(json.dumps({k: v for k, v in raw.items() if k != "rows"}).encode())
    assert bound + metadata + 2_000_000 < 32 * 1024 * 1024


def test_explicit_external_run_collects_without_absolute_locator_in_report(debug_component_gate):
    from .test_phase3_provenance import execute

    execute(
        debug_component_gate,
        """
import json
from pathlib import Path
from silent_cascade.config import resolve_config
from silent_cascade.train.config import Phase3Config
from silent_cascade.train.trainer import run_training
from silent_cascade.train.evidence import collect_component_gate
root = Path.cwd()
old = json.loads((root/'component-gate.json').read_bytes())
p = old['provenance']
config = resolve_config(Phase3Config, tuple(Path(v) for v in p['config_paths']))
external = root.parent/'owned-external-experiment'
trained = run_training(config, validation_manifest=root/'component-manifest.json',
    run_dir=external, source_commit=p['source_commit'], device='cpu')
result = collect_component_gate(config, manifest_path=root/'component-manifest.json',
    weights_path=external/trained.weights.relative_path,
    expected_checkpoint_sha256=trained.weights.file_sha256, training_run=external,
    source_commit=p['source_commit'], plan_revision=p['plan_revision'],
    output_path=root/'external-gate.json')
assert str(external) not in result.model_dump_json()
assert result.training.weights.file_sha256 == trained.weights.file_sha256
assert result.training.selected.optimizer_step > 0
""",
    )
