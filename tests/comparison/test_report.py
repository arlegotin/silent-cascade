"""Comparison reports must be derived from authenticated saved rows only."""

import json

import pytest
import torch

from silent_cascade.eval.comparison_data import (
    build_manifest,
    diagnostic_allocations,
    publish_manifest,
)
from silent_cascade.eval.comparison_runner import run_comparison
from silent_cascade.eval.comparison_types import (
    BudgetLedger,
    ComparisonConfig,
    ComparisonIdentity,
    ConditionResult,
)
from silent_cascade.eval.metrics import EvaluationError
from silent_cascade.eventflow.neural import NeuralModelIdentity
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.report.comparison import build_comparison_report


@pytest.fixture
def small_run(tmp_path, monkeypatch):
    from silent_cascade.eval import comparison_runner

    config = ComparisonConfig()
    torch.manual_seed(11)
    model = EventFlowModel(config.phase4_config.neural).eval()
    calls = 0

    def controlled(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = ConditionResult(
            stop_reason="dynamics_error" if calls == 1 else "fixture",
            error=EvaluationError(code="dynamics_error") if calls == 1 else None,
        )
        if "trace_sink" in kwargs:
            kwargs["trace_sink"](result, None)
        return result

    monkeypatch.setattr(comparison_runner, "run_intact_episode", controlled)
    monkeypatch.setattr(comparison_runner, "_run_compressed_episode", controlled)
    for name in ("iid", "depth"):
        allocation = diagnostic_allocations()[name]
        first = allocation.blocks[0].model_copy(update={"episode_count": 4})
        manifest = build_manifest(config, name, allocation.model_copy(update={"blocks": (first,)}))
        publish_manifest(manifest, tmp_path / "manifests" / f"{name}.json")
        for condition in ("intact_eventflow", "compressed_eventflow"):
            identity = ComparisonIdentity(
                condition=condition,
                manifest_name=name,
                protocol_sha256=config.protocol_sha256,
                config_sha256=config.config_sha256,
                generator_sha256=config.generator_sha256,
                manifest_sha256=sha256_bytes(canonical_json_bytes(manifest)),
                checkpoint_sha256="1" * 64,
                model_state_sha256=NeuralModelIdentity.from_model(
                    model, source_revision="1" * 40
                ).model_state_sha256,
                producing_source_revision="1" * 40,
                execution_source_revision="2" * 40,
            )
            run_comparison(
                config=config,
                manifest=manifest,
                identity=identity,
                model=model,
                output_dir=tmp_path / "conditions" / condition / name,
                budget=BudgetLedger(),
            )
    return tmp_path


def test_report_is_artifact_only_and_never_certifies_phase5(small_run, tmp_path) -> None:
    output = build_comparison_report(small_run, tmp_path / "report")
    report = json.loads((output / "report.json").read_text())
    assert report["purpose"] == "exploratory_comparison"
    assert not report["complete_a"]
    assert report["decision"]["status"] == "deferred_after_a"
    assert report["corpora"]["iid"]["intact_eventflow"]["episodes"] == 4
    assert report["corpora"]["iid"]["intact_eventflow"]["errors"] == 1
    assert report["corpora"]["iid"]["intact_eventflow"]["timed_successes"] <= 4
    assert "Phase 5 gate passed" not in (output / "report.md").read_text()
    assert "Single-seed exploratory" in (output / "report.md").read_text()
    assert any((small_run / "conditions" / "intact_eventflow" / "iid" / "traces").iterdir())


def test_report_rejects_missing_pair_and_corrupt_rows(small_run, tmp_path) -> None:
    missing = small_run / "conditions" / "compressed_eventflow" / "depth" / "inventory.json"
    missing.unlink()
    with pytest.raises((ValueError, FileNotFoundError), match=r"missing|inventory"):
        build_comparison_report(small_run, tmp_path / "missing-report")
    missing.write_bytes(b"{}")
    rows = small_run / "conditions" / "intact_eventflow" / "iid" / "rows.jsonl.gz"
    rows.write_bytes(rows.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match=r"hash|inventory|corrupt"):
        build_comparison_report(small_run, tmp_path / "corrupt-report")
