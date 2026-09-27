"""Milestone A admission is fixed before reading diagnostic scores."""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from silent_cascade.env.episode import EpisodeVariant
from silent_cascade.env.reward import score_actions
from silent_cascade.eval.comparison_data import build_manifest, diagnostic_allocations, iter_bundles
from silent_cascade.eval.comparison_protocol import (
    decide_after_a,
    legacy_compatibility_indices,
    timing_indices,
)
from silent_cascade.eval.comparison_runner import run_comparison
from silent_cascade.eval.comparison_types import (
    BudgetLedger,
    ComparisonConfig,
    ComparisonIdentity,
    ComparisonRow,
    ConditionResult,
)
from silent_cascade.eventflow.neural import NeuralModelIdentity
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.schemas import Action


def _rows(iid_success: int = 461, depth_success: int = 384):
    config = ComparisonConfig()
    allocation = diagnostic_allocations()["iid"]
    first = allocation.blocks[0].model_copy(update={"episode_count": 4})
    negative = next(
        bundle
        for bundle in iter_bundles(
            build_manifest(config, "iid", allocation.model_copy(update={"blocks": (first,)})),
            config,
        )
        if bundle.truth.recipe.variant is EpisodeVariant.SAFE_NEGATIVE
    )
    truth = negative.truth
    rows = []
    for name, successes in (("iid", iid_success), ("depth", depth_success)):
        for condition in ("intact_eventflow", "compressed_eventflow"):
            for index in range(512):
                actions = (
                    ()
                    if condition == "compressed_eventflow" or index < successes
                    else (Action(0, truth.activation_time, 99),)
                )
                result = ConditionResult(actions=actions, stop_reason="fixture")
                rows.append(
                    ComparisonRow(
                        public_id=f"{name}-{index}",
                        condition=condition,
                        manifest_name=name,
                        protocol_sha256=config.protocol_sha256,
                        episode_sha256=f"{index:064x}",
                        identity_sha256="1" * 64 if condition == "intact_eventflow" else "2" * 64,
                        variant=truth.recipe.variant,
                        path_length=truth.recipe.requested_path_length,
                        truth=truth,
                        result=result,
                        score=score_actions(truth, actions),
                        timed_success=not actions,
                    )
                )
    return config, rows


def test_a_admission_is_predeclared() -> None:
    config, rows = _rows()
    decision = decide_after_a(rows, protocol_sha256=config.protocol_sha256)
    assert decision.proceed_to_b
    assert decision.iid_intact_successes == 461
    assert decision.depth_intact_successes == 384
    assert decision.iid_denominator == decision.depth_denominator == 512
    assert not decide_after_a(
        _rows(460, 384)[1], protocol_sha256=config.protocol_sha256
    ).proceed_to_b
    assert not decide_after_a(
        _rows(461, 383)[1], protocol_sha256=config.protocol_sha256
    ).proceed_to_b
    assert not decide_after_a(rows[:-1], protocol_sha256=config.protocol_sha256).proceed_to_b
    assert not decide_after_a(
        [rows[0].model_copy(update={"protocol_sha256": "0" * 64}), *rows[1:]],
        protocol_sha256=config.protocol_sha256,
    ).proceed_to_b
    compute = rows[0].result.end_to_end_compute.model_copy(update={"foundation_model_calls": 1})
    result = rows[0].result.model_copy(update={"end_to_end_compute": compute})
    assert not decide_after_a(
        [rows[0].model_copy(update={"result": result}), *rows[1:]],
        protocol_sha256=config.protocol_sha256,
    ).proceed_to_b
    damaged = rows[0].model_copy(update={"timed_success": False})
    assert not decide_after_a(
        [damaged, *rows[1:]], protocol_sha256=config.protocol_sha256
    ).proceed_to_b
    compression_losses = []
    for row in rows:
        if row.condition == "compressed_eventflow":
            action = Action(0, row.truth.activation_time, 99)
            result = row.result.model_copy(update={"actions": (action,)})
            row = row.model_copy(
                update={
                    "result": result,
                    "score": score_actions(row.truth, (action,)),
                    "timed_success": False,
                }
            )
        compression_losses.append(row)
    assert decide_after_a(compression_losses, protocol_sha256=config.protocol_sha256).proceed_to_b


def test_legacy_compatibility_subset_is_fixed_before_scores() -> None:
    depths = [2, 3, 4] * 40
    indices = legacy_compatibility_indices(depths)
    assert len(indices) == 32
    assert [depths[index] for index in indices].count(2) == 12
    assert [depths[index] for index in indices].count(3) == 12
    assert [depths[index] for index in indices].count(4) == 8


def test_resume_charges_attempts_and_preserves_completed_rows(tmp_path, monkeypatch) -> None:
    from silent_cascade.eval import comparison_runner

    config = ComparisonConfig()
    allocation = diagnostic_allocations()["iid"]
    first = allocation.blocks[0].model_copy(update={"episode_count": 4})
    manifest = build_manifest(config, "iid", allocation.model_copy(update={"blocks": (first,)}))
    assert timing_indices(manifest) == (0, 1, 2, 3)
    torch.manual_seed(11)
    model = EventFlowModel(config.phase4_config.neural).eval()
    identity = ComparisonIdentity(
        condition="intact_eventflow",
        manifest_name="iid",
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
    initial = BudgetLedger(
        elapsed_scientific_seconds=7200.0,
        extensions=("approved additional local storage",),
    )
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        result = ConditionResult(stop_reason="fixture")
        kwargs["trace_sink"](result, None)
        return result

    monkeypatch.setattr(comparison_runner, "run_intact_episode", interrupted)
    output = tmp_path / "intact"
    with pytest.raises(KeyboardInterrupt):
        run_comparison(
            config=config,
            manifest=manifest,
            identity=identity,
            model=model,
            output_dir=output,
            budget=initial,
        )
    interrupted_budget = BudgetLedger.model_validate_json((output / "budget.json").read_bytes())
    assert interrupted_budget.attempts == 1
    assert interrupted_budget.extensions[0] == initial.extensions[0]
    assert any("time target" in note for note in interrupted_budget.extensions)
    assert len(list((output / "rows").glob("*.json.gz"))) == 2

    def resumed(*args, **kwargs):
        result = ConditionResult(stop_reason="fixture")
        kwargs["trace_sink"](result, None)
        return result

    monkeypatch.setattr(comparison_runner, "run_intact_episode", resumed)
    run_comparison(
        config=config,
        manifest=manifest,
        identity=identity,
        model=model,
        output_dir=output,
        budget=initial,
    )
    final_budget = BudgetLedger.model_validate_json((output / "budget.json").read_bytes())
    assert final_budget.attempts == 2
    assert final_budget.elapsed_scientific_seconds >= interrupted_budget.elapsed_scientific_seconds
    assert final_budget.extensions == interrupted_budget.extensions
    assert json.loads((output / "inventory.json").read_text())["row_count"] == 4


def test_cli_ponder_refuses_unadmitted_milestone(tmp_path) -> None:
    script = Path(__file__).parents[2] / "scripts/run_phase5a.py"
    spec = importlib.util.spec_from_file_location("run_phase5a", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    decision = {"status": "deferred_after_a", "proceed_to_b": False}
    report = tmp_path / "report"
    report.mkdir()
    (report / "decision.json").write_text(json.dumps(decision))
    with pytest.raises(ValueError, match="deferred"):
        module.main(["ponder", "--run-dir", str(tmp_path)])


def test_checkpoint_check_authenticates_frozen_producer_without_current_tree(tmp_path, monkeypatch):
    retained = Path(
        "/Users/artemlegotin/Library/Application Support/silent-cascade/runs/"
        "phase4-pilot-artifact-fix-v1/event_flow/11/pilot"
    )
    if not (retained / "training-result.json").is_file():
        pytest.skip("retained accepted Phase 4 archive is unavailable")
    script = Path(__file__).parents[2] / "scripts/run_phase5a.py"
    spec = importlib.util.spec_from_file_location("run_phase5a_checkpoint", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "authenticate_run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("historical authentication must not compare current package inventory")
        ),
        raising=False,
    )
    module._checkpoint_check(
        Namespace(
            config=Path(__file__).parents[2] / "configs/eval/phase5a.yaml",
            run_dir=tmp_path,
            phase4_run_dir=retained,
        )
    )
    assert json.loads((tmp_path / "checkpoint.json").read_text())["weights_sha256"].startswith(
        "bc2850"
    )
