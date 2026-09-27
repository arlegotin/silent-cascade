"""Independent local audit of the executed Phase 5A exploratory comparison."""

import gzip
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from silent_cascade.env.episode import episode_sha256
from silent_cascade.env.reward import score_actions
from silent_cascade.eval.comparison_data import iter_bundles
from silent_cascade.eval.comparison_protocol import decide_after_a, ponder_checkpoint_rank, timing_indices
from silent_cascade.eval.comparison_runner import (
    _run_compressed_episode,
    ponder_state_sha256,
    run_intact_episode,
    run_ponder_episode,
)
from silent_cascade.eval.comparison_types import ComparisonConfig, ComparisonIdentity, ComparisonManifest, ComparisonRow, PonderTrainingResult
from silent_cascade.eventflow.neural_weights import load_neural_weights
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from silent_cascade.models.activation_ponder import ActivationPonderModel, matched_ponder_config
from silent_cascade.report.comparison import _read_condition
from silent_cascade.train.activation_ponder import load_main_checkpoint, validate_primary_validation_receipt
from silent_cascade.train.pilot_data import PilotManifest


ROOT = Path("/Users/artemlegotin/Library/Application Support/silent-cascade/runs/phase5a-v1")
REPO = Path("/Volumes/git/legotin/silent-cascade")
START = perf_counter()
config = ComparisonConfig()
report_a = json.loads((ROOT / "report/report.json").read_bytes())
report_b = json.loads((ROOT / "ponder-report/report.json").read_bytes())
binding = json.loads((ROOT / "checkpoint.json").read_bytes())
eventflow = load_neural_weights(
    Path(binding["weights_path"]), expected_sha256=binding["weights_sha256"], device="cpu"
).model.eval()
trained = PonderTrainingResult.model_validate_json((ROOT / "ponder/result.json").read_bytes())
ponder = ActivationPonderModel(matched_ponder_config()).eval()
optimizer = torch.optim.AdamW(ponder.parameters(), lr=3e-4)
selected_path = ROOT / "ponder" / trained.selected_checkpoint
selected_update, _ = load_main_checkpoint(
    selected_path,
    ponder,
    optimizer,
    config_sha256=config.config_sha256,
    source_revision=trained.source_revision,
)
assert selected_update == trained.selected_update == 12000
assert sha256_file(selected_path) == trained.selected_checkpoint_sha256
assert trained.parameters == report_b["parameters"]
assert trained.foundation_model_calls == 0
assert config.protocol_sha256 == report_a["protocol_sha256"]

primary_path = REPO / "manifests/validation/phase4/primary.json"
primary = PilotManifest.model_validate_json(primary_path.read_bytes())
ranks = []
for update in range(1000, 12001, 1000):
    checkpoint = ROOT / "ponder" / f"main-{update:04d}.pt"
    receipt = validate_primary_validation_receipt(
        ROOT / "ponder" / f"validation-{update:04d}/receipt.json",
        entries=primary.entries,
        update=update,
        checkpoint_sha256=sha256_file(checkpoint),
        manifest_sha256=sha256_file(primary_path),
    )
    ranks.append((ponder_checkpoint_rank(
        update=update,
        cap=24,
        primary_successes=receipt["timed_successes"],
        negative_false_actions=receipt["negative_false_actions"],
    ), update))
assert max(ranks) == (trained.selected_rank, trained.selected_update)


def independent_interval(reference, candidate):
    difference = np.asarray(
        [int(b.timed_success) - int(a.timed_success) for a, b in zip(reference, candidate, strict=True)],
        dtype=np.int8,
    )
    strata = defaultdict(list)
    for index, row in enumerate(reference):
        strata[(row.path_length, row.variant.value)].append(index)
    rng = np.random.default_rng(8009)
    draws = np.zeros(10000, dtype=np.float64)
    for members in strata.values():
        sample = difference[np.asarray(members)]
        positions = rng.integers(0, len(sample), size=(10000, len(sample)))
        draws += sample[positions].sum(axis=1)
    draws /= len(reference)
    return float(difference.mean()), [float(x) for x in np.quantile(draws, [0.025, 0.975])]


replay_seconds = {"a": 0.0, "b": 0.0}
groups = {}
all_a_rows = []
replay_total = 0
for corpus in ("iid", "depth"):
    manifest_path = REPO / "manifests/validation/phase5a-v1" / f"{corpus}.json"
    manifest = ComparisonManifest.model_validate_json(manifest_path.read_bytes())
    assert (ROOT / "manifests" / f"{corpus}.json").read_bytes() == manifest_path.read_bytes()
    assert len(manifest.entries) == 512
    bundles = {index: bundle for index, bundle in enumerate(iter_bundles(manifest, config))
               if index in timing_indices(manifest)[::2]}
    assert len(bundles) == 16
    intact_rows = None
    compressed_rows = None
    for condition, cap in (
        ("intact_eventflow", None),
        ("compressed_eventflow", None),
        *(("activation_ponder", cap) for cap in (4, 8, 12, 16, 24)),
    ):
        identity, rows, budget = _read_condition(ROOT, condition=condition, manifest=manifest, cap=cap)
        assert identity.execution_source_revision == "332cc41492deedfbf041c3b025f94be1c9d9fbb8"
        assert identity.protocol_sha256 == config.protocol_sha256
        assert identity.generator_sha256 == config.generator_sha256
        assert identity.foundation_model_calls == 0
        if cap is None:
            assert identity.checkpoint_sha256 == binding["weights_sha256"]
            summary = report_a["corpora"][corpus][condition]
            all_a_rows.extend(rows)
            scope = "a"
        else:
            assert identity.checkpoint_sha256 == trained.selected_checkpoint_sha256
            assert identity.model_state_sha256 == ponder_state_sha256(ponder)
            summary = report_b["corpora"][corpus][f"activation_ponder_cap{cap}"]
            scope = "b"
        assert summary["episodes"] == len(rows) == 512
        assert summary["timed_successes"] == sum(row.timed_success for row in rows)
        assert summary["errors"] == sum(row.error is not None for row in rows) == 0
        assert summary["mean_end_to_end_forward_macs"] == sum(
            row.result.end_to_end_compute.forward_macs for row in rows
        ) / 512
        assert summary["mean_post_activation_forward_macs"] == sum(
            row.result.post_activation_compute.forward_macs for row in rows
        ) / 512
        assert summary["mean_records_scored"] == sum(
            row.result.end_to_end_compute.records_scored for row in rows
        ) / 512
        by_stratum = defaultdict(lambda: {"episodes": 0, "timed_successes": 0, "errors": 0})
        for row, entry in zip(rows, manifest.entries, strict=True):
            assert row.public_id == entry.public_id
            assert row.episode_sha256 == entry.episode_sha256
            assert row.score == score_actions(row.truth, row.result.actions)
            assert row.timed_success == row.score.timed_success
            assert row.result.end_to_end_compute.foundation_model_calls == 0
            for key in (f"variant:{row.variant.value}", f"depth:{row.path_length}"):
                by_stratum[key]["episodes"] += 1
                by_stratum[key]["timed_successes"] += int(row.timed_success)
                by_stratum[key]["errors"] += int(row.error is not None)
        assert dict(by_stratum) == summary["by_stratum"]
        for index, bundle in bundles.items():
            assert rows[index].episode_sha256 == episode_sha256(bundle)
            started = perf_counter()
            if condition == "intact_eventflow":
                rerun = run_intact_episode(bundle, model=eventflow, config=config)
            elif condition == "compressed_eventflow":
                rerun = _run_compressed_episode(bundle, model=eventflow, config=config)
            else:
                rerun = run_ponder_episode(bundle, model=ponder, cap=cap)
            replay_seconds[scope] += perf_counter() - started
            saved = rows[index].result
            assert rerun.actions == saved.actions
            assert rerun.stop_reason == saved.stop_reason
            assert rerun.error == saved.error
            assert rerun.steps == saved.steps
            assert rerun.trace_sha256 == saved.trace_sha256
            assert rerun.end_to_end_compute.forward_macs == saved.end_to_end_compute.forward_macs
            assert rerun.end_to_end_compute.records_scored == saved.end_to_end_compute.records_scored
            assert rerun.post_activation_compute.forward_macs == saved.post_activation_compute.forward_macs
            replay_total += 1
        label = f"{corpus}/{condition}" if cap is None else f"{corpus}/activation_ponder_cap{cap}"
        groups[label] = {
            "identity_sha256": identity.sha256,
            "rows_sha256": sha256_file(ROOT / "conditions" / condition /
                (corpus if cap is None else f"{corpus}-cap{cap}") / "rows.jsonl.gz"),
            "episodes": len(rows),
            "timed_successes": sum(row.timed_success for row in rows),
            "errors": 0,
            "replays_matched": 16,
            "budget_seconds": budget.elapsed_scientific_seconds,
        }
        if condition == "intact_eventflow":
            intact_rows = rows
        elif condition == "compressed_eventflow":
            compressed_rows = rows
        else:
            interval = independent_interval(intact_rows, rows)
            saved_interval = report_b["paired_intervals"][corpus][f"activation_ponder_cap{cap}"]
            assert interval == (saved_interval["ponder_minus_intact"], saved_interval["interval_95"])
    assert independent_interval(intact_rows, compressed_rows) == (
        report_a["paired_intervals"][corpus]["compressed_minus_intact"],
        report_a["paired_intervals"][corpus]["interval_95"],
    )

decision = decide_after_a(all_a_rows, protocol_sha256=config.protocol_sha256)
assert decision.model_dump(mode="json") == report_a["decision"]
assert json.loads((ROOT / "report/decision.json").read_bytes()) == report_a["decision"]
assert replay_total == 224
assert report_b["selected_update"] == selected_update
assert report_b["nearest_iid_compute_cap"] is None

final_a_budget = json.loads((ROOT / "conditions/compressed_eventflow/depth/budget.json").read_bytes())
final_b_budget = json.loads((ROOT / "conditions/activation_ponder/depth-cap24/budget.json").read_bytes())
receipt = {
    "schema_version": "phase5a-independent-local-verification-v1",
    "execution_source_revision": "332cc41492deedfbf041c3b025f94be1c9d9fbb8",
    "report_a_sha256": sha256_file(ROOT / "report/report.json"),
    "report_b_sha256": sha256_file(ROOT / "ponder-report/report.json"),
    "selected_checkpoint_sha256": trained.selected_checkpoint_sha256,
    "selected_update": selected_update,
    "primary_validation_boundaries_recounted": 12,
    "primary_validation_rows_recounted": 120000,
    "condition_rows_verified": 7168,
    "replays_matched": replay_total,
    "groups": groups,
    "replay_scientific_seconds": replay_seconds,
    "cumulative_scientific_seconds_including_replay": {
        "a": final_a_budget["elapsed_scientific_seconds"] + replay_seconds["a"],
        "b": final_b_budget["elapsed_scientific_seconds"] + replay_seconds["b"],
    },
    "foundation_model_calls": 0,
    "elapsed_engineering_seconds": perf_counter() - START,
}
Path("/private/tmp/phase5a_independent_verify.json").write_bytes(canonical_json_bytes(receipt) + b"\n")
print(json.dumps({"matched": replay_total, "selected": selected_update, "replay_seconds": replay_seconds}))
