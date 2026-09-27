"""Offline, phase-scoped exploratory comparison commands."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

from silent_cascade.config import resolve_config
from silent_cascade.env.pilot import curriculum_to_bundle
from silent_cascade.env.reward import score_actions
from silent_cascade.eval.comparison_data import prepare_manifests, publish_manifest
from silent_cascade.eval.comparison_protocol import legacy_compatibility_indices
from silent_cascade.eval.comparison_runner import (
    ponder_state_sha256,
    run_comparison,
    run_intact_episode,
)
from silent_cascade.eval.comparison_types import (
    ACCEPTED_EVENTFLOW_STATE_SHA256,
    ACCEPTED_EVENTFLOW_WEIGHTS_SHA256,
    ACCEPTED_PHASE4_GATE_SHA256,
    ACCEPTED_PHASE4_SOURCE,
    BudgetLedger,
    ComparisonConfig,
    ComparisonIdentity,
    ComparisonManifest,
)
from silent_cascade.eval.metrics import TimedEpisodeRow
from silent_cascade.eventflow.neural_weights import load_neural_weights
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from silent_cascade.io import atomic_create_bytes, atomic_write_bytes
from silent_cascade.models.activation_ponder import ActivationPonderModel, matched_ponder_config
from silent_cascade.report.comparison import build_comparison_report, build_ponder_report
from silent_cascade.train.activation_ponder import (
    fit_ponder,
    load_main_checkpoint,
    run_debug_competence,
)
from silent_cascade.train.curriculum_data import make_curriculum_example
from silent_cascade.train.pilot_data import PilotManifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "manifests/validation/phase5a-v1"
PHASE4_GATE = ROOT / "manifests/validation/phase4/autonomous-gate-v1.json"


def _config(path: Path) -> ComparisonConfig:
    return resolve_config(ComparisonConfig, [path]).config


def _source_revision() -> str:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.call(
        ["git", "diff", "--quiet", "HEAD", "--", "src", "scripts", "configs"], cwd=ROOT
    ):
        raise ValueError("execution source has uncommitted implementation changes")
    return revision


def _committed_manifest(path: Path) -> None:
    relative = path.relative_to(ROOT)
    if subprocess.call(
        ["git", "ls-files", "--error-unmatch", "--", str(relative)],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ):
        raise ValueError("diagnostic manifest must be committed before evaluation")
    if subprocess.call(["git", "diff", "--quiet", "HEAD", "--", str(relative)], cwd=ROOT):
        raise ValueError("diagnostic manifest differs from committed revision")


def _prepare(args: argparse.Namespace) -> None:
    config = _config(args.config)
    manifests = prepare_manifests(config, output_dir=MANIFEST_DIR)
    for name, manifest in manifests.items():
        publish_manifest(manifest, args.run_dir / "manifests" / f"{name}.json")
    print(
        "Prepared and oracle-audited diagnostic manifests: "
        + ", ".join(f"{name}={len(manifest.entries)}" for name, manifest in manifests.items())
    )


def _checkpoint_check(args: argparse.Namespace) -> None:
    config = _config(args.config)
    if sha256_file(PHASE4_GATE) != ACCEPTED_PHASE4_GATE_SHA256:
        raise ValueError("accepted Phase 4 gate artifact hash differs")
    gate = json.loads(PHASE4_GATE.read_bytes())
    training_path = args.phase4_run_dir / "training-result.json"
    training = json.loads(training_path.read_bytes())
    selected = training.get("selected_weights")
    receipt = gate.get("local_verification", {})
    if (
        gate.get("outcome") != "passed"
        or gate.get("offline_passed") is not True
        or gate.get("foundation_model_calls") != 0
        or gate.get("source", {}).get("source_commit") != ACCEPTED_PHASE4_SOURCE
        or gate.get("selected_weights_sha256") != ACCEPTED_EVENTFLOW_WEIGHTS_SHA256
        or gate.get("model_state_sha256") != ACCEPTED_EVENTFLOW_STATE_SHA256
        or gate.get("upstream_artifact_hashes", {}).get("training-result.json")
        != sha256_file(training_path)
        or training.get("status") != "robustness_complete"
        or training.get("gate_eligible") is not True
        or selected is None
        or gate.get("training_result", {}).get("selected_weights") != selected
        or receipt.get("passed") is not True
        or sha256_bytes(canonical_json_bytes(receipt.get("receipt")))
        != receipt.get("receipt_sha256")
        or sha256_file(ROOT / receipt.get("path", "missing")) != receipt.get("receipt_sha256")
    ):
        raise ValueError("retained Phase 4 gate or historical receipt differs")
    if subprocess.call(["git", "cat-file", "-e", f"{ACCEPTED_PHASE4_SOURCE}^{{commit}}"], cwd=ROOT):
        raise ValueError("accepted Phase 4 producer revision is unavailable")
    weights_path = args.phase4_run_dir / selected["path"]
    weights = load_neural_weights(
        weights_path, expected_sha256=ACCEPTED_EVENTFLOW_WEIGHTS_SHA256, device="cpu"
    )
    if (
        weights.sha256 != ACCEPTED_EVENTFLOW_WEIGHTS_SHA256
        or weights.identity.model_state_sha256 != ACCEPTED_EVENTFLOW_STATE_SHA256
        or weights.identity.source_revision != ACCEPTED_PHASE4_SOURCE
        or weights.model.config != config.phase4_config.neural
        or selected.get("sha256") != weights.sha256
        or selected.get("model_state_sha256") != ACCEPTED_EVENTFLOW_STATE_SHA256
        or selected.get("global_step") != 12000
        or selected.get("eligible") is not True
    ):
        raise ValueError("retained selected EventFlow checkpoint differs from accepted Phase 4")
    descriptor = {
        "schema_version": "phase5a-checkpoint-binding-v1",
        "phase4_gate_sha256": ACCEPTED_PHASE4_GATE_SHA256,
        "producing_source_revision": ACCEPTED_PHASE4_SOURCE,
        "weights_path": str(weights_path),
        "weights_sha256": weights.sha256,
        "model_state_sha256": weights.identity.model_state_sha256,
        "comparison_protocol_sha256": config.protocol_sha256,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    path = args.run_dir / "checkpoint.json"
    raw = canonical_json_bytes(descriptor) + b"\n"
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("existing Phase 5A checkpoint binding differs")
    else:
        atomic_create_bytes(path, raw)
    print("Authenticated accepted Phase 4 selected portable checkpoint")


def _evaluate_a(args: argparse.Namespace) -> None:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in {"", "0"}:
        raise ValueError("MPS fallback must be disabled")
    config = _config(args.config)
    binding = json.loads((args.run_dir / "checkpoint.json").read_bytes())
    if (
        binding["comparison_protocol_sha256"] != config.protocol_sha256
        or binding["weights_sha256"] != ACCEPTED_EVENTFLOW_WEIGHTS_SHA256
        or binding["model_state_sha256"] != ACCEPTED_EVENTFLOW_STATE_SHA256
    ):
        raise ValueError("checkpoint binding differs from accepted protocol")
    weights = load_neural_weights(
        Path(binding["weights_path"]), expected_sha256=binding["weights_sha256"], device="cpu"
    )
    execution_revision = _source_revision()
    cumulative_budget = BudgetLedger(extensions=tuple(args.extension))
    for name in ("iid", "depth"):
        committed = MANIFEST_DIR / f"{name}.json"
        _committed_manifest(committed)
        run_manifest = args.run_dir / "manifests" / f"{name}.json"
        if run_manifest.read_bytes() != committed.read_bytes():
            raise ValueError("run manifest differs from committed diagnostic manifest")
        manifest = ComparisonManifest.model_validate_json(run_manifest.read_bytes())
        if len(manifest.entries) != 512:
            raise ValueError("Milestone A requires all 512 diagnostic episodes per corpus")
        for condition in ("intact_eventflow", "compressed_eventflow"):
            identity = ComparisonIdentity(
                condition=condition,
                manifest_name=name,
                protocol_sha256=config.protocol_sha256,
                config_sha256=config.config_sha256,
                generator_sha256=config.generator_sha256,
                manifest_sha256=sha256_bytes(canonical_json_bytes(manifest)),
                checkpoint_sha256=weights.sha256,
                model_state_sha256=weights.identity.model_state_sha256,
                producing_source_revision=ACCEPTED_PHASE4_SOURCE,
                execution_source_revision=execution_revision,
            )
            destination = args.run_dir / "conditions" / condition / name
            run_comparison(
                config=config,
                manifest=manifest,
                identity=identity,
                model=weights.model,
                output_dir=destination,
                budget=cumulative_budget,
            )
            cumulative_budget = BudgetLedger.model_validate_json(
                (destination / "budget.json").read_bytes()
            )
            print(f"Completed {condition}/{name}: {len(manifest.entries)} episodes")


def _compatibility_check(args: argparse.Namespace) -> None:
    """Reexecute 32 fixed retained primary rows against their accepted decisions."""
    config = _config(args.config)
    binding = json.loads((args.run_dir / "checkpoint.json").read_bytes())
    if binding["comparison_protocol_sha256"] != config.protocol_sha256:
        raise ValueError("compatibility checkpoint protocol differs")
    weights = load_neural_weights(
        Path(binding["weights_path"]), expected_sha256=binding["weights_sha256"], device="cpu"
    )
    manifest = PilotManifest.model_validate_json(
        (ROOT / "manifests/validation/phase4/primary.json").read_bytes()
    )
    indices = legacy_compatibility_indices(
        [entry.projected.path_length for entry in manifest.entries]
    )
    selected = set(indices)
    row_path = args.phase4_run_dir / "final/eval/primary/rows.jsonl"
    matched = 0
    with row_path.open("rb") as handle:
        for index, line in enumerate(handle):
            if index > indices[-1]:
                break
            if index not in selected:
                continue
            retained = TimedEpisodeRow.model_validate_json(line)
            entry = manifest.entries[index]
            example = make_curriculum_example(config.phase4_config, entry.key)
            bundle = curriculum_to_bundle(example, config=config.phase4_config)
            if (
                retained.public_id != bundle.public.init.episode_public_id
                or retained.checkpoint_sha256 != weights.sha256
                or retained.episode_sha256 != entry.projected.episode_sha256
            ):
                raise ValueError("retained primary compatibility identity differs")
            compared = run_intact_episode(bundle, model=weights.model, config=config)
            if (
                compared.actions != retained.actions
                or (compared.error is None) != (retained.error is None)
                or score_actions(bundle.truth, compared.actions) != retained.score
            ):
                raise ValueError(f"intact bridge differs from retained primary row {index}")
            matched += 1
    if matched != 32:
        raise ValueError("retained primary compatibility subset is incomplete")
    receipt = {
        "schema_version": "phase5a-retained-compatibility-v1",
        "protocol_sha256": config.protocol_sha256,
        "checkpoint_sha256": weights.sha256,
        "retained_rows_sha256": sha256_file(row_path),
        "selected_indices": list(indices),
        "matching_rows": matched,
        "foundation_model_calls": 0,
    }
    path = args.run_dir / "retained-compatibility.json"
    raw = canonical_json_bytes(receipt) + b"\n"
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("existing compatibility receipt differs")
    else:
        atomic_create_bytes(path, raw)
    print("Retained intact compatibility: 32/32 matching selected primary rows")


def _report(args: argparse.Namespace) -> None:
    output = args.output or args.run_dir / "report"
    build_comparison_report(args.run_dir, output)
    print(f"Wrote artifact-only comparison report: {output}")


def _competence(args: argparse.Namespace) -> None:
    result = run_debug_competence(
        config=_config(args.config),
        run_dir=args.run_dir / "competence",
        source_revision=_source_revision(),
    )
    print(
        f"Ponderer DEBUG competence: {result['status']}; "
        f"{result.get('last_autonomous_successes', '?')}/64 at cap 24"
    )


def _ponder(args: argparse.Namespace) -> None:
    decision_path = args.run_dir / "report/decision.json"
    if not decision_path.is_file():
        raise ValueError("Milestone A decision is missing")
    decision = json.loads(decision_path.read_bytes())
    if not decision.get("proceed_to_b"):
        raise ValueError("Milestone B was deferred after A")
    config = _config(args.config)
    competence_path = args.run_dir / "competence/result.json"
    if not competence_path.is_file():
        raise ValueError("fixed DEBUG competence result is missing")
    competence = json.loads(competence_path.read_bytes())
    if (
        competence.get("status") != "competent"
        or competence.get("last_autonomous_successes") != 64
        or competence.get("autonomous_denominator") != 64
        or competence.get("promotable") is not False
        or competence.get("foundation_model_calls") != 0
    ):
        raise ValueError("fixed DEBUG competence admission failed")
    debug_checkpoint = (
        args.run_dir / "competence" / f"debug-{competence['completed_update']:04d}.pt"
    )
    if sha256_file(debug_checkpoint) != competence["last_checkpoint_sha256"]:
        raise ValueError("fixed DEBUG competence checkpoint differs")
    competence_budget = BudgetLedger.model_validate_json(
        (args.run_dir / "competence/budget.json").read_bytes()
    )
    fresh_budget = BudgetLedger(
        elapsed_scientific_seconds=competence_budget.elapsed_scientific_seconds,
        retained_bytes=competence_budget.retained_bytes,
        extensions=competence_budget.extensions,
    )
    trained = fit_ponder(
        config=config,
        run_dir=args.run_dir,
        budget=fresh_budget,
        resume=args.resume,
    )
    if trained.status == "profiling" or trained.status == "inconclusive_budget":
        print(f"Ponderer main trajectory: {trained.status}; updates={trained.completed_updates}")
        return
    if trained.completed_updates != trained.chosen_updates or not trained.selected_checkpoint:
        raise ValueError("ponder training has no completed selected checkpoint")
    model = ActivationPonderModel(matched_ponder_config()).to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)
    selected_path = args.run_dir / "ponder" / trained.selected_checkpoint
    selected_update, _ = load_main_checkpoint(
        selected_path,
        model,
        optimizer,
        config_sha256=config.config_sha256,
        source_revision=trained.source_revision,
    )
    if (
        selected_update != trained.selected_update
        or sha256_file(selected_path) != trained.selected_checkpoint_sha256
    ):
        raise ValueError("selected ponder checkpoint differs from primary validation rank")
    model.eval()
    state_sha = ponder_state_sha256(model)
    cumulative = BudgetLedger.model_validate_json(
        (args.run_dir / "ponder/budget.json").read_bytes()
    )
    for name in ("iid", "depth"):
        committed = MANIFEST_DIR / f"{name}.json"
        _committed_manifest(committed)
        path = args.run_dir / "manifests" / f"{name}.json"
        if path.read_bytes() != committed.read_bytes():
            raise ValueError("run manifest differs from committed diagnostic manifest")
        manifest = ComparisonManifest.model_validate_json(path.read_bytes())
        for cap in (4, 8, 12, 16, 24):
            identity = ComparisonIdentity(
                condition="activation_ponder",
                manifest_name=name,
                transition_cap=cap,
                protocol_sha256=config.protocol_sha256,
                config_sha256=config.config_sha256,
                generator_sha256=config.generator_sha256,
                manifest_sha256=sha256_bytes(canonical_json_bytes(manifest)),
                checkpoint_sha256=trained.selected_checkpoint_sha256,
                model_state_sha256=state_sha,
                producing_source_revision=trained.source_revision,
                execution_source_revision=_source_revision(),
            )
            destination = args.run_dir / "conditions/activation_ponder" / f"{name}-cap{cap}"
            run_comparison(
                config=config,
                manifest=manifest,
                identity=identity,
                model=model,
                output_dir=destination,
                budget=cumulative,
            )
            cumulative = BudgetLedger.model_validate_json(
                (destination / "budget.json").read_bytes()
            )
            print(f"Ponderer {name}/cap{cap}: complete {len(manifest.entries)} rows", flush=True)
    print("Ponderer all ten diagnostic units complete; artifact report follows", flush=True)
    report_dir = args.run_dir / "ponder-report"
    if not report_dir.exists():
        build_ponder_report(args.run_dir, report_dir)
    else:
        report = json.loads((report_dir / "report.json").read_bytes())
        if (
            report.get("complete_b") is not True
            or report.get("selected_update") != trained.selected_update
        ):
            raise ValueError("existing ponder report differs from selected checkpoint")
    if trained.status != "completed_exploratory" or not trained.execution_complete:
        finished = trained.model_copy(
            update={
                "status": "completed_exploratory",
                "execution_complete": True,
            }
        )
        atomic_write_bytes(
            args.run_dir / "ponder/result.json", canonical_json_bytes(finished) + b"\n"
        )
    print(f"Ponderer exploratory report: {report_dir}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (
        "prepare",
        "checkpoint-check",
        "compatibility-check",
        "evaluate-a",
        "report",
        "competence",
        "ponder",
    ):
        sub = commands.add_parser(command)
        sub.add_argument("--run-dir", type=Path, required=True)
        if command in {
            "prepare",
            "checkpoint-check",
            "compatibility-check",
            "evaluate-a",
            "competence",
            "ponder",
        }:
            sub.add_argument("--config", type=Path, default=ROOT / "configs/eval/phase5a.yaml")
        if command in {"checkpoint-check", "compatibility-check"}:
            sub.add_argument("--phase4-run-dir", type=Path, required=True)
        if command == "evaluate-a":
            sub.add_argument("--extension", action="append", default=[])
        if command == "report":
            sub.add_argument("--output", type=Path)
        if command == "ponder":
            sub.add_argument("--resume", type=Path)
    args = parser.parse_args(argv)
    {
        "prepare": _prepare,
        "checkpoint-check": _checkpoint_check,
        "compatibility-check": _compatibility_check,
        "evaluate-a": _evaluate_a,
        "report": _report,
        "competence": _competence,
        "ponder": _ponder,
    }[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
