"""Artifact-only exploratory Phase 5A report with strict paired coverage checks."""

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from silent_cascade.eval.comparison_protocol import (
    decide_after_a,
    nearest_ponder_cap,
    ponder_checkpoint_rank,
    timing_indices,
)
from silent_cascade.eval.comparison_types import (
    BudgetLedger,
    ComparisonIdentity,
    ComparisonManifest,
    ComparisonRow,
    PonderTrainingResult,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from silent_cascade.io import atomic_create_bytes


def _read_condition(
    run_dir: Path, *, condition: str, manifest: ComparisonManifest, cap: int | None = None
) -> tuple[ComparisonIdentity, list[ComparisonRow], BudgetLedger]:
    name = manifest.name if cap is None else f"{manifest.name}-cap{cap}"
    root = run_dir / "conditions" / condition / name
    for name in ("identity.json", "inventory.json", "rows.jsonl.gz", "budget.json"):
        if not (root / name).is_file():
            raise ValueError(f"missing comparison {condition}/{manifest.name}/{name}")
    identity = ComparisonIdentity.model_validate_json((root / "identity.json").read_bytes())
    inventory = json.loads((root / "inventory.json").read_bytes())
    budget = BudgetLedger.model_validate_json((root / "budget.json").read_bytes())
    if (
        identity.condition != condition
        or identity.manifest_name != manifest.name
        or identity.manifest_sha256 != sha256_bytes(canonical_json_bytes(manifest))
        or identity.protocol_sha256 != manifest.protocol_sha256
        or identity.config_sha256 != manifest.config_sha256
        or identity.foundation_model_calls != 0
        or identity.dirty_source
        or identity.transition_cap != cap
    ):
        raise ValueError("comparison identity differs from diagnostic manifest")
    if (
        inventory.get("schema_version") != "phase5a-run-inventory-v1"
        or inventory.get("identity_sha256") != identity.sha256
        or inventory.get("manifest_sha256") != identity.manifest_sha256
        or inventory.get("rows_sha256") != sha256_file(root / "rows.jsonl.gz")
    ):
        raise ValueError("comparison inventory or rows hash is corrupt")
    with gzip.open(root / "rows.jsonl.gz", "rb") as handle:
        rows = [ComparisonRow.model_validate_json(line) for line in handle if line.strip()]
    if len(rows) != inventory.get("row_count") or len(rows) != len(manifest.entries):
        raise ValueError("comparison inventory has missing or extra rows")
    if sum(row.error is not None for row in rows) != inventory.get("error_count"):
        raise ValueError("comparison inventory error count differs")
    for index, (row, entry) in enumerate(zip(rows, manifest.entries, strict=True)):
        if (
            row.public_id != entry.public_id
            or row.episode_sha256 != entry.episode_sha256
            or row.variant is not entry.variant
            or row.path_length != entry.requested_path_length
            or row.truth.recipe.accepted_attempt != entry.accepted_attempt
            or row.condition != condition
            or row.manifest_name != manifest.name
            or row.protocol_sha256 != manifest.protocol_sha256
            or row.identity_sha256 != identity.sha256
            or row.result.end_to_end_compute.foundation_model_calls != 0
            or row.result.post_activation_compute.foundation_model_calls != 0
        ):
            raise ValueError("comparison row differs from manifest, identity, or offline protocol")
        if row.full_trace_ref is not None:
            ref = Path(row.full_trace_ref)
            if ref.parts != ("traces", f"{index:05d}.trajectory.json.gz"):
                raise ValueError("comparison full trace reference is invalid")
            trace_path = root / ref
            if not trace_path.is_file() or sha256_file(trace_path) != row.full_trace_sha256:
                raise ValueError("comparison full trace hash is corrupt")
            trace = json.loads(gzip.decompress(trace_path.read_bytes()))
            if (
                trace.get("identity_sha256") != identity.sha256
                or trace.get("episode_sha256") != row.episode_sha256
            ):
                raise ValueError("comparison full trace binding differs")
        elif not row.timed_success:
            raise ValueError("failed comparison episode lacks a retained full trace")
    retained_successes = sum(row.timed_success and row.full_trace_ref is not None for row in rows)
    if retained_successes < min(32, sum(row.timed_success for row in rows)):
        raise ValueError("comparison lacks 32 retained successful traces")
    return identity, rows, budget


def _summary(rows: list[ComparisonRow], budget: BudgetLedger, timing: tuple[int, ...]) -> dict:
    groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "timed_successes": 0, "errors": 0}
    )
    errors: Counter[str] = Counter()
    for row in rows:
        for key in (f"variant:{row.variant.value}", f"depth:{row.path_length}"):
            counts = groups[key]
            counts["episodes"] += 1
            counts["timed_successes"] += int(row.timed_success)
            counts["errors"] += int(row.error is not None)
        if row.error is not None:
            errors[row.error.code] += 1
    n = len(rows)
    return {
        "episodes": n,
        "timed_successes": sum(row.timed_success for row in rows),
        "errors": sum(row.error is not None for row in rows),
        "error_categories": dict(sorted(errors.items())),
        "by_stratum": dict(sorted(groups.items())),
        "mean_end_to_end_forward_macs": sum(
            row.result.end_to_end_compute.forward_macs for row in rows
        )
        / n
        if n
        else None,
        "mean_post_activation_forward_macs": sum(
            row.result.post_activation_compute.forward_macs for row in rows
        )
        / n
        if n
        else None,
        "mean_records_scored": sum(row.result.end_to_end_compute.records_scored for row in rows) / n
        if n
        else None,
        "mean_inference_wall_seconds": sum(row.inference_wall_seconds for row in rows) / n
        if n
        else None,
        "timing_subset": {
            "episodes": len(timing),
            "indices": list(timing),
            "mean_inference_wall_seconds": sum(
                rows[index].inference_wall_seconds for index in timing
            )
            / len(timing)
            if timing
            else None,
        },
        "inference_wall_seconds": sum(row.inference_wall_seconds for row in rows),
        "cumulative_scientific_seconds": budget.elapsed_scientific_seconds,
        "cumulative_retained_bytes": budget.retained_bytes,
        "attempts": budget.attempts,
        "extensions": list(budget.extensions),
    }


def _paired_interval(intact: list[ComparisonRow], compressed: list[ComparisonRow]) -> dict:
    """10,000 fixed-seed episode bootstrap draws within depth/variant strata."""
    by_key: dict[tuple[int, str], list[int]] = defaultdict(list)
    differences = np.array(
        [
            int(b.timed_success) - int(a.timed_success)
            for a, b in zip(intact, compressed, strict=True)
        ],
        dtype=np.int8,
    )
    for index, row in enumerate(intact):
        by_key[(row.path_length, row.variant.value)].append(index)
    rng = np.random.default_rng(8009)
    replicates = np.zeros(10_000, dtype=np.float64)
    for indices in by_key.values():
        stratum = differences[np.asarray(indices)]
        draws = rng.integers(0, len(stratum), size=(10_000, len(stratum)))
        replicates += stratum[draws].sum(axis=1)
    replicates /= len(differences)
    lo, hi = np.quantile(replicates, [0.025, 0.975])
    return {
        "compressed_minus_intact": float(differences.mean()),
        "interval_95": [float(lo), float(hi)],
        "interval_label": "single-seed exploratory episode interval",
        "bootstrap_replicates": 10_000,
        "analysis_seed": 8009,
    }


def build_comparison_report(run_dir: Path, output_dir: Path) -> Path:
    """Authenticate saved evidence, then publish only exploratory counts and intervals."""
    if output_dir.exists():
        raise ValueError("comparison report destination must be fresh")
    manifests = {}
    for name in ("iid", "depth"):
        path = run_dir / "manifests" / f"{name}.json"
        if not path.is_file():
            raise ValueError(f"missing comparison manifest {name}")
        manifest = ComparisonManifest.model_validate_json(path.read_bytes())
        if manifest.name != name or manifest.gate_eligible or len(manifest.entries) > 512:
            raise ValueError("diagnostic manifest has invalid purpose or coverage")
        manifests[name] = manifest
    if manifests["iid"].protocol_sha256 != manifests["depth"].protocol_sha256:
        raise ValueError("diagnostic manifests have different protocols")
    if {e.public_id for e in manifests["iid"].entries} & {
        e.public_id for e in manifests["depth"].entries
    }:
        raise ValueError("diagnostic manifests overlap")
    identities = {}
    all_rows = []
    corpora = {}
    intervals = {}
    for name, manifest in manifests.items():
        corpora[name] = {}
        by_condition = {}
        for condition in ("intact_eventflow", "compressed_eventflow"):
            identity, rows, budget = _read_condition(
                run_dir, condition=condition, manifest=manifest
            )
            identities[(name, condition)] = identity
            by_condition[condition] = rows
            all_rows.extend(rows)
            corpora[name][condition] = _summary(rows, budget, timing_indices(manifest))
        intact = by_condition["intact_eventflow"]
        compressed = by_condition["compressed_eventflow"]
        if [(r.public_id, r.episode_sha256) for r in intact] != [
            (r.public_id, r.episode_sha256) for r in compressed
        ]:
            raise ValueError("missing paired comparison rows")
        intervals[name] = _paired_interval(intact, compressed)
    checkpoint_bindings = {
        (
            identity.checkpoint_sha256,
            identity.model_state_sha256,
            identity.producing_source_revision,
        )
        for identity in identities.values()
    }
    if len(checkpoint_bindings) != 1:
        raise ValueError("same-checkpoint comparison identities differ")
    protocol_sha256 = manifests["iid"].protocol_sha256
    decision = decide_after_a(all_rows, protocol_sha256=protocol_sha256)
    complete_a = all(len(manifest.entries) == 512 for manifest in manifests.values())
    report = {
        "schema_version": "phase5a-comparison-report-v1",
        "purpose": "exploratory_comparison",
        "gate_eligible": False,
        "complete_a": complete_a,
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": {
            name: sha256_bytes(canonical_json_bytes(manifest))
            for name, manifest in manifests.items()
        },
        "identities": {
            f"{name}/{condition}": identity.sha256
            for (name, condition), identity in identities.items()
        },
        "corpora": corpora,
        "paired_intervals": intervals,
        "decision": decision.model_dump(mode="json"),
        "claim_boundary": (
            "Same-checkpoint execution-regime diagnostic only; no competitive architectural "
            "merit, final Phase 5 gate, or confirmatory scientific-support claim."
        ),
    }
    lines = [
        "# Phase 5A comparative pilot — Milestone A",
        "",
        "Single-seed exploratory diagnostic. No final Phase 5 gate or scientific-support claim.",
        "",
        f"Protocol: `{protocol_sha256}`. Complete paired A corpora: {complete_a}.",
        f"Decision: **{decision.status}**; reasons: {', '.join(decision.reasons) or 'none'}.",
        "",
        "| Corpus | Condition | Timed success | Errors | Inference seconds |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name in ("iid", "depth"):
        for condition in ("intact_eventflow", "compressed_eventflow"):
            summary = corpora[name][condition]
            lines.append(
                f"| {name} | {condition} | {summary['timed_successes']}/{summary['episodes']} "
                f"| {summary['errors']} | {summary['inference_wall_seconds']:.3f} |"
            )
    lines.extend(["", report["claim_boundary"], ""])
    output_dir.mkdir(parents=True)
    atomic_create_bytes(output_dir / "report.json", canonical_json_bytes(report) + b"\n")
    atomic_create_bytes(output_dir / "report.md", "\n".join(lines).encode())
    atomic_create_bytes(output_dir / "decision.json", canonical_json_bytes(decision) + b"\n")
    return output_dir


def build_ponder_report(run_dir: Path, output_dir: Path) -> Path:
    """Derive B from saved full rows and primary-only checkpoint selection evidence."""
    if output_dir.exists():
        raise ValueError("ponder report destination must be fresh")
    trained_path = run_dir / "ponder/result.json"
    if not trained_path.is_file():
        raise ValueError("missing ponder training result")
    trained = PonderTrainingResult.model_validate_json(trained_path.read_bytes())
    if (
        trained.chosen_updates is None
        or trained.completed_updates != trained.chosen_updates
        or trained.selected_update is None
        or trained.selected_checkpoint is None
        or sha256_file(run_dir / "ponder/profile.json") != trained.profile_sha256
        or sha256_file(run_dir / "ponder/admission.json") != trained.admission_sha256
    ):
        raise ValueError("ponder training or admission is incomplete")
    expected_boundaries = tuple(range(1000, trained.completed_updates + 1, 1000))
    if trained.validation_boundaries != expected_boundaries:
        raise ValueError("primary selection lacks complete validation boundaries")
    ranked = []
    validation_receipts = {}
    for update in expected_boundaries:
        path = run_dir / "ponder" / f"validation-{update:04d}/receipt.json"
        receipt = json.loads(path.read_bytes())
        checkpoint = run_dir / "ponder" / f"main-{update:04d}.pt"
        if (
            receipt.get("denominator") != 10000
            or receipt.get("cap") != 24
            or receipt.get("foundation_model_calls") != 0
            or receipt.get("checkpoint_sha256") != sha256_file(checkpoint)
            or receipt.get("rows_sha256") != sha256_file(path.parent / "rows.jsonl.gz")
        ):
            raise ValueError("primary validation receipt is incomplete or corrupt")
        rank = ponder_checkpoint_rank(
            update=update,
            cap=24,
            primary_successes=receipt["timed_successes"],
            negative_false_actions=receipt["negative_false_actions"],
        )
        ranked.append((rank, update))
        validation_receipts[update] = sha256_file(path)
    best_rank, best_update = max(ranked)
    if (
        best_rank != trained.selected_rank
        or best_update != trained.selected_update
        or trained.selected_checkpoint != f"main-{best_update:04d}.pt"
        or trained.selected_checkpoint_sha256
        != sha256_file(run_dir / "ponder" / trained.selected_checkpoint)
    ):
        raise ValueError("selected ponder checkpoint differs from primary-only rank")
    manifests = {
        name: ComparisonManifest.model_validate_json(
            (run_dir / "manifests" / f"{name}.json").read_bytes()
        )
        for name in ("iid", "depth")
    }
    if any(len(manifest.entries) != 512 for manifest in manifests.values()):
        raise ValueError("ponder comparison needs both complete 512-row corpora")
    corpora = {}
    paired = {}
    identities = {}
    for name, manifest in manifests.items():
        intact_identity, intact, _ = _read_condition(
            run_dir, condition="intact_eventflow", manifest=manifest
        )
        corpora[name] = {
            "intact_eventflow": _summary(
                intact,
                BudgetLedger.model_validate_json(
                    (run_dir / "conditions/intact_eventflow" / name / "budget.json").read_bytes()
                ),
                timing_indices(manifest),
            )
        }
        identities[f"{name}/intact_eventflow"] = intact_identity.sha256
        paired[name] = {}
        for cap in (4, 8, 12, 16, 24):
            identity, rows, budget = _read_condition(
                run_dir, condition="activation_ponder", manifest=manifest, cap=cap
            )
            if (
                identity.checkpoint_sha256 != trained.selected_checkpoint_sha256
                or identity.producing_source_revision != trained.source_revision
                or [(r.public_id, r.episode_sha256) for r in rows]
                != [(r.public_id, r.episode_sha256) for r in intact]
            ):
                raise ValueError("ponder rows are not paired to the selected checkpoint")
            key = f"activation_ponder_cap{cap}"
            corpora[name][key] = _summary(rows, budget, timing_indices(manifest))
            interval = _paired_interval(intact, rows)
            paired[name][key] = {
                "ponder_minus_intact": interval["compressed_minus_intact"],
                "interval_95": interval["interval_95"],
                "interval_label": interval["interval_label"],
                "bootstrap_replicates": interval["bootstrap_replicates"],
                "analysis_seed": interval["analysis_seed"],
            }
            identities[f"{name}/{key}"] = identity.sha256
    intact_macs = corpora["iid"]["intact_eventflow"]["mean_end_to_end_forward_macs"]
    ponder_macs = {
        cap: corpora["iid"][f"activation_ponder_cap{cap}"]["mean_end_to_end_forward_macs"]
        for cap in (4, 8, 12, 16, 24)
    }
    nearest = nearest_ponder_cap(intact_macs, ponder_macs)
    depth_overlap = (
        nearest is not None
        and abs(
            corpora["depth"][f"activation_ponder_cap{nearest}"]["mean_end_to_end_forward_macs"]
            - corpora["depth"]["intact_eventflow"]["mean_end_to_end_forward_macs"]
        )
        / corpora["depth"]["intact_eventflow"]["mean_end_to_end_forward_macs"]
        <= 0.10
    )
    report = {
        "schema_version": "phase5a-ponder-report-v1",
        "purpose": "exploratory_comparison",
        "gate_eligible": False,
        "complete_b": True,
        "training_status": "completed_exploratory",
        "last_training_stage": trained.last_stage,
        "completed_updates": trained.completed_updates,
        "selected_update": best_update,
        "selected_checkpoint_sha256": trained.selected_checkpoint_sha256,
        "selected_rank": best_rank,
        "parameters": trained.parameters,
        "entity_parameters": trained.entity_parameters,
        "profile_sha256": trained.profile_sha256,
        "admission_sha256": trained.admission_sha256,
        "validation_receipt_sha256": validation_receipts,
        "identities": identities,
        "corpora": corpora,
        "paired_intervals": paired,
        "nearest_iid_compute_cap": nearest,
        "nearest_iid_compute_overlap_within_10_percent": nearest is not None,
        "same_cap_depth_compute_overlap_within_10_percent": depth_overlap,
        "claim_boundary": (
            "Single-seed exploratory comparison only; no converged-baseline certification, "
            "final Phase 5 gate, or confirmatory scientific-support claim."
        ),
    }
    lines = [
        "# Phase 5A comparative pilot — Milestone B",
        "",
        report["claim_boundary"],
        "",
        f"Training updates: {trained.completed_updates}; last stage: {trained.last_stage}; "
        f"selected at update {best_update} by primary validation cap 24.",
        "",
        "| Corpus | Condition | Timed success | Errors | Mean forward MACs |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name in ("iid", "depth"):
        for key in (
            "intact_eventflow",
            *(f"activation_ponder_cap{cap}" for cap in (4, 8, 12, 16, 24)),
        ):
            item = corpora[name][key]
            lines.append(
                f"| {name} | {key} | {item['timed_successes']}/{item['episodes']} "
                f"| {item['errors']} | {item['mean_end_to_end_forward_macs']:.0f} |"
            )
    lines.extend(
        [
            "",
            "Nearest measured IID compute cap: "
            f"{nearest if nearest is not None else 'none within 10%'}.",
            f"Same cap overlaps on depth: {depth_overlap}.",
            "",
        ]
    )
    output_dir.mkdir(parents=True)
    atomic_create_bytes(output_dir / "report.json", canonical_json_bytes(report) + b"\n")
    atomic_create_bytes(output_dir / "report.md", "\n".join(lines).encode())
    return output_dir
