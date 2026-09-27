"""Artifact-only exploratory Phase 5A report with strict paired coverage checks."""

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from silent_cascade.eval.comparison_protocol import decide_after_a, timing_indices
from silent_cascade.eval.comparison_types import (
    BudgetLedger,
    ComparisonIdentity,
    ComparisonManifest,
    ComparisonRow,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from silent_cascade.io import atomic_create_bytes


def _read_condition(
    run_dir: Path, *, condition: str, manifest: ComparisonManifest
) -> tuple[ComparisonIdentity, list[ComparisonRow], BudgetLedger]:
    root = run_dir / "conditions" / condition / manifest.name
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
