"""Regenerate pilot tables and figures solely from verified retained artifacts."""

import gzip
import io
import json
from collections import Counter
from pathlib import Path

from silent_cascade.eval.artifacts import read_evaluation_artifact
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.report.pilot_artifacts import (
    evaluation_directories,
    load_evaluation,
    load_training_result,
    read_json,
)
from silent_cascade.train.pilot_data import _check_path, _publish_pilot_bytes


def _figures(root, rows):
    import matplotlib
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 8), layout="constrained")
    axes = figure.subplots(2, 2)
    for axis, positive in zip(axes[0], (True, False), strict=True):
        candidates = [r for r in rows if r.is_positive == positive and r.full_trace_ref]
        axis.set_title("Positive timeline" if positive else "Negative timeline")
        axis.set_xlabel("time since activation (seconds)")
        if candidates:
            row = candidates[0]
            events = read_json(root / row.neural_trace_ref)["events"]
            kinds = sorted({event["kind"] for event in events})
            for kind in kinds:
                axis.scatter(
                    [
                        e["timestamp"] - row.truth.activation_time
                        for e in events
                        if e["kind"] == kind
                    ],
                    [kind] * sum(e["kind"] == kind for e in events),
                    s=12,
                )
            if positive:
                axis.axvspan(
                    row.truth.action_window_start - row.truth.activation_time,
                    row.truth.action_window_end - row.truth.activation_time,
                    alpha=0.15,
                    color="green",
                )
        else:
            axis.text(0.05, 0.5, "No retained example in this corpus", transform=axis.transAxes)
    errors = [a.timestamp - r.truth.action_target for r in rows if r.is_positive for a in r.actions]
    axes[1, 0].set_title("Action timing error (all positive actions)")
    axes[1, 0].set_xlabel("action time minus target (seconds)")
    if errors:
        axes[1, 0].hist(errors, bins=min(20, len(errors)))
    else:
        axes[1, 0].text(0.05, 0.5, "No positive actions", transform=axes[1, 0].transAxes)
    axes[1, 1].set_title("Retained positive guard accumulators (anchors)")
    positives = [r for r in rows if r.is_positive and r.full_trace_ref]
    if positives:
        row = positives[0]
        anchors = json.loads(gzip.decompress(read_evaluation_artifact(root / row.full_trace_ref)))[
            "anchors"
        ]
        for index, kind in enumerate(("RECALL", "COMPOSE", "ACT")):
            axes[1, 1].plot(
                [anchor["time"] - row.truth.activation_time for anchor in anchors],
                [
                    anchor["core"]["continuous"]["guard_accumulators"]["values"][index]
                    for anchor in anchors
                ],
                marker=".",
                label=kind,
            )
        axes[1, 1].axhline(1, color="black", linewidth=0.5)
        axes[1, 1].legend()
    axes[1, 1].set_xlabel("time since activation (seconds)")
    target = io.BytesIO()
    with matplotlib.rc_context({"svg.hashsalt": "silent-cascade-pilot-v1"}):
        FigureCanvasSVG(figure).print_svg(target, metadata={"Date": None})
    return target.getvalue()


def build_pilot_report(*, run_dir: Path, output_dir: Path) -> Path:
    """Verify every raw input before publishing any artifact-derived report."""
    _check_path(run_dir)
    _check_path(output_dir)
    evaluations = evaluation_directories(run_dir)
    training = None
    result_path = run_dir / "training-result.json"
    if result_path.exists():
        training = load_training_result(run_dir, result_path)
    tables, plots, identities = [], {}, {}
    lines = [
        "# Autonomous timed pilot",
        "",
        "Pilot validation evidence; training-exposed selection. No baseline, continuity, "
        "novelty, frozen-test, or final Phase 4 gate claim.",
        "",
        "Conclusion: production learning gate not established by this report. "
        "A completed execution is distinct from a passed scientific gate.",
        "",
    ]
    if training is not None:
        lines.extend(
            [
                f"Training status: {training['status']}; updates: "
                f"{training['progress']['global_step']}; seed: 11.",
                "Weights: "
                + (
                    "eligible selected"
                    if training["selected_weights"]
                    else "latest diagnostic; no eligible selected weights"
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Training exposure and selection are not supplied; fixture-only or standalone "
                "evaluation evidence cannot certify training.",
                "",
            ]
        )
    for number, root in enumerate(evaluations):
        identity, rows, metrics, hashes = load_evaluation(root)
        identities[str(root.relative_to(run_dir))] = identity.sha256
        variants, classes, events, misses, action_diagnostics = (
            {},
            {},
            Counter(),
            Counter(),
            Counter(),
        )
        for row in rows:
            variant = row.truth.recipe.variant.value
            counts = variants.setdefault(variant, dict(episodes=0, timed_success=0, errors=0))
            counts["episodes"] += 1
            counts["timed_success"] += row.timed_success
            counts["errors"] += row.error is not None
            label = str(row.truth.relevant_hazard_type) if row.is_positive else "negative"
            counts = classes.setdefault(label, dict(episodes=0, correct_class=0, timed_success=0))
            counts["episodes"] += 1
            counts["correct_class"] += row.score.correct_class is True
            counts["timed_success"] += row.timed_success
            events.update(dict(row.event_counts))
            misses.update([row.miss_category or "success"])
            observed = read_json(root / row.neural_trace_ref)["events"]
            for event in observed:
                if event["raw_action_argmax"] is not None:
                    action_diagnostics["raw_class_" + str(event["raw_action_argmax"])] += 1
                    action_diagnostics["legal_class_" + str(event["legal_action_argmax"])] += 1
                    action_diagnostics["status_disagreements"] += event["status_disagrees"]
        compute_fields = (
            "forward_macs",
            "backward_macs",
            "records_scored",
            "flow_evaluations",
            "jump_applications",
            "foundation_model_calls",
        )
        compute = {
            field: sum(getattr(row.compute, field) for row in rows) for field in compute_fields
        }
        table = dict(
            metrics.model_dump(mode="json"),
            variants=variants,
            classes=classes,
            events=dict(events),
            misses=dict(misses),
            compute=compute,
            action_diagnostics=dict(action_diagnostics),
            identity_sha256=identity.sha256,
            manifest_sha256=identity.manifest_sha256,
            checkpoint_sha256=identity.checkpoint_sha256,
            source_revision=identity.execution_source_revision,
            raw_rows=str(root.relative_to(run_dir) / "rows.jsonl"),
            rows_sha256=hashes["rows.jsonl"],
        )
        if (
            sum(v["episodes"] for v in variants.values()) != metrics.episode_count
            or sum(v["timed_success"] for v in variants.values()) != metrics.timed_success_count
            or sum(events.values()) != sum(row.event_count for row in rows)
            or sum(misses.values()) != metrics.episode_count
        ):
            raise ValueError("report table denominator mismatch")
        tables.append(table)
        lines.extend(
            [
                f"## {root.relative_to(run_dir)}",
                "",
                f"Experiment: {identity.experiment}; "
                f"purpose: {identity.purpose}; split: {identity.split}; stage: {identity.stage}.",
                f"Source: `{identity.execution_source_revision}`; config: "
                f"`{rows[0].config_sha256}`; manifest: `{identity.manifest_sha256}`; "
                f"checkpoint: `{identity.checkpoint_sha256}`; model seed: 11.",
                "",
                "| Outcome | Count / denominator |",
                "| --- | ---: |",
                f"| Timed success | {metrics.timed_success_count} / {metrics.episode_count} |",
                f"| Negative false action | {metrics.false_action_count} / "
                f"{metrics.negative_count} |",
                f"| Errors | {metrics.error_count} / {metrics.episode_count} |",
                "",
            ]
        )
        for title, groups in (("Variant", variants), ("Class", classes)):
            lines.extend([f"| {title} | Episodes | Timed success |", "| --- | ---: | ---: |"])
            lines.extend(
                f"| {name} | {counts['episodes']} | {counts['timed_success']} |"
                for name, counts in sorted(groups.items())
            )
            lines.append("")
        for title, counts in (
            ("Event counts", events),
            ("Miss categories", misses),
            ("Compute", compute),
            ("Action diagnostics", action_diagnostics),
        ):
            lines.extend([f"| {title} | Count |", "| --- | ---: |"])
            lines.extend(f"| {name} | {count} |" for name, count in sorted(counts.items()))
            lines.append("")
        name = f"trajectories-{number}.svg"
        plots[name] = _figures(root, rows)
        lines.extend([f"![Retained trajectories and timing diagnostics]({name})", ""])
    table_bytes = canonical_json_bytes({"evaluations": tables})
    report_bytes = ("\n".join(lines) + "\n").encode()
    outputs = {**plots, "tables.json": table_bytes, "report.md": report_bytes}
    for name, raw in outputs.items():
        _publish_pilot_bytes(output_dir / name, raw)
    _publish_pilot_bytes(
        output_dir / "report-index.json",
        canonical_json_bytes(
            {
                "artifact_hashes": {name: sha256_bytes(raw) for name, raw in outputs.items()},
                "evaluations": identities,
            }
        ),
    )
    return output_dir / "report.md"
