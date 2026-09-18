"""Regenerate pilot tables and figures solely from verified retained artifacts."""

import io
from collections import Counter
from pathlib import Path

from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.report.pilot_artifacts import (
    evaluation_directories,
    load_abandoned_evaluation,
    load_training_result,
    training_evaluation_status,
)
from silent_cascade.train.pilot_data import _check_path, _publish_pilot_bytes


def _figures(examples, errors):
    import matplotlib
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 8), layout="constrained")
    axes = figure.subplots(2, 2)
    for axis, positive in zip(axes[0], (True, False), strict=True):
        selected = examples.get(positive)
        axis.set_title("Positive timeline" if positive else "Negative timeline")
        axis.set_xlabel("time since activation (seconds)")
        if selected is not None:
            row = selected.row
            events = selected.sidecar["events"]
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
    axes[1, 0].set_title("Action timing error (all positive actions)")
    axes[1, 0].set_xlabel("action time minus target (seconds)")
    if errors:
        axes[1, 0].hist(errors, bins=min(20, len(errors)))
    else:
        axes[1, 0].text(0.05, 0.5, "No positive actions", transform=axes[1, 0].transAxes)
    axes[1, 1].set_title("Retained positive guard accumulators (anchors)")
    selected = examples.get(True)
    if selected is not None:
        row = selected.row
        anchors = selected.trajectory["anchors"]
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


def build_pilot_report(*, run_dir: Path, output_dir: Path, evidence_context=None) -> Path:
    """Verify every raw input before publishing any artifact-derived report."""
    _check_path(run_dir)
    _check_path(output_dir)
    from silent_cascade.archive.readers import evaluation_header, iter_evaluation_episodes

    evaluations = evaluation_directories(run_dir, evidence_context=evidence_context)
    training = None
    result_path = run_dir / "training-result.json"
    if result_path.exists():
        training = load_training_result(run_dir, result_path, evidence_context=evidence_context)
    required, abandoned = (
        training_evaluation_status(run_dir, training) if training is not None else (set(), set())
    )
    if not required <= set(evaluations):
        raise ValueError("missing committed evaluation corpus")
    incomplete = []
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
        if not (root / "DONE").exists() and training is not None and root not in required:
            partial = load_abandoned_evaluation(
                root, run_dir=run_dir, training=training, abandoned=abandoned
            )
            incomplete.append(partial)
            lines.extend(
                [
                    f"## {partial['corpus']} — abandoned incomplete",
                    "",
                    f"Retained rows: {partial['retained_rows']} / "
                    f"{partial['planned_episodes']}; retained errors: "
                    f"{partial['retained_errors']}; unknown episodes: "
                    f"{partial['unknown_episodes']}.",
                    "Partial evidence is excluded from completed-corpus metrics; "
                    "unknown episodes have no inferred outcome. All raw files remain retained.",
                    "",
                ]
            )
            continue
        with evaluation_header(root, evidence_context=evidence_context) as (header, _):
            pass
        identity, metrics, hashes = header.identity, header.metrics, header.hashes
        identities[str(root.relative_to(run_dir))] = identity.sha256
        variants, classes, events, misses, action_diagnostics = (
            {},
            {},
            Counter(),
            Counter(),
            Counter(),
        )
        compute_fields = (
            "forward_macs",
            "backward_macs",
            "records_scored",
            "flow_evaluations",
            "jump_applications",
            "foundation_model_calls",
        )
        compute, examples, action_errors = Counter(), {}, []
        event_total = 0
        for episode in iter_evaluation_episodes(
            root, evidence_context=evidence_context, expected_header=header, plot_examples=True
        ):
            row = episode.row
            event_total += row.event_count
            compute.update({field: getattr(row.compute, field) for field in compute_fields})
            if row.is_positive:
                action_errors.extend(
                    action.timestamp - row.truth.action_target for action in row.actions
                )
            if episode.trajectory is not None:
                examples[row.is_positive] = episode
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
            observed = episode.sidecar["events"]
            for event in observed:
                if event["raw_action_argmax"] is not None:
                    action_diagnostics["raw_class_" + str(event["raw_action_argmax"])] += 1
                    action_diagnostics["legal_class_" + str(event["legal_action_argmax"])] += 1
                    action_diagnostics["status_disagreements"] += event["status_disagrees"]
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
            or sum(events.values()) != event_total
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
                f"`{sha256_bytes(identity.evaluation_config_canonical_json.encode())}`; "
                f"manifest: `{identity.manifest_sha256}`; "
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
        plots[name] = _figures(examples, action_errors)
        lines.extend([f"![Retained trajectories and timing diagnostics]({name})", ""])
    table_bytes = canonical_json_bytes(
        {"evaluations": tables, "incomplete_evaluations": incomplete}
    )
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
