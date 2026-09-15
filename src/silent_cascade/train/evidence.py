"""Real source-bound component collection; raw decisions precede private scoring."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from pydantic import TypeAdapter

from silent_cascade.errors import ArtifactIntegrityError
from silent_cascade.eventflow.checkpoint_rng import restore_rng_snapshot
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.rng import seed_all, snapshot_global_rng
from silent_cascade.train.batches import next_training_batch
from silent_cascade.train.checkpoints import (
    _read,
    load_checkpoint_index,
    load_training_checkpoint,
    load_weight_bundle,
)
from silent_cascade.train.component_eval import (
    ComponentPredictions,
    evaluate_components,
    score_components,
)
from silent_cascade.train.curriculum_data import ComponentManifest
from silent_cascade.train.evidence_types import (
    MAX_ARTIFACT_BYTES,
    MAX_METADATA_BYTES,
    ComponentGateReport,
    ComponentRow,
    ComponentSummary,
    ComputeBatch,
    NumericEvidence,
    PredictionData,
    ReplaySample,
    TrainingEvidence,
    decode_json,
    read_bytes,
)
from silent_cascade.train.provenance import (
    authenticate_source,
    collect_provenance,
    publish,
    relative,
)
from silent_cascade.train.trainer import (
    TrainingRunResult,
    _json,
    _manifest_corpus,
    _verified_history,
)
from silent_cascade.train.verification import (
    measure_cpu_resume,
    measure_device_parity,
    measure_mps_resume,
    measure_offline_imports,
)


def canonical_json_bytes(value):
    return json.dumps(
        _json(value), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode()


def require(condition, message):
    if not condition:
        raise ArtifactIntegrityError(message)


def score_row(
    prediction, target, *, position, public_id, entry, model_hash, provenance, batch_index
):
    """Use the actual component contract; the independent verifier duplicates no call here."""
    metrics = score_components(ComponentPredictions((prediction,)), (target,))
    scored = metrics.rows[0]
    raw = asdict(prediction)
    projected = {name: raw[name] for name in PredictionData.model_fields}
    return ComponentRow.model_validate_json(
        canonical_json_bytes(
            dict(
                public_id=public_id,
                example_hash=entry.example_hash,
                position=position,
                variant=scored.category,
                target=asdict(target),
                prediction=projected,
                required_recall_count=len(target.record_ids),
                correct_recall_count=sum(scored.recall_correct),
                required_composition_count=len(target.content),
                correct_composition_count=sum(scored.composition_correct),
                complete_chain_correct=scored.chain_correct,
                action_correct=scored.action_correct,
                capped=prediction.stop_reason == "capped",
                invalid_prediction=prediction.stop_reason == "malformed",
                log_delay_absolute_errors=metrics.log_delay_absolute_errors,
                normalized_deadline_absolute_errors=metrics.normalized_deadline_absolute_errors,
                compute_batch=batch_index,
                model_state_sha256=model_hash,
                config_sha256=provenance.config_sha256,
                source_commit=provenance.source_commit,
                foundation_model_calls=prediction.foundation_model_calls,
            )
        )
    )


def summarize_rows(rows):
    n = len(rows)
    required_r = sum(r.required_recall_count for r in rows)
    required_c = sum(r.required_composition_count for r in rows)
    correct_r = sum(r.correct_recall_count for r in rows)
    correct_c = sum(r.correct_composition_count for r in rows)
    chains = sum(r.complete_chain_correct for r in rows)
    actions = sum(r.action_correct for r in rows)
    return ComponentSummary(
        episode_count=n,
        required_recall_count=required_r,
        correct_recall_count=correct_r,
        required_composition_count=required_c,
        correct_composition_count=correct_c,
        complete_chain_correct=chains,
        action_correct_count=actions,
        variant_counts={
            v: sum(r.variant == v for r in rows) for v in ("positive", "safe", "disconnected")
        },
        cap_count=sum(r.capped for r in rows),
        invalid_prediction_count=sum(r.invalid_prediction for r in rows),
        required_recall_accuracy=correct_r / required_r if required_r else 0.0,
        required_composition_accuracy=correct_c / required_c if required_c else 0.0,
        complete_chain_accuracy=chains / n if n else 0.0,
        action_accuracy=actions / n if n else 0.0,
        content_gates_pass=all(
            total > 0 and 100 * correct > 99 * total
            for correct, total in ((correct_r, required_r), (correct_c, required_c), (chains, n))
        ),
    )


def row_chain(rows):
    state = hashlib.sha256(b"silent-cascade/phase3/rows/v1\0").digest()
    for position, row in enumerate(rows):
        raw = canonical_json_bytes(row)
        state = hashlib.sha256(
            state + position.to_bytes(4, "big") + len(raw).to_bytes(8, "big") + raw
        ).digest()
    return state.hex()


def _bound(path, digest, limit=MAX_METADATA_BYTES):
    raw = read_bytes(path, limit)
    require(hashlib.sha256(raw).hexdigest() == digest, "training journal hash mismatch")
    return decode_json(raw, limit)


def _training_evidence(root, run, restored_weights, config, source, manifest_hash, corpus):
    require(
        run.is_absolute() and run == run.resolve(),
        "training run must be an explicit canonical real directory",
    )
    run_name = f"runs/{run.name}"  # Portable identity; physical location is an execution input.
    index = load_checkpoint_index(run)
    require(
        index.config_sha256 == config.sha256
        and index.source_commit == source
        and index.validation_manifest_sha256 == manifest_hash,
        "run index identity mismatch",
    )
    candidates = []
    for path in sorted(run.glob("attempt-*/result.json")):
        raw = read_bytes(path)
        decode_json(raw)
        result = TypeAdapter(TrainingRunResult).validate_json(raw)
        if result.latest == index.latest:
            candidates.append((path, raw, result))
    require(bool(candidates), "run lacks a completed result for its latest committed checkpoint")
    path, raw, result = candidates[-1]
    require(
        bool(index.best) and result.selected == index.best[0],
        "selected checkpoint differs from real index",
    )
    require(
        result.weights == restored_weights.descriptor, "export descriptor differs from selected run"
    )
    require(
        result.selected.model_state_sha256 == result.weights.model_state_sha256,
        "portable weights do not contain selected model state",
    )
    require(
        result.weights.optimizer_step == 0 and result.selected.optimizer_step > 0,
        "portable and selected training step identities disagree",
    )
    _, _, latest_meta, _ = _read(run / result.latest.relative_path, weights=False)
    require(
        latest_meta.progress == result.progress, "result progress differs from latest full archive"
    )
    selected = load_training_checkpoint(
        run / result.selected.relative_path,
        expected_config_sha256=config.sha256,
        expected_source_commit=source,
        device=result.device,
    )
    require(selected.descriptor == result.selected, "selected full archive descriptor mismatch")
    history = _verified_history(run, result.progress)
    require(
        list(result.validation_history) == history and bool(history),
        "run selection history mismatch",
    )
    best = max(history, key=lambda row: (row["chain"], row["composition"], -row["step"]))
    require(
        (best["step"], best["chain"], best["composition"])
        == (
            result.selected.optimizer_step,
            result.selected.validation_metric,
            result.selected.validation_composition_metric,
        ),
        "selected validation ranking mismatch",
    )
    cursor = result.progress.training_journal_sha256
    seen, gradient_max, step_count = set(), 0.0, 0
    measured_history = []
    step_counters, diagnostic_samples = [], []
    objective_step_counts = {}
    while cursor is not None:
        require(cursor not in seen and len(seen) <= 75000, "cyclic or oversized journal")
        seen.add(cursor)
        journal = _bound(run / f"journal-{cursor}.json", cursor)
        attempt = run / journal["attempt"]
        for entry in journal["steps"]:
            step = _bound(attempt / entry["path"], entry["sha256"])
            values = step["result"]
            require(
                values["objective_version"] == config.config.training.objective_version
                and values["auxiliary_coefficient"]
                == config.config.training.content_auxiliary_weight,
                "training objective identity mismatch",
            )
            objective_step_counts[values["objective_version"]] = (
                objective_step_counts.get(values["objective_version"], 0) + 1
            )
            step_counters.append(step["step"] - 1)
            if step["step"] <= 2:
                diagnostic_samples.append(
                    {
                        "step": step["step"],
                        "example_hashes": step["example_hashes"],
                        "result": values,
                    }
                )
            require(
                type(values["gradient_norm"]) is float and values["gradient_norm"] >= 0,
                "invalid measured training gradient norm",
            )
            require(
                type(values["compute"]["foundation_model_calls"]) is int
                and values["compute"]["foundation_model_calls"] == 0,
                "nonzero training foundation calls",
            )
            gradient_max = max(gradient_max, values["gradient_norm"])
            step_count += 1
        if journal["validation"] is not None:
            entry = journal["validation_file"]
            validation = _bound(attempt / entry["path"], entry["sha256"], MAX_ARTIFACT_BYTES)
            require(
                validation["targets"] == json.loads(canonical_json_bytes(corpus.targets)),
                "validation targets differ from fixed manifest",
            )
            predictions = TypeAdapter(ComponentPredictions).validate_json(
                canonical_json_bytes(
                    {
                        "rows": [
                            {
                                key: value
                                for key, value in row.items()
                                if key not in ("failed", "capped", "stopped")
                            }
                            for row in validation["predictions"]["rows"]
                        ],
                        "evaluation_mode": "unassisted_content",
                    }
                )
            )
            metrics = score_components(predictions, corpus.targets)
            require(
                metrics.to_primitive() == validation["metrics"],
                "stored validation arithmetic mismatch",
            )
            stored = journal["validation"]
            measured_history.append(
                {
                    **stored,
                    "episode_count": metrics.episode_count,
                    "complete_chain_correct": metrics.correct_chain_count,
                    "required_recall_count": metrics.required_recall_count,
                    "correct_recall_count": metrics.correct_recall_count,
                    "required_composition_count": metrics.required_composition_count,
                    "correct_composition_count": metrics.correct_composition_count,
                }
            )
            require(
                (stored["chain"], stored["composition"], stored["recall"], stored["gates_pass"])
                == (
                    metrics.complete_chain_accuracy,
                    metrics.required_composition_accuracy,
                    metrics.required_recall_accuracy,
                    metrics.gates_pass,
                ),
                "journal validation summary mismatch",
            )
        cursor = journal["prior"]
    require(step_count == result.progress.optimizer_step, "training steps missing from journal")
    return TrainingEvidence.model_validate_json(
        canonical_json_bytes(
            dict(
                objective_version=config.config.training.objective_version,
                auxiliary_coefficient=config.config.training.content_auxiliary_weight,
                objective_step_counts=objective_step_counts,
                step_counters=sorted(step_counters),
                diagnostic_samples=sorted(diagnostic_samples, key=lambda row: row["step"]),
                run_path=run_name,
                result_path=relative(run, path),
                result_sha256=hashlib.sha256(raw).hexdigest(),
                index_sha256=hashlib.sha256(read_bytes(run / "checkpoint-index.json")).hexdigest(),
                journal_sha256=result.progress.training_journal_sha256,
                latest=result.latest,
                selected=result.selected,
                weights=result.weights,
                step_count=step_count,
                gradient_norm_max=gradient_max,
                next_batch_counter=result.progress.next_batch_counter,
                stop_reason=result.stop_reason,
                device=result.device,
                validation_history=tuple(sorted(measured_history, key=lambda row: row["step"])),
                patience_counter=result.progress.patience_counter,
                validation_manifest_sha256=manifest_hash,
            )
        )
    )


def collect_component_gate(
    config,
    *,
    manifest_path: Path,
    weights_path: Path,
    expected_checkpoint_sha256: str,
    training_run: Path,
    source_commit: str,
    plan_revision: str,
    output_path: Path,
) -> ComponentGateReport:
    """Collect real decisions and measurements; publication can honestly report failure."""
    root = Path(__file__).resolve().parents[3]
    provenance = collect_provenance(config, root, manifest_path, source_commit, plan_revision)
    require(
        weights_path.parent == training_run and weights_path.name == Path(weights_path.name).name,
        "weights must belong to the explicitly supplied run",
    )
    restored = load_weight_bundle(
        weights_path, expected_sha256=expected_checkpoint_sha256, device="cpu"
    )
    require(
        restored.config.sha256 == config.sha256
        and restored.descriptor.source_commit == source_commit,
        "weights source/configuration mismatch",
    )
    raw_manifest = read_bytes(manifest_path, MAX_ARTIFACT_BYTES)
    decode_json(raw_manifest, MAX_ARTIFACT_BYTES)
    manifest = ComponentManifest.model_validate_json(raw_manifest)
    require(manifest.plan_revision == plan_revision, "manifest plan revision mismatch")
    corpus, manifest_hash = _manifest_corpus(config, manifest_path, source_commit)
    require(
        manifest_hash == provenance.validation_manifest_sha256, "manifest changed during collection"
    )
    training = _training_evidence(
        root, training_run, restored, config, source_commit, manifest_hash, corpus
    )
    evaluation = evaluate_components(
        restored.model, corpus, batch_size=config.config.training.batch_size
    )
    # Full decisions already exist before any private target is used below.
    rows = tuple(
        score_row(
            prediction,
            target,
            position=i,
            public_id=public.init.episode_public_id,
            entry=entry,
            model_hash=restored.descriptor.model_state_sha256,
            provenance=provenance,
            batch_index=i // config.config.training.batch_size,
        )
        for i, (prediction, target, public, entry) in enumerate(
            zip(
                evaluation.predictions.rows,
                corpus.targets,
                corpus.public_examples,
                manifest.entries,
                strict=True,
            )
        )
    )
    summary = summarize_rows(rows)
    expected_counts = (
        {"positive": 5000, "safe": 2500, "disconnected": 2500}
        if manifest.publication == "production"
        else {"positive": 8, "safe": 4, "disconnected": 4}
    )
    require(summary.variant_counts == expected_counts, "fixed corpus variant counts mismatch")
    repeated = evaluate_components(
        restored.model, corpus, batch_size=config.config.training.batch_size
    )
    mismatches = sum(
        canonical_json_bytes(a) != canonical_json_bytes(b)
        for a, b in zip(evaluation.predictions.rows, repeated.predictions.rows, strict=True)
    )
    samples = tuple(
        ReplaySample(
            position=i,
            variant=variant,
            output_sha256=hashlib.sha256(canonical_json_bytes(rows[i].prediction)).hexdigest(),
        )
        for variant in ("positive", "safe", "disconnected")
        for i in [next(i for i, row in enumerate(rows) if row.variant == variant)]
    )
    before = snapshot_global_rng()
    try:
        seed_all(11)
        model = EventFlowModel(config.config.neural)
        batch = next_training_batch(config.config, stage="one_hop", batch_counter=0)
        parity = measure_device_parity(model, batch, training=config.config.training)
        cpu_resume = measure_cpu_resume(config, source_commit)
        mps_resume = measure_mps_resume(config, source_commit)
    finally:
        restore_rng_snapshot(before, restore_mps=before.torch_mps_state is not None)
    compute = tuple(
        ComputeBatch.model_validate_json(
            canonical_json_bytes(
                dict(
                    start=i * config.config.training.batch_size,
                    count=min(
                        config.config.training.batch_size,
                        len(rows) - i * config.config.training.batch_size,
                    ),
                    **_json(snapshot),
                )
            )
        )
        for i, snapshot in enumerate(evaluation.compute)
    )
    passed = (
        manifest.publication == "production"
        and summary.content_gates_pass
        and summary.cap_count == summary.invalid_prediction_count == mismatches == 0
        and parity.passed
        and cpu_resume.passed
        and mps_resume.passed
    )
    report = ComponentGateReport(
        objective_version=config.config.training.objective_version,
        auxiliary_coefficient=config.config.training.content_auxiliary_weight,
        publication=manifest.publication,
        provenance=provenance,
        training=training,
        rows=rows,
        rows_sha256=row_chain(rows),
        summary=summary,
        compute=compute,
        numeric=NumericEvidence.model_validate_json(
            canonical_json_bytes(dict(parity=parity, cpu_resume=cpu_resume, mps_resume=mps_resume))
        ),
        replay_samples=samples,
        repeated_prediction_count=len(rows),
        repeated_prediction_mismatches=mismatches,
        offline=measure_offline_imports(config.config.training),
        passed=passed,
    )
    metadata = report.model_dump(mode="json", exclude={"rows", "compute"})
    require(
        len(canonical_json_bytes(metadata)) <= MAX_METADATA_BYTES,
        "gate metadata byte limit exceeded",
    )
    authenticate_source(root, source_commit, plan_revision)
    publish(output_path, report)
    return report
