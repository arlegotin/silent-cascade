"""Measured full trainer/scalar-runner workload; no training or acceptance credit."""

import math
import os
import platform
import resource
import shutil
import statistics
import sys
from collections import Counter
from copy import deepcopy

import torch

from silent_cascade.eventflow.neural import NeuralModelIdentity
from silent_cascade.hashing import sha256_bytes
from silent_cascade.train.pilot_trainer import pilot_train_one_step, publish_json
from silent_cascade.train.trainer import make_optimizer


def process_highwater_bytes():
    """Process lifetime RSS high-water, not per-operation or tensor working memory."""
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError):
        return None
    return int(value if sys.platform == "darwin" else value * 1024)


def model_hash(model):
    return NeuralModelIdentity.from_model(model, source_revision="0" * 40).model_state_sha256


def measure_training_trial(
    config,
    *,
    model,
    stage,
    device,
    output_dir,
    warmup_updates=16,
    measured_updates=32,
    source_revision="0" * 40,
):
    from silent_cascade.eventflow.neural_weights import save_neural_weights
    from silent_cascade.train.pilot_verification import (
        TrainingTrial,
        diagnostic_batch,
        measure_operation,
    )

    if not 1 <= warmup_updates <= 16 or not 1 <= measured_updates <= 32:
        raise ValueError("diagnostic trial exceeds the bounded update recipe")
    model = deepcopy(model).to(device)
    initial = model_hash(model)
    optimizer = make_optimizer(model, config.config.pilot)
    durations, rows = [], []
    for counter in range(warmup_updates + measured_updates):

        def operation(counter=counter):
            batch = diagnostic_batch(config.config, stage=stage, counter=counter).to(device)
            step = pilot_train_one_step(model, optimizer, batch, config.config)
            return batch, step

        (batch, step), elapsed = measure_operation(operation, device=device)
        if counter >= warmup_updates:
            durations.append(elapsed)
        row = {
            "diagnostic_update": counter + 1,
            "warmup": counter < warmup_updates,
            "elapsed_seconds": elapsed,
            "example_keys": batch.example_keys,
            "example_hashes": batch.example_hashes,
            "trace_steps": batch.targets.kind.shape[1],
            "step": step,
        }
        publish_json(output_dir / f"update-{counter:03d}.json", row)
        rows.append(step)
    final_weights_sha = save_neural_weights(
        output_dir / "final-weights.safetensors",
        model=model,
        identity=NeuralModelIdentity.from_model(model, source_revision=source_revision),
    )
    resources = {
        "process_highwater_bytes": process_highwater_bytes(),
        "mps_driver_allocation_bytes": torch.mps.driver_allocated_memory()
        if device == "mps"
        else None,
        "final_weights_sha256": final_weights_sha,
    }
    publish_json(output_dir / "resources.json", resources)
    result = TrainingTrial(
        device=device,
        stage=stage,
        batch_size=config.config.pilot.batch_size,
        max_trace_steps=batch.targets.kind.shape[1],
        warmup_updates=warmup_updates,
        measured_updates=measured_updates,
        update_seconds=tuple(durations),
        raw_step_rows=len(rows),
        initial_model_sha256=initial,
        final_model_sha256=model_hash(model),
        parameter_count=sum(p.numel() for p in model.parameters()),
        total_forward_macs=sum(r.compute.forward_macs for r in rows),
        total_backward_macs=sum(r.compute.backward_macs for r in rows),
        tensor_working_memory_bytes=max(r.compute.memory_bytes for r in rows),
        process_highwater_bytes=resources["process_highwater_bytes"],
        mps_driver_allocation_bytes=resources["mps_driver_allocation_bytes"],
        foundation_model_calls=sum(r.compute.foundation_model_calls for r in rows),
    )
    publish_json(output_dir / "trial.json", result)
    return result


def forecast_workload(
    *,
    update_seconds,
    evaluation_seconds_per_episode,
    component_seconds_per_episode,
    retained_bytes_per_episode,
    checkpoint_bytes,
    manifest_bytes,
    free_bytes,
    journal_bytes_per_update=0,
    component_bytes_per_episode=0,
    diagnostic_retained_bytes=0,
):
    from silent_cascade.train.pilot_verification import WorkloadForecast

    for low, high in (update_seconds, evaluation_seconds_per_episode, retained_bytes_per_episode):
        if not 0 < low <= high or not math.isfinite(high):
            raise ValueError("forecast requires positive finite measured ranges")
    if (
        component_seconds_per_episode <= 0
        or checkpoint_bytes <= 0
        or manifest_bytes <= 0
        or free_bytes < 0
    ):
        raise ValueError("invalid measured resource inputs")
    # Coverage estimate only: six checkpoint locations x up to six modes x three
    # variants x two exact replays. Legal overlap makes twelve an optimistic floor.
    final = 40000 + 512 + 128
    audit_context = 4 * 6376.053914
    lower = (
        4000 * update_seconds[0]
        + (50000 + final + 12) * evaluation_seconds_per_episode[0]
        + 10000 * component_seconds_per_episode
        + audit_context
    )
    upper = (
        75000 * update_seconds[1]
        + (1500000 + final + 216) * evaluation_seconds_per_episode[1]
        + 750000 * component_seconds_per_episode
        + audit_context
    )
    # All failure trajectories retained. Per-episode bounds are scenario inputs,
    # never a storage cap or permission to discard larger/adverse observations.
    if any(
        type(n) is not int or n < 0
        for n in (journal_bytes_per_update, component_bytes_per_episode, diagnostic_retained_bytes)
    ):
        raise ValueError("retention measurements must be nonnegative byte counts")
    retention_components = {
        "autonomous_and_final_rows": math.ceil(
            (1500000 + final + 216) * retained_bytes_per_episode[1]
        ),
        "training_journals": 75000 * journal_bytes_per_update,
        "component_validations": 750000 * component_bytes_per_episode,
        "four_manifests": 4 * manifest_bytes,
        "audit_artifact_allowance": 4 * manifest_bytes,
        "retained_diagnostics": diagnostic_retained_bytes,
    }
    retained_low = math.ceil(
        (50000 + final + 12) * retained_bytes_per_episode[0]
        + 8 * manifest_bytes
        + 4000 * journal_bytes_per_update
        + 10000 * component_bytes_per_episode
        + diagnostic_retained_bytes
    )
    retained_high = sum(retention_components.values())
    working = 6 * checkpoint_bytes  # best3 + latest1 + pending archive + portable export
    required = math.ceil(1.2 * retained_high) + working
    return WorkloadForecast(
        audit_context_seconds=audit_context,
        compute_seconds_lower=lower,
        compute_seconds_upper=upper,
        projected_retained_bytes_lower=retained_low,
        projected_retained_bytes_upper=retained_high,
        checkpoint_working_bytes=working,
        required_free_bytes=required,
        observed_free_bytes=free_bytes,
        space_sufficient=free_bytes > required,
        assumptions=(
            "Optimistic successful path: four stages with one global 1000-step boundary each; "
            "no convergence promise.",
            "Upper envelope: 75 boundaries, dual autonomous checks plus one-hop component work; "
            "maxima need not coincide.",
            "Final work: primary twice, two-hop, robustness (40000); 256 delay pairs (512); "
            "64 episodes on each device (128).",
            "Replay estimate: 12-216 complete-episode equivalents from six locations, six modes, "
            "three variants, two replays; Task11 must establish actual coverage.",
            "Four leakage audits use historical one-hop 6376.053914s each as context, "
            "not measurements of the other stages.",
            "Compute-only rates depend on observed cascade/error distribution; future competent "
            "cascades and artifacts may cost more.",
            "Retention keeps every failure; remeasure headroom before production. "
            "Checkpoint working space is added after the 1.2 safety factor.",
            "Audit artifact storage allowance is four measured primary-manifest sizes, "
            "an explicit estimate pending actual four-audit output sizes.",
        ),
        retention_components=retention_components,
    )


def _evaluation_trial(
    config, *, model, identity, weights_sha, manifest, subset, device, output_dir
):
    from silent_cascade.report.pilot_artifacts import load_evaluation
    from silent_cascade.train.component_eval import evaluate_components
    from silent_cascade.train.curriculum_data import ComponentCorpus
    from silent_cascade.train.pilot_verification import (
        EvaluationTrial,
        _evaluate_subset,
        diagnostic_batch,
        measure_operation,
    )
    from silent_cascade.train.traces import build_teacher_trace, component_target

    def operation():
        return _evaluate_subset(
            config,
            model=model,
            identity=identity,
            weights_sha=weights_sha,
            manifest=manifest,
            subset=subset,
            device=device,
            output_dir=output_dir / "autonomous",
        )

    _, elapsed = measure_operation(operation, device=device)
    _, rows, _, _ = load_evaluation(output_dir / "autonomous")
    events, misses = Counter(), Counter()
    for row in rows:
        events.update(dict(row.event_counts))
        misses.update([row.miss_category or "success"])
    files = list((output_dir / "autonomous").rglob("*"))
    byte_count = sum(p.stat().st_size for p in files if p.is_file())
    per_episode = [
        sum(p.stat().st_size for p in (output_dir / "autonomous/episodes").glob(f"{index:05d}.*"))
        for index in range(len(rows))
    ]
    examples = tuple(
        e
        for counter in range(2)
        for e in diagnostic_batch(config.config, stage="one_hop", counter=counter).examples
    )
    corpus = ComponentCorpus(
        tuple(e.public for e in examples),
        tuple(component_target(build_teacher_trace(e)) for e in examples),
    )
    component_model = deepcopy(model).to(device)

    def components():
        result = evaluate_components(
            component_model, corpus, batch_size=config.config.pilot.batch_size
        )
        publish_json(
            output_dir / "components.json",
            {"example_hashes": tuple(e.example_hash for e in examples), "result": result},
        )
        return result

    component, component_seconds = measure_operation(components, device=device)
    observations = {
        "elapsed_seconds": elapsed,
        "component_elapsed_seconds": component_seconds,
        "process_highwater_bytes": process_highwater_bytes(),
    }
    publish_json(output_dir / "observations.json", observations)
    result = EvaluationTrial(
        device=device,
        episodes=len(rows),
        elapsed_seconds=elapsed,
        mean_episode_seconds=elapsed / len(rows),
        event_counts=dict(events),
        miss_counts=dict(misses),
        error_episodes=sum(r.error is not None for r in rows),
        forward_macs=sum(r.compute.forward_macs for r in rows),
        foundation_model_calls=sum(r.compute.foundation_model_calls for r in rows)
        + sum(c.foundation_model_calls for c in component.compute),
        retained_raw_bytes=byte_count,
        largest_episode_bytes=max(per_episode),
        process_highwater_bytes=observations["process_highwater_bytes"],
        tensor_working_memory_bytes=max(r.compute.memory_bytes for r in rows),
        component_episodes=len(examples),
        component_elapsed_seconds=component_seconds,
        component_raw_bytes=(output_dir / "components.json").stat().st_size,
    )
    publish_json(output_dir / "trial.json", result)
    return result


def _profile_projection(training, evaluation, numeric, output_dir, *, free_bytes):
    workload_scores = {
        device: 75000
        * max(
            statistics.mean(training[f"{device}/{stage}"].update_seconds)
            for stage in ("one_hop", "primary")
        )
        + 1500000 * trial.mean_episode_seconds
        + 750000 * trial.component_elapsed_seconds / trial.component_episodes
        for device, trial in evaluation.items()
    }
    chosen = (
        "mps"
        if numeric.device_checks_passed and workload_scores["mps"] < workload_scores["cpu"]
        else "cpu"
    )
    reason = (
        "MPS full workload projection is faster and all numerical checks pass"
        if chosen == "mps"
        else (
            "CPU retained: native MPS evidence is missing"
            if numeric.missing_devices
            else "CPU retained: native MPS numerical checks failed"
            if not numeric.device_checks_passed
            else "CPU full workload projection is no slower"
        )
    )
    updates = [
        statistics.mean(training[f"{chosen}/{s}"].update_seconds) for s in ("one_hop", "primary")
    ]
    measured = evaluation[chosen]
    events = sum(measured.event_counts.values()) / measured.episodes
    ordinary = (
        sum(measured.event_counts.get(n, 0) for n in ("fact", "activate", "terminal"))
        / measured.episodes
    )
    expansion = max(1.0, (64 + ordinary) / max(1.0, events))
    cpu_rate = evaluation["cpu"].mean_episode_seconds
    retained = measured.retained_raw_bytes / measured.episodes
    checkpoint_bytes = (output_dir / "numerics/resume-cpu/resume.safetensors").stat().st_size
    forecast = forecast_workload(
        update_seconds=(min(updates), max(updates)),
        evaluation_seconds_per_episode=(
            min(measured.mean_episode_seconds, cpu_rate),
            max(measured.mean_episode_seconds, cpu_rate) * expansion,
        ),
        component_seconds_per_episode=max(
            e.component_elapsed_seconds / e.component_episodes for e in evaluation.values()
        ),
        retained_bytes_per_episode=(
            retained,
            max(retained, measured.largest_episode_bytes) * expansion,
        ),
        checkpoint_bytes=checkpoint_bytes,
        manifest_bytes=(output_dir / "numerics/manifest.json").stat().st_size,
        free_bytes=free_bytes,
        journal_bytes_per_update=max(
            p.stat().st_size for p in output_dir.glob("*/*/update-*.json")
        ),
        component_bytes_per_episode=math.ceil(
            max(e.component_raw_bytes / e.component_episodes for e in evaluation.values())
        ),
        diagnostic_retained_bytes=sum(
            p.stat().st_size
            for p in output_dir.rglob("*")
            if p.is_file() and p not in (output_dir / "DONE", output_dir / "throughput-report.json")
        ),
    )
    return chosen, reason, checkpoint_bytes, forecast


def verify_profile_raw(report, output_dir, *, config, source_commit):
    """Rebuild reusable summaries from required saved work, without running a model."""
    from dataclasses import asdict, fields

    from silent_cascade.eventflow.neural_weights import load_neural_weights
    from silent_cascade.report.pilot_artifacts import load_evaluation, read_json
    from silent_cascade.train import pilot_verification as verification
    from silent_cascade.train.pilot_data import PilotManifest, _read_pilot_bytes
    from silent_cascade.train.trainer import StepResult

    numeric = verification.read_numeric_report(
        output_dir / "numerics", config=config, source_commit=source_commit
    )
    if (
        report.numeric_report_sha256
        != sha256_bytes(_read_pilot_bytes(output_dir / "numerics/numeric-report.json"))
        or numeric.source != report.source
        or numeric.initial_model_sha256 != report.initial_model_sha256
        or numeric.portable_weights_sha256 != report.portable_weights_sha256
        or numeric.missing_devices != report.missing_devices
        or numeric.device_checks_passed != report.numerical_checks_passed
    ):
        raise ValueError("raw numeric profile binding differs")
    manifest = PilotManifest.model_validate_json(
        _read_pilot_bytes(output_dir / "numerics/manifest.json")
    )
    report.manifest_subset.validate_manifest(manifest)
    if read_json(output_dir / "profile-subset.json") != report.manifest_subset.model_dump(
        mode="json"
    ):
        raise ValueError("raw profile subset differs")
    weights = load_neural_weights(
        output_dir / "numerics/unchanged-weights.safetensors",
        expected_sha256=numeric.portable_weights_sha256,
        device="cpu",
    )
    required = {"intent.json", "profile-subset.json", "measurement-context.json"}
    required.update(
        "numerics/" + name for name in (*numeric.artifact_hashes, "numeric-report.json", "DONE")
    )
    warmup, measured = (16, 32) if config.config.pilot.is_production else (1, 2)
    devices = ("cpu",) if report.missing_devices else ("cpu", "mps")
    for device in devices:
        for stage in ("one_hop", "primary"):
            required.update(
                f"{device}/{stage}/{name}"
                for name in (
                    "trial.json",
                    "resources.json",
                    "final-weights.safetensors",
                    *(f"update-{i:03d}.json" for i in range(warmup + measured)),
                )
            )
        required.update(
            f"{device}/evaluation/{name}"
            for name in ("trial.json", "observations.json", "components.json")
        )
        required.update(
            f"{device}/evaluation/autonomous/{name}"
            for name in verification._runtime_artifact_inventory(
                output_dir / device / "evaluation/autonomous",
                config=config,
                manifest=manifest,
                subset=report.manifest_subset,
                weights=(numeric.portable_weights_sha256, weights.identity),
            )
        )
    verification._check_artifact_closure(
        output_dir, report.artifact_hashes, required, report_name="throughput-report.json"
    )
    training, evaluation = {}, {}
    for device in devices:
        for stage in ("one_hop", "primary"):
            root = output_dir / device / stage
            observations = read_json(root / "resources.json")
            if set(observations) != {
                "process_highwater_bytes",
                "mps_driver_allocation_bytes",
                "final_weights_sha256",
            }:
                raise ValueError("raw training resource inventory differs")
            final = load_neural_weights(
                root / "final-weights.safetensors",
                expected_sha256=observations["final_weights_sha256"],
                device="cpu",
            )
            if (
                final.identity.source_revision != source_commit
                or final.model.config != config.config.neural
            ):
                raise ValueError("raw final diagnostic weights differ")
            rows = []
            for i in range(warmup + measured):
                row = read_json(root / f"update-{i:03d}.json")
                batch = verification.diagnostic_batch(config.config, stage=stage, counter=i)
                if (
                    set(row)
                    != {
                        "diagnostic_update",
                        "warmup",
                        "elapsed_seconds",
                        "example_keys",
                        "example_hashes",
                        "trace_steps",
                        "step",
                    }
                    or row["diagnostic_update"] != i + 1
                    or row["warmup"] is not (i < warmup)
                    or row["example_hashes"] != list(batch.example_hashes)
                    or row["example_keys"] != [asdict(k) for k in batch.example_keys]
                    or row["trace_steps"] != batch.targets.kind.shape[1]
                    or type(row["elapsed_seconds"]) not in (int, float)
                    or row["elapsed_seconds"] <= 0
                    or set(row["step"]) != {f.name for f in fields(StepResult)}
                    or row["step"]["objective_version"] != config.config.pilot.objective_version
                    or row["step"]["auxiliary_coefficient"]
                    != config.config.pilot.content_auxiliary_weight
                ):
                    raise ValueError("raw training update recipe differs")
                rows.append(row)
            computes = [row["step"]["compute"] for row in rows]
            for compute in computes:
                if any(
                    type(compute[n]) is not int or compute[n] < 0
                    for n in (
                        "forward_macs",
                        "backward_macs",
                        "memory_bytes",
                        "foundation_model_calls",
                    )
                ):
                    raise ValueError("raw compute counter differs")
            rebuilt = verification.TrainingTrial(
                device=device,
                stage=stage,
                batch_size=config.config.pilot.batch_size,
                max_trace_steps=rows[-1]["trace_steps"],
                warmup_updates=warmup,
                measured_updates=measured,
                update_seconds=tuple(row["elapsed_seconds"] for row in rows[warmup:]),
                raw_step_rows=len(rows),
                initial_model_sha256=numeric.initial_model_sha256,
                final_model_sha256=final.identity.model_state_sha256,
                parameter_count=sum(p.numel() for p in final.model.parameters()),
                total_forward_macs=sum(c["forward_macs"] for c in computes),
                total_backward_macs=sum(c["backward_macs"] for c in computes),
                tensor_working_memory_bytes=max(c["memory_bytes"] for c in computes),
                process_highwater_bytes=observations["process_highwater_bytes"],
                mps_driver_allocation_bytes=observations["mps_driver_allocation_bytes"],
                foundation_model_calls=sum(c["foundation_model_calls"] for c in computes),
            )
            if rebuilt != report.training_trials[f"{device}/{stage}"] or read_json(
                root / "trial.json"
            ) != rebuilt.model_dump(mode="json"):
                raise ValueError("raw training summary differs")
            training[f"{device}/{stage}"] = rebuilt
        root = output_dir / device / "evaluation"
        _, rows, _, _ = load_evaluation(root / "autonomous")
        observations = read_json(root / "observations.json")
        if set(observations) != {
            "elapsed_seconds",
            "component_elapsed_seconds",
            "process_highwater_bytes",
        }:
            raise ValueError("raw evaluation observations differ")
        component = read_json(root / "components.json")
        examples = tuple(
            e
            for i in range(2)
            for e in verification.diagnostic_batch(
                config.config, stage="one_hop", counter=i
            ).examples
        )
        if component["example_hashes"] != [e.example_hash for e in examples] or len(
            component["result"]["predictions"]["rows"]
        ) != len(examples):
            raise ValueError("raw component sample inventory differs")
        events, misses = Counter(), Counter()
        for row in rows:
            events.update(dict(row.event_counts))
            misses.update([row.miss_category or "success"])
        rebuilt = verification.EvaluationTrial(
            device=device,
            episodes=len(rows),
            elapsed_seconds=observations["elapsed_seconds"],
            mean_episode_seconds=observations["elapsed_seconds"] / len(rows),
            event_counts=dict(events),
            miss_counts=dict(misses),
            error_episodes=sum(r.error is not None for r in rows),
            forward_macs=sum(r.compute.forward_macs for r in rows),
            foundation_model_calls=sum(r.compute.foundation_model_calls for r in rows)
            + sum(c["foundation_model_calls"] for c in component["result"]["compute"]),
            retained_raw_bytes=sum(
                p.stat().st_size for p in (root / "autonomous").rglob("*") if p.is_file()
            ),
            largest_episode_bytes=max(
                sum(p.stat().st_size for p in (root / "autonomous/episodes").glob(f"{i:05d}.*"))
                for i in range(len(rows))
            ),
            process_highwater_bytes=observations["process_highwater_bytes"],
            tensor_working_memory_bytes=max(r.compute.memory_bytes for r in rows),
            component_episodes=len(examples),
            component_elapsed_seconds=observations["component_elapsed_seconds"],
            component_raw_bytes=(root / "components.json").stat().st_size,
        )
        if rebuilt != report.evaluation_trials[device] or read_json(
            root / "trial.json"
        ) != rebuilt.model_dump(mode="json"):
            raise ValueError("raw evaluation summary differs")
        evaluation[device] = rebuilt
    context = read_json(output_dir / "measurement-context.json")
    chosen, reason, checkpoint_bytes, forecast = _profile_projection(
        training, evaluation, numeric, output_dir, free_bytes=context["free_bytes"]
    )
    if (
        report.chosen_device != chosen
        or report.device_choice_reason != reason
        or report.forecast != forecast
        or report.checkpoint_bytes != checkpoint_bytes
        or report.retained_sample_bytes
        != sum((output_dir / name).stat().st_size for name in required)
        or report.environment != context["environment"]
        or report.destination != context["destination"]
        or report.destination_is_system_temp
        != report.destination.startswith(
            ("/private/tmp/", "/private/var/folders/", "/tmp/", "/var/folders/")
        )
    ):
        raise ValueError("raw profile choice/forecast summary differs")


def profile_pilot(config, *, output_dir):
    from silent_cascade.eventflow.checkpoint_rng import restore_rng_snapshot
    from silent_cascade.eventflow.neural_weights import load_neural_weights
    from silent_cascade.rng import snapshot_global_rng
    from silent_cascade.train import pilot_verification as verification
    from silent_cascade.train.pilot_data import PilotManifest, _read_pilot_bytes

    source = verification.authenticate_numeric_source(config, None)
    devices, missing = verification.native_devices()
    if config.config.pilot.is_production and (
        os.environ.get("OMP_NUM_THREADS") != "1" or torch.get_num_threads() != 1
    ):
        raise ValueError(
            "production CPU measurement requires OMP_NUM_THREADS=1 and one torch thread"
        )
    if output_dir.exists():
        raise ValueError("profile destination must be fresh; retain previous failures")
    publish_json(
        output_dir / "intent.json",
        {"source": source, "config": config.canonical_json.decode(), "devices": devices},
    )
    before = snapshot_global_rng()
    try:
        numeric = verification.verify_pilot_numerics(
            config, checkpoint=None, output_dir=output_dir / "numerics"
        )
        weights = load_neural_weights(
            output_dir / "numerics/unchanged-weights.safetensors",
            expected_sha256=numeric.portable_weights_sha256,
            device="cpu",
        )
        model = weights.model
        model.requires_grad_(True)
        manifest = PilotManifest.model_validate_json(
            _read_pilot_bytes(output_dir / "numerics/manifest.json")
        )
        subset = verification.declare_subset(
            manifest,
            count=256 if config.config.pilot.is_production else 16,
            output_path=output_dir / "profile-subset.json",
        )
        training, evaluation = {}, {}
        initial_rng = snapshot_global_rng()
        for device in devices:
            for stage in ("one_hop", "primary"):
                restore_rng_snapshot(initial_rng, restore_mps=device == "mps")
                training[f"{device}/{stage}"] = measure_training_trial(
                    config,
                    model=model,
                    stage=stage,
                    device=device,
                    output_dir=output_dir / device / stage,
                    warmup_updates=16 if config.config.pilot.is_production else 1,
                    measured_updates=32 if config.config.pilot.is_production else 2,
                    source_revision=source.source_commit,
                )
            evaluation[device] = _evaluation_trial(
                config,
                model=model,
                identity=weights.identity,
                weights_sha=numeric.portable_weights_sha256,
                manifest=manifest,
                subset=subset,
                device=device,
                output_dir=output_dir / device / "evaluation",
            )
        environment = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_threads": str(torch.get_num_threads()),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
            "memory_measurement": "process-lifetime getrusage RSS highwater; tensor "
            "working-memory estimate separately; MPS current driver allocation separately",
        }
        context = {
            "free_bytes": shutil.disk_usage(output_dir).free,
            "destination": str(output_dir.absolute()),
            "environment": environment,
        }
        publish_json(output_dir / "measurement-context.json", context)
        chosen, reason, checkpoint_bytes, forecast = _profile_projection(
            training, evaluation, numeric, output_dir, free_bytes=context["free_bytes"]
        )
        if verification.authenticate_numeric_source(config, None) != source:
            raise ValueError("source changed during workload measurement")
        artifact_hashes = verification._artifact_hashes(output_dir)
        result = verification.PilotThroughputReport(
            config_canonical_json=config.canonical_json.decode(),
            config_sha256=config.sha256,
            source=source,
            initial_model_sha256=numeric.initial_model_sha256,
            portable_weights_sha256=numeric.portable_weights_sha256,
            production_workload_measured=config.config.pilot.is_production,
            training_trials=training,
            evaluation_trials=evaluation,
            manifest_subset=subset,
            numeric_report_sha256=sha256_bytes(
                _read_pilot_bytes(output_dir / "numerics/numeric-report.json")
            ),
            numerical_checks_passed=numeric.device_checks_passed,
            chosen_device=chosen,
            device_choice_reason=reason,
            missing_devices=missing,
            checkpoint_bytes=checkpoint_bytes,
            retained_sample_bytes=sum(
                (output_dir / name).stat().st_size for name in artifact_hashes
            ),
            destination=str(output_dir.absolute()),
            destination_is_system_temp=str(output_dir.absolute()).startswith(
                ("/private/tmp/", "/private/var/folders/", "/tmp/", "/var/folders/")
            ),
            forecast=forecast,
            environment=environment,
            artifact_hashes=artifact_hashes,
        )
        digest = publish_json(output_dir / "throughput-report.json", result)
        publish_json(output_dir / "DONE", {"report_sha256": digest})
        return result
    except Exception as error:
        publish_json(
            output_dir / "failure.json", {"error_type": type(error).__name__, "error": str(error)}
        )
        raise
    finally:
        restore_rng_snapshot(before, restore_mps=before.torch_mps_state is not None)
