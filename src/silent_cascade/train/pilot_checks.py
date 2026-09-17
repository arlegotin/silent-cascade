"""One package-owned final execution path; collection itself never executes models."""

from dataclasses import asdict
from pathlib import Path

import torch

from silent_cascade.env.episode import episode_sha256
from silent_cascade.eval.artifacts import DelayTransform, EpisodeBinding, EvaluationIdentity
from silent_cascade.eval.runner import evaluate_episodes
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.neural import NeuralEventFlowAgent
from silent_cascade.eventflow.neural_checkpoint import (
    load_neural_runtime_checkpoint,
    publish_neural_runtime_checkpoint,
    restore_neural_runtime,
    snapshot_neural_runtime,
)
from silent_cascade.eventflow.neural_replay import verify_neural_replay, write_neural_replay
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.report.pilot_artifacts import child, load_evaluation
from silent_cascade.train.evidence_types import read_bytes
from silent_cascade.train.pilot_data import _publish_pilot_bytes
from silent_cascade.train.pilot_evidence import (
    SUITES,
    authenticate_run,
    collect_pilot_evidence,
    expected_bundles,
    local_verification_inventory,
    verify_phase4_gate_artifact,
)
from silent_cascade.train.pilot_evidence_types import (
    MAX_BYTES,
    REQUIRED_COVERAGE,
    LocalVerificationReceipt,
    Phase4GateArtifact,
    strict_json,
)


def record_local_verification(*, repo_root: Path, output_dir: Path) -> LocalVerificationReceipt:
    """Execute the fixed local gate once, separately from scientific collection."""
    import os
    import subprocess

    from silent_cascade.train.provenance import git

    root = repo_root.resolve()
    revision = git(root, "rev-parse", "HEAD").decode().strip()
    before = local_verification_inventory(root, revision, working=True)
    if before != local_verification_inventory(root, revision):
        raise ValueError("local verification requires committed unchanged inputs")
    if output_dir.exists():
        raise ValueError("local verification attempt already exists")
    output_dir.mkdir(parents=True, exist_ok=False)
    _publish(
        output_dir / "intent.json",
        dict(command=["make", "verify"], verification_commit=revision, before_inventory=before),
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"MAKEFLAGS", "MFLAGS", "MAKEFILES", "MAKEOVERRIDES", "GNUMAKEFLAGS"}
    }
    environment.update(OMP_NUM_THREADS="1", UV_OFFLINE="1", UV_FROZEN="1")
    with (
        (output_dir / "stdout.log").open("xb") as stdout,
        (output_dir / "stderr.log").open("xb") as stderr,
    ):
        completed = subprocess.run(
            ["make", "verify"], cwd=root, stdout=stdout, stderr=stderr, env=environment, check=False
        )
    after_commit = git(root, "rev-parse", "HEAD").decode().strip()
    after = local_verification_inventory(root, after_commit, working=True)
    stdout_sha = sha256_bytes(read_bytes(output_dir / "stdout.log"))
    stderr_sha = sha256_bytes(read_bytes(output_dir / "stderr.log"))
    process = dict(
        command=["make", "verify"],
        returncode=completed.returncode,
        stdout_sha256=stdout_sha,
        stderr_sha256=stderr_sha,
    )
    _publish(output_dir / "process-result.json", process)
    receipt = LocalVerificationReceipt(
        verification_commit=revision,
        after_commit=after_commit,
        before_inventory=before,
        after_inventory=after,
        source_unchanged=before == after and revision == after_commit,
        returncode=completed.returncode,
        stdout_sha256=stdout_sha,
        stderr_sha256=stderr_sha,
        process_result_sha256=sha256_bytes(canonical_json_bytes(process)),
    )
    _publish(output_dir / "receipt.json", receipt)
    return receipt


def _publish(path, value):
    _publish_pilot_bytes(path, canonical_json_bytes(value))


def _evaluate(*, run_dir, config, source, result, manifests, weights, name):
    stage = name if name in {"two_hop", "robustness"} else "primary"
    manifest = manifests[stage]
    descriptor = result.selected_weights or result.latest_weights
    bundles = tuple(expected_bundles(manifest, config, name))
    paired = name.startswith("delay_")
    parent_hash = sha256_bytes(canonical_json_bytes(manifest))
    parents = [e for e in manifest.entries if e.projected.variant == "positive"]
    bindings = tuple(
        EpisodeBinding.from_bundle(
            bundle,
            transform=DelayTransform(
                parent_public_id=parents[index].projected.public_id,
                parent_episode_sha256=parents[index].projected.episode_sha256,
                version="phase4-delay-pair-v1",
                parameters_canonical_json=canonical_json_bytes(
                    {"delay": 12.0 if name == "delay_12" else 48.0}
                ).decode(),
            )
            if paired
            else None,
        )
        for index, bundle in enumerate(bundles)
    )
    identity = EvaluationIdentity(
        experiment="phase4-pilot-v1",
        stage=stage,
        split=manifest.split,
        purpose="delay_swap"
        if paired
        else ("pilot_validation" if config.config.pilot.is_production else "debug"),
        manifest_schema=manifest.schema_version,
        manifest_sha256=parent_hash,
        parent_manifest_sha256=parent_hash if paired else None,
        episodes=bindings,
        checkpoint_sha256=descriptor.sha256,
        model_identity=weights.identity,
        evaluation_config_canonical_json=config.canonical_json.decode(),
        execution_source_revision=source.source_commit,
    )
    output = run_dir / "final/eval" / name
    request = dict(identity_sha256=identity.sha256, device="cpu", purpose=name)
    _publish(output.parent / ("." + name + ".request.json"), request)
    if (output / "DONE").exists():
        previous, _, _, _ = load_evaluation(output)
        execution = strict_json(output / "execution.json")
        if previous != identity or execution["device"] != "cpu" or execution["purpose"] != name:
            raise ValueError("incompatible final execution reuse")
        return
    if output.exists():
        raise ValueError("partial final evaluation retained; cannot silently overwrite")
    if any(t.device.type != "cpu" for t in (*weights.model.parameters(), *weights.model.buffers())):
        raise ValueError("final evaluation model must actually execute on CPU")
    evaluate_episodes(
        weights.model,
        identity=identity,
        config=config.config,
        episodes=bundles,
        output_dir=output,
        device="cpu",
    )
    _publish(
        output / "execution.json",
        dict(
            schema_version="phase4-cpu-execution-v1",
            device=next(weights.model.parameters()).device.type,
            purpose=name,
            identity_sha256=identity.sha256,
            done_sha256=sha256_bytes(read_bytes(output / "DONE")),
            weights_sha256=descriptor.sha256,
            model_state_sha256=weights.identity.model_state_sha256,
            source_commit=source.source_commit,
        ),
    )


def _terminal(session):
    return dict(
        actions=[asdict(a) for a in session.state.core.actions],
        score=asdict(session.terminal_score),
        trace_sha256=session.trace.snapshot().sha256,
    )


def _continuations(*, run_dir, config, source, result, manifests, weights):
    path = run_dir / "final/continuation.json"
    if path.exists():
        return
    descriptor = result.selected_weights or result.latest_weights
    weights_path = child(run_dir, descriptor.path)
    _publish(
        run_dir / "final/runtime/intent.json",
        dict(source_commit=source.source_commit, weights_sha256=descriptor.sha256, device="cpu"),
    )
    engine = EventEngine(config.config.event_flow)
    records, covered = [], set()
    _, evaluation_rows, _, _ = load_evaluation(run_dir / "final/eval/primary")
    for bundle, row in zip(
        expected_bundles(manifests["primary"], config, "primary"), evaluation_rows, strict=True
    ):
        if row.error is not None:
            continue
        variant = "variant:" + bundle.truth.recipe.variant.value
        observed = strict_json(child(run_dir / "final/eval/primary", row.neural_trace_ref))[
            "causal_events"
        ]
        possible = {variant, "boundary:mid_flow"}
        possible.update("boundary:" + event["kind"] for event in observed)
        possible.update("mode:" + event["post_mode"] for event in observed)
        if (possible & set(REQUIRED_COVERAGE)) <= covered:
            continue
        agent = NeuralEventFlowAgent(weights.model, identity=weights.identity, device="cpu")
        with torch.no_grad():
            uninterrupted = engine.run_episode(bundle, agent)
        if uninterrupted.trace.sha256 != row.causal_trace_sha256:
            raise ValueError("selected runtime differs from final primary trace")
        events = uninterrupted.trace.events
        candidates = []
        prior_time, prior_mode = 0.0, "observing"
        for event in events:
            if event.timestamp > prior_time:
                candidates.append(
                    (None, "mid_flow", (event.timestamp + prior_time) / 2.0, prior_mode)
                )
                break
            prior_time, prior_mode = event.timestamp, event.post_mode.value
        candidates.extend(
            (index, event.kind, event.timestamp, event.post_mode.value)
            for index, event in enumerate(events)
        )
        replay_path = run_dir / "final/runtime" / (str(len(records)) + ".replay.json")
        replay_pair = None
        for index, boundary, pause_time, mode in candidates:
            coverage = {variant, "boundary:" + boundary, "mode:" + mode} & set(REQUIRED_COVERAGE)
            if coverage <= covered:
                continue
            agent = NeuralEventFlowAgent(weights.model, identity=weights.identity, device="cpu")
            with torch.no_grad():
                session = engine.start_episode(bundle, agent)
                if index is None:
                    engine.run_until(session, agent, pause_time)
                else:
                    for _ in range(index + 1):
                        engine.step(session, agent)
                snapshot = snapshot_neural_runtime(
                    session,
                    agent,
                    config=config.config.event_flow,
                    source_revision=source.source_commit,
                    experiment_config_canonical_json=config.canonical_json.decode(),
                )
                checkpoint = publish_neural_runtime_checkpoint(
                    run_dir / "final/runtime" / (str(len(records)) + ".safetensors"), snapshot
                )
                loaded = load_neural_runtime_checkpoint(
                    checkpoint.path,
                    expected_sha256=checkpoint.sha256,
                    config=config.config.event_flow,
                    source_revision=source.source_commit,
                    device="cpu",
                )
                restored, restored_agent = restore_neural_runtime(
                    loaded, device="cpu", restore_rng=False
                )
                while not engine.step(session, agent):
                    pass
                while not engine.step(restored, restored_agent):
                    pass
            original, resumed = _terminal(session), _terminal(restored)
            if replay_pair is None:
                write_neural_replay(
                    replay_path,
                    bundle=bundle,
                    result=uninterrupted,
                    identity=weights.identity,
                    config=config.config.event_flow,
                    weights=weights_path,
                    source_revision=source.source_commit,
                    experiment_config_canonical_json=config.canonical_json.decode(),
                )
                replay_pair = [
                    verify_neural_replay(replay_path, weights_path=weights_path).model_dump(
                        mode="json"
                    )
                    for _ in range(2)
                ]
            matched = (
                canonical_json_bytes(original) == canonical_json_bytes(resumed)
                and original["trace_sha256"] == uninterrupted.trace.sha256
                and replay_pair[0] == replay_pair[1]
            )
            records.append(
                dict(
                    public_id=row.public_id,
                    episode_sha256=episode_sha256(bundle),
                    weights_sha256=descriptor.sha256,
                    model_state_sha256=weights.identity.model_state_sha256,
                    source_commit=source.source_commit,
                    device="cpu",
                    boundary=boundary,
                    pause_time=pause_time,
                    event_index=index,
                    mode=mode,
                    variant=bundle.truth.recipe.variant.value,
                    checkpoint_path=checkpoint.path.relative_to(run_dir).as_posix(),
                    checkpoint_sha256=checkpoint.sha256,
                    replay_path=replay_path.relative_to(run_dir).as_posix(),
                    replay_sha256=sha256_bytes(read_bytes(replay_path)),
                    original=original,
                    restored=resumed,
                    replay_comparisons=replay_pair,
                    coverage=sorted(coverage),
                    matched=matched,
                )
            )
            if matched:
                covered.update(coverage)
        if covered >= set(REQUIRED_COVERAGE):
            break
    _publish(path, {"schema_version": "phase4-continuation-evidence-v1", "records": records})


def _run_pilot_checks_owned(*, run_dir, config, output_path):
    """Called only under the workflow's existing permanent run ownership."""
    root, source, result, manifests, weights, archive = authenticate_run(run_dir, config)
    if output_path.exists():
        verified = verify_phase4_gate_artifact(output_path, repo_root=root, raw_run_dir=run_dir)
        if verified["missing_raw_attachments"]:
            raise ValueError("missing raw evidence prevents compatible execution reuse")
        return Phase4GateArtifact.model_validate_json(read_bytes(output_path, limit=MAX_BYTES))
    if weights is not None:
        for name in SUITES:
            _evaluate(
                run_dir=run_dir,
                config=config,
                source=source,
                result=result,
                manifests=manifests,
                weights=weights,
                name=name,
            )
        _continuations(
            run_dir=run_dir,
            config=config,
            source=source,
            result=result,
            manifests=manifests,
            weights=weights,
        )
        from silent_cascade.train.pilot_offline import measure_pilot_offline
        from silent_cascade.train.pilot_verification import verify_pilot_numerics

        if not (run_dir / "final/numerics/DONE").exists():
            verify_pilot_numerics(config, checkpoint=archive, output_dir=run_dir / "final/numerics")
        if not (run_dir / "final/offline/offline.json").exists():
            measure_pilot_offline(output_dir=run_dir / "final/offline")
    return collect_pilot_evidence(run_dir=run_dir, config=config, output_path=output_path)


def run_pilot_checks(*, run_dir: Path, config, output_path: Path) -> Phase4GateArtifact:
    from silent_cascade.train.pilot_workflow import pilot_ownership

    owner = strict_json(run_dir.parent / ("." + run_dir.name + ".owner.json"))
    with pilot_ownership(
        run_dir,
        run_identity=owner["run_identity"],
        recover=lambda: authenticate_run(run_dir, config),
    ):
        _, source, _, _, _, _ = authenticate_run(run_dir, config)
        workflow = strict_json(run_dir / "workflow.json")
        manifest = source.data_introductions["primary/manifest"]["path"]
        identity = sha256_bytes(
            canonical_json_bytes(
                dict(
                    config_sha256=config.sha256,
                    executable_sha256=source.source_sha256,
                    device=workflow["device"],
                    manifest_directory=Path(manifest).parent.as_posix(),
                )
            )
        )
        if identity != owner["run_identity"]:
            raise ValueError("standalone ownership identity differs")
        return _run_pilot_checks_owned(run_dir=run_dir, config=config, output_path=output_path)
