"""Offline autonomous evaluator; private episodes never cross the agent boundary."""

import copy
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path

import torch

from silent_cascade.env.episode import EpisodeBundle, episode_sha256
from silent_cascade.errors import DynamicsError
from silent_cascade.eval.artifacts import EvaluationIdentity, PilotEvaluation, write_evaluation
from silent_cascade.eval.compute import NeuralComputeMeter, aggregate_runtime_compute
from silent_cascade.eval.metrics import EvaluationError, TimedEpisodeRow
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.neural import NeuralEventFlowAgent
from silent_cascade.eventflow.state import ComputeCounters
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.io import atomic_create_bytes
from silent_cascade.logging.neural_trace import observe_neural_event, write_full_neural_trace
from silent_cascade.logging.runtime_diagnostics import runtime_diagnostic_identity
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.train.pilot_config import Phase4Config


def _counter_delta(after, before):
    return ComputeCounters(
        **{
            f.name: getattr(after, f.name) - getattr(before, f.name)
            for f in fields(ComputeCounters)
        }
    )


def _run_one(*, bundle, identity, agent, engine, output_dir, ordinal, retain):
    session, error = None, None
    snapshots, observations = [], []
    episode_operations = engine.operational_compute_snapshot()
    entity_parameters = agent.model.record_encoder.entity_embedding.weight.numel()
    meter = NeuralComputeMeter(agent.model)
    try:
        with meter, torch.no_grad():
            session = engine.start_episode(bundle, agent)
        snapshots.append(meter.snapshot())
        while True:
            before = session.state
            event_operations = engine.operational_compute_snapshot()
            meter = NeuralComputeMeter(agent.model)
            with meter, torch.no_grad():
                finished = engine.step(session, agent)
            snapshots.append(meter.snapshot())
            summary = session.trace.snapshot().events[-1]
            observations.append(
                observe_neural_event(
                    before=before,
                    after=session.state,
                    summary=summary,
                    diagnostics=agent.diagnostic_snapshot(),
                    compute=aggregate_runtime_compute(
                        _counter_delta(session.state.core.counters, before.core.counters),
                        (snapshots[-1],),
                        entity_parameters=entity_parameters,
                        causal_flow_evaluations=(
                            engine.operational_compute_snapshot().causal_flow_evaluations
                            - event_operations.causal_flow_evaluations
                        ),
                    ),
                )
            )
            if finished:
                break
    except DynamicsError as caught:
        snapshots.append(meter.snapshot())
        diagnostic = runtime_diagnostic_identity(caught)
        error = EvaluationError(code=diagnostic.code, invariant=diagnostic.invariant)
    actions = session.state.core.actions if session is not None else ()
    counters = session.state.core.counters if session is not None else ComputeCounters()
    trace = session.trace.snapshot() if session is not None else None
    counts = Counter(e.kind for e in trace.events) if trace is not None else Counter()
    stem = f"episodes/{ordinal:05d}"
    neural_ref = f"{stem}.neural.json"
    neural_raw = canonical_json_bytes(
        {
            "schema": "phase4-neural-observations-v1",
            "identity_sha256": identity.sha256,
            "episode_sha256": episode_sha256(bundle),
            "initial_compute": aggregate_runtime_compute(
                ComputeCounters(),
                (snapshots[0],),
                entity_parameters=entity_parameters,
            ).model_dump(mode="json"),
            "events": [o.model_dump(mode="json") for o in observations],
            "causal_events": [e.to_payload() for e in trace.events] if trace else [],
        }
    )
    atomic_create_bytes(output_dir / neural_ref, neural_raw)
    row = TimedEpisodeRow.from_outcome(
        identity=identity,
        bundle=bundle,
        actions=actions,
        error=error,
        crash_ref=None
        if error is None
        else f"crashes/index.json#{bundle.public.init.episode_public_id}",
        event_count=sum(counts.values()),
        event_counts=tuple(sorted(counts.items())),
        causal_trace_sha256=trace.sha256 if trace is not None else None,
        neural_trace_ref=neural_ref,
        neural_trace_sha256=sha256_bytes(neural_raw),
        compute=aggregate_runtime_compute(
            counters,
            tuple(snapshots),
            entity_parameters=entity_parameters,
            checkpoint_serializations=int(error is not None and session is not None),
            causal_flow_evaluations=(
                engine.operational_compute_snapshot().causal_flow_evaluations
                - episode_operations.causal_flow_evaluations
            ),
        ),
    )
    if retain or not row.timed_success or error is not None:
        full_ref = f"{stem}.trajectory.json.gz"
        digest = write_full_neural_trace(
            output_dir / full_ref,
            identity_sha256=identity.sha256,
            episode_sha256=row.episode_sha256,
            trajectory=session.trajectory if session is not None else None,
        )
        row = row.model_copy(update={"full_trace_ref": full_ref, "full_trace_sha256": digest})
    # Timing is deliberately outside semantic rows, event hashes and decisions.
    atomic_create_bytes(
        output_dir / f"{stem}.telemetry.json",
        canonical_json_bytes(
            {
                "elapsed_seconds": sum(s.elapsed_seconds for s in snapshots),
            }
        ),
    )
    return row


def evaluate_episodes(
    model: EventFlowModel,
    *,
    identity: EvaluationIdentity,
    config: Phase4Config,
    episodes: Iterable[EpisodeBundle],
    output_dir: Path,
    device: str,
) -> PilotEvaluation:
    """Own one frozen inference copy and stream each completed episode's evidence."""
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in {"", "0"}:
        raise ValueError("enabled MPS fallback is forbidden for evaluation")
    if device not in {"cpu", "mps"}:
        raise ValueError("evaluation requires native CPU or MPS")
    if canonical_json_bytes(config).decode() != identity.evaluation_config_canonical_json:
        raise ValueError("evaluation configuration differs from identity")
    # Deepcopy preserves shared non-owning module bindings without touching caller mode,
    # gradient flags, parameter identities, gradients, buffers, device or optimizer.
    inference = copy.deepcopy(model).to(device)
    agent = NeuralEventFlowAgent(inference, identity=identity.model_identity, device=device)
    if output_dir.exists():
        raise ValueError("evaluation destination must be fresh")
    output_dir.mkdir(parents=True)
    engine = EventEngine(
        config.event_flow,
        crash_root=output_dir / "crashes",
        source_revision=identity.execution_source_revision,
        experiment_config_canonical_json=identity.evaluation_config_canonical_json,
    )
    selected = identity.retained_public_ids() | frozenset(identity.report_example_public_ids)

    def rows():
        for ordinal, bundle in enumerate(episodes):
            if ordinal >= len(identity.episodes):
                raise ValueError("extra episode in evaluation inventory")
            binding = identity.episodes[ordinal]
            if (
                bundle.public.init.episode_public_id,
                episode_sha256(bundle),
                bundle.truth.recipe.variant.value,
                bundle.truth.recipe.requested_path_length,
            ) != (
                binding.public_id,
                binding.episode_sha256,
                binding.variant,
                binding.path_length,
            ):
                raise ValueError("episode differs from ordered evaluation inventory")
            if bundle.truth.key.split_namespace.value != identity.split:
                raise ValueError("episode split differs from evaluation inventory")
            yield _run_one(
                bundle=bundle,
                identity=identity,
                agent=agent,
                engine=engine,
                output_dir=output_dir,
                ordinal=ordinal,
                retain=binding.public_id in selected or identity.purpose == "delay_swap",
            )

    return write_evaluation(identity=identity, rows=rows(), output_dir=output_dir)
