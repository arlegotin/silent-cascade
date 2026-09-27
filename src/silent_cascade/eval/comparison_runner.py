"""Private scoring boundary for paired Phase 5A condition execution."""

import copy
import gzip
import hashlib
import json
import os
import shutil
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import torch

from silent_cascade.env.episode import EpisodeBundle, PublicEpisode, episode_sha256
from silent_cascade.env.reward import score_actions
from silent_cascade.eval.comparison_data import iter_bundles
from silent_cascade.eval.comparison_types import (
    ACCEPTED_PHASE4_SOURCE,
    BudgetLedger,
    ComparisonConfig,
    ComparisonIdentity,
    ComparisonManifest,
    ComparisonRow,
    ComparisonStep,
    ConditionResult,
)
from silent_cascade.eval.compute import (
    NeuralComputeMeter,
    RuntimeCompute,
    aggregate_runtime_compute,
)
from silent_cascade.eval.metrics import EvaluationError
from silent_cascade.eval.ponder_policy import PonderExecutionError, ponder_public
from silent_cascade.eventflow.compressed import (
    CompressedExecutionError,
    compressed_state_sha256,
    run_compressed_from_activation,
)
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.invariants import validate_post_jump, validate_session_boundary
from silent_cascade.eventflow.neural import NeuralEventFlowAgent, NeuralModelIdentity
from silent_cascade.eventflow.state import ComputeCounters
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from silent_cascade.io import atomic_create_bytes, atomic_write_bytes
from silent_cascade.logging.neural_trace import _host, write_full_neural_trace
from silent_cascade.models.activation_ponder import ActivationPonderModel
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.schemas import Action, Mode


@dataclass(frozen=True, slots=True)
class PublicProposal:
    """Actions proposed from public observations, before private arbitration."""

    actions: tuple[Action, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.actions, tuple) or any(
            not isinstance(action, Action) for action in self.actions
        ):
            raise TypeError("public proposals require an Action tuple")


def arbitrate_proposal(bundle: EpisodeBundle, proposal: PublicProposal) -> PublicProposal:
    """Evaluator only: private terminal wins equal-time ties and truncates later actions."""
    if not isinstance(bundle, EpisodeBundle) or not isinstance(proposal, PublicProposal):
        raise TypeError("private evaluator requires a bundle and a public proposal")
    if tuple(sorted(proposal.actions, key=lambda action: action.timestamp)) != proposal.actions:
        raise ValueError("proposed actions must be chronological")
    terminal = bundle.truth.private_terminal.timestamp
    return PublicProposal(
        tuple(action for action in proposal.actions if action.timestamp < terminal)
    )


def run_public_policy(
    bundle: EpisodeBundle, policy: Callable[[PublicEpisode], PublicProposal]
) -> PublicProposal:
    """Pass only public input to a condition, then arbitrate privately."""
    proposal = policy(bundle.public)
    if not isinstance(proposal, PublicProposal):
        raise TypeError("public policy must return a PublicProposal")
    return arbitrate_proposal(bundle, proposal)


def ponder_state_sha256(model: ActivationPonderModel) -> str:
    """Portable logical tensor identity, independent of checkpoint serialization."""
    if type(model) is not ActivationPonderModel:
        raise TypeError("ponder state identity requires the exact ponderer architecture")
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(
            canonical_json_bytes(
                {"name": name, "dtype": str(value.dtype), "shape": tuple(value.shape)}
            )
        )
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_ponder_episode(
    bundle: EpisodeBundle,
    *,
    model: ActivationPonderModel,
    cap: int,
    trace_sink: Callable[[ConditionResult, object | None], None] | None = None,
) -> ConditionResult:
    """Own the private terminal; the model sees only the complete public episode."""
    started = perf_counter()
    trace_steps = None
    try:
        decision = ponder_public(model, bundle.public, cap=cap)
        trace_steps = tuple(asdict(step) for step in decision.steps)
        proposal = PublicProposal(actions=() if decision.action is None else (decision.action,))
        actions = arbitrate_proposal(bundle, proposal).actions
        steps = tuple(
            ComparisonStep(
                event_id=bundle.public.events[-1].event_id + index + 1,
                kind="ponder",
                timestamp=step.cognitive_timestamp,
                state_sha256=sha256_bytes(canonical_json_bytes(asdict(step))),
                selected_record_id=step.selected_record_id,
                halt_probability=step.halt_probability,
                action_class=step.action_class,
                action_offset=step.action_offset,
            )
            for index, step in enumerate(decision.steps)
        )
        result = ConditionResult(
            actions=actions,
            stop_reason=decision.stop_reason,
            steps=steps,
            trace_sha256=sha256_bytes(canonical_json_bytes({"steps": list(trace_steps)})),
            end_to_end_compute=decision.compute,
            post_activation_compute=decision.compute,
            inference_wall_seconds=perf_counter() - started,
        )
    except PonderExecutionError as caught:
        trace_steps = tuple(asdict(step) for step in caught.steps)
        steps = tuple(
            ComparisonStep(
                event_id=bundle.public.events[-1].event_id + index + 1,
                kind="ponder",
                timestamp=step.cognitive_timestamp,
                state_sha256=sha256_bytes(canonical_json_bytes(asdict(step))),
                selected_record_id=step.selected_record_id,
                halt_probability=step.halt_probability,
                action_class=step.action_class,
                action_offset=step.action_offset,
            )
            for index, step in enumerate(caught.steps)
        )
        result = ConditionResult(
            stop_reason="dynamics_error",
            steps=steps,
            trace_sha256=sha256_bytes(canonical_json_bytes({"steps": list(trace_steps)})),
            end_to_end_compute=caught.compute,
            post_activation_compute=caught.compute,
            error=EvaluationError(code="dynamics_error", invariant=caught.cause_type),
            inference_wall_seconds=perf_counter() - started,
        )
    except Exception as caught:
        result = ConditionResult(
            stop_reason="dynamics_error",
            error=EvaluationError(code="dynamics_error", invariant=type(caught).__name__),
            inference_wall_seconds=perf_counter() - started,
        )
    if trace_sink is not None:
        trace_sink(result, trace_steps)
    return result


def _counter_delta(after: ComputeCounters, before: ComputeCounters) -> ComputeCounters:
    return ComputeCounters(
        **{
            item.name: getattr(after, item.name) - getattr(before, item.name)
            for item in fields(ComputeCounters)
        }
    )


def run_intact_episode(
    bundle: EpisodeBundle,
    *,
    model: EventFlowModel,
    config: ComparisonConfig,
    trace_sink: Callable[[ConditionResult, object | None], None] | None = None,
) -> ConditionResult:
    """Bridge the unchanged EventFlow engine into the new exploratory record."""
    started = perf_counter()
    if type(model) is not EventFlowModel:
        raise TypeError("intact comparison requires an EventFlowModel")
    if model.config != config.phase4_config.neural:
        raise ValueError("intact checkpoint architecture differs from accepted Phase 4")
    identity = NeuralModelIdentity.from_model(model, source_revision=ACCEPTED_PHASE4_SOURCE)
    agent = NeuralEventFlowAgent(model, identity=identity, device="cpu")
    engine = EventEngine(config.phase4_config.event_flow)
    entity_parameters = model.record_encoder.entity_embedding.weight.numel()
    snapshots = []
    post_snapshots = []
    session = None
    error = None
    activation_counters = ComputeCounters()
    activation_flows = 0
    current_meter = NeuralComputeMeter(model)
    try:
        with current_meter, torch.no_grad():
            session = engine.start_episode(bundle, agent)
        snapshots.append(current_meter.snapshot())
        while True:
            before_mode = session.state.core.mode
            current_meter = NeuralComputeMeter(model)
            try:
                with current_meter, torch.no_grad():
                    finished = engine.step(session, agent)
            finally:
                snapshots.append(current_meter.snapshot())
                if before_mode is not Mode.OBSERVING:
                    post_snapshots.append(snapshots[-1])
            if before_mode is Mode.OBSERVING and session.state.core.mode is Mode.SEARCHING:
                activation_counters = session.state.core.counters
                activation_flows = engine.operational_compute_snapshot().causal_flow_evaluations
            if finished:
                break
    except Exception as caught:
        error = EvaluationError(
            code="dynamics_error",
            invariant=getattr(caught, "context", {}).get("invariant")
            if isinstance(getattr(caught, "context", None), dict)
            else None,
        )
    counters = session.state.core.counters if session is not None else ComputeCounters()
    actions = session.state.core.actions if session is not None else ()
    trace = session.trace.snapshot() if session is not None else None
    performed_flows = engine.operational_compute_snapshot().causal_flow_evaluations
    end_to_end = aggregate_runtime_compute(
        counters,
        tuple(snapshots),
        entity_parameters=entity_parameters,
        causal_flow_evaluations=performed_flows,
    )
    post_counters = _counter_delta(counters, activation_counters)
    post_activation = aggregate_runtime_compute(
        post_counters,
        tuple(post_snapshots),
        entity_parameters=entity_parameters,
        causal_flow_evaluations=performed_flows - activation_flows,
    )
    result = ConditionResult(
        actions=actions,
        stop_reason="terminal" if error is None else "dynamics_error",
        steps=()
        if trace is None
        else tuple(
            ComparisonStep(
                event_id=event.event_id,
                kind=event.kind,
                timestamp=event.timestamp,
                state_sha256=event.state_sha256,
            )
            for event in trace.events
        ),
        trace_sha256=trace.sha256 if trace else None,
        end_to_end_compute=end_to_end,
        post_activation_compute=post_activation,
        error=error,
    )
    result = result.model_copy(update={"inference_wall_seconds": perf_counter() - started})
    if trace_sink is not None:
        trace_sink(result, session.trajectory if session is not None else None)
    return result


def _row(
    bundle: EpisodeBundle,
    identity: ComparisonIdentity,
    result: ConditionResult,
    *,
    full_trace_ref: str | None = None,
    full_trace_sha256: str | None = None,
) -> ComparisonRow:
    score = score_actions(bundle.truth, result.actions)
    return ComparisonRow(
        public_id=bundle.public.init.episode_public_id,
        condition=identity.condition,
        manifest_name=identity.manifest_name,
        protocol_sha256=identity.protocol_sha256,
        episode_sha256=episode_sha256(bundle),
        identity_sha256=identity.sha256,
        variant=bundle.truth.recipe.variant,
        path_length=bundle.truth.recipe.requested_path_length,
        truth=bundle.truth,
        result=result,
        score=score,
        timed_success=result.error is None and score.timed_success,
        error=result.error,
        inference_wall_seconds=result.inference_wall_seconds,
        full_trace_ref=full_trace_ref,
        full_trace_sha256=full_trace_sha256,
    )


def _sum_compute(*parts: RuntimeCompute) -> RuntimeCompute:
    """Combine disjoint measured scopes while retaining the peak memory and model size."""
    if not parts:
        return RuntimeCompute()
    engine = ComputeCounters(
        **{
            item.name: sum(getattr(part.engine, item.name) for part in parts)
            for item in fields(ComputeCounters)
        }
    )
    modules: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    for part in parts:
        modules.update(part.module_calls)
        operations.update(part.operation_estimates)
    special = {
        "engine",
        "module_calls",
        "operation_estimates",
        "parameters",
        "entity_parameters",
        "memory_bytes",
    }
    totals = {
        name: sum(getattr(part, name) for part in parts)
        for name in RuntimeCompute.model_fields
        if name not in special
    }
    return RuntimeCompute(
        engine=engine,
        module_calls=dict(modules),
        operation_estimates=dict(operations),
        parameters=max(part.parameters for part in parts),
        entity_parameters=max(part.entity_parameters for part in parts),
        memory_bytes=max(part.memory_bytes for part in parts),
        **totals,
    )


def _run_compressed_episode(
    bundle: EpisodeBundle,
    *,
    model: EventFlowModel,
    config: ComparisonConfig,
    trace_sink: Callable[[ConditionResult, object | None], None] | None = None,
) -> ConditionResult:
    """Own terminal arbitration; pass only a detached public runtime state to compression."""
    started = perf_counter()
    identity = NeuralModelIdentity.from_model(model, source_revision=ACCEPTED_PHASE4_SOURCE)
    agent = NeuralEventFlowAgent(model, identity=identity, device="cpu")
    engine = EventEngine(config.phase4_config.event_flow)
    entity_parameters = model.record_encoder.entity_embedding.weight.numel()
    prefix_snapshots = []
    session = None
    intervention_states = []
    try:
        meter = NeuralComputeMeter(model)
        with meter, torch.no_grad():
            session = engine.start_episode(bundle, agent)
        prefix_snapshots.append(meter.snapshot())
        while session.state.core.mode is Mode.OBSERVING:
            meter = NeuralComputeMeter(model)
            try:
                with meter, torch.no_grad():
                    engine.step(session, agent)
            finally:
                prefix_snapshots.append(meter.snapshot())
        if session.state.core.mode is not Mode.SEARCHING:
            raise ValueError("diagnostic episode did not reach public activation")
        activation = session.state
        prefix_flows = engine.operational_compute_snapshot().causal_flow_evaluations
        prefix = aggregate_runtime_compute(
            activation.core.counters,
            tuple(prefix_snapshots),
            entity_parameters=entity_parameters,
            causal_flow_evaluations=prefix_flows,
        )
        decision = run_compressed_from_activation(
            agent, copy.deepcopy(activation), state_observer=intervention_states.append
        )
        steps = [
            ComparisonStep(
                event_id=event.event_id,
                kind=event.kind,
                timestamp=event.timestamp,
                state_sha256=event.state_sha256,
            )
            for event in session.trace.snapshot().events
        ]
        steps.extend(
            ComparisonStep(
                event_id=step.event_id,
                kind=step.kind.value,
                timestamp=step.executed_at,
                state_sha256=step.state_sha256,
                predicted_crossing_at=step.predicted_crossing_at,
                predicted_delta=step.predicted_delta,
                bypassed_refractory_until=step.bypassed_refractory_until,
                selected_record_id=step.selected_record_id,
            )
            for step in decision.steps
        )
        current = decision.final_state
        post = decision.compute
        actions = ()
        predicted = decision.predicted_act
        terminal = bundle.truth.private_terminal
        if predicted is not None and predicted.timestamp < terminal.timestamp:
            validate_session_boundary(current, next_event=predicted)
            before_flows = engine.operational_compute_snapshot().causal_flow_evaluations
            before = engine.advance_to(current, predicted.timestamp)
            meter = NeuralComputeMeter(model)
            with meter, torch.no_grad():
                after, emitted = agent.on_internal(before, predicted)
            validate_post_jump(before, predicted, after)
            if tuple(emitted) != after.core.actions[len(before.core.actions) :]:
                raise ValueError("compressed ACT emission differs from runtime state")
            actions = tuple(emitted)
            post = _sum_compute(
                post,
                aggregate_runtime_compute(
                    _counter_delta(after.core.counters, current.core.counters),
                    (meter.snapshot(),),
                    entity_parameters=entity_parameters,
                    causal_flow_evaluations=(
                        engine.operational_compute_snapshot().causal_flow_evaluations - before_flows
                    ),
                ),
            )
            current = after
            intervention_states.append(after)
            steps.append(
                ComparisonStep(
                    event_id=predicted.event_id,
                    kind="act",
                    timestamp=predicted.timestamp,
                    state_sha256=compressed_state_sha256(after),
                    predicted_crossing_at=predicted.timestamp,
                    predicted_delta=predicted.predicted_delta,
                )
            )
        before_flows = engine.operational_compute_snapshot().causal_flow_evaluations
        after_terminal_advance = engine.advance_to(current, terminal.timestamp)
        post = _sum_compute(
            post,
            aggregate_runtime_compute(
                _counter_delta(after_terminal_advance.core.counters, current.core.counters),
                (),
                entity_parameters=entity_parameters,
                causal_flow_evaluations=(
                    engine.operational_compute_snapshot().causal_flow_evaluations - before_flows
                ),
            ),
        )
        steps.append(
            ComparisonStep(
                event_id=terminal.event_id,
                kind="terminal",
                timestamp=terminal.timestamp,
                state_sha256=compressed_state_sha256(after_terminal_advance),
            )
        )
        ordered = tuple(steps)
        intervention_states.append(after_terminal_advance)
        result = ConditionResult(
            actions=actions,
            stop_reason=decision.stop_reason,
            steps=ordered,
            trace_sha256=sha256_bytes(
                canonical_json_bytes({"steps": [step.model_dump(mode="json") for step in ordered]})
            ),
            end_to_end_compute=_sum_compute(prefix, post),
            post_activation_compute=post,
        )
        result = result.model_copy(update={"inference_wall_seconds": perf_counter() - started})
        if trace_sink is not None:
            prefix_states = session.trajectory.checkpoint_snapshots()
            trace_sink(result, (*prefix_states, *intervention_states))
        return result
    except Exception as caught:
        counters = session.state.core.counters if session is not None else ComputeCounters()
        prefix = aggregate_runtime_compute(
            counters,
            tuple(prefix_snapshots),
            entity_parameters=entity_parameters,
            causal_flow_evaluations=engine.operational_compute_snapshot().causal_flow_evaluations,
        )
        if isinstance(caught, CompressedExecutionError) and session is not None:
            partial = caught.partial
            prefix_steps = [
                ComparisonStep(
                    event_id=event.event_id,
                    kind=event.kind,
                    timestamp=event.timestamp,
                    state_sha256=event.state_sha256,
                )
                for event in session.trace.snapshot().events
            ]
            prefix_steps.extend(
                ComparisonStep(
                    event_id=step.event_id,
                    kind=step.kind.value,
                    timestamp=step.executed_at,
                    state_sha256=step.state_sha256,
                    predicted_crossing_at=step.predicted_crossing_at,
                    predicted_delta=step.predicted_delta,
                    bypassed_refractory_until=step.bypassed_refractory_until,
                    selected_record_id=step.selected_record_id,
                )
                for step in partial.steps
            )
            result = ConditionResult(
                actions=(),
                stop_reason="dynamics_error",
                steps=tuple(prefix_steps),
                end_to_end_compute=_sum_compute(prefix, partial.compute),
                post_activation_compute=partial.compute,
                error=EvaluationError(
                    code="dynamics_error",
                    invariant=type(caught.__cause__).__name__ if caught.__cause__ else None,
                ),
            )
            result = result.model_copy(update={"inference_wall_seconds": perf_counter() - started})
            if trace_sink is not None:
                trace_sink(
                    result, (*session.trajectory.checkpoint_snapshots(), *intervention_states)
                )
            return result
        result = ConditionResult(
            actions=(),
            stop_reason="dynamics_error",
            end_to_end_compute=prefix,
            error=EvaluationError(code="dynamics_error", invariant=type(caught).__name__),
        )
        result = result.model_copy(update={"inference_wall_seconds": perf_counter() - started})
        if trace_sink is not None:
            trace_sink(
                result,
                (*session.trajectory.checkpoint_snapshots(), *intervention_states)
                if session is not None
                else None,
            )
        return result


def run_comparison(
    *,
    config: ComparisonConfig,
    manifest: ComparisonManifest,
    identity: ComparisonIdentity,
    model: torch.nn.Module,
    output_dir: Path,
    budget: BudgetLedger,
) -> Path:
    """Stream one condition over an ordered diagnostic manifest, retaining failures."""
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") not in {"", "0"}:
        raise ValueError("MPS fallback is forbidden for the CPU comparison")
    if identity.protocol_sha256 != config.protocol_sha256:
        raise ValueError("comparison identity protocol mismatch")
    if identity.config_sha256 != config.config_sha256:
        raise ValueError("comparison identity config mismatch")
    if identity.generator_sha256 != config.generator_sha256:
        raise ValueError("comparison identity generator mismatch")
    if identity.manifest_sha256 != sha256_bytes(canonical_json_bytes(manifest)):
        raise ValueError("comparison identity manifest mismatch")
    if identity.manifest_name != manifest.name:
        raise ValueError("comparison identity manifest name mismatch")
    if identity.condition == "activation_ponder":
        if type(model) is not ActivationPonderModel:
            raise TypeError("ponder comparison requires the exact ponderer architecture")
        actual_state = ponder_state_sha256(model)
    else:
        if type(model) is not EventFlowModel or model.config != config.phase4_config.neural:
            raise TypeError("intact comparison requires the accepted EventFlow architecture")
        actual_state = NeuralModelIdentity.from_model(
            model, source_revision=identity.producing_source_revision
        ).model_state_sha256
    if actual_state != identity.model_state_sha256:
        raise ValueError("comparison identity model-state mismatch")
    inference = copy.deepcopy(model).to("cpu").eval()
    identity_bytes = canonical_json_bytes(identity) + b"\n"
    resumed = output_dir.exists()
    if output_dir.exists():
        if (
            not (output_dir / "identity.json").exists()
            or (output_dir / "identity.json").read_bytes() != identity_bytes
        ):
            raise ValueError("existing comparison has incompatible identity")
        if (output_dir / "inventory.json").exists():
            inventory = json.loads((output_dir / "inventory.json").read_bytes())
            if inventory["identity_sha256"] != identity.sha256 or inventory[
                "rows_sha256"
            ] != sha256_file(output_dir / "rows.jsonl.gz"):
                raise ValueError("completed comparison inventory is corrupt")
            return output_dir
        if not (output_dir / "budget.json").exists():
            raise ValueError("existing comparison lacks durable budget")
        retained = BudgetLedger.model_validate_json((output_dir / "budget.json").read_bytes())
        extensions = tuple(dict.fromkeys((*retained.extensions, *budget.extensions)))
        budget = retained.model_copy(update={"extensions": extensions})
    else:
        output_dir.mkdir(parents=True)
        atomic_create_bytes(output_dir / "identity.json", identity_bytes)
    (output_dir / "rows").mkdir(exist_ok=True)

    def local_retained_bytes() -> int:
        return sum(
            path.stat().st_size
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "budget.json"
        )

    local_start_bytes = local_retained_bytes()
    base_retained_bytes = (
        budget.retained_bytes - local_start_bytes if resumed else budget.retained_bytes
    )
    if base_retained_bytes < 0:
        raise ValueError("resumed comparison budget understates retained evidence")
    started = perf_counter()
    budget = budget.model_copy(update={"attempts": budget.attempts + 1})

    def charge() -> None:
        charged = budget.model_copy(
            update={
                "elapsed_scientific_seconds": budget.elapsed_scientific_seconds
                + (perf_counter() - started),
                "retained_bytes": base_retained_bytes + local_retained_bytes(),
            }
        )
        atomic_write_bytes(output_dir / "budget.json", canonical_json_bytes(charged) + b"\n")

    charge()
    count = 0
    failures = 0
    retained_successes = 0
    try:
        for index, bundle in enumerate(iter_bundles(manifest, config)):
            shard = output_dir / "rows" / f"{index:05d}.json.gz"
            if shard.exists():
                row = ComparisonRow.model_validate_json(gzip.decompress(shard.read_bytes()))
                if (
                    row.public_id != bundle.public.init.episode_public_id
                    or row.episode_sha256 != episode_sha256(bundle)
                    or row.identity_sha256 != identity.sha256
                ):
                    raise ValueError("existing row shard differs from comparison identity")
            else:
                free = shutil.disk_usage(output_dir).free
                if free < config.working_reserve_bytes:
                    raise OSError("comparison storage reserve is unavailable")
                elapsed = budget.elapsed_scientific_seconds + (perf_counter() - started)
                notes = list(budget.extensions)
                unit = f"{identity.condition}/{manifest.name}"
                milestone = (
                    "Milestone B" if identity.condition == "activation_ponder" else "Milestone A"
                )
                time_target = (
                    config.milestone_b_time_target_seconds
                    if identity.condition == "activation_ponder"
                    else config.milestone_a_time_target_seconds
                )
                if elapsed >= time_target and not any(
                    f"{milestone} time target" in note for note in notes
                ):
                    notes.append(
                        f"{milestone} time target extended before {unit} "
                        f"episode {index}; {len(manifest.entries) - index} in this unit remain, "
                        f"{free} free bytes, {elapsed:.3f} cumulative seconds; "
                        "complete fixed paired coverage while reserve remains"
                    )
                retained_estimate = base_retained_bytes + local_retained_bytes()
                if retained_estimate >= config.retained_artifact_target_bytes and not any(
                    f"{milestone} storage target" in note for note in notes
                ):
                    notes.append(
                        f"{milestone} storage target extended before {unit} "
                        f"episode {index}; {free} free bytes and {len(manifest.entries) - index} "
                        "episodes remain in this unit"
                    )
                if tuple(notes) != budget.extensions:
                    budget = budget.model_copy(update={"extensions": tuple(notes)})
                    charge()
                trace_binding: list[tuple[str, str]] = []

                def trace_sink(
                    result: ConditionResult,
                    states: object | None,
                    *,
                    current_bundle: EpisodeBundle = bundle,
                    current_index: int = index,
                    successful_traces: int = retained_successes,
                    bindings: list[tuple[str, str]] = trace_binding,
                ) -> None:
                    score = score_actions(current_bundle.truth, result.actions)
                    if result.error is None and score.timed_success and successful_traces >= 32:
                        return
                    reference = f"traces/{current_index:05d}.trajectory.json.gz"
                    path = output_dir / reference
                    path.parent.mkdir(exist_ok=True)
                    if identity.condition == "intact_eventflow":
                        destination = (
                            path
                            if not path.exists()
                            else path.with_name(f".{path.name}.{uuid4().hex}.candidate")
                        )
                        digest = write_full_neural_trace(
                            destination,
                            identity_sha256=identity.sha256,
                            episode_sha256=episode_sha256(current_bundle),
                            trajectory=states,
                        )
                        if destination != path:
                            try:
                                if sha256_file(path) != digest:
                                    raise ValueError("retained intact trace differs on resume")
                            finally:
                                destination.unlink(missing_ok=True)
                    else:
                        if hasattr(states, "checkpoint_snapshots"):
                            states = states.checkpoint_snapshots()
                        payload = gzip.compress(
                            canonical_json_bytes(
                                {
                                    "schema_version": (
                                        "phase5a-ponder-trajectory-v1"
                                        if identity.condition == "activation_ponder"
                                        else "phase5a-compressed-trajectory-v1"
                                    ),
                                    "identity_sha256": identity.sha256,
                                    "episode_sha256": episode_sha256(current_bundle),
                                    "states": []
                                    if states is None
                                    else [_host(state) for state in states],
                                    "steps": [
                                        step.model_dump(mode="json") for step in result.steps
                                    ],
                                }
                            ),
                            mtime=0,
                        )
                        digest = sha256_bytes(payload)
                        if path.exists():
                            if sha256_file(path) != digest:
                                raise ValueError("retained compressed trace differs on resume")
                        else:
                            atomic_create_bytes(path, payload)
                    bindings.append((reference, digest))

                if identity.condition == "intact_eventflow":
                    result = run_intact_episode(
                        bundle, model=inference, config=config, trace_sink=trace_sink
                    )
                elif identity.condition == "compressed_eventflow":
                    result = _run_compressed_episode(
                        bundle, model=inference, config=config, trace_sink=trace_sink
                    )
                else:
                    result = run_ponder_episode(
                        bundle,
                        model=inference,
                        cap=identity.transition_cap,
                        trace_sink=trace_sink,
                    )
                if not trace_binding and (
                    result.error is not None
                    or not score_actions(bundle.truth, result.actions).timed_success
                ):
                    raise ValueError("failed comparison episode lacks a full trace")
                reference, digest = trace_binding[0] if trace_binding else (None, None)
                row = _row(
                    bundle,
                    identity,
                    result,
                    full_trace_ref=reference,
                    full_trace_sha256=digest,
                )
                atomic_create_bytes(shard, gzip.compress(canonical_json_bytes(row), mtime=0))
                charge()
            count += 1
            failures += int(row.error is not None)
            if row.timed_success and row.full_trace_ref:
                retained_successes += 1
    finally:
        charge()
    row_file = output_dir / "rows.jsonl.gz"
    temporary = output_dir / f".rows.jsonl.gz.{uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                for index in range(count):
                    compressed.write(
                        gzip.decompress((output_dir / "rows" / f"{index:05d}.json.gz").read_bytes())
                        + b"\n"
                    )
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, row_file)
    finally:
        temporary.unlink(missing_ok=True)
    inventory = {
        "schema_version": "phase5a-run-inventory-v1",
        "identity_sha256": identity.sha256,
        "manifest_sha256": identity.manifest_sha256,
        "row_count": count,
        "error_count": failures,
        "rows_sha256": sha256_file(row_file),
    }
    atomic_create_bytes(output_dir / "inventory.json", canonical_json_bytes(inventory) + b"\n")
    charge()
    return output_dir
