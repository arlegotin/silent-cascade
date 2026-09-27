"""Private scoring boundary for paired Phase 5A condition execution."""

import copy
import gzip
import os
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from time import perf_counter

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
from silent_cascade.eval.compute import NeuralComputeMeter, aggregate_runtime_compute
from silent_cascade.eval.metrics import EvaluationError
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.neural import NeuralEventFlowAgent, NeuralModelIdentity
from silent_cascade.eventflow.state import ComputeCounters
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from silent_cascade.io import atomic_create_bytes
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


def _counter_delta(after: ComputeCounters, before: ComputeCounters) -> ComputeCounters:
    return ComputeCounters(
        **{
            item.name: getattr(after, item.name) - getattr(before, item.name)
            for item in fields(ComputeCounters)
        }
    )


def run_intact_episode(
    bundle: EpisodeBundle, *, model: EventFlowModel, config: ComparisonConfig
) -> ConditionResult:
    """Bridge the unchanged EventFlow engine into the new exploratory record."""
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
                if before_mode is Mode.SEARCHING:
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
    return ConditionResult(
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


def _row(
    bundle: EpisodeBundle, identity: ComparisonIdentity, result: ConditionResult
) -> ComparisonRow:
    score = score_actions(bundle.truth, result.actions)
    return ComparisonRow(
        public_id=bundle.public.init.episode_public_id,
        episode_sha256=episode_sha256(bundle),
        identity_sha256=identity.sha256,
        variant=bundle.truth.recipe.variant,
        path_length=bundle.truth.recipe.requested_path_length,
        truth=bundle.truth,
        result=result,
        score=score,
        timed_success=result.error is None and score.timed_success,
        error=result.error,
    )


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
    if output_dir.exists():
        raise ValueError("comparison destination must be fresh")
    if identity.protocol_sha256 != config.protocol_sha256:
        raise ValueError("comparison identity protocol mismatch")
    if identity.config_sha256 != config.config_sha256:
        raise ValueError("comparison identity config mismatch")
    if identity.generator_sha256 != config.generator_sha256:
        raise ValueError("comparison identity generator mismatch")
    if identity.manifest_sha256 != sha256_bytes(canonical_json_bytes(manifest)):
        raise ValueError("comparison identity manifest mismatch")
    if identity.condition != "intact_eventflow":
        raise NotImplementedError("condition implementation is assigned to a later task")
    if type(model) is not EventFlowModel or model.config != config.phase4_config.neural:
        raise TypeError("intact comparison requires the accepted EventFlow architecture")
    actual_state = NeuralModelIdentity.from_model(
        model, source_revision=identity.producing_source_revision
    ).model_state_sha256
    if actual_state != identity.model_state_sha256:
        raise ValueError("comparison identity model-state mismatch")
    inference = copy.deepcopy(model).to("cpu").eval()
    output_dir.mkdir(parents=True)
    started = perf_counter()
    atomic_create_bytes(output_dir / "identity.json", canonical_json_bytes(identity) + b"\n")
    count = 0
    failures = 0
    with (
        (output_dir / "rows.jsonl.gz").open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed,
    ):
        for bundle in iter_bundles(manifest, config):
            result = run_intact_episode(bundle, model=inference, config=config)
            row = _row(bundle, identity, result)
            compressed.write(canonical_json_bytes(row) + b"\n")
            count += 1
            failures += int(row.error is not None)
    row_file = output_dir / "rows.jsonl.gz"
    inventory = {
        "schema_version": "phase5a-run-inventory-v1",
        "identity_sha256": identity.sha256,
        "manifest_sha256": identity.manifest_sha256,
        "row_count": count,
        "error_count": failures,
        "rows_sha256": sha256_file(row_file),
    }
    atomic_create_bytes(output_dir / "inventory.json", canonical_json_bytes(inventory) + b"\n")
    charged = budget.model_copy(
        update={
            "elapsed_scientific_seconds": budget.elapsed_scientific_seconds
            + (perf_counter() - started),
            "retained_bytes": budget.retained_bytes
            + sum(path.stat().st_size for path in output_dir.iterdir() if path.is_file()),
            "attempts": budget.attempts + 1,
        }
    )
    atomic_create_bytes(output_dir / "budget.json", canonical_json_bytes(charged) + b"\n")
    return output_dir
