"""Independent public-fact graph oracle and deterministic cognitive timing."""

import math
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from silent_cascade.env.config import OracleTimingConfig
from silent_cascade.env.episode import EpisodeTruth, EpisodeVariant, PublicEpisode
from silent_cascade.errors import OracleError
from silent_cascade.schemas import (
    MAX_ENTITY_ID,
    MAX_HAZARD_TYPE,
    Action,
    ActivationPayload,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    InternalEventKind,
    LinkFact,
    SafeFact,
)

_TIME_TOLERANCE = 1.0e-9


def _require_exact_int(
    value: object,
    name: str,
    *,
    maximum: int | None = None,
) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is out of bounds")


def _require_optional_int(
    value: object,
    name: str,
    *,
    maximum: int | None = None,
) -> None:
    if value is not None:
        _require_exact_int(value, name, maximum=maximum)


def _require_int_tuple(
    value: object,
    name: str,
    *,
    nonempty: bool = False,
    maximum: int | None = None,
) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if nonempty and not value:
        raise ValueError(f"{name} must be nonempty")
    for item in value:
        _require_exact_int(item, name, maximum=maximum)


def _require_nonnegative_float(value: object, name: str, *, positive: bool = False) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value) or value < 0.0 or (positive and value == 0.0):
        qualifier = "positive and finite" if positive else "nonnegative and finite"
        raise ValueError(f"{name} must be {qualifier}")


class OraclePolicy(StrEnum):
    """Explicitly authorized graph semantics for an evaluation suite."""

    PRIMARY = "primary"
    BRANCHING = "branching"
    CONTRADICTION = "contradiction"


class OracleTerminalKind(StrEnum):
    """Publicly derivable terminal decision."""

    HAZARD = "hazard"
    SAFE = "safe"
    DISCONNECTED = "disconnected"


@dataclass(frozen=True, slots=True)
class OracleTraceStep:
    """One event in the oracle's separate cognitive-trace namespace."""

    trace_step_id: int
    parent_trace_step_id: int | None
    kind: InternalEventKind
    timestamp: float
    delta: float
    selected_record_id: int | None
    focus_before: int | None
    focus_after: int | None

    def __post_init__(self) -> None:
        _validate_trace_step(self)


@dataclass(frozen=True, slots=True)
class OracleSolution:
    """Unique graph decision derived exclusively from public events."""

    terminal_kind: OracleTerminalKind
    node_path: tuple[int, ...]
    link_record_ids: tuple[int, ...]
    terminal_record_id: int | None
    hazard_type: int | None
    public_delay: float | None
    superseded_terminal_record_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _validate_solution(self)


@dataclass(frozen=True, slots=True)
class OracleTrace:
    """A timed oracle trace and its externally scored action sequence."""

    solution: OracleSolution
    steps: tuple[OracleTraceStep, ...]
    actions: tuple[Action, ...]

    def __post_init__(self) -> None:
        _validate_trace(self)


def _validate_solution(solution: OracleSolution) -> None:
    if not isinstance(solution.terminal_kind, OracleTerminalKind):
        raise TypeError("terminal_kind must be an OracleTerminalKind")
    _require_int_tuple(solution.node_path, "node_path", nonempty=True, maximum=MAX_ENTITY_ID)
    _require_int_tuple(solution.link_record_ids, "link_record_ids")
    _require_optional_int(solution.terminal_record_id, "terminal_record_id")
    _require_optional_int(solution.hazard_type, "hazard_type", maximum=MAX_HAZARD_TYPE)
    if solution.public_delay is not None:
        _require_nonnegative_float(solution.public_delay, "public_delay", positive=True)
    _require_int_tuple(
        solution.superseded_terminal_record_ids,
        "superseded_terminal_record_ids",
    )
    if len(solution.node_path) != len(solution.link_record_ids) + 1:
        raise ValueError("node_path and link_record_ids lengths disagree")
    if len(set(solution.node_path)) != len(solution.node_path):
        raise ValueError("node_path must not repeat nodes")
    support_ids = solution.link_record_ids + (
        () if solution.terminal_record_id is None else (solution.terminal_record_id,)
    )
    if len(set(support_ids)) != len(support_ids):
        raise ValueError("oracle support record IDs must be unique")
    if solution.terminal_kind is OracleTerminalKind.HAZARD:
        if (
            solution.terminal_record_id is None
            or solution.hazard_type is None
            or solution.public_delay is None
        ):
            raise ValueError("hazard solution requires terminal record, hazard type, and delay")
    elif solution.terminal_kind is OracleTerminalKind.SAFE:
        if (
            solution.terminal_record_id is None
            or solution.hazard_type is not None
            or solution.public_delay is not None
        ):
            raise ValueError("safe solution requires only a terminal record")
    elif (
        solution.terminal_record_id is not None
        or solution.hazard_type is not None
        or solution.public_delay is not None
    ):
        raise ValueError("disconnected solution cannot contain terminal hazard fields")


def _validate_trace_step(step: OracleTraceStep) -> None:
    _require_exact_int(step.trace_step_id, "trace_step_id")
    _require_optional_int(step.parent_trace_step_id, "parent_trace_step_id")
    if not isinstance(step.kind, InternalEventKind):
        raise TypeError("kind must be an InternalEventKind")
    _require_nonnegative_float(step.timestamp, "timestamp")
    _require_nonnegative_float(step.delta, "delta")
    _require_optional_int(step.selected_record_id, "selected_record_id")
    _require_optional_int(step.focus_before, "focus_before", maximum=MAX_ENTITY_ID)
    _require_optional_int(step.focus_after, "focus_after", maximum=MAX_ENTITY_ID)
    if step.parent_trace_step_id is not None and step.parent_trace_step_id >= step.trace_step_id:
        raise ValueError("parent_trace_step_id must precede trace_step_id")
    memory_values = (step.selected_record_id, step.focus_before, step.focus_after)
    if step.kind in (InternalEventKind.RECALL, InternalEventKind.COMPOSE):
        if any(value is None for value in memory_values):
            raise ValueError("memory trace steps require selection and focus")
    elif any(value is not None for value in memory_values):
        raise ValueError("non-memory trace steps cannot contain selection or focus")


def _validate_action(action: Action) -> None:
    if not isinstance(action, Action):
        raise TypeError("actions must contain Action values")
    _require_exact_int(action.hazard_type, "action.hazard_type", maximum=MAX_HAZARD_TYPE)
    _require_nonnegative_float(action.timestamp, "action.timestamp")
    _require_exact_int(action.caused_by_event_id, "action.caused_by_event_id")


def _validate_trace(trace: OracleTrace) -> None:
    if not isinstance(trace.solution, OracleSolution):
        raise TypeError("solution must be an OracleSolution")
    _validate_solution(trace.solution)
    if not isinstance(trace.steps, tuple):
        raise TypeError("steps must be a tuple")
    if not isinstance(trace.actions, tuple):
        raise TypeError("actions must be a tuple")
    support_ids = set(trace.solution.link_record_ids)
    if trace.solution.terminal_record_id is not None:
        support_ids.add(trace.solution.terminal_record_id)
    previous: OracleTraceStep | None = None
    for index, step in enumerate(trace.steps):
        if not isinstance(step, OracleTraceStep):
            raise TypeError("steps must contain OracleTraceStep values")
        _validate_trace_step(step)
        if step.trace_step_id != index:
            raise ValueError("trace step IDs must be zero-based and contiguous")
        expected_parent = None if index == 0 else index - 1
        if step.parent_trace_step_id != expected_parent:
            raise ValueError("trace parent links must form one contiguous chain")
        if (
            step.kind in (InternalEventKind.RECALL, InternalEventKind.COMPOSE)
            and step.selected_record_id not in support_ids
        ):
            raise ValueError("trace step selects a record outside oracle support")
        if previous is not None and (
            step.timestamp < previous.timestamp
            or not math.isclose(
                step.timestamp - previous.timestamp,
                step.delta,
                rel_tol=1.0e-12,
                abs_tol=_TIME_TOLERANCE,
            )
        ):
            raise ValueError("trace timestamps and deltas must be monotonic and aligned")
        previous = step
    for action in trace.actions:
        _validate_action(action)

    act_steps = tuple(step for step in trace.steps if step.kind is InternalEventKind.ACT)
    if trace.solution.terminal_kind is OracleTerminalKind.HAZARD:
        if len(act_steps) != 1 or not trace.steps or trace.steps[-1] is not act_steps[0]:
            raise ValueError("hazard trace requires one final ACT step")
        if len(trace.actions) != 1:
            raise ValueError("hazard trace requires exactly one action")
        action = trace.actions[0]
        act = act_steps[0]
        if (
            action.hazard_type != trace.solution.hazard_type
            or action.timestamp != act.timestamp
            or action.caused_by_event_id != act.trace_step_id
        ):
            raise ValueError("hazard action must match the oracle solution and ACT step")
    elif act_steps or trace.actions:
        raise ValueError("negative oracle traces must abstain and contain no ACT step")


def _require_valid_solution(solution: OracleSolution) -> None:
    try:
        _validate_solution(solution)
    except (TypeError, ValueError) as error:
        raise OracleError(f"invalid oracle solution: {error}") from error


def _require_valid_trace(trace: OracleTrace) -> None:
    try:
        _validate_trace(trace)
    except (TypeError, ValueError) as error:
        raise OracleError(f"invalid oracle trace: {error}") from error


type _Fact = LinkFact | HazardFact | SafeFact


def _fact_identity(payload: _Fact) -> tuple[object, ...]:
    if isinstance(payload, LinkFact):
        return ("link", payload.source_node, payload.target_node)
    if isinstance(payload, HazardFact):
        if payload.delay <= 0.0:
            raise OracleError("hazard delay must be positive")
        return ("hazard", payload.node, payload.hazard_type, payload.delay)
    if isinstance(payload, SafeFact):
        return ("safe", payload.node)
    raise OracleError("invalid FACT payload")


def solve_public_episode(
    public: PublicEpisode,
    policy: OraclePolicy = OraclePolicy.PRIMARY,
) -> OracleSolution:
    """Derive the unique primary legal trace from public facts and activation."""
    if not isinstance(public, PublicEpisode):
        raise TypeError("public must be a PublicEpisode")
    if not isinstance(policy, OraclePolicy):
        raise TypeError("policy must be an OraclePolicy")
    if policy is not OraclePolicy.PRIMARY:
        raise OracleError(f"oracle policy {policy.value!r} is not implemented in Phase 1 Task 5")

    links_by_source: dict[int, list[ExternalEvent]] = {}
    terminals_by_node: dict[int, list[ExternalEvent]] = {}
    identities: set[tuple[object, ...]] = set()
    for event in public.events[:-1]:
        if event.kind is not ExternalEventKind.FACT:
            raise OracleError("public pre-activation events must be FACT records")
        payload = event.payload
        if not isinstance(payload, (LinkFact, HazardFact, SafeFact)):
            raise OracleError("invalid FACT payload")
        identity = _fact_identity(payload)
        if identity in identities:
            raise OracleError("duplicate equivalent fact record")
        identities.add(identity)
        if isinstance(payload, LinkFact):
            links_by_source.setdefault(payload.source_node, []).append(event)
        else:
            terminals_by_node.setdefault(payload.node, []).append(event)

    activation = public.events[-1]
    if activation.kind is not ExternalEventKind.ACTIVATE or not isinstance(
        activation.payload, ActivationPayload
    ):
        raise OracleError("public episode must end in a valid activation")

    current_node = activation.payload.start_node
    node_path: list[int] = [current_node]
    link_record_ids: list[int] = []
    visited: set[int] = set()
    while True:
        if current_node in visited:
            raise OracleError("primary graph contains a reachable cycle")
        visited.add(current_node)

        terminals = terminals_by_node.get(current_node, [])
        if len(terminals) > 1:
            raise OracleError("primary graph has multiple terminals at a reachable node")
        if terminals:
            if links_by_source.get(current_node):
                raise OracleError("reachable terminal has an outgoing continuation")
            terminal = terminals[0]
            payload = terminal.payload
            if isinstance(payload, HazardFact):
                return OracleSolution(
                    OracleTerminalKind.HAZARD,
                    tuple(node_path),
                    tuple(link_record_ids),
                    terminal.event_id,
                    payload.hazard_type,
                    payload.delay,
                )
            if isinstance(payload, SafeFact):
                return OracleSolution(
                    OracleTerminalKind.SAFE,
                    tuple(node_path),
                    tuple(link_record_ids),
                    terminal.event_id,
                    None,
                    None,
                )
            raise OracleError("invalid reachable terminal payload")

        outgoing = links_by_source.get(current_node, [])
        if len(outgoing) > 1:
            raise OracleError("primary graph contains a reachable branch")
        if not outgoing:
            return OracleSolution(
                OracleTerminalKind.DISCONNECTED,
                tuple(node_path),
                tuple(link_record_ids),
                None,
                None,
                None,
            )
        link = outgoing[0]
        payload = link.payload
        if not isinstance(payload, LinkFact):
            raise OracleError("invalid reachable link payload")
        link_record_ids.append(link.event_id)
        current_node = payload.target_node
        node_path.append(current_node)


def verify_oracle_truth(solution: OracleSolution, truth: EpisodeTruth) -> None:
    """Fail if independently derived public support disagrees with private truth."""
    if not isinstance(solution, OracleSolution) or not isinstance(truth, EpisodeTruth):
        raise TypeError("solution and truth must be oracle and episode values")
    _require_valid_solution(solution)
    expected_kind = {
        EpisodeVariant.POSITIVE: OracleTerminalKind.HAZARD,
        EpisodeVariant.SAFE_NEGATIVE: OracleTerminalKind.SAFE,
        EpisodeVariant.DISCONNECTED_NEGATIVE: OracleTerminalKind.DISCONNECTED,
    }[truth.recipe.variant]
    expected_records = solution.link_record_ids + (
        () if solution.terminal_record_id is None else (solution.terminal_record_id,)
    )
    expected_private_kind = (
        ExternalEventKind.OUTCOME
        if solution.terminal_kind is OracleTerminalKind.HAZARD
        else ExternalEventKind.END
    )
    comparisons = (
        (solution.terminal_kind, expected_kind, "terminal kind"),
        (solution.node_path, truth.relevant_node_path, "node path"),
        (expected_records, truth.relevant_record_ids, "record support"),
        (solution.terminal_record_id, truth.terminal_record_id, "terminal record"),
        (solution.hazard_type, truth.relevant_hazard_type, "hazard type"),
        (truth.private_terminal.kind, expected_private_kind, "private terminal kind"),
    )
    for public_value, private_value, field in comparisons:
        if public_value != private_value:
            raise OracleError(f"oracle truth mismatch: {field}")
    if solution.terminal_kind is OracleTerminalKind.HAZARD:
        assert solution.public_delay is not None
        if not math.isclose(
            solution.public_delay,
            truth.episode_delay,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE,
        ):
            raise OracleError("oracle truth mismatch: episode delay")


def _delay_bucket(delay: float) -> int:
    if not math.isfinite(delay) or delay <= 0.0:
        raise OracleError("hazard delay must be positive")
    return math.floor(math.log2(delay))


def _competitive_count(selected: _Fact, facts: tuple[_Fact, ...]) -> int:
    count = 0
    for candidate in facts:
        if candidate is selected or type(candidate) is not type(selected):
            continue
        if isinstance(selected, LinkFact):
            assert isinstance(candidate, LinkFact)
            matches = (
                selected.source_node == candidate.source_node
                or selected.target_node == candidate.target_node
            )
        elif isinstance(selected, HazardFact):
            assert isinstance(candidate, HazardFact)
            matches = (
                selected.node == candidate.node
                or selected.hazard_type == candidate.hazard_type
                or _delay_bucket(selected.delay) == _delay_bucket(candidate.delay)
            )
        else:
            assert isinstance(selected, SafeFact) and isinstance(candidate, SafeFact)
            matches = selected.node == candidate.node
        count += matches
    return count


def _selected_trace_records(
    public: PublicEpisode, solution: OracleSolution
) -> tuple[tuple[ExternalEvent, int, int], ...]:
    facts_by_id = {
        event.event_id: event for event in public.events if event.kind is ExternalEventKind.FACT
    }
    if len(solution.node_path) != len(solution.link_record_ids) + 1:
        raise OracleError("oracle solution path and link support disagree")
    selected: list[tuple[ExternalEvent, int, int]] = []
    for index, record_id in enumerate(solution.link_record_ids):
        event = facts_by_id.get(record_id)
        if event is None or not isinstance(event.payload, LinkFact):
            raise OracleError("oracle link support does not name a public link")
        before = solution.node_path[index]
        after = solution.node_path[index + 1]
        if (event.payload.source_node, event.payload.target_node) != (before, after):
            raise OracleError("oracle link support does not match its node path")
        selected.append((event, before, after))

    if solution.terminal_kind is OracleTerminalKind.DISCONNECTED:
        if solution.terminal_record_id is not None:
            raise OracleError("disconnected oracle solution cannot select a terminal")
        return tuple(selected)
    terminal = facts_by_id.get(solution.terminal_record_id)
    expected_type = HazardFact if solution.terminal_kind is OracleTerminalKind.HAZARD else SafeFact
    if terminal is None or not isinstance(terminal.payload, expected_type):
        raise OracleError("oracle terminal support does not name the expected public terminal")
    node = solution.node_path[-1]
    if terminal.payload.node != node:
        raise OracleError("oracle terminal support does not match its node path")
    if isinstance(terminal.payload, HazardFact) and (
        terminal.payload.hazard_type != solution.hazard_type
        or solution.public_delay is None
        or not math.isclose(
            terminal.payload.delay,
            solution.public_delay,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE,
        )
    ):
        raise OracleError("oracle solution disagrees with selected public hazard")
    selected.append((terminal, node, node))
    return tuple(selected)


def build_oracle_trace(
    public: PublicEpisode,
    solution: OracleSolution,
    terminal_horizon: ExternalEvent,
    timing: OracleTimingConfig,
    rng: np.random.Generator,
) -> OracleTrace:
    """Build the unique deterministic-feasible trace using a local jitter RNG."""
    if not isinstance(public, PublicEpisode) or not isinstance(solution, OracleSolution):
        raise TypeError("public and solution must be oracle episode values")
    _require_valid_solution(solution)
    if not isinstance(terminal_horizon, ExternalEvent) or terminal_horizon.kind not in (
        ExternalEventKind.OUTCOME,
        ExternalEventKind.END,
    ):
        raise OracleError("terminal horizon must be a private OUTCOME or END event")
    if not isinstance(timing, OracleTimingConfig):
        raise TypeError("timing must be an OracleTimingConfig")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")

    activation = public.events[-1]
    if activation.kind is not ExternalEventKind.ACTIVATE:
        raise OracleError("public episode must end in activation")
    activation_time = activation.timestamp
    delay = terminal_horizon.timestamp - activation_time
    if not math.isfinite(delay) or delay <= 0.0:
        raise OracleError("terminal horizon must follow activation")
    expected_terminal_kind = (
        ExternalEventKind.OUTCOME
        if solution.terminal_kind is OracleTerminalKind.HAZARD
        else ExternalEventKind.END
    )
    if terminal_horizon.kind is not expected_terminal_kind:
        raise OracleError("terminal horizon kind disagrees with oracle solution")
    if solution.terminal_kind is OracleTerminalKind.HAZARD:
        assert solution.public_delay is not None
        if not math.isclose(
            solution.public_delay,
            delay,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE,
        ):
            raise OracleError("public hazard delay disagrees with terminal horizon")

    selected_records = _selected_trace_records(public, solution)
    facts = tuple(
        event.payload
        for event in public.events
        if event.kind is ExternalEventKind.FACT
        and isinstance(event.payload, (LinkFact, HazardFact, SafeFact))
    )
    event_specs: list[tuple[InternalEventKind, int, int, int, int]] = []
    for event, before, after in selected_records:
        assert isinstance(event.payload, (LinkFact, HazardFact, SafeFact))
        competitors = _competitive_count(event.payload, facts)
        event_specs.extend(
            (
                (InternalEventKind.RECALL, event.event_id, before, before, competitors),
                (InternalEventKind.COMPOSE, event.event_id, before, after, competitors),
            )
        )

    compose_budget = timing.terminal_compose_fraction * delay
    deltas: list[float] = []
    elapsed = 0.0
    for index, (*_, competitive_count) in enumerate(event_specs):
        current_time = activation_time + elapsed
        remaining_budget = activation_time + compose_budget - current_time
        remaining_event_count = len(event_specs) - index
        urgency = min(
            1.0,
            remaining_event_count * timing.delta_0 / max(remaining_budget, timing.delta_min),
        )
        raw = timing.delta_0 * (1.0 + 0.15 * competitive_count) / (1.0 + 0.5 * urgency)
        jittered = raw * math.exp(float(rng.normal(0.0, timing.jitter_log_std)))
        if not math.isfinite(jittered):
            raise OracleError("oracle trace jitter produced a nonfinite interval")
        delta = min(timing.delta_max, max(timing.delta_min, jittered))
        deltas.append(delta)
        elapsed += delta

    if elapsed > compose_budget:
        scale = compose_budget / elapsed
        deltas = [delta * scale for delta in deltas]
        if any(delta < timing.delta_min for delta in deltas):
            raise OracleError("oracle trace is temporally infeasible")

    steps: list[OracleTraceStep] = []
    current_time = activation_time
    for index, (spec, delta) in enumerate(zip(event_specs, deltas, strict=True)):
        kind, selected_id, before, after, _ = spec
        current_time += delta
        steps.append(
            OracleTraceStep(
                trace_step_id=index,
                parent_trace_step_id=None if index == 0 else index - 1,
                kind=kind,
                timestamp=current_time,
                delta=delta,
                selected_record_id=selected_id,
                focus_before=before,
                focus_after=after,
            )
        )

    actions: tuple[Action, ...] = ()
    if solution.terminal_kind is OracleTerminalKind.HAZARD:
        if solution.hazard_type is None:
            raise OracleError("hazard solution is missing its hazard type")
        target = activation_time + timing.action_target_fraction * delay
        act_id = len(steps)
        act_delta = target - current_time
        if act_delta < 0.0:
            raise OracleError("oracle action target precedes terminal composition")
        steps.append(
            OracleTraceStep(
                trace_step_id=act_id,
                parent_trace_step_id=None if act_id == 0 else act_id - 1,
                kind=InternalEventKind.ACT,
                timestamp=target,
                delta=act_delta,
                selected_record_id=None,
                focus_before=None,
                focus_after=None,
            )
        )
        actions = (Action(solution.hazard_type, target, act_id),)
    return OracleTrace(solution, tuple(steps), actions)


def scale_oracle_trace(trace: OracleTrace, factor: float) -> OracleTrace:
    """Scale all trace and action time values by one positive finite factor."""
    if not isinstance(trace, OracleTrace):
        raise TypeError("trace must be an OracleTrace")
    _require_valid_trace(trace)
    if type(factor) is not float or not math.isfinite(factor) or factor <= 0.0:
        raise OracleError("oracle trace scale factor must be positive and finite")
    scaled_step_times = tuple(
        (step.timestamp * factor, step.delta * factor) for step in trace.steps
    )
    scaled_action_times = tuple(action.timestamp * factor for action in trace.actions)
    if any(not math.isfinite(value) for values in scaled_step_times for value in values) or any(
        not math.isfinite(timestamp) for timestamp in scaled_action_times
    ):
        raise OracleError("scaled oracle trace contains nonfinite time values")
    return OracleTrace(
        solution=trace.solution,
        steps=tuple(
            replace(step, timestamp=timestamp, delta=delta)
            for step, (timestamp, delta) in zip(
                trace.steps,
                scaled_step_times,
                strict=True,
            )
        ),
        actions=tuple(
            replace(action, timestamp=timestamp)
            for action, timestamp in zip(
                trace.actions,
                scaled_action_times,
                strict=True,
            )
        ),
    )


def oracle_actions(trace: OracleTrace) -> tuple[Action, ...]:
    """Return the immutable externally scored action sequence for a trace."""
    if not isinstance(trace, OracleTrace):
        raise TypeError("trace must be an OracleTrace")
    return trace.actions
