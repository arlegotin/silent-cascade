"""Training-private content labels derived from an already validated oracle trace."""

import math
from dataclasses import dataclass

from silent_cascade.schemas import HazardFact, InternalEventKind, LinkFact, Mode, SafeFact
from silent_cascade.train.curriculum_data import (
    ComponentContent,
    ComponentTarget,
    CurriculumExample,
)


@dataclass(frozen=True, slots=True)
class TeacherStep:
    kind: InternalEventKind
    timestamp: float
    delta: float
    selected_record_id: int | None
    pre_mode: Mode
    post_mode: Mode
    active_record_id_before: int | None
    active_record_id_after: int | None
    support_before: tuple[int, ...]
    support_after: tuple[int, ...]
    focus_before: int | None
    focus: int | None = None
    role: int | None = None
    hazard_type: int | None = None
    log_delay: float | None = None
    normalized_deadline: float | None = None
    status: int | None = None
    confidence: float | None = None
    append_support: bool | None = None
    continue_search: bool | None = None
    action_class: int | None = None
    action_lead: float | None = None


@dataclass(frozen=True, slots=True)
class TeacherTrace:
    steps: tuple[TeacherStep, ...]
    terminal_class: int
    terminal_status: int
    final_mode: Mode
    final_support: tuple[int, ...]


def build_teacher_trace(example: CurriculumExample) -> TeacherTrace:
    facts = {event.event_id: event.payload for event in example.public.events[:-1]}
    oracle = example.oracle_trace
    activation_time = example.public.events[-1].timestamp
    terminal_class = example.solution.hazard_type if example.solution.hazard_type is not None else 4
    terminal_status = {"hazard": 0, "safe": 1, "disconnected": 2}[example.solution.terminal_kind]
    support: tuple[int, ...] = ()
    active = None
    mode = Mode.SEARCHING
    steps = []
    for index, event in enumerate(oracle.steps):
        before_support, before_active, before_mode = support, active, mode
        labels: dict[str, object] = {}
        if event.kind is InternalEventKind.RECALL:
            active = event.selected_record_id
            mode = Mode.HAVE_MEMORY
        elif event.kind is InternalEventKind.COMPOSE:
            fact = facts[event.selected_record_id]
            last_compose = (
                index == len(oracle.steps) - 1
                or oracle.steps[index + 1].kind is InternalEventKind.ACT
            )
            support += (event.selected_record_id,)
            active = None
            labels.update(confidence=fact.confidence, append_support=True)
            if isinstance(fact, LinkFact):
                labels.update(role=0, focus=event.focus_after, continue_search=not last_compose)
                mode = Mode.QUIESCENT if last_compose else Mode.SEARCHING
                if last_compose:
                    labels.update(status=2, action_class=4)
            elif isinstance(fact, HazardFact):
                labels.update(
                    role=1,
                    hazard_type=fact.hazard_type,
                    log_delay=math.log1p(fact.delay),
                    normalized_deadline=fact.delay / (1.0 + event.timestamp - activation_time),
                    status=0,
                )
                mode = Mode.HOLDING_HAZARD
            elif isinstance(fact, SafeFact):
                labels.update(role=2, status=1, action_class=4)
                mode = Mode.QUIESCENT
        elif event.kind is InternalEventKind.ACT:
            labels.update(action_class=terminal_class, action_lead=0.175)
            mode = Mode.QUIESCENT
        else:
            raise ValueError("teacher traces cannot synthesize NOOP or absent events")
        steps.append(
            TeacherStep(
                event.kind,
                event.timestamp,
                event.delta,
                event.selected_record_id,
                before_mode,
                mode,
                before_active,
                active,
                before_support,
                support,
                event.focus_before,
                **labels,
            )
        )
    return TeacherTrace(tuple(steps), terminal_class, terminal_status, mode, support)


def component_target(trace: TeacherTrace) -> ComponentTarget:
    compositions = tuple(step for step in trace.steps if step.kind is InternalEventKind.COMPOSE)
    return ComponentTarget(
        tuple(step.selected_record_id for step in compositions),
        tuple(
            ComponentContent(
                step.selected_record_id,
                step.role,
                step.focus,
                step.hazard_type,
                step.log_delay,
                step.normalized_deadline,
                step.status,
                step.confidence,
                step.append_support,
                step.continue_search,
            )
            for step in compositions
        ),
        trace.terminal_class,
        trace.terminal_status,
    )
