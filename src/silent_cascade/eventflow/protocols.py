"""Public agent and closed-form flow interfaces for the event engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from silent_cascade.eventflow.guards import GuardCrossing
    from silent_cascade.eventflow.state import ComputeCounters, RuntimeState
    from silent_cascade.schemas import Action, AgentInit, Condition, ExternalEvent, InternalEvent


class AgentCondition(Protocol):
    name: Condition

    def initialize(self, init: AgentInit) -> RuntimeState: ...

    def on_external(self, state: RuntimeState, event: ExternalEvent) -> RuntimeState: ...

    def next_internal_event(self, state: RuntimeState) -> InternalEvent | None: ...

    def on_internal(
        self, state: RuntimeState, event: InternalEvent
    ) -> tuple[RuntimeState, list[Action]]: ...

    def compute_counters(self) -> ComputeCounters: ...


class ClosedFormFlow(Protocol):
    def advance(self, state: RuntimeState, target_time: float) -> RuntimeState: ...

    def next_crossings(self, state: RuntimeState) -> tuple[GuardCrossing, ...]: ...
