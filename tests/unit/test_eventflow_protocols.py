import inspect

from silent_cascade.eventflow.protocols import AgentCondition, ClosedFormFlow


def test_agent_protocol_declares_only_public_runtime_callbacks() -> None:
    methods = {
        name
        for name, value in AgentCondition.__dict__.items()
        if inspect.isfunction(value) and not name.startswith("_")
    }

    assert methods == {
        "initialize",
        "on_external",
        "next_internal_event",
        "on_internal",
        "compute_counters",
    }
    assert tuple(inspect.signature(AgentCondition.initialize).parameters) == ("self", "init")
    assert tuple(inspect.signature(AgentCondition.on_external).parameters) == (
        "self",
        "state",
        "event",
    )
    assert tuple(inspect.signature(AgentCondition.on_internal).parameters) == (
        "self",
        "state",
        "event",
    )
    assert tuple(inspect.signature(AgentCondition.compute_counters).parameters) == ("self",)


def test_agent_protocol_does_not_receive_external_horizon() -> None:
    parameters = inspect.signature(AgentCondition.next_internal_event).parameters
    assert tuple(parameters) == ("self", "state")
    assert "external_horizon" not in parameters


def test_closed_form_flow_exposes_only_analytic_state_operations() -> None:
    methods = {
        name
        for name, value in ClosedFormFlow.__dict__.items()
        if inspect.isfunction(value) and not name.startswith("_")
    }

    assert methods == {"advance", "next_crossings"}
    assert tuple(inspect.signature(ClosedFormFlow.advance).parameters) == (
        "self",
        "state",
        "target_time",
    )
    assert tuple(inspect.signature(ClosedFormFlow.next_crossings).parameters) == (
        "self",
        "state",
    )
