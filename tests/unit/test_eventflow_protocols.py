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
        "dt",
    )
    assert tuple(inspect.signature(ClosedFormFlow.next_crossings).parameters) == (
        "self",
        "state",
    )


def test_closed_form_flow_uses_the_canonical_continuous_state_contract() -> None:
    advance = inspect.signature(ClosedFormFlow.advance)
    crossings = inspect.signature(ClosedFormFlow.next_crossings)

    assert advance.parameters["state"].annotation == "ContinuousState"
    assert advance.parameters["dt"].annotation == "float"
    assert advance.return_annotation == "ContinuousState"
    assert crossings.parameters["state"].annotation == "ContinuousState"
    assert crossings.return_annotation == "list[GuardCrossing]"
