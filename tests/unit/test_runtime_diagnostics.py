"""Safe crash identities must distinguish invariant failures without private text."""

import json

import pytest

from silent_cascade.errors import DynamicsError, TimeOrderError
from silent_cascade.eventflow.scheduling import selected_gap_clamp_streak
from silent_cascade.logging.crash_bundle import CrashContext, write_crash_bundle


def captured(error):
    try:
        # Deliberately identical source location for every semantic failure.
        raise error
    except DynamicsError as caught:
        return caught


def write_diagnostic(tmp_path, invariant):
    error = captured(
        DynamicsError(
            "PRIVATE-MESSAGE",
            context={"invariant": invariant, "path": "/PRIVATE-PATH", "window": "PRIVATE-WINDOW"},
        )
    )
    artifact = write_crash_bundle(
        tmp_path, error=error, context=CrashContext(), sanitize_diagnostics=True
    )
    return error, json.loads(artifact.path.read_text())


def test_same_code_distinct_invariants_survive_sanitized_publication(tmp_path):
    _, cap = write_diagnostic(tmp_path, "internal_event_cap")
    _, refractory = write_diagnostic(tmp_path, "same_kind_refractory")
    assert cap["error"]["code"] == refractory["error"]["code"] == "dynamics_error"
    assert cap["error"] != refractory["error"]
    assert cap["error"]["context"] == {
        "invariant": "internal_event_cap",
        "replay_certifiable": True,
    }
    assert refractory["error"]["context"] == {
        "invariant": "same_kind_refractory",
        "replay_certifiable": True,
    }
    assert "PRIVATE-" not in json.dumps([cap, refractory])


def test_repeated_identity_has_one_reusable_canonical_projection(tmp_path):
    from silent_cascade.logging.runtime_diagnostics import runtime_diagnostic_identity

    error, first = write_diagnostic(tmp_path, "internal_event_cap")
    _, repeated = write_diagnostic(tmp_path, "internal_event_cap")
    identity = runtime_diagnostic_identity(error)
    assert identity.replay_certifiable
    assert first["error"] == repeated["error"] == identity.to_payload()


@pytest.mark.parametrize(
    "context", [{}, {"invariant": "PRIVATE-UNKNOWN"}, {"invariant": ["internal_event_cap"]}]
)
def test_unknown_diagnostic_is_explicitly_unsuitable_for_replay_certification(tmp_path, context):
    from silent_cascade.logging.runtime_diagnostics import runtime_diagnostic_identity

    error = captured(DynamicsError("PRIVATE-MESSAGE", context={**context, "seed": "PRIVATE-SEED"}))
    identity = runtime_diagnostic_identity(error)
    assert not identity.replay_certifiable
    assert identity.to_payload()["context"] == {"invariant": None, "replay_certifiable": False}
    artifact = write_crash_bundle(
        tmp_path, error=error, context=CrashContext(), sanitize_diagnostics=True
    )
    serialized = artifact.path.read_text()
    assert json.loads(serialized)["error"] == identity.to_payload()
    assert "PRIVATE-" not in serialized


def test_error_subclass_cannot_publish_arbitrary_private_code_or_claim_classification(tmp_path):
    from silent_cascade.logging.runtime_diagnostics import runtime_diagnostic_identity

    class UnknownCallbackFailure(DynamicsError):
        code = "PRIVATE-CODE"

    error = captured(
        UnknownCallbackFailure("PRIVATE-MESSAGE", context={"invariant": "internal_event_cap"})
    )
    identity = runtime_diagnostic_identity(error)
    assert not identity.replay_certifiable
    artifact = write_crash_bundle(
        tmp_path, error=error, context=CrashContext(), sanitize_diagnostics=True
    )
    assert "PRIVATE-" not in artifact.path.read_text()


def test_typed_time_failure_keeps_distinct_code():
    from silent_cascade.logging.runtime_diagnostics import runtime_diagnostic_identity

    error = TimeOrderError("PRIVATE-MESSAGE", context={"invariant": "next_event_time"})
    assert runtime_diagnostic_identity(error).to_payload() == {
        "code": "time_order_error",
        "message": "runtime execution failed",
        "context": {"invariant": "next_event_time", "replay_certifiable": True},
    }


def test_traceback_only_names_explicitly_approved_project_frames(tmp_path):
    # This approved production check raises before using its choice argument.
    try:
        selected_gap_clamp_streak(-1, None)
    except DynamicsError as error:
        artifact = write_crash_bundle(
            tmp_path, error=error, context=CrashContext(), sanitize_diagnostics=True
        )
    payload = json.loads(artifact.path.read_text())
    assert (
        "silent_cascade.eventflow.scheduling.selected_gap_clamp_streak:"
        in payload["traceback_text"]
    )
    assert "test_traceback_only_names" not in payload["traceback_text"]
    assert str(tmp_path) not in payload["traceback_text"]


def test_unknown_filename_and_function_are_not_copied_into_safe_traceback(tmp_path):
    source = "def PRIVATE_FUNCTION():\n    raise failure\nPRIVATE_FUNCTION()\n"
    namespace = {
        "failure": DynamicsError("PRIVATE-MESSAGE", context={"invariant": "PRIVATE-UNKNOWN"})
    }
    try:
        exec(compile(source, "/PRIVATE-PATH/silent_cascade/eventflow/engine.py", "exec"), namespace)
    except DynamicsError as error:
        artifact = write_crash_bundle(
            tmp_path, error=error, context=CrashContext(), sanitize_diagnostics=True
        )
    payload = artifact.path.read_text()
    assert "PRIVATE" not in payload
    assert json.loads(payload)["traceback_text"]
