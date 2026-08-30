import math

import pytest

from silent_cascade.errors import (
    ArtifactError,
    AtomicWriteError,
    DynamicsError,
    SilentCascadeError,
    TimeOrderError,
)


def test_typed_error_payload_has_stable_code_message_and_context() -> None:
    error = AtomicWriteError("replace failed", context={"path": "artifact.json"})
    assert error.to_payload() == {
        "code": "atomic_write_error",
        "message": "replace failed",
        "context": {"path": "artifact.json"},
    }


def test_error_hierarchy_supports_precise_and_family_catches() -> None:
    assert issubclass(AtomicWriteError, ArtifactError)
    assert issubclass(ArtifactError, SilentCascadeError)
    assert issubclass(TimeOrderError, DynamicsError)


def test_typed_error_rejects_non_json_context_values() -> None:
    with pytest.raises(TypeError):
        AtomicWriteError("invalid", context={"value": object()})


def test_typed_error_rejects_nested_nonfinite_values_and_non_string_keys() -> None:
    with pytest.raises(ValueError):
        AtomicWriteError("invalid", context={"items": [{"value": math.inf}]})
    with pytest.raises(TypeError):
        AtomicWriteError("invalid", context={"items": {1: "value"}})
