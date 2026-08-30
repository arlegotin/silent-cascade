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
