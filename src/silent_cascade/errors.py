"""Stable typed errors for fail-loud execution."""

from collections.abc import Mapping
from typing import ClassVar

from silent_cascade.validation import JsonValue


class SilentCascadeError(Exception):
    code: ClassVar[str] = "silent_cascade_error"

    def __init__(
        self,
        message: str,
        *,
        context: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.context = dict(context or {})

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }


class ConfigurationError(SilentCascadeError):
    code = "configuration_error"


class ArtifactError(SilentCascadeError):
    code = "artifact_error"


class AtomicWriteError(ArtifactError):
    code = "atomic_write_error"


class ArtifactIntegrityError(ArtifactError):
    code = "artifact_integrity_error"


class CrashBundleError(ArtifactError):
    code = "crash_bundle_error"


class DoctorError(SilentCascadeError):
    code = "doctor_error"


class DynamicsError(SilentCascadeError):
    code = "dynamics_error"


class TimeOrderError(DynamicsError):
    code = "time_order_error"


class ProvenanceError(SilentCascadeError):
    code = "provenance_error"


class ReplayError(SilentCascadeError):
    code = "replay_error"
