"""Stable typed errors for fail-loud execution."""

import math
from collections.abc import Mapping
from typing import ClassVar

from silent_cascade.validation import JsonValue


def _validated_json_value(value: object, *, path: str = "context") -> JsonValue:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain finite numbers")
        return value
    if isinstance(value, list):
        return [
            _validated_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            result[key] = _validated_json_value(item, path=f"{path}.{key}")
        return result
    raise TypeError(f"{path} contains a non-JSON value: {type(value).__name__}")


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
        self.context = _validated_json_value(dict(context or {}))

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
