"""Strict shared validation types."""

from pydantic import BaseModel, ConfigDict

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class StrictModel(BaseModel):
    """Base model for immutable, non-coercing repository contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )
