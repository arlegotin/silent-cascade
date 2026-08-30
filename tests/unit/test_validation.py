import math

import pytest
from pydantic import ValidationError

from silent_cascade.validation import StrictModel


class ExampleModel(StrictModel):
    count: int
    value: float


def test_strict_model_rejects_unknown_fields_and_coercion() -> None:
    with pytest.raises(ValidationError):
        ExampleModel.model_validate({"count": 1, "value": 2.0, "extra": True})
    with pytest.raises(ValidationError):
        ExampleModel.model_validate({"count": "1", "value": 2.0})


def test_strict_model_is_frozen_and_rejects_nonfinite_values() -> None:
    model = ExampleModel(count=1, value=2.0)
    with pytest.raises(ValidationError):
        model.count = 2
    with pytest.raises(ValidationError):
        ExampleModel(count=1, value=math.nan)
