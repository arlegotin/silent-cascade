"""Typed failures raised by neural model contracts."""

from silent_cascade.errors import SilentCascadeError


class NeuralError(SilentCascadeError):
    """A bounded neural input, state, or output contract was violated."""

    code = "neural_error"
