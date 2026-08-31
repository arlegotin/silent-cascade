"""Shared private action-window timing derived from the active OFD contract."""

from dataclasses import dataclass

from silent_cascade.env.config import OracleTimingConfig


@dataclass(frozen=True, slots=True)
class ActionWindow:
    """A target action interval whose right boundary is excluded."""

    start: float
    end: float
    target: float

    def contains(self, timestamp: float) -> bool:
        """Return whether ``timestamp`` lies in this half-open interval."""
        return self.start <= timestamp < self.end


def action_window(
    activation_time: float,
    delay: float,
    timing: OracleTimingConfig,
) -> ActionWindow:
    """Derive the configured action window from public activation and delay."""
    if not isinstance(timing, OracleTimingConfig):
        raise TypeError("timing must be an OracleTimingConfig")
    return ActionWindow(
        start=activation_time + timing.action_window_start_fraction * delay,
        end=activation_time + timing.action_window_end_fraction * delay,
        target=activation_time + timing.action_target_fraction * delay,
    )
