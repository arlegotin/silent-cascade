"""Half-open OFD scoring and the public-only diagnostic random policy."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from silent_cascade.env.config import OracleTimingConfig
from silent_cascade.env.episode import EpisodeTruth, EpisodeVariant, PublicEpisode
from silent_cascade.env.timing import action_window
from silent_cascade.errors import ScoringError
from silent_cascade.schemas import Action, ExternalEventKind, HazardFact

RANDOM_BASELINE_POSITIVE_SUCCESS_PROBABILITY = 0.125
RANDOM_BASELINE_NEGATIVE_SUCCESS_PROBABILITY = 0.5
RANDOM_BASELINE_BALANCED_SUCCESS_PROBABILITY = 0.3125


@dataclass(frozen=True, slots=True)
class EpisodeScore:
    """The private scorer's result for one episode action sequence."""

    timed_success: bool
    is_positive: bool
    action_count: int
    correct_class: bool | None
    in_window: bool | None
    false_action: bool
    reason: str


def score_actions(truth: EpisodeTruth, actions: Sequence[Action]) -> EpisodeScore:
    """Score actions against private episode truth with exact half-open semantics."""
    if any(not math.isfinite(action.timestamp) for action in actions):
        raise ScoringError("action timestamp must be finite")
    if any(action.timestamp < truth.activation_time for action in actions):
        raise ScoringError("action precedes activation")
    if truth.recipe.variant is not EpisodeVariant.POSITIVE:
        return EpisodeScore(
            timed_success=not actions,
            is_positive=False,
            action_count=len(actions),
            correct_class=None,
            in_window=None,
            false_action=bool(actions),
            reason="negative_abstention" if not actions else "negative_false_action",
        )
    if len(actions) != 1:
        return EpisodeScore(
            timed_success=False,
            is_positive=True,
            action_count=len(actions),
            correct_class=None,
            in_window=None,
            false_action=False,
            reason="action_count",
        )
    action = actions[0]
    if (
        truth.relevant_hazard_type is None
        or truth.action_window_start is None
        or truth.action_window_end is None
    ):
        raise ScoringError("positive episode is missing scorer truth")
    correct_class = action.hazard_type == truth.relevant_hazard_type
    in_window = truth.action_window_start <= action.timestamp < truth.action_window_end
    return EpisodeScore(
        timed_success=correct_class and in_window,
        is_positive=True,
        action_count=1,
        correct_class=correct_class,
        in_window=in_window,
        false_action=False,
        reason="success" if correct_class and in_window else "incorrect",
    )


def random_baseline_actions(
    public: PublicEpisode,
    timing: OracleTimingConfig,
    rng: np.random.Generator,
) -> tuple[Action, ...]:
    """Sample the diagnostic policy using only agent-visible episode fields."""
    activation = public.events[-1]
    if activation.kind is not ExternalEventKind.ACTIVATE:
        raise ScoringError("public episode has no final activation")
    delays = [
        event.payload.delay
        for event in public.events
        if event.kind is ExternalEventKind.FACT and isinstance(event.payload, HazardFact)
    ]
    if len(delays) != 2 or delays[0] != delays[1]:
        raise ScoringError("random baseline requires two equal public hazard delays")
    if float(rng.random()) < 0.5:
        return ()
    window = action_window(activation.timestamp, delays[0], timing)
    return (
        Action(
            hazard_type=int(rng.integers(0, public.init.hazard_type_count)),
            timestamp=float(rng.uniform(window.start, window.end)),
            caused_by_event_id=activation.event_id,
        ),
    )
