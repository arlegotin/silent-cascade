"""Behavioral contracts for OFD action scoring and diagnostic random policy."""

import inspect
import math
from typing import get_type_hints

import numpy as np
import pytest

from silent_cascade.env.config import OracleTimingConfig, SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    EpisodeKey,
    EpisodeRecipe,
    EpisodeTruth,
    EpisodeVariant,
    MatchedEpisodeCoordinate,
    PublicEpisode,
)
from silent_cascade.env.reward import (
    RANDOM_BASELINE_BALANCED_SUCCESS_PROBABILITY,
    RANDOM_BASELINE_NEGATIVE_SUCCESS_PROBABILITY,
    RANDOM_BASELINE_POSITIVE_SUCCESS_PROBABILITY,
    action_window,
    random_baseline_actions,
    score_actions,
)
from silent_cascade.errors import ScoringError
from silent_cascade.schemas import (
    Action,
    ActivationPayload,
    AgentInit,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
)


def positive_truth() -> EpisodeTruth:
    """Return hand-authored private truth for a 24-second positive episode."""
    return EpisodeTruth(
        key=EpisodeKey(
            "ofd-v1",
            SplitNamespace.PHASE1_GATE,
            SuiteName.IID_PRIMARY,
            71,
            MatchedEpisodeCoordinate("matched", 1, 0),
        ),
        recipe=EpisodeRecipe(1, EpisodeVariant.POSITIVE, 0, SuiteName.IID_PRIMARY, 0),
        relevant_node_path=(1, 2),
        relevant_record_ids=(10, 11),
        terminal_record_id=11,
        relevant_hazard_type=2,
        private_terminal=ExternalEvent(99, 124.0, ExternalEventKind.OUTCOME, None),
        activation_time=100.0,
        episode_delay=24.0,
        action_window_start=118.0,
        action_window_end=121.6,
        action_target=119.8,
        rejection_count=0,
        rejection_reasons=(),
    )


def negative_truth() -> EpisodeTruth:
    """Return hand-authored private truth for a 24-second negative episode."""
    positive = positive_truth()
    return EpisodeTruth(
        key=positive.key,
        recipe=EpisodeRecipe(1, EpisodeVariant.SAFE_NEGATIVE, 0, SuiteName.IID_PRIMARY, 0),
        relevant_node_path=positive.relevant_node_path,
        relevant_record_ids=positive.relevant_record_ids,
        terminal_record_id=positive.terminal_record_id,
        relevant_hazard_type=None,
        private_terminal=ExternalEvent(99, 124.0, ExternalEventKind.END, None),
        activation_time=positive.activation_time,
        episode_delay=positive.episode_delay,
        action_window_start=None,
        action_window_end=None,
        action_target=None,
        rejection_count=0,
        rejection_reasons=(),
    )


def public_episode(*, hazard_delays: tuple[float, float] = (24.0, 24.0)) -> PublicEpisode:
    """Return a public episode with the diagnostic policy's two hazard facts."""
    return PublicEpisode(
        init=AgentInit("00000000-0000-4000-8000-000000000004", 64, 4, 0.0),
        events=(
            ExternalEvent(10, 1.0, ExternalEventKind.FACT, HazardFact(2, 1, hazard_delays[0])),
            ExternalEvent(11, 2.0, ExternalEventKind.FACT, HazardFact(3, 3, hazard_delays[1])),
            ExternalEvent(12, 100.0, ExternalEventKind.ACTIVATE, ActivationPayload(1)),
        ),
    )


def correct_action() -> Action:
    return Action(2, 119.8, 12)


def wrong_class_action() -> Action:
    return Action(1, 119.8, 12)


def early_action() -> Action:
    return Action(2, 117.99999999999999, 12)


def late_action() -> Action:
    return Action(2, 121.6, 12)


def unchecked_action(timestamp: float) -> Action:
    """Construct an invalid action solely to test scorer-side finite-time validation."""
    action = object.__new__(Action)
    object.__setattr__(action, "hazard_type", 2)
    object.__setattr__(action, "timestamp", timestamp)
    object.__setattr__(action, "caused_by_event_id", 12)
    return action


def test_action_window_is_half_open() -> None:
    window = action_window(100.0, 24.0, OracleTimingConfig())

    assert (window.start, window.end, window.target) == (118.0, 121.6, 119.8)
    assert window.contains(window.start)
    assert window.contains(math.nextafter(window.end, -math.inf))
    assert not window.contains(window.end)


@pytest.mark.parametrize(
    "actions",
    [
        (),
        (wrong_class_action(),),
        (early_action(),),
        (late_action(),),
        (correct_action(), correct_action()),
    ],
)
def test_positive_requires_one_correct_in_window_action(actions: tuple[Action, ...]) -> None:
    assert not score_actions(positive_truth(), actions).timed_success


def test_positive_score_records_success_for_one_correct_target_action() -> None:
    score = score_actions(positive_truth(), (correct_action(),))

    assert score.timed_success
    assert (score.is_positive, score.action_count, score.correct_class, score.in_window) == (
        True,
        1,
        True,
        True,
    )
    assert not score.false_action
    assert score.reason == "success"


@pytest.mark.parametrize("timestamp", [121.6, 124.0])
def test_positive_rejects_right_window_boundary_and_outcome_time(timestamp: float) -> None:
    score = score_actions(positive_truth(), (Action(2, timestamp, 12),))

    assert not score.timed_success
    assert score.in_window is False
    assert score.reason == "incorrect"


def test_negative_requires_abstention() -> None:
    abstention = score_actions(negative_truth(), ())
    false_action = score_actions(negative_truth(), (correct_action(),))

    assert abstention.timed_success
    assert abstention.reason == "negative_abstention"
    assert not false_action.timed_success
    assert false_action.false_action
    assert false_action.reason == "negative_false_action"


def test_scorer_rejects_actions_before_activation() -> None:
    with pytest.raises(ScoringError, match="precedes activation"):
        score_actions(positive_truth(), (Action(2, 99.99999999999999, 12),))


@pytest.mark.parametrize("timestamp", [math.inf, -math.inf, math.nan])
def test_scorer_rejects_nonfinite_action_times(timestamp: float) -> None:
    with pytest.raises(ScoringError, match="must be finite"):
        score_actions(positive_truth(), (unchecked_action(timestamp),))


def test_random_baseline_is_public_only_and_uses_no_private_truth_parameter() -> None:
    signature = inspect.signature(random_baseline_actions)
    hints = get_type_hints(random_baseline_actions)

    assert tuple(signature.parameters) == ("public", "timing", "rng")
    assert hints["public"] is PublicEpisode
    assert "truth" not in signature.parameters


def test_random_baseline_requires_two_equal_public_hazard_delays() -> None:
    with pytest.raises(ScoringError, match="two equal public hazard delays"):
        random_baseline_actions(
            public_episode(hazard_delays=(24.0, 25.0)),
            OracleTimingConfig(),
            np.random.default_rng(9),
        )


def test_random_baseline_seeded_abstention_class_and_timestamp_distribution() -> None:
    public = public_episode()
    rng = np.random.default_rng(20260831)
    window = action_window(100.0, 24.0, OracleTimingConfig())
    abstentions = 0
    class_counts = [0, 0, 0, 0]

    for _ in range(10_000):
        actions = random_baseline_actions(public, OracleTimingConfig(), rng)
        if not actions:
            abstentions += 1
            continue
        action = actions[0]
        class_counts[action.hazard_type] += 1
        assert window.start <= action.timestamp < window.end
        assert action.caused_by_event_id == 12

    assert abstentions == 4_960
    assert class_counts == [1_291, 1_275, 1_245, 1_229]


def test_random_baseline_analytic_success_probabilities() -> None:
    assert RANDOM_BASELINE_POSITIVE_SUCCESS_PROBABILITY == 0.5 * 0.25 == 0.125
    assert RANDOM_BASELINE_NEGATIVE_SUCCESS_PROBABILITY == 0.5
    assert RANDOM_BASELINE_BALANCED_SUCCESS_PROBABILITY == 0.3125
