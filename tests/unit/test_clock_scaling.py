"""Contracts for immutable paired OFD clock transforms."""

from dataclasses import replace
from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    EpisodeVariant,
    canonical_episode_bytes,
    episode_from_bytes,
    episode_sha256,
    scale_episode_time,
)
from silent_cascade.env.generator import IndependentEpisodeRequest, generate_independent_episode
from silent_cascade.env.oracle import solve_public_episode
from silent_cascade.errors import EpisodeInvariantError
from silent_cascade.schemas import HazardFact


def _config() -> Phase1Config:
    return resolve_config(
        Phase1Config,
        [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
    ).config


def _parent_bundle():
    request = IndependentEpisodeRequest(
        SplitNamespace.PHASE1_GATE,
        SuiteName.IID_PRIMARY,
        7,
        3,
        2,
        EpisodeVariant.POSITIVE,
        0,
    )
    return generate_independent_episode(_config(), request, public_id_seed=91)


@pytest.mark.parametrize(
    ("target_suite", "factor"),
    (
        (SuiteName.CLOCK_SCALE_0_1X, 0.1),
        (SuiteName.CLOCK_SCALE_10X, 10.0),
    ),
)
def test_paired_clock_transform_preserves_source_semantics_and_scales_time(
    target_suite: SuiteName, factor: float
) -> None:
    """Missing a time field or changing source semantics breaks paired comparison."""
    parent = _parent_bundle()
    parent_bytes = canonical_episode_bytes(parent)
    child_id = "10000000-0000-4000-8000-000000000001"

    child = scale_episode_time(parent, target_suite, child_id)

    assert canonical_episode_bytes(parent) == parent_bytes
    assert episode_from_bytes(canonical_episode_bytes(child)) == child
    assert child.public.init.episode_public_id == child_id
    assert child.truth.key == parent.truth.key
    assert child.truth.recipe.evaluation_suite is target_suite
    assert child.truth.recipe.clock_scale == factor
    assert child.truth.recipe.parent_public_id == parent.public.init.episode_public_id
    assert child.truth.recipe.parent_episode_sha256 == episode_sha256(parent)
    assert child.truth.relevant_node_path == parent.truth.relevant_node_path
    assert child.truth.relevant_record_ids == parent.truth.relevant_record_ids
    assert child.truth.terminal_record_id == parent.truth.terminal_record_id
    assert child.truth.relevant_hazard_type == parent.truth.relevant_hazard_type
    assert child.truth.recipe.variant is parent.truth.recipe.variant
    assert child.truth.recipe.distractor_link_count == parent.truth.recipe.distractor_link_count
    assert child.truth.recipe.accepted_attempt == parent.truth.recipe.accepted_attempt
    assert child.truth.rejection_count == parent.truth.rejection_count
    assert child.truth.rejection_reasons == parent.truth.rejection_reasons
    parent_solution = solve_public_episode(parent.public)
    child_solution = solve_public_episode(child.public)
    assert (
        child_solution.terminal_kind,
        child_solution.node_path,
        child_solution.link_record_ids,
        child_solution.terminal_record_id,
        child_solution.hazard_type,
        child_solution.superseded_terminal_record_ids,
    ) == (
        parent_solution.terminal_kind,
        parent_solution.node_path,
        parent_solution.link_record_ids,
        parent_solution.terminal_record_id,
        parent_solution.hazard_type,
        parent_solution.superseded_terminal_record_ids,
    )
    assert child_solution.public_delay == parent_solution.public_delay * factor
    assert child.public.init.initial_time == parent.public.init.initial_time * factor
    assert [event.timestamp for event in child.public.events] == [
        event.timestamp * factor for event in parent.public.events
    ]
    assert child.truth.activation_time == parent.truth.activation_time * factor
    assert child.truth.episode_delay == parent.truth.episode_delay * factor
    assert child.truth.private_terminal.timestamp == (
        parent.truth.private_terminal.timestamp * factor
    )
    assert child.truth.action_window_start == parent.truth.action_window_start * factor
    assert child.truth.action_window_end == parent.truth.action_window_end * factor
    assert child.truth.action_target == parent.truth.action_target * factor
    assert child.truth.recipe.oracle_timing.delta_0 == (
        parent.truth.recipe.oracle_timing.delta_0 * factor
    )
    assert child.truth.recipe.oracle_timing.delta_min == (
        parent.truth.recipe.oracle_timing.delta_min * factor
    )
    assert child.truth.recipe.oracle_timing.delta_max == (
        parent.truth.recipe.oracle_timing.delta_max * factor
    )
    assert [
        event.payload.delay
        for event in child.public.events
        if isinstance(event.payload, HazardFact)
    ] == [
        event.payload.delay * factor
        for event in parent.public.events
        if isinstance(event.payload, HazardFact)
    ]
    child_start_ratio = (
        child.truth.action_window_start - child.truth.activation_time
    ) / child.truth.episode_delay
    assert child_start_ratio == pytest.approx(
        (parent.truth.action_window_start - parent.truth.activation_time)
        / parent.truth.episode_delay
    )
    child_target_ratio = (
        child.truth.action_target - child.truth.activation_time
    ) / child.truth.episode_delay
    assert child_target_ratio == pytest.approx(
        (parent.truth.action_target - parent.truth.activation_time) / parent.truth.episode_delay
    )
    child_end_ratio = (
        child.truth.action_window_end - child.truth.activation_time
    ) / child.truth.episode_delay
    assert child_end_ratio == pytest.approx(
        (parent.truth.action_window_end - parent.truth.activation_time) / parent.truth.episode_delay
    )


def test_paired_clock_transform_rejects_non_iid_or_already_scaled_parent() -> None:
    """Accepting derived or non-IID parents would resample a semantic source."""
    parent = _parent_bundle()
    child = scale_episode_time(
        parent,
        SuiteName.CLOCK_SCALE_0_1X,
        "10000000-0000-4000-8000-000000000001",
    )

    with pytest.raises(EpisodeInvariantError, match="unscaled IID"):
        scale_episode_time(
            child,
            SuiteName.CLOCK_SCALE_10X,
            "10000000-0000-4000-8000-000000000002",
        )
    non_iid = replace(
        parent,
        truth=replace(
            parent.truth,
            key=replace(parent.truth.key, suite=SuiteName.OOD_LONG_DELAY),
            recipe=replace(parent.truth.recipe, evaluation_suite=SuiteName.OOD_LONG_DELAY),
        ),
    )
    with pytest.raises(EpisodeInvariantError, match="unscaled IID"):
        scale_episode_time(
            non_iid,
            SuiteName.CLOCK_SCALE_10X,
            "10000000-0000-4000-8000-000000000002",
        )


@pytest.mark.parametrize(
    "target_suite",
    (SuiteName.IID_PRIMARY, SuiteName.OOD_LONG_DELAY, "clock_scale_10x"),
)
def test_paired_clock_transform_rejects_non_clock_target(target_suite: object) -> None:
    """Changing the suite-to-factor mapping must fail instead of silently choosing a scale."""
    with pytest.raises((TypeError, ValueError), match="clock"):
        scale_episode_time(
            _parent_bundle(),
            target_suite,  # type: ignore[arg-type]
            "10000000-0000-4000-8000-000000000001",
        )


def test_paired_clock_transform_rejects_scaled_nonfinite_time() -> None:
    """Overflow must not create an artifact with nonfinite timing provenance."""
    parent = _parent_bundle()
    delay = 1.0e307
    activation = 1.0e307
    fact_count = len(parent.public.events) - 1
    public = replace(
        parent.public,
        events=tuple(
            replace(
                event,
                timestamp=(
                    (index + 1) * activation / (fact_count + 1)
                    if event.kind.value == "fact"
                    else activation
                ),
                payload=(
                    replace(event.payload, delay=delay)
                    if isinstance(event.payload, HazardFact)
                    else event.payload
                ),
            )
            for index, event in enumerate(parent.public.events)
        ),
    )
    overflowing = replace(
        parent,
        public=public,
        truth=replace(
            parent.truth,
            private_terminal=replace(parent.truth.private_terminal, timestamp=activation + delay),
            activation_time=activation,
            episode_delay=delay,
            action_window_start=activation + 0.75 * delay,
            action_window_end=activation + 0.90 * delay,
            action_target=activation + 0.825 * delay,
        ),
    )

    with pytest.raises(EpisodeInvariantError, match="nonfinite"):
        scale_episode_time(
            overflowing,
            SuiteName.CLOCK_SCALE_10X,
            "10000000-0000-4000-8000-000000000001",
        )
