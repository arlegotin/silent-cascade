"""Acceptance-scale regressions for exact Phase 1 oracle trace feasibility."""

import math
from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SuiteName
from silent_cascade.env.episode import EpisodeVariant
from silent_cascade.env.generator import (
    generate_independent_episode,
    iter_phase1_gate_requests,
)
from silent_cascade.env.oracle import build_oracle_trace, solve_public_episode
from silent_cascade.env.reward import score_actions
from silent_cascade.rng import (
    IndependentCounterSeedKey,
    SeedStream,
    independent_local_generator,
)
from silent_cascade.schemas import InternalEventKind

_GATE_ROOT_SEED = 2026083011
_GATE_PUBLIC_ID_SEED = 2026083012


@pytest.fixture(scope="module")
def config() -> Phase1Config:
    return resolve_config(
        Phase1Config,
        [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
    ).config


def _trace_rng(bundle: object):
    truth = bundle.truth  # type: ignore[union-attr]
    coordinate = truth.key.coordinate
    return independent_local_generator(
        IndependentCounterSeedKey(
            "ofd-v1",
            truth.key.split_namespace,
            truth.key.suite,
            truth.key.root_seed,
            coordinate.episode_index,
            SeedStream.TRACE_JITTER,
            truth.recipe.accepted_attempt,
        )
    )


def test_all_frozen_ood_short_draws_build_exact_authenticated_oracle_traces(
    config: Phase1Config,
) -> None:
    """The old minimum-count proxy accepted 47 jitter-infeasible short-delay traces."""
    count = 0
    exact_regressions: dict[int, tuple[EpisodeVariant, int]] = {}
    for request in iter_phase1_gate_requests(_GATE_ROOT_SEED):
        if request.suite is not SuiteName.OOD_SHORT_DELAY:
            continue
        bundle = generate_independent_episode(config, request, _GATE_PUBLIC_ID_SEED)
        trace = build_oracle_trace(
            bundle.public,
            solve_public_episode(bundle.public),
            bundle.truth.private_terminal,
            bundle.truth.recipe.oracle_timing,
            _trace_rng(bundle),
        )
        non_action_steps = tuple(
            step for step in trace.steps if step.kind is not InternalEventKind.ACT
        )
        trace_delay = bundle.truth.private_terminal.timestamp - bundle.truth.activation_time
        prefix_deltas: list[float] = []
        for step in non_action_steps:
            prefix_deltas.append(step.delta)
            assert (
                bundle.truth.recipe.oracle_timing.delta_min
                <= step.delta
                <= bundle.truth.recipe.oracle_timing.delta_max
            )
            assert step.timestamp == math.fsum((bundle.truth.activation_time, *prefix_deltas))
        assert non_action_steps[-1].timestamp <= math.fsum(
            (
                bundle.truth.activation_time,
                bundle.truth.recipe.oracle_timing.terminal_compose_fraction * trace_delay,
            )
        )
        if bundle.truth.recipe.variant is EpisodeVariant.POSITIVE:
            expected_target = math.fsum(
                (
                    bundle.truth.activation_time,
                    bundle.truth.recipe.oracle_timing.action_target_fraction * trace_delay,
                )
            )
            assert trace.steps[-1].kind is InternalEventKind.ACT
            assert trace.steps[-1].timestamp == expected_target
            assert trace.steps[-1].delta == (expected_target - non_action_steps[-1].timestamp)
            assert trace.actions[0].timestamp == expected_target
        else:
            assert trace.steps == non_action_steps
            assert trace.actions == ()
        assert score_actions(bundle.truth, trace.actions).timed_success
        count += 1
        if request.episode_index in {16219, 16500}:
            exact_regressions[request.episode_index] = (
                bundle.truth.recipe.variant,
                bundle.truth.recipe.accepted_attempt,
            )

    assert count == 24_000
    assert exact_regressions == {
        16219: (EpisodeVariant.SAFE_NEGATIVE, 1),
        16500: (EpisodeVariant.POSITIVE, 1),
    }
