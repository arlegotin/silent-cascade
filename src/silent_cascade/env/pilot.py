"""Private, authenticated curriculum projections for the ordinary event engine.

The retained generator key names the original parent, not the transformed
episode. The Phase 4 manifest binds that parent and the separate projection.
"""

import math
from dataclasses import replace

import numpy as np

from silent_cascade.env.config import OracleTimingConfig, SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    EpisodeBundle,
    EpisodeRecipe,
    EpisodeTruth,
    PublicEpisode,
    episode_sha256,
)
from silent_cascade.env.generator import IndependentEpisodeRequest, generate_independent_episode
from silent_cascade.env.oracle import build_oracle_trace, solve_public_episode, verify_oracle_truth
from silent_cascade.env.timing import action_window
from silent_cascade.errors import EpisodeInvariantError
from silent_cascade.schemas import HazardFact
from silent_cascade.train.curriculum_data import (
    MAX_PARENT_ATTEMPTS,
    CurriculumExample,
    curriculum_allocation,
    curriculum_rng,
    make_curriculum_example,
)
from silent_cascade.train.pilot_config import Phase4Config


def curriculum_to_bundle(example: CurriculumExample, *, config: Phase4Config) -> EpisodeBundle:
    """Regenerate the parent and transform before constructing private truth."""
    if not isinstance(example, CurriculumExample) or not isinstance(config, Phase4Config):
        raise TypeError("expected CurriculumExample and Phase4Config")
    root, path, variant = curriculum_allocation(example.key)
    request = IndependentEpisodeRequest(
        SplitNamespace(example.key.split),
        SuiteName.IID_PRIMARY,
        root,
        example.key.episode_index * MAX_PARENT_ATTEMPTS + example.accepted_attempt,
        path,
        variant,
        example.key.episode_index // 4,
        example.key.episode_index % 4,
    )
    parent = generate_independent_episode(config, request, example.key.public_id_seed)
    if episode_sha256(parent) != example.parent_hash:
        raise EpisodeInvariantError("pilot parent identity mismatch")
    if make_curriculum_example(config, example.key) != example:
        raise EpisodeInvariantError("pilot curriculum identity/hash mismatch")
    solution = solve_public_episode(example.public)
    timing_values = config.data.oracle_timing.model_dump()
    for name in ("delta_0", "delta_min", "delta_max"):
        timing_values[name] *= example.time_factor
    timing = OracleTimingConfig(**timing_values)
    delay = example.terminal_horizon.timestamp - example.public.events[-1].timestamp
    window = (
        action_window(example.public.events[-1].timestamp, delay, timing)
        if solution.hazard_type is not None
        else None
    )
    truth = EpisodeTruth(
        key=parent.truth.key,
        recipe=EpisodeRecipe(
            requested_path_length=len(solution.link_record_ids),
            variant=variant,
            distractor_link_count=example.nuisance_link_count,
            evaluation_suite=SuiteName.IID_PRIMARY,
            accepted_attempt=parent.truth.recipe.accepted_attempt,
            oracle_timing=timing,
        ),
        relevant_node_path=solution.node_path,
        relevant_record_ids=solution.link_record_ids
        + (() if solution.terminal_record_id is None else (solution.terminal_record_id,)),
        terminal_record_id=solution.terminal_record_id,
        relevant_hazard_type=solution.hazard_type,
        private_terminal=example.terminal_horizon,
        activation_time=example.public.events[-1].timestamp,
        episode_delay=delay,
        action_window_start=None if window is None else window.start,
        action_window_end=None if window is None else window.end,
        action_target=None if window is None else window.target,
        rejection_count=parent.truth.rejection_count,
        rejection_reasons=parent.truth.rejection_reasons,
    )
    bundle = EpisodeBundle(example.public, truth)
    verify_oracle_truth(solution, truth)
    trace = build_oracle_trace(
        bundle.public,
        solution,
        truth.private_terminal,
        timing,
        curriculum_rng(example.key, "trace_jitter", example.accepted_attempt),
    )
    if trace != example.oracle_trace:
        raise EpisodeInvariantError("pilot feasible trace identity mismatch")
    return bundle


def delay_pair(
    bundle: EpisodeBundle, *, delays: tuple[float, float]
) -> tuple[EpisodeBundle, EpisodeBundle]:
    """Two interventions; each changes both hazard delays and private scoring.

    Public IDs are retained for pairing. Consumers bind each child with
    EpisodeBinding.from_bundle and a DelayTransform naming this input hash.
    """
    if not isinstance(bundle, EpisodeBundle):
        raise TypeError("delay pair requires an EpisodeBundle")
    if (
        type(delays) is not tuple
        or len(delays) != 2
        or any(
            type(delay) is not float or not math.isfinite(delay) or delay <= 0 for delay in delays
        )
        or delays[0] == delays[1]
    ):
        raise ValueError("delay pair requires two distinct positive finite floats")
    if sum(isinstance(e.payload, HazardFact) for e in bundle.public.events) != 2:
        raise EpisodeInvariantError("delay pair requires exactly two public hazards")
    verify_oracle_truth(solve_public_episode(bundle.public), bundle.truth)
    children = []
    for delay in delays:
        public = PublicEpisode(
            bundle.public.init,
            tuple(
                replace(e, payload=replace(e.payload, delay=delay))
                if isinstance(e.payload, HazardFact)
                else e
                for e in bundle.public.events
            ),
        )
        solution = solve_public_episode(public)
        timing = bundle.truth.recipe.oracle_timing
        window = (
            action_window(bundle.truth.activation_time, delay, timing)
            if solution.hazard_type is not None
            else None
        )
        truth = replace(
            bundle.truth,
            private_terminal=replace(
                bundle.truth.private_terminal, timestamp=bundle.truth.activation_time + delay
            ),
            episode_delay=delay,
            action_window_start=None if window is None else window.start,
            action_window_end=None if window is None else window.end,
            action_target=None if window is None else window.target,
        )
        child = EpisodeBundle(public, truth)
        verify_oracle_truth(solution, truth)
        build_oracle_trace(
            public, solution, truth.private_terminal, timing, np.random.default_rng(0)
        )
        children.append(child)
    return children[0], children[1]
