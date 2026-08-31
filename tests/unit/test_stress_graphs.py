"""Structural contracts for the Phase 1 branching and irrelevant-cycle suites."""

from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    EpisodeBundle,
    EpisodeVariant,
    StressMetadata,
    canonical_episode_bytes,
)
from silent_cascade.env.generator import IndependentEpisodeRequest, independent_seed_tokens
from silent_cascade.env.oracle import OraclePolicy, OracleTerminalKind, solve_public_episode
from silent_cascade.errors import EpisodeInvariantError, OracleError
from silent_cascade.schemas import HazardFact, LinkFact, SafeFact


@pytest.fixture
def config() -> Phase1Config:
    return resolve_config(
        Phase1Config,
        [
            Path("configs/base.yaml"),
            Path("configs/data/primary.yaml"),
            Path("configs/data/stress.yaml"),
        ],
    ).config


def _request(suite: SuiteName, episode_index: int = 17) -> IndependentEpisodeRequest:
    variant = (
        EpisodeVariant.DISCONNECTED_NEGATIVE
        if suite is SuiteName.NULL_NEAR_MISS_STRESS
        else EpisodeVariant.POSITIVE
    )
    return IndependentEpisodeRequest(
        split_namespace=SplitNamespace.DEBUG,
        suite=suite,
        root_seed=20260831,
        episode_index=episode_index,
        requested_path_length=3,
        variant=variant,
        allocation_quartet_index=4,
    )


def _generate(config: Phase1Config, request: IndependentEpisodeRequest) -> EpisodeBundle:
    import silent_cascade.env.generator as generator

    generate_stress_episode = getattr(generator, "generate_stress_episode", None)
    assert callable(generate_stress_episode), "structural stress generation API is missing"
    return generate_stress_episode(config, request, public_id_seed=91)


def _activation_reachable_nodes(bundle: EpisodeBundle) -> set[int]:
    links: dict[int, list[int]] = defaultdict(list)
    for event in bundle.public.events[:-1]:
        if isinstance(event.payload, LinkFact):
            links[event.payload.source_node].append(event.payload.target_node)
    start = bundle.public.events[-1].payload.start_node
    reachable = {start}
    pending = [start]
    while pending:
        node = pending.pop()
        for target in links[node]:
            if target not in reachable:
                reachable.add(target)
                pending.append(target)
    return reachable


def _directed_cycles(bundle: EpisodeBundle) -> tuple[tuple[int, ...], ...]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for event in bundle.public.events[:-1]:
        if isinstance(event.payload, LinkFact):
            adjacency[event.payload.source_node].append(event.payload.target_node)
    cycles: set[tuple[int, ...]] = set()

    def visit(node: int, path: tuple[int, ...]) -> None:
        for target in adjacency[node]:
            if target in path:
                cycle = path[path.index(target) :]
                rotations = tuple(cycle[index:] + cycle[:index] for index in range(len(cycle)))
                cycles.add(min(rotations))
            else:
                visit(target, (*path, target))

    for node in tuple(adjacency):
        visit(node, (node,))
    return tuple(sorted(cycles))


def _replace_fact(
    bundle: EpisodeBundle,
    record_id: int,
    payload: LinkFact | HazardFact | SafeFact,
) -> EpisodeBundle:
    events = tuple(
        replace(event, payload=payload) if event.event_id == record_id else event
        for event in bundle.public.events
    )
    return EpisodeBundle(replace(bundle.public, events=events), bundle.truth)


def _assert_invalid(bundle: EpisodeBundle, config: Phase1Config) -> None:
    from silent_cascade.env.invariants import validate_episode_invariants

    with pytest.raises(EpisodeInvariantError):
        validate_episode_invariants(bundle, config)


def test_branching_suite_has_one_hazard_path_and_bounded_terminal_free_branches(
    config: Phase1Config,
) -> None:
    """Removing branch isolation would create another solution or wrong branch count."""
    bundle = _generate(config, _request(SuiteName.BRANCHING_STRESS))
    solution = solve_public_episode(bundle.public, OraclePolicy.BRANCHING)
    facts = bundle.public.events[:-1]
    links = tuple(event for event in facts if isinstance(event.payload, LinkFact))
    terminal_nodes = {
        event.payload.node for event in facts if isinstance(event.payload, (HazardFact, SafeFact))
    }
    solution_nodes = set(solution.node_path)
    branch_links = tuple(
        event
        for event in links
        if event.event_id not in solution.link_record_ids
        and event.payload.source_node in solution_nodes
    )
    outgoing_sources = {event.payload.source_node for event in links}

    assert solution.terminal_kind is OracleTerminalKind.HAZARD
    assert len(solution.link_record_ids) == 3
    assert config.stress is not None
    assert (
        config.stress.branching_records[0]
        <= len(branch_links)
        <= config.stress.branching_records[1]
    )
    assert all(
        event.payload.target_node not in terminal_nodes
        and event.payload.target_node not in outgoing_sources
        for event in branch_links
    )
    assert _activation_reachable_nodes(bundle) - solution_nodes == {
        event.payload.target_node for event in branch_links
    }

    with pytest.raises(OracleError, match="branch"):
        solve_public_episode(bundle.public, OraclePolicy.PRIMARY)

    from silent_cascade.env.invariants import validate_episode_invariants

    assert validate_episode_invariants(bundle, config).valid


def test_irrelevant_cycle_suite_is_outside_activation_reachability(
    config: Phase1Config,
) -> None:
    """Connecting the cycle to activation would turn it into an invalid reachable loop."""
    bundle = _generate(config, _request(SuiteName.CYCLES_STRESS))
    cycles = _directed_cycles(bundle)
    reachable = _activation_reachable_nodes(bundle)

    assert config.stress is not None
    assert len(cycles) == 1
    assert (
        config.stress.irrelevant_cycle_length[0]
        <= len(cycles[0])
        <= config.stress.irrelevant_cycle_length[1]
    )
    assert all(node not in reachable for node in cycles[0])
    assert solve_public_episode(bundle.public).terminal_kind is OracleTerminalKind.HAZARD

    from silent_cascade.env.invariants import validate_episode_invariants

    assert validate_episode_invariants(bundle, config).valid


def test_overflow_retains_all_facts_and_keeps_capacity_diagnostic_private(
    config: Phase1Config,
) -> None:
    """Eviction or a public capacity cue would lose the declared overflow stress condition."""
    bundle = _generate(config, _request(SuiteName.MEMORY_OVERFLOW_STRESS))
    facts = bundle.public.events[:-1]
    metadata = bundle.truth.stress_metadata

    assert config.stress is not None
    assert (
        config.stress.overflow_record_count[0]
        <= len(facts)
        <= config.stress.overflow_record_count[1]
    )
    assert len(facts) > bundle.public.init.memory_capacity
    assert metadata is not None
    assert metadata.over_capacity_record_count == len(facts)
    assert metadata.near_miss_missing_edges is None
    assert sum(isinstance(event.payload, HazardFact) for event in facts) == 2
    assert sum(isinstance(event.payload, SafeFact) for event in facts) == 1
    assert solve_public_episode(bundle.public).terminal_kind is OracleTerminalKind.HAZARD
    assert "over_capacity" not in repr(bundle.public)

    from silent_cascade.env.invariants import validate_episode_invariants
    from silent_cascade.env.oracle import verify_oracle_truth

    verify_oracle_truth(solve_public_episode(bundle.public), bundle.truth)
    assert validate_episode_invariants(bundle, config).valid
    primary_truth = replace(
        bundle.truth,
        key=replace(bundle.truth.key, suite=SuiteName.IID_PRIMARY),
        recipe=replace(bundle.truth.recipe, evaluation_suite=SuiteName.IID_PRIMARY),
        stress_metadata=None,
    )
    _assert_invalid(EpisodeBundle(bundle.public, primary_truth), config)


def test_null_near_miss_is_a_disconnected_negative_with_one_private_missing_edge(
    config: Phase1Config,
) -> None:
    """Adding the named missing LINK would make the private near-miss hazard reachable."""
    bundle = _generate(config, _request(SuiteName.NULL_NEAR_MISS_STRESS))
    metadata = bundle.truth.stress_metadata
    facts = bundle.public.events[:-1]
    reachable = _activation_reachable_nodes(bundle)
    endpoint = bundle.truth.relevant_node_path[-1]

    assert metadata is not None
    assert metadata.near_miss_missing_edges == 1
    assert metadata.near_miss_hazard_node is not None
    assert bundle.truth.recipe.variant is EpisodeVariant.DISCONNECTED_NEGATIVE
    assert solve_public_episode(bundle.public).terminal_kind is OracleTerminalKind.DISCONNECTED
    assert bundle.truth.terminal_record_id is None
    assert metadata.near_miss_hazard_node not in reachable
    assert any(
        isinstance(event.payload, HazardFact)
        and event.payload.node == metadata.near_miss_hazard_node
        for event in facts
    )
    assert not any(
        isinstance(event.payload, LinkFact)
        and (event.payload.source_node, event.payload.target_node)
        == (endpoint, metadata.near_miss_hazard_node)
        for event in facts
    )
    assert "near_miss" not in repr(bundle.public)

    from silent_cascade.env.invariants import validate_episode_invariants
    from silent_cascade.env.oracle import verify_oracle_truth

    verify_oracle_truth(solve_public_episode(bundle.public), bundle.truth)
    assert validate_episode_invariants(bundle, config).valid


@pytest.mark.parametrize(
    "suite",
    (
        SuiteName.BRANCHING_STRESS,
        SuiteName.CYCLES_STRESS,
        SuiteName.MEMORY_OVERFLOW_STRESS,
        SuiteName.NULL_NEAR_MISS_STRESS,
    ),
)
def test_structural_stress_regeneration_order_and_global_rng_are_stable(
    config: Phase1Config, suite: SuiteName
) -> None:
    """Changing call order, regeneration, or ambient RNG must not alter a stress artifact."""
    import silent_cascade.env.generator as generator

    request = _request(suite, episode_index=29)
    other_request = _request(
        SuiteName.CYCLES_STRESS
        if suite is SuiteName.BRANCHING_STRESS
        else SuiteName.BRANCHING_STRESS,
        episode_index=30,
    )
    regenerate = getattr(generator, "regenerate_stress_episode", None)
    assert callable(regenerate), "structural stress regeneration API is missing"
    np.random.seed(20260902)
    before = np.random.get_state()
    first = _generate(config, request)
    _generate(config, other_request)
    second = _generate(config, request)
    after = np.random.get_state()

    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert canonical_episode_bytes(first) == canonical_episode_bytes(second)
    assert (
        regenerate(
            config,
            request,
            91,
            first.public.init.episode_public_id,
            first.truth.recipe.accepted_attempt,
        )
        == first
    )
    assert set(independent_seed_tokens(request, 0)).isdisjoint(
        independent_seed_tokens(other_request, 0)
    )


def test_branching_and_cycle_mutations_fail_closed(config: Phase1Config) -> None:
    """Reachable branch/cycle corruption must fail independent structural validation."""
    branching = _generate(config, _request(SuiteName.BRANCHING_STRESS, 41))
    branch_solution = solve_public_episode(branching.public, OraclePolicy.BRANCHING)
    branch_links = tuple(
        event
        for event in branching.public.events[:-1]
        if isinstance(event.payload, LinkFact)
        and event.event_id not in branch_solution.link_record_ids
        and event.payload.source_node in branch_solution.node_path
    )
    assert len(branch_links) >= 2
    _assert_invalid(
        _replace_fact(
            branching,
            branch_links[0].event_id,
            LinkFact(branch_links[0].payload.source_node, branch_solution.node_path[0]),
        ),
        config,
    )
    _assert_invalid(
        _replace_fact(
            branching,
            branch_links[1].event_id,
            LinkFact(branch_links[0].payload.target_node, branch_links[1].payload.target_node),
        ),
        config,
    )
    _assert_invalid(
        EpisodeBundle(
            branching.public,
            replace(
                branching.truth,
                recipe=replace(branching.truth.recipe, distractor_link_count=0),
            ),
        ),
        config,
    )

    cycles = _generate(config, _request(SuiteName.CYCLES_STRESS, 42))
    cycle = _directed_cycles(cycles)[0]
    cycle_event = next(
        event
        for event in cycles.public.events[:-1]
        if isinstance(event.payload, LinkFact)
        and (event.payload.source_node, event.payload.target_node) == (cycle[0], cycle[1])
    )
    _assert_invalid(
        _replace_fact(cycles, cycle_event.event_id, LinkFact(cycle[0], cycle[0])), config
    )
    _assert_invalid(
        _replace_fact(
            cycles,
            cycle_event.event_id,
            LinkFact(cycles.truth.relevant_node_path[0], cycle[1]),
        ),
        config,
    )


def test_overflow_and_near_miss_mutations_fail_closed(config: Phase1Config) -> None:
    """Capacity metadata and one-edge near-miss provenance cannot be repaired silently."""
    overflow = _generate(config, _request(SuiteName.MEMORY_OVERFLOW_STRESS, 43))
    assert overflow.truth.stress_metadata is not None
    _assert_invalid(
        EpisodeBundle(
            overflow.public,
            replace(
                overflow.truth,
                stress_metadata=StressMetadata(
                    over_capacity_record_count=overflow.truth.stress_metadata.over_capacity_record_count
                    + 1
                ),
            ),
        ),
        config,
    )

    near_miss = _generate(config, _request(SuiteName.NULL_NEAR_MISS_STRESS, 44))
    metadata = near_miss.truth.stress_metadata
    assert metadata is not None and metadata.near_miss_hazard_node is not None
    last_link_id = near_miss.truth.relevant_record_ids[-1]
    _assert_invalid(
        _replace_fact(
            near_miss,
            last_link_id,
            LinkFact(
                near_miss.truth.relevant_node_path[-2],
                metadata.near_miss_hazard_node,
            ),
        ),
        config,
    )
    _assert_invalid(
        EpisodeBundle(
            near_miss.public,
            replace(
                near_miss.truth,
                stress_metadata=StressMetadata(
                    near_miss_missing_edges=2,
                    near_miss_hazard_node=metadata.near_miss_hazard_node,
                ),
            ),
        ),
        config,
    )


def test_overflow_and_near_miss_reject_the_wrong_primary_variant(
    config: Phase1Config,
) -> None:
    """Suite policy must not permit a positive near miss or a negative overflow trace."""
    with pytest.raises(ValueError, match="positive hazard path"):
        _generate(
            config,
            replace(
                _request(SuiteName.MEMORY_OVERFLOW_STRESS, 45),
                variant=EpisodeVariant.DISCONNECTED_NEGATIVE,
            ),
        )
    with pytest.raises(ValueError, match="disconnected negative path"):
        _generate(
            config,
            replace(
                _request(SuiteName.NULL_NEAR_MISS_STRESS, 46),
                variant=EpisodeVariant.POSITIVE,
            ),
        )
