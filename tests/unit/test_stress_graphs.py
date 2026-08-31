"""Structural contracts for the Phase 1 branching and irrelevant-cycle suites."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import EpisodeBundle, EpisodeVariant, canonical_episode_bytes
from silent_cascade.env.generator import IndependentEpisodeRequest, independent_seed_tokens
from silent_cascade.env.oracle import OraclePolicy, OracleTerminalKind, solve_public_episode
from silent_cascade.errors import OracleError
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
    return IndependentEpisodeRequest(
        split_namespace=SplitNamespace.DEBUG,
        suite=suite,
        root_seed=20260831,
        episode_index=episode_index,
        requested_path_length=3,
        variant=EpisodeVariant.POSITIVE,
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


@pytest.mark.parametrize("suite", (SuiteName.BRANCHING_STRESS, SuiteName.CYCLES_STRESS))
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
