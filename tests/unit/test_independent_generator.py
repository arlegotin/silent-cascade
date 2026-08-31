"""Behavioral contracts for independent Phase 1 claim-data generation."""

import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace
from silent_cascade.env.episode import EpisodeVariant, canonical_episode_bytes
from silent_cascade.env.generator import PHASE1_GATE_ALLOCATION, iter_independent_requests
from silent_cascade.env.oracle import solve_public_episode, verify_oracle_truth
from silent_cascade.errors import GenerationError
from silent_cascade.schemas import ExternalEventKind, HazardFact, SafeFact


@pytest.fixture
def config() -> Phase1Config:
    return resolve_config(
        Phase1Config,
        [Path("configs/base.yaml"), Path("configs/data/primary.yaml")],
    ).config


def first_quartet(root_seed: int = 41) -> tuple[object, object, object, object]:
    requests = tuple(iter_independent_requests(PHASE1_GATE_ALLOCATION, root_seed))
    return requests[0], requests[1], requests[2], requests[3]


def test_independent_quartet_has_exact_labels_and_disjoint_nuisance_tokens(
    config: Phase1Config,
) -> None:
    """Quartets stratify labels only; they cannot share construction randomness."""
    from silent_cascade.env.generator import independent_seed_tokens

    requests = first_quartet()
    assert Counter(request.variant for request in requests) == {
        EpisodeVariant.POSITIVE: 2,
        EpisodeVariant.SAFE_NEGATIVE: 1,
        EpisodeVariant.DISCONNECTED_NEGATIVE: 1,
    }
    token_sets = [set(independent_seed_tokens(request, accepted_attempt=0)) for request in requests]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(token_sets)
        for right in token_sets[index + 1 :]
    )
    assert all(len(tokens) == 6 for tokens in token_sets)


def test_independent_episode_has_requested_public_oracle_and_private_contract(
    config: Phase1Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Independent construction must preserve the complete primary episode contract."""
    import silent_cascade.env.generator as generator
    from silent_cascade.env.invariants import validate_episode_invariants

    monkeypatch.setattr(
        generator,
        "sample_cohort_template",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cohort sampler used")),
    )

    for request in first_quartet():
        bundle = generator.generate_independent_episode(config, request, public_id_seed=91)
        facts = bundle.public.events[:-1]
        fact_payloads = tuple(event.payload for event in facts)
        solution = solve_public_episode(bundle.public)

        assert sum(isinstance(payload, HazardFact) for payload in fact_payloads) == 2
        assert sum(isinstance(payload, SafeFact) for payload in fact_payloads) == 1
        assert len(facts) <= config.data.primary_memory_capacity
        assert [event.event_id for event in facts] == list(range(len(facts)))
        assert bundle.public.events[-1].kind is ExternalEventKind.ACTIVATE
        assert bundle.public.events[-1].event_id == len(facts)
        assert bundle.truth.private_terminal.event_id == len(facts) + 1
        assert len(solution.link_record_ids) == request.requested_path_length
        assert bundle.truth.key.coordinate.mode == "independent"
        assert bundle.truth.key.coordinate.episode_index == request.episode_index
        assert (
            bundle.truth.key.coordinate.allocation_quartet_index == request.allocation_quartet_index
        )
        assert "root_seed" not in repr(bundle.public)
        assert "private_terminal" not in repr(bundle.public)
        verify_oracle_truth(solution, bundle.truth)
        assert validate_episode_invariants(bundle, config).valid


def test_independent_retry_is_episode_local_and_exhaustion_is_opaque(
    config: Phase1Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rejected draw may retry only its own episode and never disclose its coordinates."""
    import silent_cascade.env.generator as generator

    requests = first_quartet()
    baseline = {
        request.episode_index: generator.generate_independent_episode(config, request, 91)
        for request in requests
    }
    original = generator._build_independent_candidate
    rejected_index = requests[0].episode_index

    def reject_one(*args: object, **kwargs: object) -> object:
        request = args[1]
        attempt = args[2]
        if request.episode_index == rejected_index and attempt < 3:
            raise ValueError("forced independent rejection")
        return original(*args, **kwargs)

    monkeypatch.setattr(generator, "_build_independent_candidate", reject_one)
    generated = {
        request.episode_index: generator.generate_independent_episode(config, request, 91)
        for request in requests
    }
    assert generated[rejected_index].truth.recipe.accepted_attempt == 3
    for request in requests[1:]:
        assert generated[request.episode_index].truth.recipe.accepted_attempt == 0
        assert canonical_episode_bytes(generated[request.episode_index]) == canonical_episode_bytes(
            baseline[request.episode_index]
        )

    monkeypatch.setattr(
        generator,
        "_build_independent_candidate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("forced independent rejection")),
    )
    short = config.model_copy(
        update={"data": config.data.model_copy(update={"max_generation_attempts": 3})}
    )
    with pytest.raises(GenerationError, match="independent episode generation exhausted") as raised:
        generator.generate_independent_episode(short, requests[0], 91)
    context = raised.value.to_payload()["context"]
    assert set(context) == {"request_hash", "attempt_count", "rejection_reasons"}
    assert re.fullmatch(r"[0-9a-f]{64}", context["request_hash"])
    assert "41" not in str(context)
    assert requests[0].variant.value not in str(context)


def test_independent_regeneration_and_execution_topology_are_exact(
    config: Phase1Config,
) -> None:
    """Call order, chunking, and global NumPy state cannot change independent artifacts."""
    from silent_cascade.env.generator import (
        generate_independent_episode,
        regenerate_independent_episode,
    )

    requests = first_quartet()
    np.random.seed(20260831)
    before = np.random.get_state()
    forward = {
        request.episode_index: generate_independent_episode(config, request, 91)
        for request in requests
    }
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    reverse = {
        request.episode_index: generate_independent_episode(config, request, 91)
        for request in reversed(requests)
    }
    for chunk_size in (1, 3, 7):
        chunked = {
            request.episode_index: generate_independent_episode(config, request, 91)
            for start in range(0, len(requests), chunk_size)
            for request in requests[start : start + chunk_size]
        }
        for request in requests:
            expected = forward[request.episode_index]
            assert canonical_episode_bytes(
                chunked[request.episode_index]
            ) == canonical_episode_bytes(expected)
            assert canonical_episode_bytes(
                reverse[request.episode_index]
            ) == canonical_episode_bytes(expected)
    expected = forward[requests[0].episode_index]
    assert (
        regenerate_independent_episode(
            config,
            requests[0],
            91,
            expected.public.init.episode_public_id,
            expected.truth.recipe.accepted_attempt,
        )
        == expected
    )
    with pytest.raises(GenerationError, match="regeneration"):
        regenerate_independent_episode(config, requests[0], 91, "bad", 0)


def test_independent_frozen_namespace_uses_the_same_private_recipe_path(
    config: Phase1Config,
) -> None:
    """Frozen claim data may only change its namespace, never its generator primitive."""
    from silent_cascade.env.generator import (
        generate_independent_episode,
        independent_seed_tokens,
    )
    from silent_cascade.env.invariants import validate_episode_invariants

    gate_request = first_quartet()[0]
    frozen_request = replace(gate_request, split_namespace=SplitNamespace.FROZEN)
    assert independent_seed_tokens(gate_request, 0) != independent_seed_tokens(frozen_request, 0)
    gate = generate_independent_episode(config, gate_request, 91)
    frozen = generate_independent_episode(config, frozen_request, 91)
    assert gate.truth.key.coordinate == frozen.truth.key.coordinate
    assert gate.truth.recipe.requested_path_length == frozen.truth.recipe.requested_path_length
    assert gate.truth.recipe.variant is frozen.truth.recipe.variant
    assert gate.public.init.episode_public_id != frozen.public.init.episode_public_id
    assert validate_episode_invariants(gate, config).valid
    assert validate_episode_invariants(frozen, config).valid


def test_independent_generation_is_process_hash_seed_stable() -> None:
    """No unordered or Python-hash path may enter claim-data construction."""
    script = """
from pathlib import Path
from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config
from silent_cascade.env.episode import canonical_episode_bytes
from silent_cascade.env.generator import generate_independent_episode, iter_phase1_gate_requests
config = resolve_config(
    Phase1Config, [Path('configs/base.yaml'), Path('configs/data/primary.yaml')]
).config
request = next(iter_phase1_gate_requests(41))
print(canonical_episode_bytes(generate_independent_episode(config, request, 91)).hex())
"""
    outputs = []
    for hash_seed in ("0", "1"):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path.cwd(),
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(result.stdout.strip())
    assert outputs[0] and outputs[0] == outputs[1]
