import json
import os
import subprocess
import sys
import uuid
from dataclasses import replace

import numpy as np
import pytest
import torch

from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.episode import EpisodeVariant
from silent_cascade.rng import (
    AllocationLabelKey,
    CounterSeedKey,
    IndependentCounterSeedKey,
    IndependentPublicIdKey,
    PublicIdBatchKey,
    SeedStream,
    allocate_independent_public_id,
    allocate_independent_variants,
    allocate_public_ids,
    derive_counter_seed,
    derive_independent_counter_seed,
    independent_local_generator,
    local_generator,
    seed_key_primitive,
    snapshot_global_rng,
)


def counter_seed_key(cohort_index: int) -> CounterSeedKey:
    return CounterSeedKey(
        generator_version="ofd-v1",
        split_namespace=SplitNamespace.VALIDATION,
        suite=SuiteName.IID_PRIMARY,
        root_seed=41,
        cohort_index=cohort_index,
        member_index=2,
        stream=SeedStream.PRESENTATION,
        attempt=0,
    )


def assert_rng_snapshots_equal(actual: object, expected: object) -> None:
    assert hasattr(actual, "python_state") and hasattr(expected, "python_state")
    assert actual.python_state == expected.python_state
    assert actual.numpy_state[0] == expected.numpy_state[0]
    assert np.array_equal(actual.numpy_state[1], expected.numpy_state[1])
    assert actual.numpy_state[2:] == expected.numpy_state[2:]
    assert torch.equal(actual.torch_cpu_state, expected.torch_cpu_state)
    if expected.torch_mps_state is not None:
        assert actual.torch_mps_state is not None
        assert torch.equal(actual.torch_mps_state, expected.torch_mps_state)


def test_counter_rng_is_order_independent_and_does_not_touch_globals() -> None:
    before = snapshot_global_rng()
    keys = [counter_seed_key(index) for index in range(20)]

    forward = {item.cohort_index: local_generator(item).integers(0, 2**63) for item in keys}
    reverse = {
        item.cohort_index: local_generator(item).integers(0, 2**63) for item in reversed(keys)
    }

    assert forward == reverse
    assert_rng_snapshots_equal(snapshot_global_rng(), before)


def test_cohort_counter_seed_matches_frozen_vector_and_canonical_key_bytes() -> None:
    key = counter_seed_key(7)

    assert derive_counter_seed(key).token == (
        "14dfc485480dfb97aa21f99f8cb152ffe09e7b183e9a0bffbe424b28df52f8e2"
    )
    assert derive_counter_seed(key).seed == 27746428027079338998295180152501523199
    assert json.dumps(
        seed_key_primitive(key),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode() == (
        b'{"attempt":0,"cohort_index":7,"generator_version":"ofd-v1",'
        b'"member_index":2,"root_seed":41,"split_namespace":"validation",'
        b'"stream":"presentation","suite":"iid_primary"}'
    )


def test_seed_key_primitive_accepts_only_typed_counter_keys() -> None:
    with pytest.raises(TypeError, match="CounterSeedKey"):
        seed_key_primitive({})  # type: ignore[arg-type]


def test_independent_counter_seed_matches_frozen_vector_and_is_local() -> None:
    key = IndependentCounterSeedKey(
        generator_version="ofd-v1",
        split_namespace=SplitNamespace.PHASE1_GATE,
        suite=SuiteName.IID_PRIMARY,
        root_seed=41,
        episode_index=7,
        stream=SeedStream.PRESENTATION,
        attempt=0,
    )

    assert derive_independent_counter_seed(key).token == (
        "b93ed757b3430cc59003146a0afeead8f43d3919db05872185a7b504de1d90bf"
    )
    assert derive_independent_counter_seed(key).seed == 246233469291832394751030201046666242776
    assert independent_local_generator(key).integers(0, 2**63) == independent_local_generator(
        key
    ).integers(0, 2**63)


def test_matched_public_ids_match_frozen_vector_and_are_rfc4122_v4() -> None:
    ids = allocate_public_ids(
        PublicIdBatchKey(
            generator_version="ofd-v1",
            split_namespace=SplitNamespace.VALIDATION,
            suite=SuiteName.IID_PRIMARY,
            public_id_seed=91,
            cohort_index=7,
            accepted_attempt=0,
        )
    )

    assert ids == (
        "eb78e7f5-c0bb-4b38-88f3-5d9d9024fc32",
        "bd8da50f-9499-4c1f-a23e-09ceeeb68e53",
        "91800f56-56f7-4815-8db1-8e4c7c892aa2",
        "8a4f6ea2-4a78-490b-bf96-d007436f1e5c",
    )
    assert all(uuid.UUID(value).version == 4 for value in ids)
    assert all(uuid.UUID(value).variant == uuid.RFC_4122 for value in ids)


def test_independent_public_id_matches_frozen_vector_and_is_rfc4122_v4() -> None:
    public_id = allocate_independent_public_id(
        IndependentPublicIdKey(
            generator_version="ofd-v1",
            split_namespace=SplitNamespace.PHASE1_GATE,
            suite=SuiteName.IID_PRIMARY,
            public_id_seed=91,
            episode_index=7,
            accepted_attempt=0,
        )
    )

    parsed = uuid.UUID(public_id)
    assert public_id == "8d1d0713-5f5e-4c49-a6a1-f3620133d2d7"
    assert parsed.version == 4
    assert parsed.variant == uuid.RFC_4122


@pytest.mark.parametrize(
    ("key", "match"),
    [
        (replace(counter_seed_key(0), member_index=0, stream=SeedStream.LABEL), "scope"),
        (replace(counter_seed_key(0), member_index=-1), "scope"),
        (replace(counter_seed_key(0), root_seed=True), "root_seed"),
        (replace(counter_seed_key(0), root_seed=2**128), "root_seed"),
        (replace(counter_seed_key(0), cohort_index=True), "cohort_index"),
        (replace(counter_seed_key(0), attempt=True), "attempt"),
        (replace(counter_seed_key(0), attempt=1_000), "attempt"),
    ],
)
def test_counter_seed_rejects_invalid_scope_and_exact_integer_ranges(
    key: CounterSeedKey, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        derive_counter_seed(key)


@pytest.mark.parametrize(
    ("key", "match"),
    [
        (
            IndependentCounterSeedKey(
                "ofd-v1",
                SplitNamespace.PHASE1_GATE,
                SuiteName.IID_PRIMARY,
                41,
                7,
                SeedStream.LABEL,
                0,
            ),
            "LABEL",
        ),
        (
            IndependentCounterSeedKey(
                "ofd-v1",
                SplitNamespace.PHASE1_GATE,
                SuiteName.IID_PRIMARY,
                True,
                7,
                SeedStream.PRESENTATION,
                0,
            ),
            "root_seed",
        ),
        (
            IndependentCounterSeedKey(
                "ofd-v1",
                SplitNamespace.PHASE1_GATE,
                SuiteName.IID_PRIMARY,
                41,
                True,
                SeedStream.PRESENTATION,
                0,
            ),
            "episode_index",
        ),
        (
            IndependentCounterSeedKey(
                "ofd-v1",
                SplitNamespace.PHASE1_GATE,
                SuiteName.IID_PRIMARY,
                41,
                7,
                SeedStream.PRESENTATION,
                1_000,
            ),
            "attempt",
        ),
    ],
)
def test_independent_counter_seed_rejects_label_and_exact_integer_ranges(
    key: IndependentCounterSeedKey, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        derive_independent_counter_seed(key)


@pytest.mark.parametrize(
    ("key", "match"),
    [
        (
            PublicIdBatchKey(
                "ofd-v1", SplitNamespace.VALIDATION, SuiteName.IID_PRIMARY, True, 0, 0
            ),
            "public_id_seed",
        ),
        (
            PublicIdBatchKey(
                "ofd-v1", SplitNamespace.VALIDATION, SuiteName.IID_PRIMARY, 0, True, 0
            ),
            "cohort_index",
        ),
        (
            PublicIdBatchKey(
                "ofd-v1", SplitNamespace.VALIDATION, SuiteName.IID_PRIMARY, 0, 0, True
            ),
            "accepted_attempt",
        ),
        (
            PublicIdBatchKey(
                "ofd-v1", SplitNamespace.VALIDATION, SuiteName.IID_PRIMARY, 2**128, 0, 0
            ),
            "public_id_seed",
        ),
        (
            PublicIdBatchKey(
                "ofd-v1", SplitNamespace.VALIDATION, SuiteName.IID_PRIMARY, 0, 0, 1_000
            ),
            "accepted_attempt",
        ),
    ],
)
def test_matched_public_id_rejects_exact_integer_ranges(key: PublicIdBatchKey, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        allocate_public_ids(key)


@pytest.mark.parametrize(
    ("key", "match"),
    [
        (
            IndependentPublicIdKey(
                "ofd-v1", SplitNamespace.PHASE1_GATE, SuiteName.IID_PRIMARY, True, 0, 0
            ),
            "public_id_seed",
        ),
        (
            IndependentPublicIdKey(
                "ofd-v1", SplitNamespace.PHASE1_GATE, SuiteName.IID_PRIMARY, 0, True, 0
            ),
            "episode_index",
        ),
        (
            IndependentPublicIdKey(
                "ofd-v1", SplitNamespace.PHASE1_GATE, SuiteName.IID_PRIMARY, 0, 0, True
            ),
            "accepted_attempt",
        ),
    ],
)
def test_independent_public_id_rejects_exact_integer_ranges(
    key: IndependentPublicIdKey, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        allocate_independent_public_id(key)


def test_independent_allocation_returns_one_stratified_quartet_without_globals() -> None:
    before = snapshot_global_rng()
    key = AllocationLabelKey(
        generator_version="ofd-v1",
        split_namespace=SplitNamespace.PHASE1_GATE,
        suite=SuiteName.IID_PRIMARY,
        root_seed=41,
        requested_path_length=3,
        allocation_quartet_index=7,
    )

    variants = allocate_independent_variants(key)

    assert variants == allocate_independent_variants(key)
    assert len(variants) == 4
    assert variants.count(EpisodeVariant.POSITIVE) == 2
    assert variants.count(EpisodeVariant.SAFE_NEGATIVE) == 1
    assert variants.count(EpisodeVariant.DISCONNECTED_NEGATIVE) == 1
    assert_rng_snapshots_equal(snapshot_global_rng(), before)


@pytest.mark.parametrize(
    ("key", "match"),
    [
        (
            AllocationLabelKey(
                "ofd-v1", SplitNamespace.PHASE1_GATE, SuiteName.IID_PRIMARY, True, 3, 0
            ),
            "root_seed",
        ),
        (
            AllocationLabelKey(
                "ofd-v1", SplitNamespace.PHASE1_GATE, SuiteName.IID_PRIMARY, 41, True, 0
            ),
            "requested_path_length",
        ),
        (
            AllocationLabelKey(
                "ofd-v1", SplitNamespace.PHASE1_GATE, SuiteName.IID_PRIMARY, 41, 3, True
            ),
            "allocation_quartet_index",
        ),
    ],
)
def test_independent_allocation_rejects_exact_integer_ranges(
    key: AllocationLabelKey, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        allocate_independent_variants(key)


def test_counter_seed_tokens_have_no_collisions_in_a_focused_sample() -> None:
    tokens = {derive_counter_seed(counter_seed_key(index)).token for index in range(10_000)}

    assert len(tokens) == 10_000


def test_accepted_draw_token_sequences_include_trace_jitter_in_frozen_order() -> None:
    """Omitting acceptance-affecting jitter would make construction evidence incomplete."""
    import silent_cascade.env.generator as generator_module
    from silent_cascade.env.generator import (
        CohortRequest,
        independent_seed_tokens,
        iter_phase1_gate_requests,
    )

    matched_request = CohortRequest(
        SplitNamespace.VALIDATION,
        SuiteName.VALIDATION,
        41,
        0,
        2,
    )
    matched = generator_module._matched_seed_tokens(matched_request, 0)
    independent_request = next(iter_phase1_gate_requests(41))
    independent = independent_seed_tokens(independent_request, 0)

    matched_stream_groups = (
        (
            -1,
            (
                SeedStream.LABEL,
                SeedStream.TEMPLATE,
                SeedStream.STRUCTURE,
                SeedStream.TIMESTAMPS,
            ),
        ),
        *(
            (
                member,
                (
                    SeedStream.NODE_PERMUTATION,
                    SeedStream.TERMINALS,
                    SeedStream.PRESENTATION,
                    SeedStream.TRACE_JITTER,
                ),
            )
            for member in range(4)
        ),
    )
    expected_matched = tuple(
        derive_counter_seed(
            CounterSeedKey(
                "ofd-v1",
                matched_request.split_namespace,
                matched_request.suite,
                matched_request.root_seed,
                matched_request.cohort_index,
                member_index,
                stream,
                0,
            )
        ).token
        for member_index, streams in matched_stream_groups
        for stream in streams
    )
    expected_independent = tuple(
        derive_independent_counter_seed(
            IndependentCounterSeedKey(
                "ofd-v1",
                independent_request.split_namespace,
                independent_request.suite,
                independent_request.root_seed,
                independent_request.episode_index,
                stream,
                0,
            )
        ).token
        for stream in (
            SeedStream.TEMPLATE,
            SeedStream.STRUCTURE,
            SeedStream.TIMESTAMPS,
            SeedStream.NODE_PERMUTATION,
            SeedStream.TERMINALS,
            SeedStream.PRESENTATION,
            SeedStream.TRACE_JITTER,
        )
    )

    assert matched == expected_matched
    assert independent == expected_independent


def test_independent_and_matched_public_id_domains_have_no_collisions_in_focused_samples() -> None:
    matched = {
        public_id
        for cohort_index in range(10_000)
        for public_id in allocate_public_ids(
            PublicIdBatchKey(
                "ofd-v1",
                SplitNamespace.PHASE1_GATE,
                SuiteName.IID_PRIMARY,
                91,
                cohort_index,
                0,
            )
        )
    }
    independent = {
        allocate_independent_public_id(
            IndependentPublicIdKey(
                "ofd-v1",
                SplitNamespace.PHASE1_GATE,
                SuiteName.IID_PRIMARY,
                91,
                episode_index,
                0,
            )
        )
        for episode_index in range(10_000)
    }

    assert len(matched) == 40_000
    assert len(independent) == 10_000
    assert matched.isdisjoint(independent)


def test_counter_rng_results_do_not_depend_on_python_hash_seed() -> None:
    script = """
import json
from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.rng import (
    AllocationLabelKey,
    CounterSeedKey,
    IndependentPublicIdKey,
    PublicIdBatchKey,
    SeedStream,
    allocate_independent_public_id,
    allocate_independent_variants,
    allocate_public_ids,
    derive_counter_seed,
    local_generator,
)
key = CounterSeedKey(
    'ofd-v1', SplitNamespace.VALIDATION, SuiteName.IID_PRIMARY, 41, 7, 2,
    SeedStream.PRESENTATION, 0,
)
allocation_key = AllocationLabelKey(
    'ofd-v1', SplitNamespace.PHASE1_GATE, SuiteName.IID_PRIMARY, 41, 3, 7,
)
matched_key = PublicIdBatchKey(
    'ofd-v1', SplitNamespace.VALIDATION, SuiteName.IID_PRIMARY, 91, 7, 0,
)
independent_key = IndependentPublicIdKey(
    'ofd-v1', SplitNamespace.PHASE1_GATE, SuiteName.IID_PRIMARY, 91, 7, 0,
)
print(json.dumps({
    'token': derive_counter_seed(key).token,
    'draw': int(local_generator(key).integers(0, 2**63)),
    'variants': [value.value for value in allocate_independent_variants(allocation_key)],
    'matched': allocate_public_ids(matched_key),
    'independent': allocate_independent_public_id(independent_key),
}, sort_keys=True))
"""
    outputs = []
    for hash_seed in ("0", "1"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            cwd=os.getcwd(),
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            text=True,
        )
        outputs.append(completed.stdout)

    assert outputs[0] == outputs[1]
