"""Diagnostic allocation and immutable reconstruction contracts."""

from collections import Counter

import pytest

from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.episode import EpisodeVariant
from silent_cascade.env.generator import iter_independent_requests
from silent_cascade.eval.comparison_data import (
    build_manifest,
    diagnostic_allocations,
    iter_bundles,
)
from silent_cascade.eval.comparison_types import ComparisonConfig


def test_diagnostic_allocations_and_identity() -> None:
    """A changed depth allocation or label ratio would invalidate the paired sample."""
    allocations = diagnostic_allocations()
    assert set(allocations) == {"iid", "depth"}
    assert {
        block.requested_path_length: block.episode_count for block in allocations["iid"].blocks
    } == {
        2: 172,
        3: 172,
        4: 168,
    }
    assert {
        block.requested_path_length: block.episode_count for block in allocations["depth"].blocks
    } == {
        5: 128,
        6: 128,
        7: 128,
        8: 128,
    }
    config = ComparisonConfig()
    coordinates = set()
    for name, allocation in allocations.items():
        assert allocation.split_namespace is SplitNamespace.DEBUG
        root_seed = config.iid_root_seed if name == "iid" else config.depth_root_seed
        requests = tuple(iter_independent_requests(allocation, root_seed))
        assert len(requests) == 512
        assert Counter(request.variant for request in requests) == {
            EpisodeVariant.POSITIVE: 256,
            EpisodeVariant.SAFE_NEGATIVE: 128,
            EpisodeVariant.DISCONNECTED_NEGATIVE: 128,
        }
        coordinates.update((request.suite, request.episode_index) for request in requests)
        assert {request.suite for request in requests} == {
            SuiteName.IID_PRIMARY if name == "iid" else SuiteName.OOD_DEPTH
        }
    assert len(coordinates) == 1024
    assert config.protocol_sha256 == ComparisonConfig().protocol_sha256


def test_every_diagnostic_episode_satisfies_invariants_and_oracle() -> None:
    """A hash-only manifest would miss corrupted generated truth or score."""
    config = ComparisonConfig()
    allocation = diagnostic_allocations()["iid"].model_copy(
        update={"blocks": diagnostic_allocations()["iid"].blocks[:1]}
    )
    first = allocation.blocks[0].model_copy(update={"episode_count": 4})
    allocation = allocation.model_copy(update={"blocks": (first,)})
    manifest = build_manifest(config, "iid", allocation)
    assert manifest.gate_eligible is False
    assert manifest.split_namespace is SplitNamespace.DEBUG
    assert len(manifest.entries) == 4
    assert len({entry.public_id for entry in manifest.entries}) == 4
    assert all(len(entry.episode_sha256) == 64 for entry in manifest.entries)
    assert build_manifest(config, "iid", allocation) == manifest
    assert len(tuple(iter_bundles(manifest, config))) == 4


def test_manifest_rejects_corrupt_episode_hash() -> None:
    """Reconstruction must fail when retained episode bytes differ."""
    config = ComparisonConfig()
    allocation = diagnostic_allocations()["iid"]
    first = allocation.blocks[0].model_copy(update={"episode_count": 4})
    allocation = allocation.model_copy(update={"blocks": (first,)})
    manifest = build_manifest(config, "iid", allocation)
    bad_entry = manifest.entries[0].model_copy(update={"episode_sha256": "0" * 64})
    bad = manifest.model_copy(update={"entries": (bad_entry, *manifest.entries[1:])})
    with pytest.raises(ValueError, match="episode hash"):
        tuple(iter_bundles(bad, config))


def test_manifest_rejects_changed_allocation_or_selective_valid_rows() -> None:
    """Valid episode hashes cannot excuse a substituted diagnostic allocation."""
    config = ComparisonConfig()
    allocation = diagnostic_allocations()["iid"]
    first = allocation.blocks[0].model_copy(update={"episode_count": 4})
    allocation = allocation.model_copy(update={"blocks": (first,)})
    manifest = build_manifest(config, "iid", allocation)
    changed_allocation = manifest.model_copy(update={"allocation_sha256": "0" * 64})
    with pytest.raises(ValueError, match="allocation"):
        tuple(iter_bundles(changed_allocation, config))
    duplicated = manifest.model_copy(
        update={"entries": (manifest.entries[0], manifest.entries[0], *manifest.entries[2:])}
    )
    with pytest.raises(ValueError, match=r"coordinate|allocation|duplicate"):
        tuple(iter_bundles(duplicated, config))
