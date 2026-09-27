"""Phase-scoped protocol identity and publication guards."""

from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import SplitNamespace
from silent_cascade.eval.comparison_data import (
    build_manifest,
    diagnostic_allocations,
    iter_bundles,
    publish_manifest,
)
from silent_cascade.eval.comparison_types import ComparisonConfig, ComparisonManifest


def test_config_and_manifest_roundtrip() -> None:
    """The committed YAML and portable JSON must reconstruct the same protocol."""
    config = resolve_config(ComparisonConfig, [Path("configs/eval/phase5a.yaml")]).config
    assert config == ComparisonConfig()
    allocation = diagnostic_allocations()["iid"]
    first = allocation.blocks[0].model_copy(update={"episode_count": 4})
    tiny = allocation.model_copy(update={"blocks": (first,)})
    manifest = build_manifest(config, "iid", tiny)
    restored = ComparisonManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest
    assert len(tuple(iter_bundles(restored, config))) == 4


def test_reject_frozen_overwrite_or_protocol_mutation(tmp_path) -> None:
    """A copied or edited manifest must never silently become comparison input."""
    config = ComparisonConfig()
    allocation = diagnostic_allocations()["iid"]
    first = allocation.blocks[0].model_copy(update={"episode_count": 4})
    allocation = allocation.model_copy(update={"blocks": (first,)})
    manifest = build_manifest(config, "iid", allocation)
    frozen = tmp_path / "manifests" / "frozen" / "iid.json"
    with pytest.raises(ValueError, match="frozen"):
        publish_manifest(manifest, frozen)
    path = tmp_path / "iid.json"
    publish_manifest(manifest, path)
    assert publish_manifest(manifest, path) == path
    changed = manifest.model_copy(update={"protocol_sha256": "0" * 64})
    with pytest.raises(ValueError, match=r"overwrite|different"):
        publish_manifest(changed, path)
    with pytest.raises(ValueError, match="protocol"):
        tuple(iter_bundles(changed, config))
    wrong_split = manifest.model_copy(update={"split_namespace": SplitNamespace.FROZEN})
    with pytest.raises(ValueError, match=r"DEBUG|debug"):
        tuple(iter_bundles(wrong_split, config))
