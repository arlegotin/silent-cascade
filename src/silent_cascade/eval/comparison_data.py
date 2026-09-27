"""Deterministic, evaluator-owned Phase 5A diagnostic episodes."""

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.episode import EpisodeBundle, EpisodeVariant, episode_sha256
from silent_cascade.env.generator import (
    EpisodeBlock,
    IndependentAllocation,
    generate_independent_episode,
    iter_independent_requests,
)
from silent_cascade.env.invariants import validate_episode_invariants
from silent_cascade.env.oracle import solve_public_episode, verify_oracle_truth
from silent_cascade.env.reward import score_actions
from silent_cascade.eval.comparison_types import (
    ComparisonConfig,
    ComparisonEpisodeRef,
    ComparisonManifest,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.schemas import Action


def diagnostic_allocations() -> dict[str, IndependentAllocation]:
    """Predeclared complete quartets, disjoint from all frozen namespaces."""
    return {
        "iid": IndependentAllocation(
            allocation_id="phase5a-iid-v1",
            split_namespace=SplitNamespace.DEBUG,
            blocks=tuple(
                EpisodeBlock(
                    suite=SuiteName.IID_PRIMARY,
                    requested_path_length=depth,
                    first_episode_index=first,
                    episode_count=count,
                )
                for depth, first, count in ((2, 0, 172), (3, 172, 172), (4, 344, 168))
            ),
        ),
        "depth": IndependentAllocation(
            allocation_id="phase5a-depth-v1",
            split_namespace=SplitNamespace.DEBUG,
            blocks=tuple(
                EpisodeBlock(
                    suite=SuiteName.OOD_DEPTH,
                    requested_path_length=depth,
                    first_episode_index=first,
                    episode_count=128,
                )
                for depth, first in ((5, 0), (6, 128), (7, 256), (8, 384))
            ),
        ),
    }


def _seeds(config: ComparisonConfig, name: str) -> tuple[int, int]:
    if name == "iid":
        return config.iid_root_seed, config.iid_public_id_seed
    if name == "depth":
        return config.depth_root_seed, config.depth_public_id_seed
    raise ValueError("unknown diagnostic corpus")


def _audit_bundle(bundle: EpisodeBundle, config: ComparisonConfig) -> None:
    report = validate_episode_invariants(bundle, config.phase1_config)
    if not report.valid:
        raise ValueError("generated diagnostic episode violates invariants")
    solution = solve_public_episode(bundle.public)
    verify_oracle_truth(solution, bundle.truth)
    if bundle.truth.recipe.variant is EpisodeVariant.POSITIVE:
        assert solution.hazard_type is not None
        action = Action(
            solution.hazard_type,
            bundle.truth.activation_time
            + config.phase1_config.data.oracle_timing.action_target_fraction
            * bundle.truth.episode_delay,
            bundle.public.events[-1].event_id,
        )
        actions = (action,)
    else:
        actions = ()
    if not score_actions(bundle.truth, actions).timed_success:
        raise ValueError("generated diagnostic episode fails independent oracle score")


def build_manifest(
    config: ComparisonConfig, name: str, allocation: IndependentAllocation
) -> ComparisonManifest:
    """Generate and audit one allocation; small allocations support fast unit fixtures."""
    root_seed, public_id_seed = _seeds(config, name)
    expected_suite = SuiteName.IID_PRIMARY if name == "iid" else SuiteName.OOD_DEPTH
    if allocation.split_namespace is not SplitNamespace.DEBUG or any(
        block.suite is not expected_suite for block in allocation.blocks
    ):
        raise ValueError("diagnostic allocation must use DEBUG and its declared suite")
    entries = []
    seen_ids: set[str] = set()
    for request in iter_independent_requests(allocation, root_seed):
        bundle = generate_independent_episode(config.phase1_config, request, public_id_seed)
        _audit_bundle(bundle, config)
        public_id = bundle.public.init.episode_public_id
        if public_id in seen_ids:
            raise ValueError("duplicate diagnostic public ID")
        seen_ids.add(public_id)
        entries.append(
            ComparisonEpisodeRef(
                episode_index=request.episode_index,
                requested_path_length=request.requested_path_length,
                variant=request.variant,
                allocation_quartet_index=request.allocation_quartet_index,
                quartet_member_index=request.quartet_member_index,
                public_id=public_id,
                episode_sha256=episode_sha256(bundle),
                accepted_attempt=bundle.truth.recipe.accepted_attempt,
            )
        )
    return ComparisonManifest(
        name=name,
        config_sha256=config.config_sha256,
        protocol_sha256=config.protocol_sha256,
        root_seed=root_seed,
        public_id_seed=public_id_seed,
        allocation=allocation,
        allocation_sha256=sha256_bytes(canonical_json_bytes(allocation)),
        entries=tuple(entries),
    )


def iter_bundles(manifest: ComparisonManifest, config: ComparisonConfig) -> Iterator[EpisodeBundle]:
    """Reconstruct private bundles locally and authenticate every published byte identity."""
    if manifest.split_namespace is not SplitNamespace.DEBUG or manifest.gate_eligible:
        raise ValueError("comparison manifest must be DEBUG and ineligible for gates")
    if manifest.protocol_sha256 != config.protocol_sha256:
        raise ValueError("comparison protocol hash mismatch")
    if manifest.config_sha256 != config.config_sha256:
        raise ValueError("comparison config hash mismatch")
    root_seed, public_id_seed = _seeds(config, manifest.name)
    if (manifest.root_seed, manifest.public_id_seed) != (root_seed, public_id_seed):
        raise ValueError("comparison seed mismatch")
    suite = SuiteName.IID_PRIMARY if manifest.name == "iid" else SuiteName.OOD_DEPTH
    if manifest.allocation_sha256 != sha256_bytes(canonical_json_bytes(manifest.allocation)):
        raise ValueError("diagnostic allocation hash mismatch")
    if manifest.allocation.split_namespace is not SplitNamespace.DEBUG or any(
        block.suite is not suite for block in manifest.allocation.blocks
    ):
        raise ValueError("diagnostic allocation has wrong split or suite")
    expected_requests = tuple(iter_independent_requests(manifest.allocation, root_seed))
    if len(expected_requests) != len(manifest.entries):
        raise ValueError("diagnostic allocation row count mismatch")
    seen_ids: set[str] = set()
    for entry, request in zip(manifest.entries, expected_requests, strict=True):
        if (
            entry.episode_index,
            entry.requested_path_length,
            entry.variant,
            entry.allocation_quartet_index,
            entry.quartet_member_index,
        ) != (
            request.episode_index,
            request.requested_path_length,
            request.variant,
            request.allocation_quartet_index,
            request.quartet_member_index,
        ):
            raise ValueError("diagnostic allocation coordinate mismatch")
        bundle = generate_independent_episode(config.phase1_config, request, public_id_seed)
        if bundle.public.init.episode_public_id != entry.public_id:
            raise ValueError("diagnostic public ID mismatch")
        if episode_sha256(bundle) != entry.episode_sha256:
            raise ValueError("diagnostic episode hash mismatch")
        if bundle.truth.recipe.accepted_attempt != entry.accepted_attempt:
            raise ValueError("diagnostic accepted attempt mismatch")
        if entry.public_id in seen_ids:
            raise ValueError("duplicate diagnostic public ID")
        seen_ids.add(entry.public_id)
        yield bundle


def publish_manifest(manifest: ComparisonManifest, path: Path) -> Path:
    """Write once; compatible repeated preparation authenticates existing bytes."""
    if "frozen" in path.parts:
        raise ValueError("diagnostic manifest cannot be published under frozen")
    payload = canonical_json_bytes(manifest) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("refusing to overwrite a different diagnostic manifest")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def prepare_manifests(
    config: ComparisonConfig, *, output_dir: Path
) -> dict[str, ComparisonManifest]:
    """Publish the exact 512+512 diagnostic protocol without overwriting evidence."""
    result = {}
    for name, allocation in diagnostic_allocations().items():
        path = output_dir / f"{name}.json"
        if path.exists():
            manifest = ComparisonManifest.model_validate_json(path.read_bytes())
            expected_root, expected_public = _seeds(config, name)
            if (
                manifest.name != name
                or manifest.protocol_sha256 != config.protocol_sha256
                or manifest.config_sha256 != config.config_sha256
                or manifest.allocation != allocation
                or manifest.allocation_sha256 != sha256_bytes(canonical_json_bytes(allocation))
                or manifest.root_seed != expected_root
                or manifest.public_id_seed != expected_public
                or len(manifest.entries) != 512
            ):
                raise ValueError("existing diagnostic manifest differs from fixed protocol")
            tuple(iter_bundles(manifest, config))
        else:
            manifest = build_manifest(config, name, allocation)
            if len(manifest.entries) != 512:
                raise ValueError("diagnostic allocation is incomplete")
            publish_manifest(manifest, path)
        result[name] = manifest
    public_ids = {entry.public_id for manifest in result.values() for entry in manifest.entries}
    if len(public_ids) != 1024:
        raise ValueError("diagnostic public IDs overlap")
    return result
