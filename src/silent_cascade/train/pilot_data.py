"""Immutable Phase 4 recipes and counter-addressed teacher batches."""

import os
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from silent_cascade.config import ResolvedConfig
from silent_cascade.env.pilot import _accepted_parent, curriculum_to_bundle
from silent_cascade.eval.artifacts import EpisodeBinding
from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.train.batches import TrainingBatch, pack_training_examples
from silent_cascade.train.curriculum_data import (
    CURRICULUM_VERSION,
    ComponentCorpus,
    CurriculumExample,
    CurriculumKey,
    curriculum_allocation,
    make_curriculum_example,
)
from silent_cascade.train.pilot_config import Phase4Config, parse_phase4_canonical
from silent_cascade.validation import StrictModel

type Stage = Literal["one_hop", "two_hop", "primary", "robustness"]
type Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
MAX_PILOT_MANIFEST_BYTES = 64 * 1024 * 1024
_ROOTS = {"train": (431, 433), "validation": (439, 443), "debug": (449, 457)}


class PilotManifestEntry(StrictModel):
    key: CurriculumKey
    parent_hash: Hash
    parent_public_id: str = Field(min_length=1)
    accepted_attempt: int = Field(ge=0, lt=1000)
    public_hash: Hash
    target_hash: Hash
    example_hash: Hash
    projected: EpisodeBinding
    time_factor: float = Field(ge=0.5, le=2.0)
    nuisance_link_count: int = Field(ge=0, le=12)


class PilotManifest(StrictModel):
    schema_version: Literal["phase4-data-v1"] = "phase4-data-v1"
    experiment: Literal["phase4-pilot-v1"] = "phase4-pilot-v1"
    generator_version: Literal["ofd-v1"] = "ofd-v1"
    transform_version: Literal["ofd-one-hop-v1"] = "ofd-one-hop-v1"
    projection_version: Literal["phase4-projection-v1"] = "phase4-projection-v1"
    stage: Stage
    split: Literal["validation", "debug"]
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_hash: Hash
    config_canonical_json: str
    count: int = Field(ge=16, le=10_000)
    variant_counts: dict[str, int]
    entries: tuple[PilotManifestEntry, ...]
    foundation_model_calls: Literal[0] = 0

    @model_validator(mode="after")
    def validate_inventory(self):
        config = parse_phase4_canonical(self.config_canonical_json)
        if self.config_hash != sha256_bytes(self.config_canonical_json.encode()):
            raise ValueError("pilot configuration hash mismatch")
        split = "validation" if config.pilot.is_production else "debug"
        expected_count = 10_000 if split == "validation" else 16
        if self.split != split or self.count != expected_count or len(self.entries) != self.count:
            raise ValueError("pilot manifest split/count disagree with configuration")
        if len({e.key for e in self.entries}) != self.count:
            raise ValueError("duplicate pilot curriculum keys")
        for values in (
            [e.projected.public_id for e in self.entries],
            [e.public_hash for e in self.entries],
            [e.projected.episode_sha256 for e in self.entries],
        ):
            if len(set(values)) != self.count:
                raise ValueError("duplicate pilot public identity")
        for index, entry in enumerate(self.entries):
            if entry.key != _key(self.stage, self.split, index):
                raise ValueError("pilot requires exact ordered namespace keys")
            _, path, variant = curriculum_allocation(entry.key)
            if entry.projected.variant != variant.value or entry.projected.path_length != (
                1 if self.stage == "one_hop" else path
            ):
                raise ValueError("pilot allocation inventory mismatch")
            if self.stage != "robustness" and entry.time_factor != 1.0:
                raise ValueError("only robustness carries arbitrary scaling")
        expected_variants = {
            "positive": self.count // 2,
            "safe_negative": self.count // 4,
            "disconnected_negative": self.count // 4,
        }
        if (
            self.variant_counts != expected_variants
            or dict(Counter(e.projected.variant for e in self.entries)) != expected_variants
        ):
            raise ValueError("pilot variant counts disagree")
        return self


def _key(stage: str, split: str, index: int) -> CurriculumKey:
    root, public_id = _ROOTS[split]
    return CurriculumKey(CURRICULUM_VERSION, split, root, public_id, index, stage)


def _entry(example: CurriculumExample, config: Phase4Config) -> PilotManifestEntry:
    bundle = curriculum_to_bundle(example, config=config)
    return PilotManifestEntry(
        key=example.key,
        parent_hash=example.parent_hash,
        parent_public_id=_parent_public_id(example, config),
        accepted_attempt=example.accepted_attempt,
        public_hash=example.public_hash,
        target_hash=example.target_hash,
        example_hash=example.example_hash,
        projected=EpisodeBinding.from_bundle(bundle),
        time_factor=example.time_factor,
        nuisance_link_count=example.nuisance_link_count,
    )


def _parent_public_id(example: CurriculumExample, config: Phase4Config) -> str:
    # The original parent ID cannot be inferred from the transformed public ID.
    return _accepted_parent(example, config=config).public.init.episode_public_id


def _check_path(path: Path) -> None:
    parts = path.absolute().parts
    if ".." in parts or "frozen" in parts or "frozen_test" in parts:
        raise ValueError("pilot artifacts cannot access the frozen-test namespace")


def _read_pilot_bytes(path: Path) -> bytes:
    _check_path(path)
    with archive_parent(path, error_factory=ValueError) as (parent, name):
        return read_archive_at(
            parent, name, max_bytes=MAX_PILOT_MANIFEST_BYTES, error_factory=ValueError
        )


def _publish_pilot_bytes(path: Path, payload: bytes) -> None:
    """Create-only durable publication with pinned, no-follow directory FDs."""
    _check_path(path)
    absolute = path.absolute()
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in absolute.parts[1:-1]:
            with suppress(FileExistsError):
                os.mkdir(component, dir_fd=descriptor)
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        temporary = f".pilot-{uuid.uuid4().hex}.tmp"
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
            dir_fd=descriptor,
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(
                    temporary,
                    absolute.name,
                    src_dir_fd=descriptor,
                    dst_dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                if (
                    read_archive_at(
                        descriptor,
                        absolute.name,
                        max_bytes=MAX_PILOT_MANIFEST_BYTES,
                        error_factory=ValueError,
                    )
                    != payload
                ):
                    raise ValueError("pilot artifact already exists; refusing overwrite") from None
            os.fsync(descriptor)
        finally:
            os.unlink(temporary, dir_fd=descriptor)
    finally:
        os.close(descriptor)


def freeze_pilot_manifest(
    config: ResolvedConfig[Phase4Config], *, stage: str, output_path: Path, source_commit: str
) -> PilotManifest:
    _check_path(output_path)
    if config.sha256 != sha256_bytes(
        canonical_json_bytes(config.config)
    ) or config.canonical_json != canonical_json_bytes(config.config):
        raise ValueError("resolved pilot config identity mismatch")
    split = "validation" if config.config.pilot.is_production else "debug"
    count = 10_000 if split == "validation" else 16
    entries = tuple(
        _entry(make_curriculum_example(config.config, _key(stage, split, i)), config.config)
        for i in range(count)
    )
    manifest = PilotManifest(
        stage=stage,
        split=split,
        source_commit=source_commit,
        config_hash=config.sha256,
        config_canonical_json=config.canonical_json.decode(),
        count=count,
        entries=entries,
        variant_counts=dict(Counter(e.projected.variant for e in entries)),
    )
    _publish_pilot_bytes(output_path, canonical_json_bytes(manifest))
    return manifest


def load_pilot_manifest(path: Path, *, config: ResolvedConfig[Phase4Config]) -> PilotManifest:
    payload = _read_pilot_bytes(path)
    manifest = PilotManifest.model_validate_json(payload)
    # A canonical byte check also rejects duplicate JSON keys and nonfinite literals.
    if canonical_json_bytes(manifest) != payload:
        raise ValueError("pilot manifest must be strict canonical JSON")
    if (
        config.sha256 != manifest.config_hash
        or config.canonical_json.decode() != manifest.config_canonical_json
    ):
        raise ValueError("resolved pilot configuration hash mismatch")
    for _ in iter_pilot_examples(manifest, config=config.config):
        pass
    return manifest


def iter_pilot_examples(
    manifest: PilotManifest, *, config: Phase4Config
) -> Iterator[CurriculumExample]:
    # Revalidate even model_copy/model_construct values at this private boundary.
    manifest = PilotManifest.model_validate_json(canonical_json_bytes(manifest))
    if canonical_json_bytes(config).decode() != manifest.config_canonical_json:
        raise ValueError("pilot manifest configuration mismatch")
    for entry in manifest.entries:
        example = make_curriculum_example(config, entry.key)
        if _entry(example, config) != entry:
            raise ValueError("pilot manifest example/projection hash mismatch")
        yield example


def next_pilot_batch(config: Phase4Config, *, stage: str, batch_counter: int) -> TrainingBatch:
    if type(batch_counter) is not int or batch_counter < 0:
        raise ValueError("pilot batch counter must be a nonnegative exact integer")
    start = batch_counter * config.pilot.batch_size
    examples = tuple(
        make_curriculum_example(config, _key(stage, "train", i))
        for i in range(start, start + config.pilot.batch_size)
    )
    return pack_training_examples(examples, next_batch_counter=batch_counter + 1)


def pilot_component_corpus(manifest: PilotManifest, *, config: Phase4Config) -> ComponentCorpus:
    from silent_cascade.train.traces import build_teacher_trace, component_target

    if manifest.stage != "one_hop":
        raise ValueError("component corpus requires the one-hop stage")
    examples = tuple(iter_pilot_examples(manifest, config=config))
    return ComponentCorpus(
        tuple(e.public for e in examples),
        tuple(component_target(build_teacher_trace(e)) for e in examples),
    )
