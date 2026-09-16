"""Private, counter-addressed curriculum transforms and component corpus recipes.

Only this training layer solves graphs. Public projections never contain these
keys, parent recipes, solutions, or terminal horizons.
"""

import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from silent_cascade.config import canonical_json_bytes
from silent_cascade.env.config import OracleTimingConfig, Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    EpisodeRecipe,
    EpisodeVariant,
    PublicEpisode,
    episode_sha256,
)
from silent_cascade.env.generator import IndependentEpisodeRequest, generate_independent_episode
from silent_cascade.env.oracle import (
    OracleSolution,
    OracleTrace,
    build_oracle_trace,
    solve_public_episode,
)
from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at
from silent_cascade.rng import AllocationLabelKey, allocate_independent_variants
from silent_cascade.schemas import (
    ActivationPayload,
    AgentInit,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    LinkFact,
    SafeFact,
)
from silent_cascade.train.config import Phase3Config

CURRICULUM_VERSION = "ofd-one-hop-v1"
MAX_PARENT_ATTEMPTS = 1000
MAX_COMPONENT_MANIFEST_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CurriculumKey:
    version: str
    split: str
    root_seed: int
    public_id_seed: int
    episode_index: int
    stage: str

    def __post_init__(self) -> None:
        if self.version != CURRICULUM_VERSION:
            raise ValueError("unsupported curriculum version")
        if self.split not in ("train", "validation", "debug"):
            raise ValueError("curriculum split must be train, validation, or debug")
        if self.stage not in ("one_hop", "two_hop", "primary", "robustness"):
            raise ValueError("unsupported curriculum stage")
        for name in ("root_seed", "public_id_seed", "episode_index"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative exact integer")
        if max(self.root_seed, self.public_id_seed) >= 2**128:
            raise ValueError("curriculum root and public-ID seeds must fit 128 bits")


def _phase3_seed(stage, split, root_seed, episode_index, stream_name, attempt):
    material = json.dumps(
        [
            "silent-cascade/phase3/data/v1",
            stage,
            split,
            root_seed,
            episode_index,
            stream_name,
            attempt,
        ],
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest(), "big")


def curriculum_rng(key: CurriculumKey, stream_name: str, attempt: int = 0) -> np.random.Generator:
    return np.random.default_rng(
        _phase3_seed(
            key.stage,
            key.split,
            key.root_seed,
            key.episode_index,
            stream_name,
            attempt,
        )
    )


def _hash(domain: str, value: object) -> str:
    return hashlib.sha256(canonical_json_bytes({"domain": domain, "value": value})).hexdigest()


def public_episode_hash(public: PublicEpisode) -> str:
    return _hash("silent-cascade/phase3/public/v1", asdict(public))


def _public_id(key: CurriculumKey) -> str:
    material = canonical_json_bytes(
        {
            "domain": "silent-cascade/phase3/public-id/v1",
            "key": asdict(key),
        }
    )
    digest = hmac.new(key.public_id_seed.to_bytes(16, "big"), material, hashlib.sha256).digest()
    return str(UUID(bytes=digest[:16], version=4))


def curriculum_allocation(key: CurriculumKey) -> tuple[int, int, EpisodeVariant]:
    """A quartet shares the root/path/label key across members and all retries."""
    quartet = replace(key, episode_index=key.episode_index // 4 * 4)
    # The frozen generator requires 128-bit roots; NumPy itself accepts all 256 bits.
    root = (
        _phase3_seed(
            quartet.stage, quartet.split, quartet.root_seed, quartet.episode_index, "parent", 0
        )
        % 2**128
    )
    path = (
        2
        if key.stage in ("one_hop", "two_hop")
        else int(curriculum_rng(quartet, "augmentation").integers(2, 5))
    )
    variants = allocate_independent_variants(
        AllocationLabelKey(
            "ofd-v1",
            SplitNamespace(key.split),
            SuiteName.IID_PRIMARY,
            root,
            path,
            key.episode_index // 4,
        )
    )
    return root, path, variants[key.episode_index % 4]


@dataclass(frozen=True, slots=True)
class CurriculumExample:
    key: CurriculumKey
    public: PublicEpisode
    solution: OracleSolution
    oracle_trace: OracleTrace
    terminal_horizon: ExternalEvent
    parent_recipe: EpisodeRecipe
    parent_hash: str
    transform_identity: str
    accepted_attempt: int
    nuisance_link_count: int
    variant: EpisodeVariant
    time_factor: float
    paired_unscaled_public: PublicEpisode | None
    paired_unscaled_trace: OracleTrace | None
    public_hash: str
    target_hash: str
    example_hash: str
    foundation_model_calls: Literal[0] = 0

    @property
    def init(self) -> AgentInit:
        return self.public.init

    @property
    def events(self) -> tuple[ExternalEvent, ...]:
        return self.public.events


def make_curriculum_example(config: Phase1Config, key: CurriculumKey) -> CurriculumExample:
    if not isinstance(config, Phase1Config) or not isinstance(key, CurriculumKey):
        raise TypeError("expected Phase1Config and CurriculumKey")
    root, path, variant = curriculum_allocation(key)
    desired = int(curriculum_rng(key, "subset").integers(0, 5))
    for attempt in range(MAX_PARENT_ATTEMPTS):
        request = IndependentEpisodeRequest(
            SplitNamespace(key.split),
            SuiteName.IID_PRIMARY,
            root,
            key.episode_index * MAX_PARENT_ATTEMPTS + attempt,
            path,
            variant,
            key.episode_index // 4,
            key.episode_index % 4,
        )
        parent = generate_independent_episode(config, request, key.public_id_seed)
        solved = solve_public_episode(parent.public)
        facts = parent.public.events[:-1]
        activation = parent.public.events[-1].payload
        assert isinstance(activation, ActivationPayload)
        if key.stage == "one_hop":
            off_path = tuple(
                event
                for event in facts
                if isinstance(event.payload, LinkFact)
                and event.event_id not in solved.link_record_ids
            )
            if len(off_path) < desired:
                continue
            last_link = next(
                event for event in facts if event.event_id == solved.link_record_ids[-1]
            )
            assert isinstance(last_link.payload, LinkFact)
            activation = ActivationPayload(last_link.payload.source_node)
            selected = curriculum_rng(key, "subset", attempt).choice(
                len(off_path), desired, replace=False
            )
            nuisance = tuple(off_path[int(index)] for index in selected)
            reachable = {last_link.payload.source_node, last_link.payload.target_node}
            if any(event.payload.source_node in reachable for event in nuisance):
                raise ValueError("one-hop nuisance source is reachable")
            facts = (
                last_link,
                *nuisance,
                *(event for event in facts if isinstance(event.payload, (HazardFact, SafeFact))),
            )
        break
    else:
        raise ValueError("one-hop parent construction exhausted 1000 attempts")

    order = curriculum_rng(key, "presentation", attempt).permutation(len(facts))
    gaps = np.exp(
        curriculum_rng(key, "timestamps", attempt).uniform(
            math.log(0.1), math.log(8.0), len(facts) + 1
        )
    )
    times = np.cumsum(gaps)
    events = tuple(
        ExternalEvent(i, float(times[i]), ExternalEventKind.FACT, facts[int(index)].payload)
        for i, index in enumerate(order)
    )
    events += (ExternalEvent(len(facts), float(times[-1]), ExternalEventKind.ACTIVATE, activation),)
    public = PublicEpisode(
        replace(parent.public.init, episode_public_id=_public_id(key), initial_time=0.0), events
    )
    terminal = replace(
        parent.truth.private_terminal,
        event_id=len(events),
        timestamp=events[-1].timestamp + parent.truth.episode_delay,
    )
    solution = solve_public_episode(public)
    trace = build_oracle_trace(
        public,
        solution,
        terminal,
        config.data.oracle_timing,
        curriculum_rng(key, "trace_jitter", attempt),
    )
    paired_public = paired_trace = None
    factor = 1.0
    if key.stage == "robustness":
        paired_public, paired_trace = public, trace
        factor = float(
            np.exp(
                curriculum_rng(key, "augmentation", attempt).uniform(math.log(0.5), math.log(2.0))
            )
        )
        scaled_events = tuple(
            replace(
                event,
                timestamp=event.timestamp * factor,
                payload=replace(event.payload, delay=event.payload.delay * factor)
                if isinstance(event.payload, HazardFact)
                else event.payload,
            )
            for event in events
        )
        public = PublicEpisode(public.init, scaled_events)
        terminal = replace(terminal, timestamp=terminal.timestamp * factor)
        timing_values = config.data.oracle_timing.model_dump()
        for name in ("delta_0", "delta_min", "delta_max"):
            timing_values[name] *= factor
        solution = solve_public_episode(public)
        trace = build_oracle_trace(
            public,
            solution,
            terminal,
            OracleTimingConfig(**timing_values),
            curriculum_rng(key, "trace_jitter", attempt),
        )
    expected_links = 1 if key.stage == "one_hop" else path
    if (
        len(solution.link_record_ids) != expected_links
        or solution.terminal_kind != solved.terminal_kind
    ):
        raise ValueError("curriculum transform changed graph semantics")
    if (
        sum(isinstance(e.payload, HazardFact) for e in public.events) != 2
        or sum(isinstance(e.payload, SafeFact) for e in public.events) != 1
    ):
        raise ValueError("curriculum transform changed terminal counts")
    if (
        len(facts) > config.data.primary_memory_capacity
        or len(trace.steps) > 2 * (expected_links + 1) + 1
    ):
        raise ValueError("curriculum exceeded capacity")
    parent_hash = episode_sha256(parent)
    public_hash = public_episode_hash(public)
    target_hash = _hash(
        "silent-cascade/phase3/target/v1",
        {
            "solution": asdict(solution),
            "trace": asdict(trace),
            "terminal": asdict(terminal),
        },
    )
    identity = f"{CURRICULUM_VERSION}/{key.stage}"
    example_hash = _hash(
        "silent-cascade/phase3/example/v1",
        {
            "key": asdict(key),
            "public_hash": public_hash,
            "target_hash": target_hash,
            "parent_hash": parent_hash,
            "transform": identity,
            "accepted_attempt": attempt,
        },
    )
    return CurriculumExample(
        key,
        public,
        solution,
        trace,
        terminal,
        parent.truth.recipe,
        parent_hash,
        identity,
        attempt,
        len(facts) - expected_links - 3,
        variant,
        factor,
        paired_public,
        paired_trace,
        public_hash,
        target_hash,
        example_hash,
    )


@dataclass(frozen=True, slots=True)
class ComponentContent:
    """One ordered composition target; None marks an unused output field."""

    record_id: int
    role: int
    focus: int | None
    hazard_type: int | None
    log_delay: float | None
    normalized_deadline: float | None
    status: int | None
    confidence: float
    append_support: bool
    continue_search: bool | None


@dataclass(frozen=True, slots=True)
class ComponentTarget:
    """Immutable scoring-only content chain; never passed to a model."""

    record_ids: tuple[int, ...]
    content: tuple[ComponentContent, ...]
    terminal_class: int
    terminal_status: int


@dataclass(frozen=True, slots=True)
class ComponentCorpus:
    public_examples: tuple[PublicEpisode, ...]
    targets: tuple[ComponentTarget, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.public_examples, tuple)
            or not isinstance(self.targets, tuple)
            or len(self.public_examples) != len(self.targets)
        ):
            raise ValueError("component public and target tuples must have equal lengths")

    def public_batch(self, start: int, count: int):
        from silent_cascade.train.batches import pack_public_examples

        self._check_slice(start, count)
        return pack_public_examples(self.public_examples[start : start + count])

    def scoring_slice(self, start: int, count: int) -> tuple[ComponentTarget, ...]:
        self._check_slice(start, count)
        return self.targets[start : start + count]

    def _check_slice(self, start: int, count: int) -> None:
        if (
            type(start) is not int
            or type(count) is not int
            or start < 0
            or not 1 <= count <= 128
            or start + count > len(self.public_examples)
        ):
            raise ValueError("component slice must be in bounds with 1 to 128 rows")


class ComponentManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    key: CurriculumKey
    public_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    example_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted_attempt: int = Field(ge=0, lt=1000)


class ComponentManifest(BaseModel):
    """Bounded recipes; source-history authentication is an independent gate."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["phase3-component-manifest-v1"] = "phase3-component-manifest-v1"
    publication: Literal["debug", "production"] = "debug"
    curriculum_version: Literal["ofd-one-hop-v1"] = "ofd-one-hop-v1"
    generator_version: Literal["ofd-v1"] = "ofd-v1"
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    plan_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    count: int = Field(ge=1, le=10_000)
    entries: tuple[ComponentManifestEntry, ...] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_entries(self):
        if len(self.entries) != self.count or len({e.key for e in self.entries}) != self.count:
            raise ValueError("manifest counts or unique keys disagree")
        if len({e.public_hash for e in self.entries}) != self.count:
            raise ValueError("manifest public hashes must be unique")
        if self.publication == "production":
            if self.count != 10_000:
                raise ValueError("production component manifests require exactly 10000 episodes")
            for index, entry in enumerate(self.entries):
                key = entry.key
                if (key.split, key.root_seed, key.public_id_seed, key.episode_index, key.stage) != (
                    "validation",
                    313,
                    337,
                    index,
                    "one_hop",
                ):
                    raise ValueError("production manifest must use exact ordered validation keys")
        return self

    @classmethod
    def from_examples(
        cls,
        examples: tuple[CurriculumExample, ...],
        config: Phase3Config,
        *,
        source_revision: str,
        plan_revision: str,
    ):
        return cls(
            source_revision=source_revision,
            plan_revision=plan_revision,
            config_hash=hashlib.sha256(canonical_json_bytes(config)).hexdigest(),
            count=len(examples),
            entries=tuple(
                ComponentManifestEntry(
                    key=e.key,
                    public_hash=e.public_hash,
                    example_hash=e.example_hash,
                    accepted_attempt=e.accepted_attempt,
                )
                for e in examples
            ),
        )

    def _examples(self, config: Phase3Config) -> tuple[CurriculumExample, ...]:
        if self.config_hash != hashlib.sha256(canonical_json_bytes(config)).hexdigest():
            raise ValueError("manifest configuration hash mismatch")
        if self.publication == "production" and (
            not config.training.is_production or config.neural.architecture_profile != "production"
        ):
            raise ValueError("production manifest requires the production one-hop configuration")
        examples = tuple(make_curriculum_example(config, entry.key) for entry in self.entries)
        for entry, example in zip(self.entries, examples, strict=True):
            if (entry.public_hash, entry.example_hash, entry.accepted_attempt) != (
                example.public_hash,
                example.example_hash,
                example.accepted_attempt,
            ):
                raise ValueError("manifest example hash or accepted attempt mismatch")
        if self.publication == "production":
            counts = {
                variant: sum(e.variant == variant for e in examples) for variant in EpisodeVariant
            }
            if counts != {
                EpisodeVariant.POSITIVE: 5000,
                EpisodeVariant.SAFE_NEGATIVE: 2500,
                EpisodeVariant.DISCONNECTED_NEGATIVE: 2500,
            }:
                raise ValueError("production manifest variant allocation mismatch")
        return examples

    def build_corpus(self, config: Phase3Config) -> ComponentCorpus:
        from silent_cascade.train.traces import build_teacher_trace, component_target

        examples = self._examples(config)
        return ComponentCorpus(
            tuple(e.public for e in examples),
            tuple(component_target(build_teacher_trace(e)) for e in examples),
        )


def load_component_manifest(path: Path, config: Phase3Config) -> ComponentManifest:
    with archive_parent(path, error_factory=ValueError) as (parent, name):
        payload = read_archive_at(
            parent, name, max_bytes=MAX_COMPONENT_MANIFEST_BYTES, error_factory=ValueError
        )
    manifest = ComponentManifest.model_validate_json(payload)
    manifest._examples(config)
    return manifest
