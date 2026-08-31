"""Fail-closed, deterministic audits for public OFD shortcut features.

The auditor is deliberately the sole Phase 1 component that sees an
``EpisodeBundle``.  Feature extraction reads only ``bundle.public``; private
truth is used only to construct the explicitly declared audit targets and, in
named positive-control mode, to synthesize a known leak.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import pairwise, permutations
from pathlib import Path
from tempfile import mkdtemp
from typing import Literal, NoReturn, Protocol

import numpy as np
import psutil
from pydantic import Field, model_validator
from scipy.optimize import minimize
from scipy.special import logsumexp

from silent_cascade.env.config import LeakageAuditConfig, Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    CorpusDigestEntry,
    CorpusHashBuilder,
    EpisodeBundle,
    EpisodeVariant,
    IndependentEpisodeCoordinate,
    MatchedEpisodeCoordinate,
    PublicEpisode,
    episode_sha256,
    scale_episode_time,
)
from silent_cascade.env.invariants import validate_cohort_invariants, validate_episode_invariants
from silent_cascade.env.oracle import solve_public_episode, verify_oracle_truth
from silent_cascade.env.timing import action_window
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.provenance import EvidenceProvenance, LeakageAuditEvidenceAnchor
from silent_cascade.schemas import (
    ActivationPayload,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    LinkFact,
    SafeFact,
)
from silent_cascade.validation import StrictModel

type HexDigest = str
_FEATURE_DIMENSIONS = (17, 256, 278, 4, 768, 194, 10, 33)
_TOTAL_FEATURE_DIMENSION = sum(_FEATURE_DIMENSIONS)


class LeakageAuditProfileName(StrEnum):
    TEST = "test"
    PHASE1_GATE = "phase1_gate"


class ShortcutTask(StrEnum):
    POSITIVE_BINARY = "positive_binary"
    VARIANT_THREE_WAY = "variant_three_way"
    POSITIVE_HAZARD_CLASS = "positive_hazard_class"


class ShortcutFeatureGroup(StrEnum):
    ID_POSITION = "id_position"
    ACTIVATION_NODE = "activation_node"
    FIRST_LAST_FACT = "first_last_fact"
    COUNTS = "counts"
    ORDER_RECORD_IDS = "order_record_ids"
    TIMES = "times"
    TERMINAL_MULTISET = "terminal_multiset"
    LINK_TOPOLOGY = "link_topology"
    COMBINED = "combined"


_GROUP_SLICES = {
    ShortcutFeatureGroup.ID_POSITION: slice(0, 17),
    ShortcutFeatureGroup.ACTIVATION_NODE: slice(17, 273),
    ShortcutFeatureGroup.FIRST_LAST_FACT: slice(273, 551),
    ShortcutFeatureGroup.COUNTS: slice(551, 555),
    ShortcutFeatureGroup.ORDER_RECORD_IDS: slice(555, 1323),
    ShortcutFeatureGroup.TIMES: slice(1323, 1517),
    ShortcutFeatureGroup.TERMINAL_MULTISET: slice(1517, 1527),
    ShortcutFeatureGroup.LINK_TOPOLOGY: slice(1527, 1560),
}


@dataclass(frozen=True, slots=True)
class AuditSeedKey:
    schema_version: Literal["leakage-v1"]
    audit_seed: int
    corpus_hash: HexDigest
    task: ShortcutTask
    replicate_index: int
    suite: SuiteName
    requested_path_length: int
    randomization_block_index: int
    episode_position: int


@dataclass(frozen=True, slots=True)
class ShortcutFeatureSet:
    schema_version: Literal["leakage-features-v1"]
    episode_public_id: str
    generation_mode: Literal["matched", "independent"]
    audit_group_id: str
    suite: SuiteName
    requested_path_length: int
    vectors: Mapping[ShortcutFeatureGroup, np.ndarray]


class ShortcutProbeResult(StrictModel):
    task: ShortcutTask
    feature_group: ShortcutFeatureGroup
    feature_dimension: int
    train_examples: int
    test_examples: int
    train_class_counts: dict[str, int]
    test_class_counts: dict[str, int]
    raw_accuracy: float
    balanced_accuracy: float
    balanced_chance: float
    raw_permutation_p: float
    holm_adjusted_p: float
    optimizer_iterations: int
    optimizer_converged: bool
    passed: bool


class PositiveControlResult(StrictModel):
    control_id: str
    target_task: ShortcutTask
    expected_detector_id: str
    observed_detector_ids: tuple[str, ...]
    base_subset_corpus_sha256: HexDigest
    injected_corpus_sha256: HexDigest
    split_membership_sha256: HexDigest
    balanced_accuracy: float | None
    holm_adjusted_p: float | None
    passed: bool


@dataclass(frozen=True, slots=True)
class AuditExample:
    bundle: EpisodeBundle
    manifest_rank: int
    generation_mode: Literal["matched", "independent"]
    randomization_block_index: int
    episode_position: int


class AuditSourceDescriptor(StrictModel):
    schema_version: Literal["leakage-source-v1"]
    generation_mode: Literal["matched", "independent"]
    allocation_id: str
    allocation_or_manifest_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    split_namespace: SplitNamespace
    root_seed: int
    public_id_seed_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    generator_source_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    episode_count: int = Field(gt=0, multiple_of=4)


def audit_source_descriptor_sha256(descriptor: AuditSourceDescriptor) -> str:
    """Hash the descriptor independently authenticated by the source constructor."""
    if not isinstance(descriptor, AuditSourceDescriptor):
        raise TypeError("descriptor must be an AuditSourceDescriptor")
    return sha256_bytes(canonical_json_bytes(descriptor))


class AuditSourceAuthentication(StrictModel):
    """Source-agnostic evidence supplied by an independently authenticated source."""

    schema_version: Literal["leakage-source-auth-v1"]
    profile: LeakageAuditProfileName
    descriptor_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    source_manifest_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    suite_path_denominators: dict[str, int]
    clock_pair_manifest_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    clock_scale_pair_counts: dict[Literal["scale_0_1x", "scale_10x"], int]

    @model_validator(mode="after")
    def require_positive_complete_counts(self) -> AuditSourceAuthentication:
        if (
            not self.suite_path_denominators
            or any(
                not key or type(value) is not int or value <= 0
                for key, value in self.suite_path_denominators.items()
            )
            or set(self.clock_scale_pair_counts) != {"scale_0_1x", "scale_10x"}
            or any(
                type(value) is not int or value <= 0
                for value in self.clock_scale_pair_counts.values()
            )
        ):
            raise ValueError("authenticated profile counts must be complete positive integers")
        return self


class ReiterableAuditSource(Protocol):
    @property
    def descriptor(self) -> AuditSourceDescriptor: ...

    @property
    def episode_count(self) -> int: ...

    @property
    def authentication(self) -> AuditSourceAuthentication: ...

    @property
    def publishable(self) -> bool: ...

    def iter_examples(self) -> Iterator[AuditExample]: ...

    def iter_clock_pairs(self) -> Iterator[PairedClockAuditPair]: ...


@dataclass(frozen=True, slots=True)
class PairedClockAuditPair:
    """One independently authenticated unscaled parent and paired clock child."""

    parent: AuditExample
    child: EpisodeBundle

    def __post_init__(self) -> None:
        if not isinstance(self.parent, AuditExample) or not isinstance(self.child, EpisodeBundle):
            raise TypeError("paired clock audit values must be an audit parent and episode child")
        if self.child.truth.recipe.evaluation_suite not in {
            SuiteName.CLOCK_SCALE_0_1X,
            SuiteName.CLOCK_SCALE_10X,
        }:
            raise ValueError("paired clock child has no declared clock scale")


class _SourceManifestHashBuilder:
    """Accumulate the canonical source-manifest JSON without retaining its entries."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(
            b'{"domain":"silent-cascade/ofd-v1/leakage-source-manifest/v1","episodes":['
        )
        self._count = 0
        self._finalized = False

    def add(self, example: AuditExample, digest: str) -> None:
        if self._finalized:
            raise RuntimeError("cannot add to a finalized source manifest")
        if self._count:
            self._digest.update(b",")
        self._digest.update(
            canonical_json_bytes(
                {
                    "manifest_rank": example.manifest_rank,
                    "generation_mode": example.generation_mode,
                    "randomization_block_index": example.randomization_block_index,
                    "episode_position": example.episode_position,
                    "public_id": example.bundle.public.init.episode_public_id,
                    "episode_sha256": digest,
                }
            )
        )
        self._count += 1

    def finalize(self) -> str:
        if self._finalized:
            raise RuntimeError("source manifest has already been finalized")
        self._finalized = True
        self._digest.update(b"]}")
        return self._digest.hexdigest()


def _source_manifest_sha256(examples: Sequence[AuditExample]) -> str:
    """Bind a source descriptor to its exact ordered artifact manifest."""
    builder = _SourceManifestHashBuilder()
    for example in examples:
        builder.add(example, episode_sha256(example.bundle))
    return builder.finalize()


@dataclass(frozen=True, slots=True)
class InMemoryAuditSource:
    descriptor: AuditSourceDescriptor
    examples: tuple[AuditExample, ...]
    validation_config: Phase1Config | None = None
    authentication: AuditSourceAuthentication | None = None
    clock_pairs: tuple[PairedClockAuditPair, ...] = ()
    publishable: Literal[False] = False

    @property
    def episode_count(self) -> int:
        return len(self.examples)

    @property
    def manifest_sha256(self) -> str:
        return _source_manifest_sha256(self.examples)

    def iter_examples(self) -> Iterator[AuditExample]:
        return iter(self.examples)

    def iter_clock_pairs(self) -> Iterator[PairedClockAuditPair]:
        return iter(self.clock_pairs)


def _clock_pair_manifest_sha256(pairs: Sequence[PairedClockAuditPair]) -> str:
    builder = _ClockPairManifestHashBuilder()
    for pair in pairs:
        builder.add(pair)
    return builder.finalize()


class _ClockPairManifestHashBuilder:
    """Accumulate the authenticated clock-pair manifest in bounded memory."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(
            b'{"domain":"silent-cascade/ofd-v1/leakage-clock-pair-manifest/v1","pairs":['
        )
        self._count = 0
        self._finalized = False

    def add(self, pair: PairedClockAuditPair) -> None:
        if self._finalized:
            raise RuntimeError("clock-pair manifest has already been finalized")
        if self._count:
            self._digest.update(b",")
        self._digest.update(
            canonical_json_bytes(
                {
                    "parent_manifest_rank": pair.parent.manifest_rank,
                    "parent_public_id": pair.parent.bundle.public.init.episode_public_id,
                    "parent_episode_sha256": episode_sha256(pair.parent.bundle),
                    "scale": pair.child.truth.recipe.clock_scale,
                    "child_public_id": pair.child.public.init.episode_public_id,
                    "child_episode_sha256": episode_sha256(pair.child),
                }
            )
        )
        self._count += 1

    def finalize(self) -> str:
        if self._finalized:
            raise RuntimeError("clock-pair manifest has already been finalized")
        self._finalized = True
        self._digest.update(b"]}")
        return self._digest.hexdigest()


@dataclass(frozen=True, slots=True)
class NamedLeakInjector:
    control_id: str
    target_task: ShortcutTask
    expected_detector_id: str
    apply: Callable[[ReiterableAuditSource], ReiterableAuditSource]


@dataclass(frozen=True, slots=True)
class _InjectedAuditSource:
    """A source tagged for an explicitly authorized positive-control overlay."""

    source: ReiterableAuditSource
    injector: NamedLeakInjector

    @property
    def descriptor(self) -> AuditSourceDescriptor:
        return self.source.descriptor

    @property
    def episode_count(self) -> int:
        return self.source.episode_count

    @property
    def manifest_sha256(self) -> str:
        return self.source.manifest_sha256  # type: ignore[attr-defined]

    @property
    def validation_config(self) -> Phase1Config | None:
        return getattr(self.source, "validation_config", None)

    def iter_examples(self) -> Iterator[AuditExample]:
        return self.source.iter_examples()

    @property
    def authentication(self) -> AuditSourceAuthentication:
        return self.source.authentication

    @property
    def publishable(self) -> bool:
        return self.source.publishable

    def iter_clock_pairs(self) -> Iterator[PairedClockAuditPair]:
        return self.source.iter_clock_pairs()


def _injector(
    control_id: str, task: ShortcutTask, group: ShortcutFeatureGroup
) -> NamedLeakInjector:
    expected_detector_id = f"{task.value}:{group.value}"

    def apply(source: ReiterableAuditSource) -> ReiterableAuditSource:
        if not isinstance(source.descriptor, AuditSourceDescriptor):
            raise TypeError("positive control requires an audit source")
        return _InjectedAuditSource(
            source,
            NamedLeakInjector(control_id, task, expected_detector_id, apply),
        )

    return NamedLeakInjector(control_id, task, expected_detector_id, apply)


leak_record_count = _injector(
    "PC_COUNT_BY_LABEL", ShortcutTask.POSITIVE_BINARY, ShortcutFeatureGroup.COUNTS
)
leak_timestamp = _injector(
    "PC_ACTIVATION_GAP_BY_LABEL", ShortcutTask.POSITIVE_BINARY, ShortcutFeatureGroup.TIMES
)
leak_terminal_order = _injector(
    "PC_TERMINAL_ORDER_BY_VARIANT",
    ShortcutTask.VARIANT_THREE_WAY,
    ShortcutFeatureGroup.ORDER_RECORD_IDS,
)
leak_node_range = _injector(
    "PC_ACTIVATION_ID_BY_LABEL",
    ShortcutTask.POSITIVE_BINARY,
    ShortcutFeatureGroup.ACTIVATION_NODE,
)
leak_record_id = _injector(
    "PC_RECORD_ID_BY_VARIANT",
    ShortcutTask.VARIANT_THREE_WAY,
    ShortcutFeatureGroup.ORDER_RECORD_IDS,
)
leak_public_id = _injector(
    "PC_PUBLIC_ID_BY_LABEL", ShortcutTask.POSITIVE_BINARY, ShortcutFeatureGroup.ID_POSITION
)
leak_delay_layout = _injector(
    "PC_DELAY_BY_LABEL", ShortcutTask.POSITIVE_BINARY, ShortcutFeatureGroup.TERMINAL_MULTISET
)
leak_hazard_layout = _injector(
    "PC_HAZARD_LAYOUT_BY_CLASS",
    ShortcutTask.POSITIVE_HAZARD_CLASS,
    ShortcutFeatureGroup.ORDER_RECORD_IDS,
)
leak_manifest_order = _injector(
    "PC_MANIFEST_ORDER_BY_VARIANT",
    ShortcutTask.VARIANT_THREE_WAY,
    ShortcutFeatureGroup.ID_POSITION,
)
NAMED_LEAK_INJECTORS = (
    leak_record_count,
    leak_timestamp,
    leak_terminal_order,
    leak_node_range,
    leak_record_id,
    leak_public_id,
    leak_delay_layout,
    leak_hazard_layout,
    leak_manifest_order,
)


def _rebuild_control_bundle(
    bundle: EpisodeBundle,
    facts: Sequence[ExternalEvent],
    *,
    event_ids: Sequence[int] | None = None,
    activation_time: float | None = None,
    initial_time: float | None = None,
    public_id: str | None = None,
    relevant_node_path: tuple[int, ...] | None = None,
    hazard_type: int | None = None,
    delay: float | None = None,
    preserve_fact_timestamps: bool = False,
    fact_timestamps: Sequence[float] | None = None,
) -> EpisodeBundle:
    """Rebind private auditor truth to one deliberately rewritten public artifact."""
    original_facts = _fact_events(bundle)
    activation = bundle.public.events[-1]
    activation_time = activation.timestamp if activation_time is None else activation_time
    initial_time = bundle.public.init.initial_time if initial_time is None else initial_time
    ids = tuple(range(len(facts))) if event_ids is None else tuple(event_ids)
    if len(ids) != len(facts) or len(set(ids)) != len(ids):
        raise ValueError("positive-control FACT IDs must be complete and unique")
    if fact_timestamps is not None:
        if len(fact_timestamps) != len(facts):
            raise ValueError("positive-control FACT timestamps must be complete")
        timestamps = tuple(float(timestamp) for timestamp in fact_timestamps)
    elif preserve_fact_timestamps:
        timestamps = tuple(event.timestamp for event in facts)
    elif len(facts) == len(original_facts):
        timestamps = tuple(sorted(event.timestamp for event in original_facts))
    else:
        start = initial_time
        timestamps = tuple(
            float(start + (activation_time - start) * (index + 1) / (len(facts) + 1))
            for index in range(len(facts))
        )
    rewritten = tuple(
        ExternalEvent(event_id, timestamp, ExternalEventKind.FACT, event.payload)
        for event_id, timestamp, event in zip(ids, timestamps, facts, strict=True)
    )
    old_to_new = {
        event.event_id: event_id
        for event, event_id in zip(facts, ids, strict=True)
        if event.event_id in {item.event_id for item in original_facts}
    }
    activation_id = 384 if event_ids is not None else len(facts)
    activation_payload = activation.payload
    if relevant_node_path is not None:
        activation_payload = ActivationPayload(relevant_node_path[0])
    public = PublicEpisode(
        replace(
            bundle.public.init,
            initial_time=float(initial_time),
            episode_public_id=(
                bundle.public.init.episode_public_id if public_id is None else public_id
            ),
        ),
        (
            *rewritten,
            ExternalEvent(
                activation_id,
                float(activation_time),
                ExternalEventKind.ACTIVATE,
                activation_payload,
            ),
        ),
    )
    truth = bundle.truth
    actual_delay = truth.episode_delay if delay is None else delay
    actual_hazard = truth.relevant_hazard_type if hazard_type is None else hazard_type
    path = truth.relevant_node_path if relevant_node_path is None else relevant_node_path
    relevant_ids = tuple(old_to_new[item] for item in truth.relevant_record_ids)
    terminal_id = None if truth.terminal_record_id is None else old_to_new[truth.terminal_record_id]
    window = (
        action_window(float(activation_time), actual_delay, truth.recipe.oracle_timing)
        if truth.recipe.variant is EpisodeVariant.POSITIVE
        else None
    )
    truth = replace(
        truth,
        recipe=replace(
            truth.recipe,
            distractor_link_count=sum(isinstance(event.payload, LinkFact) for event in facts)
            - truth.recipe.requested_path_length,
        ),
        relevant_node_path=path,
        relevant_record_ids=relevant_ids,
        terminal_record_id=terminal_id,
        relevant_hazard_type=(
            actual_hazard if truth.recipe.variant is EpisodeVariant.POSITIVE else None
        ),
        private_terminal=replace(
            truth.private_terminal,
            event_id=activation_id + 1,
            timestamp=float(activation_time + actual_delay),
        ),
        activation_time=float(activation_time),
        episode_delay=float(actual_delay),
        action_window_start=None if window is None else window.start,
        action_window_end=None if window is None else window.end,
        action_target=None if window is None else window.target,
    )
    return EpisodeBundle(public, truth)


def _unused_link(facts: Sequence[ExternalEvent], forbidden_nodes: set[int]) -> LinkFact:
    used = {
        (event.payload.source_node, event.payload.target_node)
        for event in facts
        if isinstance(event.payload, LinkFact)
    }
    adjacency: dict[int, set[int]] = defaultdict(set)
    for source, target in used:
        adjacency[source].add(target)

    def reaches(source: int, target: int) -> bool:
        pending = [source]
        visited: set[int] = set()
        while pending:
            node = pending.pop()
            if node == target:
                return True
            if node not in visited:
                visited.add(node)
                pending.extend(adjacency[node])
        return False

    for source in range(64):
        for target in range(64):
            if (
                source != target
                and source not in forbidden_nodes
                and target not in forbidden_nodes
                and (source, target) not in used
                and not reaches(target, source)
            ):
                return LinkFact(source, target)
    raise ValueError("positive control cannot allocate an unreachable sentinel LINK")


def _rewrite_positive_control(
    injector: NamedLeakInjector,
    example: AuditExample,
    corpus_position: int,
    *,
    encoded_manifest_rank: int | None = None,
    injected_hazard_target: int | None = None,
    minimum_observation_gap: float = 0.1,
) -> AuditExample:
    """Construct one exact declared public control after subset/split freezing."""
    bundle = example.bundle
    positive = bundle.truth.recipe.variant is EpisodeVariant.POSITIVE
    variant_index = list(EpisodeVariant).index(bundle.truth.recipe.variant)
    facts = list(_fact_events(bundle))
    if injector.control_id == "PC_COUNT_BY_LABEL":
        target = 48 if positive else 56
        required = set(bundle.truth.relevant_record_ids)
        terminals = {
            event.event_id for event in facts if isinstance(event.payload, (HazardFact, SafeFact))
        }
        essential = required | terminals
        removable = [
            event
            for event in facts
            if event.event_id not in essential and isinstance(event.payload, LinkFact)
        ]
        remove_count = max(0, len(facts) - target)
        if remove_count > len(removable):
            raise ValueError("count control cannot remove a relevant FACT record")
        removed_ids = (
            {event.event_id for event in removable[-remove_count:]} if remove_count else set()
        )
        kept = [event for event in facts if event.event_id not in removed_ids]
        forbidden = set(bundle.truth.relevant_node_path)
        padding_count = target - len(kept)
        for _ in range(padding_count):
            payload = _unused_link(kept, forbidden)
            kept.append(
                ExternalEvent(
                    max(event.event_id for event in kept) + 1,
                    0.0,
                    ExternalEventKind.FACT,
                    payload,
                )
            )
        if padding_count:
            last_timestamp = facts[-1].timestamp
            activation_timestamp = bundle.public.events[-1].timestamp
            first_padding = len(kept) - padding_count
            for offset in range(padding_count):
                timestamp = last_timestamp + (
                    (activation_timestamp - last_timestamp) * (offset + 1) / (padding_count + 1)
                )
                kept[first_padding + offset] = replace(
                    kept[first_padding + offset], timestamp=float(timestamp)
                )
        bundle = _rebuild_control_bundle(
            bundle,
            kept,
            preserve_fact_timestamps=True,
        )
    elif injector.control_id == "PC_ACTIVATION_GAP_BY_LABEL":
        bundle = _rebuild_control_bundle(
            bundle,
            facts,
            activation_time=float(facts[-1].timestamp + (1.0 if positive else 4.0)),
        )
    elif injector.control_id == "PC_TERMINAL_ORDER_BY_VARIANT":
        hazards = [event for event in facts if isinstance(event.payload, HazardFact)]
        safe = next(event for event in facts if isinstance(event.payload, SafeFact))
        terminals = (hazards[0], hazards[1], safe)
        codes = ((0, 1, 2), (0, 2, 1), (2, 0, 1))
        selected = [terminals[index] for index in codes[variant_index]]
        selected.extend(event for event in facts if event not in terminals)
        if tuple(event.payload for event in selected) == tuple(event.payload for event in facts):
            link_positions = [
                index for index, event in enumerate(selected) if isinstance(event.payload, LinkFact)
            ]
            left, right = link_positions[:2]
            selected[left], selected[right] = selected[right], selected[left]
        bundle = _rebuild_control_bundle(bundle, selected)
    elif injector.control_id == "PC_ACTIVATION_ID_BY_LABEL":
        activation = bundle.public.events[-1]
        assert isinstance(activation.payload, ActivationPayload)
        target = 0 if positive else 63
        source = activation.payload.start_node
        swaps = {source: target, target: source}
        if source == target:
            left, right = bundle.truth.relevant_node_path[1:3]
            swaps.update({left: right, right: left})

        def remap(value: int) -> int:
            return swaps.get(value, value)

        rewritten = []
        for event in facts:
            payload = event.payload
            if isinstance(payload, LinkFact):
                payload = replace(
                    payload,
                    source_node=remap(payload.source_node),
                    target_node=remap(payload.target_node),
                )
            elif isinstance(payload, (HazardFact, SafeFact)):
                payload = replace(payload, node=remap(payload.node))
            rewritten.append(replace(event, payload=payload))
        bundle = _rebuild_control_bundle(
            bundle,
            rewritten,
            relevant_node_path=tuple(remap(node) for node in bundle.truth.relevant_node_path),
        )
    elif injector.control_id == "PC_RECORD_ID_BY_VARIANT":
        band = (0, 128, 256)[variant_index]
        bundle = _rebuild_control_bundle(
            bundle,
            facts,
            event_ids=tuple(band + index for index in range(len(facts))),
        )
    elif injector.control_id == "PC_PUBLIC_ID_BY_LABEL":
        raw = bytearray(
            bytes.fromhex(
                sha256_bytes(
                    canonical_json_bytes(
                        {
                            "domain": "silent-cascade/ofd-v1/pc-public-id/v1",
                            "position": corpus_position,
                        }
                    )
                )
            )[:16]
        )
        raw[0] = 0 if positive else 255
        raw[6] = (raw[6] & 0x0F) | 0x40
        raw[8] = (raw[8] & 0x3F) | 0x80
        bundle = _rebuild_control_bundle(
            bundle,
            facts,
            public_id=str(uuid.UUID(bytes=bytes(raw))),
        )
    elif injector.control_id == "PC_DELAY_BY_LABEL":
        delay = 1.0 if positive else 1024.0
        rewritten = [
            replace(
                event,
                payload=(
                    replace(event.payload, delay=delay)
                    if isinstance(event.payload, HazardFact)
                    else event.payload
                ),
            )
            for event in facts
        ]
        bundle = _rebuild_control_bundle(bundle, rewritten, delay=delay)
    elif injector.control_id == "PC_HAZARD_LAYOUT_BY_CLASS":
        if not positive:
            return example
        if injected_hazard_target not in range(4):
            raise ValueError("hazard-layout control requires its frozen target")
        target = int(injected_hazard_target)
        hazards = [event for event in facts if isinstance(event.payload, HazardFact)]
        safe = next(event for event in facts if isinstance(event.payload, SafeFact))
        terminal_id = bundle.truth.terminal_record_id
        assert terminal_id is not None
        rewritten_hazards = [
            replace(
                event,
                payload=replace(
                    event.payload,
                    hazard_type=(target if event.event_id == terminal_id else (target + 1) % 4),
                ),
            )
            for event in hazards
        ]
        sentinel = next(
            (
                event
                for event in facts
                if isinstance(event.payload, LinkFact)
                and event.event_id not in bundle.truth.relevant_record_ids
            ),
            None,
        )
        if sentinel is None:
            last_timestamp = facts[-1].timestamp
            activation_timestamp = bundle.public.events[-1].timestamp
            sentinel = ExternalEvent(
                max(event.event_id for event in facts) + 1,
                float(last_timestamp + (activation_timestamp - last_timestamp) / 2.0),
                ExternalEventKind.FACT,
                _unused_link(facts, set(bundle.truth.relevant_node_path)),
            )
            facts.append(sentinel)
        replaced = {event.event_id: event for event in rewritten_hazards}
        facts = [replaced.get(event.event_id, event) for event in facts]
        safe = next(event for event in facts if isinstance(event.payload, SafeFact))
        rewritten_hazards = [event for event in facts if isinstance(event.payload, HazardFact)]
        sentinel = next(event for event in facts if event.event_id == sentinel.event_id)
        selected: list[ExternalEvent | None] = [None, None, None, None]
        selected[target] = safe
        remaining = [*rewritten_hazards, sentinel]
        for index in range(4):
            if selected[index] is None:
                selected[index] = remaining.pop(0)
        prefix = [event for event in selected if event is not None]
        reserved = {event.event_id for event in prefix}
        prefix.extend(event for event in facts if event.event_id not in reserved)
        bundle = _rebuild_control_bundle(
            bundle,
            prefix,
            hazard_type=target,
            fact_timestamps=tuple(sorted(event.timestamp for event in facts)),
        )
    elif injector.control_id == "PC_MANIFEST_ORDER_BY_VARIANT":
        if encoded_manifest_rank is None:
            raise ValueError("manifest-order control requires its frozen encoded rank")
        example = replace(example, manifest_rank=encoded_manifest_rank)
    return replace(example, bundle=bundle)


class CounterfactualCheckId(StrEnum):
    TERMINAL_DELAY_SWAP = "terminal_delay_swap"
    PRESENTATION_PERMUTATION = "presentation_permutation"
    PAIRED_CLOCK_SCALE = "paired_clock_scale"


class CounterfactualPairResult(StrictModel):
    pair_key_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    decision_mismatch: bool
    temporal_mismatch: bool


class CounterfactualCheckResult(StrictModel):
    check_id: CounterfactualCheckId
    checked_pairs: int = Field(gt=0)
    decision_mismatch_count: int = Field(ge=0)
    temporal_mismatch_count: int = Field(ge=0)
    result_payload_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    passed: bool

    @model_validator(mode="after")
    def require_derived_pass_state(self) -> CounterfactualCheckResult:
        if self.passed != (self.decision_mismatch_count == 0 and self.temporal_mismatch_count == 0):
            raise ValueError("counterfactual passed state must equal its mismatch counts")
        return self


class LeakageReport(StrictModel):
    schema_version: Literal["leakage-report-v1"]
    provenance: EvidenceProvenance
    generation_mode: Literal["matched", "independent"]
    profile: LeakageAuditProfileName
    corpus_hash: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    feature_schema_hash: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    split_membership_hash: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    train_membership_hash: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    test_membership_hash: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    episode_count: int
    randomization_block_count: int
    suite_path_denominators: dict[str, int]
    construction_checks: dict[str, bool]
    probes: tuple[ShortcutProbeResult, ...]
    positive_controls: tuple[PositiveControlResult, ...]
    counterfactual_checks: tuple[CounterfactualCheckResult, ...]
    label_shuffled_control_passed: bool
    passed: bool

    @model_validator(mode="after")
    def require_complete_counterfactual_family(self) -> LeakageReport:
        if tuple(item.check_id for item in self.counterfactual_checks) != tuple(
            CounterfactualCheckId
        ):
            raise ValueError(
                "counterfactual checks must contain every check exactly once in enum order"
            )
        clean_pass = self.profile is LeakageAuditProfileName.TEST or all(
            probe.passed for probe in self.probes
        )
        expected = (
            not self.positive_controls
            and clean_pass
            and self.label_shuffled_control_passed
            and all(item.passed for item in self.positive_controls)
            and all(item.passed for item in self.counterfactual_checks)
        )
        if self.passed != expected:
            raise ValueError("leakage report passed state is not derived from its evidence")
        return self


@dataclass(frozen=True, slots=True)
class _StoredExample:
    public_id: str
    digest: str
    group_id: str
    suite: SuiteName
    path_length: int
    variant: EpisodeVariant
    hazard_class: int | None
    block: int
    position: int
    # This is extracted from public FACT records during the first stream.  It
    # is metadata for the permitted within-episode null, never a predictor.
    public_hazard_classes: tuple[int, ...] = ()


def _fact_events(bundle: EpisodeBundle):
    # This helper is intentionally public-only.  Never move truth access here.
    return tuple(event for event in bundle.public.events if event.kind is ExternalEventKind.FACT)


def _one_hot(length: int, index: int | None) -> np.ndarray:
    vector = np.zeros(length, dtype=np.float32)
    if index is not None:
        if type(index) is not int or not 0 <= index < length:
            raise ValueError("public categorical value is out of range")
        vector[index] = 1.0
    return vector


def _record_parts(event: object) -> tuple[int, int, int | None, int | None, float | None]:
    payload = event.payload  # type: ignore[union-attr]
    if isinstance(payload, LinkFact):
        return 0, payload.source_node, payload.target_node, None, None
    if isinstance(payload, HazardFact):
        return 1, payload.node, None, payload.hazard_type, payload.delay
    if isinstance(payload, SafeFact):
        return 2, payload.node, None, None, None
    raise ValueError("public FACT payload is unsupported")


def _first_last_vector(event: object | None) -> np.ndarray:
    vector = np.zeros(139, dtype=np.float32)
    if event is None:
        return vector
    kind, subject, object_id, hazard, delay = _record_parts(event)
    vector[kind] = 1.0
    vector[3 + subject] = 1.0
    vector[67 + (64 if object_id is None else object_id)] = 1.0
    if hazard is not None:
        vector[132] = 1.0
        vector[133 + hazard] = 1.0
    if delay is not None:
        vector[137] = 1.0
        vector[138] = math.log1p(delay)
    return vector


def _topology_vector(facts: Sequence[object]) -> np.ndarray:
    links = [event.payload for event in facts if isinstance(event.payload, LinkFact)]
    nodes = sorted({node for link in links for node in (link.source_node, link.target_node)})
    in_degree = Counter(link.target_node for link in links)
    out_degree = Counter(link.source_node for link in links)
    adjacent: dict[int, set[int]] = {node: set() for node in nodes}
    directed: dict[int, list[int]] = {node: [] for node in nodes}
    for link in links:
        adjacent[link.source_node].add(link.target_node)
        adjacent[link.target_node].add(link.source_node)
        directed[link.source_node].append(link.target_node)

    components: list[int] = []
    remaining = set(nodes)
    while remaining:
        pending = [remaining.pop()]
        size = 0
        while pending:
            node = pending.pop()
            size += 1
            for neighbour in adjacent[node] & remaining:
                remaining.remove(neighbour)
                pending.append(neighbour)
        components.append(size)

    # Kosaraju keeps this graph summary independent of activation/rooted reachability.
    seen: set[int] = set()
    order: list[int] = []

    def visit(node: int) -> None:
        seen.add(node)
        for child in directed[node]:
            if child not in seen:
                visit(child)
        order.append(node)

    for node in nodes:
        if node not in seen:
            visit(node)
    reverse: dict[int, list[int]] = {node: [] for node in nodes}
    for node, children in directed.items():
        for child in children:
            reverse[child].append(node)
    seen.clear()
    scc_sizes: list[int] = []
    for start in reversed(order):
        if start in seen:
            continue
        pending, size = [start], 0
        seen.add(start)
        while pending:
            node = pending.pop()
            size += 1
            for parent in reverse[node]:
                if parent not in seen:
                    seen.add(parent)
                    pending.append(parent)
        scc_sizes.append(size)
    cycle = any(size > 1 for size in scc_sizes) or any(
        link.source_node == link.target_node for link in links
    )
    norm = max(len(nodes), 1)
    result = np.zeros(33, dtype=np.float32)
    result[:5] = (
        len(nodes) / 64,
        len(links) / 64,
        len(components) / 64,
        len(scc_sizes) / 64,
        cycle,
    )
    for offset, values in ((5, in_degree), (10, out_degree)):
        for node in nodes:
            result[offset + min(values[node], 4)] += 1.0 / norm
    for size in components:
        result[15 + min(size - 1, 7)] += 1.0 / max(len(components), 1)
    degrees = [
        np.asarray([values[node] for node in nodes], dtype=np.float32)
        for values in (in_degree, out_degree)
    ]
    for index, values in enumerate(degrees):
        base = 23 + 4 * index
        if len(values):
            result[base : base + 4] = (values.min(), values.max(), values.mean(), values.std())
    if nodes:
        result[31] = sum(in_degree[node] == 0 for node in nodes) / len(nodes)
        result[32] = sum(out_degree[node] == 0 for node in nodes) / len(nodes)
    return result


def _audit_group_id(example: AuditExample) -> str:
    # Coordinates are private provenance, not predictor inputs.  The exposed
    # feature vectors do not include this value.
    return f"{example.generation_mode}:{example.randomization_block_index}"


def extract_shortcut_features(example: AuditExample, corpus_size: int) -> ShortcutFeatureSet:
    """Encode the frozen 1,560-column predictor from public events alone."""
    if not isinstance(example, AuditExample) or type(corpus_size) is not int or corpus_size <= 0:
        raise TypeError("example and corpus_size must be valid exact values")
    public = example.bundle.public
    facts = _fact_events(example.bundle)
    activation = public.events[-1]
    if not isinstance(activation.payload, ActivationPayload):
        raise ValueError("public episode has no valid activation")
    try:
        identifier = np.frombuffer(uuid.UUID(public.init.episode_public_id).bytes, dtype=np.uint8)
    except ValueError as error:
        raise ValueError("public episode ID is not a UUID") from error
    id_position = np.concatenate(
        (
            identifier.astype(np.float32) / 255.0,
            np.asarray([example.manifest_rank / max(corpus_size - 1, 1)], dtype=np.float32),
        )
    )
    activation_node = np.zeros(256, dtype=np.float32)
    activation_node[:64] = _one_hot(64, activation.payload.start_node)
    for event in facts:
        _kind, subject, object_id, _hazard, _delay = _record_parts(event)
        activation_node[64 + subject] += 1.0
        if object_id is not None:
            activation_node[128 + object_id] += 1.0
        activation_node[192 + subject] += 1.0
        if object_id is not None:
            activation_node[192 + object_id] += 1.0
    first_last = np.concatenate(
        (
            _first_last_vector(facts[0] if facts else None),
            _first_last_vector(facts[-1] if facts else None),
        )
    )
    kinds = [_record_parts(event)[0] for event in facts]
    counts = (
        np.asarray([len(facts), kinds.count(0), kinds.count(1), kinds.count(2)], dtype=np.float32)
        / 64.0
    )
    order = np.zeros(768, dtype=np.float32)
    times = np.zeros(194, dtype=np.float32)
    previous = public.init.initial_time
    for slot, event in enumerate(facts[:64]):
        kind, subject, object_id, hazard, delay = _record_parts(event)
        base = slot * 12
        order[base : base + 4] = (1.0, *(_one_hot(3, kind)))
        order[base + 4 : base + 12] = (
            subject / 63.0,
            float(object_id is not None),
            0.0 if object_id is None else object_id / 63.0,
            float(hazard is not None),
            0.0 if hazard is None else hazard / 3.0,
            float(delay is not None),
            0.0 if delay is None else math.log1p(delay),
            event.event_id / 63.0,
        )
        time_base = slot * 3
        times[time_base : time_base + 3] = (
            1.0,
            math.log1p(event.timestamp - public.init.initial_time),
            math.log1p(event.timestamp - previous),
        )
        previous = event.timestamp
    times[-2:] = (
        math.log1p(activation.timestamp - public.init.initial_time),
        math.log1p(activation.timestamp - previous),
    )
    hazards = sorted(
        (event.payload for event in facts if isinstance(event.payload, HazardFact)),
        key=lambda item: (item.delay, item.hazard_type),
    )
    terminals = np.zeros(10, dtype=np.float32)
    for hazard in hazards:
        terminals[hazard.hazard_type] += 0.5
    for index, hazard in enumerate(hazards[:2]):
        terminals[4 + index] = math.log1p(hazard.delay)
        terminals[6 + index] = hazard.hazard_type / 3.0
    terminals[8:] = (kinds.count(1) / 64.0, kinds.count(2) / 64.0)
    vectors: dict[ShortcutFeatureGroup, np.ndarray] = {
        ShortcutFeatureGroup.ID_POSITION: id_position.astype(np.float32),
        ShortcutFeatureGroup.ACTIVATION_NODE: activation_node,
        ShortcutFeatureGroup.FIRST_LAST_FACT: first_last.astype(np.float32),
        ShortcutFeatureGroup.COUNTS: counts,
        ShortcutFeatureGroup.ORDER_RECORD_IDS: order,
        ShortcutFeatureGroup.TIMES: times,
        ShortcutFeatureGroup.TERMINAL_MULTISET: terminals,
        ShortcutFeatureGroup.LINK_TOPOLOGY: _topology_vector(facts),
    }
    vectors[ShortcutFeatureGroup.COMBINED] = np.concatenate(tuple(vectors.values())).astype(
        np.float32
    )
    if tuple(vector.shape[0] for vector in vectors.values()) != (
        *_FEATURE_DIMENSIONS,
        _TOTAL_FEATURE_DIMENSION,
    ):
        raise RuntimeError("frozen leakage feature dimensions changed")
    if not all(np.isfinite(vector).all() for vector in vectors.values()):
        raise ValueError("public feature vector contains nonfinite data")
    return ShortcutFeatureSet(
        "leakage-features-v1",
        public.init.episode_public_id,
        example.generation_mode,
        _audit_group_id(example),
        example.bundle.truth.key.suite,
        example.bundle.truth.recipe.requested_path_length,
        vectors,
    )


def counterfactual_pair_key(
    check_id: CounterfactualCheckId, source_public_ids: Sequence[str], transform: str
) -> str:
    if not isinstance(check_id, CounterfactualCheckId) or not isinstance(transform, str):
        raise TypeError("counterfactual key inputs are invalid")
    allowed_tags: dict[CounterfactualCheckId, tuple[str, int]] = {
        CounterfactualCheckId.TERMINAL_DELAY_SWAP: ("swap_terminal_delay", 2),
        CounterfactualCheckId.PRESENTATION_PERMUTATION: ("permute_presentation", 1),
        CounterfactualCheckId.PAIRED_CLOCK_SCALE: ("scale_0_1x", 1),
    }
    allowed_tag, expected_count = allowed_tags[check_id]
    if check_id is CounterfactualCheckId.PAIRED_CLOCK_SCALE:
        if transform not in {"scale_0_1x", "scale_10x"}:
            raise ValueError("counterfactual transform is invalid")
    elif transform != allowed_tag:
        raise ValueError("counterfactual transform is invalid")
    if len(source_public_ids) != expected_count:
        raise ValueError("counterfactual source-ID cardinality is invalid")
    if any(not isinstance(value, str) for value in source_public_ids):
        raise TypeError("counterfactual source IDs must be strings")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "silent-cascade/ofd-v1/counterfactual-pair/v1",
                "check_id": check_id.value,
                "source_public_ids": list(source_public_ids),
                "transform": transform,
            }
        )
    )


def counterfactual_result_payload_hash(
    check_id: CounterfactualCheckId, pairs: Sequence[CounterfactualPairResult]
) -> str:
    if not isinstance(check_id, CounterfactualCheckId):
        raise TypeError("check_id must be a CounterfactualCheckId")
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "silent-cascade/ofd-v1/counterfactual-check/v1",
                "check_id": check_id.value,
                "pairs": [pair.model_dump(mode="json") for pair in pairs],
            }
        )
    )


def _profile_config(config: LeakageAuditConfig, profile: LeakageAuditProfileName):
    return config.test if profile is LeakageAuditProfileName.TEST else config.phase1_gate


def _validate_holm_attainability(config: LeakageAuditConfig, profile: object) -> None:
    """Reject a confirmatory profile whose finite null cannot reach Holm alpha."""
    if getattr(profile, "enforce_clean_statistical_gate", False) and (
        27 / (profile.permutation_replicates + 1) >= config.alpha
    ):
        raise ValueError("permutation replicate count cannot attain the Holm threshold")


def _validate_provenance(
    source: ReiterableAuditSource, config: LeakageAuditConfig, provenance: EvidenceProvenance
) -> None:
    descriptor = source.descriptor
    if (
        not isinstance(descriptor, AuditSourceDescriptor)
        or source.episode_count != descriptor.episode_count
    ):
        raise ValueError("source descriptor episode count is invalid")
    if (
        descriptor.generation_mode != provenance.generation_mode
        or descriptor.allocation_id != provenance.allocation_id
        or descriptor.split_namespace is not provenance.split_namespace
        or descriptor.root_seed != provenance.root_seed
        or descriptor.public_id_seed_sha256 != provenance.public_id_seed_sha256
        or descriptor.config_sha256 != provenance.config_sha256
        or descriptor.generator_source_sha256 != provenance.generator_source.sha256
        or provenance.analysis_seeds
        != {"audit_seed": config.audit_seed, "positive_control_seed": config.positive_control_seed}
        or provenance.foundation_model_calls != 0
    ):
        raise ValueError("source descriptor/provenance mismatch")
    validation_config = getattr(source, "validation_config", None)
    if not isinstance(validation_config, Phase1Config):
        raise ValueError("audit source has no canonical validation configuration")
    if sha256_bytes(canonical_json_bytes(validation_config)) != descriptor.config_sha256:
        raise ValueError("audit source configuration digest is not authenticated")


_PHASE1_GATE_DENOMINATORS = {
    "distractor_flood:2": 8_000,
    "distractor_flood:3": 8_000,
    "distractor_flood:4": 8_000,
    "iid_primary:2": 8_000,
    "iid_primary:3": 8_000,
    "iid_primary:4": 8_000,
    "ood_depth:5": 4_000,
    "ood_depth:6": 4_000,
    "ood_depth:7": 4_000,
    "ood_depth:8": 4_000,
    "ood_long_delay:2": 4_000,
    "ood_long_delay:3": 4_000,
    "ood_long_delay:4": 4_000,
    "ood_short_delay:2": 8_000,
    "ood_short_delay:3": 8_000,
    "ood_short_delay:4": 8_000,
}


def _expected_profile_denominators(
    profile: LeakageAuditProfileName, episode_count: int
) -> dict[str, int]:
    if profile is LeakageAuditProfileName.PHASE1_GATE:
        if episode_count != 100_000:
            raise ValueError("phase1 gate requires exactly 100,000 episodes")
        return _PHASE1_GATE_DENOMINATORS
    if episode_count % 12:
        raise ValueError("test profile must contain equal complete cohorts at paths 2, 3, and 4")
    per_path = episode_count // 3
    return {f"iid_primary:{path}": per_path for path in (2, 3, 4)}


def _validate_source_authentication(
    source: ReiterableAuditSource,
    profile: LeakageAuditProfileName,
) -> AuditSourceAuthentication:
    authentication = getattr(source, "authentication", None)
    if not isinstance(authentication, AuditSourceAuthentication):
        raise ValueError("audit source has no independent authentication record")
    if (
        authentication.profile is not profile
        or authentication.descriptor_sha256 != audit_source_descriptor_sha256(source.descriptor)
    ):
        raise ValueError("audit source authentication does not bind the descriptor and profile")
    expected = _expected_profile_denominators(profile, source.episode_count)
    if authentication.suite_path_denominators != expected:
        raise ValueError("audit source profile denominators are not exact")
    clock_counts = authentication.clock_scale_pair_counts
    if (
        set(clock_counts) != {"scale_0_1x", "scale_10x"}
        or any(type(value) is not int or value <= 0 for value in clock_counts.values())
        or (
            profile is LeakageAuditProfileName.PHASE1_GATE
            and clock_counts != {"scale_0_1x": 5_000, "scale_10x": 2_000}
        )
    ):
        raise ValueError("audit source clock-pair denominators are not exact")
    return authentication


def _validate_independent_trust_anchor(
    source: ReiterableAuditSource,
    profile: LeakageAuditProfileName,
    provenance: EvidenceProvenance,
    authentication: AuditSourceAuthentication,
) -> LeakageAuditEvidenceAnchor:
    """Bind source-owned witnesses to the separately supplied evidence capability."""
    anchor = provenance.leakage_audit
    descriptor = source.descriptor
    if not isinstance(anchor, LeakageAuditEvidenceAnchor):
        raise ValueError("independent leakage trust anchor is required")
    if (
        anchor.profile != profile.value
        or anchor.allocation_id != descriptor.allocation_id
        or anchor.allocation_id != provenance.allocation_id
        or anchor.allocation_or_manifest_sha256 != descriptor.allocation_or_manifest_sha256
        or anchor.config_sha256 != descriptor.config_sha256
        or anchor.config_sha256 != provenance.config_sha256
        or anchor.descriptor_sha256 != audit_source_descriptor_sha256(descriptor)
        or anchor.descriptor_sha256 != authentication.descriptor_sha256
        or anchor.source_manifest_sha256 != authentication.source_manifest_sha256
        or anchor.suite_path_denominators != authentication.suite_path_denominators
        or anchor.clock_pair_manifest_sha256 != authentication.clock_pair_manifest_sha256
        or anchor.clock_scale_pair_counts != authentication.clock_scale_pair_counts
        or anchor.episode_count != descriptor.episode_count
        or anchor.episode_count != source.episode_count
    ):
        raise ValueError("independent leakage trust anchor does not match the source")
    if profile is LeakageAuditProfileName.PHASE1_GATE and not source.publishable:
        raise ValueError("in-memory or test leakage sources are nonpublishable")
    return anchor


def _resource_guard(config: LeakageAuditConfig, phase: str = "unspecified") -> None:
    del phase
    if psutil.Process().memory_info().rss > config.max_resident_working_bytes:
        raise MemoryError("leakage audit resident working-set ceiling exceeded")


def _allocate_feature_store(path: Path, rows: int) -> None:
    allocated = np.memmap(
        path,
        dtype=np.float32,
        mode="w+",
        shape=(rows, _TOTAL_FEATURE_DIMENSION),
    )
    allocated.flush()
    allocated._mmap.close()


def _write_feature_batch(
    path: Path,
    total_rows: int,
    start: int,
    batch: np.ndarray,
    config: LeakageAuditConfig,
) -> None:
    if (
        batch.ndim != 2
        or batch.dtype != np.float32
        or batch.shape[1] != _TOTAL_FEATURE_DIMENSION
        or not 0 <= start < start + len(batch) <= total_rows
    ):
        raise ValueError("feature-store batch coordinates are invalid")
    _resource_guard(config, "extraction")
    mapped = np.memmap(
        path,
        dtype=np.float32,
        mode="r+",
        shape=(total_rows, _TOTAL_FEATURE_DIMENSION),
    )
    try:
        mapped[start : start + len(batch)] = batch
        mapped.flush()
    finally:
        mapped._mmap.close()
    _resource_guard(config, "extraction")


def _validate_audit_coordinate(
    example: AuditExample, rank: int, descriptor: AuditSourceDescriptor
) -> None:
    if example.manifest_rank != rank:
        raise ValueError("audit manifest rank is not canonical source order")
    key = example.bundle.truth.key
    coordinate = key.coordinate
    if (
        key.split_namespace is not descriptor.split_namespace
        or key.root_seed != descriptor.root_seed
        or key.suite is not example.bundle.truth.recipe.evaluation_suite
    ):
        raise ValueError("audit episode key is not descriptor-bound")
    if example.generation_mode == "matched":
        if not isinstance(coordinate, MatchedEpisodeCoordinate) or (
            coordinate.cohort_index,
            coordinate.member_index,
        ) != (example.randomization_block_index, example.episode_position):
            raise ValueError("matched audit coordinate is not authenticated")
    elif (
        not isinstance(coordinate, IndependentEpisodeCoordinate)
        or coordinate.allocation_quartet_index != example.randomization_block_index
        or coordinate.episode_index != example.episode_position
    ):
        raise ValueError("independent audit coordinate is not authenticated")


def _validate_independent_quartets(rows: Sequence[_StoredExample]) -> None:
    by_block: dict[int, list[_StoredExample]] = defaultdict(list)
    for row in rows:
        by_block[row.block].append(row)
    expected_variants = Counter(
        (
            EpisodeVariant.POSITIVE,
            EpisodeVariant.POSITIVE,
            EpisodeVariant.SAFE_NEGATIVE,
            EpisodeVariant.DISCONNECTED_NEGATIVE,
        )
    )
    for block, quartet in by_block.items():
        positions = sorted(row.position for row in quartet)
        if (
            len(quartet) != 4
            or positions != list(range(positions[0], positions[0] + 4))
            or positions[0] % 4
            or Counter(row.variant for row in quartet) != expected_variants
            or len({(row.suite, row.path_length) for row in quartet}) != 1
            or {row.group_id for row in quartet} != {f"independent:{block}"}
        ):
            raise ValueError("independent allocation quartet is incomplete or corrupted")


def _split_memberships(
    rows: Sequence[_StoredExample],
    seed: int,
    corpus_hash: str,
    mode: str,
    *,
    strict_divisible: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    by_stratum: dict[tuple[SuiteName, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(rows):
        by_stratum[(row.suite, row.path_length)][row.group_id].append(index)
    train: list[int] = []
    test: list[int] = []
    membership: list[dict[str, object]] = []
    for (suite, path), groups in sorted(
        by_stratum.items(), key=lambda item: (item[0][0].value, item[0][1])
    ):
        if strict_divisible and len(groups) % 5:
            raise ValueError("full-gate stratum group count must be divisible by five")
        if len(groups) < 5:
            raise ValueError("shortcut stratum has fewer than five groups")
        ranked = sorted(
            groups,
            key=lambda group: sha256_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": "leakage-v1",
                        "audit_seed": seed,
                        "corpus_hash": corpus_hash,
                        "generation_mode": mode,
                        "suite": suite.value,
                        "requested_path_length": path,
                        "randomization_block_index": int(group.rsplit(":", 1)[1]),
                    }
                )
            ),
        )
        cut = len(ranked) * 4 // 5
        selected_train = set(ranked[:cut])
        for group in ranked:
            target = train if group in selected_train else test
            target.extend(groups[group])
            membership.append(
                {
                    "suite": suite.value,
                    "path": path,
                    "group": group,
                    "split": "train" if group in selected_train else "test",
                }
            )
    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
        sha256_bytes(
            canonical_json_bytes(
                {"domain": "silent-cascade/ofd-v1/leakage-split/v1", "memberships": membership}
            )
        ),
    )


def _hazard_identity_columns(group: ShortcutFeatureGroup) -> np.ndarray:
    size = (
        _TOTAL_FEATURE_DIMENSION
        if group is ShortcutFeatureGroup.COMBINED
        else _FEATURE_DIMENSIONS[list(ShortcutFeatureGroup).index(group)]
    )
    mask = np.zeros(size, dtype=bool)
    if group is ShortcutFeatureGroup.FIRST_LAST_FACT:
        mask[[133, 134, 135, 136, 272, 273, 274, 275]] = True
    elif group is ShortcutFeatureGroup.ORDER_RECORD_IDS:
        mask[np.arange(64) * 12 + 8] = True
    elif group is ShortcutFeatureGroup.TERMINAL_MULTISET:
        mask[[0, 1, 2, 3, 6, 7]] = True
    elif group is ShortcutFeatureGroup.COMBINED:
        first_last_offset = _GROUP_SLICES[ShortcutFeatureGroup.FIRST_LAST_FACT].start
        assert first_last_offset is not None
        mask[first_last_offset + 133 : first_last_offset + 137] = True
        mask[first_last_offset + 272 : first_last_offset + 276] = True
        order_offset = _GROUP_SLICES[ShortcutFeatureGroup.ORDER_RECORD_IDS].start
        assert order_offset is not None
        mask[order_offset + np.arange(64) * 12 + 8] = True
        terminal_offset = _GROUP_SLICES[ShortcutFeatureGroup.TERMINAL_MULTISET].start
        assert terminal_offset is not None
        mask[terminal_offset : terminal_offset + 4] = True
        mask[terminal_offset + 6 : terminal_offset + 8] = True
    return mask


def _continuous_columns(group: ShortcutFeatureGroup, width: int) -> np.ndarray:
    # Presence/category channels are excluded by construction.  The remaining
    # scalar summaries may be standardized from train rows only.
    continuous = np.ones(width, dtype=bool)
    if group is ShortcutFeatureGroup.ID_POSITION:
        continuous[:16] = False
    elif group is ShortcutFeatureGroup.ACTIVATION_NODE:
        continuous[:64] = False
    elif group is ShortcutFeatureGroup.FIRST_LAST_FACT:
        for offset in (0, 139):
            continuous[offset : offset + 137] = False
            continuous[offset + 137] = False
    elif group is ShortcutFeatureGroup.ORDER_RECORD_IDS:
        for slot in range(64):
            base = slot * 12
            continuous[base : base + 4] = False
            continuous[base + 5] = False
            continuous[base + 7 : base + 10] = False
    elif group is ShortcutFeatureGroup.TIMES:
        continuous[np.arange(64) * 3] = False
    elif group is ShortcutFeatureGroup.TERMINAL_MULTISET:
        continuous[:4] = False
        continuous[6:8] = False
    elif group is ShortcutFeatureGroup.LINK_TOPOLOGY:
        # Counts and degree moments are scalar quantities; the cycle bit and
        # every histogram bin are categorical/presence channels.
        continuous[4:23] = False
    elif group is ShortcutFeatureGroup.COMBINED:
        continuous = np.concatenate(
            tuple(
                _continuous_columns(item, width)
                for item, width in zip(
                    tuple(ShortcutFeatureGroup)[:-1], _FEATURE_DIMENSIONS, strict=True
                )
            )
        )
    return continuous


def _standardize(
    train: np.ndarray,
    test: np.ndarray,
    group: ShortcutFeatureGroup,
    *,
    continuous_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (
        _continuous_columns(group, train.shape[1]) if continuous_mask is None else continuous_mask
    )
    if mask.shape != (train.shape[1],):
        raise ValueError("standardization mask does not match selected feature width")
    result_train, result_test = (
        train.astype(np.float64, copy=True),
        test.astype(np.float64, copy=True),
    )
    mean = result_train[:, mask].mean(axis=0)
    std = result_train[:, mask].std(axis=0)
    zero = std == 0.0
    std[zero] = 1.0
    result_train[:, mask] = (result_train[:, mask] - mean) / std
    result_test[:, mask] = (result_test[:, mask] - mean) / std
    if np.any(zero):
        indices = np.flatnonzero(mask)[zero]
        result_train[:, indices] = 0.0
        result_test[:, indices] = 0.0
    if not (np.isfinite(result_train).all() and np.isfinite(result_test).all()):
        raise ValueError("standardization produced nonfinite data")
    return result_train, result_test


@dataclass(slots=True)
class _BatchedFeatureReader:
    """Read only bounded row batches from a dense or memory-mapped feature store."""

    values: np.ndarray
    start: int
    stop: int
    allowed_columns: np.ndarray | None
    batch_size: int
    config: LeakageAuditConfig
    max_batch_seen: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.values, np.ndarray) or self.values.ndim != 2:
            raise TypeError("feature reader requires a two-dimensional array")
        if not 0 <= self.start < self.stop <= self.values.shape[1]:
            raise ValueError("feature reader column bounds are invalid")
        if not 1 <= self.batch_size <= self.config.feature_batch_size:
            raise ValueError("feature reader batch size exceeds the configured ceiling")
        if self.allowed_columns is not None and self.allowed_columns.shape != (
            self.stop - self.start,
        ):
            raise ValueError("feature reader column mask has the wrong width")
        # Bound the simultaneous raw/float64/centering arrays independently
        # of the row-count ceiling. Narrow feature groups may still use the
        # full configured row batch.
        self.batch_size = min(
            self.batch_size,
            max(1, 16_000_000 // max(self.width * 16, 1)),
        )

    @property
    def width(self) -> int:
        if self.allowed_columns is None:
            return self.stop - self.start
        return int(np.sum(self.allowed_columns))

    def batches(self, indices: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if indices.ndim != 1:
            raise ValueError("feature reader indices must be one-dimensional")
        for offset in range(0, len(indices), self.batch_size):
            _resource_guard(self.config, "batch_read")
            batch_indices = indices[offset : offset + self.batch_size]
            self.max_batch_seen = max(self.max_batch_seen, len(batch_indices))
            temporary: np.memmap | None = None
            source = self.values
            if isinstance(self.values, np.memmap):
                temporary = np.memmap(
                    self.values.filename,
                    dtype=self.values.dtype,
                    mode="r",
                    offset=self.values.offset,
                    shape=self.values.shape,
                )
                source = temporary
            batch = source[batch_indices, self.start : self.stop]
            if self.allowed_columns is not None:
                batch = batch[:, self.allowed_columns]
            copied = np.array(batch, copy=True)
            if temporary is not None:
                temporary._mmap.close()
                del temporary
            yield batch_indices, copied


def _batched_moments(
    reader: _BatchedFeatureReader,
    train: np.ndarray,
    continuous_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute train-only population moments with stable bounded batch merging."""
    if continuous_mask.shape != (reader.width,):
        raise ValueError("batched standardization mask has the wrong width")
    columns = np.flatnonzero(continuous_mask)
    mean = np.zeros(len(columns), dtype=np.float64)
    second = np.zeros(len(columns), dtype=np.float64)
    count = 0
    for _indices, raw in reader.batches(train):
        values = raw[:, columns].astype(np.float64, copy=False)
        batch_count = len(values)
        if batch_count == 0:
            continue
        batch_mean = values.mean(axis=0)
        batch_second = np.square(values - batch_mean).sum(axis=0)
        delta = batch_mean - mean
        total = count + batch_count
        second += batch_second + np.square(delta) * count * batch_count / total
        mean += delta * batch_count / total
        count = total
        _resource_guard(reader.config, "moments")
    if count != len(train) or count == 0:
        raise ValueError("batched moments did not consume the complete training set")
    std = np.sqrt(second / count)
    zero = std == 0.0
    std[zero] = 1.0
    if not (np.isfinite(mean).all() and np.isfinite(std).all()):
        raise ValueError("batched standardization produced nonfinite moments")
    return mean, std, zero


def _standardize_batch(
    raw: np.ndarray,
    continuous_mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    zero: np.ndarray,
) -> np.ndarray:
    values = raw.astype(np.float64, copy=True)
    if np.any(continuous_mask):
        values[:, continuous_mask] = (values[:, continuous_mask] - mean) / std
        if np.any(zero):
            values[:, np.flatnonzero(continuous_mask)[zero]] = 0.0
    if not np.isfinite(values).all():
        raise ValueError("batched standardization produced nonfinite data")
    return values


def _fit_predict_batched(
    reader: _BatchedFeatureReader,
    train: np.ndarray,
    labels: np.ndarray,
    test: np.ndarray,
    classes: np.ndarray,
    continuous_mask: np.ndarray,
    config: LeakageAuditConfig,
) -> tuple[np.ndarray, int]:
    """Fit and predict without materializing a full selected or float64 matrix."""
    if labels.ndim != 1 or labels.shape[0] != reader.values.shape[0]:
        raise ValueError("batched optimizer labels must be feature-store aligned")
    targets = np.searchsorted(classes, labels[train])
    n_features, n_classes = reader.width, len(classes)
    counts = np.bincount(targets, minlength=n_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("optimizer received a missing class")
    class_weights = 1.0 / (n_classes * counts)
    mean, std, zero = _batched_moments(reader, train, continuous_mask)
    eye = np.eye(n_classes)

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = flat[: n_features * n_classes].reshape(n_features, n_classes)
        intercept = flat[n_features * n_classes :]
        loss = 0.0
        coefficient_gradient = np.zeros_like(coefficients)
        intercept_gradient = np.zeros_like(intercept)
        for batch_indices, raw in reader.batches(train):
            values = _standardize_batch(raw, continuous_mask, mean, std, zero)
            batch_targets = np.searchsorted(classes, labels[batch_indices])
            weights = class_weights[batch_targets]
            logits = values @ coefficients + intercept
            log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
            residual = (np.exp(log_probs) - eye[batch_targets]) * weights[:, None]
            loss -= float(np.sum(weights * log_probs[np.arange(len(batch_indices)), batch_targets]))
            coefficient_gradient += values.T @ residual
            intercept_gradient += residual.sum(axis=0)
            _resource_guard(config, "optimization")
        loss += 0.5 * config.l2_penalty * float(np.sum(coefficients * coefficients))
        coefficient_gradient += config.l2_penalty * coefficients
        return loss, np.concatenate((coefficient_gradient.ravel(), intercept_gradient))

    result = minimize(
        objective,
        np.zeros(n_features * n_classes + n_classes),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": config.optimizer_max_iterations,
            "gtol": config.optimizer_gradient_tolerance,
            "ftol": config.optimizer_function_tolerance,
        },
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("leakage logistic optimizer failed")
    coefficients = result.x[: n_features * n_classes].reshape(n_features, n_classes)
    intercept = result.x[n_features * n_classes :]
    predictions = np.empty(len(test), dtype=classes.dtype)
    offset = 0
    for _batch_indices, raw in reader.batches(test):
        values = _standardize_batch(raw, continuous_mask, mean, std, zero)
        count = len(values)
        predictions[offset : offset + count] = classes[
            np.argmax(values @ coefficients + intercept, axis=1)
        ]
        offset += count
        _resource_guard(config, "optimization")
    if offset != len(test):
        raise ValueError("batched prediction did not consume the complete test set")
    return predictions, int(result.nit)


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray, classes: np.ndarray) -> float:
    recalls = [np.mean(predictions[labels == value] == value) for value in classes]
    if any(not np.isfinite(value) for value in recalls):
        raise ValueError("balanced accuracy has a missing class")
    return float(np.mean(recalls))


def _permutation_exceeds_observed(
    labels: np.ndarray,
    predictions: np.ndarray,
    classes: np.ndarray,
    observed: float,
) -> bool:
    # A conditional hazard-label permutation can remove a class in a bounded
    # sample. Count that replicate as an exceedance instead of assigning an
    # anti-conservative statistic to an undefined class recall.
    if any(not np.any(labels == value) for value in classes):
        return True
    return _balanced_accuracy(labels, predictions, classes) >= observed


def _permuted_labels(
    rows: Sequence[_StoredExample],
    labels: np.ndarray,
    task: ShortcutTask,
    replicate: int,
    corpus_hash: str,
    audit_seed: int,
) -> np.ndarray:
    result = labels.copy()
    if task is ShortcutTask.POSITIVE_HAZARD_CLASS:
        if labels.shape != (len(rows),):
            raise ValueError("hazard permutation labels must be source-aligned")
        for index, row in enumerate(rows):
            if row.variant is not EpisodeVariant.POSITIVE:
                if labels[index] != -1:
                    raise ValueError("non-positive hazard label is invalid")
                continue
            classes = tuple(sorted(row.public_hazard_classes))
            if len(classes) != 2 or any(type(value) is not int for value in classes):
                raise ValueError("positive public hazard multiset is invalid")
            if classes[0] == classes[1]:
                continue
            if labels[index] not in classes:
                raise ValueError("hazard target is not public")
            key = AuditSeedKey(
                "leakage-v1",
                audit_seed,
                corpus_hash,
                task,
                replicate,
                row.suite,
                row.path_length,
                row.block,
                row.position,
            )
            digest = sha256_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": key.schema_version,
                        "audit_seed": key.audit_seed,
                        "corpus_hash": key.corpus_hash,
                        "task": key.task.value,
                        "replicate_index": key.replicate_index,
                        "suite": key.suite.value,
                        "requested_path_length": key.requested_path_length,
                        "randomization_block_index": key.randomization_block_index,
                        "episode_position": key.episode_position,
                    }
                )
            )
            if int(digest[:16], 16) & 1:
                result[index] = classes[1] if labels[index] == classes[0] else classes[0]
        return result
    groups: dict[tuple[SuiteName, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row.suite, row.path_length, row.block)].append(index)
    for (suite, path, block), indices in groups.items():
        group_labels = tuple(int(labels[index]) for index in indices)
        candidates = tuple(sorted(set(permutations(group_labels))))
        expected = 6 if task is ShortcutTask.POSITIVE_BINARY else 12
        if len(candidates) != expected:
            raise ValueError("audit group does not have its declared label multiset")
        key = AuditSeedKey(
            "leakage-v1",
            audit_seed,
            corpus_hash,
            task,
            replicate,
            suite,
            path,
            block,
            min(rows[index].position for index in indices),
        )
        digest = sha256_bytes(
            canonical_json_bytes(
                {
                    "schema_version": key.schema_version,
                    "audit_seed": key.audit_seed,
                    "corpus_hash": key.corpus_hash,
                    "task": key.task.value,
                    "replicate_index": key.replicate_index,
                    "suite": key.suite.value,
                    "requested_path_length": key.requested_path_length,
                    "randomization_block_index": key.randomization_block_index,
                    "episode_position": key.episode_position,
                }
            )
        )
        selected = candidates[int(digest[:16], 16) % len(candidates)]
        for index, value in zip(indices, selected, strict=True):
            result[index] = value
    return result


def _holm(probes: list[ShortcutProbeResult], alpha: float) -> list[ShortcutProbeResult]:
    ordered = sorted(enumerate(probes), key=lambda item: item[1].raw_permutation_p)
    adjusted = [0.0] * len(probes)
    running = 0.0
    for rank, (index, probe) in enumerate(ordered):
        running = max(running, min(1.0, (len(probes) - rank) * probe.raw_permutation_p))
        adjusted[index] = running
    return [
        probe.model_copy(
            update={
                "holm_adjusted_p": adjusted[index],
                "passed": not (
                    probe.balanced_accuracy > probe.balanced_chance and adjusted[index] < alpha
                ),
            }
        )
        for index, probe in enumerate(probes)
    ]


def _feature_bounds(group: ShortcutFeatureGroup) -> tuple[int, int]:
    if group is ShortcutFeatureGroup.COMBINED:
        return 0, _TOTAL_FEATURE_DIMENSION
    selected = _GROUP_SLICES[group]
    assert selected.start is not None and selected.stop is not None
    return selected.start, selected.stop


def _run_probes(
    values: np.ndarray,
    rows: Sequence[_StoredExample],
    train: np.ndarray,
    test: np.ndarray,
    config: LeakageAuditConfig,
    profile,
    corpus_hash: str,
    *,
    label_shuffled: bool = False,
) -> list[ShortcutProbeResult]:
    probes: list[ShortcutProbeResult] = []
    labels_by_task = {
        ShortcutTask.POSITIVE_BINARY: np.asarray(
            [row.variant is EpisodeVariant.POSITIVE for row in rows], dtype=np.int8
        ),
        ShortcutTask.VARIANT_THREE_WAY: np.asarray(
            [list(EpisodeVariant).index(row.variant) for row in rows], dtype=np.int8
        ),
    }
    hazard_indices = np.asarray(
        [index for index, row in enumerate(rows) if row.variant is EpisodeVariant.POSITIVE],
        dtype=np.int64,
    )
    labels_by_task[ShortcutTask.POSITIVE_HAZARD_CLASS] = np.asarray(
        [row.hazard_class if row.variant is EpisodeVariant.POSITIVE else -1 for row in rows],
        dtype=np.int8,
    )
    for task, source_labels in labels_by_task.items():
        if label_shuffled:
            source_labels = _permuted_labels(
                rows, source_labels, task, -1, corpus_hash, config.audit_seed
            )
        indices = (
            hazard_indices if task is ShortcutTask.POSITIVE_HAZARD_CLASS else np.arange(len(rows))
        )
        task_train = np.intersect1d(train, indices, assume_unique=True)
        task_test = np.intersect1d(test, indices, assume_unique=True)
        classes = np.unique(source_labels[indices])
        if len(classes) < (
            4
            if task is ShortcutTask.POSITIVE_HAZARD_CLASS
            else (3 if task is ShortcutTask.VARIANT_THREE_WAY else 2)
        ):
            raise ValueError("shortcut task is missing a required class")
        if (
            min(np.sum(source_labels[task_test] == item) for item in classes)
            < profile.minimum_test_examples_per_class
        ):
            raise ValueError("shortcut task has insufficient held-out examples")
        for group in ShortcutFeatureGroup:
            start, stop = _feature_bounds(group)
            allowed: np.ndarray | None = None
            continuous_mask = _continuous_columns(group, stop - start)
            if task is ShortcutTask.POSITIVE_HAZARD_CLASS:
                allowed = ~_hazard_identity_columns(group)
                continuous_mask = continuous_mask[allowed]
            reader = _BatchedFeatureReader(
                values,
                start,
                stop,
                allowed,
                config.feature_batch_size,
                config,
            )
            predictions, iterations = _fit_predict_batched(
                reader,
                task_train,
                source_labels,
                task_test,
                classes,
                continuous_mask,
                config,
            )
            observed = _balanced_accuracy(source_labels[task_test], predictions, classes)
            exceed = 0
            for replicate_start in range(
                0, profile.permutation_replicates, config.permutation_batch_size
            ):
                _resource_guard(config, "permutation")
                for replicate in range(
                    replicate_start,
                    min(
                        replicate_start + config.permutation_batch_size,
                        profile.permutation_replicates,
                    ),
                ):
                    permuted_all = _permuted_labels(
                        rows, labels_by_task[task], task, replicate, corpus_hash, config.audit_seed
                    )
                    permuted = permuted_all[task_test]
                    exceed += _permutation_exceeds_observed(
                        permuted,
                        predictions,
                        classes,
                        observed,
                    )
            raw_p = (1 + exceed) / (profile.permutation_replicates + 1)
            probes.append(
                ShortcutProbeResult(
                    task=task,
                    feature_group=group,
                    feature_dimension=reader.width,
                    train_examples=len(task_train),
                    test_examples=len(task_test),
                    train_class_counts={
                        str(value): int(np.sum(source_labels[task_train] == value))
                        for value in classes
                    },
                    test_class_counts={
                        str(value): int(np.sum(source_labels[task_test] == value))
                        for value in classes
                    },
                    raw_accuracy=float(np.mean(predictions == source_labels[task_test])),
                    balanced_accuracy=observed,
                    balanced_chance=1.0 / len(classes),
                    raw_permutation_p=raw_p,
                    holm_adjusted_p=1.0,
                    optimizer_iterations=iterations,
                    optimizer_converged=True,
                    passed=True,
                )
            )
    return _holm(probes, config.alpha)


class _CounterfactualResultBuilder:
    """Accumulate one canonical counterfactual result without retaining its pairs."""

    def __init__(self, check_id: CounterfactualCheckId) -> None:
        if not isinstance(check_id, CounterfactualCheckId):
            raise TypeError("check_id must be a CounterfactualCheckId")
        self.check_id = check_id
        self._digest = hashlib.sha256()
        self._digest.update(
            b'{"check_id":'
            + b'"'
            + check_id.value.encode("ascii")
            + b'"'
            + b',"domain":"silent-cascade/ofd-v1/counterfactual-check/v1","pairs":['
        )
        self.checked_pairs = 0
        self.decision_mismatches = 0
        self.temporal_mismatches = 0
        self._finalized = False

    def add(self, pair: CounterfactualPairResult) -> None:
        if self._finalized:
            raise RuntimeError("counterfactual result has already been finalized")
        if not isinstance(pair, CounterfactualPairResult):
            raise TypeError("counterfactual builder requires typed pair results")
        if self.checked_pairs:
            self._digest.update(b",")
        self._digest.update(canonical_json_bytes(pair.model_dump(mode="json")))
        self.checked_pairs += 1
        self.decision_mismatches += pair.decision_mismatch
        self.temporal_mismatches += pair.temporal_mismatch

    def finalize(self) -> CounterfactualCheckResult:
        if self._finalized:
            raise RuntimeError("counterfactual result has already been finalized")
        if not self.checked_pairs:
            raise ValueError(f"counterfactual source has no {self.check_id.value} pairs")
        self._finalized = True
        self._digest.update(b"]}")
        return CounterfactualCheckResult(
            check_id=self.check_id,
            checked_pairs=self.checked_pairs,
            decision_mismatch_count=self.decision_mismatches,
            temporal_mismatch_count=self.temporal_mismatches,
            result_payload_sha256=self._digest.hexdigest(),
            passed=self.decision_mismatches == 0 and self.temporal_mismatches == 0,
        )


def _counterfactual_result(
    check_id: CounterfactualCheckId, pairs: Sequence[CounterfactualPairResult]
) -> CounterfactualCheckResult:
    builder = _CounterfactualResultBuilder(check_id)
    for pair in pairs:
        builder.add(pair)
    return builder.finalize()


def _derived_public_id(domain: str, source_id: str) -> str:
    payload = bytearray(
        bytes.fromhex(
            sha256_bytes(canonical_json_bytes({"domain": domain, "source_id": source_id}))
        )[:16]
    )
    payload[6] = (payload[6] & 0x0F) | 0x40
    payload[8] = (payload[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(payload)))


def _reordered_public(bundle: EpisodeBundle, seed: str) -> PublicEpisode:
    facts = list(_fact_events(bundle))
    if len(facts) < 2:
        raise ValueError("presentation counterfactual requires two FACT records")
    order = sorted(
        range(len(facts)),
        key=lambda index: sha256_bytes(
            canonical_json_bytes(
                {
                    "domain": "silent-cascade/ofd-v1/presentation-counterfactual/v1",
                    "seed": seed,
                    "record_id": facts[index].event_id,
                }
            )
        ),
    )
    if order == list(range(len(facts))):
        order = [*order[1:], order[0]]
    sorted_times = sorted(event.timestamp for event in facts)
    reordered = tuple(
        ExternalEvent(
            facts[original].event_id,
            float(sorted_times[position]),
            ExternalEventKind.FACT,
            facts[original].payload,
        )
        for position, original in enumerate(order)
    )
    return PublicEpisode(bundle.public.init, (*reordered, bundle.public.events[-1]))


def _with_terminal_delay(bundle: EpisodeBundle, delay: float) -> PublicEpisode:
    solution = solve_public_episode(bundle.public)
    hazards = tuple(
        event for event in _fact_events(bundle) if isinstance(event.payload, HazardFact)
    )
    if (
        solution.terminal_record_id is None
        or solution.public_delay is None
        or len(hazards) != 2
        or len({event.payload.delay for event in hazards}) != 1
    ):
        raise ValueError("terminal-delay counterfactual requires a public hazard")
    rewritten: list[ExternalEvent] = []
    for event in bundle.public.events:
        payload = event.payload
        if isinstance(payload, HazardFact):
            payload = replace(payload, delay=delay)
        rewritten.append(ExternalEvent(event.event_id, event.timestamp, event.kind, payload))
    return PublicEpisode(bundle.public.init, tuple(rewritten))


def _decision_signature(public: PublicEpisode) -> tuple[object, ...]:
    solved = solve_public_episode(public)
    return (
        solved.terminal_kind,
        solved.node_path,
        solved.link_record_ids,
        solved.terminal_record_id,
        solved.hazard_type,
        solved.superseded_terminal_record_ids,
    )


def _normalized_window_signature(bundle: EpisodeBundle) -> tuple[float, float, float] | None:
    solved = solve_public_episode(bundle.public)
    if solved.public_delay is None:
        return None
    activation = bundle.public.events[-1].timestamp
    window = action_window(activation, solved.public_delay, bundle.truth.recipe.oracle_timing)
    return (
        (window.start - activation) / solved.public_delay,
        (window.target - activation) / solved.public_delay,
        (window.end - activation) / solved.public_delay,
    )


def _normalized_windows_equal(left: EpisodeBundle, right: EpisodeBundle) -> bool:
    left_values = _normalized_window_signature(left)
    right_values = _normalized_window_signature(right)
    if left_values is None or right_values is None:
        return left_values is right_values
    return all(
        math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=1.0e-12)
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )


def _swapped_action_window_matches(
    bundle: EpisodeBundle,
    public: PublicEpisode,
    expected_delay: float,
) -> bool:
    solved = solve_public_episode(public)
    if solved.public_delay is None or not math.isclose(
        solved.public_delay,
        expected_delay,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        return False
    activation = public.events[-1].timestamp
    timing = bundle.truth.recipe.oracle_timing
    observed = action_window(activation, solved.public_delay, timing)
    expected = (
        activation + timing.action_window_start_fraction * expected_delay,
        activation + timing.action_target_fraction * expected_delay,
        activation + timing.action_window_end_fraction * expected_delay,
    )
    return all(
        math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1.0e-12)
        for actual, wanted in zip(
            (observed.start, observed.target, observed.end),
            expected,
            strict=True,
        )
    )


def _counterfactual_checks(
    source: ReiterableAuditSource,
    profile: LeakageAuditProfileName,
    config: LeakageAuditConfig | None = None,
) -> tuple[CounterfactualCheckResult, ...]:
    delay_builder = _CounterfactualResultBuilder(CounterfactualCheckId.TERMINAL_DELAY_SWAP)
    presentation_builder = _CounterfactualResultBuilder(
        CounterfactualCheckId.PRESENTATION_PERMUTATION
    )
    clock_builder = _CounterfactualResultBuilder(CounterfactualCheckId.PAIRED_CLOCK_SCALE)
    authentication = source.authentication
    clock_manifest = _ClockPairManifestHashBuilder()
    counts = Counter()
    parent_identities: dict[int, tuple[str, str]] = {}
    prior_order: tuple[int, int] | None = None
    for pair_index, pair in enumerate(source.iter_clock_pairs()):
        if config is not None and pair_index % config.feature_batch_size == 0:
            _resource_guard(config, "counterfactual")
        clock_manifest.add(pair)
        parent = pair.parent
        child = pair.child
        actual_parent = (
            parent.bundle.public.init.episode_public_id,
            episode_sha256(parent.bundle),
        )
        prior_identity = parent_identities.setdefault(parent.manifest_rank, actual_parent)
        if prior_identity != actual_parent:
            raise ValueError("paired clock manifest disagrees about a parent identity")
        suite = child.truth.recipe.evaluation_suite
        tag = "scale_0_1x" if suite is SuiteName.CLOCK_SCALE_0_1X else "scale_10x"
        scale_order = 0 if tag == "scale_0_1x" else 1
        order = (scale_order, parent.manifest_rank)
        if prior_order is not None and order <= prior_order:
            raise ValueError("paired clock source is not in canonical scale/parent order")
        prior_order = order
        counts[tag] += 1
        expected_child = scale_episode_time(
            parent.bundle,
            suite,
            child.public.init.episode_public_id,
        )
        if child != expected_child:
            raise ValueError("paired clock child is not the exact declared transform")
        clock_builder.add(
            CounterfactualPairResult(
                pair_key_sha256=counterfactual_pair_key(
                    CounterfactualCheckId.PAIRED_CLOCK_SCALE,
                    (parent.bundle.public.init.episode_public_id,),
                    tag,
                ),
                decision_mismatch=_decision_signature(child.public)
                != _decision_signature(parent.bundle.public),
                temporal_mismatch=not _normalized_windows_equal(child, parent.bundle),
            )
        )
    if clock_manifest.finalize() != authentication.clock_pair_manifest_sha256:
        raise ValueError("paired clock manifest does not match source authentication")
    if dict(counts) != authentication.clock_scale_pair_counts:
        raise ValueError("paired clock counts do not match source authentication")

    pending_positive: dict[tuple[SuiteName, int], AuditExample] = {}
    matched_parent_ranks: set[int] = set()
    for source_index, example in enumerate(source.iter_examples()):
        if config is not None and source_index % config.feature_batch_size == 0:
            _resource_guard(config, "counterfactual")
        bundle = example.bundle
        public_id = bundle.public.init.episode_public_id
        expected_parent = parent_identities.get(example.manifest_rank)
        if expected_parent is not None:
            if expected_parent != (public_id, episode_sha256(bundle)):
                raise ValueError("paired clock parent is not a member of the authenticated source")
            matched_parent_ranks.add(example.manifest_rank)
        original = solve_public_episode(bundle.public)
        permuted = _reordered_public(bundle, public_id)
        permuted_bundle = replace(bundle, public=permuted)
        presentation_builder.add(
            CounterfactualPairResult(
                pair_key_sha256=counterfactual_pair_key(
                    CounterfactualCheckId.PRESENTATION_PERMUTATION,
                    (public_id,),
                    "permute_presentation",
                ),
                decision_mismatch=_decision_signature(permuted)
                != _decision_signature(bundle.public),
                temporal_mismatch=not _normalized_windows_equal(permuted_bundle, bundle),
            )
        )
        if original.public_delay is not None:
            key = (bundle.truth.key.suite, bundle.truth.recipe.requested_path_length)
            partner = pending_positive.pop(key, None)
            if partner is None:
                pending_positive[key] = example
            else:
                partner_solution = solve_public_episode(partner.bundle.public)
                left_public = _with_terminal_delay(bundle, partner_solution.public_delay)
                right_public = _with_terminal_delay(partner.bundle, original.public_delay)
                delay_builder.add(
                    CounterfactualPairResult(
                        pair_key_sha256=counterfactual_pair_key(
                            CounterfactualCheckId.TERMINAL_DELAY_SWAP,
                            (
                                partner.bundle.public.init.episode_public_id,
                                public_id,
                            ),
                            "swap_terminal_delay",
                        ),
                        decision_mismatch=(
                            _decision_signature(left_public) != _decision_signature(bundle.public)
                            or _decision_signature(right_public)
                            != _decision_signature(partner.bundle.public)
                        ),
                        temporal_mismatch=(
                            not _swapped_action_window_matches(
                                bundle,
                                left_public,
                                partner_solution.public_delay,
                            )
                            or not _swapped_action_window_matches(
                                partner.bundle,
                                right_public,
                                original.public_delay,
                            )
                        ),
                    )
                )
    if pending_positive:
        raise ValueError("counterfactual source leaves an unpaired positive episode")
    if matched_parent_ranks != set(parent_identities):
        raise ValueError("paired clock parent is not a member of the authenticated source")
    if profile is LeakageAuditProfileName.PHASE1_GATE and (
        delay_builder.checked_pairs != 25_000
        or presentation_builder.checked_pairs != 100_000
        or dict(counts) != {"scale_0_1x": 5_000, "scale_10x": 2_000}
    ):
        raise ValueError("phase1 counterfactual denominators are not exact")
    return (
        delay_builder.finalize(),
        presentation_builder.finalize(),
        clock_builder.finalize(),
    )


def _corpus_hash_for_rows(rows: Sequence[_StoredExample], indices: Sequence[int]) -> str:
    builder = CorpusHashBuilder(len(indices))
    for index in indices:
        row = rows[index]
        builder.add(CorpusDigestEntry(row.public_id, row.digest))
    return builder.finalize()


def _positive_control_subset(
    rows: Sequence[_StoredExample],
    profile: object,
    audit_seed: int,
    corpus_hash: str,
) -> np.ndarray:
    desired = profile.positive_control_episode_count
    by_stratum: dict[tuple[SuiteName, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(rows):
        by_stratum[(row.suite, row.path_length)][row.group_id].append(index)
    groups_needed, remainder = divmod(desired, 4 * len(by_stratum))
    if remainder or groups_needed <= 0:
        raise ValueError("positive-control count cannot be balanced across source strata")
    selected_groups: set[str] = set()
    for (suite, path), groups in sorted(
        by_stratum.items(), key=lambda item: (item[0][0].value, item[0][1])
    ):
        ranked = sorted(
            groups,
            key=lambda group: sha256_bytes(
                canonical_json_bytes(
                    {
                        "domain": "silent-cascade/ofd-v1/pc-subset/v1",
                        "audit_seed": audit_seed,
                        "corpus_hash": corpus_hash,
                        "suite": suite.value,
                        "requested_path_length": path,
                        "group_id": group,
                    }
                )
            ),
        )
        if len(ranked) < groups_needed:
            raise ValueError("positive-control stratum lacks its frozen subset quota")
        selected_groups.update(ranked[:groups_needed])
    selected = np.asarray(
        [index for index, row in enumerate(rows) if row.group_id in selected_groups],
        dtype=np.int64,
    )
    if len(selected) != desired:
        raise ValueError("positive-control subset has the wrong frozen size")
    return selected


def _positive_control_split(
    rows: Sequence[_StoredExample],
    positive_control_seed: int,
    clean_subset_hash: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    by_stratum: dict[tuple[SuiteName, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(rows):
        by_stratum[(row.suite, row.path_length)][row.group_id].append(index)
    train: list[int] = []
    test: list[int] = []
    memberships: list[dict[str, object]] = []
    for (suite, path), groups in sorted(
        by_stratum.items(), key=lambda item: (item[0][0].value, item[0][1])
    ):
        if len(groups) % 5:
            raise ValueError("positive-control stratum must have an exact 80/20 group split")
        ranked = sorted(
            groups,
            key=lambda group: sha256_bytes(
                canonical_json_bytes(
                    {
                        "domain": "silent-cascade/ofd-v1/pc-split/v1",
                        "positive_control_seed": positive_control_seed,
                        "clean_subset_corpus_hash": clean_subset_hash,
                        "suite": suite.value,
                        "requested_path_length": path,
                        "group_id": group,
                    }
                )
            ),
        )
        cut = len(ranked) * 4 // 5
        for rank, group in enumerate(ranked):
            target = train if rank < cut else test
            target.extend(groups[group])
            memberships.append(
                {
                    "suite": suite.value,
                    "path": path,
                    "group": group,
                    "split": "train" if rank < cut else "test",
                }
            )
    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
        sha256_bytes(
            canonical_json_bytes(
                {
                    "domain": "silent-cascade/ofd-v1/pc-split-membership/v1",
                    "memberships": memberships,
                }
            )
        ),
    )


def _task_labels(rows: Sequence[_StoredExample], task: ShortcutTask) -> np.ndarray:
    if task is ShortcutTask.POSITIVE_BINARY:
        return np.asarray([row.variant is EpisodeVariant.POSITIVE for row in rows], dtype=np.int8)
    if task is ShortcutTask.VARIANT_THREE_WAY:
        return np.asarray([list(EpisodeVariant).index(row.variant) for row in rows], dtype=np.int8)
    return np.asarray(
        [row.hazard_class if row.variant is EpisodeVariant.POSITIVE else -1 for row in rows],
        dtype=np.int8,
    )


def _run_positive_control_probe(
    values: np.ndarray,
    rows: Sequence[_StoredExample],
    train: np.ndarray,
    test: np.ndarray,
    injector: NamedLeakInjector,
    group: ShortcutFeatureGroup,
    config: LeakageAuditConfig,
    profile: object,
    injected_hash: str,
) -> ShortcutProbeResult:
    labels = _task_labels(rows, injector.target_task)
    eligible = np.asarray(
        [
            index
            for index, row in enumerate(rows)
            if injector.target_task is not ShortcutTask.POSITIVE_HAZARD_CLASS
            or row.variant is EpisodeVariant.POSITIVE
        ],
        dtype=np.int64,
    )
    eligible_set = set(eligible)
    task_train = np.asarray([index for index in train if index in eligible_set], dtype=np.int64)
    task_test = np.asarray([index for index in test if index in eligible_set], dtype=np.int64)
    classes = np.unique(labels[eligible])
    expected_classes = (
        4
        if injector.target_task is ShortcutTask.POSITIVE_HAZARD_CLASS
        else (3 if injector.target_task is ShortcutTask.VARIANT_THREE_WAY else 2)
    )
    if len(classes) != expected_classes:
        raise ValueError("positive-control target is missing a class")
    start, stop = _feature_bounds(group)
    allowed: np.ndarray | None = None
    continuous = _continuous_columns(group, stop - start)
    if injector.target_task is ShortcutTask.POSITIVE_HAZARD_CLASS:
        allowed = ~_hazard_identity_columns(group)
        continuous = continuous[allowed]
    reader = _BatchedFeatureReader(
        values,
        start,
        stop,
        allowed,
        config.feature_batch_size,
        config,
    )
    predictions, iterations = _fit_predict_batched(
        reader,
        task_train,
        labels,
        task_test,
        classes,
        continuous,
        config,
    )
    observed = _balanced_accuracy(labels[task_test], predictions, classes)
    exceed = 0
    for replicate in range(profile.positive_control_permutation_replicates):
        if replicate % config.permutation_batch_size == 0:
            _resource_guard(config, "control")
        permuted = _permuted_labels(
            rows,
            labels,
            injector.target_task,
            replicate,
            injected_hash,
            config.positive_control_seed,
        )[task_test]
        exceed += _permutation_exceeds_observed(
            permuted,
            predictions,
            classes,
            observed,
        )
    raw_p = (1 + exceed) / (profile.positive_control_permutation_replicates + 1)
    return ShortcutProbeResult(
        task=injector.target_task,
        feature_group=group,
        feature_dimension=reader.width,
        train_examples=len(task_train),
        test_examples=len(task_test),
        train_class_counts={
            str(value): int(np.sum(labels[task_train] == value)) for value in classes
        },
        test_class_counts={
            str(value): int(np.sum(labels[task_test] == value)) for value in classes
        },
        raw_accuracy=float(np.mean(predictions == labels[task_test])),
        balanced_accuracy=observed,
        balanced_chance=1.0 / len(classes),
        raw_permutation_p=raw_p,
        holm_adjusted_p=1.0,
        optimizer_iterations=iterations,
        optimizer_converged=True,
        passed=False,
    )


_EXPECTED_CONTROL_FAILURE_ID: dict[str, str | None] = {
    "PC_COUNT_BY_LABEL": "recipe_distractor_count",
    "PC_ACTIVATION_GAP_BY_LABEL": "timing_provenance",
    "PC_TERMINAL_ORDER_BY_VARIANT": "presentation_provenance",
    "PC_ACTIVATION_ID_BY_LABEL": "node_permutation_provenance",
    "PC_RECORD_ID_BY_VARIANT": "record_identity",
    "PC_PUBLIC_ID_BY_LABEL": None,
    "PC_DELAY_BY_LABEL": "presentation_provenance",
    "PC_HAZARD_LAYOUT_BY_CLASS": "presentation_provenance",
    "PC_MANIFEST_ORDER_BY_VARIANT": None,
}


def _control_mutation_error(control_id: str) -> NoReturn:
    raise ValueError(f"positive-control public mutation was not exactly declared: {control_id}")


def _expected_control_unused_link(
    facts: Sequence[ExternalEvent], forbidden_nodes: set[int]
) -> LinkFact:
    """Independently derive the first unique off-path DAG-safe padding edge."""
    edges = {
        (event.payload.source_node, event.payload.target_node)
        for event in facts
        if isinstance(event.payload, LinkFact)
    }
    adjacency: dict[int, set[int]] = defaultdict(set)
    for source, target in edges:
        adjacency[source].add(target)

    def reaches(source: int, target: int) -> bool:
        pending = [source]
        visited: set[int] = set()
        while pending:
            node = pending.pop()
            if node == target:
                return True
            if node not in visited:
                visited.add(node)
                pending.extend(adjacency[node])
        return False

    for source in range(64):
        for target in range(64):
            if (
                source != target
                and source not in forbidden_nodes
                and target not in forbidden_nodes
                and (source, target) not in edges
                and not reaches(target, source)
            ):
                return LinkFact(source, target)
    raise ValueError("control preflight cannot derive its expected unreachable LINK")


def _assemble_expected_control_public(
    original: EpisodeBundle,
    ordered_facts: Sequence[ExternalEvent],
    *,
    event_ids: Sequence[int] | None = None,
    fact_timestamps: Sequence[float] | None = None,
    activation_id: int | None = None,
    activation_time: float | None = None,
    activation_payload: ActivationPayload | None = None,
    public_id: str | None = None,
) -> PublicEpisode:
    """Build the exact expected public artifact without the production rewriter."""
    source_facts = _fact_events(original)
    activation = original.public.events[-1]
    ids = tuple(range(len(ordered_facts))) if event_ids is None else tuple(event_ids)
    timestamps = (
        tuple(sorted(event.timestamp for event in source_facts))
        if fact_timestamps is None
        else tuple(float(value) for value in fact_timestamps)
    )
    if len(ids) != len(ordered_facts) or len(timestamps) != len(ordered_facts):
        raise ValueError("control preflight expected FACT coordinates are incomplete")
    expected_facts = tuple(
        ExternalEvent(event_id, timestamp, ExternalEventKind.FACT, source.payload)
        for event_id, timestamp, source in zip(ids, timestamps, ordered_facts, strict=True)
    )
    expected_activation = ExternalEvent(
        len(ordered_facts) if activation_id is None else activation_id,
        activation.timestamp if activation_time is None else float(activation_time),
        activation.kind,
        activation.payload if activation_payload is None else activation_payload,
    )
    return PublicEpisode(
        replace(
            original.public.init,
            episode_public_id=(
                original.public.init.episode_public_id if public_id is None else public_id
            ),
        ),
        (*expected_facts, expected_activation),
    )


def _expected_control_artifact(
    injector: NamedLeakInjector,
    original: AuditExample,
    *,
    corpus_position: int,
    encoded_manifest_rank: int,
    injected_hazard_target: int | None,
) -> tuple[PublicEpisode, int]:
    """Derive one canonical overlay solely from clean inputs and frozen coordinates."""
    bundle = original.bundle
    facts = list(_fact_events(bundle))
    activation = bundle.public.events[-1]
    positive = bundle.truth.recipe.variant is EpisodeVariant.POSITIVE
    control_id = injector.control_id
    expected_rank = original.manifest_rank

    if control_id == "PC_COUNT_BY_LABEL":
        target = 48 if positive else 56
        protected_ids = set(bundle.truth.relevant_record_ids)
        protected_ids.update(
            event.event_id for event in facts if isinstance(event.payload, (HazardFact, SafeFact))
        )
        removable = [
            event
            for event in facts
            if event.event_id not in protected_ids and isinstance(event.payload, LinkFact)
        ]
        remove_count = max(0, len(facts) - target)
        if remove_count > len(removable):
            raise ValueError("control preflight cannot remove a protected FACT record")
        removed_ids = (
            {event.event_id for event in removable[-remove_count:]} if remove_count else set()
        )
        expected_facts = [event for event in facts if event.event_id not in removed_ids]
        padding_count = target - len(expected_facts)
        for _ in range(padding_count):
            expected_facts.append(
                ExternalEvent(
                    max(event.event_id for event in expected_facts) + 1,
                    0.0,
                    ExternalEventKind.FACT,
                    _expected_control_unused_link(
                        expected_facts, set(bundle.truth.relevant_node_path)
                    ),
                )
            )
        if padding_count:
            last_source_time = facts[-1].timestamp
            first_padding = len(expected_facts) - padding_count
            for offset in range(padding_count):
                expected_facts[first_padding + offset] = replace(
                    expected_facts[first_padding + offset],
                    timestamp=float(
                        last_source_time
                        + (activation.timestamp - last_source_time)
                        * (offset + 1)
                        / (padding_count + 1)
                    ),
                )
        public = _assemble_expected_control_public(
            bundle,
            expected_facts,
            fact_timestamps=tuple(event.timestamp for event in expected_facts),
        )
    elif control_id == "PC_ACTIVATION_GAP_BY_LABEL":
        public = _assemble_expected_control_public(
            bundle,
            facts,
            activation_time=facts[-1].timestamp + (1.0 if positive else 4.0),
        )
    elif control_id == "PC_TERMINAL_ORDER_BY_VARIANT":
        hazards = [event for event in facts if isinstance(event.payload, HazardFact)]
        safe = next(event for event in facts if isinstance(event.payload, SafeFact))
        terminals = (hazards[0], hazards[1], safe)
        code_by_variant = {
            EpisodeVariant.POSITIVE: (0, 1, 2),
            EpisodeVariant.SAFE_NEGATIVE: (0, 2, 1),
            EpisodeVariant.DISCONNECTED_NEGATIVE: (2, 0, 1),
        }
        expected_facts = [
            terminals[index] for index in code_by_variant[bundle.truth.recipe.variant]
        ]
        expected_facts.extend(event for event in facts if event not in terminals)
        if tuple(event.payload for event in expected_facts) == tuple(
            event.payload for event in facts
        ):
            link_positions = [
                index
                for index, event in enumerate(expected_facts)
                if isinstance(event.payload, LinkFact)
            ]
            left, right = link_positions[:2]
            expected_facts[left], expected_facts[right] = (
                expected_facts[right],
                expected_facts[left],
            )
        public = _assemble_expected_control_public(bundle, expected_facts)
    elif control_id == "PC_ACTIVATION_ID_BY_LABEL":
        if not isinstance(activation.payload, ActivationPayload):
            raise ValueError("control preflight expected an activation payload")
        target = 0 if positive else 63
        source = activation.payload.start_node
        replacements = {source: target, target: source}
        if source == target:
            left, right = bundle.truth.relevant_node_path[1:3]
            replacements.update({left: right, right: left})

        def remap(value: int) -> int:
            return replacements.get(value, value)

        expected_facts = []
        for event in facts:
            payload = event.payload
            if isinstance(payload, LinkFact):
                payload = replace(
                    payload,
                    source_node=remap(payload.source_node),
                    target_node=remap(payload.target_node),
                )
            elif isinstance(payload, (HazardFact, SafeFact)):
                payload = replace(payload, node=remap(payload.node))
            expected_facts.append(replace(event, payload=payload))
        public = _assemble_expected_control_public(
            bundle,
            expected_facts,
            activation_payload=ActivationPayload(remap(source)),
        )
    elif control_id == "PC_RECORD_ID_BY_VARIANT":
        band_by_variant = {
            EpisodeVariant.POSITIVE: 0,
            EpisodeVariant.SAFE_NEGATIVE: 128,
            EpisodeVariant.DISCONNECTED_NEGATIVE: 256,
        }
        band = band_by_variant[bundle.truth.recipe.variant]
        public = _assemble_expected_control_public(
            bundle,
            facts,
            event_ids=tuple(band + index for index in range(len(facts))),
            activation_id=384,
        )
    elif control_id == "PC_PUBLIC_ID_BY_LABEL":
        derived = bytearray(
            bytes.fromhex(
                sha256_bytes(
                    canonical_json_bytes(
                        {
                            "domain": "silent-cascade/ofd-v1/pc-public-id/v1",
                            "position": corpus_position,
                        }
                    )
                )
            )[:16]
        )
        derived[0] = 0 if positive else 255
        derived[6] = (derived[6] & 0x0F) | 0x40
        derived[8] = (derived[8] & 0x3F) | 0x80
        public = _assemble_expected_control_public(
            bundle, facts, public_id=str(uuid.UUID(bytes=bytes(derived)))
        )
    elif control_id == "PC_DELAY_BY_LABEL":
        delay = 1.0 if positive else 1024.0
        expected_facts = [
            replace(
                event,
                payload=(
                    replace(event.payload, delay=delay)
                    if isinstance(event.payload, HazardFact)
                    else event.payload
                ),
            )
            for event in facts
        ]
        public = _assemble_expected_control_public(bundle, expected_facts)
    elif control_id == "PC_HAZARD_LAYOUT_BY_CLASS":
        if not positive:
            public = bundle.public
        else:
            if injected_hazard_target not in range(4):
                raise ValueError("control preflight requires its frozen hazard target")
            target = int(injected_hazard_target)
            terminal_id = bundle.truth.terminal_record_id
            if terminal_id is None:
                raise ValueError("control preflight expected a terminal hazard record")
            sentinel = next(
                (
                    event
                    for event in facts
                    if isinstance(event.payload, LinkFact)
                    and event.event_id not in bundle.truth.relevant_record_ids
                ),
                None,
            )
            if sentinel is None:
                sentinel = ExternalEvent(
                    max(event.event_id for event in facts) + 1,
                    float(facts[-1].timestamp + (activation.timestamp - facts[-1].timestamp) / 2.0),
                    ExternalEventKind.FACT,
                    _expected_control_unused_link(facts, set(bundle.truth.relevant_node_path)),
                )
                facts.append(sentinel)
            expected_facts = [
                replace(
                    event,
                    payload=(
                        replace(
                            event.payload,
                            hazard_type=(
                                target if event.event_id == terminal_id else (target + 1) % 4
                            ),
                        )
                        if isinstance(event.payload, HazardFact)
                        else event.payload
                    ),
                )
                for event in facts
            ]
            safe = next(event for event in expected_facts if isinstance(event.payload, SafeFact))
            hazards = [event for event in expected_facts if isinstance(event.payload, HazardFact)]
            sentinel = next(
                event for event in expected_facts if event.event_id == sentinel.event_id
            )
            reserved: list[ExternalEvent | None] = [None, None, None, None]
            reserved[target] = safe
            stable_fill = [*hazards, sentinel]
            for index in range(4):
                if reserved[index] is None:
                    reserved[index] = stable_fill.pop(0)
            prefix = [event for event in reserved if event is not None]
            reserved_ids = {event.event_id for event in prefix}
            prefix.extend(event for event in expected_facts if event.event_id not in reserved_ids)
            public = _assemble_expected_control_public(
                bundle,
                prefix,
                fact_timestamps=tuple(sorted(event.timestamp for event in expected_facts)),
            )
    elif control_id == "PC_MANIFEST_ORDER_BY_VARIANT":
        public = bundle.public
        expected_rank = encoded_manifest_rank
    else:
        _control_mutation_error(control_id)
    return public, expected_rank


def _require_exact_control_public_mutation(
    injector: NamedLeakInjector,
    original: AuditExample,
    transformed: AuditExample,
    *,
    corpus_position: int,
    encoded_manifest_rank: int,
    injected_hazard_target: int | None,
) -> None:
    """Reject any public artifact other than the independently reconstructed overlay."""
    expected_public, expected_rank = _expected_control_artifact(
        injector,
        original,
        corpus_position=corpus_position,
        encoded_manifest_rank=encoded_manifest_rank,
        injected_hazard_target=injected_hazard_target,
    )
    if transformed.bundle.public != expected_public or transformed.manifest_rank != expected_rank:
        _control_mutation_error(injector.control_id)


def _has_control_observation_gap_consequence(
    bundle: EpisodeBundle, validation_config: Phase1Config
) -> bool:
    facts = _fact_events(bundle)
    lower, upper = validation_config.data.observation_gap_log_uniform
    previous = bundle.public.init.initial_time
    for event in (*facts, bundle.public.events[-1]):
        gap = event.timestamp - previous
        if not lower <= gap <= upper:
            return True
        previous = event.timestamp
    return False


def _expected_control_failure_id(
    injector: NamedLeakInjector,
    row: _StoredExample,
    original: AuditExample,
    transformed: AuditExample,
    validation_config: Phase1Config,
) -> str | None:
    changed_fact_count = len(_fact_events(original.bundle)) != len(_fact_events(transformed.bundle))
    if (
        injector.control_id in {"PC_COUNT_BY_LABEL", "PC_HAZARD_LAYOUT_BY_CLASS"}
        and changed_fact_count
        and _has_control_observation_gap_consequence(transformed.bundle, validation_config)
    ):
        return "observation_gap"
    if injector.control_id == "PC_COUNT_BY_LABEL":
        return "recipe_distractor_count"
    if (
        injector.control_id == "PC_HAZARD_LAYOUT_BY_CLASS"
        and row.variant is not EpisodeVariant.POSITIVE
    ):
        return None
    if injector.control_id == "PC_HAZARD_LAYOUT_BY_CLASS" and changed_fact_count:
        return "recipe_distractor_count"
    return _EXPECTED_CONTROL_FAILURE_ID[injector.control_id]


def _control_code(
    values: np.ndarray,
    row: _StoredExample,
    injector: NamedLeakInjector,
) -> object:
    """Read only the exact predeclared disjoint code from one targeted view."""
    group = ShortcutFeatureGroup(injector.expected_detector_id.split(":", 1)[1])
    start, stop = _feature_bounds(group)
    view = values[start:stop]
    control_id = injector.control_id
    if control_id == "PC_COUNT_BY_LABEL":
        return float(view[0])
    if control_id == "PC_ACTIVATION_GAP_BY_LABEL":
        return float(view[-1])
    if control_id == "PC_TERMINAL_ORDER_BY_VARIANT":
        return tuple(int(np.argmax(view[slot * 12 + 1 : slot * 12 + 4])) for slot in range(3))
    if control_id == "PC_ACTIVATION_ID_BY_LABEL":
        return int(np.argmax(view[:64]))
    if control_id == "PC_RECORD_ID_BY_VARIANT":
        return float(view[11])
    if control_id == "PC_PUBLIC_ID_BY_LABEL":
        return float(view[0])
    if control_id == "PC_DELAY_BY_LABEL":
        return tuple(float(value) for value in view[4:6])
    if control_id == "PC_HAZARD_LAYOUT_BY_CLASS":
        return next(
            slot for slot in range(4) if int(np.argmax(view[slot * 12 + 1 : slot * 12 + 4])) == 2
        )
    if control_id == "PC_MANIFEST_ORDER_BY_VARIANT":
        return float(view[-1])
    raise ValueError("unknown positive-control code")


def _expected_control_code(row: _StoredExample, injector: NamedLeakInjector) -> object:
    variant_index = list(EpisodeVariant).index(row.variant)
    if injector.control_id == "PC_COUNT_BY_LABEL":
        return (48 if row.variant is EpisodeVariant.POSITIVE else 56) / 64.0
    if injector.control_id == "PC_ACTIVATION_GAP_BY_LABEL":
        return math.log1p(1.0 if row.variant is EpisodeVariant.POSITIVE else 4.0)
    if injector.control_id == "PC_TERMINAL_ORDER_BY_VARIANT":
        return ((1, 1, 2), (1, 2, 1), (2, 1, 1))[variant_index]
    if injector.control_id == "PC_ACTIVATION_ID_BY_LABEL":
        return 0 if row.variant is EpisodeVariant.POSITIVE else 63
    if injector.control_id == "PC_RECORD_ID_BY_VARIANT":
        return (0, 128, 256)[variant_index] / 63.0
    if injector.control_id == "PC_PUBLIC_ID_BY_LABEL":
        return 0.0 if row.variant is EpisodeVariant.POSITIVE else 1.0
    if injector.control_id == "PC_DELAY_BY_LABEL":
        encoded = math.log1p(1.0 if row.variant is EpisodeVariant.POSITIVE else 1024.0)
        return (encoded, encoded)
    if injector.control_id == "PC_HAZARD_LAYOUT_BY_CLASS":
        if row.hazard_class is None:
            raise ValueError("hazard-layout positive row has no target")
        return row.hazard_class
    raise ValueError("manifest-order codes use ordered disjoint intervals")


def _control_codes_equal(actual: object, expected: object) -> bool:
    if isinstance(actual, float) and isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6)
    if isinstance(actual, tuple) and isinstance(expected, tuple):
        return len(actual) == len(expected) and all(
            _control_codes_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def _require_control_feature_disjointness(
    values: np.ndarray,
    rows: Sequence[_StoredExample],
    indices: np.ndarray,
    injector: NamedLeakInjector,
    split_name: str,
) -> None:
    eligible = [
        int(index)
        for index in indices
        if injector.target_task is not ShortcutTask.POSITIVE_HAZARD_CLASS
        or rows[int(index)].variant is EpisodeVariant.POSITIVE
    ]
    if not eligible:
        raise ValueError(f"positive-control {split_name} target features are not disjoint")
    by_label: dict[int, set[object]] = defaultdict(set)
    labels = _task_labels(rows, injector.target_task)
    for index in eligible:
        code = _control_code(values[index], rows[index], injector)
        by_label[int(labels[index])].add(code)
        if injector.control_id != "PC_MANIFEST_ORDER_BY_VARIANT" and not _control_codes_equal(
            code,
            _expected_control_code(rows[index], injector),
        ):
            raise ValueError(f"positive-control {split_name} target features are not disjoint")
    if len(by_label) not in {2, 3, 4} or any(
        left & right
        for left_label, left in by_label.items()
        for right_label, right in by_label.items()
        if left_label < right_label
    ):
        raise ValueError(f"positive-control {split_name} target features are not disjoint")
    if injector.control_id == "PC_MANIFEST_ORDER_BY_VARIANT":
        ordered = [by_label[label] for label in sorted(by_label)]
        if any(max(left) >= min(right) for left, right in pairwise(ordered)):
            raise ValueError(f"positive-control {split_name} target features are not disjoint")


def _execute_positive_control(
    source: ReiterableAuditSource,
    rows: Sequence[_StoredExample],
    injector: NamedLeakInjector,
    config: LeakageAuditConfig,
    profile: object,
    corpus_hash: str,
    validation_config: Phase1Config,
    work: Path,
) -> PositiveControlResult:
    selected = _positive_control_subset(rows, profile, config.audit_seed, corpus_hash)
    selected_rows = tuple(rows[index] for index in selected)
    clean_hash = _corpus_hash_for_rows(rows, selected)
    train, test, split_hash = _positive_control_split(
        selected_rows, config.positive_control_seed, clean_hash
    )
    hazard_targets: dict[int, int] = {}
    for membership in (train, test):
        positives = [
            index for index in membership if selected_rows[index].variant is EpisodeVariant.POSITIVE
        ]
        if len(positives) % 4:
            raise ValueError("hazard-layout control split cannot balance four target classes")
        hazard_targets.update({index: rank % 4 for rank, index in enumerate(positives)})
    manifest_order = sorted(
        range(len(selected_rows)),
        key=lambda index: (list(EpisodeVariant).index(selected_rows[index].variant), index),
    )
    encoded_ranks = {index: rank for rank, index in enumerate(manifest_order)}
    global_to_local = {global_index: local for local, global_index in enumerate(selected)}
    control_path = work / "positive-control.f32"
    control_values = np.memmap(
        control_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(selected), _TOTAL_FEATURE_DIMENSION),
    )
    control_rows: list[_StoredExample] = []
    injected_entries: list[CorpusDigestEntry | None] = [None] * len(selected)
    seen = 0
    try:
        for source_index, example in enumerate(source.iter_examples()):
            local = global_to_local.get(source_index)
            if local is None:
                continue
            if local % config.feature_batch_size == 0:
                _resource_guard(config, "control")
            transformed = _rewrite_positive_control(
                injector,
                example,
                local,
                encoded_manifest_rank=encoded_ranks[local],
                injected_hazard_target=hazard_targets.get(local),
                minimum_observation_gap=validation_config.data.observation_gap_log_uniform[0],
            )
            _require_exact_control_public_mutation(
                injector,
                example,
                transformed,
                corpus_position=local,
                encoded_manifest_rank=encoded_ranks[local],
                injected_hazard_target=hazard_targets.get(local),
            )
            report = validate_episode_invariants(
                transformed.bundle, validation_config, strict=False
            )
            expected_failure = _expected_control_failure_id(
                injector,
                selected_rows[local],
                example,
                transformed,
                validation_config,
            )
            actual_failure = (
                None
                if report.valid
                else (report.check_ids[0] if len(report.check_ids) == 1 else "multiple")
            )
            if actual_failure != expected_failure:
                raise ValueError(
                    "positive-control construction failure was not exactly authorized: "
                    f"{injector.control_id} expected={expected_failure} "
                    f"actual={actual_failure}"
                )
            feature_set = extract_shortcut_features(transformed, len(selected))
            control_values[local] = feature_set.vectors[ShortcutFeatureGroup.COMBINED]
            bundle = transformed.bundle
            digest = episode_sha256(bundle)
            injected_entries[local] = CorpusDigestEntry(
                bundle.public.init.episode_public_id, digest
            )
            control_rows.append(
                _StoredExample(
                    bundle.public.init.episode_public_id,
                    digest,
                    feature_set.audit_group_id,
                    bundle.truth.key.suite,
                    bundle.truth.recipe.requested_path_length,
                    bundle.truth.recipe.variant,
                    bundle.truth.relevant_hazard_type,
                    transformed.randomization_block_index,
                    transformed.episode_position,
                    tuple(
                        sorted(
                            event.payload.hazard_type
                            for event in _fact_events(bundle)
                            if isinstance(event.payload, HazardFact)
                        )
                    ),
                )
            )
            seen += 1
        if seen != len(selected):
            raise ValueError("positive-control source regeneration is incomplete")
        control_values.flush()
        injected_digest = CorpusHashBuilder(len(selected))
        for local in range(len(selected)):
            entry = injected_entries[local]
            if entry is None:
                raise ValueError("positive-control injected digest is incomplete")
            injected_digest.add(entry)
        injected_hash = injected_digest.finalize()
        _require_control_feature_disjointness(
            control_values, control_rows, train, injector, "train"
        )
        _require_control_feature_disjointness(control_values, control_rows, test, injector, "test")
        probes: list[ShortcutProbeResult] = []
        for group in ShortcutFeatureGroup:
            probe = _run_positive_control_probe(
                control_values,
                control_rows,
                train,
                test,
                injector,
                group,
                config,
                profile,
                injected_hash,
            )
            if probe.task is not injector.target_task or probe.feature_group is not group:
                raise ValueError("positive-control detector identity was not actually observed")
            probes.append(probe)
        probes = _holm(probes, config.alpha)
    finally:
        del control_values
        control_path.unlink(missing_ok=True)
    full_gate = profile.enforce_clean_statistical_gate
    observed = tuple(
        f"{probe.task.value}:{probe.feature_group.value}"
        for probe in probes
        if probe.balanced_accuracy >= config.positive_control_min_balanced_accuracy
        and probe.raw_permutation_p <= 0.05
        and (not full_gate or probe.holm_adjusted_p < config.alpha)
    )
    expected_group = ShortcutFeatureGroup(injector.expected_detector_id.split(":", 1)[1])
    expected_probe = next(
        probe
        for probe in probes
        if probe.task is injector.target_task and probe.feature_group is expected_group
    )
    passed = injector.expected_detector_id in observed
    return PositiveControlResult(
        control_id=injector.control_id,
        target_task=injector.target_task,
        expected_detector_id=injector.expected_detector_id,
        observed_detector_ids=observed,
        base_subset_corpus_sha256=clean_hash,
        injected_corpus_sha256=injected_hash,
        split_membership_sha256=split_hash,
        balanced_accuracy=expected_probe.balanced_accuracy,
        holm_adjusted_p=expected_probe.holm_adjusted_p,
        passed=passed,
    )


def audit_leakage(
    source: ReiterableAuditSource,
    config: LeakageAuditConfig,
    profile: LeakageAuditProfileName,
    provenance: EvidenceProvenance,
    workspace: Path,
    positive_control: str | None = None,
) -> LeakageReport:
    """Run a two-pass bounded audit, failing closed on evidence or resource corruption."""
    if (
        not isinstance(config, LeakageAuditConfig)
        or not isinstance(profile, LeakageAuditProfileName)
        or not isinstance(provenance, EvidenceProvenance)
        or not isinstance(workspace, Path)
    ):
        raise TypeError("leakage audit received invalid typed inputs")
    _validate_provenance(source, config, provenance)
    authentication = _validate_source_authentication(source, profile)
    _validate_independent_trust_anchor(source, profile, provenance, authentication)
    validation_config = source.validation_config  # type: ignore[attr-defined]
    assert isinstance(validation_config, Phase1Config)
    selected_profile = _profile_config(config, profile)
    _validate_holm_attainability(config, selected_profile)
    if source.episode_count != selected_profile.episode_count:
        raise ValueError("source episode count does not equal frozen audit profile")
    if not workspace.exists() or not workspace.is_dir():
        raise ValueError("leakage workspace must be caller-owned existing directory")
    required = source.episode_count * _TOTAL_FEATURE_DIMENSION * np.dtype(np.float32).itemsize
    if required > config.max_feature_store_bytes:
        raise MemoryError("leakage feature-store ceiling exceeded")
    work = Path(mkdtemp(prefix="silent-cascade-leakage-", dir=workspace))
    feature_path = work / "features.f32"
    features: np.memmap | None = None
    public_values: np.ndarray | None = None
    try:
        _allocate_feature_store(feature_path, source.episode_count)
        feature_buffer = np.empty(
            (
                min(config.feature_batch_size, source.episode_count),
                _TOTAL_FEATURE_DIMENSION,
            ),
            dtype=np.float32,
        )
        rows: list[_StoredExample] = []
        digest = CorpusHashBuilder(source.episode_count)
        source_manifest = _SourceManifestHashBuilder()
        seen_tokens: set[tuple[int, int]] = set()
        denominators: Counter[str] = Counter()
        active_matched_block: int | None = None
        active_matched_bundles: list[EpisodeBundle] = []

        def flush_matched_group() -> None:
            if active_matched_block is not None:
                validate_cohort_invariants(tuple(active_matched_bundles), validation_config)  # type: ignore[arg-type]

        for index, example in enumerate(source.iter_examples()):
            if index % config.feature_batch_size == 0:
                _resource_guard(config, "extraction")
            if example.generation_mode != source.descriptor.generation_mode:
                raise ValueError("example generation mode differs from source")
            _validate_audit_coordinate(example, index, source.descriptor)
            validate_episode_invariants(example.bundle, validation_config)
            if example.generation_mode == "matched":
                if active_matched_block is None:
                    active_matched_block = example.randomization_block_index
                elif example.randomization_block_index != active_matched_block:
                    flush_matched_group()
                    active_matched_block = example.randomization_block_index
                    active_matched_bundles.clear()
                active_matched_bundles.append(example.bundle)
            solution = solve_public_episode(example.bundle.public)
            verify_oracle_truth(solution, example.bundle.truth)
            token = (example.randomization_block_index, example.episode_position)
            if token in seen_tokens:
                raise ValueError("audit seed-token collision")
            seen_tokens.add(token)
            feature_set = extract_shortcut_features(example, source.episode_count)
            feature_buffer[index % len(feature_buffer)] = feature_set.vectors[
                ShortcutFeatureGroup.COMBINED
            ]
            if (index + 1) % len(feature_buffer) == 0:
                _write_feature_batch(
                    feature_path,
                    source.episode_count,
                    index + 1 - len(feature_buffer),
                    feature_buffer,
                    config,
                )
            bundle = example.bundle
            bundle_digest = episode_sha256(bundle)
            digest.add(CorpusDigestEntry(bundle.public.init.episode_public_id, bundle_digest))
            source_manifest.add(example, bundle_digest)
            rows.append(
                _StoredExample(
                    bundle.public.init.episode_public_id,
                    bundle_digest,
                    feature_set.audit_group_id,
                    bundle.truth.key.suite,
                    bundle.truth.recipe.requested_path_length,
                    bundle.truth.recipe.variant,
                    bundle.truth.relevant_hazard_type,
                    example.randomization_block_index,
                    example.episode_position,
                    tuple(
                        sorted(
                            event.payload.hazard_type
                            for event in _fact_events(bundle)
                            if isinstance(event.payload, HazardFact)
                        )
                    ),
                )
            )
            denominators[
                f"{bundle.truth.key.suite.value}:{bundle.truth.recipe.requested_path_length}"
            ] += 1
        if len(rows) != source.episode_count:
            raise ValueError("audit source yielded the wrong number of examples")
        remainder = len(rows) % len(feature_buffer)
        if remainder:
            _write_feature_batch(
                feature_path,
                source.episode_count,
                len(rows) - remainder,
                feature_buffer[:remainder],
                config,
            )
        del feature_buffer
        if source.descriptor.generation_mode == "matched":
            flush_matched_group()
        else:
            _validate_independent_quartets(rows)
        corpus_hash = digest.finalize()
        if source_manifest.finalize() != authentication.source_manifest_sha256:
            raise ValueError("audit source order or membership differs from authentication")
        if dict(sorted(denominators.items())) != authentication.suite_path_denominators:
            raise ValueError("audit source strata differ from authenticated profile denominators")
        # Regeneration authentication includes source order, public ID, and digest.
        second_count = 0
        for second_count, example in enumerate(source.iter_examples(), start=1):
            if (second_count - 1) % config.feature_batch_size == 0:
                _resource_guard(config, "extraction")
            index = second_count - 1
            if index >= len(rows) or (
                example.bundle.public.init.episode_public_id,
                episode_sha256(example.bundle),
            ) != (rows[index].public_id, rows[index].digest):
                raise ValueError("audit source second pass differs from first pass")
        if second_count != len(rows):
            raise ValueError("audit source second pass differs from first pass")
        train, test, split_hash = _split_memberships(
            rows,
            config.audit_seed,
            corpus_hash,
            source.descriptor.generation_mode,
            strict_divisible=profile is LeakageAuditProfileName.PHASE1_GATE,
        )
        features = np.memmap(
            feature_path,
            dtype=np.float32,
            mode="r",
            shape=(source.episode_count, _TOTAL_FEATURE_DIMENSION),
        )
        public_values = features
        # Every actual counterfactual stream is authenticated before the first
        # statistical fit.  A corrupt clock child/order/count/manifest must not
        # consume clean, shuffled, or control optimizer work.
        counterfactual = _counterfactual_checks(source, profile, config)
        probes = _run_probes(
            public_values, rows, train, test, config, selected_profile, corpus_hash
        )
        shuffled_probes = _run_probes(
            public_values,
            rows,
            train,
            test,
            config,
            selected_profile,
            corpus_hash,
            label_shuffled=True,
        )
        train_hash = sha256_bytes(
            canonical_json_bytes(
                {
                    "domain": "silent-cascade/ofd-v1/leakage-members/v1",
                    "public_ids": [rows[index].public_id for index in train],
                }
            )
        )
        test_hash = sha256_bytes(
            canonical_json_bytes(
                {
                    "domain": "silent-cascade/ofd-v1/leakage-members/v1",
                    "public_ids": [rows[index].public_id for index in test],
                }
            )
        )
        active_injector = source.injector if isinstance(source, _InjectedAuditSource) else None
        if positive_control is not None and (
            active_injector is None or active_injector.control_id != positive_control
        ):
            raise ValueError("positive-control request does not match injected source")
        controls: tuple[PositiveControlResult, ...] = ()
        if active_injector is not None:
            controls = (
                _execute_positive_control(
                    source,
                    rows,
                    active_injector,
                    config,
                    selected_profile,
                    corpus_hash,
                    validation_config,
                    work,
                ),
            )
        clean_pass = not selected_profile.enforce_clean_statistical_gate or all(
            probe.passed for probe in probes
        )
        return LeakageReport(
            schema_version="leakage-report-v1",
            provenance=provenance,
            generation_mode=source.descriptor.generation_mode,
            profile=profile,
            corpus_hash=corpus_hash,
            feature_schema_hash=sha256_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": "leakage-features-v1",
                        "dimensions": _FEATURE_DIMENSIONS,
                        "groups": [group.value for group in ShortcutFeatureGroup],
                    }
                )
            ),
            split_membership_hash=split_hash,
            train_membership_hash=train_hash,
            test_membership_hash=test_hash,
            episode_count=len(rows),
            randomization_block_count=len({row.block for row in rows}),
            suite_path_denominators=dict(sorted(denominators.items())),
            construction_checks={
                "provenance": True,
                "public_ids": True,
                "seed_tokens": True,
                "invariants": True,
                "finite_features": True,
                "two_pass_identity": True,
            },
            probes=tuple(probes),
            positive_controls=controls,
            counterfactual_checks=counterfactual,
            label_shuffled_control_passed=all(probe.passed for probe in shuffled_probes),
            passed=(
                not controls
                and clean_pass
                and all(probe.passed for probe in shuffled_probes)
                and all(item.passed for item in counterfactual)
                and all(item.passed for item in controls)
            ),
        )
    finally:
        if public_values is not None:
            del public_values
        if features is not None:
            del features
        try:
            shutil.rmtree(work)
        except OSError as error:
            raise RuntimeError("leakage audit workspace cleanup failed") from error
        _resource_guard(config, "cleanup")
