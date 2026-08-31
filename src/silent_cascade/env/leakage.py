"""Fail-closed, deterministic audits for public OFD shortcut features.

The auditor is deliberately the sole Phase 1 component that sees an
``EpisodeBundle``.  Feature extraction reads only ``bundle.public``; private
truth is used only to construct the explicitly declared audit targets and, in
named positive-control mode, to synthesize a known leak.
"""

from __future__ import annotations

import math
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import permutations
from pathlib import Path
from tempfile import mkdtemp
from typing import Literal, Protocol

import numpy as np
import psutil
from pydantic import Field, model_validator
from scipy.optimize import minimize
from scipy.special import logsumexp

from silent_cascade.env.config import LeakageAuditConfig, SplitNamespace, SuiteName
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
from silent_cascade.env.oracle import solve_public_episode, verify_oracle_truth
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.provenance import EvidenceProvenance
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


class ReiterableAuditSource(Protocol):
    @property
    def descriptor(self) -> AuditSourceDescriptor: ...

    @property
    def episode_count(self) -> int: ...

    def iter_examples(self) -> Iterator[AuditExample]: ...


@dataclass(frozen=True, slots=True)
class InMemoryAuditSource:
    descriptor: AuditSourceDescriptor
    examples: tuple[AuditExample, ...]

    @property
    def episode_count(self) -> int:
        return len(self.examples)

    def iter_examples(self) -> Iterator[AuditExample]:
        return iter(self.examples)


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

    def iter_examples(self) -> Iterator[AuditExample]:
        return (
            _rewrite_positive_control(self.injector, example, index)
            for index, example in enumerate(self.source.iter_examples())
        )


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


def _public_with_facts(bundle: EpisodeBundle, facts: Sequence[ExternalEvent]) -> EpisodeBundle:
    activation = bundle.public.events[-1]
    if len(facts) >= 2:
        start = bundle.public.init.initial_time
        stop = activation.timestamp
        step = (stop - start) / (len(facts) + 1)
        facts = tuple(
            ExternalEvent(
                event.event_id, float(start + step * (index + 1)), event.kind, event.payload
            )
            for index, event in enumerate(facts)
        )
    return replace(bundle, public=PublicEpisode(bundle.public.init, (*facts, activation)))


def _rewrite_positive_control(
    injector: NamedLeakInjector, example: AuditExample, corpus_position: int
) -> AuditExample:
    """Construct the declared injected public artifact; no predictor overlay exists."""
    bundle = example.bundle
    positive = bundle.truth.recipe.variant is EpisodeVariant.POSITIVE
    variant_index = list(EpisodeVariant).index(bundle.truth.recipe.variant)
    facts = list(_fact_events(bundle))
    if injector.control_id == "PC_COUNT_BY_LABEL":
        target = 48 if positive else 56
        if len(facts) > target:
            raise ValueError("count positive control cannot remove public records")
        next_id = max(event.event_id for event in bundle.public.events) + 1
        for offset in range(target - len(facts)):
            facts.append(
                ExternalEvent(
                    next_id + offset,
                    facts[-1].timestamp,
                    ExternalEventKind.FACT,
                    LinkFact(48 + offset // 8, 56 + offset % 8),
                )
            )
        bundle = _public_with_facts(bundle, facts)
    elif injector.control_id == "PC_ACTIVATION_GAP_BY_LABEL":
        activation = bundle.public.events[-1]
        replacement = ExternalEvent(
            activation.event_id,
            float(facts[-1].timestamp + (1.0 if positive else 4.0)),
            activation.kind,
            activation.payload,
        )
        bundle = replace(bundle, public=PublicEpisode(bundle.public.init, (*facts, replacement)))
    elif injector.control_id == "PC_TERMINAL_ORDER_BY_VARIANT":
        hazards = [event for event in facts if isinstance(event.payload, HazardFact)]
        safes = [event for event in facts if isinstance(event.payload, SafeFact)]
        other = [event for event in facts if event not in hazards and event not in safes]
        code = ((0, 1, 2), (0, 2, 1), (2, 0, 1))[variant_index]
        terminals = (hazards[0], hazards[1], safes[0])
        bundle = _public_with_facts(bundle, [terminals[index] for index in code] + other)
    elif injector.control_id == "PC_ACTIVATION_ID_BY_LABEL":
        activation = bundle.public.events[-1]
        assert isinstance(activation.payload, ActivationPayload)
        target = 63 if positive else 0
        source = activation.payload.start_node

        def remap(value: int) -> int:
            if value == source:
                return target
            if value == target:
                return source
            return value

        rewritten: list[ExternalEvent] = []
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
            rewritten.append(ExternalEvent(event.event_id, event.timestamp, event.kind, payload))
        bundle = replace(
            bundle,
            public=PublicEpisode(
                bundle.public.init,
                (
                    *rewritten,
                    ExternalEvent(
                        activation.event_id,
                        activation.timestamp,
                        activation.kind,
                        ActivationPayload(target),
                    ),
                ),
            ),
        )
    elif injector.control_id == "PC_RECORD_ID_BY_VARIANT":
        band = (0, 128, 256)[variant_index]
        rewritten = [
            ExternalEvent(band + index, event.timestamp, event.kind, event.payload)
            for index, event in enumerate(facts)
        ]
        activation = bundle.public.events[-1]
        bundle = replace(
            bundle,
            public=PublicEpisode(
                bundle.public.init,
                (
                    *rewritten,
                    ExternalEvent(384, activation.timestamp, activation.kind, activation.payload),
                ),
            ),
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
        raw[0] = 255 if positive else 0
        raw[6] = (raw[6] & 0x0F) | 0x40
        raw[8] = (raw[8] & 0x3F) | 0x80
        bundle = replace(
            bundle,
            public=replace(
                bundle.public,
                init=replace(
                    bundle.public.init, episode_public_id=str(uuid.UUID(bytes=bytes(raw)))
                ),
            ),
        )
    elif injector.control_id == "PC_DELAY_BY_LABEL":
        delay = 1.0 if positive else 1024.0
        rewritten = [
            ExternalEvent(
                event.event_id,
                event.timestamp,
                event.kind,
                replace(event.payload, delay=delay)
                if isinstance(event.payload, HazardFact)
                else event.payload,
            )
            for event in facts
        ]
        bundle = _public_with_facts(bundle, rewritten)
    elif injector.control_id == "PC_HAZARD_LAYOUT_BY_CLASS":
        if not positive:
            return example
        target = bundle.truth.relevant_hazard_type
        assert target is not None
        hazards = [event for event in facts if isinstance(event.payload, HazardFact)]
        safes = [event for event in facts if isinstance(event.payload, SafeFact)]
        links = [event for event in facts if isinstance(event.payload, LinkFact)]
        sentinel = links[0]
        others = [
            event for event in facts if event not in {hazards[0], hazards[1], safes[0], sentinel}
        ]
        slots: list[ExternalEvent | None] = [None, None, None, None]
        slots[target] = safes[0]
        remaining = [hazards[0], hazards[1], sentinel]
        for index in range(4):
            if slots[index] is None:
                slots[index] = remaining.pop(0)
        bundle = _public_with_facts(bundle, [item for item in slots if item is not None] + others)
    elif injector.control_id == "PC_MANIFEST_ORDER_BY_VARIANT":
        example = replace(example, manifest_rank=variant_index * 1_000_000 + corpus_position)
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
            clean_pass
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


@dataclass(frozen=True, slots=True)
class _FitResult:
    predictions: np.ndarray
    iterations: int
    converged: bool


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


def _resource_guard(config: LeakageAuditConfig) -> None:
    if psutil.Process().memory_info().rss > config.max_resident_working_bytes:
        raise MemoryError("leakage audit resident working-set ceiling exceeded")


def _validate_audit_coordinate(example: AuditExample, rank: int) -> None:
    if example.manifest_rank != rank:
        raise ValueError("audit manifest rank is not canonical source order")
    coordinate = example.bundle.truth.key.coordinate
    if example.generation_mode == "matched":
        if not isinstance(coordinate, MatchedEpisodeCoordinate) or (
            coordinate.cohort_index,
            coordinate.member_index,
        ) != (example.randomization_block_index, example.episode_position):
            raise ValueError("matched audit coordinate is not authenticated")
    elif not isinstance(coordinate, IndependentEpisodeCoordinate) or (
        coordinate.allocation_quartet_index != example.randomization_block_index
    ):
        raise ValueError("independent audit coordinate is not authenticated")


def _matched_nuisance_signature(bundle: EpisodeBundle) -> tuple[object, ...]:
    facts = _fact_events(bundle)
    degree: dict[int, list[int]] = {}
    for event in facts:
        if isinstance(event.payload, LinkFact):
            degree.setdefault(event.payload.source_node, [0, 0])[1] += 1
            degree.setdefault(event.payload.target_node, [0, 0])[0] += 1
    links = tuple(sorted((values[0], values[1]) for values in degree.values()))
    counts = (
        sum(isinstance(event.payload, HazardFact) for event in facts),
        sum(isinstance(event.payload, SafeFact) for event in facts),
    )
    return (
        links,
        tuple(event.timestamp for event in facts),
        counts,
        bundle.truth.episode_delay,
    )


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


def _fit_logistic(
    x: np.ndarray, labels: np.ndarray, classes: np.ndarray, config: LeakageAuditConfig
) -> _FitResult:
    targets = np.searchsorted(classes, labels)
    n_classes = len(classes)
    n_features = x.shape[1]
    class_counts = np.bincount(targets, minlength=n_classes).astype(np.float64)
    if np.any(class_counts == 0):
        raise ValueError("optimizer received a missing class")
    weights = 1.0 / class_counts[targets]
    weights /= weights.sum()

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = flat[: n_features * n_classes].reshape(n_features, n_classes)
        intercept = flat[n_features * n_classes :]
        logits = x @ coefficients + intercept
        log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -np.sum(
            weights * log_probs[np.arange(len(labels)), targets]
        ) + 0.5 * config.l2_penalty * np.sum(coefficients**2)
        probs = np.exp(log_probs)
        residual = (probs - np.eye(n_classes)[targets]) * weights[:, None]
        gradient = np.concatenate(
            ((x.T @ residual + config.l2_penalty * coefficients).ravel(), residual.sum(axis=0))
        )
        return float(loss), gradient

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
    parameters = result.x
    logits = (
        x @ parameters[: n_features * n_classes].reshape(n_features, n_classes)
        + parameters[n_features * n_classes :]
    )
    return _FitResult(classes[np.argmax(logits, axis=1)], int(result.nit), True)


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray, classes: np.ndarray) -> float:
    recalls = [np.mean(predictions[labels == value] == value) for value in classes]
    if any(not np.isfinite(value) for value in recalls):
        raise ValueError("balanced accuracy has a missing class")
    return float(np.mean(recalls))


def _permuted_labels(
    rows: Sequence[_StoredExample],
    labels: np.ndarray,
    task: ShortcutTask,
    replicate: int,
    corpus_hash: str,
    audit_seed: int,
) -> np.ndarray:
    result = labels.copy()
    groups: dict[tuple[SuiteName, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row.suite, row.path_length, row.block)].append(index)
    for (suite, path, block), indices in groups.items():
        group_labels = tuple(int(labels[index]) for index in indices)
        if task is ShortcutTask.POSITIVE_HAZARD_CLASS:
            # Hazards are independently swapped inside a public episode by the
            # counterfactual/control source; a group label shuffle is invalid.
            continue
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


def _feature_views(values: np.ndarray) -> dict[ShortcutFeatureGroup, np.ndarray]:
    result: dict[ShortcutFeatureGroup, np.ndarray] = {}
    start = 0
    for group, width in zip(tuple(ShortcutFeatureGroup)[:-1], _FEATURE_DIMENSIONS, strict=True):
        result[group] = values[:, start : start + width]
        start += width
    result[ShortcutFeatureGroup.COMBINED] = values
    return result


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
        [rows[index].hazard_class for index in hazard_indices], dtype=np.int8
    )
    for task, labels in labels_by_task.items():
        if label_shuffled and task is not ShortcutTask.POSITIVE_HAZARD_CLASS:
            labels = _permuted_labels(rows, labels, task, -1, corpus_hash, config.audit_seed)
        indices = (
            hazard_indices if task is ShortcutTask.POSITIVE_HAZARD_CLASS else np.arange(len(rows))
        )
        task_train = np.intersect1d(train, indices, assume_unique=True)
        task_test = np.intersect1d(test, indices, assume_unique=True)
        local_labels = labels
        if task is ShortcutTask.POSITIVE_HAZARD_CLASS:
            local_indices = {global_index: local for local, global_index in enumerate(indices)}
            train_rows = np.asarray([local_indices[index] for index in task_train], dtype=np.int64)
            test_rows = np.asarray([local_indices[index] for index in task_test], dtype=np.int64)
        else:
            train_rows, test_rows = task_train, task_test
        classes = np.unique(local_labels)
        if len(classes) < (
            4
            if task is ShortcutTask.POSITIVE_HAZARD_CLASS
            else (3 if task is ShortcutTask.VARIANT_THREE_WAY else 2)
        ):
            raise ValueError("shortcut task is missing a required class")
        if (
            min(np.sum(local_labels[test_rows] == item) for item in classes)
            < profile.minimum_test_examples_per_class
        ):
            raise ValueError("shortcut task has insufficient held-out examples")
        for group, all_values in _feature_views(values).items():
            selected = all_values[indices]
            continuous_mask = _continuous_columns(group, all_values.shape[1])
            if task is ShortcutTask.POSITIVE_HAZARD_CLASS:
                allowed = ~_hazard_identity_columns(group)
                selected = selected[:, allowed]
                continuous_mask = continuous_mask[allowed]
            x_train, x_test = _standardize(
                selected[train_rows],
                selected[test_rows],
                group,
                continuous_mask=continuous_mask,
            )
            # Fit once, then stream held-out label permutations without refitting.
            predictions, iterations = _fit_predict(
                x_train, local_labels[train_rows], x_test, classes, config
            )
            observed = _balanced_accuracy(local_labels[test_rows], predictions, classes)
            exceed = 0
            test_rows_global = indices[test_rows]
            for replicate_start in range(
                0, profile.permutation_replicates, config.permutation_batch_size
            ):
                for replicate in range(
                    replicate_start,
                    min(
                        replicate_start + config.permutation_batch_size,
                        profile.permutation_replicates,
                    ),
                ):
                    permuted_all = _permuted_labels(
                        rows,
                        np.asarray(
                            [row.variant is EpisodeVariant.POSITIVE for row in rows], dtype=np.int8
                        )
                        if task is ShortcutTask.POSITIVE_BINARY
                        else (
                            np.asarray(
                                [list(EpisodeVariant).index(row.variant) for row in rows],
                                dtype=np.int8,
                            )
                            if task is ShortcutTask.VARIANT_THREE_WAY
                            else np.asarray(
                                [
                                    row.hazard_class
                                    if row.variant is EpisodeVariant.POSITIVE
                                    else -1
                                    for row in rows
                                ],
                                dtype=np.int8,
                            )
                        ),
                        task,
                        replicate,
                        corpus_hash,
                        config.audit_seed,
                    )
                    permuted = permuted_all[test_rows_global]
                    exceed += _balanced_accuracy(permuted, predictions, classes) >= observed
            raw_p = (1 + exceed) / (profile.permutation_replicates + 1)
            probes.append(
                ShortcutProbeResult(
                    task=task,
                    feature_group=group,
                    feature_dimension=selected.shape[1],
                    train_examples=len(train_rows),
                    test_examples=len(test_rows),
                    train_class_counts={
                        str(value): int(np.sum(local_labels[train_rows] == value))
                        for value in classes
                    },
                    test_class_counts={
                        str(value): int(np.sum(local_labels[test_rows] == value))
                        for value in classes
                    },
                    raw_accuracy=float(np.mean(predictions == local_labels[test_rows])),
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


def _fit_predict(
    x_train: np.ndarray,
    labels: np.ndarray,
    x_test: np.ndarray,
    classes: np.ndarray,
    config: LeakageAuditConfig,
) -> tuple[np.ndarray, int]:
    targets = np.searchsorted(classes, labels)
    n_features, n_classes = x_train.shape[1], len(classes)
    counts = np.bincount(targets, minlength=n_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("optimizer received a missing class")
    weights = 1.0 / counts[targets]
    weights /= weights.sum()
    eye = np.eye(n_classes)

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = flat[: n_features * n_classes].reshape(n_features, n_classes)
        intercept = flat[n_features * n_classes :]
        logits = x_train @ coefficients + intercept
        log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
        residual = (np.exp(log_probs) - eye[targets]) * weights[:, None]
        return float(
            -np.sum(weights * log_probs[np.arange(len(labels)), targets])
            + 0.5 * config.l2_penalty * np.sum(coefficients * coefficients)
        ), np.concatenate(
            (
                (x_train.T @ residual + config.l2_penalty * coefficients).ravel(),
                residual.sum(axis=0),
            )
        )

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
    return classes[np.argmax(x_test @ coefficients + intercept, axis=1)], int(result.nit)


def _counterfactual_result(
    check_id: CounterfactualCheckId, pairs: Sequence[CounterfactualPairResult]
) -> CounterfactualCheckResult:
    if not pairs:
        raise ValueError(f"counterfactual source has no {check_id.value} pairs")
    decision_mismatches = sum(pair.decision_mismatch for pair in pairs)
    temporal_mismatches = sum(pair.temporal_mismatch for pair in pairs)
    return CounterfactualCheckResult(
        check_id=check_id,
        checked_pairs=len(pairs),
        decision_mismatch_count=decision_mismatches,
        temporal_mismatch_count=temporal_mismatches,
        result_payload_sha256=counterfactual_result_payload_hash(check_id, pairs),
        passed=decision_mismatches == 0 and temporal_mismatches == 0,
    )


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
    if solution.terminal_record_id is None or solution.public_delay is None:
        raise ValueError("terminal-delay counterfactual requires a public hazard")
    rewritten: list[ExternalEvent] = []
    for event in bundle.public.events:
        payload = event.payload
        if event.event_id == solution.terminal_record_id and isinstance(payload, HazardFact):
            payload = replace(payload, delay=delay)
        rewritten.append(ExternalEvent(event.event_id, event.timestamp, event.kind, payload))
    return PublicEpisode(bundle.public.init, tuple(rewritten))


def _solution_signature(public: PublicEpisode) -> tuple[object, ...]:
    solved = solve_public_episode(public)
    return (
        solved.terminal_kind,
        solved.hazard_type,
        solved.node_path,
        solved.public_delay,
    )


def _counterfactual_checks(source: ReiterableAuditSource) -> tuple[CounterfactualCheckResult, ...]:
    delay_pairs: list[CounterfactualPairResult] = []
    presentation_pairs: list[CounterfactualPairResult] = []
    clock_pairs: list[CounterfactualPairResult] = []
    pending_positive: dict[tuple[SuiteName, int], AuditExample] = {}
    for example in source.iter_examples():
        bundle = example.bundle
        public_id = bundle.public.init.episode_public_id
        original = solve_public_episode(bundle.public)
        permuted = _reordered_public(bundle, public_id)
        presentation_pairs.append(
            CounterfactualPairResult(
                pair_key_sha256=counterfactual_pair_key(
                    CounterfactualCheckId.PRESENTATION_PERMUTATION,
                    (public_id,),
                    "permute_presentation",
                ),
                decision_mismatch=_solution_signature(permuted)[:3]
                != _solution_signature(bundle.public)[:3],
                temporal_mismatch=False,
            )
        )
        if original.public_delay is not None:
            key = (bundle.truth.key.suite, bundle.truth.recipe.requested_path_length)
            partner = pending_positive.pop(key, None)
            if partner is None:
                pending_positive[key] = example
            else:
                partner_solution = solve_public_episode(partner.bundle.public)
                left = solve_public_episode(
                    _with_terminal_delay(bundle, partner_solution.public_delay)
                )
                right = solve_public_episode(
                    _with_terminal_delay(partner.bundle, original.public_delay)
                )
                delay_pairs.append(
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
                            left.terminal_kind is not original.terminal_kind
                            or right.terminal_kind is not partner_solution.terminal_kind
                            or left.hazard_type != original.hazard_type
                            or right.hazard_type != partner_solution.hazard_type
                        ),
                        temporal_mismatch=(
                            left.public_delay != partner_solution.public_delay
                            or right.public_delay != original.public_delay
                        ),
                    )
                )
        for suite, tag in (
            (SuiteName.CLOCK_SCALE_0_1X, "scale_0_1x"),
            (SuiteName.CLOCK_SCALE_10X, "scale_10x"),
        ):
            scaled = scale_episode_time(bundle, suite, _derived_public_id(tag, public_id))
            scaled_solution = solve_public_episode(scaled.public)
            clock_pairs.append(
                CounterfactualPairResult(
                    pair_key_sha256=counterfactual_pair_key(
                        CounterfactualCheckId.PAIRED_CLOCK_SCALE, (public_id,), tag
                    ),
                    decision_mismatch=(
                        scaled_solution.terminal_kind is not original.terminal_kind
                        or scaled_solution.hazard_type != original.hazard_type
                    ),
                    temporal_mismatch=(
                        original.public_delay is not None
                        and scaled_solution.public_delay
                        != original.public_delay * scaled.truth.recipe.clock_scale
                    ),
                )
            )
    if pending_positive:
        raise ValueError("counterfactual source leaves an unpaired positive episode")
    return (
        _counterfactual_result(CounterfactualCheckId.TERMINAL_DELAY_SWAP, delay_pairs),
        _counterfactual_result(CounterfactualCheckId.PRESENTATION_PERMUTATION, presentation_pairs),
        _counterfactual_result(CounterfactualCheckId.PAIRED_CLOCK_SCALE, clock_pairs),
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
    selected_profile = _profile_config(config, profile)
    if source.episode_count != selected_profile.episode_count:
        raise ValueError("source episode count does not equal frozen audit profile")
    if not workspace.exists() or not workspace.is_dir():
        raise ValueError("leakage workspace must be caller-owned existing directory")
    required = source.episode_count * _TOTAL_FEATURE_DIMENSION * np.dtype(np.float32).itemsize
    if required > config.max_feature_store_bytes:
        raise MemoryError("leakage feature-store ceiling exceeded")
    work = Path(mkdtemp(prefix="silent-cascade-leakage-", dir=workspace))
    feature_path = work / "features.f32"
    try:
        features = np.memmap(
            feature_path,
            dtype=np.float32,
            mode="w+",
            shape=(source.episode_count, _TOTAL_FEATURE_DIMENSION),
        )
        rows: list[_StoredExample] = []
        digest = CorpusHashBuilder(source.episode_count)
        seen_tokens: set[tuple[int, int]] = set()
        denominators: Counter[str] = Counter()
        matched_groups: dict[int, list[tuple[EpisodeVariant, tuple[object, ...]]]] = defaultdict(
            list
        )
        for index, example in enumerate(source.iter_examples()):
            if index % config.feature_batch_size == 0:
                _resource_guard(config)
            if example.generation_mode != source.descriptor.generation_mode:
                raise ValueError("example generation mode differs from source")
            _validate_audit_coordinate(example, index)
            solution = solve_public_episode(example.bundle.public)
            verify_oracle_truth(solution, example.bundle.truth)
            token = (example.randomization_block_index, example.episode_position)
            if token in seen_tokens:
                raise ValueError("audit seed-token collision")
            seen_tokens.add(token)
            feature_set = extract_shortcut_features(example, source.episode_count)
            features[index] = feature_set.vectors[ShortcutFeatureGroup.COMBINED]
            bundle = example.bundle
            digest.add(
                CorpusDigestEntry(bundle.public.init.episode_public_id, episode_sha256(bundle))
            )
            rows.append(
                _StoredExample(
                    bundle.public.init.episode_public_id,
                    episode_sha256(bundle),
                    feature_set.audit_group_id,
                    bundle.truth.key.suite,
                    bundle.truth.recipe.requested_path_length,
                    bundle.truth.recipe.variant,
                    bundle.truth.relevant_hazard_type,
                    example.randomization_block_index,
                    example.episode_position,
                )
            )
            denominators[
                f"{bundle.truth.key.suite.value}:{bundle.truth.recipe.requested_path_length}"
            ] += 1
            if example.generation_mode == "matched":
                matched_groups[example.randomization_block_index].append(
                    (bundle.truth.recipe.variant, _matched_nuisance_signature(bundle))
                )
        if len(rows) != source.episode_count:
            raise ValueError("audit source yielded the wrong number of examples")
        if source.descriptor.generation_mode == "matched":
            expected = Counter(
                {
                    EpisodeVariant.POSITIVE: 2,
                    EpisodeVariant.SAFE_NEGATIVE: 1,
                    EpisodeVariant.DISCONNECTED_NEGATIVE: 1,
                }
            )
            for members in matched_groups.values():
                if (
                    len(members) != 4
                    or Counter(variant for variant, _signature in members) != expected
                ):
                    raise ValueError("matched audit group does not contain the declared cohort")
                if len({signature for _variant, signature in members}) != 1:
                    raise ValueError("matched audit cohort nuisance controls differ")
        corpus_hash = digest.finalize()
        # Regeneration authentication includes source order, public ID, and digest.
        second = tuple(
            (example.bundle.public.init.episode_public_id, episode_sha256(example.bundle))
            for example in source.iter_examples()
        )
        if second != tuple((row.public_id, row.digest) for row in rows):
            raise ValueError("audit source second pass differs from first pass")
        features.flush()
        train, test, split_hash = _split_memberships(
            rows,
            config.audit_seed,
            corpus_hash,
            source.descriptor.generation_mode,
            strict_divisible=profile is LeakageAuditProfileName.PHASE1_GATE,
        )
        public_values = np.asarray(features)
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
            control_probes = _run_probes(
                public_values, rows, train, test, config, selected_profile, corpus_hash
            )
            expected = active_injector.expected_detector_id
            observed = tuple(
                f"{probe.task.value}:{probe.feature_group.value}"
                for probe in control_probes
                if probe.balanced_accuracy >= config.positive_control_min_balanced_accuracy
                and probe.raw_permutation_p <= 0.05
            )
            matched = next(
                (
                    probe
                    for probe in control_probes
                    if f"{probe.task.value}:{probe.feature_group.value}" == expected
                ),
                None,
            )
            if matched is None:
                raise ValueError("positive-control detector family is incomplete")
            full_gate = profile is LeakageAuditProfileName.PHASE1_GATE
            control_passed = (
                expected in observed
                and matched.balanced_accuracy >= config.positive_control_min_balanced_accuracy
                and matched.raw_permutation_p <= 0.05
                and (not full_gate or matched.holm_adjusted_p < config.alpha)
            )
            controls = (
                PositiveControlResult(
                    control_id=active_injector.control_id,
                    target_task=active_injector.target_task,
                    expected_detector_id=expected,
                    observed_detector_ids=observed,
                    base_subset_corpus_sha256=corpus_hash,
                    injected_corpus_sha256=sha256_bytes(
                        canonical_json_bytes(
                            {
                                "domain": "silent-cascade/ofd-v1/positive-control/v1",
                                "base_subset_corpus_sha256": corpus_hash,
                                "control_id": active_injector.control_id,
                            }
                        )
                    ),
                    split_membership_sha256=split_hash,
                    balanced_accuracy=matched.balanced_accuracy,
                    holm_adjusted_p=matched.holm_adjusted_p,
                    passed=control_passed,
                ),
            )
        counterfactual = _counterfactual_checks(source)
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
                clean_pass
                and all(probe.passed for probe in shuffled_probes)
                and all(item.passed for item in counterfactual)
                and all(item.passed for item in controls)
            ),
        )
    finally:
        with suppress(UnboundLocalError):
            del features
        shutil.rmtree(work, ignore_errors=True)
