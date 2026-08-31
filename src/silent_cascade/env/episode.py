"""Private episode bundles and canonical OFD serialization."""

import hashlib
import math
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from enum import StrEnum
from typing import Literal, Self

from pydantic import field_validator

from silent_cascade.env.config import OracleTimingConfig, SplitNamespace, SuiteName
from silent_cascade.env.timing import action_window
from silent_cascade.errors import EpisodeInvariantError
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.schemas import AgentInit, ExternalEvent, ExternalEventKind, HazardFact
from silent_cascade.validation import StrictModel

_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_TIME_TOLERANCE = 1.0e-9


def _require_int(value: object, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact integer")
    if value < minimum:
        raise ValueError(f"{name} is out of bounds")


def _require_float(value: object, name: str, *, positive: bool = False) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    if not math.isfinite(value) or (positive and value <= 0.0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))


def _times_equal(left: float, right: float) -> bool:
    return abs(left - right) <= _TIME_TOLERANCE


class EpisodeVariant(StrEnum):
    POSITIVE = "positive"
    SAFE_NEGATIVE = "safe_negative"
    DISCONNECTED_NEGATIVE = "disconnected_negative"


@dataclass(frozen=True, slots=True)
class PublicEpisode:
    init: AgentInit
    events: tuple[ExternalEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.init, AgentInit):
            raise TypeError("init must be an AgentInit")
        if not isinstance(self.events, tuple) or not self.events:
            raise ValueError("events must be a nonempty tuple")
        ids: set[int] = set()
        activation_count = 0
        previous_time = self.init.initial_time
        for event in self.events:
            if not isinstance(event, ExternalEvent):
                raise TypeError("events must contain ExternalEvent values")
            if event.event_id in ids:
                raise EpisodeInvariantError("public event IDs must be unique")
            if event.timestamp < previous_time:
                raise EpisodeInvariantError("public events must be chronological")
            if event.kind not in (ExternalEventKind.FACT, ExternalEventKind.ACTIVATE):
                raise EpisodeInvariantError("private terminal events cannot be public")
            ids.add(event.event_id)
            activation_count += event.kind is ExternalEventKind.ACTIVATE
            previous_time = event.timestamp
        if activation_count != 1 or self.events[-1].kind is not ExternalEventKind.ACTIVATE:
            raise EpisodeInvariantError("public events require one final activation")


@dataclass(frozen=True, slots=True)
class MatchedEpisodeCoordinate:
    mode: Literal["matched"]
    cohort_index: int
    member_index: int

    def __post_init__(self) -> None:
        if self.mode != "matched":
            raise ValueError("matched coordinates require mode='matched'")
        _require_int(self.cohort_index, "cohort_index")
        _require_int(self.member_index, "member_index")


@dataclass(frozen=True, slots=True)
class IndependentEpisodeCoordinate:
    mode: Literal["independent"]
    episode_index: int
    allocation_quartet_index: int

    def __post_init__(self) -> None:
        if self.mode != "independent":
            raise ValueError("independent coordinates require mode='independent'")
        _require_int(self.episode_index, "episode_index")
        _require_int(self.allocation_quartet_index, "allocation_quartet_index")


type EpisodeCoordinate = MatchedEpisodeCoordinate | IndependentEpisodeCoordinate


@dataclass(frozen=True, slots=True)
class EpisodeKey:
    generator_version: str
    split_namespace: SplitNamespace
    suite: SuiteName
    root_seed: int
    coordinate: EpisodeCoordinate

    def __post_init__(self) -> None:
        if self.generator_version != "ofd-v1":
            raise ValueError("generator_version must be ofd-v1")
        if not isinstance(self.split_namespace, SplitNamespace) or not isinstance(
            self.suite, SuiteName
        ):
            raise TypeError("split_namespace and suite must be strict enums")
        _require_int(self.root_seed, "root_seed")
        if not isinstance(
            self.coordinate, (MatchedEpisodeCoordinate, IndependentEpisodeCoordinate)
        ):
            raise TypeError("coordinate must be a discriminated episode coordinate")


@dataclass(frozen=True, slots=True)
class EpisodeRecipe:
    requested_path_length: int
    variant: EpisodeVariant
    distractor_link_count: int
    evaluation_suite: SuiteName
    accepted_attempt: int
    clock_scale: float = 1.0
    parent_public_id: str | None = None
    parent_episode_sha256: str | None = None
    oracle_timing: OracleTimingConfig = dataclass_field(default_factory=OracleTimingConfig)

    def __post_init__(self) -> None:
        _require_int(self.requested_path_length, "requested_path_length", minimum=1)
        if not isinstance(self.variant, EpisodeVariant) or not isinstance(
            self.evaluation_suite, SuiteName
        ):
            raise TypeError("variant and evaluation_suite must be strict enums")
        if not isinstance(self.oracle_timing, OracleTimingConfig):
            raise TypeError("oracle_timing must be an OracleTimingConfig")
        _require_int(self.distractor_link_count, "distractor_link_count")
        _require_int(self.accepted_attempt, "accepted_attempt")
        _require_float(self.clock_scale, "clock_scale", positive=True)
        parent_values = (self.parent_public_id, self.parent_episode_sha256)
        if (parent_values[0] is None) != (parent_values[1] is None):
            raise ValueError("parent public ID and hash must be supplied together")
        if self.parent_public_id is not None:
            _validate_public_id(self.parent_public_id)
            _validate_digest(self.parent_episode_sha256)
        clock_factors = {
            SuiteName.CLOCK_SCALE_0_1X: 0.1,
            SuiteName.CLOCK_SCALE_10X: 10.0,
        }
        expected_factor = clock_factors.get(self.evaluation_suite)
        if expected_factor is None:
            if self.clock_scale != 1.0 or self.parent_public_id is not None:
                raise ValueError("only paired clock suites may carry clock transform provenance")
        elif self.clock_scale != expected_factor or self.parent_public_id is None:
            raise ValueError("paired clock suite must carry its exact factor and parent provenance")


@dataclass(frozen=True, slots=True)
class StressMetadata:
    over_capacity_record_count: int | None = None
    near_miss_missing_edges: int | None = None
    near_miss_hazard_node: int | None = None
    proposed_checkpoint_pause_time: float | None = None
    minimum_feasible_delay: float | None = None

    def __post_init__(self) -> None:
        if self.over_capacity_record_count is not None:
            _require_int(self.over_capacity_record_count, "over_capacity_record_count", minimum=65)
        if self.near_miss_missing_edges is not None:
            _require_int(self.near_miss_missing_edges, "near_miss_missing_edges", minimum=1)
        if self.near_miss_hazard_node is not None:
            _require_int(self.near_miss_hazard_node, "near_miss_hazard_node")
            if self.near_miss_hazard_node > 63:
                raise ValueError("near_miss_hazard_node is out of bounds")
        if self.proposed_checkpoint_pause_time is not None:
            _require_float(self.proposed_checkpoint_pause_time, "proposed_checkpoint_pause_time")
        if self.minimum_feasible_delay is not None:
            _require_float(self.minimum_feasible_delay, "minimum_feasible_delay", positive=True)


@dataclass(frozen=True, slots=True)
class EpisodeTruth:
    key: EpisodeKey
    recipe: EpisodeRecipe
    relevant_node_path: tuple[int, ...]
    relevant_record_ids: tuple[int, ...]
    terminal_record_id: int | None
    relevant_hazard_type: int | None
    private_terminal: ExternalEvent
    activation_time: float
    episode_delay: float
    action_window_start: float | None
    action_window_end: float | None
    action_target: float | None
    rejection_count: int
    rejection_reasons: tuple[str, ...]
    stress_metadata: StressMetadata | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, EpisodeKey) or not isinstance(self.recipe, EpisodeRecipe):
            raise TypeError("key and recipe must be episode values")
        if self.recipe.parent_public_id is None:
            if self.key.suite is not self.recipe.evaluation_suite:
                raise EpisodeInvariantError("recipe suite must match episode key")
        elif self.key.suite is not SuiteName.IID_PRIMARY or self.recipe.evaluation_suite not in (
            SuiteName.CLOCK_SCALE_0_1X,
            SuiteName.CLOCK_SCALE_10X,
        ):
            raise EpisodeInvariantError("paired clock recipe must retain an IID source key")
        if not isinstance(self.relevant_node_path, tuple) or not self.relevant_node_path:
            raise ValueError("relevant_node_path must be a nonempty tuple")
        if len(self.relevant_node_path) != self.recipe.requested_path_length + 1:
            raise EpisodeInvariantError("path length must match recipe")
        for node in self.relevant_node_path:
            _require_int(node, "relevant_node_path")
        _require_record_ids(self.relevant_record_ids, "relevant_record_ids")
        if self.terminal_record_id is not None:
            _require_int(self.terminal_record_id, "terminal_record_id")
        if self.relevant_hazard_type is not None:
            _require_int(self.relevant_hazard_type, "relevant_hazard_type", minimum=0)
            if self.relevant_hazard_type > 3:
                raise ValueError("relevant_hazard_type is out of bounds")
        if not isinstance(
            self.private_terminal, ExternalEvent
        ) or self.private_terminal.kind not in (
            ExternalEventKind.OUTCOME,
            ExternalEventKind.END,
        ):
            raise EpisodeInvariantError("private terminal must be OUTCOME or END")
        _require_float(self.activation_time, "activation_time")
        _require_float(self.episode_delay, "episode_delay", positive=True)
        if not _times_equal(
            self.private_terminal.timestamp, self.activation_time + self.episode_delay
        ):
            raise EpisodeInvariantError("private terminal timestamp must match delay")
        _require_int(self.rejection_count, "rejection_count")
        if not isinstance(self.rejection_reasons, tuple) or not all(
            isinstance(reason, str) for reason in self.rejection_reasons
        ):
            raise TypeError("rejection_reasons must be a tuple of strings")
        self._validate_variant_fields()
        self._validate_stress_metadata()

    def _validate_variant_fields(self) -> None:
        action_values = (self.action_window_start, self.action_window_end, self.action_target)
        if self.recipe.variant is EpisodeVariant.POSITIVE:
            if self.private_terminal.kind is not ExternalEventKind.OUTCOME:
                raise EpisodeInvariantError("positive episodes require an OUTCOME terminal")
            if self.terminal_record_id is None or self.relevant_hazard_type is None:
                raise EpisodeInvariantError("positive episodes require terminal hazard truth")
            if any(value is None for value in action_values):
                raise EpisodeInvariantError("positive episodes require an action window")
            start, end, target = action_values
            assert start is not None and end is not None and target is not None
            _require_float(start, "action_window_start")
            _require_float(end, "action_window_end")
            _require_float(target, "action_target")
            expected = action_window(
                self.activation_time,
                self.episode_delay,
                self.recipe.oracle_timing,
            )
            if not (
                _times_equal(start, expected.start)
                and _times_equal(end, expected.end)
                and _times_equal(target, expected.target)
            ):
                raise EpisodeInvariantError("positive action window must match OFD timing")
        else:
            if self.private_terminal.kind is not ExternalEventKind.END:
                raise EpisodeInvariantError("negative episodes require an END terminal")
            if (
                any(value is not None for value in action_values)
                or self.relevant_hazard_type is not None
            ):
                raise EpisodeInvariantError("negative episodes expose no action or hazard truth")

    def _validate_stress_metadata(self) -> None:
        suite = self.key.suite
        special = {
            SuiteName.MEMORY_OVERFLOW_STRESS,
            SuiteName.NULL_NEAR_MISS_STRESS,
            SuiteName.CHECKPOINT_STRESS,
            SuiteName.MINIMUM_DURATION_STRESS,
        }
        if suite not in special:
            if self.stress_metadata is not None:
                raise EpisodeInvariantError(
                    "stress metadata is only valid for its matching stress suite"
                )
            return
        if not isinstance(self.stress_metadata, StressMetadata):
            raise EpisodeInvariantError("matching stress suites require stress metadata")
        metadata = self.stress_metadata
        if suite is SuiteName.MEMORY_OVERFLOW_STRESS:
            if metadata.over_capacity_record_count is None or any(
                value is not None
                for value in (
                    metadata.near_miss_missing_edges,
                    metadata.near_miss_hazard_node,
                    metadata.proposed_checkpoint_pause_time,
                    metadata.minimum_feasible_delay,
                )
            ):
                raise EpisodeInvariantError(
                    "memory overflow metadata must contain only record count"
                )
        elif suite is SuiteName.NULL_NEAR_MISS_STRESS:
            if (
                metadata.near_miss_missing_edges is None
                or metadata.near_miss_hazard_node is None
                or any(
                    value is not None
                    for value in (
                        metadata.over_capacity_record_count,
                        metadata.proposed_checkpoint_pause_time,
                        metadata.minimum_feasible_delay,
                    )
                )
            ):
                raise EpisodeInvariantError(
                    "null near-miss metadata must contain only missing-edge provenance"
                )
        elif suite is SuiteName.CHECKPOINT_STRESS:
            pause = metadata.proposed_checkpoint_pause_time
            if (
                pause is None
                or not self.activation_time < pause < self.private_terminal.timestamp
                or any(
                    value is not None
                    for value in (
                        metadata.over_capacity_record_count,
                        metadata.near_miss_missing_edges,
                        metadata.near_miss_hazard_node,
                        metadata.minimum_feasible_delay,
                    )
                )
            ):
                raise EpisodeInvariantError(
                    "checkpoint metadata must contain only an in-window pause"
                )
        elif metadata.minimum_feasible_delay is None or any(
            value is not None
            for value in (
                metadata.over_capacity_record_count,
                metadata.near_miss_missing_edges,
                metadata.near_miss_hazard_node,
                metadata.proposed_checkpoint_pause_time,
            )
        ):
            raise EpisodeInvariantError(
                "minimum-duration metadata must contain only feasible delay"
            )


@dataclass(frozen=True, slots=True)
class EpisodeBundle:
    public: PublicEpisode
    truth: EpisodeTruth

    def __post_init__(self) -> None:
        if not isinstance(self.public, PublicEpisode) or not isinstance(self.truth, EpisodeTruth):
            raise TypeError("bundle requires public episode and private truth")
        activation = self.public.events[-1]
        if activation.timestamp != self.truth.activation_time:
            raise EpisodeInvariantError("truth activation time must match public activation")
        public_ids = {event.event_id for event in self.public.events}
        if self.truth.private_terminal.event_id in public_ids:
            raise EpisodeInvariantError("private terminal ID must not be public")
        facts_by_id = {
            event.event_id: event
            for event in self.public.events
            if event.kind is ExternalEventKind.FACT
        }
        fact_ids = set(facts_by_id)
        if not set(self.truth.relevant_record_ids).issubset(fact_ids):
            raise EpisodeInvariantError("relevant record IDs must refer to FACT events")
        if (
            self.truth.terminal_record_id is not None
            and self.truth.terminal_record_id not in fact_ids
        ):
            raise EpisodeInvariantError("terminal record ID must refer to a FACT event")
        if self.truth.recipe.variant is EpisodeVariant.POSITIVE:
            self._validate_positive_terminal(facts_by_id)

    def _validate_positive_terminal(self, facts_by_id: dict[int, ExternalEvent]) -> None:
        terminal_id = self.truth.terminal_record_id
        hazard_type = self.truth.relevant_hazard_type
        assert terminal_id is not None and hazard_type is not None
        if terminal_id not in self.truth.relevant_record_ids:
            raise EpisodeInvariantError("positive terminal hazard must be a relevant record")
        payload = facts_by_id[terminal_id].payload
        if not isinstance(payload, HazardFact):
            raise EpisodeInvariantError("positive terminal hazard must select a HazardFact")
        if (
            payload.node != self.truth.relevant_node_path[-1]
            or payload.hazard_type != hazard_type
            or not _times_equal(payload.delay, self.truth.episode_delay)
        ):
            raise EpisodeInvariantError(
                "positive terminal hazard must match node, type, and delay truth"
            )


def public_projection(bundle: EpisodeBundle) -> PublicEpisode:
    """Return the agent-visible episode without private scorer truth."""
    if not isinstance(bundle, EpisodeBundle):
        raise TypeError("public projection requires an EpisodeBundle")
    return bundle.public


def _tuple_from_list(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


class PublicEpisodeArtifact(StrictModel):
    schema_version: Literal[1] = 1
    generator_version: Literal["ofd-v1"] = "ofd-v1"
    init: AgentInit
    events: tuple[ExternalEvent, ...]

    @field_validator("events", mode="before")
    @classmethod
    def list_events_are_tuples(cls, value: object) -> object:
        return _tuple_from_list(value)

    @classmethod
    def from_public(cls, public: PublicEpisode) -> Self:
        return cls(init=public.init, events=public.events)

    def to_public(self) -> PublicEpisode:
        return PublicEpisode(init=self.init, events=self.events)


class EpisodeArtifact(StrictModel):
    schema_version: Literal[1] = 1
    generator_version: Literal["ofd-v1"] = "ofd-v1"
    public: PublicEpisodeArtifact
    truth: EpisodeTruth

    @field_validator("truth", mode="before")
    @classmethod
    def private_lists_are_tuples(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for field in ("relevant_node_path", "relevant_record_ids", "rejection_reasons"):
            normalized[field] = _tuple_from_list(normalized.get(field))
        return normalized

    @classmethod
    def from_bundle(cls, bundle: EpisodeBundle) -> Self:
        return cls(public=PublicEpisodeArtifact.from_public(bundle.public), truth=bundle.truth)

    def to_bundle(self) -> EpisodeBundle:
        return EpisodeBundle(public=self.public.to_public(), truth=self.truth)


def canonical_episode_bytes(bundle: EpisodeBundle) -> bytes:
    return canonical_json_bytes(EpisodeArtifact.from_bundle(bundle))


def episode_sha256(bundle: EpisodeBundle) -> str:
    return sha256_bytes(canonical_episode_bytes(bundle))


_CLOCK_FACTORS = {
    SuiteName.CLOCK_SCALE_0_1X: 0.1,
    SuiteName.CLOCK_SCALE_10X: 10.0,
}


def _scaled_time(value: float, factor: float) -> float:
    scaled = value * factor
    if not math.isfinite(scaled):
        raise EpisodeInvariantError("scaled episode contains nonfinite time values")
    return scaled


def _scale_optional_time(value: float | None, factor: float) -> float | None:
    return None if value is None else _scaled_time(value, factor)


def _scale_timing_provenance(timing: OracleTimingConfig, factor: float) -> OracleTimingConfig:
    return OracleTimingConfig(
        delta_0=_scaled_time(timing.delta_0, factor),
        delta_min=_scaled_time(timing.delta_min, factor),
        delta_max=_scaled_time(timing.delta_max, factor),
        jitter_log_std=timing.jitter_log_std,
        terminal_compose_fraction=timing.terminal_compose_fraction,
        action_window_start_fraction=timing.action_window_start_fraction,
        action_target_fraction=timing.action_target_fraction,
        action_window_end_fraction=timing.action_window_end_fraction,
    )


def _scale_fact_delay(event: ExternalEvent, factor: float) -> ExternalEvent:
    payload = event.payload
    if isinstance(payload, HazardFact):
        payload = replace(payload, delay=_scaled_time(payload.delay, factor))
    return replace(event, timestamp=_scaled_time(event.timestamp, factor), payload=payload)


def scale_episode_time(
    bundle: EpisodeBundle,
    target_suite: SuiteName,
    paired_public_id: str,
) -> EpisodeBundle:
    """Create an immutable paired clock child from one unscaled IID source."""
    if not isinstance(bundle, EpisodeBundle):
        raise TypeError("bundle must be an EpisodeBundle")
    if not isinstance(target_suite, SuiteName):
        raise TypeError("target_suite must be a clock scaling SuiteName")
    try:
        factor = _CLOCK_FACTORS[target_suite]
    except KeyError as error:
        raise ValueError("target_suite must be a clock scaling suite") from error
    _validate_public_id(paired_public_id)

    recipe = bundle.truth.recipe
    if (
        bundle.truth.key.suite is not SuiteName.IID_PRIMARY
        or recipe.evaluation_suite is not SuiteName.IID_PRIMARY
        or recipe.clock_scale != 1.0
        or recipe.parent_public_id is not None
    ):
        raise EpisodeInvariantError("clock transform requires an unscaled IID parent")
    if paired_public_id == bundle.public.init.episode_public_id:
        raise EpisodeInvariantError("paired clock child must have a new public ID")

    scaled_events = tuple(_scale_fact_delay(event, factor) for event in bundle.public.events)
    scaled_terminal = replace(
        bundle.truth.private_terminal,
        timestamp=_scaled_time(bundle.truth.private_terminal.timestamp, factor),
    )
    scaled_recipe = replace(
        recipe,
        evaluation_suite=target_suite,
        clock_scale=factor,
        parent_public_id=bundle.public.init.episode_public_id,
        parent_episode_sha256=episode_sha256(bundle),
        oracle_timing=_scale_timing_provenance(recipe.oracle_timing, factor),
    )
    scaled_truth = replace(
        bundle.truth,
        recipe=scaled_recipe,
        private_terminal=scaled_terminal,
        activation_time=_scaled_time(bundle.truth.activation_time, factor),
        episode_delay=_scaled_time(bundle.truth.episode_delay, factor),
        action_window_start=_scale_optional_time(bundle.truth.action_window_start, factor),
        action_window_end=_scale_optional_time(bundle.truth.action_window_end, factor),
        action_target=_scale_optional_time(bundle.truth.action_target, factor),
    )
    return EpisodeBundle(
        PublicEpisode(
            AgentInit(
                paired_public_id,
                bundle.public.init.memory_capacity,
                bundle.public.init.hazard_type_count,
                _scaled_time(bundle.public.init.initial_time, factor),
            ),
            scaled_events,
        ),
        scaled_truth,
    )


def episode_from_bytes(payload: bytes) -> EpisodeBundle:
    return EpisodeArtifact.model_validate_json(payload).to_bundle()


def _validate_public_id(value: object) -> None:
    if not isinstance(value, str):
        raise TypeError("episode public ID must be a string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as error:
        raise ValueError("episode public ID must be a canonical UUID v4") from error
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError("episode public ID must be a canonical UUID v4")


def _validate_digest(value: object) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError("episode SHA-256 must be lowercase 64-hex")


def _require_record_ids(value: object, name: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must be unique")
    for item in value:
        _require_int(item, name)


@dataclass(frozen=True, slots=True)
class CorpusDigestEntry:
    episode_public_id: str
    episode_sha256: str

    def __post_init__(self) -> None:
        _validate_public_id(self.episode_public_id)
        _validate_digest(self.episode_sha256)


class CorpusHashBuilder:
    """Incrementally calculate a corpus digest in declared source order."""

    def __init__(self, expected_count: int) -> None:
        _require_int(expected_count, "expected_count", minimum=1)
        self._expected_count = expected_count
        self._digest = hashlib.sha256()
        self._digest.update(b"silent-cascade/corpus/v1\0")
        self._digest.update(expected_count.to_bytes(8, "big", signed=False))
        self._public_ids: set[str] = set()
        self._count = 0
        self._finalized = False

    def add(self, entry: CorpusDigestEntry) -> None:
        if self._finalized:
            raise RuntimeError("cannot add after finalization")
        if not isinstance(entry, CorpusDigestEntry):
            raise TypeError("entry must be a CorpusDigestEntry")
        if self._count >= self._expected_count:
            raise ValueError("too many corpus entries")
        if entry.episode_public_id in self._public_ids:
            raise ValueError("duplicate corpus public ID")
        public_id = entry.episode_public_id.encode("utf-8")
        self._digest.update(self._count.to_bytes(8, "big", signed=False))
        self._digest.update(len(public_id).to_bytes(4, "big", signed=False))
        self._digest.update(public_id)
        self._digest.update(bytes.fromhex(entry.episode_sha256))
        self._public_ids.add(entry.episode_public_id)
        self._count += 1

    def finalize(self) -> str:
        if self._finalized:
            raise RuntimeError("corpus hash has already been finalized")
        if self._count != self._expected_count:
            raise ValueError("wrong number of corpus entries")
        self._finalized = True
        return self._digest.hexdigest()


def corpus_sha256(entries: Iterable[CorpusDigestEntry], *, expected_count: int) -> str:
    builder = CorpusHashBuilder(expected_count)
    for entry in entries:
        builder.add(entry)
    return builder.finalize()
