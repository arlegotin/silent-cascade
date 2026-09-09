"""Strict private validation archives and deterministic scripted CPU replay.

The registry is a closed v1 mapping. Archives carry private scoring truth and
must never enter online traces or agent callbacks. CPU replay compares exact
canonical hashes; a timing diagnostic tolerance does not relax hash equality.
"""

import json
import math
import os
import stat
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, ValidationError, ValidationInfo, field_validator

from silent_cascade.env.episode import EpisodeArtifact, EpisodeBundle
from silent_cascade.env.reward import EpisodeScore
from silent_cascade.errors import AtomicWriteError, ReplayError, SilentCascadeError
from silent_cascade.eventflow.config import EventFlowConfig
from silent_cascade.eventflow.engine import EpisodeResult, EventEngine
from silent_cascade.eventflow.scheduling import TieResolution
from silent_cascade.eventflow.scripted import ScriptedEventFlowAgent
from silent_cascade.eventflow.state import ComputeCounters
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.io import atomic_create_bytes
from silent_cascade.logging.trace import CausalEventSummary, CausalTrace, SegmentSummary
from silent_cascade.schemas import Action, Condition, ExternalPayload, Hypothesis, Mode
from silent_cascade.validation import StrictModel

MAX_REPLAY_BYTES = 32 * 1024 * 1024
CPU_TIMESTAMP_TOLERANCE = 1.0e-9
_SOURCE = {
    "fact": "external",
    "activate": "external",
    "recall": "internal",
    "compose": "internal",
    "act": "internal",
    "terminal": "terminal",
}


def _mismatch(field: str) -> ReplayError:
    return ReplayError(f"replay mismatch: {field}", context={"field": field})


class CausalEventArtifact(StrictModel):
    kind: Literal["fact", "activate", "recall", "compose", "act", "terminal"]
    source: Literal["external", "internal", "terminal"]
    event_id: int
    parent_event_id: int | None
    timestamp: float
    prediction_snapshot_sha256: str
    selected_record_id: int | None
    selected_rank: int | None
    hypothesis_before: Hypothesis | None
    hypothesis_after: Hypothesis | None
    support_before: tuple[int, ...]
    support_after: tuple[int, ...]
    actions: tuple[Action, ...]
    state_sha256: str
    payload: ExternalPayload
    pre_mode: Mode
    post_mode: Mode
    delta: float
    state_norm: float
    counter_delta: ComputeCounters
    counters_after: ComputeCounters
    segment: SegmentSummary | None
    tie: TieResolution | None
    tie_id: str | None
    was_gap_clamped: bool
    raw_predicted_delta: float | None

    @field_validator("source")
    @classmethod
    def consistent_source(cls, value: str, info: ValidationInfo) -> str:
        if "kind" in info.data and value != _SOURCE[info.data["kind"]]:
            raise ValueError("source is inconsistent with kind")
        return value

    @classmethod
    def from_summary(cls, summary: CausalEventSummary) -> Self:
        payload = summary.to_payload()
        payload["source"] = _SOURCE[summary.kind]
        return cls.model_validate_json(canonical_json_bytes(payload))


class CausalTraceArtifact(StrictModel):
    schema_version: Literal[1] = 1
    events: tuple[CausalEventArtifact, ...] = Field(max_length=130)
    sha256: str

    @classmethod
    def from_trace(cls, trace: CausalTrace) -> Self:
        return cls(
            events=tuple(CausalEventArtifact.from_summary(row) for row in trace.events),
            sha256=trace.sha256,
        )


class EpisodeResultArtifact(StrictModel):
    """Public result projection with causal counters only, like the online trace."""

    public_id: str
    score: EpisodeScore
    actions: tuple[Action, ...]
    counters: ComputeCounters

    @field_validator("counters")
    @classmethod
    def require_causal_counters(cls, value: ComputeCounters) -> ComputeCounters:
        if value.checkpoint_flow_evaluations != 0:
            raise ValueError("checkpoint_flow_evaluations must be zero in causal replay")
        return value

    @classmethod
    def from_result(cls, result: EpisodeResult) -> Self:
        return cls(
            public_id=result.public_id,
            score=result.score,
            actions=result.actions,
            counters=replace(result.counters, checkpoint_flow_evaluations=0),
        )


class ReplayArtifact(StrictModel):
    schema_version: Literal["phase2-replay-v1"]
    access_class: Literal["validation_private"]
    episode: EpisodeArtifact
    config_canonical_json: str
    config_sha256: str
    condition: Literal[Condition.EVENT_FLOW]
    agent_implementation: Literal["scripted-event-flow-v1"]
    expected_result: EpisodeResultArtifact
    trace: CausalTraceArtifact
    payload_sha256: str


class ReplayComparison(StrictModel):
    schema_version: Literal["phase2-replay-comparison-v1"] = "phase2-replay-comparison-v1"
    matched: Literal[True] = True
    event_count: int
    trace_sha256: str
    payload_sha256: str

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self))


def _payload_hash(payload: dict) -> str:
    return sha256_bytes(
        canonical_json_bytes({k: v for k, v in payload.items() if k != "payload_sha256"})
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise _mismatch("duplicate_json_key")
        result[key] = value
    return result


def _parse_artifact(raw: bytes) -> ReplayArtifact:
    if len(raw) > MAX_REPLAY_BYTES:
        raise _mismatch("archive.byte_limit")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(payload, dict):
            raise _mismatch("archive.envelope")
        if payload.get("payload_sha256") != _payload_hash(payload):
            raise _mismatch("payload_sha256")
        artifact = ReplayArtifact.model_validate_json(raw)
        if raw != canonical_json_bytes(artifact):
            raise _mismatch("archive.canonical_json")
        _validated_inputs(artifact)
        return artifact
    except ValidationError as error:
        location = ".".join(str(part) for part in error.errors()[0]["loc"])
        raise _mismatch(location or "archive.schema") from error
    except SilentCascadeError as error:
        if isinstance(error, ReplayError):
            raise
        raise _mismatch("archive.schema") from error
    except (ValueError, TypeError, KeyError, RecursionError) as error:
        raise _mismatch("archive.json") from error


def _validated_inputs(artifact: ReplayArtifact) -> tuple[EpisodeBundle, EventFlowConfig]:
    encoded = artifact.config_canonical_json.encode("utf-8")
    if sha256_bytes(encoded) != artifact.config_sha256:
        raise _mismatch("config_sha256")
    try:
        config = EventFlowConfig.model_validate_json(encoded)
        if canonical_json_bytes(config) != encoded:
            raise _mismatch("config_canonical_json")
    except ValidationError as error:
        raise _mismatch("config_canonical_json") from error
    try:
        bundle = artifact.episode.to_bundle()
    except (SilentCascadeError, ValueError, TypeError) as error:
        raise _mismatch("episode") from error
    if artifact.expected_result.public_id != bundle.public.init.episode_public_id:
        raise _mismatch("expected_result.public_id")
    expected_trace_payload = {
        "schema_version": artifact.trace.schema_version,
        "events": [
            row.model_dump(mode="json", exclude={"source"}) for row in artifact.trace.events
        ],
    }
    if sha256_bytes(canonical_json_bytes(expected_trace_payload)) != artifact.trace.sha256:
        raise _mismatch("trace.sha256")
    return bundle, config


@contextmanager
def _parent_descriptor(path: Path):
    """Pin each directory without following symlinks, including intermediate ones."""
    absolute = path.absolute()
    if ".." in absolute.parts or not absolute.name:
        raise _mismatch("archive.path")
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in absolute.parts[1:-1]:
            next_descriptor = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        yield descriptor, absolute.name
    finally:
        os.close(descriptor)


def _read_at(parent: int, name: str) -> bytes:
    descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise _mismatch("archive.regular_file")
        if info.st_size > MAX_REPLAY_BYTES:
            raise _mismatch("archive.byte_limit")
        chunks, remaining = [], MAX_REPLAY_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            remaining -= len(chunk)
        raise _mismatch("archive.byte_limit")
    finally:
        os.close(descriptor)


def load_replay_artifact(path: Path) -> ReplayArtifact:
    """Validate one bounded, canonical regular-file snapshot before any rerun."""
    try:
        with _parent_descriptor(path) as (parent, name):
            raw = _read_at(parent, name)
        return _parse_artifact(raw)
    except OSError as error:
        raise _mismatch("archive.path_or_file") from error


def write_replay_artifact(
    path: Path,
    *,
    bundle: EpisodeBundle,
    config: EventFlowConfig,
    result: EpisodeResult,
) -> ReplayArtifact:
    """Publish a private v1 artifact atomically, accepting only identical existing bytes."""
    config_bytes = canonical_json_bytes(config)
    artifact = ReplayArtifact(
        schema_version="phase2-replay-v1",
        access_class="validation_private",
        episode=EpisodeArtifact.from_bundle(bundle),
        config_canonical_json=config_bytes.decode("utf-8"),
        config_sha256=sha256_bytes(config_bytes),
        condition=Condition.EVENT_FLOW,
        agent_implementation="scripted-event-flow-v1",
        expected_result=EpisodeResultArtifact.from_result(result),
        trace=CausalTraceArtifact.from_trace(result.trace),
        payload_sha256="",
    )
    payload = artifact.model_dump(mode="json")
    payload["payload_sha256"] = _payload_hash(payload)
    raw = canonical_json_bytes(payload)
    artifact = _parse_artifact(raw)
    try:
        with _parent_descriptor(path) as (parent, name):
            # macOS cannot traverse a directory via /dev/fd. Validate parents
            # here, publish using the shared no-clobber writer, and read any
            # existing target through the pinned descriptor. Hostile concurrent
            # ancestor replacement is outside this local publication boundary.
            try:
                atomic_create_bytes(path, raw, mode=0o600)
            except AtomicWriteError as error:
                if not isinstance(error.__cause__, FileExistsError):
                    raise _mismatch("archive.publication") from error
                if _read_at(parent, name) != raw:
                    raise _mismatch("archive.existing_divergent") from error
    except OSError as error:
        raise _mismatch("archive.path_or_file") from error
    return artifact


def _compare(expected: object, actual: object, path: str) -> None:
    """Report the first differing field in deterministic semantic field order."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if expected.keys() != actual.keys():
            raise _mismatch(path)
        for key in actual:
            _compare(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise _mismatch(path)
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare(left, right, f"{path}.{index}")
    elif path.endswith(".timestamp") and type(expected) is type(actual) is float:
        if not math.isclose(expected, actual, rel_tol=0.0, abs_tol=CPU_TIMESTAMP_TOLERANCE):
            raise _mismatch(path)
    elif type(expected) is not type(actual) or expected != actual:
        raise _mismatch(path)


def verify_replay(artifact: ReplayArtifact) -> ReplayComparison:
    """Rerun on CPU, stopping immediately at the first differing causal boundary."""
    # Revalidate even a model_copy/model_construct object; those APIs bypass validators.
    artifact = _parse_artifact(canonical_json_bytes(artifact))
    bundle, config = _validated_inputs(artifact)
    factory = {"scripted-event-flow-v1": ScriptedEventFlowAgent}[artifact.agent_implementation]
    engine = EventEngine(config)
    agent = factory(device="cpu")
    try:
        session = engine.start_episode(bundle, agent)
        index = 0
        while True:
            if index >= len(artifact.trace.events):
                raise _mismatch("trace.events.length")
            terminal = engine.step(session, agent)
            actual = CausalEventArtifact.from_summary(session.trace.snapshot().events[-1])
            expected_row = artifact.trace.events[index].model_dump(mode="json")
            actual_row = actual.model_dump(mode="json")
            _compare(
                expected_row,
                actual_row,
                f"trace.events.{index}",
            )
            # Diagnostic timing tolerance does not permit continuing past an
            # exact CPU row mismatch, even in a coherently rehashed archive.
            if canonical_json_bytes(expected_row) != canonical_json_bytes(actual_row):
                raise _mismatch(f"trace.events.{index}")
            index += 1
            if terminal:
                break
        if index != len(artifact.trace.events):
            raise _mismatch("trace.events.length")
        result = EpisodeResultArtifact(
            public_id=session.public_id,
            score=session.terminal_score,
            actions=session.state.core.actions,
            counters=session.state.core.counters,
        )
        _compare(
            artifact.expected_result.model_dump(mode="json"),
            result.model_dump(mode="json"),
            "expected_result",
        )
        actual_hash = session.trace.snapshot().sha256
        if artifact.trace.sha256 != actual_hash:
            raise _mismatch("trace.sha256")
    except SilentCascadeError as error:
        if isinstance(error, ReplayError):
            raise
        raise _mismatch("runtime") from error
    return ReplayComparison(
        event_count=index,
        trace_sha256=session.trace.snapshot().sha256,
        payload_sha256=artifact.payload_sha256,
    )
