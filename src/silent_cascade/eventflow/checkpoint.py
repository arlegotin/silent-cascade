"""Private immutable safetensors checkpoints for the closed scripted-v1 runtime.

Configuration identity covers EventFlowConfig, not the full experiment config.
CPU source continuation is exact; cross-device analytic rounding is not certified.
All untrusted bytes, tensor hashes and causal anchors are validated before globals
or a caller agent are changed. The archive is integrity checked, not authenticated.
"""

import copy
import json
import math
import os
import re
import stat
import struct
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

import torch
from pydantic import Field, ValidationError
from safetensors import SafetensorError
from safetensors.torch import load, save

from silent_cascade.env.episode import EpisodeTruth, PublicEpisode
from silent_cascade.env.reward import EpisodeScore
from silent_cascade.errors import AtomicWriteError, ReplayError, SilentCascadeError
from silent_cascade.eventflow.checkpoint_rng import (
    RngMetadata,
    decode_rng,
    encode_rng,
    restore_rng_snapshot,
    validate_rng_restore,
)
from silent_cascade.eventflow.config import EventFlowConfig
from silent_cascade.eventflow.engine import (
    EventEngine,
    RuntimeSession,
    _core_metadata,
    _public_state_anchor,
)
from silent_cascade.eventflow.flow import state_at
from silent_cascade.eventflow.guards import allowed_mode_mask
from silent_cascade.eventflow.invariants import validate_runtime_state, validate_session_boundary
from silent_cascade.eventflow.replay import CausalTraceArtifact
from silent_cascade.eventflow.scheduling import (
    ExternalEventQueue,
    PredictionCache,
    PredictionSnapshot,
    choose_next_event,
    normalize_internal_gap,
)
from silent_cascade.eventflow.scripted import ScriptedEventFlowAgent
from silent_cascade.eventflow.state import (
    INTERNAL_EVENT_ID_BASE,
    AnalyticSegment,
    ComputeCounters,
    ContinuousChannels,
    ContinuousState,
    RuntimeCore,
    RuntimeState,
    SegmentParameters,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.io import atomic_create_bytes
from silent_cascade.logging.trace import (
    CausalEventSummary,
    CausalTrace,
    SegmentSummary,
    TraceRecorder,
    Trajectory,
    tensor_sha256,
)
from silent_cascade.memory import BoundedMemory
from silent_cascade.rng import RngSnapshot, snapshot_global_rng
from silent_cascade.schemas import (
    Action,
    AgentInit,
    Condition,
    ExternalEvent,
    ExternalEventKind,
    Hypothesis,
    InternalEvent,
    InternalEventKind,
    Mode,
)
from silent_cascade.validation import StrictModel

MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024


class CoreMetadata(StrictModel):
    mode: Mode
    memory: BoundedMemory
    focus_node_id: int | None
    active_record_id: int | None
    active_record_rank: int | None
    support_ids: tuple[int, ...]
    hypothesis: Hypothesis | None
    activation_time: float | None
    actions: tuple[Action, ...]
    same_kind_refractory_until: tuple[float, float, float]
    executed_internal_events: int
    consecutive_gap_clamps: int
    last_event_id: int | None
    last_event_time: float | None
    counters: ComputeCounters


class StateMetadata(StrictModel):
    core: CoreMetadata
    time: float
    segment_started_at: float
    segment_parent_event_id: int
    prediction_snapshot_sha256: str


class TensorMetadata(StrictModel):
    dtype: Literal["torch.float32", "torch.uint8"]
    shape: tuple[int, ...]
    sha256: str


class RuntimeCheckpointMetadata(StrictModel):
    schema_version: Literal["phase2-runtime-checkpoint-v1"]
    engine_schema: Literal["event-engine-v1"]
    access_class: Literal["validation_private"]
    semantics: Literal["paused_world"]
    config_scope: Literal["EventFlowConfig"]
    config_canonical_json: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_revision: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    condition: Literal[Condition.EVENT_FLOW]
    agent_implementation: Literal["scripted-event-flow-v1"]
    checkpoint_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    original_device: Literal["cpu", "mps"]
    public_id: str
    truth: EpisodeTruth
    state: StateMetadata
    pause_cursor: float
    external_queue: tuple[ExternalEvent, ...] = Field(max_length=66)
    prediction_cache: PredictionSnapshot | None
    trace: CausalTraceArtifact
    trajectory: tuple[StateMetadata, ...] = Field(min_length=1, max_length=131)
    agent_counters: ComputeCounters
    terminal_score: EpisodeScore | None
    rng: RngMetadata
    tensors: dict[str, TensorMetadata]
    metadata_sha256: str


@dataclass(frozen=True, slots=True)
class RuntimeCheckpointArtifact:
    metadata: RuntimeCheckpointMetadata
    tensors: dict[str, torch.Tensor]
    path: Path | None = None
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeCheckpointStage:
    """Pre-callback clone; no archive encoding or filesystem work on success."""

    session: RuntimeSession
    agent_counters: ComputeCounters
    rng: RngSnapshot


def stage_runtime(session: RuntimeSession, agent: ScriptedEventFlowAgent) -> RuntimeCheckpointStage:
    if type(agent) is not ScriptedEventFlowAgent or session.failure is not None:
        raise ReplayError("checkpoint requires a valid registered scripted session")
    EventEngine._check_session(session)
    # State owns independent tensor/semantic storage. Trajectory already owns
    # inaccessible cloned anchors and immutable prefixes; share it while staging.
    staged = replace(
        session,
        state=copy.deepcopy(session.state),
        truth=copy.deepcopy(session.truth),
        external_queue=ExternalEventQueue.from_snapshot(
            copy.deepcopy(session.external_queue.snapshot())
        ),
        prediction_cache=PredictionCache.from_snapshot(
            copy.deepcopy(session.prediction_cache.snapshot())
        ),
        trace=TraceRecorder(copy.deepcopy(session.trace.snapshot())),
        failure=None,
    )
    return RuntimeCheckpointStage(staged, replace(agent.compute_counters()), snapshot_global_rng())


def _state_metadata(
    state: RuntimeState, prefix: str, tensors: dict[str, torch.Tensor]
) -> StateMetadata:
    for group, channels in (
        ("current", state.core.continuous),
        ("origin", state.segment.origin),
        ("targets", state.segment.parameters.flow_targets),
        ("rates", state.segment.parameters.flow_rates),
    ):
        for item in fields(channels):
            tensors[f"{prefix}.{group}.{item.name}"] = (
                getattr(channels, item.name).detach().cpu().contiguous().clone()
            )
    for name in ("guard_targets", "guard_rates"):
        tensors[f"{prefix}.{name}"] = (
            getattr(state.segment.parameters, name).detach().cpu().contiguous().clone()
        )
    return StateMetadata(
        core=CoreMetadata.model_validate_json(canonical_json_bytes(_core_metadata(state))),
        time=state.time,
        segment_started_at=state.segment.started_at,
        segment_parent_event_id=state.segment.parent_event_id,
        prediction_snapshot_sha256=state.segment.prediction_snapshot_sha256,
    )


def _tensor_metadata(name: str, tensor: torch.Tensor) -> TensorMetadata:
    return TensorMetadata(
        dtype=str(tensor.dtype),
        shape=tuple(tensor.shape),
        sha256=sha256_bytes(
            canonical_json_bytes({"name": name, "tensor_sha256": tensor_sha256(tensor)})
        ),
    )


def _metadata_hash(payload: dict) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {key: value for key, value in payload.items() if key != "metadata_sha256"}
        )
    )


def snapshot_runtime(
    session: RuntimeSession,
    agent: ScriptedEventFlowAgent,
    *,
    config: EventFlowConfig,
    source_revision: str,
    checkpoint_id: str | None = None,
) -> RuntimeCheckpointArtifact:
    """Capture the current simulated cursor without advancing or predicting."""
    return snapshot_staged_runtime(
        stage_runtime(session, agent),
        config=config,
        source_revision=source_revision,
        checkpoint_id=checkpoint_id,
    )


def snapshot_staged_runtime(
    stage: RuntimeCheckpointStage,
    *,
    config: EventFlowConfig,
    source_revision: str,
    checkpoint_id: str | None = None,
) -> RuntimeCheckpointArtifact:
    session = stage.session
    rng, tensors = encode_rng(stage.rng)
    state = _state_metadata(session.state, "state", tensors)
    trajectory = tuple(
        _state_metadata(anchor, f"trajectory.{index}", tensors)
        for index, anchor in enumerate(session.trajectory.checkpoint_snapshots())
    )
    config_bytes = canonical_json_bytes(config)
    try:
        metadata = RuntimeCheckpointMetadata(
            schema_version="phase2-runtime-checkpoint-v1",
            engine_schema="event-engine-v1",
            access_class="validation_private",
            semantics="paused_world",
            config_scope="EventFlowConfig",
            config_canonical_json=config_bytes.decode(),
            config_sha256=sha256_bytes(config_bytes),
            source_revision=source_revision,
            condition=Condition.EVENT_FLOW,
            agent_implementation="scripted-event-flow-v1",
            checkpoint_id=checkpoint_id or uuid4().hex,
            original_device=session.state.core.continuous.device.type,
            public_id=session.public_id,
            truth=session.truth,
            state=state,
            pause_cursor=session.pause_cursor,
            external_queue=session.external_queue.snapshot(),
            prediction_cache=session.prediction_cache.snapshot(),
            trace=CausalTraceArtifact.from_trace(session.trace.snapshot()),
            trajectory=trajectory,
            agent_counters=stage.agent_counters,
            terminal_score=session.terminal_score,
            rng=rng,
            tensors={name: _tensor_metadata(name, tensor) for name, tensor in tensors.items()},
            metadata_sha256="",
        )
        metadata = metadata.model_copy(
            update={"metadata_sha256": _metadata_hash(metadata.model_dump(mode="json"))}
        )
        artifact = RuntimeCheckpointArtifact(metadata, tensors)
        _validate_artifact(
            artifact,
            config=config,
            source_revision=source_revision,
            device=metadata.original_device,
        )
        return artifact
    except (ValidationError, TypeError, ValueError) as error:
        raise ReplayError("invalid checkpoint metadata") from error


def _decode_state(
    metadata: StateMetadata, prefix: str, tensors: dict[str, torch.Tensor], device: str
) -> RuntimeState:
    def channels(group, cls):
        return cls(
            **{
                item.name: tensors.pop(f"{prefix}.{group}.{item.name}").to(device)
                for item in fields(cls)
            }
        )

    current, origin = channels("current", ContinuousState), channels("origin", ContinuousState)
    parameters = SegmentParameters(
        channels("targets", ContinuousChannels),
        channels("rates", ContinuousChannels),
        tensors.pop(f"{prefix}.guard_targets").to(device),
        tensors.pop(f"{prefix}.guard_rates").to(device),
    )
    core = RuntimeCore(
        continuous=current,
        **{name: getattr(metadata.core, name) for name in CoreMetadata.model_fields},
    )
    state = RuntimeState(
        core,
        AnalyticSegment(
            metadata.segment_started_at,
            origin,
            parameters,
            metadata.segment_parent_event_id,
            metadata.prediction_snapshot_sha256,
        ),
        metadata.time,
    )
    validate_runtime_state(state)
    return state


def _trace_from_metadata(metadata: CausalTraceArtifact) -> CausalTrace:
    rows = tuple(
        CausalEventSummary(
            **{name: getattr(row, name) for name in CausalEventSummary.__dataclass_fields__}
        )
        for row in metadata.events
    )
    trace = CausalTrace(rows)
    if trace.sha256 != metadata.sha256:
        raise ReplayError("checkpoint trace hash differs")
    recorder = TraceRecorder()
    for row in rows:
        recorder.append(row)
        if row.tie_id != metadata.events[len(recorder.snapshot().events) - 1].tie_id:
            raise ReplayError("checkpoint tie hash differs")
    return trace


def _validate_history(
    state: RuntimeState, anchors: tuple[RuntimeState, ...], trace: CausalTrace
) -> None:
    if len(anchors) != len(trace.events) + 1:
        raise ReplayError("checkpoint trajectory and trace lengths differ")
    initial = anchors[0]
    if (
        initial.core.last_event_id is not None
        or initial.core.actions
        or initial.core.memory.records
        or initial.core.mode is not Mode.OBSERVING
        or initial.core.counters.jump_applications
    ):
        raise ReplayError("invalid initial causal anchor")
    recorder = TraceRecorder()
    for previous, after, row in zip(anchors, anchors[1:], trace.events, strict=False):
        if row.kind in {"fact", "activate", "terminal"}:
            event = ExternalEvent(
                row.event_id,
                row.timestamp,
                ExternalEventKind.END if row.kind == "terminal" else ExternalEventKind(row.kind),
                row.payload,
            )
        else:
            kind = InternalEventKind(row.kind)
            event = InternalEvent(
                row.event_id,
                row.parent_event_id,
                row.timestamp,
                kind,
                ("recall", "compose", "act").index(row.kind),
                row.raw_predicted_delta if row.was_gap_clamped else row.delta,
            )
        before = replace(
            previous,
            time=row.timestamp,
            core=replace(previous.core, continuous=state_at(previous, row.timestamp)),
        )
        actual = recorder.record(
            event,
            before,
            after.core if row.kind == "terminal" else after,
            selected_record_id=row.selected_record_id,
            selected_rank=row.selected_rank,
            tie=row.tie,
            was_gap_clamped=row.was_gap_clamped,
            raw_predicted_delta=row.raw_predicted_delta,
        )
        if canonical_json_bytes(actual.to_payload()) != canonical_json_bytes(row.to_payload()):
            raise ReplayError("checkpoint causal anchor differs from trace")
    last = anchors[-1]
    if _public_state_anchor(last) != _public_state_anchor(state) or SegmentSummary.from_segment(
        last.segment
    ) != SegmentSummary.from_segment(state.segment):
        raise ReplayError("checkpoint current state differs from causal anchor")
    for item in fields(ComputeCounters):
        a, b = getattr(last.core.counters, item.name), getattr(state.core.counters, item.name)
        if item.name == "checkpoint_flow_evaluations":
            valid = b >= a
        elif item.name == "guard_predictions":
            valid = b in {a, a + 1}
        else:
            valid = a == b
        if not valid:
            raise ReplayError("checkpoint counters differ from causal anchor")


def _validate_artifact(
    artifact: RuntimeCheckpointArtifact,
    *,
    config: EventFlowConfig,
    source_revision: str,
    device: str | None,
) -> tuple[RuntimeSession | None, RngSnapshot]:
    try:
        raw = canonical_json_bytes(artifact.metadata)
        if len(raw) > MAX_METADATA_BYTES:
            raise ReplayError("checkpoint metadata exceeds byte limit")
        metadata = RuntimeCheckpointMetadata.model_validate_json(raw)
        if metadata.metadata_sha256 != _metadata_hash(metadata.model_dump(mode="json")):
            raise ReplayError("checkpoint metadata hash differs")
        if (
            not isinstance(source_revision, str)
            or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source_revision)
            or metadata.source_revision != source_revision
        ):
            raise ReplayError("checkpoint source revision differs")
        config_bytes = canonical_json_bytes(config)
        if (
            metadata.config_canonical_json.encode() != config_bytes
            or metadata.config_sha256 != sha256_bytes(config_bytes)
        ):
            raise ReplayError("checkpoint runtime configuration differs")
        if set(artifact.tensors) != set(metadata.tensors):
            raise ReplayError("checkpoint tensor set differs")
        for name, tensor in artifact.tensors.items():
            if (
                tensor.device.type != "cpu"
                or _tensor_metadata(name, tensor) != metadata.tensors[name]
            ):
                raise ReplayError("checkpoint tensor schema or hash differs")
        _validate_tensor_layout(metadata, artifact.tensors, config)
        _validate_external_history(metadata)
        if device is None:
            # Portable inspection stops at schema, storage and integrity checks.
            # Backend-dependent analytic/cross-anchor validation belongs to
            # same-device runnable restoration, before any caller mutation.
            _trace_from_metadata(metadata.trace)
            ExternalEventQueue.from_snapshot(metadata.external_queue)
            return None, decode_rng(
                metadata.rng,
                {
                    name: value
                    for name, value in artifact.tensors.items()
                    if name.startswith("rng.")
                },
            )
        if device not in {"cpu", "mps"} or (
            device == "mps" and not torch.backends.mps.is_available()
        ):
            raise ReplayError("checkpoint device is unavailable")
        remaining = dict(artifact.tensors)
        state = _decode_state(metadata.state, "state", remaining, device)
        anchors = tuple(
            _decode_state(item, f"trajectory.{index}", remaining, device)
            for index, item in enumerate(metadata.trajectory)
        )
        rng = decode_rng(metadata.rng, remaining)
        trace = _trace_from_metadata(metadata.trace)
        trajectory = Trajectory.from_snapshots(anchors)
        _validate_history(state, anchors, trace)
        if metadata.pause_cursor != state.time:
            raise ReplayError("checkpoint pause cursor differs")
        queue = ExternalEventQueue.from_snapshot(metadata.external_queue)
        terminal = metadata.truth.private_terminal
        pending = queue.snapshot()
        is_terminal = state.core.mode is Mode.TERMINAL
        if (
            not is_terminal
            and tuple(
                event
                for event in pending
                if event.kind in {ExternalEventKind.OUTCOME, ExternalEventKind.END}
            )
            != (terminal,)
        ) or (is_terminal and pending):
            raise ReplayError("checkpoint terminal queue differs")
        if any(
            event.timestamp < state.time or event.event_id in {row.event_id for row in trace.events}
            for event in pending
        ):
            raise ReplayError("checkpoint queue overlaps causal history")
        if (metadata.terminal_score is not None) != is_terminal:
            raise ReplayError("checkpoint terminal score differs")
        cache = PredictionCache.from_snapshot(metadata.prediction_cache)
        session = RuntimeSession(
            metadata.public_id,
            metadata.truth,
            state,
            queue,
            cache,
            TraceRecorder(trace),
            metadata.pause_cursor,
            trajectory,
            terminal_score=metadata.terminal_score,
            segment_anchor=SegmentSummary.from_segment(state.segment),
            public_state_anchor=_public_state_anchor(state),
        )
        EventEngine._check_session(session)
        if cache.is_computed:
            cached = cache.snapshot()
            active = any(
                allowed and float(state.segment.parameters.guard_targets[index]) > 1.0
                for index, allowed in enumerate(allowed_mode_mask(state.core.mode))
            )
            if (cached.event is not None) != active:
                raise ReplayError("checkpoint cached dormancy differs from guard activity")
            if cached.event is not None:
                # A losing near-tie event may precede a pause cursor. Validate at
                # its causal anchor and race there before checking the winner.
                validate_session_boundary(anchors[-1], next_event=cached.event)
                if (
                    cached.event.event_id
                    != INTERNAL_EVENT_ID_BASE + state.core.executed_internal_events
                ):
                    raise ReplayError("checkpoint cached event ordinal differs")
                if not math.isclose(
                    cached.event.predicted_delta,
                    cached.event.timestamp - state.segment.started_at,
                    rel_tol=0.0,
                    abs_tol=2 * math.ulp(cached.event.timestamp),
                ):
                    raise ReplayError("checkpoint cached delta differs from its causal origin")
                normalized, _ = normalize_internal_gap(
                    cached.event, origin_time=state.segment.started_at
                )
                winner = choose_next_event(queue, normalized, current_time=state.segment.started_at)
                if winner is None or winner.timestamp < state.time:
                    raise ReplayError("checkpoint cached winner precedes cursor")
        if state.core.counters.guard_predictions != anchors[
            -1
        ].core.counters.guard_predictions + int(cache.is_computed):
            raise ReplayError("checkpoint prediction counter differs from cache")
        if metadata.agent_counters.foundation_model_calls or any(
            getattr(metadata.agent_counters, item.name) > getattr(state.core.counters, item.name)
            for item in fields(ComputeCounters)
        ):
            raise ReplayError("checkpoint agent counters exceed authoritative runtime counters")
        return session, rng
    except ReplayError:
        raise
    except (
        SilentCascadeError,
        ValidationError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        AttributeError,
    ) as error:
        raise ReplayError("invalid runtime checkpoint") from error


def _validate_tensor_layout(metadata, tensors, config):
    expected = {"rng.cpu"}
    if metadata.rng.has_mps:
        expected.add("rng.mps")
    for prefix in ("state", *(f"trajectory.{index}" for index in range(len(metadata.trajectory)))):
        for group, cls in (
            ("current", ContinuousState),
            ("origin", ContinuousState),
            ("targets", ContinuousChannels),
            ("rates", ContinuousChannels),
        ):
            for item in fields(cls):
                name = f"{prefix}.{group}.{item.name}"
                expected.add(name)
                tensor = tensors[name]
                if (
                    tensor.dtype is not torch.float32
                    or tensor.shape != (getattr(config.dimensions, item.name),)
                    or not bool(torch.isfinite(tensor).all())
                ):
                    raise ReplayError("checkpoint continuous tensor layout differs")
        for name in ("guard_targets", "guard_rates"):
            key = f"{prefix}.{name}"
            expected.add(key)
            tensor = tensors[key]
            if (
                tensor.dtype is not torch.float32
                or tensor.shape != (3,)
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ReplayError("checkpoint guard tensor layout differs")
    if expected != set(tensors):
        raise ReplayError("checkpoint tensor names differ from schema")


def _validate_external_history(metadata: RuntimeCheckpointMetadata) -> None:
    """Bind pending public events to the consumed prefix without private replay."""
    consumed = tuple(
        ExternalEvent(row.event_id, row.timestamp, ExternalEventKind(row.kind), row.payload)
        for row in metadata.trace.events
        if row.kind in {"fact", "activate"}
    )
    pending = tuple(
        event
        for event in metadata.external_queue
        if event.kind in {ExternalEventKind.FACT, ExternalEventKind.ACTIVATE}
    )
    public = PublicEpisode(
        AgentInit(metadata.public_id, 64, 4, metadata.trajectory[0].time), (*consumed, *pending)
    )
    if public.events[-1].timestamp != metadata.truth.activation_time:
        raise ReplayError("checkpoint public and private activation times differ")
    facts = {event.event_id for event in public.events if event.kind is ExternalEventKind.FACT}
    if not set(metadata.truth.relevant_record_ids).issubset(facts):
        raise ReplayError("checkpoint private references include absent public facts")


def publish_runtime_checkpoint(
    path: Path, artifact: RuntimeCheckpointArtifact
) -> RuntimeCheckpointArtifact:
    """Atomically create exactly one archive; even identical existing bytes refuse."""
    metadata = artifact.metadata
    config = EventFlowConfig.model_validate_json(metadata.config_canonical_json)
    _validate_artifact(
        artifact,
        config=config,
        source_revision=metadata.source_revision,
        device=metadata.original_device,
    )
    raw = save(
        {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in artifact.tensors.items()
        },
        metadata={"runtime": canonical_json_bytes(metadata).decode()},
    )
    if len(raw) > MAX_CHECKPOINT_BYTES:
        raise ReplayError("checkpoint exceeds byte limit")
    try:
        _check_parent(path)
        atomic_create_bytes(path, raw, mode=0o600)
    except (OSError, AtomicWriteError) as error:
        raise ReplayError("checkpoint publication failed") from error
    return replace(artifact, path=path, sha256=sha256_bytes(raw))


def _open_parent(path: Path) -> tuple[int, str]:
    absolute = path.absolute()
    if ".." in absolute.parts or not absolute.name:
        raise ReplayError("invalid checkpoint path")
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in absolute.parts[1:-1]:
            next_descriptor = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor, absolute.name
    except BaseException:
        os.close(descriptor)
        raise


def _check_parent(path: Path) -> None:
    descriptor, _ = _open_parent(path)
    os.close(descriptor)


def _read_checkpoint(path: Path) -> bytes:
    parent, name = _open_parent(path)
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
    finally:
        os.close(parent)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_CHECKPOINT_BYTES:
            raise ReplayError("checkpoint must be a bounded regular file")
        chunks, remaining = [], MAX_CHECKPOINT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
            remaining -= len(chunk)
        raise ReplayError("checkpoint exceeds byte limit")
    finally:
        os.close(descriptor)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReplayError("duplicate checkpoint JSON key")
        result[key] = value
    return result


def load_runtime_checkpoint(
    path: Path,
    *,
    config: EventFlowConfig,
    source_revision: str,
    agent_implementation: str = "scripted-event-flow-v1",
) -> RuntimeCheckpointArtifact:
    """Load safely without touching global generators or constructing an agent."""
    try:
        raw = _read_checkpoint(path)
        size = struct.unpack("<Q", raw[:8])[0]
        if size > MAX_METADATA_BYTES or size > len(raw) - 8:
            raise ReplayError("checkpoint header exceeds byte limit")
        header = json.loads(raw[8 : 8 + size], object_pairs_hook=_unique_object)
        if set(header.get("__metadata__", {})) != {"runtime"}:
            raise ReplayError("checkpoint metadata envelope differs")
        encoded = header["__metadata__"]["runtime"].encode()
        payload = json.loads(encoded, object_pairs_hook=_unique_object)
        metadata = RuntimeCheckpointMetadata.model_validate_json(encoded)
        if encoded != canonical_json_bytes(metadata) or metadata.metadata_sha256 != _metadata_hash(
            payload
        ):
            raise ReplayError("checkpoint metadata is noncanonical or hash differs")
        if agent_implementation != metadata.agent_implementation:
            raise ReplayError("checkpoint agent identity differs")
        artifact = RuntimeCheckpointArtifact(metadata, load(raw), path, sha256_bytes(raw))
        # CPU archives validate analytically here. MPS archives remain safely
        # inspectable without MPS; runnable restore validates natively later.
        _validate_artifact(
            artifact,
            config=config,
            source_revision=source_revision,
            device="cpu" if metadata.original_device == "cpu" else None,
        )
        return artifact
    except ReplayError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        struct.error,
        SafetensorError,
        RecursionError,
    ) as error:
        raise ReplayError("invalid checkpoint file") from error


def restore_runtime_session(
    artifact: RuntimeCheckpointArtifact,
    agent: ScriptedEventFlowAgent | None = None,
    *,
    config: EventFlowConfig,
    source_revision: str,
    device: str = "cpu",
    restore_rng: bool = True,
    restore_mps: bool = False,
) -> tuple[RuntimeSession, ScriptedEventFlowAgent]:
    """Validate completely, then transactionally install RNG/public agent counters."""
    if restore_mps and not restore_rng:
        raise ReplayError("MPS restoration requires RNG restoration")
    if device != artifact.metadata.original_device:
        raise ReplayError("runtime checkpoint continuation requires same-device restoration")
    if agent is not None and (
        type(agent) is not ScriptedEventFlowAgent or agent.device != torch.device(device)
    ):
        raise ReplayError("checkpoint requires the registered agent on the target device")
    session, rng = _validate_artifact(
        artifact, config=config, source_revision=source_revision, device=device
    )
    assert session is not None
    if restore_rng:
        validate_rng_restore(rng, restore_mps=restore_mps)
    result_agent = agent if agent is not None else ScriptedEventFlowAgent(device=device)
    counters_before = result_agent.compute_counters()
    try:
        ScriptedEventFlowAgent.restore_compute_counters(
            result_agent, artifact.metadata.agent_counters
        )
        if restore_rng:
            restore_rng_snapshot(rng, restore_mps=restore_mps)
    except Exception as error:
        ScriptedEventFlowAgent.restore_compute_counters(result_agent, counters_before)
        raise ReplayError("checkpoint restoration failed transactionally") from error
    return session, result_agent
