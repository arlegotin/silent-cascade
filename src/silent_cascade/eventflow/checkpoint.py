"""Private immutable safetensors checkpoints for the closed scripted-v1 runtime.

Configuration identity covers EventFlowConfig, not the full experiment config.
CPU source continuation is exact; cross-device analytic rounding is not certified.
All untrusted bytes, tensor hashes and causal anchors are validated before globals
or a caller agent are changed. The archive is integrity checked, not authenticated.
"""

import copy
import json
import re
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

import torch
from pydantic import Field, ValidationError
from safetensors import SafetensorError
from safetensors.torch import load, save

from silent_cascade.env.episode import EpisodeTruth
from silent_cascade.env.reward import EpisodeScore
from silent_cascade.errors import AtomicWriteError, ReplayError, SilentCascadeError
from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at
from silent_cascade.eventflow.checkpoint_rng import (
    RngMetadata,
    encode_rng,
    restore_rng_snapshot,
    validate_rng_restore,
)
from silent_cascade.eventflow.checkpoint_state import (
    StateMetadata,
    TensorMetadata,
    _metadata_hash,
    _state_metadata,
    _tensor_metadata,
    validate_runtime_contents,
)
from silent_cascade.eventflow.config import EventFlowConfig
from silent_cascade.eventflow.engine import (
    EventEngine,
    RuntimeSession,
)
from silent_cascade.eventflow.replay import CausalTraceArtifact
from silent_cascade.eventflow.scheduling import (
    ExternalEventQueue,
    PredictionCache,
    PredictionSnapshot,
)
from silent_cascade.eventflow.scripted import ScriptedEventFlowAgent
from silent_cascade.eventflow.state import (
    ComputeCounters,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.io import atomic_create_bytes
from silent_cascade.logging.trace import (
    TraceRecorder,
)
from silent_cascade.rng import RngSnapshot, snapshot_global_rng
from silent_cascade.schemas import (
    Condition,
    ExternalEvent,
)
from silent_cascade.validation import StrictModel

MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024


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
        return validate_runtime_contents(metadata, artifact.tensors, config, device)
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


def _archive_error(field: str) -> ReplayError:
    return ReplayError(f"invalid checkpoint: {field}", context={"field": field})


def _check_parent(path: Path) -> None:
    with archive_parent(path, error_factory=_archive_error):
        pass


def _read_checkpoint(path: Path) -> bytes:
    with archive_parent(path, error_factory=_archive_error) as (parent, name):
        return read_archive_at(
            parent, name, max_bytes=MAX_CHECKPOINT_BYTES, error_factory=_archive_error
        )


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
        if not isinstance(header, dict):
            raise ReplayError("checkpoint header must be an object")
        envelope = header.get("__metadata__")
        if not isinstance(envelope, dict) or set(envelope) != {"runtime"}:
            raise ReplayError("checkpoint metadata envelope differs")
        if not isinstance(envelope["runtime"], str):
            raise ReplayError("checkpoint runtime metadata must be a string")
        encoded = envelope["runtime"].encode()
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
