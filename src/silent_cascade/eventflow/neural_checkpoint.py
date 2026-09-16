"""Private learned-runtime snapshots, safe restoration and crash staging."""

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

import torch
from pydantic import Field

from silent_cascade.errors import ReplayError
from silent_cascade.eventflow.checkpoint import RuntimeCheckpointMetadata
from silent_cascade.eventflow.checkpoint_rng import (
    encode_rng,
    restore_rng_snapshot,
    validate_rng_restore,
)
from silent_cascade.eventflow.checkpoint_state import (
    _metadata_hash,
    _state_metadata,
    _tensor_metadata,
    validate_runtime_contents,
)
from silent_cascade.eventflow.config import EventFlowConfig
from silent_cascade.eventflow.engine import EventEngine, RuntimeSession
from silent_cascade.eventflow.neural import NeuralDiagnostics, NeuralEventFlowAgent
from silent_cascade.eventflow.neural_weights import (
    MAX_METADATA_BYTES,
    WeightsMetadata,
    decode_archive,
    encode_archive,
    load_neural_weights,
    publish_bytes,
    read_bytes,
    reconstruct_weights,
    snapshot_weights,
)
from silent_cascade.eventflow.replay import CausalTraceArtifact
from silent_cascade.eventflow.scheduling import ExternalEventQueue, PredictionCache
from silent_cascade.eventflow.state import ComputeCounters
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.trace import TraceRecorder
from silent_cascade.rng import RngSnapshot, snapshot_global_rng
from silent_cascade.schemas import AgentInit
from silent_cascade.train.pilot_config import parse_phase4_canonical


class NeuralRuntimeMetadata(RuntimeCheckpointMetadata):
    schema_version: Literal["phase4-neural-runtime-v1"] = "phase4-neural-runtime-v1"
    agent_implementation: Literal["neural-event-flow-v1"] = "neural-event-flow-v1"
    public_init: AgentInit
    diagnostics: NeuralDiagnostics
    experiment_config_canonical_json: str
    experiment_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    weights: WeightsMetadata
    weights_ref: str | None = Field(default=None, pattern=r"^weights-[0-9a-f]{64}\.safetensors$")
    weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class NeuralRuntimeCheckpoint:
    metadata: NeuralRuntimeMetadata
    tensors: dict[str, torch.Tensor]
    path: Path | None = None
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class NeuralRuntimeStage:
    session: RuntimeSession
    public_init: AgentInit
    counters: ComputeCounters
    diagnostics: NeuralDiagnostics
    rng: RngSnapshot
    weights: WeightsMetadata
    weights_ref: str
    weights_sha256: str


def validate_experiment_config(raw, config, identity):
    try:
        full = parse_phase4_canonical(raw)
        if (
            full.event_flow != config
            or canonical_json_bytes(full.neural).decode() != identity.model_config_json
        ):
            raise ValueError("full configuration engine/model sections differ")
        return sha256_bytes(raw.encode())
    except (TypeError, ValueError, AttributeError) as error:
        raise ReplayError("invalid full experiment configuration") from error


def _clone_session(session):
    EventEngine._check_session(session)
    if session.failure is not None:
        raise ReplayError("cannot snapshot failed runtime")
    return replace(
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


def _metadata(
    session,
    *,
    public_init,
    counters,
    diagnostics,
    rng,
    weights,
    config,
    source_revision,
    experiment_config_canonical_json,
    weights_ref=None,
    weights_sha256=None,
):
    full_hash = validate_experiment_config(
        experiment_config_canonical_json, config, weights.identity
    )
    rng_metadata, tensors = encode_rng(rng)
    state = _state_metadata(session.state, "state", tensors)
    trajectory = tuple(
        _state_metadata(anchor, f"trajectory.{i}", tensors)
        for i, anchor in enumerate(session.trajectory.checkpoint_snapshots())
    )
    metadata = NeuralRuntimeMetadata(
        engine_schema="event-engine-v1",
        access_class="validation_private",
        semantics="paused_world",
        config_scope="EventFlowConfig",
        config_canonical_json=canonical_json_bytes(config).decode(),
        config_sha256=sha256_bytes(canonical_json_bytes(config)),
        source_revision=source_revision,
        condition="event_flow",
        checkpoint_id=uuid4().hex,
        original_device=session.state.core.continuous.device.type,
        public_id=session.public_id,
        truth=session.truth,
        state=state,
        pause_cursor=session.pause_cursor,
        external_queue=session.external_queue.snapshot(),
        prediction_cache=session.prediction_cache.snapshot(),
        trace=CausalTraceArtifact.from_trace(session.trace.snapshot()),
        trajectory=trajectory,
        agent_counters=counters,
        terminal_score=session.terminal_score,
        rng=rng_metadata,
        tensors={name: _tensor_metadata(name, value) for name, value in tensors.items()},
        metadata_sha256="",
        public_init=public_init,
        diagnostics=diagnostics,
        weights=weights,
        weights_ref=weights_ref,
        weights_sha256=weights_sha256,
        experiment_config_canonical_json=experiment_config_canonical_json,
        experiment_config_sha256=full_hash,
    )
    return metadata.model_copy(
        update={"metadata_sha256": _metadata_hash(metadata.model_dump(mode="json"))}
    ), tensors


def snapshot_neural_runtime(
    session: RuntimeSession,
    agent: NeuralEventFlowAgent,
    *,
    config: EventFlowConfig,
    source_revision: str,
    experiment_config_canonical_json: str,
) -> NeuralRuntimeCheckpoint:
    try:
        if type(agent) is not NeuralEventFlowAgent:
            raise ReplayError("unregistered neural agent")
        agent._check_model()
        weights, weight_tensors = snapshot_weights(agent.model, agent.identity)
        metadata, tensors = _metadata(
            _clone_session(session),
            public_init=agent._init,
            counters=agent.compute_counters(),
            diagnostics=agent.diagnostic_snapshot(),
            rng=snapshot_global_rng(),
            weights=weights,
            config=config,
            source_revision=source_revision,
            experiment_config_canonical_json=experiment_config_canonical_json,
        )
        artifact = NeuralRuntimeCheckpoint(metadata, {**tensors, **weight_tensors})
        _validate(artifact, config=config, source_revision=source_revision, device=agent.device)
        return artifact
    except ReplayError:
        raise
    except Exception as error:
        raise ReplayError("invalid neural snapshot") from error


def _validated_weights(artifact, device):
    metadata = artifact.metadata
    embedded = {
        name: value
        for name, value in artifact.tensors.items()
        if name.startswith(("parameter/", "buffer/"))
    }
    if metadata.weights_ref is None:
        if metadata.weights_sha256 is not None:
            raise ReplayError("standalone weights cannot claim a containing-file hash")
        return reconstruct_weights(metadata.weights, embedded, device)
    if (
        embedded
        or artifact.path is None
        or metadata.weights_ref != f"weights-{metadata.weights_sha256}.safetensors"
    ):
        raise ReplayError("invalid referenced weights")
    loaded = load_neural_weights(
        artifact.path.parent / metadata.weights_ref,
        expected_sha256=metadata.weights_sha256,
        device=device,
    )
    if (
        loaded.identity != metadata.weights.identity
        or snapshot_weights(loaded.model, loaded.identity)[0] != metadata.weights
    ):
        raise ReplayError("referenced weights identity differs")
    return loaded.model


def _validate(artifact, *, config, source_revision, device):
    try:
        raw = canonical_json_bytes(artifact.metadata)
        if len(raw) > MAX_METADATA_BYTES:
            raise ReplayError("neural metadata byte limit")
        metadata = NeuralRuntimeMetadata.model_validate_json(raw)
        if metadata.metadata_sha256 != _metadata_hash(metadata.model_dump(mode="json")):
            raise ReplayError("neural metadata hash differs")
        if metadata.source_revision != source_revision:
            raise ReplayError("executing source revision differs")
        config_bytes = canonical_json_bytes(config)
        if (
            metadata.config_canonical_json.encode() != config_bytes
            or metadata.config_sha256 != sha256_bytes(config_bytes)
        ):
            raise ReplayError("runtime configuration differs")
        if (
            validate_experiment_config(
                metadata.experiment_config_canonical_json, config, metadata.weights.identity
            )
            != metadata.experiment_config_sha256
        ):
            raise ReplayError("full configuration hash differs")
        if device != metadata.original_device:
            raise ReplayError("neural continuation requires same-device restoration")
        state_tensors = {
            name: value
            for name, value in artifact.tensors.items()
            if not name.startswith(("parameter/", "buffer/"))
        }
        if set(state_tensors) != set(metadata.tensors):
            raise ReplayError("runtime tensor names differ")
        for name, tensor in state_tensors.items():
            if (
                tensor.device.type != "cpu"
                or _tensor_metadata(name, tensor) != metadata.tensors[name]
            ):
                raise ReplayError("runtime tensor schema/hash differs")
        session, rng = validate_runtime_contents(metadata, state_tensors, config, device)
        if metadata.public_init != AgentInit(
            metadata.public_id, 64, 4, metadata.trajectory[0].time
        ):
            raise ReplayError("public initialization differs from initial anchor")
        last = (
            metadata.trajectory[-2]
            if metadata.terminal_score is not None
            else metadata.trajectory[-1]
        )
        if metadata.agent_counters != last.core.counters:
            raise ReplayError("neural counters differ from last callback anchor")
        callback = next(
            (row.kind for row in reversed(metadata.trace.events) if row.kind != "terminal"),
            "initialize",
        )
        if metadata.diagnostics.callback != callback:
            raise ReplayError("neural diagnostics differ from last callback")
        diagnostic = metadata.diagnostics
        if callback == "initialize":
            expected_calls = ()
        elif callback in {"fact", "activate"}:
            expected_calls = (
                ("observe", 1),
                ("record_encoder", 1),
                ("current_record_encoder", 2 if callback == "fact" else 1),
                ("controller", 1),
                ("scorer", 0 if callback == "fact" else 1),
            )
        else:
            expected_calls = (
                ("record_encoder", 2),
                ("jump", 1),
                ("controller", 1),
                ("scorer", 2 if callback == "recall" else 1),
                (callback, 1),
            )
        row = next((row for row in reversed(metadata.trace.events) if row.kind != "terminal"), None)
        if (
            diagnostic.records_scored != (row.counter_delta.records_scored if row else 0)
            or diagnostic.record_rows_encoded
            != {
                "initialize": 0,
                "fact": 66,
                "activate": 65,
                "recall": 128,
                "compose": 128,
                "act": 128,
            }[callback]
            or diagnostic.module_calls != expected_calls
        ):
            raise ReplayError("neural diagnostic counters differ from causal callback")
        model = _validated_weights(artifact, device)
        return session, rng, model
    except ReplayError:
        raise
    except Exception as error:
        raise ReplayError("invalid neural runtime checkpoint") from error


def publish_neural_runtime_checkpoint(
    path: Path, artifact: NeuralRuntimeCheckpoint
) -> NeuralRuntimeCheckpoint:
    candidate = replace(artifact, path=path)
    _validate(
        candidate,
        config=EventFlowConfig.model_validate_json(artifact.metadata.config_canonical_json),
        source_revision=artifact.metadata.source_revision,
        device=artifact.metadata.original_device,
    )
    raw = encode_archive(artifact.tensors, artifact.metadata, "runtime")
    digest = publish_bytes(path, raw)
    return replace(candidate, sha256=digest)


def load_neural_runtime_checkpoint(
    path: Path, *, expected_sha256: str, config: EventFlowConfig, source_revision: str, device: str
) -> NeuralRuntimeCheckpoint:
    raw = read_bytes(path)
    metadata, tensors = decode_archive(raw, expected_sha256, NeuralRuntimeMetadata, "runtime")
    artifact = NeuralRuntimeCheckpoint(metadata, tensors, path, sha256_bytes(raw))
    _validate(artifact, config=config, source_revision=source_revision, device=device)
    return artifact


def restore_neural_runtime(
    artifact: NeuralRuntimeCheckpoint, *, device: str, restore_rng: bool = True
) -> tuple[RuntimeSession, NeuralEventFlowAgent]:
    """Restore all recorded, available generators after complete validation.

    CPU archives retain opaque MPS RNG bytes on CPU-only hosts. An explicit
    MPS target always requires both the backend and a valid captured MPS RNG.
    """
    metadata = artifact.metadata
    session, rng, model = _validate(
        artifact,
        config=EventFlowConfig.model_validate_json(metadata.config_canonical_json),
        source_revision=metadata.source_revision,
        device=device,
    )
    agent = NeuralEventFlowAgent(model, identity=metadata.weights.identity, device=device)
    agent._init, agent._counters, agent._diagnostics = (
        metadata.public_init,
        metadata.agent_counters,
        metadata.diagnostics,
    )
    if restore_rng:
        restore_mps = device == "mps" or (
            rng.torch_mps_state is not None and torch.backends.mps.is_available()
        )
        validate_rng_restore(rng, restore_mps=restore_mps)
        restore_rng_snapshot(rng, restore_mps=restore_mps)
    return session, agent


def stage_neural_runtime(session, agent, *, engine):
    agent._check_model()
    validate_experiment_config(
        engine._experiment_config_canonical_json, engine.config, agent.identity
    )
    reference = engine._neural_weights_reference
    if reference is None or reference[0].identity != agent.identity:
        weights, tensors = snapshot_weights(agent.model, agent.identity)
        raw = encode_archive(tensors, weights, "weights")
        digest = sha256_bytes(raw)
        name = f"weights-{digest}.safetensors"
        engine.crash_root.mkdir(parents=True, exist_ok=True)
        path = engine.crash_root / name
        if path.exists():
            if read_bytes(path) != raw:
                raise ReplayError("existing frozen weights differ")
        else:
            publish_bytes(path, raw)
        reference = weights, name, digest
        engine._neural_weights_reference = reference
    return NeuralRuntimeStage(
        _clone_session(session),
        agent._init,
        agent.compute_counters(),
        agent.diagnostic_snapshot(),
        snapshot_global_rng(),
        *reference,
    )


def publish_neural_crash(path, stage, *, engine):
    metadata, tensors = _metadata(
        stage.session,
        public_init=stage.public_init,
        counters=stage.counters,
        diagnostics=stage.diagnostics,
        rng=stage.rng,
        weights=stage.weights,
        config=engine.config,
        source_revision=engine.source_revision,
        experiment_config_canonical_json=engine._experiment_config_canonical_json,
        weights_ref=stage.weights_ref,
        weights_sha256=stage.weights_sha256,
    )
    return publish_neural_runtime_checkpoint(path, NeuralRuntimeCheckpoint(metadata, tensors))
