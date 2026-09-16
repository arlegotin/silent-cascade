"""Private learned replay reconstructs every decision from portable weights."""

from pathlib import Path
from typing import Literal

from pydantic import Field

from silent_cascade.env.episode import EpisodeArtifact
from silent_cascade.errors import ReplayError, SilentCascadeError
from silent_cascade.eventflow.checkpoint_state import _trace_from_metadata
from silent_cascade.eventflow.config import EventFlowConfig
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.neural import NeuralEventFlowAgent, NeuralModelIdentity
from silent_cascade.eventflow.neural_checkpoint import validate_experiment_config
from silent_cascade.eventflow.neural_weights import (
    MAX_METADATA_BYTES,
    bounded_json,
    load_neural_weights,
    publish_bytes,
    read_bytes,
)
from silent_cascade.eventflow.replay import (
    CausalTraceArtifact,
    EpisodeResultArtifact,
    _compare,
    _payload_hash,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.validation import StrictModel


class NeuralReplayArtifact(StrictModel):
    schema_version: Literal["phase4-neural-replay-v1"] = "phase4-neural-replay-v1"
    access_class: Literal["validation_private"] = "validation_private"
    episode: EpisodeArtifact
    identity: NeuralModelIdentity
    source_revision: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    config_canonical_json: str
    config_sha256: str
    experiment_config_canonical_json: str
    experiment_config_sha256: str
    weights_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_device: Literal["cpu", "mps"]
    expected_result: EpisodeResultArtifact
    trace: CausalTraceArtifact
    payload_sha256: str


class NeuralReplayComparison(StrictModel):
    matched: Literal[True] = True
    event_count: int
    trace_sha256: str
    weights_sha256: str
    trace: CausalTraceArtifact
    result: EpisodeResultArtifact
    mismatches: tuple[str, ...] = ()


def _parse(raw):
    try:
        payload = bounded_json(raw)
        artifact = NeuralReplayArtifact.model_validate_json(raw)
        if raw != canonical_json_bytes(artifact) or artifact.payload_sha256 != _payload_hash(
            payload
        ):
            raise ReplayError("neural replay canonical bytes/hash differ")
        config = EventFlowConfig.model_validate_json(artifact.config_canonical_json)
        if artifact.config_canonical_json.encode() != canonical_json_bytes(
            config
        ) or artifact.config_sha256 != sha256_bytes(canonical_json_bytes(config)):
            raise ReplayError("neural replay engine configuration differs")
        if (
            validate_experiment_config(
                artifact.experiment_config_canonical_json, config, artifact.identity
            )
            != artifact.experiment_config_sha256
        ):
            raise ReplayError("neural replay full configuration hash differs")
        bundle = artifact.episode.to_bundle()
        if artifact.expected_result.public_id != bundle.public.init.episode_public_id:
            raise ReplayError("neural replay public identity differs")
        _trace_from_metadata(artifact.trace)
        return artifact, bundle, config
    except ReplayError:
        raise
    except Exception as error:
        raise ReplayError("invalid neural replay") from error


def write_neural_replay(
    path: Path,
    *,
    bundle,
    result,
    identity,
    config,
    weights: Path,
    source_revision: str,
    experiment_config_canonical_json: str,
) -> str:
    digest = sha256_bytes(read_bytes(weights))
    loaded = load_neural_weights(weights, expected_sha256=digest, device="cpu")
    if loaded.identity != identity:
        raise ReplayError("replay logical model identity differs")
    initial = result.trajectory.checkpoint_snapshots()[0]
    artifact = NeuralReplayArtifact(
        episode=EpisodeArtifact.from_bundle(bundle),
        identity=identity,
        source_revision=source_revision,
        config_canonical_json=canonical_json_bytes(config).decode(),
        config_sha256=sha256_bytes(canonical_json_bytes(config)),
        experiment_config_canonical_json=experiment_config_canonical_json,
        experiment_config_sha256=validate_experiment_config(
            experiment_config_canonical_json, config, identity
        ),
        weights_sha256=digest,
        original_device=initial.core.continuous.device.type,
        expected_result=EpisodeResultArtifact.from_result(result),
        trace=CausalTraceArtifact.from_trace(result.trace),
        payload_sha256="",
    )
    payload = artifact.model_dump(mode="json")
    payload["payload_sha256"] = _payload_hash(payload)
    raw = canonical_json_bytes(payload)
    _parse(raw)
    return publish_bytes(path, raw)


def verify_neural_replay(path, *, weights_path: Path) -> NeuralReplayComparison:
    artifact, bundle, config = _parse(read_bytes(path, max_bytes=MAX_METADATA_BYTES))
    weights = load_neural_weights(
        weights_path, expected_sha256=artifact.weights_sha256, device=artifact.original_device
    )
    if weights.identity != artifact.identity:
        raise ReplayError("replay model identity differs")
    agent = NeuralEventFlowAgent(
        weights.model, identity=weights.identity, device=artifact.original_device
    )
    try:
        result = EventEngine(config).run_episode(bundle, agent)
    except SilentCascadeError as error:
        raise ReplayError(
            "neural replay mismatch: runtime", context={"field": "runtime"}
        ) from error
    actual_trace, actual_result = (
        CausalTraceArtifact.from_trace(result.trace),
        EpisodeResultArtifact.from_result(result),
    )
    for name, expected, actual in (
        ("trace", artifact.trace, actual_trace),
        ("result", artifact.expected_result, actual_result),
    ):
        _compare(expected.model_dump(mode="json"), actual.model_dump(mode="json"), name)
        if canonical_json_bytes(expected) != canonical_json_bytes(actual):
            raise ReplayError(f"neural replay mismatch: {name}", context={"field": name})
    return NeuralReplayComparison(
        event_count=len(result.trace.events),
        trace_sha256=result.trace.sha256,
        weights_sha256=weights.sha256,
        trace=actual_trace,
        result=actual_result,
    )
