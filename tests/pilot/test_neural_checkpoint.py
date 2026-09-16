"""Portable learned continuation and rejection must preserve caller state."""

import json
import struct
from dataclasses import fields, is_dataclass

import pytest
import torch
from pydantic import BaseModel
from safetensors.torch import load, save

from silent_cascade.errors import ReplayError
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes


def snapshot(case):
    from silent_cascade.eventflow.neural_checkpoint import snapshot_neural_runtime

    return snapshot_neural_runtime(
        case.session,
        case.agent,
        config=case.config.event_flow,
        source_revision=case.revision,
        experiment_config_canonical_json=case.canonical,
    )


def _mutable_metadata_containers(value, path="metadata"):
    """Inspect every model/dataclass/tuple descendant, not only weights aliases."""
    if isinstance(value, BaseModel):
        children = ((name, getattr(value, name)) for name in type(value).model_fields)
    elif is_dataclass(value):
        children = ((field.name, getattr(value, field.name)) for field in fields(value))
    elif isinstance(value, dict):
        yield path, value
        children = tuple(value.items())
    elif isinstance(value, list | tuple):
        if isinstance(value, list):
            yield path, value
        children = enumerate(value)
    else:
        return
    for name, child in children:
        yield from _mutable_metadata_containers(child, f"{path}.{name}")


@pytest.mark.parametrize("access", ["constructor", "returned"])
@pytest.mark.parametrize("damage", ["tensor", "mapping", "nested_metadata"])
def test_runtime_archive_owns_immutable_snapshot(neural_archive_case, tmp_path, access, damage):
    from silent_cascade.eventflow.neural_checkpoint import (
        NeuralRuntimeCheckpoint,
        load_neural_runtime_checkpoint,
        publish_neural_runtime_checkpoint,
        restore_neural_runtime,
    )

    case = neural_archive_case
    original = snapshot(case)
    metadata, tensors = original.metadata, original.tensors
    expected_metadata = canonical_json_bytes(metadata)
    expected_tensor = tensors["state.current.z_fast"].clone()
    artifact = NeuralRuntimeCheckpoint(metadata, tensors)
    if access == "returned":
        metadata, tensors = artifact.metadata, artifact.tensors
    if damage == "tensor":
        tensors["state.current.z_fast"][0] += 1
    elif damage == "mapping":
        tensors.clear()
    else:
        containers = list(_mutable_metadata_containers(metadata))
        assert {
            "metadata.weights.aliases",
            "metadata.tensors",
            "metadata.rng.python_state",
            "metadata.rng.python_state.tuple",
            "metadata.rng.python_state.tuple.1.tuple",
        } <= {path for path, _ in containers}
        for _, container in containers:
            container.clear()
        assert canonical_json_bytes(metadata) != expected_metadata
    assert canonical_json_bytes(artifact.metadata) == expected_metadata
    assert torch.equal(artifact.tensors["state.current.z_fast"], expected_tensor)
    published = publish_neural_runtime_checkpoint(tmp_path / "isolated.safetensors", artifact)
    loaded = load_neural_runtime_checkpoint(
        published.path,
        expected_sha256=published.sha256,
        config=case.config.event_flow,
        source_revision=case.revision,
        device="cpu",
    )
    # Loaded descriptors and live restored sessions must not expose archive storage either.
    view = loaded.tensors["state.current.z_fast"]
    view.add_(1)
    assert not torch.equal(view, expected_tensor)
    loaded.metadata.weights.aliases.clear()
    session, agent = restore_neural_runtime(loaded, device="cpu", restore_rng=False)
    while not case.engine.step(session, agent):
        pass
    while not case.engine.step(case.session, case.agent):
        pass
    assert session.trace.snapshot().sha256 == case.session.trace.snapshot().sha256
    assert canonical_json_bytes(loaded.metadata) == expected_metadata
    assert torch.equal(loaded.tensors["state.current.z_fast"], expected_tensor)


def test_midflow_neural_resume_is_exact(neural_archive_case, tmp_path):
    from silent_cascade.eventflow.neural_checkpoint import (
        load_neural_runtime_checkpoint,
        publish_neural_runtime_checkpoint,
        restore_neural_runtime,
    )

    case = neural_archive_case
    artifact = publish_neural_runtime_checkpoint(tmp_path / "runtime.safetensors", snapshot(case))
    loaded = load_neural_runtime_checkpoint(
        artifact.path,
        expected_sha256=artifact.sha256,
        config=case.config.event_flow,
        source_revision=case.revision,
        device="cpu",
    )
    session, agent = restore_neural_runtime(loaded, device="cpu")
    assert agent.identity.source_revision == "a" * 40
    assert loaded.metadata.source_revision == "b" * 40
    assert agent.diagnostic_snapshot() == case.agent.diagnostic_snapshot()
    while not case.engine.step(case.session, case.agent):
        pass
    while not case.engine.step(session, agent):
        pass
    assert session.state.core.actions == case.session.state.core.actions
    assert session.terminal_score == case.session.terminal_score
    assert session.trace.snapshot().sha256 == case.session.trace.snapshot().sha256


@pytest.mark.parametrize(
    "damage",
    [
        "queue",
        "cache",
        "support",
        "provenance",
        "init",
        "counters",
        "neural_counters",
        "module_calls",
        "origin",
        "rates",
        "shape",
        "weights",
        "aliases",
        "source",
        "config",
        "full_config",
        "trace",
    ],
)
def test_neural_corruption_preserves_rng_and_model(neural_archive_case, tmp_path, damage):
    from silent_cascade.eventflow.neural_checkpoint import (
        load_neural_runtime_checkpoint,
        publish_neural_runtime_checkpoint,
    )

    case = neural_archive_case
    published = publish_neural_runtime_checkpoint(tmp_path / "runtime.safetensors", snapshot(case))
    raw = published.path.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    payload = json.loads(json.loads(raw[8 : 8 + size])["__metadata__"]["runtime"])
    tensors = load(raw)
    if damage == "queue":
        payload["external_queue"] = []
    elif damage == "cache":
        payload["prediction_cache"]["event"]["timestamp"] += 0.1
    elif damage == "support":
        payload["state"]["core"]["support_ids"] = [999]
    elif damage == "provenance":
        payload["state"]["core"]["memory"]["records"][0]["record"]["provenance"] = "simulated"
    elif damage == "init":
        payload["public_init"]["initial_time"] = 0.1
    elif damage == "counters":
        payload["agent_counters"]["foundation_model_calls"] = 1
    elif damage == "neural_counters":
        payload["diagnostics"]["records_scored"] = -1
    elif damage == "module_calls":
        payload["diagnostics"]["module_calls"][0][1] = 999
    elif damage in {"origin", "rates"}:
        tensors[f"state.{damage}.z_fast"][0] = 100.0
    elif damage == "shape":
        tensors["state.current.z_fast"] = torch.zeros(1)
    elif damage == "weights":
        tensors[next(k for k in tensors if k.startswith("parameter/"))].zero_()
    elif damage == "aliases":
        payload["weights"]["aliases"] = {}
    elif damage == "source":
        payload["source_revision"] = "c" * 40
    elif damage == "config":
        payload["config_sha256"] = "0" * 64
    elif damage == "full_config":
        payload["experiment_config_canonical_json"] = "{}"
    elif damage == "trace":
        payload["trace"]["sha256"] = "0" * 64
    if damage == "weights":
        tensors[next(k for k in tensors if k.startswith("parameter/"))].fill_(0.25)
    if damage in {"origin", "rates", "shape"}:
        from silent_cascade.eventflow.checkpoint_state import _tensor_metadata

        for name in payload["tensors"]:
            payload["tensors"][name] = _tensor_metadata(name, tensors[name]).model_dump(mode="json")
    payload["metadata_sha256"] = sha256_bytes(
        canonical_json_bytes({k: v for k, v in payload.items() if k != "metadata_sha256"})
    )
    corrupt = save(tensors, metadata={"runtime": canonical_json_bytes(payload).decode()})
    published.path.write_bytes(corrupt)
    before = torch.get_rng_state().clone()
    counters = case.agent.compute_counters()
    with pytest.raises(ReplayError):
        load_neural_runtime_checkpoint(
            published.path,
            expected_sha256=sha256_bytes(corrupt),
            config=case.config.event_flow,
            source_revision=case.revision,
            device="cpu",
        )
    assert torch.equal(torch.get_rng_state(), before)
    case.agent._check_model()
    assert counters == case.agent.compute_counters()


@pytest.mark.parametrize("raw", [None, "{}", "[" * 70 + "]" * 70])
def test_full_config_required(neural_archive_case, raw):
    from silent_cascade.eventflow.neural_checkpoint import snapshot_neural_runtime

    case = neural_archive_case
    with pytest.raises(ReplayError):
        snapshot_neural_runtime(
            case.session,
            case.agent,
            config=case.config.event_flow,
            source_revision=case.revision,
            experiment_config_canonical_json=raw,
        )


def rng_signature():
    from silent_cascade.rng import snapshot_global_rng

    state = snapshot_global_rng()
    return (
        state.python_state,
        state.numpy_state[0],
        state.numpy_state[1].tobytes(),
        state.numpy_state[2:],
        state.torch_cpu_state.numpy().tobytes(),
        None if state.torch_mps_state is None else state.torch_mps_state.numpy().tobytes(),
    )


def test_restore_without_rng_or_forward_preserves_all_generators(neural_archive_case, monkeypatch):
    from silent_cascade.eventflow.neural_checkpoint import restore_neural_runtime
    from silent_cascade.models.event_flow import EventFlowModel

    artifact = snapshot(neural_archive_case)
    before = rng_signature()

    def forbidden(*args, **kwargs):
        pytest.fail("restore ran a learned forward")

    for name in ("initial_context", "preview_and_control", "compose", "action", "recall_scores"):
        monkeypatch.setattr(EventFlowModel, name, forbidden)
    session, agent = restore_neural_runtime(artifact, device="cpu", restore_rng=False)
    assert (
        session.prediction_cache.snapshot()
        == neural_archive_case.session.prediction_cache.snapshot()
    )
    assert agent.identity == neural_archive_case.identity
    assert rng_signature() == before


def test_restore_rng_restores_all_captured_generators(neural_archive_case):
    import random

    import numpy as np

    from silent_cascade.eventflow.neural_checkpoint import restore_neural_runtime

    artifact = snapshot(neural_archive_case)
    before = rng_signature()
    random.random()
    np.random.random()
    torch.rand(1)
    if torch.backends.mps.is_available():
        torch.rand(1, device="mps")
    restore_neural_runtime(artifact, device="cpu", restore_rng=True)
    assert rng_signature() == before


def test_cpu_restore_ignores_unavailable_mps_rng(neural_archive_case, monkeypatch):
    from silent_cascade.eventflow.neural_checkpoint import restore_neural_runtime

    artifact = snapshot(neural_archive_case)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.mps, "set_rng_state", lambda *args: pytest.fail("CPU-only restore touched MPS")
    )
    before = rng_signature()
    torch.rand(1)
    restore_neural_runtime(artifact, device="cpu", restore_rng=True)
    assert rng_signature() == before


def test_invalid_native_mps_rng_is_transactional(neural_archive_case):
    if not torch.backends.mps.is_available():
        pytest.skip("native MPS backend unavailable")
    from silent_cascade.eventflow.checkpoint_state import _metadata_hash, _tensor_metadata
    from silent_cascade.eventflow.neural_checkpoint import (
        NeuralRuntimeCheckpoint,
        restore_neural_runtime,
    )

    artifact = snapshot(neural_archive_case)
    tensors = dict(artifact.tensors)
    tensors["rng.mps"] = torch.zeros(1, dtype=torch.uint8)
    fields = dict(artifact.metadata.tensors)
    fields["rng.mps"] = _tensor_metadata("rng.mps", tensors["rng.mps"])
    metadata = artifact.metadata.model_copy(update={"tensors": fields})
    metadata = metadata.model_copy(
        update={"metadata_sha256": _metadata_hash(metadata.model_dump(mode="json"))}
    )
    invalid = NeuralRuntimeCheckpoint(metadata, tensors)
    before = rng_signature()
    with pytest.raises(ReplayError):
        restore_neural_runtime(invalid, device="cpu")
    assert rng_signature() == before


@pytest.mark.parametrize(
    "damage", ["aliases", "shape", "extra", "logical", "header", "depth", "file_limit", "symlink"]
)
def test_weights_corruption_and_limits_before_allocation(
    neural_archive_case, tmp_path, monkeypatch, damage
):
    import os

    from silent_cascade.eventflow import neural_weights as codec

    case = neural_archive_case
    path = tmp_path / "weights.safetensors"
    codec.save_neural_weights(path, model=case.model, identity=case.identity)
    raw = path.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    payload = json.loads(json.loads(raw[8 : 8 + size])["__metadata__"]["weights"])
    tensors = load(raw)
    if damage == "aliases":
        payload["aliases"] = {}
    elif damage == "shape":
        name = next(
            k for k in sorted(tensors) if k.startswith("parameter/") and tensors[k].numel() > 1
        )
        original_shape = tensors[name].shape
        tensors[name] = tensors[name].flatten()[:1]
        assert tensors[name].shape != original_shape
        from silent_cascade.models.weights import model_state_sha256

        payload["identity"]["model_state_sha256"] = model_state_sha256(tensors, payload["aliases"])
    elif damage == "extra":
        tensors["unregistered"] = torch.ones(1)
    elif damage == "logical":
        payload["identity"]["model_state_sha256"] = "0" * 64
    elif damage == "header":
        raw = struct.pack("<Q", 16 * 1024 * 1024 + 1)
    elif damage == "depth":
        header = b"[" * 65 + b"]" * 65
        raw = struct.pack("<Q", len(header)) + header
    if damage in {"aliases", "shape", "extra", "logical"}:
        modified = save(tensors, metadata={"weights": canonical_json_bytes(payload).decode()})
        assert modified != raw
        raw = modified
    path.write_bytes(raw)
    expected = sha256_bytes(raw)
    if damage == "file_limit":
        with path.open("r+b") as stream:
            stream.truncate(128 * 1024 * 1024 + 1)
    if damage == "symlink":
        link = tmp_path / "linked.safetensors"
        os.symlink(path, link)
        path = link
    if damage in {"header", "depth", "file_limit", "symlink"}:
        monkeypatch.setattr(
            codec, "load", lambda *args: pytest.fail("unsafe bytes reached tensor allocation")
        )
    before = rng_signature()
    with pytest.raises(ReplayError):
        codec.load_neural_weights(path, expected_sha256=expected, device="cpu")
    assert rng_signature() == before
    case.agent._check_model()


def test_full_config_model_mismatch_is_rejected(neural_archive_case):
    from silent_cascade.eventflow.neural_checkpoint import snapshot_neural_runtime
    from silent_cascade.train.pilot_config import resolve_pilot_config

    case = neural_archive_case
    raw = canonical_json_bytes(resolve_pilot_config("phase4_pilot").config).decode()
    with pytest.raises(ReplayError):
        snapshot_neural_runtime(
            case.session,
            case.agent,
            config=case.config.event_flow,
            source_revision=case.revision,
            experiment_config_canonical_json=raw,
        )


def test_publication_refuses_overwrite_and_symlinked_parent(neural_archive_case, tmp_path):
    from silent_cascade.eventflow.neural_weights import save_neural_weights

    case = neural_archive_case
    path = tmp_path / "weights.safetensors"
    save_neural_weights(path, model=case.model, identity=case.identity)
    before = path.read_bytes()
    with pytest.raises(ReplayError):
        save_neural_weights(path, model=case.model, identity=case.identity)
    assert path.read_bytes() == before
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ReplayError):
        save_neural_weights(
            tmp_path / "link/other.safetensors", model=case.model, identity=case.identity
        )
    assert not (tmp_path / "other.safetensors").exists()


def test_cross_device_runtime_restore_is_rejected(neural_archive_case):
    from silent_cascade.eventflow.neural_checkpoint import restore_neural_runtime

    with pytest.raises(ReplayError, match="same-device"):
        restore_neural_runtime(snapshot(neural_archive_case), device="mps", restore_rng=False)


def test_native_mps_neural_resume(neural_archive_case, tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip("native MPS backend unavailable")
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent
    from silent_cascade.eventflow.neural_checkpoint import (
        load_neural_runtime_checkpoint,
        publish_neural_runtime_checkpoint,
        restore_neural_runtime,
    )

    case = neural_archive_case
    case.model.to("mps")
    case.agent = NeuralEventFlowAgent(case.model, identity=case.identity, device="mps")
    case.session = case.engine.start_episode(case.bundle, case.agent)
    while case.session.state.core.mode.value != "have_memory":
        assert not case.engine.step(case.session, case.agent)
    event = case.engine.next_internal_event(case.session, case.agent)
    case.engine.run_until(case.session, case.agent, (case.session.state.time + event.timestamp) / 2)
    artifact = publish_neural_runtime_checkpoint(tmp_path / "mps.safetensors", snapshot(case))
    loaded = load_neural_runtime_checkpoint(
        artifact.path,
        expected_sha256=artifact.sha256,
        config=case.config.event_flow,
        source_revision=case.revision,
        device="mps",
    )
    session, agent = restore_neural_runtime(loaded, device="mps")
    while not case.engine.step(case.session, case.agent):
        pass
    while not case.engine.step(session, agent):
        pass
    assert session.state.core.actions == case.session.state.core.actions
    assert session.trace.snapshot().sha256 == case.session.trace.snapshot().sha256
