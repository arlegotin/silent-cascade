"""Safe one-file archive validation must precede any caller mutation."""

import json
import os
import random
import struct

import numpy as np
import pytest
import torch
from conftest import CASES, bundle_for
from safetensors.torch import load, save

from silent_cascade.errors import ReplayError
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.scripted import ScriptedEventFlowAgent
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.rng import snapshot_global_rng

REVISION = "a" * 40


@pytest.mark.parametrize(
    "header",
    [
        b"[]",
        b'{"__metadata__":[]}',
        b'{"__metadata__":{"runtime":1}}',
        b'{"__metadata__":{"runtime":null}}',
        b'{"__metadata__":{"runtime":[]}}',
        b"[" * 1500 + b"0" + b"]" * 1500,
        json.dumps({"__metadata__": {"runtime": "[" * 1500 + "0" + "]" * 1500}}).encode(),
    ],
    ids=[
        "list-header",
        "list-envelope",
        "integer-runtime",
        "null-runtime",
        "list-runtime",
        "deep-header",
        "deep-runtime",
    ],
)
def test_malformed_checkpoint_shapes_are_typed_without_mutation(
    tmp_path, config, monkeypatch, header
):
    from silent_cascade.eventflow.checkpoint import load_runtime_checkpoint

    path = tmp_path / "malformed.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    caller = ScriptedEventFlowAgent()
    before, counters = rng_signature(), caller.compute_counters()

    def forbidden(*args, **kwargs):
        pytest.fail("malformed archive reached execution or a mutation boundary")

    monkeypatch.setattr(EventEngine, "start_episode", forbidden)
    monkeypatch.setattr(random, "setstate", forbidden)
    monkeypatch.setattr(np.random, "set_state", forbidden)
    monkeypatch.setattr(torch, "set_rng_state", forbidden)
    monkeypatch.setattr(ScriptedEventFlowAgent, "restore_compute_counters", forbidden)
    with pytest.raises(ReplayError):
        load_runtime_checkpoint(path, config=config, source_revision=REVISION)
    assert rng_signature() == before
    assert caller.compute_counters() == counters


def archive(tmp_path, config):
    from silent_cascade.eventflow.checkpoint import publish_runtime_checkpoint, snapshot_runtime

    engine, agent = EventEngine(config), ScriptedEventFlowAgent()
    session = engine.start_episode(bundle_for(CASES[0]), agent)
    engine.run_until(session, agent, 1.1)
    artifact = snapshot_runtime(session, agent, config=config, source_revision=REVISION)
    path = tmp_path / "runtime.safetensors"
    publish_runtime_checkpoint(path, artifact)
    return path, artifact, session, agent


def unpack(path):
    raw = path.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + size])
    return json.loads(header["__metadata__"]["runtime"]), load(raw)


def rewrite(path, payload, tensors, *, canonical=True):
    payload["metadata_sha256"] = sha256_bytes(
        canonical_json_bytes({k: v for k, v in payload.items() if k != "metadata_sha256"})
    )
    encoded = canonical_json_bytes(payload).decode() if canonical else json.dumps(payload, indent=2)
    path.write_bytes(save(tensors, metadata={"runtime": encoded}))


def rng_signature():
    snapshot = snapshot_global_rng()
    return (
        snapshot.python_state,
        snapshot.numpy_state[0],
        snapshot.numpy_state[1].tobytes(),
        snapshot.numpy_state[2:],
        bytes(snapshot.torch_cpu_state.tolist()),
        None if snapshot.torch_mps_state is None else bytes(snapshot.torch_mps_state.tolist()),
    )


@pytest.mark.parametrize(
    "damage",
    [
        "schema",
        "config",
        "source",
        "agent",
        "missing_tensor",
        "extra_tensor",
        "dtype",
        "shape",
        "hash",
        "nonfinite",
        "queue",
        "trace",
        "noncanonical",
        "rng",
        "unknown_field",
    ],
)
def test_corruption_is_refused_without_global_or_agent_mutation(tmp_path, config, damage):
    from silent_cascade.eventflow.checkpoint import load_runtime_checkpoint

    path, _, _, agent = archive(tmp_path, config)
    payload, tensors = unpack(path)
    first = "state.current.z_fast"
    if damage == "schema":
        payload["schema_version"] = "unknown"
    elif damage == "config":
        payload["config_sha256"] = "0" * 64
    elif damage == "source":
        payload["source_revision"] = "b" * 40
    elif damage == "agent":
        payload["agent_implementation"] = "arbitrary.module.Agent"
    elif damage == "missing_tensor":
        del tensors[first]
    elif damage == "extra_tensor":
        tensors["extra"] = torch.zeros(1)
    elif damage == "dtype":
        tensors[first] = tensors[first].double()
    elif damage == "shape":
        tensors[first] = tensors[first][:1]
    elif damage == "hash":
        tensors[first][0] += 0.001
    elif damage == "nonfinite":
        tensors[first][0] = float("nan")
    elif damage == "queue":
        payload["external_queue"] = []
    elif damage == "trace":
        payload["trace"]["sha256"] = "0" * 64
    elif damage == "rng":
        payload["rng"]["numpy_position"] = 999999
    elif damage == "unknown_field":
        payload["arbitrary_class"] = "anything"
    rewrite(path, payload, tensors, canonical=damage != "noncanonical")
    before, counters = rng_signature(), agent.compute_counters()
    with pytest.raises(ReplayError):
        load_runtime_checkpoint(path, config=config, source_revision=REVISION)
    assert rng_signature() == before
    assert agent.compute_counters() == counters


@pytest.mark.parametrize("kind", ["truncated", "symlink", "ancestor_symlink", "fifo", "oversize"])
def test_nonregular_or_unbounded_archive_refused(tmp_path, config, kind):
    from silent_cascade.eventflow.checkpoint import MAX_CHECKPOINT_BYTES, load_runtime_checkpoint

    path, _, _, _ = archive(tmp_path, config)
    if kind == "truncated":
        path.write_bytes(path.read_bytes()[:-10])
    elif kind == "symlink":
        link = tmp_path / "link.safetensors"
        link.symlink_to(path)
        path = link
    elif kind == "ancestor_symlink":
        link = tmp_path / "directory"
        link.symlink_to(tmp_path, target_is_directory=True)
        path = link / path.name
    elif kind == "fifo":
        path = tmp_path / "fifo"
        os.mkfifo(path)
    elif kind == "oversize":
        with path.open("r+b") as stream:
            stream.truncate(MAX_CHECKPOINT_BYTES + 1)
    with pytest.raises(ReplayError):
        load_runtime_checkpoint(path, config=config, source_revision=REVISION)


def test_existing_destination_is_never_overwritten(tmp_path, config):
    from silent_cascade.eventflow.checkpoint import publish_runtime_checkpoint

    path, artifact, _, _ = archive(tmp_path, config)
    before = path.read_bytes()
    with pytest.raises(ReplayError):
        publish_runtime_checkpoint(path, artifact)
    assert path.read_bytes() == before


def test_rng_roundtrip_restores_draws_and_preserves_archived_mps_on_cpu(tmp_path, config):
    from silent_cascade.eventflow.checkpoint import restore_runtime_session

    path, artifact, _, agent = archive(tmp_path, config)
    expected = (random.random(), np.random.random(), torch.rand(5))
    restored, resumed = restore_runtime_session(artifact, config=config, source_revision=REVISION)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    assert torch.equal(torch.rand(5), expected[2])
    assert resumed.compute_counters() == agent.compute_counters()
    assert restored.failure is None
    _, tensors = unpack(path)
    if "rng.mps" in tensors:
        assert tensors["rng.mps"].dtype is torch.uint8


def test_failed_restore_revalidates_mutable_tensor_storage(tmp_path, config):
    from silent_cascade.eventflow.checkpoint import restore_runtime_session

    _, artifact, _, agent = archive(tmp_path, config)
    artifact.tensors["state.current.z_fast"][0] = 0.99
    before, counters = rng_signature(), agent.compute_counters()
    with pytest.raises(ReplayError):
        restore_runtime_session(artifact, agent, config=config, source_revision=REVISION)
    assert rng_signature() == before and agent.compute_counters() == counters


def test_mps_restore_requires_rng_restoration(tmp_path, config):
    from silent_cascade.eventflow.checkpoint import restore_runtime_session

    _, artifact, _, agent = archive(tmp_path, config)
    before = rng_signature()
    with pytest.raises(ReplayError):
        restore_runtime_session(
            artifact,
            agent,
            config=config,
            source_revision=REVISION,
            restore_rng=False,
            restore_mps=True,
        )
    assert rng_signature() == before


@pytest.mark.parametrize(
    "damage",
    [
        "dormancy",
        "event_id",
        "delta",
        "agent_counters",
        "nonfinite_rehashed",
        "extra_tensor_rehashed",
    ],
)
def test_coherent_envelope_cannot_hide_invalid_semantics(tmp_path, config, damage):
    from silent_cascade.eventflow.checkpoint import load_runtime_checkpoint
    from silent_cascade.logging.trace import tensor_sha256

    path, _, _, _ = archive(tmp_path, config)
    payload, tensors = unpack(path)
    if damage == "dormancy":
        payload["prediction_cache"]["event"] = None
    elif damage == "event_id":
        payload["prediction_cache"]["event"]["event_id"] += 100
    elif damage == "delta":
        payload["prediction_cache"]["event"]["predicted_delta"] += 0.01
    elif damage == "agent_counters":
        payload["agent_counters"]["controller_calls"] += 100
    else:
        name = "state.current.z_fast" if damage == "nonfinite_rehashed" else "extra"
        if damage == "nonfinite_rehashed":
            tensors[name][0] = float("nan")
        else:
            tensors[name] = torch.zeros(1)
        payload["tensors"][name] = {
            "dtype": str(tensors[name].dtype),
            "shape": list(tensors[name].shape),
            "sha256": sha256_bytes(
                canonical_json_bytes({"name": name, "tensor_sha256": tensor_sha256(tensors[name])})
            ),
        }
    rewrite(path, payload, tensors)
    with pytest.raises(ReplayError):
        load_runtime_checkpoint(path, config=config, source_revision=REVISION)


def test_rng_install_failure_rolls_back_all_rngs_and_caller_counters(tmp_path, config, monkeypatch):
    from silent_cascade.eventflow.checkpoint import restore_runtime_session

    _, artifact, _, _ = archive(tmp_path, config)
    caller = ScriptedEventFlowAgent()
    random.random()
    np.random.random()
    torch.rand(7)
    before, counters = rng_signature(), caller.compute_counters()
    original = torch.set_rng_state
    failed = False

    def fail_once(value):
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("injected CPU RNG installation failure")
        original(value)

    monkeypatch.setattr(torch, "set_rng_state", fail_once)
    with pytest.raises(ReplayError):
        restore_runtime_session(artifact, caller, config=config, source_revision=REVISION)
    assert rng_signature() == before
    assert caller.compute_counters() == counters


def test_archived_mps_rng_is_retained_without_requiring_backend(tmp_path, config, monkeypatch):
    from silent_cascade.eventflow.checkpoint import load_runtime_checkpoint, restore_runtime_session
    from silent_cascade.logging.trace import tensor_sha256

    path, _, _, _ = archive(tmp_path, config)
    payload, tensors = unpack(path)
    # Stand-in bytes exercise portable storage, never claim valid MPS execution.
    tensors["rng.mps"] = torch.arange(16, dtype=torch.uint8)
    payload["rng"]["has_mps"] = True
    payload["tensors"]["rng.mps"] = {
        "dtype": "torch.uint8",
        "shape": [16],
        "sha256": sha256_bytes(
            canonical_json_bytes(
                {"name": "rng.mps", "tensor_sha256": tensor_sha256(tensors["rng.mps"])}
            )
        ),
    }
    rewrite(path, payload, tensors)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    loaded = load_runtime_checkpoint(path, config=config, source_revision=REVISION)
    restore_runtime_session(loaded, config=config, source_revision=REVISION)
    assert torch.equal(loaded.tensors["rng.mps"], torch.arange(16, dtype=torch.uint8))
    before = rng_signature()
    with pytest.raises(ReplayError):
        restore_runtime_session(loaded, config=config, source_revision=REVISION, restore_mps=True)
    assert rng_signature() == before


def test_archive_has_only_json_metadata_and_safetensors_payload(tmp_path, config):
    import ast
    import inspect

    import silent_cascade.eventflow.checkpoint as checkpoints
    import silent_cascade.eventflow.checkpoint_rng as rng_codec

    path, _, _, _ = archive(tmp_path, config)
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + length])
    assert set(header["__metadata__"]) == {"runtime"}
    assert all(
        value["dtype"] in {"F32", "U8"} for key, value in header.items() if key != "__metadata__"
    )
    assert str(tmp_path).encode() not in raw
    for module in (checkpoints, rng_codec):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert ast.unparse(node.func) not in {
                    "torch.save",
                    "torch.load",
                    "__import__",
                    "importlib.import_module",
                    "pickle.loads",
                    "pickle.load",
                }


def test_mps_origin_archive_loads_on_cpu_without_claiming_runnable_cpu_state(
    tmp_path, config, monkeypatch
):
    from silent_cascade.eventflow.checkpoint import load_runtime_checkpoint, restore_runtime_session

    path, _, _, caller = archive(tmp_path, config)
    payload, tensors = unpack(path)
    payload["original_device"] = "mps"
    rewrite(path, payload, tensors)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    before, counters = rng_signature(), caller.compute_counters()
    loaded = load_runtime_checkpoint(path, config=config, source_revision=REVISION)
    assert loaded.metadata.original_device == "mps"
    assert all(torch.equal(value, loaded.tensors[name]) for name, value in tensors.items())
    with pytest.raises(ReplayError, match="same-device"):
        restore_runtime_session(loaded, caller, config=config, source_revision=REVISION)
    assert rng_signature() == before and caller.compute_counters() == counters


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="native MPS required")
def test_native_mps_rng_validation_uses_isolated_generator(tmp_path, config, monkeypatch):
    from silent_cascade.eventflow.checkpoint import restore_runtime_session
    from silent_cascade.eventflow.checkpoint_rng import restore_rng_snapshot
    from silent_cascade.rng import snapshot_global_rng

    _, artifact, _, _ = archive(tmp_path, config)
    restore_runtime_session(artifact, config=config, source_revision=REVISION, restore_mps=True)
    snapshot = snapshot_global_rng()
    before = rng_signature()

    # A malformed archived MPS state must be rejected before any global setter.
    def forbidden(*args, **kwargs):
        pytest.fail("MPS validation touched a global generator")

    monkeypatch.setattr(random, "setstate", forbidden)
    monkeypatch.setattr(np.random, "set_state", forbidden)
    monkeypatch.setattr(torch, "set_rng_state", forbidden)
    from dataclasses import replace

    with pytest.raises(ReplayError):
        restore_rng_snapshot(
            replace(snapshot, torch_mps_state=torch.zeros(1, dtype=torch.uint8)), restore_mps=True
        )
    assert rng_signature() == before


def test_missing_pending_activation_cannot_be_hidden_by_rehash(tmp_path, config):
    from silent_cascade.eventflow.checkpoint import (
        load_runtime_checkpoint,
        publish_runtime_checkpoint,
        snapshot_runtime,
    )

    agent = ScriptedEventFlowAgent()
    session = EventEngine(config).start_episode(bundle_for(CASES[0]), agent)
    checkpoint = snapshot_runtime(session, agent, config=config, source_revision=REVISION)
    path = tmp_path / "initial.safetensors"
    publish_runtime_checkpoint(path, checkpoint)
    payload, tensors = unpack(path)
    payload["external_queue"] = [
        row for row in payload["external_queue"] if row["kind"] != "activate"
    ]
    rewrite(path, payload, tensors)
    with pytest.raises(ReplayError):
        load_runtime_checkpoint(path, config=config, source_revision=REVISION)


def test_successful_crash_enabled_execution_does_not_encode_archives(tmp_path, config, monkeypatch):
    import silent_cascade.eventflow.checkpoint as checkpoints

    def forbidden(*args, **kwargs):
        pytest.fail("successful execution encoded a safetensors archive")

    monkeypatch.setattr(checkpoints, "save", forbidden)
    result = EventEngine(config, crash_root=tmp_path, source_revision=REVISION).run_episode(
        bundle_for(CASES[0]), ScriptedEventFlowAgent()
    )
    assert result.score.timed_success
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "damage",
    [
        "fact_memory",
        "cached_root",
        "historical_root",
        "historical_flow_counter",
        "terminal_success",
        "terminal_count",
        "terminal_identity",
        "terminal_time",
    ],
)
def test_rehashed_history_root_and_score_corruption_refused_before_any_setter(
    tmp_path, config, monkeypatch, damage
):
    from silent_cascade.eventflow.checkpoint import (
        RuntimeCheckpointArtifact,
        RuntimeCheckpointMetadata,
        restore_runtime_session,
        snapshot_runtime,
    )

    engine, agent = EventEngine(config), ScriptedEventFlowAgent()
    session = engine.start_episode(bundle_for(CASES[0]), agent)
    if damage.startswith("terminal"):
        while not engine.step(session, agent):
            pass
    else:
        engine.run_until(session, agent, 1.1)
    artifact = snapshot_runtime(session, agent, config=config, source_revision=REVISION)
    payload = artifact.metadata.model_dump(mode="json")
    if damage == "historical_root":
        from dataclasses import replace

        from silent_cascade.eventflow.guards import allowed_mode_mask, prediction_snapshot_sha256
        from silent_cascade.logging.trace import tensor_sha256

        # Change the historical ACTIVATE segment's RECALL rate while leaving
        # its recorded RECALL time fixed; coherently bind all affected hashes.
        name = "trajectory.5.guard_rates"
        artifact.tensors[name][0] *= 0.8
        anchor = session.trajectory.checkpoint_snapshots()[5]
        parameters = replace(anchor.segment.parameters, guard_rates=artifact.tensors[name])
        digest = prediction_snapshot_sha256(
            anchor.segment.origin,
            parameters,
            started_at=anchor.segment.started_at,
            allowed_mode_mask=allowed_mode_mask(anchor.core.mode),
            parent_event_id=anchor.segment.parent_event_id,
        )
        payload["trajectory"][5]["prediction_snapshot_sha256"] = digest
        segment = payload["trace"]["events"][4]["segment"]
        segment["prediction_snapshot_sha256"] = digest
        for entry in segment["parameter_hashes"]:
            if entry[0] == "guard_rates":
                entry[1] = tensor_sha256(artifact.tensors[name])
        payload["trace"]["events"][5]["prediction_snapshot_sha256"] = digest
        payload["tensors"][name]["sha256"] = sha256_bytes(
            canonical_json_bytes(
                {"name": name, "tensor_sha256": tensor_sha256(artifact.tensors[name])}
            )
        )
    elif damage == "historical_flow_counter":
        # This remains internally monotone and trace-delta consistent, but
        # invents a causal flow evaluation in the first executed FACT.
        for item in payload["trajectory"][1:]:
            item["core"]["counters"]["flow_evaluations"] += 1
        payload["state"]["core"]["counters"]["flow_evaluations"] += 1
        for row in payload["trace"]["events"]:
            row["counters_after"]["flow_evaluations"] += 1
        payload["trace"]["events"][0]["counter_delta"]["flow_evaluations"] += 1
    elif damage == "fact_memory":
        payload["trace"]["events"][0]["payload"]["node"] = 63
    elif damage == "cached_root":
        payload["prediction_cache"]["event"]["timestamp"] += 0.1
        payload["prediction_cache"]["event"]["predicted_delta"] += 0.1
    elif damage == "terminal_success":
        payload["terminal_score"]["timed_success"] = False
    elif damage == "terminal_count":
        payload["terminal_score"]["action_count"] = 999
    elif damage == "terminal_identity":
        payload["truth"]["private_terminal"]["event_id"] += 100
    elif damage == "terminal_time":
        payload["truth"]["episode_delay"] = 3.0
        payload["truth"]["private_terminal"]["timestamp"] = 4.0
        payload["truth"]["action_window_start"] = 3.25
        payload["truth"]["action_window_end"] = 3.7
        payload["truth"]["action_target"] = 3.475
    online = {
        "schema_version": 1,
        "events": [
            {key: value for key, value in row.items() if key != "source"}
            for row in payload["trace"]["events"]
        ],
    }
    payload["trace"]["sha256"] = sha256_bytes(canonical_json_bytes(online))
    payload["metadata_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {key: value for key, value in payload.items() if key != "metadata_sha256"}
        )
    )
    corrupt = RuntimeCheckpointArtifact(
        RuntimeCheckpointMetadata.model_validate_json(canonical_json_bytes(payload)),
        artifact.tensors,
    )
    caller = ScriptedEventFlowAgent()
    before, counters = rng_signature(), caller.compute_counters()

    def forbidden(*args, **kwargs):
        pytest.fail("corrupt checkpoint reached a global RNG or caller counter setter")

    monkeypatch.setattr(random, "setstate", forbidden)
    monkeypatch.setattr(np.random, "set_state", forbidden)
    monkeypatch.setattr(torch, "set_rng_state", forbidden)
    monkeypatch.setattr(ScriptedEventFlowAgent, "restore_compute_counters", forbidden)
    with pytest.raises(ReplayError):
        restore_runtime_session(corrupt, caller, config=config, source_revision=REVISION)
    assert rng_signature() == before and caller.compute_counters() == counters
