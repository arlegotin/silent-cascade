from dataclasses import replace

import pytest
import torch

from silent_cascade.models.errors import NeuralError


def test_core_digest_preserves_legacy_bytes(model):
    from silent_cascade.models.weights import model_layout, model_state_sha256
    from silent_cascade.train.checkpoints import _model_hash, _snapshot

    tensors, aliases, _ = _snapshot(model)
    assert model_layout(model)[1] == aliases
    assert model_state_sha256(tensors, aliases) == _model_hash(tensors, aliases)

    # Execute the original hash/layout bodies from the reviewed parent revision.
    import ast
    import hashlib
    import subprocess

    from silent_cascade.hashing import canonical_json_bytes

    source = subprocess.check_output(
        [
            "git",
            "show",
            "78d80170ed16e5f007aa647400df174262d7b04d:src/silent_cascade/train/checkpoints.py",
        ],
        text=True,
    )
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in {"_layout", "_model_hash"}
    ]
    namespace = {
        "torch": torch,
        "EventFlowModel": type(model),
        "hashlib": hashlib,
        "canonical_json_bytes": canonical_json_bytes,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "legacy_weights", "exec"), namespace)
    assert namespace["_layout"](model)[1] == aliases
    assert namespace["_model_hash"](tensors, aliases) == model_state_sha256(tensors, aliases)


@pytest.mark.parametrize(
    "mutation", ["version", "storage", "parameter", "alias", "buffer", "buffer_version"]
)
def test_mutated_model_rejected_on_reuse(runtime_case, controlled_model, mutation):
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent

    agent = NeuralEventFlowAgent(controlled_model, identity=runtime_case.identity, device="cpu")
    parameter = controlled_model.mode_embedding.weight
    with torch.no_grad():
        if mutation == "version":
            parameter.add_(1)
        elif mutation == "storage":
            parameter.data = parameter.data.clone()
        elif mutation == "parameter":
            controlled_model.mode_embedding.weight = torch.nn.Parameter(parameter.clone())
        elif mutation == "alias":
            object.__setattr__(
                controlled_model.external_encoder, "_injected_record_encoder", torch.nn.Identity()
            )
        elif mutation == "buffer":
            controlled_model.register_buffer("new_buffer", torch.zeros(1))
        else:
            controlled_model.controller._allowed_guard_masks.logical_not_()
    with pytest.raises(NeuralError, match="mutat"):
        agent.initialize(runtime_case.init)


def test_wrong_identity_and_nonfinite_weights_rejected(runtime_case, controlled_model):
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent

    with pytest.raises(NeuralError, match="identity"):
        NeuralEventFlowAgent(
            controlled_model,
            identity=replace(runtime_case.identity, model_state_sha256="0" * 64),
            device="cpu",
        )
    with torch.no_grad():
        controlled_model.mode_embedding.weight[0, 0] = float("nan")
    with pytest.raises(NeuralError, match="nonfinite"):
        NeuralEventFlowAgent(controlled_model, identity=runtime_case.identity, device="cpu")


def test_reuse_never_hashes_or_serializes_weights(runtime_case, controlled_model, monkeypatch):
    import silent_cascade.models.weights as weights
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent

    agent = NeuralEventFlowAgent(controlled_model, identity=runtime_case.identity, device="cpu")

    def forbidden(*args, **kwargs):
        pytest.fail("weight serialization during episode")

    monkeypatch.setattr(weights, "model_state_sha256", forbidden)
    monkeypatch.setattr(controlled_model, "state_dict", forbidden)
    runtime_case.run(agent)
    runtime_case.run(agent)


def test_wrong_configuration_identity_is_rejected(runtime_case, controlled_model):
    import json

    from silent_cascade.eventflow.neural import NeuralEventFlowAgent, NeuralModelIdentity

    with pytest.raises(NeuralError, match="identity"):
        NeuralModelIdentity(
            json.dumps({"truth": 3}), runtime_case.identity.model_state_sha256, "a" * 40
        )
    from silent_cascade.hashing import canonical_json_bytes
    from silent_cascade.models.config import NeuralModelConfig

    debug = NeuralModelConfig(
        architecture_profile="debug",
        record_hidden_dim=96,
        external_hidden_dim=64,
        query_dim=64,
        controller_hidden_dim=128,
        jump_hidden_dim=128,
        head_hidden_dim=64,
    )
    with pytest.raises(NeuralError, match="identity"):
        NeuralEventFlowAgent(
            controlled_model,
            identity=replace(
                runtime_case.identity, model_config_json=canonical_json_bytes(debug).decode()
            ),
            device="cpu",
        )
