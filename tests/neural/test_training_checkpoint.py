"""Archive adversarial tests exercise real safe bytes and filesystem boundaries."""

import hashlib
import importlib
import json
import struct

import numpy as np
import pytest
import torch
from safetensors.torch import load, save

from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.rng import snapshot_global_rng
from silent_cascade.train.state import CheckpointDescriptor, TrainingError, TrainProgress

SOURCE = "a" * 40


def codec():
    return importlib.import_module("silent_cascade.train.checkpoints")


def progress(step=0, metric=None):
    return TrainProgress(
        optimizer_step=step,
        next_batch_counter=step,
        stage="one_hop",
        train_root_seed=311,
        train_public_id_seed=331,
        best_metric=metric,
        best_step=step if metric is not None else None,
        patience_counter=0,
        retained_checkpoints=(),
        validation_manifest_sha256="b" * 64,
    )


def optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
        fused=False,
    )


def unpack(path):
    raw = path.read_bytes()
    header_size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_size])
    return json.loads(header["__metadata__"]["training"]), load(raw)


def rewrite(path, metadata, tensors):
    raw = save(tensors, metadata={"training": json.dumps(metadata)})
    replacement = path.parent / ("training-" + hashlib.sha256(raw).hexdigest() + ".safetensors")
    replacement.write_bytes(raw)
    return replacement


@pytest.fixture
def saved(tmp_path, resolved_neural_config):
    module = codec()
    model = EventFlowModel(resolved_neural_config.config.neural)
    opt = optimizer(model)
    descriptor = module.save_training_checkpoint(
        tmp_path, model, opt, progress(), config=resolved_neural_config, source_commit=SOURCE
    )
    path = tmp_path / descriptor.relative_path
    return module, path, descriptor, model, opt


def restore(module, path, config, **kwargs):
    return module.load_training_checkpoint(
        path,
        expected_config_sha256=config.sha256,
        expected_source_commit=SOURCE,
        device="cpu",
        **kwargs,
    )


def test_empty_optimizer_shared_names_and_explicit_rng_restore(saved, resolved_neural_config):
    module, path, descriptor, model, opt = saved
    before = snapshot_global_rng()
    restored = restore(module, path, resolved_neural_config)
    assert torch.equal(before.torch_cpu_state, torch.get_rng_state())
    assert restored.optimizer.state == opt.state == {}
    assert restored.progress == progress()
    assert restored.config.config == resolved_neural_config.config
    assert descriptor.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    metadata, tensors = unpack(path)
    assert metadata["schema_version"] == "phase3-training-checkpoint-v1"
    assert set(k for k in tensors if k.startswith("parameter/")) == {
        "parameter/" + name for name, _ in model.named_parameters()
    }
    assert set(metadata["aliases"]) == set(model.state_dict())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[key], rtol=0, atol=0)
    assert len(list(restored.model.parameters())) == len(list(model.parameters()))


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "phase3-training-checkpoint-v2"),
        ("source_commit", "c" * 40),
        ("config_sha256", "d" * 64),
        ("extra", 1),
        ("optimizer_step", True),
    ],
)
def test_rejects_bad_metadata_without_rng_mutation(saved, resolved_neural_config, field, value):
    module, path, _, model, _ = saved
    metadata, tensors = unpack(path)
    if field == "optimizer_step":
        metadata["progress"][field] = value
    else:
        metadata[field] = value
    path = rewrite(path, metadata, tensors)
    before = snapshot_global_rng()
    weights = {key: value.clone() for key, value in model.state_dict().items()}
    with pytest.raises(TrainingError):
        restore(module, path, resolved_neural_config)
    assert torch.equal(before.torch_cpu_state, torch.get_rng_state())
    for key, value in model.state_dict().items():
        assert torch.equal(weights[key], value)


@pytest.mark.parametrize("mutation", ["shape", "dtype", "nan", "missing", "extra", "alias", "rng"])
def test_rejects_invalid_tensors_and_aliases_transactionally(
    saved, resolved_neural_config, mutation
):
    module, path, _, _, _ = saved
    metadata, tensors = unpack(path)
    key = next(k for k in tensors if k.startswith("parameter/"))
    if mutation == "shape":
        tensors[key] = tensors[key].reshape(-1)[:0]
    elif mutation == "dtype":
        tensors[key] = tensors[key].double()
    elif mutation == "nan":
        tensors[key].view(-1)[0] = float("nan")
    elif mutation == "missing":
        del tensors[key]
    elif mutation == "extra":
        tensors["unknown"] = torch.zeros(1)
    elif mutation == "alias":
        metadata["aliases"][next(iter(metadata["aliases"]))] = "parameter/unknown"
    else:
        tensors["rng.cpu"] = torch.zeros(2, dtype=torch.uint8)
    path = rewrite(path, metadata, tensors)
    before = snapshot_global_rng()
    with pytest.raises(TrainingError):
        restore(module, path, resolved_neural_config)
    assert torch.equal(before.torch_cpu_state, torch.get_rng_state())


@pytest.mark.parametrize(
    "mutation", ["truncated", "oversized", "deep", "metadata_limit", "duplicate"]
)
def test_rejects_bounded_header_before_tensor_load(saved, resolved_neural_config, mutation):
    module, path, _, _, _ = saved
    if mutation == "truncated":
        path.write_bytes(path.read_bytes()[:-1])
    elif mutation == "oversized":
        with path.open("r+b") as stream:
            stream.truncate(256 * 1024 * 1024 + 1)
    elif mutation == "deep":
        raw = ('{"x":' * 70 + "0" + "}" * 70).encode()
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    elif mutation == "metadata_limit":
        path.write_bytes(struct.pack("<Q", 8 * 1024 * 1024 + 1))
    else:
        raw = b'{"__metadata__":{},"__metadata__":{}}'
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    with pytest.raises(TrainingError):
        restore(module, path, resolved_neural_config)


def test_no_clobber_symlinks_and_identical_reuse(saved, resolved_neural_config, tmp_path):
    module, path, descriptor, model, opt = saved
    # Saving must not itself advance any global generator, so these bytes match.
    assert (
        module.save_training_checkpoint(
            path.parent, model, opt, progress(), config=resolved_neural_config, source_commit=SOURCE
        )
        == descriptor
    )
    path.write_bytes(b"conflicting existing archive")
    before = path.read_bytes()
    with pytest.raises(TrainingError):
        module.save_training_checkpoint(
            path.parent, model, opt, progress(), config=resolved_neural_config, source_commit=SOURCE
        )
    assert path.read_bytes() == before
    link = tmp_path / "link.safetensors"
    link.symlink_to(path)
    with pytest.raises(TrainingError):
        restore(module, link, resolved_neural_config)
    parent_link = tmp_path / "parent"
    parent_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(TrainingError):
        module.export_weights(
            parent_link, model, config=resolved_neural_config, source_commit=SOURCE
        )


def test_export_is_portable_and_hash_bound(saved, resolved_neural_config, tmp_path):
    module, _, _, model, _ = saved
    descriptor = module.export_weights(
        tmp_path, model, config=resolved_neural_config, source_commit=SOURCE
    )
    path = tmp_path / descriptor.relative_path
    restored = module.load_weights(path, expected_sha256=descriptor.file_sha256, device="cpu")
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)
    with pytest.raises(TrainingError):
        module.load_weights(path, expected_sha256="0" * 64, device="cpu")


@pytest.mark.parametrize("path", ["x//y", "x/./y", "./x", "x/", "../x", "/x", "x\\y"])
def test_descriptor_rejects_path_alias_spellings(path):
    with pytest.raises(ValueError):
        CheckpointDescriptor(
            relative_path=path,
            file_sha256="a" * 64,
            model_state_sha256="b" * 64,
            config_sha256="c" * 64,
            source_commit=SOURCE,
            optimizer_step=0,
            stage="one_hop",
        )


def test_index_retains_latest_best_three_and_protected_evidence(
    saved, resolved_neural_config, tmp_path
):
    module, _, _, model, opt = saved
    run = tmp_path / "run"
    run.mkdir()
    descriptors = []
    for step, metric in enumerate((0.9, 0.8, 0.7, 0.6, 0.5)):
        # Empty optimizer can be valid for untouched parameters at any global step.
        descriptor = module.save_training_checkpoint(
            run,
            model,
            opt,
            progress(step, metric),
            config=resolved_neural_config,
            source_commit=SOURCE,
        )
        descriptors.append(descriptor)
        index = module.update_checkpoint_index(
            run, descriptor, protected=(descriptors[3],) if step >= 3 else ()
        )
    assert index.latest == descriptors[4]
    assert index.best == tuple(descriptors[:3])
    unrelated = run / "unrelated.safetensors"
    unrelated.write_bytes(b"unrelated")
    assert module.prune_training_checkpoints(run) == ()
    assert unrelated.read_bytes() == b"unrelated"
    assert all((run / d.relative_path).is_file() for d in descriptors)
    module.update_checkpoint_index(run, descriptors[4], protected=())
    # Protection is durable: an omitted prior pin cannot remove evidence protection.
    assert module.prune_training_checkpoints(run) == ()


def test_index_prunes_only_authenticated_owned_unreferenced_archives(
    saved, resolved_neural_config, tmp_path
):
    module, _, _, model, opt = saved
    run = tmp_path / "run"
    run.mkdir()
    descriptors = []
    for step in range(5):
        item = module.save_training_checkpoint(
            run,
            model,
            opt,
            progress(step, float(step) / 10),
            config=resolved_neural_config,
            source_commit=SOURCE,
        )
        descriptors.append(item)
        module.update_checkpoint_index(run, item)
    removed = module.prune_training_checkpoints(run)
    assert set(removed) == {d.relative_path for d in descriptors[:2]}
    assert all((run / d.relative_path).exists() for d in descriptors[2:])
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    copied = foreign / descriptors[4].relative_path
    copied.write_bytes((run / descriptors[4].relative_path).read_bytes())
    with pytest.raises(TrainingError):
        module.update_checkpoint_index(foreign, descriptors[4])


def test_current_evaluated_metric_is_distinct_from_historical_best(
    saved, resolved_neural_config, tmp_path
):
    module, _, _, model, opt = saved
    evaluated = progress(4, 0.9).model_dump()
    evaluated.update(best_step=2, validation_metric=0.8)
    evaluated = TrainProgress(**evaluated)
    descriptor = module.save_training_checkpoint(
        tmp_path, model, opt, evaluated, config=resolved_neural_config, source_commit=SOURCE
    )
    assert descriptor.validation_metric == 0.8
    restored = restore(module, tmp_path / descriptor.relative_path, resolved_neural_config)
    assert restored.progress.best_metric == 0.9
    assert restored.progress.best_step == 2
    assert restored.progress.validation_metric == 0.8


def test_missing_owned_archive_cannot_reset_or_replace_index(
    saved, resolved_neural_config, tmp_path
):
    module, _, _, model, opt = saved
    first = module.save_training_checkpoint(
        tmp_path, model, opt, progress(), config=resolved_neural_config, source_commit=SOURCE
    )
    module.update_checkpoint_index(tmp_path, first)
    before = (tmp_path / "checkpoint-index.json").read_bytes()
    (tmp_path / first.relative_path).unlink()
    second = module.save_training_checkpoint(
        tmp_path, model, opt, progress(1), config=resolved_neural_config, source_commit=SOURCE
    )
    with pytest.raises(TrainingError):
        module.update_checkpoint_index(tmp_path, second)
    assert (tmp_path / "checkpoint-index.json").read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "shape",
        "dtype",
        "nan",
        "negative",
        "step",
        "step_bool",
        "group",
        "option",
        "bool_option",
    ],
)
def test_malformed_named_optimizer_state_is_rejected(saved, resolved_neural_config, mutation):
    module, path, _, model, opt = saved
    # A real optimizer step on all named parameters, independent of the full trainer.
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    opt.step()
    descriptor = module.save_training_checkpoint(
        path.parent, model, opt, progress(1), config=resolved_neural_config, source_commit=SOURCE
    )
    path = path.parent / descriptor.relative_path
    metadata, tensors = unpack(path)
    key = next(
        key for key in tensors if key.startswith("optimizer/") and key.endswith("/exp_avg_sq")
    )
    if mutation == "missing":
        del tensors[key]
    elif mutation == "extra":
        tensors[key + "_unknown"] = tensors[key].clone()
    elif mutation == "shape":
        tensors[key] = torch.zeros(0)
    elif mutation == "dtype":
        tensors[key] = tensors[key].double()
    elif mutation == "nan":
        tensors[key].view(-1)[0] = float("nan")
    elif mutation == "negative":
        tensors[key].view(-1)[0] = -1.0
    elif mutation in {"step", "step_bool"}:
        step_key = key.removesuffix("exp_avg_sq") + "step"
        tensors[step_key] = torch.tensor(2.0) if mutation == "step" else torch.tensor(True)
    elif mutation == "group":
        metadata["groups"][0]["names"].append(metadata["groups"][0]["names"][0])
    elif mutation == "option":
        metadata["groups"][0]["options"]["lr"] = 0.1
    else:
        metadata["groups"][0]["options"]["foreach"] = 0
    path = rewrite(path, metadata, tensors)
    before = snapshot_global_rng()
    with pytest.raises(TrainingError):
        restore(module, path, resolved_neural_config)
    assert torch.equal(before.torch_cpu_state, torch.get_rng_state())


def test_tensor_header_is_validated_before_safetensors_allocation(
    saved, resolved_neural_config, monkeypatch
):
    module, path, _, _, _ = saved
    metadata, tensors = unpack(path)
    tensors["parameter/unknown"] = torch.zeros(1)
    path = rewrite(path, metadata, tensors)

    def forbidden(_):
        raise AssertionError("unvalidated tensor allocation")

    monkeypatch.setattr(module, "load_tensors", forbidden)
    with pytest.raises(TrainingError, match="tensor names"):
        restore(module, path, resolved_neural_config)


def test_explicit_rng_restore_revalidates_all_generators_transactionally(
    saved, resolved_neural_config
):
    module, path, _, _, _ = saved
    restored = restore(module, path, resolved_neural_config)
    restored.rng.torch_cpu_state.resize_(2)
    before = snapshot_global_rng()
    with pytest.raises(TrainingError):
        restored.restore_rng()
    after = snapshot_global_rng()
    assert before.python_state == after.python_state
    assert np.array_equal(before.numpy_state[1], after.numpy_state[1])
    assert torch.equal(before.torch_cpu_state, after.torch_cpu_state)


@pytest.mark.parametrize("field,value", [("numpy_has_gauss", False), ("numpy_cached_gaussian", 0)])
def test_rng_metadata_rejects_coercible_primitives(saved, resolved_neural_config, field, value):
    module, path, _, _, _ = saved
    metadata, tensors = unpack(path)
    metadata["rng"][field] = value
    path = rewrite(path, metadata, tensors)
    with pytest.raises(TrainingError):
        restore(module, path, resolved_neural_config)


@pytest.mark.parametrize("operation", ["update", "prune", "load"])
def test_corrupt_indexed_archive_prevents_any_index_mutation(saved, tmp_path, operation):
    module, path, descriptor, _, _ = saved
    module.update_checkpoint_index(tmp_path, descriptor)
    before = (tmp_path / "checkpoint-index.json").read_bytes()
    path.write_bytes(b"corrupt")
    with pytest.raises(TrainingError):
        if operation == "update":
            module.update_checkpoint_index(tmp_path, descriptor)
        elif operation == "prune":
            module.prune_training_checkpoints(tmp_path)
        else:
            module.load_checkpoint_index(tmp_path)
    assert (tmp_path / "checkpoint-index.json").read_bytes() == before
    assert path.read_bytes() == b"corrupt"


@pytest.mark.parametrize("kind", ["archive", "index", "lock"])
def test_symlinked_index_paths_are_never_followed(saved, tmp_path, kind):
    module, path, descriptor, _, _ = saved
    module.update_checkpoint_index(tmp_path, descriptor)
    target = {
        "archive": path,
        "index": tmp_path / "checkpoint-index.json",
        "lock": tmp_path / ".checkpoint-index.lock",
    }[kind]
    original = target.read_bytes()
    outside = tmp_path / "outside"
    target.rename(outside)
    target.symlink_to(outside)
    with pytest.raises(TrainingError):
        module.prune_training_checkpoints(tmp_path)
    assert outside.read_bytes() == original


def test_protected_checkpoint_cannot_advance_past_latest(saved, resolved_neural_config, tmp_path):
    module, _, first, model, opt = saved
    future = module.save_training_checkpoint(
        tmp_path, model, opt, progress(1), config=resolved_neural_config, source_commit=SOURCE
    )
    with pytest.raises(TrainingError):
        module.update_checkpoint_index(tmp_path, first, protected=(future,))
    assert not (tmp_path / "checkpoint-index.json").exists()


def test_overlapping_offsets_reject_before_tensor_loading(
    saved, resolved_neural_config, monkeypatch
):
    module, path, _, _, _ = saved
    raw = path.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + size])
    name = next(
        name
        for name in header
        if name.startswith("parameter/") and header[name]["data_offsets"][0] > 0
    )
    start, end = header[name]["data_offsets"]
    header[name]["data_offsets"] = [0, end - start]
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    raw = struct.pack("<Q", len(encoded)) + encoded + raw[8 + size :]
    path = path.parent / ("training-" + hashlib.sha256(raw).hexdigest() + ".safetensors")
    path.write_bytes(raw)

    def forbidden(_):
        raise AssertionError("offsets reached tensor allocator")

    monkeypatch.setattr(module, "load_tensors", forbidden)
    with pytest.raises(TrainingError):
        restore(module, path, resolved_neural_config)


@pytest.mark.parametrize("field", ["expected_config_sha256", "expected_source_commit"])
def test_expected_bindings_are_required_exact_hashes(saved, resolved_neural_config, field):
    module, path, _, _, _ = saved
    kwargs = {
        "expected_config_sha256": resolved_neural_config.sha256,
        "expected_source_commit": SOURCE,
        "device": "cpu",
    }
    kwargs[field] = None
    with pytest.raises(TrainingError):
        module.load_training_checkpoint(path, **kwargs)


def test_index_breaks_chain_ties_by_composition_then_earliest_step(
    saved, resolved_neural_config, tmp_path
):
    module, _, _, model, opt = saved
    run = tmp_path / "selection"
    run.mkdir()
    descriptors = []
    for step, (chain, composition) in enumerate(
        ((0.9, 0.8), (0.9, 0.9), (0.9, 0.9), (0.8, 1.0), (0.9, 0.7))
    ):
        values = progress(step, 0.9).model_dump()
        values.update(validation_metric=chain, validation_composition_metric=composition)
        descriptor = module.save_training_checkpoint(
            run,
            model,
            opt,
            TrainProgress(**values),
            config=resolved_neural_config,
            source_commit=SOURCE,
        )
        descriptors.append(descriptor)
        index = module.update_checkpoint_index(run, descriptor)
    assert index.best == (descriptors[1], descriptors[2], descriptors[0])
    assert index.latest == descriptors[4]
    assert descriptors[1].validation_composition_metric == 0.9
