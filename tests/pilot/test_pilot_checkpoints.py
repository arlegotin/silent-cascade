import pytest
import torch


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_real_next_update_matches_after_restore(tmp_path, device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("native MPS unavailable")

    def same(actual, expected):
        if device == "cpu":
            assert torch.equal(actual, expected)
        else:
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.train.pilot_checkpoints import load_pilot_checkpoint, save_pilot_checkpoint
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_data import next_pilot_batch
    from silent_cascade.train.pilot_provenance import PilotSourceIdentity
    from silent_cascade.train.pilot_state import PilotProgress
    from silent_cascade.train.pilot_trainer import pilot_train_one_step
    from silent_cascade.train.trainer import make_optimizer

    config = resolve_pilot_config("phase4_smoke")
    source = PilotSourceIdentity(
        source_commit="a" * 40,
        source_files={"test": "b" * 64},
        source_sha256="c" * 64,
        plan_revision="d" * 40,
        plan_sha256="e" * 64,
        spec_sha256="f" * 64,
        config_sha256=config.sha256,
    )
    torch.manual_seed(11)
    model = EventFlowModel(config.config.neural).to(device)
    optimizer = make_optimizer(model, config.config.pilot)
    pilot_train_one_step(
        model,
        optimizer,
        next_pilot_batch(config.config, stage="one_hop", batch_counter=0).to(device),
        config.config,
    )
    path = tmp_path / "checkpoint.safetensors"
    digest = save_pilot_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        progress=PilotProgress(global_step=1, batch_counter=1),
        config=config,
        source=source,
    )
    expected_random = torch.rand(5)
    expected_device_random = torch.rand(5, device=device)
    batch = next_pilot_batch(config.config, stage="one_hop", batch_counter=1).to(device)
    expected = pilot_train_one_step(model, optimizer, batch, config.config)
    restored = load_pilot_checkpoint(
        path, expected_sha256=digest, config=config, source=source, device=device
    )
    assert torch.equal(torch.rand(5), expected_random)
    assert torch.equal(torch.rand(5, device=device), expected_device_random)
    actual = pilot_train_one_step(restored.model, restored.optimizer, batch, config.config)
    assert actual.loss == (
        expected.loss if device == "cpu" else pytest.approx(expected.loss, rel=1e-5, abs=1e-6)
    )
    for a, b in zip(model.parameters(), restored.model.parameters(), strict=True):
        same(a, b)
    restored_parameters = dict(restored.model.named_parameters())
    for name, parameter in model.named_parameters():
        expected_state = optimizer.state.get(parameter, {})
        actual_state = restored.optimizer.state.get(restored_parameters[name], {})
        assert expected_state.keys() == actual_state.keys()
        for k, value in expected_state.items():
            same(actual_state[k], value)
    snapshot = torch.get_rng_state().clone()
    from silent_cascade.train.state import TrainingError

    with pytest.raises(TrainingError):
        load_pilot_checkpoint(
            path, expected_sha256="0" * 64, config=config, source=source, device=device
        )
    assert torch.equal(snapshot, torch.get_rng_state())
    from dataclasses import replace

    from silent_cascade.eventflow.neural_weights import decode_archive, encode_archive
    from silent_cascade.hashing import sha256_bytes
    from silent_cascade.train.pilot_checkpoints import PilotCheckpoint

    metadata, tensors = decode_archive(path.read_bytes(), digest, PilotCheckpoint, "pilot_training")
    bad_weights = metadata.weights.model_copy(
        update={"identity": replace(metadata.weights.identity, source_revision="f" * 40)}
    )
    raw = encode_archive(
        tensors, metadata.model_copy(update={"weights": bad_weights}), "pilot_training"
    )
    wrong_source = tmp_path / "wrong-source.safetensors"
    wrong_source.write_bytes(raw)
    with pytest.raises(TrainingError, match="source"):
        load_pilot_checkpoint(
            wrong_source,
            expected_sha256=sha256_bytes(raw),
            config=config,
            source=source,
            device=device,
        )
    assert torch.equal(snapshot, torch.get_rng_state())
    bad_group = metadata.groups[0].model_copy(
        update={"options": {**metadata.groups[0].options, "maximize": 0}}
    )
    raw = encode_archive(
        tensors, metadata.model_copy(update={"groups": (bad_group,)}), "pilot_training"
    )
    wrong_options = tmp_path / "wrong-options.safetensors"
    wrong_options.write_bytes(raw)
    with pytest.raises(TrainingError, match="options"):
        load_pilot_checkpoint(
            wrong_options,
            expected_sha256=sha256_bytes(raw),
            config=config,
            source=source,
            device=device,
        )
    assert torch.equal(snapshot, torch.get_rng_state())


def test_best_three_latest_one_retains_adverse_evidence(tmp_path):
    from silent_cascade.hashing import sha256_bytes
    from silent_cascade.train.pilot_state import PilotCheckpointDescriptor
    from silent_cascade.train.pilot_trainer import retain_checkpoints

    descriptors = []
    for step in range(1, 6):
        raw = str(step).encode()
        digest = sha256_bytes(raw)
        name = f"training-{step}-{digest}.safetensors"
        (tmp_path / name).write_bytes(raw)
        descriptors.append(
            PilotCheckpointDescriptor(
                path=name,
                sha256=digest,
                model_state_sha256="a" * 64,
                global_step=step,
                stage="primary",
                rank=(float(6 - step), 0.0, -float(step)),
            )
        )
    (tmp_path / "adverse.json").write_text("failure")
    retain_checkpoints(tmp_path, descriptors, descriptors[-1])
    assert [d.global_step for d in descriptors if (tmp_path / d.path).exists()] == [1, 2, 3, 5]
    assert (tmp_path / "adverse.json").read_text() == "failure"


def test_retention_keeps_publication_and_pruning_in_pinned_parent(tmp_path, monkeypatch):
    import silent_cascade.train.pilot_trainer as trainer
    from silent_cascade.hashing import sha256_bytes
    from silent_cascade.train.pilot_state import PilotCheckpointDescriptor

    run, parked, outside = (tmp_path / p for p in ("run", "parked", "outside"))
    run.mkdir()
    outside.mkdir()
    descriptors = []
    for step in (0, 1):
        raw = str(step).encode()
        digest = sha256_bytes(raw)
        name = f"training-{step}-{digest}.safetensors"
        (run / name).write_bytes(raw)
        (outside / name).write_bytes(raw)
        descriptors.append(
            PilotCheckpointDescriptor(
                path=name,
                sha256=digest,
                model_state_sha256="a" * 64,
                global_step=step,
                stage="one_hop",
            )
        )
    original = trainer._write_index

    def swap_after_publication(*args):
        original(*args)
        run.rename(parked)
        run.symlink_to(outside, target_is_directory=True)

    monkeypatch.setattr(trainer, "_write_index", swap_after_publication)
    trainer.retain_checkpoints(run, descriptors, descriptors[-1])
    assert (outside / descriptors[0].path).read_bytes() == b"0"
    assert not (parked / descriptors[0].path).exists()
    assert (parked / descriptors[1].path).is_file()
    assert (parked / "checkpoint-index.json").is_file()
    assert not (outside / "checkpoint-index.json").exists()


def test_durable_publication_and_cleanup_share_pinned_parents(tmp_path, monkeypatch):
    import silent_cascade.train.pilot_trainer as trainer
    from silent_cascade.hashing import sha256_bytes

    run, parked, outside = (tmp_path / p for p in ("run", "parked", "outside"))
    (run / "attempt-test").mkdir(parents=True)
    (outside / "attempt-test").mkdir(parents=True)
    pending = run / "attempt-test/checkpoint-0.safetensors"
    pending.write_bytes(b"owned archive")
    unrelated = outside / "attempt-test/checkpoint-0.safetensors"
    unrelated.write_bytes(b"unrelated archive")
    original = trainer._publish_bytes_at

    def swap_before_publication(*args, **kwargs):
        run.rename(parked)
        run.symlink_to(outside, target_is_directory=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(trainer, "_publish_bytes_at", swap_before_publication)
    name = trainer._publish_checkpoint(run, pending, 0, sha256_bytes(b"owned archive"))
    assert (parked / name).read_bytes() == b"owned archive"
    assert not (parked / "attempt-test/checkpoint-0.safetensors").exists()
    assert unrelated.read_bytes() == b"unrelated archive"
    assert not (outside / name).exists()


def test_selected_descriptor_is_hash_bound_immutable_and_reusable(tmp_path):
    """Publication-only synthetic progress, not production learning evidence."""
    import silent_cascade.train.pilot_trainer as trainer
    from silent_cascade.eventflow.neural import NeuralModelIdentity
    from silent_cascade.eventflow.neural_weights import save_neural_weights
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_state import PilotCheckpointDescriptor, PilotProgress

    model = EventFlowModel(resolve_pilot_config("phase4_smoke").config.neural)
    identity = NeuralModelIdentity.from_model(model, source_revision="a" * 40)
    weights = tmp_path / "weights.safetensors"
    digest = save_neural_weights(weights, model=model, identity=identity)
    descriptor = PilotCheckpointDescriptor(
        path=weights.name,
        sha256=digest,
        model_state_sha256=identity.model_state_sha256,
        global_step=4000,
        stage="robustness",
        rank=(0.95, -0.01, -4000.0),
        eligible=True,
    )
    progress = PilotProgress(
        global_step=4000,
        batch_counter=4000,
        stage="robustness",
        status="robustness_complete",
        selected=descriptor,
    )
    config = resolve_pilot_config("phase4_pilot")
    trainer._publish_selected(tmp_path, progress, config)
    selected = tmp_path / "selected.json"
    assert selected.read_bytes() == canonical_json_bytes(descriptor)
    actual = PilotCheckpointDescriptor.model_validate_json(selected.read_bytes())
    assert actual.sha256 == sha256_bytes((tmp_path / actual.path).read_bytes())
    assert actual.model_state_sha256 == identity.model_state_sha256
    trainer._publish_selected(tmp_path, progress, config)
    changed = progress.model_copy(
        update={"selected": descriptor.model_copy(update={"rank": (0.96, -0.01, -4000.0)})}
    )
    with pytest.raises(ValueError, match="refusing overwrite"):
        trainer._publish_selected(tmp_path, changed, config)
    assert selected.read_bytes() == canonical_json_bytes(descriptor)
