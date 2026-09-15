from pathlib import Path

import pytest
import torch

from silent_cascade.config import resolve_config
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.train.batches import next_training_batch
from silent_cascade.train.config import Phase3Config
from silent_cascade.train.state import TrainingError


@pytest.fixture
def smoke_config():
    return resolve_config(
        Phase3Config,
        tuple(
            Path(p)
            for p in (
                "configs/base.yaml",
                "configs/data/primary.yaml",
                "configs/model/event_flow.yaml",
                "configs/model/neural_components.yaml",
                "configs/train/smoke.yaml",
            )
        ),
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="native MPS unavailable"
            ),
        ),
    ],
)
def test_one_step_updates_parameters_and_reports_backward(smoke_config, device):
    from silent_cascade.train.trainer import make_optimizer, train_one_step

    model = EventFlowModel(smoke_config.config.neural).to(device)
    optimizer = make_optimizer(model, smoke_config.config.training)
    before = next(model.parameters()).detach().clone()
    batch = next_training_batch(smoke_config.config, stage="one_hop", batch_counter=0).to(device)
    result = train_one_step(model, optimizer, batch, smoke_config.config)
    assert result.loss > 0
    assert len(result.terms) == 10
    assert result.compute.backward_macs > 0
    assert not torch.equal(before, next(model.parameters()))


def test_nonfinite_gradient_fails_before_optimizer_update(smoke_config):
    from silent_cascade.train.trainer import make_optimizer, train_one_step

    model = EventFlowModel(smoke_config.config.neural)
    optimizer = make_optimizer(model, smoke_config.config.training)
    parameter = next(model.parameters())
    before = parameter.detach().clone()
    handle = parameter.register_hook(lambda grad: grad * torch.nan)
    batch = next_training_batch(smoke_config.config, stage="one_hop", batch_counter=0)
    with pytest.raises(TrainingError, match="Nonfinite gradient"):
        train_one_step(model, optimizer, batch, smoke_config.config)
    handle.remove()
    assert torch.equal(before, parameter)
    assert not optimizer.state


def test_nonfinite_internal_final_guard_cannot_be_masked(smoke_config, monkeypatch):
    from dataclasses import replace

    from silent_cascade.train import trainer

    model = EventFlowModel(smoke_config.config.neural)
    optimizer = trainer.make_optimizer(model, smoke_config.config.training)
    batch = next_training_batch(smoke_config.config, stage="one_hop", batch_counter=0)
    original = trainer.teacher_forced_unroll

    def corrupt(*args):
        unroll = original(*args)
        parameter = unroll.steps[-1].post_parameters
        import copy

        corrupted = copy.copy(parameter)
        object.__setattr__(corrupted, "flow_targets", parameter.flow_targets * torch.nan)
        return replace(
            unroll, steps=(*unroll.steps[:-1], replace(unroll.steps[-1], post_parameters=corrupted))
        )

    monkeypatch.setattr(trainer, "teacher_forced_unroll", corrupt)
    with pytest.raises(TrainingError, match="Nonfinite internal"):
        trainer.train_one_step(model, optimizer, batch, smoke_config.config)


def test_explicit_device_override_does_not_change_config(smoke_config, tmp_path):
    from silent_cascade.train.trainer import run_training

    manifest = _manifest(smoke_config, tmp_path / "manifest.json")
    result = run_training(
        smoke_config,
        validation_manifest=manifest,
        run_dir=tmp_path / "run",
        source_commit="a" * 40,
        device="cpu",
    )
    assert result.device == "cpu"
    assert result.latest.config_sha256 == smoke_config.sha256
    with pytest.raises(TrainingError, match="device"):
        run_training(
            smoke_config,
            validation_manifest=manifest,
            run_dir=tmp_path / "invalid",
            source_commit="a" * 40,
            device="cuda",
        )


def test_stopping_ceiling_patience_and_strict_gates():
    from silent_cascade.train.trainer import stopping_reason

    assert (
        stopping_reason(step=75000, ceiling=75000, patience=0, scheduled=False, gates_pass=False)
        == "step_ceiling"
    )
    assert (
        stopping_reason(step=15000, ceiling=75000, patience=15, scheduled=True, gates_pass=False)
        == "early_stopping"
    )
    assert (
        stopping_reason(step=1000, ceiling=75000, patience=0, scheduled=True, gates_pass=True)
        == "component_gate"
    )
    assert (
        stopping_reason(step=999, ceiling=75000, patience=0, scheduled=False, gates_pass=True)
        is None
    )


def _manifest(config, path):
    from silent_cascade.train.curriculum_data import (
        ComponentManifest,
        CurriculumKey,
        make_curriculum_example,
    )

    examples = tuple(
        make_curriculum_example(
            config.config, CurriculumKey("ofd-one-hop-v1", "debug", 313, 337, i, "one_hop")
        )
        for i in range(16)
    )
    manifest = ComponentManifest.from_examples(
        examples, config.config, source_revision="a" * 40, plan_revision="b" * 40
    )
    path.write_text(manifest.model_dump_json())
    return path


def test_real_run_resumes_exactly_and_keeps_selected_archive(smoke_config, tmp_path, monkeypatch):
    from silent_cascade.train import trainer
    from silent_cascade.train.checkpoints import load_checkpoint_index, load_training_checkpoint

    manifest = _manifest(smoke_config, tmp_path / "manifest.json")
    expected = trainer.run_training(
        smoke_config,
        validation_manifest=manifest,
        run_dir=tmp_path / "complete",
        source_commit="a" * 40,
    )
    original = trainer.train_one_step
    calls = 0

    def interrupted(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TrainingError("injected interruption")
        return original(*args)

    monkeypatch.setattr(trainer, "train_one_step", interrupted)
    run_dir = tmp_path / "resumed"
    with pytest.raises(TrainingError, match="injected interruption"):
        trainer.run_training(
            smoke_config, validation_manifest=manifest, run_dir=run_dir, source_commit="a" * 40
        )
    prior = load_checkpoint_index(run_dir).latest
    assert prior.optimizer_step == 2
    monkeypatch.setattr(trainer, "train_one_step", original)
    actual = trainer.run_training(
        smoke_config,
        validation_manifest=manifest,
        run_dir=run_dir,
        source_commit="a" * 40,
        resume=run_dir / prior.relative_path,
    )
    assert actual.progress.optimizer_step == 4
    assert actual.progress.next_batch_counter == 4
    assert len(actual.progress.training_journal_sha256) == 64
    assert actual.stop_reason == "step_ceiling"
    assert actual.selected.optimizer_step in (2, 4)
    assert (run_dir / actual.selected.relative_path).is_file()
    assert actual.selected in load_checkpoint_index(run_dir).protected

    def restored(directory, descriptor):
        return load_training_checkpoint(
            directory / descriptor.relative_path,
            expected_config_sha256=smoke_config.sha256,
            expected_source_commit="a" * 40,
            device="cpu",
        )

    left = restored(tmp_path / "complete", expected.latest)
    right = restored(run_dir, actual.latest)
    for a, b in zip(left.model.parameters(), right.model.parameters(), strict=True):
        mps = torch.backends.mps.is_available()
        torch.testing.assert_close(a, b, rtol=1e-4 if mps else 0, atol=1e-5 if mps else 0)
    assert len(actual.validation_history) == 2
    assert list(run_dir.glob("attempt-*/failure.json"))


def test_checkpoint_publication_interruption_replays_uncommitted_segment(
    smoke_config, tmp_path, monkeypatch
):
    from silent_cascade.train import trainer
    from silent_cascade.train.checkpoints import load_checkpoint_index

    manifest = _manifest(smoke_config, tmp_path / "manifest.json")
    run_dir = tmp_path / "run"
    original = trainer.update_checkpoint_index

    def interrupted(directory, descriptor, **kwargs):
        if descriptor.optimizer_step == 2:
            raise TrainingError("interrupted index publication")
        return original(directory, descriptor, **kwargs)

    monkeypatch.setattr(trainer, "update_checkpoint_index", interrupted)
    with pytest.raises(TrainingError, match="interrupted index"):
        trainer.run_training(
            smoke_config, validation_manifest=manifest, run_dir=run_dir, source_commit="a" * 40
        )
    prior = load_checkpoint_index(run_dir).latest
    assert prior.optimizer_step == 0
    monkeypatch.setattr(trainer, "update_checkpoint_index", original)
    result = trainer.run_training(
        smoke_config,
        validation_manifest=manifest,
        run_dir=run_dir,
        source_commit="a" * 40,
        resume=run_dir / prior.relative_path,
    )
    assert result.progress.optimizer_step == 4
    assert len(result.validation_history) == 2
    journal = trainer._read_json(
        run_dir / f"journal-{result.progress.training_journal_sha256}.json"
    )
    step = run_dir / journal["attempt"] / journal["steps"][0]["path"]
    step.write_text("{}")
    with pytest.raises(TrainingError, match="journal"):
        trainer.run_training(
            smoke_config,
            validation_manifest=manifest,
            run_dir=run_dir,
            source_commit="a" * 40,
            resume=run_dir / result.latest.relative_path,
        )


def test_changed_validation_manifest_prevents_resume(smoke_config, tmp_path):
    from silent_cascade.train.trainer import run_training

    manifest = _manifest(smoke_config, tmp_path / "manifest.json")
    run_dir = tmp_path / "run"
    result = run_training(
        smoke_config, validation_manifest=manifest, run_dir=run_dir, source_commit="a" * 40
    )
    with pytest.raises(TrainingError, match="source"):
        run_training(
            smoke_config,
            validation_manifest=manifest,
            run_dir=run_dir,
            source_commit="b" * 40,
            resume=run_dir / result.latest.relative_path,
        )
    changed = resolve_config(
        Phase3Config,
        tuple(
            Path(p)
            for p in (
                "configs/base.yaml",
                "configs/data/primary.yaml",
                "configs/model/event_flow.yaml",
                "configs/model/neural_components.yaml",
                "configs/train/smoke.yaml",
            )
        ),
        set_overrides=("runtime.device_preference=[cpu]",),
    )
    with pytest.raises(TrainingError, match="configuration"):
        run_training(
            changed,
            validation_manifest=manifest,
            run_dir=run_dir,
            source_commit="a" * 40,
            resume=run_dir / result.latest.relative_path,
        )
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(TrainingError, match="manifest"):
        run_training(
            smoke_config,
            validation_manifest=manifest,
            run_dir=run_dir,
            source_commit="a" * 40,
            resume=run_dir / result.latest.relative_path,
        )


def test_manifest_hash_binds_the_bytes_that_generated_corpus(smoke_config, tmp_path, monkeypatch):
    import hashlib

    from silent_cascade.train import curriculum_data, trainer

    manifest = _manifest(smoke_config, tmp_path / "manifest.json")
    expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
    original = curriculum_data.make_curriculum_example
    changed = False

    def mutate_after_manifest_read(*args):
        nonlocal changed
        if not changed:
            manifest.write_text(manifest.read_text() + "\n")
            changed = True
        return original(*args)

    monkeypatch.setattr(curriculum_data, "make_curriculum_example", mutate_after_manifest_read)
    result = trainer.run_training(
        smoke_config,
        validation_manifest=manifest,
        run_dir=tmp_path / "run",
        source_commit="a" * 40,
        device="cpu",
    )
    assert result.progress.validation_manifest_sha256 == expected


def test_production_manifest_requires_exact_complete_validation_keys(neural_config):
    import hashlib

    from pydantic import ValidationError

    from silent_cascade.hashing import canonical_json_bytes
    from silent_cascade.train.curriculum_data import (
        ComponentManifest,
        ComponentManifestEntry,
        CurriculumKey,
    )

    entries = tuple(
        ComponentManifestEntry(
            key=CurriculumKey("ofd-one-hop-v1", "validation", 313, 337, i, "one_hop"),
            public_hash=f"{i:064x}",
            example_hash=f"{i + 10000:064x}",
            accepted_attempt=0,
        )
        for i in range(10000)
    )
    kwargs = dict(
        publication="production",
        source_revision="a" * 40,
        plan_revision="b" * 40,
        config_hash=hashlib.sha256(canonical_json_bytes(neural_config)).hexdigest(),
        count=10000,
        entries=entries,
    )
    manifest = ComponentManifest(**kwargs)
    assert manifest.publication == "production"
    for change in ({"entries": entries[:-1], "count": 9999}, {"entries": tuple(reversed(entries))}):
        with pytest.raises(ValidationError):
            ComponentManifest(**{**kwargs, **change})


def test_post_publication_failure_attributes_the_durable_checkpoint(
    smoke_config, tmp_path, monkeypatch
):
    from silent_cascade.train import trainer
    from silent_cascade.train.checkpoints import load_checkpoint_index

    manifest = _manifest(smoke_config, tmp_path / "manifest.json")
    run_dir = tmp_path / "run"
    original = trainer.prune_training_checkpoints

    def fail_after_publication(directory):
        if load_checkpoint_index(directory).latest.optimizer_step == 2:
            raise TrainingError("injected retention failure")
        return original(directory)

    monkeypatch.setattr(trainer, "prune_training_checkpoints", fail_after_publication)
    with pytest.raises(TrainingError) as caught:
        trainer.run_training(
            smoke_config,
            validation_manifest=manifest,
            run_dir=run_dir,
            source_commit="a" * 40,
            device="cpu",
        )
    failure = trainer._read_json(Path(caught.value.context["failure_path"]))
    assert failure["prior_checkpoint"]["optimizer_step"] == 2
    assert load_checkpoint_index(run_dir).latest.optimizer_step == 2
