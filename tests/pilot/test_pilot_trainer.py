import pytest
import torch

from .test_pilot_source import DATA_SETUP, checkout


def _sparse_artifact(root, name, size):
    """Byte-boundary fixture only; deliberately not scientific evaluation rows."""
    import hashlib

    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return path, digest


def _check_artifact_consumer(root, name, digest, consumer):
    from silent_cascade.train.pilot_state import PilotProgress
    from silent_cascade.train.pilot_trainer import _journal, verify_journal

    if consumer == "journal":
        progress = _journal(
            root, PilotProgress(), {"kind": "validation", "artifacts": {name: digest}}
        )
        assert verify_journal(root, progress) == {progress.journal_sha256}
    else:
        from silent_cascade.train.pilot_trainer import _collect_artifact_hashes

        assert _collect_artifact_hashes(root)[name] == digest


@pytest.mark.parametrize("consumer", ["journal", "inventory"])
@pytest.mark.parametrize("leaf", ["rows.jsonl", "traces/example.jsonl"])
@pytest.mark.parametrize("size", [64 * 1024 * 1024 + 1, 128 * 1024 * 1024])
def test_autonomous_hash_consumers_accept_larger_than_manifest_limit(
    tmp_path, consumer, leaf, size
):
    name = "attempt-test/validation-0-one_hop/autonomous/" + leaf
    _, digest = _sparse_artifact(tmp_path, name, size)
    _check_artifact_consumer(tmp_path, name, digest, consumer)


@pytest.mark.parametrize("consumer", ["journal", "inventory"])
@pytest.mark.parametrize(
    ("name", "size"),
    [
        ("attempt-test/validation-0-one_hop/autonomous/rows.jsonl", 128 * 1024 * 1024 + 1),
        ("attempt-test/validation-0-one_hop/validation.json", 64 * 1024 * 1024 + 1),
    ],
)
def test_artifact_consumers_keep_domain_byte_limits(tmp_path, consumer, name, size):
    _, digest = _sparse_artifact(tmp_path, name, size)
    with pytest.raises(ValueError, match="byte_limit"):
        _check_artifact_consumer(tmp_path, name, digest, consumer)


@pytest.mark.parametrize("consumer", ["journal", "inventory"])
def test_artifact_consumers_reject_leaf_symlinks(tmp_path, consumer):
    name = "attempt-test/validation-0-one_hop/autonomous/rows.jsonl"
    path, digest = _sparse_artifact(tmp_path, name, 1)
    outside = tmp_path / "unrelated"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(OSError):
        _check_artifact_consumer(tmp_path, name, digest, consumer)


@pytest.mark.parametrize("kind", ["changed", "parent_symlink", "absolute", "traversal", "frozen"])
def test_journal_artifact_hash_and_path_guards(tmp_path, kind):
    from silent_cascade.train.state import TrainingError

    name = "attempt-test/validation-0-one_hop/autonomous/rows.jsonl"
    path, digest = _sparse_artifact(tmp_path, name, 1)
    if kind == "changed":
        path.write_bytes(b"x")
    elif kind == "parent_symlink":
        outside = tmp_path / "unrelated"
        path.parent.rename(outside)
        path.parent.symlink_to(outside, target_is_directory=True)
    elif kind == "absolute":
        name = str(path)
    elif kind == "traversal":
        name = "attempt-test/../" + name
    else:
        name = "attempt-test/validation-0-one_hop/autonomous/frozen/rows.jsonl"
        _sparse_artifact(tmp_path, name, 1)
    with pytest.raises((TrainingError, ValueError, OSError)):
        _check_artifact_consumer(tmp_path, name, digest, "journal")


def test_real_update_changes_weights_and_records_full_losses():
    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_data import next_pilot_batch
    from silent_cascade.train.pilot_trainer import pilot_train_one_step
    from silent_cascade.train.trainer import make_optimizer

    config = resolve_pilot_config("phase4_smoke").config
    torch.manual_seed(11)
    model = EventFlowModel(config.neural)
    optimizer = make_optimizer(model, config.pilot)
    before = [p.detach().clone() for p in model.parameters()]
    result = pilot_train_one_step(
        model, optimizer, next_pilot_batch(config, stage="one_hop", batch_counter=0), config
    )
    assert result.loss > 0 and result.gradient_norm > 0
    assert result.content_loss > 0 and result.timed_loss > 0
    assert result.per_position and result.compute.foundation_model_calls == 0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


@pytest.mark.parametrize("interrupt_after", [2, 4])
def test_authenticated_training_resume_and_latest_debug_export(tmp_path, interrupt_after):
    root, execute = checkout(tmp_path / "repo")
    execute(
        root,
        DATA_SETUP
        + f"\nstop_at = {interrupt_after}\n"
        + """
import json
import silent_cascade.train.pilot_trainer as trainer
complete = trainer.run_pilot_training(config, manifests=manifests, run_dir=root/'whole',
                                     device='cpu', source_commit=source)
assert complete.progress.global_step == 4 and complete.status == 'step_ceiling'
assert complete.selected_checkpoint is None and not complete.gate_eligible
assert not (root/'whole/selected.json').exists()
assert complete.latest_weights.sha256 and complete.latest_weights.path
assert len(list((root/'whole').rglob('autonomous/rows.jsonl'))) == 2
real_step, calls = trainer.pilot_train_one_step, 0
def interrupt(*args, **kwargs):
    global calls
    calls += 1
    if calls == stop_at:
        raise RuntimeError('injected interruption after one uncommitted update')
    return real_step(*args, **kwargs)
trainer.pilot_train_one_step = interrupt
try:
    trainer.run_pilot_training(config, manifests=manifests, run_dir=root/'split',
                               device='cpu', source_commit=source)
except RuntimeError:
    pass
else:
    raise AssertionError('injected interruption did not occur')
trainer.pilot_train_one_step = real_step
index = json.loads((root/'split/checkpoint-index.json').read_bytes())
assert index['latest']['global_step'] == (stop_at - 1) // 2 * 2
resumed = trainer.run_pilot_training(config, manifests=manifests, run_dir=root/'split',
    device='cpu', source_commit=source, resume=root/'split'/index['latest']['path'])
assert resumed.model_identity == complete.model_identity
assert resumed.progress.global_step == 4 and resumed.progress.batch_counter == 4
assert not (root/'split/selected.json').exists()
restart = json.loads(next((root/'split').glob('restart-*.json')).read_bytes())
assert len(restart['uncommitted_journal_tail']) == 1
""",
    )
