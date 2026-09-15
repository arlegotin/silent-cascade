"""Run path rejection must not mutate a symlink target or training RNG."""

import random

import numpy as np
import pytest
import torch

from silent_cascade.train.state import TrainingError

from .test_trainer import _manifest, smoke_config  # noqa: F401


@pytest.mark.parametrize("ancestor", [False, True])
def test_symlink_run_rejection_has_no_target_side_effects(tmp_path, smoke_config, ancestor):  # noqa: F811
    from silent_cascade.train.trainer import run_training

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "run-link"
    link.symlink_to(target, target_is_directory=True)
    manifest = _manifest(smoke_config, tmp_path / "manifest.json")
    before = tuple(target.iterdir())
    with pytest.raises((TrainingError, OSError)) as rejected:
        run_training(
            smoke_config,
            validation_manifest=manifest,
            run_dir=link / "new/run" if ancestor else link,
            source_commit="a" * 40,
            device="cpu",
        )
    assert tuple(target.iterdir()) == before
    assert isinstance(rejected.value, TrainingError)


def test_real_missing_empty_and_resumed_directories_preserve_rng(tmp_path):
    from silent_cascade.train.run_directory import create_attempt_directory, prepare_run_directory

    run = tmp_path / "new/parents/run"
    assert prepare_run_directory(run, resume=False) == run
    assert prepare_run_directory(run, resume=False) == run
    python = random.getstate()
    numpy = np.random.get_state()
    tensor = torch.get_rng_state().clone()
    first, second = create_attempt_directory(run), create_attempt_directory(run)
    assert first != second and first.is_dir() and second.is_dir()
    assert first.parent == run and first.name.startswith("attempt-")
    assert random.getstate() == python
    assert np.array_equal(np.random.get_state()[1], numpy[1])
    assert torch.equal(torch.get_rng_state(), tensor)
    assert prepare_run_directory(run, resume=True) == run
    with pytest.raises(TrainingError, match="resume"):
        prepare_run_directory(run, resume=False)
    missing = tmp_path / "missing/resume"
    with pytest.raises(TrainingError):
        prepare_run_directory(missing, resume=True)
    assert not missing.parent.exists()


@pytest.mark.parametrize("ancestor", [False, True])
def test_attempt_creation_rejects_symlinks_without_side_effects(tmp_path, ancestor):
    from silent_cascade.train.run_directory import create_attempt_directory

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(TrainingError):
        create_attempt_directory(link / "missing" if ancestor else link)
    assert tuple(target.iterdir()) == ()
