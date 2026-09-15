"""Real two-step AdamW continuation; no dependency on the future trainer."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from silent_cascade.config import resolve_config
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.models.losses import event_flow_loss
from silent_cascade.rng import snapshot_global_rng
from silent_cascade.train.batches import next_training_batch
from silent_cascade.train.config import Phase3Config
from silent_cascade.train.state import TrainProgress


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
def test_checkpoint_resume_reproduces_next_step_and_global_draws(tmp_path, device):
    from silent_cascade.train.checkpoints import (
        export_weights,
        load_training_checkpoint,
        load_weights,
        save_training_checkpoint,
    )
    from silent_cascade.train.unroll import teacher_forced_unroll

    config = resolve_config(
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
    torch.manual_seed(11)
    model = EventFlowModel(config.config.neural).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4, eps=1e-7, foreach=False, fused=False
    )

    def step(model, optimizer, counter):
        batch = next_training_batch(config.config, stage="one_hop", batch_counter=counter).to(
            device
        )
        optimizer.zero_grad(set_to_none=True)
        loss = event_flow_loss(
            teacher_forced_unroll(model, batch).loss_inputs(), config.config.training.loss_weights
        ).total
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return batch.example_hashes, loss.detach().cpu()

    def draws():
        return (
            random.random(),
            float(np.random.random()),
            torch.rand(5),
            torch.rand(5, device=device).cpu(),
        )

    step(model, optimizer, 0)
    progress = TrainProgress(
        optimizer_step=1,
        next_batch_counter=1,
        stage="one_hop",
        train_root_seed=311,
        train_public_id_seed=331,
        best_metric=None,
        best_step=None,
        patience_counter=0,
        retained_checkpoints=(),
        validation_manifest_sha256="b" * 64,
    )
    saved = save_training_checkpoint(
        tmp_path, model, optimizer, progress, config=config, source_commit="a" * 40
    )
    path = tmp_path / saved.relative_path
    expected_draws = draws()
    expected_hashes, expected_loss = step(model, optimizer, 1)
    before = snapshot_global_rng()
    restored = load_training_checkpoint(
        path, expected_config_sha256=config.sha256, expected_source_commit="a" * 40, device=device
    )
    assert torch.equal(before.torch_cpu_state, torch.get_rng_state())
    if device == "mps":
        assert torch.equal(before.torch_mps_state, torch.mps.get_rng_state())
    restored.restore_rng()
    actual_draws = draws()
    assert expected_draws[:2] == actual_draws[:2]
    assert all(torch.equal(a, b) for a, b in zip(expected_draws[2:], actual_draws[2:], strict=True))
    hashes, loss = step(restored.model, restored.optimizer, restored.progress.next_batch_counter)
    assert hashes == expected_hashes
    rtol, atol = (0, 0) if device == "cpu" else (1e-4, 1e-5)
    torch.testing.assert_close(loss, expected_loss, rtol=rtol, atol=atol)
    for left, right in zip(model.parameters(), restored.model.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=rtol, atol=atol)
        for name in optimizer.state[left]:
            torch.testing.assert_close(
                optimizer.state[left][name],
                restored.optimizer.state[right][name],
                rtol=rtol,
                atol=atol,
            )
        assert restored.optimizer.state[right]["step"].device.type == "cpu"
        assert restored.optimizer.state[right]["step"].dtype == torch.float32
    descriptor = export_weights(tmp_path, restored.model, config=config, source_commit="a" * 40)
    export = tmp_path / descriptor.relative_path
    portable = load_weights(export, expected_sha256=descriptor.file_sha256, device="cpu")
    for left, right in zip(restored.model.parameters(), portable.parameters(), strict=True):
        torch.testing.assert_close(left.cpu(), right, rtol=0, atol=0)
