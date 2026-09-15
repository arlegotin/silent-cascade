"""The later collector and these tests execute the same numerical comparisons."""

import pytest
import torch

from silent_cascade.config import resolve_config
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.train.batches import next_training_batch
from silent_cascade.train.config import Phase3Config

from .test_training_cli import CONFIGS


def test_cpu_resume_evidence_is_exact_and_measured():
    from silent_cascade.train.verification import measure_cpu_resume

    config = resolve_config(Phase3Config, CONFIGS)
    evidence = measure_cpu_resume(config, source_commit="a" * 40)
    assert evidence.passed
    assert evidence.parameters.compared > 100_000
    assert evidence.optimizer.compared > evidence.parameters.compared
    assert evidence.parameters.max_absolute_error == 0.0
    assert evidence.optimizer.max_absolute_error == 0.0
    assert evidence.rng_mismatches == 0
    assert evidence.batch_hash_mismatches == 0
    assert len(evidence.checkpoint_sha256) == 64
    assert evidence.config_sha256 == config.sha256
    assert evidence.objective_version == "teacher_timed_v1"
    assert evidence.auxiliary_coefficient == 0.0
    assert evidence.optimizer_options["eps"] == 1e-7
    assert len(evidence.first_example_hashes) == 8


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="native MPS unavailable")
def test_neural_one_step_cpu_mps_agree():
    from silent_cascade.train.verification import measure_device_parity

    config = resolve_config(Phase3Config, CONFIGS)
    torch.manual_seed(11)
    model = EventFlowModel(config.config.neural)
    batch = next_training_batch(config.config, stage="one_hop", batch_counter=0)
    evidence = measure_device_parity(model, batch)
    assert evidence.passed, evidence.model_dump_json(indent=2)
    assert evidence.devices == ("cpu", "mps")
    assert evidence.output_dtype == "float32"
    assert evidence.recalled_record_mismatches == 0
    assert evidence.action_class_mismatches == 0
    assert evidence.forward.rtol == 1e-4 and evidence.forward.atol == 1e-5
    assert evidence.gradients.rtol == 1e-3 and evidence.gradients.atol == 1e-5
    assert evidence.updated_weights.compared > 100_000
    assert evidence.active_crossings > 0 and evidence.dormant_crossings > 0
    assert evidence.empty_memory_rows > 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="native MPS unavailable")
def test_nonempty_probe_is_not_counted_as_empty(monkeypatch):
    from silent_cascade.train.verification import measure_device_parity

    config = resolve_config(Phase3Config, CONFIGS)
    torch.manual_seed(11)
    model = EventFlowModel(config.config.neural)
    original = EventFlowModel.initial_context

    def nonempty(self, batch_size, *, device):
        context = original(self, batch_size, device=device)
        if batch_size == 1:
            eligibility = context.eligibility.clone()
            eligibility[:, 0] = True
            context = context._updated(eligibility=eligibility)
        return context

    monkeypatch.setattr(EventFlowModel, "initial_context", nonempty)
    batch = next_training_batch(config.config, stage="one_hop", batch_counter=0)
    evidence = measure_device_parity(model, batch)
    assert evidence.empty_memory_rows == 0
    assert not evidence.passed


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="native MPS unavailable")
def test_mps_resume_evidence_measures_native_draws_and_restores_rng(monkeypatch):
    from silent_cascade.rng import snapshot_global_rng
    from silent_cascade.train.verification import measure_mps_resume

    config = resolve_config(Phase3Config, CONFIGS)
    before = snapshot_global_rng()
    draws = []
    original = torch.rand

    def tracked(*args, **kwargs):
        value = original(*args, **kwargs)
        if value.device.type == "mps":
            draws.append(value.detach().cpu())
        return value

    monkeypatch.setattr(torch, "rand", tracked)
    evidence = measure_mps_resume(config, source_commit="a" * 40)
    assert evidence.passed and evidence.device == "mps"
    assert evidence.schema_version == "phase3-mps-resume-v2"
    assert evidence.parameters.rtol == evidence.optimizer.rtol == evidence.losses.rtol == 1e-4
    assert evidence.parameters.atol == evidence.optimizer.atol == evidence.losses.atol == 1e-5
    assert len(draws) == 2 and torch.equal(draws[0], draws[1])
    after = snapshot_global_rng()
    assert torch.equal(before.torch_cpu_state, after.torch_cpu_state)
    assert torch.equal(before.torch_mps_state, after.torch_mps_state)
