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
