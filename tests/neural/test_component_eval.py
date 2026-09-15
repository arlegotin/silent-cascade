"""Unassisted decisions retain errors and unconditional scoring denominators."""

import json
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.train.batches import next_training_batch
from silent_cascade.train.curriculum_data import ComponentCorpus
from silent_cascade.train.traces import component_target


@pytest.fixture
def component_corpus(neural_config):
    batch = next_training_batch(neural_config, stage="one_hop", batch_counter=0)
    return ComponentCorpus(
        tuple(e.public for e in batch.examples[:8]),
        tuple(component_target(t) for t in batch.teacher_traces[:8]),
    )


def test_predictions_isolated_from_labels_and_bounded(neural_config, component_corpus):
    from silent_cascade.train.component_eval import predict_components

    torch.manual_seed(11)
    model = EventFlowModel(neural_config.neural)
    first = predict_components(model, component_corpus.public_batch(0, 8))
    changed = replace(component_corpus, targets=tuple(reversed(component_corpus.targets)))
    second = predict_components(model, changed.public_batch(0, 8))
    assert first.to_primitive() == second.to_primitive()
    assert all(row.atomic_transitions <= 4 for row in first.rows)
    assert first.evaluation_mode == "unassisted_content"


def test_missing_content_never_reduces_denominators(component_corpus):
    from silent_cascade.train.component_eval import (
        ComponentPrediction,
        ComponentPredictions,
        score_components,
    )

    predictions = ComponentPredictions(
        tuple(ComponentPrediction(stop_reason="dormant") for _ in component_corpus.targets)
    )
    score = score_components(predictions, component_corpus.targets)
    required = sum(len(t.record_ids) for t in component_corpus.targets)
    assert score.required_recall_count == required
    assert score.required_composition_count == required
    assert score.episode_count == 8
    assert score.complete_chain_accuracy == 0.0
    assert not score.gates_pass
    assert score.to_primitive()["category_metrics"]["positive"]["accuracy"] == 0.0
    assert predictions.to_primitive()["timed"] is False


def test_evaluation_restores_training_mode(neural_config, component_corpus):
    from silent_cascade.train.component_eval import evaluate_components

    model = EventFlowModel(neural_config.neural).train()
    evaluation = evaluate_components(model, component_corpus, batch_size=4)
    assert model.training
    assert len(evaluation.predictions.rows) == 8
    assert evaluation.metrics.episode_count == 8
    assert all(c.foundation_model_calls == 0 for c in evaluation.compute)
    assert sum(row.records_scored for row in evaluation.predictions.rows) == sum(
        counter.records_scored for counter in evaluation.compute
    )


def test_prediction_fresh_process_blocks_all_private_environment_imports(
    neural_config, component_corpus, tmp_path
):
    from dataclasses import fields

    from safetensors.torch import save_file

    public = component_corpus.public_batch(0, 8)
    tensors = {}
    metadata = {}
    for prefix, value in (("public", public), ("records", public.records)):
        for field in fields(value):
            item = getattr(value, field.name)
            if isinstance(item, torch.Tensor):
                tensors[prefix + "/" + field.name] = item.contiguous()
            elif field.name != "records":
                metadata[prefix + "/" + field.name] = item
    save_file(tensors, tmp_path / "public.safetensors")
    (tmp_path / "public.json").write_text(json.dumps(metadata))
    script = """
import importlib.abc, json, sys
from dataclasses import fields
from pathlib import Path
class BlockPrivate(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        private_env = (fullname.startswith("silent_cascade.env.")
                       and fullname != "silent_cascade.env.config")
        if private_env or fullname in {
            "silent_cascade.train.batches", "silent_cascade.train.unroll",
            "silent_cascade.train.curriculum_data"}:
            raise AssertionError("private import: " + fullname)
sys.meta_path.insert(0, BlockPrivate())
from safetensors.torch import load_file
from silent_cascade.models.config import NeuralModelConfig
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.models.types import PublicInputBatch
from silent_cascade.memory.tensor_store import RecordTensorBatch
from silent_cascade.train.component_eval import predict_components
root=Path(sys.argv[1]); values=load_file(root / "public.safetensors")
def tuples(v):
    return tuple(tuples(x) for x in v) if isinstance(v,list) else v
values.update({k:tuples(v) for k,v in json.loads((root / "public.json").read_text()).items()})
records=RecordTensorBatch(**{f.name:values["records/"+f.name] for f in fields(RecordTensorBatch)})
public=PublicInputBatch(records=records, **{
    f.name:values["public/"+f.name] for f in fields(PublicInputBatch) if f.name != "records"})
rows=predict_components(EventFlowModel(NeuralModelConfig()), public)
assert len(rows.rows)==8
print("public-only prediction passed")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_score_exact_chain_extra_cap_and_disconnected_preview_stop():
    from silent_cascade.train.component_eval import (
        ComponentPrediction,
        ComponentPredictions,
        ContentDecision,
        score_components,
    )
    from silent_cascade.train.curriculum_data import ComponentContent, ComponentTarget

    target = ComponentTarget(
        (7,), (ComponentContent(7, 0, 12, None, None, None, 2, 1.0, True, False),), 4, 2
    )
    decision = ContentDecision(7, 0, 12, 3, 0.3, 0.5, 0, 0.9, True, True)
    row = ComponentPrediction((7,), (decision,), "dormant", 2)
    score = score_components(ComponentPredictions((row,)), (target,))
    assert score.complete_chain_accuracy == 1.0
    for wrong in (
        replace(row, stop_reason="capped"),
        replace(row, record_ids=(7, 9)),
        replace(row, content=(replace(decision, append_support=False),)),
    ):
        score = score_components(ComponentPredictions((wrong,)), (target,))
        assert score.complete_chain_accuracy == 0.0
        assert score.required_recall_count == score.required_composition_count == 1


def test_internal_nan_is_typed_failure(neural_config, component_corpus):
    from silent_cascade.models.errors import NeuralError
    from silent_cascade.train.component_eval import predict_components

    model = EventFlowModel(neural_config.neural)
    with torch.no_grad():
        next(model.parameters()).fill_(torch.nan)
    with pytest.raises(NeuralError):
        predict_components(model, component_corpus.public_batch(0, 8))


def test_prediction_rejects_wrong_model_dtype_with_typed_error(neural_config, component_corpus):
    from silent_cascade.models.errors import NeuralError
    from silent_cascade.train.component_eval import predict_components

    model = EventFlowModel(neural_config.neural).double()
    with pytest.raises(NeuralError, match="dtype"):
        predict_components(model, component_corpus.public_batch(0, 8))
