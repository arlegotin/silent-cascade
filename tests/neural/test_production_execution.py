"""Real checkout production routes reject misattribution before expensive work."""

import shutil

import pytest

from .test_phase3_provenance import ROOT, checkout, execute

SETUP = r"""
from pathlib import Path
import json, subprocess, sys, types
from silent_cascade.config import resolve_config
from silent_cascade.train.config import Phase3Config
from silent_cascade.train.curriculum_data import ComponentManifest
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.train.checkpoints import export_weights, save_training_checkpoint
from silent_cascade.train.state import TrainProgress
from silent_cascade.train import trainer, cli
from silent_cascade.errors import ProvenanceError
root = Path.cwd()
source = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
paths = [root / p for p in ('configs/base.yaml','configs/data/primary.yaml',
    'configs/model/event_flow.yaml','configs/model/neural_components.yaml',
    'configs/train/one_hop_content_v2.yaml')]
config = resolve_config(Phase3Config, paths)
manifest_path = root / 'manifest.json'
value = json.loads(manifest_path.read_bytes())
value.update(source_revision=source, plan_revision=source, config_hash=config.sha256)
manifest_path.write_text(json.dumps(value))
subprocess.run(['git','add','manifest.json'], check=True)
subprocess.run(['git','commit','-qm','introduce fixed production metadata'], check=True)
archive_dir = root.parent / 'archives'
archive_dir.mkdir()
weights = export_weights(archive_dir, EventFlowModel(config.config.neural),
                         config=config, source_commit=source)
import hashlib
model = EventFlowModel(config.config.neural)
progress = TrainProgress(optimizer_step=0, next_batch_counter=0, stage='one_hop',
    train_root_seed=config.config.training.train_root_seed,
    train_public_id_seed=config.config.training.train_public_id_seed,
    patience_counter=0, retained_checkpoints=(),
    validation_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
checkpoint = save_training_checkpoint(archive_dir, model,
    trainer.make_optimizer(model, config.config.training), progress,
    config=config, source_commit=source)
output = root.parent / 'output.json'
run = root.parent / 'new-run'
class DownstreamReached(Exception):
    pass
def downstream(*args, **kwargs):
    raise DownstreamReached('unauthenticated corpus/model/RNG boundary crossed')
ComponentManifest.build_corpus = downstream
trainer.EventFlowModel = downstream
trainer.load_training_checkpoint = downstream
cli.load_weight_bundle = downstream
"""


def _checkout(tmp_path):
    root = checkout(tmp_path / "source")
    shutil.copy2(
        ROOT / "manifests/validation/phase3/one-hop-10000-content-v2.json",
        root / "manifest.json",
    )
    return root


@pytest.mark.parametrize("route", ["callable", "cli", "callable_resume", "cli_resume", "evaluate"])
@pytest.mark.parametrize("mutation", ["dirty", "committed", "foreign", "clean"])
def test_production_routes_authenticate_before_compute(tmp_path, route, mutation):
    root = _checkout(tmp_path)
    execute(
        root,
        SETUP
        + f"route, mutation = {route!r}, {mutation!r}\n"
        + r"""
if mutation in ('dirty', 'committed'):
    path = root / 'src/silent_cascade/models/controller.py'
    path.write_bytes(path.read_bytes() + b'\n# execution changed\n')
    if mutation == 'committed':
        subprocess.run(['git','add',str(path)], check=True)
        subprocess.run(['git','commit','-qm','unreviewed executable'], check=True)
elif mutation == 'foreign':
    import importlib.util, shutil
    foreign = root.parent / 'foreign-controller.py'
    shutil.copy2(root / 'src/silent_cascade/models/controller.py', foreign)
    name = 'silent_cascade.models.controller'
    spec = importlib.util.spec_from_file_location(name, foreign)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
resume = archive_dir / checkpoint.relative_path if 'resume' in route else None
try:
    if route.startswith('callable'):
        trainer.run_training(config, validation_manifest=manifest_path, run_dir=run,
                             source_commit=source, resume=resume, device='cpu')
    elif route.startswith('cli'):
        from typer.testing import CliRunner
        arguments = ['fit', *(v for p in paths for v in ('--config', str(p))),
                     '--validation-manifest', str(manifest_path), '--run-dir', str(run),
                     '--expected-source-commit', source, '--device', 'cpu']
        if resume is not None:
            arguments += ['--resume', str(resume)]
        result = CliRunner().invoke(cli.app, arguments, catch_exceptions=False)
        assert mutation != 'clean' and result.exit_code == 1, result.output
        assert json.loads(result.stdout)['error']['code'] == 'provenance_error', result.output
        raise ProvenanceError('checked CLI rejection')
    else:
        cli.evaluate(archive_dir / weights.relative_path, weights.file_sha256,
                     manifest_path, output)
except ProvenanceError:
    assert mutation != 'clean', 'clean production checkout rejected'
except DownstreamReached:
    assert mutation == 'clean', 'changed/foreign source reached downstream execution'
else:
    raise AssertionError('route returned without rejection or downstream sentinel')
assert not run.exists() and not output.exists()
""",
    )


@pytest.mark.parametrize("mutation", ["reordered", "changed", "empty_changed", "forged_object"])
def test_production_rejects_noncanonical_recipe_before_compute(tmp_path, mutation):
    root = _checkout(tmp_path)
    execute(
        root,
        SETUP
        + f"mutation = {mutation!r}\n"
        + r"""
from dataclasses import replace
if mutation == 'reordered':
    config = replace(config, source_paths=tuple(reversed(config.source_paths)))
elif mutation in ('changed', 'empty_changed'):
    config = resolve_config(Phase3Config, paths, set_overrides=['training.model_seed=11',
                            'runtime.device_preference=[cpu]'])
    if mutation == 'empty_changed':
        config = replace(config, source_paths=())
    value['config_hash'] = config.sha256
    manifest_path.write_text(json.dumps(value))
    subprocess.run(['git','add','manifest.json'], check=True)
    subprocess.run(['git','commit','-qm','noncanonical runtime recipe'], check=True)
else:
    changed = resolve_config(Phase3Config, paths, set_overrides=['runtime.device_preference=[cpu]'])
    config = replace(config, config=changed.config)
try:
    trainer.run_training(config, validation_manifest=manifest_path, run_dir=run,
                         source_commit=source, device='cpu')
except ProvenanceError:
    pass
except DownstreamReached:
    raise AssertionError('noncanonical recipe reached compute')
else:
    raise AssertionError('noncanonical recipe accepted')
assert not run.exists()
""",
    )


def test_evaluation_checks_expected_archive_hash_before_source_or_model(tmp_path):
    root = _checkout(tmp_path)
    execute(
        root,
        SETUP
        + r"""
from silent_cascade.train.state import TrainingError
path = root / 'src/silent_cascade/models/controller.py'
path.write_bytes(path.read_bytes() + b'\n# dirty source\n')
try:
    cli.evaluate(archive_dir / weights.relative_path, '0'*64, manifest_path, output)
except TrainingError as error:
    assert 'hash' in str(error)
else:
    raise AssertionError('wrong expected hash accepted')
assert not output.exists()
""",
    )


def test_training_rechecks_source_before_initial_durable_checkpoint(tmp_path):
    root = _checkout(tmp_path)
    execute(
        root,
        SETUP
        + r"""
def changed_corpus(*args):
    path = root / 'src/silent_cascade/models/controller.py'
    path.write_bytes(path.read_bytes() + b'\n# changed during work\n')
    return None
ComponentManifest.build_corpus = changed_corpus
trainer.EventFlowModel = EventFlowModel
trainer.next_training_batch = downstream
try:
    trainer.run_training(config, validation_manifest=manifest_path, run_dir=run,
                         source_commit=source, device='cpu')
except ProvenanceError:
    pass
else:
    raise AssertionError('changed source published initial checkpoint')
assert not tuple(run.glob('*.safetensors'))
assert not tuple(run.glob('journal-*.json'))
""",
    )


def test_evaluation_rechecks_source_before_publication(tmp_path):
    root = _checkout(tmp_path)
    execute(
        root,
        SETUP
        + r"""
from silent_cascade.train.checkpoints import load_weight_bundle
from silent_cascade.train.component_eval import evaluate_components
from silent_cascade.train.curriculum_data import (
    ComponentCorpus, CurriculumKey, make_curriculum_example,
)
from silent_cascade.train.traces import build_teacher_trace, component_target
example = make_curriculum_example(config.config,
    CurriculumKey('ofd-one-hop-v1', 'debug', 313, 337, 0, 'one_hop'))
corpus = ComponentCorpus((example.public,), (component_target(build_teacher_trace(example)),))
ComponentManifest.build_corpus = lambda *args: corpus
cli.load_weight_bundle = load_weight_bundle
def changed_evaluation(*args, **kwargs):
    result = evaluate_components(*args, **kwargs)
    path = root / 'src/silent_cascade/models/controller.py'
    path.write_bytes(path.read_bytes() + b'\n# changed during evaluation\n')
    return result
cli.evaluate_components = changed_evaluation
try:
    cli.evaluate(archive_dir / weights.relative_path, weights.file_sha256, manifest_path, output)
except ProvenanceError:
    pass
else:
    raise AssertionError('changed evaluator published final rows')
assert not output.exists()
""",
    )
