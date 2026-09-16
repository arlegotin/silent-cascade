"""Source identity, fixed recipes, and bounded primitive decoding."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def checkout(path):
    """Real isolated Git history containing the currently executing implementation."""
    path.mkdir()
    for directory in ("src", "configs", "scripts", "docs/superpowers"):
        shutil.copytree(
            ROOT / directory, path / directory, ignore=shutil.ignore_patterns("__pycache__")
        )
    for name in ("pyproject.toml", "uv.lock", ".gitignore"):
        shutil.copy2(ROOT / name, path / name)
    mapping = Path("manifests/validation/phase3/delivery.json")
    if (ROOT / mapping).exists():
        (path / mapping).parent.mkdir(parents=True)
        shutil.copy2(ROOT / mapping, path / mapping)
    for args in (
        ("init", "-q"),
        ("config", "user.email", "test@example.invalid"),
        ("config", "user.name", "Local test"),
        ("add", "."),
        ("commit", "-qm", "reviewed source"),
    ):
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)
    return path


def execute(root, program):
    completed = subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout


SETUP = """
from pathlib import Path
import subprocess
from silent_cascade.config import resolve_config
from silent_cascade.train.config import Phase3Config
from silent_cascade.train.provenance import freeze_component_manifest
root = Path.cwd()
source = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
paths = ['configs/base.yaml','configs/data/primary.yaml','configs/model/event_flow.yaml',
         'configs/model/neural_components.yaml','configs/train/smoke_content_v2.yaml']
config = resolve_config(Phase3Config, tuple(Path(p) for p in paths))
manifest_path = root / 'component-manifest.json'
manifest = freeze_component_manifest(config, source_commit=source, plan_revision=source,
                                     output_path=manifest_path)
"""


def test_source_closure_includes_ancestor_initializers_and_execution_consumers():
    from silent_cascade.train.provenance import PHASE3_SOURCE_PATHS

    assert tuple(sorted(set(PHASE3_SOURCE_PATHS))) == PHASE3_SOURCE_PATHS
    for name in (
        "train/observations.py",
        "train/checkpoints.py",
        "train/verification.py",
        "train/objective.py",
        "train/numeric_inventory.py",
        "train/content_unroll.py",
        "models/content_loss.py",
        "models/content_types.py",
        "models/transitions.py",
        "models/weights.py",
        "train/evidence_types.py",
        "models/__init__.py",
        "train/__init__.py",
        "eval/__init__.py",
        "memory/__init__.py",
        "__init__.py",
    ):
        assert f"src/silent_cascade/{name}" in PHASE3_SOURCE_PATHS


def test_freeze_real_debug_recipe_and_refuse_overwrite(tmp_path):
    from silent_cascade.train import provenance  # Missing boundary must fail before fixture setup.

    assert provenance is not None
    root = checkout(tmp_path / "source")
    execute(
        root,
        SETUP
        + """
from silent_cascade.errors import ProvenanceError
assert manifest.count == 16 and manifest.publication == 'debug'
assert [e.key.episode_index for e in manifest.entries] == list(range(16))
assert all(e.key.root_seed == 313 and e.key.public_id_seed == 337 for e in manifest.entries)
original = manifest_path.read_bytes()
subprocess.run(['git','add','component-manifest.json'], check=True)
subprocess.run(['git','commit','-qm','introduce immutable manifest'], check=True)
try:
    from silent_cascade.train.provenance import publish
    publish(manifest_path, {'collision': 'different manifest bytes'})
except ProvenanceError as error:
    assert 'publication' in str(error) or 'exist' in str(error)
else:
    raise AssertionError('freeze overwrote an existing recipe')
assert manifest_path.read_bytes() == original
""",
    )


def test_collector_cannot_authenticate_a_different_clean_checkout(tmp_path):
    from silent_cascade.errors import ProvenanceError
    from silent_cascade.train.provenance import authenticate_source

    root = checkout(tmp_path / "other")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    with pytest.raises(ProvenanceError, match="execut"):
        authenticate_source(root, source, source)


def test_freeze_refuses_corpus_outside_source_checkout(tmp_path):
    root = checkout(tmp_path / "source")
    execute(
        root,
        SETUP[: SETUP.index("manifest = freeze_component_manifest")]
        + """
from silent_cascade.errors import ProvenanceError
external = root.parent / 'external-manifest.json'
try:
    freeze_component_manifest(config, source_commit=source, plan_revision=source,
                              output_path=external)
except ProvenanceError:
    pass
else:
    raise AssertionError('corpus escaped its source checkout')
assert not external.exists()
""",
    )


@pytest.mark.parametrize(
    "raw",
    [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b"[" * 1000 + b"0" + b"]" * 1000],
    ids=["duplicate", "nan", "infinity", "deep"],
)
def test_bounded_decoder_rejects_ambiguous_or_deep_json(raw):
    from silent_cascade.errors import ArtifactIntegrityError
    from silent_cascade.train.evidence_types import decode_json

    with pytest.raises(ArtifactIntegrityError):
        decode_json(raw)


def test_component_manifest_reads_above_old_limit_and_rejects_above_32mib(tmp_path):
    from silent_cascade.config import resolve_config
    from silent_cascade.train.config import Phase3Config
    from silent_cascade.train.curriculum_data import load_component_manifest

    from .test_trainer import _manifest
    from .test_training_cli import CONFIGS

    config = resolve_config(Phase3Config, CONFIGS)
    path = _manifest(config, tmp_path / "manifest.json")
    original = path.read_bytes()
    path.write_bytes(original + b" " * (8 * 1024 * 1024))
    assert load_component_manifest(path, config.config).count == 16
    with path.open("r+b") as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    with pytest.raises(ValueError):
        load_component_manifest(path, config.config)


@pytest.mark.parametrize("committed", [False, True])
def test_changed_executable_cannot_reuse_reviewed_source(tmp_path, committed):
    root = checkout(tmp_path / "source")
    execute(
        root,
        SETUP
        + f"""
from silent_cascade.train.provenance import authenticate_source
from silent_cascade.errors import ProvenanceError
subprocess.run(['git','add','component-manifest.json'], check=True)
subprocess.run(['git','commit','-qm','introduce recipe'], check=True)
path = root / 'src/silent_cascade/models/controller.py'
path.write_bytes(path.read_bytes() + b'\\n# changed source\\n')
if {committed!r}:
    subprocess.run(['git','add',str(path)], check=True)
    subprocess.run(['git','commit','-qm','unreviewed model change'], check=True)
try:
    authenticate_source(root, source, source)
except ProvenanceError:
    pass
else:
    raise AssertionError('changed executable authenticated')
""",
    )


@pytest.mark.parametrize("module", ["transitions.py", "weights.py"])
def test_uncommitted_shared_core_cannot_reuse_reviewed_source(tmp_path, module):
    root = checkout(tmp_path / "source")
    execute(
        root,
        SETUP
        + f"""
from silent_cascade.train.provenance import authenticate_source
from silent_cascade.errors import ProvenanceError
path = root / 'src/silent_cascade/models/{module}'
path.write_bytes(path.read_bytes() + b'\\n# uncommitted shared transition mutation\\n')
try:
    authenticate_source(root, source, source)
except ProvenanceError:
    pass
else:
    raise AssertionError('uncommitted shared transition authenticated')
""",
    )


def build_debug_gate(root):
    checkout(root)
    execute(
        root,
        SETUP
        + """
import json
from silent_cascade.train.trainer import run_training
from silent_cascade.train.evidence import collect_component_gate
subprocess.run(['git','add','component-manifest.json'], check=True)
subprocess.run(['git','commit','-qm','introduce fixed corpus'], check=True)
run = root / 'runs' / 'debug'
trained = run_training(config, validation_manifest=manifest_path, run_dir=run,
                       source_commit=source, device='cpu')
weights = run / trained.weights.relative_path
report = collect_component_gate(config, manifest_path=manifest_path, weights_path=weights,
    expected_checkpoint_sha256=trained.weights.file_sha256, training_run=run,
    source_commit=source, plan_revision=source, output_path=root / 'component-gate.json')
assert report.summary.episode_count == 16
subprocess.run(['git','add','component-gate.json'], check=True)
subprocess.run(['git','commit','-qm','archive debug evidence'], check=True)
""",
    )
    return root
