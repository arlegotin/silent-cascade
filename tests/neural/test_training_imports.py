"""Fresh offline processes exercise real training/evaluation with optional ML blocked."""

import os
import subprocess
import sys

from .test_training_cli import CONFIGS, ROOT

OFFLINE_SCRIPT = r"""
import importlib.abc, json, pathlib, socket, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = {"mlx", "mlx_lm", "mlx_vlm", "transformers", "huggingface_hub", "qwen"}
        if fullname.split(".")[0] in blocked:
            raise AssertionError("optional import: " + fullname)
sys.meta_path.insert(0, BlockOptional())
def no_network(*args, **kwargs):
    raise AssertionError("network access forbidden")
socket.socket.connect = no_network
socket.socket.connect_ex = no_network
socket.create_connection = no_network
socket.getaddrinfo = no_network
from silent_cascade.train.cli import app
from silent_cascade.config import resolve_config
from silent_cascade.train.config import Phase3Config
from silent_cascade.train.curriculum_data import (
    ComponentManifest, CurriculumKey, make_curriculum_example,
)
import silent_cascade.train.cli as cli
assert pathlib.Path(cli.__file__).is_relative_to(pathlib.Path(sys.argv[1]))
paths = tuple(pathlib.Path(p) for p in sys.argv[3:])
config = resolve_config(Phase3Config, paths)
directory = pathlib.Path(sys.argv[2])
manifest = directory / "manifest.json"
examples = tuple(make_curriculum_example(
    config.config, CurriculumKey("ofd-one-hop-v1", "debug", 313, 337, i, "one_hop")
) for i in range(16))
manifest.write_text(ComponentManifest.from_examples(
    examples, config.config, source_revision="a"*40, plan_revision="b"*40
).model_dump_json())
app(args=["fit", *(v for p in paths for v in ("--config", str(p))),
          "--validation-manifest", str(manifest), "--run-dir", str(directory / "run"),
          "--expected-source-commit", "a"*40, "--device", "cpu"], standalone_mode=False)
result = json.loads(next((directory / "run").glob("attempt-*/result.json")).read_bytes())
weights = result["weights"]
app(args=["evaluate-components", "--weights", str(directory / "run" / weights["relative_path"]),
          "--expected-checkpoint-sha256", weights["file_sha256"], "--manifest", str(manifest),
          "--output", str(directory / "evaluation.json")], standalone_mode=False)
"""


def test_fresh_process_executes_offline_smoke(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            OFFLINE_SCRIPT,
            str(ROOT / "src"),
            str(tmp_path),
            *map(str, CONFIGS),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "evaluation.json").is_file()


def test_model_memory_imports_do_not_reach_training_or_environment_implementations(tmp_path):
    script = """
import importlib.abc, sys
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        private_env = (fullname.startswith("silent_cascade.env.")
                       and fullname != "silent_cascade.env.config")
        if fullname.startswith("silent_cascade.train") or private_env:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Boundary())
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.models.losses import event_flow_loss
from silent_cascade.memory import encoder, tensor_store, retrieval, eviction
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
