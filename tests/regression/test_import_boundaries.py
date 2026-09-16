import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    EpisodeBundle,
    EpisodeKey,
    EpisodeRecipe,
    EpisodeTruth,
    EpisodeVariant,
    MatchedEpisodeCoordinate,
    PublicEpisode,
    public_projection,
)
from silent_cascade.schemas import (
    ActivationPayload,
    AgentInit,
    ExternalEvent,
    ExternalEventKind,
    LinkFact,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src"


def _module_name(path: Path) -> str:
    return ".".join(path.relative_to(SOURCE_ROOT).with_suffix("").parts)


def imported_modules(path: Path) -> set[str]:
    """Return imports from source syntax without executing module side effects."""
    module = _module_name(path)
    package = module.split(".")[:-1]
    imported: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package) - node.level + 1
                base_parts = package[:keep]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base:
                imported.add(base)
            imported.update(f"{base}.{alias.name}" if base else alias.name for alias in node.names)
    return imported


def test_core_imports_do_not_load_optional_qwen_modules() -> None:
    program = r"""
import importlib.abc
import sys

blocked = {"mlx", "mlx_vlm", "huggingface_hub"}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError(f"optional import attempted: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
import silent_cascade
import silent_cascade.config
import silent_cascade.rng
import silent_cascade.io
import silent_cascade.doctor
import silent_cascade.cli
import silent_cascade.eventflow.engine
import silent_cascade.eventflow.scripted
import silent_cascade.eventflow.replay
import silent_cascade.eventflow.checkpoint
assert not blocked.intersection(sys.modules)
assert not any("qwen" in name.lower() for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_generator_and_invariants_do_not_import_oracle() -> None:
    generator_imports = imported_modules(SOURCE_ROOT / "silent_cascade/env/generator.py")
    invariant_imports = imported_modules(SOURCE_ROOT / "silent_cascade/env/invariants.py")
    assert "silent_cascade.env.oracle" not in generator_imports
    assert "silent_cascade.env.oracle" not in invariant_imports
    assert "silent_cascade.env.generator" not in invariant_imports


def test_scripted_codec_dispatch_keeps_neural_training_modules_unloaded(tmp_path):
    program = """
import importlib.abc
import importlib.util
import sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('silent_cascade.train', 'silent_cascade.models',
                                'silent_cascade.eventflow.neural')):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Blocker())
from pathlib import Path
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.scripted import ScriptedEventFlowAgent
from silent_cascade.eventflow.checkpoint import load_runtime_checkpoint, restore_runtime_session
from silent_cascade.errors import DynamicsError
from silent_cascade.config import resolve_config
from silent_cascade.eventflow.config import Phase2Config
paths = ('configs/base.yaml', 'configs/data/primary.yaml', 'configs/model/event_flow.yaml')
config = resolve_config(Phase2Config, [Path(p) for p in paths]).config.event_flow
engine = EventEngine(config, crash_root=Path(sys.argv[1]), source_revision='a' * 40)
agent = ScriptedEventFlowAgent()
spec = importlib.util.spec_from_file_location('scripted_fixtures', 'tests/conftest.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
session = engine.start_episode(fixtures.bundle_for(fixtures.CASES[0]), agent)
def fail(state, event):
    raise DynamicsError('injected callback failure')
agent.on_external = fail
try:
    engine.step(session, agent)
except DynamicsError:
    pass
else:
    raise AssertionError('scripted callback did not fail')
archives = list(Path(sys.argv[1]).glob('*.safetensors'))
assert len(archives) == 1
assert len(list(Path(sys.argv[1]).glob('*.json'))) == 1
artifact = load_runtime_checkpoint(archives[0], config=config, source_revision='a' * 40)
restored, restored_agent = restore_runtime_session(
    artifact, config=config, source_revision='a' * 40, restore_rng=False)
assert engine.step(restored, restored_agent) is False
"""
    completed = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_oracle_does_not_import_generator_or_invariants() -> None:
    modules = imported_modules(SOURCE_ROOT / "silent_cascade/env/oracle.py")
    assert "silent_cascade.env.generator" not in modules
    assert "silent_cascade.env.invariants" not in modules


def test_agent_facing_modules_cannot_import_private_environment_or_foundations() -> None:
    """Any later agent-facing module is picked up without importing it at runtime."""
    package = SOURCE_ROOT / "silent_cascade"
    forbidden = {
        "silent_cascade.env.generator",
        "silent_cascade.env.invariants",
        "silent_cascade.env.oracle",
        "silent_cascade.env.services",
        "silent_cascade.env.episode",
        "silent_cascade.env.reward",
        "silent_cascade.eventflow.engine",
        "silent_cascade.eventflow.scheduling",
        "mlx",
        "mlx_vlm",
        "huggingface_hub",
    }
    for relative in ("eventflow", "memory", "models", "eval/conditions"):
        directory = package / relative
        for path in directory.rglob("*.py") if directory.exists() else ():
            # Private infrastructure has narrowly scoped environment access.
            relative_path = path.relative_to(package).as_posix()
            allowed = set()
            if relative_path == "eventflow/engine.py":
                allowed = {
                    "silent_cascade.env.episode",
                    "silent_cascade.env.reward",
                    "silent_cascade.eventflow.scheduling",
                }
            elif relative_path == "eventflow/scheduling.py":
                allowed = {"silent_cascade.env.episode"}
            elif relative_path in {
                "eventflow/replay.py",
                "eventflow/checkpoint.py",
                "eventflow/checkpoint_state.py",
                "eventflow/neural_checkpoint.py",
                "eventflow/neural_replay.py",
            }:
                allowed = {
                    "silent_cascade.env.episode",
                    "silent_cascade.env.reward",
                    "silent_cascade.eventflow.engine",
                    "silent_cascade.eventflow.scheduling",
                }
            elif relative_path == "eventflow/evidence.py":
                # Validation-private collector; never imported by an agent.
                allowed = {
                    "silent_cascade.env.episode",
                    "silent_cascade.env.reward",
                    "silent_cascade.env.services",
                    "silent_cascade.eventflow.engine",
                }
            modules = imported_modules(path)
            assert not any(
                module == blocked or module.startswith(f"{blocked}.")
                for blocked in forbidden - allowed
                for module in modules
            ), path


def test_engine_private_environment_imports_are_limited_to_episode_and_reward() -> None:
    path = SOURCE_ROOT / "silent_cascade/eventflow/engine.py"
    allowed = {"silent_cascade.env.episode", "silent_cascade.env.reward"}
    private = {name for name in imported_modules(path) if name.startswith("silent_cascade.env")}
    assert private
    assert all(
        any(name == base or name.startswith(f"{base}.") for base in allowed) for name in private
    )


def test_scripted_agent_has_no_environment_imports_or_private_symbols() -> None:
    path = SOURCE_ROOT / "silent_cascade/eventflow/scripted.py"
    assert not any("env" in name.split(".") for name in imported_modules(path))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {"oracle", "EpisodeTruth", "EpisodeBundle"}
    symbols = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            symbols.add(node.id)
        elif isinstance(node, ast.Attribute):
            symbols.add(node.attr)
        elif isinstance(node, ast.alias):
            symbols.update(node.name.split("."))
            if node.asname:
                symbols.add(node.asname)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.arg):
            symbols.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # Include string annotations and dynamic-import module names.
            symbols.update(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", node.value))
    assert not forbidden.intersection(symbols)


def test_public_projection_spy_observes_no_private_truth_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default public projection must not consult private truth even transiently."""
    public = PublicEpisode(
        init=AgentInit("00000000-0000-4000-8000-000000000001", 64, 4, 0.0),
        events=(
            ExternalEvent(0, 0.5, ExternalEventKind.FACT, LinkFact(1, 2)),
            ExternalEvent(
                1,
                1.0,
                ExternalEventKind.ACTIVATE,
                ActivationPayload(1),
            ),
        ),
    )
    truth = EpisodeTruth(
        key=EpisodeKey(
            "ofd-v1",
            SplitNamespace.DEBUG,
            SuiteName.IID_PRIMARY,
            998_877,
            MatchedEpisodeCoordinate("matched", 0, 0),
        ),
        recipe=EpisodeRecipe(
            1,
            EpisodeVariant.DISCONNECTED_NEGATIVE,
            0,
            SuiteName.IID_PRIMARY,
            0,
        ),
        relevant_node_path=(1, 2),
        relevant_record_ids=(0,),
        terminal_record_id=None,
        relevant_hazard_type=None,
        private_terminal=ExternalEvent(2, 2.0, ExternalEventKind.END, None),
        activation_time=1.0,
        episode_delay=1.0,
        action_window_start=None,
        action_window_end=None,
        action_target=None,
        rejection_count=0,
        rejection_reasons=("PRIVATE-ROOT-SEED-998877",),
    )
    bundle = EpisodeBundle(public, truth)
    private_reads: list[str] = []
    original_getattribute = EpisodeBundle.__getattribute__

    def observe_private_read(instance: EpisodeBundle, name: str) -> object:
        if name == "truth":
            private_reads.append(name)
        return original_getattribute(instance, name)

    monkeypatch.setattr(EpisodeBundle, "__getattribute__", observe_private_read)

    assert public_projection(bundle) is public
    assert private_reads == []
