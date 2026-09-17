"""Real command boundaries, including strict profile and legacy freeze selection."""

from typer.testing import CliRunner

from silent_cascade.cli import app


def test_pilot_commands_reject_missing_inputs_and_unsupported_profiles(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            "configs/train/pilot_smoke.yaml",
            "--manifest-dir",
            str(tmp_path),
            "--run-dir",
            str(tmp_path / "run"),
            "--seed",
            "23",
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code != 0
    assert "seed" in result.output.lower()
    result = runner.invoke(
        app,
        [
            "report",
            "build",
            "--pilot",
            "--run-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code != 0
    assert "artifact" in result.output.lower()


def test_freeze_keeps_conditional_seed_requirements(tmp_path):
    runner = CliRunner()
    result = runner.invoke(app, ["data", "freeze", "--output", str(tmp_path / "old.json")])
    assert result.exit_code != 0
    assert "root-seed" in result.output
    result = runner.invoke(
        app,
        [
            "data",
            "freeze",
            "--pilot-stage",
            "primary",
            "--config",
            "configs/train/pilot_smoke.yaml",
            "--root-seed",
            "1",
            "--output",
            str(tmp_path / "pilot.json"),
        ],
    )
    assert result.exit_code != 0
    assert "override" in result.output.lower()


def test_real_cli_smoke_training_evaluation_and_replay(tmp_path):
    from .test_pilot_source import DATA_SETUP, checkout

    root, execute = checkout(tmp_path / "repo")
    execute(
        root,
        DATA_SETUP
        + """
from typer.testing import CliRunner
from silent_cascade.cli import app
import json
runner = CliRunner()
run = root / 'runs/cli-smoke'
result = runner.invoke(app, ['train', '--config', 'configs/train/pilot_smoke.yaml',
    '--manifest-dir', 'pilot-data', '--run-dir', str(run), '--seed', '11', '--device', 'cpu'])
assert result.exit_code == 0, (result.output, result.exception)
result_file = next(run.glob('attempt-*/result.json'))
training = json.loads(result_file.read_bytes())
assert training['progress']['global_step'] == 4
descriptor = run / 'latest.json'
descriptor.write_text(json.dumps(training['latest_weights']))
result = runner.invoke(app, ['evaluate', '--checkpoint', str(descriptor),
    '--manifest', 'pilot-data/primary.json', '--output', str(run / 'eval/primary'),
    '--device', 'cpu'])
assert result.exit_code == 0, (result.output, result.exception)
assert json.loads((run / 'eval/primary/metrics.json').read_bytes())['episode_count'] == 16
replay = run / 'eval/primary/replay.json'
assert replay.is_file()
result = runner.invoke(app, ['replay', str(replay), '--weights',
    str(run / training['latest_weights']['path']), '--json'])
assert result.exit_code == 0, (result.output, result.exception)
assert json.loads(result.output)['matched'] is True
assert 'trace' not in json.loads(result.output) and 'result' not in json.loads(result.output)
# A concrete checkpoint descriptor must remain bound to the trainer's result.
for change in ({'global_step': 99}, {'eligible': True}, {'path': '../outside'}):
    descriptor.write_text(json.dumps(training['latest_weights'] | change))
    rejected = runner.invoke(app, ['evaluate', '--checkpoint', str(descriptor),
        '--manifest', 'pilot-data/primary.json', '--output', str(run / 'eval/rejected'),
        '--device', 'cpu'])
    assert rejected.exit_code != 0, change
    assert not (run / 'eval/rejected').exists()
""",
    )
