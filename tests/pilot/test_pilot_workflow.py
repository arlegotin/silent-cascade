"""Real immutable input introduction, restart and process ownership boundaries."""

import os

import pytest


def test_new_smoke_inputs_stop_before_fit_without_committing(tmp_path):
    from .test_pilot_source import checkout

    root, execute = checkout(tmp_path / "repo")
    execute(
        root,
        """
from pathlib import Path
import subprocess
import json
from typer.testing import CliRunner
from silent_cascade.cli import app
from silent_cascade.train.pilot_workflow import run_pilot
root = Path.cwd()
before = subprocess.check_output(['git','rev-parse','HEAD'])
frozen = CliRunner().invoke(app, ['data', 'freeze', '--pilot-stage', 'one_hop',
    '--config', 'configs/train/pilot_smoke.yaml', '--output', 'pilot-data/one_hop.json', '--json'])
assert frozen.exit_code == 0, (frozen.output, frozen.exception)
assert json.loads(frozen.output)['count'] == 16
try:
    run_pilot(config_path=Path('configs/train/pilot_smoke.yaml'),
              manifest_dir=Path('pilot-data'), run_dir=Path('runs/smoke'), device='cpu')
except ValueError as error:
    assert 'commit-data precondition' in str(error), str(error)
    assert 'pilot-data/primary.json' in str(error)
    assert 'pilot-data/audits/primary/report.json' in str(error)
else:
    raise AssertionError('unintroduced data trained')
assert subprocess.check_output(['git','rev-parse','HEAD']) == before
assert not Path('runs/smoke').exists()
assert len(list(Path('pilot-data/audits').glob('*/report.json'))) == 4
""",
    )


def test_real_smoke_workflow_reuses_finished_artifacts(tmp_path):
    from .test_pilot_source import checkout

    root, execute = checkout(tmp_path / "repo")
    execute(
        root,
        """
import json
import subprocess
import sys
from pathlib import Path
from silent_cascade.train.pilot_workflow import run_pilot, _restore_result
args = dict(config_path=Path('configs/train/pilot_smoke.yaml'),
            manifest_dir=Path('pilot-data'), run_dir=Path('runs/smoke'), device='cpu')
try:
    run_pilot(**args)
except ValueError as error:
    assert 'commit-data precondition' in str(error)
else:
    raise AssertionError('unintroduced input fit')
stages = ('one_hop', 'two_hop', 'primary', 'robustness')
paths = [path for stage in stages for path in (
    f'pilot-data/{stage}.json', f'pilot-data/audits/{stage}/report.json')]
subprocess.run(['git', 'add', *paths], check=True)
subprocess.run(['git', 'commit', '-qm', 'introduce exact canonical debug inputs'], check=True)
subprocess.run(['git', 'commit', '--allow-empty', '-qm', 'compatible training revision'],
               check=True)
completed = subprocess.run([sys.executable, 'scripts/run_pilot.py',
    '--config', 'configs/train/pilot_smoke.yaml', '--manifest-dir', 'pilot-data',
    '--run-dir', 'runs/smoke', '--device', 'cpu'], capture_output=True, text=True)
assert completed.returncode == 0, completed.stdout + completed.stderr
assert 'production learning gate=unmet' in completed.stdout
first = _restore_result(json.loads(Path('runs/smoke/training-result.json').read_bytes()))
assert first.progress.global_step == 4 and not first.gate_eligible
before = {str(p): p.read_bytes() for p in Path('runs/smoke').rglob('*') if p.is_file()}
second = run_pilot(**args)
assert second == first
after = {str(p): p.read_bytes() for p in Path('runs/smoke').rglob('*') if p.is_file()}
assert before == after
assert Path('runs/smoke/report/report.md').is_file()
try:
    run_pilot(**(args | {'device': 'mps'}))
except ValueError as error:
    assert 'new-run-path' in str(error)
else:
    raise AssertionError('incompatible directory accepted')
""",
    )


def test_ownership_distinguishes_start_and_run_identity(tmp_path):
    from silent_cascade.train.pilot_workflow import pilot_ownership, process_identity

    path = tmp_path / "run"
    assert process_identity(os.getpid()) is not None
    with pilot_ownership(path, run_identity="a" * 64):
        with pytest.raises(ValueError, match="live"), pilot_ownership(path, run_identity="a" * 64):
            pass
        with (
            pytest.raises(ValueError, match="identity"),
            pilot_ownership(path, run_identity="b" * 64),
        ):
            pass


def test_stale_pid_reuse_requires_durable_recovery_and_keeps_other_process(tmp_path):
    import json

    from silent_cascade.train.pilot_workflow import pilot_ownership, process_identity

    run = tmp_path / "run"
    run.mkdir()
    lock = tmp_path / ".run.owner.json"
    current = process_identity(os.getpid())
    lock.write_text(
        json.dumps(
            dict(
                pid=os.getpid(),
                process_start={"create_time": current["create_time"] - 1},
                run_identity="a" * 64,
                active=True,
            )
        )
    )
    with (
        pytest.raises(ValueError, match="durable checkpoint"),
        pilot_ownership(run, run_identity="a" * 64),
    ):
        pass
    assert process_identity(os.getpid()) == current
    # A stale record alone cannot authorize continuation; actual workflows supply
    # the checkpoint authenticator. The missing run has no durable state to recover.
    run.rmdir()
    with pilot_ownership(run, run_identity="a" * 64):
        assert json.loads(lock.read_text())["process_start"] == current


def test_ownership_unknown_process_is_not_assumed_dead(tmp_path, monkeypatch):
    import psutil

    from silent_cascade.train.pilot_workflow import process_identity

    def inaccessible(pid):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(psutil, "Process", inaccessible)
    with pytest.raises(ValueError, match="cannot inspect process identity"):
        process_identity(123)


@pytest.mark.parametrize("location", ["run-parent", "owner"])
def test_ownership_never_follows_links(tmp_path, location):
    from silent_cascade.train.pilot_workflow import pilot_ownership

    real = tmp_path / "real"
    real.mkdir()
    if location == "run-parent":
        linked = tmp_path / "linked"
        linked.symlink_to(real, target_is_directory=True)
        run = linked / "run"
    else:
        run = tmp_path / "run"
        (real / "owner").write_text("{}")
        (tmp_path / ".run.owner.json").symlink_to(real / "owner")
    with pytest.raises((ValueError, OSError)), pilot_ownership(run, run_identity="a" * 64):
        raise AssertionError("symbolic ownership path followed")


def test_interrupted_smoke_resumes_last_durable_archive(tmp_path):
    from .test_pilot_source import DATA_SETUP, checkout

    root, execute = checkout(tmp_path / "repo")
    execute(
        root,
        DATA_SETUP
        + """
from silent_cascade.train.pilot_workflow import run_pilot
import silent_cascade.train.pilot_trainer as trainer
original = trainer.pilot_train_one_step
calls = 0
def interrupted(*args, **kwargs):
    global calls
    calls += 1
    if calls == 3:
        raise RuntimeError('injected interruption after durable step 2')
    return original(*args, **kwargs)
trainer.pilot_train_one_step = interrupted
args = dict(config_path=Path('configs/train/pilot_smoke.yaml'), manifest_dir=Path('pilot-data'),
            run_dir=Path('runs/interrupted'), device='cpu')
try:
    run_pilot(**args)
except RuntimeError as error:
    assert 'injected interruption' in str(error)
else:
    raise AssertionError('interruption not observed')
trainer.pilot_train_one_step = original
result = run_pilot(**args)
assert result.progress.global_step == 4
assert len(list(Path('runs/interrupted').glob('attempt-*/failure.json'))) == 1
assert len(list(Path('runs/interrupted').glob('restart-*.json'))) == 1
assert Path('runs/interrupted/report/report.md').is_file()
""",
    )


def test_process_identity_tracks_actual_child_lifetime():
    import subprocess
    import sys

    from silent_cascade.train.pilot_workflow import process_identity

    with subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    ) as child:
        start = process_identity(child.pid)
        assert start is not None and start["create_time"] > 0
        assert process_identity(child.pid) == start
        child.stdin.close()
        child.wait(timeout=10)
        assert process_identity(child.pid) is None


def test_malformed_owner_cannot_be_reclaimed(tmp_path):
    import json

    from silent_cascade.train.pilot_workflow import pilot_ownership

    (tmp_path / ".run.owner.json").write_text(
        json.dumps({"pid": os.getpid(), "run_identity": "a" * 64, "active": False})
    )
    with (
        pytest.raises(ValueError, match="ownership record"),
        pilot_ownership(tmp_path / "run", run_identity="a" * 64),
    ):
        pass


def test_competing_run_is_refused_before_any_input_preparation(tmp_path):
    from .test_pilot_source import checkout

    root, execute = checkout(tmp_path / "repo")
    execute(
        root,
        """
from pathlib import Path
import silent_cascade.train.pilot_workflow as workflow
def forbidden(*args, **kwargs):
    raise AssertionError('competing writer reached freeze/audit preparation')
workflow._inputs = forbidden
run = Path('runs/concurrent')
with workflow.pilot_ownership(run, run_identity='a' * 64):
    try:
        workflow.run_pilot(config_path=Path('configs/train/pilot_smoke.yaml'),
            manifest_dir=Path('pilot-data'), run_dir=run, device='cpu')
    except ValueError as error:
        assert 'ownership' in str(error) or 'live' in str(error)
    else:
        raise AssertionError('competing invocation entered workflow')
assert not Path('pilot-data').exists()
""",
    )


def test_recovered_attempt_result_must_match_durable_checkpoint(tmp_path):
    from .test_pilot_source import DATA_SETUP, checkout

    root, execute = checkout(tmp_path / "repo")
    execute(
        root,
        DATA_SETUP
        + """
import json
from silent_cascade.train.pilot_workflow import train_pilot
args = dict(config_path=Path('configs/train/pilot_smoke.yaml'), manifest_dir=Path('pilot-data'),
            run_dir=Path('runs/recovery'), device='cpu')
first = train_pilot(**args)
wrapper = Path('runs/recovery/training-result.json')
original = wrapper.read_bytes()
wrapper.unlink()  # trainer result durable, wrapper publication interrupted
attempt = next(Path('runs/recovery').glob('attempt-*/result.json'))
raw = attempt.read_bytes()
payload = json.loads(raw)
payload['status'] = payload['progress']['status'] = 'robustness_complete'
payload['gate_eligible'] = True
attempt.write_text(json.dumps(payload))
try:
    train_pilot(**args)
except ValueError as error:
    assert 'last durable checkpoint' in str(error), str(error)
else:
    raise AssertionError('checkpoint-inconsistent recovered result accepted')
assert not wrapper.exists(), 'unverified result was republished'
attempt.write_bytes(raw)
assert train_pilot(**args) == first
assert wrapper.read_bytes() == original
""",
    )


INTERRUPTED_VALIDATION_SETUP = """
import json
from silent_cascade.train.pilot_workflow import run_pilot, train_pilot
from silent_cascade.hashing import sha256_bytes
import silent_cascade.eval.runner as runner
original = runner._run_one
calls = 0
interruption_call = globals().get('validation_interruption_call', 2)
def interrupted(*args, **kwargs):
    global calls
    calls += 1
    if calls == interruption_call:
        raise RuntimeError('injected interruption during validation')
    if globals().get('fail_first_row', False):
        from silent_cascade.models.event_flow import EventFlowModel
        from silent_cascade.models.errors import NeuralError
        compose = EventFlowModel.compose
        def failed_compose(self, context):
            raise NeuralError('injected retained partial dynamics error')
        EventFlowModel.compose = failed_compose
        try:
            return original(*args, **kwargs)
        finally:
            EventFlowModel.compose = compose
    return original(*args, **kwargs)
runner._run_one = interrupted
args = dict(config_path=Path('configs/train/pilot_smoke.yaml'), manifest_dir=Path('pilot-data'),
            run_dir=Path('runs/interrupted-validation'), device='cpu')
try:
    run_pilot(**args)
except RuntimeError as error:
    assert 'injected interruption during validation' in str(error)
else:
    raise AssertionError('interruption not observed')
runner._run_one = original
run = args['run_dir']
partial = next(p for p in run.glob('attempt-*/validation-*/autonomous')
               if not (p / 'DONE').exists())
assert (partial / 'identity.json').is_file() and not (partial / 'DONE').exists()
assert json.loads((run / 'checkpoint-index.json').read_bytes())['latest']['global_step'] == (
    2 if interruption_call == 18 else 0)
assert len((partial / '.rows.pending.jsonl').read_bytes().splitlines()) == 1
retained = {str(p.relative_to(run)): sha256_bytes(p.read_bytes())
            for p in partial.rglob('*') if p.is_file()}
assert any(name.endswith('.trajectory.json.gz') for name in retained)
Path('partial-evidence.json').write_text(json.dumps(retained))
result = train_pilot(**args)
assert result.progress.global_step == 4
assert len(list(run.glob('restart-*.json'))) == 1
"""


def test_interrupted_validation_finishes_workflow_without_discarding_evidence(tmp_path):
    from .test_pilot_source import DATA_SETUP, checkout

    root, execute = checkout(tmp_path / "repo")
    execute(
        root,
        DATA_SETUP
        + INTERRUPTED_VALIDATION_SETUP
        + """
result = run_pilot(**args)
assert result.progress.global_step == 4 and not result.gate_eligible
table = json.loads((run / 'report/tables.json').read_bytes())
assert len(table['evaluations']) == 3
partial = table['incomplete_evaluations']
assert len(partial) == 1
assert partial[0]['status'] == 'abandoned_incomplete'
assert partial[0]['planned_episodes'] == 16
assert partial[0]['retained_rows'] == 1
assert partial[0]['unknown_episodes'] == 15
assert 'timed_success_count' not in partial[0]
for name, digest in retained.items():
    assert sha256_bytes((run / name).read_bytes()) == digest
before = {str(p): p.read_bytes() for p in run.rglob('*') if p.is_file()}
assert run_pilot(**args) == result
assert before == {str(p): p.read_bytes() for p in run.rglob('*') if p.is_file()}
""",
    )
