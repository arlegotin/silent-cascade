"""Real tiny pilot execution under network and optional-model denial hooks."""

import pytest


def test_fresh_offline_boundary_denies_cloud_and_subprocess_escape(tmp_path):
    import json
    import os
    import subprocess
    import sys

    from silent_cascade.train import pilot_offline

    program = r"""
import importlib, json, os, socket, subprocess, sys
from silent_cascade.train.pilot_offline import install_offline_boundary
attempts, blocked = install_offline_boundary()
denied = []
for name, operation in [
    ('dns', lambda: socket.getaddrinfo('example.invalid', 443)),
    ('socket', lambda: socket.socket().connect(('127.0.0.1', 9))),
    ('cloud', lambda: importlib.import_module('boto3')),
    ('aws', lambda: subprocess.run(['aws', '--version'])),
    ('shell', lambda: subprocess.run('true', shell=True)),
    ('python', lambda: subprocess.run([sys.executable, '-c', 'pass'])),
    ('system', lambda: os.system('true')),
]:
    try:
        operation()
    except RuntimeError:
        denied.append(name)
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
print(json.dumps({'denied': denied, 'revision': revision,
                  'credential': os.environ.get('AWS_SECRET_ACCESS_KEY')}))
"""
    environment = pilot_offline.offline_environment(tmp_path)
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    result = subprocess.run(
        [sys.executable, "-B", "-c", program],
        env={**environment, "AWS_SECRET_ACCESS_KEY": "fixture-secret"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["denied"] == ["dns", "socket", "cloud", "aws", "shell", "python", "system"]
    assert len(payload["revision"]) == 40
    assert payload["credential"] is None
    assert (
        "AWS_SECRET_ACCESS_KEY" not in os.environ
        or os.environ["AWS_SECRET_ACCESS_KEY"] != "fixture-secret"
    )


def test_offline_child_cannot_write_outside_counted_workspace(tmp_path):
    import subprocess
    import sys

    from silent_cascade.train.pilot_offline import offline_environment

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    program = r"""
import os, sys
from pathlib import Path
from silent_cascade.train.pilot_offline import install_offline_boundary
root = Path(sys.argv[1])
install_offline_boundary(workspace_root=root)
(root / 'retained.txt').write_text('inside')
fd = os.open(root, os.O_RDONLY)
try:
    nested = os.open('descriptor.txt', os.O_WRONLY | os.O_CREAT, dir_fd=fd)
    os.write(nested, b'inside')
    os.close(nested)
finally:
    os.close(fd)
for path in (root.parent / 'escaped.txt', root / '..' / 'escaped.txt'):
    try:
        path.write_text('outside')
    except RuntimeError:
        pass
    else:
        raise AssertionError('uncounted output')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, str(allowed)],
        env=offline_environment(allowed),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (allowed / "retained.txt").read_text() == "inside"
    assert (allowed / "descriptor.txt").read_bytes() == b"inside"
    assert not (tmp_path / "escaped.txt").exists()


def test_only_exact_nested_offline_diagnostic_is_admitted(tmp_path):
    import subprocess
    import sys

    from silent_cascade.train.pilot_offline import offline_environment

    program = r"""
import os, subprocess, sys
from pathlib import Path
from silent_cascade.train import pilot_offline as module
root = Path(sys.argv[1])
repo = Path(module.__file__).resolve().parents[3]
module.install_offline_boundary(workspace_root=root)
output = root / 'nested'
argv = [sys.executable, '-B', '-c', module._PROGRAM, str(output)]
environment = module.offline_environment(output)
mutations = [
    (argv[:3] + [module._PROGRAM + ' ', str(output)], environment, repo),
    (argv + ['extra'], environment, repo),
    (argv[:4] + [str(root.parent / 'escape')],
     module.offline_environment(root.parent / 'escape'), repo),
    (argv, dict(environment, AWS_SECRET_ACCESS_KEY='fixture'), repo),
    (argv, dict(environment, PYTHONPATH='/tmp'), repo),
    (argv, environment, root),
]
for changed, env, cwd in mutations:
    try:
        subprocess.run(changed, env=env, cwd=cwd, check=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError('altered nested diagnostic accepted')
result = module.measure_pilot_offline(output_dir=output)
assert result.foundation_model_calls == 0
print('exact nested diagnostic passed')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, str(tmp_path)],
        env=offline_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "exact nested diagnostic passed"


@pytest.fixture
def verification():
    from silent_cascade.train import pilot_verification

    return pilot_verification


def test_offline_probe_runs_training_evaluation_replay_and_artifact_report(verification, tmp_path):
    result = verification.measure_pilot_offline(output_dir=tmp_path / "offline")
    assert result.training_updates == 1
    assert result.evaluation_episodes == 16
    assert result.replayed_episodes == 1
    assert result.artifact_reports == 1
    assert result.backward_macs > 0
    assert result.foundation_model_calls == 0
    assert result.network_attempts == result.optional_import_attempts == 0
    assert result.forbidden_modules == ()
