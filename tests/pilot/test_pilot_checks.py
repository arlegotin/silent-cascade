"""Both entrypoints share a package-owned, training-free final execution path."""

import importlib.util
from pathlib import Path

import pytest


def test_shared_driver_is_installed_package_code(tmp_path):
    import os
    import subprocess
    import sys

    assert importlib.util.find_spec("silent_cascade.train.pilot_checks") is not None
    source = (Path(__file__).parents[2] / "scripts/check_phase4_pilot.py").read_text()
    assert "run_pilot_checks" in source
    assert "run_pilot_training" not in source
    package_root = Path(__file__).resolve().parents[2] / "src"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            """
import importlib.abc
import sys
class NoSourceScripts(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'scripts' or fullname.startswith('scripts.'):
            raise AssertionError('installed package imported source-only scripts')
sys.meta_path.insert(0, NoSourceScripts())
from silent_cascade.train.pilot_checks import run_pilot_checks
assert run_pilot_checks.__module__ == 'silent_cascade.train.pilot_checks'
""",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(package_root)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.fixture(scope="session")
def gate_artifact_case(tmp_path_factory):
    from .test_pilot_source import DATA_SETUP, checkout_phase4

    root, execute = checkout_phase4(tmp_path_factory.mktemp("task11-fixture-only") / "repo")
    execute(
        root,
        DATA_SETUP
        + """
import json
import sys
from silent_cascade.train.pilot_workflow import train_pilot
from silent_cascade.train.pilot_checks import run_pilot_checks
from silent_cascade.train.pilot_evidence import verify_phase4_gate_artifact
run = Path('runs/checks')
trained = train_pilot(config_path=Path('configs/train/pilot_smoke.yaml'),
                      manifest_dir=Path('pilot-data'), run_dir=run, device='cpu')
assert trained.progress.global_step == 4
assert trained.selected_checkpoint is None and not trained.gate_eligible
before = (run / 'training-result.json').read_bytes()
serialized = json.loads(before)
assert serialized['schema_version'] == 'phase4-training-result-v2'
assert 'artifact_hashes' not in serialized
assert serialized['artifact_index']['entry_count'] == len(trained.artifact_hashes)
artifact = run_pilot_checks(run_dir=run, config=config, output_path=run/'gate.json')
assert artifact.outcome == 'debug_non_acceptance'
assert artifact.selected_checkpoint_sha256 is None
assert 'no_selected_checkpoint' in artifact.failures
assert artifact.suites['primary']['episode_count'] == 16
assert artifact.suites['primary_repeat']['episode_count'] == 16
assert artifact.repeat_equal
assert all(e['execution']['device'] == 'cpu' for e in artifact.execution_evidence.values()
           if isinstance(e, dict) and 'execution' in e)
assert artifact.continuation_evidence
assert (run / 'training-result.json').read_bytes() == before
again = run_pilot_checks(run_dir=run, config=config, output_path=run/'gate.json')
assert again == artifact
checked = verify_phase4_gate_artifact(run/'gate.json', repo_root=root, raw_run_dir=run)
assert checked['valid'] and not checked['passed']
assert checked['neural_replay'] == 'not_rerun'
completed = subprocess.run([sys.executable, 'scripts/check_phase4_pilot.py',
    '--config', 'configs/train/pilot_smoke.yaml', '--run-dir', str(run),
    '--output', str(run/'gate.json')], capture_output=True, text=True)
assert completed.returncode == 0, completed.stdout + completed.stderr
assert (run / 'training-result.json').read_bytes() == before
""",
        timeout=900,
    )
    return root, execute


CASE_HEADER = """
import json
from pathlib import Path
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.train.pilot_config import resolve_pilot_config
from silent_cascade.train.pilot_evidence import verify_phase4_gate_artifact
from silent_cascade.train.pilot_evidence_types import read_compact_rows
root = Path.cwd()
run = root / 'runs/checks'
config = resolve_pilot_config('phase4_smoke')
payload = json.loads((run/'gate.json').read_bytes())
"""


def test_real_debug_driver_runs_cpu_without_second_fit_and_reuses(gate_artifact_case):
    root, _ = gate_artifact_case
    assert (root / "runs/checks/gate.json").is_file()


@pytest.mark.parametrize(
    "mutation",
    [
        "source_omission",
        "introduction",
        "denominator",
        "conditional_only",
        "cpu_device",
        "stale_weights",
        "missing_replay",
        "hidden_pair",
        "untested_tensors",
        "untested_mps_tensors",
        "debug_as_production",
    ],
)
def test_rehashed_real_gate_rejects_coordinated_claim_tampering(gate_artifact_case, mutation):
    root, execute = gate_artifact_case
    if mutation == "untested_mps_tensors":
        import json

        actual = json.loads((root / "runs/checks/gate.json").read_bytes())
        if not actual["numeric_evidence"]["device_parity"]:
            pytest.skip("CPU-only fixture: native MPS tensor omission not exercised")
    execute(
        root,
        CASE_HEADER
        + f"mutation = {mutation!r}\n"
        + """
if mutation == 'source_omission':
    payload['source']['source_files'].pop('scripts/check_phase4_pilot.py')
    payload['source']['source_sha256'] = sha256_bytes(
        canonical_json_bytes(payload['source']['source_files']))
elif mutation == 'introduction':
    payload['source']['data_introductions']['primary/manifest']['introduction'] = 'a'*40
elif mutation == 'denominator':
    payload['suites']['primary']['safe_count'] -= 1
elif mutation == 'conditional_only':
    payload['suites']['primary']['timed_success_count'] = 10000
elif mutation == 'cpu_device':
    payload['execution_evidence']['primary']['execution']['device'] = 'mps'
elif mutation == 'stale_weights':
    payload['model_state_sha256'] = 'a'*64
elif mutation == 'missing_replay':
    payload['continuation_evidence'][0]['replay_comparisons'] = []
elif mutation == 'hidden_pair':
    payload['pair_rows'].pop()
elif mutation == 'untested_tensors':
    payload['numeric_evidence']['cpu_inventory']['primary']['gradients']['tested_names'].pop()
elif mutation == 'untested_mps_tensors':
    payload['numeric_evidence']['device_parity']['primary']['gradients']['tested_names'].pop()
else:
    payload['config_canonical_json'] = resolve_pilot_config('phase4_pilot').canonical_json.decode()
    payload['outcome'] = 'passed'
    payload['failures'] = []
tampered = run/'tampered-gate.json'
tampered.write_bytes(canonical_json_bytes(payload))
try:
    verify_phase4_gate_artifact(tampered, repo_root=root)
except ValueError:
    pass
else:
    raise AssertionError('coordinated claim accepted: ' + mutation)
""",
    )


def test_rehashed_gate_cannot_omit_a_failed_episode(gate_artifact_case):
    root, execute = gate_artifact_case
    execute(
        root,
        CASE_HEADER
        + """
import gzip
part = payload['attachments']['primary'][0]
rows = list(read_compact_rows(run/part['path']))
failed = next(i for i, item in enumerate(rows) if not item['row']['timed_success'])
rows.pop(failed)
replacement = gzip.compress(b''.join(canonical_json_bytes(row)+b'\\n' for row in rows), mtime=0)
part.update(path='omitted-row.jsonl.gz', rows=len(rows), sha256=sha256_bytes(replacement))
(run/part['path']).write_bytes(replacement)
payload['suites']['primary']['episode_count'] -= 1
tampered = run/'omitted-gate.json'
tampered.write_bytes(canonical_json_bytes(payload))
try:
    verify_phase4_gate_artifact(tampered, repo_root=root)
except ValueError as error:
    assert 'episode inventory' in str(error), str(error)
else:
    raise AssertionError('failed episode omitted from denominator')
""",
    )


@pytest.mark.parametrize("target", ["plan", "source", "foreign_module"])
def test_actual_gate_rejects_changed_execution_closure(gate_artifact_case, target):
    root, execute = gate_artifact_case
    execute(
        root,
        CASE_HEADER
        + f"target = {target!r}\n"
        + """
import sys
import types
from silent_cascade.errors import ProvenanceError
path = None
original = None
if target == 'foreign_module':
    foreign = types.ModuleType('silent_cascade.foreign')
    foreign.__file__ = '/private/tmp/foreign-package.py'
    sys.modules[foreign.__name__] = foreign
else:
    path = root / ('docs/superpowers/plans/2026-09-16-phase-4-autonomous-eventflow.md'
                   if target == 'plan' else 'src/silent_cascade/models/event_flow.py')
    original = path.read_bytes()
    path.write_bytes(original + b'\\n# changed during execution\\n')
try:
    try:
        verify_phase4_gate_artifact(run/'gate.json', repo_root=root)
    except (ValueError, ProvenanceError):
        pass
    else:
        raise AssertionError('changed source accepted')
finally:
    if path is not None:
        path.write_bytes(original)
""",
    )


def test_verification_is_artifact_only_and_portable_scope_is_explicit(gate_artifact_case):
    root, execute = gate_artifact_case
    execute(
        root,
        CASE_HEADER
        + """
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.models.event_flow import EventFlowModel
import silent_cascade.train.pilot_trainer as trainer
import silent_cascade.train.pilot_checks as checks
def forbidden(*args, **kwargs):
    raise AssertionError('artifact verification executed a neural workload')
EventEngine.run_episode = forbidden
EventEngine.step = forbidden
EventFlowModel.forward = forbidden
trainer.pilot_train_one_step = forbidden
checks.record_local_verification = forbidden
portable = verify_phase4_gate_artifact(run/'gate.json', repo_root=root)
assert portable['valid'] and not portable['passed']
assert portable['missing_raw_attachments']
assert portable['neural_replay'] == 'not_rerun'
assert 'numeric.complete_cross_record_validation' in portable['unavailable_semantic_checks']
assert 'training.archives' in portable['unavailable_semantic_checks']
raw = verify_phase4_gate_artifact(run/'gate.json', repo_root=root, raw_run_dir=run)
assert raw['valid'] and not raw['missing_raw_attachments']
assert raw['neural_replay'] == 'not_rerun'
assert raw['unavailable_semantic_checks'] == []
""",
    )


def test_rehashed_row_and_summary_cannot_replace_private_outcome(gate_artifact_case):
    root, execute = gate_artifact_case
    execute(
        root,
        CASE_HEADER
        + """
import gzip
part = payload['attachments']['primary'][0]
rows = list(read_compact_rows(run/part['path']))
record = next(item for item in rows if not item['row']['timed_success'])
record['row']['timed_success'] = True
record['row']['miss_category'] = None
payload['suites']['primary']['timed_success_count'] += 1
replacement = gzip.compress(b''.join(canonical_json_bytes(row)+b'\\n' for row in rows), mtime=0)
part.update(path='substituted-row.jsonl.gz', sha256=sha256_bytes(replacement))
(run/part['path']).write_bytes(replacement)
tampered = run/'substituted-gate.json'
tampered.write_bytes(canonical_json_bytes(payload))
try:
    verify_phase4_gate_artifact(tampered, repo_root=root)
except ValueError as error:
    assert 'timed success' in str(error), str(error)
else:
    raise AssertionError('private failure replaced by rehashed success')
""",
    )


def test_collection_reauthenticates_source_after_reading_all_rows(gate_artifact_case):
    root, execute = gate_artifact_case
    execute(
        root,
        CASE_HEADER
        + """
import silent_cascade.train.pilot_evidence as evidence
path = root/'docs/superpowers/plans/2026-09-16-phase-4-autonomous-eventflow.md'
original = path.read_bytes()
real = evidence.discover_local_verification
def mutate_after_reading(**arguments):
    result = real(**arguments)
    path.write_bytes(original+b'\\n# changed during collection\\n')
    return result
evidence.discover_local_verification = mutate_after_reading
try:
    try:
        evidence.collect_pilot_evidence(run_dir=run, config=config,
                                       output_path=run/'changed-source-gate.json')
    except ValueError as error:
        assert 'source' in str(error), str(error)
    else:
        raise AssertionError('source changed during collection accepted')
finally:
    path.write_bytes(original)
assert not (run/'changed-source-gate.json').exists()
""",
    )


def test_production_cli_separates_eligible_training_from_failed_final_gate(gate_artifact_case):
    root, execute = gate_artifact_case
    execute(
        root,
        CASE_HEADER
        + """
import importlib.util
import sys
from types import SimpleNamespace
spec = importlib.util.spec_from_file_location('pilot_source_cli', root/'scripts/run_pilot.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
# Exercise the CLI status boundary without an unauthorized production fit.
module.run_pilot = lambda **kwargs: SimpleNamespace(
    progress=SimpleNamespace(global_step=4000), gate_eligible=True)
module.resolve_pilot_path = lambda path: SimpleNamespace(
    config=SimpleNamespace(pilot=SimpleNamespace(is_production=True)))
cli_run = root/'runs/cli-status-fixture'
cli_run.mkdir()
payload['outcome'] = 'failed'
(cli_run/'phase4-gate.json').write_bytes(canonical_json_bytes(payload))
sys.argv = ['run_pilot.py', '--config', 'configs/train/pilot.yaml', '--manifest-dir', 'pilot-data',
            '--run-dir', str(cli_run)]
assert module.main() == 1
""",
    )


def test_production_cardinality_gate_serialization_stays_bounded(gate_artifact_case):
    root, execute = gate_artifact_case
    execute(
        root,
        CASE_HEADER
        + """
from silent_cascade.train.pilot_evidence_types import Phase4GateArtifact, MAX_FILE_BYTES
# Size-only fixture: no generated outcome is treated as scientific evidence.
sizes = {name: (256 if name.startswith('delay_') else 10000)
         for name in payload['execution_evidence']}
for name, count in sizes.items():
    identity = payload['execution_evidence'][name]['identity']
    sample = identity['episodes']
    identity['episodes'] = [sample[i % len(sample)] for i in range(count)]
upstream = payload['upstream_artifact_hashes']
for key in list(upstream):
    if key.startswith('final/eval/') and '/episodes/' in key:
        del upstream[key]
for name, count in sizes.items():
    for index in range(count):
        for suffix in ('neural.json', 'telemetry.json', 'trajectory.json.gz'):
            upstream[f'final/eval/{name}/episodes/{index:05d}.{suffix}'] = 'a'*64
index = payload['training_result']['artifact_index']
sample = index['shards'][0]
index['entry_count'] = 3000000
index['shards'] = [dict(sample, path=f'attempt-size-fixture/artifact-index.{start:05d}.jsonl.gz',
                       rows=50000, decompressed_bytes=15000000)
                   for start in range(0,3000000,50000)]
for name, count in sizes.items():
    sample = payload['attachments'][name][0]
    payload['attachments'][name] = [dict(sample, path=f'gate.{name}.{start:05d}.jsonl.gz',
                                       rows=min(1000,count-start))
                                   for start in range(0,count,1000)]
pairs = payload['pair_rows']
payload['pair_rows'] = [pairs[i % len(pairs)] for i in range(256)]
# At most one new record is needed per required coverage obligation. Keep the
# actual retained trace payloads, repeating to all15 possible obligations.
records = payload['continuation_evidence']
payload['continuation_evidence'] = [records[i % len(records)] for i in range(15)]
parsed = Phase4GateArtifact.model_validate_json(canonical_json_bytes(payload))
encoded = canonical_json_bytes(parsed)
assert len(encoded) + 2*1024**2 < MAX_FILE_BYTES  # reserve for the source-bound receipt
(run/'production-cardinality-size.json').write_bytes(canonical_json_bytes(dict(
    scope='serialization fixture only; not model execution or acceptance',
    final_rows=sum(sizes.values()), training_index_entries=3000000,
    gate_bytes=len(encoded), file_limit_bytes=MAX_FILE_BYTES,
    continuation_records=15)))
""",
    )


def test_standalone_refuses_live_owner_and_missing_raw_reuse(gate_artifact_case):
    root, execute = gate_artifact_case
    execute(
        root,
        CASE_HEADER
        + """
from silent_cascade.train.pilot_checks import run_pilot_checks
from silent_cascade.train.pilot_workflow import pilot_ownership
owner = json.loads((run.parent / ('.' + run.name + '.owner.json')).read_bytes())
with pilot_ownership(run, run_identity=owner['run_identity']):
    try:
        run_pilot_checks(run_dir=run, config=config, output_path=run/'gate.json')
    except ValueError as error:
        assert 'live' in str(error)
    else:
        raise AssertionError('competing checker accepted')
path = run/payload['continuation_evidence'][0]['checkpoint_path']
retained = path.with_suffix('.retained-for-test')
path.rename(retained)
try:
    try:
        run_pilot_checks(run_dir=run, config=config, output_path=run/'gate.json')
    except ValueError as error:
        assert 'missing raw' in str(error), str(error)
    else:
        raise AssertionError('missing raw evidence reused')
finally:
    retained.rename(path)
""",
    )


@pytest.mark.parametrize("family", ["numeric", "replay", "numeric_partial", "replay_partial"])
def test_missing_training_sidecar_never_suppresses_present_semantics(gate_artifact_case, family):
    root, execute = gate_artifact_case
    execute(
        root,
        CASE_HEADER
        + f"family = {family!r}\n"
        + """
from silent_cascade.train.pilot_artifact_index import iter_artifact_index
from silent_cascade.eventflow.engine import EventEngine
from silent_cascade.eventflow.replay import _payload_hash
import silent_cascade.train.pilot_verification as numeric
def forbidden(*a, **k):
    raise AssertionError('artifact-only check executed a workload')
EventEngine.step = forbidden
numeric.capture_update = forbidden
numeric.training_objective = forbidden
entries = dict(iter_artifact_index(run, payload['training_result']['artifact_index']))
missing = run/next(p for p in entries if p.endswith('.neural.json'))
retained = missing.with_suffix('.retained-test')
missing.rename(retained)
originals = {}
additionally_missing = []
def replace(path, value):
    originals.setdefault(path, path.read_bytes())
    path.write_bytes(canonical_json_bytes(value))
    payload['upstream_artifact_hashes'][path.relative_to(run).as_posix()] = sha256_bytes(
        path.read_bytes())
try:
    if family == 'numeric_partial':
        path = run/'final/numerics/runtime-cpu/DONE'
        renamed = path.with_suffix('.retained-test')
        path.rename(renamed)
        additionally_missing.append((path, renamed))
    elif family == 'replay_partial':
        path = run/payload['continuation_evidence'][0]['checkpoint_path']
        renamed = path.with_suffix('.retained-test')
        path.rename(renamed)
        additionally_missing.append((path, renamed))
    available = verify_phase4_gate_artifact(run/'gate.json', repo_root=root, raw_run_dir=run)
    assert available['missing_raw_attachments'] and not available['passed']
    unavailable = available['unavailable_semantic_checks']
    assert ('numeric.complete_cross_record_validation' in unavailable) == (
        family == 'numeric_partial')
    if family == 'replay_partial':
        assert 'continuation.checkpoint:' + payload['continuation_evidence'][0][
            'checkpoint_path'] in unavailable
    if family.startswith('numeric'):
        path = run/'final/numerics/primary-cpu.json'
        value = json.loads(path.read_bytes())
        value['update']['compute']['foundation_model_calls'] = 1
        replace(path, value)
        report = payload['numeric_evidence']
        report['artifact_hashes']['primary-cpu.json'] = sha256_bytes(path.read_bytes())
        replace(run/'final/numerics/numeric-report.json', report)
        replace(run/'final/numerics/DONE', {'report_sha256': sha256_bytes(
            (run/'final/numerics/numeric-report.json').read_bytes())})
    else:
        record = payload['continuation_evidence'][0]
        path = run/record['replay_path']
        value = json.loads(path.read_bytes())
        value['source_revision'] = 'f'*40
        value['payload_sha256'] = _payload_hash(value)
        replace(path, value)
        for record in payload['continuation_evidence']:
            if record['replay_path'] == path.relative_to(run).as_posix():
                record['replay_sha256'] = sha256_bytes(path.read_bytes())
    candidate = run/('missing-unrelated-' + family + '.json')
    candidate.write_bytes(canonical_json_bytes(payload))
    try:
        verify_phase4_gate_artifact(candidate, repo_root=root, raw_run_dir=run)
    except ValueError as error:
        assert ('operation' if family.startswith('numeric') else 'replay') in str(error), str(error)
    else:
        raise AssertionError('unrelated absence suppressed present ' + family + ' semantics')
finally:
    for path, raw in originals.items():
        path.write_bytes(raw)
    retained.rename(missing)
    for path, renamed in additionally_missing:
        renamed.rename(path)
""",
    )


@pytest.mark.parametrize("damage", ["missing_archive", "wrong_backup", "existing_destination"])
def test_recovery_refuses_unrestorable_inputs_before_execution(gate_artifact_case, damage):
    root, execute = gate_artifact_case
    execute(
        root,
        CASE_HEADER
        + f"damage = {damage!r}\n"
        + """
import silent_cascade.train.pilot_checks as checks
def forbidden(*a, **k):
    raise AssertionError('recovery executed before authenticating inputs')
checks._run_pilot_checks_owned = forbidden
destination = root/('runs/refused-recovery-' + damage)
backup = root/('retained-refusal-' + damage)
backup.mkdir()
descriptor = payload['training_result']['progress']['latest']
path = run/descriptor['path']
retained = path.with_suffix('.retained-test')
if damage == 'existing_destination':
    destination.mkdir()
else:
    path.rename(retained)
    if damage == 'wrong_backup':
        target = backup/descriptor['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'wrong retained checkpoint')
try:
    try:
        checks.recover_pilot_checks(artifact_path=run/'gate.json', raw_run_dir=run,
            destination=destination, retained_run_dir=backup)
    except ValueError as error:
        assert any(term in str(error) for term in (
            'restore-required', 'hash differs', 'absent')), str(error)
    else:
        raise AssertionError('invalid recovery input accepted')
    if damage != 'existing_destination':
        assert not destination.exists()
finally:
    if retained.exists():
        retained.rename(path)
""",
    )


def test_recovery_rejects_schema_valid_gate_change_after_real_verification(gate_artifact_case):
    root, execute = gate_artifact_case
    execute(
        root,
        CASE_HEADER
        + """
import silent_cascade.train.pilot_checks as checks
from silent_cascade.train.pilot_evidence_types import Phase4GateArtifact
original = (run/'gate.json').read_bytes()
destination = root/'runs/recovery-changed-gate'
real_verify = checks.verify_phase4_gate_artifact
def verified_then_changed(path, **kwargs):
    result = real_verify(path, **kwargs)
    changed = json.loads(path.read_bytes())
    changed['raw_regeneration_command'] += ' --changed-after-verification'
    raw = canonical_json_bytes(changed)
    Phase4GateArtifact.model_validate_json(raw)
    path.write_bytes(raw)
    return result
def forbidden(*a, **k):
    raise AssertionError('changed gate reached numerical execution')
checks.verify_phase4_gate_artifact = verified_then_changed
checks._run_pilot_checks_owned = forbidden
try:
    try:
        checks.recover_pilot_checks(artifact_path=run/'gate.json', raw_run_dir=run,
            destination=destination)
    except ValueError as error:
        assert 'gate changed' in str(error), str(error)
    else:
        raise AssertionError('schema-valid changed gate accepted')
    assert not destination.exists()
finally:
    (run/'gate.json').write_bytes(original)
""",
    )


def test_explicit_recovery_restores_training_and_reruns_final_in_fresh_destination(
    gate_artifact_case,
):
    root, execute = gate_artifact_case
    execute(
        root,
        CASE_HEADER
        + """
import shutil
import sys
from runpy import run_path
import silent_cascade.train.pilot_checks as checks
from silent_cascade.train.pilot_artifact_index import iter_artifact_index
assert callable(checks.recover_pilot_checks)
import silent_cascade.train.pilot_trainer as trainer
def forbidden_fit(*a, **k):
    raise AssertionError('recovery refit training')
trainer.run_pilot_training = forbidden_fit
backup = root/'retained-recovery-inputs'
shutil.copytree(run, backup)
entries = dict(iter_artifact_index(run, payload['training_result']['artifact_index']))
training = next(p for p in entries if p.endswith('.neural.json'))
final = next(p for p in payload['upstream_artifact_hashes'] if p.startswith('final/eval/primary/')
             and p.endswith('.neural.json'))
for name in (training, final):
    path = run/name
    path.rename(path.with_suffix('.retained-test'))
before = {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}
destination = root/'runs/recovered-final'
# This extra fixture proves recovery, not a duplicate native diagnostic.
import silent_cascade.train.pilot_verification as numeric
numeric.native_devices = lambda: (('cpu',), ('mps',))
sys.argv = ['check_phase4_pilot.py', '--recover-from', str(run/'gate.json'),
            '--run-dir', str(run), '--retained-run-dir', str(backup),
            '--destination', str(destination)]
namespace = run_path(str(root/'scripts/check_phase4_pilot.py'))
assert namespace['main']() == 0
observed = json.loads((destination/'phase4-gate.json').read_bytes())
assert observed['source'] == payload['source']
assert observed['selected_weights_sha256'] == payload['selected_weights_sha256']
assert observed['model_state_sha256'] == payload['model_state_sha256']
assert observed['outcome'] == 'debug_non_acceptance'
assert observed['training_result'] == payload['training_result']
intent = json.loads((destination/'recovery-intent.json').read_bytes())
assert intent['original_gate_sha256'] == sha256_bytes((run/'gate.json').read_bytes())
assert intent['source_commit'] == payload['source']['source_commit']
assert intent['selected_weights_sha256'] == payload['selected_weights_sha256']
assert intent['model_state_sha256'] == payload['model_state_sha256']
assert (destination/training).read_bytes() == (backup/training).read_bytes()
assert (destination/final).is_file()
assert verify_phase4_gate_artifact(destination/'phase4-gate.json', repo_root=root,
    raw_run_dir=destination)['missing_raw_attachments'] == []
assert before == {p.relative_to(run): p.read_bytes() for p in run.rglob('*') if p.is_file()}
for name in (training, final):
    path = run/name
    path.with_suffix('.retained-test').rename(path)
""",
        timeout=900,
    )
