from pathlib import Path

import pytest


def checkout(path):
    import importlib.util

    helper = Path(__file__).parents[1] / "neural/test_phase3_provenance.py"
    spec = importlib.util.spec_from_file_location("phase3_test_utilities", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.checkout(path), module.execute


def checkout_phase4(path):
    """Run historical gate fixtures against the accepted Phase 4 source closure."""
    import io
    import subprocess
    import tarfile

    revision = "3b132253512f02c0ed9f2d774fefce3036019d91"
    repository = Path(__file__).resolve().parents[2]
    path.mkdir()
    archive = subprocess.check_output(["git", "archive", "--format=tar", revision], cwd=repository)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
        source.extractall(path, filter="data")
    for args in (
        ("init", "-q"),
        ("config", "user.email", "test@example.invalid"),
        ("config", "user.name", "Local test"),
        ("add", "."),
        ("commit", "-qm", "accepted Phase 4 source"),
    ):
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)
    import importlib.util

    helper = Path(__file__).parents[1] / "neural/test_phase3_provenance.py"
    spec = importlib.util.spec_from_file_location("phase3_test_utilities", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return path, module.execute


DATA_SETUP = """
from pathlib import Path
import subprocess
from silent_cascade.train.pilot_config import resolve_pilot_config
from silent_cascade.train.pilot_data import freeze_pilot_manifest
from silent_cascade.eval.pilot_audit import audit_pilot_manifest
from silent_cascade.train.pilot_provenance import authenticate_pilot_source, verify_pilot_data
root = Path.cwd()
config = resolve_pilot_config('phase4_smoke')
producer = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
authenticate_pilot_source(repo_root=root, source_commit=producer, config=config)
manifests = {}
for stage in ('one_hop','two_hop','primary','robustness'):
    path = root / 'pilot-data' / (stage + '.json')
    manifest = freeze_pilot_manifest(config, stage=stage, output_path=path, source_commit=producer)
    audit_pilot_manifest(manifest, config=config.config, output_dir=path.parent / 'audits' / stage)
    manifests[stage] = path
subprocess.run(['git','add','pilot-data'], check=True)
subprocess.run(['git','commit','-qm','introduce immutable debug data'], check=True)
subprocess.run(['git','commit','--allow-empty','-qm','training attempt'], check=True)
source = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
"""


def test_historical_phase4_checkout_has_exact_accepted_package_inventory(tmp_path):
    from silent_cascade.train.pilot_evidence import REQUIRED_PACKAGE_FILES

    root, _ = checkout_phase4(tmp_path / "phase4")
    actual = {str(path.relative_to(root)) for path in (root / "src/silent_cascade").rglob("*.py")}
    assert actual == set(REQUIRED_PACKAGE_FILES)


def test_real_introduction_chain_and_dirty_archive_helper(tmp_path):
    root, execute = checkout(tmp_path / "repo")
    execute(
        root,
        DATA_SETUP
        + """
loaded, introductions = verify_pilot_data(repo_root=root, source_commit=source,
                                          config=config, manifests=manifests)
assert len(loaded) == 4 and len(introductions) == 8
assert all(i['producer'] == producer and i['introduction'] != producer
           for i in introductions.values())
import json
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
audit_path = root / 'pilot-data/audits/one_hop/report.json'
original = audit_path.read_bytes()
for mutation in ('omitted', 'additional', 'unsafe', 'substituted'):
    report = json.loads(original)
    if mutation == 'omitted':
        report['executing_source_files'].pop(next(iter(report['executing_source_files'])))
    elif mutation == 'additional':
        report['executing_source_files']['invented.py'] = 'a' * 64
    elif mutation == 'unsafe':
        report['executing_source_files']['../escaped.py'] = 'a' * 64
    else:
        report['executing_source_files']['__init__.py'] = 'a' * 64
    report['executing_source_sha256'] = sha256_bytes(
        canonical_json_bytes(report['executing_source_files']))
    audit_path.write_bytes(canonical_json_bytes(report))
    try:
        verify_pilot_data(repo_root=root, source_commit=source, config=config, manifests=manifests)
    except ValueError:
        pass
    else:
        raise AssertionError(mutation + ' audit accepted')
audit_path.write_bytes(original)
helper = root / 'src/silent_cascade/train/archive_tensors.py'
helper.write_text(helper.read_text() + '\\n# dirt\\n')
try:
    authenticate_pilot_source(repo_root=root, source_commit=source, config=config)
except ValueError as error:
    assert 'archive_tensors.py' in str(error)
else:
    raise AssertionError('dirty extracted helper accepted')
helper.unlink()
helper.symlink_to('objective.py')
subprocess.run(['git','add',str(helper)], check=True)
subprocess.run(['git','commit','-qm','invalid symbolic source blob'], check=True)
bad_source = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
try:
    authenticate_pilot_source(repo_root=root, source_commit=bad_source, config=config)
except ValueError as error:
    assert 'regular Git blobs' in str(error)
else:
    raise AssertionError('symbolic source blob accepted')
""",
    )


def test_uncommitted_executing_source_is_rejected(tmp_path):
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_provenance import authenticate_pilot_source

    with pytest.raises(ValueError):
        authenticate_pilot_source(
            repo_root=tmp_path, source_commit="a" * 40, config=resolve_pilot_config("phase4_smoke")
        )


def test_missing_production_audits_fail_closed(tmp_path):
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_provenance import verify_pilot_data

    with pytest.raises(ValueError):
        verify_pilot_data(
            repo_root=Path.cwd(),
            source_commit="a" * 40,
            config=resolve_pilot_config("phase4_pilot"),
            manifests={"one_hop": tmp_path / "one_hop.json"},
        )


@pytest.mark.parametrize("boundary", ["source", "data"])
def test_forged_or_truncated_config_source_paths_are_rejected(tmp_path, boundary):
    root, execute = checkout(tmp_path / "repo")
    execute(
        root,
        DATA_SETUP
        + f"\nboundary = {boundary!r}\n"
        + """
from dataclasses import replace
for paths in ((), config.source_paths[:-1], (*config.source_paths[:-1], root/'uv.lock')):
    forged = replace(config, source_paths=paths)
    try:
        if boundary == 'source':
            authenticate_pilot_source(repo_root=root, source_commit=source, config=forged)
        else:
            verify_pilot_data(repo_root=root, source_commit=source,
                              config=forged, manifests=manifests)
    except ValueError as error:
        assert 'overlay' in str(error)
    else:
        raise AssertionError('forged/truncated overlay inventory accepted')
""",
    )
