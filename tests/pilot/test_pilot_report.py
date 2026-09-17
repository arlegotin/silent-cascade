"""Reports regenerate metrics from actual retained evaluation evidence."""

import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def pilot_artifacts(neural_archive_case, tmp_path):
    from silent_cascade.eval.runner import evaluate_episodes

    from .test_timed_metrics import make_identity

    case = neural_archive_case
    identity = make_identity(
        case, (case.bundle,), tmp_path, experiment="task9-fixture-only-non-acceptance"
    )
    run = tmp_path / "run"
    evaluate_episodes(
        case.model,
        identity=identity,
        config=case.config,
        episodes=[case.bundle],
        output_dir=run / "eval/primary",
        device="cpu",
    )
    return SimpleNamespace(run_dir=run, report_dir=tmp_path / "report")


def test_report_does_not_import_or_call_training(pilot_artifacts, monkeypatch):
    import silent_cascade.train.pilot_trainer as trainer
    from silent_cascade.report.pilot import build_pilot_report

    def forbidden(*args, **kwargs):
        raise AssertionError("reporting must not train")

    monkeypatch.setattr(trainer, "run_pilot_training", forbidden)
    report = build_pilot_report(
        run_dir=pilot_artifacts.run_dir, output_dir=pilot_artifacts.report_dir
    )
    text = report.read_text()
    assert "Autonomous timed pilot" in text
    assert "production learning gate not established" in text
    assert "fixture-only" in text
    table = json.loads((report.parent / "tables.json").read_bytes())
    assert table["evaluations"][0]["episode_count"] == 1
    assert table["evaluations"][0]["error_count"] == 0
    assert (
        build_pilot_report(
            run_dir=pilot_artifacts.run_dir, output_dir=pilot_artifacts.report_dir
        ).read_bytes()
        == report.read_bytes()
    )


@pytest.mark.parametrize("corruption", ["missing", "hash", "unsafe", "symlink", "metrics"])
def test_report_rejects_corrupt_or_missing_artifacts(pilot_artifacts, corruption):
    from silent_cascade.report.pilot import build_pilot_report

    root = pilot_artifacts.run_dir / "eval/primary"
    if corruption == "missing":
        (root / "DONE").unlink()
    elif corruption == "symlink":
        original = root / "rows.jsonl"
        original.rename(root / "rows-copy.jsonl")
        original.symlink_to("rows-copy.jsonl")
    elif corruption == "metrics":
        (root / "metrics.json").write_text("{}")
    else:
        done = json.loads((root / "DONE").read_bytes())
        if corruption == "hash":
            done["artifact_hashes"]["rows.jsonl"] = "0" * 64
        else:
            done["artifact_hashes"]["../outside"] = "0" * 64
        (root / "DONE").write_text(json.dumps(done))
    with pytest.raises((ValueError, OSError)):
        build_pilot_report(run_dir=pilot_artifacts.run_dir, output_dir=pilot_artifacts.report_dir)
    assert not (pilot_artifacts.report_dir / "report.md").exists()


def test_report_keeps_runtime_failures_in_denominator(neural_archive_case, tmp_path, monkeypatch):
    from dataclasses import replace

    from silent_cascade.eval.runner import evaluate_episodes
    from silent_cascade.models.errors import NeuralError
    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.report.pilot import build_pilot_report

    from .test_timed_metrics import make_identity

    case = neural_archive_case
    other = replace(
        case.bundle,
        public=replace(
            case.bundle.public,
            init=replace(
                case.bundle.public.init, episode_public_id="00000000-0000-4000-8000-000000000002"
            ),
        ),
    )
    identity = make_identity(
        case, (case.bundle, other), tmp_path, experiment="task9-fixture-only-injected-failure"
    )
    original = EventFlowModel.compose
    calls = 0

    def fail_once(self, context):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise NeuralError("injected invalid dynamics")
        return original(self, context)

    monkeypatch.setattr(EventFlowModel, "compose", fail_once)
    run = tmp_path / "run"
    evaluate_episodes(
        case.model,
        identity=identity,
        config=case.config,
        episodes=[case.bundle, other],
        output_dir=run / "eval/primary",
        device="cpu",
    )
    report = build_pilot_report(run_dir=run, output_dir=tmp_path / "report")
    table = json.loads((report.parent / "tables.json").read_bytes())["evaluations"][0]
    assert table["episode_count"] == 2 and table["error_count"] == 1
    assert table["variants"]["positive"]["episodes"] == 2
    assert "| Errors | 1 / 2 |" in report.read_text()


def test_report_does_not_omit_a_corpus_missing_its_identity(pilot_artifacts):
    import shutil

    from silent_cascade.report.pilot import build_pilot_report

    root = pilot_artifacts.run_dir
    shutil.copytree(root / "eval/primary", root / "eval/secondary")
    (root / "eval/secondary/identity.json").unlink()
    with pytest.raises((ValueError, OSError)):
        build_pilot_report(run_dir=root, output_dir=pilot_artifacts.report_dir)


def test_artifact_hash_verification_retains_only_requested_bytes(tmp_path):
    from silent_cascade.hashing import sha256_bytes
    from silent_cascade.report.pilot_artifacts import verify_hashes

    summary = b'{"count":1}'
    archive = b"binary artifact" * 1024
    (tmp_path / "summary.json").write_bytes(summary)
    (tmp_path / "weights.bin").write_bytes(archive)
    hashes = {"summary.json": sha256_bytes(summary), "weights.bin": sha256_bytes(archive)}
    assert verify_hashes(tmp_path, hashes, retain={"summary.json"}) == {"summary.json": summary}
    (tmp_path / "weights.bin").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="integrity"):
        verify_hashes(tmp_path, hashes, retain={"summary.json"})


@pytest.fixture(scope="module")
def recovered_validation_run(tmp_path_factory):
    from .test_pilot_source import DATA_SETUP, checkout
    from .test_pilot_workflow import INTERRUPTED_VALIDATION_SETUP

    root, execute = checkout(tmp_path_factory.mktemp("recovered-validation") / "repo")
    execute(root, DATA_SETUP + INTERRUPTED_VALIDATION_SETUP)
    return root / "runs/interrupted-validation"


def test_report_separates_authenticated_abandoned_partial_evidence(
    recovered_validation_run, tmp_path
):
    from silent_cascade.report.pilot import build_pilot_report

    report = build_pilot_report(run_dir=recovered_validation_run, output_dir=tmp_path / "report")
    tables = json.loads((report.parent / "tables.json").read_bytes())
    assert len(tables["evaluations"]) == 2
    assert all(table["episode_count"] == 16 for table in tables["evaluations"])
    incomplete = tables["incomplete_evaluations"]
    assert len(incomplete) == 1
    assert incomplete[0]["planned_episodes"] == 16
    assert incomplete[0]["retained_rows"] == 1
    assert incomplete[0]["unknown_episodes"] == 15
    assert incomplete[0]["retained_errors"] == 0
    assert "abandoned incomplete" in report.read_text()
    assert "unknown" in report.read_text()


@pytest.mark.parametrize("corruption", ["restart", "journal", "committed-done"])
def test_abandoned_classification_never_excuses_corrupt_committed_evidence(
    recovered_validation_run, tmp_path, corruption
):
    import shutil

    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
    from silent_cascade.report.pilot import build_pilot_report

    run = tmp_path / "run"
    shutil.copytree(recovered_validation_run, run)
    from silent_cascade.report.pilot_artifacts import load_training_result

    result = load_training_result(run, run / "training-result.json")
    hashes = result["artifact_hashes"]
    restart = next(run.glob("restart-*.json"))
    payload = json.loads(restart.read_bytes())
    if corruption == "restart":
        payload["uncommitted_journal_tail"] = []
        restart.write_bytes(canonical_json_bytes(payload))
        hashes[restart.name] = sha256_bytes(restart.read_bytes())
    elif corruption == "journal":
        journal = run / payload["uncommitted_journal_tail"][0]
        entry = json.loads(journal.read_bytes())
        entry["attempt"] = "attempt-" + "0" * 32
        journal.write_bytes(canonical_json_bytes(entry))
        hashes[journal.name] = sha256_bytes(journal.read_bytes())
    else:
        done = next(run.glob("attempt-*/validation-*/autonomous/DONE"))
        hashes.pop(str(done.relative_to(run)))
        done.unlink()
    # Rebind the outer inventory deliberately: semantic journal checks, not just
    # the already-covered raw-file digest mismatch, must still reject corruption.
    (run / "training-result.json").write_bytes(canonical_json_bytes(result))
    with pytest.raises((ValueError, OSError)):
        build_pilot_report(run_dir=run, output_dir=tmp_path / "report")
    assert not (tmp_path / "report/report.md").exists()


@pytest.mark.parametrize("missing", ["bundle", "checkpoint"])
def test_abandoned_error_row_requires_its_actual_crash_bundle(tmp_path, missing):
    from silent_cascade.hashing import canonical_json_bytes
    from silent_cascade.report.pilot import build_pilot_report

    from .test_pilot_source import DATA_SETUP, checkout
    from .test_pilot_workflow import INTERRUPTED_VALIDATION_SETUP

    root, execute = checkout(tmp_path / "repo")
    execute(root, DATA_SETUP + "\nfail_first_row = True\n" + INTERRUPTED_VALIDATION_SETUP)
    run = root / "runs/interrupted-validation"
    report = build_pilot_report(run_dir=run, output_dir=tmp_path / "report")
    table = json.loads((report.parent / "tables.json").read_bytes())["incomplete_evaluations"][0]
    assert table["retained_rows"] == 1 and table["retained_errors"] == 1
    assert table["unknown_episodes"] == 15
    partial = run / table["corpus"]
    crash = next((partial / "crashes").glob("*.json"))
    assert crash.name != "index.json", "partial failure should precede crash-index publication"
    if missing == "checkpoint":
        reference = json.loads(crash.read_bytes())["context"]["checkpoint_ref"]
        assert reference is not None
        crash = partial / "crashes" / reference
    from silent_cascade.report.pilot_artifacts import load_training_result

    result = load_training_result(run, run / "training-result.json")
    result["artifact_hashes"].pop(str(crash.relative_to(run)))
    crash.unlink()
    (run / "training-result.json").write_bytes(canonical_json_bytes(result))
    with pytest.raises(ValueError, match="crash evidence"):
        build_pilot_report(run_dir=run, output_dir=tmp_path / "missing-crash-report")
    assert not (tmp_path / "missing-crash-report/report.md").exists()


def test_abandoned_attempt_cannot_excuse_its_previously_committed_validation(tmp_path):
    from silent_cascade.hashing import canonical_json_bytes
    from silent_cascade.report.pilot import build_pilot_report

    from .test_pilot_source import DATA_SETUP, checkout
    from .test_pilot_workflow import INTERRUPTED_VALIDATION_SETUP

    root, execute = checkout(tmp_path / "repo")
    execute(
        root, DATA_SETUP + "\nvalidation_interruption_call = 18\n" + INTERRUPTED_VALIDATION_SETUP
    )
    run = root / "runs/interrupted-validation"
    report = build_pilot_report(run_dir=run, output_dir=tmp_path / "report")
    tables = json.loads((report.parent / "tables.json").read_bytes())
    assert len(tables["evaluations"]) == 2
    incomplete = tables["incomplete_evaluations"]
    assert len(incomplete) == 1 and incomplete[0]["unknown_episodes"] == 15
    assert incomplete[0]["corpus"].endswith("validation-4-one_hop/autonomous")
    partial = run / incomplete[0]["corpus"]
    committed = partial.parents[1] / "validation-2-one_hop/autonomous/DONE"
    assert committed.is_file(), "same abandoned attempt must retain its durable validation2"
    from silent_cascade.report.pilot_artifacts import load_training_result

    result = load_training_result(run, run / "training-result.json")
    result["artifact_hashes"].pop(str(committed.relative_to(run)))
    committed.unlink()
    (run / "training-result.json").write_bytes(canonical_json_bytes(result))
    with pytest.raises((ValueError, OSError)):
        build_pilot_report(run_dir=run, output_dir=tmp_path / "missing-committed-report")
    assert not (tmp_path / "missing-committed-report/report.md").exists()
