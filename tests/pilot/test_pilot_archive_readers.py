"""Scientific closure across bounded, authenticated episode reads."""

from dataclasses import replace

import pytest

from silent_cascade.eval.runner import evaluate_episodes
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.report import pilot_artifacts

from .test_pilot_archive_producer import producer_case
from .test_timed_metrics import make_identity


@pytest.fixture
def evaluation_case(neural_archive_case, tmp_path):
    case = neural_archive_case
    second = replace(
        case.bundle,
        public=replace(
            case.bundle.public,
            init=replace(
                case.bundle.public.init, episode_public_id="00000000-0000-4000-8000-000000000002"
            ),
        ),
    )
    bundles = (case.bundle, second)
    return case, bundles, make_identity(case, bundles, tmp_path)


def evaluate(case, bundles, identity, output, *, producer=None, session=None):
    return evaluate_episodes(
        case.model,
        identity=identity,
        config=case.config,
        episodes=bundles,
        output_dir=output,
        device="cpu",
        archive_producer=producer,
        evidence_context=session,
    )


def test_scanner_requires_final_row_closure(evaluation_case, tmp_path):
    # Omitting the final closure check would accept a scientifically incomplete prefix.
    case, bundles, identity = evaluation_case
    root = tmp_path / "evaluation"
    evaluate(case, bundles, identity, root)
    assert hasattr(pilot_artifacts, "load_evaluation_header"), "missing common evaluation scanner"
    header = pilot_artifacts.load_evaluation_header(root)
    accumulator = pilot_artifacts.EvaluationScanAccumulator()
    rows = pilot_artifacts.iter_evaluation_rows(root, header)
    first = next(rows)
    verified = pilot_artifacts.verify_evaluation_episode(root, header=header, indexed_row=first)
    accumulator.add(verified)
    with pytest.raises(ValueError, match="incomplete"):
        pilot_artifacts.finish_evaluation_scan(header, accumulator)
    second = next(rows)
    accumulator.add(
        pilot_artifacts.verify_evaluation_episode(root, header=header, indexed_row=second)
    )
    with pytest.raises(StopIteration):
        next(rows)
    pilot_artifacts.finish_evaluation_scan(header, accumulator)
    assert accumulator.metrics().episode_count == 2


@pytest.mark.parametrize("failure", ["none", "runtime", "initialization", "mixed-initialization"])
def test_cold_evaluation_consumes_exactly_one_episode(
    evaluation_case, tmp_path, monkeypatch, failure
):
    # Falling back to resident Path reads fails: every episode and metadata owner is cold.
    case, bundles, identity = evaluation_case
    if failure != "none":
        from silent_cascade.eventflow.neural import NeuralEventFlowAgent
        from silent_cascade.models.errors import NeuralError
        from silent_cascade.models.event_flow import EventFlowModel

        def fail(*args):
            raise NeuralError("retained reader failure")

        if failure == "mixed-initialization":
            initialize = NeuralEventFlowAgent.initialize

            def initialize_once(agent, init):
                if init.episode_public_id == bundles[1].public.init.episode_public_id:
                    fail()
                return initialize(agent, init)

            monkeypatch.setattr(NeuralEventFlowAgent, "initialize", initialize_once)
        else:
            monkeypatch.setattr(
                NeuralEventFlowAgent if failure == "initialization" else EventFlowModel,
                "initialize" if failure == "initialization" else "compose",
                fail,
            )
    with producer_case(tmp_path, monkeypatch) as (producer, session, server):
        root = session.run_dir / "evaluation"
        result = evaluate(case, bundles, identity, root, producer=producer, session=session)
        (root / "execution.json").write_bytes(b"{}")
        producer.after_execution("evaluation")
        assert not (root / "DONE").exists()
        assert not (root / "episodes/00000.neural.json").exists()
        maximum = 0
        original = session._request

        def request(operation, payload, **kwargs):
            nonlocal maximum
            response = original(operation, payload, **kwargs)
            maximum = max(
                maximum,
                sum(value["ref"].kind == "episode_pack" for value in server.leases.values()),
            )
            return response

        monkeypatch.setattr(session, "_request", request)
        actual_identity, rows, metrics, hashes = pilot_artifacts.load_evaluation(
            root, evidence_context=session
        )
        assert actual_identity == identity
        assert [row.public_id for row in rows] == [item.public_id for item in identity.episodes]
        assert metrics == result.metrics
        assert hashes == dict(result.artifact_hashes)
        assert maximum == 1
        assert not server.leases
        assert not (root / "episodes/00001.neural.json").exists()


@pytest.mark.parametrize("field", ["run_id", "logical_root", "ordinal", "commit_sha256"])
def test_cold_commit_discovery_rejects_substituted_binding(
    evaluation_case, tmp_path, monkeypatch, field
):
    # The child must reject even a syntactically valid binding from the wrong corpus/run/row.
    case, bundles, identity = evaluation_case
    with producer_case(tmp_path, monkeypatch) as (producer, session, server):
        root = session.run_dir / "evaluation"
        evaluate(case, bundles, identity, root, producer=producer, session=session)
        original = session._request

        def request(operation, payload, **kwargs):
            response = original(operation, payload, **kwargs)
            if operation == "status" and "episode_commit" in payload:
                response["binding"][field] = (
                    1
                    if field == "ordinal"
                    else ("0" * 64 if field == "commit_sha256" else "foreign")
                )
            return response

        monkeypatch.setattr(session, "_request", request)
        with pytest.raises(ValueError):
            session.episode_commit("evaluation", 0)
        assert not server.leases


def test_discovery_does_not_authorize_substituted_lease(evaluation_case, tmp_path, monkeypatch):
    from dataclasses import asdict

    case, bundles, identity = evaluation_case
    with producer_case(tmp_path, monkeypatch) as (producer, session, server):
        root = session.run_dir / "evaluation"
        evaluate(case, bundles, identity, root, producer=producer, session=session)
        first = session.episode_commit("evaluation", 0)
        second = session.episode_commit("evaluation", 1)
        assert first.row_sha256 != second.row_sha256
        with (
            pytest.raises(ValueError, match="commit identity"),
            session.episode(
                "evaluation", 1, commit_sha256=sha256_bytes(canonical_json_bytes(asdict(first)))
            ),
        ):
            pytest.fail("substituted episode became readable")
        assert not server.leases


def test_cold_discovery_authenticates_the_binding_leaf(evaluation_case, tmp_path, monkeypatch):
    from silent_cascade.archive.transport import _object_key

    case, bundles, identity = evaluation_case
    with producer_case(tmp_path, monkeypatch) as (producer, session, server):
        root = session.run_dir / "evaluation"
        evaluate(case, bundles, identity, root, producer=producer, session=session)
        assert (
            session.episode_commit("evaluation", 0).episode_public_id
            == identity.episodes[0].public_id
        )
        head = pilot_artifacts.read_json(session.control_dir / "episode-bindings/head.json")
        remote = server.transport.root
        binding_root = pilot_artifacts.read_json(
            remote / _object_key("task4", f"episode-bindings/roots/{head['root_sha256']}.json")
        )
        leaf = "episode-bindings/" + binding_root["episodes"]["path"]
        assert not (session.control_dir / leaf).exists()
        (remote / _object_key("task4", leaf)).write_bytes(b"corrupt committed binding leaf")
        with pytest.raises(ValueError):
            session.episode_commit("evaluation", 0)
        assert not server.leases


def test_compact_rows_consume_cold_sidecars_inside_lease(evaluation_case, tmp_path, monkeypatch):
    from silent_cascade.train.pilot_evidence import compact_evaluation

    case, bundles, identity = evaluation_case
    with producer_case(tmp_path, monkeypatch) as (producer, session, server):
        root = session.run_dir / "evaluation"
        result = evaluate(case, bundles, identity, root, producer=producer, session=session)
        (root / "execution.json").write_bytes(b"{}")
        producer.after_execution("evaluation")
        actual_identity, records, hashes = compact_evaluation(root, evidence_context=session)
        records = tuple(records)
        assert actual_identity == identity
        assert hashes == dict(result.artifact_hashes)
        assert [record["row"]["public_id"] for record in records] == [
            item.public_id for item in identity.episodes
        ]
        assert all(record["causal_events"] for record in records)
        assert not server.leases


def test_cold_report_matches_eager_tables_and_figures(evaluation_case, tmp_path, monkeypatch):
    from silent_cascade.report.pilot import build_pilot_report

    case, bundles, identity = evaluation_case
    evaluate(case, bundles, identity, tmp_path / "local/eval/primary")
    eager = build_pilot_report(run_dir=tmp_path / "local", output_dir=tmp_path / "eager-report")
    with producer_case(tmp_path, monkeypatch) as (producer, session, server):
        root = session.run_dir / "eval/primary"
        evaluate(case, bundles, identity, root, producer=producer, session=session)
        (root / "execution.json").write_bytes(b"{}")
        producer.after_execution("eval/primary")
        cold = build_pilot_report(
            run_dir=session.run_dir, output_dir=tmp_path / "cold-report", evidence_context=session
        )
        for name in ("tables.json", "trajectories-0.svg", "report.md"):
            assert (cold.parent / name).read_bytes() == (eager.parent / name).read_bytes()
        assert not server.leases


def test_cold_journal_scans_all_segments_and_exact_update_coverage(tmp_path, monkeypatch):
    from silent_cascade.archive import readers
    from silent_cascade.train.pilot_state import PilotProgress
    from silent_cascade.train.pilot_trainer import verify_journal

    with producer_case(tmp_path, monkeypatch, journal_records=2) as (producer, session, server):
        producer.bind_training(source_commit="b" * 40, config_sha256="a" * 64)
        head = None
        for step in range(1, 7):
            raw = canonical_json_bytes(
                {
                    "prior": head,
                    "global_step": step,
                    "kind": "update",
                    "attempt": "attempt-" + "a" * 32,
                }
            )
            head = sha256_bytes(raw)
            path = session.run_dir / f"journal-{head}.json"
            path.write_bytes(raw)
            producer.after_journal(path.name, head)
        assert hasattr(readers, "iter_journal_records"), "missing authenticated journal iterator"
        records = list(
            readers.iter_journal_records(session.run_dir, head, evidence_context=session)
        )
        assert [record["global_step"] for record in records] == [6, 5, 4, 3, 2, 1]
        progress = PilotProgress(global_step=6, batch_counter=6, journal_sha256=head)
        assert len(verify_journal(session.run_dir, progress, evidence_context=session)) == 6
        with pytest.raises(Exception, match="missing or duplicate"):
            verify_journal(
                session.run_dir,
                progress.model_copy(update={"global_step": 7}),
                evidence_context=session,
            )
        assert not server.leases


@pytest.mark.parametrize("damage", ["last-index", "unclassified", "crash-extra"])
def test_completed_scanner_rejects_rehashed_invalid_closure(evaluation_case, tmp_path, damage):
    # Rehashing metadata cannot authorize unknown scientific members or a false final row range.
    case, bundles, identity = evaluation_case
    root = tmp_path / "evaluation"
    evaluate(case, bundles, identity, root)
    done = pilot_artifacts.read_json(root / "DONE")
    if damage == "last-index":
        index = pilot_artifacts.read_json(root / "index.json")
        index["rows"][-1]["offset"] += 1
        name, raw = "index.json", canonical_json_bytes(index)
    elif damage == "crash-extra":
        name, raw = "crashes/index.json", canonical_json_bytes({"foreign": {}})
    else:
        name, raw = "unclassified.json", b"{}"
    (root / name).parent.mkdir(exist_ok=True)
    (root / name).write_bytes(raw)
    done["artifact_hashes"][name] = sha256_bytes(raw)
    (root / "DONE").write_bytes(canonical_json_bytes(done))
    with pytest.raises(ValueError):
        pilot_artifacts.load_evaluation(root)
