"""Scientific closure across bounded, authenticated episode reads."""

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from silent_cascade.eval.runner import evaluate_episodes
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.report import pilot_artifacts

from .test_pilot_archive_producer import compact_archive_policy, producer_case
from .test_timed_metrics import make_identity


def _canonical_partial_identity():
    fixture = Path(__file__).parents[1] / "fixtures/pilot/canonical_evaluation_identity.json"
    raw = canonical_json_bytes(json.loads(fixture.read_bytes()))
    assert len(raw) == 7111
    assert sha256_bytes(raw) == "26583e594139ab8be066ad7144f0f1e0fc3586d5fdc133838626bfdce985e647"
    return raw


class _StrictEvidenceContext:
    def __init__(self, run_dir, cold_root, payloads):
        from silent_cascade.archive.types import FileEntry

        self.run_dir = run_dir
        self.cold_root = cold_root
        self._entries = tuple(
            FileEntry(name, sha256_bytes(raw), len(raw)) for name, raw in sorted(payloads.items())
        )
        self.maximum_leases = 0
        self._active = 0

    def entries(self):
        yield from self._entries

    def evaluation_roots(self):
        roots = {
            Path(entry.path).parent.as_posix()
            for entry in self._entries
            if Path(entry.path).name
            in {"identity.json", "DONE", "rows.jsonl", "metrics.json", ".rows.pending.jsonl"}
        }
        return tuple(sorted(roots))

    @contextmanager
    def _leased(self, payload):
        assert set(payload) == {"path"}
        self._active += 1
        self.maximum_leases = max(self.maximum_leases, self._active)
        try:
            yield SimpleNamespace(local_root=self.cold_root)
        finally:
            self._active -= 1


def test_cold_partial_rejects_unbound_predecessor(tmp_path):
    from silent_cascade.archive.types import FileEntry
    from silent_cascade.report.pilot_artifacts import training_evaluation_status

    run = tmp_path / "run"
    run.mkdir()
    committed_raw = canonical_json_bytes(
        {
            "prior": None,
            "global_step": 1,
            "kind": "update",
            "attempt": "attempt-" + "1" * 32,
        }
    )
    committed = sha256_bytes(committed_raw)
    committed_name = f"journal-{committed}.json"
    abandoned_raw = canonical_json_bytes(
        {
            "prior": "f" * 64,
            "global_step": 2,
            "kind": "update",
            "attempt": "attempt-" + "2" * 32,
        }
    )
    abandoned = sha256_bytes(abandoned_raw)
    abandoned_name = f"journal-{abandoned}.json"
    restart_name = "restart-" + "3" * 32 + ".json"
    restart_raw = canonical_json_bytes(
        {"last_durable_step": 1, "uncommitted_journal_tail": [abandoned_name]}
    )
    payloads = {
        committed_name: committed_raw,
        abandoned_name: abandoned_raw,
        restart_name: restart_raw,
    }
    for name, raw in payloads.items():
        (run / name).write_bytes(raw)
    context = _StrictEvidenceContext(run, run, payloads)
    training = {
        "progress": {"global_step": 1, "journal_sha256": committed},
        "artifact_hashes": {name: sha256_bytes(raw) for name, raw in payloads.items()},
    }
    assert tuple(context.entries()) == tuple(
        FileEntry(name, sha256_bytes(raw), len(raw)) for name, raw in sorted(payloads.items())
    )
    with pytest.raises(ValueError, match="missing abandoned journal predecessor"):
        training_evaluation_status(run, training, evidence_context=context)


def test_cold_partial_refuses_newline_complete_row(tmp_path):
    from silent_cascade.report.pilot_artifacts import load_abandoned_evaluation

    run = tmp_path / "run"
    cold = tmp_path / "cold"
    logical_root = "attempt-" + "2" * 32 + "/validation-1-primary/autonomous"
    payloads = {
        f"{logical_root}/identity.json": _canonical_partial_identity(),
        f"{logical_root}/.rows.pending.jsonl": b"{}\n",
    }
    for name, raw in payloads.items():
        path = cold / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    context = _StrictEvidenceContext(run, cold, payloads)
    training = {"artifact_hashes": {name: sha256_bytes(raw) for name, raw in payloads.items()}}
    with pytest.raises(ValueError, match="cold complete abandoned rows"):
        load_abandoned_evaluation(
            run / logical_root,
            run_dir=run,
            training=training,
            abandoned={"attempt-" + "2" * 32},
            evidence_context=context,
        )
    assert context.maximum_leases == 1


def test_cold_root_training_result_is_discovered_from_authenticated_inventory(tmp_path):
    from silent_cascade.report.pilot import build_pilot_report
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_state import PilotCheckpointDescriptor, PilotProgress

    run, cold = tmp_path / "run", tmp_path / "cold"
    run.mkdir()
    attempt = "attempt-" + "2" * 32
    logical_root = f"{attempt}/validation-1-primary/autonomous"
    committed_raw = canonical_json_bytes(
        {
            "prior": None,
            "global_step": 1,
            "kind": "update",
            "attempt": "attempt-" + "1" * 32,
        }
    )
    committed = sha256_bytes(committed_raw)
    committed_name = f"journal-{committed}.json"
    abandoned_raw = canonical_json_bytes(
        {"prior": committed, "global_step": 2, "kind": "update", "attempt": attempt}
    )
    abandoned = sha256_bytes(abandoned_raw)
    abandoned_name = f"journal-{abandoned}.json"
    restart_name = "restart-" + "3" * 32 + ".json"
    restart_raw = canonical_json_bytes(
        {"last_durable_step": 1, "uncommitted_journal_tail": [abandoned_name]}
    )
    weights_name = f"{attempt}/latest-weights.safetensors"
    weights_raw = b"bounded latest weights"
    descriptor = PilotCheckpointDescriptor(
        path=weights_name,
        sha256=sha256_bytes(weights_raw),
        model_state_sha256="4" * 64,
        global_step=1,
        stage="one_hop",
    )
    progress = PilotProgress(
        global_step=1,
        batch_counter=1,
        latest=descriptor,
        journal_sha256=committed,
        status="step_ceiling",
    )
    artifacts = {
        committed_name: committed_raw,
        abandoned_name: abandoned_raw,
        restart_name: restart_raw,
        weights_name: weights_raw,
        f"{logical_root}/identity.json": _canonical_partial_identity(),
        f"{logical_root}/.rows.pending.jsonl": b'{"unfinished":1}',
    }
    training = {
        "status": progress.status,
        "progress": progress.model_dump(mode="json"),
        "selected_checkpoint": None,
        "selected_weights": None,
        "latest_weights": descriptor.model_dump(mode="json"),
        "artifact_hashes": {name: sha256_bytes(raw) for name, raw in artifacts.items()},
        "model_identity": {
            "model_config_json": canonical_json_bytes(
                resolve_pilot_config("phase4_smoke").config.neural
            ).decode(),
            "model_state_sha256": descriptor.model_state_sha256,
            "source_revision": "b" * 40,
        },
        "gate_eligible": False,
    }
    result_raw = canonical_json_bytes(training)
    payloads = artifacts | {"training-result.json": result_raw}
    for name, raw in artifacts.items():
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    result = cold / "training-result.json"
    result.parent.mkdir(parents=True)
    result.write_bytes(result_raw)
    context = _StrictEvidenceContext(run, cold, payloads)
    report = build_pilot_report(
        run_dir=run, output_dir=tmp_path / "report", evidence_context=context
    )
    tables = json.loads((report.parent / "tables.json").read_bytes())
    assert tables["evaluations"] == []
    assert tables["incomplete_evaluations"][0]["unknown_episodes"] == 1
    assert "Training status: step_ceiling; updates: 1" in report.read_text()
    assert context.maximum_leases == 1


def test_cold_zero_row_partial_report_uses_authenticated_inventory(tmp_path, monkeypatch):
    """Cold partial and journal evidence remains reportable with pinned controls."""
    from silent_cascade.report.pilot import build_pilot_report
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_state import PilotCheckpointDescriptor, PilotProgress
    from silent_cascade.train.pilot_workflow import pilot_ownership

    attempt = "attempt-" + "2" * 32
    logical_root = f"{attempt}/validation-1-primary/autonomous"
    with producer_case(
        tmp_path,
        monkeypatch,
        stopped=True,
        policy=compact_archive_policy(),
        isolate_authority=True,
    ) as (producer, session, server):
        producer.track_file(session.run_dir / "checkpoint-index.json")
        for path in session.run_dir.glob("training-*.safetensors"):
            producer.track_file(path)
        producer.bind_training(source_commit="b" * 40, config_sha256="c" * 64)
        committed_raw = canonical_json_bytes(
            {
                "prior": None,
                "global_step": 1,
                "kind": "update",
                "attempt": "attempt-" + "1" * 32,
            }
        )
        committed = sha256_bytes(committed_raw)
        committed_name = f"journal-{committed}.json"
        (session.run_dir / committed_name).write_bytes(committed_raw)
        producer.after_journal(committed_name, committed)
        abandoned_raw = canonical_json_bytes(
            {
                "prior": committed,
                "global_step": 2,
                "kind": "update",
                "attempt": attempt,
            }
        )
        abandoned = sha256_bytes(abandoned_raw)
        abandoned_name = f"journal-{abandoned}.json"
        (session.run_dir / abandoned_name).write_bytes(abandoned_raw)
        producer.after_journal(abandoned_name, abandoned)

        partial = session.run_dir / logical_root
        partial.mkdir(parents=True)
        identity_raw = _canonical_partial_identity()
        pending_raw = b'{"unfinished":1}'
        assert len(pending_raw) == 16
        orphan_raw = b"opaque orphan crash data"
        assert len(orphan_raw) == 24
        members = {
            f"{logical_root}/identity.json": identity_raw,
            f"{logical_root}/.rows.pending.jsonl": pending_raw,
            f"{logical_root}/orphan-crash.bin": orphan_raw,
        }
        for name, raw in members.items():
            (session.run_dir / name).write_bytes(raw)

        restart_name = "restart-" + "3" * 32 + ".json"
        restart_raw = canonical_json_bytes(
            {"last_durable_step": 1, "uncommitted_journal_tail": [abandoned_name]}
        )
        (session.run_dir / restart_name).write_bytes(restart_raw)
        weights_raw = b"bounded latest weights"
        weights_name = f"{attempt}/latest-weights.safetensors"
        (session.run_dir / weights_name).write_bytes(weights_raw)
        descriptor = PilotCheckpointDescriptor(
            path=weights_name,
            sha256=sha256_bytes(weights_raw),
            model_state_sha256="4" * 64,
            global_step=1,
            stage="one_hop",
        )
        progress = PilotProgress(
            global_step=1,
            batch_counter=1,
            latest=descriptor,
            journal_sha256=committed,
            status="step_ceiling",
        )
        artifact_hashes = {
            committed_name: committed,
            abandoned_name: abandoned,
            restart_name: sha256_bytes(restart_raw),
            weights_name: descriptor.sha256,
            **{name: sha256_bytes(raw) for name, raw in members.items()},
        }
        training = {
            "status": progress.status,
            "progress": progress.model_dump(mode="json"),
            "selected_checkpoint": None,
            "selected_weights": None,
            "latest_weights": descriptor.model_dump(mode="json"),
            "artifact_hashes": artifact_hashes,
            "model_identity": {
                "model_config_json": canonical_json_bytes(
                    resolve_pilot_config("phase4_smoke").config.neural
                ).decode(),
                "model_state_sha256": descriptor.model_state_sha256,
                "source_revision": "b" * 40,
            },
            "gate_eligible": False,
        }
        result_name = "training-result.json"
        result_raw = canonical_json_bytes(training)
        (session.run_dir / result_name).write_bytes(result_raw)

        with pilot_ownership(
            session.run_dir,
            run_identity="a" * 64,
            recover=lambda: None,
            capture_stopped=True,
        ) as custody:
            producer.adopt_stopped(
                custody,
                source_commit="b" * 40,
                config_sha256="c" * 64,
            )
        control_paths = (restart_name, result_name)
        for name in control_paths:
            producer.track_file(session.run_dir / name)
        ref = producer._seal(
            logical_root=".",
            paths=control_paths,
            kind="control_snapshot",
            identity=producer._training_identity.model_copy(
                update={
                    "checkpoint_sha256": producer.stopped_checkpoint.sha256,
                    "checkpoint_committed": True,
                }
            ).model_dump(mode="json"),
        )
        producer._archive(ref, evict=False)
        assert all(not (session.run_dir / name).exists() for name in members)
        assert all((session.run_dir / name).is_file() for name in control_paths)
        assert not list(session.run_dir.rglob("DONE"))
        assert not (session.control_dir / "episode-bindings/head.json").exists()

        maximum = 0
        original = session._request

        def request(operation, payload, **kwargs):
            nonlocal maximum
            response = original(operation, payload, **kwargs)
            maximum = max(maximum, len(server.leases))
            return response

        monkeypatch.setattr(session, "_request", request)
        assert session.evaluation_roots() == (logical_root,)
        monkeypatch.setattr(
            "silent_cascade.report.pilot._figures",
            lambda *args, **kwargs: pytest.fail("partial-only report invoked matplotlib"),
        )
        report = build_pilot_report(
            run_dir=session.run_dir,
            output_dir=tmp_path / "report",
            evidence_context=session,
        )
        tables = json.loads((report.parent / "tables.json").read_bytes())
        assert tables["evaluations"] == []
        assert tables["incomplete_evaluations"] == [
            {
                "artifact_hashes": {
                    name.removeprefix(logical_root + "/"): artifact_hashes[name] for name in members
                },
                "corpus": logical_root,
                "identity_sha256": (
                    "26583e594139ab8be066ad7144f0f1e0fc3586d5fdc133838626bfdce985e647"
                ),
                "planned_episodes": 1,
                "raw_rows": f"{logical_root}/.rows.pending.jsonl",
                "retained_errors": 0,
                "retained_rows": 0,
                "status": "abandoned_incomplete",
                "unknown_episodes": 1,
                "unparsed_trailing_bytes": 16,
            }
        ]
        assert "abandoned incomplete" in report.read_text()
        assert "unknown episodes: 1" in report.read_text()
        assert maximum == 1
        assert not server.leases
        assert not (session.control_dir / "episode-bindings/head.json").exists()


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

    with producer_case(
        tmp_path,
        monkeypatch,
        journal_records=2,
        policy=compact_archive_policy(journal_records=2),
        isolate_authority=True,
    ) as (producer, session, server):
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
