"""Wiring-only final-control admission tests; no scientific execution."""

from pathlib import Path
from types import SimpleNamespace

import pytest

FINAL_CONTROL_BYTES = 134479872


def test_final_control_bound_is_fixed():
    from silent_cascade.archive.producer import before_work_bounds

    assert before_work_bounds("final_evaluation_control") == {"spool": FINAL_CONTROL_BYTES}


@pytest.mark.parametrize(
    "remaining,retained,permitted",
    [
        (FINAL_CONTROL_BYTES, 0, True),
        (FINAL_CONTROL_BYTES - 1, 0, False),
        (FINAL_CONTROL_BYTES, 1, False),
    ],
)
def test_final_control_dispatch_uses_exact_remaining_capacity(remaining, retained, permitted):
    from silent_cascade.archive.ledger import StorageBlocked
    from silent_cascade.archive.supervisor import _ArchiveServer

    server = object.__new__(_ArchiveServer)
    active = {
        "before": {"spool": 0},
        "amounts": {"spool": remaining},
        "admission": {"source_bound_fixture": True},
    }
    server.budget = SimpleNamespace(
        active_reservation="fixture",
        _state=lambda: {"reservations": {"fixture": active}},
    )
    server._check_budget = lambda: {"spool": retained}
    request = {"before_work": "final_evaluation_control"}
    snapshot = dict(active, before=dict(active["before"]), amounts=dict(active["amounts"]))
    if permitted:
        assert server._dispatch("status", request) == {"admitted": True}
        assert server._dispatch("status", request) == {"admitted": True}
        assert active == snapshot
    else:
        with pytest.raises(StorageBlocked, match="exceeds remaining admission"):
            server._dispatch("status", request)
        assert active == snapshot


@pytest.mark.parametrize("admitted", [True, False])
def test_final_control_producer_sends_only_closed_operation(admitted):
    from silent_cascade.archive.ledger import StorageBlocked
    from silent_cascade.archive.producer import ArchiveProducer

    calls = []
    producer = object.__new__(ArchiveProducer)
    producer.session = SimpleNamespace(
        _request=lambda operation, payload: (
            calls.append((operation, payload)) or {"admitted": admitted}
        )
    )
    if admitted:
        assert producer.before_final_evaluation_control() is None
    else:
        with pytest.raises(StorageBlocked, match="admission not authenticated"):
            producer.before_final_evaluation_control()
    assert calls == [("status", {"before_work": "final_evaluation_control"})]


class _ControlModel:
    def __init__(self):
        self.parameter = SimpleNamespace(device=SimpleNamespace(type="cpu"))

    def parameters(self):
        return iter((self.parameter,))

    def buffers(self):
        return iter(())


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("success", ["admit", "request", "evaluate", "admit", "execution", "after"]),
        ("deny_first", ["admit"]),
        ("deny_second", ["admit", "request", "evaluate", "admit"]),
        ("no_archive", ["request", "evaluate", "execution"]),
        ("completed", ["admit", "request"]),
    ],
)
def test_final_control_admission_orders_real_evaluate_branching(
    tmp_path, monkeypatch, mode, expected
):
    from silent_cascade.archive.ledger import StorageBlocked
    from silent_cascade.train import pilot_checks

    events = []
    identities = []
    output = tmp_path / "run/final/eval/primary"
    original_exists = Path.exists

    class Identity:
        sha256 = "identity"

        def __init__(self, **kwargs):
            identities.append(self)

    class Producer:
        def __init__(self):
            self.admissions = 0

        def before_final_evaluation_control(self):
            self.admissions += 1
            events.append("admit")
            if mode == "deny_first" or (mode == "deny_second" and self.admissions == 2):
                raise StorageBlocked("storage_blocked: fixture denial")

        def logical_root(self, directory):
            assert directory == output
            return "final/eval/primary"

        def after_execution(self, logical_root):
            assert logical_root == "final/eval/primary"
            events.append("after")

    def exists(path):
        if path == output / "DONE":
            return mode == "completed"
        if path == output:
            return False
        return original_exists(path)

    def publish(path, value):
        if path.name == ".primary.request.json":
            events.append("request")
        elif path.name == "execution.json":
            events.append("execution")
        else:
            raise AssertionError("unexpected control publication")

    def evaluate(*args, **kwargs):
        assert kwargs["output_dir"] == output
        events.append("evaluate")

    monkeypatch.setattr(Path, "exists", exists)
    monkeypatch.setattr(pilot_checks, "canonical_json_bytes", lambda _: b"canonical")
    monkeypatch.setattr(pilot_checks, "sha256_bytes", lambda _: "hash")
    monkeypatch.setattr(pilot_checks, "expected_bundles", lambda *args: ())
    monkeypatch.setattr(pilot_checks, "EvaluationIdentity", Identity)
    monkeypatch.setattr(pilot_checks, "_publish", publish)
    monkeypatch.setattr(pilot_checks, "evaluate_episodes", evaluate)
    monkeypatch.setattr(pilot_checks, "read_bytes", lambda _: b"done")
    monkeypatch.setattr(
        pilot_checks,
        "load_evaluation",
        lambda _: (identities[-1], None, None, None),
    )
    monkeypatch.setattr(
        pilot_checks,
        "strict_json",
        lambda _: {"device": "cpu", "purpose": "primary"},
    )

    manifest = SimpleNamespace(entries=(), split="debug", schema_version="fixture-v1")
    config = SimpleNamespace(
        canonical_json=b"{}",
        config=SimpleNamespace(pilot=SimpleNamespace(is_production=False)),
    )
    descriptor = SimpleNamespace(sha256="weights")
    result = SimpleNamespace(selected_weights=descriptor, latest_weights=None)
    weights = SimpleNamespace(
        model=_ControlModel(),
        identity=SimpleNamespace(model_state_sha256="model-state"),
    )
    producer = None if mode == "no_archive" else Producer()

    def call():
        return pilot_checks._evaluate(
            run_dir=tmp_path / "run",
            config=config,
            source=SimpleNamespace(source_commit="a" * 40),
            result=result,
            manifests={"primary": manifest},
            weights=weights,
            name="primary",
            archive_producer=producer,
        )

    if mode.startswith("deny_"):
        with pytest.raises(StorageBlocked, match="fixture denial"):
            call()
    else:
        assert call() is None
    assert events == expected
    assert not original_exists(output)
