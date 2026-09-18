"""Real initial checkpoint reads through scoped paths; no completed-run claims."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from silent_cascade.archive.readers import evidence_path
from silent_cascade.eventflow.neural import NeuralModelIdentity
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.train import pilot_checkpoints
from silent_cascade.train.pilot_config import resolve_pilot_config
from silent_cascade.train.pilot_provenance import PilotSourceIdentity
from silent_cascade.train.pilot_state import PilotCheckpointDescriptor, PilotProgress
from silent_cascade.train.pilot_trainer import retain_checkpoints
from silent_cascade.train.pilot_workflow import _durable
from silent_cascade.train.state import TrainingError
from silent_cascade.train.trainer import make_optimizer


class StrictPathContext:
    """Expose only the requested new fixture file, and revoke it on release."""

    def __init__(self, run_dir, files, lease_root):
        self.run_dir = run_dir
        self.files = files
        self.lease_root = lease_root
        self.active = 0
        self.maximum = 0
        self.requests = []

    @contextmanager
    def _leased(self, request):
        assert set(request) == {"path"}
        name = request["path"]
        assert name in self.files
        assert self.active == 0, "checkpoint reads must release prior leases"
        local = self.lease_root / name
        local.parent.mkdir(parents=True, exist_ok=True)
        original = self.files[name]
        original.rename(local)
        self.requests.append(name)
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        try:
            yield SimpleNamespace(local_root=self.lease_root)
        finally:
            local.rename(original)
            self.active -= 1


@pytest.fixture(scope="module")
def initial_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("initial-checkpoint")
    run = root / "run"
    run.mkdir()
    config = resolve_pilot_config("phase4_smoke")
    # Unit expected-source identity only; this is not authenticated run evidence.
    source = PilotSourceIdentity(
        source_commit="a" * 40,
        source_files={"unit-checkpoint-fixture": "b" * 64},
        source_sha256="c" * 64,
        plan_revision="d" * 40,
        plan_sha256="e" * 64,
        spec_sha256="f" * 64,
        config_sha256=config.sha256,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = EventFlowModel(config.config.neural)
        optimizer = make_optimizer(model, config.config.pilot)
        progress = PilotProgress()
        publish = pilot_checkpoints._publish_pilot_bytes

        def bounded_publish(path, raw):
            assert len(raw) <= 4 * 1024**2, "checkpoint exceeds admitted fixture payload"
            return publish(path, raw)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pilot_checkpoints, "_publish_pilot_bytes", bounded_publish)
            pending = run / "initial.safetensors"
            digest = pilot_checkpoints.save_pilot_checkpoint(
                pending,
                model=model,
                optimizer=optimizer,
                progress=progress,
                config=config,
                source=source,
            )
        identity = NeuralModelIdentity.from_model(model, source_revision=source.source_commit)
        descriptor = PilotCheckpointDescriptor(
            path=f"training-0-{digest}.safetensors",
            sha256=digest,
            model_state_sha256=identity.model_state_sha256,
            global_step=0,
            stage="one_hop",
        )
        checkpoint = run / descriptor.path
        pending.rename(checkpoint)
        retain_checkpoints(run, [descriptor], descriptor)
        eager = _durable(run, config, source, "cpu")
    assert eager == checkpoint
    backing = root / "backing"
    backing.mkdir()
    checkpoint.rename(backing / descriptor.path)
    index = run / "checkpoint-index.json"
    index.rename(backing / index.name)
    return SimpleNamespace(
        run=run,
        backing=backing,
        config=config,
        source=source,
        model=model,
        progress=progress,
        descriptor=descriptor,
        identity=identity,
        eager=eager,
    )


def cold_context(case, tmp_path, *, descriptor=None):
    if descriptor is None:
        raw = (case.backing / "checkpoint-index.json").read_bytes()
    else:
        descriptor = descriptor.model_dump(mode="json")
        raw = canonical_json_bytes({"latest": descriptor, "best": [], "history": [descriptor]})
    assert len(raw) <= 16 * 1024
    index = tmp_path / "checkpoint-index.json"
    index.write_bytes(raw)
    return StrictPathContext(
        case.run,
        {"checkpoint-index.json": index, case.descriptor.path: case.backing / case.descriptor.path},
        tmp_path / "leased",
    )


def test_cold_durable_returns_logical_path_and_restores_real_tensors(initial_checkpoint, tmp_path):
    case = initial_checkpoint
    context = cold_context(case, tmp_path)
    with torch.random.fork_rng(devices=[]):
        returned = _durable(case.run, case.config, case.source, "cpu", evidence_context=context)
        assert returned == case.eager
        assert not returned.exists(), "the return value must remain a logical run path"
        assert context.requests == ["checkpoint-index.json", case.descriptor.path]
        assert context.maximum == 1 and context.active == 0
        assert not any(path.is_file() for path in context.lease_root.rglob("*"))
        with evidence_path(case.run, case.descriptor.path, evidence_context=context) as local:
            restored = pilot_checkpoints.load_pilot_checkpoint(
                local,
                expected_sha256=case.descriptor.sha256,
                config=case.config,
                source=case.source,
                device="cpu",
            )
        assert not local.exists()
        assert restored.progress == case.progress
        assert not restored.optimizer.state
        assert (
            NeuralModelIdentity.from_model(
                restored.model, source_revision=case.source.source_commit
            )
            == case.identity
        )
        expected = case.model.state_dict()
        actual = restored.model.state_dict()
        assert actual.keys() == expected.keys()
        assert all(torch.equal(actual[name], value) for name, value in expected.items())
    assert context.active == 0


@pytest.mark.parametrize("damage", ["digest", "step", "model"])
def test_cold_durable_rejects_changed_descriptor_and_releases_leases(
    initial_checkpoint, tmp_path, damage
):
    case = initial_checkpoint
    change = {
        "digest": {"sha256": "0" * 64},
        "step": {"global_step": 1},
        "model": {"model_state_sha256": "0" * 64},
    }[damage]
    descriptor = case.descriptor.model_copy(update=change)
    context = cold_context(case, tmp_path, descriptor=descriptor)
    error = TrainingError if damage == "digest" else ValueError
    match = "digest|sha256|hash" if damage == "digest" else "descriptor mismatch"
    with torch.random.fork_rng(devices=[]), pytest.raises(error, match=match):
        _durable(case.run, case.config, case.source, "cpu", evidence_context=context)
    assert context.requests == ["checkpoint-index.json", case.descriptor.path]
    assert context.maximum == 1 and context.active == 0
    assert not any(path.is_file() for path in context.lease_root.rglob("*"))
