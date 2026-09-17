"""Numerical evidence must expose omissions and preserve real continuation."""

import importlib
import importlib.util

import pytest
import torch


@pytest.fixture
def verification():
    name = "silent_cascade.train.pilot_verification"
    assert importlib.util.find_spec(name) is not None, "pilot numerical verification is missing"
    return importlib.import_module(name)


@pytest.fixture
def numeric_case(verification):
    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.trainer import make_optimizer

    config = resolve_pilot_config("phase4_smoke")
    torch.manual_seed(11)
    model = EventFlowModel(config.config.neural)
    batch = verification.diagnostic_batch(config.config, stage="primary", counter=0)
    result = verification.capture_update(
        model, make_optimizer(model, config.config.pilot), batch, config.config
    )
    return config, model, batch, result


def test_complete_inventory_comes_from_real_model_and_declared_outputs(verification, numeric_case):
    _, model, batch, result = numeric_case
    expected = verification.expected_inventory(model)
    assert len(expected["gradients"]) == len(tuple(model.named_parameters())) > 30
    assert len(expected["optimizer"]) == 3 * len(tuple(model.named_parameters()))
    assert batch.targets.kind.shape[1] == 11
    report = verification.compare_update(result, result, expected=expected, resume_device="cpu")
    assert report.passed
    assert report.gradients.rtol == report.gradients.atol == 0.0
    report.validate_complete_inventory(model)


@pytest.mark.parametrize("group", ["forward", "losses", "gradients", "parameters", "optimizer"])
def test_same_omission_on_both_sides_fails_inventory(verification, numeric_case, group):
    _, model, _, result = numeric_case
    damaged = {name: dict(values) for name, values in result.tensors.items()}
    damaged[group].pop(next(iter(damaged[group])))
    from dataclasses import replace

    with pytest.raises(ValueError, match="inventory"):
        bad = replace(result, tensors=damaged)
        verification.compare_update(bad, bad, expected=verification.expected_inventory(model))


def test_changed_comparison_tolerance_is_rejected(verification, numeric_case):
    _, model, _, result = numeric_case
    report = verification.compare_update(
        result, result, expected=verification.expected_inventory(model)
    )
    damaged = report.model_copy(
        update={"gradients": report.gradients.model_copy(update={"atol": 0.1})}
    )
    with pytest.raises(ValueError, match="tolerance"):
        damaged.validate_complete_inventory(model)


def test_forged_gradient_shape_cannot_pass_complete_inventory(verification, numeric_case):
    _, model, _, result = numeric_case
    report = verification.compare_update(
        result, result, expected=verification.expected_inventory(model)
    )
    shapes = dict(report.gradients.tensor_shapes)
    name = next(n for n, shape in shapes.items() if len(shape) == 2)
    shapes[name] = (1, 1)
    changed = report.model_copy(
        update={"gradients": report.gradients.model_copy(update={"tensor_shapes": shapes})}
    )
    with pytest.raises(ValueError, match="inventory"):
        changed.validate_complete_inventory(model)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="native MPS missing; not parity evidence"
)
def test_real_native_mps_recipe_and_resume(verification, resume_case, tmp_path):
    from silent_cascade.eventflow.checkpoint_rng import restore_rng_snapshot
    from silent_cascade.rng import seed_all, snapshot_global_rng

    config, model, optimizer, progress, source = resume_case
    seed_all(11)
    before = snapshot_global_rng()
    batch = verification.diagnostic_batch(config.config, stage="primary", counter=0)
    cpu_model, cpu_optimizer = verification.clone_training_state(model, optimizer, "cpu")
    cpu = verification.capture_update(cpu_model, cpu_optimizer, batch, config.config)
    restore_rng_snapshot(before, restore_mps=True)
    mps_model, mps_optimizer = verification.clone_training_state(model, optimizer, "mps")
    mps = verification.capture_update(mps_model, mps_optimizer, batch.to("mps"), config.config)
    assert verification.compare_update(
        cpu, mps, expected=verification.expected_inventory(model)
    ).passed
    assert verification.measure_resume(
        config,
        model=model,
        optimizer=optimizer,
        progress=progress,
        source=source,
        device="mps",
        output_dir=tmp_path / "native-resume",
    ).passed


def test_runtime_decision_difference_fails_despite_equal_scores(verification):
    cpu = [
        {
            "public_id": "a",
            "score": {"timed_success": False},
            "error": None,
            "events": [
                {
                    "kind": "recall",
                    "timestamp": 2.0,
                    "selected_record": 3,
                    "predicted_class": None,
                    "actions": [],
                }
            ],
        }
    ]
    mps = [{**cpu[0], "events": [{**cpu[0]["events"][0], "selected_record": 4}]}]
    result = verification.compare_runtime(cpu, mps)
    assert not result.passed
    assert result.decision_mismatches == 1
    assert result.score_mismatches == 0


def test_primary_diagnostic_batch_is_max_trace_and_debug_addressed(verification):
    from silent_cascade.train.pilot_config import resolve_pilot_config

    config = resolve_pilot_config("phase4_pilot")
    batch = verification.diagnostic_batch(config.config, stage="primary", counter=0)
    assert len(batch.examples) == 128
    assert batch.targets.kind.shape == (128, 11)
    assert {e.key.split for e in batch.examples} == {"debug"}
    from silent_cascade.train.curriculum_data import curriculum_allocation

    assert {curriculum_allocation(e.key)[1] for e in batch.examples} == {4}
    assert len(set(batch.example_hashes)) == 128


@pytest.fixture
def resume_case(numeric_case):
    from silent_cascade.train.pilot_provenance import PilotSourceIdentity
    from silent_cascade.train.pilot_state import PilotProgress
    from silent_cascade.train.trainer import make_optimizer

    config, model, _, _ = numeric_case
    source = PilotSourceIdentity(
        source_commit="a" * 40,
        source_files={"fixture": "b" * 64},
        source_sha256="c" * 64,
        plan_revision="d" * 40,
        plan_sha256="e" * 64,
        spec_sha256="f" * 64,
        config_sha256=config.sha256,
    )
    return config, model, make_optimizer(model, config.config.pilot), PilotProgress(), source


def test_exact_cpu_resume_checks_rng_batch_and_all_optimizer_moments(
    verification, resume_case, tmp_path
):
    config, model, optimizer, progress, source = resume_case
    result = verification.measure_resume(
        config,
        model=model,
        optimizer=optimizer,
        progress=progress,
        source=source,
        device="cpu",
        output_dir=tmp_path / "resume",
    )
    assert result.passed
    assert result.rng_before_sha256 == result.restored_rng_sha256
    assert result.next_batch_hashes == result.restored_batch_hashes
    assert len(result.comparison.optimizer.tested_names) > 100
    assert result.next_batch_counter == 2
    assert result.diagnostic_updates == 3


def test_portable_weights_are_not_a_resumable_training_checkpoint(
    verification, resume_case, tmp_path
):
    from silent_cascade.eventflow.neural import NeuralModelIdentity
    from silent_cascade.eventflow.neural_weights import save_neural_weights

    config, model, _, _, source = resume_case
    path = tmp_path / "portable.safetensors"
    save_neural_weights(
        path,
        model=model,
        identity=NeuralModelIdentity.from_model(model, source_revision=source.source_commit),
    )
    with pytest.raises(ValueError, match=r"full.*training archive"):
        verification.load_numeric_checkpoint(path, config=config, source=source, device="cpu")


@pytest.mark.parametrize("changed", ["source", "config"])
def test_full_archive_rejects_changed_identity(verification, resume_case, tmp_path, changed):
    from silent_cascade.train.pilot_checkpoints import save_pilot_checkpoint
    from silent_cascade.train.state import TrainingError

    config, model, optimizer, progress, source = resume_case
    path = tmp_path / "full.safetensors"
    save_pilot_checkpoint(
        path, model=model, optimizer=optimizer, progress=progress, config=config, source=source
    )
    source = source.model_copy(
        update={
            "source_commit" if changed == "source" else "config_sha256": "0"
            * (40 if changed == "source" else 64)
        }
    )
    with pytest.raises((ValueError, TrainingError), match="source/config"):
        verification.load_numeric_checkpoint(path, config=config, source=source, device="cpu")


def test_unavailable_mps_is_explicit_missing_evidence(verification, monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    devices, missing = verification.native_devices()
    assert devices == ("cpu",)
    assert missing == ("mps",)


def test_public_numeric_report_is_diagnostic_and_authenticates_artifacts(
    verification, resume_case, tmp_path, monkeypatch
):
    config, _, _, _, source = resume_case
    monkeypatch.setattr(
        verification, "authenticate_numeric_source", lambda config, checkpoint: source
    )
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    result = verification.verify_pilot_numerics(
        config, checkpoint=None, output_dir=tmp_path / "numeric"
    )
    assert result.evidence_kind == "fresh_numerical_diagnostic"
    assert result.missing_devices == ("mps",)
    assert result.cpu_resume.passed
    assert not result.device_checks_passed
    assert not result.selection_verified
    assert result.initial_model_sha256 == result.runtime_model_sha256
    assert result.cpu_inventory.keys() == {"one_hop", "primary"}
    assert result.runtime_episodes == 16
    restored = verification.read_numeric_report(
        tmp_path / "numeric", config=config, source_commit=source.source_commit
    )
    assert restored == result
    with pytest.raises(ValueError, match="source"):
        verification.read_numeric_report(
            tmp_path / "numeric", config=config, source_commit="f" * 40
        )
    import json

    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

    root = tmp_path / "numeric"
    originals = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    for mutation in (
        "omitted_capture",
        "changed_summary",
        "extra_file",
        "resume_summary",
        "raw_rng",
        "count",
        "dtype",
    ):
        payload = json.loads(
            originals[next(p for p in originals if str(p) == "numeric-report.json")]
        )
        if mutation == "omitted_capture":
            name = "primary-cpu.safetensors"
            (root / name).unlink()
            payload["artifact_hashes"].pop(name)
        elif mutation == "changed_summary":
            payload["cpu_inventory"]["primary"]["gradients"]["max_absolute_error"] = 0.125
        elif mutation == "resume_summary":
            payload["cpu_resume"]["rng_after_sha256"] = "1" * 64
            payload["cpu_resume"]["restored_rng_after_sha256"] = "1" * 64
        elif mutation == "raw_rng":
            name = "resume-cpu/observations.json"
            value = json.loads((root / name).read_bytes())
            value["draws"]["python"] = 0.123
            (root / name).write_bytes(canonical_json_bytes(value))
            payload["artifact_hashes"][name] = sha256_bytes((root / name).read_bytes())
        elif mutation == "count":
            payload["diagnostic_updates"] += 1
        elif mutation == "dtype":
            from safetensors.torch import load, save

            name = "primary-cpu.safetensors"
            values = load((root / name).read_bytes())
            key = next(k for k in values if k.startswith("gradients/"))
            values[key] = values[key].to(torch.int64)
            (root / name).write_bytes(save(values))
            payload["artifact_hashes"][name] = sha256_bytes((root / name).read_bytes())
        else:
            (root / "unrelated.json").write_bytes(b"{}")
            payload["artifact_hashes"]["unrelated.json"] = sha256_bytes(b"{}")
        raw = canonical_json_bytes(payload)
        (root / "numeric-report.json").write_bytes(raw)
        (root / "DONE").write_bytes(canonical_json_bytes({"report_sha256": sha256_bytes(raw)}))
        with pytest.raises(ValueError, match=r"artifact|raw|summary|closure"):
            verification.read_numeric_report(
                root, config=config, source_commit=source.source_commit
            )
        for path, content in originals.items():
            (root / path).write_bytes(content)
        if (root / "unrelated.json").exists():
            (root / "unrelated.json").unlink()
    artifact = tmp_path / "numeric" / next(iter(result.artifact_hashes))
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact"):
        verification.read_numeric_report(
            tmp_path / "numeric", config=config, source_commit=source.source_commit
        )


@pytest.mark.parametrize("binding", ["resume_device", "subset_count"])
def test_strict_numeric_report_binds_device_and_runtime_count(
    verification, resume_case, tmp_path, monkeypatch, binding
):
    import json

    from silent_cascade.hashing import canonical_json_bytes

    config, _, _, _, source = resume_case
    monkeypatch.setattr(verification, "authenticate_numeric_source", lambda *args: source)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    result = verification.verify_pilot_numerics(
        config, checkpoint=None, output_dir=tmp_path / "numeric"
    )
    payload = json.loads(result.model_dump_json())
    if binding == "resume_device":
        payload["cpu_resume"]["device"] = "mps"
        payload["cpu_resume"]["comparison"]["mode"] = "resume_mps"
        for group in ("forward", "losses", "gradients", "parameters", "optimizer"):
            payload["cpu_resume"]["comparison"][group].update(rtol=1e-4, atol=1e-5)
    else:
        payload["runtime_subset"]["indices"] = payload["runtime_subset"]["indices"][:8]
        payload["runtime_subset"]["episode_sha256s"] = payload["runtime_subset"]["episode_sha256s"][
            :8
        ]
        payload["runtime_subset"]["variant_counts"] = {
            "positive": 4,
            "safe_negative": 2,
            "disconnected_negative": 2,
        }
    with pytest.raises(ValueError, match=r"resume device|runtime.*inventory"):
        verification.PilotNumericReport.model_validate_json(canonical_json_bytes(payload))


def test_full_input_archive_is_retained_and_runtime_export_is_untouched(
    verification, resume_case, tmp_path, monkeypatch
):
    from silent_cascade.eventflow.neural import NeuralModelIdentity
    from silent_cascade.train.pilot_checkpoints import save_pilot_checkpoint

    config, model, optimizer, progress, source = resume_case
    monkeypatch.setattr(verification, "authenticate_numeric_source", lambda *args: source)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    checkpoint = tmp_path / "input.safetensors"
    digest = save_pilot_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        progress=progress,
        config=config,
        source=source,
    )
    original = checkpoint.read_bytes()
    result = verification.verify_pilot_numerics(
        config, checkpoint=checkpoint, output_dir=tmp_path / "numeric"
    )
    assert result.evidence_kind == "checkpoint_numerical_diagnostic"
    assert not result.selection_verified and not result.data_introductions_verified
    assert result.original_checkpoint_sha256 == digest
    assert (
        result.runtime_model_sha256
        == NeuralModelIdentity.from_model(
            model, source_revision=source.source_commit
        ).model_state_sha256
    )
    assert (
        (tmp_path / "numeric/input-training.safetensors").read_bytes()
        == original
        == checkpoint.read_bytes()
    )
    assert (
        verification.read_numeric_report(
            tmp_path / "numeric", config=config, source_commit=source.source_commit
        )
        == result
    )
    import json

    from silent_cascade.errors import ReplayError
    from silent_cascade.eventflow.neural_weights import decode_archive, encode_archive
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
    from silent_cascade.train.pilot_checkpoints import PilotCheckpoint

    metadata, tensors = decode_archive(original, digest, PilotCheckpoint, "pilot_training")
    name = next(n for n in tensors if n.startswith("parameter/"))
    tensors[name] = tensors[name] + 0.25
    changed = encode_archive(tensors, metadata, "pilot_training")
    root = tmp_path / "numeric"
    (root / "input-training.safetensors").write_bytes(changed)
    payload = json.loads((root / "numeric-report.json").read_bytes())
    payload["original_checkpoint_sha256"] = sha256_bytes(changed)
    payload["artifact_hashes"]["input-training.safetensors"] = sha256_bytes(changed)
    raw = canonical_json_bytes(payload)
    (root / "numeric-report.json").write_bytes(raw)
    (root / "DONE").write_bytes(canonical_json_bytes({"report_sha256": sha256_bytes(raw)}))
    with pytest.raises((ValueError, ReplayError), match=r"archive|weight|model|state"):
        verification.read_numeric_report(root, config=config, source_commit=source.source_commit)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="native MPS missing; not parity evidence"
)
def test_native_raw_runtime_summary_cannot_be_rehashed(
    verification, resume_case, tmp_path, monkeypatch
):
    import json

    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

    config, _, _, _, source = resume_case
    monkeypatch.setattr(verification, "authenticate_numeric_source", lambda *args: source)
    root = tmp_path / "numeric"
    result = verification.verify_pilot_numerics(config, checkpoint=None, output_dir=root)
    assert not result.missing_devices
    assert (
        verification.read_numeric_report(root, config=config, source_commit=source.source_commit)
        == result
    )
    payload = json.loads((root / "numeric-report.json").read_bytes())
    payload["runtime_comparison"]["decision_mismatches"] += 1
    raw = canonical_json_bytes(payload)
    (root / "numeric-report.json").write_bytes(raw)
    (root / "DONE").write_bytes(canonical_json_bytes({"report_sha256": sha256_bytes(raw)}))
    with pytest.raises(ValueError, match="raw runtime comparison"):
        verification.read_numeric_report(root, config=config, source_commit=source.source_commit)
