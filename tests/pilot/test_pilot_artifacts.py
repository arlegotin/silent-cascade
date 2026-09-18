import json

import pytest

from .test_timed_metrics import make_identity, make_row


def test_row_fsync_precedes_envelope_and_interruption_retains_uncommitted_bytes(
    neural_archive_case, tmp_path, monkeypatch
):
    import os

    from silent_cascade.eval.runner import evaluate_episodes

    from .test_pilot_archive_producer import producer_case

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    with producer_case(tmp_path, monkeypatch) as (producer, session, server):
        output = session.run_dir / "evaluation"
        synced = []
        original_fsync = os.fsync

        def fsync(descriptor):
            original_fsync(descriptor)
            row_path = output / ".rows.pending.jsonl"
            if row_path.exists() and os.fstat(descriptor).st_ino == row_path.stat().st_ino:
                synced.append(row_path.read_bytes())

        monkeypatch.setattr(os, "fsync", fsync)

        def before_envelope(**kwargs):
            assert synced == [kwargs["payload"]], "row descriptor was not durably synced"
            assert not (session.control_dir / "episode-pending.json").exists()
            assert not tuple(server._units())
            raise RuntimeError("row durable; envelope absent")

        monkeypatch.setattr(producer, "make_commit", before_envelope)
        with pytest.raises(RuntimeError, match="envelope absent"):
            evaluate_episodes(
                case.model,
                identity=identity,
                config=case.config,
                episodes=[case.bundle],
                output_dir=output,
                device="cpu",
                archive_producer=producer,
                evidence_context=session,
            )
        assert (output / ".rows.pending.jsonl").read_bytes() == synced[0]
        assert (output / "episodes/00000.neural.json").is_file()
        assert not (output / "DONE").exists()
        assert not (session.control_dir / "episode-pending.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"purpose": "unknown"},
        {"manifest_schema": "unknown"},
        {"split": "test"},
        {"purpose": "pilot_validation", "split": "validation"},
        {"manifest_schema": "phase3-component-manifest-v1"},
    ],
)
def test_invalid_identity_cannot_certify_pilot(neural_archive_case, tmp_path, change):
    with pytest.raises(ValueError):
        make_identity(neural_archive_case, (neural_archive_case.bundle,), tmp_path, **change)


def test_historical_diagnostic_keeps_actual_manifest(neural_archive_case, tmp_path):
    from silent_cascade.eval.runner import evaluate_episodes

    case = neural_archive_case
    identity = make_identity(
        case,
        (case.bundle,),
        tmp_path,
        purpose="action_diagnostic",
        manifest_schema="phase3-component-manifest-v1",
    )
    result = evaluate_episodes(
        case.model,
        identity=identity,
        config=case.config,
        episodes=[case.bundle],
        output_dir=tmp_path / "eval",
        device="cpu",
    )
    assert not result.metrics.gate_passed
    stored = json.loads((result.output_path / "identity.json").read_text())
    assert stored["manifest_schema"] == "phase3-component-manifest-v1"
    assert stored["manifest_sha256"] == "1" * 64


def test_incomplete_write_has_no_done(neural_archive_case, tmp_path):
    from silent_cascade.eval.artifacts import write_evaluation

    identity = make_identity(neural_archive_case, (neural_archive_case.bundle,), tmp_path)
    with pytest.raises(ValueError, match="inventory"):
        write_evaluation(identity=identity, rows=[], output_dir=tmp_path / "eval")
    assert not (tmp_path / "eval/DONE").exists()


def test_serialized_rows_reconstruct_scores(neural_archive_case, tmp_path):
    from silent_cascade.eval.metrics import TimedEpisodeRow, summarize_timed_rows
    from silent_cascade.eval.runner import evaluate_episodes
    from silent_cascade.hashing import sha256_file

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    result = evaluate_episodes(
        case.model,
        identity=identity,
        config=case.config,
        episodes=[case.bundle],
        output_dir=tmp_path / "eval",
        device="cpu",
    )
    rows = [
        TimedEpisodeRow.model_validate_json(line)
        for line in (result.output_path / "rows.jsonl").read_text().splitlines()
    ]
    assert rows[0].score == rows[0].recompute_score()
    assert summarize_timed_rows(rows) == result.metrics
    for name, digest in result.artifact_hashes:
        assert sha256_file(result.output_path / name) == digest


def test_done_requires_runtime_evidence(neural_archive_case, tmp_path):
    from silent_cascade.eval.artifacts import write_evaluation

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    with pytest.raises(ValueError, match="evidence"):
        write_evaluation(
            identity=identity,
            rows=[make_row(case, identity, case.bundle)],
            output_dir=tmp_path / "bad",
        )
    assert not (tmp_path / "bad/DONE").exists()


def test_delay_identity_preserves_parent_and_transform(neural_archive_case, tmp_path):
    from silent_cascade.eval.artifacts import DelayTransform, EpisodeBinding, EvaluationIdentity

    case = neural_archive_case
    transform = DelayTransform(
        parent_public_id=case.bundle.public.init.episode_public_id,
        parent_episode_sha256="2" * 64,
        version="delay-swap-v1",
        parameters_canonical_json='{"delay":8.0}',
    )
    identity = make_identity(
        case,
        (case.bundle,),
        tmp_path,
        purpose="delay_swap",
        parent_manifest_sha256="3" * 64,
        episodes=(EpisodeBinding.from_bundle(case.bundle, transform=transform),),
    )
    restored = EvaluationIdentity.model_validate_json(identity.model_dump_json())
    assert restored.episodes[0].transform == transform
    assert restored.parent_manifest_sha256 == "3" * 64
    assert not restored.gate_eligible
    for update in ({"parent_manifest_sha256": None}, {"purpose": "debug"}):
        with pytest.raises(ValueError):
            EvaluationIdentity.model_validate(dict(identity) | update)


def test_writer_rejects_corrupt_trace_before_done(neural_archive_case, tmp_path):
    from silent_cascade.eval.artifacts import write_evaluation

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    row = make_row(case, identity, case.bundle).model_copy(
        update={
            "neural_trace_ref": "sidecar.json",
            "neural_trace_sha256": "0" * 64,
        }
    )
    destination = tmp_path / "eval"
    destination.mkdir()
    (destination / "sidecar.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        write_evaluation(identity=identity, rows=[row], output_dir=destination)
    assert not (destination / "DONE").exists()


def test_writer_rejects_substituted_truth_even_with_matching_hash(neural_archive_case, tmp_path):
    from dataclasses import replace

    from silent_cascade.env.episode import EpisodeVariant
    from silent_cascade.eval.artifacts import write_evaluation
    from silent_cascade.schemas import ExternalEventKind

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    negative = replace(
        case.bundle,
        truth=replace(
            case.bundle.truth,
            recipe=replace(case.bundle.truth.recipe, variant=EpisodeVariant.DISCONNECTED_NEGATIVE),
            terminal_record_id=None,
            relevant_hazard_type=None,
            action_window_start=None,
            action_window_end=None,
            action_target=None,
            private_terminal=replace(
                case.bundle.truth.private_terminal, kind=ExternalEventKind.END
            ),
        ),
    )
    row = make_row(case, identity, negative).model_copy(
        update={
            "episode_sha256": identity.episodes[0].episode_sha256,
        }
    )
    with pytest.raises(ValueError, match="inventory"):
        write_evaluation(identity=identity, rows=[row], output_dir=tmp_path / "bad")


def test_stratified_retention_is_predeclared_and_bounded(neural_archive_case, tmp_path):
    from silent_cascade.eval.artifacts import EpisodeBinding

    case = neural_archive_case
    base = EpisodeBinding.from_bundle(case.bundle)
    variants = ("positive", "safe_negative", "disconnected_negative")
    inventory = tuple(
        base.model_copy(
            update={
                "public_id": f"retention-{n}",
                "episode_sha256": f"{n:064x}",
                "variant": variants[n % 3],
                "path_length": 2 + (n // 3) % 3,
            }
        )
        for n in range(600)
    )
    identity = make_identity(case, (case.bundle,), tmp_path, episodes=inventory)
    selected = identity.retained_public_ids()
    assert len(selected) == 500
    counts = {}
    for episode in identity.episodes:
        if episode.public_id in selected:
            key = (episode.variant, episode.path_length)
            counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 9
    assert max(counts.values()) - min(counts.values()) <= 1


@pytest.fixture
def retained_evaluation(neural_archive_case, tmp_path):
    import shutil

    from silent_cascade.eval.metrics import TimedEpisodeRow
    from silent_cascade.eval.runner import evaluate_episodes

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    original = tmp_path / "original"
    evaluate_episodes(
        case.model,
        identity=identity,
        config=case.config,
        episodes=[case.bundle],
        output_dir=original,
        device="cpu",
    )
    row = TimedEpisodeRow.model_validate_json((original / "rows.jsonl").read_bytes())
    destination = tmp_path / "republished"
    shutil.copytree(original / "episodes", destination / "episodes")
    return case, identity, row, destination


def test_trajectory_writer_records_origin_and_reads_legacy_exact_cpu(retained_evaluation):
    import gzip

    from silent_cascade.hashing import canonical_json_bytes
    from silent_cascade.logging.neural_trace import validate_full_neural_trace

    _, identity, row, root = retained_evaluation
    payload = json.loads(gzip.decompress((root / row.full_trace_ref).read_bytes()))
    assert payload["schema"] == "phase4-neural-trajectory-v2"
    assert payload["origin_device"] == "cpu"
    events = json.loads((root / row.neural_trace_ref).read_bytes())["causal_events"]
    payload["schema"] = "phase4-neural-trajectory-v1"
    del payload["origin_device"]
    validate_full_neural_trace(
        gzip.compress(canonical_json_bytes(payload), mtime=0),
        identity_sha256=identity.sha256,
        episode_sha256=row.episode_sha256,
        events=events,
        initialization_failed=False,
    )


def test_portable_terminal_rounding_does_not_relax_live_checkpoint_or_cpu(retained_evaluation):
    import gzip

    import torch

    from silent_cascade.errors import DynamicsError
    from silent_cascade.eventflow.invariants import validate_runtime_state
    from silent_cascade.logging.neural_trace import _trajectory_state

    _, _, row, root = retained_evaluation
    anchor = json.loads(gzip.decompress((root / row.full_trace_ref).read_bytes()))["anchors"][-1]
    assert anchor["core"]["mode"] == "terminal"
    assert anchor["time"] > anchor["segment"]["started_at"]
    values = anchor["core"]["continuous"]["focus_key"]["values"]
    values[0] = torch.nextafter(torch.tensor(values[0]), torch.tensor(1.0)).item()
    with pytest.raises(DynamicsError):
        _trajectory_state(anchor)
    # Portable construction retains original bits; it cannot silently normalize
    # the tensor or weaken the ordinary exact live/checkpoint validator.
    state = _trajectory_state(anchor, origin_device="mps")
    assert state.core.continuous.focus_key[0].item() == values[0]
    with pytest.raises(DynamicsError):
        validate_runtime_state(state)
    with pytest.raises(DynamicsError):
        _trajectory_state(anchor, origin_device="cpu")
    values[0] += 0.1
    values[0] = torch.tensor(values[0]).item()
    with pytest.raises(DynamicsError):
        _trajectory_state(anchor, origin_device="mps")


@pytest.mark.parametrize(
    "corruption", ["zero_time", "nonterminal", "guards", "snapshot", "event_time", "provenance"]
)
def test_portable_validation_keeps_other_invariants_exact(retained_evaluation, corruption):
    import gzip

    import torch

    from silent_cascade.errors import DynamicsError
    from silent_cascade.logging.neural_trace import _trajectory_state

    _, _, row, root = retained_evaluation
    anchors = json.loads(gzip.decompress((root / row.full_trace_ref).read_bytes()))["anchors"]
    anchor = anchors[0] if corruption == "zero_time" else anchors[-1]
    if corruption in {"zero_time", "nonterminal"}:
        values = anchor["core"]["continuous"]["focus_key"]["values"]
        values[0] = torch.nextafter(torch.tensor(values[0]), torch.tensor(1.0)).item()
        if corruption == "nonterminal":
            anchor["core"]["mode"] = "observing"
    elif corruption == "guards":
        anchor["core"]["continuous"]["guard_accumulators"]["values"][0] = torch.tensor(1e-7).item()
    elif corruption == "snapshot":
        anchor["segment"]["prediction_snapshot_sha256"] = "f" * 64
    elif corruption == "event_time":
        anchor["core"]["last_event_time"] = anchor["time"] + 1.0
    else:
        anchor["core"]["memory"]["records"][0]["record"]["provenance"] = "inferred"
    with pytest.raises((DynamicsError, ValueError)):
        _trajectory_state(anchor, origin_device="mps")


@pytest.mark.parametrize("origin", [None, "cuda", "MPS", 1])
def test_nonempty_v2_trajectory_rejects_invalid_origin(retained_evaluation, origin):
    import gzip

    from silent_cascade.hashing import canonical_json_bytes
    from silent_cascade.logging.neural_trace import validate_full_neural_trace

    _, identity, row, root = retained_evaluation
    payload = json.loads(gzip.decompress((root / row.full_trace_ref).read_bytes()))
    payload.update(schema="phase4-neural-trajectory-v2", origin_device=origin)
    events = json.loads((root / row.neural_trace_ref).read_bytes())["causal_events"]
    with pytest.raises(ValueError, match="trajectory"):
        validate_full_neural_trace(
            gzip.compress(canonical_json_bytes(payload), mtime=0),
            identity_sha256=identity.sha256,
            episode_sha256=row.episode_sha256,
            events=events,
            initialization_failed=False,
        )


def test_empty_initialization_trajectory_requires_null_origin(tmp_path):
    import gzip

    from silent_cascade.hashing import canonical_json_bytes
    from silent_cascade.logging.neural_trace import (
        validate_full_neural_trace,
        write_full_neural_trace,
    )

    path = tmp_path / "empty.gz"
    write_full_neural_trace(
        path, identity_sha256="a" * 64, episode_sha256="b" * 64, trajectory=None
    )
    payload = json.loads(gzip.decompress(path.read_bytes()))
    assert payload["origin_device"] is None
    validate_full_neural_trace(
        path.read_bytes(),
        identity_sha256="a" * 64,
        episode_sha256="b" * 64,
        events=[],
        initialization_failed=True,
    )
    payload["origin_device"] = "mps"
    with pytest.raises(ValueError, match="trajectory"):
        validate_full_neural_trace(
            gzip.compress(canonical_json_bytes(payload)),
            identity_sha256="a" * 64,
            episode_sha256="b" * 64,
            events=[],
            initialization_failed=True,
        )


def test_writer_rejects_mixed_snapshot_devices_before_publication(retained_evaluation, tmp_path):
    import gzip
    from dataclasses import fields, is_dataclass, replace
    from types import SimpleNamespace

    import torch

    from silent_cascade.logging.neural_trace import _trajectory_state, write_full_neural_trace

    if not torch.backends.mps.is_available():
        pytest.skip("native MPS required for genuinely mixed CPU/MPS snapshots")
    _, identity, row, root = retained_evaluation
    anchor = json.loads(gzip.decompress((root / row.full_trace_ref).read_bytes()))["anchors"][0]
    cpu = _trajectory_state(anchor)

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to("mps")
        if is_dataclass(value):
            return replace(value, **{f.name: move(getattr(value, f.name)) for f in fields(value)})
        if isinstance(value, tuple):
            return tuple(move(v) for v in value)
        return value

    native = move(cpu)
    from silent_cascade.eventflow.invariants import validate_runtime_state

    validate_runtime_state(native)
    destination = tmp_path / "mixed.gz"
    with pytest.raises(ValueError, match=r"uniform.*origin"):
        write_full_neural_trace(
            destination,
            identity_sha256=identity.sha256,
            episode_sha256=row.episode_sha256,
            trajectory=SimpleNamespace(checkpoint_snapshots=lambda: (cpu, native)),
        )
    assert not destination.exists()


@pytest.mark.parametrize("substitution", ["neural_sidecar", "other_episode", "other_evaluation"])
def test_retained_trajectory_substitution_never_publishes_done(
    retained_evaluation, tmp_path, substitution
):
    from dataclasses import replace

    from silent_cascade.eval.artifacts import EpisodeBinding, EvaluationIdentity, write_evaluation
    from silent_cascade.eval.metrics import TimedEpisodeRow
    from silent_cascade.eval.runner import evaluate_episodes
    from silent_cascade.hashing import sha256_bytes

    case, identity, row, destination = retained_evaluation
    if substitution == "neural_sidecar":
        row = row.model_copy(
            update={
                "full_trace_ref": row.neural_trace_ref,
                "full_trace_sha256": row.neural_trace_sha256,
            }
        )
    else:
        other_bundle = case.bundle
        if substitution == "other_episode":
            other_bundle = replace(
                case.bundle,
                public=replace(
                    case.bundle.public,
                    init=replace(
                        case.bundle.public.init,
                        episode_public_id="00000000-0000-4000-8000-000000000002",
                    ),
                ),
            )
        values = {name: getattr(identity, name) for name in EvaluationIdentity.model_fields}
        other_identity = EvaluationIdentity.model_validate(
            values
            | {
                "experiment": "other-evaluation",
                "episodes": (EpisodeBinding.from_bundle(other_bundle),),
            }
        )
        other_output = tmp_path / "other"
        evaluate_episodes(
            case.model,
            identity=other_identity,
            config=case.config,
            episodes=[other_bundle],
            output_dir=other_output,
            device="cpu",
        )
        other_row = TimedEpisodeRow.model_validate_json((other_output / "rows.jsonl").read_bytes())
        raw = (other_output / other_row.full_trace_ref).read_bytes()
        (destination / row.full_trace_ref).write_bytes(raw)
        row = row.model_copy(update={"full_trace_sha256": sha256_bytes(raw)})
    with pytest.raises(ValueError, match="trajectory"):
        write_evaluation(identity=identity, rows=[row], output_dir=destination)
    assert not (destination / "DONE").exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "schema",
        "episode_binding",
        "evaluation_binding",
        "missing_anchors",
        "empty_anchors",
        "missing_initial",
        "missing_core",
        "core_not_object",
        "missing_segment",
        "wrong_anchor_type",
        "tensor_shape",
        "tensor_values",
        "tensor_dtype",
        "anchor_event",
        "anchor_time",
        "not_gzip",
        "truncated_gzip",
        "trailing_gzip",
        "invalid_json",
        "duplicate_key",
        "oversized",
    ],
)
def test_malformed_retained_trajectory_never_publishes_done(
    retained_evaluation, monkeypatch, corruption
):
    import gzip

    from silent_cascade.eval.artifacts import write_evaluation
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
    from silent_cascade.logging import neural_trace

    _, identity, row, destination = retained_evaluation
    path = destination / row.full_trace_ref
    original = path.read_bytes()
    payload = json.loads(gzip.decompress(original))
    if corruption == "schema":
        payload["schema"] = "phase4-neural-observations-v1"
    elif corruption == "episode_binding":
        payload["episode_sha256"] = "0" * 64
    elif corruption == "evaluation_binding":
        payload["identity_sha256"] = "0" * 64
    elif corruption == "missing_anchors":
        del payload["anchors"]
    elif corruption == "empty_anchors":
        payload["anchors"] = []
    elif corruption == "missing_initial":
        payload["anchors"].pop(0)
    elif corruption == "missing_core":
        del payload["anchors"][0]["core"]
    elif corruption == "core_not_object":
        payload["anchors"][0]["core"] = list(payload["anchors"][0]["core"].items())
    elif corruption == "missing_segment":
        del payload["anchors"][0]["segment"]
    elif corruption == "wrong_anchor_type":
        payload["anchors"][0] = 0
    elif corruption.startswith("tensor_"):
        tensor = payload["anchors"][0]["core"]["continuous"]["z_fast"]
        if corruption == "tensor_shape":
            tensor["shape"] = [255]
        elif corruption == "tensor_values":
            tensor["values"] = ["not-a-float"] * 256
        else:
            tensor["dtype"] = "torch.float64"
    elif corruption == "anchor_event":
        payload["anchors"][1]["core"]["last_event_id"] = 777
    elif corruption == "anchor_time":
        payload["anchors"][1]["time"] = 777.0
    raw = gzip.compress(canonical_json_bytes(payload), mtime=0)
    if corruption == "not_gzip":
        raw = canonical_json_bytes(payload)
    elif corruption == "truncated_gzip":
        raw = raw[:-3]
    elif corruption == "trailing_gzip":
        raw += gzip.compress(b"{}", mtime=0)
    elif corruption == "invalid_json":
        raw = gzip.compress(b"{", mtime=0)
    elif corruption == "duplicate_key":
        raw = gzip.compress(b'{"schema":"forged",' + canonical_json_bytes(payload)[1:], mtime=0)
    elif corruption == "oversized":
        monkeypatch.setattr(neural_trace, "MAX_TRAJECTORY_JSON_BYTES", 1024, raising=False)
        # Otherwise-valid evidence must fail because of the bound, not JSON syntax.
        raw = original
    path.write_bytes(raw)
    row = row.model_copy(update={"full_trace_sha256": sha256_bytes(raw)})
    with pytest.raises(ValueError, match="trajectory"):
        write_evaluation(identity=identity, rows=[row], output_dir=destination)
    assert not (destination / "DONE").exists()


@pytest.mark.parametrize("failure_stage", ["initialize", "first_event"])
def test_zero_event_failure_requires_its_exact_anchor_structure(
    neural_archive_case, tmp_path, monkeypatch, failure_stage
):
    import gzip
    import shutil

    from silent_cascade.eval.artifacts import write_evaluation
    from silent_cascade.eval.metrics import TimedEpisodeRow
    from silent_cascade.eval.runner import evaluate_episodes
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent
    from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
    from silent_cascade.models.errors import NeuralError

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)

    def fail(self, *args):
        raise NeuralError("injected callback failure")

    method = "initialize" if failure_stage == "initialize" else "on_external"
    monkeypatch.setattr(NeuralEventFlowAgent, method, fail)
    result = evaluate_episodes(
        case.model,
        identity=identity,
        config=case.config,
        episodes=[case.bundle],
        output_dir=tmp_path / "eval",
        device="cpu",
    )
    row = json.loads((result.output_path / "rows.jsonl").read_bytes())
    assert result.metrics.error_count == 1
    assert row["event_count"] == 0
    assert (row["causal_trace_sha256"] is None) == (failure_stage == "initialize")
    trajectory = json.loads(
        gzip.decompress((result.output_path / row["full_trace_ref"]).read_bytes())
    )
    assert len(trajectory["anchors"]) == (0 if failure_stage == "initialize" else 1)
    assert (result.output_path / "DONE").exists()
    destination = tmp_path / "bad"
    shutil.copytree(result.output_path / "episodes", destination / "episodes")
    shutil.copytree(
        result.output_path / "crashes",
        destination / "crashes",
        ignore=shutil.ignore_patterns("index.json"),
    )
    trajectory["anchors"] = [{}] if failure_stage == "initialize" else []
    raw = gzip.compress(canonical_json_bytes(trajectory), mtime=0)
    (destination / row["full_trace_ref"]).write_bytes(raw)
    updated = TimedEpisodeRow.model_validate_json(json.dumps(row)).model_copy(
        update={"full_trace_sha256": sha256_bytes(raw)}
    )
    with pytest.raises(ValueError, match="trajectory"):
        write_evaluation(identity=identity, rows=[updated], output_dir=destination)
    assert not (destination / "DONE").exists()
