import json

import pytest

from .test_timed_metrics import make_identity, make_row


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
