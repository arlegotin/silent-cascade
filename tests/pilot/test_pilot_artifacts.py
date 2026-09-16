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
