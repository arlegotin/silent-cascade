"""Scoring regressions use actual private truth and authoritative EpisodeScore."""

from dataclasses import replace

import pytest

from silent_cascade.env.episode import EpisodeVariant
from silent_cascade.schemas import Action, ExternalEventKind


def make_identity(case, bundles, tmp_path, **changes):
    from silent_cascade.eval.artifacts import EpisodeBinding, EvaluationIdentity
    from silent_cascade.eventflow.neural_weights import save_neural_weights

    weights = tmp_path / "source.safetensors"
    digest = save_neural_weights(weights, model=case.model, identity=case.identity)
    values = dict(
        experiment="test-evaluation",
        stage="primary",
        split="debug",
        purpose="debug",
        manifest_schema="phase4-data-v1",
        manifest_sha256="1" * 64,
        episodes=tuple(EpisodeBinding.from_bundle(bundle) for bundle in bundles),
        checkpoint_sha256=digest,
        model_identity=case.identity,
        evaluation_config_canonical_json=case.canonical,
        execution_source_revision=case.revision,
    )
    return EvaluationIdentity(**(values | changes))


def make_row(case, identity, bundle, actions=(), error=None):
    from silent_cascade.eval.metrics import TimedEpisodeRow

    return TimedEpisodeRow.from_outcome(
        identity=identity, bundle=bundle, actions=actions, error=error
    )


@pytest.mark.parametrize(
    ("actions", "success", "miss"),
    [
        ((Action(2, 2.5, 11),), True, None),
        ((Action(2, 2.8, 11),), False, "late"),
        ((Action(2, 2.49, 11),), False, "premature"),
        ((Action(1, 2.6, 11),), False, "wrong_class"),
        ((), False, "no_action"),
        ((Action(2, 2.5, 11), Action(2, 2.6, 12)), False, "multiple_actions"),
    ],
)
def test_exact_half_open_scores(neural_archive_case, tmp_path, actions, success, miss):
    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    row = make_row(case, identity, case.bundle, actions)
    assert row.timed_success is success
    assert row.miss_category == miss
    assert row.score.action_count == len(actions)
    assert row.recompute_score() == row.score


def test_failed_episodes_are_not_removed_from_denominators(neural_archive_case, tmp_path):
    from silent_cascade.eval.metrics import EvaluationError, summarize_timed_rows

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
    rows = [
        make_row(case, identity, case.bundle, (Action(2, 2.5, 11),)),
        make_row(case, identity, case.bundle),
        make_row(case, identity, negative),
        make_row(case, identity, case.bundle, error=EvaluationError(code="dynamics_error")),
    ]
    metrics = summarize_timed_rows(rows)
    assert metrics.episode_count == 4
    assert metrics.timed_success_count == 2
    assert metrics.dynamics_error_count == 1
    assert metrics.timed_success_rate == 0.5
    assert metrics.gate_passed is False
    false_action = make_row(case, identity, negative, (Action(2, 2.6, 11),))
    assert summarize_timed_rows([false_action]).negative_false_action_rate == 1.0


def test_perfect_tiny_run_cannot_pass_gate(neural_archive_case, tmp_path):
    from silent_cascade.eval.metrics import summarize_timed_rows

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    row = make_row(case, identity, case.bundle, (Action(2, 2.5, 11),))
    assert not summarize_timed_rows([row]).gate_passed


def test_gate_uses_integer_full_corpus_thresholds(neural_archive_case, tmp_path):
    """Synthetic aggregation only: no generated validation corpus or scientific run."""
    from silent_cascade.eval.metrics import EvaluationError, summarize_timed_rows

    case = neural_archive_case
    identity = make_identity(case, (case.bundle,), tmp_path)
    positive_success = make_row(case, identity, case.bundle, (Action(2, 2.5, 11),))
    positive_miss = make_row(case, identity, case.bundle)
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
    abstain = make_row(case, identity, negative)
    false_action = make_row(case, identity, negative, (Action(2, 2.5, 11),))
    templates = (
        [positive_success] * 4500 + [positive_miss] * 500 + [abstain] * 4500 + [false_action] * 500
    )
    rows = [
        row.model_copy(
            update={
                "public_id": f"metric-{n}",
                "episode_sha256": f"{n:064x}",
                "purpose": "pilot_validation",
                "gate_eligible": True,
            }
        )
        for n, row in enumerate(templates)
    ]
    metrics = summarize_timed_rows(rows)
    assert metrics.timed_success_count == 9000
    assert metrics.false_action_count == 500
    assert metrics.gate_passed
    assert not summarize_timed_rows(rows[:-1]).gate_passed
    missed = rows[0].model_copy(update={"timed_success": False})
    assert not summarize_timed_rows([missed, *rows[1:]]).gate_passed
    corrupted = rows[4500].model_copy(update={"error": EvaluationError(code="dynamics_error")})
    assert not summarize_timed_rows([*rows[:4500], corrupted, *rows[4501:]]).gate_passed
    too_many_false = rows[5000].model_copy(update={"actions": (Action(2, 2.5, 11),)})
    assert not summarize_timed_rows([*rows[:5000], too_many_false, *rows[5001:]]).gate_passed
