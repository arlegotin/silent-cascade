"""Policy-only synthetic rows are not scientific validation artifacts."""

import pytest

from silent_cascade.hashing import canonical_json_bytes, sha256_bytes


@pytest.fixture
def pilot_training_case():
    from silent_cascade.train.pilot_state import (
        PilotCheckpointDescriptor,
        PilotProgress,
        ValidationRecord,
    )

    rows = tuple(
        {
            "recall_correct": [True],
            "composition_correct": [True],
            "chain_correct": True,
            "action_correct": True,
            "category": "positive",
            "error": None,
        }
        for _ in range(10000)
    )
    record = ValidationRecord(
        stage="one_hop",
        global_step=2000,
        manifest_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        model_state_sha256="c" * 64,
        rows=rows,
        rows_sha256=sha256_bytes(canonical_json_bytes({"rows": rows})),
        production=True,
        evidence_kind="content_validation",
    )
    progress = PilotProgress(
        global_step=2000,
        batch_counter=2000,
        manifest_hashes={"one_hop": "a" * 64},
        evaluation_weights=PilotCheckpointDescriptor(
            path="synthetic.safetensors",
            sha256="b" * 64,
            model_state_sha256="c" * 64,
            global_step=2000,
            stage="one_hop",
        ),
    )
    return progress, record


def test_stage_promotion_does_not_reset_budget(pilot_training_case):
    from silent_cascade.train.pilot_state import apply_validation

    before, record = pilot_training_case
    after = apply_validation(before, record)
    assert (after.stage, after.global_step, after.batch_counter) == ("two_hop", 2000, 2000)
    assert after.stage_start_step == 2000
    assert after.patience == 0
    assert len(after.promotion_hashes) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"rows": ()},
        {"stage": "primary"},
        {"manifest_sha256": "d" * 64},
        {"global_step": 1000},
        {"evidence_kind": "phase4_overfit_engineering_diagnostic"},
        {"checkpoint_sha256": "e" * 64},
        {"model_state_sha256": "f" * 64},
    ],
)
def test_incomplete_or_wrong_evidence_rejects(pilot_training_case, change):
    from silent_cascade.train.pilot_state import apply_validation

    progress, record = pilot_training_case
    with pytest.raises(ValueError):
        apply_validation(progress, record.model_copy(update=change))


def test_debug_cannot_promote(pilot_training_case):
    from silent_cascade.train.pilot_state import apply_validation

    progress, record = pilot_training_case
    after = apply_validation(progress, record.model_copy(update={"production": False}))
    assert after.stage == "one_hop"


def test_one_hop_content_success_with_autonomous_error_cannot_promote(pilot_training_case):
    from silent_cascade.train.pilot_state import apply_validation

    progress, record = pilot_training_case
    rows = [{**row, "error": None} for row in record.rows]
    rows[0]["error"] = {"code": "dynamics_error", "invariant": "internal_event_cap"}
    record = record.model_copy(
        update={
            "rows": tuple(rows),
            "rows_sha256": sha256_bytes(canonical_json_bytes({"rows": rows})),
        }
    )
    after = apply_validation(progress, record)
    assert after.stage == "one_hop" and not after.best_rank


def test_global_ceiling_rejects_next_update():
    from silent_cascade.train.pilot_state import PilotProgress, advance_progress

    with pytest.raises(ValueError, match="budget"):
        advance_progress(PilotProgress(global_step=75000, batch_counter=75000), ceiling=75000)


def test_patience_and_earlier_step_tie_selection(pilot_training_case):
    from silent_cascade.train.pilot_state import apply_validation

    progress, record = pilot_training_case
    record = record.model_copy(update={"production": False})
    progress = progress.model_copy(update={"best_rank": (1.0, 1.0, -1000.0), "patience": 14})
    after = apply_validation(progress, record)
    assert after.status == "early_stopping" and after.patience == 15
    assert after.best_rank == (1.0, 1.0, -1000.0)


@pytest.fixture
def synthetic_timed_record():
    """Complete, hashed policy fixture; never emitted as scientific evidence."""
    from types import SimpleNamespace

    from silent_cascade.env.pilot import curriculum_to_bundle
    from silent_cascade.eval.metrics import TimedEpisodeRow
    from silent_cascade.schemas import Action
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_overfit import fixed_overfit_examples
    from silent_cascade.train.pilot_state import ValidationRecord

    config = resolve_pilot_config("phase4_smoke")
    examples = fixed_overfit_examples(config)[:4]
    identity = SimpleNamespace(
        checkpoint_sha256="b" * 64,
        model_identity=SimpleNamespace(model_state_sha256="c" * 64, source_revision="d" * 40),
        execution_source_revision="d" * 40,
        evaluation_config_canonical_json="{}",
        sha256="e" * 64,
        manifest_sha256="a" * 64,
        purpose="pilot_validation",
        gate_eligible=False,
    )
    prototypes = []
    for example in examples:
        bundle = curriculum_to_bundle(example, config=config.config)
        truth = bundle.truth
        actions = (
            ()
            if truth.relevant_hazard_type is None
            else (
                Action(
                    truth.relevant_hazard_type,
                    (truth.action_window_start + truth.action_window_end) / 2,
                    999,
                ),
            )
        )
        prototypes.append(
            TimedEpisodeRow.from_outcome(
                identity=identity, bundle=bundle, actions=actions
            ).model_dump(mode="json")
        )
    rows = tuple(
        {**prototypes[i % 4], "public_id": f"synthetic-{i}", "episode_sha256": f"{i:064x}"}
        for i in range(10000)
    )
    return ValidationRecord(
        stage="robustness",
        global_step=3000,
        manifest_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        model_state_sha256="c" * 64,
        rows=rows,
        rows_sha256=sha256_bytes(canonical_json_bytes({"rows": rows})),
        production=True,
        evidence_kind="autonomous_validation",
    )


def test_robustness_requires_current_primary_and_minimum_updates(synthetic_timed_record):
    from silent_cascade.train.pilot_state import (
        PilotCheckpointDescriptor,
        PilotProgress,
        apply_validation,
    )

    record = synthetic_timed_record
    progress = PilotProgress(
        global_step=3000,
        batch_counter=3000,
        stage="robustness",
        stage_start_step=2000,
        manifest_hashes={"robustness": "a" * 64, "primary": "a" * 64},
        evaluation_weights=PilotCheckpointDescriptor(
            path="synthetic",
            sha256="b" * 64,
            model_state_sha256="c" * 64,
            global_step=3000,
            stage="robustness",
        ),
    )
    primary = record.model_copy(update={"stage": "primary"})
    assert apply_validation(progress, record, primary=primary).status == "robustness_complete"
    assert (
        apply_validation(
            progress.model_copy(update={"stage_start_step": 2500}), record, primary=primary
        ).status
        == "running"
    )
    with pytest.raises(ValueError, match="primary"):
        apply_validation(progress, record)
    with pytest.raises(ValueError, match="primary"):
        apply_validation(progress, record, primary=primary.model_copy(update={"global_step": 2000}))


def test_any_timed_dynamics_error_makes_candidate_ineligible(synthetic_timed_record):
    record = synthetic_timed_record
    rows = list(record.rows)
    rows[0] = {
        **rows[0],
        "error": {"code": "dynamics_error", "invariant": "internal_event_cap"},
        "timed_success": False,
        "miss_category": "dynamics_error",
    }
    record = record.model_copy(
        update={
            "rows": tuple(rows),
            "rows_sha256": sha256_bytes(canonical_json_bytes({"rows": rows})),
        }
    )
    rank, passed = record.rank_and_gate()
    assert rank == () and passed is False


@pytest.mark.parametrize("mutation", ["one_hop", "duplicate_episode", "mixed_identity"])
def test_timed_evidence_rejects_wrong_gate_or_mixed_inventory(synthetic_timed_record, mutation):
    from silent_cascade.train.pilot_state import ValidationRecord

    record = synthetic_timed_record
    rows = [dict(r) for r in record.rows]
    stage = record.stage
    if mutation == "one_hop":
        stage = "one_hop"
    elif mutation == "duplicate_episode":
        rows[0]["episode_sha256"] = rows[1]["episode_sha256"]
    else:
        rows[0]["identity_sha256"] = "f" * 64
    changed = record.model_copy(
        update={
            "stage": stage,
            "rows": tuple(rows),
            "rows_sha256": sha256_bytes(canonical_json_bytes({"rows": rows})),
        }
    )
    with pytest.raises(ValueError):
        ValidationRecord.model_validate_json(canonical_json_bytes(changed))
