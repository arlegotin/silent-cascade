"""Gate arithmetic must remain independent from convenient cached pass flags."""

import pytest


@pytest.mark.parametrize(
    "change",
    [
        "wrong_denominator",
        "conditional_only",
        "hidden_error",
        "hidden_pair",
        "missing_replay",
        "missing_mps",
        "delay_12_error",
        "delay_48_error",
        "debug",
    ],
)
def test_acceptance_requires_every_independent_obligation(change):
    from silent_cascade.train.pilot_evidence import acceptance_failures
    from silent_cascade.train.pilot_evidence_types import REQUIRED_COVERAGE

    suite = dict(
        episode_count=10000,
        timed_success_count=9000,
        positive_count=5000,
        safe_count=2500,
        disconnected_count=2500,
        false_action_count=500,
        error_count=0,
        foundation_model_calls=0,
    )
    arguments = dict(
        production=True,
        selected=True,
        suites={name: dict(suite) for name in ("primary", "two_hop", "robustness")},
        repeat=True,
        pair_successes=231,
        coverage=REQUIRED_COVERAGE,
        numeric=True,
        offline=True,
        local_verified=True,
    )
    assert acceptance_failures(**arguments) == ()
    if change == "wrong_denominator":
        arguments["suites"]["primary"]["safe_count"] = 2499
    elif change == "conditional_only":
        arguments["suites"]["primary"]["timed_success_count"] = 5000
    elif change == "hidden_error":
        arguments["suites"]["two_hop"]["error_count"] = 1
    elif change == "hidden_pair":
        arguments["pair_successes"] = 230
    elif change == "missing_replay":
        arguments["coverage"] = ()
    elif change == "missing_mps":
        arguments["numeric"] = False
    elif change in {"delay_12_error", "delay_48_error"}:
        arguments["suites"][change.removesuffix("_error")] = dict(suite, error_count=1)
    else:
        arguments["production"] = False
    assert acceptance_failures(**arguments)


def test_continuation_claim_cannot_be_a_cached_flag():
    from silent_cascade.train.pilot_evidence import verify_continuation_records

    with pytest.raises(ValueError, match="continuation"):
        verify_continuation_records(
            [{"matched": True, "coverage": ["boundary:act"]}],
            rows=(),
            config=None,
            source=None,
            weights_sha="a" * 64,
        )


def test_numeric_flag_without_report_is_not_evidence():
    from silent_cascade.train.pilot_evidence import verify_numeric_evidence

    with pytest.raises(ValueError, match="numeric"):
        verify_numeric_evidence({"passed": True}, config=None, source=None, checkpoint=None)


@pytest.mark.parametrize("damage", ["trace", "event_count", "mismatches", "failed_duplicate"])
def test_portable_replay_comparison_binds_actual_trace(neural_archive_case, damage):
    from copy import deepcopy
    from types import SimpleNamespace

    from silent_cascade.env.episode import episode_sha256
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent
    from silent_cascade.eventflow.neural_replay import NeuralReplayComparison
    from silent_cascade.eventflow.replay import CausalTraceArtifact, EpisodeResultArtifact
    from silent_cascade.train.pilot_evidence import verify_continuation_records

    case = neural_archive_case
    result = case.engine.run_episode(
        case.bundle, NeuralEventFlowAgent(case.model, identity=case.identity, device="cpu")
    )
    trace = CausalTraceArtifact.from_trace(result.trace)
    outcome = EpisodeResultArtifact.from_result(result).model_dump(mode="json")
    comparison = NeuralReplayComparison(
        event_count=len(result.trace.events),
        trace_sha256=result.trace.sha256,
        weights_sha256="a" * 64,
        trace=trace,
        result=EpisodeResultArtifact.from_result(result),
    ).model_dump(mode="json")
    row = dict(
        public_id=result.public_id,
        episode_sha256=episode_sha256(case.bundle),
        model_state_sha256=case.identity.model_state_sha256,
        causal_trace_sha256=result.trace.sha256,
        actions=outcome["actions"],
        score=outcome["score"],
        truth={"recipe": {"variant": "positive"}},
    )
    events = trace.model_dump(mode="json")["events"]
    event = events[0]
    terminal = dict(
        actions=row["actions"], score=row["score"], trace_sha256=row["causal_trace_sha256"]
    )
    record = dict(
        public_id=row["public_id"],
        episode_sha256=row["episode_sha256"],
        weights_sha256="a" * 64,
        model_state_sha256=row["model_state_sha256"],
        source_commit=case.revision,
        device="cpu",
        boundary=event["kind"],
        pause_time=event["timestamp"],
        event_index=0,
        mode=event["post_mode"],
        variant="positive",
        checkpoint_path="runtime.safetensors",
        checkpoint_sha256="b" * 64,
        replay_path="replay.json",
        replay_sha256="c" * 64,
        original=terminal,
        restored=terminal,
        replay_comparisons=[deepcopy(comparison), deepcopy(comparison)],
        coverage=["boundary:fact", "mode:observing", "variant:positive"],
        matched=True,
    )
    arguments = dict(
        rows=[dict(row=row, causal_events=events)],
        config=None,
        source=SimpleNamespace(source_commit=case.revision),
        weights_sha="a" * 64,
    )
    assert verify_continuation_records([record], **arguments) == (
        "boundary:fact",
        "mode:observing",
        "variant:positive",
    )
    if damage == "failed_duplicate":
        failed = deepcopy(record)
        failed["restored"]["trace_sha256"] = "d" * 64
        # Avoid sharing the original/restored fixture dictionary in this case.
        failed["original"] = deepcopy(record["original"])
        failed["matched"] = False
        with pytest.raises(ValueError, match="continuation"):
            verify_continuation_records([record, failed], **arguments)
        return
    for value in record["replay_comparisons"]:
        if damage == "trace":
            value["trace"] = CausalTraceArtifact.from_trace(
                case.session.trace.snapshot()
            ).model_dump(mode="json")
        elif damage == "event_count":
            value["event_count"] += 1
        else:
            value["mismatches"] = ["actual replay differed"]
    with pytest.raises(ValueError, match="continuation replay"):
        verify_continuation_records([record], **arguments)
