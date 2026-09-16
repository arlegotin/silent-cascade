"""Private timed outcome records and integer-denominator single-corpus metrics."""

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import Field, model_validator

from silent_cascade.env.episode import EpisodeBundle, EpisodeTruth, episode_sha256
from silent_cascade.env.reward import EpisodeScore, score_actions
from silent_cascade.eval.compute import RuntimeCompute
from silent_cascade.hashing import sha256_bytes
from silent_cascade.schemas import Action
from silent_cascade.validation import StrictModel

Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")]
Purpose = Literal["pilot_validation", "debug", "action_diagnostic", "delay_swap"]


class EvaluationError(StrictModel):
    code: Literal["dynamics_error", "time_order_error", "validation_error"]
    invariant: str | None = None


class TimedEpisodeRow(StrictModel):
    public_id: str
    episode_sha256: Hash
    checkpoint_sha256: Hash
    model_state_sha256: Hash
    producing_source_revision: Revision
    execution_source_revision: Revision
    config_sha256: Hash
    identity_sha256: Hash
    manifest_sha256: Hash
    purpose: Purpose
    evaluation_mode: Literal["autonomous_timed"] = "autonomous_timed"
    model_seed: Literal[11] = 11
    truth: EpisodeTruth
    actions: tuple[Action, ...]
    score: EpisodeScore
    timed_success: bool
    is_positive: bool
    miss_category: (
        Literal[
            "no_action",
            "wrong_class",
            "premature",
            "late",
            "multiple_actions",
            "dynamics_error",
            "negative_false_action",
        ]
        | None
    )
    error: EvaluationError | None = None
    crash_ref: str | None = None
    event_count: int = Field(default=0, ge=0)
    event_counts: tuple[tuple[str, int], ...] = ()
    causal_trace_sha256: Hash | None = None
    neural_trace_ref: str | None = None
    neural_trace_sha256: Hash | None = None
    full_trace_ref: str | None = None
    full_trace_sha256: Hash | None = None
    compute: RuntimeCompute = Field(default_factory=RuntimeCompute)
    gate_eligible: bool = False

    def recompute_score(self) -> EpisodeScore:
        return score_actions(self.truth, self.actions)

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.score != self.recompute_score():
            raise ValueError("persisted score differs from actions and private scoring facts")
        if self.timed_success != (self.error is None and self.score.timed_success):
            raise ValueError("invalid timed success or error denominator")
        if self.is_positive != self.score.is_positive:
            raise ValueError("invalid outcome class")
        if self.miss_category != _miss(self.score, self.actions, self.truth, self.error):
            raise ValueError("invalid miss category")
        if self.gate_eligible and self.purpose != "pilot_validation":
            raise ValueError("non-acceptance evaluations cannot pass the pilot gate")
        return self

    @classmethod
    def from_outcome(cls, *, identity, bundle: EpisodeBundle, actions, error=None, **details):
        score = score_actions(bundle.truth, actions)
        return cls(
            public_id=bundle.public.init.episode_public_id,
            episode_sha256=episode_sha256(bundle),
            checkpoint_sha256=identity.checkpoint_sha256,
            model_state_sha256=identity.model_identity.model_state_sha256,
            producing_source_revision=identity.model_identity.source_revision,
            execution_source_revision=identity.execution_source_revision,
            config_sha256=sha256_bytes(identity.evaluation_config_canonical_json.encode()),
            identity_sha256=identity.sha256,
            manifest_sha256=identity.manifest_sha256,
            purpose=identity.purpose,
            truth=bundle.truth,
            actions=tuple(actions),
            score=score,
            timed_success=error is None and score.timed_success,
            is_positive=score.is_positive,
            miss_category=_miss(score, actions, bundle.truth, error),
            error=error,
            gate_eligible=identity.gate_eligible,
            **details,
        )


def _miss(score, actions, truth, error):
    if error is not None:
        return "dynamics_error"
    if score.timed_success:
        return None
    if not score.is_positive:
        return "negative_false_action"
    if not actions:
        return "no_action"
    if len(actions) > 1:
        return "multiple_actions"
    if not score.correct_class:
        return "wrong_class"
    return "premature" if actions[0].timestamp < truth.action_window_start else "late"


class PilotMetrics(StrictModel):
    episode_count: int
    timed_success_count: int
    positive_count: int
    negative_count: int
    positive_success_count: int
    negative_success_count: int
    false_action_count: int
    dynamics_error_count: int
    error_count: int
    foundation_model_calls: int
    timed_success_rate: float
    negative_false_action_rate: float
    miss_counts: tuple[tuple[str, int], ...]
    validation_valid: bool
    gate_passed: bool
    gate_scope: Literal["single_corpus_timed_only"] = "single_corpus_timed_only"


def summarize_timed_rows(rows: Sequence[TimedEpisodeRow]) -> PilotMetrics:
    """Every attempted episode contributes, including crashes and invalid candidates."""
    from collections import Counter

    count = len(rows)
    success = sum(row.error is None and row.timed_success for row in rows)
    errors = sum(row.error is not None for row in rows)
    negatives = sum(not row.is_positive for row in rows)
    false_actions = sum(not row.is_positive and bool(row.actions) for row in rows)
    foundation = sum(row.compute.foundation_model_calls for row in rows)
    eligible = (
        count == 10_000
        and negatives == 5_000
        and len({row.public_id for row in rows}) == count
        and len({row.episode_sha256 for row in rows}) == count
        and len({row.identity_sha256 for row in rows}) == 1
        and all(row.gate_eligible for row in rows)
    )
    return PilotMetrics(
        episode_count=count,
        timed_success_count=success,
        positive_count=count - negatives,
        negative_count=negatives,
        positive_success_count=sum(r.is_positive and r.timed_success for r in rows),
        negative_success_count=sum(not r.is_positive and r.timed_success for r in rows),
        false_action_count=false_actions,
        dynamics_error_count=sum(
            r.error is not None and r.error.code in {"dynamics_error", "time_order_error"}
            for r in rows
        ),
        error_count=errors,
        foundation_model_calls=foundation,
        timed_success_rate=success / count if count else 0.0,
        negative_false_action_rate=false_actions / negatives if negatives else 0.0,
        miss_counts=tuple(
            sorted(Counter(r.miss_category for r in rows if r.miss_category).items())
        ),
        validation_valid=bool(count) and errors == 0 and foundation == 0,
        gate_passed=eligible
        and success >= 9_000
        and false_actions <= 500
        and errors == 0
        and foundation == 0,
    )
