"""Canonical, bounded public identities for runtime crashes and replay comparison.

Only declared invariant identifiers carry replay-certifiable meaning. Consumers
must reject unclassified identities before comparing payloads; two generic errors
are not evidence that the same failure was reproduced. Locations aid diagnosis
and are intentionally separate from semantic identity, since replay call stacks
can differ from the original execution entry point.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from silent_cascade.errors import DynamicsError, SilentCascadeError, TimeOrderError
from silent_cascade.validation import JsonValue

# Closed vocabulary from the actual runtime boundary checks. Never infer an
# identity from arbitrary exception messages, context fields or class names.
_PUBLIC_INVARIANTS = frozenset(
    {
        "act_emission",
        "action_after_activation",
        "action_causal_id",
        "action_mode",
        "action_type",
        "actions_preserved",
        "activation_initializes_public_focus",
        "activation_time",
        "activation_time_preserved",
        "active_focus",
        "active_mode_requires_activation",
        "active_record_id",
        "active_record_legal",
        "active_record_mode",
        "active_record_rank_bound",
        "at_most_one_action",
        "cached_prediction_snapshot",
        "clamp_owner",
        "compose_clears_active",
        "compose_consumes_active",
        "compose_support_append",
        "continuous_matches_segment",
        "counters_monotone",
        "endogenous_kind",
        "event_type",
        "eventflow_noop",
        "external_before_activation",
        "external_event_id",
        "guards_match_segment",
        "have_memory_active_record",
        "holding_hazard_hypothesis",
        "hypothesis_support_ledger",
        "inferred_hypothesis",
        "inferred_support_nonempty",
        "internal_count_owner",
        "internal_event_id",
        "jump_time",
        "jump_transition",
        "last_event_id",
        "last_event_identity",
        "last_event_time",
        "memory_metadata_monotone",
        "mode",
        "next_event_time",
        "no_event_after_terminal",
        "observing_before_activation",
        "offline_zero_model_calls",
        "one_jump",
        "only_act_emits",
        "origin_guards_reset",
        "perceived_memory",
        "perceived_records_preserved",
        "perceived_support",
        "post_jump_guards_reset",
        "post_jump_identity",
        "post_jump_segment",
        "post_jump_time",
        "prediction_after_activation",
        "prediction_delta",
        "prediction_mode",
        "prediction_parent",
        "prediction_snapshot",
        "public_focus",
        "record_observation_time",
        "refractory_owner",
        "refractory_shape",
        "runtime_core",
        "runtime_device",
        "runtime_state",
        "runtime_structure",
        "same_kind_refractory",
        "segment_origin_time",
        "segment_parent",
        "segment_time",
        "support_ledger_preserved",
        "terminal_destination",
        "terminal_guards_reset",
        "executed_internal_events",
        "consecutive_gap_clamps",
        "internal_event_cap",
        "gap_clamp_limit",
        "gap_clamp_streak",
        "source_revision_format",
        "cached_prediction_parent",
        "segment_storage_changed",
        "public_history_changed",
        "initial_runtime_mode",
        "initial_runtime_time",
        "initial_public_history",
        "prediction_input_mutated",
        "missing_next_event",
        "selected_event_time",
        "advance_after_terminal",
        "advance_time",
        "callback_runtime_result",
        "callback_counter_owner",
        "callback_actions_rewritten",
        "callback_emission_mismatch",
        "jump_input_mutated",
        "pause_after_terminal",
        "pause_time",
        "missing_terminal_score",
    }
)


@dataclass(frozen=True, slots=True)
class RuntimeDiagnosticIdentity:
    code: str
    invariant: str | None

    @property
    def replay_certifiable(self) -> bool:
        return (
            self.code in {"dynamics_error", "time_order_error"}
            and self.invariant in _PUBLIC_INVARIANTS
        )

    def to_payload(self) -> dict[str, JsonValue]:
        """Canonical error payload shared by publication and future replay checks."""
        return {
            "code": self.code,
            "message": "runtime execution failed",
            "context": {"invariant": self.invariant, "replay_certifiable": self.replay_certifiable},
        }


def runtime_diagnostic_identity(error: SilentCascadeError) -> RuntimeDiagnosticIdentity:
    """Project only an exact built-in typed error and a declared invariant ID.

    Unknown callback subclasses cannot publish arbitrary codes or claim a known
    identity. Their broad built-in error category is retained for diagnosis,
    with invariant=None and replay_certifiable=False.
    """
    code = (
        "time_order_error"
        if isinstance(error, TimeOrderError)
        else "dynamics_error"
        if isinstance(error, DynamicsError)
        else "silent_cascade_error"
    )
    invariant = error.context.get("invariant")
    if (
        type(error) not in (DynamicsError, TimeOrderError)
        or type(invariant) is not str
        or invariant not in _PUBLIC_INVARIANTS
    ):
        invariant = None
    return RuntimeDiagnosticIdentity(code, invariant)


# File identity must match the installed project's exact module path, not a
# suffix supplied by arbitrary code. Qualnames and output labels are explicit.
_PROJECT_FUNCTIONS = {
    "eventflow.engine": frozenset(
        {
            "_failure_boundary.<locals>.guarded",
            "EventEngine.__init__",
            "EventEngine._check_session",
            "EventEngine.start_episode",
            "EventEngine.next_internal_event",
            "EventEngine.next_internal_event.<locals>.predict",
            "EventEngine._next_choice",
            "EventEngine.advance_to",
            "EventEngine._require_callback_result",
            "EventEngine._require_emitted_matches_state",
            "EventEngine._execute_choice",
            "EventEngine.step",
            "EventEngine.run_until",
            "EventEngine.run_episode",
        }
    ),
    "eventflow.invariants": frozenset(
        {
            "_require",
            "_typed_boundary.<locals>.checked",
            "_validate_core",
            "validate_runtime_state",
            "validate_session_boundary",
            "validate_post_jump",
        }
    ),
    "eventflow.scheduling": frozenset(
        {
            "ExternalEventQueue.__init__",
            "ExternalEventQueue.consume",
            "PredictionCache.get_or_predict",
            "TieCandidate.__post_init__",
            "TieResolution.__post_init__",
            "selected_gap_clamp_streak",
            "normalize_internal_gap",
            "choose_next_event",
        }
    ),
    "eventflow.state": frozenset(
        {
            "require_time",
            "_validate_tensor",
            "_require_bounds",
            "ContinuousChannels.__post_init__",
            "ContinuousState.__post_init__",
            "SegmentParameters.__post_init__",
            "ComputeCounters.__post_init__",
            "AnalyticSegment.__post_init__",
            "RuntimeCore.__post_init__",
            "RuntimeState.__post_init__",
        }
    ),
}
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_APPROVED_LOCATIONS = {
    (
        str(_PACKAGE_ROOT.joinpath(*module.split(".")).with_suffix(".py")),
        qualname,
    ): f"silent_cascade.{module}.{qualname}"
    for module, names in _PROJECT_FUNCTIONS.items()
    for qualname in names
}


def safe_project_traceback(error: SilentCascadeError) -> str:
    """Return at most 20 approved module/function labels and bounded line numbers.

    Omit unknown frames, chained exceptions, source, locals and all raw filenames.
    Never claim an unrecognized traceback is a replay-certifiable failure.
    """
    locations = deque(maxlen=20)
    cursor = error.__traceback__
    while cursor is not None:
        code = cursor.tb_frame.f_code
        label = _APPROVED_LOCATIONS.get((code.co_filename, code.co_qualname))
        if label is not None and 0 < cursor.tb_lineno <= 1_000_000:
            locations.append(f"{label}:{cursor.tb_lineno}")
        cursor = cursor.tb_next
    return "\n".join(locations) or "no approved project frames"
