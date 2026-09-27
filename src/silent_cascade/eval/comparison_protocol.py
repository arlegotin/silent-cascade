"""Predeclared, artifact-derived Milestone A resource admission."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from silent_cascade.eval.comparison_types import (
    BudgetLedger,
    ComparisonManifest,
    ComparisonRow,
    CostEstimate,
    CostProfile,
    TrainingAdmission,
)
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.validation import StrictModel


class MilestoneDecision(StrictModel):
    schema_version: Literal["phase5a-a-decision-v1"] = "phase5a-a-decision-v1"
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["admitted_to_b", "deferred_after_a"]
    proceed_to_b: bool
    iid_denominator: int = Field(ge=0)
    depth_denominator: int = Field(ge=0)
    iid_intact_successes: int = Field(ge=0)
    depth_intact_successes: int = Field(ge=0)
    condition_counts: dict[str, int]
    reasons: tuple[str, ...]


def timing_indices(manifest: ComparisonManifest) -> tuple[int, ...]:
    """Predeclared balanced-quartet prefix: IID 12/12/8, depth 8 each."""
    quota = {2: 12, 3: 12, 4: 8} if manifest.name == "iid" else {5: 8, 6: 8, 7: 8, 8: 8}
    chosen = []
    for depth, count in quota.items():
        indices = [
            index
            for index, entry in enumerate(manifest.entries)
            if entry.requested_path_length == depth
        ]
        chosen.extend(indices[: min(count, len(indices))])
    return tuple(sorted(chosen))


def legacy_compatibility_indices(depths: Sequence[int]) -> tuple[int, ...]:
    """Select the first 12/12/8 retained primary rows by depth, independent of scores."""
    quota = {2: 12, 3: 12, 4: 8}
    chosen = []
    for index, depth in enumerate(depths):
        if quota.get(depth, 0):
            chosen.append(index)
            quota[depth] -= 1
        if len(chosen) == 32:
            break
    if len(chosen) != 32:
        raise ValueError("retained primary corpus lacks 32 predeclared compatibility rows")
    return tuple(chosen)


def decide_after_a(rows: Sequence[ComparisonRow], *, protocol_sha256: str) -> MilestoneDecision:
    """Admit B only from complete paired, valid, offline A rows and intact scores."""
    groups: dict[tuple[str, str], list[ComparisonRow]] = defaultdict(list)
    reasons: set[str] = set()
    for row in rows:
        try:
            valid = ComparisonRow.model_validate_json(canonical_json_bytes(row))
        except Exception:
            reasons.add("invalid_row")
            continue
        if valid.protocol_sha256 != protocol_sha256:
            reasons.add("protocol_mismatch")
        if valid.condition not in {"intact_eventflow", "compressed_eventflow"}:
            reasons.add("unexpected_condition")
        if valid.error is not None:
            reasons.add("dynamics_or_integrity_failure")
        if (
            valid.result.end_to_end_compute.foundation_model_calls
            or valid.result.post_activation_compute.foundation_model_calls
        ):
            reasons.add("offline_failure")
        groups[(valid.manifest_name, valid.condition)].append(valid)
    keys = (
        ("iid", "intact_eventflow"),
        ("iid", "compressed_eventflow"),
        ("depth", "intact_eventflow"),
        ("depth", "compressed_eventflow"),
    )
    for key in keys:
        if len(groups[key]) != 512:
            reasons.add("incomplete_corpus")
        ids = [(row.public_id, row.episode_sha256) for row in groups[key]]
        if len(set(ids)) != len(ids):
            reasons.add("duplicate_episode")
    for name in ("iid", "depth"):
        intact = {(row.public_id, row.episode_sha256) for row in groups[(name, "intact_eventflow")]}
        compressed = {
            (row.public_id, row.episode_sha256) for row in groups[(name, "compressed_eventflow")]
        }
        if intact != compressed:
            reasons.add("unpaired_episode")
    iid_successes = sum(row.timed_success for row in groups[("iid", "intact_eventflow")])
    depth_successes = sum(row.timed_success for row in groups[("depth", "intact_eventflow")])
    if iid_successes < 461:
        reasons.add("intact_iid_below_90_percent")
    if depth_successes < 384:
        reasons.add("intact_depth_below_75_percent")
    proceed = not reasons
    return MilestoneDecision(
        protocol_sha256=protocol_sha256,
        status="admitted_to_b" if proceed else "deferred_after_a",
        proceed_to_b=proceed,
        iid_denominator=len(groups[("iid", "intact_eventflow")]),
        depth_denominator=len(groups[("depth", "intact_eventflow")]),
        iid_intact_successes=iid_successes,
        depth_intact_successes=depth_successes,
        condition_counts={"/".join(key): len(groups[key]) for key in keys},
        reasons=tuple(sorted(reasons)),
    )


def ponder_checkpoint_rank(
    *, update: int, cap: int, primary_successes: int, negative_false_actions: int
) -> tuple[int, int, int]:
    """Only complete primary validation at cap 24 may rank a checkpoint."""
    if cap != 24 or update < 1000 or update % 1000:
        raise ValueError("ponder selection requires a 1,000-update primary cap-24 boundary")
    if not 0 <= primary_successes <= 10000 or not 0 <= negative_false_actions <= 5000:
        raise ValueError("invalid complete primary validation totals")
    return primary_successes, -negative_false_actions, -update


def admit_ponder_budget(profile: CostProfile, budget: BudgetLedger) -> TrainingAdmission:
    """Include every full validation, final cap episode, and deterministic replay."""
    required = {"one_hop", "two_hop", "primary", "robustness"}
    if set(profile.update_seconds_by_stage) != required or any(
        value < 0 for value in profile.update_seconds_by_stage.values()
    ):
        raise ValueError("cost profile lacks the fixed four-stage workload")
    estimates = {}
    for n in range(1000, 12001, 1000):
        stage_counts = {
            "one_hop": min(n, 9000),
            "two_hop": min(max(n - 9000, 0), 1000),
            "primary": min(max(n - 10000, 0), 1000),
            "robustness": min(max(n - 11000, 0), 1000),
        }
        validation_episodes = 10000 * (n // 1000)
        final_episodes = 2 * 512 * 5
        replay_episodes = 16 * (4 + 2 * 5)
        measured = (
            sum(stage_counts[stage] * profile.update_seconds_by_stage[stage] for stage in required)
            + validation_episodes * profile.validation_episode_seconds
            + final_episodes * profile.final_episode_seconds
            + replay_episodes * profile.replay_episode_seconds
            + (n // 1000 + 1) * profile.checkpoint_seconds
            + profile.report_seconds
        )
        bytes_used = (
            (n // 1000 + 1) * profile.retained_bytes_per_update_boundary
            + validation_episodes * profile.retained_bytes_per_validation_episode
            + final_episodes * profile.retained_bytes_per_final_episode
        )
        estimates[n] = CostEstimate(
            updates=n,
            validation_episodes=validation_episodes,
            final_episodes=final_episodes,
            replay_episodes=replay_episodes,
            projected_seconds=budget.elapsed_scientific_seconds + 2 * measured,
            projected_retained_bytes=budget.retained_bytes + 2 * bytes_used,
        )
    fits = [
        n
        for n, item in estimates.items()
        if item.projected_seconds <= 21600 and item.projected_retained_bytes <= 10737418240
    ]
    chosen = max(fits) if fits else None
    return TrainingAdmission(
        status="admitted" if chosen else "extension_required",
        chosen_updates=chosen,
        estimates=estimates,
        rationale=(
            "largest fixed 1,000-update boundary within 2x time/storage planning targets"
            if chosen
            else "no boundary fits 2x planning targets; extension needs a resource ruling"
        ),
    )


def nearest_ponder_cap(intact_iid_macs: float, ponder_iid_macs: dict[int, float]) -> int | None:
    """Pick the measured IID cap nearest intact compute, if within 10%."""
    if (
        intact_iid_macs <= 0
        or set(ponder_iid_macs) != {4, 8, 12, 16, 24}
        or any(value < 0 for value in ponder_iid_macs.values())
    ):
        raise ValueError("nearest-compute selection requires all five measured IID caps")
    selected = min(
        ponder_iid_macs, key=lambda cap: (abs(ponder_iid_macs[cap] - intact_iid_macs), cap)
    )
    return (
        selected
        if abs(ponder_iid_macs[selected] - intact_iid_macs) / intact_iid_macs <= 0.10
        else None
    )
