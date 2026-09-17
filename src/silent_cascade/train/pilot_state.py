"""Global curriculum progress and gates derived from complete bound validation rows."""

from typing import Literal

from pydantic import Field, model_validator

from silent_cascade.eval.metrics import EvaluationError, TimedEpisodeRow, summarize_timed_rows
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.train.pilot_data import Hash, Stage
from silent_cascade.validation import JsonValue, StrictModel

STAGES = ("one_hop", "two_hop", "primary", "robustness")


class PilotCheckpointDescriptor(StrictModel):
    path: str
    sha256: Hash
    model_state_sha256: Hash
    global_step: int = Field(ge=0, le=75000)
    stage: Stage
    rank: tuple[float, ...] = ()
    eligible: bool = False


class PilotProgress(StrictModel):
    global_step: int = Field(default=0, ge=0, le=75000)
    batch_counter: int = Field(default=0, ge=0, le=75000)
    stage: Stage = "one_hop"
    stage_start_step: int = Field(default=0, ge=0, le=75000)
    patience: int = Field(default=0, ge=0, le=15)
    best_rank: tuple[float, ...] = ()
    selected: PilotCheckpointDescriptor | None = None
    latest: PilotCheckpointDescriptor | None = None
    evaluation_weights: PilotCheckpointDescriptor | None = None
    promotion_hashes: tuple[Hash, ...] = ()
    manifest_hashes: dict[str, Hash] = Field(default_factory=dict)
    journal_sha256: Hash | None = None
    status: Literal["running", "robustness_complete", "step_ceiling", "early_stopping"] = "running"

    @model_validator(mode="after")
    def counters(self):
        if self.batch_counter != self.global_step or self.stage_start_step > self.global_step:
            raise ValueError("pilot global counters disagree")
        return self


class ValidationRecord(StrictModel):
    stage: Stage
    global_step: int = Field(ge=0, le=75000)
    manifest_sha256: Hash
    checkpoint_sha256: Hash
    model_state_sha256: Hash
    rows: tuple[dict[str, JsonValue], ...]
    rows_sha256: Hash
    production: bool
    evidence_kind: Literal["content_validation", "autonomous_validation"]

    @model_validator(mode="after")
    def row_integrity(self):
        if not self.rows or self.rows_sha256 != sha256_bytes(
            canonical_json_bytes({"rows": self.rows})
        ):
            raise ValueError("missing validation rows or row hash mismatch")
        if self.production and len(self.rows) != 10000:
            raise ValueError("production validation requires 10000 rows")
        if self.evidence_kind == "content_validation":
            if self.stage != "one_hop":
                raise ValueError("content validation belongs to one-hop")
            for row in self.rows:
                if (
                    set(row)
                    != {
                        "recall_correct",
                        "composition_correct",
                        "chain_correct",
                        "action_correct",
                        "category",
                        "error",
                    }
                    or any(type(row[k]) is not bool for k in ("chain_correct", "action_correct"))
                    or row["category"] not in {"positive", "safe", "disconnected"}
                    or any(
                        not isinstance(row[k], list)
                        or not row[k]
                        or any(type(v) is not bool for v in row[k])
                        for k in ("recall_correct", "composition_correct")
                    )
                ):
                    raise ValueError("invalid content validation row")
                if row["error"] is not None:
                    EvaluationError.model_validate_json(canonical_json_bytes(row["error"]))
        else:
            if self.stage == "one_hop":
                raise ValueError("one-hop promotion requires content validation")
            parsed = self.timed_rows()
            if len({r.public_id for r in parsed}) != len(parsed) or len(
                {r.episode_sha256 for r in parsed}
            ) != len(parsed):
                raise ValueError("duplicate validation row")
            for field in (
                "identity_sha256",
                "config_sha256",
                "producing_source_revision",
                "execution_source_revision",
            ):
                if len({getattr(row, field) for row in parsed}) != 1:
                    raise ValueError("mixed validation identity")
            for row in parsed:
                if (
                    row.checkpoint_sha256 != self.checkpoint_sha256
                    or row.model_state_sha256 != self.model_state_sha256
                    or row.manifest_sha256 != self.manifest_sha256
                    or (self.production and row.purpose != "pilot_validation")
                ):
                    raise ValueError("validation row identity mismatch")
        return self

    def timed_rows(self):
        return tuple(
            TimedEpisodeRow.model_validate_json(canonical_json_bytes(r)) for r in self.rows
        )

    def rank_and_gate(self):
        if self.evidence_kind == "content_validation":
            if any(row["error"] is not None for row in self.rows):
                return (), False
            recall = [v for r in self.rows for v in r["recall_correct"]]
            compose = [v for r in self.rows for v in r["composition_correct"]]
            chain = sum(r["chain_correct"] for r in self.rows) / len(self.rows)
            composition = sum(compose) / len(compose)
            return (chain, composition, -float(self.global_step)), self.production and all(
                n > 0.99 for n in (chain, composition, sum(recall) / len(recall))
            )
        metrics = summarize_timed_rows(self.timed_rows())
        valid = metrics.validation_valid
        return (
            (
                metrics.timed_success_rate,
                -metrics.negative_false_action_rate,
                -float(self.global_step),
            )
            if valid
            else ()
        ), bool(
            self.production
            and valid
            and metrics.episode_count == 10000
            and metrics.negative_count == 5000
            and metrics.timed_success_count >= 9000
            and metrics.false_action_count <= 500
        )


def apply_validation(progress, record, *, primary=None, interval=1000):
    progress = PilotProgress.model_validate_json(canonical_json_bytes(progress))
    record = ValidationRecord.model_validate_json(canonical_json_bytes(record))
    if (
        record.stage != progress.stage
        or record.global_step != progress.global_step
        or progress.global_step % interval
        or progress.manifest_hashes.get(record.stage) != record.manifest_sha256
    ):
        raise ValueError("validation stage, step or manifest mismatch")
    weights = progress.evaluation_weights
    if (
        weights is None
        or weights.global_step != record.global_step
        or weights.stage != record.stage
        or weights.sha256 != record.checkpoint_sha256
        or weights.model_state_sha256 != record.model_state_sha256
    ):
        raise ValueError("validation checkpoint differs from current exported weights")
    rank, passed = record.rank_and_gate()
    if progress.stage == "robustness":
        if primary is None:
            raise ValueError("robustness requires current primary validation")
        primary = ValidationRecord.model_validate_json(canonical_json_bytes(primary))
        if (
            primary.stage != "primary"
            or primary.global_step != record.global_step
            or primary.checkpoint_sha256 != record.checkpoint_sha256
            or primary.model_state_sha256 != record.model_state_sha256
            or primary.manifest_sha256 != progress.manifest_hashes.get("primary")
        ):
            raise ValueError("stale or mismatched primary validation")
        primary_rank, primary_pass = primary.rank_and_gate()
        passed = (
            passed and primary_pass and progress.global_step - progress.stage_start_step >= 1000
        )
        rank = (*primary_rank[:2], rank[0], -float(progress.global_step)) if passed else ()
    improved = bool(rank) and (not progress.best_rank or rank > progress.best_rank)
    updates = dict(
        best_rank=rank if improved else progress.best_rank,
        patience=0 if improved else min(15, progress.patience + 1),
    )
    if passed:
        certificate = sha256_bytes(
            canonical_json_bytes(
                {
                    "validation": record.model_dump(mode="json"),
                    "primary": primary.model_dump(mode="json") if primary else None,
                }
            )
        )
        updates["promotion_hashes"] = (*progress.promotion_hashes, certificate)
        if progress.stage == "robustness":
            updates["status"] = "robustness_complete"
        else:
            updates.update(
                stage=STAGES[STAGES.index(progress.stage) + 1],
                stage_start_step=progress.global_step,
                patience=0,
                best_rank=(),
            )
    elif updates["patience"] >= 15:
        updates["status"] = "early_stopping"
    return PilotProgress.model_validate_json(
        canonical_json_bytes(progress.model_copy(update=updates))
    )


def advance_progress(progress, *, ceiling):
    if progress.status != "running" or progress.global_step >= min(ceiling, 75000):
        raise ValueError("pilot update exceeds global budget or natural stop")
    return progress.model_copy(
        update={
            "global_step": progress.global_step + 1,
            "batch_counter": progress.batch_counter + 1,
        }
    )
