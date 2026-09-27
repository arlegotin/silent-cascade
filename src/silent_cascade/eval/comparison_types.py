"""Closed identities for the exploratory Phase 5A diagnostic comparison."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace
from silent_cascade.env.episode import EpisodeTruth, EpisodeVariant
from silent_cascade.env.generator import IndependentAllocation
from silent_cascade.env.reward import EpisodeScore, score_actions
from silent_cascade.eval.compute import RuntimeCompute
from silent_cascade.eval.metrics import EvaluationError
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from silent_cascade.schemas import Action
from silent_cascade.train.pilot_config import Phase4Config, resolve_pilot_config
from silent_cascade.validation import StrictModel

_ROOT = Path(__file__).resolve().parents[3]
_GENERATOR = _ROOT / "src/silent_cascade/env/generator.py"


def _phase1_config() -> Phase1Config:
    return resolve_config(
        Phase1Config,
        [_ROOT / "configs/base.yaml", _ROOT / "configs/data/primary.yaml"],
    ).config


def _phase4_config() -> Phase4Config:
    return resolve_pilot_config("phase4_pilot").config


class ComparisonConfig(StrictModel):
    """Fixed science choices, with transparent soft resource targets."""

    experiment_version: Literal["phase5a-v1"] = "phase5a-v1"
    purpose: Literal["exploratory_comparison"] = "exploratory_comparison"
    gate_eligible: Literal[False] = False
    iid_root_seed: Literal[7919] = 7919
    iid_public_id_seed: Literal[7927] = 7927
    depth_root_seed: Literal[7933] = 7933
    depth_public_id_seed: Literal[7937] = 7937
    analysis_seed: Literal[8009] = 8009
    scientific_seed: Literal[11] = 11
    milestone_a_time_target_seconds: Literal[7200] = 7200
    milestone_b_time_target_seconds: Literal[21600] = 21600
    retained_artifact_target_bytes: Literal[10737418240] = 10737418240
    working_reserve_bytes: Literal[2147483648] = 2147483648
    main_update_ceiling: Literal[12000] = 12000
    phase1_config: Phase1Config = Field(default_factory=_phase1_config)
    phase4_config: Phase4Config = Field(default_factory=_phase4_config)

    @model_validator(mode="after")
    def require_matching_generator(self):
        if self.phase1_config.data != self.phase4_config.data:
            raise ValueError("Phase 5A and accepted Phase 4 generator settings differ")
        return self

    @property
    def config_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self))

    @property
    def generator_sha256(self) -> str:
        return sha256_file(_GENERATOR)

    @property
    def protocol_sha256(self) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "config_sha256": self.config_sha256,
                    "generator_sha256": self.generator_sha256,
                    "schema": "phase5a-diagnostic-v1",
                    "allocation": {"iid": [172, 172, 168], "depth": [128, 128, 128, 128]},
                    "split": "debug",
                    "selection": "primary-validation-cap24-v1",
                }
            )
        )


class ComparisonEpisodeRef(StrictModel):
    episode_index: int = Field(ge=0)
    requested_path_length: int = Field(ge=1, le=8)
    variant: EpisodeVariant
    allocation_quartet_index: int = Field(ge=0)
    quartet_member_index: int = Field(ge=0, le=3)
    public_id: str = Field(min_length=1)
    episode_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted_attempt: int = Field(ge=0)


class ComparisonManifest(StrictModel):
    schema_version: Literal["phase5a-diagnostic-v1"] = "phase5a-diagnostic-v1"
    name: Literal["iid", "depth"]
    purpose: Literal["exploratory_comparison"] = "exploratory_comparison"
    gate_eligible: Literal[False] = False
    split_namespace: Literal[SplitNamespace.DEBUG] = SplitNamespace.DEBUG
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    root_seed: int = Field(ge=0)
    public_id_seed: int = Field(ge=0)
    allocation: IndependentAllocation
    allocation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: tuple[ComparisonEpisodeRef, ...]

    @field_validator("entries", mode="before")
    @classmethod
    def entries_from_json(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class BudgetLedger(StrictModel):
    """Cumulative scientific resource usage, including recorded extensions."""

    schema_version: Literal["phase5a-budget-v1"] = "phase5a-budget-v1"
    elapsed_scientific_seconds: float = Field(default=0.0, ge=0.0)
    retained_bytes: int = Field(default=0, ge=0)
    attempts: int = Field(default=0, ge=0)
    updates: int = Field(default=0, ge=0, le=12000)
    extensions: tuple[str, ...] = ()
    finish_reason: str | None = None


class ComparisonIdentity(StrictModel):
    """Immutable condition provenance, separate from private episode truth."""

    schema_version: Literal["phase5a-identity-v1"] = "phase5a-identity-v1"
    experiment_version: Literal["phase5a-v1"] = "phase5a-v1"
    purpose: Literal["exploratory_comparison"] = "exploratory_comparison"
    gate_eligible: Literal[False] = False
    condition: Literal["intact_eventflow", "compressed_eventflow", "activation_ponder"]
    manifest_name: Literal["iid", "depth"]
    transition_cap: int | None = Field(default=None, ge=4, le=24)
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    producing_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    execution_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dirty_source: bool = False
    scientific_seed: Literal[11] = 11
    device: Literal["cpu"] = "cpu"
    foundation_model_calls: Literal[0] = 0

    @model_validator(mode="after")
    def require_cap_for_ponder(self):
        if self.condition == "activation_ponder" and self.transition_cap not in {4, 8, 12, 16, 24}:
            raise ValueError("ponderer requires a predeclared cap")
        if self.condition != "activation_ponder" and self.transition_cap is not None:
            raise ValueError("EventFlow condition cannot carry a ponderer cap")
        return self

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self))


class ComparisonStep(StrictModel):
    event_id: int = Field(ge=0)
    kind: str
    timestamp: float
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predicted_crossing_at: float | None = None
    predicted_delta: float | None = Field(default=None, ge=0.0)
    bypassed_refractory_until: float | None = None
    selected_record_id: int | None = Field(default=None, ge=0)


class ConditionResult(StrictModel):
    actions: tuple[Action, ...] = ()
    stop_reason: str
    steps: tuple[ComparisonStep, ...] = ()
    trace_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    end_to_end_compute: RuntimeCompute = Field(default_factory=RuntimeCompute)
    post_activation_compute: RuntimeCompute = Field(default_factory=RuntimeCompute)
    error: EvaluationError | None = None
    inference_wall_seconds: float = Field(default=0.0, ge=0.0)


class ComparisonRow(StrictModel):
    schema_version: Literal["phase5a-row-v1"] = "phase5a-row-v1"
    public_id: str
    condition: Literal["intact_eventflow", "compressed_eventflow", "activation_ponder"]
    manifest_name: Literal["iid", "depth"]
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    episode_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    variant: EpisodeVariant
    path_length: int = Field(ge=1, le=8)
    truth: EpisodeTruth
    result: ConditionResult
    score: EpisodeScore
    timed_success: bool
    error: EvaluationError | None = None
    inference_wall_seconds: float = Field(default=0.0, ge=0.0)
    full_trace_ref: str | None = None
    full_trace_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    gate_eligible: Literal[False] = False

    @model_validator(mode="after")
    def verify_score(self):
        if self.score != score_actions(self.truth, self.result.actions):
            raise ValueError("comparison score differs from private scorer")
        if self.timed_success != (self.result.error is None and self.score.timed_success):
            raise ValueError("comparison success or error denominator is invalid")
        if self.error != self.result.error:
            raise ValueError("comparison error differs from condition result")
        if self.variant is not self.truth.recipe.variant:
            raise ValueError("comparison variant differs from private truth")
        if self.path_length != self.truth.recipe.requested_path_length:
            raise ValueError("comparison depth differs from private truth")
        if (self.full_trace_ref is None) != (self.full_trace_sha256 is None):
            raise ValueError("comparison full trace reference and hash must be paired")
        return self


ACCEPTED_PHASE4_SOURCE = "3b132253512f02c0ed9f2d774fefce3036019d91"
ACCEPTED_PHASE4_GATE_SHA256 = "6353acb150fc5f46d10215e5b4206f08744818184cce3933f6d5b33e0eb75ab1"
ACCEPTED_EVENTFLOW_WEIGHTS_SHA256 = (
    "bc2850deb2bacb64d336750c5e824390e4a8eb45a3144467e3a61af84c49285b"
)
ACCEPTED_EVENTFLOW_STATE_SHA256 = "2c5e5b2a0cb3749404ec3f2a5b70dc20c00fd8a2560a4035f577311fa375e30f"
