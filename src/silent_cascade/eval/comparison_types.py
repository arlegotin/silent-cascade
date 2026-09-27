"""Closed identities for the exploratory Phase 5A diagnostic comparison."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace
from silent_cascade.env.episode import EpisodeVariant
from silent_cascade.env.generator import IndependentAllocation
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from silent_cascade.validation import StrictModel

_ROOT = Path(__file__).resolve().parents[3]
_GENERATOR = _ROOT / "src/silent_cascade/env/generator.py"


def _phase1_config() -> Phase1Config:
    return resolve_config(
        Phase1Config,
        [_ROOT / "configs/base.yaml", _ROOT / "configs/data/primary.yaml"],
    ).config


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

    @property
    def config_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self))

    @property
    def protocol_sha256(self) -> str:
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "config_sha256": self.config_sha256,
                    "generator_sha256": sha256_file(_GENERATOR),
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


ACCEPTED_PHASE4_SOURCE = "3b132253512f02c0ed9f2d774fefce3036019d91"
ACCEPTED_PHASE4_GATE_SHA256 = "6353acb150fc5f46d10215e5b4206f08744818184cce3933f6d5b33e0eb75ab1"
ACCEPTED_EVENTFLOW_WEIGHTS_SHA256 = (
    "bc2850deb2bacb64d336750c5e824390e4a8eb45a3144467e3a61af84c49285b"
)
ACCEPTED_EVENTFLOW_STATE_SHA256 = "2c5e5b2a0cb3749404ec3f2a5b70dc20c00fd8a2560a4035f577311fa375e30f"
