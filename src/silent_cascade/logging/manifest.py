"""Strict, immutable environment-private episode manifest publication."""

import json
import math
import uuid
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.errors import AtomicWriteError, ManifestAccessError, ManifestError
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from silent_cascade.io import atomic_create_bytes
from silent_cascade.provenance import EvidenceProvenance, public_id_seed_sha256
from silent_cascade.validation import StrictModel


class ManifestAccessClass(StrEnum):
    FIXTURE = "fixture"
    DEBUG = "debug"
    VALIDATION = "validation"
    FROZEN_TEST = "frozen_test"


class MatchedManifestCoordinate(StrictModel):
    mode: Literal["matched"] = "matched"
    cohort_index: int = Field(ge=0)
    member_index: int = Field(ge=0, le=3)


class IndependentManifestCoordinate(StrictModel):
    mode: Literal["independent"] = "independent"
    episode_index: int = Field(ge=0)
    allocation_quartet_index: int = Field(ge=0)
    quartet_member_index: int = Field(ge=0, le=3)


type ManifestCoordinate = Annotated[
    MatchedManifestCoordinate | IndependentManifestCoordinate, Field(discriminator="mode")
]


class EpisodeManifestEntry(StrictModel):
    episode_public_id: str
    split_namespace: SplitNamespace
    suite: SuiteName
    coordinate: ManifestCoordinate
    requested_path_length: int
    accepted_attempt: int
    episode_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    subset_memberships: tuple[str, ...] = ()
    parent_public_id: str | None = None
    parent_episode_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    clock_scale: float = 1.0

    @field_validator("episode_public_id", "parent_public_id")
    @classmethod
    def require_canonical_v4_uuid(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            parsed = uuid.UUID(value)
        except ValueError as error:
            raise ValueError("public ID must be a canonical UUID v4") from error
        if parsed.version != 4 or str(parsed) != value:
            raise ValueError("public ID must be a canonical UUID v4")
        return value

    @model_validator(mode="after")
    def validate_recipe_fields(self) -> "EpisodeManifestEntry":
        if type(self.requested_path_length) is not int or self.requested_path_length < 1:
            raise ValueError("requested_path_length must be an exact positive integer")
        if type(self.accepted_attempt) is not int or not 0 <= self.accepted_attempt < 1_000:
            raise ValueError("accepted_attempt must be an exact bounded integer")
        if (
            not isinstance(self.subset_memberships, tuple)
            or any(not membership for membership in self.subset_memberships)
            or len(set(self.subset_memberships)) != len(self.subset_memberships)
        ):
            raise ValueError("subset memberships must be unique nonempty strings")
        if (self.parent_public_id is None) != (self.parent_episode_sha256 is None):
            raise ValueError("paired parent ID and hash must be supplied together")
        if (
            type(self.clock_scale) is not float
            or not math.isfinite(self.clock_scale)
            or self.clock_scale <= 0
        ):
            raise ValueError("clock_scale must be an exact positive finite float")
        expected = {SuiteName.CLOCK_SCALE_0_1X: 0.1, SuiteName.CLOCK_SCALE_10X: 10.0}.get(
            self.suite
        )
        if expected is None:
            if self.clock_scale != 1.0 or self.parent_public_id is not None:
                raise ValueError("only paired clock suites may carry parent provenance")
        else:
            if self.clock_scale != expected or self.parent_public_id is None:
                raise ValueError("paired clock suites require exact factor and parent provenance")
            if (
                self.parent_public_id == self.episode_public_id
                or self.parent_episode_sha256 == self.episode_sha256
            ):
                raise ValueError(
                    "paired clock parent identity and digest must be distinct from child"
                )
        return self


def _coordinate_key(entry: EpisodeManifestEntry) -> tuple[int, ...]:
    coordinate = entry.coordinate
    if isinstance(coordinate, MatchedManifestCoordinate):
        return coordinate.cohort_index, coordinate.member_index
    return (
        coordinate.episode_index,
        coordinate.allocation_quartet_index,
        coordinate.quartet_member_index,
    )


class EpisodeManifest(StrictModel):
    schema_version: Literal[1]
    experiment_version: str = Field(min_length=1)
    access_class: ManifestAccessClass
    provenance: EvidenceProvenance
    suite: SuiteName
    public_id_seed: int
    episode_count: int
    entries: tuple[EpisodeManifestEntry, ...]

    @model_validator(mode="after")
    def validate_immutable_content(self) -> "EpisodeManifest":
        if type(self.public_id_seed) is not int or not 0 <= self.public_id_seed < 2**128:
            raise ValueError("public_id_seed must be an exact 128-bit unsigned integer")
        if self.provenance.public_id_seed_sha256 != public_id_seed_sha256(self.public_id_seed):
            raise ValueError("public_id_seed does not match provenance fingerprint")
        if type(self.episode_count) is not int or self.episode_count != len(self.entries):
            raise ValueError("episode_count must match entries")
        if not self.entries:
            raise ValueError("manifest must contain entries")
        if any(
            entry.split_namespace is not self.provenance.split_namespace
            or entry.suite is not self.suite
            for entry in self.entries
        ):
            raise ValueError("entry suite and namespace must match manifest provenance")
        if len({entry.episode_public_id for entry in self.entries}) != len(self.entries):
            raise ValueError("manifest public IDs must be unique")
        if len({entry.episode_sha256 for entry in self.entries}) != len(self.entries):
            raise ValueError("manifest episode hashes must be unique")
        coordinates = tuple(_coordinate_key(entry) for entry in self.entries)
        if len(set(coordinates)) != len(coordinates):
            raise ValueError("manifest coordinates must be unique")
        if tuple(sorted(coordinates)) != coordinates:
            raise ValueError("manifest entries must be in canonical coordinate order")
        self._validate_coordinates()
        self._validate_access_class()
        return self

    def _validate_coordinates(self) -> None:
        modes = {entry.coordinate.mode for entry in self.entries}
        if self.provenance.generation_mode not in modes or len(modes) != 1:
            raise ValueError("entry coordinate mode must match generation mode")
        if self.provenance.generation_mode == "matched":
            groups: dict[int, set[int]] = {}
            for entry in self.entries:
                coordinate = entry.coordinate
                assert isinstance(coordinate, MatchedManifestCoordinate)
                groups.setdefault(coordinate.cohort_index, set()).add(coordinate.member_index)
            if any(indices != {0, 1, 2, 3} for indices in groups.values()):
                raise ValueError("matched cohorts must contain member indices 0 through 3")
            return
        groups = {}
        for entry in self.entries:
            coordinate = entry.coordinate
            assert isinstance(coordinate, IndependentManifestCoordinate)
            groups.setdefault(coordinate.allocation_quartet_index, set()).add(
                coordinate.quartet_member_index
            )
        if any(indices != {0, 1, 2, 3} for indices in groups.values()):
            raise ValueError("independent quartets must contain member indices 0 through 3")

    def _validate_access_class(self) -> None:
        provenance = self.provenance
        if self.access_class is ManifestAccessClass.VALIDATION:
            if not (
                self.suite is SuiteName.VALIDATION
                and provenance.split_namespace is SplitNamespace.VALIDATION
                and provenance.allocation_id == "validation-v1"
                and provenance.generation_mode == "matched"
                and all(not entry.subset_memberships for entry in self.entries)
            ):
                raise ValueError("validation manifests require canonical validation provenance")
        elif self.access_class is ManifestAccessClass.FROZEN_TEST:
            if (
                provenance.split_namespace is not SplitNamespace.FROZEN
                or provenance.generation_mode != "independent"
                or not provenance.allocation_id.startswith("frozen-")
            ):
                raise ValueError(
                    "frozen-test manifests require a distinct frozen independent allocation"
                )
        elif self.access_class is ManifestAccessClass.DEBUG:
            if (
                provenance.split_namespace is not SplitNamespace.DEBUG
                or not provenance.allocation_id.startswith("test-")
            ):
                raise ValueError("debug manifests require a test- debug allocation")
        elif self.access_class is ManifestAccessClass.FIXTURE and provenance.allocation_id == "":
            raise ValueError("fixture manifests require an allocation ID")


class ManifestEnvelope(StrictModel):
    payload: EpisodeManifest
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ManifestPublication(StrictModel):
    path: str
    created: bool
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _manifest_error(message: str, *, path: Path, cause: Exception | None = None) -> ManifestError:
    context: dict[str, object] = {"path": str(path)}
    if isinstance(cause, AtomicWriteError):
        context.update(cause.context)
    elif cause is not None:
        context["reason"] = str(cause)
    return ManifestError(message, context=context)  # type: ignore[arg-type]


def _envelope_for(payload: EpisodeManifest) -> ManifestEnvelope:
    return ManifestEnvelope(
        payload=payload, payload_sha256=sha256_bytes(canonical_json_bytes(payload))
    )


def verify_manifest(envelope: ManifestEnvelope) -> EpisodeManifest:
    if not isinstance(envelope, ManifestEnvelope):
        raise ManifestError("manifest envelope has an invalid type")
    actual = sha256_bytes(canonical_json_bytes(envelope.payload))
    if actual != envelope.payload_sha256:
        raise ManifestError(
            "manifest payload hash mismatch",
            context={"expected": envelope.payload_sha256, "actual": actual},
        )
    return envelope.payload


def load_manifest(path: Path) -> EpisodeManifest:
    try:
        raw = path.read_bytes()
        envelope = ManifestEnvelope.model_validate_json(raw)
        canonical = canonical_json_bytes(envelope)
        if raw != canonical:
            raise ManifestError("manifest envelope is not canonical", context={"path": str(path)})
        return verify_manifest(envelope)
    except ManifestError:
        raise
    except (OSError, ValidationError, ValueError, json.JSONDecodeError) as error:
        raise _manifest_error(
            "manifest load or schema verification failed", path=path, cause=error
        ) from error


def publish_manifest(path: Path, manifest: EpisodeManifest) -> ManifestPublication:
    if not isinstance(path, Path) or not isinstance(manifest, EpisodeManifest):
        raise TypeError("path must be a Path and manifest must be an EpisodeManifest")
    try:
        manifest = EpisodeManifest.model_validate(manifest.model_dump())
    except ValidationError as error:
        raise ManifestError(
            "manifest content cannot be verified", context={"reason": str(error)}
        ) from error
    envelope = _envelope_for(manifest)
    candidate = canonical_json_bytes(envelope)
    try:
        atomic_create_bytes(path, candidate)
        created = True
    except AtomicWriteError as error:
        if error.message != "artifact already exists":
            raise _manifest_error(
                "immutable manifest publication failed", path=path, cause=error
            ) from error
        try:
            existing = load_manifest(path)
            existing_bytes = path.read_bytes()
        except ManifestError as load_error:
            raise _manifest_error(
                "existing manifest cannot be verified", path=path, cause=error
            ) from load_error
        if existing != manifest or existing_bytes != candidate:
            raise _manifest_error(
                "different immutable manifest already exists", path=path, cause=error
            ) from error
        created = False
    try:
        loaded = load_manifest(path)
        actual_bytes = path.read_bytes()
    except ManifestError as error:
        raise _manifest_error(
            "published manifest cannot be verified", path=path, cause=error
        ) from error
    if loaded != manifest or actual_bytes != candidate:
        raise ManifestError(
            "published manifest differs from candidate", context={"path": str(path)}
        )
    return ManifestPublication(
        path=str(path),
        created=created,
        payload_sha256=envelope.payload_sha256,
        file_sha256=sha256_file(path),
    )


def require_oracle_inspection_allowed(manifest: EpisodeManifest) -> None:
    if manifest.access_class is ManifestAccessClass.FROZEN_TEST:
        raise ManifestAccessError("oracle inspection is forbidden for frozen test data")
