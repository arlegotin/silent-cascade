"""Private Phase 1 manifest construction and authenticated regeneration.

This module is also the only Phase 1 seam which binds the isolated generator,
manifest, oracle, and leakage-audit services into publishable evidence.
"""

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal, Self
from uuid import UUID

import numpy as np
from pydantic import Field, model_validator
from scipy.stats import binomtest

from silent_cascade.config import ResolvedConfig, resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    CorpusDigestEntry,
    CorpusHashBuilder,
    EpisodeBundle,
    EpisodeVariant,
    IndependentEpisodeCoordinate,
    MatchedEpisodeCoordinate,
    PublicEpisodeArtifact,
    episode_sha256,
    scale_episode_time,
)
from silent_cascade.env.generator import (
    PHASE1_GATE_ALLOCATION,
    VALIDATION_ALLOCATION,
    CohortAllocation,
    CohortRequest,
    IndependentAllocation,
    IndependentEpisodeRequest,
    generate_independent_episode,
    generate_matched_cohort,
    iter_cohort_requests,
    iter_independent_requests,
    regenerate_independent_episode,
    regenerate_matched_episode,
    regenerate_stress_episode,
    validate_phase1_gate_allocation,
    validate_validation_allocation,
)
from silent_cascade.env.invariants import validate_cohort_invariants, validate_episode_invariants
from silent_cascade.env.leakage import (
    AuditExample,
    AuditSourceAuthentication,
    AuditSourceDescriptor,
    LeakageAuditProfileName,
    LeakageReport,
    PairedClockAuditPair,
    ReiterableAuditSource,
    _ClockPairManifestHashBuilder,
    _SourceManifestHashBuilder,
    audit_leakage,
    audit_source_descriptor_sha256,
)
from silent_cascade.env.oracle import OraclePolicy, OracleTerminalKind, solve_public_episode
from silent_cascade.env.reward import (
    RANDOM_BASELINE_NEGATIVE_SUCCESS_PROBABILITY,
    RANDOM_BASELINE_POSITIVE_SUCCESS_PROBABILITY,
    random_baseline_actions,
    score_actions,
)
from silent_cascade.env.timing import action_window
from silent_cascade.errors import AtomicWriteError
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from silent_cascade.io import atomic_create_bytes
from silent_cascade.logging.manifest import (
    EpisodeManifest,
    EpisodeManifestEntry,
    IndependentManifestCoordinate,
    ManifestAccessClass,
    MatchedManifestCoordinate,
    load_manifest,
    require_oracle_inspection_allowed,
)
from silent_cascade.provenance import (
    EvidenceProvenance,
    EvidenceProvenanceCollector,
    LeakageAuditEvidenceAnchor,
    collect_evidence_provenance,
    public_id_seed_sha256,
)
from silent_cascade.rng import (
    AllocationLabelKey,
    IndependentPublicIdKey,
    allocate_independent_public_id,
    allocate_independent_variants,
)
from silent_cascade.schemas import Action
from silent_cascade.validation import StrictModel

type HexDigest = str


class ConfigSelection(StrictModel):
    """The complete, reproducible Phase 1 configuration selection."""

    base_path: Path = Path("configs/base.yaml")
    data_path: Path = Path("configs/data/primary.yaml")
    set_overrides: tuple[str, ...] = ()


class ManifestCorpusSource(StrictModel):
    mode: Literal["manifest"] = "manifest"
    manifest_path: Path


class Phase1GateCorpusSource(StrictModel):
    mode: Literal["phase1_gate"] = "phase1_gate"
    allocation_id: str = Field(min_length=1)
    root_seed: int
    public_id_seed: int


type CorpusSource = Annotated[
    ManifestCorpusSource | Phase1GateCorpusSource, Field(discriminator="mode")
]


class FreezeValidationRequest(StrictModel):
    config: ConfigSelection
    output_path: Path
    episode_count: int = Field(default=10_000, gt=0, multiple_of=4)
    root_seed: int
    public_id_seed: int


class InspectEpisodeRequest(StrictModel):
    config: ConfigSelection
    manifest_path: Path
    episode_public_id: UUID | None = None
    entry_index: int | None = Field(default=None, ge=0)
    include_oracle: bool = False

    @model_validator(mode="after")
    def require_one_selector(self) -> Self:
        if (self.episode_public_id is None) == (self.entry_index is None):
            raise ValueError("select exactly one of episode_public_id or entry_index")
        return self


class OracleEvaluationRequest(StrictModel):
    config: ConfigSelection
    source: CorpusSource
    output_path: Path | None = None


class LeakageAuditRequest(StrictModel):
    config: ConfigSelection
    source: CorpusSource
    profile: LeakageAuditProfileName
    output_path: Path | None = None


class ArtifactPublication(StrictModel):
    path: str
    created: bool
    payload_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    file_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")


class ManifestFreezeReport(StrictModel):
    schema_version: Literal["manifest-freeze-report-v1"]
    manifest_payload_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: EvidenceProvenance
    access_class: Literal[ManifestAccessClass.DEBUG, ManifestAccessClass.VALIDATION]
    episode_count: int = Field(gt=0, multiple_of=4)
    cohort_count: int = Field(gt=0)
    positive_count: int = Field(gt=0)
    safe_negative_count: int = Field(gt=0)
    disconnected_negative_count: int = Field(gt=0)


class ManifestFreezeResult(StrictModel):
    report: ManifestFreezeReport
    publication: ArtifactPublication


class OracleInspection(StrictModel):
    terminal_kind: OracleTerminalKind
    node_path: tuple[int, ...]
    support_record_ids: tuple[int, ...]
    hazard_type: int | None
    action_window_start: float | None
    action_window_end: float | None
    action_target: float | None


class EpisodeInspectionReport(StrictModel):
    schema_version: Literal["episode-inspection-report-v1"]
    manifest_payload_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    access_class: ManifestAccessClass
    entry_index: int = Field(ge=0)
    episode_public_id: str
    public_episode: PublicEpisodeArtifact
    oracle: OracleInspection | None = None


class ExactRandomCheck(StrictModel):
    successes: int
    total: int
    observed_rate: float
    expected_rate: float
    absolute_error: float
    exact_binomial_p: float
    passed: bool


class OracleEvaluationReport(StrictModel):
    schema_version: Literal["oracle-evaluation-report-v1"]
    provenance: EvidenceProvenance
    source_mode: Literal["manifest", "phase1_gate"]
    requested_episode_count: int
    verified_episode_count: int
    positive_count: int
    safe_negative_count: int
    disconnected_negative_count: int
    suite_path_denominators: dict[str, int]
    invariant_failures: Literal[0]
    oracle_ambiguities: Literal[0]
    seed_token_collisions: Literal[0]
    public_id_collisions: Literal[0]
    oracle_successes: int
    oracle_failures: Literal[0]
    random_positive: ExactRandomCheck
    random_negative: ExactRandomCheck
    random_pooled_observed_rate: float
    random_pooled_expected_rate: Literal[0.3125]
    clock_0_1x_episode_count: int
    clock_10x_episode_count: int
    clock_decision_mismatches: Literal[0]
    rejection_reason_counts: dict[str, int]
    generation_attempt_count: int
    rejected_draw_count: int
    rejected_draw_rate: float
    corpus_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    passed: bool


class OracleEvaluationResult(StrictModel):
    report: OracleEvaluationReport
    publication: ArtifactPublication | None


class LeakageAuditResult(StrictModel):
    report: LeakageReport
    publication: ArtifactPublication | None


@dataclass(frozen=True, slots=True)
class Phase1ServiceDependencies:
    """Bound service capabilities; production and test sources cannot be mixed."""

    production_mode: bool
    validation_allocation: CohortAllocation
    independent_allocation: IndependentAllocation
    collect_provenance: EvidenceProvenanceCollector
    build_manifest: Callable[..., EpisodeManifest]
    regenerate_manifest_entry: Callable[..., EpisodeBundle]
    iter_independent: Callable[..., Iterator[IndependentEpisodeRequest]]
    generate_independent: Callable[..., EpisodeBundle]
    build_audit_source: Callable[..., ReiterableAuditSource]
    build_audit_anchor: Callable[..., LeakageAuditEvidenceAnchor]
    create_audit_workspace: Callable[[], AbstractContextManager[Path]]

    @classmethod
    def for_test(
        cls,
        *,
        validation_allocation: CohortAllocation,
        collect_provenance: EvidenceProvenanceCollector,
        build_manifest: Callable[..., EpisodeManifest],
        independent_allocation: IndependentAllocation | None = None,
        regenerate_manifest_entry: Callable[..., EpisodeBundle] | None = None,
        iter_independent: Callable[..., Iterator[IndependentEpisodeRequest]] | None = None,
        generate_independent: Callable[..., EpisodeBundle] | None = None,
        build_audit_source: Callable[..., ReiterableAuditSource] | None = None,
        build_audit_anchor: Callable[..., LeakageAuditEvidenceAnchor] | None = None,
        create_audit_workspace: Callable[[], AbstractContextManager[Path]] | None = None,
    ) -> "Phase1ServiceDependencies":
        from silent_cascade.env.generator import EpisodeBlock, IndependentAllocation

        if (
            validation_allocation.split_namespace is not SplitNamespace.DEBUG
            or not validation_allocation.allocation_id.startswith("test-")
        ):
            raise ValueError("test dependencies require a test- DEBUG validation allocation")
        independent = independent_allocation or IndependentAllocation(
            allocation_id="test-independent-v1",
            split_namespace=SplitNamespace.DEBUG,
            blocks=(
                EpisodeBlock(
                    suite=SuiteName.IID_PRIMARY,
                    requested_path_length=2,
                    first_episode_index=0,
                    episode_count=8,
                ),
            ),
        )
        # A test caller that needs independent services must supply a valid allocation;
        # freeze-only dependencies deliberately receive no iterable gate source.
        if independent.blocks and (
            independent.split_namespace is not SplitNamespace.DEBUG
            or not independent.allocation_id.startswith("test-")
        ):
            raise ValueError("test dependencies require a test- DEBUG independent allocation")
        return cls(
            False,
            validation_allocation,
            independent,
            collect_provenance,
            build_manifest,
            regenerate_manifest_entry or regenerate_entry,
            iter_independent or iter_independent_requests,
            generate_independent or generate_independent_episode,
            build_audit_source or _unconfigured_audit_source,
            build_audit_anchor or _unconfigured_audit_anchor,
            create_audit_workspace or _temporary_audit_workspace,
        )


def _unconfigured_audit_source(*args: object, **kwargs: object) -> ReiterableAuditSource:
    del args, kwargs
    raise ValueError("test dependencies require an explicit audit source builder")


def _unconfigured_audit_anchor(*args: object, **kwargs: object) -> LeakageAuditEvidenceAnchor:
    del args, kwargs
    raise ValueError("test dependencies require an independent audit anchor authority")


def _require_service_dependencies(deps: Phase1ServiceDependencies) -> None:
    if deps is PRODUCTION_DEPENDENCIES:
        return
    if deps.production_mode:
        raise ValueError("production services require sealed production dependencies")
    validation_count = sum(block.cohort_count for block in deps.validation_allocation.blocks) * 4
    independent_count = sum(block.episode_count for block in deps.independent_allocation.blocks)
    if (
        deps.validation_allocation.split_namespace is not SplitNamespace.DEBUG
        or not deps.validation_allocation.allocation_id.startswith("test-")
        or not 8 <= validation_count <= 16
        or deps.independent_allocation.split_namespace is not SplitNamespace.DEBUG
        or not deps.independent_allocation.allocation_id.startswith("test-")
        or not 8 <= independent_count <= 16
    ):
        raise ValueError("test dependencies require exact 8-16 episode DEBUG allocations")


def _require_test_output_boundary(path: Path, deps: Phase1ServiceDependencies) -> None:
    if deps.production_mode:
        return
    candidate = path if path.is_absolute() else Path.cwd() / path
    canonical_validation = (Path.cwd() / "manifests" / "validation").resolve()
    try:
        candidate.resolve().relative_to(canonical_validation)
    except ValueError:
        return
    raise ValueError("test dependencies may not publish under manifests/validation")


@contextmanager
def _temporary_audit_workspace() -> Iterator[Path]:
    with TemporaryDirectory(prefix="silent-cascade-audit-") as raw:
        yield Path(raw)


def _resolve(selection: ConfigSelection) -> ResolvedConfig[Phase1Config]:
    return resolve_config(
        Phase1Config,
        (selection.base_path, selection.data_path),
        set_overrides=selection.set_overrides,
    )


def _publish_report(path: Path, report: StrictModel) -> ArtifactPublication:
    """Atomically publish report-only canonical bytes with manifest-equivalent reuse."""
    payload = canonical_json_bytes(report)
    payload_sha256 = sha256_bytes(payload)
    try:
        atomic_create_bytes(path, payload)
        created = True
    except AtomicWriteError as error:
        if error.message != "artifact already exists":
            raise ValueError("immutable report publication failed") from error
        existing = path.read_bytes()
        if existing != payload:
            raise ValueError("different immutable report already exists") from error
        created = False
    if path.read_bytes() != payload:
        raise ValueError("published report differs from candidate")
    return ArtifactPublication(
        path=str(path),
        created=created,
        payload_sha256=payload_sha256,
        file_sha256=sha256_file(path),
    )


def _validation_provenance(
    resolved: ResolvedConfig[Phase1Config],
    request: FreezeValidationRequest,
    deps: Phase1ServiceDependencies,
) -> EvidenceProvenance:
    return deps.collect_provenance(
        resolved,
        repo_root=Path.cwd(),
        generation_mode="matched",
        allocation_id=deps.validation_allocation.allocation_id,
        split_namespace=deps.validation_allocation.split_namespace,
        root_seed=request.root_seed,
        public_id_seed=request.public_id_seed,
        analysis_seeds={},
    )


def freeze_validation(
    request: FreezeValidationRequest,
    *,
    deps: Phase1ServiceDependencies = None,  # type: ignore[assignment]
) -> ManifestFreezeResult:
    """Freeze the only publishable validation manifest and its report atomically."""
    if deps is None:
        deps = PRODUCTION_DEPENDENCIES
    _require_service_dependencies(deps)
    _require_test_output_boundary(request.output_path, deps)
    resolved = _resolve(request.config)
    allocation = deps.validation_allocation
    if deps.production_mode:
        if request.episode_count != 10_000 or allocation != VALIDATION_ALLOCATION:
            raise ValueError("production freeze requires the canonical 10,000 episode allocation")
        validate_validation_allocation(allocation, resolved.config)
    elif (
        allocation.split_namespace is not SplitNamespace.DEBUG
        or not allocation.allocation_id.startswith("test-")
        or request.episode_count != sum(block.cohort_count for block in allocation.blocks) * 4
    ):
        raise ValueError("test freeze requires its exact test- DEBUG allocation")
    provenance = _validation_provenance(resolved, request, deps)
    if (
        provenance.generation_mode != "matched"
        or provenance.allocation_id != allocation.allocation_id
        or provenance.split_namespace is not allocation.split_namespace
        or provenance.config_sha256 != resolved.sha256
        or provenance.root_seed != request.root_seed
        or provenance.public_id_seed_sha256 != public_id_seed_sha256(request.public_id_seed)
        or provenance.foundation_model_calls != 0
    ):
        raise ValueError("freeze provenance does not bind resolved inputs")
    manifest = deps.build_manifest(
        resolved.config, provenance, request.root_seed, request.public_id_seed
    )
    expected_access = (
        ManifestAccessClass.VALIDATION if deps.production_mode else ManifestAccessClass.DEBUG
    )
    if (
        manifest.access_class is not expected_access
        or manifest.episode_count != request.episode_count
        or manifest.provenance != provenance
    ):
        raise ValueError("manifest does not match the frozen service boundary")
    counts = {variant: 0 for variant in EpisodeVariant}
    for entry in manifest.entries:
        bundle = deps.regenerate_manifest_entry(resolved.config, manifest, entry)
        counts[bundle.truth.recipe.variant] += 1
    if sum(counts.values()) != manifest.episode_count:
        raise ValueError("manifest regeneration count is incomplete")
    if deps.production_mode and (
        manifest.episode_count,
        len(manifest.entries) // 4,
        counts[EpisodeVariant.POSITIVE],
        counts[EpisodeVariant.SAFE_NEGATIVE],
        counts[EpisodeVariant.DISCONNECTED_NEGATIVE],
    ) != (10_000, 2_500, 5_000, 2_500, 2_500):
        raise ValueError("production validation manifest has an invalid allocation")
    report = ManifestFreezeReport(
        schema_version="manifest-freeze-report-v1",
        manifest_payload_sha256=sha256_bytes(canonical_json_bytes(manifest)),
        provenance=provenance,
        access_class=manifest.access_class,
        episode_count=manifest.episode_count,
        cohort_count=manifest.episode_count // 4,
        positive_count=counts[EpisodeVariant.POSITIVE],
        safe_negative_count=counts[EpisodeVariant.SAFE_NEGATIVE],
        disconnected_negative_count=counts[EpisodeVariant.DISCONNECTED_NEGATIVE],
    )
    return ManifestFreezeResult(
        report=report, publication=_publish_report(request.output_path, report)
    )


def _oracle_policy_for_suite(suite: SuiteName) -> OraclePolicy:
    if suite is SuiteName.BRANCHING_STRESS:
        return OraclePolicy.BRANCHING
    if suite is SuiteName.CONTRADICTION_STRESS:
        return OraclePolicy.CONTRADICTION
    return OraclePolicy.PRIMARY


def inspect_episode(
    request: InspectEpisodeRequest,
    *,
    deps: Phase1ServiceDependencies = None,  # type: ignore[assignment]
) -> EpisodeInspectionReport:
    """Inspect a regenerated public episode; private oracle fields require access authority."""
    if deps is None:
        deps = PRODUCTION_DEPENDENCIES
    _require_service_dependencies(deps)
    resolved = _resolve(request.config)
    manifest = load_manifest(request.manifest_path)
    if manifest.provenance.config_sha256 != resolved.sha256:
        raise ValueError("manifest config does not match resolved inspection config")
    if request.entry_index is not None:
        if request.entry_index >= len(manifest.entries):
            raise ValueError("inspection entry index is outside the manifest")
        index = request.entry_index
    else:
        assert request.episode_public_id is not None
        matches = [
            index
            for index, entry in enumerate(manifest.entries)
            if entry.episode_public_id == str(request.episode_public_id)
        ]
        if len(matches) != 1:
            raise ValueError("inspection public ID is not in the manifest")
        index = matches[0]
    entry = manifest.entries[index]
    bundle = deps.regenerate_manifest_entry(resolved.config, manifest, entry)
    if bundle.public.init.episode_public_id != entry.episode_public_id:
        raise ValueError("inspection regeneration does not match manifest")
    oracle: OracleInspection | None = None
    if request.include_oracle:
        require_oracle_inspection_allowed(manifest)
        solution = solve_public_episode(bundle.public, _oracle_policy_for_suite(entry.suite))
        support = solution.link_record_ids + (
            () if solution.terminal_record_id is None else (solution.terminal_record_id,)
        )
        oracle = OracleInspection(
            terminal_kind=solution.terminal_kind,
            node_path=solution.node_path,
            support_record_ids=support,
            hazard_type=solution.hazard_type,
            action_window_start=bundle.truth.action_window_start,
            action_window_end=bundle.truth.action_window_end,
            action_target=bundle.truth.action_target,
        )
    return EpisodeInspectionReport(
        schema_version="episode-inspection-report-v1",
        manifest_payload_sha256=sha256_bytes(canonical_json_bytes(manifest)),
        access_class=manifest.access_class,
        entry_index=index,
        episode_public_id=entry.episode_public_id,
        public_episode=PublicEpisodeArtifact.from_public(bundle.public),
        oracle=oracle,
    )


def _random_generator(public_id: str) -> np.random.Generator:
    seed = int(sha256_bytes(public_id.encode("utf-8"))[:16], 16)
    return np.random.default_rng(seed)


def _random_check(successes: int, total: int, expected_rate: float) -> ExactRandomCheck:
    if total <= 0:
        raise ValueError("random diagnostic stratum is empty")
    observed = successes / total
    error = abs(observed - expected_rate)
    exact_p = float(binomtest(successes, total, expected_rate, alternative="two-sided").pvalue)
    return ExactRandomCheck(
        successes=successes,
        total=total,
        observed_rate=observed,
        expected_rate=expected_rate,
        absolute_error=error,
        exact_binomial_p=exact_p,
        passed=exact_p >= 0.001 and error <= 0.01,
    )


def _independent_provenance(
    resolved: ResolvedConfig[Phase1Config],
    source: Phase1GateCorpusSource,
    deps: Phase1ServiceDependencies,
) -> EvidenceProvenance:
    allocation = deps.independent_allocation
    if source.allocation_id != allocation.allocation_id:
        raise ValueError("source allocation does not match bound dependencies")
    if deps.production_mode:
        if (
            allocation != PHASE1_GATE_ALLOCATION
            or allocation.allocation_id != "phase1-independent-gate-v1"
        ):
            raise ValueError("production requires the canonical independent gate allocation")
        validate_phase1_gate_allocation(allocation, resolved.config)
    elif (
        allocation.split_namespace is not SplitNamespace.DEBUG
        or not allocation.allocation_id.startswith("test-")
    ):
        raise ValueError("test dependencies require a test- DEBUG independent allocation")
    provenance = deps.collect_provenance(
        resolved,
        repo_root=Path.cwd(),
        generation_mode="independent",
        allocation_id=allocation.allocation_id,
        split_namespace=allocation.split_namespace,
        root_seed=source.root_seed,
        public_id_seed=source.public_id_seed,
        analysis_seeds={},
    )
    if (
        provenance.generation_mode != "independent"
        or provenance.allocation_id != allocation.allocation_id
        or provenance.split_namespace is not allocation.split_namespace
        or provenance.root_seed != source.root_seed
        or provenance.public_id_seed_sha256 != public_id_seed_sha256(source.public_id_seed)
        or provenance.config_sha256 != resolved.sha256
        or provenance.foundation_model_calls != 0
    ):
        raise ValueError("independent provenance does not bind resolved inputs")
    return provenance


def _require_independent_bundle_matches_request(
    bundle: EpisodeBundle,
    request: IndependentEpisodeRequest,
    public_id_seed: int,
) -> None:
    """Authenticate a generated bundle against its exact allocation recipe."""
    truth = bundle.truth
    key = truth.key
    coordinate = key.coordinate
    recipe = truth.recipe
    expected_public_id = allocate_independent_public_id(
        IndependentPublicIdKey(
            generator_version="ofd-v1",
            split_namespace=request.split_namespace,
            suite=request.suite,
            public_id_seed=public_id_seed,
            episode_index=request.episode_index,
            accepted_attempt=recipe.accepted_attempt,
        )
    )
    if (
        not isinstance(coordinate, IndependentEpisodeCoordinate)
        or key.split_namespace is not request.split_namespace
        or key.suite is not request.suite
        or key.root_seed != request.root_seed
        or coordinate.episode_index != request.episode_index
        or coordinate.allocation_quartet_index != request.allocation_quartet_index
        or recipe.requested_path_length != request.requested_path_length
        or recipe.variant is not request.variant
        or recipe.evaluation_suite is not request.suite
        or recipe.parent_public_id is not None
        or bundle.public.init.episode_public_id != expected_public_id
    ):
        raise ValueError("generated episode does not match its bound allocation request")


def _require_manifest_bundle_matches_entry(
    bundle: EpisodeBundle,
    entry: EpisodeManifestEntry,
) -> None:
    """Authenticate regenerated content and its coordinate against one manifest entry."""
    _assert_entry(bundle, entry)
    key = bundle.truth.key
    recipe = bundle.truth.recipe
    coordinate = key.coordinate
    manifest_coordinate = entry.coordinate
    if isinstance(manifest_coordinate, MatchedManifestCoordinate):
        coordinate_matches = (
            isinstance(coordinate, MatchedEpisodeCoordinate)
            and coordinate.cohort_index == manifest_coordinate.cohort_index
            and coordinate.member_index == manifest_coordinate.member_index
        )
    else:
        coordinate_matches = (
            isinstance(coordinate, IndependentEpisodeCoordinate)
            and coordinate.episode_index == manifest_coordinate.episode_index
            and coordinate.allocation_quartet_index == manifest_coordinate.allocation_quartet_index
        )
    expected_key_suite = (
        SuiteName.IID_PRIMARY
        if entry.suite in {SuiteName.CLOCK_SCALE_0_1X, SuiteName.CLOCK_SCALE_10X}
        else entry.suite
    )
    if (
        not coordinate_matches
        or key.split_namespace is not entry.split_namespace
        or key.suite is not expected_key_suite
        or recipe.evaluation_suite is not entry.suite
        or recipe.requested_path_length != entry.requested_path_length
    ):
        raise ValueError("regenerated episode does not match its manifest coordinate")


def _manifest_matches_matched_allocation(
    manifest: EpisodeManifest,
    allocation: CohortAllocation,
) -> bool:
    expected_count = sum(block.cohort_count for block in allocation.blocks) * 4
    if (
        manifest.provenance.generation_mode != "matched"
        or manifest.provenance.allocation_id != allocation.allocation_id
        or manifest.provenance.split_namespace is not allocation.split_namespace
        or manifest.episode_count != expected_count
        or len({block.suite for block in allocation.blocks}) != 1
        or manifest.suite is not allocation.blocks[0].suite
    ):
        return False
    expected = tuple(
        (block.suite, block.requested_path_length, cohort_index, member_index)
        for block in allocation.blocks
        for cohort_index in range(
            block.first_cohort_index,
            block.first_cohort_index + block.cohort_count,
        )
        for member_index in range(4)
    )
    actual = tuple(
        (
            entry.suite,
            entry.requested_path_length,
            entry.coordinate.cohort_index,
            entry.coordinate.member_index,
        )
        for entry in manifest.entries
        if isinstance(entry.coordinate, MatchedManifestCoordinate)
    )
    return actual == expected


def _manifest_matches_independent_allocation(
    manifest: EpisodeManifest,
    allocation: IndependentAllocation,
) -> bool:
    expected_count = sum(block.episode_count for block in allocation.blocks)
    if (
        manifest.provenance.generation_mode != "independent"
        or manifest.provenance.allocation_id != allocation.allocation_id
        or manifest.provenance.split_namespace is not allocation.split_namespace
        or manifest.episode_count != expected_count
    ):
        return False
    requests = tuple(iter_independent_requests(allocation, manifest.provenance.root_seed))
    if len(requests) != len(manifest.entries):
        return False
    for rank, (entry, request) in enumerate(zip(manifest.entries, requests, strict=True)):
        coordinate = entry.coordinate
        suite_matches = entry.suite is request.suite or (
            request.suite is SuiteName.IID_PRIMARY
            and entry.suite in {SuiteName.CLOCK_SCALE_0_1X, SuiteName.CLOCK_SCALE_10X}
        )
        if (
            not isinstance(coordinate, IndependentManifestCoordinate)
            or not suite_matches
            or entry.requested_path_length != request.requested_path_length
            or coordinate.episode_index != request.episode_index
            or coordinate.allocation_quartet_index != request.allocation_quartet_index
            or coordinate.quartet_member_index != rank % 4
        ):
            return False
    return True


def _require_manifest_source_boundary(
    manifest: EpisodeManifest,
    deps: Phase1ServiceDependencies,
) -> None:
    """Bind a manifest source to the exact sealed production or injected allocation."""
    if deps.production_mode:
        valid = (
            manifest.access_class is ManifestAccessClass.VALIDATION
            and deps.validation_allocation == VALIDATION_ALLOCATION
            and _manifest_matches_matched_allocation(manifest, VALIDATION_ALLOCATION)
        )
        message = "production manifest source is not the canonical validation allocation"
    else:
        valid = manifest.access_class is ManifestAccessClass.DEBUG and (
            _manifest_matches_matched_allocation(manifest, deps.validation_allocation)
            or _manifest_matches_independent_allocation(manifest, deps.independent_allocation)
        )
        message = "test manifest source does not match its bound DEBUG allocation"
    if not valid:
        raise ValueError(message)


def _bound_manifest_sha256(
    path: Path,
    deps: Phase1ServiceDependencies,
    *,
    expected_sha256: str | None = None,
) -> tuple[EpisodeManifest, str]:
    manifest = load_manifest(path)
    _require_manifest_source_boundary(manifest, deps)
    digest = sha256_bytes(canonical_json_bytes(manifest))
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("manifest source changed before evidence publication")
    return manifest, digest


def evaluate_oracle(
    request: OracleEvaluationRequest,
    *,
    deps: Phase1ServiceDependencies = None,  # type: ignore[assignment]
) -> OracleEvaluationResult:
    """Regenerate and score an authenticated source without retaining episode traces."""
    if deps is None:
        deps = PRODUCTION_DEPENDENCIES
    _require_service_dependencies(deps)
    if request.output_path is not None:
        _require_test_output_boundary(request.output_path, deps)
    resolved = _resolve(request.config)
    source = request.source
    manifest_source_sha256: str | None = None
    if isinstance(source, ManifestCorpusSource):
        manifest, manifest_source_sha256 = _bound_manifest_sha256(
            source.manifest_path,
            deps,
        )
        if manifest.provenance.config_sha256 != resolved.sha256:
            raise ValueError("manifest config does not match resolved oracle config")
        embedded = manifest.provenance
        provenance = deps.collect_provenance(
            resolved,
            repo_root=Path.cwd(),
            generation_mode=embedded.generation_mode,
            allocation_id=embedded.allocation_id,
            split_namespace=embedded.split_namespace,
            root_seed=embedded.root_seed,
            public_id_seed=manifest.public_id_seed,
            analysis_seeds={},
        )
        if (
            provenance.source_dirty
            or provenance.foundation_model_calls != 0
            or provenance.generation_mode != embedded.generation_mode
            or provenance.allocation_id != embedded.allocation_id
            or provenance.split_namespace is not embedded.split_namespace
            or provenance.root_seed != embedded.root_seed
            or provenance.public_id_seed_sha256 != embedded.public_id_seed_sha256
            or provenance.config_sha256 != embedded.config_sha256
            or provenance.generator_source != embedded.generator_source
        ):
            raise ValueError("current oracle provenance does not authenticate immutable manifest")
        ordered = (
            (deps.regenerate_manifest_entry(resolved.config, manifest, entry), entry)
            for entry in manifest.entries
        )
        expected_count = manifest.episode_count
        source_mode: Literal["manifest", "phase1_gate"] = "manifest"
        expected_denominators = Counter(
            f"{entry.suite.value}:{entry.requested_path_length}" for entry in manifest.entries
        )
        clock_01 = sum(entry.suite is SuiteName.CLOCK_SCALE_0_1X for entry in manifest.entries)
        clock_10 = sum(entry.suite is SuiteName.CLOCK_SCALE_10X for entry in manifest.entries)
        clock_pairs = _make_manifest_clock_pairs(
            resolved.config,
            manifest,
            deps.regenerate_manifest_entry,
        )
        manifest_entries_by_id = {entry.episode_public_id: entry for entry in manifest.entries}
    elif isinstance(source, Phase1GateCorpusSource):
        provenance = _independent_provenance(resolved, source, deps)
        requests = deps.iter_independent(deps.independent_allocation, source.root_seed)
        ordered = (
            (
                deps.generate_independent(resolved.config, item, source.public_id_seed),
                item,
            )
            for item in requests
        )
        expected_count = sum(block.episode_count for block in deps.independent_allocation.blocks)
        source_mode = "phase1_gate"
        expected_denominators = Counter(
            {
                f"{block.suite.value}:{block.requested_path_length}": block.episode_count
                for block in deps.independent_allocation.blocks
            }
        )
        clock_01 = sum(
            block.scale_0_1x_episode_count for block in deps.independent_allocation.clock_blocks
        )
        clock_10 = sum(
            block.scale_10x_episode_count for block in deps.independent_allocation.clock_blocks
        )
        clock_pairs = _make_clock_pairs(
            resolved.config,
            deps.independent_allocation,
            source.root_seed,
            source.public_id_seed,
            deps.generate_independent,
        )
        manifest_entries_by_id = {}
    else:
        raise TypeError("unsupported oracle source")
    corpus = CorpusHashBuilder(expected_count)
    denominators: Counter[str] = Counter()
    variants: Counter[EpisodeVariant] = Counter()
    rejection_reasons: Counter[str] = Counter()
    random_successes: Counter[str] = Counter()
    seed_tokens: set[tuple[object, ...]] = set()
    verified = 0
    rejected_draws = 0
    generation_attempts = 0
    active_cohort_index: int | None = None
    active_cohort: list[EpisodeBundle] = []

    def flush_cohort() -> None:
        if active_cohort_index is not None:
            if len(active_cohort) != 4:
                raise ValueError("matched oracle source contains an incomplete cohort")
            validate_cohort_invariants(tuple(active_cohort), resolved.config)  # type: ignore[arg-type]

    for bundle, source_recipe in ordered:
        if isinstance(source_recipe, EpisodeManifestEntry):
            _require_manifest_bundle_matches_entry(bundle, source_recipe)
        else:
            _require_independent_bundle_matches_request(
                bundle,
                source_recipe,
                source.public_id_seed,
            )
        key = bundle.truth.key
        coordinate = key.coordinate
        token = (
            key.split_namespace,
            key.suite,
            key.root_seed,
            coordinate,
        )
        if token in seed_tokens:
            raise ValueError("oracle source contains a seed-token collision")
        seed_tokens.add(token)
        if bundle.truth.recipe.parent_public_id is None:
            validate_episode_invariants(bundle, resolved.config)
        else:
            entry = manifest_entries_by_id.get(bundle.public.init.episode_public_id)
            if entry is None:
                raise ValueError("clock child is absent from its manifest")
            validate_episode_invariants(
                _regenerate_manifest_clock_parent(resolved.config, manifest, entry),
                resolved.config,
            )
        if isinstance(coordinate, MatchedEpisodeCoordinate):
            if active_cohort_index is None:
                active_cohort_index = coordinate.cohort_index
            elif coordinate.cohort_index != active_cohort_index:
                flush_cohort()
                active_cohort.clear()
                active_cohort_index = coordinate.cohort_index
            active_cohort.append(bundle)
        elif active_cohort_index is not None:
            raise ValueError("matched and independent oracle entries may not be interleaved")
        solution = solve_public_episode(
            bundle.public, _oracle_policy_for_suite(bundle.truth.recipe.evaluation_suite)
        )
        from silent_cascade.env.oracle import verify_oracle_truth

        verify_oracle_truth(solution, bundle.truth)
        activation = bundle.public.events[-1]
        actions: tuple[Action, ...]
        if solution.terminal_kind is OracleTerminalKind.HAZARD:
            assert solution.hazard_type is not None and solution.public_delay is not None
            target = bundle.truth.activation_time + (
                bundle.truth.recipe.oracle_timing.action_target_fraction * solution.public_delay
            )
            actions = (Action(solution.hazard_type, target, activation.event_id),)
        else:
            actions = ()
        if not score_actions(bundle.truth, actions).timed_success:
            raise ValueError("oracle action disagrees with scorer")
        random_score = score_actions(
            bundle.truth,
            random_baseline_actions(
                bundle.public,
                bundle.truth.recipe.oracle_timing,
                _random_generator(bundle.public.init.episode_public_id),
            ),
        )
        random_stratum = (
            "positive" if bundle.truth.recipe.variant is EpisodeVariant.POSITIVE else "negative"
        )
        random_successes[random_stratum] += int(random_score.timed_success)
        variants[bundle.truth.recipe.variant] += 1
        denominators[
            f"{bundle.truth.recipe.evaluation_suite.value}:{bundle.truth.recipe.requested_path_length}"
        ] += 1
        corpus.add(CorpusDigestEntry(bundle.public.init.episode_public_id, episode_sha256(bundle)))
        verified += 1
        rejected_draws += bundle.truth.rejection_count
        generation_attempts += bundle.truth.rejection_count + 1
        rejection_reasons.update(bundle.truth.rejection_reasons)
    flush_cohort()
    if verified != expected_count or dict(sorted(denominators.items())) != dict(
        sorted(expected_denominators.items())
    ):
        raise ValueError("oracle source denominator is incomplete or out of canonical order")
    positive = variants[EpisodeVariant.POSITIVE]
    safe = variants[EpisodeVariant.SAFE_NEGATIVE]
    disconnected = variants[EpisodeVariant.DISCONNECTED_NEGATIVE]
    random_positive = _random_check(
        random_successes["positive"], positive, RANDOM_BASELINE_POSITIVE_SUCCESS_PROBABILITY
    )
    random_negative = _random_check(
        random_successes["negative"],
        safe + disconnected,
        RANDOM_BASELINE_NEGATIVE_SUCCESS_PROBABILITY,
    )
    verified_clock_counts: Counter[str] = Counter()
    clock_decision_mismatches = 0
    for pair in clock_pairs:
        if not _clock_decision_matches(pair.parent.bundle, pair.child):
            clock_decision_mismatches += 1
        verified_clock_counts[
            "scale_0_1x"
            if pair.child.truth.recipe.evaluation_suite is SuiteName.CLOCK_SCALE_0_1X
            else "scale_10x"
        ] += 1
    if (
        verified_clock_counts["scale_0_1x"] != clock_01
        or verified_clock_counts["scale_10x"] != clock_10
    ):
        raise ValueError("clock source count does not match the frozen allocation")
    if clock_decision_mismatches:
        raise ValueError("clock transform changes the oracle decision")
    pooled = (random_successes["positive"] + random_successes["negative"]) / verified
    report = OracleEvaluationReport(
        schema_version="oracle-evaluation-report-v1",
        provenance=provenance,
        source_mode=source_mode,
        requested_episode_count=expected_count,
        verified_episode_count=verified,
        positive_count=positive,
        safe_negative_count=safe,
        disconnected_negative_count=disconnected,
        suite_path_denominators=dict(sorted(denominators.items())),
        invariant_failures=0,
        oracle_ambiguities=0,
        seed_token_collisions=0,
        public_id_collisions=0,
        oracle_successes=verified,
        oracle_failures=0,
        random_positive=random_positive,
        random_negative=random_negative,
        random_pooled_observed_rate=pooled,
        random_pooled_expected_rate=0.3125,
        clock_0_1x_episode_count=clock_01,
        clock_10x_episode_count=clock_10,
        clock_decision_mismatches=clock_decision_mismatches,
        rejection_reason_counts=dict(sorted(rejection_reasons.items())),
        generation_attempt_count=generation_attempts,
        rejected_draw_count=rejected_draws,
        rejected_draw_rate=rejected_draws / generation_attempts if generation_attempts else 0.0,
        corpus_sha256=corpus.finalize(),
        passed=random_positive.passed and random_negative.passed,
    )
    if request.output_path is not None and isinstance(source, ManifestCorpusSource):
        assert manifest_source_sha256 is not None
        _bound_manifest_sha256(
            source.manifest_path,
            deps,
            expected_sha256=manifest_source_sha256,
        )
    publication = _publish_report(request.output_path, report) if request.output_path else None
    return OracleEvaluationResult(report=report, publication=publication)


@dataclass(frozen=True, slots=True)
class _BoundAuditSource:
    """A service-minted source whose data is regenerated in canonical order."""

    descriptor: AuditSourceDescriptor
    validation_config: Phase1Config
    authentication: AuditSourceAuthentication
    _examples: Callable[[], Iterator[AuditExample]]
    _clock_pairs: Callable[[], Iterator[PairedClockAuditPair]]
    publishable: bool

    @property
    def episode_count(self) -> int:
        return self.descriptor.episode_count

    def iter_examples(self) -> Iterator[AuditExample]:
        return self._examples()

    def iter_clock_pairs(self) -> Iterator[PairedClockAuditPair]:
        return self._clock_pairs()


def _base_rank(allocation: "IndependentAllocation", request: IndependentEpisodeRequest) -> int:
    rank = 0
    for block in allocation.blocks:
        if (
            block.suite is request.suite
            and block.requested_path_length == request.requested_path_length
            and block.first_episode_index
            <= request.episode_index
            < block.first_episode_index + block.episode_count
        ):
            return rank + request.episode_index - block.first_episode_index
        rank += block.episode_count
    raise ValueError("clock parent is outside the exact independent allocation")


def _clock_parent_requests(
    allocation: "IndependentAllocation", root_seed: int
) -> Iterator[tuple[SuiteName, IndependentEpisodeRequest]]:
    """Yield exactly the declared clock children in scale-then-parent canonical order."""
    for suite, count_field in (
        (SuiteName.CLOCK_SCALE_0_1X, "scale_0_1x_episode_count"),
        (SuiteName.CLOCK_SCALE_10X, "scale_10x_episode_count"),
    ):
        expected = {
            (block.requested_path_length, block.source_first_episode_index + offset)
            for block in allocation.clock_blocks
            for offset in range(getattr(block, count_field))
        }
        yielded = 0
        for request in iter_independent_requests(allocation, root_seed):
            if (
                request.suite is SuiteName.IID_PRIMARY
                and (
                    request.requested_path_length,
                    request.episode_index,
                )
                in expected
            ):
                yielded += 1
                yield suite, request
        if yielded != len(expected):
            raise ValueError("clock parent is missing from independent allocation")


def _make_clock_pairs(
    config: Phase1Config,
    allocation: "IndependentAllocation",
    root_seed: int,
    public_id_seed: int,
    generate: Callable[..., EpisodeBundle],
) -> Iterator[PairedClockAuditPair]:
    for suite, request in _clock_parent_requests(allocation, root_seed):
        parent = generate(config, request, public_id_seed)
        _require_independent_bundle_matches_request(parent, request, public_id_seed)
        child_id = allocate_independent_public_id(
            IndependentPublicIdKey(
                generator_version="ofd-v1",
                split_namespace=request.split_namespace,
                suite=suite,
                public_id_seed=public_id_seed,
                episode_index=request.episode_index,
                accepted_attempt=parent.truth.recipe.accepted_attempt,
            )
        )
        yield PairedClockAuditPair(
            parent=AuditExample(
                bundle=parent,
                manifest_rank=_base_rank(allocation, request),
                generation_mode="independent",
                randomization_block_index=request.allocation_quartet_index,
                episode_position=request.episode_index,
            ),
            child=scale_episode_time(parent, suite, child_id),
        )


def _make_manifest_clock_pairs(
    config: Phase1Config,
    manifest: EpisodeManifest,
    regenerate: Callable[..., EpisodeBundle],
) -> Iterator[PairedClockAuditPair]:
    """Regenerate every declared manifest parent and exact child in canonical order."""
    prior_coordinate: tuple[int, int, int] | None = None
    for rank, entry in enumerate(manifest.entries):
        if entry.suite not in {SuiteName.CLOCK_SCALE_0_1X, SuiteName.CLOCK_SCALE_10X}:
            continue
        coordinate = entry.coordinate
        if not isinstance(coordinate, IndependentManifestCoordinate):
            raise ValueError("clock manifest entry has a non-independent coordinate")
        current_coordinate = (
            coordinate.episode_index,
            coordinate.allocation_quartet_index,
            coordinate.quartet_member_index,
        )
        if prior_coordinate is not None and current_coordinate <= prior_coordinate:
            raise ValueError("clock manifest entries are not in canonical order")
        prior_coordinate = current_coordinate
        parent = _regenerate_manifest_clock_parent(config, manifest, entry)
        child = regenerate(config, manifest, entry)
        expected = scale_episode_time(parent, entry.suite, entry.episode_public_id)
        if child != expected:
            raise ValueError("clock child is not the exact authenticated transform")
        _require_manifest_bundle_matches_entry(child, entry)
        yield PairedClockAuditPair(
            parent=AuditExample(
                bundle=parent,
                manifest_rank=rank,
                generation_mode="independent",
                randomization_block_index=coordinate.allocation_quartet_index,
                episode_position=coordinate.episode_index,
            ),
            child=child,
        )


def _clock_decision_matches(parent: EpisodeBundle, child: EpisodeBundle) -> bool:
    parent_solution = solve_public_episode(parent.public)
    child_solution = solve_public_episode(child.public)
    if (
        parent_solution.terminal_kind,
        parent_solution.node_path,
        parent_solution.link_record_ids,
        parent_solution.terminal_record_id,
        parent_solution.hazard_type,
    ) != (
        child_solution.terminal_kind,
        child_solution.node_path,
        child_solution.link_record_ids,
        child_solution.terminal_record_id,
        child_solution.hazard_type,
    ):
        return False
    if parent_solution.public_delay is None:
        return child_solution.public_delay is None
    if child_solution.public_delay is None:
        return False
    parent_window = action_window(
        parent.public.events[-1].timestamp,
        parent_solution.public_delay,
        parent.truth.recipe.oracle_timing,
    )
    child_window = action_window(
        child.public.events[-1].timestamp,
        child_solution.public_delay,
        child.truth.recipe.oracle_timing,
    )
    parent_ratios = tuple(
        (value - parent.public.events[-1].timestamp) / parent_solution.public_delay
        for value in (parent_window.start, parent_window.target, parent_window.end)
    )
    child_ratios = tuple(
        (value - child.public.events[-1].timestamp) / child_solution.public_delay
        for value in (child_window.start, child_window.target, child_window.end)
    )
    return all(
        abs(left - right) <= 1.0e-12
        for left, right in zip(parent_ratios, child_ratios, strict=True)
    )


def _bind_independent_audit_source(
    resolved: ResolvedConfig[Phase1Config],
    source: Phase1GateCorpusSource,
    provenance: EvidenceProvenance,
    deps: Phase1ServiceDependencies,
) -> ReiterableAuditSource:
    allocation = deps.independent_allocation
    allocation_sha = sha256_bytes(canonical_json_bytes(allocation))
    request_count = sum(block.episode_count for block in allocation.blocks)
    if sum(1 for _ in deps.iter_independent(allocation, source.root_seed)) != request_count:
        raise ValueError("independent source iterator is incomplete")

    def examples() -> Iterator[AuditExample]:
        for rank, request in enumerate(deps.iter_independent(allocation, source.root_seed)):
            bundle = deps.generate_independent(resolved.config, request, source.public_id_seed)
            _require_independent_bundle_matches_request(
                bundle,
                request,
                source.public_id_seed,
            )
            yield AuditExample(
                bundle=bundle,
                manifest_rank=rank,
                generation_mode="independent",
                randomization_block_index=request.allocation_quartet_index,
                episode_position=request.episode_index,
            )

    def clock_pairs() -> Iterator[PairedClockAuditPair]:
        yield from _make_clock_pairs(
            resolved.config,
            allocation,
            source.root_seed,
            source.public_id_seed,
            deps.generate_independent,
        )

    descriptor = AuditSourceDescriptor(
        schema_version="leakage-source-v1",
        generation_mode="independent",
        allocation_id=allocation.allocation_id,
        allocation_or_manifest_sha256=allocation_sha,
        split_namespace=allocation.split_namespace,
        root_seed=source.root_seed,
        public_id_seed_sha256=public_id_seed_sha256(source.public_id_seed),
        config_sha256=resolved.sha256,
        generator_source_sha256=provenance.generator_source.sha256,
        episode_count=request_count,
    )
    source_manifest = _SourceManifestHashBuilder()
    for example in examples():
        source_manifest.add(example, episode_sha256(example.bundle))
    clock_manifest = _ClockPairManifestHashBuilder()
    clock_counts: Counter[str] = Counter()
    for pair in clock_pairs():
        clock_manifest.add(pair)
        clock_counts[
            "scale_0_1x"
            if pair.child.truth.recipe.evaluation_suite is SuiteName.CLOCK_SCALE_0_1X
            else "scale_10x"
        ] += 1
    denominators = Counter(
        {
            f"{block.suite.value}:{block.requested_path_length}": block.episode_count
            for block in allocation.blocks
        }
    )
    authentication = AuditSourceAuthentication(
        schema_version="leakage-source-auth-v1",
        profile=(
            LeakageAuditProfileName.PHASE1_GATE
            if deps.production_mode
            else LeakageAuditProfileName.TEST
        ),
        descriptor_sha256=audit_source_descriptor_sha256(descriptor),
        source_manifest_sha256=source_manifest.finalize(),
        suite_path_denominators=dict(sorted(denominators.items())),
        clock_pair_manifest_sha256=clock_manifest.finalize(),
        clock_scale_pair_counts=dict(clock_counts),
    )
    return _BoundAuditSource(
        descriptor,
        resolved.config,
        authentication,
        examples,
        clock_pairs,
        deps.production_mode,
    )


def _bind_manifest_audit_source(
    resolved: ResolvedConfig[Phase1Config],
    manifest: EpisodeManifest,
    deps: Phase1ServiceDependencies,
) -> ReiterableAuditSource:
    """Bind a frozen manifest by its canonical bytes; never accept caller bundles."""
    _require_manifest_source_boundary(manifest, deps)
    provenance = manifest.provenance
    if provenance.generation_mode != "matched":
        raise ValueError("manifest leakage audit requires a matched base source")

    def examples() -> Iterator[AuditExample]:
        for rank, entry in enumerate(manifest.entries):
            bundle = deps.regenerate_manifest_entry(resolved.config, manifest, entry)
            _require_manifest_bundle_matches_entry(bundle, entry)
            coordinate = entry.coordinate
            if isinstance(coordinate, MatchedManifestCoordinate):
                block, position = coordinate.cohort_index, coordinate.member_index
                mode: Literal["matched", "independent"] = "matched"
            else:
                block, position = coordinate.allocation_quartet_index, coordinate.episode_index
                mode = "independent"
            yield AuditExample(bundle, rank, mode, block, position)

    def clock_pairs() -> Iterator[PairedClockAuditPair]:
        for suite in (SuiteName.CLOCK_SCALE_0_1X, SuiteName.CLOCK_SCALE_10X):
            for rank, entry in enumerate(manifest.entries):
                coordinate = entry.coordinate
                if not isinstance(coordinate, MatchedManifestCoordinate):
                    raise ValueError("matched manifest entry has an invalid coordinate")
                parent = deps.regenerate_manifest_entry(resolved.config, manifest, entry)
                _require_manifest_bundle_matches_entry(parent, entry)
                child_id = allocate_independent_public_id(
                    IndependentPublicIdKey(
                        generator_version="ofd-v1",
                        split_namespace=entry.split_namespace,
                        suite=suite,
                        public_id_seed=manifest.public_id_seed,
                        episode_index=rank,
                        accepted_attempt=entry.accepted_attempt,
                    )
                )
                yield PairedClockAuditPair(
                    AuditExample(
                        parent,
                        rank,
                        "matched",
                        coordinate.cohort_index,
                        coordinate.member_index,
                    ),
                    scale_episode_time(parent, suite, child_id),
                )

    descriptor = AuditSourceDescriptor(
        schema_version="leakage-source-v1",
        generation_mode=provenance.generation_mode,
        allocation_id=provenance.allocation_id,
        allocation_or_manifest_sha256=sha256_bytes(canonical_json_bytes(manifest)),
        split_namespace=provenance.split_namespace,
        root_seed=provenance.root_seed,
        public_id_seed_sha256=provenance.public_id_seed_sha256,
        config_sha256=resolved.sha256,
        generator_source_sha256=provenance.generator_source.sha256,
        episode_count=manifest.episode_count,
    )
    source_manifest = _SourceManifestHashBuilder()
    for example in examples():
        source_manifest.add(example, episode_sha256(example.bundle))
    clock_manifest = _ClockPairManifestHashBuilder()
    clock_counts: Counter[str] = Counter()
    for pair in clock_pairs():
        clock_manifest.add(pair)
        clock_counts[
            "scale_0_1x"
            if pair.child.truth.recipe.evaluation_suite is SuiteName.CLOCK_SCALE_0_1X
            else "scale_10x"
        ] += 1
    denominators = Counter(
        f"{entry.suite.value}:{entry.requested_path_length}" for entry in manifest.entries
    )
    authentication = AuditSourceAuthentication(
        schema_version="leakage-source-auth-v1",
        profile=LeakageAuditProfileName.TEST,
        descriptor_sha256=audit_source_descriptor_sha256(descriptor),
        source_manifest_sha256=source_manifest.finalize(),
        suite_path_denominators=dict(sorted(denominators.items())),
        clock_pair_manifest_sha256=clock_manifest.finalize(),
        clock_scale_pair_counts=dict(clock_counts),
    )
    return _BoundAuditSource(
        descriptor, resolved.config, authentication, examples, clock_pairs, deps.production_mode
    )


def _validate_manifest_boundary(
    allocation: CohortAllocation,
    provenance: EvidenceProvenance,
    root_seed: int,
    public_id_seed: int,
    access_class: ManifestAccessClass,
) -> None:
    if type(root_seed) is not int or root_seed != provenance.root_seed:
        raise ValueError("root seed does not match provenance")
    if (
        type(public_id_seed) is not int
        or public_id_seed < 0
        or provenance.public_id_seed_sha256 != public_id_seed_sha256(public_id_seed)
    ):
        raise ValueError("public ID seed does not match provenance")
    if provenance.generation_mode != "matched":
        raise ValueError("matched manifest builder requires matched provenance")
    if provenance.allocation_id != allocation.allocation_id:
        raise ValueError("allocation does not match provenance")
    if provenance.split_namespace is not allocation.split_namespace:
        raise ValueError("split namespace does not match provenance")
    if access_class is ManifestAccessClass.DEBUG:
        if (
            allocation.split_namespace is not SplitNamespace.DEBUG
            or not allocation.allocation_id.startswith("test-")
        ):
            raise ValueError("debug manifests require a test- DEBUG allocation")
    elif access_class is not ManifestAccessClass.VALIDATION:
        raise ValueError("cohort builder can create only DEBUG or VALIDATION manifests")


def build_cohort_manifest(
    config: Phase1Config,
    allocation: CohortAllocation,
    provenance: EvidenceProvenance,
    root_seed: int,
    public_id_seed: int,
    *,
    access_class: Literal[ManifestAccessClass.DEBUG, ManifestAccessClass.VALIDATION],
) -> EpisodeManifest:
    """Build only immutable recipe entries; episode payloads never leave this seam."""
    if not isinstance(config, Phase1Config) or not isinstance(allocation, CohortAllocation):
        raise TypeError("config and allocation have invalid types")
    if not isinstance(provenance, EvidenceProvenance) or not isinstance(
        access_class, ManifestAccessClass
    ):
        raise TypeError("provenance and access class have invalid types")
    # The validation guard needs the exact config; keep DEBUG structurally reusable.
    if access_class is ManifestAccessClass.VALIDATION:
        validate_validation_allocation(allocation, config)
    _validate_manifest_boundary(allocation, provenance, root_seed, public_id_seed, access_class)
    suites = {block.suite for block in allocation.blocks}
    if len(suites) != 1:
        raise ValueError("a manifest must contain exactly one suite")
    entries: list[EpisodeManifestEntry] = []
    for request in iter_cohort_requests(allocation, root_seed):
        cohort = generate_matched_cohort(config, request, public_id_seed)
        for member_index, bundle in enumerate(cohort.episodes):
            entries.append(
                EpisodeManifestEntry(
                    episode_public_id=bundle.public.init.episode_public_id,
                    split_namespace=request.split_namespace,
                    suite=request.suite,
                    coordinate=MatchedManifestCoordinate(
                        cohort_index=request.cohort_index, member_index=member_index
                    ),
                    requested_path_length=request.requested_path_length,
                    accepted_attempt=cohort.accepted_attempt,
                    episode_sha256=episode_sha256(bundle),
                )
            )
    return EpisodeManifest(
        schema_version=1,
        experiment_version=config.experiment_version,
        access_class=access_class,
        provenance=provenance,
        suite=next(iter(suites)),
        public_id_seed=public_id_seed,
        episode_count=len(entries),
        entries=tuple(entries),
    )


def build_validation_manifest(
    config: Phase1Config,
    provenance: EvidenceProvenance,
    root_seed: int,
    public_id_seed: int,
) -> EpisodeManifest:
    """Build the sole production validation allocation and reject dirty evidence."""
    if provenance.source_dirty:
        raise ValueError("production validation provenance must be clean")
    manifest = build_cohort_manifest(
        config,
        VALIDATION_ALLOCATION,
        provenance,
        root_seed,
        public_id_seed,
        access_class=ManifestAccessClass.VALIDATION,
    )
    counts = {variant: 0 for variant in EpisodeVariant}
    for entry in manifest.entries:
        bundle = regenerate_entry(config, manifest, entry)
        counts[bundle.truth.recipe.variant] += 1
    if counts != {
        EpisodeVariant.POSITIVE: 5_000,
        EpisodeVariant.SAFE_NEGATIVE: 2_500,
        EpisodeVariant.DISCONNECTED_NEGATIVE: 2_500,
    }:
        raise ValueError("frozen validation manifest has an invalid variant allocation")
    return manifest


def _assert_entry(bundle: EpisodeBundle, entry: EpisodeManifestEntry) -> EpisodeBundle:
    if (
        bundle.public.init.episode_public_id != entry.episode_public_id
        or bundle.truth.recipe.accepted_attempt != entry.accepted_attempt
        or episode_sha256(bundle) != entry.episode_sha256
    ):
        raise ValueError("regenerated episode does not match manifest entry")
    return bundle


def regenerate_entry(
    config: Phase1Config, manifest: EpisodeManifest, entry: EpisodeManifestEntry
) -> EpisodeBundle:
    """Regenerate one authenticated recipe, never inferring a quartet position."""
    if entry not in manifest.entries:
        raise ValueError("entry is not a member of the manifest")
    if entry.split_namespace is not manifest.provenance.split_namespace:
        raise ValueError("entry split namespace differs from manifest provenance")
    coordinate = entry.coordinate
    if isinstance(coordinate, MatchedManifestCoordinate):
        request = CohortRequest(
            split_namespace=entry.split_namespace,
            suite=entry.suite,
            root_seed=manifest.provenance.root_seed,
            cohort_index=coordinate.cohort_index,
            requested_path_length=entry.requested_path_length,
        )
        return _assert_entry(
            regenerate_matched_episode(
                config,
                request,
                manifest.public_id_seed,
                entry.episode_public_id,
                entry.accepted_attempt,
            ),
            entry,
        )
    if not isinstance(coordinate, IndependentManifestCoordinate):
        raise TypeError("unsupported manifest coordinate")
    request = _independent_request_for_manifest_entry(manifest, entry)
    source_suite = request.suite

    if entry.suite in {SuiteName.CLOCK_SCALE_0_1X, SuiteName.CLOCK_SCALE_10X}:
        parent = _regenerate_manifest_clock_parent(config, manifest, entry)
        return _assert_entry(
            scale_episode_time(parent, entry.suite, entry.episode_public_id), entry
        )
    regenerate = (
        regenerate_independent_episode
        if source_suite
        in {
            SuiteName.IID_PRIMARY,
            SuiteName.OOD_DEPTH,
            SuiteName.OOD_SHORT_DELAY,
            SuiteName.OOD_LONG_DELAY,
            SuiteName.DISTRACTOR_FLOOD,
        }
        else regenerate_stress_episode
    )
    return _assert_entry(
        regenerate(
            config,
            request,
            manifest.public_id_seed,
            entry.episode_public_id,
            entry.accepted_attempt,
        ),
        entry,
    )


def _independent_request_for_manifest_entry(
    manifest: EpisodeManifest, entry: EpisodeManifestEntry
) -> IndependentEpisodeRequest:
    coordinate = entry.coordinate
    if not isinstance(coordinate, IndependentManifestCoordinate):
        raise ValueError("clock manifest entry has a non-independent coordinate")
    source_suite = (
        SuiteName.IID_PRIMARY
        if entry.suite in {SuiteName.CLOCK_SCALE_0_1X, SuiteName.CLOCK_SCALE_10X}
        else entry.suite
    )
    variants = allocate_independent_variants(
        AllocationLabelKey(
            generator_version="ofd-v1",
            split_namespace=entry.split_namespace,
            suite=source_suite,
            root_seed=manifest.provenance.root_seed,
            requested_path_length=entry.requested_path_length,
            allocation_quartet_index=coordinate.allocation_quartet_index,
        )
    )
    return IndependentEpisodeRequest(
        split_namespace=entry.split_namespace,
        suite=source_suite,
        root_seed=manifest.provenance.root_seed,
        episode_index=coordinate.episode_index,
        requested_path_length=entry.requested_path_length,
        variant=variants[coordinate.quartet_member_index],
        allocation_quartet_index=coordinate.allocation_quartet_index,
    )


def _regenerate_manifest_clock_parent(
    config: Phase1Config, manifest: EpisodeManifest, entry: EpisodeManifestEntry
) -> EpisodeBundle:
    if entry.suite not in {SuiteName.CLOCK_SCALE_0_1X, SuiteName.CLOCK_SCALE_10X}:
        raise ValueError("manifest entry is not a clock child")
    if entry.parent_public_id is None or entry.parent_episode_sha256 is None:
        raise ValueError("clock manifest entry is missing its parent")
    request = _independent_request_for_manifest_entry(manifest, entry)
    parent = regenerate_independent_episode(
        config,
        request,
        manifest.public_id_seed,
        entry.parent_public_id,
        entry.accepted_attempt,
    )
    _require_independent_bundle_matches_request(parent, request, manifest.public_id_seed)
    if episode_sha256(parent) != entry.parent_episode_sha256:
        raise ValueError("clock parent hash does not match manifest entry")
    return parent


def _production_audit_source(
    resolved: ResolvedConfig[Phase1Config],
    source: CorpusSource,
    provenance: EvidenceProvenance,
    profile: LeakageAuditProfileName,
) -> ReiterableAuditSource:
    if isinstance(source, Phase1GateCorpusSource):
        bound = _bind_independent_audit_source(
            resolved, source, provenance, PRODUCTION_DEPENDENCIES
        )
    elif isinstance(source, ManifestCorpusSource):
        manifest = load_manifest(source.manifest_path)
        _require_current_provenance_authenticates_manifest(resolved, manifest, provenance)
        bound = _bind_manifest_audit_source(resolved, manifest, PRODUCTION_DEPENDENCIES)
    else:
        raise TypeError("unsupported audit source")
    if bound.authentication.profile is not profile:
        raise ValueError("audit source profile does not match request")
    return bound


def _require_current_provenance_authenticates_manifest(
    resolved: ResolvedConfig[Phase1Config],
    manifest: EpisodeManifest,
    current: EvidenceProvenance,
) -> None:
    """Compare immutable source identity without conflating it with execution provenance."""
    embedded = manifest.provenance
    if (
        embedded.config_sha256 != resolved.sha256
        or current.generation_mode != embedded.generation_mode
        or current.allocation_id != embedded.allocation_id
        or current.split_namespace is not embedded.split_namespace
        or current.root_seed != embedded.root_seed
        or current.public_id_seed_sha256 != embedded.public_id_seed_sha256
        or current.config_sha256 != embedded.config_sha256
        or current.generator_source != embedded.generator_source
        or current.foundation_model_calls != 0
    ):
        raise ValueError("current audit provenance does not authenticate immutable manifest")


def _anchor_for_bound_source(
    source: ReiterableAuditSource, profile: LeakageAuditProfileName
) -> LeakageAuditEvidenceAnchor:
    descriptor = source.descriptor
    authentication = source.authentication
    if authentication.profile is not profile:
        raise ValueError("audit anchor profile does not match authenticated source")
    return LeakageAuditEvidenceAnchor(
        schema_version="phase1-leakage-audit-anchor-v1",
        profile=profile.value,
        allocation_id=descriptor.allocation_id,
        allocation_or_manifest_sha256=descriptor.allocation_or_manifest_sha256,
        config_sha256=descriptor.config_sha256,
        descriptor_sha256=audit_source_descriptor_sha256(descriptor),
        source_manifest_sha256=authentication.source_manifest_sha256,
        suite_path_denominators=authentication.suite_path_denominators,
        clock_pair_manifest_sha256=authentication.clock_pair_manifest_sha256,
        clock_scale_pair_counts=authentication.clock_scale_pair_counts,
        episode_count=descriptor.episode_count,
    )


def _production_audit_anchor(
    resolved: ResolvedConfig[Phase1Config],
    source: CorpusSource,
    provenance: EvidenceProvenance,
    profile: LeakageAuditProfileName,
) -> LeakageAuditEvidenceAnchor:
    """Independently regenerate the immutable source authority from frozen inputs."""
    if isinstance(source, Phase1GateCorpusSource):
        bound = _bind_independent_audit_source(
            resolved, source, provenance, PRODUCTION_DEPENDENCIES
        )
    elif isinstance(source, ManifestCorpusSource):
        manifest = load_manifest(source.manifest_path)
        _require_current_provenance_authenticates_manifest(resolved, manifest, provenance)
        bound = _bind_manifest_audit_source(resolved, manifest, PRODUCTION_DEPENDENCIES)
    else:
        raise TypeError("unsupported audit source")
    return _anchor_for_bound_source(bound, profile)


def _collect_final_phase1_provenance(
    resolved: ResolvedConfig[Phase1Config],
    *,
    repo_root: Path,
    generation_mode: Literal["matched", "independent"],
    allocation_id: str,
    split_namespace: SplitNamespace,
    root_seed: int,
    public_id_seed: int,
    analysis_seeds: dict[str, int],
) -> EvidenceProvenance:
    return collect_evidence_provenance(
        resolved,
        repo_root=repo_root,
        generation_mode=generation_mode,
        allocation_id=allocation_id,
        split_namespace=split_namespace,
        root_seed=root_seed,
        public_id_seed=public_id_seed,
        analysis_seeds=analysis_seeds,
        analysis_scope="phase1_analysis",
    )


def _audit_provenance(
    resolved: ResolvedConfig[Phase1Config],
    source: CorpusSource,
    deps: Phase1ServiceDependencies,
) -> EvidenceProvenance:
    seeds = {
        "audit_seed": resolved.config.data.leakage_audit.audit_seed,
        "positive_control_seed": resolved.config.data.leakage_audit.positive_control_seed,
    }
    if isinstance(source, Phase1GateCorpusSource):
        allocation = deps.independent_allocation
        if source.allocation_id != allocation.allocation_id:
            raise ValueError("audit allocation does not match bound dependencies")
        if deps.production_mode:
            validate_phase1_gate_allocation(allocation, resolved.config)
        provenance = deps.collect_provenance(
            resolved,
            repo_root=Path.cwd(),
            generation_mode="independent",
            allocation_id=allocation.allocation_id,
            split_namespace=allocation.split_namespace,
            root_seed=source.root_seed,
            public_id_seed=source.public_id_seed,
            analysis_seeds=seeds,
        )
    elif isinstance(source, ManifestCorpusSource):
        manifest = load_manifest(source.manifest_path)
        _require_manifest_source_boundary(manifest, deps)
        embedded = manifest.provenance
        if embedded.config_sha256 != resolved.sha256:
            raise ValueError("manifest config does not match resolved audit config")
        provenance = deps.collect_provenance(
            resolved,
            repo_root=Path.cwd(),
            generation_mode=embedded.generation_mode,
            allocation_id=embedded.allocation_id,
            split_namespace=embedded.split_namespace,
            root_seed=embedded.root_seed,
            public_id_seed=manifest.public_id_seed,
            analysis_seeds=seeds,
        )
        if (
            provenance.generation_mode != embedded.generation_mode
            or provenance.allocation_id != embedded.allocation_id
            or provenance.split_namespace is not embedded.split_namespace
            or provenance.root_seed != embedded.root_seed
            or provenance.public_id_seed_sha256 != embedded.public_id_seed_sha256
            or provenance.config_sha256 != embedded.config_sha256
            or provenance.generator_source != embedded.generator_source
        ):
            raise ValueError("current audit provenance does not authenticate immutable manifest")
    else:
        raise TypeError("unsupported audit source")
    if (
        provenance.config_sha256 != resolved.sha256
        or provenance.foundation_model_calls != 0
        or provenance.analysis_seeds != seeds
        or (deps.production_mode and provenance.source_dirty)
    ):
        raise ValueError("audit provenance does not bind resolved clean inputs")
    return provenance


def run_leakage_audit(
    request: LeakageAuditRequest,
    *,
    deps: Phase1ServiceDependencies = None,  # type: ignore[assignment]
) -> LeakageAuditResult:
    """Bind a trusted source, run Task 15, and publish only its complete report."""
    if deps is None:
        deps = PRODUCTION_DEPENDENCIES
    _require_service_dependencies(deps)
    if request.output_path is not None:
        _require_test_output_boundary(request.output_path, deps)
    resolved = _resolve(request.config)
    manifest_source_sha256: str | None = None
    if isinstance(request.source, ManifestCorpusSource):
        _, manifest_source_sha256 = _bound_manifest_sha256(
            request.source.manifest_path,
            deps,
        )
    provenance = _audit_provenance(resolved, request.source, deps)
    anchor = deps.build_audit_anchor(resolved, request.source, provenance, request.profile)
    if not isinstance(anchor, LeakageAuditEvidenceAnchor):
        raise ValueError("audit anchor authority returned an invalid capability")
    source = deps.build_audit_source(resolved, request.source, provenance, request.profile)
    provenance = provenance.model_copy(update={"leakage_audit": anchor})
    with deps.create_audit_workspace() as workspace:
        report = audit_leakage(
            source,
            resolved.config.data.leakage_audit,
            request.profile,
            provenance,
            workspace,
        )
    if request.output_path is not None and isinstance(request.source, ManifestCorpusSource):
        assert manifest_source_sha256 is not None
        _bound_manifest_sha256(
            request.source.manifest_path,
            deps,
            expected_sha256=manifest_source_sha256,
        )
    publication = _publish_report(request.output_path, report) if request.output_path else None
    return LeakageAuditResult(report=report, publication=publication)


PRODUCTION_DEPENDENCIES = Phase1ServiceDependencies(
    production_mode=True,
    validation_allocation=VALIDATION_ALLOCATION,
    independent_allocation=PHASE1_GATE_ALLOCATION,
    collect_provenance=_collect_final_phase1_provenance,
    build_manifest=build_validation_manifest,
    regenerate_manifest_entry=regenerate_entry,
    iter_independent=iter_independent_requests,
    generate_independent=generate_independent_episode,
    build_audit_source=_production_audit_source,
    build_audit_anchor=_production_audit_anchor,
    create_audit_workspace=_temporary_audit_workspace,
)
