"""Verify the five immutable Phase 1 gate artifacts without regeneration."""

import argparse
import base64
import binascii
import hashlib
import sys
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.episode import EpisodeVariant
from silent_cascade.env.generator import (
    PHASE1_GATE_ALLOCATION,
    VALIDATION_ALLOCATION,
    CohortRequest,
    IndependentEpisodeRequest,
    independent_seed_tokens,
    iter_independent_requests,
    matched_seed_tokens,
)
from silent_cascade.env.leakage import (
    NAMED_LEAK_INJECTORS,
    PC_BASE_SUBSET_MEMBERSHIP_DOMAIN,
    AuditSourceDescriptor,
    CounterfactualCheckId,
    CounterfactualCheckResult,
    LeakageAuditProfileName,
    LeakageConstructionStatistics,
    LeakageMembershipEvidence,
    LeakageReport,
    OrderedSha256Pack,
    PositiveControlResult,
    ShortcutFeatureGroup,
    ShortcutProbeResult,
    ShortcutTask,
    audit_source_descriptor_sha256,
)
from silent_cascade.env.reproducibility import (
    IndependentSourceDescriptor,
    ReproducibilityReport,
    independent_sample_membership_sha256,
    manifest_sample_membership_sha256,
    select_independent_reproducibility_sample,
    select_manifest_reproducibility_sample,
)
from silent_cascade.env.services import OracleEvaluationReport
from silent_cascade.errors import (
    ArtifactIntegrityError,
    ManifestError,
    ProvenanceError,
    SilentCascadeError,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.manifest import (
    EpisodeManifest,
    ManifestAccessClass,
    MatchedManifestCoordinate,
    load_manifest,
)
from silent_cascade.provenance import (
    AcceptedAttemptRun,
    ConstructionNamespaceBuilder,
    ConstructionNamespaceEvidence,
    EvidenceProvenance,
    authenticate_final_phase1_provenance,
    require_namespace_evidence_provenance,
)
from silent_cascade.rng import (
    IndependentPublicIdKey,
    PublicIdBatchKey,
    allocate_independent_public_id,
    allocate_public_ids,
)
from silent_cascade.validation import StrictModel

_VALIDATION_EPISODE_COUNT = 10_000
_INDEPENDENT_EPISODE_COUNT = 100_000
_INDEPENDENT_VARIANT_COUNTS = (50_000, 25_000, 25_000)
_CLOCK_COUNTS = (5_000, 2_000)
_COUNTERFACTUAL_COUNTS = {
    CounterfactualCheckId.TERMINAL_DELAY_SWAP: 25_000,
    CounterfactualCheckId.PRESENTATION_PERMUTATION: 100_000,
    CounterfactualCheckId.PAIRED_CLOCK_SCALE: 7_000,
}
_GATE_DENOMINATORS = {
    "distractor_flood:2": 8_000,
    "distractor_flood:3": 8_000,
    "distractor_flood:4": 8_000,
    "iid_primary:2": 8_000,
    "iid_primary:3": 8_000,
    "iid_primary:4": 8_000,
    "ood_depth:5": 4_000,
    "ood_depth:6": 4_000,
    "ood_depth:7": 4_000,
    "ood_depth:8": 4_000,
    "ood_long_delay:2": 4_000,
    "ood_long_delay:3": 4_000,
    "ood_long_delay:4": 4_000,
    "ood_short_delay:2": 8_000,
    "ood_short_delay:3": 8_000,
    "ood_short_delay:4": 8_000,
}
_REPRODUCIBILITY_MODES = ("forward", "reverse", "chunked", "fresh_process")
_REPRODUCIBILITY_SAMPLE_SIZE = 1_000
_REPRODUCIBILITY_CHUNK_SIZES = (1, 3, 7)
_REPRODUCIBILITY_PYTHON_HASH_SEEDS = (0, 1)
_VALIDATION_EXPERIMENT_VERSION = "v1"
_VALIDATION_COHORT_BLOCKS = ((0, 834, 2), (834, 1_667, 3), (1_667, 2_500, 4))
_PHASE1_GATE_ALLOCATION_SHA256 = "9e032f6993af9f2d53c1dde3a3e3e72e097cf143ae2e72d02609f2d2d6f1ce18"
_CLOCK_PAIR_COUNTS = {"scale_0_1x": 5_000, "scale_10x": 2_000}
_VALIDATION_ROOT_SEED = 2026083001
_VALIDATION_PUBLIC_ID_SEED = 2026083002
_VALIDATION_PUBLIC_ID_SEED_SHA256 = (
    "0454fca622eb08379a5d88ecbe0a5ef70f6a15e9ca7d333acecf840f2df38802"
)
_INDEPENDENT_ROOT_SEED = 2026083011
_INDEPENDENT_PUBLIC_ID_SEED = 2026083012
_INDEPENDENT_PUBLIC_ID_SEED_SHA256 = (
    "f21ac562825bfd96e875eedf90c8ba5ed09883ebe2c18449843b03acebec778a"
)
_PRIMARY_CONFIG_SHA256 = "8eede957c7d69cc85d33bddcd1af69eb76bc5d7bba48bd1cbbe166bbb757ecd4"
_PHASE1_PERMUTATION_DENOMINATOR = 5_000
_MAX_LEAKAGE_REPORT_BYTES = 16 * 1024 * 1024
_PHASE1_FEATURE_DIMENSIONS = {
    ShortcutTask.POSITIVE_BINARY: (17, 256, 278, 4, 768, 194, 10, 33, 1_560),
    ShortcutTask.VARIANT_THREE_WAY: (17, 256, 278, 4, 768, 194, 10, 33, 1_560),
    ShortcutTask.POSITIVE_HAZARD_CLASS: (17, 256, 270, 4, 704, 194, 4, 33, 1_482),
}
_CONSTRUCTION_CHECK_IDS = (
    "provenance",
    "public_ids",
    "seed_tokens",
    "invariants",
    "finite_features",
    "two_pass_identity",
)
_CONSTRUCTION_CHECKS = {key: True for key in sorted(_CONSTRUCTION_CHECK_IDS)}
_FEATURE_SCHEMA_SHA256 = sha256_bytes(
    canonical_json_bytes(
        {
            "schema_version": "leakage-features-v1",
            "dimensions": (17, 256, 278, 4, 768, 194, 10, 33),
            "groups": [group.value for group in ShortcutFeatureGroup],
        }
    )
)

type HexDigest = str


class Phase1GateVerificationResult(StrictModel):
    """Canonical summary proving the five files agree at their public boundaries."""

    schema_version: Literal["phase1-gate-verification-v3"]
    validation_episode_count: Literal[10_000]
    independent_episode_count: Literal[100_000]
    matched_accepted_draw_count: Literal[2_500]
    independent_accepted_draw_count: Literal[100_000]
    matched_public_id_seed: Literal[2026083002]
    independent_public_id_seed: Literal[2026083012]
    construction_token_count: Literal[750_000]
    public_id_count: Literal[117_000]
    validation_manifest_payload_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    independent_corpus_sha256: HexDigest = Field(pattern=r"^[0-9a-f]{64}$")
    foundation_model_calls: Literal[0]
    passed: Literal[True]


def _artifact_error(message: str, *, path: Path | None = None) -> ArtifactIntegrityError:
    context = {} if path is None else {"path": str(path)}
    return ArtifactIntegrityError(message, context=context)


def _load_report[ReportT: StrictModel](path: Path, model: type[ReportT], *, name: str) -> ReportT:
    try:
        if model is LeakageReport and path.stat().st_size > _MAX_LEAKAGE_REPORT_BYTES:
            raise ValueError("canonical leakage report exceeds 16 MiB")
        raw = path.read_bytes()
        if model is LeakageReport and len(raw) > _MAX_LEAKAGE_REPORT_BYTES:
            raise ValueError("canonical leakage report exceeds 16 MiB")
        report = model.model_validate_json(raw)
    except (OSError, ValidationError, TypeError, ValueError) as error:
        raise _artifact_error(f"{name} artifact schema verification failed", path=path) from error
    if raw != canonical_json_bytes(report):
        raise _artifact_error(f"{name} artifact is not canonical", path=path)
    return report


def _load_validation(path: Path) -> EpisodeManifest:
    try:
        return load_manifest(path)
    except ManifestError as error:
        raise _artifact_error("validation artifact verification failed", path=path) from error


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise _artifact_error(message)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _holm_adjusted_p_values(raw_p_values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = sorted(enumerate(raw_p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(raw_p_values)
    running = 0.0
    for rank, (index, raw_p) in enumerate(ordered):
        running = max(running, min(1.0, (len(raw_p_values) - rank) * raw_p))
        adjusted[index] = running
    return tuple(adjusted)


def _probe_sufficient_statistics_are_consistent(probe: ShortcutProbeResult) -> bool:
    """Independently derive arithmetic; no producer statistics helper is trusted."""
    try:
        counts = probe.test_class_counts
        matrix = probe.test_confusion_counts
        if (
            type(counts) is not dict
            or not counts
            or type(matrix) is not dict
            or set(matrix) != set(counts)
            or any(
                type(label) is not str or type(count) is not int or count <= 0
                for label, count in counts.items()
            )
        ):
            return False
        correct = 0
        # Match the chosen Python 3.12 compensated sum in canonical label order,
        # while deriving the recalls independently from the supplied counts.
        recalls: list[float] = []
        for label in sorted(counts):
            row = matrix[label]
            if (
                type(row) is not dict
                or set(row) != set(counts)
                or any(
                    type(key) is not str or type(value) is not int or value < 0
                    for key, value in row.items()
                )
                or sum(row.values()) != counts[label]
            ):
                return False
            correct += row[label]
            recalls.append(row[label] / counts[label])
        return (
            type(probe.permutation_exceedance_count) is int
            and type(probe.permutation_replicate_count) is int
            and 0 <= probe.permutation_exceedance_count <= probe.permutation_replicate_count
            and probe.permutation_replicate_count > 0
            and sum(counts.values()) == probe.test_examples
            and probe.raw_accuracy == correct / sum(counts.values())
            and probe.balanced_accuracy == sum(recalls) / len(counts)
            and probe.raw_permutation_p
            == (probe.permutation_exceedance_count + 1) / (probe.permutation_replicate_count + 1)
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _base_subset_membership_sha256(source_public_ids, selected) -> str:
    """Commit the exact source-index/public-ID subsequence with explicit lengths."""
    if not source_public_ids or not selected:
        raise ValueError("subset source and members must be nonempty")
    members = []
    prior = -1
    for index in selected:
        if type(index) is not int or index <= prior or index >= len(source_public_ids):
            raise ValueError("subset indices must increase within the source")
        public_id = source_public_ids[index]
        if type(public_id) is not str or str(uuid.UUID(public_id)) != public_id:
            raise ValueError("subset source ID is not a canonical UUID")
        members.append({"episode_public_id": public_id, "source_index": index})
        prior = index
    return _base_subset_membership_payload_sha256(
        {
            "domain": PC_BASE_SUBSET_MEMBERSHIP_DOMAIN,
            "item_count": len(members),
            "members": members,
            "source_episode_count": len(source_public_ids),
        },
        source_public_ids,
    )


def _base_subset_membership_payload_sha256(payload, source_public_ids) -> str:
    required = {"domain", "item_count", "members", "source_episode_count"}
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("subset payload keys are invalid")
    if (
        payload["domain"] != PC_BASE_SUBSET_MEMBERSHIP_DOMAIN
        or type(payload["source_episode_count"]) is not int
        or payload["source_episode_count"] != len(source_public_ids)
        or not 1 <= len(source_public_ids) <= 100_000
        or type(payload["item_count"]) is not int
        or type(payload["members"]) is not list
        or payload["item_count"] != len(payload["members"])
        or not 1 <= payload["item_count"] <= len(source_public_ids)
    ):
        raise ValueError("subset payload domain or counts are invalid")
    last_index = -1
    for item in payload["members"]:
        if type(item) is not dict or set(item) != {"episode_public_id", "source_index"}:
            raise ValueError("subset member keys are invalid")
        index = item["source_index"]
        if (
            type(index) is not int
            or index <= last_index
            or index >= len(source_public_ids)
            or type(item["episode_public_id"]) is not str
            or item["episode_public_id"] != source_public_ids[index]
        ):
            raise ValueError("subset source order or public ID differs")
        last_index = index
    return sha256_bytes(canonical_json_bytes(payload))


def _decode_leakage_pack(pack: OrderedSha256Pack, count: int) -> tuple[str, ...]:
    """Recheck bytes independently, including model_construct/model_copy callers."""
    _require(
        type(pack) is OrderedSha256Pack
        and type(pack.item_count) is int
        and pack.item_count == count
        and pack.schema_version == "ordered-sha256-pack-v1"
        and pack.encoding == "base64-concatenated-sha256-v1"
        and type(pack.payload_base64) is str
        and len(pack.payload_base64) == 4 * ((32 * count + 2) // 3)
        and _is_sha256(pack.payload_sha256),
        "leakage digest pack shape or count differs",
    )
    try:
        raw = base64.b64decode(pack.payload_base64, validate=True)
    except (ValueError, binascii.Error) as error:
        raise _artifact_error("leakage digest pack Base64 differs") from error
    _require(
        len(raw) == 32 * count
        and base64.b64encode(raw).decode("ascii") == pack.payload_base64
        and hashlib.sha256(raw).hexdigest() == pack.payload_sha256,
        "leakage digest pack payload hash differs",
    )
    digests = tuple(raw[offset : offset + 32].hex() for offset in range(0, len(raw), 32))
    _require(len(set(digests)) == count, "leakage digest pack repeats an item")
    return digests


def _leakage_corpus_sha256(public_ids: Sequence[str], digests: Sequence[str]) -> str:
    """Task 2 ordered binary framing, independent of the producer corpus builder."""
    _require(
        len(public_ids) == len(digests) > 0 and len(set(public_ids)) == len(public_ids),
        "leakage corpus IDs or digest count differs",
    )
    digest = hashlib.sha256(b"silent-cascade/corpus/v1\0")
    digest.update(len(public_ids).to_bytes(8, "big"))
    for rank, (public_id, episode_digest) in enumerate(zip(public_ids, digests, strict=True)):
        encoded = public_id.encode("utf-8")
        digest.update(rank.to_bytes(8, "big"))
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(episode_digest))
    return digest.hexdigest()


def _leakage_source_manifest_sha256(requests, public_ids, digests) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "silent-cascade/ofd-v1/leakage-source-manifest/v2",
                "episodes": [
                    {
                        "manifest_rank": rank,
                        "generation_mode": "independent",
                        "randomization_block_index": request.allocation_quartet_index,
                        "episode_position": request.episode_index,
                        "quartet_member_index": request.quartet_member_index,
                        "public_id": public_id,
                        "episode_sha256": digest,
                    }
                    for rank, (request, public_id, digest) in enumerate(
                        zip(requests, public_ids, digests, strict=True)
                    )
                ],
            }
        )
    )


def _leakage_clock_manifest_sha256(requests, public_ids, digests, parents, child_digests) -> str:
    order = tuple((0 if scale == 0.1 else 1, rank) for rank, scale, _ in parents)
    _require(
        len(order) == len(set(order)) and order == tuple(sorted(order)),
        "leakage clock parent order differs",
    )
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "silent-cascade/ofd-v1/leakage-clock-pair-manifest/v2",
                "pairs": [
                    {
                        "parent_manifest_rank": rank,
                        "parent_quartet_member_index": requests[rank].quartet_member_index,
                        "parent_public_id": public_ids[rank],
                        "parent_episode_sha256": digests[rank],
                        "scale": scale,
                        "child_public_id": child_id,
                        "child_episode_sha256": child_digest,
                    }
                    for (rank, scale, child_id), child_digest in zip(
                        parents, child_digests, strict=True
                    )
                ],
            }
        )
    )


def _require_complete_membership(train, test, count: int) -> None:
    _require(
        len(train) + len(test) == count
        and all(type(index) is int for index in (*train, *test))
        and len(set(train)) == len(train)
        and len(set(test)) == len(test)
        and set(train).isdisjoint(test)
        and set(train).union(test) == set(range(count)),
        "leakage train/test membership is incomplete, overlapping, or non-covering",
    )


@dataclass(frozen=True)
class _LeakageMembership:
    train: tuple[int, ...]
    test: tuple[int, ...]
    sha256: str


def _leakage_groups(requests):
    groups = defaultdict(lambda: defaultdict(list))
    for index, request in enumerate(requests):
        groups[(request.suite.value, request.requested_path_length)][
            f"independent:{request.allocation_quartet_index}"
        ].append(index)
    for stratum in groups.values():
        for indices in stratum.values():
            _require(
                len(indices) == 4
                and tuple(requests[index].quartet_member_index for index in indices) == (0, 1, 2, 3)
                and Counter(requests[index].variant for index in indices)
                == {
                    EpisodeVariant.POSITIVE: 2,
                    EpisodeVariant.SAFE_NEGATIVE: 1,
                    EpisodeVariant.DISCONNECTED_NEGATIVE: 1,
                },
                "leakage group has incomplete explicit quartet members or labels",
            )
    return groups


def _leakage_membership(requests, corpus_hash: str, *, control: bool = False) -> _LeakageMembership:
    train, test, records = [], [], []
    for (suite, path), groups in sorted(_leakage_groups(requests).items()):
        _require(len(groups) >= 5 and len(groups) % 5 == 0, "leakage stratum cannot split 80/20")
        keys = {}
        for group in groups:
            payload = (
                {
                    "domain": "silent-cascade/ofd-v1/pc-split/v1",
                    "positive_control_seed": 2026083092,
                    "clean_subset_corpus_hash": corpus_hash,
                    "suite": suite,
                    "requested_path_length": path,
                    "group_id": group,
                }
                if control
                else {
                    "schema_version": "leakage-v1",
                    "audit_seed": 2026083091,
                    "corpus_hash": corpus_hash,
                    "generation_mode": "independent",
                    "suite": suite,
                    "requested_path_length": path,
                    "randomization_block_index": requests[
                        groups[group][0]
                    ].allocation_quartet_index,
                }
            )
            keys[group] = sha256_bytes(canonical_json_bytes(payload))
        ranked = sorted(groups, key=keys.__getitem__)
        for rank, group in enumerate(ranked):
            is_train = rank < len(ranked) * 4 // 5
            (train if is_train else test).extend(groups[group])
            records.append(
                {
                    "suite": suite,
                    "path": path,
                    "group": group,
                    "split": "train" if is_train else "test",
                }
            )
    _require_complete_membership(train, test, len(requests))
    return _LeakageMembership(
        tuple(train),
        tuple(test),
        sha256_bytes(
            canonical_json_bytes(
                {
                    "domain": (
                        "silent-cascade/ofd-v1/pc-split-membership/v1"
                        if control
                        else "silent-cascade/ofd-v1/leakage-split/v1"
                    ),
                    "memberships": records,
                }
            )
        ),
    )


def _leakage_control_subset(requests, corpus_hash: str) -> tuple[int, ...]:
    strata = _leakage_groups(requests)
    quota, remainder = divmod(8_000, 4 * len(strata))
    _require(quota > 0 and remainder == 0, "leakage control subset cannot balance strata")
    selected = []
    for (suite, path), groups in sorted(strata.items()):
        ranked = sorted(
            groups,
            key=lambda group: sha256_bytes(
                canonical_json_bytes(
                    {
                        "domain": "silent-cascade/ofd-v1/pc-subset/v1",
                        "audit_seed": 2026083091,
                        "corpus_hash": corpus_hash,
                        "suite": suite,
                        "requested_path_length": path,
                        "group_id": group,
                    }
                )
            ),
        )
        _require(len(ranked) >= quota, "leakage control subset stratum is too small")
        for group in ranked[:quota]:
            selected.extend(groups[group])
    _require(len(selected) == 8_000, "leakage control subset count differs")
    return tuple(sorted(selected))


def _leakage_counterfactual_sha256(check: CounterfactualCheckId, pairs: Iterable) -> str:
    """Hash the exact all-false stream only after the caller rejects nonzero counts."""
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "silent-cascade/ofd-v1/counterfactual-check/v1",
                "check_id": check.value,
                "pairs": [
                    {
                        "pair_key_sha256": sha256_bytes(
                            canonical_json_bytes(
                                {
                                    "domain": "silent-cascade/ofd-v1/counterfactual-pair/v1",
                                    "check_id": check.value,
                                    "source_public_ids": list(ids),
                                    "transform": tag,
                                }
                            )
                        ),
                        "decision_mismatch": False,
                        "temporal_mismatch": False,
                    }
                    for ids, tag in pairs
                ],
            }
        )
    )


@lru_cache(maxsize=2)
def _leakage_coordinates(attempts: tuple[int, ...]):
    """Retain only immutable frozen requests, IDs, and authenticated clock ranks."""
    requests = tuple(iter_independent_requests(PHASE1_GATE_ALLOCATION, _INDEPENDENT_ROOT_SEED))
    _require(len(requests) == len(attempts) == 100_000, "leakage source draw count differs")
    public_ids = tuple(
        allocate_independent_public_id(
            IndependentPublicIdKey(
                generator_version="ofd-v1",
                split_namespace=request.split_namespace,
                suite=request.suite,
                public_id_seed=_INDEPENDENT_PUBLIC_ID_SEED,
                episode_index=request.episode_index,
                accepted_attempt=attempt,
            )
        )
        for request, attempt in zip(requests, attempts, strict=True)
    )
    ranks = {
        (request.suite, request.requested_path_length, request.episode_index): rank
        for rank, request in enumerate(requests)
    }
    clock_ids = iter(_independent_clock_public_ids(requests, attempts))
    parents = tuple(
        (
            ranks[
                (
                    SuiteName.IID_PRIMARY,
                    block.requested_path_length,
                    block.source_first_episode_index + offset,
                )
            ],
            scale,
            next(clock_ids),
        )
        for scale, field in ((0.1, "scale_0_1x_episode_count"), (10.0, "scale_10x_episode_count"))
        for block in PHASE1_GATE_ALLOCATION.clock_blocks
        for offset in range(getattr(block, field))
    )
    _require(
        len(set(public_ids).union(child_id for _, _, child_id in parents)) == 107_000,
        "leakage base or clock public IDs collide",
    )
    return requests, public_ids, parents


@dataclass(frozen=True)
class _LeakageBaseEvidence:
    corpus: str
    manifest: str
    membership: _LeakageMembership
    subset: tuple[int, ...]
    subset_corpus: str
    subset_membership: str
    control_membership: _LeakageMembership
    counterfactuals: tuple[str, ...]


@lru_cache(maxsize=2)
def _leakage_base_evidence(
    attempts: tuple[int, ...], digests: tuple[str, ...]
) -> _LeakageBaseEvidence:
    # Cache keys contain the complete immutable primitive input, never a report
    # commitment or pass flag. Cached results contain no mutable report objects.
    requests, public_ids, parents = _leakage_coordinates(attempts)
    corpus = _leakage_corpus_sha256(public_ids, digests)
    membership = _leakage_membership(requests, corpus)
    subset = _leakage_control_subset(requests, corpus)
    subset_ids = tuple(public_ids[index] for index in subset)
    subset_corpus = _leakage_corpus_sha256(subset_ids, tuple(digests[index] for index in subset))
    control_membership = _leakage_membership(
        tuple(requests[index] for index in subset), subset_corpus, control=True
    )
    pending, delay_pairs = {}, []
    for request, public_id in zip(requests, public_ids, strict=True):
        if request.variant is EpisodeVariant.POSITIVE:
            key = (request.suite, request.requested_path_length)
            if key in pending:
                delay_pairs.append(((pending.pop(key), public_id), "swap_terminal_delay"))
            else:
                pending[key] = public_id
    _require(not pending and len(delay_pairs) == 25_000, "leakage delay pairing is incomplete")
    counterfactuals = (
        _leakage_counterfactual_sha256(CounterfactualCheckId.TERMINAL_DELAY_SWAP, delay_pairs),
        _leakage_counterfactual_sha256(
            CounterfactualCheckId.PRESENTATION_PERMUTATION,
            (((public_id,), "permute_presentation") for public_id in public_ids),
        ),
        _leakage_counterfactual_sha256(
            CounterfactualCheckId.PAIRED_CLOCK_SCALE,
            (
                ((public_ids[rank],), "scale_0_1x" if scale == 0.1 else "scale_10x")
                for rank, scale, _ in parents
            ),
        ),
    )
    return _LeakageBaseEvidence(
        corpus,
        _leakage_source_manifest_sha256(requests, public_ids, digests),
        membership,
        subset,
        subset_corpus,
        _base_subset_membership_sha256(public_ids, subset),
        control_membership,
        counterfactuals,
    )


def _require_leakage_workloads(probes, requests, membership, *, hazard_control: bool = False):
    for probe in probes:
        train, test = membership.train, membership.test
        if probe.task is ShortcutTask.POSITIVE_HAZARD_CLASS:
            train = tuple(
                index for index in train if requests[index].variant is EpisodeVariant.POSITIVE
            )
            test = tuple(
                index for index in test if requests[index].variant is EpisodeVariant.POSITIVE
            )
        _require(
            probe.train_examples == len(train) and probe.test_examples == len(test),
            "leakage task-filtered probe workload differs from reconstructed membership",
        )
        for indices, reported in (
            (train, probe.train_class_counts),
            (test, probe.test_class_counts),
        ):
            if probe.task is ShortcutTask.POSITIVE_HAZARD_CLASS:
                # Main hazard identities and conditional shuffled identities are
                # nuisance evidence, not allocation labels. Their exact totals
                # and confusion primitives are checked; the control explicitly
                # assigns four balanced target classes within each membership.
                expected = (
                    Counter(str(rank % 4) for rank in range(len(indices)))
                    if hazard_control
                    else None
                )
            else:
                expected = Counter(
                    str(int(requests[index].variant is EpisodeVariant.POSITIVE))
                    if probe.task is ShortcutTask.POSITIVE_BINARY
                    else str(tuple(EpisodeVariant).index(requests[index].variant))
                    for index in indices
                )
            _require(
                expected is None or reported == expected,
                "leakage task class counts differ from allocation-derived labels",
            )


def _require_rederived_leakage(leakage: LeakageReport) -> None:
    statistics = leakage.construction_statistics
    digests = _decode_leakage_pack(statistics.source_episode_sha256s, 100_000)
    children = _decode_leakage_pack(statistics.clock_child_episode_sha256s, 7_000)
    attempts = _accepted_attempts(leakage.namespace_evidence)
    requests, public_ids, parents = _leakage_coordinates(attempts)
    base = _leakage_base_evidence(attempts, digests)
    anchor = leakage.provenance.leakage_audit
    assert anchor is not None
    _require(base.corpus == leakage.corpus_hash, "leakage corpus hash is not independently derived")
    _require(
        base.manifest
        == anchor.source_manifest_sha256
        == statistics.first_pass_source_manifest_sha256
        == statistics.second_pass_source_manifest_sha256,
        "leakage source manifest is not independently derived",
    )
    _require(
        _leakage_clock_manifest_sha256(requests, public_ids, digests, parents, children)
        == anchor.clock_pair_manifest_sha256,
        "leakage clock manifest is not independently derived",
    )
    evidence = leakage.membership_evidence
    _require(
        base.membership.sha256 == evidence.split_membership_sha256 == leakage.split_membership_hash
        and len(base.membership.train) == evidence.train_episode_count == 80_000
        and len(base.membership.test) == evidence.test_episode_count == 20_000,
        "leakage split hash or membership counts are not independently derived",
    )
    for indices, reported, outer in (
        (base.membership.train, evidence.train_membership_sha256, leakage.train_membership_hash),
        (base.membership.test, evidence.test_membership_sha256, leakage.test_membership_hash),
    ):
        _require(
            sha256_bytes(
                canonical_json_bytes(
                    {
                        "domain": "silent-cascade/ofd-v1/leakage-members/v1",
                        "public_ids": [public_ids[index] for index in indices],
                    }
                )
            )
            == reported
            == outer,
            "leakage ordered member hash is not independently derived",
        )
    _require_leakage_workloads(leakage.probes, requests, base.membership)
    _require_leakage_workloads(leakage.label_shuffled_probes, requests, base.membership)
    subset_requests = tuple(requests[index] for index in base.subset)
    subset_ids = tuple(public_ids[index] for index in base.subset)
    for control in leakage.positive_controls:
        _require(
            control.base_subset_membership_sha256 == base.subset_membership
            and control.base_subset_corpus_sha256 == base.subset_corpus
            and control.split_membership_sha256 == base.control_membership.sha256,
            "leakage positive control subset or split is not independently derived",
        )
        injected_ids = subset_ids
        if control.control_id == "PC_PUBLIC_ID_BY_LABEL":
            ids = []
            for position, request in enumerate(subset_requests):
                raw = bytearray(
                    hashlib.sha256(
                        canonical_json_bytes(
                            {
                                "domain": "silent-cascade/ofd-v1/pc-public-id/v1",
                                "position": position,
                            }
                        )
                    ).digest()[:16]
                )
                raw[0] = 0 if request.variant is EpisodeVariant.POSITIVE else 255
                raw[6] = (raw[6] & 0x0F) | 0x40
                raw[8] = (raw[8] & 0x3F) | 0x80
                ids.append(str(uuid.UUID(bytes=bytes(raw))))
            injected_ids = tuple(ids)
        _require(
            _leakage_corpus_sha256(
                injected_ids, _decode_leakage_pack(control.injected_episode_sha256s, 8_000)
            )
            == control.injected_corpus_sha256,
            "leakage injected corpus hash is not independently derived",
        )
        _require_leakage_workloads(
            control.probes,
            subset_requests,
            base.control_membership,
            hazard_control=control.control_id == "PC_HAZARD_LAYOUT_BY_CLASS",
        )
    namespace = leakage.namespace_evidence
    checks = dict(
        sorted(
            {
                "provenance": base.manifest == anchor.source_manifest_sha256
                and anchor.episode_count == len(requests),
                "public_ids": namespace.public_id_collision_count == 0
                and namespace.base_public_id_count == len(public_ids)
                and namespace.clock_public_id_count == len(parents)
                and namespace.total_public_id_count == len(public_ids) + len(parents),
                "seed_tokens": namespace.seed_token_collision_count == 0
                and namespace.seed_token_count == 700_000,
                "invariants": statistics.invariant_verified_count == len(requests),
                "finite_features": statistics.feature_row_count
                == statistics.finite_feature_row_count
                == len(requests),
                "two_pass_identity": statistics.second_pass_verified_count
                == statistics.second_pass_match_count
                == len(requests)
                and statistics.first_pass_source_manifest_sha256
                == statistics.second_pass_source_manifest_sha256
                == base.manifest,
            }.items()
        )
    )
    _require(
        leakage.construction_checks == checks and all(checks.values()),
        "leakage construction checks are not independently derived",
    )
    _require(
        all(
            result.result_payload_sha256 == expected
            for result, expected in zip(
                leakage.counterfactual_checks, base.counterfactuals, strict=True
            )
        ),
        "leakage counterfactual result hash is not independently derived",
    )


def _probe_primitive_types_are_exact(probe: ShortcutProbeResult) -> bool:
    return (
        type(probe.task) is ShortcutTask
        and type(probe.feature_group) is ShortcutFeatureGroup
        and type(probe.feature_dimension) is int
        and type(probe.train_examples) is int
        and type(probe.test_examples) is int
        and type(probe.train_class_counts) is dict
        and type(probe.test_class_counts) is dict
        and _probe_sufficient_statistics_are_consistent(probe)
        and all(
            type(key) is str and type(value) is int
            for counts in (probe.train_class_counts, probe.test_class_counts)
            for key, value in counts.items()
        )
        and type(probe.raw_accuracy) is float
        and type(probe.balanced_accuracy) is float
        and type(probe.balanced_chance) is float
        and type(probe.raw_permutation_p) is float
        and type(probe.holm_adjusted_p) is float
        and type(probe.optimizer_iterations) is int
        and type(probe.optimizer_converged) is bool
        and type(probe.passed) is bool
    )


def _positive_control_primitive_types_are_exact(control: PositiveControlResult) -> bool:
    return (
        type(control.control_id) is str
        and type(control.target_task) is ShortcutTask
        and type(control.expected_detector_id) is str
        and type(control.observed_detector_ids) is tuple
        and all(type(value) is str for value in control.observed_detector_ids)
        and _is_sha256(control.base_subset_corpus_sha256)
        and _is_sha256(control.injected_corpus_sha256)
        and _is_sha256(control.split_membership_sha256)
        and type(control.balanced_accuracy) is float
        and type(control.holm_adjusted_p) is float
        and type(control.probes) is tuple
        and type(control.passed) is bool
    )


def _namespace_evidence_primitive_types_are_exact(
    evidence: ConstructionNamespaceEvidence,
) -> bool:
    integer_fields = (
        evidence.public_id_seed,
        evidence.accepted_draw_count,
        evidence.rejected_draw_count,
        evidence.generation_attempt_count,
        evidence.seed_token_count,
        evidence.seed_token_collision_count,
        evidence.base_public_id_count,
        evidence.clock_public_id_count,
        evidence.total_public_id_count,
        evidence.public_id_collision_count,
    )
    return (
        type(evidence) is ConstructionNamespaceEvidence
        and type(evidence.schema_version) is str
        and evidence.schema_version == "construction-namespace-evidence-v1"
        and type(evidence.generation_mode) is str
        and all(type(value) is int for value in integer_fields)
        and type(evidence.accepted_attempt_runs) is tuple
        and all(
            type(run) is AcceptedAttemptRun
            and type(run.first_draw_index) is int
            and type(run.draw_count) is int
            and type(run.accepted_attempt) is int
            for run in evidence.accepted_attempt_runs
        )
        and _is_sha256(evidence.seed_token_sequence_sha256)
        and _is_sha256(evidence.public_id_sequence_sha256)
    )


def _require_namespace_evidence(
    evidence: ConstructionNamespaceEvidence,
    provenance: EvidenceProvenance,
    *,
    generation_mode: Literal["matched", "independent"],
    public_id_seed: int,
    accepted_draw_count: int,
    seed_token_count: int,
    base_public_id_count: int,
    clock_public_id_count: int,
) -> None:
    try:
        require_namespace_evidence_provenance(evidence, provenance)
    except (TypeError, ValueError) as error:
        raise _artifact_error("construction namespace evidence contradicts provenance") from error
    _require(
        _namespace_evidence_primitive_types_are_exact(evidence)
        and evidence.generation_mode == generation_mode
        and evidence.public_id_seed == public_id_seed
        and evidence.accepted_draw_count == accepted_draw_count
        and evidence.seed_token_count == seed_token_count
        and evidence.base_public_id_count == base_public_id_count
        and evidence.clock_public_id_count == clock_public_id_count
        and evidence.total_public_id_count == base_public_id_count + clock_public_id_count,
        "construction namespace evidence has a wrong frozen seed or denominator",
    )


def _leakage_report_primitive_types_are_exact(leakage: LeakageReport) -> bool:
    digest_fields = (
        leakage.corpus_hash,
        leakage.feature_schema_hash,
        leakage.split_membership_hash,
        leakage.train_membership_hash,
        leakage.test_membership_hash,
    )
    return (
        type(leakage) is LeakageReport
        and type(leakage.schema_version) is str
        and leakage.schema_version == "leakage-report-v3"
        and type(leakage.provenance) is EvidenceProvenance
        and _namespace_evidence_primitive_types_are_exact(leakage.namespace_evidence)
        and type(leakage.generation_mode) is str
        and type(leakage.profile) is LeakageAuditProfileName
        and all(_is_sha256(value) for value in digest_fields)
        and type(leakage.episode_count) is int
        and type(leakage.randomization_block_count) is int
        and type(leakage.suite_path_denominators) is dict
        and all(
            type(key) is str and type(value) is int
            for key, value in leakage.suite_path_denominators.items()
        )
        and type(leakage.construction_check_ids) is tuple
        and all(type(value) is str for value in leakage.construction_check_ids)
        and type(leakage.construction_checks) is dict
        and all(
            type(key) is str and type(value) is bool
            for key, value in leakage.construction_checks.items()
        )
        and type(leakage.probes) is tuple
        and type(leakage.label_shuffled_probes) is tuple
        and type(leakage.positive_controls) is tuple
        and type(leakage.counterfactual_checks) is tuple
        and type(leakage.label_shuffled_control_passed) is bool
        and type(leakage.passed) is bool
    )


def _counterfactual_primitive_types_are_exact(result: CounterfactualCheckResult) -> bool:
    return (
        type(result) is CounterfactualCheckResult
        and type(result.check_id) is CounterfactualCheckId
        and type(result.checked_pairs) is int
        and type(result.decision_mismatch_count) is int
        and type(result.temporal_mismatch_count) is int
        and _is_sha256(result.result_payload_sha256)
        and type(result.passed) is bool
    )


def _expected_class_counts(
    task: ShortcutTask,
    episode_count: int,
) -> tuple[dict[str, int] | None, dict[str, int] | None]:
    if task is ShortcutTask.POSITIVE_BINARY:
        return (
            {"0": episode_count * 2 // 5, "1": episode_count * 2 // 5},
            {"0": episode_count // 10, "1": episode_count // 10},
        )
    if task is ShortcutTask.VARIANT_THREE_WAY:
        return (
            {"0": episode_count * 2 // 5, "1": episode_count // 5, "2": episode_count // 5},
            {"0": episode_count // 10, "1": episode_count // 20, "2": episode_count // 20},
        )
    if episode_count == 8_000:
        return (
            {str(index): 800 for index in range(4)},
            {str(index): 200 for index in range(4)},
        )
    return None, None


def _probe_metadata_is_consistent(
    probe: ShortcutProbeResult,
    *,
    episode_count: int,
) -> bool:
    if not _probe_primitive_types_are_exact(probe):
        return False
    group_index = tuple(ShortcutFeatureGroup).index(probe.feature_group)
    task_examples = (
        episode_count // 2 if probe.task is ShortcutTask.POSITIVE_HAZARD_CLASS else episode_count
    )
    expected_train = task_examples * 4 // 5
    expected_test = task_examples - expected_train
    expected_labels = {
        ShortcutTask.POSITIVE_BINARY: {"0", "1"},
        ShortcutTask.VARIANT_THREE_WAY: {"0", "1", "2"},
        ShortcutTask.POSITIVE_HAZARD_CLASS: {"0", "1", "2", "3"},
    }[probe.task]
    expected_chance = {
        ShortcutTask.POSITIVE_BINARY: 0.5,
        ShortcutTask.VARIANT_THREE_WAY: 1.0 / 3.0,
        ShortcutTask.POSITIVE_HAZARD_CLASS: 0.25,
    }[probe.task]
    exact_train, exact_test = _expected_class_counts(probe.task, episode_count)
    scaled_p = probe.raw_permutation_p * _PHASE1_PERMUTATION_DENOMINATOR
    permutation_numerator = round(scaled_p)
    bounded_values = (
        probe.raw_accuracy,
        probe.balanced_accuracy,
        probe.balanced_chance,
        probe.raw_permutation_p,
        probe.holm_adjusted_p,
    )
    return (
        all(0.0 <= value <= 1.0 for value in bounded_values)
        and probe.balanced_chance == expected_chance
        and probe.feature_dimension == _PHASE1_FEATURE_DIMENSIONS[probe.task][group_index]
        and probe.train_examples == expected_train
        and probe.test_examples == expected_test
        and probe.permutation_replicate_count == 4_999
        and set(probe.train_class_counts) == expected_labels
        and set(probe.test_class_counts) == expected_labels
        and all(value >= 0 for value in probe.train_class_counts.values())
        and all(value >= 0 for value in probe.test_class_counts.values())
        and sum(probe.train_class_counts.values()) == probe.train_examples
        and sum(probe.test_class_counts.values()) == probe.test_examples
        and min(probe.train_class_counts.values()) >= 200
        and min(probe.test_class_counts.values()) >= 200
        and (exact_train is None or probe.train_class_counts == exact_train)
        and (exact_test is None or probe.test_class_counts == exact_test)
        and 0 <= probe.optimizer_iterations <= 500
        and probe.optimizer_converged
        and 1 <= permutation_numerator <= _PHASE1_PERMUTATION_DENOMINATOR
        and probe.raw_permutation_p == permutation_numerator / _PHASE1_PERMUTATION_DENOMINATOR
    )


def _probe_family_is_consistent(
    probes: tuple[ShortcutProbeResult, ...],
    *,
    episode_count: int,
) -> bool:
    expected = tuple((task, group) for task in ShortcutTask for group in ShortcutFeatureGroup)
    if tuple((probe.task, probe.feature_group) for probe in probes) != expected:
        return False
    adjusted = _holm_adjusted_p_values(tuple(probe.raw_permutation_p for probe in probes))
    signatures: dict[ShortcutTask, tuple[object, ...]] = {}
    for index, probe in enumerate(probes):
        if (
            not _probe_metadata_is_consistent(probe, episode_count=episode_count)
            or probe.holm_adjusted_p != adjusted[index]
            or probe.passed
            != (not (probe.balanced_accuracy > probe.balanced_chance and adjusted[index] < 0.01))
        ):
            return False
        signature = (
            probe.train_examples,
            probe.test_examples,
            probe.train_class_counts,
            probe.test_class_counts,
        )
        if signatures.setdefault(probe.task, signature) != signature:
            return False
    return True


def _positive_control_is_consistent(control: PositiveControlResult) -> bool:
    if not _positive_control_primitive_types_are_exact(control):
        return False
    expected_family = tuple((control.target_task, group) for group in ShortcutFeatureGroup)
    observed_family = tuple((probe.task, probe.feature_group) for probe in control.probes)
    if observed_family != expected_family or any(
        not _probe_metadata_is_consistent(probe, episode_count=8_000) for probe in control.probes
    ):
        return False
    adjusted = _holm_adjusted_p_values(tuple(probe.raw_permutation_p for probe in control.probes))
    if any(
        not probe.optimizer_converged
        or probe.holm_adjusted_p != adjusted[index]
        or probe.passed
        != (not (probe.balanced_accuracy > probe.balanced_chance and adjusted[index] < 0.01))
        for index, probe in enumerate(control.probes)
    ):
        return False
    derived_observed = tuple(
        f"{probe.task.value}:{probe.feature_group.value}"
        for index, probe in enumerate(control.probes)
        if probe.balanced_accuracy >= 0.95
        and probe.raw_permutation_p <= 0.05
        and adjusted[index] < 0.01
    )
    expected_group = ShortcutFeatureGroup(control.expected_detector_id.split(":", 1)[1])
    expected_probe = next(
        (probe for probe in control.probes if probe.feature_group is expected_group),
        None,
    )
    derived_pass = control.expected_detector_id in derived_observed
    return (
        control.observed_detector_ids == derived_observed
        and expected_probe is not None
        and control.balanced_accuracy == expected_probe.balanced_accuracy
        and control.holm_adjusted_p == expected_probe.holm_adjusted_p
        and control.passed is derived_pass
    )


def _common_provenance_key(provenance: EvidenceProvenance) -> tuple[object, ...]:
    return (
        provenance.plan_base_revision,
        provenance.source_commit,
        provenance.source_dirty,
        provenance.generator_version,
        provenance.config_sha256,
        provenance.generator_source,
        provenance.analysis_source,
    )


def _independent_source_key(provenance: EvidenceProvenance) -> tuple[object, ...]:
    return (
        provenance.generation_mode,
        provenance.allocation_id,
        provenance.split_namespace,
        provenance.root_seed,
        provenance.public_id_seed_sha256,
    )


def _require_zero_call_common_provenance(
    provenances: tuple[EvidenceProvenance, ...],
) -> None:
    _require(
        all(item.foundation_model_calls == 0 for item in provenances),
        "Phase 1 evidence records nonzero foundation-model calls",
    )
    _require(
        all(not item.source_dirty for item in provenances),
        "Phase 1 evidence provenance records a dirty source tree",
    )
    expected = _common_provenance_key(provenances[0])
    _require(
        all(_common_provenance_key(item) == expected for item in provenances[1:]),
        "Phase 1 plan/source/config/generator provenance disagrees",
    )


def _require_empty_non_leakage_metadata(
    provenances: tuple[EvidenceProvenance, ...],
) -> None:
    _require(
        all(not item.analysis_seeds and item.leakage_audit is None for item in provenances),
        "non-leakage provenance metadata must be exactly empty",
    )


def _validation_path_length(cohort_index: int) -> int | None:
    for first, stop, path_length in _VALIDATION_COHORT_BLOCKS:
        if first <= cohort_index < stop:
            return path_length
    return None


def _require_validation_recipe(manifest: EpisodeManifest) -> None:
    _require(
        manifest.experiment_version == _VALIDATION_EXPERIMENT_VERSION,
        "validation artifact has the wrong frozen experiment version",
    )
    for entry_index, entry in enumerate(manifest.entries):
        coordinate = entry.coordinate
        expected_cohort_index, expected_member_index = divmod(entry_index, 4)
        _require(
            isinstance(coordinate, MatchedManifestCoordinate)
            and coordinate.cohort_index == expected_cohort_index
            and coordinate.member_index == expected_member_index
            and entry.requested_path_length == _validation_path_length(expected_cohort_index),
            "validation artifact does not match the ordered 834/833/833 allocation recipe",
        )


def _accepted_attempts(
    evidence: ConstructionNamespaceEvidence,
) -> tuple[int, ...]:
    attempts = tuple(
        run.accepted_attempt
        for run in evidence.accepted_attempt_runs
        for _ in range(run.draw_count)
    )
    _require(
        len(attempts) == evidence.accepted_draw_count,
        "construction attempt runs do not cover the declared corpus",
    )
    return attempts


def _reconstruct_matched_namespace(
    manifest: EpisodeManifest,
) -> tuple[ConstructionNamespaceEvidence, set[bytes], set[bytes]]:
    builder = ConstructionNamespaceBuilder(
        generation_mode="matched",
        public_id_seed=_VALIDATION_PUBLIC_ID_SEED,
        accepted_draw_count=2_500,
        clock_public_id_count=0,
    )
    token_values: set[bytes] = set()
    public_id_values: set[bytes] = set()
    for cohort_index, request in enumerate(
        CohortRequest(
            split_namespace=VALIDATION_ALLOCATION.split_namespace,
            suite=block.suite,
            root_seed=_VALIDATION_ROOT_SEED,
            cohort_index=index,
            requested_path_length=block.requested_path_length,
        )
        for block in VALIDATION_ALLOCATION.blocks
        for index in range(
            block.first_cohort_index,
            block.first_cohort_index + block.cohort_count,
        )
    ):
        members = manifest.entries[cohort_index * 4 : cohort_index * 4 + 4]
        attempt = members[0].accepted_attempt
        _require(
            len(members) == 4 and all(member.accepted_attempt == attempt for member in members),
            "validation cohort accepted attempts disagree",
        )
        expected_ids = allocate_public_ids(
            PublicIdBatchKey(
                generator_version="ofd-v1",
                split_namespace=request.split_namespace,
                suite=request.suite,
                public_id_seed=_VALIDATION_PUBLIC_ID_SEED,
                cohort_index=request.cohort_index,
                accepted_attempt=attempt,
            )
        )
        actual_ids = tuple(member.episode_public_id for member in members)
        _require(
            actual_ids == expected_ids,
            "validation public ID is not derivable from its recipe",
        )
        tokens = matched_seed_tokens(request, attempt)
        builder.add_draw(attempt, tokens, actual_ids)
        token_values.update(bytes.fromhex(token) for token in tokens)
        public_id_values.update(uuid.UUID(public_id).bytes for public_id in actual_ids)
    return builder.finalize(), token_values, public_id_values


def _independent_clock_public_ids(
    requests: tuple[IndependentEpisodeRequest, ...],
    attempts: tuple[int, ...],
) -> tuple[str, ...]:
    request_attempts = {
        (request.suite, request.requested_path_length, request.episode_index): attempt
        for request, attempt in zip(requests, attempts, strict=True)
    }
    identifiers: list[str] = []
    for suite, count_field in (
        (SuiteName.CLOCK_SCALE_0_1X, "scale_0_1x_episode_count"),
        (SuiteName.CLOCK_SCALE_10X, "scale_10x_episode_count"),
    ):
        for block in PHASE1_GATE_ALLOCATION.clock_blocks:
            for offset in range(getattr(block, count_field)):
                episode_index = block.source_first_episode_index + offset
                key = (SuiteName.IID_PRIMARY, block.requested_path_length, episode_index)
                attempt = request_attempts.get(key)
                _require(attempt is not None, "independent clock parent is absent")
                identifiers.append(
                    allocate_independent_public_id(
                        IndependentPublicIdKey(
                            generator_version="ofd-v1",
                            split_namespace=SplitNamespace.PHASE1_GATE,
                            suite=suite,
                            public_id_seed=_INDEPENDENT_PUBLIC_ID_SEED,
                            episode_index=episode_index,
                            accepted_attempt=attempt,
                        )
                    )
                )
    return tuple(identifiers)


def _reconstruct_independent_namespace(
    evidence: ConstructionNamespaceEvidence,
    *,
    matched_tokens: set[bytes],
    matched_public_ids: set[bytes],
) -> ConstructionNamespaceEvidence:
    attempts = _accepted_attempts(evidence)
    requests, public_ids, clock_parents = _leakage_coordinates(attempts)
    _require(
        len(requests) == len(attempts) == _INDEPENDENT_EPISODE_COUNT,
        "independent namespace draw count is incomplete",
    )
    clock_ids = tuple(public_id for _, _, public_id in clock_parents)
    builder = ConstructionNamespaceBuilder(
        generation_mode="independent",
        public_id_seed=_INDEPENDENT_PUBLIC_ID_SEED,
        accepted_draw_count=_INDEPENDENT_EPISODE_COUNT,
        clock_public_id_count=len(clock_ids),
    )
    for request, attempt, public_id in zip(requests, attempts, public_ids, strict=True):
        tokens = independent_seed_tokens(request, attempt)
        _require(
            not any(bytes.fromhex(token) in matched_tokens for token in tokens),
            "matched and independent construction tokens collide",
        )
        _require(
            uuid.UUID(public_id).bytes not in matched_public_ids,
            "matched and independent base public IDs collide",
        )
        builder.add_draw(attempt, tokens, (public_id,))
    for public_id in clock_ids:
        _require(
            uuid.UUID(public_id).bytes not in matched_public_ids,
            "matched and independent clock public IDs collide",
        )
        builder.add_clock_public_id(public_id)
    return builder.finalize()


def _has_frozen_reproducibility_matrix(report: ReproducibilityReport) -> bool:
    return (
        report.sample_size == _REPRODUCIBILITY_SAMPLE_SIZE
        and report.modes == _REPRODUCIBILITY_MODES
        and report.chunk_sizes == _REPRODUCIBILITY_CHUNK_SIZES
        and report.python_hash_seeds == _REPRODUCIBILITY_PYTHON_HASH_SEEDS
        and report.mismatch_count == 0
    )


def _require_validation_artifacts(
    manifest: EpisodeManifest,
    reproducibility: ReproducibilityReport,
) -> tuple[str, set[bytes], set[bytes]]:
    _require_namespace_evidence(
        reproducibility.namespace_evidence,
        reproducibility.provenance,
        generation_mode="matched",
        public_id_seed=_VALIDATION_PUBLIC_ID_SEED,
        accepted_draw_count=2_500,
        seed_token_count=50_000,
        base_public_id_count=10_000,
        clock_public_id_count=0,
    )
    _require(
        manifest.access_class is ManifestAccessClass.VALIDATION
        and manifest.suite is SuiteName.VALIDATION
        and manifest.provenance.generation_mode == "matched"
        and manifest.provenance.allocation_id == "validation-v1"
        and manifest.provenance.split_namespace is SplitNamespace.VALIDATION
        and manifest.provenance.root_seed == _VALIDATION_ROOT_SEED
        and manifest.public_id_seed == _VALIDATION_PUBLIC_ID_SEED
        and manifest.provenance.public_id_seed_sha256 == _VALIDATION_PUBLIC_ID_SEED_SHA256
        and manifest.episode_count == _VALIDATION_EPISODE_COUNT,
        "validation artifact does not have the frozen 10,000-episode recipe",
    )
    _require_validation_recipe(manifest)
    manifest_payload_sha256 = sha256_bytes(canonical_json_bytes(manifest))
    expected_corpus = _leakage_corpus_sha256(
        tuple(entry.episode_public_id for entry in manifest.entries),
        tuple(entry.episode_sha256 for entry in manifest.entries),
    )
    selected = select_manifest_reproducibility_sample(
        manifest,
        manifest_payload_sha256,
        _REPRODUCIBILITY_SAMPLE_SIZE,
    )
    expected_membership = manifest_sample_membership_sha256(
        manifest_payload_sha256,
        selected,
    )
    _require(
        reproducibility.passed
        and reproducibility.source_mode == "manifest"
        and _has_frozen_reproducibility_matrix(reproducibility)
        and reproducibility.verified_source_entries == _VALIDATION_EPISODE_COUNT
        and reproducibility.source_payload_sha256 == manifest_payload_sha256
        and reproducibility.reference_corpus_sha256 == expected_corpus
        and reproducibility.sample_membership_sha256 == expected_membership
        and reproducibility.modes == _REPRODUCIBILITY_MODES,
        "validation reproducibility artifact is failed or inconsistent",
    )
    _require(
        reproducibility.provenance == manifest.provenance,
        "validation manifest and reproducibility provenance disagree",
    )
    try:
        namespace, tokens, public_ids = _reconstruct_matched_namespace(manifest)
    except (TypeError, ValueError, ProvenanceError) as error:
        raise _artifact_error("validation namespace reconstruction failed") from error
    _require(
        reproducibility.namespace_evidence == namespace,
        "validation namespace evidence is not derivable from the manifest",
    )
    return manifest_payload_sha256, tokens, public_ids


def _require_oracle(oracle: OracleEvaluationReport) -> None:
    try:
        reparsed = OracleEvaluationReport.model_validate(oracle.model_dump(mode="python"))
    except ValidationError as error:
        raise _artifact_error("oracle derived report validation failed") from error
    _require(reparsed == oracle, "oracle report is not its strict derived form")
    _require_namespace_evidence(
        oracle.namespace_evidence,
        oracle.provenance,
        generation_mode="independent",
        public_id_seed=_INDEPENDENT_PUBLIC_ID_SEED,
        accepted_draw_count=_INDEPENDENT_EPISODE_COUNT,
        seed_token_count=700_000,
        base_public_id_count=_INDEPENDENT_EPISODE_COUNT,
        clock_public_id_count=sum(_CLOCK_COUNTS),
    )
    _require(oracle.passed, "oracle inner report passed flag is false")
    _require(
        oracle.source_mode == "phase1_gate"
        and oracle.requested_episode_count == _INDEPENDENT_EPISODE_COUNT
        and oracle.verified_episode_count == _INDEPENDENT_EPISODE_COUNT
        and (
            oracle.positive_count,
            oracle.safe_negative_count,
            oracle.disconnected_negative_count,
        )
        == _INDEPENDENT_VARIANT_COUNTS
        and oracle.oracle_successes == _INDEPENDENT_EPISODE_COUNT
        and oracle.oracle_failures == 0
        and oracle.invariant_failures == 0
        and oracle.oracle_ambiguities == 0
        and oracle.seed_token_collisions == 0
        and oracle.public_id_collisions == 0
        and oracle.random_positive.total == 50_000
        and oracle.random_negative.total == 50_000
        and oracle.random_positive.passed
        and oracle.random_negative.passed
        and (
            oracle.clock_0_1x_episode_count,
            oracle.clock_10x_episode_count,
        )
        == _CLOCK_COUNTS
        and oracle.suite_path_denominators == _GATE_DENOMINATORS,
        "oracle artifact has a wrong frozen denominator or failed diagnostic",
    )


def _require_sufficient_leakage_evidence(leakage: LeakageReport) -> None:
    try:
        payload = leakage.model_dump(mode="json", warnings="error")
        payload_size = len(canonical_json_bytes(payload))
    except (AttributeError, TypeError, ValueError) as error:
        raise _artifact_error("leakage sufficient evidence cannot be serialized exactly") from error
    _require(
        payload_size <= _MAX_LEAKAGE_REPORT_BYTES,
        "canonical leakage report exceeds 16 MiB",
    )
    try:
        statistics = LeakageConstructionStatistics.model_validate(
            leakage.construction_statistics.model_dump(mode="python")
        )
        membership = LeakageMembershipEvidence.model_validate(
            leakage.membership_evidence.model_dump(mode="python")
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise _artifact_error("leakage sufficient evidence schema is invalid") from error
    anchor = leakage.provenance.leakage_audit
    _require(anchor is not None, "leakage sufficient evidence anchor is missing")
    _require(
        statistics.source_episode_sha256s.item_count == leakage.episode_count == 100_000
        and statistics.clock_child_episode_sha256s.item_count == 7_000
        and statistics.invariant_verified_count == 100_000
        and statistics.feature_row_count == statistics.finite_feature_row_count == 100_000
        and statistics.second_pass_verified_count == statistics.second_pass_match_count == 100_000
        and statistics.first_pass_source_manifest_sha256
        == statistics.second_pass_source_manifest_sha256
        == anchor.source_manifest_sha256
        and membership.train_episode_count == 80_000
        and membership.test_episode_count == 20_000
        and membership.split_membership_sha256 == leakage.split_membership_hash
        and membership.train_membership_sha256 == leakage.train_membership_hash
        and membership.test_membership_sha256 == leakage.test_membership_hash,
        "leakage sufficient construction or membership evidence is inconsistent",
    )
    for control in leakage.positive_controls:
        try:
            validated = PositiveControlResult.model_validate(control.model_dump(mode="python"))
        except (AttributeError, TypeError, ValueError) as error:
            raise _artifact_error("leakage control sufficient evidence is invalid") from error
        _require(
            validated.injected_episode_sha256s.item_count == 8_000,
            "leakage control digest pack count is not exact",
        )
    _require(
        len({control.base_subset_membership_sha256 for control in leakage.positive_controls}) == 1,
        "leakage control subset memberships disagree",
    )


def _require_leakage(leakage: LeakageReport) -> None:
    _require_sufficient_leakage_evidence(leakage)
    _require_namespace_evidence(
        leakage.namespace_evidence,
        leakage.provenance,
        generation_mode="independent",
        public_id_seed=_INDEPENDENT_PUBLIC_ID_SEED,
        accepted_draw_count=_INDEPENDENT_EPISODE_COUNT,
        seed_token_count=700_000,
        base_public_id_count=_INDEPENDENT_EPISODE_COUNT,
        clock_public_id_count=sum(_CLOCK_COUNTS),
    )
    _require(
        _leakage_report_primitive_types_are_exact(leakage),
        "leakage outer report types or containers are invalid",
    )
    _require(
        type(leakage.passed) is bool and leakage.passed,
        "leakage inner report passed flag is false or mistyped",
    )
    _require(
        leakage.generation_mode == "independent"
        and leakage.profile is LeakageAuditProfileName.PHASE1_GATE
        and leakage.episode_count == _INDEPENDENT_EPISODE_COUNT
        and leakage.randomization_block_count == 25_000
        and leakage.suite_path_denominators == _GATE_DENOMINATORS
        and type(leakage.construction_checks) is dict
        and leakage.construction_check_ids == _CONSTRUCTION_CHECK_IDS
        and tuple(leakage.construction_checks) == tuple(sorted(_CONSTRUCTION_CHECK_IDS))
        and all(type(value) is bool for value in leakage.construction_checks.values())
        and type(leakage.label_shuffled_control_passed) is bool
        and leakage.label_shuffled_control_passed
        and type(leakage.feature_schema_hash) is str
        and leakage.feature_schema_hash == _FEATURE_SCHEMA_SHA256,
        "leakage artifact has a wrong frozen denominator or failed check",
    )
    _require(
        _probe_family_is_consistent(
            leakage.probes,
            episode_count=_INDEPENDENT_EPISODE_COUNT,
        )
        and all(probe.passed for probe in leakage.probes),
        "leakage clean-probe family is absent, incomplete, or failed",
    )
    _require(
        _probe_family_is_consistent(
            leakage.label_shuffled_probes,
            episode_count=_INDEPENDENT_EPISODE_COUNT,
        )
        and all(probe.passed for probe in leakage.label_shuffled_probes)
        and leakage.label_shuffled_control_passed,
        "leakage label-shuffled probe family is absent, incomplete, or failed",
    )
    expected_counterfactuals = tuple(_COUNTERFACTUAL_COUNTS.items())
    _require(
        len(leakage.counterfactual_checks) == len(expected_counterfactuals)
        and all(
            _counterfactual_primitive_types_are_exact(result)
            and result.check_id is expected_id
            and result.checked_pairs == expected_count
            and result.decision_mismatch_count == 0
            and result.temporal_mismatch_count == 0
            and result.passed
            for result, (expected_id, expected_count) in zip(
                leakage.counterfactual_checks,
                expected_counterfactuals,
                strict=True,
            )
        ),
        "leakage counterfactual check failed or has a wrong denominator",
    )
    expected_controls = tuple(
        (injector.control_id, injector.target_task, injector.expected_detector_id)
        for injector in NAMED_LEAK_INJECTORS
    )
    observed_controls = tuple(
        (control.control_id, control.target_task, control.expected_detector_id)
        for control in leakage.positive_controls
    )
    _require(
        observed_controls == expected_controls
        and all(
            _positive_control_is_consistent(control) and control.passed
            for control in leakage.positive_controls
        )
        and len({control.base_subset_corpus_sha256 for control in leakage.positive_controls}) == 1
        and len({control.split_membership_sha256 for control in leakage.positive_controls}) == 1,
        "leakage positive-control family is absent, incomplete, or failed",
    )
    anchor = leakage.provenance.leakage_audit
    _require(anchor is not None, "leakage provenance is missing its frozen audit authority")
    assert anchor is not None
    expected_descriptor = AuditSourceDescriptor(
        schema_version="leakage-source-v1",
        generation_mode="independent",
        allocation_id="phase1-independent-gate-v1",
        allocation_or_manifest_sha256=_PHASE1_GATE_ALLOCATION_SHA256,
        split_namespace=SplitNamespace.PHASE1_GATE,
        root_seed=leakage.provenance.root_seed,
        public_id_seed_sha256=leakage.provenance.public_id_seed_sha256,
        config_sha256=leakage.provenance.config_sha256,
        generator_source_sha256=leakage.provenance.generator_source.sha256,
        episode_count=_INDEPENDENT_EPISODE_COUNT,
    )
    _require(
        leakage.provenance.analysis_seeds
        == {"audit_seed": 2026083091, "positive_control_seed": 2026083092}
        and anchor.profile == LeakageAuditProfileName.PHASE1_GATE.value
        and anchor.allocation_id == leakage.provenance.allocation_id == "phase1-independent-gate-v1"
        and anchor.allocation_or_manifest_sha256 == _PHASE1_GATE_ALLOCATION_SHA256
        and anchor.config_sha256 == leakage.provenance.config_sha256
        and anchor.descriptor_sha256 == audit_source_descriptor_sha256(expected_descriptor)
        and anchor.suite_path_denominators == leakage.suite_path_denominators == _GATE_DENOMINATORS
        and anchor.clock_scale_pair_counts == _CLOCK_PAIR_COUNTS
        and anchor.episode_count == leakage.episode_count == _INDEPENDENT_EPISODE_COUNT,
        "leakage provenance contradicts its frozen audit authority",
    )
    _require_rederived_leakage(leakage)


def _require_independent_reproducibility(report: ReproducibilityReport) -> None:
    _require_namespace_evidence(
        report.namespace_evidence,
        report.provenance,
        generation_mode="independent",
        public_id_seed=_INDEPENDENT_PUBLIC_ID_SEED,
        accepted_draw_count=_INDEPENDENT_EPISODE_COUNT,
        seed_token_count=700_000,
        base_public_id_count=_INDEPENDENT_EPISODE_COUNT,
        clock_public_id_count=sum(_CLOCK_COUNTS),
    )
    _require(
        report.passed
        and report.source_mode == "independent_allocation"
        and _has_frozen_reproducibility_matrix(report)
        and report.verified_source_entries == _INDEPENDENT_EPISODE_COUNT
        and report.mismatch_count == 0,
        "independent reproducibility artifact is failed or has a wrong denominator",
    )
    expected_source = IndependentSourceDescriptor(
        schema_version="phase1-independent-source-v1",
        allocation_id="phase1-independent-gate-v1",
        allocation_sha256=_PHASE1_GATE_ALLOCATION_SHA256,
        split_namespace=SplitNamespace.PHASE1_GATE,
        root_seed=report.provenance.root_seed,
        public_id_seed_sha256=report.provenance.public_id_seed_sha256,
        config_sha256=report.provenance.config_sha256,
        generator_source_sha256=report.provenance.generator_source.sha256,
    )
    _require(
        report.source_payload_sha256 == sha256_bytes(canonical_json_bytes(expected_source)),
        "independent reproducibility source descriptor disagrees with frozen provenance",
    )
    requests = tuple(iter_independent_requests(PHASE1_GATE_ALLOCATION, _INDEPENDENT_ROOT_SEED))
    selected = select_independent_reproducibility_sample(
        requests,
        report.source_payload_sha256,
        _REPRODUCIBILITY_SAMPLE_SIZE,
    )
    _require(
        report.sample_membership_sha256
        == independent_sample_membership_sha256(report.source_payload_sha256, selected),
        "independent reproducibility sample membership is not derivable",
    )


def verify_phase1_gate_artifacts(
    *,
    validation_path: Path,
    oracle_path: Path,
    leakage_path: Path,
    validation_reproducibility_path: Path,
    independent_reproducibility_path: Path,
) -> Phase1GateVerificationResult:
    """Strictly load and cross-check the five frozen Phase 1 evidence files."""
    manifest = _load_validation(validation_path)
    oracle = _load_report(oracle_path, OracleEvaluationReport, name="oracle")
    leakage = _load_report(leakage_path, LeakageReport, name="leakage")
    validation_reproducibility = _load_report(
        validation_reproducibility_path,
        ReproducibilityReport,
        name="validation reproducibility",
    )
    independent_reproducibility = _load_report(
        independent_reproducibility_path,
        ReproducibilityReport,
        name="independent reproducibility",
    )

    _require_empty_non_leakage_metadata(
        (
            manifest.provenance,
            oracle.provenance,
            validation_reproducibility.provenance,
            independent_reproducibility.provenance,
        )
    )

    manifest_payload_sha256, matched_tokens, matched_public_ids = _require_validation_artifacts(
        manifest, validation_reproducibility
    )
    _require_oracle(oracle)
    _require_leakage(leakage)
    _require_independent_reproducibility(independent_reproducibility)

    provenances = (
        manifest.provenance,
        oracle.provenance,
        leakage.provenance,
        validation_reproducibility.provenance,
        independent_reproducibility.provenance,
    )
    _require_zero_call_common_provenance(provenances)
    _require(
        all(item.config_sha256 == _PRIMARY_CONFIG_SHA256 for item in provenances),
        "Phase 1 evidence does not use the frozen primary configuration",
    )
    for provenance in provenances:
        try:
            authenticate_final_phase1_provenance(provenance, repo_root=Path.cwd())
        except (TypeError, ProvenanceError) as error:
            raise _artifact_error("historical Git provenance authentication failed") from error
    independent_key = _independent_source_key(oracle.provenance)
    _require(
        all(
            _independent_source_key(item) == independent_key
            for item in (leakage.provenance, independent_reproducibility.provenance)
        )
        and independent_key
        == (
            "independent",
            "phase1-independent-gate-v1",
            SplitNamespace.PHASE1_GATE,
            _INDEPENDENT_ROOT_SEED,
            _INDEPENDENT_PUBLIC_ID_SEED_SHA256,
        ),
        "independent gate source provenance disagrees",
    )
    _require(
        oracle.corpus_sha256
        == leakage.corpus_hash
        == independent_reproducibility.reference_corpus_sha256,
        "independent gate corpus SHA-256 values disagree",
    )
    _require(
        oracle.namespace_evidence
        == leakage.namespace_evidence
        == independent_reproducibility.namespace_evidence,
        "independent namespace evidence disagrees across reports",
    )
    try:
        independent_namespace = _reconstruct_independent_namespace(
            oracle.namespace_evidence,
            matched_tokens=matched_tokens,
            matched_public_ids=matched_public_ids,
        )
    except (TypeError, ValueError, ProvenanceError) as error:
        raise _artifact_error("independent namespace reconstruction failed") from error
    _require(
        independent_namespace == oracle.namespace_evidence,
        "independent namespace evidence is not derivable from frozen coordinates",
    )
    construction_token_count = (
        validation_reproducibility.namespace_evidence.seed_token_count
        + independent_namespace.seed_token_count
    )
    public_id_count = (
        validation_reproducibility.namespace_evidence.total_public_id_count
        + independent_namespace.total_public_id_count
    )
    _require(
        construction_token_count == 750_000 and public_id_count == 117_000,
        "combined Phase 1 namespace counts are not exact",
    )
    return Phase1GateVerificationResult(
        schema_version="phase1-gate-verification-v3",
        validation_episode_count=_VALIDATION_EPISODE_COUNT,
        independent_episode_count=_INDEPENDENT_EPISODE_COUNT,
        matched_accepted_draw_count=2_500,
        independent_accepted_draw_count=_INDEPENDENT_EPISODE_COUNT,
        matched_public_id_seed=_VALIDATION_PUBLIC_ID_SEED,
        independent_public_id_seed=_INDEPENDENT_PUBLIC_ID_SEED,
        construction_token_count=construction_token_count,
        public_id_count=public_id_count,
        validation_manifest_payload_sha256=manifest_payload_sha256,
        independent_corpus_sha256=oracle.corpus_sha256,
        foundation_model_calls=0,
        passed=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the five immutable Phase 1 gate artifacts locally."
    )
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--leakage", type=Path, required=True)
    parser.add_argument("--validation-reproducibility", type=Path, required=True)
    parser.add_argument("--independent-reproducibility", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = verify_phase1_gate_artifacts(
            validation_path=arguments.validation,
            oracle_path=arguments.oracle,
            leakage_path=arguments.leakage,
            validation_reproducibility_path=arguments.validation_reproducibility,
            independent_reproducibility_path=arguments.independent_reproducibility,
        )
    except SilentCascadeError as error:
        sys.stderr.buffer.write(canonical_json_bytes(error.to_payload()) + b"\n")
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
