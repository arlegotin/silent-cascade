"""Contracts for the fail-closed public-feature leakage auditor."""

from pathlib import Path

import numpy as np
import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.generator import CohortRequest, generate_matched_cohort
from silent_cascade.provenance import (
    GENERATOR_SOURCE_PATHS,
    PHASE1_ANALYSIS_SOURCE_PATHS,
    EvidenceProvenance,
    SourceTreeFingerprint,
    public_id_seed_sha256,
)


def _config() -> Phase1Config:
    return resolve_config(
        Phase1Config, [Path("configs/base.yaml"), Path("configs/data/primary.yaml")]
    ).config


def _provenance(config_sha256: str) -> EvidenceProvenance:
    return EvidenceProvenance(
        schema_version="phase1-evidence-provenance-v1",
        plan_base_revision="c" * 40,
        source_commit="d" * 40,
        source_dirty=False,
        generator_version="ofd-v1",
        generation_mode="matched",
        allocation_id="test-leakage-v1",
        split_namespace=SplitNamespace.DEBUG,
        config_sha256=config_sha256,
        generator_source=SourceTreeFingerprint(
            frame_version="sc-source-tree-v1",
            scope="generator",
            paths=GENERATOR_SOURCE_PATHS,
            sha256="a" * 64,
        ),
        analysis_source=SourceTreeFingerprint(
            frame_version="sc-source-tree-v1",
            scope="phase1_analysis",
            paths=PHASE1_ANALYSIS_SOURCE_PATHS,
            sha256="b" * 64,
        ),
        root_seed=41,
        public_id_seed_sha256=public_id_seed_sha256(91),
        analysis_seeds={"audit_seed": 2026083091, "positive_control_seed": 2026083092},
    )


def test_shortcut_features_have_fixed_public_only_dimensions() -> None:
    """Adding a private field or changing one declared feature width must fail this contract."""
    from silent_cascade.env.leakage import (
        AuditExample,
        ShortcutFeatureGroup,
        extract_shortcut_features,
    )

    config = _config()
    bundle = generate_matched_cohort(
        config,
        CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 41, 7, 3),
        91,
    ).episodes[0]
    features = extract_shortcut_features(AuditExample(bundle, 3, "matched", 7, 0), corpus_size=8)

    assert {group: vector.shape for group, vector in features.vectors.items()} == {
        ShortcutFeatureGroup.ID_POSITION: (17,),
        ShortcutFeatureGroup.ACTIVATION_NODE: (256,),
        ShortcutFeatureGroup.FIRST_LAST_FACT: (278,),
        ShortcutFeatureGroup.COUNTS: (4,),
        ShortcutFeatureGroup.ORDER_RECORD_IDS: (768,),
        ShortcutFeatureGroup.TIMES: (194,),
        ShortcutFeatureGroup.TERMINAL_MULTISET: (10,),
        ShortcutFeatureGroup.LINK_TOPOLOGY: (33,),
        ShortcutFeatureGroup.COMBINED: (1560,),
    }
    assert all(vector.dtype == np.float32 for vector in features.vectors.values())
    assert np.array_equal(
        features.vectors[ShortcutFeatureGroup.COMBINED],
        np.concatenate(
            [
                features.vectors[group]
                for group in ShortcutFeatureGroup
                if group is not ShortcutFeatureGroup.COMBINED
            ]
        ),
    )


def test_audit_rejects_unauthenticated_source_before_iteration(tmp_path: Path) -> None:
    """Removing descriptor/provenance binding must fail before private bundles are read."""
    from silent_cascade.env.leakage import (
        AuditSourceDescriptor,
        InMemoryAuditSource,
        LeakageAuditProfileName,
        audit_leakage,
    )

    config = _config()
    source = InMemoryAuditSource(
        AuditSourceDescriptor(
            schema_version="leakage-source-v1",
            generation_mode="matched",
            allocation_id="wrong-allocation",
            allocation_or_manifest_sha256="f" * 64,
            split_namespace=SplitNamespace.DEBUG,
            root_seed=41,
            public_id_seed_sha256=public_id_seed_sha256(91),
            config_sha256="e" * 64,
            generator_source_sha256="a" * 64,
            episode_count=4,
        ),
        (),
    )

    with pytest.raises(ValueError, match="source descriptor"):
        audit_leakage(
            source,
            config.data.leakage_audit,
            LeakageAuditProfileName.TEST,
            _provenance("e" * 64),
            tmp_path,
        )


def test_counterfactual_hash_known_answers_are_canonical() -> None:
    """Changing pair framing/order must fail the frozen counterfactual evidence answers."""
    from silent_cascade.env.leakage import (
        CounterfactualCheckId,
        CounterfactualPairResult,
        counterfactual_pair_key,
        counterfactual_result_payload_hash,
    )

    key = counterfactual_pair_key(
        CounterfactualCheckId.TERMINAL_DELAY_SWAP,
        (
            "00000000-0000-4000-8000-000000000001",
            "00000000-0000-4000-8000-000000000002",
        ),
        "swap_terminal_delay",
    )
    assert key == "a72ebffcd5bedcb5da8932a6bacdf307994dddeb8951e8976638b9c1c1e961ca"
    pair = CounterfactualPairResult(
        pair_key_sha256=key, decision_mismatch=False, temporal_mismatch=False
    )
    assert counterfactual_result_payload_hash(
        CounterfactualCheckId.TERMINAL_DELAY_SWAP, (pair,)
    ) == ("28b15afe82e20b9e8db3c2f5bf4402e454858bfdaf7ce9663001d5aac99f1372")


def test_named_positive_controls_have_one_disjoint_targeted_detector() -> None:
    """Deleting a declared leak family or retargeting it to COMBINED must fail this contract."""
    from silent_cascade.env.leakage import NAMED_LEAK_INJECTORS

    assert {
        (injector.control_id, injector.expected_detector_id) for injector in NAMED_LEAK_INJECTORS
    } == {
        ("PC_COUNT_BY_LABEL", "positive_binary:counts"),
        ("PC_ACTIVATION_GAP_BY_LABEL", "positive_binary:times"),
        ("PC_TERMINAL_ORDER_BY_VARIANT", "variant_three_way:order_record_ids"),
        ("PC_ACTIVATION_ID_BY_LABEL", "positive_binary:activation_node"),
        ("PC_RECORD_ID_BY_VARIANT", "variant_three_way:order_record_ids"),
        ("PC_PUBLIC_ID_BY_LABEL", "positive_binary:id_position"),
        ("PC_DELAY_BY_LABEL", "positive_binary:terminal_multiset"),
        ("PC_HAZARD_LAYOUT_BY_CLASS", "positive_hazard_class:order_record_ids"),
        ("PC_MANIFEST_ORDER_BY_VARIANT", "variant_three_way:id_position"),
    }


def test_counterfactual_schemas_reject_forged_passes_and_unknown_tags() -> None:
    """Publishing a mismatch as passing or accepting an undeclared transform must fail."""
    from silent_cascade.env.leakage import (
        CounterfactualCheckId,
        CounterfactualCheckResult,
        counterfactual_pair_key,
    )

    with pytest.raises(ValueError, match="passed"):
        CounterfactualCheckResult(
            check_id=CounterfactualCheckId.TERMINAL_DELAY_SWAP,
            checked_pairs=1,
            decision_mismatch_count=1,
            temporal_mismatch_count=0,
            result_payload_sha256="f" * 64,
            passed=True,
        )
    with pytest.raises(ValueError, match="transform"):
        counterfactual_pair_key(
            CounterfactualCheckId.PRESENTATION_PERMUTATION,
            ("00000000-0000-4000-8000-000000000001",),
            "untrusted-tag",
        )


def test_hazard_task_physically_removes_combined_identity_channels_and_preserves_flags() -> None:
    """Wrong offsets or standardized presence flags must fail this preprocessor contract."""
    from silent_cascade.env.leakage import (
        ShortcutFeatureGroup,
        _hazard_identity_columns,
        _standardize,
    )

    mask = _hazard_identity_columns(ShortcutFeatureGroup.COMBINED)
    assert {406, 407, 408, 409, 545, 546, 547, 548}.issubset(set(np.flatnonzero(mask)))
    values = np.zeros((4, 1560), dtype=np.float32)
    values[:, 0] = (1.0, 1.0, 0.0, 0.0)
    train, test = _standardize(values[:2], values[2:], ShortcutFeatureGroup.COMBINED)
    assert np.array_equal(train[:, 0], (1.0, 1.0))
    assert np.array_equal(test[:, 0], (0.0, 0.0))


def test_group_permutations_select_only_complete_declared_assignments() -> None:
    """A per-position remap that changes a quartet class multiset must fail this null contract."""
    from silent_cascade.env.episode import EpisodeVariant
    from silent_cascade.env.leakage import ShortcutTask, _permuted_labels, _StoredExample

    rows = tuple(
        _StoredExample(
            public_id=f"00000000-0000-4000-8000-{index:012d}",
            digest="a" * 64,
            group_id="matched:7",
            suite=SuiteName.IID_PRIMARY,
            path_length=3,
            variant=variant,
            hazard_class=None,
            block=7,
            position=index,
        )
        for index, variant in enumerate(
            (
                EpisodeVariant.POSITIVE,
                EpisodeVariant.POSITIVE,
                EpisodeVariant.SAFE_NEGATIVE,
                EpisodeVariant.DISCONNECTED_NEGATIVE,
            )
        )
    )
    binary = np.asarray((1, 1, 0, 0), dtype=np.int8)
    variant = np.asarray((0, 0, 1, 2), dtype=np.int8)
    for replicate in range(19):
        assert sorted(
            _permuted_labels(
                rows, binary, ShortcutTask.POSITIVE_BINARY, replicate, "b" * 64, 91
            ).tolist()
        ) == [0, 0, 1, 1]
        assert sorted(
            _permuted_labels(
                rows, variant, ShortcutTask.VARIANT_THREE_WAY, replicate, "b" * 64, 91
            ).tolist()
        ) == [0, 0, 1, 2]


def test_audit_streams_a_small_complete_source_and_cleans_its_memmaps(tmp_path: Path) -> None:
    """Retaining bundles or omitting a report family must fail this bounded audit contract."""
    from silent_cascade.env.leakage import (
        AuditExample,
        AuditSourceDescriptor,
        CounterfactualCheckId,
        InMemoryAuditSource,
        LeakageAuditProfileName,
        audit_leakage,
    )

    config = _config()
    examples = tuple(
        AuditExample(bundle, rank, "matched", cohort_index, member_index)
        for cohort_index in range(16)
        for member_index, bundle in enumerate(
            generate_matched_cohort(
                config,
                CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 41, cohort_index, 3),
                91,
            ).episodes
        )
        for rank in (cohort_index * 4 + member_index,)
    )
    audit_config = config.data.leakage_audit.model_copy(
        update={
            "test": config.data.leakage_audit.test.model_copy(
                update={
                    "episode_count": len(examples),
                    "permutation_replicates": 1,
                    "minimum_test_examples_per_class": 1,
                }
            )
        }
    )
    source = InMemoryAuditSource(
        AuditSourceDescriptor(
            schema_version="leakage-source-v1",
            generation_mode="matched",
            allocation_id="test-leakage-v1",
            allocation_or_manifest_sha256="f" * 64,
            split_namespace=SplitNamespace.DEBUG,
            root_seed=41,
            public_id_seed_sha256=public_id_seed_sha256(91),
            config_sha256="e" * 64,
            generator_source_sha256="a" * 64,
            episode_count=len(examples),
        ),
        examples,
    )

    report = audit_leakage(
        source,
        audit_config,
        LeakageAuditProfileName.TEST,
        _provenance("e" * 64),
        tmp_path,
    )

    assert report.episode_count == 64
    assert len(report.probes) == 27
    assert tuple(item.check_id for item in report.counterfactual_checks) == tuple(
        CounterfactualCheckId
    )
    assert not list(tmp_path.iterdir())
