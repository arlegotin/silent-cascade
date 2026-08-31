"""Contracts for the fail-closed public-feature leakage auditor."""

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.generator import CohortRequest, generate_matched_cohort
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
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


def _config_sha256(config: Phase1Config) -> str:
    return sha256_bytes(canonical_json_bytes(config))


def _authenticated_test_source(config: Phase1Config, groups_per_path: int = 100):
    from silent_cascade.env.episode import scale_episode_time
    from silent_cascade.env.leakage import (
        AuditExample,
        AuditSourceAuthentication,
        AuditSourceDescriptor,
        InMemoryAuditSource,
        LeakageAuditProfileName,
        PairedClockAuditPair,
        _clock_pair_manifest_sha256,
        _source_manifest_sha256,
        audit_source_descriptor_sha256,
    )

    examples = tuple(
        AuditExample(bundle, rank, "matched", cohort_index, member_index)
        for path_index, path in enumerate((2, 3, 4))
        for local_group in range(groups_per_path)
        for cohort_index in (path_index * groups_per_path + local_group,)
        for member_index, bundle in enumerate(
            generate_matched_cohort(
                config,
                CohortRequest(
                    SplitNamespace.DEBUG,
                    SuiteName.IID_PRIMARY,
                    41,
                    cohort_index,
                    path,
                ),
                91,
            ).episodes
        )
        for rank in ((path_index * groups_per_path + local_group) * 4 + member_index,)
    )
    clock_pairs = (
        PairedClockAuditPair(
            examples[0],
            scale_episode_time(
                examples[0].bundle,
                SuiteName.CLOCK_SCALE_0_1X,
                "00000000-0000-4000-8000-000000000001",
            ),
        ),
        PairedClockAuditPair(
            examples[groups_per_path * 4],
            scale_episode_time(
                examples[groups_per_path * 4].bundle,
                SuiteName.CLOCK_SCALE_10X,
                "00000000-0000-4000-8000-000000000002",
            ),
        ),
    )
    descriptor = AuditSourceDescriptor(
        schema_version="leakage-source-v1",
        generation_mode="matched",
        allocation_id="test-leakage-v1",
        allocation_or_manifest_sha256="1" * 64,
        split_namespace=SplitNamespace.DEBUG,
        root_seed=41,
        public_id_seed_sha256=public_id_seed_sha256(91),
        config_sha256=_config_sha256(config),
        generator_source_sha256="a" * 64,
        episode_count=len(examples),
    )
    authentication = AuditSourceAuthentication(
        schema_version="leakage-source-auth-v1",
        profile=LeakageAuditProfileName.TEST,
        descriptor_sha256=audit_source_descriptor_sha256(descriptor),
        source_manifest_sha256=_source_manifest_sha256(examples),
        suite_path_denominators={f"iid_primary:{path}": groups_per_path * 4 for path in (2, 3, 4)},
        clock_pair_manifest_sha256=_clock_pair_manifest_sha256(clock_pairs),
        clock_scale_pair_counts={"scale_0_1x": 1, "scale_10x": 1},
    )
    return InMemoryAuditSource(
        descriptor,
        examples,
        validation_config=config,
        authentication=authentication,
        clock_pairs=clock_pairs,
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


def test_counterfactual_result_builder_matches_canonical_sequence_hash() -> None:
    """Full-profile result hashing must not retain 132,000 pair models in memory."""
    from silent_cascade.env.leakage import (
        CounterfactualCheckId,
        CounterfactualPairResult,
        _CounterfactualResultBuilder,
        counterfactual_result_payload_hash,
    )

    pairs = tuple(
        CounterfactualPairResult(
            pair_key_sha256=f"{index:064x}",
            decision_mismatch=index == 1,
            temporal_mismatch=index == 2,
        )
        for index in range(3)
    )
    builder = _CounterfactualResultBuilder(CounterfactualCheckId.PRESENTATION_PERMUTATION)
    for pair in pairs:
        builder.add(pair)

    result = builder.finalize()

    assert result.checked_pairs == 3
    assert result.decision_mismatch_count == 1
    assert result.temporal_mismatch_count == 1
    assert result.result_payload_sha256 == counterfactual_result_payload_hash(
        CounterfactualCheckId.PRESENTATION_PERMUTATION, pairs
    )


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


def test_named_positive_controls_write_the_exact_declared_public_codes() -> None:
    """A one-field encoding drift can create an overlapping or underpowered control."""
    from silent_cascade.env.leakage import (
        NAMED_LEAK_INJECTORS,
        AuditExample,
        _fact_events,
        _rewrite_positive_control,
    )
    from silent_cascade.schemas import ActivationPayload, HazardFact, SafeFact

    config = _config()
    bundles = generate_matched_cohort(
        config,
        CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 41, 17, 3),
        91,
    ).episodes
    examples = tuple(
        AuditExample(bundle, index, "matched", 17, index) for index, bundle in enumerate(bundles)
    )
    by_id = {injector.control_id: injector for injector in NAMED_LEAK_INJECTORS}

    counts = [
        _rewrite_positive_control(by_id["PC_COUNT_BY_LABEL"], example, index)
        for index, example in enumerate(examples)
    ]
    assert [len(_fact_events(item.bundle)) for item in counts] == [48, 56, 48, 56]

    gaps = [
        _rewrite_positive_control(by_id["PC_ACTIVATION_GAP_BY_LABEL"], example, index)
        for index, example in enumerate(examples)
    ]
    assert [
        item.bundle.public.events[-1].timestamp - item.bundle.public.events[-2].timestamp
        for item in gaps
    ] == [1.0, 4.0, 1.0, 4.0]

    terminal_order = [
        _rewrite_positive_control(by_id["PC_TERMINAL_ORDER_BY_VARIANT"], example, index)
        for index, example in enumerate(examples)
    ]
    assert [
        "".join(
            "H" if isinstance(event.payload, HazardFact) else "S"
            for event in _fact_events(item.bundle)[:3]
        )
        for item in terminal_order
    ] == ["HHS", "SHH", "HHS", "HSH"]

    activation_ids = [
        _rewrite_positive_control(by_id["PC_ACTIVATION_ID_BY_LABEL"], example, index)
        for index, example in enumerate(examples)
    ]
    assert [
        item.bundle.public.events[-1].payload.start_node
        for item in activation_ids
        if isinstance(item.bundle.public.events[-1].payload, ActivationPayload)
    ] == [0, 63, 0, 63]

    record_ids = [
        _rewrite_positive_control(by_id["PC_RECORD_ID_BY_VARIANT"], example, index)
        for index, example in enumerate(examples)
    ]
    assert [
        tuple(event.event_id for event in _fact_events(item.bundle))[:2] for item in record_ids
    ] == [
        (0, 1),
        (256, 257),
        (0, 1),
        (128, 129),
    ]

    public_ids = [
        _rewrite_positive_control(by_id["PC_PUBLIC_ID_BY_LABEL"], example, index)
        for index, example in enumerate(examples)
    ]
    assert [
        bytes.fromhex(item.bundle.public.init.episode_public_id.replace("-", ""))[0]
        for item in public_ids
    ] == [
        0,
        255,
        0,
        255,
    ]

    delays = [
        _rewrite_positive_control(by_id["PC_DELAY_BY_LABEL"], example, index)
        for index, example in enumerate(examples)
    ]
    assert [
        {
            event.payload.delay
            for event in _fact_events(item.bundle)
            if isinstance(event.payload, HazardFact)
        }
        for item in delays
    ] == [{1.0}, {1024.0}, {1.0}, {1024.0}]

    for target in range(4):
        layout = _rewrite_positive_control(
            by_id["PC_HAZARD_LAYOUT_BY_CLASS"],
            examples[0],
            target,
            injected_hazard_target=target,
        )
        first_four = _fact_events(layout.bundle)[:4]
        assert isinstance(first_four[target].payload, SafeFact)
        assert sum(isinstance(item.payload, HazardFact) for item in first_four) == 2

    ordered = [
        _rewrite_positive_control(
            by_id["PC_MANIFEST_ORDER_BY_VARIANT"],
            example,
            index,
            encoded_manifest_rank=100 + index,
        )
        for index, example in enumerate(examples)
    ]
    assert [item.manifest_rank for item in ordered] == [100, 101, 102, 103]


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


def test_link_topology_mask_freezes_cycle_flag_and_combined_1560_bit_layout() -> None:
    """A categorical topology flag must never be transformed as a continuous scalar."""
    from silent_cascade.env.leakage import ShortcutFeatureGroup, _continuous_columns

    topology = _continuous_columns(ShortcutFeatureGroup.LINK_TOPOLOGY, 33)
    assert np.array_equal(
        topology,
        np.asarray(
            (
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ),
            dtype=bool,
        ),
    )
    combined = _continuous_columns(ShortcutFeatureGroup.COMBINED, 1560)
    assert combined.shape == (1560,)
    assert not combined[1527 + 4]
    assert hashlib.sha256(np.packbits(combined.astype(np.uint8)).tobytes()).hexdigest() == (
        "6ca18896cbfa1cd90a65b0851758f769b5fd340c94ddfdb956f5517f5cebbe51"
    )


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


def test_hazard_permutations_swap_only_distinct_public_positive_classes() -> None:
    """Leaving hazard targets unchanged makes the declared null distribution degenerate."""
    from silent_cascade.env.episode import EpisodeVariant
    from silent_cascade.env.leakage import ShortcutTask, _permuted_labels, _StoredExample

    rows = tuple(
        _StoredExample(
            public_id=f"00000000-0000-4000-8000-{index:012d}",
            digest="a" * 64,
            group_id="independent:7",
            suite=SuiteName.IID_PRIMARY,
            path_length=3,
            variant=EpisodeVariant.POSITIVE if index < 2 else EpisodeVariant.SAFE_NEGATIVE,
            hazard_class=index if index < 2 else None,
            block=7,
            position=index,
            public_hazard_classes=(0, 1) if index == 0 else ((2, 2) if index == 1 else ()),
        )
        for index in range(4)
    )
    labels = np.asarray((0, 2, -1, -1), dtype=np.int8)
    permutations = {
        tuple(_permuted_labels(rows, labels, ShortcutTask.POSITIVE_HAZARD_CLASS, rep, "b" * 64, 91))
        for rep in range(64)
    }
    assert permutations <= {(0, 2, -1, -1), (1, 2, -1, -1)}
    assert (1, 2, -1, -1) in permutations


def test_independent_coordinate_rejects_forged_source_position() -> None:
    """A quartet index alone does not authenticate an independent episode."""
    from silent_cascade.env.episode import EpisodeVariant
    from silent_cascade.env.generator import IndependentEpisodeRequest, generate_independent_episode
    from silent_cascade.env.leakage import (
        AuditExample,
        AuditSourceDescriptor,
        _validate_audit_coordinate,
    )

    config = _config()
    request = IndependentEpisodeRequest(
        split_namespace=SplitNamespace.DEBUG,
        suite=SuiteName.IID_PRIMARY,
        root_seed=41,
        episode_index=7,
        allocation_quartet_index=1,
        requested_path_length=3,
        variant=EpisodeVariant.POSITIVE,
    )
    bundle = generate_independent_episode(config, request, 91)
    descriptor = AuditSourceDescriptor(
        schema_version="leakage-source-v1",
        generation_mode="independent",
        allocation_id="test-leakage-v1",
        allocation_or_manifest_sha256="f" * 64,
        split_namespace=SplitNamespace.DEBUG,
        root_seed=41,
        public_id_seed_sha256=public_id_seed_sha256(91),
        config_sha256="e" * 64,
        generator_source_sha256="a" * 64,
        episode_count=4,
    )
    with pytest.raises(ValueError, match="independent audit coordinate"):
        _validate_audit_coordinate(AuditExample(bundle, 0, "independent", 1, 99), 0, descriptor)


def test_audit_streams_a_small_complete_source_and_cleans_its_memmaps(tmp_path: Path) -> None:
    """Retaining bundles or omitting a report family must fail this bounded audit contract."""
    from silent_cascade.env.leakage import (
        CounterfactualCheckId,
        LeakageAuditProfileName,
        audit_leakage,
    )

    config = _config()
    source = _authenticated_test_source(config, groups_per_path=20)
    audit_config = config.data.leakage_audit.model_copy(
        update={
            "test": config.data.leakage_audit.test.model_copy(
                update={
                    "episode_count": source.episode_count,
                    "permutation_replicates": 1,
                    "minimum_test_examples_per_class": 1,
                }
            )
        }
    )

    report = audit_leakage(
        source,
        audit_config,
        LeakageAuditProfileName.TEST,
        _provenance(_config_sha256(config)),
        tmp_path,
    )

    assert report.episode_count == 240
    assert len(report.probes) == 27
    assert tuple(item.check_id for item in report.counterfactual_checks) == tuple(
        CounterfactualCheckId
    )
    assert not list(tmp_path.iterdir())


def test_audit_fails_closed_when_workspace_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Suppressing a failed evidence cleanup can leave a publishable partial artifact."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    source = _authenticated_test_source(config, groups_per_path=20)
    audit_config = config.data.leakage_audit.model_copy(
        update={
            "test": config.data.leakage_audit.test.model_copy(
                update={
                    "episode_count": source.episode_count,
                    "permutation_replicates": 1,
                    "minimum_test_examples_per_class": 1,
                }
            )
        }
    )

    def fail_cleanup(_path: Path) -> None:
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(leakage.shutil, "rmtree", fail_cleanup)

    with pytest.raises(RuntimeError, match="cleanup"):
        leakage.audit_leakage(
            source,
            audit_config,
            leakage.LeakageAuditProfileName.TEST,
            _provenance(_config_sha256(config)),
            tmp_path,
        )


def test_feature_store_ceiling_refuses_before_source_iteration(tmp_path: Path) -> None:
    """The exact 1,560-float row budget must be checked before opening the source."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    source = _authenticated_test_source(config, groups_per_path=20)

    class CountingSource:
        descriptor = source.descriptor
        authentication = source.authentication
        validation_config = source.validation_config
        episode_count = source.episode_count
        iter_calls = 0

        def iter_examples(self):
            self.iter_calls += 1
            return source.iter_examples()

        def iter_clock_pairs(self):
            self.iter_calls += 1
            return source.iter_clock_pairs()

    counting = CountingSource()
    required = source.episode_count * 1_560 * np.dtype(np.float32).itemsize
    audit_config = config.data.leakage_audit.model_copy(
        update={
            "max_feature_store_bytes": required - 1,
            "test": config.data.leakage_audit.test.model_copy(
                update={"episode_count": source.episode_count}
            ),
        }
    )

    with pytest.raises(MemoryError, match="feature-store ceiling"):
        leakage.audit_leakage(
            counting,
            audit_config,
            leakage.LeakageAuditProfileName.TEST,
            _provenance(_config_sha256(config)),
            tmp_path,
        )
    assert counting.iter_calls == 0
    assert not list(tmp_path.iterdir())


def test_second_pass_identity_failure_precedes_probe_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regenerating source that truncates pass two must fail before publication."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    source = _authenticated_test_source(config, groups_per_path=20)

    class TruncatingSource:
        descriptor = source.descriptor
        authentication = source.authentication
        validation_config = source.validation_config
        episode_count = source.episode_count
        calls = 0

        def iter_examples(self):
            self.calls += 1
            if self.calls == 2:
                return iter(source.examples[:-1])
            return source.iter_examples()

        def iter_clock_pairs(self):
            return source.iter_clock_pairs()

    def prohibited_fit(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("probe fitting began before second-pass authentication")

    monkeypatch.setattr(leakage, "_run_probes", prohibited_fit)
    audit_config = config.data.leakage_audit.model_copy(
        update={
            "test": config.data.leakage_audit.test.model_copy(
                update={"episode_count": source.episode_count}
            )
        }
    )

    with pytest.raises(ValueError, match="second pass differs"):
        leakage.audit_leakage(
            TruncatingSource(),
            audit_config,
            leakage.LeakageAuditProfileName.TEST,
            _provenance(_config_sha256(config)),
            tmp_path,
        )
    assert not list(tmp_path.iterdir())


def test_audit_rejects_validation_config_digest_before_source_iteration(tmp_path: Path) -> None:
    """Trusting a typed config without hashing it would fit against unauthenticated inputs."""
    from silent_cascade.env.leakage import (
        AuditSourceAuthentication,
        AuditSourceDescriptor,
        LeakageAuditProfileName,
        audit_leakage,
        audit_source_descriptor_sha256,
    )

    config = _config()
    descriptor = AuditSourceDescriptor(
        schema_version="leakage-source-v1",
        generation_mode="matched",
        allocation_id="test-leakage-v1",
        allocation_or_manifest_sha256="1" * 64,
        split_namespace=SplitNamespace.DEBUG,
        root_seed=41,
        public_id_seed_sha256=public_id_seed_sha256(91),
        config_sha256="e" * 64,
        generator_source_sha256="a" * 64,
        episode_count=1_200,
    )
    authentication = AuditSourceAuthentication(
        schema_version="leakage-source-auth-v1",
        profile=LeakageAuditProfileName.TEST,
        descriptor_sha256=audit_source_descriptor_sha256(descriptor),
        source_manifest_sha256="2" * 64,
        suite_path_denominators={
            "iid_primary:2": 400,
            "iid_primary:3": 400,
            "iid_primary:4": 400,
        },
        clock_pair_manifest_sha256="3" * 64,
        clock_scale_pair_counts={"scale_0_1x": 1, "scale_10x": 1},
    )

    class CountingSource:
        validation_config = config
        episode_count = 1_200
        iter_calls = 0

        def __init__(self) -> None:
            self.descriptor = descriptor
            self.authentication = authentication

        def iter_examples(self):
            self.iter_calls += 1
            return iter(())

        def iter_clock_pairs(self):
            self.iter_calls += 1
            return iter(())

    source = CountingSource()
    with pytest.raises(ValueError, match="configuration digest"):
        audit_leakage(
            source,
            config.data.leakage_audit,
            LeakageAuditProfileName.TEST,
            _provenance("e" * 64),
            tmp_path,
        )
    assert source.iter_calls == 0


def test_audit_rejects_forged_authenticated_profile_before_source_iteration(
    tmp_path: Path,
) -> None:
    """A missing TEST stratum in the authentication record must fail before fitting."""
    from silent_cascade.env.leakage import (
        AuditSourceAuthentication,
        AuditSourceDescriptor,
        LeakageAuditProfileName,
        audit_leakage,
        audit_source_descriptor_sha256,
    )

    config = _config()
    config_sha256 = _config_sha256(config)
    descriptor = AuditSourceDescriptor(
        schema_version="leakage-source-v1",
        generation_mode="matched",
        allocation_id="test-leakage-v1",
        allocation_or_manifest_sha256="1" * 64,
        split_namespace=SplitNamespace.DEBUG,
        root_seed=41,
        public_id_seed_sha256=public_id_seed_sha256(91),
        config_sha256=config_sha256,
        generator_source_sha256="a" * 64,
        episode_count=1_200,
    )
    authentication = AuditSourceAuthentication(
        schema_version="leakage-source-auth-v1",
        profile=LeakageAuditProfileName.TEST,
        descriptor_sha256=audit_source_descriptor_sha256(descriptor),
        source_manifest_sha256="2" * 64,
        suite_path_denominators={"iid_primary:2": 1_200},
        clock_pair_manifest_sha256="3" * 64,
        clock_scale_pair_counts={"scale_0_1x": 1, "scale_10x": 1},
    )

    class CountingSource:
        validation_config = config
        episode_count = 1_200
        iter_calls = 0

        def __init__(self) -> None:
            self.descriptor = descriptor
            self.authentication = authentication

        def iter_examples(self):
            self.iter_calls += 1
            return iter(())

        def iter_clock_pairs(self):
            self.iter_calls += 1
            return iter(())

    source = CountingSource()
    with pytest.raises(ValueError, match="profile denominators"):
        audit_leakage(
            source,
            config.data.leakage_audit,
            LeakageAuditProfileName.TEST,
            _provenance(config_sha256),
            tmp_path,
        )
    assert source.iter_calls == 0


def test_audit_rejects_self_reordered_source_before_any_probe_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-ranking a reordered stream must not manufacture new source authentication."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    source = _authenticated_test_source(config, groups_per_path=20)
    reordered = tuple(
        replace(example, manifest_rank=rank)
        for rank, example in enumerate(reversed(source.examples))
    )
    forged = replace(source, examples=reordered)
    audit_config = config.data.leakage_audit.model_copy(
        update={
            "test": config.data.leakage_audit.test.model_copy(
                update={"episode_count": len(reordered)}
            )
        }
    )

    def prohibited_fit(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("probe fitting began before source authentication")

    monkeypatch.setattr(leakage, "_run_probes", prohibited_fit)

    with pytest.raises(ValueError, match="source order or membership"):
        leakage.audit_leakage(
            forged,
            audit_config,
            leakage.LeakageAuditProfileName.TEST,
            _provenance(_config_sha256(config)),
            tmp_path,
        )


def test_terminal_delay_counterfactual_swaps_both_public_hazard_delays() -> None:
    """Rewriting only the reachable hazard would not implement the common-delay swap."""
    from silent_cascade.env.leakage import _fact_events, _with_terminal_delay
    from silent_cascade.env.oracle import solve_public_episode
    from silent_cascade.schemas import HazardFact

    config = _config()
    left = generate_matched_cohort(
        config,
        CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 41, 1, 2),
        91,
    ).episodes[0]
    right = generate_matched_cohort(
        config,
        CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 41, 2, 2),
        91,
    ).episodes[0]
    right_delay = solve_public_episode(right.public).public_delay
    assert right_delay is not None

    swapped = _with_terminal_delay(left, right_delay)

    assert {
        event.payload.delay
        for event in _fact_events(type("BundleView", (), {"public": swapped})())
        if isinstance(event.payload, HazardFact)
    } == {right_delay}
    assert solve_public_episode(swapped).public_delay == right_delay


def test_counterfactual_engine_consumes_only_declared_paired_clock_children() -> None:
    """Synthesizing two clock children per base row violates the authenticated pair profile."""
    from silent_cascade.env.episode import scale_episode_time
    from silent_cascade.env.leakage import (
        AuditExample,
        AuditSourceAuthentication,
        AuditSourceDescriptor,
        CounterfactualCheckId,
        InMemoryAuditSource,
        LeakageAuditProfileName,
        PairedClockAuditPair,
        _clock_pair_manifest_sha256,
        _counterfactual_checks,
        _source_manifest_sha256,
        audit_source_descriptor_sha256,
    )

    config = _config()
    examples = tuple(
        AuditExample(bundle, rank, "matched", path_index, member_index)
        for path_index, path in enumerate((2, 3, 4))
        for member_index, bundle in enumerate(
            generate_matched_cohort(
                config,
                CohortRequest(
                    SplitNamespace.DEBUG,
                    SuiteName.IID_PRIMARY,
                    41,
                    path_index,
                    path,
                ),
                91,
            ).episodes
        )
        for rank in (path_index * 4 + member_index,)
    )
    clock_pairs = (
        PairedClockAuditPair(
            examples[0],
            scale_episode_time(
                examples[0].bundle,
                SuiteName.CLOCK_SCALE_0_1X,
                "00000000-0000-4000-8000-000000000001",
            ),
        ),
        PairedClockAuditPair(
            examples[4],
            scale_episode_time(
                examples[4].bundle,
                SuiteName.CLOCK_SCALE_10X,
                "00000000-0000-4000-8000-000000000002",
            ),
        ),
    )
    descriptor = AuditSourceDescriptor(
        schema_version="leakage-source-v1",
        generation_mode="matched",
        allocation_id="test-leakage-v1",
        allocation_or_manifest_sha256="1" * 64,
        split_namespace=SplitNamespace.DEBUG,
        root_seed=41,
        public_id_seed_sha256=public_id_seed_sha256(91),
        config_sha256=_config_sha256(config),
        generator_source_sha256="a" * 64,
        episode_count=len(examples),
    )
    authentication = AuditSourceAuthentication(
        schema_version="leakage-source-auth-v1",
        profile=LeakageAuditProfileName.TEST,
        descriptor_sha256=audit_source_descriptor_sha256(descriptor),
        source_manifest_sha256=_source_manifest_sha256(examples),
        suite_path_denominators={f"iid_primary:{path}": 4 for path in (2, 3, 4)},
        clock_pair_manifest_sha256=_clock_pair_manifest_sha256(clock_pairs),
        clock_scale_pair_counts={"scale_0_1x": 1, "scale_10x": 1},
    )
    source = InMemoryAuditSource(
        descriptor,
        examples,
        validation_config=config,
        authentication=authentication,
        clock_pairs=clock_pairs,
    )

    checks = _counterfactual_checks(source, LeakageAuditProfileName.TEST)

    assert {item.check_id: item.checked_pairs for item in checks} == {
        CounterfactualCheckId.TERMINAL_DELAY_SWAP: 3,
        CounterfactualCheckId.PRESENTATION_PERMUTATION: 12,
        CounterfactualCheckId.PAIRED_CLOCK_SCALE: 2,
    }
    assert all(item.passed for item in checks)


@pytest.mark.parametrize("control_index", range(9))
def test_named_positive_control_executes_exact_isolated_detector_end_to_end(
    control_index: int,
    tmp_path: Path,
) -> None:
    """A registry-only or pre-injected malformed control would never test detector power."""
    from silent_cascade.env.leakage import (
        NAMED_LEAK_INJECTORS,
        LeakageAuditProfileName,
        audit_leakage,
    )

    config = _config()
    clean_source = _authenticated_test_source(config)
    injector = NAMED_LEAK_INJECTORS[control_index]
    audit_config = config.data.leakage_audit.model_copy(
        update={
            "test": config.data.leakage_audit.test.model_copy(
                update={
                    "episode_count": clean_source.episode_count,
                    "permutation_replicates": 1,
                    "positive_control_episode_count": clean_source.episode_count,
                    "positive_control_permutation_replicates": 19,
                    "minimum_test_examples_per_class": 1,
                }
            )
        }
    )

    report = audit_leakage(
        injector.apply(clean_source),
        audit_config,
        LeakageAuditProfileName.TEST,
        _provenance(_config_sha256(config)),
        tmp_path,
        positive_control=injector.control_id,
    )

    assert not report.passed
    assert len(report.positive_controls) == 1
    result = report.positive_controls[0]
    assert result.control_id == injector.control_id
    assert result.target_task is injector.target_task
    assert result.observed_detector_ids == (injector.expected_detector_id,), result
    assert result.base_subset_corpus_sha256 != result.injected_corpus_sha256
    assert result.balanced_accuracy is not None and result.balanced_accuracy >= 0.95
    assert result.passed


def test_frozen_optimizer_fits_three_standardized_contiguous_rank_blocks() -> None:
    """The manifest-order KAT isolates weighting, preprocessing, and the frozen L2 fit."""
    from silent_cascade.env.leakage import (
        ShortcutFeatureGroup,
        _balanced_accuracy,
        _fit_predict,
        _standardize,
    )

    count = 1_200
    values = np.zeros((count, 17), dtype=np.float32)
    values[:, -1] = np.arange(count, dtype=np.float32) / (count - 1)
    labels = np.repeat(np.arange(3, dtype=np.int8), count // 3)
    test = np.arange(0, count, 5, dtype=np.int64)
    train = np.setdiff1d(np.arange(count, dtype=np.int64), test, assume_unique=True)
    x_train, x_test = _standardize(values[train], values[test], ShortcutFeatureGroup.ID_POSITION)

    predictions, _iterations = _fit_predict(
        x_train,
        labels[train],
        x_test,
        np.arange(3, dtype=np.int8),
        _config().data.leakage_audit,
    )

    assert _balanced_accuracy(labels[test], predictions, np.arange(3)) >= 0.95


def test_batched_optimizer_matches_dense_fit_without_oversized_reads() -> None:
    """A full-row advanced-index or float64 materialization would violate this KAT."""
    from silent_cascade.env.leakage import (
        ShortcutFeatureGroup,
        _BatchedFeatureReader,
        _fit_predict,
        _fit_predict_batched,
        _standardize,
    )

    row_count = 60
    values = np.asarray(
        [
            (
                float(index % 2),
                float((index // 2) % 3),
                float(index) / row_count,
                float((index * index) % 11),
            )
            for index in range(row_count)
        ],
        dtype=np.float32,
    )
    labels = np.asarray([index % 3 for index in range(row_count)], dtype=np.int8)
    train = np.arange(0, 48, dtype=np.int64)
    test = np.arange(48, row_count, dtype=np.int64)
    classes = np.arange(3, dtype=np.int8)
    config = _config().data.leakage_audit.model_copy(update={"feature_batch_size": 7})
    continuous = np.ones(values.shape[1], dtype=bool)
    dense_train, dense_test = _standardize(
        values[train],
        values[test],
        ShortcutFeatureGroup.COUNTS,
        continuous_mask=continuous,
    )
    expected, _ = _fit_predict(dense_train, labels[train], dense_test, classes, config)
    reader = _BatchedFeatureReader(
        values,
        0,
        values.shape[1],
        None,
        config.feature_batch_size,
        config,
    )

    actual, _ = _fit_predict_batched(reader, train, labels, test, classes, continuous, config)

    assert np.array_equal(actual, expected)
    assert reader.max_batch_seen <= config.feature_batch_size


def test_full_profile_feature_shape_streams_under_resident_ceiling(tmp_path: Path) -> None:
    """The 624 MB full-profile store must execute moments below the 512 MB RSS gate."""
    import psutil

    from silent_cascade.env.leakage import (
        _batched_moments,
        _BatchedFeatureReader,
    )

    config = _config().data.leakage_audit
    feature_path = tmp_path / "full-profile.f32"
    values = np.memmap(
        feature_path,
        dtype=np.float32,
        mode="w+",
        shape=(100_000, 1_560),
    )
    reader = _BatchedFeatureReader(
        values,
        0,
        1_560,
        None,
        config.feature_batch_size,
        config,
    )

    mean, std, zero = _batched_moments(
        reader,
        np.arange(100_000, dtype=np.int64),
        np.ones(1_560, dtype=bool),
    )

    assert feature_path.stat().st_size == 624_000_000
    assert np.array_equal(mean, np.zeros(1_560))
    assert np.array_equal(std, np.ones(1_560))
    assert np.all(zero)
    assert reader.max_batch_seen <= 4_096
    assert psutil.Process().memory_info().rss <= config.max_resident_working_bytes
