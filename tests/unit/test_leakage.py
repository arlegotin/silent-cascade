"""Contracts for the fail-closed public-feature leakage auditor."""

import ast
import hashlib
import inspect
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config, SplitNamespace, SuiteName
from silent_cascade.env.generator import CohortRequest, generate_matched_cohort
from silent_cascade.errors import ProvenanceError
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.provenance import (
    GENERATOR_SOURCE_PATHS,
    PHASE1_ANALYSIS_SOURCE_PATHS,
    EvidenceProvenance,
    SourceTreeFingerprint,
    build_construction_namespace_evidence,
    public_id_seed_sha256,
)


def _quartet_descriptor(split: SplitNamespace = SplitNamespace.DEBUG):
    from silent_cascade.env.leakage import AuditSourceDescriptor

    return AuditSourceDescriptor(
        schema_version="leakage-source-v1",
        generation_mode="independent",
        allocation_id="test-quartet-v1",
        allocation_or_manifest_sha256="1" * 64,
        split_namespace=split,
        root_seed=41,
        public_id_seed_sha256=public_id_seed_sha256(91),
        config_sha256="2" * 64,
        generator_source_sha256="3" * 64,
        episode_count=4,
    )


def _nonaligned_quartet_examples():
    from silent_cascade.env.generator import IndependentEpisodeRequest, generate_independent_episode
    from silent_cascade.env.leakage import AuditExample
    from silent_cascade.rng import AllocationLabelKey, allocate_independent_variants

    variants = allocate_independent_variants(
        AllocationLabelKey("ofd-v1", SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 41, 3, 7)
    )
    return tuple(
        AuditExample(
            generate_independent_episode(
                _config(),
                IndependentEpisodeRequest(
                    SplitNamespace.DEBUG,
                    SuiteName.IID_PRIMARY,
                    41,
                    5 + member,
                    3,
                    variant,
                    7,
                    member,
                ),
                91,
            ),
            member,
            "independent",
            7,
            5 + member,
            quartet_member_index=member,
        )
        for member, variant in enumerate(variants)
    )


def _quartet_rows(examples):
    from silent_cascade.env.episode import episode_sha256
    from silent_cascade.env.leakage import _StoredExample

    return [
        _StoredExample(
            public_id=example.bundle.public.init.episode_public_id,
            digest=episode_sha256(example.bundle),
            group_id="independent:7",
            suite=example.bundle.truth.key.suite,
            path_length=example.bundle.truth.recipe.requested_path_length,
            variant=example.bundle.truth.recipe.variant,
            hazard_class=example.bundle.truth.relevant_hazard_type,
            block=example.randomization_block_index,
            position=example.episode_position,
            quartet_member_index=example.quartet_member_index,
        )
        for example in examples
    ]


def test_audit_authenticates_nonaligned_explicit_quartet() -> None:
    from silent_cascade.env.leakage import (
        _validate_audit_coordinate,
        _validate_independent_quartets,
    )

    examples = _nonaligned_quartet_examples()
    assert tuple((e.episode_position, e.quartet_member_index) for e in examples) == (
        (5, 0),
        (6, 1),
        (7, 2),
        (8, 3),
    )
    for rank, example in enumerate(examples):
        _validate_audit_coordinate(example, rank, _quartet_descriptor())
    _validate_independent_quartets(list(reversed(_quartet_rows(examples))), _quartet_descriptor())


@pytest.mark.parametrize("mutation", ("coordinate", "allocation_label"))
def test_audit_member_is_bound_to_coordinate_and_allocation(mutation: str) -> None:
    from silent_cascade.env.leakage import _validate_audit_coordinate

    example = _nonaligned_quartet_examples()[0]
    if mutation == "coordinate":
        example = replace(example, quartet_member_index=1)
    else:
        # Keep a real artifact and variant; agree on a member allocated another label.
        other = next(
            e
            for e in _nonaligned_quartet_examples()
            if e.bundle.truth.recipe.variant != example.bundle.truth.recipe.variant
        )
        truth = example.bundle.truth
        member = other.quartet_member_index
        coordinate = replace(truth.key.coordinate, quartet_member_index=member)
        example = replace(
            example,
            quartet_member_index=member,
            bundle=replace(
                example.bundle, truth=replace(truth, key=replace(truth.key, coordinate=coordinate))
            ),
        )
    with pytest.raises(ValueError, match="independent audit"):
        _validate_audit_coordinate(example, 0, _quartet_descriptor())


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate_member",
        "five_rows",
        "eight_rows",
        "gap",
        "member_order",
        "path",
        "suite",
        "group",
        "allocation_label",
    ),
)
def test_audit_quartet_rejects_structural_and_label_forgery(mutation: str) -> None:
    from silent_cascade.env.leakage import _validate_independent_quartets

    rows = _quartet_rows(_nonaligned_quartet_examples())
    if mutation == "duplicate_member":
        rows[3] = replace(rows[3], quartet_member_index=0)
    elif mutation in {"five_rows", "eight_rows"}:
        rows += [
            replace(row, position=row.position + 4)
            for row in rows[: 1 if mutation == "five_rows" else 4]
        ]
    elif mutation == "gap":
        rows[3] = replace(rows[3], position=9)
    elif mutation == "member_order":
        rows[0], rows[1] = replace(rows[0], position=6), replace(rows[1], position=5)
    elif mutation == "path":
        rows[3] = replace(rows[3], path_length=4)
    elif mutation == "suite":
        rows[3] = replace(rows[3], suite=SuiteName.OOD_LONG_DELAY)
    elif mutation == "group":
        rows[3] = replace(rows[3], group_id="independent:8")
    else:
        different = next(index for index, row in enumerate(rows) if row.variant != rows[0].variant)
        rows[0], rows[different] = (
            replace(rows[0], variant=rows[different].variant),
            replace(rows[different], variant=rows[0].variant),
        )
    with pytest.raises(ValueError, match="independent allocation quartet"):
        _validate_independent_quartets(rows, _quartet_descriptor())


@pytest.mark.parametrize("member", (False, True, -1, 4))
def test_audit_and_stored_members_reject_non_exact_indices(member: object) -> None:
    example = _nonaligned_quartet_examples()[0]
    row = _quartet_rows((example,))[0]
    for value in (example, row):
        with pytest.raises((TypeError, ValueError), match="quartet_member_index"):
            replace(value, quartet_member_index=member)


def test_audit_and_stored_members_are_mandatory_keyword_only() -> None:
    from dataclasses import MISSING, fields

    from silent_cascade.env.leakage import AuditExample, _StoredExample

    for cls in (AuditExample, _StoredExample):
        members = {field.name: field for field in fields(cls)}
        assert "quartet_member_index" in members
        member = members["quartet_member_index"]
        assert member.kw_only and member.default is MISSING and member.default_factory is MISSING

    example = _nonaligned_quartet_examples()[0]
    row = _quartet_rows((example,))[0]
    for value in (example, row):
        arguments = {
            field.name: getattr(value, field.name)
            for field in fields(value)
            if field.name != "quartet_member_index"
        }
        with pytest.raises(TypeError, match="quartet_member_index"):
            type(value)(**arguments)


def test_matched_audit_member_agrees_with_private_coordinate() -> None:
    from silent_cascade.env.leakage import AuditExample, _validate_audit_coordinate

    bundle = generate_matched_cohort(
        _config(), CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 41, 7, 3), 91
    ).episodes[0]
    example = AuditExample(bundle, 0, "matched", 7, 0, quartet_member_index=1)
    descriptor = _quartet_descriptor().model_copy(update={"generation_mode": "matched"})
    with pytest.raises(ValueError, match="matched audit coordinate"):
        _validate_audit_coordinate(example, 0, descriptor)


def test_quartet_validation_never_recovers_members_from_absolute_indices() -> None:
    from silent_cascade.env.leakage import (
        _validate_audit_coordinate,
        _validate_independent_quartets,
    )

    for validate in (_validate_audit_coordinate, _validate_independent_quartets):
        tree = ast.parse(inspect.getsource(validate))
        assert not any(isinstance(node, (ast.Mod, ast.FloorDiv)) for node in ast.walk(tree))
        assert not any(
            isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "divmod")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "divmod")
            )
            for node in ast.walk(tree)
        )


@pytest.mark.parametrize(
    ("member", "source_digest", "clock_digest"),
    (
        (
            0,
            "471303a236c702d0c5e40ef449fc6eee0aca96d7de2c1e7a255084d7ef487cb5",
            "3b255cd73bf2bc8140cdc949a8b8fd268fb4ad1218c5d03868eb640f406b1b4f",
        ),
        (
            1,
            "d648a3c39fe7c8ba6dea2a375b54f52754498ecf08fd72af1e6744a11dba8303",
            "47cc5f757a306567ee763fca6d6b594b05e8a909a26d0fada71b9a29c0585780",
        ),
    ),
)
def test_explicit_member_has_independent_source_and_clock_hash_known_answers(
    member: int, source_digest: str, clock_digest: str
) -> None:
    from silent_cascade.env.episode import scale_episode_time
    from silent_cascade.env.leakage import (
        PairedClockAuditPair,
        _clock_pair_manifest_sha256,
        _source_manifest_sha256,
    )

    # Literal answers derive from independently framed, sorted compact JSON.
    example = replace(_nonaligned_quartet_examples()[0], quartet_member_index=member)
    pair = PairedClockAuditPair(
        example,
        scale_episode_time(
            example.bundle,
            SuiteName.CLOCK_SCALE_0_1X,
            "00000000-0000-4000-8000-000000000011",
        ),
    )
    assert _source_manifest_sha256((example,)) == source_digest
    assert _clock_pair_manifest_sha256((pair,)) == clock_digest


def _config() -> Phase1Config:
    return resolve_config(
        Phase1Config, [Path("configs/base.yaml"), Path("configs/data/primary.yaml")]
    ).config


def _provenance(config_sha256: str, source: object | None = None) -> EvidenceProvenance:
    leakage_audit = None
    if source is not None:
        from silent_cascade.provenance import LeakageAuditEvidenceAnchor

        descriptor = source.descriptor
        authentication = source.authentication
        leakage_audit = LeakageAuditEvidenceAnchor(
            schema_version="phase1-leakage-audit-anchor-v2",
            profile=authentication.profile.value,
            allocation_id=descriptor.allocation_id,
            allocation_or_manifest_sha256=descriptor.allocation_or_manifest_sha256,
            config_sha256=descriptor.config_sha256,
            descriptor_sha256=authentication.descriptor_sha256,
            source_manifest_sha256=authentication.source_manifest_sha256,
            suite_path_denominators=authentication.suite_path_denominators,
            clock_pair_manifest_sha256=authentication.clock_pair_manifest_sha256,
            clock_scale_pair_counts=authentication.clock_scale_pair_counts,
            episode_count=descriptor.episode_count,
        )
    return EvidenceProvenance(
        schema_version="phase1-evidence-provenance-v1",
        plan_base_revision="c" * 40,
        source_commit="d" * 40,
        source_dirty=False,
        generator_version="ofd-v1",
        generation_mode=(source.descriptor.generation_mode if source is not None else "matched"),
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
        leakage_audit=leakage_audit,
    )


def _config_sha256(config: Phase1Config) -> str:
    return sha256_bytes(canonical_json_bytes(config))


def _authenticated_test_source(
    config: Phase1Config,
    groups_per_path: int = 100,
    clock_counts: tuple[int, int] = (1, 1),
):
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
        AuditExample(
            bundle, rank, "matched", cohort_index, member_index, quartet_member_index=member_index
        )
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
    clock_pairs = tuple(
        PairedClockAuditPair(
            examples[parent_rank],
            scale_episode_time(
                examples[parent_rank].bundle,
                suite,
                f"00000000-0000-4000-8000-{identifier:012x}",
            ),
        )
        for suite, count, first_parent, first_identifier in (
            (SuiteName.CLOCK_SCALE_0_1X, clock_counts[0], 0, 1),
            (SuiteName.CLOCK_SCALE_10X, clock_counts[1], groups_per_path * 4, 2),
        )
        for local in range(count)
        for parent_rank, identifier in ((first_parent + local, first_identifier + 2 * local),)
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
        schema_version="leakage-source-auth-v2",
        profile=LeakageAuditProfileName.TEST,
        descriptor_sha256=audit_source_descriptor_sha256(descriptor),
        source_manifest_sha256=_source_manifest_sha256(examples),
        suite_path_denominators={f"iid_primary:{path}": groups_per_path * 4 for path in (2, 3, 4)},
        clock_pair_manifest_sha256=_clock_pair_manifest_sha256(clock_pairs),
        clock_scale_pair_counts={
            "scale_0_1x": clock_counts[0],
            "scale_10x": clock_counts[1],
        },
    )
    return InMemoryAuditSource(
        descriptor,
        examples,
        validation_config=config,
        authentication=authentication,
        clock_pairs=clock_pairs,
        public_id_seed=91,
    )


def _authenticated_independent_test_source(
    config: Phase1Config,
    groups_per_path: int = 20,
):
    from silent_cascade.env.episode import scale_episode_time
    from silent_cascade.env.generator import (
        IndependentEpisodeRequest,
        generate_independent_episode,
    )
    from silent_cascade.env.invariants import _allocation_variants
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

    examples = []
    for path_index, path in enumerate((2, 3, 4)):
        for local_group in range(groups_per_path):
            block = path_index * groups_per_path + local_group
            variants = _allocation_variants(
                SplitNamespace.DEBUG,
                SuiteName.IID_PRIMARY,
                41,
                path,
                block,
            )
            for local_position, variant in enumerate(variants):
                episode_index = block * 4 + local_position
                bundle = generate_independent_episode(
                    config,
                    IndependentEpisodeRequest(
                        split_namespace=SplitNamespace.DEBUG,
                        suite=SuiteName.IID_PRIMARY,
                        root_seed=41,
                        episode_index=episode_index,
                        requested_path_length=path,
                        variant=variant,
                        allocation_quartet_index=block,
                        quartet_member_index=local_position,
                    ),
                    91,
                )
                examples.append(
                    AuditExample(
                        bundle,
                        episode_index,
                        "independent",
                        block,
                        episode_index,
                        quartet_member_index=local_position,
                    )
                )
    clock_pairs = (
        PairedClockAuditPair(
            examples[0],
            scale_episode_time(
                examples[0].bundle,
                SuiteName.CLOCK_SCALE_0_1X,
                "00000000-0000-4000-8000-000000000011",
            ),
        ),
        PairedClockAuditPair(
            examples[groups_per_path * 4],
            scale_episode_time(
                examples[groups_per_path * 4].bundle,
                SuiteName.CLOCK_SCALE_10X,
                "00000000-0000-4000-8000-000000000012",
            ),
        ),
    )
    descriptor = AuditSourceDescriptor(
        schema_version="leakage-source-v1",
        generation_mode="independent",
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
        schema_version="leakage-source-auth-v2",
        profile=LeakageAuditProfileName.TEST,
        descriptor_sha256=audit_source_descriptor_sha256(descriptor),
        source_manifest_sha256=_source_manifest_sha256(examples),
        suite_path_denominators={f"iid_primary:{path}": groups_per_path * 4 for path in (2, 3, 4)},
        clock_pair_manifest_sha256=_clock_pair_manifest_sha256(clock_pairs),
        clock_scale_pair_counts={"scale_0_1x": 1, "scale_10x": 1},
    )
    return InMemoryAuditSource(
        descriptor,
        tuple(examples),
        validation_config=config,
        authentication=authentication,
        clock_pairs=clock_pairs,
        public_id_seed=91,
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
    features = extract_shortcut_features(
        AuditExample(
            bundle,
            3,
            "matched",
            7,
            0,
            quartet_member_index=bundle.truth.key.coordinate.member_index,
        ),
        corpus_size=8,
    )

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
    assert {
        group: hashlib.sha256(vector.tobytes()).hexdigest()
        for group, vector in features.vectors.items()
    } == {
        ShortcutFeatureGroup.ID_POSITION: (
            "775d241019fbf62e480f652e3ed5b251f44a09e7cb0eac31fec3878ab6e303e2"
        ),
        ShortcutFeatureGroup.ACTIVATION_NODE: (
            "aaefc8ee7169331756b5ac8d16815f8d89898d44ec000a0caf7b73deb74cf34f"
        ),
        ShortcutFeatureGroup.FIRST_LAST_FACT: (
            "a9303ef175356511290714bfdd202e9eb04af319f19362f370d8a39b91210eaa"
        ),
        ShortcutFeatureGroup.COUNTS: (
            "b6b1e7da7520d24132d49e193135f085758738d67b32a47af877d1d162a875cd"
        ),
        ShortcutFeatureGroup.ORDER_RECORD_IDS: (
            "d56ef376688172a240ff4fbae1cf0364957192b2a3fe4da859184cd4801fcc83"
        ),
        ShortcutFeatureGroup.TIMES: (
            "666e935fced16b57bd17d603c3ec9b9e3418f96d8b99e2b1dd211118083a905a"
        ),
        ShortcutFeatureGroup.TERMINAL_MULTISET: (
            "64cb3f6e3b13d2a942bd9e78c479ce4b318fe956c24f8d50c06f048493ef85db"
        ),
        ShortcutFeatureGroup.LINK_TOPOLOGY: (
            "2bf69b460adc292b2ac0fe745c81a5337557f8955a9278c841359f0b57deb153"
        ),
        ShortcutFeatureGroup.COMBINED: (
            "cc0e5c1a40857b6b8458a21ed2d001081e42012325ea27b1500aa5d09ff05484"
        ),
    }
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


def test_feature_extraction_ignores_private_targets_and_rejects_unknown_payloads() -> None:
    """Private truth is forbidden input and public payload vocabulary is closed."""
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
    example = AuditExample(
        bundle, 0, "matched", 7, 0, quartet_member_index=bundle.truth.key.coordinate.member_index
    )
    baseline = extract_shortcut_features(example, corpus_size=8)
    private_mutation = replace(
        bundle,
        truth=replace(
            bundle.truth,
            rejection_reasons=(*bundle.truth.rejection_reasons, "private-only-mutation"),
        ),
    )
    mutated = extract_shortcut_features(
        replace(example, bundle=private_mutation),
        corpus_size=8,
    )
    for group in ShortcutFeatureGroup:
        assert np.array_equal(baseline.vectors[group], mutated.vectors[group])

    with pytest.raises(ValueError, match="payload"):
        replace(bundle.public.events[0], payload=object())


@pytest.mark.parametrize(
    ("mutation", "expected_changed"),
    (
        ("public_id", {"id_position"}),
        ("manifest_rank", {"id_position"}),
        ("activation", {"activation_node"}),
        ("timestamp", {"times"}),
    ),
)
def test_public_feature_mutations_change_only_the_declared_vocabulary_family(
    mutation: str,
    expected_changed: set[str],
) -> None:
    """Targeted public changes must not leak into unrelated feature vocabularies."""
    from silent_cascade.env.leakage import (
        AuditExample,
        ShortcutFeatureGroup,
        extract_shortcut_features,
    )
    from silent_cascade.schemas import ActivationPayload

    bundle = generate_matched_cohort(
        _config(),
        CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 41, 7, 3),
        91,
    ).episodes[0]
    example = AuditExample(
        bundle, 3, "matched", 7, 0, quartet_member_index=bundle.truth.key.coordinate.member_index
    )
    if mutation == "public_id":
        mutated = replace(
            example,
            bundle=replace(
                bundle,
                public=replace(
                    bundle.public,
                    init=replace(
                        bundle.public.init,
                        episode_public_id="ffffffff-ffff-4fff-bfff-ffffffffffff",
                    ),
                ),
            ),
        )
    elif mutation == "manifest_rank":
        mutated = replace(example, manifest_rank=4)
    elif mutation == "activation":
        activation = bundle.public.events[-1]
        assert isinstance(activation.payload, ActivationPayload)
        mutated = replace(
            example,
            bundle=replace(
                bundle,
                public=replace(
                    bundle.public,
                    events=(
                        *bundle.public.events[:-1],
                        replace(
                            activation,
                            payload=ActivationPayload((activation.payload.start_node + 1) % 64),
                        ),
                    ),
                ),
            ),
        )
    else:
        first = bundle.public.events[0]
        mutated = replace(
            example,
            bundle=replace(
                bundle,
                public=replace(
                    bundle.public,
                    events=(
                        replace(first, timestamp=first.timestamp + 1.0e-6),
                        *bundle.public.events[1:],
                    ),
                ),
            ),
        )
    baseline = extract_shortcut_features(example, corpus_size=8)
    changed = extract_shortcut_features(mutated, corpus_size=8)

    assert {
        group.value
        for group in ShortcutFeatureGroup
        if group is not ShortcutFeatureGroup.COMBINED
        and not np.array_equal(baseline.vectors[group], changed.vectors[group])
    } == expected_changed


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


def test_phase1_gate_schedules_every_named_positive_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The acceptance profile cannot silently substitute fixture-scale control evidence."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    source = _authenticated_test_source(config, groups_per_path=20)
    observed: list[str] = []

    def execute(
        injected_source: object,
        _rows: object,
        injector: object,
        _config: object,
        profile: object,
        _corpus_hash: object,
        _validation_config: object,
        _work: object,
    ) -> leakage.PositiveControlResult:
        assert profile is config.data.leakage_audit.phase1_gate
        assert injected_source.injector == injector  # type: ignore[attr-defined]
        observed.append(injector.control_id)  # type: ignore[attr-defined]
        expected_group = injector.expected_detector_id.split(":", 1)[1]  # type: ignore[attr-defined]
        chance = {
            leakage.ShortcutTask.POSITIVE_BINARY: 0.5,
            leakage.ShortcutTask.VARIANT_THREE_WAY: 1.0 / 3.0,
            leakage.ShortcutTask.POSITIVE_HAZARD_CLASS: 0.25,
        }[injector.target_task]  # type: ignore[attr-defined]
        return leakage.PositiveControlResult(
            control_id=injector.control_id,  # type: ignore[attr-defined]
            target_task=injector.target_task,  # type: ignore[attr-defined]
            expected_detector_id=injector.expected_detector_id,  # type: ignore[attr-defined]
            observed_detector_ids=(injector.expected_detector_id,),  # type: ignore[attr-defined]
            base_subset_corpus_sha256="1" * 64,
            injected_corpus_sha256="2" * 64,
            split_membership_sha256="3" * 64,
            balanced_accuracy=1.0,
            holm_adjusted_p=9 * 0.0002,
            probes=tuple(
                leakage.ShortcutProbeResult(
                    task=injector.target_task,  # type: ignore[attr-defined]
                    feature_group=group,
                    feature_dimension=1,
                    train_examples=6_400,
                    test_examples=1_600,
                    train_class_counts={"0": 3_200, "1": 3_200},
                    test_class_counts={"0": 800, "1": 800},
                    raw_accuracy=1.0 if group.value == expected_group else chance,
                    balanced_accuracy=1.0 if group.value == expected_group else chance,
                    balanced_chance=chance,
                    raw_permutation_p=0.0002 if group.value == expected_group else 1.0,
                    holm_adjusted_p=(9 * 0.0002 if group.value == expected_group else 1.0),
                    optimizer_iterations=1,
                    optimizer_converged=True,
                    passed=group.value != expected_group,
                )
                for group in leakage.ShortcutFeatureGroup
            ),
            passed=True,
        )

    monkeypatch.setattr(leakage, "_execute_positive_control", execute)

    controls = leakage._execute_required_positive_controls(
        source,
        (),
        config.data.leakage_audit,
        leakage.LeakageAuditProfileName.PHASE1_GATE,
        config.data.leakage_audit.phase1_gate,
        "4" * 64,
        config,
        tmp_path,
    )

    assert tuple(observed) == tuple(
        injector.control_id for injector in leakage.NAMED_LEAK_INJECTORS
    )
    assert tuple(item.control_id for item in controls) == tuple(observed)


def test_holm_adjustment_has_an_exact_known_answer() -> None:
    from silent_cascade.env.leakage import _holm_adjusted_p_values

    assert _holm_adjusted_p_values((0.01, 0.04, 0.03)) == (0.03, 0.06, 0.06)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("episode_count", 99_996),
        ("permutation_replicates", 4_998),
        ("positive_control_episode_count", 7_980),
        ("positive_control_permutation_replicates", 4_998),
        ("minimum_test_examples_per_class", 199),
        ("enforce_clean_statistical_gate", False),
    ],
)
def test_phase1_gate_rejects_noncanonical_profile_before_source_access(
    field: str,
    replacement: int | bool,
    tmp_path: Path,
) -> None:
    import silent_cascade.env.leakage as leakage

    config = _config()
    phase1_gate = config.data.leakage_audit.phase1_gate.model_copy(update={field: replacement})
    audit_config = config.data.leakage_audit.model_copy(update={"phase1_gate": phase1_gate})

    class ProhibitedSource:
        @property
        def descriptor(self) -> object:
            raise AssertionError("noncanonical profile reached source authentication")

    with pytest.raises(ValueError, match="phase1 gate profile is not exact"):
        leakage.audit_leakage(
            ProhibitedSource(),  # type: ignore[arg-type]
            audit_config,
            leakage.LeakageAuditProfileName.PHASE1_GATE,
            _provenance(_config_sha256(config)),
            tmp_path,
        )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("schema_version", "leakage-v2"),
        ("audit_seed", 0),
        ("positive_control_seed", 0),
        ("train_fraction", 0.5),
        ("alpha", 1.0),
        ("l2_penalty", 0.0),
        ("optimizer_max_iterations", 499),
        ("optimizer_gradient_tolerance", 1.0),
        ("optimizer_function_tolerance", 1.0),
        ("positive_control_min_balanced_accuracy", 0.0),
        ("feature_batch_size", 1),
        ("permutation_batch_size", 1),
        ("max_feature_store_bytes", 1_000_000_000_000),
        ("max_resident_working_bytes", 1_000_000_000_000),
    ],
)
def test_phase1_gate_rejects_noncanonical_audit_config_before_source_access(
    field: str,
    replacement: str | int | float,
    tmp_path: Path,
) -> None:
    import silent_cascade.env.leakage as leakage

    config = _config()
    audit_config = config.data.leakage_audit.model_copy(update={field: replacement})

    class ProhibitedSource:
        @property
        def descriptor(self) -> object:
            raise AssertionError("noncanonical audit config reached source authentication")

    with pytest.raises(ValueError, match="phase1 gate audit config is not exact"):
        leakage.audit_leakage(
            ProhibitedSource(),  # type: ignore[arg-type]
            audit_config,
            leakage.LeakageAuditProfileName.PHASE1_GATE,
            _provenance(_config_sha256(config)),
            tmp_path,
        )


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
        AuditExample(bundle, index, "matched", 17, index, quartet_member_index=index)
        for index, bundle in enumerate(bundles)
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


def test_count_control_preserves_existing_records_activation_and_initial_time() -> None:
    """Count encodings may add only unreachable padding LINKs to a short episode."""
    from silent_cascade.env.leakage import (
        NAMED_LEAK_INJECTORS,
        AuditExample,
        _fact_events,
        _rewrite_positive_control,
    )
    from silent_cascade.schemas import ActivationPayload, LinkFact

    config = _config()
    bundles = generate_matched_cohort(
        config,
        CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 0, 0, 2),
        91,
    ).episodes
    from silent_cascade.env.episode import EpisodeVariant

    bundle = next(item for item in bundles if item.truth.recipe.variant is EpisodeVariant.POSITIVE)
    assert bundle.truth.recipe.distractor_link_count == 0
    example = AuditExample(
        bundle, 0, "matched", 0, 0, quartet_member_index=bundle.truth.key.coordinate.member_index
    )
    injector = next(item for item in NAMED_LEAK_INJECTORS if item.control_id == "PC_COUNT_BY_LABEL")

    transformed = _rewrite_positive_control(injector, example, 0)

    original_facts = _fact_events(bundle)
    transformed_facts = _fact_events(transformed.bundle)
    assert transformed.bundle.public.init == bundle.public.init
    assert transformed.bundle.public.events[-1].timestamp == bundle.public.events[-1].timestamp
    assert transformed.bundle.public.events[-1].payload == bundle.public.events[-1].payload
    assert isinstance(transformed.bundle.public.events[-1].payload, ActivationPayload)
    assert transformed_facts[: len(original_facts)] == original_facts
    padding = transformed_facts[len(original_facts) :]
    assert len(transformed_facts) == 48
    assert all(isinstance(event.payload, LinkFact) for event in padding)
    assert all(
        event.payload.source_node not in bundle.truth.relevant_node_path
        and event.payload.target_node not in bundle.truth.relevant_node_path
        for event in padding
        if isinstance(event.payload, LinkFact)
    )
    assert all(
        left.timestamp < right.timestamp
        for left, right in zip(
            (*original_facts[-1:], *padding),
            (*padding, transformed.bundle.public.events[-1]),
            strict=True,
        )
    )


def test_hazard_layout_adds_an_unreachable_sentinel_for_zero_distractors() -> None:
    """The sentinel must never borrow a relevant path LINK."""
    from silent_cascade.env.invariants import validate_episode_invariants
    from silent_cascade.env.leakage import (
        NAMED_LEAK_INJECTORS,
        AuditExample,
        _fact_events,
        _rewrite_positive_control,
    )
    from silent_cascade.schemas import LinkFact, SafeFact

    config = _config()
    bundles = generate_matched_cohort(
        config,
        CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 0, 0, 2),
        91,
    ).episodes
    from silent_cascade.env.episode import EpisodeVariant

    bundle = next(item for item in bundles if item.truth.recipe.variant is EpisodeVariant.POSITIVE)
    assert bundle.truth.recipe.distractor_link_count == 0
    example = AuditExample(
        bundle, 0, "matched", 0, 0, quartet_member_index=bundle.truth.key.coordinate.member_index
    )
    injector = next(
        item for item in NAMED_LEAK_INJECTORS if item.control_id == "PC_HAZARD_LAYOUT_BY_CLASS"
    )

    transformed = _rewrite_positive_control(injector, example, 0, injected_hazard_target=3)

    original_links = {
        event.payload for event in _fact_events(bundle) if isinstance(event.payload, LinkFact)
    }
    first_four = _fact_events(transformed.bundle)[:4]
    sentinels = [
        event.payload
        for event in first_four
        if isinstance(event.payload, LinkFact) and event.payload not in original_links
    ]
    assert isinstance(first_four[3].payload, SafeFact)
    assert len(sentinels) == 1
    sentinel = sentinels[0]
    assert sentinel.source_node not in bundle.truth.relevant_node_path
    assert sentinel.target_node not in bundle.truth.relevant_node_path
    report = validate_episode_invariants(transformed.bundle, config, strict=False)
    assert report.check_ids == ("recipe_distractor_count",)


def test_count_control_minimum_duration_keeps_the_original_time_boundary() -> None:
    """Short windows use increasing padding times and one precise admitted consequence."""
    from silent_cascade.config import resolve_config
    from silent_cascade.env.episode import EpisodeVariant
    from silent_cascade.env.generator import IndependentEpisodeRequest, generate_stress_episode
    from silent_cascade.env.invariants import validate_episode_invariants
    from silent_cascade.env.leakage import (
        NAMED_LEAK_INJECTORS,
        AuditExample,
        _fact_events,
        _rewrite_positive_control,
    )

    config = resolve_config(
        Phase1Config,
        [
            Path("configs/base.yaml"),
            Path("configs/data/primary.yaml"),
            Path("configs/data/stress.yaml"),
        ],
    ).config
    bundle = generate_stress_episode(
        config,
        IndependentEpisodeRequest(
            split_namespace=SplitNamespace.DEBUG,
            suite=SuiteName.MINIMUM_DURATION_STRESS,
            root_seed=20260831,
            episode_index=41,
            requested_path_length=3,
            variant=EpisodeVariant.POSITIVE,
            allocation_quartet_index=10,
            quartet_member_index=1,
        ),
        91,
    )
    injector = next(item for item in NAMED_LEAK_INJECTORS if item.control_id == "PC_COUNT_BY_LABEL")

    transformed = _rewrite_positive_control(
        injector,
        AuditExample(
            bundle,
            0,
            "independent",
            10,
            41,
            quartet_member_index=bundle.truth.key.coordinate.quartet_member_index,
        ),
        0,
    )

    assert transformed.bundle.truth.key.coordinate.quartet_member_index == 1
    assert transformed.bundle.public.init == bundle.public.init
    assert transformed.bundle.public.events[-1].timestamp == bundle.public.events[-1].timestamp
    assert all(
        left.timestamp < right.timestamp
        for left, right in zip(
            _fact_events(transformed.bundle),
            transformed.bundle.public.events[1:],
            strict=True,
        )
    )
    report = validate_episode_invariants(transformed.bundle, config, strict=False)
    assert report.check_ids == ("observation_gap",)


def test_count_control_short_observation_window_has_one_precise_admission() -> None:
    """Sub-minimum inserted gaps are explicit; existing timestamps never move."""
    from silent_cascade.env.episode import EpisodeVariant
    from silent_cascade.env.invariants import validate_episode_invariants
    from silent_cascade.env.leakage import (
        NAMED_LEAK_INJECTORS,
        AuditExample,
        _fact_events,
        _rewrite_positive_control,
    )

    config = _config()
    bundles = generate_matched_cohort(
        config,
        CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 83, 0, 2),
        91,
    ).episodes
    bundle = next(item for item in bundles if item.truth.recipe.variant is EpisodeVariant.POSITIVE)
    original_facts = _fact_events(bundle)
    assert bundle.public.events[-1].timestamp - original_facts[-1].timestamp < 0.11
    injector = next(item for item in NAMED_LEAK_INJECTORS if item.control_id == "PC_COUNT_BY_LABEL")

    transformed = _rewrite_positive_control(
        injector,
        AuditExample(
            bundle,
            0,
            "matched",
            0,
            0,
            quartet_member_index=bundle.truth.key.coordinate.member_index,
        ),
        0,
    )

    assert _fact_events(transformed.bundle)[: len(original_facts)] == original_facts
    report = validate_episode_invariants(transformed.bundle, config, strict=False)
    assert report.check_ids == ("observation_gap",)


def test_count_control_admits_exact_phase1_gate_identity_episode() -> None:
    """An already-48-fact positive must not invent a construction failure."""
    from silent_cascade.env.episode import EpisodeVariant, episode_sha256
    from silent_cascade.env.generator import (
        IndependentEpisodeRequest,
        generate_independent_episode,
    )
    from silent_cascade.env.invariants import validate_episode_invariants
    from silent_cascade.env.leakage import (
        NAMED_LEAK_INJECTORS,
        AuditExample,
        ShortcutFeatureGroup,
        _expected_control_failure_id,
        _fact_events,
        _rewrite_positive_control,
        _StoredExample,
        extract_shortcut_features,
    )
    from silent_cascade.schemas import HazardFact, LinkFact

    config = resolve_config(
        Phase1Config,
        [
            Path("configs/base.yaml"),
            Path("configs/data/primary.yaml"),
            Path("configs/data/stress.yaml"),
        ],
    ).config
    request = IndependentEpisodeRequest(
        split_namespace=SplitNamespace.PHASE1_GATE,
        suite=SuiteName.DISTRACTOR_FLOOD,
        root_seed=2026083011,
        episode_index=341,
        requested_path_length=2,
        variant=EpisodeVariant.POSITIVE,
        allocation_quartet_index=19085,
        quartet_member_index=1,
    )
    bundle = generate_independent_episode(config, request, 2026083012)
    assert bundle.public.init.episode_public_id == "7f8d4768-be0b-4a05-8a84-794136396559"
    facts = _fact_events(bundle)
    assert len(facts) == 48
    assert bundle.truth.recipe.distractor_link_count == 43
    assert sum(isinstance(event.payload, LinkFact) for event in facts) - 2 == 43

    original = AuditExample(
        bundle,
        76341,
        "independent",
        19085,
        341,
        quartet_member_index=bundle.truth.key.coordinate.quartet_member_index,
    )
    injector = next(item for item in NAMED_LEAK_INJECTORS if item.control_id == "PC_COUNT_BY_LABEL")
    transformed = _rewrite_positive_control(injector, original, 6533)
    assert transformed == original

    features = extract_shortcut_features(transformed, 8_000)
    assert features.vectors[ShortcutFeatureGroup.COUNTS][0] == 48 / 64
    row = _StoredExample(
        public_id=bundle.public.init.episode_public_id,
        digest=episode_sha256(bundle),
        group_id=features.audit_group_id,
        suite=SuiteName.DISTRACTOR_FLOOD,
        path_length=2,
        variant=EpisodeVariant.POSITIVE,
        hazard_class=bundle.truth.relevant_hazard_type,
        block=19085,
        position=341,
        quartet_member_index=1,
        public_hazard_classes=tuple(
            sorted(
                event.payload.hazard_type
                for event in facts
                if isinstance(event.payload, HazardFact)
            )
        ),
    )
    report = validate_episode_invariants(transformed.bundle, config, strict=False)
    assert report.valid
    actual_failure = (
        None
        if report.valid
        else (report.check_ids[0] if len(report.check_ids) == 1 else "multiple")
    )
    assert actual_failure is None
    assert _expected_control_failure_id(injector, row, original, transformed, config) is None


def test_activation_id_control_identifies_exact_phase1_gate_provenance_failure() -> None:
    """The first frozen activation-ID row must retain its authorized failure identity."""
    from silent_cascade.env.episode import EpisodeVariant, episode_sha256
    from silent_cascade.env.generator import (
        IndependentEpisodeRequest,
        generate_independent_episode,
    )
    from silent_cascade.env.invariants import validate_episode_invariants
    from silent_cascade.env.leakage import (
        NAMED_LEAK_INJECTORS,
        AuditExample,
        _expected_control_artifact,
        _expected_control_failure_id,
        _fact_events,
        _rewrite_positive_control,
        _StoredExample,
        extract_shortcut_features,
    )
    from silent_cascade.schemas import HazardFact

    config = resolve_config(
        Phase1Config,
        [
            Path("configs/base.yaml"),
            Path("configs/data/primary.yaml"),
            Path("configs/data/stress.yaml"),
        ],
    ).config
    request = IndependentEpisodeRequest(
        split_namespace=SplitNamespace.PHASE1_GATE,
        suite=SuiteName.IID_PRIMARY,
        root_seed=2026083011,
        episode_index=32,
        requested_path_length=2,
        variant=EpisodeVariant.POSITIVE,
        allocation_quartet_index=8,
        quartet_member_index=0,
    )
    bundle = generate_independent_episode(config, request, 2026083012)
    assert bundle.public.init.episode_public_id == "366e995a-3211-403b-9da6-b4a376c4829b"

    original = AuditExample(
        bundle,
        32,
        "independent",
        8,
        32,
        quartet_member_index=bundle.truth.key.coordinate.quartet_member_index,
    )
    injector = next(
        item for item in NAMED_LEAK_INJECTORS if item.control_id == "PC_ACTIVATION_ID_BY_LABEL"
    )
    transformed = _rewrite_positive_control(injector, original, 0)
    expected_public, expected_rank = _expected_control_artifact(
        injector,
        original,
        corpus_position=0,
        encoded_manifest_rank=0,
        injected_hazard_target=None,
    )
    assert transformed.bundle.public == expected_public
    assert transformed.manifest_rank == expected_rank
    assert transformed.bundle.truth.key.coordinate.quartet_member_index == 0

    features = extract_shortcut_features(original, 8_000)
    facts = _fact_events(bundle)
    row = _StoredExample(
        public_id=bundle.public.init.episode_public_id,
        digest=episode_sha256(bundle),
        group_id=features.audit_group_id,
        suite=SuiteName.IID_PRIMARY,
        path_length=2,
        variant=EpisodeVariant.POSITIVE,
        hazard_class=bundle.truth.relevant_hazard_type,
        block=8,
        position=32,
        quartet_member_index=0,
        public_hazard_classes=tuple(
            sorted(
                event.payload.hazard_type
                for event in facts
                if isinstance(event.payload, HazardFact)
            )
        ),
    )
    expected_failure = _expected_control_failure_id(injector, row, original, transformed, config)
    report = validate_episode_invariants(transformed.bundle, config, strict=False)
    actual_failure = (
        None
        if report.valid
        else (report.check_ids[0] if len(report.check_ids) == 1 else "multiple")
    )
    assert expected_failure == "node_permutation_provenance"
    assert actual_failure == "node_permutation_provenance"


@pytest.mark.parametrize(
    "attack",
    ("public_id_suffix", "terminal_alternate_code", "hazard_alternate_fill"),
)
def test_exact_control_preflight_rejects_round_six_reviewer_attacks(attack: str) -> None:
    """Alternate but still decodable interventions are not the declared controls."""
    from silent_cascade.env.episode import EpisodeVariant
    from silent_cascade.env.leakage import (
        NAMED_LEAK_INJECTORS,
        AuditExample,
        _fact_events,
        _require_exact_control_public_mutation,
        _rewrite_positive_control,
    )

    config = _config()
    bundles = generate_matched_cohort(
        config,
        CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 41, 17, 3),
        91,
    ).episodes
    encoded_rank = 100
    hazard_target = None
    corpus_position = 7
    if attack == "public_id_suffix":
        injector = next(
            item for item in NAMED_LEAK_INJECTORS if item.control_id == "PC_PUBLIC_ID_BY_LABEL"
        )
        bundle = next(
            item for item in bundles if item.truth.recipe.variant is EpisodeVariant.SAFE_NEGATIVE
        )
        original = AuditExample(
            bundle,
            0,
            "matched",
            17,
            0,
            quartet_member_index=bundle.truth.key.coordinate.member_index,
        )
        transformed = _rewrite_positive_control(injector, original, corpus_position)
        corrupted = replace(
            transformed,
            bundle=replace(
                transformed.bundle,
                public=replace(
                    transformed.bundle.public,
                    init=replace(
                        transformed.bundle.public.init,
                        episode_public_id="aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa",
                    ),
                ),
            ),
        )
    elif attack == "terminal_alternate_code":
        injector = next(
            item
            for item in NAMED_LEAK_INJECTORS
            if item.control_id == "PC_TERMINAL_ORDER_BY_VARIANT"
        )
        bundle = next(
            item for item in bundles if item.truth.recipe.variant is EpisodeVariant.SAFE_NEGATIVE
        )
        original = AuditExample(
            bundle,
            0,
            "matched",
            17,
            0,
            quartet_member_index=bundle.truth.key.coordinate.member_index,
        )
        transformed = _rewrite_positive_control(injector, original, corpus_position)
        events = list(transformed.bundle.public.events)
        events[1], events[2] = (
            replace(events[1], payload=events[2].payload),
            replace(events[2], payload=events[1].payload),
        )
        corrupted = replace(
            transformed,
            bundle=replace(
                transformed.bundle,
                public=replace(transformed.bundle.public, events=tuple(events)),
            ),
        )
    else:
        injector = next(
            item for item in NAMED_LEAK_INJECTORS if item.control_id == "PC_HAZARD_LAYOUT_BY_CLASS"
        )
        zero_distractor = generate_matched_cohort(
            config,
            CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 0, 0, 2),
            91,
        ).episodes
        bundle = next(
            item for item in zero_distractor if item.truth.recipe.variant is EpisodeVariant.POSITIVE
        )
        original = AuditExample(
            bundle,
            0,
            "matched",
            0,
            0,
            quartet_member_index=bundle.truth.key.coordinate.member_index,
        )
        hazard_target = 3
        transformed = _rewrite_positive_control(
            injector,
            original,
            corpus_position,
            injected_hazard_target=hazard_target,
        )
        assert [type(event.payload).__name__ for event in _fact_events(transformed.bundle)[:4]] == [
            "HazardFact",
            "HazardFact",
            "LinkFact",
            "SafeFact",
        ]
        events = list(transformed.bundle.public.events)
        events[1], events[2] = (
            replace(events[1], payload=events[2].payload),
            replace(events[2], payload=events[1].payload),
        )
        corrupted = replace(
            transformed,
            bundle=replace(
                transformed.bundle,
                public=replace(transformed.bundle.public, events=tuple(events)),
            ),
        )

    _require_exact_control_public_mutation(
        injector,
        original,
        transformed,
        corpus_position=corpus_position,
        encoded_manifest_rank=encoded_rank,
        injected_hazard_target=hazard_target,
    )
    with pytest.raises(ValueError, match="not exactly declared"):
        _require_exact_control_public_mutation(
            injector,
            original,
            corrupted,
            corpus_position=corpus_position,
            encoded_manifest_rank=encoded_rank,
            injected_hazard_target=hazard_target,
        )


@pytest.mark.parametrize(
    ("case", "control_id", "variant_name", "hazard_target"),
    (
        ("count_add_padding", "PC_COUNT_BY_LABEL", "positive", None),
        ("count_add_retained", "PC_COUNT_BY_LABEL", "positive", None),
        ("count_remove_identity", "PC_COUNT_BY_LABEL", "flood_positive", None),
        ("count_remove_retained", "PC_COUNT_BY_LABEL", "flood_positive", None),
        ("gap_positive_target", "PC_ACTIVATION_GAP_BY_LABEL", "positive", None),
        ("gap_negative_target", "PC_ACTIVATION_GAP_BY_LABEL", "safe_negative", None),
        ("gap_preserved_fact", "PC_ACTIVATION_GAP_BY_LABEL", "positive", None),
        ("terminal_positive_alternate", "PC_TERMINAL_ORDER_BY_VARIANT", "positive", None),
        ("terminal_safe_alternate", "PC_TERMINAL_ORDER_BY_VARIANT", "safe_negative", None),
        (
            "terminal_disconnected_alternate",
            "PC_TERMINAL_ORDER_BY_VARIANT",
            "disconnected_negative",
            None,
        ),
        ("terminal_positive_identity", "PC_TERMINAL_ORDER_BY_VARIANT", "positive", None),
        ("terminal_safe_identity", "PC_TERMINAL_ORDER_BY_VARIANT", "safe_negative", None),
        (
            "terminal_disconnected_identity",
            "PC_TERMINAL_ORDER_BY_VARIANT",
            "disconnected_negative",
            None,
        ),
        ("terminal_suffix", "PC_TERMINAL_ORDER_BY_VARIANT", "positive", None),
        ("activation_positive_target", "PC_ACTIVATION_ID_BY_LABEL", "positive", None),
        ("activation_negative_target", "PC_ACTIVATION_ID_BY_LABEL", "safe_negative", None),
        ("activation_partial_bijection", "PC_ACTIVATION_ID_BY_LABEL", "positive", None),
        ("record_positive_band", "PC_RECORD_ID_BY_VARIANT", "positive", None),
        ("record_safe_band", "PC_RECORD_ID_BY_VARIANT", "safe_negative", None),
        (
            "record_disconnected_band",
            "PC_RECORD_ID_BY_VARIANT",
            "disconnected_negative",
            None,
        ),
        ("record_local_rank", "PC_RECORD_ID_BY_VARIANT", "safe_negative", None),
        ("record_preserved_time", "PC_RECORD_ID_BY_VARIANT", "positive", None),
        ("public_positive_byte", "PC_PUBLIC_ID_BY_LABEL", "positive", None),
        ("public_negative_byte", "PC_PUBLIC_ID_BY_LABEL", "safe_negative", None),
        ("public_suffix", "PC_PUBLIC_ID_BY_LABEL", "safe_negative", None),
        ("public_collision", "PC_PUBLIC_ID_BY_LABEL", "safe_negative", None),
        ("public_preserved_event", "PC_PUBLIC_ID_BY_LABEL", "positive", None),
        ("delay_positive_first", "PC_DELAY_BY_LABEL", "positive", None),
        ("delay_positive_second", "PC_DELAY_BY_LABEL", "positive", None),
        ("delay_negative_first", "PC_DELAY_BY_LABEL", "safe_negative", None),
        ("delay_negative_second", "PC_DELAY_BY_LABEL", "safe_negative", None),
        ("delay_preserved_safe", "PC_DELAY_BY_LABEL", "positive", None),
        ("hazard_hlh_0", "PC_HAZARD_LAYOUT_BY_CLASS", "zero_positive", 0),
        ("hazard_hlh_1", "PC_HAZARD_LAYOUT_BY_CLASS", "zero_positive", 1),
        ("hazard_hlh_2", "PC_HAZARD_LAYOUT_BY_CLASS", "zero_positive", 2),
        ("hazard_hlh_3", "PC_HAZARD_LAYOUT_BY_CLASS", "zero_positive", 3),
        ("hazard_lhh_0", "PC_HAZARD_LAYOUT_BY_CLASS", "zero_positive", 0),
        ("hazard_lhh_1", "PC_HAZARD_LAYOUT_BY_CLASS", "zero_positive", 1),
        ("hazard_lhh_2", "PC_HAZARD_LAYOUT_BY_CLASS", "zero_positive", 2),
        ("hazard_lhh_3", "PC_HAZARD_LAYOUT_BY_CLASS", "zero_positive", 3),
        ("hazard_relevant_sentinel", "PC_HAZARD_LAYOUT_BY_CLASS", "zero_positive", 3),
        ("hazard_sentinel_identity", "PC_HAZARD_LAYOUT_BY_CLASS", "zero_positive", 3),
        ("hazard_timestamp", "PC_HAZARD_LAYOUT_BY_CLASS", "zero_positive", 3),
        ("hazard_suffix", "PC_HAZARD_LAYOUT_BY_CLASS", "zero_positive", 3),
        ("hazard_class", "PC_HAZARD_LAYOUT_BY_CLASS", "zero_positive", 3),
        ("hazard_activation_parent", "PC_HAZARD_LAYOUT_BY_CLASS", "zero_positive", 3),
        ("manifest_rank", "PC_MANIFEST_ORDER_BY_VARIANT", "positive", None),
        ("manifest_public_order", "PC_MANIFEST_ORDER_BY_VARIANT", "positive", None),
    ),
)
def test_control_value_mutation_table_refuses_before_fit(
    case: str,
    control_id: str,
    variant_name: str,
    hazard_target: int | None,
) -> None:
    """Every declared target and preserved public family is authenticated by value."""
    import uuid

    from silent_cascade.env.episode import EpisodeVariant
    from silent_cascade.env.generator import IndependentEpisodeRequest, generate_independent_episode
    from silent_cascade.env.leakage import (
        NAMED_LEAK_INJECTORS,
        AuditExample,
        _fact_events,
        _require_exact_control_public_mutation,
        _rewrite_positive_control,
    )
    from silent_cascade.env.timing import action_window
    from silent_cascade.schemas import ActivationPayload, HazardFact, LinkFact

    config = _config()
    bundles = generate_matched_cohort(
        config,
        CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 41, 17, 3),
        91,
    ).episodes
    variant_by_name = {
        "positive": EpisodeVariant.POSITIVE,
        "safe_negative": EpisodeVariant.SAFE_NEGATIVE,
        "disconnected_negative": EpisodeVariant.DISCONNECTED_NEGATIVE,
    }
    if variant_name == "flood_positive":
        bundle = generate_independent_episode(
            config,
            IndependentEpisodeRequest(
                SplitNamespace.DEBUG,
                SuiteName.DISTRACTOR_FLOOD,
                41,
                1,
                4,
                EpisodeVariant.POSITIVE,
                0,
                1,
            ),
            91,
        )
        assert len(_fact_events(bundle)) == 55
        original = AuditExample(
            bundle,
            0,
            "independent",
            0,
            1,
            quartet_member_index=bundle.truth.key.coordinate.quartet_member_index,
        )
    elif variant_name == "zero_positive":
        zero_distractor = generate_matched_cohort(
            config,
            CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 0, 0, 2),
            91,
        ).episodes
        bundle = next(
            item for item in zero_distractor if item.truth.recipe.variant is EpisodeVariant.POSITIVE
        )
        assert bundle.truth.recipe.distractor_link_count == 0
        original = AuditExample(
            bundle,
            0,
            "matched",
            0,
            0,
            quartet_member_index=bundle.truth.key.coordinate.member_index,
        )
    else:
        variant = variant_by_name[variant_name]
        bundle = next(item for item in bundles if item.truth.recipe.variant is variant)
        original = AuditExample(
            bundle,
            0,
            "matched",
            17,
            0,
            quartet_member_index=bundle.truth.key.coordinate.member_index,
        )
    injector = next(item for item in NAMED_LEAK_INJECTORS if item.control_id == control_id)
    corpus_position = 7
    encoded_rank = 100
    transformed = _rewrite_positive_control(
        injector,
        original,
        corpus_position,
        encoded_manifest_rank=encoded_rank,
        injected_hazard_target=hazard_target,
    )
    _require_exact_control_public_mutation(
        injector,
        original,
        transformed,
        corpus_position=corpus_position,
        encoded_manifest_rank=encoded_rank,
        injected_hazard_target=hazard_target,
    )

    def with_events(events: list[object], *, truth: object | None = None) -> object:
        return replace(
            transformed,
            bundle=replace(
                transformed.bundle,
                public=replace(transformed.bundle.public, events=tuple(events)),
                truth=transformed.bundle.truth if truth is None else truth,
            ),
        )

    def truth_for_payload_permutation(before: list[object], after: list[object]) -> object:
        """Rebind private record references so the corrupt public bundle stays valid."""
        destinations: dict[int, int] = {}
        unmatched = list(after[:-1])
        for source in before[:-1]:
            destination = next(event for event in unmatched if event.payload == source.payload)
            unmatched.remove(destination)
            destinations[source.event_id] = destination.event_id
        truth = transformed.bundle.truth
        return replace(
            truth,
            relevant_record_ids=tuple(
                destinations[event_id] for event_id in truth.relevant_record_ids
            ),
            terminal_record_id=(
                None if truth.terminal_record_id is None else destinations[truth.terminal_record_id]
            ),
        )

    def truth_with_timing(*, activation_time: float, delay: float) -> object:
        truth = transformed.bundle.truth
        window = (
            action_window(activation_time, delay, truth.recipe.oracle_timing)
            if truth.recipe.variant is EpisodeVariant.POSITIVE
            else None
        )
        return replace(
            truth,
            private_terminal=replace(truth.private_terminal, timestamp=activation_time + delay),
            activation_time=activation_time,
            episode_delay=delay,
            action_window_start=None if window is None else window.start,
            action_window_end=None if window is None else window.end,
            action_target=None if window is None else window.target,
        )

    events = list(transformed.bundle.public.events)
    if case == "count_add_padding":
        padding = events[-2]
        assert isinstance(padding.payload, LinkFact)
        events[-2] = replace(padding, payload=replace(padding.payload, confidence=0.5))
        corrupted = with_events(events)
    elif case in {"count_add_retained", "count_remove_retained"}:
        events[0] = replace(events[0], timestamp=events[0].timestamp + 1.0e-6)
        corrupted = with_events(events)
    elif case == "count_remove_identity":
        events[0], events[1] = (
            replace(events[0], payload=events[1].payload),
            replace(events[1], payload=events[0].payload),
        )
        corrupted = with_events(events)
    elif case in {"gap_positive_target", "gap_negative_target"}:
        activation_time = events[-1].timestamp + 0.5
        events[-1] = replace(events[-1], timestamp=activation_time)
        corrupted = with_events(
            events,
            truth=truth_with_timing(
                activation_time=activation_time,
                delay=transformed.bundle.truth.episode_delay,
            ),
        )
    elif case == "gap_preserved_fact":
        events[0] = replace(events[0], timestamp=events[0].timestamp + 1.0e-6)
        corrupted = with_events(events)
    elif case.startswith("terminal_") and case.endswith("_alternate"):
        before = list(events)
        safe_slot = next(
            index for index in range(3) if type(events[index].payload).__name__ == "SafeFact"
        )
        hazard_slot = next(index for index in range(3) if index != safe_slot)
        events[safe_slot], events[hazard_slot] = (
            replace(events[safe_slot], payload=events[hazard_slot].payload),
            replace(events[hazard_slot], payload=events[safe_slot].payload),
        )
        corrupted = with_events(events, truth=truth_for_payload_permutation(before, events))
    elif case.startswith("terminal_") and case.endswith("_identity"):
        corrupted = replace(transformed, bundle=original.bundle)
    elif case == "terminal_suffix":
        events[3], events[4] = (
            replace(events[3], payload=events[4].payload),
            replace(events[4], payload=events[3].payload),
        )
        corrupted = with_events(events)
    elif case in {"activation_positive_target", "activation_negative_target"}:
        activation = events[-1]
        assert isinstance(activation.payload, ActivationPayload)
        wrong = 1 if case == "activation_positive_target" else 62
        events[-1] = replace(activation, payload=ActivationPayload(wrong))
        corrupted = with_events(events)
    elif case == "activation_partial_bijection":
        original_events = original.bundle.public.events
        changed_index = next(
            index
            for index, (before, after) in enumerate(
                zip(original_events[:-1], events[:-1], strict=True)
            )
            if before.payload != after.payload
        )
        events[changed_index] = replace(
            events[changed_index], payload=original_events[changed_index].payload
        )
        corrupted = with_events(events)
    elif case.startswith("record_") and case.endswith("_band"):
        mutable_index = next(
            index
            for index, event in enumerate(events[:-1])
            if event.event_id not in transformed.bundle.truth.relevant_record_ids
        )
        events[mutable_index] = replace(
            events[mutable_index], event_id=events[mutable_index].event_id + 64
        )
        corrupted = with_events(events)
    elif case == "record_local_rank":
        events[0], events[1] = (
            replace(events[0], event_id=events[1].event_id),
            replace(events[1], event_id=events[0].event_id),
        )
        corrupted = with_events(events)
    elif case == "record_preserved_time":
        events[0] = replace(events[0], timestamp=events[0].timestamp + 1.0e-6)
        corrupted = with_events(events)
    elif case.startswith("public_") and case != "public_preserved_event":
        raw = bytearray(uuid.UUID(transformed.bundle.public.init.episode_public_id).bytes)
        if case == "public_positive_byte":
            raw[0] = 1
        elif case == "public_negative_byte":
            raw[0] = 254
        elif case == "public_suffix":
            raw[-1] ^= 1
        else:
            collision = _rewrite_positive_control(
                injector, original, corpus_position + 1
            ).bundle.public.init.episode_public_id
            raw = bytearray(uuid.UUID(collision).bytes)
        corrupted = replace(
            transformed,
            bundle=replace(
                transformed.bundle,
                public=replace(
                    transformed.bundle.public,
                    init=replace(
                        transformed.bundle.public.init,
                        episode_public_id=str(uuid.UUID(bytes=bytes(raw))),
                    ),
                ),
            ),
        )
    elif case == "public_preserved_event":
        events[0] = replace(events[0], timestamp=events[0].timestamp + 1.0e-6)
        corrupted = with_events(events)
    elif case.startswith("delay_") and case != "delay_preserved_safe":
        hazard_indices = [
            index for index, event in enumerate(events) if isinstance(event.payload, HazardFact)
        ]
        selected = hazard_indices[0 if case.endswith("first") else 1]
        payload = events[selected].payload
        assert isinstance(payload, HazardFact)
        events[selected] = replace(
            events[selected], payload=replace(payload, delay=payload.delay + 1.0)
        )
        if events[selected].event_id == transformed.bundle.truth.terminal_record_id:
            corrupted = with_events(
                events,
                truth=truth_with_timing(
                    activation_time=transformed.bundle.truth.activation_time,
                    delay=payload.delay + 1.0,
                ),
            )
        else:
            corrupted = with_events(events)
    elif case == "delay_preserved_safe":
        safe_index = next(
            index
            for index, event in enumerate(events)
            if type(event.payload).__name__ == "SafeFact"
        )
        events[safe_index] = replace(
            events[safe_index],
            payload=replace(events[safe_index].payload, confidence=0.5),
        )
        corrupted = with_events(events)
    elif case.startswith("hazard_hlh_") or case.startswith("hazard_lhh_"):
        assert hazard_target is not None
        before = list(events)
        other = [index for index in range(4) if index != hazard_target]
        payloads = [events[index].payload for index in other]
        if case.startswith("hazard_hlh_"):
            payloads[1], payloads[2] = payloads[2], payloads[1]
        else:
            payloads = [payloads[2], payloads[0], payloads[1]]
        for index, payload in zip(other, payloads, strict=True):
            events[index] = replace(events[index], payload=payload)
        corrupted = with_events(events, truth=truth_for_payload_permutation(before, events))
    elif case in {"hazard_relevant_sentinel", "hazard_sentinel_identity"}:
        assert hazard_target is not None
        sentinel_index = next(
            index
            for index in range(4)
            if index != hazard_target and isinstance(events[index].payload, LinkFact)
        )
        if case == "hazard_relevant_sentinel":
            relevant_id = original.bundle.truth.relevant_record_ids[0]
            replacement = next(
                event.payload
                for event in _fact_events(original.bundle)
                if event.event_id == relevant_id
            )
        else:
            sentinel = events[sentinel_index].payload
            assert isinstance(sentinel, LinkFact)
            replacement = replace(sentinel, confidence=0.5)
        events[sentinel_index] = replace(events[sentinel_index], payload=replacement)
        corrupted = with_events(events)
    elif case == "hazard_timestamp":
        events[0] = replace(events[0], timestamp=events[0].timestamp + 1.0e-6)
        corrupted = with_events(events)
    elif case == "hazard_suffix":
        events[4] = replace(events[4], payload=replace(events[4].payload, confidence=0.5))
        corrupted = with_events(events)
    elif case == "hazard_class":
        hazard_index = next(
            index
            for index in range(4)
            if isinstance(events[index].payload, HazardFact)
            and events[index].event_id != transformed.bundle.truth.terminal_record_id
        )
        payload = events[hazard_index].payload
        assert isinstance(payload, HazardFact)
        events[hazard_index] = replace(
            events[hazard_index],
            payload=replace(payload, hazard_type=(payload.hazard_type + 2) % 4),
        )
        corrupted = with_events(events)
    elif case == "hazard_activation_parent":
        activation = events[-1]
        assert isinstance(activation.payload, ActivationPayload)
        events[-1] = replace(
            activation,
            payload=ActivationPayload((activation.payload.start_node + 1) % 64),
        )
        truth = transformed.bundle.truth
        corrupted = with_events(
            events,
            truth=replace(
                truth,
                relevant_node_path=(
                    events[-1].payload.start_node,
                    *truth.relevant_node_path[1:],
                ),
            ),
        )
    elif case == "manifest_rank":
        corrupted = replace(transformed, manifest_rank=encoded_rank + 1)
    else:
        before = list(events)
        events[0], events[1] = (
            replace(events[0], payload=events[1].payload),
            replace(events[1], payload=events[0].payload),
        )
        corrupted = with_events(events, truth=truth_for_payload_permutation(before, events))

    fit_calls = 0

    def fit_spy() -> None:
        nonlocal fit_calls
        fit_calls += 1

    def preflight_then_fit() -> None:
        _require_exact_control_public_mutation(
            injector,
            original,
            corrupted,
            corpus_position=corpus_position,
            encoded_manifest_rank=encoded_rank,
            injected_hazard_target=hazard_target,
        )
        fit_spy()

    with pytest.raises(ValueError, match="not exactly declared"):
        preflight_then_fit()
    assert fit_calls == 0


@pytest.mark.parametrize("control_index", range(9))
def test_every_control_preflight_rejects_an_undeclared_public_field(
    control_index: int,
) -> None:
    """Every control has an exact mutation-field contract before feature extraction."""
    from silent_cascade.env.episode import EpisodeVariant
    from silent_cascade.env.leakage import (
        NAMED_LEAK_INJECTORS,
        AuditExample,
        _require_exact_control_public_mutation,
        _rewrite_positive_control,
    )

    bundles = generate_matched_cohort(
        _config(),
        CohortRequest(SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 41, 17, 3),
        91,
    ).episodes
    injector = NAMED_LEAK_INJECTORS[control_index]
    bundle = (
        next(item for item in bundles if item.truth.recipe.variant is EpisodeVariant.POSITIVE)
        if injector.control_id == "PC_HAZARD_LAYOUT_BY_CLASS"
        else bundles[0]
    )
    original = AuditExample(
        bundle, 0, "matched", 17, 0, quartet_member_index=bundle.truth.key.coordinate.member_index
    )
    transformed = _rewrite_positive_control(
        injector,
        original,
        0,
        encoded_manifest_rank=100,
        injected_hazard_target=2,
    )
    _require_exact_control_public_mutation(
        injector,
        original,
        transformed,
        corpus_position=0,
        encoded_manifest_rank=100,
        injected_hazard_target=2,
    )
    corrupted = replace(
        transformed,
        bundle=replace(
            transformed.bundle,
            public=replace(
                transformed.bundle.public,
                init=replace(transformed.bundle.public.init, memory_capacity=63),
            ),
        ),
    )

    with pytest.raises(ValueError, match="not exactly declared"):
        _require_exact_control_public_mutation(
            injector,
            original,
            corrupted,
            corpus_position=0,
            encoded_manifest_rank=100,
            injected_hazard_target=2,
        )


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


def test_every_standardization_mask_has_a_frozen_known_answer() -> None:
    """Every feature family freezes categorical and continuous vocabulary roles."""
    from silent_cascade.env.leakage import (
        ShortcutFeatureGroup,
        _continuous_columns,
        _feature_bounds,
    )

    expected = {
        ShortcutFeatureGroup.ID_POSITION: (
            17,
            1,
            "d3a064592dc6bda7ad009a53d77dc197fd15e0fd3b291f30eb8ab37862e2fc95",
        ),
        ShortcutFeatureGroup.ACTIVATION_NODE: (
            256,
            192,
            "f6e117451b82a006c21b97960fb6dd5203b5ebedc0700004771b1fa1f31edf6b",
        ),
        ShortcutFeatureGroup.FIRST_LAST_FACT: (
            278,
            2,
            "a44fb51bf98102fa816d746ca68f483713b672e0b6d5dbe1af7bc29c27d30419",
        ),
        ShortcutFeatureGroup.COUNTS: (
            4,
            4,
            "fde502858306c235a3121e42326b53228b7ef4690eeed92a2b2eafe73c03a3ef",
        ),
        ShortcutFeatureGroup.ORDER_RECORD_IDS: (
            768,
            256,
            "ba084cee7b744153155fb8ea34183bde8dbebd50db9196266bba2b752a0a6e67",
        ),
        ShortcutFeatureGroup.TIMES: (
            194,
            130,
            "4c8ce184e41c4f76706f279ae276deeb658897d1f3f58e4f885e9c548c078254",
        ),
        ShortcutFeatureGroup.TERMINAL_MULTISET: (
            10,
            4,
            "46aa68f4b1ee3cc5e0d0ff27299d91b99c9e6ac64b2a0588fcc9074a08c44a62",
        ),
        ShortcutFeatureGroup.LINK_TOPOLOGY: (
            33,
            14,
            "4469147f074ed578389d5ac0b1d19eb7c02440110d2edc25ddafe0ad998adc71",
        ),
        ShortcutFeatureGroup.COMBINED: (
            1560,
            603,
            "6ca18896cbfa1cd90a65b0851758f769b5fd340c94ddfdb956f5517f5cebbe51",
        ),
    }
    for group, (width, continuous_count, digest) in expected.items():
        start, stop = _feature_bounds(group)
        mask = _continuous_columns(group, stop - start)
        assert (len(mask), int(mask.sum())) == (width, continuous_count)
        assert hashlib.sha256(np.packbits(mask.astype(np.uint8)).tobytes()).hexdigest() == digest


def test_split_membership_has_an_exact_known_answer() -> None:
    """Seed framing, quartet grouping, rank order, and 80/20 selection are frozen."""
    from silent_cascade.env.episode import EpisodeVariant
    from silent_cascade.env.leakage import _split_memberships, _StoredExample

    variants = (
        EpisodeVariant.POSITIVE,
        EpisodeVariant.POSITIVE,
        EpisodeVariant.SAFE_NEGATIVE,
        EpisodeVariant.DISCONNECTED_NEGATIVE,
    )
    rows = tuple(
        _StoredExample(
            public_id=f"00000000-0000-4000-8000-{block * 4 + position:012d}",
            digest=f"{block * 4 + position + 1:064x}",
            group_id=f"matched:{block}",
            suite=SuiteName.IID_PRIMARY,
            path_length=3,
            variant=variant,
            hazard_class=0 if variant is EpisodeVariant.POSITIVE else None,
            block=block,
            position=position,
            quartet_member_index=position,
        )
        for block in range(5)
        for position, variant in enumerate(variants)
    )

    train, test, digest = _split_memberships(
        rows,
        2026083091,
        "b" * 64,
        "matched",
        strict_divisible=True,
    )

    assert train.tolist() == [
        0,
        1,
        2,
        3,
        8,
        9,
        10,
        11,
        4,
        5,
        6,
        7,
        16,
        17,
        18,
        19,
    ]
    assert test.tolist() == [12, 13, 14, 15]
    assert digest == "4ac3287de1886b3d10644f95363b955d6cf4219bd578ac701522cb32ef473d3e"


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
            quartet_member_index=index,
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
            quartet_member_index=index,
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
        quartet_member_index=0,
        requested_path_length=3,
        variant=EpisodeVariant.POSITIVE,
    )
    bundle = generate_independent_episode(config, request, 91)
    assert bundle.truth.key.coordinate.quartet_member_index == 0
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
        _validate_audit_coordinate(
            AuditExample(
                bundle,
                0,
                "independent",
                1,
                99,
                quartet_member_index=bundle.truth.key.coordinate.quartet_member_index,
            ),
            0,
            descriptor,
        )


@pytest.mark.parametrize("mutation", ("position", "variant", "stratum"))
def test_independent_quartet_aggregate_rejects_corruption(mutation: str) -> None:
    """Individually valid coordinates cannot forge a complete allocation quartet."""
    from silent_cascade.env.episode import EpisodeVariant
    from silent_cascade.env.leakage import _StoredExample, _validate_independent_quartets
    from silent_cascade.rng import AllocationLabelKey, allocate_independent_variants

    variants = allocate_independent_variants(
        AllocationLabelKey("ofd-v1", SplitNamespace.DEBUG, SuiteName.IID_PRIMARY, 41, 3, 7)
    )
    rows = [
        _StoredExample(
            public_id=f"00000000-0000-4000-8000-{index:012d}",
            digest=f"{index + 1:064x}",
            group_id="independent:7",
            suite=SuiteName.IID_PRIMARY,
            path_length=3,
            variant=variant,
            hazard_class=0 if variant is EpisodeVariant.POSITIVE else None,
            block=7,
            position=28 + index,
            quartet_member_index=index,
            public_hazard_classes=(0, 1),
        )
        for index, variant in enumerate(variants)
    ]
    _validate_independent_quartets(rows, _quartet_descriptor())
    if mutation == "position":
        rows[3] = replace(rows[3], position=30)
    elif mutation == "variant":
        rows[3] = replace(rows[3], variant=EpisodeVariant.POSITIVE)
    else:
        rows[3] = replace(rows[3], path_length=4)

    with pytest.raises(ValueError, match="independent allocation quartet"):
        _validate_independent_quartets(rows, _quartet_descriptor())


def test_all_frozen_gate_block_boundaries_are_valid_suite_scoped_quartets() -> None:
    """Every frozen block boundary, including suite resets, is authoritative."""
    from itertools import islice

    from silent_cascade.env.generator import (
        PHASE1_GATE_ALLOCATION,
        iter_phase1_gate_requests,
    )
    from silent_cascade.env.leakage import _StoredExample, _validate_independent_quartets

    requests = iter_phase1_gate_requests(41)
    observed = []
    for block in PHASE1_GATE_ALLOCATION.blocks:
        quartet = tuple(islice(requests, 4))
        rows = tuple(
            _StoredExample(
                public_id=f"00000000-0000-4000-8000-{request.episode_index:012d}",
                digest=f"{index + 1:064x}",
                group_id=f"independent:{request.allocation_quartet_index}",
                suite=request.suite,
                path_length=request.requested_path_length,
                variant=request.variant,
                hazard_class=0 if request.variant.value == "positive" else None,
                block=request.allocation_quartet_index,
                position=request.episode_index,
                quartet_member_index=request.quartet_member_index,
            )
            for index, request in enumerate(quartet)
        )
        _validate_independent_quartets(rows, _quartet_descriptor(SplitNamespace.PHASE1_GATE))
        observed.append(
            (
                quartet[0].suite,
                quartet[0].requested_path_length,
                quartet[0].episode_index,
                quartet[0].allocation_quartet_index,
            )
        )
        tuple(islice(requests, block.episode_count - 4))

    assert len(observed) == 16
    assert observed[3] == (SuiteName.OOD_DEPTH, 5, 0, 6_000)
    assert observed[7] == (SuiteName.OOD_SHORT_DELAY, 2, 0, 10_000)
    assert observed[10] == (SuiteName.OOD_LONG_DELAY, 2, 0, 16_000)
    assert observed[13] == (SuiteName.DISTRACTOR_FLOOD, 2, 0, 19_000)


def test_independently_authenticated_source_executes_all_fits_with_frozen_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent quartets run clean/shuffled fits under the exact same protocol."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    source = _authenticated_independent_test_source(config)
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
    fit_configs = []

    def exact_label_fit(
        _reader,
        _train,
        labels,
        test,
        _classes,
        _continuous,
        fit_config,
    ):
        fit_configs.append(fit_config)
        return labels[test].copy(), 1

    monkeypatch.setattr(leakage, "_fit_predict_batched", exact_label_fit)
    report = leakage.audit_leakage(
        source,
        audit_config,
        leakage.LeakageAuditProfileName.TEST,
        _provenance(_config_sha256(config), source),
        tmp_path,
    )

    assert report.generation_mode == "independent"
    assert report.episode_count == 240
    assert report.schema_version == "leakage-report-v2"
    assert report.namespace_evidence.generation_mode == "independent"
    assert report.namespace_evidence.public_id_seed == 91
    assert report.namespace_evidence.accepted_draw_count == 240
    assert report.namespace_evidence.seed_token_count == 1_680
    assert report.namespace_evidence.base_public_id_count == 240
    assert report.namespace_evidence.clock_public_id_count == 2
    assert len(fit_configs) == 54
    assert {id(item) for item in fit_configs} == {id(audit_config)}
    assert {item.l2_penalty for item in fit_configs} == {0.03}


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
        _provenance(_config_sha256(config), source),
        tmp_path,
    )

    assert report.episode_count == 240
    assert report.schema_version == "leakage-report-v2"
    assert report.namespace_evidence.generation_mode == "matched"
    assert report.namespace_evidence.public_id_seed == 91
    assert report.namespace_evidence.accepted_draw_count == 60
    assert report.namespace_evidence.seed_token_count == 1_200
    assert report.namespace_evidence.base_public_id_count == 240
    assert report.namespace_evidence.clock_public_id_count == 2
    assert len(report.probes) == 27
    assert tuple(item.check_id for item in report.counterfactual_checks) == tuple(
        CounterfactualCheckId
    )
    assert report.label_shuffled_control_passed
    assert not list(tmp_path.iterdir())


def test_audit_rejects_colliding_actual_construction_tokens_before_probe_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The leakage construction check must consume derived tokens rather than coordinates."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    source = _authenticated_test_source(config, groups_per_path=1)
    audit_config = config.data.leakage_audit.model_copy(
        update={
            "test": config.data.leakage_audit.test.model_copy(
                update={"episode_count": source.episode_count}
            )
        }
    )
    probe_calls = 0

    def never_probe(*_args: object, **_kwargs: object):
        nonlocal probe_calls
        probe_calls += 1
        return []

    monkeypatch.setattr(leakage, "_run_probes", never_probe)
    monkeypatch.setattr(
        leakage,
        "matched_seed_tokens",
        lambda _request, _attempt: ("0" * 64,) * 20,
    )

    with pytest.raises(ProvenanceError, match="construction token collision"):
        leakage.audit_leakage(
            source,
            audit_config,
            leakage.LeakageAuditProfileName.TEST,
            _provenance(_config_sha256(config), source),
            tmp_path,
        )
    assert probe_calls == 0


def test_construction_check_labels_derive_from_actual_namespace_collisions() -> None:
    """Coordinate uniqueness cannot make token or public-ID construction checks pass."""
    import silent_cascade.env.leakage as leakage

    evidence = build_construction_namespace_evidence(
        generation_mode="matched",
        public_id_seed=91,
        accepted_attempts=(0,),
        seed_tokens=tuple(f"{index:064x}" for index in range(20)),
        base_public_ids=tuple(str(uuid.UUID(int=index + 1)) for index in range(4)),
        clock_public_ids=(),
    )

    assert leakage._construction_checks_from_namespace(evidence)["seed_tokens"] is True
    token_collision = evidence.model_copy(update={"seed_token_collision_count": 1})
    public_id_collision = evidence.model_copy(update={"public_id_collision_count": 1})
    assert leakage._construction_checks_from_namespace(token_collision)["seed_tokens"] is False
    assert leakage._construction_checks_from_namespace(public_id_collision)["public_ids"] is False


def test_complete_report_hash_is_stable_in_a_fresh_process() -> None:
    """A fresh interpreter reproduces the exact canonical leakage report bytes."""
    import subprocess
    import sys

    script = """
import runpy
import tempfile
from pathlib import Path
from silent_cascade.env.leakage import LeakageAuditProfileName, audit_leakage
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
ns = runpy.run_path('tests/unit/test_leakage.py')
config = ns['_config']()
source = ns['_authenticated_test_source'](config, groups_per_path=20)
audit_config = config.data.leakage_audit.model_copy(update={
    'test': config.data.leakage_audit.test.model_copy(update={
        'episode_count': source.episode_count,
        'permutation_replicates': 1,
        'minimum_test_examples_per_class': 1,
    }),
})
with tempfile.TemporaryDirectory() as directory:
    report = audit_leakage(
        source,
        audit_config,
        LeakageAuditProfileName.TEST,
        ns['_provenance'](ns['_config_sha256'](config), source),
        Path(directory),
    )
print(sha256_bytes(canonical_json_bytes(report)))
"""
    completed = subprocess.run(
        (sys.executable, "-c", script),
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stderr == ""
    assert completed.stdout.strip() == (
        "3723b8c8baf310cc9134e2630b24cc3cc1a20fd50bd837f254fdb0f1c45a1e35"
    )


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
            _provenance(_config_sha256(config), source),
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
        public_id_seed = source.public_id_seed
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
            _provenance(_config_sha256(config), source),
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
        public_id_seed = source.public_id_seed
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
            _provenance(_config_sha256(config), source),
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
        schema_version="leakage-source-auth-v2",
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
        schema_version="leakage-source-auth-v2",
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
            _provenance(_config_sha256(config), source),
            tmp_path,
        )


def test_audit_rejects_fully_reminted_reordered_source_before_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stream cannot replace itself and recompute every source-owned witness."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    trusted = _authenticated_test_source(config, groups_per_path=20)
    reordered = tuple(
        replace(example, manifest_rank=rank)
        for rank, example in enumerate(reversed(trusted.examples))
    )
    forged_descriptor = trusted.descriptor.model_copy(
        update={
            "allocation_or_manifest_sha256": leakage._source_manifest_sha256(reordered),
        }
    )
    forged_clock_pairs = tuple(
        leakage.PairedClockAuditPair(
            reordered[index],
            leakage.scale_episode_time(
                reordered[index].bundle,
                suite,
                f"00000000-0000-4000-8000-{suffix:012d}",
            ),
        )
        for index, suite, suffix in (
            (0, SuiteName.CLOCK_SCALE_0_1X, 101),
            (80, SuiteName.CLOCK_SCALE_10X, 102),
        )
    )
    forged_authentication = trusted.authentication.model_copy(
        update={
            "descriptor_sha256": leakage.audit_source_descriptor_sha256(forged_descriptor),
            "source_manifest_sha256": leakage._source_manifest_sha256(reordered),
            "clock_pair_manifest_sha256": leakage._clock_pair_manifest_sha256(forged_clock_pairs),
        }
    )

    class RemintedSource:
        descriptor = forged_descriptor
        authentication = forged_authentication
        validation_config = config
        episode_count = len(reordered)
        iter_calls = 0

        def iter_examples(self):
            self.iter_calls += 1
            return iter(reordered)

        def iter_clock_pairs(self):
            self.iter_calls += 1
            return iter(forged_clock_pairs)

    source = RemintedSource()

    def prohibited_fit(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("probe fitting began before independent trust validation")

    monkeypatch.setattr(leakage, "_fit_predict_batched", prohibited_fit)
    audit_config = config.data.leakage_audit.model_copy(
        update={
            "test": config.data.leakage_audit.test.model_copy(
                update={"episode_count": len(reordered)}
            )
        }
    )

    with pytest.raises(ValueError, match="independent leakage trust anchor"):
        leakage.audit_leakage(
            source,
            audit_config,
            leakage.LeakageAuditProfileName.TEST,
            _provenance(_config_sha256(config), trusted),
            tmp_path,
        )
    assert source.iter_calls == 0


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
        AuditExample(
            bundle, rank, "matched", path_index, member_index, quartet_member_index=member_index
        )
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
        schema_version="leakage-source-auth-v2",
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
    assert {item.check_id: item.result_payload_sha256 for item in checks} == {
        CounterfactualCheckId.TERMINAL_DELAY_SWAP: (
            "02a14285d1611cc621b2e378db943863a69b4bb83e2b73f0c6aa5977353e52af"
        ),
        CounterfactualCheckId.PRESENTATION_PERMUTATION: (
            "1c517d250c47f8fa94a2a0874c3c9141d271a229cb7b4dd2295b4637780c5c92"
        ),
        CounterfactualCheckId.PAIRED_CLOCK_SCALE: (
            "6f12f1e20b581940f1395bde9ae8af34053c4b6fd129a880c92b2afc238c8cb0"
        ),
    }
    assert all(item.passed for item in checks)


def test_delay_swap_compares_the_full_action_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A correct swapped delay is insufficient when start/target/end are wrong."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    source = _authenticated_test_source(config)
    real_action_window = leakage.action_window

    def wrong_target(activation_time, delay, timing):
        window = real_action_window(activation_time, delay, timing)
        return replace(window, target=window.target + 1.0)

    monkeypatch.setattr(leakage, "action_window", wrong_target)
    checks = leakage._counterfactual_checks(
        source,
        leakage.LeakageAuditProfileName.TEST,
    )
    delay = next(
        item
        for item in checks
        if item.check_id is leakage.CounterfactualCheckId.TERMINAL_DELAY_SWAP
    )

    assert not delay.passed
    assert delay.temporal_mismatch_count == delay.checked_pairs


def test_presentation_counterfactual_forces_a_nonidentity_permutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even an identity-producing rank schedule must yield a real permutation."""
    import silent_cascade.env.leakage as leakage

    example = next(_authenticated_test_source(_config()).iter_examples())
    rank = iter(range(1_000))
    monkeypatch.setattr(leakage, "sha256_bytes", lambda _payload: f"{next(rank):064x}")

    reordered = leakage._reordered_public(
        example.bundle,
        example.bundle.public.init.episode_public_id,
    )
    original_ids = tuple(event.event_id for event in leakage._fact_events(example.bundle))
    reordered_ids = tuple(event.event_id for event in reordered.events[:-1])

    assert reordered_ids != original_ids


@pytest.mark.parametrize(
    "mutation",
    ("child", "order", "count", "extra", "zero", "hash", "parent", "scale"),
)
def test_clock_corruption_is_refused_before_every_probe_fit(
    mutation: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clock identity/order/count/manifest/parent/scale corruption prohibits all fitting."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    trusted = _authenticated_test_source(config, groups_per_path=20)
    pairs = list(trusted.clock_pairs)
    source_authentication = trusted.authentication
    if mutation == "child":
        child = pairs[0].child
        pairs[0] = leakage.PairedClockAuditPair(
            pairs[0].parent,
            replace(
                child,
                public=replace(
                    child.public,
                    init=replace(child.public.init, memory_capacity=63),
                ),
            ),
        )
    elif mutation == "order":
        pairs.reverse()
    elif mutation == "count":
        pairs.pop()
    elif mutation == "extra":
        parent = tuple(trusted.iter_examples())[20 * 4 + 1]
        pairs.append(
            leakage.PairedClockAuditPair(
                parent,
                leakage.scale_episode_time(
                    parent.bundle,
                    SuiteName.CLOCK_SCALE_10X,
                    "00000000-0000-4000-8000-000000000105",
                ),
            )
        )
    elif mutation == "zero":
        source_authentication = trusted.authentication.model_copy(
            update={"clock_scale_pair_counts": {"scale_0_1x": 0, "scale_10x": 1}}
        )
    elif mutation == "hash":
        pairs[0] = leakage.PairedClockAuditPair(
            pairs[0].parent,
            leakage.scale_episode_time(
                pairs[0].parent.bundle,
                SuiteName.CLOCK_SCALE_0_1X,
                "00000000-0000-4000-8000-000000000103",
            ),
        )
    elif mutation == "parent":
        parent = replace(pairs[0].parent, manifest_rank=999)
        pairs[0] = leakage.PairedClockAuditPair(parent, pairs[0].child)
    elif mutation == "scale":
        pairs[0] = leakage.PairedClockAuditPair(
            pairs[0].parent,
            leakage.scale_episode_time(
                pairs[0].parent.bundle,
                SuiteName.CLOCK_SCALE_10X,
                "00000000-0000-4000-8000-000000000104",
            ),
        )

    class CorruptedClockSource:
        descriptor = trusted.descriptor
        authentication = source_authentication
        validation_config = config
        public_id_seed = trusted.public_id_seed
        episode_count = trusted.episode_count
        publishable = False

        def iter_examples(self):
            return trusted.iter_examples()

        def iter_clock_pairs(self):
            return iter(pairs)

    fit_calls = 0

    def prohibited_fit(*_args: object, **_kwargs: object) -> object:
        nonlocal fit_calls
        fit_calls += 1
        raise AssertionError("clock corruption reached a statistical fit")

    monkeypatch.setattr(leakage, "_fit_predict_batched", prohibited_fit)
    audit_config = config.data.leakage_audit.model_copy(
        update={
            "test": config.data.leakage_audit.test.model_copy(
                update={
                    "episode_count": trusted.episode_count,
                    "permutation_replicates": 1,
                    "minimum_test_examples_per_class": 1,
                }
            )
        }
    )

    with pytest.raises(ValueError):
        leakage.audit_leakage(
            CorruptedClockSource(),
            audit_config,
            leakage.LeakageAuditProfileName.TEST,
            _provenance(_config_sha256(config), trusted),
            tmp_path,
        )
    assert fit_calls == 0


@pytest.mark.parametrize("clock_counts", ((1, 1), (2, 2), (2, 1)))
def test_test_profile_consumes_all_authenticated_positive_clock_pairs(
    clock_counts: tuple[int, int],
) -> None:
    """TEST clock denominators come from independent authentication, not a 1/1 shortcut."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    source = _authenticated_test_source(
        config,
        groups_per_path=20,
        clock_counts=clock_counts,
    )

    authentication = leakage._validate_source_authentication(
        source, leakage.LeakageAuditProfileName.TEST
    )
    leakage._validate_independent_trust_anchor(
        source,
        leakage.LeakageAuditProfileName.TEST,
        _provenance(_config_sha256(config), source),
        authentication,
    )
    result = next(
        item
        for item in leakage._counterfactual_checks(source, leakage.LeakageAuditProfileName.TEST)
        if item.check_id is leakage.CounterfactualCheckId.PAIRED_CLOCK_SCALE
    )

    assert authentication.clock_scale_pair_counts == {
        "scale_0_1x": clock_counts[0],
        "scale_10x": clock_counts[1],
    }
    assert result.checked_pairs == sum(clock_counts)


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
                    "minimum_test_examples_per_class": 1,
                }
            )
        }
    )

    report = audit_leakage(
        injector.apply(clean_source),
        audit_config,
        LeakageAuditProfileName.TEST,
        _provenance(_config_sha256(config), clean_source),
        tmp_path,
        positive_control=injector.control_id,
    )

    assert not report.passed
    assert len(report.positive_controls) == 1
    result = report.positive_controls[0]
    assert audit_config.test.positive_control_permutation_replicates == 199
    assert result.control_id == injector.control_id
    assert result.target_task is injector.target_task
    assert injector.expected_detector_id in result.observed_detector_ids, result
    if injector.control_id == "PC_MANIFEST_ORDER_BY_VARIANT":
        assert result.injected_corpus_sha256 == result.base_subset_corpus_sha256
        assert result.injected_corpus_sha256 == report.corpus_hash
    else:
        assert result.base_subset_corpus_sha256 != result.injected_corpus_sha256
    assert result.balanced_accuracy is not None and result.balanced_accuracy >= 0.95
    assert result.passed


def test_bounded_control_executes_the_exact_4999_replicate_gate_semantics(
    tmp_path: Path,
) -> None:
    """The finite-null Holm gate runs actual fits/results at the frozen denominator."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    clean_source = _authenticated_test_source(config, groups_per_path=20)
    profile = config.data.leakage_audit.test.model_copy(
        update={
            "episode_count": clean_source.episode_count,
            "permutation_replicates": 4_999,
            "positive_control_episode_count": clean_source.episode_count,
            "positive_control_permutation_replicates": 4_999,
            "minimum_test_examples_per_class": 1,
            "enforce_clean_statistical_gate": True,
        }
    )
    audit_config = config.data.leakage_audit.model_copy(update={"test": profile})
    rows = []
    digest = leakage.CorpusHashBuilder(clean_source.episode_count)
    for example in clean_source.iter_examples():
        bundle = example.bundle
        bundle_digest = leakage.episode_sha256(bundle)
        feature_set = leakage.extract_shortcut_features(example, clean_source.episode_count)
        rows.append(
            leakage._StoredExample(
                bundle.public.init.episode_public_id,
                bundle_digest,
                feature_set.audit_group_id,
                bundle.truth.key.suite,
                bundle.truth.recipe.requested_path_length,
                bundle.truth.recipe.variant,
                bundle.truth.relevant_hazard_type,
                example.randomization_block_index,
                example.episode_position,
                tuple(
                    sorted(
                        event.payload.hazard_type
                        for event in leakage._fact_events(bundle)
                        if isinstance(event.payload, leakage.HazardFact)
                    )
                ),
                quartet_member_index=example.quartet_member_index,
            )
        )
        digest.add(
            leakage.CorpusDigestEntry(
                bundle.public.init.episode_public_id,
                bundle_digest,
            )
        )
    leakage._validate_holm_attainability(audit_config, profile)
    result = leakage._execute_positive_control(
        leakage.leak_record_count.apply(clean_source),
        rows,
        leakage.leak_record_count,
        audit_config,
        profile,
        digest.finalize(),
        config,
        tmp_path,
    )

    assert profile.permutation_replicates == 4_999
    assert profile.positive_control_permutation_replicates == 4_999
    assert result.expected_detector_id in result.observed_detector_ids
    assert result.holm_adjusted_p is not None
    assert result.holm_adjusted_p < audit_config.alpha
    assert result.passed


def test_positive_control_rejects_a_passing_probe_with_the_wrong_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observed detector evidence must come from the task/group actually executed."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    clean_source = _authenticated_test_source(config)
    injector = leakage.leak_record_count
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

    def wrong_probe(*_args: object, **_kwargs: object) -> leakage.ShortcutProbeResult:
        return leakage.ShortcutProbeResult(
            task=leakage.ShortcutTask.VARIANT_THREE_WAY,
            feature_group=leakage.ShortcutFeatureGroup.LINK_TOPOLOGY,
            feature_dimension=33,
            train_examples=960,
            test_examples=240,
            train_class_counts={"0": 480, "1": 480},
            test_class_counts={"0": 120, "1": 120},
            raw_accuracy=1.0,
            balanced_accuracy=1.0,
            balanced_chance=1.0 / 3.0,
            raw_permutation_p=0.05,
            holm_adjusted_p=0.001,
            optimizer_iterations=1,
            optimizer_converged=True,
            passed=False,
        )

    monkeypatch.setattr(leakage, "_run_positive_control_probe", wrong_probe)

    with pytest.raises(ValueError, match="detector identity"):
        leakage.audit_leakage(
            injector.apply(clean_source),
            audit_config,
            leakage.LeakageAuditProfileName.TEST,
            _provenance(_config_sha256(config), clean_source),
            tmp_path,
            positive_control=injector.control_id,
        )


def test_positive_control_rejects_generic_unrelated_invariant_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catch-all `invalid` ID cannot authorize an unrelated corruption."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    clean_source = _authenticated_test_source(config)
    injector = leakage.leak_record_count
    real_validate = leakage.validate_episode_invariants

    def unrelated_invalid(bundle, validation_config, strict=True):
        report = real_validate(bundle, validation_config, strict=strict)
        if strict is False:
            return replace(report, valid=False, check_ids=("invalid",))
        return report

    monkeypatch.setattr(leakage, "validate_episode_invariants", unrelated_invalid)
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

    with pytest.raises(ValueError, match="exactly authorized"):
        leakage.audit_leakage(
            injector.apply(clean_source),
            audit_config,
            leakage.LeakageAuditProfileName.TEST,
            _provenance(_config_sha256(config), clean_source),
            tmp_path,
            positive_control=injector.control_id,
        )


def test_positive_control_overlapping_code_is_refused_before_control_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both frozen train and test partitions need disjoint target-feature codes."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    clean_source = _authenticated_test_source(config)
    injector = leakage.leak_record_count
    monkeypatch.setattr(
        leakage,
        "_rewrite_positive_control",
        lambda _injector, example, _position, **_kwargs: example,
    )
    monkeypatch.setattr(
        leakage,
        "_require_exact_control_public_mutation",
        lambda *_args, **_kwargs: None,
    )

    def prohibited_control_fit(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("overlapping positive-control features reached fitting")

    monkeypatch.setattr(leakage, "_run_positive_control_probe", prohibited_control_fit)
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

    with pytest.raises(ValueError, match="disjoint"):
        leakage.audit_leakage(
            injector.apply(clean_source),
            audit_config,
            leakage.LeakageAuditProfileName.TEST,
            _provenance(_config_sha256(config), clean_source),
            tmp_path,
            positive_control=injector.control_id,
        )


def test_frozen_optimizer_fits_three_standardized_contiguous_rank_blocks() -> None:
    """The manifest-order KAT isolates weighting, preprocessing, and the frozen L2 fit."""
    from silent_cascade.env.leakage import (
        ShortcutFeatureGroup,
        _balanced_accuracy,
        _BatchedFeatureReader,
        _continuous_columns,
        _fit_predict_batched,
    )

    count = 1_200
    values = np.zeros((count, 17), dtype=np.float32)
    values[:, -1] = np.arange(count, dtype=np.float32) / (count - 1)
    labels = np.repeat(np.arange(3, dtype=np.int8), count // 3)
    test = np.arange(0, count, 5, dtype=np.int64)
    train = np.setdiff1d(np.arange(count, dtype=np.int64), test, assume_unique=True)
    classes = np.arange(3, dtype=np.int8)
    continuous = _continuous_columns(ShortcutFeatureGroup.ID_POSITION, values.shape[1])
    sweep = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
    expected = (0.745833, 0.837500, 0.912500, 0.970833, 0.995833, 0.995833, 0.995833)
    observed: dict[float, float] = {}
    for penalty, expected_ba in zip(sweep, expected, strict=True):
        config = _config().data.leakage_audit.model_copy(update={"l2_penalty": penalty})
        reader = _BatchedFeatureReader(
            values,
            0,
            values.shape[1],
            None,
            config.feature_batch_size,
            config,
        )
        predictions, _iterations = _fit_predict_batched(
            reader,
            train,
            labels,
            test,
            classes,
            continuous,
            config,
        )
        observed[penalty] = _balanced_accuracy(labels[test], predictions, classes)
        assert observed[penalty] == pytest.approx(expected_ba, abs=5e-7)

    assert max(penalty for penalty, ba in observed.items() if ba >= 0.95) == 0.03


def test_batched_optimizer_matches_dense_fit_without_oversized_reads() -> None:
    """Batch-size changes preserve fits without oversized materialization."""
    from silent_cascade.env.leakage import (
        _BatchedFeatureReader,
        _fit_predict_batched,
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
    continuous = np.ones(values.shape[1], dtype=bool)
    predictions = []
    iterations = []
    for batch_size in (7, 13):
        config = _config().data.leakage_audit.model_copy(update={"feature_batch_size": batch_size})
        reader = _BatchedFeatureReader(
            values,
            0,
            values.shape[1],
            None,
            config.feature_batch_size,
            config,
        )
        actual, iteration_count = _fit_predict_batched(
            reader,
            train,
            labels,
            test,
            classes,
            continuous,
            config,
        )
        predictions.append(actual)
        iterations.append(iteration_count)
        assert reader.max_batch_seen <= config.feature_batch_size

    assert np.array_equal(predictions[0], predictions[1])
    assert iterations[0] == iterations[1]


@pytest.mark.parametrize("failure", ("nonconverged", "nonfinite"))
def test_batched_optimizer_fails_closed_on_invalid_solver_result(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither optimizer exhaustion nor a nonfinite solution can publish evidence."""
    from types import SimpleNamespace

    import silent_cascade.env.leakage as leakage

    values = np.asarray(((0.0, 1.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0)))
    labels = np.asarray((0, 1, 0, 1), dtype=np.int8)
    config = _config().data.leakage_audit.model_copy(update={"feature_batch_size": 2})
    reader = leakage._BatchedFeatureReader(values, 0, 2, None, 2, config)
    result = SimpleNamespace(
        success=failure != "nonconverged",
        x=(np.asarray((np.nan,)) if failure == "nonfinite" else np.asarray((0.0,))),
        nit=0,
    )
    monkeypatch.setattr(leakage, "minimize", lambda *_args, **_kwargs: result)

    with pytest.raises(ValueError, match="optimizer failed"):
        leakage._fit_predict_batched(
            reader,
            np.asarray((0, 1), dtype=np.int64),
            labels,
            np.asarray((2, 3), dtype=np.int64),
            np.asarray((0, 1), dtype=np.int8),
            np.ones(2, dtype=bool),
            config,
        )


def test_holm_and_threshold_boundaries_are_exact() -> None:
    """Holm monotonicity and strict alpha/chance comparisons are protocol KATs."""
    from silent_cascade.env.leakage import (
        ShortcutFeatureGroup,
        ShortcutProbeResult,
        ShortcutTask,
        _holm,
    )

    def probe(raw_p: float, balanced: float = 0.75) -> ShortcutProbeResult:
        return ShortcutProbeResult(
            task=ShortcutTask.POSITIVE_BINARY,
            feature_group=ShortcutFeatureGroup.COUNTS,
            feature_dimension=4,
            train_examples=8,
            test_examples=4,
            train_class_counts={"0": 4, "1": 4},
            test_class_counts={"0": 2, "1": 2},
            raw_accuracy=balanced,
            balanced_accuracy=balanced,
            balanced_chance=0.5,
            raw_permutation_p=raw_p,
            holm_adjusted_p=1.0,
            optimizer_iterations=1,
            optimizer_converged=True,
            passed=True,
        )

    adjusted = _holm([probe(0.001), probe(0.01), probe(0.04)], 0.05)
    assert [item.holm_adjusted_p for item in adjusted] == pytest.approx((0.003, 0.02, 0.04))
    assert not any(item.passed for item in adjusted)
    assert _holm([probe(0.05)], 0.05)[0].passed
    assert _holm([probe(0.001, balanced=0.5)], 0.05)[0].passed


def test_full_profile_phases_stream_under_resident_ceiling_and_cleanup(tmp_path: Path) -> None:
    """100k storage/write/moments/fit/predict phases stay bounded and clean up."""
    import psutil

    from silent_cascade.env.leakage import (
        _allocate_feature_store,
        _batched_moments,
        _BatchedFeatureReader,
        _fit_predict_batched,
        _write_feature_batch,
    )

    config = _config().data.leakage_audit
    feature_path = tmp_path / "full-profile.f32"
    rss_by_phase = {}
    _allocate_feature_store(feature_path, 100_000)
    rss_by_phase["allocate"] = psutil.Process().memory_info().rss
    _write_feature_batch(
        feature_path,
        100_000,
        0,
        np.zeros((4_096, 1_560), dtype=np.float32),
        config,
    )
    rss_by_phase["write"] = psutil.Process().memory_info().rss
    values = np.memmap(feature_path, dtype=np.float32, mode="r", shape=(100_000, 1_560))
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
    rss_by_phase["moments"] = psutil.Process().memory_info().rss
    narrow_reader = _BatchedFeatureReader(
        values,
        0,
        2,
        None,
        config.feature_batch_size,
        config,
    )
    labels = np.arange(100_000, dtype=np.int8) % 2
    predictions, _iterations = _fit_predict_batched(
        narrow_reader,
        np.arange(80_000, dtype=np.int64),
        labels,
        np.arange(80_000, 100_000, dtype=np.int64),
        np.asarray((0, 1), dtype=np.int8),
        np.ones(2, dtype=bool),
        config,
    )
    rss_by_phase["fit_predict"] = psutil.Process().memory_info().rss

    assert feature_path.stat().st_size == 624_000_000
    assert np.array_equal(mean, np.zeros(1_560))
    assert np.array_equal(std, np.ones(1_560))
    assert np.all(zero)
    assert reader.max_batch_seen <= 4_096
    assert predictions.shape == (20_000,)
    assert narrow_reader.max_batch_seen <= 4_096
    assert all(value <= config.max_resident_working_bytes for value in rss_by_phase.values())
    values._mmap.close()
    del values
    feature_path.unlink()
    rss_by_phase["cleanup"] = psutil.Process().memory_info().rss
    assert not feature_path.exists()
    assert rss_by_phase["cleanup"] <= config.max_resident_working_bytes


@pytest.mark.parametrize(
    "phase",
    (
        "extraction",
        "moments",
        "optimization",
        "permutation",
        "control",
        "counterfactual",
        "cleanup",
    ),
)
def test_over_rss_at_every_audit_phase_cleans_without_a_partial_report(
    phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every production phase fails closed, and cleanup precedes its own RSS check."""
    import silent_cascade.env.leakage as leakage

    config = _config()
    clean_source = _authenticated_test_source(config, groups_per_path=20)
    source = leakage.leak_record_count.apply(clean_source) if phase == "control" else clean_source
    audit_config = config.data.leakage_audit.model_copy(
        update={
            "test": config.data.leakage_audit.test.model_copy(
                update={
                    "episode_count": clean_source.episode_count,
                    "permutation_replicates": 1,
                    "positive_control_episode_count": clean_source.episode_count,
                    "positive_control_permutation_replicates": 1,
                    "minimum_test_examples_per_class": 1,
                }
            )
        }
    )
    observed: list[str] = []

    def injected_guard(_config: object, current_phase: str = "unspecified") -> None:
        observed.append(current_phase)
        if current_phase == phase:
            raise MemoryError("leakage audit resident working-set ceiling exceeded")

    monkeypatch.setattr(leakage, "_resource_guard", injected_guard)
    if phase in {"moments", "optimization", "permutation", "control"}:
        monkeypatch.setattr(leakage, "_counterfactual_checks", lambda *_args: ())
    if phase in {"control", "cleanup"}:
        monkeypatch.setattr(leakage, "_run_probes", lambda *_args, **_kwargs: [])
    if phase == "permutation":

        def exact_fit(
            _reader: object,
            _train: object,
            labels: np.ndarray,
            test: np.ndarray,
            _classes: object,
            _continuous: object,
            _config: object,
        ) -> tuple[np.ndarray, int]:
            return labels[test].copy(), 1

        monkeypatch.setattr(leakage, "_fit_predict_batched", exact_fit)
    if phase == "counterfactual":
        monkeypatch.setattr(
            leakage,
            "_run_probes",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("counterfactual RSS failure reached a fit")
            ),
        )

    with pytest.raises(MemoryError, match="resident working-set ceiling"):
        leakage.audit_leakage(
            source,
            audit_config,
            leakage.LeakageAuditProfileName.TEST,
            _provenance(_config_sha256(config), clean_source),
            tmp_path,
            positive_control=("PC_COUNT_BY_LABEL" if phase == "control" else None),
        )

    assert phase in observed
    assert not list(tmp_path.iterdir())
