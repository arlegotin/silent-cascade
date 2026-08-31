"""Behavioral contracts for private OFD bundles and canonical artifacts."""

import json
import math
from dataclasses import replace

import pytest

from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    CorpusDigestEntry,
    CorpusHashBuilder,
    EpisodeBundle,
    EpisodeKey,
    EpisodeRecipe,
    EpisodeTruth,
    EpisodeVariant,
    IndependentEpisodeCoordinate,
    MatchedEpisodeCoordinate,
    PublicEpisode,
    PublicEpisodeArtifact,
    StressMetadata,
    canonical_episode_bytes,
    corpus_sha256,
    episode_from_bytes,
    episode_sha256,
    public_projection,
)
from silent_cascade.errors import EpisodeInvariantError
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.schemas import (
    ActivationPayload,
    AgentInit,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    LinkFact,
)


@pytest.fixture
def positive_bundle() -> EpisodeBundle:
    public = PublicEpisode(
        init=AgentInit("00000000-0000-4000-8000-000000000001", 64, 4, 0.0),
        events=(
            ExternalEvent(10, 0.25, ExternalEventKind.FACT, LinkFact(1, 2)),
            ExternalEvent(11, 0.50, ExternalEventKind.FACT, HazardFact(2, 1, 3.0)),
            ExternalEvent(12, 1.00, ExternalEventKind.ACTIVATE, ActivationPayload(1)),
        ),
    )
    truth = EpisodeTruth(
        key=EpisodeKey(
            "ofd-v1",
            SplitNamespace.PHASE1_GATE,
            SuiteName.IID_PRIMARY,
            987654321,
            MatchedEpisodeCoordinate("matched", 3, 2),
        ),
        recipe=EpisodeRecipe(2, EpisodeVariant.POSITIVE, 0, SuiteName.IID_PRIMARY, 4),
        relevant_node_path=(1, 2),
        relevant_record_ids=(10, 11),
        terminal_record_id=11,
        relevant_hazard_type=1,
        private_terminal=ExternalEvent(99, 4.0, ExternalEventKind.OUTCOME, None),
        activation_time=1.0,
        episode_delay=3.0,
        action_window_start=3.2,
        action_window_end=3.8,
        action_target=3.5,
        rejection_count=0,
        rejection_reasons=(),
    )
    return EpisodeBundle(public, truth)


def test_public_projection_has_no_private_truth_fields(positive_bundle: EpisodeBundle) -> None:
    public = public_projection(positive_bundle)
    dumped = repr(public)
    for forbidden in (
        "root_seed",
        "episode_index",
        "positive",
        "relevant_record_ids",
        "terminal_time",
        "action_window",
        "rejection",
    ):
        assert forbidden not in dumped
    assert {event.kind for event in public.events} <= {
        ExternalEventKind.FACT,
        ExternalEventKind.ACTIVATE,
    }


def test_public_artifact_and_public_error_payload_do_not_leak_private_sentinel(
    positive_bundle: EpisodeBundle,
) -> None:
    sentinel = "PRIVATE-ROOT-SEED-998877"
    truth = positive_bundle.truth
    private = EpisodeTruth(
        key=EpisodeKey(
            truth.key.generator_version,
            truth.key.split_namespace,
            truth.key.suite,
            998877,
            IndependentEpisodeCoordinate("independent", 987654, 123),
        ),
        recipe=truth.recipe,
        relevant_node_path=truth.relevant_node_path,
        relevant_record_ids=truth.relevant_record_ids,
        terminal_record_id=truth.terminal_record_id,
        relevant_hazard_type=truth.relevant_hazard_type,
        private_terminal=truth.private_terminal,
        activation_time=truth.activation_time,
        episode_delay=truth.episode_delay,
        action_window_start=truth.action_window_start,
        action_window_end=truth.action_window_end,
        action_target=truth.action_target,
        rejection_count=998877,
        rejection_reasons=(sentinel,),
    )
    bundle = EpisodeBundle(positive_bundle.public, private)
    public = public_projection(bundle)
    serialized = canonical_json_bytes(PublicEpisodeArtifact.from_public(public)).decode()

    assert sentinel not in repr(public)
    assert sentinel not in serialized
    with pytest.raises(ValueError) as raised:
        CorpusDigestEntry(sentinel, "11" * 32)
    assert sentinel not in repr(raised.value)


@pytest.mark.parametrize("variant", tuple(EpisodeVariant))
def test_episode_variants_round_trip_through_private_canonical_serialization(
    positive_bundle: EpisodeBundle, variant: EpisodeVariant
) -> None:
    truth = positive_bundle.truth
    if variant is EpisodeVariant.POSITIVE:
        private_truth = truth
    else:
        private_truth = EpisodeTruth(
            key=truth.key,
            recipe=EpisodeRecipe(2, variant, 0, SuiteName.IID_PRIMARY, 4),
            relevant_node_path=truth.relevant_node_path,
            relevant_record_ids=truth.relevant_record_ids,
            terminal_record_id=truth.terminal_record_id,
            relevant_hazard_type=None,
            private_terminal=ExternalEvent(99, 4.0, ExternalEventKind.END, None),
            activation_time=truth.activation_time,
            episode_delay=truth.episode_delay,
            action_window_start=None,
            action_window_end=None,
            action_target=None,
            rejection_count=truth.rejection_count,
            rejection_reasons=truth.rejection_reasons,
        )
    bundle = EpisodeBundle(
        positive_bundle.public,
        private_truth
        if variant is not EpisodeVariant.POSITIVE
        else replace(private_truth, recipe=EpisodeRecipe(2, variant, 0, SuiteName.IID_PRIMARY, 4)),
    )

    assert episode_from_bytes(canonical_episode_bytes(bundle)) == bundle


def test_private_episode_artifact_is_byte_identical_after_round_trip(
    positive_bundle: EpisodeBundle,
) -> None:
    payload = canonical_episode_bytes(positive_bundle)

    assert episode_from_bytes(payload) == positive_bundle
    assert canonical_episode_bytes(episode_from_bytes(payload)) == payload
    assert episode_sha256(positive_bundle) == episode_sha256(episode_from_bytes(payload))
    decoded = json.loads(payload)
    assert decoded["schema_version"] == 1
    assert decoded["generator_version"] == "ofd-v1"


@pytest.mark.parametrize("mutation", ["unknown", "event_order", "duplicate_id", "nonfinite"])
def test_episode_artifact_rejects_invalid_serialized_data(
    positive_bundle: EpisodeBundle, mutation: str
) -> None:
    decoded = json.loads(canonical_episode_bytes(positive_bundle))
    if mutation == "unknown":
        decoded["unexpected"] = True
    elif mutation == "event_order":
        decoded["public"]["events"][:2] = reversed(decoded["public"]["events"][:2])
    elif mutation == "duplicate_id":
        decoded["public"]["events"][1]["event_id"] = decoded["public"]["events"][0]["event_id"]
    else:
        decoded["truth"]["episode_delay"] = math.inf

    with pytest.raises((TypeError, ValueError, EpisodeInvariantError)):
        episode_from_bytes(json.dumps(decoded).encode("utf-8"))


def test_private_episode_artifact_rejects_unknown_variant_spelling(
    positive_bundle: EpisodeBundle,
) -> None:
    decoded = json.loads(canonical_episode_bytes(positive_bundle))
    decoded["truth"]["recipe"]["variant"] = "hidden_positive"

    with pytest.raises((TypeError, ValueError, EpisodeInvariantError)):
        episode_from_bytes(json.dumps(decoded).encode("utf-8"))


def test_bundle_rejects_terminal_event_in_public_timeline(positive_bundle: EpisodeBundle) -> None:
    with pytest.raises((TypeError, ValueError, EpisodeInvariantError)):
        PublicEpisode(
            positive_bundle.public.init,
            (*positive_bundle.public.events, positive_bundle.truth.private_terminal),
        )


def test_stress_metadata_is_private_and_limited_to_its_matching_suite(
    positive_bundle: EpisodeBundle,
) -> None:
    with pytest.raises((TypeError, ValueError, EpisodeInvariantError)):
        replace(positive_bundle.truth, stress_metadata=StressMetadata(65))


def test_corpus_hash_builder_matches_known_answer_and_is_single_use() -> None:
    entries = (
        CorpusDigestEntry("00000000-0000-4000-8000-000000000001", "11" * 32),
        CorpusDigestEntry("00000000-0000-4000-8000-000000000002", "22" * 32),
    )
    builder = CorpusHashBuilder(2)
    for entry in entries:
        builder.add(entry)

    expected = "d38af712baf94c88101b00aabfba0178e9e5957d8faf1ea58f6faa83d37dd41f"
    assert builder.finalize() == expected
    assert corpus_sha256(entries, expected_count=2) == expected
    with pytest.raises((RuntimeError, ValueError)):
        builder.finalize()


@pytest.mark.parametrize(
    "entry",
    [
        CorpusDigestEntry("00000000-0000-4000-8000-000000000001", "11" * 32),
        CorpusDigestEntry("00000000-0000-4000-8000-000000000001", "11" * 32),
    ],
)
def test_corpus_hash_builder_rejects_duplicate_public_ids(entry: CorpusDigestEntry) -> None:
    builder = CorpusHashBuilder(2)
    builder.add(entry)
    with pytest.raises((RuntimeError, ValueError)):
        builder.add(entry)
