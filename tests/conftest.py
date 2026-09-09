"""Shared hand-authored checkpoint fixtures; no generator or oracle calls."""

import json
from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.episode import (
    EpisodeBundle,
    EpisodeKey,
    EpisodeRecipe,
    EpisodeTruth,
    EpisodeVariant,
    MatchedEpisodeCoordinate,
    PublicEpisode,
)
from silent_cascade.eventflow.config import Phase2Config
from silent_cascade.schemas import (
    ActivationPayload,
    AgentInit,
    ExternalEvent,
    ExternalEventKind,
    HazardFact,
    LinkFact,
    SafeFact,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures/phase2"
CASES = json.loads((FIXTURES / "scripted_cases.json").read_text())["cases"]


@pytest.fixture(scope="module")
def config():
    return resolve_config(
        Phase2Config,
        [
            Path("configs/base.yaml"),
            Path("configs/data/primary.yaml"),
            Path("configs/model/event_flow.yaml"),
        ],
    ).config.event_flow


def bundle_for(case):
    payloads = {"link": LinkFact, "hazard": HazardFact, "safe": SafeFact}
    facts = tuple(
        ExternalEvent(record, time, ExternalEventKind.FACT, payloads[kind](**payload))
        for record, time, kind, payload in case["facts"]
    )
    event_id, time, start = case["activation"]
    public = PublicEpisode(
        AgentInit("00000000-0000-4000-8000-000000000001", 64, 4, 0.0),
        (
            *facts,
            ExternalEvent(event_id, time, ExternalEventKind.ACTIVATE, ActivationPayload(start)),
        ),
    )
    positive = case["name"] == "positive"
    truth = EpisodeTruth(
        EpisodeKey(
            "ofd-v1",
            SplitNamespace.DEBUG,
            SuiteName.IID_PRIMARY,
            1,
            MatchedEpisodeCoordinate("matched", 0, 0),
        ),
        EpisodeRecipe(2, EpisodeVariant(case["name"]), 0, SuiteName.IID_PRIMARY, 0),
        (0, 1, 2),
        (8, 3, 5) if positive else (8, 3),
        5 if positive else None,
        2 if positive else None,
        ExternalEvent(
            10, 3.0, ExternalEventKind.OUTCOME if positive else ExternalEventKind.END, None
        ),
        1.0,
        2.0,
        2.5 if positive else None,
        2.8 if positive else None,
        2.65 if positive else None,
        0,
        (),
    )
    return EpisodeBundle(public, truth)


def episode(links=0):
    facts = tuple(
        ExternalEvent(i + 1, 0.0, ExternalEventKind.FACT, LinkFact(i, i + 1)) for i in range(links)
    )
    facts += (ExternalEvent(links + 1, 0.0, ExternalEventKind.FACT, HazardFact(links, 2, 100.0)),)
    return EpisodeBundle(
        PublicEpisode(
            AgentInit("public-runtime-safety", 64, 4, 0.0),
            (
                *facts,
                ExternalEvent(links + 2, 1.0, ExternalEventKind.ACTIVATE, ActivationPayload(0)),
            ),
        ),
        EpisodeTruth(
            EpisodeKey(
                "ofd-v1",
                SplitNamespace.DEBUG,
                SuiteName.IID_PRIMARY,
                998877,
                MatchedEpisodeCoordinate("matched", 0, 0),
            ),
            EpisodeRecipe(max(1, links), EpisodeVariant.POSITIVE, 0, SuiteName.IID_PRIMARY, 0),
            tuple(range(links + 1)) if links else (63, 0),
            tuple(range(1, links + 2)),
            links + 1,
            2,
            ExternalEvent(999, 101.0, ExternalEventKind.OUTCOME, None),
            1.0,
            100.0,
            76.0,
            91.0,
            83.5,
            1,
            ("PRIVATE-LABEL-SENTINEL",),
        ),
    )
