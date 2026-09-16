import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from silent_cascade.config import resolve_config
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.train.config import Phase3Config


@pytest.fixture
def jump_fixtures():
    # Reuse the existing public jump fixtures without making tests a package.
    import importlib.util

    path = Path(__file__).parents[1] / "unit/test_jumps.py"
    spec = importlib.util.spec_from_file_location("pilot_jump_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def neural_config():
    return resolve_config(
        Phase3Config,
        tuple(
            Path(p)
            for p in (
                "configs/base.yaml",
                "configs/data/primary.yaml",
                "configs/model/event_flow.yaml",
                "configs/model/neural_components.yaml",
                "configs/train/one_hop.yaml",
            )
        ),
    ).config


@pytest.fixture
def model(neural_config):
    torch.manual_seed(11)
    return EventFlowModel(neural_config.neural)


class ControlledModel(EventFlowModel):
    """Real encoders/jumps, with explicitly chosen public prediction logits."""

    def __init__(self, config):
        super().__init__(config)
        self.role = 1
        self.focus = 9
        self.hazard = 3
        self.status = 0
        self.action_logits = [0.0, 8.0, 0.0, 0.0, 10.0]
        self.retrieval_logits = [0.0, 0.0, 9.0, 0.0] + [0.0] * 60
        self.guard_rate = math.log(3.0) / 0.1
        self.dormant = False
        self.append = True
        self.continue_search = False
        self.calls = []

    def set_hazard_class(self, value):
        self.hazard = value

    def set_action_logits(self, values):
        self.action_logits = values

    def preview_and_control(self, context):
        self.calls.append(("control", context))
        preview, parameters = super().preview_and_control(context)
        targets = torch.full_like(parameters.guard_targets, 0.5 if self.dormant else 1.5)
        return preview, replace(
            parameters,
            guard_targets=targets,
            guard_rates=torch.full_like(targets, self.guard_rate),
        )

    def recall_scores(self, context):
        from silent_cascade.memory.retrieval import RetrievalScores

        self.calls.append(("recall", context))
        scores = context.workspace.latent.new_tensor([self.retrieval_logits])
        return RetrievalScores(
            scores, scores.masked_fill(~context.eligibility, -torch.inf), context.eligibility
        )

    def compose(self, context):
        self.calls.append(("compose", context))
        predictions = super().compose(context)

        def logits(size, choice):
            result = context.workspace.latent.new_zeros((1, size))
            result[0, choice] = 8.0
            return result

        return replace(
            predictions,
            role_logits=logits(5, self.role),
            next_focus_logits=logits(64, self.focus),
            hazard_logits=logits(4, self.hazard),
            status_logits=logits(3, self.status),
            normalized_deadline=context.workspace.latent.new_tensor([7.0]),
            confidence_logit=context.workspace.latent.new_tensor([0.0]),
            append_support_logit=context.workspace.latent.new_tensor(
                [1.0 if self.append else -1.0]
            ),
            continue_search_logit=context.workspace.latent.new_tensor(
                [1.0 if self.continue_search else -1.0]
            ),
        )

    def action(self, context):
        from silent_cascade.models.heads import ActionPredictions

        self.calls.append(("action", context))
        return ActionPredictions(
            context.workspace.latent.new_tensor([self.action_logits]),
            context.workspace.latent.new_tensor([0.99]),
        )


@pytest.fixture
def controlled_model(neural_config):
    torch.manual_seed(11)
    return ControlledModel(neural_config.neural)


class RuntimeCase:
    def __init__(self, model, engine_config):
        from silent_cascade.eventflow.neural import NeuralModelIdentity
        from silent_cascade.schemas import (
            ActivationPayload,
            AgentInit,
            ExternalEvent,
            ExternalEventKind,
            HazardFact,
            LinkFact,
            SafeFact,
        )

        self.init = AgentInit("00000000-0000-4000-8000-000000000001", 64, 4, 0.0)
        self.events = (
            ExternalEvent(10, 1.0, ExternalEventKind.FACT, LinkFact(1, 2)),
            ExternalEvent(20, 2.0, ExternalEventKind.FACT, HazardFact(2, 3, 24.0)),
            ExternalEvent(30, 3.0, ExternalEventKind.FACT, HazardFact(4, 0, 24.0)),
            ExternalEvent(40, 4.0, ExternalEventKind.FACT, SafeFact(5)),
            ExternalEvent(50, 5.0, ExternalEventKind.ACTIVATE, ActivationPayload(1)),
        )
        self.identity = NeuralModelIdentity.from_model(model, source_revision="a" * 40)
        self.safe_events = (
            *self.events[:-1],
            replace(self.events[-1], payload=ActivationPayload(5)),
        )
        self.disconnected_events = (
            *self.events[:-1],
            replace(self.events[-1], payload=ActivationPayload(6)),
        )
        self.diagnostics = ()
        self.engine_config = engine_config

    def run(self, agent, *, negative=False, terminal_delay=24.0):
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
        from silent_cascade.eventflow.engine import EventEngine
        from silent_cascade.schemas import ExternalEvent, ExternalEventKind

        truth = EpisodeTruth(
            EpisodeKey(
                "ofd-v1",
                SplitNamespace.DEBUG,
                SuiteName.IID_PRIMARY,
                1,
                MatchedEpisodeCoordinate("matched", 0, 0),
            ),
            EpisodeRecipe(
                1,
                EpisodeVariant.DISCONNECTED_NEGATIVE if negative else EpisodeVariant.POSITIVE,
                0,
                SuiteName.IID_PRIMARY,
                0,
            ),
            (1, 2),
            (10, 20),
            None if negative else 20,
            None if negative else 3,
            ExternalEvent(
                99,
                5.0 + terminal_delay,
                ExternalEventKind.END if negative else ExternalEventKind.OUTCOME,
                None,
            ),
            5.0,
            terminal_delay,
            None if negative else 23.0,
            None if negative else 26.6,
            None if negative else 24.8,
            0,
            (),
        )
        engine = EventEngine(self.engine_config)
        result = engine.run_episode(
            EpisodeBundle(PublicEpisode(self.init, self.events), truth), agent
        )
        self.diagnostics = agent.diagnostic_snapshot()
        return result


@pytest.fixture
def runtime_case(controlled_model, neural_config):
    return RuntimeCase(controlled_model, neural_config.event_flow)
