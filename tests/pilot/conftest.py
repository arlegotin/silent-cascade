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
