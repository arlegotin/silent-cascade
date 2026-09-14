from pathlib import Path

import pytest

from silent_cascade.config import resolve_config
from silent_cascade.train.config import Phase3Config

PHASE3_CONFIG_PATHS = (
    Path("configs/base.yaml"),
    Path("configs/data/primary.yaml"),
    Path("configs/model/event_flow.yaml"),
    Path("configs/model/neural_components.yaml"),
    Path("configs/train/one_hop.yaml"),
)


@pytest.fixture(scope="module")
def resolved_neural_config():
    return resolve_config(Phase3Config, PHASE3_CONFIG_PATHS)


@pytest.fixture(scope="module")
def neural_config(resolved_neural_config):
    return resolved_neural_config.config
