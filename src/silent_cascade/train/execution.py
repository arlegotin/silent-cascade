"""Authenticate the executing production producer before compute or publication."""

from pathlib import Path

from silent_cascade.config import ResolvedConfig
from silent_cascade.errors import ProvenanceError
from silent_cascade.train.config import Phase3Config
from silent_cascade.train.curriculum_data import ComponentManifest
from silent_cascade.train.provenance import authenticate_source, validate_config


def authenticate_production_execution(
    config: ResolvedConfig[Phase3Config],
    *,
    manifest: ComponentManifest,
    source_commit: str,
) -> None:
    """Reject misattributed production execution; debug is non-acceptance."""
    if not config.config.training.is_production:
        return
    if manifest.publication != "production" or manifest.source_revision != source_commit:
        raise ProvenanceError("Production execution manifest/source mismatch")
    root = Path(__file__).resolve().parents[3]
    authenticate_source(root, source_commit, manifest.plan_revision)
    validate_config(config, root, allow_empty_sources=True)
    if manifest.config_hash != config.sha256:
        raise ProvenanceError("Production execution manifest/configuration mismatch")
