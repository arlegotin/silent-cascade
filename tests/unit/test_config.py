from pathlib import Path

import pytest

from silent_cascade.config import ProjectConfig, resolve_config
from silent_cascade.errors import ConfigurationError


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_resolve_config_deep_merges_in_declared_order_then_applies_overrides(
    tmp_path: Path,
) -> None:
    base = write_yaml(
        tmp_path / "base.yaml",
        """
schema_version: 1
experiment_version: v1
limits:
  batch_size: 128
  primary_memory_records: 64
""",
    )
    layer = write_yaml(
        tmp_path / "layer.yaml",
        """
limits:
  batch_size: 64
""",
    )
    resolved = resolve_config(ProjectConfig, [base, layer], set_overrides=["limits.batch_size=32"])
    assert resolved.config.limits.batch_size == 32
    assert resolved.config.limits.primary_memory_records == 64
    assert resolved.source_paths == (base, layer)


def test_resolve_config_rejects_unknown_keys_and_nonfinite_numbers(tmp_path: Path) -> None:
    unknown = write_yaml(
        tmp_path / "unknown.yaml", "schema_version: 1\nexperiment_version: v1\nunknown: true\n"
    )
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(ProjectConfig, [unknown])
    nonfinite = write_yaml(
        tmp_path / "nonfinite.yaml",
        "schema_version: 1\nexperiment_version: v1\nlimits:\n  batch_size: .nan\n",
    )
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(ProjectConfig, [nonfinite])


def test_equivalent_configs_have_identical_canonical_bytes_and_hash(tmp_path: Path) -> None:
    first = write_yaml(
        tmp_path / "first.yaml",
        "schema_version: 1\nexperiment_version: v1\nruntime:\n  dtype: float32\n",
    )
    second = write_yaml(
        tmp_path / "second.yaml",
        "runtime:\n  dtype: float32\nexperiment_version: v1\nschema_version: 1\n",
    )
    resolved_first = resolve_config(ProjectConfig, [first])
    resolved_second = resolve_config(ProjectConfig, [second])
    assert resolved_first.canonical_json == resolved_second.canonical_json
    assert resolved_first.sha256 == resolved_second.sha256
    assert b'"max_trainable_parameters":5000000' in resolved_first.canonical_json
    assert b'"retain_all_failure_traces":true' in resolved_first.canonical_json


@pytest.mark.parametrize(
    "override", ["", "missing_equals", ".leading=1", "trailing.=1", "double..dot=1"]
)
def test_invalid_set_override_raises_typed_error(tmp_path: Path, override: str) -> None:
    base = write_yaml(tmp_path / "base.yaml", "schema_version: 1\nexperiment_version: v1\n")
    with pytest.raises(ConfigurationError, match="invalid --set override"):
        resolve_config(ProjectConfig, [base], set_overrides=[override])


def test_committed_paths_must_be_relative(tmp_path: Path) -> None:
    config = write_yaml(
        tmp_path / "absolute.yaml",
        "schema_version: 1\nexperiment_version: v1\npaths:\n  runs: /tmp/runs\n",
    )
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(ProjectConfig, [config])


def test_runtime_device_preference_must_include_cpu(tmp_path: Path) -> None:
    config = write_yaml(
        tmp_path / "mps-only.yaml",
        "schema_version: 1\nexperiment_version: v1\nruntime:\n  device_preference: [mps]\n",
    )
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(ProjectConfig, [config])
