from pathlib import Path

import pytest
import yaml
from pydantic import Field

from silent_cascade.config import ProjectConfig, parse_set_override, resolve_config
from silent_cascade.errors import ConfigurationError
from silent_cascade.validation import StrictModel


class MergeConfig(StrictModel):
    literal_merge: int = Field(alias="<<")
    merged_value: int


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


@pytest.mark.parametrize(
    ("text", "key"),
    [
        (
            "schema_version: 1\nexperiment_version: v1\nexperiment_version: v2\n",
            "experiment_version",
        ),
        (
            "schema_version: 1\nexperiment_version: v1\n"
            "limits:\n  batch_size: 1\n  batch_size: 2\n",
            "batch_size",
        ),
        (
            "schema_version: 1\nexperiment_version: v1\n"
            "limits:\n  unknown_limit: true\n"
            "limits:\n  batch_size: 32\n",
            "limits",
        ),
    ],
)
def test_resolve_config_rejects_duplicate_yaml_keys_with_file_and_key_context(
    tmp_path: Path, text: str, key: str
) -> None:
    config = write_yaml(tmp_path / "duplicate.yaml", text)

    with pytest.raises(ConfigurationError, match="duplicate configuration key") as raised:
        resolve_config(ProjectConfig, [config])

    assert raised.value.message == "duplicate configuration key"
    assert raised.value.context["path"] == str(config)
    assert raised.value.context["key"] == key


def test_resolve_config_keeps_root_string_key_validation(tmp_path: Path) -> None:
    config = write_yaml(tmp_path / "non-string-key.yaml", "1: value\n")

    with pytest.raises(
        ConfigurationError, match="configuration root must be a string-keyed mapping"
    ):
        resolve_config(ProjectConfig, [config])


def test_resolve_config_wraps_unhashable_yaml_keys_as_configuration_error(tmp_path: Path) -> None:
    config = write_yaml(tmp_path / "unhashable-key.yaml", "? [one, two]\n: value\n")

    with pytest.raises(ConfigurationError, match="configuration load failed"):
        resolve_config(ProjectConfig, [config])


def test_parse_set_override_rejects_duplicate_mapping_keys() -> None:
    with pytest.raises(ConfigurationError, match="duplicate configuration key") as raised:
        parse_set_override("limits={batch_size: 1, batch_size: 2}")

    assert raised.value.context["expression"] == "limits={batch_size: 1, batch_size: 2}"
    assert raised.value.context["key"] == "batch_size"


def test_resolve_config_rejects_repeated_yaml_merge_keys_with_file_context(
    tmp_path: Path,
) -> None:
    config = write_yaml(
        tmp_path / "repeated-merge.yaml",
        "schema_version: 1\nexperiment_version: v1\nlimits:\n"
        "  <<: {batch_size: 32}\n"
        "  <<: {primary_memory_records: 64}\n",
    )

    with pytest.raises(ConfigurationError, match="duplicate configuration key") as raised:
        resolve_config(ProjectConfig, [config])

    assert raised.value.context == {"path": str(config), "key": "<<"}


def test_parse_set_override_rejects_repeated_yaml_merge_keys_with_expression_context() -> None:
    expression = "limits={<<: {batch_size: 32}, <<: {primary_memory_records: 64}}"

    with pytest.raises(ConfigurationError, match="duplicate configuration key") as raised:
        parse_set_override(expression)

    assert raised.value.context == {"expression": expression, "key": "<<"}


def test_resolve_config_allows_quoted_merge_literal_with_merge_directive(
    tmp_path: Path,
) -> None:
    config = write_yaml(
        tmp_path / "merge-literal.yaml",
        '"<<": 7\n<<: {merged_value: 9}\n',
    )

    resolved = resolve_config(MergeConfig, [config])

    assert yaml.safe_load(config.read_text(encoding="utf-8")) == {
        "<<": 7,
        "merged_value": 9,
    }
    assert resolved.config.literal_merge == 7
    assert resolved.config.merged_value == 9


def test_parse_set_override_allows_quoted_merge_literal_with_merge_directive() -> None:
    expression = 'config={"<<": 7, <<: {merged_value: 9}}'

    path, value = parse_set_override(expression)

    assert path == ("config",)
    assert value == yaml.safe_load(expression.split("=", 1)[1])


def test_resolve_config_allows_explicit_key_to_override_yaml_merge(tmp_path: Path) -> None:
    config = write_yaml(
        tmp_path / "merge-override.yaml",
        "schema_version: 1\nexperiment_version: v1\nlimits:\n"
        "  <<: {batch_size: 32}\n"
        "  batch_size: 64\n",
    )

    resolved = resolve_config(ProjectConfig, [config])

    assert resolved.config.limits.batch_size == 64


def test_resolve_config_keeps_safe_loader_merge_sequence_precedence(tmp_path: Path) -> None:
    config = write_yaml(
        tmp_path / "merge-sequence.yaml",
        "schema_version: 1\nexperiment_version: v1\nlimits:\n"
        "  <<:\n"
        "    - {batch_size: 32}\n"
        "    - {batch_size: 64}\n",
    )

    resolved = resolve_config(ProjectConfig, [config])

    assert resolved.config.limits.batch_size == 32


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "true"),
        ("runtime.primary_offline", "1"),
        ("runtime.allow_mps_fallback", "0"),
        ("runtime.primary_foundation_model_calls", "false"),
        ("limits.retain_all_failure_traces", "1"),
    ],
)
def test_literal_fields_reject_bool_integer_equivalents(
    tmp_path: Path, field: str, value: str
) -> None:
    section, _, name = field.partition(".")
    field_yaml = f"{field}: {value}" if not name else f"{section}:\n  {name}: {value}"
    schema_version_yaml = "" if field == "schema_version" else "schema_version: 1\n"
    config = write_yaml(
        tmp_path / "coercion.yaml",
        f"{schema_version_yaml}experiment_version: v1\n{field_yaml}\n",
    )
    with pytest.raises(ConfigurationError, match="configuration validation failed"):
        resolve_config(ProjectConfig, [config])
