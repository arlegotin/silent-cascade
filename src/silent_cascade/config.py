"""Strict layered configuration with canonical hashes."""

import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationError, field_validator
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode, Node

from silent_cascade.errors import ConfigurationError
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.validation import JsonValue, StrictModel


class RuntimeConfig(StrictModel):
    device_preference: tuple[Literal["mps", "cpu"], ...] = ("mps", "cpu")
    dtype: Literal["float32"] = "float32"
    primary_offline: Literal[True] = True
    allow_mps_fallback: Literal[False] = False
    primary_foundation_model_calls: Literal[0] = 0

    @field_validator("primary_offline", "allow_mps_fallback", mode="before")
    @classmethod
    def require_exact_boolean_type(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("value must be an exact bool")
        return value

    @field_validator("primary_foundation_model_calls", mode="before")
    @classmethod
    def require_exact_integer_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("value must be an exact int")
        return value

    @field_validator("device_preference", mode="before")
    @classmethod
    def tuple_from_yaml_list(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("device_preference")
    @classmethod
    def require_cpu_path(
        cls, value: tuple[Literal["mps", "cpu"], ...]
    ) -> tuple[Literal["mps", "cpu"], ...]:
        if "cpu" not in value:
            raise ValueError("device_preference must include the mandatory CPU path")
        if len(value) != len(set(value)):
            raise ValueError("device_preference may not contain duplicates")
        return value


class LimitsConfig(StrictModel):
    max_trainable_parameters: int = Field(default=5_000_000, ge=1, le=5_000_000)
    primary_memory_records: int = Field(default=64, ge=1, le=64)
    max_eventflow_events: int = Field(default=64, ge=1, le=64)
    max_fixed_grid_opportunities: int = Field(default=25_000, ge=1, le=25_000)
    batch_size: int = Field(default=128, ge=1, le=128)
    retained_full_traces: int = Field(default=500, ge=0, le=500)
    retain_all_failure_traces: Literal[True] = True
    foundation_model_workers: int = Field(default=1, ge=0, le=1)

    @field_validator("retain_all_failure_traces", mode="before")
    @classmethod
    def require_exact_boolean_type(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("value must be an exact bool")
        return value


class PathsConfig(StrictModel):
    runs: str = "runs"
    reports: str = "reports"
    manifests: str = "manifests"

    @field_validator("runs", "reports", "manifests")
    @classmethod
    def relative_repository_path(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("repository path must be relative and may not contain '..'")
        return value


class ProjectConfig(StrictModel):
    schema_version: Literal[1] = 1
    experiment_version: str = Field(default="v1", pattern=r"^[a-z0-9][a-z0-9._-]*$")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @field_validator("schema_version", mode="before")
    @classmethod
    def require_exact_integer_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("value must be an exact int")
        return value


@dataclass(frozen=True, slots=True)
class ResolvedConfig[TConfig: StrictModel]:
    config: TConfig
    canonical_json: bytes
    sha256: str
    source_paths: tuple[Path, ...]


_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


class _DuplicateKeyError(ConstructorError):
    """Internal loader error retaining the key that would be overwritten."""

    def __init__(self, key: object, node: MappingNode, key_node: Node) -> None:
        super().__init__(
            "while constructing a mapping",
            node.start_mark,
            "found duplicate key",
            key_node.start_mark,
        )
        self.key = key


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects duplicate keys in every mapping."""

    def construct_mapping(self, node: Node, deep: bool = False) -> dict[object, object]:
        if isinstance(node, MappingNode):
            self.flatten_mapping(node)
        if not isinstance(node, MappingNode):
            raise ConstructorError(
                None, None, f"expected a mapping node, but found {node.id}", node.start_mark
            )

        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as error:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark,
                ) from error
            if key in mapping:
                raise _DuplicateKeyError(key, node, key_node)
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _read_yaml_mapping(path: Path) -> dict[str, JsonValue]:
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictSafeLoader)
    except _DuplicateKeyError as error:
        raise ConfigurationError(
            "duplicate configuration key",
            context={"path": str(path), "key": str(error.key)},
        ) from error
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(
            "configuration load failed", context={"path": str(path), "reason": str(error)}
        ) from error
    if loaded is None:
        return {}
    if not isinstance(loaded, dict) or not all(isinstance(key, str) for key in loaded):
        raise ConfigurationError(
            "configuration root must be a string-keyed mapping", context={"path": str(path)}
        )
    return loaded


def _deep_merge(
    base: Mapping[str, JsonValue], overlay: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    merged = deepcopy(dict(base))
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def parse_set_override(expression: str) -> tuple[tuple[str, ...], JsonValue]:
    if "=" not in expression:
        raise ConfigurationError("invalid --set override", context={"expression": expression})
    dotted_key, raw_value = expression.split("=", 1)
    path = tuple(dotted_key.split("."))
    if not path or any(not segment or not _KEY.fullmatch(segment) for segment in path):
        raise ConfigurationError("invalid --set override", context={"expression": expression})
    try:
        value = yaml.load(raw_value, Loader=_StrictSafeLoader)
    except _DuplicateKeyError as error:
        raise ConfigurationError(
            "duplicate configuration key",
            context={"expression": expression, "key": str(error.key)},
        ) from error
    except yaml.YAMLError as error:
        raise ConfigurationError(
            "invalid --set override",
            context={"expression": expression, "reason": str(error)},
        ) from error
    return path, value


def _apply_override(target: dict[str, JsonValue], path: tuple[str, ...], value: JsonValue) -> None:
    cursor = target
    for segment in path[:-1]:
        child = cursor.setdefault(segment, {})
        if not isinstance(child, dict):
            raise ConfigurationError(
                "override traverses a non-mapping value", context={"path": ".".join(path)}
            )
        cursor = child
    cursor[path[-1]] = value


def resolve_config[TConfig: StrictModel](
    model_type: type[TConfig],
    yaml_paths: Sequence[Path],
    *,
    set_overrides: Sequence[str] = (),
) -> ResolvedConfig[TConfig]:
    merged: dict[str, JsonValue] = {}
    for path in yaml_paths:
        merged = _deep_merge(merged, _read_yaml_mapping(path))
    for expression in set_overrides:
        key_path, value = parse_set_override(expression)
        _apply_override(merged, key_path, value)
    try:
        config = model_type.model_validate(merged)
    except ValidationError as error:
        raise ConfigurationError(
            "configuration validation failed", context={"details": str(error)}
        ) from error
    canonical = canonical_json_bytes(config)
    return ResolvedConfig(
        config=config,
        canonical_json=canonical,
        sha256=sha256_bytes(canonical),
        source_paths=tuple(yaml_paths),
    )
