"""Closed evidence data and bounded reads; no model, scoring or pass arithmetic."""

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from silent_cascade.errors import ArtifactIntegrityError
from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at
from silent_cascade.validation import StrictModel

MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_METADATA_BYTES = 8 * 1024 * 1024
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Count = Annotated[int, Field(strict=True, ge=0)]
Zero = Annotated[int, Field(strict=True, ge=0, le=0)]
Variant = Literal["positive", "safe", "disconnected"]


def decode_json(raw: bytes, limit: int = MAX_METADATA_BYTES) -> dict:
    """Reject ambiguous JSON, nonfinite constants and excessive nesting before schemas."""

    def pairs(items):
        value = {}
        for key, child in items:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = child
        return value

    def constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    try:
        if len(raw) > limit:
            raise ValueError("JSON byte limit exceeded")
        result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        if not isinstance(result, dict):
            raise ValueError("JSON envelope must be an object")
        pending = [(result, 0)]
        while pending:
            item, depth = pending.pop()
            if depth > 64:
                raise ValueError("JSON nesting limit exceeded")
            if isinstance(item, dict):
                pending.extend((child, depth + 1) for child in item.values())
            elif isinstance(item, list):
                pending.extend((child, depth + 1) for child in item)
            elif isinstance(item, float):
                import math

                if not math.isfinite(item):
                    raise ValueError("nonfinite JSON number")
        return result
    except (ValueError, TypeError, RecursionError, UnicodeError) as error:
        raise ArtifactIntegrityError(str(error)) from error


def read_bytes(path: Path, limit: int = MAX_METADATA_BYTES) -> bytes:
    with archive_parent(path, error_factory=ArtifactIntegrityError) as (parent, name):
        return read_archive_at(parent, name, max_bytes=limit, error_factory=ArtifactIntegrityError)


class ComparisonData(StrictModel):
    rtol: float = Field(ge=0)
    atol: float = Field(ge=0)
    compared: Count
    mismatches: Count
    max_absolute_error: float = Field(ge=0)
    max_relative_error: float = Field(ge=0)
    max_absolute_name: str
    max_relative_name: str
    tested_names: tuple[str, ...]
    failed_names: tuple[str, ...]
    tensor_elements: dict[str, Count]
    tensor_shapes: dict[str, tuple[Count, ...]]


class ParityData(StrictModel):
    schema_version: Literal["phase3-device-parity-v3"]
    objective_version: Literal["teacher_timed_v1", "teacher_timed_plus_content_v2"]
    auxiliary_coefficient: float
    optimizer_options: dict[str, object]
    devices: tuple[Literal["cpu"], Literal["mps"]]
    output_dtype: Literal["float32"]
    python_version: str
    torch_version: str
    threads: int = Field(ge=1)
    model_state_sha256: Hash
    example_hashes: tuple[Hash, ...]
    forward: ComparisonData
    losses: ComparisonData
    gradients: ComparisonData
    updated_weights: ComparisonData
    recalled_record_mismatches: Count
    action_class_mismatches: Count
    active_crossings: Count
    dormant_crossings: Count
    empty_memory_rows: Count
    foundation_model_calls: Zero


class ResumeData(StrictModel):
    schema_version: Literal["phase3-cpu-resume-v3", "phase3-mps-resume-v3"]
    objective_version: Literal["teacher_timed_v1", "teacher_timed_plus_content_v2"]
    auxiliary_coefficient: float
    optimizer_options: dict[str, object]
    device: Literal["cpu", "mps"]
    python_version: str
    torch_version: str
    threads: int = Field(ge=1)
    config_sha256: Hash
    source_commit: Revision
    checkpoint_sha256: Hash
    model_state_sha256: Hash
    resumed_step: Annotated[int, Field(strict=True, ge=2, le=2)]
    next_batch_counter: Annotated[int, Field(strict=True, ge=2, le=2)]
    example_hashes: tuple[Hash, ...]
    first_example_hashes: tuple[Hash, ...]
    parameters: ComparisonData
    optimizer: ComparisonData
    losses: ComparisonData
    rng_mismatches: Count
    batch_hash_mismatches: Count
    foundation_model_calls: Zero


class NumericEvidence(StrictModel):
    parity: ParityData
    cpu_resume: ResumeData
    mps_resume: ResumeData


class Phase3EvidenceProvenance(StrictModel):
    schema_version: Literal["phase3-evidence-provenance-v1"] = "phase3-evidence-provenance-v1"
    source_commit: Revision
    plan_revision: Revision
    evidence_base_revision: Revision
    source_paths: tuple[str, ...]
    source_tree_sha256: Hash
    specification_sha256: Hash
    approved_plan_sha256: Hash
    config_paths: tuple[str, ...]
    config_canonical_json: str
    config_sha256: Hash
    validation_manifest_path: str
    validation_manifest_commit: Revision
    validation_manifest_sha256: Hash
    generator_version: Literal["ofd-v1"] = "ofd-v1"
    transform_version: Literal["ofd-one-hop-v1"] = "ofd-one-hop-v1"
    model_seed: Annotated[int, Field(strict=True, ge=11, le=11)] = 11
    train_root_seed: Annotated[int, Field(strict=True, ge=311, le=311)] = 311
    train_public_id_seed: Annotated[int, Field(strict=True, ge=331, le=331)] = 331
    validation_root_seed: Annotated[int, Field(strict=True, ge=313, le=313)] = 313
    validation_public_id_seed: Annotated[int, Field(strict=True, ge=337, le=337)] = 337
    python_version: str = Field(pattern=r"^3\.12\.[0-9]+$")
    torch_version: str
    device: Literal["cpu"] = "cpu"
    threads: int = Field(ge=1)
    foundation_model_calls: Zero = 0


class ContentData(StrictModel):
    record_id: int
    role: int
    focus: int | None
    hazard_type: int | None
    log_delay: float | None
    normalized_deadline: float | None
    status: int | None
    confidence: float
    append_support: bool
    continue_search: bool | None


class TargetData(StrictModel):
    record_ids: tuple[int, ...] = Field(min_length=1, max_length=2)
    content: tuple[ContentData, ...] = Field(min_length=1, max_length=2)
    terminal_class: int
    terminal_status: int


class PredictionData(StrictModel):
    record_ids: tuple[int, ...] = Field(max_length=2)
    content: tuple[ContentData, ...] = Field(max_length=2)
    terminal_class: int
    terminal_status: int
    action_class: int
    stop_reason: Literal["capped", "malformed", "dormant", "explicit_null", "hazard", "safe"]
    atomic_transitions: Count
    observation_events: Count
    scorer_calls: Count
    records_scored: Count


class ComponentRow(StrictModel):
    public_id: str
    example_hash: Hash
    position: Count
    variant: Variant
    target: TargetData
    prediction: PredictionData
    required_recall_count: Count
    correct_recall_count: Count
    required_composition_count: Count
    correct_composition_count: Count
    complete_chain_correct: bool
    action_correct: bool
    capped: bool
    invalid_prediction: bool
    log_delay_absolute_errors: tuple[float, ...]
    normalized_deadline_absolute_errors: tuple[float, ...]
    compute_batch: Count
    model_state_sha256: Hash
    config_sha256: Hash
    source_commit: Revision
    foundation_model_calls: Zero = 0


class ComponentSummary(StrictModel):
    episode_count: Count
    required_recall_count: Count
    correct_recall_count: Count
    required_composition_count: Count
    correct_composition_count: Count
    complete_chain_correct: Count
    action_correct_count: Count
    variant_counts: dict[Variant, Count]
    cap_count: Count
    invalid_prediction_count: Count
    required_recall_accuracy: float
    required_composition_accuracy: float
    complete_chain_accuracy: float
    action_accuracy: float
    content_gates_pass: bool


class ComputeBatch(StrictModel):
    start: Count
    count: int = Field(ge=1, le=128)
    module_calls: dict[str, Count]
    row_transitions: Count
    records_scored: Count
    eligible_records: Count
    estimated_macs: Count
    forward_macs: Count
    backward_macs: Count
    operation_estimates: dict[str, Count]
    flow_evaluations: Count
    jump_applications: Count
    opportunities: Count
    parameters: Count
    memory_bytes: Count
    foundation_model_calls: Zero
    elapsed_seconds: float = Field(ge=0)
    mps_peak_allocation_bytes: Count | None


class DescriptorData(StrictModel):
    schema_version: Literal["phase3-checkpoint-descriptor-v1"]
    relative_path: str
    file_sha256: Hash
    model_state_sha256: Hash
    config_sha256: Hash
    source_commit: Revision
    optimizer_step: Count
    stage: Literal["smoke", "one_hop"]
    validation_metric: float | None
    validation_composition_metric: float | None


class StepDiagnostic(StrictModel):
    step: int = Field(strict=True, ge=1, le=2)
    example_hashes: tuple[Hash, ...]
    result: dict[str, object]


class TrainingEvidence(StrictModel):
    objective_version: Literal["teacher_timed_v1", "teacher_timed_plus_content_v2"]
    auxiliary_coefficient: float
    objective_step_counts: dict[str, Count]
    step_counters: tuple[Count, ...] = Field(max_length=75000)
    diagnostic_samples: tuple[StepDiagnostic, StepDiagnostic]
    run_path: str
    result_path: str
    result_sha256: Hash
    index_sha256: Hash
    journal_sha256: Hash
    latest: DescriptorData
    selected: DescriptorData
    weights: DescriptorData
    step_count: Count
    gradient_norm_max: float = Field(ge=0)
    next_batch_counter: Count
    patience_counter: Count
    stop_reason: Literal["component_gate", "early_stopping", "step_ceiling"]
    device: Literal["cpu", "mps"]
    validation_manifest_sha256: Hash
    validation_history: tuple["ValidationPoint", ...]
    foundation_model_calls: Zero = 0


class ValidationPoint(StrictModel):
    step: int = Field(strict=True, ge=1, le=75000)
    chain: float = Field(ge=0, le=1)
    composition: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    gates_pass: bool
    episode_count: Count
    complete_chain_correct: Count
    required_recall_count: Count
    correct_recall_count: Count
    required_composition_count: Count
    correct_composition_count: Count


class OfflineEvidence(StrictModel):
    schema_version: Literal["phase3-offline-import-probe-v2"]
    objective_version: Literal["teacher_timed_v1", "teacher_timed_plus_content_v2"]
    auxiliary_coefficient: float
    config_sha256: Hash
    training_steps: Count
    predicted_rows: Count
    backward_macs: Count
    blocked_import_attempts: Count
    network_attempts: Count
    forbidden_modules: tuple[str, ...]
    foundation_model_calls: Zero


class WeightEnvironmentData(StrictModel):
    python: str
    system: str
    machine: str
    torch: str
    numpy: str
    safetensors: str
    package: str
    device: Literal["cpu", "mps"]
    threads: int = Field(strict=True, ge=1, le=1024)
    deterministic_algorithms: bool


class WeightMetadataData(StrictModel):
    schema_version: Literal["phase3-model-weights-v1"]
    config_json: str
    config_sha256: Hash
    source_commit: Revision
    model_state_sha256: Hash
    aliases: dict[str, str]
    training: bool
    environment: WeightEnvironmentData
    owner_directory: str = Field(min_length=1, max_length=4096)
    progress: None
    groups: tuple[()]
    optimizer_names: tuple[()]
    rng: None


class ReplaySample(StrictModel):
    position: Count
    variant: Variant
    output_sha256: Hash


class ComponentGateReport(StrictModel):
    schema_version: Literal["phase3-component-gate-v3"] = "phase3-component-gate-v3"
    objective_version: Literal["teacher_timed_v1", "teacher_timed_plus_content_v2"]
    auxiliary_coefficient: float
    publication: Literal["debug", "production"]
    evidence_kind: Literal["neural component engineering evidence"] = (
        "neural component engineering evidence"
    )
    evaluation_mode: Literal["unassisted_content"] = "unassisted_content"
    timed: bool = False
    threshold_percent: Annotated[int, Field(strict=True, ge=99, le=99)] = 99
    provenance: Phase3EvidenceProvenance
    training: TrainingEvidence
    rows: tuple[ComponentRow, ...] = Field(min_length=16, max_length=10000)
    rows_sha256: Hash
    summary: ComponentSummary
    compute: tuple[ComputeBatch, ...]
    numeric: NumericEvidence
    replay_samples: tuple[ReplaySample, ReplaySample, ReplaySample]
    repeated_prediction_count: Count
    repeated_prediction_mismatches: Count
    offline: OfflineEvidence
    passed: bool
    foundation_model_calls: Zero = 0


class Phase3VerificationResult(StrictModel):
    schema_version: Literal["phase3-component-verification-v1"] = "phase3-component-verification-v1"
    valid: bool
    passed: bool
    publication: Literal["debug", "production"]
    episode_count: Count
    source_commit: Revision
    artifact_file_sha256: Hash
    checkpoint_sha256: Hash
    foundation_model_calls: Zero = 0
