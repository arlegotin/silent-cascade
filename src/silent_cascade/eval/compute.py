"""Non-invasive measured compute accounting for neural component execution."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType

import torch
from pydantic import Field
from torch import nn

from silent_cascade.eventflow.state import ComputeCounters
from silent_cascade.models.errors import NeuralError
from silent_cascade.validation import StrictModel

_ACTIVE_METERS: ContextVar[tuple[NeuralComputeMeter, ...]] = ContextVar(
    "silent_cascade_neural_compute_meters", default=()
)


@dataclass(frozen=True, slots=True)
class NeuralComputeSnapshot:
    """Immutable counters captured from one completed measurement scope."""

    module_calls: Mapping[str, int]
    row_transitions: int
    records_scored: int
    eligible_records: int
    estimated_macs: int
    forward_macs: int
    backward_macs: int
    operation_estimates: Mapping[str, int]
    flow_evaluations: int
    jump_applications: int
    opportunities: int
    parameters: int
    memory_bytes: int
    foundation_model_calls: int
    elapsed_seconds: float
    mps_peak_allocation_bytes: int | None


@dataclass(frozen=True, slots=True)
class _VersionedTensorReference:
    tensor: torch.Tensor
    version: int
    name: str


def _linear_macs(module: nn.Linear, inputs: tuple[torch.Tensor, ...]) -> int:
    values = inputs[0]
    rows = values.numel() // module.in_features
    return rows * module.in_features * module.out_features


def _tensor_elements(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, (tuple, list)):
        return sum(_tensor_elements(item) for item in value)
    return 0


def _record_flow_evaluation(batch_size: int, value_count: int) -> None:
    """Record executed functional flow work for every active meter."""
    for meter in _ACTIVE_METERS.get():
        meter._flow_evaluations += batch_size
        meter._row_transitions += batch_size
        meter._operation_estimates["exp_log_ops"] += value_count


def _record_jump_application(batch_size: int) -> None:
    """Record rows whose workspace was successfully changed by a jump."""
    for meter in _ACTIVE_METERS.get():
        meter._jump_applications += batch_size
        meter._row_transitions += batch_size


def _record_functional_operations(**operations: int) -> None:
    """Record already-executed functional work using integer metadata only."""
    for meter in _ACTIVE_METERS.get():
        meter._operation_estimates.update(operations)


def _record_ponder_retrieval(valid_mask: torch.Tensor, query_dim: int) -> None:
    """Account for executed record and null dot products after key projection."""
    if valid_mask.ndim != 2 or valid_mask.dtype is not torch.bool or query_dim <= 0:
        raise ValueError("ponder retrieval requires a [B,S] bool mask and positive width")
    rows, slots = valid_mask.shape
    macs = rows * (slots + 1) * query_dim
    for meter in _ACTIVE_METERS.get():
        meter._forward_macs += macs
        meter._operation_estimates["bilinear_dot_macs"] += macs
        meter._records_scored += int(valid_mask.sum())
        meter._eligibility_references.append(
            meter._capture_versioned_tensor(valid_mask, "ponder valid memory")
        )


class NeuralComputeMeter:
    """Observe module and functional neural work without changing execution."""

    def __init__(self, module: nn.Module) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("module must be an nn.Module")
        self.module = module
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._tensor_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._token: Token[tuple[NeuralComputeMeter, ...]] | None = None
        self._entered = False
        self._finished = False
        self._snapshot_valid = False
        self._invalid_reason: str | None = None
        self._module_calls: Counter[str] = Counter()
        self._operation_estimates: Counter[str] = Counter()
        self._row_transitions = 0
        self._records_scored = 0
        self._eligibility_references: list[_VersionedTensorReference] = []
        self._eligible_records = 0
        self._forward_macs = 0
        self._backward_macs = 0
        self._flow_evaluations = 0
        self._jump_applications = 0
        self._mode_references: list[_VersionedTensorReference] = []
        self._opportunities = 0
        self._memory_bytes = 0
        self._elapsed_seconds = 0.0
        self._started_at = 0.0
        self._uses_mps = any(
            value.device.type == "mps"
            for value in (*tuple(module.parameters()), *tuple(module.buffers()))
        )

    def __enter__(self) -> NeuralComputeMeter:
        if self._entered:
            raise NeuralError("a compute meter cannot be entered more than once")
        self._entered = True
        try:
            if self._uses_mps:
                torch.mps.synchronize()
            for submodule in self._observed_modules():
                self._handles.append(submodule.register_forward_hook(self._forward_hook))
            current = _ACTIVE_METERS.get()
            self._token = _ACTIVE_METERS.set((*current, self))
            self._started_at = perf_counter()
            return self
        except BaseException:
            self._invalid_reason = "compute meter entry did not complete"
            self._cleanup()
            raise

    def _observed_modules(self) -> tuple[nn.Module, ...]:
        """Include explicit non-owning shared dependencies once by identity."""
        observed: list[nn.Module] = []
        seen: set[int] = set()

        def append_tree(root: nn.Module) -> None:
            for candidate in root.modules():
                if id(candidate) not in seen:
                    seen.add(id(candidate))
                    observed.append(candidate)

        append_tree(self.module)
        for candidate in tuple(observed):
            if type(candidate).__name__ == "ExternalEncoder":
                append_tree(candidate.record_encoder)  # type: ignore[union-attr]
            if type(candidate).__name__ in {
                "RetrievalScorer",
                "FlowGuardController",
                "SharedJump",
                "ComposeHeads",
                "ActionHeads",
            }:
                append_tree(candidate.mode_embedding)  # type: ignore[union-attr]
        return tuple(observed)

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        try:
            if self._uses_mps:
                torch.mps.synchronize()
            self._elapsed_seconds = perf_counter() - self._started_at
            if self._invalid_reason is None:
                self._freeze_public_counts()
                self._snapshot_valid = True
            elif exc_type is None:
                raise NeuralError(self._invalid_reason)
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Release every observer resource, including after partial entry."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for handle in self._tensor_handles:
            handle.remove()
        self._tensor_handles.clear()
        self._eligibility_references.clear()
        self._mode_references.clear()
        if self._token is not None:
            _ACTIVE_METERS.reset(self._token)
            self._token = None
        self._finished = True

    def _capture_versioned_tensor(
        self, tensor: torch.Tensor, name: str
    ) -> _VersionedTensorReference:
        """Capture host mutation metadata without executing a tensor operation."""
        try:
            version = tensor._version
        except RuntimeError as error:
            self._invalid_reason = f"{name} tensor mutation version is unavailable"
            raise NeuralError(self._invalid_reason) from error
        return _VersionedTensorReference(tensor=tensor, version=version, name=name)

    def _validate_version(self, reference: _VersionedTensorReference) -> None:
        try:
            current = reference.tensor._version
        except RuntimeError as error:
            self._invalid_reason = f"{reference.name} tensor mutation version is unavailable"
            raise NeuralError(self._invalid_reason) from error
        if current != reference.version:
            self._invalid_reason = f"{reference.name} tensor was modified in place"
            raise NeuralError(self._invalid_reason)

    def _freeze_public_counts(self) -> None:
        """Copy observed public masks to CPU and reduce after measured timing."""
        references = (*self._eligibility_references, *self._mode_references)
        for reference in references:
            self._validate_version(reference)
        eligible_records = sum(
            int(reference.tensor.detach().to(device="cpu", copy=True).sum().item())
            for reference in self._eligibility_references
        )
        opportunities = sum(
            int((reference.tensor.detach().to(device="cpu", copy=True) != 0).sum().item())
            for reference in self._mode_references
        )
        self._eligible_records = eligible_records
        self._opportunities = opportunities

    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[object, ...],
        output: object,
    ) -> None:
        name = type(module).__name__
        self._module_calls[name] += 1
        if isinstance(module, nn.Linear):
            tensor_inputs = inputs  # PyTorch supplied the tensor contract for Linear.
            macs = _linear_macs(module, tensor_inputs)  # type: ignore[arg-type]
            self._forward_macs += macs
            rows = tensor_inputs[0].numel() // module.in_features  # type: ignore[union-attr]
            if module.bias is not None:
                self._operation_estimates["linear_bias_adds"] += rows * module.out_features
            if isinstance(output, torch.Tensor) and output.requires_grad:
                self._tensor_handles.append(output.register_hook(self._backward_counter(macs)))
        elif isinstance(module, nn.GRUCell):
            values = inputs[0]
            assert isinstance(values, torch.Tensor)
            rows = values.numel() // module.input_size
            macs = 3 * rows * module.hidden_size * (module.input_size + module.hidden_size)
            self._forward_macs += macs
            self._jump_applications += rows
            self._row_transitions += rows
            self._operation_estimates["gru_bias_adds"] += 6 * rows * module.hidden_size
            self._operation_estimates["gru_gate_ops"] += 8 * rows * module.hidden_size
            if isinstance(output, torch.Tensor) and output.requires_grad:
                self._tensor_handles.append(output.register_hook(self._backward_counter(macs)))
        elif isinstance(module, nn.SiLU):
            self._operation_estimates["silu_ops"] += 4 * _tensor_elements(output)
        elif isinstance(module, nn.LayerNorm):
            self._operation_estimates["layer_norm_ops"] += 5 * _tensor_elements(output)
        elif isinstance(module, nn.Embedding) and isinstance(output, torch.Tensor):
            self._operation_estimates["embedding_output_bytes"] += (
                output.numel() * output.element_size()
            )

        if name == "RecordEncoder":
            self._operation_estimates["record_rows_encoded"] += inputs[0].subject_ids.numel()
        elif name == "ExternalEncoder":
            # Its direct record-network call also encodes the current event row,
            # even on ACTIVATE where the result is subsequently masked away.
            self._operation_estimates["record_rows_encoded"] += inputs[0].batch_size
        elif name == "RetrievalScorer":
            context = inputs[0]
            batch_size = context.batch_size  # type: ignore[union-attr]
            slots = context.memory_embeddings.shape[1]  # type: ignore[union-attr]
            query_dim = module.config.query_dim  # type: ignore[union-attr]
            bilinear_macs = batch_size * slots * query_dim
            self._forward_macs += bilinear_macs
            self._operation_estimates["bilinear_dot_macs"] += bilinear_macs
            self._records_scored += batch_size * slots
            self._eligibility_references.append(  # type: ignore[union-attr]
                self._capture_versioned_tensor(context.eligibility, "eligibility")
            )
            self._memory_bytes = max(
                self._memory_bytes,
                context.memory_embeddings.numel()  # type: ignore[union-attr]
                * context.memory_embeddings.element_size(),  # type: ignore[union-attr]
            )
            self._row_transitions += batch_size
        elif name == "FlowGuardController":
            context = inputs[0]
            batch_size = context.batch_size  # type: ignore[union-attr]
            self._mode_references.append(  # type: ignore[union-attr]
                self._capture_versioned_tensor(context.modes, "modes")
            )
            self._row_transitions += batch_size
            self._operation_estimates["tanh_ops"] += batch_size * 456
            self._operation_estimates["sigmoid_ops"] += batch_size * 3
            self._operation_estimates["exp_log_ops"] += batch_size * (456 + 3)
        elif name == "ComposeHeads":
            batch_size = inputs[0].batch_size  # type: ignore[union-attr]
            self._row_transitions += batch_size
            self._operation_estimates["exp_log_ops"] += batch_size
        elif name == "ActionHeads":
            batch_size = inputs[0].batch_size  # type: ignore[union-attr]
            self._row_transitions += batch_size
            self._operation_estimates["sigmoid_ops"] += batch_size

    def _backward_counter(self, forward_macs: int):
        def record(gradient: torch.Tensor) -> torch.Tensor:
            if not self._finished:
                self._backward_macs += 2 * forward_macs
            return gradient

        return record

    def snapshot(self) -> NeuralComputeSnapshot:
        """Return counters frozen independently from future internal mutation."""
        if not self._snapshot_valid:
            raise NeuralError("compute meter has no valid snapshot")
        return NeuralComputeSnapshot(
            module_calls=MappingProxyType(dict(self._module_calls)),
            row_transitions=self._row_transitions,
            records_scored=self._records_scored,
            eligible_records=self._eligible_records,
            estimated_macs=self._forward_macs,
            forward_macs=self._forward_macs,
            backward_macs=self._backward_macs,
            operation_estimates=MappingProxyType(dict(self._operation_estimates)),
            flow_evaluations=self._flow_evaluations,
            jump_applications=self._jump_applications,
            opportunities=self._opportunities,
            parameters=sum(parameter.numel() for parameter in self.module.parameters()),
            memory_bytes=self._memory_bytes,
            foundation_model_calls=0,
            elapsed_seconds=self._elapsed_seconds,
            mps_peak_allocation_bytes=None,
        )


def parameter_counts(model: nn.Module) -> dict[str, int]:
    """Count unique trainable parameters and the shared entity table exactly once."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    unique: dict[int, nn.Parameter] = {}
    entity_ids: set[int] = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        unique[id(parameter)] = parameter
        if name.endswith("entity_embedding.weight"):
            entity_ids.add(id(parameter))
    total = sum(parameter.numel() for parameter in unique.values())
    entity_table = sum(unique[parameter_id].numel() for parameter_id in entity_ids)
    return {
        "total": total,
        "entity_table": entity_table,
        "non_entity": total - entity_table,
    }


class RuntimeCompute(StrictModel):
    """Disjoint engine and neural observations; EventFlow has no opportunity ticks."""

    engine: ComputeCounters = Field(default_factory=ComputeCounters)
    module_calls: dict[str, int] = Field(default_factory=dict)
    operation_estimates: dict[str, int] = Field(default_factory=dict)
    parameters: int = Field(default=0, ge=0)
    entity_parameters: int = Field(default=0, ge=0)
    forward_macs: int = Field(default=0, ge=0)
    backward_macs: int = Field(default=0, ge=0)
    records_scored: int = Field(default=0, ge=0)
    record_rows_encoded: int = Field(default=0, ge=0)
    eligible_records: int = Field(default=0, ge=0)
    neural_flow_evaluations: int = Field(default=0, ge=0)
    neural_jump_applications: int = Field(default=0, ge=0)
    neural_active_controller_rows: int = Field(default=0, ge=0)
    flow_evaluations: int = Field(default=0, ge=0)
    uncommitted_flow_evaluations: int = Field(default=0, ge=0)
    jump_applications: int = Field(default=0, ge=0)
    opportunities: int = Field(default=0, ge=0, le=0)
    foundation_model_calls: int = Field(default=0, ge=0, le=0)
    memory_bytes: int = Field(default=0, ge=0)
    pause_count: int = Field(default=0, ge=0)
    checkpoint_serializations: int = Field(default=0, ge=0)
    replay_executions: int = Field(default=0, ge=0)


def aggregate_runtime_compute(
    engine: ComputeCounters,
    snapshots: tuple[NeuralComputeSnapshot, ...],
    *,
    entity_parameters: int,
    checkpoint_serializations: int = 0,
    causal_flow_evaluations: int | None = None,
) -> RuntimeCompute:
    """Sum disjoint callback scopes, counting scalar analytic flow only in engine."""
    modules, operations = Counter(), Counter()
    for snapshot in snapshots:
        modules.update(snapshot.module_calls)
        operations.update(snapshot.operation_estimates)
    if engine.foundation_model_calls or any(s.foundation_model_calls for s in snapshots):
        raise NeuralError("runtime must have zero foundation-model calls")
    performed_flow = (
        engine.flow_evaluations if causal_flow_evaluations is None else causal_flow_evaluations
    )
    if performed_flow < engine.flow_evaluations:
        raise NeuralError("performed flow cannot be below committed flow")
    return RuntimeCompute(
        engine=engine,
        module_calls=dict(modules),
        operation_estimates=dict(operations),
        parameters=max((s.parameters for s in snapshots), default=0),
        entity_parameters=entity_parameters,
        forward_macs=sum(s.forward_macs for s in snapshots),
        backward_macs=sum(s.backward_macs for s in snapshots),
        records_scored=sum(s.records_scored for s in snapshots),
        record_rows_encoded=operations["record_rows_encoded"],
        eligible_records=sum(s.eligible_records for s in snapshots),
        neural_flow_evaluations=sum(s.flow_evaluations for s in snapshots),
        neural_jump_applications=sum(s.jump_applications for s in snapshots),
        neural_active_controller_rows=sum(s.opportunities for s in snapshots),
        flow_evaluations=performed_flow,
        uncommitted_flow_evaluations=performed_flow - engine.flow_evaluations,
        jump_applications=engine.jump_applications,
        memory_bytes=max((s.memory_bytes for s in snapshots), default=0),
        checkpoint_serializations=checkpoint_serializations,
    )
