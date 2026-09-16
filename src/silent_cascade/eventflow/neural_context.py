"""Validated single-row bridges from public runtime state to learned components."""

import math
from dataclasses import fields, replace

import torch

from silent_cascade.eventflow.state import (
    ContinuousChannels,
    ContinuousState,
    RuntimeState,
    SegmentParameters,
)
from silent_cascade.memory.tensor_store import pack_memory
from silent_cascade.models.dynamics import BatchedSegmentParameters
from silent_cascade.models.errors import NeuralError
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.models.types import ModelContext, TensorWorkspace
from silent_cascade.schemas import AgentInit, Mode


def workspace_from_continuous(state: ContinuousState) -> TensorWorkspace:
    if not isinstance(state, ContinuousState):
        raise TypeError("state must be ContinuousState")
    state = replace(state)  # Revalidate exposed tensor storage and retain autograd.
    return TensorWorkspace(
        torch.cat([getattr(state, f.name) for f in fields(ContinuousChannels)])[None],
        state.guard_accumulators[None],
    )


def continuous_from_workspace(workspace: TensorWorkspace) -> ContinuousState:
    if not isinstance(workspace, TensorWorkspace):
        raise TypeError("workspace must be TensorWorkspace")
    if workspace.batch_size != 1:
        raise NeuralError("runtime conversion requires exactly one row")
    return workspace.row(0)


def segment_from_batch(parameters: BatchedSegmentParameters) -> SegmentParameters:
    if not isinstance(parameters, BatchedSegmentParameters):
        raise TypeError("parameters must be BatchedSegmentParameters")
    if parameters.batch_size != 1:
        raise NeuralError("runtime segment requires exactly one row")
    parameters = replace(parameters)

    def channels(tensor):
        return ContinuousChannels(
            **dict(
                zip(
                    (field.name for field in fields(ContinuousChannels)),
                    tensor[0].split((256, 64, 8, 64, 64)),
                    strict=True,
                )
            )
        )

    return SegmentParameters(
        channels(parameters.flow_targets),
        channels(parameters.flow_rates),
        parameters.guard_targets[0],
        parameters.guard_rates[0],
    )


def context_from_runtime(
    model: EventFlowModel,
    state: RuntimeState,
    init: AgentInit,
) -> ModelContext:
    if not isinstance(model, EventFlowModel) or not isinstance(state, RuntimeState):
        raise TypeError("context requires EventFlowModel and RuntimeState")
    if not isinstance(init, AgentInit):
        raise TypeError("init must be AgentInit")
    core, now = state.core, state.time
    workspace = workspace_from_continuous(core.continuous)
    device = workspace.device
    if any(parameter.device != device for parameter in model.parameters()):
        raise NeuralError("model and runtime must use one device")
    elapsed = 0.0 if core.activation_time is None else now - core.activation_time
    if now < init.initial_time or elapsed < 0:
        raise NeuralError("runtime clocks must follow initialization and activation")
    records = pack_memory((core.memory,), (init.initial_time,), device=str(device))
    ids = records.record_ids[0]
    eligible = torch.zeros((1, 64), dtype=torch.bool, device=device)
    support = torch.zeros_like(eligible)
    for index, slot in enumerate(core.memory.records):
        eligible[0, index] = (
            slot.valid
            and slot.record.observed_at <= now
            and not slot.consumed
            and slot.refractory_until <= now
        )
    for record_id in core.support_ids:
        support[0, ids.index(record_id)] = True
    active = -1 if core.active_record_id is None else ids.index(core.active_record_id)
    hypothesis = [0.0] * 8
    if core.hypothesis is not None:
        prediction = core.hypothesis
        hypothesis[0] = 1.0
        hypothesis[5] = float(prediction.is_safe)
        hypothesis[6] = prediction.confidence
        if not prediction.is_safe:
            hypothesis[1 + prediction.hazard_type] = 1.0
            remaining = prediction.deadline - now
            hypothesis[7] = math.copysign(math.log1p(abs(remaining)), remaining)
    return ModelContext(
        workspace=workspace,
        memory_embeddings=model.record_encoder(records),
        eligibility=eligible,
        support_mask=support,
        active_slot_indices=torch.tensor([active], device=device),
        modes=torch.tensor([tuple(Mode).index(core.mode)], device=device),
        time_features=torch.tensor(
            [[math.log1p(now - init.initial_time), math.log1p(elapsed)]],
            dtype=torch.float32,
            device=device,
        ),
        hypothesis_features=torch.tensor([hypothesis], dtype=torch.float32, device=device),
    )
