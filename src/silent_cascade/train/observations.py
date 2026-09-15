"""Public-only chronological observation flow shared by prediction and training."""

from dataclasses import dataclass, fields

import torch

from silent_cascade.memory.retrieval import RetrievalPreview
from silent_cascade.models.dynamics import BatchedSegmentParameters, flow_batch
from silent_cascade.models.types import ExternalFeatures, ModelContext


@dataclass(frozen=True, slots=True)
class Boundary:
    context: ModelContext
    parameters: BatchedSegmentParameters
    preview: RetrievalPreview
    mask: torch.Tensor

    @property
    def raw_guard_targets(self) -> torch.Tensor:
        return self.parameters.raw_guard_targets


def _parameters(parameters, rows, replacement=None):
    return BatchedSegmentParameters(
        **{
            item.name: getattr(parameters, item.name)[rows]
            if replacement is None
            else getattr(parameters, item.name).index_copy(0, rows, getattr(replacement, item.name))
            for item in fields(parameters)
        }
    )


def _external(prefix, rows):
    records = prefix.records
    slots = prefix.observation_slots[rows, 0].clamp_min(0)
    fact = prefix.observation_kind[rows, 0] == 0
    values = {}
    for source, target in (
        ("subject_ids", "subject_ids"),
        ("object_ids", "object_ids"),
        ("kind_ids", "record_kind_ids"),
        ("hazard_ids", "hazard_ids"),
        ("provenance_ids", "provenance_ids"),
    ):
        values[target] = torch.where(fact, getattr(records, source)[rows, slots], 0)
    return ExternalFeatures(
        **values,
        record_scalar_features=records.scalar_features[rows, slots] * fact[:, None],
        activation_entity_ids=prefix.activation_entities[rows, 0].clamp_min(0),
        event_kinds=prefix.observation_kind[rows, 0],
        time_features=prefix.observation_time_features[rows, 0],
    )


def observe_public(model, public):
    size = public.records.batch_size
    device = public.records.device
    context = model.initial_context(size, device=str(device))
    times = torch.tensor(public.initial_times, dtype=torch.float64)
    parameters = preview = None
    observations, before, after, masks = [], [], [], []
    for col in range(public.observation_mask.shape[1]):
        mask = public.observation_mask[:, col]
        rows = mask.nonzero(as_tuple=True)[0]
        if not rows.numel():
            continue
        prefix = public.at_observation(col)
        current = context._gather(rows)
        if parameters is not None:
            dt = (public.observation_times[rows.cpu(), col] - times[rows.cpu()]).to(
                device, dtype=torch.float32
            )
            current = current._updated(
                workspace=flow_batch(current.workspace, _parameters(parameters, rows), dt)
            )
        pre = context._scatter(rows, current).workspace.latent
        event = _external(prefix, rows)
        current = model.observe(current, event)
        fact_rows = (event.event_kinds == 0).nonzero(as_tuple=True)[0]
        if fact_rows.numel():
            fact_event = ExternalFeatures(
                **{item.name: getattr(event, item.name)[fact_rows] for item in fields(event)}
            )
            encoded = model.external_encoder._current_record_embedding(fact_event)
            slots = prefix.observation_slots[rows[fact_rows], 0]
            flat_indices = fact_rows * 64 + slots
            memory = current.memory_embeddings.flatten(0, 1).index_copy(0, flat_indices, encoded)
            current = current._updated(memory_embeddings=memory.reshape(-1, 64, 96))
        current = current._updated(eligibility=prefix.records.valid_mask[rows])
        active_preview, active_parameters = model.preview_and_control(current)
        context = context._scatter(rows, current)
        parameters = (
            active_parameters
            if parameters is None
            else _parameters(parameters, rows, active_parameters)
        )
        preview = (
            active_preview
            if preview is None
            else RetrievalPreview(
                preview.features.index_copy(0, rows, active_preview.features),
                preview.has_candidate.index_copy(0, rows, active_preview.has_candidate),
            )
        )
        times = times.index_copy(0, rows.cpu(), public.observation_times[rows.cpu(), col])
        observations.append(Boundary(context, parameters, preview, mask))
        before.append(pre)
        after.append(context.workspace.latent)
        masks.append(mask)

    return context, parameters, preview, tuple(observations), before, after, masks
