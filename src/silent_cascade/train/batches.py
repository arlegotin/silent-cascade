"""Bounded public observation staging and separate teacher-forced targets."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import torch

from silent_cascade.env.episode import PublicEpisode
from silent_cascade.memory.store import BoundedMemory, append_perceived_fact
from silent_cascade.memory.tensor_store import pack_memory
from silent_cascade.models.types import PublicInputBatch
from silent_cascade.schemas import ActivationPayload, InternalEventKind, Mode
from silent_cascade.train.config import Phase3Config
from silent_cascade.train.curriculum_data import (
    CurriculumExample,
    CurriculumKey,
    curriculum_rng,
    make_curriculum_example,
)
from silent_cascade.train.traces import TeacherTrace, build_teacher_trace

TARGET_NAMES = (
    "kind",
    "delta",
    "selected_slot",
    "focus",
    "role",
    "hazard_type",
    "log_delay",
    "normalized_deadline",
    "status",
    "confidence",
    "append_support",
    "continue_search",
    "action_class",
    "action_lead",
)
INTEGER_TARGETS = frozenset(
    ("kind", "selected_slot", "focus", "role", "hazard_type", "status", "action_class")
)


@dataclass(frozen=True, slots=True)
class TeacherTargets:
    kind: torch.Tensor
    delta: torch.Tensor
    selected_slot: torch.Tensor
    focus: torch.Tensor
    role: torch.Tensor
    hazard_type: torch.Tensor
    log_delay: torch.Tensor
    normalized_deadline: torch.Tensor
    status: torch.Tensor
    confidence: torch.Tensor
    append_support: torch.Tensor
    continue_search: torch.Tensor
    action_class: torch.Tensor
    action_lead: torch.Tensor
    validity: Mapping[str, torch.Tensor]
    real_step_mask: torch.Tensor
    final_dormancy_mask: torch.Tensor

    def __post_init__(self) -> None:
        shape = self.kind.shape
        if len(shape) != 2 or not 1 <= shape[0] <= 128 or not 1 <= shape[1] <= 11:
            raise ValueError("teacher targets must have bounded [B,T] shape")
        if set(self.validity) != set(TARGET_NAMES):
            raise ValueError("each teacher target requires its own validity mask")
        for name in TARGET_NAMES:
            value, mask = getattr(self, name), self.validity[name]
            dtype = torch.int64 if name in INTEGER_TARGETS else torch.float32
            if value.shape != shape or value.dtype != dtype or value.device != self.kind.device:
                raise ValueError(f"invalid {name} target")
            if mask.shape != shape or mask.dtype != torch.bool or mask.device != self.kind.device:
                raise ValueError(f"invalid {name} validity mask")
            if bool((mask & ~self.real_step_mask).any()):
                raise ValueError("padding must not contribute to target loss")
            if name in INTEGER_TARGETS and bool((value[~mask] != -1).any()):
                raise ValueError("invalid integer labels must use -1")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError("teacher floating labels must be finite")
            object.__setattr__(self, name, value.clone())
        for name in ("real_step_mask", "final_dormancy_mask"):
            value = getattr(self, name)
            if (
                value.shape != shape
                or value.dtype != torch.bool
                or value.device != self.kind.device
            ):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value.clone())
        if bool((self.final_dormancy_mask & ~self.real_step_mask).any()):
            raise ValueError("final dormancy must follow a real step, never padding")
        object.__setattr__(
            self,
            "validity",
            MappingProxyType({name: mask.clone() for name, mask in self.validity.items()}),
        )

    def to(self, device: str) -> "TeacherTargets":
        return TeacherTargets(
            **{name: getattr(self, name).to(device) for name in TARGET_NAMES},
            validity={name: mask.to(device) for name, mask in self.validity.items()},
            real_step_mask=self.real_step_mask.to(device),
            final_dormancy_mask=self.final_dormancy_mask.to(device),
        )


def pack_public_examples(examples: tuple[PublicEpisode, ...]) -> PublicInputBatch:
    if not isinstance(examples, tuple) or not 1 <= len(examples) <= 128:
        raise ValueError("public batch must contain 1 to 128 examples")
    memories = []
    for public in examples:
        memory = BoundedMemory(64)
        for event in public.events[:-1]:
            memory = append_perceived_fact(memory, event)
        memories.append(memory)
    initial_times = tuple(public.init.initial_time for public in examples)
    records = pack_memory(tuple(memories), initial_times, device="cpu")
    shape = (len(examples), max(len(public.events) for public in examples))
    kinds = torch.full(shape, -1, dtype=torch.int64)
    slots = torch.full_like(kinds, -1)
    entities = torch.full_like(kinds, -1)
    mask = torch.zeros(shape, dtype=torch.bool)
    times = torch.zeros(shape, dtype=torch.float64)
    time_features = torch.zeros((*shape, 2), dtype=torch.float32)
    for row, public in enumerate(examples):
        times[row].fill_(public.events[-1].timestamp)
        for col, event in enumerate(public.events):
            mask[row, col] = True
            times[row, col] = event.timestamp
            time_features[row, col, 0] = math.log1p(event.timestamp - public.init.initial_time)
            if isinstance(event.payload, ActivationPayload):
                kinds[row, col] = 1
                entities[row, col] = event.payload.start_node
            else:
                kinds[row, col] = 0
                slots[row, col] = records.record_ids[row].index(event.event_id)
    return PublicInputBatch(
        records, kinds, slots, entities, mask, time_features, times, initial_times
    )


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    public: PublicInputBatch
    targets: TeacherTargets
    teacher_times: torch.Tensor
    pre_modes: torch.Tensor
    post_modes: torch.Tensor
    active_slots_before: torch.Tensor
    active_slots_after: torch.Tensor
    support_before: torch.Tensor
    support_after: torch.Tensor
    examples: tuple[CurriculumExample, ...]
    teacher_traces: tuple[TeacherTrace, ...]
    example_keys: tuple[CurriculumKey, ...]
    example_hashes: tuple[str, ...]
    next_batch_counter: int

    def public_inputs(self) -> PublicInputBatch:
        return self.public

    @property
    def operation_masks(self) -> Mapping[str, torch.Tensor]:
        return MappingProxyType(
            {
                "observations": self.public.observation_mask,
                "real_steps": self.targets.real_step_mask,
                "retrieval": self.targets.validity["selected_slot"],
                "composition": self.targets.validity["role"],
                "action": self.targets.kind == 2,
                "abstention": self.targets.action_class == 4,
                "final_dormancy": self.targets.final_dormancy_mask,
            }
        )

    def to(self, device: str) -> "TrainingBatch":
        public = self.public.to(device)
        return replace(
            self,
            public=public,
            targets=self.targets.to(device),
            **{
                name: getattr(self, name).to(device)
                for name in (
                    "pre_modes",
                    "post_modes",
                    "active_slots_before",
                    "active_slots_after",
                    "support_before",
                    "support_after",
                )
            },
        )

    def permute_slots(self, permutations: torch.Tensor) -> "TrainingBatch":
        public = self.public.permute_slots(permutations)
        # Remap all record correspondence, including teacher active/support state.
        inverse = torch.argsort(permutations, dim=1)

        def remap(value):
            return torch.where(value >= 0, inverse.gather(1, value.clamp_min(0)), -1)

        targets = replace(self.targets, selected_slot=remap(self.targets.selected_slot))
        indices = permutations[:, None, :].expand_as(self.support_before)
        return replace(
            self,
            public=public,
            targets=targets,
            active_slots_before=remap(self.active_slots_before),
            active_slots_after=remap(self.active_slots_after),
            support_before=self.support_before.gather(2, indices),
            support_after=self.support_after.gather(2, indices),
        )


def pack_training_examples(
    examples: tuple[CurriculumExample, ...], *, next_batch_counter: int = 0
) -> TrainingBatch:
    public = pack_public_examples(tuple(e.public for e in examples))
    traces = tuple(build_teacher_trace(e) for e in examples)
    shape = (len(examples), max(len(trace.steps) for trace in traces))
    values = {
        name: torch.full(shape, -1, dtype=torch.int64)
        if name in INTEGER_TARGETS
        else torch.zeros(shape, dtype=torch.float32)
        for name in TARGET_NAMES
    }
    masks = {name: torch.zeros(shape, dtype=torch.bool) for name in TARGET_NAMES}
    real = torch.zeros(shape, dtype=torch.bool)
    dormancy = torch.zeros_like(real)
    times = torch.zeros(shape, dtype=torch.float64)
    pre_modes = torch.full(shape, -1, dtype=torch.int64)
    post_modes = torch.full_like(pre_modes, -1)
    active_before = torch.full_like(pre_modes, -1)
    active_after = torch.full_like(pre_modes, -1)
    support_before = torch.zeros((*shape, 64), dtype=torch.bool)
    support_after = torch.zeros_like(support_before)
    for row, trace in enumerate(traces):
        ids = public.records.record_ids[row]
        for col, step in enumerate(trace.steps):
            real[row, col] = True
            times[row, col] = step.timestamp
            pre_modes[row, col] = list(Mode).index(step.pre_mode)
            post_modes[row, col] = list(Mode).index(step.post_mode)
            if step.active_record_id_before is not None:
                active_before[row, col] = ids.index(step.active_record_id_before)
            if step.active_record_id_after is not None:
                active_after[row, col] = ids.index(step.active_record_id_after)
            for record_id in step.support_before:
                support_before[row, col, ids.index(record_id)] = True
            for record_id in step.support_after:
                support_after[row, col, ids.index(record_id)] = True
            labels = {
                name: getattr(step, name)
                for name in TARGET_NAMES
                if name not in ("kind", "selected_slot")
            }
            labels["kind"] = list(InternalEventKind).index(step.kind)
            labels["selected_slot"] = (
                ids.index(step.selected_record_id)
                if step.kind is InternalEventKind.RECALL
                else None
            )
            for name, value in labels.items():
                if value is not None:
                    values[name][row, col] = value
                    masks[name][row, col] = True
        dormancy[row, len(trace.steps) - 1] = True
    batch = TrainingBatch(
        public,
        TeacherTargets(**values, validity=masks, real_step_mask=real, final_dormancy_mask=dormancy),
        times,
        pre_modes,
        post_modes,
        active_before,
        active_after,
        support_before,
        support_after,
        examples,
        traces,
        tuple(e.key for e in examples),
        tuple(e.example_hash for e in examples),
        next_batch_counter,
    )
    # This independent presentation of memory is not an entity/relevance feature.
    permutations = torch.tensor(
        [
            curriculum_rng(e.key, "augmentation", e.accepted_attempt + MAX_SLOT_DOMAIN_OFFSET)
            .permutation(64)
            .tolist()
            for e in examples
        ],
        dtype=torch.int64,
    )
    return batch.permute_slots(permutations)


MAX_SLOT_DOMAIN_OFFSET = 1000


def next_training_batch(config: Phase3Config, *, stage: str, batch_counter: int) -> TrainingBatch:
    if type(batch_counter) is not int or batch_counter < 0:
        raise ValueError("batch_counter must be a nonnegative exact integer")
    training = config.training
    examples = tuple(
        make_curriculum_example(
            config,
            CurriculumKey(
                training.curriculum_version,
                "train",
                training.train_root_seed,
                training.train_public_id_seed,
                batch_counter * training.batch_size + offset,
                stage,
            ),
        )
        for offset in range(training.batch_size)
    )
    return pack_training_examples(examples, next_batch_counter=batch_counter + 1)
