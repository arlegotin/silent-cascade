"""Autonomous public-only learned policy over the scalar causal runtime."""

import math
import re
from dataclasses import dataclass, fields, replace
from functools import wraps

import torch

from silent_cascade.errors import DynamicsError
from silent_cascade.eval.compute import _record_functional_operations
from silent_cascade.eventflow.flow import start_segment
from silent_cascade.eventflow.guards import (
    allowed_mode_mask,
    next_crossings,
    prediction_snapshot_sha256,
)
from silent_cascade.eventflow.jumps import (
    ComposeDecision,
    ComposeRole,
    apply_act,
    apply_activate,
    apply_compose,
    apply_fact,
    apply_recall,
    begin_post_jump_segment,
)
from silent_cascade.eventflow.neural_context import (
    context_from_runtime,
    continuous_from_workspace,
    segment_from_batch,
)
from silent_cascade.eventflow.state import (
    ComputeCounters,
    ContinuousChannels,
    RuntimeCore,
    RuntimeState,
    SegmentParameters,
)
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.memory import BoundedMemory, append_perceived_fact
from silent_cascade.memory.retrieval import RetrievalPreview, RetrievalScores, select_record_ids
from silent_cascade.memory.tensor_store import pack_memory
from silent_cascade.models.config import NeuralModelConfig
from silent_cascade.models.errors import NeuralError
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.models.heads import ActionPredictions, ComposePredictions
from silent_cascade.models.transitions import (
    ComposeTransition,
    RecallTransition,
    apply_action_context,
    apply_compose_context,
    apply_recall_context,
)
from silent_cascade.models.types import ExternalFeatures, _validate_tensor
from silent_cascade.models.weights import model_layout, model_state_sha256
from silent_cascade.schemas import (
    AgentInit,
    Condition,
    ExternalEventKind,
    InternalEvent,
    InternalEventKind,
    Mode,
)


def _snapshot(model):
    unique, aliases = model_layout(model)
    device = next(model.parameters()).device
    if any(
        value.device != device
        or value.dtype not in {torch.float32, torch.bool, torch.int64}
        or (name.startswith("parameter/") and value.dtype != torch.float32)
        for name, value in unique.items()
    ):
        raise NeuralError("model requires one device, float32 parameters and typed buffers")
    tensors = {key: value.detach().cpu().contiguous().clone() for key, value in unique.items()}
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise NeuralError("nonfinite model state")
    return model_state_sha256(tensors, aliases)


def _shared_bindings(model):
    consumers = (
        model.scorer,
        model.controller,
        model.shared_jump,
        model.compose_heads,
        model.action_heads,
    )
    if (
        any(module.mode_embedding is not model.mode_embedding for module in consumers)
        or model.external_encoder.record_encoder is not model.record_encoder
    ):
        raise NeuralError("model alias identity mutated")


def _versions(model):
    # No state_dict, CPU copies or tensor serialization on the callback path.
    tensors = (
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    )
    return tuple(
        (
            name,
            id(value),
            value.data_ptr(),
            value._version,
            tuple(value.shape),
            value.dtype,
            value.device,
            tuple(value.stride()),
        )
        for name, value in tensors
    )


@dataclass(frozen=True, slots=True)
class NeuralModelIdentity:
    model_config_json: str
    model_state_sha256: str
    source_revision: str

    def __post_init__(self):
        try:
            config = NeuralModelConfig.model_validate_json(self.model_config_json)
            if canonical_json_bytes(config).decode() != self.model_config_json:
                raise ValueError("noncanonical model configuration")
            if not re.fullmatch(r"[0-9a-f]{64}", self.model_state_sha256):
                raise ValueError("invalid logical model digest")
            if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.source_revision):
                raise ValueError("invalid source revision")
        except (TypeError, ValueError) as error:
            raise NeuralError("invalid neural model identity") from error

    @classmethod
    def from_model(cls, model: EventFlowModel, *, source_revision: str):
        _shared_bindings(model)
        return cls(canonical_json_bytes(model.config).decode(), _snapshot(model), source_revision)


@dataclass(frozen=True, slots=True)
class NeuralDiagnostics:
    """Latest callback snapshot, copied to immutable host values only."""

    callback: str = "initialize"
    module_calls: tuple[tuple[str, int], ...] = ()
    scores: tuple[float, ...] = ()
    predictions: tuple[tuple[str, tuple[float, ...]], ...] = ()
    raw_action_argmax: int | None = None
    legal_action_argmax: int | None = None
    status_disagrees: bool = False
    record_rows_encoded: int = 0
    records_scored: int = 0


def _callback(function):
    @wraps(function)
    @torch.no_grad()
    def checked(self, *args, **kwargs):
        try:
            self._check_model()
            return function(self, *args, **kwargs)
        except NeuralError:
            raise
        except (
            DynamicsError,
            TypeError,
            ValueError,
            RuntimeError,
            AttributeError,
            IndexError,
            KeyError,
            OverflowError,
        ) as error:
            raise NeuralError(f"invalid neural {function.__name__} callback") from error

    return checked


class NeuralEventFlowAgent:
    """Frozen evaluator. Every causal episode value resides in RuntimeState."""

    name = Condition.EVENT_FLOW
    implementation_id = "neural-event-flow-v1"

    def __init__(self, model: EventFlowModel, *, identity: NeuralModelIdentity, device: str):
        if not isinstance(model, EventFlowModel) or not isinstance(identity, NeuralModelIdentity):
            raise NeuralError("agent requires EventFlowModel and NeuralModelIdentity")
        if device not in {"cpu", "mps"} or any(p.device.type != device for p in model.parameters()):
            raise NeuralError("model must already use the declared CPU/MPS device")
        _shared_bindings(model)
        if (
            _snapshot(model) != identity.model_state_sha256
            or canonical_json_bytes(model.config).decode() != identity.model_config_json
        ):
            raise NeuralError("model identity differs from frozen evaluator")
        model.eval()
        model.requires_grad_(False)
        self.model, self.identity, self.device = model, identity, device
        self._model_versions = _versions(model)
        self._init = None
        self._counters = ComputeCounters()
        self._diagnostics = NeuralDiagnostics()

    def _check_model(self):
        _shared_bindings(self.model)
        if (
            _versions(self.model) != self._model_versions
            or canonical_json_bytes(self.model.config).decode() != self.identity.model_config_json
            or any(module.training for module in self.model.modules())
        ):
            raise NeuralError("frozen evaluator model mutated")

    @_callback
    def initialize(self, init: AgentInit) -> RuntimeState:
        if not isinstance(init, AgentInit):
            raise NeuralError("initialize requires public AgentInit")
        self._init = init
        context = self.model.initial_context(1, device=self.device)
        core = RuntimeCore(
            continuous_from_workspace(context.workspace), memory=BoundedMemory(init.memory_capacity)
        )
        channels = {
            item.name: getattr(core.continuous, item.name) for item in fields(ContinuousChannels)
        }
        parameters = SegmentParameters(
            ContinuousChannels(**channels),
            ContinuousChannels(
                **{name: torch.ones_like(value) for name, value in channels.items()}
            ),
            torch.full_like(core.continuous.guard_accumulators, 0.5),
            torch.ones_like(core.continuous.guard_accumulators),
        )
        digest = prediction_snapshot_sha256(
            core.continuous,
            parameters,
            started_at=init.initial_time,
            allowed_mode_mask=allowed_mode_mask(core.mode),
            parent_event_id=0,
        )
        self._counters = core.counters
        self._diagnostics = NeuralDiagnostics()
        return start_segment(
            core,
            parameters,
            time=init.initial_time,
            parent_event_id=0,
            prediction_snapshot_sha256=digest,
        )

    def _context(self, state):
        return context_from_runtime(self.model, state, self._init)

    def _install(self, core, time, context, diagnostics):
        preview, batch = self.model.preview_and_control(context)
        if not isinstance(preview, RetrievalPreview):
            raise NeuralError("invalid preview schema")
        _validate_tensor(
            preview.features,
            shape=(1, 100),
            dtype=torch.float32,
            device=context.device,
            name="preview",
            finite=True,
        )
        _validate_tensor(
            preview.has_candidate,
            shape=(1,),
            dtype=torch.bool,
            device=context.device,
            name="preview candidate mask",
        )
        parameters = segment_from_batch(batch)
        # Actual scorer implementations evaluate all 64 rows, including padding.
        scored = 0 if core.mode is Mode.OBSERVING else 64
        core = replace(
            core,
            counters=replace(
                core.counters,
                controller_calls=core.counters.controller_calls + 1,
                records_scored=core.counters.records_scored + scored,
            ),
        )
        state = begin_post_jump_segment(core, parameters, time=time)
        self._counters = state.core.counters
        self._diagnostics = diagnostics
        return state

    @_callback
    def on_external(self, state, event):
        if event.kind not in {ExternalEventKind.FACT, ExternalEventKind.ACTIVATE}:
            raise NeuralError("only delivered public FACT/ACTIVATE events are accepted")
        context = self._context(state)
        fact = event.kind is ExternalEventKind.FACT
        memory = append_perceived_fact(state.core.memory, event) if fact else state.core.memory
        records = pack_memory((memory,), (self._init.initial_time,), device=self.device)
        slot = len(memory.records) - 1 if fact else 0
        zero = torch.zeros(1, dtype=torch.int64, device=self.device)
        external = ExternalFeatures(
            **{
                target: getattr(records, source)[:, slot] if fact else zero
                for source, target in (
                    ("subject_ids", "subject_ids"),
                    ("object_ids", "object_ids"),
                    ("kind_ids", "record_kind_ids"),
                    ("hazard_ids", "hazard_ids"),
                    ("provenance_ids", "provenance_ids"),
                )
            },
            record_scalar_features=records.scalar_features[:, slot]
            if fact
            else torch.zeros((1, 6), device=self.device),
            activation_entity_ids=zero if fact else zero + event.payload.start_node,
            event_kinds=zero if fact else zero + 1,
            time_features=torch.tensor(
                [[math.log1p(state.time - self._init.initial_time), 0.0]],
                dtype=torch.float32,
                device=self.device,
            ),
        )
        context = self.model.observe(context, external)
        if fact:
            encoded = self.model.external_encoder._current_record_embedding(external)
            embeddings = context.memory_embeddings.clone()
            embeddings[:, slot] = encoded
            context = context._updated(memory_embeddings=embeddings)
        context = context._updated(eligibility=records.valid_mask.clone())
        continuous = continuous_from_workspace(context.workspace)
        core = (apply_fact if fact else apply_activate)(state, event, continuous=continuous)
        diagnostics = NeuralDiagnostics(
            event.kind.value,
            (
                ("observe", 1),
                ("record_encoder", 1),
                ("current_record_encoder", 2 if fact else 1),
                ("controller", 1),
                ("scorer", 0 if fact else 1),
            ),
            record_rows_encoded=66 if fact else 65,
            records_scored=0 if fact else 64,
        )
        return self._install(core, state.time, context, diagnostics)

    def next_internal_event(self, state):
        # Analytic query only: no forward, mutation check, jump, or latent mutation.
        candidates = next_crossings(state)
        if not candidates:
            return None
        crossing = candidates[0]
        return InternalEvent(
            crossing.event_id,
            crossing.parent_event_id,
            crossing.timestamp,
            crossing.kind,
            crossing.guard_index,
            crossing.predicted_delta,
        )

    def _predictions(self, predictions, expected, context):
        if not isinstance(predictions, expected):
            raise NeuralError("invalid prediction schema")
        widths = {
            "role_logits": 5,
            "next_focus_logits": 64,
            "hazard_logits": 4,
            "status_logits": 3,
            "class_logits": 5,
        }
        result = []
        for item in fields(predictions):
            value = getattr(predictions, item.name)
            shape = (1, widths[item.name]) if item.name in widths else (1,)
            _validate_tensor(
                value,
                shape=shape,
                dtype=torch.float32,
                device=context.device,
                name=item.name,
                finite=True,
            )
            result.append((item.name, tuple(value.detach().cpu().flatten().tolist())))
        return tuple(result)

    @_callback
    def on_internal(self, state, event):
        context = self._context(state)
        scores = ()
        predictions = ()
        raw = legal = None
        disagreement = False
        if event.kind is InternalEventKind.RECALL:
            output = self.model.recall_scores(context)
            if not isinstance(output, RetrievalScores):
                raise NeuralError("invalid recall schema")
            _validate_tensor(
                output.raw_scores,
                shape=(1, 64),
                dtype=torch.float32,
                device=context.device,
                name="recall scores",
                finite=True,
            )
            _validate_tensor(
                output.eligible_mask,
                shape=(1, 64),
                dtype=torch.bool,
                device=context.device,
                name="recall eligibility",
            )
            if not torch.equal(output.eligible_mask, context.eligibility):
                raise NeuralError("recall eligibility must reflect public structural legality")
            _validate_tensor(
                output.masked_logits,
                shape=(1, 64),
                dtype=torch.float32,
                device=context.device,
                name="recall logits",
            )
            if not torch.equal(
                output.masked_logits,
                output.raw_scores.masked_fill(~context.eligibility, -torch.inf),
            ):
                raise NeuralError("recall logits must mask only structurally ineligible slots")
            ids = tuple(slot.record.record_id for slot in state.core.memory.records)
            record_id = select_record_ids(output, (ids + (None,) * (64 - len(ids)),))[0]
            if record_id is None:
                raise NeuralError("empty learned recall")
            selected = ids.index(record_id)
            context = apply_recall_context(
                self.model,
                context,
                RecallTransition(
                    torch.tensor([selected], device=self.device),
                    torch.tensor([tuple(Mode).index(Mode.HAVE_MEMORY)], device=self.device),
                ),
            )
            core = apply_recall(
                state,
                event,
                record_id=record_id,
                selected_rank=1,
                continuous=continuous_from_workspace(context.workspace),
            )
            core = replace(
                core,
                counters=replace(core.counters, records_scored=core.counters.records_scored + 64),
            )
            scores = tuple(output.raw_scores[0].cpu().tolist())
        elif event.kind is InternalEventKind.COMPOSE:
            output = self.model.compose(context)
            predictions = self._predictions(output, ComposePredictions, context)
            role = tuple(ComposeRole)[int(output.role_logits[0].argmax().item())]
            confidence = float(torch.sigmoid(output.confidence_logit[0]).item())
            _record_functional_operations(sigmoid_ops=1)
            decision = ComposeDecision(
                role,
                next_focus_node_id=int(output.next_focus_logits[0].argmax().item())
                if role is ComposeRole.LINK
                else None,
                hazard_type=int(output.hazard_logits[0].argmax().item())
                if role is ComposeRole.HAZARD
                else None,
                deadline=float(
                    state.core.activation_time
                    + output.normalized_deadline[0].item()
                    * (1.0 + state.time - state.core.activation_time)
                )
                if role is ComposeRole.HAZARD
                else None,
                confidence=confidence,
                append_support=bool(output.append_support_logit[0].item() >= 0),
                continue_search=bool(output.continue_search_logit[0].item() >= 0),
            )
            # Typed metadata is authoritative; supplying the existing continuous state
            # bypasses scripted impulses while assembling the learned jump registers.
            core = apply_compose(state, event, decision, continuous=state.core.continuous)
            supports = context.support_mask.clone()
            if decision.append_support:
                supports[0, int(context.active_slot_indices[0].item())] = True
            hypothesis = context.hypothesis_features.clone()
            if role in {ComposeRole.HAZARD, ComposeRole.SAFE}:
                hypothesis.zero_()
                hypothesis[0, 0] = 1
                hypothesis[0, 5] = float(role is ComposeRole.SAFE)
                hypothesis[0, 6] = confidence
                if role is ComposeRole.HAZARD:
                    hypothesis[0, 1 + decision.hazard_type] = 1
                    remaining = decision.deadline - state.time
                    hypothesis[0, 7] = math.copysign(math.log1p(abs(remaining)), remaining)
            context = apply_compose_context(
                self.model,
                context,
                ComposeTransition(
                    supports,
                    torch.tensor([tuple(Mode).index(core.mode)], device=self.device),
                    hypothesis,
                    torch.tensor([decision.next_focus_node_id or 0], device=self.device),
                    torch.tensor([role is ComposeRole.LINK], device=self.device),
                ),
            )
            core = replace(core, continuous=continuous_from_workspace(context.workspace))
            expected_status = (
                0 if role is ComposeRole.HAZARD else 1 if role is ComposeRole.SAFE else 2
            )
            disagreement = int(output.status_logits[0].argmax().item()) != expected_status
        elif event.kind is InternalEventKind.ACT:
            if state.core.mode is not Mode.HOLDING_HAZARD:
                raise NeuralError("ACT requires HOLDING_HAZARD")
            output = self.model.action(context)
            predictions = self._predictions(output, ActionPredictions, context)
            raw = int(output.class_logits[0].argmax().item())
            legal = int(output.class_logits[0, :4].argmax().item())
            context = apply_action_context(self.model, context)
            core = apply_act(
                state,
                event,
                hazard_type=legal,
                continuous=continuous_from_workspace(context.workspace),
            )
        else:
            raise NeuralError("unsupported internal event")
        # Consumption, refractory eligibility and hypothesis time features come
        # from typed runtime metadata; never retain aliased decision tensors.
        context = self._context(replace(state, core=core))
        diagnostics = NeuralDiagnostics(
            event.kind.value,
            (
                ("record_encoder", 2),
                ("jump", 1),
                ("controller", 1),
                ("scorer", 2 if event.kind is InternalEventKind.RECALL else 1),
                (event.kind.value, 1),
            ),
            scores,
            predictions,
            raw,
            legal,
            disagreement,
            record_rows_encoded=128,
            records_scored=128 if event.kind is InternalEventKind.RECALL else 64,
        )
        result = self._install(core, state.time, context, diagnostics)
        return result, list(core.actions[len(state.core.actions) :])

    def compute_counters(self):
        return self._counters

    def diagnostic_snapshot(self):
        return self._diagnostics
