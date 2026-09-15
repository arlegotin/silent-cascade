"""Bounded learned jumps for current external and endogenous events."""

import torch
from torch import nn
from torch.nn import functional as functional

from silent_cascade.memory.encoder import RecordEncoder
from silent_cascade.models.common import context_features
from silent_cascade.models.config import NeuralModelConfig
from silent_cascade.models.errors import NeuralError
from silent_cascade.models.types import ExternalFeatures, ModelContext, TensorWorkspace


def _require_batch_tensor(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    name: str,
    finite: bool = False,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise NeuralError(f"{name} must be a tensor")
    if value.shape != shape or value.dtype is not dtype or value.device != device:
        raise NeuralError(f"{name} must match the declared batch, dtype, and device")
    if finite and not bool(torch.isfinite(value).all()):
        raise NeuralError(f"{name} must contain only finite values")
    return value


def _bounded_jump(previous: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    targets, gate_logits = raw.chunk(2, dim=1)
    gates = torch.sigmoid(gate_logits)
    bounded_targets = torch.tanh(targets)
    from silent_cascade.eval.compute import _record_functional_operations

    _record_functional_operations(sigmoid_ops=gates.numel(), tanh_ops=bounded_targets.numel())
    return (1.0 - gates) * previous + gates * bounded_targets


class ExternalEncoder(nn.Module):
    """Encode only the current public FACT or ACTIVATE event for injection."""

    def __init__(self, config: NeuralModelConfig, record_encoder: RecordEncoder) -> None:
        super().__init__()
        if not isinstance(config, NeuralModelConfig):
            raise TypeError("config must be a NeuralModelConfig")
        if not isinstance(record_encoder, RecordEncoder):
            raise TypeError("record_encoder must be a RecordEncoder")
        self.config = config
        object.__setattr__(self, "_injected_record_encoder", record_encoder)
        self.network = nn.Sequential(
            nn.Linear(config.external_input_dim, config.external_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.external_hidden_dim, config.external_dim),
        )
        self.injection = nn.Linear(config.external_dim, config.external_injection_output_dim)

    @property
    def record_encoder(self) -> RecordEncoder:
        return object.__getattribute__(self, "_injected_record_encoder")

    def _current_record_embedding(self, event: ExternalFeatures) -> torch.Tensor:
        encoder = self.record_encoder
        object_present = event.record_scalar_features[:, :1]
        inputs = torch.cat(
            (
                encoder.encode_entities(event.subject_ids),
                encoder.encode_entities(event.object_ids) * object_present,
                encoder.kind_embedding(event.record_kind_ids),
                encoder.hazard_embedding(event.hazard_ids),
                encoder.provenance_embedding(event.provenance_ids),
                event.record_scalar_features,
            ),
            dim=1,
        )
        return encoder.network(inputs)

    def forward(self, event: ExternalFeatures) -> torch.Tensor:
        if not isinstance(event, ExternalFeatures):
            raise TypeError("event must be ExternalFeatures")
        encoder = self.record_encoder
        if (
            next(self.network.parameters()).device != event.device
            or encoder.entity_embedding.weight.device != event.device
        ):
            raise NeuralError("external features and encoders must use one device")
        fact = (event.event_kinds == 0).unsqueeze(1)
        activation = ~fact
        record_embedding = self._current_record_embedding(event) * fact
        activation_embedding = encoder.encode_entities(event.activation_entity_ids) * activation
        kind_indicators = functional.one_hot(event.event_kinds, num_classes=2).to(torch.float32)
        encoded = self.network(
            torch.cat(
                (record_embedding, activation_embedding, kind_indicators, event.time_features),
                dim=1,
            )
        )
        return self.injection(encoded)


def inject_external(state: TensorWorkspace, raw_injection: torch.Tensor) -> TensorWorkspace:
    """Apply a convex fast/slow-only external injection and reset all guards."""
    if not isinstance(state, TensorWorkspace):
        raise TypeError("state must be a TensorWorkspace")
    raw = _require_batch_tensor(
        raw_injection,
        shape=(state.batch_size, 640),
        dtype=torch.float32,
        device=state.device,
        name="raw external injection",
        finite=True,
    )
    updated_prefix = _bounded_jump(state.latent[:, :320], raw)
    latent = torch.cat((updated_prefix, state.latent[:, 320:]), dim=1)
    result = TensorWorkspace._from_functional_update(latent, torch.zeros_like(state.accumulators))
    from silent_cascade.eval.compute import _record_jump_application

    _record_jump_application(state.batch_size)
    return result


class SharedJump(nn.Module):
    """Apply one event-conditioned convex jump across every latent channel."""

    def __init__(
        self,
        config: NeuralModelConfig,
        *,
        mode_embedding: nn.Embedding | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, NeuralModelConfig):
            raise TypeError("config must be a NeuralModelConfig")
        self.config = config
        if mode_embedding is None:
            self._owned_mode_embedding = nn.Embedding(config.mode_count, config.mode_dim)
        else:
            if not isinstance(mode_embedding, nn.Embedding):
                raise TypeError("mode_embedding must be an nn.Embedding")
            if (
                mode_embedding.num_embeddings != config.mode_count
                or mode_embedding.embedding_dim != config.mode_dim
            ):
                raise NeuralError("mode embedding does not match the neural configuration")
            object.__setattr__(self, "_injected_mode_embedding", mode_embedding)
        self.event_embedding = nn.Embedding(3, config.kind_dim)
        self.network = nn.Sequential(
            nn.Linear(config.jump_input_dim, config.jump_hidden_dim),
            nn.LayerNorm(config.jump_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.jump_hidden_dim, config.jump_output_dim),
        )

    @property
    def mode_embedding(self) -> nn.Embedding:
        owned = self._modules.get("_owned_mode_embedding")
        if owned is not None:
            return owned  # type: ignore[return-value]
        return object.__getattribute__(self, "_injected_mode_embedding")

    def forward(self, context: ModelContext, event_kinds: torch.Tensor) -> TensorWorkspace:
        if not isinstance(context, ModelContext):
            raise TypeError("context must be a ModelContext")
        kinds = _require_batch_tensor(
            event_kinds,
            shape=(context.batch_size,),
            dtype=torch.int64,
            device=context.device,
            name="event kinds",
        )
        if bool(((kinds < 0) | (kinds >= 3)).any()):
            raise NeuralError("event kinds must encode RECALL, COMPOSE, or ACT")
        if next(self.network.parameters()).device != context.device:
            raise NeuralError("jump network and context must use one device")
        safe_indices = context.active_slot_indices.clamp_min(0)
        rows = torch.arange(context.batch_size, device=context.device)
        active = context.memory_embeddings[rows, safe_indices]
        active = active * (context.active_slot_indices >= 0).unsqueeze(1)
        raw = self.network(
            torch.cat(
                (
                    context_features(context, self.mode_embedding),
                    active,
                    self.event_embedding(kinds),
                ),
                dim=1,
            )
        )
        latent = _bounded_jump(context.workspace.latent, raw)
        result = TensorWorkspace._from_functional_update(
            latent, torch.zeros_like(context.workspace.accumulators)
        )
        from silent_cascade.eval.compute import _record_jump_application

        _record_jump_application(context.batch_size)
        return result
