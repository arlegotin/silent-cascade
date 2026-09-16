"""Core-owned logical weight identity, independent of training/archive parsers."""

import hashlib

import torch

from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.models.event_flow import EventFlowModel


def model_layout(model: EventFlowModel):
    """Canonical Phase 3 unique tensor names and state-dict alias bindings."""
    unique = {"parameter/" + k: v for k, v in model.named_parameters(remove_duplicate=True)}
    unique.update({"buffer/" + k: v for k, v in model.named_buffers(remove_duplicate=True)})
    identities = {id(value): key for key, value in unique.items()}
    aliases = {
        name: identities[id(value)] for name, value in model.state_dict(keep_vars=True).items()
    }
    return unique, aliases


def model_state_sha256(tensors: dict[str, torch.Tensor], aliases: dict[str, str]) -> str:
    """Hash the unchanged Phase 3 logical bytes (CPU contiguous snapshots)."""
    digest = hashlib.sha256(canonical_json_bytes(aliases))
    for name in sorted(k for k in tensors if k.startswith(("parameter/", "buffer/"))):
        value = tensors[name]
        digest.update(
            canonical_json_bytes(
                {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
            )
        )
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()
