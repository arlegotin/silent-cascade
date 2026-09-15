"""Closed tensor layouts without neural construction or producer arithmetic.

These are expected storage contracts, not measurements. Trace widths come from
authenticated examples. Actual comparisons enumerate executed tensors separately.
"""

from silent_cascade.models.config import NeuralModelConfig


def parameter_shapes(config: NeuralModelConfig) -> dict[str, tuple[int, ...]]:
    """Unique parameter layout of either approved assembled architecture."""
    c = config
    result = {
        "/mode_embedding.weight": (c.mode_count, c.mode_dim),
        "/record_encoder.entity_embedding.weight": (c.entity_count, c.entity_dim),
        "/record_encoder.kind_embedding.weight": (3, c.kind_dim),
        "/record_encoder.hazard_embedding.weight": (5, c.hazard_dim),
        "/record_encoder.provenance_embedding.weight": (3, c.provenance_dim),
        "/shared_jump.event_embedding.weight": (3, c.kind_dim),
        "/scorer.record_projection.weight": (c.query_dim, c.record_dim),
    }

    def linear(name, inputs, outputs):
        result[f"/{name}.weight"] = (outputs, inputs)
        result[f"/{name}.bias"] = (outputs,)

    def norm(name, width):
        result[f"/{name}.weight"] = result[f"/{name}.bias"] = (width,)

    for name, inputs, outputs in (
        ("record_encoder.network.0", c.record_input_dim, c.record_hidden_dim),
        ("record_encoder.network.2", c.record_hidden_dim, c.record_dim),
        ("scorer.query_network.0", c.retrieval_query_input_dim, c.query_dim),
        ("scorer.pair_network.0", c.retrieval_pair_input_dim, c.query_dim),
        ("scorer.pair_network.2", c.query_dim, 1),
        ("controller.network.0", c.controller_input_dim, c.controller_hidden_dim),
        ("controller.network.3", c.controller_hidden_dim, c.controller_hidden_dim),
        ("controller.network.6", c.controller_hidden_dim, c.controller_output_dim),
        ("shared_jump.network.0", c.jump_input_dim, c.jump_hidden_dim),
        ("shared_jump.network.3", c.jump_hidden_dim, c.jump_output_dim),
        ("external_encoder.network.0", c.external_input_dim, c.external_hidden_dim),
        ("external_encoder.network.2", c.external_hidden_dim, c.external_dim),
        ("external_encoder.injection", c.external_dim, c.external_injection_output_dim),
        ("focus_projection", c.entity_dim, 64),
        ("compose_heads.trunk.0", c.composition_input_dim, c.head_hidden_dim),
        ("action_heads.trunk.0", c.action_input_dim, c.head_hidden_dim),
    ):
        linear(name, inputs, outputs)
    for name, width in (
        ("controller.network.1", c.controller_hidden_dim),
        ("controller.network.4", c.controller_hidden_dim),
        ("shared_jump.network.1", c.jump_hidden_dim),
        ("compose_heads.trunk.1", c.head_hidden_dim),
        ("action_heads.trunk.1", c.head_hidden_dim),
    ):
        norm(name, width)
    for name, outputs in (
        ("role", 5),
        ("next_focus", c.entity_count),
        ("hazard", 4),
        ("delay", 1),
        ("deadline", 1),
        ("status", 3),
        ("confidence", 1),
        ("append_support", 1),
        ("continue_search", 1),
    ):
        linear("compose_heads." + name, c.head_hidden_dim, outputs)
    linear("action_heads.classifier", c.head_hidden_dim, 5)
    linear("action_heads.lead", c.head_hidden_dim, 1)
    return result


def forward_shapes(config, batch_size, observations, steps, *, content):
    """Every retained objective tensor plus empty-memory and guard parity probes."""
    c, b = config, batch_size
    result = {}

    def put(prefix, fields, size=b):
        for name, tail in fields.items():
            result[f"{prefix}/{name}"] = (size, *tail)

    context = {
        "workspace/latent": (c.latent_dim,),
        "workspace/accumulators": (3,),
        "memory_embeddings": (c.memory_slots, c.record_dim),
        "eligibility": (c.memory_slots,),
        "support_mask": (c.memory_slots,),
        "active_slot_indices": (),
        "modes": (),
        "time_features": (c.time_feature_dim,),
        "hypothesis_features": (c.hypothesis_feature_dim,),
    }
    parameters = {
        "flow_targets": (c.latent_dim,),
        "flow_rates": (c.latent_dim,),
        "raw_guard_targets": (3,),
        "guard_targets": (3,),
        "guard_rates": (3,),
    }
    preview = {"features": (c.preview_dim,), "has_candidate": ()}
    retrieval = {
        name: (c.memory_slots,) for name in ("raw_scores", "masked_logits", "eligible_mask")
    }
    composition = {
        "role_logits": (5,),
        "next_focus_logits": (c.entity_count,),
        "hazard_logits": (4,),
        "status_logits": (3,),
        "log_delay": (),
        "normalized_deadline": (),
        "confidence_logit": (),
        "append_support_logit": (),
        "continue_search_logit": (),
    }
    targets = (
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

    def boundary(prefix):
        put(prefix + "/context", context)
        put(prefix + "/parameters", parameters)
        put(prefix + "/preview", preview)
        put(prefix, {"mask": ()})

    for branch in ("timed", "content") if content else ("timed",):
        prefix = "/" + branch
        put(prefix + "/final_context", context)
        put(
            prefix + "/targets",
            {name: (steps,) for name in (*targets, "real_step_mask", "final_dormancy_mask")},
        )
        put(prefix + "/targets/validity", {name: (steps,) for name in targets})
        for col in range(observations):
            boundary(f"{prefix}/observations/{col}")
        for col in range(steps):
            operation = f"{prefix}/steps/{col}"
            put(operation + "/prediction_context", context)
            put(operation + "/post_context", context)
            put(operation + "/retrieval", retrieval)
            put(operation + "/composition", composition)
            if branch == "timed":
                boundary(f"{prefix}/boundaries/{col}")
                put(operation + "/post_parameters", parameters)
                put(operation, {"crossings": (3,), "legal_guards": (3,)})
                put(operation + "/action", {"class_logits": (5,), "lead_fraction": ()})
            else:
                put(operation, {"row_mask": (), "kind": (), "raw_recall_target": ()})
        if branch == "timed":
            put(
                prefix,
                {
                    "pre_jump_latent": (observations + steps, c.latent_dim),
                    "post_jump_latent": (observations + steps, c.latent_dim),
                    "jump_mask": (observations + steps,),
                },
            )
    put("empty/0", preview, size=1)
    put("empty/1", parameters, size=1)
    result["guard_probe"] = (1, 3)
    return result


def reduction_position_shape(key, batch_size, observations, steps):
    """Raw reductions retain padded boundaries; state instead retains every jump."""
    if key in ("state_slow_change", "state_bound"):
        return (batch_size, observations + steps)
    if key == "guard":
        return (batch_size, steps, 2)
    return (batch_size, steps)
