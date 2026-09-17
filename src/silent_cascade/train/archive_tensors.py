"""Schema-independent, finite AdamW tensors for closed training archives."""

import torch

from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.train.state import TrainingError


def validate_moments(tensors, global_step):
    for name, value in tensors.items():
        if not name.startswith("optimizer/"):
            continue
        if not bool(torch.isfinite(value).all()):
            raise TrainingError("nonfinite AdamW state")
        if name.endswith("/step"):
            step = float(value)
            if not 1 <= step <= global_step or not step.is_integer():
                raise TrainingError("invalid AdamW step")
        elif name.endswith("/exp_avg_sq") and bool((value < 0).any()):
            raise TrainingError("negative AdamW second moment")


def optimizer_options(recipe):
    return dict(
        lr=recipe.learning_rate,
        betas=list(recipe.betas),
        eps=recipe.epsilon,
        weight_decay=recipe.weight_decay,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
        decoupled_weight_decay=True,
    )


def snapshot_optimizer(model, optimizer, recipe, global_step):
    if type(optimizer) is not torch.optim.AdamW:
        raise TrainingError("expected AdamW")
    by_id = {id(p): name for name, p in model.named_parameters()}
    groups, names, tensors, state_names = [], [], {}, []
    for group in optimizer.param_groups:
        group_names = tuple(by_id[id(p)] for p in group["params"])
        options = {
            key: list(value) if key == "betas" else value
            for key, value in group.items()
            if key != "params"
        }
        if canonical_json_bytes(options) != canonical_json_bytes(optimizer_options(recipe)):
            raise TrainingError("AdamW options differ from configuration")
        groups.append({"names": group_names, "options": options})
        names.extend(group_names)
    if len(names) != len(set(names)) or set(names) != set(by_id.values()):
        raise TrainingError("optimizer must own every unique parameter exactly once")
    for parameter, state in optimizer.state.items():
        name = by_id[id(parameter)]
        if not state:
            continue
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise TrainingError("unknown AdamW state")
        state_names.append(name)
        for kind, value in state.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or value.shape != (() if kind == "step" else parameter.shape)
                or value.device.type != ("cpu" if kind == "step" else parameter.device.type)
            ):
                raise TrainingError("AdamW state tensor layout differs")
            tensors[f"optimizer/{name}/{kind}"] = value.detach().cpu().contiguous().clone()
    validate_moments(tensors, global_step)
    return tuple(groups), tuple(sorted(state_names)), tensors
