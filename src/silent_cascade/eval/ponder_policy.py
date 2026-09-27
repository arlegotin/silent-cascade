"""Public episode adapter for autonomous Phase 5A ponderer inference."""

import torch
import torch.nn.functional as F

from silent_cascade.env.episode import PublicEpisode
from silent_cascade.eval.compute import (
    NeuralComputeMeter,
    RuntimeCompute,
    aggregate_runtime_compute,
)
from silent_cascade.eventflow.state import ComputeCounters
from silent_cascade.models.activation_ponder import (
    PONDER_CAPS,
    ActivationPonderModel,
    PonderDecision,
    PonderStep,
)
from silent_cascade.schemas import Action
from silent_cascade.train.batches import pack_public_examples


class PonderExecutionError(RuntimeError):
    """Inference failure with the public cognitive prefix and measured work."""

    def __init__(
        self, cause: Exception, steps: tuple[PonderStep, ...], compute: RuntimeCompute
    ) -> None:
        super().__init__(str(cause))
        self.cause_type = type(cause).__name__
        self.steps = steps
        self.compute = compute


def ponder_public(
    model: ActivationPonderModel, public: PublicEpisode, *, cap: int
) -> PonderDecision:
    """Act from the public stream only; private terminal arbitration stays with evaluator."""
    if not isinstance(model, ActivationPonderModel) or not isinstance(public, PublicEpisode):
        raise TypeError("ponder inference requires a model and PublicEpisode")
    if cap not in PONDER_CAPS:
        raise ValueError("ponder cap must be one of the five predeclared budgets")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("Phase 5A ponder inference requires CPU")
    meter = NeuralComputeMeter(model)
    steps = []
    try:
        with meter, torch.no_grad():
            encoded = model.encode_public(pack_public_examples((public,)))
            hidden = encoded.hidden
            output = None
            for _ in range(cap):
                output = model.transition(encoded, hidden)
                hidden = output.hidden
                probability = float(torch.sigmoid(output.heads["halt"][0, 0]))
                action_class = int(output.heads["action"][0].argmax())
                offset = float(F.softplus(output.heads["action_offset"][0, 0]))
                steps.append(
                    PonderStep(
                        cognitive_timestamp=public.events[-1].timestamp,
                        selected_record_id=output.selected_record_ids[0],
                        halt_probability=probability,
                        action_class=action_class,
                        action_offset=offset,
                    )
                )
                if probability >= 0.5:
                    break
        assert output is not None
        last = steps[-1]
        action = (
            None
            if last.action_class == 4
            else Action(
                last.action_class,
                public.events[-1].timestamp + last.action_offset,
                public.events[-1].event_id,
            )
        )
        compute = aggregate_runtime_compute(
            ComputeCounters(),
            (meter.snapshot(),),
            entity_parameters=model.record_encoder.entity_embedding.weight.numel(),
        )
        return PonderDecision(
            action=action,
            steps=tuple(steps),
            stop_reason="halt" if last.halt_probability >= 0.5 else "cap_reached",
            compute=compute,
        )
    except Exception as caught:
        try:
            compute = aggregate_runtime_compute(
                ComputeCounters(),
                (meter.snapshot(),),
                entity_parameters=model.record_encoder.entity_embedding.weight.numel(),
            )
        except Exception:
            compute = RuntimeCompute()
        raise PonderExecutionError(caught, tuple(steps), compute) from caught
