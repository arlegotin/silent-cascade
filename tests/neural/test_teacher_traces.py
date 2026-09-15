import math

import pytest

from silent_cascade.schemas import InternalEventKind, Mode


def test_teacher_content_modes_support_and_dormancy(neural_config):
    from silent_cascade.train.curriculum_data import CurriculumKey, make_curriculum_example
    from silent_cascade.train.traces import build_teacher_trace

    for i in range(12):
        example = make_curriculum_example(
            neural_config, CurriculumKey("ofd-one-hop-v1", "debug", 311, 331, i, "one_hop")
        )
        teacher = build_teacher_trace(example)
        assert len(teacher.steps) == len(example.oracle_trace.steps)
        for step in teacher.steps:
            assert step.delta > 0
            if step.kind is InternalEventKind.RECALL:
                assert step.pre_mode is Mode.SEARCHING
                assert step.post_mode is Mode.HAVE_MEMORY
                assert step.active_record_id_after == step.selected_record_id
            if step.kind is InternalEventKind.COMPOSE:
                assert step.pre_mode is Mode.HAVE_MEMORY
                assert step.append_support
                assert step.selected_record_id in step.support_after
                assert step.role in (0, 1, 2)
                if step.role in (1, 2):
                    assert step.focus is None
                    assert step.continue_search is None
        if example.solution.hazard_type is None:
            assert teacher.terminal_class == 4
            assert teacher.final_mode is Mode.QUIESCENT
            assert all(s.kind is not InternalEventKind.ACT for s in teacher.steps)
        else:
            assert teacher.steps[-1].action_class == example.solution.hazard_type
            assert teacher.steps[-1].action_lead == 0.175
            assert teacher.final_mode is Mode.QUIESCENT
            terminal_compose = teacher.steps[-2]
            assert terminal_compose.hazard_type == example.solution.hazard_type
            assert terminal_compose.log_delay == pytest.approx(
                math.log1p(example.solution.public_delay)
            )
            assert terminal_compose.normalized_deadline == pytest.approx(
                example.solution.public_delay
                / (1.0 + terminal_compose.timestamp - example.public.events[-1].timestamp)
            )


def test_hand_authored_contradiction_fixture_exercises_role_and_no_append_loss(
    neural_config,
    model_context,
):
    """An auxiliary fixture only; contradictory worlds are never curriculum parents."""
    from dataclasses import replace

    import torch
    from torch.nn import functional as functional

    from silent_cascade.memory.encoder import RecordEncoder
    from silent_cascade.memory.store import BoundedMemory, append_perceived_fact
    from silent_cascade.memory.tensor_store import pack_memory
    from silent_cascade.models.heads import ComposeHeads
    from silent_cascade.schemas import ExternalEvent, ExternalEventKind, HazardFact, SafeFact

    memory = BoundedMemory(64)
    for event in (
        ExternalEvent(0, 1.0, ExternalEventKind.FACT, HazardFact(7, 2, 4.0)),
        ExternalEvent(1, 2.0, ExternalEventKind.FACT, SafeFact(7)),
    ):
        memory = append_perceived_fact(memory, event)
    records = pack_memory((memory, memory), (0.0, 0.0), device="cpu")
    support = torch.zeros((2, 64), dtype=torch.bool)
    support[:, 0] = True
    hypothesis = torch.zeros((2, 8), dtype=torch.float32)
    hypothesis[:, 0] = 1.0  # Existing hazard hypothesis at the same subject.
    hypothesis[:, 3] = 1.0  # Hazard class 2.
    context = replace(
        model_context,
        memory_embeddings=RecordEncoder(neural_config.neural)(records),
        eligibility=records.valid_mask,
        support_mask=support,
        active_slot_indices=torch.ones(2, dtype=torch.int64),
        modes=torch.full((2,), 2, dtype=torch.int64),
        hypothesis_features=hypothesis,
    )
    heads = ComposeHeads(neural_config.neural)
    prediction = heads(context)
    role_target = torch.full((2,), 4, dtype=torch.int64)  # CONTRADICTORY, in ComposeRole order.
    loss = functional.cross_entropy(prediction.role_logits, role_target)
    loss += functional.binary_cross_entropy_with_logits(
        prediction.append_support_logit, torch.zeros(2)
    )
    loss.backward()
    assert heads.role.bias.grad[4] < 0
    assert heads.append_support.bias.grad.item() > 0
    assert torch.isfinite(loss)
