"""Behavioral tests for learned retrieval and its non-mutating preview."""

import math
from dataclasses import fields, replace

import pytest
import torch
from torch import nn


def _zero_scorer(scorer) -> None:
    with torch.no_grad():
        for parameter in scorer.parameters():
            parameter.zero_()


def _configure_bilinear_scorer(scorer, query_inputs: tuple[int, ...]) -> None:
    """Make selected query inputs compare against matching record coordinates."""
    _zero_scorer(scorer)
    with torch.no_grad():
        query_layer = scorer.query_network[0]
        for coordinate, input_index in enumerate(query_inputs):
            query_layer.weight[coordinate, input_index] = 1.0
            scorer.record_projection.weight[coordinate, coordinate] = 1.0


def _context_with(
    model_context,
    *,
    latent: torch.Tensor | None = None,
    embeddings: torch.Tensor | None = None,
    eligible: torch.Tensor | None = None,
    support: torch.Tensor | None = None,
):
    from silent_cascade.models.types import ModelContext, TensorWorkspace

    workspace = model_context.workspace
    if latent is not None:
        workspace = TensorWorkspace(latent, workspace.accumulators)
    return ModelContext(
        workspace=workspace,
        memory_embeddings=(model_context.memory_embeddings if embeddings is None else embeddings),
        eligibility=model_context.eligibility if eligible is None else eligible,
        support_mask=model_context.support_mask if support is None else support,
        active_slot_indices=model_context.active_slot_indices,
        modes=model_context.modes,
        time_features=model_context.time_features,
        hypothesis_features=model_context.hypothesis_features,
    )


def test_context_features_pool_consumed_support_independent_of_eligibility(
    model_context,
) -> None:
    from silent_cascade.models.common import context_features

    embeddings = torch.zeros_like(model_context.memory_embeddings)
    embeddings[0, 0] = 2.0
    embeddings[0, 1] = 4.0
    support = torch.zeros_like(model_context.support_mask)
    support[0, :2] = True
    eligible = model_context.eligibility.clone()
    eligible[0, 0] = False
    context = _context_with(
        model_context, embeddings=embeddings, eligible=eligible, support=support
    )
    mode_embedding = nn.Embedding(6, 8)
    with torch.no_grad():
        mode_embedding.weight.copy_(torch.arange(48, dtype=torch.float32).reshape(6, 8))

    features = context_features(context, mode_embedding)

    assert features.shape == (2, 570)
    torch.testing.assert_close(features[0, 456:464], mode_embedding.weight[1])
    assert features[0, 464:560].eq(3.0).all()
    assert features[1, 464:560].eq(0.0).all()


def test_wrong_subject_candidate_is_not_removed_by_retrieval_mask(
    model_context, neural_config
) -> None:
    from silent_cascade.memory.retrieval import RetrievalScorer

    scorer = RetrievalScorer(neural_config.neural)
    _configure_bilinear_scorer(scorer, (0,))
    embeddings = torch.zeros_like(model_context.memory_embeddings)
    embeddings[0, 0, 0] = 1.0  # subject 1: the activation focus in this test scenario
    embeddings[0, 1, 0] = 9.0  # subject 2: deliberately wrong, but schema-legal
    context = _context_with(model_context, embeddings=embeddings)
    with torch.no_grad():
        scorer.query_network[0].bias[0] = 1.0

    scores = scorer(context)

    assert scores.raw_scores.shape == (2, 64)
    assert scores.masked_logits.shape == (2, 64)
    assert scores.eligible_mask[0, 1]
    assert torch.isfinite(scores.masked_logits[0, 1])
    assert scores.raw_scores[0, 1] > scores.raw_scores[0, 0]


def test_preview_is_finite_for_empty_memory(model_context, neural_config) -> None:
    from silent_cascade.memory.retrieval import RetrievalScorer

    context = replace(model_context, eligibility=torch.zeros_like(model_context.eligibility))
    preview = RetrievalScorer(neural_config.neural).preview(context)
    assert torch.isfinite(preview.features).all()
    assert preview.features.eq(0).all()
    assert not preview.has_candidate.any()


def test_preview_one_candidate_has_zero_margin_and_entropy(model_context, neural_config) -> None:
    from silent_cascade.memory.retrieval import RetrievalScorer

    scorer = RetrievalScorer(neural_config.neural)
    _zero_scorer(scorer)
    embeddings = torch.zeros_like(model_context.memory_embeddings)
    embeddings[0, 7] = torch.arange(96, dtype=torch.float32)
    eligible = torch.zeros_like(model_context.eligibility)
    eligible[0, 7] = True
    preview = scorer.preview(_context_with(model_context, embeddings=embeddings, eligible=eligible))

    torch.testing.assert_close(preview.features[0, :4], torch.tensor([0.0, 0.0, 0.0, 1 / 64]))
    torch.testing.assert_close(preview.features[0, 4:], embeddings[0, 7])
    assert preview.has_candidate.tolist() == [True, False]


def test_preview_includes_every_fourth_place_boundary_tie_and_is_permutation_invariant(
    model_context, neural_config
) -> None:
    from silent_cascade.memory.retrieval import RetrievalScorer

    scorer = RetrievalScorer(neural_config.neural)
    _zero_scorer(scorer)
    embeddings = torch.zeros_like(model_context.memory_embeddings)
    embeddings[0, :5, 0] = torch.arange(1, 6, dtype=torch.float32)
    eligible = torch.zeros_like(model_context.eligibility)
    eligible[0, :5] = True
    context = _context_with(model_context, embeddings=embeddings, eligible=eligible)

    preview = scorer.preview(context)
    permutation = torch.arange(64).roll(11)
    permuted = _context_with(
        model_context,
        embeddings=embeddings[:, permutation],
        eligible=eligible[:, permutation],
    )
    permuted_preview = scorer.preview(permuted)

    assert preview.features[0, 4].item() == pytest.approx(3.0)
    assert preview.features[0, 2].item() == pytest.approx(math.log(5.0))
    torch.testing.assert_close(permuted_preview.features, preview.features, rtol=1e-6, atol=1e-6)


def test_select_record_ids_uses_lowest_record_id_for_exact_ties(
    model_context, neural_config
) -> None:
    from silent_cascade.memory.retrieval import RetrievalScorer, select_record_ids

    scorer = RetrievalScorer(neural_config.neural)
    _zero_scorer(scorer)
    eligible = torch.zeros_like(model_context.eligibility)
    eligible[0, :2] = True
    scores = scorer(_context_with(model_context, eligible=eligible))
    record_ids = (
        (10, 3, *(None for _ in range(62))),
        tuple(None for _ in range(64)),
    )

    assert select_record_ids(scores, record_ids) == (3, None)


def test_preview_does_not_mutate_context(model_context, neural_config) -> None:
    from silent_cascade.memory.retrieval import RetrievalScorer

    snapshots = {
        field.name: (
            getattr(model_context, field.name).latent.clone()
            if field.name == "workspace"
            else getattr(model_context, field.name).clone()
        )
        for field in fields(model_context)
    }
    accumulator_snapshot = model_context.workspace.accumulators.clone()

    RetrievalScorer(neural_config.neural).preview(model_context)

    torch.testing.assert_close(model_context.workspace.latent, snapshots["workspace"])
    torch.testing.assert_close(model_context.workspace.accumulators, accumulator_snapshot)
    for name, snapshot in snapshots.items():
        if name != "workspace":
            torch.testing.assert_close(getattr(model_context, name), snapshot)


def test_flowed_query_is_rescored_and_can_change_selection(model_context, neural_config) -> None:
    from silent_cascade.memory.retrieval import RetrievalScorer, select_record_ids

    scorer = RetrievalScorer(neural_config.neural)
    # Retrieval query order is focus(64), fast(256), slow(64), drives(8), mode(8), support(96).
    _configure_bilinear_scorer(scorer, (64, 65))
    embeddings = torch.zeros_like(model_context.memory_embeddings)
    embeddings[0, 0, 0] = 1.0
    embeddings[0, 1, 1] = 1.0
    eligible = torch.zeros_like(model_context.eligibility)
    eligible[0, :2] = True
    initial_latent = torch.zeros_like(model_context.workspace.latent)
    initial_latent[0, 0] = 1.0
    flowed_latent = initial_latent.clone()
    flowed_latent[0, :2] = torch.tensor([0.0, 1.0])
    record_ids = (
        (10, 20, *(None for _ in range(62))),
        tuple(None for _ in range(64)),
    )

    initial_context = _context_with(
        model_context,
        latent=initial_latent,
        embeddings=embeddings,
        eligible=eligible,
    )
    flowed_context = _context_with(
        model_context,
        latent=flowed_latent,
        embeddings=embeddings,
        eligible=eligible,
    )
    initial = scorer(initial_context)
    flowed = scorer(flowed_context)
    initial_preview = scorer.preview(initial_context)
    flowed_preview = scorer.preview(flowed_context)

    assert select_record_ids(initial, record_ids)[0] == 10
    assert select_record_ids(flowed, record_ids)[0] == 20
    assert initial_preview.features[0, 4] > initial_preview.features[0, 5]
    assert flowed_preview.features[0, 5] > flowed_preview.features[0, 4]


def test_injected_mode_embedding_has_single_parameter_owner(neural_config) -> None:
    from silent_cascade.memory.retrieval import RetrievalScorer

    mode_embedding = nn.Embedding(6, 8)
    scorer = RetrievalScorer(neural_config.neural, mode_embedding=mode_embedding)

    assert scorer.mode_embedding is mode_embedding
    assert all("mode_embedding" not in name for name, _ in scorer.named_parameters())


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_encoder_and_scorer_have_finite_gradients_and_permutation_equivariance(
    public_memories, neural_config, device
) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    from silent_cascade.memory.encoder import RecordEncoder
    from silent_cascade.memory.retrieval import RetrievalScorer
    from silent_cascade.memory.tensor_store import pack_memory
    from silent_cascade.models.types import ModelContext, TensorWorkspace

    config = neural_config.neural
    encoder = RecordEncoder(config).to(device)
    scorer = RetrievalScorer(config).to(device)
    records = pack_memory(public_memories, (0.0, 9.0), device=device)
    embeddings = encoder(records)
    context = ModelContext(
        workspace=TensorWorkspace.zeros(2, device),
        memory_embeddings=embeddings,
        eligibility=records.valid_mask,
        support_mask=torch.zeros((2, 64), dtype=torch.bool, device=device),
        active_slot_indices=torch.full((2,), -1, dtype=torch.int64, device=device),
        modes=torch.tensor([1, 4], dtype=torch.int64, device=device),
        time_features=torch.zeros((2, 2), dtype=torch.float32, device=device),
        hypothesis_features=torch.zeros((2, 8), dtype=torch.float32, device=device),
    )
    permutation = torch.arange(64, device=device).roll(9).repeat(2, 1)
    permuted_context = replace(
        context,
        memory_embeddings=embeddings.gather(1, permutation.unsqueeze(-1).expand(-1, -1, 96)),
        eligibility=records.valid_mask.gather(1, permutation),
        support_mask=context.support_mask.gather(1, permutation),
    )

    scores = scorer(context)
    preview = scorer.preview(context)
    permuted_scores = scorer(permuted_context)
    permuted_preview = scorer.preview(permuted_context)
    torch.testing.assert_close(
        permuted_scores.raw_scores,
        scores.raw_scores.gather(1, permutation),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(permuted_preview.features, preview.features, rtol=1e-5, atol=1e-6)

    loss = scores.raw_scores[context.eligibility].sum() + preview.features.sum()
    loss.backward()
    for module in (encoder, scorer):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(bool(gradient.any()) for gradient in gradients)
