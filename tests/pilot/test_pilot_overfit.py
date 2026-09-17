from collections import Counter


def test_fixed64_recipe_has_actual_workload_and_no_pilot_manifest():
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_overfit import fixed_overfit_examples

    config = resolve_pilot_config("phase4_smoke")
    examples = fixed_overfit_examples(config)
    assert len(examples) == 64
    assert [e.key.episode_index for e in examples] == list(range(64))
    assert Counter(e.key.root_seed for e in examples) == {449: 64}
    assert config.config.pilot.batch_size == 8 and config.config.pilot.max_steps == 4


def test_actual_boundary_preserves_all_outcomes_without_a_manifest(tmp_path):
    import torch

    from silent_cascade.models.event_flow import EventFlowModel
    from silent_cascade.train.pilot_config import resolve_pilot_config
    from silent_cascade.train.pilot_overfit import evaluate_overfit_boundary, fixed_overfit_examples

    config = resolve_pilot_config("phase4_smoke")
    torch.manual_seed(11)
    model = EventFlowModel(config.config.neural)
    result = evaluate_overfit_boundary(
        model,
        fixed_overfit_examples(config)[:4],
        config=config,
        device="cpu",
        output_dir=tmp_path,
        source_commit="a" * 40,
        update=0,
        ordered_data_sha256="b" * 64,
        weights_sha256="c" * 64,
    )
    assert len(result["outcomes"]) == 4
    assert result["gate_eligible"] is False
    assert all("score" in r and "error" in r and "actions" in r for r in result["outcomes"])
    assert all(p.requires_grad for p in model.parameters())
