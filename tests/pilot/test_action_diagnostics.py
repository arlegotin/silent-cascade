"""Real tiny archives prove context separation; none certify historical reproduction."""

import copy
import errno
import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from silent_cascade.config import ResolvedConfig, resolve_config
from silent_cascade.eventflow.neural import NeuralModelIdentity
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.models.event_flow import EventFlowModel
from silent_cascade.train.checkpoints import export_weights
from silent_cascade.train.config import Phase3Config
from silent_cascade.train.curriculum_data import (
    MAX_COMPONENT_MANIFEST_BYTES,
    ComponentManifest,
    CurriculumKey,
    make_curriculum_example,
)
from silent_cascade.train.pilot_config import resolve_pilot_config
from silent_cascade.train.state import TrainingError


@pytest.fixture
def diagnostic_case(tmp_path):
    config = resolve_pilot_config("phase4_smoke").config
    original = resolve_config(
        Phase3Config,
        tuple(
            Path(p)
            for p in (
                "configs/base.yaml",
                "configs/data/primary.yaml",
                "configs/model/event_flow.yaml",
                "configs/model/neural_components.yaml",
                "configs/train/smoke.yaml",
            )
        ),
    )
    original_config = original.config.model_copy(update={"neural": config.neural})
    original_raw = canonical_json_bytes(original_config)
    original = ResolvedConfig(original_config, original_raw, sha256_bytes(original_raw), ())
    examples = tuple(
        make_curriculum_example(
            original.config, CurriculumKey("ofd-one-hop-v1", "debug", 313, 337, index, "one_hop")
        )
        for index in range(4)
    )
    positive = next(e for e in examples if e.variant.value == "positive")
    true_class = positive.solution.hazard_type
    torch.manual_seed(11)
    model = EventFlowModel(config.neural)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.controller.network[-1].bias[-6:-3].fill_(math.log(3.0))
        model.controller.network[-1].bias[-3:].fill_(10.0)
        model.compose_heads.role.bias[1] = 8.0
        model.compose_heads.deadline.bias[0] = 2.0
        model.compose_heads.hazard.bias[true_class] = 8.0
        model.action_heads.classifier.bias[true_class] = 8.0

    def prepare(*, items=(positive,), candidate=model):
        directory = tmp_path / f"case-{len(list(tmp_path.iterdir()))}"
        directory.mkdir()
        descriptor = export_weights(directory, candidate, config=original, source_commit="a" * 40)
        manifest = ComponentManifest.from_examples(
            tuple(items), original.config, source_revision="a" * 40, plan_revision="b" * 40
        )
        manifest_path = directory / "manifest.json"
        manifest_path.write_bytes(canonical_json_bytes(manifest))
        return dict(
            model=candidate,
            examples=items,
            identity=NeuralModelIdentity.from_model(candidate, source_revision="a" * 40),
            config=config,
            source_weights_path=directory / descriptor.relative_path,
            expected_source_weights_sha256=descriptor.file_sha256,
            source_manifest_path=manifest_path,
            expected_source_manifest_sha256=sha256_bytes(manifest_path.read_bytes()),
            output_dir=directory / "diagnosis",
        )

    def run(**kwargs):
        from silent_cascade.eval.action_diagnostics import diagnose_actions

        return diagnose_actions(**prepare(**kwargs))

    return SimpleNamespace(
        model=model,
        positive=positive,
        examples=examples,
        true_class=true_class,
        run=run,
        prepare=prepare,
    )


def read_rows(report):
    return [
        json.loads(line) for line in (report.output_dir / "readouts.jsonl").read_text().splitlines()
    ]


def test_missing_autonomous_act_remains_a_failure(diagnostic_case):
    case = diagnostic_case
    with torch.no_grad():
        case.model.controller.network[-1].bias[-4] = -10.0  # ACT target dormant.
    report = case.run()
    assert report.autonomous.positive_count == 1
    assert report.autonomous.action_count == 0
    assert report.autonomous.missing_action_count == 1
    assert report.autonomous.timed_success_count == 0
    assert report.teacher_timed.label == "teacher_context_diagnostic"
    assert report.teacher_timed.correct_count == 1
    assert report.autonomous_raw.correct_count == 0
    assert report.autonomous_raw.missing_action_count == 1
    assert not report.gate_eligible
    assert not report.historical_reproduction
    rows = read_rows(report)
    missing = next(r for r in rows if r["context_kind"] == "autonomous_act_raw_five_way")
    assert missing["logits"] is None and not missing["act_occurred"]
    assert missing["raw_five_way_choice"] is None
    assert missing["guard_targets"][2] < 1.0
    assert missing["guard_crossings"][2] is None
    assert missing["state_sha256"] is not None
    assert missing["segment_sha256"] is not None
    assert (report.output_dir / "autonomous/episodes/00000.trajectory.json.gz").exists()


@pytest.mark.parametrize("abstain", [False, True])
def test_legal_shield_and_raw_argmax_use_the_same_actual_act(diagnostic_case, abstain):
    case = diagnostic_case
    wrong = (case.true_class + 1) % 4
    with torch.no_grad():
        case.model.action_heads.classifier.bias.zero_()
        case.model.action_heads.classifier.bias[wrong] = 8.0
        case.model.action_heads.classifier.bias[4] = 10.0 if abstain else 0.0
    report = case.run()
    assert report.autonomous.action_count == 1
    assert report.autonomous.correct_count == 0
    assert report.autonomous.timed_success_count == 0
    rows = read_rows(report)
    legal = next(r for r in rows if r["context_kind"] == "autonomous_legal_shield")
    raw = next(r for r in rows if r["context_kind"] == "autonomous_act_raw_five_way")
    assert legal["legal_act_shield_choice"] == wrong
    assert raw["raw_five_way_choice"] == (4 if abstain else wrong)
    assert legal["state_sha256"] == raw["state_sha256"]
    assert legal["segment_sha256"] == raw["segment_sha256"]
    assert legal["logits"] == raw["logits"]
    assert legal["act_occurred"] and raw["act_occurred"]
    assert legal["trajectory_act_state_verified"]
    assert legal["guard_targets"][2] > 1.0
    assert legal["guard_crossings"][2] > 0.0


def test_immediate_observer_is_transparent_and_future_context_is_distinct(diagnostic_case):
    case = diagnostic_case
    with torch.no_grad():
        head = case.model.action_heads
        head.trunk[0].weight[0, 561] = 1.0  # elapsed-since-activation log feature
        head.trunk[1].weight.fill_(1.0)
        head.classifier.weight[(case.true_class + 1) % 4, 0] = 30.0
    report = case.run()
    immediate, teacher = [
        next(r for r in read_rows(report) if r["context_kind"] == kind)
        for kind in ("historical_immediate_five_way", "teacher_context_diagnostic")
    ]
    assert immediate["raw_five_way_choice"] == case.true_class
    assert teacher["raw_five_way_choice"] != case.true_class
    assert immediate["elapsed_since_activation"] == 0.0
    assert teacher["elapsed_since_activation"] > 0.0
    assert immediate["segment_sha256"] is None
    assert immediate["guard_context"] == "not_installed_in_historical_recipe"
    assert not immediate["act_occurred"]
    assert report.observer_prediction_equivalence


def test_negative_teacher_readout_is_post_composition_and_all_variants_count(diagnostic_case):
    report = diagnostic_case.run(items=diagnostic_case.examples)
    teacher = [r for r in read_rows(report) if r["context_kind"] == "teacher_context_diagnostic"]
    assert len(teacher) == 4
    assert sum(r["true_class"] == 4 for r in teacher) == 2
    assert all(r["readout_position"] == "post_composition" for r in teacher if r["true_class"] == 4)
    assert set(report.teacher_timed.by_variant) == {
        "positive",
        "safe_negative",
        "disconnected_negative",
    }
    assert report.autonomous.episode_count == report.autonomous_raw.episode_count == 4
    assert sum(sum(row) for row in report.autonomous_raw.confusion) == 4


@pytest.mark.parametrize(
    "mutation", ["weights", "manifest", "model", "identity", "order", "example"]
)
def test_original_authentication_precedes_autonomous_execution(
    diagnostic_case, monkeypatch, mutation
):
    from silent_cascade.eval import action_diagnostics as diagnostics

    args = diagnostic_case.prepare(items=diagnostic_case.examples)
    if mutation in {"weights", "manifest"}:
        args[f"expected_source_{mutation}_sha256"] = "0" * 64
    elif mutation == "model":
        args["model"] = copy.deepcopy(args["model"])
        with torch.no_grad():
            args["model"].action_heads.classifier.bias[0] += 1.0
    elif mutation == "identity":
        args["identity"] = replace(args["identity"], source_revision="c" * 40)
    elif mutation == "order":
        args["examples"] = tuple(reversed(args["examples"]))
    else:
        args["examples"] = (
            replace(args["examples"][0], target_hash="0" * 64),
            *args["examples"][1:],
        )

    def forbidden(*args, **kwargs):
        pytest.fail("unauthenticated inputs reached autonomous execution")

    monkeypatch.setattr(diagnostics, "evaluate_episodes", forbidden)
    with pytest.raises((ValueError, TrainingError), match=r"hash|identity|model|examples"):
        diagnostics.diagnose_actions(**args)
    assert not args["output_dir"].exists()


def test_timed_action_gradients_and_caller_state_are_preserved(diagnostic_case):
    case = diagnostic_case
    torch.manual_seed(19)
    candidate = EventFlowModel(case.model.config).train()
    for parameter in candidate.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = copy.deepcopy(candidate.state_dict())
    report = case.run(candidate=candidate)
    gradients = report.gradient_diagnostics
    assert gradients["action_head_l2"] > 0
    assert gradients["recurrent_l2"] > 0
    assert gradients["all_finite"]
    assert gradients["optimizer_steps"] == 0
    assert candidate.training
    assert all(torch.equal(value, before[name]) for name, value in candidate.state_dict().items())
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in candidate.parameters())


def test_diagnostic_runtime_receives_only_public_inputs(diagnostic_case, monkeypatch):
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent
    from silent_cascade.models.types import ModelContext
    from silent_cascade.schemas import AgentInit, ExternalEventKind

    initialize, external = NeuralEventFlowAgent.initialize, NeuralEventFlowAgent.on_external

    def init(self, value):
        assert type(value) is AgentInit
        return initialize(self, value)

    def event(self, state, value):
        assert value.kind in {ExternalEventKind.FACT, ExternalEventKind.ACTIVATE}
        return external(self, state, value)

    monkeypatch.setattr(NeuralEventFlowAgent, "initialize", init)
    monkeypatch.setattr(NeuralEventFlowAgent, "on_external", event)
    action = EventFlowModel.action

    def public_context(self, context):
        assert type(context) is ModelContext
        return action(self, context)

    monkeypatch.setattr(EventFlowModel, "action", public_context)
    assert diagnostic_case.run().foundation_model_calls == 0


def test_script_rejects_wrong_original_hash_before_diagnosis(diagnostic_case, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "diagnose_script", "scripts/diagnose_phase4_actions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = diagnostic_case.prepare()
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose_phase4_actions.py",
            "--weights",
            str(args["source_weights_path"]),
            "--manifest",
            str(args["source_manifest_path"]),
            "--expected-weights-sha256",
            "0" * 64,
            "--expected-manifest-sha256",
            args["expected_source_manifest_sha256"],
            "--profile",
            "phase4_smoke",
            "--device",
            "cpu",
            "--output",
            str(args["output_dir"]),
        ],
    )
    assert module.main() == 1
    assert not args["output_dir"].exists()


def test_runtime_error_keeps_positive_denominator_and_crash_evidence(diagnostic_case, monkeypatch):
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent
    from silent_cascade.models.errors import NeuralError

    def broken_runtime(self, state, event):
        raise NeuralError("controlled autonomous failure")

    monkeypatch.setattr(NeuralEventFlowAgent, "on_internal", broken_runtime)
    report = diagnostic_case.run()
    assert report.autonomous.episode_count == report.autonomous.positive_count == 1
    assert report.autonomous.error_count == 1
    assert report.autonomous.timed_success_count == 0
    assert report.teacher_timed.correct_count == 1
    assert (report.output_dir / "autonomous/crashes/index.json").exists()


def test_disk_preflight_precedes_predictions(diagnostic_case, monkeypatch):
    from silent_cascade.eval import action_diagnostics as diagnostics

    args = diagnostic_case.prepare()
    monkeypatch.setattr(diagnostics.shutil, "disk_usage", lambda path: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="disk space"):
        diagnostics.diagnose_actions(**args)
    assert not args["output_dir"].exists()


def test_script_authenticates_once_and_publishes_tiny_diagnostic(diagnostic_case, monkeypatch):
    from silent_cascade.eval import action_diagnostics as diagnostics

    spec = importlib.util.spec_from_file_location(
        "diagnose_script_success", "scripts/diagnose_phase4_actions.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = diagnostic_case.prepare()
    original = diagnostics.load_weight_bundle
    calls = []

    def load_once(*positional, **keywords):
        calls.append(keywords["expected_sha256"])
        return original(*positional, **keywords)

    monkeypatch.setattr(diagnostics, "load_weight_bundle", load_once)
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose_phase4_actions.py",
            "--weights",
            str(args["source_weights_path"]),
            "--manifest",
            str(args["source_manifest_path"]),
            "--expected-weights-sha256",
            args["expected_source_weights_sha256"],
            "--expected-manifest-sha256",
            args["expected_source_manifest_sha256"],
            "--profile",
            "phase4_smoke",
            "--device",
            "cpu",
            "--output",
            str(args["output_dir"]),
        ],
    )
    assert module.main() == 0
    assert calls == [args["expected_source_weights_sha256"]]
    report = json.loads((args["output_dir"] / "report.json").read_text())
    assert not report["historical_reproduction"]
    assert report["autonomous"]["episode_count"] == 1
    assert not report["gate_eligible"]
    assert str(args["source_weights_path"]) not in json.dumps(report)


def test_teacher_register_is_not_reported_as_the_composition_prediction(diagnostic_case):
    case = diagnostic_case
    wrong = (case.true_class + 1) % 4
    with torch.no_grad():
        case.model.compose_heads.hazard.bias.zero_()
        case.model.compose_heads.hazard.bias[wrong] = 8.0
    report = case.run(items=case.examples)
    teacher = [r for r in read_rows(report) if r["context_kind"] == "teacher_context_diagnostic"]
    positive = next(r for r in teacher if r["public_id"] == case.positive.init.episode_public_id)
    assert positive["context_hypothesis_class"] == case.true_class
    assert positive["predicted_hypothesis_class"] == wrong
    assert all(r["predicted_hypothesis_class"] is None for r in teacher if r["true_class"] == 4)


def test_oversized_original_manifest_is_rejected_before_digest_read(diagnostic_case):
    from silent_cascade.eval.action_diagnostics import diagnose_actions

    args = diagnostic_case.prepare()
    with args["source_manifest_path"].open("ab") as handle:
        handle.write(b" " * MAX_COMPONENT_MANIFEST_BYTES)
    # A digest mismatch must not mask the earlier required bounded-reader check.
    args["expected_source_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=r"archive\.byte_limit"):
        diagnose_actions(**args)
    assert not args["output_dir"].exists()


def test_symlink_original_manifest_is_rejected_before_digest_read(diagnostic_case):
    from silent_cascade.eval.action_diagnostics import diagnose_actions

    args = diagnostic_case.prepare()
    link = args["source_manifest_path"].with_name("linked-manifest.json")
    link.symlink_to(args["source_manifest_path"])
    args["source_manifest_path"] = link
    args["expected_source_manifest_sha256"] = "0" * 64
    with pytest.raises(OSError) as caught:
        diagnose_actions(**args)
    assert caught.value.errno == errno.ELOOP
    assert not args["output_dir"].exists()


def test_original_manifest_digest_binds_the_snapshot_used_by_the_loader(
    diagnostic_case, monkeypatch
):
    from silent_cascade.eval import action_diagnostics as diagnostics

    args = diagnostic_case.prepare()
    other = diagnostic_case.prepare(items=diagnostic_case.examples)
    path = args["source_manifest_path"]
    original_bytes = path.read_bytes()
    other_bytes = other["source_manifest_path"].read_bytes()
    loader = diagnostics.load_component_manifest

    def substituted_snapshot(path, config):
        path.write_bytes(other_bytes)
        try:
            return loader(path, config)
        finally:
            path.write_bytes(original_bytes)

    monkeypatch.setattr(diagnostics, "load_component_manifest", substituted_snapshot)
    with pytest.raises(ValueError, match=r"manifest.*changed"):
        diagnostics._authenticate_files(
            source_weights_path=args["source_weights_path"],
            expected_source_weights_sha256=args["expected_source_weights_sha256"],
            source_manifest_path=path,
            expected_source_manifest_sha256=args["expected_source_manifest_sha256"],
            device="cpu",
        )


@pytest.mark.parametrize("failure_stage", ["prediction_validation", "segment_installation"])
def test_failure_after_action_head_retains_uncommitted_attempt_and_denominator(
    diagnostic_case, monkeypatch, failure_stage
):
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent
    from silent_cascade.models.errors import NeuralError
    from silent_cascade.models.heads import ActionPredictions

    if failure_stage == "prediction_validation":
        original = NeuralEventFlowAgent._predictions

        def fail_validation(self, output, expected, context):
            if expected is ActionPredictions:
                raise NeuralError("controlled failure after actual action-head prediction")
            return original(self, output, expected, context)

        monkeypatch.setattr(NeuralEventFlowAgent, "_predictions", fail_validation)
    else:
        original = NeuralEventFlowAgent._install

        def fail_installation(self, core, time, context, diagnostics):
            if diagnostics.callback == "act":
                raise NeuralError("controlled failure after action before segment installation")
            return original(self, core, time, context, diagnostics)

        monkeypatch.setattr(NeuralEventFlowAgent, "_install", fail_installation)

    report = diagnostic_case.run()
    assert report.autonomous.episode_count == report.autonomous.positive_count == 1
    assert report.autonomous.error_count == report.autonomous_raw.error_count == 1
    assert report.autonomous.action_count == report.autonomous_raw.action_count == 0
    assert report.autonomous.missing_action_count == report.autonomous_raw.missing_action_count == 1
    assert report.autonomous.timed_success_count == report.autonomous_raw.correct_count == 0
    for row in read_rows(report):
        if row["context_kind"].startswith("autonomous"):
            assert not row["act_occurred"]
            assert row["raw_five_way_choice"] is None
            assert row["legal_act_shield_choice"] is None
            assert row["logits"] is None
            assert len(row["failed_action_attempts"]) == 1
            attempt = row["failed_action_attempts"][0]
            assert attempt["committed"] is False
            assert len(attempt["state_sha256"]) == 64
            assert attempt["context"]
    assert (report.output_dir / "DONE").exists()
    assert (report.output_dir / "autonomous/crashes/index.json").exists()
    assert (report.output_dir / "autonomous/episodes/00000.trajectory.json.gz").exists()


def test_successful_act_still_requires_its_observed_context(diagnostic_case, monkeypatch):
    from silent_cascade.eventflow.neural import NeuralEventFlowAgent

    initialize = NeuralEventFlowAgent.initialize

    def drop_runtime_observer(self, init):
        # Deliberately damage only diagnostic observation on the inference copy.
        self.model.action_heads._forward_hooks.clear()
        return initialize(self, init)

    monkeypatch.setattr(NeuralEventFlowAgent, "initialize", drop_runtime_observer)
    with pytest.raises(ValueError, match="ACT observation inventory"):
        diagnostic_case.run()
