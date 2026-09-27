"""Teacher-forced content projection and masked objective for Phase 5A pondering."""

import copy
import gzip
import io
import json
import math
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from silent_cascade.env.episode import EpisodeBundle, episode_sha256
from silent_cascade.env.pilot import curriculum_to_bundle
from silent_cascade.env.reward import score_actions
from silent_cascade.eval.comparison_protocol import admit_ponder_budget, ponder_checkpoint_rank
from silent_cascade.eval.comparison_runner import (
    PublicProposal,
    arbitrate_proposal,
    ponder_state_sha256,
)
from silent_cascade.eval.comparison_types import (
    BudgetLedger,
    ComparisonConfig,
    CostProfile,
    PonderTrainingResult,
)
from silent_cascade.eval.compute import NeuralComputeMeter, parameter_counts
from silent_cascade.eval.ponder_policy import ponder_public
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from silent_cascade.io import atomic_create_bytes, atomic_write_bytes
from silent_cascade.models.activation_ponder import (
    PONDER_CAPS,
    ActivationPonderModel,
    matched_ponder_config,
)
from silent_cascade.models.types import PublicInputBatch
from silent_cascade.schemas import InternalEventKind
from silent_cascade.train.batches import TrainingBatch, pack_training_examples
from silent_cascade.train.curriculum_data import (
    CURRICULUM_VERSION,
    CurriculumKey,
    make_curriculum_example,
)
from silent_cascade.train.pilot_config import Phase4Config
from silent_cascade.train.pilot_data import PilotManifest

_INTEGER = ("role", "focus", "hazard", "status", "action_class")
_FLOAT = (
    "log_delay",
    "deadline",
    "confidence",
    "support",
    "continue",
    "action_offset",
)
_ALL = (*_INTEGER, *_FLOAT)


@dataclass(frozen=True, slots=True)
class PonderBatch:
    """Public inputs remain separate from the training-only teacher correspondence."""

    public: PublicInputBatch
    teacher_slots: torch.Tensor
    step_mask: torch.Tensor
    rollout_mask: torch.Tensor
    caps: torch.Tensor
    targets: dict[str, torch.Tensor]
    validity: dict[str, torch.Tensor]
    halt_target: torch.Tensor
    teacher_delay: torch.Tensor

    def to(self, device: str) -> "PonderBatch":
        return PonderBatch(
            public=self.public.to(device),
            teacher_slots=self.teacher_slots.to(device),
            step_mask=self.step_mask.to(device),
            rollout_mask=self.rollout_mask.to(device),
            caps=self.caps.to(device),
            targets={name: value.to(device) for name, value in self.targets.items()},
            validity={name: value.to(device) for name, value in self.validity.items()},
            halt_target=self.halt_target.to(device),
            teacher_delay=self.teacher_delay.to(device),
        )


@dataclass(frozen=True, slots=True)
class PonderLoss:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]
    denominators: dict[str, int]


def evaluate_competence(
    model: ActivationPonderModel, bundles: Sequence[EpisodeBundle], *, cap: int
) -> tuple[dict, ...]:
    """Keep private terminal and scoring in the evaluator, outside model callbacks."""
    rows = []
    model.eval()
    for bundle in bundles:
        decision = ponder_public(model, bundle.public, cap=cap)
        proposal = PublicProposal(actions=() if decision.action is None else (decision.action,))
        actions = arbitrate_proposal(bundle, proposal).actions
        score = score_actions(bundle.truth, actions)
        rows.append(
            {
                "public_id": bundle.public.init.episode_public_id,
                "episode_sha256": episode_sha256(bundle),
                "variant": bundle.truth.recipe.variant.value,
                "path_length": bundle.truth.recipe.requested_path_length,
                "actions": [asdict(action) for action in actions],
                "timed_success": score.timed_success,
                "score_reason": score.reason,
                "stop_reason": decision.stop_reason,
                "steps": [asdict(step) for step in decision.steps],
                "compute": decision.compute.model_dump(mode="json"),
            }
        )
    return tuple(rows)


def build_fixed_competence_set(config: Phase4Config) -> tuple[tuple, tuple[str, ...]]:
    """Separate 64-example DEBUG set, complete quartets at roots 7963/7993."""
    if not isinstance(config, Phase4Config):
        raise TypeError("competence set requires the accepted Phase 4 data settings")
    examples = tuple(
        make_curriculum_example(
            config,
            CurriculumKey(CURRICULUM_VERSION, "debug", 7963, 7993, index, stage),
        )
        for stage, count in (("one_hop", 16), ("two_hop", 16), ("primary", 32))
        for index in range(count)
    )
    public_ids = [example.public.init.episode_public_id for example in examples]
    if len(set(public_ids)) != 64:
        raise ValueError("fixed competence set has duplicate public identities")
    return examples, tuple(example.example_hash for example in examples)


def save_debug_checkpoint(
    path: Path,
    model: ActivationPonderModel,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    dataset_sha256: str,
    source_revision: str | None = None,
) -> None:
    """Write a complete CPU continuation capsule and a hash-bound descriptor."""
    if step < 0 or len(dataset_sha256) != 64:
        raise ValueError("invalid debug checkpoint step or dataset hash")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("debug checkpoint requires CPU tensors")
    payload = {
        "schema_version": "phase5a-ponder-debug-checkpoint-v1",
        "step": step,
        "dataset_sha256": dataset_sha256,
        "model_config_sha256": sha256_bytes(canonical_json_bytes(model.config)),
        "source_revision": source_revision,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    atomic_create_bytes(path, raw)
    atomic_create_bytes(
        path.with_suffix(".json"),
        canonical_json_bytes(
            {
                "schema_version": "phase5a-ponder-debug-checkpoint-index-v1",
                "step": step,
                "dataset_sha256": dataset_sha256,
                "model_config_sha256": payload["model_config_sha256"],
                "source_revision": source_revision,
                "checkpoint_sha256": sha256_bytes(raw),
            }
        )
        + b"\n",
    )


def load_debug_checkpoint(
    path: Path,
    model: ActivationPonderModel,
    optimizer: torch.optim.Optimizer,
    *,
    expected_dataset_sha256: str,
    expected_source_revision: str | None = None,
) -> int:
    descriptor = json.loads(path.with_suffix(".json").read_bytes())
    if (
        descriptor.get("checkpoint_sha256") != sha256_file(path)
        or descriptor.get("dataset_sha256") != expected_dataset_sha256
        or descriptor.get("model_config_sha256") != sha256_bytes(canonical_json_bytes(model.config))
        or descriptor.get("source_revision") != expected_source_revision
    ):
        raise ValueError("debug checkpoint binding differs")
    payload = torch.load(io.BytesIO(path.read_bytes()), map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != "phase5a-ponder-debug-checkpoint-v1"
        or payload.get("step") != descriptor.get("step")
        or payload.get("dataset_sha256") != expected_dataset_sha256
        or payload.get("model_config_sha256") != descriptor.get("model_config_sha256")
        or payload.get("source_revision") != expected_source_revision
    ):
        raise ValueError("debug checkpoint payload differs")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"])
    return int(payload["step"])


def save_main_checkpoint(
    path: Path,
    model: ActivationPonderModel,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    budget: BudgetLedger,
    config_sha256: str,
    source_revision: str,
) -> None:
    """Durable CPU continuation with bound source, optimizer, RNG, and consumed cost."""
    if (
        not 0 <= step <= 12000
        or len(config_sha256) != 64
        or len(source_revision) != 40
        or next(model.parameters()).device.type != "cpu"
    ):
        raise ValueError("invalid main checkpoint identity or device")
    payload = {
        "schema_version": "phase5a-ponder-main-checkpoint-v1",
        "step": step,
        "budget": budget.model_dump(mode="json"),
        "config_sha256": config_sha256,
        "source_revision": source_revision,
        "model_config_sha256": sha256_bytes(canonical_json_bytes(model.config)),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    atomic_create_bytes(path, raw)
    atomic_create_bytes(
        path.with_suffix(".json"),
        canonical_json_bytes(
            {
                "schema_version": "phase5a-ponder-main-checkpoint-index-v1",
                "step": step,
                "config_sha256": config_sha256,
                "source_revision": source_revision,
                "model_config_sha256": payload["model_config_sha256"],
                "checkpoint_sha256": sha256_bytes(raw),
            }
        )
        + b"\n",
    )


def load_main_checkpoint(
    path: Path,
    model: ActivationPonderModel,
    optimizer: torch.optim.Optimizer,
    *,
    config_sha256: str,
    source_revision: str,
) -> tuple[int, BudgetLedger]:
    descriptor = json.loads(path.with_suffix(".json").read_bytes())
    if (
        descriptor.get("schema_version") != "phase5a-ponder-main-checkpoint-index-v1"
        or descriptor.get("checkpoint_sha256") != sha256_file(path)
        or descriptor.get("config_sha256") != config_sha256
        or descriptor.get("source_revision") != source_revision
        or descriptor.get("model_config_sha256") != sha256_bytes(canonical_json_bytes(model.config))
    ):
        raise ValueError("main checkpoint binding differs")
    payload = torch.load(io.BytesIO(path.read_bytes()), map_location="cpu", weights_only=True)
    if (
        payload.get("schema_version") != "phase5a-ponder-main-checkpoint-v1"
        or payload.get("step") != descriptor.get("step")
        or payload.get("config_sha256") != config_sha256
        or payload.get("source_revision") != source_revision
        or payload.get("model_config_sha256") != descriptor.get("model_config_sha256")
    ):
        raise ValueError("main checkpoint payload differs")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"])
    return int(payload["step"]), BudgetLedger.model_validate_json(
        canonical_json_bytes(payload["budget"])
    )


def rebind_zero_update_checkpoint(
    original: Path,
    corrected: Path,
    model: ActivationPonderModel,
    optimizer: torch.optim.Optimizer,
    *,
    config_sha256: str,
    prior_source: str,
    corrected_source: str,
) -> dict:
    """Preserve the sole seed-11 initialization across a pre-update wiring correction."""
    if original.name != "main-0000.pt" or corrected_source == prior_source:
        raise ValueError("only the original zero-update capsule can be rebound once")
    step, budget = load_main_checkpoint(
        original,
        model,
        optimizer,
        config_sha256=config_sha256,
        source_revision=prior_source,
    )
    if step != 0 or budget.updates != 0:
        raise ValueError("a learned checkpoint cannot cross source revisions")
    state_sha = ponder_state_sha256(model)
    rng = torch.get_rng_state().clone()
    save_main_checkpoint(
        corrected,
        model,
        optimizer,
        step=0,
        budget=budget,
        config_sha256=config_sha256,
        source_revision=corrected_source,
    )
    if ponder_state_sha256(model) != state_sha or not torch.equal(torch.get_rng_state(), rng):
        raise ValueError("source rebind mutated model state or RNG")
    return {
        "schema_version": "phase5a-ponder-zero-update-source-correction-v1",
        "reason": "Phase 4 pilot config uses pilot, not the Phase 3 training field",
        "prior_source_revision": prior_source,
        "corrected_source_revision": corrected_source,
        "prior_checkpoint_sha256": sha256_file(original),
        "corrected_checkpoint_sha256": sha256_file(corrected),
        "model_state_sha256": state_sha,
        "completed_updates": 0,
        "foundation_model_calls": 0,
    }


def main_training_stage(step: int) -> str:
    """Fixed Phase 4 exposure schedule, indexed by the next optimizer update."""
    if type(step) is not int or not 1 <= step <= 12000:
        raise ValueError("main training update must be in [1, 12000]")
    if step <= 9000:
        return "one_hop"
    if step <= 10000:
        return "two_hop"
    if step <= 11000:
        return "primary"
    return "robustness"


def main_training_batch(config: ComparisonConfig, *, step: int) -> TrainingBatch:
    """Reuse Phase 4's exact train roots, batch size, and counter-addressed IDs."""
    if not isinstance(config, ComparisonConfig):
        raise TypeError("main training batch requires the fixed comparison config")
    stage = main_training_stage(step)
    training = config.phase4_config.pilot
    first_index = (step - 1) * training.batch_size
    examples = tuple(
        make_curriculum_example(
            config.phase4_config,
            CurriculumKey(
                CURRICULUM_VERSION,
                "train",
                training.train_root_seed,
                training.train_public_id_seed,
                first_index + offset,
                stage,
            ),
        )
        for offset in range(training.batch_size)
    )
    return pack_training_examples(examples, next_batch_counter=step)


def require_clean_main_source() -> str:
    """A learned trajectory is bound to committed implementation bytes."""
    root = Path(__file__).resolve().parents[3]
    execution_paths = (
        "src/silent_cascade/env",
        "src/silent_cascade/eventflow",
        "src/silent_cascade/models",
        "src/silent_cascade/train",
        "src/silent_cascade/eval",
        "scripts/run_phase5a.py",
        "configs/eval/phase5a.yaml",
    )
    revision = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", *execution_paths],
        cwd=root,
        text=True,
    ).strip()
    if len(revision) != 40 or subprocess.call(
        ["git", "diff", "--quiet", "HEAD", "--", *execution_paths], cwd=root
    ):
        raise ValueError("main training source has uncommitted implementation changes")
    if subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *execution_paths],
        cwd=root,
        text=True,
    ).strip():
        raise ValueError("main training source has uncommitted implementation changes")
    return revision


def project_ponder_batch(batch: TrainingBatch, *, caps: torch.Tensor) -> PonderBatch:
    """Align one transition to each oracle record interpretation, then null termination."""
    if not isinstance(batch, TrainingBatch):
        raise TypeError("ponder projection requires a TrainingBatch")
    if (
        not isinstance(caps, torch.Tensor)
        or caps.dtype != torch.int64
        or caps.shape != (len(batch.teacher_traces),)
        or bool(((caps < 1) | (caps > 24)).any())
    ):
        raise ValueError("ponder caps must be integer [B] values in [1,24]")
    sequences = []
    for trace in batch.teacher_traces:
        records = [step for step in trace.steps if step.kind is InternalEventKind.COMPOSE]
        if trace.terminal_status == 2:
            records.append(None)
        if not records:
            raise ValueError("teacher has no record interpretation or null termination")
        sequences.append(records)
    size = (len(sequences), int(caps.max()))
    slots = torch.full(size, -1, dtype=torch.int64)
    mask = torch.zeros(size, dtype=torch.bool)
    rollout_mask = torch.arange(size[1])[None, :] < caps[:, None]
    targets = {
        name: torch.full(size, -1, dtype=torch.int64)
        if name in _INTEGER
        else torch.zeros(size, dtype=torch.float32)
        for name in _ALL
    }
    validity = {name: torch.zeros(size, dtype=torch.bool) for name in _ALL}
    halt = torch.zeros(size, dtype=torch.bool)
    delays = torch.ones(size, dtype=torch.float32)
    for row, (trace, sequence) in enumerate(zip(batch.teacher_traces, sequences, strict=True)):
        record_ids = batch.public.records.record_ids[row]
        for col, step in enumerate(sequence):
            if col >= int(caps[row]):
                break
            mask[row, col] = True
            final = col == len(sequence) - 1
            halt[row, col] = final
            if step is None:
                slots[row, col] = 64
                labels = {"role": 4, "status": 2, "support": False}
            else:
                if step.selected_record_id not in record_ids:
                    raise ValueError("teacher-selected record is absent from public memory")
                slots[row, col] = record_ids.index(step.selected_record_id)
                labels = {
                    "role": step.role,
                    "focus": step.focus,
                    "hazard": step.hazard_type,
                    "log_delay": step.log_delay,
                    "deadline": math.expm1(step.log_delay) if step.log_delay is not None else None,
                    "status": step.status,
                    "confidence": step.confidence,
                    "support": step.append_support,
                    "continue": step.continue_search,
                }
            if final:
                labels["action_class"] = trace.terminal_class
                if trace.terminal_class != 4:
                    hazard = next(
                        (
                            candidate
                            for candidate in sequence
                            if candidate is not None and candidate.log_delay is not None
                        ),
                        None,
                    )
                    if hazard is None:
                        raise ValueError("positive oracle has no public hazard delay")
                    delays[row, col] = math.expm1(hazard.log_delay)
                    labels["action_offset"] = 0.825
            for name, value in labels.items():
                if value is None:
                    continue
                targets[name][row, col] = int(value) if name in _INTEGER else float(value)
                validity[name][row, col] = True
    return PonderBatch(
        batch.public, slots, mask, rollout_mask, caps.clone(), targets, validity, halt, delays
    )


def _masked_ce(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, int]:
    count = int(mask.sum())
    if not count:
        return logits.sum() * 0.0, 0
    return F.cross_entropy(logits[mask], target[mask]), count


def _masked_bce(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, int]:
    count = int(mask.sum())
    if not count:
        return logits.sum() * 0.0, 0
    return F.binary_cross_entropy_with_logits(logits[mask], target[mask]), count


def _masked_smooth(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, int]:
    count = int(mask.sum())
    if not count:
        return prediction.sum() * 0.0, 0
    return F.smooth_l1_loss(prediction[mask], target[mask]), count


def ponder_loss(model: ActivationPonderModel, batch: PonderBatch) -> PonderLoss:
    """Independent masks prevent absent content labels from inventing examples."""
    if not isinstance(model, ActivationPonderModel) or not isinstance(batch, PonderBatch):
        raise TypeError("ponder loss requires a model and projected batch")
    predicted = model.forward_teacher(batch.public, batch.teacher_slots, batch.step_mask)
    terms = {}
    counts = {}

    def save(name: str, reduced: tuple[torch.Tensor, int]) -> None:
        terms[name], counts[name] = reduced

    save("retrieval", _masked_ce(predicted.retrieval_logits, batch.teacher_slots, batch.step_mask))
    for name, head in (
        ("role", "role"),
        ("focus", "focus"),
        ("hazard", "hazard"),
        ("status", "status"),
        ("action_class", "action"),
    ):
        term = "action" if name == "action_class" else name
        save(term, _masked_ce(predicted.heads[head], batch.targets[name], batch.validity[name]))
    for name, head in (
        ("confidence", "confidence"),
        ("support", "support"),
        ("continue", "continue"),
    ):
        save(
            name,
            _masked_bce(
                predicted.heads[head].squeeze(-1), batch.targets[name], batch.validity[name]
            ),
        )
    for name, head in (("log_delay", "log_delay"), ("deadline", "deadline")):
        save(
            name,
            _masked_smooth(
                predicted.heads[head].squeeze(-1), batch.targets[name], batch.validity[name]
            ),
        )
    normalized_offset = F.softplus(predicted.heads["action_offset"].squeeze(-1)) / (
        batch.teacher_delay
    )
    save(
        "action_offset",
        _masked_smooth(
            normalized_offset,
            batch.targets["action_offset"],
            batch.validity["action_offset"],
        ),
    )
    save(
        "halt",
        _masked_bce(
            predicted.heads["halt"].squeeze(-1),
            batch.halt_target.float(),
            batch.step_mask,
        ),
    )
    hidden_energy = predicted.hidden.square().mean(dim=-1)
    terms["state_bound"] = hidden_energy[batch.step_mask].mean()
    counts["state_bound"] = int(batch.step_mask.sum())
    halt_probability = torch.sigmoid(predicted.heads["halt"].squeeze(-1))
    before = torch.cat(
        (torch.ones_like(halt_probability[:, :1]), 1.0 - halt_probability[:, :-1]), dim=1
    ).cumprod(dim=1)
    terms["event_cost"] = (before * batch.rollout_mask).sum(dim=1).mean()
    counts["event_cost"] = batch.rollout_mask.shape[0]

    def applicable_mean(names: tuple[str, ...]) -> torch.Tensor:
        relevant = [terms[name] for name in names if counts[name] > 0]
        return torch.stack(relevant).mean() if relevant else predicted.hidden.sum() * 0.0

    terms["compose_type"] = applicable_mean(("role", "status", "confidence", "support", "continue"))
    terms["deadline_group"] = applicable_mean(("log_delay", "deadline"))
    total = (
        terms["retrieval"]
        + terms["compose_type"]
        + 0.5 * terms["focus"]
        + terms["hazard"]
        + 0.25 * terms["deadline_group"]
        + terms["action"]
        + 0.25 * terms["action_offset"]
        + terms["halt"]
        + 0.05 * terms["state_bound"]
        + 0.001 * terms["event_cost"]
    )
    if not bool(torch.isfinite(total)):
        raise ValueError("nonfinite ponder loss")
    return PonderLoss(total=total, terms=terms, denominators=counts)


def run_debug_competence(*, config: ComparisonConfig, run_dir: Path, source_revision: str) -> dict:
    """One nonpromotable seed-11 fit on the fixed 64-example DEBUG set."""
    if not isinstance(config, ComparisonConfig) or len(source_revision) != 40:
        raise ValueError("competence fit requires the fixed config and committed source")
    examples, hashes = build_fixed_competence_set(config.phase4_config)
    manifest = {
        "schema_version": "phase5a-ponder-competence-data-v1",
        "root_seed": 7963,
        "public_id_seed": 7993,
        "counts": {"one_hop": 16, "two_hop": 16, "primary": 32},
        "example_hashes": list(hashes),
        "public_ids": [example.public.init.episode_public_id for example in examples],
        "source_revision": source_revision,
        "promotable": False,
    }
    dataset_sha256 = sha256_bytes(canonical_json_bytes(manifest))
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("existing competence dataset or source differs")
    else:
        atomic_create_bytes(manifest_path, manifest_bytes)
    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_bytes())
        if (
            result.get("dataset_sha256") != dataset_sha256
            or result.get("source_revision") != source_revision
            or sha256_file(run_dir / f"debug-{result.get('completed_update', -1):04d}.pt")
            != result.get("last_checkpoint_sha256")
        ):
            raise ValueError("existing competence result has incompatible identity")
        return result
    batch = pack_training_examples(examples)
    bundles = tuple(
        curriculum_to_bundle(example, config=config.phase4_config) for example in examples
    )
    torch.manual_seed(11)
    model = ActivationPonderModel(matched_ponder_config()).to("cpu")
    counts = parameter_counts(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        weight_decay=1.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-6,
        foreach=False,
        fused=False,
    )
    budget_path = run_dir / "budget.json"
    if budget_path.exists():
        budget = BudgetLedger.model_validate_json(budget_path.read_bytes())
        budget = budget.model_copy(update={"attempts": budget.attempts + 1})
    else:
        budget = BudgetLedger(attempts=1)
    checkpoints = sorted(run_dir.glob("debug-*.pt"))
    if checkpoints:
        latest = checkpoints[-1]
        start_step = load_debug_checkpoint(
            latest,
            model,
            optimizer,
            expected_dataset_sha256=dataset_sha256,
            expected_source_revision=source_revision,
        )
    else:
        start_step = 0
        latest = run_dir / "debug-0000.pt"
        save_debug_checkpoint(
            latest,
            model,
            optimizer,
            step=0,
            dataset_sha256=dataset_sha256,
            source_revision=source_revision,
        )

    def charge(elapsed: float, *, updates: int = 0) -> None:
        nonlocal budget
        retained = sum(
            path.stat().st_size
            for path in run_dir.rglob("*")
            if path.is_file() and path != budget_path
        )
        budget = budget.model_copy(
            update={
                "elapsed_scientific_seconds": budget.elapsed_scientific_seconds + elapsed,
                "retained_bytes": retained,
                "updates": budget.updates + updates,
            }
        )
        atomic_write_bytes(budget_path, canonical_json_bytes(budget) + b"\n")

    charge(0.0)
    best_successes = 0
    for path in sorted(run_dir.glob("eval-*.json")):
        evaluated = json.loads(path.read_bytes())
        best_successes = max(best_successes, evaluated["timed_successes"])

    def evaluate(step: int) -> tuple[int, Path]:
        nonlocal best_successes
        started = perf_counter()
        rows = evaluate_competence(model, bundles, cap=24)
        elapsed = perf_counter() - started
        successes = sum(row["timed_success"] for row in rows)
        if any(row["compute"]["foundation_model_calls"] != 0 for row in rows):
            raise ValueError("DEBUG competence evaluation violated the offline protocol")
        if any(len(row["steps"]) > 24 for row in rows):
            raise ValueError("competence rollout exceeded the fixed cap")
        payload = {
            "schema_version": "phase5a-ponder-competence-eval-v1",
            "step": step,
            "cap": 24,
            "dataset_sha256": dataset_sha256,
            "checkpoint_sha256": sha256_file(run_dir / f"debug-{step:04d}.pt"),
            "timed_successes": successes,
            "denominator": len(rows),
            "foundation_model_calls": sum(row["compute"]["foundation_model_calls"] for row in rows),
            "rows": rows,
        }
        path = run_dir / f"eval-{step:04d}.json"
        raw = canonical_json_bytes(payload) + b"\n"
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError("existing competence evaluation differs")
        else:
            atomic_create_bytes(path, raw)
        charge(elapsed)
        best_successes = max(best_successes, successes)
        print(f"competence update {step}: {successes}/64 autonomous successes", flush=True)
        return successes, path

    if (
        start_step
        and start_step % 50 == 0
        and not (run_dir / f"eval-{start_step:04d}.json").exists()
    ):
        evaluate(start_step)
    last_step = start_step
    last_successes = 0
    if start_step and (run_dir / f"eval-{start_step:04d}.json").exists():
        last_successes = json.loads((run_dir / f"eval-{start_step:04d}.json").read_bytes())[
            "timed_successes"
        ]
    for step in range(start_step + 1, 1001):
        if budget.updates >= 1000 or last_successes == 64:
            break
        free = shutil.disk_usage(run_dir).free
        if free < config.working_reserve_bytes:
            raise OSError("competence artifact volume lacks the required 2 GiB reserve")
        if budget.elapsed_scientific_seconds >= config.milestone_b_time_target_seconds and not any(
            "Milestone B time target" in note for note in budget.extensions
        ):
            note = (
                f"Milestone B time target extended before competence update {step}; "
                f"{1001 - step} fixed debug updates remain, {free} free bytes; "
                "complete the predeclared competence test while practical"
            )
            budget = budget.model_copy(update={"extensions": (*budget.extensions, note)})
            charge(0.0)
        started = perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        choices = torch.randint(0, len(PONDER_CAPS), (64,))
        caps = torch.tensor(PONDER_CAPS, dtype=torch.int64)[choices]
        projected = project_ponder_batch(batch, caps=caps)
        with NeuralComputeMeter(model) as meter:
            losses = ponder_loss(model, projected)
            losses.total.backward()
        snapshot = meter.snapshot()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_step = step
        charge(perf_counter() - started, updates=1)
        if step % 50 == 0:
            latest = run_dir / f"debug-{step:04d}.pt"
            save_debug_checkpoint(
                latest,
                model,
                optimizer,
                step=step,
                dataset_sha256=dataset_sha256,
                source_revision=source_revision,
            )
            last_successes, _ = evaluate(step)
            compute_path = run_dir / f"compute-{step:04d}.json"
            atomic_create_bytes(
                compute_path,
                canonical_json_bytes(
                    {
                        "step": step,
                        "last_loss": float(losses.total.detach()),
                        "last_update_forward_macs": snapshot.forward_macs,
                        "last_update_backward_macs": snapshot.backward_macs,
                        "last_update_gru_calls": snapshot.module_calls.get("GRUCell", 0),
                    }
                )
                + b"\n",
            )
            charge(0.0)
    if last_step != int(latest.stem.split("-")[-1]):
        latest = run_dir / f"debug-{last_step:04d}.pt"
        save_debug_checkpoint(
            latest,
            model,
            optimizer,
            step=last_step,
            dataset_sha256=dataset_sha256,
            source_revision=source_revision,
        )
        last_successes, _ = evaluate(last_step)
    status = "competent" if last_successes == 64 else "inconclusive_training"
    result = {
        "schema_version": "phase5a-ponder-competence-result-v1",
        "status": status,
        "dataset_sha256": dataset_sha256,
        "source_revision": source_revision,
        "scientific_seed": 11,
        "width": model.config.width,
        "parameters": counts["total"],
        "entity_parameters": counts["entity_table"],
        "completed_update": last_step,
        "physical_update_attempts": budget.updates,
        "best_autonomous_successes": best_successes,
        "last_autonomous_successes": last_successes,
        "autonomous_denominator": 64,
        "last_checkpoint_sha256": sha256_file(latest),
        "elapsed_scientific_seconds": budget.elapsed_scientific_seconds,
        "retained_bytes": budget.retained_bytes,
        "foundation_model_calls": 0,
        "promotable": False,
    }
    atomic_create_bytes(result_path, canonical_json_bytes(result) + b"\n")
    return result


def _write_json(path: Path, value: object) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def _main_update(
    model: ActivationPonderModel,
    optimizer: torch.optim.Optimizer,
    config: ComparisonConfig,
    step: int,
) -> tuple[float, float, float]:
    """One and only one counter-addressed train batch and optimizer update."""
    started = perf_counter()
    batch = main_training_batch(config, step=step)
    generated = perf_counter() - started
    model.train()
    optimizer.zero_grad(set_to_none=True)
    choices = torch.randint(0, len(PONDER_CAPS), (len(batch.teacher_traces),))
    caps = torch.tensor(PONDER_CAPS, dtype=torch.int64)[choices]
    projected = project_ponder_batch(batch, caps=caps)
    loss = ponder_loss(model, projected).total
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return generated, perf_counter() - started - generated, float(loss.detach())


def _primary_manifest(config: ComparisonConfig) -> PilotManifest:
    path = Path(__file__).resolve().parents[3] / "manifests/validation/phase4/primary.json"
    manifest = PilotManifest.model_validate_json(path.read_bytes())
    if manifest.stage != "primary" or manifest.split != "validation" or manifest.count != 10000:
        raise ValueError("accepted primary validation manifest differs")
    if manifest.config_hash != sha256_bytes(manifest.config_canonical_json.encode()):
        raise ValueError("primary validation config hash differs")
    if manifest.config_canonical_json != canonical_json_bytes(config.phase4_config).decode():
        raise ValueError("primary validation config differs from the fixed Phase 4 recipe")
    return manifest


def _profile_main(
    model: ActivationPonderModel,
    optimizer: torch.optim.Optimizer,
    config: ComparisonConfig,
    manifest: PilotManifest,
    run_dir: Path,
    first_100_costs: list[tuple[float, float]],
    checkpoint_seconds: float,
) -> CostProfile:
    """Measure timing rows and disposable stage updates while preserving the trajectory."""
    rng_state = torch.get_rng_state().clone()
    original_model = copy.deepcopy(model.state_dict())
    original_optimizer = copy.deepcopy(optimizer.state_dict())
    stage_costs = {"one_hop": sum(sum(pair) for pair in first_100_costs) / len(first_100_costs)}
    for stage, first_step in (("two_hop", 9001), ("primary", 10001), ("robustness", 11001)):
        disposable = ActivationPonderModel(matched_ponder_config()).to("cpu")
        disposable.load_state_dict(original_model)
        trial_optimizer = torch.optim.AdamW(
            disposable.parameters(),
            lr=3.0e-4,
            weight_decay=1.0e-4,
            betas=(0.9, 0.999),
            eps=1.0e-6,
            foreach=False,
            fused=False,
        )
        trial_optimizer.load_state_dict(original_optimizer)
        samples = []
        for offset in range(3):
            generated, trained, _ = _main_update(
                disposable, trial_optimizer, config, first_step + offset
            )
            samples.append(generated + trained)
        stage_costs[stage] = max(samples)
    torch.set_rng_state(rng_state)
    assert all(
        torch.equal(model.state_dict()[name], tensor) for name, tensor in original_model.items()
    )
    sample_path = run_dir / "profile-timing-64.json"
    if sample_path.exists():
        sample = json.loads(sample_path.read_bytes())
        rows = sample["rows"]
        seconds = sample["elapsed_seconds"]
    else:
        started = perf_counter()
        rows = []
        model.eval()
        for entry in manifest.entries[:64]:
            example = make_curriculum_example(config.phase4_config, entry.key)
            bundle = curriculum_to_bundle(example, config=config.phase4_config)
            if episode_sha256(bundle) != entry.projected.episode_sha256:
                raise ValueError("primary timing row differs from accepted manifest")
            rows.extend(evaluate_competence(model, (bundle,), cap=24))
        seconds = perf_counter() - started
        atomic_create_bytes(
            sample_path,
            canonical_json_bytes(
                {
                    "schema_version": "phase5a-ponder-profile-timing-v1",
                    "cap": 24,
                    "elapsed_seconds": seconds,
                    "rows": rows,
                }
            )
            + b"\n",
        )
    if len(rows) != 64 or any(row["compute"]["foundation_model_calls"] for row in rows):
        raise ValueError("fixed profile timing rows are incomplete or online")
    row_bytes = max(len(canonical_json_bytes(row)) for row in rows)
    return CostProfile(
        update_seconds_by_stage=stage_costs,
        validation_episode_seconds=seconds / 64,
        final_episode_seconds=seconds / 64,
        replay_episode_seconds=seconds / 64,
        checkpoint_seconds=checkpoint_seconds,
        report_seconds=5.0,
        retained_bytes_per_update_boundary=sum(
            path.stat().st_size for path in run_dir.glob("main-0100.*")
        ),
        retained_bytes_per_validation_episode=row_bytes,
        retained_bytes_per_final_episode=row_bytes,
    )


def _evaluate_primary_boundary(
    model: ActivationPonderModel,
    config: ComparisonConfig,
    manifest: PilotManifest,
    *,
    step: int,
    checkpoint_sha256: str,
    run_dir: Path,
) -> tuple[dict, float]:
    """Stream all 10,000 accepted primary rows, preserving every adverse outcome."""
    destination = run_dir / f"validation-{step:04d}"
    receipt_path = destination / "receipt.json"
    manifest_sha = sha256_file(
        Path(__file__).resolve().parents[3] / "manifests/validation/phase4/primary.json"
    )
    if receipt_path.exists():
        return validate_primary_validation_receipt(
            receipt_path,
            entries=manifest.entries,
            update=step,
            checkpoint_sha256=checkpoint_sha256,
            manifest_sha256=manifest_sha,
        ), 0.0
    destination.mkdir(exist_ok=True)
    temporary = destination / "rows.jsonl.gz.tmp"
    successes = 0
    false_actions = 0
    errors = 0
    started = perf_counter()
    model.eval()
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                for index, entry in enumerate(manifest.entries):
                    example = make_curriculum_example(config.phase4_config, entry.key)
                    bundle = curriculum_to_bundle(example, config=config.phase4_config)
                    if episode_sha256(bundle) != entry.projected.episode_sha256:
                        raise ValueError(f"primary validation row {index} differs from manifest")
                    row = evaluate_competence(model, (bundle,), cap=24)[0]
                    successes += int(row["timed_success"])
                    false_actions += int(row["variant"] != "positive" and bool(row["actions"]))
                    errors += int(row["compute"]["foundation_model_calls"] != 0)
                    compressed.write(canonical_json_bytes(row) + b"\n")
                    if (index + 1) % 1000 == 0:
                        print(f"primary validation {step}: {index + 1}/10000", flush=True)
            raw.flush()
            import os

            os.fsync(raw.fileno())
        temporary.replace(destination / "rows.jsonl.gz")
    finally:
        temporary.unlink(missing_ok=True)
    elapsed = perf_counter() - started
    if errors:
        raise ValueError("primary validation recorded a foundation-model call")
    receipt = {
        "schema_version": "phase5a-ponder-primary-validation-v1",
        "update": step,
        "cap": 24,
        "denominator": 10000,
        "timed_successes": successes,
        "negative_false_actions": false_actions,
        "foundation_model_calls": 0,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha,
        "rows_sha256": sha256_file(destination / "rows.jsonl.gz"),
        "elapsed_seconds": elapsed,
    }
    atomic_create_bytes(receipt_path, canonical_json_bytes(receipt) + b"\n")
    return validate_primary_validation_receipt(
        receipt_path,
        entries=manifest.entries,
        update=step,
        checkpoint_sha256=checkpoint_sha256,
        manifest_sha256=manifest_sha,
    ), elapsed


def validate_primary_validation_receipt(
    receipt_path: Path,
    *,
    entries: Sequence,
    update: int,
    checkpoint_sha256: str,
    manifest_sha256: str,
) -> dict:
    """Recount every bound validation row before using its checkpoint rank."""
    receipt = json.loads(receipt_path.read_bytes())
    rows_path = receipt_path.parent / "rows.jsonl.gz"
    if (
        receipt.get("schema_version") != "phase5a-ponder-primary-validation-v1"
        or receipt.get("update") != update
        or receipt.get("cap") != 24
        or receipt.get("denominator") != len(entries)
        or receipt.get("checkpoint_sha256") != checkpoint_sha256
        or receipt.get("manifest_sha256") != manifest_sha256
        or receipt.get("rows_sha256") != sha256_file(rows_path)
        or receipt.get("foundation_model_calls") != 0
    ):
        raise ValueError("primary validation receipt binding differs")
    successes = false_actions = 0
    with gzip.open(rows_path, "rt") as handle:
        for index, (line, entry) in enumerate(zip(handle, entries, strict=True)):
            row = json.loads(line)
            projected = entry.projected
            if (
                row.get("public_id") != projected.public_id
                or row.get("episode_sha256") != projected.episode_sha256
                or row.get("variant") != projected.variant
                or row.get("path_length") != projected.path_length
                or row.get("compute", {}).get("foundation_model_calls") != 0
                or type(row.get("timed_success")) is not bool
                or (row.get("score_reason") in {"success", "negative_abstention"})
                != row["timed_success"]
                or (
                    projected.variant == "positive"
                    and row.get("score_reason") not in {"success", "incorrect", "action_count"}
                )
                or (
                    projected.variant != "positive"
                    and row.get("score_reason")
                    not in {"negative_abstention", "negative_false_action"}
                )
                or not isinstance(row.get("actions"), list)
            ):
                raise ValueError(f"primary validation row {index} binding differs")
            successes += int(row["timed_success"])
            false_actions += int(projected.variant != "positive" and bool(row["actions"]))
    if (
        receipt.get("timed_successes") != successes
        or receipt.get("negative_false_actions") != false_actions
    ):
        raise ValueError("primary validation receipt count differs from bound rows")
    return receipt


def recover_orphan_main_boundary(
    main_dir: Path,
    *,
    latest_step: int,
    budget: BudgetLedger,
    model: ActivationPonderModel,
    optimizer: torch.optim.Optimizer,
    config_sha256: str,
    source_revision: str,
) -> tuple[int, Path] | None:
    """Restore a complete immutable boundary written before its progress pointer."""
    future = sorted(
        path
        for path in main_dir.glob("main-[0-9][0-9][0-9][0-9].pt")
        if int(path.stem.split("-")[1]) > latest_step
    )
    if not future:
        return None
    expected = 1000 if latest_step < 1000 else latest_step + 1000
    if len(future) != 1 or future[0].name != f"main-{expected:04d}.pt":
        raise ValueError("orphaned main checkpoint is not the next validation boundary")
    path = future[0]
    if not path.with_suffix(".json").is_file():
        raise ValueError("orphaned main checkpoint lacks its immutable descriptor")
    step, checkpoint_budget = load_main_checkpoint(
        path,
        model,
        optimizer,
        config_sha256=config_sha256,
        source_revision=source_revision,
    )
    if (
        step != expected
        or checkpoint_budget.updates != expected
        or budget.updates != expected
        or budget.elapsed_scientific_seconds < checkpoint_budget.elapsed_scientific_seconds
        or budget.retained_bytes < checkpoint_budget.retained_bytes
    ):
        raise ValueError("orphaned main checkpoint differs from physical progress")
    return step, path


def fit_ponder(
    *,
    config: ComparisonConfig,
    run_dir: Path,
    budget: BudgetLedger,
    resume: Path | None = None,
) -> PonderTrainingResult:
    """Run one fresh seed-11 trajectory in durable, complete update boundaries."""
    if not isinstance(config, ComparisonConfig) or not isinstance(budget, BudgetLedger):
        raise TypeError("ponder fit requires the frozen comparison config and ledger")
    source = require_clean_main_source()
    main_dir = run_dir / "ponder"
    main_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(main_dir).free < config.working_reserve_bytes:
        raise OSError("ponder artifact volume lacks the required 2 GiB reserve")
    model = ActivationPonderModel(matched_ponder_config()).to("cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3.0e-4,
        weight_decay=1.0e-4,
        betas=(0.9, 0.999),
        eps=1.0e-6,
        foreach=False,
        fused=False,
    )
    budget_path = main_dir / "budget.json"
    result_path = main_dir / "result.json"
    base_retained = budget.retained_bytes

    def main_bytes() -> int:
        return sum(
            path.stat().st_size
            for path in main_dir.rglob("*")
            if path.is_file() and path != budget_path
        )

    def save_budget() -> None:
        nonlocal budget
        budget = budget.model_copy(update={"retained_bytes": base_retained + main_bytes()})
        _write_json(budget_path, budget)

    if result_path.exists():
        result = PonderTrainingResult.model_validate_json(result_path.read_bytes())
        if result.config_sha256 != config.config_sha256:
            raise ValueError("existing ponder trajectory has incompatible config")
        if result.source_revision != source:
            recorded = BudgetLedger.model_validate_json(budget_path.read_bytes())
            if (
                result.completed_updates != 0
                or recorded.updates != 0
                or result.latest_checkpoint != "main-0000.pt"
            ):
                raise ValueError("learned ponder trajectory cannot change source revision")
            correction = main_dir / "main-0000-rebound.pt"
            receipt_path = main_dir / "zero-update-source-correction.json"
            if correction.exists() or receipt_path.exists():
                if not correction.exists() or not receipt_path.exists():
                    raise ValueError("incomplete zero-update source correction")
                receipt = json.loads(receipt_path.read_bytes())
                if (
                    receipt.get("prior_source_revision") != result.source_revision
                    or receipt.get("corrected_source_revision") != source
                    or receipt.get("prior_checkpoint_sha256") != result.latest_checkpoint_sha256
                    or receipt.get("corrected_checkpoint_sha256") != sha256_file(correction)
                ):
                    raise ValueError("existing zero-update correction differs")
            else:
                receipt = rebind_zero_update_checkpoint(
                    main_dir / "main-0000.pt",
                    correction,
                    model,
                    optimizer,
                    config_sha256=config.config_sha256,
                    prior_source=result.source_revision,
                    corrected_source=source,
                )
                atomic_create_bytes(receipt_path, canonical_json_bytes(receipt) + b"\n")
            result = result.model_copy(
                update={
                    "source_revision": source,
                    "latest_checkpoint": correction.name,
                    "latest_checkpoint_sha256": sha256_file(correction),
                }
            )
            _write_json(result_path, result)
        if result.latest_checkpoint is None:
            raise ValueError("existing ponder trajectory lacks a checkpoint")
        checkpoint = resume or main_dir / result.latest_checkpoint
        step, checkpoint_budget = load_main_checkpoint(
            checkpoint,
            model,
            optimizer,
            config_sha256=config.config_sha256,
            source_revision=source,
        )
        if (
            sha256_file(checkpoint) != result.latest_checkpoint_sha256
            or step != result.completed_updates
        ):
            raise ValueError("latest ponder checkpoint differs from recorded progress")
        retained = BudgetLedger.model_validate_json(budget_path.read_bytes())
        if (
            retained.elapsed_scientific_seconds < checkpoint_budget.elapsed_scientific_seconds
            or retained.updates < checkpoint_budget.updates
            or retained.retained_bytes < checkpoint_budget.retained_bytes
        ):
            raise ValueError("resumed ponder ledger lost consumed resources")
        budget = retained.model_copy(update={"attempts": retained.attempts + 1})
        base_retained = max(0, budget.retained_bytes - main_bytes())
        recovered = main_dir / "main-0100.pt"
        progress_file = main_dir / "profile-progress.json"
        if step == 0 and recovered.is_file() and progress_file.is_file():
            progress = json.loads(progress_file.read_bytes())
            if progress.get("checkpoint_sha256") != sha256_file(recovered):
                raise ValueError("orphaned profile checkpoint differs from progress")
            step, _ = load_main_checkpoint(
                recovered,
                model,
                optimizer,
                config_sha256=config.config_sha256,
                source_revision=source,
            )
            if step != 100:
                raise ValueError("orphaned profile checkpoint has wrong update")
            result = result.model_copy(
                update={
                    "completed_updates": 100,
                    "latest_checkpoint": recovered.name,
                    "latest_checkpoint_sha256": sha256_file(recovered),
                }
            )
            _write_json(result_path, result)
        orphan = recover_orphan_main_boundary(
            main_dir,
            latest_step=step,
            budget=budget,
            model=model,
            optimizer=optimizer,
            config_sha256=config.config_sha256,
            source_revision=source,
        )
        if orphan is not None:
            step, checkpoint = orphan
            result = result.model_copy(
                update={
                    "completed_updates": step,
                    "last_stage": main_training_stage(step),
                    "latest_checkpoint": checkpoint.name,
                    "latest_checkpoint_sha256": sha256_file(checkpoint),
                }
            )
            _write_json(result_path, result)
            save_budget()
    else:
        if resume is not None or budget.updates != 0:
            raise ValueError("fresh ponder trajectory cannot resume or inherit main updates")
        torch.manual_seed(11)
        model = ActivationPonderModel(matched_ponder_config()).to("cpu")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3.0e-4,
            weight_decay=1.0e-4,
            betas=(0.9, 0.999),
            eps=1.0e-6,
            foreach=False,
            fused=False,
        )
        budget = budget.model_copy(update={"attempts": budget.attempts + 1})
        checkpoint = main_dir / "main-0000.pt"
        save_main_checkpoint(
            checkpoint,
            model,
            optimizer,
            step=0,
            budget=budget,
            config_sha256=config.config_sha256,
            source_revision=source,
        )
        result = PonderTrainingResult(
            status="profiling",
            source_revision=source,
            config_sha256=config.config_sha256,
            completed_updates=0,
            last_stage="one_hop",
            latest_checkpoint=checkpoint.name,
            latest_checkpoint_sha256=sha256_file(checkpoint),
            parameters=parameter_counts(model)["total"],
            entity_parameters=parameter_counts(model)["entity_table"],
        )
        _write_json(result_path, result)
        save_budget()
        step = 0
    manifest_started = perf_counter()
    manifest = _primary_manifest(config)
    budget = budget.model_copy(
        update={
            "elapsed_scientific_seconds": budget.elapsed_scientific_seconds
            + perf_counter()
            - manifest_started,
        }
    )
    save_budget()
    profile_path = main_dir / "profile.json"
    admission_path = main_dir / "admission.json"
    profile_progress_path = main_dir / "profile-progress.json"

    def advance_one(update: int) -> tuple[float, float]:
        nonlocal budget
        if budget.updates >= config.main_update_ceiling:
            raise ValueError("physical main-update ceiling reached")
        if shutil.disk_usage(main_dir).free < config.working_reserve_bytes:
            raise OSError("ponder artifact reserve exhausted before update")
        generated, trained, loss = _main_update(model, optimizer, config, update)
        budget = budget.model_copy(
            update={
                "updates": budget.updates + 1,
                "elapsed_scientific_seconds": budget.elapsed_scientific_seconds
                + generated
                + trained,
            }
        )
        _write_json(budget_path, budget)
        if update % 50 == 0 or update == 100:
            print(f"ponder update {update}: loss={loss:.4f}", flush=True)
        return generated, trained

    if step < 100:
        costs = []
        for update in range(step + 1, 101):
            costs.append(advance_one(update))
        started = perf_counter()
        checkpoint = main_dir / "main-0100.pt"
        save_main_checkpoint(
            checkpoint,
            model,
            optimizer,
            step=100,
            budget=budget,
            config_sha256=config.config_sha256,
            source_revision=source,
        )
        checkpoint_seconds = perf_counter() - started
        budget = budget.model_copy(
            update={
                "elapsed_scientific_seconds": budget.elapsed_scientific_seconds + checkpoint_seconds
            }
        )
        save_budget()
        result = result.model_copy(
            update={
                "completed_updates": 100,
                "latest_checkpoint": checkpoint.name,
                "latest_checkpoint_sha256": sha256_file(checkpoint),
            }
        )
        _write_json(
            profile_progress_path,
            {
                "schema_version": "phase5a-ponder-profile-progress-v1",
                "checkpoint_sha256": sha256_file(checkpoint),
                "first_100_costs": costs,
                "checkpoint_seconds": checkpoint_seconds,
            },
        )
        _write_json(result_path, result)
        step = 100

    if step == 100 and not admission_path.exists():
        progress = json.loads(profile_progress_path.read_bytes())
        if (
            progress.get("schema_version") != "phase5a-ponder-profile-progress-v1"
            or progress.get("checkpoint_sha256") != result.latest_checkpoint_sha256
            or len(progress.get("first_100_costs", [])) != 100
        ):
            raise ValueError("first 100 counted updates lack complete profile costs")
        profiled = perf_counter()
        if profile_path.exists():
            profile = CostProfile.model_validate_json(profile_path.read_bytes())
        else:
            profile = _profile_main(
                model,
                optimizer,
                config,
                manifest,
                main_dir,
                [tuple(pair) for pair in progress["first_100_costs"]],
                progress["checkpoint_seconds"],
            )
            budget = budget.model_copy(
                update={
                    "elapsed_scientific_seconds": budget.elapsed_scientific_seconds
                    + perf_counter()
                    - profiled,
                }
            )
            atomic_create_bytes(profile_path, canonical_json_bytes(profile) + b"\n")
            save_budget()
        admission = admit_ponder_budget(profile, budget)
        if admission.chosen_updates is None:
            free = shutil.disk_usage(main_dir).free
            practical = [
                n
                for n, estimate in admission.estimates.items()
                if estimate.projected_retained_bytes - budget.retained_bytes
                < free - config.working_reserve_bytes
            ]
            if practical:
                selected = max(practical)
                note = (
                    f"Milestone B time target extended at profile update 100 to {selected} "
                    f"fixed updates; projected 2x total "
                    f"{admission.estimates[selected].projected_seconds:.1f}s, "
                    f"free {free} bytes; decision uses throughput and storage only"
                )
                budget = budget.model_copy(update={"extensions": (*budget.extensions, note)})
                admission = admission.model_copy(
                    update={
                        "status": "admitted",
                        "chosen_updates": selected,
                        "rationale": note,
                    }
                )
            else:
                admission = admission.model_copy(
                    update={
                        "status": "budget_insufficient",
                        "rationale": "2x retention projection exceeds working reserve",
                    }
                )
        atomic_create_bytes(admission_path, canonical_json_bytes(admission) + b"\n")
        save_budget()
        result = result.model_copy(
            update={
                "status": "profiling" if admission.chosen_updates else "inconclusive_budget",
                "profile_sha256": sha256_file(profile_path),
                "admission_sha256": sha256_file(admission_path),
                "chosen_updates": admission.chosen_updates,
            }
        )
        _write_json(result_path, result)
        print(
            f"ponder admission: N={admission.chosen_updates}, "
            f"profiled 100 counted updates; status={admission.status}",
            flush=True,
        )
        return result

    profile = CostProfile.model_validate_json(profile_path.read_bytes())
    admission = admit_ponder_budget(profile, budget)
    # The published admission is immutable; later actual costs cannot reselect N.
    from silent_cascade.eval.comparison_types import TrainingAdmission

    frozen_admission = TrainingAdmission.model_validate_json(admission_path.read_bytes())
    if (
        sha256_file(profile_path) != result.profile_sha256
        or sha256_file(admission_path) != result.admission_sha256
        or frozen_admission.chosen_updates != result.chosen_updates
    ):
        raise ValueError("published ponder profile or admission differs")
    del admission
    if result.chosen_updates is None:
        return result
    target = result.chosen_updates

    def finish_validation(boundary: int) -> None:
        nonlocal budget, result
        checkpoint_path = main_dir / f"main-{boundary:04d}.pt"
        checkpoint_sha = sha256_file(checkpoint_path)
        started = perf_counter()
        receipt, measured = _evaluate_primary_boundary(
            model,
            config,
            manifest,
            step=boundary,
            checkpoint_sha256=checkpoint_sha,
            run_dir=main_dir,
        )
        if measured:
            budget = budget.model_copy(
                update={"elapsed_scientific_seconds": budget.elapsed_scientific_seconds + measured}
            )
            save_budget()
        rank = ponder_checkpoint_rank(
            update=boundary,
            cap=receipt["cap"],
            primary_successes=receipt["timed_successes"],
            negative_false_actions=receipt["negative_false_actions"],
        )
        if result.selected_rank is None or rank > result.selected_rank:
            result = result.model_copy(
                update={
                    "selected_checkpoint": checkpoint_path.name,
                    "selected_checkpoint_sha256": checkpoint_sha,
                    "selected_update": boundary,
                    "selected_rank": rank,
                }
            )
        result = result.model_copy(
            update={
                "validation_boundaries": tuple(
                    sorted(set((*result.validation_boundaries, boundary)))
                ),
                "status": "running",
            }
        )
        _write_json(result_path, result)
        if measured:
            print(
                f"primary validation {boundary}: {receipt['timed_successes']}/10000, "
                f"false actions={receipt['negative_false_actions']}, "
                f"elapsed={perf_counter() - started:.1f}s",
                flush=True,
            )

    if step % 1000 == 0 and step not in result.validation_boundaries:
        finish_validation(step)
    for update in range(step + 1, target + 1):
        if budget.updates >= config.main_update_ceiling:
            result = result.model_copy(update={"status": "inconclusive_budget"})
            _write_json(result_path, result)
            return result
        if update % 1000 == 1:
            free = shutil.disk_usage(main_dir).free
            if free < config.working_reserve_bytes:
                result = result.model_copy(update={"status": "inconclusive_budget"})
                _write_json(result_path, result)
                return result
            notes = list(budget.extensions)
            if (
                budget.elapsed_scientific_seconds >= config.milestone_b_time_target_seconds
                and not any("Milestone B actual time target" in note for note in notes)
            ):
                notes.append(
                    f"Milestone B actual time target extended before update {update}; "
                    f"{target - update + 1} fixed updates and remaining complete validations, "
                    f"free {free} bytes; finish admitted trajectory while practical"
                )
            if budget.retained_bytes >= config.retained_artifact_target_bytes and not any(
                "Milestone B actual storage target" in note for note in notes
            ):
                notes.append(
                    f"Milestone B actual storage target extended before update {update}; "
                    f"free {free} bytes with 2 GiB reserve"
                )
            if tuple(notes) != budget.extensions:
                budget = budget.model_copy(update={"extensions": tuple(notes)})
                save_budget()
        advance_one(update)
        if update % 1000 == 0:
            started = perf_counter()
            checkpoint = main_dir / f"main-{update:04d}.pt"
            save_main_checkpoint(
                checkpoint,
                model,
                optimizer,
                step=update,
                budget=budget,
                config_sha256=config.config_sha256,
                source_revision=source,
            )
            budget = budget.model_copy(
                update={
                    "elapsed_scientific_seconds": budget.elapsed_scientific_seconds
                    + perf_counter()
                    - started,
                }
            )
            save_budget()
            result = result.model_copy(
                update={
                    "completed_updates": update,
                    "last_stage": main_training_stage(update),
                    "latest_checkpoint": checkpoint.name,
                    "latest_checkpoint_sha256": sha256_file(checkpoint),
                }
            )
            _write_json(result_path, result)
            finish_validation(update)
    return result
