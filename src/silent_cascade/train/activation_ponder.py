"""Teacher-forced content projection and masked objective for Phase 5A pondering."""

import io
import json
import math
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from silent_cascade.env.episode import EpisodeBundle, episode_sha256
from silent_cascade.env.pilot import curriculum_to_bundle
from silent_cascade.env.reward import score_actions
from silent_cascade.eval.comparison_runner import PublicProposal, arbitrate_proposal
from silent_cascade.eval.comparison_types import BudgetLedger, ComparisonConfig
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
