"""Immutable evaluation identities and atomic publication of streamed evidence."""

import json
import os
from collections import Counter
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from silent_cascade.env.episode import EpisodeArtifact, EpisodeBundle, episode_sha256
from silent_cascade.eval.metrics import (
    Hash,
    PilotMetrics,
    Purpose,
    Revision,
    TimedEpisodeRow,
    summarize_timed_rows,
)
from silent_cascade.eventflow.archive_io import archive_parent, read_archive_at
from silent_cascade.eventflow.neural import NeuralModelIdentity
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.io import atomic_create_bytes
from silent_cascade.logging.crash_bundle import CrashBundleManifest
from silent_cascade.train.pilot_config import parse_phase4_canonical
from silent_cascade.validation import StrictModel


class DelayTransform(StrictModel):
    parent_public_id: str
    parent_episode_sha256: Hash
    version: str = Field(min_length=1)
    parameters_canonical_json: str

    @model_validator(mode="after")
    def validate_parameters(self):
        import json

        values = json.loads(self.parameters_canonical_json)
        if (
            not isinstance(values, dict)
            or canonical_json_bytes(values).decode() != self.parameters_canonical_json
        ):
            raise ValueError("transform parameters must be a canonical object")
        return self


class EpisodeBinding(StrictModel):
    public_id: str = Field(min_length=1)
    episode_sha256: Hash
    scoring_truth_sha256: Hash
    variant: Literal["positive", "safe_negative", "disconnected_negative"]
    path_length: int = Field(ge=1, le=8)
    generator_version: Literal["ofd-v1"] = "ofd-v1"
    transform: DelayTransform | None = None

    @classmethod
    def from_bundle(cls, bundle: EpisodeBundle, *, transform=None):
        return cls(
            public_id=bundle.public.init.episode_public_id,
            episode_sha256=episode_sha256(bundle),
            scoring_truth_sha256=sha256_bytes(
                canonical_json_bytes(
                    EpisodeArtifact.from_bundle(bundle).model_dump(mode="json")["truth"]
                )
            ),
            variant=bundle.truth.recipe.variant.value,
            path_length=bundle.truth.recipe.requested_path_length,
            transform=transform,
        )


class EvaluationIdentity(StrictModel):
    experiment: str = Field(min_length=1, max_length=128)
    stage: Literal["one_hop", "two_hop", "primary", "robustness"]
    split: Literal["validation", "debug"]
    purpose: Purpose
    manifest_schema: Literal["phase4-data-v1", "phase3-component-manifest-v1"]
    manifest_sha256: Hash
    episodes: tuple[EpisodeBinding, ...] = Field(min_length=1)
    checkpoint_sha256: Hash
    model_identity: NeuralModelIdentity
    evaluation_config_canonical_json: str
    execution_source_revision: Revision
    parent_manifest_sha256: Hash | None = None
    full_trace_public_ids: tuple[str, ...] = Field(default=(), max_length=500)
    report_example_public_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_identity(self):
        config = parse_phase4_canonical(self.evaluation_config_canonical_json)
        if canonical_json_bytes(config.neural).decode() != self.model_identity.model_config_json:
            raise ValueError("evaluation and neural configurations differ")
        ids = [e.public_id for e in self.episodes]
        hashes = [e.episode_sha256 for e in self.episodes]
        if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
            raise ValueError("duplicate evaluation inventory")
        if (
            self.manifest_schema == "phase3-component-manifest-v1"
            and self.purpose != "action_diagnostic"
        ):
            raise ValueError("historical manifest requires explicit action diagnostic purpose")
        if self.purpose == "debug" and self.split != "debug":
            raise ValueError("debug purpose requires debug split")
        if self.purpose == "pilot_validation":
            counts = Counter(e.variant for e in self.episodes)
            if (
                self.split != "validation"
                or not config.pilot.is_production
                or counts
                != {
                    "positive": 5000,
                    "safe_negative": 2500,
                    "disconnected_negative": 2500,
                }
            ):
                raise ValueError("pilot validation requires the full balanced validation inventory")
        if self.purpose == "delay_swap":
            if (
                self.parent_manifest_sha256 is None
                or self.stage != "primary"
                or any(e.transform is None for e in self.episodes)
            ):
                raise ValueError(
                    "delay swaps require primary parent manifest and transform bindings"
                )
            parents = [
                (e.transform.parent_public_id, e.transform.parent_episode_sha256)
                for e in self.episodes
            ]
            if len(set(parents)) != len(parents):
                raise ValueError("duplicate parent inventory")
        elif self.parent_manifest_sha256 is not None or any(e.transform for e in self.episodes):
            raise ValueError("parent transforms require delay_swap purpose")
        for selected in (self.full_trace_public_ids, self.report_example_public_ids):
            if len(set(selected)) != len(selected) or not set(selected) <= set(ids):
                raise ValueError("trace retention IDs must be unique inventory members")
        return self

    @cached_property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self))

    @property
    def gate_eligible(self) -> bool:
        return self.purpose == "pilot_validation" and self.stage == "primary"

    def retained_public_ids(self) -> frozenset[str]:
        """Predeclare a deterministic round-robin sample across variant/length strata."""
        if self.full_trace_public_ids:
            return frozenset(self.full_trace_public_ids)
        strata = {}
        for episode in self.episodes:
            strata.setdefault((episode.variant, episode.path_length), []).append(episode.public_id)
        selected = []
        for offset in range(max(map(len, strata.values()))):
            for key in sorted(strata):
                if offset < len(strata[key]):
                    selected.append(strata[key][offset])
                    if len(selected) == 500:
                        return frozenset(selected)
        return frozenset(selected)


class PilotEvaluation(StrictModel):
    identity: EvaluationIdentity
    metrics: PilotMetrics
    output_path: Path
    artifact_hashes: tuple[tuple[str, Hash], ...]


def read_evaluation_artifact(path: Path, *, max_bytes: int = 128 * 1024 * 1024) -> bytes:
    with archive_parent(path, error_factory=ValueError) as (parent, name):
        return read_archive_at(parent, name, max_bytes=max_bytes, error_factory=ValueError)


def _crash_index(root, rows):
    failed = {row.public_id for row in rows if row.error is not None}
    entries = {}
    for path in sorted((root / "crashes").glob("*.json")):
        raw = read_evaluation_artifact(path)
        manifest = CrashBundleManifest.model_validate_json(raw)
        public_id = manifest.context.episode_public_id
        if public_id not in failed or public_id in entries:
            raise ValueError("unexpected or duplicate crash evidence")
        entries[public_id] = {"path": str(path.relative_to(root)), "sha256": sha256_bytes(raw)}
    if set(entries) != failed:
        raise ValueError("missing crash evidence")
    if entries:
        atomic_create_bytes(root / "crashes/index.json", canonical_json_bytes(entries))


def _verify_evidence(root, row, *, retain):
    from silent_cascade.logging.neural_trace import (
        NeuralEventObservation,
        validate_full_neural_trace,
    )

    verified_bytes = {}
    for reference, digest in (
        (row.neural_trace_ref, row.neural_trace_sha256),
        (row.full_trace_ref, row.full_trace_sha256),
    ):
        if (reference is None) != (digest is None):
            raise ValueError("incomplete trace reference")
        if reference is not None:
            if Path(reference).is_absolute() or ".." in Path(reference).parts:
                raise ValueError("unsafe trace reference")
            if reference not in verified_bytes:
                verified_bytes[reference] = read_evaluation_artifact(root / reference)
            if sha256_bytes(verified_bytes[reference]) != digest:
                raise ValueError("trace integrity mismatch")
    if row.neural_trace_ref is None or (
        (retain or not row.timed_success) and row.full_trace_ref is None
    ):
        raise ValueError("missing required runtime evidence")
    sidecar = json.loads(verified_bytes[row.neural_trace_ref])
    if (sidecar["identity_sha256"], sidecar["episode_sha256"], sidecar["schema"]) != (
        row.identity_sha256,
        row.episode_sha256,
        "phase4-neural-observations-v1",
    ):
        raise ValueError("neural evidence identity mismatch")
    events = sidecar["causal_events"]
    observed = [
        NeuralEventObservation.model_validate_json(json.dumps(event)) for event in sidecar["events"]
    ]
    if len(events) != row.event_count or len(observed) != len(events):
        raise ValueError("neural evidence event count mismatch")
    expected_hash = sha256_bytes(canonical_json_bytes({"schema_version": 1, "events": events}))
    if row.causal_trace_sha256 is not None and row.causal_trace_sha256 != expected_hash:
        raise ValueError("causal evidence hash mismatch")
    if tuple(sorted(Counter(event["kind"] for event in events).items())) != row.event_counts:
        raise ValueError("causal evidence event kinds mismatch")
    if [action for event in events for action in event["actions"]] != row.model_dump(mode="json")[
        "actions"
    ]:
        raise ValueError("actions differ from causal evidence")
    for event, neural in zip(events, observed, strict=True):
        if (
            event["event_id"],
            event["parent_event_id"],
            event["timestamp"],
            event["kind"],
            event["prediction_snapshot_sha256"],
        ) != (
            neural.event_id,
            neural.parent_event_id,
            neural.timestamp,
            neural.kind,
            neural.prediction_snapshot_sha256,
        ):
            raise ValueError("neural and causal evidence differ")
    if row.error is None:
        if not events or events[-1]["kind"] != "terminal" or row.crash_ref is not None:
            raise ValueError("completed episode requires terminal evidence")
    elif row.crash_ref != f"crashes/index.json#{row.public_id}":
        raise ValueError("failed episode requires its crash index reference")
    if row.full_trace_ref is not None:
        validate_full_neural_trace(
            verified_bytes[row.full_trace_ref],
            identity_sha256=row.identity_sha256,
            episode_sha256=row.episode_sha256,
            events=events,
            initialization_failed=(row.error is not None and row.causal_trace_sha256 is None),
        )


def write_evaluation(
    *, identity: EvaluationIdentity, rows: Iterable[TimedEpisodeRow], output_dir: Path
) -> PilotEvaluation:
    """Stream rows immediately; DONE authenticates completion, separately from the gate.

    Partial evidence is deliberately retained on any failure and never marked DONE.
    A fresh destination is required; no previous artifact is replaced.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_create_bytes(output_dir / "identity.json", canonical_json_bytes(identity))
    selected = identity.retained_public_ids()
    atomic_create_bytes(
        output_dir / "retention.json",
        canonical_json_bytes(
            {
                "stratified_public_ids": sorted(selected),
                "report_example_public_ids": list(identity.report_example_public_ids),
                "every_failure": True,
                "every_delay_swap": True,
            }
        ),
    )
    collected, index = [], []
    pending = output_dir / ".rows.pending.jsonl"
    with pending.open("xb") as handle:
        for number, row in enumerate(rows):
            if number >= len(identity.episodes):
                raise ValueError("extra episode in evaluation inventory")
            binding = identity.episodes[number]
            if (row.public_id, row.episode_sha256, row.identity_sha256) != (
                binding.public_id,
                binding.episode_sha256,
                identity.sha256,
            ):
                raise ValueError("row differs from ordered evaluation inventory")
            row = TimedEpisodeRow.model_validate_json(row.model_dump_json())
            if (
                sha256_bytes(canonical_json_bytes(row.model_dump(mode="json")["truth"]))
                != binding.scoring_truth_sha256
                or row.truth.recipe.variant.value != binding.variant
                or row.truth.recipe.requested_path_length != binding.path_length
            ):
                raise ValueError("scoring facts differ from evaluation inventory")
            if (
                row.checkpoint_sha256 != identity.checkpoint_sha256
                or row.model_state_sha256 != identity.model_identity.model_state_sha256
                or row.producing_source_revision != identity.model_identity.source_revision
                or row.execution_source_revision != identity.execution_source_revision
                or row.purpose != identity.purpose
                or row.gate_eligible != identity.gate_eligible
                or row.config_sha256
                != sha256_bytes(identity.evaluation_config_canonical_json.encode())
                or row.manifest_sha256 != identity.manifest_sha256
            ):
                raise ValueError("row provenance differs from identity")
            payload = canonical_json_bytes(row) + b"\n"
            index.append(
                {
                    "public_id": row.public_id,
                    "offset": handle.tell(),
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                }
            )
            handle.write(payload)
            handle.flush()
            collected.append(row)
        handle.flush()
        os.fsync(handle.fileno())
    if len(collected) != len(identity.episodes):
        raise ValueError("missing episodes in evaluation inventory")
    _crash_index(output_dir, collected)
    for row in collected:
        _verify_evidence(
            output_dir,
            row,
            retain=(
                row.public_id in selected
                or row.public_id in identity.report_example_public_ids
                or identity.purpose == "delay_swap"
            ),
        )
    os.link(pending, output_dir / "rows.jsonl")
    pending.unlink()
    metrics = summarize_timed_rows(collected)
    atomic_create_bytes(output_dir / "index.json", canonical_json_bytes({"rows": index}))
    atomic_create_bytes(output_dir / "metrics.json", canonical_json_bytes(metrics))
    artifacts = tuple(
        (str(path.relative_to(output_dir)), sha256_bytes(read_evaluation_artifact(path)))
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    )
    atomic_create_bytes(
        output_dir / "DONE",
        canonical_json_bytes(
            {
                "identity_sha256": identity.sha256,
                "artifact_hashes": dict(artifacts),
                "execution_complete": True,
                "gate_passed": metrics.gate_passed,
                "validation_valid": metrics.validation_valid,
            }
        ),
    )
    return PilotEvaluation(
        identity=identity, metrics=metrics, output_path=output_dir, artifact_hashes=artifacts
    )
