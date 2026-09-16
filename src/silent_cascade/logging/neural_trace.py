"""Private observers of committed neural events; no policy callback authority."""

import gzip
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path

import torch

from silent_cascade.eval.compute import RuntimeCompute
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.io import atomic_create_bytes
from silent_cascade.logging.trace import SegmentSummary
from silent_cascade.validation import StrictModel


class RecallObservation(StrictModel):
    slot: int
    record_id: int
    score: float
    eligible: bool
    rank: int | None


class NeuralEventObservation(StrictModel):
    event_id: int
    parent_event_id: int | None
    timestamp: float
    kind: str
    prediction_snapshot_sha256: str
    segment_sha256: str
    next_segment_sha256: str | None
    guard_targets: tuple[float, ...]
    guard_rates: tuple[float, ...]
    recall_scores: tuple[RecallObservation, ...]
    predictions: tuple[tuple[str, tuple[float, ...]], ...]
    raw_action_argmax: int | None
    legal_action_argmax: int | None
    status_disagrees: bool
    compute: RuntimeCompute


def observe_neural_event(*, before, after, summary, diagnostics, compute):
    """Use pre-jump slots at actual execution time, never invoke a learned scorer."""
    terminal = summary.kind == "terminal"
    candidates = []
    if not terminal and summary.kind == "recall":
        for slot, entry in enumerate(before.core.memory.records):
            eligible = (
                entry.valid
                and not entry.consumed
                and entry.record.observed_at <= summary.timestamp
                and entry.refractory_until <= summary.timestamp
            )
            candidates.append((slot, entry.record.record_id, diagnostics.scores[slot], eligible))
    ordered = sorted(
        ((score, record_id) for _, record_id, score, valid in candidates if valid),
        key=lambda item: (-item[0], item[1]),
    )
    ranks = {record_id: index + 1 for index, (_, record_id) in enumerate(ordered)}
    return NeuralEventObservation(
        event_id=summary.event_id,
        parent_event_id=summary.parent_event_id,
        timestamp=summary.timestamp,
        kind=summary.kind,
        prediction_snapshot_sha256=summary.prediction_snapshot_sha256,
        segment_sha256=sha256_bytes(
            canonical_json_bytes(asdict(SegmentSummary.from_segment(before.segment)))
        ),
        next_segment_sha256=None
        if terminal
        else sha256_bytes(canonical_json_bytes(asdict(summary.segment))),
        guard_targets=()
        if terminal
        else tuple(after.segment.parameters.guard_targets.cpu().tolist()),
        guard_rates=() if terminal else tuple(after.segment.parameters.guard_rates.cpu().tolist()),
        recall_scores=tuple(
            RecallObservation(
                slot=slot,
                record_id=record_id,
                score=score,
                eligible=valid,
                rank=ranks.get(record_id),
            )
            for slot, record_id, score, valid in candidates
        ),
        predictions=() if terminal else diagnostics.predictions,
        raw_action_argmax=None if terminal else diagnostics.raw_action_argmax,
        legal_action_argmax=None if terminal else diagnostics.legal_action_argmax,
        status_disagrees=False if terminal else diagnostics.status_disagrees,
        compute=compute,
    )


def _host(value):
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        }
    if is_dataclass(value):
        return {field.name: _host(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_host(item) for item in value]
    if isinstance(value, dict):
        return {key: _host(item) for key, item in value.items()}
    return value


def write_full_neural_trace(path: Path, *, identity_sha256, episode_sha256, trajectory):
    """Lossless float32 JSON anchors compressed deterministically, one episode at a time."""
    snapshots = () if trajectory is None else trajectory.checkpoint_snapshots()
    raw = gzip.compress(
        canonical_json_bytes(
            {
                "schema": "phase4-neural-trajectory-v1",
                "identity_sha256": identity_sha256,
                "episode_sha256": episode_sha256,
                "anchors": [_host(state) for state in snapshots],
            }
        ),
        mtime=0,
    )
    atomic_create_bytes(path, raw)
    return sha256_bytes(raw)
