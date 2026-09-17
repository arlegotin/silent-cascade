"""Private observers of committed neural events; no policy callback authority."""

import gzip
import json
import math
import zlib
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path

import torch

from silent_cascade.eval.compute import RuntimeCompute
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.io import atomic_create_bytes
from silent_cascade.logging.trace import SegmentSummary
from silent_cascade.validation import StrictModel

MAX_TRAJECTORY_JSON_BYTES = 128 * 1024 * 1024


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
    devices = _snapshot_devices(snapshots)
    if (snapshots and (len(devices) != 1 or not devices <= {"cpu", "mps"})) or (
        not snapshots and devices
    ):
        raise ValueError("trajectory requires one uniform native origin device")
    origin_device = next(iter(devices)) if devices else None
    raw = gzip.compress(
        canonical_json_bytes(
            {
                "schema": "phase4-neural-trajectory-v2",
                "origin_device": origin_device,
                "identity_sha256": identity_sha256,
                "episode_sha256": episode_sha256,
                "anchors": [_host(state) for state in snapshots],
            }
        ),
        mtime=0,
    )
    atomic_create_bytes(path, raw)
    return sha256_bytes(raw)


def _snapshot_devices(value):
    if isinstance(value, torch.Tensor):
        return {value.device.type}
    if is_dataclass(value):
        return set().union(*(_snapshot_devices(getattr(value, f.name)) for f in fields(value)))
    if isinstance(value, dict):
        return set().union(*(_snapshot_devices(v) for v in value.values()))
    if isinstance(value, (tuple, list)):
        return set().union(*(_snapshot_devices(v) for v in value))
    return set()


def _trajectory_object(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("invalid trajectory object fields")
    return value


def _trajectory_state(anchor, *, origin_device="cpu"):
    """Keep exact archive decoding separate from declared portable MPS evidence."""
    from silent_cascade.eventflow.checkpoint_state import (
        CoreMetadata,
        StateMetadata,
        _construct_state,
        _decode_state,
    )
    from silent_cascade.eventflow.invariants import _validate_portable_mps_trajectory_state

    if origin_device not in {"cpu", "mps"}:
        raise ValueError("invalid trajectory origin device")

    anchor = _trajectory_object(anchor, ("core", "segment", "time"))
    core = dict(_trajectory_object(anchor["core"], (*CoreMetadata.model_fields, "continuous")))
    continuous = core.pop("continuous")
    segment = _trajectory_object(
        anchor["segment"],
        (
            "started_at",
            "origin",
            "parameters",
            "parent_event_id",
            "prediction_snapshot_sha256",
        ),
    )
    parameters = _trajectory_object(
        segment["parameters"],
        (
            "flow_targets",
            "flow_rates",
            "guard_targets",
            "guard_rates",
        ),
    )
    tensors = {}

    def vector(value, size):
        value = _trajectory_object(value, ("dtype", "shape", "values"))
        if (
            value["dtype"] != "torch.float32"
            or value["shape"] != [size]
            or any(type(dimension) is not int for dimension in value["shape"])
            or not isinstance(value["values"], list)
            or len(value["values"]) != size
            or any(type(item) is not float or not math.isfinite(item) for item in value["values"])
        ):
            raise ValueError("invalid trajectory float32 vector")
        tensor = torch.tensor(value["values"], dtype=torch.float32)
        if tensor.tolist() != value["values"]:
            raise ValueError("trajectory values are not exact float32 values")
        return tensor

    dimensions = {
        "z_fast": 256,
        "z_slow": 64,
        "drives": 8,
        "focus_key": 64,
        "hypothesis_latent": 64,
    }
    for group, values in (
        ("current", continuous),
        ("origin", segment["origin"]),
        ("targets", parameters["flow_targets"]),
        ("rates", parameters["flow_rates"]),
    ):
        sizes = dimensions | ({"guard_accumulators": 3} if group in {"current", "origin"} else {})
        _trajectory_object(values, sizes)
        for name, size in sizes.items():
            tensors[f"anchor.{group}.{name}"] = vector(values[name], size)
    for name in ("guard_targets", "guard_rates"):
        tensors[f"anchor.{name}"] = vector(parameters[name], 3)
    metadata = StateMetadata.model_validate_json(
        canonical_json_bytes(
            {
                "core": core,
                "time": anchor["time"],
                "segment_started_at": segment["started_at"],
                "segment_parent_event_id": segment["parent_event_id"],
                "prediction_snapshot_sha256": segment["prediction_snapshot_sha256"],
            }
        )
    )
    if origin_device == "cpu":
        return _decode_state(metadata, "anchor", tensors, "cpu")
    state = _construct_state(metadata, "anchor", tensors, "cpu")
    _validate_portable_mps_trajectory_state(state)
    return state


def validate_full_neural_trace(
    raw, *, identity_sha256, episode_sha256, events, initialization_failed
):
    """Validate one bounded gzip member and its complete committed anchor structure."""
    from silent_cascade.errors import SilentCascadeError
    from silent_cascade.logging.trace import _state_summary
    from silent_cascade.schemas import Mode

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate trajectory JSON key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("invalid trajectory JSON constant")

    try:
        decoder = zlib.decompressobj(wbits=31)
        decoded = decoder.decompress(raw, MAX_TRAJECTORY_JSON_BYTES + 1)
        if (
            len(decoded) > MAX_TRAJECTORY_JSON_BYTES
            or not decoder.eof
            or decoder.unused_data
            or decoder.unconsumed_tail
        ):
            raise ValueError("invalid or oversized trajectory gzip member")
        payload = json.loads(
            decoded, object_pairs_hook=unique_object, parse_constant=reject_constant
        )
        version = payload.get("schema") if isinstance(payload, dict) else None
        if version not in {"phase4-neural-trajectory-v1", "phase4-neural-trajectory-v2"}:
            raise ValueError("invalid trajectory schema")
        keys = ("schema", "identity_sha256", "episode_sha256", "anchors")
        payload = _trajectory_object(
            payload, keys + (("origin_device",) if version.endswith("v2") else ())
        )
        origin_device = payload.get("origin_device", "cpu")
        if (payload["schema"], payload["identity_sha256"], payload["episode_sha256"]) != (
            version,
            identity_sha256,
            episode_sha256,
        ):
            raise ValueError("trajectory identity mismatch")
        anchors = payload["anchors"]
        expected_count = 0 if initialization_failed else len(events) + 1
        if not isinstance(anchors, list) or len(anchors) != expected_count:
            raise ValueError("trajectory anchor count mismatch")
        if version.endswith("v2") and (
            (not anchors and origin_device is not None)
            or (anchors and origin_device not in ("cpu", "mps"))
        ):
            raise ValueError("trajectory origin device differs from anchor inventory")
        if initialization_failed:
            if events:
                raise ValueError("initialization failure cannot have trajectory events")
            return
        previous = _trajectory_state(anchors[0], origin_device=origin_device)
        if (
            previous.core.mode is not Mode.OBSERVING
            or previous.core.last_event_id is not None
            or previous.core.last_event_time is not None
            or previous.core.actions
            or previous.core.memory.records
            or previous.core.counters.jump_applications
            or previous.time != previous.segment.started_at
        ):
            raise ValueError("invalid initial trajectory anchor")
        for anchor, event in zip(anchors[1:], events, strict=True):
            state = _trajectory_state(anchor, origin_device=origin_device)
            if (
                previous.core.mode is Mode.TERMINAL
                or state.time < previous.time
                or (
                    state.time,
                    state.core.last_event_time,
                    state.core.last_event_id,
                    state.core.mode.value,
                    state.core.counters.jump_applications,
                )
                != (
                    event["timestamp"],
                    event["timestamp"],
                    event["event_id"],
                    event["post_mode"],
                    previous.core.counters.jump_applications + 1,
                )
                or _state_summary(state.core.continuous)[1] != event["state_sha256"]
            ):
                raise ValueError("trajectory anchor differs from causal event")
            if state.core.mode is not Mode.TERMINAL and (
                state.segment.started_at != state.time
                or canonical_json_bytes(asdict(SegmentSummary.from_segment(state.segment)))
                != canonical_json_bytes(event["segment"])
            ):
                raise ValueError("trajectory segment differs from causal event")
            previous = state
    except (
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        RecursionError,
        zlib.error,
        SilentCascadeError,
    ) as error:
        raise ValueError("invalid retained trajectory evidence") from error
