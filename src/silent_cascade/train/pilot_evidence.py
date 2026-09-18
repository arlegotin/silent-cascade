"""Artifact-only Phase4 authentication and independently recomputed gate arithmetic."""

from collections import Counter
from itertools import pairwise, zip_longest
from pathlib import Path
from shlex import quote

from silent_cascade.env.episode import episode_sha256
from silent_cascade.env.pilot import curriculum_to_bundle, delay_pair
from silent_cascade.eval.metrics import TimedEpisodeRow
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.report.pilot_artifacts import (
    child,
    load_evaluation,
    load_training_result,
    validate_training_result,
)
from silent_cascade.train.evidence_types import read_bytes
from silent_cascade.train.pilot_artifact_index import iter_artifact_index, parse_training_envelope
from silent_cascade.train.pilot_data import _publish_pilot_bytes, iter_pilot_examples
from silent_cascade.train.pilot_evidence_types import (
    MAX_BYTES,
    MAX_FILE_BYTES,
    REQUIRED_COVERAGE,
    LocalVerificationReceipt,
    Phase4GateArtifact,
    publish_gate_artifact,
    read_compact_rows,
    strict_json,
    write_compact_shards,
)
from silent_cascade.train.pilot_provenance import (
    PLAN,
    R2_PLAN,
    R2_SPEC,
    SPEC,
    _blobs,
    authenticate_pilot_source,
    verify_pilot_data,
)
from silent_cascade.train.provenance import git


def local_verification_inventory(repo_root, revision, *, working=False):
    """Exact tracked execution/test/build inputs; never generated acceptance outputs."""
    required = {"Makefile", ".python-version", "pyproject.toml", "uv.lock", PLAN, SPEC}
    tracked = git(repo_root, "ls-tree", "-r", "--name-only", revision).decode().splitlines()
    paths = required | {
        p for p in tracked if p.startswith(("src/", "scripts/", "tests/", "configs/"))
    }
    blobs = _blobs(repo_root, revision, paths)
    if working:
        blobs = {p: read_bytes(child(repo_root, p), limit=64 * 1024**2) for p in blobs}
        # New tracked inputs during the process change the after inventory too.
        current = set(git(repo_root, "ls-files").decode().splitlines())
        extras = {
            p for p in current if p.startswith(("src/", "scripts/", "tests/", "configs/"))
        } - paths
        blobs.update({p: read_bytes(child(repo_root, p), limit=64 * 1024**2) for p in extras})
    return {p: sha256_bytes(raw) for p, raw in sorted(blobs.items())}


def discover_local_verification(*, repo_root, source_commit):
    """Newest compatible ancestor, including failures; corrupt attempts never vanish."""
    expected = local_verification_inventory(repo_root, source_commit)
    head = git(repo_root, "rev-parse", "HEAD").decode().strip()
    git(repo_root, "merge-base", "--is-ancestor", source_commit, head)
    if local_verification_inventory(repo_root, head, working=True) != expected:
        raise ValueError("local verification inputs changed")
    ancestry = git(repo_root, "rev-list", "--topo-order", source_commit).decode().splitlines()
    base = repo_root / "artifacts/phase4-local-verification"
    for revision in ancestry:
        directory = base / revision
        if not directory.exists():
            continue
        if directory.is_symlink():
            raise ValueError("local verification directory must be regular")
        # A malformed attempt is never silently ignored, even if its claimed
        # inventory cannot be decoded to establish compatibility.
        if not (directory / "receipt.json").exists():
            raise ValueError("incomplete local verification attempt")
        receipt = LocalVerificationReceipt.model_validate_json(
            canonical_json_bytes(strict_json(directory / "receipt.json"))
        )
        committed = local_verification_inventory(repo_root, revision)
        if receipt.verification_commit != revision or receipt.before_inventory != committed:
            raise ValueError("local verification commit/inventory differs")
        for name, digest in (
            ("stdout.log", receipt.stdout_sha256),
            ("stderr.log", receipt.stderr_sha256),
            ("process-result.json", receipt.process_result_sha256),
        ):
            if (
                not (directory / name).exists()
                or sha256_bytes(read_bytes(directory / name)) != digest
            ):
                raise ValueError("local verification bound log/result missing or changed")
        process = strict_json(directory / "process-result.json")
        if process != dict(
            command=list(receipt.command),
            returncode=receipt.returncode,
            stdout_sha256=receipt.stdout_sha256,
            stderr_sha256=receipt.stderr_sha256,
        ):
            raise ValueError("local verification process result differs")
        unchanged = (
            receipt.before_inventory == receipt.after_inventory
            and receipt.verification_commit == receipt.after_commit
        )
        if receipt.source_unchanged != unchanged:
            raise ValueError("local verification source change claim differs")
        if committed != expected:
            continue
        return dict(
            path=(directory / "receipt.json").relative_to(repo_root).as_posix(),
            receipt_sha256=sha256_bytes(read_bytes(directory / "receipt.json")),
            receipt=receipt.model_dump(mode="json"),
            passed=receipt.returncode == 0 and unchanged,
        )
    return None


WORKLOAD = {
    "primary": 10000,
    "two_hop": 10000,
    "robustness": 10000,
    "primary_repeat": 10000,
    "delay_pairs": 256,
    "delays": [12.0, 48.0],
    "minimum_pair_successes": 231,
    "runtime_smoke": 64,
    "minimum_timed_successes": 9000,
    "maximum_negative_false_actions": 500,
    "maximum_dynamics_errors": 0,
    "foundation_model_calls": 0,
}
SUITES = ("primary", "two_hop", "robustness", "primary_repeat", "delay_12", "delay_48")
REQUIRED_PACKAGE_FILES = (
    "src/silent_cascade/__init__.py",
    "src/silent_cascade/cli.py",
    "src/silent_cascade/config.py",
    "src/silent_cascade/doctor.py",
    "src/silent_cascade/env/__init__.py",
    "src/silent_cascade/env/config.py",
    "src/silent_cascade/env/episode.py",
    "src/silent_cascade/env/generator.py",
    "src/silent_cascade/env/invariants.py",
    "src/silent_cascade/env/leakage.py",
    "src/silent_cascade/env/oracle.py",
    "src/silent_cascade/env/pilot.py",
    "src/silent_cascade/env/reproducibility.py",
    "src/silent_cascade/env/reward.py",
    "src/silent_cascade/env/services.py",
    "src/silent_cascade/env/timing.py",
    "src/silent_cascade/errors.py",
    "src/silent_cascade/eval/__init__.py",
    "src/silent_cascade/eval/action_diagnostics.py",
    "src/silent_cascade/eval/artifacts.py",
    "src/silent_cascade/eval/compute.py",
    "src/silent_cascade/eval/metrics.py",
    "src/silent_cascade/eval/pilot_audit.py",
    "src/silent_cascade/eval/runner.py",
    "src/silent_cascade/eventflow/__init__.py",
    "src/silent_cascade/eventflow/archive_io.py",
    "src/silent_cascade/eventflow/checkpoint.py",
    "src/silent_cascade/eventflow/checkpoint_rng.py",
    "src/silent_cascade/eventflow/checkpoint_state.py",
    "src/silent_cascade/eventflow/config.py",
    "src/silent_cascade/eventflow/engine.py",
    "src/silent_cascade/eventflow/evidence.py",
    "src/silent_cascade/eventflow/flow.py",
    "src/silent_cascade/eventflow/guards.py",
    "src/silent_cascade/eventflow/invariants.py",
    "src/silent_cascade/eventflow/jumps.py",
    "src/silent_cascade/eventflow/neural.py",
    "src/silent_cascade/eventflow/neural_checkpoint.py",
    "src/silent_cascade/eventflow/neural_context.py",
    "src/silent_cascade/eventflow/neural_replay.py",
    "src/silent_cascade/eventflow/neural_weights.py",
    "src/silent_cascade/eventflow/protocols.py",
    "src/silent_cascade/eventflow/provenance.py",
    "src/silent_cascade/eventflow/replay.py",
    "src/silent_cascade/eventflow/scheduling.py",
    "src/silent_cascade/eventflow/scripted.py",
    "src/silent_cascade/eventflow/state.py",
    "src/silent_cascade/hashing.py",
    "src/silent_cascade/io.py",
    "src/silent_cascade/logging/__init__.py",
    "src/silent_cascade/logging/crash_bundle.py",
    "src/silent_cascade/logging/manifest.py",
    "src/silent_cascade/logging/neural_trace.py",
    "src/silent_cascade/logging/runtime_diagnostics.py",
    "src/silent_cascade/logging/trace.py",
    "src/silent_cascade/memory/__init__.py",
    "src/silent_cascade/memory/encoder.py",
    "src/silent_cascade/memory/eviction.py",
    "src/silent_cascade/memory/provenance.py",
    "src/silent_cascade/memory/retrieval.py",
    "src/silent_cascade/memory/store.py",
    "src/silent_cascade/memory/tensor_store.py",
    "src/silent_cascade/models/__init__.py",
    "src/silent_cascade/models/common.py",
    "src/silent_cascade/models/config.py",
    "src/silent_cascade/models/content_loss.py",
    "src/silent_cascade/models/content_types.py",
    "src/silent_cascade/models/controller.py",
    "src/silent_cascade/models/dynamics.py",
    "src/silent_cascade/models/errors.py",
    "src/silent_cascade/models/event_flow.py",
    "src/silent_cascade/models/heads.py",
    "src/silent_cascade/models/jump.py",
    "src/silent_cascade/models/losses.py",
    "src/silent_cascade/models/transitions.py",
    "src/silent_cascade/models/types.py",
    "src/silent_cascade/models/weights.py",
    "src/silent_cascade/provenance.py",
    "src/silent_cascade/report/__init__.py",
    "src/silent_cascade/report/pilot.py",
    "src/silent_cascade/report/pilot_artifacts.py",
    "src/silent_cascade/rng.py",
    "src/silent_cascade/schemas.py",
    "src/silent_cascade/train/__init__.py",
    "src/silent_cascade/train/__main__.py",
    "src/silent_cascade/train/archive_tensors.py",
    "src/silent_cascade/train/batches.py",
    "src/silent_cascade/train/checkpoints.py",
    "src/silent_cascade/train/cli.py",
    "src/silent_cascade/train/component_eval.py",
    "src/silent_cascade/train/config.py",
    "src/silent_cascade/train/content_unroll.py",
    "src/silent_cascade/train/curriculum.py",
    "src/silent_cascade/train/curriculum_data.py",
    "src/silent_cascade/train/evidence.py",
    "src/silent_cascade/train/evidence_types.py",
    "src/silent_cascade/train/execution.py",
    "src/silent_cascade/train/numeric_inventory.py",
    "src/silent_cascade/train/objective.py",
    "src/silent_cascade/train/observations.py",
    "src/silent_cascade/train/pilot_checkpoints.py",
    "src/silent_cascade/train/pilot_checks.py",
    "src/silent_cascade/train/pilot_artifact_index.py",
    "src/silent_cascade/train/pilot_cli.py",
    "src/silent_cascade/train/pilot_config.py",
    "src/silent_cascade/train/pilot_data.py",
    "src/silent_cascade/train/pilot_evidence.py",
    "src/silent_cascade/train/pilot_evidence_types.py",
    "src/silent_cascade/train/pilot_measurement.py",
    "src/silent_cascade/train/pilot_offline.py",
    "src/silent_cascade/train/pilot_overfit.py",
    "src/silent_cascade/train/pilot_provenance.py",
    "src/silent_cascade/train/pilot_state.py",
    "src/silent_cascade/train/pilot_trainer.py",
    "src/silent_cascade/train/pilot_verification.py",
    "src/silent_cascade/train/pilot_workflow.py",
    "src/silent_cascade/train/provenance.py",
    "src/silent_cascade/train/run_directory.py",
    "src/silent_cascade/train/state.py",
    "src/silent_cascade/train/traces.py",
    "src/silent_cascade/train/trainer.py",
    "src/silent_cascade/train/unroll.py",
    "src/silent_cascade/train/verification.py",
    "src/silent_cascade/validation.py",
)

# Preserve the historical source inventory while adding the current storage closure.
HISTORICAL_REQUIRED_PACKAGE_FILES = REQUIRED_PACKAGE_FILES
STORAGE_PACKAGE_FILES = (
    "src/silent_cascade/archive/__init__.py",
    "src/silent_cascade/archive/_qualification.py",
    "src/silent_cascade/archive/_qualification_continuation.py",
    "src/silent_cascade/archive/_retained.py",
    "src/silent_cascade/archive/bundles.py",
    "src/silent_cascade/archive/catalog.py",
    "src/silent_cascade/archive/ledger.py",
    "src/silent_cascade/archive/operational.py",
    "src/silent_cascade/archive/preflight.py",
    "src/silent_cascade/archive/producer.py",
    "src/silent_cascade/archive/readers.py",
    "src/silent_cascade/archive/session.py",
    "src/silent_cascade/archive/supervisor.py",
    "src/silent_cascade/archive/transport.py",
    "src/silent_cascade/archive/types.py",
)
REQUIRED_PACKAGE_FILES = tuple(sorted((*HISTORICAL_REQUIRED_PACKAGE_FILES, *STORAGE_PACKAGE_FILES)))


def acceptance_failures(
    *,
    production,
    selected,
    suites,
    repeat,
    pair_successes,
    coverage,
    numeric,
    offline,
    local_verified=False,
):
    failures = []
    if not production:
        failures.append("debug_non_acceptance")
    if not selected:
        failures.append("no_selected_checkpoint")
    for name in ("primary", "two_hop", "robustness"):
        value = suites.get(name)
        if value is None:
            failures.append("missing_" + name)
        elif (
            value["episode_count"] != 10000
            or value["positive_count"] != 5000
            or value["safe_count"] != 2500
            or value["disconnected_count"] != 2500
            or value["timed_success_count"] < 9000
            or value["false_action_count"] > 500
            or value["error_count"]
            or value["foundation_model_calls"]
        ):
            failures.append(name + "_gate_unmet")
    if not repeat:
        failures.append("primary_repeat_unmet")
    if pair_successes < 231:
        failures.append("delay_pairs_unmet")
    for name in ("delay_12", "delay_48"):
        if name in suites and suites[name]["error_count"]:
            failures.append(name + "_invalid_execution")
    failures.extend("missing_" + key for key in REQUIRED_COVERAGE if key not in coverage)
    if not numeric:
        failures.append("selected_native_numerics_unmet")
    if not offline:
        failures.append("offline_check_unmet")
    if not local_verified:
        failures.append("local_verification_unmet")
    return tuple(failures)


def authenticate_run(run_dir, config, *, repo_root=None):
    """Discover immutable inputs from the full archive; reconstruct actual CPU tensors."""
    from silent_cascade.eventflow.checkpoint_rng import restore_rng_snapshot
    from silent_cascade.eventflow.neural import NeuralModelIdentity
    from silent_cascade.eventflow.neural_weights import decode_archive, load_neural_weights
    from silent_cascade.rng import snapshot_global_rng
    from silent_cascade.train.pilot_checkpoints import PilotCheckpoint, load_pilot_checkpoint
    from silent_cascade.train.pilot_workflow import _durable, _restore_result

    root = repo_root or Path(__file__).resolve().parents[3]
    payload = load_training_result(run_dir, run_dir / "training-result.json")
    result = _restore_result(payload)
    descriptor = result.selected_checkpoint or result.progress.latest
    if descriptor is None:
        raise ValueError("missing full training archive")
    archive_path = child(run_dir, descriptor.path)
    metadata, _ = decode_archive(
        read_bytes(archive_path, limit=1024**3),
        descriptor.sha256,
        PilotCheckpoint,
        "pilot_training",
    )
    source = authenticate_pilot_source(
        repo_root=root, source_commit=metadata.source.source_commit, config=config
    )
    manifests = {
        stage: child(root, metadata.source.data_introductions[stage + "/manifest"]["path"])
        for stage in ("one_hop", "two_hop", "primary", "robustness")
    }
    loaded, introductions = verify_pilot_data(
        repo_root=root, source_commit=source.source_commit, config=config, manifests=manifests
    )
    source = source.model_copy(update={"data_introductions": introductions})
    if metadata.source != source:
        raise ValueError("training data introduction identity differs")
    before = snapshot_global_rng()
    try:
        _durable(run_dir, config, source, "cpu", result=payload)
        session = load_pilot_checkpoint(
            archive_path,
            expected_sha256=descriptor.sha256,
            config=config,
            source=source,
            device="cpu",
        )
    finally:
        restore_rng_snapshot(before, restore_mps=False)
    model_identity = NeuralModelIdentity.from_model(
        session.model, source_revision=source.source_commit
    )
    if model_identity.model_state_sha256 != descriptor.model_state_sha256:
        raise ValueError("full archive logical model differs")
    weights_descriptor = result.selected_weights or (
        result.latest_weights if not config.config.pilot.is_production else None
    )
    weights = None
    if weights_descriptor is not None:
        weights = load_neural_weights(
            child(run_dir, weights_descriptor.path),
            expected_sha256=weights_descriptor.sha256,
            device="cpu",
        )
        if (
            weights.identity != model_identity
            or weights_descriptor.model_state_sha256 != descriptor.model_state_sha256
        ):
            raise ValueError("selected archive/portable tensor identity differs")
    if (
        config.config.pilot.is_production
        and result.selected_checkpoint is not None
        and (
            not result.gate_eligible
            or not descriptor.eligible
            or descriptor.stage != "robustness"
            or result.selected_weights is None
        )
    ):
        raise ValueError("selected checkpoint lacks training eligibility")
    return root, source, result, loaded, weights, archive_path


def expected_bundles(manifest, config, suite):
    bundles = (
        curriculum_to_bundle(example, config=config.config)
        for example in iter_pilot_examples(manifest, config=config.config)
    )
    if suite.startswith("delay_"):
        count = 0
        for bundle in bundles:
            if bundle.truth.recipe.variant.value != "positive":
                continue
            yield delay_pair(bundle, delays=(12.0, 48.0))[0 if suite == "delay_12" else 1]
            count += 1
            if count == (256 if config.config.pilot.is_production else 8):
                return
    else:
        yield from bundles


def verify_evaluation_identity(
    identity, *, manifest, config, suite, weights_sha, model_sha, source
):
    from silent_cascade.eval.artifacts import DelayTransform

    paired = suite.startswith("delay_")
    digest = sha256_bytes(canonical_json_bytes(manifest))
    purpose = (
        "delay_swap"
        if paired
        else ("pilot_validation" if config.config.pilot.is_production else "debug")
    )
    if (
        identity.experiment != "phase4-pilot-v1"
        or identity.stage != manifest.stage
        or identity.split != manifest.split
        or identity.purpose != purpose
        or identity.manifest_sha256 != digest
        or identity.parent_manifest_sha256 != (digest if paired else None)
        or identity.checkpoint_sha256 != weights_sha
        or identity.model_identity.model_state_sha256 != model_sha
        or identity.model_identity.source_revision != source
        or identity.execution_source_revision != source
        or identity.evaluation_config_canonical_json != config.canonical_json.decode()
    ):
        raise ValueError("final evaluation identity differs from selected workload")
    if paired:
        parents = [e.projected for e in manifest.entries if e.projected.variant == "positive"]
        for binding, parent in zip(identity.episodes, parents, strict=False):
            expected = DelayTransform(
                parent_public_id=parent.public_id,
                parent_episode_sha256=parent.episode_sha256,
                version="phase4-delay-pair-v1",
                parameters_canonical_json=canonical_json_bytes(
                    {"delay": 12.0 if suite == "delay_12" else 48.0}
                ).decode(),
            )
            if binding.transform != expected:
                raise ValueError("delay parent/transform inventory differs")


def compact_evaluation(directory, *, evidence_context=None):
    from silent_cascade.archive.readers import evaluation_header, iter_evaluation_episodes

    with evaluation_header(directory, evidence_context=evidence_context) as (header, _):
        pass
    episodes = iter_evaluation_episodes(
        directory,
        evidence_context=evidence_context,
        expected_header=header,
    )
    return header.identity, _compact_rows(episodes), header.hashes


def _compact_rows(episodes):
    for episode in episodes:
        row, sidecar = episode.row, episode.sidecar
        # Full semantic observations are hashed after excluding only measured telemetry.
        semantic = []
        for value in sidecar["events"]:
            item = dict(value)
            compute = dict(item.get("compute", {}))
            compute.pop("elapsed_seconds", None)
            compute.pop("mps_peak_allocation_bytes", None)
            item["compute"] = compute
            semantic.append(item)
        yield {
            "row": row.model_dump(mode="json"),
            "causal_events": sidecar["causal_events"],
            "compute_observations": [sidecar["initial_compute"]]
            + [e["compute"] for e in sidecar["events"]],
            "semantic_sha256": sha256_bytes(canonical_json_bytes({"events": semantic})),
        }


def compact_attachment_records(root, attachments):
    """Fully validate each bounded shard while yielding every row in order."""
    for attachment in attachments:
        path = child(root, attachment.path)
        if sha256_bytes(read_bytes(path, limit=MAX_BYTES)) != attachment.sha256:
            raise ValueError("compact attachment hash mismatch")
        count = 0
        for record in read_compact_rows(path):
            count += 1
            yield record
        if count != attachment.rows:
            raise ValueError("episode inventory mismatch")


def _observe_records(records, *, digests, projections, retained, continuation_ids):
    for record in records:
        # One canonical record per digest makes ordering and row boundaries
        # explicit; comparisons never concatenate ambiguous unframed values.
        digests.append(sha256_bytes(canonical_json_bytes(record)))
        projections.append(sha256_bytes(canonical_json_bytes(repeat_projection(record))))
        if record["row"]["public_id"] in continuation_ids:
            retained.append(record)
        yield record


def verify_compute_observations(compute, observations):
    from silent_cascade.eval.compute import RuntimeCompute

    parsed = [
        RuntimeCompute.model_validate_json(canonical_json_bytes(value)) for value in observations
    ]
    if not parsed:
        raise ValueError("missing compute observations")
    for name in RuntimeCompute.model_fields:
        actual = getattr(compute, name)
        values = [getattr(value, name) for value in parsed]
        if name in {"parameters", "entity_parameters", "memory_bytes"}:
            expected = max(values)
        elif name in {"module_calls", "operation_estimates"}:
            expected = dict(sum((Counter(value) for value in values), Counter()))
        elif name == "engine":
            expected = {
                key: sum(getattr(value, key) for value in values)
                for key in values[0].__dataclass_fields__
            }
            from dataclasses import asdict

            actual = asdict(actual)
        else:
            expected = sum(values)
        if actual != expected:
            raise ValueError("runtime compute aggregate differs: " + name)


def verify_rows(records, bundles, *, identity, weights_sha, model_sha, source, config):
    from dataclasses import fields

    from silent_cascade.eval.artifacts import EpisodeBinding
    from silent_cascade.eventflow.replay import CausalEventArtifact
    from silent_cascade.logging.trace import CausalEventSummary, CausalTrace

    bundles = tuple(bundles)
    if len(identity.episodes) != len(bundles):
        raise ValueError("episode inventory mismatch")
    rows = []
    for record, bundle, binding in zip_longest(records, bundles, identity.episodes):
        if record is None or bundle is None or binding is None:
            raise ValueError("episode inventory mismatch")
        if set(record) != {"row", "causal_events", "semantic_sha256", "compute_observations"}:
            raise ValueError("compact row schema differs")
        row = TimedEpisodeRow.model_validate_json(canonical_json_bytes(record["row"]))
        expected = EpisodeBinding.from_bundle(bundle, transform=binding.transform)
        if (
            binding != expected
            or row.public_id != expected.public_id
            or row.episode_sha256 != episode_sha256(bundle)
            or row.truth != bundle.truth
        ):
            raise ValueError("episode inventory/private truth mismatch")
        if (
            row.checkpoint_sha256 != weights_sha
            or row.model_state_sha256 != model_sha
            or row.producing_source_revision != source
            or row.execution_source_revision != source
            or row.config_sha256 != config.sha256
            or row.identity_sha256 != identity.sha256
            or row.manifest_sha256 != identity.manifest_sha256
            or row.purpose != identity.purpose
        ):
            raise ValueError("selected row provenance mismatch")
        events = record["causal_events"]
        summaries = []
        for event in events:
            event_source = (
                "external"
                if event["kind"] in {"fact", "activate"}
                else ("terminal" if event["kind"] == "terminal" else "internal")
            )
            parsed = CausalEventArtifact.model_validate_json(
                canonical_json_bytes(event | {"source": event_source})
            )
            summaries.append(
                CausalEventSummary(
                    **{f.name: getattr(parsed, f.name) for f in fields(CausalEventSummary)}
                )
            )
        CausalTrace(tuple(summaries))
        digest = sha256_bytes(canonical_json_bytes({"schema_version": 1, "events": events}))
        if row.causal_trace_sha256 is not None and row.causal_trace_sha256 != digest:
            raise ValueError("causal trace hash differs")
        if (
            row.event_count != len(events)
            or tuple(sorted(Counter(e["kind"] for e in events).items())) != row.event_counts
            or [a for event in events for a in event["actions"]] != record["row"]["actions"]
        ):
            raise ValueError("causal action/event inventory differs")
        if len({e["event_id"] for e in events}) != len(events) or any(
            b["timestamp"] < a["timestamp"] for a, b in pairwise(events)
        ):
            raise ValueError("illegal event order")
        if row.error is None and (not events or events[-1]["kind"] != "terminal"):
            raise ValueError("missing terminal event")
        if sum(e["kind"] in {"act", "recall", "compose"} for e in events) > 64:
            raise ValueError("internal event cap exceeded")
        if row.compute.parameters > 5_000_000 or row.compute.foundation_model_calls:
            raise ValueError("compute ceiling violated")
        if row.error is None:
            verify_compute_observations(row.compute, record["compute_observations"])
        rows.append(row)
    return tuple(rows)


def suite_counts(rows):
    result = dict(
        episode_count=len(rows),
        timed_success_count=sum(r.timed_success for r in rows),
        positive_count=sum(r.is_positive for r in rows),
        safe_count=sum(r.truth.recipe.variant.value == "safe_negative" for r in rows),
        disconnected_count=sum(
            r.truth.recipe.variant.value == "disconnected_negative" for r in rows
        ),
        false_action_count=sum(not r.is_positive and bool(r.actions) for r in rows),
        error_count=sum(r.error is not None for r in rows),
        foundation_model_calls=sum(r.compute.foundation_model_calls for r in rows),
    )
    for variant in ("positive", "safe_negative", "disconnected_negative"):
        selected = [r for r in rows if r.truth.recipe.variant.value == variant]
        result[variant + "_successes"] = sum(r.timed_success for r in selected)
        result[variant + "_denominator"] = len(selected)
    return result


def repeat_projection(record):
    row = record["row"]
    return {
        key: row[key]
        for key in (
            "public_id",
            "episode_sha256",
            "actions",
            "score",
            "error",
            "causal_trace_sha256",
            "compute",
        )
    } | {"causal_events": record["causal_events"], "semantic_sha256": record["semantic_sha256"]}


def pair_results(left, right):
    if len(left) != len(right):
        raise ValueError("delay pair inventory mismatch")
    result = []
    for a, b in zip(left, right, strict=True):
        if a.public_id != b.public_id:
            raise ValueError("delay pair order mismatch")
        times = [r.actions[0].timestamp if len(r.actions) == 1 else None for r in (a, b)]
        # Positive normalized action windows are [0.75, 0.90).
        starts = [
            r.truth.action_window_start - 0.75 * d
            for r, d in zip((a, b), (12.0, 48.0), strict=True)
        ]
        result.append(
            dict(
                public_id=a.public_id,
                passed=a.timed_success and b.timed_success,
                action_times=times,
                delta=None if None in times else times[1] - times[0],
                normalized_times=[
                    None if t is None else (t - s) / d
                    for t, s, d in zip(times, starts, (12.0, 48.0), strict=True)
                ],
            )
        )
    return tuple(result)


def verify_continuation_records(
    records, *, rows, config, source, weights_sha, run_dir=None, unavailable=None
):
    """Validate retained comparisons and real pause/replay identities, never rerun."""
    from silent_cascade.eventflow.checkpoint_state import _trace_from_metadata
    from silent_cascade.eventflow.neural_checkpoint import load_neural_runtime_checkpoint
    from silent_cascade.eventflow.neural_replay import NeuralReplayComparison, _parse

    required = {
        "public_id",
        "episode_sha256",
        "weights_sha256",
        "model_state_sha256",
        "source_commit",
        "device",
        "boundary",
        "pause_time",
        "event_index",
        "mode",
        "variant",
        "checkpoint_path",
        "checkpoint_sha256",
        "replay_path",
        "replay_sha256",
        "original",
        "restored",
        "replay_comparisons",
        "coverage",
        "matched",
    }
    by_id = {value["row"]["public_id"]: value for value in rows}
    unavailable = [] if unavailable is None else unavailable
    coverage = set()
    for record in records:
        if set(record) != required:
            raise ValueError("continuation evidence schema differs")
        original = by_id.get(record["public_id"])
        if original is None:
            raise ValueError("continuation episode missing from primary")
        row, events = original["row"], original["causal_events"]
        if (
            record["weights_sha256"] != weights_sha
            or record["device"] != "cpu"
            or record["source_commit"] != source.source_commit
            or record["model_state_sha256"] != row["model_state_sha256"]
            or record["episode_sha256"] != row["episode_sha256"]
            or record["variant"] != row["truth"]["recipe"]["variant"]
        ):
            raise ValueError("continuation selected identity differs")
        index = record["event_index"]
        if record["boundary"] == "mid_flow":
            previous = [e for e in events if e["timestamp"] <= record["pause_time"]]
            following = [e for e in events if e["timestamp"] > record["pause_time"]]
            if not following or any(e["timestamp"] == record["pause_time"] for e in events):
                raise ValueError("continuation is not a genuine mid-flow pause")
            mode = previous[-1]["post_mode"] if previous else "observing"
            prefix = len(previous)
        else:
            if type(index) is not int or not 0 <= index < len(events):
                raise ValueError("continuation boundary index differs")
            event = events[index]
            if event["kind"] != record["boundary"] or event["timestamp"] != record["pause_time"]:
                raise ValueError("continuation boundary differs")
            mode, prefix = event["post_mode"], index + 1
        expected_coverage = sorted(
            {"boundary:" + record["boundary"], "mode:" + mode, "variant:" + record["variant"]}
            & set(REQUIRED_COVERAGE)
        )
        if record["coverage"] != expected_coverage or record["mode"] != mode:
            raise ValueError("continuation coverage differs")
        expected_outcome = {
            "actions": row["actions"],
            "score": row["score"],
            "trace_sha256": row["causal_trace_sha256"],
        }
        comparisons = tuple(
            NeuralReplayComparison.model_validate_json(canonical_json_bytes(value))
            for value in record["replay_comparisons"]
        )
        if len(comparisons) != 2:
            raise ValueError("continuation requires two actual replay comparisons")
        for comparison in comparisons:
            trace = _trace_from_metadata(comparison.trace)
            if (
                trace.sha256 != comparison.trace_sha256
                or comparison.event_count != len(trace.events)
                or comparison.mismatches
                or comparison.weights_sha256 != weights_sha
                or comparison.trace_sha256 != row["causal_trace_sha256"]
                or comparison.result.public_id != row["public_id"]
                or comparison.result.model_dump(mode="json")["actions"] != row["actions"]
                or comparison.result.model_dump(mode="json")["score"] != row["score"]
            ):
                raise ValueError("continuation replay identity/outcome differs")
        matched = (
            record["original"] == record["restored"] == expected_outcome
            and comparisons[0] == comparisons[1]
        )
        if record["matched"] is not matched:
            raise ValueError("continuation comparison flag differs")
        if not matched:
            raise ValueError("continuation comparison failed; later coverage cannot excuse it")
        checkpoint_present = (
            run_dir is not None and child(run_dir, record["checkpoint_path"]).exists()
        )
        replay_present = run_dir is not None and child(run_dir, record["replay_path"]).exists()
        if checkpoint_present:
            checkpoint = load_neural_runtime_checkpoint(
                child(run_dir, record["checkpoint_path"]),
                expected_sha256=record["checkpoint_sha256"],
                config=config.config.event_flow,
                source_revision=source.source_commit,
                device="cpu",
            )
            metadata = checkpoint.metadata
            if (
                metadata.public_id != row["public_id"]
                or metadata.original_device != "cpu"
                or metadata.weights.identity.model_state_sha256 != row["model_state_sha256"]
                or metadata.state.time != record["pause_time"]
                or metadata.state.core.mode.value != mode
                or len(metadata.trace.events) != prefix
                or metadata.experiment_config_canonical_json != config.canonical_json.decode()
            ):
                raise ValueError("continuation paused archive differs")
        else:
            unavailable.append("continuation.checkpoint:" + record["checkpoint_path"])
        if replay_present:
            replay_raw = read_bytes(child(run_dir, record["replay_path"]), limit=MAX_BYTES)
            if sha256_bytes(replay_raw) != record["replay_sha256"]:
                raise ValueError("continuation replay hash differs")
            replay, bundle, _ = _parse(replay_raw)
            if (
                episode_sha256(bundle) != row["episode_sha256"]
                or replay.original_device != "cpu"
                or replay.source_revision != source.source_commit
                or replay.weights_sha256 != weights_sha
                or replay.trace != comparisons[0].trace
                or replay.expected_result != comparisons[0].result
            ):
                raise ValueError("continuation replay archive differs")
        else:
            unavailable.append("continuation.replay:" + record["replay_path"])
        if matched:
            coverage.update(expected_coverage)
    return tuple(sorted(coverage))


def verify_numeric_evidence(value, *, config, source, checkpoint, run_dir=None, unavailable=None):
    from silent_cascade.train.pilot_verification import PilotNumericReport, read_numeric_report

    try:
        report = PilotNumericReport.model_validate_json(canonical_json_bytes(value))
    except Exception as error:
        raise ValueError("numeric evidence schema differs") from error
    if (
        report.source != source
        or report.config_sha256 != config.sha256
        or report.original_checkpoint_sha256 != checkpoint["sha256"]
        or report.initial_model_sha256 != checkpoint["model_state_sha256"]
    ):
        raise ValueError("numeric selected full archive differs")
    unavailable = [] if unavailable is None else unavailable
    directory = run_dir / "final/numerics" if run_dir is not None else None
    names = {*report.artifact_hashes, "numeric-report.json", "DONE"}
    if directory is not None and all(child(directory, name).exists() for name in names):
        actual = read_numeric_report(
            run_dir / "final/numerics", config=config, source_commit=source.source_commit
        )
        if actual != report:
            raise ValueError("numeric evidence differs from raw tensor comparisons")
    else:
        unavailable.append("numeric.complete_cross_record_validation")
        if directory is not None:
            import torch

            from silent_cascade.models.event_flow import EventFlowModel
            from silent_cascade.train.pilot_verification import (
                _read_capture,
                _read_capture_operations,
                _read_checked_archive,
            )

            with torch.random.fork_rng(devices=[]):
                model = EventFlowModel(config.config.neural)
            stems = {
                name.removesuffix(".safetensors")
                for name in names
                if name.endswith(".safetensors")
                and (
                    name.startswith(("one_hop-", "primary-"))
                    or name.rsplit("/", 1)[-1]
                    in {"expected.safetensors", "restored.safetensors", "bootstrap.safetensors"}
                )
            }
            for stem in sorted(stems):
                path = child(directory, stem)
                if path.with_suffix(".json").exists():
                    if path.with_suffix(".safetensors").exists():
                        _read_capture(path, model=model, config=config.config)
                    else:
                        _read_capture_operations(path, model=model, config=config.config)
                        unavailable.append("numeric.capture_tensors:" + stem)
                else:
                    unavailable.append("numeric.capture_operations:" + stem)
            for name in (
                "input-training.safetensors",
                "resume-cpu/resume.safetensors",
                "resume-mps/resume.safetensors",
            ):
                if name in names:
                    if child(directory, name).exists():
                        _read_checked_archive(child(directory, name), config=config, source=source)
                    else:
                        unavailable.append("numeric.archive:" + name)
    return report.device_checks_passed


def verify_offline_evidence(value, *, source, run_dir=None, missing_raw=(), unavailable=None):
    from silent_cascade.train.pilot_offline import PilotOfflineReport

    report = PilotOfflineReport.model_validate_json(canonical_json_bytes(value))
    if report.source_commit != source.source_commit or report.forbidden_modules:
        raise ValueError("offline source/import evidence differs")
    unavailable = [] if unavailable is None else unavailable
    if run_dir is not None:
        from silent_cascade.eventflow.neural_replay import _parse

        root = run_dir / "final/offline"
        expected = {p: h for p, h in source.source_files.items() if p.startswith("src/")}
        if (root / "executed-source.json").exists():
            executed = strict_json(root / "executed-source.json")
            if executed != expected or report.executed_source_sha256 != sha256_bytes(
                canonical_json_bytes(executed)
            ):
                raise ValueError("offline executing source differs")
        else:
            unavailable.append("offline.executed_source")
        if (root / "step.json").exists():
            step = strict_json(root / "step.json")
            if (
                step["compute"]["backward_macs"] != report.backward_macs
                or step["compute"]["foundation_model_calls"] != 0
            ):
                raise ValueError("offline raw update/count evidence differs")
        else:
            unavailable.append("offline.update")
        if (root / "replay.json").exists():
            replay_raw = read_bytes(root / "replay.json", limit=MAX_BYTES)
            replay, _, _ = _parse(replay_raw)
            if (
                replay.weights_sha256 != report.weights_sha256
                or replay.source_revision != source.source_commit
                or sha256_bytes(replay_raw) != report.replay_sha256
                or replay.expected_result.counters.foundation_model_calls
            ):
                raise ValueError("offline raw replay/count evidence differs")
        else:
            unavailable.append("offline.replay")
        for name, digest in (
            ("weights.safetensors", report.weights_sha256),
            ("report/report.md", report.artifact_report_sha256),
        ):
            if (root / name).exists():
                if sha256_bytes(read_bytes(root / name)) != digest:
                    raise ValueError("offline raw attachment differs")
            else:
                unavailable.append("offline.attachment:" + name)
        prefix = "final/offline/run/eval/primary/"
        if (root / "run/eval/primary/DONE").exists() and not any(
            name.startswith(prefix) for name in missing_raw
        ):
            identity, rows, metrics, _ = load_evaluation(root / "run/eval/primary")
            if (
                identity.purpose != "debug"
                or len(rows) != 16
                or report.weights_sha256 != identity.checkpoint_sha256
                or report.manifest_sha256 != identity.manifest_sha256
                or metrics.foundation_model_calls
            ):
                raise ValueError("offline raw evaluation/count evidence differs")
        else:
            unavailable.append("offline.evaluation")
    else:
        unavailable.extend(
            "offline." + name
            for name in ("executed_source", "update", "replay", "evaluation", "weights", "report")
        )
    return True


def _verify_available_training(run_dir, *, training, config, source, unavailable):
    """Archive/weights authentication must not depend on old trajectory availability."""
    from silent_cascade.eventflow.checkpoint_rng import restore_rng_snapshot
    from silent_cascade.eventflow.neural import NeuralModelIdentity
    from silent_cascade.eventflow.neural_weights import load_neural_weights
    from silent_cascade.rng import snapshot_global_rng
    from silent_cascade.train.pilot_checkpoints import load_pilot_checkpoint
    from silent_cascade.train.pilot_state import PilotProgress
    from silent_cascade.train.pilot_workflow import _durable

    if run_dir is None:
        unavailable.extend(("training.archives", "training.weights", "training.durable_journal"))
        return
    progress = PilotProgress.model_validate_json(canonical_json_bytes(training["progress"]))
    for label, descriptor in (
        ("selected", training["selected_checkpoint"]),
        ("latest", training["progress"]["latest"]),
    ):
        if descriptor is None:
            continue
        path = child(run_dir, descriptor["path"])
        if not path.exists():
            unavailable.append("training.archive:" + label)
            continue
        before = snapshot_global_rng()
        try:
            session = load_pilot_checkpoint(
                path,
                expected_sha256=descriptor["sha256"],
                config=config,
                source=source,
                device="cpu",
            )
        finally:
            restore_rng_snapshot(before, restore_mps=False)
        identity = NeuralModelIdentity.from_model(
            session.model, source_revision=source.source_commit
        )
        if identity.model_state_sha256 != descriptor["model_state_sha256"]:
            raise ValueError("available training archive model differs")
        portable = training["selected_weights" if label == "selected" else "latest_weights"]
        if portable is not None and identity.model_state_sha256 != portable["model_state_sha256"]:
            raise ValueError("available training archive/portable binding differs")
        if (
            label == "selected"
            and config.config.pilot.is_production
            and (
                not training["gate_eligible"]
                or not descriptor["eligible"]
                or descriptor["stage"] != "robustness"
                or portable is None
            )
        ):
            raise ValueError("available selected archive lacks training eligibility")
        if label == "latest" and (
            session.progress.model_dump(exclude={"latest"})
            != progress.model_dump(exclude={"latest"})
            or identity != NeuralModelIdentity(**training["model_identity"])
        ):
            raise ValueError("available latest archive/result differs")
    for label in ("selected_weights", "latest_weights"):
        descriptor = training[label]
        if descriptor is None:
            continue
        path = child(run_dir, descriptor["path"])
        if not path.exists():
            unavailable.append("training.weights:" + label)
            continue
        weights = load_neural_weights(path, expected_sha256=descriptor["sha256"], device="cpu")
        if (
            weights.identity.model_state_sha256 != descriptor["model_state_sha256"]
            or weights.identity.source_revision != source.source_commit
        ):
            raise ValueError("available training portable identity differs")
    complete = (run_dir / "checkpoint-index.json").exists()
    cursor, seen = progress.journal_sha256, set()
    while cursor is not None:
        if cursor in seen:
            raise ValueError("available training journal is cyclic")
        seen.add(cursor)
        path = child(run_dir, "journal-" + cursor + ".json")
        if not path.exists():
            complete = False
            break
        raw = read_bytes(path)
        if sha256_bytes(raw) != cursor:
            raise ValueError("available training journal hash differs")
        record = strict_json(path)
        complete = complete and all(
            child(run_dir, name).exists() for name in record.get("artifacts", {})
        )
        cursor = record["prior"]
    latest = progress.latest
    complete = complete and latest is not None and child(run_dir, latest.path).exists()
    if complete:
        before = snapshot_global_rng()
        try:
            _durable(run_dir, config, source, "cpu", result=training)
        finally:
            restore_rng_snapshot(before, restore_mps=False)
    else:
        unavailable.append("training.durable_journal")


def collect_pilot_evidence(*, run_dir, config, output_path):
    root, source, result, manifests, weights, _ = authenticate_run(run_dir, config)
    attachments, projections, rows_by_suite, execution, upstream = {}, {}, {}, {}, {}
    continuation = ()
    continuation_path = run_dir / "final/continuation.json"
    if continuation_path.exists():
        continuation = tuple(strict_json(continuation_path)["records"])
        upstream["final/continuation.json"] = sha256_bytes(read_bytes(continuation_path))
    continuation_ids = {record["public_id"] for record in continuation}
    primary_records = []
    if weights is not None:
        for name in SUITES:
            directory = run_dir / "final/eval" / name
            if not directory.exists():
                continue
            identity, records, hashes = compact_evaluation(directory)
            observed = strict_json(directory / "execution.json")
            expected_execution = dict(
                schema_version="phase4-cpu-execution-v1",
                device="cpu",
                purpose=name,
                identity_sha256=identity.sha256,
                done_sha256=sha256_bytes(read_bytes(directory / "DONE")),
                weights_sha256=identity.checkpoint_sha256,
                model_state_sha256=weights.identity.model_state_sha256,
                source_commit=source.source_commit,
            )
            if observed != expected_execution:
                raise ValueError("CPU execution identity/purpose mismatch")
            stage = name if name in {"two_hop", "robustness"} else "primary"
            manifest = manifests[stage]
            verify_evaluation_identity(
                identity,
                manifest=manifest,
                config=config,
                suite=name,
                weights_sha=(result.selected_weights or result.latest_weights).sha256,
                model_sha=weights.identity.model_state_sha256,
                source=source.source_commit,
            )
            attachments[name] = write_compact_shards(
                output_path.parent / f"{output_path.stem}.{name}", records
            )
            projections[name] = []
            observed_records = _observe_records(
                compact_attachment_records(output_path.parent, attachments[name]),
                digests=[],
                projections=projections[name],
                retained=primary_records,
                continuation_ids=continuation_ids if name == "primary" else set(),
            )
            rows = verify_rows(
                observed_records,
                expected_bundles(manifest, config, name),
                identity=identity,
                weights_sha=identity.checkpoint_sha256,
                model_sha=weights.identity.model_state_sha256,
                source=source.source_commit,
                config=config,
            )
            if (
                identity.checkpoint_sha256
                != (result.selected_weights or result.latest_weights).sha256
            ):
                raise ValueError("stale selected weights")
            rows_by_suite[name] = rows
            execution[name] = {"identity": identity.model_dump(mode="json"), "execution": observed}
            upstream.update({f"final/eval/{name}/{p}": h for p, h in hashes.items()})
            for p in ("DONE", "execution.json"):
                upstream[f"final/eval/{name}/{p}"] = sha256_bytes(read_bytes(directory / p))
    coverage = verify_continuation_records(
        continuation,
        rows=primary_records,
        config=config,
        source=source,
        weights_sha=(result.selected_weights or result.latest_weights).sha256,
        run_dir=run_dir,
    )
    for record in continuation:
        for kind in ("checkpoint", "replay"):
            upstream[record[kind + "_path"]] = record[kind + "_sha256"]
    pair_rows = pair_results(rows_by_suite.get("delay_12", ()), rows_by_suite.get("delay_48", ()))
    repeat = (
        "primary" in projections
        and "primary_repeat" in projections
        and projections["primary"] == projections["primary_repeat"]
    )
    numeric, numeric_evidence = False, None
    if (run_dir / "final/numerics/DONE").exists():
        from silent_cascade.train.pilot_verification import read_numeric_report

        report = read_numeric_report(
            run_dir / "final/numerics", config=config, source_commit=source.source_commit
        )
        if result.selected_checkpoint is not None and (
            report.original_checkpoint_sha256 != result.selected_checkpoint.sha256
            or report.initial_model_sha256 != result.selected_checkpoint.model_state_sha256
        ):
            raise ValueError("numeric selected archive/model mismatch")
        numeric_evidence = report.model_dump(mode="json")
        numeric = verify_numeric_evidence(
            numeric_evidence,
            config=config,
            source=source,
            checkpoint=(result.selected_checkpoint or result.progress.latest).model_dump(
                mode="json"
            ),
        )
        upstream.update({"final/numerics/" + p: h for p, h in report.artifact_hashes.items()})
        upstream["final/numerics/numeric-report.json"] = sha256_bytes(
            read_bytes(run_dir / "final/numerics/numeric-report.json")
        )
        upstream["final/numerics/DONE"] = sha256_bytes(read_bytes(run_dir / "final/numerics/DONE"))
    offline, offline_evidence = False, None
    if (run_dir / "final/offline/offline.json").exists():
        from silent_cascade.train.pilot_offline import PilotOfflineReport

        report = PilotOfflineReport.model_validate_json(
            canonical_json_bytes(strict_json(run_dir / "final/offline/offline.json"))
        )
        executed = strict_json(run_dir / "final/offline/executed-source.json")
        expected = {p: h for p, h in source.source_files.items() if p.startswith("src/")}
        if (
            executed != expected
            or report.executed_source_sha256 != sha256_bytes(canonical_json_bytes(executed))
            or report.source_commit != source.source_commit
        ):
            raise ValueError("offline executing source differs")
        offline_evidence = report.model_dump(mode="json")
        offline = verify_offline_evidence(offline_evidence, source=source, run_dir=run_dir)
        for path in (run_dir / "final/offline").rglob("*"):
            if path.is_file():
                upstream[path.relative_to(run_dir).as_posix()] = sha256_bytes(
                    read_bytes(path, limit=1024**3)
                )
    suites = {name: suite_counts(rows) for name, rows in rows_by_suite.items()}
    local_verification = discover_local_verification(
        repo_root=root, source_commit=source.source_commit
    )
    failures = acceptance_failures(
        production=config.config.pilot.is_production,
        selected=result.selected_checkpoint is not None,
        suites=suites,
        repeat=repeat,
        pair_successes=sum(row["passed"] for row in pair_rows),
        coverage=coverage,
        numeric=numeric,
        offline=offline,
        local_verified=local_verification is not None and local_verification["passed"],
    )
    for descriptor in (result.selected_checkpoint, result.selected_weights, result.progress.latest):
        if descriptor is not None:
            upstream[descriptor.path] = descriptor.sha256
    upstream["training-result.json"] = sha256_bytes(read_bytes(run_dir / "training-result.json"))
    training_payload = strict_json(run_dir / "training-result.json")
    training_envelope = parse_training_envelope(training_payload)
    # Preserve exact descriptor roots and original serialized-result bytes.
    # All training file references remain in the bounded index, never a second
    # flattened copy inside this gate's upstream mapping.
    for part in training_envelope.artifact_index.shards:
        raw = read_bytes(child(run_dir, part.path), limit=MAX_BYTES)
        if sha256_bytes(raw) != part.sha256:
            raise ValueError("training index shard changed during collection")
        _publish_pilot_bytes(child(output_path.parent, part.path), raw)
        upstream[part.path] = part.sha256
    artifact = Phase4GateArtifact(
        source=source,
        config_canonical_json=config.canonical_json.decode(),
        outcome="debug_non_acceptance"
        if not config.config.pilot.is_production
        else ("failed" if failures else "passed"),
        selected_checkpoint_sha256=result.selected_checkpoint.sha256
        if result.selected_checkpoint
        else None,
        selected_weights_sha256=result.selected_weights.sha256 if result.selected_weights else None,
        model_state_sha256=weights.identity.model_state_sha256 if weights else None,
        training_result=training_payload,
        workload=WORKLOAD,
        suites=suites,
        attachments=attachments,
        upstream_artifact_hashes=upstream,
        execution_evidence=execution,
        continuation_evidence=continuation,
        pair_rows=pair_rows,
        repeat_equal=repeat,
        numeric_passed=numeric,
        offline_passed=offline,
        coverage=coverage,
        numeric_evidence=numeric_evidence,
        offline_evidence=offline_evidence,
        local_verification=local_verification,
        failures=failures,
        raw_regeneration_command=(
            "uv run python scripts/check_phase4_pilot.py --recover-from "
            + quote(str(output_path))
            + " --run-dir "
            + quote(str(run_dir))
            + ' --destination "${PHASE4_RECOVERY_DEST:?set an absent fresh destination}"'
            + ' --retained-run-dir "${PHASE4_RETAINED_RUN:?set retained training input mirror}"'
        ),
    )
    if (
        authenticate_pilot_source(
            repo_root=root, source_commit=source.source_commit, config=config
        ).source_files
        != source.source_files
    ):
        raise ValueError("source changed during collection")
    publish_gate_artifact(output_path, artifact)
    return artifact


def verify_phase4_gate_artifact(artifact_path, *, repo_root, raw_run_dir=None):
    from silent_cascade.eval.artifacts import EvaluationIdentity
    from silent_cascade.train.pilot_config import resolve_pilot_config

    artifact = Phase4GateArtifact.model_validate_json(
        canonical_json_bytes(strict_json(artifact_path, limit=MAX_FILE_BYTES))
    )
    config = resolve_pilot_config(
        "phase4_pilot"
        if '"profile":"phase4_pilot"' in artifact.config_canonical_json
        else "phase4_smoke"
    )
    source = authenticate_pilot_source(
        repo_root=repo_root, source_commit=artifact.source.source_commit, config=config
    )
    if {name for name in source.source_files if name.startswith("src/")} != set(
        REQUIRED_PACKAGE_FILES
    ):
        raise ValueError("required independent source inventory differs")
    required_nonpackage = {
        R2_PLAN,
        R2_SPEC,
        "configs/base.yaml",
        "configs/data/primary.yaml",
        "configs/model/event_flow.yaml",
        "configs/model/neural_components.yaml",
        "configs/train/pilot.yaml",
        "pyproject.toml",
        "uv.lock",
        "docs/superpowers/plans/2026-09-16-phase-4-autonomous-eventflow.md",
        "docs/superpowers/specs/2026-08-30-silent-cascade-design.md",
        "scripts/run_pilot.py",
        "scripts/check_phase4_pilot.py",
        "scripts/verify_phase4_gate_artifact.py",
        "scripts/record_phase4_local_verify.py",
        "Makefile",
        ".python-version",
    }
    if not config.config.pilot.is_production:
        required_nonpackage.add("configs/train/pilot_smoke.yaml")
    if set(source.source_files) != set(REQUIRED_PACKAGE_FILES) | required_nonpackage:
        raise ValueError("required independent entrypoint/config inventory differs")
    if (
        source.source_files != artifact.source.source_files
        or config.canonical_json.decode() != artifact.config_canonical_json
        or artifact.workload != WORKLOAD
    ):
        raise ValueError("gate source/config/workload differs")
    manifests = {
        stage: child(repo_root, artifact.source.data_introductions[stage + "/manifest"]["path"])
        for stage in ("one_hop", "two_hop", "primary", "robustness")
    }
    loaded, introductions = verify_pilot_data(
        repo_root=repo_root, source_commit=source.source_commit, config=config, manifests=manifests
    )
    if source.model_copy(update={"data_introductions": introductions}) != artifact.source:
        raise ValueError("gate introduction history differs")
    source = artifact.source
    selected_checkpoint = artifact.training_result["selected_checkpoint"]
    selected_weights = artifact.training_result["selected_weights"]
    if artifact.selected_checkpoint_sha256 != (
        selected_checkpoint["sha256"] if selected_checkpoint else None
    ) or artifact.selected_weights_sha256 != (
        selected_weights["sha256"] if selected_weights else None
    ):
        raise ValueError("gate selected descriptor differs")
    record_hashes, projections, suites, rows_by_suite = {}, {}, {}, {}
    primary_records = []
    continuation_ids = {record["public_id"] for record in artifact.continuation_evidence}
    if set(artifact.execution_evidence) != set(artifact.attachments):
        raise ValueError("execution/attachment inventory differs")
    for name, attachments in artifact.attachments.items():
        if name not in SUITES:
            raise ValueError("unexpected suite")
        record_hashes[name], projections[name] = [], []
        records = _observe_records(
            compact_attachment_records(artifact_path.parent, attachments),
            digests=record_hashes[name],
            projections=projections[name],
            retained=primary_records,
            continuation_ids=continuation_ids if name == "primary" else set(),
        )
        identity = EvaluationIdentity.model_validate_json(
            canonical_json_bytes(artifact.execution_evidence[name]["identity"])
        )
        execution = artifact.execution_evidence[name]["execution"]
        expected_execution = dict(
            schema_version="phase4-cpu-execution-v1",
            device="cpu",
            purpose=name,
            identity_sha256=identity.sha256,
            done_sha256=artifact.upstream_artifact_hashes[f"final/eval/{name}/DONE"],
            weights_sha256=identity.checkpoint_sha256,
            model_state_sha256=artifact.model_state_sha256,
            source_commit=source.source_commit,
        )
        if execution != expected_execution:
            raise ValueError("CPU execution identity/purpose mismatch")
        selected = (
            artifact.training_result["selected_weights"]
            or artifact.training_result["latest_weights"]
        )
        if (
            config.config.pilot.is_production
            and artifact.training_result["selected_weights"] is None
        ):
            raise ValueError("production evidence without selection")
        stage = name if name in {"two_hop", "robustness"} else "primary"
        verify_evaluation_identity(
            identity,
            manifest=loaded[stage],
            config=config,
            suite=name,
            weights_sha=selected["sha256"],
            model_sha=artifact.model_state_sha256,
            source=source.source_commit,
        )
        expected_manifest_sha = sha256_bytes(canonical_json_bytes(loaded[stage]))
        expected_purpose = (
            "delay_swap"
            if name.startswith("delay_")
            else ("pilot_validation" if config.config.pilot.is_production else "debug")
        )
        if (
            identity.stage != stage
            or identity.split != loaded[stage].split
            or identity.purpose != expected_purpose
            or identity.manifest_sha256 != expected_manifest_sha
            or identity.parent_manifest_sha256
            != (expected_manifest_sha if name.startswith("delay_") else None)
        ):
            raise ValueError("final evaluation manifest/purpose differs")
        rows = verify_rows(
            records,
            expected_bundles(loaded[stage], config, name),
            identity=identity,
            weights_sha=selected["sha256"],
            model_sha=artifact.model_state_sha256,
            source=source.source_commit,
            config=config,
        )
        suites[name], rows_by_suite[name] = suite_counts(rows), rows
    if suites != artifact.suites:
        raise ValueError("success aggregate mismatch")
    repeat = (
        "primary" in projections
        and "primary_repeat" in projections
        and projections["primary"] == projections["primary_repeat"]
    )
    pairs = pair_results(rows_by_suite.get("delay_12", ()), rows_by_suite.get("delay_48", ()))
    if (
        canonical_json_bytes({"pairs": pairs})
        != canonical_json_bytes({"pairs": artifact.pair_rows})
        or repeat != artifact.repeat_equal
    ):
        raise ValueError("repeat/pair evidence mismatch")
    missing, unavailable = [], []
    envelope = parse_training_envelope(artifact.training_result)
    validate_training_result(
        envelope.model_dump(mode="json", exclude={"schema_version", "artifact_index"})
        | {"artifact_hashes": {}}
    )
    if not envelope.artifact_index.entry_count:
        raise ValueError("missing training artifact inventory")
    required_upstream = {part.path: part.sha256 for part in envelope.artifact_index.shards}
    required_upstream["training-result.json"] = sha256_bytes(
        canonical_json_bytes(artifact.training_result)
    )
    for descriptor in (
        selected_checkpoint,
        selected_weights,
        artifact.training_result["progress"]["latest"],
    ):
        if descriptor is not None:
            required_upstream[descriptor["path"]] = descriptor["sha256"]
    for record in artifact.continuation_evidence:
        for kind in ("checkpoint", "replay"):
            required_upstream[record[kind + "_path"]] = record[kind + "_sha256"]
    if artifact.numeric_evidence is not None:
        required_upstream.update(
            {
                "final/numerics/" + name: digest
                for name, digest in artifact.numeric_evidence["artifact_hashes"].items()
            }
        )
        required_upstream["final/numerics/numeric-report.json"] = sha256_bytes(
            canonical_json_bytes(artifact.numeric_evidence)
        )
    if any(
        artifact.upstream_artifact_hashes.get(name) != digest
        for name, digest in required_upstream.items()
    ):
        raise ValueError("upstream artifact inventory differs")

    def verify_raw(name, digest):
        if raw_run_dir is None or not child(raw_run_dir, name).exists():
            missing.append(name)
        elif sha256_bytes(read_bytes(child(raw_run_dir, name), limit=1024**3)) != digest:
            raise ValueError("raw attachment hash mismatch: " + name)

    for name, digest in artifact.upstream_artifact_hashes.items():
        verify_raw(name, digest)
    for name, digest in iter_artifact_index(artifact_path.parent, envelope.artifact_index):
        if name in artifact.upstream_artifact_hashes:
            if artifact.upstream_artifact_hashes[name] != digest:
                raise ValueError("training/upstream artifact hash differs")
        else:
            verify_raw(name, digest)
    raw_complete = raw_run_dir is not None and not missing
    _verify_available_training(
        raw_run_dir,
        training=artifact.training_result,
        config=config,
        source=source,
        unavailable=unavailable,
    )
    if raw_complete:
        authenticate_run(raw_run_dir, config, repo_root=repo_root)
        if strict_json(raw_run_dir / "training-result.json") != artifact.training_result:
            raise ValueError("raw selected training result differs")
    if raw_run_dir is not None:
        for name in record_hashes:
            if any(path.startswith(f"final/eval/{name}/") for path in missing):
                unavailable.append("evaluation:" + name)
                continue
            _, raw_records, _ = compact_evaluation(raw_run_dir / "final/eval" / name)
            if record_hashes[name] != [
                sha256_bytes(canonical_json_bytes(record)) for record in raw_records
            ]:
                raise ValueError("raw episode inventory differs")
    else:
        unavailable.extend("evaluation:" + name for name in record_hashes)
    weights = selected_weights or artifact.training_result["latest_weights"]
    coverage = verify_continuation_records(
        artifact.continuation_evidence,
        rows=primary_records,
        config=config,
        source=source,
        weights_sha=weights["sha256"],
        run_dir=raw_run_dir,
        unavailable=unavailable,
    )
    numeric = artifact.numeric_evidence is not None and verify_numeric_evidence(
        artifact.numeric_evidence,
        config=config,
        source=source,
        checkpoint=selected_checkpoint or artifact.training_result["progress"]["latest"],
        run_dir=raw_run_dir,
        unavailable=unavailable,
    )
    offline = artifact.offline_evidence is not None and verify_offline_evidence(
        artifact.offline_evidence,
        source=source,
        run_dir=raw_run_dir,
        missing_raw=missing,
        unavailable=unavailable,
    )
    if (
        coverage != artifact.coverage
        or numeric != artifact.numeric_passed
        or offline != artifact.offline_passed
    ):
        raise ValueError("gate execution evidence differs from recorded comparisons")
    local_verification = discover_local_verification(
        repo_root=repo_root, source_commit=source.source_commit
    )
    if local_verification != artifact.local_verification:
        raise ValueError("local verification receipt differs from published gate")
    failures = acceptance_failures(
        production=config.config.pilot.is_production,
        selected=artifact.selected_checkpoint_sha256 is not None,
        suites=suites,
        repeat=repeat,
        pair_successes=sum(r["passed"] for r in pairs),
        coverage=coverage,
        numeric=numeric,
        offline=offline,
        local_verified=local_verification is not None and local_verification["passed"],
    )
    outcome = (
        "debug_non_acceptance"
        if not config.config.pilot.is_production
        else ("failed" if failures else "passed")
    )
    if failures != artifact.failures or outcome != artifact.outcome:
        raise ValueError("gate outcome differs from recomputed conditions")
    return dict(
        valid=True,
        passed=outcome == "passed" and not missing,
        recorded_outcome=outcome,
        missing_raw_attachments=missing,
        unavailable_semantic_checks=sorted(set(unavailable)),
        verification_scope="recorded_evidence_integrity",
        neural_replay="not_rerun",
        source_commit=source.source_commit,
        selected_weights_sha256=artifact.selected_weights_sha256,
        artifact_sha256=sha256_bytes(read_bytes(artifact_path, limit=MAX_BYTES)),
    )
