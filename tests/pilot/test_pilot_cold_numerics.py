"""Artifact-only numerical semantic readers through cold leases."""

import inspect
import json
import shutil

import pytest

from .cold_semantics_fixture import (
    NUMERIC_RUNTIME_ROOT,
    HistoricalContext,
    historical_root,
    numeric_inventory,
)
from .test_pilot_cold_semantics import install_no_execution_tripwires


@pytest.fixture(scope="module")
def historical_numeric_case():
    backing = historical_root()
    if backing is None:
        pytest.skip("requires private retained Task 4 smoke evidence")

    from silent_cascade.config import ResolvedConfig
    from silent_cascade.hashing import sha256_bytes
    from silent_cascade.train.pilot_config import parse_phase4_canonical
    from silent_cascade.train.pilot_provenance import PilotSourceIdentity

    gate = json.loads((backing / "phase4-gate.json").read_bytes())
    canonical = gate["config_canonical_json"].encode()
    config = ResolvedConfig(
        config=parse_phase4_canonical(canonical.decode()),
        canonical_json=canonical,
        sha256=sha256_bytes(canonical),
        source_paths=(),
    )
    source = PilotSourceIdentity.model_validate(gate["source"])
    inventory = numeric_inventory(gate)
    assert source.source_commit == "42b8ab0ba8a647794b08ce29327fce105e3a75f7"
    assert len(inventory) == 75
    return dict(backing=backing, gate=gate, config=config, source=source, inventory=inventory)


def cold_kwargs(function, context):
    return (
        {"evidence_context": context}
        if "evidence_context" in inspect.signature(function).parameters
        else {}
    )


def context_for(case, tmp_path):
    run = tmp_path / "logical-run"
    run.mkdir(parents=True)
    return run, HistoricalContext(
        run,
        case["backing"],
        case["inventory"],
        evaluation_root=NUMERIC_RUNTIME_ROOT,
    )


def selected_checkpoint(case):
    return case["gate"]["training_result"]["progress"]["latest"]


def test_genuine_numeric_report_matches_cold_read(historical_numeric_case, tmp_path, monkeypatch):
    from silent_cascade.train.pilot_verification import read_numeric_report

    case = historical_numeric_case
    install_no_execution_tripwires(monkeypatch)
    eager = read_numeric_report(
        case["backing"] / "final/numerics",
        config=case["config"],
        source_commit=case["source"].source_commit,
    )

    run, context = context_for(case, tmp_path)
    with context.guarded_reads(monkeypatch):
        cold = read_numeric_report(
            run / "final/numerics",
            config=case["config"],
            source_commit=case["source"].source_commit,
            **cold_kwargs(read_numeric_report, context),
        )

    assert cold == eager
    assert cold.source.source_commit == "42b8ab0ba8a647794b08ce29327fce105e3a75f7"
    assert cold.missing_devices == ("mps",)
    assert not cold.device_checks_passed
    assert cold.runtime_episodes == 16
    assert context.payload_maximum == 1 and context.active == 0


def test_genuine_numeric_verifier_and_runtime_rows_match_cold_reads(
    historical_numeric_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_evidence import verify_numeric_evidence
    from silent_cascade.train.pilot_verification import _runtime_rows

    case = historical_numeric_case
    install_no_execution_tripwires(monkeypatch)
    eager_unavailable = []
    eager = verify_numeric_evidence(
        case["gate"]["numeric_evidence"],
        config=case["config"],
        source=case["source"],
        checkpoint=selected_checkpoint(case),
        run_dir=case["backing"],
        unavailable=eager_unavailable,
    )
    eager_rows = _runtime_rows(case["backing"] / NUMERIC_RUNTIME_ROOT)

    run, context = context_for(case, tmp_path)
    cold_unavailable = []
    with context.guarded_reads(monkeypatch):
        cold = verify_numeric_evidence(
            case["gate"]["numeric_evidence"],
            config=case["config"],
            source=case["source"],
            checkpoint=selected_checkpoint(case),
            run_dir=run,
            unavailable=cold_unavailable,
            **cold_kwargs(verify_numeric_evidence, context),
        )
        cold_rows = _runtime_rows(run / NUMERIC_RUNTIME_ROOT, **cold_kwargs(_runtime_rows, context))

    assert cold == eager is False
    assert cold_unavailable == eager_unavailable == []
    assert cold_rows == eager_rows
    assert len(cold_rows) == 16
    assert context.payload_maximum == 1 and context.active == 0


def test_missing_capture_tensor_validates_available_operations(
    historical_numeric_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_evidence import verify_numeric_evidence

    case = historical_numeric_case
    run, context = context_for(case, tmp_path)
    tensor = "final/numerics/one_hop-cpu.safetensors"
    operation = "final/numerics/one_hop-cpu.json"
    context.hidden.add(tensor)
    unavailable = []
    install_no_execution_tripwires(monkeypatch)
    with context.guarded_reads(monkeypatch):
        result = verify_numeric_evidence(
            case["gate"]["numeric_evidence"],
            config=case["config"],
            source=case["source"],
            checkpoint=selected_checkpoint(case),
            run_dir=run,
            unavailable=unavailable,
            evidence_context=context,
        )

    assert result is False
    assert unavailable == [
        "numeric.complete_cross_record_validation",
        "numeric.capture_tensors:one_hop-cpu",
    ]
    assert "file:" + operation in context.requests
    assert "file:" + tensor not in context.requests
    assert context.payload_maximum == 1 and context.active == 0


def test_missing_capture_tensor_cannot_hide_corrupt_operations(
    historical_numeric_case, tmp_path, monkeypatch
):
    from silent_cascade.hashing import canonical_json_bytes
    from silent_cascade.train.pilot_evidence import verify_numeric_evidence

    case = historical_numeric_case
    run, context = context_for(case, tmp_path)
    tensor = "final/numerics/one_hop-cpu.safetensors"
    operation = "final/numerics/one_hop-cpu.json"
    context.hidden.add(tensor)
    control_root = tmp_path / "corrupt-operation"
    control = control_root / operation
    value = json.loads((case["backing"] / operation).read_bytes())
    value["update"]["compute"]["foundation_model_calls"] = 1
    raw = canonical_json_bytes(value)
    assert (case["backing"] / operation).stat().st_size == 21_337
    assert len(raw) <= 65_536 and ((len(raw) + 4095) // 4096) * 4096 <= 65_536
    control.parent.mkdir(parents=True)
    control.write_bytes(raw)
    context.overrides[operation] = control_root
    install_no_execution_tripwires(monkeypatch)
    with context.guarded_reads(monkeypatch), pytest.raises(ValueError, match="compute identity"):
        verify_numeric_evidence(
            case["gate"]["numeric_evidence"],
            config=case["config"],
            source=case["source"],
            checkpoint=selected_checkpoint(case),
            run_dir=run,
            evidence_context=context,
        )
    assert "file:" + operation in context.requests
    assert context.payload_maximum == 1 and context.active == 0


def test_final_runtime_corruption_is_rejected(historical_numeric_case, tmp_path, monkeypatch):
    from silent_cascade.train.pilot_verification import read_numeric_report

    case = historical_numeric_case
    run, context = context_for(case, tmp_path)
    entries = context.episode_commit(NUMERIC_RUNTIME_ROOT, 15)
    entries = (*entries.owned, *entries.borrowed)
    assert len(entries) == 4
    assert sum(((entry.bytes + 4095) // 4096) * 4096 for entry in entries) <= 3_366_912
    context.corrupt_final_episode(tmp_path / "final-runtime-control")
    install_no_execution_tripwires(monkeypatch)
    with context.guarded_reads(monkeypatch), pytest.raises(ValueError, match="integrity mismatch"):
        read_numeric_report(
            run / "final/numerics",
            config=case["config"],
            source_commit=case["source"].source_commit,
            evidence_context=context,
        )
    assert f"episode:{NUMERIC_RUNTIME_ROOT}:15" in context.requests
    assert context.payload_maximum == 1 and context.active == 0


def test_substituted_runtime_done_is_rejected(historical_numeric_case, tmp_path, monkeypatch):
    from silent_cascade.hashing import canonical_json_bytes
    from silent_cascade.train.pilot_verification import read_numeric_report

    case = historical_numeric_case
    run, context = context_for(case, tmp_path)
    control_root = tmp_path / "runtime-metadata-control"
    names = (
        "DONE",
        "identity.json",
        "retention.json",
        "rows.jsonl",
        "index.json",
        "metrics.json",
    )
    sources = tuple(case["backing"] / NUMERIC_RUNTIME_ROOT / name for name in names)
    assert sum(source.stat().st_size for source in sources) == 81_907
    rounded = sum(((source.stat().st_size + 4095) // 4096) * 4096 for source in sources)
    assert rounded == 94_208 and rounded + 4 * 4096 <= 128 * 1024
    for name, source in zip(names, sources, strict=True):
        target = control_root / NUMERIC_RUNTIME_ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    target = control_root / NUMERIC_RUNTIME_ROOT / "DONE"
    done = json.loads(target.read_bytes())
    done["identity_sha256"] = "0" * 64
    target.write_bytes(canonical_json_bytes(done))
    assert (
        sum(path.stat().st_size for path in control_root.rglob("*") if path.is_file()) < 128 * 1024
    )
    context.overrides[("metadata", NUMERIC_RUNTIME_ROOT)] = control_root
    install_no_execution_tripwires(monkeypatch)
    with context.guarded_reads(monkeypatch), pytest.raises(ValueError, match="identity hash"):
        read_numeric_report(
            run / "final/numerics",
            config=case["config"],
            source_commit=case["source"].source_commit,
            evidence_context=context,
        )
    assert context.payload_maximum <= 1 and context.active == 0


def test_extra_resident_numeric_inventory_is_rejected(
    historical_numeric_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_verification import read_numeric_report

    case = historical_numeric_case
    run, context = context_for(case, tmp_path)
    extra = run / "final/numerics/unbound.txt"
    assert len(b"unbound") <= 4096
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"unbound")
    install_no_execution_tripwires(monkeypatch)
    with context.guarded_reads(monkeypatch), pytest.raises(ValueError, match="closure"):
        read_numeric_report(
            run / "final/numerics",
            config=case["config"],
            source_commit=case["source"].source_commit,
            evidence_context=context,
        )
    assert context.payload_maximum == 1 and context.active == 0


def test_wrong_numeric_root_and_late_inventory_failure_precede_reads(
    historical_numeric_case, tmp_path, monkeypatch
):
    from silent_cascade.train.pilot_verification import read_numeric_report

    case = historical_numeric_case
    run, context = context_for(case, tmp_path)
    _, late_context = context_for(case, tmp_path / "late")
    with context.guarded_reads(monkeypatch), pytest.raises(ValueError, match="root differs"):
        read_numeric_report(
            run / "wrong",
            config=case["config"],
            source_commit=case["source"].source_commit,
            evidence_context=context,
        )
    assert context.requests == []

    late_context.late_failure = ValueError("late numeric inventory failure")
    with late_context.guarded_reads(monkeypatch), pytest.raises(ValueError, match="late numeric"):
        read_numeric_report(
            late_context.run_dir / "final/numerics",
            config=case["config"],
            source_commit=case["source"].source_commit,
            evidence_context=late_context,
        )
    assert late_context.requests == []
