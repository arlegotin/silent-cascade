"""Phase4 must never inherit completion from prior phases or debug evidence."""

from pathlib import Path
from runpy import run_path

import pytest


def receipt_repository(tmp_path, *, command="@echo verified"):
    import subprocess

    root = tmp_path / "receipt-repo"
    root.mkdir()
    inputs = {
        "Makefile": "verify:\n\t" + command + "\n",
        ".python-version": "3.13\n",
        "pyproject.toml": "# fixture\n",
        "uv.lock": "# fixture\n",
        "src/silent_cascade/fixture.py": "# executable fixture\n",
        "scripts/fixture.py": "# script fixture\n",
        "tests/test_fixture.py": "# concrete tracked test input\n",
        "configs/fixture.yaml": "fixture: true\n",
        "docs/superpowers/plans/2026-09-16-phase-4-autonomous-eventflow.md": "fixture plan\n",
        "docs/superpowers/specs/2026-08-30-silent-cascade-design.md": "fixture spec\n",
    }
    for name, value in inputs.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    for args in (
        ("init", "-q"),
        ("config", "user.name", "Fixture"),
        ("config", "user.email", "fixture@example.invalid"),
        ("add", "."),
        ("commit", "-qm", "receipt fixture"),
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    return root


def receipt_head(root):
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


@pytest.mark.parametrize(
    "command,passed", [("@echo verified", True), ("@echo failed >&2; exit 7", False)]
)
def test_local_receipt_records_actual_fixed_process_and_create_only_attempt(
    tmp_path, command, passed
):
    from silent_cascade.train.pilot_checks import record_local_verification
    from silent_cascade.train.pilot_evidence import discover_local_verification

    root = receipt_repository(tmp_path, command=command)
    commit = receipt_head(root)
    output = root / "artifacts/phase4-local-verification" / commit
    receipt = record_local_verification(repo_root=root, output_dir=output)
    assert receipt.command == ("make", "verify")
    assert (receipt.returncode == 0) is passed
    assert receipt.before_inventory == receipt.after_inventory
    assert "Makefile" in receipt.before_inventory
    assert "tests/test_fixture.py" in receipt.before_inventory
    observed = discover_local_verification(repo_root=root, source_commit=commit)
    assert observed["passed"] is passed
    assert observed["receipt_sha256"]
    with pytest.raises(ValueError, match="exists"):
        record_local_verification(repo_root=root, output_dir=output)


@pytest.mark.parametrize(
    "mutation",
    ["command", "returncode", "missing_log", "corrupt", "changed_test", "changed_makefile"],
)
def test_local_receipt_rejects_mutations(tmp_path, mutation):
    import json

    from silent_cascade.train.pilot_checks import record_local_verification
    from silent_cascade.train.pilot_evidence import discover_local_verification

    root = receipt_repository(tmp_path)
    commit = receipt_head(root)
    output = root / "artifacts/phase4-local-verification" / commit
    record_local_verification(repo_root=root, output_dir=output)
    path = output / "receipt.json"
    payload = json.loads(path.read_text())
    if mutation in {"command", "returncode"}:
        payload[mutation] = ["true"] if mutation == "command" else 7
        path.write_text(json.dumps(payload))
    elif mutation == "missing_log":
        (output / "stdout.log").rename(output / "retained-stdout.log")
    elif mutation == "corrupt":
        path.write_text("{broken")
    else:
        target = "Makefile" if mutation == "changed_makefile" else "tests/test_fixture.py"
        (root / target).write_text("# changed input\n")
    with pytest.raises(ValueError):
        discover_local_verification(repo_root=root, source_commit=commit)


def test_local_receipt_data_only_ancestry_newer_failure_and_incomplete_attempt(tmp_path):
    import subprocess

    from silent_cascade.train.pilot_checks import record_local_verification
    from silent_cascade.train.pilot_evidence import discover_local_verification

    # Fixed Makefile reads an external flag: both runs exercise the actual command,
    # while unchanged tracked inputs allow the newer failure to supersede the pass.
    root = receipt_repository(tmp_path, command="@test ! -f fail-flag")
    first = receipt_head(root)
    base = root / "artifacts/phase4-local-verification"
    record_local_verification(repo_root=root, output_dir=base / first)
    (root / "data.json").write_text("{}")
    subprocess.run(["git", "add", "data.json"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "data only introduction"], cwd=root, check=True)
    second = receipt_head(root)
    assert discover_local_verification(repo_root=root, source_commit=second)["passed"]
    (root / "fail-flag").write_text("fixture")
    record_local_verification(repo_root=root, output_dir=base / second)
    assert not discover_local_verification(repo_root=root, source_commit=second)["passed"]
    (base / second / "receipt.json").rename(base / second / "retained-receipt.json")
    with pytest.raises(ValueError, match="incomplete"):
        discover_local_verification(repo_root=root, source_commit=second)


def test_local_receipt_source_changes_during_process_are_recorded_not_passed(tmp_path):
    from silent_cascade.train.pilot_checks import record_local_verification

    root = receipt_repository(tmp_path, command="@echo changed >> tests/test_fixture.py")
    output = root / "artifacts/phase4-local-verification" / receipt_head(root)
    receipt = record_local_verification(repo_root=root, output_dir=output)
    assert receipt.returncode == 0
    assert receipt.before_inventory != receipt.after_inventory
    assert not receipt.source_unchanged


def test_local_receipt_ignores_inherited_make_dry_run_flags(tmp_path, monkeypatch):
    from silent_cascade.train.pilot_checks import record_local_verification

    root = receipt_repository(tmp_path, command="@exit 7")
    monkeypatch.setenv("MAKEFLAGS", "--just-print")
    output = root / "artifacts/phase4-local-verification" / receipt_head(root)
    receipt = record_local_verification(repo_root=root, output_dir=output)
    assert receipt.returncode != 0


def test_phase4_delivery_refuses_unearned_complete(tmp_path):
    namespace = run_path(str(Path(__file__).parents[1] / "integration/test_phase0_repository.py"))
    assert "_assert_phase4_delivery_state" in namespace
    check = namespace["_assert_phase4_delivery_state"]
    check(tmp_path, "| 4 — Autonomous pilot | plan | In progress. |")
    with pytest.raises(AssertionError):
        check(tmp_path, "| 4 — Autonomous pilot | plan | Complete. |")
    gate = tmp_path / "manifests/validation/phase4/autonomous-gate-v1.json"
    gate.parent.mkdir(parents=True)
    gate.write_bytes(b"unmapped gate")
    with pytest.raises(AssertionError):
        check(tmp_path, "| 4 — Autonomous pilot | plan | In progress. |")


@pytest.mark.parametrize("canonical", [True, False])
def test_phase4_delivery_map_accepts_only_canonical_gate_path(canonical):
    from silent_cascade.train.pilot_evidence_types import Phase4DeliveryMap

    value = dict(
        gate=(
            "manifests/validation/phase4/autonomous-gate-v1.json"
            if canonical
            else "manifests/validation/phase4/autonomous-gate.json"
        ),
        gate_sha256="a" * 64,
        source_commit="b" * 40,
        selected_weights_sha256="c" * 64,
    )
    if canonical:
        assert Phase4DeliveryMap.model_validate(value).gate == value["gate"]
    else:
        with pytest.raises(ValueError):
            Phase4DeliveryMap.model_validate(value)


@pytest.mark.parametrize(
    "damage", ["missing_gate", "old_name", "hash", "source", "weights", "receipt"]
)
def test_phase4_delivery_binds_independently_verified_identity(tmp_path, damage):
    import json
    import shutil

    from silent_cascade.hashing import canonical_json_bytes

    namespace = run_path(str(Path(__file__).parents[1] / "integration/test_phase0_repository.py"))
    check = namespace["_assert_phase4_delivery_state"]
    gate = tmp_path / "manifests/validation/phase4/autonomous-gate-v1.json"
    gate.parent.mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).parents[2] / "manifests/validation/phase4/autonomous-gate-v1.json", gate
    )
    delivery = gate.parent / "delivery.json"
    shutil.copyfile(
        Path(__file__).parents[2] / "manifests/validation/phase4/delivery.json", delivery
    )
    receipt_source = json.loads(gate.read_bytes())["local_verification"]["path"]
    receipt = tmp_path / receipt_source
    receipt.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parents[2] / receipt_source, receipt)
    row = next(
        line
        for line in (Path(__file__).parents[2] / "docs/PLAN.md").read_text().splitlines()
        if line.startswith("| 4 —")
    )
    check(tmp_path, row)
    if damage == "missing_gate":
        gate.rename(gate.with_suffix(".retained"))
    elif damage == "old_name":
        gate.rename(gate.with_name("autonomous-gate.json"))
    elif damage == "hash":
        gate.write_bytes(b"different gate")
    elif damage == "source":
        mapping = json.loads(delivery.read_bytes())
        mapping["source_commit"] = "c" * 40
        delivery.write_bytes(canonical_json_bytes(mapping))
    elif damage == "weights":
        mapping = json.loads(delivery.read_bytes())
        mapping["selected_weights_sha256"] = "c" * 64
        delivery.write_bytes(canonical_json_bytes(mapping))
    else:
        receipt.write_bytes(b"damaged receipt")
    with pytest.raises(AssertionError):
        check(tmp_path, row)
