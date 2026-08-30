import json
import shutil
import subprocess

from typer.testing import CliRunner

from silent_cascade.cli import app

runner = CliRunner()


def test_cli_doctor_json_matches_report_schema(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--json", "--config", "-"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["torch_cpu_ok"] is True
    assert payload["numeric"]["flow_ok"] is True


def test_installed_console_script_runs_doctor(tmp_path) -> None:
    executable = shutil.which("silent-cascade")
    assert executable is not None
    completed = subprocess.run(
        [executable, "doctor", "--json", "--config", "-"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["ok"] is True


def test_cli_has_no_unimplemented_commands() -> None:
    assert {command.name for command in app.registered_commands} == {"doctor"}
