.PHONY: setup doctor lint test smoke verify

setup:
	uv sync --locked --group dev

doctor:
	uv run silent-cascade doctor

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q --ignore=tests/integration/test_phase1_services.py --ignore=tests/unit/test_leakage.py
	uv run pytest -q tests/integration/test_phase1_services.py
	uv run pytest -q tests/unit/test_leakage.py

smoke:
	uv run pytest -q tests/integration/test_cli_doctor.py tests/integration/test_cli_replay.py tests/integration/test_cli_phase1.py tests/integration/test_phase1_gate_verifier.py tests/regression/test_phase1_fixtures.py tests/regression/test_import_boundaries.py tests/neural/test_training_cli.py tests/neural/test_training_imports.py

verify: lint test doctor
	uv build
