.PHONY: setup doctor lint test smoke verify

setup:
	uv sync --locked --group dev

doctor:
	uv run silent-cascade doctor

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest -q

smoke:
	uv run pytest -q tests/integration/test_cli_doctor.py tests/integration/test_cli_phase1.py tests/integration/test_phase1_gate_verifier.py tests/regression/test_phase1_fixtures.py tests/regression/test_import_boundaries.py

verify: lint test doctor
	uv build
