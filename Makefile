.PHONY: setup doctor lint test smoke ci

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
	uv run pytest -q tests/integration/test_cli_doctor.py tests/regression/test_import_boundaries.py

ci: lint test doctor
