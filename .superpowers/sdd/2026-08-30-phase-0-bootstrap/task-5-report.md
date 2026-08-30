# Task 5 Report: Reproducible Global RNG State

## RED

Added the exact CPU and capability tests from the Task 5 brief. Before
implementation, `uv run pytest tests/unit/test_rng.py -q` failed during
collection with `ModuleNotFoundError: No module named 'silent_cascade.rng'`.

## GREEN

Implemented immutable dataclass snapshots for Python, NumPy, Torch CPU, and
optional MPS state; strict serializable round-trip reports; global seeding;
capability detection; and a diagnostic that restores the caller's RNG state in
`finally`.

Focused tests: `2 passed, 1 skipped` (MPS capability-gated).

Full unit tests: `37 passed, 1 skipped`.

Ruff check passed and Ruff format check passed. `git diff --check` passed.

## MPS environment evidence

The active runtime reports:

```text
mps_available: False
mps_rng_state_supported: False
```

The MPS-only test is therefore skipped as required by the local capability
gate. The CPU round-trip diagnostic reports Python, NumPy, and Torch CPU all
successful, with MPS unchecked and `torch_mps_ok: null`.
