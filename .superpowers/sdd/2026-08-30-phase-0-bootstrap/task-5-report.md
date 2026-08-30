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

## Fix Round 1

Added failure-atomicity tests that establish a distinct current RNG stream
before invoking an invalid seed or an MPS-incompatible restore. Both tests
failed before the fix: `seed_all(2**64)` reached Torch after mutating Python
and NumPy, while MPS restore mutated CPU state before raising `DoctorError`.

`seed_all` now performs exact-`int` (rejecting `bool`) and inclusive
`0..2**64-1` validation before any RNG mutation. `restore_global_rng` now
preflights MPS snapshot capability before changing Python, NumPy, or Torch
CPU state.

Fix focused tests: `4 passed, 1 skipped`.

## Fix Round 2

Wrapped both failure-atomicity tests in outer RNG snapshots with `finally`
restoration, preventing process-global state leakage. Parameterized invalid
seed stream-preservation coverage for `-1` (`ValueError`), `True`
(`TypeError`), and `2**64` (`ValueError`), and added acceptance coverage for
exact endpoint seeds `0` and `2**64-1`.

Fix Round 2 focused tests: `8 passed, 1 skipped`.

## Fix Round 3

Scoped the simulated unavailable-MPS monkeypatch with
`monkeypatch.context()`, ensuring it ends before the outer snapshot is
restored. Parameterized invalid-seed tests now conditionally clone and compare
MPS RNG state in addition to Python, NumPy, and Torch CPU streams. The direct
MPS branch is skipped on this CPU host; reviewers should re-run it on an MPS
capable runtime.

Fix Round 3 focused tests: `8 passed, 1 skipped` locally.
