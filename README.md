# Silent Cascade

Silent Cascade is a public research project testing persistent, learned event-flow computation on the Observation-Free Deadline benchmark.

The repository is in Phase 0 bootstrap and reports no benchmark results. The project tests a bounded computational mechanism; it makes no claim about consciousness, sentience, biological fidelity, or general intelligence.

## Current working path

```bash
uv sync --locked --group dev
uv run silent-cascade doctor
make ci
```

The primary path is offline at runtime and uses exactly zero foundation-model calls. Qwen is optional, isolated, and outside the scientific path.

## Protocol

- [Canonical design specification](docs/superpowers/specs/2026-08-30-silent-cascade-design.md)
- [Implementation-plan index](docs/PLAN.md)
- [Research and claim boundary](docs/research-boundary.md)
- [Approved deviations](docs/deviations.md)

Only the Phase 0 `doctor` command is implemented. Commands for data generation, training, evaluation, intervention, replay, reporting, and demos enter the repository with their owning implementation phase.
