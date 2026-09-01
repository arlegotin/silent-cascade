# Silent Cascade

Silent Cascade is a public research project testing persistent, learned event-flow computation on the Observation-Free Deadline benchmark.

The repository contains the Phase 1 data-generator, oracle, leakage-audit, and
reproducibility infrastructure. It reports no learned benchmark results: the
fixed production evidence gate is a separate local run, and no trained model
result exists yet. The project tests a bounded computational mechanism; it
makes no claim about consciousness, sentience, biological fidelity, or general
intelligence.

## Current working path

```bash
uv sync --locked --group dev
uv run silent-cascade doctor
make verify
```

The implemented command surface is:

```text
silent-cascade doctor
silent-cascade data freeze --output PATH --root-seed INT --public-id-seed INT
silent-cascade episode inspect MANIFEST (--episode-id UUID | --entry-index INT) [--oracle]
silent-cascade oracle evaluate (--manifest PATH | --allocation phase1-gate)
silent-cascade leakage audit (--manifest PATH | --allocation phase1-gate) --profile (test | phase1-gate)
```

The Phase 1 commands also accept `--config`, `--data-config`, repeatable
`--set`, and `--json` options. Allocation mode requires both seed options;
manifest mode obtains its private recipe metadata from the verified manifest
and forbids caller-supplied seeds. The exact production-size commands and
artifact paths are frozen in the approved Phase 1 plan.

The primary path is offline at runtime and uses exactly zero foundation-model calls. Qwen is optional, isolated, and outside the scientific path.

No CI/CD is configured for this project. Linting, tests, environment checks,
and package builds run only in the local workspace.

## Protocol

- [Canonical design specification](docs/superpowers/specs/2026-08-30-silent-cascade-design.md)
- [Implementation-plan index](docs/PLAN.md)
- [Research and claim boundary](docs/research-boundary.md)
- [Approved deviations](docs/deviations.md)

Training, learned-condition evaluation, intervention, replay, reporting, and
demo commands remain unimplemented until their owning phases.
