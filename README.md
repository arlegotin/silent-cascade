# Silent Cascade

Silent Cascade is a public research project testing persistent, learned event-flow computation on the Observation-Free Deadline benchmark.

The repository contains Phase 1 data-generator, oracle, leakage-audit, and
reproducibility infrastructure plus the Phase 2 EventFlow runtime. Phase 2 is
non-neural runtime engineering evidence, not learned-model or benchmark
evidence. The project reports no learned benchmark results and makes no claim
about consciousness, sentience, biological fidelity, or general intelligence.

## Current working path

```bash
uv sync --locked --group dev
uv run silent-cascade doctor
make verify
```

The implemented command surface is:

```text
silent-cascade doctor
silent-cascade replay ARTIFACT [--json]
silent-cascade data freeze --output PATH --root-seed INT --public-id-seed INT
silent-cascade episode inspect MANIFEST (--episode-id UUID | --entry-index INT) [--oracle]
silent-cascade oracle evaluate (--manifest PATH | --allocation phase1-gate)
silent-cascade leakage audit (--manifest PATH | --allocation phase1-gate) --profile (test | phase1-gate)
silent-cascade data freeze --pilot-stage primary --config configs/train/pilot.yaml --output PATH
silent-cascade train --config configs/train/pilot.yaml --manifest-dir PATH --run-dir PATH --seed 11 --device cpu
silent-cascade evaluate --checkpoint PATH/selected.json --manifest PATH/primary.json --output PATH --device cpu
silent-cascade report build --pilot --run-dir PATH --output PATH
```

The Phase 1 commands also accept `--config`, `--data-config`, repeatable
`--set`, and `--json` options. Allocation mode requires both seed options;
manifest mode obtains its private recipe metadata from the verified manifest
and forbids caller-supplied seeds. The exact production-size commands and
artifact paths are frozen in the approved Phase 1 plan.

`silent-cascade replay` verifies a closed private validation replay artifact or
a certified crash bundle on CPU. Its output is public-safe: it reports hashes,
event counts, and replay status, never episode truth or private terminal data.

The primary path is offline at runtime and uses exactly zero foundation-model calls. Qwen is optional, isolated, and outside the scientific path.

No CI/CD is configured for this project. Linting, tests, environment checks,
and package builds run only in the local workspace.

## Protocol

- [Canonical design specification](docs/superpowers/specs/2026-08-30-silent-cascade-design.md)
- [Implementation-plan index](docs/PLAN.md)
- [Research and claim boundary](docs/research-boundary.md)
- [Approved deviations](docs/deviations.md)

The local Phase 4 pilot workflow is available through `make pilot` and
`make pilot-smoke`. Both first require separately introduced immutable manifests
and audit reports, followed by a distinct compatible training revision. Initial
setup creates missing inputs and stops before fitting; it never commits your
checkout. See [pilot setup and command details](docs/phase4-autonomous-eventflow.md).
The four-update smoke is debug engineering evidence, not a passed production
learning gate. Intervention, general reporting and demo commands remain future work.
