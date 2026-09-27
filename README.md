# Silent Cascade

Silent Cascade is a public research project testing persistent, learned event-flow computation on the Observation-Free Deadline benchmark.

The corrected, offline, one-seed Phase 4 autonomous pilot passed its unchanged
engineering gate. Its selected 12,000-update model achieved 9,994/10,000 timed
successes on primary validation, 9,991/10,000 on two-hop validation and
9,966/10,000 on robustness validation, with zero dynamics failures and zero
foundation-model calls. See the [measured pilot and adverse history](docs/phase4-autonomous-eventflow.md),
[canonical gate](manifests/validation/phase4/autonomous-gate-v1.json), and
[published report](reports/phase4-pilot-v1/report.md). These are training-exposed
validation results from one seed. No claim about consciousness, sentience,
biological fidelity, or general intelligence follows.

The **Phase 5A exploratory comparison** tested fresh DEBUG episodes against a
separately trained activation-time ponderer. EventFlow scored 512/512 on IID and
504/512 on unseen depths 5–8; at caps 12–24, the ponderer scored 512/512 on both
and used about thirteen times fewer measured forward MACs. The same EventFlow
weights scored much lower when its cognitive work was compressed to activation
time. See the [paired report and limits](docs/phase5a-comparative-pilot.md).
This single-seed diagnostic is not the full strong-baseline program; the
remaining controls, five-seed tests, and frozen final-test results are pending.

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

The accepted corrected run used `PILOT_DEVICE=cpu` and
`PILOT_MANIFEST_DIR=manifests/validation/phase4-artifact-fix-v1`. A fresh pilot
uses an absent `PILOT_RUN_DIR` and a compatible committed data/source sequence:

```sh
make pilot PILOT_DEVICE=cpu PILOT_MANIFEST_DIR=manifests/validation/phase4-artifact-fix-v1 PILOT_RUN_DIR="$PILOT_RUN_DIR"
```

The fresh run needs its own source-bound local verification before separate gate
collection; follow the [pilot guide](docs/phase4-autonomous-eventflow.md). To
verify the published canonical gate, set `PILOT_RUN_DIR` to the retained corrected
raw run, then use:

```sh
uv run python scripts/verify_phase4_gate_artifact.py --artifact manifests/validation/phase4/autonomous-gate-v1.json --raw-run-dir "$PILOT_RUN_DIR"
```

Artifact-only verification without the raw run reports missing coverage.
`make pilot-smoke` remains debug engineering evidence. The guide records the
exact run history, local verification receipt and source-bound setup.
The full canonical intervention suite, general reporting and demo commands
remain future work.
