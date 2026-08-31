"""Thin local adapter for the fail-closed Phase 1 reproducibility harness."""

import json
from pathlib import Path
from typing import Annotated

import typer

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config
from silent_cascade.env.reproducibility import (
    IndependentAllocationReproducibilitySource,
    ReproducibilityRequest,
    check_reproducibility,
)
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.io import atomic_create_bytes

app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: Annotated[Path, typer.Option()] = Path("configs/base.yaml"),
    data_config: Annotated[Path, typer.Option()] = Path("configs/data/primary.yaml"),
    allocation: Annotated[str, typer.Option()] = "phase1-gate",
    root_seed: Annotated[int, typer.Option()] = ...,
    public_id_seed: Annotated[int, typer.Option()] = ...,
    verify_all_source_entries: Annotated[bool, typer.Option()] = False,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    if allocation != "phase1-gate":
        raise typer.BadParameter("only the canonical phase1-gate selector is permitted")
    resolved = resolve_config(Phase1Config, [config, data_config])
    report = check_reproducibility(
        ReproducibilityRequest(
            source=IndependentAllocationReproducibilitySource(
                allocation_id="phase1-independent-gate-v1",
                root_seed=root_seed,
                public_id_seed=public_id_seed,
            ),
            sample_size=1_000,
            chunk_sizes=(1, 3, 7),
            python_hash_seeds=(0, 1),
            verify_all_source_entries=verify_all_source_entries,
            output_path=output,
        ),
        resolved,
    )
    data = canonical_json_bytes(report)
    if output is not None:
        atomic_create_bytes(output, data)
    typer.echo(json.dumps(report.model_dump(mode="json"), sort_keys=True))


if __name__ == "__main__":
    app()
