"""Thin local adapter for the fail-closed Phase 1 reproducibility harness."""

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from silent_cascade.config import resolve_config
from silent_cascade.env.config import Phase1Config
from silent_cascade.env.reproducibility import (
    IndependentAllocationReproducibilitySource,
    ManifestReproducibilitySource,
    ReproducibilityRequest,
    check_reproducibility,
)
from silent_cascade.errors import ArtifactIntegrityError, AtomicWriteError, SilentCascadeError
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.io import atomic_create_bytes

app = typer.Typer(add_completion=False)

_PHASE1_ALLOCATION_SELECTOR = "phase1-gate"
_PHASE1_ALLOCATION_ID = "phase1-independent-gate-v1"
_PRODUCTION_SAMPLE_SIZE = 1_000
_PRODUCTION_CHUNK_SIZES = (1, 3, 7)
_PRODUCTION_PYTHON_HASH_SEEDS = (0, 1)


def _seed(value: int, *, option: str) -> int:
    if not 0 <= value < 2**128:
        raise typer.BadParameter(
            "seed must be an unsigned 128-bit integer",
            param_hint=option,
        )
    return value


def _source(
    *,
    manifest: Path | None,
    allocation: str | None,
    root_seed: int | None,
    public_id_seed: int | None,
) -> ManifestReproducibilitySource | IndependentAllocationReproducibilitySource:
    if (manifest is None) == (allocation is None):
        raise typer.BadParameter(
            "select exactly one of --manifest or --allocation",
            param_hint="--manifest/--allocation",
        )
    if manifest is not None:
        if root_seed is not None or public_id_seed is not None:
            raise typer.BadParameter(
                "manifest mode forbids --root-seed and --public-id-seed",
                param_hint="--manifest",
            )
        return ManifestReproducibilitySource(manifest_path=manifest)
    assert allocation is not None
    if allocation != _PHASE1_ALLOCATION_SELECTOR:
        raise typer.BadParameter(
            "only the canonical phase1-gate selector is permitted",
            param_hint="--allocation",
        )
    if root_seed is None or public_id_seed is None:
        raise typer.BadParameter(
            "allocation mode requires --root-seed and --public-id-seed",
            param_hint="--allocation",
        )
    return IndependentAllocationReproducibilitySource(
        allocation_id=_PHASE1_ALLOCATION_ID,
        root_seed=_seed(root_seed, option="--root-seed"),
        public_id_seed=_seed(public_id_seed, option="--public-id-seed"),
    )


def _publish_report(path: Path, data: bytes) -> None:
    try:
        atomic_create_bytes(path, data)
    except AtomicWriteError as error:
        if error.message != "artifact already exists":
            raise
        try:
            existing = path.read_bytes()
        except OSError as read_error:
            raise ArtifactIntegrityError(
                "existing reproducibility report could not be verified"
            ) from read_error
        if existing != data:
            raise ArtifactIntegrityError(
                "different immutable reproducibility report already exists"
            ) from error
    try:
        published = path.read_bytes()
    except OSError as error:
        raise ArtifactIntegrityError(
            "published reproducibility report could not be verified"
        ) from error
    if published != data:
        raise ArtifactIntegrityError("published reproducibility report verification failed")


def _render_error(error: SilentCascadeError) -> None:
    payload = {"code": error.code, "context": {}, "message": error.message}
    sys.stderr.buffer.write(canonical_json_bytes(payload) + b"\n")


@app.command()
def main(
    config: Annotated[Path, typer.Option()] = Path("configs/base.yaml"),
    data_config: Annotated[Path, typer.Option()] = Path("configs/data/primary.yaml"),
    manifest: Annotated[Path | None, typer.Option("--manifest")] = None,
    allocation: Annotated[str | None, typer.Option("--allocation")] = None,
    root_seed: Annotated[int | None, typer.Option("--root-seed")] = None,
    public_id_seed: Annotated[int | None, typer.Option("--public-id-seed")] = None,
    sample_size: Annotated[int, typer.Option("--sample-size", min=1)] = _PRODUCTION_SAMPLE_SIZE,
    chunk_size: Annotated[list[int] | None, typer.Option("--chunk-size", min=1)] = None,
    python_hash_seed: Annotated[list[int] | None, typer.Option("--python-hash-seed", min=0)] = None,
    verify_all_source_entries: Annotated[bool, typer.Option()] = False,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    source = _source(
        manifest=manifest,
        allocation=allocation,
        root_seed=root_seed,
        public_id_seed=public_id_seed,
    )
    chunks = tuple(chunk_size or _PRODUCTION_CHUNK_SIZES)
    hash_seeds = tuple(python_hash_seed or _PRODUCTION_PYTHON_HASH_SEEDS)
    try:
        resolved = resolve_config(Phase1Config, [config, data_config])
        try:
            report = check_reproducibility(
                ReproducibilityRequest(
                    source=source,
                    sample_size=sample_size,
                    chunk_sizes=chunks,
                    python_hash_seeds=hash_seeds,
                    verify_all_source_entries=verify_all_source_entries,
                    output_path=None,
                ),
                resolved,
            )
        except ValueError as error:
            raise ArtifactIntegrityError("reproducibility verification failed") from error
        data = canonical_json_bytes(report)
        if output is not None:
            _publish_report(output, data)
    except SilentCascadeError as error:
        _render_error(error)
        raise typer.Exit(code=1) from None
    typer.echo(
        json.dumps(
            report.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app()
