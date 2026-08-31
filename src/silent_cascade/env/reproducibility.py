"""Fail-closed order, chunk, and fresh-interpreter regeneration checks."""

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import Field

from silent_cascade.config import ResolvedConfig
from silent_cascade.env.config import Phase1Config, SplitNamespace
from silent_cascade.env.episode import (
    CorpusDigestEntry,
    EpisodeBundle,
    corpus_sha256,
    episode_sha256,
)
from silent_cascade.env.generator import (
    PHASE1_GATE_ALLOCATION,
    IndependentAllocation,
    IndependentEpisodeRequest,
    generate_independent_episode,
    iter_independent_requests,
    validate_phase1_gate_allocation,
)
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes
from silent_cascade.logging.manifest import EpisodeManifest, EpisodeManifestEntry, load_manifest
from silent_cascade.provenance import (
    EvidenceProvenance,
    EvidenceProvenanceCollector,
    collect_evidence_provenance,
    public_id_seed_sha256,
)
from silent_cascade.rng import AllocationLabelKey, allocate_independent_variants
from silent_cascade.validation import StrictModel


class ManifestReproducibilitySource(StrictModel):
    mode: Literal["manifest"] = "manifest"
    manifest_path: Path


class IndependentAllocationReproducibilitySource(StrictModel):
    mode: Literal["independent_allocation"] = "independent_allocation"
    allocation_id: str = Field(min_length=1)
    root_seed: int
    public_id_seed: int


class IndependentSourceDescriptor(StrictModel):
    schema_version: Literal["phase1-independent-source-v1"]
    allocation_id: str
    allocation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_namespace: SplitNamespace
    root_seed: int
    public_id_seed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


type ReproducibilitySource = Annotated[
    ManifestReproducibilitySource | IndependentAllocationReproducibilitySource,
    Field(discriminator="mode"),
]


class ReproducibilityRequest(StrictModel):
    source: ReproducibilitySource
    sample_size: int = Field(gt=0)
    chunk_sizes: tuple[int, ...]
    python_hash_seeds: tuple[int, ...]
    verify_all_source_entries: bool
    output_path: Path | None = None


class ReproducibilityReport(StrictModel):
    schema_version: Literal["phase1-reproducibility-v1"]
    source_mode: Literal["manifest", "independent_allocation"]
    source_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_size: int
    verified_source_entries: int
    modes: tuple[str, ...]
    chunk_sizes: tuple[int, ...]
    python_hash_seeds: tuple[int, ...]
    reference_corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_membership_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mismatch_count: Literal[0]
    provenance: EvidenceProvenance
    passed: bool


class FreshProcessRunner(Protocol):
    def __call__(self, work_order_bytes: bytes, python_hash_seed: int) -> bytes: ...


@dataclass(frozen=True, slots=True)
class ReproducibilityDependencies:
    production_mode: bool
    independent_allocation: IndependentAllocation
    collect_provenance: EvidenceProvenanceCollector
    load_verified_manifest: Callable[[Path], EpisodeManifest]
    regenerate_manifest_entry: Callable[..., EpisodeBundle]
    iter_independent: Callable[..., Iterator[IndependentEpisodeRequest]]
    generate_independent: Callable[..., EpisodeBundle]
    run_fresh_process: FreshProcessRunner

    @classmethod
    def for_test(
        cls,
        allocation: IndependentAllocation,
        provenance: EvidenceProvenance,
        *,
        collect_provenance: EvidenceProvenanceCollector,
        load_verified_manifest: Callable[[Path], EpisodeManifest],
        iter_independent: Callable[..., Iterator[IndependentEpisodeRequest]],
        generate_independent: Callable[..., EpisodeBundle],
    ) -> "ReproducibilityDependencies":
        from silent_cascade.env.services import regenerate_entry

        if (
            allocation.split_namespace is not SplitNamespace.DEBUG
            or not allocation.allocation_id.startswith("test-")
        ):
            raise ValueError("test dependencies require a test- DEBUG allocation")
        if provenance.source_dirty or provenance.allocation_id != allocation.allocation_id:
            raise ValueError("test provenance must be clean and bound to its allocation")
        return cls(
            False,
            allocation,
            collect_provenance,
            load_verified_manifest,
            regenerate_entry,
            iter_independent,
            generate_independent,
            _run_fresh_process,
        )


def _run_fresh_process(work_order_bytes: bytes, python_hash_seed: int) -> bytes:
    descriptor, raw_path = tempfile.mkstemp(prefix="silent-cascade-repro-", suffix=".json")
    path = Path(raw_path)
    try:
        os.write(descriptor, work_order_bytes)
        os.close(descriptor)
        environment = {**os.environ, "PYTHONHASHSEED": str(python_hash_seed)}
        result = subprocess.run(
            [sys.executable, "-m", "silent_cascade.env.reproducibility", "--work-order", str(path)],
            check=False,
            capture_output=True,
            env=environment,
        )
        if result.returncode != 0:
            raise ValueError(
                f"fresh reproducibility process failed: {result.stderr.decode('utf-8', 'replace')}"
            )
        return result.stdout
    finally:
        with suppress(OSError):
            os.close(descriptor)
        path.unlink(missing_ok=True)


PRODUCTION_REPRODUCIBILITY_DEPENDENCIES = ReproducibilityDependencies(
    True,
    PHASE1_GATE_ALLOCATION,
    collect_evidence_provenance,
    load_manifest,
    lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("services unavailable")),
    iter_independent_requests,
    generate_independent_episode,
    _run_fresh_process,
)


def _request_key(request: IndependentEpisodeRequest) -> tuple[str, int, int]:
    return request.suite.value, request.requested_path_length, request.episode_index


def _entry_for(request: IndependentEpisodeRequest, bundle: EpisodeBundle) -> CorpusDigestEntry:
    return CorpusDigestEntry(bundle.public.init.episode_public_id, episode_sha256(bundle))


def _independent_descriptor(
    allocation: IndependentAllocation,
    source: IndependentAllocationReproducibilitySource,
    resolved: ResolvedConfig[Phase1Config],
    provenance: EvidenceProvenance,
) -> IndependentSourceDescriptor:
    return IndependentSourceDescriptor(
        schema_version="phase1-independent-source-v1",
        allocation_id=allocation.allocation_id,
        allocation_sha256=sha256_bytes(canonical_json_bytes(allocation)),
        split_namespace=allocation.split_namespace,
        root_seed=source.root_seed,
        public_id_seed_sha256=public_id_seed_sha256(source.public_id_seed),
        config_sha256=resolved.sha256,
        generator_source_sha256=provenance.generator_source.sha256,
    )


def _validate_independent_provenance(
    provenance: EvidenceProvenance,
    *,
    allocation_id: str,
    split_namespace: SplitNamespace,
    root_seed: int,
    public_id_seed: int,
    config_sha256: str,
) -> None:
    if (
        provenance.generation_mode != "independent"
        or provenance.allocation_id != allocation_id
        or provenance.split_namespace is not split_namespace
        or provenance.root_seed != root_seed
        or provenance.public_id_seed_sha256 != public_id_seed_sha256(public_id_seed)
        or provenance.config_sha256 != config_sha256
        or provenance.foundation_model_calls != 0
    ):
        raise ValueError("reproducibility provenance mismatch")


def _independent_sample(
    requests: tuple[IndependentEpisodeRequest, ...], source_payload_sha256: str, count: int
) -> tuple[IndependentEpisodeRequest, ...]:
    strata: dict[tuple[str, int], list[IndependentEpisodeRequest]] = {}
    for request in requests:
        strata.setdefault((request.suite.value, request.requested_path_length), []).append(request)
    per_stratum, remainder = divmod(min(count, len(requests)), len(strata))
    selected: list[IndependentEpisodeRequest] = []
    for index, values in enumerate(strata.values()):
        quota = per_stratum + (index < remainder)
        ranked = sorted(
            values,
            key=lambda request: (
                sha256_bytes(
                    canonical_json_bytes(
                        {
                            "domain": "silent-cascade/ofd-v1/repro-sample-rank/v1",
                            "source_payload_sha256": source_payload_sha256,
                            "suite": request.suite.value,
                            "requested_path_length": request.requested_path_length,
                            "episode_index": request.episode_index,
                        }
                    )
                ),
                request.episode_index,
            ),
        )
        if len(ranked) < quota:
            raise ValueError("source stratum lacks its reproducibility quota")
        selected.extend(ranked[:quota])
    return tuple(selected)


def production_sample_quotas() -> tuple[int, ...]:
    """The fixed 1,000-sample allocation is 62 each plus eight leading 63s."""
    return tuple(63 if index < 8 else 62 for index in range(16))


def _membership_hash(
    source_payload_sha256: str, requests: tuple[IndependentEpisodeRequest, ...]
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "silent-cascade/ofd-v1/repro-sample-membership/v1",
                "source_payload_sha256": source_payload_sha256,
                "coordinates": [
                    {
                        "suite": request.suite.value,
                        "requested_path_length": request.requested_path_length,
                        "episode_index": request.episode_index,
                    }
                    for request in requests
                ],
            }
        )
    )


def _manifest_sample(
    manifest: EpisodeManifest, source_payload_sha256: str, count: int
) -> tuple[EpisodeManifestEntry, ...]:
    strata: dict[tuple[str, int], list[EpisodeManifestEntry]] = {}
    for entry in manifest.entries:
        strata.setdefault((entry.suite.value, entry.requested_path_length), []).append(entry)
    per_stratum, remainder = divmod(min(count, manifest.episode_count), len(strata))
    selected: list[EpisodeManifestEntry] = []
    for index, values in enumerate(strata.values()):
        quota = per_stratum + (index < remainder)
        ranked = sorted(
            values,
            key=lambda entry: sha256_bytes(
                canonical_json_bytes(
                    {
                        "domain": "silent-cascade/ofd-v1/repro-manifest-sample-rank/v1",
                        "source_payload_sha256": source_payload_sha256,
                        "suite": entry.suite.value,
                        "requested_path_length": entry.requested_path_length,
                        "coordinate": entry.coordinate.model_dump(mode="json"),
                        "episode_public_id": entry.episode_public_id,
                    }
                )
            ),
        )
        if len(ranked) < quota:
            raise ValueError("manifest stratum lacks its reproducibility quota")
        selected.extend(ranked[:quota])
    return tuple(selected)


def _manifest_membership_hash(
    source_payload_sha256: str, entries: tuple[EpisodeManifestEntry, ...]
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "domain": "silent-cascade/ofd-v1/repro-manifest-sample-membership/v1",
                "source_payload_sha256": source_payload_sha256,
                "entries": [
                    {
                        "episode_public_id": entry.episode_public_id,
                        "coordinate": entry.coordinate.model_dump(mode="json"),
                    }
                    for entry in entries
                ],
            }
        )
    )


def _fresh_work_order(
    config: Phase1Config,
    source: IndependentAllocationReproducibilitySource,
    selected: tuple[IndependentEpisodeRequest, ...],
    expected: dict[tuple[str, int, int], CorpusDigestEntry],
) -> bytes:
    return canonical_json_bytes(
        {
            "config": config.model_dump(mode="json"),
            "public_id_seed": source.public_id_seed,
            "entries": [
                {
                    "split_namespace": request.split_namespace.value,
                    "suite": request.suite.value,
                    "root_seed": request.root_seed,
                    "episode_index": request.episode_index,
                    "requested_path_length": request.requested_path_length,
                    "allocation_quartet_index": request.allocation_quartet_index,
                    "quartet_member_index": list(
                        allocate_independent_variants(
                            AllocationLabelKey(
                                "ofd-v1",
                                request.split_namespace,
                                request.suite,
                                request.root_seed,
                                request.requested_path_length,
                                request.allocation_quartet_index,
                            )
                        )
                    ).index(request.variant),
                    "episode_public_id": expected[_request_key(request)].episode_public_id,
                    "episode_sha256": expected[_request_key(request)].episode_sha256,
                }
                for request in selected
            ],
        }
    )


def check_reproducibility(
    request: ReproducibilityRequest,
    resolved: ResolvedConfig[Phase1Config],
    *,
    deps: ReproducibilityDependencies = PRODUCTION_REPRODUCIBILITY_DEPENDENCIES,
) -> ReproducibilityReport:
    """Regenerate an authenticated independent allocation under every execution order."""
    if isinstance(request.source, ManifestReproducibilitySource):
        manifest = deps.load_verified_manifest(request.source.manifest_path)
        embedded = manifest.provenance
        current = deps.collect_provenance(
            resolved,
            repo_root=Path.cwd(),
            generation_mode=embedded.generation_mode,
            allocation_id=embedded.allocation_id,
            split_namespace=embedded.split_namespace,
            root_seed=embedded.root_seed,
            public_id_seed=manifest.public_id_seed,
            analysis_seeds={},
        )
        if current != embedded:
            raise ValueError("manifest provenance does not match current reproducibility inputs")
        source_hash = sha256_bytes(canonical_json_bytes(manifest))
        source_entries = tuple(manifest.entries)
        selected = _manifest_sample(manifest, source_hash, request.sample_size)
        to_verify = source_entries if request.verify_all_source_entries else selected
        for entry in to_verify:
            bundle = deps.regenerate_manifest_entry(resolved.config, manifest, entry)
            if (
                bundle.public.init.episode_public_id != entry.episode_public_id
                or episode_sha256(bundle) != entry.episode_sha256
            ):
                raise ValueError("manifest entry regeneration mismatch")
        for ordered in (selected, tuple(reversed(selected))):
            for entry in ordered:
                bundle = deps.regenerate_manifest_entry(resolved.config, manifest, entry)
                if episode_sha256(bundle) != entry.episode_sha256:
                    raise ValueError("manifest order reproducibility mismatch")
        reference = corpus_sha256(
            (
                CorpusDigestEntry(entry.episode_public_id, entry.episode_sha256)
                for entry in source_entries
            ),
            expected_count=len(source_entries),
        )
        return ReproducibilityReport(
            schema_version="phase1-reproducibility-v1",
            source_mode="manifest",
            source_payload_sha256=source_hash,
            sample_size=len(selected),
            verified_source_entries=len(to_verify),
            modes=("forward", "reverse", "chunked"),
            chunk_sizes=request.chunk_sizes,
            python_hash_seeds=request.python_hash_seeds,
            reference_corpus_sha256=reference,
            sample_membership_sha256=_manifest_membership_hash(source_hash, selected),
            mismatch_count=0,
            provenance=current,
            passed=True,
        )
    if not isinstance(request.source, IndependentAllocationReproducibilitySource):
        raise TypeError("unsupported reproducibility source")
    source = request.source
    allocation = deps.independent_allocation
    if source.allocation_id != allocation.allocation_id:
        raise ValueError("source allocation does not match bound dependencies")
    if deps.production_mode:
        validate_phase1_gate_allocation(allocation, resolved.config)
        if allocation.allocation_id != "phase1-independent-gate-v1":
            raise ValueError("production requires the canonical independent gate allocation")
    elif (
        allocation.split_namespace is not SplitNamespace.DEBUG
        or not allocation.allocation_id.startswith("test-")
    ):
        raise ValueError("non-production dependencies require a test- DEBUG allocation")
    provenance = deps.collect_provenance(
        resolved,
        repo_root=Path.cwd(),
        generation_mode="independent",
        allocation_id=allocation.allocation_id,
        split_namespace=allocation.split_namespace,
        root_seed=source.root_seed,
        public_id_seed=source.public_id_seed,
        analysis_seeds={},
    )
    _validate_independent_provenance(
        provenance,
        allocation_id=allocation.allocation_id,
        split_namespace=allocation.split_namespace,
        root_seed=source.root_seed,
        public_id_seed=source.public_id_seed,
        config_sha256=resolved.sha256,
    )
    descriptor = _independent_descriptor(allocation, source, resolved, provenance)
    payload_hash = sha256_bytes(canonical_json_bytes(descriptor))
    requests = tuple(deps.iter_independent(allocation, source.root_seed))
    expected: dict[tuple[str, int, int], CorpusDigestEntry] = {}
    for item in requests:
        bundle = deps.generate_independent(resolved.config, item, source.public_id_seed)
        expected[_request_key(item)] = _entry_for(item, bundle)
    reference = corpus_sha256(
        (expected[_request_key(item)] for item in requests), expected_count=len(requests)
    )
    selected = _independent_sample(requests, payload_hash, request.sample_size)
    for mode, ordered in (("forward", selected), ("reverse", tuple(reversed(selected)))):
        for item in ordered:
            if (
                _entry_for(
                    item, deps.generate_independent(resolved.config, item, source.public_id_seed)
                )
                != expected[_request_key(item)]
            ):
                raise ValueError(f"reproducibility mismatch in {mode}")
    for size in request.chunk_sizes:
        if type(size) is not int or size <= 0:
            raise ValueError("chunk sizes must be positive exact integers")
        for start in range(0, len(selected), size):
            for item in selected[start : start + size]:
                if (
                    _entry_for(
                        item,
                        deps.generate_independent(resolved.config, item, source.public_id_seed),
                    )
                    != expected[_request_key(item)]
                ):
                    raise ValueError("reproducibility mismatch in chunked generation")
    work_order = _fresh_work_order(resolved.config, source, selected, expected)
    for hash_seed in request.python_hash_seeds:
        raw = deps.run_fresh_process(work_order, hash_seed)
        try:
            output = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("fresh process emitted malformed reproducibility output") from error
        if output.get("reference_corpus_sha256") != corpus_sha256(
            (expected[_request_key(item)] for item in selected), expected_count=len(selected)
        ):
            raise ValueError("fresh process reproducibility mismatch")
    return ReproducibilityReport(
        schema_version="phase1-reproducibility-v1",
        source_mode="independent_allocation",
        source_payload_sha256=payload_hash,
        sample_size=len(selected),
        verified_source_entries=len(requests),
        modes=("forward", "reverse", "chunked", "fresh_process"),
        chunk_sizes=request.chunk_sizes,
        python_hash_seeds=request.python_hash_seeds,
        reference_corpus_sha256=reference,
        sample_membership_sha256=_membership_hash(payload_hash, selected),
        mismatch_count=0,
        provenance=provenance,
        passed=True,
    )


def _worker(path: Path) -> None:
    try:
        payload = json.loads(path.read_bytes())
        config_payload = payload["config"]
        phase1_gate = config_payload["data"]["phase1_gate"]
        for name, values in phase1_gate.items():
            phase1_gate[name] = {int(key): value for key, value in values.items()}
        config = Phase1Config.model_validate(config_payload)
        entries = []
        for value in payload["entries"]:
            split = SplitNamespace(value["split_namespace"])
            suite = __import__("silent_cascade.env.config", fromlist=["SuiteName"]).SuiteName(
                value["suite"]
            )
            variants = allocate_independent_variants(
                AllocationLabelKey(
                    "ofd-v1",
                    split,
                    suite,
                    value["root_seed"],
                    value["requested_path_length"],
                    value["allocation_quartet_index"],
                )
            )
            item = IndependentEpisodeRequest(
                split,
                suite,
                value["root_seed"],
                value["episode_index"],
                value["requested_path_length"],
                variants[value["quartet_member_index"]],
                value["allocation_quartet_index"],
            )
            bundle = generate_independent_episode(config, item, payload["public_id_seed"])
            entry = _entry_for(item, bundle)
            if (
                entry.episode_public_id != value["episode_public_id"]
                or entry.episode_sha256 != value["episode_sha256"]
            ):
                raise ValueError("fresh process entry mismatch")
            entries.append(entry)
        print(
            json.dumps(
                {"reference_corpus_sha256": corpus_sha256(entries, expected_count=len(entries))},
                sort_keys=True,
            )
        )
    finally:
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--work-order":
        raise SystemExit(2)
    _worker(Path(sys.argv[2]))
