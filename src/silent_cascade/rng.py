"""Reproducible global RNG capture for Python, NumPy, Torch CPU, and MPS."""

import copy
import hashlib
import hmac
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from itertools import permutations
from typing import Any, Literal

import numpy as np
import torch

from silent_cascade.env.config import SplitNamespace, SuiteName
from silent_cascade.env.episode import EpisodeVariant
from silent_cascade.errors import DoctorError, GenerationError
from silent_cascade.hashing import canonical_json_bytes
from silent_cascade.validation import JsonValue, StrictModel


@dataclass(frozen=True, slots=True)
class RngSnapshot:
    python_state: object
    numpy_state: tuple[Any, ...]
    torch_cpu_state: torch.Tensor
    torch_mps_state: torch.Tensor | None


class RngRoundTripReport(StrictModel):
    python_ok: bool
    numpy_ok: bool
    torch_cpu_ok: bool
    torch_mps_checked: bool
    torch_mps_ok: bool | None


class SeedStream(StrEnum):
    LABEL = "label"
    TEMPLATE = "template"
    STRUCTURE = "structure"
    NODE_PERMUTATION = "node_permutation"
    TERMINALS = "terminals"
    PRESENTATION = "presentation"
    TIMESTAMPS = "timestamps"
    TRACE_JITTER = "trace_jitter"
    RANDOM_BASELINE = "random_baseline"


@dataclass(frozen=True, slots=True)
class CounterSeedKey:
    generator_version: Literal["ofd-v1"]
    split_namespace: SplitNamespace
    suite: SuiteName
    root_seed: int
    cohort_index: int
    member_index: int
    stream: SeedStream
    attempt: int


@dataclass(frozen=True, slots=True)
class CounterSeed:
    token: str
    seed: int


@dataclass(frozen=True, slots=True)
class IndependentCounterSeedKey:
    generator_version: Literal["ofd-v1"]
    split_namespace: SplitNamespace
    suite: SuiteName
    root_seed: int
    episode_index: int
    stream: SeedStream
    attempt: int


@dataclass(frozen=True, slots=True)
class AllocationLabelKey:
    generator_version: Literal["ofd-v1"]
    split_namespace: SplitNamespace
    suite: SuiteName
    root_seed: int
    requested_path_length: int
    allocation_quartet_index: int


@dataclass(frozen=True, slots=True)
class PublicIdBatchKey:
    generator_version: Literal["ofd-v1"]
    split_namespace: SplitNamespace
    suite: SuiteName
    public_id_seed: int
    cohort_index: int
    accepted_attempt: int


@dataclass(frozen=True, slots=True)
class IndependentPublicIdKey:
    generator_version: Literal["ofd-v1"]
    split_namespace: SplitNamespace
    suite: SuiteName
    public_id_seed: int
    episode_index: int
    accepted_attempt: int


COHORT_SCOPED_STREAMS = frozenset(
    {SeedStream.LABEL, SeedStream.TEMPLATE, SeedStream.STRUCTURE, SeedStream.TIMESTAMPS}
)


def _validate_generator_identity(
    generator_version: object, split_namespace: object, suite: object
) -> None:
    if generator_version != "ofd-v1":
        raise ValueError("generator_version must be ofd-v1")
    if not isinstance(split_namespace, SplitNamespace):
        raise ValueError("split_namespace must be a SplitNamespace")
    if not isinstance(suite, SuiteName):
        raise ValueError("suite must be a SuiteName")


def _validate_exact_range(
    name: str, value: object, *, lower: int, upper: int | None = None
) -> None:
    if type(value) is not int or value < lower or (upper is not None and value >= upper):
        raise ValueError(f"{name} is outside its exact integer range")


def validate_seed_key_scope(key: CounterSeedKey) -> None:
    if not isinstance(key, CounterSeedKey):
        raise TypeError("key must be a CounterSeedKey")
    _validate_generator_identity(key.generator_version, key.split_namespace, key.suite)
    if type(key.member_index) is not int:
        raise ValueError("member_index must be an exact integer")
    if not isinstance(key.stream, SeedStream):
        raise ValueError("stream must be a SeedStream")
    _validate_exact_range("root_seed", key.root_seed, lower=0, upper=2**128)
    _validate_exact_range("cohort_index", key.cohort_index, lower=0)
    _validate_exact_range("attempt", key.attempt, lower=0, upper=1_000)
    cohort_scoped = key.stream in COHORT_SCOPED_STREAMS
    if (cohort_scoped and key.member_index != -1) or (
        not cohort_scoped and key.member_index not in range(4)
    ):
        raise ValueError("member_index does not match stream scope")


def seed_key_primitive(key: CounterSeedKey) -> dict[str, JsonValue]:
    if not isinstance(key, CounterSeedKey):
        raise TypeError("key must be a CounterSeedKey")
    return {
        "generator_version": key.generator_version,
        "split_namespace": key.split_namespace.value,
        "suite": key.suite.value,
        "root_seed": key.root_seed,
        "cohort_index": key.cohort_index,
        "member_index": key.member_index,
        "stream": key.stream.value,
        "attempt": key.attempt,
    }


def derive_counter_seed(key: CounterSeedKey) -> CounterSeed:
    validate_seed_key_scope(key)
    payload = canonical_json_bytes(
        {"domain": "silent-cascade/ofd-v1/counter-seed/v1", **seed_key_primitive(key)}
    )
    digest = hashlib.sha256(payload).digest()
    return CounterSeed(token=digest.hex(), seed=int.from_bytes(digest[:16], "big"))


def local_generator(key: CounterSeedKey) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(derive_counter_seed(key).seed))


def _validate_independent_seed_key(key: IndependentCounterSeedKey) -> None:
    if not isinstance(key, IndependentCounterSeedKey):
        raise TypeError("key must be an IndependentCounterSeedKey")
    _validate_generator_identity(key.generator_version, key.split_namespace, key.suite)
    if not isinstance(key.stream, SeedStream):
        raise ValueError("stream must be a SeedStream")
    if key.stream is SeedStream.LABEL:
        raise ValueError("independent counter seed keys cannot use LABEL")
    _validate_exact_range("root_seed", key.root_seed, lower=0, upper=2**128)
    _validate_exact_range("episode_index", key.episode_index, lower=0)
    _validate_exact_range("attempt", key.attempt, lower=0, upper=1_000)


def derive_independent_counter_seed(key: IndependentCounterSeedKey) -> CounterSeed:
    _validate_independent_seed_key(key)
    payload = canonical_json_bytes(
        {
            "domain": "silent-cascade/ofd-v1/independent-counter-seed/v1",
            "generator_version": key.generator_version,
            "split_namespace": key.split_namespace.value,
            "suite": key.suite.value,
            "root_seed": key.root_seed,
            "episode_index": key.episode_index,
            "stream": key.stream.value,
            "attempt": key.attempt,
        }
    )
    digest = hashlib.sha256(payload).digest()
    return CounterSeed(token=digest.hex(), seed=int.from_bytes(digest[:16], "big"))


def independent_local_generator(key: IndependentCounterSeedKey) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(derive_independent_counter_seed(key).seed))


def _validate_allocation_label_key(key: AllocationLabelKey) -> None:
    if not isinstance(key, AllocationLabelKey):
        raise TypeError("key must be an AllocationLabelKey")
    _validate_generator_identity(key.generator_version, key.split_namespace, key.suite)
    _validate_exact_range("root_seed", key.root_seed, lower=0, upper=2**128)
    _validate_exact_range("requested_path_length", key.requested_path_length, lower=1)
    _validate_exact_range("allocation_quartet_index", key.allocation_quartet_index, lower=0)


def _allocation_permutations() -> tuple[tuple[EpisodeVariant, ...], ...]:
    variants = tuple(
        sorted(
            (
                EpisodeVariant.POSITIVE,
                EpisodeVariant.POSITIVE,
                EpisodeVariant.SAFE_NEGATIVE,
                EpisodeVariant.DISCONNECTED_NEGATIVE,
            ),
            key=lambda variant: variant.value,
        )
    )
    result: list[tuple[EpisodeVariant, ...]] = []
    for candidate in permutations(variants):
        if candidate not in result:
            result.append(candidate)
    return tuple(result)


ALLOCATION_PERMUTATIONS = _allocation_permutations()


def allocate_independent_variants(key: AllocationLabelKey) -> tuple[EpisodeVariant, ...]:
    _validate_allocation_label_key(key)
    payload = canonical_json_bytes(
        {
            "domain": "silent-cascade/ofd-v1/allocation-label/v1",
            "generator_version": key.generator_version,
            "split_namespace": key.split_namespace.value,
            "suite": key.suite.value,
            "root_seed": key.root_seed,
            "requested_path_length": key.requested_path_length,
            "allocation_quartet_index": key.allocation_quartet_index,
        }
    )
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    return ALLOCATION_PERMUTATIONS[int(generator.integers(len(ALLOCATION_PERMUTATIONS)))]


def validate_public_id_key(key: PublicIdBatchKey) -> None:
    if not isinstance(key, PublicIdBatchKey):
        raise TypeError("key must be a PublicIdBatchKey")
    _validate_generator_identity(key.generator_version, key.split_namespace, key.suite)
    _validate_exact_range("public_id_seed", key.public_id_seed, lower=0, upper=2**128)
    _validate_exact_range("cohort_index", key.cohort_index, lower=0)
    _validate_exact_range("accepted_attempt", key.accepted_attempt, lower=0, upper=1_000)


def _public_id_hmac_key(public_id_seed: int) -> bytes:
    return b"silent-cascade/ofd-v1/public-id-key/v1\0" + public_id_seed.to_bytes(16, "big")


def _version_4_uuid(raw: bytes) -> str:
    uuid_bytes = bytearray(raw[:16])
    uuid_bytes[6] = (uuid_bytes[6] & 0x0F) | 0x40
    uuid_bytes[8] = (uuid_bytes[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(uuid_bytes)))


def allocate_public_ids(key: PublicIdBatchKey) -> tuple[str, str, str, str]:
    validate_public_id_key(key)
    hmac_key = _public_id_hmac_key(key.public_id_seed)
    context: dict[str, JsonValue] = {
        "generator_version": key.generator_version,
        "split_namespace": key.split_namespace.value,
        "suite": key.suite.value,
        "cohort_index": key.cohort_index,
        "attempt": key.accepted_attempt,
    }
    candidates = [
        _version_4_uuid(
            hmac.new(
                hmac_key,
                canonical_json_bytes(
                    {
                        "domain": "silent-cascade/ofd-v1/id-pool/v1",
                        **context,
                        "counter": counter,
                    }
                ),
                hashlib.sha256,
            ).digest()
        )
        for counter in range(4)
    ]
    if len(set(candidates)) != 4:
        raise GenerationError("public ID collision")
    assigned = sorted(
        candidates,
        key=lambda public_id: (
            hmac.new(
                hmac_key,
                canonical_json_bytes(
                    {
                        "domain": "silent-cascade/ofd-v1/id-assignment/v1",
                        **context,
                        "public_id": public_id,
                    }
                ),
                hashlib.sha256,
            ).digest(),
            public_id,
        ),
    )
    return assigned[0], assigned[1], assigned[2], assigned[3]


def _validate_independent_public_id_key(key: IndependentPublicIdKey) -> None:
    if not isinstance(key, IndependentPublicIdKey):
        raise TypeError("key must be an IndependentPublicIdKey")
    _validate_generator_identity(key.generator_version, key.split_namespace, key.suite)
    _validate_exact_range("public_id_seed", key.public_id_seed, lower=0, upper=2**128)
    _validate_exact_range("episode_index", key.episode_index, lower=0)
    _validate_exact_range("accepted_attempt", key.accepted_attempt, lower=0, upper=1_000)


def allocate_independent_public_id(key: IndependentPublicIdKey) -> str:
    _validate_independent_public_id_key(key)
    hmac_key = _public_id_hmac_key(key.public_id_seed)
    message = canonical_json_bytes(
        {
            "domain": "silent-cascade/ofd-v1/independent-id/v1",
            "generator_version": key.generator_version,
            "split_namespace": key.split_namespace.value,
            "suite": key.suite.value,
            "episode_index": key.episode_index,
            "attempt": key.accepted_attempt,
        }
    )
    return _version_4_uuid(hmac.new(hmac_key, message, hashlib.sha256).digest())


def mps_rng_state_supported() -> bool:
    return bool(
        torch.backends.mps.is_available()
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "get_rng_state")
        and hasattr(torch.mps, "set_rng_state")
    )


def seed_all(seed: int) -> None:
    if type(seed) is not int:
        raise TypeError("seed must be an int")
    if not 0 <= seed <= 2**64 - 1:
        raise ValueError("seed must be in range 0..2**64-1")

    def apply_seed() -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        # torch.manual_seed also seeds the MPS default generator when present.
        torch.manual_seed(seed)

    _apply_transactionally("seed global RNGs", apply_seed)


def snapshot_global_rng() -> RngSnapshot:
    mps_state = torch.mps.get_rng_state().clone() if mps_rng_state_supported() else None
    return RngSnapshot(
        python_state=copy.deepcopy(random.getstate()),
        numpy_state=copy.deepcopy(np.random.get_state()),
        torch_cpu_state=torch.get_rng_state().clone(),
        torch_mps_state=mps_state,
    )


def restore_global_rng(snapshot: RngSnapshot) -> None:
    _prevalidate_snapshot(snapshot)
    _apply_transactionally("restore global RNGs", lambda: _apply_snapshot(snapshot))


def _prevalidate_snapshot(snapshot: RngSnapshot) -> None:
    if snapshot.torch_mps_state is not None and not mps_rng_state_supported():
        raise DoctorError("MPS RNG state cannot be restored on this runtime")

    isolated_python = random.Random()
    isolated_python.setstate(copy.deepcopy(snapshot.python_state))

    isolated_numpy = np.random.RandomState()
    isolated_numpy.set_state(copy.deepcopy(snapshot.numpy_state))

    isolated_torch_cpu = torch.Generator(device="cpu")
    isolated_torch_cpu.set_state(snapshot.torch_cpu_state.clone())


def _apply_snapshot(snapshot: RngSnapshot) -> None:
    random.setstate(copy.deepcopy(snapshot.python_state))
    np.random.set_state(copy.deepcopy(snapshot.numpy_state))
    torch.set_rng_state(snapshot.torch_cpu_state.clone())
    if snapshot.torch_mps_state is not None:
        torch.mps.set_rng_state(snapshot.torch_mps_state.clone())


def _apply_transactionally(operation: str, mutation: Callable[[], None]) -> None:
    caller_snapshot = snapshot_global_rng()
    try:
        mutation()
    except Exception as original_error:
        rollback_errors = _rollback_global_rng(caller_snapshot)
        if rollback_errors:
            rollback_detail = "; ".join(rollback_errors)
            raise DoctorError(
                f"{operation} failed and RNG rollback also failed: "
                f"{type(original_error).__name__}: {original_error}; {rollback_detail}",
                context={
                    "operation": operation,
                    "original_error_type": type(original_error).__name__,
                    "original_error": str(original_error),
                    "rollback_errors": rollback_errors,
                },
            ) from original_error
        raise


def _rollback_global_rng(snapshot: RngSnapshot) -> list[str]:
    """Best-effort rollback that never re-enters the public restore path."""

    rollback_errors: list[str] = []
    for backend, restore in (
        ("Python", lambda: random.setstate(copy.deepcopy(snapshot.python_state))),
        ("NumPy", lambda: np.random.set_state(copy.deepcopy(snapshot.numpy_state))),
        ("Torch CPU", lambda: torch.set_rng_state(snapshot.torch_cpu_state.clone())),
    ):
        try:
            restore()
        except Exception as error:
            rollback_errors.append(f"{backend}: {type(error).__name__}: {error}")

    if snapshot.torch_mps_state is not None:
        try:
            if not mps_rng_state_supported():
                raise DoctorError("MPS RNG state cannot be restored on this runtime")
            torch.mps.set_rng_state(snapshot.torch_mps_state.clone())
        except Exception as error:
            rollback_errors.append(f"MPS: {type(error).__name__}: {error}")

    return rollback_errors


def verify_rng_round_trip(*, include_mps: bool) -> RngRoundTripReport:
    outer = snapshot_global_rng()
    try:
        seed_all(0x5A17)
        checkpoint = snapshot_global_rng()
        first_python = random.random()
        first_numpy = float(np.random.random())
        first_cpu = torch.rand(8)
        first_mps = (
            torch.rand(8, device="mps").cpu() if include_mps and mps_rng_state_supported() else None
        )

        restore_global_rng(checkpoint)
        second_python = random.random()
        second_numpy = float(np.random.random())
        second_cpu = torch.rand(8)
        second_mps = (
            torch.rand(8, device="mps").cpu() if include_mps and mps_rng_state_supported() else None
        )
        if include_mps and torch.backends.mps.is_available():
            torch.mps.synchronize()

        mps_checked = include_mps and mps_rng_state_supported()
        mps_ok = (
            torch.equal(first_mps, second_mps)
            if first_mps is not None and second_mps is not None
            else None
        )
        return RngRoundTripReport(
            python_ok=first_python == second_python,
            numpy_ok=first_numpy == second_numpy,
            torch_cpu_ok=torch.equal(first_cpu, second_cpu),
            torch_mps_checked=mps_checked,
            torch_mps_ok=mps_ok,
        )
    finally:
        restore_global_rng(outer)
