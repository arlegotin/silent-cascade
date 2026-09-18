"""Scientific scanners consume authenticated leases without exporting deferred paths."""

import re
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path

from silent_cascade.hashing import canonical_json_bytes, sha256_bytes


def logical_root(root: Path, evidence_context):
    """A caller's path cannot silently name a different session or leased corpus."""
    absolute = root.absolute()
    run = evidence_context.run_dir
    if not absolute.is_relative_to(run) or ".." in absolute.parts:
        raise ValueError("evidence context logical root differs")
    return absolute.relative_to(run).as_posix()


@contextmanager
def evidence_path(run_dir: Path, name: str, *, evidence_context=None):
    from silent_cascade.report.pilot_artifacts import child

    path = child(run_dir, name)
    if evidence_context is not None and logical_root(run_dir, evidence_context) != ".":
        raise ValueError("evidence path requires the session run root")
    if evidence_context is None or path.exists():
        yield path
    else:
        with evidence_context._leased({"path": name}) as lease:
            yield child(lease.local_root, name)


def _verify_journal_segment(run_dir, entries, commitment_sha256, evidence_context):
    from silent_cascade.archive.producer import JournalSegmentCommit
    from silent_cascade.report.pilot_artifacts import _decode_json
    from silent_cascade.train.pilot_data import _read_pilot_bytes

    if entries:
        links = {}
        for entry in entries:
            match = re.fullmatch(r"journal-([0-9a-f]{64})\.json", entry.path)
            if match is None or match[1] != entry.sha256:
                raise ValueError("journal segment member filename/hash differs")
            with evidence_path(run_dir, entry.path, evidence_context=evidence_context) as path:
                raw = _read_pilot_bytes(path)
            if len(raw) != entry.bytes or sha256_bytes(raw) != entry.sha256:
                raise ValueError("journal segment raw member differs")
            record = _decode_json(raw)
            if not isinstance(record, dict) or "prior" not in record:
                raise ValueError("journal segment predecessor missing")
            links[entry.sha256] = record["prior"]
        heads = links.keys() - set(links.values())
        if len(heads) != 1 or len(links) != len(entries):
            raise ValueError("journal segment has gaps/cycles/extras")
        head = next(iter(heads))
        cursor, chain = head, []
        while cursor in links:
            if cursor in chain:
                raise ValueError("journal segment cycle")
            chain.append(cursor)
            cursor = links[cursor]
        if len(chain) != len(links):
            raise ValueError("journal segment has unconsumed members")
        commit = JournalSegmentCommit(
            base_head=cursor, sealed_head=head, record_sha256s=tuple(chain)
        )
        if sha256_bytes(canonical_json_bytes(commit)) != commitment_sha256:
            raise ValueError("journal segment commitment differs")
    else:
        raise ValueError("empty journal segment")


def iter_journal_records(run_dir: Path, head: str | None, *, evidence_context=None):
    from silent_cascade.report.pilot_artifacts import _decode_json, child
    from silent_cascade.train.pilot_data import _read_pilot_bytes

    if evidence_context is not None and logical_root(run_dir, evidence_context) != ".":
        raise ValueError("journal context run root differs")
    cursor, seen, segments = head, set(), set()
    while cursor is not None:
        if (
            not isinstance(cursor, str)
            or not re.fullmatch(r"[0-9a-f]{64}", cursor)
            or cursor in seen
        ):
            raise ValueError("cyclic or invalid pilot journal")
        seen.add(cursor)
        name = f"journal-{cursor}.json"
        if evidence_context is not None and not (run_dir / name).exists():
            from silent_cascade.archive.catalog import _opened_unit, iter_unit_files

            segment = None
            with evidence_context._leased({"path": name}) as lease:
                if lease.ref.unit_id not in segments:
                    with _opened_unit(lease.metadata_root, lease.ref) as (_, manifest):
                        if manifest.kind != "journal" or manifest.logical_root != ".":
                            raise ValueError("journal owner kind/root differs")
                        entries = tuple(iter_unit_files(lease.metadata_root, lease.ref))
                        if len(entries) != lease.ref.file_count:
                            raise ValueError("journal segment count differs")
                        segment = (
                            lease.ref.unit_id,
                            entries,
                            manifest.identity.evidence_identity_sha256,
                        )
                raw = _read_pilot_bytes(lease.local_root / name)
            if segment is not None:
                unit_id, entries, commitment = segment
                _verify_journal_segment(run_dir, entries, commitment, evidence_context)
                segments.add(unit_id)
        else:
            raw = _read_pilot_bytes(run_dir / name)
        if sha256_bytes(raw) != cursor:
            raise ValueError("pilot journal hash mismatch")
        record = _decode_json(raw)
        if (
            not isinstance(record, dict)
            or record.get("kind") not in {"update", "validation"}
            or not re.fullmatch(r"attempt-[0-9a-f]{32}", record.get("attempt", ""))
            or type(record.get("global_step")) is not int
            or record["global_step"] <= 0
            or "prior" not in record
            or (
                record["prior"] is not None
                and (
                    not isinstance(record["prior"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", record["prior"])
                )
            )
        ):
            raise ValueError("invalid recovery journal")
        artifacts = record.get("artifacts", {})
        if not isinstance(artifacts, dict) or (record["kind"] == "validation" and not artifacts):
            raise ValueError("missing committed validation artifacts")
        for path, digest in artifacts.items():
            relative = child(run_dir, path).relative_to(run_dir)
            if (
                relative.parts[0] != record["attempt"]
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ValueError("committed validation artifact mismatch")
        yield record
        cursor = record["prior"]


def read_training_envelope(run_dir: Path, path: Path, *, evidence_context=None):
    from silent_cascade.report.pilot_artifacts import read_json, validate_training_result
    from silent_cascade.train.pilot_artifact_index import parse_training_envelope

    name = path.absolute().relative_to(run_dir.absolute()).as_posix()
    with evidence_path(run_dir, name, evidence_context=evidence_context) as local:
        envelope = parse_training_envelope(read_json(local))
    projection = envelope.model_dump(mode="json", exclude={"schema_version", "artifact_index"})
    validate_training_result(projection | {"artifact_hashes": {}})
    return envelope


def verify_training_inventory(*, run_dir: Path, envelope, evidence_context):
    from silent_cascade.eval.artifacts import read_evaluation_artifact
    from silent_cascade.report.pilot_artifacts import validate_training_result
    from silent_cascade.train.pilot_artifact_index import iter_artifact_index

    projection = envelope.model_dump(mode="json", exclude={"schema_version", "artifact_index"})
    validate_training_result(projection | {"artifact_hashes": {}})
    for name, digest in iter_artifact_index(
        run_dir, envelope.artifact_index, evidence_context=evidence_context
    ):
        with evidence_path(run_dir, name, evidence_context=evidence_context) as path:
            if sha256_bytes(read_evaluation_artifact(path)) != digest:
                raise ValueError(f"artifact integrity mismatch: {name}")
    for key in ("latest_weights", "selected_weights", "selected_checkpoint"):
        descriptor = projection[key]
        if descriptor is not None:
            with evidence_path(
                run_dir, descriptor["path"], evidence_context=evidence_context
            ) as path:
                if sha256_bytes(read_evaluation_artifact(path)) != descriptor["sha256"]:
                    raise ValueError("training descriptor hash differs")


@contextmanager
def evaluation_header(root: Path, *, evidence_context=None):
    from silent_cascade.report.pilot_artifacts import load_evaluation_header

    name = None if evidence_context is None else logical_root(root, evidence_context)
    metadata = nullcontext(None) if evidence_context is None else evidence_context.metadata(name)
    with metadata as lease:
        if lease is not None and (
            lease.ref.logical_root != name or lease.ref.kind != "evaluation_metadata"
        ):
            raise ValueError("evaluation metadata lease identity differs")
        metadata_root = root if lease is None else lease.local_root / name
        yield load_evaluation_header(metadata_root), metadata_root


def iter_evaluation_episodes(
    root: Path, *, evidence_context=None, expected_header=None, plot_examples=False
):
    from silent_cascade.report.pilot_artifacts import (
        EvaluationScanAccumulator,
        finish_evaluation_scan,
        iter_evaluation_rows,
        verify_evaluation_episode,
    )

    name = None if evidence_context is None else logical_root(root, evidence_context)
    with evaluation_header(root, evidence_context=evidence_context) as (header, metadata_root):
        if expected_header is not None and header != expected_header:
            raise ValueError("evaluation header changed between reads")
        accumulator = EvaluationScanAccumulator()
        decoded = set()
        for indexed in iter_evaluation_rows(metadata_root, header):
            episode = nullcontext(None)
            commit = None
            if evidence_context is not None:
                commit = evidence_context.episode_commit(name, indexed.ordinal)
                if (
                    commit.identity_sha256 != header.identity.sha256
                    or commit.ordinal != indexed.ordinal
                    or commit.row_offset != indexed.offset
                    or commit.row_bytes != len(indexed.raw)
                    or commit.row_sha256 != sha256_bytes(indexed.raw)
                    or commit.episode_public_id != indexed.row.public_id
                    or commit.episode_sha256 != indexed.row.episode_sha256
                ):
                    raise ValueError("episode commitment differs from indexed row")
                for entry in (*commit.owned, *commit.borrowed):
                    if not Path(entry.path).is_relative_to(name):
                        raise ValueError("episode commitment escapes evaluation root")
                    relative = Path(entry.path).relative_to(name).as_posix()
                    if header.hashes.get(relative) != entry.sha256:
                        raise ValueError("episode commitment differs from original DONE")
                episode = evidence_context.episode(
                    name,
                    indexed.ordinal,
                    commit_sha256=sha256_bytes(canonical_json_bytes(asdict(commit))),
                )
            with episode as episode_lease:
                if episode_lease is not None and episode_lease.ref.logical_root != name:
                    raise ValueError("episode lease logical root differs")
                payload_root = root if episode_lease is None else episode_lease.local_root / name
                verified = verify_evaluation_episode(
                    payload_root,
                    header=header,
                    indexed_row=indexed,
                    decode_trajectory=plot_examples and indexed.row.is_positive not in decoded,
                    borrowed_paths=None
                    if commit is None
                    else frozenset(
                        Path(entry.path).relative_to(name).as_posix() for entry in commit.borrowed
                    ),
                )
                if verified.trajectory is not None:
                    decoded.add(indexed.row.is_positive)
                if commit is not None:
                    owned = frozenset(
                        Path(entry.path).relative_to(name).as_posix() for entry in commit.owned
                    )
                    borrowed = frozenset(
                        Path(entry.path).relative_to(name).as_posix() for entry in commit.borrowed
                    )
                    if (owned, borrowed) != (verified.owned, verified.shared):
                        raise ValueError(
                            "episode commitment ownership differs from scientific evidence"
                        )
                accumulator.add(verified)
                yield verified
        finish_evaluation_scan(header, accumulator)


def scan_evaluation(root: Path, *, evidence_context=None, consume=None):
    with evaluation_header(root, evidence_context=evidence_context) as (header, _):
        pass
    for verified in iter_evaluation_episodes(
        root, evidence_context=evidence_context, expected_header=header
    ):
        if consume is not None:
            consume(verified)
    return header
