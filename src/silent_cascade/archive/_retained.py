"""Bounded private retained-inventory generations with immutable leaf partitions.

This module serializes/authenticates rows supplied by the custody walker. It does
not discover files, authorize eviction, grant capacity, or mutate a ledger head.
Every consumer must exhaust its record iterator before accepting the snapshot.
"""

import hashlib
import json
import re
import stat
from pathlib import PurePosixPath

from silent_cascade.archive.catalog import _control_reader, _create_control_object
from silent_cascade.archive.ledger import StorageBlocked
from silent_cascade.hashing import canonical_json_bytes, sha256_bytes

_MAX_PARTITIONS = 4096
_MAX_INDEX_PAGES = 4096
_MAX_INTEGER = 2**63 - 1
_LEAF = "phase4-engineering-leaf-v2"
_INDEX = "phase4-engineering-index-v2"
_SNAPSHOT = "phase4-engineering-snapshot-v2"
_LAYOUT = {"partition", "lower", "upper", "original_entries"}
_REFERENCE = _LAYOUT | {"sha256", "encoded_bytes", "entries", "allocated"}
_SNAPSHOT_KEYS = {
    "schema_version",
    "head",
    "index_pages",
    "pages",
    "layout_sha256",
    "original_entries",
    "entries",
    "allocated",
    "encoded_bytes",
}
_ROW = {
    "path",
    "device",
    "inode",
    "mode",
    "links",
    "bytes",
    "allocated",
    "mtime_ns",
    "ctime_ns",
    "sha256",
}


def _deny(reason):
    raise StorageBlocked(f"storage_blocked: retained inventory {reason}")


def _integer(value, *, minimum=0, maximum=_MAX_INTEGER):
    if type(value) is not int or not minimum <= value <= maximum:
        _deny("integer exceeds schema")
    return value


def _hash(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _deny("hash differs")
    return value


def _key(path):
    if type(path) is not str or not path or len(path.encode()) > 4096:
        _deny("path exceeds schema")
    value = PurePosixPath(path)
    if value.is_absolute() or ".." in value.parts or str(value) != path:
        _deny("unsafe path")
    return () if path == "." else value.parts


def _row(record):
    if type(record) is not dict or set(record) != _ROW:
        _deny("row fields differ")
    _key(record["path"])
    for name in _ROW - {"path", "sha256", "mtime_ns", "ctime_ns"}:
        _integer(record[name])
    for name in ("mtime_ns", "ctime_ns"):
        _integer(record[name], minimum=-_MAX_INTEGER - 1)
    if record["allocated"] % 512:
        _deny("allocation counter differs")
    if record["sha256"] is not None:
        _hash(record["sha256"])
        if not (
            stat.S_ISREG(record["mode"])
            and record["links"] == 1
            and record["bytes"] <= 128 * 1024**2
            and record["allocated"] >= record["bytes"]
        ):
            _deny("unsupported row carries content authority")
    return record


def _layout(reference):
    return {name: reference[name] for name in _LAYOUT}


def _layout_hash(references):
    digest = hashlib.sha256()
    for reference in references:
        digest.update(canonical_json_bytes(_layout(reference)))
    return digest.hexdigest()


def _reference(value, policy):
    if type(value) is not dict or set(value) != _REFERENCE:
        _deny("index reference fields differ")
    _integer(value["partition"], maximum=_MAX_PARTITIONS - 1)
    _integer(value["original_entries"], minimum=1, maximum=policy.page_entries)
    _integer(value["entries"], maximum=value["original_entries"])
    _integer(value["encoded_bytes"], minimum=1, maximum=policy.page_bytes)
    _integer(value["allocated"])
    if _key(value["lower"]) > _key(value["upper"]):
        _deny("partition range reversed")
    _hash(value["sha256"])


def _descriptor(snapshot, policy):
    if type(snapshot) is not dict or snapshot.get("schema_version") != _SNAPSHOT:
        _deny("legacy encoding requires explicit same-ledger transition")
    if set(snapshot) != _SNAPSHOT_KEYS:
        _deny("snapshot fields differ")
    _integer(snapshot["pages"], maximum=_MAX_PARTITIONS)
    _integer(snapshot["index_pages"], maximum=_MAX_INDEX_PAGES)
    _integer(snapshot["original_entries"], maximum=snapshot["pages"] * policy.page_entries)
    _integer(snapshot["entries"], maximum=snapshot["original_entries"])
    _integer(snapshot["allocated"])
    _integer(snapshot["encoded_bytes"], maximum=policy.metadata_bytes)
    _hash(snapshot["layout_sha256"])
    if snapshot["pages"]:
        _hash(snapshot["head"])
        if not 0 < snapshot["index_pages"] <= snapshot["pages"]:
            _deny("index cardinality differs")
    elif any(snapshot[name] for name in _SNAPSHOT_KEYS - {"schema_version", "layout_sha256"}):
        _deny("empty snapshot differs")


def _object(control_dir, kind, digest, policy):
    _hash(digest)
    try:
        raw = _control_reader(control_dir)(f"retained/{kind}/{digest}.json", policy.page_bytes)
        if sha256_bytes(raw) != digest:
            _deny("object authentication failed")
        value = json.loads(raw)
        if type(value) is not dict or canonical_json_bytes(value) != raw:
            _deny("object encoding is not canonical")
    except (OSError, ValueError, TypeError) as error:
        raise StorageBlocked(
            "storage_blocked: retained inventory object missing or invalid"
        ) from error
    return raw, value


def _references(control_dir, snapshot, policy):
    _descriptor(snapshot, policy)
    reverse, visited, head, index_bytes = [], set(), snapshot["head"], 0
    expected = snapshot["pages"] - 1
    for _ in range(snapshot["index_pages"]):
        if head is None or head in visited:
            _deny("index is missing or cyclic")
        visited.add(head)
        raw, page = _object(control_dir, "index", head, policy)
        if type(page) is not dict or set(page) != {
            "schema_version",
            "next",
            "first_partition",
            "leaves",
        }:
            _deny("index fields differ")
        if page["schema_version"] != _INDEX or type(page["leaves"]) is not list:
            _deny("index schema differs")
        if not 0 < len(page["leaves"]) <= policy.page_entries:
            _deny("index page exceeds entries")
        _integer(page["first_partition"], maximum=_MAX_PARTITIONS - 1)
        if page["first_partition"] + len(page["leaves"]) - 1 != expected:
            _deny("index partition order differs")
        for reference in reversed(page["leaves"]):
            _reference(reference, policy)
            if reference["partition"] != expected:
                _deny("duplicate or reordered partition")
            reverse.append(reference)
            expected -= 1
            if len(reverse) > _MAX_PARTITIONS:
                _deny("partition capacity exceeded")
        head = page["next"]
        if head is not None:
            _hash(head)
        index_bytes += len(raw)
    if head is not None or expected != -1:
        _deny("index chain incomplete")
    reverse.reverse()
    references = reverse
    previous, hashes = None, set()
    for reference in references:
        if previous is not None and previous >= _key(reference["lower"]):
            _deny("partition ranges overlap or reorder")
        previous = _key(reference["upper"])
        if reference["sha256"] in hashes:
            _deny("duplicate leaf")
        hashes.add(reference["sha256"])
    totals = {
        "original_entries": sum(ref["original_entries"] for ref in references),
        "entries": sum(ref["entries"] for ref in references),
        "allocated": sum(ref["allocated"] for ref in references),
        "encoded_bytes": index_bytes + sum(ref["encoded_bytes"] for ref in references),
    }
    if any(snapshot[name] != value for name, value in totals.items()):
        _deny("snapshot totals differ")
    if _layout_hash(references) != snapshot["layout_sha256"]:
        _deny("original partition layout differs")
    return references


def _leaf(control_dir, reference, policy):
    raw, value = _object(control_dir, "leaves", reference["sha256"], policy)
    if type(value) is not dict or set(value) != _LAYOUT | {"schema_version", "entries"}:
        _deny("leaf fields differ")
    if value["schema_version"] != _LEAF or _layout(value) != _layout(reference):
        _deny("leaf original partition differs")
    _integer(value["partition"], maximum=_MAX_PARTITIONS - 1)
    _integer(value["original_entries"], minimum=1, maximum=policy.page_entries)
    _key(value["lower"])
    _key(value["upper"])
    if type(value["entries"]) is not list or len(value["entries"]) != reference["entries"]:
        _deny("leaf entries differ")
    if len(raw) != reference["encoded_bytes"]:
        _deny("leaf byte count differs")
    previous = None
    for row in value["entries"]:
        key = _key(_row(row)["path"])
        if not _key(reference["lower"]) <= key <= _key(reference["upper"]):
            _deny("row escapes original partition")
        if previous is not None and previous >= key:
            _deny("leaf row order differs")
        previous = key
    if sum(row["allocated"] for row in value["entries"]) != reference["allocated"]:
        _deny("leaf allocation count differs")
    return value


def read_records(control_dir, snapshot, policy, *, reverse=False):
    references = _references(control_dir, snapshot, policy)
    for reference in reversed(references) if reverse else references:
        rows = _leaf(control_dir, reference, policy)["entries"]
        yield from reversed(rows) if reverse else rows


def _worst_row(row):
    value = dict(row)
    if stat.S_ISDIR(value["mode"]):
        for name in ("links", "bytes", "allocated", "mtime_ns", "ctime_ns"):
            value[name] = 10**20 - 1  # conservative signed-counter textual width
    return value


def _payload(layout, rows):
    return {"schema_version": _LEAF, **layout, "entries": rows}


def _initial_leaves(records, policy):
    rows, encoded, partition, previous = [], 0, 0, None
    for row in records:
        key = _key(_row(row)["path"])
        if previous is not None and previous >= key:
            _deny("initial DFS order differs")
        previous = key
        size = len(canonical_json_bytes(_worst_row(row))) + 1

        def header(lower, upper=row["path"]):
            return len(
                canonical_json_bytes(
                    _payload(
                        {
                            "partition": _MAX_PARTITIONS - 1,
                            "lower": lower,
                            "upper": upper,
                            "original_entries": policy.page_entries,
                        },
                        [],
                    )
                )
            )

        if rows and (
            len(rows) == policy.page_entries
            or header(rows[0]["path"]) + encoded + size > policy.page_bytes
        ):
            yield _payload(
                {
                    "partition": partition,
                    "lower": rows[0]["path"],
                    "upper": rows[-1]["path"],
                    "original_entries": len(rows),
                },
                rows,
            )
            partition += 1
            rows, encoded = [], 0
        if partition >= _MAX_PARTITIONS or header(row["path"]) + size > policy.page_bytes:
            _deny("initial leaf exceeds capacity")
        rows.append(row)
        encoded += size
    if rows:
        yield _payload(
            {
                "partition": partition,
                "lower": rows[0]["path"],
                "upper": rows[-1]["path"],
                "original_entries": len(rows),
            },
            rows,
        )


def _index_pages(references, policy):
    head, page, encoded = None, [], 0

    def payload(refs):
        return {
            "schema_version": _INDEX,
            "next": head,
            "first_partition": refs[0]["partition"],
            "leaves": refs,
        }

    for reference in references:
        size = len(canonical_json_bytes(reference)) + 1
        first = page[0]["partition"] if page else reference["partition"]
        header = len(
            canonical_json_bytes(
                {"schema_version": _INDEX, "next": head, "first_partition": first, "leaves": []}
            )
        )
        if page and (
            len(page) == policy.page_entries or header + encoded + size - 1 > policy.page_bytes
        ):
            raw = canonical_json_bytes(payload(page))
            head = sha256_bytes(raw)
            yield head, raw
            page, encoded = [], 0
            header = len(
                canonical_json_bytes(
                    {
                        "schema_version": _INDEX,
                        "next": head,
                        "first_partition": reference["partition"],
                        "leaves": [],
                    }
                )
            )
        if header + size - 1 > policy.page_bytes:
            _deny("index reference exceeds page")
        page.append(reference)
        encoded += size
    if page:
        raw = canonical_json_bytes(payload(page))
        yield sha256_bytes(raw), raw


def _selected(paths):
    if type(paths) is not tuple or len(paths) > 1000 or len(set(paths)) != len(paths):
        _deny("removal authority exceeds bound")
    for path in paths:
        _key(path)
    return set(paths)


def _generation_bound(policy):
    value = {name: _MAX_INTEGER for name in _SNAPSHOT_KEYS}
    value.update(schema_version=_SNAPSHOT, head="f" * 64, layout_sha256="f" * 64)
    size = len(canonical_json_bytes(value))
    if size > policy.page_bytes:
        _deny("generation descriptor exceeds reader")
    return size


def _affected(row, selected):
    path = row["path"]
    return path in selected or (
        stat.S_ISDIR(row["mode"])
        and any(path == "." or member.startswith(path + "/") for member in selected)
    )


def _publication_plan(control_dir, snapshot, policy, selected_paths, references=None):
    selected = _selected(selected_paths)
    references = _references(control_dir, snapshot, policy) if references is None else references
    affected, found, total = set(), set(), 0

    def bounded_references():
        nonlocal total
        for ref in references:
            leaf = _leaf(control_dir, ref, policy)
            for row in leaf["entries"]:
                if row["path"] in selected:
                    if row["sha256"] is None or not stat.S_ISREG(row["mode"]):
                        _deny("removal is not an eligible ordinary row")
                    found.add(row["path"])
            bound = dict(ref)
            if any(_affected(row, selected) for row in leaf["entries"]):
                affected.add(ref["partition"])
                encoded = len(
                    canonical_json_bytes(
                        _payload(_layout(ref), [_worst_row(row) for row in leaf["entries"]])
                    )
                )
                if encoded > policy.page_bytes:
                    _deny("changed leaf exceeds original capacity")
                total += encoded
                bound["encoded_bytes"], bound["allocated"] = encoded, _MAX_INTEGER
            yield bound

    index_bytes, index_count = _index_bound(bounded_references(), policy)
    if found != selected:
        _deny("removal path absent from authenticated inventory")
    total += _generation_bound(policy)
    if total + index_bytes > policy.metadata_bytes or index_count > _MAX_INDEX_PAGES:
        _deny("publication exceeds metadata capacity")
    return total + index_bytes, len(affected) + index_count + 1, affected


def _index_bound(references, policy):
    count, encoded, maximum = 0, 0, 0
    for ref in references:
        size = len(canonical_json_bytes(ref)) + 1
        count += 1
        encoded += size
        maximum = max(maximum, size)
    if not count:
        return 0, 0
    header = len(
        canonical_json_bytes(
            {
                "schema_version": _INDEX,
                "next": "f" * 64,
                "first_partition": _MAX_PARTITIONS - 1,
                "leaves": [],
            }
        )
    )
    capacity = min(policy.page_entries, (policy.page_bytes - header) // maximum)
    if capacity < 1:
        _deny("bounded index reference exceeds page")
    pages = (count + capacity - 1) // capacity
    if pages > _MAX_INDEX_PAGES:
        _deny("bounded index capacity exceeded")
    return encoded + pages * header, pages


def publication_bound(control_dir, snapshot, policy, selected_paths):
    encoded, objects, _affected_set = _publication_plan(
        control_dir, snapshot, policy, selected_paths
    )
    return encoded, objects


def initial_bound(records, policy):
    """No-write bootstrap bound, including mutable-directory widths and index."""
    count, encoded, maximum, longest = 0, 0, 0, "."
    for row in records:
        size = len(canonical_json_bytes(_worst_row(row))) + 1
        count += 1
        encoded += size
        maximum = max(maximum, size)
        if len(canonical_json_bytes({"p": row["path"]})) > len(
            canonical_json_bytes({"p": longest})
        ):
            longest = row["path"]
    if not count:
        return _generation_bound(policy), 1
    layout = {
        "partition": _MAX_PARTITIONS - 1,
        "lower": longest,
        "upper": longest,
        "original_entries": policy.page_entries,
    }
    header = len(canonical_json_bytes(_payload(layout, [])))
    capacity = min(policy.page_entries, (policy.page_bytes - header) // maximum)
    if capacity < 1:
        _deny("bootstrap row exceeds page")
    leaves = (count + capacity - 1) // capacity
    if leaves > _MAX_PARTITIONS:
        _deny("bootstrap partition capacity exceeded")
    refs = (
        {
            **layout,
            "partition": index,
            "sha256": "f" * 64,
            "encoded_bytes": policy.page_bytes,
            "entries": policy.page_entries,
            "allocated": _MAX_INTEGER,
        }
        for index in range(leaves)
    )
    index_bytes, indexes = _index_bound(refs, policy)
    total = encoded + leaves * header + _generation_bound(policy) + index_bytes
    if total > policy.metadata_bytes or indexes > _MAX_INDEX_PAGES:
        _deny("bootstrap metadata capacity exceeded")
    return total, leaves + indexes + 1


def build_snapshot(
    records, policy, *, control_dir=None, previous=None, removed_paths=(), publish=False
):
    selected = _selected(removed_paths)
    if publish and control_dir is None:
        _deny("publication has no control root")
    references, changed, total = [], False, 0
    if previous is None:
        if selected:
            _deny("initial snapshot cannot remove rows")
        leaves = ((leaf, None) for leaf in _initial_leaves(records, policy))
    else:
        old = _references(control_dir, previous, policy)
        references = old
        _bytes, _objects, affected = _publication_plan(
            control_dir, previous, policy, removed_paths, old
        )
        current_stream = iter(records)

        def updated():
            current = next(current_stream, None)
            for ref in old:
                leaf = _leaf(control_dir, ref, policy)
                rows = []
                for expected in leaf["entries"]:
                    if current is None or current["path"] != expected["path"]:
                        if expected["path"] in selected:
                            continue
                        _deny("unlisted row removal or addition")
                    _row(current)
                    if current != expected and not (
                        stat.S_ISDIR(expected["mode"])
                        and _affected(expected, selected)
                        and all(
                            current[name] == expected[name]
                            for name in ("path", "device", "inode", "mode")
                        )
                    ):
                        _deny("unlisted row change")
                    rows.append(current)
                    current = next(current_stream, None)
                replacement = _payload(_layout(ref), rows)
                if replacement != leaf and ref["partition"] not in affected:
                    _deny("changed leaf outside admitted affected set")
                yield replacement, ref
            if current is not None:
                _deny("unlisted trailing row addition")

        leaves = updated()
    for leaf, old_ref in leaves:
        raw = canonical_json_bytes(leaf)
        if len(raw) > policy.page_bytes:
            _deny("leaf exceeds reader limit")
        digest = sha256_bytes(raw)
        ref = {
            **_layout(leaf),
            "sha256": digest,
            "encoded_bytes": len(raw),
            "entries": len(leaf["entries"]),
            "allocated": sum(row["allocated"] for row in leaf["entries"]),
        }
        if old_ref is None:
            references.append(ref)
        else:
            references[ref["partition"]] = ref
        total += len(raw)
        if total > policy.metadata_bytes:
            _deny("snapshot exceeds metadata capacity")
        if old_ref is None or old_ref["sha256"] != digest:
            changed = True
            if publish:
                _create_control_object(control_dir, f"retained/leaves/{digest}.json", raw)
    head, index_count = None, 0
    for head, raw in _index_pages(references, policy):
        total += len(raw)
        index_count += 1
        if total > policy.metadata_bytes or index_count > _MAX_INDEX_PAGES:
            _deny("index exceeds metadata capacity")
        if publish and (previous is None or changed):
            _create_control_object(control_dir, f"retained/index/{head}.json", raw)
    snapshot = {
        "schema_version": _SNAPSHOT,
        "head": head,
        "index_pages": index_count,
        "pages": len(references),
        "layout_sha256": _layout_hash(references),
        "original_entries": sum(ref["original_entries"] for ref in references),
        "entries": sum(ref["entries"] for ref in references),
        "allocated": sum(ref["allocated"] for ref in references),
        "encoded_bytes": total,
    }
    _descriptor(snapshot, policy)
    if previous is not None and any(
        snapshot[name] != previous[name] for name in ("pages", "layout_sha256", "original_entries")
    ):
        _deny("update changed original partition layout")
    if publish and (previous is None or changed):
        raw = canonical_json_bytes(snapshot)
        if len(raw) > policy.page_bytes:
            _deny("generation descriptor exceeds reader")
        _create_control_object(control_dir, f"retained/generations/{sha256_bytes(raw)}.json", raw)
    return snapshot
